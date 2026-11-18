"""Solvated geometry optimization and explicitly defined thermochemistry."""
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.io import read
from core import backend
from core.paths import DFT, UMA, LIBRARY, HARTREE_EV, write_xyz
from core.protocol import GEOMETRY, file_hash, protocol_hash, load_record, write_record, runtime_versions
from .sp import make_mf, electronic_checks
from .jobs import run_rows


def _atoms(molecule):
    return Atoms([molecule.atom_symbol(i) for i in range(molecule.natm)],
                 positions=molecule.atom_coords(unit="Angstrom"))


def thermal_correction(mf, *, directory, temperature=298.15, pressure=101325.0,
                       model="rrho", cutoffs=(25.0, 50.0, 100.0)):
    """Native DFT Hessian; RRHO or an explicitly separate entropy-only qRRHO model.

    This is not the manuscript's certified xTB-kernel partition implementation.
    Two Hessian columns are checked at two displacements, not the whole Hessian.
    """
    from pyscf.hessian import thermo
    from pyscf.data import nist
    from scipy.constants import Boltzmann, Planck, speed_of_light
    if temperature <= 0 or pressure <= 0 or model not in ("rrho", "qrrho"):
        raise ValueError("positive thermodynamic conditions and rrho/qrrho required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(mf, "to_cpu"):
        mf = mf.to_cpu()
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    center = gradient.kernel()
    if not np.isfinite(center).all() or np.max(np.abs(center)) > 4.5e-4:
        raise ValueError("geometry fails maximum Cartesian gradient threshold")
    hessian_driver = mf.Hessian()
    hessian_driver.grid_response = True
    hessian = hessian_driver.kernel()
    n = mf.mol.natm
    matrix = hessian.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    np.save(directory / "hessian.npy", hessian)
    np.save(directory / "gradient.npy", center)
    if not np.isfinite(matrix).all() or np.max(np.abs(matrix - matrix.T)) > 1e-6:
        raise ValueError("Hessian fails finite/symmetry check")
    original_molecule = mf.mol.copy()
    original_energy = float(mf.e_tot)
    coords = original_molecule.atom_coords()
    target = next((i for i in range(n) if mf.mol.atom_charge(i) == 8), n - 1)
    columns = [(0, 1), (target, 2)]
    probes = []
    scanner = gradient.as_scanner()
    for atom, axis in columns:
        for step in (0.0005, 0.001):
            gradients = []
            for sign in (-1, 1):
                displaced = coords.copy()
                displaced[atom, axis] += sign * step
                molecule = original_molecule.copy().set_geom_(displaced, unit="Bohr")
                energy, value = scanner(molecule)
                if not scanner.base.converged:
                    raise ValueError("displaced-gradient SCF did not converge")
                gradients.append(np.asarray(value).ravel())
            fd = (gradients[1] - gradients[0]) / (2 * step)
            exact = matrix[:, 3 * atom + axis]
            error = float(np.linalg.norm(fd - exact) / max(np.linalg.norm(exact), 1e-12))
            probes.append({"atom": atom, "axis": axis, "step_bohr": step, "relative_error": error})
            np.save(directory / f"probe_{atom}_{axis}_{step}.npy", gradients)
    # Restore the central geometry before the partition calculation.
    mf.reset(original_molecule)
    mf.e_tot = original_energy
    if any(not np.isfinite(p["relative_error"]) or p["relative_error"] > 0.01 for p in probes):
        raise ValueError(f"Hessian derivative check failed: {probes}")
    modes = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
    frequencies = np.asarray(modes["freq_wavenumber"], dtype=float)
    if not np.isfinite(frequencies).all() or np.any(frequencies < -5) or np.any(frequencies == 0):
        raise ValueError("unstable or zero internal vibrational mode")
    partition = thermo.thermo(mf, np.abs(modes["freq_au"]), temperature, pressure)
    # The nuclear partition excludes the electronic spin contribution.
    nuclear = sum(partition[k][0] for k in ("G_trans", "G_rot", "G_vib"))
    electronic_entropy = -temperature * partition["S_elec"][0]
    rrho = float((nuclear + electronic_entropy) * HARTREE_EV)
    values = {}
    if model == "qrrho":
        # Grimme-style entropy interpolation with a fixed 1e-44 kg m² rotor inertia.
        nu = np.abs(frequencies) * speed_of_light * 100
        x = Planck * nu / (Boltzmann * temperature)
        expneg = np.exp(-x)
        s_ho = Boltzmann * (x * expneg / (-np.expm1(-x)) - np.log(-np.expm1(-x)))
        inertia = Planck / (8 * np.pi**2 * nu)
        reduced = inertia * 1e-44 / (inertia + 1e-44)
        s_rotor = Boltzmann * (0.5 + np.log(np.sqrt(8 * np.pi**3 * reduced * Boltzmann * temperature / Planck**2)))
        for cutoff in cutoffs:
            if cutoff <= 0:
                raise ValueError("positive rotor cutoff required")
            weight = 1 / (1 + (cutoff / np.abs(frequencies))**4)
            correction = -temperature * float(np.sum((1 - weight) * (s_rotor - s_ho))) / nist.HARTREE2J * HARTREE_EV
            values[str(float(cutoff))] = rrho + correction
        if "50.0" not in values:
            raise ValueError("primary 50 cm^-1 cutoff required")
    protocol = {"model": "PySCF-RRHO" if model == "rrho" else "entropy-only-qRRHO-fixed-inertia",
                "engine": "pyscf_cpu", "implementation_versions": runtime_versions(),
                "temperature_K": temperature, "pressure_Pa": pressure,
                "standard_state": "1atm" if pressure == 101325 else f"{pressure}Pa",
                "mode_projection": "PySCF mass-weighted translation/rotation exclusion",
                "spin_entropy": "-kT ln(multiplicity)", "frequency_scale": 1.0,
                "primary_cutoff_cm": 50 if model == "qrrho" else None,
                "rotor_inertia_kg_m2": 1e-44 if model == "qrrho" else None,
                "manuscript_kernel_equivalent": False}
    return {"thermal_status": "passed", "thermal_protocol": protocol,
            "g_thermal_eV": values["50.0"] if model == "qrrho" else rrho,
            "thermal_by_cutoff_eV": values, "frequencies_cm": frequencies.tolist(),
            "symmetry_number": int(partition["sym_number"][0]), "derivative_probes": probes,
            "gradient_max_Ha_bohr": float(np.max(np.abs(center)))}


def optimize_state(xyz, charge, mult, *, directory, settings=None, thermal=False,
                   thermal_model="rrho", maxsteps=100, memory_mb=4000, threads=4):
    from pyscf.geomopt.geometric_solver import kernel
    xyz = Path(xyz).resolve()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = {**GEOMETRY, **(settings or {})}
    if settings is None and charge < 0:
        chosen["basis"] = "def2-svpd"
    method = {**chosen, "engine": backend.get_dft_backend_name(), "implementation_versions": runtime_versions()}
    request = {**method, "maxsteps": maxsteps, "thermal": thermal, "thermal_model": thermal_model}
    key = protocol_hash(file_hash(xyz), charge, mult, request)
    work = directory / "attempts" / key[:20]
    work.mkdir(parents=True, exist_ok=True)
    cached = work / "result.json"
    if cached.exists():
        old = load_record(cached)
        if old["status"] == "complete":
            return write_record(directory / "result.json", old)
    restart = work / "checkpoint.xyz"
    atoms = read(restart if restart.exists() else xyz)
    mf = make_mf(atoms, charge, mult, chosen, memory_mb=memory_mb, threads=threads)
    mf.chkfile = str(work / "scf.chk")
    scanner = mf.nuc_grad_method()
    scanner.grid_response = True
    def checkpoint(env):
        write_xyz(_atoms(env["mol"]), restart)
    converged, molecule = kernel(scanner, maxsteps=maxsteps, callback=checkpoint,
                                 assert_convergence=True, convergence_gmax=4.5e-4)
    write_xyz(_atoms(molecule), work / "opt.xyz")
    mf = make_mf(_atoms(molecule), charge, mult, chosen, memory_mb=memory_mb, threads=threads)
    mf.chkfile = str(work / "optimized.chk")
    mf.kernel()
    qc = electronic_checks(mf)
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    maximum = float(np.max(np.abs(gradient.kernel())))
    if not converged or not np.isfinite(maximum) or maximum > 4.5e-4:
        raise RuntimeError(f"geometry optimization failed (max gradient {maximum})")
    record = {"status": "complete", "optimization_status": "passed", "thermal_status": "not_requested",
              "charge": int(charge), "mult": int(mult), "method": method, "protocol_hash": key,
              "energy_eV": float(mf.e_tot) * HARTREE_EV,
              "qc": {**qc, "gradient_max_Ha_bohr": maximum},
              "geometry": str((work / "opt.xyz").resolve()),
              "sources": {str(xyz): file_hash(xyz), str((work / "opt.xyz").resolve()): file_hash(work / "opt.xyz")}}
    if chosen.get("solvent"):
        record["e_smd_eV"] = record["energy_eV"]
    record["thermal_geometry_family"] = ({**method, "basis": "def2-svpd if anionic; def2-svp otherwise"}
                                         if settings is None else method)
    # Preserve geometry even if expensive Hessian work fails or is interrupted.
    write_record(work / "geometry.json", record)
    if thermal:
        record.update(thermal_correction(mf, directory=work, model=thermal_model))
        for path in work.glob("*.npy"):
            record["sources"][str(path.resolve())] = file_hash(path)
    write_record(cached, record)
    return write_record(directory / "result.json", record)


def run_batch(*, from_library=False, **kwargs):
    def worker(row, directory, **options):
        if from_library:
            paths = sorted((LIBRARY / row["id"] / "conformers").glob("conf_*.xyz"))
            mult = int(row["mult_hint"])
        else:
            record = load_record(UMA / row["id"] / row["state"] / "result.json")
            if record.get('status') != 'complete':
                raise ValueError('UMA state did not complete with a valid graph')
            # Keep the declared spin until DFT spin ordering is checked.
            mult = int(row["mult_hint"])
            paths = [Path(c["geometry"]) for c in record["conformers"] if c["mult"] == mult]
        if not paths:
            raise ValueError("no conformers for declared multiplicity")
        completed, failures = [], []
        for index, path in enumerate(paths):
            try:
                completed.append(optimize_state(path, int(row["charge"]), mult,
                                  directory=directory / "conformers" / f"conf_{index:03}", **options))
            except Exception as exc:
                failures.append({"input": str(path), "error": str(exc)})
        if not completed:
            raise RuntimeError(f"all DFT conformers failed: {failures}")
        from core.properties.integrity import assess
        checked = []
        for record in completed:
            integrity = assess(row["smiles"], read(record["geometry"]))
            if integrity["passed"]:
                record["integrity"] = integrity
                checked.append(record)
            else:
                failures.append({"input": record["geometry"], "error": integrity["errors"]})
        if not checked:
            raise RuntimeError(f"no DFT geometry preserves the declared graph: {failures}")
        completed = checked
        best = min(completed, key=lambda r: r["energy_eV"])
        return write_record(directory / "result.json", {
            **best, "identity": row["smiles"], "id": row["id"], "state": row["state"],
            "selection": "lowest geometry-level electronic energy among completed conformers",
            "sampling_complete": not failures, "failed_attempts": failures,
            "conformers": [{k: r[k] for k in ("geometry", "energy_eV", "mult")} for r in completed]})
    return run_rows(worker, DFT, **kwargs)


if __name__ == "__main__":
    from core.cli import main
    raise SystemExit(main(["dft", *sys.argv[1:]]))
