"""DFT single points and matched-reference bookkeeping."""
from pathlib import Path
import sys

import numpy as np
from ase.io import read
from core import backend
from core.paths import DFT, SP, HARTREE_EV, free_energy
from core.protocol import ACTIVE_SP, file_hash, protocol_hash, load_record, write_record, audit, runtime_versions
from .jobs import run_rows


def make_mf(atoms, charge, mult, settings=None, *, memory_mb=4000, threads=4, dft_module=None):
    from pyscf import gto, lib
    settings = {**ACTIVE_SP, **(settings or {})}
    electrons = int(sum(atoms.numbers)) - int(charge)
    if int(charge) != charge or int(mult) != mult or mult < 1 or mult - 1 > electrons or (electrons - mult + 1) % 2:
        raise ValueError("invalid charge/multiplicity for this geometry")
    if memory_mb < 256 or threads < 1:
        raise ValueError("positive threads and at least 256 MB required")
    lib.num_threads(threads)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                unit="Angstrom", basis=settings["basis"], charge=int(charge), spin=int(mult) - 1,
                max_memory=memory_mb, verbose=0)
    module = dft_module if dft_module is not None else backend.get_dft_module()
    mf = module.RKS(mol) if mult == 1 else module.UKS(mol)
    mf.xc = settings["xc"]
    if settings.get("dispersion"):
        mf.disp = settings["dispersion"]
    if settings["density_fit"]:
        mf = mf.density_fit(only_dfj=settings["only_dfj"])
    mf.grids.level = settings["grid_level"]
    if settings.get("solvent"):
        mf = mf.SMD()
        mf.with_solvent.solvent = settings["solvent"]
        if "solvent_grid_order" in settings:
            mf.with_solvent.lebedev_order = settings["solvent_grid_order"]
    mf.conv_tol = settings["conv_tol"]
    mf.max_cycle = settings["max_cycle"]
    return mf


def electronic_checks(mf):
    if not mf.converged or not np.isfinite(float(mf.e_tot)):
        raise RuntimeError("SCF did not converge to a finite energy")
    spin = mf.mol.spin / 2
    actual = float(mf.spin_square()[0])
    error = abs(actual - spin * (spin + 1))
    if not np.isfinite(actual) or error > 0.1:
        raise RuntimeError(f"spin contamination: |S²-S(S+1)|={error}")
    return {"scf_converged": True, "s2": actual, "spin_error": error,
            "spin_ordering_verified": False}


def single_point(xyz, charge, mult, *, settings=None, out=None, memory_mb=4000, threads=4):
    xyz = Path(xyz).resolve()
    settings = {**ACTIVE_SP, **(settings or {})}
    engine = backend.get_sp_backend_name()
    versions = runtime_versions()
    dft_module = None
    if engine == "gpu4pyscf-rocm":
        dft_module = backend.get_rocm_dft_module()
        import cupy
        import gpu4pyscf
        versions.update(cupy=cupy.__version__, gpu4pyscf=gpu4pyscf.__version__,
                        hip_runtime=cupy.cuda.runtime.runtimeGetVersion())
    method = {**settings, "engine": engine, "implementation_versions": versions}
    key = protocol_hash(file_hash(xyz), charge, mult, method)
    if out and Path(out).exists():
        try:
            old = load_record(out)
        except ValueError:
            old = {}
        if old.get("protocol_hash") == key and old.get("status") == "complete":
            return old
    atoms = read(xyz)
    mf = make_mf(atoms, charge, mult, settings, memory_mb=memory_mb, threads=threads,
                 dft_module=dft_module)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        mf.chkfile = str(Path(out).with_suffix(".chk"))
    mf.kernel()
    checks = electronic_checks(mf)
    record = {"status": "complete", "quantity": "electronic_energy", "energy_eV": float(mf.e_tot) * HARTREE_EV,
              "charge": int(charge), "mult": int(mult), "method": method, "qc": checks,
              "geometry": str(xyz), "protocol_hash": key, "sources": {str(xyz): file_hash(xyz)}}
    if settings.get("solvent"):
        record["e_smd_eV"] = record["energy_eV"]
    if out:
        record = write_record(out, record)
    return record


def fc_reference(oxidized, reduced):
    from core.properties.context import require_compatible
    require_compatible([oxidized, reduced], thermodynamic=True)
    if oxidized["charge"] != 1 or reduced["charge"] != 0:
        raise ValueError("Fc reference requires +1 and neutral states")
    if oxidized.get("identity") != reduced.get("identity"):
        raise ValueError("Fc states must have the same exact identity")
    return {"absolute_potential_V": free_energy(oxidized) - free_energy(reduced),
            "identity": oxidized["identity"], "method": oxidized["method"],
            "thermal_protocol": oxidized["thermal_protocol"],
            "scope": "caller must establish that this exact identity is ferrocene"}


def run_batch(**kwargs):
    def worker(row, directory, **options):
        source = DFT / row["id"] / row["state"] / "result.json"
        geometry = load_record(source)
        if geometry.get("optimization_status") != "passed":
            raise ValueError("DFT optimization must pass before the active single point")
        record = single_point(geometry["geometry"], int(row["charge"]), int(geometry["mult"]),
                              out=directory / "electronic.json", **options)
        record.update(identity=row["smiles"], id=row["id"], state=row["state"],
                      sources={**record["sources"], str(source.resolve()): file_hash(source)})
        from collections import Counter
        record["composition"] = dict(Counter(read(geometry["geometry"]).get_chemical_symbols()))
        for name in ("g_thermal_eV", "thermal_status", "thermal_protocol", "thermal_by_cutoff_eV"):
            if name in geometry:
                record[name] = geometry[name]
        if "thermal_protocol" in record:
            record["thermal_protocol"] = {**record["thermal_protocol"],
                                           "geometry_protocol": geometry["thermal_geometry_family"]}
        return write_record(directory / "result.json", record)
    return run_rows(worker, SP, **kwargs)


if __name__ == "__main__":
    from core.cli import main
    raise SystemExit(main(["sp", *sys.argv[1:]]))
