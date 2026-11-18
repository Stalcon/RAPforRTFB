"""Gas-phase UMA preparation on CPU, CUDA, or ROCm PyTorch."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from ase.io import read
from ase.optimize import BFGS
from core import backend
from core.paths import LIBRARY, UMA, write_xyz
from core.protocol import file_hash, protocol_hash, load_record, write_record, runtime_versions
from .jobs import run_rows

DEFAULT_MODEL = "uma-s-1p2"
TASK = "omol"
INFERENCE = {"compile": False, "tf32": False, "merge_mole": False, "execution_mode": "general"}


@lru_cache(maxsize=2)
def load_model(model=DEFAULT_MODEL, device=None):
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings
    device = device or backend.get_uma_device()
    # Eager execution avoids CUDA-only accelerated inference paths on AMD.
    predictor = pretrained_mlip.get_predict_unit(
        model, device="cuda" if str(device).startswith("cuda") else "cpu",
        inference_settings=InferenceSettings(**INFERENCE))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    return FAIRChemCalculator(predictor, task_name=TASK)


@lru_cache(maxsize=2)
def model_identity(model=DEFAULT_MODEL):
    from fairchem.core.calculate import pretrained_mlip
    checkpoint = Path(pretrained_mlip.pretrained_checkpoint_path_from_name(model))
    return {"model": model, "task": TASK, "checkpoint_sha256": file_hash(checkpoint),
            "inference": INFERENCE, "versions": runtime_versions()}


def candidate_mults(n_elec, hint=1):
    if int(n_elec) != n_elec or n_elec < 1:
        raise ValueError("positive integer electron count required")
    candidates = {1, 3} if n_elec % 2 == 0 else {2}
    if hint is not None:
        if int(hint) != hint:
            raise ValueError('multiplicity must be an integer')
        candidates.add(int(hint))
    if any(m < 1 or m - 1 > n_elec or (n_elec - (m - 1)) % 2 for m in candidates):
        raise ValueError("spin multiplicity is inconsistent with electron count")
    return sorted(candidates)


def relax_one(calc, atoms, charge, mult, fmax=0.02, steps=300, directory=None):
    candidate_mults(int(sum(atoms.numbers)) - charge, mult)
    if fmax <= 0 or steps < 1:
        raise ValueError("positive force tolerance and steps required")
    atoms = atoms.copy()
    atoms.info.update(charge=int(charge), spin=int(mult))
    atoms.calc = calc
    options = {"logfile": None}
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        options.update(logfile=str(directory / "optimization.log"),
                       trajectory=str(directory / "optimization.traj"))
    optimizer = BFGS(atoms, **options)
    if directory:
        optimizer.attach(lambda: write_xyz(atoms, directory / "checkpoint.xyz"), interval=1)
    converged = optimizer.run(fmax=fmax, steps=steps)
    forces = atoms.get_forces()
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(forces).all() or not np.isfinite(energy):
        raise RuntimeError("UMA returned nonfinite energy or forces")
    maximum = float(np.linalg.norm(forces, axis=1).max())
    return atoms, {"energy_eV": energy, "fmax_eV_A": maximum,
                   "converged": bool(converged and maximum <= fmax),
                   "steps": optimizer.nsteps, "charge": int(charge), "mult": int(mult)}


def preopt_state(conf_paths, charge, mult_hint, fmax=0.02, steps=300,
                 *, directory, model=DEFAULT_MODEL, scan_spins=False):
    paths = [Path(p).resolve() for p in conf_paths]
    if not paths:
        raise ValueError("no input conformers")
    directory = Path(directory)
    calc = load_model(model)
    identity = model_identity(model)
    settings = {**identity, "fmax": fmax, "steps": steps, "scan_spins": scan_spins,
                "device": backend.get_uma_device()}
    selected = []
    failures = []
    for path in paths:
        initial = read(path)
        possible = candidate_mults(int(sum(initial.numbers)) - charge, mult_hint)
        for mult in (possible if scan_spins else [int(mult_hint)]):
            key = protocol_hash(file_hash(path), charge, mult, settings)
            work = directory / "attempts" / f"{path.stem}_m{mult}_{key[:16]}"
            record_file = work / "result.json"
            record = load_record(record_file) if record_file.exists() else None
            if not record or record["status"] != "complete":
                try:
                    start = read(work / "checkpoint.xyz") if (work / "checkpoint.xyz").exists() else initial
                    atoms, result = relax_one(calc, start, charge, mult, fmax, steps, work)
                    write_xyz(atoms, work / "opt.xyz")
                    record = write_record(record_file, {
                        **result, "status": "complete" if result["converged"] else "unconverged",
                        "protocol_hash": key, "method": settings,
                        "sources": {str(path): file_hash(path), str((work / "opt.xyz").resolve()): file_hash(work / "opt.xyz")},
                        "geometry": str((work / "opt.xyz").resolve()), "input_conformer": path.name})
                except Exception as exc:
                    failures.append({"conformer": path.name, "mult": mult, "error": str(exc)})
                    continue
            if record["status"] == "complete":
                selected.append(record)
            else:
                failures.append({"conformer": path.name, "mult": mult, "error": "not converged"})
    if not selected:
        raise RuntimeError(f"No converged UMA conformers: {failures}")
    best = min(selected, key=lambda r: r["energy_eV"])
    write_xyz(read(best["geometry"]), directory / "opt.xyz")
    return write_record(directory / "result.json", {
        **best, "geometry": str((directory / "opt.xyz").resolve()),
        "sources": {**best["sources"], str((directory / "opt.xyz").resolve()): file_hash(directory / "opt.xyz")},
        "scope": "gas-phase ML preoptimization, not a DFT ground-state decision",
        "sampling_complete": not failures, "failed_attempts": failures,
        "conformers": [{k: r[k] for k in ("geometry", "mult", "energy_eV", "input_conformer")} for r in selected]})


def run_batch(**kwargs):
    def worker(row, directory, **options):
        from core.properties.integrity import assess
        result = preopt_state(sorted((LIBRARY / row["id"] / "conformers").glob("conf_*.xyz")),
                              int(row["charge"]), int(row["mult_hint"]), directory=directory, **options)
        checks = [assess(row["smiles"], read(item["geometry"])) for item in result["conformers"]]
        good = [item for item, check in zip(result["conformers"], checks) if check["passed"]]
        if not good:
            write_record(directory / 'result.json', {**result, 'status': 'failed', 'integrity_checks': checks})
            raise RuntimeError("no UMA minimum preserves the declared graph")
        best = min(good, key=lambda item: item["energy_eV"])
        write_xyz(read(best["geometry"]), directory / "opt.xyz")
        result.update(conformers=good, energy_eV=best["energy_eV"], mult=best["mult"],
                      integrity_checks=checks, identity=row["smiles"])
        result["sources"][str((directory / "opt.xyz").resolve())] = file_hash(directory / "opt.xyz")
        if len(good) != len(checks):
            result["sampling_complete"] = False
        return write_record(directory / "result.json", result)
    return run_rows(worker, UMA, **kwargs)


def main():
    from core.cli import main as cli_main
    return cli_main(["uma", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
