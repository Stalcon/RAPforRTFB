"""Prepare molecules and write reports for the validation runs."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ase import Atoms
from ase.io import read
from rdkit import Chem

from core.paths import HARTREE_EV, ROOT, atomic_write_json, canonical_smiles, write_xyz
from core.prep.build import check_stereo, conformer_ensemble
from core.properties.integrity import assess
from core.protocol import file_hash, write_record

REFERENCE_DATA = ROOT / "data" / "validation"
OUTPUT = ROOT / "calcs" / "validation"


def calculation_options(parser):
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=4000)


def resources(args):
    return {"threads": args.threads, "memory_mb": args.memory_mb}


def run_report(parser, runner):
    args = parser.parse_args()
    try:
        report = runner(args)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    atomic_write_json(args.out, report)
    print(f"Wrote {args.out}")
    return int(report.get("status") != "complete")


def seed_xyz(smiles, path):
    """Build a reproducible starting geometry in the atom order used by the graph checks."""
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    if check_stereo(mol):
        raise ValueError(f"unspecified stereochemistry: {smiles}")
    if len(Chem.AddHs(mol).GetAtoms()) == 1:
        write_xyz(Atoms([mol.GetAtomWithIdx(0).GetSymbol()], positions=[[0, 0, 0]]), path)
        return path
    molecule, indices, _, _ = conformer_ensemble(mol, top_k=1)
    positions = molecule.GetConformer(indices[0]).GetPositions()
    write_xyz(Atoms([a.GetSymbol() for a in molecule.GetAtoms()], positions=positions), path)
    return path


def check_graph(smiles, xyz):
    geometry = read(xyz)
    check = assess(smiles, geometry)
    if not check["passed"]:
        raise ValueError(f"{xyz}: {', '.join(check['errors'])}")
    return geometry


def state_free_energy(xyz, charge, mult, directory, *, smiles=None, thermal_model,
                      monatomic_protocol=None, **limits):
    """Calculate the thermal correction and final electronic energy for one solution state."""
    from core.modeling.dft import optimize_state
    from core.modeling.sp import electronic_checks, make_mf, single_point

    geometry = read(xyz)
    if len(geometry) == 1:
        # A monatomic ion needs only the translational and electronic thermal terms.
        from pyscf.hessian import thermo
        from core.protocol import GEOMETRY

        if monatomic_protocol is None:
            raise ValueError("a matched molecular thermal protocol is required for an isolated ion")
        mf = make_mf(geometry, charge, mult, {**GEOMETRY, "basis": "def2-svpd"}, **limits)
        mf.kernel()
        electronic_checks(mf)
        temperature = monatomic_protocol["temperature_K"]
        model = SimpleNamespace(mol=mf.mol, e_tot=float(mf.e_tot))
        partition = thermo.thermo(model, np.array([]), temperature, monatomic_protocol["pressure_Pa"])
        correction = (partition["G_trans"][0] - temperature * partition["S_elec"][0]) * HARTREE_EV
        thermal = {"thermal_status": "passed", "thermal_protocol": monatomic_protocol,
                   "g_thermal_eV": float(correction)}
        geometry_path = str(Path(xyz).resolve())
        sources = {}
    else:
        thermal = optimize_state(xyz, charge, mult, directory=directory,
                                 thermal=True, thermal_model=thermal_model, **limits)
        geometry_path = thermal["geometry"]
        result_path = directory / "result.json"
        sources = {**thermal["sources"], str(result_path.resolve()): file_hash(result_path)}
    if smiles is not None:
        geometry = check_graph(smiles, geometry_path)
    else:
        geometry = read(geometry_path)
        if Counter(geometry.get_chemical_symbols()) != Counter({"Fe": 1, "C": 10, "H": 10}):
            raise ValueError("expected the curated ferrocene reference geometry")
    electronic = single_point(geometry_path, charge, mult, out=directory / "electronic.json", **limits)
    record = {
        **electronic,
        "identity": canonical_smiles(smiles) if smiles is not None else "ferrocene",
        "composition": dict(Counter(geometry.get_chemical_symbols())),
        "sources": {**electronic["sources"], **sources},
    }
    record.update({name: thermal[name] for name in ("thermal_status", "thermal_protocol", "g_thermal_eV")})
    return write_record(directory / "free_energy.json", record)


def input_hashes(*paths):
    return {str(path.resolve()): file_hash(path) for path in paths}
