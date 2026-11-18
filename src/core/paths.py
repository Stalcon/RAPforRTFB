"""Shared units, paths, and file operations."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(os.environ.get("RAP_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG_DIR = ROOT / "config"
LIBRARY = ROOT / "library"
CALCS = ROOT / "calcs"
UMA, DFT, SP, SPINCHECK = (CALCS / n for n in ("uma", "dft", "sp", "spincheck"))
RESULTS = ROOT / "results"
HARTREE_EV = 27.211386245988
EV_KJ = 96.4853321233
KB_EV = 8.617333262145e-5
KT_EV = KB_EV * 298.15
FARADAY = 96485.3321233
MANIFEST_FIELDS = ["id", "name", "family", "state", "charge", "mult_hint",
                   "n_e", "n_conf", "solv_preopt", "smiles"]


def safe_name(value):
    value = str(value)
    if not value or value in (".", "..") or any(c not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-." for c in value):
        raise ValueError(f"unsafe record name: {value!r}")
    return value


def load_config(name):
    name = safe_name(name)
    path = CONFIG_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    path = CONFIG_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"rap_config_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if k.isupper()}


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_smiles(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def structure_id(smiles):
    return hashlib.sha1(canonical_smiles(smiles).encode()).hexdigest()[:12]


def state_label(n_e):
    return "rest" if n_e == 0 else f"{'red' if n_e < 0 else 'ox'}{abs(n_e)}"


def read_manifest(path=None):
    path = Path(path or LIBRARY / "manifest.csv")
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = (safe_name(row["id"]), safe_name(row["state"]))
        if key in seen:
            raise ValueError(f"duplicate manifest state: {key}")
        seen.add(key)
    return rows


def write_manifest(rows, path=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path or LIBRARY / "manifest.csv", stream.getvalue())


def upsert_manifest(rows, path=None):
    ids = {r["id"] for r in rows}
    merged = [r for r in read_manifest(path) if r["id"] not in ids] + rows
    write_manifest(sorted(merged, key=lambda r: (r["id"], int(r["n_e"]))), path)


def read_result(gid, state, root=DFT):
    path = Path(root) / safe_name(gid) / safe_name(state) / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def to_float(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a physical quantity")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite physical quantity")
    return number


def free_energy(record):
    """Never substitute zero for absent thermal terms."""
    if not record or record.get("status") != "complete" or record.get("thermal_status") != "passed":
        raise ValueError("complete electronic and checked thermal records required")
    return to_float(record["e_smd_eV"]) + to_float(record["g_thermal_eV"])


def group_ids(rows=None):
    return sorted({r["id"] for r in (read_manifest() if rows is None else rows)})


def state_names(gid, rows=None):
    return [r["state"] for r in (read_manifest() if rows is None else rows) if r["id"] == gid]


def write_xyz(atoms, path, comment=""):
    lines = [str(len(atoms)), comment]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"{symbol} " + " ".join(f"{x:.12f}" for x in position))
    atomic_write_text(path, "\n".join(lines) + "\n")


def smiles_to_xyz_map(smiles, atoms):
    """Use preserved RDKit AddHs order; reject reordered XYZ inputs."""
    from rdkit import Chem
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles(smiles)))
    if [a.GetSymbol() for a in molecule.GetAtoms()] != atoms.get_chemical_symbols():
        raise ValueError("XYZ atom order differs from the declared SMILES order")
    return {i: i for i in range(molecule.GetNumAtoms())}


def redox_core_xyz_indices(smiles, atoms, core_smarts):
    from rdkit import Chem
    mapping = smiles_to_xyz_map(smiles, atoms)
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles(smiles)))
    pattern = Chem.MolFromSmarts(core_smarts)
    if pattern is None:
        raise ValueError("invalid redox core SMARTS")
    matches = molecule.GetSubstructMatches(pattern)
    if len(matches) != 1:
        raise ValueError("provide an unambiguous redox core mapping")
    return [mapping[i] for i in matches[0]]
