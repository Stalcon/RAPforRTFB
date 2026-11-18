"""Content-addressed calculation records and source checks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .paths import SP, atomic_write_json, safe_name

ACTIVE_SP = {"xc": "wb97m-v", "basis": "def2-tzvpd", "dispersion": None,
             "solvent": "acetonitrile", "density_fit": True, "only_dfj": False,
             "grid_level": 4, "conv_tol": 1e-9, "max_cycle": 200}
GEOMETRY = {**ACTIVE_SP, "xc": "r2scan", "basis": "def2-svp", "dispersion": "d4"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_hash(path):
    return file_hash(path)


def runtime_versions():
    result = {}
    for name in ("rapforrtfb", "numpy", "ase", "pyscf", "pyscf-dispersion", "gpu4pyscf", "fairchem-core", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def source_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def protocol_hash(geometry, charge, mult, settings=None):
    return digest({"geometry": geometry, "charge": int(charge), "mult": int(mult),
                   "settings": settings or ACTIVE_SP, "versions": runtime_versions(),
                   "core_source": source_hash()})


def record_path(gid, state, key, root=SP):
    return Path(root) / safe_name(gid) / safe_name(state) / f"{safe_name(key)}.json"


def write_record(path, record):
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    payload["record_hash"] = digest(payload)
    atomic_write_json(path, payload)
    return payload


def record_provenance_ok(record):
    if not isinstance(record, dict) or not record.get("record_hash"):
        return False
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    if digest(payload) != record["record_hash"]:
        return False
    return all(Path(path).is_file() and file_hash(path) == expected
               for path, expected in record.get("sources", {}).items())


def load_record(path):
    record = json.loads(Path(path).read_text())
    if not record_provenance_ok(record):
        raise ValueError(f"changed record or source: {path}")
    return record


def audit(root=SP):
    return [{"path": str(path), "valid": record_provenance_ok(json.loads(path.read_text()))}
            for path in sorted(Path(root).rglob("*.json"))]
