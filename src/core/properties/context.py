"""Check that calculation records can be compared before using their energies."""
from core.paths import to_float


def require_compatible(records, *, thermodynamic=False):
    if not records:
        raise ValueError("at least one record required")
    fields = ["method"] + (["thermal_protocol"] if thermodynamic else [])
    for record in records:
        if record.get("status") != "complete":
            raise ValueError("incomplete calculation")
        for field in fields:
            if not record.get(field) or record[field] != records[0].get(field):
                raise ValueError(f"missing or mismatched {field}")
        if thermodynamic and record.get("thermal_status") != "passed":
            raise ValueError("missing checked thermal correction")


def energy(record, *, free=False):
    if free:
        from core.paths import free_energy
        return free_energy(record)
    return to_float(record["energy_eV"])
