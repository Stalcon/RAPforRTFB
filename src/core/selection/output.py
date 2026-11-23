"""Compare candidate trade-offs and write CSV tables."""
import csv
import io
from core.paths import atomic_write_text, to_float


def pareto_shortlist(rows, objectives):
    """Objectives map property names to min/max. Missing objectives are excluded."""
    if not objectives or any(v not in ("min", "max") for v in objectives.values()):
        raise ValueError("declare min/max for each objective")
    eligible = [r for r in rows if r.get("status") == "eligible"]
    if not eligible:
        return []
    for field in ("electrode_pool", "operating_context", "route_context"):
        if not eligible[0].get(field) or any(r.get(field) != eligible[0][field] for r in eligible):
            raise ValueError(f"separate Pareto comparisons by {field}")
    return _pareto_front(eligible, objectives)


def raw_pareto_front(candidates, *, pool, e_key="E", cap_key="cap"):
    """Return the voltage–capacity front for a set already chosen for one electrode.

    Pass candidates from the same operating context and mass convention.
    Rows with missing objective values are skipped. Ties keep their input order.
    """
    if pool not in ("anolyte", "catholyte"):
        raise ValueError("pool must be anolyte or catholyte")
    if e_key == cap_key:
        raise ValueError("voltage and capacity must use different fields")
    voltage_direction = "min" if pool == "anolyte" else "max"
    return _pareto_front(candidates, {e_key: voltage_direction, cap_key: "max"})


def _pareto_front(rows, objectives):
    # Orient all objectives so a smaller value is preferred.
    points = []
    for row in rows:
        if any(row.get(key) is None for key in objectives):
            continue
        values = [to_float(row[key]) * (1 if direction == "min" else -1)
                  for key, direction in objectives.items()]
        points.append((row, values))
    return [row for row, values in points
            if not any(_dominates(other_values, values) for _, other_values in points)]


def _dominates(left, right):
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def write_table(rows, path):
    if not rows:
        atomic_write_text(path, "")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())
