"""Compare calculated redox potentials with measured values in acetonitrile."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from math import sqrt

from ase.io import read

from common import (OUTPUT, REFERENCE_DATA, calculation_options, input_hashes, resources,
                     run_report, seed_xyz, state_free_energy)
from core.paths import to_float
from core.properties.potentials import disproportionation, redox_potential

PANEL = REFERENCE_DATA / "redox.json"
FC_SEEDS = REFERENCE_DATA / "ferrocene"


def endpoints(row):
    oxidized, reduced = row["event"].split("->")
    return row["id"], oxidized, reduced


def first_charge(row):
    """Read the charge of the oxidized state from its label."""
    _, state, _ = endpoints(row)
    if state.startswith("ox"):
        return int(state[2:] or "1")
    if state == "neu":
        return 0
    if state.startswith("red"):
        return -int(state[3:])
    raise ValueError(f"unknown redox state {state}")


def scores(rows):
    errors = [r["error_V"] for r in rows]
    if not errors:
        return {"n": 0}
    return {"n": len(errors), "MAE_V": mean(map(abs, errors)),
            "RMSE_V": sqrt(mean(e * e for e in errors)), "bias_V": mean(errors),
            "max_abs_error_V": max(map(abs, errors))}


def summarize(rows):
    by_family, by_wave, by_study, pairs = (defaultdict(list) for _ in range(4))
    for row in rows:
        by_family[row["family"]].append(row)
        by_wave[str(row["wave"])].append(row)
        by_study[row["study_id"]].append(row)
        pairs[row["id"], row["study_id"]].append(row)
    spacings = []
    for (molecule, study_id), group in pairs.items():
        if sorted(r["wave"] for r in group) != [1, 2]:
            continue
        first, second = sorted(group, key=lambda r: r["wave"])
        exp = first["exp_V"] - second["exp_V"]
        calc = first["calc_V"] - second["calc_V"]
        high, low = sorted(group, key=first_charge, reverse=True)
        spacings.append({"id": molecule, "study_id": study_id, "exp_V": exp, "calc_V": calc,
                         "error_V": calc - exp,
                         "calc_disproportionation": disproportionation(high["calc_V"], low["calc_V"]),
                         "literature_implied_disproportionation": disproportionation(high["exp_V"], low["exp_V"])})
    return {"overall": scores(rows), "molecules": len({r["id"] for r in rows}),
            "studies": len(by_study), "by_family": {k: scores(v) for k, v in by_family.items()},
            "by_wave": {k: scores(v) for k, v in by_wave.items()},
            "by_study": {k: scores(v) for k, v in by_study.items()},
            "wave1_minus_wave2": spacings, "spacing_metrics": scores(spacings)}


def inputs(panel):
    data = json.loads(PANEL.read_text())[panel]
    rows = data["endpoints"]
    if panel == "primary":
        coverage = (len(rows), len({r["id"] for r in rows}), len({r["study_id"] for r in rows}))
        if coverage != (39, 30, 7) or any(row["tier"] != "A" for row in rows):
            raise ValueError("expected the frozen tier-A panel of 39 endpoints, 30 molecules, and 7 studies")
    states = {tuple(key.split(":")): to_float(value) for key, value in data["state_G_eV"].items()}
    return rows, states, data["protocol"]


def archived_record(total, identity, charge, label):
    # Replay has only the total free energy. Carry it through the usual potential formula.
    return {"status": "complete", "thermal_status": "passed", "identity": identity,
            "charge": charge, "method": {"snapshot": label, "solvent": "acetonitrile"},
            "thermal_protocol": {"snapshot": label}, "e_smd_eV": total, "g_thermal_eV": 0.0}


def run(args):
    rows, archived, label = inputs(args.panel)
    hashes = input_hashes(PANEL)
    if args.ids:
        unknown = set(args.ids) - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"unknown panel IDs: {', '.join(sorted(unknown))}")
        rows = [r for r in rows if r["id"] in args.ids]
    requirements = {("ferrocene", "ox"): (None, 1), ("ferrocene", "neu"): (None, 0)}
    for row in rows:
        gid, oxidized, reduced = endpoints(row)
        charge = first_charge(row)
        for state, q in ((oxidized, charge), (reduced, charge - 1)):
            key = gid, state
            current = row["parent_smiles"], q
            if key in requirements and requirements[key] != current:
                raise ValueError(f"inconsistent identity or charge for {gid}:{state}")
            requirements[key] = current

    records, failures = {}, {}
    for (gid, state), (smiles, charge) in requirements.items():
        key = gid, state
        if args.mode == "replay":
            records[key] = archived_record(archived[key], smiles or gid, charge, label)
            continue
        directory = args.work_dir / gid
        try:
            if gid == "ferrocene":
                seed = FC_SEEDS / ("oxidized.xyz" if charge else "neutral.xyz")
                hashes.update(input_hashes(seed))
            else:
                seed = seed_xyz(smiles, directory / "seed.xyz")
            mult = 1 + (sum(read(seed).numbers) - charge) % 2
            records[key] = state_free_energy(seed, charge, int(mult), directory / state,
                                             smiles=smiles, thermal_model=args.thermal_model,
                                             **resources(args))
        except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
            failures[f"{gid}:{state}"] = str(exc)

    results = []
    fc = [("ferrocene", state) for state in ("ox", "neu")]
    for row in rows:
        gid, oxidized, reduced = endpoints(row)
        keys = [(gid, oxidized), (gid, reduced), *fc]
        if any(key not in records for key in keys):
            continue
        calc = redox_potential(*(records[key] for key in keys))["E_vs_Fc_V"]
        if args.mode == "replay" and abs(calc - row["calc_V"]) > 1e-8:
            raise ValueError(f"archived state totals disagree with benchmark {gid}:{row['event']}")
        results.append({**row, "archived_calc_V": row["calc_V"], "calc_V": calc,
                        "error_V": calc - row["exp_V"]})
    return {"status": "complete" if len(results) == len(rows) else "incomplete",
            "mode": args.mode, "panel": args.panel,
            "protocol": label if args.mode == "replay" else "RAPforRTFB portable DFT + " + args.thermal_model,
            "fresh_spin_policy": "lowest multiplicity of correct electron parity; spin ordering unchecked"
            if args.mode == "fresh" else None,
            "scope": "Retrospective measured parent comparison in MeCN on the Fc/Fc+ scale. No fitting.",
            "input_hashes": hashes,
            "requested_endpoints": len(rows), "state_failures": failures,
            "state_record_hashes": {":".join(k): r["record_hash"] for k, r in records.items()
                                    if "record_hash" in r},
            "results": results, "summary": summarize(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("primary", "bmepz"), default="primary")
    parser.add_argument("--mode", choices=("replay", "fresh"), default="replay")
    parser.add_argument("--id", dest="ids", action="append", help="Select an exact panel ID, repeatable.")
    parser.add_argument("--thermal-model", choices=("qrrho", "rrho"), default="qrrho")
    parser.add_argument("--work-dir", type=Path, default=OUTPUT / "redox_states")
    parser.add_argument("--out", type=Path, default=OUTPUT / "redox.json")
    calculation_options(parser)
    raise SystemExit(run_report(parser, run))
