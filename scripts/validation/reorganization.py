"""Compare gas-phase four-point reorganization energies with D3TaLES calculations."""

import argparse
import csv
from pathlib import Path

from ase.io import read

from common import OUTPUT, calculation_options, check_graph, input_hashes, resources, run_report, seed_xyz
from core.modeling.dft import optimize_state
from core.paths import ROOT, to_float
from core.properties.reorg import calculate_pair, nelsen_4pt

D3TALES = ROOT / "data" / "D3TaLES" / "d3tales_public.csv"
CHANNELS = {"hole": (1, "cation1"), "electron": (0, "anion1")}
# Allow for rounding in the two component energies and the reported total.
ROUNDING_TOLERANCE_EV = 0.00015


def run(args):
    with args.dataset.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["_id"] == args.id]
    if len(rows) != 1:
        raise ValueError(f"expected one D3TaLES record for {args.id}, got {len(rows)}")
    row = rows[0]
    if to_float(row["groundState_charge"]) != 0:
        raise ValueError("this comparator requires a neutral D3TaLES ground state")
    q_ox, ion = CHANNELS[args.channel]
    reference = to_float(row[f"{args.channel}_reorganization_energy"])
    ion_relaxation = to_float(row[f"relaxation_groundState_{ion}"])
    neutral_relaxation = to_float(row[f"relaxation_{ion}_groundState"])
    ox_dist, red_dist = ((ion_relaxation, neutral_relaxation) if args.channel == "hole"
                         else (neutral_relaxation, ion_relaxation))
    replay = nelsen_4pt(0.0, ox_dist, 0.0, red_dist, surface="gas")
    error = replay["four_point_sum_eV"] - reference
    report = {"status": "complete", "id": args.id, "smiles": row["smiles"],
              "mode": args.mode, "channel": args.channel, "reference_eV": reference,
              "reference_replay": replay, "reference_rounding_error_eV": error,
              "scope": "Gas-phase computed comparator at a different electronic level.",
              "input_hashes": input_hashes(args.dataset)}
    if not replay["minimum_consistent"] or abs(error) > ROUNDING_TOLERANCE_EV + 1e-9:
        raise ValueError("D3TaLES directional terms are negative or inconsistent with the total")
    if args.mode == "replay":
        return report

    directory = args.work_dir / args.id / args.channel
    seed = seed_xyz(row["smiles"], directory / "seed.xyz")
    electron_count = int(sum(read(seed).numbers))
    limits = resources(args)
    minima, mults = {}, {}
    for label, charge in (("oxidized", q_ox), ("reduced", q_ox - 1)):
        mults[label] = 1 + (electron_count - charge) % 2
        minima[label] = optimize_state(
            seed, charge, mults[label], directory=directory / label,
            settings={"solvent": None, "basis": "def2-svpd" if charge < 0 else "def2-svp"},
            **limits,
        )
        check_graph(row["smiles"], minima[label]["geometry"])
    comparison = calculate_pair(
        minima["oxidized"]["geometry"], minima["reduced"]["geometry"], q_ox,
        mults["oxidized"], mults["reduced"], directory=directory / "cross_points",
        settings={"solvent": None}, **limits,
    )
    report.update({
        "status": "complete" if comparison["minimum_consistent"] else "invalid_minimum_ordering",
        "fresh_result": comparison, "fresh_minus_reference_eV": comparison["four_point_sum_eV"] - reference,
        "geometry_records": {label: {"hash": data["record_hash"], "method": data["method"]}
                             for label, data in minima.items()},
        "spin_policy": "lowest multiplicity of correct electron parity; spin ordering unchecked",
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=D3TALES)
    parser.add_argument("--id", default="06ASFG", help="Exact D3TaLES ID (default: benzene).")
    parser.add_argument("--channel", choices=CHANNELS, default="hole")
    parser.add_argument("--mode", choices=("replay", "fresh"), default="replay")
    parser.add_argument("--work-dir", type=Path, default=OUTPUT / "reorganization_states")
    parser.add_argument("--out", type=Path, default=OUTPUT / "reorganization.json")
    calculation_options(parser)
    raise SystemExit(run_report(parser, run))
