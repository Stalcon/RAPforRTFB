"""Compare SMD hydration energies with measured values for neutral molecules."""

import argparse
import csv
from math import sqrt
from pathlib import Path
from statistics import mean

from rdkit import Chem

from common import (OUTPUT, REFERENCE_DATA, calculation_options, check_graph, input_hashes,
                     resources, run_report, seed_xyz)
from core.modeling.dft import optimize_state
from core.modeling.sp import single_point
from core.paths import EV_KJ, to_float


def run(args):
    path = REFERENCE_DATA / "hydration.csv"
    with path.open(newline="") as stream:
        panel = {row["id"]: row for row in csv.DictReader(stream)}
    if args.ids:
        unknown = set(args.ids) - panel.keys()
        if unknown:
            raise ValueError(f"unknown hydration IDs: {', '.join(sorted(unknown))}")
    chosen = list(panel) if args.all else args.ids
    results, failures = [], {}
    limits = resources(args)
    for gid in dict.fromkeys(chosen):
        row = panel[gid]
        try:
            molecule = Chem.MolFromSmiles(row["smiles"])
            if (row["solvent"] != "water" or Chem.GetFormalCharge(molecule) != 0
                    or any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())):
                raise ValueError("expected a closed-shell neutral hydration reference")
            directory = args.work_dir / gid
            seed = seed_xyz(row["smiles"], directory / "seed.xyz")
            geometry = optimize_state(seed, 0, 1, directory=directory / "gas_opt",
                                      settings={"solvent": None}, **limits)
            xyz = geometry["geometry"]
            check_graph(row["smiles"], xyz)
            gas = single_point(xyz, 0, 1, settings={"solvent": None},
                               out=directory / "gas.json", **limits)
            water = single_point(xyz, 0, 1, settings={"solvent": "water"},
                                 out=directory / "smd_water.json", **limits)
            calculated = (water["energy_eV"] - gas["energy_eV"]) * EV_KJ / 4.184
            observed = to_float(row["exp_kcal"])
            results.append({"id": gid, "smiles": row["smiles"],
                            "experimental_kcal_mol": observed, "calc_kcal_mol": calculated,
                            "experimental_uncertainty_kcal_mol": to_float(row["exp_unc"]) if row["exp_unc"] else None,
                            "error_kcal_mol": calculated - observed, "geometry_record_hash": geometry["record_hash"],
                            "gas_record_hash": gas["record_hash"], "water_record_hash": water["record_hash"]})
        except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
            failures[gid] = str(exc)
    errors = [row["error_kcal_mol"] for row in results]
    metrics = {"n": len(errors)}
    if errors:
        metrics.update(MAE_kcal_mol=mean(map(abs, errors)),
                       RMSE_kcal_mol=sqrt(mean(x * x for x in errors)), bias_kcal_mol=mean(errors))
    return {"status": "incomplete" if failures else "complete", "metrics": metrics,
            "scope": "Water SMD minus gas on each gas minimum, with a common concentration convention. "
                     "Solvent-model control for small neutral solutes.",
            "input_hashes": input_hashes(path), "results": results, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--id", dest="ids", action="append", help="Select an exact reference ID, repeatable.")
    selection.add_argument("--all", action="store_true", help="Run every water reference.")
    parser.add_argument("--work-dir", type=Path, default=OUTPUT / "hydration_states")
    parser.add_argument("--out", type=Path, default=OUTPUT / "hydration.json")
    calculation_options(parser)
    raise SystemExit(run_report(parser, run))
