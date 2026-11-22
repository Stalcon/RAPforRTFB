"""Compare proton-transfer energies with reported basicity values in acetonitrile."""

import argparse
import json
from math import log
from pathlib import Path

from scipy.constants import R

from common import (OUTPUT, REFERENCE_DATA, calculation_options, input_hashes, resources, run_report,
                     seed_xyz, state_free_energy)
from core.properties.polymer import reaction_delta_g

BASICITY = REFERENCE_DATA / "basicity.json"
RELATIVE = {"dipea": -1, "dbuh": -1, "dipeah": 1, "dbu": 1}


def capture(base):
    return {base: -1, "hcl": -1, base + "h": 1, "chloride": 1}


def run(args):
    data = json.loads(BASICITY.read_text())
    reference_values = data["reference_values"]
    temperature = data["temperature_K"]
    hcl = reference_values["HCl_pKa"]
    dbu = reference_values["DBUH_pKa"]
    dipea = reference_values["DIPEA_log_KH"]
    factor = R * temperature * log(10) / 1000

    references = {base: -factor * (pka - hcl) for base, pka in (("dbu", dbu), ("dipea", dipea))}
    references["relative"] = factor * (dbu - dipea)
    computed = data["archived_kJ_mol"].copy()
    state_hashes = {}
    if args.mode == "fresh":
        states = {}
        # Run HCl first so chloride can use the same thermal settings.
        for name in ("dbu", "dbuh", "dipea", "dipeah", "hcl", "chloride"):
            smiles, charge = data["species"][name]
            directory = args.work_dir / name
            seed = seed_xyz(smiles, directory / "seed.xyz")
            states[name] = state_free_energy(
                seed, charge, 1, directory, smiles=smiles, thermal_model=args.thermal_model,
                monatomic_protocol=states["hcl"]["thermal_protocol"] if name == "chloride" else None,
                **resources(args),
            )
        for base, stoichiometry in (("dbu", capture("dbu")), ("dipea", capture("dipea")),
                                    ("relative", RELATIVE)):
            computed[base] = reaction_delta_g(states, stoichiometry)["delta_G_kJ_mol"]
        state_hashes = {name: state["record_hash"] for name, state in states.items()}

    return {
        "status": "complete", "mode": args.mode,
        "scope": "Conditional comparison of separated-ion proton transfer; activity and indicator conventions differ.",
        "temperature_K": temperature,
        "reference_values": reference_values,
        "input_hashes": input_hashes(BASICITY), "state_record_hashes": state_hashes,
        "results": [{"comparison": key, "stoichiometry": RELATIVE if key == "relative" else capture(key),
                     "computed_kJ_mol": value, "literature_implied_kJ_mol": references[key],
                     "difference_kJ_mol": value - references[key]} for key, value in computed.items()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "fresh"), default="replay")
    parser.add_argument("--thermal-model", choices=("qrrho", "rrho"), default="qrrho")
    parser.add_argument("--work-dir", type=Path, default=OUTPUT / "basicity_states")
    parser.add_argument("--out", type=Path, default=OUTPUT / "basicity.json")
    calculation_options(parser)
    raise SystemExit(run_report(parser, run))
