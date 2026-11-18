"""Small command-line interface for the staged molecular workflow."""
import argparse
import json
import sys
from .paths import LIBRARY, RESULTS, atomic_write_json, read_manifest, safe_name, load_config, upsert_manifest


def prepare(smiles, name=None, *, graft=False, conformers=1):
    from rdkit import Chem
    from .prep.screen import RedoxMonomer, screen_mol, identify, build_states
    from .prep.build import write_dataset
    if graft:
        monomers = screen_mol(smiles, name)
    else:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid SMILES")
        families = identify(molecule)
        monomer = RedoxMonomer(smiles, smiles, families, families, "none", name)
        charges, spins = build_states(molecule, families)
        monomer.initialize_states(charges, spins)
        monomers = [monomer]
    if not monomers:
        raise ValueError("no eligible graph attachment generated")
    for monomer in monomers:
        molecule = Chem.AddHs(monomer.to_mol())
        electrons = sum(a.GetAtomicNum() for a in molecule.GetAtoms())
        from .modeling.uma import candidate_mults
        for charge in monomer.charge_states:
            candidate_mults(electrons - charge, monomer.spin_map[charge])
    rows = write_dataset(monomers, top_k=conformers)
    return {"states": rows, "scope": "declared state hypotheses; no route or material admission"}


def prepare_config(name, conformers=1):
    from rdkit import Chem
    from .prep.screen import RedoxMonomer, identify, build_states, screen_mol
    from .prep.build import write_monomer
    from .modeling.uma import candidate_mults
    config = load_config(name)
    groups = config.get('GROUPS', config.get('groups', []))
    if not groups:
        raise ValueError('the selected cohort is empty; supply exact graphs first')
    rows = []
    for group in groups:
        fragment = Chem.MolFromSmiles(group['frag'])
        if fragment is None:
            raise ValueError(f"invalid fragment: {group['frag']}")
        if any(a.GetAtomicNum() == 0 for a in fragment.GetAtoms()):
            scaffold = Chem.MolFromSmiles(config.get('SCAFFOLD', load_config('redox_groups')['SCAFFOLD']))
            product = Chem.molzip(Chem.CombineMols(fragment, scaffold))
            Chem.SanitizeMol(product)
            if any(a.GetAtomicNum() == 0 for a in product.GetAtoms()) or len(Chem.GetMolFrags(product)) != 1:
                raise ValueError('fragment/scaffold dummy maps must form one exact product')
            monomers = [RedoxMonomer(group['frag'], Chem.MolToSmiles(product), identify(fragment),
                                    identify(product), 'curated mapped attachment', group.get('name'))]
        elif group.get('as_product', False):
            monomers = [RedoxMonomer(group['frag'], Chem.MolToSmiles(fragment), identify(fragment),
                                    identify(fragment), 'exact supplied graph', group.get('name'))]
        else:
            monomers = screen_mol(group['frag'], group.get('name'))
        if not monomers:
            raise ValueError(f"no product for {group.get('name', group['frag'])}")
        for monomer in monomers:
            if group.get('family'):
                monomer.families = [safe_name(group['family'])]
                monomer.id = None
            if group.get('states'):
                states = group['states']
                q0 = Chem.GetFormalCharge(monomer.to_mol())
                if len({s[1] for s in states}) != len(states) or len({s[0] for s in states}) != len(states):
                    raise ValueError('state charges and labels must be unique')
                if states[0][1] != q0 or states[0][3] != 0:
                    raise ValueError('first state must be the graph resting charge with n_e=0')
                electrons = sum(a.GetAtomicNum() for a in Chem.AddHs(monomer.to_mol()).GetAtoms())
                for label, charge, mult, delta in states:
                    safe_name(label)
                    if charge - q0 != delta:
                        raise ValueError('n_e is signed charge change from the resting state')
                    candidate_mults(electrons - charge, mult)
                monomer.initialize_states([s[1] for s in states], {s[1]: s[2] for s in states})
            elif not monomer.charge_states:
                monomer.initialize_states(*build_states(monomer.to_mol(), monomer.families))
            built = write_monomer(monomer, top_k=conformers)
            if group.get('states'):
                labels = {s[1]: s[0] for s in group['states']}
                for row in built:
                    row['state'] = labels[row['charge']]
            rows.extend(built)
    upsert_manifest(rows)
    return {'states': rows, 'scope': 'curated graph/state hypotheses, not synthesis admission'}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rap")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="report actual backend capabilities")
    p = commands.add_parser("prepare", help="prepare exact supplied graph, optionally graft a precursor")
    p.add_argument("--smiles", required=True)
    p.add_argument("--name")
    p.add_argument("--graft", action="store_true")
    p.add_argument("--conformers", type=int, default=1)
    p = commands.add_parser('prepare-config', help='build a curated GROUPS configuration with state overrides')
    p.add_argument('name', choices=('redox_groups', 'starting_candidates'))
    p.add_argument('--conformers', type=int, default=1)
    for name in ("uma", "dft", "sp"):
        p = commands.add_parser(name)
        p.add_argument("--id", action="append", dest="ids")
        p.add_argument("--state", action="append", dest="states")
        p.add_argument("--limit", type=int)
        p.add_argument("--threads", type=int, default=4)
        if name == "uma":
            p.add_argument("--model", default="uma-s-1p2")
            p.add_argument("--fmax", type=float, default=0.02)
            p.add_argument("--steps", type=int, default=300)
            p.add_argument("--scan-spins", action="store_true")
        else:
            p.add_argument("--memory-mb", type=int, default=4000)
        if name == "dft":
            p.add_argument("--from-library", action="store_true")
            p.add_argument("--thermal", action="store_true")
            p.add_argument("--thermal-model", choices=("rrho", "qrrho"), default="rrho")
            p.add_argument("--maxsteps", type=int, default=100)
    p = commands.add_parser("report", help="export state status, energies, and scope")
    p.add_argument("--out", default=str(RESULTS / "states.csv"))
    args = parser.parse_args(argv)
    try:
        if args.command == "env":
            from .backend import describe
            result = describe()
        elif args.command == "prepare":
            result = prepare(args.smiles, args.name, graft=args.graft, conformers=args.conformers)
        elif args.command == 'prepare-config':
            result = prepare_config(args.name, args.conformers)
        elif args.command == "report":
            result = report(args.out)
        else:
            from .backend import configure_threads
            configure_threads(args.threads, args.threads)
            import importlib
            stage = importlib.import_module(f"core.modeling.{args.command}")
            options = vars(args).copy()
            options.pop("command")
            if args.command == "uma":
                import torch
                torch.set_num_threads(options.pop("threads"))
            result = stage.run_batch(**options)
            if any(r["status"] != "complete" for r in result):
                return 1
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def report(path):
    from .paths import UMA, DFT, SP
    from .protocol import load_record
    from .selection.output import write_table
    rows = []
    for entry in read_manifest():
        for stage, root in (("uma", UMA), ("dft", DFT), ("sp", SP)):
            file = root / safe_name(entry["id"]) / safe_name(entry["state"]) / "result.json"
            record = load_record(file) if file.exists() else {}
            latest = file.parent / 'status.json'
            execution = json.loads(latest.read_text()).get('status') if latest.exists() else None
            rows.append({"id": entry["id"], "state": entry["state"], "stage": stage,
                         "status": record.get("status", "missing"),
                         "latest_attempt_status": execution,
                         "energy_eV": record.get("energy_eV"),
                         "g_thermal_eV": record.get("g_thermal_eV"),
                         "thermal_status": record.get("thermal_status", "not_available"),
                         "engine": record.get("method", {}).get("engine", "UMA" if stage == "uma" else ""),
                         "material_decision": "pending"})
    write_table(rows, path)
    return {"output": str(path), "rows": len(rows)}
