import csv
import math
from pathlib import Path
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import paths
import hashlib

## Counts unspecified stereo elements (chiral centers and E/Z); symmetry-equivalent centers are ignored
def check_stereo(mol:Chem.Mol):
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    count = 0
    for element in Chem.FindPotentialStereo(mol):
        if element.specified == Chem.StereoSpecified.Specified:
            continue
        if element.type.name.startswith("Atom"):
            atom = mol.GetAtomWithIdx(element.centeredOn)
            neighbor_ranks = []
            for neighbor in atom.GetNeighbors():
                neighbor_rank = ranks[neighbor.GetIdx()]
                neighbor_ranks.append(neighbor_rank)
            if neighbor_ranks and len(set(neighbor_ranks)) == 1:
                continue
        count += 1
    return count

## Estimates the number of conformers of a molecule's rotatable bond count
def n_conformers(mol):
    n_rot = Descriptors.NumRotatableBonds(mol)
    if n_rot <= 3:
        return 25
    if n_rot <= 6:
        return 75
    return 150

## Embeds, relaxes, prunes duplicates and returns the top-k lowest-FF-energy conformers
def conformer_ensemble(mol, top_k=10, seed=0xC0FFEE, prune_rms=0.1):
    molH = Chem.AddHs(mol)
    n_guess = n_conformers(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.useRandomCoords = False

    conformer_ids = list(AllChem.EmbedMultipleConfs(molH,params=params,numConfs=n_guess))

    if not conformer_ids:
        params.useRandomCoords = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(molH,params=params,numConfs=n_guess))
    if not conformer_ids:
        raise RuntimeError(f"Embedding failed for {Chem.MolToSmiles(mol)}")

    net_charge = Chem.GetFormalCharge(mol)
    max_iters = 1000
    if net_charge != 0:
        max_iters = 200
    if AllChem.MMFFHasAllMoleculeParams(molH):
        ff_name = "MMFF"
        results = AllChem.MMFFOptimizeMoleculeConfs(molH,maxIters=max_iters)
    else:
        ff_name = "UFF"
        results = AllChem.UFFOptimizeMoleculeConfs(molH,maxIters=max_iters)

    energies = []
    for converged, energy in results:
        energies.append(energy)

    order = sorted(range(len(conformer_ids)), key = lambda i: energies[i])[:top_k]
    kept_ids = []
    kept_energies = []
    for i in order:
        kept_ids.append(conformer_ids[i])
        kept_energies.append(energies[i])

    return molH, kept_ids, ff_name, kept_energies

MANIFEST_FIELDS = ["id", "name", "family", "state", "charge",
                   "mult_hint", "n_e", "n_conf", "solv_preopt", "smiles"]

## Generates unique identifier for a monomer
def monomer_id(monomer):
    mol = monomer.to_mol()
    if mol is None:
        raise ValueError(f"bad grafted SMILES: {monomer.smiles!r}")
    smi = Chem.MolToSmiles(mol)
    return "mono-" + hashlib.sha1(smi.encode()).hexdigest()[:12]

## Turns conformer into xyz text
def to_xyz(molH, comment=""):
    conformer = molH.GetConformer()
    lines = [str(molH.GetNumAtoms()), comment]
    for atom in molH.GetAtoms():
        p = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines) + "\n"

## Maps electrons transferred from the resting state
def state_label(n_e):
    if n_e == 0:
        return "rest"
    return f"red{abs(n_e)}" if n_e < 0 else f"ox{n_e}"

## Writes monomer directory (conformer xyz text) and returns state manifest rows
def write_monomer(monomer, top_k=10, seed=0xC0FFEE):
    mol = monomer.to_mol()
    if mol is None:
        raise ValueError(f"bad grafted SMILES: {monomer.smiles!r}")

    n_unspec = check_stereo(mol)
    if n_unspec:
        raise ValueError(
            f"{monomer.name or monomer.smiles}: {n_unspec} unspecified stereo element(s)"
        )

    if not monomer.charge_states:
        raise ValueError("monomer has no states")

    dir_id = monomer.id or monomer_id(monomer)
    monomer_dir = paths.LIBRARY / dir_id
    monomer_dir.mkdir(parents=True, exist_ok=True)

    smi = Chem.MolToSmiles(mol)
    (monomer_dir / f"{dir_id}.smiles").write_text(smi + "\n")

    molH, conformer_ids, ff_name, energies = conformer_ensemble(mol, top_k=top_k, seed=seed)

    conformer_dir = monomer_dir / "conformers"
    conformer_dir.mkdir(exist_ok=True)
    for stale in conformer_dir.glob("conf_*.xyz"):
        stale.unlink()

    for i, (conformer_id, energy) in enumerate(zip(conformer_ids, energies)):
        single = Chem.Mol(molH)
        single.RemoveAllConformers()
        single.AddConformer(molH.GetConformer(conformer_id), assignId=True)
        (conformer_dir / f"conf_{i:02d}.xyz").write_text(
            to_xyz(single, f"conf {i} ff={ff_name} E={energy:.3f} smiles={smi}")
        )

    q0 = Chem.GetFormalCharge(mol)
    rows = []
    for charge in monomer.charge_states:
        n_e = charge - q0
        rows.append(dict(
            id=dir_id,
            name=monomer.name or dir_id,
            family="+".join(monomer.families),
            state=state_label(n_e),
            charge=charge,
            mult_hint=monomer.spin_map[charge],
            n_e=n_e,
            n_conf=len(conformer_ids),
            solv_preopt=int(charge != 0),
            smiles=smi,
        ))
    return rows

## Replaces manifest.csv with given rows
def write_manifest(rows, path=None):
    path = path or (paths.LIBRARY / "manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

## Writes all given monomers's directories and write manifest.csv
def write_library(monomers, top_k=10, seed=0xC0FFEE):
    rows = []
    for monomer in monomers:
        rows.extend(write_monomer(monomer, top_k=top_k, seed=seed))
    write_manifest(rows)
    return rows

def load_manifest(path=None):
    path = path or (paths.LIBRARY / "manifest.csv")
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))

## Replace given rows and keeps other ids
def upsert_manifest(rows, path=None):
    path = path or (paths.LIBRARY / "manifest.csv")
    new_ids = {r["id"] for r in rows}
    kept = [r for r in load_manifest(path) if r["id"] not in new_ids]
    merged = kept + rows
    merged.sort(key=lambda r: (r["id"], r["n_e"]))   # stable diffs across runs
    write_manifest(merged, path)

## Screen+embed one dataset and merge it inot the mainfest
def write_dataset(monomers, top_k=10, seed=0xC0FFEE):
    rows = []
    for m in monomers:
        rows.extend(write_monomer(m, top_k=top_k, seed=seed))
    upsert_manifest(rows)
    return rows

    