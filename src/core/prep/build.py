"""Generate conformers and write the molecular library."""

import csv
import hashlib
import math
import json
from datetime import datetime, timezone

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from core import paths


def check_stereo(mol: Chem.Mol):
    """Count unspecified stereocenters and double-bond stereochemistry."""
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


def n_conformers(mol):
    """Choose an embedding budget based on the number of rotatable bonds."""
    n_rot = Descriptors.NumRotatableBonds(mol)
    if n_rot <= 3:
        return 25
    if n_rot <= 6:
        return 75
    return 150


def conformer_ensemble(mol, top_k=10, seed=0xC0FFEE, prune_rms=0.1):
    """Return the lowest-energy converged conformers after RMSD pruning."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    molH = Chem.AddHs(mol)
    n_guess = n_conformers(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.useRandomCoords = False

    conformer_ids = list(AllChem.EmbedMultipleConfs(molH, params=params, numConfs=n_guess))

    if not conformer_ids:
        params.useRandomCoords = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(molH, params=params, numConfs=n_guess))
    if not conformer_ids:
        raise RuntimeError(f"Embedding failed for {Chem.MolToSmiles(mol)}")

    net_charge = Chem.GetFormalCharge(mol)
    max_iters = 1000
    if net_charge != 0:
        max_iters = 200
    if AllChem.MMFFHasAllMoleculeParams(molH):
        ff_name = "MMFF"
        results = AllChem.MMFFOptimizeMoleculeConfs(molH, maxIters=max_iters)
    else:
        ff_name = "UFF"
        results = AllChem.UFFOptimizeMoleculeConfs(molH, maxIters=max_iters)

    converged = [
        (conf_id, energy)
        for conf_id, (status, energy) in zip(conformer_ids, results)
        if status == 0 and math.isfinite(energy)
    ]
    if not converged:
        raise RuntimeError("No force-field conformers converged. Increase the optimization steps.")

    converged.sort(key=lambda conformer: conformer[1])
    kept_ids, kept_energies = zip(*converged[:top_k])
    return molH, list(kept_ids), ff_name, list(kept_energies)


MANIFEST_FIELDS = ["id", "name", "family", "state", "charge",
                   "mult_hint", "n_e", "n_conf", "solv_preopt", "smiles"]


def structure_id(smiles):
    """Return a short identifier for the canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def monomer_id(families, smiles):
    """Build an identifier from the redox families and canonical structure."""
    if not families:
        raise ValueError("monomer_id() needs at least one family; call identify() first")
    return "+".join(families) + "_" + structure_id(smiles)


def to_xyz(molH, comment=""):
    """Write one RDKit conformer as XYZ text."""
    conformer = molH.GetConformer()
    lines = [str(molH.GetNumAtoms()), comment]
    for atom in molH.GetAtoms():
        p = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines) + "\n"


def state_label(n_e):
    """Label the charge change relative to the resting state."""
    if n_e == 0:
        return "rest"
    return f"red{abs(n_e)}" if n_e < 0 else f"ox{n_e}"


def write_monomer(monomer, top_k=10, seed=0xC0FFEE):
    """Write conformers and return the corresponding manifest rows."""
    mol = monomer.to_mol()
    if mol is None:
        raise ValueError(f"bad grafted SMILES: {monomer.smiles!r}")

    n_unspec = check_stereo(mol)
    if n_unspec:
        raise ValueError(
            f"{monomer.name or monomer.smiles}: {n_unspec} unspecified stereo element(s)"
        )

    if not monomer.charge_states:
        raise ValueError("monomer has no states; call initialize_states() first")

    missing = [q for q in monomer.charge_states if q not in monomer.spin_map]
    if missing:
        raise ValueError(f"spin_map is missing charge states: {missing}")

    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    dir_id = monomer_id(monomer.families, smi)
    if monomer.id and monomer.id != dir_id:
        raise ValueError(
            f"stale monomer.id {monomer.id!r}; structure id is {dir_id!r}"
        )
    monomer.id = dir_id

    monomer_dir = paths.LIBRARY / dir_id
    monomer_dir.mkdir(parents=True, exist_ok=True)

    (monomer_dir / f"{dir_id}.smiles").write_text(smi + "\n")
    (monomer_dir / "meta.json").write_text(json.dumps({
        "id": dir_id,
        "name": monomer.name,
        "families": list(monomer.families),
        "fragment_families": list(monomer.fragment_families),
        "attachment": monomer.attachment,
        "fragment": monomer.fragment_smiles,
        "smiles": smi,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")

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


def write_manifest(rows, path=None):
    """Replace the manifest with the supplied rows."""
    path = path or (paths.LIBRARY / "manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_manifest(path=None):
    """Read the manifest or return an empty list if it does not exist."""
    path = path or (paths.LIBRARY / "manifest.csv")
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def upsert_manifest(rows, path=None):
    """Replace rows for the supplied monomer IDs."""
    path = path or (paths.LIBRARY / "manifest.csv")
    new_ids = {r["id"] for r in rows}
    kept = [r for r in load_manifest(path) if r["id"] not in new_ids]
    merged = kept + rows
    merged.sort(key=lambda r: (r["id"], int(r["n_e"])))
    write_manifest(merged, path)


def write_library(monomers, top_k=10, seed=0xC0FFEE):
    """Build the monomers and replace the manifest."""
    rows = []
    for monomer in monomers:
        rows.extend(write_monomer(monomer, top_k=top_k, seed=seed))
    write_manifest(rows)
    return rows


def write_dataset(monomers, top_k=10, seed=0xC0FFEE):
    """Build the monomers and merge their rows into the manifest."""
    rows = []
    for monomer in monomers:
        rows.extend(write_monomer(monomer, top_k=top_k, seed=seed))
    upsert_manifest(rows)
    return rows
