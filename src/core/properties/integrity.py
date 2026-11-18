"""Composition, connectivity, stereochemistry, and close-contact checks."""
import numpy as np


def assess(smiles, atoms, *, clash_fraction=0.55):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from ase.data import covalent_radii
    from core.paths import canonical_smiles
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles(smiles)))
    errors = []
    if [a.GetSymbol() for a in molecule.GetAtoms()] != atoms.get_chemical_symbols():
        return {"passed": False, "errors": ["composition_or_atom_order"], "stereochemistry_checked": False}
    positions = np.asarray(atoms.positions)
    if not np.isfinite(positions).all():
        return {"passed": False, "errors": ["nonfinite_coordinates"], "stereochemistry_checked": False}
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    radii = covalent_radii[atoms.numbers]
    for i in range(len(atoms)):
        for j in range(i):
            if distances[i, j] < clash_fraction * (radii[i] + radii[j]):
                errors.append(f"clash:{j}-{i}")
    block = str(len(atoms)) + "\n\n" + "\n".join(
        f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(atoms.get_chemical_symbols(), positions))
    inferred = Chem.MolFromXYZBlock(block)
    if inferred is None:
        return {"passed": False, "errors": ["xyz_parse_failed"], "stereochemistry_checked": False}
    rdDetermineBonds.DetermineConnectivity(inferred, useVdw=True)
    def bond_pairs(mol):
        return {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in mol.GetBonds()
        }

    if bond_pairs(inferred) != bond_pairs(molecule):
        errors.append("connectivity_changed")
    geometry = Chem.Mol(molecule)
    conf = Chem.Conformer(len(atoms))
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, tuple(float(v) for v in p))
    geometry.AddConformer(conf)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    Chem.AssignStereochemistryFrom3D(geometry, replaceExistingTags=True)
    for original, observed in zip(molecule.GetAtoms(), geometry.GetAtoms()):
        if original.HasProp("_CIPCode") and (not observed.HasProp("_CIPCode") or
                original.GetProp("_CIPCode") != observed.GetProp("_CIPCode")):
            errors.append(f"stereochemistry_changed:{original.GetIdx()}")
    for original, observed in zip(molecule.GetBonds(), geometry.GetBonds()):
        if original.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ) and original.GetStereo() != observed.GetStereo():
            errors.append(f"bond_stereochemistry_changed:{original.GetIdx()}")
    return {"passed": not errors, "errors": errors, "stereochemistry_checked": True,
            "scope": "distance-based graph screen; bond orders are not inferred from DFT density"}
