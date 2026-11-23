"""Import exact polymer graphs and construct audited frozen capped fragments."""
from collections import Counter
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read
from core.paths import canonical_smiles, safe_name, write_xyz, atomic_write_json
from core.protocol import file_hash


def audit_backbone(smiles, backbone_indices):
    """Backbone indices refer to the canonical SMILES without added hydrogens."""
    from rdkit import Chem
    from .build import check_stereo
    molecule = Chem.MolFromSmiles(canonical_smiles(smiles))
    indices = list(backbone_indices)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError('a simple ordered open-chain backbone is required')
    if len(Chem.GetMolFrags(molecule)) != 1 or check_stereo(molecule):
        raise ValueError('connected exact stereochemical graph required')
    expected_edges = {tuple(sorted(pair)) for pair in zip(indices, indices[1:])}
    actual_edges = set()
    for index in indices:
        atom = molecule.GetAtomWithIdx(index)
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            raise ValueError('backbone must consist of nonaromatic carbon atoms')
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            if j in indices:
                bond = molecule.GetBondBetweenAtoms(index, j)
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    raise ValueError('backbone bonds must be single')
                actual_edges.add(tuple(sorted((index, j))))
    if actual_edges != expected_edges:
        raise ValueError('backbone is not the declared open path')
    return {'passed': True, 'canonical_smiles': Chem.MolToSmiles(molecule),
            'backbone_indices': indices}


def build_occupancy_case(case, *, directory, conformers=2):
    """Build a curated exact graph, rather than infer a graft distribution."""
    from rdkit import Chem
    from .build import conformer_ensemble, to_xyz
    from core.paths import atomic_write_text
    required = ('id', 'smiles', 'backbone_indices', 'backbone_id', 'backbone_configuration',
                'reactive_sites', 'occupied_sites', 'end_caps', 'ion_convention', 'charge', 'mult')
    if any(k not in case for k in required):
        raise ValueError(f'case requires {required}')
    if not set(case['occupied_sites']) <= set(case['reactive_sites']):
        raise ValueError('occupied sites must be declared reactive sites')
    audit = audit_backbone(case['smiles'], case['backbone_indices'])
    molecule = Chem.MolFromSmiles(audit['canonical_smiles'])
    if Chem.GetFormalCharge(molecule) != case['charge']:
        raise ValueError('resting graph charge differs from case charge')
    directory = Path(directory) / safe_name(case['id'])
    directory.mkdir(parents=True, exist_ok=True)
    mol, ids, ff, energies = conformer_ensemble(molecule, top_k=conformers)
    from core.modeling.uma import candidate_mults
    candidate_mults(sum(a.GetAtomicNum() for a in mol.GetAtoms()) - case['charge'], case['mult'])
    for i, conf_id in enumerate(ids):
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(mol.GetConformer(conf_id), assignId=True)
        atomic_write_text(directory / f'conf_{i:02}.xyz', to_xyz(single))
    result = {**case, 'smiles': audit['canonical_smiles'], 'backbone_audit': audit,
              'composition': dict(Counter(a.GetSymbol() for a in mol.GetAtoms())),
              'forcefield': ff, 'seed_energies_kcal_mol': energies,
              'site_mapping_independently_verified': False,
              'scope': 'curated graph; loading probability and site assignment are not inferred'}
    atomic_write_json(directory / 'case.json', result)
    return result


def make_frozen_fragments(parent_xyz, fragments, *, directory, parent_frame, cap_length_A=1.09):
    """Each fragment declares retained atom indices, cut bonds, and charge/spin.

    A cut is (retained carbon index, removed atom index). Its replacement H is
    placed along that original bond. Original retained coordinates never move.
    """
    if len(fragments) != 2 or cap_length_A <= 0:
        raise ValueError('exactly two fragments and positive C-H cap length required')
    parent_xyz = Path(parent_xyz).resolve()
    parent = read(parent_xyz)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sets = [set(f['indices']) for f in fragments]
    if sets[0] & sets[1]:
        raise ValueError('retained fragment atom sets must not overlap')
    outputs, geometries = [], []
    from core.modeling.uma import candidate_mults
    for label, fragment in zip(('A', 'B'), fragments):
        indices = fragment['indices']
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(parent) for i in indices):
            raise ValueError('invalid retained atom indices')
        atoms = parent[indices]
        for kept, removed in fragment['cuts']:
            if kept not in indices or removed in indices or not 0 <= removed < len(parent) or parent[kept].symbol != 'C':
                raise ValueError('declare a retained carbon and removed bond partner')
            delta = parent.positions[removed] - parent.positions[kept]
            length = np.linalg.norm(delta)
            if not np.isfinite(length) or length < 0.1:
                raise ValueError('invalid cut bond geometry')
            atoms += Atoms('H', positions=[parent.positions[kept] + delta * cap_length_A / length])
        candidate_mults(int(sum(atoms.numbers)) - fragment['charge'], fragment['mult'])
        path = directory / f'{label}.xyz'
        write_xyz(atoms, path)
        geometries.append(atoms)
        outputs.append({'label': label, 'xyz': str(path.resolve()), 'charge': fragment['charge'],
                        'mult': fragment['mult'], 'retained_parent_indices': indices,
                        'cuts': fragment['cuts'], 'composition': dict(Counter(atoms.get_chemical_symbols()))})
    pair = geometries[0] + geometries[1]
    write_xyz(pair, directory / 'AB.xyz')
    result = {'parent_frame': parent_frame, 'parent_xyz_sha256': file_hash(parent_xyz),
              'fragments': outputs, 'pair_xyz': str((directory / 'AB.xyz').resolve()),
              'pair_charge': sum(f['charge'] for f in fragments),
              'pair_multiplicity': None, 'frozen_coordinates_verified': True,
              'cap_definition': {'element': 'H', 'length_A': cap_length_A, 'direction': 'original cut bond'},
              'scope': 'set pair spin explicitly; fragment selection chemistry requires review'}
    atomic_write_json(directory / 'fragments.json', result)
    return result


def core_contact(atoms, indices_a, indices_b):
    """Core centroid separation and acute least-squares plane angle."""
    a, b = atoms.positions[indices_a], atoms.positions[indices_b]
    if len(a) < 3 or len(b) < 3:
        raise ValueError('at least three atoms per core required')
    normals = []
    for points in (a, b):
        _, singular, axes = np.linalg.svd(points - points.mean(axis=0))
        if singular[1] < 1e-8:
            raise ValueError('core points are collinear')
        normals.append(axes[-1])
    return {'centroid_distance_A': float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))),
            'plane_angle_degrees': float(np.degrees(np.arccos(np.clip(abs(normals[0] @ normals[1]), 0, 1))))}
