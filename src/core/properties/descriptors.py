"""Calculate RMSD, theoretical capacity, and a molecular solubility proxy."""
import numpy as np
from core.paths import FARADAY, to_float


def rmsd(reference, target):
    """Align two geometries with the Kabsch method and report RMSD in the input units.

    The caller supplies coordinates with matching atom order.
    """
    a, b = np.asarray(reference, float), np.asarray(target, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.isfinite([a, b]).all():
        raise ValueError("matching finite N x 3 coordinates required")
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b)**2, axis=1))))


def capacity(molecular_weight_g_mol, electrons):
    mass, electrons = to_float(molecular_weight_g_mol), to_float(electrons)
    if mass <= 0 or electrons < 0:
        raise ValueError("positive mass and nonnegative assumed electron count required")
    return electrons * FARADAY / (3.6 * mass)


def solubility_proxy(smiles):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES")
    return {"mol_weight_g_mol": Descriptors.MolWt(mol), "cLogP": Crippen.MolLogP(mol),
            "tpsa_A2": rdMolDescriptors.CalcTPSA(mol), "formal_charge": Chem.GetFormalCharge(mol),
            "scope": "descriptors only; no calibrated MeCN solubility or polymer swelling prediction"}
