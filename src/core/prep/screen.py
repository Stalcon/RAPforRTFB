from rdkit import Chem
from rdkit.Chem import AllChem
import build

FAMILIES = {
    "quinone": {
        "smarts": ["[#6;R]1(=O)[#6;R]=,:[#6;R][#6;R](=O)[#6;R]=,:[#6;R]1",
                   "[#6;R]1(=O)[#6;R](=O)[#6;R]=,:[#6;R][#6;R]=,:[#6;R]1"],
        "n_e": 2, "multi": True, "direction": -1,
    },
    "bis-imide": {
        "smarts": ["[#7]1[#6](=O)c2cc3[#6](=O)[#7][#6](=O)c3cc2[#6]1=O",
                   "[#7]1[#6](=O)c2ccc3c4c(ccc(c24)[#6]1=O)[#6](=O)[#7][#6]3=O"],
        "n_e": 2, "multi": True, "direction": -1,
    },
    "mono-imide": {
        "smarts": ["O=[#6;R](c)[#7;R][#6;R](=O)c"],
        "n_e": 1, "multi": False, "direction": -1,
    },
    "viologen": {
        "smarts": ["[n+;R]1ccc(-c2cc[n+]cc2)cc1"],
        "n_e": 2, "multi": True, "direction": -1,
    },
    "pyridinium": {
        "smarts": ["[n+;R;H0]"],
        "n_e": 1, "multi": False, "direction": -1,
    },
    "phenothiazine": {
        "smarts": ["c1cccc2c(c1)[#16]c1ccccc1[#7]2"],
        "n_e": 1, "multi": False, "direction": +1,
    },
    "nitroxide": {
        "smarts": ["[#7;X3]([#6])([#6])[#8;X1;v1]"],
        "n_e": 1, "multi": False, "direction": +1,
    },
    "other": {
        "smarts": None,
        "n_e": 1, "multi": False, "direction": -1,
    },
}
FAMILY_PAT = {}
for family, data in FAMILIES.items():
    compiled_patterns = []
    for family_smarts in (data.get("smarts") or []):
        compiled_pattern = Chem.MolFromSmarts(family_smarts)
        compiled_patterns.append(compiled_pattern)
    FAMILY_PAT[family] = compiled_patterns
BENZYL_TETHER = "[CH2](c1ccc(C)cc1)"
GRAFTS = {
    "O-benzyl ether":
        "[OX2H1:9]-[#6;!$([CX3]=O):2]>>" + BENZYL_TETHER + "[O:9]-[#6:2]",
    "N-benzyl amine":
        "[NX3;!a;H1,H2;!$(N[#6]=O);!$(N[#16]=O);!$(N=*):9]>>" + BENZYL_TETHER + "[#7:9]",
    "N-benzyl imine":
        "[#7X3;H1:9]([#6:2]=[O:3])[#6:4]=[O:5]>>" + BENZYL_TETHER + "[#7:9]([#6:2]=[O:3])[#6:4]=[O:5]",
    "N-benzyl pyridinium":
        "[nX2;H0;+0:9]>>" + BENZYL_TETHER + "[n+:9]",
    "S-benzyl thioether":
        "[SX2H1:9]>>" + BENZYL_TETHER + "[S:9]",
    "benzyl ester":
        "[CX3:2](=[O:3])[OX2H1:9]>>[CX3:2](=[O:3])[O:9]" + BENZYL_TETHER,
}
GRAFTS_PAT = {}
for chemistry, smarts in GRAFTS.items():
    graft_smarts = smarts.split(">>")[0]
    compiled_graft = Chem.MolFromSmarts(graft_smarts)
    GRAFTS_PAT[chemistry] = compiled_graft
GRAFTS_RXN = {}
for chemistry, smarts in GRAFTS.items():
    compiled_rxn = AllChem.ReactionFromSmarts(smarts)
    GRAFTS_RXN[chemistry] = compiled_rxn
ATTACH_MAP = 9


class RedoxMonomer:
    def __init__(self, frag_smi, functionalized_smi, frag_families, families, attachment, name=None):
        self.name = name
        self.fragment_smiles = frag_smi
        self.smiles = functionalized_smi
        self.attachment = attachment
        self.fragment_families = frag_families
        self.families = families
        self.charge_states = []
        self.spin_map = {}
        self.props = {}
    def to_mol(self):
        return Chem.MolFromSmiles(self.smiles)
    def initialize_states(self, charge_states, spin_map):
        self.charge_states = charge_states
        self.spin_map = spin_map


def identify(mol):
    matched = []
    for family in FAMILIES.items():
        if family == "other":
            if matched == []:
                matched.append(family)
                continue
        smarts_list = FAMILY_PAT.get(family)
        for smarts in smarts_list:
            group = Chem.MolFromSmarts(smarts)
            if group and mol.HasSubstructMatch(group):
                matched.append(family)  
    if "bis-imide" in matched:
        matched.remove("mono-imide")
    if "viologen" in matched:
        matched.remove("pyridinium")
    return matched

def attachment_points(mol):
    points = []
    for chemistry, pattern in GRAFTS_PAT.items():
        for q in range(pattern.GetNumAtoms()):
            if pattern.GetAtomWithIdx(q).GetAtomMapNum() == ATTACH_MAP:
                attach_q = q
            else:
                continue
        for match in mol.GetSubstructMatches(pattern):
            point = (match[attach_q], chemistry)
            points.append(point)
    return points


def graft_products(mol_smi):
    precursor = Chem.MolFromSmiles(mol_smi)
    if precursor is None:
        raise ValueError(f"bad SMILES")

    points = attachment_points(precursor)

    seen = []
    products = []

    for point in points:
        chemistry = point[1]
        product_sets = GRAFTS_RXN[chemistry].RunReactants((precursor,))
        for product_set in product_sets:
            product = product_set[0]
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            smi = Chem.MolToSmiles(product)
            if smi not in seen:
                seen.append(smi)
                products.append((smi,chemistry))
    return products

def build_states(mol_smi, families):
    mol = Chem.MolFromSmiles(mol_smi)
    q0 = Chem.GetFormalCharge(mol)
    unpaired0 = mol.GetNumRadicalElectrons()

    charge_states = [q0]
    spin_map = {q0:unpaired0+1}

    for family in families:
        if family == "other":
            continue
        family_e = FAMILIES[family]["n_e"]
        direction = FAMILIES[family]["direction"]

        for n in range(1, family_e+1):
            q = q0 + direction*n
            unpaired = (unpaired0 + n) % 2
            if q not in charge_states:
                charge_states.append(q)
                spin_map[q] = unpaired + 1
    return charge_states, spin_map

## Add stereo guarding later
def screen_mol(mol_smi, name=None):
    precursor = Chem.MolFromSmiles(mol_smi)
    frag_families = identify(precursor)

    monomoers = []
    
    for smi, chemistry in graft_products(mol_smi):
        grafted = Chem.MolFromSmiles(smi)
        families = identify(grafted)
        charge_states, spin_map = build_states(smi, families)
        redox_monomoer = RedoxMonomer(
            frag_smi=mol_smi, 
            functionalized_smi=grafted, 
            frag_families=frag_families, 
            families=families, 
            attachment=chemistry, 
            name=name)
        redox_monomoer.initialize_states(charge_states, spin_map)
        monomoers.append(redox_monomoer)

    return monomoers

