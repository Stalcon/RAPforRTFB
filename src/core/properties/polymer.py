"""Calculate grafting energies and compare finite polymer models."""
from collections import Counter
import math

from core.paths import EV_KJ, KB_EV, to_float
from .context import energy, require_compatible
from .descriptors import capacity


def reaction_delta_g(components, stoichiometry, *, target_standard_state="1M"):
    """Use positive coefficients for products and negative ones for reactants.

    Include hydrogen in the elemental counts.
    """
    if not stoichiometry or set(stoichiometry) - set(components):
        raise ValueError("every reaction component must be supplied")
    records = [components[k] for k in stoichiometry]
    require_compatible(records, thermodynamic=True)
    balance = Counter()
    charge = 0
    for name, coefficient in stoichiometry.items():
        if not isinstance(coefficient, int) or coefficient == 0:
            raise ValueError("nonzero integer stoichiometric coefficients required")
        record = components[name]
        if not record.get("composition"):
            raise ValueError("explicit elemental composition required")
        for element, count in record["composition"].items():
            balance[element] += coefficient * count
        charge += coefficient * record["charge"]
    if charge or any(balance.values()):
        raise ValueError(f"reaction is not atom/charge balanced: {dict(balance)}, charge={charge}")
    value = sum(c * energy(components[k], free=True) for k, c in stoichiometry.items())
    protocol = records[0]["thermal_protocol"]
    initial = protocol["standard_state"]
    correction = 0.0
    if initial != target_standard_state:
        if (initial, target_standard_state) != ("1atm", "1M"):
            raise ValueError("only explicit 1atm -> 1M conversion is implemented")
        t = protocol["temperature_K"]
        correction = sum(stoichiometry.values()) * KB_EV * t * math.log(8.314462618 * t * 1000 / 101325)
    return {"delta_G_kJ_mol": (value + correction) * EV_KJ,
            "standard_state_correction_kJ_mol": correction * EV_KJ,
            "standard_state": target_standard_state, "stoichiometry": stoichiometry,
            "scope": "defined reaction thermodynamics; not a barrier, conversion, or synthesis gate"}


def occupancy_cycle(empty, single_i, single_j, double, *, free=True):
    records = [empty, single_i, single_j, double]
    require_compatible(records, thermodynamic=free)
    for field in ("backbone_id", "backbone_configuration"):
        if not empty.get(field) or any(r.get(field) != empty[field] for r in records):
            raise ValueError(f"mismatched or absent {field}")
    occupancies = [set(r["occupied_sites"]) for r in records]
    if occupancies[0] or len(occupancies[1]) != 1 or len(occupancies[2]) != 1 or occupancies[1] == occupancies[2] or occupancies[3] != occupancies[1] | occupancies[2]:
        raise ValueError("require matched empty, i, j, and ij occupancy graphs")
    # The empty and doubly occupied chains must balance the two singly occupied chains.
    for field in ("charge",):
        if empty[field] + double[field] != single_i[field] + single_j[field]:
            raise ValueError("unbalanced occupancy charge")
    elements = set().union(*(r["composition"] for r in records))
    if any(empty["composition"].get(e, 0) + double["composition"].get(e, 0) !=
           single_i["composition"].get(e, 0) + single_j["composition"].get(e, 0) for e in elements):
        raise ValueError("unbalanced occupancy composition")
    value = energy(double, free=free) + energy(empty, free=free) - energy(single_i, free=free) - energy(single_j, free=free)
    return {"quantity": "delta_delta_G" if free else "delta_delta_E", "value_kJ_mol": value * EV_KJ,
            "scope": "conditional whole-chain nonadditivity; not isolated motif binding or coupling"}


def frozen_motif_interaction(pair, fragment_a, fragment_b):
    require_compatible([pair, fragment_a, fragment_b])
    if pair["charge"] != fragment_a["charge"] + fragment_b["charge"]:
        raise ValueError("fragment charges do not sum to the pair")
    for field in ("parent_frame", "cap_definition"):
        if not pair.get(field) or any(r.get(field) != pair[field] for r in (fragment_a, fragment_b)):
            raise ValueError(f"fragments must share {field}")
    if not all(r.get("frozen_coordinates_verified") is True for r in (pair, fragment_a, fragment_b)):
        raise ValueError("frozen fragment coordinate correspondence must be verified")
    elements = set().union(*(r["composition"] for r in (pair, fragment_a, fragment_b)))
    if any(pair["composition"].get(e, 0) != fragment_a["composition"].get(e, 0) + fragment_b["composition"].get(e, 0) for e in elements):
        raise ValueError("fragment composition does not sum to the pair")
    return {"interaction_E_kJ_mol": (energy(pair) - energy(fragment_a) - energy(fragment_b)) * EV_KJ,
            "scope": "electronic interaction of declared caps; no thermal or counterpoise correction"}


def finite_ensemble(energies_eV, *, temperature=298.15, duplicate_groups=None):
    values = [to_float(e) for e in energies_eV]
    if not values or temperature <= 0:
        raise ValueError("nonempty finite energies and positive temperature required")
    if duplicate_groups is None or len(duplicate_groups) != len(values):
        raise ValueError("explicit numerical duplicate-group assignments required")
    unique = {}
    for group, value in zip(duplicate_groups, values):
        unique[group] = min(unique.get(group, value), value)
    minimum = min(unique.values())
    terms = [math.exp(-(e - minimum) / (KB_EV * temperature)) for e in unique.values()]
    return {"finite_set_G_eV": minimum - KB_EV * temperature * math.log(sum(terms)),
            "unique_minima": len(unique), "weights": [w / sum(terms) for w in terms],
            "converged_ensemble": False}


def polymer_capacity(fragment_mass_g_mol, graft_count, electrons_per_graft, *, extra_mass_g_mol=0):
    mass = to_float(fragment_mass_g_mol) + to_float(extra_mass_g_mol)
    if graft_count < 0 or int(graft_count) != graft_count or extra_mass_g_mol < 0:
        raise ValueError("nonnegative integer graft count and additional mass required")
    return {"capacity_ceiling_mAh_g": capacity(mass, graft_count * electrons_per_graft),
            "mass_g_mol": mass, "scope": "declared composition and full assumed electron utilization"}
