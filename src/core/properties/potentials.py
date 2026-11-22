"""Calculate redox potentials, wave spacing, and voltage differences."""
import math

from core.paths import EV_KJ, KB_EV, free_energy, to_float
from .context import require_compatible


def redox_potential(oxidized, reduced, fc_oxidized, fc_reduced):
    records = [oxidized, reduced, fc_oxidized, fc_reduced]
    require_compatible(records, thermodynamic=True)
    for left, right in ((oxidized, reduced), (fc_oxidized, fc_reduced)):
        if not left.get("identity") or left["identity"] != right.get("identity"):
            raise ValueError("redox states must retain the same nuclear graph identity")
        if left["charge"] - right["charge"] != 1:
            raise ValueError("this function requires a one-electron couple")
    absolute = free_energy(oxidized) - free_energy(reduced)
    reference = free_energy(fc_oxidized) - free_energy(fc_reduced)
    return {"E_absolute_V": absolute, "E_vs_Fc_V": absolute - reference,
            "reference_absolute_V": reference, "identity": oxidized["identity"],
            "reference_identity": fc_oxidized["identity"],
            "method": oxidized["method"], "thermal_protocol": oxidized["thermal_protocol"],
            "experimental_accuracy_validated": False}


def disproportionation(first_reduction_V, second_reduction_V, temperature=298.15):
    first, second, temperature = map(to_float, (first_reduction_V, second_reduction_V, temperature))
    if temperature <= 0:
        raise ValueError("positive temperature required")
    delta = first - second
    return {"step_spacing_V": delta, "disproportionation_dG_kJ_mol": delta * EV_KJ,
            "log10_K_disproportionation": -delta / (KB_EV * temperature * math.log(10)),
            "scope": "2 intermediate -> oxidized + reduced; same solvent and reference required"}


def matched_shift(reference, substituted):
    for field in ("scaffold", "attachment", "redox_step", "method", "reference", "thermal_protocol"):
        if not reference.get(field) or reference[field] != substituted.get(field):
            raise ValueError(f"matched comparison requires identical {field}")
    return {"potential_shift_V": to_float(substituted["potential_V"]) - to_float(reference["potential_V"]),
            "scope": "declared matched structures; transferability and additivity are untested"}


def proposed_cell(anolyte, catholyte):
    for field in ("method", "reference", "thermal_protocol", "operating_context", "potential_definition"):
        if not anolyte.get(field) or anolyte[field] != catholyte.get(field):
            raise ValueError(f"cell proposal requires matched {field}")
    return {"nominal_voltage_V": to_float(catholyte["potential_V"]) - to_float(anolyte["potential_V"]),
            "scope": "half-cell potential difference only; no demonstrated device voltage, efficiency, or cycling"}
