"""Check whether a candidate meets the requirements for selection."""
from core.paths import to_float

REQUIRED_CRITERIA = ("redox_window", "material_capacity", "state_stability", "charge_transfer",
                     "practical_attachment", "route_thermodynamics", "motif_association",
                     "intersite_coupling", "occupancy_sensitivity", "polymer_accessibility")


def contiguous_paths(events, resting_charge, *, direction, window):
    """Only count sequential accessible events from the as-prepared charge."""
    if direction not in (-1, 1) or len(window) != 2 or window[0] >= window[1]:
        raise ValueError("direction +/-1 and an ordered potential window required")
    lookup = {}
    for event in events:
        key = (int(event["from_charge"]), int(event["to_charge"]))
        if key in lookup:
            raise ValueError("duplicate redox event")
        lookup[key] = event
    path, charge = [], int(resting_charge)
    while (charge, charge + direction) in lookup:
        event = lookup[(charge, charge + direction)]
        interval = event.get("qualified_interval_V")
        if event.get("status") != "pass" or interval is None:
            break
        low, high = map(to_float, interval)
        if low > high or low < window[0] or high > window[1]:
            break
        path.append(event)
        charge += direction
    return path


def evaluate_criterion(evidence, bounds):
    if not evidence or evidence.get("qualified") is not True:
        return "pending"
    if evidence.get("decision") in ("pass", "fail", "pending"):
        return evidence["decision"]
    if bounds is None or evidence.get("interval") is None:
        return "pending"
    low, high = map(to_float, evidence["interval"])
    lower, upper = bounds
    if low > high or (lower is not None and upper is not None and lower > upper):
        raise ValueError("invalid evidence interval or bounds")
    if (lower is None or low >= lower) and (upper is None or high <= upper):
        return "pass"
    if (lower is not None and high < lower) or (upper is not None and low > upper):
        return "fail"
    return "pending"


def capacity_ceiling_decision(ceiling_mAh_g, minimum_mAh_g):
    """Exclude only when the declared polymer's upper bound is below the floor."""
    if ceiling_mAh_g is None or minimum_mAh_g is None:
        return "pending"
    ceiling, minimum = map(to_float, (ceiling_mAh_g, minimum_mAh_g))
    if ceiling < 0 or minimum < 0:
        raise ValueError("capacity ceiling and minimum must be nonnegative")
    return "fail" if ceiling < minimum else "pending"


def _candidate_criterion(candidate, criterion, item, bounds):
    # Attachment evidence belongs to the synthesis route. Other evidence belongs to the operating conditions.
    context = ("route_context" if criterion in ("practical_attachment", "route_thermodynamics")
               else "operating_context")
    if item and (item.get("identity") != candidate["identity"] or item.get(context) != candidate[context]):
        raise ValueError(f"mismatched identity/context for {criterion}")
    if criterion == "material_capacity" and item and "capacity_ceiling_mAh_g" in item:
        if item.get("qualified") is not True:
            return "pending"
        ceiling_decision = capacity_ceiling_decision(
            item["capacity_ceiling_mAh_g"], bounds[0] if bounds else None)
        if ceiling_decision == "fail" or item.get("interval") is None:
            return ceiling_decision
    return evaluate_criterion(item, bounds)


def _candidate_status(decisions):
    if "fail" in decisions:
        return "excluded"
    if decisions and all(decision == "pass" for decision in decisions):
        return "eligible"
    return "pending"


def scorecard_row(candidate, evidence, thresholds, required=REQUIRED_CRITERIA):
    if not all(candidate.get(key) for key in ("identity", "operating_context", "route_context")):
        raise ValueError("exact identity and operating/route contexts required")
    decisions = {
        criterion: _candidate_criterion(candidate, criterion, evidence.get(criterion), thresholds.get(criterion))
        for criterion in required
    }
    return {**candidate, "status": _candidate_status(decisions.values()), "criteria": decisions}
