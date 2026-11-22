"""Calculate reorganization energies from four-point, solvent, and continuum models."""
from core.paths import to_float
from .context import require_compatible

COULOMB_EV_A = 14.3996454784255


def nelsen_4pt(e_ox_at_ox, e_ox_at_red, e_red_at_red, e_red_at_ox, *, surface):
    if surface not in ("gas", "equilibrium_solution"):
        raise ValueError("declare gas or equilibrium_solution surface")
    oo, or_, rr, ro = map(to_float, (e_ox_at_ox, e_ox_at_red, e_red_at_red, e_red_at_ox))
    ox, red = or_ - oo, ro - rr
    return {"oxidized_distortion_eV": ox, "reduced_distortion_eV": red,
            "four_point_sum_eV": ox + red, "surface": surface,
            "minimum_consistent": min(ox, red) >= -1e-6,
            "scope": "electronic distortion costs; not a measured transfer rate"}


def calculate_pair(ox_xyz, red_xyz, q_ox, mult_ox, mult_red, *, directory, settings=None, **resources):
    from pathlib import Path
    from ase.io import read
    from core.modeling.sp import single_point
    if read(ox_xyz).get_chemical_symbols() != read(red_xyz).get_chemical_symbols():
        raise ValueError("paired XYZ files must have identical atom order")
    points = {}
    for label, xyz, charge, mult in (("oo", ox_xyz, q_ox, mult_ox),
                                    ("or", red_xyz, q_ox, mult_ox),
                                    ("rr", red_xyz, q_ox - 1, mult_red),
                                    ("ro", ox_xyz, q_ox - 1, mult_red)):
        points[label] = single_point(xyz, charge, mult, settings=settings,
                                     out=Path(directory) / f"{label}.json", **resources)
    require_compatible(list(points.values()))
    surface = "equilibrium_solution" if points["oo"]["method"].get("solvent") else "gas"
    return {**nelsen_4pt(*(points[k]["energy_eV"] for k in ("oo", "or", "rr", "ro")), surface=surface),
            "points": points}


def directional_relaxation(initial_eq, final_at_initial_eq, final_nonequilibrium, final_eq):
    a, b, c, d = map(to_float, (initial_eq, final_at_initial_eq, final_nonequilibrium, final_eq))
    return {"total_solution_relaxation_eV": c - d,
            "equilibrium_geometry_component_eV": b - d,
            "frozen_slow_increment_eV": c - b, "vertical_charge_change_eV": c - a}


def summarize_relaxations(forward, reverse, tolerance=0.05):
    a, b = (to_float(r["total_solution_relaxation_eV"]) for r in (forward, reverse))
    if not 0 < tolerance < 1:
        raise ValueError("relative tolerance must be between zero and one")
    average = (a + b) / 2
    asymmetry = abs(a - b) / average if average > 0 else None
    return {"forward_eV": a, "reverse_eV": b, "relative_asymmetry": asymmetry,
            "symmetric_approximation_passes": min(a, b) >= 0 and asymmetry is not None and asymmetry <= tolerance,
            "scope": "requires matched nonequilibrium inputs; do not add gas or outer-sphere terms again"}


def _pekar(eps_s, eps_op):
    eps_s, eps_op = to_float(eps_s), to_float(eps_op)
    if not 1 <= eps_op <= eps_s:
        raise ValueError("require 1 <= optical <= static dielectric")
    return 1 / eps_op - 1 / eps_s


def lambda_o(radius_A, eps_s, eps_op, z=1):
    """Estimate the Born solvent response for a single spherical cavity."""
    radius_A, z = to_float(radius_A), to_float(z)
    if radius_A <= 0:
        raise ValueError("positive cavity radius required")
    return COULOMB_EV_A * z**2 / (2 * radius_A) * _pekar(eps_s, eps_op)


def lambda_het(radius_A, electrode_distance_A, eps_s, eps_op, z=1):
    """Estimate the solvent response next to a planar electrode.

    Measure the distance from the ion center to the electrode.
    """
    a, d, z = map(to_float, (radius_A, electrode_distance_A, z))
    if a <= 0 or d < a:
        raise ValueError("sphere must not intersect the electrode")
    return COULOMB_EV_A * z**2 * (1 / (2 * a) - 1 / (4 * d)) * _pekar(eps_s, eps_op)


def lambda_se_contact(radius_A, eps_s, eps_op, z=1, *, partner_radius_A=None, distance_A=None):
    """Estimate the solvent response between two nonoverlapping spheres.

    By default, use equal radii and place the spheres in contact.
    """
    a = to_float(radius_A)
    b = a if partner_radius_A is None else to_float(partner_radius_A)
    d = a + b if distance_A is None else to_float(distance_A)
    z = to_float(z)
    if min(a, b) <= 0 or d < a + b:
        raise ValueError("positive, nonoverlapping cavities required")
    return COULOMB_EV_A * z**2 * (1 / (2 * a) + 1 / (2 * b) - 1 / d) * _pekar(eps_s, eps_op)
