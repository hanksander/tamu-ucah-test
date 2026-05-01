import math
import numpy as np

# Assumes this script lives in the same repo and can import your inlet module.
# If needed, adjust the import path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "combined_cycle_liquid_ram_scram"))

import pyc_run

inlet2 = pyc_run._inlet2
R_AIR = 287.05
DEFAULT_OFF_DESIGN_MACH = 5
DEFAULT_OFF_DESIGN_ALTITUDE_M = 18_000.0
DEFAULT_OFF_DESIGN_ALPHA_DEG = 6


def gamma_mass_function(gamma, R):
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return math.sqrt(gamma / R) * (2.0 / gp1) ** (gp1 / (2.0 * gm1))


def evaluate_inlet_capacity_and_startability(
        design,
        M0,
        altitude_m,
        alpha_deg,
        margin_unstart=0.85,
):
    """
    First-pass inlet capacity / startability check for a fixed inlet geometry.

    Returns a dict with:
      mdot_swallowed_kgs
      mdot_kantrowitz_limit_kgs
      mdot_throat_limit_kgs
      mdot_passed_kgs
      spill_mdot_kgs
      spill_fraction
      state: 'started', 'spillage', or 'unstart'
      reason
    """

    # Re-run the frozen-geometry shock chain at this condition.
    up = inlet2._upstream_shock_chain(design, M0, altitude_m, alpha_deg)
    if not up.get("ok", False):
        return {
            "ok": False,
            "state": "unstart",
            "reason": f"Upstream shock chain failed: {up.get('reason', 'unknown')}",
        }

    # Swallowed streamtube from your new geometry model.
    swallowed = inlet2.compute_swallowed_streamtube(
        design, M0=M0, altitude_m=altitude_m, alpha_deg=alpha_deg
    )
    if not swallowed.get("ok", False):
        return {
            "ok": False,
            "state": "unstart",
            "reason": f"Swallowed streamtube failed: {swallowed.get('reason', 'unknown')}",
        }

    mdot_swallowed_raw = float(swallowed.get("mdot_swallowed_kgs", 0.0))
    A_swallowed_raw = float(swallowed.get("A_swallowed_m2", 0.0))
    A_swallowed = float(
        pyc_run._corrected_capture_area(
            design["A_capture_required_m2"], A_swallowed_raw
        )
    )
    fs = inlet2.freestream_state(M0, altitude_m)
    mdot_swallowed = float(fs["rho0"] * fs["V0"] * A_swallowed)

    # Fixed throat geometry.
    A_throat = float(design["throat_area_actual_m2"])

    # Pre-throat / post-ramp2 state for Kantrowitz-like limit.
    M2 = float(up["M2"])
    T2 = float(up["T2_s"])

    # Static pressure after forebody+ramp1+ramp2.
    p2 = float(up["p0"] * up["p_fore_ratio"] * up["p21"] * up["p32"])
    gamma2 = float(inlet2.gamma_air(T2))
    a2 = math.sqrt(gamma2 * R_AIR * T2)
    V2 = M2 * a2
    theta2 = float(design["theta2_deg"])
    Vx2 = V2 * math.cos(math.radians(theta2))
    rho2 = p2 / (R_AIR * T2)

    # Kantrowitz contraction ratio limit at the pre-throat supersonic state.
    CR_k_raw = float(inlet2.kantrowitz_contraction_ratio(M2))
    CR_k = CR_k_raw if CR_k_raw > 1.0 else 1.0 / CR_k_raw

    # Maximum startable capture area consistent with current throat.
    A_capture_max_k = CR_k * A_throat
    mdot_kantrowitz_limit = rho2 * Vx2 * A_capture_max_k

    # Throat choked-flow limit.
    # Use post-cowl total pressure with freestream total temperature as a first-pass
    # estimate, consistent with the rest of your current inlet model.
    Pt_throat_in = float(up["Pt_after_cowl"])
    Tt_throat_in = float(up["Tt0"])

    gamma_throat = float(inlet2.gamma_air(max(T2, 200.0)))
    Gamma_mass = gamma_mass_function(gamma_throat, R_AIR)
    mdot_throat_limit = Pt_throat_in * A_throat * Gamma_mass / math.sqrt(Tt_throat_in)

    mdot_passed = min(mdot_swallowed, mdot_kantrowitz_limit, mdot_throat_limit)
    spill_mdot = max(0.0, mdot_swallowed - mdot_passed)
    spill_fraction = spill_mdot / mdot_swallowed if mdot_swallowed > 1.0e-12 else 0.0

    limiting_mode = min(
        (
            ("swallowed", mdot_swallowed),
            ("kantrowitz", mdot_kantrowitz_limit),
            ("throat", mdot_throat_limit),
        ),
        key=lambda kv: kv[1],
    )[0]

    # Decision logic:
    # - If the startability limit is violated badly, flag unstart.
    # - If swallowed > capacity but not catastrophically, flag spillage.
    # - Otherwise started.
    #
    # The margin_unstart lets you be conservative near the limit.
    if mdot_swallowed > mdot_kantrowitz_limit / margin_unstart:
        state = "unstart"
        reason = "Available swallowed flow exceeds startability/Kantrowitz capacity."
    elif spill_mdot > 0.0:
        state = "spillage"
        reason = "Available swallowed flow exceeds passed flow capacity; excess must spill."
    else:
        state = "started"
        reason = "Swallowed flow is within both startability and throat capacity."

    return {
        "ok": True,
        "state": state,
        "reason": reason,
        "limiting_mode": limiting_mode,
        "M0": float(M0),
        "altitude_m": float(altitude_m),
        "alpha_deg": float(alpha_deg),
        "A_swallowed_raw_m2": A_swallowed_raw,
        "A_swallowed_m2": A_swallowed,
        "A_throat_m2": A_throat,
        "CR_k": float(CR_k),
        "A_capture_max_k_m2": float(A_capture_max_k),
        "mdot_swallowed_raw_kgs": float(mdot_swallowed_raw),
        "mdot_swallowed_kgs": float(mdot_swallowed),
        "mdot_kantrowitz_limit_kgs": float(mdot_kantrowitz_limit),
        "mdot_throat_limit_kgs": float(mdot_throat_limit),
        "mdot_passed_kgs": float(mdot_passed),
        "spill_mdot_kgs": float(spill_mdot),
        "spill_fraction": float(spill_fraction),
        "swallowed_region_tag": str(swallowed.get("region_tag", "")),
        "x_capture_m": float(swallowed.get("x_capture_m", np.nan)),
        "upper_capture_xy": np.asarray(swallowed.get("upper_capture_xy", [np.nan, np.nan]), dtype=float),
        "lower_capture_xy": np.asarray(swallowed.get("lower_capture_xy", [np.nan, np.nan]), dtype=float),
    }


def analyze_pyc_config_fixed_geometry_high_condition(
        M0=None,
        altitude_m=None,
        alpha_deg=None,
        margin_unstart=0.85,
):
    """
    Evaluate the fixed inlet geometry built from pyc_config at a higher
    Mach/altitude operating point.

    If M0, altitude_m, or alpha_deg are omitted, this uses the pyc_config
    design geometry and evaluates it at the DEFAULT_OFF_DESIGN_* settings.
    """
    design = pyc_run._get_inlet_design()
    M_eval = (
        float(M0) if M0 is not None
        else DEFAULT_OFF_DESIGN_MACH
    )
    altitude_eval = (
        float(altitude_m) if altitude_m is not None
        else DEFAULT_OFF_DESIGN_ALTITUDE_M
    )
    alpha_eval = (
        float(alpha_deg) if alpha_deg is not None
        else DEFAULT_OFF_DESIGN_ALPHA_DEG
    )

    result = evaluate_inlet_capacity_and_startability(
        design=design,
        M0=M_eval,
        altitude_m=altitude_eval,
        alpha_deg=alpha_eval,
        margin_unstart=margin_unstart,
    )
    result["geometry_design_M0"] = float(design["design_M0"])
    result["geometry_design_altitude_m"] = float(design["design_altitude_m"])
    result["geometry_design_alpha_deg"] = float(design["alpha_deg"])
    return result


def print_startability_result(title, result):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    # Uses your cached/default inlet design from pyc_config + pyc_run.
    design = pyc_run._get_inlet_design()

    design_result = evaluate_inlet_capacity_and_startability(
        design=design,
        M0=float(design["design_M0"]),
        altitude_m=float(design["design_altitude_m"]),
        alpha_deg= float(design["alpha_deg"]),
    )

    high_result = analyze_pyc_config_fixed_geometry_high_condition()

    print_startability_result(
        "INLET CAPACITY / STARTABILITY CHECK - DESIGN CONDITION",
        design_result,
    )
    print_startability_result(
        "INLET CAPACITY / STARTABILITY CHECK - HIGHER MACH/ALTITUDE",
        high_result,
    )
