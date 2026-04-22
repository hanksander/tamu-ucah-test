"""
Combustor — constant-area Rayleigh heat addition with JP-10 thermodynamics.

Energy balance
--------------
Heat added per kg of air (at combustion efficiency η_c):
    q = η_c · φ · f_stoich · LHV_JP10   [J/kg_air]

Total temperature rise via enthalpy balance (replaces the old cp·ΔT):
    h_products(Tt4, phi, P3) = h_air(Tt3, P3) + q

    This correctly accounts for:
      - cp varying strongly with temperature (≈1050 → 1340 J/kg·K over
        800–2500 K), which the old fixed-cp estimate ignored.
      - Product composition (CO₂, H₂O, CO, etc.) having a different cp
        than pure air at the same temperature.
      - Pressure-dependent dissociation equilibrium via Cantera at P3.

Rayleigh flow then gives M4 and Pt4/Pt3 for the computed Tt4/Tt3.

Exit gas properties (γ4, R4) come from the JP10Thermo Cantera equilibrium
at the actual T4, P4, and φ, capturing the real gas effect of dissociation.

CHANGES vs original
-------------------
- Replaced cp3·(Tt4−Tt3) energy balance with a brentq enthalpy iteration
  using thermo.h() and thermo.h_air() at the actual combustor pressure P3.
- thermo.gamma() and thermo.R() calls now pass the static pressure at each
  station (state3.P for entry, P4 for exit) so Cantera evaluates equilibrium
  at realistic pressures instead of always at 1 atm.
"""

import numpy as np
from scipy.optimize import brentq

from gas_dynamics import FlowState, rayleigh_exit, isentropic_T, isentropic_P
from pyc_config import F_STOICH, LHV_JP10, ETA_COMBUSTOR


def compute_combustor(
    state3: FlowState,
    phi: float,
    thermo,
    eta_c: float = ETA_COMBUSTOR,
    area_ratio: float = 1.0,
) -> tuple[FlowState, bool]:
    """
    Constant-area combustor with Rayleigh heat addition (subsonic branch).

    Parameters
    ----------
    state3 : FlowState   Combustor entrance (station 3)
    phi    : float       Equivalence ratio
    thermo : JP10Thermo  Thermodynamic model
    eta_c  : float       Combustion efficiency
    area_ratio : float   Exit/inlet area ratio

    Returns
    -------
    state4 : FlowState   Combustor exit (station 4)
    choked : bool        True if thermal choking occurred (Tt4/Tt* ≥ 1)
    """
    # Heat addition per kg air
    q = eta_c * phi * F_STOICH * LHV_JP10

    # Fuel-air mass ratio for this operating point
    f_ratio = phi * F_STOICH   # [kg_fuel / kg_air]

    Tt3 = state3.Tt
    P3  = state3.P

    # ── Enthalpy-balance iteration to find Tt4 ────────────────────────────
    # First-law balance for 1 kg of entering air:
    #
    #   h_air(Tt3) + q  =  (1 + f) · h_products(Tt4)
    #
    # Rearranged for the brentq target (J/kg_mix):
    #
    #   h_products(Tt4)  =  (h_air(Tt3) + q) / (1 + f)
    #
    # The (1 + f) divisor is essential because thermo.h() returns J/kg_mix
    # (per unit mass of the product mixture, which includes the added fuel
    # mass f kg per kg of air), while q and h_air are both J/kg_air.
    # Omitting the factor inflates the effective heat release by ~5% at
    # phi = 0.8 (f ≈ 0.053), raising Tt4 by ~75 K and biasing the
    # thermal-choking Mach number upward.
    #
    # Using the actual combustor static pressure P3 for both calls
    # correctly captures:
    #   1. The strong temperature dependence of cp (cp rises ~30% from
    #      inlet to exit temperatures — a fixed cp would overestimate Tt4).
    #   2. Product-mixture cp differing from pure-air cp.
    #   3. Pressure-shifted dissociation equilibrium at high T.
    h3     = thermo.h_air(Tt3, P3)
    target = (h3 + q) / (1.0 + f_ratio)   # [J/kg_mix]

    Tt4 = brentq(
        lambda T: thermo.h(T, phi, P3) - target,
        Tt3,
        Tt3 + 5000.0,
        xtol=5.0,
    )

    # ── Rayleigh flow: first pass with inlet γ ────────────────────────────
    # The Rayleigh constant-γ equations require a representative γ for the
    # combustor.  At inlet, γ_air ≈ 1.40; at exit, γ_products ≈ 1.20–1.25.
    # Using only γ_inlet overestimates Tt* (the sonic total temperature),
    # making the combustor appear harder to choke than it is.  For example,
    # at M3 = 0.35: γ=1.40 gives Tt*/Tt3 ≈ 3.10, while γ=1.25 gives 2.35.
    # A two-pass scheme is used: first pass with γ_inlet to get a first-guess
    # T4/P4, then look up γ_exit from Cantera and redo Rayleigh with
    # γ_avg = (γ_inlet + γ_exit) / 2 for a consistent choking check.
    M4, Pt4_Pt3, choked = rayleigh_exit(state3.M, Tt4 / Tt3,
                                         state3.gamma, supersonic=False)

    # First-pass exit static conditions using inlet γ
    Pt4 = state3.Pt * Pt4_Pt3
    T4  = isentropic_T(Tt4, M4, state3.gamma)
    P4  = isentropic_P(Pt4, M4, state3.gamma)

    # Look up exit γ and R at first-guess conditions
    gamma4 = thermo.gamma(T4, phi, P4)
    R4     = thermo.R(T4, phi, P4)

    # ── Rayleigh flow: refined pass with average γ ────────────────────────
    gamma_avg = 0.5 * (state3.gamma + gamma4)
    M4, Pt4_Pt3, choked = rayleigh_exit(state3.M, Tt4 / Tt3,
                                         gamma_avg, supersonic=False)

    # Recompute exit static conditions with refined M4 and exit γ
    Pt4 = state3.Pt * Pt4_Pt3
    T4  = isentropic_T(Tt4, M4, gamma4)
    P4  = isentropic_P(Pt4, M4, gamma4)

    # Final refinement of γ and R at actual exit (T4, P4)
    gamma4 = thermo.gamma(T4, phi, P4)
    R4     = thermo.R(T4, phi, P4)
    T4     = isentropic_T(Tt4, M4, gamma4)
    P4     = isentropic_P(Pt4, M4, gamma4)

    return FlowState(M=M4, T=T4, P=P4, Pt=Pt4, Tt=Tt4,
                     gamma=gamma4, R=R4), choked


def _mixture_total_enthalpy(thermo, Tt: float, phi: float, P_ref: float) -> float:
    """Mixture total enthalpy at the current composition basis."""
    if phi <= 1.0e-12:
        return thermo.h_air(Tt, P_ref)
    return thermo.h(Tt, phi, P_ref)


def _solve_step_total_temperature(
    Tt_in: float,
    P_ref: float,
    phi_in: float,
    phi_out: float,
    dq_step: float,
    thermo,
) -> float:
    """Solve total temperature after a small heat-addition step.

    Heat is specified per kg of incoming air. Thermo enthalpy is on a
    per-kg-mixture basis, so mixture mass growth from fuel addition is
    accounted for explicitly across the step.
    """
    f_in = phi_in * F_STOICH
    f_out = phi_out * F_STOICH
    h_t_in = _mixture_total_enthalpy(thermo, Tt_in, phi_in, P_ref)
    target = ((1.0 + f_in) * h_t_in + dq_step) / max(1.0 + f_out, 1.0e-12)

    lo = Tt_in
    hi = Tt_in + 5000.0
    f_lo = thermo.h(lo, phi_out, P_ref) - target
    f_hi = thermo.h(hi, phi_out, P_ref) - target

    if f_lo > 0.0:
        # This should be rare for a positive heat-addition step, but allow a
        # small backoff in case the mixture-basis change makes the target dip.
        for _ in range(12):
            lo = max(50.0, 0.8 * lo)
            f_lo = thermo.h(lo, phi_out, P_ref) - target
            if f_lo <= 0.0:
                break

    if f_hi < 0.0:
        for _ in range(12):
            hi = hi + max(1000.0, 0.5 * hi)
            f_hi = thermo.h(hi, phi_out, P_ref) - target
            if f_hi >= 0.0:
                break

    if f_lo * f_hi > 0.0:
        raise ValueError(
            "Could not bracket total-temperature root in "
            f"_solve_step_total_temperature: phi_in={phi_in:.6f}, "
            f"phi_out={phi_out:.6f}, Tt_in={Tt_in:.3f}, "
            f"f(lo)={f_lo:.6e}, f(hi)={f_hi:.6e}"
        )

    return brentq(
        lambda T: thermo.h(T, phi_out, P_ref) - target,
        lo,
        hi,
        xtol=5.0,
    )


def _advance_variable_rayleigh_step(
    state_in: FlowState,
    phi_in: float,
    phi_out: float,
    dq_step: float,
    thermo,
) -> tuple[FlowState, bool]:
    """Advance one small constant-area heat-addition step."""
    Tt_out = _solve_step_total_temperature(
        Tt_in=state_in.Tt,
        P_ref=state_in.P,
        phi_in=phi_in,
        phi_out=phi_out,
        dq_step=dq_step,
        thermo=thermo,
    )

    M_out, Pt_ratio, choked = rayleigh_exit(
        state_in.M,
        Tt_out / max(state_in.Tt, 1.0e-12),
        state_in.gamma,
        supersonic=False,
    )

    Pt_out = state_in.Pt * Pt_ratio
    T_out = isentropic_T(Tt_out, M_out, state_in.gamma)
    P_out = isentropic_P(Pt_out, M_out, state_in.gamma)
    gamma_out = thermo.gamma(T_out, phi_out, P_out)
    R_out = thermo.R(T_out, phi_out, P_out)

    gamma_avg = 0.5 * (state_in.gamma + gamma_out)
    M_out, Pt_ratio, choked = rayleigh_exit(
        state_in.M,
        Tt_out / max(state_in.Tt, 1.0e-12),
        gamma_avg,
        supersonic=False,
    )

    Pt_out = state_in.Pt * Pt_ratio
    T_out = isentropic_T(Tt_out, M_out, gamma_out)
    P_out = isentropic_P(Pt_out, M_out, gamma_out)

    gamma_out = thermo.gamma(T_out, phi_out, P_out)
    R_out = thermo.R(T_out, phi_out, P_out)
    T_out = isentropic_T(Tt_out, M_out, gamma_out)
    P_out = isentropic_P(Pt_out, M_out, gamma_out)

    return FlowState(
        M=M_out,
        T=T_out,
        P=P_out,
        Pt=Pt_out,
        Tt=Tt_out,
        gamma=gamma_out,
        R=R_out,
    ), choked


def _bisect_variable_rayleigh_sonic_step(
    state_in: FlowState,
    phi_in: float,
    phi_out_full: float,
    dq_step_full: float,
    thermo,
    n_iter: int = 24,
) -> FlowState:
    """Back out the exact heat-addition fraction where the march reaches sonic."""
    lo = 0.0
    hi = 1.0
    sonic_state = None

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        phi_mid = phi_in + mid * (phi_out_full - phi_in)
        dq_mid = mid * dq_step_full
        state_mid, choked_mid = _advance_variable_rayleigh_step(
            state_in=state_in,
            phi_in=phi_in,
            phi_out=phi_mid,
            dq_step=dq_mid,
            thermo=thermo,
        )
        if choked_mid or state_mid.M >= 1.0:
            sonic_state = state_mid
            hi = mid
        else:
            lo = mid

    if sonic_state is None:
        sonic_state, _ = _advance_variable_rayleigh_step(
            state_in=state_in,
            phi_in=phi_in,
            phi_out=phi_out_full,
            dq_step=dq_step_full,
            thermo=thermo,
        )
    return sonic_state


def compute_combustor_variable_rayleigh(
    state3: FlowState,
    phi: float,
    thermo,
    eta_c: float = ETA_COMBUSTOR,
    area_ratio: float = 1.0,
    n_steps: int = 50,
    sonic_bisect_iters: int = 24,
) -> tuple[FlowState, bool]:
    """Variable-property Rayleigh march for a constant-area combustor.

    This solver distributes the total heat addition over many small steps,
    updates thermo properties at each step, and uses local Rayleigh relations
    with a representative gamma for each increment. It is implemented for a
    constant-area combustor only and is not wired into the cycle yet.
    """
    if not np.isfinite(phi) or phi < 0.0:
        raise ValueError("phi must be finite and non-negative.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if abs(area_ratio - 1.0) > 1.0e-9:
        raise ValueError(
            "compute_combustor_variable_rayleigh currently supports constant-area "
            "combustors only (area_ratio must be 1.0)."
        )
    if phi <= 1.0e-12:
        return state3, False

    q_total = eta_c * phi * F_STOICH * LHV_JP10
    dq_step = q_total / float(n_steps)
    dphi = phi / float(n_steps)

    state = state3
    phi_curr = 0.0
    choked = False

    for _ in range(n_steps):
        phi_next = min(phi, phi_curr + dphi)
        state_next, choked_step = _advance_variable_rayleigh_step(
            state_in=state,
            phi_in=phi_curr,
            phi_out=phi_next,
            dq_step=dq_step,
            thermo=thermo,
        )

        if choked_step or state_next.M >= 1.0:
            state = _bisect_variable_rayleigh_sonic_step(
                state_in=state,
                phi_in=phi_curr,
                phi_out_full=phi_next,
                dq_step_full=dq_step,
                thermo=thermo,
                n_iter=sonic_bisect_iters,
            )
            choked = True
            break

        state = state_next
        phi_curr = phi_next

    return state, choked


def build_station3_state_from_totals(
    Pt3: float,
    Tt3: float,
    M3: float,
    thermo,
) -> FlowState:
    """Reconstruct combustor-inlet static state from totals + Mach.

    Uses a first-pass perfect-air inversion followed by a thermo-refined
    gamma/R update so standalone combustor queries match the burner path.
    """
    gamma3 = 1.40
    R3 = 287.05

    T3 = isentropic_T(Tt3, M3, gamma3)
    P3 = isentropic_P(Pt3, M3, gamma3)

    gamma3 = thermo.gamma(T3, 0.0, P3)
    R3 = thermo.R(T3, 0.0, P3)

    T3 = isentropic_T(Tt3, M3, gamma3)
    P3 = isentropic_P(Pt3, M3, gamma3)

    return FlowState(M=M3, T=T3, P=P3, Pt=Pt3, Tt=Tt3, gamma=gamma3, R=R3)


# ---------------------------------------------------------------------------
# Standalone wrapper for the inlet-limited RAM closure (Phase 1).
#
# The φ-envelope solver (Stage B of the new closure) needs to evaluate the
# combustor-face response at many trial φ values while sweeping for the
# inlet's Ps_max, the thermal-choke limit, and the Tt4 material limit. It
# cannot afford to go through pyCycle. This wrapper takes station-3 total
# conditions + Mach (which is all the envelope solver has) and returns the
# station-4 numbers the solver needs, with zero OpenMDAO/pyCycle coupling.
# ---------------------------------------------------------------------------

def combustor_face_response(
    Pt3: float,
    Tt3: float,
    M3:  float,
    phi: float,
    thermo,
    area_ratio: float = 1.0,
    eta_c: float = ETA_COMBUSTOR,
    model: str = "variable_property_rayleigh",
    n_steps: int = 80,
    sonic_bisect_iters: int = 24,
) -> dict:
    """Standalone combustor evaluation for the φ-envelope solver.

    Builds a station-3 FlowState from total conditions + Mach, runs
    compute_combustor, and returns the station-4 observables the envelope
    solver roots on: Ps4 (matched against the inlet's Ps_max), M4 (thermal-
    choke limit), Tt4 (material limit), plus Pt4 for diagnostics.
    """
    state3 = build_station3_state_from_totals(
        Pt3=Pt3,
        Tt3=Tt3,
        M3=M3,
        thermo=thermo,
    )

    if model == "avg_gamma_rayleigh":
        state4, choked = compute_combustor(
            state3,
            phi=phi,
            thermo=thermo,
            eta_c=eta_c,
            area_ratio=area_ratio,
        )
    elif model == "variable_property_rayleigh":
        state4, choked = compute_combustor_variable_rayleigh(
            state3,
            phi=phi,
            thermo=thermo,
            eta_c=eta_c,
            area_ratio=area_ratio,
            n_steps=n_steps,
            sonic_bisect_iters=sonic_bisect_iters,
        )
    else:
        raise ValueError(
            "model must be 'avg_gamma_rayleigh' or "
            "'variable_property_rayleigh'."
        )

    return {
        'Ps4':     float(state4.P),
        'M4':      float(state4.M),
        'Tt4':     float(state4.Tt),
        'Pt4':     float(state4.Pt),
        'gamma4':  float(state4.gamma),
        'R4':      float(state4.R),
        'choked':  bool(choked),
        'f_ratio': float(phi * F_STOICH),
    }
