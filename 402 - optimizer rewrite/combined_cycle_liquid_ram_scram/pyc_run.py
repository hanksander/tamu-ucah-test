"""
pyc_run.py
==========
Analysis runner for the pyCycle ramjet.

User-visible functions
----------------------
    analyze(M0, altitude_m, phi, ramp_angles, verbose)
    mach_sweep(mach_range, altitude_m, phi)

Unit conventions
----------------
All public inputs / outputs are SI (m, K, Pa, N, s, kg/s).
pyCycle internals use American engineering units; conversions are handled
here via OpenMDAO's unit-aware get_val / set_val.

FloatingPointError fix
----------------------
pyCycle's FlightConditions internally solves a Newton balance for Tt and Pt.
The default initial guesses (500 degR, 14.696 psi) are wildly wrong at
high Mach (e.g., Tt0 ~ 5500 degR at M=8).  CEA hits log(0) on species
concentrations within the first Newton step.  The fix: seed
fc.conv.balance.Tt and fc.conv.balance.Pt with isentropic estimates before
every run_model() call.
"""


import sys, os
sys.path.insert(0, os.path.dirname(__file__))
#sys.path.insert(0, _HERE)
import warnings
warnings.filterwarnings('ignore')

import importlib.util
import numpy as np
import openmdao.api as om
from ambiance import Atmosphere

from gas_dynamics import FlowState, isentropic_P, isentropic_T
from combustor import combustor_face_response
from pyc_config import (
    F_STOICH_JP10, ETA_COMBUSTOR,
    ETA_NOZZLE_CV,
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M,
    INLET_DESIGN_ALPHA_DEG, INLET_DESIGN_LEADING_EDGE_ANGLE_DEG,
    INLET_DESIGN_MDOT_KGS, INLET_DESIGN_WIDTH_M,
    INLET_FOREBODY_SEP_MARGIN, INLET_RAMP_SEP_MARGIN,
    INLET_KANTROWITZ_MARGIN, INLET_SHOCK_FOCUS_FACTOR,
    NOZZLE_TYPE,
    TT4_MAX_K, M4_MAX, COMBUSTOR_LENGTH_M_DEFAULT,
    DIFFUSER_AREA_RATIO, NOZZLE_AR, NOZZLE_AR_DEFAULT,
    DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH,
    PHI_DEFAULT,
)
from pyc_ram_cycle   import RamCycle
import nozzle_design
from thermo import get_thermo

# 402inlet2.py — module name begins with a digit, import via importlib
_inlet2_spec = importlib.util.spec_from_file_location(
    'inlet2', os.path.join(os.path.dirname(__file__), '402inlet2.py'))
_inlet2 = importlib.util.module_from_spec(_inlet2_spec)
_inlet2_spec.loader.exec_module(_inlet2)

G0      = 9.80665    # m/s^2
AIR_GAM = 1.40
AIR_R   = 287.05     # J/(kg*K)
K2R     = 1.8        # Kelvin -> Rankine
PA2PSI  = 1.0 / 6894.757
M2FT    = 3.28084
KG2LBM  = 2.20462


# --- New: parameterized design builder ---
def build_design(
    M0=None, altitude_m=None, alpha_deg=None,
    leading_edge_angle_deg=None, mdot_required=None, width_m=None,
    forebody_separation_margin=None, ramp_separation_margin=None,
    kantrowitz_margin=None, shock_focus_factor=None,
    diffuser_area_ratio=None, diffuser_min_shock_accommodation_dh=None,
    combustor_length_m=None, nozzle_AR=None, design_phi=None,
):
    """Build an inlet+geometry design dict. Any arg left as None falls
    back to the pyc_config default."""
    def pick(x, d): return d if x is None else x
    alpha_deg_eff = pick(alpha_deg, INLET_DESIGN_ALPHA_DEG)
    M0_eff        = pick(M0, INLET_DESIGN_M0)
    alt_eff       = pick(altitude_m, INLET_DESIGN_ALT_M)
    design = _inlet2.design_2ramp_shock_matched_inlet(
        M0=M0_eff,
        altitude_m=alt_eff,
        alpha_deg=alpha_deg_eff,
        leading_edge_angle_deg=pick(leading_edge_angle_deg,
                                    INLET_DESIGN_LEADING_EDGE_ANGLE_DEG),
        mdot_required=pick(mdot_required, INLET_DESIGN_MDOT_KGS),
        width_m=pick(width_m, INLET_DESIGN_WIDTH_M),
        forebody_separation_margin=pick(forebody_separation_margin,
                                        INLET_FOREBODY_SEP_MARGIN),
        ramp_separation_margin=pick(ramp_separation_margin,
                                    INLET_RAMP_SEP_MARGIN),
        kantrowitz_margin=pick(kantrowitz_margin, INLET_KANTROWITZ_MARGIN),
        shock_focus_factor=pick(shock_focus_factor, INLET_SHOCK_FOCUS_FACTOR),
    )
    diffuser_area_ratio_eff = pick(diffuser_area_ratio, DIFFUSER_AREA_RATIO)
    diffuser_min_shock_accommodation_dh_eff = pick(
        diffuser_min_shock_accommodation_dh,
        DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH,
    )
    diffuser = _inlet2.build_subsonic_diffuser(
        T_upper=np.asarray(design["throat_upper_xy"], dtype=float),
        T_lower=np.asarray(design["throat_lower_xy"], dtype=float),
        h_throat=float(design["throat_height_m"]),
        width_m=pick(width_m, INLET_DESIGN_WIDTH_M),
        area_ratio_exit_to_throat=diffuser_area_ratio_eff,
        min_shock_accommodation_dh=diffuser_min_shock_accommodation_dh_eff,
    )
    # attach the extras not consumed by the 2-ramp solver. alpha_deg is stored
    # here so the RAM closure evaluates at the same α used to design geometry.
    design['alpha_deg']           = float(alpha_deg_eff)
    design['design_M0']           = float(M0_eff)
    design['design_altitude_m']   = float(alt_eff)
    design['design_phi']          = float(pick(design_phi, PHI_DEFAULT))
    design['diffuser']            = diffuser
    design['combustor_face_xy_upper'] = diffuser["exit_upper_xy"]
    design['combustor_face_xy_lower'] = diffuser["exit_lower_xy"]
    design['A_combustor_face']        = diffuser["A_exit"]
    design['diffuser_area_ratio'] = diffuser_area_ratio_eff
    design['diffuser_min_shock_accommodation_dh'] = diffuser_min_shock_accommodation_dh_eff
    design['combustor_length_m']  = float(pick(combustor_length_m, COMBUSTOR_LENGTH_M_DEFAULT))
    design['nozzle_AR']           = float(pick(nozzle_AR, NOZZLE_AR))

    # Commit A_noz_throat analytically at the design point. The choked-nozzle
    # mass balance ties mass flow, combustor-exit totals, and throat area:
    #   mdot_total = Pt4 · A_t · Γ(γ,R) / √Tt4
    # With mdot known from freestream capture, and (Tt4, Pt4, γ4, R4) from
    # the combustor at the design φ, we invert for A_t. This value is frozen
    # so that every off-design phi-envelope solve references the same throat.
    # A_noz_exit is set from the commanded nozzle_AR (pyc_config NOZZLE_AR or
    # caller override) — NOT from ideal-expansion sizing — so the geometry is
    # in general under-, ideally-, or over-expanded at any given flight point.
    _design_pt = _compute_design_point_A_noz_throat(design)
    design['A_noz_throat']  = float(_design_pt['A_noz_throat'])
    design['A_noz_exit']    = float(design['A_noz_throat'] * design['nozzle_AR'])
    design['design_point']  = _design_pt
    return design


def _compute_design_point_A_noz_throat(design: dict) -> dict:
    """Analytic A_noz_throat at the design (M0, alt, alpha, phi) from the
    cascade+diffuser + Rayleigh + choked-nozzle balance."""
    M0_des   = float(design['design_M0'])
    alt_des  = float(design['design_altitude_m'])
    alpha_des = float(design['alpha_deg'])
    phi_des  = float(design['design_phi'])

    cap = _inlet2.compute_inlet_capability(
        design, M0=M0_des, altitude_m=alt_des, alpha_deg=alpha_des,
    )
    if not cap.get('ok', False):
        raise RuntimeError(
            f"Design-point inlet capability failed: {cap.get('reason','?')}"
        )

    atm  = Atmosphere(alt_des)
    T0   = float(atm.temperature[0])
    rho0 = float(atm.density[0])
    V0   = M0_des * np.sqrt(AIR_GAM * AIR_R * T0)
    A_capture = float(design['A_capture_required_m2'])
    W_air = rho0 * V0 * A_capture

    face = combustor_face_response(
        Pt3=float(cap['Pt3_deliverable']),
        Tt3=float(cap['Tt0']),
        M3=float(cap['M3_diffuser_exit']),
        phi=phi_des,
        thermo=get_thermo(),
        area_ratio=1.0,   # constant-area combustor
    )
    gamma4 = float(face['gamma4'])
    R4     = float(face['R4'])
    Pt4    = float(face['Pt4'])
    Tt4    = float(face['Tt4'])
    FAR    = float(face['f_ratio'])
    mdot_total = W_air * (1.0 + FAR)

    gp1 = gamma4 + 1.0
    gm1 = gamma4 - 1.0
    Gamma_mass = np.sqrt(gamma4 / R4) * (2.0 / gp1) ** (gp1 / (2.0 * gm1))
    A_noz_throat = mdot_total * np.sqrt(Tt4) / (Pt4 * Gamma_mass)

    P0_des = float(Atmosphere(alt_des).pressure[0])
    return {
        'A_noz_throat':     float(A_noz_throat),
        'Pt4_design':       Pt4,
        'Tt4_design':       Tt4,
        'gamma4_design':    gamma4,
        'R4_design':        R4,
        'mdot_air_design':  float(W_air),
        'mdot_total_design':float(mdot_total),
        'M0_design':        M0_des,
        'altitude_m_design':alt_des,
        'alpha_deg_design': alpha_des,
        'phi_design':       phi_des,
        'P0_design':        float(P0_des),
    }




# ---------------------------------------------------------------------------
# Isentropic initial-guess seeding
# ---------------------------------------------------------------------------

def _seed_fc_initial_guess(prob, M0, T0_K, P0_Pa):
    """
    Seed the FlightConditions Newton balance with isentropic estimates.

    pyCycle's FlightConditions contains:
        fc.conv.balance.Tt  (default 500 degR)
        fc.conv.balance.Pt  (default 14.696 psi)

    At high Mach these defaults are 10x or more below the correct values,
    driving CEA species concentrations to zero on the first Newton step and
    raising FloatingPointError: divide by zero encountered in log.

    Setting them to the isentropic total conditions gives the Newton solver
    a starting point close enough for CEA to remain physical.
    """
    gam    = AIR_GAM
    Tt0_K  = T0_K  * (1.0 + 0.5 * (gam - 1.0) * M0 ** 2)
    Pt0_Pa = P0_Pa * (Tt0_K / T0_K) ** (gam / (gam - 1.0))

    Tt0_R   = Tt0_K  * K2R
    Pt0_psi = Pt0_Pa * PA2PSI

    try:
        prob.set_val('fc.conv.balance.Tt', Tt0_R,   units='degR')
        prob.set_val('fc.conv.balance.Pt', Pt0_psi, units='psi')
    except (KeyError, AttributeError):
        pass   # path may differ in some pyCycle versions; best-effort only


# ---------------------------------------------------------------------------
# Problem singleton  (built once, reused across the Mach sweep)
# ---------------------------------------------------------------------------

_ram_prob = None


def _make_problem(CycleClass):
    """Build and set up one pyCycle Problem."""
    prob = om.Problem(reports=False)
    prob.model = CycleClass(design=True)

    # Newton with solve_subsystems=True lets each pyCycle element's own
    # internal solver (FlightConditions balance, Nozzle PR_bal, etc.) run
    # to convergence on the first Newton step, giving a good initial Jacobian.
    # The props_calcs.py n_moles[0] patch (applied to the installed package)
    # is required for compute_partials to work without a ValueError.
    nlsolver = om.NewtonSolver(solve_subsystems=True)
    nlsolver.options['maxiter']             = 12
    nlsolver.options['atol']               = 1e-5
    nlsolver.options['rtol']               = 1e-5
    nlsolver.options['iprint']             = -1
    nlsolver.options['err_on_non_converge'] = False
    nlsolver.linesearch = om.BoundsEnforceLS()
    nlsolver.linesearch.options['bound_enforcement'] = 'scalar'
    nlsolver.linesearch.options['iprint']  = -1
    prob.model.nonlinear_solver = nlsolver

    lsolver = om.DirectSolver()
    lsolver.options['iprint'] = -1
    prob.model.linear_solver = lsolver

    prob.setup(force_alloc_complex=False)

    # Fixed model parameters
    prob.set_val('nozz.Cv', ETA_NOZZLE_CV)

    return prob


def _get_problem():
    """Lazily build and return the singleton RAM Problem."""
    global _ram_prob
    if _ram_prob is None:
        _ram_prob = _make_problem(RamCycle)
    return _ram_prob


# ---------------------------------------------------------------------------
# Inlet physics  (fixed 2-ramp shock-matched geometry from 402inlet2.py)
# ---------------------------------------------------------------------------
#
# Geometry is designed once at first use from pyc_config defaults, then
# evaluate_fixed_geometry_at_condition() is called per flight point.
# If the frozen geometry cannot pass the flow at a given (M0, alpha),
# the evaluator returns success=False and we fall back to MIL-E-5007D.

_inlet_design = None


def _get_inlet_design():
    """Build and cache the frozen inlet geometry from pyc_config defaults.

    Routes through build_design() so the returned dict includes the
    diffuser/combustor/nozzle extras and the alpha_deg key that downstream
    RAM-closure sites read.
    """
    global _inlet_design
    if _inlet_design is None:
        _inlet_design = build_design()
    return _inlet_design


def compute_inlet_conditions(M0, alt_m, ramp_angles=None, alpha_deg=0.0):
    """
    analyze() closes the inlet/back-pressure consistency with a bracketed
    scalar solve outside the pyCycle model, so no per-call evaluation is
    performed here.
    """
    del ramp_angles, M0, alt_m, alpha_deg
    return None, None


# ---------------------------------------------------------------------------
# Cascade-only inlet-limited RAM closure
# ---------------------------------------------------------------------------
#
# The old v2 closure swept Ps_back through the diffuser and picked a terminal
# shock station — doubly counting Pt losses that the reflection cascade was
# already applying. The new model is cascade-only:
#
#   (a) _inlet2.compute_inlet_capability returns one number for Pt3 delivered
#       to the combustor face (cascade Pt-loss + diffuser friction).
#   (b) _solve_phi_envelope sweeps φ through the combustor at fixed (Pt3,
#       Tt3, M3), then enforces three caps:
#         – phi_Tt4          : material Tt4_max
#         – phi_choke        : thermal-choke margin on M4
#         – phi_inlet_limit  : the φ at which Pt3_req(φ) = Pt3_deliverable,
#                              where Pt3_req is inverted from the choked
#                              nozzle + Rayleigh chain given the frozen
#                              A_noz_throat. Beyond this φ, the combustor
#                              needs more Pt3 than the inlet can deliver
#                              → inlet-expulsion unstart.
#       All three are soft-minned with phi_request.
#   (c) analyze() runs pyCycle exactly once at the clipped φ with
#       ram_recovery = Pt3_deliverable / Pt0 and MN = M3_diffuser_exit.

from scipy.interpolate import PchipInterpolator as _Pchip

_SOFTMIN_K = 15.0  # larger → sharper soft-min (closer to true min)               #LOWER

_inlet_cap_cache: dict = {}
_INLET_CAP_CACHE_MAX = 2048


def _softmin(values, k: float = _SOFTMIN_K) -> float:
    """Smooth min via  -log(Σ exp(-k·x)) / k. Stable with offset trick."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    if arr.size == 1:
        return float(arr[0])
    x_min = arr.min()
    return float(x_min - np.log(np.sum(np.exp(-k * (arr - x_min)))) / k)


def _design_digest(design: dict) -> tuple:
    """Stable cache key for a frozen inlet design. Rounds geometry to the
    tolerance of physics-level cache keys so FD perturbations still hit.
    """
    return (
        round(float(design.get('theta_fore_deg', 0.0)), 3),
        round(float(design.get('theta1_deg', 0.0)), 3),
        round(float(design.get('theta2_deg', 0.0)), 3),
        round(float(design.get('theta_cowl_deg', 0.0)), 3),
        round(float(design.get('A_capture_required_m2', 0.0)), 6),
        round(float(design.get('throat_area_actual_m2', 0.0)), 6),
        round(float(design.get('A_noz_throat', 0.0)), 6),
        round(float(design.get('diffuser_area_ratio', 0.0)), 4),
        round(float(design.get('diffuser_min_shock_accommodation_dh', 0.0)), 4),
    )


def _get_capability(design, M0, altitude_m, alpha_deg):
    """Memoized inlet capability lookup."""
    key = (
        round(float(M0), 2),
        round(float(altitude_m), -2),
        round(float(alpha_deg), 2),
        _design_digest(design),
    )
    cap = _inlet_cap_cache.get(key)
    if cap is None:
        cap = _inlet2.compute_inlet_capability(
            design, M0=M0, altitude_m=altitude_m, alpha_deg=alpha_deg,
        )
        if len(_inlet_cap_cache) >= _INLET_CAP_CACHE_MAX:
            _inlet_cap_cache.clear()
        _inlet_cap_cache[key] = cap
    return cap


def _choked_nozzle_mass_param(gamma: float, R: float) -> float:
    """Γ(γ,R) = √(γ/R) · (2/(γ+1))^((γ+1)/(2(γ-1)))  — the choked-nozzle
    mass-flow coefficient: mdot = Pt · A · Γ / √Tt."""
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return float(np.sqrt(gamma / R) * (2.0 / gp1) ** (gp1 / (2.0 * gm1)))


def _solve_phi_envelope(capability, A_noz_throat, W_air,
                         Tt4_max, phi_request, thermo,
                         M4_max: float = 0.95,
                         phi_grid: np.ndarray | None = None):
    """Cascade-only φ-envelope.

    Caps applied (soft-min over all four):
      φ_request        — user command
      φ_Tt4            — material Tt4 limit
      φ_choke          — thermal-choke margin on M4
      φ_inlet_limit    — Pt3_req(φ) = Pt3_deliverable (inlet expulsion)

    Each cap is found by Brent root-finding on obs(φ) − target, so cap
    values are smooth in flight condition (no coarse-grid PCHIP topology
    artefacts). Pt3_req(φ) is computed inline from the choked-nozzle mass
    balance and the Rayleigh combustor face response at φ:
        Pt4_req = W_tot · √Tt4 / (A_noz_throat · Γ(γ4, R4))
        π_comb  = Pt4 / Pt3
        Pt3_req = Pt4_req / π_comb,  W_tot = W_air · (1 + FAR(φ)).
    """
    if not capability.get('ok', False):
        return {
            'ok': False,
            'reason': capability.get('reason', 'capability not available'),
            'phi_cap': 0.0,
            'phi_inlet_limit': 0.0,
        }

    Pt3 = float(capability['Pt3_deliverable'])
    Tt3 = float(capability['Tt0'])
    M3  = float(capability['M3_diffuser_exit'])

    PHI_LO = 0.02
    phi_top = max(0.95, float(phi_request) + 0.10)

    def _obs_at(phi):
        face = combustor_face_response(
            Pt3=Pt3, Tt3=Tt3, M3=M3, phi=float(phi),
            thermo=thermo, area_ratio=1.0,  # constant-area combustor
        )
        Tt4      = float(face['Tt4'])
        Pt4_here = float(face['Pt4'])
        gamma4   = float(face['gamma4'])
        R4       = float(face['R4'])
        FAR      = float(face['f_ratio'])
        M4_here  = float(face['M4'])
        Gamma_mass = _choked_nozzle_mass_param(gamma4, R4)
        mdot_total = W_air * (1.0 + FAR)
        Pt4_req    = mdot_total * np.sqrt(Tt4) / (A_noz_throat * Gamma_mass)
        pi_comb    = Pt4_here / max(Pt3, 1.0e-9)
        Pt3_req    = Pt4_req / max(pi_comb, 1.0e-9)
        return Tt4, M4_here, Pt4_here, Pt3_req

    def _invert(obs_index, target):
        """Brentq on obs[obs_index](φ) − target over [PHI_LO, phi_top].

        target ≥ obs(phi_top) → cap never binds → +inf
        target ≤ obs(PHI_LO)  → already violated at floor → PHI_LO
        """
        target = float(target)
        obs_lo = _obs_at(PHI_LO)[obs_index]
        obs_hi = _obs_at(phi_top)[obs_index]
        if not (np.isfinite(obs_lo) and np.isfinite(obs_hi)):
            return float('nan')
        if target >= obs_hi:
            return float('inf')
        if target <= obs_lo:
            return float(PHI_LO)
        def _res(phi):
            return _obs_at(phi)[obs_index] - target
        try:
            return float(_brentq(_res, PHI_LO, phi_top,
                                  xtol=1.0e-5, rtol=1.0e-6, maxiter=60))
        except Exception:
            return float('nan')

    phi_Tt4         = _invert(0, Tt4_max)
    phi_choke       = _invert(1, M4_max)
    phi_inlet_limit = _invert(3, Pt3)

    phi_caps = np.array(
        [float(phi_request), phi_Tt4, phi_choke, phi_inlet_limit],
        dtype=float,
    )
    phi_cap = _softmin(phi_caps, k=_SOFTMIN_K)
    phi_cap = float(max(PHI_LO, phi_cap))

    face_cap = combustor_face_response(
        Pt3=Pt3, Tt3=Tt3, M3=M3, phi=phi_cap,
        thermo=thermo, area_ratio=1.0,
    )

    # Optional diagnostic grid (not used by the solver; kept so downstream
    # plotting code can still introspect the φ-observable curves).
    phi_grid_diag = (np.linspace(PHI_LO, phi_top, 12) if phi_grid is None
                     else np.asarray(phi_grid, dtype=float))
    Tt4_arr     = np.empty_like(phi_grid_diag)
    M4_arr      = np.empty_like(phi_grid_diag)
    Pt4_arr     = np.empty_like(phi_grid_diag)
    Pt3_req_arr = np.empty_like(phi_grid_diag)
    for i, phi in enumerate(phi_grid_diag):
        Tt4_arr[i], M4_arr[i], Pt4_arr[i], Pt3_req_arr[i] = _obs_at(phi)

    return {
        'ok':              True,
        'phi_inlet_limit': phi_inlet_limit,
        'phi_choke':       phi_choke,
        'phi_Tt4':         phi_Tt4,
        'phi_request':     float(phi_request),
        'phi_cap':         phi_cap,
        'M3':              M3,
        'Pt3':             Pt3,
        'Tt3':             Tt3,
        'Ps4':             float(face_cap['Ps4']),
        'M4':              float(face_cap['M4']),
        'Tt4':             float(face_cap['Tt4']),
        'Pt4':             float(face_cap['Pt4']),
        'grid_phi':        phi_grid_diag,
        'Tt4_arr':         Tt4_arr,
        'M4_arr':          M4_arr,
        'Pt4_arr':         Pt4_arr,
        'Pt3_req_arr':     Pt3_req_arr,
    }


def _build_air_state_from_totals(M, Tt, Pt, thermo=None):
    """
    Reconstruct an air FlowState from total conditions and Mach.

    Used for reporting pre-combustor diagnostics at stations not represented
    by an explicit pyCycle flow port, such as the diffuser entrance.
    """
    if thermo is None:
        thermo = get_thermo()

    gamma = AIR_GAM
    gas_r = AIR_R
    T = float(Tt)
    P = float(Pt)

    for _ in range(3):
        T = isentropic_T(Tt, M, gamma)
        P = isentropic_P(Pt, M, gamma)
        gamma = thermo.gamma(T, 0.0, P)
        gas_r = thermo.R(T, 0.0, P)

    return FlowState(
        M=float(M),
        T=float(T),
        P=float(P),
        Pt=float(Pt),
        Tt=float(Tt),
        gamma=float(gamma),
        R=float(gas_r),
    )


# ---------------------------------------------------------------------------
# Combustor geometry helpers
# ---------------------------------------------------------------------------

def compute_combustor_geometry(
    combustor_length_m: float,
    design: dict | None = None,
    cross_section_area_m2: float | None = None,
    cross_section_shape: str | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
) -> dict:
    """
    Size a constant-area combustor from a user-supplied chamber length.

    Cross-section is taken from the diffuser exit area when a diffuser block
    is attached to `design` (circular). A rectangular fallback is retained
    for designs without a diffuser. Volume = A_cc · L_cc.
    """
    if combustor_length_m <= 0.0:
        raise ValueError("combustor_length_m must be positive.")

    if cross_section_area_m2 is None:
        diff = design.get("diffuser") if isinstance(design, dict) else None
        if diff is not None:
            cross_section_area_m2 = float(diff["A_exit"])
            if cross_section_shape is None:
                cross_section_shape = "circular"
        else:
            if width_m is None:
                width_m = INLET_DESIGN_WIDTH_M
            if height_m is None:
                if design is None:
                    raise ValueError("geometry inputs or design must be provided.")
                if "throat_height_m" in design:
                    height_m = float(design["throat_height_m"])
                else:
                    t_up = np.asarray(design["throat_upper_xy"], dtype=float)
                    t_lo = np.asarray(design["throat_lower_xy"], dtype=float)
                    height_m = float(abs(t_up[1] - t_lo[1]))
            if width_m <= 0.0 or height_m <= 0.0:
                raise ValueError("Combustor width and height must be positive.")
            cross_section_area_m2 = width_m * height_m
            if cross_section_shape is None:
                cross_section_shape = "rectangular"

    combustor_area = float(cross_section_area_m2)
    if combustor_area <= 0.0:
        raise ValueError("Combustor cross_section_area_m2 must be positive.")
    combustor_length = float(combustor_length_m)
    combustor_volume = combustor_area * combustor_length

    geometry = {
        "cross_section_shape": str(cross_section_shape or "circular"),
        "cross_section_area_m2": float(combustor_area),
        "volume_m3": float(combustor_volume),
        "length_m": float(combustor_length),
    }

    if geometry["cross_section_shape"] == "circular":
        radius_m = np.sqrt(combustor_area / np.pi)
        geometry.update({
            "radius_m": float(radius_m),
            "diameter_m": float(2.0 * radius_m),
            "width_m": float(2.0 * radius_m),
            "height_m": float(2.0 * radius_m),
        })
    else:
        geometry.update({
            "width_m": float(width_m),
            "height_m": float(height_m),
            "radius_m": float(np.sqrt(combustor_area / np.pi)),
            "diameter_m": float(np.sqrt(4.0 * combustor_area / np.pi)),
        })

    return geometry


def _build_inlet_geometry_summary(design: dict) -> dict:
    width_m = float(
        design.get("width_m", design.get("diffuser", {}).get("width_m", INLET_DESIGN_WIDTH_M))
    )
    return {
        "capture_area_m2": float(design["A_capture_required_m2"]),
        "post_cowl_area_m2": float(design["post_cowl_area_m2"]),
        "throat_area_m2": float(design["throat_area_actual_m2"]),
        "throat_height_m": float(design["throat_height_m"]),
        "width_m": width_m,
        "diffuser_exit_shape": str(design.get("diffuser", {}).get("exit_shape", "unknown")),
        "diffuser_exit_diameter_m": float(design.get("diffuser", {}).get("exit_diameter_m", np.nan)),
        "forebody_length_m": float(design["forebody_length_m"]),
        "ramp1_length_m": float(design["ramp1_length_m"]),
        "shock2_to_lip_m": float(design["shock2_distance_from_break2_to_lip_m"]),
    }


def _build_combustor_section_summary(combustor_geometry: dict) -> dict:
    return {
        "shape": str(combustor_geometry["cross_section_shape"]),
        "width_m": float(combustor_geometry["width_m"]),
        "height_m": float(combustor_geometry["height_m"]),
        "radius_m": float(combustor_geometry["radius_m"]),
        "diameter_m": float(combustor_geometry["diameter_m"]),
        "area_m2": float(combustor_geometry["cross_section_area_m2"]),
    }


def _estimate_nozzle_geometry(
    throat_area_m2: float,
    nozzle_AR: float,
    combustor_diameter_m: float | None = None,
    half_angle_deg: float = 12.0,
) -> dict:
    if throat_area_m2 <= 0.0:
        raise ValueError("throat_area_m2 must be positive.")
    if nozzle_AR <= 0.0:
        raise ValueError("nozzle_AR must be positive.")

    exit_area_m2 = throat_area_m2 * nozzle_AR
    throat_radius_m = float(np.sqrt(throat_area_m2 / np.pi))
    exit_radius_m = float(np.sqrt(exit_area_m2 / np.pi))
    length_m = (exit_radius_m - throat_radius_m) / max(np.tan(np.radians(half_angle_deg)), 1.0e-12)
    if combustor_diameter_m is not None:
        length_m = max(length_m, 0.25 * float(combustor_diameter_m))

    return {
        "throat_area_m2": float(throat_area_m2),
        "exit_area_m2": float(exit_area_m2),
        "throat_radius_m": float(throat_radius_m),
        "throat_diameter_m": float(2.0 * throat_radius_m),
        "exit_radius_m": float(exit_radius_m),
        "exit_diameter_m": float(2.0 * exit_radius_m),
        "area_ratio": float(nozzle_AR),
        "expansion_state": "geometry_only",
        "length_m": float(max(length_m, 0.0)),
    }


def _compute_diffuser_volume_m3(diffuser: dict | None) -> float:
    if not isinstance(diffuser, dict):
        return 0.0
    xs = np.asarray(diffuser.get("x_stations", []), dtype=float)
    areas = np.asarray(diffuser.get("A_stations", []), dtype=float)
    if xs.size < 2 or areas.size != xs.size:
        return 0.0
    return float(np.trapezoid(areas, xs))


def _compute_nozzle_converging_volume_m3(inlet_area_m2: float, throat_area_m2: float) -> float:
    if inlet_area_m2 <= 0.0 or throat_area_m2 <= 0.0:
        return 0.0
    converging_length_m = float(
        nozzle_design.default_bell_converging_length(inlet_area_m2, throat_area_m2)
    )
    r_in = float(np.sqrt(inlet_area_m2 / np.pi))
    r_th = float(np.sqrt(throat_area_m2 / np.pi))
    return float((np.pi * converging_length_m / 3.0) * (r_in * r_in + r_in * r_th + r_th * r_th))


def _build_geometry_packaging_summary(
    design: dict,
    inlet_geometry: dict,
    combustor_geometry: dict,
    combustor_section: dict,
    nozzle_geometry: dict,
) -> dict:
    diffuser = design.get("diffuser", {})
    total_length_m = (
        float(design["forebody_length_m"])
        + float(design["ramp1_length_m"])
        + float(design["shock2_distance_from_break2_to_lip_m"])
        + float(diffuser.get("length_m", 0.0))
        + float(combustor_geometry["length_m"])
        + float(nozzle_geometry.get("length_m", 0.0))
    )
    max_diameter_m = max(
        float(combustor_section.get("diameter_m", 0.0)),
        float(nozzle_geometry.get("exit_diameter_m", 0.0)),
        float(inlet_geometry.get("diffuser_exit_diameter_m", 0.0)),
    )
    max_width_m = max(
        float(inlet_geometry.get("width_m", 0.0)),
        float(combustor_section.get("width_m", 0.0)),
        float(nozzle_geometry.get("exit_diameter_m", 0.0)),
    )
    max_height_m = max(
        float(design.get("opening_normal_to_ramp2_m", 0.0)),
        float(design.get("post_cowl_height_m", 0.0)),
        float(design.get("throat_height_m", 0.0)),
        float(inlet_geometry.get("diffuser_exit_diameter_m", 0.0)),
        float(combustor_section.get("height_m", 0.0)),
        float(combustor_section.get("diameter_m", 0.0)),
        float(nozzle_geometry.get("throat_diameter_m", 0.0)),
        float(nozzle_geometry.get("exit_diameter_m", 0.0)),
    )
    return {
        "total_length_m": float(total_length_m),
        "max_diameter_m": float(max_diameter_m),
        "max_width_m": float(max_width_m),
        "max_height_m": float(max_height_m),
    }


def build_geometry_summary(design: dict) -> dict:
    """
    Construct a geometry-only engine summary from a frozen inlet design dict.

    This path is intentionally cheaper than `analyze()`: it reuses the inlet
    geometry already built by `build_design()`, sizes the combustor directly
    from user-supplied chamber length and diffuser exit area, and estimates
    nozzle packaging from throat area and commanded nozzle area ratio. No
    pyCycle / nozzle performance solve.
    """
    if not isinstance(design, dict):
        raise TypeError("design must be a dict returned by build_design().")

    nozzle_throat_area = float(design["A_noz_throat"])
    combustor_length_m = float(design.get("combustor_length_m", COMBUSTOR_LENGTH_M_DEFAULT))
    nozzle_AR = float(design.get("nozzle_AR", NOZZLE_AR_DEFAULT))

    combustor_geometry = compute_combustor_geometry(
        combustor_length_m=combustor_length_m,
        design=design,
    )
    combustor_section = _build_combustor_section_summary(combustor_geometry)
    inlet_geometry = _build_inlet_geometry_summary(design)
    nozzle_geometry = _estimate_nozzle_geometry(
        throat_area_m2=nozzle_throat_area,
        nozzle_AR=nozzle_AR,
        combustor_diameter_m=combustor_section["diameter_m"],
    )
    diffuser_volume_m3 = _compute_diffuser_volume_m3(design.get("diffuser"))
    nozzle_converging_volume_m3 = _compute_nozzle_converging_volume_m3(
        inlet_area_m2=float(combustor_geometry["cross_section_area_m2"]),
        throat_area_m2=float(nozzle_throat_area),
    )
    internal_precombustion_volume_m3 = (
        float(combustor_geometry["volume_m3"])
        + diffuser_volume_m3
        + nozzle_converging_volume_m3
    )
    packaging = _build_geometry_packaging_summary(
        design=design,
        inlet_geometry=inlet_geometry,
        combustor_geometry=combustor_geometry,
        combustor_section=combustor_section,
        nozzle_geometry=nozzle_geometry,
    )

    summary = dict(design)
    summary.update({
        "inlet_geometry": inlet_geometry,
        "combustor_section": combustor_section,
        "combustor_geometry": combustor_geometry,
        "nozzle_geometry": nozzle_geometry,
        "geometry": packaging,
        "capture_area_m2": float(inlet_geometry["capture_area_m2"]),
        "throat_area_m2": float(inlet_geometry["throat_area_m2"]),
        "total_length_m": float(packaging["total_length_m"]),
        "max_diameter_m": float(packaging["max_diameter_m"]),
        "max_width_m": float(packaging["max_width_m"]),
        "max_height_m": float(packaging["max_height_m"]),
        "combustor_volume_m3": float(combustor_geometry["volume_m3"]),
        "diffuser_volume_m3": float(diffuser_volume_m3),
        "nozzle_converging_volume_m3": float(nozzle_converging_volume_m3),
        "internal_precombustion_volume_m3": float(internal_precombustion_volume_m3),
    })
    return summary


# ---------------------------------------------------------------------------
# Single-point analysis
# ---------------------------------------------------------------------------

def analyze(
    M0:           float,
    altitude_m:   float,
    phi:          float,
    alpha_deg:    float | None = None,
    ramp_angles:  list  | None = None,
    combustor_length_m: float | None = None,
    design = None,
    verbose:      bool         = False,
) -> dict:
    """
    Run the ramjet cycle at a single flight condition.

    Parameters
    ----------
    M0           : Freestream Mach number
    altitude_m   : Geometric altitude [m]
    phi          : Equivalence ratio
    alpha_deg    : Flight angle of attack for off-design inlet evaluation [deg]
    ramp_angles  : Ramp deflection angles [deg] (None -> config default)
    verbose      : Print cycle table

    Returns
    -------
    dict with keys:
        M0, altitude, phi,
        Isp [s], F_sp [N*s/kg_air], thrust [N],
        mdot_air [kg/s], mdot_fuel [kg/s],
        eta_pt, choked,
        T_stations, Tt_stations, M_stations, Pt_stations
    """
    if design is None:
        design = _get_inlet_design()  # baseline fallback
    eval_alpha_deg = float(
        design.get('alpha_deg', INLET_DESIGN_ALPHA_DEG)
        if alpha_deg is None else alpha_deg
    )

    # Freestream conditions
    atm  = Atmosphere(altitude_m)
    T0   = float(atm.temperature[0])   # K
    P0   = float(atm.pressure[0])      # Pa
    rho0 = float(atm.density[0])       # kg/m^3

    V0      = M0 * np.sqrt(AIR_GAM * AIR_R * T0)
    # Capture area from the frozen 2-ramp inlet geometry (402inlet2.py).
    # design['A_capture_required_m2'] is the geometric opening sized at the
    # design point; mass flow at off-design scales as rho0*V0*A_capture.
    A_capture_m2 = float(design['A_capture_required_m2'])
    W_kgs   = rho0 * V0 * A_capture_m2         # kg/s
    W_lbms  = W_kgs * KG2LBM                   # lbm/s

    combustor_length_m_eff = (float(combustor_length_m) if combustor_length_m is not None
                              else float(design.get("combustor_length_m",
                                                    COMBUSTOR_LENGTH_M_DEFAULT)))
    # Chamber cross-section comes from the diffuser exit area and length is
    # user-supplied, so the geometry is fully determined before pyCycle runs
    # (no dependency on nozzle throat). burner.area_ratio uses A_cc / A_throat_inlet.
    combustor_geometry = compute_combustor_geometry(
        combustor_length_m=combustor_length_m_eff,
        design=design,
    )
    combustor_area_ratio = (
        combustor_geometry["cross_section_area_m2"] / design["throat_area_actual_m2"]
    )

    prob = _get_problem()

    # ── Seed FlightConditions balance with isentropic initial guess ──────────
    _seed_fc_initial_guess(prob, M0, T0, P0)

    # ── Flight conditions ────────────────────────────────────────────────────
    prob.set_val('fc.alt', altitude_m * M2FT, units='ft')
    prob.set_val('fc.MN',  M0)
    prob.set_val('fc.W',   W_lbms, units='lbm/s')
    prob.set_val("burner.area_ratio", 1.0)

    # ── Inlet capability + φ envelope (cascade-only) ─────────────────────────
    capability = _get_capability(design, M0, altitude_m, eval_alpha_deg)
    if not capability.get('ok', False):
        raise RuntimeError(
            f"Inlet capability failed at M0={M0}, alt={altitude_m}: "
            f"{capability.get('reason','unknown')}"
        )
    A_noz_throat = float(design['A_noz_throat'])
    envelope_info = _solve_phi_envelope(
        capability=capability,
        A_noz_throat=A_noz_throat,
        W_air=float(W_kgs),
        Tt4_max=TT4_MAX_K,
        phi_request=phi,
        thermo=get_thermo(),
        M4_max=M4_MAX,
    )
    if not envelope_info.get('ok', False):
        raise RuntimeError(
            f"φ-envelope solve failed at M0={M0}, alt={altitude_m}: "
            f"{envelope_info.get('reason', 'unknown')}"
        )
    phi_effective = float(envelope_info['phi_cap'])
    FAR = phi_effective * F_STOICH_JP10
    prob.set_val("burner.Fl_I:FAR", FAR)

    # Inlet inputs: ram_recovery and MN3 come straight from cascade+diffuser.
    Pt0 = float(capability['Pt0'])
    ram_recovery = float(capability['Pt3_deliverable']) / max(Pt0, 1.0e-12)
    inlet_MN = float(np.clip(capability['M3_diffuser_exit'], 0.05, 0.95))
    prob.set_val('inlet.ram_recovery', ram_recovery)
    prob.set_val('inlet.MN',           inlet_MN)
    prob.run_model()

    phi_inlet_limit = float(envelope_info.get('phi_inlet_limit', float('inf')))
    unstart_flag = 0.0
    # A finite near-zero inlet limit means the frozen inlet cannot support even
    # the minimum combustor loading without expulsion/swallow issues. By
    # contrast, +inf means the inlet cap never binds, which is a healthy case.
    if np.isfinite(phi_inlet_limit) and phi_inlet_limit < 0.05:
        unstart_flag = 1.0

    inlet_inputs = {
        'ram_recovery':   ram_recovery,
        'MN_exit':        inlet_MN,
        'Pt_after_cowl':  float(capability['Pt_after_cowl']),
        'M_after_cowl':   float(capability['M3_cascade_inlet']),
        'Tt_after_cowl':  float(capability['Tt0']),
        'unstart_flag':   unstart_flag,
        'status':         'cascade',
    }
    ram_closure = {
        'envelope':     envelope_info,
        'phi_clipped':  phi_effective,
        'capability':   capability,
        'inlet_inputs': inlet_inputs,
    }

    def _K(p):  return float(prob.get_val(p, units='degK')[0])
    def _Pa(p): return float(prob.get_val(p, units='Pa')[0])
    def _mn(p): return float(prob.get_val(p)[0])

    # ── Nozzle & thrust: frozen-geometry override ─────────────────────────
    # pyCycle runs the CD nozzle in design mode, sizing A_exit each call so
    # that Ps_exit = ambient (perfect expansion at every off-design point).
    # We don't want that — the physical nozzle has fixed throat and exit area
    # set at the design point.  Fg is recomputed here at frozen A_exit, with
    # the over/under-expansion pressure term (Ps_exit − P0)·A_exit included.
    # ram_drag comes from pyCycle (inlet momentum deficit is unaffected).
    Pt4_Pa = _Pa('burner.Fl_O:tot:P')
    Tt4_K  = _K('burner.Fl_O:tot:T')
    gamma4_K = float(prob.get_val('burner.Fl_O:stat:gamma')[0])

    Wfuel  = float(prob.get_val('perf.Wfuel', units='kg/s')[0])
    ram_drag_N = float(prob.get_val('perf.ram_drag', units='N')[0])

    throat_area_m2 = float(design['A_noz_throat'])
    exit_area_m2   = float(design['A_noz_exit'])

    # Supersonic exit Mach at frozen area ratio, with a gamma refinement pass.
    mdot_tot_kgs = W_kgs + Wfuel
    area_ratio_frozen = exit_area_m2 / throat_area_m2
    thermo = get_thermo()
    if Pt4_Pa <= P0 or area_ratio_frozen < 1.0 + 1.0e-9:
        # Engine cannot sustain supersonic exit: either no pressure head
        # (Pt4 ≤ ambient) or a degenerate frozen geometry.  Treat thrust as
        # zero and net force as pure ram drag.
        M9     = 1.0
        T9     = Tt4_K / (1.0 + 0.5 * (gamma4_K - 1.0))
        gamma9 = gamma4_K
        R9     = float(thermo.R(T9, phi_effective, max(P0, 1.0)))
        Ps9    = P0
        V9     = 0.0
        Fg_N   = 0.0
        Fn_N   = -ram_drag_N
    else:
        M9 = float(_inlet2.invert_area_mach_ratio_supersonic(area_ratio_frozen,
                                                              gamma=gamma4_K))
        T9 = Tt4_K / (1.0 + 0.5 * (gamma4_K - 1.0) * M9 * M9)
        gamma9 = float(thermo.gamma(T9, phi_effective, P0))
        R9     = float(thermo.R    (T9, phi_effective, P0))
        M9 = float(_inlet2.invert_area_mach_ratio_supersonic(area_ratio_frozen,
                                                              gamma=gamma9))
        T9 = Tt4_K / (1.0 + 0.5 * (gamma9 - 1.0) * M9 * M9)
        Ps9 = Pt4_Pa / (1.0 + 0.5 * (gamma9 - 1.0) * M9 * M9) ** (gamma9 / (gamma9 - 1.0))
        V9_ideal = M9 * np.sqrt(gamma9 * R9 * T9)
        V9 = ETA_NOZZLE_CV * V9_ideal
        Fg_N = mdot_tot_kgs * V9 + (Ps9 - P0) * exit_area_m2
        Fn_N = Fg_N - ram_drag_N
    F_sp = Fn_N / max(W_kgs, 1.0e-12)
    Isp  = Fn_N / max(Wfuel * G0, 1.0e-12)

    nozzle_throat = {
        "area":  throat_area_m2,
        "T":     _K('nozz.Throat:stat:T'),
        "P":     _Pa('nozz.Throat:stat:P'),
        "M":     _mn('nozz.Throat:stat:MN'),
        "Pt":    Pt4_Pa,
        "Tt":    Tt4_K,
        "gamma": float(prob.get_val('nozz.Throat:stat:gamma')[0]),
    }
    if Ps9 > P0 * 1.02:
        expansion_state = "under-expanded"
    elif Ps9 < P0 * 0.98:
        expansion_state = "over-expanded"
    else:
        expansion_state = "ideally expanded"
    nozzle_exit = {
        "area":  exit_area_m2,
        "T":     float(T9),
        "P":     float(Ps9),
        "M":     float(M9),
        "Pt":    _Pa('nozz.Fl_O:tot:P'),
        "Tt":    _K('nozz.Fl_O:tot:T'),
        "gamma": float(gamma9),
    }
    nozzle_perf = {
        "F_cruise":        Fg_N,
        "area_ratio":      (exit_area_m2 / throat_area_m2) if throat_area_m2 > 0.0 else float('nan'),
        "expansion_state": expansion_state,
    }

    diffuser_entry_station = None
    try:
        diffuser_entry_station = _build_air_state_from_totals(
            M=ram_closure['inlet_inputs']['M_after_cowl'],
            Tt=ram_closure['inlet_inputs']['Tt_after_cowl'],
            Pt=ram_closure['inlet_inputs']['Pt_after_cowl'],
        )
    except Exception:
        diffuser_entry_station = None

    choked = False
    try:
        choked = bool(_mn('burner.choked') > 0.5)
    except Exception:
        pass

    combustor_section = _build_combustor_section_summary(combustor_geometry)
    inlet_geometry = _build_inlet_geometry_summary(design)
    nozzle_length_m = (
        float(np.sqrt(float(nozzle_exit["area"]) / np.pi))
        - float(np.sqrt(float(nozzle_throat["area"]) / np.pi))
    ) / max(np.tan(np.radians(12.0)), 1.0e-12)
    nozzle_geometry = {
        "throat_area_m2": float(nozzle_throat["area"]),
        "exit_area_m2": float(nozzle_exit["area"]),
        "throat_radius_m": float(np.sqrt(float(nozzle_throat["area"]) / np.pi)),
        "throat_diameter_m": float(np.sqrt(4.0 * float(nozzle_throat["area"]) / np.pi)),
        "exit_radius_m": float(np.sqrt(float(nozzle_exit["area"]) / np.pi)),
        "exit_diameter_m": float(np.sqrt(4.0 * float(nozzle_exit["area"]) / np.pi)),
        "area_ratio": float(nozzle_perf["area_ratio"]),
        "expansion_state": nozzle_perf["expansion_state"],
        "length_m": float(max(nozzle_length_m, 0.3)),
    }
    diffuser_volume_m3 = _compute_diffuser_volume_m3(design.get("diffuser"))
    nozzle_converging_volume_m3 = _compute_nozzle_converging_volume_m3(
        inlet_area_m2=float(combustor_geometry["cross_section_area_m2"]),
        throat_area_m2=float(nozzle_throat["area"]),
    )
    internal_precombustion_volume_m3 = (
        float(combustor_geometry["volume_m3"])
        + diffuser_volume_m3
        + nozzle_converging_volume_m3
    )
    packaging = _build_geometry_packaging_summary(
        design=design,
        inlet_geometry=inlet_geometry,
        combustor_geometry=combustor_geometry,
        combustor_section=combustor_section,
        nozzle_geometry=nozzle_geometry,
    )


    result = dict(
        M0=M0, altitude=altitude_m, phi=phi_effective,
        alpha_deg=eval_alpha_deg,
        design_alpha_deg=float(design.get('alpha_deg', INLET_DESIGN_ALPHA_DEG)),
        phi_request=float(phi),
        phi_effective=phi_effective,
        envelope=envelope_info,
        capability=capability,
        Isp=Isp, F_sp=F_sp, thrust=Fn_N,
        mdot_air=W_kgs, mdot_fuel=Wfuel,
        eta_pt=ram_closure['inlet_inputs']['ram_recovery'],
        choked=choked,
        unstart_flag=ram_closure['inlet_inputs']['unstart_flag'],
        T_stations={
            0: _K('fc.Fl_O:stat:T'),
            2: (float(diffuser_entry_station.T)
                if diffuser_entry_station is not None else np.nan),
            3: _K('inlet.Fl_O:stat:T'),
            4: _K('burner.Fl_O:stat:T'),
            9: float(nozzle_exit['T']),
        },
        Tt_stations={
            0: _K('fc.Fl_O:tot:T'),
            2: (float(diffuser_entry_station.Tt)
                if diffuser_entry_station is not None else np.nan),
            3: _K('inlet.Fl_O:tot:T'),
            4: _K('burner.Fl_O:tot:T'),
            9: float(nozzle_exit['Tt']),
        },
        P_stations={
            0: _Pa('fc.Fl_O:stat:P'),
            2: (float(diffuser_entry_station.P)
                if diffuser_entry_station is not None else np.nan),
            3: _Pa('inlet.Fl_O:stat:P'),
            4: _Pa('burner.Fl_O:stat:P'),
            9: float(nozzle_exit['P']),
        },
        M_stations={
            0: M0,
            2: (float(diffuser_entry_station.M)
                if diffuser_entry_station is not None else np.nan),
            3: _mn('inlet.Fl_O:stat:MN'),
            4: _mn('burner.Fl_O:stat:MN'),
            9: float(nozzle_exit['M']),
        },
        Pt_stations={
            0: _Pa('fc.Fl_O:tot:P'),
            2: (float(diffuser_entry_station.Pt)
                if diffuser_entry_station is not None else np.nan),
            3: _Pa('inlet.Fl_O:tot:P'),
            4: _Pa('burner.Fl_O:tot:P'),
            9: float(nozzle_exit['Pt']),
        },
        nozzle_throat_area=float(nozzle_throat['area']),
        nozzle_exit_area=float(nozzle_exit['area']),
        nozzle_area_ratio=float(nozzle_perf['area_ratio']),
        nozzle_expansion=nozzle_perf['expansion_state'],
        inlet_geometry=inlet_geometry,
        combustor_section=combustor_section,
        combustor_geometry=combustor_geometry,
        nozzle_geometry=nozzle_geometry,
        geometry=packaging,
        diffuser_volume_m3=float(diffuser_volume_m3),
        nozzle_converging_volume_m3=float(nozzle_converging_volume_m3),
        internal_precombustion_volume_m3=float(internal_precombustion_volume_m3),

    )

    if verbose:
        _print_cycle(result)

    return result


# ---------------------------------------------------------------------------
# Mach sweep
# ---------------------------------------------------------------------------

def mach_sweep(mach_range, altitude_m, phi, **kwargs):
    """
    Run analyze() over a Mach array.  Returns None for failed points.
    """
    results = []
    for M in mach_range:
        try:
            results.append(analyze(M, altitude_m, phi, **kwargs))
        except Exception as e:
            print(f'  [warn] M={M:.2f} failed: {e}')
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_cycle(r):
    inlet_geom = r.get('inlet_geometry', {})
    combustor_section = r.get('combustor_section', {})
    combustor_geom = r.get('combustor_geometry')
    nozzle_geom = r.get('nozzle_geometry', {})

    print(f"\n{'='*70}")
    print(f"  RAM  |  M0={r['M0']:.2f}  "
          f"|  alt={r['altitude']/1e3:.1f} km  "
          f"|  phi={r['phi']:.2f}")
    print(f"{'='*70}")
    print("  Station states")
    print(f"  {'Stn':>8}  {'M':>7}  {'Ts [K]':>8}  {'Tt [K]':>8}  "
          f"{'Ps [kPa]':>10}  {'Pt [kPa]':>10}")
    print(f"  {'-'*64}")
    lbl = {0: 'Free', 2: 'Post-cowl', 3: 'Comb.in', 4: 'Comb.out', 9: 'Nozzle'}
    for s in (0, 2, 3, 4, 9):
        print(f"  {lbl[s]:>8}  {r['M_stations'][s]:>7.3f}  "
              f"{r['T_stations'][s]:>8.1f}  {r['Tt_stations'][s]:>8.1f}  "
              f"{r['P_stations'][s]/1e3:>10.2f}  "
              f"{r['Pt_stations'][s]/1e3:>10.2f}")
    print()
    print("  Inlet geometry")
    print(f"  capture area     = {inlet_geom.get('capture_area_m2', float('nan')):>8.5f} m^2")
    print(f"  post-cowl area   = {inlet_geom.get('post_cowl_area_m2', float('nan')):>8.5f} m^2")
    print(f"  throat area      = {inlet_geom.get('throat_area_m2', float('nan')):>8.5f} m^2")
    print(f"  throat height    = {inlet_geom.get('throat_height_m', float('nan')):>8.5f} m")
    print(f"  width            = {inlet_geom.get('width_m', float('nan')):>8.5f} m")
    print(f"  diffuser exit    = {inlet_geom.get('diffuser_exit_shape', 'n/a')}")
    print(f"  diffuser exit dia= {inlet_geom.get('diffuser_exit_diameter_m', float('nan')):>8.5f} m")
    print(f"  diffuser volume  = {r.get('diffuser_volume_m3', float('nan')):>8.5f} m^3")
    print(f"  forebody length  = {inlet_geom.get('forebody_length_m', float('nan')):>8.5f} m")
    print(f"  ramp1 length     = {inlet_geom.get('ramp1_length_m', float('nan')):>8.5f} m")
    print(f"  break2->lip dist = {inlet_geom.get('shock2_to_lip_m', float('nan')):>8.5f} m")
    print()
    print("  Combustor geometry")
    print(f"  section shape    = {combustor_section.get('shape', 'n/a')}")
    print(f"  section width    = {combustor_section.get('width_m', float('nan')):>8.5f} m")
    print(f"  section height   = {combustor_section.get('height_m', float('nan')):>8.5f} m")
    print(f"  section radius   = {combustor_section.get('radius_m', float('nan')):>8.5f} m")
    print(f"  section diameter = {combustor_section.get('diameter_m', float('nan')):>8.5f} m")
    print(f"  section area     = {combustor_section.get('area_m2', float('nan')):>8.5f} m^2")
    if combustor_geom is not None:
        print(f"  chamber length   = {combustor_geom.get('length_m', float('nan')):>8.5f} m")
        print(f"  chamber volume   = {combustor_geom.get('volume_m3', float('nan')):>8.5f} m^3")
    print()
    print("  Nozzle geometry")
    print(f"  throat area At   = {nozzle_geom.get('throat_area_m2', float('nan')):>8.5f} m^2")
    print(f"  throat diameter  = {nozzle_geom.get('throat_diameter_m', float('nan')):>8.5f} m")
    print(f"  exit area Ae     = {nozzle_geom.get('exit_area_m2', float('nan')):>8.5f} m^2")
    print(f"  exit diameter    = {nozzle_geom.get('exit_diameter_m', float('nan')):>8.5f} m")
    print(f"  area ratio Ae/At = {nozzle_geom.get('area_ratio', float('nan')):>8.5f}")
    print(f"  expansion state  = {nozzle_geom.get('expansion_state', 'n/a')}")
    print(f"  conv. sec volume = {r.get('nozzle_converging_volume_m3', float('nan')):>8.5f} m^3")
    print(f"  sum vol (cc+diff+conv) = {r.get('internal_precombustion_volume_m3', float('nan')):>8.5f} m^3")
    print()
    print("  Performance")
    print(f"  Isp       = {r['Isp']:>8.1f} s")
    print(f"  F_sp      = {r['F_sp']:>8.1f} N*s/kg_air")
    print(f"  Thrust    = {r['thrust']/1e3:>8.2f} kN ")
    print(f"  mdot_air  = {r['mdot_air']:>7.3f} kg/s")
    print(f"  mdot_fuel = {r['mdot_fuel']*1e3:>7.3f} g/s")
    print(f"  eta_pt    = {r['eta_pt']:.4f}   choked={r['choked']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':


    from pyc_config import PHI_DEFAULT


    print("\n--- Design Point Performance")
    design = _get_inlet_design()
    design_result = analyze(
        M0=INLET_DESIGN_M0,
        altitude_m=INLET_DESIGN_ALT_M,
        phi=PHI_DEFAULT,
        alpha_deg=INLET_DESIGN_ALPHA_DEG,
        verbose=True,
    )

    print("\n--- Design Point Flowpath Plot")
    try:
        import os
        import plots_pycycle
        plots_pycycle.OUTDIR = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(plots_pycycle.OUTDIR, exist_ok=True)
        plots_pycycle.fig_flowpath(
            design,
            design_result,
            combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
        )
    except Exception as exc:
        print(f"  [warn] flowpath plot generation failed: {type(exc).__name__}: {exc}")



    '''
    
    BEFORE RUNNING IN VIRTUAL ENVIRONMENTS OR ROOT DIRECTORY INTERPRETER PYCYCLE MUST BE MODIFIED
    
#    - pycycle\thermo\cea\props_rhs.py
    change outputs['rhs_P'][num_element] = inputs['n_moles'] to inputs['n_moles'][0]
    
#    - pycycle\ thermo\cea\props_calcs.py
    change both n_moles = inputs['n_moles'] assignments to inputs['n_moles'][0]
    
    '''



#mach 4-5
#altitude 15-24km
