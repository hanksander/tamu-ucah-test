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

import warnings
warnings.filterwarnings('ignore')

import importlib.util
import numpy as np
import openmdao.api as om
from ambiance import Atmosphere

from gas_dynamics import FlowState, isentropic_P, isentropic_T
from nozzle import compute_nozzle
from combustor import combustor_face_response
from pyc_config import (
    F_STOICH_JP10, ETA_COMBUSTOR,
    ETA_NOZZLE_CV, ISOLATOR_PT_RECOVERY,
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M,
    INLET_DESIGN_ALPHA_DEG, INLET_DESIGN_LEADING_EDGE_ANGLE_DEG,
    INLET_DESIGN_MDOT_KGS, INLET_DESIGN_WIDTH_M,
    INLET_FOREBODY_SEP_MARGIN, INLET_RAMP_SEP_MARGIN,
    INLET_KANTROWITZ_MARGIN, INLET_SHOCK_FOCUS_FACTOR,
    NOZZLE_TYPE,
    TT4_MAX_K, M4_MAX, PS3_BIAS, COMBUSTOR_L_STAR_DEFAULT,
    DIFFUSER_AREA_RATIO, NOZZLE_AR_DEFAULT,
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
    diffuser_area_ratio=None, combustor_L_star=None, nozzle_AR=None,
):
    """Build an inlet+geometry design dict. Any arg left as None falls
    back to the pyc_config default."""
    def pick(x, d): return d if x is None else x
    alpha_deg_eff = pick(alpha_deg, INLET_DESIGN_ALPHA_DEG)
    design = _inlet2.design_2ramp_shock_matched_inlet(
        M0=pick(M0, INLET_DESIGN_M0),
        altitude_m=pick(altitude_m, INLET_DESIGN_ALT_M),
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
    # attach the extras not consumed by the 2-ramp solver. alpha_deg is stored
    # here so the RAM closure evaluates at the same α used to design geometry.
    design['alpha_deg']           = float(alpha_deg_eff)
    design['diffuser_area_ratio'] = pick(diffuser_area_ratio, DIFFUSER_AREA_RATIO)
    design['combustor_L_star']    = pick(combustor_L_star, COMBUSTOR_L_STAR_DEFAULT)
    design['nozzle_AR']           = pick(nozzle_AR, NOZZLE_AR_DEFAULT)
    return design


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


def _ram_inlet_case_to_cycle_inputs(case, M0, Pt0, iso_pt=ISOLATOR_PT_RECOVERY):
    """
    Map a frozen-geometry inlet evaluation to the pyCycle inlet inputs used by
    the RAM outer closure.
    """
    if case.get('success', False):
        return {
            'ram_recovery': float(case['pt_frac_after_terminal_shock']) * iso_pt,
            'MN_exit': float(np.clip(case['M_at_combustor_face'], 0.05, 0.95)),
            'Pt_after_cowl': float(case['Pt_after_cowl']),
            'M_after_cowl': float(case['M_after_cowl_shock']),
            'Tt_after_cowl': float(case['Tt0']),
            'x_shock': float(case['x_terminal_shock']),
            'unstart_flag': 0.0,
            'status': str(case.get('status', 'normal')),
            'terminal': case.get('terminal'),
        }

    status = case.get('status', 'unknown')
    term = case.get('terminal', {})
    Pt_ac = float(case.get('Pt_after_cowl', term.get('Pt_after_shock', 1.0)))

    if status == 'expelled':
        _, _, _, _, pt_ratio_bow = _inlet2.normal_shock(M0)
        ram_recovery = float(pt_ratio_bow) * iso_pt
        inlet_mn = float(np.clip(term.get('M_exit', term.get('M_sub', 0.05)), 0.05, 0.95))
        unstart_flag = +1.0
        x_shock = float(term.get('x_s', 0.0))
    elif status == 'swallowed':
        Pt_after_shock = float(term.get('Pt_after_shock', Pt_ac))
        ram_recovery = (Pt_after_shock / max(Pt0, 1.0e-12)) * iso_pt
        inlet_mn = float(np.clip(term.get('M_exit', term.get('M_sub', 0.95)), 0.05, 0.95))
        unstart_flag = -1.0
        x_shock = float(term.get('x_s', 0.0))
    else:
        ram_recovery = 0.05 * iso_pt
        inlet_mn = 0.3
        unstart_flag = +1.0
        x_shock = 0.0

    return {
        'ram_recovery': float(ram_recovery),
        'MN_exit': float(inlet_mn),
        'Pt_after_cowl': Pt_ac,
        'M_after_cowl': float(case.get('M_after_cowl_shock', np.nan)),
        'Tt_after_cowl': float(term.get('Tt0', np.nan)),
        'x_shock': x_shock,
        'unstart_flag': unstart_flag,
        'status': str(status),
        'terminal': term,
    }


def _evaluate_ram_outer_residual(prob, design, M0, altitude_m, p_back):
    """
    Run one RAM trial at a specified combustor-face back pressure.
    """
    case = _inlet2.evaluate_fixed_geometry_at_condition(
        design,
        M0=M0,
        altitude_m=altitude_m,
        alpha_deg=float(design.get('alpha_deg', INLET_DESIGN_ALPHA_DEG)),
        p_back=float(p_back),
    )

    flight = _inlet2.freestream_state(M0, altitude_m)
    inlet_inputs = _ram_inlet_case_to_cycle_inputs(case, M0, float(flight['pt0']))
    inlet_inputs['Tt_after_cowl'] = float(flight['Tt0'])

    prob.set_val('inlet.ram_recovery', inlet_inputs['ram_recovery'])
    prob.set_val('inlet.MN', inlet_inputs['MN_exit'])
    prob.run_model()

    cycle_ps_back = float(prob.get_val('inlet.Fl_O:stat:P', units='Pa')[0])
    return {
        'residual': cycle_ps_back - float(p_back),
        'cycle_ps_back': cycle_ps_back,
        'case': case,
        'inlet_inputs': inlet_inputs,
    }


# ---------------------------------------------------------------------------
# Inlet-limited RAM closure (v2)
# ---------------------------------------------------------------------------
#
# v1 (_solve_ram_outer_closure) iterates on p_back until pyCycle's inlet
# produces a Ps that matches what the 402 inlet model says should stand at
# that terminal-shock station. Every bisection step runs pyCycle — typically
# 5–12 calls per flight point.
#
# v2 replaces the bisection with three steps, entirely outside pyCycle:
#   (a) inlet capability once per (M0, alt, α): Ps_max/min + PCHIP inverse
#       splines for (x_s, Pt_frac, M_exit) of Ps, from compute_inlet_capability.
#   (b) φ envelope: sweep φ at 8 points through combustor_face_response, find
#       the three caps (inlet Ps_max, M4_max, Tt4_max) via monotonic PCHIP
#       inversion, soft-min to φ_cap.
#   (c) one pyCycle call at the clipped φ with a single inlet-input eval at
#       the matched p_back. O(1) pyCycle calls per flight point vs O(12).
#
# The SOFTMIN gives SLSQP a smooth gradient across cap transitions — hard
# min would create a kink that the optimizer can't handle.

from scipy.interpolate import PchipInterpolator as _Pchip
from scipy.optimize import brentq as _brentq

_SOFTMIN_K = 20.0  # larger → sharper soft-min (closer to true min)

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
    tolerance of physics-level cache keys so tiny FD perturbations still hit.
    """
    return (
        round(float(design.get('theta_fore_deg', 0.0)), 3),
        round(float(design.get('theta1_deg', 0.0)), 3),
        round(float(design.get('theta2_deg', 0.0)), 3),
        round(float(design.get('theta_cowl_deg', 0.0)), 3),
        round(float(design.get('A_capture_required_m2', 0.0)), 6),
        round(float(design.get('throat_area_actual_m2', 0.0)), 6),
        round(float(design.get('diffuser_area_ratio', 0.0)), 4),
    )


def _get_capability(design, M0, altitude_m, alpha_deg):
    """Memoized inlet capability lookup. Keyed on rounded flight state +
    inlet geometry digest so brentq re-entries and FD perturbations reuse.
    """
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


def _solve_phi_envelope(capability, area_ratio, Tt4_max, phi_request, thermo,
                         mdot_freestream: float | None = None,
                         A3: float | None = None,
                         M4_max: float = 0.98,
                         phi_grid: np.ndarray | None = None,
                         ps3_bias: float = PS3_BIAS):
    """Inlet-limited φ-envelope.

    Parameters
    ----------
    capability       : dict from compute_inlet_capability
    area_ratio       : combustor A_exit/A_entrance (≈1 for constant-area)
    Tt4_max          : material-limit on burner-exit Tt [K]
    phi_request      : commanded equivalence ratio
    thermo           : JP10Thermo singleton
    mdot_freestream  : (accepted for signature compatibility; unused)
    A3               : (accepted for signature compatibility; unused)
    M4_max           : thermal-choke margin
    phi_grid         : optional explicit φ sample grid (default 8 points)
    ps3_bias         : where along [Ps_min, Ps_max] to place the terminal
                       shock. 0 = strong/exit, 1 = weak/throat. Default
                       PS3_BIAS from pyc_config (~0.7, weak shock).

    Physics
    -------
    Stage A gave us a one-parameter family of post-terminal-shock combustor-
    face states:

        for Ps3 ∈ [Ps_min, Ps_max]:
            M3  = M_exit_of_Ps(Ps3)                 # subsonic
            Pt3 = Pt_after_cowl · Pt_frac_of_Ps(Ps3)
            Tt3 = Tt0                               # adiabatic

    Mass flow is conserved through the diffuser *identically* — any Ps3
    in [Ps_min, Ps_max] passes the same mdot by construction of the
    capability splines. The shock position is not set by a standalone
    station-3 balance; it is set by downstream (combustor + nozzle)
    demand, which pyCycle resolves internally. So outside pyCycle we
    pick Ps3 pragmatically along the capability range via PS3_BIAS and
    let pyCycle's own balance converge at that back-pressure.

    phi_inlet_limit is therefore +∞ unconditionally here — we can't
    detect inlet unstart from this standalone closure. Unstart shows up
    downstream as a pyCycle convergence failure or a violated Tt4/M4
    constraint in the optimizer.
    """
    if not capability.get('ok', False):
        return {
            'ok': False,
            'reason': capability.get('reason', 'capability not available'),
            'phi_cap': 0.0,
        }

    Ps_min        = capability['Ps_min']
    Ps_max        = capability['Ps_max']
    Pt_after_cowl = capability['Pt_after_cowl']
    Tt0           = capability['Tt0']
    M_exit_of_Ps  = capability['M_exit_of_Ps']
    Pt_frac_of_Ps = capability['Pt_frac_of_Ps']

    # ── Stage B1: pick Ps3 along the capability range ──────────────────────
    # Mass-flow balance can't determine Ps3 (invariant through the diffuser),
    # so we bias along [Ps_min, Ps_max]. PS3_BIAS=0.7 places the shock near
    # the throat (weak terminal shock, good recovery, typical started mode).
    ps3_bias = float(np.clip(ps3_bias, 0.0, 1.0))
    Ps3 = float(Ps_min + ps3_bias * (Ps_max - Ps_min))

    M3  = float(M_exit_of_Ps(Ps3))
    Pt3 = float(Pt_after_cowl) * float(Pt_frac_of_Ps(Ps3))
    Tt3 = float(Tt0)

    # ── Stage B2: sweep φ through the combustor at this fixed state 3 ──────
    if phi_grid is None:
        phi_top = max(0.95, float(phi_request) + 0.1)
        phi_grid = np.linspace(0.05, phi_top, 8)
    else:
        phi_grid = np.asarray(phi_grid, dtype=float)

    Tt4_arr = np.empty_like(phi_grid)
    M4_arr  = np.empty_like(phi_grid)
    Ps4_arr = np.empty_like(phi_grid)
    Pt4_arr = np.empty_like(phi_grid)

    for i, phi in enumerate(phi_grid):
        face = combustor_face_response(
            Pt3=Pt3, Tt3=Tt3, M3=M3, phi=float(phi),
            thermo=thermo, area_ratio=area_ratio,
        )
        Tt4_arr[i] = face['Tt4']
        M4_arr[i]  = face['M4']
        Ps4_arr[i] = face['Ps4']
        Pt4_arr[i] = face['Pt4']

    # ── Stage B3: invert Tt4(φ) and M4(φ) for caps ─────────────────────────
    # Both Tt4 and M4 are monotone increasing in φ for subsonic Rayleigh.
    # Semantics:
    #   target > max(obs) → cap never binds within the grid → +∞ (no cap)
    #   target < min(obs) → target already violated at the lowest sampled φ
    #                       → cap at that lowest φ (floor)
    def _phi_at_obs_target(obs_arr, target):
        x = np.asarray(obs_arr, dtype=float)
        y = np.asarray(phi_grid, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if x.size < 2:
            return float('nan')
        order = np.argsort(y)        # sort by φ ascending
        x = x[order]; y = y[order]
        # Enforce strict monotone increasing in x for PCHIP.
        keep = np.concatenate(([True], np.diff(x) > 0.0))
        x = x[keep]; y = y[keep]
        if x.size < 2:
            return float('nan')
        if target >= x[-1]:
            return float('inf')
        if target <= x[0]:
            return float(y[0])
        inv = _Pchip(x, y, extrapolate=False)
        return float(inv(target))

    phi_Tt4   = _phi_at_obs_target(Tt4_arr, Tt4_max)
    phi_choke = _phi_at_obs_target(M4_arr,  M4_max)

    # Inlet-expulsion cap is inactive in this standalone closure — unstart
    # can't be detected without the downstream pyCycle balance. Treat as
    # never-binding (+∞); downstream solves and optimizer constraints pick
    # up true violations.
    phi_inlet_limit = float('inf')

    # ── Stage C: soft-min over caps and phi_request ────────────────────────
    phi_caps = np.array([float(phi_request), phi_Tt4, phi_choke, phi_inlet_limit],
                        dtype=float)
    phi_cap  = _softmin(phi_caps, k=_SOFTMIN_K)
    phi_cap  = float(max(0.02, phi_cap))

    face_cap = combustor_face_response(
        Pt3=Pt3, Tt3=Tt3, M3=M3, phi=phi_cap,
        thermo=thermo, area_ratio=area_ratio,
    )

    return {
        'ok':              True,
        'phi_inlet_limit': phi_inlet_limit,
        'phi_choke':       phi_choke,
        'phi_Tt4':         phi_Tt4,
        'phi_request':     float(phi_request),
        'phi_cap':         phi_cap,
        'Ps3':             float(Ps3),
        'M3':              float(M3),
        'Pt3':             float(Pt3),
        'Tt3':             float(Tt3),
        'Ps4':             float(face_cap['Ps4']),
        'M4':              float(face_cap['M4']),
        'Tt4':             float(face_cap['Tt4']),
        'Pt4':             float(face_cap['Pt4']),
        'ps3_bias':        float(ps3_bias),
        'Ps_min':          float(Ps_min),
        'Ps_max':          float(Ps_max),
        'grid_phi':        phi_grid,
        'Tt4_arr':         Tt4_arr,
        'M4_arr':          M4_arr,
        'Ps4_arr':         Ps4_arr,
        'Pt4_arr':         Pt4_arr,
    }


def _solve_ram_outer_closure(prob, design, M0, altitude_m,
                             rel_tol=1.0e-3, abs_tol_pa=500.0, max_iter=12):
    """
    Close the RAM inlet/back-pressure consistency with a bracketed scalar solve
    on combustor-face static pressure.
    """
    seed_case = _inlet2.evaluate_fixed_geometry_at_condition(
        design,
        M0=M0,
        altitude_m=altitude_m,
        alpha_deg=float(design.get('alpha_deg', INLET_DESIGN_ALPHA_DEG)),
        p_back=1.0,
    )
    terminal = seed_case.get('terminal', {})
    p_min = float(terminal.get('Ps_min', np.nan))
    p_max = float(terminal.get('Ps_max', np.nan))

    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        raise RuntimeError('Could not determine a valid RAM back-pressure bracket.')

    span = max(p_max - p_min, 1.0)
    p_lo = p_min + 1.0e-4 * span
    p_hi = p_max - 1.0e-4 * span
    if p_hi <= p_lo:
        p_lo = p_min
        p_hi = p_max

    lo_eval = _evaluate_ram_outer_residual(prob, design, M0, altitude_m, p_lo)
    hi_eval = _evaluate_ram_outer_residual(prob, design, M0, altitude_m, p_hi)

    best = min((lo_eval, hi_eval), key=lambda item: abs(item['residual']))
    if abs(best['residual']) <= max(abs_tol_pa, rel_tol * max(best['cycle_ps_back'], 1.0)):
        return best

    if lo_eval['residual'] == 0.0:
        return lo_eval
    if hi_eval['residual'] == 0.0:
        return hi_eval

    if np.sign(lo_eval['residual']) == np.sign(hi_eval['residual']):
        return best

    for _ in range(max_iter):
        p_mid = 0.5 * (p_lo + p_hi)
        mid_eval = _evaluate_ram_outer_residual(prob, design, M0, altitude_m, p_mid)
        if abs(mid_eval['residual']) < abs(best['residual']):
            best = mid_eval

        if abs(mid_eval['residual']) <= max(abs_tol_pa, rel_tol * max(mid_eval['cycle_ps_back'], 1.0)):
            return mid_eval

        if np.sign(mid_eval['residual']) == np.sign(lo_eval['residual']):
            p_lo = p_mid
            lo_eval = mid_eval
        else:
            p_hi = p_mid
            hi_eval = mid_eval

    return best


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
    nozzle_throat_area: float,
    combustor_L_star: float,
    design: dict | None = None,
    cross_section_area_m2: float | None = None,
    cross_section_shape: str | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
) -> dict:
    """
    Size a constant-area combustor from characteristic length.

    When a diffuser is present, the combustor inherits the diffuser exit area
    and is treated as circular. Legacy rectangular sizing is retained as a
    fallback for cases without a diffuser block.
    """
    if nozzle_throat_area <= 0.0:
        raise ValueError("nozzle_throat_area must be positive.")
    if combustor_L_star <= 0.0:
        raise ValueError("combustor_L_star must be positive.")

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
    combustor_volume = combustor_L_star * nozzle_throat_area
    combustor_length = combustor_volume / combustor_area

    geometry = {
        "L_star": float(combustor_L_star),
        "cross_section_shape": str(cross_section_shape or "circular"),
        "cross_section_area_m2": float(combustor_area),
        "throat_area_m2": float(nozzle_throat_area),
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
    from L* and throat area, and estimates nozzle packaging from throat area
    and commanded nozzle area ratio. No pyCycle / nozzle performance solve.
    """
    if not isinstance(design, dict):
        raise TypeError("design must be a dict returned by build_design().")

    nozzle_throat_area = float(design["throat_area_actual_m2"])
    combustor_L_star = float(design.get("combustor_L_star", COMBUSTOR_L_STAR_DEFAULT))
    nozzle_AR = float(design.get("nozzle_AR", NOZZLE_AR_DEFAULT))

    combustor_geometry = compute_combustor_geometry(
        nozzle_throat_area=nozzle_throat_area,
        combustor_L_star=combustor_L_star,
        design=design,
    )
    combustor_section = _build_combustor_section_summary(combustor_geometry)
    inlet_geometry = _build_inlet_geometry_summary(design)
    nozzle_geometry = _estimate_nozzle_geometry(
        throat_area_m2=nozzle_throat_area,
        nozzle_AR=nozzle_AR,
        combustor_diameter_m=combustor_section["diameter_m"],
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
    })
    return summary


def _reconstruct_ram_nozzle_geometry(state4, state9, mass_flow, phi, thermo, eta_n):
    """
    Build throat and exit areas from the local nozzle solution.
    """
    gamma_star = state4.gamma
    r_star = state4.R
    T_star = isentropic_T(state4.Tt, 1.0, gamma_star)
    P_star = isentropic_P(state4.Pt, 1.0, gamma_star)

    for _ in range(2):
        gamma_star = thermo.gamma(T_star, phi, P_star)
        r_star = thermo.R(T_star, phi, P_star)
        T_star = isentropic_T(state4.Tt, 1.0, gamma_star)
        P_star = isentropic_P(state4.Pt, 1.0, gamma_star)

    rho_star = P_star / max(r_star * T_star, 1.0e-12)
    a_star = np.sqrt(gamma_star * r_star * T_star)
    throat_area = mass_flow / max(rho_star * a_star, 1.0e-12)

    rho9 = state9.P / max(state9.R * state9.T, 1.0e-12)
    v9 = eta_n * state9.V
    exit_area = mass_flow / max(rho9 * v9, 1.0e-12)

    throat = {
        "area": float(throat_area),
        "T": float(T_star),
        "P": float(P_star),
        "M": 1.0,
        "Pt": float(state4.Pt),
        "Tt": float(state4.Tt),
        "gamma": float(gamma_star),
    }
    exit = {
        "area": float(exit_area),
        "T": float(state9.T),
        "P": float(state9.P),
        "M": float(state9.M),
        "Pt": float(state9.Pt),
        "Tt": float(state9.Tt),
        "gamma": float(state9.gamma),
    }
    perf = {
        "F_cruise": np.nan,
        "area_ratio": float(exit_area / throat_area) if throat_area > 0.0 else np.nan,
        "expansion_state": "ideally expanded",
    }
    return throat, exit, perf


# ---------------------------------------------------------------------------
# Single-point analysis
# ---------------------------------------------------------------------------

def analyze(
    M0:           float,
    altitude_m:   float,
    phi:          float,
    ramp_angles:  list  | None = None,
    combustor_L_star: float | None = None,
    design = None,
    verbose:      bool         = False,
    closure_version: str = 'v2',
) -> dict:
    """
    Run the ramjet cycle at a single flight condition.

    Parameters
    ----------
    M0           : Freestream Mach number
    altitude_m   : Geometric altitude [m]
    phi          : Equivalence ratio
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

    combustor_L_star_eff = (float(combustor_L_star) if combustor_L_star is not None
                             else float(COMBUSTOR_L_STAR_DEFAULT))
    combustor_geometry = compute_combustor_geometry(
        nozzle_throat_area=float(design["throat_area_actual_m2"]),
        combustor_L_star=combustor_L_star_eff,
        design=design,
    )
    combustor_area_ratio = (
        combustor_geometry["cross_section_area_m2"] / design["throat_area_actual_m2"]
    )

    ram_closure = None

    prob = _get_problem()

    # ── Seed FlightConditions balance with isentropic initial guess ──────────
    # This is the critical fix for the FloatingPointError / log(0) in CEA.
    # Must be called before set_val of flight conditions so the balance
    # starts from a physically reasonable state.
    _seed_fc_initial_guess(prob, M0, T0, P0)

    # ── Flight conditions ────────────────────────────────────────────────────
    prob.set_val('fc.alt', altitude_m * M2FT, units='ft')
    prob.set_val('fc.MN',  M0)
    prob.set_val('fc.W',   W_lbms, units='lbm/s')
    prob.set_val("burner.area_ratio", combustor_area_ratio)

    # ── Inlet / RAM closure ──────────────────────────────────────────────────
    # v1: bisection on p_back until pyCycle's inlet self-consistent.
    # v2: compute inlet capability + φ envelope outside pyCycle, clip φ
    #     against the soft-min of (inlet Ps_max, M4_max, Tt4_max), then run
    #     pyCycle once. φ used downstream for FAR, nozzle, and reporting is
    #     the clipped value — not the commanded one.
    phi_effective = float(phi)
    envelope_info = None
    if closure_version == 'v2':
        # Clip φ via envelope BEFORE setting FAR, then run a single pyCycle
        # pass at p_back = Ps3 (combustor-entrance static from envelope).
        A3_combustor = float(combustor_geometry["cross_section_area_m2"])
        capability = _get_capability(
            design, M0, altitude_m,
            float(design.get('alpha_deg', INLET_DESIGN_ALPHA_DEG)),
        )
        envelope_info = _solve_phi_envelope(
            capability=capability,
            area_ratio=combustor_area_ratio,
            Tt4_max=TT4_MAX_K,
            phi_request=phi,
            thermo=get_thermo(),
            mdot_freestream=W_kgs,
            A3=A3_combustor,
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
        # p_back is the combustor-entrance static pressure (Ps3), which is
        # what pyCycle's inlet output static matches against.
        ram_closure = _evaluate_ram_outer_residual(
            prob, design, M0, altitude_m, envelope_info['Ps3'],
        )
        ram_closure['envelope']    = envelope_info
        ram_closure['phi_clipped'] = phi_effective
        # Guard against envelope/pyCycle divergence. Inputs are self-consistent
        # by construction — a large residual means one of them is wrong.
        _res_pa  = abs(float(ram_closure['residual']))
        _ref_pa  = max(float(ram_closure['cycle_ps_back']), 1.0)
        if _res_pa > max(2000.0, 0.02 * _ref_pa):
            import warnings as _warnings
            _warnings.warn(
                f"v2 closure residual {_res_pa:.0f} Pa at M0={M0:.2f}, "
                f"alt={altitude_m:.0f} m (cycle Ps_back={_ref_pa:.0f} Pa) — "
                f"envelope Ps3 may be inconsistent with pyCycle.",
                RuntimeWarning, stacklevel=2,
            )
    else:
        FAR = phi_effective * F_STOICH_JP10
        prob.set_val("burner.Fl_I:FAR", FAR)
        ram_closure = _solve_ram_outer_closure(prob, design, M0, altitude_m)

    def _K(p):  return float(prob.get_val(p, units='degK')[0])
    def _Pa(p): return float(prob.get_val(p, units='Pa')[0])
    def _mn(p): return float(prob.get_val(p)[0])

    # ── Nozzle: replace pyCycle's internal nozz with nozzle_design.py ───────
    # Burner exit totals and mass flow feed a standalone FlowStart -> Nozzle
    # problem built by nozzle_design.build_pycycle_problem (via
    # run_pycycle_nozzle).  Thrust, Isp, throat area, exit area, and station
    # 9 values below come from that run — pyCycle's perf element is ignored.
    Pt4_Pa = _Pa('burner.Fl_O:tot:P')
    Tt4_K  = _K('burner.Fl_O:tot:T')
    MN4    = _mn('burner.Fl_O:stat:MN')
    W4_kgs = float(prob.get_val('burner.Fl_O:stat:W', units='kg/s')[0])
    state4 = FlowState(
        M=MN4,
        T=_K('burner.Fl_O:stat:T'),
        P=_Pa('burner.Fl_O:stat:P'),
        Pt=Pt4_Pa,
        Tt=Tt4_K,
        gamma=float(prob.get_val('burner.Fl_O:stat:gamma')[0]),
        R=float(prob.get_val('burner.Fl_O:stat:R', units='J/(kg*K)')[0]),
    )

    thermo = get_thermo()
    state0 = FlowState(
        M=M0,
        T=T0,
        P=P0,
        Pt=_Pa('fc.Fl_O:tot:P'),
        Tt=_K('fc.Fl_O:tot:T'),
        gamma=AIR_GAM,
        R=AIR_R,
    )
    F_sp, Isp, state9 = compute_nozzle(
        state4=state4,
        state0=state0,
        P0=P0,
        phi=phi_effective,
        thermo=thermo,
        eta_n=ETA_NOZZLE_CV,
    )
    nozzle_throat, nozzle_exit, nozzle_perf = _reconstruct_ram_nozzle_geometry(
        state4=state4,
        state9=state9,
        mass_flow=W4_kgs,
        phi=phi_effective,
        thermo=thermo,
        eta_n=ETA_NOZZLE_CV,
    )
    Fn_N = F_sp * W_kgs
    # F_cruise = momentum + pressure thrust at ambient (gross thrust).
    # Net thrust for a ramjet subtracts ram drag on the captured air stream:
    #     Fn = F_cruise - mdot_air * V0
    Wfuel = float(W_kgs * FAR)

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
    packaging = _build_geometry_packaging_summary(
        design=design,
        inlet_geometry=inlet_geometry,
        combustor_geometry=combustor_geometry,
        combustor_section=combustor_section,
        nozzle_geometry=nozzle_geometry,
    )


    result = dict(
        M0=M0, altitude=altitude_m, phi=phi_effective,
        phi_request=float(phi),
        phi_effective=phi_effective,
        closure_version=closure_version,
        envelope=envelope_info,
        Isp=Isp, F_sp=F_sp, thrust=Fn_N,
        mdot_air=W_kgs, mdot_fuel=Wfuel,
        eta_pt=ram_closure['inlet_inputs']['ram_recovery'],
        choked=choked,
        x_shock=ram_closure['inlet_inputs']['x_shock'],
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
        print(f"  L*               = {combustor_geom.get('L_star', float('nan')):>8.5f} m")
        print(f"  throat area At   = {combustor_geom.get('throat_area_m2', float('nan')):>8.5f} m^2")
        print(f"  chamber volume   = {combustor_geom.get('volume_m3', float('nan')):>8.5f} m^3")
        print(f"  chamber length   = {combustor_geom.get('length_m', float('nan')):>8.5f} m")
    print()
    print("  Nozzle geometry")
    print(f"  throat area At   = {nozzle_geom.get('throat_area_m2', float('nan')):>8.5f} m^2")
    print(f"  throat diameter  = {nozzle_geom.get('throat_diameter_m', float('nan')):>8.5f} m")
    print(f"  exit area Ae     = {nozzle_geom.get('exit_area_m2', float('nan')):>8.5f} m^2")
    print(f"  exit diameter    = {nozzle_geom.get('exit_diameter_m', float('nan')):>8.5f} m")
    print(f"  area ratio Ae/At = {nozzle_geom.get('area_ratio', float('nan')):>8.5f}")
    print(f"  expansion state  = {nozzle_geom.get('expansion_state', 'n/a')}")
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
    print("=" * 60)
    print("  pyCycle Ramjet")
    print("=" * 60)

    print("\n--- Design Point Performance")
    analyze(M0=INLET_DESIGN_M0, altitude_m=INLET_DESIGN_ALT_M, phi=PHI_DEFAULT, verbose=True)



    '''
    
    BEFORE RUNNING IN VIRTUAL ENVIRONMENTS OR ROOT DIRECTORY INTERPRETER PYCYCLE MUST BE MODIFIED
    
#    - pycycle\thermo\cea\props_rhs.py
    change outputs['rhs_P'][num_element] = inputs['n_moles'] to inputs['n_moles'][0]
    
#    - pycycle\ thermo\cea\props_calcs.py
    change both n_moles = inputs['n_moles'] assignments to inputs['n_moles'][0]
    
    '''



#mach 4-5
#altitude 15-24km
