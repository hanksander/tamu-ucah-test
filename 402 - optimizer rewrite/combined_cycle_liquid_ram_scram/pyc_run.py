"""
pyc_run.py
==========
Analysis runner for the pyCycle dual-mode ram/scramjet.

Mode selection
--------------
    M0 < M_transition  ->  RAM   (RayleighCombustor, subsonic Rayleigh)
    M0 >= M_transition ->  SCRAM (ScramCombustor, Rayleigh)

User-visible functions
----------------------
    analyze(M0, altitude_m, phi, M_transition, ramp_angles, verbose)
    mach_sweep(mach_range, altitude_m, phi, M_transition)

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

from gas_dynamics import FlowState, isentropic_P, isentropic_T, pi_milspec
from nozzle import compute_nozzle
from pyc_config import (
    F_STOICH_JP10, ETA_COMBUSTOR,
    ETA_NOZZLE_CV, ISOLATOR_PT_RECOVERY,
    M_TRANSITION,
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M,
    INLET_DESIGN_ALPHA_DEG, INLET_DESIGN_LEADING_EDGE_ANGLE_DEG,
    INLET_DESIGN_MDOT_KGS, INLET_DESIGN_WIDTH_M,
    INLET_FOREBODY_SEP_MARGIN, INLET_RAMP_SEP_MARGIN,
    INLET_KANTROWITZ_MARGIN, INLET_SHOCK_FOCUS_FACTOR,
    NOZZLE_TYPE,
)
from pyc_ram_cycle   import RamCycle
from pyc_scram_cycle import ScramCycle
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
# Problem singletons  (built once, reused across the Mach sweep)
# ---------------------------------------------------------------------------

_ram_prob   = None
_scram_prob = None


def _make_problem(CycleClass):
    """Build and set up one pyCycle Problem."""
    prob = om.Problem()
    prob.model = CycleClass(design=True)

    # Newton with solve_subsystems=True lets each pyCycle element's own
    # internal solver (FlightConditions balance, Nozzle PR_bal, etc.) run
    # to convergence on the first Newton step, giving a good initial Jacobian.
    # The props_calcs.py n_moles[0] patch (applied to the installed package)
    # is required for compute_partials to work without a ValueError.
    nlsolver = om.NewtonSolver(solve_subsystems=True)
    nlsolver.options['maxiter']             = 30
    nlsolver.options['atol']               = 1e-8
    nlsolver.options['rtol']               = 1e-8
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


def _get_problem(mode):
    """Lazily build and return the singleton Problem for the given mode."""
    global _ram_prob, _scram_prob
    if mode == 'ram':
        if _ram_prob is None:
            _ram_prob = _make_problem(RamCycle)
        return _ram_prob
    else:
        if _scram_prob is None:
            _scram_prob = _make_problem(ScramCycle)
        return _scram_prob


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
    """Build and cache the frozen inlet geometry from pyc_config defaults."""
    # Per-call override of design inputs is intentionally disabled for now;
    # plumb arguments through here when that becomes useful.
    # def _get_inlet_design(M0=None, alt=None, alpha=None, le_angle=None,
    #                      mdot=None, width=None): ...
    global _inlet_design
    if _inlet_design is None:
        _inlet_design = _inlet2.design_2ramp_shock_matched_inlet(
            M0=INLET_DESIGN_M0,
            altitude_m=INLET_DESIGN_ALT_M,
            alpha_deg=INLET_DESIGN_ALPHA_DEG,
            leading_edge_angle_deg=INLET_DESIGN_LEADING_EDGE_ANGLE_DEG,
            mdot_required=INLET_DESIGN_MDOT_KGS,
            width_m=INLET_DESIGN_WIDTH_M,
            forebody_separation_margin=INLET_FOREBODY_SEP_MARGIN,
            ramp_separation_margin=INLET_RAMP_SEP_MARGIN,
            kantrowitz_margin=INLET_KANTROWITZ_MARGIN,
            shock_focus_factor=INLET_SHOCK_FOCUS_FACTOR,
        )
    return _inlet_design


def compute_precowl_state(M0, alt_m, alpha_deg=0.0):
    """
    Explicit, non-iterative: run the frozen oblique-shock chain up to the
    cowl-shock exit and return Pt_after_cowl, Tt0, M3, plus a success flag.
    Separated so it can be called once per design point and its outputs
    fed into the Newton-looped DiffuserTerminalShock component.
    """
    design = _get_inlet_design()
    case = _inlet2.evaluate_fixed_geometry_at_condition(
        design, M0=M0, altitude_m=alt_m, alpha_deg=alpha_deg,
        p_back=1.0,
    )
    return design, case


def compute_inlet_conditions(M0, alt_m, mode, ramp_angles=None, alpha_deg=0.0):
    """
    RAM mode: returns (None, None). ram_recovery and MN_exit are driven by
    DiffuserTerminalShock inside RamCycle's Newton loop; analyze() sets
    diff.M0 / diff.alt_m instead.

    SCRAM mode: explicit (shock is swallowed past the throat).
    """
    del ramp_angles

    if mode == 'ram':
        return None, None

    design = _get_inlet_design()
    case = _inlet2.evaluate_fixed_geometry_at_condition(
        design, M0=M0, altitude_m=alt_m, alpha_deg=alpha_deg,
        p_back=1.0,
    )
    if not case.get('success', False):
        Pt_ratio = pi_milspec(M0) * ISOLATOR_PT_RECOVERY
        exit_MN  = max(M0 * 0.75, 1.05)
        return float(Pt_ratio), float(exit_MN)

    # R2 reflection cascade is now the canonical isolator pt-loss model.
    # It already includes all oblique reflections and a terminating normal
    # shock to subsonic, so no additional ISOLATOR_PT_RECOVERY factor is
    # applied here.
    Pt_ratio = float(case['pt_frac_after_reflection_cascade'])
    exit_MN  = float(case['M_after_reflection_cascade'])
    return Pt_ratio, exit_MN


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
    M_transition: float | None = None,
    ramp_angles:  list  | None = None,
    combustor_L_star: float | None = None,
    verbose:      bool         = False,
) -> dict:
    """
    Run the dual-mode cycle at a single flight condition.

    Parameters
    ----------
    M0           : Freestream Mach number
    altitude_m   : Geometric altitude [m]
    phi          : Equivalence ratio
    M_transition : RAM->SCRAM switch Mach (None -> pyc_config.M_TRANSITION)
    ramp_angles  : Ramp deflection angles [deg] (None -> config default)
    verbose      : Print cycle table

    Returns
    -------
    dict with keys:
        mode, M0, altitude, phi, M_trans,
        Isp [s], F_sp [N*s/kg_air], thrust [N],
        mdot_air [kg/s], mdot_fuel [kg/s],
        eta_pt, choked,
        T_stations, Tt_stations, M_stations, Pt_stations
    """
    M_trans = float(M_transition) if M_transition is not None else M_TRANSITION
    mode    = 'scram' if M0 >= M_trans else 'ram'
    design  = _get_inlet_design()

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

    # True fuel-air ratio by mass. Combustion efficiency is applied to the
    # heat release inside compute_combustor (not to the fuel mass flow), so
    # FAR here is the physical injected-fuel ratio. This prevents eta_c from
    # being double-counted between mass bookkeeping and heat release.
    FAR = phi * F_STOICH_JP10

    combustor_L_star_eff = float(combustor_L_star) if combustor_L_star is not None else 1.0
    combustor_geometry = compute_combustor_geometry(
        nozzle_throat_area=float(design["throat_area_actual_m2"]),
        combustor_L_star=combustor_L_star_eff,
        design=design,
    )
    combustor_area_ratio = (
        combustor_geometry["cross_section_area_m2"] / design["throat_area_actual_m2"]
    )

    # Inlet shock recovery + exit Mach
    ram_recovery, inlet_MN = compute_inlet_conditions(
        M0, altitude_m, mode, ramp_angles
    )

    prob = _get_problem(mode)

    # ── Seed FlightConditions balance with isentropic initial guess ──────────
    # This is the critical fix for the FloatingPointError / log(0) in CEA.
    # Must be called before set_val of flight conditions so the balance
    # starts from a physically reasonable state.
    _seed_fc_initial_guess(prob, M0, T0, P0)

    # ── Flight conditions ────────────────────────────────────────────────────
    prob.set_val('fc.alt', altitude_m * M2FT, units='ft')
    prob.set_val('fc.MN',  M0)
    prob.set_val('fc.W',   W_lbms, units='lbm/s')

    # ── Inlet ────────────────────────────────────────────────────────────────
    if mode == 'ram':
        # ram_recovery and inlet.MN are driven by DiffuserTerminalShock via
        # the Newton loop in RamCycle. Feed it M0 / altitude so it can rerun
        # the frozen shock chain at the current flight condition.
        prob.set_val('diff.M0',    M0)
        prob.set_val('diff.alt_m', altitude_m, units='m')
    else:
        prob.set_val('inlet.ram_recovery', ram_recovery)
        prob.set_val('inlet.MN',           inlet_MN)

    # ── Combustor ────────────────────────────────────────────────────────────
    prob.set_val('burner.Fl_I:FAR', FAR)
    if mode == 'ram':
        prob.set_val('burner.area_ratio', combustor_area_ratio)

    # ── Solve ────────────────────────────────────────────────────────────────
    prob.run_model()

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

    if mode == 'ram':
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
            phi=phi,
            thermo=thermo,
            eta_n=ETA_NOZZLE_CV,
        )
        nozzle_throat, nozzle_exit, nozzle_perf = _reconstruct_ram_nozzle_geometry(
            state4=state4,
            state9=state9,
            mass_flow=W4_kgs,
            phi=phi,
            thermo=thermo,
            eta_n=ETA_NOZZLE_CV,
        )
        Fn_N = F_sp * W_kgs
    else:
        burner_exit_flowstation = {
            'Fl_O:tot:h': prob.get_val('burner.Fl_O:tot:h'),
            'Fl_O:tot:T': prob.get_val('burner.Fl_O:tot:T'),
            'Fl_O:tot:P': prob.get_val('burner.Fl_O:tot:P'),
            'Fl_O:tot:S': prob.get_val('burner.Fl_O:tot:S'),
            'Fl_O:tot:rho': prob.get_val('burner.Fl_O:tot:rho'),
            'Fl_O:tot:gamma': prob.get_val('burner.Fl_O:tot:gamma'),
            'Fl_O:tot:Cp': prob.get_val('burner.Fl_O:tot:Cp'),
            'Fl_O:tot:Cv': prob.get_val('burner.Fl_O:tot:Cv'),
            'Fl_O:tot:R': prob.get_val('burner.Fl_O:tot:R'),
            'Fl_O:tot:composition': prob.get_val('burner.Fl_O:tot:composition'),
            'Fl_O:stat:h': prob.get_val('burner.Fl_O:stat:h'),
            'Fl_O:stat:T': prob.get_val('burner.Fl_O:stat:T'),
            'Fl_O:stat:P': prob.get_val('burner.Fl_O:stat:P'),
            'Fl_O:stat:S': prob.get_val('burner.Fl_O:stat:S'),
            'Fl_O:stat:rho': prob.get_val('burner.Fl_O:stat:rho'),
            'Fl_O:stat:gamma': prob.get_val('burner.Fl_O:stat:gamma'),
            'Fl_O:stat:Cp': prob.get_val('burner.Fl_O:stat:Cp'),
            'Fl_O:stat:Cv': prob.get_val('burner.Fl_O:stat:Cv'),
            'Fl_O:stat:R': prob.get_val('burner.Fl_O:stat:R'),
            'Fl_O:stat:V': prob.get_val('burner.Fl_O:stat:V'),
            'Fl_O:stat:Vsonic': prob.get_val('burner.Fl_O:stat:Vsonic'),
            'Fl_O:stat:MN': prob.get_val('burner.Fl_O:stat:MN'),
            'Fl_O:stat:area': prob.get_val('burner.Fl_O:stat:area'),
            'Fl_O:stat:W': prob.get_val('burner.Fl_O:stat:W'),
        }
        burner_exit_flow_port_data = prob.model._get_subsystem('burner').Fl_O_data['Fl_O']

        noz = nozzle_design.run_pycycle_nozzle(
            m_inlet=MN4,
            pt_inlet=Pt4_Pa,
            tt_inlet=Tt4_K,
            ps_exhaust=P0,
            cv=ETA_NOZZLE_CV,
            nozzle_type=NOZZLE_TYPE,
            mass_flow=W4_kgs,
            flowstation=burner_exit_flowstation,
            flow_port_data=burner_exit_flow_port_data,
            ambient_pressure=P0,
        )
        nozzle_exit   = noz['exit']
        nozzle_throat = noz['throat']
        nozzle_perf   = noz['performance']
        Fn_N  = float(nozzle_perf['F_cruise']) - W_kgs * V0
    # F_cruise = momentum + pressure thrust at ambient (gross thrust).
    # Net thrust for a ramjet subtracts ram drag on the captured air stream:
    #     Fn = F_cruise - mdot_air * V0
    Wfuel = float(W_kgs * FAR)
    if mode != 'ram':
        F_sp = Fn_N / max(W_kgs, 1e-12)
        Isp  = Fn_N / max(Wfuel * G0, 1e-12)

    choked = False
    if mode == 'ram':
        try:
            choked = bool(_mn('burner.choked') > 0.5)
        except Exception:
            pass
    else:
        try:
            choked = bool(_mn('burner.rayleigh.choked') > 0.5)
        except Exception:
            pass

    combustor_section = {
        "shape": str(combustor_geometry["cross_section_shape"]),
        "width_m": float(combustor_geometry["width_m"]),
        "height_m": float(combustor_geometry["height_m"]),
        "radius_m": float(combustor_geometry["radius_m"]),
        "diameter_m": float(combustor_geometry["diameter_m"]),
        "area_m2": float(combustor_geometry["cross_section_area_m2"]),
    }
    inlet_geometry = {
        "capture_area_m2": float(design["A_capture_required_m2"]),
        "post_cowl_area_m2": float(design["post_cowl_area_m2"]),
        "throat_area_m2": float(design["throat_area_actual_m2"]),
        "throat_height_m": float(design["throat_height_m"]),
        "width_m": float(INLET_DESIGN_WIDTH_M),
        "diffuser_exit_shape": str(design.get("diffuser", {}).get("exit_shape", "unknown")),
        "diffuser_exit_diameter_m": float(design.get("diffuser", {}).get("exit_diameter_m", np.nan)),
        "forebody_length_m": float(design["forebody_length_m"]),
        "ramp1_length_m": float(design["ramp1_length_m"]),
        "shock2_to_lip_m": float(design["shock2_distance_from_break2_to_lip_m"]),
    }
    nozzle_geometry = {
        "throat_area_m2": float(nozzle_throat["area"]),
        "exit_area_m2": float(nozzle_exit["area"]),
        "throat_radius_m": float(np.sqrt(float(nozzle_throat["area"]) / np.pi)),
        "throat_diameter_m": float(np.sqrt(4.0 * float(nozzle_throat["area"]) / np.pi)),
        "exit_radius_m": float(np.sqrt(float(nozzle_exit["area"]) / np.pi)),
        "exit_diameter_m": float(np.sqrt(4.0 * float(nozzle_exit["area"]) / np.pi)),
        "area_ratio": float(nozzle_perf["area_ratio"]),
        "expansion_state": nozzle_perf["expansion_state"],
    }

    result = dict(
        mode=mode, M0=M0, altitude=altitude_m, phi=phi, M_trans=M_trans,
        Isp=Isp, F_sp=F_sp, thrust=Fn_N,
        mdot_air=W_kgs, mdot_fuel=Wfuel,
        eta_pt=(float(prob.get_val('diff.ram_recovery')[0])
                if mode == 'ram' else ram_recovery),
        choked=choked,
        T_stations={
            0: _K('fc.Fl_O:stat:T'),
            3: _K('inlet.Fl_O:stat:T'),
            4: _K('burner.Fl_O:stat:T'),
            9: float(nozzle_exit['T']),
        },
        Tt_stations={
            0: _K('fc.Fl_O:tot:T'),
            3: _K('inlet.Fl_O:tot:T'),
            4: _K('burner.Fl_O:tot:T'),
            9: float(nozzle_exit['Tt']),
        },
        P_stations={
            0: _Pa('fc.Fl_O:stat:P'),
            3: _Pa('inlet.Fl_O:stat:P'),
            4: _Pa('burner.Fl_O:stat:P'),
            9: float(nozzle_exit['P']),
        },
        M_stations={
            0: M0,
            3: _mn('inlet.Fl_O:stat:MN'),
            4: _mn('burner.Fl_O:stat:MN'),
            9: float(nozzle_exit['M']),
        },
        Pt_stations={
            0: _Pa('fc.Fl_O:tot:P'),
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
    )

    if verbose:
        _print_cycle(result)

    return result


# ---------------------------------------------------------------------------
# Mach sweep
# ---------------------------------------------------------------------------

def mach_sweep(mach_range, altitude_m, phi,
               M_transition=None, **kwargs):
    """
    Run analyze() over a Mach array.  Returns None for failed points.
    """
    results = []
    for M in mach_range:
        try:
            results.append(analyze(M, altitude_m, phi,
                                   M_transition=M_transition, **kwargs))
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
    print(f"  {r['mode'].upper():6s}  |  M0={r['M0']:.2f}  "
          f"|  alt={r['altitude']/1e3:.1f} km  "
          f"|  phi={r['phi']:.2f}  "
          f"|  M_trans={r['M_trans']:.1f}")
    print(f"{'='*70}")
    print("  Station states")
    print(f"  {'Stn':>8}  {'M':>7}  {'Ts [K]':>8}  {'Tt [K]':>8}  "
          f"{'Ps [kPa]':>10}  {'Pt [kPa]':>10}")
    print(f"  {'-'*64}")
    lbl = {0: 'Free', 3: 'Comb.in', 4: 'Comb.out', 9: 'Nozzle'}
    for s in (0, 3, 4, 9):
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
    print("=" * 60)
    print("  pyCycle Dual-Mode Ram/Scramjet")
    print("=" * 60)

    print("\n--- Design point, phi=0.8 ---")
    analyze(M0=INLET_DESIGN_M0, altitude_m=INLET_DESIGN_ALT_M, phi=0.8, M_transition=5.2, verbose=True)



    '''
    
    BEFORE RUNNING IN VIRTUAL ENVIRONMENTS OR ROOT DIRECTORY INTERPRETER PYCYCLE MUST BE MODIFIED
    
#    - pycycle\thermo\cea\props_rhs.py
    change outputs['rhs_P'][num_element] = inputs['n_moles'] to inputs['n_moles'][0]
    
#    - pycycle\ thermo\cea\props_calcs.py
    change both n_moles = inputs['n_moles'] assignments to inputs['n_moles'][0]
    
    '''



