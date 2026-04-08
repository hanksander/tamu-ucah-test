"""
pyc_run.py
==========
Analysis runner for the pyCycle dual-mode ram/scramjet.

Mode selection
--------------
    M0 < M_transition  ->  RAM   (pyc.Combustor,  constant-pressure)
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

import numpy as np
import openmdao.api as om
from ambiance import Atmosphere

from gas_dynamics import oblique_shock, normal_shock, pi_milspec
from pyc_config import (
    A_CAPTURE, INLET_RAMPS_DEG,
    F_STOICH_JP7, ETA_COMBUSTOR,
    ETA_NOZZLE_CV, ISOLATOR_PT_RECOVERY,
    M_TRANSITION, RAM_COMBUSTOR_EXIT_MN,
)
from pyc_ram_cycle   import RamCycle
from pyc_scram_cycle import ScramCycle

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
# Inlet physics  (pre-computed outside pyCycle using gas_dynamics.py)
# ---------------------------------------------------------------------------

def compute_inlet_conditions(M0, alt_m, mode, ramp_angles=None):
    """
    Compute (ram_recovery, inlet_exit_MN) for pyCycle's Inlet element.

    RAM  : oblique shocks + terminal normal shock -> subsonic exit MN=0.35
           Pt loss includes oblique + normal shock + isolator recovery.
    SCRAM: oblique shocks only -> supersonic exit
           Pt loss includes oblique shock train + isolator recovery.
    Falls back to MIL-E-5007D if any shock detaches.
    """
    if ramp_angles is None:
        ramp_angles = INLET_RAMPS_DEG

    M, Pt_ratio = M0, 1.0
    detached = False

    for theta in ramp_angles:
        M2, _, _, Pt2Pt1, _ = oblique_shock(M, theta, AIR_GAM)
        if M2 is None:
            detached = True
            break
        Pt_ratio *= Pt2Pt1
        M = M2

    if detached:
        Pt_ratio = pi_milspec(M0)
        M = M0 * 0.75

    if mode == 'ram':
        M_ns, _, _, Pt_ns = normal_shock(M, AIR_GAM)
        Pt_ratio *= Pt_ns * ISOLATOR_PT_RECOVERY
        exit_MN   = min(float(M_ns), 0.35)
    else:
        Pt_ratio *= ISOLATOR_PT_RECOVERY
        exit_MN   = float(M) * 0.97

    return float(Pt_ratio), float(exit_MN)


# ---------------------------------------------------------------------------
# Single-point analysis
# ---------------------------------------------------------------------------

def analyze(
    M0:           float,
    altitude_m:   float,
    phi:          float,
    M_transition: float | None = None,
    ramp_angles:  list  | None = None,
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

    # Freestream conditions
    atm  = Atmosphere(altitude_m)
    T0   = float(atm.temperature[0])   # K
    P0   = float(atm.pressure[0])      # Pa
    rho0 = float(atm.density[0])       # kg/m^3

    V0      = M0 * np.sqrt(AIR_GAM * AIR_R * T0)
    W_kgs   = rho0 * V0 * A_CAPTURE            # kg/s
    W_lbms  = W_kgs * KG2LBM                   # lbm/s

    # Effective FAR (combustion efficiency applied here)
    FAR = ETA_COMBUSTOR * phi * F_STOICH_JP7

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
    prob.set_val('inlet.ram_recovery', ram_recovery)
    prob.set_val('inlet.MN',           inlet_MN)

    # ── Combustor ────────────────────────────────────────────────────────────
    prob.set_val('burner.Fl_I:FAR', FAR)
    if mode == 'ram':
        prob.set_val('burner.MN', RAM_COMBUSTOR_EXIT_MN)

    # ── Solve ────────────────────────────────────────────────────────────────
    prob.run_model()

    # ── Extract results in SI ────────────────────────────────────────────────
    Fn_N    = float(prob.get_val('perf.Fn',    units='N')[0])
    Wfuel   = float(prob.get_val('perf.Wfuel', units='kg/s')[0])

    F_sp = Fn_N / max(W_kgs, 1e-12)
    Isp  = Fn_N / max(Wfuel * G0, 1e-12)

    def _K(p):  return float(prob.get_val(p, units='degK')[0])
    def _Pa(p): return float(prob.get_val(p, units='Pa')[0])
    def _mn(p): return float(prob.get_val(p)[0])

    choked = False
    if mode == 'scram':
        try:
            choked = bool(_mn('burner.rayleigh.choked') > 0.5)
        except Exception:
            pass

    result = dict(
        mode=mode, M0=M0, altitude=altitude_m, phi=phi, M_trans=M_trans,
        Isp=Isp, F_sp=F_sp, thrust=Fn_N,
        mdot_air=W_kgs, mdot_fuel=Wfuel,
        eta_pt=ram_recovery, choked=choked,
        T_stations={
            0: _K('fc.Fl_O:stat:T'),
            3: _K('inlet.Fl_O:stat:T'),
            4: _K('burner.Fl_O:stat:T'),
            9: _K('nozz.Fl_O:stat:T'),
        },
        Tt_stations={
            0: _K('fc.Fl_O:tot:T'),
            3: _K('inlet.Fl_O:tot:T'),
            4: _K('burner.Fl_O:tot:T'),
            9: _K('nozz.Fl_O:tot:T'),
        },
        M_stations={
            0: M0,
            3: _mn('inlet.Fl_O:stat:MN'),
            4: _mn('burner.Fl_O:stat:MN'),
            9: _mn('nozz.Fl_O:stat:MN'),
        },
        Pt_stations={
            0: _Pa('fc.Fl_O:tot:P'),
            3: _Pa('inlet.Fl_O:tot:P'),
            4: _Pa('burner.Fl_O:tot:P'),
            9: _Pa('nozz.Fl_O:tot:P'),
        },
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
    print(f"\n{'='*70}")
    print(f"  {r['mode'].upper():6s}  |  M0={r['M0']:.2f}  "
          f"|  alt={r['altitude']/1e3:.1f} km  "
          f"|  phi={r['phi']:.2f}  "
          f"|  M_trans={r['M_trans']:.1f}")
    print(f"{'='*70}")
    print(f"  {'Stn':>8}  {'M':>7}  {'Ts [K]':>8}  {'Tt [K]':>8}  "
          f"{'Pt [kPa]':>10}")
    print(f"  {'-'*50}")
    lbl = {0: 'Free', 3: 'Comb.in', 4: 'Comb.out', 9: 'Nozzle'}
    for s in (0, 3, 4, 9):
        print(f"  {lbl[s]:>8}  {r['M_stations'][s]:>7.3f}  "
              f"{r['T_stations'][s]:>8.1f}  {r['Tt_stations'][s]:>8.1f}  "
              f"{r['Pt_stations'][s]/1e3:>10.2f}")
    print()
    print(f"  Isp       = {r['Isp']:>8.1f} s")
    print(f"  F_sp      = {r['F_sp']:>8.1f} N*s/kg_air")
    print(f"  Thrust    = {r['thrust']/1e3:>8.2f} kN   (A_cap={A_CAPTURE} m^2)")
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

    print("\n--- Design point: RAM  M=4.5, alt=20km, phi=0.8 ---")
    analyze(M0=4.5, altitude_m=20_000, phi=0.8, M_transition=5.0, verbose=True)
