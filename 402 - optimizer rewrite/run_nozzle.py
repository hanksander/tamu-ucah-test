"""
run_nozzle.py
=============
Complete hypersonic ramjet nozzle analysis using pyCycle + OpenMDAO.

What this script does
---------------------
  1. Standard atmosphere lookup at the design flight condition.
  2. Isentropic pre-sizing (analytical, no pyCycle needed) – quick geometry
     estimate and sanity-check before the full thermochemical solve.
  3. Design-point pyCycle solve (CD nozzle with CEA thermodynamics).
       – Sizes throat area A* and exit area A_exit from a target exit Mach.
       – Extracts all performance metrics.
  4. Nozzle start / unstart check (normal shock analysis).
  5. Off-design solves with FIXED geometry from step 3.
       – Varies back-pressure (altitude) to sweep expansion state.
       – Varies inlet total conditions (Tt, Pt) to simulate throttle.
  6. Back-pressure sweep: Ps_exhaust from 0.5x to 2.0x design ambient.
  7. Inlet Tt sweep: simulate variable combustor exit temperature.
  8. Inlet Pt sweep: simulate variable inlet recovery.
  9. All results printed in both English and SI units.
 10. SQLite case recording for post-processing.

Usage
-----
  python run_nozzle.py                    # full analysis, default conditions
  python run_nozzle.py --no-pycycle       # analytical pre-sizing only
  python run_nozzle.py --mach 7           # change design Mach
  python run_nozzle.py --Tt 4000 --Pt 80 # change nozzle inlet conditions

Design-point default conditions
---------------------------------
  Flight Mach        : 6.0
  Altitude           : 25 908 m  (85 000 ft)
  Nozzle inlet Tt    : 4 200 °R  (2 333 K)  – combustor exit
  Nozzle inlet Pt    : 70 psia   (482 kPa)  – after inlet + combustor losses
  Mass flow W        : 150 lbm/s (68 kg/s)
  FAR                : 0.04      – fuel-to-air ratio (combustion products)
  Target exit Mach   : 5.5
  Cfg                : 0.985     – gross thrust coefficient
"""

import argparse
import sys
import numpy as np
import openmdao.api as om

from nozzle_model import RamjetNozzle


# ============================================================================
# Unit conversion constants
# ============================================================================

FT_PER_M    = 3.28084
LBM_PER_KG  = 2.20462
LBF_PER_N   = 0.224809
PSI_PER_PA  = 1.0 / 6894.757
IN2_PER_M2  = 1550.003
FPS_PER_MPS = FT_PER_M
K_PER_R     = 5.0 / 9.0       # Rankine → Kelvin
R_PER_K     = 9.0 / 5.0       # Kelvin → Rankine


# ============================================================================
# 1976 US Standard Atmosphere (0 – 86 km)
# ============================================================================

_ATM_LAYERS = [
    # (base_alt_m,  base_T_K,  lapse_K/m,  base_P_Pa)
    (0.0,     288.15, -0.0065,  101325.0),
    (11000.0, 216.65,  0.0,      22632.1),
    (20000.0, 216.65,  0.001,    5474.89),
    (32000.0, 228.65,  0.0028,   868.019),
    (47000.0, 270.65,  0.0,      110.906),
    (51000.0, 270.65, -0.0028,   66.9389),
    (71000.0, 214.65, -0.002,    3.95642),
    (86000.0, 186.87,  0.0,      0.3734),
]
_G0 = 9.80665
_R  = 287.058


def atmosphere(alt_m: float) -> dict:
    """1976 US Standard Atmosphere – returns T [K], P [Pa], rho, a."""
    alt_m = float(np.clip(alt_m, 0, 86000))
    layer = _ATM_LAYERS[0]
    for lyr in _ATM_LAYERS:
        if alt_m >= lyr[0]:
            layer = lyr
    h0, T0, L, P0 = layer
    dh = alt_m - h0
    T  = T0 + L * dh
    P  = P0 * (np.exp(-_G0 * dh / (_R * T0)) if abs(L) < 1e-12
               else (T / T0) ** (-_G0 / (L * _R)))
    return {"T_K": T, "P_Pa": P, "rho": P / (_R * T),
            "a_ms": np.sqrt(1.4 * _R * T)}


def flight_state(alt_m: float, mach: float) -> dict:
    """Full freestream + total conditions for a given altitude and Mach."""
    atm = atmosphere(alt_m)
    T, P, a = atm["T_K"], atm["P_Pa"], atm["a_ms"]
    Tt = T  * (1 + 0.2 * mach**2)
    Pt = P  * (Tt / T) ** (1.4 / 0.4)
    V  = mach * a
    return {**atm,
            "Mach": mach, "V_ms": V, "Tt_K": Tt, "Pt_Pa": Pt,
            "Tt_R": Tt * R_PER_K, "Pt_psia": Pt * PSI_PER_PA,
            "P_psia": P * PSI_PER_PA,
            "T_R": T * R_PER_K}


# ============================================================================
# Isentropic relations - for initial sizing
# ============================================================================

def area_ratio(M: float, g: float = 1.3) -> float:
    """A/A* as a function of Mach and γ."""
    t = 1 + 0.5 * (g - 1) * M**2
    return (1 / M) * (t / (0.5 * (g + 1))) ** (0.5 * (g + 1) / (g - 1))


def mach_from_area_ratio(AR: float, g: float = 1.3,
                          supersonic: bool = True) -> float:
    """Invert area_ratio() via Newton iteration."""
    M = 3.0 if supersonic else 0.5
    for _ in range(200):
        f  = area_ratio(M, g) - AR
        dM = M * 1e-7
        df = (area_ratio(M + dM, g) - area_ratio(M - dM, g)) / (2 * dM)
        M -= f / df
        M  = max(M, 1e-6)
        if abs(f) < 1e-10:
            break
    return M


def isentropic_presizing(Tt_K: float, Pt_Pa: float, mdot_kgs: float,
                          M_exit: float, g: float = 1.3) -> dict:
    """
    Compute throat and exit areas from 1D isentropic nozzle relations.
    Used as a quick estimate before the pyCycle CEA solve.
    """
    R = 287.058   # J/(kg·K) – approximate; CEA uses mixture-specific R
    # Throat (choked, M=1)
    T_th = Tt_K  / (1 + 0.5 * (g - 1))
    P_th = Pt_Pa / ((1 + 0.5 * (g - 1)) ** (g / (g - 1)))
    V_th = np.sqrt(g * R * T_th)   # = speed of sound at throat
    A_th = mdot_kgs / (P_th / (R * T_th) * V_th)   # m²
    # Exit
    AR   = area_ratio(M_exit, g)
    A_ex = A_th * AR
    # Exit static conditions
    T_ex = Tt_K  / (1 + 0.5 * (g - 1) * M_exit**2)
    P_ex = Pt_Pa / ((1 + 0.5 * (g - 1) * M_exit**2) ** (g / (g - 1)))
    V_ex = M_exit * np.sqrt(g * R * T_ex)
    Fg_id = mdot_kgs * V_ex   # ideal momentum thrust [N]
    return {
        "A_throat_m2": A_th, "A_exit_m2": A_ex, "AR": AR,
        "T_throat_K": T_th,  "P_throat_Pa": P_th, "V_throat_ms": V_th,
        "T_exit_K": T_ex,    "P_exit_Pa": P_ex,   "V_exit_ms": V_ex,
        "Fg_ideal_N": Fg_id,
    }


def normal_shock(M1: float, g: float = 1.3) -> dict:
    """Normal shock relations upstream Mach M1."""
    M2  = np.sqrt(((g-1)*M1**2 + 2) / (2*g*M1**2 - (g-1)))
    P2P1 = (2*g*M1**2 - (g-1)) / (g+1)
    T2T1 = P2P1 * (2 + (g-1)*M1**2) / ((g+1)*M1**2)
    Pt2Pt1 = (((g+1)*M1**2) / (2+(g-1)*M1**2))**(g/(g-1)) * \
             ((g+1)/(2*g*M1**2-(g-1)))**(1/(g-1))
    return {"M2": M2, "P2_P1": P2P1, "T2_T1": T2T1, "Pt2_Pt1": Pt2Pt1}


# ============================================================================
# OpenMDAO problem builders
# ============================================================================

def build_problem(design: bool, record_file: str = None) -> om.Problem:
    """Create a bare Problem with the RamjetNozzle group."""
    prob = om.Problem()
    prob.model.add_subsystem("nozzle_grp", RamjetNozzle(design=design),
                             promotes=["*"])
    if record_file:
        rec = om.SqliteRecorder(record_file)
        prob.add_recorder(rec)
    prob.setup(check=False, force_alloc_complex=True)
    return prob


def set_inlet_conditions(prob: om.Problem,
                          Tt_R: float, Pt_psia: float,
                          W_lbms: float, FAR: float) -> None:
    """Helper: set nozzle inlet total conditions."""
    prob.set_val("nozzle_in.Tt",  Tt_R,   units="degR")
    prob.set_val("nozzle_in.Pt",  Pt_psia, units="psia")
    prob.set_val("nozzle_in.W",   W_lbms,  units="lbm/s")
    prob.set_val("nozzle_in.FAR", FAR)


def get_results(prob: om.Problem) -> dict:
    """Extract all key outputs from a solved Problem into a flat dict."""
    g = lambda v, **kw: float(prob.get_val(v, **kw))

    Fg          = g("nozzle.Fg",                      units="lbf")
    MN          = g("nozzle.Fl_O:stat:MN")
    V_fps       = g("nozzle.Fl_O:stat:V",             units="ft/s")
    Ps_exit     = g("nozzle.Fl_O:stat:P",             units="psia")
    Ts_exit     = g("nozzle.Fl_O:stat:T",             units="degR")
    Tt_exit     = g("nozzle.Fl_O:tot:T",              units="degR")
    A_exit_in2  = g("nozzle.Fl_O:stat:area",          units="inch**2")
    A_th_in2    = g("nozzle.Throat:Fl_O:stat:area",   units="inch**2")
    Pt_in       = g("nozzle_in.Fl_O:tot:P",           units="psia")
    Tt_in       = g("nozzle_in.Fl_O:tot:T",           units="degR")
    Ps_amb      = g("nozzle.Ps_exhaust",               units="psia")
    AR          = g("post.AR")
    CF          = g("post.CF")
    NPR         = g("post.NPR")
    delta_P     = g("post.delta_P",                    units="psia")
    Cfg         = g("nozzle.Cfg")

    # Expansion state
    tol = 0.03 * Ps_amb
    if abs(delta_P) < tol:
        expansion = "Perfectly expanded"
    elif delta_P > 0:
        expansion = f"Under-expanded  (ΔP = +{delta_P:.4f} psia)"
    else:
        expansion = f"Over-expanded   (ΔP = {delta_P:.4f} psia)"

    # Nozzle start check: is there a normal shock in the diverging section?
    ns = normal_shock(MN)
    Ps_exit_after_ns = Ps_exit * ns["P2_P1"]   # if shock sat at exit plane
    started = MN > 1.0

    # Unit conversions to SI
    A_exit_m2  = A_exit_in2 / IN2_PER_M2
    A_th_m2    = A_th_in2   / IN2_PER_M2
    V_ms       = V_fps      / FPS_PER_MPS
    Fg_N       = Fg         / LBF_PER_N
    Ts_exit_K  = Ts_exit    * K_PER_R
    Tt_exit_K  = Tt_exit    * K_PER_R
    Tt_in_K    = Tt_in      * K_PER_R
    Pt_in_Pa   = Pt_in      / PSI_PER_PA
    Ps_exit_Pa = Ps_exit    / PSI_PER_PA
    Ps_amb_Pa  = Ps_amb     / PSI_PER_PA

    return {
        # Thrust
        "Fg_lbf":        Fg,
        "Fg_N":          Fg_N,
        # Nozzle inlet
        "Tt_in_R":       Tt_in,
        "Tt_in_K":       Tt_in_K,
        "Pt_in_psia":    Pt_in,
        "Pt_in_Pa":      Pt_in_Pa,
        # Nozzle exit (total)
        "Tt_exit_R":     Tt_exit,
        "Tt_exit_K":     Tt_exit_K,
        # Nozzle exit (static)
        "MN_exit":       MN,
        "V_exit_fps":    V_fps,
        "V_exit_ms":     V_ms,
        "Ts_exit_R":     Ts_exit,
        "Ts_exit_K":     Ts_exit_K,
        "Ps_exit_psia":  Ps_exit,
        "Ps_exit_Pa":    Ps_exit_Pa,
        # Ambient
        "Ps_amb_psia":   Ps_amb,
        "Ps_amb_Pa":     Ps_amb_Pa,
        # Geometry
        "A_throat_in2":  A_th_in2,
        "A_throat_m2":   A_th_m2,
        "A_exit_in2":    A_exit_in2,
        "A_exit_m2":     A_exit_m2,
        "area_ratio":    AR,
        # Coefficients
        "CF":            CF,
        "NPR":           NPR,
        "Cfg":           Cfg,
        # Expansion and start
        "delta_P_psia":  delta_P,
        "expansion":     expansion,
        "nozzle_started": started,
        "M_after_normal_shock": ns["M2"],
        "Pt_loss_normal_shock": 1 - ns["Pt2_Pt1"],
    }


# ============================================================================
# Print helpers
# ============================================================================

DIVIDER  = "=" * 68
THIN_DIV = "─" * 68


def _hdr(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_atmosphere(alt_m: float, mach: float) -> None:
    fc = flight_state(alt_m, mach)
    _hdr(f"FLIGHT CONDITION  M = {mach:.2f}  Alt = {alt_m/1000:.1f} km "
         f"({alt_m*FT_PER_M:.0f} ft)")
    print(f"  Static  T  : {fc['T_K']:>9.2f} K     {fc['T_R']:>9.2f} °R")
    print(f"  Static  P  : {fc['P_Pa']:>9.1f} Pa    {fc['P_psia']:>9.4f} psia")
    print(f"  Total   Tt : {fc['Tt_K']:>9.2f} K     {fc['Tt_R']:>9.2f} °R")
    print(f"  Total   Pt : {fc['Pt_Pa']:>9.1f} Pa    {fc['Pt_psia']:>9.4f} psia")
    print(f"  Velocity   : {fc['V_ms']:>9.2f} m/s")
    print(f"  Speed snd  : {fc['a_ms']:>9.2f} m/s")


def print_presizing(iso: dict, M_exit: float, g: float) -> None:
    AR_check = area_ratio(M_exit, g)
    _hdr(f"ISENTROPIC PRE-SIZING  (γ = {g})  M_exit = {M_exit:.2f}")
    print(f"  Area ratio     : {AR_check:.5f}")
    print(f"  Throat area    : {iso['A_throat_m2']*1e4:>9.4f} cm²   "
          f"{iso['A_throat_m2']:>10.6f} m²")
    print(f"  Exit area      : {iso['A_exit_m2']*1e4:>9.4f} cm²   "
          f"{iso['A_exit_m2']:>10.6f} m²")
    print(f"  Throat T       : {iso['T_throat_K']:>9.2f} K")
    print(f"  Throat P       : {iso['P_throat_Pa']:>9.2f} Pa")
    print(f"  Throat V       : {iso['V_throat_ms']:>9.2f} m/s  (= sonic speed)")
    print(f"  Exit T         : {iso['T_exit_K']:>9.2f} K")
    print(f"  Exit P         : {iso['P_exit_Pa']:>9.4f} Pa")
    print(f"  Exit V         : {iso['V_exit_ms']:>9.2f} m/s")
    print(f"  Ideal Fg       : {iso['Fg_ideal_N']:>9.2f} N  (momentum only, no Cfg)")


def print_results(r: dict, label: str = "NOZZLE PERFORMANCE") -> None:
    _hdr(label)
    print(f"  ── Inlet ──────────────────────────────────────")
    print(f"  Total temp  Tt : {r['Tt_in_R']:>9.2f} °R   {r['Tt_in_K']:>9.2f} K")
    print(f"  Total pres  Pt : {r['Pt_in_psia']:>9.4f} psia  {r['Pt_in_Pa']:>9.1f} Pa")
    print(f"  NPR  Pt/Pamb   : {r['NPR']:>9.4f}")
    print(f"  ── Exit ───────────────────────────────────────")
    print(f"  Exit Mach      : {r['MN_exit']:>9.5f}")
    print(f"  Exit velocity  : {r['V_exit_fps']:>9.2f} ft/s  {r['V_exit_ms']:>9.2f} m/s")
    print(f"  Exit Ts        : {r['Ts_exit_R']:>9.2f} °R   {r['Ts_exit_K']:>9.2f} K")
    print(f"  Exit Ps        : {r['Ps_exit_psia']:>9.4f} psia  {r['Ps_exit_Pa']:>9.2f} Pa")
    print(f"  Ambient Ps     : {r['Ps_amb_psia']:>9.4f} psia  {r['Ps_amb_Pa']:>9.2f} Pa")
    print(f"  Expansion      : {r['expansion']}")
    print(f"  ── Geometry ───────────────────────────────────")
    print(f"  Throat area A* : {r['A_throat_in2']:>9.4f} in²   {r['A_throat_m2']*1e4:>9.4f} cm²")
    print(f"  Exit area      : {r['A_exit_in2']:>9.4f} in²   {r['A_exit_m2']*1e4:>9.4f} cm²")
    print(f"  Area ratio A/A*: {r['area_ratio']:>9.5f}")
    print(f"  ── Performance ────────────────────────────────")
    print(f"  Gross thrust Fg: {r['Fg_lbf']:>9.2f} lbf   {r['Fg_N']:>9.2f} N")
    print(f"  Thrust coeff CF: {r['CF']:>9.5f}")
    print(f"  Cfg (loss)     : {r['Cfg']:>9.4f}")
    print(f"  ── Start Check ────────────────────────────────")
    if r["nozzle_started"]:
        print(f"  ✓ Nozzle STARTED (supersonic throughout diverging section)")
        print(f"    M after N.S. at exit plane : {r['M_after_normal_shock']:.4f} (not physical)")
        print(f"    Pt loss if N.S. at exit    : {r['Pt_loss_normal_shock']*100:.2f}%")
    else:
        print(f"  ✗ Nozzle NOT STARTED (M_exit = {r['MN_exit']:.4f} < 1)")


# ============================================================================
# Design-point solve
# ============================================================================

def run_design(Tt_R: float, Pt_psia: float, W_lbms: float, FAR: float,
               MN_exit: float, Cfg: float, Ps_exhaust_psia: float,
               verbose: bool = True) -> tuple:
    """
    Run the pyCycle design-point solve.

    Returns
    -------
    prob    : solved om.Problem (kept alive for geometry extraction)
    results : dict from get_results()
    """
    prob = build_problem(design=True, record_file="nozzle_design.sql")

    set_inlet_conditions(prob, Tt_R, Pt_psia, W_lbms, FAR)
    prob.set_val("nozzle.MN",          MN_exit)
    prob.set_val("nozzle.Cfg",         Cfg)
    prob.set_val("nozzle.Ps_exhaust",  Ps_exhaust_psia, units="psia")

    if verbose:
        print("\n  Running pyCycle design-point solve...")

    prob.run_model()

    r = get_results(prob)
    if verbose:
        print_results(r, label="DESIGN-POINT NOZZLE PERFORMANCE")

    return prob, r


# ============================================================================
# Off-design solve (fixed geometry from design point)
# ============================================================================

def run_offdesign(design_prob: om.Problem,
                  Tt_R: float, Pt_psia: float, W_lbms: float, FAR: float,
                  Ps_exhaust_psia: float, Cfg: float,
                  label: str = "OFF-DESIGN") -> dict:
    """
    Run one off-design solve with fixed nozzle geometry from the design problem.

    The throat area (A*) and exit area are transferred from the design solve
    so the nozzle operates at the correct geometry.
    """
    prob = build_problem(design=False)

    # Transfer fixed geometry from design solve
    A_throat_in2 = float(design_prob.get_val(
        "nozzle.Throat:Fl_O:stat:area", units="inch**2"))
    A_exit_in2   = float(design_prob.get_val(
        "nozzle.Fl_O:stat:area", units="inch**2"))

    prob.set_val("nozzle.area_Throat", A_throat_in2, units="inch**2")
    prob.set_val("nozzle.area_exit",   A_exit_in2,   units="inch**2")

    set_inlet_conditions(prob, Tt_R, Pt_psia, W_lbms, FAR)
    prob.set_val("nozzle.Cfg",         Cfg)
    prob.set_val("nozzle.Ps_exhaust",  Ps_exhaust_psia, units="psia")

    prob.run_model()
    r = get_results(prob)
    prob.cleanup()

    return r


# ============================================================================
# Sweeps
# ============================================================================

def sweep_backpressure(design_prob: om.Problem, design_results: dict,
                        base_Ps_psia: float,
                        Tt_R: float, Pt_psia: float,
                        W_lbms: float, FAR: float, Cfg: float) -> None:
    """
    Sweep ambient back-pressure from 50 % to 200 % of design ambient.
    Shows how expansion state evolves as altitude changes off-design.
    """
    _hdr("BACK-PRESSURE SWEEP  (fixed geometry, varying altitude / Ps_amb)")
    ratios = [0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 2.00]
    print(f"  {'Ps_amb/Ps_des':>14}  {'Ps_amb[psia]':>13}  "
          f"{'MN_exit':>8}  {'Fg[N]':>9}  {'CF':>7}  {'Expansion'}")
    print(f"  {THIN_DIV}")
    for ratio in ratios:
        Ps = base_Ps_psia * ratio
        try:
            r = run_offdesign(design_prob, Tt_R, Pt_psia, W_lbms, FAR, Ps, Cfg)
            exp_short = r["expansion"].split("(")[0].strip()[:20]
            print(f"  {ratio:>14.2f}  {Ps:>13.4f}  "
                  f"{r['MN_exit']:>8.4f}  {r['Fg_N']:>9.2f}  "
                  f"{r['CF']:>7.5f}  {exp_short}")
        except Exception as e:
            print(f"  {ratio:>14.2f}  {Ps:>13.4f}  SOLVER FAILED: {e}")


def sweep_inlet_Tt(design_prob: om.Problem,
                    base_Tt_R: float, Pt_psia: float,
                    W_lbms: float, FAR: float, Cfg: float,
                    Ps_exhaust_psia: float) -> None:
    """
    Sweep nozzle inlet total temperature (Tt) to simulate throttle / combustor
    exit temperature variation.  Geometry stays fixed from design point.

    In a ramjet, Tt is driven by the fuel-to-air ratio (FAR). Here we vary
    Tt directly to decouple the thermodynamic effect from the combustor model.
    """
    _hdr("INLET TEMPERATURE (Tt) SWEEP  (fixed geometry, fixed Pt)")
    Tt_range = [base_Tt_R * f for f in [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]]
    print(f"  {'Tt [°R]':>9}  {'Tt [K]':>8}  {'MN_exit':>8}  "
          f"{'V_exit [m/s]':>13}  {'Fg [N]':>9}  {'CF':>7}")
    print(f"  {THIN_DIV}")
    for Tt in Tt_range:
        try:
            r = run_offdesign(design_prob, Tt, Pt_psia, W_lbms, FAR,
                              Ps_exhaust_psia, Cfg)
            print(f"  {Tt:>9.1f}  {Tt*K_PER_R:>8.1f}  {r['MN_exit']:>8.4f}  "
                  f"{r['V_exit_ms']:>13.2f}  {r['Fg_N']:>9.2f}  {r['CF']:>7.5f}")
        except Exception as e:
            print(f"  {Tt:>9.1f}  SOLVER FAILED: {e}")


def sweep_inlet_Pt(design_prob: om.Problem,
                    Tt_R: float, base_Pt_psia: float,
                    W_lbms: float, FAR: float, Cfg: float,
                    Ps_exhaust_psia: float) -> None:
    """
    Sweep nozzle inlet total pressure (Pt) to simulate varying inlet recovery.
    Lower Pt = worse inlet shock system (higher flight Mach off-design).
    """
    _hdr("INLET PRESSURE (Pt) SWEEP  (fixed geometry, fixed Tt)")
    Pt_range = [base_Pt_psia * f for f in [0.50, 0.65, 0.80, 1.00, 1.20]]
    print(f"  {'Pt [psia]':>10}  {'NPR':>8}  {'MN_exit':>8}  "
          f"{'Fg [N]':>9}  {'CF':>7}  {'Expansion'}")
    print(f"  {THIN_DIV}")
    for Pt in Pt_range:
        try:
            r = run_offdesign(design_prob, Tt_R, Pt, W_lbms, FAR,
                              Ps_exhaust_psia, Cfg)
            exp_short = r["expansion"].split("(")[0].strip()[:20]
            print(f"  {Pt:>10.4f}  {r['NPR']:>8.4f}  {r['MN_exit']:>8.4f}  "
                  f"{r['Fg_N']:>9.2f}  {r['CF']:>7.5f}  {exp_short}")
        except Exception as e:
            print(f"  {Pt:>10.4f}  SOLVER FAILED: {e}")


def sweep_Cfg(design_prob: om.Problem,
               Tt_R: float, Pt_psia: float,
               W_lbms: float, FAR: float,
               Ps_exhaust_psia: float) -> None:
    """
    Sweep gross thrust coefficient Cfg to quantify the impact of nozzle
    efficiency losses (divergence, viscous friction, shock inside nozzle).
    """
    _hdr("Cfg SENSITIVITY SWEEP  (fixed geometry)")
    Cfg_range = [0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000]
    print(f"  {'Cfg':>7}  {'Fg [N]':>9}  {'CF':>7}  {'ΔFg vs ideal [%]':>18}")
    print(f"  {THIN_DIV}")
    Fg_ideal = None
    rows = []
    for Cfg in Cfg_range:
        try:
            r = run_offdesign(design_prob, Tt_R, Pt_psia, W_lbms, FAR,
                              Ps_exhaust_psia, Cfg)
            rows.append((Cfg, r["Fg_N"], r["CF"]))
            if Cfg == 1.000:
                Fg_ideal = r["Fg_N"]
        except Exception as e:
            print(f"  {Cfg:.4f}  SOLVER FAILED: {e}")
    if Fg_ideal is None and rows:
        Fg_ideal = rows[-1][1]
    for Cfg, Fg_N, CF in rows:
        delta = (Fg_N - Fg_ideal) / Fg_ideal * 100 if Fg_ideal else 0
        print(f"  {Cfg:>7.4f}  {Fg_N:>9.2f}  {CF:>7.5f}  {delta:>+18.3f}%")


def sweep_exit_mach_geometry(Tt_R: float, Pt_psia: float,
                               W_lbms: float, FAR: float,
                               Cfg: float, Ps_exhaust_psia: float) -> None:
    """
    Design a nozzle for each target exit Mach number and compare geometry
    and ideal performance.  Each case is a fresh design-point solve.
    Useful for selecting the optimal nozzle expansion ratio.
    """
    _hdr("EXIT MACH / AREA RATIO TRADE  (separate design solves per Mach)")
    MN_range = [3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0]
    print(f"  {'MN_exit':>8}  {'AR (pyCycle)':>14}  {'AR (isent.)':>12}  "
          f"{'A* [cm²]':>10}  {'A_exit [cm²]':>13}  {'Fg [N]':>9}  {'CF':>7}")
    print(f"  {THIN_DIV}")
    for MN in MN_range:
        try:
            prob = build_problem(design=True)
            set_inlet_conditions(prob, Tt_R, Pt_psia, W_lbms, FAR)
            prob.set_val("nozzle.MN",         MN)
            prob.set_val("nozzle.Cfg",        Cfg)
            prob.set_val("nozzle.Ps_exhaust", Ps_exhaust_psia, units="psia")
            prob.run_model()
            r  = get_results(prob)
            AR_iso = area_ratio(MN, g=1.28)
            print(f"  {MN:>8.1f}  {r['area_ratio']:>14.5f}  {AR_iso:>12.5f}  "
                  f"{r['A_throat_m2']*1e4:>10.4f}  {r['A_exit_m2']*1e4:>13.4f}  "
                  f"{r['Fg_N']:>9.2f}  {r['CF']:>7.5f}")
            prob.cleanup()
        except Exception as e:
            print(f"  {MN:>8.1f}  SOLVER FAILED: {e}")


# ============================================================================
# CLI + main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hypersonic Ramjet CD Nozzle Analysis — pyCycle + OpenMDAO")
    p.add_argument("--mach",       type=float, default=6.0,
                   help="Flight Mach number (default 6.0)")
    p.add_argument("--alt",        type=float, default=25908.0,
                   help="Altitude [m] (default 25908 = 85 000 ft)")
    p.add_argument("--Tt",         type=float, default=4200.0,
                   help="Nozzle inlet total temp [°R] (default 4200)")
    p.add_argument("--Pt",         type=float, default=70.0,
                   help="Nozzle inlet total pressure [psia] (default 70)")
    p.add_argument("--W",          type=float, default=150.0,
                   help="Mass flow rate [lbm/s] (default 150)")
    p.add_argument("--FAR",        type=float, default=0.04,
                   help="Fuel-to-air ratio (default 0.04)")
    p.add_argument("--MN-exit",    type=float, default=5.5,
                   help="Target exit Mach (design point, default 5.5)")
    p.add_argument("--Cfg",        type=float, default=0.985,
                   help="Nozzle gross thrust coefficient (default 0.985)")
    p.add_argument("--gamma",      type=float, default=1.28,
                   help="γ for isentropic pre-sizing only (default 1.28)")
    p.add_argument("--no-pycycle", action="store_true",
                   help="Skip pyCycle solve; run analytical pre-sizing only")
    p.add_argument("--no-sweeps",  action="store_true",
                   help="Skip all parameter sweeps after design point")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + DIVIDER)
    print("  HYPERSONIC RAMJET — CD NOZZLE ANALYSIS")
    print("  pyCycle + OpenMDAO | CEA Thermodynamics | Design + Off-Design")
    print(DIVIDER)

    # ── 1. Flight condition ───────────────────────────────────────────────
    print_atmosphere(args.alt, args.mach)
    fc = flight_state(args.alt, args.mach)
    Ps_amb_psia = fc["P_psia"]   # ambient back-pressure at design altitude

    # ── 2. Isentropic pre-sizing ──────────────────────────────────────────
    mdot_kgs = args.W / LBM_PER_KG
    Tt_K     = args.Tt * K_PER_R
    Pt_Pa    = args.Pt / PSI_PER_PA
    iso = isentropic_presizing(Tt_K, Pt_Pa, mdot_kgs, args.MN_exit, args.gamma)
    print_presizing(iso, args.MN_exit, args.gamma)

    # Exit here if pyCycle is not wanted
    if args.no_pycycle:
        print("\n  [--no-pycycle] Skipping pyCycle solve. Done.\n")
        return

    # ── 3. Design-point solve ─────────────────────────────────────────────
    design_prob, design_r = run_design(
        Tt_R            = args.Tt,
        Pt_psia         = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        MN_exit         = args.MN_exit,
        Cfg             = args.Cfg,
        Ps_exhaust_psia = Ps_amb_psia,
        verbose         = True,
    )

    if args.no_sweeps:
        design_prob.cleanup()
        print("\n  [--no-sweeps] Skipping off-design sweeps. Done.\n")
        return

    # ── 4. Back-pressure sweep ────────────────────────────────────────────
    sweep_backpressure(
        design_prob, design_r,
        base_Ps_psia    = Ps_amb_psia,
        Tt_R            = args.Tt,
        Pt_psia         = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        Cfg             = args.Cfg,
    )

    # ── 5. Inlet temperature sweep (throttle simulation) ──────────────────
    sweep_inlet_Tt(
        design_prob,
        base_Tt_R       = args.Tt,
        Pt_psia         = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        Cfg             = args.Cfg,
        Ps_exhaust_psia = Ps_amb_psia,
    )

    # ── 6. Inlet pressure sweep (varying inlet recovery) ──────────────────
    sweep_inlet_Pt(
        design_prob,
        Tt_R            = args.Tt,
        base_Pt_psia    = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        Cfg             = args.Cfg,
        Ps_exhaust_psia = Ps_amb_psia,
    )

    # ── 7. Cfg sensitivity ────────────────────────────────────────────────
    sweep_Cfg(
        design_prob,
        Tt_R            = args.Tt,
        Pt_psia         = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        Ps_exhaust_psia = Ps_amb_psia,
    )

    # ── 8. Exit Mach / area ratio geometry trade ──────────────────────────
    sweep_exit_mach_geometry(
        Tt_R            = args.Tt,
        Pt_psia         = args.Pt,
        W_lbms          = args.W,
        FAR             = args.FAR,
        Cfg             = args.Cfg,
        Ps_exhaust_psia = Ps_amb_psia,
    )

    design_prob.cleanup()
    print(f"\n  Analysis complete. Case data saved to nozzle_design.sql\n")


if __name__ == "__main__":
    main()
