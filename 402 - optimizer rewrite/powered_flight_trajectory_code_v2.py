import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

from combined_cycle_liquid_ram_scram import analyze
from combined_cycle_liquid_ram_scram import compute_inlet
from combined_cycle_liquid_ram_scram import freestream
from combined_cycle_liquid_ram_scram import FlowState

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
G0    = 9.80665        # m/s²
RGAS  = 287.058        # J/(kg·K)
GAMMA = 1.4
RE    = 6_371_000.0    # m

# HCM mass budget  (total 900 kg, integral booster design)
M0             = 1088.0    # kg  gross mass at F-35 release
M_PROP_BOOST   = 280.0    # kg  solid rocket propellant (boosts to M2.5+)
M_FUEL_SFRJ    = 130.0    # kg  HTPB solid fuel grain in SFRJ combustion chamber
M_STRUCT       = M0 - M_PROP_BOOST - M_FUEL_SFRJ   # 490 kg (airframe+inlet+nozzle+warhead)

# Aero reference
S_REF    = 0.10           # m²  body cross-section reference area
D_BODY   = 0.34          # m   body diameter  (matches S_REF = pi*D²/4)

# ── SFRJ hardware geometry ────────────────────────────────────────────────────
# Designed for M4 cruise at 12 km altitude
# Inlet: annular mixed-compression type (2-shock)
A_INLET   = 0.012         # m²  inlet capture area (M4 design point)
A_THROAT  = 0.008         # m²  nozzle throat area
AREA_RATIO = 6.5          # Ae/At  nozzle expansion ratio (optimised ~39 kft)

# ── SFRJ fuel grain (HTPB-based) ─────────────────────────────────────────────
RHO_FUEL   = 920.0        # kg/m³  HTPB density
A_BURN_0   = 0.18         # m²   initial fuel grain burning surface area
A_BURN_MIN = 0.04         # m²   minimum (near-burnout) burning area
# Saint-Robert regression law: r_dot [m/s] = a_reg * Pc[Pa]^n_reg
# Calibrated at design point: M4, h=12km, Pc≈0.7MPa, FAR≈0.06
A_REG      = 1.44e-5      # regression coefficient (HTPB, calibrated)
N_REG      = 0.35         # pressure exponent (typical HTPB: 0.3–0.4)

# ── SFRJ thermodynamic constants ─────────────────────────────────────────────
GAMMA_C    = 1.25          # ratio of specific heats in combustion gas (HTPB products)
CP_C       = 1800.0        # J/(kg·K)  combustion gas Cp
ETA_COMB   = 0.94          # combustion efficiency
H_FUEL     = 42.5e6        # J/kg  lower heating value of HTPB
FAR_DESIGN = 0.060         # design fuel-air ratio (slightly lean of stoich 0.068)

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD ATMOSPHERE (1976)
# ──────────────────────────────────────────────────────────────────────────────
def atmosphere(h_m):
    """Scalar atmosphere – returns (rho, P, T, a)."""
    h = float(np.clip(h_m, 0.0, 79_900.0))
    T0, P0 = 288.15, 101_325.0
    if h <= 11_000:
        T = T0 - 0.0065 * h
        P = P0 * (T / T0) ** 5.2561
    elif h <= 25_000:
        T = 216.65
        P = 22_632.1 * np.exp(-0.0001577 * (h - 11_000))
    elif h <= 47_000:
        T = 216.65 + 0.003 * (h - 25_000)
        P = 2_488.4 * (T / 216.65) ** (-34.164)
    else:
        T = 282.65 - 0.0028 * (h - 47_000)
        P = 141.3 * (T / 282.65) ** 17.082
    rho = P / (RGAS * T)
    a   = np.sqrt(GAMMA * RGAS * T)
    return rho, P, T, a

def atm_vec(h_arr):
    """Vectorised atmosphere."""
    h_arr = np.atleast_1d(np.asarray(h_arr, dtype=float))
    out = [atmosphere(hi) for hi in h_arr]
    return tuple(np.array([o[i] for o in out]) for i in range(4))

# ──────────────────────────────────────────────────────────────────────────────
# AERODYNAMICS (Aakash you can put the aerodatabase here)
# ──────────────────────────────────────────────────────────────────────────────
def cd0(M):
    M = np.asarray(M, dtype=float)
    return np.where(M < 0.8,  0.018 + 0.002 * M,
           np.where(M < 1.2,  0.018 + 0.050 * (M - 0.8),
           np.where(M < 2.0,  0.038 - 0.012 * (M - 1.2),
           np.where(M < 5.0,  0.028 - 0.003 * (M - 2.0),
                               0.019 + 0.001 * (M - 5.0)))))

def cl_alpha(M):
    """Lift-curve slope [1/deg]."""
    M = np.asarray(M, dtype=float)
    cla = np.where(M < 1.0,  0.045,
          np.where(M < 3.0,  0.045 - 0.008 * (M - 1.0),
                              0.029 - 0.002 * (M - 3.0)))
    return np.clip(cla, 0.010, 0.05)

def ki(M):
    return 0.12 + 0.04 * np.maximum(0.0, 3.0 - M)

def aero(M, alpha_deg, rho, V):
    q  = 0.5 * rho * V**2
    CL = cl_alpha(M) * alpha_deg
    CD = cd0(M) + ki(M) * CL**2
    L  = q * S_REF * CL
    D  = q * S_REF * CD
    return CL, CD, L, D, q

# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION
# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION — SOLID ROCKET BOOSTER
# ──────────────────────────────────────────────────────────────────────────────
def boost_thrust(M, h):
    """
    Solid rocket motor thrust [N].
    Vacuum thrust ~105 kN; corrected for ambient back-pressure.
    T = Cf(h) * Pc * At  with constant Pc (regressive grain approximated flat).
    """
    _, P0, _, _ = atmosphere(h)
    Pc_boost = 8.0e6              # N/m²  chamber pressure (8 MPa)
    At_boost = 0.012              # m²    booster nozzle throat
    Cf_vac   = 1.65               # vacuum thrust coefficient
    # Pressure thrust correction: subtract P0*Ae (over-expanded at altitude)
    Ae_boost = At_boost * 8.0    # expansion ratio 8
    T = Cf_vac * Pc_boost * At_boost - (P0 - 0.0) * Ae_boost
    return float(np.clip(T, 70_000.0, 115_000.0))

def boost_isp(M=None, h=None):
    """Solid rocket Isp — ~265 s vacuum, ~240 s sea-level."""
    return 265.0   # s  (vacuum)

# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION — SOLID FUEL RAMJET (SFRJ)
# Physics reference: Waltrup et al. (1976); Netzer & Gany (1993)
# ──────────────────────────────────────────────────────────────────────────────

def sfrj_inlet_recovery(M):
    """
    Total-pressure recovery η_r for a 2-shock (isentropic + normal)
    mixed-compression inlet. MIL-E-5007D correlation.
    """
    M = float(M)
    if M <= 1.0:
        return 1.0
    elif M <= 5.0:
        # Empirical fit to Goldsmith–Griggs data for conical inlets
        eta = 1.0 - 0.075 * (M - 1.0)**1.35
    else:
        eta = 0.6 - 0.1 * (M - 5.0)
    return float(np.clip(eta, 0.05, 1.0))

def sfrj_total_conditions(M, h):
    """
    Ram-air total temperature and total pressure entering combustion chamber.
    """
    rho, P0, T0, a = atmosphere(h)
    gm = GAMMA
    Tt2 = T0  * (1.0 + (gm - 1)/2 * M**2)       # freestream total temperature
    Pt2 = P0  * (1.0 + (gm - 1)/2 * M**2)**(gm/(gm-1))
    eta_r = sfrj_inlet_recovery(M)
    Pt3 = eta_r * Pt2                              # total pressure at inlet exit
    return Tt2, Pt3

def sfrj_chamber_pressure(mdot_air, Tt4, Pc_guess=1.5e6):
    """
    Iterate to find combustion chamber pressure Pc [Pa] given
    air massflow and chamber temperature.
    Mass-flow continuity through nozzle throat:
        mdot_total = Pc * At * sqrt(gamma_c / (R_c * Tt4)) * (2/(gamma_c+1))^((gamma_c+1)/(2*(gamma_c-1)))
    """
    gm = GAMMA_C
    R_c = 8314.0 / 29.5        # J/(kg·K)  combustion products mol wt ~29.5
    Gamma_fn = np.sqrt(gm) * (2.0/(gm+1))**((gm+1)/(2*(gm-1)))
    # Pc = mdot_total * sqrt(R_c * Tt4) / (A_THROAT * Gamma_fn)
    # mdot_total ≈ mdot_air + mdot_fuel; iterate once
    mdot_est = mdot_air * 1.05   # initial estimate (5 % fuel addition)
    Pc = mdot_est * np.sqrt(R_c * Tt4) / (A_THROAT * Gamma_fn)
    return float(np.clip(Pc, 0.1e6, 6.0e6))

def sfrj_fuel_regression(Pc, A_burn):
    """
    Saint-Robert (Vieille) law: r_dot [m/s] = a_reg * Pc^n_reg
    Fuel mass-flow: mdot_f = rho_fuel * r_dot * A_burn
    """
    r_dot  = A_REG * (Pc ** N_REG)       # m/s
    mdot_f = RHO_FUEL * r_dot * A_burn   # kg/s
    return mdot_f, r_dot

def sfrj_thrust_and_isp(M, h, A_burn, verbose=False):
    """
    Full SFRJ thermodynamic cycle — returns (T_net [N], Isp_fuel [s], mdot_f [kg/s], Pc [Pa]).

    Station numbering:
      0  Freestream
      2  Inlet exit / combustor entry (after inlet shock system)
      4  Combustor exit
      5  Nozzle exit

    Design point: M4, h=12 km → T_net≈3 kN, Isp(fuel)≈1160 s, Pc≈0.7 MPa
    Operational range: M2.5–M5.5, h=5–25 km
    """
    M = float(M)
    rho0, P0, T0, a0 = atmosphere(h)
    V0 = M * a0

    # Below SFRJ ignition Mach — no combustion
    if M < 2.0:
        return 0.0, 0.0, 0.0, 0.0

    # ── Inlet: 2-shock mixed-compression (isentropic + normal shock)
    gm  = GAMMA
    Tt2 = T0  * (1.0 + (gm - 1) / 2.0 * M**2)   # total temperature (conserved)
    Pt2 = P0  * (1.0 + (gm - 1) / 2.0 * M**2) ** (gm / (gm - 1))
    eta_r = sfrj_inlet_recovery(M)
    Pt3   = eta_r * Pt2                            # total pressure entering combustor

    # Air mass-flow captured by inlet
    mdot_air = rho0 * V0 * A_INLET

    # ── Combustion: fixed FAR schedule (slightly lean, lean-blow-out limited)
    # FAR ramps up from M2 light-off to design FAR at M3+
    FAR = FAR_DESIGN * np.clip((M - 1.8) / 0.8, 0.0, 1.0)
    Q_release = ETA_COMB * FAR * H_FUEL               # J per kg of air
    Tt4 = Tt2 + Q_release / (CP_C * (1.0 + FAR))
    Tt4 = float(np.clip(Tt4, Tt2, 3500.0))

    # ── Chamber pressure from nozzle throat continuity
    gm_c = GAMMA_C
    R_c  = 8314.0 / 29.5                              # combustion gas gas constant
    Gfn  = np.sqrt(gm_c) * (2.0 / (gm_c + 1)) ** ((gm_c + 1) / (2 * (gm_c - 1)))
    mdot_f_est = FAR * mdot_air
    mdot_total = mdot_air + mdot_f_est
    Pc = mdot_total * np.sqrt(R_c * Tt4) / (A_THROAT * Gfn)
    Pc = float(np.clip(Pc, 0.05e6, 5.0e6))

    # ── Fuel regression (Saint-Robert / Vieille law)
    mdot_f, r_dot = sfrj_fuel_regression(Pc, A_burn)

    # ── Converged FAR with actual regression rate
    FAR_actual = mdot_f / mdot_air if mdot_air > 1e-6 else FAR
    Q_act  = ETA_COMB * FAR_actual * H_FUEL
    Tt4    = float(np.clip(Tt2 + Q_act / (CP_C * (1.0 + FAR_actual)), Tt2, 3500.0))
    mdot_total = mdot_air + mdot_f
    Pc = float(np.clip(
        mdot_total * np.sqrt(R_c * Tt4) / (A_THROAT * Gfn), 0.05e6, 5.0e6))

    # ── Nozzle exit Mach (Newton-Raphson on isentropic area-Mach relation)
    def ar_eq(Me):
        return ((2 / (gm_c + 1)) * (1 + (gm_c - 1) / 2 * Me**2)) \
               ** ((gm_c + 1) / (2 * (gm_c - 1))) / Me - AREA_RATIO
    Me = 3.0
    for _ in range(20):
        f  = ar_eq(Me)
        df = (ar_eq(Me + 0.001) - ar_eq(Me - 0.001)) / 0.002
        Me = max(Me - f / (df + 1e-12), 1.01)

    Te  = Tt4 / (1.0 + (gm_c - 1) / 2.0 * Me**2)
    Ve  = Me * np.sqrt(gm_c * R_c * Te)              # nozzle exit velocity
    Ae  = A_THROAT * AREA_RATIO
    Pe  = Pc * (1.0 + (gm_c - 1) / 2.0 * Me**2) ** (-gm_c / (gm_c - 1))

    # ── Net thrust
    T_gross = mdot_total * Ve + (Pe - P0) * Ae       # gross thrust
    D_ram   = mdot_air * V0                           # inlet momentum (ram drag)
    T_net   = T_gross - D_ram

    # ── Fuel-based Isp (oxidiser is free ram air — only fuel mass expended)
    Isp_fuel = T_net / (mdot_f * G0) if mdot_f > 1e-6 else 0.0

    if verbose:
        print(f"  M={M:.2f} h={h/1000:.1f}km  "
              f"mdot_air={mdot_air:.2f} kg/s  mdot_f={mdot_f:.3f} kg/s  "
              f"FAR={FAR_actual:.4f}  Tt4={Tt4:.0f} K  Pc={Pc/1e6:.2f} MPa  "
              f"T_net={T_net:.0f} N  Isp={Isp_fuel:.0f} s")

    return (float(np.clip(T_net, -5000.0, 80_000.0)),
            float(np.clip(Isp_fuel, 0.0, 2000.0)),
            float(mdot_f),
            float(Pc))


def lfrj_performance(
        M: float,
        h: float,
        phi: float,
        ramp_angles: list[float] | None = None,
        verbose: bool = False,
) -> tuple[float, float, float, float]:
    result = analyze(M0=M, altitude=h, phi=phi,
                     ramp_angles=ramp_angles, verbose=False)

    T_net = float(result['thrust'])*1000
    Isp = float(result['Isp'])
    mdot_f = float(result['mdot_fuel'])

    Pt3 = float(result['Pt_stations'][3])
    M3 = float(result['M_stations'][3])
    Pc = Pt3 / (1.0 + (GAMMA - 1.0) / 2.0 * M3 ** 2) ** (GAMMA / (GAMMA - 1.0))
    Pc = Pc / 1000

    return T_net, Isp, mdot_f, Pc



def lfrj_pressure_recovery(M: float, altitude: float, gamma: float = 1.4, R: float = 287.0):

    T, P, rho = freestream(altitude)
    a = (gamma * R * T) ** 0.5
    Tt = T * (1 + (gamma - 1)/2 * M**2)
    Pt = P * (1 + (gamma - 1)/2 * M**2) ** (gamma/(gamma - 1))

    state =  FlowState(
        M=M,
        T=T,
        P=P,
        Pt=Pt,
        Tt=Tt,
        gamma=gamma,
        R=R
    )

    recovery_ratio = compute_inlet(state, verbose=False)[1]

    return recovery_ratio



def lfrj_pressure_recovery():

    inlet_exit, total_pressure_recovery = compute_inlet()
    return total_pressure_recovery

# ──────────────────────────────────────────────────────────────────────────────
# EQUATIONS OF MOTION
# State: [x_range (m), h (m), V (m/s), gamma (rad), m (kg), A_burn (m²)]
# Control: alpha (deg)
# A_burn is only active during 'cruise' (SFRJ); fixed in boost/descent.
# ──────────────────────────────────────────────────────────────────────────────
def eom(state, alpha_deg, phase):
    x_r, h, V, gamma, m, A_burn = state
    h      = float(np.clip(h,     50.0, 79_900.0))
    V      = float(np.clip(V,     20.0, 5_000.0))
    m      = float(np.clip(m,     M_STRUCT * 0.95, M0 + 10))
    A_burn = float(np.clip(A_burn, A_BURN_MIN, A_BURN_0 * 1.05))
    rho, _, _, a = atmosphere(h)
    M_num  = V / a
    CL, CD, L, D, q = aero(M_num, alpha_deg, rho, V)
    g      = G0 * (RE / (RE + h))**2
    alpha_r = np.deg2rad(alpha_deg)

    if phase == 'boost':
        T    = boost_thrust(M_num, h)
        isp  = boost_isp()
        mdot = -T / (isp * G0)
        dA   = 0.0                          # grain does not change during boost

    elif phase == 'cruise':
        T_net, isp, mdot_f, Pc = sfrj_thrust_and_isp(M_num, h, A_burn)
        T    = T_net
        mdot = -mdot_f                      # only fuel is consumed (no oxidiser)
        # Grain surface recedes: dA_burn/dt = -perimeter_factor * r_dot * (A_burn / A_burn_0)
        _, r_dot = sfrj_fuel_regression(Pc, A_burn)
        # Burning area decreases as web burns back (simplified linear taper)
        dA   = -r_dot * (A_burn / (A_BURN_0 * 0.5)) * 0.12  # m²/s

    else:   # descent — engine off
        T    = 0.0
        mdot = 0.0
        dA   = 0.0

    dx     = V * np.cos(gamma)
    dh     = V * np.sin(gamma)
    dV     = (T * np.cos(alpha_r) - D) / m - g * np.sin(gamma)
    dgamma = ((T * np.sin(alpha_r) + L) / (m * V)
              - (g / V - V / (RE + h)) * np.cos(gamma))

    return np.array([dx, dh, dV, dgamma, mdot, dA])

# ──────────────────────────────────────────────────────────────────────────────
# RK4 INTEGRATOR
# state = [x_range, h, V, gamma, m, A_burn]
# ──────────────────────────────────────────────────────────────────────────────
def simulate_phase(state0, phase, t_end, dt, ctrl_fn,
                   m_prop_budget, stop_fn=None):
    """Simulate one phase. state0 must have 6 elements."""
    t   = 0.0
    s   = np.array(state0, dtype=float)
    m_p = float(m_prop_budget)

    keys = ['t','x_range','h','V','gamma','m','M','q',
            'alpha','L','D','T','isp','ax','CL','CD','Pc','A_burn']
    traj = {k: [] for k in keys}

    def snap(t, s):
        x_r, h, V, gamma, m, A_b = s
        h   = max(h, 10.0); V = max(V, 10.0)
        A_b = np.clip(A_b, A_BURN_MIN, A_BURN_0 * 1.05)
        rho, _, _, a = atmosphere(h)
        M_n = V / a
        al  = ctrl_fn(t, s)
        CL, CD, L, D, q = aero(M_n, al, rho, V)
        if phase == 'boost':
            Tp = boost_thrust(M_n, h); isp = boost_isp(); Pc = 8.0e6
        elif phase == 'cruise':
            Tp, isp, _, Pc = sfrj_thrust_and_isp(M_n, h, A_b)
        else:
            Tp = 0.0; isp = 0.0; Pc = 0.0
        g   = G0 * (RE / (RE + h))**2
        axg = (Tp * np.cos(np.deg2rad(al)) - D) / m / G0
        for k, v in zip(keys,
            [t, x_r, h, V, gamma, m, M_n, q, al, L, D, Tp, isp, axg, CL, CD, Pc, A_b]):
            traj[k].append(float(v))

    snap(t, s)
    while t < t_end:
        if stop_fn and stop_fn(s):
            break
        al = ctrl_fn(t, s)
        k1 = eom(s,            al, phase)
        k2 = eom(s + dt/2*k1,  al, phase)
        k3 = eom(s + dt/2*k2,  al, phase)
        k4 = eom(s + dt*k3,    al, phase)
        ds = dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        if phase in ('boost', 'cruise'):
            m_p -= abs(ds[4])
            if m_p <= 0.0:
                break
        s_new    = s + ds
        s_new[4] = max(s_new[4], M_STRUCT * 0.95)
        s_new[2] = max(s_new[2], 20.0)
        s_new[5] = np.clip(s_new[5], A_BURN_MIN, A_BURN_0 * 1.05)
        t += dt; s = s_new
        snap(t, s)

    return {k: np.array(v) for k, v in traj.items()}

# ──────────────────────────────────────────────────────────────────────────────
# ALPHA SCHEDULES
# s = [x_range, h, V, gamma, m, A_burn]
# ──────────────────────────────────────────────────────────────────────────────
CRUISE_ALT = 12_000.0   # m  (~39,400 ft) — SFRJ optimal q/Mach band

def alpha_boost(t, s):
    """Climb to SFRJ light-off altitude, maximize acceleration."""
    _, h, V, gamma, m, A_b = s
    rho, _, _, a = atmosphere(max(h, 10.0))
    M = V / a
    h_err = CRUISE_ALT - h
    base  = np.clip(5.0 - 0.9*(M - 1.0), 1.5, 12.0)
    corr  = np.clip(0.003 * h_err, -4.0, 5.0)
    return float(np.clip(base + corr, -5.0, 15.0))

def alpha_cruise(t, s):
    """Lift = Weight trim AoA + gentle altitude hold at CRUISE_ALT."""
    _, h, V, gamma, m, A_b = s
    h   = max(h, 100.0)
    rho, _, _, a = atmosphere(h)
    M   = V / a
    q   = 0.5 * rho * V**2
    W   = m * G0 * (RE / (RE + h))**2
    cla = cl_alpha(M)
    al_trim = (W / (q * S_REF * cla)) if (q > 0 and cla > 0) else 3.0
    h_err   = CRUISE_ALT - h
    al_corr = np.clip(0.004 * h_err, -4.0, 5.0)
    return float(np.clip(al_trim + al_corr, 0.0, 12.0))

def alpha_descent(t, s):
    """Pull nose over to achieve ≥80° FPA at impact."""
    _, h, V, gamma, m, A_b = s
    gam_tgt = np.deg2rad(-85.0)
    err     = gam_tgt - gamma
    return float(np.clip(2.5 * np.rad2deg(err), -30.0, 5.0))

# ──────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
def run_trajectory():
    print("=" * 65)
    print("  HCM Trajectory Simulation")
    print("  Air-Launch: F-35 @ Mach 0.8, 35,000 ft")
    print("=" * 65)

    # Initial conditions  (6-state: x_range, h, V, gamma, m, A_burn)
    h0_ft  = 35_000.0
    h0_m   = h0_ft * 0.3048
    _, _, _, a0 = atmosphere(h0_m)
    V0     = 0.8 * a0
    gamma0 = np.deg2rad(2.0)
    state0 = [0.0, h0_m, V0, gamma0, M0, A_BURN_0]   # A_burn not used in boost

    print(f"\nLaunch : {h0_ft:,.0f} ft | Mach 0.8 | V={V0:.1f} m/s | m={M0:.0f} kg")
    print(f"Mass budget: Boost propellant {M_PROP_BOOST:.0f} kg | "
          f"SFRJ fuel grain {M_FUEL_SFRJ:.0f} kg | Structure {M_STRUCT:.0f} kg")

    # ── PHASE 1: SOLID ROCKET BOOST ─────────────────────────────────────────
    print("\n[Phase 1] Solid Rocket Boost ...")
    print(f"  Target: accelerate to M2.5+ for SFRJ light-off")
    tb = simulate_phase(
        state0, 'boost', t_end=90, dt=0.5,
        ctrl_fn=alpha_boost,
        m_prop_budget=M_PROP_BOOST,
        stop_fn=lambda s: s[4] <= M0 - M_PROP_BOOST + 0.5
    )
    sf_b = [tb[k][-1] for k in ['x_range','h','V','gamma','m','A_burn']]
    _, _, _, a_b = atmosphere(sf_b[1])
    M_boost = sf_b[2] / a_b
    print(f"  Burnout: M={M_boost:.2f} | h={sf_b[1]/0.3048:,.0f} ft | "
          f"range={sf_b[0]/1000:.1f} km | m={sf_b[4]:.0f} kg")
    print(f"  Peak thrust ≈ {np.max(tb['T'])/1000:.1f} kN")
    # Diagnose SFRJ at handoff conditions
    print(f"\n  SFRJ light-off check at M={M_boost:.2f}, h={sf_b[1]/1000:.1f} km:")
    sfrj_thrust_and_isp(M_boost, sf_b[1], A_BURN_0, verbose=True)

    # ── PHASE 2: SFRJ CRUISE ────────────────────────────────────────────────
    print(f"\n[Phase 2] Solid Fuel Ramjet (SFRJ) Cruise ...")
    print(f"  HTPB grain: A_burn_0={A_BURN_0:.3f} m², ρ_f={RHO_FUEL:.0f} kg/m³")
    sf_b[3] = np.deg2rad(1.0)   # level off slightly for cruise entry
    t_c0 = tb['t'][-1]
    tc = simulate_phase(
        sf_b, 'cruise', t_end=t_c0 + 900, dt=1.0,
        ctrl_fn=alpha_cruise,
        m_prop_budget=M_FUEL_SFRJ,
        stop_fn=lambda s: s[4] <= M_STRUCT + 2.0
    )
    tc['t'] += t_c0
    sf_c = [tc[k][-1] for k in ['x_range','h','V','gamma','m','A_burn']]
    M_cruise  = float(np.mean(tc['M']))
    q_cruise  = float(np.mean(tc['q']))
    Pc_cruise = float(np.mean(tc['Pc']))
    print(f"  Mean Mach      : {M_cruise:.2f}")
    print(f"  Mean altitude  : {np.mean(tc['h'])/0.3048:,.0f} ft")
    print(f"  Mean q         : {q_cruise/1000:.1f} kPa  ({q_cruise*0.020885:.0f} psf)")
    print(f"  Mean Pc        : {Pc_cruise/1e6:.2f} MPa")
    print(f"  Mean Isp(fuel) : {float(np.mean(tc['isp'][tc['isp']>0])):.0f} s")
    print(f"  Fuel consumed  : {sf_b[4] - sf_c[4]:.1f} kg / {M_FUEL_SFRJ:.0f} kg")
    print(f"  End range      : {sf_c[0]/1000:.1f} km")

    # ── PHASE 3: TERMINAL DESCENT ───────────────────────────────────────────
    print("\n[Phase 3] Unpowered Terminal Descent ...")
    sf_c[3] = np.deg2rad(-40.0)
    t_d0 = tc['t'][-1]
    td = simulate_phase(
        sf_c, 'descent', t_end=t_d0 + 150, dt=0.25,
        ctrl_fn=alpha_descent,
        m_prop_budget=0.0,
        stop_fn=lambda s: s[1] <= 10.0
    )
    td['t'] += t_d0
    sf_d = [td[k][-1] for k in ['x_range','h','V','gamma','m','A_burn']]
    _, _, _, a_i = atmosphere(max(sf_d[1], 10.0))
    M_impact   = sf_d[2] / a_i
    fpa_impact = abs(np.rad2deg(sf_d[3]))
    total_range = sf_d[0] / 1000.0
    print(f"  Impact Mach : {M_impact:.2f}")
    print(f"  Impact FPA  : {fpa_impact:.1f}° (constraint ≥80°)")
    print(f"  Total range : {total_range:.1f} km")

    # ── Constraint Check + System Features ───────────────────────────────────
    all_q = np.concatenate([tb['q'], tc['q'], td['q']])
    max_q = float(np.max(all_q))

    print("\n─────────────────────────────────────────")
    print("CONSTRAINT CHECK:")
    ok_cruise = M_cruise >= 4.0
    ok_Mi     = M_impact >= 2.0
    ok_fpa    = fpa_impact >= 80.0
    print(f"  Cruise Mach ≥ 4.0 : {M_cruise:.2f}  {'✓' if ok_cruise else '✗ VIOLATION'}")
    print(f"  Impact Mach ≥ 2.0 : {M_impact:.2f}  {'✓' if ok_Mi     else '✗ VIOLATION'}")
    print(f"  Impact FPA  ≥ 80° : {fpa_impact:.1f}°  {'✓' if ok_fpa    else '✗ VIOLATION'}")

    feats = dict(M_boost=M_boost, M_cruise=M_cruise,
                 q_cruise=q_cruise, max_q=max_q,
                 Pc_cruise=Pc_cruise,
                 M_impact=M_impact, fpa_impact=fpa_impact,
                 total_range=total_range)

    print("\n══════════ SYSTEM FEATURES ══════════")
    print(f"  Boost Mach Number    : {M_boost:.2f}")
    print(f"  Cruise Mach Number   : {M_cruise:.2f}")
    print(f"  Cruise Dyn. Pressure : {q_cruise/1000:.2f} kPa  ({q_cruise*0.020885:.1f} psf)")
    print(f"  Max  Dyn. Pressure   : {max_q/1000:.2f} kPa  ({max_q*0.020885:.1f} psf)")
    print(f"  SFRJ Chamber Pressure: {Pc_cruise/1e6:.2f} MPa  (cruise mean)")
    print("═════════════════════════════════════\n")

    return tb, tc, td, feats

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
PC = {'boost': '#e74c3c', 'cruise': '#2980b9', 'descent': '#27ae60'}

def do_plots(tb, tc, td, feats):
    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.46, wspace=0.36)

    def ax_style(sub):
        ax = fig.add_subplot(sub)
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.grid(color='#21262d', lw=0.7, ls='--')
        return ax

    lkw   = dict(color='#c9d1d9', fontsize=9)
    tkw   = dict(color='#f0f6fc', fontsize=10, fontweight='bold', pad=8)
    legkw = dict(fontsize=7, facecolor='#21262d', labelcolor='#c9d1d9', edgecolor='none')

    phases = [('boost', tb), ('cruise', tc), ('descent', td)]

    def triplot(ax, xkey, ykey, xscale=1, yscale=1, ylabel='', xlabel='', title=''):
        for name, tr in phases:
            ax.plot(tr[xkey]*xscale, tr[ykey]*yscale,
                    color=PC[name], lw=1.8, label=name.capitalize())
        ax.set_ylabel(ylabel, **lkw); ax.set_title(title, **tkw)
        if xlabel: ax.set_xlabel(xlabel, **lkw)

    # 1. Range vs Time
    ax = ax_style(gs[0, 0])
    triplot(ax, 't', 'x_range', yscale=1e-3, ylabel='Range [km]',
            xlabel='Time [s]', title='Range vs Time')
    ax.legend(**legkw)

    # 2. Mach vs Time
    ax = ax_style(gs[0, 1])
    triplot(ax, 't', 'M', ylabel='Mach', xlabel='Time [s]', title='Mach vs Time')
    ax.axhline(4.0, color='#f39c12', lw=1.2, ls=':', label='M=4 (min cruise)')
    ax.axhline(2.0, color='#e74c3c', lw=1.2, ls=':', label='M=2 (min impact)')
    ax.legend(**legkw)

    # 3. Altitude vs Range
    ax = ax_style(gs[1, 0])
    triplot(ax, 'x_range', 'h', xscale=1e-3, yscale=1e-3,
            ylabel='Altitude [km]', xlabel='Range [km]', title='Altitude vs Range')

    # 4. Dynamic Pressure vs Range
    ax = ax_style(gs[1, 1])
    triplot(ax, 'x_range', 'q', xscale=1e-3, yscale=1e-3,
            ylabel='Dyn. Pressure [kPa]', xlabel='Range [km]',
            title='Dynamic Pressure vs Range')
    ax.axhline(feats['max_q']/1e3, color='#e74c3c', lw=1, ls=':',
               label=f"Max q = {feats['max_q']/1e3:.1f} kPa")
    ax.legend(**legkw)

    # 5. Acceleration vs Range
    ax = ax_style(gs[2, 0])
    triplot(ax, 'x_range', 'ax', xscale=1e-3,
            ylabel='Axial Accel. [g]', xlabel='Range [km]',
            title='Acceleration vs Range')

    # 6. Weight vs Range
    ax = ax_style(gs[2, 1])
    for name, tr in phases:
        ax.plot(tr['x_range']*1e-3, tr['m']*G0/1e3,
                color=PC[name], lw=1.8, label=name.capitalize())
    ax.set_ylabel('Weight [kN]', **lkw)
    ax.set_xlabel('Range [km]', **lkw)
    ax.set_title('Weight vs Range', **tkw)
    ax.legend(**legkw)

    # ── Mach-sweep analytical curves ─────────────────────────────────────────
    M_sw  = np.linspace(0.5, 6.0, 300)
    h_ref = CRUISE_ALT   # SFRJ reference altitude

    def trim_alpha(M):
        rho, _, _, a = atmosphere(h_ref)
        V = M * a; q = 0.5*rho*V**2
        m_cr = M_STRUCT + M_FUEL_SFRJ * 0.5   # mid-cruise mass
        W    = m_cr * G0
        cla  = cl_alpha(M)
        al   = W / (q*S_REF*cla) if (q > 0 and cla > 0) else 0.0
        return float(np.clip(al, -2, 20))

    # Pre-compute SFRJ performance sweep
    sfrj_T  = []; sfrj_isp = []; sfrj_Pc = []
    for M in M_sw:
        T_s, isp_s, _, Pc_s = sfrj_thrust_and_isp(M, h_ref, A_BURN_0*0.7)
        sfrj_T.append(T_s); sfrj_isp.append(isp_s); sfrj_Pc.append(Pc_s)
    sfrj_T = np.array(sfrj_T); sfrj_isp = np.array(sfrj_isp)

    # 7. T/D vs Mach
    ax = ax_style(gs[3, 0])
    td_b = []; td_s = []
    for i, M in enumerate(M_sw):
        rho, _, _, a = atmosphere(h_ref)
        V  = M * a; q = 0.5*rho*V**2
        al = trim_alpha(M)
        CL = cl_alpha(M) * al
        CD = cd0(M) + ki(M)*CL**2
        D  = q*S_REF*CD
        Tb = boost_thrust(M, h_ref)
        Ts = sfrj_T[i]
        td_b.append(Tb/D if D > 0 else 0.0)
        td_s.append(Ts/D if D > 0 else 0.0)
    ax.plot(M_sw, td_b, color=PC['boost'],  lw=1.8, label='Solid Rocket (Boost)')
    ax.plot(M_sw, td_s, color=PC['cruise'], lw=1.8, label='SFRJ (Cruise)')
    ax.axhline(1.0, color='#8b949e', lw=0.8, ls='--', label='T/D = 1')
    ax.axvline(2.5, color='#f39c12', lw=0.8, ls=':', label='SFRJ light-off M2.5')
    ax.set_ylim(0, min(30, max(max(td_b), max(td_s)+0.1)*1.15))
    ax.set_xlabel('Mach', **lkw); ax.set_ylabel('T/D', **lkw)
    ax.set_title('T/D vs Mach', **tkw); ax.legend(**legkw)

    # 8. Isp vs Mach  (fuel-based Isp for SFRJ; propellant Isp for rocket)
    ax = ax_style(gs[3, 1])
    ax.plot(M_sw, [boost_isp() for _ in M_sw], color=PC['boost'],
            lw=1.8, ls='--', label='Solid Rocket (propellant Isp)')
    ax.plot(M_sw, sfrj_isp, color=PC['cruise'], lw=1.8, label='SFRJ (fuel-based Isp)')
    ax.axvline(2.5, color='#f39c12', lw=0.8, ls=':', label='SFRJ light-off M2.5')
    ax.set_xlabel('Mach', **lkw); ax.set_ylabel('Isp [s]', **lkw)
    ax.set_title('Specific Impulse vs Mach', **tkw)
    ax.set_ylim(0, 1800); ax.legend(**legkw)

    # 9. AoA vs Mach
    ax = ax_style(gs[4, 0])
    al_arr = [trim_alpha(M) for M in M_sw]
    ax.plot(M_sw, al_arr, color='#9b59b6', lw=1.8)
    ax.axhline(0, color='#8b949e', lw=0.6, ls='--')
    ax.set_xlabel('Mach', **lkw); ax.set_ylabel('Trim AoA [deg]', **lkw)
    ax.set_title(f'Trim AoA vs Mach (h={h_ref/1000:.0f} km)', **tkw)

    # 10. L/D vs Mach
    ax = ax_style(gs[4, 1])
    ld_arr = []
    for M, al in zip(M_sw, al_arr):
        CL = cl_alpha(M) * al
        CD = cd0(M) + ki(M)*CL**2
        ld_arr.append(CL/CD if CD > 0 else 0.0)
    ax.plot(M_sw, ld_arr, color='#f39c12', lw=1.8)
    ax.axhline(0, color='#8b949e', lw=0.6, ls='--')
    ax.set_xlabel('Mach', **lkw); ax.set_ylabel('L/D', **lkw)
    ax.set_title(f'L/D vs Mach (h={h_ref/1000:.0f} km)', **tkw)

    # Suptitle
    fig.suptitle(
        "HCM Trajectory Analysis  ─  Solid Rocket → SFRJ  |  F-35 Air-Launch @ M0.8 / 35,000 ft\n"
        f"Boost Mach: {feats['M_boost']:.2f}   "
        f"SFRJ Cruise Mach: {feats['M_cruise']:.2f}   "
        f"Cruise q: {feats['q_cruise']/1000:.1f} kPa   "
        f"Max q: {feats['max_q']/1000:.1f} kPa   "
        f"Range: {feats['total_range']:.0f} km",
        color='#f0f6fc', fontsize=11, fontweight='bold', y=0.997
    )

    out = r'C:\Users\romag\Documents\capstone shit\402 - Dymos Rewrite\plots.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Plots saved → {out}")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# DYMOS OPTIMAL CONTROL PROBLEM (3-phase)
# ──────────────────────────────────────────────────────────────────────────────
class BoostODE(om.ExplicitComponent):
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*','*',method='cs')
    def compute(self, inputs, outputs):
        h  = np.clip(inputs['h'],   500., 79_000.)
        V  = np.clip(inputs['V'],    50., 4_000.)
        gm = inputs['gamma']; m = np.clip(inputs['m'], M_STRUCT, M0+10)
        al = inputs['alpha']
        rho,_,_,a = atm_vec(h); M_n = V/a
        CL,CD,L,D,q = aero(M_n,al,rho,V)
        T  = np.array([boost_thrust(Mi,hi) for Mi,hi in zip(M_n,h)])
        isp= np.full_like(M_n, boost_isp()); g = G0*(RE/(RE+h))**2; ar = np.deg2rad(al)
        outputs['x_range_dot'] = V*np.cos(gm)
        outputs['h_dot']       = V*np.sin(gm)
        outputs['V_dot']       = (T*np.cos(ar)-D)/m - g*np.sin(gm)
        outputs['gamma_dot']   = ((T*np.sin(ar)+L)/(m*V) - (g/V - V/(RE+h))*np.cos(gm))
        outputs['m_dot']       = -T/(isp*G0)

class CruiseODE(om.ExplicitComponent):
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*','*',method='cs')
    def compute(self, inputs, outputs):
        h  = np.clip(inputs['h'],   500., 79_000.)
        V  = np.clip(inputs['V'],   100., 4_000.)
        gm = inputs['gamma']; m = np.clip(inputs['m'], M_STRUCT, M0+10)
        al = inputs['alpha']
        rho,_,_,a = atm_vec(h); M_n = V/a
        CL,CD,L,D,q = aero(M_n,al,rho,V)
        # SFRJ: use mean A_burn at mid-grain for Dymos ODE (scalar grain approximation)
        A_b_mid = A_BURN_0 * 0.65
        sfrj_res = [sfrj_thrust_and_isp(Mi, hi, A_b_mid) for Mi, hi in zip(M_n, h)]
        T   = np.array([r[0] for r in sfrj_res])
        isp = np.array([r[1] if r[1] > 0 else 1.0 for r in sfrj_res])
        mdot_f = np.array([r[2] for r in sfrj_res])
        g = G0*(RE/(RE+h))**2; ar = np.deg2rad(al)
        outputs['x_range_dot'] = V*np.cos(gm)
        outputs['h_dot']       = V*np.sin(gm)
        outputs['V_dot']       = (T*np.cos(ar)-D)/m - g*np.sin(gm)
        outputs['gamma_dot']   = ((T*np.sin(ar)+L)/(m*V) - (g/V - V/(RE+h))*np.cos(gm))
        outputs['m_dot']       = -mdot_f

class DescentODE(om.ExplicitComponent):
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*','*',method='cs')
    def compute(self, inputs, outputs):
        h  = np.clip(inputs['h'],  5., 79_000.)
        V  = np.clip(inputs['V'], 50.,  4_000.)
        gm = inputs['gamma']; m = np.clip(inputs['m'], M_STRUCT*0.9, M0+10)
        al = inputs['alpha']
        rho,_,_,a = atm_vec(h); M_n = V/a
        CL,CD,L,D,q = aero(M_n,al,rho,V)
        g = G0*(RE/(RE+h))**2; ar = np.deg2rad(al)
        outputs['x_range_dot'] = V*np.cos(gm)
        outputs['h_dot']       = V*np.sin(gm)
        outputs['V_dot']       = -D/m - g*np.sin(gm)
        outputs['gamma_dot']   = (L/(m*V) - (g/V - V/(RE+h))*np.cos(gm))
        outputs['m_dot']       = np.zeros(len(h))

def build_dymos_problem(h0_m, V0):
    p    = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    seg_b, seg_c, seg_d = 8, 15, 10

    boost = dm.Phase(ode_class=BoostODE,
                     transcription=dm.GaussLobatto(num_segments=seg_b, order=3))
    traj.add_phase('boost', boost)
    boost.set_time_options(fix_initial=True, initial_val=0.0,
                           duration_bounds=(20, 80), units='s')
    boost.add_state('x_range', fix_initial=True, rate_source='x_range_dot', lower=0, ref=1e5)
    boost.add_state('h',       fix_initial=True, rate_source='h_dot', lower=1000, upper=50000)
    boost.add_state('V',       fix_initial=True, rate_source='V_dot', lower=50, upper=5000)
    boost.add_state('gamma',   fix_initial=True, rate_source='gamma_dot', lower=-0.5, upper=1.5)
    boost.add_state('m',       fix_initial=True, rate_source='m_dot', lower=M_STRUCT, upper=M0)
    boost.add_control('alpha', lower=-10, upper=20, units='deg', ref=5.)
    boost.add_boundary_constraint('m', loc='final', lower=M0-M_PROP_BOOST, units='kg')

    cruise = dm.Phase(ode_class=CruiseODE,
                      transcription=dm.GaussLobatto(num_segments=seg_c, order=3))
    traj.add_phase('cruise', cruise)
    cruise.set_time_options(fix_initial=False, duration_bounds=(200, 1200), units='s')
    for st, lb, ub in [('x_range',0,5e6),('h',5000,40000),
                        ('V',500,5000),('gamma',-0.5,0.5),('m',M_STRUCT,M0)]:
        cruise.add_state(st, fix_initial=False, rate_source=f'{st}_dot', lower=lb, upper=ub)
    cruise.add_control('alpha', lower=-5, upper=15, units='deg')
    # Cruise Mach ≥4: V ≥ 4 * a_ref(12km) ≈ 4*340=1360 m/s
    cruise.add_boundary_constraint('V', loc='initial', lower=1360., units='m/s')
    cruise.add_boundary_constraint('m', loc='final', lower=M_STRUCT, units='kg')

    descent = dm.Phase(ode_class=DescentODE,
                       transcription=dm.GaussLobatto(num_segments=seg_d, order=3))
    traj.add_phase('descent', descent)
    descent.set_time_options(fix_initial=False, duration_bounds=(20, 200), units='s')
    for st, lb, ub in [('x_range',0,5e6),('h',0,50000),
                        ('V',100,5000),('gamma',-np.pi,0),('m',M_STRUCT*0.9,M0)]:
        descent.add_state(st, fix_initial=False, rate_source=f'{st}_dot', lower=lb, upper=ub)
    descent.add_control('alpha', lower=-30, upper=10, units='deg')
    descent.add_boundary_constraint('h',     loc='final', equals=0.,     units='m')
    descent.add_boundary_constraint('V',     loc='final', lower=680.,    units='m/s')   # Mach 2
    descent.add_boundary_constraint('gamma', loc='final', upper=np.deg2rad(-80.), units='rad')

    traj.link_phases(['boost','cruise'],  vars=['time','x_range','h','V','gamma','m'])
    traj.link_phases(['cruise','descent'],vars=['time','x_range','h','V','gamma','m'])

    descent.add_objective('x_range', loc='final', scaler=-1e-5)

    p.driver = om.pyOptSparseDriver(optimizer='SLSQP')
    p.driver.opt_settings['ACC'] = 1e-5
    p.setup(force_alloc_complex=True)

    # Seed initial guesses
    nn_b = seg_b*3+1; nn_c = seg_c*3+1; nn_d = seg_d*3+1
    p.set_val('traj.boost.t_duration', 55.0)
    p.set_val('traj.boost.states:h',     np.linspace(h0_m, CRUISE_ALT, nn_b))
    p.set_val('traj.boost.states:V',     np.linspace(V0, 1500, nn_b))
    p.set_val('traj.boost.states:gamma', np.linspace(0.05, 0.02, nn_b))
    p.set_val('traj.boost.states:m',     np.linspace(M0, M0-M_PROP_BOOST, nn_b))
    p.set_val('traj.cruise.t_duration',  600.0)
    p.set_val('traj.cruise.states:h',    np.full(nn_c, CRUISE_ALT))
    p.set_val('traj.cruise.states:V',    np.full(nn_c, 1500.0))
    p.set_val('traj.cruise.states:gamma',np.zeros(nn_c))
    p.set_val('traj.cruise.states:m',    np.linspace(M0-M_PROP_BOOST, M_STRUCT+10, nn_c))
    p.set_val('traj.descent.t_duration', 60.0)
    p.set_val('traj.descent.states:h',   np.linspace(CRUISE_ALT, 0, nn_d))
    p.set_val('traj.descent.states:V',   np.linspace(1500, 900, nn_d))
    p.set_val('traj.descent.states:gamma',np.linspace(-0.1, -1.5, nn_d))
    return p

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Physics-based simulation
    tb, tc, td, feats = run_trajectory()
    plot_file = do_plots(tb, tc, td, feats)

    if '--optimize' in sys.argv:
        print("\n[Dymos] Building 3-phase optimal control problem ...")
        h0_m = 35_000 * 0.3048
        _, _, _, a0 = atmosphere(h0_m)
        try:
            prob = build_dymos_problem(h0_m, 0.8 * a0)
            dm.run_problem(prob, run_driver=True, simulate=True)
            print("[Dymos] ✓ Optimization complete.")
        except Exception as e:
            print(f"[Dymos] Solver note: {e}")
    else:
        print("Tip: pass --optimize flag to run full Dymos/IPOPT optimal control solve.\n")

    print(f"\n✓ Plot file : {plot_file}")
