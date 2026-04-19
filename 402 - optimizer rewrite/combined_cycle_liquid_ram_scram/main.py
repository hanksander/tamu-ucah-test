import numpy as np
from ambiance import Atmosphere

from thermo       import get_thermo
from gas_dynamics import make_state
from inlet        import compute_inlet
from isolator     import compute_isolator
from combustor    import compute_combustor
from nozzle       import compute_nozzle
from config       import M_TRANSITION, A_CAPTURE, F_STOICH

# Initialise Cantera once at import time
_thermo = get_thermo()

AIR_GAMMA = 1.40
AIR_R     = 287.05   # J/(kg·K)
G0        = 9.80665  # standard gravity [m/s²]


def analyze(
    M0:          float,
    altitude:    float,
    phi:         float,
    ramp_angles: list[float] | None = None,
    verbose:     bool = False,
) -> dict:

    # ── Freestream ─────────────────────────────────────────────────────────────
    atm = Atmosphere(altitude)
    T0  = float(atm.temperature[0])
    P0  = float(atm.pressure[0])

    state0 = make_state(M0, T0, P0, gamma=AIR_GAMMA, R=AIR_R)

    # ── Mode selection: M_TRANSITION ceiling ─────────────────────────────────
    #
    # The RAM->SCRAM transition is governed by terminal normal-shock total-
    # pressure recovery, not by Rayleigh thermal choking.  As M0 increases,
    # the normal shock Pt recovery collapses (e.g. ~0.44 at M=4, ~0.03 at
    # M=6), making RAM mode thermodynamically untenable above M_TRANSITION.
    #
    # Rayleigh thermal choking (choked=True) must NOT be used to promote to
    # SCRAM.  The reason is that the criterion is inverted:
    #
    #   Tt3 = Tt0  proportional to  M0^2   (adiabatic inlet)
    #   Tt4/Tt3  =  (Tt3 + q_eff/cp) / Tt3  ->  1  as  M0 -> inf
    #   Rayleigh choke threshold  Tt*/Tt3 ~ 2.33  (constant, M3=0.35)
    #
    #   => choke fires at LOW M0 (RAM is correct there)
    #      choke clears at HIGH M0 (SCRAM is necessary there)
    #
    # Promoting to SCRAM on choke would produce SCRAM at M=4 and RAM at M=6,
    # which is exactly backwards.  choked=True is a valid operating point
    # (combustor at the Rayleigh limit, M4=1); it is not a mode-change signal.
    # The flag is preserved in the output as a diagnostic for the user.

    if M0 >= M_TRANSITION:
        mode           = 'scram'
        state2, eta_pt = compute_inlet(state0, ramp_angles, mode='scram')
        state3         = compute_isolator(state2, mode='scram')
        state4, choked = compute_combustor(state3, phi, _thermo, mode='scram')
    else:
        mode           = 'ram'
        state2, eta_pt = compute_inlet(state0, ramp_angles, mode='ram')
        state3         = compute_isolator(state2, mode='ram')
        state4, choked = compute_combustor(state3, phi, _thermo)

    # ── Nozzle ─────────────────────────────────────────────────────────────────
    F_sp, Isp, state9 = compute_nozzle(state4, state0, P0, phi, _thermo)

    # ── Mass flow rates ────────────────────────────────────────────────────────
    mdot_air  = state0.rho * state0.V * A_CAPTURE   # [kg/s]
    mdot_fuel = phi * F_STOICH * mdot_air            # [kg/s]
    thrust    = F_sp * mdot_air                      # [N]

    result = {
        'mode':      mode,
        'M0':        M0,
        'altitude':  altitude,
        'phi':       phi,
        'Isp':       Isp,
        'F_sp':      F_sp,
        'thrust':    thrust,
        'mdot_air':  mdot_air,
        'mdot_fuel': mdot_fuel,
        'choked':    choked,
        'eta_pt':    eta_pt,
        'T_stations':  {0: state0.T,  2: state2.T,  3: state3.T,
                        4: state4.T,  9: state9.T},
        'Tt_stations': {0: state0.Tt, 2: state2.Tt, 3: state3.Tt,
                        4: state4.Tt, 9: state9.Tt},
        'M_stations':  {0: state0.M,  2: state2.M,  3: state3.M,
                        4: state4.M,  9: state9.M},
        'Pt_stations': {0: state0.Pt, 2: state2.Pt, 3: state3.Pt,
                        4: state4.Pt, 9: state9.Pt},
        'states':    {0: state0, 2: state2, 3: state3, 4: state4, 9: state9},
    }

    if verbose:
        _print_cycle(result)

    return result


def mach_sweep(
    mach_range: np.ndarray,
    altitude:   float,
    phi:        float,
    **kwargs,
) -> list[dict]:
    """Run analyze() over a Mach array. Returns list of result dicts."""
    return [analyze(M, altitude, phi, **kwargs) for M in mach_range]


def _print_cycle(r: dict):
    """Pretty-print a single cycle result."""
    st = r['states']
    print(f"\n{'='*74}")
    print(f"  Mode: {r['mode'].upper()}  |  M0={r['M0']:.2f}  "
          f"|  alt={r['altitude']/1e3:.1f} km  |  φ={r['phi']:.2f}")
    print(f"{'='*74}")
    print(f"  {'Stn':>6}  {'M':>7}  {'Ts [K]':>8}  {'Tt [K]':>8}  "
          f"{'Ps [kPa]':>10}  {'Pt [kPa]':>10}")
    print(f"  {'-'*64}")
    labels = {0: 'Free', 2: 'Inlet', 3: 'Isol.', 4: 'Comb.', 9: 'Nozzle'}
    for stn, s in st.items():
        print(f"  {labels[stn]:>6}  {s.M:>7.3f}  {s.T:>8.1f}  {s.Tt:>8.1f}  "
              f"{s.P/1e3:>10.2f}  {s.Pt/1e3:>10.2f}")
    print(f"\n  Isp    = {r['Isp']:>8.1f} s")
    print(f"  F_sp   = {r['F_sp']:>8.1f} N·s/kg_air")
    print(f"  Thrust = {r['thrust']/1e3:>8.2f} kN      (A_cap = {A_CAPTURE:.2f} m²)")
    print(f"  ṁ_air  = {r['mdot_air']:>8.3f} kg/s")
    print(f"  ṁ_fuel = {r['mdot_fuel']*1e3:>8.3f} g/s   "
          f"(f = {r['phi']*F_STOICH:.5f})")
    print(f"  η_inlet = {r['eta_pt']:.4f}   choked = {r['choked']}")


if __name__ == '__main__':

    r = analyze(M0=5.0, altitude=20_000, phi=0.8, verbose=True)


