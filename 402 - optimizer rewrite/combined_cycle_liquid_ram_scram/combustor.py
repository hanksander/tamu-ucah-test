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

from scipy.optimize import brentq

from gas_dynamics import FlowState, rayleigh_exit, isentropic_T, isentropic_P
from pyc_config import F_STOICH, LHV_JP10, ETA_COMBUSTOR


def compute_combustor(
    state3: FlowState,
    phi: float,
    thermo,
    eta_c: float = ETA_COMBUSTOR,
    area_ratio: float = 1.0,
    mode: str = 'scram',
) -> tuple[FlowState, bool]:
    """
    Constant-area combustor with Rayleigh heat addition.

    Parameters
    ----------
    state3 : FlowState   Combustor entrance (station 3)
    phi    : float       Equivalence ratio
    thermo : JP10Thermo  Thermodynamic model
    eta_c  : float       Combustion efficiency
    area_ratio : float   Exit/inlet area ratio
    mode   : 'ram' | 'scram'

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
        xtol=0.1,
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
    supersonic = (mode == 'scram')
    M4, Pt4_Pt3, choked = rayleigh_exit(state3.M, Tt4 / Tt3,
                                         state3.gamma, supersonic=supersonic)

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
                                         gamma_avg, supersonic=supersonic)

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
