"""
Combustor — constant-area Rayleigh heat addition with JP-10 thermodynamics.

Energy balance
--------------
Heat added per kg of air (at combustion efficiency η_c):
    q = η_c · φ · f_stoich · LHV_JP10   [J/kg_air]

Total temperature rise (using air cp at T3 for the first estimate):
    Tt4 ≈ Tt3 + q / cp_air(T3)

Rayleigh flow then gives M4 and Pt4/Pt3 for the computed Tt4/Tt3.

Exit gas properties (γ4, R4) come from the JP10Thermo Cantera equilibrium
at the actual T4 and φ, capturing the real gas effect of dissociation.
"""

from gas_dynamics import FlowState, rayleigh_exit, isentropic_T, isentropic_P
from config import F_STOICH, LHV_JP10, ETA_COMBUSTOR


def compute_combustor(
    state3: FlowState,
    phi: float,
    thermo,
    eta_c: float = ETA_COMBUSTOR,
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
    mode   : 'ram' | 'scram'

    Returns
    -------
    state4 : FlowState   Combustor exit (station 4)
    choked : bool        True if thermal choking occurred (Tt4/Tt* ≥ 1)
    """
    # Heat addition per kg air
    q = eta_c * phi * F_STOICH * LHV_JP10

    # Total temperature rise via air cp (conservative first estimate)
    cp3  = thermo.cp(state3.T, phi=0.0)     # air cp at T3 [J/(kg·K)]
    Tt3  = state3.Tt
    Tt4  = Tt3 + q / cp3

    # Rayleigh flow to find M4 and Pt loss
    supersonic = (mode == 'scram')
    M4, Pt4_Pt3, choked = rayleigh_exit(state3.M, Tt4 / Tt3,
                                         state3.gamma, supersonic=supersonic)

    # Exit static conditions using inlet γ (good first-order approximation)
    Pt4 = state3.Pt * Pt4_Pt3
    T4  = isentropic_T(Tt4, M4, state3.gamma)

    # Refine γ and R using Cantera at actual exit conditions
    gamma4 = thermo.gamma(T4, phi)
    R4     = thermo.R(T4, phi)

    # One refinement pass with correct γ
    T4  = isentropic_T(Tt4, M4, gamma4)
    P4  = isentropic_P(Pt4, M4, gamma4)

    return FlowState(M=M4, T=T4, P=P4, Pt=Pt4, Tt=Tt4,
                     gamma=gamma4, R=R4), choked
