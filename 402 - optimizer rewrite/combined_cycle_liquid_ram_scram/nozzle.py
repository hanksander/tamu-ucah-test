"""
Nozzle — isentropic expansion to ambient pressure.

Assumes perfect expansion (P9 = P0), which gives the ideal upper bound
on specific thrust.  The nozzle velocity coefficient η_n accounts for
viscous and divergence losses.

Thrust accounting
-----------------
Per unit mass of air (1D stream tube, momentum equation):

    F/ṁ_air = (1 + f)·V9 − V0     [N·s/kg_air]

where f = φ · f_stoich is the fuel-air ratio by mass.
The (P9−P0)·A9/ṁ term vanishes for perfect expansion.

Isp is based on fuel mass flow (propulsion convention):
    Isp = F_sp / (f · g₀)         [s]
"""

import numpy as np
from gas_dynamics import FlowState, isentropic_T, isentropic_M_from_Pt_P
from config import F_STOICH, ETA_NOZZLE

G0 = 9.80665   # standard gravity [m/s²]


def compute_nozzle(
    state4: FlowState,
    state0: FlowState,
    P0: float,
    phi: float,
    thermo,
    eta_n: float = ETA_NOZZLE,
) -> tuple[float, float, FlowState]:
    """
    Isentropic nozzle expansion (perfectly expanded, P9 = P0).

    Parameters
    ----------
    state4  : FlowState   Combustor exit / nozzle entrance (station 4)
    state0  : FlowState   Freestream (for V0)
    P0      : float       Ambient static pressure [Pa]
    phi     : float       Equivalence ratio
    thermo  : JP10Thermo  For exit gas properties at T9
    eta_n   : float       Nozzle velocity coefficient

    Returns
    -------
    F_sp   : float       Specific thrust F/ṁ_air [N·s/kg]
    Isp    : float       Specific impulse [s]
    state9 : FlowState   Nozzle exit state
    """
    gam4 = state4.gamma
    Tt4  = state4.Tt
    Pt4  = state4.Pt

    # Perfectly expanded exit: find M9 from Pt4/P0
    if Pt4 <= P0:
        # No thrust — engine is off or in drag
        return 0.0, 0.0, state4

    M9  = isentropic_M_from_Pt_P(Pt4 / P0, gam4)
    T9  = isentropic_T(Tt4, M9, gam4)

    # Exit gas properties at T9 with product composition
    gamma9 = thermo.gamma(T9, phi)
    R9     = thermo.R(T9, phi)

    # Ideal exit velocity, degraded by nozzle efficiency
    V9_ideal = M9 * (gam4 * R9 * T9) ** 0.5
    V9       = eta_n * V9_ideal

    # Specific thrust (per kg air, momentum equation)
    f    = phi * F_STOICH     # fuel-air mass ratio
    V0   = state0.V
    F_sp = (1.0 + f) * V9 - V0   # [N·s/kg_air]
    Isp  = F_sp / (f * G0)        # [s]

    state9 = FlowState(M=M9, T=T9, P=P0, Pt=Pt4,
                       Tt=Tt4, gamma=gamma9, R=R9)
    return F_sp, Isp, state9
