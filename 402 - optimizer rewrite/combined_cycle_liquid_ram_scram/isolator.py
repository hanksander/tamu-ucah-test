"""
Isolator — duct between inlet exit and combustor entrance.

RAM mode:  Normal shock train → subsonic M3 ≈ 0.30–0.40
           The isolator's job is to shield the inlet from combustor pressure
           rise.  Modelled here as a single normal shock for simplicity.

SCRAM mode: Pseudo-shock / oblique shock train with small Pt loss.
            No mode change (flow stays supersonic through).
"""

from gas_dynamics import FlowState, normal_shock, isentropic_T, isentropic_P
from config import ISOLATOR_PT_RECOVERY_SCRAM


# Target subsonic Mach after isolator in ram mode.
# ~0.35 is representative of a well-designed subsonic diffuser exit.
_M3_RAM_TARGET = 0.35


def compute_isolator(state2: FlowState, mode: str = 'scram') -> FlowState:
    """
    Isolator loss model.

    RAM:  Normal shock at M2 → subsonic, then isentropic diffusion to M3_target.
          Tt is conserved (adiabatic).  Pt loss comes from the normal shock.

    SCRAM: Apply fixed Pt loss (shock train) with slight Mach drop.
           Tt conserved.

    Returns FlowState at combustor entrance (station 3).
    """
    M, T, P, Pt, Tt, gam, R = (state2.M, state2.T, state2.P,
                                 state2.Pt, state2.Tt,
                                 state2.gamma, state2.R)

    if mode == 'ram':
        # Normal shock
        M_ns, P3P2, T3T2, Pt3Pt2 = normal_shock(M, gam)
        Pt3 = Pt  * Pt3Pt2
        # Isentropic diffusion to subsonic target Mach
        M3  = min(M_ns, _M3_RAM_TARGET)   # can't accelerate subsonically here
        T3  = isentropic_T(Tt, M3, gam)
        P3  = isentropic_P(Pt3, M3, gam)

    else:  # scram
        M3   = M * 0.97             # slight Mach reduction from shock train
        Pt3  = Pt * ISOLATOR_PT_RECOVERY_SCRAM
        T3   = isentropic_T(Tt, M3, gam)
        P3   = isentropic_P(Pt3, M3, gam)

    return FlowState(M=M3, T=T3, P=P3, Pt=Pt3, Tt=Tt,
                     gamma=gam, R=R)
