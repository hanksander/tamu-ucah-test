"""
Inlet — multi-ramp oblique shock compression.

RAM mode:  N oblique shocks + terminal normal shock → subsonic M2
SCRAM mode: N oblique shocks only → supersonic M2

Falls back to MIL-E-5007D Pt recovery if any shock detaches
(unstarted inlet — useful for plotting the full Mach range without crashing).
"""

from gas_dynamics import FlowState, make_state, oblique_shock, normal_shock
from gas_dynamics import isentropic_T, isentropic_P, pi_milspec
from config import INLET_RAMPS_DEG


def compute_inlet(
    state0: FlowState,
    ramp_angles: list[float] | None = None,
    mode: str = 'scram',
) -> tuple[FlowState, float]:
    """
    Process freestream through multi-ramp inlet.

    Parameters
    ----------
    state0      : FlowState  Freestream (station 0)
    ramp_angles : list[float] Deflection angles [deg]. None → config default.
    mode        : 'ram' | 'scram'

    Returns
    -------
    state2  : FlowState  Conditions at isolator entrance (station 2)
    eta_pt  : float      Overall Pt2/Pt0 (inlet total pressure recovery)
    """
    if ramp_angles is None:
        ramp_angles = INLET_RAMPS_DEG

    M, T, P, Pt, gam = (state0.M, state0.T, state0.P,
                         state0.Pt, state0.gamma)
    detached = False

    for theta in ramp_angles:
        M2, P2P1, T2T1, Pt2Pt1, _ = oblique_shock(M, theta, gam)
        if M2 is None:
            detached = True
            break
        M, T, P, Pt = M2, T*T2T1, P*P2P1, Pt*Pt2Pt1

    if detached:
        # Inlet unstarted — use MIL-SPEC Pt recovery as fallback
        eta = pi_milspec(state0.M)
        Pt  = state0.Pt * eta
        # Approximate post-shock Mach/T using a single normal shock
        M, _, T2T1, _ = normal_shock(state0.M, gam)
        T = state0.T * T2T1
        P = isentropic_P(Pt, M, gam)

    if mode == 'ram':
        # Terminal normal shock brings flow subsonic
        M2n, P2P1, T2T1, Pt2Pt1 = normal_shock(M, gam)
        M, T, P, Pt = M2n, T*T2T1, P*P2P1, Pt*Pt2Pt1

    fac  = 1.0 + (gam - 1.0) / 2.0 * M**2
    Tt2  = T * fac
    state2 = FlowState(M=M, T=T, P=P, Pt=Pt, Tt=Tt2,
                       gamma=gam, R=state0.R)
    return state2, Pt / state0.Pt
