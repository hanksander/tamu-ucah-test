"""
Inlet — multi-ramp oblique shock compression.

RAM mode:  N oblique shocks + terminal normal shock → subsonic M2
SCRAM mode: N oblique shocks only → supersonic M2

Falls back to MIL-E-5007D Pt recovery if any shock detaches
(unstarted inlet — useful for plotting the full Mach range without crashing).
"""

from gas_dynamics import FlowState, make_state, oblique_shock, normal_shock, beta_from_theta
from gas_dynamics import isentropic_T, isentropic_P, pi_milspec
from config import INLET_RAMPS_DEG
import numpy as np


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


def inlet_geometry(
        M_design: float,
        ramp_angles_deg: list[float],
        H_capture: float,
        gamma: float = 1.4,
) -> dict:
    """
    Compute ramp corner positions enforcing the shock-on-lip condition.

    Design criterion
    ----------------
    At the design Mach every oblique shock must arrive at the cowl lip
    (x_lip, H_capture).  Shock k originates at the (k-1)-th corner
    P_{k-1} = (x_{k-1}, y_{k-1}) and travels in the lab frame at angle

        alpha_k  =  sum(theta_1 .. theta_{k-1})  +  beta_k

    where the first term is the cumulative flow deflection before ramp k
    and beta_k is the local weak-shock wave angle (from the theta-beta-M
    relation).  The constraint that this ray passes through (x_lip, H_capture)
    determines each corner position algebraically:

        (H_capture - y_{k-1}) / (x_lip - x_{k-1}) = tan(alpha_k)   ... shock ray
        y_k = y_{k-1} + (x_k - x_{k-1}) * tan(sum theta_1..k)      ... ramp surface

    Rearranging the two equations simultaneously for x_k:

        x_k = (H_capture - y_{k-1} + x_{k-1}*A - x_lip*B) / (A - B)

    where A = tan(sum theta_1..k)  and  B = tan(alpha_{k+1}).

    The first shock (k=1) from the ramp nose P_0 = (0, 0) sets x_lip:

        x_lip = H_capture / tan(alpha_1) = H_capture / tan(beta_1)

    Parameters
    ----------
    M_design        : float        Freestream Mach number at design point
    ramp_angles_deg : list[float]  External deflection angle of each ramp [deg]
    H_capture       : float        Cowl-lip to ramp-nose height [m]
    gamma           : float        Ratio of specific heats (default air 1.4)

    Returns
    -------
    dict with keys:
        'corners'        : list[(x,y)]   ramp nose + all kink points + duct entry
                           Length = n_ramps + 1  (first = (0,0), last = at x_lip)
        'x_lip'          : float         cowl lip x-position [m]
        'betas_deg'      : list[float]   shock wave angles at M_design [deg]
        'lab_angles_deg' : list[float]   lab-frame shock angles [deg]
        'throat_height'  : float         internal duct height at x_lip [m]
        'ramp_lengths'   : list[float]   horizontal projection of each ramp [m]

    Raises
    ------
    ValueError if any shock detaches at M_design.
    """
    n = len(ramp_angles_deg)
    thetas = [np.radians(a) for a in ramp_angles_deg]

    # ── Shock wave angles at design Mach (iterating local Mach through shocks) ──
    betas = []
    M_local = M_design
    for i, theta_deg in enumerate(ramp_angles_deg):
        beta = beta_from_theta(theta_deg, M_local, gamma)
        if beta is None:
            raise ValueError(
                f"Shock detaches at ramp {i + 1}: M={M_local:.3f}, θ={theta_deg}°")
        betas.append(beta)
        M2, *_ = oblique_shock(M_local, theta_deg, gamma)
        if M2 is None:
            raise ValueError(f"oblique_shock failed at ramp {i + 1}")
        M_local = M2

    # Lab-frame angle for shock k  =  cumulative deflection before ramp k  +  beta_k
    cum_before = [sum(thetas[:i]) for i in range(n)]  # [0, θ1, θ1+θ2, …]
    lab_angles = [cum_before[i] + betas[i] for i in range(n)]

    # ── Cowl-lip x from shock 1 (originates at ramp nose = (0, 0)) ──────────────
    # (H_capture - 0) / (x_lip - 0) = tan(lab_angles[0])
    x_lip = H_capture / np.tan(lab_angles[0])

    # ── Iteratively solve for each intermediate ramp corner ───────────────────────
    # Corner i is the kink at the end of ramp i / start of ramp i+1.
    # Shock i+1 from corner i must hit (x_lip, H_capture).
    corners = [(0.0, 0.0)]  # corner 0 = ramp nose

    for i in range(1, n):
        x_prev, y_prev = corners[-1]
        # A = slope of ramp i  (cumulative angle after i ramps)
        A = np.tan(sum(thetas[:i]))
        # B = lab-frame slope of shock i+1
        B = np.tan(lab_angles[i])
        # Derived formula (A != B because beta > 0):
        xi = (H_capture - y_prev + x_prev * A - x_lip * B) / (A - B)
        yi = y_prev + (xi - x_prev) * A
        corners.append((xi, yi))

    # ── Final corner: end of last ramp, placed at x = x_lip ─────────────────────
    x_prev, y_prev = corners[-1]
    slope_last = np.tan(sum(thetas))
    y_last = y_prev + (x_lip - x_prev) * slope_last
    corners.append((x_lip, y_last))

    throat_height = H_capture - y_last
    ramp_lengths = [corners[i + 1][0] - corners[i][0] for i in range(n)]

    return {
        'corners': corners,
        'x_lip': x_lip,
        'betas_deg': [np.degrees(b) for b in betas],
        'lab_angles_deg': [np.degrees(a) for a in lab_angles],
        'throat_height': throat_height,
        'ramp_lengths': ramp_lengths,
    }