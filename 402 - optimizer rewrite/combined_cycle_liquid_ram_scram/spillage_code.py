import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# =============================================================================
# Oblique shock relations
# =============================================================================
 
def oblique_shock(M1, theta, gamma=1.4):

    def tbm(beta):
        num = 2.0 / np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1.0)
        den = M1**2 * (gamma + np.cos(2.0 * beta)) + 2.0
        return np.arctan2(num, den) - theta  # <-- arctan2 avoids wrapping issues

    mu = np.arcsin(1.0 / M1)
    beta_max = np.pi / 2.0

    # Sample the interval to find a sign change
    betas = np.linspace(mu + 1e-9, beta_max - 1e-9, 2000)
    vals  = np.array([tbm(b) for b in betas])
    sign_changes = np.where(np.diff(np.sign(vals)))[0]

    if len(sign_changes) == 0:
        raise ValueError(
            f"No oblique shock solution: theta={np.degrees(theta):.2f}° exceeds "
            f"the maximum deflection angle for M1={M1}."
        )

    # Weak shock = first (lowest beta) sign change
    i = sign_changes[0]
    beta = brentq(tbm, betas[i], betas[i + 1])

    # Normal component of upstream Mach number
    Mn1 = M1 * np.sin(beta)

    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    Mn2_sq    = (Mn1**2 + 2.0 / gm1) / (2.0 * gamma / gm1 * Mn1**2 - 1.0)
    p2_p1     = (2.0 * gamma * Mn1**2 - gm1) / gp1
    rho2_rho1 = gp1 * Mn1**2 / (gm1 * Mn1**2 + 2.0)
    T2_T1     = p2_p1 / rho2_rho1

    p02_p01 = (
        ((gp1 * Mn1**2) / (gm1 * Mn1**2 + 2.0)) ** (gamma / gm1)
        * (gp1 / (2.0 * gamma * Mn1**2 - gm1)) ** (1.0 / gm1)
    )

    M2 = np.sqrt(Mn2_sq) / np.sin(beta - theta)

    return {
        "beta"      : beta,
        "M2"        : M2,
        "p2_p1"     : p2_p1,
        "T2_T1"     : T2_T1,
        "rho2_rho1" : rho2_rho1,
        "p02_p01"   : p02_p01,
    }


def conical_shock(M1, theta_c_deg, gamma=1.4, shock="weak"):
    """
    Conical shock via Taylor-Maccoll ODE integration.
 
    Parameters
    ----------
    M1          : freestream Mach number (>= 1)
    theta_c_deg : cone half-angle (degrees)
    gamma       : ratio of specific heats (default 1.4)
    shock       : "weak" (default) or "strong"
 
    Returns
    -------
    (M_surface, p2/p1)
        M_surface : Mach number at the cone surface
        p2/p1     : static pressure ratio across the shock
    """
    g = gamma
    theta_c  = np.radians(theta_c_deg)
    beta_min = np.arcsin(1.0 / M1) + 1e-6
 
    def oblique_shock_post(beta):
        Mn1   = M1 * np.sin(beta)
        Mn2sq = (1 + (g-1)/2 * Mn1**2) / (g * Mn1**2 - (g-1)/2)
        delta = np.arctan(2/np.tan(beta) * (M1**2*np.sin(beta)**2 - 1) /
                          (M1**2*(g + np.cos(2*beta)) + 2))
        M2    = np.sqrt(Mn2sq / np.sin(beta - delta)**2)
        p2p1  = 1 + 2*g/(g+1) * (Mn1**2 - 1)
        return M2, p2p1, delta
 
    def M_to_V(M):
        """Normalise speed by stagnation speed of sound: V' = V/a0."""
        return M / np.sqrt(1 + (g-1)/2 * M**2)
 
    def tm_ode(phi, y):
        Vr, Vt = y
        V2  = Vr**2 + Vt**2
        a2  = 1 - (g-1)/2 * V2
        dVr = Vt
        dVt = (Vr*Vt**2 - a2*(2*Vr + Vt/np.tan(phi))) / (a2 - Vt**2)
        return [dVr, dVt]
 
    def vphi_at_surface(beta):
        M2, _, delta = oblique_shock_post(beta)
        V   = M_to_V(M2)
        Vr0 =  V * np.cos(beta - delta)
        Vt0 = -V * np.sin(beta - delta)
        sol = solve_ivp(tm_ode, [beta, theta_c], [Vr0, Vt0],
                        method='RK45', max_step=1e-4, rtol=1e-9, atol=1e-12)
        return sol.y[1, -1]
 
    # Scan for sign changes (weak and strong shock brackets)
    n     = 100
    betas = [beta_min + i*(np.pi/2 - 1e-4 - beta_min)/n for i in range(n+1)]
    brackets = []
    v_prev = vphi_at_surface(betas[0])
    for i in range(1, len(betas)):
        v_curr = vphi_at_surface(betas[i])
        if v_prev * v_curr < 0:
            brackets.append((betas[i-1], betas[i]))
        v_prev = v_curr
 
    if not brackets:
        raise ValueError(
            f"No attached conical shock solution for M1={M1}, theta_c={theta_c_deg} deg. "
            "Cone angle may exceed the detachment limit.")
 
    if shock == "weak":
        bracket = brackets[0]
    elif shock == "strong":
        bracket = brackets[-1]
    else:
        raise ValueError("shock must be 'weak' or 'strong'")
 
    beta_sol = brentq(vphi_at_surface, bracket[0], bracket[1], xtol=1e-8, rtol=1e-8)
 
    M2_sh, p2p1, delta = oblique_shock_post(beta_sol)
    V   = M_to_V(M2_sh)
    Vr0 =  V * np.cos(beta_sol - delta)
    Vt0 = -V * np.sin(beta_sol - delta)
 
    sol  = solve_ivp(tm_ode, [beta_sol, theta_c], [Vr0, Vt0],
                     method='RK45', max_step=1e-4, rtol=1e-9, atol=1e-12)
    Vr_c = sol.y[0, -1]
    Vt_c = sol.y[1, -1]
    V2_c = Vr_c**2 + Vt_c**2
    a2_c = 1 - (g-1)/2 * V2_c
    M_surface = np.sqrt(V2_c / a2_c)
 
    return M_surface, p2p1, beta_sol
 
# =============================================================================
# Geometry / intersection
# =============================================================================


def shock_streamline_intersect(x_r, y_r, x_s, y_s, theta, beta):

    """
    Finds intersection point between the shock off the ramp, and the streamline on the edge of the capture zone.
 
    Parameters
    ----------
    x_r, y_r : float  — origin of the shock ray (on the ramp)       (angle = beta + theta)
    x_s, y_s : float  — origin of the streamline ray (from the previos point, or from the cowl lip point)  (angle = theta)
    theta     : float — flow angle in radians
    beta      : float — shock angle (deflection) in radians - find from theta beta mach relations
    """
    x = (y_s - y_r - np.tan(theta)*x_s + np.tan(beta + theta)*x_r)/(np.tan(theta+beta)- np.tan(theta))
    y = np.tan(theta)*(x - x_s) + y_s

    return (x, y)
 
def plot_intersection(x_r0, y_r0, beta_0, theta_0,
                      x_r1, y_r1, beta_1, theta_1,
                      x_r2, y_r2, beta_2, theta_2,
                      x_r3, y_r3,
                      x_c0, y_c0,
                      x_c1, y_c1,
                      x_c2, y_c2):

    # --- intersections ---
    x_int1, y_int1 = shock_streamline_intersect(x_r2, y_r2, x_c1,   y_c1,   theta_2, beta_2)
    x_int2, y_int2 = shock_streamline_intersect(x_r1, y_r1, x_int1, y_int1, theta_1, beta_1)
    x_int3, y_int3 = shock_streamline_intersect(x_r0, y_r0, x_int2, y_int2, theta_0, beta_0)

    def extend_to_x(x_start, y_start, angle, x_target):
        """Continue a ray from (x_start, y_start) at angle until x = x_target."""
        x_end = x_target
        y_end = y_start + np.tan(angle) * (x_target - x_start)
        return x_end, y_end

    # extend each shock past its intersection point to x_c1
    x_ext1, y_ext1 = extend_to_x(x_int1, y_int1, theta_2 + beta_2, x_c1)
    x_ext2, y_ext2 = extend_to_x(x_int2, y_int2, theta_1 + beta_1, x_c1)
    x_ext3, y_ext3 = extend_to_x(x_int3, y_int3, theta_0 + beta_0, x_c1)

    fig, ax = plt.subplots(figsize=(8, 7))

    # --- ramp geometry: r_0 → r_1 → r_2 → r_3 ---
    ax.plot([x_r0, x_r1, x_r2, x_r3], [y_r0, y_r1, y_r2, y_r3],
            color='black', linewidth=2.0, marker='o', markersize=8,
            label='Ramp geometry')

    # --- cowl geometry: c_1 → c_2 ---
    ax.plot([x_c1, x_c2], [y_c1, y_c2],
            color='black', linewidth=2.0, marker='o', markersize=8,
            label='Cowl geometry')

    # --- shock lines: solid r → intersection, dotted intersection → x_c1 ---
    ax.plot([x_r2, x_int1], [y_r2, y_int1],
            color='red', linewidth=1.8, label='Shock')
    ax.plot([x_int1, x_ext1], [y_int1, y_ext1],
            color='red', linewidth=1.8, linestyle='--')

    ax.plot([x_r1, x_int2], [y_r1, y_int2],
            color='red', linewidth=1.8)
    ax.plot([x_int2, x_ext2], [y_int2, y_ext2],
            color='red', linewidth=1.8, linestyle='--')

    ax.plot([x_r0, x_int3], [y_r0, y_int3],
            color='red', linewidth=1.8)
    ax.plot([x_int3, x_ext3], [y_int3, y_ext3],
            color='red', linewidth=1.8, linestyle='--')

    # --- streamlines ---
    ax.plot([x_c1,   x_int1], [y_c1,   y_int1],
            color='steelblue', linewidth=1.8, label='Streamline')
    ax.plot([x_int1, x_int2], [y_int1, y_int2],
            color='steelblue', linewidth=1.8)
    ax.plot([x_int2, x_int3], [y_int2, y_int3],
            color='steelblue', linewidth=1.8)
    ax.plot([x_c0, x_int3], [y_c0, y_int3],
            color='steelblue', linewidth=1.8)

    # --- intersection points ---
    ax.plot(x_int1, y_int1, '*', color="#23bb5d", markersize=14, zorder=6,
            label=f'int_1  ({x_int1:.3f}, {y_int1:.3f})')
    ax.plot(x_int2, y_int2, '*', color="#1FD12E", markersize=14, zorder=6,
            label=f'int_2  ({x_int2:.3f}, {y_int2:.3f})')
    ax.plot(x_int3, y_int3, '*', color="#80CF26", markersize=14, zorder=6,
            label=f'int_3  ({x_int3:.3f}, {y_int3:.3f})')

    # --- point labels ---
    offset = 0.015
    ax.annotate('$r_0$', (x_r0, y_r0), (x_r0 - offset, y_r0 - offset), fontsize=9)
    ax.annotate('$r_1$', (x_r1, y_r1), (x_r1 - offset, y_r1 - offset), fontsize=9)
    ax.annotate('$r_2$', (x_r2, y_r2), (x_r2 - offset, y_r2 - offset), fontsize=9)
    ax.annotate('$r_3$', (x_r3, y_r3), (x_r3 - offset, y_r3 - offset), fontsize=9)
    ax.annotate('$c_1$', (x_c1, y_c1), (x_c1 - offset, y_c1 - offset), fontsize=9)
    ax.annotate('$c_2$', (x_c2, y_c2), (x_c2 - offset, y_c2 - offset), fontsize=9)
    ax.annotate('$c_0$', (x_c0, y_c0), (x_c0 - offset, y_c0 - offset), fontsize=9)

    # --- reference axes ---
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)

    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('y  (increases downward)')
    ax.set_title('shock_streamline_intersect — three steps')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.4, alpha=0.4)

    plt.tight_layout()
    plt.savefig("inlet_plot.png", dpi=150, bbox_inches='tight')

def prandtl_meyer(M1, theta_deg, gamma=1.4):
    """
    Prandtl-Meyer expansion fan.
    Returns (M2, p2/p1) given upstream Mach and turning angle (degrees).
    """
    def nu(M):
        a = np.sqrt((gamma + 1) / (gamma - 1))
        return np.degrees(a * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1)))
 
    def p_ratio(M):
        return (1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1))
 
    nu2 = nu(M1) + theta_deg
 
    lo, hi = 1.0, 100.0
    for _ in range(1000):
        mid = (lo + hi) / 2
        if nu(mid) < nu2:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-8:
            break
    M2 = (lo + hi) / 2
 
    return M2, p_ratio(M2) / p_ratio(M1)


def print_report(output_path="inlet_analysis.txt",
                 # geometry
                 theta_0=None, theta_1=None, theta_2=None, theta_c=None,
                 alpha=None,
                 # effective turning angles (with AOA)
                 eff_theta_0=None, eff_theta_1=None, eff_theta_2=None,
                 # section 0 (conical forebody)
                 M_fb=None, beta_fb=None, P_fb=None,
                 # section 1
                 M1=None, beta_1=None, P1=None,
                 # section 2
                 M2=None, beta_2=None, P2=None,
                 # cowl
                 Pc=None,
                 # streamtube lengths
                 l1=None, l2=None, lc=None,
                 # spillage
                 D_spill=None, L_spill=None,
                 #eff capture area
                 cap_height=None, cap_area=None):

    SEP  = "=" * 52
    sep  = "-" * 52

    def deg(r):
        return f"{np.degrees(r):>10.4f} deg"
    def flt(v, unit=""):
        return f"{v:>14.6f} {unit}".rstrip()

    lines = [
        SEP,
        "  INLET ANALYSIS — SUMMARY REPORT",
        SEP,
        "",
        "GEOMETRIC TURNING ANGLES",
        sep,
        f"  theta_0   (ramp section 0) : {deg(theta_0)}",
        f"  theta_1   (ramp section 1) : {deg(theta_1)}",
        f"  theta_2   (ramp section 2) : {deg(theta_2)}",
        f"  theta_c   (cowl)           : {deg(theta_c)}",
        "",
        "ANGLE OF ATTACK",
        sep,
        f"  alpha                      : {deg(alpha)}",
        "",
        "EFFECTIVE TURNING ANGLES  (geometric + AoA)",
        sep,
        f"  eff theta_0                : {deg(eff_theta_0)}",
        f"  eff theta_1  (delta_1-0)   : {deg(eff_theta_1)}",
        f"  eff theta_2  (delta_2-1)   : {deg(eff_theta_2)}",
        "",
        "FLOW CONDITIONS BY SECTION",
        sep,
        "  Section 0 — Conical forebody shock",
        f"    beta_fb  : {deg(beta_fb)}",
        f"    M_fb     : {flt(M_fb)}",
        f"    P_fb     : {flt(P_fb, 'Pa')}",
        "",
        "  Section 1 — 1st oblique shock",
        f"    beta_1   : {deg(beta_1)}",
        f"    M1       : {flt(M1)}",
        f"    P1       : {flt(P1, 'Pa')}",
        "",
        "  Section 2 — 2nd oblique shock",
        f"    beta_2   : {deg(beta_2)}",
        f"    M2       : {flt(M2)}",
        f"    P2       : {flt(P2, 'Pa')}",
        "",
        "  Cowl surface",
        f"    Pc       : {flt(Pc, 'Pa')}",
        "",
        "CAPTURE AREA",
        sep,
        f"  cap_height                 : {flt(cap_height, 'm')}",
        f"  cap_area                   : {flt(cap_area, 'm^2')}",
        "",
        "STREAMTUBE SEGMENT LENGTHS",
        sep,
        f"  l1  (int1 -> int2)          : {flt(l1, 'm')}",
        f"  l2  (c1   -> int1)          : {flt(l2, 'm')}",
        f"  lc  (cowl segment)         : {flt(lc, 'm')}",
        "",
        "SPILLAGE FORCES",
        sep,
        f"  D_spill (drag)             : {flt(D_spill, 'N')}",
        f"  L_spill (lift)             : {flt(L_spill, 'N')}",
        "",
        SEP,
    
    ]

    report = "\n".join(lines)
    print(report)

    with open(output_path, "w") as f:       # "w" overwrites automatically
        f.write(report + "\n")

    print(f"\n  [Report written to '{output_path}']")




# =============================================================================
# Main
# =============================================================================
 
if __name__ == '__main__':

    # --- freestream ---
    M_inf = 4 # 4-5
    alpha = np.radians(2) # -2 to 4
    # Pfb = something
    P0 = 8850 # Pa (16-19km wanted)
    inlet_width = 0.3 #m

    # --- points ---
    x_r0, y_r0 = 0.0,      0.0
    x_r1, y_r1 = 0.643674, 0.090463
    x_r2, y_r2 = 0.877544, 0.146077
    x_r3, y_r3 = 1.370300, 0.310814
    x_c1, y_c1 = 1.326994, 0.440350
    x_c2, y_c2 = 1.579634, 0.434318

    print()
 
    # --- flow angles from ramp geometry ---
    theta_0 = np.arctan((y_r1 - y_r0) / (x_r1 - x_r0))
    theta_1 = np.arctan((y_r2 - y_r1) / (x_r2 - x_r1))
    theta_2 = np.arctan((y_r3 - y_r2) / (x_r3 - x_r2))
    theta_c = np.arctan((y_c2 - y_c1) / (x_c2 - x_c1))

 
    # --- oblique shock relations ---
    
    Mfb, p2_p1fb, betafb = conical_shock(M_inf, np.degrees(theta_0))
    
    shock_0 = oblique_shock(M_inf, theta_0+alpha)
    shock_1 = oblique_shock(Mfb, theta_1 - theta_0)
    shock_2 = oblique_shock(shock_1['M2'], theta_2 - theta_1)

 
    beta_0 = betafb
    beta_1 = shock_1['beta']
    beta_2 = shock_2['beta']
    betafb2d = shock_0['beta']

    x_int1, y_int1 = shock_streamline_intersect(x_r2, y_r2, x_c1,   y_c1,   theta_2, beta_2)
    x_int2, y_int2 = shock_streamline_intersect(x_r1, y_r1, x_int1, y_int1, theta_1, beta_1)
    x_int3, y_int3 = shock_streamline_intersect(x_r0, y_r0, x_int2, y_int2, theta_0, beta_0)

    

    theta_c_ex = theta_2 - theta_c
    
    piggy_term, Pc_P2 = prandtl_meyer(shock_2["M2"], np.degrees(theta_c_ex))

    piggy_term, chungus_term = prandtl_meyer(Mfb, np.degrees(theta_0 - theta_c))


    Pfb = P0*p2_p1fb
    Pfbc = Pfb*chungus_term


    P1 = Pfb*shock_1["p2_p1"]
    P2 = P1*shock_2["p2_p1"]
    Pc = Pc_P2 * P2

    
    l1 = np.sqrt((x_int2 - x_int1)**2 + (y_int2 - y_int1)**2)
    l2 = np.sqrt((x_c1 - x_int1)**2 + (y_c1 - y_int1)**2)
    lc= 0.5 ### get this

    D_spill = ((P1 - Pfb)*l1*np.sin(theta_1) + (P2 - Pfb)*l2*np.sin(theta_2) + (Pc - Pfbc)*lc*np.sin(theta_c))*inlet_width # N
    L_spill = ((P1 - Pfb)*l1*np.cos(theta_1) + (P2 - Pfb)*l2*np.cos(theta_2) + (Pc - Pfbc)*lc*np.cos(theta_c))*inlet_width


    fb_s2s = np.sqrt((x_int3)**2 + (y_int3)**2)
    cap_height = fb_s2s*np.sin(alpha+beta_0+theta_0)
    cap_area = cap_height * inlet_width  # m^2

    x_c0, y_c0 = cap_height*np.sin(alpha), cap_height*np.cos(alpha)

    plot_intersection(
        x_r0=x_r0, y_r0=y_r0, beta_0=beta_0, theta_0=theta_0,
        x_r1=x_r1, y_r1=y_r1, beta_1=beta_1, theta_1=theta_1,
        x_r2=x_r2, y_r2=y_r2, beta_2=beta_2, theta_2=theta_2,
        x_r3=x_r3, y_r3=y_r3,
        x_c0=x_c0, y_c0=y_c0,
        x_c1=x_c1, y_c1=y_c1,
        x_c2=x_c2, y_c2=y_c2,
    )



    print_report(
        output_path="inlet_analysis.txt",
        theta_0=theta_0, theta_1=theta_1, theta_2=theta_2, theta_c=theta_c,
        alpha=alpha,
        eff_theta_0=theta_0 + alpha,
        eff_theta_1=theta_1 - theta_0,
        eff_theta_2=theta_2 - theta_1,
        M_fb=Mfb,   beta_fb=beta_0, P_fb=Pfb,
        M1=shock_1['M2'], beta_1=beta_1, P1=P1,
        M2=shock_2['M2'], beta_2=beta_2, P2=P2,
        Pc=Pc,
        l1=l1, l2=l2, lc=lc,
        D_spill=D_spill, L_spill=L_spill,
        cap_height=cap_height, cap_area=cap_area,
    )