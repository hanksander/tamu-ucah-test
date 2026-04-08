import math
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.4
R = 287.05


# =========================
# Atmosphere
# =========================
def std_atmosphere_1976(h_m):
    """
    Standard atmosphere up to 25 km.
    Returns T [K], p [Pa], rho [kg/m^3], a [m/s]
    """
    g0 = 9.80665

    if h_m < 11000.0:
        T = 288.15 - 0.0065 * h_m
        p = 101325.0 * (T / 288.15) ** (g0 / (R * 0.0065))
    else:
        T = 216.65
        p11 = 22632.06
        p = p11 * math.exp(-g0 * (h_m - 11000.0) / (R * T))

    rho = p / (R * T)
    a = math.sqrt(GAMMA * R * T)
    print(f"Temp: {T}, Pressure: {p}, density: {rho}, speed of sound: {a}")
    return T, p, rho, a


# =========================
# Shock helpers
# =========================
def theta_beta_m_residual(beta, M, theta, gamma=GAMMA):
    num = 2.0 / math.tan(beta) * (M**2 * math.sin(beta)**2 - 1.0)
    den = M**2 * (gamma + math.cos(2.0 * beta)) + 2.0
    return math.tan(theta) - num / den


def solve_weak_beta(M, theta_rad, gamma=GAMMA):
    """
    Weak oblique-shock solution for beta [rad].
    Returns None if no attached weak solution exists.
    """
    if M <= 1.0 or theta_rad <= 0.0:
        return None

    mu = math.asin(1.0 / M)
    betas = np.linspace(mu + 1e-6, math.radians(89.9), 5000)
    vals = np.array([theta_beta_m_residual(b, M, theta_rad, gamma) for b in betas])

    roots = []
    for i in range(len(betas) - 1):
        f1 = vals[i]
        f2 = vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            roots.append(betas[i])
        elif f1 * f2 < 0.0:
            a = betas[i]
            b = betas[i + 1]
            for _ in range(80):
                c = 0.5 * (a + b)
                fc = theta_beta_m_residual(c, M, theta_rad, gamma)
                fa = theta_beta_m_residual(a, M, theta_rad, gamma)
                if fa * fc <= 0.0:
                    b = c
                else:
                    a = c
            roots.append(0.5 * (a + b))

    if not roots:
        return None

    return min(roots)


def normal_shock(M1, gamma=GAMMA):
    """
    Returns:
        M2, p2/p1, rho2/rho1, T2/T1, pt2/pt1
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    M1sq = M1**2

    p2_p1 = 1.0 + 2.0 * gamma / gp1 * (M1sq - 1.0)
    rho2_rho1 = gp1 * M1sq / (gm1 * M1sq + 2.0)
    T2_T1 = p2_p1 / rho2_rho1

    M2sq = (1.0 + 0.5 * gm1 * M1sq) / (gamma * M1sq - 0.5 * gm1)
    M2 = math.sqrt(M2sq)

    pt1_p1 = (1.0 + 0.5 * gm1 * M1sq) ** (gamma / gm1)
    pt2_p2 = (1.0 + 0.5 * gm1 * M2sq) ** (gamma / gm1)
    pt2_pt1 = (pt2_p2 * p2_p1) / pt1_p1

    return M2, p2_p1, rho2_rho1, T2_T1, pt2_pt1


def oblique_shock(M1, theta_deg, gamma=GAMMA):
    """
    Returns:
        beta_deg, M2, p2/p1, pt2/pt1
    """
    theta = math.radians(theta_deg)
    beta = solve_weak_beta(M1, theta, gamma)
    if beta is None:
        return None

    Mn1 = M1 * math.sin(beta)
    Mn2, p2_p1, _, _, pt2_pt1 = normal_shock(Mn1, gamma)
    M2 = Mn2 / math.sin(beta - theta)

    return math.degrees(beta), M2, p2_p1, pt2_pt1


def theta_max_attached(M, gamma=GAMMA):
    """
    Maximum attached-shock turning angle [deg] from the theta-beta-M relation.
    Used here as a proxy for separation limit.
    """
    if M <= 1.0:
        return None

    mu = math.asin(1.0 / M)
    betas = np.linspace(mu + 1e-4, math.radians(89.5), 8000)

    thetas = []
    for beta in betas:
        num = 2.0 / math.tan(beta) * (M**2 * math.sin(beta)**2 - 1.0)
        den = M**2 * (gamma + math.cos(2.0 * beta)) + 2.0
        tan_theta = num / den
        if tan_theta > 0.0:
            thetas.append(math.degrees(math.atan(tan_theta)))

    if not thetas:
        return None

    return max(thetas)


# =========================
# Geometry helpers
# =========================
def unit_from_angle_deg(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def total_temperature(T, M, gamma=GAMMA):
    return T * (1.0 + 0.5 * (gamma - 1.0) * M**2)


def total_pressure(p, M, gamma=GAMMA):
    return p * (1.0 + 0.5 * (gamma - 1.0) * M**2) ** (gamma / (gamma - 1.0))

# =========================
# Design function
# =========================
def design_2ramp_shock_matched_inlet(
    M0,
    altitude_m,
    alpha_deg,
    leading_edge_angle_deg,
    mdot_required,
    width_m,
    separation_margin=0.95,
    throat_area_factor=0.95,
    shock_focus_factor=1.25,
):
    """
    3-panel external-compression geometry:
      forebody -> ramp 1 -> ramp 2 -> angled cowl -> cowl shock -> throat

    This version:
      - sizes throat from post-cowl-shock Mach using the area-Mach relation
      - uses a sharp internal corner from Ramp 2 end to throat lower point
      - ignores shocks/expansions caused by that sharp internal corner
      - cowl is angled downward by the leading-edge angle
      - ramp shocks are focused on a point located shock_focus_factor times the
        opening-normal distance downstream along the Ramp 2 shock beyond the cowl lip
      - includes a Kantrowitz check as a yes/no flag only
      - reports a normal shock immediately after the cowl shock
    """
    if M0 <= 1.0:
        raise ValueError("M0 must be supersonic.")
    if mdot_required <= 0.0:
        raise ValueError("mdot_required must be positive.")
    if width_m <= 0.0:
        raise ValueError("width_m must be positive.")
    if not (0.0 < separation_margin < 1.0):
        raise ValueError("separation_margin must be between 0 and 1.")
    if not (0.0 < throat_area_factor < 1.0):
        raise ValueError("throat_area_factor must be between 0 and 1.")
    if shock_focus_factor <= 0.0:
        raise ValueError("shock_focus_factor must be positive.")

    def area_mach_ratio(M, gamma=GAMMA):
        return (1.0 / M) * (
            (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
        ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))

    def kantrowitz_contraction_ratio(M1, gamma=GAMMA):
        g = gamma
        term1 = ((g + 1) * M1**2) / ((g - 1) * M1**2 + 2)
        term2 = (g + 1) / (2 * g * M1**2 - (g - 1))
        term3 = ((1 + (g - 1) / 2 * M1**2) / ((g + 1) / 2)) ** (g / (g - 1))
        return (1 / M1) * term1 ** (1 / (g - 1)) * term2 ** (1 / (g - 1)) * term3

    T0, p0, rho0, a0 = std_atmosphere_1976(altitude_m)
    V0 = M0 * a0
    pt0 = total_pressure(p0, M0)
    Tt0 = total_temperature(T0, M0)

    # Required opening from freestream mass flux
    A_capture_required = mdot_required / (rho0 * V0)
    h_req_normal = A_capture_required / width_m

    # -------------------------------------------------
    # Forebody
    # -------------------------------------------------
    theta_fore = leading_edge_angle_deg
    dtheta_fore_eff = theta_fore + alpha_deg
    if dtheta_fore_eff <= 0.0:
        raise ValueError("Forebody effective turn must be positive.")

    theta_fore_eff_max = theta_max_attached(M0)
    if theta_fore_eff_max is None:
        raise ValueError("Could not determine attached-shock limit for forebody.")

    if dtheta_fore_eff >= separation_margin * theta_fore_eff_max:
        raise ValueError(
            f"Forebody effective turn ({dtheta_fore_eff:.3f} deg) exceeds "
            f"{separation_margin:.2f} of attached-shock limit ({theta_fore_eff_max:.3f} deg)."
        )

    shf = oblique_shock(M0, dtheta_fore_eff)
    if shf is None:
        raise ValueError("Forebody has no attached weak shock solution.")

    beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio = shf
    shock_fore_abs = beta_fore_rel

    # -------------------------------------------------
    # Ramp 1
    # -------------------------------------------------
    dtheta1_max = theta_max_attached(M_fore)
    if dtheta1_max is None:
        raise ValueError("Could not determine attached-shock limit for Ramp 1.")

    dtheta1 = separation_margin * dtheta1_max
    theta1 = theta_fore + dtheta1

    sh1 = oblique_shock(M_fore, dtheta1)
    if sh1 is None:
        raise ValueError("Ramp 1 has no attached weak shock solution.")

    beta1_rel, M1, p21, pt21 = sh1
    shock1_abs = theta_fore + beta1_rel

    # -------------------------------------------------
    # Ramp 2
    # -------------------------------------------------
    dtheta2_max = theta_max_attached(M1)
    if dtheta2_max is None:
        raise ValueError("Could not determine attached-shock limit for Ramp 2.")

    dtheta2 = separation_margin * dtheta2_max
    theta2 = theta1 + dtheta2

    sh2 = oblique_shock(M1, dtheta2)
    if sh2 is None:
        raise ValueError("Ramp 2 has no attached weak shock solution.")

    beta2_rel, M2, p32, pt32 = sh2
    shock2_abs = theta1 + beta2_rel

    # -------------------------------------------------
    # Geometry: use a shock focus point downstream of the cowl lip
    # -------------------------------------------------
    P0 = np.array([0.0, 0.0], dtype=float)

    ramp1_dir = unit_from_angle_deg(theta1)
    ramp2_dir = unit_from_angle_deg(theta2)
    shock1_dir = unit_from_angle_deg(shock1_abs)
    shock2_dir = unit_from_angle_deg(shock2_abs)

    sep_angle = math.radians(shock2_abs - theta2)
    sep_sin = math.sin(sep_angle)
    if sep_sin <= 1e-10:
        raise ValueError("Shock 2 is nearly parallel to Ramp 2. Cannot place geometry.")

    # Distance from Ramp 2 start to cowl lip along shock2 based on required opening
    lam2_lip = h_req_normal / sep_sin

    # Focus point is farther downstream on shock2
    lam2_focus = shock_focus_factor * lam2_lip

    # Solve for P1 so focus point lies on shock1
    denom = cross2(ramp1_dir, shock1_dir)
    numer = cross2(shock2_dir, shock1_dir)
    if abs(denom) < 1e-10:
        raise ValueError("Ramp 1 is nearly collinear with its shock. Cannot solve geometry.")

    s1 = -lam2_focus * numer / denom
    if s1 <= 0.0:
        raise ValueError("Computed Ramp 2 start location is not downstream of Ramp 1 start.")

    P1 = s1 * ramp1_dir
    focus_point = P1 + lam2_focus * shock2_dir

    if abs(cross2(focus_point - P0, shock1_dir)) > 1e-6:
        raise ValueError("Internal geometry check failed: focus point is not on Ramp 1 shock.")

    # Cowl lip is upstream of the focus point on shock2
    C = P1 + lam2_lip * shock2_dir

    # Actual forebody point so ACTUAL forebody shock hits the cowl lip
    tan_bf = math.tan(math.radians(shock_fore_abs))
    if abs(tan_bf) < 1e-12:
        raise ValueError("Forebody shock angle is too small to solve forebody placement.")

    x_fore = C[0] - C[1] / tan_bf
    P_fore = np.array([x_fore, 0.0], dtype=float)

    # Foot of normal from cowl lip to Ramp 2
    t2 = ramp2_dir
    n2 = np.array([-t2[1], t2[0]], dtype=float)
    if np.dot(C - P1, n2) < 0.0:
        n2 = -n2
    F = C - h_req_normal * n2

    # -------------------------------------------------
    # Angled cowl and cowl shock
    # -------------------------------------------------
    theta_cowl = -leading_edge_angle_deg
    cowl_turn_mag = theta2 - theta_cowl
    if cowl_turn_mag <= 0.0:
        raise ValueError("Cowl turn magnitude must be positive.")

    shc = oblique_shock(M2, cowl_turn_mag)
    if shc is None:
        raise ValueError("Cowl shock has no attached weak shock solution.")

    beta_cowl_rel, M3, p43, pt43 = shc
    cowl_shock_abs = theta2 - beta_cowl_rel  # downward from lip

    # -------------------------------------------------
    # Immediate normal shock after cowl shock
    # -------------------------------------------------
    M4, p54, rho54, T54, pt54 = normal_shock(M3)
    T4 = T0 * (1.0 + 0.5 * (GAMMA - 1.0) * M0**2) / (1.0 + 0.5 * (GAMMA - 1.0) * M4**2)
    a4 = math.sqrt(GAMMA * R * T4)
    V4 = M4 * a4

    # Total-pressure fractions relative to freestream pt
    pt_frac_after_forebody = pt_fore_ratio
    pt_frac_after_shock1 = pt_fore_ratio * pt21
    pt_frac_after_shock2 = pt_fore_ratio * pt21 * pt32
    pt_frac_after_cowl = pt_fore_ratio * pt21 * pt32 * pt43
    pt_frac_after_immediate_normal = pt_frac_after_cowl * pt54

    # -------------------------------------------------
    # Throat sizing from post-cowl-shock Mach
    # -------------------------------------------------
    h_post_cowl = C[1] - F[1]
    A_post_cowl = width_m * h_post_cowl

    A_over_Astar_post_cowl = area_mach_ratio(M3)
    A_star_ideal = A_post_cowl / A_over_Astar_post_cowl
    A_throat = throat_area_factor * A_star_ideal
    h_throat = A_throat / width_m

    if h_throat <= 0.0:
        raise ValueError("Computed throat height is non-positive.")
    if h_throat >= C[1]:
        raise ValueError("Computed throat height is larger than available cowl height.")

    # -------------------------------------------------
    # Place throat lower point on cowl shock
    # -------------------------------------------------
    y_throat_lower = C[1] - h_throat

    tan_cs = math.tan(math.radians(cowl_shock_abs))
    if abs(tan_cs) < 1e-12:
        raise ValueError("Cowl shock is too shallow to place throat on it.")

    x_throat = C[0] + (y_throat_lower - C[1]) / tan_cs
    T_lower = np.array([x_throat, y_throat_lower], dtype=float)
    T_upper = np.array([x_throat, C[1]], dtype=float)

    if x_throat <= F[0]:
        raise ValueError("Computed throat is not downstream of Ramp 2 lower endpoint.")

    # -------------------------------------------------
    # Kantrowitz check only, no failure
    # -------------------------------------------------
    A_capture = width_m * C[1]
    CR_geom = A_capture / A_throat
    CR_k_raw = kantrowitz_contraction_ratio(M3)
    CR_k = CR_k_raw if CR_k_raw > 1.0 else 1.0 / CR_k_raw
    kantrowitz_pass = CR_geom <= CR_k

    return {
        "success": True,

        "rho0_kgm3": rho0,
        "V0_ms": V0,
        "A_capture_required_m2": A_capture_required,
        "opening_normal_to_ramp2_m": h_req_normal,

        "alpha_deg": alpha_deg,
        "leading_edge_angle_deg": leading_edge_angle_deg,

        "theta_fore_deg": theta_fore,
        "dtheta_fore_eff_deg": dtheta_fore_eff,
        "theta1_deg": theta1,
        "theta2_deg": theta2,
        "theta_cowl_deg": theta_cowl,
        "dtheta1_deg": dtheta1,
        "dtheta2_deg": dtheta2,
        "cowl_turn_mag_deg": cowl_turn_mag,

        "beta_fore_rel_deg": beta_fore_rel,
        "beta1_rel_deg": beta1_rel,
        "beta2_rel_deg": beta2_rel,
        "beta_cowl_rel_deg": beta_cowl_rel,

        "shock_fore_abs_deg": shock_fore_abs,
        "shock1_abs_deg": shock1_abs,
        "shock2_abs_deg": shock2_abs,
        "cowl_shock_abs_deg": cowl_shock_abs,

        "M_after_forebody_shock": M_fore,
        "M_after_shock1": M1,
        "M_after_shock2": M2,
        "M_after_cowl_shock": M3,
        "M_after_immediate_normal_shock": M4,

        "V_after_immediate_normal_shock_ms": V4,

        "pt_frac_after_forebody_shock": pt_frac_after_forebody,
        "pt_frac_after_shock1": pt_frac_after_shock1,
        "pt_frac_after_shock2": pt_frac_after_shock2,
        "pt_frac_after_cowl_shock": pt_frac_after_cowl,
        "pt_frac_after_immediate_normal_shock": pt_frac_after_immediate_normal,

        "forebody_xy": P_fore,
        "nose_xy": P0,
        "break2_xy": P1,
        "cowl_lip_xy": C,
        "ramp2_normal_foot_xy": F,
        "shock_focus_xy": focus_point,

        "post_cowl_area_m2": A_post_cowl,
        "post_cowl_height_m": h_post_cowl,
        "A_over_Astar_post_cowl": A_over_Astar_post_cowl,

        "throat_area_ideal_m2": A_star_ideal,
        "throat_area_actual_m2": A_throat,
        "throat_height_m": h_throat,
        "throat_upper_xy": T_upper,
        "throat_lower_xy": T_lower,

        "capture_area_m2": A_capture,
        "geometric_contraction_ratio": CR_geom,
        "kantrowitz_limit_CR": CR_k,
        "kantrowitz_pass": kantrowitz_pass,

        "forebody_length_m": float(np.linalg.norm(P0 - P_fore)),
        "ramp1_length_m": float(np.linalg.norm(P1 - P0)),
        "shock2_distance_from_break2_to_lip_m": lam2_lip,
        "shock_focus_factor": shock_focus_factor,
    }

def print_d2r_report(result):
    """
    Pretty-print report for design_2ramp_shock_matched_inlet().
    """
    if not result.get("success", False):
        print("FAILED")
        if "notes" in result:
            for line in result["notes"]:
                print(" -", line)
        return

    print("\n==============================")
    print("   2-RAMP SHOCK MATCHED INLET")
    print("==============================\n")

    print("FREESTREAM / CAPTURE")
    print("------------------------------")
    print(f"Density                         = {result['rho0_kgm3']:.4f} kg/m^3")
    print(f"Velocity                        = {result['V0_ms']:.2f} m/s")
    print(f"Capture area required           = {result['A_capture_required_m2']:.6f} m^2")
    print(f"Opening normal to Ramp 2        = {result['opening_normal_to_ramp2_m']:.6f} m")

    print("\nGEOMETRY / PANEL ANGLES")
    print("------------------------------")
    print(f"Alpha                           = {result['alpha_deg']:.3f} deg")
    print(f"Leading edge angle              = {result['leading_edge_angle_deg']:.3f} deg")
    print(f"Forebody angle                  = {result['theta_fore_deg']:.3f} deg")
    print(f"Forebody effective turn         = {result['dtheta_fore_eff_deg']:.3f} deg")
    print(f"Ramp 1 angle                    = {result['theta1_deg']:.3f} deg")
    print(f"Ramp 2 angle                    = {result['theta2_deg']:.3f} deg")
    print(f"Cowl angle                      = {result['theta_cowl_deg']:.3f} deg")
    print(f"Ramp 1 incremental turn         = {result['dtheta1_deg']:.3f} deg")
    print(f"Ramp 2 incremental turn         = {result['dtheta2_deg']:.3f} deg")
    print(f"Cowl turn magnitude             = {result['cowl_turn_mag_deg']:.3f} deg")

    print("\nSHOCK ANGLES")
    print("------------------------------")
    print(f"Forebody shock abs              = {result['shock_fore_abs_deg']:.3f} deg")
    print(f"Ramp 1 shock abs                = {result['shock1_abs_deg']:.3f} deg")
    print(f"Ramp 2 shock abs                = {result['shock2_abs_deg']:.3f} deg")
    print(f"Cowl shock abs                  = {result['cowl_shock_abs_deg']:.3f} deg")
    print(f"Forebody beta                   = {result['beta_fore_rel_deg']:.3f} deg")
    print(f"Ramp 1 beta                     = {result['beta1_rel_deg']:.3f} deg")
    print(f"Ramp 2 beta                     = {result['beta2_rel_deg']:.3f} deg")
    print(f"Cowl beta                       = {result['beta_cowl_rel_deg']:.3f} deg")

    print("\nFLOW STATE AFTER EACH SHOCK")
    print("------------------------------")
    print(f"Mach after forebody shock       = {result['M_after_forebody_shock']:.4f}")
    print(f"Mach after Ramp 1 shock         = {result['M_after_shock1']:.4f}")
    print(f"Mach after Ramp 2 shock         = {result['M_after_shock2']:.4f}")
    print(f"Mach after cowl shock           = {result['M_after_cowl_shock']:.4f}")
    print(f"Mach after immediate normal     = {result['M_after_immediate_normal_shock']:.4f}")
    print(f"Speed after immediate normal    = {result['V_after_immediate_normal_shock_ms']:.2f} m/s")

    print("\nTOTAL PRESSURE RECOVERY")
    print("------------------------------")
    print(f"pt/pt0 after forebody shock     = {result['pt_frac_after_forebody_shock']:.6f}")
    print(f"pt/pt0 after Ramp 1 shock       = {result['pt_frac_after_shock1']:.6f}")
    print(f"pt/pt0 after Ramp 2 shock       = {result['pt_frac_after_shock2']:.6f}")
    print(f"pt/pt0 after cowl shock         = {result['pt_frac_after_cowl_shock']:.6f}")
    print(f"pt/pt0 after immediate normal   = {result['pt_frac_after_immediate_normal_shock']:.6f}")

    print("\nKEY GEOMETRY POINTS")
    print("------------------------------")
    print(f"Forebody start                  = ({result['forebody_xy'][0]:.6f}, {result['forebody_xy'][1]:.6f}) m")
    print(f"Forebody/Ramp1 junction         = ({result['nose_xy'][0]:.6f}, {result['nose_xy'][1]:.6f}) m")
    print(f"Ramp1/Ramp2 junction            = ({result['break2_xy'][0]:.6f}, {result['break2_xy'][1]:.6f}) m")
    print(f"Cowl lip                        = ({result['cowl_lip_xy'][0]:.6f}, {result['cowl_lip_xy'][1]:.6f}) m")
    print(f"Ramp 2 normal foot              = ({result['ramp2_normal_foot_xy'][0]:.6f}, {result['ramp2_normal_foot_xy'][1]:.6f}) m")
    print(f"Shock focus point               = ({result['shock_focus_xy'][0]:.6f}, {result['shock_focus_xy'][1]:.6f}) m")

    print("\nPANEL LENGTHS")
    print("------------------------------")
    print(f"Forebody length                 = {result['forebody_length_m']:.6f} m")
    print(f"Ramp 1 length                   = {result['ramp1_length_m']:.6f} m")
    print(f"Shock2 to lip distance          = {result['shock2_distance_from_break2_to_lip_m']:.6f} m")
    print(f"Shock focus factor              = {result['shock_focus_factor']:.3f}")

    print("\nTHROAT")
    print("------------------------------")
    print(f"Post-cowl area                  = {result['post_cowl_area_m2']:.6f} m^2")
    print(f"Post-cowl height                = {result['post_cowl_height_m']:.6f} m")
    print(f"A/A* at post-cowl station       = {result['A_over_Astar_post_cowl']:.6f}")
    print(f"Ideal throat area               = {result['throat_area_ideal_m2']:.6f} m^2")
    print(f"Actual throat area              = {result['throat_area_actual_m2']:.6f} m^2")
    print(f"Throat height                   = {result['throat_height_m']:.6f} m")
    print(f"Throat lower point              = ({result['throat_lower_xy'][0]:.6f}, {result['throat_lower_xy'][1]:.6f}) m")
    print(f"Throat upper point              = ({result['throat_upper_xy'][0]:.6f}, {result['throat_upper_xy'][1]:.6f}) m")

    print("\nKANTROWITZ CHECK")
    print("------------------------------")
    print(f"Capture area                    = {result['capture_area_m2']:.6f} m^2")
    print(f"Geometric contraction ratio     = {result['geometric_contraction_ratio']:.6f}")
    print(f"Kantrowitz limit                = {result['kantrowitz_limit_CR']:.6f}")
    print(f"Kantrowitz pass                 = {result['kantrowitz_pass']}")

    print("\n==============================\n")

# =========================
# Plot
# =========================
def plot_2ramp_shock_matched_inlet(result, shock_extension_factor=1.20, cowl_extension_factor=1.35):
    if not result.get("success", False):
        raise ValueError("Result is not successful.")

    P_fore = result["forebody_xy"]
    P0 = result["nose_xy"]
    P1 = result["break2_xy"]
    C = result["cowl_lip_xy"]
    F = result["ramp2_normal_foot_xy"]
    T_lower = result["throat_lower_xy"]
    T_upper = result["throat_upper_xy"]
    focus = result["shock_focus_xy"]

    shock_fore_abs = result["shock_fore_abs_deg"]
    shock1_abs = result["shock1_abs_deg"]
    shock2_abs = result["shock2_abs_deg"]
    cowl_shock_abs = result["cowl_shock_abs_deg"]
    theta_cowl = result["theta_cowl_deg"]

    x_end = shock_extension_factor * max(T_upper[0], focus[0], C[0])
    cowl_dir = unit_from_angle_deg(theta_cowl)

    fore_shock_dir = unit_from_angle_deg(shock_fore_abs)
    shock1_dir = unit_from_angle_deg(shock1_abs)
    shock2_dir = unit_from_angle_deg(shock2_abs)
    cowl_shock_dir = unit_from_angle_deg(cowl_shock_abs)

    lam_fore_end = (x_end - P_fore[0]) / fore_shock_dir[0]
    y_fore_end = P_fore[1] + lam_fore_end * fore_shock_dir[1]

    lam1_end = (x_end - P0[0]) / shock1_dir[0]
    y_s1_end = P0[1] + lam1_end * shock1_dir[1]

    lam2_end = (x_end - P1[0]) / shock2_dir[0]
    y_s2_end = P1[1] + lam2_end * shock2_dir[1]

    lamc_end = (x_end - C[0]) / cowl_shock_dir[0]
    y_sc_end = C[1] + lamc_end * cowl_shock_dir[1]

    # cowl line
    cowl_len = cowl_extension_factor * max(T_upper[0] - C[0], 0.2)
    cowl_end = C + cowl_len * cowl_dir

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Geometry
    ax.plot([P_fore[0], P0[0]], [P_fore[1], P0[1]], linewidth=2, label="Forebody")
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], linewidth=2, label="Ramp 1")
    ax.plot([P1[0], F[0]], [P1[1], F[1]], linewidth=2, label="Ramp 2")
    ax.plot([F[0], T_lower[0]], [F[1], T_lower[1]], linewidth=2, label="Internal sharp edge")
    ax.plot([F[0], C[0]], [F[1], C[1]], linewidth=2, label="Opening (normal to Ramp 2)")
    ax.plot([C[0], cowl_end[0]], [C[1], cowl_end[1]], linewidth=2, label="Angled cowl")
    ax.plot([T_lower[0], T_upper[0]], [T_lower[1], T_upper[1]], linewidth=2, label="Throat")

    # Shocks
    ax.plot([P_fore[0], x_end], [P_fore[1], y_fore_end], linestyle="--", linewidth=1.5, label="Forebody shock")
    ax.plot([P0[0], x_end], [P0[1], y_s1_end], linestyle="--", linewidth=1.5, label="Ramp 1 shock")
    ax.plot([P1[0], x_end], [P1[1], y_s2_end], linestyle="--", linewidth=1.5, label="Ramp 2 shock")
    ax.plot([C[0], x_end], [C[1], y_sc_end], linestyle="--", linewidth=1.5, label="Cowl shock")

    # Focus point
    ax.plot(focus[0], focus[1], marker="x", markersize=8)
    ax.text(focus[0], focus[1], "  Shock focus", va="bottom")

    # Key points
    for P in [P_fore, P0, P1, F, C, T_lower, T_upper]:
        ax.plot(P[0], P[1], marker="o")

    ax.text(P_fore[0], P_fore[1], "  Forebody start", va="bottom")
    ax.text(P0[0], P0[1], "  Forebody/Ramp 1", va="bottom")
    ax.text(P1[0], P1[1], "  Ramp 1/Ramp 2", va="bottom")
    ax.text(F[0], F[1], "  Ramp 2 end", va="top")
    ax.text(C[0], C[1], "  Cowl lip", va="bottom")
    ax.text(T_lower[0], T_lower[1], "  Throat lower", va="top")
    ax.text(T_upper[0], T_upper[1], "  Throat upper", va="bottom")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("2-Ramp Shock-Matched Inlet with Cowl Angle and Throat")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    y_max = max(C[1], F[1], y_fore_end, y_s1_end, y_s2_end, y_sc_end, T_upper[1], focus[1], cowl_end[1])
    y_min = min(P_fore[1], P0[1], P1[1], F[1], C[1], y_sc_end, T_lower[1], focus[1], cowl_end[1], 0.0)

    ax.set_xlim(P_fore[0] - 0.02 * (x_end - P_fore[0]), 1.05 * max(x_end, cowl_end[0], T_upper[0]))
    ax.set_ylim(y_min - 0.05 * max(y_max - y_min, 1e-6), y_max + 0.15 * max(y_max - y_min, 1e-6))

    plt.show()


# =========================
# Example
# =========================
if __name__ == "__main__":
    result = design_2ramp_shock_matched_inlet(
        M0=5.0,
        altitude_m=12000.0,
        alpha_deg=4.0,
        leading_edge_angle_deg=4.0,
        mdot_required=6.0,
        width_m=0.25,
        separation_margin=0.3,
        throat_area_factor=0.95,
    )

    print_d2r_report(result)

    plot_2ramp_shock_matched_inlet(result)