import math
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.4
R = 287.05

# Subsonic-diffuser (throat -> combustor face) geometry defaults.
DIFFUSER_AREA_RATIO = 2.0
DIFFUSER_HALF_ANGLE_DEG = 3.0

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
    #print(f"Temp: {T}, Pressure: {p}, density: {rho}, speed of sound: {a}")
    return T, p, rho, a

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

def unit_from_angle_deg(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=float)

def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def total_temperature(T, M, gamma=GAMMA):
    return T * (1.0 + 0.5 * (gamma - 1.0) * M**2)

def total_pressure(p, M, gamma=GAMMA):
    return p * (1.0 + 0.5 * (gamma - 1.0) * M**2) ** (gamma / (gamma - 1.0))

def validate_d2r_inputs(M0,mdot_required,width_m,forebody_separation_margin,
    ramp_separation_margin,throat_area_factor,shock_focus_factor,):
    if M0 <= 1.0:
        raise ValueError("M0 must be supersonic.")
    if mdot_required <= 0.0:
        raise ValueError("mdot_required must be positive.")
    if width_m <= 0.0:
        raise ValueError("width_m must be positive.")
    if not (0.0 < forebody_separation_margin < 1.0):
        raise ValueError("forebody_separation_margin must be between 0 and 1.")
    if not (0.0 < ramp_separation_margin < 1.0):
        raise ValueError("ramp_separation_margin must be between 0 and 1.")
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

def freestream_state(M0, altitude_m):
    T0, p0, rho0, a0 = std_atmosphere_1976(altitude_m)
    V0 = M0 * a0
    pt0 = total_pressure(p0, M0)
    Tt0 = total_temperature(T0, M0)
    return {
        "T0": T0,
        "p0": p0,
        "rho0": rho0,
        "a0": a0,
        "V0": V0,
        "pt0": pt0,
        "Tt0": Tt0,
    }

def required_opening_from_mdot(mdot_required, rho0, V0, width_m):
    A_capture_required = mdot_required / (rho0 * V0)
    h_req_normal = A_capture_required / width_m
    return A_capture_required, h_req_normal

def solve_forebody_stage(
    M0,alpha_deg,leading_edge_angle_deg,forebody_separation_margin,):
    theta_fore = leading_edge_angle_deg

    # Actual operating forebody turn
    dtheta_fore_eff = theta_fore + alpha_deg

    # Margin/reference turn
    dtheta_fore_margin = theta_fore + max(alpha_deg - 4.0, 0.0)

    if dtheta_fore_eff <= 0.0:
        raise ValueError("Forebody effective turn must be positive.")

    theta_fore_eff_max = theta_max_attached(M0)
    if theta_fore_eff_max is None:
        raise ValueError("Could not determine attached-shock limit for forebody.")

    if dtheta_fore_eff >= forebody_separation_margin * theta_fore_eff_max:
        raise ValueError(
            f"Forebody effective turn ({dtheta_fore_eff:.3f} deg) exceeds "
            f"{forebody_separation_margin:.2f} of attached-shock limit ({theta_fore_eff_max:.3f} deg)."
        )

    # ACTUAL SHOCK
    shf = oblique_shock(M0, dtheta_fore_eff)
    if shf is None:
        raise ValueError("Forebody has no attached weak shock solution.")

    beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio = shf

    # MARGIN SHOCK
    shf_margin = oblique_shock(M0, dtheta_fore_margin)
    if shf_margin is None:
        raise ValueError("Margin forebody shock invalid.")

    beta_fore_margin_rel = shf_margin[0]

    return {
        "theta_fore": theta_fore,

        "dtheta_fore_eff": dtheta_fore_eff,
        "dtheta_fore_margin": dtheta_fore_margin,

        "beta_fore_rel": beta_fore_rel,
        "beta_fore_margin_rel": beta_fore_margin_rel,

        "M_fore": M_fore,
        "p_fore_ratio": p_fore_ratio,
        "pt_fore_ratio": pt_fore_ratio,

        "shock_fore_abs": beta_fore_rel,
        "shock_fore_margin_abs": beta_fore_margin_rel,
    }

def solve_ramp_stage(M_up, theta_base, ramp_separation_margin, stage_name):
    dtheta_max = theta_max_attached(M_up)
    if dtheta_max is None:
        raise ValueError(f"Could not determine attached-shock limit for {stage_name}.")

    dtheta = ramp_separation_margin * dtheta_max
    theta_abs = theta_base + dtheta

    sh = oblique_shock(M_up, dtheta)
    if sh is None:
        raise ValueError(f"{stage_name} has no attached weak shock solution.")

    beta_rel, M_down, p_ratio, pt_ratio = sh
    shock_abs = theta_base + beta_rel

    return {
        "dtheta": dtheta,
        "theta_abs": theta_abs,
        "beta_rel": beta_rel,
        "M_down": M_down,
        "p_ratio": p_ratio,
        "pt_ratio": pt_ratio,
        "shock_abs": shock_abs,
    }

def solve_external_geometry(theta1,theta2,shock1_abs,shock2_abs,h_req_normal,
    shock_focus_factor,):

    P0 = np.array([0.0, 0.0], dtype=float)

    ramp1_dir = unit_from_angle_deg(theta1)
    ramp2_dir = unit_from_angle_deg(theta2)
    shock1_dir = unit_from_angle_deg(shock1_abs)
    shock2_dir = unit_from_angle_deg(shock2_abs)

    # Normal pointing from Ramp 2 toward the cowl side
    n2 = np.array([-ramp2_dir[1], ramp2_dir[0]], dtype=float)

    sep_angle = math.radians(shock2_abs - theta2)
    sep_sin = math.sin(sep_angle)
    if sep_sin <= 1e-10:
        raise ValueError("Shock 2 is nearly parallel to Ramp 2. Cannot place geometry.")

    # Focus lies on Ramp 2 shock
    lam2_focus = shock_focus_factor * h_req_normal / sep_sin

    # Focus also lies on Ramp 1 shock
    denom = cross2(ramp1_dir,shock1_dir)
    numer = cross2(lam2_focus * shock2_dir,shock1_dir)

    if abs(denom) < 1e-10:
        raise ValueError("Ramp 1 is nearly collinear with its shock. Cannot solve geometry.")

    s1 = -numer / denom
    if s1 <= 0.0:
        raise ValueError("Computed Ramp 2 start location is not downstream of Ramp 1 start.")

    P1 = s1 * ramp1_dir
    focus_point = P1 + lam2_focus * shock2_dir

    if abs(cross2(focus_point - P0,shock1_dir)) > 1e-6:
        raise ValueError("Internal geometry check failed: focus point is not on Ramp 1 shock.")

    # Put cowl lip below focus along the opening normal
    focus_to_lip_normal = (shock_focus_factor - 1.0) * h_req_normal
    C = focus_point - focus_to_lip_normal * n2

    # Foot of opening on Ramp 2
    F = C - h_req_normal * n2

    # Diagnostic distance along Ramp 2 shock to the lip projection
    lam2_lip = np.dot(C - P1,shock2_dir)

    return {
        "P0": P0,
        "P1": P1,
        "C": C,
        "F": F,
        "n2": n2,
        "ramp1_dir": ramp1_dir,
        "ramp2_dir": ramp2_dir,
        "shock1_dir": shock1_dir,
        "shock2_dir": shock2_dir,
        "lam2_lip": lam2_lip,
        "lam2_focus": lam2_focus,
        "focus_point": focus_point,
    }

def solve_forebody_start_from_focus(focus_point,shock_fore_abs,):
    tan_bf = math.tan(math.radians(shock_fore_abs))
    if abs(tan_bf) < 1e-12:
        raise ValueError("Forebody shock angle is too small to solve forebody placement.")

    x_fore = focus_point[0] - focus_point[1] / tan_bf
    return np.array([x_fore, 0.0], dtype=float)

def solve_cowl_stage(M2, theta2, leading_edge_angle_deg):
    theta_cowl = -leading_edge_angle_deg
    cowl_turn_mag = theta2 - theta_cowl
    if cowl_turn_mag <= 0.0:
        raise ValueError("Cowl turn magnitude must be positive.")

    shc = oblique_shock(M2, cowl_turn_mag)
    if shc is None:
        raise ValueError("Cowl shock has no attached weak shock solution.")

    beta_cowl_rel, M3, p43, pt43 = shc
    cowl_shock_abs = theta2 - beta_cowl_rel

    return {
        "theta_cowl": theta_cowl,
        "cowl_turn_mag": cowl_turn_mag,
        "beta_cowl_rel": beta_cowl_rel,
        "M3": M3,
        "p43": p43,
        "pt43": pt43,
        "cowl_shock_abs": cowl_shock_abs,
    }

def solve_immediate_normal_after_cowl(M3, pt_frac_after_cowl, Tt0):
    M4, p54, rho54, T54, pt54 = normal_shock(M3)
    pt_frac_after_immediate_normal = pt_frac_after_cowl * pt54

    T4 = Tt0 / (1.0 + 0.5 * (GAMMA - 1.0) * M4**2)
    a4 = math.sqrt(GAMMA * R * T4)
    V4 = M4 * a4

    return {
        "M4": M4,
        "p54": p54,
        "rho54": rho54,
        "T54": T54,
        "pt54": pt54,
        "pt_frac_after_immediate_normal": pt_frac_after_immediate_normal,
        "T4": T4,
        "a4": a4,
        "V4": V4,
    }

def size_throat_from_post_cowl(M3, width_m, h_req_normal, throat_area_factor):
    A_post_cowl = width_m * h_req_normal
    A_over_Astar_post_cowl = area_mach_ratio(M3)
    A_star_ideal = A_post_cowl / A_over_Astar_post_cowl
    A_throat = throat_area_factor * A_star_ideal
    h_throat = A_throat / width_m

    if h_throat <= 0.0:
        raise ValueError("Computed throat height is non-positive.")

    return {
        "A_post_cowl": A_post_cowl,
        "A_over_Astar_post_cowl": A_over_Astar_post_cowl,
        "A_star_ideal": A_star_ideal,
        "A_throat": A_throat,
        "h_throat": h_throat,
    }

def place_throat_on_cowl_and_cowl_shock(C, theta_cowl, cowl_shock_abs, h_throat, ramp2_end_point,leading_edge_angle_deg):
    m_cowl = math.tan(math.radians(theta_cowl))
    m_cs = math.tan(math.radians(cowl_shock_abs))
    denom_gap = m_cowl - m_cs
    if abs(denom_gap) < 1e-12:
        raise ValueError("Cowl and cowl shock are nearly parallel. Cannot place throat.")

    x_throat = C[0] + h_throat / denom_gap
    y_throat_upper = C[1] + m_cowl * (x_throat - C[0])
    y_throat_lower = C[1] + m_cs * (x_throat - C[0])

    T_upper = np.array([x_throat, y_throat_upper], dtype=float)
    T_lower = np.array([x_throat, y_throat_lower], dtype=float)
    T_upper_body = point_in_body_frame(T_upper, leading_edge_angle_deg)
    if x_throat <= ramp2_end_point[0]:
        raise ValueError("Computed throat is not downstream of Ramp 2 lower endpoint.")

    return {
        "T_upper": T_upper,
        "T_lower": T_lower,
        "x_throat": x_throat,
        "y_throat_upper": y_throat_upper,
        "y_throat_lower": y_throat_lower,
        "throat_upper_body_x_m": T_upper_body[0],
        "throat_upper_body_y_m": T_upper_body[1],
    }

def evaluate_kantrowitz(M3, width_m, h_req_normal, A_throat):
    A_capture_k = width_m * h_req_normal
    CR_geom = A_capture_k / A_throat
    CR_k_raw = kantrowitz_contraction_ratio(M3)
    CR_k = CR_k_raw if CR_k_raw > 1.0 else 1.0 / CR_k_raw
    kantrowitz_pass = CR_geom <= CR_k

    return {
        "A_capture_k": A_capture_k,
        "CR_geom": CR_geom,
        "CR_k": CR_k,
        "kantrowitz_pass": kantrowitz_pass,
    }

def point_in_body_frame(point_xy,leading_edge_angle_deg,):
    theta = math.radians(leading_edge_angle_deg)
    x = point_xy[0]
    y = point_xy[1]

    x_body = x * math.cos(theta) + y * math.sin(theta)
    y_body = -x * math.sin(theta) + y * math.cos(theta)

    return np.array([x_body, y_body], dtype=float)

def invert_area_mach_ratio_supersonic(target_A_over_Astar,tol=1e-8,max_iter=200,):
    if target_A_over_Astar < 1.0:
        raise ValueError("Supersonic A/A* target must be >= 1.")

    lo = 1.0 + 1e-8
    hi = 50.0

    f_lo = area_mach_ratio(lo) - target_A_over_Astar
    f_hi = area_mach_ratio(hi) - target_A_over_Astar

    if f_lo > 0.0:
        return lo
    if f_hi < 0.0:
        raise ValueError("Could not bracket supersonic Mach from A/A* target.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = area_mach_ratio(mid) - target_A_over_Astar
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)

def size_throat_from_kantrowitz(M2,width_m,h_req_normal,kantrowitz_margin=0.95,):
    A_capture_k = width_m * h_req_normal

    CR_k_raw = kantrowitz_contraction_ratio(M2)
    CR_k = CR_k_raw if CR_k_raw > 1.0 else 1.0 / CR_k_raw
    CR_target = kantrowitz_margin * CR_k

    A_throat = A_capture_k / CR_target
    h_throat = A_throat / width_m

    return {
        "A_capture_k": A_capture_k,
        "CR_k": CR_k,
        "CR_target": CR_target,
        "A_throat": A_throat,
        "h_throat": h_throat,
    }

def target_post_cowl_mach_from_throat(A_post_cowl,A_throat,):
    A_over_Astar_target = A_post_cowl / A_throat
    M3_target = invert_area_mach_ratio_supersonic(A_over_Astar_target)
    return {
        "A_over_Astar_target": A_over_Astar_target,
        "M3_target": M3_target,
    }

def solve_cowl_stage_for_target_postshock(M2,theta2,M3_target,):
    theta_max = theta_max_attached(M2)
    if theta_max is None:
        raise ValueError("Could not determine attached-shock limit for cowl stage.")

    def eval_candidate(turn_deg):
        sh = oblique_shock(M2, turn_deg)
        if sh is None:
            return None

        beta_rel, M3, p43, pt43 = sh
        err = abs(M3 - M3_target)

        return {
            "theta_cowl": theta2 - turn_deg,
            "cowl_turn_mag": turn_deg,
            "beta_cowl_rel": beta_rel,
            "M3": M3,
            "p43": p43,
            "pt43": pt43,
            "cowl_shock_abs": theta2 - beta_rel,
            "M3_target": M3_target,
            "M3_error": err,
        }

    # -------------------------
    # Stage 1: coarse search
    # -------------------------
    coarse_candidates = np.linspace(1e-3, theta_max * 0.999, 20)

    best = None
    best_err = float("inf")

    for turn_deg in coarse_candidates:
        cand = eval_candidate(turn_deg)
        if cand is None:
            continue
        if cand["M3_error"] < best_err:
            best_err = cand["M3_error"]
            best = cand

    if best is None:
        raise ValueError("Could not solve cowl stage for target post-shock Mach.")

    # -------------------------
    # Stage 2: local refinement
    # -------------------------
    coarse_step = coarse_candidates[1] - coarse_candidates[0]
    refine_lo = max(1e-3, best["cowl_turn_mag"] - 2.0 * coarse_step)
    refine_hi = min(theta_max * 0.999, best["cowl_turn_mag"] + 2.0 * coarse_step)

    refine_candidates = np.linspace(refine_lo, refine_hi, 40)

    for turn_deg in refine_candidates:
        cand = eval_candidate(turn_deg)
        if cand is None:
            continue
        if cand["M3_error"] < best_err:
            best_err = cand["M3_error"]
            best = cand

    return best

def build_d2r_result(fs,A_capture_required,h_req_normal,
    alpha_deg,leading_edge_angle_deg,forebody_separation_margin,
    ramp_separation_margin,forebody,ramp1,ramp2,geom,P_fore,
    cowl,normal_after_cowl,throat,throat_geom,kant,shock_focus_factor,):
    pt_frac_after_forebody = forebody["pt_fore_ratio"]
    pt_frac_after_shock1 = forebody["pt_fore_ratio"] * ramp1["pt_ratio"]
    pt_frac_after_shock2 = forebody["pt_fore_ratio"] * ramp1["pt_ratio"] * ramp2["pt_ratio"]
    pt_frac_after_cowl = pt_frac_after_shock2 * cowl["pt43"]
    return {
        "success": True,

        "rho0_kgm3": fs["rho0"],
        "V0_ms": fs["V0"],
        "A_capture_required_m2": A_capture_required,
        "opening_normal_to_ramp2_m": h_req_normal,

        "alpha_deg": alpha_deg,
        "leading_edge_angle_deg": leading_edge_angle_deg,
        "forebody_separation_margin": forebody_separation_margin,
        "ramp_separation_margin": ramp_separation_margin,

        "theta_fore_deg": forebody["theta_fore"],
        "dtheta_fore_eff_deg": forebody["dtheta_fore_eff"],
        "theta1_deg": ramp1["theta_abs"],
        "theta2_deg": ramp2["theta_abs"],
        "theta_cowl_deg": cowl["theta_cowl"],
        "dtheta1_deg": ramp1["dtheta"],
        "dtheta2_deg": ramp2["dtheta"],
        "cowl_turn_mag_deg": cowl["cowl_turn_mag"],

        "beta_fore_rel_deg": forebody["beta_fore_rel"],
        "beta1_rel_deg": ramp1["beta_rel"],
        "beta2_rel_deg": ramp2["beta_rel"],
        "beta_cowl_rel_deg": cowl["beta_cowl_rel"],

        "shock_fore_margin_abs_deg": forebody["shock_fore_margin_abs"],
        "dtheta_fore_margin_deg": forebody["dtheta_fore_margin"],
        "shock_fore_abs_deg": forebody["shock_fore_abs"],
        "shock1_abs_deg": ramp1["shock_abs"],
        "shock2_abs_deg": ramp2["shock_abs"],
        "cowl_shock_abs_deg": cowl["cowl_shock_abs"],

        "M_after_forebody_shock": forebody["M_fore"],
        "M_after_shock1": ramp1["M_down"],
        "M_after_shock2": ramp2["M_down"],
        "M_after_cowl_shock": cowl["M3"],
        "M_after_immediate_normal_shock": normal_after_cowl["M4"],

        "V_after_immediate_normal_shock_ms": normal_after_cowl["V4"],

        "pt_frac_after_forebody_shock": pt_frac_after_forebody,
        "pt_frac_after_shock1": pt_frac_after_shock1,
        "pt_frac_after_shock2": pt_frac_after_shock2,
        "pt_frac_after_cowl_shock": pt_frac_after_cowl,
        "pt_frac_after_immediate_normal_shock": normal_after_cowl["pt_frac_after_immediate_normal"],

        "forebody_xy": P_fore,
        "nose_xy": geom["P0"],
        "break2_xy": geom["P1"],
        "cowl_lip_xy": geom["C"],
        "ramp2_normal_foot_xy": geom["F"],
        "shock_focus_xy": geom["focus_point"],

        "post_cowl_area_m2": throat["A_post_cowl"],
        "post_cowl_height_m": h_req_normal,
        "A_over_Astar_post_cowl": throat["A_over_Astar_post_cowl"],

        "throat_area_ideal_m2": throat["A_star_ideal"],
        "throat_area_actual_m2": throat["A_throat"],
        "throat_height_m": throat["h_throat"],
        "throat_upper_xy": throat_geom["T_upper"],
        "throat_lower_xy": throat_geom["T_lower"],

        "capture_area_kantrowitz_m2": kant["A_capture_k"],
        "geometric_contraction_ratio": kant["CR_geom"],
        "kantrowitz_limit_CR": kant["CR_k"],
        "kantrowitz_pass": kant["kantrowitz_pass"],

        "forebody_length_m": float(np.linalg.norm(geom["P0"] - P_fore)),
        "ramp1_length_m": float(np.linalg.norm(geom["P1"] - geom["P0"])),
        "shock2_distance_from_break2_to_lip_m": geom["lam2_lip"],
        "shock_focus_factor": shock_focus_factor,
    }

# ----------------------------------------------------------------------------
# Point 1: Subsonic diffuser geometry (station 4 -> station 4.5)
# ----------------------------------------------------------------------------
def build_subsonic_diffuser(T_upper, T_lower, h_throat, width_m,
                            area_ratio_exit_to_throat,
                            half_angle_deg=None, length_m=None,
                            n_stations=51):
    """
    Build a straight-walled symmetric subsonic diffuser downstream of the
    throat. Exactly one of (half_angle_deg, length_m) must be provided.
    """
    import numpy as np, math

    if area_ratio_exit_to_throat <= 1.0:
        raise ValueError("Diffuser exit/throat area ratio must exceed 1.")
    if (half_angle_deg is None) == (length_m is None):
        raise ValueError("Specify exactly one of half_angle_deg or length_m.")

    A_throat = h_throat * width_m
    A_exit   = area_ratio_exit_to_throat * A_throat
    h_exit   = A_exit / width_m

    if length_m is None:
        dh_per_side = 0.5 * (h_exit - h_throat)
        length_m = dh_per_side / math.tan(math.radians(half_angle_deg))

    x0 = 0.5 * (T_upper[0] + T_lower[0])
    y_mid = 0.5 * (T_upper[1] + T_lower[1])
    x_exit = x0 + length_m

    xs = np.linspace(x0, x_exit, n_stations)
    s  = (xs - x0) / length_m
    A_of_x = A_throat + s * (A_exit - A_throat)
    h_of_x = A_of_x / width_m

    upper_wall = np.column_stack([xs, y_mid + 0.5 * h_of_x])
    lower_wall = np.column_stack([xs, y_mid - 0.5 * h_of_x])

    exit_upper = upper_wall[-1]
    exit_lower = lower_wall[-1]

    return {
        "x_throat":      x0,
        "x_exit":        x_exit,
        "length_m":      float(length_m),
        "A_throat":      float(A_throat),
        "A_exit":        float(A_exit),
        "h_throat":      float(h_throat),
        "h_exit":        float(h_exit),
        "area_ratio":    float(area_ratio_exit_to_throat),
        "width_m":       float(width_m),
        "y_centerline":  float(y_mid),
        "x_stations":    xs,
        "A_stations":    A_of_x,
        "h_stations":    h_of_x,
        "upper_wall_xy": upper_wall,
        "lower_wall_xy": lower_wall,
        "exit_upper_xy": exit_upper,
        "exit_lower_xy": exit_lower,
    }


# ----------------------------------------------------------------------------
# Point 2: Terminal normal-shock position driven by back pressure
# ----------------------------------------------------------------------------
def _static_over_total(M, gamma=GAMMA):
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-gamma / (gamma - 1.0))


def _subsonic_mach_from_area_ratio(A_over_Astar, tol=1e-10, max_iter=200):
    """Subsonic branch of the isentropic area-Mach relation."""
    if A_over_Astar < 1.0:
        raise ValueError("A/A* must be >= 1 for a real subsonic solution.")
    lo, hi = 1e-6, 1.0 - 1e-8
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = area_mach_ratio(mid) - A_over_Astar
        if abs(f) < tol:
            return mid
        if f > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _exit_static_pressure_for_shock_at(x_s, diffuser, Pt_after_cowl,
                                       M_throat=1.0):
    """
    Given a trial shock axial station x_s in the diverging (supersonic)
    region, propagate to the diffuser exit and return predicted exit static
    pressure plus intermediate state.
    """
    import numpy as np

    A_throat = diffuser["A_throat"]
    A_exit   = diffuser["A_exit"]
    xs       = diffuser["x_stations"]
    A_stats  = diffuser["A_stations"]

    if x_s <= xs[0] or x_s >= xs[-1]:
        raise ValueError("Trial shock station outside diffuser.")

    A_s = float(np.interp(x_s, xs, A_stats))

    M_sup = invert_area_mach_ratio_supersonic(A_s / A_throat)

    M_sub, p2_p1, _, _, pt2_pt1 = normal_shock(M_sup)
    Pt_after_shock = Pt_after_cowl * pt2_pt1

    Astar_sub = A_s / area_mach_ratio(M_sub)
    M_exit    = _subsonic_mach_from_area_ratio(A_exit / Astar_sub)

    Ps_exit = Pt_after_shock * _static_over_total(M_exit)
    return {
        "x_s":            x_s,
        "A_s":            A_s,
        "M_sup":          M_sup,
        "M_sub":          M_sub,
        "pt_ratio_shock": pt2_pt1,
        "Pt_after_shock": Pt_after_shock,
        "M_exit":         M_exit,
        "Ps_exit":        Ps_exit,
    }


def solve_terminal_shock_position(result, p_back, Pt_after_cowl, Tt0,
                                  tol=1e-4, max_iter=80):
    """
    Find axial shock station x_s in the subsonic diffuser such that the
    predicted exit static pressure equals p_back.
    """
    diffuser = result["diffuser"]
    xs = diffuser["x_stations"]

    eps = 1e-4 * (xs[-1] - xs[0])
    x_lo = xs[0] + eps
    x_hi = xs[-1] - eps

    hi_state = _exit_static_pressure_for_shock_at(x_hi, diffuser, Pt_after_cowl)
    lo_state = _exit_static_pressure_for_shock_at(x_lo, diffuser, Pt_after_cowl)

    Ps_max = lo_state["Ps_exit"]
    Ps_min = hi_state["Ps_exit"]

    if p_back > Ps_max:
        return {"status": "expelled", "p_back": p_back,
                "Ps_max": Ps_max, "Ps_min": Ps_min,
                "pt_frac_after_terminal_shock":
                    Pt_after_cowl / Pt_after_cowl,
                **lo_state}
    if p_back < Ps_min:
        return {"status": "swallowed", "p_back": p_back,
                "Ps_max": Ps_max, "Ps_min": Ps_min,
                "pt_frac_after_terminal_shock":
                    hi_state["Pt_after_shock"] / Pt_after_cowl,
                **hi_state}

    lo, hi = x_lo, x_hi
    state = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        state = _exit_static_pressure_for_shock_at(mid, diffuser, Pt_after_cowl)
        err = state["Ps_exit"] - p_back
        if abs(err) < tol * max(p_back, 1.0):
            break
        if err > 0.0:
            lo = mid
        else:
            hi = mid

    pt_frac_terminal = state["Pt_after_shock"] / Pt_after_cowl

    T_exit = Tt0 / (1.0 + 0.5 * (GAMMA - 1.0) * state["M_exit"] ** 2)
    a_exit = math.sqrt(GAMMA * R * T_exit)
    V_exit = state["M_exit"] * a_exit

    return {
        "status":                        "normal",
        "p_back":                        p_back,
        "Ps_max":                        Ps_max,
        "Ps_min":                        Ps_min,
        "x_s":                           state["x_s"],
        "A_s":                           state["A_s"],
        "M_sup":                         state["M_sup"],
        "M_sub":                         state["M_sub"],
        "M_exit":                        state["M_exit"],
        "Ps_exit":                       state["Ps_exit"],
        "Pt_after_terminal_shock":       state["Pt_after_shock"],
        "pt_frac_after_terminal_shock":  pt_frac_terminal,
        "T_exit":                        T_exit,
        "a_exit":                        a_exit,
        "V_exit":                        V_exit,
    }


def design_2ramp_shock_matched_inlet(M0,altitude_m,alpha_deg,leading_edge_angle_deg,
    mdot_required,width_m,forebody_separation_margin=0.95,
    ramp_separation_margin=0.95,kantrowitz_margin=0.95,
    shock_focus_factor=1.25,):

    validate_d2r_inputs(
        M0=M0,
        mdot_required=mdot_required,
        width_m=width_m,
        forebody_separation_margin=forebody_separation_margin,
        ramp_separation_margin=ramp_separation_margin,
        throat_area_factor=kantrowitz_margin,
        shock_focus_factor=shock_focus_factor,
    )

    fs = freestream_state(M0,altitude_m)

    A_capture_required, h_req_normal = required_opening_from_mdot(
        mdot_required=mdot_required,
        rho0=fs["rho0"],
        V0=fs["V0"],
        width_m=width_m,
    )

    forebody = solve_forebody_stage(
        M0=M0,
        alpha_deg=alpha_deg,
        leading_edge_angle_deg=leading_edge_angle_deg,
        forebody_separation_margin=forebody_separation_margin,
    )

    ramp1 = solve_ramp_stage(
        M_up=forebody["M_fore"],
        theta_base=forebody["theta_fore"],
        ramp_separation_margin=ramp_separation_margin,
        stage_name="Ramp 1",
    )

    ramp2 = solve_ramp_stage(
        M_up=ramp1["M_down"],
        theta_base=ramp1["theta_abs"],
        ramp_separation_margin=ramp_separation_margin,
        stage_name="Ramp 2",
    )

    geom = solve_external_geometry(
        theta1=ramp1["theta_abs"],
        theta2=ramp2["theta_abs"],
        shock1_abs=ramp1["shock_abs"],
        shock2_abs=ramp2["shock_abs"],
        h_req_normal=h_req_normal,
        shock_focus_factor=shock_focus_factor,
    )

    P_fore = solve_forebody_start_from_focus(
        focus_point=geom["focus_point"],
        shock_fore_abs=forebody["shock_fore_abs"],
    )

    # 1) Size throat from Kantrowitz using pre-cowl-shock Mach M2
    throat_k = size_throat_from_kantrowitz(
        M2=ramp2["M_down"],
        width_m=width_m,
        h_req_normal=h_req_normal,
        kantrowitz_margin=kantrowitz_margin,
    )

    # 2) Reverse out target post-cowl-shock Mach from throat area
    post_cowl_target = target_post_cowl_mach_from_throat(
        A_post_cowl=throat_k["A_capture_k"],
        A_throat=throat_k["A_throat"],
    )

    # 3) Solve cowl angle to produce that post-shock Mach
    cowl = solve_cowl_stage_for_target_postshock(
        M2=ramp2["M_down"],
        theta2=ramp2["theta_abs"],
        M3_target=post_cowl_target["M3_target"],
    )

    pt_frac_after_cowl = (
        forebody["pt_fore_ratio"]
        * ramp1["pt_ratio"]
        * ramp2["pt_ratio"]
        * cowl["pt43"]
    )

    normal_after_cowl = solve_immediate_normal_after_cowl(
        M3=cowl["M3"],
        pt_frac_after_cowl=pt_frac_after_cowl,
        Tt0=fs["Tt0"],
    )

    throat_geom = place_throat_on_cowl_and_cowl_shock(
        C=geom["C"],
        theta_cowl=cowl["theta_cowl"],
        cowl_shock_abs=cowl["cowl_shock_abs"],
        h_throat=throat_k["h_throat"],
        ramp2_end_point=geom["F"],
        leading_edge_angle_deg= -leading_edge_angle_deg
    )

    diffuser = build_subsonic_diffuser(
        T_upper=throat_geom["T_upper"],
        T_lower=throat_geom["T_lower"],
        h_throat=throat_k["h_throat"],
        width_m=width_m,
        area_ratio_exit_to_throat=DIFFUSER_AREA_RATIO,
        half_angle_deg=DIFFUSER_HALF_ANGLE_DEG,
    )

    # Report actual Kantrowitz using M2 and chosen throat
    kant = {
        "A_capture_k": throat_k["A_capture_k"],
        "CR_geom": throat_k["A_capture_k"] / throat_k["A_throat"],
        "CR_k": throat_k["CR_k"],
        "kantrowitz_pass": (throat_k["A_capture_k"] / throat_k["A_throat"]) <= throat_k["CR_k"],
    }

    pt_frac_after_forebody = forebody["pt_fore_ratio"]
    pt_frac_after_shock1 = forebody["pt_fore_ratio"] * ramp1["pt_ratio"]
    pt_frac_after_shock2 = forebody["pt_fore_ratio"] * ramp1["pt_ratio"] * ramp2["pt_ratio"]

    return {
        "success": True,

        "rho0_kgm3": fs["rho0"],
        "V0_ms": fs["V0"],
        "A_capture_required_m2": A_capture_required,
        "opening_normal_to_ramp2_m": h_req_normal,

        "alpha_deg": alpha_deg,
        "leading_edge_angle_deg": leading_edge_angle_deg,
        "forebody_separation_margin": forebody_separation_margin,
        "ramp_separation_margin": ramp_separation_margin,
        "kantrowitz_margin": kantrowitz_margin,

        "theta_fore_deg": forebody["theta_fore"],
        "dtheta_fore_eff_deg": forebody["dtheta_fore_eff"],
        "theta1_deg": ramp1["theta_abs"],
        "theta2_deg": ramp2["theta_abs"],
        "theta_cowl_deg": cowl["theta_cowl"],
        "dtheta1_deg": ramp1["dtheta"],
        "dtheta2_deg": ramp2["dtheta"],
        "cowl_turn_mag_deg": cowl["cowl_turn_mag"],

        "beta_fore_rel_deg": forebody["beta_fore_rel"],
        "beta1_rel_deg": ramp1["beta_rel"],
        "beta2_rel_deg": ramp2["beta_rel"],
        "beta_cowl_rel_deg": cowl["beta_cowl_rel"],

        "shock_fore_abs_deg": forebody["shock_fore_abs"],
        "shock1_abs_deg": ramp1["shock_abs"],
        "shock2_abs_deg": ramp2["shock_abs"],
        "cowl_shock_abs_deg": cowl["cowl_shock_abs"],

        "M_after_forebody_shock": forebody["M_fore"],
        "M_after_shock1": ramp1["M_down"],
        "M_after_shock2": ramp2["M_down"],
        "M_after_cowl_shock": cowl["M3"],
        "M_target_after_cowl_shock": cowl["M3_target"],
        "M_after_cowl_shock_error": cowl["M3_error"],
        "M_after_immediate_normal_shock": normal_after_cowl["M4"],

        "V_after_immediate_normal_shock_ms": normal_after_cowl["V4"],

        "pt_frac_after_forebody_shock": pt_frac_after_forebody,
        "pt_frac_after_shock1": pt_frac_after_shock1,
        "pt_frac_after_shock2": pt_frac_after_shock2,
        "pt_frac_after_cowl_shock": pt_frac_after_cowl,
        "pt_frac_after_immediate_normal_shock": normal_after_cowl["pt_frac_after_immediate_normal"],

        "forebody_xy": P_fore,
        "nose_xy": geom["P0"],
        "break2_xy": geom["P1"],
        "cowl_lip_xy": geom["C"],
        "ramp2_normal_foot_xy": geom["F"],
        "shock_focus_xy": geom["focus_point"],

        "throat_upper_body_x_m": throat_geom["throat_upper_body_x_m"],
        "throat_upper_body_y_m": throat_geom["throat_upper_body_y_m"],
        "post_cowl_area_m2": throat_k["A_capture_k"],
        "post_cowl_height_m": h_req_normal,
        "A_over_Astar_post_cowl": post_cowl_target["A_over_Astar_target"],

        "throat_area_ideal_m2": throat_k["A_throat"] / kantrowitz_margin,
        "throat_area_actual_m2": throat_k["A_throat"],
        "throat_height_m": throat_k["h_throat"],
        "throat_upper_xy": throat_geom["T_upper"],
        "throat_lower_xy": throat_geom["T_lower"],

        "capture_area_kantrowitz_m2": throat_k["A_capture_k"],
        "geometric_contraction_ratio": kant["CR_geom"],
        "kantrowitz_limit_CR": kant["CR_k"],
        "kantrowitz_pass": kant["kantrowitz_pass"],

        "forebody_length_m": float(np.linalg.norm(geom["P0"] - P_fore)),
        "ramp1_length_m": float(np.linalg.norm(geom["P1"] - geom["P0"])),
        "shock2_distance_from_break2_to_lip_m": geom["lam2_lip"],
        "shock2_distance_from_break2_to_focus_m": geom["lam2_focus"],
        "shock_focus_factor": shock_focus_factor,

        "diffuser": diffuser,
        "combustor_face_xy_upper": diffuser["exit_upper_xy"],
        "combustor_face_xy_lower": diffuser["exit_lower_xy"],
        "A_combustor_face": diffuser["A_exit"],
    }

def print_d2r_report(result):
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
    print(f"Forebody separation margin      = {result['forebody_separation_margin']:.3f}")
    print(f"Ramp separation margin          = {result['ramp_separation_margin']:.3f}")
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


    print("\nTHROAT")
    print("------------------------------")
    print(f"Post-cowl area                  = {result['post_cowl_area_m2']:.6f} m^2")
    print(f"A/A* at post-cowl station       = {result['A_over_Astar_post_cowl']:.6f}")
    print(f"Ideal throat area               = {result['throat_area_ideal_m2']:.6f} m^2")
    print(f"Actual throat area              = {result['throat_area_actual_m2']:.6f} m^2")
    print(f"Throat height                   = {result['throat_height_m']:.6f} m")
    print(f"Throat lower point              = ({result['throat_lower_xy'][0]:.6f}, {result['throat_lower_xy'][1]:.6f}) m")
    print(f"Throat upper point              = ({result['throat_upper_xy'][0]:.6f}, {result['throat_upper_xy'][1]:.6f}) m")
    print(f"Throat top body-frame x          = {result['throat_upper_body_x_m']:.6f} m")
    print(f"Throat top body-frame y          = {result['throat_upper_body_y_m']:.6f} m")

    print("\nKANTROWITZ CHECK")
    print("------------------------------")
    print(f"Capture area for check          = {result['capture_area_kantrowitz_m2']:.6f} m^2")
    print(f"Geometric contraction ratio     = {result['geometric_contraction_ratio']:.6f}")
    print(f"Kantrowitz limit                = {result['kantrowitz_limit_CR']:.6f}")
    print(f"Kantrowitz pass                 = {result['kantrowitz_pass']}")

    print("\n==============================\n")

def plot_2ramp_shock_matched_inlet(result,shock_extension_factor=1.20,
    cowl_extension_factor=1.35,):

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

    x_end = shock_extension_factor * max(T_upper[0], C[0], focus[0])
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

    cowl_len = cowl_extension_factor * max(T_upper[0] - C[0], 0.2)
    cowl_end = C + cowl_len * cowl_dir

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Geometry
    ax.plot([P_fore[0], P0[0]], [P_fore[1], P0[1]],
        linewidth=2, label="Forebody")
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]],
        linewidth=2, label="Ramp 1")
    ax.plot([P1[0], F[0]], [P1[1], F[1]],
        linewidth=2, label="Ramp 2")
    ax.plot([F[0], T_lower[0]], [F[1], T_lower[1]],
        linewidth=2, label="Internal sharp edge")
    ax.plot([F[0], C[0]], [F[1], C[1]],
        linewidth=2, label="Opening (normal to Ramp 2)")
    ax.plot([C[0], cowl_end[0]], [C[1], cowl_end[1]],
        linewidth=2, label="Angled cowl")
    ax.plot([T_lower[0], T_upper[0]], [T_lower[1], T_upper[1]],
        linewidth=2, label="Throat Shock", linestyle= "--")

    # Shocks
    ax.plot([P_fore[0], x_end], [P_fore[1], y_fore_end],
        linestyle="--", linewidth=1.5, label="Forebody shock")
    ax.plot([P0[0], x_end], [P0[1], y_s1_end],
        linestyle="--", linewidth=1.5, label="Ramp 1 shock")
    ax.plot([P1[0], x_end], [P1[1], y_s2_end],
        linestyle="--", linewidth=1.5, label="Ramp 2 shock")
    ax.plot([C[0], x_end], [C[1], y_sc_end],
        linestyle="--", linewidth=1.5, label="Cowl shock")

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

    y_max = max(C[1], F[1], y_fore_end, y_s1_end, y_s2_end,
        y_sc_end, T_upper[1], focus[1], cowl_end[1])
    y_min = min(P_fore[1], P0[1], P1[1], F[1], C[1],
        y_sc_end, T_lower[1], focus[1], cowl_end[1], 0.0)

    ax.set_xlim(P_fore[0] - 0.02 * (x_end - P_fore[0]),
        1.05 * max(x_end, cowl_end[0], T_upper[0]))
    ax.set_ylim(y_min - 0.05 * max(y_max - y_min, 1e-6),
        y_max + 0.15 * max(y_max - y_min, 1e-6))

    plt.show()

def sweep_d2r_margins(M0,altitude_m,alpha_deg,leading_edge_angle_deg,
    mdot_required,width_m,forebody_margin_values,ramp_margin_values,
    kantrowitz_margin=0.95,shock_focus_factor=1.25,verbose=True,):

    best_result = None
    best_forebody_margin = None
    best_ramp_margin = None
    best_pt = -float("inf")
    records = []

    for fb_margin in forebody_margin_values:
        for rp_margin in ramp_margin_values:
            try:
                result = design_2ramp_shock_matched_inlet(
                    M0=M0,
                    altitude_m=altitude_m,
                    alpha_deg=alpha_deg,
                    leading_edge_angle_deg=leading_edge_angle_deg,
                    mdot_required=mdot_required,
                    width_m=width_m,
                    forebody_separation_margin=fb_margin,
                    ramp_separation_margin=rp_margin,
                    kantrowitz_margin=kantrowitz_margin,
                    shock_focus_factor=shock_focus_factor,
                )

                pt = result["pt_frac_after_immediate_normal_shock"]

                records.append({
                    "success": True,
                    "forebody_separation_margin": fb_margin,
                    "ramp_separation_margin": rp_margin,
                    "pt_frac_after_cowl_shock": pt,
                    "kantrowitz_pass": result.get("kantrowitz_pass", None),
                })

                if pt > best_pt:
                    best_pt = pt
                    best_result = result
                    best_forebody_margin = fb_margin
                    best_ramp_margin = rp_margin

                if verbose:
                    print(
                        f"OK  fb={fb_margin:.4f}, rp={rp_margin:.4f}, "
                        f"pt/pt0={pt:.6f}, Kantrowitz={result.get('kantrowitz_pass', None)}"
                    )

            except Exception as e:
                records.append({
                    "success": False,
                    "forebody_separation_margin": fb_margin,
                    "ramp_separation_margin": rp_margin,
                    "error": str(e),
                })

                if verbose:
                    print(
                        f"FAIL fb={fb_margin:.4f}, rp={rp_margin:.4f}, error={e}"
                    )

                continue

    summary = {
        "best_result": best_result,
        "best_forebody_separation_margin": best_forebody_margin,
        "best_ramp_separation_margin": best_ramp_margin,
        "best_pt_frac_after_cowl_shock": best_pt if best_result is not None else None,
        "records": records,
    }

    return summary

def print_d2r_sweep_summary(summary,):
    print("\n==============================")
    print("      D2R SWEEP SUMMARY")
    print("==============================\n")

    if summary["best_result"] is None:
        print("No successful cases found.")
        return

    print(f"Best forebody sep margin        = {summary['best_forebody_separation_margin']:.6f}")
    print(f"Best ramp sep margin            = {summary['best_ramp_separation_margin']:.6f}")
    print(f"Best pt/pt0 after cowl shock    = {summary['best_pt_frac_after_cowl_shock']:.6f}")

    ok_count = sum(1 for r in summary["records"] if r["success"])
    fail_count = sum(1 for r in summary["records"] if not r["success"])

    print(f"Successful cases                = {ok_count}")
    print(f"Failed cases                    = {fail_count}")
    print()

def evaluate_fixed_geometry_at_condition(result, M0, altitude_m, alpha_deg,
                                         p_back):
    """
    Re-evaluate the shock system for a fixed geometry at a new flight
    condition AND back pressure. Geometry (including the subsonic diffuser
    from build_subsonic_diffuser) is frozen inside `result`.

    The terminal normal shock is no longer pinned to the cowl lip. Its axial
    station is solved from p_back via solve_terminal_shock_position.
    """
    T0, p0, rho0, a0 = std_atmosphere_1976(altitude_m)
    V0 = M0 * a0

    theta_fore = result["theta_fore_deg"]
    theta1     = result["theta1_deg"]
    theta2     = result["theta2_deg"]
    theta_cowl = result["theta_cowl_deg"]

    P_fore  = result["forebody_xy"]
    P0_xy   = result["nose_xy"]
    P1      = result["break2_xy"]
    C       = result["cowl_lip_xy"]
    F       = result["ramp2_normal_foot_xy"]
    T_lower = result["throat_lower_xy"]
    T_upper = result["throat_upper_xy"]
    focus   = result["shock_focus_xy"]

    # ---- Forebody shock
    dtheta_fore_eff = theta_fore + alpha_deg
    if dtheta_fore_eff <= 0.0:
        return {"success": False,
                "reason": "Non-positive forebody effective turn",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    shf = oblique_shock(M0, dtheta_fore_eff)
    if shf is None:
        return {"success": False, "reason": "Forebody shock unattached",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio = shf
    shock_fore_abs = beta_fore_rel

    # ---- Ramp 1
    dtheta1 = theta1 - theta_fore
    if dtheta1 <= 0.0:
        return {"success": False,
                "reason": "Invalid ramp 1 turn from frozen geometry",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    sh1 = oblique_shock(M_fore, dtheta1)
    if sh1 is None:
        return {"success": False, "reason": "Ramp 1 shock unattached",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    beta1_rel, M1, p21, pt21 = sh1
    shock1_abs = theta_fore + beta1_rel

    # ---- Ramp 2
    dtheta2 = theta2 - theta1
    if dtheta2 <= 0.0:
        return {"success": False,
                "reason": "Invalid ramp 2 turn from frozen geometry",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    sh2 = oblique_shock(M1, dtheta2)
    if sh2 is None:
        return {"success": False, "reason": "Ramp 2 shock unattached",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    beta2_rel, M2, p32, pt32 = sh2
    shock2_abs = theta1 + beta2_rel

    # ---- Cowl shock
    cowl_turn_mag = theta2 - theta_cowl
    if cowl_turn_mag <= 0.0:
        return {"success": False, "reason": "Non-positive cowl turn",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    shc = oblique_shock(M2, cowl_turn_mag)
    if shc is None:
        return {"success": False, "reason": "Cowl shock unattached",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
    beta_cowl_rel, M3, p43, pt43 = shc
    cowl_shock_abs = theta2 - beta_cowl_rel

    pt_frac_after_forebody = pt_fore_ratio
    pt_frac_after_shock1   = pt_fore_ratio * pt21
    pt_frac_after_shock2   = pt_fore_ratio * pt21 * pt32
    pt_frac_after_cowl     = pt_fore_ratio * pt21 * pt32 * pt43

    Tt0 = total_temperature(T0, M0)
    Pt0 = p0 * (1.0 + 0.5 * (GAMMA - 1.0) * M0 * M0) ** (GAMMA / (GAMMA - 1.0))
    Pt_after_cowl = Pt0 * pt_frac_after_cowl

    # ---- Terminal shock: driven by back pressure inside the diffuser.
    if "diffuser" not in result:
        return {"success": False,
                "reason": "Frozen geometry has no diffuser block. Rebuild "
                          "with build_subsonic_diffuser first.",
                "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}

    terminal = solve_terminal_shock_position(result, p_back,
                                             Pt_after_cowl, Tt0)

    if terminal["status"] in ("expelled", "swallowed"):
        return {
            "success":        False,
            "reason":         f"Terminal shock {terminal['status']} "
                              f"(p_back={p_back:.1f} Pa, "
                              f"Ps range=[{terminal['Ps_min']:.1f}, "
                              f"{terminal['Ps_max']:.1f}] Pa).",
            "status":         terminal["status"],
            "M0":             M0,
            "alpha_deg":      alpha_deg,
            "p_back":         p_back,
            "Pt_after_cowl":  Pt_after_cowl,
            "terminal":       terminal,
        }

    pt_frac_after_terminal = (pt_frac_after_cowl
                              * terminal["pt_frac_after_terminal_shock"])

    return {
        "success":        True,
        "status":         terminal["status"],
        "M0":             M0,
        "alpha_deg":      alpha_deg,
        "p_back":         p_back,
        "V0_ms":          V0,

        "theta_fore_deg":  theta_fore,
        "theta1_deg":      theta1,
        "theta2_deg":      theta2,
        "theta_cowl_deg":  theta_cowl,

        "shock_fore_abs_deg": shock_fore_abs,
        "shock1_abs_deg":     shock1_abs,
        "shock2_abs_deg":     shock2_abs,
        "cowl_shock_abs_deg": cowl_shock_abs,

        "M_after_forebody_shock":  M_fore,
        "M_after_shock1":          M1,
        "M_after_shock2":          M2,
        "M_after_cowl_shock":      M3,

        "x_terminal_shock":            terminal["x_s"],
        "A_at_terminal_shock":         terminal["A_s"],
        "M_before_terminal_shock":     terminal["M_sup"],
        "M_after_terminal_shock":      terminal["M_sub"],
        "M_at_combustor_face":         terminal["M_exit"],
        "Ps_at_combustor_face":        terminal["Ps_exit"],
        "V_at_combustor_face_ms":      terminal["V_exit"],

        "pt_frac_after_forebody_shock":   pt_frac_after_forebody,
        "pt_frac_after_shock1":           pt_frac_after_shock1,
        "pt_frac_after_shock2":           pt_frac_after_shock2,
        "pt_frac_after_cowl_shock":       pt_frac_after_cowl,
        "pt_frac_after_terminal_shock":   pt_frac_after_terminal,

        "pt_frac_after_immediate_normal_shock": pt_frac_after_terminal,
        "M_after_immediate_normal_shock":       terminal["M_exit"],
        "V_after_immediate_normal_shock_ms":    terminal["V_exit"],

        "forebody_xy":         P_fore,
        "nose_xy":             P0_xy,
        "break2_xy":           P1,
        "cowl_lip_xy":         C,
        "ramp2_normal_foot_xy": F,
        "throat_lower_xy":     T_lower,
        "throat_upper_xy":     T_upper,
        "shock_focus_xy":      focus,

        "diffuser":            result["diffuser"],
        "Pt_after_cowl":       Pt_after_cowl,
        "Tt0":                 Tt0,
    }

def plot_fixed_geometry_case(ax,case,shock_extension_factor=1.20,
    cowl_extension_factor=1.35,):
    """
    Plot one frozen-geometry case on an existing axis.
    """
    P_fore = case["forebody_xy"]
    P0 = case["nose_xy"]
    P1 = case["break2_xy"]
    C = case["cowl_lip_xy"]
    F = case["ramp2_normal_foot_xy"]
    T_lower = case["throat_lower_xy"]
    T_upper = case["throat_upper_xy"]
    focus = case["shock_focus_xy"]

    shock_fore_abs = case["shock_fore_abs_deg"]
    shock1_abs = case["shock1_abs_deg"]
    shock2_abs = case["shock2_abs_deg"]
    cowl_shock_abs = case["cowl_shock_abs_deg"]
    theta_cowl = case["theta_cowl_deg"]

    x_end = shock_extension_factor * max(T_upper[0], C[0], focus[0])
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

    cowl_len = cowl_extension_factor * max(T_upper[0] - C[0], 0.2)
    cowl_end = C + cowl_len * cowl_dir

    ax.plot([P_fore[0], P0[0]], [P_fore[1], P0[1]], linewidth=1.5)
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], linewidth=1.5)
    ax.plot([P1[0], F[0]], [P1[1], F[1]], linewidth=1.5)
    ax.plot([F[0], T_lower[0]], [F[1], T_lower[1]], linewidth=1.5)
    ax.plot([F[0], C[0]], [F[1], C[1]], linewidth=1.5)
    ax.plot([C[0], cowl_end[0]], [C[1], cowl_end[1]], linewidth=1.5)
    ax.plot([T_lower[0], T_upper[0]], [T_lower[1], T_upper[1]], linewidth=1.5, linestyle=":")

    ax.plot([P_fore[0], x_end], [P_fore[1], y_fore_end], linestyle="--", linewidth=1.0)
    ax.plot([P0[0], x_end], [P0[1], y_s1_end], linestyle="--", linewidth=1.0)
    ax.plot([P1[0], x_end], [P1[1], y_s2_end], linestyle="--", linewidth=1.0)
    ax.plot([C[0], x_end], [C[1], y_sc_end], linestyle="--", linewidth=1.0)

    ax.plot(focus[0], focus[1], marker="x", markersize=5)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(
        f"M={case['M0']:.2f}, α={case['alpha_deg']:.1f}\n"
        f"pt/pt0={case['pt_frac_after_immediate_normal_shock']:.3f}")

def plot_fixed_geometry_3x3_grid(result,altitude_m,mach_values,alpha_values,p_back,):
    """
    Plot a 3x3 grid of frozen-geometry shock systems.

    mach_values: length 3
    alpha_values: length 3
    """
    if len(mach_values) != 3 or len(alpha_values) != 3:
        raise ValueError("mach_values and alpha_values must each have length 3.")

    fig, axes = plt.subplots(3,3,figsize=(14,12))

    for i, alpha_deg in enumerate(alpha_values):
        for j, M0 in enumerate(mach_values):
            ax = axes[i, j]

            case = evaluate_fixed_geometry_at_condition(
                result=result,
                M0=M0,
                altitude_m=altitude_m,
                alpha_deg=alpha_deg,
                p_back=p_back,
            )

            if case["success"]:
                plot_fixed_geometry_case(ax,case)
            else:
                ax.text(0.5,0.5,
                    f"M={M0:.2f}, α={alpha_deg:.1f}\nFAIL\n{case['reason']}",
                    ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"M={M0:.2f}, α={alpha_deg:.1f}")
                ax.grid(True)

    plt.tight_layout()
    plt.show()

def sweep_fixed_geometry_vs_mach(result,altitude_m,mach_values,alpha_deg,p_back,):
    """
    Recompute frozen-geometry performance vs Mach at fixed alpha.
    """
    out = []

    for M0 in mach_values:
        case = evaluate_fixed_geometry_at_condition(
            result=result,
            M0=M0,
            altitude_m=altitude_m,
            alpha_deg=alpha_deg,
            p_back=p_back,
        )
        out.append(case)

    return out

def sweep_fixed_geometry_vs_alpha(result,altitude_m,alpha_values,M0,p_back,):
    """
    Recompute frozen-geometry performance vs alpha at fixed Mach.
    """
    out = []

    for alpha_deg in alpha_values:
        case = evaluate_fixed_geometry_at_condition(
            result=result,
            M0=M0,
            altitude_m=altitude_m,
            alpha_deg=alpha_deg,
            p_back=p_back,
        )
        out.append(case)

    return out

def plot_pt_vs_mach(cases,use_immediate_normal=False,):
    """
    Plot total pressure recovery vs Mach.
    Also plots MIL-E-5008B:
        pt/pt0 = 1 - 0.075 * (M - 1)^1.35
    """
    xs = []
    ys = []

    for case in cases:
        if not case["success"]:
            continue
        xs.append(case["M0"])
        if use_immediate_normal:
            ys.append(case["pt_frac_after_immediate_normal_shock"])
        else:
            ys.append(case["pt_frac_after_cowl_shock"])

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o", label="Computed")

    if len(xs) > 0:
        xs_ref = np.linspace(min(xs), max(xs), 300)
        ys_ref = 1.0 - 0.075 * (xs_ref - 1.0) ** 1.35
        plt.plot(xs_ref, ys_ref, label="MIL-E-5008B")

    plt.grid(True)
    plt.xlabel("Freestream Mach")
    plt.ylabel("pt/pt0")
    plt.title("Total Pressure Recovery vs Mach")
    plt.legend()
    plt.show()

def plot_pt_vs_alpha(cases,use_immediate_normal=False,):
    """
    Plot total pressure recovery vs alpha.
    """
    xs = []
    ys = []

    for case in cases:
        if not case["success"]:
            continue
        xs.append(case["alpha_deg"])
        if use_immediate_normal:
            ys.append(case["pt_frac_after_immediate_normal_shock"])
        else:
            ys.append(case["pt_frac_after_cowl_shock"])

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.grid(True)
    plt.xlabel("Alpha [deg]")
    plt.ylabel("pt/pt0")
    plt.title("Total Pressure Recovery vs Alpha")
    plt.show()



##SWEEP METHOD TO FIND BEST GEOMETRY
Sweep = True
Sweep = False    #Comment out this line when new cruise conditions are used to first determine the best geometry
if Sweep:
    if __name__ == "__main__":
        forebody_margin_values = np.linspace(0.2, 0.3, 10)
        ramp_margin_values = np.linspace(0.25, 0.35, 10)

        sweep = sweep_d2r_margins(
            M0=5.0,
            altitude_m=12000.0,
            alpha_deg=2.0,
            leading_edge_angle_deg=6.0,
            mdot_required=6.0,
            width_m=0.25,
            forebody_margin_values=forebody_margin_values,
            ramp_margin_values=ramp_margin_values,
            kantrowitz_margin=0.95,
            shock_focus_factor=1.25,
            verbose=True,
        )

        print_d2r_sweep_summary(sweep)

        if sweep["best_result"] is not None:
            print_d2r_report(sweep["best_result"])
            plot_2ramp_shock_matched_inlet(sweep["best_result"])
else:
###PLOT A SINGLE GEOMETRY
    if __name__ == "__main__":
        result = design_2ramp_shock_matched_inlet(
            M0=5.0,
            altitude_m=12000.0,
            alpha_deg=2.0,
            leading_edge_angle_deg=6.0,
            mdot_required=6.0,
            width_m=0.25,
            forebody_separation_margin=0.2,
            ramp_separation_margin=0.28,
            kantrowitz_margin=0.95,
            shock_focus_factor=1.25,
        )

        print_d2r_report(result)
        plot_2ramp_shock_matched_inlet(result)

        # 3x3 shock grid
        mach_grid = [4.0, 4.5, 5.0]
        alpha_grid = [2.0, 5.0, 8.0]
        plot_fixed_geometry_3x3_grid(
            result=result,
            altitude_m=12000.0,
            mach_values=mach_grid,
            alpha_values=alpha_grid,
        )

        # pt vs Mach at fixed alpha
        mach_sweep = np.linspace(4.0, 5.0, 15)
        cases_mach = sweep_fixed_geometry_vs_mach(
            result=result,
            altitude_m=12000.0,
            mach_values=mach_sweep,
            alpha_deg=5.0,
        )
        plot_pt_vs_mach(cases_mach, use_immediate_normal=True)

        # pt vs alpha at fixed Mach
        alpha_sweep = np.linspace(0.0, 10.0, 17)
        cases_alpha = sweep_fixed_geometry_vs_alpha(
            result=result,
            altitude_m=12000.0,
            alpha_values=alpha_sweep,
            M0=5.0,
        )
        plot_pt_vs_alpha(cases_alpha, use_immediate_normal=True)