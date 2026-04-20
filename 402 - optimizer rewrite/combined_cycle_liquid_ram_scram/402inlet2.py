import math
import numpy as np
import matplotlib.pyplot as plt

from pyc_config import (
    AIR_GAMMA_REF as GAMMA,
    AIR_R as R,
    AIR_X_N2 as _X_N2_AIR,
    AIR_X_O2 as _X_O2_AIR,
    INLET_LEGACY_FOREBODY_SEP_MARGIN as _LEG_FB_MARGIN,
    INLET_LEGACY_RAMP_SEP_MARGIN as _LEG_RP_MARGIN,
    INLET_LEGACY_KANTROWITZ_MARGIN as _LEG_KZ_MARGIN,
    INLET_LEGACY_SHOCK_FOCUS_FACTOR as _LEG_FOCUS_FACTOR,
    INLET_SHOCK_EXTENSION_FACTOR as _SHOCK_EXT_FACTOR,
    INLET_COWL_EXTENSION_FACTOR as _COWL_EXT_FACTOR,
    INLET_COWL_MIN_LENGTH_M as _COWL_MIN_LEN_M,
)

# ----------------------------------------------------------------------------
# Thermally-perfect frozen-air specific heat ratio, gamma(T)
#
# NASA-7 polynomial cp/R coefficients for N2 and O2 (from CEA thermo.inp).
# Air is treated as a frozen mixture of 79% N2 + 21% O2 by mole (ignoring
# Ar/CO2 -- <1% effect on cp). Valid 200--6000 K. This matches the frozen-
# chemistry assumption of the classical Rankine-Hugoniot shock relations
# used in this file: cp is temperature-dependent (vibrational modes excite),
# but chemistry is frozen across the shock. Equilibrium (dissociation) is
# NOT included -- above ~3000 K the recovery computed here is still
# optimistic, but by a few percent rather than 10-15%.
# ----------------------------------------------------------------------------
_NASA7_N2_LO = (3.53100528, -1.23660987e-4, -5.02999437e-7,  2.43530612e-9,  -1.40881235e-12)
_NASA7_N2_HI = (2.95257626,  1.39690057e-3, -4.92631691e-7,  7.86010190e-11, -4.60755348e-15)
_NASA7_O2_LO = (3.78245636, -2.99673415e-3,  9.84730200e-6, -9.68129508e-9,   3.24372836e-12)
_NASA7_O2_HI = (3.66096083,  6.56365523e-4, -1.41149485e-7,  2.05797658e-11, -1.29913248e-15)


def _cp_over_R_nasa7(T, lo, hi):
    a = lo if T < 1000.0 else hi
    return a[0] + a[1] * T + a[2] * T * T + a[3] * T ** 3 + a[4] * T ** 4


def gamma_air(T):
    """
    Frozen-air specific heat ratio gamma = cp/cv as a function of static
    temperature T [K]. Uses NASA-7 polynomials for N2/O2 at standard mole
    fractions. Clamped to the polynomial's validity range [200, 6000] K.

    Spot values (for reference):
        T=  300 K -> 1.400
        T= 1000 K -> 1.336
        T= 1500 K -> 1.321
        T= 2500 K -> 1.297
    """
    T = max(200.0, min(float(T), 6000.0))
    cpR = (_X_N2_AIR * _cp_over_R_nasa7(T, _NASA7_N2_LO, _NASA7_N2_HI)
         + _X_O2_AIR * _cp_over_R_nasa7(T, _NASA7_O2_LO, _NASA7_O2_HI))
    return cpR / (cpR - 1.0)

# Subsonic-diffuser (throat -> combustor face) geometry defaults.
from pyc_config import (DIFFUSER_AREA_RATIO, DIFFUSER_HALF_ANGLE_DEG,
                        DIFFUSER_PHYSICS_EQUIV_HALF_ANGLE_DEG, DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH,
                        ETA_DIFFUSER)


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

def normal_shock_tpg(M1, T1, n_iter=6, tol=1e-3):
    """
    Thermally-perfect (frozen-chemistry, T-dependent gamma) normal shock.

    gamma is iterated using gamma_air evaluated at the arithmetic mean of
    the upstream and downstream static temperatures. This is the standard
    "effective constant-gamma" approximation for thermally perfect gases
    (Anderson, Modern Compressible Flow, 3rd ed., ch. 16).

    Returns: (M2, p2/p1, rho2/rho1, T2/T1, pt2/pt1, T2 [K], gamma_eff)
    """
    gamma = gamma_air(T1)
    T2 = T1
    for _ in range(n_iter):
        M2, p2p1, rho2rho1, T2T1, pt2pt1 = normal_shock(M1, gamma)
        T2_new = T1 * T2T1
        gamma_new = gamma_air(0.5 * (T1 + T2_new))
        if abs(gamma_new - gamma) < tol:
            gamma = gamma_new
            T2 = T2_new
            break
        gamma = gamma_new
        T2 = T2_new
    M2, p2p1, rho2rho1, T2T1, pt2pt1 = normal_shock(M1, gamma)
    return M2, p2p1, rho2rho1, T2T1, pt2pt1, T1 * T2T1, gamma


def oblique_shock_tpg(M1, theta_deg, T1, n_iter=6, tol=1e-3):
    """
    Thermally-perfect oblique shock. gamma is iterated at the mean static T
    across the shock (see normal_shock_tpg).

    Returns: (beta_deg, M2, p2/p1, pt2/pt1, T2/T1, T2 [K], gamma_eff)
    or None if no attached weak solution exists at the converged gamma.
    """
    theta = math.radians(theta_deg)
    gamma = gamma_air(T1)
    last = None
    for _ in range(n_iter):
        beta = solve_weak_beta(M1, theta, gamma)
        if beta is None:
            return None
        Mn1 = M1 * math.sin(beta)
        Mn2, p2p1, _, T2T1, pt2pt1 = normal_shock(Mn1, gamma)
        M2 = Mn2 / math.sin(beta - theta)
        T2 = T1 * T2T1
        last = (math.degrees(beta), M2, p2p1, pt2pt1, T2T1, T2, gamma)
        gamma_new = gamma_air(0.5 * (T1 + T2))
        if abs(gamma_new - gamma) < tol:
            break
        gamma = gamma_new
    return last


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


def _rectangle_polar_boundary(theta, half_width, half_height):
    c = math.cos(theta)
    s = math.sin(theta)
    denom_x = max(abs(c), 1.0e-12)
    denom_y = max(abs(s), 1.0e-12)
    ray = min(half_width / denom_x, half_height / denom_y)
    return np.array([ray * c, ray * s], dtype=float)


def _section_polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def morphed_rectangle_to_circle_section(
    area_m2,
    blend,
    rect_width_m,
    rect_height_m,
    circle_radius_m,
    n_points=128,
):
    """
    Build a smooth cross-section polygon that morphs from a rectangle at
    blend=0 to a circle at blend=1, then uniformly rescales it to the target
    area. The polygon lives in the (z, y) plane and is centered at the origin.
    """
    if area_m2 <= 0.0:
        raise ValueError("area_m2 must be positive.")
    if rect_width_m <= 0.0 or rect_height_m <= 0.0 or circle_radius_m <= 0.0:
        raise ValueError("Section dimensions must be positive.")

    blend = float(np.clip(blend, 0.0, 1.0))
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    half_width = 0.5 * rect_width_m
    half_height = 0.5 * rect_height_m

    pts = []
    for theta in angles:
        rect_pt = _rectangle_polar_boundary(theta, half_width, half_height)
        circ_pt = np.array(
            [circle_radius_m * math.cos(theta), circle_radius_m * math.sin(theta)],
            dtype=float,
        )
        pts.append((1.0 - blend) * rect_pt + blend * circ_pt)

    pts = np.asarray(pts, dtype=float)
    raw_area = _section_polygon_area(pts)
    scale = math.sqrt(area_m2 / max(raw_area, 1.0e-12))
    return pts * scale


def _quintic_smoothstep(u):
    """
    C2-continuous 0->1 blend with zero slope and curvature at both ends.
    """
    u = float(np.clip(u, 0.0, 1.0))
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def hydraulic_diameter_rect(width_m, height_m):
    area = width_m * height_m
    wetted_perimeter = 2.0 * (width_m + height_m)
    return 4.0 * area / max(wetted_perimeter, 1.0e-12)


def size_diffuser_length_physics_based(
    A_throat,
    A_exit,
    throat_width_m,
    throat_height_m,
    max_equiv_half_angle_deg=DIFFUSER_PHYSICS_EQUIV_HALF_ANGLE_DEG,
    min_shock_accommodation_dh=DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH,
):
    """
    Size diffuser length from the physics the present 1D model can support.

    In this model, expulsion/swallow thresholds are governed by diffuser
    area ratio; axial length only determines how much room the terminal shock
    has to sit inside the diffuser. Length is therefore set by the larger of:

    1) a conservative equivalent circular half-angle to avoid subsonic
       diffuser separation, and
    2) a minimum shock-accommodation length measured in throat hydraulic
       diameters so the terminal-shock / interaction region has axial room.
    """
    if A_throat <= 0.0 or A_exit <= 0.0:
        raise ValueError("Diffuser areas must be positive.")
    if throat_width_m <= 0.0 or throat_height_m <= 0.0:
        raise ValueError("Throat dimensions must be positive.")
    if max_equiv_half_angle_deg <= 0.0:
        raise ValueError("max_equiv_half_angle_deg must be positive.")
    if min_shock_accommodation_dh <= 0.0:
        raise ValueError("min_shock_accommodation_dh must be positive.")

    r_throat = math.sqrt(A_throat / math.pi)
    r_exit = math.sqrt(A_exit / math.pi)
    radius_change = max(r_exit - r_throat, 0.0)

    length_from_angle = radius_change / math.tan(math.radians(max_equiv_half_angle_deg))
    dh_throat = hydraulic_diameter_rect(throat_width_m, throat_height_m)
    length_from_shock = min_shock_accommodation_dh * dh_throat

    if length_from_angle >= length_from_shock:
        governing_mode = "equivalent_half_angle"
    else:
        governing_mode = "shock_accommodation"

    return {
        "length_m": float(max(length_from_angle, length_from_shock)),
        "governing_mode": governing_mode,
        "max_equiv_half_angle_deg": float(max_equiv_half_angle_deg),
        "min_shock_accommodation_dh": float(min_shock_accommodation_dh),
        "length_from_angle_m": float(length_from_angle),
        "length_from_shock_m": float(length_from_shock),
        "throat_hydraulic_diameter_m": float(dh_throat),
        "throat_equivalent_radius_m": float(r_throat),
        "exit_equivalent_radius_m": float(r_exit),
    }

def solve_forebody_stage(
    M0,alpha_deg,leading_edge_angle_deg,forebody_separation_margin,
    T0=None,):
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

    # ACTUAL SHOCK (thermally perfect if T0 supplied; legacy gamma=1.4 otherwise)
    if T0 is None:
        shf = oblique_shock(M0, dtheta_fore_eff)
        if shf is None:
            raise ValueError("Forebody has no attached weak shock solution.")
        beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio = shf
        T_fore = None
        T_ratio_fore = None
        gamma_eff_fore = GAMMA
    else:
        shf = oblique_shock_tpg(M0, dtheta_fore_eff, T0)
        if shf is None:
            raise ValueError("Forebody has no attached weak shock solution.")
        beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio, T_ratio_fore, T_fore, gamma_eff_fore = shf

    # MARGIN SHOCK (same gamma treatment as actual shock for consistency)
    if T0 is None:
        shf_margin = oblique_shock(M0, dtheta_fore_margin)
    else:
        shf_margin = oblique_shock_tpg(M0, dtheta_fore_margin, T0)
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

        "T_ratio": T_ratio_fore,
        "T_down": T_fore,
        "gamma_eff": gamma_eff_fore,

        "shock_fore_abs": beta_fore_rel,
        "shock_fore_margin_abs": beta_fore_margin_rel,
    }

def solve_ramp_stage(M_up, theta_base, ramp_separation_margin, stage_name,
                     T_up=None):
    dtheta_max = theta_max_attached(M_up, gamma=gamma_air(T_up) if T_up is not None else GAMMA)
    if dtheta_max is None:
        raise ValueError(f"Could not determine attached-shock limit for {stage_name}.")

    dtheta = ramp_separation_margin * dtheta_max
    theta_abs = theta_base + dtheta

    if T_up is None:
        sh = oblique_shock(M_up, dtheta)
        if sh is None:
            raise ValueError(f"{stage_name} has no attached weak shock solution.")
        beta_rel, M_down, p_ratio, pt_ratio = sh
        T_down = None
        T_ratio = None
        gamma_eff = GAMMA
    else:
        sh = oblique_shock_tpg(M_up, dtheta, T_up)
        if sh is None:
            raise ValueError(f"{stage_name} has no attached weak shock solution.")
        beta_rel, M_down, p_ratio, pt_ratio, T_ratio, T_down, gamma_eff = sh

    shock_abs = theta_base + beta_rel

    return {
        "dtheta": dtheta,
        "theta_abs": theta_abs,
        "beta_rel": beta_rel,
        "M_down": M_down,
        "p_ratio": p_ratio,
        "pt_ratio": pt_ratio,
        "T_ratio": T_ratio,
        "T_down": T_down,
        "gamma_eff": gamma_eff,
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

def solve_cowl_stage(M2, theta2, leading_edge_angle_deg, T2=None):
    theta_cowl = -leading_edge_angle_deg
    cowl_turn_mag = theta2 - theta_cowl
    if cowl_turn_mag <= 0.0:
        raise ValueError("Cowl turn magnitude must be positive.")

    if T2 is None:
        shc = oblique_shock(M2, cowl_turn_mag)
        if shc is None:
            raise ValueError("Cowl shock has no attached weak shock solution.")
        beta_cowl_rel, M3, p43, pt43 = shc
        T_ratio = None
        T3 = None
        gamma_eff = GAMMA
    else:
        shc = oblique_shock_tpg(M2, cowl_turn_mag, T2)
        if shc is None:
            raise ValueError("Cowl shock has no attached weak shock solution.")
        beta_cowl_rel, M3, p43, pt43, T_ratio, T3, gamma_eff = shc

    cowl_shock_abs = theta2 - beta_cowl_rel

    return {
        "theta_cowl": theta_cowl,
        "cowl_turn_mag": cowl_turn_mag,
        "beta_cowl_rel": beta_cowl_rel,
        "M3": M3,
        "p43": p43,
        "pt43": pt43,
        "T_ratio": T_ratio,
        "T_down": T3,
        "gamma_eff": gamma_eff,
        "cowl_shock_abs": cowl_shock_abs,
    }

def solve_immediate_normal_after_cowl(M3, pt_frac_after_cowl, Tt0, T3=None):
    """
    Immediate normal shock downstream of the cowl-shock region.

    If T3 (static temperature upstream of the normal shock) is provided,
    the shock is solved with temperature-dependent gamma via
    normal_shock_tpg. The downstream static T4 then uses the converged
    gamma, preserving adiabatic energy balance h_t = cp*Tt = const within
    the thermally-perfect approximation.
    """
    if T3 is None:
        M4, p54, rho54, T54, pt54 = normal_shock(M3)
        pt_frac_after_immediate_normal = pt_frac_after_cowl * pt54
        T4 = Tt0 / (1.0 + 0.5 * (GAMMA - 1.0) * M4 ** 2)
        a4 = math.sqrt(GAMMA * R * T4)
        gamma_eff = GAMMA
    else:
        M4, p54, rho54, T54, pt54, T4, gamma_eff = normal_shock_tpg(M3, T3)
        pt_frac_after_immediate_normal = pt_frac_after_cowl * pt54
        a4 = math.sqrt(gamma_eff * R * T4)

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
        "gamma_eff": gamma_eff,
    }

def solve_reflection_cascade(M3, T3, pt_frac_before, C, F, T_upper, T_lower,
                             theta_cowl_deg, max_reflections=20):
    """
    R2 isolator pressure-loss model: oblique-shock reflection cascade.

    Supersonic flow enters the isolator at direction theta_cowl_deg with
    state (M3, T3) and pt_frac_before (cumulative pt/pt0 up to and including
    the cowl shock). It reflects alternately off the floor (F -> T_lower)
    and the roof (C -> T_upper). Each reflection is a wall-aligned oblique
    shock solved with gamma(T). The cascade terminates when:
      - the flow is subsonic (accept subsonic exit),
      - the wall turn exceeds the detachment limit (cap with normal shock),
      - max_reflections is reached (cap with normal shock).

    Returns the subsonic exit state and a list of reflection records suitable
    for plotting.
    """
    phi_floor = math.degrees(math.atan2(T_lower[1] - F[1], T_lower[0] - F[0]))
    phi_roof  = math.degrees(math.atan2(T_upper[1] - C[1], T_upper[0] - C[0]))

    M = float(M3)
    T = float(T3)
    pt_frac = float(pt_frac_before)
    flow_dir = float(theta_cowl_deg)
    reflections = []
    wall = "floor"  # cowl shock emanates from roof, first reflection on floor

    # Geometric bounce points: start at F (cowl shock focus on floor in design)
    point = np.array(F, dtype=float).copy()

    for _ in range(max_reflections):
        if M <= 1.0 + 1e-4:
            break
        if wall == "floor":
            turn = phi_floor - flow_dir
            wall_dir = phi_floor
        else:
            turn = flow_dir - phi_roof
            wall_dir = phi_roof

        if turn <= 1e-6:
            # Flow already parallel (or diverging) - no reflection shock
            flow_dir = wall_dir
            wall = "roof" if wall == "floor" else "floor"
            continue

        sh = oblique_shock_tpg(M, turn, T)
        if sh is None:
            ns = normal_shock_tpg(M, T)
            M_ns, _, _, _, pt_r, T_ns, _ = ns
            pt_frac *= pt_r
            reflections.append({
                "wall": wall, "turn_deg": turn, "detached": True,
                "x": float(point[0]), "y": float(point[1]),
                "M_up": M, "M_down": M_ns, "pt_ratio": pt_r,
            })
            M, T = M_ns, T_ns
            break

        beta_rel, M_new, _, pt_r, _, T_new, _ = sh
        pt_frac *= pt_r
        reflections.append({
            "wall": wall, "turn_deg": turn, "beta_rel_deg": beta_rel,
            "x": float(point[0]), "y": float(point[1]),
            "flow_dir_in_deg": flow_dir, "wall_dir_deg": wall_dir,
            "M_up": M, "M_down": M_new, "pt_ratio": pt_r,
        })
        M, T = M_new, T_new
        flow_dir = wall_dir
        wall = "roof" if wall == "floor" else "floor"

    # Cap any residual supersonic with a normal shock (isolator terminates)
    if M > 1.0:
        ns = normal_shock_tpg(M, T)
        M_ns, _, _, _, pt_r, T_ns, _ = ns
        pt_frac *= pt_r
        reflections.append({
            "wall": "terminal", "turn_deg": 0.0,
            "M_up": M, "M_down": M_ns, "pt_ratio": pt_r,
        })
        M, T = M_ns, T_ns

    gamma_eff = gamma_air(T)
    a = math.sqrt(gamma_eff * R * T)
    V = M * a
    return {
        "M_exit": M,
        "T_exit": T,
        "V_exit": V,
        "a_exit": a,
        "gamma_eff": gamma_eff,
        "pt_frac_after_cascade": pt_frac,
        "reflections": reflections,
        "phi_floor_deg": phi_floor,
        "phi_roof_deg": phi_roof,
        "n_reflections": len(reflections),
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

def invert_area_mach_ratio_supersonic(target_A_over_Astar, gamma=GAMMA,
                                      tol=1e-4, max_iter=50,):
    if target_A_over_Astar < 1.0:
        raise ValueError("Supersonic A/A* target must be >= 1.")

    lo = 1.0 + 1e-8
    hi = 50.0

    f_lo = area_mach_ratio(lo, gamma=gamma) - target_A_over_Astar
    f_hi = area_mach_ratio(hi, gamma=gamma) - target_A_over_Astar

    if f_lo > 0.0:
        return lo
    if f_hi < 0.0:
        raise ValueError("Could not bracket supersonic Mach from A/A* target.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = area_mach_ratio(mid, gamma=gamma) - target_A_over_Astar
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)

def size_throat_from_kantrowitz(M2,width_m,h_req_normal,kantrowitz_margin=_LEG_KZ_MARGIN,):
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

def solve_cowl_stage_for_target_postshock(M2,theta2,M3_target, T2=None,):
    gamma_up = gamma_air(T2) if T2 is not None else GAMMA
    theta_max = theta_max_attached(M2, gamma=gamma_up)
    if theta_max is None:
        raise ValueError("Could not determine attached-shock limit for cowl stage.")

    def eval_candidate(turn_deg):
        if T2 is None:
            sh = oblique_shock(M2, turn_deg)
            if sh is None:
                return None
            beta_rel, M3, p43, pt43 = sh
            T3 = None
            T_ratio = None
            gamma_eff = GAMMA
        else:
            sh = oblique_shock_tpg(M2, turn_deg, T2)
            if sh is None:
                return None
            beta_rel, M3, p43, pt43, T_ratio, T3, gamma_eff = sh

        err = abs(M3 - M3_target)

        return {
            "theta_cowl": theta2 - turn_deg,
            "cowl_turn_mag": turn_deg,
            "beta_cowl_rel": beta_rel,
            "M3": M3,
            "p43": p43,
            "pt43": pt43,
            "T_ratio": T_ratio,
            "T_down": T3,
            "gamma_eff": gamma_eff,
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
                            min_shock_accommodation_dh=None,
                            n_stations=201):
    """
    Build a straight-walled symmetric subsonic diffuser downstream of the
    throat. Exactly one of (half_angle_deg, length_m) must be provided.
    """
    import numpy as np, math

    if area_ratio_exit_to_throat <= 1.0:
        raise ValueError("Diffuser exit/throat area ratio must exceed 1.")
    A_throat = h_throat * width_m
    A_exit   = area_ratio_exit_to_throat * A_throat
    exit_radius_m = math.sqrt(A_exit / math.pi)
    exit_diameter_m = 2.0 * exit_radius_m

    throat_equiv_radius_m = math.sqrt(A_throat / math.pi)
    length_sizing = None
    if length_m is None and half_angle_deg is None:
        length_sizing = size_diffuser_length_physics_based(
            A_throat=A_throat,
            A_exit=A_exit,
            throat_width_m=width_m,
            throat_height_m=h_throat,
            min_shock_accommodation_dh=(
                DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH
                if min_shock_accommodation_dh is None
                else min_shock_accommodation_dh
            ),
        )
        length_m = length_sizing["length_m"]
    elif length_m is None:
        radius_change = max(exit_radius_m - throat_equiv_radius_m, 0.0)
        length_m = radius_change / math.tan(math.radians(half_angle_deg))
    elif half_angle_deg is not None:
        raise ValueError("Specify either half_angle_deg or length_m, not both.")

    x0 = 0.5 * (T_upper[0] + T_lower[0])
    y_mid = 0.5 * (T_upper[1] + T_lower[1])
    x_exit = x0 + length_m

    xs = np.linspace(x0, x_exit, n_stations)
    s  = (xs - x0) / length_m
    smooth_s = np.array([_quintic_smoothstep(si) for si in s], dtype=float)
    r_of_x = throat_equiv_radius_m + smooth_s * (exit_radius_m - throat_equiv_radius_m)
    A_of_x = math.pi * r_of_x**2
    section_half_heights = np.empty_like(A_of_x)
    for i, (Ai, si) in enumerate(zip(A_of_x, smooth_s)):
        section = morphed_rectangle_to_circle_section(
            area_m2=Ai,
            blend=si,
            rect_width_m=width_m,
            rect_height_m=h_throat,
            circle_radius_m=exit_radius_m,
        )
        section_half_heights[i] = np.max(np.abs(section[:, 1]))

    upper_wall = np.column_stack([xs, y_mid + section_half_heights])
    lower_wall = np.column_stack([xs, y_mid - section_half_heights])

    exit_upper = upper_wall[-1]
    exit_lower = lower_wall[-1]

    return {
        "x_throat":      x0,
        "x_exit":        x_exit,
        "length_m":      float(length_m),
        "A_throat":      float(A_throat),
        "A_exit":        float(A_exit),
        "h_throat":      float(h_throat),
        "h_exit":        float(2.0 * exit_radius_m),
        "area_ratio":    float(area_ratio_exit_to_throat),
        "width_m":       float(width_m),
        "throat_shape":  "rectangular",
        "exit_shape":    "circular",
        "shape_transition": "rectangle_to_circle",
        "throat_width_m": float(width_m),
        "throat_height_m": float(h_throat),
        "throat_equivalent_radius_m": float(throat_equiv_radius_m),
        "exit_radius_m": float(exit_radius_m),
        "exit_diameter_m": float(exit_diameter_m),
        "y_centerline":  float(y_mid),
        "x_stations":    xs,
        "A_stations":    A_of_x,
        "radius_stations": r_of_x,
        "h_stations":    2.0 * section_half_heights,
        "section_half_height_stations": section_half_heights,
        "section_blend_stations": smooth_s,
        "section_axial_fraction_stations": s,
        "profile_type": "quintic_smooth_radius",
        "length_sizing": length_sizing,
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


def _subsonic_mach_from_area_ratio(A_over_Astar, gamma=GAMMA,
                                   tol=1e-5, max_iter=50):
    """Subsonic branch of the isentropic area-Mach relation."""
    if A_over_Astar < 1.0:
        raise ValueError("A/A* must be >= 1 for a real subsonic solution.")
    lo, hi = 1e-6, 1.0 - 1e-8
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = area_mach_ratio(mid, gamma=gamma) - A_over_Astar
        if abs(f) < tol:
            return mid
        if f > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _iterate_supersonic_mach_tpg(A_over_Astar, Tt, n_iter=8, tol=1e-3):
    """
    Supersonic-branch Mach from A/A*, iterating gamma at the local static T
    implied by Tt and the current Mach.
    """
    gamma = gamma_air(Tt * 2.0 / (GAMMA + 1.0))  # warm start at throat-like T
    for _ in range(n_iter):
        M = invert_area_mach_ratio_supersonic(A_over_Astar, gamma=gamma)
        T = Tt / (1.0 + 0.5 * (gamma - 1.0) * M * M)
        gamma_new = gamma_air(T)
        if abs(gamma_new - gamma) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new
    M = invert_area_mach_ratio_supersonic(A_over_Astar, gamma=gamma)
    return M, gamma


def _iterate_subsonic_mach_tpg(A_over_Astar, Tt, n_iter=8, tol=1e-3):
    """
    Subsonic-branch Mach from A/A*, iterating gamma at local static T.
    """
    gamma = gamma_air(Tt)  # warm start near stagnation
    for _ in range(n_iter):
        M = _subsonic_mach_from_area_ratio(A_over_Astar, gamma=gamma)
        T = Tt / (1.0 + 0.5 * (gamma - 1.0) * M * M)
        gamma_new = gamma_air(T)
        if abs(gamma_new - gamma) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new
    M = _subsonic_mach_from_area_ratio(A_over_Astar, gamma=gamma)
    return M, gamma


def _diffuser_subsonic_exit_state(diffuser, M_in, T_in, Pt_in, Tt0,
                                  eta_diffuser=ETA_DIFFUSER):
    """
    Subsonic diffuser exit state (station 3, combustor face) for the
    cascade-only model: the reflection cascade has already made the flow
    subsonic (or sonic after its terminal normal shock), so the diffuser is
    a pure area-ruled subsonic diffusion with a small friction Pt loss.

    Parameters
    ----------
    diffuser      : result["diffuser"] block (has A_throat, A_exit).
    M_in, T_in    : cascade exit Mach, static T (subsonic by construction).
    Pt_in         : cascade exit total pressure [Pa].
    Tt0           : freestream total temperature [K].
    eta_diffuser  : subsonic diffuser Pt-recovery factor (friction).

    Returns
    -------
    dict with:
        Pt3, M3, T3, Ps3, gamma3, A_throat, A_exit
    """
    A_throat = float(diffuser["A_throat"])
    A_exit   = float(diffuser["A_exit"])

    # Pin M_in to the subsonic branch. The cascade's terminal normal-shock
    # clamp guarantees M_exit <= 1, but protect against the M==1 edge case.
    M_in = min(float(M_in), 1.0 - 1e-6)

    # Sonic reference area for this streamtube (same on both sides of a
    # frictionless diverging duct). With friction we bump Pt afterwards.
    gamma_in = gamma_air(float(T_in))
    Astar    = A_throat / area_mach_ratio(M_in, gamma=gamma_in)

    # Subsonic branch: diverging duct -> lower Mach at larger area.
    M3, gamma3 = _iterate_subsonic_mach_tpg(A_exit / Astar, Tt0)
    T3  = Tt0 / (1.0 + 0.5 * (gamma3 - 1.0) * M3 * M3)
    Pt3 = float(Pt_in) * eta_diffuser
    Ps3 = Pt3 * _static_over_total(M3, gamma=gamma3)
    return {
        "Pt3":      Pt3,
        "M3":       M3,
        "T3":       T3,
        "Ps3":      Ps3,
        "gamma3":   gamma3,
        "A_throat": A_throat,
        "A_exit":   A_exit,
    }


def design_2ramp_shock_matched_inlet(M0,altitude_m,alpha_deg,leading_edge_angle_deg,
    mdot_required,width_m,forebody_separation_margin=_LEG_FB_MARGIN,
    ramp_separation_margin=_LEG_RP_MARGIN,kantrowitz_margin=_LEG_KZ_MARGIN,
    shock_focus_factor=_LEG_FOCUS_FACTOR,):

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
        T0=fs["T0"],
    )

    ramp1 = solve_ramp_stage(
        M_up=forebody["M_fore"],
        theta_base=forebody["theta_fore"],
        ramp_separation_margin=ramp_separation_margin,
        stage_name="Ramp 1",
        T_up=forebody["T_down"],
    )

    ramp2 = solve_ramp_stage(
        M_up=ramp1["M_down"],
        theta_base=ramp1["theta_abs"],
        ramp_separation_margin=ramp_separation_margin,
        stage_name="Ramp 2",
        T_up=ramp1["T_down"],
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
        T2=ramp2["T_down"],
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
        T3=cowl["T_down"],
    )

    throat_geom = place_throat_on_cowl_and_cowl_shock(
        C=geom["C"],
        theta_cowl=cowl["theta_cowl"],
        cowl_shock_abs=cowl["cowl_shock_abs"],
        h_throat=throat_k["h_throat"],
        ramp2_end_point=geom["F"],
        leading_edge_angle_deg= -leading_edge_angle_deg
    )

    cascade = solve_reflection_cascade(
        M3=cowl["M3"],
        T3=cowl["T_down"],
        pt_frac_before=pt_frac_after_cowl,
        C=geom["C"],
        F=geom["F"],
        T_upper=throat_geom["T_upper"],
        T_lower=throat_geom["T_lower"],
        theta_cowl_deg=cowl["theta_cowl"],
    )

    diffuser = build_subsonic_diffuser(
        T_upper=throat_geom["T_upper"],
        T_lower=throat_geom["T_lower"],
        h_throat=throat_k["h_throat"],
        width_m=width_m,
        area_ratio_exit_to_throat=DIFFUSER_AREA_RATIO,
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

        "pt_frac_after_reflection_cascade": cascade["pt_frac_after_cascade"],
        "M_after_reflection_cascade":       cascade["M_exit"],
        "T_after_reflection_cascade":       cascade["T_exit"],
        "V_after_reflection_cascade_ms":    cascade["V_exit"],
        "reflection_list":                  cascade["reflections"],
        "reflection_phi_floor_deg":         cascade["phi_floor_deg"],
        "reflection_phi_roof_deg":          cascade["phi_roof_deg"],
        "n_reflections":                    cascade["n_reflections"],

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

    diffuser = result.get("diffuser")
    if diffuser is not None:
        print("\nDIFFUSER")
        print("------------------------------")
        print(f"Diffuser transition             = {diffuser.get('shape_transition', 'n/a')}")
        print(f"Diffuser profile                = {diffuser.get('profile_type', 'n/a')}")
        print(f"Diffuser length                 = {diffuser.get('length_m', float('nan')):.6f} m")
        print(f"Diffuser exit area              = {diffuser.get('A_exit', float('nan')):.6f} m^2")
        print(f"Diffuser exit shape             = {diffuser.get('exit_shape', 'n/a')}")
        print(f"Diffuser exit diameter          = {diffuser.get('exit_diameter_m', float('nan')):.6f} m")
        length_sizing = diffuser.get("length_sizing")
        if isinstance(length_sizing, dict):
            print(f"Diffuser length basis           = {length_sizing.get('governing_mode', 'n/a')}")
            print(f"Length from half-angle          = {length_sizing.get('length_from_angle_m', float('nan')):.6f} m")
            print(f"Length from shock room          = {length_sizing.get('length_from_shock_m', float('nan')):.6f} m")
            print(f"Throat hydraulic diameter       = {length_sizing.get('throat_hydraulic_diameter_m', float('nan')):.6f} m")

    print("\nKANTROWITZ CHECK")
    print("------------------------------")
    print(f"Capture area for check          = {result['capture_area_kantrowitz_m2']:.6f} m^2")
    print(f"Geometric contraction ratio     = {result['geometric_contraction_ratio']:.6f}")
    print(f"Kantrowitz limit                = {result['kantrowitz_limit_CR']:.6f}")
    print(f"Kantrowitz pass                 = {result['kantrowitz_pass']}")

    print("\n==============================\n")

def plot_2ramp_shock_matched_inlet(result,shock_extension_factor=_SHOCK_EXT_FACTOR,
    cowl_extension_factor=_COWL_EXT_FACTOR,):

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

    cowl_len = cowl_extension_factor * max(T_upper[0] - C[0], _COWL_MIN_LEN_M)
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
    kantrowitz_margin=_LEG_KZ_MARGIN,shock_focus_factor=_LEG_FOCUS_FACTOR,verbose=True,):

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

def _upstream_shock_chain(result, M0, altitude_m, alpha_deg):
    """Run forebody → ramp1 → ramp2 → cowl → reflection cascade ONCE for a
    frozen geometry at (M0, alt, α). Shared by compute_inlet_capability and
    evaluate_fixed_geometry_at_condition.
    """
    T0, p0, rho0, a0 = std_atmosphere_1976(altitude_m)
    V0 = M0 * a0

    theta_fore = result["theta_fore_deg"]
    theta1     = result["theta1_deg"]
    theta2     = result["theta2_deg"]
    theta_cowl = result["theta_cowl_deg"]

    # ---- Forebody shock
    dtheta_fore_eff = theta_fore + alpha_deg
    if dtheta_fore_eff <= 0.0:
        return {"ok": False, "reason": "Non-positive forebody effective turn"}
    # Thermally-perfect shock train: gamma(T) propagates stage-by-stage.
    shf = oblique_shock_tpg(M0, dtheta_fore_eff, T0)
    if shf is None:
        return {"ok": False, "reason": "Forebody shock unattached"}
    beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio, _, T_fore, _ = shf

    # ---- Ramp 1
    dtheta1 = theta1 - theta_fore
    if dtheta1 <= 0.0:
        return {"ok": False, "reason": "Invalid ramp 1 turn from frozen geometry"}
    sh1 = oblique_shock_tpg(M_fore, dtheta1, T_fore)
    if sh1 is None:
        return {"ok": False, "reason": "Ramp 1 shock unattached"}
    beta1_rel, M1, p21, pt21, _, T1_s, _ = sh1

    # ---- Ramp 2
    dtheta2 = theta2 - theta1
    if dtheta2 <= 0.0:
        return {"ok": False, "reason": "Invalid ramp 2 turn from frozen geometry"}
    sh2 = oblique_shock_tpg(M1, dtheta2, T1_s)
    if sh2 is None:
        return {"ok": False, "reason": "Ramp 2 shock unattached"}
    beta2_rel, M2, p32, pt32, _, T2_s, _ = sh2

    # ---- Cowl shock
    cowl_turn_mag = theta2 - theta_cowl
    if cowl_turn_mag <= 0.0:
        return {"ok": False, "reason": "Non-positive cowl turn"}
    shc = oblique_shock_tpg(M2, cowl_turn_mag, T2_s)
    if shc is None:
        return {"ok": False, "reason": "Cowl shock unattached"}
    beta_cowl_rel, M3, p43, pt43, _, T3_s, _ = shc

    pt_frac_after_forebody = pt_fore_ratio
    pt_frac_after_shock1   = pt_fore_ratio * pt21
    pt_frac_after_shock2   = pt_fore_ratio * pt21 * pt32
    pt_frac_after_cowl     = pt_fore_ratio * pt21 * pt32 * pt43

    # R2 reflection cascade through the isolator (canonical pt loss model).
    cascade = solve_reflection_cascade(
        M3=M3, T3=T3_s, pt_frac_before=pt_frac_after_cowl,
        C=result["cowl_lip_xy"],
        F=result["ramp2_normal_foot_xy"],
        T_upper=result["throat_upper_xy"],
        T_lower=result["throat_lower_xy"],
        theta_cowl_deg=theta_cowl,
    )

    # Freestream totals use cold-air gamma (T0~220K, gamma_air~1.400); the
    # difference from constant GAMMA=1.4 is <0.1% so we leave it.
    Tt0 = total_temperature(T0, M0)
    Pt0 = p0 * (1.0 + 0.5 * (GAMMA - 1.0) * M0 * M0) ** (GAMMA / (GAMMA - 1.0))
    Pt_after_cowl = Pt0 * pt_frac_after_cowl

    # Reflection cascade is the isolator Pt-loss model. Everything downstream
    # (diffuser terminal shock, pyCycle inlet ram_recovery) references
    # Pt_after_cascade so there is no separate lumped ISOLATOR_PT_RECOVERY.
    pt_frac_after_cascade = float(cascade["pt_frac_after_cascade"])
    Pt_after_cascade      = Pt0 * pt_frac_after_cascade

    return {
        "ok":             True,
        "T0":             T0,        "p0": p0,      "rho0": rho0, "a0": a0,
        "V0":             V0,        "Tt0": Tt0,    "Pt0": Pt0,
        "M_fore":         M_fore,    "T_fore": T_fore,
        "M1":             M1,        "T1_s": T1_s,
        "M2":             M2,        "T2_s": T2_s,
        "M3":             M3,        "T3_s": T3_s,
        "p_fore_ratio":   p_fore_ratio,
        "p21": p21, "p32": p32, "p43": p43,
        "pt_fore_ratio":  pt_fore_ratio,
        "pt21": pt21, "pt32": pt32, "pt43": pt43,
        "pt_frac_after_forebody": pt_frac_after_forebody,
        "pt_frac_after_shock1":   pt_frac_after_shock1,
        "pt_frac_after_shock2":   pt_frac_after_shock2,
        "pt_frac_after_cowl":     pt_frac_after_cowl,
        "pt_frac_after_cascade":  pt_frac_after_cascade,
        "Pt_after_cowl":          Pt_after_cowl,
        "Pt_after_cascade":       Pt_after_cascade,
        # Absolute shock angles for plotting consumers
        "shock_fore_abs_deg": beta_fore_rel,
        "shock1_abs_deg":     theta_fore + beta1_rel,
        "shock2_abs_deg":     theta1 + beta2_rel,
        "cowl_shock_abs_deg": theta2 - beta_cowl_rel,
        "cascade":            cascade,
    }


def compute_inlet_capability(result, M0, altitude_m, alpha_deg):
    """Cascade-only inlet capability at (M0, altitude, alpha).

    Runs forebody → ramps → cowl → reflection-cascade → subsonic diffuser
    ONCE and returns the total pressure delivered at the combustor face. No
    terminal-shock sweep, no Ps_min/Ps_max bracket — the reflection cascade
    is the canonical isolator Pt-loss mechanism, and the subsonic diffuser
    applies a small friction factor ETA_DIFFUSER on top.

    Returns
    -------
    dict with keys:
        ok : bool
        reason : str (if not ok)
        Pt_after_cowl, Pt_after_cascade : Pt at cowl-shock and cascade exit [Pa]
        Pt3_deliverable : Pt at diffuser exit / combustor face [Pa]
        M3_diffuser_exit, T3_diffuser_exit, Ps3_diffuser_exit
        M_cascade_exit, T_cascade_exit
        pt_frac_after_cowl, pt_frac_after_cascade, pt_frac_deliverable
        Tt0, A_throat, A_exit
    """
    up = _upstream_shock_chain(result, M0, altitude_m, alpha_deg)
    if not up["ok"]:
        return {"ok": False, "reason": up["reason"]}

    if "diffuser" not in result:
        return {"ok": False,
                "reason": "Frozen geometry has no diffuser block. Rebuild "
                          "with build_subsonic_diffuser first."}

    diffuser         = result["diffuser"]
    cascade          = up["cascade"]
    Pt_after_cowl    = up["Pt_after_cowl"]
    Pt_after_cascade = up["Pt_after_cascade"]
    Tt0              = up["Tt0"]

    diff = _diffuser_subsonic_exit_state(
        diffuser,
        M_in=cascade["M_exit"],
        T_in=cascade["T_exit"],
        Pt_in=Pt_after_cascade,
        Tt0=Tt0,
    )
    Pt0 = up["Pt0"]
    pt_frac_deliverable = diff["Pt3"] / Pt0 if Pt0 > 0.0 else 0.0

    return {
        "ok":                     True,
        "Pt_after_cowl":          float(Pt_after_cowl),
        "Pt_after_cascade":       float(Pt_after_cascade),
        "Pt3_deliverable":        float(diff["Pt3"]),
        "M3_diffuser_exit":       float(diff["M3"]),
        "T3_diffuser_exit":       float(diff["T3"]),
        "Ps3_diffuser_exit":      float(diff["Ps3"]),
        "gamma3_diffuser_exit":   float(diff["gamma3"]),
        "M_cascade_exit":         float(cascade["M_exit"]),
        "T_cascade_exit":         float(cascade["T_exit"]),
        "Tt0":                    float(Tt0),
        "Pt0":                    float(Pt0),
        "M3_cascade_inlet":       float(up["M3"]),
        "T3_cascade_inlet":       float(up["T3_s"]),
        "pt_frac_after_cowl":     float(up["pt_frac_after_cowl"]),
        "pt_frac_after_cascade":  float(up["pt_frac_after_cascade"]),
        "pt_frac_deliverable":    float(pt_frac_deliverable),
        "A_throat":               float(diff["A_throat"]),
        "A_exit":                 float(diff["A_exit"]),
    }


def evaluate_fixed_geometry_at_condition(result, M0, altitude_m, alpha_deg):
    """
    Re-evaluate the shock system for a fixed geometry at a new flight
    condition. Geometry (including the subsonic diffuser from
    build_subsonic_diffuser) is frozen inside `result`.

    Cascade-only Pt-loss model: forebody + ramps + cowl + reflection cascade
    (+ friction through the subsonic diffuser). No back-pressure-driven
    terminal shock.
    """
    up = _upstream_shock_chain(result, M0, altitude_m, alpha_deg)
    if not up["ok"]:
        return {"success": False, "reason": up["reason"],
                "M0": M0, "alpha_deg": alpha_deg}

    if "diffuser" not in result:
        return {"success": False,
                "reason": "Frozen geometry has no diffuser block. Rebuild "
                          "with build_subsonic_diffuser first.",
                "M0": M0, "alpha_deg": alpha_deg}

    Pt_after_cowl    = up["Pt_after_cowl"]
    Pt_after_cascade = up["Pt_after_cascade"]
    Tt0              = up["Tt0"]
    cascade          = up["cascade"]

    diff = _diffuser_subsonic_exit_state(
        result["diffuser"],
        M_in=cascade["M_exit"],
        T_in=cascade["T_exit"],
        Pt_in=Pt_after_cascade,
        Tt0=Tt0,
    )
    Pt0 = up["Pt0"]
    pt_frac_deliverable = diff["Pt3"] / Pt0 if Pt0 > 0.0 else 0.0

    return {
        "success":        True,
        "status":         "cascade",
        "M0":             M0,
        "alpha_deg":      alpha_deg,
        "V0_ms":          up["V0"],

        "theta_fore_deg":  result["theta_fore_deg"],
        "theta1_deg":      result["theta1_deg"],
        "theta2_deg":      result["theta2_deg"],
        "theta_cowl_deg":  result["theta_cowl_deg"],

        "shock_fore_abs_deg": up["shock_fore_abs_deg"],
        "shock1_abs_deg":     up["shock1_abs_deg"],
        "shock2_abs_deg":     up["shock2_abs_deg"],
        "cowl_shock_abs_deg": up["cowl_shock_abs_deg"],

        "M_after_forebody_shock":  up["M_fore"],
        "M_after_shock1":          up["M1"],
        "M_after_shock2":          up["M2"],
        "M_after_cowl_shock":      up["M3"],

        "M_at_combustor_face":  diff["M3"],
        "Ps_at_combustor_face": diff["Ps3"],
        "T_at_combustor_face":  diff["T3"],

        "pt_frac_after_forebody_shock":   up["pt_frac_after_forebody"],
        "pt_frac_after_shock1":           up["pt_frac_after_shock1"],
        "pt_frac_after_shock2":           up["pt_frac_after_shock2"],
        "pt_frac_after_cowl_shock":       up["pt_frac_after_cowl"],
        "pt_frac_after_cascade":          up["pt_frac_after_cascade"],
        "pt_frac_deliverable":            pt_frac_deliverable,

        "pt_frac_after_reflection_cascade": cascade["pt_frac_after_cascade"],
        "M_after_reflection_cascade":       cascade["M_exit"],
        "T_after_reflection_cascade":       cascade["T_exit"],
        "V_after_reflection_cascade_ms":    cascade["V_exit"],
        "reflection_list":                  cascade["reflections"],
        "reflection_phi_floor_deg":         cascade["phi_floor_deg"],
        "reflection_phi_roof_deg":          cascade["phi_roof_deg"],
        "n_reflections":                    cascade["n_reflections"],

        "forebody_xy":          result["forebody_xy"],
        "nose_xy":              result["nose_xy"],
        "break2_xy":            result["break2_xy"],
        "cowl_lip_xy":          result["cowl_lip_xy"],
        "ramp2_normal_foot_xy": result["ramp2_normal_foot_xy"],
        "throat_lower_xy":      result["throat_lower_xy"],
        "throat_upper_xy":      result["throat_upper_xy"],
        "shock_focus_xy":       result["shock_focus_xy"],

        "diffuser":            result["diffuser"],
        "Pt_after_cowl":       Pt_after_cowl,
        "Pt_after_cascade":    Pt_after_cascade,
        "Pt3_deliverable":     diff["Pt3"],
        "Tt0":                 Tt0,
    }

def plot_fixed_geometry_case(ax,case,shock_extension_factor=_SHOCK_EXT_FACTOR,
    cowl_extension_factor=_COWL_EXT_FACTOR,):
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

    cowl_len = cowl_extension_factor * max(T_upper[0] - C[0], _COWL_MIN_LEN_M)
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
        f"pt/pt0={case['pt_frac_deliverable']:.3f}")

def plot_fixed_geometry_3x3_grid(result,altitude_m,mach_values,alpha_values):
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

def sweep_fixed_geometry_vs_mach(result,altitude_m,mach_values,alpha_deg):
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
        )
        out.append(case)

    return out

def sweep_fixed_geometry_vs_alpha(result,altitude_m,alpha_values,M0):
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
        )
        out.append(case)

    return out

def plot_pt_vs_mach(cases):
    """
    Plot total pressure recovery vs Mach (cascade+diffuser model).
    Also plots MIL-E-5008B:
        pt/pt0 = 1 - 0.075 * (M - 1)^1.35
    """
    xs = []
    ys = []

    for case in cases:
        if not case["success"]:
            continue
        xs.append(case["M0"])
        ys.append(case["pt_frac_deliverable"])

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o", label="Computed")

    if len(xs) > 0:
        xs_ref = np.linspace(min(xs), max(xs), 300)
        ys_ref = 1.0 - 0.075 * (xs_ref - 1.0) ** 1.35
        plt.plot(xs_ref, ys_ref, label="MIL-E-5008B")

    plt.grid(True)
    plt.xlabel("Freestream Mach")
    plt.ylabel("pt/pt0")
    plt.title("Total Pressure Recovery vs Mach (cascade + diffuser)")
    plt.legend()
    plt.show()

def plot_pt_vs_alpha(cases):
    """
    Plot total pressure recovery vs alpha (cascade+diffuser model).
    """
    xs = []
    ys = []

    for case in cases:
        if not case["success"]:
            continue
        xs.append(case["alpha_deg"])
        ys.append(case["pt_frac_deliverable"])

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.grid(True)
    plt.xlabel("Alpha [deg]")
    plt.ylabel("pt/pt0")
    plt.title("Total Pressure Recovery vs Alpha (cascade + diffuser)")
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
        plot_pt_vs_mach(cases_mach)

        # pt vs alpha at fixed Mach
        alpha_sweep = np.linspace(0.0, 10.0, 17)
        cases_alpha = sweep_fixed_geometry_vs_alpha(
            result=result,
            altitude_m=12000.0,
            alpha_values=alpha_sweep,
            M0=5.0,
        )
        plot_pt_vs_alpha(cases_alpha)
