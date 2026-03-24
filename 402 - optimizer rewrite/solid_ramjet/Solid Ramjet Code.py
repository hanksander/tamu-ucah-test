import math
import os
import warnings
from types import SimpleNamespace

from ambiance import Atmosphere
import openmdao.api as om
from openmdao.utils.om_warnings import SolverWarning
from pycycle.elements.combustor import Combustor
from pycycle.elements.flow_start import FlowStart
from pycycle.elements.nozzle import Nozzle
from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf

os.environ.setdefault("OPENMDAO_REPORTS", "0")
warnings.filterwarnings("ignore", category=SolverWarning)

LBM_PER_KG = 2.2046226218487757
PA_PER_PSI = 6894.757293168361
PSI_PER_PA = 1.0 / PA_PER_PSI
M_PER_FT = 0.3048
M2_PER_IN2 = 0.00064516
BTU_PER_LBM_DEGR_TO_J_PER_KG_K = 4186.8

HTPB_CEA_DATA = SimpleNamespace(
    products=janaf.products,
    element_wts=janaf.element_wts,
    reactants={**janaf.reactants, "HTPB": {"C": 10.0, "H": 15.0, "O": 1.0}},
)


class SolidRamjetCycle(Cycle):
    def setup(self):
        self.add_subsystem("flow_start", FlowStart())
        self.add_subsystem("burner", Combustor(fuel_type="HTPB"))
        self.add_subsystem("nozz", Nozzle(nozzType="CD", lossCoef="Cfg", internal_solver=True))
        self.pyc_connect_flow("flow_start.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "nozz.Fl_I")
        super().setup()


def stagnation_temperature(T, M, gamma=1.4):
    return T * (1.0 + 0.5 * (gamma - 1.0) * M**2)


def stagnation_pressure(P, M, gamma=1.4):
    return P * (1.0 + 0.5 * (gamma - 1.0) * M**2) ** (gamma / (gamma - 1.0))


def velocity_from_mach(M, T, gamma=1.4, R=287.0):
    return M * math.sqrt(gamma * R * T)


def normal_shock(M1, gamma=1.4):
    p2_p1 = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1**2 - 1.0)
    rho2_rho1 = ((gamma + 1.0) * M1**2) / ((gamma - 1.0) * M1**2 + 2.0)
    T2_T1 = p2_p1 / rho2_rho1
    M2 = math.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * M1**2)
        / (gamma * M1**2 - 0.5 * (gamma - 1.0))
    )
    term1 = ((gamma + 1.0) * M1**2) / ((gamma - 1.0) * M1**2 + 2.0)
    term2 = (gamma + 1.0) / (2.0 * gamma * M1**2 - (gamma - 1.0))
    p02_p01 = term1 ** (gamma / (gamma - 1.0)) * term2 ** (1.0 / (gamma - 1.0))
    return {"M2": M2, "T2_T1": T2_T1, "p2_p1": p2_p1, "rho2_rho1": rho2_rho1, "p02_p01": p02_p01}


def theta_beta_mach_residual(beta, M1, theta, gamma=1.4):
    sin_beta = math.sin(beta)
    numerator = 2.0 * (M1**2 * sin_beta**2 - 1.0)
    denominator = math.tan(beta) * (M1**2 * (gamma + math.cos(2.0 * beta)) + 2.0)
    return math.tan(theta) - numerator / denominator


def solve_beta(M1, theta_deg, gamma=1.4):
    theta = math.radians(theta_deg)
    beta_lo = math.asin(1.0 / M1) + 1e-8
    beta_hi = 0.5 * math.pi - 1e-8
    samples = 2000
    betas = [beta_lo + i * (beta_hi - beta_lo) / samples for i in range(samples + 1)]
    values = [theta_beta_mach_residual(beta, M1, theta, gamma) for beta in betas]

    for i in range(samples):
        if values[i] * values[i + 1] < 0.0:
            lo = betas[i]
            hi = betas[i + 1]
            break
    else:
        raise ValueError("No attached three-ramp solution found.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = theta_beta_mach_residual(mid, M1, theta, gamma)
        if abs(value) < 1e-10 or abs(hi - lo) < 1e-10:
            return math.degrees(mid)
        if theta_beta_mach_residual(lo, M1, theta, gamma) * value < 0.0:
            hi = mid
        else:
            lo = mid
    return math.degrees(0.5 * (lo + hi))


def oblique_shock(M1, theta_deg, gamma=1.4):
    beta = math.radians(solve_beta(M1, theta_deg, gamma))
    theta = math.radians(theta_deg)
    normal = normal_shock(M1 * math.sin(beta), gamma)
    M2 = normal["M2"] / math.sin(beta - theta)
    return {
        "M1": M1,
        "M2": M2,
        "theta_deg": theta_deg,
        "beta_deg": math.degrees(beta),
        "p02_p01": normal["p02_p01"],
        "p2_p1": normal["p2_p1"],
        "rho2_rho1": normal["rho2_rho1"],
        "T2_T1": normal["T2_T1"],
    }


def analyze_three_ramp_inlet(M_inf, T_inf, P_inf, rho_inf, angles, gamma=1.4):
    if len(angles) != 3:
        raise ValueError("Three ramp angles are required.")

    ramps = []
    M = M_inf
    T = T_inf
    P = P_inf
    rho = rho_inf
    P0 = stagnation_pressure(P_inf, M_inf, gamma)

    for angle in angles:
        shock = oblique_shock(M, angle, gamma)
        M = shock["M2"]
        T *= shock["T2_T1"]
        P *= shock["p2_p1"]
        rho *= shock["rho2_rho1"]
        P0 *= shock["p02_p01"]
        ramps.append(shock)

    if M <= 1.0:
        raise ValueError("Three-ramp inlet must remain supersonic ahead of the isolator shock.")

    normal = normal_shock(M, gamma)
    return {
        "angles": [float(angle) for angle in angles],
        "ramps": ramps,
        "M_exit": M,
        "T_exit": T,
        "P_exit": P,
        "rho_exit": rho,
        "P0_exit": P0,
        "recovery_after_isolator_shock": (P0 / stagnation_pressure(P_inf, M_inf, gamma)) * normal["p02_p01"],
    }


def optimize_three_ramp_inlet(
    M_inf,
    T_inf,
    P_inf,
    rho_inf,
    gamma=1.4,
    angle_min_deg=4.0,
    angle_max_deg=14.0,
    total_turn_min_deg=18.0,
    total_turn_max_deg=45.0,
):
    best = None
    best_recovery = -1.0

    for theta1 in range(int(angle_min_deg), int(angle_max_deg) + 1):
        for theta2 in range(int(angle_min_deg), int(angle_max_deg) + 1):
            for theta3 in range(int(angle_min_deg), int(angle_max_deg) + 1):
                angles = [float(theta1), float(theta2), float(theta3)]
                total_turn = sum(angles)
                if not (total_turn_min_deg <= total_turn <= total_turn_max_deg):
                    continue
                try:
                    inlet = analyze_three_ramp_inlet(M_inf, T_inf, P_inf, rho_inf, angles, gamma)
                except ValueError:
                    continue

                if inlet["recovery_after_isolator_shock"] > best_recovery:
                    best_recovery = inlet["recovery_after_isolator_shock"]
                    best = inlet

    if best is None:
        raise ValueError("No valid three-ramp inlet found.")
    return best


def apply_isolator_entrance_shock(inlet, gamma=1.4, pt_recovery=1.0):
    shock = normal_shock(inlet["M_exit"], gamma)
    M_exit = shock["M2"]
    T_exit = inlet["T_exit"] * shock["T2_T1"]
    P_exit = inlet["P_exit"] * shock["p2_p1"]
    rho_exit = inlet["rho_exit"] * shock["rho2_rho1"]
    P0_exit = inlet["P0_exit"] * shock["p02_p01"] * pt_recovery
    return {
        "M1": inlet["M_exit"],
        "M_exit": M_exit,
        "T_exit": T_exit,
        "P_exit": P_exit,
        "rho_exit": rho_exit,
        "P0_exit": P0_exit,
        "T0_exit": stagnation_temperature(T_exit, M_exit, gamma),
        "shock": shock,
        "pt_recovery": pt_recovery,
    }


def estimate_regression_rate(
    mdot_air,
    inlet_state,
    D_port,
    L,
    a,
    n,
    rho_fuel,
    pressure_loss_frac,
    eta_comb,
    pressure_exp,
):
    area = math.pi * D_port**2 / 4.0
    perimeter = math.pi * D_port
    dx = L / 80.0
    mdot_core = mdot_air
    static_pressure = max(inlet_state["P_exit"] * (1.0 - 0.5 * pressure_loss_frac), 1.0)
    pressure_factor = (static_pressure / 101325.0) ** pressure_exp
    mdot_fuel = 0.0
    g_sum = 0.0
    rdot_sum = 0.0

    for _ in range(80):
        flux = mdot_core / area
        erosive_factor = 1.0 + 0.15 * max(flux / 500.0 - 1.0, 0.0)
        rdot = a * flux**n * pressure_factor * erosive_factor
        dmdot_fuel = rho_fuel * perimeter * dx * rdot
        mdot_core += dmdot_fuel
        mdot_fuel += dmdot_fuel
        g_sum += flux
        rdot_sum += rdot

    velocity_guess = (mdot_air + 0.5 * mdot_fuel) / (max(inlet_state["rho_exit"], 1e-6) * area)
    residence_efficiency = 1.0 - math.exp(-L / max(velocity_guess, 1e-9) / 0.003)
    mdot_fuel_effective = mdot_fuel * eta_comb * residence_efficiency

    return {
        "A_port": area,
        "G": g_sum / 80.0,
        "rdot": rdot_sum / 80.0,
        "mdot_fuel": mdot_fuel,
        "mdot_fuel_effective": mdot_fuel_effective,
        "pressure_factor": pressure_factor,
        "combustion_efficiency": eta_comb * residence_efficiency,
    }


def htpb_stoichiometric_far():
    carbon = 10.0
    hydrogen = 15.0
    oxygen = 1.0
    fuel_molecular_weight = 12.0 * carbon + hydrogen + 16.0 * oxygen
    required_o2_moles = carbon + hydrogen / 4.0 - oxygen / 2.0
    return fuel_molecular_weight * 0.232 / (32.0 * required_o2_moles)


def solve_regression_a_for_design_far(
    target_far,
    mdot_air,
    inlet_state,
    D_port,
    L,
    n,
    rho_fuel,
    pressure_loss_frac,
    eta_comb,
    pressure_exp,
):
    lo = 1e-8
    hi = 1e-5

    regression = estimate_regression_rate(
        mdot_air, inlet_state, D_port, L, hi, n, rho_fuel, pressure_loss_frac, eta_comb, pressure_exp
    )
    while regression["mdot_fuel_effective"] / mdot_air < target_far:
        hi *= 2.0
        regression = estimate_regression_rate(
            mdot_air, inlet_state, D_port, L, hi, n, rho_fuel, pressure_loss_frac, eta_comb, pressure_exp
        )
        if hi > 1.0:
            raise ValueError("Could not bracket the design-point fuel-air ratio.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        regression = estimate_regression_rate(
            mdot_air, inlet_state, D_port, L, mid, n, rho_fuel, pressure_loss_frac, eta_comb, pressure_exp
        )
        far = regression["mdot_fuel_effective"] / mdot_air
        if abs(far - target_far) / target_far < 1e-6:
            return mid, regression
        if far < target_far:
            lo = mid
        else:
            hi = mid

    final_a = 0.5 * (lo + hi)
    final_regression = estimate_regression_rate(
        mdot_air, inlet_state, D_port, L, final_a, n, rho_fuel, pressure_loss_frac, eta_comb, pressure_exp
    )
    return final_a, final_regression


def run_combustor_and_nozzle(
    isolator_state,
    mdot_air,
    mdot_fuel,
    pressure_loss_frac,
    burner_mach,
    ambient_pressure,
    nozzle_pressure_loss_frac,
    nozzle_cfg,
):
    problem = om.Problem()
    problem.model = SolidRamjetCycle(thermo_method="CEA", thermo_data=HTPB_CEA_DATA)
    problem.setup()
    problem.set_solver_print(level=-1)

    problem.set_val("flow_start.P", isolator_state["P0_exit"] * PSI_PER_PA, units="psi")
    problem.set_val("flow_start.T", isolator_state["T0_exit"] * 9.0 / 5.0, units="degR")
    problem.set_val("flow_start.W", mdot_air * LBM_PER_KG, units="lbm/s")
    problem.set_val("flow_start.MN", isolator_state["M_exit"])
    problem.set_val("burner.Fl_I:FAR", mdot_fuel / mdot_air)
    problem.set_val("burner.dPqP", pressure_loss_frac)
    problem.set_val("burner.MN", burner_mach)
    problem.set_val("nozz.Ps_exhaust", ambient_pressure * PSI_PER_PA, units="psi")
    problem.set_val("nozz.dPqP", nozzle_pressure_loss_frac)
    problem.set_val("nozz.Cfg", nozzle_cfg)
    problem.run_model()

    cp = problem.get_val("burner.Fl_O:tot:Cp", units="Btu/(lbm*degR)").item()
    cv = problem.get_val("burner.Fl_O:tot:Cv", units="Btu/(lbm*degR)").item()
    gross_thrust_actual = problem.get_val("nozz.Fg", units="lbf").item() * 4.4482216152605
    gross_thrust = problem.get_val("nozz.perf_calcs.Fg_ideal", units="lbf").item() * 4.4482216152605
    throat_area = problem.get_val("nozz.Throat:stat:area", units="inch**2").item() * M2_PER_IN2
    combustor_exit_total_pressure = problem.get_val("burner.Fl_O:tot:P", units="psi").item() * PA_PER_PSI

    return {
        "T0_out": problem.get_val("burner.Fl_O:tot:T", units="degR").item() * 5.0 / 9.0,
        "P0_out": combustor_exit_total_pressure,
        "gamma_out": problem.get_val("burner.Fl_O:tot:gamma").item(),
        "cp_out": cp * BTU_PER_LBM_DEGR_TO_J_PER_KG_K,
        "R_out": (cp - cv) * BTU_PER_LBM_DEGR_TO_J_PER_KG_K,
        "is_choked": bool(problem.get_val("nozz.mux.choked").item()),
        "A_throat_required": throat_area,
        "A_exit": problem.get_val("nozz.Fl_O:stat:area", units="inch**2").item() * M2_PER_IN2,
        "V_exit": problem.get_val("nozz.Fl_O:stat:V", units="ft/s").item() * M_PER_FT,
        "M_exit": problem.get_val("nozz.Fl_O:stat:MN").item(),
        "P_exit": problem.get_val("nozz.Fl_O:stat:P", units="psi").item() * PA_PER_PSI,
        "gross_thrust": gross_thrust_actual,
        "gross_thrust_ideal": gross_thrust,
        "gross_thrust_coefficient": gross_thrust_actual / (combustor_exit_total_pressure * throat_area),
        "gross_thrust_coefficient_ideal": gross_thrust / (combustor_exit_total_pressure * throat_area),
    }


def compute_external_drags(
    rho_inf,
    V_inf,
    inlet_lip_diameter,
    body_diameter,
    body_length,
    capture_efficiency,
    cowl_drag_coefficient,
    body_friction_coefficient,
    spillage_drag_factor,
):
    lip_area = math.pi * inlet_lip_diameter**2 / 4.0
    body_frontal_area = math.pi * body_diameter**2 / 4.0
    body_wetted_area = math.pi * body_diameter * body_length
    mdot_streamtube = rho_inf * V_inf * lip_area
    mdot_air = mdot_streamtube * capture_efficiency
    q_inf = 0.5 * rho_inf * V_inf**2
    cowl_drag = cowl_drag_coefficient * q_inf * body_frontal_area
    skin_friction_drag = body_friction_coefficient * q_inf * body_wetted_area
    spillage_drag = spillage_drag_factor * max(mdot_streamtube - mdot_air, 0.0) * V_inf
    ram_drag = mdot_air * V_inf
    return {
        "lip_area": lip_area,
        "body_frontal_area": body_frontal_area,
        "body_wetted_area": body_wetted_area,
        "mdot_streamtube": mdot_streamtube,
        "mdot_air": mdot_air,
        "q_inf": q_inf,
        "ram_drag": ram_drag,
        "cowl_drag": cowl_drag,
        "skin_friction_drag": skin_friction_drag,
        "spillage_drag": spillage_drag,
        "external_drag": cowl_drag + skin_friction_drag + spillage_drag,
    }


def apply_capture_efficiency_aoa_correction(
    nominal_capture_efficiency,
    alpha_deg,
    penalty_per_deg=0.01,
    min_fraction_of_nominal=0.80,
):
    """Apply a conservative first-order AoA penalty to inlet capture.

    This is an integration-level approximation only. It keeps the nominal
    capture at alpha=0 and reduces capture linearly with |alpha| at 1%
    of the nominal value per degree, clamped to avoid nonphysical values.
    """
    penalty_factor = max(1.0 - penalty_per_deg * abs(alpha_deg), min_fraction_of_nominal)
    return nominal_capture_efficiency * penalty_factor


def apply_pressure_recovery_aoa_correction(
    nominal_pressure_recovery,
    alpha_deg,
    penalty_per_deg=0.015,
    min_fraction_of_nominal=0.80,
):
    """Apply a conservative first-order AoA penalty to total-pressure recovery.

    This is an integration-level approximation only. It keeps the nominal
    recovery at alpha=0 and reduces recovery linearly with |alpha| at 1.5%
    of the nominal value per degree, clamped to avoid nonphysical values.
    """
    penalty_factor = max(1.0 - penalty_per_deg * abs(alpha_deg), min_fraction_of_nominal)
    return nominal_pressure_recovery * penalty_factor


def run_solid_ramjet(M_inf, h, alpha_deg=0.0):
    """Run the solid ramjet model for a single Mach/altitude point.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    h : float
        Altitude in meters for the ambiance atmosphere model.
    alpha_deg : float, optional
        Vehicle angle of attack in degrees. Defaults to 0.
    """
    gamma = 1.4
    inlet_lip_diameter = 0.25
    nominal_capture_efficiency = 0.95
    capture_efficiency = apply_capture_efficiency_aoa_correction(nominal_capture_efficiency, alpha_deg)
    fixed_ramp_angles_deg = None
    ramp_angle_min_deg = 4.0
    ramp_angle_max_deg = 14.0
    total_turn_min_deg = 18.0
    total_turn_max_deg = 45.0
    body_diameter = 0.34
    body_length = 3.58
    missile_mass = 1088.0
    D_port = 0.10
    L = 1.20
    n = 0.5
    rho_fuel = 930.0
    energy_density = 44e6
    eta_comb = 0.95
    pressure_exp = 0.20
    pressure_loss_frac = 0.06
    nominal_isolator_pt_recovery = 0.98
    isolator_pt_recovery = apply_pressure_recovery_aoa_correction(nominal_isolator_pt_recovery, alpha_deg)
    burner_mach = 0.20
    nozzle_pressure_loss_frac = 0.02
    nozzle_cfg = 0.985
    cowl_drag_coefficient = 0.06
    body_friction_coefficient = 0.003
    spillage_drag_factor = 1.0
    # Fixed regression coefficient calibrated from the original baseline case.
    regression_a = 2.589155137e-05

    atmosphere = Atmosphere(h)
    T_inf = atmosphere.temperature[0]
    P_inf = atmosphere.pressure[0]
    rho_inf = atmosphere.density[0]
    V_inf = velocity_from_mach(M_inf, T_inf, gamma)
    drags = compute_external_drags(
        rho_inf,
        V_inf,
        inlet_lip_diameter,
        body_diameter,
        body_length,
        capture_efficiency,
        cowl_drag_coefficient,
        body_friction_coefficient,
        spillage_drag_factor,
    )
    mdot_air = drags["mdot_air"]

    if fixed_ramp_angles_deg is None:
        inlet = optimize_three_ramp_inlet(
            M_inf,
            T_inf,
            P_inf,
            rho_inf,
            gamma,
            angle_min_deg=ramp_angle_min_deg,
            angle_max_deg=ramp_angle_max_deg,
            total_turn_min_deg=total_turn_min_deg,
            total_turn_max_deg=total_turn_max_deg,
        )
        inlet_mode = "optimized"
    else:
        inlet = analyze_three_ramp_inlet(M_inf, T_inf, P_inf, rho_inf, fixed_ramp_angles_deg, gamma)
        inlet_mode = "fixed"
    isolator = apply_isolator_entrance_shock(inlet, gamma, pt_recovery=isolator_pt_recovery)
    stoich_far = htpb_stoichiometric_far()
    regression = estimate_regression_rate(
        mdot_air, isolator, D_port, L, regression_a, n, rho_fuel, pressure_loss_frac, eta_comb, pressure_exp
    )
    nozzle = run_combustor_and_nozzle(
        isolator,
        mdot_air,
        regression["mdot_fuel_effective"],
        pressure_loss_frac,
        burner_mach,
        P_inf,
        nozzle_pressure_loss_frac,
        nozzle_cfg,
    )
    total_drag = drags["ram_drag"] + drags["external_drag"]
    net_thrust = nozzle["gross_thrust"] - total_drag
    isp = net_thrust / (regression["mdot_fuel_effective"] * 9.80665)
    specific_thrust = net_thrust / mdot_air
    tsfc_mg_per_ns = (
        1.0e6 * regression["mdot_fuel_effective"] / net_thrust if net_thrust > 0.0 else float("inf")
    )
    thrust_to_weight = net_thrust / (missile_mass * 9.80665)
    axial_acceleration = net_thrust / missile_mass
    fuel_air_ratio_effective = regression["mdot_fuel_effective"] / mdot_air
    equivalence_ratio_effective = fuel_air_ratio_effective / stoich_far

    return {
        "M_inf": M_inf,
        "h": h,
        "ambient": {
            "T_inf": T_inf,
            "P_inf": P_inf,
            "rho_inf": rho_inf,
            "V_inf": V_inf,
        },
        "settings": {
            "gamma": gamma,
            "inlet_lip_diameter": inlet_lip_diameter,
            "alpha_deg": alpha_deg,
            "nominal_capture_efficiency": nominal_capture_efficiency,
            "capture_efficiency": capture_efficiency,
            "fixed_ramp_angles_deg": fixed_ramp_angles_deg,
            "ramp_angle_min_deg": ramp_angle_min_deg,
            "ramp_angle_max_deg": ramp_angle_max_deg,
            "total_turn_min_deg": total_turn_min_deg,
            "total_turn_max_deg": total_turn_max_deg,
            "body_diameter": body_diameter,
            "body_length": body_length,
            "missile_mass": missile_mass,
            "D_port": D_port,
            "L": L,
            "n": n,
            "rho_fuel": rho_fuel,
            "energy_density": energy_density,
            "eta_comb": eta_comb,
            "pressure_exp": pressure_exp,
            "pressure_loss_frac": pressure_loss_frac,
            "nominal_isolator_pt_recovery": nominal_isolator_pt_recovery,
            "isolator_pt_recovery": isolator_pt_recovery,
            "burner_mach": burner_mach,
            "nozzle_pressure_loss_frac": nozzle_pressure_loss_frac,
            "nozzle_cfg": nozzle_cfg,
            "cowl_drag_coefficient": cowl_drag_coefficient,
            "body_friction_coefficient": body_friction_coefficient,
            "spillage_drag_factor": spillage_drag_factor,
            "regression_a": regression_a,
        },
        "mdot_air": mdot_air,
        "inlet_area": drags["lip_area"],
        "captured_inlet_area": drags["lip_area"] * capture_efficiency,
        "areas": {
            "inlet_lip_area": drags["lip_area"],
            "captured_inlet_area": drags["lip_area"] * capture_efficiency,
            "body_frontal_area": drags["body_frontal_area"],
            "body_wetted_area": drags["body_wetted_area"],
            "port_area": regression["A_port"],
            "nozzle_throat_area": nozzle["A_throat_required"],
            "nozzle_exit_area": nozzle["A_exit"],
            "nozzle_area_ratio": nozzle["A_exit"] / nozzle["A_throat_required"],
        },
        "drags": drags,
        "inlet_mode": inlet_mode,
        "inlet": inlet,
        "isolator": isolator,
        "stoich_far": stoich_far,
        "fuel_air_ratio_effective": fuel_air_ratio_effective,
        "equivalence_ratio_effective": equivalence_ratio_effective,
        "a": regression_a,
        "regression": regression,
        "nozzle": nozzle,
        "total_drag": total_drag,
        "net_thrust": net_thrust,
        "isp": isp,
        "specific_thrust": specific_thrust,
        "tsfc_mg_per_ns": tsfc_mg_per_ns,
        "thrust_to_weight": thrust_to_weight,
        "axial_acceleration": axial_acceleration,
    }


def print_solid_ramjet_report(results):
    areas = results["areas"]
    settings = results["settings"]
    drags = results["drags"]
    inlet = results["inlet"]
    isolator = results["isolator"]
    regression = results["regression"]
    nozzle = results["nozzle"]
    capture_efficiency_alpha_0 = apply_capture_efficiency_aoa_correction(
        settings["nominal_capture_efficiency"], 0.0
    )
    capture_efficiency_alpha_2 = apply_capture_efficiency_aoa_correction(
        settings["nominal_capture_efficiency"], 2.0
    )
    pressure_recovery_alpha_0 = apply_pressure_recovery_aoa_correction(
        settings["nominal_isolator_pt_recovery"], 0.0
    )
    pressure_recovery_alpha_2 = apply_pressure_recovery_aoa_correction(
        settings["nominal_isolator_pt_recovery"], 2.0
    )

    print(f"air mass flow              = {results['mdot_air']:.6f} kg/s")
    print(f"streamtube mass flow       = {drags['mdot_streamtube']:.6f} kg/s")
    print(f"missile mass               = {settings['missile_mass']:.6f} kg")
    print(f"body diameter              = {settings['body_diameter']:.6f} m")
    print(f"body length                = {settings['body_length']:.6f} m")
    print(f"assumed inlet lip diameter = {settings['inlet_lip_diameter']:.6f} m")
    print("fueling mode               = fixed regression coefficient")
    print(f"regression a               = {settings['regression_a']:.9e} m/s*(kg/m^2/s)^(-n)")
    print(f"resulting equivalence ratio= {results['equivalence_ratio_effective']:.6f}")
    print(f"resulting fuel-air ratio   = {results['fuel_air_ratio_effective']:.6f}")
    print(f"dynamic pressure           = {drags['q_inf']:.6f} Pa")

    print("\n=== AOA AND CAPTURE ===")
    print(f"angle of attack            = {settings['alpha_deg']:.6f} deg")
    print(f"nominal capture efficiency = {settings['nominal_capture_efficiency']:.6f}")
    print(f"capture efficiency         = {settings['capture_efficiency']:.6f}")
    print(f"capture efficiency @ 0 deg = {capture_efficiency_alpha_0:.6f}")
    print(f"capture efficiency @ 2 deg = {capture_efficiency_alpha_2:.6f}")
    print(f"nominal Pt recovery factor = {settings['nominal_isolator_pt_recovery']:.6f}")
    print(f"Pt recovery factor         = {settings['isolator_pt_recovery']:.6f}")
    print(f"Pt recovery factor @ 0 deg = {pressure_recovery_alpha_0:.6f}")
    print(f"Pt recovery factor @ 2 deg = {pressure_recovery_alpha_2:.6f}")

    print("\n=== AREAS ===")
    print(f"inlet lip area             = {areas['inlet_lip_area']:.6f} m^2")
    print(f"captured inlet area        = {areas['captured_inlet_area']:.6f} m^2")
    print(f"body frontal area          = {areas['body_frontal_area']:.6f} m^2")
    print(f"body wetted area           = {areas['body_wetted_area']:.6f} m^2")
    print(f"combustor port area        = {areas['port_area']:.6f} m^2")
    print(f"nozzle throat area         = {areas['nozzle_throat_area']:.6f} m^2")
    print(f"nozzle exit area           = {areas['nozzle_exit_area']:.6f} m^2")
    print(f"nozzle area ratio          = {areas['nozzle_area_ratio']:.6f}")

    print("\n=== THREE-RAMP INLET ===")
    print(f"inlet mode                 = {results['inlet_mode']}")
    print(f"angles (deg)               = {inlet['angles']}")
    for i, ramp in enumerate(inlet["ramps"], start=1):
        print(
            f"ramp {i}: theta={ramp['theta_deg']:.3f}, beta={ramp['beta_deg']:.3f}, "
            f"M1={ramp['M1']:.4f}, M2={ramp['M2']:.4f}, p02/p01={ramp['p02_p01']:.6f}"
        )
    print(f"inlet exit Mach            = {inlet['M_exit']:.6f}")

    print("\n=== ISOLATOR ENTRANCE NORMAL SHOCK ===")
    print(f"M1                         = {isolator['M1']:.6f}")
    print(f"M2                         = {isolator['M_exit']:.6f}")
    print(f"p02/p01                    = {isolator['shock']['p02_p01']:.6f}")
    print(f"isolator Pt recovery       = {isolator['pt_recovery']:.6f}")

    print("\n=== SOLID FUEL RAMJET COMBUSTION ===")
    print(f"mdot_fuel_raw              = {regression['mdot_fuel']:.6f} kg/s")
    print(f"mdot_fuel_effective        = {regression['mdot_fuel_effective']:.6f} kg/s")
    print(f"fuel_air_ratio_effective   = {results['fuel_air_ratio_effective']:.6f}")
    print(f"G                          = {regression['G']:.6f} kg/m^2-s")
    print(f"rdot                       = {regression['rdot']:.8f} m/s")
    print(f"pressure_factor            = {regression['pressure_factor']:.6f}")
    print(f"combustion_efficiency      = {regression['combustion_efficiency']:.6f}")
    print(f"T0_out                     = {nozzle['T0_out']:.6f} K")
    print(f"P0_out                     = {nozzle['P0_out']:.6f} Pa")
    print(f"gamma_out                  = {nozzle['gamma_out']:.6f}")
    print(f"heat_release_rate          = {regression['mdot_fuel_effective'] * settings['energy_density']:.6f} W")

    print("\n=== NOZZLE AND PERFORMANCE ===")
    print(f"is_choked                  = {nozzle['is_choked']}")
    print(f"A_throat_required          = {nozzle['A_throat_required']:.6f} m^2")
    print(f"V_exit                     = {nozzle['V_exit']:.6f} m/s")
    print(f"M_exit                     = {nozzle['M_exit']:.6f}")
    print(f"P_exit                     = {nozzle['P_exit']:.6f} Pa")
    print(f"gross thrust ideal         = {nozzle['gross_thrust_ideal']:.6f} N")
    print(f"gross thrust               = {nozzle['gross_thrust']:.6f} N")
    print(f"gross thrust coeff ideal   = {nozzle['gross_thrust_coefficient_ideal']:.6f}")
    print(f"gross thrust coeff         = {nozzle['gross_thrust_coefficient']:.6f}")
    print(f"ram drag                   = {drags['ram_drag']:.6f} N")
    print(f"cowl drag                  = {drags['cowl_drag']:.6f} N")
    print(f"skin friction drag         = {drags['skin_friction_drag']:.6f} N")
    print(f"spillage drag              = {drags['spillage_drag']:.6f} N")
    print(f"net thrust                 = {results['net_thrust']:.6f} N")
    print(f"thrust-to-weight           = {results['thrust_to_weight']:.6f}")
    print(f"axial acceleration         = {results['axial_acceleration']:.6f} m/s^2")
    print(f"specific thrust            = {results['specific_thrust']:.6f} N/(kg/s)")
    print(f"TSFC                       = {results['tsfc_mg_per_ns']:.6f} mg/(N*s)")
    print(f"Isp                        = {results['isp']:.6f} s")


def main():
    results = run_solid_ramjet(M_inf=5.0, h=15000.0)
    print_solid_ramjet_report(results)


if __name__ == "__main__":
    main()
