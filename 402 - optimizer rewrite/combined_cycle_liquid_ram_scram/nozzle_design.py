"""
Nozzle design and performance analysis using pyCycle.

The active nozzle analysis path is:
    OpenMDAO Problem -> pyCycle Cycle -> FlowStart -> Nozzle

The remaining standalone equations are only used to estimate flight-mode inputs,
rectangular vehicle packaging, and the bell contour geometry around the pyCycle
station solution. No ASME/isentropic nozzle solver is used for the reported
nozzle station properties or performance metrics.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import openmdao.api as om
from pycycle.elements.flow_start import FlowStart
from pycycle.elements.nozzle import Nozzle
from pycycle.element_base import Element
from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf
from pyc_config import INLET_DESIGN_WIDTH_M


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_PLOT_PATH = OUTPUT_DIR / "nozzle_analysis.png"
DEFAULT_GEOMETRY_PATH = OUTPUT_DIR / "nozzle_geometry.csv"
FT_PER_M = 3.280839895013123

# Assumed liquid-ramjet baseline used by the no-argument run.
# These constants intentionally hardcode the current reference case for now.
# They are assigned as CLI defaults, then passed explicitly into the solver and
# geometry functions so the functions do not depend on hidden global state.
# missile_length_m = 4.0
# nozzle_length_fraction = 0.20
# cruise_altitude_m = 12000.0
# cruise_mach = 5.0
# vehicle_width_m = 0.40
# vehicle_height_m = 0.57
# inlet_width_m = 0.25
# inlet_height_m = 0.05
# nozzle_diameter_clearance = 1.0
# fuel = "JP-7"
# fuel_lhv_j_per_kg = 43.5e6
# combustor_exit_total_temperature_k = 2500.0
# inlet_total_pressure_recovery = 0.45
# combustor_total_pressure_ratio = 0.95
DEFAULT_MISSILE_LENGTH = 4.0
DEFAULT_NOZZLE_LENGTH_FRACTION = 0.10
DEFAULT_CRUISE_ALTITUDE_M = 18000.0
DEFAULT_CRUISE_MACH = 5.0
DEFAULT_VEHICLE_WIDTH = 0.40
DEFAULT_VEHICLE_HEIGHT = 0.57
DEFAULT_INLET_WIDTH = 0.25
DEFAULT_INLET_HEIGHT = 0.05
DEFAULT_NOZZLE_DIAMETER_CLEARANCE = 1.0
DEFAULT_FUEL_NAME = "JP-7"
DEFAULT_FUEL_LHV = 43.5e6
DEFAULT_COMBUSTOR_EXIT_TT = 2500.0
DEFAULT_INLET_PRESSURE_RECOVERY = 0.45
DEFAULT_COMBUSTOR_PRESSURE_RATIO = 0.95
DEFAULT_CONV_HALF_ANGLE = 25.0   # degrees
DEFAULT_DIV_HALF_ANGLE  = 12.0
G0 = 9.80665
LBM_PER_KG = 2.2046226218
LBF_PER_N = 0.2248089431


def scalar(value):
    """Return a scalar float from OpenMDAO's array-shaped values."""

    return float(np.asarray(value).ravel()[0])


class FlowStationSource(Element):
    """
    Element that emits a fully specified flowstation.

    This lets the standalone nozzle consume the exact solved burner-exit
    flowstate, including reacted composition, through pyCycle's normal
    flow-connection graph.
    """

    def initialize(self):
        self.options.declare("flowstation", recordable=False)
        self.options.declare("flow_port_data", recordable=False)
        super().initialize()

    def pyc_setup_output_ports(self):
        flow_port_data = self.options["flow_port_data"]
        if flow_port_data is None:
            raise ValueError("FlowStationSource requires flow_port_data for pyCycle flow metadata.")
        self.init_output_flow("Fl_O", flow_port_data)

    def setup(self):
        flowstation = self.options["flowstation"]
        src = om.IndepVarComp()
        units_map = {
            "Fl_O:tot:h": "Btu/lbm",
            "Fl_O:tot:T": "degR",
            "Fl_O:tot:P": "lbf/inch**2",
            "Fl_O:tot:S": "Btu/(lbm*degR)",
            "Fl_O:tot:rho": "lbm/ft**3",
            "Fl_O:tot:gamma": None,
            "Fl_O:tot:Cp": "Btu/(lbm*degR)",
            "Fl_O:tot:Cv": "Btu/(lbm*degR)",
            "Fl_O:tot:R": "Btu/(lbm*degR)",
            "Fl_O:tot:composition": None,
            "Fl_O:stat:h": "Btu/lbm",
            "Fl_O:stat:T": "degR",
            "Fl_O:stat:P": "lbf/inch**2",
            "Fl_O:stat:S": "Btu/(lbm*degR)",
            "Fl_O:stat:rho": "lbm/ft**3",
            "Fl_O:stat:gamma": None,
            "Fl_O:stat:Cp": "Btu/(lbm*degR)",
            "Fl_O:stat:Cv": "Btu/(lbm*degR)",
            "Fl_O:stat:R": "Btu/(lbm*degR)",
            "Fl_O:stat:V": "ft/s",
            "Fl_O:stat:Vsonic": "ft/s",
            "Fl_O:stat:MN": None,
            "Fl_O:stat:area": "inch**2",
            "Fl_O:stat:W": "lbm/s",
        }
        for name, value in flowstation.items():
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size == 1:
                units = units_map.get(name)
                if units is None:
                    src.add_output(name, val=float(arr.ravel()[0]))
                else:
                    src.add_output(name, val=float(arr.ravel()[0]), units=units)
            else:
                src.add_output(name, val=arr)
        self.add_subsystem("src", src, promotes=["*"])
        super().setup()


class CompositionFlowStart(Element):
    """
    Flow-start element for a fully specified composition vector.

    pyCycle's stock FlowStart expects an elemental/reactant composition
    description. For the standalone nozzle driven from burner exit, we already
    have the reacted mixture vector and need to propagate it directly.
    """

    def initialize(self):
        self.options.declare("composition", default=None, recordable=False)
        super().initialize()

    def pyc_setup_output_ports(self):
        composition = self.options["composition"]
        if composition is None:
            raise ValueError("CompositionFlowStart requires a composition vector.")
        self.init_output_flow("Fl_O", composition)

    def setup(self):
        thermo_method = self.options["thermo_method"]
        thermo_data = self.options["thermo_data"]
        composition = self.options["composition"]

        totals = Thermo(
            mode="total_TP",
            fl_name="Fl_O:tot",
            method=thermo_method,
            thermo_kwargs={"composition": composition, "spec": thermo_data},
        )
        self.add_subsystem(
            "totals",
            totals,
            promotes_inputs=("T", "P"),
            promotes_outputs=("Fl_O:tot:*",),
        )

        exit_static = Thermo(
            mode="static_MN",
            fl_name="Fl_O:stat",
            method=thermo_method,
            thermo_kwargs={"composition": composition, "spec": thermo_data},
        )
        self.add_subsystem(
            "exit_static",
            exit_static,
            promotes_inputs=("MN", "W"),
            promotes_outputs=("Fl_O:stat:*",),
        )

        self.connect("totals.h", "exit_static.ht")
        self.connect("totals.S", "exit_static.S")
        self.connect("Fl_O:tot:P", "exit_static.guess:Pt")
        self.connect("totals.gamma", "exit_static.guess:gamt")

        super().setup()


def rectangular_vehicle_packaging(width, height, nozzle_diameter_clearance, max_exit_area):
    """
    Compute packaging limits for an axisymmetric nozzle inside a rectangular body.

    A revolved bell nozzle has a circular exit, so the limiting diameter is the
    smaller body dimension multiplied by the selected clearance factor.
    """

    if width <= 0.0 or height <= 0.0:
        raise ValueError("Vehicle width and height must be positive.")
    if nozzle_diameter_clearance <= 0.0 or nozzle_diameter_clearance > 1.0:
        raise ValueError("nozzle_diameter_clearance must be in the range (0, 1].")
    if max_exit_area is not None and max_exit_area <= 0.0:
        raise ValueError("max_exit_area must be positive.")

    frontal_area = width * height
    max_diameter = min(width, height) * nozzle_diameter_clearance
    max_radius = 0.5 * max_diameter
    rectangular_fit_area = np.pi * max_radius**2
    selected_exit_area = rectangular_fit_area if max_exit_area is None else min(max_exit_area, rectangular_fit_area)

    return {
        "width": width,
        "height": height,
        "frontal_area": frontal_area,
        "max_circular_diameter": max_diameter,
        "max_circular_radius": max_radius,
        "rectangular_fit_exit_area": rectangular_fit_area,
        "max_exit_area": selected_exit_area,
        "user_exit_area_limit": max_exit_area,
        "nozzle_diameter_clearance": nozzle_diameter_clearance,
    }


def rectangular_inlet_capture(inlet_width, inlet_height, capture_area_ratio):
    """
    Compute captured inlet area from the actual rectangular inlet opening.
    """

    if inlet_width <= 0.0 or inlet_height <= 0.0:
        raise ValueError("Inlet width and height must be positive.")
    if capture_area_ratio <= 0.0 or capture_area_ratio > 1.0:
        raise ValueError("capture_area_ratio must be in the range (0, 1].")

    inlet_area = inlet_width * inlet_height
    return {
        "width": inlet_width,
        "height": inlet_height,
        "area": inlet_area,
        "capture_area_ratio": capture_area_ratio,
        "capture_area": inlet_area * capture_area_ratio,
    }


def standard_atmosphere(altitude_ft):
    """
    1976-standard-atmosphere estimate used only to prepare flight-mode inputs.
    """

    h = altitude_ft * 0.3048
    t0 = 288.15
    p0 = 101325.0
    g = 9.80665
    r_air = 287.05
    gamma = 1.4

    if h < 11000.0:
        t_static = t0 - 0.0065 * h
        p_static = p0 * (t_static / t0) ** (g / (0.0065 * r_air))
    elif h < 20000.0:
        t_static = 216.65
        p_static = 22632.0 * np.exp(-g * (h - 11000.0) / (r_air * t_static))
    else:
        t_static = 216.65 + 0.001 * (h - 20000.0)
        p_static = 5474.9 * (t_static / 216.65) ** (g / (0.001 * r_air))

    rho = p_static / (r_air * t_static)
    speed_of_sound = np.sqrt(gamma * r_air * t_static)

    return {
        "P": p_static,
        "T": t_static,
        "rho": rho,
        "a": speed_of_sound,
        "altitude_ft": altitude_ft,
        "altitude_m": h,
    }


def standard_atmosphere_m(altitude_m):
    """1976-standard-atmosphere estimate with altitude supplied in meters."""

    return standard_atmosphere(altitude_m * FT_PER_M)


def isentropic_total_ratios(mach, gamma):
    """
    Total/static ratios used only to estimate nozzle inlet inputs in flight mode.
    """

    temp_ratio = 1.0 + 0.5 * (gamma - 1.0) * mach**2
    pressure_ratio = temp_ratio ** (gamma / (gamma - 1.0))
    return pressure_ratio, temp_ratio


def normal_shock_relations(mach_1, gamma):
    """
    Normal-shock estimate used only to prepare flight-mode nozzle inlet inputs.
    """

    mach_2 = np.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * mach_1**2)
        / (gamma * mach_1**2 - 0.5 * (gamma - 1.0))
    )
    pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mach_1**2 - 1.0)
    temperature_ratio = pressure_ratio * (2.0 + (gamma - 1.0) * mach_1**2) / (
        (gamma + 1.0) * mach_1**2
    )
    total_pressure_recovery = (
        ((gamma + 1.0) * mach_1**2 / (2.0 + (gamma - 1.0) * mach_1**2))
        ** (gamma / (gamma - 1.0))
        * ((gamma + 1.0) / (2.0 * gamma * mach_1**2 - (gamma - 1.0)))
        ** (1.0 / (gamma - 1.0))
    )

    return {
        "M2": mach_2,
        "P2_P1": pressure_ratio,
        "T2_T1": temperature_ratio,
        "Pt2_Pt1": total_pressure_recovery,
    }


def initial_exit_pressure_guess(pt_inlet, target_mach, gamma):
    """
    Provide an initial pressure guess for OpenMDAO's target-Mach balance.

    The final pressure and exit state are solved by pyCycle. This only improves
    convergence speed for the balance.
    """

    pressure_ratio, _ = isentropic_total_ratios(max(target_mach, 1.0e-6), gamma)
    return max(pt_inlet / pressure_ratio, 100.0)


def estimate_fuel_air_ratio(tt_before_combustor, tt_after_combustor, fuel_lhv, combustor_efficiency, cp_gas):
    """
    Estimate liquid-fuel/air ratio from a simple combustor energy balance.

    This is an input-preparation approximation. The pyCycle nozzle still solves
    the nozzle expansion; this only estimates fuel flow for ramjet metrics.
    """

    if tt_after_combustor <= tt_before_combustor:
        return 0.0

    denominator = combustor_efficiency * fuel_lhv - cp_gas * tt_after_combustor
    if denominator <= 0.0:
        raise ValueError(
            "Fuel heating value/combustor efficiency is too low for the requested combustor exit temperature."
        )

    return cp_gas * (tt_after_combustor - tt_before_combustor) / denominator


def estimate_flight_inputs(
    mach_flight,
    altitude_m,
    vehicle_width,
    vehicle_height,
    inlet_width,
    inlet_height,
    nozzle_diameter_clearance,
    max_exit_area,
    capture_area_ratio,
    combustor_exit_temperature,
    inlet_pressure_recovery,
    combustor_pressure_ratio,
    fuel_name,
    fuel_lhv,
    combustor_efficiency,
    combustor_cp,
    fuel_air_ratio,
    gamma,
    r_air,
):
    """
    Estimate nozzle-inlet total conditions and captured mass flow for flight mode.

    This is a simple inlet/combustor input estimate. The actual nozzle expansion
    and performance analysis are still done by pyCycle.
    """

    if combustor_exit_temperature <= 0.0:
        raise ValueError("combustor_exit_temperature must be positive.")
    if inlet_pressure_recovery <= 0.0 or inlet_pressure_recovery > 1.0:
        raise ValueError("inlet_pressure_recovery must be in the range (0, 1].")
    if combustor_pressure_ratio <= 0.0 or combustor_pressure_ratio > 1.0:
        raise ValueError("combustor_pressure_ratio must be in the range (0, 1].")

    atm = standard_atmosphere_m(altitude_m)
    p_total_ratio, t_total_ratio = isentropic_total_ratios(mach_flight, gamma)
    pt_freestream = atm["P"] * p_total_ratio
    tt_freestream = atm["T"] * t_total_ratio
    shock = normal_shock_relations(mach_flight, gamma)

    vehicle = rectangular_vehicle_packaging(
        width=vehicle_width,
        height=vehicle_height,
        nozzle_diameter_clearance=nozzle_diameter_clearance,
        max_exit_area=max_exit_area,
    )
    inlet = rectangular_inlet_capture(
        inlet_width=inlet_width,
        inlet_height=inlet_height,
        capture_area_ratio=capture_area_ratio,
    )

    pt_after_inlet = pt_freestream * inlet_pressure_recovery
    tt_after_inlet = tt_freestream
    pt_nozzle_inlet = pt_after_inlet * combustor_pressure_ratio
    tt_nozzle_inlet = combustor_exit_temperature

    if fuel_air_ratio is None:
        fuel_air_ratio = estimate_fuel_air_ratio(
            tt_before_combustor=tt_after_inlet,
            tt_after_combustor=tt_nozzle_inlet,
            fuel_lhv=fuel_lhv,
            combustor_efficiency=combustor_efficiency,
            cp_gas=combustor_cp,
        )

    capture_area = inlet["capture_area"]
    flight_velocity = mach_flight * atm["a"]
    mdot_air = atm["rho"] * flight_velocity * capture_area
    mdot_fuel = mdot_air * fuel_air_ratio
    mdot_nozzle = mdot_air + mdot_fuel

    return {
        "atm": atm,
        "shock": shock,
        "M_inlet": 0.3,
        "Pt_inlet": pt_nozzle_inlet,
        "Tt_inlet": tt_nozzle_inlet,
        "W": mdot_nozzle,
        "W_air": mdot_air,
        "W_fuel": mdot_fuel,
        "fuel_air_ratio": fuel_air_ratio,
        "Tt_before_combustor": tt_after_inlet,
        "P_ambient": atm["P"],
        "capture_area": capture_area,
        "inlet": inlet,
        "vehicle_area": vehicle["frontal_area"],
        "vehicle": vehicle,
        "pt_freestream": pt_freestream,
        "tt_freestream": tt_freestream,
        "inlet_pressure_recovery": inlet_pressure_recovery,
        "combustor_pressure_ratio": combustor_pressure_ratio,
        "flight_velocity": flight_velocity,
        "fuel_name": fuel_name,
        "fuel_lhv": fuel_lhv,
        "combustor_efficiency": combustor_efficiency,
        "combustor_cp": combustor_cp,
        "r_air": r_air,
        "gamma": gamma,
        "combustor_exit_temperature": combustor_exit_temperature,
    }


def build_pycycle_problem(
    m_inlet,
    pt_inlet,
    tt_inlet,
    mass_flow,
    ps_exhaust,
    cv,
    nozzle_type,
    flowstation=None,
    flow_port_data=None,
    target_exit_mach=None,
    gamma_for_guess=1.4,
):
    """
    Build a pyCycle FlowStart -> Nozzle problem.
    """

    prob = om.Problem(reports=False)
    cycle = prob.model = Cycle()
    cycle.options["thermo_method"] = "CEA"
    cycle.options["thermo_data"] = janaf

    cycle.add_subsystem(
        "nozzle",
        Nozzle(nozzType=nozzle_type, lossCoef="Cv", internal_solver=True),
    )

    if flowstation is None:
        flow_start = FlowStart()
        cycle.add_subsystem("flow_start", flow_start)
        cycle.pyc_connect_flow("flow_start.Fl_O", "nozzle.Fl_I")
    else:
        flow_source = FlowStationSource(flowstation=flowstation, flow_port_data=flow_port_data)
        cycle.add_subsystem("flow_source", flow_source)
        cycle.pyc_connect_flow("flow_source.Fl_O", "nozzle.Fl_I")

    if flowstation is None:
        cycle.set_input_defaults("flow_start.MN", m_inlet)
        cycle.set_input_defaults("flow_start.P", pt_inlet, units="Pa")
        cycle.set_input_defaults("flow_start.T", tt_inlet, units="K")
        cycle.set_input_defaults("flow_start.W", mass_flow, units="kg/s")
    cycle.set_input_defaults("nozzle.Cv", cv)

    if target_exit_mach is None:
        cycle.set_input_defaults("nozzle.Ps_exhaust", ps_exhaust, units="Pa")
    else:
        guess = initial_exit_pressure_guess(pt_inlet, target_exit_mach, gamma_for_guess)
        guess = min(max(guess, 100.0), 0.999 * pt_inlet)

        balance = cycle.add_subsystem("exit_mach_balance", om.BalanceComp())
        balance.add_balance(
            "Ps_exhaust",
            val=guess,
            units="Pa",
            lower=100.0,
            upper=max(0.999 * pt_inlet, 101.0),
            eq_units=None,
            lhs_name="MN_actual",
            rhs_name="MN_target",
        )
        cycle.connect("exit_mach_balance.Ps_exhaust", "nozzle.Ps_exhaust")
        cycle.connect("nozzle.Fl_O:stat:MN", "exit_mach_balance.MN_actual")
        cycle.set_input_defaults("exit_mach_balance.MN_target", target_exit_mach)

        newton = cycle.nonlinear_solver = om.NewtonSolver()
        newton.options["solve_subsystems"] = True
        newton.options["maxiter"] = 12
        newton.options["atol"] = 1.0e-8
        newton.options["rtol"] = 1.0e-8
        newton.options["iprint"] = 0
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1
        cycle.linear_solver = om.DirectSolver()

    prob.set_solver_print(level=-1)
    prob.setup(check=False)
    return prob


def run_model(prob, show_solver_warnings=False):
    """
    Run pyCycle and optionally suppress solver warnings from intermediate guesses.
    """

    if show_solver_warnings:
        prob.run_model()
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()


def get_value(prob, name, units=None):
    if units is None:
        return scalar(prob.get_val(name))
    return scalar(prob.get_val(name, units=units))


def station_from_prob(prob, prefix):
    """
    Read one pyCycle flow station in SI units.
    """

    return {
        "M": get_value(prob, f"{prefix}:stat:MN"),
        "P": get_value(prob, f"{prefix}:stat:P", "Pa"),
        "T": get_value(prob, f"{prefix}:stat:T", "K"),
        "rho": get_value(prob, f"{prefix}:stat:rho", "kg/m**3"),
        "V": get_value(prob, f"{prefix}:stat:V", "m/s"),
        "Vsonic": get_value(prob, f"{prefix}:stat:Vsonic", "m/s"),
        "area": get_value(prob, f"{prefix}:stat:area", "m**2"),
        "W": get_value(prob, f"{prefix}:stat:W", "kg/s"),
        "gamma": get_value(prob, f"{prefix}:stat:gamma"),
    }


def classify_expansion(exit_pressure, ambient_pressure, rel_tol=0.02):
    """
    Classify expansion relative to the cruise ambient pressure.
    """

    if ambient_pressure <= 0.0:
        return "vacuum reference"

    mismatch = (exit_pressure - ambient_pressure) / ambient_pressure
    if abs(mismatch) <= rel_tol:
        return "ideally expanded"
    if mismatch > 0.0:
        return "underexpanded"
    return "overexpanded"


def derive_performance(
    inlet,
    throat,
    exit_station,
    fg_pycycle,
    fg_ideal,
    ps_exhaust,
    ambient_pressure,
    cv,
    pr,
    choked,
    target_exit_mach=None,
    requested_throat_area=None,
):
    """
    Derive nozzle metrics from pyCycle station outputs.
    """

    mass_flow = exit_station["W"]
    denom = inlet["Pt"] * throat["area"]
    exit_pressure = exit_station["P"]
    exit_area = exit_station["area"]

    momentum_thrust = mass_flow * exit_station["V"] * cv
    pressure_thrust_pycycle = (exit_pressure - ps_exhaust) * exit_area
    pressure_thrust_ambient = (exit_pressure - ambient_pressure) * exit_area
    pressure_thrust_vacuum = exit_pressure * exit_area
    force_at_ambient = fg_pycycle + (ps_exhaust - ambient_pressure) * exit_area
    force_vacuum = fg_pycycle + ps_exhaust * exit_area

    return {
        "Fg": fg_pycycle,
        "Fg_ideal": fg_ideal,
        "F_momentum": momentum_thrust,
        "pressure_thrust_pycycle": pressure_thrust_pycycle,
        "pressure_thrust_ambient": pressure_thrust_ambient,
        "pressure_thrust_vac": pressure_thrust_vacuum,
        "F_cruise": force_at_ambient,
        "Fg_vacuum": force_vacuum,
        "pycycle_Cfg_effective": fg_pycycle / fg_ideal if abs(fg_ideal) > 1.0e-12 else np.nan,
        "pycycle_thrust_coeff": fg_pycycle / denom if denom > 0.0 else np.nan,
        "thrust_coeff": force_at_ambient / denom if denom > 0.0 else np.nan,
        "thrust_coeff_vacuum": force_vacuum / denom if denom > 0.0 else np.nan,
        "Cv": cv,
        "Isp": force_at_ambient / (mass_flow * G0) if mass_flow > 0.0 else np.nan,
        "Isp_pycycle": fg_pycycle / (mass_flow * G0) if mass_flow > 0.0 else np.nan,
        "Isp_vacuum": force_vacuum / (mass_flow * G0) if mass_flow > 0.0 else np.nan,
        "specific_thrust": force_at_ambient / mass_flow if mass_flow > 0.0 else np.nan,
        "specific_thrust_pycycle": fg_pycycle / mass_flow if mass_flow > 0.0 else np.nan,
        "specific_thrust_vacuum": force_vacuum / mass_flow if mass_flow > 0.0 else np.nan,
        "c_star": inlet["Pt"] * throat["area"] / mass_flow if mass_flow > 0.0 else np.nan,
        "PR": pr,
        "area_ratio": exit_area / throat["area"] if throat["area"] > 0.0 else np.nan,
        "choked": choked,
        "P_exhaust": ps_exhaust,
        "P_ambient": ambient_pressure,
        "Pe_over_Pamb": exit_pressure / ambient_pressure if ambient_pressure > 0.0 else np.nan,
        "expansion_state": classify_expansion(exit_pressure, ambient_pressure),
        "target_exit_mach": target_exit_mach,
        "requested_throat_area": requested_throat_area,
    }


def collect_results(prob, target_exit_mach=None, requested_throat_area=None, ambient_pressure=None):
    """
    Collect pyCycle nozzle outputs and derived performance metrics.
    """
    source_prefix = "flow_start.Fl_O"
    try:
        prob.get_val("flow_start.Fl_O:stat:MN")
    except Exception:
        source_prefix = "flow_source.Fl_O"

    inlet = station_from_prob(prob, source_prefix)
    throat = station_from_prob(prob, "nozzle.Throat")
    exit_station = station_from_prob(prob, "nozzle.Fl_O")

    inlet["Pt"] = get_value(prob, f"{source_prefix}:tot:P", "Pa")
    inlet["Tt"] = get_value(prob, f"{source_prefix}:tot:T", "K")
    exit_station["Pt"] = get_value(prob, "nozzle.Fl_O:tot:P", "Pa")
    exit_station["Tt"] = get_value(prob, "nozzle.Fl_O:tot:T", "K")

    fg = get_value(prob, "nozzle.Fg", "N")
    fg_ideal = get_value(prob, "nozzle.perf_calcs.Fg_ideal", "N")
    ps_exhaust = get_value(prob, "nozzle.Ps_exhaust", "Pa")
    if ambient_pressure is None:
        ambient_pressure = ps_exhaust
    pr = get_value(prob, "nozzle.PR")
    choked = bool(round(get_value(prob, "nozzle.mux.choked")))

    return {
        "inlet": inlet,
        "throat": throat,
        "exit": exit_station,
        "performance": derive_performance(
            inlet=inlet,
            throat=throat,
            exit_station=exit_station,
            fg_pycycle=fg,
            fg_ideal=fg_ideal,
            ps_exhaust=ps_exhaust,
            ambient_pressure=ambient_pressure,
            cv=get_value(prob, "nozzle.Cv"),
            pr=pr,
            choked=choked,
            target_exit_mach=target_exit_mach,
            requested_throat_area=requested_throat_area,
        ),
    }


def scale_results_by_area_factor(results, scale, requested_throat_area=None):
    """
    Scale pyCycle's mass-flow-dependent outputs by a geometric area factor.

    For a fixed thermodynamic state and pressure ratio, pyCycle station areas,
    mass flow, and thrust scale linearly together. This avoids rerunning the
    nonlinear model just to resize the nozzle.
    """

    if scale <= 0.0:
        raise ValueError("Area scale factor must be positive.")

    for station_name in ("inlet", "throat", "exit"):
        results[station_name]["area"] *= scale
        results[station_name]["W"] *= scale

    perf = results["performance"]
    fg_pycycle = perf["Fg"] * scale
    fg_ideal = perf["Fg_ideal"] * scale
    results["performance"] = derive_performance(
        inlet=results["inlet"],
        throat=results["throat"],
        exit_station=results["exit"],
        fg_pycycle=fg_pycycle,
        fg_ideal=fg_ideal,
        ps_exhaust=perf["P_exhaust"],
        ambient_pressure=perf["P_ambient"],
        cv=perf["Cv"],
        pr=perf["PR"],
        choked=perf["choked"],
        target_exit_mach=perf["target_exit_mach"],
        requested_throat_area=requested_throat_area,
    )
    return results


def scale_results_to_throat_area(results, throat_area):
    """
    Scale pyCycle's mass-flow-dependent outputs to a requested throat area.
    """

    current_area = results["throat"]["area"]
    if current_area <= 0.0:
        raise ValueError("Cannot scale results because pyCycle returned a nonpositive throat area.")

    return scale_results_by_area_factor(
        results,
        scale=throat_area / current_area,
        requested_throat_area=throat_area,
    )


def apply_exit_area_limit(results, case, max_exit_area, quiet=False):
    """
    Limit nozzle size by scaling mass flow/areas to a maximum exit area.

    This keeps the same pyCycle thermodynamic state and area ratio. It is a
    packaging rescale, not a new inlet/combustor solution.
    """

    if max_exit_area is None:
        return results
    if max_exit_area <= 0.0:
        raise ValueError("max_exit_area must be positive.")

    current_exit_area = results["exit"]["area"]
    perf = results["performance"]
    perf["exit_area_limit"] = max_exit_area
    perf["exit_area_limited"] = False
    perf["area_limit_scale"] = 1.0
    perf["area_constraint_mode"] = "within exit-area limit"

    if current_exit_area <= max_exit_area:
        return results

    scale = max_exit_area / current_exit_area
    requested_throat_area = perf["requested_throat_area"]
    if requested_throat_area is not None:
        requested_throat_area *= scale

    results = scale_results_by_area_factor(
        results,
        scale=scale,
        requested_throat_area=requested_throat_area,
    )

    perf = results["performance"]
    perf["exit_area_limit"] = max_exit_area
    perf["exit_area_limited"] = True
    perf["area_limit_scale"] = scale
    perf["area_constraint_mode"] = "mass-flow/area scaled fallback"
    perf["unlimited_exit_area"] = current_exit_area
    perf["unlimited_mass_flow"] = case["W"]

    for key in ("W", "air_mass_flow", "fuel_mass_flow"):
        if case.get(key) is not None:
            case[key] *= scale

    if case.get("flight") is not None:
        for key in ("W", "W_air", "W_fuel"):
            if case["flight"].get(key) is not None:
                case["flight"][key] *= scale

    if not quiet:
        print("\n" + "=" * 72)
        print("NOZZLE PACKAGING RESCALE")
        print("=" * 72)
        print(f"Unlimited pyCycle exit area: {current_exit_area:.6f} m^2")
        print(f"Maximum allowed exit area:   {max_exit_area:.6f} m^2")
        print(f"Area/mass-flow scale factor: {scale:.6f}")
        print(f"Scaled nozzle mass flow:     {results['exit']['W']:.4f} kg/s")
        print("Note: thermodynamic state and Ae/At are unchanged; mass flow and thrust are scaled.")

    return results


def area_mach_ratio(mach, gamma):
    """
    Isentropic area ratio A/A* for a given Mach number.
    """

    if mach <= 0.0:
        raise ValueError("Mach number must be positive.")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")

    factor = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return factor**exponent / mach


def supersonic_mach_from_area_ratio(area_ratio, gamma):
    """
    Estimate supersonic Mach number from area ratio using a constant-gamma relation.

    This is used only to select the pyCycle target Mach for an exit-area-limited
    nozzle. pyCycle still solves the final station properties.
    """

    if area_ratio < 1.0:
        raise ValueError("Supersonic nozzle area ratio must be at least 1.")
    if area_ratio <= 1.0001:
        return 1.0001

    lower = 1.0001
    upper = 8.0
    while area_mach_ratio(upper, gamma) < area_ratio:
        upper *= 1.5
        if upper > 25.0:
            raise ValueError("Could not bracket supersonic Mach for the requested area ratio.")

    for _ in range(80):
        mid = 0.5 * (lower + upper)
        if area_mach_ratio(mid, gamma) < area_ratio:
            lower = mid
        else:
            upper = mid

    return 0.5 * (lower + upper)


def solve_exit_area_constrained_nozzle(
    initial_results,
    case,
    max_exit_area,
    cv,
    nozzle_type,
    show_solver_warnings=False,
):
    """
    Re-run pyCycle with a target exit Mach so fixed mass flow fits max exit area.
    """

    if max_exit_area is None:
        return initial_results

    if max_exit_area <= 0.0:
        raise ValueError("max_exit_area must be positive.")

    initial_perf = initial_results["performance"]
    initial_perf["exit_area_limit"] = max_exit_area
    initial_perf["exit_area_limited"] = False
    initial_perf["area_constraint_mode"] = "within exit-area limit"

    if initial_results["exit"]["area"] <= max_exit_area:
        return initial_results

    throat_area = initial_results["throat"]["area"]
    if throat_area <= 0.0:
        raise ValueError("Cannot apply exit-area constraint because throat area is nonpositive.")

    target_area_ratio = max_exit_area / throat_area
    if target_area_ratio < 1.0:
        print("\n" + "=" * 72)
        print("NOZZLE GEOMETRY CONSTRAINT")
        print("=" * 72)
        print(f"Ideal-expanded exit area: {initial_results['exit']['area']:.6f} m^2")
        print(f"Maximum allowed exit area:{max_exit_area:.6f} m^2")
        print(f"Required throat area:     {throat_area:.6f} m^2")
        print("Fixed mass flow cannot fit because the throat area exceeds the exit-area limit.")
        print("Falling back to mass-flow/area scaling.")
        return apply_exit_area_limit(initial_results, case, max_exit_area)

    gamma_guess = initial_results["throat"].get("gamma", case.get("gamma", 1.4))
    commanded_area_ratio = target_area_ratio
    target_exit_mach = supersonic_mach_from_area_ratio(commanded_area_ratio, gamma_guess)
    constrained = None

    for _ in range(4):
        used_target_exit_mach = target_exit_mach
        used_commanded_area_ratio = commanded_area_ratio
        constrained = run_pycycle_nozzle(
            m_inlet=case["M_inlet"],
            pt_inlet=case["Pt_inlet"],
            tt_inlet=case["Tt_inlet"],
            ps_exhaust=case["P_ambient"],
            cv=cv,
            nozzle_type=nozzle_type,
            mass_flow=case["W"],
            throat_area=case["A_throat"],
            target_exit_mach=used_target_exit_mach,
            ambient_pressure=case["P_ambient"],
            gamma_for_guess=case["gamma"],
            show_solver_warnings=show_solver_warnings,
        )

        exit_area_error = constrained["exit"]["area"] - max_exit_area
        if abs(exit_area_error) <= max(5.0e-4, 0.005 * max_exit_area):
            break

        actual_area_ratio = constrained["exit"]["area"] / constrained["throat"]["area"]
        commanded_area_ratio = max(1.0001, commanded_area_ratio + (target_area_ratio - actual_area_ratio))
        target_exit_mach = supersonic_mach_from_area_ratio(
            commanded_area_ratio,
            constrained["throat"].get("gamma", gamma_guess),
        )

    perf = constrained["performance"]
    perf["exit_area_limit"] = max_exit_area
    perf["exit_area_limited"] = True
    perf["area_constraint_mode"] = "fixed mass flow, exit-area-constrained solve"
    perf["ideal_expanded_exit_area"] = initial_results["exit"]["area"]
    perf["target_area_ratio_for_limit"] = target_area_ratio
    perf["commanded_area_ratio_for_mach"] = used_commanded_area_ratio
    perf["geometric_target_exit_mach"] = used_target_exit_mach
    perf["area_limit_error"] = constrained["exit"]["area"] - max_exit_area

    print("\n" + "=" * 72)
    print("NOZZLE GEOMETRY CONSTRAINT SOLVE")
    print("=" * 72)
    print(f"Ideal-expanded exit area: {initial_results['exit']['area']:.6f} m^2")
    print(f"Maximum allowed exit area:{max_exit_area:.6f} m^2")
    print(f"Target Ae/At for limit:   {target_area_ratio:.4f}")
    print(f"Solved target exit Mach:  {target_exit_mach:.4f}")
    print(f"Constrained exit area:    {constrained['exit']['area']:.6f} m^2")
    print(f"Exit area error:          {perf['area_limit_error']:.6e} m^2")
    print(f"Exit pressure ratio:      {perf['Pe_over_Pamb']:.4f}")
    print("Note: mass flow is preserved; smaller exit area usually means underexpanded flow.")

    return constrained


def run_pycycle_nozzle(
    m_inlet,
    pt_inlet,
    tt_inlet,
    ps_exhaust,
    cv,
    nozzle_type,
    mass_flow=None,
    throat_area=None,
    flowstation=None,
    flow_port_data=None,
    target_exit_mach=None,
    ambient_pressure=None,
    gamma_for_guess=1.4,
    show_solver_warnings=False,
):
    """
    Run pyCycle. If throat_area is supplied without mass_flow, pyCycle is first
    run at 1 kg/s and then mass flow is scaled to match the pyCycle throat area.
    """

    if mass_flow is None and throat_area is None:
        mass_flow = 1.0

    first_mass_flow = mass_flow if mass_flow is not None else 1.0
    prob = build_pycycle_problem(
        m_inlet=m_inlet,
        pt_inlet=pt_inlet,
        tt_inlet=tt_inlet,
        mass_flow=first_mass_flow,
        ps_exhaust=ps_exhaust,
        cv=cv,
        nozzle_type=nozzle_type,
        flowstation=flowstation,
        flow_port_data=flow_port_data,
        target_exit_mach=target_exit_mach,
        gamma_for_guess=gamma_for_guess,
    )

    run_model(prob, show_solver_warnings=show_solver_warnings)
    requested_throat_area = throat_area

    results = collect_results(
        prob,
        target_exit_mach=target_exit_mach,
        requested_throat_area=requested_throat_area,
        ambient_pressure=ambient_pressure if ambient_pressure is not None else ps_exhaust,
    )

    if mass_flow is None and throat_area is not None:
        results = scale_results_to_throat_area(results, throat_area)

    return results


def print_flight_estimate(flight):
    atm = flight["atm"]
    shock = flight["shock"]
    vehicle = flight["vehicle"]
    inlet = flight["inlet"]

    print("\n" + "=" * 72)
    print("FLIGHT-MODE INPUT ESTIMATE")
    print("=" * 72)
    print(f"Altitude:                 {atm['altitude_m']:.0f} m ({atm['altitude_ft']:.0f} ft)")
    print(f"Atmospheric pressure:     {atm['P']/1e3:.3f} kPa")
    print(f"Atmospheric temperature:  {atm['T']:.2f} K")
    print(f"Atmospheric density:      {atm['rho']:.6f} kg/m^3")
    print(f"Flight velocity:          {flight['flight_velocity']:.2f} m/s")
    print(f"Freestream total pressure:{flight['pt_freestream']/1e3:.2f} kPa")
    print(f"Freestream total temp.:   {flight['tt_freestream']:.2f} K")
    print(f"Inlet pressure recovery:  {flight['inlet_pressure_recovery']:.3f}")
    print(f"Normal-shock recovery ref:{shock['Pt2_Pt1']:.3f}")
    print(f"Combustor pressure ratio: {flight['combustor_pressure_ratio']:.3f}")
    print(f"Combustor exit Tt target: {flight['combustor_exit_temperature']:.2f} K")
    print(f"Fuel:                     {flight['fuel_name']}")
    print(f"Fuel LHV:                 {flight['fuel_lhv']/1e6:.2f} MJ/kg")
    print(f"Vehicle width x height:   {vehicle['width']:.3f} m x {vehicle['height']:.3f} m")
    print(f"Vehicle frontal area:     {vehicle['frontal_area']:.6f} m^2")
    print(f"Max circular exit diam.:  {vehicle['max_circular_diameter']:.4f} m")
    print(f"Rect. fit exit area:      {vehicle['rectangular_fit_exit_area']:.6f} m^2")
    print(f"Selected max exit area:   {vehicle['max_exit_area']:.6f} m^2")
    print(f"Inlet width x height:     {inlet['width']:.3f} m x {inlet['height']:.3f} m")
    print(f"Inlet geometric area:     {inlet['area']:.6f} m^2")
    print(f"Inlet capture fraction:   {inlet['capture_area_ratio']:.3f}")
    print(f"Capture area:             {flight['capture_area']:.6f} m^2")
    print(f"Captured air flow:        {flight['W_air']:.4f} kg/s")
    print(f"Estimated fuel flow:      {flight['W_fuel']:.4f} kg/s")
    print(f"Fuel-air ratio:           {flight['fuel_air_ratio']:.5f}")
    print(f"Total nozzle mass flow:   {flight['W']:.4f} kg/s")
    print(f"Combustor inlet Tt est.:  {flight['Tt_before_combustor']:.2f} K")
    print(f"Nozzle inlet Pt estimate: {flight['Pt_inlet']/1e3:.2f} kPa")
    print(f"Nozzle inlet Tt estimate: {flight['Tt_inlet']:.2f} K")


def fuel_split_from_total(total_mass_flow, fuel_air_ratio=None, fuel_flow=None):
    """
    Split total nozzle flow into air and fuel flow if fuel information is available.
    """

    if fuel_flow is not None:
        air_flow = total_mass_flow - fuel_flow
        if air_flow <= 0.0:
            raise ValueError("Fuel flow must be smaller than total nozzle mass flow.")
        return air_flow, fuel_flow, fuel_flow / air_flow

    if fuel_air_ratio is not None:
        air_flow = total_mass_flow / (1.0 + fuel_air_ratio)
        fuel_flow = total_mass_flow - air_flow
        return air_flow, fuel_flow, fuel_air_ratio

    return None, None, None


def add_ramjet_metrics(results, case, args):
    """
    Add liquid-ramjet fuel-based Isp, ram drag, and TSFC metrics when possible.
    """

    perf = results["performance"]
    total_mass_flow = results["exit"]["W"]
    flight_velocity = case.get("flight_velocity")

    air_flow = case.get("air_mass_flow")
    fuel_flow = case.get("fuel_mass_flow")
    fuel_air_ratio = case.get("fuel_air_ratio")

    if air_flow is None or fuel_flow is None:
        air_flow, fuel_flow, fuel_air_ratio = fuel_split_from_total(
            total_mass_flow,
            fuel_air_ratio=args.fuel_air_ratio,
            fuel_flow=args.fuel_flow,
        )

    if fuel_flow is None or fuel_flow <= 0.0:
        perf["ramjet_metrics_available"] = False
        return results

    ram_drag = air_flow * flight_velocity if flight_velocity is not None else np.nan
    net_thrust = perf["F_cruise"] - ram_drag if flight_velocity is not None else np.nan

    perf.update(
        {
            "ramjet_metrics_available": True,
            "air_mass_flow": air_flow,
            "fuel_mass_flow": fuel_flow,
            "fuel_air_ratio": fuel_air_ratio,
            "flight_velocity": flight_velocity,
            "ram_drag": ram_drag,
            "ramjet_net_thrust": net_thrust,
            "fuel_Isp_gross": perf["F_cruise"] / (fuel_flow * G0),
            "fuel_Isp_net": net_thrust / (fuel_flow * G0) if np.isfinite(net_thrust) else np.nan,
            "fuel_specific_thrust_gross": perf["F_cruise"] / fuel_flow,
            "fuel_specific_thrust_net": net_thrust / fuel_flow if np.isfinite(net_thrust) else np.nan,
            "TSFC_kg_per_N_s": fuel_flow / net_thrust if np.isfinite(net_thrust) and net_thrust > 0.0 else np.nan,
            "TSFC_lbm_per_lbf_hr": (
                fuel_flow / net_thrust * LBM_PER_KG / LBF_PER_N * 3600.0
                if np.isfinite(net_thrust) and net_thrust > 0.0
                else np.nan
            ),
        }
    )
    return results


def print_results(results):
    inlet = results["inlet"]
    throat = results["throat"]
    exit_station = results["exit"]
    perf = results["performance"]

    print("\n" + "=" * 72)
    print("PYCYCLE NOZZLE RESULTS")
    print("=" * 72)
    print("Analysis engine: pyCycle FlowStart + pyCycle Nozzle")

    if perf["target_exit_mach"] is not None:
        print(f"Target exit Mach:         {perf['target_exit_mach']:.4f}")
    print(f"pyCycle exhaust pressure: {perf['P_exhaust']/1e3:.3f} kPa")
    print(f"Cruise ambient pressure:  {perf['P_ambient']/1e3:.3f} kPa")
    print(f"Nozzle pressure ratio PR: {perf['PR']:.4f}")
    print(f"Choked:                   {'Yes' if perf['choked'] else 'No'}")
    print(f"Expansion state:          {perf['expansion_state']}")

    print("\nStation properties")
    print("-" * 72)
    print(f"{'Station':<12}{'Mach':>10}{'P [kPa]':>12}{'T [K]':>12}{'V [m/s]':>12}{'Area [m^2]':>14}")
    for name, station in (
        ("Inlet", inlet),
        ("Throat", throat),
        ("Exit", exit_station),
    ):
        print(
            f"{name:<12}"
            f"{station['M']:>10.4f}"
            f"{station['P']/1e3:>12.3f}"
            f"{station['T']:>12.2f}"
            f"{station['V']:>12.2f}"
            f"{station['area']:>14.6f}"
        )

    print("\nPerformance")
    print("-" * 72)
    print(f"Mass flow:                {exit_station['W']:.4f} kg/s")
    print(f"pyCycle gross thrust Fg:  {perf['Fg']:.2f} N")
    print(f"Ideal gross thrust:       {perf['Fg_ideal']:.2f} N")
    print(f"Momentum thrust:          {perf['F_momentum']:.2f} N")
    print(f"Cruise pressure thrust:   {perf['pressure_thrust_ambient']:.2f} N")
    print(f"Cruise nozzle force:      {perf['F_cruise']:.2f} N")
    print(f"Vacuum pressure thrust:   {perf['pressure_thrust_vac']:.2f} N")
    print(f"Vacuum gross thrust:      {perf['Fg_vacuum']:.2f} N")
    print(f"pyCycle Cfg eff.:         {perf['pycycle_Cfg_effective']:.4f}")
    print(f"pyCycle thrust coeff:     {perf['pycycle_thrust_coeff']:.4f}")
    print(f"Cruise thrust coeff Cf:   {perf['thrust_coeff']:.4f}")
    print(f"Vacuum thrust coeff Cfv:  {perf['thrust_coeff_vacuum']:.4f}")
    print(f"Velocity coefficient, Cv: {perf['Cv']:.4f}")
    print(f"Specific thrust:          {perf['specific_thrust']:.2f} N/(kg/s)")
    print(f"pyCycle specific thrust:  {perf['specific_thrust_pycycle']:.2f} N/(kg/s)")
    print(f"Vacuum specific thrust:   {perf['specific_thrust_vacuum']:.2f} N/(kg/s)")
    print(f"Total-flow Isp:           {perf['Isp']:.2f} s")
    print(f"pyCycle total-flow Isp:   {perf['Isp_pycycle']:.2f} s")
    print(f"Vacuum total-flow Isp:    {perf['Isp_vacuum']:.2f} s")
    print(f"Characteristic velocity:  {perf['c_star']:.2f} m/s")
    print(f"Area ratio Ae/At:         {perf['area_ratio']:.4f}")
    print(f"Exit pressure ratio:      {perf['Pe_over_Pamb']:.4f}")
    if perf.get("exit_area_limit") is not None:
        print(f"Max exit area limit:      {perf['exit_area_limit']:.6f} m^2")
        print(f"Exit area limited:        {'Yes' if perf.get('exit_area_limited') else 'No'}")
        if perf.get("area_constraint_mode"):
            print(f"Area constraint mode:     {perf['area_constraint_mode']}")
        if perf.get("ideal_expanded_exit_area") is not None:
            print(f"Ideal-expanded exit area: {perf['ideal_expanded_exit_area']:.6f} m^2")
            print(f"Target Mach for area fit: {perf['geometric_target_exit_mach']:.4f}")
            print(f"Exit area error:          {perf['area_limit_error']:.6e} m^2")
        if perf.get("exit_area_limited"):
            if perf.get("unlimited_exit_area") is not None:
                print(f"Unlimited exit area:      {perf['unlimited_exit_area']:.6f} m^2")
            if perf.get("area_limit_scale") is not None:
                print(f"Area/mass-flow scale:     {perf['area_limit_scale']:.6f}")

    if perf.get("ramjet_metrics_available"):
        print("\nLiquid ramjet fuel-based metrics")
        print("-" * 72)
        print(f"Air mass flow:            {perf['air_mass_flow']:.4f} kg/s")
        print(f"Fuel mass flow:           {perf['fuel_mass_flow']:.4f} kg/s")
        print(f"Fuel-air ratio:           {perf['fuel_air_ratio']:.5f}")
        if perf["flight_velocity"] is not None:
            print(f"Flight velocity:          {perf['flight_velocity']:.2f} m/s")
            print(f"Ram drag estimate:        {perf['ram_drag']:.2f} N")
            print(f"Ramjet net thrust est.:   {perf['ramjet_net_thrust']:.2f} N")
            print(f"Fuel Isp, net:            {perf['fuel_Isp_net']:.2f} s")
            print(f"TSFC:                     {perf['TSFC_lbm_per_lbf_hr']:.4f} lbm/(lbf*hr)")
        else:
            print("Ram drag/net thrust:      not available without flight velocity")
        print(f"Fuel Isp, gross nozzle:   {perf['fuel_Isp_gross']:.2f} s")

    if perf["area_ratio"] < 1.05:
        print(
            "NOTE: Ae/At is near 1 because the nozzle pressure ratio is low. "
            "Use a lower cruise ambient pressure or a higher inlet total pressure "
            "to get a larger divergent section."
        )

    if perf["requested_throat_area"] is not None:
        error = throat["area"] - perf["requested_throat_area"]
        print(f"Requested throat area:    {perf['requested_throat_area']:.6f} m^2")
        print(f"Throat area error:        {error:.6e} m^2")


def default_bell_diverging_length(throat_area, exit_area):
    """
    Estimate a starting bell-nozzle divergent length from throat/exit areas.
    """

    r_throat = np.sqrt(max(throat_area, 1.0e-12) / np.pi)
    r_exit = np.sqrt(max(exit_area, 1.0e-12) / np.pi)
    radius_change = max(r_exit - r_throat, 1.0e-6)
    reference_angle = np.radians(15.0)
    # Roughly a 75%-length bell compared with a 15-degree conical nozzle.
    return max(0.50, 0.75 * radius_change / np.tan(reference_angle))


def default_bell_converging_length(inlet_area, throat_area):
    """
    Estimate converging length from the actual area contraction.

    This uses an equivalent conical contraction half-angle of 30 degrees to
    size the axial length from the inlet and throat radii.
    """

    r_inlet = np.sqrt(max(inlet_area, 1.0e-12) / np.pi)
    r_throat = np.sqrt(max(throat_area, 1.0e-12) / np.pi)
    radius_change = max(r_inlet - r_throat, 1.0e-6)
    reference_angle = np.radians(30.0)
    return max(0.02, radius_change / np.tan(reference_angle))


def size_bell_nozzle_for_vehicle(
    inlet_area,
    throat_area,
    exit_area,
    converging_length,
    diverging_length,
    missile_length,
    nozzle_length_fraction,
    nozzle_length,
    vehicle_radius,
):
    """
    Choose bell-contour lengths that fit a missile packaging budget.

    This sizes the generated geometry only. pyCycle still defines the station
    areas and thermodynamic performance.
    """

    if missile_length <= 0.0:
        raise ValueError("missile_length must be positive.")
    if nozzle_length_fraction <= 0.0:
        raise ValueError("nozzle_length_fraction must be positive.")
    if nozzle_length is not None and nozzle_length <= 0.0:
        raise ValueError("nozzle_length must be positive.")

    r_inlet = np.sqrt(max(inlet_area, 1.0e-12) / np.pi)
    r_throat = np.sqrt(max(throat_area, 1.0e-12) / np.pi)
    r_exit = np.sqrt(max(exit_area, 1.0e-12) / np.pi)

    length_budget = nozzle_length if nozzle_length is not None else missile_length * nozzle_length_fraction
    if length_budget <= 0.0:
        raise ValueError("Nozzle length budget must be positive.")

    unconstrained_converging_length = default_bell_converging_length(inlet_area, throat_area)
    unconstrained_diverging_length = default_bell_diverging_length(throat_area, exit_area)
    notes = []

    if converging_length is None:
        sized_converging_length = unconstrained_converging_length
        notes.append("Converging length was solved from inlet/throat area contraction.")
    else:
        if converging_length <= 0.0:
            raise ValueError("converging_length must be positive.")
        sized_converging_length = converging_length
    if diverging_length is None and sized_converging_length >= 0.55 * length_budget:
        sized_converging_length = max(0.05, 0.35 * length_budget)
        notes.append("Converging length was reduced to leave room for the divergent bell.")

    if diverging_length is None:
        available_diverging_length = length_budget - sized_converging_length
        if available_diverging_length <= 0.05:
            raise ValueError(
                "Nozzle length budget is too short for the requested converging section. "
                "Increase --nozzle_length/--nozzle_length_fraction or reduce --converging_length."
            )
        sized_diverging_length = min(unconstrained_diverging_length, available_diverging_length)
        length_limited = unconstrained_diverging_length > available_diverging_length
        if length_limited:
            notes.append("Diverging length was capped by the missile length budget.")
    else:
        if diverging_length <= 0.0:
            raise ValueError("diverging_length must be positive.")
        sized_diverging_length = diverging_length
        length_limited = sized_converging_length + sized_diverging_length > length_budget
        if length_limited:
            notes.append("User-specified nozzle length exceeds the missile length budget.")

    total_length = sized_converging_length + sized_diverging_length
    radius_change = r_exit - r_throat
    equivalent_half_angle = np.degrees(np.arctan2(max(radius_change, 0.0), sized_diverging_length))
    if equivalent_half_angle > 20.0:
        notes.append("Equivalent divergent half-angle is high; expect losses in a real nozzle.")

    if vehicle_radius is not None and vehicle_radius > 0.0 and r_exit > vehicle_radius:
        notes.append("Exit radius is larger than the vehicle radius; reduce exit area or mass flow.")

    return {
        "missile_length": missile_length,
        "nozzle_length_fraction": nozzle_length_fraction,
        "nozzle_length_overridden": nozzle_length is not None,
        "length_budget": length_budget,
        "converging_length": sized_converging_length,
        "diverging_length": sized_diverging_length,
        "total_length": total_length,
        "length_fraction": total_length / missile_length,
        "unconstrained_converging_length": unconstrained_converging_length,
        "unconstrained_diverging_length": unconstrained_diverging_length,
        "length_limited": length_limited,
        "r_inlet": r_inlet,
        "r_throat": r_throat,
        "r_exit": r_exit,
        "equivalent_converging_half_angle_deg": np.degrees(
            np.arctan2(max(r_inlet - r_throat, 0.0), sized_converging_length)
        ),
        "equivalent_divergent_half_angle_deg": equivalent_half_angle,
        "notes": notes,
    }


def hermite_curve(x0, y0, slope0, x1, y1, slope1, n_points):
    """
    Cubic Hermite curve for a smooth bell-nozzle wall contour.
    """

    t = np.linspace(0.0, 1.0, n_points)
    dx = x1 - x0
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    x = x0 + t * dx
    y = h00 * y0 + h10 * dx * slope0 + h01 * y1 + h11 * dx * slope1
    return x, np.maximum(y, 1.0e-6)


def generate_bell_contour(
    inlet_area,
    throat_area,
    exit_area,
    converging_length,
    diverging_length,
    throat_angle_deg,
    exit_angle_deg,
    n_points,
):
    """
    Build a smooth nozzle contour from pyCycle station areas.

    pyCycle gives station areas, not wall shape. This function creates a
    reasonable bell-style profile for visualization and preliminary geometry.
    Performance still comes from pyCycle station areas. For plotting, the
    area distribution is converted to a 2D rectangular duct of constant width.
    """

    r_inlet = np.sqrt(max(inlet_area, 1.0e-12) / np.pi)
    r_throat = np.sqrt(max(throat_area, 1.0e-12) / np.pi)
    r_exit = np.sqrt(max(exit_area, 1.0e-12) / np.pi)

    if n_points < 90:
        raise ValueError("n_points must be at least 90 for a smooth bell contour.")

    if converging_length is None:
        converging_length = default_bell_converging_length(inlet_area, throat_area)
    elif converging_length <= 0.0:
        raise ValueError("converging_length must be positive.")

    throat_angle = np.radians(throat_angle_deg)
    exit_angle = np.radians(exit_angle_deg)

    if diverging_length is None:
        diverging_length = default_bell_diverging_length(throat_area, exit_area)
    elif diverging_length <= 0.0:
        raise ValueError("diverging_length must be positive.")

    n_conv = max(30, int(n_points * 0.35))
    n_div = max(60, n_points - n_conv)

    x_conv = np.linspace(-converging_length, 0.0, n_conv)
    s = (x_conv + converging_length) / converging_length
    # Cosine easing gives zero wall slope at the inlet and throat.
    r_conv = r_throat + (r_inlet - r_throat) * 0.5 * (1.0 + np.cos(np.pi * s))

    x_div, r_div = hermite_curve(
        0.0,
        r_throat,
        np.tan(throat_angle),
        diverging_length,
        r_exit,
        np.tan(exit_angle),
        n_div,
    )

    x = np.concatenate([x_conv, x_div[1:]])
    r = np.concatenate([r_conv, r_div[1:]])
    area = np.pi * r**2
    width = float(INLET_DESIGN_WIDTH_M)
    height = area / width
    half_height = 0.5 * height

    return {
        "x": x,
        "radius": r,
        "area": area,
        "width": width,
        "height": height,
        "half_height": half_height,
        "upper_wall": half_height,
        "lower_wall": -half_height,
        "r_inlet": r_inlet,
        "r_throat": r_throat,
        "r_exit": r_exit,
        "converging_length": converging_length,
        "diverging_length": diverging_length,
        "throat_angle_deg": throat_angle_deg,
        "exit_angle_deg": exit_angle_deg,
    }


def prepare_plot_geometry_areas(inlet_area, throat_area, exit_area, min_gap_fraction=0.005):
    """
    Build a physically monotonic area set for geometry display only.

    pyCycle station areas are retained for reporting/performance, but the bell
    contour plot should always represent a converging-diverging nozzle with
    A_inlet > A_throat < A_exit. If the raw areas violate that ordering, clamp
    the displayed throat area slightly below the smaller end area and record a
    note explaining the adjustment.
    """

    if inlet_area <= 0.0 or throat_area <= 0.0 or exit_area <= 0.0:
        raise ValueError("Plot geometry areas must all be positive.")

    display_inlet_area = inlet_area
    display_throat_area = throat_area
    display_exit_area = exit_area
    notes = []
    adjusted = False

    min_end_area = min(display_inlet_area, display_exit_area)
    if display_throat_area >= min_end_area:
        adjusted = True
        gap = max(min_end_area * min_gap_fraction, 1.0e-9)
        display_throat_area = max(min_end_area - gap, 1.0e-12)
        notes.append(
            "Display geometry throat area was clamped below the smaller end area "
            "because the raw pyCycle station areas did not form a physical C-D nozzle sequence."
        )

    return {
        "raw_inlet_area": inlet_area,
        "raw_throat_area": throat_area,
        "raw_exit_area": exit_area,
        "inlet_area": display_inlet_area,
        "throat_area": display_throat_area,
        "exit_area": display_exit_area,
        "adjusted": adjusted,
        "notes": notes,
    }


def save_contour_csv(contour, output_path):
    """
    Save the generated nozzle contour for CAD/sketch import.
    """

    data = np.column_stack((contour["x"], contour["upper_wall"], contour["lower_wall"], contour["height"], contour["area"]))
    header = "x_m,wall_upper_m,wall_lower_m,height_m,area_m2"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def plot_results(
    results,
    output_path,
    geometry_csv_path,
    converging_length,
    diverging_length,
    bell_throat_angle,
    bell_exit_angle,
    bell_points,
    missile_length,
    nozzle_length_fraction,
    nozzle_length,
    vehicle_radius,
):
    inlet = results["inlet"]
    throat = results["throat"]
    exit_station = results["exit"]
    perf = results["performance"]
    plot_geometry = prepare_plot_geometry_areas(
        inlet_area=inlet["area"],
        throat_area=throat["area"],
        exit_area=exit_station["area"],
    )

    stations = ["Inlet", "Throat", "Exit"]
    areas = np.array(
        [
            plot_geometry["inlet_area"],
            plot_geometry["throat_area"],
            plot_geometry["exit_area"],
        ]
    )
    sizing = size_bell_nozzle_for_vehicle(
        inlet_area=plot_geometry["inlet_area"],
        throat_area=plot_geometry["throat_area"],
        exit_area=plot_geometry["exit_area"],
        converging_length=converging_length,
        diverging_length=diverging_length,
        missile_length=missile_length,
        nozzle_length_fraction=nozzle_length_fraction,
        nozzle_length=nozzle_length,
        vehicle_radius=vehicle_radius,
    )
    contour = generate_bell_contour(
        inlet_area=plot_geometry["inlet_area"],
        throat_area=plot_geometry["throat_area"],
        exit_area=plot_geometry["exit_area"],
        converging_length=sizing["converging_length"],
        diverging_length=sizing["diverging_length"],
        throat_angle_deg=bell_throat_angle,
        exit_angle_deg=bell_exit_angle,
        n_points=bell_points,
    )

    if geometry_csv_path is not None:
        save_contour_csv(contour, geometry_csv_path)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(contour["x"], contour["upper_wall"], color="black", linewidth=2)
    ax0.plot(contour["x"], contour["lower_wall"], color="black", linewidth=2)
    ax0.fill_between(contour["x"], contour["lower_wall"], contour["upper_wall"], color="steelblue", alpha=0.28)
    ax0.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="Throat")
    ax0.scatter(
        [-contour["converging_length"], 0.0, contour["diverging_length"]],
        [
            0.5 * contour["height"][0],
            0.5 * contour["height"][np.argmin(contour["area"])],
            0.5 * contour["height"][-1],
        ],
        color=["#4C78A8", "#F58518", "#54A24B"],
        zorder=5,
        label="pyCycle station heights/2",
    )
    ax0.set_title("Rectangular Nozzle Contour From pyCycle Areas", fontweight="bold")
    ax0.set_xlabel("Axial position (m)")
    ax0.set_ylabel("Half-height from centerline (m)")
    ax0.grid(True, alpha=0.25)
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(stations, [inlet["M"], throat["M"], exit_station["M"]], marker="o", linewidth=2.2)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    ax1.set_title("Mach Number", fontweight="bold")
    ax1.set_ylabel("Mach")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(stations, [inlet["P"] / 1e3, throat["P"] / 1e3, exit_station["P"] / 1e3], marker="o", linewidth=2.2)
    ax2.set_title("Static Pressure", fontweight="bold")
    ax2.set_ylabel("kPa")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(stations, [inlet["T"], throat["T"], exit_station["T"]], marker="o", linewidth=2.2)
    ax3.set_title("Static Temperature", fontweight="bold")
    ax3.set_ylabel("K")
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(stations, areas, color=["#4C78A8", "#F58518", "#54A24B"])
    ax4.set_title("Displayed Geometry Area", fontweight="bold")
    ax4.set_ylabel("m^2")
    ax4.grid(True, axis="y", alpha=0.25)

    ax5 = fig.add_subplot(gs[2, 1])
    thrust_values = [perf["Fg_ideal"], perf["Fg"], perf["F_cruise"], perf["Fg_vacuum"]]
    bars = ax5.bar(
        ["Ideal Fg", "pyCycle Fg", "Cruise F", "Vacuum Fg"],
        thrust_values,
        color=["#9ECAE9", "#E45756", "#F58518", "#54A24B"],
    )
    ax5.set_title("Gross Thrust", fontweight="bold")
    ax5.set_ylabel("N")
    ax5.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, thrust_values):
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    summary = [
        "pyCycle Performance",
        f"mdot  = {exit_station['W']:.4f} kg/s",
        f"Atraw = {throat['area']:.5f} m^2",
        f"Atrnd = {plot_geometry['throat_area']:.5f} m^2",
        f"Ae    = {exit_station['area']:.5f} m^2",
        f"Ae/At = {perf['area_ratio']:.4f}",
        f"Ltot  = {sizing['total_length']:.3f} m",
        f"Lbud  = {sizing['length_budget']:.3f} m",
        f"Ldiv  = {contour['diverging_length']:.3f} m",
        f"theta = {sizing['equivalent_divergent_half_angle_deg']:.1f} deg",
        f"Exp   = {perf['expansion_state']}",
        f"PR    = {perf['PR']:.4f}",
        f"Cfg_e = {perf['pycycle_Cfg_effective']:.4f}",
        f"Cf_p  = {perf['pycycle_thrust_coeff']:.4f}",
        f"Cf    = {perf['thrust_coeff']:.4f}",
        f"Cfv   = {perf['thrust_coeff_vacuum']:.4f}",
        f"Cv    = {perf['Cv']:.4f}",
        f"IspW  = {perf['Isp']:.2f} s",
        f"IspWv = {perf['Isp_vacuum']:.2f} s",
        f"c*    = {perf['c_star']:.2f} m/s",
    ]
    if perf.get("ramjet_metrics_available"):
        summary.extend(
            [
                "",
                "Liquid Ramjet",
                f"f     = {perf['fuel_air_ratio']:.5f}",
                f"mdotf = {perf['fuel_mass_flow']:.4f} kg/s",
                f"IspfG = {perf['fuel_Isp_gross']:.1f} s",
            ]
        )
        if np.isfinite(perf["fuel_Isp_net"]):
            summary.append(f"IspfN = {perf['fuel_Isp_net']:.1f} s")
    if plot_geometry["adjusted"]:
        summary.extend(["", "Display geometry adjusted"])
    ax6.text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F7F7F7", edgecolor="#BBBBBB"),
        transform=ax6.transAxes,
    )

    fig.suptitle("Nozzle Analysis From pyCycle", fontweight="bold", fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print("\nGeometry sizing")
    print("-" * 72)
    print(f"Missile length:           {sizing['missile_length']:.3f} m")
    if sizing["nozzle_length_overridden"]:
        print(
            f"Nozzle length budget:     {sizing['length_budget']:.3f} m "
            f"({100.0 * sizing['length_budget'] / sizing['missile_length']:.1f}% of missile length override)"
        )
    else:
        print(
            f"Nozzle length budget:     {sizing['length_budget']:.3f} m "
            f"({100.0 * sizing['nozzle_length_fraction']:.1f}% of missile length)"
        )
    print(f"Generated total length:   {sizing['total_length']:.3f} m")
    print(f"Converging length:        {sizing['converging_length']:.3f} m")
    print(f"Diverging length:         {sizing['diverging_length']:.3f} m")
    print(f"Unconstrained bell Ldiv:  {sizing['unconstrained_diverging_length']:.3f} m")
    print(f"Equivalent div. angle:    {sizing['equivalent_divergent_half_angle_deg']:.2f} deg")
    print(f"Exit radius:              {sizing['r_exit']:.4f} m")
    if plot_geometry["adjusted"]:
        print(f"Displayed inlet area:     {plot_geometry['inlet_area']:.6f} m^2")
        print(f"Displayed throat area:    {plot_geometry['throat_area']:.6f} m^2")
        print(f"Displayed exit area:      {plot_geometry['exit_area']:.6f} m^2")
    if vehicle_radius is not None:
        print(f"Vehicle radius reference: {vehicle_radius:.4f} m")
    for note in sizing["notes"]:
        print(f"NOTE: {note}")
    for note in plot_geometry["notes"]:
        print(f"NOTE: {note}")

    print(f"\nPlot saved to: {output_path}")
    if geometry_csv_path is not None:
        print(f"Bell contour saved to: {geometry_csv_path}")
    return fig


def resolve_case_from_args(args):
    gamma = args.gamma
    r_air = args.R
    flight_altitude_m = args.altitude * 0.3048 if args.altitude is not None else args.altitude_m
    manual_cruise_altitude_m = (
        args.cruise_altitude * 0.3048 if args.cruise_altitude is not None else args.cruise_altitude_m
    )
    vehicle = rectangular_vehicle_packaging(
        width=args.vehicle_width,
        height=args.vehicle_height,
        nozzle_diameter_clearance=args.nozzle_diameter_clearance,
        max_exit_area=args.max_exit_area,
    )

    if args.flight and args.manual:
        raise ValueError("Use either --flight or --manual, not both.")

    use_flight_mode = args.flight or not args.manual

    if use_flight_mode:
        flight = estimate_flight_inputs(
            mach_flight=args.M_flight,
            altitude_m=flight_altitude_m,
            vehicle_width=args.vehicle_width,
            vehicle_height=args.vehicle_height,
            inlet_width=args.inlet_width,
            inlet_height=args.inlet_height,
            nozzle_diameter_clearance=args.nozzle_diameter_clearance,
            max_exit_area=args.max_exit_area,
            capture_area_ratio=args.capture_area_ratio,
            combustor_exit_temperature=args.combustor_exit_temp,
            inlet_pressure_recovery=args.inlet_pressure_recovery,
            combustor_pressure_ratio=args.combustor_pressure_ratio,
            fuel_name=args.fuel,
            fuel_lhv=args.fuel_LHV,
            combustor_efficiency=args.combustor_efficiency,
            combustor_cp=args.combustor_cp,
            fuel_air_ratio=args.fuel_air_ratio,
            gamma=gamma,
            r_air=r_air,
        )
        print_flight_estimate(flight)

        return {
            "M_inlet": flight["M_inlet"],
            "Pt_inlet": flight["Pt_inlet"],
            "Tt_inlet": flight["Tt_inlet"],
            "P_exhaust": args.P_ambient if args.P_ambient is not None else flight["P_ambient"],
            "P_ambient": args.P_ambient if args.P_ambient is not None else flight["P_ambient"],
            "ambient_source": "user --P_ambient" if args.P_ambient is not None else "flight altitude atmosphere",
            "W": flight["W"],
            "air_mass_flow": flight["W_air"],
            "fuel_mass_flow": flight["W_fuel"],
            "fuel_air_ratio": flight["fuel_air_ratio"],
            "flight_velocity": flight["flight_velocity"],
            "A_throat": None,
            "target_exit_mach": args.M_exit,
            "gamma": gamma,
            "flight": flight,
            "vehicle": flight["vehicle"],
            "max_exit_area": flight["vehicle"]["max_exit_area"],
            "mode": "flight",
        }

    if args.P_ambient is not None:
        ambient_pressure = args.P_ambient
        ambient_source = "user --P_ambient"
    elif args.Ps is not None:
        ambient_pressure = args.Ps
        ambient_source = "user --Ps"
    else:
        ambient_pressure = standard_atmosphere_m(manual_cruise_altitude_m)["P"]
        ambient_source = f"standard atmosphere at {manual_cruise_altitude_m:.0f} m"

    return {
        "M_inlet": args.M_inlet,
        "Pt_inlet": args.Pt,
        "Tt_inlet": args.Tt,
        "P_exhaust": ambient_pressure,
        "P_ambient": ambient_pressure,
        "ambient_source": ambient_source,
        "W": args.W,
        "air_mass_flow": None,
        "fuel_mass_flow": args.fuel_flow,
        "fuel_air_ratio": args.fuel_air_ratio,
        "flight_velocity": args.flight_velocity,
        "A_throat": None if args.W is not None else args.A_throat,
        "target_exit_mach": args.M_exit,
        "gamma": gamma,
        "flight": None,
        "vehicle": vehicle,
        "max_exit_area": vehicle["max_exit_area"],
        "mode": "manual",
    }


def run_exit_mach_sweep(case, args):
    """
    Sweep target exit Mach numbers and rank by nozzle force at cruise ambient.
    """

    if not args.sweep_exit_mach:
        return None

    if args.sweep_exit_points < 2:
        raise ValueError("--sweep_exit_points must be at least 2.")
    if args.M_exit_max <= args.M_exit_min:
        raise ValueError("--M_exit_max must be greater than --M_exit_min.")

    mach_values = np.linspace(args.M_exit_min, args.M_exit_max, args.sweep_exit_points)
    rows = []
    ramjet_sweep = case.get("fuel_mass_flow") is not None and case.get("flight_velocity") is not None
    max_exit_area = None if args.no_exit_area_limit else case.get("max_exit_area")

    print("\n" + "=" * 72)
    print("EXIT MACH SWEEP")
    print("=" * 72)
    if ramjet_sweep:
        print("Objective: maximize estimated ramjet net thrust at cruise.")
    else:
        print("Objective: maximize nozzle force evaluated at cruise ambient pressure.")
    print("Note: each row is a pyCycle solve with Ps_exhaust balanced to the target Mach.")
    force_label = "F_net[N]" if ramjet_sweep else "F_cruise[N]"
    isp_label = "IspfN[s]" if ramjet_sweep else "IspW[s]"
    print(
        f"{'M_exit':>8}"
        f"{'Pe[kPa]':>11}"
        f"{'Ae/At':>10}"
        f"{force_label:>15}"
        f"{isp_label:>10}"
        f"{'Cf':>9}"
        f"{'State':>16}"
    )
    print("-" * 79)

    for mach in mach_values:
        sweep_case = case.copy()
        if case.get("flight") is not None:
            sweep_case["flight"] = case["flight"].copy()
        try:
            result = run_pycycle_nozzle(
                m_inlet=sweep_case["M_inlet"],
                pt_inlet=sweep_case["Pt_inlet"],
                tt_inlet=sweep_case["Tt_inlet"],
                ps_exhaust=sweep_case["P_ambient"],
                cv=args.Cv,
                nozzle_type=args.nozzle_type,
                mass_flow=sweep_case["W"],
                throat_area=sweep_case["A_throat"],
                target_exit_mach=mach,
                ambient_pressure=sweep_case["P_ambient"],
                gamma_for_guess=sweep_case["gamma"],
                show_solver_warnings=args.show_solver_warnings,
            )
            if max_exit_area is not None:
                result["performance"]["exit_area_limit"] = max_exit_area
                result["performance"]["exit_area_limited"] = result["exit"]["area"] > max_exit_area
                result["performance"]["area_constraint_mode"] = "sweep geometry filter"
                if result["exit"]["area"] > max_exit_area:
                    print(
                        f"{mach:>8.3f}"
                        f"{result['exit']['P']/1e3:>11.3f}"
                        f"{result['performance']['area_ratio']:>10.4f}"
                        f"{'REJECT':>15}"
                        f"{'':>10}"
                        f"{'':>9}"
                        f"{'Ae too large':>16}"
                    )
                    continue
            result = add_ramjet_metrics(result, sweep_case, args)
        except Exception as exc:
            print(f"{mach:>8.3f}{'FAILED':>71}  {exc}")
            continue

        perf = result["performance"]
        exit_station = result["exit"]
        rows.append(result)
        force_value = perf["ramjet_net_thrust"] if ramjet_sweep else perf["F_cruise"]
        isp_value = perf["fuel_Isp_net"] if ramjet_sweep else perf["Isp"]
        print(
            f"{mach:>8.3f}"
            f"{exit_station['P']/1e3:>11.3f}"
            f"{perf['area_ratio']:>10.4f}"
            f"{force_value:>15.2f}"
            f"{isp_value:>10.2f}"
            f"{perf['thrust_coeff']:>9.4f}"
            f"{perf['expansion_state']:>16}"
        )

    if not rows:
        print("No sweep cases completed.")
        return None

    first_perf = rows[0]["performance"]
    objective_key = (
        "ramjet_net_thrust"
        if first_perf.get("ramjet_metrics_available") and np.isfinite(first_perf.get("ramjet_net_thrust", np.nan))
        else "F_cruise"
    )
    best = max(rows, key=lambda item: item["performance"][objective_key])
    best_perf = best["performance"]
    best_exit = best["exit"]

    print("-" * 79)
    print(
        "Best sweep case: "
        f"M_exit={best_exit['M']:.3f}, "
        f"Ae/At={best_perf['area_ratio']:.4f}, "
        f"{objective_key}={best_perf[objective_key]:.2f} N, "
        f"total-flow Isp={best_perf['Isp']:.2f} s, "
        f"{best_perf['expansion_state']}"
    )

    return {"cases": rows, "best": best}


def main(args):
    case = resolve_case_from_args(args)

    print("\n" + "=" * 72)
    print("PYCYCLE NOZZLE INPUTS")
    print("=" * 72)
    print(f"Run mode:                 {case['mode']}")
    print(f"Nozzle type:              {args.nozzle_type}")
    print(f"Inlet Mach:               {case['M_inlet']:.4f}")
    print(f"Inlet total pressure:     {case['Pt_inlet']/1e3:.3f} kPa")
    print(f"Inlet total temperature:  {case['Tt_inlet']:.2f} K")
    print(f"Cruise ambient pressure:  {case['P_ambient']/1e3:.3f} kPa")
    print(f"Ambient pressure source:  {case['ambient_source']}")
    print(f"Vehicle width x height:   {case['vehicle']['width']:.3f} m x {case['vehicle']['height']:.3f} m")
    print(f"Vehicle frontal area:     {case['vehicle']['frontal_area']:.6f} m^2")
    print(f"Max circular exit diam.:  {case['vehicle']['max_circular_diameter']:.4f} m")
    if case.get("flight") is not None:
        inlet = case["flight"]["inlet"]
        print(f"Inlet width x height:     {inlet['width']:.3f} m x {inlet['height']:.3f} m")
        print(f"Inlet capture area:       {inlet['capture_area']:.6f} m^2")
    if case["target_exit_mach"] is None:
        print(f"Expansion assumption:     ideal expansion to cruise ambient")
        print(f"pyCycle exhaust pressure: {case['P_exhaust']/1e3:.3f} kPa")
    else:
        print("Expansion assumption:     off-design target exit Mach")
        print("pyCycle exhaust pressure: solved by pyCycle balance")
    print(f"Velocity coefficient Cv:  {args.Cv:.4f}")
    if args.no_exit_area_limit:
        print("Max exit area limit:      disabled")
    else:
        print(f"Max exit area limit:      {case['max_exit_area']:.6f} m^2")
    if args.nozzle_length is None:
        print(
            f"Nozzle length budget:     {args.missile_length * args.nozzle_length_fraction:.3f} m "
            f"({100.0 * args.nozzle_length_fraction:.1f}% of {args.missile_length:.3f} m)"
        )
    else:
        print(f"Nozzle length budget:     {args.nozzle_length:.3f} m")
    if case["target_exit_mach"] is not None:
        print(f"Target exit Mach balance: {case['target_exit_mach']:.4f}")
    if case["W"] is not None:
        print(f"Total nozzle mass input:  {case['W']:.4f} kg/s")
    if case.get("air_mass_flow") is not None:
        print(f"Air mass flow estimate:   {case['air_mass_flow']:.4f} kg/s")
    if case.get("fuel_mass_flow") is not None:
        print(f"Fuel mass flow estimate:  {case['fuel_mass_flow']:.4f} kg/s")
    if case.get("fuel_air_ratio") is not None:
        print(f"Fuel-air ratio:           {case['fuel_air_ratio']:.5f}")
    if case["A_throat"] is not None:
        print(f"Requested throat area:    {case['A_throat']:.6f} m^2")
        print("Mass flow will be scaled from pyCycle throat area.")

    results = run_pycycle_nozzle(
        m_inlet=case["M_inlet"],
        pt_inlet=case["Pt_inlet"],
        tt_inlet=case["Tt_inlet"],
        ps_exhaust=case["P_exhaust"],
        cv=args.Cv,
        nozzle_type=args.nozzle_type,
        mass_flow=case["W"],
        throat_area=case["A_throat"],
        target_exit_mach=case["target_exit_mach"],
        ambient_pressure=case["P_ambient"],
        gamma_for_guess=case["gamma"],
        show_solver_warnings=args.show_solver_warnings,
    )
    max_exit_area = None if args.no_exit_area_limit else case["max_exit_area"]
    if case["target_exit_mach"] is None:
        results = solve_exit_area_constrained_nozzle(
            initial_results=results,
            case=case,
            max_exit_area=max_exit_area,
            cv=args.Cv,
            nozzle_type=args.nozzle_type,
            show_solver_warnings=args.show_solver_warnings,
        )
    else:
        results = apply_exit_area_limit(results, case, max_exit_area)
    results = add_ramjet_metrics(results, case, args)

    print_results(results)
    sweep_results = run_exit_mach_sweep(case, args)
    geometry_csv_path = None if args.no_geometry_csv else args.geometry_csv
    fig = plot_results(
        results,
        output_path=args.plot_path,
        geometry_csv_path=geometry_csv_path,
        converging_length=args.converging_length,
        diverging_length=args.diverging_length,
        bell_throat_angle=args.bell_throat_angle,
        bell_exit_angle=args.bell_exit_angle,
        bell_points=args.bell_points,
        missile_length=args.missile_length,
        nozzle_length_fraction=args.nozzle_length_fraction,
        nozzle_length=args.nozzle_length,
        vehicle_radius=case["vehicle"]["max_circular_radius"],
    )

    backend = plt.get_backend().lower()
    if any(name in backend for name in ("agg", "pdf", "pgf", "ps", "svg", "template")):
        print(f"Display skipped ({plt.get_backend()} backend)")
    else:
        plt.show()

    return results, sweep_results, fig


def build_parser():
    parser = argparse.ArgumentParser(
        description="Nozzle design analysis using pyCycle FlowStart and Nozzle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default liquid-ramjet flight estimate
  python nozzle_design.py

  # Manual cold-nozzle check case
  python nozzle_design.py --manual --M_inlet 0.5 --Pt 200000 --Tt 320 --A_throat 0.01

  # Try several exit Mach values and rank them by cruise-ambient force
  python nozzle_design.py --sweep_exit_mach --M_exit_min 1.5 --M_exit_max 3.0 --sweep_exit_points 4

  # Override mission and rectangular packaging assumptions
  python nozzle_design.py --M_flight 5.0 --altitude_m 12000 --vehicle_width 0.40 --vehicle_height 0.57 --inlet_width 0.25 --inlet_height 0.05

  # Use an explicit exit-area cap instead of the rectangular-body circular-fit value
  python nozzle_design.py --max_exit_area 0.10

  # Manual mode with user-specified liquid fuel-air ratio for fuel-based Isp
  python nozzle_design.py --manual --W 10 --fuel_air_ratio 0.04 --flight_velocity 1200
        """,
    )

    parser.add_argument("--flight", action="store_true", help="Use flight-estimated liquid-ramjet inputs. This is the default.")
    parser.add_argument("--manual", action="store_true", help="Use manual nozzle inlet inputs instead of the flight estimate.")
    parser.add_argument("--M_inlet", type=float, default=0.5, help="Nozzle inlet Mach number.")
    parser.add_argument("--Pt", type=float, default=200000.0, help="Nozzle inlet total pressure (Pa).")
    parser.add_argument("--Tt", type=float, default=320.0, help="Nozzle inlet total temperature (K).")
    parser.add_argument(
        "--Ps",
        type=float,
        help="Manual-mode exhaust/back pressure override if --P_ambient is not supplied (Pa).",
    )
    parser.add_argument("--P_ambient", type=float, help="Exhaust/back pressure for the pyCycle nozzle (Pa).")
    parser.add_argument(
        "--cruise_altitude",
        type=float,
        help="Legacy manual-mode cruise altitude in feet. Overrides --cruise_altitude_m.",
    )
    parser.add_argument(
        "--cruise_altitude_m",
        type=float,
        default=DEFAULT_CRUISE_ALTITUDE_M,
        help="Manual-mode cruise altitude in meters when --P_ambient/--Ps are omitted.",
    )
    parser.add_argument("--W", type=float, help="Mass flow rate through the nozzle (kg/s).")
    parser.add_argument(
        "--A_throat",
        type=float,
        default=0.01,
        help="Requested throat area (m^2). Ignored if --W is supplied or --flight is used.",
    )
    parser.add_argument(
        "--max_exit_area",
        type=float,
        help=(
            "Maximum packaged nozzle exit area (m^2). If omitted, the code uses the largest "
            "circular exit that fits inside --vehicle_width x --vehicle_height."
        ),
    )
    parser.add_argument("--no_exit_area_limit", action="store_true", help="Disable the packaged exit-area limit.")
    parser.add_argument(
        "--M_exit",
        type=float,
        help=(
            "Optional off-design target exit Mach. If omitted, pyCycle assumes ideal expansion "
            "to the cruise ambient/back pressure."
        ),
    )
    parser.add_argument("--Cv", type=float, default=1.0, help="pyCycle nozzle velocity coefficient.")
    parser.add_argument(
        "--nozzle_type",
        choices=("CD", "CV", "CD_CV"),
        default="CD",
        help="pyCycle nozzle type.",
    )

    parser.add_argument("--M_flight", type=float, default=DEFAULT_CRUISE_MACH, help="Flight Mach number for --flight mode.")
    parser.add_argument("--altitude", type=float, help="Legacy flight altitude in feet. Overrides --altitude_m.")
    parser.add_argument(
        "--altitude_m",
        type=float,
        default=DEFAULT_CRUISE_ALTITUDE_M,
        help="Flight altitude in meters for --flight mode.",
    )
    parser.add_argument(
        "--flight_velocity",
        type=float,
        help="Manual-mode flight velocity for ram-drag/net-ramjet-thrust estimates (m/s).",
    )
    parser.add_argument(
        "--vehicle_width",
        type=float,
        default=DEFAULT_VEHICLE_WIDTH,
        help="Rectangular vehicle body width available for nozzle packaging (m).",
    )
    parser.add_argument(
        "--vehicle_height",
        type=float,
        default=DEFAULT_VEHICLE_HEIGHT,
        help="Rectangular vehicle body height available for nozzle packaging (m).",
    )
    parser.add_argument(
        "--nozzle_diameter_clearance",
        type=float,
        default=DEFAULT_NOZZLE_DIAMETER_CLEARANCE,
        help="Fraction of the smaller rectangular body dimension allowed for the circular nozzle exit.",
    )
    parser.add_argument(
        "--inlet_width",
        type=float,
        default=DEFAULT_INLET_WIDTH,
        help="Rectangular inlet opening width used for air mass-flow estimate (m).",
    )
    parser.add_argument(
        "--inlet_height",
        type=float,
        default=DEFAULT_INLET_HEIGHT,
        help="Rectangular inlet opening height used for air mass-flow estimate (m).",
    )
    parser.add_argument(
        "--missile_length",
        type=float,
        default=DEFAULT_MISSILE_LENGTH,
        help="Total missile length used for nozzle geometry packaging checks (m).",
    )
    parser.add_argument(
        "--nozzle_length_fraction",
        type=float,
        default=DEFAULT_NOZZLE_LENGTH_FRACTION,
        help="Fraction of missile length allocated to the nozzle geometry when --nozzle_length is omitted.",
    )
    parser.add_argument(
        "--nozzle_length",
        type=float,
        help="Total generated nozzle geometry length override (m).",
    )
    parser.add_argument(
        "--capture_area_ratio",
        type=float,
        default=1.0,
        help="Captured fraction of the rectangular inlet opening area.",
    )
    parser.add_argument(
        "--combustor_exit_temp",
        type=float,
        default=DEFAULT_COMBUSTOR_EXIT_TT,
        help="Combustor-exit/nozzle-inlet total temperature estimate for flight mode (K).",
    )
    parser.add_argument(
        "--inlet_pressure_recovery",
        type=float,
        default=DEFAULT_INLET_PRESSURE_RECOVERY,
        help="Total-pressure recovery from freestream to combustor inlet for flight-mode estimates.",
    )
    parser.add_argument(
        "--combustor_pressure_ratio",
        type=float,
        default=DEFAULT_COMBUSTOR_PRESSURE_RATIO,
        help="Combustor total-pressure ratio from combustor inlet to nozzle inlet.",
    )
    parser.add_argument(
        "--fuel",
        default=DEFAULT_FUEL_NAME,
        help="Fuel label used in output.",
    )
    parser.add_argument(
        "--fuel_air_ratio",
        type=float,
        help="Liquid fuel-air ratio. In --flight mode this overrides the heat-balance fuel estimate.",
    )
    parser.add_argument(
        "--fuel_flow",
        type=float,
        help="Manual-mode liquid fuel flow rate (kg/s) for fuel-based Isp.",
    )
    parser.add_argument("--fuel_LHV", type=float, default=DEFAULT_FUEL_LHV, help="Liquid fuel lower heating value (J/kg).")
    parser.add_argument(
        "--combustor_efficiency",
        type=float,
        default=0.95,
        help="Combustor efficiency used for flight-mode fuel-flow estimate.",
    )
    parser.add_argument(
        "--combustor_cp",
        type=float,
        default=1200.0,
        help="Approximate hot-gas cp used for flight-mode fuel-flow estimate (J/kg/K).",
    )
    parser.add_argument("--gamma", type=float, default=1.4, help="Gamma used for flight/input estimates only.")
    parser.add_argument("--R", type=float, default=287.0, help="Gas constant used for flight/input estimates only.")
    parser.add_argument(
        "--sweep_exit_mach",
        action="store_true",
        help="Sweep target exit Mach values and rank by nozzle force at cruise ambient pressure.",
    )
    parser.add_argument("--M_exit_min", type=float, default=1.2, help="Minimum target exit Mach for sweep mode.")
    parser.add_argument("--M_exit_max", type=float, default=3.0, help="Maximum target exit Mach for sweep mode.")
    parser.add_argument("--sweep_exit_points", type=int, default=4, help="Number of target Mach values in sweep mode.")
    parser.add_argument(
        "--converging_length",
        type=float,
        default=0.35,
        help="Converging-section length used for generated bell contour plot/CSV (m).",
    )
    parser.add_argument(
        "--diverging_length",
        type=float,
        help="Diverging-section length for generated bell contour (m). If omitted, it is estimated from radii.",
    )
    parser.add_argument(
        "--bell_throat_angle",
        type=float,
        default=30.0,
        help="Initial diverging wall angle for the generated bell contour (deg).",
    )
    parser.add_argument(
        "--bell_exit_angle",
        type=float,
        default=8.0,
        help="Exit wall angle for the generated bell contour (deg).",
    )
    parser.add_argument(
        "--bell_points",
        type=int,
        default=240,
        help="Number of points used for generated bell contour plot/CSV.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Output path for the nozzle analysis plot.",
    )
    parser.add_argument(
        "--geometry_csv",
        type=Path,
        default=DEFAULT_GEOMETRY_PATH,
        help="CSV path for generated bell contour geometry.",
    )
    parser.add_argument("--no_geometry_csv", action="store_true", help="Do not write the generated contour CSV.")
    parser.add_argument(
        "--show_solver_warnings",
        action="store_true",
        help="Show pyCycle/OpenMDAO solver warnings from intermediate nonlinear iterations.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
