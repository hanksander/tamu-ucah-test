"""
Nozzle design and performance analysis using pyCycle.

The active nozzle analysis path is:
    OpenMDAO Problem -> pyCycle Cycle -> FlowStart -> Nozzle

The remaining standalone equations are only used to estimate optional flight-mode
inputs before the pyCycle nozzle run. No ASME/isentropic nozzle solver is used
for the reported nozzle station properties or performance metrics.
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
from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_PLOT_PATH = OUTPUT_DIR / "nozzle_analysis.png"
DEFAULT_GEOMETRY_PATH = OUTPUT_DIR / "nozzle_geometry.csv"
G0 = 9.80665
LBM_PER_KG = 2.2046226218
LBF_PER_N = 0.2248089431


def scalar(value):
    """Return a scalar float from OpenMDAO's array-shaped values."""

    return float(np.asarray(value).ravel()[0])


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
    altitude_ft,
    max_radius_inches,
    capture_area_ratio,
    combustor_temp_ratio,
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

    atm = standard_atmosphere(altitude_ft)
    p_total_ratio, t_total_ratio = isentropic_total_ratios(mach_flight, gamma)
    pt_freestream = atm["P"] * p_total_ratio
    tt_freestream = atm["T"] * t_total_ratio
    shock = normal_shock_relations(mach_flight, gamma)

    pt_after_inlet = pt_freestream * shock["Pt2_Pt1"]
    tt_after_inlet = tt_freestream
    pt_nozzle_inlet = pt_after_inlet * 0.95
    tt_nozzle_inlet = tt_after_inlet * combustor_temp_ratio

    if fuel_air_ratio is None:
        fuel_air_ratio = estimate_fuel_air_ratio(
            tt_before_combustor=tt_after_inlet,
            tt_after_combustor=tt_nozzle_inlet,
            fuel_lhv=fuel_lhv,
            combustor_efficiency=combustor_efficiency,
            cp_gas=combustor_cp,
        )

    max_radius_m = max_radius_inches * 0.0254
    vehicle_area = np.pi * max_radius_m**2
    capture_area = vehicle_area * capture_area_ratio
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
        "vehicle_area": vehicle_area,
        "max_radius_m": max_radius_m,
        "flight_velocity": flight_velocity,
        "fuel_lhv": fuel_lhv,
        "combustor_efficiency": combustor_efficiency,
        "combustor_cp": combustor_cp,
        "r_air": r_air,
        "gamma": gamma,
        "combustor_temp_ratio": combustor_temp_ratio,
    }


def build_pycycle_problem(
    m_inlet,
    pt_inlet,
    tt_inlet,
    mass_flow,
    ps_exhaust,
    cv,
    nozzle_type,
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

    cycle.add_subsystem("flow_start", FlowStart())
    cycle.add_subsystem(
        "nozzle",
        Nozzle(nozzType=nozzle_type, lossCoef="Cv", internal_solver=True),
    )
    cycle.pyc_connect_flow("flow_start.Fl_O", "nozzle.Fl_I")

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

    inlet = station_from_prob(prob, "flow_start.Fl_O")
    throat = station_from_prob(prob, "nozzle.Throat")
    exit_station = station_from_prob(prob, "nozzle.Fl_O")

    inlet["Pt"] = get_value(prob, "flow_start.Fl_O:tot:P", "Pa")
    inlet["Tt"] = get_value(prob, "flow_start.Fl_O:tot:T", "K")
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


def scale_results_to_throat_area(results, throat_area):
    """
    Scale pyCycle's mass-flow-dependent outputs to a requested throat area.

    For a fixed thermodynamic state and pressure ratio, pyCycle station areas,
    mass flow, and thrust scale linearly together. This avoids rerunning the
    nonlinear model just to resize the nozzle.
    """

    current_area = results["throat"]["area"]
    if current_area <= 0.0:
        raise ValueError("Cannot scale results because pyCycle returned a nonpositive throat area.")

    scale = throat_area / current_area
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
        requested_throat_area=throat_area,
    )
    return results


def run_pycycle_nozzle(
    m_inlet,
    pt_inlet,
    tt_inlet,
    ps_exhaust,
    cv,
    nozzle_type,
    mass_flow=None,
    throat_area=None,
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

    print("\n" + "=" * 72)
    print("FLIGHT-MODE INPUT ESTIMATE")
    print("=" * 72)
    print(f"Altitude:                 {atm['altitude_ft']:.0f} ft")
    print(f"Atmospheric pressure:     {atm['P']/1e3:.3f} kPa")
    print(f"Atmospheric temperature:  {atm['T']:.2f} K")
    print(f"Atmospheric density:      {atm['rho']:.6f} kg/m^3")
    print(f"Flight velocity:          {flight['flight_velocity']:.2f} m/s")
    print(f"Normal-shock M2:          {shock['M2']:.3f}")
    print(f"Total pressure recovery:  {shock['Pt2_Pt1']:.3f}")
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
    converging_length=0.35,
    diverging_length=None,
    throat_angle_deg=30.0,
    exit_angle_deg=8.0,
    n_points=240,
):
    """
    Build a smooth axisymmetric nozzle contour from pyCycle station areas.

    pyCycle gives station areas, not wall shape. This function creates a
    reasonable bell-style profile for visualization and preliminary geometry.
    """

    r_inlet = np.sqrt(max(inlet_area, 1.0e-12) / np.pi)
    r_throat = np.sqrt(max(throat_area, 1.0e-12) / np.pi)
    r_exit = np.sqrt(max(exit_area, 1.0e-12) / np.pi)

    if converging_length <= 0.0:
        raise ValueError("converging_length must be positive.")

    throat_angle = np.radians(throat_angle_deg)
    exit_angle = np.radians(exit_angle_deg)

    if diverging_length is None:
        radius_change = max(r_exit - r_throat, 1.0e-6)
        reference_angle = np.radians(15.0)
        # Roughly a 75%-length bell compared with a 15-degree conical nozzle.
        diverging_length = max(0.50, 0.75 * radius_change / np.tan(reference_angle))
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

    return {
        "x": x,
        "radius": r,
        "area": np.pi * r**2,
        "r_inlet": r_inlet,
        "r_throat": r_throat,
        "r_exit": r_exit,
        "converging_length": converging_length,
        "diverging_length": diverging_length,
        "throat_angle_deg": throat_angle_deg,
        "exit_angle_deg": exit_angle_deg,
    }


def save_contour_csv(contour, output_path):
    """
    Save the generated nozzle contour for CAD/sketch import.
    """

    data = np.column_stack((contour["x"], contour["radius"], -contour["radius"], contour["area"]))
    header = "x_m,radius_upper_m,radius_lower_m,area_m2"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def plot_results(
    results,
    output_path=DEFAULT_PLOT_PATH,
    geometry_csv_path=None,
    converging_length=0.35,
    diverging_length=None,
    bell_throat_angle=30.0,
    bell_exit_angle=8.0,
):
    inlet = results["inlet"]
    throat = results["throat"]
    exit_station = results["exit"]
    perf = results["performance"]

    stations = ["Inlet", "Throat", "Exit"]
    areas = np.array([inlet["area"], throat["area"], exit_station["area"]])
    contour = generate_bell_contour(
        inlet_area=inlet["area"],
        throat_area=throat["area"],
        exit_area=exit_station["area"],
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=bell_throat_angle,
        exit_angle_deg=bell_exit_angle,
    )

    if geometry_csv_path is not None:
        save_contour_csv(contour, geometry_csv_path)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(contour["x"], contour["radius"], color="black", linewidth=2)
    ax0.plot(contour["x"], -contour["radius"], color="black", linewidth=2)
    ax0.fill_between(contour["x"], -contour["radius"], contour["radius"], color="steelblue", alpha=0.28)
    ax0.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="Throat")
    ax0.scatter(
        [-converging_length, 0.0, contour["diverging_length"]],
        [contour["r_inlet"], contour["r_throat"], contour["r_exit"]],
        color=["#4C78A8", "#F58518", "#54A24B"],
        zorder=5,
        label="pyCycle station radii",
    )
    ax0.set_title("Bell Nozzle Contour From pyCycle Areas", fontweight="bold")
    ax0.set_xlabel("Axial position (m)")
    ax0.set_ylabel("Equivalent radius (m)")
    ax0.grid(True, alpha=0.25)
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
    ax4.set_title("Flow Area", fontweight="bold")
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
        f"At    = {throat['area']:.5f} m^2",
        f"Ae    = {exit_station['area']:.5f} m^2",
        f"Ae/At = {perf['area_ratio']:.4f}",
        f"Ldiv  = {contour['diverging_length']:.3f} m",
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
    print(f"\nPlot saved to: {output_path}")
    if geometry_csv_path is not None:
        print(f"Bell contour saved to: {geometry_csv_path}")
    return fig


def resolve_case_from_args(args):
    gamma = args.gamma
    r_air = args.R

    if args.flight and args.manual:
        raise ValueError("Use either --flight or --manual, not both.")

    use_flight_mode = args.flight or not args.manual

    if use_flight_mode:
        flight = estimate_flight_inputs(
            mach_flight=args.M_flight,
            altitude_ft=args.altitude,
            max_radius_inches=args.max_radius,
            capture_area_ratio=args.capture_area_ratio,
            combustor_temp_ratio=args.combustor_temp_ratio,
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
            "mode": "flight",
        }

    if args.P_ambient is not None:
        ambient_pressure = args.P_ambient
        ambient_source = "user --P_ambient"
    elif args.Ps is not None:
        ambient_pressure = args.Ps
        ambient_source = "user --Ps"
    else:
        ambient_pressure = standard_atmosphere(args.cruise_altitude)["P"]
        ambient_source = f"standard atmosphere at {args.cruise_altitude:.0f} ft"

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
        try:
            result = run_pycycle_nozzle(
                m_inlet=case["M_inlet"],
                pt_inlet=case["Pt_inlet"],
                tt_inlet=case["Tt_inlet"],
                ps_exhaust=case["P_ambient"],
                cv=args.Cv,
                nozzle_type=args.nozzle_type,
                mass_flow=case["W"],
                throat_area=case["A_throat"],
                target_exit_mach=mach,
                ambient_pressure=case["P_ambient"],
                gamma_for_guess=case["gamma"],
                show_solver_warnings=args.show_solver_warnings,
            )
            result = add_ramjet_metrics(result, case, args)
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
    if case["target_exit_mach"] is None:
        print(f"Expansion assumption:     ideal expansion to cruise ambient")
        print(f"pyCycle exhaust pressure: {case['P_exhaust']/1e3:.3f} kPa")
    else:
        print("Expansion assumption:     off-design target exit Mach")
        print("pyCycle exhaust pressure: solved by pyCycle balance")
    print(f"Velocity coefficient Cv:  {args.Cv:.4f}")
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
    results = add_ramjet_metrics(results, case, args)

    print_results(results)
    sweep_results = run_exit_mach_sweep(case, args)
    geometry_csv_path = None if args.no_geometry_csv else args.geometry_csv
    fig = plot_results(
        results,
        geometry_csv_path=geometry_csv_path,
        converging_length=args.converging_length,
        diverging_length=args.diverging_length,
        bell_throat_angle=args.bell_throat_angle,
        bell_exit_angle=args.bell_exit_angle,
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

  # Override flight-estimated inlet conditions
  python nozzle_design.py --M_flight 5.0 --altitude 60000 --max_radius 18.9

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
        default=60000.0,
        help="Manual-mode cruise altitude used to estimate ambient pressure when --P_ambient/--Ps are omitted (ft).",
    )
    parser.add_argument("--W", type=float, help="Mass flow rate through the nozzle (kg/s).")
    parser.add_argument(
        "--A_throat",
        type=float,
        default=0.01,
        help="Requested throat area (m^2). Ignored if --W is supplied or --flight is used.",
    )
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

    parser.add_argument("--M_flight", type=float, default=5.0, help="Flight Mach number for --flight mode.")
    parser.add_argument("--altitude", type=float, default=60000.0, help="Flight altitude in feet for --flight mode.")
    parser.add_argument(
        "--flight_velocity",
        type=float,
        help="Manual-mode flight velocity for ram-drag/net-ramjet-thrust estimates (m/s).",
    )
    parser.add_argument("--max_radius", type=float, default=18.9, help="Maximum vehicle radius in inches.")
    parser.add_argument("--capture_area_ratio", type=float, default=0.8, help="Captured fraction of vehicle frontal area.")
    parser.add_argument(
        "--combustor_temp_ratio",
        type=float,
        default=3.5,
        help="Estimated combustor Tt ratio for flight-mode nozzle-inlet inputs.",
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
    parser.add_argument("--fuel_LHV", type=float, default=43.0e6, help="Liquid fuel lower heating value (J/kg).")
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
