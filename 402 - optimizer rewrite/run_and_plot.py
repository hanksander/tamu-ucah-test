from __future__ import annotations
import os
import json
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dymos as dm

from trajectory_opt.optimize import run
from trajectory_opt.engine_adapter import PyCycleRamAdapter
from trajectory_opt.perf_surrogate import PerfTable
from trajectory_opt.trajectory_problem import build_inner_problem, extract_metrics
from powered_flight_trajectory_code_v4 import atmosphere, M0 as M0_MASS
from combined_cycle_liquid_ram_scram.pyc_config import M4_MAX, TT4_MAX_K, Q_MAX_PA

OUT = "plots_opt"
PHASES = ("cruise",)
PHASE_COLORS = {"cruise": "#d62728"}


def ts(p, phase, name):
    """Fetch timeseries value, handling old/new Dymos naming."""
    for path in (f"traj.{phase}.timeseries.{name}",
                 f"traj.{phase}.timeseries.states:{name}",
                 f"traj.{phase}.timeseries.controls:{name}"):
        try:
            return np.asarray(p.get_val(path)).ravel()
        except Exception:
            continue
    raise KeyError(f"no timeseries path found for {phase}.{name}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)


def plot_vs(data, x, y, xlabel, ylabel, title, fname, scale_x=1.0, scale_y=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ph in PHASES:
        ax.plot(data[ph][x] * scale_x, data[ph][y] * scale_y,
                label=ph, color=PHASE_COLORS[ph], lw=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, fname)


def main(maxiter: int = 1):
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print(" OUTER OPTIMIZER")
    print("=" * 60)
    design, res = run(maxiter=maxiter)
    range_km_opt = -float(res.fun)
    print(f"\n-> optimal range: {range_km_opt:.2f} km")
    print(f"-> optimal geometry:\n{design}\n")

    with open(os.path.join(OUT, "optimal_design.json"), "w") as f:
        json.dump({**{k: float(v) for k, v in asdict(design).items()
                      if isinstance(v, (int, float))},
                   "range_km": range_km_opt}, f, indent=2)

    print("=" * 60)
    print(" INNER SOLVE WITH OPTIMAL DESIGN (simulate=True)")
    print("=" * 60)
    table = PerfTable(design, PyCycleRamAdapter()).build()
    # LFRJ cruise entry: Mach 4 at 20 km (solid rocket delivers these ICs externally)
    h0 = 20_000.0
    V0 = 4.0 * atmosphere(h0)[3]
    p = build_inner_problem(table, h0, V0, phi_init=0.8)
    dm.run_problem(p, run_driver=True, simulate=False)
    metrics = extract_metrics(p)
    print(f"\n-> final range: {metrics['range_m']/1000:.2f} km")
    print(f"-> fuel burned: {metrics['fuel_kg']:.2f} kg\n")

    data = {}
    for ph in PHASES:
        d = {"t": ts(p, ph, "time")}
        for s in ("x_range", "h", "V", "gamma", "m"):
            d[s] = ts(p, ph, s)
        d["alpha"] = ts(p, ph, "alpha")
        d["q"]     = ts(p, ph, "q")
        a_local = np.array([atmosphere(hh)[3] for hh in d["h"]])
        d["Mach"] = d["V"] / a_local
        data[ph] = d
    data["cruise"]["M4"]           = ts(p, "cruise", "M4")
    data["cruise"]["Tt4"]          = ts(p, "cruise", "Tt4")
    data["cruise"]["unstart_flag"] = ts(p, "cruise", "unstart_flag")

    plot_vs(data, "t", "h",        "t [s]", "altitude [m]",      "Altitude vs Time",          "01_altitude_vs_time.png")
    plot_vs(data, "x_range", "h",  "range [km]", "altitude [km]", f"Altitude vs Range (R = {range_km_opt:.1f} km)",
            "02_altitude_vs_range.png", scale_x=1e-3, scale_y=1e-3)
    plot_vs(data, "t", "Mach",     "t [s]", "Mach",              "Mach vs Time",              "03_mach_vs_time.png")
    plot_vs(data, "t", "V",        "t [s]", "V [m/s]",           "Velocity vs Time",          "04_velocity_vs_time.png")
    plot_vs(data, "t", "m",        "t [s]", "mass [kg]",         "Mass vs Time",              "05_mass_vs_time.png")
    plot_vs(data, "t", "gamma",    "t [s]", "γ [rad]",           "Flight Path Angle vs Time", "06_gamma_vs_time.png")
    plot_vs(data, "t", "alpha",    "t [s]", "α [deg]",           "Angle of Attack vs Time",   "07_alpha_vs_time.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for ph in PHASES:
        ax.plot(data[ph]["t"], data[ph]["q"], label=ph, color=PHASE_COLORS[ph], lw=2)
    ax.axhline(Q_MAX_PA, color="k", ls="--", lw=1, label=f"q_max = {Q_MAX_PA/1e3:.0f} kPa")
    ax.set_xlabel("t [s]"); ax.set_ylabel("q [Pa]"); ax.set_title("Dynamic Pressure vs Time")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "08_q_vs_time.png")

    tc = data["cruise"]["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(tc, data["cruise"]["M4"], color=PHASE_COLORS["cruise"], lw=2)
    axes[0].axhline(M4_MAX, color="k", ls="--", lw=1, label=f"limit = {M4_MAX}")
    axes[0].set_ylabel("M4"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(tc, data["cruise"]["Tt4"], color=PHASE_COLORS["cruise"], lw=2)
    axes[1].axhline(TT4_MAX_K, color="k", ls="--", lw=1, label=f"limit = {TT4_MAX_K} K")
    axes[1].set_ylabel("Tt4 [K]"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(tc, data["cruise"]["unstart_flag"], color=PHASE_COLORS["cruise"], lw=2)
    axes[2].axhline(0.0, color="k", ls="--", lw=1)
    axes[2].set_ylabel("unstart flag"); axes[2].set_xlabel("t [s]"); axes[2].grid(True, alpha=0.3)
    fig.suptitle("Cruise Path-Constraint Observables")
    save(fig, "09_cruise_path_constraints.png")

    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(f"range_km     : {range_km_opt:.3f}\n")
        f.write(f"fuel_burn_kg : {metrics['fuel_kg']:.3f}\n")
        f.write(f"launch_mass  : {M0_MASS:.1f} kg\n\n")
        f.write("optimal design:\n")
        for k, v in asdict(design).items():
            f.write(f"  {k:<22s} = {v}\n")

    print(f"\n-> wrote {len(os.listdir(OUT))} files to ./{OUT}/")


if __name__ == "__main__":
    main()
