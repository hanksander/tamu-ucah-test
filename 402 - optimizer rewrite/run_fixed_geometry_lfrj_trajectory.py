from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from powered_flight_trajectory_code_v4 import (
    A_BURN_0,
    G0,
    M0,
    M_FUEL_LFRJ,
    M_PROP_BOOST,
    RE,
    S_REF,
    aero,
    atmosphere,
    boost_isp,
    boost_thrust,
    cl_alpha,
)
from trajectory_opt.engine_adapter import PyCycleRamAdapter
from trajectory_opt.fixed_lfrj import (
    BOOST_HANDOFF_MACH,
    CRUISE_ALT_M,
    CRUISE_MACH,
    build_fixed_perf_table,
    fixed_lfrj_design,
)


OUTDIR = Path("fixed_lfrj_trajectory_outputs")
LFRJ_DRY_MASS = M0 - M_PROP_BOOST - M_FUEL_LFRJ
CRUISE_DRAG_N = 4_000.0
MACH_THRUST_GAIN_N_PER_MACH = 35_000.0


def _mach(h_m: float, V: float) -> float:
    return float(V / atmosphere(h_m)[3])


def _trim_alpha(M: float, h_m: float, V: float, m_kg: float) -> float:
    rho, _, _, _ = atmosphere(h_m)
    q = 0.5 * rho * V * V
    cla = float(cl_alpha(M))
    W = m_kg * G0 * (RE / (RE + h_m)) ** 2
    alpha = W / (q * S_REF * cla) if q > 1.0 and cla > 1e-9 else 0.0
    return float(np.clip(alpha, -5.0, 35.0))


def _alpha_boost(t: float, s: np.ndarray) -> float:
    del t
    _, h, V, gamma, m = s
    M = _mach(max(h, 10.0), max(V, 10.0))
    h_err = CRUISE_ALT_M - h
    gamma_cmd = np.clip(0.18 + h_err / 85_000.0 - 0.03 * max(M - 2.0, 0.0), 0.02, 0.32)
    alpha_cmd = 5.0 + 38.0 * (gamma_cmd - gamma)
    return float(np.clip(alpha_cmd, -2.0, 14.0))


def _alpha_cruise(t: float, s: np.ndarray) -> float:
    del t
    _, h, V, _, m = s
    base_alpha = _trim_alpha(_mach(max(h, 10.0), max(V, 10.0)), h, V, m)
    gamma_deg = np.rad2deg(s[3])
    alpha_cmd = 0.78 * base_alpha - 2.0 * gamma_deg - 0.010 * (h - CRUISE_ALT_M)
    return float(np.clip(alpha_cmd, -5.0, 35.0))


def _alpha_descent(t: float, s: np.ndarray) -> float:
    del t
    gamma_target = np.deg2rad(-70.0)
    err_deg = np.rad2deg(gamma_target - s[3])
    return float(np.clip(2.2 * err_deg, -30.0, 8.0))


def _select_cruise_phi(M: float, h_m: float, table, drag_N: float) -> tuple[float, object]:
    """Choose phi so thrust roughly matches drag while holding Mach 4.8."""
    mach_target = CRUISE_MACH + 0.03 if M < CRUISE_MACH else CRUISE_MACH
    desired_thrust = drag_N + MACH_THRUST_GAIN_N_PER_MACH * (mach_target - M)
    if desired_thrust <= 0.0:
        return 0.0, None

    phis = np.asarray(table.GRID_PHI, dtype=float)
    thrusts = np.array([table.lookup(M, h_m, phi).thrust_N for phi in phis])

    order = np.argsort(thrusts)
    thrust_sorted = thrusts[order]
    phi_sorted = phis[order]

    if desired_thrust <= thrust_sorted[0]:
        if M >= CRUISE_MACH:
            return 0.0, None
        phi = float(phi_sorted[0])
    elif desired_thrust >= thrust_sorted[-1]:
        phi = float(phi_sorted[-1])
    else:
        phi = float(np.interp(desired_thrust, thrust_sorted, phi_sorted))

    return phi, table.lookup(M, h_m, phi)


def _forces(s: np.ndarray, alpha_deg: float, phase: str, table=None) -> dict:
    _, h, V, _, m = s
    h = float(np.clip(h, 10.0, 79_900.0))
    V = float(np.clip(V, 20.0, 5_000.0))
    M = _mach(h, V)
    rho, _, _, _ = atmosphere(h)
    CL, CD, L, D, q = aero(M, alpha_deg, rho, V)

    if phase == "boost":
        T = boost_thrust(M, h)
        mdot = T / (boost_isp() * G0)
        phi = 0.0
        mdot_f = 0.0
        M4 = 0.0
        Tt4 = 0.0
        unstart = 0.0
    elif phase == "cruise":
        D = CRUISE_DRAG_N
        CD = D / (q * S_REF) if q > 1.0 else 0.0
        phi, perf = _select_cruise_phi(M, h, table, D)
        if perf is None:
            T = 0.0
            mdot_f = 0.0
            mdot = 0.0
            M4 = 0.0
            Tt4 = 0.0
            unstart = 0.0
        else:
            T = perf.thrust_N
            mdot_f = perf.mdot_fuel_kgs
            mdot = mdot_f
            M4 = perf.M4
            Tt4 = perf.Tt4_K
            unstart = perf.unstart_flag
    else:
        T = 0.0
        mdot = 0.0
        phi = 0.0
        mdot_f = 0.0
        M4 = 0.0
        Tt4 = 0.0
        unstart = 0.0

    return {
        "M": M,
        "rho": rho,
        "CL": float(CL),
        "CD": float(CD),
        "L": float(L),
        "D": float(D),
        "q": float(q),
        "T": float(T),
        "mdot": float(mdot),
        "mdot_f": float(mdot_f),
        "phi": float(phi),
        "M4": float(M4),
        "Tt4": float(Tt4),
        "unstart": float(unstart),
    }


def _eom(s: np.ndarray, alpha_deg: float, phase: str, table=None) -> np.ndarray:
    _, h, V, gamma, m = s
    f = _forces(s, alpha_deg, phase, table)
    g = G0 * (RE / (RE + max(h, 1.0))) ** 2
    ar = np.deg2rad(alpha_deg)
    dx = V * np.cos(gamma)
    dh = V * np.sin(gamma)
    dV = (f["T"] * np.cos(ar) - f["D"]) / m - g * np.sin(gamma)
    dgamma = ((f["T"] * np.sin(ar) + f["L"]) / (m * max(V, 1.0))
              - (g / max(V, 1.0) - V / (RE + max(h, 1.0))) * np.cos(gamma))
    dm = -f["mdot"] if phase in ("boost", "cruise") else 0.0
    return np.array([dx, dh, dV, dgamma, dm], dtype=float)


def _record(rows: list[dict], t: float, s: np.ndarray, phase: str, alpha: float, table=None):
    f = _forces(s, alpha, phase, table)
    rows.append({
        "t_s": float(t),
        "phase": phase,
        "x_m": float(s[0]),
        "h_m": float(s[1]),
        "V_mps": float(s[2]),
        "gamma_rad": float(s[3]),
        "m_kg": float(s[4]),
        "Mach": f["M"],
        "alpha_deg": float(alpha),
        "phi": f["phi"],
        "q_Pa": f["q"],
        "CL": f["CL"],
        "CD": f["CD"],
        "L_N": f["L"],
        "D_N": f["D"],
        "T_N": f["T"],
        "mdot_fuel_kgs": f["mdot_f"],
        "M4": f["M4"],
        "Tt4_K": f["Tt4"],
        "unstart_flag": f["unstart"],
    })


def simulate_phase(
    s0: np.ndarray,
    phase: str,
    t0: float,
    duration_s: float,
    dt_s: float,
    alpha_fn,
    rows: list[dict],
    table=None,
    stop_fn=None,
) -> tuple[np.ndarray, float]:
    s = np.array(s0, dtype=float)
    t = float(t0)
    tend = t0 + duration_s
    _record(rows, t, s, phase, alpha_fn(t, s), table)
    while t < tend:
        if stop_fn is not None and stop_fn(s):
            break
        dt = min(dt_s, tend - t)
        a1 = alpha_fn(t, s)
        k1 = _eom(s, a1, phase, table)
        k2 = _eom(s + 0.5 * dt * k1, a1, phase, table)
        k3 = _eom(s + 0.5 * dt * k2, a1, phase, table)
        k4 = _eom(s + dt * k3, a1, phase, table)
        s = s + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        s[1] = max(s[1], 0.0)
        s[2] = max(s[2], 20.0)
        if phase == "boost":
            s[4] = max(s[4], M0 - M_PROP_BOOST)
        else:
            s[4] = max(s[4], LFRJ_DRY_MASS)
        t += dt
        _record(rows, t, s, phase, alpha_fn(t, s), table)
        if s[1] <= 0.0 and phase == "descent":
            break
    return s, t


def _initial_state() -> np.ndarray:
    h0_m = 35_000.0 * 0.3048
    V0 = 0.8 * atmosphere(h0_m)[3]
    return np.array([0.0, h0_m, V0, np.deg2rad(2.0), M0], dtype=float)


def _prescribed_handoff_state(handoff_mach: float) -> np.ndarray:
    V = handoff_mach * atmosphere(CRUISE_ALT_M)[3]
    return np.array([0.0, CRUISE_ALT_M, V, 0.0, M0 - M_PROP_BOOST], dtype=float)


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _phase_array(rows: list[dict], key: str, phase: str | None = None) -> np.ndarray:
    vals = [r[key] for r in rows if phase is None or r["phase"] == phase]
    return np.asarray(vals, dtype=float)


def _plot(rows: list[dict], outdir: Path):
    phase_colors = {"boost": "#1f77b4", "cruise": "#d62728", "descent": "#2ca02c"}

    def plot_xy(xkey, ykey, xlabel, ylabel, title, fname, sx=1.0, sy=1.0):
        fig, ax = plt.subplots(figsize=(8, 5))
        for phase, color in phase_colors.items():
            xs = _phase_array(rows, xkey, phase) * sx
            ys = _phase_array(rows, ykey, phase) * sy
            if xs.size:
                ax.plot(xs, ys, label=phase, lw=2, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=140)
        plt.close(fig)

    plot_xy("x_m", "h_m", "range [km]", "altitude [km]", "Altitude vs Range",
            "01_altitude_vs_range.png", 1e-3, 1e-3)
    plot_xy("t_s", "Mach", "time [s]", "Mach", "Mach vs Time", "02_mach_vs_time.png")
    plot_xy("t_s", "V_mps", "time [s]", "V [m/s]", "Velocity vs Time", "03_velocity_vs_time.png")
    plot_xy("t_s", "m_kg", "time [s]", "mass [kg]", "Mass vs Time", "04_mass_vs_time.png")
    plot_xy("t_s", "T_N", "time [s]", "thrust [kN]", "Thrust vs Time", "05_thrust_vs_time.png", 1.0, 1e-3)
    plot_xy("t_s", "q_Pa", "time [s]", "q [kPa]", "Dynamic Pressure vs Time", "06_q_vs_time.png", 1.0, 1e-3)
    plot_xy("t_s", "phi", "time [s]", "phi", "Equivalence Ratio vs Time", "07_phi_vs_time.png")

    cruise = [r for r in rows if r["phase"] == "cruise"]
    if cruise:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        tc = _phase_array(rows, "t_s", "cruise")
        axes[0].plot(tc, _phase_array(rows, "M4", "cruise"), color=phase_colors["cruise"], lw=2)
        axes[0].set_ylabel("M4")
        axes[1].plot(tc, _phase_array(rows, "Tt4_K", "cruise"), color=phase_colors["cruise"], lw=2)
        axes[1].set_ylabel("Tt4 [K]")
        axes[2].plot(tc, _phase_array(rows, "unstart_flag", "cruise"), color=phase_colors["cruise"], lw=2)
        axes[2].set_ylabel("unstart")
        axes[2].set_xlabel("time [s]")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle("Cruise Engine Path Observables")
        fig.tight_layout()
        fig.savefig(outdir / "08_engine_constraints.png", dpi=140)
        plt.close(fig)


def _summarize(rows: list[dict], prescribed_handoff: bool) -> dict:
    boost = [r for r in rows if r["phase"] == "boost"]
    cruise = [r for r in rows if r["phase"] == "cruise"]
    descent = [r for r in rows if r["phase"] == "descent"]
    final = rows[-1]
    boost_final = boost[-1] if boost else None
    cruise_final = cruise[-1] if cruise else None
    summary = {
        "prescribed_handoff": prescribed_handoff,
        "total_range_km": final["x_m"] / 1000.0,
        "final_mach": final["Mach"],
        "final_altitude_m": final["h_m"],
        "max_q_kPa": max(r["q_Pa"] for r in rows) / 1000.0,
        "boost": None,
        "cruise": None,
        "descent": None,
        "constraints": {},
    }
    if boost_final is not None:
        summary["boost"] = {
            "burnout_mach": boost_final["Mach"],
            "burnout_altitude_m": boost_final["h_m"],
            "burnout_range_km": boost_final["x_m"] / 1000.0,
            "prop_used_kg": M0 - boost_final["m_kg"],
        }
    if cruise_final is not None:
        cruise_machs = np.asarray([r["Mach"] for r in cruise])
        cruise_alts = np.asarray([r["h_m"] for r in cruise])
        summary["cruise"] = {
            "range_km": (cruise_final["x_m"] - cruise[0]["x_m"]) / 1000.0,
            "fuel_used_kg": cruise[0]["m_kg"] - cruise_final["m_kg"],
            "mean_mach": float(np.mean(cruise_machs)),
            "max_mach": float(np.max(cruise_machs)),
            "mean_altitude_m": float(np.mean(cruise_alts)),
            "min_altitude_m": float(np.min(cruise_alts)),
            "max_altitude_m": float(np.max(cruise_alts)),
            "max_unstart_flag": float(np.max([r["unstart_flag"] for r in cruise])),
            "max_Tt4_K": float(np.max([r["Tt4_K"] for r in cruise])),
            "max_M4": float(np.max([r["M4"] for r in cruise])),
        }
    if descent:
        summary["descent"] = {
            "impact_mach": descent[-1]["Mach"],
            "impact_fpa_deg": abs(np.rad2deg(descent[-1]["gamma_rad"])),
        }
    summary["constraints"] = {
        "boost_handoff_mach_met": bool(
            prescribed_handoff or (boost_final is not None and boost_final["Mach"] >= BOOST_HANDOFF_MACH)
        ),
        "cruise_reached_mach_4p8": bool(cruise_final is not None and summary["cruise"]["max_mach"] >= CRUISE_MACH),
        "cruise_stayed_in_table_mach": bool(cruise_final is not None and summary["cruise"]["max_mach"] <= 5.0 + 1e-6),
        "cruise_stayed_in_table_altitude": bool(
            cruise_final is not None
            and summary["cruise"]["min_altitude_m"] >= 18_000.0 - 1e-6
            and summary["cruise"]["max_altitude_m"] <= 20_000.0 + 1e-6
        ),
        "cruise_no_unstart": bool(cruise_final is not None and summary["cruise"]["max_unstart_flag"] <= 0.5),
        "fuel_within_budget": bool(cruise_final is None or summary["cruise"]["fuel_used_kg"] <= M_FUEL_LFRJ + 1e-6),
    }
    return summary


def run_trajectory(
    prescribed_handoff: bool,
    include_descent: bool,
    force_table: bool,
    handoff_mach: float,
) -> tuple[list[dict], dict]:
    design = fixed_lfrj_design()
    table = build_fixed_perf_table(design, engine=PyCycleRamAdapter(), force=force_table)
    rows: list[dict] = []

    if prescribed_handoff:
        s = _prescribed_handoff_state(handoff_mach)
        t = 0.0
    else:
        s0 = _initial_state()
        s, t = simulate_phase(
            s0,
            "boost",
            t0=0.0,
            duration_s=90.0,
            dt_s=0.1,
            alpha_fn=_alpha_boost,
            rows=rows,
            stop_fn=lambda st: (
                _mach(st[1], st[2]) >= BOOST_HANDOFF_MACH
                or st[4] <= M0 - M_PROP_BOOST + 1e-6
            ),
        )

    handoff_mach = _mach(s[1], s[2])
    if handoff_mach >= BOOST_HANDOFF_MACH:
        fuel_floor = s[4] - M_FUEL_LFRJ
        s, t = simulate_phase(
            s,
            "cruise",
            t0=t,
            duration_s=1200.0,
            dt_s=1.0,
            alpha_fn=_alpha_cruise,
            rows=rows,
            table=table,
            stop_fn=lambda st: (
                st[4] <= fuel_floor + 1e-6
                or st[1] <= table.GRID_H[0]
                or st[1] >= table.GRID_H[-1]
                or _mach(st[1], st[2]) >= table.GRID_M[-1]
            ),
        )
    else:
        print(
            f"[warn] boost ended at Mach {handoff_mach:.2f}; "
            f"cruise skipped because fixed table starts at Mach {BOOST_HANDOFF_MACH:.1f}."
        )

    if include_descent and rows:
        s[3] = min(s[3], np.deg2rad(-10.0))
        s, t = simulate_phase(
            s,
            "descent",
            t0=t,
            duration_s=400.0,
            dt_s=0.5,
            alpha_fn=_alpha_descent,
            rows=rows,
            stop_fn=lambda st: st[1] <= 1.0,
        )

    summary = _summarize(rows, prescribed_handoff) if rows else {}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the fixed-geometry LFRJ trajectory using the pyc_run performance table."
    )
    parser.add_argument(
        "--prescribed-handoff",
        action="store_true",
        help="start powered cruise at Mach 4 and 19 km instead of simulating the solid boost",
    )
    parser.add_argument(
        "--handoff-mach",
        type=float,
        default=BOOST_HANDOFF_MACH,
        help="Mach number to use with --prescribed-handoff",
    )
    parser.add_argument(
        "--include-descent",
        action="store_true",
        help="append an unpowered descent after cruise",
    )
    parser.add_argument(
        "--force-table",
        action="store_true",
        help="rebuild the fixed performance table before trajectory simulation",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTDIR),
        help="directory for trajectory CSV, summary, and plots",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_trajectory(
        args.prescribed_handoff,
        args.include_descent,
        args.force_table,
        args.handoff_mach,
    )
    if not rows:
        raise RuntimeError("No trajectory rows generated.")

    _write_csv(outdir / "trajectory.csv", rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, outdir)

    print("=" * 72)
    print("Fixed-geometry LFRJ trajectory")
    print("=" * 72)
    print(f"  output dir       : {outdir}")
    print(f"  prescribed boost : {args.prescribed_handoff}")
    print(f"  total range      : {summary['total_range_km']:.1f} km")
    print(f"  final Mach       : {summary['final_mach']:.2f}")
    print(f"  max q            : {summary['max_q_kPa']:.1f} kPa")
    if summary.get("boost"):
        b = summary["boost"]
        print(f"  boost burnout    : M={b['burnout_mach']:.2f}, h={b['burnout_altitude_m']/1000.0:.1f} km")
    if summary.get("cruise"):
        c = summary["cruise"]
        print(f"  cruise range     : {c['range_km']:.1f} km")
        print(f"  cruise fuel      : {c['fuel_used_kg']:.1f} kg")
        print(f"  cruise max Mach  : {c['max_mach']:.2f}")
        print(f"  engine max Tt4   : {c['max_Tt4_K']:.0f} K")
    print("  constraints:")
    for key, val in summary["constraints"].items():
        print(f"    {key:<26s}: {val}")


if __name__ == "__main__":
    main()
