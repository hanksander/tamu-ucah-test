from __future__ import annotations

import csv
import contextlib
import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from .engine_adapter import PyCycleRamAdapter
    from .engine_interface import Design, PerfResult
    from .perf_surrogate import PerfTable
except ImportError:
    from trajectory_opt.engine_adapter import PyCycleRamAdapter
    from trajectory_opt.engine_interface import Design, PerfResult
    from trajectory_opt.perf_surrogate import PerfTable


BOOST_HANDOFF_MACH = 4.0
CRUISE_MACH = 4.8
CRUISE_ALT_M = 19_000.0
PHI_CRUISE = 0.8

# Direct pyCycle validation points before the trajectory uses interpolation.
VALIDATION_MACHS = np.array([4.0, 4.4, 4.8, 5.0])
VALIDATION_ALTS_M = np.array([18_000.0, 19_000.0, 20_000.0])

# Focused surrogate grid for the fixed-geometry Mach 4 boost handoff and
# Mach 4.8 / 19 km cruise mission.
SURROGATE_GRID_M = np.array([4.0, 4.4, 4.8, 5.0])
SURROGATE_GRID_H_M = np.array([18_000.0, 19_000.0, 20_000.0])
SURROGATE_GRID_PHI = np.array([0.40, 0.50, 0.65, 0.80, 0.90])


def fixed_lfrj_design(base: Design | None = None) -> Design:
    """Frozen engine geometry from pyc_config via Design defaults."""
    return Design() if base is None else base


def _perf_row(M: float, h_m: float, phi: float, result: PerfResult) -> dict:
    return {
        "Mach": float(M),
        "altitude_m": float(h_m),
        "phi": float(phi),
        "status": result.status.value,
        "thrust_N": result.thrust_N,
        "Isp_s": result.Isp_s,
        "mdot_air_kgs": result.mdot_air_kgs,
        "mdot_fuel_kgs": result.mdot_fuel_kgs,
        "M4": result.M4,
        "Tt4_K": result.Tt4_K,
        "unstart_flag": result.unstart_flag,
        "Pc_Pa": result.Pc_Pa,
        "engine_length_m": result.engine_length_m,
        "max_diameter_m": result.max_diameter_m,
    }


@contextlib.contextmanager
def _quiet_solver_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def validate_fixed_engine(
    design: Design,
    engine: PyCycleRamAdapter | None = None,
    machs=VALIDATION_MACHS,
    altitudes_m=VALIDATION_ALTS_M,
    phi: float = PHI_CRUISE,
    quiet: bool = True,
) -> list[dict]:
    """Run direct pyCycle points for the frozen geometry."""
    engine = PyCycleRamAdapter() if engine is None else engine
    rows = []
    for h_m in altitudes_m:
        for M in machs:
            with _quiet_solver_output(quiet):
                result = engine.evaluate(M, h_m, phi, design)
            rows.append(_perf_row(M, h_m, phi, result))
    return rows


def build_fixed_perf_table(
    design: Design,
    engine: PyCycleRamAdapter | None = None,
    force: bool = False,
    quiet: bool = True,
) -> PerfTable:
    """Build the fixed-geometry performance table used by the trajectory."""
    engine = PyCycleRamAdapter() if engine is None else engine
    table = PerfTable(
        design,
        engine,
        grid_M=SURROGATE_GRID_M,
        grid_H=SURROGATE_GRID_H_M,
        grid_PHI=SURROGATE_GRID_PHI,
        cache_label="lfrj_m48_19km",
    )
    with _quiet_solver_output(quiet):
        return table.build(force=force)


def write_fixed_engine_report(
    output_dir: str | Path,
    design: Design,
    validation_rows: list[dict],
    table: PerfTable | None,
) -> dict:
    """Persist the fixed-geometry design, validation map, and grid metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    design_path = output_dir / "fixed_lfrj_design.json"
    validation_path = output_dir / "fixed_lfrj_validation.csv"
    summary_path = output_dir / "fixed_lfrj_summary.json"

    design_path.write_text(json.dumps(asdict(design), indent=2))

    with validation_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    surrogate_grid = None
    if table is not None:
        surrogate_grid = {
            "Mach": table.GRID_M.tolist(),
            "altitude_m": table.GRID_H.tolist(),
            "phi": table.GRID_PHI.tolist(),
            "cache_path": str(table._cache_path()),
        }

    summary = {
        "design_digest": design.digest(),
        "boost_handoff_mach": BOOST_HANDOFF_MACH,
        "cruise_mach": CRUISE_MACH,
        "cruise_alt_m": CRUISE_ALT_M,
        "phi_cruise": PHI_CRUISE,
        "surrogate_grid": surrogate_grid,
        "files": {
            "design": str(design_path),
            "validation": str(validation_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
