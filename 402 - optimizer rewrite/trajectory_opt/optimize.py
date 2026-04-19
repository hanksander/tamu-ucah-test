from __future__ import annotations
import os
import sys
import numpy as np
import dymos as dm
from dataclasses import replace, asdict
from scipy.optimize import minimize

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from combined_cycle_liquid_ram_scram.pyc_config import (
    ENGINE_L_MAX_M, ENGINE_D_MAX_M,
)
from powered_flight_trajectory_code_v4 import atmosphere
try:
    from .engine_interface import Design
    from .engine_adapter import PyCycleRamAdapter
    from .perf_surrogate import PerfTable
    from .trajectory_problem import (
        build_inner_problem, extract_metrics,
        snapshot_for_warmstart, apply_warmstart,
    )
except ImportError:
    from trajectory_opt.engine_interface import Design
    from trajectory_opt.engine_adapter import PyCycleRamAdapter
    from trajectory_opt.perf_surrogate import PerfTable
    from trajectory_opt.trajectory_problem import (
        build_inner_problem, extract_metrics,
        snapshot_for_warmstart, apply_warmstart,
    )

# ----- DV schema (single source of truth for outer driver) -----
# Reduced to the three geometry/sizing DVs the user cares about.
DV_NAMES  = ('diffuser_AR', 'design_mdot_kgs', 'design_M0')
DV_BOUNDS = {
    'diffuser_AR':     (1.5, 3.0),
    'design_mdot_kgs': (5.0, 15.0),
    'design_M0':       (4.0, 5.0),
}

_engine = PyCycleRamAdapter()
_warm_snap = None

def design_from_x(x: np.ndarray, base: Design) -> Design:
    return replace(base, **dict(zip(DV_NAMES, x)))

def _evaluate_design(design: Design, verbose: bool=True) -> float:
    """Returns NEGATIVE range [km] (scipy minimizes)."""
    global _warm_snap

    # Fast geometry-only feasibility gate
    geom = _engine.geometry(design)
    L = float(geom.get('total_length_m', 0.0))
    D = float(geom.get('max_diameter_m', 0.0))
    if L > ENGINE_L_MAX_M or D > ENGINE_D_MAX_M:
        if verbose: print(f"  [skip] geom infeasible L={L:.2f} D={D:.2f}")
        return 1e6

    try:
        table = PerfTable(design, _engine).build()
    except Exception as e:
        if verbose: print(f"  [skip] surrogate build failed: {e}")
        return 1e6

    # LFRJ cruise entry: assume solid rocket already boosted to Mach 4 at 20 km
    h0_m = 20_000.
    V0   = 4.0 * atmosphere(h0_m)[3]

    try:
        p = build_inner_problem(table, h0_m, V0, phi_init=0.8)
        if _warm_snap is not None:
            apply_warmstart(p, _warm_snap)
        dm.run_problem(p, run_driver=True, simulate=False)
        m = extract_metrics(p)
        _warm_snap = snapshot_for_warmstart(p)
        range_km = m['range_m'] / 1000.0
        if verbose:
            print(f"  design={design.digest()}  range={range_km:.1f} km  "
                    f"L={L:.2f} D={D:.2f}")
        return -range_km
    except Exception as e:
        if verbose: print(f"  [fail] inner solve: {e}")
        return 1e6

def run(baseline: Design | None = None, maxiter: int = 30) -> tuple[Design, object]:
    base = baseline or Design()
    x0   = np.array([getattr(base, n) for n in DV_NAMES])
    bnds = [DV_BOUNDS[n] for n in DV_NAMES]

    history = []
    def obj(x):
        d = design_from_x(x, base)
        v = _evaluate_design(d)
        history.append((d, v))
        return v

    result = minimize(
        obj, x0=x0, bounds=bnds, method='L-BFGS-B',
        options={'eps': 2e-3, 'maxiter': maxiter, 'ftol': 1e-4, 'disp': True},
    )
    opt = design_from_x(result.x, base)
    return opt, result


if __name__ == "__main__":
    run()
