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
    'diffuser_AR':     (1.5, 2.5),
    'design_mdot_kgs': (7.0, 12.0),
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



    """
     1. Create the conda env (Anaconda Prompt or any terminal):

  conda create -n ramjet -c conda-forge python=3.11 -y
  conda activate ramjet
  conda install -c conda-forge pyoptsparse openmdao dymos numpy scipy matplotlib cantera ambiance -y
  pip install openmdao-pycycle

  Verify IPOPT is present:
  python -c "from pyoptsparse import IPOPT; print(IPOPT().name)"

  2. Point PyCharm at the conda env:

  - File → Settings → Project: ... → Python Interpreter → Add Interpreter → Conda Environment → Existing environment
  - Set "Conda executable" to your conda install (e.g., C:\Users\hanks\miniconda3\Scripts\conda.exe) and pick ramjet
  from the "Use existing environment" dropdown.
  - Apply. The interpreter at the bottom-right of the editor should now read "Python 3.11 (ramjet)".
  - Verify: open the PyCharm terminal — the prompt should say (ramjet). If not, add conda activate ramjet to your
  terminal startup or use PyCharm's "Start with conda" option under Settings → Tools → Terminal.

  3. Activate IPOPT in the code:

  In trajectory_opt/trajectory_problem.py you already have the IPOPT block commented at lines 83-91. Swap which block is
   live:
  - Comment out lines 79-81 (ScipyOptimizeDriver / SLSQP).
  - Uncomment lines 84-89 (pyOptSparseDriver / IPOPT + declare_coloring()).

  The declare_coloring() call is the biggest speed lever — it detects sparsity once per setup and reuses it.

  4. Run once to seed the PerfTable cache:

  python run_and_plot.py

  First run rebuilds the surrogate pickle (one pyCycle call per grid node). Second run and beyond skip this entirely for
   the same design.digest().
    
    """
