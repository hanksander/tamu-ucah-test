from __future__ import annotations
import os
import sys
import pickle
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .engine_interface import Design, PerfResult, Status, EngineModel
except ImportError:
    from trajectory_opt.engine_interface import Design, PerfResult, Status, EngineModel

CACHE_DIR = Path(__file__).resolve().parent / ".perf_cache"

class PerfTable:
    """M × h × phi interpolator for a fixed Design. Cached on disk."""

    # Cruise-only envelope: M=4-5, alt=15-24 km (user-specified design window).
    GRID_M   = np.linspace(4.0, 5.0, 3)
    GRID_H   = np.linspace(15_000., 24_000., 4)
    GRID_PHI = np.array([0.7, 0.9])
    _FIELDS = ('thrust','isp','mdot_a','mdot_f','M4','Tt4',
                'unstart','Pc','L','D')

    def __init__(self, design: Design, engine: EngineModel):
        self.design = design
        self.engine = engine
        self._interps = None
        self._geom    = None

    def _cache_path(self) -> Path:
        # Include the grid signature so a grid-bound change (e.g. widening
        # the altitude range) doesn't collide with an older pickle of the
        # same design.
        grid_tag = f"{len(self.GRID_M)}x{len(self.GRID_H)}x{len(self.GRID_PHI)}"
        return CACHE_DIR / f"table_{self.design.digest()}_{grid_tag}.pkl"

    def build(self, force: bool=False) -> "PerfTable":
        p = self._cache_path()
        if p.exists() and not force:
            with open(p, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._evaluate_grid()
            CACHE_DIR.mkdir(exist_ok=True)
            with open(p, 'wb') as f:
                pickle.dump(data, f)
        self._build_interps(data)
        return self

    def _evaluate_grid(self) -> dict:
        shape = (len(self.GRID_M), len(self.GRID_H), len(self.GRID_PHI))
        arrs = {k: np.zeros(shape) for k in self._FIELDS}
        for i, M in enumerate(self.GRID_M):
            for j, h in enumerate(self.GRID_H):
                for k, phi in enumerate(self.GRID_PHI):
                    r = self.engine.evaluate(M, h, phi, self.design)
                    arrs['thrust'][i,j,k]  = r.thrust_N
                    arrs['isp'][i,j,k]     = r.Isp_s
                    arrs['mdot_a'][i,j,k]  = r.mdot_air_kgs
                    arrs['mdot_f'][i,j,k]  = r.mdot_fuel_kgs
                    arrs['M4'][i,j,k]      = r.M4
                    arrs['Tt4'][i,j,k]     = r.Tt4_K
                    arrs['unstart'][i,j,k] = r.unstart_flag
                    arrs['Pc'][i,j,k]      = r.Pc_Pa
                    arrs['L'][i,j,k]       = r.engine_length_m
                    arrs['D'][i,j,k]       = r.max_diameter_m
        # Cache constant geometry separately
        arrs['_L_m'] = float(np.mean(arrs['L']))
        arrs['_D_m'] = float(np.mean(arrs['D']))
        return arrs

    def _build_interps(self, data: dict):
        pts = (self.GRID_M, self.GRID_H, self.GRID_PHI)
        self._interps = {
            k: RegularGridInterpolator(pts, data[k], bounds_error=False,
                                        fill_value=None)  # nearest extrapolation
            for k in self._FIELDS if k in ('thrust','isp','mdot_a','mdot_f',
                                            'M4','Tt4','unstart','Pc')
        }
        self._geom = (data['_L_m'], data['_D_m'])

    def lookup(self, M: float, h_m: float, phi: float) -> PerfResult:
        x = np.array([[np.clip(M,    self.GRID_M[0],   self.GRID_M[-1]),
                        np.clip(h_m,  self.GRID_H[0],   self.GRID_H[-1]),
                        np.clip(phi,  self.GRID_PHI[0], self.GRID_PHI[-1])]])
        L, D = self._geom
        return PerfResult(
            thrust_N       = float(self._interps['thrust'](x).item()),
            Isp_s          = float(self._interps['isp'](x).item()),
            mdot_air_kgs   = float(self._interps['mdot_a'](x).item()),
            mdot_fuel_kgs  = max(float(self._interps['mdot_f'](x).item()), 1e-6),
            M4             = float(self._interps['M4'](x).item()),
            Tt4_K          = float(self._interps['Tt4'](x).item()),
            unstart_flag   = float(self._interps['unstart'](x).item()),
            Pc_Pa          = float(self._interps['Pc'](x).item()),
            engine_length_m= L,
            max_diameter_m = D,
            status         = Status.OK,
        )

    def lookup_batch(self, M_arr, h_arr, phi: float) -> dict:
        """Vectorized lookup over N flight points.
        Returns dict of length-N arrays keyed by {thrust, isp, mdot_a,
        mdot_f, M4, Tt4, unstart, Pc}."""
        M_arr = np.asarray(M_arr, dtype=float).ravel()
        h_arr = np.asarray(h_arr, dtype=float).ravel()
        n = M_arr.size
        x = np.empty((n, 3))
        x[:, 0] = np.clip(M_arr, self.GRID_M[0], self.GRID_M[-1])
        x[:, 1] = np.clip(h_arr, self.GRID_H[0], self.GRID_H[-1])
        x[:, 2] = np.clip(float(phi), self.GRID_PHI[0], self.GRID_PHI[-1])
        out = {k: np.asarray(interp(x)).ravel()
               for k, interp in self._interps.items()}
        out['mdot_f'] = np.maximum(out['mdot_f'], 1e-6)
        return out
