from __future__ import annotations
import os
import sys
import warnings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from combined_cycle_liquid_ram_scram import pyc_run
try:
    from .engine_interface import Design, PerfResult, Status, EngineModel
except ImportError:
    from trajectory_opt.engine_interface import Design, PerfResult, Status, EngineModel

def _design_to_overrides(d: Design) -> dict:
    """Map Design fields → pyc_run.build_design kwargs (step 3)."""
    return dict(
        M0=d.design_M0,
        altitude_m=d.design_alt_m,
        alpha_deg=d.design_alpha_deg,
        leading_edge_angle_deg=d.LE_angle_deg,
        mdot_required=d.design_mdot_kgs,
        width_m=d.inlet_width_m,
        forebody_separation_margin=d.forebody_sep_margin,
        ramp_separation_margin=d.ramp_sep_margin,
        kantrowitz_margin=d.kantrowitz_margin,
        shock_focus_factor=d.shock_focus_factor,
        diffuser_min_shock_accommodation_dh=d.diffuser_min_shock_accommodation_dh,
        diffuser_area_ratio=d.diffuser_AR,
        combustor_length_m=d.combustor_length_m,
        nozzle_AR=d.nozzle_AR,
    )

class PyCycleRamAdapter:
    """Wraps pyc_run.analyze() in the stable PerfResult contract."""

    def __init__(self):
        self._design_cache = {}    # digest -> pyc_run design dict
        self._geometry_cache = {}  # digest -> enriched geometry summary

    def _get_design_dict(self, d: Design) -> dict:
        key = d.digest()
        if key not in self._design_cache:
            self._design_cache[key] = pyc_run.build_design(
                **_design_to_overrides(d)
            )
        return self._design_cache[key]

    def _get_geometry_summary(self, d: Design) -> dict:
        key = d.digest()
        if key not in self._geometry_cache:
            self._geometry_cache[key] = pyc_run.build_geometry_summary(
                self._get_design_dict(d)
            )
        return self._geometry_cache[key]

    def evaluate(self, M, h_m, phi, design: Design) -> PerfResult:
        try:
            pyc_design = self._get_design_dict(design)
            r = pyc_run.analyze(M0=M, altitude_m=h_m, phi=phi,
                                design=pyc_design)
        except Exception as e:
            warnings.warn(f"analyze() failed at M={M:.2f} h={h_m:.0f}: {e}")
            return PerfResult.infeasible()

        unstart = float(r.get('unstart_flag', 0.0))
        status  = Status.UNSTART if abs(unstart) > 0.5 else Status.OK
        geom    = r.get('geometry', {})
        Pt3     = float(r['Pt_stations'].get(3, 0.0))
        M3      = float(r['M_stations'].get(3, 0.0))
        GAMMA   = 1.4
        Pc      = Pt3 / (1.0 + 0.5*(GAMMA-1)*M3**2) ** (GAMMA/(GAMMA-1))

        return PerfResult(
            thrust_N       = float(r.get('thrust', 0.0)),
            Isp_s          = float(r.get('Isp', 0.0)),
            mdot_air_kgs   = float(r.get('mdot_air', 0.0)),
            mdot_fuel_kgs  = float(r.get('mdot_fuel', 1e-6)),
            M4             = float(r['M_stations'].get(4, 0.0)),
            Tt4_K          = float(r['Tt_stations'].get(4, 0.0)),
            unstart_flag   = unstart,
            Pc_Pa          = Pc,
            engine_length_m= float(geom.get('total_length_m', 2.5)),
            max_diameter_m = float(geom.get('max_diameter_m', 0.34)),
            status         = status,
        )

    def geometry(self, design: Design) -> dict:
        return self._get_geometry_summary(design)
