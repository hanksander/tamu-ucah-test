from __future__ import annotations
import os, sys
from dataclasses import dataclass, asdict, fields
from enum import Enum
from typing import Protocol, runtime_checkable
import hashlib, json

# pyc_config is the single source of truth for engine design defaults.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from combined_cycle_liquid_ram_scram.pyc_config import (
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M, INLET_DESIGN_ALPHA_DEG,
    INLET_DESIGN_LEADING_EDGE_ANGLE_DEG, INLET_DESIGN_MDOT_KGS,
    INLET_DESIGN_WIDTH_M, INLET_FOREBODY_SEP_MARGIN,
    INLET_RAMP_SEP_MARGIN, INLET_KANTROWITZ_MARGIN,
    INLET_SHOCK_FOCUS_FACTOR, DIFFUSER_AREA_RATIO,
    COMBUSTOR_L_STAR_DEFAULT, NOZZLE_AR_DEFAULT,
)

class Status(Enum):
    OK         = "ok"
    UNSTART    = "unstart"
    SWALLOWED  = "swallowed"
    OFF_DESIGN = "off_design"
    INFEASIBLE = "infeasible"

@dataclass(frozen=True)
class Design:
    """All geometry + operating DVs. Defaults are sourced from pyc_config.py
    so `Design()` and `pyc_run.build_design()` agree bit-for-bit.
    Add fields with defaults; never break existing callers."""
    kantrowitz_margin:   float = float(INLET_KANTROWITZ_MARGIN)
    diffuser_AR:         float = float(DIFFUSER_AREA_RATIO)
    combustor_L_star:    float = float(COMBUSTOR_L_STAR_DEFAULT)
    nozzle_AR:           float = float(NOZZLE_AR_DEFAULT)
    LE_angle_deg:        float = float(INLET_DESIGN_LEADING_EDGE_ANGLE_DEG)
    ramp_sep_margin:     float = float(INLET_RAMP_SEP_MARGIN)
    forebody_sep_margin: float = float(INLET_FOREBODY_SEP_MARGIN)
    inlet_width_m:       float = float(INLET_DESIGN_WIDTH_M)
    shock_focus_factor:  float = float(INLET_SHOCK_FOCUS_FACTOR)
    # design-point anchor (sizing condition, not an operating point)
    design_M0:           float = float(INLET_DESIGN_M0)
    design_alt_m:        float = float(INLET_DESIGN_ALT_M)
    design_alpha_deg:    float = float(INLET_DESIGN_ALPHA_DEG)
    design_mdot_kgs:     float = float(INLET_DESIGN_MDOT_KGS)

    def digest(self) -> str:
        return hashlib.sha1(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]

@dataclass(frozen=True)
class PerfResult:
    """Trajectory contract. Order doesn't matter; names do.
    Any field the trajectory might read should have a safe default
    for the INFEASIBLE path."""
    thrust_N:        float
    Isp_s:           float
    mdot_air_kgs:    float
    mdot_fuel_kgs:   float
    # Path-constraint observables
    M4:              float   # combustor-exit Mach (>1 = thermal choke)
    Tt4_K:           float
    unstart_flag:    float   # 0 = healthy; !=0 = unstart/swallow
    Pc_Pa:           float
    # Geometry (constant for a given Design; echoed for convenience)
    engine_length_m: float
    max_diameter_m: float
    status:          Status = Status.OK

    @classmethod
    def infeasible(cls) -> "PerfResult":
        return cls(
            thrust_N=0.0, Isp_s=0.0, mdot_air_kgs=0.0, mdot_fuel_kgs=1e-6,
            M4=2.0, Tt4_K=0.0, unstart_flag=1.0, Pc_Pa=0.0,
            engine_length_m=99.0, max_diameter_m=99.0,
            status=Status.INFEASIBLE,
        )



@runtime_checkable
class EngineModel(Protocol):
    """Any physics implementation that conforms to this is plug-compatible."""
    """Plug-compatible physics contract. Any class with these methods
          is accepted by the surrogate builder, the trajectory ODEs, and the
          outer optimizer."""

    def evaluate(
            self,
            M: float,
            h_m: float,
            phi: float,
            design: Design,
    ) -> PerfResult:
        """Full engine cycle at one flight point. Must never raise —
        return PerfResult.infeasible() on failure so the optimizer can
        keep moving."""
        ...

    def geometry(self, design: Design) -> dict:
        """Geometry-only build for `design`, WITHOUT running the cycle.
        Used by the outer optimizer for the fast feasibility gate
        (length / diameter limits) before paying for a surrogate rebuild.

        Required keys:
            'total_length_m'  : float
            'max_diameter_m'  : float
            'max_width_m'     : float
            'max_height_m'    : float
            'capture_area_m2' : float
            'throat_area_m2'  : float
            'combustor_volume_m3' : float
        Extra keys are permitted and ignored by core consumers."""
        ...


