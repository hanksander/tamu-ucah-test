from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from enum import Enum
from typing import Protocol, runtime_checkable
import hashlib, json

class Status(Enum):
    OK         = "ok"
    UNSTART    = "unstart"
    SWALLOWED  = "swallowed"
    OFF_DESIGN = "off_design"
    INFEASIBLE = "infeasible"

@dataclass(frozen=True)
class Design:
    """All geometry + operating DVs. Defaults match pyc_config.py baselines.
    Add fields with defaults; never break existing callers."""
    kantrowitz_margin:   float = 0.80
    diffuser_AR:         float = 2.0
    combustor_L_star:    float = 1.25
    nozzle_AR:           float = 5.0
    LE_angle_deg:        float = 4.0
    ramp_sep_margin:     float = 0.25
    forebody_sep_margin: float = 0.25
    inlet_width_m:       float = 0.25
    shock_focus_factor:  float = 1.10
    # design-point anchor (sizing condition, not an operating point)
    design_M0:           float = 5.0
    design_alt_m:        float = 18_000.0
    design_alpha_deg:    float = 4.0
    design_mdot_kgs:     float = 10.0

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
            'capture_area_m2' : float
            'throat_area_m2'  : float
        Extra keys are permitted and ignored by core consumers."""
        ...


