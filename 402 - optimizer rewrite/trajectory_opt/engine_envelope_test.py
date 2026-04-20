from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from typing import Iterable
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_IMPORT_ERROR = None
try:
    from combined_cycle_liquid_ram_scram import pyc_run
    from combined_cycle_liquid_ram_scram.pyc_config import (
        ENGINE_COMBUSTOR_D_MAX_M,
        ENGINE_D_MAX_M,
        ENGINE_L_MAX_M,
        ENGINE_MIN_THRUST_N,
        ENGINE_NOZZLE_EXIT_D_MAX_M,
        M4_MAX,
        TT4_MAX_K,
    )

    try:
        from .engine_adapter import PyCycleRamAdapter
        from .engine_interface import Design
    except ImportError:
        from trajectory_opt.engine_adapter import PyCycleRamAdapter
        from trajectory_opt.engine_interface import Design
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    pyc_run = None
    PyCycleRamAdapter = None
    Design = None
    ENGINE_COMBUSTOR_D_MAX_M   = 0.35
    ENGINE_D_MAX_M             = 0.38
    ENGINE_L_MAX_M             = 3.8
    ENGINE_MIN_THRUST_N        = 6_000.0
    ENGINE_NOZZLE_EXIT_D_MAX_M = 0.38


# ═══════════════════════════════════════════════════════════════════════════
# ENVELOPE + CONSTRAINTS  — edit here to change what the search must satisfy.
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class EnvelopeSpec:
    # ── ENVELOPE (flight conditions the design must work at) ──
    mach_min: float = 4.0
    mach_max: float = 5.0
    alt_min_m: float = 19_000.0
    alt_max_m: float = 21_000.0
    aoa_min_deg: float = 0.0
    aoa_max_deg: float = 3.0
    nominal_aoa_deg: float = 3.0

    # ── CONSTRAINTS (pass/fail limits applied to every candidate) ──
    max_total_length_m: float = 3.8
    max_width_m: float = 0.275
    max_height_m: float = 0.38
    max_diameter_m: float = 0.38             # overall frontal cap
    max_combustor_diameter_m: float = 0.35   # combustion-chamber cap
    max_nozzle_exit_diameter_m: float = 0.38 # nozzle exit cap
    min_combustor_volume_m3: float = 0.08
    min_thrust_N: float = 6_000.0
    phi_for_thrust: float = 1.0
    reject_if_phi_clipped: bool = False


@dataclass(frozen=True)
class CandidateResult:
    design: Design
    feasible: bool
    score: float
    reasons: tuple[str, ...]
    geometry: dict
    points_checked: int
    worst_pt_recovery: float
    worst_capture_mdot_kgs: float
    worst_thrust_N: float
    worst_M4: float
    worst_Tt4_K: float
    worst_phi_effective: float
    worst_max_height_m: float
    combustor_volume_m3: float


# ═══════════════════════════════════════════════════════════════════════════
# FROZEN VARIABLES  — baseline design values that are NOT swept.
# Anything listed here that doesn't appear in _candidate_grid / _refined_grid
# below stays fixed across every candidate.  Move a knob out of the grids and
# into here to freeze it; move it out of here and into the grids to sweep it.
# ═══════════════════════════════════════════════════════════════════════════
def _build_baseline_design(spec: EnvelopeSpec) -> Design:
    return Design(
        kantrowitz_margin=0.85,
        diffuser_AR=1.75,
        combustor_L_star=1.5,
        nozzle_AR=9.0,                       # overridden by pyc_run ideal-expansion sizing
        LE_angle_deg=4.5,
        ramp_sep_margin=0.30,
        forebody_sep_margin=0.30,
        inlet_width_m=spec.max_width_m,      # 0.275 m hard requirement
        shock_focus_factor=1.10,
        diffuser_min_shock_accommodation_dh=3.0,
        design_M0=5.0,
        design_alt_m=0.5 * (spec.alt_min_m + spec.alt_max_m),   # 20 000 m
        design_alpha_deg=spec.nominal_aoa_deg,                  # 3.0 deg
        design_mdot_kgs=5.0,
    )


def _build_design_dict(design: Design) -> dict:
    return pyc_run.build_design(
        M0=design.design_M0,
        altitude_m=design.design_alt_m,
        alpha_deg=design.design_alpha_deg,
        leading_edge_angle_deg=design.LE_angle_deg,
        mdot_required=design.design_mdot_kgs,
        width_m=design.inlet_width_m,
        forebody_separation_margin=design.forebody_sep_margin,
        ramp_separation_margin=design.ramp_sep_margin,
        kantrowitz_margin=design.kantrowitz_margin,
        shock_focus_factor=design.shock_focus_factor,
        diffuser_min_shock_accommodation_dh=design.diffuser_min_shock_accommodation_dh,
        diffuser_area_ratio=min(design.diffuser_AR, 2.5),
        combustor_L_star=design.combustor_L_star,
        nozzle_AR=design.nozzle_AR,
    )


def _envelope_points(spec: EnvelopeSpec) -> list[tuple[float, float, float]]:
    # 8 envelope corners + the nominal-α midpoint.  Sufficient for a first-pass
    # feasibility sweep (captures all extreme combinations of M, alt, α).
    mach_mid = 0.5 * (spec.mach_min + spec.mach_max)
    alt_mid  = 0.5 * (spec.alt_min_m + spec.alt_max_m)
    points: set[tuple[float, float, float]] = set()
    for m, a, aoa in itertools.product(
        (spec.mach_min, spec.mach_max),
        (spec.alt_min_m, spec.alt_max_m),
        (spec.aoa_min_deg, spec.aoa_max_deg),
    ):
        points.add((m, a, aoa))
    points.add((mach_mid, alt_mid, spec.nominal_aoa_deg))
    return sorted(points)


def _score_geometry(geometry: dict) -> float:
    forebody = float(geometry.get("forebody_length_m", 99.0))
    total = float(geometry.get("total_length_m", 99.0))
    diameter = float(geometry.get("max_diameter_m", 99.0))
    width = float(geometry.get("max_width_m", 99.0))
    height = float(geometry.get("max_height_m", 99.0))
    volume_shortfall = max(0.0, 0.14 - float(geometry.get("combustor_volume_m3", 0.0)))
    return forebody + 0.15 * total + 0.05 * diameter + 0.05 * width + 0.05 * height + volume_shortfall


def _cheap_max_height_m(geometry: dict, design: Design) -> float:
    vals = [
        float(geometry.get("opening_normal_to_ramp2_m", 0.0)),
        float(geometry.get("post_cowl_height_m", 0.0)),
        float(geometry.get("throat_height_m", 0.0)),
        float(geometry.get("diffuser", {}).get("exit_diameter_m", 0.0)),
        float(design.inlet_width_m),
    ]
    return max(vals)


def _analysis_max_height_m(analysis: dict, design_dict: dict) -> float:
    inlet = analysis.get("inlet_geometry", {})
    combustor = analysis.get("combustor_section", {})
    nozzle = analysis.get("nozzle_geometry", {})
    vals = [
        float(design_dict.get("opening_normal_to_ramp2_m", 0.0)),
        float(design_dict.get("post_cowl_height_m", 0.0)),
        float(design_dict.get("throat_height_m", 0.0)),
        float(design_dict.get("diffuser", {}).get("exit_diameter_m", 0.0)),
        float(inlet.get("throat_height_m", 0.0)),
        float(inlet.get("diffuser_exit_diameter_m", 0.0)),
        float(combustor.get("height_m", 0.0)),
        float(combustor.get("diameter_m", 0.0)),
        float(nozzle.get("exit_diameter_m", 0.0)),
        float(nozzle.get("throat_diameter_m", 0.0)),
    ]
    return max(vals)


# ═══════════════════════════════════════════════════════════════════════════
# FREE VARIABLES  — knobs the search actually sweeps.
# `_candidate_grid` is the coarse pass (runs on every candidate).
# `_refined_grid`  is the local refinement around the top-n coarse seeds.
# Add/remove keys from the `knobs` dict to change what is swept; edit the
# tuple values to change the discretization.  Anything NOT in these dicts
# is inherited from _build_baseline_design above and stays frozen.
# ═══════════════════════════════════════════════════════════════════════════
def _candidate_grid(base: Design, spec: EnvelopeSpec) -> Iterable[Design]:
    # Five-knob coarse grid (2·3·2·2·3 = 72 candidates) for a sub-hour sweep.
    # Forebody/ramp margins, shock focus, Lstar, shock-accommodation D_h, and
    # design altitude are frozen in _build_baseline_design.
    knobs = {
        "design_M0":         (4.5, 5.0),
        "design_mdot_kgs":   (4.0, 5.0, 6.0),
        "kantrowitz_margin": (0.80, 0.85),
        "LE_angle_deg":      (3.5, 4.5),
        "diffuser_AR":       (1.5, 1.75, 2.0),
    }
    names = tuple(knobs.keys())
    for values in itertools.product(*(knobs[name] for name in names)):
        yield replace(base, **dict(zip(names, values)))


def _refined_grid(seed: Design, spec: EnvelopeSpec) -> Iterable[Design]:
    def around(value: float, delta: float, low: float, high: float) -> tuple[float, ...]:
        vals = sorted({max(low, min(high, value - delta)), value, max(low, min(high, value + delta))})
        return tuple(vals)

    knobs = {
        "design_M0":         around(seed.design_M0,         0.25,  4.5,  5.0),
        "design_mdot_kgs":   around(seed.design_mdot_kgs,   0.5,   3.5,  6.5),
        "kantrowitz_margin": around(seed.kantrowitz_margin, 0.025, 0.80, 0.90),
        "LE_angle_deg":      around(seed.LE_angle_deg,      0.25,  3.0,  5.0),
        "diffuser_AR":       around(seed.diffuser_AR,       0.125, 1.4,  2.0),
    }
    names = tuple(knobs.keys())
    for values in itertools.product(*(knobs[name] for name in names)):
        yield replace(seed, **dict(zip(names, values)))


def evaluate_candidate(
    design: Design,
    spec: EnvelopeSpec,
    adapter: PyCycleRamAdapter,
) -> CandidateResult:
    reasons: list[str] = []

    try:
        geometry = adapter.geometry(design)
    except Exception as exc:
        return CandidateResult(
            design=design,
            feasible=False,
            score=1.0e9,
            reasons=(f"geometry build failed: {exc}",),
            geometry={},
            points_checked=0,
            worst_pt_recovery=0.0,
            worst_capture_mdot_kgs=0.0,
            worst_thrust_N=0.0,
            worst_M4=0.0,
            worst_Tt4_K=0.0,
            worst_phi_effective=0.0,
            worst_max_height_m=0.0,
            combustor_volume_m3=0.0,
        )

    total_length = float(geometry.get("total_length_m", math.inf))
    max_diameter = float(geometry.get("max_diameter_m", math.inf))
    max_width = float(
        max(
            geometry.get("width_m", 0.0),
            geometry.get("inlet_width_m", 0.0),
            design.inlet_width_m,
        )
    )
    max_height = float(_cheap_max_height_m(geometry, design))
    geometry["max_width_m"] = max_width
    geometry["max_height_m"] = max_height
    geometry["combustor_volume_m3"] = float(geometry.get("combustor_volume_m3", 0.0))
    if total_length > spec.max_total_length_m:
        reasons.append(
            f"length {total_length:.3f} m exceeds limit {spec.max_total_length_m:.3f} m"
        )
    if max_diameter > spec.max_diameter_m:
        reasons.append(
            f"diameter {max_diameter:.3f} m exceeds limit {spec.max_diameter_m:.3f} m"
        )
    if max_width > spec.max_width_m:
        reasons.append(
            f"width {max_width:.3f} m exceeds limit {spec.max_width_m:.3f} m"
        )
    if design.inlet_width_m > spec.max_width_m:
        reasons.append(
            f"inlet width {design.inlet_width_m:.3f} m exceeds limit {spec.max_width_m:.3f} m"
        )
    if max_height > spec.max_height_m:
        reasons.append(
            f"height {max_height:.3f} m exceeds limit {spec.max_height_m:.3f} m"
        )
    if reasons:
        return CandidateResult(
            design=design,
            feasible=False,
            score=1.0e6 + _score_geometry(geometry),
            reasons=tuple(reasons),
            geometry=geometry,
            points_checked=0,
            worst_pt_recovery=0.0,
            worst_capture_mdot_kgs=0.0,
            worst_thrust_N=0.0,
            worst_M4=0.0,
            worst_Tt4_K=0.0,
            worst_phi_effective=0.0,
            worst_max_height_m=0.0,
            combustor_volume_m3=0.0,
        )

    design_dict = _build_design_dict(design)
    worst_pt_recovery = math.inf
    worst_capture_mdot_kgs = math.inf
    worst_thrust_N = math.inf
    worst_M4 = -math.inf
    worst_Tt4_K = -math.inf
    worst_phi_effective = math.inf
    worst_max_height_m = -math.inf
    combustor_volume_m3 = 0.0
    checked = 0

    for mach, alt_m, aoa_deg in _envelope_points(spec):
        try:
            capability = pyc_run._inlet2.compute_inlet_capability(
                design_dict,
                M0=mach,
                altitude_m=alt_m,
                alpha_deg=aoa_deg,
            )
        except Exception as exc:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: capability exception: {exc}"
            )
            break

        checked += 1
        if not capability.get("ok", False):
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: {capability.get('reason', 'capability failed')}"
            )
            break

        try:
            case = pyc_run._inlet2.evaluate_fixed_geometry_at_condition(
                design_dict,
                M0=mach,
                altitude_m=alt_m,
                alpha_deg=aoa_deg,
            )
        except Exception as exc:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: operating-point exception: {exc}"
            )
            break

        if not case.get("success", False):
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: {case.get('reason', 'off-design failure')}"
            )
            break

        worst_pt_recovery = min(
            worst_pt_recovery,
            float(case.get("pt_frac_deliverable", 0.0)),
        )
        worst_capture_mdot_kgs = min(worst_capture_mdot_kgs, float(py_catch_mdot(mach, alt_m, geometry)))

        analysis = pyc_run.analyze(
            M0=mach,
            altitude_m=alt_m,
            phi=spec.phi_for_thrust,
            design=design_dict,
            verbose=False,
        )
        combustor_volume_m3 = float(analysis.get("combustor_geometry", {}).get("volume_m3", combustor_volume_m3))
        geometry["combustor_volume_m3"] = combustor_volume_m3
        if combustor_volume_m3 < spec.min_combustor_volume_m3:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: combustor volume {combustor_volume_m3:.3f} m^3 "
                f"below minimum {spec.min_combustor_volume_m3:.3f} m^3"
            )
            break
        point_max_height = float(_analysis_max_height_m(analysis, design_dict))
        worst_max_height_m = max(worst_max_height_m, point_max_height)
        geometry["max_height_m"] = max(float(geometry.get("max_height_m", 0.0)), point_max_height)
        if point_max_height > spec.max_height_m:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: height {point_max_height:.3f} m "
                f"exceeds limit {spec.max_height_m:.3f} m"
            )
            break

        combustor_diam_m = float(analysis.get("combustor_section", {}).get("diameter_m", 0.0))
        if combustor_diam_m > spec.max_combustor_diameter_m:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: combustor diameter {combustor_diam_m:.3f} m "
                f"exceeds limit {spec.max_combustor_diameter_m:.3f} m"
            )
            break

        nozzle_exit_diam_m = float(analysis.get("nozzle_geometry", {}).get("exit_diameter_m", 0.0))
        if nozzle_exit_diam_m > spec.max_nozzle_exit_diameter_m:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: nozzle exit diameter {nozzle_exit_diam_m:.3f} m "
                f"exceeds limit {spec.max_nozzle_exit_diameter_m:.3f} m"
            )
            break

        unstart_flag = float(analysis.get("unstart_flag", 0.0))
        if abs(unstart_flag) > 0.5:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: ram-cycle closure flagged "
                f"unstart/swallow (unstart_flag={unstart_flag:.1f})"
            )
            break

        thrust = float(analysis.get("thrust", 0.0))
        phi_eff = float(analysis.get("phi_effective", spec.phi_for_thrust))
        worst_thrust_N = min(worst_thrust_N, thrust)
        worst_M4 = max(worst_M4, float(analysis.get("M_stations", {}).get(4, 0.0)))
        worst_Tt4_K = max(worst_Tt4_K, float(analysis.get("Tt_stations", {}).get(4, 0.0)))
        worst_phi_effective = min(worst_phi_effective, phi_eff)
        if thrust < spec.min_thrust_N:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: thrust {thrust/1000.0:.3f} kN "
                f"below minimum {spec.min_thrust_N/1000.0:.3f} kN at phi_effective={phi_eff:.3f}"
            )
            break
        if float(analysis.get("M_stations", {}).get(4, 0.0)) > M4_MAX:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: M4 {float(analysis.get('M_stations', {}).get(4, 0.0)):.3f} "
                f"exceeds M4_MAX={M4_MAX:.3f}"
            )
            break
        if float(analysis.get("Tt_stations", {}).get(4, 0.0)) > TT4_MAX_K:
            reasons.append(
                f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: Tt4 {float(analysis.get('Tt_stations', {}).get(4, 0.0)):.1f} K "
                f"exceeds TT4_MAX_K={TT4_MAX_K:.1f} K"
            )
            break

        if spec.reject_if_phi_clipped:
            phi_req = float(analysis.get("phi_request", spec.phi_for_thrust))
            if phi_eff + 1.0e-6 < phi_req:
                reasons.append(
                    f"M={mach:.2f} h={alt_m:.0f} aoa={aoa_deg:.1f}: closure clipped phi "
                    f"from {phi_req:.3f} to {phi_eff:.3f}"
                )
                break

    feasible = not reasons
    if not math.isfinite(worst_pt_recovery):
        worst_pt_recovery = 0.0
    if not math.isfinite(worst_capture_mdot_kgs):
        worst_capture_mdot_kgs = 0.0
    if not math.isfinite(worst_thrust_N):
        worst_thrust_N = 0.0
    if not math.isfinite(worst_M4):
        worst_M4 = 0.0
    if not math.isfinite(worst_Tt4_K):
        worst_Tt4_K = 0.0
    if not math.isfinite(worst_phi_effective):
        worst_phi_effective = 0.0
    if not math.isfinite(worst_max_height_m):
        worst_max_height_m = float(geometry.get("max_height_m", 0.0))

    score = _score_geometry(geometry)
    if not feasible:
        score += 1.0e5 + 1000.0 * len(reasons)

    return CandidateResult(
        design=design,
        feasible=feasible,
        score=score,
        reasons=tuple(reasons),
        geometry=geometry,
        points_checked=checked,
        worst_pt_recovery=worst_pt_recovery,
        worst_capture_mdot_kgs=worst_capture_mdot_kgs,
        worst_thrust_N=worst_thrust_N,
        worst_M4=worst_M4,
        worst_Tt4_K=worst_Tt4_K,
        worst_phi_effective=worst_phi_effective,
        worst_max_height_m=worst_max_height_m,
        combustor_volume_m3=combustor_volume_m3,
    )


def py_catch_mdot(mach: float, alt_m: float, geometry: dict) -> float:
    fs = pyc_run._inlet2.freestream_state(mach, alt_m)
    capture_area = float(geometry.get("A_capture_required_m2", 0.0))
    return float(fs["rho0"]) * float(fs["V0"]) * capture_area


def _progress_heartbeat(
    timer_state: dict[str, float],
    stage: str,
    done: int,
    total: int,
    feasible_count: int,
    best_score: float | None,
    extra: str = "",
) -> None:
    now = time.perf_counter()
    last = float(timer_state.get(stage, 0.0))
    if done < total and (now - last) < 0.5:
        return
    timer_state[stage] = now
    best_text = "n/a"
    if best_score is not None and math.isfinite(best_score):
        best_text = f"{best_score:.4f}"
    suffix = f" {extra}" if extra else ""
    print(
        f"[progress] {stage} {done}/{total} feasible={feasible_count} "
        f"best_score={best_text}{suffix}"
    )


# ── Parallel evaluation plumbing ──────────────────────────────────────────
#
# Workers each hold a lazy, process-local PyCycleRamAdapter (and therefore
# their own pyCycle Problem), so the ~10–20 s setup cost is paid once per
# worker and amortized across every candidate that worker handles.
# Windows ProcessPoolExecutor uses spawn, so the worker entrypoint MUST be
# defined at module top level.

_WORKER_ADAPTER = None  # populated inside each worker process on first use


def _get_worker_adapter():
    global _WORKER_ADAPTER
    if _WORKER_ADAPTER is None:
        _WORKER_ADAPTER = PyCycleRamAdapter()
    return _WORKER_ADAPTER


def _infeasible_result(design: Design, reason: str) -> CandidateResult:
    return CandidateResult(
        design=design,
        feasible=False,
        score=1.0e9,
        reasons=(reason,),
        geometry={},
        points_checked=0,
        worst_pt_recovery=0.0,
        worst_capture_mdot_kgs=0.0,
        worst_thrust_N=0.0,
        worst_M4=0.0,
        worst_Tt4_K=0.0,
        worst_phi_effective=0.0,
        worst_max_height_m=0.0,
        combustor_volume_m3=0.0,
    )


def evaluate_candidate_worker(payload: tuple[Design, EnvelopeSpec]) -> CandidateResult:
    """Top-level worker entrypoint for ProcessPoolExecutor.

    Accepts pickle-safe (Design, EnvelopeSpec) and returns a
    CandidateResult.  Never raises — exceptions are wrapped into an
    infeasible result so the parent can keep collecting."""
    design, spec = payload
    try:
        adapter = _get_worker_adapter()
        return evaluate_candidate(design, spec, adapter)
    except Exception as exc:
        return _infeasible_result(design, f"worker exception: {exc!r}")


def evaluate_candidates_batch(
    candidates: list[Design],
    spec: EnvelopeSpec,
    stage_name: str,
    max_workers: int,
) -> list[CandidateResult]:
    """Evaluate a batch of candidates serially (max_workers<=1) or in
    parallel via ProcessPoolExecutor (max_workers>1).  Progress heartbeats
    and per-stage summary logging happen on the parent side."""
    if not candidates:
        print(f"[{stage_name}] skipped: 0 candidates")
        return []

    n_pts = len(_envelope_points(spec))
    print(
        f"[{stage_name}] start: candidates={len(candidates)} "
        f"envelope_points={n_pts} workers={max_workers}"
    )

    timer_state: dict[str, float] = {}
    results: list[CandidateResult] = []

    if max_workers <= 1:
        adapter = _get_worker_adapter()
        for idx, candidate in enumerate(candidates, start=1):
            try:
                result = evaluate_candidate(candidate, spec, adapter)
            except Exception as exc:
                result = _infeasible_result(candidate, f"serial exception: {exc!r}")
            results.append(result)
            _progress_heartbeat(
                timer_state,
                stage=stage_name,
                done=idx,
                total=len(candidates),
                feasible_count=sum(1 for r in results if r.feasible),
                best_score=min((r.score for r in results), default=None),
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_candidate = {
                pool.submit(evaluate_candidate_worker, (cand, spec)): cand
                for cand in candidates
            }
            for fut in as_completed(future_to_candidate):
                candidate = future_to_candidate[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = _infeasible_result(
                        candidate, f"future exception: {exc!r}"
                    )
                results.append(result)
                _progress_heartbeat(
                    timer_state,
                    stage=stage_name,
                    done=len(results),
                    total=len(candidates),
                    feasible_count=sum(1 for r in results if r.feasible),
                    best_score=min((r.score for r in results), default=None),
                )

    feasible_count = sum(1 for r in results if r.feasible)
    scores = [r.score for r in results]
    best_score = min(scores) if scores else None
    best_text = (
        f"{best_score:.4f}"
        if best_score is not None and math.isfinite(best_score)
        else "n/a"
    )
    print(
        f"[{stage_name}] summary: evaluated={len(results)} "
        f"feasible={feasible_count} best_score={best_text}"
    )
    return results


def search_envelope(
    spec: EnvelopeSpec,
    top_n: int = 1,
    max_workers: int = 1,
) -> tuple[CandidateResult | None, list[CandidateResult]]:
    baseline = _build_baseline_design(spec)

    # Stage 1: coarse grid — parent builds candidates, batch helper evaluates.
    coarse_candidates = list(_candidate_grid(baseline, spec))
    coarse_results = evaluate_candidates_batch(
        coarse_candidates, spec, "coarse", max_workers
    )
    coarse_results.sort(key=lambda item: item.score)

    # Stage 2: refinement — expand each top-n seed, dedupe in parent, then
    # evaluate the union in one parallel batch.
    seeds = coarse_results[:top_n]
    refined_candidates: list[Design] = []
    seen: set[str] = set()
    for seed in seeds:
        for candidate in _refined_grid(seed.design, spec):
            key = candidate.digest()
            if key in seen:
                continue
            seen.add(key)
            refined_candidates.append(candidate)
    refined_results = evaluate_candidates_batch(
        refined_candidates, spec, "refine", max_workers
    )

    all_results = coarse_results + refined_results
    all_results.sort(key=lambda item: item.score)
    feasible = [item for item in all_results if item.feasible]
    best = feasible[0] if feasible else None
    return best, all_results[:top_n]


def _serialize_candidate(result: CandidateResult) -> dict:
    return {
        "feasible": result.feasible,
        "score": result.score,
        "points_checked": result.points_checked,
        "worst_pt_recovery": result.worst_pt_recovery,
        "worst_capture_mdot_kgs": result.worst_capture_mdot_kgs,
        "worst_thrust_N": result.worst_thrust_N,
        "worst_M4": result.worst_M4,
        "worst_Tt4_K": result.worst_Tt4_K,
        "worst_phi_effective": result.worst_phi_effective,
        "worst_max_height_m": result.worst_max_height_m,
        "combustor_volume_m3": result.combustor_volume_m3,
        "design": asdict(result.design),
        "geometry": result.geometry,
        "reasons": list(result.reasons),
    }


def _format_kv(key: str, value, indent: int = 2) -> str:
    pad = " " * indent
    if isinstance(value, float):
        return f"{pad}{key:<40s} {value:.6f}"
    return f"{pad}{key:<40s} {value}"


def _append_candidate_section(
    lines: list[str],
    result: CandidateResult,
    heading: str,
) -> None:
    geom = result.geometry
    lines.append(heading)
    lines.append("-" * 78)
    lines.append(f"feasible:        {result.feasible}")
    lines.append(f"score:           {result.score:.4f}")
    lines.append(f"points_checked:  {result.points_checked}")
    lines.append("")
    lines.append("Design knobs:")
    for key, value in asdict(result.design).items():
        lines.append(_format_kv(key, value))
    lines.append("")
    lines.append("Geometry:")
    for key in sorted(geom.keys()):
        value = geom[key]
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for sub_key in sorted(value.keys()):
                lines.append(_format_kv(sub_key, value[sub_key], indent=4))
        else:
            lines.append(_format_kv(key, value))
    lines.append("")
    lines.append("Worst-case performance across envelope:")
    lines.append(f"  worst_pt_recovery:      {result.worst_pt_recovery:.4f}")
    lines.append(f"  worst_capture_mdot:     {result.worst_capture_mdot_kgs:.3f} kg/s")
    lines.append(f"  worst_thrust:           {result.worst_thrust_N/1000.0:.3f} kN")
    lines.append(f"  worst_M4:               {result.worst_M4:.3f}")
    lines.append(f"  worst_Tt4:              {result.worst_Tt4_K:.1f} K")
    lines.append(f"  worst_phi_effective:    {result.worst_phi_effective:.3f}")
    lines.append(f"  worst_max_height:       {result.worst_max_height_m:.3f} m")
    lines.append(f"  combustor_volume:       {result.combustor_volume_m3:.3f} m^3")
    if result.reasons:
        lines.append("")
        lines.append("Infeasibility reasons:")
        for reason in result.reasons:
            lines.append(f"  - {reason}")
    lines.append("")


def save_geometry_report(
    path: str,
    spec: EnvelopeSpec,
    best: CandidateResult | None,
    top: list[CandidateResult],
) -> None:
    """Write a plain-text summary of the selected geometry (if feasible) or
    the closest near-feasible candidates (if none were feasible)."""
    lines: list[str] = []
    lines.append("ENGINE ENVELOPE SEARCH RESULTS")
    lines.append("=" * 78)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("ENVELOPE SPEC")
    lines.append("-" * 78)
    lines.append(f"Mach range:       {spec.mach_min:.2f} - {spec.mach_max:.2f}")
    lines.append(f"Altitude range:   {spec.alt_min_m/1000:.1f} - {spec.alt_max_m/1000:.1f} km")
    lines.append(f"AoA range:        {spec.aoa_min_deg:.1f} - {spec.aoa_max_deg:.1f} deg")
    lines.append(f"Nominal AoA:      {spec.nominal_aoa_deg:.1f} deg")
    lines.append(f"Max length:       {spec.max_total_length_m:.3f} m")
    lines.append(f"Max width:        {spec.max_width_m:.3f} m")
    lines.append(f"Max height:       {spec.max_height_m:.3f} m")
    lines.append(f"Max diameter:     {spec.max_diameter_m:.3f} m")
    lines.append(f"Max combustor D:  {spec.max_combustor_diameter_m:.3f} m")
    lines.append(f"Max nozzle exit:  {spec.max_nozzle_exit_diameter_m:.3f} m")
    lines.append(f"Min combustor V:  {spec.min_combustor_volume_m3:.3f} m^3")
    lines.append(
        f"Min thrust:       {spec.min_thrust_N/1000.0:.2f} kN at phi={spec.phi_for_thrust:.2f}"
    )
    lines.append("")

    if best is not None:
        lines.append("RESULT: FEASIBLE GEOMETRY FOUND")
        lines.append("=" * 78)
        lines.append("")
        _append_candidate_section(lines, best, "SELECTED GEOMETRY")
    else:
        lines.append("RESULT: NO FULLY FEASIBLE GEOMETRY FOUND")
        lines.append("=" * 78)
        lines.append(
            f"Reporting closest {len(top)} near-feasible candidate(s) by score."
        )
        lines.append("")
        for idx, result in enumerate(top, start=1):
            _append_candidate_section(lines, result, f"CANDIDATE #{idx}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    if _IMPORT_ERROR is not None:
        print(
            "Unable to start engine envelope search because a required Python "
            f"dependency is missing: {_IMPORT_ERROR}"
        )
        print(
            "Run this script in the same environment you use for `pyc_run.py` "
            "so `ambiance`, OpenMDAO, and pyCycle dependencies are available."
        )
        return 2

    spec = EnvelopeSpec()
    # ═══════════════════════════════════════════════════════════════════════
    # WORKER COUNT  — how many parallel processes evaluate candidates.
    # Preferred: set the ENGINE_ENVELOPE_WORKERS=N env var before running
    # (e.g. `set ENGINE_ENVELOPE_WORKERS=6` on Windows cmd, or
    # `$env:ENGINE_ENVELOPE_WORKERS=6` in PowerShell).
    # Otherwise: change the "1" default on the next line to your desired
    # worker count.  Start with 1 to confirm correctness, then scale up.
    # ═══════════════════════════════════════════════════════════════════════
    max_workers_env = os.environ.get("ENGINE_ENVELOPE_WORKERS", "4")
    try:
        max_workers = max(1, int(max_workers_env))
    except ValueError:
        print(f"[warn] invalid ENGINE_ENVELOPE_WORKERS={max_workers_env!r}; defaulting to 1")
        max_workers = 1
    print(f"[search] max_workers={max_workers} "
          f"(set ENGINE_ENVELOPE_WORKERS=N to override)")
    best, top = search_envelope(spec, top_n=1, max_workers=max_workers)

    default_report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "envelope_selected_geometry.txt",
    )
    report_path = os.environ.get("ENGINE_ENVELOPE_REPORT_PATH", default_report_path)
    try:
        save_geometry_report(report_path, spec, best, top)
        print(f"[report] wrote geometry summary to {report_path}")
    except OSError as exc:
        print(f"[warn] failed to write geometry report to {report_path}: {exc}")

    print("Engine envelope search")
    print(
        f"Envelope: M={spec.mach_min:.2f}-{spec.mach_max:.2f}, "
        f"alt={spec.alt_min_m/1000:.1f}-{spec.alt_max_m/1000:.1f} km, "
        f"AoA={spec.aoa_min_deg:.1f}-{spec.aoa_max_deg:.1f} deg"
    )
    print(
        f"Geometry limits: L<={spec.max_total_length_m:.2f} m, "
        f"D<={spec.max_diameter_m:.2f} m, "
        f"W<={spec.max_width_m:.3f} m, "
        f"H<={spec.max_height_m:.2f} m, "
        f"D_comb<={spec.max_combustor_diameter_m:.2f} m, "
        f"D_noz_exit<={spec.max_nozzle_exit_diameter_m:.2f} m, "
        f"Vc>={spec.min_combustor_volume_m3:.2f} m^3"
    )
    print(
        f"Performance limit: thrust>={spec.min_thrust_N/1000.0:.2f} kN "
        f"using closure-final phi_effective (requested phi={spec.phi_for_thrust:.2f})"
    )
    print(
        f"Operability rules: no expulsion/swallow, no ram-cycle unstart, "
        f"M4<={M4_MAX:.2f}, Tt4<={TT4_MAX_K:.0f} K"
        + (", no phi clipping" if spec.reject_if_phi_clipped else "")
    )

    if best is None:
        print("\nNo fully feasible geometry found.")
        print("\nBest near-feasible candidates:")
        for idx, result in enumerate(top, start=1):
            geom = result.geometry
            print(
                f"{idx}. score={result.score:.4f} feasible={result.feasible} "
                f"forebody={geom.get('forebody_length_m', float('nan')):.3f} m "
                f"total={geom.get('total_length_m', float('nan')):.3f} m "
                f"diam={geom.get('max_diameter_m', float('nan')):.3f} m "
                f"width={geom.get('max_width_m', float('nan')):.3f} m "
                f"height={geom.get('max_height_m', float('nan')):.3f} m "
                f"Vc={geom.get('combustor_volume_m3', float('nan')):.3f} m^3"
            )
            if result.reasons:
                print(f"   reason: {result.reasons[0]}")
        return 1

    geom = best.geometry
    print("\nBest feasible geometry:")
    print(json.dumps(_serialize_candidate(best), indent=2, sort_keys=True))
    print(
        "\nSummary: "
        f"forebody={geom.get('forebody_length_m', float('nan')):.3f} m, "
        f"total={geom.get('total_length_m', float('nan')):.3f} m, "
        f"diam={geom.get('max_diameter_m', float('nan')):.3f} m, "
        f"width={geom.get('max_width_m', float('nan')):.3f} m, "
        f"height={geom.get('max_height_m', float('nan')):.3f} m, "
        f"Vc={geom.get('combustor_volume_m3', float('nan')):.3f} m^3, "
        f"worst pt/pt0={best.worst_pt_recovery:.3f}, "
        f"worst thrust={best.worst_thrust_N/1000.0:.3f} kN, "
        f"worst M4={best.worst_M4:.3f}, "
        f"worst Tt4={best.worst_Tt4_K:.1f} K, "
        f"worst phi_effective={best.worst_phi_effective:.3f}, "
        f"worst height={best.worst_max_height_m:.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
