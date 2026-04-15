"""
plots_pycycle.py
================
Stand-alone plotting driver for the pyCycle-based cycle (pyc_run.py) plus
the 2-ramp shock-matched inlet (402inlet2.py) and the nozzle contour
generator (nozzle_design.py).

Run directly:
    python plots_pycycle.py

Figures saved to ./figures_pycycle/.
"""

import os
import sys
import argparse
import warnings

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pyc_run
from pyc_config import (
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M, INLET_DESIGN_ALPHA_DEG,
    INLET_DESIGN_WIDTH_M,
    M_TRANSITION, M_MIN, M_MAX,
)
import nozzle_design

# 402inlet2 — module name starts with a digit, import through pyc_run's cache
_inlet2 = pyc_run._inlet2

OUTDIR = os.path.join(_HERE, 'figures_pycycle')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'axes.grid':        True,
    'grid.alpha':       0.30,
    'figure.dpi':       120,
    'lines.linewidth':  1.8,
})

PHI_DEFAULT = 0.8
ALT_DEFAULT = 18_000.0
COMBUSTOR_L_STAR_DEFAULT = 1.5
NOZZLE_CONVERGING_LENGTH_DEFAULT = None
NOZZLE_DIVERGING_LENGTH_DEFAULT = None
NOZZLE_THROAT_ANGLE_DEFAULT = 25.0
NOZZLE_EXIT_ANGLE_DEFAULT = 12.0
NOZZLE_BELL_POINTS_DEFAULT = 240
CAD_WALL_THICKNESS_M = 0.003
CAD_RING_POINTS_DEFAULT = 192
CAD_DIFFUSER_SECTION_COUNT_DEFAULT = 64
CAD_NOZZLE_SECTION_COUNT_DEFAULT = 128
CAD_SILVER_RGBA = np.array([192, 192, 192, 255], dtype=np.uint8)


def _save(fig, name):
    path = os.path.join(OUTDIR, name + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# Mach sweep through pyc_run
# ---------------------------------------------------------------------------

def mach_sweep(mach_range, altitude=ALT_DEFAULT, phi=PHI_DEFAULT):
    results = []
    for M in mach_range:
        try:
            r = pyc_run.analyze(M0=float(M), altitude_m=altitude, phi=phi)
        except Exception as e:
            print(f'  [warn] M={M:.2f} failed: {e}')
            r = None
        results.append(r)
    return results


def _arr(results, key, sub=None):
    out = []
    for r in results:
        if r is None:
            out.append(np.nan); continue
        v = r[key][sub] if sub is not None else r[key]
        out.append(float(v))
    return np.array(out)


def _polyline_segment_lengths(points):
    diffs = np.diff(points, axis=0)
    return np.sqrt(np.sum(diffs * diffs, axis=1))


def _resample_polyline(points, n_samples):
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 1:
        return np.repeat(points, n_samples, axis=0)

    seg_lengths = _polyline_segment_lengths(points)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if s[-1] <= 1.0e-12:
        return np.repeat(points[:1], n_samples, axis=0)

    targets = np.linspace(0.0, s[-1], int(n_samples))
    x = np.interp(targets, s, points[:, 0])
    y = np.interp(targets, s, points[:, 1])
    return np.column_stack([x, y])


def _polyline_y_at_x(points, x_val):
    points = np.asarray(points, dtype=float)
    xs = points[:, 0]
    ys = points[:, 1]

    if x_val <= xs[0]:
        if len(xs) == 1:
            return float(ys[0])
        slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1.0e-12)
        return float(ys[0] + slope * (x_val - xs[0]))
    if x_val >= xs[-1]:
        if len(xs) == 1:
            return float(ys[-1])
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1.0e-12)
        return float(ys[-1] + slope * (x_val - xs[-1]))

    return float(np.interp(x_val, xs, ys))


def _quintic_floor_curve(start_xy, end_xy, start_slope, end_slope, n_points=80):
    x0, y0 = map(float, start_xy)
    x1, y1 = map(float, end_xy)
    dx = x1 - x0
    if dx <= 1.0e-9:
        return np.array([[x0, y0], [x1, y1]], dtype=float)

    m0 = float(start_slope) * dx
    m1 = float(end_slope) * dx
    coeff = np.linalg.solve(
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 2.0, 6.0, 12.0, 20.0],
        ], dtype=float),
        np.array([y0, m0, 0.0, y1, m1, 0.0], dtype=float),
    )
    s = np.linspace(0.0, 1.0, int(n_points))
    y = np.polyval(coeff[::-1], s)
    x = x0 + dx * s
    return np.column_stack([x, y])


def _polyline_slope(points, at_start=False):
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return 0.0
    if at_start:
        p0, p1 = points[0], points[1]
    else:
        p0, p1 = points[-2], points[-1]
    return float((p1[1] - p0[1]) / max(p1[0] - p0[0], 1.0e-12))


def _build_inlet_floor_curve(ramp2_start_xy, foot_xy, throat_lower_xy, diffuser_lower_xy):
    ramp2_start_xy = np.asarray(ramp2_start_xy, dtype=float)
    foot_xy = np.asarray(foot_xy, dtype=float)
    throat_lower_xy = np.asarray(throat_lower_xy, dtype=float)
    diffuser_lower_xy = np.asarray(diffuser_lower_xy, dtype=float)

    if diffuser_lower_xy.shape[0] >= 2:
        end_slope = _polyline_slope(diffuser_lower_xy, at_start=True)
    else:
        end_slope = 0.0

    dx_ramp = foot_xy[0] - ramp2_start_xy[0]
    if abs(dx_ramp) <= 1.0e-12:
        start_slope = end_slope
    else:
        start_slope = float((foot_xy[1] - ramp2_start_xy[1]) / dx_ramp)
    if throat_lower_xy[0] <= foot_xy[0] + 1.0e-9:
        return np.array([foot_xy, throat_lower_xy], dtype=float)
    return _quintic_floor_curve(
        foot_xy,
        throat_lower_xy,
        start_slope=start_slope,
        end_slope=end_slope,
    )


def _flowpath_layout(
    design,
    design_cycle,
    combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
):
    """Shared geometry layout for flowpath and axial-property plots."""
    fore = np.asarray(design['forebody_xy'], dtype=float)
    nose = np.asarray(design['nose_xy'], dtype=float)
    brk2 = np.asarray(design['break2_xy'], dtype=float)
    cowl = np.asarray(design['cowl_lip_xy'], dtype=float)
    foot = np.asarray(design['ramp2_normal_foot_xy'], dtype=float)
    t_up = np.asarray(design['throat_upper_xy'], dtype=float)
    t_lo = np.asarray(design['throat_lower_xy'], dtype=float)

    throat_h  = float(abs(t_up[1] - t_lo[1]))
    throat_x0 = max(t_up[0], t_lo[0])
    y_center  = 0.5 * (t_up[1] + t_lo[1])

    # Subsonic diffuser contour between inlet throat and combustor face.
    diff = design.get('diffuser')
    if diff is None:
        diff_len    = 0.0
        diff_h_exit = throat_h
        diff_upper  = np.array([[throat_x0, y_center + 0.5 * throat_h]])
        diff_lower  = np.array([[throat_x0, y_center - 0.5 * throat_h]])
    else:
        diff_len    = float(diff['length_m'])
        diff_h_exit = float(diff['h_exit'])
        diff_upper  = np.asarray(diff['upper_wall_xy'], dtype=float)
        diff_lower  = np.asarray(diff['lower_wall_xy'], dtype=float)

    inlet_floor = _build_inlet_floor_curve(brk2, foot, t_lo, diff_lower)
    inlet_lower = np.vstack([
        fore,
        nose,
        brk2,
        foot,
        inlet_floor[1:],
    ])
    inlet_upper = np.vstack([cowl, t_up])

    duct_x0   = throat_x0 + diff_len        # combustor face

    combustor = design_cycle.get('combustor_geometry')
    A_throat = float(design_cycle['nozzle_throat_area'])
    if combustor is None or abs(float(combustor.get('L_star', np.nan)) - combustor_L_star) > 1.0e-12:
        combustor = pyc_run.compute_combustor_geometry(
            nozzle_throat_area=A_throat,
            combustor_L_star=combustor_L_star,
            design=design,
        )

    duct_len = float(combustor['length_m'])
    duct_x1 = duct_x0 + duct_len
    duct_area = float(combustor['cross_section_area_m2'])
    duct_radius = float(combustor['radius_m'])

    A_exit = float(design_cycle['nozzle_exit_area'])
    A_inlet = duct_area
    bell = nozzle_design.generate_bell_contour(
        inlet_area=A_inlet,
        throat_area=A_throat,
        exit_area=A_exit,
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=throat_angle_deg,
        exit_angle_deg=exit_angle_deg,
        n_points=n_points,
    )

    x_shift = duct_x1 - bell['x'][0]
    bx = bell['x'] + x_shift

    return {
        'fore': fore,
        'nose': nose,
        'brk2': brk2,
        'cowl': cowl,
        'foot': foot,
        't_up': t_up,
        't_lo': t_lo,
        'inlet_upper': inlet_upper,
        'inlet_lower': inlet_lower,
        'inlet_floor': inlet_floor,
        'throat_h':  throat_h,
        'throat_x0': throat_x0,
        'diff_upper': diff_upper,
        'diff_lower': diff_lower,
        'diff_len':   diff_len,
        'diff_h_exit': diff_h_exit,
        'duct_area': duct_area,
        'duct_radius': duct_radius,
        'duct_x0': duct_x0,
        'duct_len': duct_len,
        'duct_x1': duct_x1,
        'combustor': combustor,
        'A_inlet': A_inlet,
        'A_throat': A_throat,
        'A_exit': A_exit,
        'bell': bell,
        'bx': bx,
        'y_center': y_center,
        'station_x': {
            0: float(fore[0]),
            2: float(throat_x0),
            3: float(duct_x0),
            4: float(duct_x1),
            9: float(bx[-1]),
        },
        'station_labels': {
            0: 'Freestream',
            2: 'Throat',
            3: 'Combustor face',
            4: 'Combustor exit',
            9: 'Nozzle exit',
        },
    }


# ---------------------------------------------------------------------------
# Figures driven by 402inlet2 (reuse its existing plotting functions)
# ---------------------------------------------------------------------------

def fig_inlet_design_detail(design):
    """2-ramp shock-matched inlet geometry at design point."""
    _inlet2.plot_2ramp_shock_matched_inlet(design)
    fig = plt.gcf()
    fig.suptitle(f'2-ramp inlet — design M={INLET_DESIGN_M0}, '
                 f'alt={INLET_DESIGN_ALT_M/1e3:.0f} km, '
                 f'α={INLET_DESIGN_ALPHA_DEG}°', y=1.02)
    _save(fig, 'fig01_inlet_design_detail')


def _design_back_pressure(design):
    """Freestream static pressure at design altitude — sweep p_back default."""
    _, p0, _, _ = _inlet2.std_atmosphere_1976(INLET_DESIGN_ALT_M)
    return float(p0)


def fig_inlet_fixed_grid(design):
    """Off-design 3x3 (Mach × alpha) grid on frozen geometry."""
    mach_vals  = [3.0, 4.0, 5.0]
    alpha_vals = [-2.0, 0.0, 2.0]
    _inlet2.plot_fixed_geometry_3x3_grid(design, INLET_DESIGN_ALT_M,
                                         mach_vals, alpha_vals,
                                         _design_back_pressure(design))
    fig = plt.gcf()
    _save(fig, 'fig02_inlet_fixed_geometry_grid')


def fig_inlet_pt_vs_mach(design):
    """Inlet Pt recovery vs Mach (402inlet2 shock train)."""
    mach_vals = np.linspace(4.0, 5.5, 10)
    cases = _inlet2.sweep_fixed_geometry_vs_mach(
        design, INLET_DESIGN_ALT_M, mach_vals, INLET_DESIGN_ALPHA_DEG,
        _design_back_pressure(design))
    _inlet2.plot_pt_vs_mach(cases, use_immediate_normal=True)
    fig = plt.gcf()
    _save(fig, 'fig03_inlet_pt_vs_mach')


def fig_inlet_pt_vs_alpha(design):
    """Inlet Pt recovery vs angle of attack."""
    alpha_vals = np.linspace(-4.0, 6.0, 21)
    cases = _inlet2.sweep_fixed_geometry_vs_alpha(
        design, INLET_DESIGN_ALT_M, alpha_vals, INLET_DESIGN_M0,
        _design_back_pressure(design))
    _inlet2.plot_pt_vs_alpha(cases, use_immediate_normal=True)
    fig = plt.gcf()
    _save(fig, 'fig04_inlet_pt_vs_alpha')


# ---------------------------------------------------------------------------
# Flowpath to-scale: inlet profile + combustor duct + nozzle bell
# ---------------------------------------------------------------------------

def fig_flowpath(
    design,
    design_cycle,
    combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
):
    """
    Nose-to-tail to-scale flowpath.  Inlet is 2D side view (from 402inlet2);
    combustor and nozzle are axisymmetric after the diffuser exit, with the
    inlet retained as the existing 2D side-view geometry.
    """
    layout = _flowpath_layout(
        design, design_cycle,
        combustor_L_star=combustor_L_star,
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=throat_angle_deg,
        exit_angle_deg=exit_angle_deg,
        n_points=n_points,
    )

    # Inlet geometry (corners and throat)
    nose    = layout['nose']
    brk2    = layout['brk2']
    cowl    = layout['cowl']
    foot    = layout['foot']
    fore    = layout['fore']
    t_up    = layout['t_up']
    t_lo    = layout['t_lo']
    inlet_lower = layout['inlet_lower']
    inlet_upper = layout['inlet_upper']
    inlet_floor = layout['inlet_floor']

    # Subsonic diffuser + combustor duct + nozzle pulled from the shared layout
    diff_upper = layout['diff_upper']
    diff_lower = layout['diff_lower']
    duct_radius = layout['duct_radius']
    duct_x0   = layout['duct_x0']
    duct_len  = layout['duct_len']
    duct_x1   = layout['duct_x1']
    combustor = layout['combustor']
    A_throat  = layout['A_throat']
    A_exit    = layout['A_exit']
    A_inlet   = layout['A_inlet']
    bell      = layout['bell']
    bx        = layout['bx']
    y_center  = layout['y_center']
    r_nozz    = bell['radius']
    duct_h    = 2.0 * duct_radius
    duct_y_lo = y_center - duct_radius
    h_nozz    = 2.0 * r_nozz
    y_nozz_up = y_center + r_nozz
    y_nozz_lo = y_center - r_nozz

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # ── Inlet ramps (lower surface) ─────────────────────────────────────────
    ax.plot(inlet_lower[:, 0], inlet_lower[:, 1], '-', color='steelblue', lw=2.2,
            label='Ramps')

    # Cowl: lip -> throat_upper
    ax.plot(inlet_upper[:, 0], inlet_upper[:, 1], '-', color='firebrick', lw=2.2,
            label='Cowl')

    # Throat line (drawn between upper and lower xy)
    ax.plot([t_up[0], t_lo[0]], [t_up[1], t_lo[1]], '--',
            color='gray', lw=1.2, label='Throat')

    # ── Primary compression shocks (forebody, ramp1, ramp2) + cowl shock ───
    # Forebody/ramp1/ramp2 shocks all converge at the shock focus point (by
    # design). The cowl shock emanates from the cowl lip and impinges on the
    # ramp-2 line at the focus (shock-on-lip) or at F.
    focus  = np.asarray(design['shock_focus_xy'], dtype=float)
    shock_origins = [
        (fore, design['shock_fore_abs_deg'], 'Forebody shock'),
        (nose, design['shock1_abs_deg'],     'Ramp-1 shock'),
        (brk2, design['shock2_abs_deg'],     'Ramp-2 shock'),
    ]
    shock_label_used = False
    for origin, ang_deg, lbl in shock_origins:
        ax.plot([origin[0], focus[0]], [origin[1], focus[1]],
                '-.', color='crimson', lw=1.1,
                label='Compression shocks' if not shock_label_used else None)
        shock_label_used = True

    # Cowl shock: from lip C, directed downward at cowl_shock_abs_deg,
    # impinging on the ramp-2 surface. At the design point the impingement
    # lies at the ramp-2 normal foot F (shock-on-lip closure).
    cowl_shock_end = foot
    ax.plot([cowl[0], cowl_shock_end[0]], [cowl[1], cowl_shock_end[1]],
            '-.', color='darkred', lw=1.3, label='Cowl shock')

    # ── R2 reflection cascade rays (isolator) ───────────────────────────────
    # The cowl shock hits the floor at `cowl_shock_end`. From there, the
    # reflection cascade begins: each reflection is a wall-anchored oblique
    # shock whose angle relative to the incoming flow is beta_rel. We
    # ray-march wall to wall to obtain true bounce geometry.
    refls = design.get('reflection_list', []) or []
    if refls:
        phi_floor = math.radians(design.get('reflection_phi_floor_deg', 0.0))
        phi_roof  = math.radians(design.get('reflection_phi_roof_deg', 0.0))
        # Floor line anchor = foot F (ramp-2 end); roof line anchor = cowl C
        def _ray_wall_intersect(P, dir_rad, Q, wall_rad):
            d = np.array([math.cos(dir_rad), math.sin(dir_rad)])
            w = np.array([math.cos(wall_rad), math.sin(wall_rad)])
            M_mat = np.array([[d[0], -w[0]], [d[1], -w[1]]])
            rhs = np.array([Q[0] - P[0], Q[1] - P[1]])
            try:
                t, _ = np.linalg.solve(M_mat, rhs)
            except np.linalg.LinAlgError:
                return None
            if t <= 0:
                return None
            return P + t * d

        current = np.asarray(cowl_shock_end, dtype=float).copy()
        ray_label_used = False
        for r in refls:
            if r.get('wall') == 'terminal':
                # Terminal normal shock: draw vertical-ish line across duct
                ax.plot([current[0], current[0]], [t_lo[1], t_up[1]],
                        '-', color='darkmagenta', lw=1.6,
                        label='Terminal normal'
                        if not ray_label_used else None)
                ray_label_used = True
                break
            flow_in = math.radians(r.get('flow_dir_in_deg', 0.0))
            beta    = math.radians(r.get('beta_rel_deg', 0.0))
            if r.get('wall') == 'floor':
                # Shock emanates upward from floor into the flow:
                # absolute direction = flow_in + beta
                ray_dir = flow_in + beta
                hit = _ray_wall_intersect(current, ray_dir, cowl,  phi_roof)
            else:
                # Shock emanates downward from roof:
                ray_dir = flow_in - beta
                hit = _ray_wall_intersect(current, ray_dir, foot, phi_floor)
            if hit is None:
                break
            ax.plot([current[0], hit[0]], [current[1], hit[1]],
                    ':', color='darkmagenta', lw=1.1,
                    label='Reflected shocks' if not ray_label_used else None)
            ray_label_used = True
            current = hit

    # ── Subsonic diffuser walls (throat -> combustor face) ─────────────────
    ax.plot(inlet_floor[:, 0], inlet_floor[:, 1],
            '-', color='slateblue', lw=2.0, label='Inlet floor closure')
    if layout['diff_len'] > 0.0:
        ax.plot(diff_upper[:, 0], diff_upper[:, 1],
                '-', color='slateblue', lw=2.0, label='Diffuser')
        ax.plot(diff_lower[:, 0], diff_lower[:, 1],
                '-', color='slateblue', lw=2.0)
        ax.fill_between(diff_upper[:, 0], diff_lower[:, 1], diff_upper[:, 1],
                        color='slateblue', alpha=0.06)

    # ── Combustor duct ──────────────────────────────────────────────────────
    duct = Rectangle((duct_x0, duct_y_lo), duct_len, duct_h,
                     fill=False, ec='darkgreen', lw=2.0)
    ax.add_patch(duct)

    # ── Nozzle bell (axisymmetric about duct centerline) ────────────────────
    ax.plot(bx, y_nozz_up, '-', color='darkorange', lw=2.2, label='Nozzle')
    ax.plot(bx, y_nozz_lo, '-', color='darkorange', lw=2.2)

    # ── Annotations ─────────────────────────────────────────────────────────

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Flowpath (to scale) — design M={INLET_DESIGN_M0}, '
                 f'alt={INLET_DESIGN_ALT_M/1e3:.0f} km')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    _save(fig, 'flowpath_geometry')


# ---------------------------------------------------------------------------
# 3D CAD export (trimesh) — inlet + diffuser + combustor + nozzle shell
# ---------------------------------------------------------------------------

def fig_cad_model(
    design,
    design_cycle,
    wall_thickness_m,
    output_path=None,
    ring_points=CAD_RING_POINTS_DEFAULT,
    diffuser_section_count=CAD_DIFFUSER_SECTION_COUNT_DEFAULT,
    nozzle_section_count=CAD_NOZZLE_SECTION_COUNT_DEFAULT,
    combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
):
    """
    Build a 3D CAD model of the full internal flowpath shell.

    The existing downstream diffuser/combustor/nozzle loft is retained. The
    inlet forebody floor, cowl roof, and a smooth floor closure from ramp 2
    into the throat are appended as prismatic shell surfaces across the inlet
    width. Inlet and exit planes are left open (no end caps).

    Parameters
    ----------
    design            : dict from 402inlet2.design_2ramp_shock_matched_inlet.
    design_cycle      : dict returned by pyc_run.analyze at the design point.
    wall_thickness_m  : shell wall thickness [m].
    output_path       : destination file ('.stl', '.obj', '.ply', '.glb');
                        defaults to <OUTDIR>/engine_cad.stl.

    Returns
    -------
    trimesh.Trimesh  the hollow engine body.
    """
    import trimesh

    t = float(wall_thickness_m)
    if t <= 0.0:
        raise ValueError("wall_thickness_m must be positive.")
    ring_points = int(ring_points)
    diffuser_section_count = int(diffuser_section_count)
    nozzle_section_count = int(nozzle_section_count)
    if ring_points < 24:
        raise ValueError("ring_points must be at least 24.")
    if diffuser_section_count < 3 or nozzle_section_count < 3:
        raise ValueError("section counts must be at least 3.")

    layout = _flowpath_layout(
        design, design_cycle,
        combustor_L_star=combustor_L_star,
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=throat_angle_deg,
        exit_angle_deg=exit_angle_deg,
        n_points=n_points,
    )

    y_center = layout['y_center']
    fore = layout['fore']
    cowl = layout['cowl']
    inlet_lower = layout['inlet_lower']
    inlet_upper = layout['inlet_upper']
    duct_x0 = layout['duct_x0']
    duct_x1 = layout['duct_x1']
    duct_radius = layout['duct_radius']
    bx = layout['bx']
    nozzle_r = layout['bell']['radius']
    diff = design.get('diffuser')

    throat_w = float(diff['throat_width_m']) if diff is not None else float(INLET_DESIGN_WIDTH_M)
    throat_h = float(diff['throat_height_m']) if diff is not None else float(layout['throat_h'])
    diffuser_exit_radius = float(diff['exit_radius_m']) if diff is not None else float(duct_radius)
    half_width = 0.5 * throat_w

    def _ring_to_xyz(x_val, ring_zy):
        y = ring_zy[:, 1] + y_center
        z = ring_zy[:, 0]
        x = np.full_like(y, x_val, dtype=float)
        return np.column_stack([x, y, z])

    def _circle_ring(radius_m, n_theta):
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        return np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta)])

    def _rectangle_ring(width_m, height_m):
        half_w = 0.5 * float(width_m)
        half_h = 0.5 * float(height_m)
        return np.array([
            [-half_w, -half_h],
            [-half_w,  half_h],
            [ half_w,  half_h],
            [ half_w, -half_h],
        ], dtype=float)

    def _morph_ring(area_m2, blend, outer=False, n_theta=None):
        if n_theta is None:
            n_theta = ring_points
        return _inlet2.morphed_rectangle_to_circle_section(
            area_m2=area_m2,
            blend=blend,
            rect_width_m=throat_w + (2.0 * t if outer else 0.0),
            rect_height_m=throat_h + (2.0 * t if outer else 0.0),
            circle_radius_m=diffuser_exit_radius + (t if outer else 0.0),
            n_points=n_theta,
        )

    def _bridge_rings(ring_a, ring_b):
        n = ring_a.shape[0]
        verts = np.vstack([ring_a, ring_b])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + i])
            faces.append([j, n + j, n + i])
        return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)

    def _bridge_polylines(poly_a, poly_b):
        poly_a = np.asarray(poly_a, dtype=float)
        poly_b = np.asarray(poly_b, dtype=float)
        if poly_a.shape != poly_b.shape or poly_a.shape[0] < 2:
            raise ValueError("polylines must have matching shape and at least two points.")
        verts = np.vstack([poly_a, poly_b])
        n = poly_a.shape[0]
        faces = []
        for i in range(n - 1):
            faces.append([i, i + 1, n + i])
            faces.append([i + 1, n + i + 1, n + i])
        return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)

    def _polyline_to_xyz(poly_xy, z_val, y_offset=0.0):
        poly_xy = np.asarray(poly_xy, dtype=float)
        return np.column_stack([
            poly_xy[:, 0],
            poly_xy[:, 1] + y_offset,
            np.full(poly_xy.shape[0], float(z_val), dtype=float),
        ])

    def _wall_shell_from_polyline(poly_xy, is_upper):
        y_offset = t if is_upper else -t
        z_outer = half_width + t
        left_inner = _polyline_to_xyz(poly_xy, -half_width, 0.0)
        right_inner = _polyline_to_xyz(poly_xy, half_width, 0.0)
        left_outer = _polyline_to_xyz(poly_xy, -z_outer, y_offset)
        right_outer = _polyline_to_xyz(poly_xy, z_outer, y_offset)
        return [
            _bridge_polylines(left_inner, right_inner),
            _bridge_polylines(right_outer, left_outer),
            _bridge_polylines(left_outer, left_inner),
            _bridge_polylines(right_inner, right_outer),
        ]

    def _side_shell_from_upper_lower(upper_xy, lower_xy):
        upper_xy = np.asarray(upper_xy, dtype=float)
        lower_xy = np.asarray(lower_xy, dtype=float)
        if upper_xy.shape != lower_xy.shape:
            raise ValueError("upper/lower polylines must share the same sampling.")
        z_outer = half_width + t
        left_upper_inner = _polyline_to_xyz(upper_xy, -half_width, 0.0)
        left_lower_inner = _polyline_to_xyz(lower_xy, -half_width, 0.0)
        right_upper_inner = _polyline_to_xyz(upper_xy, half_width, 0.0)
        right_lower_inner = _polyline_to_xyz(lower_xy, half_width, 0.0)
        left_upper_outer = _polyline_to_xyz(upper_xy, -z_outer, t)
        left_lower_outer = _polyline_to_xyz(lower_xy, -z_outer, -t)
        right_upper_outer = _polyline_to_xyz(upper_xy, z_outer, t)
        right_lower_outer = _polyline_to_xyz(lower_xy, z_outer, -t)
        return [
            _bridge_polylines(left_lower_inner, left_upper_inner),
            _bridge_polylines(left_upper_outer, left_lower_outer),
            _bridge_polylines(left_upper_inner, left_upper_outer),
            _bridge_polylines(left_lower_outer, left_lower_inner),
            _bridge_polylines(right_upper_inner, right_lower_inner),
            _bridge_polylines(right_lower_outer, right_upper_outer),
            _bridge_polylines(right_upper_outer, right_upper_inner),
            _bridge_polylines(right_lower_inner, right_lower_outer),
        ]

    inner_sections = []
    outer_sections = []

    if diff is not None:
        xs = np.asarray(diff['x_stations'], dtype=float)
        areas = np.asarray(diff['A_stations'], dtype=float)
        blends = np.asarray(diff['section_blend_stations'], dtype=float)
        throat_x = float(xs[0])
        inner_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w, throat_h)))
        outer_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w + 2.0 * t, throat_h + 2.0 * t)))
        diffuser_idx = np.linspace(0, len(xs) - 1, diffuser_section_count, dtype=int).tolist()
        diffuser_idx = sorted(set(diffuser_idx))
        if diffuser_idx[-1] != len(xs) - 1:
            diffuser_idx.append(len(xs) - 1)
        for idx in diffuser_idx:
            if idx == 0:
                continue
            inner_sections.append(_ring_to_xyz(float(xs[idx]), _morph_ring(float(areas[idx]), float(blends[idx]), outer=False)))
            outer_sections.append(_ring_to_xyz(float(xs[idx]), _morph_ring(float(areas[idx]), float(blends[idx]), outer=True)))
    else:
        throat_area = float(design['throat_area_actual_m2'])
        inner_sections.append(_ring_to_xyz(duct_x0, _morph_ring(throat_area, 0.0, outer=False)))
        outer_sections.append(_ring_to_xyz(duct_x0, _morph_ring(throat_area, 0.0, outer=True)))

    inner_sections.append(_ring_to_xyz(duct_x1, _circle_ring(duct_radius, ring_points)))
    outer_sections.append(_ring_to_xyz(duct_x1, _circle_ring(duct_radius + t, ring_points)))

    nozzle_idx = np.linspace(0, len(bx) - 1, nozzle_section_count, dtype=int).tolist()
    nozzle_idx = sorted(set(nozzle_idx))
    if nozzle_idx[-1] != len(bx) - 1:
        nozzle_idx.append(len(bx) - 1)
    for idx in nozzle_idx[1:]:
        inner_sections.append(_ring_to_xyz(float(bx[idx]), _circle_ring(float(nozzle_r[idx]), ring_points)))
        outer_sections.append(_ring_to_xyz(float(bx[idx]), _circle_ring(float(nozzle_r[idx]) + t, ring_points)))

    meshes = []
    for ring_a, ring_b in zip(outer_sections[:-1], outer_sections[1:]):
        meshes.append(_bridge_rings(ring_a, ring_b))
    for ring_a, ring_b in zip(inner_sections[:-1], inner_sections[1:]):
        meshes.append(_bridge_rings(ring_b, ring_a))

    inlet_lower_resampled = _resample_polyline(inlet_lower, max(64, diffuser_section_count))
    inlet_upper_resampled = _resample_polyline(inlet_upper, max(32, diffuser_section_count // 2))
    for mesh in _wall_shell_from_polyline(inlet_lower_resampled, is_upper=False):
        meshes.append(mesh)
    for mesh in _wall_shell_from_polyline(inlet_upper_resampled, is_upper=True):
        meshes.append(mesh)

    overlap_x0 = float(cowl[0])
    overlap_x1 = float(layout['t_up'][0])
    if overlap_x1 > overlap_x0 + 1.0e-9:
        side_x = np.linspace(overlap_x0, overlap_x1, max(40, diffuser_section_count))
        side_lower = np.column_stack([
            side_x,
            [_polyline_y_at_x(inlet_lower, x) for x in side_x],
        ])
        side_upper = np.column_stack([
            side_x,
            [_polyline_y_at_x(inlet_upper, x) for x in side_x],
        ])
        for mesh in _side_shell_from_upper_lower(side_upper, side_lower):
            meshes.append(mesh)

    engine = trimesh.util.concatenate(meshes)
    engine.merge_vertices()
    engine.remove_unreferenced_vertices()
    engine.visual.face_colors = CAD_SILVER_RGBA
    engine.visual.vertex_colors = CAD_SILVER_RGBA

    if output_path is None:
        output_path = os.path.join(OUTDIR, 'engine_cad.stl')
    engine.export(output_path)
    print(f'  wrote {output_path}')
    return engine


# ---------------------------------------------------------------------------
# Figures driven by pyc_run Mach sweep
# ---------------------------------------------------------------------------

def fig_performance(results, mach_range):
    """Isp, specific thrust, and net thrust vs Mach."""
    Isp    = _arr(results, 'Isp')
    F_sp   = _arr(results, 'F_sp')
    thrust = _arr(results, 'thrust')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    axes[0].plot(mach_range, Isp, 'o-', color='navy')
    axes[0].set_xlabel('M0'); axes[0].set_ylabel('Isp [s]')
    axes[0].set_title('Specific impulse')

    axes[1].plot(mach_range, F_sp, 's-', color='darkgreen')
    axes[1].set_xlabel('M0'); axes[1].set_ylabel('F/ṁ_air  [N·s/kg_air]')
    axes[1].set_title('Specific thrust')

    axes[2].plot(mach_range, thrust / 1e3, '^-', color='firebrick')
    axes[2].set_xlabel('M0'); axes[2].set_ylabel('Net thrust [kN]')
    axes[2].set_title('Net thrust')

    for ax in axes:
        ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.6,
                   label=f'RAM→SCRAM  M={M_TRANSITION}')
        ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'pyc_run sweep  (alt={ALT_DEFAULT/1e3:.0f} km, φ={PHI_DEFAULT})')
    _save(fig, 'performance_vs_mach')


def fig_mass_flows(results, mach_range):
    mdot_air  = _arr(results, 'mdot_air')
    mdot_fuel = _arr(results, 'mdot_fuel')

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.plot(mach_range, mdot_air,       'o-', color='navy',   label='ṁ_air')
    ax.plot(mach_range, mdot_fuel*1e3,  's-', color='firebrick',
            label='ṁ_fuel × 1000')
    ax.set_xlabel('M0'); ax.set_ylabel('Mass flow [kg/s]')
    ax.set_title('Captured / fuel mass flow vs Mach')
    ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.6)
    ax.legend()
    _save(fig, 'mass_flows')


def fig_station_T(results, mach_range):
    """Static and total temperatures at stations 0, 3, 4, 9."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    stations = [(0, 'Freestream'), (3, 'Inlet exit'),
                (4, 'Burner exit'), (9, 'Nozzle exit')]
    colors = ['navy', 'teal', 'firebrick', 'darkorange']
    for (s, lbl), c in zip(stations, colors):
        ax1.plot(mach_range, _arr(results, 'T_stations',  s),
                 'o-', color=c, label=f'{s}: {lbl}')
        ax2.plot(mach_range, _arr(results, 'Tt_stations', s),
                 'o-', color=c, label=f'{s}: {lbl}')
    for ax, ttl, yl in [(ax1, 'Static T', 'T [K]'),
                        (ax2, 'Total Tt', 'Tt [K]')]:
        ax.set_xlabel('M0'); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.5)
        ax.legend(fontsize=8)
    _save(fig, 'station_temperatures')


def fig_station_Pt(results, mach_range):
    fig, ax = plt.subplots(figsize=(10, 5.0))
    stations = [(0, 'Freestream'), (3, 'Inlet exit'),
                (4, 'Burner exit'), (9, 'Nozzle exit')]
    colors = ['navy', 'teal', 'firebrick', 'darkorange']
    for (s, lbl), c in zip(stations, colors):
        ax.semilogy(mach_range, _arr(results, 'Pt_stations', s) / 1e3,
                    'o-', color=c, label=f'{s}: {lbl}')
    ax.set_xlabel('M0'); ax.set_ylabel('Pt [kPa]')
    ax.set_title('Total pressure through the flowpath')
    ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=8)
    _save(fig, 'station_total_pressures')


def fig_inlet_recovery_cycle(results, mach_range):
    """eta_pt reported by compute_inlet_conditions inside pyc_run."""
    eta = _arr(results, 'eta_pt')
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.plot(mach_range, eta, 'o-', color='steelblue')
    ax.set_xlabel('M0'); ax.set_ylabel('Pt_inlet_exit / Pt_freestream')
    ax.set_title('Inlet total-pressure recovery '
                 '(frozen 2-ramp geometry + isolator)')
    ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.5)
    ax.set_ylim(bottom=0)
    _save(fig, 'inlet_recovery_vs_mach')


def fig_nozzle_geom_vs_mach(results, mach_range):
    A_star = _arr(results, 'nozzle_throat_area')
    A_exit = _arr(results, 'nozzle_exit_area')
    eps    = _arr(results, 'nozzle_area_ratio')
    M9     = _arr(results, 'M_stations', 9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    axes[0].plot(mach_range, A_star*1e4, 'o-', color='firebrick', label='A*')
    axes[0].plot(mach_range, A_exit*1e4, 's-', color='darkorange', label='A_exit')
    axes[0].set_xlabel('M0'); axes[0].set_ylabel('Area [cm²]')
    axes[0].set_title('Nozzle throat & exit area (sized each point)')
    axes[0].legend()

    ax2 = axes[1]
    l1 = ax2.plot(mach_range, eps, 'o-', color='teal',  label='Ae/A*')[0]
    ax2.set_xlabel('M0'); ax2.set_ylabel('Area ratio Ae/A*', color='teal')
    ax2b = ax2.twinx()
    l2 = ax2b.plot(mach_range, M9, 's-', color='firebrick', label='M9')[0]
    ax2b.set_ylabel('Exit Mach  M9', color='firebrick')
    ax2.set_title('Area ratio and exit Mach')
    ax2.legend(handles=[l1, l2], loc='best')
    for ax in (axes[0], ax2):
        ax.axvline(M_TRANSITION, color='gray', ls='--', alpha=0.5)
    _save(fig, 'nozzle_geometry_vs_mach')


def _plot_engine_profile(
    design,
    design_cycle,
    quantity_key,
    total_key,
    ylabel,
    title,
    outfile,
):
    """Plot one property along the engine using the design-point geometry."""
    layout = _flowpath_layout(
        design,
        design_cycle,
        combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
        converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
        diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
        throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
        exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
        n_points=NOZZLE_BELL_POINTS_DEFAULT,
    )

    stations = (0, 3, 4, 9)
    x = np.array([layout['station_x'][s] for s in stations], dtype=float)
    y_static = np.array([design_cycle[quantity_key][s] for s in stations], dtype=float)
    y_total = np.array([design_cycle[total_key][s] for s in stations], dtype=float)

    scale = 1e3 if 'Pa' in ylabel else 1.0
    y_static_plot = y_static / scale
    y_total_plot = y_total / scale

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, y_static_plot, 'o-', color='firebrick', label='Static')
    ax.plot(x, y_total_plot, 's--', color='navy', label='Total')

    for s in stations:
        xs = layout['station_x'][s]
        ax.axvline(xs, color='gray', ls=':', lw=0.9, alpha=0.6)
        ax.text(xs, ax.get_ylim()[1], layout['station_labels'][s],
                rotation=90, ha='right', va='top', fontsize=8, color='gray')

    ax.axvspan(layout['station_x'][3], layout['station_x'][4],
               color='darkgreen', alpha=0.08, label='Combustor')
    ax.axvspan(layout['station_x'][4], layout['station_x'][9],
               color='darkorange', alpha=0.06, label='Nozzle')

    ax.set_xlabel('Axial distance x [m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    _save(fig, outfile)


def fig_engine_pressure_profile(design, design_cycle):
    _plot_engine_profile(
        design,
        design_cycle,
        quantity_key='P_stations',
        total_key='Pt_stations',
        ylabel='Pressure [kPa]',
        title='Pressure Along Engine at Design Point',
        outfile='engine_pressure_profile',
    )


def fig_engine_temperature_profile(design, design_cycle):
    _plot_engine_profile(
        design,
        design_cycle,
        quantity_key='T_stations',
        total_key='Tt_stations',
        ylabel='Temperature [K]',
        title='Temperature Along Engine at Design Point',
        outfile='engine_temperature_profile',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate pyCycle plots and CAD geometry.')
    parser.add_argument(
        '--cad-wall-thickness-mm',
        type=float,
        default=CAD_WALL_THICKNESS_M * 1e3,
        help='CAD shell wall thickness in mm.',
    )
    args = parser.parse_args()
    cad_wall_thickness_m = float(args.cad_wall_thickness_mm) * 1.0e-3
    print('=' * 64)
    print('  plots_pycycle — pyc_run + 402inlet2 + nozzle_design')
    print('=' * 64)

    # Frozen inlet design (cached inside pyc_run)
    design = pyc_run._get_inlet_design()
    print(f'  inlet design success: {design["success"]}')

    # Design-point cycle run (for nozzle throat/exit area in flowpath plot)
    print('  running design-point cycle ...')
    design_cycle = pyc_run.analyze(
        M0=INLET_DESIGN_M0,
        altitude_m=INLET_DESIGN_ALT_M,
        phi=PHI_DEFAULT,
        combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
    )

    # Mach sweep
    mach_range = np.linspace(max(M_MIN, 4.0), min(M_MAX, 5.5), 10)
    print(f'  Mach sweep over {len(mach_range)} points '
          f'at alt={ALT_DEFAULT/1e3:.0f} km, φ={PHI_DEFAULT}')
    results = mach_sweep(mach_range, altitude=ALT_DEFAULT, phi=PHI_DEFAULT)

    print('\n  writing figures:')
    fig_flowpath(
        design,
        design_cycle,
        combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
        converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
        diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
        throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
        exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
        n_points=NOZZLE_BELL_POINTS_DEFAULT,
    )
    fig_performance(results, mach_range)
    fig_mass_flows(results, mach_range)
    fig_station_T(results, mach_range)
    fig_station_Pt(results, mach_range)
    fig_engine_pressure_profile(design, design_cycle)
    fig_engine_temperature_profile(design, design_cycle)

    print(f'\n  generating 3D CAD model (wall={cad_wall_thickness_m*1e3:.1f} mm) ...')
    try:
        fig_cad_model(
            design,
            design_cycle,
            wall_thickness_m=cad_wall_thickness_m,
            combustor_L_star=COMBUSTOR_L_STAR_DEFAULT,
            converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
            diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
            throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
            exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
            n_points=NOZZLE_BELL_POINTS_DEFAULT,
        )
    except Exception as exc:
        import traceback
        print(f'  [warn] CAD export failed: {type(exc).__name__}: {exc}')
        traceback.print_exc()

    print(f'\n  done — figures in {OUTDIR}')


if __name__ == '__main__':
    main()
