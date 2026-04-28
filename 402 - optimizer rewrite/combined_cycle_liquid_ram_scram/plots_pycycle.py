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
import time
import warnings

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import BoundaryNorm, ListedColormap

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pyc_run
from pyc_config import (
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M, INLET_DESIGN_ALPHA_DEG,
    INLET_DESIGN_WIDTH_M, COMBUSTOR_LENGTH_M_DEFAULT, COMBUSTOR_WIDTH_M_DEFAULT,
    INLET_CONSTANT_AREA_LENGTH_M, COMBUSTOR_Y_OFFSET_M,
    M_MIN, M_MAX,
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

from pyc_config import PHI_DEFAULT, INLET_DESIGN_ALT_M
ALT_DEFAULT = INLET_DESIGN_ALT_M #18_000.0
ENVELOPE_MACH_VALUES_DEFAULT = (4.0, 4.5, 5.0)
ENVELOPE_ALT_RANGE_DEFAULT = (19_000.0, 21_500.0)
ENVELOPE_ALPHA_RANGE_DEFAULT = (-1.0, 5.0)
ENVELOPE_ALT_COUNT_DEFAULT = 6
ENVELOPE_ALPHA_COUNT_DEFAULT = 7
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

def mach_sweep(mach_range, altitude=ALT_DEFAULT, phi=PHI_DEFAULT):            #############################################################################
    results = []
    total = len(mach_range)
    for idx, M in enumerate(mach_range, start=1):
        t0 = time.perf_counter()
        try:
            r = pyc_run.analyze(M0=float(M), altitude_m=altitude, phi=phi)
            dt = time.perf_counter() - t0
            print(f'    [{idx:02d}/{total:02d}] M={M:.2f} in {dt:.1f}s')
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f'  [warn] [{idx:02d}/{total:02d}] M={M:.2f} failed after {dt:.1f}s: {e}')
            r = None
        results.append(r)
    return results


def altitude_sweep(alt_range, M0=INLET_DESIGN_M0,
                   alpha_deg=INLET_DESIGN_ALPHA_DEG, phi=PHI_DEFAULT):
    results = []
    total = len(alt_range)
    for idx, alt in enumerate(alt_range, start=1):
        t0 = time.perf_counter()
        try:
            r = pyc_run.analyze(M0=float(M0), altitude_m=float(alt),
                                phi=phi, alpha_deg=alpha_deg)
            dt = time.perf_counter() - t0
            print(f'    [{idx:02d}/{total:02d}] alt={alt/1e3:.2f} km in {dt:.1f}s')
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f'  [warn] [{idx:02d}/{total:02d}] alt={alt/1e3:.2f} km '
                  f'failed after {dt:.1f}s: {e}')
            r = None
        results.append(r)
    return results


def aoa_sweep(alpha_range, M0=INLET_DESIGN_M0,
              altitude_m=INLET_DESIGN_ALT_M, phi=PHI_DEFAULT):
    results = []
    total = len(alpha_range)
    for idx, alpha_deg in enumerate(alpha_range, start=1):
        t0 = time.perf_counter()
        try:
            r = pyc_run.analyze(M0=float(M0), altitude_m=float(altitude_m),
                                phi=phi, alpha_deg=float(alpha_deg))
            dt = time.perf_counter() - t0
            print(f'    [{idx:02d}/{total:02d}] alpha={alpha_deg:+.2f} deg in {dt:.1f}s')
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f'  [warn] [{idx:02d}/{total:02d}] alpha={alpha_deg:+.2f} deg '
                  f'failed after {dt:.1f}s: {e}')
            r = None
        results.append(r)
    return results


def mach_alt_sweep(mach_range, alt_range,
                   alpha_deg=INLET_DESIGN_ALPHA_DEG, phi=PHI_DEFAULT):
    """2D sweep over (M0, altitude) at fixed α and commanded φ.

    Returns a (n_alt, n_mach) object array of analyze() dicts (or None on
    failure), indexed as results[i_alt, j_mach].
    """
    mach_range = np.asarray(mach_range, dtype=float)
    alt_range  = np.asarray(alt_range,  dtype=float)
    results = np.empty((alt_range.size, mach_range.size), dtype=object)
    total = alt_range.size * mach_range.size
    k = 0
    for i, alt in enumerate(alt_range):
        for j, M in enumerate(mach_range):
            k += 1
            t0 = time.perf_counter()
            try:
                r = pyc_run.analyze(M0=float(M), altitude_m=float(alt),
                                    phi=phi, alpha_deg=alpha_deg)
                dt = time.perf_counter() - t0
                print(f'    [{k:03d}/{total:03d}] M={M:.2f} '
                      f'alt={alt/1e3:.2f} km in {dt:.1f}s')
            except Exception as e:
                dt = time.perf_counter() - t0
                print(f'  [warn] [{k:03d}/{total:03d}] M={M:.2f} '
                      f'alt={alt/1e3:.2f} km failed after {dt:.1f}s: {e}')
                r = None
            results[i, j] = r
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


def _build_inlet_floor_curve(ramp2_start_xy, foot_xy, diffuser_lower_xy,
                             blend_fraction_of_ramp2=0.35):
    """Lower-surface straight segment from the ramp-2 foot to diffuser entry."""
    diffuser_lower_xy = np.asarray(diffuser_lower_xy, dtype=float)
    end_xy = np.asarray(diffuser_lower_xy[0], dtype=float)
    foot_xy = np.asarray(foot_xy, dtype=float)
    return np.array([foot_xy, end_xy], dtype=float)


def _build_inlet_roof_curve(ramp2_start_xy, foot_xy, cowl_xy, throat_upper_xy,
                            diffuser_upper_xy):
    """Upper-surface straight segment from the cowl lip to the throat."""
    cowl_xy = np.asarray(cowl_xy, dtype=float)
    throat_upper_xy = np.asarray(throat_upper_xy, dtype=float)
    return np.array([cowl_xy, throat_upper_xy], dtype=float)


def _flowpath_layout(
    design,
    design_cycle,
    combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
    combustor_y_offset_m=0.0,
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
    inlet_center_y  = 0.5 * (t_up[1] + t_lo[1])
    const_area_len = max(0.0, float(INLET_CONSTANT_AREA_LENGTH_M))
    diffuser_entry_x = throat_x0 + const_area_len

    # Subsonic diffuser contour between inlet throat and combustor face.
    diff = design.get('diffuser')
    if diff is None:
        diff_len    = 0.0
        diff_h_exit = throat_h
        diff_upper_raw = np.array([[diffuser_entry_x, inlet_center_y + 0.5 * throat_h]])
        diff_lower_raw = np.array([[diffuser_entry_x, inlet_center_y - 0.5 * throat_h]])
        diff_h_stations = np.array([throat_h], dtype=float)
        diff_x_stations = np.array([diffuser_entry_x], dtype=float)
    else:
        diff_len    = float(diff['length_m'])
        diff_h_exit = float(diff['h_exit'])
        diff_upper_raw = np.asarray(diff['upper_wall_xy'], dtype=float)
        diff_lower_raw = np.asarray(diff['lower_wall_xy'], dtype=float)
        diff_h_stations = np.asarray(diff['h_stations'], dtype=float)
        diff_x_stations = np.asarray(diff['x_stations'], dtype=float) + const_area_len

    # Throat reference is unchanged (inlet/cowl/throat fixed by `design`).
    throat_top_y   = float(diff_upper_raw[0, 1])
    flowpath_top_y = throat_top_y   # legacy alias for downstream consumers

    inlet_floor = np.array([foot, t_lo], dtype=float)
    const_area_upper = np.array([
        t_up,
        np.array([diffuser_entry_x, t_up[1]], dtype=float),
    ])
    const_area_lower = np.array([
        t_lo,
        np.array([diffuser_entry_x, t_lo[1]], dtype=float),
    ])
    # Keep the drawn/CAD lower wall faithful to the design geometry: ramp 2
    # runs all the way to the opening foot ``foot`` before the throat blend
    # begins.
    inlet_lower = np.vstack([
        fore,
        nose,
        brk2,
        inlet_floor,
    ])

    duct_x0   = diffuser_entry_x + diff_len        # combustor face

    combustor = design_cycle.get('combustor_geometry')
    A_throat = float(design_cycle['nozzle_throat_area'])
    if combustor is None or abs(float(combustor.get('length_m', np.nan)) - combustor_length_m) > 1.0e-12:
        combustor = pyc_run.compute_combustor_geometry(
            combustor_length_m=combustor_length_m,
            design=design,
        )

    duct_len = float(combustor['length_m'])
    duct_x1 = duct_x0 + duct_len
    duct_area = float(combustor['cross_section_area_m2'])
    duct_width = float(combustor.get('width_m', COMBUSTOR_WIDTH_M_DEFAULT))
    duct_height = float(combustor['height_m'])
    # Combustor anchored at throat top + vertical offset. Positive offset
    # shifts the combustor (and nozzle) away from the body axis ("down" in the
    # ax.invert_yaxis() display) without touching the upstream geometry.
    duct_top_y    = throat_top_y + float(combustor_y_offset_m)
    duct_bottom_y = duct_top_y - duct_height
    duct_center_y = 0.5 * (duct_top_y + duct_bottom_y)

    # Diffuser walls loft smoothly from the throat-side opening to the
    # offset combustor-face opening. Internal channel height (h_stations) is
    # preserved at every station, so diffuser area-ratio physics is unchanged;
    # only the centerline tilts so the diffuser exit lands at duct_center_y.
    throat_center_y = throat_top_y - 0.5 * float(diff_h_stations[0])
    n_diff = diff_x_stations.shape[0]
    if n_diff > 1:
        x0_d = float(diff_x_stations[0])
        x1_d = float(diff_x_stations[-1])
        t_loft = (diff_x_stations - x0_d) / max(x1_d - x0_d, 1.0e-12)
        # Quintic smoothstep: zero slope and curvature at both endpoints, so
        # the loft blends C2-smoothly into the constant-area section and the
        # combustor face.
        s_loft = t_loft * t_loft * t_loft * (10.0 - 15.0 * t_loft + 6.0 * t_loft * t_loft)
        diff_center_y = throat_center_y + (duct_center_y - throat_center_y) * s_loft
    else:
        diff_center_y = np.array([throat_center_y], dtype=float)
    diff_upper = np.column_stack([
        diff_x_stations,
        diff_center_y + 0.5 * diff_h_stations,
    ])
    diff_lower = np.column_stack([
        diff_x_stations,
        diff_center_y - 0.5 * diff_h_stations,
    ])

    # inlet_roof needs the (now-built) diff_upper to land on its first point.
    inlet_roof  = _build_inlet_roof_curve(brk2, foot, cowl, t_up, diff_upper)
    inlet_upper = inlet_roof

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
        width_m=duct_width,
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
        'inlet_roof': inlet_roof,
        'inlet_floor': inlet_floor,
        'const_area_upper': const_area_upper,
        'const_area_lower': const_area_lower,
        'const_area_len': const_area_len,
        'diffuser_entry_x': diffuser_entry_x,
        'throat_h':  throat_h,
        'throat_x0': throat_x0,
        'diff_upper': diff_upper,
        'diff_lower': diff_lower,
        'diff_x_stations': diff_x_stations,
        'diff_h_stations': diff_h_stations,
        'diff_center_y': diff_center_y,
        'throat_center_y': throat_center_y,
        'diff_len':   diff_len,
        'diff_h_exit': diff_h_exit,
        'duct_area': duct_area,
        'duct_width': duct_width,
        'duct_height': duct_height,
        'duct_top_y': duct_top_y,
        'duct_bottom_y': duct_bottom_y,
        'duct_center_y': duct_center_y,
        'duct_x0': duct_x0,
        'duct_len': duct_len,
        'duct_x1': duct_x1,
        'combustor': combustor,
        'A_inlet': A_inlet,
        'A_throat': A_throat,
        'A_exit': A_exit,
        'bell': bell,
        'bx': bx,
        'flowpath_top_y': flowpath_top_y,
        'inlet_center_y': inlet_center_y,
        'station_x': {
            0: float(fore[0]),
            2: float(throat_x0),
            3: float(duct_x0),
            4: float(duct_x1),
            9: float(bx[-1]),
        },
        'station_labels': {
            0: 'Freestream',
            2: 'Post-cowl',
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


def fig_inlet_fixed_grid(design):
    """Off-design 3x3 (Mach × alpha) grid on frozen geometry."""
    mach_vals  = [3.0, 4.0, 5.0]
    alpha_vals = [-2.0, 0.0, 2.0]
    _inlet2.plot_fixed_geometry_3x3_grid(design, INLET_DESIGN_ALT_M,
                                         mach_vals, alpha_vals)
    fig = plt.gcf()
    _save(fig, 'fig02_inlet_fixed_geometry_grid')


def fig_inlet_pt_vs_mach(design):
    """Inlet Pt recovery vs Mach (402inlet2 shock train)."""
    mach_vals = np.linspace(4.0, 5.5, 10)
    cases = _inlet2.sweep_fixed_geometry_vs_mach(
        design, INLET_DESIGN_ALT_M, mach_vals, INLET_DESIGN_ALPHA_DEG)
    _inlet2.plot_pt_vs_mach(cases)
    fig = plt.gcf()
    _save(fig, 'fig03_inlet_pt_vs_mach')


def fig_inlet_pt_vs_alpha(design):
    """Inlet Pt recovery vs angle of attack."""
    alpha_vals = np.linspace(-4.0, 6.0, 21)
    cases = _inlet2.sweep_fixed_geometry_vs_alpha(
        design, INLET_DESIGN_ALT_M, alpha_vals, INLET_DESIGN_M0)
    _inlet2.plot_pt_vs_alpha(cases)
    fig = plt.gcf()
    _save(fig, 'fig04_inlet_pt_vs_alpha')


# ---------------------------------------------------------------------------
# Flowpath to-scale: inlet profile + combustor duct + nozzle bell
# ---------------------------------------------------------------------------

def print_ramp_geometry(design):
    """Print x,y coordinates of the forebody/ramp1/ramp2 junctions and throat.

    Junction naming follows 402inlet2 convention:
        forebody_xy           – forebody leading edge (start of compression)
        nose_xy               – forebody ends / ramp 1 begins
        break2_xy             – ramp 1 ends / ramp 2 begins
        ramp2_normal_foot_xy  – ramp 2 ends (throat entry, body side)
        throat_lower_xy       – throat corner on body side
        throat_upper_xy       – throat corner on cowl side
        cowl_lip_xy           – cowl leading edge
    """
    pts = [
        ('Forebody start (leading edge)',    design['forebody_xy']),
        ('Forebody ->Ramp 1 junction',       design['nose_xy']),
        ('Ramp 1 ->Ramp 2 junction',         design['break2_xy']),
        ('Ramp 2 end (throat body side)',    design['ramp2_normal_foot_xy']),
        ('Throat lower corner',              design['throat_lower_xy']),
        ('Throat upper corner',              design['throat_upper_xy']),
        ('Cowl lip',                         design['cowl_lip_xy']),
    ]
    # Re-origin to match fig_flowpath: (0, 0) at the forebody tip.
    origin = np.asarray(design['forebody_xy'], dtype=float)
    print('Ramp / inlet geometry points (origin at forebody tip):')
    print(f'  {"point":38s}  {"x [m]":>10s}  {"y [m]":>10s}')
    for name, xy in pts:
        x_shift = float(xy[0]) - float(origin[0])
        y_shift = float(xy[1]) - float(origin[1])
        print(f'  {name:38s}  {x_shift:>10.6f}  {y_shift:>10.6f}')
    print(f'  {"forebody length":38s}  {float(design["forebody_length_m"]):>10.6f}  (m)')
    print(f'  {"ramp 1 length":38s}  {float(design["ramp1_length_m"]):>10.6f}  (m)')


def fig_flowpath(
    design,
    design_cycle,
    combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
    combustor_y_offset_m=COMBUSTOR_Y_OFFSET_M,
):
    """
    Nose-to-tail to-scale flowpath. Inlet is a 2D side view (from 402inlet2);
    the diffuser, combustor, and nozzle remain rectangular downstream.
    """
    print_ramp_geometry(design)
    layout = _flowpath_layout(
        design, design_cycle,
        combustor_length_m=combustor_length_m,
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=throat_angle_deg,
        exit_angle_deg=exit_angle_deg,
        n_points=n_points,
        combustor_y_offset_m=combustor_y_offset_m,
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
    const_area_upper = layout['const_area_upper']
    const_area_lower = layout['const_area_lower']

    # Subsonic diffuser + combustor duct + nozzle pulled from the shared layout
    diff_upper = layout['diff_upper']
    diff_lower = layout['diff_lower']
    duct_width = layout['duct_width']
    duct_height = layout['duct_height']
    duct_top_y = layout['duct_top_y']
    duct_bottom_y = layout['duct_bottom_y']
    duct_x0   = layout['duct_x0']
    duct_len  = layout['duct_len']
    duct_x1   = layout['duct_x1']
    combustor = layout['combustor']
    A_throat  = layout['A_throat']
    A_exit    = layout['A_exit']
    A_inlet   = layout['A_inlet']
    bell      = layout['bell']
    bx        = layout['bx']
    duct_center_y = layout['duct_center_y']
    y_nozz_up = duct_center_y + bell['upper_wall']
    y_nozz_lo = duct_center_y + bell['lower_wall']

    # Re-origin all drawn coordinates so (0, 0) sits at the forebody tip. This
    # is a plot-only translation — _flowpath_layout and the CAD/engine-profile
    # consumers still use the original absolute coordinates.
    origin_x = float(fore[0])
    origin_y = float(fore[1])
    def _sh(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return np.array([a[0] - origin_x, a[1] - origin_y])
        return a - np.array([origin_x, origin_y])

    inlet_lower      = _sh(inlet_lower)
    inlet_upper      = _sh(inlet_upper)
    inlet_floor      = _sh(inlet_floor)
    const_area_upper = _sh(const_area_upper)
    const_area_lower = _sh(const_area_lower)
    diff_upper       = _sh(diff_upper)
    diff_lower       = _sh(diff_lower)
    t_up             = _sh(t_up)
    t_lo             = _sh(t_lo)
    fore             = _sh(fore)   # becomes (0, 0)
    nose             = _sh(nose)
    brk2             = _sh(brk2)
    cowl             = _sh(cowl)
    foot             = _sh(foot)
    swallowed_boundary = _sh(np.asarray(
        design.get('swallowed_boundary_xy', [[np.nan, np.nan]]), dtype=float))
    swallowed_upper  = _sh(np.asarray(
        design.get('upper_capture_swallowed_xy', [np.nan, np.nan]), dtype=float))
    swallowed_lower  = _sh(np.asarray(
        design.get('lower_capture_swallowed_xy', [np.nan, np.nan]), dtype=float))
    duct_x0       -= origin_x
    duct_x1       -= origin_x
    duct_bottom_y -= origin_y
    bx             = bx - origin_x
    y_nozz_up      = y_nozz_up - origin_y
    y_nozz_lo      = y_nozz_lo - origin_y

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # ── Inlet ramps (lower surface) ─────────────────────────────────────────
    ax.plot(inlet_lower[:, 0], inlet_lower[:, 1], '-', color='steelblue', lw=2.2,
            label='Ramps')

    # Cowl: lip -> throat_upper
    ax.plot(inlet_upper[:, 0], inlet_upper[:, 1], '-', color='firebrick', lw=2.2,
            label='Cowl')

    # Throat line (drawn between upper and lower xy)

    # ── Primary compression shocks (forebody, ramp1, ramp2) + cowl shock ───
    # Ramp-1 and ramp-2 shocks still converge at the stored external-shock
    # focus. The forebody shock is now drawn as its own ray from the forebody
    # leading edge so an explicit forebody length can decouple it from that
    # common focus.
    focus  = _sh(np.asarray(design['shock_focus_xy'], dtype=float))
    lower_wall_parts = [inlet_floor]
    if layout['const_area_len'] > 0.0:
        lower_wall_parts.append(const_area_lower[1:])
    if layout['diff_len'] > 0.0:
        lower_wall_parts.append(diff_lower[1:])
    lower_wall = np.vstack(lower_wall_parts)

    upper_wall_parts = [inlet_upper]
    if layout['const_area_len'] > 0.0:
        upper_wall_parts.append(const_area_upper[1:])
    if layout['diff_len'] > 0.0:
        upper_wall_parts.append(diff_upper[1:])
    upper_wall = np.vstack(upper_wall_parts)

    shock_segments = _inlet2.build_internal_shock_segments(
        C=cowl,
        F=foot,
        T_upper=t_up,
        T_lower=t_lo,
        cowl_shock_abs_deg=design['cowl_shock_abs_deg'],
        lower_wall_xy=lower_wall,
        upper_wall_xy=upper_wall,
    )
    x_end = 1.05 * max(
        float(np.max(inlet_upper[:, 0])),
        float(np.max(inlet_lower[:, 0])),
        float(focus[0]),
    )
    fore_shock_dir = _inlet2.unit_from_angle_deg(design['shock_fore_abs_deg'])
    lam_fore_end = (x_end - fore[0]) / fore_shock_dir[0]
    fore_shock_end = fore + lam_fore_end * fore_shock_dir

    # Legacy plotting kept for reference: all three compression shocks were
    # drawn to the common external-shock focus.
    # shock_origins = [
    #     (fore, design['shock_fore_abs_deg'], 'Forebody shock'),
    #     (nose, design['shock1_abs_deg'],     'Ramp-1 shock'),
    #     (brk2, design['shock2_abs_deg'],     'Ramp-2 shock'),
    # ]
    # shock_label_used = False
    # for origin, ang_deg, lbl in shock_origins:
    #     ax.plot([origin[0], focus[0]], [origin[1], focus[1]],
    #             '-.', color='crimson', lw=1.1,
    #             label='Compression shocks' if not shock_label_used else None)
    #     shock_label_used = True

    ax.plot([fore[0], fore_shock_end[0]], [fore[1], fore_shock_end[1]],
            '-.', color='crimson', lw=1.1, label='Compression shocks')
    ax.plot([nose[0], focus[0]], [nose[1], focus[1]],
            '-.', color='crimson', lw=1.1)
    ax.plot([brk2[0], focus[0]], [brk2[1], focus[1]],
            '-.', color='crimson', lw=1.1)

    if swallowed_boundary.ndim == 2 and swallowed_boundary.shape[0] >= 2 and np.all(np.isfinite(swallowed_boundary)):
        ax.plot(swallowed_boundary[:, 0], swallowed_boundary[:, 1],
                '--', color='teal', lw=1.6, label='Swallowed streamtube boundary')
    if np.all(np.isfinite(swallowed_upper)) and np.all(np.isfinite(swallowed_lower)):
        ax.plot([swallowed_lower[0], swallowed_upper[0]],
                [swallowed_lower[1], swallowed_upper[1]],
                '-', color='teal', lw=2.0, alpha=0.9, label='Freestream capture segment')

    cowl_label_used = False
    for seg in shock_segments:
        start = seg['start']
        end = seg['end']
        if seg['kind'] == 'cowl':
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    '-.', color='darkred', lw=1.3,
                    label='Cowl shock' if not cowl_label_used else None)
            cowl_label_used = True

    # ── Subsonic diffuser walls (throat -> combustor face) ─────────────────
    ax.plot(inlet_floor[:, 0], inlet_floor[:, 1],
            '-', color='slateblue', lw=2.0, label='Inlet floor closure')
    if layout['const_area_len'] > 0.0:
        ax.plot(const_area_upper[:, 0], const_area_upper[:, 1],
                '-', color='slateblue', lw=2.0)
        ax.plot(const_area_lower[:, 0], const_area_lower[:, 1],
                '-', color='slateblue', lw=2.0)
    if layout['diff_len'] > 0.0:
        ax.plot(diff_upper[:, 0], diff_upper[:, 1],
                '-', color='slateblue', lw=2.0, label='Diffuser')
        ax.plot(diff_lower[:, 0], diff_lower[:, 1],
                '-', color='slateblue', lw=2.0)

    # ── Combustor duct ──────────────────────────────────────────────────────
    duct = Rectangle((duct_x0, duct_bottom_y), duct_len, duct_height,
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
    ax.invert_yaxis()
    #ax.legend(loc='upper left')
    _save(fig, 'flowpath_geometry')


# ---------------------------------------------------------------------------
# 3D CAD export (trimesh) — inlet + diffuser + combustor + nozzle shell
# ---------------------------------------------------------------------------

def _export_step_from_trimesh_mesh(mesh, output_path):
    """
    Export a triangulated trimesh mesh to STEP via an optional cadquery/OCP
    backend by wrapping each triangle as a planar B-rep face.

    This is intentionally a best-effort path: it preserves geometry for CAD
    interchange when a STEP-capable backend is installed, while keeping the
    default mesh-export workflow dependency-light.
    """
    try:
        import cadquery as cq
    except Exception as exc:
        raise RuntimeError(
            "STEP export requires cadquery with its OCP/OpenCascade backend "
            "installed. STL/OBJ/PLY export is still available."
        ) from exc

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cq_faces = []

    for tri in faces:
        pts = [cq.Vector(*vertices[int(idx)]) for idx in tri]
        edges = [
            cq.Edge.makeLine(pts[0], pts[1]),
            cq.Edge.makeLine(pts[1], pts[2]),
            cq.Edge.makeLine(pts[2], pts[0]),
        ]
        wire = cq.Wire.assembleEdges(edges)
        cq_faces.append(cq.Face.makeFromWires(wire))

    shape = cq.Compound.makeCompound(cq_faces)
    cq.exporters.export(shape, output_path)


def _export_cad_mesh(engine, output_path):
    ext = os.path.splitext(str(output_path))[1].lower()
    if ext in ('.stp', '.step'):
        _export_step_from_trimesh_mesh(engine, output_path)
        return
    engine.export(output_path)


def fig_cad_model(
    design,
    design_cycle,
    wall_thickness_m,
    output_path=None,
    extra_output_paths=None,
    ring_points=CAD_RING_POINTS_DEFAULT,
    diffuser_section_count=CAD_DIFFUSER_SECTION_COUNT_DEFAULT,
    nozzle_section_count=CAD_NOZZLE_SECTION_COUNT_DEFAULT,
    combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
    converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
    diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
    throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
    exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
    n_points=NOZZLE_BELL_POINTS_DEFAULT,
    combustor_y_offset_m=COMBUSTOR_Y_OFFSET_M,
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
    output_path       : primary destination file ('.stl', '.obj', '.ply', '.glb',
                        '.stp', '.step');
                        defaults to <OUTDIR>/engine_cad.stl.
    extra_output_paths: optional iterable of additional export paths.

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
        combustor_length_m=combustor_length_m,
        converging_length=converging_length,
        diverging_length=diverging_length,
        throat_angle_deg=throat_angle_deg,
        exit_angle_deg=exit_angle_deg,
        n_points=n_points,
        combustor_y_offset_m=combustor_y_offset_m,
    )

    flowpath_top_y = layout['flowpath_top_y']
    duct_center_y = layout['duct_center_y']
    fore = layout['fore']
    cowl = layout['cowl']
    inlet_lower = layout['inlet_lower']
    inlet_upper = layout['inlet_upper']
    duct_x0 = layout['duct_x0']
    duct_x1 = layout['duct_x1']
    duct_width = layout['duct_width']
    duct_height = layout['duct_height']
    bx = layout['bx']
    nozzle_h = layout['bell']['height']
    const_area_len = layout['const_area_len']
    diffuser_entry_x = layout['diffuser_entry_x']
    diff = design.get('diffuser')

    throat_w = float(diff['throat_width_m']) if diff is not None else float(INLET_DESIGN_WIDTH_M)
    throat_h = float(diff['throat_height_m']) if diff is not None else float(layout['throat_h'])
    half_width = 0.5 * throat_w

    def _ring_to_xyz(x_val, ring_zy, center_y):
        y = ring_zy[:, 1] + center_y
        z = ring_zy[:, 0]
        x = np.full_like(y, x_val, dtype=float)
        return np.column_stack([x, y, z])

    def _rectangle_ring(width_m, height_m, outer=False, n_theta=None):
        if n_theta is None:
            n_theta = ring_points
        width_m = float(width_m)
        height_m = float(height_m)
        if outer:
            width_m += 2.0 * t
            height_m += 2.0 * t
        n_side = max(2, n_theta // 4)
        n_last = n_theta - 3 * n_side
        top = np.column_stack([
            np.linspace(-0.5 * width_m, 0.5 * width_m, n_side, endpoint=False),
            np.full(n_side, 0.5 * height_m, dtype=float),
        ])
        right = np.column_stack([
            np.full(n_side, 0.5 * width_m, dtype=float),
            np.linspace(0.5 * height_m, -0.5 * height_m, n_side, endpoint=False),
        ])
        bottom = np.column_stack([
            np.linspace(0.5 * width_m, -0.5 * width_m, n_side, endpoint=False),
            np.full(n_side, -0.5 * height_m, dtype=float),
        ])
        left = np.column_stack([
            np.full(n_last, -0.5 * width_m, dtype=float),
            np.linspace(-0.5 * height_m, 0.5 * height_m, n_last, endpoint=False),
        ])
        return np.vstack([top, right, bottom, left])

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
        xs = np.asarray(layout['diff_x_stations'], dtype=float)
        heights = np.asarray(layout['diff_h_stations'], dtype=float)
        widths = np.asarray(diff['width_stations'], dtype=float)
        diff_center_y_arr = np.asarray(layout['diff_center_y'], dtype=float)
        throat_x = float(layout['throat_x0'])
        # Throat + constant-area sections sit on the unchanged throat center
        # (inlet/throat geometry is fixed regardless of combustor offset).
        throat_center_y = float(layout['throat_center_y'])
        inner_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w, throat_h, outer=False), throat_center_y))
        outer_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w, throat_h, outer=True), throat_center_y))
        if const_area_len > 0.0:
            inner_sections.append(_ring_to_xyz(diffuser_entry_x, _rectangle_ring(throat_w, throat_h, outer=False), throat_center_y))
            outer_sections.append(_ring_to_xyz(diffuser_entry_x, _rectangle_ring(throat_w, throat_h, outer=True), throat_center_y))
        diffuser_idx = np.linspace(0, len(xs) - 1, diffuser_section_count, dtype=int).tolist()
        diffuser_idx = sorted(set(diffuser_idx))
        if diffuser_idx[-1] != len(xs) - 1:
            diffuser_idx.append(len(xs) - 1)
        for idx in diffuser_idx:
            if idx == 0:
                continue
            # Diffuser sections track the lofted centerline so they meet
            # the combustor face at duct_center_y when an offset is applied.
            section_center_y = float(diff_center_y_arr[idx])
            inner_sections.append(_ring_to_xyz(float(xs[idx]), _rectangle_ring(float(widths[idx]), float(heights[idx]), outer=False), section_center_y))
            outer_sections.append(_ring_to_xyz(float(xs[idx]), _rectangle_ring(float(widths[idx]), float(heights[idx]), outer=True), section_center_y))
    else:
        throat_x = float(layout['throat_x0'])
        inner_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w, throat_h, outer=False), duct_center_y))
        outer_sections.append(_ring_to_xyz(throat_x, _rectangle_ring(throat_w, throat_h, outer=True), duct_center_y))
        if const_area_len > 0.0:
            inner_sections.append(_ring_to_xyz(diffuser_entry_x, _rectangle_ring(throat_w, throat_h, outer=False), duct_center_y))
            outer_sections.append(_ring_to_xyz(diffuser_entry_x, _rectangle_ring(throat_w, throat_h, outer=True), duct_center_y))

    inner_sections.append(_ring_to_xyz(duct_x1, _rectangle_ring(duct_width, duct_height, outer=False), duct_center_y))
    outer_sections.append(_ring_to_xyz(duct_x1, _rectangle_ring(duct_width, duct_height, outer=True), duct_center_y))

    nozzle_idx = np.linspace(0, len(bx) - 1, nozzle_section_count, dtype=int).tolist()
    nozzle_idx = sorted(set(nozzle_idx))
    if nozzle_idx[-1] != len(bx) - 1:
        nozzle_idx.append(len(bx) - 1)
    for idx in nozzle_idx[1:]:
        section_center_y = duct_center_y
        inner_sections.append(_ring_to_xyz(float(bx[idx]), _rectangle_ring(duct_width, float(nozzle_h[idx]), outer=False), section_center_y))
        outer_sections.append(_ring_to_xyz(float(bx[idx]), _rectangle_ring(duct_width, float(nozzle_h[idx]), outer=True), section_center_y))

    meshes = []
    for ring_a, ring_b in zip(outer_sections[:-1], outer_sections[1:]):
        meshes.append(_bridge_rings(ring_a, ring_b))
    for ring_a, ring_b in zip(inner_sections[:-1], inner_sections[1:]):
        meshes.append(_bridge_rings(ring_b, ring_a))

    inlet_lower_resampled = _resample_polyline(inlet_lower, max(128, 2 * diffuser_section_count))
    inlet_upper_resampled = _resample_polyline(inlet_upper, max(96, diffuser_section_count))
    for mesh in _wall_shell_from_polyline(inlet_lower_resampled, is_upper=False):
        meshes.append(mesh)
    for mesh in _wall_shell_from_polyline(inlet_upper_resampled, is_upper=True):
        meshes.append(mesh)

    overlap_x0 = float(cowl[0])
    overlap_x1 = float(inlet_upper[-1, 0])
    if overlap_x1 > overlap_x0 + 1.0e-9:
        side_x = np.linspace(overlap_x0, overlap_x1, max(96, 2 * diffuser_section_count))
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

    export_paths = []
    if output_path is None:
        output_path = os.path.join(OUTDIR, 'engine_cad.stl')
    export_paths.append(output_path)
    for path in (extra_output_paths or []):
        if path is None:
            continue
        if path not in export_paths:
            export_paths.append(path)

    for path in export_paths:
        try:
            _export_cad_mesh(engine, path)
            print(f'  wrote {path}')
        except Exception as exc:
            print(f'  [warn] CAD export skipped for {path}: {type(exc).__name__}: {exc}')
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
        ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'pyc_run sweep  (alt={ALT_DEFAULT/1e3:.0f} km, φ={PHI_DEFAULT})')
    _save(fig, 'performance_vs_mach')


def fig_performance_vs_alt(results, alt_range,
                           M0=INLET_DESIGN_M0,
                           alpha_deg=INLET_DESIGN_ALPHA_DEG,
                           phi=PHI_DEFAULT):
    """Isp, specific thrust, and net thrust vs altitude at fixed M0/α/φ."""
    Isp    = _arr(results, 'Isp')
    F_sp   = _arr(results, 'F_sp')
    thrust = _arr(results, 'thrust')
    alt_km = np.asarray(alt_range, dtype=float) / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    axes[0].plot(alt_km, Isp, 'o-', color='navy')
    axes[0].set_xlabel('Altitude [km]'); axes[0].set_ylabel('Isp [s]')
    axes[0].set_title('Specific impulse')

    axes[1].plot(alt_km, F_sp, 's-', color='darkgreen')
    axes[1].set_xlabel('Altitude [km]')
    axes[1].set_ylabel('F/ṁ_air  [N·s/kg_air]')
    axes[1].set_title('Specific thrust')

    axes[2].plot(alt_km, thrust / 1e3, '^-', color='firebrick')
    axes[2].set_xlabel('Altitude [km]'); axes[2].set_ylabel('Net thrust [kN]')
    axes[2].set_title('Net thrust')

    for ax in axes:
        ax.axvline(INLET_DESIGN_ALT_M / 1e3, color='gray',
                   linestyle=':', lw=1.0, label='design alt')
        ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'pyc_run altitude sweep  '
                 f'(M0={M0}, α={alpha_deg}°, φ={phi})')
    _save(fig, 'performance_vs_altitude')


def fig_performance_vs_aoa(results, alpha_range,
                           M0=INLET_DESIGN_M0,
                           altitude_m=INLET_DESIGN_ALT_M,
                           phi=PHI_DEFAULT):
    """Fixed-Mach/fixed-altitude AoA sweep with unstart/choke markers."""
    alpha_range = np.asarray(alpha_range, dtype=float)
    thrust = _arr(results, 'thrust') / 1e3
    eta_pt = _arr(results, 'eta_pt')
    M4 = np.array([
        float(r['M_stations'].get(4, np.nan)) if r is not None else np.nan
        for r in results
    ], dtype=float)
    phi_eff = _arr(results, 'phi_effective')
    unstart = _arr(results, 'unstart_flag')
    choked = _arr(results, 'choked')

    mask_unstart = np.isfinite(unstart) & (unstart > 0.5)
    mask_choked = np.isfinite(choked) & (choked > 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    axes = axes.ravel()
    series = [
        (thrust, 'Net thrust [kN]', 'Net thrust', 'firebrick', 'o-'),
        (eta_pt, 'Pt3/Pt0 [-]', 'Inlet recovery', 'navy', 's-'),
        (M4, 'M4 [-]', 'Combustor exit Mach', 'darkgreen', '^-'),
        (phi_eff, 'Effective phi [-]', 'Effective equivalence ratio', 'darkorange', 'd-'),
    ]

    for ax, (y, ylabel, title, color, style) in zip(axes, series):
        ax.plot(alpha_range, y, style, color=color, label='response')
        if np.any(mask_unstart):
            ax.plot(alpha_range[mask_unstart], y[mask_unstart], 'x',
                    color='crimson', ms=8, mew=2, linestyle='None',
                    label='unstart')
        if np.any(mask_choked):
            ax.plot(alpha_range[mask_choked], y[mask_choked], 'o',
                    mfc='none', mec='black', ms=8, mew=1.5, linestyle='None',
                    label='choked')
        ax.axvline(INLET_DESIGN_ALPHA_DEG, color='gray',
                   linestyle=':', lw=1.0, label='design AoA')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[2].set_xlabel('Angle of attack [deg]')
    axes[3].set_xlabel('Angle of attack [deg]')

    handles, labels = axes[0].get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup.setdefault(l, h)
    axes[0].legend(dedup.values(), dedup.keys(), loc='best', fontsize=8)

    fig.suptitle(f'pyc_run AoA sweep  '
                 f'(M0={M0}, alt={altitude_m/1e3:.1f} km, phi={phi})')
    _save(fig, 'performance_vs_aoa')


def fig_performance_along_climb_path(design, phi=PHI_DEFAULT,
                                     mach_start=4.0, mach_end=5.0,
                                     alt_start_m=16_000.0, alt_end_m=19_000.0,
                                     alpha_deg=2.0, n_points=7):
    """
    Fixed-geometry performance map along a straight (M0, altitude) path.

    Geometry is frozen at the supplied design point; only the flight condition
    changes from (mach_start, alt_start_m) to (mach_end, alt_end_m).
    """
    mach_values = np.linspace(float(mach_start), float(mach_end), int(n_points))
    alt_values_m = np.linspace(float(alt_start_m), float(alt_end_m), int(n_points))

    results = []
    for M0, altitude_m in zip(mach_values, alt_values_m):
        try:
            r = pyc_run.analyze(
                M0=float(M0),
                altitude_m=float(altitude_m),
                phi=phi,
                alpha_deg=float(alpha_deg),
                design=design,
            )
        except Exception as exc:
            print(f'  [warn] climb-path point failed at M0={M0:.3f}, alt={altitude_m/1e3:.3f} km: '
                  f'{type(exc).__name__}: {exc}')
            r = None
        results.append(r)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    mach_ok = []
    alt_ok_km = []
    thrust_ok_kn = []
    isp_ok = []
    for M0, altitude_m, r in zip(mach_values, alt_values_m, results):
        if r is None:
            continue
        mach_ok.append(float(M0))
        alt_ok_km.append(float(altitude_m) / 1e3)
        thrust_ok_kn.append(float(r['thrust']) / 1e3)
        isp_ok.append(float(r['Isp']))

    ax.plot(mach_values, alt_values_m / 1e3, '--', color='lightgray', lw=1.2,
            label='Requested path')
    ax.scatter(mach_ok, alt_ok_km, s=56, color='navy', zorder=3,
               label='Fixed-geometry evaluations')

    for x, y, thrust_kn, isp_s in zip(mach_ok, alt_ok_km, thrust_ok_kn, isp_ok):
        ax.annotate(f'{thrust_kn:.1f} kN\n{isp_s:.0f} s',
                    xy=(x, y), xytext=(0, 9), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.9))

    ax.set_xlabel('M0')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'Fixed-Geometry Performance Along Climb Path (α={alpha_deg:.1f}°)')
    ax.legend(loc='best', fontsize=8)
    _save(fig, 'performance_along_climb_path')


def fig_phi_vs_mach_alt(results_2d, mach_range, alt_range,
                        alpha_deg=INLET_DESIGN_ALPHA_DEG,
                        phi_request=PHI_DEFAULT):
    """Effective (clipped) equivalence ratio over the (M0, altitude) grid.

    The pyCycle stack commands phi_request, but _solve_phi_envelope soft-mins
    it against the Tt4/choke/inlet-expulsion caps. phi_effective is what the
    engine actually burns — lower than phi_request wherever an operability
    cap binds.
    """
    mach_range = np.asarray(mach_range, dtype=float)
    alt_range  = np.asarray(alt_range,  dtype=float)
    phi_eff = np.full(results_2d.shape, np.nan, dtype=float)
    for i in range(results_2d.shape[0]):
        for j in range(results_2d.shape[1]):
            r = results_2d[i, j]
            if r is not None:
                phi_eff[i, j] = float(r.get('phi_effective', r.get('phi', np.nan)))

    fig, ax = plt.subplots(figsize=(10, 6.0))
    M_grid, A_grid = np.meshgrid(mach_range, alt_range / 1e3)
    pcm = ax.pcolormesh(M_grid, A_grid, phi_eff,
                        shading='auto', cmap='viridis',
                        vmin=0.0, vmax=max(float(phi_request), float(np.nanmax(phi_eff)) if np.isfinite(np.nanmax(phi_eff)) else float(phi_request)))
    cs = ax.contour(M_grid, A_grid, phi_eff,
                    levels=np.linspace(0.1, float(phi_request), 7),
                    colors='white', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.plot(INLET_DESIGN_M0, INLET_DESIGN_ALT_M / 1e3,
            marker='*', color='red', markersize=14,
            markeredgecolor='black', label='Design point')
    ax.set_xlabel('M0')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'Effective equivalence ratio φ_eff '
                 f'(α={alpha_deg}°, φ_request={phi_request})')
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(pcm, ax=ax, label='φ_effective')
    _save(fig, 'phi_vs_mach_alt')


def fig_mass_flows(results, mach_range):
    mdot_air  = _arr(results, 'mdot_air')
    mdot_fuel = _arr(results, 'mdot_fuel')

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.plot(mach_range, mdot_air,       'o-', color='navy',   label='ṁ_air')
    ax.plot(mach_range, mdot_fuel*1e3,  's-', color='firebrick',
            label='ṁ_fuel × 1000')
    ax.set_xlabel('M0'); ax.set_ylabel('Mass flow [kg/s]')
    ax.set_title('Captured / fuel mass flow vs Mach')
    ax.legend()
    _save(fig, 'mass_flows')


def fig_station_T(results, mach_range):
    """Static and total temperatures at stations 0, 2, 3, 4, 9."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    stations = [(0, 'Freestream'), (2, 'Post-cowl'),
                (3, 'Combustor face'), (4, 'Burner exit'),
                (9, 'Nozzle exit')]
    colors = ['navy', 'slateblue', 'teal', 'firebrick', 'darkorange']
    for (s, lbl), c in zip(stations, colors):
        ax1.plot(mach_range, _arr(results, 'T_stations',  s),
                 'o-', color=c, label=f'{s}: {lbl}')
        ax2.plot(mach_range, _arr(results, 'Tt_stations', s),
                 'o-', color=c, label=f'{s}: {lbl}')
    for ax, ttl, yl in [(ax1, 'Static T', 'T [K]'),
                        (ax2, 'Total Tt', 'Tt [K]')]:
        ax.set_xlabel('M0'); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=8)
    _save(fig, 'station_temperatures')


def fig_station_Pt(results, mach_range):
    fig, ax = plt.subplots(figsize=(10, 5.0))
    stations = [(0, 'Freestream'), (2, 'Post-cowl'),
                (3, 'Combustor face'), (4, 'Burner exit'),
                (9, 'Nozzle exit')]
    colors = ['navy', 'slateblue', 'teal', 'firebrick', 'darkorange']
    for (s, lbl), c in zip(stations, colors):
        ax.semilogy(mach_range, _arr(results, 'Pt_stations', s) / 1e3,
                    'o-', color=c, label=f'{s}: {lbl}')
    ax.set_xlabel('M0'); ax.set_ylabel('Pt [kPa]')
    ax.set_title('Total pressure through the flowpath')
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
    ax.set_ylim(bottom=0)
    _save(fig, 'inlet_recovery_vs_mach')


def fig_ram_diagnostics(results, mach_range):
    """Combustor choke state and inlet-expulsion flag through the RAM sweep."""
    choked = _arr(results, 'choked')
    unstart = _arr(results, 'unstart_flag')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    ax1.plot(mach_range, choked, 'o-', color='firebrick')
    ax1.set_xlabel('M0')
    ax1.set_ylabel('Choked flag')
    ax1.set_title('Combustor Choked vs Mach')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0.0, 1.0])

    ax2.plot(mach_range, unstart, 'o-', color='slateblue')
    ax2.set_xlabel('M0')
    ax2.set_ylabel('Unstart flag')
    ax2.set_title('Inlet-expulsion flag vs Mach')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0.0, 1.0])

    _save(fig, 'ram_diagnostics_vs_mach')


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
    _save(fig, 'nozzle_geometry_vs_mach')


def flight_envelope_sweep(
    mach_values,
    altitude_values,
    alpha_values,
    phi=PHI_DEFAULT,
    design=None,
    combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
):
    """Evaluate fixed geometry over a Mach-altitude-AoA envelope."""
    mach_values = np.asarray(mach_values, dtype=float)
    altitude_values = np.asarray(altitude_values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)
    if design is None:
        design = pyc_run._get_inlet_design()

    results = np.empty(
        (len(mach_values), len(altitude_values), len(alpha_values)),
        dtype=object,
    )
    total = results.size
    counter = 0

    for i_m, M0 in enumerate(mach_values):
        for i_alt, alt_m in enumerate(altitude_values):
            for i_alpha, alpha_deg in enumerate(alpha_values):
                counter += 1
                t0 = time.perf_counter()
                try:
                    results[i_m, i_alt, i_alpha] = pyc_run.analyze(
                        M0=float(M0),
                        altitude_m=float(alt_m),
                        phi=float(phi),
                        alpha_deg=float(alpha_deg),
                        design=design,
                        combustor_length_m=combustor_length_m,
                    )
                    dt = time.perf_counter() - t0
                    print(
                        f'    [{counter:03d}/{total:03d}] '
                        f'M={M0:.2f}, alt={alt_m/1e3:.2f} km, '
                        f'aoa={alpha_deg:.2f} deg in {dt:.1f}s'
                    )
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    print(
                        f'  [warn] [{counter:03d}/{total:03d}] '
                        f'M={M0:.2f}, alt={alt_m/1e3:.2f} km, '
                        f'aoa={alpha_deg:.2f} deg failed after {dt:.1f}s: {exc}'
                    )
                    results[i_m, i_alt, i_alpha] = None

    return {
        'mach_values': mach_values,
        'altitude_values': altitude_values,
        'alpha_values': alpha_values,
        'phi': float(phi),
        'results': results,
    }


def _envelope_slice(envelope, mach_index, getter):
    alt_values = envelope['altitude_values']
    alpha_values = envelope['alpha_values']
    grid = np.full((len(alt_values), len(alpha_values)), np.nan, dtype=float)
    for i_alt in range(len(alt_values)):
        for i_alpha in range(len(alpha_values)):
            result = envelope['results'][mach_index, i_alt, i_alpha]
            if result is None:
                continue
            try:
                grid[i_alt, i_alpha] = float(getter(result))
            except Exception:
                grid[i_alt, i_alpha] = np.nan
    return grid


def _operability_fraction(envelope, predicate):
    results = envelope['results']
    n_mach = results.shape[0]
    fraction = np.zeros(results.shape[1:], dtype=float)
    for i_alt in range(results.shape[1]):
        for i_alpha in range(results.shape[2]):
            hits = 0
            for i_m in range(n_mach):
                if predicate(results[i_m, i_alt, i_alpha]):
                    hits += 1
            fraction[i_alt, i_alpha] = hits / max(n_mach, 1)
    return fraction


def _started(result):
    return (
        result is not None
        and np.isfinite(float(result.get('unstart_flag', np.nan)))
        and abs(float(result.get('unstart_flag', np.nan))) <= 0.5
    )


def _operable(result):
    return _started(result) and (not bool(result.get('choked', False)))


def _start_state_index(result):
    if result is None:
        return 2.0
    flag = float(result.get('unstart_flag', np.nan))
    if not np.isfinite(flag):
        return 2.0
    if flag > 0.5:
        return 1.0
    return 0.0


def _choke_state_index(result):
    if result is None:
        return 2.0
    return 1.0 if bool(result.get('choked', False)) else 0.0


def _regular_edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mids, [last]])


def _plot_scalar_map(ax, alpha_values, altitude_values, grid, title,
                     cmap='viridis', vmin=None, vmax=None):
    valid = np.isfinite(grid)
    ax.set_title(title)
    ax.set_xlabel('Angle of attack [deg]')
    ax.set_ylabel('Altitude [km]')
    if not np.any(valid):
        ax.text(0.5, 0.5, 'No converged points', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        return None

    levels = np.linspace(vmin, vmax, 17) if (vmin is not None and vmax is not None) else 16
    contour = ax.contourf(
        alpha_values,
        altitude_values / 1e3,
        grid,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.contour(
        alpha_values,
        altitude_values / 1e3,
        grid,
        levels=6,
        colors='k',
        linewidths=0.35,
        alpha=0.35,
    )
    return contour


def _plot_discrete_map(ax, alpha_values, altitude_values, grid, title,
                       cmap, norm, tick_values, tick_labels):
    x_edges = _regular_edges(alpha_values)
    y_edges = _regular_edges(altitude_values / 1e3)
    mesh = ax.pcolormesh(
        x_edges, y_edges, grid,
        cmap=cmap, norm=norm, shading='flat',
    )
    ax.set_title(title)
    ax.set_xlabel('Angle of attack [deg]')
    ax.set_ylabel('Altitude [km]')
    cbar = plt.colorbar(mesh, ax=ax, ticks=tick_values, pad=0.02)
    cbar.ax.set_yticklabels(tick_labels)
    return mesh


def _metric_limits(grids):
    finite_chunks = [g[np.isfinite(g)] for g in grids if np.any(np.isfinite(g))]
    if not finite_chunks:
        return None, None
    finite = np.concatenate(finite_chunks)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) <= 1.0e-12:
        span = max(abs(vmax), 1.0)
        vmin -= 0.05 * span
        vmax += 0.05 * span
    return vmin, vmax


def fig_envelope_performance(envelope):
    mach_values = envelope['mach_values']
    alt_values = envelope['altitude_values']
    alpha_values = envelope['alpha_values']

    metrics = [
        ('Net thrust [kN]', 'thrust', lambda r: r['thrust'] / 1e3, 'viridis'),
        ('Specific thrust [N·s/kg_air]', 'specific thrust', lambda r: r['F_sp'], 'plasma'),
        ('Inlet recovery', 'eta_pt', lambda r: r['eta_pt'], 'cividis'),
    ]

    fig, axes = plt.subplots(
        len(metrics), len(mach_values),
        figsize=(4.7 * len(mach_values), 10.8),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, (cbar_label, title_stub, getter, cmap) in enumerate(metrics):
        grids = [_envelope_slice(envelope, i_m, getter) for i_m in range(len(mach_values))]
        vmin, vmax = _metric_limits(grids)
        row_mappable = None
        for col, M0 in enumerate(mach_values):
            mappable = _plot_scalar_map(
                axes[row, col], alpha_values, alt_values, grids[col],
                title=f'M0={M0:.2f}  {title_stub}',
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            if row == len(metrics) - 1:
                axes[row, col].set_xlabel('Angle of attack [deg]')
            if col > 0:
                axes[row, col].set_ylabel('')
            if row_mappable is None and mappable is not None:
                row_mappable = mappable
        if row_mappable is not None:
            fig.colorbar(row_mappable, ax=axes[row, :], shrink=0.97, pad=0.02,
                         label=cbar_label)

    fig.suptitle(
        f'Off-design performance maps  '
        f'(phi={envelope["phi"]:.2f}, fixed engine geometry)',
        fontsize=13,
    )
    _save(fig, 'offdesign_performance_altitude_aoa')


def fig_envelope_operability(envelope):
    mach_values = envelope['mach_values']
    alt_values = envelope['altitude_values']
    alpha_values = envelope['alpha_values']

    fig, axes = plt.subplots(
        2, len(mach_values),
        figsize=(4.7 * len(mach_values), 7.2),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    start_cmap = ListedColormap(['seagreen', 'darkred', 'lightgray'])
    start_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], start_cmap.N)
    choke_cmap = ListedColormap(['seagreen', 'firebrick', 'lightgray'])
    choke_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], choke_cmap.N)

    for col, M0 in enumerate(mach_values):
        start_grid = _envelope_slice(envelope, col, _start_state_index)
        choke_grid = _envelope_slice(envelope, col, _choke_state_index)

        _plot_discrete_map(
            axes[0, col], alpha_values, alt_values, start_grid,
            title=f'M0={M0:.2f}  inlet state',
            cmap=start_cmap, norm=start_norm,
            tick_values=[0.0, 1.0, 2.0],
            tick_labels=['started', 'expelled', 'failed'],
        )
        _plot_discrete_map(
            axes[1, col], alpha_values, alt_values, choke_grid,
            title=f'M0={M0:.2f}  combustor choking',
            cmap=choke_cmap, norm=choke_norm,
            tick_values=[0.0, 1.0, 2.0],
            tick_labels=['unchoked', 'choked', 'failed'],
        )

        if col > 0:
            axes[0, col].set_ylabel('')
            axes[1, col].set_ylabel('')

    fig.suptitle('Operability maps  (fixed engine geometry)', fontsize=13)
    _save(fig, 'offdesign_operability_altitude_aoa')


def fig_envelope_band_summary(envelope):
    mach_values = envelope['mach_values']
    alt_values = envelope['altitude_values']
    alpha_values = envelope['alpha_values']
    results = envelope['results']

    min_thrust = np.full(results.shape[1:], np.nan, dtype=float)
    for i_alt in range(results.shape[1]):
        for i_alpha in range(results.shape[2]):
            thrusts = []
            for i_m in range(results.shape[0]):
                result = results[i_m, i_alt, i_alpha]
                if result is not None:
                    thrusts.append(float(result['thrust']) / 1e3)
            if thrusts:
                min_thrust[i_alt, i_alpha] = float(np.min(thrusts))

    operable_fraction = 100.0 * _operability_fraction(envelope, _operable)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    thrust_vmin, thrust_vmax = _metric_limits([min_thrust])

    thrust_map = _plot_scalar_map(
        axes[0], alpha_values, alt_values, min_thrust,
        title=(
            f'Min net thrust across M={mach_values[0]:.2f}-'
            f'{mach_values[-1]:.2f} [kN]'
        ),
        cmap='viridis', vmin=thrust_vmin, vmax=thrust_vmax,
    )
    if thrust_map is not None:
        fig.colorbar(thrust_map, ax=axes[0], shrink=0.94, pad=0.02,
                     label='Min net thrust [kN]')

    op_map = _plot_scalar_map(
        axes[1], alpha_values, alt_values, operable_fraction,
        title='Operable Mach coverage [% of sampled Mach points]',
        cmap='cividis', vmin=0.0, vmax=100.0,
    )
    if op_map is not None:
        fig.colorbar(op_map, ax=axes[1], shrink=0.94, pad=0.02,
                     label='Operable coverage [%]')

    fig.suptitle('Mach-band envelope summary', fontsize=13)
    _save(fig, 'offdesign_mach_band_summary')


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
        combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
        converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
        diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
        throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
        exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
        n_points=NOZZLE_BELL_POINTS_DEFAULT,
    )

    stations = (0, 2, 3, 4, 9)
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
    parser.add_argument(
        '--envelope-mach',
        nargs='+',
        type=float,
        default=list(ENVELOPE_MACH_VALUES_DEFAULT),
        help='Mach slices for the altitude/AoA envelope plots.',
    )
    parser.add_argument(
        '--envelope-alt-min',
        type=float,
        default=ENVELOPE_ALT_RANGE_DEFAULT[0],
        help='Minimum altitude for the envelope plots [m].',
    )
    parser.add_argument(
        '--envelope-alt-max',
        type=float,
        default=ENVELOPE_ALT_RANGE_DEFAULT[1],
        help='Maximum altitude for the envelope plots [m].',
    )
    parser.add_argument(
        '--envelope-alt-count',
        type=int,
        default=ENVELOPE_ALT_COUNT_DEFAULT,
        help='Number of altitude samples for the envelope plots.',
    )
    parser.add_argument(
        '--envelope-aoa-min',
        type=float,
        default=ENVELOPE_ALPHA_RANGE_DEFAULT[0],
        help='Minimum angle of attack for the envelope plots [deg].',
    )
    parser.add_argument(
        '--envelope-aoa-max',
        type=float,
        default=ENVELOPE_ALPHA_RANGE_DEFAULT[1],
        help='Maximum angle of attack for the envelope plots [deg].',
    )
    parser.add_argument(
        '--envelope-aoa-count',
        type=int,
        default=ENVELOPE_ALPHA_COUNT_DEFAULT,
        help='Number of angle-of-attack samples for the envelope plots.',
    )
    args = parser.parse_args()
    cad_wall_thickness_m = float(args.cad_wall_thickness_mm) * 1.0e-3
    envelope_mach = np.asarray(args.envelope_mach, dtype=float)
    envelope_alt = np.linspace(
        float(args.envelope_alt_min),
        float(args.envelope_alt_max),
        int(args.envelope_alt_count),
    )
    envelope_alpha = np.linspace(
        float(args.envelope_aoa_min),
        float(args.envelope_aoa_max),
        int(args.envelope_aoa_count),
    )
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
        combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
    )

    """
    # Mach sweep
    sweep_altitude_m = INLET_DESIGN_ALT_M #######################################################
    mach_range = np.linspace(max(M_MIN, 4.0), min(M_MAX, 5.0), 15)
    print(f'  Mach sweep over {len(mach_range)} points '
          f'at alt={sweep_altitude_m/1e3:.0f} km, phi={PHI_DEFAULT}')
    results = mach_sweep(mach_range, altitude=sweep_altitude_m, phi=PHI_DEFAULT) #########################

    


    # Altitude sweep: ±2 km around design altitude, M0 and α frozen at design.
    alt_range = np.linspace(INLET_DESIGN_ALT_M - 1_000.0,
                            INLET_DESIGN_ALT_M + 3_000.0, 10)
    print(f'  Altitude sweep over {len(alt_range)} points '
          f'at M0={INLET_DESIGN_M0}, α={INLET_DESIGN_ALPHA_DEG}°, '
          f'phi={PHI_DEFAULT}')
    alt_results = altitude_sweep(
        alt_range, M0=INLET_DESIGN_M0,
        alpha_deg=INLET_DESIGN_ALPHA_DEG, phi=PHI_DEFAULT,
    )
    """
    """
    # AoA sweep at fixed design Mach and altitude.
    alpha_range = np.linspace(-3.0, 6.0, 9   )
    print(f'  AoA sweep over {len(alpha_range)} points '
          f'at M0={INLET_DESIGN_M0}, alt={INLET_DESIGN_ALT_M/1e3:.0f} km, '
          f'phi={PHI_DEFAULT}')
    aoa_results = aoa_sweep(
        alpha_range,
        M0=INLET_DESIGN_M0,
        altitude_m=INLET_DESIGN_ALT_M,
        phi=PHI_DEFAULT,
    )
    """

    """
    # 2-D (M0, altitude) sweep for the φ operability map.
    phi_mach_range = np.linspace(max(M_MIN, 4.0), min(M_MAX, 5.0), 4)
    phi_alt_range  = np.linspace(INLET_DESIGN_ALT_M - 1_000.0,
                                 INLET_DESIGN_ALT_M + 3_000.0, 4)
    print(f'  φ map over {len(phi_mach_range)}×{len(phi_alt_range)} '
          f'(M0, alt) points at α={INLET_DESIGN_ALPHA_DEG}°, '
          f'φ_request={PHI_DEFAULT}')
    phi_map_results = mach_alt_sweep(
        phi_mach_range, phi_alt_range,
        alpha_deg=INLET_DESIGN_ALPHA_DEG, phi=PHI_DEFAULT,         ########################
    )
    """

    print('\n  writing figures:')
    fig_flowpath(
        design,
        design_cycle,
        combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
        converging_length=NOZZLE_CONVERGING_LENGTH_DEFAULT,
        diverging_length=NOZZLE_DIVERGING_LENGTH_DEFAULT,
        throat_angle_deg=NOZZLE_THROAT_ANGLE_DEFAULT,
        exit_angle_deg=NOZZLE_EXIT_ANGLE_DEFAULT,
        n_points=NOZZLE_BELL_POINTS_DEFAULT,
    )

    """
    fig_performance_along_climb_path(
        design,
        phi=PHI_DEFAULT,
        mach_start=4.0,
        mach_end=5.0,
        alt_start_m=16_000.0,
        alt_end_m=19_000.0,
        alpha_deg=2.0,
        n_points=7,
    )
    """
    #fig_performance(results, mach_range)


    """
    fig_performance_vs_alt(
        alt_results, alt_range,
        M0=INLET_DESIGN_M0, alpha_deg=INLET_DESIGN_ALPHA_DEG, phi=PHI_DEFAULT,
    )
    """
    """
    fig_performance_vs_aoa(
        aoa_results, alpha_range,
        M0=INLET_DESIGN_M0, altitude_m=INLET_DESIGN_ALT_M, phi=PHI_DEFAULT,
    )
    """
    """
    
    fig_phi_vs_mach_alt(
        phi_map_results, phi_mach_range, phi_alt_range,
        alpha_deg=INLET_DESIGN_ALPHA_DEG, phi_request=PHI_DEFAULT,
    )
    
    #fig_mass_flows(results, mach_range)
    #fig_station_T(results, mach_range)
    #fig_station_Pt(results, mach_range)
    """
    #fig_ram_diagnostics(results, mach_range)
    """
    #fig_engine_pressure_profile(design, design_cycle)
    #fig_engine_temperature_profile(design, design_cycle)
    
    """

    print(f'\n  generating 3D CAD model (wall={cad_wall_thickness_m*1e3:.1f} mm) ...')
    try:
        cad_base = os.path.join(OUTDIR, 'engine_cad')
        fig_cad_model(
            design,
            design_cycle,
            wall_thickness_m=cad_wall_thickness_m,
            output_path=cad_base + '.stl',
            extra_output_paths=[cad_base + '.obj', cad_base + '.ply', cad_base + '.stp'],
            combustor_length_m=COMBUSTOR_LENGTH_M_DEFAULT,
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
