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
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pyc_run
from pyc_config import (
    INLET_DESIGN_M0, INLET_DESIGN_ALT_M, INLET_DESIGN_ALPHA_DEG,
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
ALT_DEFAULT = 20_000.0


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
    mach_vals = np.linspace(2.5, 7.0, 19)
    cases = _inlet2.sweep_fixed_geometry_vs_mach(
        design, INLET_DESIGN_ALT_M, mach_vals, INLET_DESIGN_ALPHA_DEG)
    _inlet2.plot_pt_vs_mach(cases, use_immediate_normal=True)
    fig = plt.gcf()
    _save(fig, 'fig03_inlet_pt_vs_mach')


def fig_inlet_pt_vs_alpha(design):
    """Inlet Pt recovery vs angle of attack."""
    alpha_vals = np.linspace(-4.0, 6.0, 21)
    cases = _inlet2.sweep_fixed_geometry_vs_alpha(
        design, INLET_DESIGN_ALT_M, alpha_vals, INLET_DESIGN_M0)
    _inlet2.plot_pt_vs_alpha(cases, use_immediate_normal=True)
    fig = plt.gcf()
    _save(fig, 'fig04_inlet_pt_vs_alpha')


# ---------------------------------------------------------------------------
# Flowpath to-scale: inlet profile + combustor duct + nozzle bell
# ---------------------------------------------------------------------------

def fig_flowpath(design, design_cycle):
    """
    Nose-to-tail to-scale flowpath.  Inlet is 2D side view (from 402inlet2);
    combustor is a rectangular duct; nozzle is axisymmetric (radius plotted
    symmetrically about the duct centerline), contour from
    nozzle_design.generate_bell_contour.
    """
    # Inlet geometry (corners and throat)
    nose    = np.asarray(design['nose_xy'], dtype=float)
    brk2    = np.asarray(design['break2_xy'], dtype=float)
    cowl    = np.asarray(design['cowl_lip_xy'], dtype=float)
    foot    = np.asarray(design['ramp2_normal_foot_xy'], dtype=float)
    fore    = np.asarray(design['forebody_xy'], dtype=float)
    t_up    = np.asarray(design['throat_upper_xy'], dtype=float)
    t_lo    = np.asarray(design['throat_lower_xy'], dtype=float)

    # Duct (combustor) — between inlet throat and nozzle entry, length
    # chosen for visual clarity; height = inlet throat opening.
    duct_h  = float(abs(t_up[1] - t_lo[1]))
    duct_y_lo = min(t_up[1], t_lo[1])
    duct_x0 = max(t_up[0], t_lo[0])
    duct_len = 1.8 * duct_h   # arbitrary aspect for display
    duct_x1 = duct_x0 + duct_len

    # Nozzle bell — axisymmetric; match bell inlet radius to half duct height
    A_throat = float(design_cycle['nozzle_throat_area'])
    A_exit   = float(design_cycle['nozzle_exit_area'])
    A_inlet  = np.pi * (duct_h / 2.0) ** 2
    bell = nozzle_design.generate_bell_contour(
        inlet_area=max(A_inlet, A_throat * 1.01),
        throat_area=A_throat,
        exit_area=A_exit,
    )
    # Shift bell so its -converging_length end sits at duct_x1
    x_shift = duct_x1 - bell['x'][0]
    bx = bell['x'] + x_shift
    # Map axisymmetric radius to display centered on duct centerline
    y_center = 0.5 * (t_up[1] + t_lo[1])
    r = bell['radius']

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # ── Inlet ramps (lower surface) ─────────────────────────────────────────
    ramp_pts = np.vstack([fore, nose, brk2, foot])
    ax.plot(ramp_pts[:, 0], ramp_pts[:, 1], '-', color='steelblue', lw=2.2,
            label='Ramps')

    # Cowl: lip -> throat_upper
    cowl_pts = np.vstack([cowl, t_up])
    ax.plot(cowl_pts[:, 0], cowl_pts[:, 1], '-', color='firebrick', lw=2.2,
            label='Cowl')

    # Throat line (drawn between upper and lower xy)
    ax.plot([t_up[0], t_lo[0]], [t_up[1], t_lo[1]], '--',
            color='gray', lw=1.2, label='Throat')

    # ── Combustor duct ──────────────────────────────────────────────────────
    duct = Rectangle((duct_x0, duct_y_lo), duct_len, duct_h,
                     fill=False, ec='darkgreen', lw=2.0)
    ax.add_patch(duct)
    ax.text(duct_x0 + 0.5 * duct_len, duct_y_lo + duct_h + 0.02 * duct_h,
            'Combustor', ha='center', va='bottom', color='darkgreen')

    # ── Nozzle bell (axisymmetric about duct centerline) ────────────────────
    ax.plot(bx, y_center + r, '-', color='darkorange', lw=2.2, label='Nozzle')
    ax.plot(bx, y_center - r, '-', color='darkorange', lw=2.2)

    # ── Annotations ─────────────────────────────────────────────────────────
    ax.annotate(f'Throat  A* = {A_throat*1e4:.2f} cm²',
                xy=(bx[np.argmin(r)], y_center + r.min()),
                xytext=(bx[np.argmin(r)], y_center + r.min() + 3 * duct_h),
                ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'Exit  Ae/A* = {A_exit/A_throat:.2f}',
                xy=(bx[-1], y_center + r[-1]),
                xytext=(bx[-1], y_center + r[-1] + duct_h),
                ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]  (axisymmetric about duct centerline for nozzle)')
    ax.set_title(f'Flowpath (to scale) — design M={INLET_DESIGN_M0}, '
                 f'alt={INLET_DESIGN_ALT_M/1e3:.0f} km')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    _save(fig, 'fig05_flowpath_geometry')


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
    _save(fig, 'fig06_performance_vs_mach')


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
    _save(fig, 'fig07_mass_flows')


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
    _save(fig, 'fig08_station_temperatures')


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
    _save(fig, 'fig09_station_total_pressures')


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
    _save(fig, 'fig10_inlet_recovery_vs_mach')


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
    _save(fig, 'fig11_nozzle_geometry_vs_mach')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    )

    # Mach sweep
    mach_range = np.linspace(max(M_MIN, 2.5), min(M_MAX, 7.0), 16)
    print(f'  Mach sweep over {len(mach_range)} points '
          f'at alt={ALT_DEFAULT/1e3:.0f} km, φ={PHI_DEFAULT}')
    results = mach_sweep(mach_range, altitude=ALT_DEFAULT, phi=PHI_DEFAULT)

    print('\n  writing figures:')
    fig_inlet_design_detail(design)
    fig_inlet_fixed_grid(design)
    fig_inlet_pt_vs_mach(design)
    fig_inlet_pt_vs_alpha(design)
    fig_flowpath(design, design_cycle)
    fig_performance(results, mach_range)
    fig_mass_flows(results, mach_range)
    fig_station_T(results, mach_range)
    fig_station_Pt(results, mach_range)
    fig_inlet_recovery_cycle(results, mach_range)
    fig_nozzle_geom_vs_mach(results, mach_range)

    print(f'\n  done — figures in {OUTDIR}')


if __name__ == '__main__':
    main()
