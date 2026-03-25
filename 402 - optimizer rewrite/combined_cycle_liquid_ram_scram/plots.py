"""
plots.py
========
Comprehensive performance and geometry plots for the JP-10 dual-mode
ramjet / scramjet.

All new analysis functions are defined here; the original project files
(atmosphere.py, combustor.py, config.py, gas_dynamics.py, inlet.py,
isolator.py, main.py, nozzle.py, thermo.py) are imported read-only and
are NOT modified.

Figures generated
-----------------
 1 – Nose-to-tail flowpath geometry with dimensions, angles, features
 2 – Engine component efficiencies used in cycle analysis
 3 – Thrust-to-weight ratio at min/max airbreathing Mach (q = 1000 psf)
 4 – Inlet geometry (ramp angles, shock angles) vs Mach
 5 – Inlet capture ratio A₀/Ac vs Mach at multiple angles of attack
 6 – Inlet total pressure recovery vs Mach at multiple angles of attack
 7 – Nozzle geometry (area ratio, NPR, exit Mach) vs Mach
 8 – Nozzle gross thrust coefficient Cfg vs Mach
 9 – Force accounting system schematic + equations
10 – Thrust and FnWa vs Mach, altitude, and angle of attack
11 – Specific impulse Isp vs Mach, altitude, and angle of attack

"""

import os
import sys
import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ─── project root on path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ambiance import Atmosphere
from gas_dynamics import (
    make_state,
    oblique_shock,
    normal_shock,
    beta_from_theta,
    isentropic_M_from_Pt_P,
    isentropic_T,
    isentropic_P,
    pi_milspec,
)
from inlet     import compute_inlet, inlet_geometry
from isolator  import compute_isolator
from combustor import compute_combustor
from nozzle    import compute_nozzle
from thermo    import get_thermo
from config    import (
    INLET_RAMPS_DEG,
    M_TRANSITION,
    A_CAPTURE,
    F_STOICH,
    ETA_COMBUSTOR,
    ETA_NOZZLE,
    ISOLATOR_PT_RECOVERY_SCRAM,
    LHV_JP10,
)
from main import analyze

# ─── physical constants & design parameters ───────────────────────────────────
AIR_GAMMA = 1.40
AIR_R     = 287.05       # J/(kg·K)
G0        = 9.80665      # m/s²
Q_PSF     = 1000.0       # design dynamic pressure [psf]
Q_PA      = Q_PSF * 47.8803   # → Pa  (1 psf = 47.88 Pa)
PHI       = 0.8          # default equivalence ratio
M_MIN     = 2.0
M_MAX     = 8.0
M_DESIGN  = 5.0          # inlet design Mach

# Engine weight model – conservative estimate for a small research engine.
# Scales linearly with capture area from a 50 kg base.
_EW_KG = 50.0 + 1_600.0 * A_CAPTURE   # ~130 kg for A_c=0.05 m²
_EW_N  = _EW_KG * G0

OUTDIR = os.path.join(_HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ─── singleton Cantera thermo ──────────────────────────────────────────────────
_thermo = get_thermo()

def safe_analyze(M: float, alt: float, phi: float = PHI) -> dict | None:
    """Wrapper around main.analyze() that returns None on any exception."""
    try:
        return analyze(M0=M, altitude=alt, phi=phi)
    except Exception:
        return None

def compute_transition_mach(alt: float = 25_000.0, phi: float = PHI) -> float:
    """
    Return the RAM->SCRAM transition Mach number.

    The transition is set by M_TRANSITION in config.py (the Mach above which
    the terminal normal shock in the RAM inlet becomes too lossy to sustain
    useful combustion).  This is NOT driven by Rayleigh thermal choking —
    see main.analyze() for the detailed explanation of why thermal choking
    gives an inverted transition direction and must not be used as the
    mode-selection criterion.

    The safe_analyze() sweep is kept as a cross-check: it verifies that the
    first Mach in the sweep that returns mode='scram' agrees with the
    configured M_TRANSITION.  If they diverge (e.g. due to a future code
    change), a warning is printed.

    Parameters
    ----------
    alt : float   Altitude [m] (kept for API compatibility; not used in the
                  primary return value since M_TRANSITION is altitude-independent
                  in this model).
    phi : float   Equivalence ratio (likewise altitude/phi-independent here).

    Returns
    -------
    float   M_TRANSITION from config.py.
    """
    # Cross-check: first Mach in sweep that analyze() calls 'scram'
    for M in np.linspace(M_MIN, M_MAX, 300):
        r = safe_analyze(M, alt, phi)
        if r is not None and r['mode'] == 'scram':
            sweep_trans = float(M)
            if abs(sweep_trans - M_TRANSITION) > 0.1:
                print(f"  WARNING: sweep transition M={sweep_trans:.2f} differs "
                      f"from M_TRANSITION={M_TRANSITION} in config.py")
            break
    return float(M_TRANSITION)




# Physics-backed transition Mach — replaces the hardcoded config value in plots.
# This is where RAM-mode Rayleigh combustion first hits thermal choking at φ=PHI.
_M_TRANS = compute_transition_mach()
print(f"  Physics-backed RAM→SCRAM transition: M = {_M_TRANS:.2f}"
      f"  (config fallback = {M_TRANSITION})")

# ─── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "axes.grid":        True,
    "grid.alpha":       0.30,
    "figure.dpi":       120,
    "lines.linewidth":  2.0,
    "legend.fontsize":  9,
    "legend.framealpha": 0.90,
})

_TC = plt.cm.tab10.colors   # 10-colour qualitative palette


# ══════════════════════════════════════════════════════════════════════════════
# NEW ANALYSIS UTILITIES  (do not modify any existing file)
# ══════════════════════════════════════════════════════════════════════════════

def altitude_for_q(M: float, q_Pa: float = Q_PA,
                   gamma: float = AIR_GAMMA) -> float:
    """
    Return the geometric altitude [m] at which Mach *M* gives dynamic
    pressure *q_Pa*.
        q = ½ γ P M²  →  P_req = 2 q / (γ M²)
    """
    P_req = 2.0 * q_Pa / (gamma * M**2)
    P_sl  = float(Atmosphere(0).pressure[0])       # ~101325 Pa
    P_hi  = float(Atmosphere(50_000).pressure[0])  # ~80 Pa
    if P_req >= P_sl:
        return 0.0
    if P_req <= P_hi:
        return 50_000.0
    return brentq(lambda h: float(Atmosphere(h).pressure[0]) - P_req,
                  0, 50_000, xtol=1.0)




def capture_ratio(M0: float, aoa_deg: float = 0.0,
                  ramps: list | None = None,
                  gamma: float = AIR_GAMMA) -> float:
    """
    Estimate inlet air capture ratio A₀/Ac via a 2-D oblique-shock
    streamtube model.

    Physics
    -------
    The cowl lip is positioned so that at the design Mach (M_DESIGN) the
    first ramp oblique shock impinges exactly on the lip → A₀/Ac = 1.
    Below M_DESIGN the shock angle β increases, reducing the captured
    streamtube width:

        A₀/Ac  ≈  sin(β_design) / sin(β_current)    (2-D, one-shock model)

    Positive AoA (windward) increases the effective first-ramp deflection;
    negative AoA (leeward) decreases it.
    """
    if ramps is None:
        ramps = list(INLET_RAMPS_DEG)

    theta1 = ramps[0]

    # Design shock angle
    beta_d = beta_from_theta(theta1, M_DESIGN, gamma)
    if beta_d is None:
        beta_d = np.arcsin(1.0 / M_DESIGN)   # Mach-angle limit

    # Effective first-ramp angle with AoA
    theta_eff = max(theta1 + aoa_deg, 0.5)
    beta_c    = beta_from_theta(theta_eff, M0, gamma)
    if beta_c is None:
        return 0.30   # shock detached → severe spillage

    cr = np.sin(beta_d) / np.sin(beta_c)
    return float(np.clip(cr, 0.05, 1.0))


def inlet_recovery_aoa(M0: float, aoa_deg: float = 0.0,
                        ramps: list | None = None,
                        alt: float = 25_000.0) -> float:
    """
    Compute inlet total-pressure recovery Pt2/Pt0 with angle-of-attack
    effect modelled as a shift in the first ramp deflection angle.

    Mode selection is physics-backed: try RAM first, check whether the
    combustor would thermally choke at this (M0, alt) condition, and
    promote to SCRAM if it would.  This mirrors the logic in main.analyze()
    so the recovery curve has no artificial step at a hardcoded Mach.
    """
    if ramps is None:
        ramps = list(INLET_RAMPS_DEG)
    eff = list(ramps)
    eff[0] = ramps[0] + aoa_deg

    atm = Atmosphere(alt)
    T0  = float(atm.temperature[0])
    P0  = float(atm.pressure[0])
    s0  = make_state(M0, T0, P0, gamma=AIR_GAMMA, R=AIR_R)

    # Determine mode from physics: run RAM inlet+isolator+combustor and
    # check for thermal choking.  If choked, switch to SCRAM.
    try:
        s2_ram, _ = compute_inlet(s0, eff, mode='ram')
        s3_ram    = compute_isolator(s2_ram, mode='ram')
        _, ram_choked = compute_combustor(s3_ram, PHI, _thermo, mode='ram')
    except Exception:
        ram_choked = True   # any failure (e.g. shock detachment) → treat as choked

    mode = 'scram' if ram_choked else 'ram'

    # Now compute recovery with the correct mode
    try:
        _, eta = compute_inlet(s0, eff, mode=mode)
    except Exception:
        eta = pi_milspec(M0)
    return float(eta)

def nozzle_cfg(M0: float, alt: float, phi: float = PHI) -> float:
    """
    Nozzle gross thrust coefficient:

        Cfg = Fg / (Pt4 · A_t)
            = η_n · (2/(γ+1))^((γ+1)/(2(γ-1)))
              · √( 2γ²/(γ-1) · (1 − (P0/Pt4)^((γ-1)/γ)) )

    The factor (2/(γ+1))^… arises from the choked throat mass-flux
    normalisation; η_n accounts for divergence and viscous losses.
    """
    r = safe_analyze(M0, alt, phi)
    if r is None:
        return ETA_NOZZLE
    s4  = r["states"][4]
    P0  = float(Atmosphere(alt).pressure[0])
    NPR = s4.Pt / P0
    if NPR <= 1.0:
        return 0.0
    g = s4.gamma
    throat_factor = (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    expansion     = np.sqrt(2.0 * g**2 / (g - 1.0) *
                            (1.0 - (1.0 / NPR) ** ((g - 1.0) / g)))
    return float(ETA_NOZZLE * throat_factor * expansion)


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  ✓  {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Nose-to-tail flowpath geometry
# ══════════════════════════════════════════════════════════════════════════════

def fig01_flowpath() -> None:
    """2-D side-view cross-section of the dual-mode ram/scramjet engine."""
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")

    # ── Inlet geometry from shock-on-lip condition ─────────────────────────
    W  = 0.20                       # span (into page) [m]
    Hc = A_CAPTURE / W              # capture height [m]

    geom = inlet_geometry(
        M_design        = M_DESIGN,
        ramp_angles_deg = INLET_RAMPS_DEG,
        H_capture       = Hc,
        gamma           = AIR_GAMMA,
    )

    corners = geom['corners']        # [(x0,y0), (x1,y1), ..., (xn,yn)]
    x_lip   = geom['x_lip']          # cowl lip x-position
    betas   = geom['betas_deg']      # shock wave angles at M_DESIGN [deg]
    lab_ang = geom['lab_angles_deg'] # lab-frame shock angles [deg]
    H_throat = geom['throat_height'] # internal duct height

    rx = [c[0] for c in corners]
    ry = [c[1] for c in corners]

    x_inlet = rx[-1]   # = x_lip by construction

    iso_L   = 0.30
    comb_L  = 0.42
    nozz_L  = 0.65
    x_iso2  = x_inlet  + iso_L
    x_comb2 = x_iso2   + comb_L
    x_nozz2 = x_comb2  + nozz_L

    H_nozz_exit = H_throat * 4.5   # nozzle expands ~4.5× in height

    # ── filled duct sections ───────────────────────────────────────────────
    def section(x1, x2, yb1, yb2, yt1, yt2, fc, alpha=0.32):
        verts = [(x1, yb1), (x2, yb2), (x2, yt2), (x1, yt1)]
        ax.add_patch(Polygon(verts, closed=True, facecolor=fc,
                             edgecolor="none", alpha=alpha))
        ax.plot([x1, x2], [yb1, yb2], "k-", lw=1.6)
        ax.plot([x1, x2], [yt1, yt2], "k-", lw=1.6)

    section(x_inlet, x_iso2, ry[-1], ry[-1], Hc, Hc, "#74c476")   # isolator
    section(x_iso2,  x_comb2, ry[-1], ry[-1], Hc, Hc, "#fd8d3c")  # combustor
    section(x_comb2, x_nozz2,                                       # nozzle
            ry[-1], ry[-1] - H_nozz_exit + H_throat, Hc, Hc, "#9ecae1")

    # Ramp lower-wall solid body
    ramp_body = ([(rx[0], ry[0])] + list(zip(rx, ry))
                 + [(rx[-1], ry[-1] - 0.10), (rx[0], -0.10)])
    ax.add_patch(Polygon(ramp_body, closed=True,
                         facecolor="#bdbdbd", edgecolor="k", lw=1.8))
    ax.plot(rx, ry, "k-", lw=2.8)

    # Cowl (flat upper wall) and nozzle lower-wall expansion
    ax.fill_between([x_inlet, x_nozz2], [Hc, Hc], [Hc + 0.024, Hc + 0.024],
                    color="#525252", alpha=0.80, zorder=4)
    ax.plot([x_inlet, x_nozz2], [Hc, Hc], "k-", lw=2.2, zorder=5)
    ax.plot([x_comb2, x_nozz2],
            [ry[-1], ry[-1] - (H_nozz_exit - H_throat)], "k-", lw=2.2)

    # Cowl lip dashed reference line
    ax.plot([x_inlet, x_inlet], [ry[-1], Hc + 0.024], "k--", lw=1.0)

    # ── oblique shocks at design Mach ──────────────────────────────────────
    # Each shock originates at its ramp corner and is drawn to the cowl lip
    # (x_lip, Hc) — enforced by the shock-on-lip design condition.
    shock_cols = ["#d73027", "#f46d43", "#fdae61"]
    for i in range(len(INLET_RAMPS_DEG)):
        x0, y0   = corners[i]        # shock origin = corner i
        x1s, y1s = x_lip, Hc         # shock terminus = cowl lip

        ax.annotate("", xy=(x1s, y1s), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-", color=shock_cols[i],
                                   lw=2.0, linestyle="dashed"))

        # Label at midpoint of shock ray
        '''
        xm = 0.5 * (x0 + x1s)
        ym = 0.5 * (y0 + y1s)
        ax.text(xm, ym - 0.018,
                f"OS{i+1}  β={betas[i]:.1f}°\n(θ={INLET_RAMPS_DEG[i]}°)",
                fontsize=7.5, color=shock_cols[i], ha="center", fontweight="bold")
        '''

    # ── section labels ─────────────────────────────────────────────────────
    def sec_lbl(x, y, t, bg):
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5,
                fontweight="bold",
                bbox=dict(facecolor=bg, edgecolor="#555", alpha=0.88,
                          pad=3, boxstyle="round,pad=0.3"))

    ym_duct = ry[-1] + H_throat / 2.0
    sec_lbl(0.5 * (rx[0] + rx[-1]), ry[-1] + H_throat * 0.4,
            f"External Compression Inlet\n"
            f"Ramps: {INLET_RAMPS_DEG[0]}°–{INLET_RAMPS_DEG[1]}°–"
            f"{INLET_RAMPS_DEG[2]}°  (design M={M_DESIGN})", "#c6dbef")
    sec_lbl(0.5 * (x_inlet + x_iso2),  ym_duct-0.1, f"Isolator\nL={iso_L:.2f} m", "#c7e9c0")
    sec_lbl(0.5 * (x_iso2  + x_comb2), ym_duct-0.1, f"Combustor\nL={comb_L:.2f} m\n(JP-10/air)", "#fdd0a2")
    sec_lbl(0.5 * (x_comb2 + x_nozz2),
            ry[-1] - H_nozz_exit / 2.0,
            f"Nozzle\nL={nozz_L:.2f} m", "#deebf7")

    # ── station markers ────────────────────────────────────────────────────
    stns = {"∞(0)": rx[0] - 0.14,
            "2":    x_inlet,
            "3":    x_iso2,
            "4":    x_comb2,
            "9":    x_nozz2}
    for lbl, xs in stns.items():
        ax.plot([xs, xs], [Hc + 0.09, Hc + 0.15], "k-", lw=0.9)
        ax.text(xs, Hc + 0.22, f"Stn {lbl}", ha="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#aaa",
                          pad=2, alpha=0.9))

    # ── ramp-angle arc annotations ─────────────────────────────────────────
    for i in range(len(INLET_RAMPS_DEG)):
        xm_r = 0.5 * (rx[i] + rx[i + 1])
        ym_r = 0.5 * (ry[i] + ry[i + 1]) - 0.048
        ax.text(xm_r, ym_r, f"θ_{i+1}={INLET_RAMPS_DEG[i]}°",
                ha="center", fontsize=9, color="#08519c", fontweight="bold")

    # ── top dimension arrows ───────────────────────────────────────────────
    ya_dim = -0.25
    def dim_arr(x1, x2, y, lbl, col="#333333"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        ax.text(0.5*(x1 + x2), y - 0.018, lbl,
                ha="center", fontsize=7.5, color=col)

    ramp_total = rx[-1] - rx[0]
    dim_arr(rx[0], rx[-1],  ya_dim, f"Ramps  {ramp_total:.2f} m")
    dim_arr(x_inlet, x_iso2,  ya_dim, f"{iso_L} m")
    dim_arr(x_iso2,  x_comb2, ya_dim, f"{comb_L} m")
    dim_arr(x_comb2, x_nozz2, ya_dim, f"{nozz_L} m")
    dim_arr(rx[0], x_nozz2, ya_dim - 0.055,
            f"Total engine length  {rx[0]:.2f} – {x_nozz2:.2f} m  "
            f"≈  {x_nozz2 - rx[0]:.2f} m",
            col="#08306b")

    # Capture-height annotation
    ax.annotate("", xy=(rx[0] - 0.06, 0), xytext=(rx[0] - 0.06, Hc),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
    ax.text(rx[0] - 0.09, Hc / 2,
            f"H_c\n= {Hc*100:.1f} cm",
            rotation=90, ha="right", va="center", fontsize=8)

    # Throat-height annotation (inside duct at cowl lip)
    '''
    ax.annotate("", xy=(x_lip + 0.04, ry[-1]), xytext=(x_lip + 0.04, Hc),
                arrowprops=dict(arrowstyle="<->", color="#08519c", lw=1.0))
    ax.text(x_lip + 0.07, ry[-1] + H_throat / 2,
            f"H_th\n={H_throat*100:.1f} cm",
            rotation=90, ha="left", va="center", fontsize=7.5, color="#08519c")
    '''
    # ── mode transition note (inside the duct) ─────────────────────────────
    '''
    ax.text(0.5 * (x_inlet + x_comb2), ry[-1] + H_throat * 0.15,
            f"RAM mode  M < {_M_TRANS:.1f}  |  SCRAM mode  M ≥ {_M_TRANS:.1f}",
            ha="center", fontsize=8, color="gray", style="italic")
    '''
    ax.set_xlim(-0.33, x_nozz2 + 0.10)
    ax.set_ylim(-0.4, Hc + 0.28)
    ax.invert_yaxis()

    # Geometry info box
    geom_note = (
        f"Shock-on-lip geometry (design M={M_DESIGN})\n"
        + "\n".join(
            f"  Ramp {i+1}: L={geom['ramp_lengths'][i]*100:.1f} cm, "
            f"β={betas[i]:.1f}°, lab={lab_ang[i]:.1f}°"
            for i in range(len(INLET_RAMPS_DEG))
        )
        + f"\n  x_lip = {x_lip*100:.1f} cm,  H_throat = {H_throat*100:.1f} cm"
        + f"\n  Contraction ratio H_c/H_th = {Hc/H_throat:.2f}"
    )
    ax.text(0.01, 0.01, geom_note, transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="#333",
            bbox=dict(facecolor="white", edgecolor="#aaa", alpha=0.90, pad=3))

    ax.set_title(
        "Figure 1 — JP-10 Dual-Mode Ram/Scramjet  |  Nose-to-Tail Flowpath Schematic\n"
        f"Capture Area A_c = {A_CAPTURE} m²  "
        f"({W*100:.0f} cm span × {Hc*100:.1f} cm height)   "
        f"Fuel: JP-10 (C₁₀H₁₆)   Design Mach = {M_DESIGN}   "
        f"Mode transition M = {_M_TRANS:.1f}  (config-set)",
        fontweight="bold", fontsize=10.5,
    )

    _save(fig, "fig01_flowpath_geometry.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Component efficiencies
# ══════════════════════════════════════════════════════════════════════════════

def fig02_efficiencies() -> None:
    """Bar chart + reference table of all cycle efficiency parameters."""
    η_inlet_scram = inlet_recovery_aoa(M_DESIGN, 0.0)
    η_inlet_ram   = inlet_recovery_aoa(4.0,      0.0)

    bars = [
        ("Inlet Pt recovery\nScram (M=8, AoA=0°)",  η_inlet_scram,  "#6baed6"),
        ("Inlet Pt recovery\nRam   (M=4, AoA=0°)",  η_inlet_ram,    "#2166ac"),
        ("Isolator Pt recovery\n(scram mode)",        ISOLATOR_PT_RECOVERY_SCRAM, "#74c476"),
        ("Combustion efficiency\nη_c",               ETA_COMBUSTOR,  "#fd8d3c"),
        ("Nozzle velocity coeff.\nη_n",              ETA_NOZZLE,     "#9e9ac8"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── horizontal bar chart ───────────────────────────────────────────────
    names  = [b[0] for b in bars]
    vals   = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    y_pos  = np.arange(len(names))

    bh = ax1.barh(y_pos, vals, color=colors,
                  edgecolor="k", linewidth=0.8, height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9.5)
    ax1.set_xlabel("Efficiency / Pressure Recovery Ratio")
    ax1.set_xlim(0, 1.10)
    ax1.set_title("Engine Component Efficiencies", fontweight="bold")
    ax1.axvline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    for bar, v in zip(bh, vals):
        ax1.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
                 f"{v*100:.1f}%", va="center", fontsize=10, fontweight="bold")

    # Overall cycle Pt recovery at design point
    η_cycle = η_inlet_scram * ISOLATOR_PT_RECOVERY_SCRAM
    ax1.text(0.02, -0.8,
             f"Combined inlet+isolator Pt recovery (M={M_DESIGN}): "
             f"{η_cycle*100:.1f}%",
             fontsize=9, style="italic", color="#636363",
             transform=ax1.transData)

    # ── parameter table ────────────────────────────────────────────────────
    rows = [
        ["LHV JP-10",                    f"{LHV_JP10/1e6:.1f} MJ/kg",  "config.py"],
        ["Stoich. fuel/air ratio",        f"{F_STOICH:.4f}",            "config.py"],
        ["Combustion efficiency η_c",     f"{ETA_COMBUSTOR*100:.0f}%",  "config.py"],
        ["Nozzle velocity coeff. η_n",    f"{ETA_NOZZLE*100:.0f}%",     "config.py"],
        ["Isolator Pt rec. (scram)",      f"{ISOLATOR_PT_RECOVERY_SCRAM*100:.0f}%",
                                                                         "config.py"],
        ["Mode transition Mach",          f"M = {M_TRANSITION}",        "config.py"],
        ["Inlet ramp angles",
         f"{INLET_RAMPS_DEG[0]}°/{INLET_RAMPS_DEG[1]}°/{INLET_RAMPS_DEG[2]}°  (Σ=24°)",
                                                                         "config.py"],
        ["Capture area A_c",              f"{A_CAPTURE:.3f} m²",        "config.py"],
        ["Inlet Pt rec. (M=8, 0° AoA)",  f"{η_inlet_scram*100:.1f}%",  "computed"],
        ["Inlet Pt rec. (M=4, 0° AoA)",  f"{η_inlet_ram*100:.1f}%",    "computed"],
        ["Equivalence ratio φ (default)", f"{PHI:.2f}",                 "analysis"],
        ["Engine weight (model)",         f"{_EW_KG:.0f} kg",           "estimate"],
    ]

    ax2.axis("off")
    tbl = ax2.table(
        cellText=rows,
        colLabels=["Parameter", "Value", "Source"],
        cellLoc="center", loc="center",
        colWidths=[0.48, 0.30, 0.22],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1, 1.50)

    for j in range(3):
        tbl[(0, j)].set_facecolor("#08306b")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(3):
            tbl[(i, j)].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    ax2.set_title("Cycle Analysis Model Parameters", fontweight="bold")

    fig.suptitle("Figure 2 — Component Efficiencies & Model Parameters",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig02_efficiencies.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Thrust-to-weight ratio at q = 1000 psf
# ══════════════════════════════════════════════════════════════════════════════

def fig03_thrust_to_weight() -> None:
    """T/W and gross thrust along the q = 1000 psf flight path."""
    machs  = np.linspace(M_MIN, M_MAX, 20)
    alts   = [altitude_for_q(M) for M in machs]
    thrust = []
    for M, alt in zip(machs, alts):
        r = safe_analyze(M, alt)
        thrust.append(r["thrust"] if r else np.nan)

    thrust = np.array(thrust)
    tw     = thrust / _EW_N
    alts_km = np.array(alts) / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: T/W and thrust on twin y-axes
    ax  = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(machs, tw,           "o-",  color=_TC[0], ms=5, label="T/W")
    l2, = ax2.plot(machs, thrust / 1e3, "s--", color=_TC[1], ms=5, label="Thrust [kN]")
    ax.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
               label=f"Mode switch M={_M_TRANS:.2f}")
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Thrust / Engine Weight  (T/W)", color=_TC[0])
    ax2.set_ylabel("Net Thrust  [kN]",              color=_TC[1])
    ax.tick_params(axis="y", labelcolor=_TC[0])
    ax2.tick_params(axis="y", labelcolor=_TC[1])
    ax.set_title(f"T/W and Thrust vs Mach\n(q = {Q_PSF:.0f} psf trajectory, φ = {PHI})",
                 fontweight="bold")

    # Annotate min and max T/W
    ok = ~np.isnan(tw)
    if ok.any():
        for func, lbl in [(np.nanargmin, "Min"), (np.nanargmax, "Max")]:
            ii = func(tw)
            dy = np.nanmax(tw[ok]) * 0.12
            ax.annotate(
                f"{lbl} T/W = {tw[ii]:.2f}\nM = {machs[ii]:.1f}",
                xy=(machs[ii], tw[ii]),
                xytext=(machs[ii] + 0.6, tw[ii] + dy),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9),
                bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=2),
            )
    ax.legend([l1, l2, plt.Line2D([0],[0], color="gray", ls=":")],
              ["T/W", "Thrust [kN]", f"Mode switch M={_M_TRANS:.2f}"],
              fontsize=8)

    # Right: q=1000 psf altitude trajectory coloured by T/W
    ax3 = axes[1]
    sc = ax3.scatter(machs, alts_km, c=tw, cmap="RdYlGn",
                     s=80, zorder=3, vmin=0)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("T/W ratio")
    ax3.plot(machs, alts_km, "k-", lw=0.8, alpha=0.4)
    ax3.axvline(_M_TRANS, color="gray", ls=":", lw=1.3)
    ax3.set_xlabel("Mach Number")
    ax3.set_ylabel("Altitude  [km]")
    ax3.set_title(f"q = {Q_PSF:.0f} psf Flight Path  (colour = T/W)",
                  fontweight="bold")

    # Info box
    info = (f"Engine weight model\n"
            f"  W_eng ≈ {_EW_KG:.0f} kg  ({_EW_N:.0f} N)\n"
            f"  A_c = {A_CAPTURE:.3f} m²,  φ = {PHI}\n"
            f"  q = {Q_PSF:.0f} psf = {Q_PA:.0f} Pa")
    ax3.text(0.98, 0.04, info, transform=ax3.transAxes,
             fontsize=8, va="bottom", ha="right",
             bbox=dict(facecolor="white", edgecolor="#aaa", alpha=0.9, pad=4))

    fig.suptitle(
        f"Figure 3 — Thrust-to-Weight Ratio at q = {Q_PSF:.0f} psf  "
        f"(min airbreathing M = {M_MIN}, max M = {M_MAX})",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig03_thrust_to_weight.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Inlet geometry
# ══════════════════════════════════════════════════════════════════════════════

def fig04_inlet_geometry() -> None:
    """Ramp angles (fixed geometry) and oblique-shock wave angle β vs Mach."""
    machs = np.linspace(M_MIN, M_MAX, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: individual and cumulative ramp angles (bar + step)
    ramp_idx   = np.arange(len(INLET_RAMPS_DEG))
    cum_angles = np.cumsum(INLET_RAMPS_DEG)

    bars = ax1.bar(ramp_idx, INLET_RAMPS_DEG, width=0.40,
                   color=[_TC[0], _TC[1], _TC[2]],
                   edgecolor="k", linewidth=0.9, label="Individual ramp angle")
    # Step-cumulative overlay
    xs_step = np.array([0, 0, 1, 1, 2, 2, 3]) - 0.2
    ys_step = np.array([0, INLET_RAMPS_DEG[0], INLET_RAMPS_DEG[0],
                        cum_angles[1], cum_angles[1],
                        cum_angles[2], cum_angles[2]])
    ax1.plot(xs_step, ys_step, "k--", lw=1.5, label="Cumulative deflection")

    ax1.set_xticks(ramp_idx)
    ax1.set_xticklabels([f"Ramp {i+1}" for i in ramp_idx])
    ax1.set_ylabel("Deflection Angle  [deg]")
    ax1.set_title("Inlet Ramp Angles — Fixed Geometry", fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0, max(cum_angles) * 1.25)

    for i, (ang, cum) in enumerate(zip(INLET_RAMPS_DEG, cum_angles)):
        ax1.text(i, ang + 0.35, f"{ang}°",
                 ha="center", fontsize=11, fontweight="bold")
        ax1.text(i + 0.22, cum + 0.35, f"Σ={cum}°",
                 ha="left", fontsize=9, color="k")

    ax1.text(0.50, 0.96,
             "Note: Fixed-geometry inlet — no variable ramps\n"
             "Total external deflection = 24°\n"
             f"RAM mode adds terminal normal shock (M < {_M_TRANS:.1f})",
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=8.5,
             bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=3))

    # Right: oblique shock wave angle β vs freestream Mach for each ramp
    M_arr = machs.copy()
    for i, theta in enumerate(INLET_RAMPS_DEG):
        betas = []
        for M in M_arr:
            b = beta_from_theta(theta, M, AIR_GAMMA)
            betas.append(np.degrees(b) if b is not None else np.nan)
        ax2.plot(machs, np.array(betas), color=_TC[i],
                 label=f"Ramp {i+1}  (δ={theta}°)")
        # Advance local Mach through this ramp for the next iteration
        new_M = []
        for M in M_arr:
            M2, *_ = oblique_shock(M, theta, AIR_GAMMA)
            new_M.append(M2 if M2 is not None else M)
        M_arr = np.array(new_M)

    # Mach-angle limit
    mach_angles = np.degrees(np.arcsin(1.0 / machs))
    ax2.plot(machs, mach_angles, "k:", lw=1.0, label="Mach angle μ (β_min)")
    ax2.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
                label=f"Mode switch M={_M_TRANS:.2f}")
    ax2.axvline(M_DESIGN, color="k", ls="--", lw=0.9, alpha=0.5,
                label=f"Design M={M_DESIGN}")

    ax2.set_xlabel("Freestream Mach Number M₀")
    ax2.set_ylabel("Shock Wave Angle β  [deg]")
    ax2.set_title("Oblique Shock Wave Angle β vs Freestream Mach\n"
                  "(at each ramp station, AIR γ=1.40)",
                  fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(M_MIN, M_MAX)

    fig.suptitle("Figure 4 — Inlet Geometry: Ramp Angles and Shock Wave Angles",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig04_inlet_geometry.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Inlet capture ratio A₀/Ac vs Mach
# ══════════════════════════════════════════════════════════════════════════════

def fig05_capture_ratio() -> None:
    """A₀/Ac from 2-D streamtube model at four angles of attack."""
    machs    = np.linspace(M_MIN, M_MAX, 80)
    aoa_vals = [-4.0, 0.0, 4.0, 8.0]
    aoa_cols = [_TC[3], _TC[0], _TC[2], _TC[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: A₀/Ac vs Mach at multiple AoA
    for aoa, col in zip(aoa_vals, aoa_cols):
        cr = [capture_ratio(M, aoa) for M in machs]
        ax1.plot(machs, cr, color=col, label=f"AoA = {aoa:+.0f}°")

    ax1.axvline(M_DESIGN, color="k", ls="--", lw=0.9, alpha=0.5,
                label=f"Design M={M_DESIGN} (A₀/Ac → 1)")
    ax1.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
                label=f"Mode switch M={_M_TRANS:.2f}")
    ax1.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
    ax1.set_xlabel("Freestream Mach Number M₀")
    ax1.set_ylabel("Capture Ratio  A₀/Ac")
    ax1.set_ylim(0.0, 1.12)
    ax1.set_xlim(M_MIN, M_MAX)
    ax1.set_title("Inlet Air Capture Ratio vs Mach\n"
                  "(2-D oblique-shock streamtube model)",
                  fontweight="bold")
    ax1.legend()

    note = ("Model: A₀/Ac = sin(β_design) / sin(β_current)\n"
            "β computed from first ramp θ₁ ± AoA\n"
            f"Design Mach = {M_DESIGN}  (A₀/Ac = 1 at design)\n"
            "Shock detachment → A₀/Ac ≈ 0.30 (heavily spilled)")
    ax1.text(0.02, 0.05, note, transform=ax1.transAxes, fontsize=8,
             va="bottom",
             bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=3))

    # Right: A₀/Ac vs AoA at fixed Mach numbers
    aoa_range = np.linspace(-8.0, 12.0, 60)
    mach_sel  = [3.0, 5.0, 6.0, 8.0, 10.0]
    for i, Mf in enumerate(mach_sel):
        cr = [capture_ratio(Mf, a) for a in aoa_range]
        ax2.plot(aoa_range, cr, color=_TC[i], label=f"M = {Mf}")

    ax2.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
    ax2.set_xlabel("Angle of Attack  [deg]")
    ax2.set_ylabel("Capture Ratio  A₀/Ac")
    ax2.set_ylim(0.0, 1.12)
    ax2.set_title("Capture Ratio vs Angle of Attack\n"
                  "(at selected Mach numbers)",
                  fontweight="bold")
    ax2.legend()

    fig.suptitle("Figure 5 — Inlet Air Capture Ratio A₀/Ac vs Mach Number",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig05_capture_ratio.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Inlet total-pressure recovery
# ══════════════════════════════════════════════════════════════════════════════

def fig06_inlet_pt_recovery() -> None:
    """Pt2/Pt0 vs Mach at multiple AoA, compared with MIL-E-5007D."""
    machs    = np.linspace(M_MIN, M_MAX, 60)
    aoa_vals = [-4.0, 0.0, 4.0]
    aoa_cols = [_TC[3], _TC[0], _TC[2]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: recovery vs Mach at multiple AoA
    milspec = [pi_milspec(M) for M in machs]
    ax1.plot(machs, milspec, "k--", lw=1.6, label="MIL-E-5007D (reference)")

    # ── Per-AoA transition Mach ───────────────────────────────────────────
    # AoA shifts the effective first ramp angle, changing post-shock T3 and
    # therefore where Rayleigh choking occurs.  A single _M_TRANS line (0° AoA)
    # would only be correct for the 0° curve; the ±4° curves choke at
    # different Mach numbers and the annotation arrow would point to the
    # wrong step.  Compute each transition independently.
    def transition_mach_for_aoa(aoa: float, alt: float = 25_000.0) -> float:
        """First M in sweep where inlet_recovery_aoa() switches to SCRAM mode."""
        for M in np.linspace(M_MIN, M_MAX, 300):
            try:
                eff = list(INLET_RAMPS_DEG)
                eff[0] = INLET_RAMPS_DEG[0] + aoa
                atm = Atmosphere(alt)
                T0  = float(atm.temperature[0])
                P0  = float(atm.pressure[0])
                s0  = make_state(M, T0, P0, gamma=AIR_GAMMA, R=AIR_R)
                s2, _ = compute_inlet(s0, eff, mode='ram')
                s3    = compute_isolator(s2, mode='ram')
                _, choked = compute_combustor(s3, PHI, _thermo, mode='ram')
                if choked:
                    return float(M)
            except Exception:
                return float(M)
        return _M_TRANS   # fallback

    m_trans_per_aoa = {aoa: transition_mach_for_aoa(aoa) for aoa in aoa_vals}

    for aoa, col in zip(aoa_vals, aoa_cols):
        rec = [inlet_recovery_aoa(M, aoa, alt=25_000.0) for M in machs]
        ax1.plot(machs, rec, color=col, label=f"AoA = {aoa:+.0f}°")

        # Draw a color-matched vertical line at the mode-switch Mach for
        # this AoA curve so each step is correctly identified
        mt = m_trans_per_aoa[aoa]
        ax1.axvline(mt, color=col, ls=":", lw=1.1, alpha=0.75)

    ax1.axvline(M_DESIGN, color="k", ls="--", lw=0.9, alpha=0.4,
                label=f"Design M={M_DESIGN}")
    ax1.set_xlabel("Freestream Mach Number M₀")
    ax1.set_ylabel("Total Pressure Recovery  Pt₂/Pt₀")
    ax1.set_ylim(0.0, 1.08)
    ax1.set_xlim(M_MIN, M_MAX)
    ax1.set_title("Inlet Total Pressure Recovery vs Mach\n"
                  "(fixed alt = 25 km,  φ = N/A)",
                  fontweight="bold")
    ax1.legend(fontsize=8)

    # Annotate the discontinuity for the 0° AoA curve specifically
    mt0 = m_trans_per_aoa[0.0]
    ax1.axvspan(mt0 - 0.12, mt0 + 0.12,
                color="gold", alpha=0.28, zorder=0)
    ax1.annotate(
        "Step is a model artifact:\nRAM adds terminal normal\nshock; SCRAM does not.\nNot physically real.",
        xy=(mt0, 0.60),
        xytext=(mt0 + 0.55, 0.38),
        fontsize=7.5, color="#7f4f00",
        arrowprops=dict(arrowstyle="->", color="#7f4f00", lw=0.9),
        bbox=dict(facecolor="#fffbe6", edgecolor="#c9a227", alpha=0.92, pad=3),
    )

    mt_strs = ", ".join(
        f"AoA {a:+.0f}°→M={m_trans_per_aoa[a]:.2f}" for a in aoa_vals
    )
    note = (f"Mode model:\n"
            f"  RAM: 3 oblique shocks + normal shock\n"
            f"  SCRAM: 3 oblique shocks only\n"
            f"Transition Mach (Rayleigh choke, physics-backed):\n"
            f"  {mt_strs}\n"
            "AoA effect: θ₁_eff = θ₁ + AoA  (windward +)\n"
            "Shock detachment → MIL-SPEC fallback\n"
            "Dotted lines = per-AoA mode-switch Mach")
    ax1.text(0.02, 0.05, note, transform=ax1.transAxes, fontsize=7.5,
             va="bottom",
             bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=3))

    # Right: recovery vs AoA at fixed Mach numbers
    aoa_range = np.linspace(-6.0, 8.0, 50)
    mach_sel  = [3.0, 5.0, 7.0, 10.0]
    for i, Mf in enumerate(mach_sel):
        rec = [inlet_recovery_aoa(Mf, a, alt=25_000.0) for a in aoa_range]
        ax2.plot(aoa_range, rec, color=_TC[i], label=f"M = {Mf}")

    ax2.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Angle of Attack  [deg]")
    ax2.set_ylabel("Total Pressure Recovery  Pt₂/Pt₀")
    ax2.set_title("Inlet Pt Recovery vs Angle of Attack\n"
                  "(at selected Mach numbers, alt = 25 km)",
                  fontweight="bold")
    ax2.legend()

    fig.suptitle("Figure 6 — Inlet Total Pressure Recovery vs Mach Number",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig06_inlet_pt_recovery.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Nozzle geometry
# ══════════════════════════════════════════════════════════════════════════════

def fig07_nozzle_geometry() -> None:
    """Nozzle area ratio Ae/At, NPR, and exit Mach vs freestream Mach."""
    machs  = np.linspace(M_MIN, M_MAX, 20)
    alts_q = [altitude_for_q(M) for M in machs]

    area_ratio, M9_arr, NPR_arr = [], [], []

    for M, alt in zip(machs, alts_q):
        r = safe_analyze(M, alt)
        if r is None:
            area_ratio.append(np.nan)
            M9_arr.append(np.nan)
            NPR_arr.append(np.nan)
            continue
        s4  = r["states"][4]
        s9  = r["states"][9]
        P0  = float(Atmosphere(alt).pressure[0])
        NPR = s4.Pt / P0
        M9  = s9.M
        g   = s4.gamma
        if M9 > 0:
            Ae_At = ((1.0 / M9) *
                     ((2.0/(g+1.0)) *
                      (1.0 + (g-1.0)/2.0 * M9**2)) **
                     ((g+1.0)/(2.0*(g-1.0))))
        else:
            Ae_At = np.nan
        area_ratio.append(Ae_At)
        M9_arr.append(M9)
        NPR_arr.append(NPR)

    area_ratio = np.array(area_ratio)
    M9_arr     = np.array(M9_arr)
    NPR_arr    = np.array(NPR_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: area ratio and NPR on twin axes
    ax  = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(machs, area_ratio, "o-",  color=_TC[0], ms=5,
                  label="Ae/At  (expansion ratio)")
    l2, = ax2.plot(machs, NPR_arr,   "s--", color=_TC[1], ms=5,
                   label="NPR = Pt4/P0")
    ax.axvline(_M_TRANS, color="gray", ls=":", lw=1.3)
    ax.set_xlabel("Freestream Mach Number M₀")
    ax.set_ylabel("Nozzle Expansion Ratio  Ae/At", color=_TC[0])
    ax2.set_ylabel("Nozzle Pressure Ratio  NPR",   color=_TC[1])
    ax.tick_params(axis="y", labelcolor=_TC[0])
    ax2.tick_params(axis="y", labelcolor=_TC[1])
    ax.set_title("Nozzle Expansion Ratio and NPR vs Mach\n"
                 f"(q = {Q_PSF:.0f} psf trajectory)",
                 fontweight="bold")
    lines  = [l1, l2, plt.Line2D([0],[0], color="gray", ls=":")]
    labels = ["Ae/At", "NPR", f"Mode switch M={_M_TRANS:.2f}"]
    ax.legend(lines, labels, fontsize=8)

    note = ("Nozzle type: 2-D conv-div (fixed geometry)\n"
            "Expansion: perfectly expanded  (P₉ = P₀)\n"
            f"Velocity coefficient η_n = {ETA_NOZZLE:.2f}\n"
            "Ae/At varies continuously with flight condition\n"
            "(no moving parts; off-design losses captured by η_n)")
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=8,
            va="bottom",
            bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=3))

    # Right: exit Mach vs freestream Mach
    ax3 = axes[1]
    ax3.plot(machs, M9_arr, "o-", color=_TC[2], ms=5,
             label="Nozzle exit Mach M₉")
    ax3.plot(machs, machs, "k--", lw=1.0, label="M₉ = M₀  (ref.)")
    ax3.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
                label=f"Mode switch M={_M_TRANS:.2f}")
    ax3.set_xlabel("Freestream Mach Number M₀")
    ax3.set_ylabel("Nozzle Exit Mach Number M₉")
    ax3.set_title("Nozzle Exit Mach vs Freestream Mach\n"
                  "(perfectly expanded, isentropic)",
                  fontweight="bold")
    ax3.legend()

    fig.suptitle("Figure 7 — Nozzle Geometry vs Mach Number",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig07_nozzle_geometry.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Nozzle gross thrust coefficient Cfg
# ══════════════════════════════════════════════════════════════════════════════

def fig08_nozzle_cfg() -> None:
    """Cfg vs Mach on q=1000 psf path and two fixed altitudes."""
    machs  = np.linspace(M_MIN, M_MAX, 20)
    alts_q = [altitude_for_q(M) for M in machs]

    cfg_q, cfg_25, cfg_32 = [], [], []
    for M, alt in zip(machs, alts_q):
        cfg_q.append(nozzle_cfg(M, alt))
        cfg_25.append(nozzle_cfg(M, 25_000.0))
        cfg_32.append(nozzle_cfg(M, 32_000.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Cfg vs Mach
    ax1.plot(machs, cfg_q,  "o-",  color=_TC[0], ms=5, label=f"q = {Q_PSF:.0f} psf")
    ax1.plot(machs, cfg_25, "s--", color=_TC[1], ms=5, label="Alt = 25 km")
    ax1.plot(machs, cfg_32, "^:",  color=_TC[2], ms=5, label="Alt = 32 km")
    ax1.axhline(ETA_NOZZLE, color="k", ls="--", lw=1.0,
                label=f"η_n = {ETA_NOZZLE} (kinetic limit)")
    ax1.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
                label=f"Mode switch M={_M_TRANS:.2f}")
    ax1.set_xlabel("Freestream Mach Number M₀")
    ax1.set_ylabel("Gross Thrust Coefficient  Cfg")
    ax1.set_title("Nozzle Gross Thrust Coefficient vs Mach",
                  fontweight="bold")
    ax1.legend(fontsize=8)

    definition = (
        "Definition:\n"
        "  Cfg = Fg / (Pt4 · A_t)\n"
        "      = η_n · (2/(γ+1))^((γ+1)/(2(γ-1)))\n"
        "           · √( 2γ²/(γ-1) · (1 − (P₀/Pt4)^((γ-1)/γ)) )\n"
        "Perfectly expanded nozzle  (P₉ = P₀)\n"
        "Increases with NPR; saturates at high speed"
    )
    ax1.text(0.02, 0.05, definition, transform=ax1.transAxes, fontsize=8,
             va="bottom",
             bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=3))

    # Right: Cfg vs NPR for parametric γ values
    NPR_range = np.logspace(0.5, 4.5, 200)
    for g_val, lbl in [(1.20, "γ = 1.20 (hot products, scram)"),
                        (1.30, "γ = 1.30 (intermediate)"),
                        (1.40, "γ = 1.40 (cold air, sea-level)")]:
        cfg_par = [ETA_NOZZLE
                   * (2.0/(g_val+1.0))**((g_val+1.0)/(2.0*(g_val-1.0)))
                   * np.sqrt(2.0*g_val**2/(g_val-1.0)
                             * (1.0-(1.0/NPR)**((g_val-1.0)/g_val)))
                   for NPR in NPR_range]
        ax2.semilogx(NPR_range, cfg_par, label=lbl)

    ax2.axhline(ETA_NOZZLE, color="k", ls="--", lw=0.9,
                label=f"η_n = {ETA_NOZZLE} (asymptote)")
    ax2.set_xlabel("Nozzle Pressure Ratio  NPR = Pt4/P₀")
    ax2.set_ylabel("Gross Thrust Coefficient  Cfg")
    ax2.set_title("Cfg vs NPR — Parametric (γ sensitivity)",
                  fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)

    fig.suptitle("Figure 8 — Nozzle Gross Thrust Coefficient Cfg vs Mach",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig08_nozzle_cfg.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Force accounting system
# ══════════════════════════════════════════════════════════════════════════════

def fig09_force_accounting() -> None:
    """Nose-to-tail control-volume schematic and thrust equations."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for a in (ax, ax2):
        a.set_aspect("equal")
        a.axis("off")

    # ──────────────────────────────────────────────────────────────────────
    # LEFT — engine block diagram with control volume boundary
    # ──────────────────────────────────────────────────────────────────────
    ax.set_xlim(-1.5, 11)
    ax.set_ylim(-1.8, 5)

    # Section fills
    secs = [
        (0.0, 2.5, "#6baed6", "Inlet\nRamps"),
        (2.5, 3.8, "#74c476", "Isolator"),
        (3.8, 5.5, "#fd8d3c", "Combustor"),
        (5.5, 8.0, "#9ecae1", "Nozzle"),
    ]
    for x1, x2, col, lbl in secs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, 0), x2 - x1, 2.0,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="k", lw=0.6, alpha=0.45))
        ax.text(0.5*(x1+x2), 1.0, lbl,
                ha="center", va="center", fontsize=9, fontweight="bold")

    # Outer engine casing
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 8, 2.0, boxstyle="round,pad=0.04",
        fill=False, edgecolor="k", lw=2.2))

    # Freestream arrows (left)
    for yf in [0.35, 1.0, 1.65]:
        ax.annotate("", xy=(0, yf), xytext=(-1.2, yf),
                    arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.8))
    ax.text(-1.0, 1.0,
            "Stn 0\nM₀, V₀\nρ₀, P₀",
            ha="center", va="center", fontsize=8.5, color="#2166ac",
            fontweight="bold")

    # Exhaust arrows (right) — fuel added to flow
    for yf in [0.30, 1.0, 1.70]:
        ax.annotate("", xy=(8.9, yf), xytext=(8.0, yf),
                    arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.8))
    ax.text(9.8, 1.0,
            "Stn 9\nM₉, V₉\nρ₉, P₀",
            ha="center", va="center", fontsize=8.5, color="#d73027",
            fontweight="bold")

    # Fuel injection arrow (from above into combustor)
    ax.annotate("", xy=(4.65, 2.0), xytext=(4.65, 3.5),
                arrowprops=dict(arrowstyle="->", color="#7b2d8b", lw=2.0))
    ax.text(4.65, 3.7, "Fuel\n(JP-10)\nṁ_f = φ · f_s · ṁ₀",
            ha="center", fontsize=8, color="#7b2d8b",
            bbox=dict(facecolor="#f3e5f5", edgecolor="#7b2d8b",
                      alpha=0.85, pad=2))

    # Thrust arrow
    ax.annotate("", xy=(-1.2, 3.5), xytext=(1.5, 3.5),
                arrowprops=dict(arrowstyle="<-", color="purple", lw=2.8))
    ax.text(0.15, 3.8, "NET THRUST  F",
            ha="center", fontsize=11, fontweight="bold", color="purple")


    # Station markers
    stn_x = [0.0, 2.5, 3.8, 5.5, 8.0]
    stn_id = ["0", "2", "3", "4", "9"]
    for xs, sid in zip(stn_x, stn_id):
        ax.plot([xs, xs], [-0.08, -0.40], "k-", lw=1.0)
        ax.text(xs, -0.62, f"Stn {sid}", ha="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#999",
                          pad=1.5, alpha=0.9))

    # Cp / friction note
    ax.text(4.0, -1.35,
            "Ramp pressure: oblique-shock Rankine-Hugoniot   |   "
            "Combustor: Rayleigh constant-area heat addition\n"
            "Nozzle: isentropic to P₉ = P₀   |   "
            "Friction: excluded (inviscid thermodynamic cycle)",
            ha="center", fontsize=8, color="#444",
            bbox=dict(facecolor="#f0f0f0", edgecolor="#bbb",
                      alpha=0.85, pad=3))

    ax.set_title(
        "Nose-to-Tail Control Volume\n"
        "(Stream-Thrust / Integral-Momentum Method)",
        fontweight="bold", fontsize=10,
    )

    # ──────────────────────────────────────────────────────────────────────
    # RIGHT — equations and accounting description
    # ──────────────────────────────────────────────────────────────────────
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 11)

    def txt(x, y, s, fs=9.5, fw="normal", col="k"):
        style = "italic" if fw == "italic" else "normal"
        weight = "normal" if fw == "italic" else fw
        ax2.text(x, y, s, fontsize=fs, fontweight=weight, fontstyle=style,
                 color=col, ha="left", va="center")

    # Shaded boxes
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.2, 7.6), 10.6, 2.7, boxstyle="round,pad=0.15",
        facecolor="#e8f4f8", edgecolor="#2166ac", lw=1.3, alpha=0.6))
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.2, 4.2), 10.6, 3.1, boxstyle="round,pad=0.15",
        facecolor="#fff7e6", edgecolor="#fd8d3c", lw=1.3, alpha=0.5))
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.2, 0.8), 10.6, 3.0, boxstyle="round,pad=0.15",
        facecolor="#f5f0ff", edgecolor="#7b2d8b", lw=1.3, alpha=0.5))

    txt(5.5, 10.3, "Force Accounting — Equations", 12, "bold", "k")
    txt(5.5, 9.8,  "Approach: 1-D Stream-Tube Integral Momentum", 10, "italic", "#2166ac")

    # Thrust equations box
    txt(0.5, 9.6,  "Gross thrust (momentum flux + pressure):", 9, "normal", "k")
    txt(0.5, 9.0,  "  Fg  =  ṁ₉ · V₉  +  (P₉ − P₀) · A₉",
        10, "normal", "#08306b")
    txt(0.5, 8.4,  "  For perfectly expanded nozzle (P₉ = P₀) → pressure term = 0",
        8.5, "italic", "#555")
    txt(0.5, 7.85, "  F_net  =  Fg  −  Fg_inlet  =  Fg  −  ṁ₀ · V₀  (ram drag)",
        9, "normal", "#08306b")

    # FnWa / Isp box
    txt(0.5, 7.1,  "Specific Thrust (FnWa):", 9, "bold", "k")
    txt(0.5, 6.6,  "  FnWa  =  F / ṁ_air  =  (1 + f) · V₉ − V₀    [N·s/kg_air]",
        10, "normal", "#7f2704")
    txt(0.5, 6.05, "  where  f = φ · f_stoich"
                   f"  =  {PHI} × {F_STOICH:.4f}  =  {PHI*F_STOICH:.4f}",
        9, "normal", "#7f2704")
    txt(0.5, 5.5,  "Specific Impulse:", 9, "bold", "k")
    txt(0.5, 5.0,  "  Isp  =  FnWa / (f · g₀)  =  F / (ṁ_fuel · g₀)    [s]",
        10, "normal", "#7f2704")
    txt(0.5, 4.45, f"  g₀ = {G0} m/s²", 9, "normal", "#555")

    # Surface integration box
    txt(0.5, 3.55, "Surface / Volume Pressure Accounting:", 9, "bold", "k")
    txt(0.5, 3.05, "  Inlet ramps: Oblique shock R-H relations → ΔP integrated",
        9, "normal", "#444")
    txt(0.5, 2.55, "  Isolator: Rayleigh / normal-shock Pt loss → adiabatic",
        9, "normal", "#444")
    txt(0.5, 2.05, "  Combustor: Constant-area Rayleigh heat addition",
        9, "normal", "#444")
    txt(0.5, 1.55, "  Nozzle: Isentropic expansion to P₉ = P_ambient",
        9, "normal", "#444")
    txt(0.5, 1.05, "  Skin friction: NOT modelled (inviscid thermodynamic cycle)",
        9, "italic", "#888")

    ax2.set_title("Force Accounting Equations", fontweight="bold", fontsize=10)

    fig.suptitle(
        "Figure 9 — Force Accounting System  (Nose-to-Tail Stream-Thrust Method)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig09_force_accounting.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Thrust and FnWa vs Mach, altitude, AoA
# ══════════════════════════════════════════════════════════════════════════════

def fig10_thrust_fnwa() -> None:
    """Net thrust and specific thrust on three trajectories + AoA sensitivity."""
    machs  = np.linspace(M_MIN, M_MAX, 16)
    alts_q = [altitude_for_q(M) for M in machs]

    def sweep(alts_in):
        thr, fnwa = [], []
        for M, alt in zip(machs, alts_in):
            r = safe_analyze(M, alt)
            thr.append(r["thrust"] if r else np.nan)
            fnwa.append(r["F_sp"]   if r else np.nan)
        return np.array(thr), np.array(fnwa)

    thr_q,  fnwa_q  = sweep(alts_q)
    thr_25, fnwa_25 = sweep([25_000.0] * len(machs))
    thr_32, fnwa_32 = sweep([32_000.0] * len(machs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def mode_line(a):
        a.axvline(_M_TRANS, color="gray", ls=":", lw=1.3)
        a.text(_M_TRANS + 0.15, a.get_ylim()[0], "←RAM|SCRAM→",
               fontsize=7.5, color="gray", va="bottom")

    # ── Thrust vs Mach ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(machs, thr_q  / 1e3, "o-",  color=_TC[0], ms=5, label="q = 1000 psf")
    ax.plot(machs, thr_25 / 1e3, "s--", color=_TC[1], ms=5, label="Alt = 25 km")
    ax.plot(machs, thr_32 / 1e3, "^:",  color=_TC[2], ms=5, label="Alt = 32 km")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Net Thrust  [kN]")
    ax.set_title(f"Net Thrust vs Mach  (φ = {PHI})", fontweight="bold")
    ax.legend()
    mode_line(ax)

    # ── FnWa vs Mach ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(machs, fnwa_q,  "o-",  color=_TC[0], ms=5, label="q = 1000 psf")
    ax.plot(machs, fnwa_25, "s--", color=_TC[1], ms=5, label="Alt = 25 km")
    ax.plot(machs, fnwa_32, "^:",  color=_TC[2], ms=5, label="Alt = 32 km")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Specific Thrust FnWa  [N·s/kg_air]")
    ax.set_title(f"Specific Thrust (FnWa) vs Mach  (φ = {PHI})", fontweight="bold")
    ax.legend()
    mode_line(ax)

    # ── Thrust vs altitude at fixed Mach ──────────────────────────────────
    ax = axes[1, 0]
    alts_r = np.linspace(10_000, 40_000, 16)
    for i, Mf in enumerate([3.0, 5.0, 7.0, 9.0]):
        thr_alt = []
        for alt in alts_r:
            r = safe_analyze(Mf, alt)
            thr_alt.append(r["thrust"] if r else np.nan)
        ax.plot(alts_r / 1e3, np.array(thr_alt) / 1e3,
                color=_TC[i], ms=5, label=f"M = {Mf}")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Net Thrust  [kN]")
    ax.set_title(f"Net Thrust vs Altitude  (φ = {PHI})", fontweight="bold")
    ax.legend()

    # ── FnWa vs AoA (inlet efficiency + capture scaling) ──────────────────
    ax = axes[1, 1]
    aoa_arr = np.linspace(-4.0, 8.0, 20)
    for i, Mf in enumerate([3.0, 5.5, 7.0, 9.0]):
        alt0 = altitude_for_q(Mf)
        r0   = safe_analyze(Mf, alt0)
        if r0 is None:
            continue
        eta0 = inlet_recovery_aoa(Mf, 0.0, alt=alt0)
        cr0  = capture_ratio(Mf, 0.0)
        fnwa_aoa = []
        for aoa in aoa_arr:
            eta_a = inlet_recovery_aoa(Mf, aoa, alt=alt0)
            cr_a  = capture_ratio(Mf, aoa)
            # Scale: inlet Pt loss → lowers effective combustor entry Pt;
            # capture ratio → scales mass flow and hence thrust proportionally
            scale = (eta_a / max(eta0, 1e-6)) * (cr_a / max(cr0, 1e-6))
            fnwa_aoa.append(r0["F_sp"] * scale)
        ax.plot(aoa_arr, fnwa_aoa, color=_TC[i], ms=5, label=f"M = {Mf}")

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Angle of Attack  [deg]")
    ax.set_ylabel("Approx. FnWa  [N·s/kg_air]")
    ax.set_title("FnWa vs AoA\n"
                 "(scaled via inlet Pt + capture ratio)",
                 fontweight="bold")
    ax.legend()
    ax.text(0.02, 0.04,
            "Note: AoA effect approximated by scaling\n"
            "baseline FnWa with ηinlet(AoA)/ηinlet(0°)\n"
            "and A₀/Ac(AoA)/A₀/Ac(0°)",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=2))

    fig.suptitle("Figure 10 — Net Thrust and Specific Thrust (FnWa)\n"
                 "vs Mach Number, Altitude, and Angle of Attack",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig10_thrust_fnwa.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Specific impulse
# ══════════════════════════════════════════════════════════════════════════════

def fig11_isp() -> None:
    """Isp on three trajectories, vs altitude, and vs AoA."""
    machs  = np.linspace(M_MIN, M_MAX, 16)
    alts_q = [altitude_for_q(M) for M in machs]

    def isp_sweep(alts_in):
        isps = []
        for M, alt in zip(machs, alts_in):
            r = safe_analyze(M, alt)
            isps.append(r["Isp"] if r else np.nan)
        return np.array(isps)

    isp_q  = isp_sweep(alts_q)
    isp_25 = isp_sweep([25_000.0] * len(machs))
    isp_32 = isp_sweep([32_000.0] * len(machs))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Isp vs Mach ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(machs, isp_q,  "o-",  color=_TC[0], ms=5, label="q = 1000 psf")
    ax.plot(machs, isp_25, "s--", color=_TC[1], ms=5, label="Alt = 25 km")
    ax.plot(machs, isp_32, "^:",  color=_TC[2], ms=5, label="Alt = 32 km")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(_M_TRANS, color="gray", ls=":", lw=1.3,
               label=f"Mode switch M={_M_TRANS:.2f}")
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Specific Impulse  Isp  [s]")
    ax.set_title(f"Isp vs Mach  (φ = {PHI})", fontweight="bold")
    ax.set_xlim(M_MIN, M_MAX)
    ax.legend()

    # ── Isp vs altitude ────────────────────────────────────────────────────
    ax = axes[1]
    alts_r = np.linspace(10_000, 40_000, 16)
    for i, Mf in enumerate([3.0, 5.0, 7.0, 9.0]):
        isp_alt = []
        for alt in alts_r:
            r = safe_analyze(Mf, alt)
            isp_alt.append(r["Isp"] if r else np.nan)
        ax.plot(alts_r / 1e3, isp_alt, color=_TC[i], ms=5, label=f"M = {Mf}")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Specific Impulse  Isp  [s]")
    ax.set_title(f"Isp vs Altitude  (φ = {PHI})", fontweight="bold")
    ax.legend()

    # ── Isp vs AoA ────────────────────────────────────────────────────────
    ax = axes[2]
    aoa_arr = np.linspace(-4.0, 8.0, 20)
    for i, Mf in enumerate([3.0, 5.5, 7.0, 9.0]):
        alt0 = altitude_for_q(Mf)
        r0   = safe_analyze(Mf, alt0)
        if r0 is None:
            continue
        eta0 = inlet_recovery_aoa(Mf, 0.0, alt=alt0)
        cr0  = capture_ratio(Mf, 0.0)
        isp_aoa = []
        for aoa in aoa_arr:
            eta_a = inlet_recovery_aoa(Mf, aoa, alt=alt0)
            cr_a  = capture_ratio(Mf, aoa)
            scale = (eta_a / max(eta0, 1e-6)) * (cr_a / max(cr0, 1e-6))
            isp_aoa.append(r0["Isp"] * scale)
        ax.plot(aoa_arr, isp_aoa, color=_TC[i], ms=5, label=f"M = {Mf}")

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Angle of Attack  [deg]")
    ax.set_ylabel("Approx. Isp  [s]")
    ax.set_title("Isp vs Angle of Attack\n"
                 "(scaled via inlet Pt + capture ratio)",
                 fontweight="bold")
    ax.legend()
    ax.text(0.02, 0.04,
            "Note: same AoA-scaling approximation as Fig 10",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(facecolor="lightyellow", edgecolor="#aaa", pad=2))

    fig.suptitle("Figure 11 — Specific Impulse (Isp)\n"
                 "vs Mach Number, Altitude, and Angle of Attack",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig11_isp.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print(" JP-10 Dual-Mode Ram/Scramjet — Comprehensive Analysis Plots")
    print("=" * 65)
    print(f"  Output: {OUTDIR}\n")

    steps = [
        ("Fig  1 — Nose-to-tail flowpath geometry",     fig01_flowpath),
        ("Fig  2 — Component efficiencies",             fig02_efficiencies),
        ("Fig  3 — Thrust-to-weight (q=1000 psf)",      fig03_thrust_to_weight),
        ("Fig  4 — Inlet geometry & shock angles",      fig04_inlet_geometry),
        ("Fig  5 — Inlet capture ratio vs Mach/AoA",   fig05_capture_ratio),
        ("Fig  6 — Inlet Pt recovery vs Mach/AoA",     fig06_inlet_pt_recovery),
        ("Fig  7 — Nozzle geometry vs Mach",            fig07_nozzle_geometry),
        ("Fig  8 — Nozzle Cfg vs Mach",                 fig08_nozzle_cfg),
        ("Fig  9 — Force accounting system",            fig09_force_accounting),
        ("Fig 10 — Thrust & FnWa vs Mach/alt/AoA",     fig10_thrust_fnwa),
        ("Fig 11 — Isp vs Mach/alt/AoA",               fig11_isp),
    ]

    t0 = time.time()
    for desc, fn in steps:
        t1 = time.time()
        print(f"  {desc}…", flush=True, end=" ")
        try:
            fn()
        except Exception as exc:
            print(f"\n  !! ERROR: {exc}")
        print(f"({time.time()-t1:.1f}s)")

    print(f"\n  Done — {len(steps)} figures in {time.time()-t0:.1f} s")
    print(f"  Saved to: {OUTDIR}/")


if __name__ == "__main__":
    main()