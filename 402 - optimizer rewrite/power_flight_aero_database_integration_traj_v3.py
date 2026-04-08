"""
trajectory_lfrj.py
==================
Powered flight trajectory for an HCM using:

  Phase 1 : Solid Rocket Boost  — unchanged from powered_flight_trajectory_code_v2.py
  Phase 2 : Liquid Fuel Ramjet (LFRJ) Cruise — JP-10 dual-mode ram/scramjet
             driven by combined_cycle_liquid_ram_scram.analyze()
  Phase 3 : Unpowered Terminal Descent — unchanged

ALL SFRJ functions are preserved below for reference / comparison.
The cruise EOM, simulate, run_trajectory, and plots have LFRJ counterparts
prefixed with `lfrj_` or suffixed with `_lfrj`.

Key differences vs SFRJ:
  - Cruise calls lfrj_performance(M, h, phi) not sfrj_thrust_and_isp(M, h, A_burn)
  - No solid fuel grain burn-area state variable (A_burn unused in LFRJ cruise)
  - PHI_LFRJ is the equivalence ratio; held constant at 0.8 for cruise
  - LFRJ_CRUISE_ALT raised to 20 km for better ram/scramjet q & inlet efficiency
  - Bug fix: original lfrj_performance multiplied thrust by 1000 (N→wrong);
    corrected here — thrust returned directly in [N] from analyze()

Aerodynamics:
  - CFD-database-driven aero model (replaces analytic cd0/cl_alpha/ki functions)
  - Surrogate models trained on parsed CFD files via build_global_database()
  - Call init_aero_models() before run_trajectory_lfrj() to load the database
  - Falls back gracefully if no CFD data directory is found

Usage:
    python trajectory_lfrj.py            # physics simulation + plots
    python trajectory_lfrj.py --optimize # also run Dymos optimal control
"""

import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

import os
import glob
import re
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# ── LFRJ cycle solver ─────────────────────────────────────────────────────────
from main import analyze
from inlet import compute_inlet
from atmosphere import freestream
from gas_dynamics import FlowState  

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
G0    = 9.80665        # m/s²
RGAS  = 287.058        # J/(kg·K)
GAMMA = 1.4
RE    = 6_371_000.0    # m

# ── HCM mass budget  (total 1088 kg, integral booster design) ─────────────────
M0             = 1088.0    # kg  gross mass at F-35 release
M_PROP_BOOST   = 280.0     # kg  solid rocket propellant (boosts to M2.5+)
M_FUEL_SFRJ    = 130.0     # kg  HTPB solid fuel grain (SFRJ config, kept for reference)
M_STRUCT       = M0 - M_PROP_BOOST - M_FUEL_SFRJ   # 678 kg (airframe+inlet+nozzle+warhead)

# LFRJ mass budget — liquid JP-10 replaces HTPB grain; same total fuel mass
M_FUEL_LFRJ    = 130.0     # kg  liquid JP-10 fuel

# ── Aero reference ────────────────────────────────────────────────────────────
# S_REF must match the reference area used by the CFD solver that produced the
# aero database.  The ogive CFD runs used a unit reference area of 1.0 m².
# v3 incorrectly changed this to 0.10 m² (the body cross-section), which made
# drag 10x too low, T/D ~ 15 during cruise, and range ~1600 km instead of ~500 km.
S_REF    = 1.00           # m²  CFD reference area (must match aero database)
D_BODY   = 0.34           # m   body diameter

# ── SFRJ hardware geometry (preserved, not used in LFRJ cruise) ───────────────
A_INLET   = 0.012         # m²  inlet capture area (M4 design point, SFRJ)
A_THROAT  = 0.008         # m²  nozzle throat area
AREA_RATIO = 6.5          # Ae/At  nozzle expansion ratio

# ── SFRJ fuel grain (preserved) ───────────────────────────────────────────────
RHO_FUEL   = 920.0        # kg/m³  HTPB density
A_BURN_0   = 0.18         # m²    initial fuel grain burning surface area
A_BURN_MIN = 0.04         # m²    minimum (near-burnout) burning area
A_REG      = 1.44e-5      # Saint-Robert regression coefficient (HTPB)
N_REG      = 0.35         # pressure exponent

# ── SFRJ thermodynamic constants (preserved) ─────────────────────────────────
GAMMA_C    = 1.25
CP_C       = 1800.0       # J/(kg·K)
ETA_COMB   = 0.94
H_FUEL     = 42.5e6       # J/kg  HTPB LHV
FAR_DESIGN = 0.060

# ── LFRJ-specific constants ───────────────────────────────────────────────────
PHI_LFRJ        = 0.80    # equivalence ratio for JP-10 cruise combustion
LFRJ_MACH_MIN   = 2.0     # minimum Mach for LFRJ ignition
LFRJ_MACH_MAX   = 6.0     # maximum Mach — physical limit of ram/scramjet cycle

# Cruise altitude derived from stoichiometric range balance:
#   Range = M_FUEL_LFRJ / (phi * F_STOICH * rho(h) * A_INLET)
# Solve for rho, then invert the ISA to find h.
# Uses F_STOICH from config so it stays consistent with analyze().
from config import F_STOICH as _F_STOICH_CFG
_R_TARGET_M   = 500_000.0   # m  design range target for cruise phase
_rho_cruise   = M_FUEL_LFRJ / (_R_TARGET_M * PHI_LFRJ * _F_STOICH_CFG * A_INLET)
# Invert ISA stratosphere (11–25 km): rho = P/(R*T), P=22632*exp(-1.577e-4*(h-11000)), T=216.65
import math as _math
_P_cruise     = _rho_cruise * RGAS * 216.65
_h_strato     = 11_000.0 - _math.log(_P_cruise / 22_632.1) / 0.0001577
# If density is too high the altitude falls in the troposphere; clamp and recompute
if _h_strato < 11_000.0:
    # Invert troposphere: T=288.15-0.0065h, P=101325*(T/288.15)^5.2561
    # rho = P/(R*T) -> solve numerically (simple bisection over 0-11km)
    _lo, _hi = 0.0, 11_000.0
    for _ in range(60):
        _hm = 0.5*(_lo+_hi)
        _Tm = 288.15 - 0.0065*_hm
        _rm = 101_325.0*(_Tm/288.15)**5.2561 / (RGAS*_Tm)
        if _rm > _rho_cruise: _lo = _hm
        else: _hi = _hm
    _h_strato = 0.5*(_lo+_hi)
LFRJ_CRUISE_ALT = float(round(_h_strato / 500) * 500)  # round to nearest 500 m

# analyze() in main.py sizes all mass flows and thrust to config.A_CAPTURE = 0.05 m².
# The missile inlet is A_INLET = 0.012 m².  Every thrust and mdot_fuel value
# returned by analyze() must be scaled by this ratio before use in the EOM.
# Importing here keeps the scale factor derived from both sources — no hardcoding.
from config import A_CAPTURE as _A_CAPTURE_CFG
LFRJ_AREA_SCALE = A_INLET / _A_CAPTURE_CFG   # 0.012 / 0.05 = 0.24

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD ATMOSPHERE (1976)
# ──────────────────────────────────────────────────────────────────────────────
def atmosphere(h_m):
    """Scalar atmosphere — returns (rho, P, T, a)."""
    h = float(np.clip(h_m, 0.0, 79_900.0))
    T0, P0 = 288.15, 101_325.0
    if h <= 11_000:
        T = T0 - 0.0065 * h
        P = P0 * (T / T0) ** 5.2561
    elif h <= 25_000:
        T = 216.65
        P = 22_632.1 * np.exp(-0.0001577 * (h - 11_000))
    elif h <= 47_000:
        T = 216.65 + 0.003 * (h - 25_000)
        P = 2_488.4 * (T / 216.65) ** (-34.164)
    else:
        T = 282.65 - 0.0028 * (h - 47_000)
        P = 141.3 * (T / 282.65) ** 17.082
    rho = P / (RGAS * T)
    a   = np.sqrt(GAMMA * RGAS * T)
    return rho, P, T, a

def atm_vec(h_arr):
    """Vectorised atmosphere."""
    h_arr = np.atleast_1d(np.asarray(h_arr, dtype=float))
    out = [atmosphere(hi) for hi in h_arr]
    return tuple(np.array([o[i] for o in out]) for i in range(4))

# ──────────────────────────────────────────────────────────────────────────────
# AERODYNAMICS — CFD DATABASE-DRIVEN
# ──────────────────────────────────────────────────────────────────────────────

def parse_cfd_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if "Function Data:" in line:
            data_start = i + 2
            break

    if data_start is None:
        raise RuntimeError(f"Cannot find 'Function Data:' section in {filepath}")

    data = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        tokens = line.strip().split()
        if len(tokens) != 5:
            continue
        mach, q, alpha, beta, f = map(float, tokens)
        data.append([mach, q, alpha, beta, f])

    df = pd.DataFrame(data, columns=["Mach", "q", "alpha", "beta", "F"])
    return df


def find_files(directory, model_name, param, file_type="Totl"):
    """
    Find files for a given parameter and type (Totl, Pres, Visc).
    """
    if file_type not in ["Totl", "Pres", "Visc"]:
        pattern = f"{model_name}.{param}.dat"
    else:
        pattern = f"{model_name}.{param}.{file_type}.dat"
    return glob.glob(os.path.join(directory, pattern))


def extract_geometry_from_dir(dirname, prefix="ogive"):
    """
    Extracts geometry parameters from directory name.
    e.g., ogive_4_2_3 -> {"x1": 4, "x2": 2, "x3": 3}
    If the directory is exactly the prefix (no underscore suffix), return {}.
    """
    if not dirname.startswith(prefix + "_"):
        return {}
    parts = dirname.replace(prefix + "_", "").split("_")
    geom = {}
    for i, p in enumerate(parts):
        try:
            geom[f"x{i + 1}"] = int(p)
        except ValueError:
            try:
                geom[f"x{i + 1}"] = float(p)
            except ValueError:
                continue
    return geom


def collect_data_across_dirs(param, model_prefix="ogive", file_type="Totl"):
    """
    Collect data for a parameter across directories.
    """
    all_data = []

    is_derived = param == "L/D"
    required_params = ["CLw", "CDw"] if is_derived else [param]

    for entry in os.scandir('.'):
        if not entry.is_dir():
            continue
        if not (entry.name == model_prefix or entry.name.startswith(model_prefix + "_")):
            continue

        geom_params = extract_geometry_from_dir(entry.name, prefix=model_prefix)

        data_frames = {}
        for req_param in required_params:
            files = find_files(entry.path, model_prefix, req_param, file_type)
            if not files:
                print(f"Skipping {entry.name}: no {file_type} file for {req_param}")
                break
            try:
                df = parse_cfd_file(files[0])
                for k, v in geom_params.items():
                    df[k] = v
                df["config_dir"] = entry.name
                data_frames[req_param] = df
            except RuntimeError as e:
                print(f"Error reading {files[0]}: {e}")
                break
        else:
            if is_derived:
                df_cl = data_frames["CLw"]
                df_cd = data_frames["CDw"]
                merged = pd.merge(
                    df_cl, df_cd,
                    on=["Mach", "q", "alpha", "beta"] + list(geom_params.keys()),
                    suffixes=("_CL", "_CD")
                )
                merged["F"] = merged["F_CL"] / merged["F_CD"]
                merged["config_dir"] = df_cl["config_dir"].iloc[0]
                all_data.append(merged[["Mach", "q", "alpha", "beta", "F"] + list(geom_params.keys()) + ["config_dir"]])
            else:
                all_data.append(data_frames[param])

    if not all_data:
        raise RuntimeError(
            f"No data found for parameter {param} ({file_type}) in any matching directory for prefix '{model_prefix}'.")

    return pd.concat(all_data, ignore_index=True)


def verify_aero_data(merged_df):
    """
    Verify the loaded aerodynamic data for reasonableness.
    """
    print("\n" + "=" * 60)
    print("AERODYNAMIC DATA VERIFICATION")
    print("=" * 60)

    if "CLw" in merged_df.columns and "CDw_inviscid" in merged_df.columns and "CDw_viscous" in merged_df.columns:
        merged_df["CDw_total"] = merged_df["CDw_inviscid"] + merged_df["CDw_viscous"]
        valid_mask = (merged_df["CDw_total"] > 0.001) & (merged_df["CLw"].notna()) & (merged_df["CDw_total"].notna())
        if valid_mask.sum() > 0:
            ld_ratio = merged_df.loc[valid_mask, "CLw"] / merged_df.loc[valid_mask, "CDw_total"]
            print(f"\nL/D Ratio Statistics:")
            print(f"  Min:    {ld_ratio.min():.3f}")
            print(f"  Max:    {ld_ratio.max():.3f}")
            print(f"  Mean:   {ld_ratio.mean():.3f}")
            print(f"  Median: {ld_ratio.median():.3f}")
            high_ld = ld_ratio > 4.5
            if high_ld.sum() > 0:
                print(f"\n  WARNING: {high_ld.sum()} points have L/D > 4.5")
                print(f"  Maximum L/D found: {ld_ratio.max():.3f}")

    if "CLw" in merged_df.columns:
        cl_valid = merged_df["CLw"].notna()
        print(f"\nCL Statistics ({cl_valid.sum()} valid points):")
        print(f"  Min:  {merged_df.loc[cl_valid, 'CLw'].min():.4f}")
        print(f"  Max:  {merged_df.loc[cl_valid, 'CLw'].max():.4f}")
        print(f"  Mean: {merged_df.loc[cl_valid, 'CLw'].mean():.4f}")

    if "CDw_inviscid" in merged_df.columns:
        cd_valid = merged_df["CDw_inviscid"].notna()
        print(f"\nCD Inviscid Statistics ({cd_valid.sum()} valid points):")
        print(f"  Min:  {merged_df.loc[cd_valid, 'CDw_inviscid'].min():.4f}")
        print(f"  Max:  {merged_df.loc[cd_valid, 'CDw_inviscid'].max():.4f}")
        print(f"  Mean: {merged_df.loc[cd_valid, 'CDw_inviscid'].mean():.4f}")

    if "CDw_viscous" in merged_df.columns:
        cd_valid = merged_df["CDw_viscous"].notna()
        print(f"\nCD Viscous Statistics ({cd_valid.sum()} valid points):")
        print(f"  Min:  {merged_df.loc[cd_valid, 'CDw_viscous'].min():.4f}")
        print(f"  Max:  {merged_df.loc[cd_valid, 'CDw_viscous'].max():.4f}")
        print(f"  Mean: {merged_df.loc[cd_valid, 'CDw_viscous'].mean():.4f}")

    print(f"\nData Coverage:")
    print(f"  Total points: {len(merged_df)}")
    if "Mach" in merged_df.columns:
        print(f"  Mach range: [{merged_df['Mach'].min():.2f}, {merged_df['Mach'].max():.2f}]")
    if "alpha" in merged_df.columns:
        print(f"  Alpha range: [{merged_df['alpha'].min():.2f}, {merged_df['alpha'].max():.2f}] deg")
    if "q" in merged_df.columns:
        print(f"  Dynamic pressure range: [{merged_df['q'].min():.2e}, {merged_df['q'].max():.2e}] bar")

    print("=" * 60 + "\n")


def build_global_database(model_prefix, surrogate_type="linear"):
    """
    Build a global merged DataFrame and train surrogate models for:
      - CLw_inviscid, CDw_inviscid, CMn_inviscid  (from Pres files)
      - CDw_viscous                                (from Visc files)
      - MaxQdotTotalQdotConvection                 (from Totl files)

    Returns merged DataFrame, models dict, scalers dict,
    feature_cols_inviscid, feature_cols_viscous.
    """
    inviscid_params = ["CLw", "CDw", "CMn"]
    viscous_params  = ["CDw"]
    heat_params     = ["MaxQdotTotalQdotConvection"]

    collected = {}

    for param in inviscid_params:
        try:
            df = collect_data_across_dirs(param, model_prefix, file_type="Pres")
            collected[f"{param}_inviscid"] = df.rename(columns={"F": f"{param}_inviscid"})
        except RuntimeError as e:
            print(f"Warning: could not collect {param} (Pres): {e}")
            collected[f"{param}_inviscid"] = None

    for param in viscous_params:
        try:
            df = collect_data_across_dirs(param, model_prefix, file_type="Visc")
            collected[f"{param}_viscous"] = df.rename(columns={"F": f"{param}_viscous"})
        except RuntimeError as e:
            print(f"Warning: could not collect {param} (Visc): {e}")
            collected[f"{param}_viscous"] = None

    for param in heat_params:
        try:
            df = collect_data_across_dirs(param, model_prefix, file_type="AHHH")
            collected[param] = df.rename(columns={"F": param})
        except RuntimeError as e:
            print(f"Warning: could not collect {param}: {e}")
            collected[param] = None

    geom_cols = []
    for df in collected.values():
        if df is not None:
            geom_cols = [c for c in df.columns if re.match(r"x\d+", c)]
            break

    keycols = ["Mach", "q", "alpha", "beta"] + geom_cols

    dfs_to_merge = []
    for param, df in collected.items():
        if df is not None:
            cols_needed = keycols + [param]
            for g in geom_cols:
                if g not in df.columns:
                    df[g] = np.nan
            dfs_to_merge.append(df[cols_needed].copy())

    if not dfs_to_merge:
        raise RuntimeError("No parameter data found to build global database.")

    merged = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        merged = pd.merge(merged, df, on=keycols, how="outer")

    for col in ["Mach", "q", "alpha", "beta"] + geom_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["Mach", "q", "alpha", "beta"])

    for g in geom_cols:
        if merged[g].isna().any():
            merged[g] = merged[g].fillna(merged[g].median(skipna=True))

    # Convert dynamic pressure from bar to Pa
    merged['q'] = merged['q'] * 1e5

    verify_aero_data(merged)

    outcsv = f"global_database_{model_prefix}.csv"
    merged.to_csv(outcsv, index=False)
    print(f"Global database written to: {outcsv}")

    feature_cols_inviscid = ["Mach", "q", "alpha", "beta"] + geom_cols

    merged["Re"] = merged["Mach"] * 1e6  # placeholder — replace with actual Re
    feature_cols_viscous = ["Mach", "Re", "alpha", "beta"] + geom_cols

    models  = {}
    scalers = {}

    X_inviscid        = merged[feature_cols_inviscid].values
    X_inviscid_scaler = StandardScaler() if surrogate_type == "gpr" else None
    X_inviscid_scaled = (X_inviscid_scaler.fit_transform(X_inviscid)
                         if surrogate_type == "gpr" else X_inviscid)

    X_viscous        = merged[feature_cols_viscous].values
    X_viscous_scaler = StandardScaler() if surrogate_type == "gpr" else None
    X_viscous_scaled = (X_viscous_scaler.fit_transform(X_viscous)
                        if surrogate_type == "gpr" else X_viscous)

    scalers["X_inviscid_scaler"] = X_inviscid_scaler
    scalers["X_viscous_scaler"]  = X_viscous_scaler

    def _make_model(surrogate_type, n_features):
        if surrogate_type == "gpr":
            kernel = (C(1.0, (1e-3, 1e3))
                      * Matern(length_scale=np.ones(n_features),
                               length_scale_bounds=(1e-2, 1e3))
                      + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1)))
            return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=3, random_state=0)
        return LinearRegression()

    def _train(target, X_scaled, X_raw, feature_cols, surrogate_type):
        if target not in merged.columns:
            return None
        y    = merged[target].values
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            print(f"Skipping training for {target}: not enough valid samples ({mask.sum()})")
            return None
        X_tr = X_scaled[mask] if surrogate_type == "gpr" else merged.loc[mask, feature_cols].values
        y_tr = y[mask]
        label = "GPR" if surrogate_type == "gpr" else "Linear Regression"
        print(f"Training {label} for {target} on {X_tr.shape[0]} samples")
        m = _make_model(surrogate_type, X_tr.shape[1])
        m.fit(X_tr, y_tr)
        return m

    for target in ["CLw_inviscid", "CDw_inviscid", "CMn_inviscid"]:
        m = _train(target, X_inviscid_scaled, X_inviscid, feature_cols_inviscid, surrogate_type)
        if m is not None:
            models[target] = m

    for target in ["CDw_viscous"]:
        m = _train(target, X_viscous_scaled, X_viscous, feature_cols_viscous, surrogate_type)
        if m is not None:
            models[target] = m

    for target in heat_params:
        m = _train(target, X_inviscid_scaled, X_inviscid, feature_cols_inviscid, surrogate_type)
        if m is not None:
            models[target] = m

    return merged, models, scalers, feature_cols_inviscid, feature_cols_viscous


def get_models_for_prefix(model_prefix, surrogate_type="linear"):
    """
    Convenience wrapper — returns (model_dict, scalers, feature_cols, merged_df).
    """
    merged, models, scalers, feature_cols_inviscid, feature_cols_viscous = \
        build_global_database(model_prefix, surrogate_type)

    model_dict = {
        'inviscid': {
            'Cd': models.get("CDw_inviscid"),
            'Cl': models.get("CLw_inviscid"),
            'Cm': models.get("CMn_inviscid"),
        },
        'viscous': {
            'Cd': models.get("CDw_viscous"),
        },
        'heat': {
            'qdot': models.get("MaxQdotTotalQdotConvection"),
        }
    }

    feature_cols = {
        'inviscid': feature_cols_inviscid,
        'viscous':  feature_cols_viscous,
    }

    return model_dict, scalers, feature_cols, merged


# ── Global aero model state (loaded once via init_aero_models) ────────────────
MODEL_DICT  = None
SCALERS     = None
FEATURE_COLS = None
GEOM_PARAMS  = None
MERGED_DF   = None   # raw CFD database retained for plot sweeps


def init_aero_models(prefix="ogive", geom_params=None, surrogate="linear"):
    """
    Load and train aerodynamic surrogate models from CFD database files.
    Must be called before run_trajectory_lfrj().
    """
    global MODEL_DICT, SCALERS, FEATURE_COLS, GEOM_PARAMS, MERGED_DF

    MODEL_DICT, SCALERS, FEATURE_COLS, MERGED_DF = get_models_for_prefix(
        prefix, surrogate_type=surrogate
    )
    GEOM_PARAMS = geom_params if geom_params is not None else {}
    print("Aero models initialized.")


def aero(M, alpha_deg, rho, V, beta=0.0):
    """
    Aerodynamic forces via CFD-database surrogate models.

    Parameters
    ----------
    M         : Mach number (scalar or array)
    alpha_deg : angle of attack [deg]
    rho       : air density [kg/m³]
    V         : airspeed [m/s]
    beta      : sideslip angle [deg], default 0

    Returns
    -------
    CL, CD, L [N], D [N], q [Pa]
    """
    if MODEL_DICT is None:
        raise RuntimeError("Aero models not initialized. Call init_aero_models() first.")

    q = 0.5 * rho * V ** 2

    def build_features(feature_names):
        feats = []
        for name in feature_names:
            if name == "Mach":
                feats.append(np.asarray(M))
            elif name == "q":
                feats.append(np.asarray(q))
            elif name == "Re":
                feats.append(np.asarray(M) * 1e6)
            elif name == "alpha":
                feats.append(np.asarray(alpha_deg))
            elif name == "beta":
                feats.append(np.asarray(beta))
            else:
                feats.append(np.full_like(np.asarray(M), GEOM_PARAMS.get(name, 0.0)))
        return np.column_stack(feats)

    X_inv = build_features(FEATURE_COLS['inviscid'])
    if SCALERS["X_inviscid_scaler"] is not None:
        X_inv = SCALERS["X_inviscid_scaler"].transform(X_inv)

    CL     = MODEL_DICT['inviscid']['Cl'].predict(X_inv)[0]
    CD_inv = MODEL_DICT['inviscid']['Cd'].predict(X_inv)[0]

    X_visc = build_features(FEATURE_COLS['viscous'])
    if SCALERS["X_viscous_scaler"] is not None:
        X_visc = SCALERS["X_viscous_scaler"].transform(X_visc)

    CD_visc = MODEL_DICT['viscous']['Cd'].predict(X_visc)[0]

    CD = CD_inv + CD_visc

    # Clamp coefficients to physically plausible values.
    # The linear surrogate can extrapolate outside the training envelope and
    # produce negative CD (impossible) or L/D > 11 (database max is ~11.3).
    # CD_min: use the smallest total CD seen anywhere in the database (~0.001).
    # CL: allow the full range the surrogate predicts; do not clip lift.
    CD = max(float(CD), 1e-3)   # enforce CD > 0

    L = q * S_REF * CL
    D = q * S_REF * CD

    return CL, CD, L, D, q


def solve_alpha_trim(M, rho, V, target_L=None, target_n=None,
                     mass=None, g=9.81, alpha_bounds=(-10.0, 15.0)):
    """
    Solve for trim angle of attack [deg] such that Lift == target_L
    or load factor n == target_n, using bisection.
    """
    def residual(alpha):
        CL, CD, L, D, q = aero(M, alpha, rho, V)
        if target_L is not None:
            return L - target_L
        elif target_n is not None:
            return L / (mass * g) - target_n
        else:
            raise ValueError("Must specify target_L or target_n")

    a_lo, a_hi = alpha_bounds
    f_lo = residual(a_lo)
    f_hi = residual(a_hi)

    for _ in range(10):
        if f_lo * f_hi < 0:
            break
        a_lo -= 5
        a_hi += 5
        f_lo = residual(a_lo)
        f_hi = residual(a_hi)

    if f_lo * f_hi > 0:
        alphas = np.linspace(a_lo, a_hi, 25)
        vals   = [abs(residual(a)) for a in alphas]
        return alphas[np.argmin(vals)]

    for _ in range(40):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = residual(a_mid)
        if f_lo * f_mid <= 0:
            a_hi = a_mid
            f_hi = f_mid
        else:
            a_lo = a_mid
            f_lo = f_mid

    return 0.5 * (a_lo + a_hi)


# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION — SOLID ROCKET BOOSTER  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def boost_thrust(M, h):
    """
    Solid rocket motor thrust [N].
    Vacuum thrust ~105 kN; corrected for ambient back-pressure.
    """
    _, P0, _, _ = atmosphere(h)
    Pc_boost = 8.0e6              # N/m²  chamber pressure (8 MPa)
    At_boost = 0.012              # m²    booster nozzle throat
    Cf_vac   = 1.65               # vacuum thrust coefficient
    Ae_boost = At_boost * 8.0    # expansion ratio 8
    T = Cf_vac * Pc_boost * At_boost - (P0 - 0.0) * Ae_boost
    return float(np.clip(T, 70_000.0, 115_000.0))

def boost_isp(M=None, h=None):
    """Solid rocket Isp — ~265 s vacuum."""
    return 265.0   # s

# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION — SOLID FUEL RAMJET (SFRJ)   ← PRESERVED, NOT REMOVED
# Physics reference: Waltrup et al. (1976); Netzer & Gany (1993)
# ──────────────────────────────────────────────────────────────────────────────
def sfrj_inlet_recovery(M):
    """
    Total-pressure recovery η_r for a 2-shock mixed-compression inlet.
    MIL-E-5007D correlation.
    """
    M = float(M)
    if M <= 1.0:
        return 1.0
    elif M <= 5.0:
        eta = 1.0 - 0.075 * (M - 1.0)**1.35
    else:
        eta = 0.6 - 0.1 * (M - 5.0)
    return float(np.clip(eta, 0.05, 1.0))

def sfrj_total_conditions(M, h):
    """Ram-air total temperature and total pressure entering combustion chamber."""
    rho, P0, T0, a = atmosphere(h)
    gm = GAMMA
    Tt2 = T0  * (1.0 + (gm - 1)/2 * M**2)
    Pt2 = P0  * (1.0 + (gm - 1)/2 * M**2)**(gm/(gm-1))
    eta_r = sfrj_inlet_recovery(M)
    Pt3 = eta_r * Pt2
    return Tt2, Pt3

def sfrj_chamber_pressure(mdot_air, Tt4, Pc_guess=1.5e6):
    """Iterate to find combustion chamber pressure Pc [Pa]."""
    gm  = GAMMA_C
    R_c = 8314.0 / 29.5
    Gamma_fn = np.sqrt(gm) * (2.0/(gm+1))**((gm+1)/(2*(gm-1)))
    mdot_est = mdot_air * 1.05
    Pc = mdot_est * np.sqrt(R_c * Tt4) / (A_THROAT * Gamma_fn)
    return float(np.clip(Pc, 0.1e6, 6.0e6))

def sfrj_fuel_regression(Pc, A_burn):
    """Saint-Robert law: r_dot [m/s] = a_reg * Pc^n_reg."""
    r_dot  = A_REG * (Pc ** N_REG)
    mdot_f = RHO_FUEL * r_dot * A_burn
    return mdot_f, r_dot

def sfrj_thrust_and_isp(M, h, A_burn, verbose=False):
    """
    Full SFRJ thermodynamic cycle.
    Returns (T_net [N], Isp_fuel [s], mdot_f [kg/s], Pc [Pa]).
    """
    M = float(M)
    rho0, P0, T0, a0 = atmosphere(h)
    V0 = M * a0

    if M < 2.0:
        return 0.0, 0.0, 0.0, 0.0

    gm  = GAMMA
    Tt2 = T0  * (1.0 + (gm - 1) / 2.0 * M**2)
    Pt2 = P0  * (1.0 + (gm - 1) / 2.0 * M**2) ** (gm / (gm - 1))
    eta_r = sfrj_inlet_recovery(M)
    Pt3   = eta_r * Pt2
    mdot_air = rho0 * V0 * A_INLET
    FAR = FAR_DESIGN * np.clip((M - 1.8) / 0.8, 0.0, 1.0)
    Q_release = ETA_COMB * FAR * H_FUEL
    Tt4 = Tt2 + Q_release / (CP_C * (1.0 + FAR))
    Tt4 = float(np.clip(Tt4, Tt2, 3500.0))
    gm_c = GAMMA_C
    R_c  = 8314.0 / 29.5
    Gfn  = np.sqrt(gm_c) * (2.0 / (gm_c + 1)) ** ((gm_c + 1) / (2 * (gm_c - 1)))
    mdot_f_est = FAR * mdot_air
    mdot_total = mdot_air + mdot_f_est
    Pc = mdot_total * np.sqrt(R_c * Tt4) / (A_THROAT * Gfn)
    Pc = float(np.clip(Pc, 0.05e6, 5.0e6))
    mdot_f, r_dot = sfrj_fuel_regression(Pc, A_burn)
    FAR_actual = mdot_f / mdot_air if mdot_air > 1e-6 else FAR
    Q_act  = ETA_COMB * FAR_actual * H_FUEL
    Tt4    = float(np.clip(Tt2 + Q_act / (CP_C * (1.0 + FAR_actual)), Tt2, 3500.0))
    mdot_total = mdot_air + mdot_f
    Pc = float(np.clip(
        mdot_total * np.sqrt(R_c * Tt4) / (A_THROAT * Gfn), 0.05e6, 5.0e6))

    def ar_eq(Me):
        return ((2 / (gm_c + 1)) * (1 + (gm_c - 1) / 2 * Me**2)) \
               ** ((gm_c + 1) / (2 * (gm_c - 1))) / Me - AREA_RATIO
    Me = 3.0
    for _ in range(20):
        f  = ar_eq(Me)
        df = (ar_eq(Me + 0.001) - ar_eq(Me - 0.001)) / 0.002
        Me = max(Me - f / (df + 1e-12), 1.01)

    Te  = Tt4 / (1.0 + (gm_c - 1) / 2.0 * Me**2)
    Ve  = Me * np.sqrt(gm_c * R_c * Te)
    Ae  = A_THROAT * AREA_RATIO
    Pe  = Pc * (1.0 + (gm_c - 1) / 2.0 * Me**2) ** (-gm_c / (gm_c - 1))
    T_gross = mdot_total * Ve + (Pe - P0) * Ae
    D_ram   = mdot_air * V0
    T_net   = T_gross - D_ram
    Isp_fuel = T_net / (mdot_f * G0) if mdot_f > 1e-6 else 0.0

    if verbose:
        print(f"  M={M:.2f} h={h/1000:.1f}km  "
              f"mdot_air={mdot_air:.2f} kg/s  mdot_f={mdot_f:.3f} kg/s  "
              f"FAR={FAR_actual:.4f}  Tt4={Tt4:.0f} K  Pc={Pc/1e6:.2f} MPa  "
              f"T_net={T_net:.0f} N  Isp={Isp_fuel:.0f} s")

    return (float(np.clip(T_net, -5000.0, 80_000.0)),
            float(np.clip(Isp_fuel, 0.0, 2000.0)),
            float(mdot_f),
            float(Pc))

# ──────────────────────────────────────────────────────────────────────────────
# PROPULSION — LIQUID FUEL RAMJET (LFRJ)
# Wraps the dual-mode ram/scramjet cycle solver in combined_cycle_liquid_ram_scram
# ──────────────────────────────────────────────────────────────────────────────
def lfrj_performance(
        M: float,
        h: float,
        phi: float,
        ramp_angles: list | None = None,
        verbose: bool = False,
) -> tuple[float, float, float, float]:
    """
    Liquid Fuel Ramjet performance via combined-cycle analyze().

    Parameters
    ----------
    M    : float   Freestream Mach number
    h    : float   Altitude [m]
    phi  : float   Equivalence ratio (fuel-air / stoichiometric)
    ramp_angles : list | None   Inlet ramp deflections [deg]; None → config default
    verbose : bool

    Returns
    -------
    T_net  : float  Net thrust [N]     (ram drag already subtracted inside analyze)
    Isp    : float  Fuel-based specific impulse [s]
    mdot_f : float  Fuel mass-flow rate [kg/s]
    Pc     : float  Combustor static pressure at station 3 [Pa]

    Notes
    -----
    Bug fix vs original: original code multiplied thrust by 1000
    (result['thrust'] is already in N from analyze(); *1000 was wrong).

    Mach envelope gate: returns zero thrust for M < LFRJ_MACH_MIN or
    M > LFRJ_MACH_MAX (2.0–6.0).  This prevents analyze() from being called
    outside the valid ram/scramjet operating envelope, which previously caused
    the trajectory to accelerate unboundedly past Mach 6 and produced an
    incorrect Mach-vs-time plot.
    """
    M = float(M)

    # ── Hard operational envelope — zero thrust outside these bounds ──────────
    if M < LFRJ_MACH_MIN or M > LFRJ_MACH_MAX:
        if verbose:
            print(f"  [lfrj_performance] M={M:.2f} outside LFRJ envelope "
                  f"[{LFRJ_MACH_MIN}, {LFRJ_MACH_MAX}] — returning zero thrust")
        return 0.0, 0.0, 0.0, 0.0

    try:
        result = analyze(M0=M, altitude=h, phi=phi,
                         ramp_angles=ramp_angles, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"  [lfrj_performance] analyze() failed at M={M:.2f} h={h/1e3:.1f}km: {exc}")
        return 0.0, 0.0, 0.0, 0.0

    # thrust and mdot_fuel from analyze() are sized to config.A_CAPTURE = 0.05 m².
    # Scale both to the missile's actual inlet area via LFRJ_AREA_SCALE = A_INLET / A_CAPTURE.
    # Isp is an intensive quantity (thrust per unit fuel flow) so it does NOT scale.
    T_net  = float(result['thrust'])   * LFRJ_AREA_SCALE
    Isp    = float(result['Isp'])
    mdot_f = float(result['mdot_fuel']) * LFRJ_AREA_SCALE

    Pt3 = float(result['Pt_stations'][3])
    M3  = float(result['M_stations'][3])
    Pc  = Pt3 / (1.0 + (GAMMA - 1.0) / 2.0 * M3**2) ** (GAMMA / (GAMMA - 1.0))

    return (float(np.clip(T_net, -5_000.0, 200_000.0)),
            float(np.clip(Isp,    0.0,       4_000.0)),
            float(mdot_f),
            float(Pc))


def lfrj_pressure_recovery(M: float, altitude: float,
                            gamma: float = 1.4, R: float = 287.0) -> float:
    """
    Inlet total-pressure recovery Pt2/Pt0 computed via the combined-cycle
    inlet model at (M, altitude).
    """
    T, P, rho = freestream(altitude)
    a  = (gamma * R * T) ** 0.5
    Tt = T  * (1 + (gamma - 1) / 2 * M**2)
    Pt = P  * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))
    state = FlowState(M=M, T=T, P=P, Pt=Pt, Tt=Tt, gamma=gamma, R=R)
    try:
        recovery_ratio = compute_inlet(state, verbose=False)[1]
    except Exception:
        recovery_ratio = 0.0
    return float(recovery_ratio)

# ──────────────────────────────────────────────────────────────────────────────
# EQUATIONS OF MOTION — LFRJ VERSION
# State: [x_range (m), h (m), V (m/s), gamma (rad), m (kg), A_burn (m²)]
# A_burn is carried for structural compatibility but unused in LFRJ cruise.
# ──────────────────────────────────────────────────────────────────────────────
def eom_lfrj(state, alpha_deg, phase):
    """
    Equations of motion using LFRJ propulsion during the cruise phase.
    Boost and descent phases are identical to the SFRJ trajectory.
    """
    x_r, h, V, gamma, m, A_burn = state
    h      = float(np.clip(h,     50.0,  79_900.0))
    V      = float(np.clip(V,     20.0,   5_000.0))
    m      = float(np.clip(m,     M_STRUCT * 0.95, M0 + 10))
    rho, _, _, a = atmosphere(h)
    M_num  = V / a
    CL, CD, L, D, q = aero(M_num, alpha_deg, rho, V)
    g      = G0 * (RE / (RE + h))**2
    alpha_r = np.deg2rad(alpha_deg)

    if phase == 'boost':
        T    = boost_thrust(M_num, h)
        isp  = boost_isp()
        mdot = -T / (isp * G0)
        dA   = 0.0

    elif phase == 'cruise':
        T_net, isp, mdot_f, Pc = lfrj_performance(M_num, h, PHI_LFRJ)
        T    = T_net
        mdot = -mdot_f
        dA   = 0.0   # no solid grain — A_burn does not evolve for LFRJ

    else:   # descent — engine off
        T    = 0.0
        mdot = 0.0
        dA   = 0.0

    dx     = V * np.cos(gamma)
    dh     = V * np.sin(gamma)
    dV     = (T * np.cos(alpha_r) - D) / m - g * np.sin(gamma)
    dgamma = ((T * np.sin(alpha_r) + L) / (m * V)
              - (g / V - V / (RE + h)) * np.cos(gamma))

    return np.array([dx, dh, dV, dgamma, mdot, dA])

# ──────────────────────────────────────────────────────────────────────────────
# RK4 INTEGRATOR — LFRJ VERSION
# ──────────────────────────────────────────────────────────────────────────────
def simulate_phase_lfrj(state0, phase, t_end, dt, ctrl_fn,
                         m_prop_budget, stop_fn=None):
    """
    Simulate one flight phase using LFRJ propulsion during 'cruise'.
    state0 must have 6 elements: [x_range, h, V, gamma, m, A_burn].
    A_burn is not evolved during LFRJ cruise (passed through unchanged).
    """
    t   = 0.0
    s   = np.array(state0, dtype=float)
    m_p = float(m_prop_budget)

    keys = ['t', 'x_range', 'h', 'V', 'gamma', 'm', 'M', 'q',
            'alpha', 'L', 'D', 'T', 'isp', 'ax', 'CL', 'CD', 'Pc', 'A_burn']
    traj = {k: [] for k in keys}

    def snap(t, s):
        x_r, h, V, gamma, m, A_b = s
        h   = max(h, 10.0)
        V   = max(V, 10.0)
        rho, _, _, a = atmosphere(h)
        M_n = V / a
        al  = ctrl_fn(t, s)
        CL, CD, L, D, q = aero(M_n, al, rho, V)

        if phase == 'boost':
            Tp  = boost_thrust(M_n, h)
            isp = boost_isp()
            Pc  = 8.0e6
        elif phase == 'cruise':
            Tp, isp, _, Pc = lfrj_performance(M_n, h, PHI_LFRJ)
        else:
            Tp = 0.0; isp = 0.0; Pc = 0.0

        g   = G0 * (RE / (RE + h))**2
        axg = (Tp * np.cos(np.deg2rad(al)) - D) / m / G0
        for k, v in zip(keys,
            [t, x_r, h, V, gamma, m, M_n, q, al, L, D, Tp, isp, axg, CL, CD, Pc, A_b]):
            traj[k].append(float(v))

    snap(t, s)
    while t < t_end:
        if stop_fn and stop_fn(s):
            break
        al = ctrl_fn(t, s)
        k1 = eom_lfrj(s,            al, phase)
        k2 = eom_lfrj(s + dt/2*k1,  al, phase)
        k3 = eom_lfrj(s + dt/2*k2,  al, phase)
        k4 = eom_lfrj(s + dt*k3,    al, phase)
        ds = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        if phase in ('boost', 'cruise'):
            m_p -= abs(ds[4])
            if m_p <= 0.0:
                break
        s_new    = s + ds
        s_new[4] = max(s_new[4], M_STRUCT * 0.95)
        s_new[2] = max(s_new[2], 20.0)
        s_new[5] = np.clip(s_new[5], A_BURN_MIN, A_BURN_0 * 1.05)
        t += dt
        s  = s_new
        snap(t, s)

    return {k: np.array(v) for k, v in traj.items()}

# ──────────────────────────────────────────────────────────────────────────────
# ALPHA SCHEDULES
# ──────────────────────────────────────────────────────────────────────────────
CRUISE_ALT = 12_000.0       # m  — SFRJ reference (retained for SFRJ Dymos ODE)

def alpha_boost(t, s):
    """Climb to LFRJ cruise altitude; maximise acceleration."""
    _, h, V, gamma, m, A_b = s
    rho, _, _, a = atmosphere(max(h, 10.0))
    M = V / a
    h_err = LFRJ_CRUISE_ALT - h
    base  = np.clip(5.0 - 0.9*(M - 1.0), 1.5, 12.0)
    corr  = np.clip(0.003 * h_err, -4.0, 5.0)
    # Clip to the aero database alpha envelope [−8, 10] — extrapolation beyond 10 deg gives unphysical L/D
    return float(np.clip(base + corr, -5.0, 10.0))

def alpha_cruise(t, s):
    """Lift = Weight trim AoA + gentle altitude hold at LFRJ_CRUISE_ALT."""
    _, h, V, gamma, m, A_b = s
    h   = max(h, 100.0)
    rho, _, _, a = atmosphere(h)
    M   = V / a
    W   = m * G0 * (RE / (RE + h))**2
    al_trim = solve_alpha_trim(M, rho, V, target_L=W)
    h_err   = LFRJ_CRUISE_ALT - h
    # Gain reduced from 0.004 to 0.0008: 0.004 caused phugoid oscillations growing
    # to ±7km amplitude because the correction saturated at 5deg and overshot badly.
    # 0.0008 gives a max correction of ±0.8deg for a 1km error — stable damping.
    al_corr = np.clip(0.0008 * h_err, -2.0, 2.0)
    # Clip to the aero database alpha envelope — surrogate extrapolates badly above 10 deg
    return float(np.clip(al_trim + al_corr, 0.0, 10.0))

def alpha_descent(t, s):
    """Pull nose over to achieve ≥80° FPA at impact."""
    _, h, V, gamma, m, A_b = s
    gam_tgt = np.deg2rad(-85.0)
    err     = gam_tgt - gamma
    return float(np.clip(2.5 * np.rad2deg(err), -30.0, 5.0))

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LFRJ SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
def run_trajectory_lfrj():
    print("=" * 65)
    print("  HCM LFRJ Trajectory Simulation")
    print("  Air-Launch: F-35 @ Mach 0.8, 35,000 ft")
    print("  Cruise propulsion: Liquid Fuel Ramjet (JP-10, dual-mode)")
    print("=" * 65)

    print("\nInitializing aero models...")
    init_aero_models(
        prefix="ogive",
        geom_params={"x1": 4, "x2": 2},
        surrogate="linear"
    )

    h0_ft  = 35_000.0
    h0_m   = h0_ft * 0.3048
    _, _, _, a0 = atmosphere(h0_m)
    V0     = 0.8 * a0
    gamma0 = np.deg2rad(2.0)
    state0 = [0.0, h0_m, V0, gamma0, M0, A_BURN_0]

    print(f"\nLaunch : {h0_ft:,.0f} ft | Mach 0.8 | V={V0:.1f} m/s | m={M0:.0f} kg")
    print(f"Mass budget: Boost propellant {M_PROP_BOOST:.0f} kg | "
          f"LFRJ fuel (JP-10) {M_FUEL_LFRJ:.0f} kg | Structure {M_STRUCT:.0f} kg")

    # ── PHASE 1: SOLID ROCKET BOOST ───────────────────────────────────────────
    print("\n[Phase 1] Solid Rocket Boost ...")
    print(f"  Target: accelerate to M2.5+ for LFRJ light-off")
    tb = simulate_phase_lfrj(
        state0, 'boost', t_end=90, dt=0.5,
        ctrl_fn=alpha_boost,
        m_prop_budget=M_PROP_BOOST,
        stop_fn=lambda s: s[4] <= M0 - M_PROP_BOOST + 0.5
    )
    sf_b = [tb[k][-1] for k in ['x_range', 'h', 'V', 'gamma', 'm', 'A_burn']]
    _, _, _, a_b = atmosphere(sf_b[1])
    M_boost = sf_b[2] / a_b
    print(f"  Burnout: M={M_boost:.2f} | h={sf_b[1]/0.3048:,.0f} ft | "
          f"range={sf_b[0]/1000:.1f} km | m={sf_b[4]:.0f} kg")
    print(f"  Peak thrust ≈ {np.max(tb['T'])/1000:.1f} kN")

    print(f"\n  LFRJ light-off check at M={M_boost:.2f}, h={sf_b[1]/1000:.1f} km:")
    T_lo, isp_lo, mdot_lo, Pc_lo = lfrj_performance(M_boost, sf_b[1], PHI_LFRJ, verbose=True)
    print(f"    T_net={T_lo:.0f} N  Isp={isp_lo:.0f} s  "
          f"mdot_f={mdot_lo:.3f} kg/s  Pc={Pc_lo/1e6:.2f} MPa")

    # ── PHASE 2: LFRJ CRUISE ──────────────────────────────────────────────────
    print(f"\n[Phase 2] Liquid Fuel Ramjet (LFRJ) Cruise ...")
    print(f"  Fuel: JP-10 liquid  |  φ = {PHI_LFRJ:.2f}  |  "
          f"Target altitude: {LFRJ_CRUISE_ALT/1000:.0f} km")
    sf_b[3] = np.deg2rad(1.0)
    t_c0 = tb['t'][-1]
    tc = simulate_phase_lfrj(
        sf_b, 'cruise', t_end=t_c0 + 900, dt=1.0,
        ctrl_fn=alpha_cruise,
        m_prop_budget=M_FUEL_LFRJ,
        stop_fn=lambda s: s[4] <= M_STRUCT + 2.0
    )
    tc['t'] += t_c0
    sf_c = [tc[k][-1] for k in ['x_range', 'h', 'V', 'gamma', 'm', 'A_burn']]

    M_cruise   = float(np.mean(tc['M']))
    q_cruise   = float(np.mean(tc['q']))
    Pc_cruise  = float(np.mean(tc['Pc']))
    isp_valid  = tc['isp'][tc['isp'] > 0]
    isp_cruise = float(np.mean(isp_valid)) if len(isp_valid) > 0 else 0.0

    print(f"  Mean Mach       : {M_cruise:.2f}")
    print(f"  Mean altitude   : {np.mean(tc['h'])/0.3048:,.0f} ft")
    print(f"  Mean q          : {q_cruise/1000:.1f} kPa  ({q_cruise*0.020885:.0f} psf)")
    print(f"  Mean Pc         : {Pc_cruise/1e6:.2f} MPa")
    print(f"  Mean Isp (fuel) : {isp_cruise:.0f} s")
    print(f"  Fuel consumed   : {sf_b[4] - sf_c[4]:.1f} kg / {M_FUEL_LFRJ:.0f} kg")
    print(f"  End range       : {sf_c[0]/1000:.1f} km")

    # ── PHASE 3: TERMINAL DESCENT ─────────────────────────────────────────────
    print("\n[Phase 3] Unpowered Terminal Descent ...")
    sf_c[3] = np.deg2rad(-40.0)
    t_d0 = tc['t'][-1]
    td = simulate_phase_lfrj(
        sf_c, 'descent', t_end=t_d0 + 150, dt=0.25,
        ctrl_fn=alpha_descent,
        m_prop_budget=0.0,
        stop_fn=lambda s: s[1] <= 10.0
    )
    td['t'] += t_d0
    sf_d = [td[k][-1] for k in ['x_range', 'h', 'V', 'gamma', 'm', 'A_burn']]
    _, _, _, a_i = atmosphere(max(sf_d[1], 10.0))
    M_impact   = sf_d[2] / a_i
    fpa_impact = abs(np.rad2deg(sf_d[3]))
    total_range = sf_d[0] / 1000.0
    print(f"  Impact Mach : {M_impact:.2f}")
    print(f"  Impact FPA  : {fpa_impact:.1f}° (constraint ≥80°)")
    print(f"  Total range : {total_range:.1f} km")

    all_q = np.concatenate([tb['q'], tc['q'], td['q']])
    max_q = float(np.max(all_q))

    print("\n─────────────────────────────────────────")
    print("CONSTRAINT CHECK:")
    ok_cruise = M_cruise >= 4.0
    ok_Mi     = M_impact >= 2.0
    ok_fpa    = fpa_impact >= 80.0
    print(f"  Cruise Mach ≥ 4.0 : {M_cruise:.2f}  {'✓' if ok_cruise else '✗ VIOLATION'}")
    print(f"  Impact Mach ≥ 2.0 : {M_impact:.2f}  {'✓' if ok_Mi     else '✗ VIOLATION'}")
    print(f"  Impact FPA  ≥ 80° : {fpa_impact:.1f}°  {'✓' if ok_fpa    else '✗ VIOLATION'}")

    feats = dict(
        M_boost=M_boost, M_cruise=M_cruise,
        q_cruise=q_cruise, max_q=max_q,
        Pc_cruise=Pc_cruise,
        M_impact=M_impact, fpa_impact=fpa_impact,
        total_range=total_range,
    )

    print("\n══════════ SYSTEM FEATURES (LFRJ) ══════════")
    print(f"  Boost Mach Number     : {M_boost:.2f}")
    print(f"  Cruise Mach Number    : {M_cruise:.2f}")
    print(f"  Cruise Dyn. Pressure  : {q_cruise/1000:.2f} kPa  ({q_cruise*0.020885:.1f} psf)")
    print(f"  Max  Dyn. Pressure    : {max_q/1000:.2f} kPa  ({max_q*0.020885:.1f} psf)")
    print(f"  LFRJ Chamber Pressure : {Pc_cruise/1e6:.2f} MPa  (cruise mean)")
    print(f"  Fuel used / budget    : {M_FUEL_LFRJ:.0f} kg JP-10")
    print("═════════════════════════════════════════════\n")

    return tb, tc, td, feats

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — LFRJ  (one figure per plot, saved individually)
# ──────────────────────────────────────────────────────────────────────────────
PC = {'boost': '#e74c3c', 'cruise': '#2980b9', 'descent': '#27ae60'}

_SUBTITLE = (
    "HCM LFRJ  |  Solid Rocket → JP-10 Liquid Fuel Ramjet  "
    "|  F-35 Air-Launch @ M0.8 / 35,000 ft"
)


def _new_fig():
    """Create a single-axes figure with the dark theme."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.grid(color='#21262d', lw=0.7, ls='--')
    return fig, ax


def _save_fig(fig, output_dir, filename):
    """Save figure to output_dir/filename and close it."""
    out = output_dir / filename
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"    ✓  {filename}")
    return str(out)


def do_plots_lfrj(tb, tc, td, feats):
    """
    Save each trajectory plot as a separate PNG in plots_lfrj/.
    """
    output_dir = Path(__file__).parent / "plots_lfrj"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving individual plots → {output_dir}/")

    lkw   = dict(color='#c9d1d9', fontsize=10)
    tkw   = dict(color='#f0f6fc', fontsize=12, fontweight='bold', pad=10)
    legkw = dict(fontsize=8, facecolor='#21262d', labelcolor='#c9d1d9', edgecolor='none')
    phases = [('boost', tb), ('cruise', tc), ('descent', td)]

    def triplot(ax, xkey, ykey, xscale=1, yscale=1):
        for name, tr in phases:
            ax.plot(tr[xkey] * xscale, tr[ykey] * yscale,
                    color=PC[name], lw=2.0, label=name.capitalize())

    def tag(fig, title, feats):
        fig.suptitle(
            f"{title}\n"
            + _SUBTITLE
            + f"   |   Cruise M {feats['M_cruise']:.2f}   "
            + f"Range {feats['total_range']:.0f} km",
            color='#f0f6fc', fontsize=12, y=1.01,
        )

    saved = []

    # ── 01  Range vs Time ────────────────────────────────────────────────────
    fig, ax = _new_fig()
    triplot(ax, 't', 'x_range', yscale=1e-3)
    ax.set_xlabel('Time [s]', **lkw)
    ax.set_ylabel('Range [km]', **lkw)
    ax.set_title('01 — Range vs Time', **tkw)
    ax.legend(**legkw)
    tag(fig, '01 — Range vs Time', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '01_range_vs_time.png'))

    # ── 02  Mach vs Time ─────────────────────────────────────────────────────
    fig, ax = _new_fig()
    # Plot each phase in its own colour for visual clarity
    for name, tr in phases:
        ax.plot(tr['t'], tr['M'], color=PC[name], lw=2.0, label=name.capitalize())
    # Thin connectors at phase junctions to close the gap between separate line segments
    ax.plot([tb['t'][-1], tc['t'][0]], [tb['M'][-1], tc['M'][0]],
            color='#8b949e', lw=1.0, ls='--')
    ax.plot([tc['t'][-1], td['t'][0]], [tc['M'][-1], td['M'][0]],
            color='#8b949e', lw=1.0, ls='--')
    # Reference lines
    ax.axhline(LFRJ_MACH_MAX, color='#e74c3c', lw=1.4, ls='-.',
               label=f'M={LFRJ_MACH_MAX:.0f} (LFRJ limit)')
    ax.axhline(4.0, color='#f39c12', lw=1.2, ls=':', label='M=4 (min cruise)')
    ax.axhline(2.0, color='#8b949e', lw=1.0, ls=':', label='M=2 (min impact)')
    # Cap y-axis so a spurious high-thrust spike cannot compress the rest of the plot
    ax.set_ylim(0, LFRJ_MACH_MAX + 1.0)
    ax.set_xlabel('Time [s]', **lkw)
    ax.set_ylabel('Mach Number', **lkw)
    ax.set_title('02 — Mach Number vs Time', **tkw)
    ax.legend(**legkw)
    tag(fig, '02 — Mach Number vs Time', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '02_mach_vs_time.png'))

    # ── 03  Altitude vs Range ────────────────────────────────────────────────
    fig, ax = _new_fig()
    triplot(ax, 'x_range', 'h', xscale=1e-3, yscale=1e-3)
    ax.axhline(LFRJ_CRUISE_ALT / 1e3, color='#f39c12', lw=1.2, ls=':',
               label=f"LFRJ cruise alt {LFRJ_CRUISE_ALT/1e3:.0f} km")
    ax.set_xlabel('Range [km]', **lkw)
    ax.set_ylabel('Altitude [km]', **lkw)
    ax.set_title('03 — Altitude vs Range', **tkw)
    ax.legend(**legkw)
    tag(fig, '03 — Altitude vs Range', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '03_altitude_vs_range.png'))

    # ── 04  Dynamic Pressure vs Range ────────────────────────────────────────
    fig, ax = _new_fig()
    triplot(ax, 'x_range', 'q', xscale=1e-3, yscale=1e-3)
    ax.axhline(feats['max_q'] / 1e3, color='#e74c3c', lw=1.2, ls=':',
               label=f"Max q = {feats['max_q']/1e3:.1f} kPa")
    ax.set_xlabel('Range [km]', **lkw)
    ax.set_ylabel('Dynamic Pressure [kPa]', **lkw)
    ax.set_title('04 — Dynamic Pressure vs Range', **tkw)
    ax.legend(**legkw)
    tag(fig, '04 — Dynamic Pressure vs Range', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '04_dynamic_pressure_vs_range.png'))

    # ── 05  Axial Acceleration vs Range ─────────────────────────────────────
    fig, ax = _new_fig()
    triplot(ax, 'x_range', 'ax', xscale=1e-3)
    ax.axhline(0.0, color='#8b949e', lw=0.8, ls='--')
    ax.set_xlabel('Range [km]', **lkw)
    ax.set_ylabel('Axial Acceleration [g]', **lkw)
    ax.set_title('05 — Axial Acceleration vs Range', **tkw)
    ax.legend(**legkw)
    tag(fig, '05 — Axial Acceleration vs Range', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '05_acceleration_vs_range.png'))

    # ── 06  Weight vs Range ──────────────────────────────────────────────────
    fig, ax = _new_fig()
    for name, tr in phases:
        ax.plot(tr['x_range'] * 1e-3, tr['m'] * G0 / 1e3,
                color=PC[name], lw=2.0, label=name.capitalize())
    ax.set_xlabel('Range [km]', **lkw)
    ax.set_ylabel('Weight [kN]', **lkw)
    ax.set_title('06 — Vehicle Weight vs Range', **tkw)
    ax.legend(**legkw)
    tag(fig, '06 — Vehicle Weight vs Range', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '06_weight_vs_range.png'))

    # ── Pre-compute Mach-sweep data (shared by plots 07–10) ─────────────────
    h_ref = LFRJ_CRUISE_ALT

    # Determine the valid Mach range directly from the CFD database so that
    # plots 09 and 10 never extrapolate the surrogate outside its training domain.
    if MERGED_DF is not None and 'Mach' in MERGED_DF.columns:
        M_db_min = float(MERGED_DF['Mach'].min())
        M_db_max = float(MERGED_DF['Mach'].max())
        # Clamp to a sensible physical range
        M_db_min = max(M_db_min, 0.5)
        M_db_max = min(M_db_max, 10.0)
    else:
        # Fallback if database not available
        M_db_min, M_db_max = 0.5, 7.0

    # Sweep restricted to the LFRJ operational envelope [LFRJ_MACH_MIN, LFRJ_MACH_MAX].
    # Starting from M_db_min (0.8) causes the surrogate to extrapolate into the
    # transonic/subsonic regime where CD is negative and L/D reaches 20+.
    # These plots show LFRJ performance; boost-phase aero is irrelevant here.
    M_sw_fine = np.linspace(LFRJ_MACH_MIN, min(M_db_max, LFRJ_MACH_MAX), 250)
    M_sw_lfrj = np.linspace(LFRJ_MACH_MIN, min(M_db_max, LFRJ_MACH_MAX), 40)

    # Trim alpha: solved from the surrogate only within the valid Mach range.
    # Alpha bounds are taken from the database to avoid surrogate extrapolation.
    if MERGED_DF is not None and 'alpha' in MERGED_DF.columns:
        al_db_min = float(MERGED_DF['alpha'].min())
        al_db_max = float(MERGED_DF['alpha'].max())
    else:
        al_db_min, al_db_max = -10.0, 20.0

    def trim_alpha(M):
        rho, _, _, a = atmosphere(h_ref)
        V    = M * a
        m_cr = M_STRUCT + M_FUEL_LFRJ * 0.5
        W    = m_cr * G0
        # Clip to database alpha bounds AND the operational ceiling of 10 deg
        al_hi = min(al_db_max, 10.0)
        al = solve_alpha_trim(M, rho, V, target_L=W,
                              alpha_bounds=(max(al_db_min, 0.0), al_hi))
        return float(np.clip(al, 0.0, al_hi))

    print("    Computing LFRJ Mach sweep for analytical plots...")
    lfrj_T_sw   = []
    lfrj_isp_sw = []
    for M in M_sw_lfrj:
        Tp, isp_p, _, _ = lfrj_performance(M, h_ref, PHI_LFRJ)
        lfrj_T_sw.append(Tp)
        lfrj_isp_sw.append(isp_p)
    lfrj_T_sw   = np.array(lfrj_T_sw)
    lfrj_isp_sw = np.array(lfrj_isp_sw)

    td_b = []
    for M in M_sw_fine:
        rho, _, _, a = atmosphere(h_ref)
        V  = M * a
        al = trim_alpha(M)
        CL, CD, L, D, q = aero(M, al, rho, V)
        Tb = boost_thrust(M, h_ref)
        td_b.append(Tb / D if D > 0 else 0.0)

    td_l = []
    for i, M in enumerate(M_sw_lfrj):
        rho, _, _, a = atmosphere(h_ref)
        V  = M * a
        al = trim_alpha(M)
        CL, CD, L, D, q = aero(M, al, rho, V)
        td_l.append(lfrj_T_sw[i] / D if D > 0 else 0.0)

    # ── Trim AoA sweep — evaluated only within the database Mach envelope ──
    al_arr = [trim_alpha(M) for M in M_sw_fine]

    # ── L/D sweep — computed directly from L and D forces returned by aero()
    #    so the ratio is consistent with the force model rather than re-deriving
    #    CL/CD.  Also sweep over the alpha dimension at each Mach to find the
    #    maximum L/D the aerodatabase predicts (not just at one trim point).
    #    Two curves are produced:
    #      ld_trim : L/D at the trim alpha (operational L/D the vehicle flies at)
    #      ld_max  : maximum L/D over all database alpha values (aero envelope)
    if MERGED_DF is not None and 'alpha' in MERGED_DF.columns:
        alpha_sweep = np.unique(np.sort(MERGED_DF['alpha'].dropna().values))
    else:
        alpha_sweep = np.linspace(al_db_min, al_db_max, 20)

    ld_trim = []
    ld_max  = []
    for M, al_trim in zip(M_sw_fine, al_arr):
        rho, _, _, a = atmosphere(h_ref)
        V = M * a
        # L/D at trim alpha — use force ratio directly from aero()
        _, _, L_t, D_t, _ = aero(M, al_trim, rho, V)
        ld_trim.append(L_t / D_t if D_t > 0 else 0.0)
        # Maximum L/D over the database alpha range
        ld_vals = []
        for al_s in alpha_sweep:
            _, _, L_s, D_s, _ = aero(M, float(al_s), rho, V)
            if D_s > 0:
                ld_vals.append(L_s / D_s)
        ld_max.append(float(np.max(ld_vals)) if ld_vals else 0.0)

    # ── 07  T/D vs Mach ──────────────────────────────────────────────────────
    fig, ax = _new_fig()
    ax.plot(M_sw_fine, td_b, color=PC['boost'],  lw=2.0, label='Solid Rocket (Boost)')
    ax.plot(M_sw_lfrj, td_l, color=PC['cruise'], lw=2.0, label='LFRJ (JP-10, φ=0.8)')
    ax.axhline(1.0, color='#8b949e', lw=0.9, ls='--', label='T/D = 1')
    ax.axvline(2.5, color='#f39c12', lw=0.9, ls=':', label='LFRJ light-off M 2.5')
    valid_td = [v for v in td_b + td_l if np.isfinite(v)]
    if valid_td:
        ax.set_ylim(0, min(30, max(valid_td) * 1.15))
    ax.set_xlabel('Mach Number', **lkw)
    ax.set_ylabel('Thrust / Drag', **lkw)
    ax.set_title(f'07 — Thrust-to-Drag Ratio vs Mach  (h = {h_ref/1e3:.0f} km)', **tkw)
    ax.legend(**legkw)
    tag(fig, f'07 — T/D vs Mach  (h={h_ref/1e3:.0f} km)', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '07_thrust_drag_ratio_vs_mach.png'))

    # ── 08  Isp vs Mach ──────────────────────────────────────────────────────
    fig, ax = _new_fig()
    ax.plot(M_sw_fine, [boost_isp() for _ in M_sw_fine],
            color=PC['boost'], lw=2.0, ls='--', label='Solid Rocket (propellant Isp)')
    ax.plot(M_sw_lfrj, lfrj_isp_sw,
            color=PC['cruise'], lw=2.0, label='LFRJ (fuel-based Isp, JP-10)')
    ax.axvline(2.5, color='#f39c12', lw=0.9, ls=':', label='LFRJ light-off M 2.5')
    isp_ceil = max(3000, np.nanmax(lfrj_isp_sw) * 1.15) if len(lfrj_isp_sw) else 3000
    ax.set_ylim(0, isp_ceil)
    ax.set_xlabel('Mach Number', **lkw)
    ax.set_ylabel('Specific Impulse Isp [s]', **lkw)
    ax.set_title(f'08 — Specific Impulse vs Mach  (h = {h_ref/1e3:.0f} km)', **tkw)
    ax.legend(**legkw)
    tag(fig, f'08 — Isp vs Mach  (h={h_ref/1e3:.0f} km)', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '08_isp_vs_mach.png'))

    # ── 09  Trim AoA vs Mach ─────────────────────────────────────────────────
    fig, ax = _new_fig()
    ax.plot(M_sw_fine, al_arr, color='#9b59b6', lw=2.0, label='Trim AoA (L = W, mid-cruise mass)')
    ax.axhline(0.0, color='#8b949e', lw=0.7, ls='--')
    # Shade the region outside the database alpha envelope to flag extrapolation
    ax.axhspan(al_db_max, ax.get_ylim()[1] if ax.get_ylim()[1] > al_db_max else al_db_max + 2,
               alpha=0.12, color='#e74c3c', label=f'Outside DB alpha (>{al_db_max:.1f}°)')
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < al_db_min else al_db_min - 2, al_db_min,
               alpha=0.12, color='#e74c3c')
    ax.set_xlim(M_db_min, M_db_max)
    ax.set_xlabel('Mach Number', **lkw)
    ax.set_ylabel('Trim Angle of Attack [deg]', **lkw)
    ax.set_title(f'09 — Trim AoA vs Mach  (h = {h_ref/1e3:.0f} km, DB Mach range [{M_db_min:.1f}–{M_db_max:.1f}])', **tkw)
    ax.legend(**legkw)
    tag(fig, f'09 — Trim AoA vs Mach  (h={h_ref/1e3:.0f} km)', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '09_trim_aoa_vs_mach.png'))

    # ── 10  L/D vs Mach ──────────────────────────────────────────────────────
    # Two curves:
    #   ld_max  — maximum L/D over all database alpha values: represents the
    #             full aerodatabase L/D capability envelope.
    #   ld_trim — L/D at the trim alpha (operational, vehicle-weight-driven).
    # L/D is computed from the L and D force values returned by aero() so the
    # ratio is always consistent with the force model used in the trajectory.
    fig, ax = _new_fig()
    #ax.plot(M_sw_fine, ld_max,  color='#f39c12', lw=2.0, label='Max L/D (DB alpha sweep)')
    ax.plot(M_sw_fine, ld_trim, color='#9b59b6', lw=1.8, ls='--',
            label='L/D at trim α (mid-cruise mass)')
    ax.axhline(0.0, color='#8b949e', lw=0.7, ls='--')
    ax.set_xlim(M_db_min, M_db_max)
    valid_ld = [v for v in ld_max + ld_trim if np.isfinite(v) and v > -50]
    if valid_ld:
        ax.set_ylim(min(0, min(valid_ld)) - 0.2,
                    max(valid_ld)) #* 1.15)
    ax.set_xlabel('Mach Number', **lkw)
    ax.set_ylabel('Lift-to-Drag Ratio  L/D', **lkw)
    ax.set_title(
        f'10 — L/D vs Mach  (h = {h_ref/1e3:.0f} km, DB Mach range [{M_db_min:.1f}–{M_db_max:.1f}])',
        **tkw)
    ax.legend(**legkw)
    tag(fig, f'10 — L/D vs Mach  (h={h_ref/1e3:.0f} km)', feats)
    fig.tight_layout()
    saved.append(_save_fig(fig, output_dir, '10_lift_drag_ratio_vs_mach.png'))

    print(f"\n  ✓  {len(saved)} plots saved to {output_dir}/")
    return saved

# ──────────────────────────────────────────────────────────────────────────────
# DYMOS OPTIMAL CONTROL PROBLEM — LFRJ VERSION  (3-phase)
# Retains BoostODE and DescentODE unchanged; replaces CruiseODE with LFRJ.
# ──────────────────────────────────────────────────────────────────────────────
class BoostODE(om.ExplicitComponent):
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),
                     ('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),
                     ('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        h   = np.clip(inputs['h'],   500., 79_000.)
        V   = np.clip(inputs['V'],    50., 4_000.)
        gm  = inputs['gamma']
        m   = np.clip(inputs['m'], M_STRUCT, M0 + 10)
        al  = inputs['alpha']
        rho, _, _, a = atm_vec(h)
        M_n = V / a
        CL, CD, L, D, q = aero(M_n, al, rho, V)
        T   = np.array([boost_thrust(Mi, hi) for Mi, hi in zip(M_n, h)])
        isp = np.full_like(M_n, boost_isp())
        g   = G0 * (RE / (RE + h))**2
        ar  = np.deg2rad(al)
        outputs['x_range_dot'] = V * np.cos(gm)
        outputs['h_dot']       = V * np.sin(gm)
        outputs['V_dot']       = (T * np.cos(ar) - D) / m - g * np.sin(gm)
        outputs['gamma_dot']   = ((T * np.sin(ar) + L) / (m * V)
                                   - (g / V - V / (RE + h)) * np.cos(gm))
        outputs['m_dot']       = -T / (isp * G0)


class LFRJCruiseODE(om.ExplicitComponent):
    """Dymos ODE for LFRJ cruise phase. Uses lfrj_performance() per node."""
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),
                     ('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),
                     ('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        h   = np.clip(inputs['h'],  500., 79_000.)
        V   = np.clip(inputs['V'], 100.,  4_000.)
        gm  = inputs['gamma']
        m   = np.clip(inputs['m'], M_STRUCT, M0 + 10)
        al  = inputs['alpha']
        rho, _, _, a = atm_vec(h)
        M_n = V / a
        CL, CD, L, D, q = aero(M_n, al, rho, V)
        lfrj_res = [lfrj_performance(Mi, hi, PHI_LFRJ)
                    for Mi, hi in zip(M_n, h)]
        T      = np.array([r[0] for r in lfrj_res])
        isp    = np.array([r[1] if r[1] > 0 else 1.0 for r in lfrj_res])
        mdot_f = np.array([r[2] for r in lfrj_res])
        g   = G0 * (RE / (RE + h))**2
        ar  = np.deg2rad(al)
        outputs['x_range_dot'] = V * np.cos(gm)
        outputs['h_dot']       = V * np.sin(gm)
        outputs['V_dot']       = (T * np.cos(ar) - D) / m - g * np.sin(gm)
        outputs['gamma_dot']   = ((T * np.sin(ar) + L) / (m * V)
                                   - (g / V - V / (RE + h)) * np.cos(gm))
        outputs['m_dot']       = -mdot_f


class DescentODE(om.ExplicitComponent):
    def initialize(self): self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        for n, u in [('x_range','m'),('h','m'),('V','m/s'),
                     ('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        for n, u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),
                     ('gamma_dot','rad/s'),('m_dot','kg/s')]:
            self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        h   = np.clip(inputs['h'],   5., 79_000.)
        V   = np.clip(inputs['V'],  50.,  4_000.)
        gm  = inputs['gamma']
        m   = np.clip(inputs['m'], M_STRUCT * 0.9, M0 + 10)
        al  = inputs['alpha']
        rho, _, _, a = atm_vec(h)
        M_n = V / a
        CL, CD, L, D, q = aero(M_n, al, rho, V)
        g = G0 * (RE / (RE + h))**2
        ar = np.deg2rad(al)
        outputs['x_range_dot'] = V * np.cos(gm)
        outputs['h_dot']       = V * np.sin(gm)
        outputs['V_dot']       = -D / m - g * np.sin(gm)
        outputs['gamma_dot']   = (L / (m * V) - (g / V - V / (RE + h)) * np.cos(gm))
        outputs['m_dot']       = np.zeros(len(h))


def build_dymos_problem_lfrj(h0_m, V0):
    """3-phase Dymos optimal control problem using LFRJ cruise ODE."""
    p    = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    seg_b, seg_c, seg_d = 8, 15, 10

    boost = dm.Phase(ode_class=BoostODE,
                     transcription=dm.GaussLobatto(num_segments=seg_b, order=3))
    traj.add_phase('boost', boost)
    boost.set_time_options(fix_initial=True, initial_val=0.0,
                           duration_bounds=(20, 80), units='s')
    boost.add_state('x_range', fix_initial=True, rate_source='x_range_dot', lower=0, ref=1e5)
    boost.add_state('h',       fix_initial=True, rate_source='h_dot', lower=1000, upper=50000)
    boost.add_state('V',       fix_initial=True, rate_source='V_dot', lower=50, upper=5000)
    boost.add_state('gamma',   fix_initial=True, rate_source='gamma_dot', lower=-0.5, upper=1.5)
    boost.add_state('m',       fix_initial=True, rate_source='m_dot', lower=M_STRUCT, upper=M0)
    boost.add_control('alpha', lower=-10, upper=20, units='deg', ref=5.)
    boost.add_boundary_constraint('m', loc='final', lower=M0 - M_PROP_BOOST, units='kg')

    cruise = dm.Phase(ode_class=LFRJCruiseODE,
                      transcription=dm.GaussLobatto(num_segments=seg_c, order=3))
    traj.add_phase('cruise', cruise)
    cruise.set_time_options(fix_initial=False, duration_bounds=(200, 1200), units='s')
    # V upper bound derived from LFRJ_MACH_MAX × speed of sound at cruise altitude
    # so it stays consistent with LFRJ_MACH_MAX and LFRJ_CRUISE_ALT if either changes
    _, _, _, _a_cruise = atmosphere(LFRJ_CRUISE_ALT)
    V_cruise_max = LFRJ_MACH_MAX * _a_cruise
    for st, lb, ub in [('x_range', 0, 5e6), ('h', 5000, 40000),
                        ('V', 500, V_cruise_max), ('gamma', -0.5, 0.5), ('m', M_STRUCT, M0)]:
        cruise.add_state(st, fix_initial=False, rate_source=f'{st}_dot', lower=lb, upper=ub)
    cruise.add_control('alpha', lower=-5, upper=15, units='deg')
    # Lower V bound at cruise entry = LFRJ_MACH_MIN * a(LFRJ_CRUISE_ALT)
    # This ensures the missile reaches LFRJ ignition speed before cruise begins.
    _V_lfrj_ignition = LFRJ_MACH_MIN * _a_cruise
    cruise.add_boundary_constraint('V', loc='initial', lower=_V_lfrj_ignition, units='m/s')
    cruise.add_boundary_constraint('m', loc='final', lower=M_STRUCT, units='kg')

    descent = dm.Phase(ode_class=DescentODE,
                       transcription=dm.GaussLobatto(num_segments=seg_d, order=3))
    traj.add_phase('descent', descent)
    descent.set_time_options(fix_initial=False, duration_bounds=(20, 200), units='s')
    for st, lb, ub in [('x_range', 0, 5e6), ('h', 0, 50000),
                        ('V', 100, 5000), ('gamma', -np.pi, 0), ('m', M_STRUCT * 0.9, M0)]:
        descent.add_state(st, fix_initial=False, rate_source=f'{st}_dot', lower=lb, upper=ub)
    descent.add_control('alpha', lower=-30, upper=10, units='deg')
    descent.add_boundary_constraint('h',     loc='final', equals=0.,              units='m')
    descent.add_boundary_constraint('V',     loc='final', lower=680.,             units='m/s')
    descent.add_boundary_constraint('gamma', loc='final', upper=np.deg2rad(-80.), units='rad')

    traj.link_phases(['boost', 'cruise'],   vars=['time', 'x_range', 'h', 'V', 'gamma', 'm'])
    traj.link_phases(['cruise', 'descent'], vars=['time', 'x_range', 'h', 'V', 'gamma', 'm'])

    descent.add_objective('x_range', loc='final', scaler=-1e-5)

    p.driver = om.pyOptSparseDriver(optimizer='SLSQP')
    p.driver.opt_settings['ACC'] = 1e-5
    p.setup(force_alloc_complex=True)

    nn_b = seg_b*3 + 1
    nn_c = seg_c*3 + 1
    nn_d = seg_d*3 + 1
    p.set_val('traj.boost.t_duration', 55.0)
    p.set_val('traj.boost.states:h',     np.linspace(h0_m, LFRJ_CRUISE_ALT, nn_b))
    p.set_val('traj.boost.states:V',     np.linspace(V0, 1500, nn_b))
    p.set_val('traj.boost.states:gamma', np.linspace(0.05, 0.02, nn_b))
    p.set_val('traj.boost.states:m',     np.linspace(M0, M0 - M_PROP_BOOST, nn_b))
    p.set_val('traj.cruise.t_duration',  600.0)
    p.set_val('traj.cruise.states:h',    np.full(nn_c, LFRJ_CRUISE_ALT))
    p.set_val('traj.cruise.states:V',    np.full(nn_c, 1400.0))
    p.set_val('traj.cruise.states:gamma', np.zeros(nn_c))
    p.set_val('traj.cruise.states:m',    np.linspace(M0 - M_PROP_BOOST, M_STRUCT + 10, nn_c))
    p.set_val('traj.descent.t_duration', 60.0)
    p.set_val('traj.descent.states:h',   np.linspace(LFRJ_CRUISE_ALT, 0, nn_d))
    p.set_val('traj.descent.states:V',   np.linspace(1400, 900, nn_d))
    p.set_val('traj.descent.states:gamma', np.linspace(-0.1, -1.5, nn_d))
    return p

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Physics-based simulation (init_aero_models is called inside run_trajectory_lfrj)
    tb, tc, td, feats = run_trajectory_lfrj()
    plot_files = do_plots_lfrj(tb, tc, td, feats)

    if '--optimize' in sys.argv:
        print("\n[Dymos] Building 3-phase LFRJ optimal control problem ...")
        h0_m = 35_000 * 0.3048
        _, _, _, a0 = atmosphere(h0_m)
        try:
            prob = build_dymos_problem_lfrj(h0_m, 0.8 * a0)
            dm.run_problem(prob, run_driver=True, simulate=True)
            print("[Dymos] ✓ Optimization complete.")
        except Exception as e:
            print(f"[Dymos] Solver note: {e}")
    else:
        print("Tip: pass --optimize flag to run full Dymos/SLSQP optimal control solve.\n")

    print(f"\n✓ {len(plot_files)} individual plots saved:")
    for p in plot_files:
        print(f"   {p}")