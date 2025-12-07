import openmdao.api as om
import dymos as dm
import numpy as np
import math
import matplotlib.pyplot as plt


import os
import glob
import argparse
import pandas as pd
from pathlib import Path
import re
import operator

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler




###### FIRST, CBaero Parsing Functions ######

def parse_cfd_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Locate the function data section
    data_start = None
    for i, line in enumerate(lines):
        if "Function Data:" in line:
            data_start = i + 2
            break

    if data_start is None:
        raise RuntimeError(f"Cannot find 'Function Data:' section in {filepath}")

    # Parse the function data
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

def find_files(directory, model_name, param):
    totl_params = ["CDw", "CLw", "CFx", "CFy", "CFz", "CMl", "CMn", "CMm", "CMx", "CMy", "CMz", "CSw"]
    if param in totl_params:
        pattern = f"{model_name}.{param}.Totl.dat"
    else:
        pattern = f"{model_name}.{param}.dat"
    return glob.glob(os.path.join(directory, pattern))

def extract_geometry_from_dir(dirname, prefix="ogive"):
    """
    Extracts geometry parameters from directory name.
    e.g., waverider_4_2_3 -> {"x1": 4, "x2": 2, "x3": 3}
    If the directory is exactly the prefix (no underscore suffix), return {} (no geometry).
    """
    # If there's no underscore after the prefix, treat as no geometry info.
    if not dirname.startswith(prefix + "_"):
        return {}
    parts = dirname.replace(prefix + "_", "").split("_")
    # Only numeric parts are considered geometry; ignore empty parts / non-numeric gracefully
    geom = {}
    for i, p in enumerate(parts):
        try:
            geom[f"x{i+1}"] = int(p)
        except ValueError:
            # If it's not an int, try float; otherwise skip
            try:
                geom[f"x{i+1}"] = float(p)
            except ValueError:
                # skip non-numeric suffixes
                continue
    return geom

def collect_data_across_dirs(param, model_prefix="ogive"):
    all_data = []

    is_derived = param == "L/D"
    required_params = ["CLw", "CDw"] if is_derived else [param]

    for entry in os.scandir('.'):
        # Include directories that are either exactly model_prefix (no geometry) or start with model_prefix_
        if not entry.is_dir():
            continue
        if not (entry.name == model_prefix or entry.name.startswith(model_prefix + "_")):
            continue

        geom_params = extract_geometry_from_dir(entry.name, prefix=model_prefix)

        data_frames = {}
        for req_param in required_params:
            files = find_files(entry.path, model_prefix, req_param)
            if not files:
                print(f"Skipping {entry.name}: no file for {req_param}")
                break
            try:
                df = parse_cfd_file(files[0])
                # attach geometry columns if present
                for k, v in geom_params.items():
                    df[k] = v
                df["config_dir"] = entry.name
                data_frames[req_param] = df
            except RuntimeError as e:
                print(f"Error reading {files[0]}: {e}")
                break
        else:
            # All required parameters were successfully loaded
            if is_derived:
                df_cl = data_frames["CLw"]
                df_cd = data_frames["CDw"]
                # Merge on shared variables
                merged = pd.merge(
                        df_cl, df_cd,
                        on=["Mach", "q", "alpha", "beta"] + list(geom_params.keys()),
                    suffixes=("_CL", "_CD")
                )

                merged["F"] = merged["F_CL"] / merged["F_CD"]

                # Add config_dir manually (same for whole df)
                merged["config_dir"] = df_cl["config_dir"].iloc[0]

                all_data.append(merged[["Mach", "q", "alpha", "beta", "F"] + list(geom_params.keys()) + ["config_dir"]])
            else:
                all_data.append(data_frames[param])

    if not all_data:
        raise RuntimeError(f"No data found for parameter {param} in any matching directory for prefix '{model_prefix}'.")

    return pd.concat(all_data, ignore_index=True)

def verify_aero_data(merged_df):
    """
    Verify the loaded aerodynamic data for reasonableness.
    Checks for:
    - Expected L/D ratios
    - Reasonable CL and CD values
    - Data coverage
    """
    print("\n" + "="*60)
    print("AERODYNAMIC DATA VERIFICATION")
    print("="*60)
    
    if "CLw" in merged_df.columns and "CDw" in merged_df.columns:
        # Calculate L/D where CD is not zero
        valid_mask = (merged_df["CDw"] > 0.001) & (merged_df["CLw"].notna()) & (merged_df["CDw"].notna())
        if valid_mask.sum() > 0:
            ld_ratio = merged_df.loc[valid_mask, "CLw"] / merged_df.loc[valid_mask, "CDw"]
            print(f"\nL/D Ratio Statistics:")
            print(f"  Min:    {ld_ratio.min():.3f}")
            print(f"  Max:    {ld_ratio.max():.3f}")
            print(f"  Mean:   {ld_ratio.mean():.3f}")
            print(f"  Median: {ld_ratio.median():.3f}")
            
            # Check for suspiciously high L/D
            high_ld = ld_ratio > 4.5
            if high_ld.sum() > 0:
                print(f"\n  WARNING: {high_ld.sum()} points have L/D > 4.5")
                print(f"  Maximum L/D found: {ld_ratio.max():.3f}")
                # Show some examples
                high_ld_df = merged_df.loc[valid_mask].loc[ld_ratio > 4.5].head(10)
                print("\n  Sample high L/D points:")
                print(high_ld_df[["Mach", "alpha", "CLw", "CDw"]].to_string())
    
    # CL statistics
    if "CLw" in merged_df.columns:
        cl_valid = merged_df["CLw"].notna()
        print(f"\nCL Statistics ({cl_valid.sum()} valid points):")
        print(f"  Min:  {merged_df.loc[cl_valid, 'CLw'].min():.4f}")
        print(f"  Max:  {merged_df.loc[cl_valid, 'CLw'].max():.4f}")
        print(f"  Mean: {merged_df.loc[cl_valid, 'CLw'].mean():.4f}")
    
    # CD statistics
    if "CDw" in merged_df.columns:
        cd_valid = merged_df["CDw"].notna()
        print(f"\nCD Statistics ({cd_valid.sum()} valid points):")
        print(f"  Min:  {merged_df.loc[cd_valid, 'CDw'].min():.4f}")
        print(f"  Max:  {merged_df.loc[cd_valid, 'CDw'].max():.4f}")
        print(f"  Mean: {merged_df.loc[cd_valid, 'CDw'].mean():.4f}")
    
    # Data coverage
    print(f"\nData Coverage:")
    print(f"  Total points: {len(merged_df)}")
    if "Mach" in merged_df.columns:
        print(f"  Mach range: [{merged_df['Mach'].min():.2f}, {merged_df['Mach'].max():.2f}]")
    if "alpha" in merged_df.columns:
        print(f"  Alpha range: [{merged_df['alpha'].min():.2f}, {merged_df['alpha'].max():.2f}] deg")
    if "q" in merged_df.columns:
        print(f"  Dynamic pressure range: [{merged_df['q'].min():.2e}, {merged_df['q'].max():.2e}] bar")
    
    print("="*60 + "\n")

def build_global_database(model_prefix, surrogate_type="linear"):
    """
    Build a global merged DataFrame containing CLw, CDw, CMn, MaxQdotTotalQdotConvection
    across all model_prefix (either model_prefix dir or model_prefix_* dirs). Train GaussianProcessRegressor models for
    each of these outputs and return (global_df, models, scalers, feature_cols).

    Returns:
        global_df: pandas.DataFrame with columns including Mach,q,alpha,beta,x1...,CLw,CDw,CMn,MaxQdotTotalQdotConvection
        models: dict mapping output_name -> trained sklearn GaussianProcessRegressor
        scalers: dict with feature scaler (scaler for X) and target scalers if needed (not used here)
        feature_cols: list of feature column names used for training
    """
    # parameter names to collect (these must match file param names)
    target_params = ["CLw", "CDw", "CMn", "MaxQdotTotalQdotConvection"]
    collected = {}

    # Collect data for each target param
    for param in target_params:
        try:
            collected[param] = collect_data_across_dirs(param, model_prefix=model_prefix)
            # rename column 'F' -> param to avoid collisions
            collected[param] = collected[param].rename(columns={"F": param})
        except RuntimeError as e:
            print(f"Warning: could not collect {param}: {e}")
            collected[param] = None

    # Determine geom columns from any available dataset
    geom_cols = []
    for df in collected.values():
        if df is not None:
            # geometry columns are those starting with 'x' per your extract function
            geom_cols = [c for c in df.columns if re.match(r"x\d+", c)]
            break

    # Merge datasets on Mach,q,alpha,beta and geom columns
    keycols = ["Mach", "q", "alpha", "beta"] + geom_cols
    # Start from the first available df
    dfs_to_merge = []
    for param, df in collected.items():
        if df is not None:
            # select relevant cols (may include config_dir)
            cols_needed = keycols + [param]
            # Some dfs might not include all geom cols explicitly; ensure they exist
            for g in geom_cols:
                if g not in df.columns:
                    df[g] = np.nan
            dfs_to_merge.append(df[cols_needed].copy())

    if not dfs_to_merge:
        raise RuntimeError("No parameter data found to build global database.")

    # Perform sequential merges (outer join to keep all points)
    merged = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        merged = pd.merge(merged, df, on=keycols, how="outer")

    # Ensure numeric types
    for col in ["Mach", "q", "alpha", "beta"] + geom_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Optionally drop rows with missing core variables
    merged = merged.dropna(subset=["Mach", "q", "alpha", "beta"])

    # Fill geometry NaNs with median of each geom column (so model can use them)
    for g in geom_cols:
        if merged[g].isna().any():
            median = merged[g].median(skipna=True)
            merged[g] = merged[g].fillna(median)

    # VERIFY THE AERO DATA
    verify_aero_data(merged)

    # Save global database to CSV
    outcsv = f"global_database_{model_prefix}.csv"
    merged.to_csv(outcsv, index=False)
    print(f"Global database written to: {outcsv}")

    # === Prepare training data ===
    feature_cols = ["Mach", "q", "alpha", "beta"] + geom_cols
    X = merged[feature_cols].values

    # We'll train one GPR per target that exists (non-all-nan)
    models = {}
    scalers = {}

    # standardize features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    if surrogate_type == "gpr":
        scalers["X_scaler"] = X_scaler
    else:
        scalers["X_scaler"] = None  # No scaling needed


    # --- Train models based on chosen surrogate_type ---
    for target in target_params:
        if target not in merged.columns:
            continue

        y = merged[target].values
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            print(f"Skipping training for {target}: not enough valid samples ({mask.sum()})")
            continue

        X_train = X_scaled[mask] if surrogate_type == "gpr" else merged.loc[mask, feature_cols].values
        y_train = y[mask]

        if surrogate_type == "gpr":
            # Gaussian Process Regressor
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]),
                                                  length_scale_bounds=(1e-2, 1e3)) \
                     + WhiteKernel(noise_level=1e-6,
                                   noise_level_bounds=(1e-10, 1e1))
            model = GaussianProcessRegressor(kernel=kernel,
                                             normalize_y=True,
                                             n_restarts_optimizer=3,
                                             random_state=0)
            print(f"Training GPR for {target} on {X_train.shape[0]} samples")
            model.fit(X_train, y_train)

        elif surrogate_type == "linear":
            # Simple fast linear regression
            print(f"Training Linear Regression for {target} on {X_train.shape[0]} samples")
            model = LinearRegression()
            model.fit(X_train, y_train)

        models[target] = model

    return merged, models, scalers, feature_cols

def get_models_for_prefix(model_prefix):
    """
    Convenience function requested by the user.
    Takes a model_prefix (e.g., 'waverider') and returns a list of sklearn models:
      [Cd_model, Cl_model, Cm_model, qdot_model]
    If any of the models are missing (not trained), that list entry will be None.
    """
    merged, models, scalers, feature_cols = build_global_database(model_prefix)

    # Map dictionary keys to the requested ordering
    # CDw -> Cd, CLw -> Cl, CMn -> Cm, MaxQdotTotalQdotConvection -> qdot
    Cd_model = models.get("CDw")
    Cl_model = models.get("CLw")
    Cm_model = models.get("CMn")  # you're using CMn in the training targets
    qdot_model = models.get("MaxQdotTotalQdotConvection")

    # Inform user about missing items
    if Cd_model is None:
        print("Warning: CDw model not available (insufficient data or failed to train).")
    if Cl_model is None:
        print("Warning: CLw model not available (insufficient data or failed to train).")
    if Cm_model is None:
        print("Warning: CMn (Cm) model not available (insufficient data or failed to train).")
    if qdot_model is None:
        print("Warning: MaxQdotTotalQdotConvection (qdot) model not available (insufficient data or failed to train).")

    return [Cd_model, Cl_model, Cm_model, qdot_model], scalers, feature_cols, merged

def interpolate(models, scalers, feature_cols, query_df):
    """
    Given trained models and a dataframe of query points (with feature_cols present),
    return query_df with added predicted columns for each model key.

    Args:
        models: dict target_name -> trained sklearn regressor
        scalers: dict containing X_scaler used for features
        feature_cols: list of feature column names expected in query_df
        query_df: pandas.DataFrame containing at least feature_cols

    Returns:
        pandas.DataFrame: copy of query_df with added columns "<target>_pred" (and "<target>_std" if available)
    """
    out = query_df.copy()
    # Ensure numeric and fill missing geom columns with medians if needed
    for c in feature_cols:
        if c not in out.columns:
            # Fill with median from scalers? Here, attempt to use 0
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    Xq = out[feature_cols].values
    Xq_scaled = scalers["X_scaler"].transform(Xq)

    for target, model in models.items():
        # predict with std
        try:
            y_pred, y_std = model.predict(Xq_scaled, return_std=True)
        except TypeError:
            # Some sklearn versions might not support return_std; fall back
            y_pred = model.predict(Xq_scaled)
            y_std = np.full_like(y_pred, np.nan)
        out[f"{target}_pred"] = y_pred
        out[f"{target}_std"] = y_std

    return out


# Small helper to map user requested plotting param to trained model key
_param_synonyms = {
    "q_dot": "MaxQdotTotalQdotConvection",
    "qdot": "MaxQdotTotalQdotConvection",
    "Cm": "CMn",
    "Cl": "CLw",
    "Cd": "CDw",
    # exact keys also map to themselves implicitly
}

def _map_param_to_model_key(param):
    if param in _param_synonyms:
        return _param_synonyms[param]
    return param  # assume user passed the exact key like "CLw"


#### done with CBaero parsing and model building ####
# --- 1. Define the Physics Component (Equations of Motion) ---
class RocketBoostEOM(om.ExplicitComponent):
    def __init__(self, model_list, scalers, feature_cols, **kwargs):
        super().__init__(**kwargs)
        # Store models as instance attributes
        self.model_list = model_list
        self.scalers = scalers
        self.feature_cols = feature_cols

    """
    Computes the differential equations for a 2D atmospheric rocket boost / glide with lift and AOA as control variables
    """
    def initialize(self):
        """
        Define constant parameters for the environment
        """
        self.options.declare('num_nodes', types=int)
        self.R_E = 6378137.0  # Earth radius (m)
        self.g0 = 9.80665    # Gravity at sea level (m/s^2)
        
        # Atmospheric constants (Simplified ISA-ish Model used here)
        self.rho0 = 1.225     # Density at sea level (kg/m^3)
        self.H = 7500.0       # Scale height (m)
        self.T0 = 288.15      # Temperature at sea level (K) - used for speed of sound
        self.R_gas = 287.05   # Specific Gas Constant (J/(kg*K))
        self.gamma_atm = 1.4  # Ratio of specific heats 

    def setup(self):
        """
        Define Inputs and Outputs
        """
        nn = self.options['num_nodes']

        # Inputs (States, Controls, Parameters)
        self.add_input('h', val=np.zeros(nn), units='m', desc='Altitude')
        self.add_input('V', val=np.zeros(nn), units='m/s', desc='Velocity')
        self.add_input('gamma', val=np.zeros(nn), units='rad', desc='Flight path angle')
        self.add_input('m', val=np.zeros(nn), units='kg', desc='Vehicle Mass')
        
        # Control Input: Angle of Attack (alpha)
        self.add_input('alpha', val=np.zeros(nn), units='rad', desc='Angle of attack')
        
        # Parameters (vehicle/engine/shape characteristics)
        # For a pure glide/flight optimization set T = 0
        self.add_input('T', val=0.0, units='N', desc='Engine Thrust (set to 0 for glide)')
        self.add_input('Isp', val=250.0, units='s', desc='Specific Impulse')
        self.add_input('S', val=0.0124, units='m**2', desc='Reference Area')
        self.add_input('payload_mass', val=9.0, units='kg', desc='Payload mass')
        
        # Shape Parameters (simplified)
        self.add_input('CD_0', val=0.1, desc='Zero-lift drag coefficient')
        self.add_input('CL_alpha', val=0.1, desc='Lift coefficient slope (per rad)')

        # Outputs (State Derivatives and intermediate values)
        self.add_output('h_dot', val=np.zeros(nn), units='m/s', desc='dh/dt: Rate of change of altitude')
        self.add_output('V_dot', val=np.zeros(nn), units='m/s**2', desc='dV/dt: Acceleration')
        self.add_output('gamma_dot', val=np.zeros(nn), units='rad/s', desc='dgamma/dt: Rate of change of flight path angle')
        self.add_output('m_dot', val=np.zeros(nn), units='kg/s', desc='dm/dt: Mass flow rate')
        self.add_output('x_dot', val=np.zeros(nn), units='m/s', desc='dx/dt: Rate of change of horizontal range')
        
        self.add_output('Drag', val=np.zeros(nn), units='N', desc='Total Drag Force')
        self.add_output('Lift', val=np.zeros(nn), units='N', desc='Total Lift Force')
        self.add_output('Mach', val=np.zeros(nn), desc='Mach Number')
        self.add_output('g_load', val=np.zeros(nn), units=None, desc='Load factor (L / (m*g0))')
        self.add_output('q_dot', val=np.zeros(nn), units='W/m**2', desc='Stagnation Heat flux rate')
        self.add_output('q_dyn', val=np.zeros(nn), units='Pa', desc='Dynamic pressure')
        self.add_output('rho_atm', val=np.zeros(nn), units='kg/m**3', desc='Atmospheric density')
        self.add_output('CL', val=np.zeros(nn), desc='Lift coefficient')
        self.add_output('CD', val=np.zeros(nn), desc='Drag coefficient')

        # Partials setup (fd for simplicity)
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        """
        Executes physics calculations for every point in the trajectory
        """
        h = inputs['h']
        V = inputs['V']
        gamma = inputs['gamma']
        m = inputs['m']
        alpha = inputs['alpha']
        T = inputs['T']
        S = inputs['S'] 
        CD_0 = inputs['CD_0']
        CL_alpha = inputs['CL_alpha']
        Isp = inputs['Isp']
        g0 = self.g0
        payload = inputs['payload_mass']

        # --- Sub-Models ---

        # 1. Local Gravity & Radius
        r = self.R_E + h
        g = g0 * (self.R_E / r)**2

        # 2. Atmosphere (Density and Speed of Sound)
        # FIX: Use a more robust temperature model with limits
        Temp = (15.04 - 0.00649 * h) + 273.1
        # Clamp temperature to reasonable bounds (prevent negative/very low temps)
        Temp = np.maximum(Temp, 216.65)  # Stratosphere minimum temp
        
        pressure = 101.29 * (Temp/288.08)**5.256
        rho = self.rho0 * np.exp(-h / self.H)
        a = np.sqrt(self.gamma_atm * self.R_gas * Temp) 
        
        # Protect against division by zero or very small speed of sound
        a = np.maximum(a, 200.0)  # Minimum speed of sound ~200 m/s
        outputs['Mach'] = V / a
        outputs['rho_atm'] = rho

        # 3. Aerodynamics (Shape-dependent forces) -- using your surrogate models
        q = 0.5 * rho * V**2 # Dynamic Pressure
        outputs['q_dyn'] = q
        
        # Convert alpha to degrees for the model
        alpha_deg = np.rad2deg(alpha)
        
        # Check for out-of-range alpha
        if (np.abs(alpha_deg) > 15).any():
            print(f"Warning: AoA outside trained range: min={np.min(alpha_deg):.2f}, max={np.max(alpha_deg):.2f} deg")
        
        # Build predictor input - ensure all values are finite
        predictor_input = np.column_stack((
            outputs['Mach'],
            q / 1E5,  # convert to bar
            alpha_deg,
            np.zeros_like(h)  # beta = 0 for 2D
        ))
        
        # Safety check: replace any NaN/inf with reasonable defaults before prediction
        if not np.all(np.isfinite(predictor_input)):
            print(f"Warning: Non-finite values in predictor_input detected!")
            print(f"  Mach range: [{np.min(outputs['Mach']):.2f}, {np.max(outputs['Mach']):.2f}]")
            print(f"  q range: [{np.min(q):.2e}, {np.max(q):.2e}] Pa")
            print(f"  alpha range: [{np.min(alpha_deg):.2f}, {np.max(alpha_deg):.2f}] deg")
            print(f"  h range: [{np.min(h):.2f}, {np.max(h):.2f}] m")
            print(f"  V range: [{np.min(V):.2f}, {np.max(V):.2f}] m/s")
            # Replace NaNs with reasonable defaults
            predictor_input = np.nan_to_num(predictor_input, 
                                            nan=0.0, 
                                            posinf=10.0, 
                                            neginf=0.0)
        
        # Transform and predict
        try:
            # Use scaling only if a scaler exists
            if self.scalers["X_scaler"] is not None:
                X_in = self.scalers["X_scaler"].transform(predictor_input)
            else:
                X_in = predictor_input

            # Models are sklearn models with .predict()
            CL = self.model_list[1].predict(X_in)
            CD = self.model_list[0].predict(X_in)
            q_dot = self.model_list[3].predict(X_in)

        except Exception as e:
            print(f"Error in model prediction: {e}")
            print(f"Predictor input shape: {predictor_input.shape}")
            print(f"Predictor input:\n{predictor_input}")
            raise

        Drag = q * S * CD
        Lift = q * S * CL
        outputs['Drag'] = Drag
        outputs['Lift'] = Lift
        outputs['CL'] = CL
        outputs['CD'] = CD
        outputs['q_dot'] = q_dot

        # Rest of the method remains the same...
        # 4. Mass flow rate
        m_dot = -T / (Isp * g0) if Isp != 0 else 0.0
        outputs['m_dot'][:] = m_dot 

        # 5. G-Load Calculation
        outputs['g_load'] = Lift / (m * g0)

        # --- Equations of Motion (EOMs) ---
        outputs['h_dot'] = V * np.sin(gamma)
        outputs['V_dot'] = (T * np.cos(alpha) - Drag) / m - g * np.sin(gamma)
        V_safe = np.maximum(V, 1.0)  # Prevent division by very small velocities
        outputs['gamma_dot'] = (T * np.sin(alpha) + Lift) / (m * V_safe) + (V_safe / r - g / V_safe) * np.cos(gamma)
        outputs['x_dot'] = V * np.cos(gamma)

def write_traj_file(sim_out, path):
    """
    Write trajectory data to a .traj file
    """
    time = sim_out.get_val('time')
    V = sim_out.get_val('V')
    Mach = sim_out.get_val('Mach')
    x = sim_out.get_val('x')
    h = sim_out.get_val('h')
    rho_atm = sim_out.get_val('rho_atm')
    lift = sim_out.get_val('Lift')
    drag = sim_out.get_val('Drag')
    q_dot = sim_out.get_val('q_dot')
    alpha = sim_out.get_val('alpha')
    gamma = sim_out.get_val('gamma')
    q_dyn = sim_out.get_val('q_dyn')
    m = sim_out.get_val('m')
    
    # Calculate L/D
    ld = np.where(np.abs(drag) > 1e-6, lift / drag, 0.0)
    
    traj_file = os.path.join(path, 'trajectory.traj')
    with open(traj_file, 'w') as f:
        # Header
        f.write(f"{'time':>12s} {'velocity':>12s} {'Mach':>12s} {'downrange':>12s} "
                f"{'altitude':>12s} {'rho_atm':>12s} {'L':>12s} {'D':>12s} "
                f"{'q_dot':>12s} {'alpha':>12s} {'gamma':>12s} {'L/D':>12s} "
                f"{'q_dyn':>12s} {'mass':>12s}\n")
        
        f.write(f"{'(s)':>12s} {'(m/s)':>12s} {'':>12s} {'(m)':>12s} "
                f"{'(m)':>12s} {'(kg/m³)':>12s} {'(N)':>12s} {'(N)':>12s} "
                f"{'(W/m²)':>12s} {'(deg)':>12s} {'(deg)':>12s} {'':>12s} "
                f"{'(Pa)':>12s} {'(kg)':>12s}\n")
        
        # Data rows
        for i in range(len(time)):
            # Handle both 1D and 2D array indexing
            t_val = time[i] if time.ndim == 1 else time[i][0]
            v_val = V[i] if V.ndim == 1 else V[i][0]
            m_val = Mach[i] if Mach.ndim == 1 else Mach[i][0]
            x_val = x[i] if x.ndim == 1 else x[i][0]
            h_val = h[i] if h.ndim == 1 else h[i][0]
            rho_val = rho_atm[i] if rho_atm.ndim == 1 else rho_atm[i][0]
            l_val = lift[i] if lift.ndim == 1 else lift[i][0]
            d_val = drag[i] if drag.ndim == 1 else drag[i][0]
            q_val = q_dot[i] if q_dot.ndim == 1 else q_dot[i][0]
            a_val = alpha[i] if alpha.ndim == 1 else alpha[i][0]
            g_val = gamma[i] if gamma.ndim == 1 else gamma[i][0]
            ld_val = ld[i] if ld.ndim == 1 else ld[i][0]
            qd_val = q_dyn[i] if q_dyn.ndim == 1 else q_dyn[i][0]
            mass_val = m[i] if m.ndim == 1 else m[i][0]
            
            f.write(f"{t_val:12.4f} {v_val:12.4f} {m_val:12.4f} {x_val:12.4f} "
                    f"{h_val:12.4f} {rho_val:12.6f} {l_val:12.4f} {d_val:12.4f} "
                    f"{q_val:12.4f} {np.rad2deg(a_val):12.4f} {np.rad2deg(g_val):12.4f} "
                    f"{ld_val:12.4f} {qd_val:12.4f} {mass_val:12.4f}\n")
    
    print(f"Trajectory file written to: {traj_file}")

def plot_results(sim_out, path):
    """
    Generates 12 plots from the Dymos simulation timeseries output.
    """
    time = sim_out.get_val('time')
    h = sim_out.get_val('h')
    x = sim_out.get_val('x')
    V = sim_out.get_val('V')
    Mach = sim_out.get_val('Mach')
    g_load = sim_out.get_val('g_load')
    q_dot = sim_out.get_val('q_dot')
    lift = sim_out.get_val('Lift')
    drag = sim_out.get_val('Drag')
    gamma = sim_out.get_val('gamma')
    alpha = sim_out.get_val('alpha')
    q_dyn = sim_out.get_val('q_dyn')
    CL = sim_out.get_val('CL')
    CD = sim_out.get_val('CD')
    
    # Calculate L/D safely
    ld = np.where(drag > 1e-6, lift / drag, 0.0)
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))
    fig.suptitle('Glide/Flight Optimization Results (Start: h=8km, M=8, gamma=0)', fontsize=16)

    # 1. Altitude vs. Downrange
    ax = axes[0, 0]
    ax.plot(x / 1000.0, h / 1000.0, 'b-', linewidth=2)
    ax.set_title('Altitude vs. Downrange', fontsize=12, fontweight='bold')
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True, alpha=0.3)

    # 2. Altitude vs. Time
    ax = axes[0, 1]
    ax.plot(time, h / 1000.0, 'b-', linewidth=2)
    ax.set_title('Altitude vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True, alpha=0.3)

    # 3. Velocity vs. Time
    ax = axes[0, 2]
    ax.plot(time, V, 'r-', linewidth=2)
    ax.set_title('Velocity vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(True, alpha=0.3)

    # 4. Alpha vs. Time
    ax = axes[1, 0]
    ax.plot(time, np.rad2deg(alpha), 'g-', linewidth=2)
    ax.set_title('Angle of Attack vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Alpha (deg)')
    ax.grid(True, alpha=0.3)

    # 5. Gamma vs. Time
    ax = axes[1, 1]
    ax.plot(time, np.rad2deg(gamma), 'purple', linewidth=2)
    ax.set_title('Flight Path Angle vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gamma (deg)')
    ax.grid(True, alpha=0.3)

    # 6. q_dot vs. Time
    ax = axes[1, 2]
    ax.plot(time, q_dot / 1e6, 'orange', linewidth=2)
    ax.set_title('Heat Flux vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q̇ (MW/m²)')
    ax.grid(True, alpha=0.3)

    # 7. L/D vs. Time
    ax = axes[2, 0]
    ax.plot(time, ld, 'cyan', linewidth=2)
    ax.set_title('L/D vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('L/D')
    ax.grid(True, alpha=0.3)

    # 8. Mach vs. Time
    ax = axes[2, 1]
    ax.plot(time, Mach, 'brown', linewidth=2)
    ax.set_title('Mach Number vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach')
    ax.grid(True, alpha=0.3)

    # 9. Lift and Drag vs. Time
    ax = axes[2, 2]
    ax.plot(time, lift, 'b-', linewidth=2, label='Lift')
    ax.plot(time, drag, 'r-', linewidth=2, label='Drag')
    ax.set_title('Lift and Drag vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 10. Dynamic Pressure vs. Time
    ax = axes[3, 0]
    ax.plot(time, q_dyn / 1000.0, 'magenta', linewidth=2)
    ax.set_title('Dynamic Pressure vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q (kPa)')
    ax.grid(True, alpha=0.3)

    # 11. CL and CD vs. Time
    ax = axes[3, 1]
    ax.plot(time, CL, 'b-', linewidth=2, label='CL')
    ax.plot(time, CD, 'r-', linewidth=2, label='CD')
    ax.set_title('Lift and Drag Coefficients vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 12. Lift over Weight vs. Time
    ax = axes[3, 2]
    ax.plot(time, lift / (sim_out.get_val('m') * 9.80665), 'teal', linewidth=2)
    ax.set_title('Lift over Weight vs. Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('L / (m*g0)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(path, 'glide_flight_trajectory_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plots saved to: {os.path.join(path, 'glide_flight_trajectory_results.png')}")


# --- 3. Setup the Dymos Optimization Problem (modified for flight starting at h=8km, M=8, gamma=0) ---
def run_dymos_optimization(path, plotting, surrogate_type="linear"):

### first get the CBaero models ###
    print("Computing CBaero Interpolations")
    model_list, scalers, feature_cols, merged = get_models_for_prefix("ogive")
    # model_list returns [Cd_model, Cl_model, Cm_model, qdot_model]
    print("Interpolation Models Obtained")

    # 1. Create the OpenMDAO Problem
    p = om.Problem(model=om.Group())
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['maxiter'] = 100
    p.driver.options['tol'] = 1.0E-4
    p.driver.declare_coloring()

    # 2. Initialize the Dymos Phase
    # num_segments increased to give more fidelity for descent/glide
    phase = dm.Phase(
        ode_class=lambda **kwargs: RocketBoostEOM(model_list, scalers, feature_cols, **kwargs),
        transcription=dm.GaussLobatto(num_segments=30, order=3)
    )

    # 3. Add the Phase to the Problem
    p.model.add_subsystem('phase0', phase)

    # Timeseries outputs we want to inspect from the ODE
    phase.add_timeseries_output('Mach')
    phase.add_timeseries_output('Lift')
    phase.add_timeseries_output('Drag')
    phase.add_timeseries_output('g_load') 
    phase.add_timeseries_output('q_dot')
    phase.add_timeseries_output('q_dyn')
    phase.add_timeseries_output('rho_atm')
    phase.add_timeseries_output('CL')
    phase.add_timeseries_output('CD')
    
    # States are automatically in timeseries
    # x, h, V, gamma, alpha, m are already available

    # 4. Set Time (now for the full flight; allow long duration)
    phase.set_time_options(fix_initial=True, duration_bounds=(0.0, 10000.0), duration_ref=200.0)

    # 5. Define States and their Bounds
    phase.add_state('h', units='m', rate_source='h_dot',
                    lower=0.0, upper=20000.0, ref=9000.0, defect_ref=9000.0,
                    fix_initial=True, fix_final=False) 

    phase.add_state('V', units='m/s', rate_source='V_dot',
                    lower=50.0, upper=10000.0, ref=3000.0, defect_ref=3000.0,
                    fix_initial=True, fix_final=False) 

    phase.add_state('gamma', units='rad', rate_source='gamma_dot',
                    lower=-np.pi/2, upper=np.pi/2, ref=1.0, defect_ref=1.0,
                    fix_initial=True, fix_final=False) 

    phase.add_state('m', units='kg', rate_source='m_dot',
                    lower=10, ref=45,
                    fix_initial=True, # initial mass fixed
                    fix_final=False) 

    phase.add_state('x', units='m', rate_source='x_dot',
                    fix_initial=True, ref=10000.0, defect_ref=10000.0)

    # 6. Define Controls and Boundary Constraints
    phase.add_control('alpha', units='rad', lower=np.deg2rad(-5), upper=np.deg2rad(8),
                      opt=True, continuity=True)
    
    # Path constraints
    phase.add_path_constraint('h', upper=9000.0, ref=9000.0)
    # phase.add_path_constraint('alpha', lower=np.deg2rad(-2), upper=np.deg2rad(6), ref=0.1)
    # phase.add_path_constraint('g_load', upper=25.0, ref=10.0)
    # phase.add_path_constraint('q_dot', upper=12000000.0, ref=100000.0)

    #phase.add_boundary_constraint('V', loc='final', lower=2*343.0, upper=4*343.0, ref=1000.0)

    # 7. Define Parameters (T set to 0 to model unpowered flight)
    phase.add_parameter('T', units='N', opt=False, val=0.0)   # No thrust for flight/glide optimization
    phase.add_parameter('Isp', units='s', opt=False, val=250.0)
    phase.add_parameter('S', units='m**2', opt=False, val=0.0124) 
    phase.add_parameter('CD_0', opt=False, val=0.1)
    phase.add_parameter('CL_alpha', opt=False, val=0.1)
    phase.add_parameter('payload_mass', units='kg', val=9.0, opt=False)

    # 8. Set the Objective Function: maximize final downrange x
    # Dymos objectives are minimized by default; multiply by -1 to maximize
    phase.add_objective('x', loc='final', scaler=-1.0)

    # 9. Setup the Problem and Set Initial Guesses
    p.setup(check=True)

    # Compute initial speed from Mach = 8 at h = 8000 m using same Temp formula as in the ODE
    h0 = 9000.0
    Temp0 = (15.04 - 0.00649 * h0) + 273.1
    a0 = np.sqrt(1.4 * 287.05 * Temp0)
    mach0 = 8.0
    V0 = mach0 * a0
    print(f"Initial speed for M=8 at h=8000 m: V0 = {V0:.3f} m/s (a0 = {a0:.3f} m/s)")

    # Set initial time and duration guess
    p.set_val('phase0.t_initial', 0.0)
    p.set_val('phase0.t_duration', 4000.0)  # initial guess for flight duration

    # Initial mass (fixed) - set to your nominal vehicle mass (example 45 kg)
    p.set_val('phase0.states:m', phase.interp(ys=[45.0, 40.0], nodes='state_input'))
    
    # State Guesses - start at h=8000 m, V = mach8 speed, gamma = 0
    p.set_val('phase0.states:h', phase.interp(ys=[h0, 0.0], nodes='state_input')) 
    p.set_val('phase0.states:V', phase.interp(ys=[V0, 300.0], nodes='state_input')) 
    p.set_val('phase0.states:gamma', phase.interp(ys=[0.0, np.deg2rad(-5.0)], nodes='state_input'))
    p.set_val('phase0.states:x', phase.interp(ys=[0.0, 800000.0], nodes='state_input')) 
    
    # Alpha initial guess (moderate)
    p.set_val('phase0.controls:alpha', np.deg2rad(0.0))

    # Set the parameter T to 0 (unpowered flight)
    p.set_val('phase0.parameters:T', 0.0)

    # 10. Run the Optimization
    print("Starting optimization for unpowered flight (glide) ...")
    dm.run_problem(p, simulate=True) 
    print("Optimization finished.")

    # 11. Output the Results
    sim_out = p.model.phase0.timeseries
    
    initial_gamma_rad = sim_out.get_val('gamma')[0].item()
    initial_gamma_deg = np.rad2deg(initial_gamma_rad)
    final_velocity = sim_out.get_val('V')[-1].item()
    final_altitude = sim_out.get_val('h')[-1].item()
    max_range = sim_out.get_val('x')[-1].item() 
    max_q_dot = np.max(sim_out.get_val('q_dot'))
    final_g_load = sim_out.get_val('g_load')[-1].item() 
    alpha_profile = sim_out.get_val('alpha')
    alpha_start_deg = np.rad2deg(alpha_profile[0].item())
    alpha_end_deg = np.rad2deg(alpha_profile[-1].item())

    print("\n" + "="*50)
    print(" FLIGHT OPTIMIZATION RESULTS (Start: h=8km, M=8, gamma=0; T=0)")
    print("="*50)
    print(f"Initial Flight Path Angle (gamma_0): {initial_gamma_deg:.2f} degrees")
    print(f"Final Velocity: {final_velocity:.2f} m/s")
    print(f"Final Altitude: {final_altitude:.2f} meters")
    print(f"Maximum Range Achieved: {max_range / 1000.0:.2f} km") 
    print(f"Final G-Load: {final_g_load:.2f} g")
    print(f"Angle of Attack (alpha) Profile (start to end): {alpha_start_deg:.2f} deg to {alpha_end_deg:.2f} deg")
    print("="*50 + "\n")

    print(f"MAX RANGE = {max_range}")
    print(f"MAX Q_dot = {max_q_dot}")
    
    # 12. Plot the results
    if plotting:
        plot_results(sim_out, path)
        print(f"Plots saved to {path}")
    
    # 13. Write trajectory file
    write_traj_file(sim_out, path)

    return max_range, max_q_dot


if __name__ == '__main__':
    run_dymos_optimization(r"./", plotting = True, surrogate_type="linear")