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

def extract_geometry_from_dir(dirname, prefix="waverider"):
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

def collect_data_across_dirs(param, model_prefix="waverider"):
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

def build_global_database(model_prefix):
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
    scalers["X_scaler"] = X_scaler

    for target in target_params:
        if target not in merged.columns:
            continue
        y = merged[target].values
        # drop rows where target is nan
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            print(f"Skipping training for {target}: not enough valid samples ({mask.sum()})")
            continue
        X_train = X_scaled[mask]
        y_train = y[mask]

        # Simple GPR kernel: constant * Matern + white noise
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        print(f"Training GPR for {target} on {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        gpr.fit(X_train, y_train)
        models[target] = gpr

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
    Computes the differential equations for a 2D atmospheric rocket boost with lift and AOA as control variables
    """
    def initialize(self):
        """
        Define constant parameters for the environment
        """
        self.options.declare('num_nodes', types=int)
        self.R_E = 6378137.0  # Earth radius (m)
        self.g0 = 9.80665    # Gravity at sea level (m/s^2)
        
        # Atmospheric constants (Simplified Isothermal Model)
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
        self.add_input('initV', val=np.zeros(nn), units='m/s', desc=' Initial Velocity')
        self.add_input('V', val=np.zeros(nn), units='m/s', desc='Velocity')
        self.add_input('gamma', val=np.zeros(nn), units='rad', desc='Flight path angle')
        self.add_input('m', val=np.zeros(nn), units='kg', desc='Vehicle Mass')
        
        # Control Input: Angle of Attack (alpha)
        self.add_input('alpha', val=np.zeros(nn), units='rad', desc='Angle of attack')
        
        # Parameters (defining the vehicle/engine/shape characteristics)
        self.add_input('T', val=4250.0, units='N', desc='Engine Thrust (constant)')
        self.add_input('Isp', val=250.0, units='s', desc='Specific Impulse')
        self.add_input('S', val=0.0124, units='m**2', desc='Reference Area (fixed to 0.0314 m^2)')
        self.add_input('payload_mass', val=9.0, units='kg', desc='Payload mass')
        
        # Shape Parameters (simplified)
        self.add_input('CD_0', val=0.1, desc='Zero-lift drag coefficient (related to nose/friction)')
        # Take this shit outtttt
        self.add_input('CL_alpha', val=0.1, desc='Lift coefficient slope (how much lift per rad of alpha)')

        # Outputs (State Derivatives and intermediate values)
        self.add_output('v', val = np.zeros(nn), units = 'm/s', desc = 'velocity over time')
        self.add_output('h_dot', val=np.zeros(nn), units='m/s', desc='dh/dt: Rate of change of altitude')
        self.add_output('V_dot', val=np.zeros(nn), units='m/s**2', desc='dV/dt: Acceleration')
        self.add_output('gamma_dot', val=np.zeros(nn), units='rad/s', desc='dgamma/dt: Rate of change of flight path angle')
        self.add_output('m_dot', val=np.zeros(nn), units='kg/s', desc='dm/dt: Mass flow rate')
        self.add_output('x_dot', val=np.zeros(nn), units='m/s', desc='dx/dt: Rate of change of horizontal range')
        
        self.add_output('Drag', val=np.zeros(nn), units='N', desc='Total Drag Force')
        self.add_output('Lift', val=np.zeros(nn), units='N', desc='Total Lift Force')
        self.add_output('Mach', val=np.zeros(nn), desc='Mach Number')
        self.add_output('g_load', val=np.zeros(nn), units=None, desc='Load factor (L / (m*g0))')
        self.add_output('q_dot', val=np.zeros(nn), units='W/m**2', desc='Stagnation Heat flux rate')  # Heat flux rate


        # Partials setup
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
        Temp = (15.04 - 0.00649 * h) + 273.1
        pressure = 101.29 * (Temp/288.08)**5.256
        rho = self.rho0 * np.exp(-h / self.H)
        a = np.sqrt(self.gamma_atm * self.R_gas * Temp) 
        outputs['Mach'] = V / a

        # 3. Aerodynamics (Shape-dependent forces)
        q = 0.5 * rho * V**2 # Dynamic Pressure
        
        # will get values from aakash (the goat)
        CL = self.model_list[1].predict(self.scalers['X_scaler'].transform(
            np.column_stack((
                outputs['Mach'],
                q/10E5, #convert to bar
                alpha,
                np.zeros_like(h),  # beta = 0 for 2D
            ))))
        CD = self.model_list[0].predict(self.scalers['X_scaler'].transform(
            np.column_stack((
                outputs['Mach'],
                q/10E5, #convert to bar
                alpha,
                np.zeros_like(h),  # beta = 0 for 2D
            ))))

        q_dot = self.model_list[3].predict(self.scalers['X_scaler'].transform(
            np.column_stack((
                outputs['Mach'],
                q/10E5, #convert to bar
                alpha,
                np.zeros_like(h),  # beta = 0 for 2D
            ))))

        Drag = q * S * CD
        Lift = q * S * CL
        outputs['Drag'] = Drag
        outputs['Lift'] = Lift
        outputs['q_dot'] = q_dot

        # 4. Mass flow rate
        m_dot = -T / (Isp * g0)
        outputs['m_dot'][:] = m_dot 

        # 5. G-Load Calculation
        outputs['g_load'] = Lift / (m * g0)

        # --- Equations of Motion (EOMs) ---
        outputs['h_dot'] = V * np.sin(gamma)
        outputs['V_dot'] = (T * np.cos(alpha) - Drag) / m - g * np.sin(gamma)
        outputs['gamma_dot'] = (T * np.sin(alpha) + Lift) / (m * V) + (V / r - g / V) * np.cos(gamma)
        outputs['x_dot'] = V * np.cos(gamma) # Horizontal range rate
        # outputs['v'] = 


# --- 2. Define the Plotting Function ---
def plot_results(sim_out):
    """
    Generates plots from the Dymos simulation timeseries output.
    """
    time = sim_out.get_val('time')
    h = sim_out.get_val('h')
    x = sim_out.get_val('x')
    V = sim_out.get_val('V')
    Mach = sim_out.get_val('Mach')
    g_load = sim_out.get_val('g_load')
    alpha_deg = np.rad2deg(sim_out.get_val('alpha'))

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.suptitle('Rocket Boost Trajectory Optimization Results', fontsize=16)

    # 1. Altitude vs. Time
    ax = axes[0, 0]
    ax.plot(time, h / 1000.0)
    ax.set_title('Altitude vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)

    # 2. Altitude vs. Range (Trajectory)
    ax = axes[0, 1]
    ax.plot(x / 1000.0, h / 1000.0)
    ax.set_title('Trajectory (Altitude vs. Range)')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)

    # 3. Mach vs. Time
    ax = axes[1, 0]
    ax.plot(time, Mach)
    ax.set_title('Mach Number vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach')
    ax.grid(True)

    # 4. G-Load vs. Time
    ax = axes[1, 1]
    ax.plot(time, g_load)
    ax.set_title('G-Load vs. Time (Maximized by Aerodynamics)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('G-Load (L/W)')
    ax.grid(True)

    # 5. Angle of Attack vs. Time
    ax = axes[2, 0]
    ax.plot(time, alpha_deg)
    ax.set_title('Angle of Attack vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Alpha (deg)')
    ax.grid(True)
    
    # 6. Velocity vs. Time
    ax = axes[2, 1]
    ax.plot(time, V)
    ax.set_title('Velocity vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 3. Setup the Dymos Optimization Problem ---
def run_dymos_optimization():

    ### first get the CBaero models ###
    print("Computing CBaero Interpolations")
    model_list, scalers, feature_cols, merged = get_models_for_prefix("waverider")
    #model list returns [Cd_model, Cl_model, Cm_model, qdot_model]
    print("Interpolation Models Obtained")


    # 1. Create the OpenMDAO Problem
    p = om.Problem(model=om.Group())
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['maxiter'] = 1000 
    p.driver.options['tol'] = 1.0E-6
    p.driver.declare_coloring()

    # 2. Initialize the Dymos Phase
    # num_segments set to 20 for faster computation
    phase = dm.Phase(
        ode_class=lambda **kwargs: RocketBoostEOM(model_list, scalers, feature_cols, **kwargs),
        transcription=dm.GaussLobatto(num_segments=100, order=3)
    )

    # 3. Add the Phase to the Problem
    p.model.add_subsystem('phase0', phase)

    phase.add_timeseries_output('Mach')
    phase.add_timeseries_output('Lift')
    phase.add_timeseries_output('Drag')
    phase.add_timeseries_output('x')      
    phase.add_timeseries_output('g_load') 
    phase.add_timeseries_output('alpha')

    # 4. Set Time (Booster Burn Time)
    phase.set_time_options(fix_initial=True, duration_bounds=(10, 30), duration_ref=20.0)

    # 5. Define States and their Bounds
    # Altitude is free to vary, but constrained by the path constraint (see below)
    phase.add_state('h', units='m', rate_source='h_dot',
                    lower=0.0, upper=9000.0, ref=9000.0, defect_ref=9000.0,
                    fix_initial=True, fix_final=False) 

    phase.add_state('V', units='m/s', rate_source='V_dot',
                    lower=1.0, ref=3000.0, defect_ref=3000.0,
                    fix_initial=True) 

    phase.add_state('gamma', units='rad', rate_source='gamma_dot',
                    lower=np.deg2rad(0), upper=np.pi/2, ref=1.0, defect_ref=1.0,
                    fix_initial=False) 

    phase.add_state('m', units='kg', rate_source='m_dot',
                    lower=10.0, ref=45.0,
                    fix_initial=True, # Initial mass is fixed at 45 kg
                    fix_final=False) 

    phase.add_state('x', units='m', rate_source='x_dot',
                    fix_initial=True, ref=10000.0, defect_ref=10000.0)

    # 6. Define Controls and Boundary Constraints
    phase.add_control('alpha', units='rad', lower=np.deg2rad(0), upper=np.deg2rad(6),
                      opt=True)
    
    # Boundary constraints
    phase.add_boundary_constraint('m', loc='final', lower=19.0)    
    
    # CRITICAL CHANGE: REMOVED G-LOAD CONSTRAINT
    
    # Path constraints
    # Maximum altitude is constrained to 9000m throughout the flight
    phase.add_path_constraint('h', upper=9000.0, ref=9000.0)
    phase.add_path_constraint('Mach', lower=5.0, upper=8.0, ref=0.01) # Tight Mach constraint
    phase.add_path_constraint('alpha', lower=np.deg2rad(0), upper=np.deg2rad(6), ref=0.1)
    #phase.add_path_constraint('g_load', upper=15.0, ref=10.0)  # Max g-load constraint
    #phase.add_path_constraint('q_dot', upper=500000.0, ref=100000.0)  # Max heat flux constraint
    
    # 7. Define Parameters (S is fixed at 0.0314 m^2)
    phase.add_parameter('T', units='N', opt=False, val=4250.0)
    phase.add_parameter('Isp', units='s', opt=False, val=250.0)
    phase.add_parameter('S', units='m**2', opt=False, val=0.0124) 
    phase.add_parameter('CD_0', opt=False, val=0.1)
    phase.add_parameter('CL_alpha', opt=False, val=0.1)
    phase.add_parameter('payload_mass', units='kg', val=9.0, opt=False)

    # 8. Set the Objective Function
    phase.add_objective('x', loc='final', scaler=-1.0)

    # 9. Setup the Problem and Set Initial Guesses
    p.setup(check=True)

    # Set initial values
    p.set_val('phase0.t_initial', 0.0)
    p.set_val('phase0.t_duration', 15.0)
    
    # Initial mass fixed at 45 kg
    p.set_val('phase0.states:m', phase.interp(ys=[45.0, 19.0], nodes='state_input'))
    
    # State Guesses 
    p.set_val('phase0.states:h', phase.interp(ys=[0.0, 9000], nodes='state_input')) 
    p.set_val('phase0.states:V', phase.interp(ys=[1.0, 2000], nodes='state_input')) 
    p.set_val('phase0.states:gamma', phase.interp(ys=[np.deg2rad(85), np.deg2rad(5)], nodes='state_input'))
    p.set_val('phase0.states:x', phase.interp(ys=[0.0, 450000.0], nodes='state_input')) 
    
    # Using a moderate alpha guess for stability
    p.set_val('phase0.controls:alpha', np.deg2rad(3.0))
    
    # 10. Run the Optimization
    print("Starting optimization...")
    dm.run_problem(p, simulate=True) 
    print("Optimization finished.")

    # 11. Output the Results
    sim_out = p.model.phase0.timeseries
    
    optimal_angle_rad = sim_out.get_val('gamma')[0].item()
    optimal_angle_deg = np.rad2deg(optimal_angle_rad)
    final_velocity = sim_out.get_val('V')[-1].item()
    final_altitude = sim_out.get_val('h')[-1].item()
    max_range = sim_out.get_val('x')[-1].item() 
    final_g_load = sim_out.get_val('g_load')[-1].item() 
    alpha_profile = sim_out.get_val('alpha')
    alpha_start_deg = np.rad2deg(alpha_profile[0].item())
    alpha_end_deg = np.rad2deg(alpha_profile[-1].item())

    print("\n" + "="*50)
    print(" OPTIMIZATION RESULTS (MAX RANGE, Max H=9km)")
    print("="*50)
    #print(f"Initial Mass: {p.get_val('phase0.states:m')[0].item():.2f} kg")
    print(f"Optimal Initial Pitch Angle (gamma_0): {optimal_angle_deg:.2f} degrees")
    print(f"Final Velocity at Burnout: {final_velocity:.2f} m/s")
    print(f"Final Altitude at Burnout: {final_altitude:.2f} meters")
    print(f"Maximum Range Achieved: {max_range / 1000.0:.2f} km") 
    #print(f"Final G-Load (Maneuverability): {final_g_load:.2f} g") # G-Load is now calculated, not constrained
    #print(f"Angle of Attack (alpha) Profile (start to end): {alpha_start_deg:.2f} deg to {alpha_end_deg:.2f} deg")
    print("="*50 + "\n")
    
    # 12. Plot the results
    plot_results(sim_out)

if __name__ == '__main__':
    run_dymos_optimization()
