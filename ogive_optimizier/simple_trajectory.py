import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import os
import glob
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# ====================== PHYSICAL CONSTANTS ======================
G_0 = 9.80665  # m/s²
R_e = 6371000  # m
GAMMA_PERF = 1.4
R = 287.05  # J/(kg·K)

# ====================== UNIT CONVERSIONS ======================
BAR_TO_PA = 100000.0  # 1 bar = 100,000 Pa

# ====================== CBAERO DATABASE FUNCTIONS ======================
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

def find_files(directory, model_name, param):
    totl_params = ["CDw", "CLw", "CFx", "CFy", "CFz", "CMl", "CMn", "CMm", "CMx", "CMy", "CMz", "CSw"]
    if param in totl_params:
        pattern = f"{model_name}.{param}.Totl.dat"
    else:
        pattern = f"{model_name}.{param}.dat"
    return glob.glob(os.path.join(directory, pattern))

def extract_geometry_from_dir(dirname, prefix="ogive"):
    if not dirname.startswith(prefix + "_"):
        return {}
    parts = dirname.replace(prefix + "_", "").split("_")
    geom = {}
    for i, p in enumerate(parts):
        try:
            geom[f"x{i+1}"] = int(p)
        except ValueError:
            try:
                geom[f"x{i+1}"] = float(p)
            except ValueError:
                continue
    return geom

def collect_data_across_dirs(param, model_prefix="ogive"):
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
            files = find_files(entry.path, model_prefix, req_param)
            if not files:
                print(f"Skipping {entry.name}: no file for {req_param}")
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
        raise RuntimeError(f"No data found for parameter {param} in any matching directory for prefix '{model_prefix}'.")

    return pd.concat(all_data, ignore_index=True)

def build_global_database(model_prefix, surrogate_type="linear"):
    """Build global database and train surrogate models."""
    target_params = ["CLw", "CDw", "CMn", "MaxQdotTotalQdotConvection"]
    collected = {}

    for param in target_params:
        try:
            collected[param] = collect_data_across_dirs(param, model_prefix=model_prefix)
            collected[param] = collected[param].rename(columns={"F": param})
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
            median = merged[g].median(skipna=True)
            merged[g] = merged[g].fillna(median)

    outcsv = f"global_database_{model_prefix}.csv"
    merged.to_csv(outcsv, index=False)
    print(f"Global database written to: {outcsv}")

    feature_cols = ["Mach", "q", "alpha", "beta"] + geom_cols
    X = merged[feature_cols].values

    models = {}
    scalers = {}

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    if surrogate_type == "gpr":
        scalers["X_scaler"] = X_scaler
    else:
        scalers["X_scaler"] = None

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
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]),
                                                  length_scale_bounds=(1e-2, 1e3)) \
                     + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=3, random_state=0)
            print(f"Training GPR for {target} on {X_train.shape[0]} samples")
            model.fit(X_train, y_train)
        elif surrogate_type == "linear":
            print(f"Training Linear Regression for {target} on {X_train.shape[0]} samples")
            model = LinearRegression()
            model.fit(X_train, y_train)

        models[target] = model

    return merged, models, scalers, feature_cols

# ====================== REALISTIC VEHICLE ======================
class Vehicle:
    def __init__(self):
        self.mass_ogive_kg = 45.0  # kg (glide vehicle)
        self.S_ref = 0.0124  # m²
        self.target_altitude = 8000  # m (start altitude for glide)
        
    def get_atmosphere(self, altitude):
        if altitude <= 11000:
            T = 15.04 - 0.00649 * altitude
        else:
            T = -56.46
        T_kelvin = T + 273.1
        a = np.sqrt(GAMMA_PERF * R * T_kelvin)
        return None, a, T_kelvin

# ====================== AERODYNAMIC DATABASE ======================
class AeroDatabase:
    def __init__(self, models, scalers, feature_cols, S_ref):
        """Initialize with trained surrogate models."""
        self.models = models
        self.scalers = scalers
        self.feature_cols = feature_cols
        self.S_ref = S_ref
        
    def get_coefficients(self, Mach, q_pa, alpha_deg):
        """Get CL and CD from surrogate models."""
        q_bar = q_pa / BAR_TO_PA
        
        # Clamp to reasonable ranges
        Mach = max(min(Mach, 10.0), 0.1)
        q_bar = max(min(q_bar, 20.0), 0.001)
        alpha_deg = max(min(alpha_deg, 10.0), -10.0)
        
        # Build input
        predictor_input = np.array([[Mach, q_bar, alpha_deg, 0.0]])  # beta = 0
        
        # Transform if needed
        if self.scalers["X_scaler"] is not None:
            X_in = self.scalers["X_scaler"].transform(predictor_input)
        else:
            X_in = predictor_input
        
        # Predict
        CL = self.models["CLw"].predict(X_in)[0]
        CD = self.models["CDw"].predict(X_in)[0]
        
        return float(CL), float(CD)
    
    def get_heat_flux(self, Mach, q_pa, alpha_deg):
        """Get heat flux from surrogate model."""
        q_bar = q_pa / BAR_TO_PA
        
        predictor_input = np.array([[Mach, q_bar, alpha_deg, 0.0]])
        
        if self.scalers["X_scaler"] is not None:
            X_in = self.scalers["X_scaler"].transform(predictor_input)
        else:
            X_in = predictor_input
        
        if "MaxQdotTotalQdotConvection" in self.models:
            q_dot = self.models["MaxQdotTotalQdotConvection"].predict(X_in)[0]
            return float(q_dot)
        else:
            # Fallback approximation
            return 1.83e-4 * np.sqrt(q_pa) * (Mach**3.15)

# ====================== ATMOSPHERE ======================
class Atmosphere:
    @staticmethod
    def atmo_model(h):
        """Standard atmosphere model"""
        if h <= 0:
            h = 0
        
        if h <= 11000:
            T = 15.04 - 0.00649 * h  # Celsius
        else:
            T = -56.46  # Celsius
            
        T_kelvin = T + 273.1
        
        if h <= 11000:
            p = 101.29 * ((T_kelvin) / 288.08) ** 5.256  # kPa
        else:
            p = 22.65 * np.exp(1.73 - 0.000157 * h)  # kPa
            
        rho = p / (0.2869 * T_kelvin)  # kg/m³
        a = np.sqrt(GAMMA_PERF * R * T_kelvin)  # m/s
        
        return rho, a, T_kelvin

# ====================== TRAJECTORY INTEGRATOR ======================
class TrajectoryIntegrator:
    def __init__(self, vehicle, aero_db):
        self.vehicle = vehicle
        self.aero_db = aero_db
        
    def glide_phase_with_alpha_profile(self, initial_state, alpha_profile_func, max_time=3600):
        """
        Glide phase with specified alpha profile function
        alpha_profile_func(t, state) -> alpha_deg
        
        """
        x0, h0, v0, gamma0, m0 = initial_state
        m = self.vehicle.mass_ogive_kg
        
        # Track maximum heat flux
        self.max_q_dot = 0
        self.q_dot_history = []
        self.alphas_history = []
        self.L_history = []
        self.D_history = []
        self.time_history = []
        
        def ode(t, y):
            x, h, v, gamma, m = y
            
            # CRITICAL FIX: Enforce gamma bounds to prevent backward flight
            # Limit gamma to [-85°, +85°] to keep cos(gamma) positive
            gamma = np.clip(gamma, -np.deg2rad(85), np.deg2rad(85))
            
            # Atmosphere
            rho, a, T = Atmosphere.atmo_model(h)
            Mach = v / max(a, 1.0)
            q_pa = 0.5 * rho * v**2
            
            # Get alpha from profile function
            alpha_deg = alpha_profile_func(t, y)
            alpha = np.deg2rad(alpha_deg)
            
            # Aerodynamics
            CL, CD = self.aero_db.get_coefficients(Mach, q_pa, alpha_deg)
            
            # ADDITIONAL FIX: Ensure coefficients are reasonable
            # Sometimes models can return negative or extreme values
            CL = np.clip(CL, -2.0, 5.0)  # Reasonable CL bounds
            CD = np.clip(CD, 0.001, 5.0)  # CD should always be positive
            
            L = CL * q_pa * self.vehicle.S_ref
            D = CD * q_pa * self.vehicle.S_ref
            
            # Heat flux
            q_dot = self.aero_db.get_heat_flux(Mach, q_pa, alpha_deg)
            
            # Track max heat flux
            if q_dot > self.max_q_dot:
                self.max_q_dot = q_dot
            
            # Gravity
            g = G_0 * (R_e / (R_e + h))**2
            
            # Equations of motion
            x_dot = v * np.cos(gamma)
            h_dot = v * np.sin(gamma)
            v_dot = -D/m - g * np.sin(gamma)
            
            # FIX: Improved gamma_dot calculation with limiting
            if v > 10.0:
                gamma_dot_raw = (L/m - g * np.cos(gamma)) / v
                
                # Limit rate of change of gamma to prevent numerical issues
                # Max 10 deg/s change rate
                gamma_dot = np.clip(gamma_dot_raw, -np.deg2rad(10), np.deg2rad(10))
                
                # Additional check: if gamma is approaching limits, reduce rate
                if gamma < -np.deg2rad(80):
                    gamma_dot = max(gamma_dot, 0)  # Don't let it get steeper
                elif gamma > np.deg2rad(80):
                    gamma_dot = min(gamma_dot, 0)  # Don't let it get steeper upward
            else:
                gamma_dot = 0.0
            
            return [x_dot, h_dot, v_dot, gamma_dot, 0.0]
        
        # Events
        def ground_event(t, y):
            return y[1]  # altitude
        ground_event.terminal = True
        ground_event.direction = -1
        
        # ADD: Stall/failure event for extreme gamma
        def extreme_gamma_event(t, y):
            return np.deg2rad(89) - abs(y[3])  # Stop if |gamma| > 89°
        extreme_gamma_event.terminal = True
        extreme_gamma_event.direction = -1
        
        # Initial state with gamma bounds check
        gamma0 = np.clip(gamma0, -np.deg2rad(85), np.deg2rad(85))
        y0 = [x0, h0, v0, gamma0, m]
        
        # Integrate
        sol = solve_ivp(
            ode,
            [0, max_time],
            y0,
            events=[ground_event, extreme_gamma_event],
            max_step=1.0,
            rtol=1e-4,
            atol=1e-6,
            method='RK45'
        )
        
        if not sol.success:
            return None
            
        # Calculate histories - now including L and D
        for i in range(len(sol.t)):
            h = sol.y[1, i]
            v = sol.y[2, i]
            rho, a, T = Atmosphere.atmo_model(h)
            Mach = v / max(a, 1.0)
            q_pa = 0.5 * rho * v**2
            
            # Get alpha
            alpha_deg = alpha_profile_func(sol.t[i], sol.y[:, i])
            
            # Get aerodynamic coefficients
            CL, CD = self.aero_db.get_coefficients(Mach, q_pa, alpha_deg)
            CL = np.clip(CL, -2.0, 5.0)
            CD = np.clip(CD, 0.001, 5.0)
            
            # Calculate forces
            L = CL * q_pa * self.vehicle.S_ref
            D = CD * q_pa * self.vehicle.S_ref
            
            # Get heat flux
            q_dot = self.aero_db.get_heat_flux(Mach, q_pa, alpha_deg)
            
            # Store all histories
            self.q_dot_history.append(q_dot)
            self.alphas_history.append(alpha_deg)
            self.L_history.append(L)
            self.D_history.append(D)
            self.time_history.append(sol.t[i])
        
        # Results
        final_state = sol.y[:, -1]
        x_final = final_state[0]
        
        # Check if we terminated due to extreme gamma
        if len(sol.t_events[1]) > 0:
            print(f"  ⚠️ Trajectory terminated due to extreme flight path angle")
        
        return {
            'time': sol.t,
            'states': sol.y,
            'final_range': x_final,
            'max_q_dot': self.max_q_dot,
            'q_dot': np.array(self.q_dot_history),
            'q_dot_history': self.q_dot_history,
            'time_history': self.time_history,
            'alphas': np.array(self.alphas_history),
            'L': np.array(self.L_history),
            'D': np.array(self.D_history),
        }

# def run_dymos_optimization(path, plotting=True, surrogate_type="linear"):
#     """
#     Drop-in replacement for Dymos optimization using simple trajectory solver.
#     Now includes optimization of initial deployment conditions.
    
#     Returns:
#         max_range: Maximum range achieved (m)
#         max_q_dot: Maximum heat flux (W/m²)
#     """
#     print("\n" + "="*70)
#     print("SIMPLE TRAJECTORY OPTIMIZATION WITH DEPLOYMENT OPTIMIZATION")
#     print("="*70)
    
#     # Build global database and train models
#     print("\n[1/4] Building aerodynamic database...")
#     try:
#         merged, models, scalers, feature_cols = build_global_database("ogive", surrogate_type=surrogate_type)
#     except Exception as e:
#         print(f"ERROR: Failed to build global database: {e}")
#         return 0, 1e9
    
#     # Create vehicle and aero database
#     vehicle = Vehicle()
#     aero_db = AeroDatabase(models, scalers, feature_cols, vehicle.S_ref)
    
#     # Create integrator
#     integrator = TrajectoryIntegrator(vehicle, aero_db)
    
#     # ========== DEPLOYMENT CONDITION OPTIMIZATION ==========
#     print("\n[2/4] OPTIMIZING DEPLOYMENT CONDITIONS")
#     print("-"*50)
    
#     # Define parameter ranges to test
#     altitude_range = [7000, 7500, 8000, 8500, 9000]  # meters
#     gamma_range = [-9, -7, -5, -2, 0, 2, 5]  # degrees
#     mach_fixed = 8.0  # Keep Mach constant at 8
    
#     # Store results for all deployment conditions
#     deployment_results = []
#     successful_deployments = []
    
#     print(f"Testing {len(altitude_range)} altitudes × {len(gamma_range)} flight path angles = {len(altitude_range)*len(gamma_range)} combinations")
#     print(f"Mach number fixed at: {mach_fixed}")
#     print()
    
#     # Test all combinations
#     for h0 in altitude_range:
#         for gamma0_deg in gamma_range:
#             print(f"Testing: h={h0/1000:.1f} km, γ={gamma0_deg:+.1f}°, M={mach_fixed:.1f} ... ", end="")
            
#             # Calculate initial velocity from Mach number
#             rho0, a0, T0 = Atmosphere.atmo_model(h0)
#             v0 = mach_fixed * a0
#             gamma0 = np.deg2rad(gamma0_deg)
#             x0 = 0.0
#             m0 = vehicle.mass_ogive_kg
            
#             initial_state = np.array([x0, h0, v0, gamma0, m0])
            
#             # Test with the best alpha strategy (determined from previous runs)
#             # Using a moderate descent strategy as default
#             def alpha_profile(t, state):
#                 """Moderate alpha profile for testing"""
#                 h_current = state[1]
#                 v_current = state[2]
#                 gamma_current = state[3]
                
#                 # Base alpha for moderate descent
#                 alpha = 1.5
                
#                 # Altitude control
#                 if h_current > 8900:
#                     alpha = max(0.0, alpha - 3.0)
#                 elif h_current > 8700:
#                     alpha = max(0.5, alpha - 1.5)
#                 elif h_current > 8500:
#                     alpha = max(1.0, alpha - 0.5)
                
#                 # Adjust for altitude loss
#                 h_loss = max(0, h0 - h_current)
#                 alpha += -0.3 * (h_loss / 1000.0)
                
#                 # Reduce at low speeds
#                 if v_current < 500:
#                     alpha *= (v_current / 500.0)
                
#                 # Prevent excessive climb
#                 if gamma_current > np.deg2rad(2):
#                     alpha = max(0.0, alpha - 2.0)
                
#                 # Clamp
#                 alpha = max(-2.0, min(4.0, alpha))
                
#                 return alpha
            
#             # Run trajectory
#             try:
#                 result = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile, max_time=3600)
                
#                 if result:
#                     range_km = result['final_range'] / 1000
#                     max_q = result['max_q_dot']
#                     max_alt = np.max(result['states'][1, :])
                    
#                     # Check constraints
#                     ceiling_ok = max_alt <= 9050  # 50m tolerance
#                     q_dot_ok = max_q <= 1.5e6  # Heat flux limit
                    
#                     if ceiling_ok and q_dot_ok:
#                         print(f"✓ Range: {range_km:.1f} km, q̇: {max_q:.0f} W/m²")
                        
#                         deployment_results.append({
#                             'h0': h0,
#                             'gamma0_deg': gamma0_deg,
#                             'v0': v0,
#                             'initial_state': initial_state,
#                             'result': result,
#                             'range_km': range_km,
#                             'max_q_dot': max_q,
#                             'max_altitude': max_alt,
#                             'success': True
#                         })
#                         successful_deployments.append((h0, gamma0_deg, range_km, max_q))
#                     else:
#                         reason = "ceiling" if not ceiling_ok else "heat"
#                         print(f"✗ Failed ({reason})")
#                         deployment_results.append({
#                             'h0': h0,
#                             'gamma0_deg': gamma0_deg,
#                             'success': False,
#                             'reason': reason
#                         })
#                 else:
#                     print("✗ Integration failed")
#                     deployment_results.append({
#                         'h0': h0,
#                         'gamma0_deg': gamma0_deg,
#                         'success': False,
#                         'reason': 'integration'
#                     })
#             except Exception as e:
#                 print(f"✗ Error: {e}")
#                 deployment_results.append({
#                     'h0': h0,
#                     'gamma0_deg': gamma0_deg,
#                     'success': False,
#                     'reason': 'error'
#                 })
    
#     # Find best deployment conditions
#     if not successful_deployments:
#         print("\n✗ ERROR: No successful deployment conditions found!")
#         return 0, 1e9
    
#     print("\n" + "-"*50)
#     print("DEPLOYMENT OPTIMIZATION RESULTS:")
#     print("-"*50)
#     print(f"Successful configurations: {len(successful_deployments)}/{len(deployment_results)}")
    
#     # Sort by range (descending)
#     successful_deployments.sort(key=lambda x: x[2], reverse=True)
    
#     print("\nTop 5 deployment conditions:")
#     print(f"{'Rank':<6} {'Alt (km)':<10} {'γ (deg)':<10} {'Range (km)':<12} {'q̇ (MW/m²)':<12}")
#     print("-"*60)
#     for i, (h, g, r, q) in enumerate(successful_deployments[:5]):
#         print(f"{i+1:<6} {h/1000:<10.1f} {g:<10.1f} {r:<12.1f} {q/1e6:<12.2f}")
    
#     # Select best deployment conditions
#     best_h0, best_gamma0_deg, best_range_deploy, best_q_deploy = successful_deployments[0]
    
#     print(f"\n★ SELECTED DEPLOYMENT CONDITIONS:")
#     print(f"  Altitude: {best_h0/1000:.1f} km")
#     print(f"  Flight path angle: {best_gamma0_deg:.1f}°")
#     print(f"  Mach: {mach_fixed}")
#     print(f"  Expected range: {best_range_deploy:.1f} km")
    
#     # ========== FINAL OPTIMIZATION WITH BEST DEPLOYMENT ==========
#     print("\n[3/4] FINAL TRAJECTORY OPTIMIZATION WITH BEST DEPLOYMENT")
#     print("-"*50)
    
#     # Set up initial conditions with best deployment
#     rho0, a0, T0 = Atmosphere.atmo_model(best_h0)
#     v0 = mach_fixed * a0
#     gamma0 = np.deg2rad(best_gamma0_deg)
#     x0 = 0.0
#     m0 = vehicle.mass_ogive_kg
    
#     initial_state = np.array([x0, best_h0, v0, gamma0, m0])
    
#     print(f"\nOptimal Initial Conditions:")
#     print(f"  Altitude: {best_h0/1000:.1f} km")
#     print(f"  Velocity: {v0:.0f} m/s (Mach {mach_fixed:.1f})")
#     print(f"  Flight path angle: {best_gamma0_deg:.1f}°")
    
#     # Test different alpha strategies with optimal deployment
#     test_alphas = []
#     test_results = []
    
#     alpha_strategies = [
#         ("Original (max L/D)", "original", None),
#         ("Ballistic", 0.0, -0.0),
#         ("Very low alpha", 0.5, -0.1),
#         ("Low alpha", 1.0, -0.2),
#         ("Moderate low", 1.5, -0.3),
#         ("Controlled descent", 2.0, -0.4),
#     ]
    
#     for i, strategy in enumerate(alpha_strategies):
#         if strategy[0] == "Original (max L/D)":
#             name = strategy[0]
#             print(f"\nTest {i+1}/{len(alpha_strategies)}: {name}")
            
#             def alpha_profile(t, state):
#                 """Original strategy - find alpha for max L/D"""
#                 h_current = state[1]
#                 v_current = state[2]
                
#                 rho, a, T = Atmosphere.atmo_model(h_current)
#                 Mach = v_current / max(a, 1.0)
#                 q_pa = 0.5 * rho * v_current**2
                
#                 best_alpha = 0.0
#                 best_LD = 0
                
#                 for alpha_test in np.linspace(0, 4, 9):
#                     CL, CD = aero_db.get_coefficients(Mach, q_pa, alpha_test)
#                     if CD > 0:
#                         LD = CL / CD
#                         if LD > best_LD:
#                             best_LD = LD
#                             best_alpha = alpha_test
                
#                 if h_current > 8800:
#                     best_alpha *= 0.5
                
#                 return best_alpha
            
#         else:
#             name, base_alpha, descent_rate = strategy
#             print(f"\nTest {i+1}/{len(alpha_strategies)}: {name}")
            
#             def make_alpha_profile(base, rate):
#                 def alpha_profile(t, state):
#                     h_current = state[1]
#                     v_current = state[2]
#                     gamma_current = state[3]
                    
#                     alpha = base
                    
#                     if h_current > 8900:
#                         alpha = max(0.0, alpha - 3.0)
#                     elif h_current > 8700:
#                         alpha = max(0.5, alpha - 1.5)
#                     elif h_current > 8500:
#                         alpha = max(1.0, alpha - 0.5)
                    
#                     h_loss = max(0, best_h0 - h_current)
#                     alpha += rate * (h_loss / 1000.0)
                    
#                     if v_current < 500:
#                         alpha *= (v_current / 500.0)
                    
#                     if gamma_current > np.deg2rad(2):
#                         alpha = max(0.0, alpha - 2.0)
                    
#                     alpha = max(-2.0, min(4.0, alpha))
                    
#                     return alpha
                
#                 return alpha_profile
            
#             alpha_profile = make_alpha_profile(base_alpha, descent_rate)
        
#         # Run trajectory
#         result = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile)
        
#         if result:
#             range_km = result['final_range'] / 1000
#             max_q = result['max_q_dot']
#             max_alt = np.max(result['states'][1, :])
            
#             ceiling_ok = "✓" if max_alt <= 9050 else "✗"
            
#             print(f"  → Range: {range_km:.1f} km, Max q̇: {max_q:.0f} W/m², Max alt: {max_alt/1000:.2f} km {ceiling_ok}")
            
#             test_alphas.append(name)
#             test_results.append(result)
#         else:
#             print(f"  → Failed to integrate")
    
#     # Select best result
#     if not test_results:
#         print("\nERROR: All trajectories failed!")
#         return 0, 1e9
    
#     ceiling_limit = 9050
#     q_dot_limit = 1.2e6
#     best_score = -np.inf
#     best_idx = 0
    
#     print("\n" + "-"*50)
#     print("FINAL RESULTS SUMMARY:")
#     print("-"*50)
    
#     for i, result in enumerate(test_results):
#         range_m = result['final_range']
#         max_q = result['max_q_dot']
#         max_alt = np.max(result['states'][1, :])
        
#         ceiling_penalty = 0
#         if max_alt > ceiling_limit:
#             ceiling_penalty = 10000 * (max_alt - ceiling_limit) / 1000
        
#         q_penalty = max(0, (max_q - q_dot_limit) / 1000)
        
#         score = range_m/1000 - ceiling_penalty - q_penalty
        
#         ceiling_status = "✓" if max_alt <= ceiling_limit else f"✗"
#         q_status = "✓" if max_q <= q_dot_limit else "✗"
        
#         print(f"{test_alphas[i]:20s}: Range={range_m/1000:6.1f} km, q̇={max_q:7.0f} W/m² {q_status}, Alt {ceiling_status}, Score={score:7.1f}")
        
#         if score > best_score:
#             best_score = score
#             best_idx = i
    
#     best_result = test_results[best_idx]
#     best_strategy = test_alphas[best_idx]
    
#     print(f"\n★ SELECTED: {best_strategy}")
#     print(f"   with deployment at {best_h0/1000:.1f} km, γ={best_gamma0_deg:.1f}°")
    
#     max_range = best_result['final_range']
#     max_q_dot = best_result['max_q_dot']
#     max_alt = np.max(best_result['states'][1, :])
    
#     print(f"\n[4/4] FINAL OPTIMIZED RESULTS:")
#     print(f"  Deployment: {best_h0/1000:.1f} km altitude, {best_gamma0_deg:.1f}° flight path angle")
#     print(f"  Maximum Range: {max_range/1000:.1f} km")
#     print(f"  Maximum Heat Flux: {max_q_dot:.0f} W/m²")
#     print(f"  Maximum Altitude: {max_alt/1000:.2f} km")
    
#     if max_alt > 9050:
#         print(f"  ⚠️  WARNING: Exceeded 9 km ceiling by {(max_alt-9000)/1000:.2f} km")
    
#     # ========== PLOTTING ==========
#     if plotting:
#         plot_trajectory_results(best_result, path, best_strategy)
    
#     return max_range, max_q_dot

def run_dymos_optimization(path, plotting=True, surrogate_type="linear"):
    """
    Drop-in replacement for Dymos optimization using vehicle-characteristic trajectory solver.
    Generates realistic glide-dive profiles: cruise at/below max L/D, then terminal dive.
    Scores based on range and terminal energy state (no hard Mach constraint).
    
    Returns:
        max_range: Maximum range achieved (m)
        max_q_dot: Maximum heat flux (W/m²)
    """
    print("\n" + "="*70)
    print("VEHICLE-CHARACTERISTIC TRAJECTORY OPTIMIZATION")
    print("="*70)
    
    # Build global database and train models
    print("\n[1/5] Building aerodynamic database...")
    try:
        merged, models, scalers, feature_cols = build_global_database("ogive", surrogate_type=surrogate_type)
    except Exception as e:
        print(f"ERROR: Failed to build global database: {e}")
        return 0, 1e9
    
    # Create vehicle and aero database
    vehicle = Vehicle()
    aero_db = AeroDatabase(models, scalers, feature_cols, vehicle.S_ref)
    
    # Create integrator
    integrator = TrajectoryIntegrator(vehicle, aero_db)
    
    # ========== VEHICLE CHARACTERIZATION ==========
    print("\n[2/5] ANALYZING VEHICLE AERODYNAMIC CHARACTERISTICS")
    print("-"*50)
    
    # Sample the aerodynamic database to understand vehicle behavior
    mach_samples = np.linspace(2, 8, 25)
    alpha_samples = np.linspace(-2, 4, 25)
    q_nominal = 5e4  # Nominal dynamic pressure for characterization
    
    LD_map = np.zeros((len(mach_samples), len(alpha_samples)))
    CL_map = np.zeros((len(mach_samples), len(alpha_samples)))
    CD_map = np.zeros((len(mach_samples), len(alpha_samples)))
    
    for i, M in enumerate(mach_samples):
        for j, alpha in enumerate(alpha_samples):
            CL, CD = aero_db.get_coefficients(M, q_nominal, alpha)
            CL_map[i, j] = CL
            CD_map[i, j] = CD
            if CD > 1e-6:
                LD_map[i, j] = CL / CD
            else:
                LD_map[i, j] = 0
    
    # Find optimal alpha schedule based on vehicle characteristics
    optimal_alpha_vs_mach = np.zeros(len(mach_samples))
    max_LD_vs_mach = np.zeros(len(mach_samples))
    
    for i, M in enumerate(mach_samples):
        best_idx = np.argmax(LD_map[i, :])
        optimal_alpha_vs_mach[i] = alpha_samples[best_idx]
        max_LD_vs_mach[i] = LD_map[i, best_idx]
    
    print(f"\nVehicle Aerodynamic Profile:")
    print(f"  Mach Range: {mach_samples[0]:.1f} - {mach_samples[-1]:.1f}")
    print(f"  Max L/D Range: {np.min(max_LD_vs_mach[max_LD_vs_mach > 0]):.2f} - {np.max(max_LD_vs_mach):.2f}")
    print(f"  Optimal α Range: {np.min(optimal_alpha_vs_mach):.2f}° - {np.max(optimal_alpha_vs_mach):.2f}°")
    
    # Characterize vehicle glide behavior
    avg_LD = np.mean(max_LD_vs_mach[max_LD_vs_mach > 0])
    
    print(f"  Average L/D: {avg_LD:.2f}")
    
    # Create interpolators for vehicle characteristics
    from scipy.interpolate import interp1d
    alpha_maxLD_interp = interp1d(mach_samples, optimal_alpha_vs_mach, 
                                   kind='cubic', bounds_error=False, 
                                   fill_value=(optimal_alpha_vs_mach[0], optimal_alpha_vs_mach[-1]))
    
    max_LD_interp = interp1d(mach_samples, max_LD_vs_mach,
                             kind='cubic', bounds_error=False,
                             fill_value=(max_LD_vs_mach[0], max_LD_vs_mach[-1]))
    
    # ========== DEPLOYMENT CONDITION OPTIMIZATION ==========
    print("\n[3/5] OPTIMIZING DEPLOYMENT CONDITIONS")
    print("-"*50)
    
    # Define parameter ranges - tuned for hypersonic gliders
    altitude_range = [7500, 8000, 8500, 9000]  # meters
    gamma_range = [-12, -10, -8, -6, -4, -2, -1, 0]  # degrees
    mach_fixed = 8.0
    
    deployment_results = []
    
    print(f"Testing {len(altitude_range)} × {len(gamma_range)} = {len(altitude_range)*len(gamma_range)} deployment conditions")
    print(f"Scoring: Range + Terminal Energy State\n")
    
    # Preferred terminal velocity range (for scoring, not constraint)
    preferred_mach_terminal = 2.0
    
    # Strict ceiling limit
    ceiling_limit = 9000  # meters (hard limit)
    ceiling_margin = 50   # meters (safety margin)
    
    # Test all combinations with vehicle-characteristic control
    for h0 in altitude_range:
        for gamma0_deg in gamma_range:
            print(f"h={h0/1000:.1f}km, γ={gamma0_deg:+.1f}° ... ", end="")
            
            rho0, a0, T0 = Atmosphere.atmo_model(h0)
            v0 = mach_fixed * a0
            gamma0 = np.deg2rad(gamma0_deg)
            initial_state = np.array([0.0, h0, v0, gamma0, vehicle.mass_ogive_kg])
            
            # Vehicle-characteristic alpha profile for glide-dive
            def alpha_profile_glide_dive(t, state):
                """
                Glide-dive profile with strict ceiling management:
                - Initial phase: manage deployment conditions carefully
                - Cruise phase: fly at or below max L/D to maximize range
                - Strict ceiling enforcement
                - Terminal dive: gradual transition to efficient landing
                """
                h = state[1]
                v = state[2]
                gamma = state[3]
                
                rho, a, T = Atmosphere.atmo_model(h)
                M = v / max(a, 1.0)
                
                # Get vehicle's max L/D alpha for this Mach
                alpha_maxLD = float(alpha_maxLD_interp(M))
                
                # === TERMINAL DIVE PHASE ===
                if h < 2000:
                    # Low altitude - transition to landing dive
                    # Use speed to modulate alpha for reasonable terminal velocity
                    if v > 1200:  # Very fast
                        return max(-1.0, alpha_maxLD * 0.2)
                    elif v > 800:  # Fast
                        return alpha_maxLD * 0.4
                    elif v > 500:  # Moderate
                        return alpha_maxLD * 0.6
                    else:  # Slow
                        return alpha_maxLD * 0.3
                
                # === INITIAL PHASE (first few seconds after deployment) ===
                if t < 10:  # First 10 seconds
                    # Handle initial flight path angle carefully
                    if gamma0_deg < -8:
                        # Very steep initial descent - use minimal alpha
                        alpha = alpha_maxLD * 0.3
                    elif gamma0_deg < -4:
                        # Moderate descent - use reduced alpha
                        alpha = alpha_maxLD * 0.6
                    elif gamma0_deg < 0:
                        # Slight descent - can use more alpha
                        alpha = alpha_maxLD * 0.8
                    else:
                        # Level or climbing - use conservative alpha
                        alpha = alpha_maxLD * 0.7
                    
                    # Override if too close to ceiling at start
                    if h > ceiling_limit - 200:
                        alpha = min(alpha, 0.0)
                    
                    return np.clip(alpha, -2.0, 4.0)
                
                # === CRUISE PHASE ===
                # Base strategy: fly at max L/D or below (never above)
                alpha = alpha_maxLD
                
                # STRICT ceiling management with multiple layers
                altitude_margin = ceiling_limit - h
                
                if altitude_margin < ceiling_margin:
                    # Emergency: at ceiling limit
                    alpha = -1.5  # Strong negative alpha to force descent
                elif altitude_margin < 100:
                    # Critical: very close to ceiling
                    alpha = -0.5  # Negative alpha
                elif altitude_margin < 200:
                    # Warning: approaching ceiling fast
                    alpha = min(alpha, 0.0)  # Zero alpha (ballistic)
                elif altitude_margin < 300:
                    # Caution: getting close
                    alpha = min(alpha, alpha_maxLD * 0.2)
                elif altitude_margin < 500:
                    # Moderate: reduce lift
                    alpha = min(alpha, alpha_maxLD * 0.5)
                elif altitude_margin < 700:
                    # Watch: slight reduction
                    alpha = min(alpha, alpha_maxLD * 0.7)
                
                # Additional check: if climbing near ceiling, force descent
                if h > ceiling_limit - 400 and gamma > np.deg2rad(0.5):
                    alpha = -0.5  # Negative lift to stop climb
                elif h > ceiling_limit - 500 and gamma > np.deg2rad(1.5):
                    alpha = 0.0
                
                # Prevent excessive climb anywhere
                gamma_deg = np.rad2deg(gamma)
                if gamma_deg > 5:
                    alpha = min(alpha, -0.5)
                elif gamma_deg > 3:
                    alpha = min(alpha, 0.0)
                elif gamma_deg > 2:
                    alpha = min(alpha, alpha_maxLD * 0.4)
                elif gamma_deg > 1:
                    alpha = min(alpha, alpha_maxLD * 0.7)
                
                # If in steep descent away from ceiling, allow full max L/D
                if gamma_deg < -3 and altitude_margin > 500:
                    alpha = alpha_maxLD
                
                # Smooth transition to dive as altitude decreases
                if h < 4000 and h > 2000:
                    # Transition zone
                    transition_factor = (h - 2000) / 2000  # 1.0 at 4km, 0.0 at 2km
                    dive_alpha = alpha_maxLD * 0.5
                    alpha = alpha * transition_factor + dive_alpha * (1 - transition_factor)
                
                return np.clip(alpha, -2.0, 4.0)
            
            try:
                result = integrator.glide_phase_with_alpha_profile(
                    initial_state, alpha_profile_glide_dive, max_time=3600
                )
                
                if result:
                    range_km = result['final_range'] / 1000
                    max_q = result['max_q_dot']
                    max_alt = np.max(result['states'][1, :])
                    
                    # Check terminal state
                    v_final = result['states'][2, -1]
                    h_final = result['states'][1, -1]
                    rho_f, a_f, _ = Atmosphere.atmo_model(max(h_final, 100))
                    mach_final = v_final / a_f
                    
                    # Only hard constraints are ceiling and heat flux
                    ceiling_ok = max_alt <= (ceiling_limit + ceiling_margin)
                    q_dot_ok = max_q <= 1.5e6
                    
                    # Compute score based on range and terminal energy state
                    score = range_km
                    
                    # Penalty for bad terminal Mach (prefer around 2.0, but not a constraint)
                    mach_deviation = abs(mach_final - preferred_mach_terminal)
                    if mach_deviation > 2.0:
                        # Large penalty for very bad terminal speed
                        score -= 5.0 * mach_deviation
                    elif mach_deviation > 1.0:
                        # Moderate penalty for somewhat off target
                        score -= 2.0 * mach_deviation
                    else:
                        # Small penalty for minor deviations
                        score -= 0.5 * mach_deviation
                    
                    # Penalty for ceiling violation
                    if not ceiling_ok:
                        ceiling_violation = (max_alt - ceiling_limit - ceiling_margin) / 1000
                        score -= 50.0 * ceiling_violation  # Severe penalty
                    
                    # Penalty for heat flux violation
                    if not q_dot_ok:
                        q_violation = (max_q - 1.5e6) / 1e6
                        score -= 20.0 * q_violation  # Severe penalty
                    
                    if ceiling_ok and q_dot_ok:
                        print(f"✓ {range_km:.1f}km, M={mach_final:.1f}, q̇={max_q/1e6:.2f}MW/m², score={score:.1f}")
                        deployment_results.append({
                            'h0': h0, 'gamma0_deg': gamma0_deg,
                            'range_km': range_km, 'max_q_dot': max_q,
                            'max_altitude': max_alt, 'result': result,
                            'mach_final': mach_final,
                            'score': score,
                            'success': True
                        })
                    else:
                        reasons = []
                        if not ceiling_ok: reasons.append("ceiling")
                        if not q_dot_ok: reasons.append("heat")
                        print(f"✗ {','.join(reasons)}, M={mach_final:.1f}")
                        deployment_results.append({
                            'success': False,
                            'score': score,
                            'h0': h0,
                            'gamma0_deg': gamma0_deg
                        })
                else:
                    print("✗ failed")
                    deployment_results.append({'success': False, 'score': -1e9})
            except Exception as e:
                print(f"✗ error")
                deployment_results.append({'success': False, 'score': -1e9})
    
    # Select best deployment (by score, including failures)
    if deployment_results:
        best = max(deployment_results, key=lambda x: x.get('score', -1e9))
        
        if best.get('success', False):
            print(f"\n★ OPTIMAL DEPLOYMENT: h={best['h0']/1000:.1f}km, γ={best['gamma0_deg']:.1f}°")
            print(f"  Range: {best['range_km']:.1f} km")
            print(f"  Terminal Mach: {best['mach_final']:.1f}")
            print(f"  Max Altitude: {best['max_altitude']/1000:.3f} km")
            print(f"  Score: {best['score']:.1f}")
        else:
            print("\n✗ ERROR: No fully successful trajectories!")
            print("  Attempting to use best partial result...")
            
            # Try to find any result that at least completed
            completed = [r for r in deployment_results if r.get('result') is not None]
            if not completed:
                print("✗✗ FATAL: No trajectories completed!")
                return 0, 1e9
            
            best = max(completed, key=lambda x: x.get('score', -1e9))
            print(f"  Using: h={best.get('h0', 0)/1000:.1f}km, γ={best.get('gamma0_deg', 0):.1f}°")
            print(f"  Score: {best.get('score', -1e9):.1f}")
    else:
        print("\n✗✗ FATAL: No results at all!")
        return 0, 1e9
    
    # ========== CONTROL STRATEGY REFINEMENT ==========
    print("\n[4/5] REFINING VEHICLE-SPECIFIC GLIDE-DIVE STRATEGY")
    print("-"*50)
    
    rho0, a0, T0 = Atmosphere.atmo_model(best['h0'])
    v0 = mach_fixed * a0
    gamma0 = np.deg2rad(best['gamma0_deg'])
    initial_state = np.array([0.0, best['h0'], v0, gamma0, vehicle.mass_ogive_kg])
    
    # Define multiple control strategies
    strategies = []
    
    # Strategy 1: Conservative (lower alpha throughout)
    def strategy_conservative(t, state):
        h, v, gamma = state[1], state[2], state[3]
        rho, a, _ = Atmosphere.atmo_model(h)
        M = v / max(a, 1.0)
        alpha_maxLD = float(alpha_maxLD_interp(M))
        
        # Terminal
        if h < 2000:
            if v > 1200:
                return alpha_maxLD * 0.1
            elif v > 800:
                return alpha_maxLD * 0.3
            else:
                return alpha_maxLD * 0.5
        
        # Initial phase
        if t < 10:
            if gamma0_deg < -8:
                return alpha_maxLD * 0.2
            elif gamma0_deg < -4:
                return alpha_maxLD * 0.5
            else:
                return alpha_maxLD * 0.6
        
        # Cruise - conservative (70% of max L/D)
        alpha = alpha_maxLD * 0.7
        
        # Ceiling management
        altitude_margin = ceiling_limit - h
        if altitude_margin < 50:
            return -1.5
        elif altitude_margin < 100:
            return -0.5
        elif altitude_margin < 200:
            return 0.0
        elif altitude_margin < 400:
            alpha = min(alpha, alpha_maxLD * 0.3)
        
        if gamma > np.deg2rad(2):
            alpha = min(alpha, 0.0)
        
        return np.clip(alpha, -2.0, 4.0)
    
    strategies.append(("Conservative-70%", strategy_conservative))
    
    # Strategy 2: Aggressive (closer to max L/D)
    def strategy_aggressive(t, state):
        h, v, gamma = state[1], state[2], state[3]
        rho, a, _ = Atmosphere.atmo_model(h)
        M = v / max(a, 1.0)
        alpha_maxLD = float(alpha_maxLD_interp(M))
        
        # Terminal
        if h < 2000:
            if v > 1200:
                return alpha_maxLD * 0.3
            else:
                return alpha_maxLD * 0.5
        
        # Initial
        if t < 10:
            if gamma0_deg < -8:
                return alpha_maxLD * 0.4
            else:
                return alpha_maxLD * 0.8
        
        # Cruise - aggressive (95% of max L/D)
        alpha = alpha_maxLD * 0.95
        
        # Ceiling
        altitude_margin = ceiling_limit - h
        if altitude_margin < 50:
            return -1.5
        elif altitude_margin < 150:
            return 0.0
        elif altitude_margin < 300:
            alpha = min(alpha, alpha_maxLD * 0.5)
        
        if gamma > np.deg2rad(2.5):
            alpha = min(alpha, alpha_maxLD * 0.3)
        
        return np.clip(alpha, -2.0, 4.0)
    
    strategies.append(("Aggressive-95%", strategy_aggressive))
    
    # Strategy 3: Use the deployment-optimized one
    strategies.append(("Deployment-Optimized", None))  # Will use best['result']
    
    # Test strategies
    print(f"\nTesting {len(strategies)} control strategies:\n")
    
    results = []
    for name, strategy in strategies:
        if strategy is None:
            # Use the deployment result
            result_data = best
            result = best['result']
        else:
            result = integrator.glide_phase_with_alpha_profile(initial_state, strategy)
            if result:
                range_km = result['final_range'] / 1000
                max_q = result['max_q_dot']
                max_alt = np.max(result['states'][1, :])
                
                v_final = result['states'][2, -1]
                h_final = result['states'][1, -1]
                rho_f, a_f, _ = Atmosphere.atmo_model(max(h_final, 100))
                mach_final = v_final / a_f
                
                # Compute score
                score = range_km
                mach_deviation = abs(mach_final - preferred_mach_terminal)
                if mach_deviation > 2.0:
                    score -= 5.0 * mach_deviation
                elif mach_deviation > 1.0:
                    score -= 2.0 * mach_deviation
                else:
                    score -= 0.5 * mach_deviation
                
                if max_alt > ceiling_limit + ceiling_margin:
                    score -= 50.0 * (max_alt - ceiling_limit - ceiling_margin) / 1000
                if max_q > 1.5e6:
                    score -= 20.0 * (max_q - 1.5e6) / 1e6
                
                result_data = {
                    'range_km': range_km,
                    'max_q_dot': max_q,
                    'max_altitude': max_alt,
                    'mach_final': mach_final,
                    'score': score,
                    'result': result
                }
            else:
                print(f"{name:25s}: ✗ failed")
                continue
        
        ceiling_ok = "✓" if result_data['max_altitude'] <= ceiling_limit + ceiling_margin else "✗"
        q_ok = "✓" if result_data['max_q_dot'] <= 1.5e6 else "✗"
        
        print(f"{name:25s}: {result_data['range_km']:6.1f}km, M={result_data['mach_final']:.1f}, "
              f"alt{ceiling_ok}, q̇{q_ok}, score={result_data['score']:6.1f}")
        
        results.append({
            'name': name,
            'result': result_data['result'],
            'score': result_data['score'],
            'range_km': result_data['range_km'],
            'max_q_dot': result_data['max_q_dot'],
            'max_altitude': result_data['max_altitude'],
            'mach_final': result_data['mach_final']
        })
    
    if not results:
        print("\n✗ ERROR: All strategies failed!")
        return 0, 1e9
    
    # Select best by score
    best_result = max(results, key=lambda x: x['score'])
    
    print(f"\n★ SELECTED: {best_result['name']}")
    
    # ========== FINAL RESULTS ==========
    print("\n[5/5] FINAL RESULTS")
    print("="*70)
    print(f"Vehicle Type: {vehicle.__class__.__name__}")
    print(f"Aerodynamic Signature:")
    print(f"  - Average L/D: {avg_LD:.2f}")
    print(f"  - Optimal α Range: {np.min(optimal_alpha_vs_mach):.1f}° to {np.max(optimal_alpha_vs_mach):.1f}°")
    print(f"\nOptimal Deployment:")
    print(f"  - Altitude: {best['h0']/1000:.1f} km")
    print(f"  - Flight Path Angle: {best['gamma0_deg']:.1f}°")
    print(f"  - Mach: {mach_fixed:.1f}")
    print(f"\nControl Strategy: {best_result['name']}")
    print(f"  (Glide at/below max L/D with strict ceiling)")
    print(f"\nPerformance:")
    print(f"  - Range: {best_result['range_km']:.1f} km")
    print(f"  - Terminal Mach: {best_result['mach_final']:.1f}")
    print(f"  - Max Heat Flux: {best_result['max_q_dot']/1e6:.2f} MW/m²")
    print(f"  - Max Altitude: {best_result['max_altitude']/1000:.3f} km (limit: {ceiling_limit/1000:.1f} km)")
    print(f"  - Score: {best_result['score']:.1f}")
    
    max_range = best_result['result']['final_range']
    max_q_dot = best_result['max_q_dot']
    
    if plotting:
        plot_trajectory_results(best_result['result'], path, 
                               f"{best_result['name']}")
    
    print("="*70)
    
    return max_range, max_q_dot

def plot_trajectory_results(result, path, strategy_name):
    """Plot and save trajectory results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Glide Trajectory Results ({strategy_name})', fontsize=14)
    
    time = result['time']
    states = result['states']
    x = states[0, :] / 1000  # km
    h = states[1, :] / 1000  # km
    v = states[2, :]  # m/s
    gamma = np.rad2deg(states[3, :])  # deg
    
    # Calculate Mach history
    mach = []
    for i in range(len(time)):
        rho, a, T = Atmosphere.atmo_model(states[1, i])
        mach.append(states[2, i] / max(a, 1.0))
    
    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(x, h, 'b-', linewidth=2)
    ax.axhline(y=9, color='r', linestyle='--', alpha=0.5, label='9 km ceiling')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory Profile')
    ax.grid(True)
    ax.legend()
    
    # 2. Altitude vs Time
    ax = axes[0, 1]
    ax.plot(time, h, 'g-', linewidth=2)
    ax.axhline(y=9, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude vs Time')
    ax.grid(True)
    
    # 3. Velocity vs Time
    ax = axes[0, 2]
    ax.plot(time, v, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity vs Time')
    ax.grid(True)
    
    # 4. Mach vs Time
    ax = axes[1, 0]
    ax.plot(time, mach, 'c-', linewidth=2)
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Mach 2')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach Number')
    ax.set_title('Mach Number vs Time')
    ax.grid(True)
    ax.legend()
    
    # 5. Heat Flux vs Time
    ax = axes[1, 1]
    if 'q_dot_history' in result:
        ax.plot(result['time_history'][:len(result['q_dot_history'])], 
                np.array(result['q_dot_history'])/1000,
                'm-', linewidth=2)
        ax.axhline(y=1200, color='r', linestyle='--', alpha=0.5, label='1.2 MW/m² limit')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heat Flux (kW/m²)')
        ax.set_title('Heat Flux vs Time')
        ax.grid(True)
        ax.legend()
    
    # 6. Flight Path Angle vs Time
    ax = axes[1, 2]
    ax.plot(time, gamma, 'k-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flight Path Angle (deg)')
    ax.set_title('Flight Path Angle vs Time')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(path, 'glide_flight_trajectory_results.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Test run
    max_range, max_q_dot = run_dymos_optimization('.', plotting=True)
    print(f"\nFinal: Range = {max_range/1000:.1f} km, Q_dot = {max_q_dot:.0f} W/m²")
