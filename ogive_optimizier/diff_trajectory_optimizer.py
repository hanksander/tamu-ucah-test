"""
trajectory_optimizer.py (Enhanced with adaptive q_dot limit and optimized initial_mach)
Key changes:
- Added q_dot_limit parameter to all relevant functions
- Added initial_mach as an optimization variable (6-8) in the search vector
- Propagates both parameters from run_dymos_optimization through to evaluation
- Added comprehensive 3x3 plot grid with key trajectory metrics
- Added .traj file output with full time series data
"""
import numpy as np
import random
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import warnings
import os
import matplotlib.pyplot as plt

# Local imports
try:
    from simple_trajectory import Vehicle, Atmosphere, TrajectoryIntegrator, plot_trajectory_results
except Exception:
    Vehicle = None
    Atmosphere = None
    TrajectoryIntegrator = None
    plot_trajectory_results = None

def build_alpha_profile_from_cps(control_times, control_alphas_deg,
                                 alpha_min=-5.0, alpha_max=6.0, kind='linear',
                                 t_max_expected=300.0):
    """Create an alpha(t, state) callable from control points.
    
    Args:
        control_times: Control point times in normalized [0, 1]
        control_alphas_deg: Alpha values at control points (degrees)
        alpha_min, alpha_max: Bounds for alpha
        kind: Interpolation kind ('linear' or 'cubic')
        t_max_expected: Expected maximum trajectory time (seconds)
    """
    control_times = np.asarray(control_times, dtype=float)
    control_alphas_deg = np.asarray(control_alphas_deg, dtype=float)
    
    if np.any(np.diff(control_times) < 0):
        idx = np.argsort(control_times)
        control_times = control_times[idx]
        control_alphas_deg = control_alphas_deg[idx]
    
    if kind == 'cubic' and len(control_times) >= 4:
        try:
            from scipy.interpolate import CubicSpline
        except Exception:
            kind = 'linear'
    
    if kind == 'cubic' and len(control_times) >= 4:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(control_times, control_alphas_deg, extrapolate=True)
        
        def alpha_profile(t, state):
            try:
                tval = float(t)
            except Exception:
                tval = 0.0
            
            # Normalize time based on expected trajectory duration
            t_norm = tval / t_max_expected
            t_norm = np.clip(t_norm, 0.0, 1.0)
            
            a = cs(t_norm)
            return float(np.clip(a, alpha_min, alpha_max))
    else:
        f = interp1d(control_times, control_alphas_deg, kind='linear',
                     bounds_error=False,
                     fill_value=(control_alphas_deg[0], control_alphas_deg[-1]))
        
        def alpha_profile(t, state):
            try:
                tval = float(t)
            except Exception:
                tval = 0.0
            
            # Normalize time based on expected trajectory duration
            t_norm = tval / t_max_expected
            t_norm = np.clip(t_norm, 0.0, 1.0)
            
            a = float(f(t_norm))
            return float(np.clip(a, alpha_min, alpha_max))
    
    return alpha_profile

def evaluate_trajectory_candidate(x,
                                  integrator,
                                  vehicle,
                                  target_mach=2.0,
                                  q_dot_limit=1.2e6,
                                  ceiling_limit=9050,
                                  alpha_time_grid=None,
                                  alpha_bounds=(-2.0, 4.0),
                                  penalty_weights=None,
                                  max_time=3600.0,
                                  return_result=False):
    """Evaluate a candidate defined by vector x.
    x layout: [h0 (m), gamma0_deg, initial_mach, alpha_cp1, ..., alpha_cpN]
    Returns scalar cost (lower is better) or full result if return_result=True.
    
    Note: initial_mach is now the 3rd element in the optimization vector (index 2)
    """
    if penalty_weights is None:
        penalty_weights = {'heat': 5e3, 'alt': 1e5, 'mach': 1e4, 'smooth': 1e-2}
    x = np.asarray(x, dtype=float)
    if x.size < 4:  # Need at least h0, gamma0, mach0, and one alpha_cp
        return 1e12 if not return_result else None
    
    h0 = float(x[0])
    gamma0_deg = float(x[1])
    initial_mach = float(x[2])  # Extract initial Mach from optimization vector
    alpha_cps = x[3:].astype(float)  # Alpha control points start at index 3
    
    N = alpha_cps.size
    if alpha_time_grid is None:
        alpha_time_grid = np.linspace(0.0, 1.0, N)
    alpha_profile = build_alpha_profile_from_cps(alpha_time_grid, alpha_cps,
                                                 alpha_min=alpha_bounds[0],
                                                 alpha_max=alpha_bounds[1],
                                                 kind='quadratic')
    try:
        rho0, a0, _ = Atmosphere.atmo_model(h0)
    except Exception:
        return 1e12 if not return_result else None
    
    # Use optimized initial Mach number
    v0 = initial_mach * a0
    x0 = 0.0
    gamma0 = np.deg2rad(gamma0_deg)
    m0 = vehicle.mass_ogive_kg
    initial_state = np.array([x0, h0, v0, gamma0, m0])
    
    try:
        result = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile, max_time=max_time)
        if result is None:
            return 1e12 if not return_result else None
    except Exception:
        return 1e12 if not return_result else None
    if return_result:
        return result
    final_state = result['states'][:, -1]
    v_final = final_state[2]
    h_final = final_state[1]
    try:
        _, a_final, _ = Atmosphere.atmo_model(max(0.0, h_final))
    except Exception:
        return 1e12
    mach_final = float(v_final / max(a_final, 1.0))
    max_q_dot = float(result.get('max_q_dot', 1e12))
    max_alt = float(np.max(result['states'][1, :]))
    range_m = float(result.get('final_range', 0.0))
    # Cost: negative range + penalties
    cost = -range_m
    if max_q_dot > q_dot_limit:
        cost += penalty_weights['heat'] * (max_q_dot - q_dot_limit)
    if max_alt > ceiling_limit:
        cost += penalty_weights['alt'] * (max_alt - ceiling_limit)
    cost += penalty_weights['mach'] * abs(mach_final - target_mach)
    if N >= 3:
        diffs = np.diff(alpha_cps)
        cost += penalty_weights['smooth'] * np.sum(diffs**2)
    if h_final < 0:
        cost += 1e6
    return float(cost)

def _de_objective_wrapper(x, integrator, vehicle, target_mach, q_dot_limit, 
                          ceiling_limit, alpha_time_grid, alpha_bounds):
    """Wrapper function to pass to differential_evolution."""
    return evaluate_trajectory_candidate(x, integrator, vehicle,
                                         target_mach=target_mach,
                                         q_dot_limit=q_dot_limit,
                                         ceiling_limit=ceiling_limit,
                                         alpha_time_grid=alpha_time_grid,
                                         alpha_bounds=alpha_bounds)

def optimize_with_shooting(integrator,
                           vehicle,
                           optimizer='de',
                           n_alpha_cp=7,
                           bounds_h=(7000, 9000),
                           bounds_gamma_deg=(-20, 5),
                           bounds_mach=(6.0, 8.0),
                           alpha_bounds=(-2.0, 6.0),
                           popsize=12,
                           maxiter=80,
                           seed=None,
                           workers=1,
                           q_dot_limit=1.2e6):
    """Global search over [h0, gamma0_deg, initial_mach, alpha_cp...] with multithreading support.
    
    Args:
        bounds_h: Altitude bounds in meters
        bounds_gamma_deg: Flight path angle bounds in degrees
        bounds_mach: Initial Mach number bounds (default 6.0-8.0)
        q_dot_limit: Heat flux limit in W/m² (default 1.2e6 = 1.2 MW/m²)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    bounds = []
    bounds.append(bounds_h)            # Index 0: h0
    bounds.append(bounds_gamma_deg)    # Index 1: gamma0_deg
    bounds.append(bounds_mach)         # Index 2: initial_mach (NEW)
    for _ in range(n_alpha_cp):
        bounds.append(alpha_bounds)    # Indices 3+: alpha control points
    
    obj_args = (integrator, vehicle, 
                2.0,                                      # target_mach
                q_dot_limit,                              # q_dot_limit
                9050,                                     # ceiling_limit
                np.linspace(0, 1, n_alpha_cp),            # alpha_time_grid
                alpha_bounds)
    
    print(f"[trajectory_optimizer] Running DE (dims={len(bounds)}) popsize={popsize} maxiter={maxiter} workers={workers}")
    print(f"[trajectory_optimizer] Q_dot limit: {q_dot_limit/1e6:.2f} MW/m²")
    print(f"[trajectory_optimizer] Initial Mach bounds: [{bounds_mach[0]:.1f}, {bounds_mach[1]:.1f}]")
    
    result = differential_evolution(_de_objective_wrapper, bounds, 
                                    args=obj_args,
                                    strategy='best1bin',
                                    maxiter=maxiter, popsize=popsize, tol=1e-3,
                                    polish=True, seed=seed,
                                    updating='deferred', workers=workers)
    best_x = result.x
    best_cost = result.fun
    print(f"[trajectory_optimizer] DE finished: best_cost={best_cost:.3e}")
    print(f"[trajectory_optimizer] Optimized initial Mach: {best_x[2]:.2f}")
    return {'x': best_x, 'cost': best_cost, 'result_obj': result}

def plot_comprehensive_results(trajectory, path, strategy_name, q_dot_limit=1.2e6):
    """Generate comprehensive 3x3 plot grid with key trajectory metrics."""
    
    # Extract data from trajectory
    t = trajectory.get('time', np.array([]))
    states = trajectory.get('states', np.zeros((5, 0)))
    alphas = trajectory.get('alphas', np.array([]))
    L_hist = trajectory.get('L', np.array([]))
    D_hist = trajectory.get('D', np.array([]))
    q_dot_hist = trajectory.get('q_dot', np.array([]))
    
    if len(t) == 0 or states.shape[1] == 0:
        print("  ⚠️  Warning: No trajectory data available for plots")
        return
    
    # Extract state variables
    x = states[0, :] / 1000.0  # downrange in km
    h = states[1, :] / 1000.0  # altitude in km
    v = states[2, :]           # velocity in m/s
    gamma = np.rad2deg(states[3, :])  # flight path angle in deg
    
    # Calculate derived quantities
    mach = np.zeros_like(v)
    rho = np.zeros_like(v)
    for i, (hi, vi) in enumerate(zip(h * 1000, v)):
        try:
            rho_i, a_i, _ = Atmosphere.atmo_model(max(0.0, hi))
            mach[i] = vi / max(a_i, 1.0)
            rho[i] = rho_i
        except:
            mach[i] = 0.0
            rho[i] = 0.0
    
    # Calculate L/D ratio
    LD_ratio = np.zeros_like(L_hist)
    for i in range(len(L_hist)):
        if abs(D_hist[i]) > 1e-6:
            LD_ratio[i] = L_hist[i] / D_hist[i]
        else:
            LD_ratio[i] = 0.0
    
    # Dynamic pressure in kPa
    q_dyn = 0.5 * rho * v**2 / 1000.0
    
    # Heat flux in MW/m²
    q_dot_MW = q_dot_hist / 1e6
    
    # Create 3x3 subplot figure
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'Comprehensive Trajectory Analysis\n{strategy_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Altitude vs Downrange
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x, h, 'b-', linewidth=2)
    ax1.set_xlabel('Downrange (km)', fontsize=10)
    ax1.set_ylabel('Altitude (km)', fontsize=10)
    ax1.set_title('Altitude Profile', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=9.05, color='r', linestyle='--', alpha=0.5, label='Ceiling')
    ax1.legend(fontsize=8)
    
    # Plot 2: Velocity vs Time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, v, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Velocity (m/s)', fontsize=10)
    ax2.set_title('Velocity History', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mach vs Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, mach, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Mach Number', fontsize=10)
    ax3.set_title('Mach Number History', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Alpha vs Time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t, alphas, 'b-', linewidth=2)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Angle of Attack (deg)', fontsize=10)
    ax4.set_title('Angle of Attack vs Time', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 5: L/D vs Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t, LD_ratio, 'g-', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('L/D Ratio', fontsize=10)
    ax5.set_title('Lift-to-Drag Ratio vs Time', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 6: Alpha vs Mach
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(mach, alphas, 'r-', linewidth=2)
    ax6.set_xlabel('Mach Number', fontsize=10)
    ax6.set_ylabel('Angle of Attack (deg)', fontsize=10)
    ax6.set_title('Angle of Attack vs Mach', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 7: Heat Flux vs Time
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(t, q_dot_MW, 'r-', linewidth=2, label='Heat Flux')
    ax7.axhline(y=q_dot_limit/1e6, color='orange', linestyle='--', linewidth=2, 
                label=f'Limit ({q_dot_limit/1e6:.2f} MW/m²)')
    ax7.set_xlabel('Time (s)', fontsize=10)
    ax7.set_ylabel('Heat Flux (MW/m²)', fontsize=10)
    ax7.set_title('Heat Flux History', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)
    
    # Plot 8: Dynamic Pressure vs Time
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(t, q_dyn, 'purple', linewidth=2)
    ax8.set_xlabel('Time (s)', fontsize=10)
    ax8.set_ylabel('Dynamic Pressure (kPa)', fontsize=10)
    ax8.set_title('Dynamic Pressure History', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Flight Path Angle vs Time
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(t, gamma, 'brown', linewidth=2)
    ax9.set_xlabel('Time (s)', fontsize=10)
    ax9.set_ylabel('Flight Path Angle (deg)', fontsize=10)
    ax9.set_title('Flight Path Angle History', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    filename = os.path.join(path, 'comprehensive_trajectory_analysis.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  📊 Comprehensive analysis saved: {filename}")
    plt.close()

def write_trajectory_file(trajectory, path, strategy_name, vehicle):
    """Write trajectory data to a .traj file with full time series.
    
    Format: t, v, M, downrange, altitude, rho_atm, L, D, q_dot, alpha, gamma, ...
    """
    
    # Extract data from trajectory
    t = trajectory.get('time', np.array([]))
    states = trajectory.get('states', np.zeros((5, 0)))
    alphas = trajectory.get('alphas', np.array([]))
    L_hist = trajectory.get('L', np.array([]))
    D_hist = trajectory.get('D', np.array([]))
    q_dot_hist = trajectory.get('q_dot', np.array([]))
    
    if len(t) == 0 or states.shape[1] == 0:
        print("  ⚠️  Warning: No trajectory data available for .traj file")
        return
    
    # Extract state variables
    x = states[0, :]  # downrange
    h = states[1, :]  # altitude
    v = states[2, :]  # velocity
    gamma = states[3, :]  # flight path angle
    m = states[4, :]  # mass
    
    # Calculate Mach number and atmospheric density
    mach = np.zeros_like(v)
    rho = np.zeros_like(v)
    for i, (hi, vi) in enumerate(zip(h, v)):
        try:
            rho_i, a_i, _ = Atmosphere.atmo_model(max(0.0, hi))
            mach[i] = vi / max(a_i, 1.0)
            rho[i] = rho_i
        except:
            mach[i] = 0.0
            rho[i] = 0.0
    
    # Calculate L/D ratio
    LD_ratio = np.zeros_like(L_hist)
    for i in range(len(L_hist)):
        if abs(D_hist[i]) > 1e-6:
            LD_ratio[i] = L_hist[i] / D_hist[i]
        else:
            LD_ratio[i] = 0.0
    
    # Calculate dynamic pressure
    q_dyn = 0.5 * rho * v**2
    
    # Write to file
    filename = os.path.join(path, 'best_trajectory.traj')
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"# Optimized Trajectory Data\n")
        f.write(f"# Strategy: {strategy_name}\n")
        f.write(f"# Vehicle Mass: {vehicle.mass_ogive_kg:.2f} kg\n")
        f.write(f"# Reference Area: {vehicle.S_ref:.4f} m²\n")
        f.write(f"# Generated by trajectory_optimizer.py\n")
        f.write("#\n")
        f.write("# Column definitions:\n")
        f.write("#  1: time (s)\n")
        f.write("#  2: velocity (m/s)\n")
        f.write("#  3: Mach number\n")
        f.write("#  4: downrange (m)\n")
        f.write("#  5: altitude (m)\n")
        f.write("#  6: atmospheric density (kg/m³)\n")
        f.write("#  7: lift force (N)\n")
        f.write("#  8: drag force (N)\n")
        f.write("#  9: heat flux (W/m²)\n")
        f.write("# 10: angle of attack (deg)\n")
        f.write("# 11: flight path angle (deg)\n")
        f.write("# 12: L/D ratio\n")
        f.write("# 13: dynamic pressure (Pa)\n")
        f.write("# 14: mass (kg)\n")
        f.write("#\n")
        
        # Write column headers
        f.write(f"{'time':>12s} {'velocity':>12s} {'Mach':>12s} {'downrange':>12s} "
                f"{'altitude':>12s} {'rho_atm':>12s} {'L':>12s} {'D':>12s} "
                f"{'q_dot':>12s} {'alpha':>12s} {'gamma':>12s} {'L/D':>12s} "
                f"{'q_dyn':>12s} {'mass':>12s}\n")
        
        f.write(f"{'(s)':>12s} {'(m/s)':>12s} {'':>12s} {'(m)':>12s} "
                f"{'(m)':>12s} {'(kg/m³)':>12s} {'(N)':>12s} {'(N)':>12s} "
                f"{'(W/m²)':>12s} {'(deg)':>12s} {'(deg)':>12s} {'':>12s} "
                f"{'(Pa)':>12s} {'(kg)':>12s}\n")
        
        # Write data
        for i in range(len(t)):
            f.write(f"{t[i]:12.4f} {v[i]:12.3f} {mach[i]:12.4f} {x[i]:12.1f} "
                    f"{h[i]:12.3f} {rho[i]:12.6e} {L_hist[i]:12.3f} {D_hist[i]:12.3f} "
                    f"{q_dot_hist[i]:12.1f} {alphas[i]:12.4f} {np.rad2deg(gamma[i]):12.4f} "
                    f"{LD_ratio[i]:12.4f} {q_dyn[i]:12.3f} {m[i]:12.3f}\n")
    
    print(f"  💾 Trajectory data saved: {filename}")
    print(f"     - {len(t)} time steps")
    print(f"     - Duration: {t[-1]:.1f} s")
    print(f"     - Final downrange: {x[-1]/1000:.2f} km")

def run_dymos_optimization(path, plotting=True, surrogate_type='linear', 
                          workers=16, q_dot_limit=1.2e6, bounds_mach=(7.5, 8.0)):
    """Compatibility wrapper with plotting integration and multithreading.
    Args:
        path: Output directory for plots
        plotting: Whether to generate plots
        surrogate_type: Type of surrogate model ('linear' or 'gpr')
        workers: Number of parallel workers (1 = serial, -1 = all CPUs)
        q_dot_limit: Heat flux limit in W/m² (default 1.2e6 = 1.2 MW/m²)
        bounds_mach: Tuple of (min_mach, max_mach) for optimization (default (6.0, 8.0))
    Returns: (best_range_m, best_max_qdot)
    """
    try:
        from simple_trajectory import build_global_database
    except Exception as e:
        raise RuntimeError("simple_trajectory.build_global_database not found") from e
    
    print('\n[trajectory_optimizer] Building aerodynamic database...')
    merged, models, scalers, feature_cols = build_global_database('ogive', surrogate_type=surrogate_type)
    vehicle = Vehicle()
    aero_db = None
    try:
        from simple_trajectory import AeroDatabase
        aero_db = AeroDatabase(models, scalers, feature_cols, vehicle.S_ref)
    except Exception:
        raise RuntimeError('AeroDatabase class not found in simple_trajectory')
    integrator = TrajectoryIntegrator(vehicle, aero_db)
    
    print(f'\n[trajectory_optimizer] Running shooting optimizer:')
    print(f'  Workers: {workers}')
    print(f'  Q_dot limit: {q_dot_limit/1e6:.2f} MW/m²')
    print(f'  Initial Mach bounds: [{bounds_mach[0]:.1f}, {bounds_mach[1]:.1f}]')
    
    best = optimize_with_shooting(integrator, vehicle, 
                                  n_alpha_cp=25, 
                                  maxiter=5, 
                                  popsize=5,
                                  workers=workers,
                                  q_dot_limit=q_dot_limit,
                                  bounds_mach=bounds_mach)
    
    # Re-evaluate best candidate to get full trajectory
    best_x = best['x']
    h0 = float(best_x[0])
    gamma0_deg = float(best_x[1])
    initial_mach = float(best_x[2])  # Extract optimized Mach number
    
    n_cp = len(best_x) - 3  # Now we have 3 initial conditions (h, gamma, mach)
    alpha_time_grid = np.linspace(0, 1, n_cp)
    alpha_cps = best_x[3:]  # Alpha CPs start at index 3
    alpha_bounds_used = (-2.0, 6.0)
    alpha_profile = build_alpha_profile_from_cps(alpha_time_grid, alpha_cps, 
                                                 alpha_min=alpha_bounds_used[0], 
                                                 alpha_max=alpha_bounds_used[1])
    
    rho0, a0, _ = Atmosphere.atmo_model(h0)
    v0 = initial_mach * a0  # Use optimized initial Mach
    initial_state = np.array([0.0, h0, v0, np.deg2rad(gamma0_deg), vehicle.mass_ogive_kg])
    
    trajectory = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile, max_time=3600.0)
    if trajectory is None:
        return 0.0, 1e9
    
    best_range = float(trajectory['final_range'])
    best_qdot = float(trajectory['max_q_dot'])
    max_alt = float(np.max(trajectory['states'][1, :]))
    
    print(f"\n[trajectory_optimizer] ★ OPTIMIZED TRAJECTORY RESULTS:")
    print(f"  Deployment: {h0/1000:.1f} km altitude, {gamma0_deg:.1f}° flight path angle, Mach {initial_mach:.2f}")
    print(f"  Maximum Range: {best_range/1000:.1f} km")
    print(f"  Maximum Heat Flux: {best_qdot:.0f} W/m² ({best_qdot/1e6:.2f} MW/m²)")
    print(f"  Heat Flux Limit: {q_dot_limit:.0f} W/m² ({q_dot_limit/1e6:.2f} MW/m²)")
    print(f"  Maximum Altitude: {max_alt/1000:.2f} km")
    if best_qdot > q_dot_limit:
        excess = (best_qdot - q_dot_limit) / 1e6
        print(f"  ⚠️  WARNING: Exceeded heat flux limit by {excess:.2f} MW/m²")
    if max_alt > 9050:
        print(f"  ⚠️  WARNING: Exceeded 9 km ceiling by {(max_alt-9000)/1000:.2f} km")
    
    strategy_name = f"DE Optimized (h0={h0/1000:.1f}km, γ0={gamma0_deg:.1f}°, M0={initial_mach:.2f})"
    
    if plotting:
        # Original plots
        if plot_trajectory_results is not None:
            plot_trajectory_results(trajectory, path, strategy_name)
        else:
            print("  ⚠️  Warning: plot_trajectory_results not available from simple_trajectory")
        
        # Comprehensive 3x3 plot
        plot_comprehensive_results(trajectory, path, strategy_name, q_dot_limit)
        
        # Write trajectory file
        write_trajectory_file(trajectory, path, strategy_name, vehicle)
    
    return best_range, best_qdot

if __name__ == '__main__':
    import multiprocessing
    
    # Test run with custom q_dot limit and Mach bounds
    n_workers = 16
    custom_q_dot_limit = 0.95e6  # 0.95 MW/m²
    custom_mach_bounds = (6.5, 8.0)  # Allow optimizer to find best Mach between 6.5 and 8
    
    print(f"\nRunning optimization with {n_workers} workers...")
    print(f"Using custom q_dot limit: {custom_q_dot_limit/1e6:.2f} MW/m²")
    print(f"Optimizing initial Mach in range: [{custom_mach_bounds[0]:.1f}, {custom_mach_bounds[1]:.1f}]")
    
    max_range, max_q_dot = run_dymos_optimization('.', 
                                                   plotting=True, 
                                                   workers=n_workers, 
                                                   q_dot_limit=custom_q_dot_limit,
                                                   bounds_mach=custom_mach_bounds)
    print(f"\nFinal: Range = {max_range/1000:.1f} km, Q_dot = {max_q_dot:.0f} W/m²")