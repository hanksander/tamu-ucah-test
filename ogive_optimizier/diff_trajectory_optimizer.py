"""
trajectory_optimizer.py (Enhanced with adaptive q_dot limit and optimized initial_mach)
Key changes:
- Added q_dot_limit parameter to all relevant functions
- Added initial_mach as an optimization variable (6-8) in the search vector
- Propagates both parameters from run_dymos_optimization through to evaluation
"""
import numpy as np
import random
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import warnings
import os
# Local imports
try:
    from simple_trajectory import Vehicle, Atmosphere, TrajectoryIntegrator, plot_trajectory_results
except Exception:
    Vehicle = None
    Atmosphere = None
    TrajectoryIntegrator = None
    plot_trajectory_results = None

def build_alpha_profile_from_cps(control_times, control_alphas_deg,
                                 alpha_min=-5.0, alpha_max=6.0, kind='linear'):
    """Create an alpha(t, state) callable from control points."""
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
            t_eval = np.clip(tval, 0.0, 1.0)
            a = cs(t_eval)
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
            t_eval = np.clip(tval, 0.0, 1.0)
            a = float(f(t_eval))
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
        penalty_weights = {'heat': 5e3, 'alt': 1e4, 'mach': 1e4, 'smooth': 1e-2}
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
                                                 kind='linear')
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

def run_dymos_optimization(path, plotting=True, surrogate_type='linear', 
                          workers=16, q_dot_limit=1.2e6, bounds_mach=(6.0, 8.0)):
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
                                  n_alpha_cp=5, 
                                  maxiter=10, 
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
    
    if plotting and plot_trajectory_results is not None:
        strategy_name = f"DE Optimized (h0={h0/1000:.1f}km, γ0={gamma0_deg:.1f}°, M0={initial_mach:.2f})"
        plot_trajectory_results(trajectory, path, strategy_name)
    elif plotting:
        print("  ⚠️  Warning: plot_trajectory_results not available from simple_trajectory")
    
    return best_range, best_qdot

if __name__ == '__main__':
    import multiprocessing
    
    # Test run with custom q_dot limit and Mach bounds
    n_workers = 16
    custom_q_dot_limit = 0.8e6  # 0.8 MW/m²
    custom_mach_bounds = (6.0, 8.0)  # Allow optimizer to find best Mach between 6 and 8
    
    print(f"\nRunning optimization with {n_workers} workers...")
    print(f"Using custom q_dot limit: {custom_q_dot_limit/1e6:.2f} MW/m²")
    print(f"Optimizing initial Mach in range: [{custom_mach_bounds[0]:.1f}, {custom_mach_bounds[1]:.1f}]")
    
    max_range, max_q_dot = run_dymos_optimization('.', 
                                                   plotting=True, 
                                                   workers=n_workers, 
                                                   q_dot_limit=custom_q_dot_limit,
                                                   bounds_mach=custom_mach_bounds)
    print(f"\nFinal: Range = {max_range/1000:.1f} km, Q_dot = {max_q_dot:.0f} W/m²")