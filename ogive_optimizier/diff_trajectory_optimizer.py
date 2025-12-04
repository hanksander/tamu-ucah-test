"""
trajectory_optimizer.py

A separate module implementing a robust shooting-based optimizer for the hypersonic
glide vehicle trajectory. Integrates with simple_trajectory.py plotting and includes
multithreading support.

Usage:
    from trajectory_optimizer import optimize_with_shooting, run_dymos_optimization
    from simple_trajectory import Vehicle, Atmosphere, TrajectoryIntegrator

    best = optimize_with_shooting(integrator, vehicle, n_alpha_cp=7, maxiter=80, workers=4)
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
    """Create an alpha(t, state) callable from control points.

    control_times : 1D array, values in [0,1]
    control_alphas_deg : 1D array of same length, degrees
    kind : 'linear' or 'cubic' (cubic requires >=4 points)

    Returned function signature: alpha_profile(t, state) -> alpha_deg
    """
    control_times = np.asarray(control_times, dtype=float)
    control_alphas_deg = np.asarray(control_alphas_deg, dtype=float)

    # ensure monotonic increasing times
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
        # linear fallback
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

    x layout: [h0 (m), gamma0_deg, alpha_cp1, ..., alpha_cpN]

    Returns scalar cost (lower is better) or full result if return_result=True.
    """
    if penalty_weights is None:
        penalty_weights = {'heat': 1e-3, 'alt': 1e4, 'mach': 1e4, 'smooth': 1e-2}

    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 1e12 if not return_result else None

    h0 = float(x[0])
    gamma0_deg = float(x[1])
    alpha_cps = x[2:].astype(float)
    N = alpha_cps.size

    if alpha_time_grid is None:
        alpha_time_grid = np.linspace(0.0, 1.0, N)

    alpha_profile = build_alpha_profile_from_cps(alpha_time_grid, alpha_cps,
                                                 alpha_min=alpha_bounds[0],
                                                 alpha_max=alpha_bounds[1],
                                                 kind='linear')

    # Keep Mach release fixed at 8.0
    try:
        rho0, a0, _ = Atmosphere.atmo_model(h0)
    except Exception:
        return 1e12 if not return_result else None

    mach_release = 8.0
    v0 = mach_release * a0

    x0 = 0.0
    gamma0 = np.deg2rad(gamma0_deg)
    m0 = vehicle.mass_ogive_kg
    initial_state = np.array([x0, h0, v0, gamma0, m0])

    # Run integration
    try:
        result = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile, max_time=max_time)
        if result is None:
            return 1e12 if not return_result else None
    except Exception:
        return 1e12 if not return_result else None

    # If caller wants full result, return it
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
    """
    Wrapper function to pass to differential_evolution. 
    It takes 'x' (the optimization vector) and fixed arguments.
    """
    return evaluate_trajectory_candidate(x, integrator, vehicle,
                                         target_mach=target_mach,
                                         q_dot_limit=q_dot_limit,
                                         ceiling_limit=ceiling_limit,
                                         alpha_time_grid=alpha_time_grid,
                                         alpha_bounds=alpha_bounds)
# --------------------------------------------------------


def optimize_with_shooting(integrator,
                           vehicle,
                           optimizer='de',
                           n_alpha_cp=7,
                           bounds_h=(7000, 12000),
                           bounds_gamma_deg=(-20, 5),
                           alpha_bounds=(-2.0, 6.0),
                           popsize=12,
                           maxiter=80,
                           seed=None,
                           workers=1):
    """Global search over [h0, gamma0_deg, alpha_cp...] with multithreading support."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    bounds = []
    bounds.append(bounds_h)
    bounds.append(bounds_gamma_deg)
    for _ in range(n_alpha_cp):
        bounds.append(alpha_bounds)

    # 🛑 THE FIX: Pass fixed objects/parameters via 'args' 
    # instead of relying on the unpicklable closure.
    obj_args = (integrator, vehicle, 
                2.0,                                      # target_mach
                1.2e6,                                    # q_dot_limit
                9050,                                     # ceiling_limit
                np.linspace(0, 1, n_alpha_cp),            # alpha_time_grid
                alpha_bounds)

    # Use Differential Evolution with multithreading
    print(f"[trajectory_optimizer] Running DE (dims={len(bounds)}) popsize={popsize} maxiter={maxiter} workers={workers}")
    result = differential_evolution(_de_objective_wrapper, bounds, 
                                    args=obj_args, # <-- Pass the fixed arguments here
                                    strategy='best1bin',
                                    maxiter=maxiter, popsize=popsize, tol=1e-3,
                                    polish=True, seed=seed,
                                    updating='deferred', workers=workers)

    best_x = result.x
    best_cost = result.fun
    print(f"[trajectory_optimizer] DE finished: best_cost={best_cost:.3e}")

    return {'x': best_x, 'cost': best_cost, 'result_obj': result}


def run_dymos_optimization(path, plotting=True, surrogate_type='linear', workers=16):
    """Compatibility wrapper with plotting integration and multithreading.

    Args:
        path: Output directory for plots
        plotting: Whether to generate plots
        surrogate_type: Type of surrogate model ('linear' or 'gpr')
        workers: Number of parallel workers (1 = serial, -1 = all CPUs)

    Returns: (best_range_m, best_max_qdot)
    """
    # Import locally to avoid circular imports
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

    # Run the shooting-based optimizer with multithreading
    print(f'\n[trajectory_optimizer] Running shooting optimizer (workers={workers})...')
    best = optimize_with_shooting(integrator, vehicle, 
                                  n_alpha_cp=12, 
                                  maxiter=20, 
                                  popsize=16,
                                  workers=workers)

    # Re-evaluate best candidate to get full trajectory
    best_x = best['x']
    n_cp = len(best_x) - 2
    alpha_time_grid = np.linspace(0, 1, n_cp)
    alpha_cps = best_x[2:]
    alpha_bounds_used = (-2.0, 6.0)  # Default from optimize_with_shooting
    alpha_profile = build_alpha_profile_from_cps(alpha_time_grid, alpha_cps, 
                                                 alpha_min=alpha_bounds_used[0], 
                                                 alpha_max=alpha_bounds_used[1])

    h0 = float(best_x[0])
    gamma0_deg = float(best_x[1])
    rho0, a0, _ = Atmosphere.atmo_model(h0)
    v0 = 8.0 * a0
    initial_state = np.array([0.0, h0, v0, np.deg2rad(gamma0_deg), vehicle.mass_ogive_kg])

    trajectory = integrator.glide_phase_with_alpha_profile(initial_state, alpha_profile, max_time=3600.0)

    if trajectory is None:
        return 0.0, 1e9

    best_range = float(trajectory['final_range'])
    best_qdot = float(trajectory['max_q_dot'])
    max_alt = float(np.max(trajectory['states'][1, :]))

    print(f"\n[trajectory_optimizer] ★ OPTIMIZED TRAJECTORY RESULTS:")
    print(f"  Deployment: {h0/1000:.1f} km altitude, {gamma0_deg:.1f}° flight path angle")
    print(f"  Maximum Range: {best_range/1000:.1f} km")
    print(f"  Maximum Heat Flux: {best_qdot:.0f} W/m²")
    print(f"  Maximum Altitude: {max_alt/1000:.2f} km")

    if max_alt > 9050:
        print(f"  ⚠️  WARNING: Exceeded 9 km ceiling by {(max_alt-9000)/1000:.2f} km")

    # Generate plots using simple_trajectory plotting function
    if plotting and plot_trajectory_results is not None:
        strategy_name = f"DE Optimized (h0={h0/1000:.1f}km, γ0={gamma0_deg:.1f}°)"
        plot_trajectory_results(trajectory, path, strategy_name)
    elif plotting:
        print("  ⚠️  Warning: plot_trajectory_results not available from simple_trajectory")

    return best_range, best_qdot


if __name__ == '__main__':
    import multiprocessing
    
    # Test run with multithreading
    #n_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    n_workers = 16
    print(f"\nRunning optimization with {n_workers} workers...")
    
    max_range, max_q_dot = run_dymos_optimization('.', plotting=True, workers=n_workers)
    print(f"\nFinal: Range = {max_range/1000:.1f} km, Q_dot = {max_q_dot:.0f} W/m²")