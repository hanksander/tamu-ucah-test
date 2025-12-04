import random
import math
import time
import copy
import os
import subprocess
import shutil
import numpy as np
import scipy as sp
import trimesh as tm
from fit_optimizer_3 import WaveriderCase
from simple_trajectory import run_dymos_optimization
from parametric_body_generator import (
    ParametricBody, 
    generate_parametric_body_mesh,
    write_tri_file
)
from parameter_solver import compute_reference_parameters

# ============ CONFIGURATION ============
# Toggle features on/off
USE_Z_SQUASH = False  # Set True to enable elliptical cross-section
USE_Z_CUT = False     # Set True to enable flat bottom cut

# Default values when features are disabled
DEFAULT_Z_SQUASH = 1.0  # Circular cross-section
DEFAULT_Z_CUT = None    # No cut

# Fixed control point x-positions (as fractions of length)
FIXED_CP_X = [0.1, 0.2, 0.45, 0.7, 1.0]

# GLOBAL MAXIMUM RADIUS CONSTRAINT (meters)
MAX_RADIUS_CONSTRAINT = 0.1  # Never exceed this radius
# ========================================

def set_up_ogive(params):
    """
    Set up ogive geometry from parameters and ensure valid STL.
    Control points are at fixed x-positions: 0, 0.1L, 0.2L, 0.45L, 0.7L, L
    
    CONSTRAINT: All radii are clamped to MAX_RADIUS_CONSTRAINT
    """
    length = params['length']
    max_radius = params['max_radius']
    
    # ENFORCE GLOBAL RADIUS CONSTRAINT
    max_radius = min(max_radius, MAX_RADIUS_CONSTRAINT)
    
    # Control points at fixed x-positions
    cp_x = [0.0] + [x_frac * length for x_frac in FIXED_CP_X]
    
    # Radii at each control point (scaled by max_radius)
    # Start at 0, then user-defined values
    cp_r_raw = [
        0.0,  # Nose is always radius 0
        params['cp1_r'] * max_radius,
        params['cp2_r'] * max_radius,
        params['cp3_r'] * max_radius,
        params['cp4_r'] * max_radius,
        params['cp5_r'] * max_radius
    ]
    
    # Clamp all radii to MAX_RADIUS_CONSTRAINT
    cp_r = [min(r, MAX_RADIUS_CONSTRAINT) for r in cp_r_raw]
    
    # Ensure radii are monotonically non-decreasing
    for i in range(1, len(cp_r)):
        if cp_r[i] < cp_r[i-1]:
            cp_r[i] = cp_r[i-1]  # Don't allow radius to decrease
    
    control_points = list(zip(cp_x, cp_r))
    
    # Handle z_squash and z_cut based on configuration
    z_squash = params.get('z_squash', DEFAULT_Z_SQUASH) if USE_Z_SQUASH else DEFAULT_Z_SQUASH
    z_cut = params.get('z_cut', DEFAULT_Z_CUT) if USE_Z_CUT else DEFAULT_Z_CUT
    
    # Create parametric body with higher spline order for smoother curves
    body = ParametricBody(
        length=length,
        control_points=control_points,
        z_cut=z_cut,
        z_squash=z_squash,
        spline_order=3,  # Cubic spline for smooth interpolation
        name=f"Ogive_L{length:.3f}_R{max_radius:.3f}"
    )
    
    # Generate mesh with higher resolution to capture smooth curves
    vertices, triangles, stats = generate_parametric_body_mesh(
        body, 
        n_axial=100,  # Increased from 80
        n_circumferential=50,  # Increased from 40
        add_nose_cap=True,
        add_tail_cap=True
    )
    
    # Save as .tri file
    tri_filename = './ogive.tri'
    write_tri_file(tri_filename, vertices, triangles)
    
    # Create trimesh object and ensure it's a valid volume
    mesh = tm.Trimesh(vertices=vertices, faces=triangles)
    
    # Fix mesh issues to make it a valid volume
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    
    # Fix normals to be consistent (all pointing outward)
    mesh.fix_normals()
    
    # Fill any small holes that might exist
    if not mesh.is_watertight:
        mesh.fill_holes()
    
    # Force the mesh to be a volume by ensuring it's watertight
    if not mesh.is_volume:
        # Try to repair the mesh
        mesh = mesh.convex_hull  # As a last resort, use convex hull
        print("Warning: Mesh was not a valid volume, using convex hull approximation")
    
    # Export to STL
    stl_filename = './ogive.stl'
    mesh.export(stl_filename)
    
    return body, stl_filename, tri_filename

def compute_aerodatabase(ogive_trimesh_path):
    """
    Run CBAERO simulation on ogive geometry.
    
    Assumes we are already in the case directory (from evaluate_particle's os.chdir).
    """
    current_dir = os.getcwd()
    
    # Look for run_cbaero.sh in parent directories
    search_dir = current_dir
    CBAERO_SCRIPT = None
    for _ in range(5):  # Search up to 5 levels
        candidate = os.path.join(search_dir, "run_cbaero.sh")
        if os.path.exists(candidate):
            CBAERO_SCRIPT = candidate
            break
        search_dir = os.path.dirname(search_dir)
    
    if CBAERO_SCRIPT is None:
        raise FileNotFoundError("Could not find run_cbaero.sh in parent directories")
    
    # Clean the path - remove './' prefix if present
    if ogive_trimesh_path.startswith('./'):
        ogive_trimesh_path = ogive_trimesh_path[2:]
    
    tri_source = ogive_trimesh_path
    
    # Verify input file exists (in current directory)
    if not os.path.exists(tri_source):
        raise FileNotFoundError(f"Ogive .tri file not found: {os.path.join(current_dir, tri_source)}")
    
    # Create CBAERO subdirectory in current directory
    filename = 'ogive'
    cbaero_dir = filename  # Just 'ogive' relative to current dir
    
    os.makedirs(cbaero_dir, exist_ok=True)
    
    # Copy .tri file to CBAERO directory
    tri_dest = os.path.join(cbaero_dir, f'{filename}.tri')
    shutil.copy2(tri_source, tri_dest)
    print(f"Copied {tri_source} → {tri_dest}")
    
    # Load mesh and compute reference parameters
    print("Computing reference parameters from mesh...")
    
    # Read .tri file to extract vertices and triangles
    vertices, triangles = read_tri_file(tri_dest)
    
    geom_params = compute_reference_parameters(vertices, triangles, verbose=False)
    print(f"  Sref = {geom_params['Sref']:.6f}")
    print(f"  cref = {geom_params['cref']:.6f}")
    print(f"  bref = {geom_params['bref']:.6f}")
    
    sref = geom_params['Sref']
    cref = geom_params['cref']
    bref = geom_params['bref']
    
    # Change to CBAERO directory to run the script
    os.chdir(cbaero_dir)
    
    try:
        # Run the CBAERO shell script
        cmd = ['unbuffer', 'bash', CBAERO_SCRIPT, filename, str(sref), str(cref), str(bref)]
        
        # Log file is now in current directory (inside ogive/)
        log_file = 'cbaero_run.log'
        
        print(f"Running CBAERO in {os.getcwd()}...")
        print(f"Command: {' '.join(cmd)}")
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Wait for completion with timeout (10 minutes)
            try:
                returncode = process.wait(timeout=600)
                
                if returncode != 0:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    raise subprocess.CalledProcessError(returncode, cmd, output=log_content)
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise TimeoutError("CBAERO script exceeded 10 minute timeout")
        
        # Print log for debugging
        with open(log_file, 'r') as f:
            log_content = f.read()
            print(f"CBAERO output:\n{log_content}")
        
        # Verify output files were created
        required_files = [
            f'{filename}.msh',
            f'{filename}.cbaero',
        ]
        
        for req_file in required_files:
            if not os.path.exists(req_file):
                with open(log_file, 'r') as f:
                    error_log = f.read()
                raise FileNotFoundError(
                    f"Expected output file not found: {req_file}\n"
                    f"Script output:\n{error_log}"
                )
        
        print("✓ CBAERO completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ CBAERO script failed with error:\n{e.output}")
        raise
    except subprocess.TimeoutExpired:
        print("✗ CBAERO script timed out")
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise
    finally:
        # Return to case directory (we only changed into ogive/ subdirectory)
        os.chdir(current_dir)


def read_tri_file(tri_path):
    """
    Read a .tri file and return vertices and triangles.
    """
    with open(tri_path, 'r') as f:
        # Read header
        line = f.readline().strip()
        n_verts, n_tris = map(int, line.split())
        
        # Read vertices
        vertices = []
        for _ in range(n_verts):
            line = f.readline().strip()
            x, y, z = map(float, line.split())
            vertices.append([x, y, z])
        
        # Read triangles
        triangles = []
        for _ in range(n_tris):
            line = f.readline().strip()
            indices = list(map(int, line.split()))
            triangles.append(indices)
    
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    
    # Check if indices are 1-based or 0-based
    min_idx = triangles.min()
    max_idx = triangles.max()
    
    if min_idx == 1 or max_idx >= n_verts:
        # 1-based indexing, convert to 0-based
        triangles = triangles - 1
        print(f"  Converted triangle indices from 1-based to 0-based")
    elif min_idx == 0 and max_idx < n_verts:
        # Already 0-based
        print(f"  Triangle indices are 0-based")
    else:
        print(f"  Warning: Unusual indexing detected (min={min_idx}, max={max_idx}, n_verts={n_verts})")
    
    # Verify indices are valid
    if triangles.min() < 0 or triangles.max() >= n_verts:
        raise ValueError(f"Invalid triangle indices: range [{triangles.min()}, {triangles.max()}] "
                        f"but only {n_verts} vertices (valid range [0, {n_verts-1}])")
    
    return vertices, triangles


def calculate_ogive_range(body):
    """Calculate range for ogive geometry using simple trajectory solver."""
    path = os.getcwd()
    # Use simple trajectory solver (drop-in replacement for Dymos)
    max_range, max_q_dot = run_dymos_optimization(path, plotting=False, surrogate_type="linear")
    return max_range, max_q_dot

def cost_function(range_val, max_q_dot, q_dot_limit=1200000):
    """
    Cost function combining range maximization and heat flux constraint.
    Note: q_dot_limit is in W/m² (1.2e6 = 1.2 MW/m²)
    """
    # Maximize range (minimize negative range)
    range_cost = -range_val
    
    # Penalize exceeding heat flux limit
    q_dot_penalty = max(0, max_q_dot - q_dot_limit) * 1e3
    
    return range_cost + q_dot_penalty

class Particle:
    def __init__(self, dim, bounds, vel_bounds):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [random.uniform(vb[0], vb[1]) for vb in vel_bounds]
        self.best_pos = self.position[:]
        self.best_cost = float('inf')

def enforce_bounds(x, bounds):
    return [max(b[0] + 1e-12, min(b[1] - 1e-12, xi)) for xi, b in zip(x, bounds)]

def clip_velocity(v, v_bounds):
    return [max(vb[0], min(vb[1], vi)) for vi, vb in zip(v, v_bounds)]

def enforce_monotonic_radii(cp_r):
    """Ensure radius values are monotonically non-decreasing."""
    cp_r_mono = [cp_r[0]]
    for i in range(1, len(cp_r)):
        cp_r_mono.append(max(cp_r[i], cp_r_mono[i-1]))
    return cp_r_mono

def write_control_points_log(params, particle_id, iteration, cost, log_file="control_points.log"):
    """Write control points to a log file for tracking."""
    max_radius = min(params['max_radius'], MAX_RADIUS_CONSTRAINT)
    length = params['length']
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Particle {particle_id} | Iteration {iteration} | Cost: {cost:.6e}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Length: {length:.6f} m | Max Radius: {max_radius:.6f} m\n")
        
        if USE_Z_SQUASH:
            f.write(f"Z-Squash: {params['z_squash']:.6f}\n")
        if USE_Z_CUT and params['z_cut'] is not None:
            f.write(f"Z-Cut: {params['z_cut']:.6f} m\n")
        
        f.write(f"\nControl Points (x-position, radius):\n")
        f.write(f"{'X/L':>8} {'X (m)':>10} {'r/R_max':>10} {'r (m)':>10}\n")
        f.write(f"{'-'*42}\n")
        
        # Nose (always at 0)
        f.write(f"{'0.000':>8} {'0.000':>10} {'0.000':>10} {'0.000':>10}\n")
        
        # User-defined control points
        cp_names = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r']
        for i, (x_frac, cp_name) in enumerate(zip(FIXED_CP_X, cp_names)):
            r_norm = params[cp_name]
            r_abs = r_norm * max_radius
            r_abs_clamped = min(r_abs, MAX_RADIUS_CONSTRAINT)
            x_abs = x_frac * length
            
            f.write(f"{x_frac:>8.3f} {x_abs:>10.6f} {r_norm:>10.6f} {r_abs_clamped:>10.6f}\n")
        
        f.write(f"\n")

def write_best_solution(params, iteration, cost, range_km, q_dot_kw, log_file="best_solution.log"):
    """Write the current best solution to a file."""
    max_radius = min(params['max_radius'], MAX_RADIUS_CONSTRAINT)
    length = params['length']
    
    with open(log_file, 'w') as f:  # Overwrite each time
        f.write(f"{'='*80}\n")
        f.write(f"BEST SOLUTION - Iteration {iteration}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"PERFORMANCE:\n")
        f.write(f"  Range:      {range_km:.3f} km\n")
        f.write(f"  Max q_dot:  {q_dot_kw:.1f} kW/m²\n")
        f.write(f"  Cost:       {cost:.6e}\n\n")
        
        f.write(f"GEOMETRY:\n")
        f.write(f"  Length:     {length:.6f} m\n")
        f.write(f"  Max Radius: {max_radius:.6f} m (constrained to {MAX_RADIUS_CONSTRAINT} m)\n")
        
        if USE_Z_SQUASH:
            f.write(f"  Z-Squash:   {params['z_squash']:.6f}\n")
        if USE_Z_CUT and params['z_cut'] is not None:
            f.write(f"  Z-Cut:      {params['z_cut']:.6f} m\n")
        
        f.write(f"\nCONTROL POINTS:\n")
        f.write(f"{'X/L':>8} {'X (m)':>10} {'r/R_max':>10} {'r (m)':>10}\n")
        f.write(f"{'-'*42}\n")
        
        # Nose
        f.write(f"{'0.000':>8} {'0.000':>10} {'0.000':>10} {'0.000':>10}\n")
        
        # User-defined control points
        cp_names = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r']
        for i, (x_frac, cp_name) in enumerate(zip(FIXED_CP_X, cp_names)):
            r_norm = params[cp_name]
            r_abs = r_norm * max_radius
            r_abs_clamped = min(r_abs, MAX_RADIUS_CONSTRAINT)
            x_abs = x_frac * length
            
            f.write(f"{x_frac:>8.3f} {x_abs:>10.6f} {r_norm:>10.6f} {r_abs_clamped:>10.6f}\n")
        
        f.write(f"\n{'='*80}\n")

def evaluate_particle(pos, penalty_coeff=1e6, verbose=True, particle_id=0, iteration=0):
    """Evaluate particle for ogive optimization."""
    # Unpack position vector based on active features
    idx = 0
    
    # 5 radius values (will be enforced monotonic)
    cp_r = pos[idx:idx+5]
    idx += 5
    
    length = pos[idx]
    idx += 1
    max_radius = pos[idx]
    idx += 1
    
    # ENFORCE GLOBAL RADIUS CONSTRAINT
    max_radius = min(max_radius, MAX_RADIUS_CONSTRAINT)
    
    # Optional parameters
    if USE_Z_SQUASH:
        z_squash = pos[idx]
        idx += 1
    else:
        z_squash = DEFAULT_Z_SQUASH
    
    if USE_Z_CUT:
        z_cut_raw = pos[idx]
        z_cut = z_cut_raw if z_cut_raw > -max_radius else None
        idx += 1
    else:
        z_cut = DEFAULT_Z_CUT
    
    # Enforce monotonic radii
    cp_r_mono = enforce_monotonic_radii(cp_r)
    
    # Create parameter dictionary
    params = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1], 
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3], 
        'cp5_r': cp_r_mono[4],
        'length': length,
        'max_radius': max_radius,
        'z_squash': z_squash,
        'z_cut': z_cut
    }
    
    # Create working directory
    os.makedirs('temp_ogive_cases', exist_ok=True)
    case_name = f"ogive-L{length:.3f}-R{max_radius:.3f}"
    if USE_Z_SQUASH:
        case_name += f"-zs{z_squash:.3f}"
    if USE_Z_CUT and z_cut is not None:
        case_name += f"-zc{z_cut:.3f}"
    
    case_dir = os.path.join('temp_ogive_cases', case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(case_dir)
    
    try:
        # [1/4] Set up ogive geometry
        if verbose:
            print(f"[1/4] Setting up ogive geometry...")
        try:
            body, stl_filename, tri_filename = set_up_ogive(params)
        except Exception as e:
            if verbose:
                print(f"✗ set_up_ogive failed: {e}")
            os.chdir(original_dir)
            write_control_points_log(params, particle_id, iteration, 1e9, 
                                    log_file=os.path.join(original_dir, "control_points.log"))
            return 1e9
        
        # [2/4] Check fitting
        if verbose:
            print(f"[2/4] Checking payload fit...")
        try:
            inst = WaveriderCase(stl_filename)
            fits = inst.try_fit_payload(verbose=False)
            if not fits[0]:  # fits is a tuple (success, residual)
                if verbose:
                    print(f"✗ Geometry does not fit payload, residual: {fits[1]}")
                penalty = 1e8 + abs(fits[1]) * 1e6
                os.chdir(original_dir)
                write_control_points_log(params, particle_id, iteration, penalty,
                                        log_file=os.path.join(original_dir, "control_points.log"))
                return penalty
        except Exception as e:
            if verbose:
                print(f"✗ check_fitting failed: {e}")
            os.chdir(original_dir)
            write_control_points_log(params, particle_id, iteration, 1e9,
                                    log_file=os.path.join(original_dir, "control_points.log"))
            return 1e9

        # [3/4] Compute aerodynamic database
        try:
            if verbose:
                print("[3/4] Computing aerodynamic database (running CBAERO)...")
            compute_aerodatabase('ogive.tri')
        except Exception as e:
            if verbose:
                print(f"✗ compute_aerodatabase failed: {e}")
            os.chdir(original_dir)
            write_control_points_log(params, particle_id, iteration, 1e9,
                                    log_file=os.path.join(original_dir, "control_points.log"))
            return 1e9
        
        # [4/4] Calculate range using simple trajectory solver
        try:
            if verbose:
                print("[4/4] Computing trajectory and range (simple solver)...")
            rng, max_q_dot = calculate_ogive_range(body)
            if not (isinstance(rng, (int, float)) and math.isfinite(rng)):
                if verbose:
                    print(f"✗ Invalid range returned: {rng}")
                os.chdir(original_dir)
                write_control_points_log(params, particle_id, iteration, 1e9,
                                        log_file=os.path.join(original_dir, "control_points.log"))
                return 1e9
        except Exception as e:
            if verbose:
                print(f"✗ calculate_ogive_range failed: {e}")
            os.chdir(original_dir)
            write_control_points_log(params, particle_id, iteration, 1e9,
                                    log_file=os.path.join(original_dir, "control_points.log"))
            return 1e9
        
        # Compute cost
        cost = cost_function(rng, max_q_dot)
        
        # Return to original directory before logging
        os.chdir(original_dir)
        
        # Log control points
        write_control_points_log(params, particle_id, iteration, cost)
        
        if verbose:
            print(f"✓ Evaluation complete: range={rng/1000:.1f} km, q_dot={max_q_dot/1000:.0f} kW/m², cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        os.chdir(original_dir)
        write_control_points_log(params, particle_id, iteration, 1e9)
        raise

def get_cone_profile():
    """Get radius profile for a cone at fixed x-positions."""
    # Linear growth: r(x) = x
    return [x for x in FIXED_CP_X]

def get_von_karman_profile():
    """Get radius profile for a von Karman ogive at fixed x-positions."""
    # von Karman: r(x) = sqrt(2*x - x^2) for x in [0, 1]
    # This is a circular arc tangent to the base
    return [np.sqrt(2*x - x**2) if x <= 1.0 else 1.0 for x in FIXED_CP_X]

def get_power_law_profile(n=0.75):
    """Get radius profile for a power law body at fixed x-positions."""
    # r(x) = x^n
    return [x**n for x in FIXED_CP_X]

def initialize_particle_near_profile(bounds, profile_blend=0.5, perturbation=0.15):
    """
    Initialize a particle near a blend of cone and von Karman profiles.
    
    Args:
        bounds: Parameter bounds
        profile_blend: 0 = cone, 1 = von Karman, 0.5 = 50/50 blend
        perturbation: Fraction of parameter range to perturb (noise level)
    
    Returns:
        position: List of parameter values
    """
    cone = get_cone_profile()
    von_karman = get_von_karman_profile()
    
    # Blend between cone and von Karman
    blended_profile = [
        (1 - profile_blend) * c + profile_blend * vk
        for c, vk in zip(cone, von_karman)
    ]
    
    position = []
    
    # First 5 parameters are the radius values (cp1_r through cp5_r)
    for i, r_normalized in enumerate(blended_profile):
        lo, hi = bounds[i]
        
        # Start from the blended profile value
        base_value = r_normalized
        
        # Add perturbation
        param_range = hi - lo
        noise = random.uniform(-perturbation * param_range, perturbation * param_range)
        value = base_value + noise
        
        # Clip to bounds
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
    
    # Remaining parameters (length, max_radius, optional z_squash, z_cut)
    for i in range(5, len(bounds)):
        lo, hi = bounds[i]
        
        # For length and max_radius, use middle of range with small perturbation
        if i == 5:  # length
            base_value = 0.75  # Moderate length
        elif i == 6:  # max_radius
            base_value = 0.05  # Moderate radius
        else:  # z_squash or z_cut
            base_value = (lo + hi) / 2
        
        param_range = hi - lo
        noise = random.uniform(-perturbation * param_range, perturbation * param_range)
        value = base_value + noise
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
    
    return position

def pso_optimize(num_particles=40, iterations=200, seed=None, verbose=True):
    """PSO optimization for ogive geometry."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Clear log files at start
    if os.path.exists("control_points.log"):
        os.remove("control_points.log")
    if os.path.exists("best_solution.log"):
        os.remove("best_solution.log")
    
    # Build bounds based on active features
    bounds = []
    
    # Radius control (normalized 0-1, will be scaled by max_radius)
    # These will be enforced to be monotonically non-decreasing
    cp_r_bounds = [
        (0.0, 0.3),   # cp1_r at x=0.1L
        (0.0, 0.5),   # cp2_r at x=0.2L
        (0.0, 0.8),   # cp3_r at x=0.45L
        (0.0, 1.0),   # cp4_r at x=0.7L
        (0.0, 1.0),   # cp5_r at x=1.0L (end)
    ]
    bounds.extend(cp_r_bounds)
    
    # Basic geometry parameters
    bounds.extend([
        (0.5, 1.0),     # length (meters)
        (0.02, MAX_RADIUS_CONSTRAINT),   # max_radius (constrained)
    ])
    
    # Optional parameters
    if USE_Z_SQUASH:
        bounds.append((0.3, 1.0))  # z_squash: 1.0 = circular, <1 = flattened
    
    if USE_Z_CUT:
        bounds.append((-0.5, 0.0))  # z_cut: negative z for bottom cut
    
    dim = len(bounds)
    vel_bounds = [(-(hi - lo) * 0.5, (hi - lo) * 0.5) for lo, hi in bounds]
    
    # Print configuration
    print("\n" + "="*70)
    print("OGIVE GEOMETRY OPTIMIZER CONFIGURATION")
    print("="*70)
    print(f"  Trajectory Solver: SIMPLE (shooting method)")
    print(f"  Fixed control points at: {['0.0L'] + [f'{x}L' for x in FIXED_CP_X]}")
    print(f"  Maximum Radius Constraint: {MAX_RADIUS_CONSTRAINT} m")
    print(f"  Z-Squash (elliptical): {'ENABLED' if USE_Z_SQUASH else 'DISABLED'}")
    print(f"  Z-Cut (flat bottom):   {'ENABLED' if USE_Z_CUT else 'DISABLED'}")
    print(f"  Parameter dimensions:  {dim}")
    print(f"  Initialization: Blend of cone and von Karman profiles")
    print("="*70)
    
    # Show reference profiles
    if verbose:
        print("\nReference profiles at control points:")
        cone = get_cone_profile()
        vk = get_von_karman_profile()
        print(f"{'x/L':<8} {'Cone':<10} {'von Karman':<12}")
        print("-" * 30)
        for i, x in enumerate(FIXED_CP_X):
            print(f"{x:<8.2f} {cone[i]:<10.4f} {vk[i]:<12.4f}")
        print()
    
    # Initialize swarm with smart initialization
    swarm = []
    
    # Create particles with varying blends of cone/von Karman
    for i in range(num_particles):
        p = Particle(dim, bounds, vel_bounds)
        
        if i == 0:
            # First particle: Pure cone
            blend = 0.0
            pert = 0.0
        elif i == 1:
            # Second particle: Pure von Karman
            blend = 1.0
            pert = 0.0
        elif i == 2:
            # Third particle: 50/50 blend
            blend = 0.5
            pert = 0.05
        elif i < num_particles // 2:
            # First half: Varied blends with small perturbation
            blend = i / (num_particles // 2)
            pert = 0.10
        else:
            # Second half: Random exploration
            blend = random.uniform(0.0, 1.0)
            pert = 0.20
        
        p.position = initialize_particle_near_profile(bounds, profile_blend=blend, perturbation=pert)
        p.velocity = [random.uniform(vb[0], vb[1]) for vb in vel_bounds]
        swarm.append(p)
    
    if verbose:
        print("\nParticle initialization strategy:")
        print(f"  Particle 1: Pure cone (linear)")
        print(f"  Particle 2: Pure von Karman (optimal theoretical)")
        print(f"  Particle 3: 50/50 blend")
        print(f"  Particles 4-{num_particles//2}: Gradual blend progression")
        print(f"  Particles {num_particles//2+1}-{num_particles}: Random exploration")
        print()
    
    gbest_pos = None
    gbest_cost = float('inf')
    gbest_range = 0
    gbest_q_dot = 0
    
    # Evaluate initial swarm
    print("\nEvaluating initial swarm...")
    for i, p in enumerate(swarm):
        p.position = enforce_bounds(p.position, bounds)
        cost = evaluate_particle(p.position, verbose=verbose, particle_id=i+1, iteration=0)
        p.best_cost = cost
        p.best_pos = p.position[:]
        
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_pos = p.position[:]
        
        if verbose:
            print(f"  Particle {i+1}/{num_particles}: cost = {cost:.6g}")
    
    # Write initial best solution
    idx = 0
    cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+5])
    best_params = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1],
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3],
        'cp5_r': cp_r_mono[4],
        'length': gbest_pos[5],
        'max_radius': min(gbest_pos[6], MAX_RADIUS_CONSTRAINT),
        'z_squash': gbest_pos[7] if USE_Z_SQUASH else DEFAULT_Z_SQUASH,
        'z_cut': gbest_pos[7 + int(USE_Z_SQUASH)] if USE_Z_CUT else DEFAULT_Z_CUT
    }
    # Estimate range from cost (rough approximation)
    est_range = -gbest_cost if gbest_cost < 0 else 0
    write_best_solution(best_params, 0, gbest_cost, est_range/1000, 0, "best_solution.log")
    
    history = []
    
    # PSO iterations
    for it in range(iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {it+1}/{iterations}")
        print(f"{'='*70}")
        
        for p_idx, p in enumerate(swarm):
            # Update velocity
            for i in range(dim):
                r1, r2 = random.random(), random.random()
                c1, c2, w_inertia = 1.49445, 1.49445, 0.729
                
                cognitive = c1 * r1 * (p.best_pos[i] - p.position[i])
                social = c2 * r2 * (gbest_pos[i] - p.position[i])
                p.velocity[i] = w_inertia * p.velocity[i] + cognitive + social
            
            # Clip velocity and update position
            p.velocity = clip_velocity(p.velocity, vel_bounds)
            p.position = enforce_bounds(
                [pi + vi for pi, vi in zip(p.position, p.velocity)], bounds
            )
            
            # Evaluate
            cost = evaluate_particle(p.position, verbose=verbose, 
                                    particle_id=p_idx+1, iteration=it+1)
            
            # Update personal best
            if cost < p.best_cost:
                p.best_cost = cost
                p.best_pos = p.position[:]
            
            # Update global best
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = p.position[:]
                
                # Update best solution log immediately when improved
                idx = 0
                cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+5])
                best_params = {
                    'cp1_r': cp_r_mono[0],
                    'cp2_r': cp_r_mono[1],
                    'cp3_r': cp_r_mono[2],
                    'cp4_r': cp_r_mono[3],
                    'cp5_r': cp_r_mono[4],
                    'length': gbest_pos[5],
                    'max_radius': min(gbest_pos[6], MAX_RADIUS_CONSTRAINT),
                }
                
                idx = 7
                if USE_Z_SQUASH:
                    best_params['z_squash'] = gbest_pos[idx]
                    idx += 1
                else:
                    best_params['z_squash'] = DEFAULT_Z_SQUASH
                
                if USE_Z_CUT:
                    z_cut_raw = gbest_pos[idx]
                    best_params['z_cut'] = z_cut_raw if z_cut_raw > -best_params['max_radius'] else None
                else:
                    best_params['z_cut'] = DEFAULT_Z_CUT
                
                # Estimate range from cost
                est_range = -gbest_cost if gbest_cost < 0 else 0
                write_best_solution(best_params, it+1, gbest_cost, est_range/1000, 0, "best_solution.log")
        
        history.append(gbest_cost)
        
        # Write best solution at end of each iteration
        idx = 0
        cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+5])
        best_params = {
            'cp1_r': cp_r_mono[0],
            'cp2_r': cp_r_mono[1],
            'cp3_r': cp_r_mono[2],
            'cp4_r': cp_r_mono[3],
            'cp5_r': cp_r_mono[4],
            'length': gbest_pos[5],
            'max_radius': min(gbest_pos[6], MAX_RADIUS_CONSTRAINT),
        }
        
        idx = 7
        if USE_Z_SQUASH:
            best_params['z_squash'] = gbest_pos[idx]
            idx += 1
        else:
            best_params['z_squash'] = DEFAULT_Z_SQUASH
        
        if USE_Z_CUT:
            z_cut_raw = gbest_pos[idx]
            best_params['z_cut'] = z_cut_raw if z_cut_raw > -best_params['max_radius'] else None
        else:
            best_params['z_cut'] = DEFAULT_Z_CUT
        
        est_range = -gbest_cost if gbest_cost < 0 else 0
        write_best_solution(best_params, it+1, gbest_cost, est_range/1000, 0, "best_solution.log")
        
        if verbose:
            print(f"\nIteration {it+1} complete: best cost = {gbest_cost:.6g}")
    
    # Format best solution
    idx = 0
    cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+5])
    
    best = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1],
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3],
        'cp5_r': cp_r_mono[4],
    }
    
    idx += 5
    
    best['length'] = gbest_pos[idx]
    idx += 1
    best['max_radius'] = min(gbest_pos[idx], MAX_RADIUS_CONSTRAINT)
    idx += 1
    
    if USE_Z_SQUASH:
        best['z_squash'] = gbest_pos[idx]
        idx += 1
    else:
        best['z_squash'] = DEFAULT_Z_SQUASH
    
    if USE_Z_CUT:
        z_cut_raw = gbest_pos[idx]
        best['z_cut'] = z_cut_raw if z_cut_raw > -best['max_radius'] else None
    else:
        best['z_cut'] = DEFAULT_Z_CUT
    
    return best, -gbest_cost, history

if __name__ == "__main__":
    print("="*70)
    print("OGIVE GEOMETRY OPTIMIZER (Simple Trajectory Solver)")
    print("="*70)
    
    best, best_range, hist = pso_optimize(
        num_particles=4, 
        iterations=50, 
        seed=42, 
        verbose=True
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest Range: {best_range/1000:.3f} km")
    print(f"\nOptimal Parameters:")
    print(f"  Length:     {best['length']:.6f} m")
    print(f"  Max Radius: {best['max_radius']:.6f} m (constrained to {MAX_RADIUS_CONSTRAINT} m)")
    
    if USE_Z_SQUASH:
        print(f"  Z-Squash:   {best['z_squash']:.6f}")
    if USE_Z_CUT and best['z_cut'] is not None:
        print(f"  Z-Cut:      {best['z_cut']:.6f} m")
    
    print(f"\n  Control Point Radii (normalized):")
    for i, cp_name in enumerate(['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r']):
        print(f"    {cp_name}: {best[cp_name]:.6f} (at x={FIXED_CP_X[i]}L)")
    
    print(f"\nLog files generated:")
    print(f"  - control_points.log: All particle evaluations")
    print(f"  - best_solution.log: Current best solution (updated each iteration)")