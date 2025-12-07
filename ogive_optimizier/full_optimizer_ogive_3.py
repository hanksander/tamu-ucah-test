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
import multiprocessing
from simple_trajectory import run_dymos_optimization
#from diff_trajectory_optimizer import run_dymos_optimization

from parametric_body_generator import (
    ParametricBody, 
    generate_parametric_body_mesh,
    write_tri_file
)
from parameter_solver import compute_reference_parameters
# ============ CONFIGURATION ============
# Toggle features on/off
USE_Z_SQUASH = True  # Set True to enable elliptical cross-section
USE_Z_CUT = True     # Set True to enable flat bottom cut
# Default values when features are disabled
DEFAULT_Z_SQUASH = 1.0  # Circular cross-section
DEFAULT_Z_CUT = None    # No cut
# NOSE CAP CONFIGURATION
use_hemisphere = True  # Set True to enable hemisphere nose cap
USE_NOSE_CAP = False  # Set True to enable old spherical nose cap
DEFAULT_NOSE_RADIUS = 0.005  # 5mm default nose radius
NOSE_RADIUS_BOUNDS = (0.007, 0.040)  # 1mm to 4cm
# Fixed control point x-positions (as fractions of length)
# Now includes 7 body control points
FIXED_CP_X = [0.05, 0.15, 0.25, 0.40, 0.55, 0.75, 1.0]  # 7 control points
# GLOBAL MAXIMUM RADIUS CONSTRAINT (meters)
MAX_RADIUS_CONSTRAINT = 0.1  # Never exceed this radius
# Z-CUT PERCENTAGE BOUNDS
Z_CUT_PERCENT_BOUNDS = (0.3, 1.1)  # 30% to 110% of (max_radius * z_squash)
# HEAT FLUX LIMITS (W/m²)
Q_DOT_LIMIT = 1.3e6  # Optimizer cost function limit: 1.2 MW/m²
Q_DOT_LIMIT_TRAJECTORY = 1.3e6  # Trajectory solver limit (can be different)
#MINIMUM VOLUME CONSTRAINT
MIN_VOLUME_LITERS = 6.0  # Minimum internal volume constraint (liters)
# RESTART CAPABILITY
ENABLE_RESTART = True  # Set True to initialize from previous run
RESTART_LOG_FILE = "control_points.log"
# ========================================
def set_up_ogive(params):
    """
    Set up ogive geometry from parameters with hemisphere nose support.
    """
    length = params['length']
    max_radius = params['max_radius']
    max_radius = min(max_radius, MAX_RADIUS_CONSTRAINT)
    
    control_points = []
    
    if use_hemisphere:
        nose_radius = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
        nose_radius = min(nose_radius, MAX_RADIUS_CONSTRAINT)
        # Hemisphere control point will be added by ParametricBody class
        # Just start with first body control point after hemisphere
    else:
        # Sharp or tangent nose (original behavior)
        if USE_NOSE_CAP:
            nose_radius = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
            nose_radius = min(nose_radius, MAX_RADIUS_CONSTRAINT)
            # Original tangent sphere control points
            control_points.append((0.0, 0.0))
            control_points.append((nose_radius, nose_radius))
        else:
            control_points.append((0.0, 0.0))
    
    # Body control points at fixed x-positions
    cp_x = [x_frac * length for x_frac in FIXED_CP_X]
    
    cp_r_raw = [
        params['cp1_r'] * max_radius,
        params['cp2_r'] * max_radius,
        params['cp3_r'] * max_radius,
        params['cp4_r'] * max_radius,
        params['cp5_r'] * max_radius,
        params['cp6_r'] * max_radius,
        params['cp7_r'] * max_radius,
    ]
    
    cp_r = [min(r, MAX_RADIUS_CONSTRAINT) for r in cp_r_raw]
    
    # Ensure monotonic radii
    if use_hemisphere:
        nose_radius = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
        prev_r = nose_radius
    elif USE_NOSE_CAP:
        nose_radius = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
        prev_r = nose_radius
    else:
        prev_r = 0.0
    
    for i in range(len(cp_r)):
        if cp_r[i] < prev_r:
            cp_r[i] = prev_r
        prev_r = cp_r[i]
    
    # Add body control points
    for x, r in zip(cp_x, cp_r):
        control_points.append((x, r))
    
    z_squash = params.get('z_squash', DEFAULT_Z_SQUASH) if USE_Z_SQUASH else DEFAULT_Z_SQUASH
    z_cut = params.get('z_cut', DEFAULT_Z_CUT) if USE_Z_CUT else DEFAULT_Z_CUT
    
    # Create parametric body with hemisphere option
    body = ParametricBody(
        length=length,
        control_points=control_points,
        z_cut=z_cut,
        z_squash=z_squash,
        spline_order=3,
        hemisphere_nose=use_hemisphere,
        hemisphere_radius=nose_radius if use_hemisphere else None,
        name=f"Ogive_L{length:.3f}_R{max_radius:.3f}"
    )
    
    # Generate mesh
    vertices, triangles, stats = generate_parametric_body_mesh(
        body, 
        n_axial=100,
        n_circumferential=50,
        add_nose_cap=True,
        add_tail_cap=True
    )
    
    # Save and validate mesh
    tri_filename = './ogive.tri'
    write_tri_file(tri_filename, vertices, triangles, swap_yz=False)
    
    mesh = tm.Trimesh(vertices=vertices, faces=triangles)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    
    if not mesh.is_watertight:
        mesh.fill_holes()
    
    if not mesh.is_volume:
        mesh = mesh.convex_hull
        print("Warning: Mesh was not a valid volume, using convex hull approximation")
    
    stl_filename = './ogive.stl'
    mesh.export(stl_filename)
    
    return body, stl_filename, tri_filename

def compute_volume(vertices, triangles):
    """
    Compute volume of a closed mesh using the divergence theorem.
    
    Args:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        triangles: numpy array of shape (M, 3) containing triangle vertex indices
    
    Returns:
        volume: float, volume in cubic meters
    """
    volume = 0.0
    
    for tri in triangles:
        # Get the three vertices of the triangle
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        
        # Compute signed volume of tetrahedron formed by triangle and origin
        # V = (1/6) * |v0 · (v1 × v2)|
        cross = np.cross(v1, v2)
        volume += np.dot(v0, cross)
    
    # Divide by 6 to get actual volume
    volume = abs(volume) / 6.0
    
    return volume

def compute_aerodatabase(ogive_trimesh_path):
    """Run CBAERO simulation on ogive geometry."""
    current_dir = os.getcwd()
    
    # Look for run_cbaero.sh in parent directories
    search_dir = current_dir
    CBAERO_SCRIPT = None
    for _ in range(5):
        candidate = os.path.join(search_dir, "run_cbaero.sh")
        if os.path.exists(candidate):
            CBAERO_SCRIPT = candidate
            break
        search_dir = os.path.dirname(search_dir)
    
    if CBAERO_SCRIPT is None:
        raise FileNotFoundError("Could not find run_cbaero.sh in parent directories")
    
    if ogive_trimesh_path.startswith('./'):
        ogive_trimesh_path = ogive_trimesh_path[2:]
    
    tri_source = ogive_trimesh_path
    
    if not os.path.exists(tri_source):
        raise FileNotFoundError(f"Ogive .tri file not found: {os.path.join(current_dir, tri_source)}")
    
    filename = 'ogive'
    cbaero_dir = filename
    
    os.makedirs(cbaero_dir, exist_ok=True)
    
    tri_dest = os.path.join(cbaero_dir, f'{filename}.tri')
    shutil.copy2(tri_source, tri_dest)
    print(f"Copied {tri_source} → {tri_dest}")
    
    print("Computing reference parameters from mesh...")
    vertices, triangles = read_tri_file(tri_dest)
    
    geom_params = compute_reference_parameters(vertices, triangles, verbose=False)
    print(f"  Sref = {geom_params['Sref']:.6f}")
    print(f"  cref = {geom_params['cref']:.6f}")
    print(f"  bref = {geom_params['bref']:.6f}")
    
    sref = geom_params['Sref']
    cref = geom_params['cref']
    bref = geom_params['bref']
    
    os.chdir(cbaero_dir)
    
    try:
        cmd = ['unbuffer', 'bash', CBAERO_SCRIPT, filename, str(sref), str(cref), str(bref)]
        log_file = 'cbaero_run.log'
        
        print(f"Running CBAERO in {os.getcwd()}...")
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
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
        
        with open(log_file, 'r') as f:
            log_content = f.read()
            print(f"CBAERO output:\n{log_content}")
        
        required_files = [f'{filename}.msh', f'{filename}.cbaero']
        
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
        os.chdir(current_dir)

def read_tri_file(tri_path):
    """Read a .tri file and return vertices and triangles."""
    with open(tri_path, 'r') as f:
        line = f.readline().strip()
        n_verts, n_tris = map(int, line.split())
        
        vertices = []
        for _ in range(n_verts):
            line = f.readline().strip()
            x, y, z = map(float, line.split())
            vertices.append([x, y, z])
        
        triangles = []
        for _ in range(n_tris):
            line = f.readline().strip()
            indices = list(map(int, line.split()))
            triangles.append(indices)
    
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    
    min_idx = triangles.min()
    max_idx = triangles.max()
    
    if min_idx == 1 or max_idx >= n_verts:
        triangles = triangles - 1
        print(f"  Converted triangle indices from 1-based to 0-based")
    elif min_idx == 0 and max_idx < n_verts:
        print(f"  Triangle indices are 0-based")
    else:
        print(f"  Warning: Unusual indexing detected (min={min_idx}, max={max_idx}, n_verts={n_verts})")
    
    if triangles.min() < 0 or triangles.max() >= n_verts:
        raise ValueError(f"Invalid triangle indices: range [{triangles.min()}, {triangles.max()}] "
                        f"but only {n_verts} vertices (valid range [0, {n_verts-1}])")
    
    return vertices, triangles

def calculate_ogive_range(body):
    """Calculate range for ogive geometry using simple trajectory solver."""
    path = os.getcwd()
    # Pass the q_dot_limit to the trajectory solver
    max_range, max_q_dot = run_dymos_optimization(
        path, 
        plotting=True, 
        surrogate_type="linear", 
        q_dot_limit=Q_DOT_LIMIT_TRAJECTORY,
        mach_range = [7.75, 8.0]
    )
    return max_range, max_q_dot

def cost_function(range_val, max_q_dot, q_dot_limit=None):
    """
    Cost function combining range maximization and heat flux constraint.
    """
    if q_dot_limit is None:
        q_dot_limit = Q_DOT_LIMIT
    
    range_cost = -range_val
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

def enforce_monotonic_radii(cp_r, nose_radius=None):
    """Ensure radius values are monotonically non-decreasing."""
    if nose_radius is not None:
        cp_r_mono = [max(cp_r[0], nose_radius)]
    else:
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
        
        if USE_NOSE_CAP or use_hemisphere:
            nose_r = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
            f.write(f"Nose Radius: {nose_r*1000:.3f} mm\n")
        
        if USE_Z_SQUASH:
            f.write(f"Z-Squash: {params['z_squash']:.6f}\n")
        
        # After Z-Squash logging, replace the Z-Cut section:
        
        # Write both z_cut (actual value) and z_cut_percent (optimized parameter)
        if USE_Z_CUT:
            z_cut = params.get('z_cut')
            z_cut_percent = params.get('z_cut_percent')
            if z_cut is not None:
                f.write(f"Z-Cut: {z_cut:.6f} m")
                if z_cut_percent is not None:
                    f.write(f" ({z_cut_percent*100:.1f}%)")
                f.write(f"\n")
            elif z_cut_percent is not None:
                if z_cut_percent >= 1.0:
                    f.write(f"Z-Cut: None ({z_cut_percent*100:.1f}% >= 100%)\n")
                else:
                    f.write(f"Z-Cut: None\n")
        
        f.write(f"\nControl Points (x-position, radius):\n")
        f.write(f"{'X/L':>8} {'X (m)':>10} {'r/R_max':>10} {'r (m)':>10}\n")
        f.write(f"{'-'*42}\n")
        
        # Nose
        if USE_NOSE_CAP or use_hemisphere:
            nose_r = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
            nose_r_clamped = min(nose_r, MAX_RADIUS_CONSTRAINT)
            f.write(f"{'NOSE':>8} {nose_r_clamped:>10.6f} {'N/A':>10} {nose_r_clamped:>10.6f}\n")
        else:
            f.write(f"{'0.000':>8} {'0.000':>10} {'0.000':>10} {'0.000':>10}\n")
        
        # Body control points
        cp_names = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r', 'cp6_r', 'cp7_r']
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
    
    with open(log_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"BEST SOLUTION - Iteration {iteration}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"PERFORMANCE:\n")
        f.write(f"  Range:      {range_km:.3f} km\n")
        f.write(f"  Max q_dot:  {q_dot_kw:.1f} kW/m²\n")
        f.write(f"  Q_dot limit: {Q_DOT_LIMIT/1e3:.1f} kW/m²\n")
        f.write(f"  Cost:       {cost:.6e}\n\n")
        
        f.write(f"GEOMETRY:\n")
        f.write(f"  Length:     {length:.6f} m\n")
        f.write(f"  Max Radius: {max_radius:.6f} m\n")
        
        if USE_NOSE_CAP or use_hemisphere:
            nose_r = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
            f.write(f"  Nose Radius: {nose_r*1000:.3f} mm\n")
        
        if USE_Z_SQUASH:
            f.write(f"  Z-Squash:   {params['z_squash']:.6f}\n")
        if USE_Z_CUT and params['z_cut'] is not None:
            f.write(f"  Z-Cut:      {params['z_cut']:.6f} m\n")
        
        f.write(f"\nCONTROL POINTS:\n")
        f.write(f"{'X/L':>8} {'X (m)':>10} {'r/R_max':>10} {'r (m)':>10}\n")
        f.write(f"{'-'*42}\n")
        
        if USE_NOSE_CAP or use_hemisphere:
            nose_r = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
            nose_r_clamped = min(nose_r, MAX_RADIUS_CONSTRAINT)
            f.write(f"{'NOSE':>8} {nose_r_clamped:>10.6f} {'N/A':>10} {nose_r_clamped:>10.6f}\n")
        else:
            f.write(f"{'0.000':>8} {'0.000':>10} {'0.000':>10} {'0.000':>10}\n")
        
        cp_names = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r', 'cp6_r', 'cp7_r']
        for i, (x_frac, cp_name) in enumerate(zip(FIXED_CP_X, cp_names)):
            r_norm = params[cp_name]
            r_abs = r_norm * max_radius
            r_abs_clamped = min(r_abs, MAX_RADIUS_CONSTRAINT)
            x_abs = x_frac * length
            
            f.write(f"{x_frac:>8.3f} {x_abs:>10.6f} {r_norm:>10.6f} {r_abs_clamped:>10.6f}\n")
        
        f.write(f"\n{'='*80}\n")

def evaluate_particle(pos, penalty_coeff=1e6, verbose=True, particle_id=0, iteration=0):
    """Evaluate particle for ogive optimization with 7 control points and optional nose cap."""
    idx = 0
    
     # Nose radius
    if USE_NOSE_CAP or use_hemisphere:
        nose_radius = pos[idx]
        nose_radius = min(nose_radius, MAX_RADIUS_CONSTRAINT)
        idx += 1
    else:
        nose_radius = None
    
    # 7 radius values
    cp_r = pos[idx:idx+7]
    idx += 7
    
    length = pos[idx]
    idx += 1
    max_radius = pos[idx]
    idx += 1
    max_radius = min(max_radius, MAX_RADIUS_CONSTRAINT)
    
    # Z-squash
    if USE_Z_SQUASH:
        z_squash = pos[idx]
        idx += 1
    else:
        z_squash = DEFAULT_Z_SQUASH
    
    # Z-cut as PERCENTAGE
    if USE_Z_CUT:
        z_cut_percent = pos[idx]
        idx += 1
        
        # Compute actual z_cut from percentage
        effective_radius = max_radius * z_squash
        z_cut_raw = -effective_radius * z_cut_percent
        
        # If percent > 1.0, no cut (cuts beyond body)
        if z_cut_percent >= 1.0:
            z_cut = None
        else:
            z_cut = z_cut_raw
    else:
        z_cut = DEFAULT_Z_CUT
        z_cut_percent = None
    
    # Enforce monotonic radii
    cp_r_mono = enforce_monotonic_radii(cp_r, nose_radius)
    
    # Create params
    params = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1],
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3],
        'cp5_r': cp_r_mono[4],
        'cp6_r': cp_r_mono[5],
        'cp7_r': cp_r_mono[6],
        'length': length,
        'max_radius': max_radius,
        'z_squash': z_squash,
        'z_cut': z_cut,
        'z_cut_percent': z_cut_percent  # Store for logging
    }
    
    
    if USE_NOSE_CAP or use_hemisphere:
        params['nose_radius'] = nose_radius
    
    # Working directory
    os.makedirs('temp_ogive_cases', exist_ok=True)
    case_name = f"ogive-L{length:.3f}-R{max_radius:.3f}"
    if USE_NOSE_CAP or use_hemisphere:
        case_name += f"-nr{nose_radius*1000:.1f}mm"
    if USE_Z_SQUASH:
        case_name += f"-zs{z_squash:.3f}"
    if USE_Z_CUT and z_cut is not None:
        case_name += f"-zc{abs(z_cut):.3f}"
    
    case_dir = os.path.join('temp_ogive_cases', case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(case_dir)
    
    try:
        # [1/4] Set up geometry
        if verbose:
            print(f"[1/4] Setting up ogive geometry...")
            if USE_NOSE_CAP or use_hemisphere:
                print(f"      Nose radius: {nose_radius*1000:.2f} mm")
            if USE_Z_CUT and z_cut_percent is not None:
                if z_cut is None:
                    print(f"      Z-cut: {z_cut_percent*100:.1f}% (no cut)")
                else:
                    print(f"      Z-cut: {z_cut_percent*100:.1f}% = {abs(z_cut)*1000:.1f} mm")
        
        try:
            body, stl_filename, tri_filename = set_up_ogive(params)
        except Exception as e:
            if verbose:
                print(f"✗ set_up_ogive failed: {e}")
            os.chdir(original_dir)
            write_control_points_log(params, particle_id, iteration, 1e9, 
                                    log_file=os.path.join(original_dir, "control_points.log"))
            return 1e9
        
        # [1.5/4] Check volume constraint
        if verbose:
            print(f"[1.5/4] Checking volume constraint...")
        try:
            vertices, triangles = read_tri_file(tri_filename)
            volume_m3 = compute_volume(vertices, triangles)
            volume_liters = volume_m3 * 1000.0  # Convert m³ to liters
            
            if verbose:
                print(f"      Computed volume: {volume_liters:.2f} liters")
            
            if volume_liters < MIN_VOLUME_LITERS:
                if verbose:
                    print(f"✗ Volume constraint violated: {volume_liters:.2f}L < {MIN_VOLUME_LITERS}L")
                volume_deficit = MIN_VOLUME_LITERS - volume_liters
                penalty = 1e7 + volume_deficit * 1e6
                os.chdir(original_dir)
                write_control_points_log(params, particle_id, iteration, penalty,
                                        log_file=os.path.join(original_dir, "control_points.log"))
                return penalty
            elif verbose:
                print(f"✓ Volume constraint satisfied ({volume_liters:.2f}L >= {MIN_VOLUME_LITERS}L)")
                
        except Exception as e:
            if verbose:
                print(f"✗ Volume computation failed: {e}")
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
            if not fits[0]:
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
        
        # [3/4] Compute aerodatabase
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
        
        # [4/4] Calculate range
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
        cost = cost_function(rng, max_q_dot, q_dot_limit=Q_DOT_LIMIT)
        
        os.chdir(original_dir)
        write_control_points_log(params, particle_id, iteration, cost)
        
        if verbose:
            print(f"✓ Evaluation complete: range={rng/1000:.1f} km, q_dot={max_q_dot/1000:.0f} kW/m², "
                  f"volume={volume_liters:.1f}L, cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        os.chdir(original_dir)
        write_control_points_log(params, particle_id, iteration, 1e9)
        raise

def get_cone_profile():
    """Get radius profile for a cone at fixed x-positions (7 points)."""
    return [x for x in FIXED_CP_X]

def get_von_karman_profile():
    """Get radius profile for a von Karman ogive at fixed x-positions (7 points)."""
    return [np.sqrt(2*x - x**2) if x <= 1.0 else 1.0 for x in FIXED_CP_X]

def get_power_law_profile(n=0.75):
    """Get radius profile for a power law body at fixed x-positions (7 points)."""
    return [x**n for x in FIXED_CP_X]

def initialize_particle_near_profile(bounds, profile_blend=0.5, perturbation=0.15):
    """
    Initialize a particle near a blend of cone and von Karman profiles.
    Now handles 7 control points + optional nose radius.
    """
    cone = get_cone_profile()
    von_karman = get_von_karman_profile()
    
    blended_profile = [
        (1 - profile_blend) * c + profile_blend * vk
        for c, vk in zip(cone, von_karman)
    ]
    
    position = []
    idx = 0
    
    # Nose radius (if enabled - for EITHER hemisphere or tangent sphere)
    if USE_NOSE_CAP or use_hemisphere:
        lo, hi = bounds[idx]
        # Start from default value with perturbation
        base_value = DEFAULT_NOSE_RADIUS
        param_range = hi - lo
        noise = random.uniform(-perturbation * param_range, perturbation * param_range)
        value = base_value + noise
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
        idx += 1
    
    # 7 control point radii
    for i, r_normalized in enumerate(blended_profile):
        lo, hi = bounds[idx]
        base_value = r_normalized
        param_range = hi - lo
        noise = random.uniform(-perturbation * param_range, perturbation * param_range)
        value = base_value + noise
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
        idx += 1
    
    # Remaining parameters (length, max_radius, optional z_squash, z_cut)
    for i in range(idx, len(bounds)):
        lo, hi = bounds[i]
        
        if i == idx:  # length
            base_value = 1
        elif i == idx + 1:  # max_radius
            base_value = 0.1
        else:  # z_squash or z_cut
            base_value = (lo + hi) / 2
        
        param_range = hi - lo
        noise = random.uniform(-perturbation * param_range, perturbation * param_range)
        value = base_value + noise
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
    
    return position

def parse_control_points_log(log_file):
    """Parse control_points.log to extract particle configurations."""
    if not os.path.exists(log_file):
        print(f"  Log file not found: {log_file}")
        return []
    
    particles = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"  Reading {len(lines)} lines from log file...")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for particle header (line with ====)
            if line.startswith('=') and '=' * 10 in line:
                # Next line should be "Particle N | Iteration M | Cost: ..."
                if i + 1 < len(lines):
                    header = lines[i + 1].strip()
                    if header.startswith('Particle'):
                        parts = header.split('|')
                        if len(parts) >= 3:
                            try:
                                cost_str = parts[2].split(':')[1].strip()
                                cost = float(cost_str)
                            except:
                                print(f"    Warning: Could not parse cost from: {header}")
                                i += 1
                                continue
                            
                            # Skip to next line after header separator
                            i += 2
                            if i < len(lines) and '=' in lines[i]:
                                i += 1  # Skip second separator line
                            
                            params = {}
                            
                            # Parse geometry lines
                            while i < len(lines):
                                line = lines[i].strip()
                                
                                # Check for Length and Max Radius line
                                if line.startswith('Length:'):
                                    parts = line.split('|')
                                    if len(parts) >= 2:
                                        try:
                                            length_str = parts[0].split(':')[1].strip().replace('m', '').strip()
                                            radius_str = parts[1].split(':')[1].strip().replace('m', '').strip()
                                            params['length'] = float(length_str)
                                            params['max_radius'] = float(radius_str)
                                        except Exception as e:
                                            print(f"    Warning: Could not parse length/radius: {e}")
                                    i += 1
                                    continue
                                
                                # Check for Nose Radius
                                elif line.startswith('Nose Radius:'):
                                    try:
                                        nose_str = line.split(':')[1].strip().replace('mm', '').strip()
                                        params['nose_radius'] = float(nose_str) / 1000.0
                                    except Exception as e:
                                        print(f"    Warning: Could not parse nose radius: {e}")
                                    i += 1
                                    continue
                                
                                # Check for Z-Squash
                                elif line.startswith('Z-Squash:'):
                                    try:
                                        params['z_squash'] = float(line.split(':')[1].strip())
                                    except Exception as e:
                                        print(f"    Warning: Could not parse z-squash: {e}")
                                    i += 1
                                    continue
                                
                                # Check for Z-Cut (now handles both formats)
                                elif line.startswith('Z-Cut:'):
                                    try:
                                        rest = line.split(':', 1)[1].strip()
                                        # Check if it contains percentage
                                        if '(' in rest and '%' in rest:
                                            # Extract both value and percentage
                                            z_cut_str = rest.split('(')[0].replace('m', '').strip()
                                            percent_str = rest.split('(')[1].split('%')[0].strip()
                                            if z_cut_str.lower() != 'none':
                                                params['z_cut'] = float(z_cut_str)
                                            params['z_cut_percent'] = float(percent_str) / 100.0
                                        else:
                                            # Old format: just the value
                                            z_cut_str = rest.replace('m', '').strip()
                                            if z_cut_str.lower() != 'none':
                                                params['z_cut'] = float(z_cut_str)
                                    except Exception as e:
                                        print(f"    Warning: Could not parse z-cut: {e}")
                                    i += 1
                                    continue
                                
                                # Check for Control Points section
                                elif 'Control Points' in line:
                                    i += 1  # Skip "Control Points" line
                                    if i < len(lines):
                                        i += 1  # Skip header line (X/L, X (m), ...)
                                    if i < len(lines):
                                        i += 1  # Skip separator line (----)
                                    
                                    # Skip NOSE line
                                    if i < len(lines):
                                        i += 1
                                    
                                    # Parse 7 control points
                                    cp_names = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 
                                               'cp5_r', 'cp6_r', 'cp7_r']
                                    for cp_name in cp_names:
                                        if i < len(lines):
                                            cp_line = lines[i].strip()
                                            parts = cp_line.split()
                                            if len(parts) >= 4:
                                                try:
                                                    # Third column is r/R_max (normalized radius)
                                                    params[cp_name] = float(parts[2])
                                                except:
                                                    print(f"    Warning: Could not parse {cp_name} from: {cp_line}")
                                            i += 1
                                    
                                    # Done parsing this particle
                                    break
                                
                                else:
                                    i += 1
                            
                            # Check if we got valid parameters
                            required_keys = ['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 
                                           'cp5_r', 'cp6_r', 'cp7_r', 
                                           'length', 'max_radius']
                            if all(k in params for k in required_keys):
                                particles.append({'params': params, 'cost': cost})
                            else:
                                missing = [k for k in required_keys if k not in params]
                                print(f"    Warning: Incomplete particle, missing: {missing}")
                        else:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        print(f"  ✓ Parsed {len(particles)} valid particle configurations from log")
        return particles
    
    except Exception as e:
        print(f"  ❌ Error parsing log file: {e}")
        import traceback
        traceback.print_exc()
        return []


def params_dict_to_position(params, bounds):
    """Convert parameter dictionary to position vector matching bounds order.
    
    FIXED: Properly handles z_cut_percent reconstruction from both new and old logs.
    """
    position = []
    idx = 0
    
    # Nose radius
    if USE_NOSE_CAP or use_hemisphere:
        value = params.get('nose_radius', DEFAULT_NOSE_RADIUS)
        lo, hi = bounds[idx]
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
        idx += 1
    
    # 7 control point radii
    for i in range(1, 8):
        cp_name = f'cp{i}_r'
        value = params.get(cp_name, 0.5)
        lo, hi = bounds[idx]
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
        idx += 1
    
    # Length
    value = params.get('length', 0.75)
    lo, hi = bounds[idx]
    value = max(lo + 1e-12, min(hi - 1e-12, value))
    position.append(value)
    idx += 1
    
    # Max radius
    value = params.get('max_radius', 0.05)
    lo, hi = bounds[idx]
    value = max(lo + 1e-12, min(hi - 1e-12, value))
    position.append(value)
    idx += 1
    
    # Z-squash
    if USE_Z_SQUASH:
        value = params.get('z_squash', 1.0)
        lo, hi = bounds[idx]
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
        idx += 1
    
    # Z-cut as percentage (FIXED!)
    if USE_Z_CUT:
        z_cut = params.get('z_cut')
        z_cut_percent = params.get('z_cut_percent')
        
        # Priority 1: Use z_cut_percent if available (from new logs)
        if z_cut_percent is not None:
            value = z_cut_percent
        # Priority 2: Reconstruct from z_cut (from old logs)
        elif z_cut is not None:
            max_radius = params.get('max_radius', 0.05)
            z_squash = params.get('z_squash', 1.0)
            effective_radius = max_radius * z_squash
            # z_cut is negative: z_cut = -effective_radius * z_cut_percent
            # So: z_cut_percent = -z_cut / effective_radius
            if effective_radius > 0:
                value = -z_cut / effective_radius
            else:
                value = 0.5  # Fallback
        else:
            # No cut (z_cut was None in log)
            value = 1.05  # Default to no cut
        
        lo, hi = bounds[idx]
        value = max(lo + 1e-12, min(hi - 1e-12, value))
        position.append(value)
    
    return position

def initialize_swarm_with_restart(num_particles, bounds, vel_bounds, log_file):
    """Initialize swarm, optionally using data from previous run."""
    dim = len(bounds)
    swarm = []
    restart_info = {
        'used_restart': False,
        'n_from_log': 0,
        'n_new': 0
    }
    
    previous_particles = []
    if ENABLE_RESTART and os.path.exists(log_file):
        print(f"\n{'='*70}")
        print("RESTART MODE ENABLED")
        print(f"{'='*70}")
        print(f"Loading previous particles from {log_file}...")
        previous_particles = parse_control_points_log(log_file)
        
        if previous_particles:
            previous_particles.sort(key=lambda x: x['cost'])
            
            print(f"\nFound {len(previous_particles)} previous evaluations")
            print(f"  Best cost: {previous_particles[0]['cost']:.6g}")
            print(f"  Worst cost: {previous_particles[-1]['cost']:.6g}")
            
            restart_info['used_restart'] = True
    
    n_from_log = min(len(previous_particles), num_particles)
    
    if n_from_log > 0:
        print(f"\nInitializing {n_from_log} particles from previous run...")
        
        for i in range(n_from_log):
            p = Particle(dim, bounds, vel_bounds)
            
            prev_params = previous_particles[i]['params']
            p.position = params_dict_to_position(prev_params, bounds)
            
            p.velocity = [random.uniform(vb[0], vb[1]) for vb in vel_bounds]
            
            p.best_pos = p.position[:]
            p.best_cost = previous_particles[i]['cost']
            
            swarm.append(p)
            
            print(f"  Particle {i+1}: cost={p.best_cost:.6g}, "
                  f"L={prev_params['length']:.3f}m, R={prev_params['max_radius']:.4f}m")
        
        restart_info['n_from_log'] = n_from_log
    
    n_new = num_particles - n_from_log
    
    if n_new > 0:
        print(f"\nInitializing {n_new} new particles with profile blending...")
        
        for i in range(n_new):
            p = Particle(dim, bounds, vel_bounds)
            
            particle_idx = n_from_log + i
            
            if particle_idx == 0 and n_from_log == 0:
                blend = 0.0
                pert = 0.0
            elif particle_idx == 1 and n_from_log <= 1:
                blend = 1.0
                pert = 0.0
            elif particle_idx == 2 and n_from_log <= 2:
                blend = 0.5
                pert = 0.05
            elif particle_idx < num_particles // 2:
                blend = particle_idx / (num_particles // 2)
                pert = 0.10
            else:
                blend = random.uniform(0.0, 1.0)
                pert = 0.20
            
            p.position = initialize_particle_near_profile(bounds, profile_blend=blend, perturbation=pert)
            p.velocity = [random.uniform(vb[0], vb[1]) for vb in vel_bounds]
            swarm.append(p)
        
        restart_info['n_new'] = n_new
    
    return swarm, restart_info

# PSO optimization function - add to main optimizer file
def pso_optimize(num_particles=40, iterations=200, seed=None, verbose=True):
    """PSO optimization for ogive geometry with 7 control points and optional nose cap."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Backup logs if not restarting
    if not ENABLE_RESTART:
        if os.path.exists("control_points.log"):
            import shutil
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"control_points_backup_{timestamp}.log"
            shutil.move("control_points.log", backup_name)
            print(f"\nBacked up previous log to {backup_name}")
        
        if os.path.exists("best_solution.log"):
            os.remove("best_solution.log")
    
    # Build bounds based on active features
    bounds = []
    
    # Nose radius (if enabled - for EITHER hemisphere or tangent sphere)
    if USE_NOSE_CAP or use_hemisphere:
        bounds.append(NOSE_RADIUS_BOUNDS)
    
    # 7 radius control points (normalized 0-1, will be scaled by max_radius)
    # FIXED: More reasonable bounds for cp6 and cp7
    cp_r_bounds = [
        (0.05, 0.8),   # cp1_r at x=0.05L
        (0.1, 0.8),    # cp2_r at x=0.15L
        (0.15, 0.8),   # cp3_r at x=0.25L
        (0.2, 1.0),    # cp4_r at x=0.40L
        (0.25, 1.0),   # cp5_r at x=0.55L
        (0.3, 1.0),    # cp6_r at x=0.75L - CHANGED: can explore lower values
        (0.4, 1.0),    # cp7_r at x=1.0L (end) - CHANGED: minimum raised to maintain shape
    ]
    bounds.extend(cp_r_bounds)
    
    # Basic geometry parameters
    bounds.extend([
        (0.5, 1.0),     # length (meters)
        (0.04, MAX_RADIUS_CONSTRAINT),   # max_radius (constrained)
    ])
    
    # Optional parameters
    if USE_Z_SQUASH:
        bounds.append((0.3, 1.0))  # z_squash: 1.0 = circular, <1 = flattened
    
    if USE_Z_CUT:
        #bounds.append((-0.1, 0))  # z_cut: negative z for bottom cut
        bounds.append(Z_CUT_PERCENT_BOUNDS)  # (0.3, 1.1)
    
    dim = len(bounds)
    vel_bounds = [(-(hi - lo) * 0.5, (hi - lo) * 0.5) for lo, hi in bounds]
    
    # Print configuration
    print("\n" + "="*70)
    print("OGIVE GEOMETRY OPTIMIZER CONFIGURATION")
    print("="*70)
    print(f"  Trajectory Solver: SIMPLE (shooting method)")
    print(f"  Control points: {len(FIXED_CP_X)} body points at {FIXED_CP_X}")
    if USE_NOSE_CAP or use_hemisphere:
        print(f"  Nose cap: ENABLED (range: {NOSE_RADIUS_BOUNDS[0]*1000:.1f}-{NOSE_RADIUS_BOUNDS[1]*1000:.1f} mm)")
        print(f"  Nose type: {'HEMISPHERE' if use_hemisphere else 'TANGENT SPHERE'}")
    print(f"  Maximum Radius Constraint: {MAX_RADIUS_CONSTRAINT} m")
    print(f"  Heat Flux Limits:")
    print(f"    - Optimizer cost:   {Q_DOT_LIMIT/1e6:.2f} MW/m²")
    print(f"    - Trajectory solver: {Q_DOT_LIMIT_TRAJECTORY/1e6:.2f} MW/m²")
    print(f"  Z-Squash (elliptical): {'ENABLED' if USE_Z_SQUASH else 'DISABLED'}")
    print(f"  Z-Cut (flat bottom):   {'ENABLED' if USE_Z_CUT else 'DISABLED'}")
    print(f"  Parameter dimensions:  {dim}")
    print(f"  Restart capability:    {'ENABLED' if ENABLE_RESTART else 'DISABLED'}")
    if ENABLE_RESTART:
        print(f"  Restart log file:      {RESTART_LOG_FILE}")
    print("="*70)
    
    # Show reference profiles
    if verbose:
        print("\nReference profiles at 7 control points:")
        cone = get_cone_profile()
        vk = get_von_karman_profile()
        print(f"{'x/L':<8} {'Cone':<10} {'von Karman':<12}")
        print("-" * 30)
        for i, x in enumerate(FIXED_CP_X):
            print(f"{x:<8.2f} {cone[i]:<10.4f} {vk[i]:<12.4f}")
        print()
    
    # Initialize swarm with restart capability
    swarm, restart_info = initialize_swarm_with_restart(
        num_particles, bounds, vel_bounds, RESTART_LOG_FILE
    )
    
    # Print initialization summary
    if restart_info['used_restart']:
        print(f"\n{'='*70}")
        print("INITIALIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"  From previous run: {restart_info['n_from_log']} particles")
        print(f"  Newly generated:   {restart_info['n_new']} particles")
        print(f"  Total particles:   {num_particles}")
        print(f"{'='*70}")
    else:
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
    
    # Find initial global best from swarm
    for p in swarm:
        if p.best_cost < gbest_cost:
            gbest_cost = p.best_cost
            gbest_pos = p.best_pos[:]
    
    skip_initial_eval = restart_info['used_restart'] and gbest_cost < float('inf')
    
    if not skip_initial_eval:
        print("\nEvaluating initial swarm...")
        for i, p in enumerate(swarm):
            if restart_info['used_restart'] and i < restart_info['n_from_log']:
                if verbose:
                    print(f"  Particle {i+1}/{num_particles}: cost = {p.best_cost:.6g} (from restart)")
                continue
            
            p.position = enforce_bounds(p.position, bounds)
            cost = evaluate_particle(p.position, verbose=verbose, particle_id=i+1, iteration=0)
            p.best_cost = cost
            p.best_pos = p.position[:]
            
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = p.position[:]
            
            if verbose:
                print(f"  Particle {i+1}/{num_particles}: cost = {cost:.6g}")
    else:
        print(f"\nUsing global best from restart: cost = {gbest_cost:.6g}")
    
    # Extract best parameters and write initial solution
    idx = 0
    if USE_NOSE_CAP or use_hemisphere:
        nose_radius = gbest_pos[idx]
        idx += 1
    else:
        nose_radius = None
    
    cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+7], nose_radius)
    best_params = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1],
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3],
        'cp5_r': cp_r_mono[4],
        'cp6_r': cp_r_mono[5],
        'cp7_r': cp_r_mono[6],
        'length': gbest_pos[idx+7],
        'max_radius': min(gbest_pos[idx+8], MAX_RADIUS_CONSTRAINT),
    }
    
    if USE_NOSE_CAP or use_hemisphere:
        best_params['nose_radius'] = nose_radius
    
    idx = idx + 9
    if USE_Z_SQUASH:
        best_params['z_squash'] = gbest_pos[idx]
        idx += 1
    else:
        best_params['z_squash'] = DEFAULT_Z_SQUASH
    
    if USE_Z_CUT:
        z_cut_percent = gbest_pos[idx]
        effective_radius = best_params['max_radius'] * best_params['z_squash']
        if z_cut_percent >= 1.0:
            best_params['z_cut'] = None
        else:
            best_params['z_cut'] = -effective_radius * z_cut_percent
        best_params['z_cut_percent'] = z_cut_percent
        idx += 1
    else:
        best_params['z_cut'] = DEFAULT_Z_CUT
        best_params['z_cut_percent'] = None
    
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
            
            p.velocity = clip_velocity(p.velocity, vel_bounds)
            p.position = enforce_bounds(
                [pi + vi for pi, vi in zip(p.position, p.velocity)], bounds
            )
            
            cost = evaluate_particle(p.position, verbose=verbose, 
                                    particle_id=p_idx+1, iteration=it+1)
            
            if cost < p.best_cost:
                p.best_cost = cost
                p.best_pos = p.position[:]
            
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = p.position[:]
                
                # Update best solution immediately
                idx = 0
                if USE_NOSE_CAP or use_hemisphere:
                    nose_radius = gbest_pos[idx]
                    idx += 1
                else:
                    nose_radius = None
                
                cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+7], nose_radius)
                best_params = {
                    'cp1_r': cp_r_mono[0],
                    'cp2_r': cp_r_mono[1],
                    'cp3_r': cp_r_mono[2],
                    'cp4_r': cp_r_mono[3],
                    'cp5_r': cp_r_mono[4],
                    'cp6_r': cp_r_mono[5],
                    'cp7_r': cp_r_mono[6],
                    'length': gbest_pos[idx+7],
                    'max_radius': min(gbest_pos[idx+8], MAX_RADIUS_CONSTRAINT),
                }
                
                if USE_NOSE_CAP or use_hemisphere:
                    best_params['nose_radius'] = nose_radius
                
                idx_param = idx + 9
                if USE_Z_SQUASH:
                    best_params['z_squash'] = gbest_pos[idx_param]
                    idx_param += 1
                else:
                    best_params['z_squash'] = DEFAULT_Z_SQUASH
                
                if USE_Z_CUT:
                    z_cut_percent = gbest_pos[idx_param]  # MUST BE FIRST
                    effective_radius = best_params['max_radius'] * best_params['z_squash']
                    if z_cut_percent >= 1.0:
                        best_params['z_cut'] = None
                    else:
                        best_params['z_cut'] = -effective_radius * z_cut_percent
                    best_params['z_cut_percent'] = z_cut_percent
                    idx_param += 1
                else:
                    best_params['z_cut'] = DEFAULT_Z_CUT
                    best_params['z_cut_percent'] = None
                
                est_range = -gbest_cost if gbest_cost < 0 else 0
                write_best_solution(best_params, it+1, gbest_cost, est_range/1000, 0, "best_solution.log")
        
        history.append(gbest_cost)
        
        if verbose:
            print(f"\nIteration {it+1} complete: best cost = {gbest_cost:.6g}")
    
    # Format final best solution
    idx = 0
    if USE_NOSE_CAP or use_hemisphere:
        nose_radius = gbest_pos[idx]
        idx += 1
    else:
        nose_radius = None
    
    cp_r_mono = enforce_monotonic_radii(gbest_pos[idx:idx+7], nose_radius)
    
    best = {
        'cp1_r': cp_r_mono[0],
        'cp2_r': cp_r_mono[1],
        'cp3_r': cp_r_mono[2],
        'cp4_r': cp_r_mono[3],
        'cp5_r': cp_r_mono[4],
        'cp6_r': cp_r_mono[5],
        'cp7_r': cp_r_mono[6],
    }
    
    if USE_NOSE_CAP or use_hemisphere:
        best['nose_radius'] = nose_radius
    
    best['length'] = gbest_pos[idx+7]
    best['max_radius'] = min(gbest_pos[idx+8], MAX_RADIUS_CONSTRAINT)
    
    idx = idx + 9
    if USE_Z_SQUASH:
        best['z_squash'] = gbest_pos[idx]
        idx += 1
    else:
        best['z_squash'] = DEFAULT_Z_SQUASH
    
    if USE_Z_CUT:
        z_cut_percent = gbest_pos[idx]  # MUST BE FIRST
        effective_radius = best['max_radius'] * best['z_squash']
        if z_cut_percent >= 1.0:
            best['z_cut'] = None
        else:
            best['z_cut'] = -effective_radius * z_cut_percent
        best['z_cut_percent'] = z_cut_percent
    else:
        best['z_cut'] = DEFAULT_Z_CUT
        best['z_cut_percent'] = None
    
    return best, -gbest_cost, history


def diagnose_restart_issue(log_file):
    """Debug function to see what's being parsed from restart log."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING RESTART FROM: {log_file}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    particles = parse_control_points_log(log_file)
    
    if not particles:
        print("❌ No particles parsed from log!")
        return
    
    print(f"✓ Successfully parsed {len(particles)} particles\n")
    
    # Show first 3 particles
    for i, p in enumerate(particles[:min(3, len(particles))]):
        print(f"Particle {i+1}:")
        print(f"  Cost: {p['cost']:.6g}")
        print(f"  Keys: {list(p['params'].keys())}")
        for key in ['length', 'max_radius', 'nose_radius', 'z_cut', 'z_cut_percent']:
            val = p['params'].get(key, 'MISSING')
            print(f"  {key}: {val}")
        print(f"  CPs: ", end="")
        for j in range(1, 8):
            cp = p['params'].get(f'cp{j}_r', 'X')
            print(f"{cp:.3f} " if cp != 'X' else "X ", end="")
        print("\n")
    
    # Test conversion
    print("Testing conversion to position vector...")
    print(f"Feature flags: nose={USE_NOSE_CAP or use_hemisphere}, "
          f"z_squash={USE_Z_SQUASH}, z_cut={USE_Z_CUT}")
    
    # Build bounds
    bounds = []
    if USE_NOSE_CAP or use_hemisphere:
        bounds.append(NOSE_RADIUS_BOUNDS)
    cp_r_bounds = [(0.05, 0.3), (0.1, 0.5), (0.15, 0.6), (0.2, 0.8), 
                   (0.25, 0.9), (0.3, 1.0), (0.4, 1.0)]
    bounds.extend(cp_r_bounds)
    bounds.extend([(0.5, 1.0), (0.04, MAX_RADIUS_CONSTRAINT)])
    if USE_Z_SQUASH:
        bounds.append((0.3, 0.7))
    if USE_Z_CUT:
        bounds.append(Z_CUT_PERCENT_BOUNDS)
    
    print(f"Expected position vector length: {len(bounds)}")
    
    try:
        pos = params_dict_to_position(particles[0]['params'], bounds)
        print(f"✓ Conversion successful! Position length: {len(pos)}")
        print(f"  Values: {[f'{p:.4f}' for p in pos]}")
    except Exception as e:
        print(f"❌ Conversion FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*70)
    print("OGIVE GEOMETRY OPTIMIZER (7 Control Points + Nose Cap)")
    print("="*70)

    if ENABLE_RESTART and os.path.exists(RESTART_LOG_FILE):
        diagnose_restart_issue(RESTART_LOG_FILE)
        #input("\nPress Enter to continue with optimization...")
    
    best, best_range, hist = pso_optimize(
        num_particles=2, 
        iterations=5, 
        seed=42, 
        verbose=True
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest Range: {best_range/1000:.3f} km")
    print(f"\nOptimal Parameters:")
    print(f"  Length:     {best['length']:.6f} m")
    print(f"  Max Radius: {best['max_radius']:.6f} m")
    
    if USE_NOSE_CAP or use_hemisphere:
        print(f"  Nose Radius: {best.get('nose_radius', DEFAULT_NOSE_RADIUS)*1000:.3f} mm")
    
    if USE_Z_SQUASH:
        print(f"  Z-Squash:   {best['z_squash']:.6f}")
    if USE_Z_CUT and best['z_cut'] is not None:
        print(f"  Z-Cut:      {best['z_cut']:.6f} m")
    
    print(f"\n  Control Point Radii (normalized):")
    for i, cp_name in enumerate(['cp1_r', 'cp2_r', 'cp3_r', 'cp4_r', 'cp5_r', 'cp6_r', 'cp7_r']):
        print(f"    {cp_name}: {best[cp_name]:.6f} (at x={FIXED_CP_X[i]}L)")
    
    print(f"\nLog files generated:")
    print(f"  - control_points.log: All particle evaluations")
    print(f"  - best_solution.log: Current best solution")
    
    if ENABLE_RESTART:
        print(f"\nRESTART MODE: Run again to continue optimization")