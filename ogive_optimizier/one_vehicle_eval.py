#!/usr/bin/env python3
"""
evaluate_vehicle.py

Standalone script to evaluate a specific vehicle configuration.
This script evaluates aerodynamic performance, trajectory, and constraints
for a user-specified parametric body geometry.

Usage:
    python evaluate_vehicle.py

Author: Based on ogive optimizer
"""

import random
import math
import time
import os
import subprocess
import shutil
import numpy as np
import trimesh as tm
from fit_optimizer_3 import WaveriderCase
from integrated_traj_test import run_dymos_optimization
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
DEFAULT_NOSE_RADIUS = 0.01  # 10mm default nose radius

# Fixed control point x-positions (as fractions of length)
FIXED_CP_X = [0.05, 0.15, 0.25, 0.40, 0.55, 0.75, 1.0]  # 7 control points

# GLOBAL MAXIMUM RADIUS CONSTRAINT (meters)
MAX_RADIUS_CONSTRAINT = 0.1  # Never exceed this radius

# HEAT FLUX LIMITS (W/m²)
Q_DOT_LIMIT = 1.3e6  # Cost function limit: 1.3 MW/m²
Q_DOT_LIMIT_TRAJECTORY = 1.3e6  # Trajectory solver limit

# MINIMUM VOLUME CONSTRAINT
MIN_VOLUME_LITERS = 6.0  # Minimum internal volume constraint (liters)

# MACH EVALUATION RANGE
MACH_RANGE = [7.5, 7.75, 8.0]
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
        cross = np.cross(v1, v2)
        volume += np.dot(v0, cross)
    
    # Divide by 6 to get actual volume
    volume = abs(volume) / 6.0
    
    return volume


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
    
    if triangles.min() < 0 or triangles.max() >= n_verts:
        raise ValueError(f"Invalid triangle indices: range [{triangles.min()}, {triangles.max()}] "
                        f"but only {n_verts} vertices (valid range [0, {n_verts-1}])")
    
    return vertices, triangles


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
    
    print("Computing reference parameters from mesh...")
    vertices, triangles = read_tri_file(tri_dest)
    
    geom_params = compute_reference_parameters(vertices, triangles, verbose=False)
    
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
        
    except Exception as e:
        print(f"✗ CBAERO failed: {e}")
        raise
    finally:
        os.chdir(current_dir)


def calculate_ogive_range(body):
    """Calculate range for ogive geometry using simple trajectory solver."""
    path = os.getcwd()
    max_range, max_q_dot = run_dymos_optimization(
        path, 
        plotting=True, 
        surrogate_type="linear", 
        #q_dot_limit=Q_DOT_LIMIT_TRAJECTORY,
        #mach_range=MACH_RANGE
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


def enforce_monotonic_radii(cp_r, nose_radius=None):
    """Ensure radius values are monotonically non-decreasing."""
    if nose_radius is not None:
        cp_r_mono = [max(cp_r[0], nose_radius)]
    else:
        cp_r_mono = [cp_r[0]]
    
    for i in range(1, len(cp_r)):
        cp_r_mono.append(max(cp_r[i], cp_r_mono[i-1]))
    return cp_r_mono


def get_von_karman_profile():
    """Get radius profile for a von Karman ogive at fixed x-positions (7 points)."""
    return [np.sqrt(2*x - x**2) if x <= 1.0 else 1.0 for x in FIXED_CP_X]


def evaluate_specific_vehicle(length, max_radius, cp_radii, z_squash=None, z_cut=None, 
                              nose_radius=None, verbose=True):
    """
    Evaluate a specific vehicle configuration.
    
    Args:
        length: Vehicle length in meters
        max_radius: Maximum radius in meters
        cp_radii: List of 7 normalized radii (0-1) for control points at FIXED_CP_X positions
        z_squash: Optional z-squash factor (default: 1.0 for circular cross-section)
        z_cut: Optional z-cut value in meters (negative for bottom cut, or None)
        nose_radius: Optional nose radius in meters (required if USE_NOSE_CAP or use_hemisphere is True)
        verbose: Print detailed output
    
    Returns:
        dict with keys:
            - 'range_m': Range in meters
            - 'range_km': Range in kilometers  
            - 'max_q_dot': Maximum heat flux in W/m²
            - 'max_q_dot_kw': Maximum heat flux in kW/m²
            - 'cost': Cost function value
            - 'volume_m3': Internal volume in cubic meters
            - 'volume_liters': Internal volume in liters
            - 'fits_payload': Boolean, whether payload fits
            - 'satisfies_constraints': Boolean, all constraints satisfied
            - 'params': Dictionary of all parameters used
    """
    
    # Validate inputs
    if len(cp_radii) != 7:
        raise ValueError(f"cp_radii must have exactly 7 elements, got {len(cp_radii)}")
    
    if any(r < 0 or r > 1 for r in cp_radii):
        raise ValueError("All cp_radii values must be between 0 and 1 (normalized)")
    
    if length <= 0 or max_radius <= 0:
        raise ValueError("length and max_radius must be positive")
    
    if max_radius > MAX_RADIUS_CONSTRAINT:
        print(f"Warning: max_radius {max_radius} exceeds MAX_RADIUS_CONSTRAINT {MAX_RADIUS_CONSTRAINT}, clamping")
        max_radius = MAX_RADIUS_CONSTRAINT
    
    # Handle optional parameters
    if z_squash is None:
        z_squash = DEFAULT_Z_SQUASH if not USE_Z_SQUASH else 1.0
    
    if z_cut is None:
        z_cut = DEFAULT_Z_CUT
    
    if (USE_NOSE_CAP or use_hemisphere) and nose_radius is None:
        print(f"Warning: Nose cap enabled but no nose_radius provided, using default {DEFAULT_NOSE_RADIUS*1000:.1f}mm")
        nose_radius = DEFAULT_NOSE_RADIUS
    
    if nose_radius is not None and nose_radius > MAX_RADIUS_CONSTRAINT:
        print(f"Warning: nose_radius {nose_radius} exceeds MAX_RADIUS_CONSTRAINT, clamping")
        nose_radius = MAX_RADIUS_CONSTRAINT
    
    # Enforce monotonic radii
    cp_r_mono = enforce_monotonic_radii(cp_radii, nose_radius)
    
    # Build parameter dictionary
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
    }
    
    if USE_NOSE_CAP or use_hemisphere:
        params['nose_radius'] = nose_radius
    
    # Create working directory
    os.makedirs('temp_evaluation', exist_ok=True)
    case_name = f"eval-L{length:.3f}-R{max_radius:.3f}"
    if nose_radius is not None:
        case_name += f"-nr{nose_radius*1000:.1f}mm"
    if USE_Z_SQUASH:
        case_name += f"-zs{z_squash:.3f}"
    if USE_Z_CUT and z_cut is not None:
        case_name += f"-zc{abs(z_cut):.3f}"
    
    case_dir = os.path.join('temp_evaluation', case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(case_dir)
    
    results = {
        'range_m': None,
        'range_km': None,
        'max_q_dot': None,
        'max_q_dot_kw': None,
        'cost': None,
        'volume_m3': None,
        'volume_liters': None,
        'fits_payload': False,
        'satisfies_constraints': False,
        'params': params,
        'error': None
    }
    
    try:
        # [1/5] Set up geometry
        if verbose:
            print(f"\n{'='*70}")
            print(f"EVALUATING VEHICLE: {case_name}")
            print(f"{'='*70}")
            print(f"[1/5] Setting up geometry...")
            print(f"      Length: {length:.4f} m")
            print(f"      Max radius: {max_radius:.4f} m")
            if nose_radius is not None:
                print(f"      Nose radius: {nose_radius*1000:.2f} mm ({'HEMISPHERE' if use_hemisphere else 'TANGENT SPHERE'})")
            if USE_Z_SQUASH:
                print(f"      Z-squash: {z_squash:.4f}")
            if USE_Z_CUT and z_cut is not None:
                print(f"      Z-cut: {z_cut:.4f} m")
        
        body, stl_filename, tri_filename = set_up_ogive(params)
        
        if verbose:
            print(f"      ✓ Geometry created")
        
        # [2/5] Compute volume
        if verbose:
            print(f"[2/5] Computing volume...")
        
        vertices, triangles = read_tri_file(tri_filename)
        volume_m3 = compute_volume(vertices, triangles)
        volume_liters = volume_m3 * 1000.0
        
        results['volume_m3'] = volume_m3
        results['volume_liters'] = volume_liters
        
        if verbose:
            print(f"      Volume: {volume_liters:.2f} liters ({volume_m3:.6f} m³)")
        
        if volume_liters < MIN_VOLUME_LITERS:
            if verbose:
                print(f"      ✗ Volume constraint violated: {volume_liters:.2f}L < {MIN_VOLUME_LITERS}L")
            results['error'] = f"Volume too small: {volume_liters:.2f}L < {MIN_VOLUME_LITERS}L"
            results['cost'] = 1e7 + (MIN_VOLUME_LITERS - volume_liters) * 1e6
            os.chdir(original_dir)
            return results
        
        # [3/5] Check payload fit
        if verbose:
            print(f"[3/5] Checking payload fit...")
        
        inst = WaveriderCase(stl_filename)
        fits = inst.try_fit_payload(verbose=False)
        results['fits_payload'] = fits[0]
        
        if not fits[0]:
            if verbose:
                print(f"      ✗ Geometry does not fit payload, residual: {fits[1]:.6f}")
            results['error'] = f"Payload fit failed, residual: {fits[1]:.6f}"
            results['cost'] = 1e8 + abs(fits[1]) * 1e6
            os.chdir(original_dir)
            return results
        
        if verbose:
            print(f"      ✓ Payload fits")
        
        # [4/5] Compute aerodatabase
        if verbose:
            print(f"[4/5] Computing aerodynamic database...")
        
        compute_aerodatabase('ogive.tri')
        
        if verbose:
            print(f"      ✓ Aerodatabase computed")
        
        # [5/5] Calculate trajectory and range
        if verbose:
            print(f"[5/5] Computing trajectory...")
        
        rng, max_q_dot = calculate_ogive_range(body)
        
        results['range_m'] = rng
        results['range_km'] = rng / 1000.0
        results['max_q_dot'] = max_q_dot
        results['max_q_dot_kw'] = max_q_dot / 1000.0
        
        # Compute cost
        cost = cost_function(rng, max_q_dot, q_dot_limit=Q_DOT_LIMIT)
        results['cost'] = cost
        
        # Check if all constraints satisfied
        q_dot_ok = max_q_dot <= Q_DOT_LIMIT
        results['satisfies_constraints'] = q_dot_ok
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*70}")
            print(f"  Range:          {results['range_km']:.3f} km")
            print(f"  Max heat flux:  {results['max_q_dot_kw']:.1f} kW/m²")
            print(f"  Q_dot limit:    {Q_DOT_LIMIT/1000:.1f} kW/m²")
            print(f"  Q_dot OK:       {'✓ YES' if q_dot_ok else '✗ NO (exceeds limit)'}")
            print(f"  Volume:         {results['volume_liters']:.2f} liters")
            print(f"  Payload fits:   {'✓ YES' if results['fits_payload'] else '✗ NO'}")
            print(f"  Cost:           {cost:.6g}")
            print(f"  All constraints: {'✓ SATISFIED' if results['satisfies_constraints'] else '✗ VIOLATED'}")
            print(f"{'='*70}\n")
        
        os.chdir(original_dir)
        return results
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        results['error'] = str(e)
        results['cost'] = 1e9
        os.chdir(original_dir)
        return results


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("="*70)
    print("PARAMETRIC VEHICLE EVALUATOR")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Control points: {len(FIXED_CP_X)} at x/L = {FIXED_CP_X}")
    print(f"  Nose cap: {'HEMISPHERE' if use_hemisphere else 'TANGENT SPHERE' if USE_NOSE_CAP else 'SHARP'}")
    print(f"  Z-squash (elliptical): {'ENABLED' if USE_Z_SQUASH else 'DISABLED'}")
    print(f"  Z-cut (flat bottom): {'ENABLED' if USE_Z_CUT else 'DISABLED'}")
    print(f"  Min volume: {MIN_VOLUME_LITERS} liters")
    print(f"  Max radius: {MAX_RADIUS_CONSTRAINT} m")
    print(f"  Q_dot limit: {Q_DOT_LIMIT/1e6:.2f} MW/m²")
    print("="*70)
    
    print("\n\n" + "="*70)
    print("MY CUSTOM VEHICLE")
    print("="*70)
    
    my_results = evaluate_specific_vehicle(
        length=1.0,               # Vehicle length in meters
        max_radius=0.1,          # Maximum body radius in meters
        cp_radii=[                # 7 normalized radii (0-1) at x/L = [0.05, 0.15, 0.25, 0.40, 0.55, 0.75, 1.0]
            0.238960,   # cp1 at x/L = 0.05
            0.344643,   # cp2 at x/L = 0.15
            0.434806,   # cp3 at x/L = 0.25
            0.600175,   # cp4 at x/L = 0.40
            0.785702,   # cp5 at x/L = 0.55
            1.0,   # cp6 at x/L = 0.75
            1.0,   # cp7 at x/L = 1.0 (rear)
        ],
        z_squash=0.42677,             # 1.0 = circular, <1.0 = elliptical (flattened in z)
        z_cut=-0.025,               # Flat bottom cut (negative z value in meters, or None)
        nose_radius=0.007,         # Nose cap radius in meters
        verbose=True
    )
    
    # Access results:
    if my_results['satisfies_constraints']:
        print(f"\n✓ Vehicle satisfies all constraints!")
        print(f"  Range: {my_results['range_km']:.2f} km")
        print(f"  Volume: {my_results['volume_liters']:.2f} liters")
    else:
        print(f"\n✗ Vehicle violates constraints")
        if my_results['error']:
            print(f"  Error: {my_results['error']}")
    
    # ============ SUMMARY ============
    print("\n\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    

    print(f"\n{'Configuration':<25} {'Range (km)':<15} {'Q_dot (kW/m²)':<15} {'Volume (L)':<15} {'Status':<20}")
    print("-" * 90)
    
    for name, res in examples:
        range_str = f"{res['range_km']:.2f}" if res['range_km'] is not None else "N/A"
        q_dot_str = f"{res['max_q_dot_kw']:.1f}" if res['max_q_dot_kw'] is not None else "N/A"
        vol_str = f"{res['volume_liters']:.2f}" if res['volume_liters'] is not None else "N/A"
        status = "✓ PASS" if res['satisfies_constraints'] else "✗ FAIL"
        
        print(f"{name:<25} {range_str:<15} {q_dot_str:<15} {vol_str:<15} {status:<20}")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)