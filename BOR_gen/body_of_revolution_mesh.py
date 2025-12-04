"""
Body of Revolution Mesh Generator

Generate triangulated surface meshes from radius functions r(x).
Supports any axisymmetric body defined by r = f(x).
"""

import numpy as np
from scipy.interpolate import interp1d


def generate_bor_point_cloud(r_func, x_range, n_axial=100, n_circumferential=60, 
                              r_func_args=None, closure='both'):
    """
    Generate point cloud for a body of revolution.
    
    Parameters
    ----------
    r_func : callable
        Function r(x, *args) returning radius at axial position x
    x_range : tuple
        (x_min, x_max) - axial extent
    n_axial : int
        Number of points along axis
    n_circumferential : int
        Number of points around circumference
    r_func_args : dict, optional
        Additional arguments to pass to r_func
    closure : str
        'both' - Close both ends (nose and tail)
        'nose' - Close only nose (x_min)
        'tail' - Close only tail (x_max)
        'none' - Leave open (cylinder-like)
    
    Returns
    -------
    points : np.ndarray (n_points, 3)
        Point cloud in Cartesian coordinates
    grid : dict
        Structured grid information for meshing
    """
    
    if r_func_args is None:
        r_func_args = {}
    
    x_min, x_max = x_range
    
    # Generate axial stations
    x = np.linspace(x_min, x_max, n_axial)
    
    # Compute radius at each station
    r = np.array([r_func(xi, **r_func_args) for xi in x])
    
    # Check for valid radii
    if np.any(r < 0):
        raise ValueError("Radius function returned negative values")
    
    # Generate circumferential angles
    theta = np.linspace(0, 2*np.pi, n_circumferential, endpoint=False)
    
    # Create meshgrid
    X, Theta = np.meshgrid(x, theta)
    R = np.array([[r_func(xi, **r_func_args) for xi in x] for _ in theta])
    
    # Convert to Cartesian (X-axis is axial direction)
    # Y-Z plane is the rotation plane
    X_cart = X
    Y_cart = R * np.cos(Theta)
    Z_cart = R * np.sin(Theta)
    
    # Stack into point cloud
    points = np.stack([X_cart.ravel(), Y_cart.ravel(), Z_cart.ravel()], axis=1)
    
    # Store grid information for meshing
    grid = {
        'x': x,
        'theta': theta,
        'r': r,
        'n_axial': n_axial,
        'n_circumferential': n_circumferential,
        'X': X_cart,
        'Y': Y_cart,
        'Z': Z_cart,
        'closure': closure
    }
    
    return points, grid


def triangulate_bor_surface(grid, add_nose_cap=True, add_tail_cap=True):
    """
    Triangulate body of revolution surface from structured grid.
    
    Parameters
    ----------
    grid : dict
        Output from generate_bor_point_cloud
    add_nose_cap : bool
        Add triangulated cap at nose (x_min)
    add_tail_cap : bool
        Add triangulated cap at tail (x_max)
    
    Returns
    -------
    vertices : np.ndarray (n_verts, 3)
    triangles : np.ndarray (n_tris, 3)
    """
    
    n_axial = grid['n_axial']
    n_circ = grid['n_circumferential']
    
    # Extract Cartesian coordinates
    X = grid['X']
    Y = grid['Y']
    Z = grid['Z']
    
    # Flatten to vertices
    vertices_body = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Generate triangles for body surface
    triangles_body = []
    
    for i in range(n_circ):
        i_next = (i + 1) % n_circ  # Wrap around
        
        for j in range(n_axial - 1):
            # Quad vertices (in body grid)
            v0 = i * n_axial + j
            v1 = i * n_axial + (j + 1)
            v2 = i_next * n_axial + j
            v3 = i_next * n_axial + (j + 1)
            
            # Two triangles per quad
            # FIXED: Reversed winding for outward normals
            triangles_body.append([v0, v1, v2])
            triangles_body.append([v2, v1, v3])
    
    triangles_body = np.array(triangles_body)
    
    # Start building combined mesh
    all_vertices = [vertices_body]
    all_triangles = [triangles_body]
    
    # Add nose cap if requested
    if add_nose_cap:
        r_nose = grid['r'][0]
        
        if r_nose < 1e-10:
            # Pointed nose - create cone
            # Add apex vertex
            x_nose = grid['x'][0]
            apex = np.array([[x_nose, 0, 0]])
            apex_idx = len(vertices_body)  # Index in combined array
            
            all_vertices.append(apex)
            
            # Triangles from apex to first ring
            nose_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial  # First ring
                v2 = i_next * n_axial
                # FIXED: Reversed winding for outward normal (forward direction)
                nose_tris.append([apex_idx, v2, v1])
            
            all_triangles.append(np.array(nose_tris))
        else:
            # Flat nose - create disk
            x_nose = grid['x'][0]
            center = np.array([[x_nose, 0, 0]])
            center_idx = len(vertices_body)  # Index in combined array
            
            all_vertices.append(center)
            
            # Triangles from center to first ring
            nose_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial
                v2 = i_next * n_axial
                # FIXED: Reversed winding for outward normal (forward direction)
                nose_tris.append([center_idx, v1, v2])
            
            all_triangles.append(np.array(nose_tris))
    
    if add_tail_cap:
        r_tail = grid['r'][-1]
        
        # Current vertex count
        n_verts_so_far = sum(len(v) for v in all_vertices)
        
        if r_tail < 1e-10:
            # Pointed tail - create cone
            x_tail = grid['x'][-1]
            apex = np.array([[x_tail, 0, 0]])
            apex_idx = n_verts_so_far
            
            all_vertices.append(apex)
            
            # Triangles from last ring to apex
            tail_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial + (n_axial - 1)
                v2 = i_next * n_axial + (n_axial - 1)
                # FIXED: Reversed winding for outward normal (aft direction)
                tail_tris.append([apex_idx, v1, v2])
            
            all_triangles.append(np.array(tail_tris))
        else:
            # Flat tail - create disk
            x_tail = grid['x'][-1]
            center = np.array([[x_tail, 0, 0]])
            center_idx = n_verts_so_far
            
            all_vertices.append(center)
            
            # Triangles from last ring to center
            tail_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial + (n_axial - 1)
                v2 = i_next * n_axial + (n_axial - 1)
                # FIXED: Reversed winding for outward normal (aft direction)
                tail_tris.append([center_idx, v2, v1])
            
            all_triangles.append(np.array(tail_tris))
    
    # Combine all vertices and triangles
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    
    return vertices, triangles


def generate_bor_mesh(r_func, x_range, n_axial=100, n_circumferential=60,
                      r_func_args=None, add_nose_cap=True, add_tail_cap=True,
                      merge_tolerance=1e-9, improve_quality=False):
    """
    Complete pipeline: generate mesh for body of revolution.
    
    Parameters
    ----------
    r_func : callable
        Radius function r(x, **kwargs)
    x_range : tuple
        (x_min, x_max)
    n_axial : int
        Axial resolution
    n_circumferential : int
        Circumferential resolution
    r_func_args : dict
        Additional arguments for r_func
    add_nose_cap : bool
        Close nose end
    add_tail_cap : bool
        Close tail end
    merge_tolerance : float
        Tolerance for merging duplicate vertices
    improve_quality : bool
        Apply quality improvement
    
    Returns
    -------
    vertices : np.ndarray
    triangles : np.ndarray
    stats : dict
    """
    
    print("\n" + "="*70)
    print("GENERATING BODY OF REVOLUTION MESH")
    print("="*70)
    
    if r_func_args is None:
        r_func_args = {}
    
    # Step 1: Generate point cloud
    print(f"\n[1/5] Generating point cloud...")
    print(f"  Axial points: {n_axial}")
    print(f"  Circumferential points: {n_circumferential}")
    
    points, grid = generate_bor_point_cloud(
        r_func, x_range, n_axial, n_circumferential, r_func_args
    )
    
    print(f"  Generated {len(points)} points")
    print(f"  X range: [{grid['x'].min():.6f}, {grid['x'].max():.6f}]")
    print(f"  R range: [{grid['r'].min():.6f}, {grid['r'].max():.6f}]")
    
    # Step 2: Triangulate
    print(f"\n[2/5] Triangulating surface...")
    print(f"  Nose cap: {'Yes' if add_nose_cap else 'No'}")
    print(f"  Tail cap: {'Yes' if add_tail_cap else 'No'}")
    
    vertices, triangles = triangulate_bor_surface(grid, add_nose_cap, add_tail_cap)
    
    print(f"  Initial: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Step 3: Merge duplicates
    print(f"\n[3/5] Merging duplicate vertices...")
    
    from waverider_manual_mesh import merge_duplicate_vertices
    vertices_unique, index_map = merge_duplicate_vertices(vertices, merge_tolerance)
    triangles = index_map[triangles]
    
    n_merged = len(vertices) - len(vertices_unique)
    print(f"  Merged {n_merged} duplicate vertices")
    print(f"  Result: {len(vertices_unique)} unique vertices")
    
    vertices = vertices_unique
    
    # Step 4: Remove degenerates
    print(f"\n[4/5] Removing degenerate triangles...")
    
    from waverider_manual_mesh import (
        remove_degenerate_triangles, 
        remove_duplicate_triangles,
        compute_triangle_quality
    )
    
    n_before = len(triangles)
    triangles = remove_degenerate_triangles(triangles, vertices, 1e-10)
    triangles = remove_duplicate_triangles(triangles)
    n_removed = n_before - len(triangles)
    
    print(f"  Removed {n_removed} bad triangles")
    print(f"  Result: {len(triangles)} valid triangles")
    
    # Step 5: Quality analysis
    print(f"\n[5/5] Computing quality metrics...")
    
    qualities = compute_triangle_quality(triangles, vertices)
    
    # Check watertightness
    from collections import defaultdict
    edge_count = defaultdict(int)
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_count[edge] += 1
    
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_count.values() if count > 2)
    
    stats = {
        'n_vertices': len(vertices),
        'n_triangles': len(triangles),
        'n_merged_vertices': n_merged,
        'n_removed_triangles': n_removed,
        'min_area': qualities['area'].min(),
        'max_area': qualities['area'].max(),
        'mean_aspect_ratio': qualities['aspect_ratio'].mean(),
        'max_aspect_ratio': qualities['aspect_ratio'].max(),
        'mean_skewness': qualities['skewness'].mean(),
        'max_skewness': qualities['skewness'].max(),
        'poor_quality_count': (qualities['skewness'] > 0.9).sum(),
        'boundary_edges': boundary_edges,
        'non_manifold_edges': non_manifold_edges,
        'is_watertight': boundary_edges == 0,
        'is_manifold': non_manifold_edges == 0
    }
    
    print(f"\n  Quality metrics:")
    print(f"    Mean aspect ratio: {stats['mean_aspect_ratio']:.2f}")
    print(f"    Mean skewness: {stats['mean_skewness']:.3f}")
    print(f"    Poor quality: {stats['poor_quality_count']} triangles")
    
    print(f"\n  Topology:")
    print(f"    Boundary edges: {boundary_edges}")
    print(f"    Watertight: {'✓ Yes' if stats['is_watertight'] else '✗ No'}")
    print(f"    Manifold: {'✓ Yes' if stats['is_manifold'] else '✗ No'}")
    
    # Optional quality improvement
    if improve_quality and stats['poor_quality_count'] > 0:
        print(f"\n[OPTIONAL] Improving mesh quality...")
        from mesh_quality_optimizer import advanced_quality_improvement
        vertices, triangles = advanced_quality_improvement(
            vertices, triangles,
            edge_swap_iters=5,
            smoothing_iters=3,
            smoothing_factor=0.2
        )
        print(f"  Result: {len(triangles)} triangles after optimization")
    
    return vertices, triangles, stats


def write_tri_file(filename, vertices, triangles, scale=1.0, swap_yz=True):
    """
    Write mesh to Cart3D .tri format.
    
    Parameters
    ----------
    filename : str
    vertices : np.ndarray
    triangles : np.ndarray
    scale : float
        Scale factor (default: 1.0 for no scaling)
    swap_yz : bool
        Swap Y and Z for Cart3D convention
    """
    verts = vertices.copy() * scale
    
    if swap_yz:
        verts = verts[:, [0, 2, 1]]
    
    n_verts = len(verts)
    n_tris = len(triangles)
    
    with open(filename, 'w') as f:
        f.write(f"{n_verts} {n_tris}\n")
        for v in verts:
            f.write(f"    {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in triangles:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        for _ in range(n_tris):
            f.write("1\n")
    
    print(f"\n✓ Wrote {filename}")
    print(f"  {n_verts} vertices, {n_tris} triangles")


# ============================================================================
# EXAMPLE RADIUS FUNCTIONS
# ============================================================================

def r_cone(x, length, base_radius):
    """Simple cone: r = (base_radius / length) * x"""
    return (base_radius / length) * x


def r_cylinder(x, length, radius):
    """Cylinder: constant radius"""
    return radius


def r_sphere(x, length, radius):
    """Sphere section"""
    x_center = length / 2
    r_local = radius**2 - (x - x_center)**2
    return np.sqrt(max(0, r_local))


def r_ogive(x, length, base_radius):
    """Tangent ogive (rocket nose cone)"""
    rho = (base_radius**2 + length**2) / (2 * base_radius)
    y = rho - np.sqrt(rho**2 - x**2)
    return y


def r_power_law(x, length, base_radius, n=0.5):
    """Power law: r = base_radius * (x/length)^n"""
    return base_radius * (x / length)**n


def r_haack_series(x, length, base_radius, C=0):
    """
    Haack series (minimum drag nose cone).
    C = 0: LD-Haack (minimum drag for given length and diameter)
    C = 1/3: LV-Haack (minimum drag for given length and volume)
    """
    theta = np.arccos(1 - 2*x/length)
    r = (base_radius / np.sqrt(np.pi)) * np.sqrt(
        theta - np.sin(2*theta)/2 + C * np.sin(theta)**3
    )
    return r


def r_ellipse(x, length, base_radius):
    """Elliptical profile"""
    a = length
    b = base_radius
    return b * np.sqrt(1 - (x/a)**2)


def r_parabola(x, length, base_radius):
    """Parabolic profile"""
    return base_radius * (1 - (1 - x/length)**2)


def r_biconic(x, length, base_radius, x_junction=None, r_junction=None):
    """
    Biconic: two cones joined together.
    Common for reentry vehicles.
    """
    if x_junction is None:
        x_junction = length * 0.3
    if r_junction is None:
        r_junction = base_radius * 0.4
    
    if x <= x_junction:
        # First cone
        return (r_junction / x_junction) * x
    else:
        # Second cone
        return r_junction + (base_radius - r_junction) / (length - x_junction) * (x - x_junction)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("Body of Revolution Mesh Generator")
    print("="*70)
    print("\nAvailable example shapes:")
    print("  1. Cone")
    print("  2. Cylinder")
    print("  3. Sphere")
    print("  4. Ogive (rocket nose)")
    print("  5. Power law (n=0.5)")
    print("  6. Haack series (minimum drag)")
    print("  7. Ellipse")
    print("  8. Parabola")
    print("  9. Biconic (two cones)")
    
    # Example: Generate a cone
    print("\n" + "="*70)
    print("EXAMPLE: Generating cone mesh")
    print("="*70)
    
    # Define parameters
    length = 10.0  # m
    base_radius = 1.0  # m
    
    # Generate mesh
    vertices, triangles, stats = generate_bor_mesh(
        r_func=r_cone,
        x_range=(0, length),
        n_axial=50,
        n_circumferential=40,
        r_func_args={'length': length, 'base_radius': base_radius},
        add_nose_cap=True,  # Pointed nose
        add_tail_cap=True,  # Flat base
        improve_quality=True
    )
    
    # Write output
    write_tri_file("cone.tri", vertices, triangles)
    
    print("\n" + "="*70)
    print("MESH GENERATION COMPLETE")
    print("="*70)
    print(f"\nFinal statistics:")
    for key, value in stats.items():
        if isinstance(value, bool):
            print(f"  {key}: {'✓' if value else '✗'}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Ready for panel code analysis!")
    print("  Generated: cone.tri")