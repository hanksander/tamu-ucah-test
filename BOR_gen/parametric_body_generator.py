"""
Parametric Body Generator with advanced features:
1. Flat bottom cut plane (Z = z_cut)
2. Elliptical cross-section (squash in Z direction)
3. Spline-based radius definition
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from body_of_revolution_mesh import write_tri_file
from waverider_manual_mesh import (
    merge_duplicate_vertices,
    remove_degenerate_triangles,
    remove_duplicate_triangles,
    compute_triangle_quality
)
from collections import defaultdict


class ParametricBody:
    """
    Parametric body with spline-based radius, cut plane, and elliptical squash.
    
    Parameters
    ----------
    length : float
        Total length of body
    control_points : list of tuples
        [(x1, r1), (x2, r2), ...] defining radius at axial positions
        x values should be normalized 0-1 (will be scaled by length)
    z_cut : float, optional
        Z-coordinate for flat bottom cut plane (None = no cut)
    z_squash : float, optional
        Squash factor in Z direction (1.0 = circular, <1.0 = flattened)
        Creates elliptical cross-section with semi-axes (r, r*z_squash)
    spline_order : int
        Order of spline interpolation (1=linear, 3=cubic)
    """
    
    def __init__(self, length, control_points, z_cut=None, z_squash=1.0, 
                 spline_order=3, name="Parametric Body"):
        self.length = length
        self.control_points = np.array(control_points)
        self.z_cut = z_cut
        self.z_squash = z_squash
        self.spline_order = spline_order
        self.name = name
        
        # Create spline for radius function
        x_normalized = self.control_points[:, 0]
        r_values = self.control_points[:, 1]
        
        # Ensure x is sorted
        sort_idx = np.argsort(x_normalized)
        x_normalized = x_normalized[sort_idx]
        r_values = r_values[sort_idx]
        
        # Create spline (use min order if not enough points)
        k = min(spline_order, len(x_normalized) - 1)
        
        if k == 1:
            # Linear interpolation
            self.r_spline = interp1d(x_normalized, r_values, 
                                     kind='linear', fill_value='extrapolate')
        else:
            # Spline interpolation
            self.r_spline = UnivariateSpline(x_normalized, r_values, k=k, s=0)
    
    def r(self, x):
        """Radius at axial position x (in absolute coordinates)."""
        x_norm = x / self.length
        return float(self.r_spline(x_norm))
    
    def r_y(self, x):
        """Y semi-axis at position x (circular cross-section)."""
        return self.r(x)
    
    def r_z(self, x):
        """Z semi-axis at position x (with squash applied)."""
        return self.r(x) * self.z_squash


def generate_parametric_body_point_cloud(body, n_axial=100, n_circumferential=60):
    """
    Generate point cloud for parametric body with elliptical cross-section.
    
    Parameters
    ----------
    body : ParametricBody
        Body definition
    n_axial : int
        Axial resolution
    n_circumferential : int
        Circumferential resolution
    
    Returns
    -------
    points : np.ndarray
    grid : dict
    """
    x = np.linspace(0, body.length, n_axial)
    theta = np.linspace(0, 2*np.pi, n_circumferential, endpoint=False)
    
    # Create meshgrid
    X, Theta = np.meshgrid(x, theta)
    
    # Compute semi-axes at each x position
    R_y = np.array([[body.r_y(xi) for xi in x] for _ in theta])
    R_z = np.array([[body.r_z(xi) for xi in x] for _ in theta])
    
    # Elliptical cross-section
    Y = R_y * np.cos(Theta)
    Z = R_z * np.sin(Theta)
    
    # Track which points are on cut plane (before modification)
    on_cut_plane = None
    if body.z_cut is not None:
        on_cut_plane = Z < body.z_cut
        # Project points below z_cut onto the plane
        Z[on_cut_plane] = body.z_cut
    
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    grid = {
        'x': x,
        'theta': theta,
        'n_axial': n_axial,
        'n_circumferential': n_circumferential,
        'X': X,
        'Y': Y,
        'Z': Z,
        'has_cut': body.z_cut is not None,
        'z_cut': body.z_cut,
        'z_squash': body.z_squash,
        'on_cut_plane': on_cut_plane  # Track which points were projected
    }
    
    return points, grid


def triangulate_parametric_body(grid, add_nose_cap=True, add_tail_cap=True):
    """
    Triangulate parametric body including cut plane bottom.
    
    Returns
    -------
    vertices : np.ndarray
    triangles : np.ndarray
    """
    n_axial = grid['n_axial']
    n_circ = grid['n_circumferential']
    
    X = grid['X']
    Y = grid['Y']
    Z = grid['Z']
    
    vertices_body = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Triangulate body surface
    triangles_body = []
    
    for i in range(n_circ):
        i_next = (i + 1) % n_circ
        
        for j in range(n_axial - 1):
            v0 = i * n_axial + j
            v1 = i * n_axial + (j + 1)
            v2 = i_next * n_axial + j
            v3 = i_next * n_axial + (j + 1)
            
            # Check if quad is degenerate (due to cut plane)
            pts = vertices_body[[v0, v1, v2, v3]]
            
            # Skip if all points at same location (degenerate quad on cut plane)
            if np.allclose(pts[:, 2], pts[0, 2]) and grid['has_cut']:
                # All at same Z - might be on cut plane
                if np.allclose(pts[:, [0, 1]], pts[0, [0, 1]]):
                    continue  # Completely degenerate, skip
            
            triangles_body.append([v0, v1, v2])
            triangles_body.append([v2, v1, v3])
    
    triangles_body = np.array(triangles_body) if triangles_body else np.zeros((0, 3), dtype=int)
    
    all_vertices = [vertices_body]
    all_triangles = [triangles_body] if len(triangles_body) > 0 else []
    
    # Add nose cap
    if add_nose_cap:
        x_nose = grid['x'][0]
        
        # Check if nose is pointed (radius near zero)
        first_ring = vertices_body[::n_axial][:n_circ]
        nose_radius = np.linalg.norm(first_ring[0, 1:])  # Distance from axis
        
        if nose_radius < 1e-10:
            # Pointed nose
            apex = np.array([[x_nose, 0, 0]])
            apex_idx = len(vertices_body)
            all_vertices.append(apex)
            
            nose_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial
                v2 = i_next * n_axial
                nose_tris.append([apex_idx, v2, v1])
            
            all_triangles.append(np.array(nose_tris))
        else:
            # Flat nose
            center = np.array([[x_nose, 0, grid['Z'][0, 0]]])  # Use actual Z
            center_idx = len(vertices_body)
            all_vertices.append(center)
            
            nose_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial
                v2 = i_next * n_axial
                nose_tris.append([center_idx, v1, v2])
            
            all_triangles.append(np.array(nose_tris))
    
    # Add tail cap
    if add_tail_cap:
        n_verts_so_far = sum(len(v) for v in all_vertices)
        x_tail = grid['x'][-1]
        
        last_ring = vertices_body[n_axial-1::n_axial][:n_circ]
        tail_radius = np.linalg.norm(last_ring[0, 1:])
        
        if tail_radius < 1e-10:
            # Pointed tail
            apex = np.array([[x_tail, 0, 0]])
            apex_idx = n_verts_so_far
            all_vertices.append(apex)
            
            tail_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial + (n_axial - 1)
                v2 = i_next * n_axial + (n_axial - 1)
                tail_tris.append([apex_idx, v1, v2])
            
            all_triangles.append(np.array(tail_tris))
        else:
            # Flat tail
            center = np.array([[x_tail, 0, grid['Z'][-1, -1]]])
            center_idx = n_verts_so_far
            all_vertices.append(center)
            
            tail_tris = []
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = i * n_axial + (n_axial - 1)
                v2 = i_next * n_axial + (n_axial - 1)
                tail_tris.append([center_idx, v2, v1])
            
            all_triangles.append(np.array(tail_tris))
    
    # Add flat bottom from cut plane
    if grid['has_cut']:
        n_verts_so_far = sum(len(v) for v in all_vertices)
        
        # Find vertices on cut plane
        cut_verts_idx = np.where(np.abs(vertices_body[:, 2] - grid['z_cut']) < 1e-9)[0]
        
        if len(cut_verts_idx) > 2:
            # We have a cut plane with vertices on it
            # Triangulate the flat bottom
            
            # Get cut plane vertices
            cut_verts = vertices_body[cut_verts_idx]
            
            # Sort by angle around centroid for proper triangulation
            centroid = cut_verts.mean(axis=0)
            angles = np.arctan2(cut_verts[:, 1] - centroid[1], 
                               cut_verts[:, 0] - centroid[0])
            sort_idx = np.argsort(angles)
            cut_verts_sorted_idx = cut_verts_idx[sort_idx]
            
            # Create center point for fan triangulation
            center = centroid.copy()
            center[2] = grid['z_cut']
            center_idx = n_verts_so_far
            all_vertices.append(center.reshape(1, 3))
            
            # Fan triangulation from center
            bottom_tris = []
            n_cut = len(cut_verts_sorted_idx)
            for i in range(n_cut):
                v1 = cut_verts_sorted_idx[i]
                v2 = cut_verts_sorted_idx[(i + 1) % n_cut]
                # Inward normal (pointing down, into body)
                bottom_tris.append([center_idx, v2, v1])
            
            all_triangles.append(np.array(bottom_tris))
    
    # Combine
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles) if all_triangles else np.zeros((0, 3), dtype=int)
    
    return vertices, triangles


def generate_parametric_body_mesh(body, n_axial=100, n_circumferential=60,
                                  add_nose_cap=True, add_tail_cap=True,
                                  merge_tolerance=1e-9, improve_quality=False):
    """
    Complete mesh generation for parametric body.
    
    Parameters
    ----------
    body : ParametricBody
        Body definition
    n_axial : int
        Axial resolution
    n_circumferential : int
        Circumferential resolution
    add_nose_cap : bool
        Close nose
    add_tail_cap : bool
        Close tail
    merge_tolerance : float
        Vertex merging tolerance
    improve_quality : bool
        Apply quality improvement
    
    Returns
    -------
    vertices : np.ndarray
    triangles : np.ndarray
    stats : dict
    """
    print("\n" + "="*70)
    print(f"GENERATING PARAMETRIC BODY: {body.name}")
    print("="*70)
    print(f"  Length: {body.length}")
    print(f"  Control points: {len(body.control_points)}")
    print(f"  Z-squash: {body.z_squash}")
    print(f"  Z-cut: {body.z_cut if body.z_cut is not None else 'None'}")
    
    # Generate point cloud
    print(f"\n[1/5] Generating point cloud...")
    points, grid = generate_parametric_body_point_cloud(body, n_axial, n_circumferential)
    print(f"  Generated {len(points)} points")
    
    # Triangulate
    print(f"\n[2/5] Triangulating surface...")
    vertices, triangles = triangulate_parametric_body(grid, add_nose_cap, add_tail_cap)
    print(f"  Initial: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Merge duplicates
    print(f"\n[3/5] Merging duplicate vertices...")
    vertices_unique, index_map = merge_duplicate_vertices(vertices, merge_tolerance)
    triangles = index_map[triangles]
    n_merged = len(vertices) - len(vertices_unique)
    print(f"  Merged {n_merged} vertices")
    vertices = vertices_unique
    
    # Remove degenerates
    print(f"\n[4/5] Removing degenerate triangles...")
    n_before = len(triangles)
    triangles = remove_degenerate_triangles(triangles, vertices, 1e-10)
    triangles = remove_duplicate_triangles(triangles)
    n_removed = n_before - len(triangles)
    print(f"  Removed {n_removed} triangles")
    
    # Fix topology issues (critical for cut plane meshes)
    if body.z_cut is not None:
        print(f"\n[4.5/5] Fixing topology (cut plane detected)...")
        from mesh_topology_fixer import fix_mesh_topology
        vertices, triangles, topo_report = fix_mesh_topology(vertices, triangles, verbose=False)
        print(f"  Topology fixed: manifold={topo_report['is_manifold']}, "
              f"consistent_winding={topo_report['has_consistent_winding']}")
    
    # Quality analysis
    print(f"\n[5/5] Computing quality metrics...")
    qualities = compute_triangle_quality(triangles, vertices)
    
    # Check watertightness
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
        'mean_aspect_ratio': qualities['aspect_ratio'].mean(),
        'mean_skewness': qualities['skewness'].mean(),
        'poor_quality_count': (qualities['skewness'] > 0.9).sum(),
        'boundary_edges': boundary_edges,
        'is_watertight': boundary_edges == 0,
        'is_manifold': non_manifold_edges == 0
    }
    
    print(f"\n  Quality: skewness={stats['mean_skewness']:.3f}")
    print(f"  Topology: watertight={stats['is_watertight']}, "
          f"boundary_edges={boundary_edges}")
    
    if improve_quality and stats['poor_quality_count'] > 0:
        print(f"\n[OPTIONAL] Improving mesh quality...")
        from mesh_quality_optimizer import advanced_quality_improvement
        vertices, triangles = advanced_quality_improvement(
            vertices, triangles, edge_swap_iters=5, smoothing_iters=3
        )
    
    return vertices, triangles, stats


# ============================================================================
# EXAMPLE USAGE & PRESETS
# ============================================================================

def create_missile_body(length, diameter, nose_length_frac=0.2, 
                       z_cut=None, z_squash=1.0):
    """Create missile-like body with ogive nose."""
    nose_length = length * nose_length_frac
    body_length = length - nose_length
    radius = diameter / 2
    
    # Control points: ogive nose + cylindrical body
    n_nose = 5
    x_nose = np.linspace(0, nose_length_frac, n_nose)
    
    # Ogive formula
    rho = (radius**2 + nose_length**2) / (2 * radius)
    r_nose = [rho - np.sqrt(rho**2 - (x * length)**2) for x in x_nose]
    
    # Body section
    x_body = np.linspace(nose_length_frac, 1.0, 5)
    r_body = [radius] * len(x_body)
    
    control_points = list(zip(
        np.concatenate([x_nose[:-1], x_body]),
        r_nose[:-1] + r_body
    ))
    
    return ParametricBody(
        length, control_points, z_cut, z_squash,
        name=f"Missile (L={length}, D={diameter})"
    )


def create_lifting_body(length, max_width, z_cut=None, z_squash=0.3):
    """Create lifting body with flat bottom and elliptical cross-section."""
    # Blended body shape
    control_points = [
        (0.0, 0.0),      # Pointed nose
        (0.1, 0.2),
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.9),
        (0.9, 0.95),
        (1.0, 1.0)       # Max width at tail
    ]
    
    # Scale by max_width
    control_points = [(x, r * max_width) for x, r in control_points]
    
    return ParametricBody(
        length, control_points, z_cut, z_squash,
        name=f"Lifting Body (L={length}, W={max_width*2})"
    )


if __name__ == "__main__":
    print("Parametric Body Generator with Cut & Squash")
    print("="*70)
    
    # Example 1: Standard missile
    print("\n" + "="*70)
    print("EXAMPLE 1: Standard Missile")
    print("="*70)
    
    missile = create_missile_body(length=10, diameter=1)
    v1, t1, s1 = generate_parametric_body_mesh(missile, n_axial=80, n_circumferential=40)
    write_tri_file("missile_standard.tri", v1, t1)
    
    # Example 2: Missile with flat bottom (cut plane)
    print("\n" + "="*70)
    print("EXAMPLE 2: Missile with Flat Bottom")
    print("="*70)
    
    missile_cut = create_missile_body(length=10, diameter=1, z_cut=-0.3)
    v2, t2, s2 = generate_parametric_body_mesh(missile_cut, n_axial=80, n_circumferential=40)
    write_tri_file("missile_flat_bottom.tri", v2, t2)
    
    # Example 3: Elliptical missile (squashed)
    print("\n" + "="*70)
    print("EXAMPLE 3: Elliptical Missile (Squashed)")
    print("="*70)
    
    missile_squash = create_missile_body(length=10, diameter=1, z_squash=0.5)
    v3, t3, s3 = generate_parametric_body_mesh(missile_squash, n_axial=80, n_circumferential=40)
    write_tri_file("missile_elliptical.tri", v3, t3)
    
    # Example 4: Lifting body (flat bottom + elliptical)
    print("\n" + "="*70)
    print("EXAMPLE 4: Lifting Body")
    print("="*70)
    
    lifting = create_lifting_body(length=10, max_width=1.5, z_cut=-0.1, z_squash=0.3)
    v4, t4, s4 = generate_parametric_body_mesh(lifting, n_axial=100, n_circumferential=60)
    write_tri_file("lifting_body.tri", v4, t4)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. missile_standard.tri - Standard circular missile")
    print("  2. missile_flat_bottom.tri - Missile with flat underside")
    print("  3. missile_elliptical.tri - Elliptical cross-section")
    print("  4. lifting_body.tri - Lifting body (flat + elliptical)")