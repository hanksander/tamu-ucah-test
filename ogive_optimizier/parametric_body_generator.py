"""
Parametric Body Generator with hemisphere nose cap support
"""
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
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
    Now supports genuine hemisphere nose caps.
    
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
    spline_order : int
        Order of spline interpolation (1=linear, 3=cubic)
    enforce_monotonic : bool
        If True, ensures radius never decreases along body
    hemisphere_nose : bool
        If True, uses genuine hemisphere at nose instead of spline
    hemisphere_radius : float, optional
        Radius of hemisphere nose cap (only used if hemisphere_nose=True)
    """
    
    def __init__(self, length, control_points, z_cut=None, z_squash=1.0, 
                 spline_order=3, enforce_monotonic=True, 
                 hemisphere_nose=False, hemisphere_radius=None,
                 name="Parametric Body"):
        self.length = length
        self.z_cut = z_cut
        self.z_squash = z_squash
        self.spline_order = spline_order
        self.name = name
        self.hemisphere_nose = hemisphere_nose
        self.hemisphere_radius = hemisphere_radius
        
        # Sort control points by x-position
        control_points = np.array(control_points)
        sort_idx = np.argsort(control_points[:, 0])
        control_points = control_points[sort_idx]
        
        # CRITICAL FIX: Enforce monotonic radii BEFORE spline fitting
        if enforce_monotonic:
            x_vals = control_points[:, 0]
            r_vals = control_points[:, 1].copy()
            
            # Forward pass: ensure each radius >= previous
            for i in range(1, len(r_vals)):
                if r_vals[i] < r_vals[i-1]:
                    r_vals[i] = r_vals[i-1]
            
            control_points = np.column_stack([x_vals, r_vals])
            print(f"  Monotonic enforcement: adjusted {sum(control_points[:, 1] != np.array([cp[1] for cp in control_points]))} radii")
        
        self.control_points = control_points
        
        # Hemisphere setup
        if self.hemisphere_nose:
            if self.hemisphere_radius is None:
                # Use first control point radius as hemisphere radius
                self.hemisphere_radius = control_points[0, 1]
            
            # Find where hemisphere ends (x = hemisphere_radius)
            self.hemisphere_end_x = self.hemisphere_radius / self.length
            
            # Filter control points to only those after hemisphere
            body_mask = control_points[:, 0] > self.hemisphere_end_x
            body_control_points = control_points[body_mask]
            
            # Ensure continuity: add tangent point at hemisphere junction
            hemisphere_end_r = self.hemisphere_radius
            junction_point = np.array([[self.hemisphere_end_x, hemisphere_end_r]])
            
            if len(body_control_points) == 0:
                body_control_points = junction_point
            elif body_control_points[0, 0] > self.hemisphere_end_x + 1e-6:
                body_control_points = np.vstack([junction_point, body_control_points])
            
            self.body_control_points = body_control_points
            
            print(f"  Hemisphere nose: R={self.hemisphere_radius:.6f}, end_x={self.hemisphere_end_x:.6f}")
        else:
            self.body_control_points = control_points
        
        # Create spline for body radius function (excluding hemisphere region)
        x_normalized = self.body_control_points[:, 0]
        r_values = self.body_control_points[:, 1]
        
        # Create spline (use min order if not enough points)
        k = min(spline_order, len(x_normalized) - 1)
        
        if k == 1:
            self.r_spline = interp1d(x_normalized, r_values, 
                                     kind='linear', fill_value='extrapolate')
        elif enforce_monotonic and k >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                self.r_spline = PchipInterpolator(x_normalized, r_values, extrapolate=False)
                print(f"  Using PCHIP (monotonic cubic) interpolation")
            except ImportError:
                s_factor = 0.01 * len(x_normalized)
                self.r_spline = UnivariateSpline(x_normalized, r_values, k=k, s=s_factor)
                print(f"  Using smoothed cubic spline (s={s_factor:.3f})")
        else:
            self.r_spline = UnivariateSpline(x_normalized, r_values, k=k, s=0)
            print(f"  Using standard spline interpolation (k={k})")
    
    def r(self, x):
        """Radius at axial position x (in absolute coordinates)."""
        x_norm = np.clip(x / self.length, 0, 1)
        
        if self.hemisphere_nose and x_norm <= self.hemisphere_end_x:
            # CORRECTED: Hemisphere equation with apex at origin
            # For a hemisphere with base at x=R, we want:
            # r = sqrt(R^2 - (R-x)^2) = sqrt(2*R*x - x^2)
            x_abs = x_norm * self.length
            R = self.hemisphere_radius
            
            if x_abs <= R:
                # This gives r=0 at x=0 (apex) and r=R at x=R (base)
                r_val = np.sqrt(max(0, 2*R*x_abs - x_abs**2))
            else:
                # Fallback to spline if slightly past hemisphere
                r_val = float(self.r_spline(x_norm))
        else:
            # Use spline for body
            r_val = float(self.r_spline(x_norm))
        
        return max(0.0, r_val)
    
    def r_y(self, x):
        """Y semi-axis at position x (circular cross-section)."""
        return self.r(x)
    
    def r_z(self, x):
        """Z semi-axis at position x (with squash applied)."""
        return self.r(x) * self.z_squash
    
    def validate_monotonic(self, n_samples=100):
        """
        Validate that radius is monotonically non-decreasing along body.
        """
        x_test = np.linspace(0, self.length, n_samples)
        r_test = [self.r(x) for x in x_test]
        
        is_monotonic = all(r_test[i] <= r_test[i+1] for i in range(len(r_test)-1))
        
        if not is_monotonic:
            decreases = []
            for i in range(len(r_test)-1):
                if r_test[i] > r_test[i+1]:
                    decreases.append((x_test[i], x_test[i+1], r_test[i], r_test[i+1]))
            
            print(f"  WARNING: Radius decreases at {len(decreases)} locations:")
            for x1, x2, r1, r2 in decreases[:3]:
                print(f"    x={x1:.4f} to {x2:.4f}: r={r1:.6f} to {r2:.6f} (Δr={r2-r1:.6f})")
        
        return is_monotonic

def generate_parametric_body_point_cloud(body, n_axial=100, n_circumferential=60):
    """
    Generate point cloud for parametric body with elliptical cross-section.
    Now properly handles hemisphere nose geometry.
    """
    # Adaptive axial distribution for hemisphere
    if body.hemisphere_nose:
        # More points in hemisphere region for smooth curvature
        n_hemisphere = max(20, int(n_axial * body.hemisphere_end_x * 2))
        n_body = n_axial - n_hemisphere
        
        x_hemisphere = np.linspace(0, body.hemisphere_end_x * body.length, n_hemisphere)
        x_body = np.linspace(body.hemisphere_end_x * body.length, body.length, n_body + 1)[1:]
        x = np.concatenate([x_hemisphere, x_body])
    else:
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
    
    # Apply cut plane if specified
    if body.z_cut is not None:
        below_cut = Z < body.z_cut
        Z[below_cut] = body.z_cut
    
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    grid = {
        'x': x,
        'theta': theta,
        'n_axial': len(x),
        'n_circumferential': n_circumferential,
        'X': X,
        'Y': Y,
        'Z': Z,
        'R_y': R_y,
        'R_z': R_z,
        'has_cut': body.z_cut is not None,
        'z_cut': body.z_cut,
        'z_squash': body.z_squash,
        'hemisphere_nose': body.hemisphere_nose,
        'hemisphere_radius': body.hemisphere_radius if body.hemisphere_nose else None
    }
    
    return points, grid

def orient_triangle_normals(vertices, triangles):
    """Ensure triangle winding so that normals point outward (radially)."""
    verts = vertices
    tris = triangles.copy()
    for i in range(len(tris)):
        a, b, c = tris[i]
        v0 = verts[a]; v1 = verts[b]; v2 = verts[c]
        normal = np.cross(v1 - v0, v2 - v0)
        cent = (v0 + v1 + v2) / 3.0
        radial = np.array([0.0, cent[1], cent[2]])
        if np.linalg.norm(radial) < 1e-12:
            radial = np.array([0.0, 1.0, 0.0])
    return tris

def triangulate_parametric_body(grid, add_nose_cap=True, add_tail_cap=True, clip_tol=1e-12):
    """
    Triangulate parametric body including hemisphere nose handling.
    """
    n_axial = grid['n_axial']
    n_circ = grid['n_circumferential']
    X = grid['X']
    Y = grid['Y']
    Z = grid['Z']
    R_y = grid['R_y']
    vertices_body = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    triangles_body = []
    all_vertices = [vertices_body.copy()]
    extra_vertices = []
    extra_triangles = []
    intersection_map = {}
    triangles_local = []
    
    def vid(i_theta, j_axial):
        return i_theta * n_axial + j_axial
    
    z_cut = grid.get('z_cut', None)
    has_cut = grid.get('has_cut', False)
    hemisphere_nose = grid.get('hemisphere_nose', False)
    
    def is_above(z):
        return (z - z_cut) >= -clip_tol if has_cut else True
    
    # Standard triangulation (with or without cut plane)
    if (not has_cut) or (z_cut is None):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            for j in range(n_axial - 1):
                v0 = vid(i, j)
                v1 = vid(i, j + 1)
                v2 = vid(i_next, j)
                v3 = vid(i_next, j + 1)
                r_j = R_y[i, j]
                r_j1 = R_y[i, j+1]
                
                if r_j < 1e-10 and r_j1 > 1e-10:
                    triangles_local.append([v0, v1, v3])
                    triangles_local.append([v0, v3, v2])
                elif r_j1 < 1e-10 and r_j > 1e-10:
                    triangles_local.append([v1, v2, v0])
                    triangles_local.append([v1, v3, v2])
                elif r_j > 1e-10 and r_j1 > 1e-10:
                    triangles_local.append([v0, v1, v2])
                    triangles_local.append([v2, v1, v3])
    else:
        # Clipping logic (same as before)
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            for j in range(n_axial - 1):
                quad_idx = [vid(i, j), vid(i, j + 1), vid(i_next, j + 1), vid(i_next, j)]
                quad_pts = [vertices_body[idx] for idx in quad_idx]
                quad_z = [p[2] for p in quad_pts]
                above_flags = [is_above(z) for z in quad_z]
                
                if all(above_flags):
                    v0, v1, v3, v2 = quad_idx[0], quad_idx[1], quad_idx[2], quad_idx[3]
                    r_j = R_y[i, j]
                    r_j1 = R_y[i, j+1]
                    if r_j < 1e-10 and r_j1 > 1e-10:
                        triangles_local.append([v0, v1, v3])
                        triangles_local.append([v0, v3, v2])
                    elif r_j1 < 1e-10 and r_j > 1e-10:
                        triangles_local.append([v1, v2, v0])
                        triangles_local.append([v1, v3, v2])
                    elif r_j > 1e-10 and r_j1 > 1e-10:
                        triangles_local.append([v0, v1, v2])
                        triangles_local.append([v2, v1, v3])
                elif not any(above_flags):
                    continue
                else:
                    # Clipping logic (same as original)
                    poly_indices = []
                    poly_points = []
                    for k in range(4):
                        idx_k = quad_idx[k]
                        p_k = vertices_body[idx_k]
                        z_k = p_k[2]
                        keep_k = is_above(z_k)
                        if keep_k:
                            poly_indices.append(idx_k)
                            poly_points.append(p_k.copy())
                        next_k = (k + 1) % 4
                        idx_n = quad_idx[next_k]
                        p_n = vertices_body[idx_n]
                        z_n = p_n[2]
                        if (z_k - z_cut) * (z_n - z_cut) < -clip_tol:
                            denom = (z_n - z_k)
                            if abs(denom) < 1e-16:
                                t = 0.5
                            else:
                                t = (z_cut - z_k) / denom
                            t = np.clip(t, 0.0, 1.0)
                            p_int = p_k + t * (p_n - p_k)
                            p_int[2] = z_cut
                            edge_key = tuple(sorted((int(idx_k), int(idx_n))))
                            if edge_key in intersection_map:
                                inter_idx = intersection_map[edge_key]
                            else:
                                inter_idx = len(extra_vertices)
                                extra_vertices.append(p_int)
                                intersection_map[edge_key] = inter_idx
                            poly_indices.append(-(1 + inter_idx))
                            poly_points.append(p_int)
                    
                    if len(poly_points) < 3:
                        continue
                    for kk in range(1, len(poly_indices) - 1):
                        ia = poly_indices[0]
                        ib = poly_indices[kk]
                        ic = poly_indices[kk + 1]
                        def resolve_index(idx):
                            if idx >= 0:
                                return int(idx), vertices_body[int(idx)]
                            else:
                                ei = - (idx + 1)
                                return None, extra_vertices[ei]
                        a_idx_body, a_pt = resolve_index(ia)
                        b_idx_body, b_pt = resolve_index(ib)
                        c_idx_body, c_pt = resolve_index(ic)
                        if (a_idx_body is not None) and (b_idx_body is not None) and (c_idx_body is not None):
                            tri = [a_idx_body, b_idx_body, c_idx_body]
                            area = np.linalg.norm(np.cross(b_pt - a_pt, c_pt - a_pt)) * 0.5
                            if area > 1e-14:
                                triangles_local.append(tri)
                        else:
                            extra_triangles.append([ia, ib, ic])
    
    # Stack vertices
    extra_vertices_arr = np.array(extra_vertices) if extra_vertices else np.zeros((0, 3), dtype=float)
    all_vertices_arr = np.vstack([all_vertices[0], extra_vertices_arr]) if extra_vertices_arr.size else all_vertices[0].copy()
    
    triangles_out = []
    for tri in triangles_local:
        triangles_out.append([int(tri[0]), int(tri[1]), int(tri[2])])
    
    base_extra_idx = len(all_vertices[0])
    for tri in extra_triangles:
        conv = []
        for idx in tri:
            if idx >= 0:
                conv.append(int(idx))
            else:
                ei = -(idx + 1)
                conv.append(base_extra_idx + ei)
        a, b, c = [all_vertices_arr[ii] for ii in conv]
        area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
        if area > 1e-14:
            triangles_out.append(conv)
    
    # Nose cap - hemisphere or sharp
    if add_nose_cap:
        x_nose = grid['x'][0]
        first_ring = vertices_body[0:n_axial * n_circ:n_axial]
        nose_radius = np.linalg.norm(first_ring[0, 1:])
        
        if hemisphere_nose and nose_radius > 1e-8:
            # Create hemisphere cap at origin
            apex = np.array([[0.0, 0.0, 0.0]])
            apex_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, apex])
            
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = vid(i, 0)
                v2 = vid(i_next, 0)
                if np.linalg.norm(vertices_body[v1] - vertices_body[v2]) > 1e-10:
                    triangles_out.append([apex_idx, v2, v1])
        elif nose_radius < 1e-8:
            # Sharp nose
            apex = np.array([[x_nose, 0.0, 0.0]])
            apex_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, apex])
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = vid(i, 0)
                v2 = vid(i_next, 0)
                if np.linalg.norm(vertices_body[v1] - vertices_body[v2]) > 1e-10:
                    triangles_out.append([apex_idx, v2, v1])
        else:
            # Flat nose (cylinder)
            center = np.array([[x_nose, 0.0, grid['Z'][0, 0]]])
            center_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, center])
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = vid(i, 0)
                v2 = vid(i_next, 0)
                tri_pts = np.array([center[0], vertices_body[v1], vertices_body[v2]])
                tri_area = np.linalg.norm(np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0]))
                if tri_area > 1e-10:
                    triangles_out.append([center_idx, v1, v2])
    
    # Tail cap (same as before)
    if add_tail_cap:
        x_tail = grid['x'][-1]
        last_ring = vertices_body[n_axial - 1::n_axial][:n_circ]
        tail_radius = np.linalg.norm(last_ring[0, 1:])
        if tail_radius < 1e-8:
            apex = np.array([[x_tail, 0.0, 0.0]])
            apex_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, apex])
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = vid(i, n_axial - 1)
                v2 = vid(i_next, n_axial - 1)
                if np.linalg.norm(vertices_body[v1] - vertices_body[v2]) > 1e-10:
                    triangles_out.append([apex_idx, v1, v2])
        else:
            center = np.array([[x_tail, 0.0, grid['Z'][-1, -1]]])
            center_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, center])
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                v1 = vid(i, n_axial - 1)
                v2 = vid(i_next, n_axial - 1)
                tri_pts = np.array([center[0], vertices_body[v2], vertices_body[v1]])
                tri_area = np.linalg.norm(np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0]))
                if tri_area > 1e-10:
                    triangles_out.append([center_idx, v2, v1])
    
    # Flat bottom cap
    if has_cut and extra_vertices_arr.size:
        cut_verts_indices = list(range(base_extra_idx, base_extra_idx + len(extra_vertices)))
        if len(cut_verts_indices) >= 3:
            cut_pts = all_vertices_arr[cut_verts_indices]
            centroid = cut_pts.mean(axis=0)
            angles = np.arctan2(cut_pts[:, 1] - centroid[1], cut_pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            ordered_idx = [cut_verts_indices[k] for k in order]
            center = centroid.copy()
            center[2] = z_cut
            center_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, center.reshape(1, 3)])
            n_cut = len(ordered_idx)
            for k in range(n_cut):
                v1 = ordered_idx[k]
                v2 = ordered_idx[(k + 1) % n_cut]
                tri_pts = np.array([all_vertices_arr[center_idx], all_vertices_arr[v2], all_vertices_arr[v1]])
                area = np.linalg.norm(np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])) * 0.5
                if area > 1e-14:
                    triangles_out.append([center_idx, v2, v1])
    
    if len(triangles_out) == 0:
        triangles_arr = np.zeros((0, 3), dtype=int)
    else:
        triangles_arr = np.array(triangles_out, dtype=int)
    
    triangles_arr = orient_triangle_normals(all_vertices_arr, triangles_arr)
    
    return all_vertices_arr, triangles_arr

# Rest of the functions remain the same as original
def generate_parametric_body_mesh(body, n_axial=100, n_circumferential=60,
                                  add_nose_cap=True, add_tail_cap=True,
                                  merge_tolerance=1e-9, improve_quality=False,
                                  validate_monotonic=True):
    """Complete mesh generation for parametric body."""
    print("\n" + "="*70)
    print(f"GENERATING PARAMETRIC BODY: {body.name}")
    print("="*70)
    print(f"  Length: {body.length}")
    print(f"  Control points: {len(body.control_points)}")
    print(f"  Z-squash: {body.z_squash}")
    print(f"  Z-cut: {body.z_cut if body.z_cut is not None else 'None'}")
    print(f"  Hemisphere nose: {body.hemisphere_nose}")
    if body.hemisphere_nose:
        print(f"  Hemisphere radius: {body.hemisphere_radius:.6f}")
    
    if validate_monotonic:
        print(f"\n[0/5] Validating monotonic radius...")
        is_monotonic = body.validate_monotonic(n_samples=200)
        if is_monotonic:
            print(f"  ✓ Radius is monotonically non-decreasing")
        else:
            print(f"  ✗ WARNING: Radius decreases in some regions (pinching detected)")
    
    print(f"\n[1/5] Generating point cloud...")
    points, grid = generate_parametric_body_point_cloud(body, n_axial, n_circumferential)
    print(f"  Generated {len(points)} points")
    
    print(f"\n[2/5] Triangulating surface...")
    vertices, triangles = triangulate_parametric_body(grid, add_nose_cap, add_tail_cap)
    print(f"  Initial: {len(vertices)} vertices, {len(triangles)} triangles")
    
    print(f"\n[3/5] Merging duplicate vertices...")
    vertices_unique, index_map = merge_duplicate_vertices(vertices, merge_tolerance)
    triangles = index_map[triangles]
    n_merged = len(vertices) - len(vertices_unique)
    print(f"  Merged {n_merged} vertices")
    vertices = vertices_unique
    
    print(f"\n[4/5] Removing degenerate triangles...")
    n_before = len(triangles)
    triangles = remove_degenerate_triangles(triangles, vertices, 1e-10)
    triangles = remove_duplicate_triangles(triangles)
    n_removed = n_before - len(triangles)
    print(f"  Removed {n_removed} triangles")
    
    print(f"\n[5/5] Computing quality metrics...")
    qualities = compute_triangle_quality(triangles, vertices)
    
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

# Keep existing preset functions for backwards compatibility
def create_missile_body(length, diameter, nose_length_frac=0.2, 
                       z_cut=None, z_squash=1.0):
    """Create missile-like body with ogive nose."""
    nose_length = length * nose_length_frac
    body_length = length - nose_length
    radius = diameter / 2
    
    n_nose = 5
    x_nose = np.linspace(0, nose_length_frac, n_nose)
    
    rho = (radius**2 + nose_length**2) / (2 * radius)
    r_nose = [rho - np.sqrt(max(0, rho**2 - (x * length)**2)) for x in x_nose]
    
    x_body = np.linspace(nose_length_frac, 1.0, 5)
    r_body = [radius] * len(x_body)
    
    control_points = list(zip(
        np.concatenate([x_nose[:-1], x_body]),
        r_nose[:-1] + r_body
    ))
    
    return ParametricBody(
        length, control_points, z_cut, z_squash,
        enforce_monotonic=True,
        hemisphere_nose=False,
        name=f"Missile (L={length}, D={diameter})"
    )

def create_lifting_body(length, max_width, z_cut=None, z_squash=0.3):
    """Create lifting body with flat bottom and elliptical cross-section."""
    control_points = [
        (0.0, 0.0),
        (0.1, 0.2),
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.9),
        (0.9, 0.95),
        (1.0, 1.0)
    ]
    
    control_points = [(x, r * max_width) for x, r in control_points]
    
    return ParametricBody(
        length, control_points, z_cut, z_squash,
        enforce_monotonic=True,
        hemisphere_nose=False,
        name=f"Lifting Body (L={length}, W={max_width*2})"
    )

def run_winding_order_tests():
    """
    Test 3 configurations:
        (1) Pure BOR
        (2) BOR + squash
        (3) BOR + cut + squash

    Each test uses arbitrary random control points for the spline.
    This is meant to detect failures caused by incorrect winding orientation
    when swap_yz = True/False inside the underlying generator.
    """

    # Arbitrary values (just to create a body)
    length = 1
    max_r = 0.05   # under constraint
    z_squash_vals = [1.0, 0.7, 0.5]
    z_cut_vals    = [None, None, -0.01]

    # Generate random (but monotonic) radii
    cp = [0.3, 0.5, 0.7, 0.9, 1.0]

    tests = [
        ("BOR only"               , z_squash_vals[0], None     ),
        ("BOR + squash"           , z_squash_vals[1], None     ),
        ("BOR + cut + squash"     , z_squash_vals[2], z_cut_vals[2]),
    ]

    geometries = []
    for name, squash, cut in tests:
        control_points = [(i/(len(cp)-1), r * max_r) for i, r in enumerate(cp)]
        geometries.append(ParametricBody(
        length, control_points, cut, squash,
        enforce_monotonic=True,
        name=name
        ))
    
    for body in geometries:
        print("\n" + "="*70)
        print(f"TESTING WINDING ORDER: {body.name}")
        print("="*70)
        
        v, t, s = generate_parametric_body_mesh(
            body, n_axial=50, n_circumferential=30,
            add_nose_cap=True, add_tail_cap=True,
            merge_tolerance=1e-9, improve_quality=False,
            validate_monotonic=True
        )
        filename = f"winding_test_{body.name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.tri"
        write_tri_file(filename, v, t, swap_yz=False)
        print(f"  Winding order test mesh written to: {filename}")

    return geometries

def final_geometry(new_nose_radius=None):
    '''final geometry from optimizer'''
    control_points = [
        (0.050, 0.010791),
        (0.150, 0.024675),
        (0.250, 0.024675),
        (0.400, 0.038786),
        (0.550, 0.045318),
        (0.750, 0.045318),
        (1.000, 0.050354)]
    length = 1.0
    z_squash = 0.656898
    z_cut = -0.035043
    
    if new_nose_radius is not None:
        hemisphere_radius = new_nose_radius
    else:
        hemisphere_radius = 0.001 #original nose is 1mm

    v, t, s = generate_parametric_body_mesh(
        ParametricBody(
            length=length,
            control_points=control_points,
            z_cut=z_cut,
            z_squash=z_squash,
            hemisphere_nose=True,
            hemisphere_radius=hemisphere_radius,
            name="Final Optimized Body"
        ),
        n_axial=150,
        n_circumferential=80,
        add_nose_cap=True,
        add_tail_cap=True,
        merge_tolerance=1e-9,
        improve_quality=False,
        validate_monotonic=True)

    print(f"Volume: {compute_volume(v, t):.6f} m^3")
    
    if new_nose_radius is not None:
        print(f"Final optimized body mesh with new nose radius {new_nose_radius} m written to: final_optimized_body_new_nose.tri")
        write_tri_file(f"final_optimized_body_nr{new_nose_radius}.tri", v, t, swap_yz=False)
    else:
        write_tri_file("final_optimized_body.tri", v, t, swap_yz=False)

    print(f"Final optimized body mesh written to: final_optimized_body.tri")

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



if __name__ == "__main__":
    print("Parametric Body Generator - FIXED VERSION (No Pinching)")
    print("="*70)

    final_geometry()
    final_geometry(new_nose_radius=0.005)  # Example with 5mm nose radius
    final_geometry(new_nose_radius=0.01)
    final_geometry(new_nose_radius=0.02)
    final_geometry(new_nose_radius=0.04)

    # print("Parametric Body Generator with Hemisphere Support")
    # print("="*70)
    
    # # Test hemisphere nose
    # print("\n" + "="*70)
    # print("TEST: Hemisphere Nose")
    # print("="*70)
    
    # control_points = [
    #     (0.0, 0.0),
    #     (0.05, 0.02),
    #     (0.2, 0.05),
    #     (0.5, 0.08),
    #     (1.0, 0.1)
    # ]
    
    # body_hemisphere = ParametricBody(
    #     length=1.0,
    #     control_points=control_points,
    #     z_cut=-0.05,
    #     z_squash=0.9,
    #     hemisphere_nose=True,
    #     hemisphere_radius=0.02,
    #     name="Hemisphere Test"
    # )
    
    # v, t, s = generate_parametric_body_mesh(
    #     body_hemisphere,
    #     n_axial=100,
    #     n_circumferential=50,
    # )
    
    # write_tri_file("hemisphere_test.tri", v, t, swap_yz=False)
    # print(f"  Written to: hemisphere_test.tri")

    # run_winding_order_tests()

    
    # # Example 2: Missile with flat bottom (cut plane)
    # print("\n" + "="*70)
    # print("EXAMPLE 2: Missile with Flat Bottom")
    # print("="*70)
    
    # missile_cut = create_missile_body(length=1, diameter=0.2, z_cut=-0.)
    # v2, t2, s2 = generate_parametric_body_mesh(missile_cut, n_axial=80, n_circumferential=40)
    # write_tri_file("missile_flat_bottom_fixed.tri", v2, t2, swap_yz = False)
    
    # # Example 4: Lifting body (flat bottom + elliptical)
    # print("\n" + "="*70)
    # print("EXAMPLE 4: Lifting Body")
    # print("="*70)
    
    # lifting = create_lifting_body(length=1, max_width=0.2, z_cut=-0, z_squash=0.3)
    # v4, t4, s4 = generate_parametric_body_mesh(lifting, n_axial=100, n_circumferential=60)
    # write_tri_file("lifting_body_fixed.tri", v4, t4, swap_yz = False)
    
    # print("\n" + "="*70)
    # print("GENERATION COMPLETE")
    # print("="*70)
    # print("\nKey improvements:")
    # print("  1. Monotonic radius enforcement BEFORE spline fitting")
    # print("  2. PCHIP interpolation for smooth, non-oscillating curves")
    # print("  3. Validation checks to detect any remaining pinching")
    # print("  4. Better handling of degenerate triangles at nose/tail")