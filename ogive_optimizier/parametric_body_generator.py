"""
Parametric Body Generator with advanced features - FIXED VERSION
1. Flat bottom cut plane (Z = z_cut)
2. Elliptical cross-section (squash in Z direction)
3. Spline-based radius definition
4. FIX: Prevents "pinching" between control points by ensuring smooth interpolation
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
    
    FIXED: Ensures monotonic radius enforcement happens BEFORE spline fitting
    to prevent artificial pinching between control points.
    
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
    enforce_monotonic : bool
        If True, ensures radius never decreases along body (prevents pinching)
    """
    
    def __init__(self, length, control_points, z_cut=None, z_squash=1.0, 
                 spline_order=3, enforce_monotonic=True, name="Parametric Body"):
        self.length = length
        self.z_cut = z_cut
        self.z_squash = z_squash
        self.spline_order = spline_order
        self.name = name
        
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
        
        # Create spline for radius function
        x_normalized = self.control_points[:, 0]
        r_values = self.control_points[:, 1]
        
        # Create spline (use min order if not enough points)
        k = min(spline_order, len(x_normalized) - 1)
        
        if k == 1:
            # Linear interpolation - always monotonic
            self.r_spline = interp1d(x_normalized, r_values, 
                                     kind='linear', fill_value='extrapolate')
        elif enforce_monotonic and k >= 3:
            # Use monotonic cubic spline (Hermite) to prevent oscillations
            # This prevents the spline from creating local minima between control points
            try:
                # Try to use PchipInterpolator for monotonic interpolation
                from scipy.interpolate import PchipInterpolator
                self.r_spline = PchipInterpolator(x_normalized, r_values, extrapolate=False)
                print(f"  Using PCHIP (monotonic cubic) interpolation")
            except ImportError:
                # Fallback to standard cubic with smoothing
                # Use higher smoothing factor to reduce oscillations
                s_factor = 0.01 * len(x_normalized)  # Adaptive smoothing
                self.r_spline = UnivariateSpline(x_normalized, r_values, k=k, s=s_factor)
                print(f"  Using smoothed cubic spline (s={s_factor:.3f})")
        else:
            # Standard spline interpolation
            self.r_spline = UnivariateSpline(x_normalized, r_values, k=k, s=0)
            print(f"  Using standard spline interpolation (k={k})")
    
    def r(self, x):
        """Radius at axial position x (in absolute coordinates)."""
        x_norm = np.clip(x / self.length, 0, 1)  # Ensure within bounds
        
        # Get spline value
        r_val = float(self.r_spline(x_norm))
        
        # Ensure non-negative (splines can sometimes go slightly negative near boundaries)
        r_val = max(0.0, r_val)
        
        return r_val
    
    def r_y(self, x):
        """Y semi-axis at position x (circular cross-section)."""
        return self.r(x)
    
    def r_z(self, x):
        """Z semi-axis at position x (with squash applied)."""
        return self.r(x) * self.z_squash
    
    def validate_monotonic(self, n_samples=100):
        """
        Validate that radius is monotonically non-decreasing along body.
        Returns True if monotonic, False otherwise.
        """
        x_test = np.linspace(0, self.length, n_samples)
        r_test = [self.r(x) for x in x_test]
        
        # Check if monotonic
        is_monotonic = all(r_test[i] <= r_test[i+1] for i in range(len(r_test)-1))
        
        if not is_monotonic:
            # Find where it decreases
            decreases = []
            for i in range(len(r_test)-1):
                if r_test[i] > r_test[i+1]:
                    decreases.append((x_test[i], x_test[i+1], r_test[i], r_test[i+1]))
            
            print(f"  WARNING: Radius decreases at {len(decreases)} locations:")
            for x1, x2, r1, r2 in decreases[:3]:  # Show first 3
                print(f"    x={x1:.4f} to {x2:.4f}: r={r1:.6f} to {r2:.6f} (Δr={r2-r1:.6f})")
        
        return is_monotonic


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
    
    # Apply cut plane if specified
    if body.z_cut is not None:
        # Project points below z_cut onto the plane
        below_cut = Z < body.z_cut
        Z[below_cut] = body.z_cut
    
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    grid = {
        'x': x,
        'theta': theta,
        'n_axial': n_axial,
        'n_circumferential': n_circumferential,
        'X': X,
        'Y': Y,
        'Z': Z,
        'R_y': R_y,
        'R_z': R_z,
        'has_cut': body.z_cut is not None,
        'z_cut': body.z_cut,
        'z_squash': body.z_squash
    }
    
    return points, grid

def orient_triangle_normals(vertices, triangles):
    """
    Ensure triangle winding so that normals point outward (radially).
    For each triangle compute normal = cross(v1-v0, v2-v0) and centroid.
    If dot(normal, centroid_radial) < 0 flip the triangle.
    """
    verts = vertices
    tris = triangles.copy()
    for i in range(len(tris)):
        a, b, c = tris[i]
        v0 = verts[a]; v1 = verts[b]; v2 = verts[c]
        normal = np.cross(v1 - v0, v2 - v0)
        # centroid
        cent = (v0 + v1 + v2) / 3.0
        # radial direction from axis (ignore x-axis)
        radial = np.array([0.0, cent[1], cent[2]])
        # if radial is near zero (centroid on axis), use small perturbation in Y
        if np.linalg.norm(radial) < 1e-12:
            radial = np.array([0.0, 1.0, 0.0])
        if np.dot(normal, radial) < 0:
            # flip winding
            tris[i] = [a, c, b]
    return tris


def triangulate_parametric_body(grid, add_nose_cap=True, add_tail_cap=True, clip_tol=1e-12):
    """
    Triangulate parametric body including robust clipping for a flat bottom (z = z_cut).
    This implementation clips each quad against the cut plane and reuses intersection
    vertices for shared edges to guarantee consistent connectivity.
    """
    n_axial = grid['n_axial']
    n_circ = grid['n_circumferential']

    X = grid['X']
    Y = grid['Y']
    Z = grid['Z']
    R_y = grid['R_y']

    vertices_body = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    triangles_body = []
    all_vertices = [vertices_body.copy()]  # list of arrays (we'll vstack at end)
    # We'll collect extra vertices (intersection points, caps) into extra_vertices list
    extra_vertices = []
    extra_triangles = []

    # Helper to add/reuse vertex for intersection points
    intersection_map = {}  # edge_key -> index (global index in final array will be offset)
    # We'll store intersection points in extra_vertices and map keys to their index (relative to final after stacking)
    # But we need a consistent local indexing while building triangles. We'll keep intersections indexed relative
    # to the extra_vertices array and convert later by adding base offset.

    # local list of triangles referencing body indices (initially body-only indices)
    triangles_local = []

    # Precompute a small function to get 1D index
    def vid(i_theta, j_axial):
        return i_theta * n_axial + j_axial

    z_cut = grid.get('z_cut', None)
    has_cut = grid.get('has_cut', False)

    # Tolerance check lambda
    def is_above(z):
        # consider points at or above plane as "kept"
        return (z - z_cut) >= -clip_tol

    # If no cut -> simple quad split like before
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

                # if r_j < 1e-10 and r_j1 > 1e-10:
                #     triangles_local.append([v0, v1, v3])
                #     triangles_local.append([v0, v3, v2])
                if r_j > 1e-10 and r_j1 > 1e-10:
                    # Normal quad, reversed winding to fix normals
                    triangles_body.append([v0, v2, v1])
                    triangles_body.append([v2, v3, v1])
                elif r_j1 < 1e-10 and r_j > 1e-10:
                    triangles_local.append([v1, v2, v0])
                    triangles_local.append([v1, v3, v2])
                elif r_j > 1e-10 and r_j1 > 1e-10:
                    triangles_local.append([v0, v1, v2])
                    triangles_local.append([v2, v1, v3])
                # else both tiny -> skip
        # We'll handle caps later (nose/tail), same as previous logic
    else:
        # Clipping logic: iterate quads and clip with plane z=z_cut
        # We'll also store intersection vertices in extra_vertices and reuse them via intersection_map.
        # For each quad, get its 4 points in consistent order (v0,v1,v3,v2) forming a loop.
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            for j in range(n_axial - 1):
                quad_idx = [vid(i, j), vid(i, j + 1), vid(i_next, j + 1), vid(i_next, j)]
                quad_pts = [vertices_body[idx] for idx in quad_idx]
                quad_z = [p[2] for p in quad_pts]
                above_flags = [is_above(z) for z in quad_z]

                if all(above_flags):
                    # keep the two triangles as before (no clipping)
                    # use same splitting ordering as earlier for consistency
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
                    # else skip
                elif not any(above_flags):
                    # completely below plane -> skip entire quad
                    continue
                else:
                    # Mixed: need to produce clipped polygon
                    # We'll follow quad in loop order: p0,p1,p2,p3 (note ordering)
                    poly_indices = []
                    poly_points = []

                    # There are 4 edges (p_k -> p_{k+1})
                    for k in range(4):
                        idx_k = quad_idx[k]
                        p_k = vertices_body[idx_k]
                        z_k = p_k[2]
                        keep_k = is_above(z_k)
                        # If vertex is above, include it
                        if keep_k:
                            poly_indices.append(idx_k)
                            poly_points.append(p_k.copy())

                        # Check edge to next vertex
                        next_k = (k + 1) % 4
                        idx_n = quad_idx[next_k]
                        p_n = vertices_body[idx_n]
                        z_n = p_n[2]
                        # if edge crosses plane (one above, one below), compute intersection
                        if (z_k - z_cut) * (z_n - z_cut) < -clip_tol:
                            # compute t so that z = z_cut: p = p_k + t*(p_n - p_k)
                            denom = (z_n - z_k)
                            if abs(denom) < 1e-16:
                                t = 0.5
                            else:
                                t = (z_cut - z_k) / denom
                            t = np.clip(t, 0.0, 1.0)
                            p_int = p_k + t * (p_n - p_k)
                            p_int[2] = z_cut  # enforce exact plane value
                            # Reuse intersection if present on this undirected edge
                            edge_key = tuple(sorted((int(idx_k), int(idx_n))))
                            if edge_key in intersection_map:
                                inter_idx = intersection_map[edge_key]
                            else:
                                inter_idx = len(extra_vertices)
                                extra_vertices.append(p_int)
                                intersection_map[edge_key] = inter_idx
                            # we will reference intersection by a special negative index scheme temporarily:
                            # store as negative (-(1 + inter_idx)) to convert later
                            poly_indices.append(-(1 + inter_idx))
                            poly_points.append(p_int)

                    # Now poly_points / poly_indices make up the clipped polygon in-order.
                    # Remove extremely small polygons
                    if len(poly_points) < 3:
                        continue

                    # Triangulate polygon by fan from first point (sufficient for clipped quads)
                    # But skip tiny-area triangles
                    for kk in range(1, len(poly_indices) - 1):
                        ia = poly_indices[0]
                        ib = poly_indices[kk]
                        ic = poly_indices[kk + 1]

                        # Convert indices to actual vertex positions (handle extra vertices)
                        def resolve_index(idx):
                            if idx >= 0:
                                return int(idx), vertices_body[int(idx)]
                            else:
                                ei = - (idx + 1)
                                return None, extra_vertices[ei]

                        a_idx_body, a_pt = resolve_index(ia)
                        b_idx_body, b_pt = resolve_index(ib)
                        c_idx_body, c_pt = resolve_index(ic)

                        # If all resolved to body indices, append directly to triangles_local
                        if (a_idx_body is not None) and (b_idx_body is not None) and (c_idx_body is not None):
                            tri = [a_idx_body, b_idx_body, c_idx_body]
                            # area check
                            area = np.linalg.norm(np.cross(b_pt - a_pt, c_pt - a_pt)) * 0.5
                            if area > 1e-14:
                                triangles_local.append(tri)
                        else:
                            # At least one vertex is from extra_vertices. We'll create triangle using global appended indices later.
                            # For now store triangle as tuple of possibly-negative indices
                            extra_triangles.append([ia, ib, ic])

        # End quad loop

    # Convert local triangles and extra triangles to global indices by stacking arrays
    # Final vertex ordering: [vertices_body, extra_vertices, (caps/nose/tail if added later)]
    extra_vertices_arr = np.array(extra_vertices) if extra_vertices else np.zeros((0, 3), dtype=float)
    all_vertices_arr = np.vstack([all_vertices[0], extra_vertices_arr]) if extra_vertices_arr.size else all_vertices[0].copy()

    # Build triangles list converting negative indices for extra vertices
    triangles_out = []
    for tri in triangles_local:
        triangles_out.append([int(tri[0]), int(tri[1]), int(tri[2])])

    base_extra_idx = len(all_vertices[0])  # offset where extra_vertices start

    for tri in extra_triangles:
        conv = []
        for idx in tri:
            if idx >= 0:
                conv.append(int(idx))
            else:
                ei = -(idx + 1)
                conv.append(base_extra_idx + ei)
        # area filter
        a, b, c = [all_vertices_arr[ii] for ii in conv]
        area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
        if area > 1e-14:
            triangles_out.append(conv)

    # Now handle nose and tail caps (these were previously added referencing only vertices_body indices)
    # We'll re-create them but make sure indices are adjusted to match all_vertices_arr
    # Nose cap
    if add_nose_cap:
        x_nose = grid['x'][0]
        first_ring = vertices_body[0:n_axial * n_circ:n_axial]  # first axial ring, shape (n_circ,3)
        nose_radius = np.linalg.norm(first_ring[0, 1:])
        if nose_radius < 1e-8:
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

    # Tail cap
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

    # Build flat bottom cap if we have intersection vertices on plane
    if has_cut and extra_vertices_arr.size:
        # Intersection vertices reside in all_vertices_arr[base_extra_idx : base_extra_idx + len(extra_vertices)]
        cut_verts_indices = list(range(base_extra_idx, base_extra_idx + len(extra_vertices)))
        if len(cut_verts_indices) >= 3:
            cut_pts = all_vertices_arr[cut_verts_indices]
            # sort by angle around centroid in Y-X plane (x,y) to create consistent polygon
            centroid = cut_pts.mean(axis=0)
            angles = np.arctan2(cut_pts[:, 1] - centroid[1], cut_pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            ordered_idx = [cut_verts_indices[k] for k in order]
            # create fan triangulation (center at centroid_on_plane)
            center = centroid.copy()
            center[2] = z_cut
            center_idx = len(all_vertices_arr)
            all_vertices_arr = np.vstack([all_vertices_arr, center.reshape(1, 3)])
            n_cut = len(ordered_idx)
            for k in range(n_cut):
                v1 = ordered_idx[k]
                v2 = ordered_idx[(k + 1) % n_cut]
                # triangle (center, v2, v1) to point normal inward (down) consistent with earlier code
                tri_pts = np.array([all_vertices_arr[center_idx], all_vertices_arr[v2], all_vertices_arr[v1]])
                area = np.linalg.norm(np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])) * 0.5
                if area > 1e-14:
                    triangles_out.append([center_idx, v2, v1])

    # Convert triangles_out to array and orient normals
    if len(triangles_out) == 0:
        triangles_arr = np.zeros((0, 3), dtype=int)
    else:
        triangles_arr = np.array(triangles_out, dtype=int)

    # Ensure triangle winding / orientation
    triangles_arr = orient_triangle_normals(all_vertices_arr, triangles_arr)

    return all_vertices_arr, triangles_arr



def generate_parametric_body_mesh(body, n_axial=100, n_circumferential=60,
                                  add_nose_cap=True, add_tail_cap=True,
                                  merge_tolerance=1e-9, improve_quality=False,
                                  validate_monotonic=True):
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
    validate_monotonic : bool
        Check that radius doesn't decrease (prevents pinching)
    
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
    
    # Validate monotonic if requested
    if validate_monotonic:
        print(f"\n[0/5] Validating monotonic radius...")
        is_monotonic = body.validate_monotonic(n_samples=200)
        if is_monotonic:
            print(f"  ✓ Radius is monotonically non-decreasing")
        else:
            print(f"  ✗ WARNING: Radius decreases in some regions (pinching detected)")
    
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
    r_nose = [rho - np.sqrt(max(0, rho**2 - (x * length)**2)) for x in x_nose]
    
    # Body section
    x_body = np.linspace(nose_length_frac, 1.0, 5)
    r_body = [radius] * len(x_body)
    
    control_points = list(zip(
        np.concatenate([x_nose[:-1], x_body]),
        r_nose[:-1] + r_body
    ))
    
    return ParametricBody(
        length, control_points, z_cut, z_squash,
        enforce_monotonic=True,
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
        enforce_monotonic=True,
        name=f"Lifting Body (L={length}, W={max_width*2})"
    )


if __name__ == "__main__":
    print("Parametric Body Generator - FIXED VERSION (No Pinching)")
    print("="*70)
    
    # Example 2: Missile with flat bottom (cut plane)
    print("\n" + "="*70)
    print("EXAMPLE 2: Missile with Flat Bottom")
    print("="*70)
    
    missile_cut = create_missile_body(length=1, diameter=0.2, z_cut=-0.)
    v2, t2, s2 = generate_parametric_body_mesh(missile_cut, n_axial=80, n_circumferential=40)
    write_tri_file("missile_flat_bottom_fixed.tri", v2, t2)
    
    # Example 4: Lifting body (flat bottom + elliptical)
    print("\n" + "="*70)
    print("EXAMPLE 4: Lifting Body")
    print("="*70)
    
    lifting = create_lifting_body(length=1, max_width=0.2, z_cut=-0.5, z_squash=0.3)
    v4, t4, s4 = generate_parametric_body_mesh(lifting, n_axial=100, n_circumferential=60)
    write_tri_file("lifting_body_fixed.tri", v4, t4)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nKey improvements:")
    print("  1. Monotonic radius enforcement BEFORE spline fitting")
    print("  2. PCHIP interpolation for smooth, non-oscillating curves")
    print("  3. Validation checks to detect any remaining pinching")
    print("  4. Better handling of degenerate triangles at nose/tail")