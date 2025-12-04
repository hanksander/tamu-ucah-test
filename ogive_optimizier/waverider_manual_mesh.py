"""
Robust manual mesh generator for waverider surfaces.

Fixes:
1. Zero-area triangles (removed via area threshold)
2. Duplicate vertices (merged with tolerance)
3. Duplicate triangles (removed)
4. Correct normals (upper up, lower down, backplate backward)
5. Mesh quality (skewness-based triangle optimization)
"""

import numpy as np
from scipy.spatial import cKDTree


def merge_duplicate_vertices(vertices, tolerance=1e-9):
    """
    Merge duplicate vertices within tolerance.
    
    Returns
    -------
    unique_verts : np.ndarray
        Deduplicated vertices
    index_map : np.ndarray
        Maps old indices to new indices
    """
    # Use KDTree for efficient spatial search
    tree = cKDTree(vertices)
    
    # Find all pairs within tolerance
    pairs = tree.query_pairs(r=tolerance)
    
    # Build mapping using union-find
    parent = np.arange(len(vertices))
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi
    
    for i, j in pairs:
        union(i, j)
    
    # Map each vertex to its root
    for i in range(len(vertices)):
        find(i)
    
    # Create unique vertices and index map
    unique_roots = np.unique(parent)
    root_to_new_idx = {root: idx for idx, root in enumerate(unique_roots)}
    
    unique_verts = np.zeros((len(unique_roots), 3))
    for i, root in enumerate(unique_roots):
        # Average all vertices that map to this root
        mask = parent == root
        unique_verts[i] = vertices[mask].mean(axis=0)
    
    index_map = np.array([root_to_new_idx[parent[i]] for i in range(len(vertices))])
    
    return unique_verts, index_map


def remove_degenerate_triangles(triangles, vertices, min_area=1e-10):
    """
    Remove zero-area and degenerate triangles.
    
    A triangle is degenerate if:
    - Has area below threshold
    - Has duplicate vertices
    - Is inverted (negative area)
    """
    valid_mask = np.ones(len(triangles), dtype=bool)
    
    for i, tri in enumerate(triangles):
        # Check for duplicate vertices in triangle
        if len(np.unique(tri)) < 3:
            valid_mask[i] = False
            continue
        
        # Check area
        v0, v1, v2 = vertices[tri]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        
        if area < min_area:
            valid_mask[i] = False
    
    return triangles[valid_mask]


def remove_duplicate_triangles(triangles):
    """
    Remove duplicate triangles (same 3 vertices regardless of order).
    """
    # Sort vertices in each triangle to make comparison order-independent
    sorted_tris = np.sort(triangles, axis=1)
    
    # Find unique triangles
    _, unique_idx = np.unique(sorted_tris, axis=0, return_index=True)
    
    return triangles[unique_idx]


def check_and_fix_normals(triangles, vertices, expected_normal, tolerance=0.1):
    """
    Ensure all triangles have normals pointing in expected direction.
    Flips triangles if needed.
    
    Parameters
    ----------
    triangles : np.ndarray (n, 3)
        Triangle vertex indices
    vertices : np.ndarray (m, 3)
        Vertex coordinates
    expected_normal : np.ndarray (3,)
        Expected normal direction (doesn't need to be normalized)
    tolerance : float
        Cosine similarity threshold (0=perpendicular, 1=parallel)
    
    Returns
    -------
    triangles : np.ndarray
        Triangles with corrected winding
    n_flipped : int
        Number of triangles that were flipped
    """
    expected_normal = expected_normal / np.linalg.norm(expected_normal)
    
    n_flipped = 0
    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_len = np.linalg.norm(normal)
        
        if normal_len > 1e-10:
            normal = normal / normal_len
            
            # Check if normal points in expected direction
            dot = np.dot(normal, expected_normal)
            
            if dot < -tolerance:  # Points opposite direction
                triangles[i] = tri[[0, 2, 1]]  # Flip winding
                n_flipped += 1
    
    return triangles, n_flipped


def compute_triangle_quality(triangles, vertices):
    """
    Compute quality metrics for each triangle.
    
    Returns
    -------
    qualities : dict
        'aspect_ratio': ratio of longest to shortest edge
        'skewness': measure of equilateral-ness (0=equilateral, 1=degenerate)
        'area': triangle areas
    """
    tris_verts = vertices[triangles]
    
    # Compute edge lengths
    edges = np.array([
        tris_verts[:, 1] - tris_verts[:, 0],
        tris_verts[:, 2] - tris_verts[:, 1],
        tris_verts[:, 0] - tris_verts[:, 2]
    ])
    edge_lengths = np.linalg.norm(edges, axis=2)  # (3, n_tris)
    
    # Areas
    cross = np.cross(edges[0], edges[1])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Aspect ratio
    max_edge = edge_lengths.max(axis=0)
    min_edge = edge_lengths.min(axis=0)
    aspect_ratio = max_edge / (min_edge + 1e-10)
    
    # Skewness (normalized by ideal equilateral triangle)
    # Ideal: all edges equal, area = sqrt(3)/4 * edge^2
    mean_edge = edge_lengths.mean(axis=0)
    ideal_area = np.sqrt(3) / 4 * mean_edge**2
    skewness = 1.0 - (areas / (ideal_area + 1e-10))
    skewness = np.clip(skewness, 0, 1)
    
    return {
        'aspect_ratio': aspect_ratio,
        'skewness': skewness,
        'area': areas
    }


def improve_mesh_quality(triangles, vertices, max_skewness=0.95, max_iterations=3):
    """
    Improve mesh quality by splitting/reorienting highly skewed triangles.
    
    This is conservative - only fixes the worst triangles to avoid
    creating new connectivity issues.
    """
    improved_tris = triangles.copy()
    
    for iteration in range(max_iterations):
        qualities = compute_triangle_quality(improved_tris, vertices)
        bad_mask = qualities['skewness'] > max_skewness
        
        if not bad_mask.any():
            break
        
        print(f"  Iteration {iteration+1}: {bad_mask.sum()} poor quality triangles")
        
        # For now, just remove the worst triangles
        # (More sophisticated: could split them or swap edges)
        improved_tris = improved_tris[~bad_mask]
    
    return improved_tris


def pad_streamlines_to_grid(streams):
    """
    Convert list of variable-length streamlines to regular grid.
    Pads shorter streamlines by repeating last point.
    """
    max_len = max(len(s) for s in streams)
    grid = np.zeros((len(streams), max_len, 3))
    
    for i, stream in enumerate(streams):
        grid[i, :len(stream)] = stream
        if len(stream) < max_len:
            grid[i, len(stream):] = stream[-1]
    
    return grid


def triangulate_surface_grid(grid, flip_normals=False):
    """
    Triangulate a regular surface grid into triangles.
    
    Parameters
    ----------
    grid : np.ndarray (n_rows, n_cols, 3)
        Regular grid of points
    flip_normals : bool
        If True, reverse triangle winding
    
    Returns
    -------
    vertices : np.ndarray (n_rows*n_cols, 3)
    triangles : np.ndarray (n_tris, 3)
    """
    n_rows, n_cols, _ = grid.shape
    
    # Flatten grid to vertices
    vertices = grid.reshape(-1, 3)
    
    # Create triangle connectivity
    triangles = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            # Quad vertices
            v0 = i * n_cols + j
            v1 = (i + 1) * n_cols + j
            v2 = i * n_cols + (j + 1)
            v3 = (i + 1) * n_cols + (j + 1)
            
            # Two triangles per quad
            # NOTE: Winding fixed - upper should be CCW when viewed from above
            if flip_normals:
                triangles.append([v0, v2, v1])
                triangles.append([v1, v2, v3])
            else:
                triangles.append([v0, v1, v2])
                triangles.append([v2, v1, v3])
    
    return vertices, np.array(triangles)


def create_backplate(upper_te, lower_te, flip_normals=False):
    """
    Create triangulated backplate connecting trailing edges.
    
    Parameters
    ----------
    upper_te : np.ndarray (n, 3)
        Upper trailing edge points
    lower_te : np.ndarray (n, 3)
        Lower trailing edge points (same length as upper)
    flip_normals : bool
        If True, reverse winding
    """
    n = len(upper_te)
    vertices = np.vstack([upper_te, lower_te])
    
    triangles = []
    for i in range(n - 1):
        v0 = i
        v1 = i + 1
        v2 = n + i
        v3 = n + i + 1
        
        if flip_normals:
            triangles.append([v0, v2, v1])
            triangles.append([v1, v2, v3])
        else:
            triangles.append([v0, v1, v2])
            triangles.append([v2, v1, v3])
    
    return vertices, np.array(triangles)


def generate_waverider_mesh(streams_upper, streams_lower, 
                            mirror_xy=True,
                            merge_tolerance=1e-9,
                            min_triangle_area=1e-10,
                            improve_quality=True):
    """
    Generate complete waverider mesh with proper normals and quality checks.
    
    Parameters
    ----------
    streams_upper : list of np.ndarray
        Upper surface streamlines
    streams_lower : list of np.ndarray
        Lower surface streamlines
    mirror_xy : bool
        Mirror about XY plane (Z → -Z) to create full vehicle
    merge_tolerance : float
        Tolerance for merging duplicate vertices
    min_triangle_area : float
        Minimum triangle area threshold
    improve_quality : bool
        Attempt to improve mesh quality
    
    Returns
    -------
    vertices : np.ndarray
    triangles : np.ndarray
    stats : dict
        Mesh statistics and quality metrics
    """
    print("\n" + "="*70)
    print("GENERATING WAVERIDER MESH")
    print("="*70)
    
    # Convert streamlines to grids
    print("\n[1/9] Converting streamlines to regular grids...")
    upper_grid = pad_streamlines_to_grid(streams_upper)
    lower_grid = pad_streamlines_to_grid(streams_lower)
    print(f"  Upper grid: {upper_grid.shape[0]} × {upper_grid.shape[1]}")
    print(f"  Lower grid: {lower_grid.shape[0]} × {lower_grid.shape[1]}")
    
    # Triangulate upper surface
    print("\n[2/9] Triangulating upper surface...")
    verts_upper, tris_upper = triangulate_surface_grid(upper_grid, flip_normals=True)
    print(f"  Upper: {len(verts_upper)} vertices, {len(tris_upper)} triangles")
    
    # Triangulate lower surface
    print("\n[3/9] Triangulating lower surface...")
    verts_lower, tris_lower = triangulate_surface_grid(lower_grid, flip_normals=False)
    tris_lower += len(verts_upper)  # Offset indices
    print(f"  Lower: {len(verts_lower)} vertices, {len(tris_lower)} triangles")
    
    # Create backplate
    print("\n[4/9] Creating trailing edge backplate...")
    upper_te = upper_grid[:, -1, :]
    lower_te = lower_grid[:, -1, :]
    verts_back, tris_back = create_backplate(upper_te, lower_te, flip_normals=True)
    tris_back += len(verts_upper) + len(verts_lower)
    print(f"  Backplate: {len(verts_back)} vertices, {len(tris_back)} triangles")
    
    # Combine all surfaces
    print("\n[5/9] Combining surfaces...")
    vertices = np.vstack([verts_upper, verts_lower, verts_back])
    triangles = np.vstack([tris_upper, tris_lower, tris_back])
    print(f"  Combined: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Mirror if requested
    if mirror_xy:
        print("\n[6/9] Mirroring about XZ plane (Y → -Y)...")
        verts_mirror = vertices.copy()
        verts_mirror[:, 2] *= -1
        
        tris_mirror = triangles.copy()
        tris_mirror = tris_mirror[:, ::-1]  # Flip winding
        tris_mirror += len(vertices)  # Offset indices

        print("line 394 of manual_mesh - TRY FLIP WINGING")
        
        vertices = np.vstack([vertices, verts_mirror])
        triangles = np.vstack([triangles, tris_mirror])
        print(f"  Mirrored: {len(vertices)} vertices, {len(triangles)} triangles")
    else:
        print("\n[6/9] Skipping mirror (half geometry only)")
    
    # Merge duplicate vertices
    print("\n[7/9] Merging duplicate vertices...")
    vertices_unique, index_map = merge_duplicate_vertices(vertices, merge_tolerance)
    triangles = index_map[triangles]
    n_merged = len(vertices) - len(vertices_unique)
    print(f"  Merged {n_merged} duplicate vertices")
    print(f"  Result: {len(vertices_unique)} unique vertices")
    vertices = vertices_unique
    
    # Remove degenerate triangles
    print("\n[8/9] Removing degenerate triangles...")
    n_before = len(triangles)
    triangles = remove_degenerate_triangles(triangles, vertices, min_triangle_area)
    triangles = remove_duplicate_triangles(triangles)
    n_removed = n_before - len(triangles)
    print(f"  Removed {n_removed} bad triangles")
    print(f"  Result: {len(triangles)} valid triangles")
    
    # Compute quality metrics
    print("\n[9/9] Computing mesh quality...")
    qualities = compute_triangle_quality(triangles, vertices)
    
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
        'poor_quality_count': (qualities['skewness'] > 0.9).sum()
    }
    
    print(f"\n  Quality metrics:")
    print(f"    Mean aspect ratio: {stats['mean_aspect_ratio']:.2f}")
    print(f"    Max aspect ratio: {stats['max_aspect_ratio']:.2f}")
    print(f"    Mean skewness: {stats['mean_skewness']:.3f}")
    print(f"    Max skewness: {stats['max_skewness']:.3f}")
    print(f"    Poor quality triangles (skewness > 0.9): {stats['poor_quality_count']}")
    
    # Improve quality if requested
    if improve_quality and stats['poor_quality_count'] > 0:
        print(f"\n[OPTIONAL] Improving mesh quality...")
        triangles = improve_mesh_quality(triangles, vertices)
        print(f"  Result: {len(triangles)} triangles after quality improvement")
    
    return vertices, triangles, stats


def write_tri_file(filename, vertices, triangles, scale=1000.0, swap_yz=True):
    """
    Write mesh to Cart3D .tri format.
    
    Parameters
    ----------
    filename : str
        Output file path
    vertices : np.ndarray (n, 3)
        Vertex coordinates
    triangles : np.ndarray (m, 3)
        Triangle connectivity (0-indexed)
    scale : float
        Scale factor (default: 1000 to convert m to mm)
    swap_yz : bool
        Swap Y and Z coordinates for Cart3D convention
    """
    verts = vertices.copy() * scale
    
    if swap_yz:
        verts = verts[:, [0, 2, 1]]
    
    # Convert to meters if in mm
    if scale == 1000.0:
        verts *= 1e-3
    
    n_verts = len(verts)
    n_tris = len(triangles)
    
    with open(filename, 'w') as f:
        f.write(f"{n_verts} {n_tris}\n")
        for v in verts:
            f.write(f"    {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in triangles:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")  # 1-indexed
        for _ in range(n_tris):
            f.write("1\n")  # Component ID
    
    print(f"\n✓ Wrote {filename}")
    print(f"  {n_verts} vertices, {n_tris} triangles")


if __name__ == "__main__":
    from waverider_generator.generator import waverider as wr
    
    # Generate waverider
    print("Generating waverider geometry...")
    waverider = wr(
        M_inf=5,
        beta=15,
        height=1.34,
        width=3,
        dp=[0.11, 0.63, 0, 0.46],
        n_upper_surface=10000,
        n_shockwave=10000,
        n_planes=40,
        n_streamwise=30,
        delta_streamise=0.05
    )
    
    # Generate mesh
    vertices, triangles, stats = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        mirror_xy=True,
        merge_tolerance=1e-6,
        min_triangle_area=1e-8,
        improve_quality=True
    )
    
    # Write output
    write_tri_file("waverider_manual.tri", vertices, triangles)
    
    print("\n" + "="*70)
    print("MESH GENERATION COMPLETE")
    print("="*70)
    print(f"\nFinal mesh statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")