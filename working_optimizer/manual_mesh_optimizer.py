"""
Advanced mesh quality optimization via edge swapping.

This module provides tools to improve triangle mesh quality by:
- Identifying poor quality triangles
- Swapping edges to create better triangulations
- Smoothing vertex positions (Laplacian smoothing)
"""

import numpy as np
from collections import defaultdict


def build_edge_to_triangles_map(triangles):
    """
    Build mapping from edges to triangles that share them.
    
    Returns
    -------
    edge_to_tris : dict
        Maps (v0, v1) tuple to list of triangle indices
    """
    edge_to_tris = defaultdict(list)
    
    for tri_idx, tri in enumerate(triangles):
        # Three edges per triangle
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        
        for edge in edges:
            edge_to_tris[edge].append(tri_idx)
    
    return edge_to_tris


def compute_triangle_quality_single(tri_verts):
    """
    Compute quality metric for a single triangle.
    
    Returns lower values for better triangles.
    Uses ratio of circumradius to inradius (ideal equilateral = 2.0).
    """
    v0, v1, v2 = tri_verts
    
    # Edge vectors
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    
    # Edge lengths
    l0 = np.linalg.norm(e0)
    l1 = np.linalg.norm(e1)
    l2 = np.linalg.norm(e2)
    
    # Semiperimeter
    s = (l0 + l1 + l2) / 2
    
    # Area via Heron's formula
    area_sq = s * (s - l0) * (s - l1) * (s - l2)
    if area_sq <= 0:
        return 1e10  # Degenerate
    area = np.sqrt(area_sq)
    
    # Circumradius
    R = (l0 * l1 * l2) / (4 * area)
    
    # Inradius
    r = area / s
    
    # Quality metric (ideal = 2.0, worse > 2.0)
    if r < 1e-10:
        return 1e10
    
    return R / r


def can_swap_edge(edge, triangles, edge_to_tris):
    """
    Check if an edge can be swapped (must be shared by exactly 2 triangles).
    
    Returns
    -------
    can_swap : bool
    tri_indices : tuple of 2 ints (or None)
    quad_verts : list of 4 ints (or None)
        Vertices of the quad in order
    """
    if edge not in edge_to_tris:
        return False, None, None
    
    tri_list = edge_to_tris[edge]
    
    # Edge must be shared by exactly 2 triangles
    if len(tri_list) != 2:
        return False, None, None
    
    tri0_idx, tri1_idx = tri_list
    tri0 = triangles[tri0_idx]
    tri1 = triangles[tri1_idx]
    
    # Find the 4 vertices of the quad
    v_edge = set(edge)
    v_tri0_other = set(tri0) - v_edge
    v_tri1_other = set(tri1) - v_edge
    
    if len(v_tri0_other) != 1 or len(v_tri1_other) != 1:
        return False, None, None
    
    v0, v1 = edge
    v2 = list(v_tri0_other)[0]
    v3 = list(v_tri1_other)[0]
    
    return True, (tri0_idx, tri1_idx), [v0, v1, v2, v3]


def swap_edge(triangles, tri0_idx, tri1_idx, quad_verts, vertices):
    """
    Swap diagonal of a quad if it improves quality.
    
    Parameters
    ----------
    triangles : np.ndarray
        Triangle array (modified in place)
    tri0_idx, tri1_idx : int
        Indices of the two triangles
    quad_verts : list of 4 ints
        [v0, v1, v2, v3] where (v0,v1) is current diagonal
    vertices : np.ndarray
        Vertex coordinates
    
    Returns
    -------
    swapped : bool
        True if swap was performed
    """
    v0, v1, v2, v3 = quad_verts
    
    # Current triangles: (v0,v1,v2) and (v1,v0,v3)
    # Proposed triangles: (v0,v2,v3) and (v1,v3,v2)
    
    # Compute current quality
    tri0_verts = vertices[[v0, v1, v2]]
    tri1_verts = vertices[[v1, v0, v3]]
    q0_old = compute_triangle_quality_single(tri0_verts)
    q1_old = compute_triangle_quality_single(tri1_verts)
    quality_old = max(q0_old, q1_old)
    
    # Compute new quality
    tri0_new_verts = vertices[[v0, v2, v3]]
    tri1_new_verts = vertices[[v1, v3, v2]]
    q0_new = compute_triangle_quality_single(tri0_new_verts)
    q1_new = compute_triangle_quality_single(tri1_new_verts)
    quality_new = max(q0_new, q1_new)
    
    # Swap if improvement
    if quality_new < quality_old * 0.95:  # 5% improvement threshold
        triangles[tri0_idx] = [v0, v2, v3]
        triangles[tri1_idx] = [v1, v3, v2]
        return True
    
    return False


def optimize_mesh_via_edge_swapping(triangles, vertices, max_iterations=10):
    """
    Improve mesh quality by swapping edges.
    
    Returns
    -------
    triangles : np.ndarray
        Improved triangulation
    n_swaps : int
        Total number of swaps performed
    """
    print("\n  Optimizing via edge swapping...")
    triangles = triangles.copy()
    total_swaps = 0
    
    for iteration in range(max_iterations):
        edge_to_tris = build_edge_to_triangles_map(triangles)
        
        swaps_this_iter = 0
        edges_to_check = list(edge_to_tris.keys())
        
        for edge in edges_to_check:
            can_swap, tri_indices, quad_verts = can_swap_edge(edge, triangles, edge_to_tris)
            
            if can_swap:
                if swap_edge(triangles, tri_indices[0], tri_indices[1], quad_verts, vertices):
                    swaps_this_iter += 1
        
        total_swaps += swaps_this_iter
        print(f"    Iteration {iteration+1}: {swaps_this_iter} swaps")
        
        if swaps_this_iter == 0:
            break
    
    print(f"  Total swaps: {total_swaps}")
    return triangles, total_swaps


def laplacian_smoothing(vertices, triangles, iterations=5, factor=0.5, 
                       boundary_mask=None):
    """
    Smooth vertex positions using Laplacian smoothing.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates
    triangles : np.ndarray
        Triangle connectivity
    iterations : int
        Number of smoothing iterations
    factor : float
        Smoothing factor (0=no smoothing, 1=full Laplacian)
    boundary_mask : np.ndarray of bool, optional
        Mask of vertices to keep fixed (e.g., boundary vertices)
    
    Returns
    -------
    vertices : np.ndarray
        Smoothed vertices
    """
    print(f"\n  Laplacian smoothing ({iterations} iterations, factor={factor})...")
    
    vertices = vertices.copy()
    n_verts = len(vertices)
    
    if boundary_mask is None:
        boundary_mask = np.zeros(n_verts, dtype=bool)
    
    # Build adjacency
    neighbors = [set() for _ in range(n_verts)]
    for tri in triangles:
        for i in range(3):
            neighbors[tri[i]].add(tri[(i+1) % 3])
            neighbors[tri[i]].add(tri[(i+2) % 3])
    
    for it in range(iterations):
        new_vertices = vertices.copy()
        
        for i in range(n_verts):
            if boundary_mask[i]:
                continue
            
            if len(neighbors[i]) == 0:
                continue
            
            # Average of neighbors
            neighbor_list = list(neighbors[i])
            centroid = vertices[neighbor_list].mean(axis=0)
            
            # Move toward centroid
            new_vertices[i] = vertices[i] + factor * (centroid - vertices[i])
        
        vertices = new_vertices
    
    return vertices


def identify_boundary_vertices(triangles, n_vertices):
    """
    Identify vertices on mesh boundary (edges with only 1 adjacent triangle).
    """
    edge_count = defaultdict(int)
    
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_count[edge] += 1
    
    # Boundary edges have count == 1
    boundary_verts = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_verts.update(edge)
    
    boundary_mask = np.zeros(n_vertices, dtype=bool)
    boundary_mask[list(boundary_verts)] = True
    
    return boundary_mask


def advanced_quality_improvement(vertices, triangles, 
                                edge_swap_iters=5,
                                smoothing_iters=3,
                                smoothing_factor=0.3):
    """
    Apply multiple quality improvement techniques.
    
    Returns
    -------
    vertices : np.ndarray
        Improved vertex positions
    triangles : np.ndarray  
        Improved connectivity
    """
    print("\n  Advanced quality improvement...")
    
    # Step 1: Edge swapping
    triangles, n_swaps = optimize_mesh_via_edge_swapping(
        triangles, vertices, max_iterations=edge_swap_iters
    )
    
    # Step 2: Identify boundary (don't smooth these)
    boundary_mask = identify_boundary_vertices(triangles, len(vertices))
    n_boundary = boundary_mask.sum()
    print(f"  Identified {n_boundary} boundary vertices (will not be smoothed)")
    
    # Step 3: Laplacian smoothing
    vertices = laplacian_smoothing(
        vertices, triangles,
        iterations=smoothing_iters,
        factor=smoothing_factor,
        boundary_mask=boundary_mask
    )
    
    return vertices, triangles


def visualize_quality_distribution(triangles, vertices, filename=None):
    """
    Print quality distribution statistics and optionally save histogram data.
    """
    from waverider_manual_mesh import compute_triangle_quality
    
    qualities = compute_triangle_quality(triangles, vertices)
    
    print("\n" + "="*70)
    print("MESH QUALITY DISTRIBUTION")
    print("="*70)
    
    # Aspect ratio
    print("\nAspect Ratio:")
    print(f"  Min:    {qualities['aspect_ratio'].min():.2f}")
    print(f"  Mean:   {qualities['aspect_ratio'].mean():.2f}")
    print(f"  Median: {np.median(qualities['aspect_ratio']):.2f}")
    print(f"  Max:    {qualities['aspect_ratio'].max():.2f}")
    
    # Skewness
    print("\nSkewness (0=perfect, 1=degenerate):")
    print(f"  Min:    {qualities['skewness'].min():.3f}")
    print(f"  Mean:   {qualities['skewness'].mean():.3f}")
    print(f"  Median: {np.median(qualities['skewness']):.3f}")
    print(f"  Max:    {qualities['skewness'].max():.3f}")
    
    # Quality bins
    bins = [0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]
    hist, _ = np.histogram(qualities['skewness'], bins=bins)
    
    print("\nQuality distribution:")
    print("  Excellent (< 0.3):  ", hist[0], f"({100*hist[0]/len(triangles):.1f}%)")
    print("  Good (0.3-0.6):     ", hist[1], f"({100*hist[1]/len(triangles):.1f}%)")
    print("  Fair (0.6-0.8):     ", hist[2], f"({100*hist[2]/len(triangles):.1f}%)")
    print("  Poor (0.8-0.9):     ", hist[3], f"({100*hist[3]/len(triangles):.1f}%)")
    print("  Bad (0.9-0.95):     ", hist[4], f"({100*hist[4]/len(triangles):.1f}%)")
    print("  Very bad (> 0.95):  ", hist[5], f"({100*hist[5]/len(triangles):.1f}%)")
    
    if filename:
        np.savez(filename,
                 aspect_ratio=qualities['aspect_ratio'],
                 skewness=qualities['skewness'],
                 area=qualities['area'])
        print(f"\n✓ Saved quality data to {filename}")


if __name__ == "__main__":
    from waverider_generator.generator import waverider as wr
    from waverider_manual_mesh import generate_waverider_mesh, write_tri_file
    
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
    
    # Generate initial mesh
    vertices, triangles, stats = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        mirror_xy=True,
        improve_quality=False  # We'll do advanced improvement
    )
    
    print("\n" + "="*70)
    print("INITIAL MESH QUALITY")
    print("="*70)
    visualize_quality_distribution(triangles, vertices)
    
    # Apply advanced improvement
    print("\n" + "="*70)
    print("APPLYING ADVANCED QUALITY IMPROVEMENTS")
    print("="*70)
    vertices_improved, triangles_improved = advanced_quality_improvement(
        vertices, triangles,
        edge_swap_iters=10,
        smoothing_iters=5,
        smoothing_factor=0.3
    )
    
    print("\n" + "="*70)
    print("IMPROVED MESH QUALITY")
    print("="*70)
    visualize_quality_distribution(triangles_improved, vertices_improved)
    
    # Write output
    write_tri_file("waverider_optimized.tri", vertices_improved, triangles_improved)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)