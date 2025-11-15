"""
Compare different meshing approaches and visualize results.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_tri_file(filename):
    """Load a Cart3D .tri file."""
    with open(filename, 'r') as f:
        # First line: n_verts n_tris
        n_verts, n_tris = map(int, f.readline().split())
        
        # Read vertices
        vertices = np.zeros((n_verts, 3))
        for i in range(n_verts):
            vertices[i] = list(map(float, f.readline().split()))
        
        # Read triangles (1-indexed, convert to 0-indexed)
        triangles = np.zeros((n_tris, 3), dtype=int)
        for i in range(n_tris):
            triangles[i] = [int(x)-1 for x in f.readline().split()]
        
    return vertices, triangles


def check_normal_directions(triangles, vertices):
    """
    Check if normals point in expected directions for waverider surfaces.
    
    Returns dict with average normals for different regions.
    """
    # Compute all triangle normals
    tris_verts = vertices[triangles]
    edge1 = tris_verts[:, 1] - tris_verts[:, 0]
    edge2 = tris_verts[:, 2] - tris_verts[:, 0]
    normals = np.cross(edge1, edge2)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-10)
    
    # Compute centroids
    centroids = tris_verts.mean(axis=1)
    
    # Separate by region (assume Z is vertical)
    z_mid = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    
    upper_mask = centroids[:, 2] > z_mid
    lower_mask = centroids[:, 2] <= z_mid
    
    # Average normals
    upper_normal = normals[upper_mask].mean(axis=0) if upper_mask.any() else np.array([0, 0, 0])
    lower_normal = normals[lower_mask].mean(axis=0) if lower_mask.any() else np.array([0, 0, 0])
    
    # Normalize
    if np.linalg.norm(upper_normal) > 0:
        upper_normal = upper_normal / np.linalg.norm(upper_normal)
    if np.linalg.norm(lower_normal) > 0:
        lower_normal = lower_normal / np.linalg.norm(lower_normal)
    
    return {
        'upper_normal': upper_normal,
        'lower_normal': lower_normal,
        'upper_count': upper_mask.sum(),
        'lower_count': lower_mask.sum()
    }


def check_normals_consistency(triangles, vertices):
    """
    Check if normals are consistently oriented.
    Returns fraction of triangles with consistent winding.
    """
    from collections import defaultdict
    
    # Build edge orientation count
    edge_orientations = defaultdict(lambda: {'fwd': 0, 'bwd': 0})
    
    for tri in triangles:
        edges = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0])
        ]
        
        for v0, v1 in edges:
            edge = tuple(sorted([v0, v1]))
            if v0 < v1:
                edge_orientations[edge]['fwd'] += 1
            else:
                edge_orientations[edge]['bwd'] += 1
    
    # Count consistent edges (same orientation in all triangles)
    consistent = 0
    total = 0
    for counts in edge_orientations.values():
        if counts['fwd'] > 0 and counts['bwd'] > 0:
            total += 1
        else:
            consistent += 1
            total += 1
    
    return consistent / total if total > 0 else 0


def check_manifoldness(triangles, n_vertices):
    """
    Check if mesh is manifold (each edge shared by at most 2 triangles).
    """
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
    
    # Count non-manifold edges
    non_manifold = sum(1 for count in edge_count.values() if count > 2)
    
    return {
        'is_manifold': non_manifold == 0,
        'non_manifold_edges': non_manifold,
        'total_edges': len(edge_count)
    }


def check_watertightness(triangles, n_vertices):
    """
    Check if mesh is watertight (all edges shared by exactly 2 triangles).
    """
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
    
    return {
        'is_watertight': boundary_edges == 0,
        'boundary_edges': boundary_edges,
        'total_edges': len(edge_count)
    }


def comprehensive_mesh_analysis(vertices, triangles, name="Mesh"):
    """
    Perform comprehensive analysis of mesh quality and topology.
    """
    from waverider_manual_mesh import compute_triangle_quality
    
    print("\n" + "="*70)
    print(f"COMPREHENSIVE ANALYSIS: {name}")
    print("="*70)
    
    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Vertices:  {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    print(f"  Edges:     ~{len(triangles) * 3 // 2}")
    
    # Bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    print(f"\nBounding Box:")
    print(f"  X: [{bbox_min[0]:.3f}, {bbox_max[0]:.3f}] (size: {bbox_max[0]-bbox_min[0]:.3f})")
    print(f"  Y: [{bbox_min[1]:.3f}, {bbox_max[1]:.3f}] (size: {bbox_max[1]-bbox_min[1]:.3f})")
    print(f"  Z: [{bbox_min[2]:.3f}, {bbox_max[2]:.3f}] (size: {bbox_max[2]-bbox_min[2]:.3f})")
    
    # Topology checks
    print(f"\nTopology:")
    manifold_info = check_manifoldness(triangles, len(vertices))
    watertight_info = check_watertightness(triangles, len(vertices))
    
    print(f"  Manifold: {'✓ Yes' if manifold_info['is_manifold'] else '✗ No'}")
    if not manifold_info['is_manifold']:
        print(f"    Non-manifold edges: {manifold_info['non_manifold_edges']}")
    
    print(f"  Watertight: {'✓ Yes' if watertight_info['is_watertight'] else '✗ No'}")
    if not watertight_info['is_watertight']:
        print(f"    Boundary edges: {watertight_info['boundary_edges']}")
    
    normal_consistency = check_normals_consistency(triangles, vertices)
    print(f"  Normal consistency: {normal_consistency*100:.1f}%")
    
    # Quality metrics
    print(f"\nQuality Metrics:")
    qualities = compute_triangle_quality(triangles, vertices)
    
    print(f"  Aspect Ratio:")
    print(f"    Mean: {qualities['aspect_ratio'].mean():.2f}")
    print(f"    Max:  {qualities['aspect_ratio'].max():.2f}")
    
    print(f"  Skewness (0=best, 1=worst):")
    print(f"    Mean:   {qualities['skewness'].mean():.3f}")
    print(f"    Median: {np.median(qualities['skewness']):.3f}")
    print(f"    Max:    {qualities['skewness'].max():.3f}")
    
    print(f"  Area:")
    print(f"    Min:  {qualities['area'].min():.6e}")
    print(f"    Mean: {qualities['area'].mean():.6e}")
    print(f"    Max:  {qualities['area'].max():.6e}")
    
    # Quality breakdown
    poor_quality = (qualities['skewness'] > 0.9).sum()
    bad_quality = (qualities['skewness'] > 0.95).sum()
    print(f"\n  Poor quality (skewness > 0.9):  {poor_quality} ({100*poor_quality/len(triangles):.1f}%)")
    print(f"  Bad quality (skewness > 0.95):   {bad_quality} ({100*bad_quality/len(triangles):.1f}%)")
    
    return {
        'n_vertices': len(vertices),
        'n_triangles': len(triangles),
        'is_manifold': manifold_info['is_manifold'],
        'is_watertight': watertight_info['is_watertight'],
        'normal_consistency': normal_consistency,
        'mean_aspect_ratio': qualities['aspect_ratio'].mean(),
        'max_aspect_ratio': qualities['aspect_ratio'].max(),
        'mean_skewness': qualities['skewness'].mean(),
        'max_skewness': qualities['skewness'].max(),
        'poor_quality_fraction': poor_quality / len(triangles),
        'qualities': qualities
    }


def plot_quality_comparison(results_dict, output_file='quality_comparison.png'):
    """
    Create comparison plots for multiple meshes.
    
    Parameters
    ----------
    results_dict : dict
        Keys are mesh names, values are result dicts from comprehensive_mesh_analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Skewness histograms
    ax = axes[0, 0]
    for name, results in results_dict.items():
        ax.hist(results['qualities']['skewness'], bins=50, alpha=0.5, label=name)
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Count')
    ax.set_title('Skewness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Aspect ratio histograms
    ax = axes[0, 1]
    for name, results in results_dict.items():
        ar = results['qualities']['aspect_ratio']
        ar_clipped = np.clip(ar, 0, 10)  # Clip for readability
        ax.hist(ar_clipped, bins=50, alpha=0.5, label=name)
    ax.set_xlabel('Aspect Ratio (clipped at 10)')
    ax.set_ylabel('Count')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Quality summary bars
    ax = axes[1, 0]
    names = list(results_dict.keys())
    mean_skew = [results_dict[n]['mean_skewness'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, mean_skew)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Mean Skewness')
    ax.set_title('Mean Skewness Comparison (lower is better)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Poor quality fraction
    ax = axes[1, 1]
    poor_frac = [results_dict[n]['poor_quality_fraction']*100 for n in names]
    ax.bar(x, poor_frac)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Poor Quality Triangles (%)')
    ax.set_title('Poor Quality (skewness > 0.9) Fraction')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_file}")
    plt.close()


def compare_before_after_optimization():
    """
    Generate and compare meshes before and after optimization.
    """
    from waverider_generator.generator import waverider as wr
    from waverider_manual_mesh import generate_waverider_mesh, write_tri_file
    from manual_mesh_optimizer import advanced_quality_improvement
    
    print("\n" + "="*70)
    print("GENERATING TEST MESHES FOR COMPARISON")
    print("="*70)
    
    # Generate waverider
    print("\nGenerating waverider geometry...")
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
    
    # Generate basic mesh
    print("\n" + "="*70)
    print("GENERATING BASIC MESH")
    print("="*70)
    vertices_basic, triangles_basic, _ = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        mirror_xy=True,
        improve_quality=False
    )
    write_tri_file("waverider_basic.tri", vertices_basic, triangles_basic)
    
    # Generate optimized mesh
    print("\n" + "="*70)
    print("GENERATING OPTIMIZED MESH")
    print("="*70)
    vertices_opt, triangles_opt = advanced_quality_improvement(
        vertices_basic.copy(),
        triangles_basic.copy(),
        edge_swap_iters=10,
        smoothing_iters=5,
        smoothing_factor=0.3
    )
    write_tri_file("waverider_optimized.tri", vertices_opt, triangles_opt)
    
    # Analyze both
    results = {}
    results['Basic'] = comprehensive_mesh_analysis(vertices_basic, triangles_basic, "Basic Mesh")
    results['Optimized'] = comprehensive_mesh_analysis(vertices_opt, triangles_opt, "Optimized Mesh")
    
    # Create comparison plots
    plot_quality_comparison(results)
    
    # Print improvement summary
    print("\n" + "="*70)
    print("OPTIMIZATION IMPROVEMENTS")
    print("="*70)
    
    basic = results['Basic']
    opt = results['Optimized']
    
    print(f"\nMean Skewness:")
    print(f"  Before: {basic['mean_skewness']:.4f}")
    print(f"  After:  {opt['mean_skewness']:.4f}")
    print(f"  Improvement: {(1 - opt['mean_skewness']/basic['mean_skewness'])*100:.1f}%")
    
    print(f"\nMax Skewness:")
    print(f"  Before: {basic['max_skewness']:.4f}")
    print(f"  After:  {opt['max_skewness']:.4f}")
    print(f"  Improvement: {(1 - opt['max_skewness']/basic['max_skewness'])*100:.1f}%")
    
    print(f"\nPoor Quality Triangles (skewness > 0.9):")
    print(f"  Before: {basic['poor_quality_fraction']*100:.2f}%")
    print(f"  After:  {opt['poor_quality_fraction']*100:.2f}%")
    print(f"  Reduction: {(1 - opt['poor_quality_fraction']/basic['poor_quality_fraction'])*100:.1f}%")
    
    print(f"\nMean Aspect Ratio:")
    print(f"  Before: {basic['mean_aspect_ratio']:.2f}")
    print(f"  After:  {opt['mean_aspect_ratio']:.2f}")
    print(f"  Improvement: {(1 - opt['mean_aspect_ratio']/basic['mean_aspect_ratio'])*100:.1f}%")
    
    return results


if __name__ == "__main__":
    # Run comparison
    results = compare_before_after_optimization()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - waverider_basic.tri (basic mesh)")
    print("  - waverider_optimized.tri (optimized mesh)")
    print("  - quality_comparison.png (comparison plots)")
    print("\nUse the optimized mesh for better panel code results!")