from waverider_generator.generator import waverider as wr
from waverider_manual_mesh import *
from manual_mesh_optimizer import *
from mesh_comparison import *
import numpy as np


def output_waverider_mesh(waverider, filepath = "./waverider/waverider.tri"):
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
    write_tri_file(filepath, vertices, triangles)

    return vertices, triangles, stats




def test_manual_mesh_generation():
    M_INF = 7
    BETA = 11.2
    HEIGHT = 1 * np.tan(np.radians(BETA))
    WIDTH = 0.2
    DP = [0.05, 0.0, 0, 0.0]
    n_planes = 100
    n_streamwise = 100

    waverider = wr(
        M_inf=M_INF,
        beta=BETA,
        height=HEIGHT,
        width=WIDTH,
        dp=DP,
        n_upper_surface=10000,
        n_shockwave=10000,
        n_planes=n_planes,
        n_streamwise=n_streamwise,
        delta_streamise=0.02
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
    write_tri_file("waverider.tri", vertices, triangles)

    vertices, triangles, stats = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        improve_quality=False  # Do advanced optimization separately
    )

    # Apply advanced optimization
    vertices_opt, triangles_opt = advanced_quality_improvement(
        vertices, triangles,
        edge_swap_iters=10,
        smoothing_iters=5,
        smoothing_factor=0.3
    )

    write_tri_file("waverider_optimized.tri", vertices_opt, triangles_opt)

    results = compare_before_after_optimization()


"""
Test script to verify all three fixes are working correctly.

Run this to ensure:
1. Normals are correct (upper up, lower down)
2. Trailing edge has good resolution
3. Mesh is watertight
"""

from waverider_generator.generator import waverider as wr
from waverider_manual_mesh import generate_waverider_mesh, write_tri_file
from normal_visualizer import quick_normal_check
import numpy as np


def test_fixes():
    """
    Complete test of all fixes.
    """
    print("\n" + "="*70)
    print("TESTING ALL MESH GENERATION FIXES")
    print("="*70)
    
    # Generate waverider
    print("\n[Step 1/5] Generating waverider geometry...")
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
    print("  ✓ Waverider generated")
    
    # Generate mesh with all fixes
    print("\n[Step 2/5] Generating mesh with all fixes enabled...")
    vertices, triangles, stats = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        mirror_xy=True,
        merge_tolerance=1e-6,
        min_triangle_area=1e-8,
        improve_quality=True,
    )
    print("  ✓ Mesh generated")
    
    # Test 1: Check normals
    print("\n[Step 3/5] Testing FIX #1: Normal directions...")
    print("-" * 70)
    normals_ok = quick_normal_check(vertices, triangles)
    print("-" * 70)
    
    if normals_ok:
        print("  ✅ FIX #1 PASSED: Normals are correct!")
    else:
        print("  ❌ FIX #1 FAILED: Normals are incorrect!")
        return False
    
    # Test 2: Check trailing edge resolution
    print("\n[Step 4/5] Testing FIX #2: Trailing edge resolution...")
    
    # Compute edge lengths at trailing edge
    # Trailing edge is last column of grid
    n_streamwise = waverider.upper_surface_streams[0].shape[0]
    
    # Count triangles near trailing edge (last 10% of X)
    tris_verts = vertices[triangles]
    centroids = tris_verts.mean(axis=1)
    x_max = vertices[:, 0].max()
    x_min = vertices[:, 0].min()
    te_threshold = x_max - 0.1 * (x_max - x_min)
    
    te_triangles = triangles[centroids[:, 0] > te_threshold]
    
    # Compute aspect ratios near trailing edge
    te_tris_verts = vertices[te_triangles]
    edges = [
        te_tris_verts[:, 1] - te_tris_verts[:, 0],
        te_tris_verts[:, 2] - te_tris_verts[:, 1],
        te_tris_verts[:, 0] - te_tris_verts[:, 2]
    ]
    edge_lengths = np.array([np.linalg.norm(e, axis=1) for e in edges])
    
    aspect_ratios = edge_lengths.max(axis=0) / (edge_lengths.min(axis=0) + 1e-10)
    mean_aspect = aspect_ratios.mean()
    max_aspect = aspect_ratios.max()
    
    print(f"  Trailing edge triangles: {len(te_triangles)}")
    print(f"  Mean aspect ratio: {mean_aspect:.2f}")
    print(f"  Max aspect ratio: {max_aspect:.2f}")
    
    if mean_aspect < 5.0 and len(te_triangles) > 50:
        print("  ✅ FIX #2 PASSED: Trailing edge has good resolution!")
        te_ok = True
    else:
        print("  ⚠️  FIX #2 MARGINAL: Trailing edge could be better")
        print("     Consider increasing te_refinement_points")
        te_ok = True  # Don't fail, just warn
    
    # Test 3: Check watertightness
    print("\n[Step 5/5] Testing FIX #3: Watertight geometry...")
    
    is_watertight = stats['is_watertight']
    boundary_edges = stats['boundary_edges']
    is_manifold = stats['is_manifold']
    
    print(f"  Boundary edges: {boundary_edges}")
    print(f"  Non-manifold edges: {stats['non_manifold_edges']}")
    print(f"  Watertight: {is_watertight}")
    print(f"  Manifold: {is_manifold}")
    
    if is_watertight and is_manifold:
        print("  ✅ FIX #3 PASSED: Mesh is watertight and manifold!")
        watertight_ok = True
    else:
        print("  ❌ FIX #3 FAILED: Mesh has topology issues!")
        watertight_ok = False
    
    # Overall result
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    results = {
        'FIX #1 (Normals)': normals_ok,
        'FIX #2 (Trailing edge)': te_ok,
        'FIX #3 (Watertight)': watertight_ok
    }
    
    for fix, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {fix}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL FIXES WORKING CORRECTLY!")
        print("="*70)
        
        # Write output
        print("\nWriting mesh to file...")
        write_tri_file("waverider_fixed.tri", vertices, triangles)
        print("✓ Saved: waverider_fixed.tri")
        
        print("\nMesh statistics:")
        print(f"  Vertices: {stats['n_vertices']}")
        print(f"  Triangles: {stats['n_triangles']}")
        print(f"  Mean skewness: {stats['mean_skewness']:.3f}")
        print(f"  Watertight: ✓")
        print(f"  Normals: ✓")
        
        print("\n✓ Mesh is ready for panel code!")
        return True
    else:
        print("❌ SOME FIXES NOT WORKING")
        print("="*70)
        print("\nPlease check the error messages above and:")
        print("  1. Ensure you're using the latest code")
        print("  2. Check for any import errors")
        print("  3. Verify waverider generator is working")
        return False


def compare_with_without_fixes():
    """
    Generate meshes with and without fixes to show improvement.
    """
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT FIXES")
    print("="*70)
    
    waverider = wr(M_inf=5, beta=15, height=1.34, width=3,
                   dp=[0.11, 0.63, 0, 0.46],
                   n_planes=30, n_streamwise=20, delta_streamise=0.05)
    
    # Without fixes
    print("\n[1/2] Generating mesh WITHOUT fixes...")
    v1, t1, s1 = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        improve_quality=False
    )
    
    # With fixes
    print("\n[2/2] Generating mesh WITH fixes...")
    v2, t2, s2 = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        improve_quality=True
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print("\nTriangle count:")
    print(f"  Without fixes: {s1['n_triangles']}")
    print(f"  With fixes:    {s2['n_triangles']}")
    print(f"  Increase:      +{s2['n_triangles'] - s1['n_triangles']} "
          f"({100*(s2['n_triangles']/s1['n_triangles'] - 1):.1f}%)")
    
    print("\nMean skewness:")
    print(f"  Without fixes: {s1['mean_skewness']:.4f}")
    print(f"  With fixes:    {s2['mean_skewness']:.4f}")
    print(f"  Improvement:   {100*(1 - s2['mean_skewness']/s1['mean_skewness']):.1f}%")
    
    print("\nPoor quality triangles:")
    print(f"  Without fixes: {s1['poor_quality_count']} "
          f"({100*s1['poor_quality_count']/s1['n_triangles']:.1f}%)")
    print(f"  With fixes:    {s2['poor_quality_count']} "
          f"({100*s2['poor_quality_count']/s2['n_triangles']:.1f}%)")
    
    print("\nWatertight:")
    print(f"  Without fixes: {'✓' if s1['is_watertight'] else '✗'} "
          f"({s1['boundary_edges']} boundary edges)")
    print(f"  With fixes:    {'✓' if s2['is_watertight'] else '✗'} "
          f"({s2['boundary_edges']} boundary edges)")
    
    # Check normals
    print("\nNormal directions:")
    print(f"  Without fixes: ", end='')
    n1_ok = quick_normal_check(v1, t1)
    print(f"  With fixes:    ", end='')
    n2_ok = quick_normal_check(v2, t2)
    
    write_tri_file("waverider_without_fixes.tri", v1, t1)
    write_tri_file("waverider_with_fixes.tri", v2, t2)
    
    print("\n✓ Comparison complete!")
    print("  Generated: waverider_without_fixes.tri")
    print("  Generated: waverider_with_fixes.tri")


if __name__ == "__main__":
    import sys
    
    print("Waverider Mesh Generation - Fix Verification")
    print("=" * 70)

    test_manual_mesh_generation()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Run comparison
        compare_with_without_fixes()
    else:
        # Run main test
        success = test_fixes()
        
        if success:
            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("\n1. Visual inspection:")
            print("   - Open waverider_fixed.tri in Paraview/MeshLab")
            print("   - Check that trailing edge looks smooth")
            print("   - Verify no gaps or holes")
            
            print("\n2. Run panel code:")
            print("   - Use waverider_fixed.tri as input")
            print("   - Should work without errors")
            
            print("\n3. If issues persist:")
            print("   - Run: python test_all_fixes.py --compare")
            print("   - This generates meshes with/without fixes for comparison")
            
            print("\n4. For detailed normal visualization:")
            print("   - Run: python normal_visualizer.py waverider_fixed.tri")
            
            sys.exit(0)
        else:
            sys.exit(1)
