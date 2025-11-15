"""
Visualize triangle normals to debug orientation issues.
"""

import numpy as np
import matplotlib.pyplot as plt

'''
from mpl_toolkits.mplot3d import Axes3D


def visualize_normals_3d(vertices, triangles, subsample=10, arrow_length=0.05):
    """
    Create 3D plot showing mesh with normal vectors.
    
    Parameters
    ----------
    vertices : np.ndarray
    triangles : np.ndarray
    subsample : int
        Only show every Nth triangle (for clarity)
    arrow_length : float
        Length of normal arrows relative to bounding box
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Compute scale based on bounding box
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    scale = bbox_size.max() * arrow_length
    
    # Compute triangle centers and normals
    tris_verts = vertices[triangles]
    centers = tris_verts.mean(axis=1)
    
    edge1 = tris_verts[:, 1] - tris_verts[:, 0]
    edge2 = tris_verts[:, 2] - tris_verts[:, 0]
    normals = np.cross(edge1, edge2)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-10) * scale
    
    # Separate by Z coordinate
    z_mid = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    upper_mask = centers[:, 2] > z_mid
    lower_mask = centers[:, 2] <= z_mid
    
    # Plot 1: Upper surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    upper_tris = triangles[upper_mask][::subsample]
    upper_centers = centers[upper_mask][::subsample]
    upper_normals = normals[upper_mask][::subsample]
    
    for tri in upper_tris:
        pts = vertices[tri]
        ax1.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], color='lightblue', alpha=0.3)
    
    ax1.quiver(upper_centers[:, 0], upper_centers[:, 1], upper_centers[:, 2],
               upper_normals[:, 0], upper_normals[:, 1], upper_normals[:, 2],
               color='red', length=1.0, normalize=False, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Upper Surface Normals\n(should point UP, +Z)')
    
    # Plot 2: Lower surface
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    lower_tris = triangles[lower_mask][::subsample]
    lower_centers = centers[lower_mask][::subsample]
    lower_normals = normals[lower_mask][::subsample]
    
    for tri in lower_tris:
        pts = vertices[tri]
        ax2.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], color='lightgreen', alpha=0.3)
    
    ax2.quiver(lower_centers[:, 0], lower_centers[:, 1], lower_centers[:, 2],
               lower_normals[:, 0], lower_normals[:, 1], lower_normals[:, 2],
               color='blue', length=1.0, normalize=False, arrow_length_ratio=0.3)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Lower Surface Normals\n(should point DOWN, -Z)')
    
    # Plot 3: Normal Z-component histogram
    ax3 = fig.add_subplot(2, 2, 3)
    
    all_normals = normals / scale  # Back to unit normals
    z_components = all_normals[:, 2]
    
    ax3.hist(z_components[upper_mask], bins=50, alpha=0.5, label='Upper surface', color='red')
    ax3.hist(z_components[lower_mask], bins=50, alpha=0.5, label='Lower surface', color='blue')
    ax3.axvline(0, color='black', linestyle='--', label='Z=0')
    ax3.set_xlabel('Normal Z-component')
    ax3.set_ylabel('Count')
    ax3.set_title('Normal Z-component Distribution\nUpper should be >0, Lower should be <0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    upper_avg = all_normals[upper_mask].mean(axis=0)
    lower_avg = all_normals[lower_mask].mean(axis=0)
    
    stats_text = f"""
Normal Statistics:

Upper Surface:
  Avg normal: [{upper_avg[0]:.3f}, {upper_avg[1]:.3f}, {upper_avg[2]:.3f}]
  Z > 0: {(z_components[upper_mask] > 0).sum()} / {upper_mask.sum()}
  Correct: {'✓ YES' if upper_avg[2] > 0.5 else '✗ NO'}

Lower Surface:
  Avg normal: [{lower_avg[0]:.3f}, {lower_avg[1]:.3f}, {lower_avg[2]:.3f}]
  Z < 0: {(z_components[lower_mask] < 0).sum()} / {lower_mask.sum()}
  Correct: {'✓ YES' if lower_avg[2] < -0.5 else '✗ NO'}

Expected:
  Upper: [0, 0, +1] (upward)
  Lower: [0, 0, -1] (downward)
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    return fig
'''

def quick_normal_check(vertices, triangles):
    """
    Quick text-based check of normal directions.
    """
    print("\n" + "="*70)
    print("NORMAL DIRECTION CHECK")
    print("="*70)
    
    # Compute normals
    tris_verts = vertices[triangles]
    centers = tris_verts.mean(axis=1)
    
    edge1 = tris_verts[:, 1] - tris_verts[:, 0]
    edge2 = tris_verts[:, 2] - tris_verts[:, 0]
    normals = np.cross(edge1, edge2)
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-10)
    
    # Separate by Z
    z_mid = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    upper_mask = centers[:, 2] > z_mid
    lower_mask = centers[:, 2] <= z_mid
    
    # Separate by X (leading vs trailing edge)
    x_mid = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
    leading_mask = centers[:, 0] > x_mid
    trailing_mask = centers[:, 0] <= x_mid
    
    # Check upper surface
    print("\nUpper Surface:")
    upper_normals = normals[upper_mask]
    upper_avg = upper_normals.mean(axis=0)
    print(f"  Average normal: [{upper_avg[0]:+.3f}, {upper_avg[1]:+.3f}, {upper_avg[2]:+.3f}]")
    print(f"  Expected:       [ 0.000,  0.000, +1.000] (upward)")
    
    z_positive = (upper_normals[:, 2] > 0).sum()
    z_total = len(upper_normals)
    print(f"  Triangles with Z > 0: {z_positive} / {z_total} ({100*z_positive/z_total:.1f}%)")
    
    if upper_avg[2] > 0.5:
        print(f"  Status: ✓ CORRECT (normals point upward)")
    else:
        print(f"  Status: ✗ WRONG (normals point {'downward' if upper_avg[2] < -0.5 else 'sideways'})")
    
    # Check lower surface
    print("\nLower Surface:")
    lower_normals = normals[lower_mask]
    lower_avg = lower_normals.mean(axis=0)
    print(f"  Average normal: [{lower_avg[0]:+.3f}, {lower_avg[1]:+.3f}, {lower_avg[2]:+.3f}]")
    print(f"  Expected:       [ 0.000,  0.000, -1.000] (downward)")
    
    z_negative = (lower_normals[:, 2] < 0).sum()
    z_total = len(lower_normals)
    print(f"  Triangles with Z < 0: {z_negative} / {z_total} ({100*z_negative/z_total:.1f}%)")
    
    if lower_avg[2] < -0.5:
        print(f"  Status: ✓ CORRECT (normals point downward)")
    else:
        print(f"  Status: ✗ WRONG (normals point {'upward' if lower_avg[2] > 0.5 else 'sideways'})")
    
    # Check trailing edge
    print("\nTrailing Edge (Backplate):")
    trailing_normals = normals[trailing_mask & ~upper_mask & ~lower_mask]
    if len(trailing_normals) > 0:
        trailing_avg = trailing_normals.mean(axis=0)
        print(f"  Average normal: [{trailing_avg[0]:+.3f}, {trailing_avg[1]:+.3f}, {trailing_avg[2]:+.3f}]")
        print(f"  Expected:       [-1.000,  0.000,  0.000] (backward)")
        
        if trailing_avg[0] < -0.5:
            print(f"  Status: ✓ CORRECT (normals point backward)")
        else:
            print(f"  Status: ✗ WRONG (normals point {'forward' if trailing_avg[0] > 0.5 else 'sideways'})")
    else:
        print(f"  No backplate triangles identified")
    
    print("\n" + "="*70)
    
    # Return verdict
    upper_ok = upper_avg[2] > 0.5
    lower_ok = lower_avg[2] < -0.5
    
    return upper_ok and lower_ok


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python normal_visualizer.py <mesh.tri>")
        print("\nOr use in code:")
        print("  from normal_visualizer import quick_normal_check")
        print("  quick_normal_check(vertices, triangles)")
        sys.exit(1)
    
    # Load mesh
    from mesh_comparison import load_tri_file
    
    tri_file = sys.argv[1]
    print(f"Loading {tri_file}...")
    vertices, triangles = load_tri_file(tri_file)
    
    # Quick check
    normals_ok = quick_normal_check(vertices, triangles)
    
    '''
    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_normals_3d(vertices, triangles, subsample=20)
    plt.savefig('normal_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: normal_visualization.png")
    
    plt.show()
    '''
    if not normals_ok:
        print("\n✗ NORMALS ARE INCORRECT - mesh needs fixing!")
        sys.exit(1)
    else:
        print("\n✓ Normals are correct!")
        sys.exit(0)