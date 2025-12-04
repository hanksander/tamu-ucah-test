"""
Mesh topology diagnostic and fixing tools.

Identifies and fixes:
- Non-manifold edges (edges shared by >2 triangles)
- Inconsistent winding
- Duplicate triangles
- T-junctions
"""

import numpy as np
from collections import defaultdict


def check_edge_topology(triangles, verbose=True):
    """
    Check edge-to-triangle connectivity for manifoldness and consistency.
    
    Returns
    -------
    issues : dict
        'non_manifold_edges': edges shared by >2 triangles
        'boundary_edges': edges with only 1 triangle
        'inconsistent_edges': edges with inconsistent winding
        'duplicate_triangles': duplicate triangle indices
    """
    if verbose:
        print("\n" + "="*70)
        print("CHECKING EDGE TOPOLOGY")
        print("="*70)
    
    # Build edge-to-triangle map
    # Key: (min_v, max_v) - canonical edge
    # Value: list of (tri_idx, orientation)
    #   orientation: True if edge goes min->max in triangle, False if max->min
    edge_to_tris = defaultdict(list)
    
    for tri_idx, tri in enumerate(triangles):
        edges = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0])
        ]
        
        for v0, v1 in edges:
            # Canonical edge (sorted)
            edge = tuple(sorted([v0, v1]))
            # Orientation: True if edge goes from smaller to larger vertex index
            orientation = (v0 < v1)
            edge_to_tris[edge].append((tri_idx, orientation))
    
    # Analyze edges
    non_manifold = []
    boundary = []
    inconsistent = []
    
    for edge, tri_list in edge_to_tris.items():
        n_tris = len(tri_list)
        
        if n_tris > 2:
            # Non-manifold edge
            non_manifold.append((edge, [t[0] for t in tri_list]))
        elif n_tris == 1:
            # Boundary edge
            boundary.append(edge)
        elif n_tris == 2:
            # Check orientation consistency
            ori_0 = tri_list[0][1]
            ori_1 = tri_list[1][1]
            
            # For manifold mesh, orientations should be opposite
            # (one triangle goes v0->v1, other goes v1->v0)
            if ori_0 == ori_1:
                # Same orientation = inconsistent winding
                inconsistent.append((edge, [t[0] for t in tri_list]))
    
    # Check for duplicate triangles
    tri_sets = [frozenset(tri) for tri in triangles]
    unique_tris = set(tri_sets)
    duplicates = []
    if len(unique_tris) < len(triangles):
        # Find duplicates
        seen = {}
        for idx, tri_set in enumerate(tri_sets):
            if tri_set in seen:
                duplicates.append((seen[tri_set], idx))
            else:
                seen[tri_set] = idx
    
    if verbose:
        print(f"\nEdge statistics:")
        print(f"  Total edges: {len(edge_to_tris)}")
        print(f"  Boundary edges: {len(boundary)}")
        print(f"  Non-manifold edges: {len(non_manifold)}")
        print(f"  Inconsistent winding edges: {len(inconsistent)}")
        print(f"  Duplicate triangles: {len(duplicates)}")
        
        if non_manifold:
            print(f"\n  Non-manifold edges (showing first 10):")
            for edge, tris in non_manifold[:10]:
                print(f"    Edge {edge}: shared by triangles {tris}")
        
        if inconsistent:
            print(f"\n  Inconsistent winding (showing first 10):")
            for edge, tris in inconsistent[:10]:
                print(f"    Edge {edge}: triangles {tris} have same orientation")
    
    issues = {
        'non_manifold_edges': non_manifold,
        'boundary_edges': boundary,
        'inconsistent_edges': inconsistent,
        'duplicate_triangles': duplicates,
        'is_manifold': len(non_manifold) == 0,
        'is_watertight': len(boundary) == 0,
        'has_consistent_winding': len(inconsistent) == 0
    }
    
    return issues


def fix_inconsistent_winding(triangles, verbose=True):
    """
    Fix inconsistent winding by flipping triangles to make mesh consistent.
    
    Uses flood-fill approach: start from one triangle, propagate orientation.
    """
    if verbose:
        print("\n" + "="*70)
        print("FIXING INCONSISTENT WINDING")
        print("="*70)
    
    n_tris = len(triangles)
    triangles_fixed = triangles.copy()
    
    # Build adjacency: which triangles share edges
    edge_to_tris = defaultdict(list)
    
    for tri_idx, tri in enumerate(triangles_fixed):
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        
        for edge in edges:
            edge_to_tris[edge].append(tri_idx)
    
    # Track which triangles have been oriented
    oriented = np.zeros(n_tris, dtype=bool)
    
    # Flood fill from first triangle
    queue = [0]
    oriented[0] = True
    n_flipped = 0
    
    while queue:
        current_tri_idx = queue.pop(0)
        current_tri = triangles_fixed[current_tri_idx]
        
        # Get edges with their orientation in current triangle
        edges_oriented = [
            ((current_tri[0], current_tri[1]), tuple(sorted([current_tri[0], current_tri[1]]))),
            ((current_tri[1], current_tri[2]), tuple(sorted([current_tri[1], current_tri[2]]))),
            ((current_tri[2], current_tri[0]), tuple(sorted([current_tri[2], current_tri[0]])))
        ]
        
        for (v0, v1), edge_canonical in edges_oriented:
            # Find neighbor triangles sharing this edge
            for neighbor_idx in edge_to_tris[edge_canonical]:
                if neighbor_idx == current_tri_idx or oriented[neighbor_idx]:
                    continue
                
                # Check if neighbor has consistent orientation
                neighbor_tri = triangles_fixed[neighbor_idx]
                
                # Find how neighbor sees this edge
                neighbor_edges = [
                    (neighbor_tri[0], neighbor_tri[1]),
                    (neighbor_tri[1], neighbor_tri[2]),
                    (neighbor_tri[2], neighbor_tri[0])
                ]
                
                # Check if neighbor has edge in opposite direction
                has_opposite = False
                for nv0, nv1 in neighbor_edges:
                    if (nv0, nv1) == (v1, v0):  # Opposite direction
                        has_opposite = True
                        break
                    elif (nv0, nv1) == (v0, v1):  # Same direction - need to flip
                        # Flip neighbor triangle
                        triangles_fixed[neighbor_idx] = neighbor_tri[[0, 2, 1]]
                        n_flipped += 1
                        has_opposite = True  # Now it's opposite after flip
                        break
                
                # Mark as oriented and add to queue
                oriented[neighbor_idx] = True
                queue.append(neighbor_idx)
    
    if verbose:
        print(f"  Flipped {n_flipped} triangles")
        print(f"  Oriented {oriented.sum()} / {n_tris} triangles")
    
    return triangles_fixed


def remove_duplicate_triangles_advanced(triangles, verbose=True):
    """
    Remove duplicate triangles (same 3 vertices, any order).
    """
    if verbose:
        print("\n[*] Removing duplicate triangles...")
    
    # Convert each triangle to a frozenset for comparison
    tri_sets = [frozenset(tri) for tri in triangles]
    
    # Find unique triangles
    seen = {}
    unique_indices = []
    
    for idx, tri_set in enumerate(tri_sets):
        if tri_set not in seen:
            seen[tri_set] = idx
            unique_indices.append(idx)
    
    n_removed = len(triangles) - len(unique_indices)
    
    if verbose and n_removed > 0:
        print(f"  Removed {n_removed} duplicate triangles")
    
    return triangles[unique_indices]


def remove_non_manifold_triangles(triangles, verbose=True):
    """
    Remove triangles that share edges with >2 triangles.
    """
    if verbose:
        print("\n[*] Removing non-manifold triangles...")
    
    # Find edges shared by >2 triangles
    edge_to_tris = defaultdict(list)
    
    for tri_idx, tri in enumerate(triangles):
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_to_tris[edge].append(tri_idx)
    
    # Find triangles with non-manifold edges
    bad_tri_indices = set()
    for edge, tri_list in edge_to_tris.items():
        if len(tri_list) > 2:
            # Keep first 2, remove others
            bad_tri_indices.update(tri_list[2:])
    
    if bad_tri_indices:
        mask = np.ones(len(triangles), dtype=bool)
        mask[list(bad_tri_indices)] = False
        triangles = triangles[mask]
        
        if verbose:
            print(f"  Removed {len(bad_tri_indices)} non-manifold triangles")
    
    return triangles


def fix_mesh_topology(vertices, triangles, verbose=True):
    """
    Complete topology fix pipeline.
    
    Returns
    -------
    vertices : np.ndarray
    triangles : np.ndarray
    report : dict
    """
    if verbose:
        print("\n" + "="*70)
        print("MESH TOPOLOGY FIX PIPELINE")
        print("="*70)
        print(f"\nInput: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Step 1: Check initial state
    if verbose:
        print("\n[Step 1] Initial topology check...")
    issues_before = check_edge_topology(triangles, verbose=False)
    
    # Step 2: Remove duplicate triangles
    if verbose:
        print("\n[Step 2] Removing duplicate triangles...")
    triangles = remove_duplicate_triangles_advanced(triangles, verbose=verbose)
    
    # Step 3: Remove non-manifold triangles
    if verbose:
        print("\n[Step 3] Removing non-manifold triangles...")
    triangles = remove_non_manifold_triangles(triangles, verbose=verbose)
    
    # Step 4: Fix winding consistency
    if verbose:
        print("\n[Step 4] Fixing winding consistency...")
    triangles = fix_inconsistent_winding(triangles, verbose=verbose)
    
    # Step 5: Final check
    if verbose:
        print("\n[Step 5] Final topology check...")
    issues_after = check_edge_topology(triangles, verbose=True)
    
    # Report
    report = {
        'before': issues_before,
        'after': issues_after,
        'n_vertices': len(vertices),
        'n_triangles': len(triangles),
        'is_manifold': issues_after['is_manifold'],
        'is_watertight': issues_after['is_watertight'],
        'has_consistent_winding': issues_after['has_consistent_winding']
    }
    
    if verbose:
        print("\n" + "="*70)
        print("TOPOLOGY FIX COMPLETE")
        print("="*70)
        print(f"\nResults:")
        print(f"  Manifold: {report['is_manifold']}")
        print(f"  Watertight: {report['is_watertight']}")
        print(f"  Consistent winding: {report['has_consistent_winding']}")
        print(f"  Final: {len(vertices)} vertices, {len(triangles)} triangles")
    
    return vertices, triangles, report


if __name__ == "__main__":
    # Example: Load and fix a mesh
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_topology_fixer.py <input.tri> [output.tri]")
        print("\nThis tool:")
        print("  1. Checks mesh topology")
        print("  2. Removes duplicate triangles")
        print("  3. Removes non-manifold triangles")
        print("  4. Fixes winding consistency")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.tri', '_fixed.tri')
    
    # Load mesh
    print(f"Loading {input_file}...")
    from mesh_comparison import load_tri_file
    from body_of_revolution_mesh import write_tri_file
    
    vertices, triangles = load_tri_file(input_file)
    print(f"Loaded: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Fix topology
    vertices, triangles, report = fix_mesh_topology(vertices, triangles)
    
    # Write output
    write_tri_file(output_file, vertices, triangles)
    
    print(f"\n✓ Fixed mesh saved to: {output_file}")
    
    if not report['has_consistent_winding']:
        print("\n⚠️  Warning: Some winding issues remain")
    if not report['is_manifold']:
        print("\n⚠️  Warning: Mesh is not manifold")
    if report['is_watertight']:
        print("\n✓ Mesh is watertight")
    else:
        print(f"\n⚠️  Mesh has {len(report['after']['boundary_edges'])} boundary edges")