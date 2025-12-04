"""
Calculate reference parameters (Sref, cref, bref) for waverider geometries.

These are needed for normalizing aerodynamic coefficients in panel codes.
"""

import numpy as np


def compute_reference_parameters(vertices, triangles, method='projected', verbose=True):
    """
    Compute reference area, chord, and span for a waverider mesh.
    
    Parameters
    ----------
    vertices : np.ndarray (n, 3)
        Mesh vertices
    triangles : np.ndarray (m, 3)
        Triangle connectivity
    method : str
        'projected' - Project onto XY plane (typical for waveriders)
        'wetted' - Use actual wetted surface area
        'both' - Compute both and return dict
    verbose : bool
        Print detailed breakdown
    
    Returns
    -------
    params : dict
        'Sref': Reference area
        'cref': Reference chord (mean aerodynamic chord)
        'bref': Reference span
        'xref': Reference X location (area centroid)
        'yref': Reference Y location (should be ~0 for symmetric)
        'zref': Reference Z location (area centroid)
    """
    
    if verbose:
        print("\n" + "="*70)
        print("COMPUTING REFERENCE PARAMETERS")
        print("="*70)
    
    # Get bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Separate upper and lower surfaces by Z coordinate
    tris_verts = vertices[triangles]
    centroids = tris_verts.mean(axis=1)
    z_mid = (bbox_min[2] + bbox_max[2]) / 2
    
    upper_mask = centroids[:, 2] > z_mid
    lower_mask = centroids[:, 2] <= z_mid
    
    # Also separate by X (leading vs trailing edge)
    x_mid = (bbox_min[0] + bbox_max[0]) / 2
    leading_mask = centroids[:, 0] > x_mid
    trailing_mask = centroids[:, 0] <= x_mid
    
    # Main surfaces (not backplate or leading edge)
    main_surface_mask = upper_mask | lower_mask
    
    if verbose:
        print(f"\nMesh overview:")
        print(f"  Total triangles: {len(triangles)}")
        print(f"  Upper surface: {upper_mask.sum()}")
        print(f"  Lower surface: {lower_mask.sum()}")
        print(f"  Other (edges): {len(triangles) - upper_mask.sum() - lower_mask.sum()}")
    
    # ===== Method 1: Projected Area =====
    if method in ['projected', 'both']:
        # Project triangles onto XY plane (plan view)
        # This is typical for waveriders - reference area is planform area
        
        projected_areas = []
        projected_centroids = []
        
        for i, mask in enumerate([upper_mask, lower_mask]):
            if not mask.any():
                continue
                
            tris_subset = triangles[mask]
            tris_verts_subset = vertices[tris_subset]
            
            # Project onto XY plane (Z = 0)
            tris_xy = tris_verts_subset[:, :, [0, 1]]  # Keep only X, Y
            
            # Compute areas of projected triangles
            v0 = tris_xy[:, 0]
            v1 = tris_xy[:, 1]
            v2 = tris_xy[:, 2]
            
            # Area = 0.5 * |cross product|
            edge1 = v1 - v0
            edge2 = v2 - v0
            # For 2D: cross product magnitude is just the Z component
            cross_z = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
            areas = 0.5 * np.abs(cross_z)
            
            projected_areas.append(areas)
            
            # Centroids for MAC calculation (use 3D centroids)
            tri_centroids = tris_verts_subset.mean(axis=1)
            projected_centroids.append(tri_centroids)
        
        # Combine areas (only count once since upper and lower project to same area)
        # Take maximum projection (usually upper surface for waverider)
        if len(projected_areas) == 2:
            # We have both surfaces - they project to overlapping areas
            # Use upper surface area (compression surface)
            Sref_proj = projected_areas[0].sum()
        else:
            Sref_proj = sum(a.sum() for a in projected_areas)
        
        # Combine centroids weighted by area
        all_areas = np.concatenate(projected_areas) if len(projected_areas) > 0 else np.array([])
        all_centroids = np.vstack(projected_centroids) if len(projected_centroids) > 0 else np.zeros((0, 3))
        
        if len(all_areas) > 0:
            total_area_proj = all_areas.sum()
            area_weighted_centroid = (all_centroids.T @ all_areas) / total_area_proj
        else:
            area_weighted_centroid = np.array([0, 0, 0])
    
    # ===== Method 2: Wetted Surface Area =====
    if method in ['wetted', 'both']:
        # Use actual 3D surface area (upper + lower)
        
        wetted_areas = []
        wetted_centroids = []
        
        for mask in [upper_mask, lower_mask]:
            if not mask.any():
                continue
            
            tris_subset = triangles[mask]
            tris_verts_subset = vertices[tris_subset]
            
            # Compute 3D areas
            edge1 = tris_verts_subset[:, 1] - tris_verts_subset[:, 0]
            edge2 = tris_verts_subset[:, 2] - tris_verts_subset[:, 0]
            cross = np.cross(edge1, edge2)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            wetted_areas.append(areas)
            
            tri_centroids = tris_verts_subset.mean(axis=1)
            wetted_centroids.append(tri_centroids)
        
        Sref_wetted = sum(a.sum() for a in wetted_areas)
        
        all_areas_wet = np.concatenate(wetted_areas) if len(wetted_areas) > 0 else np.array([])
        all_centroids_wet = np.vstack(wetted_centroids) if len(wetted_centroids) > 0 else np.zeros((0, 3))
        
        if len(all_areas_wet) > 0:
            area_weighted_centroid_wet = (all_centroids_wet.T @ all_areas_wet) / Sref_wetted
        else:
            area_weighted_centroid_wet = np.array([0, 0, 0])
    
    # ===== Reference Span (bref) =====
    # Maximum span in Y direction
    bref = bbox_max[1] - bbox_min[1]
    
    if verbose:
        print(f"\nReference span (bref):")
        print(f"  Y range: [{bbox_min[1]:.6f}, {bbox_max[1]:.6f}]")
        print(f"  bref = {bref:.6f}")
    
    # ===== Reference Chord (cref) =====
    # Mean Aerodynamic Chord (MAC)
    # For waverider: use planform projection
    
    if method in ['projected', 'both']:
        # MAC from projected planform
        # Simplified: use streamwise extent weighted by local span
        
        # Divide span into strips
        n_strips = 50
        y_edges = np.linspace(bbox_min[1], bbox_max[1], n_strips + 1)
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        dy = y_edges[1] - y_edges[0]
        
        local_chords = []
        local_spans = []
        
        for yc in y_centers:
            # Find all vertices in this strip
            in_strip = (vertices[:, 1] >= yc - dy/2) & (vertices[:, 1] < yc + dy/2)
            
            if in_strip.any():
                strip_verts = vertices[in_strip]
                x_min_strip = strip_verts[:, 0].min()
                x_max_strip = strip_verts[:, 0].max()
                local_chord = x_max_strip - x_min_strip
                local_chords.append(local_chord)
                local_spans.append(dy)
            else:
                local_chords.append(0)
                local_spans.append(dy)
        
        local_chords = np.array(local_chords)
        local_spans = np.array(local_spans)
        
        # MAC = ∫(c²dy) / ∫(c dy) 
        mac_numerator = (local_chords**2 * local_spans).sum()
        mac_denominator = (local_chords * local_spans).sum()
        
        if mac_denominator > 0:
            cref_mac = mac_numerator / mac_denominator
        else:
            # Fallback: simple streamwise extent
            cref_mac = bbox_max[0] - bbox_min[0]
    else:
        # Simple streamwise extent
        cref_mac = bbox_max[0] - bbox_min[0]
    
    if verbose:
        print(f"\nReference chord (cref - Mean Aerodynamic Chord):")
        print(f"  X range: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}]")
        print(f"  MAC = {cref_mac:.6f}")
    
    # ===== Prepare Results =====
    if method == 'projected':
        Sref = Sref_proj
        xref, yref, zref = area_weighted_centroid
        
        if verbose:
            print(f"\nReference area (Sref - projected onto XY plane):")
            print(f"  Sref = {Sref:.6f}")
            print(f"\nArea centroid (reference point):")
            print(f"  xref = {xref:.6f}")
            print(f"  yref = {yref:.6f} (should be ≈ 0 for symmetric)")
            print(f"  zref = {zref:.6f}")
    
    elif method == 'wetted':
        Sref = Sref_wetted
        xref, yref, zref = area_weighted_centroid_wet
        
        if verbose:
            print(f"\nReference area (Sref - wetted surface):")
            print(f"  Sref = {Sref:.6f}")
            print(f"\nArea centroid (reference point):")
            print(f"  xref = {xref:.6f}")
            print(f"  yref = {yref:.6f} (should be ≈ 0 for symmetric)")
            print(f"  zref = {zref:.6f}")
    
    elif method == 'both':
        if verbose:
            print(f"\nReference area comparison:")
            print(f"  Projected (planform): {Sref_proj:.6f}")
            print(f"  Wetted (3D surface): {Sref_wetted:.6f}")
            print(f"  Ratio (wetted/projected): {Sref_wetted/Sref_proj:.3f}")
            print(f"\n  Recommendation: Use PROJECTED for waveriders")
        
        # Return both
        params = {
            'Sref_projected': Sref_proj,
            'Sref_wetted': Sref_wetted,
            'cref': cref_mac,
            'bref': bref,
            'xref': area_weighted_centroid[0],
            'yref': area_weighted_centroid[1],
            'zref': area_weighted_centroid[2],
            'xref_wetted': area_weighted_centroid_wet[0],
            'yref_wetted': area_weighted_centroid_wet[1],
            'zref_wetted': area_weighted_centroid_wet[2]
        }
        
        if verbose:
            print("\n" + "="*70)
            print("SUMMARY (using projected/planform values)")
            print("="*70)
            print(f"  Sref = {Sref_proj:.6f}")
            print(f"  cref = {cref_mac:.6f} (MAC)")
            print(f"  bref = {bref:.6f}")
            print(f"  Reference point: ({area_weighted_centroid[0]:.6f}, "
                  f"{area_weighted_centroid[1]:.6f}, {area_weighted_centroid[2]:.6f})")
            print("="*70)
        
        return params
    
    params = {
        'Sref': Sref,
        'cref': cref_mac,
        'bref': bref,
        'xref': xref,
        'yref': yref,
        'zref': zref
    }
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  Sref = {Sref:.6f}")
        print(f"  cref = {cref_mac:.6f} (MAC)")
        print(f"  bref = {bref:.6f}")
        print(f"  Reference point: ({xref:.6f}, {yref:.6f}, {zref:.6f})")
        print("="*70)
    
    return params


def write_reference_file(params, filename="reference_params.txt", notes=""):
    """
    Write reference parameters to a text file for use with panel codes.
    
    Parameters
    ----------
    params : dict
        Output from compute_reference_parameters()
    filename : str
        Output filename
    notes : str
        Additional notes to include
    """
    
    with open(filename, 'w') as f:
        f.write("# Waverider Reference Parameters\n")
        f.write("# For use with panel codes (Cart3D, etc.)\n")
        f.write("#\n")
        
        if notes:
            f.write(f"# {notes}\n#\n")
        
        f.write("# Reference Area (Sref):\n")
        if 'Sref_projected' in params:
            f.write(f"Sref (projected) = {params['Sref_projected']:.10e}\n")
            f.write(f"Sref (wetted)    = {params['Sref_wetted']:.10e}\n")
        else:
            f.write(f"Sref = {params['Sref']:.10e}\n")
        
        f.write(f"\n# Reference Chord (cref) - Mean Aerodynamic Chord:\n")
        f.write(f"cref = {params['cref']:.10e}\n")
        
        f.write(f"\n# Reference Span (bref):\n")
        f.write(f"bref = {params['bref']:.10e}\n")
        
        f.write(f"\n# Reference Point (area centroid):\n")
        if 'xref_wetted' in params:
            f.write(f"xref (projected) = {params['xref']:.10e}\n")
            f.write(f"yref (projected) = {params['yref']:.10e}\n")
            f.write(f"zref (projected) = {params['zref']:.10e}\n")
            f.write(f"xref (wetted)    = {params['xref_wetted']:.10e}\n")
            f.write(f"yref (wetted)    = {params['yref_wetted']:.10e}\n")
            f.write(f"zref (wetted)    = {params['zref_wetted']:.10e}\n")
        else:
            f.write(f"xref = {params['xref']:.10e}\n")
            f.write(f"yref = {params['yref']:.10e}\n")
            f.write(f"zref = {params['zref']:.10e}\n")
        
        f.write(f"\n# Aspect Ratio:\n")
        if 'Sref_projected' in params:
            AR = params['bref']**2 / params['Sref_projected']
        else:
            AR = params['bref']**2 / params['Sref']
        f.write(f"AR = {AR:.6f}\n")
        
        f.write(f"\n# For normalization:\n")
        f.write(f"# CL = Lift / (0.5 * rho * V^2 * Sref)\n")
        f.write(f"# CD = Drag / (0.5 * rho * V^2 * Sref)\n")
        f.write(f"# Cm = Moment / (0.5 * rho * V^2 * Sref * cref)\n")
    
    print(f"\n✓ Reference parameters written to: {filename}")


def compare_with_simple_estimates(vertices, params):
    """
    Compare computed reference parameters with simple geometric estimates.
    """
    print("\n" + "="*70)
    print("COMPARISON WITH SIMPLE ESTIMATES")
    print("="*70)
    
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Simple estimates
    L = bbox_max[0] - bbox_min[0]  # Length
    W = bbox_max[1] - bbox_min[1]  # Width (span)
    H = bbox_max[2] - bbox_min[2]  # Height
    
    # Rectangular planform estimate
    Sref_simple = L * W
    
    print(f"\nBounding box dimensions:")
    print(f"  Length (X): {L:.6f}")
    print(f"  Width  (Y): {W:.6f}")
    print(f"  Height (Z): {H:.6f}")
    
    print(f"\nSimple rectangular planform:")
    print(f"  Sref_simple = L × W = {Sref_simple:.6f}")
    
    Sref_actual = params.get('Sref', params.get('Sref_projected', 0))
    
    print(f"\nComputed (integrated from mesh):")
    print(f"  Sref = {Sref_actual:.6f}")
    
    print(f"\nDifference:")
    print(f"  Ratio: {Sref_actual / Sref_simple:.3f}")
    print(f"  (Waveriders typically have 0.6-0.8 × rectangular planform)")


if __name__ == "__main__":
    # Example usage
    from waverider_manual_mesh import generate_waverider_mesh, write_tri_file
    from waverider_generator.generator import waverider as wr
    
    print("Generating example waverider...")
    waverider = wr(
        M_inf=5,
        beta=15,
        height=1.34,
        width=3,
        dp=[0.11, 0.63, 0, 0.46],
        n_planes=40,
        n_streamwise=30,
        delta_streamise=0.05
    )
    
    print("\nGenerating mesh...")
    vertices, triangles, stats = generate_waverider_mesh(
        waverider.upper_surface_streams,
        waverider.lower_surface_streams,
        mirror_xz=True,
        refine_te=True,
        te_refinement_points=15
    )
    
    # Compute reference parameters
    params = compute_reference_parameters(vertices, triangles, method='both')
    
    # Compare with simple estimates
    compare_with_simple_estimates(vertices, params)
    
    # Write to file
    write_reference_file(
        params, 
        "waverider_reference.txt",
        notes="M_inf=5, beta=15°, Generated from manual mesh"
    )
    
    print("\n" + "="*70)
    print("READY FOR PANEL CODE")
    print("="*70)
    print("\nUse these values in your panel code input:")
    print(f"  Sref = {params['Sref_projected']:.6e}  (recommended: projected)")
    print(f"  cref = {params['cref']:.6e}")
    print(f"  bref = {params['bref']:.6e}")
    print(f"  xref = {params['xref']:.6e}")
    print(f"  yref = {params['yref']:.6e}")
    print(f"  zref = {params['zref']:.6e}")