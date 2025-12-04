"""
Shape library and batch generator for bodies of revolution.

Includes common aerospace shapes:
- Reentry vehicles (Apollo, blunt body)
- Missiles/Rockets (Von Karman, Haack series)
- Aircraft components (nacelles, fuselage sections)
- Canonical shapes (cone, cylinder, sphere)
"""

import numpy as np
from body_of_revolution_mesh import (
    generate_bor_mesh, write_tri_file, 
    r_cone, r_cylinder, r_ogive
)


class BORShape:
    """Base class for body of revolution shapes."""
    
    def __init__(self, name, length, base_radius):
        self.name = name
        self.length = length
        self.base_radius = base_radius
    
    def r(self, x):
        """Override in subclass"""
        raise NotImplementedError
    
    def generate_mesh(self, n_axial=100, n_circumferential=60, **kwargs):
        """Generate mesh for this shape."""
        vertices, triangles, stats = generate_bor_mesh(
            r_func=self.r,
            x_range=(0, self.length),
            n_axial=n_axial,
            n_circumferential=n_circumferential,
            **kwargs
        )
        return vertices, triangles, stats
    
    def fineness_ratio(self):
        """Length / diameter ratio"""
        return self.length / (2 * self.base_radius)


# ============================================================================
# CANONICAL SHAPES
# ============================================================================

class Cone(BORShape):
    """Simple cone with pointed nose."""
    
    def __init__(self, length, base_radius):
        super().__init__("Cone", length, base_radius)
    
    def r(self, x):
        return (self.base_radius / self.length) * x


class Cylinder(BORShape):
    """Right circular cylinder."""
    
    def __init__(self, length, radius):
        super().__init__("Cylinder", length, radius)
    
    def r(self, x):
        return self.radius


class Sphere(BORShape):
    """Complete sphere."""
    
    def __init__(self, radius):
        super().__init__("Sphere", 2*radius, radius)
        self.radius = radius
    
    def r(self, x):
        x_center = self.length / 2
        r_sq = self.radius**2 - (x - x_center)**2
        return np.sqrt(max(0, r_sq))


# ============================================================================
# ROCKET NOSE CONES
# ============================================================================

class TangentOgive(BORShape):
    """Tangent ogive nose cone (common for rockets)."""
    
    def __init__(self, length, base_radius):
        super().__init__("Tangent Ogive", length, base_radius)
        self.rho = (base_radius**2 + length**2) / (2 * base_radius)
    
    def r(self, x):
        return self.rho - np.sqrt(self.rho**2 - x**2)


class SecantOgive(BORShape):
    """Secant ogive nose cone."""
    
    def __init__(self, length, base_radius, rho_factor=1.5):
        super().__init__("Secant Ogive", length, base_radius)
        # rho_factor > 1 for secant ogive
        self.rho = rho_factor * (base_radius**2 + length**2) / (2 * base_radius)
    
    def r(self, x):
        x0 = self.length - np.sqrt(self.rho**2 - self.base_radius**2)
        return np.sqrt(self.rho**2 - (x - x0)**2)


class VonKarmanNose(BORShape):
    """Von Karman nose cone (minimum drag for given length/diameter)."""
    
    def __init__(self, length, base_radius):
        super().__init__("Von Karman (LD-Haack)", length, base_radius)
    
    def r(self, x):
        theta = np.arccos(1 - 2*x/self.length)
        return (self.base_radius / np.sqrt(np.pi)) * np.sqrt(
            theta - np.sin(2*theta)/2
        )


class LVHaackNose(BORShape):
    """LV-Haack nose cone (minimum drag for given volume)."""
    
    def __init__(self, length, base_radius):
        super().__init__("LV-Haack", length, base_radius)
    
    def r(self, x):
        C = 1/3
        theta = np.arccos(1 - 2*x/self.length)
        return (self.base_radius / np.sqrt(np.pi)) * np.sqrt(
            theta - np.sin(2*theta)/2 + C * np.sin(theta)**3
        )


class PowerLawNose(BORShape):
    """Power law nose cone: r = R * (x/L)^n"""
    
    def __init__(self, length, base_radius, n=0.5):
        super().__init__(f"Power Law (n={n})", length, base_radius)
        self.n = n
    
    def r(self, x):
        return self.base_radius * (x / self.length)**self.n


class ParabolicNose(BORShape):
    """Parabolic nose cone."""
    
    def __init__(self, length, base_radius, K=0.75):
        super().__init__("Parabolic", length, base_radius)
        self.K = K  # Shape parameter
    
    def r(self, x):
        return self.base_radius * ((x / self.length)**2 - 
                                  K * (x / self.length) + K * (x / self.length)**2)


class EllipticalNose(BORShape):
    """Elliptical nose cone."""
    
    def __init__(self, length, base_radius):
        super().__init__("Elliptical", length, base_radius)
    
    def r(self, x):
        return self.base_radius * np.sqrt(1 - (x / self.length)**2)


# ============================================================================
# REENTRY VEHICLES
# ============================================================================

class BluntCone(BORShape):
    """Blunt cone (common for reentry vehicles)."""
    
    def __init__(self, length, base_radius, nose_radius):
        super().__init__("Blunt Cone", length, base_radius)
        self.nose_radius = nose_radius
    
    def r(self, x):
        # Spherical nose cap
        if x <= self.nose_radius:
            return self.nose_radius - np.sqrt(self.nose_radius**2 - x**2)
        else:
            # Conical afterbody
            x_cone = x - self.nose_radius
            l_cone = self.length - self.nose_radius
            r_junction = self.nose_radius
            return r_junction + (self.base_radius - r_junction) * (x_cone / l_cone)


class Biconic(BORShape):
    """Biconic reentry vehicle (e.g., early capsules)."""
    
    def __init__(self, length, base_radius, x_junction_frac=0.3, r_junction_frac=0.5):
        super().__init__("Biconic", length, base_radius)
        self.x_junction = length * x_junction_frac
        self.r_junction = base_radius * r_junction_frac
    
    def r(self, x):
        if x <= self.x_junction:
            # First cone (nose)
            return (self.r_junction / self.x_junction) * x
        else:
            # Second cone (aft body)
            slope = (self.base_radius - self.r_junction) / (self.length - self.x_junction)
            return self.r_junction + slope * (x - self.x_junction)


class ApolloCommand(BORShape):
    """Approximation of Apollo Command Module shape."""
    
    def __init__(self, length=3.9, base_radius=1.95):
        super().__init__("Apollo CM", length, base_radius)
        self.nose_radius = 4.7  # Heat shield radius
    
    def r(self, x):
        # Spherical section heat shield
        x_center = -self.nose_radius + self.length
        r_sq = self.nose_radius**2 - (x - x_center)**2
        if r_sq > 0:
            return np.sqrt(r_sq)
        else:
            return self.base_radius


# ============================================================================
# COMPOSITE SHAPES
# ============================================================================

class NoseConeBody(BORShape):
    """Nose cone with cylindrical body."""
    
    def __init__(self, nose_length, body_length, radius, nose_type='ogive'):
        total_length = nose_length + body_length
        super().__init__(f"{nose_type.title()} + Cylinder", total_length, radius)
        self.nose_length = nose_length
        self.body_length = body_length
        self.nose_type = nose_type
    
    def r(self, x):
        if x <= self.nose_length:
            # Nose section
            if self.nose_type == 'ogive':
                rho = (self.base_radius**2 + self.nose_length**2) / (2 * self.base_radius)
                return rho - np.sqrt(rho**2 - x**2)
            elif self.nose_type == 'cone':
                return (self.base_radius / self.nose_length) * x
            elif self.nose_type == 'haack':
                theta = np.arccos(1 - 2*x/self.nose_length)
                return (self.base_radius / np.sqrt(np.pi)) * np.sqrt(
                    theta - np.sin(2*theta)/2
                )
            else:
                raise ValueError(f"Unknown nose type: {self.nose_type}")
        else:
            # Cylindrical body
            return self.base_radius


class BoattailBody(BORShape):
    """Cylindrical body with boattail (tapered aft section)."""
    
    def __init__(self, body_length, boattail_length, body_radius, tail_radius):
        total_length = body_length + boattail_length
        super().__init__("Boattail Body", total_length, body_radius)
        self.body_length = body_length
        self.boattail_length = boattail_length
        self.tail_radius = tail_radius
    
    def r(self, x):
        if x <= self.body_length:
            return self.base_radius
        else:
            # Boattail (linear taper)
            frac = (x - self.body_length) / self.boattail_length
            return self.base_radius + frac * (self.tail_radius - self.base_radius)


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_shape_family(base_shape_class, length_range, radius_range, 
                          n_lengths=5, n_radii=5, output_dir="bor_family",
                          **mesh_kwargs):
    """
    Generate a family of shapes by varying parameters.
    
    Parameters
    ----------
    base_shape_class : class
        Shape class to instantiate
    length_range : tuple
        (min_length, max_length)
    radius_range : tuple
        (min_radius, max_radius)
    n_lengths : int
        Number of length values
    n_radii : int
        Number of radius values
    output_dir : str
        Directory for output files
    **mesh_kwargs
        Additional arguments for mesh generation
    
    Returns
    -------
    results : list of dict
        Mesh statistics for each configuration
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    lengths = np.linspace(*length_range, n_lengths)
    radii = np.linspace(*radius_range, n_radii)
    
    results = []
    
    print("\n" + "="*70)
    print(f"GENERATING FAMILY OF {base_shape_class.__name__} SHAPES")
    print("="*70)
    print(f"  Lengths: {n_lengths} values from {length_range[0]} to {length_range[1]}")
    print(f"  Radii: {n_radii} values from {radius_range[0]} to {radius_range[1]}")
    print(f"  Total: {n_lengths * n_radii} configurations")
    
    count = 0
    for i, L in enumerate(lengths):
        for j, R in enumerate(radii):
            count += 1
            
            # Create shape
            shape = base_shape_class(L, R)
            
            # Generate mesh
            print(f"\n[{count}/{n_lengths*n_radii}] {shape.name}: L={L:.3f}, R={R:.3f}")
            
            vertices, triangles, stats = shape.generate_mesh(**mesh_kwargs)
            
            # Save mesh
            filename = os.path.join(
                output_dir, 
                f"{shape.name.replace(' ', '_').lower()}_L{L:.3f}_R{R:.3f}.tri"
            )
            write_tri_file(filename, vertices, triangles)
            
            # Store results
            results.append({
                'shape': shape.name,
                'length': L,
                'radius': R,
                'fineness_ratio': shape.fineness_ratio(),
                'filename': filename,
                **stats
            })
    
    # Write summary file
    summary_file = os.path.join(output_dir, "family_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"# Family of {base_shape_class.__name__} Shapes\n")
        f.write(f"# Generated {len(results)} configurations\n\n")
        f.write("Length, Radius, Fineness, Vertices, Triangles, Mean_Skewness, Watertight\n")
        for r in results:
            f.write(f"{r['length']:.4f}, {r['radius']:.4f}, {r['fineness_ratio']:.4f}, "
                   f"{r['n_vertices']}, {r['n_triangles']}, "
                   f"{r['mean_skewness']:.4f}, {r['is_watertight']}\n")
    
    print(f"\n✓ Family generation complete!")
    print(f"  Generated {len(results)} meshes in {output_dir}/")
    print(f"  Summary: {summary_file}")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate body of revolution meshes")
    parser.add_argument('--shape', type=str, default='cone',
                       choices=['cone', 'ogive', 'haack', 'biconic', 'blunt'],
                       help='Shape to generate')
    parser.add_argument('--length', type=float, default=10.0,
                       help='Length')
    parser.add_argument('--radius', type=float, default=1.0,
                       help='Base radius')
    parser.add_argument('--n-axial', type=int, default=100,
                       help='Axial resolution')
    parser.add_argument('--n-circ', type=int, default=60,
                       help='Circumferential resolution')
    parser.add_argument('--family', action='store_true',
                       help='Generate parameter family')
    
    args = parser.parse_args()
    
    if args.family:
        # Generate family
        print("Generating shape family...")
        
        shape_map = {
            'cone': Cone,
            'ogive': TangentOgive,
            'haack': VonKarmanNose,
            'biconic': Biconic,
            'blunt': BluntCone
        }
        
        results = generate_shape_family(
            shape_map[args.shape],
            length_range=(5.0, 15.0),
            radius_range=(0.5, 2.0),
            n_lengths=3,
            n_radii=3,
            n_axial=args.n_axial,
            n_circumferential=args.n_circ,
            improve_quality=True
        )
        
    else:
        # Generate single shape
        print(f"Generating single {args.shape}...")
        
        if args.shape == 'cone':
            shape = Cone(args.length, args.radius)
        elif args.shape == 'ogive':
            shape = TangentOgive(args.length, args.radius)
        elif args.shape == 'haack':
            shape = VonKarmanNose(args.length, args.radius)
        elif args.shape == 'biconic':
            shape = Biconic(args.length, args.radius)
        elif args.shape == 'blunt':
            shape = BluntCone(args.length, args.radius, nose_radius=args.radius*0.3)
        
        vertices, triangles, stats = shape.generate_mesh(
            n_axial=args.n_axial,
            n_circumferential=args.n_circ,
            improve_quality=True
        )
        
        filename = f"{args.shape}_L{args.length}_R{args.radius}.tri"
        write_tri_file(filename, vertices, triangles)
        
        print(f"\n✓ Generated: {filename}")
        print(f"  Fineness ratio: {shape.fineness_ratio():.2f}")