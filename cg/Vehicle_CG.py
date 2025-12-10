import numpy as np
import vedo

# ============================================================================
# CONFIGURATION PARAMETERS - Modify these for your design
# ============================================================================

# Vehicle STL file
STL_FILE = 'FinalVehicle-ogive.stl'

# Shell properties
SHELL_THICKNESS = 0.005  # 5mm in meters (default thickness for main body)
SHELL_MATERIAL_DENSITY = 8840  # kg/m³ (Aluminum: 2700, Steel: 7850, Titanium: 4500)

# ============================================================================
# NOSE SECTION - Variable Density and Thickness (for first portion of vehicle)
# ============================================================================
USE_VARIABLE_NOSE_DENSITY = True  # *** CHANGE THIS: Set to False for uniform density ***
NOSE_LENGTH = 0.08                # *** CHANGE THIS: Length of nose section in meters (8cm default) ***
NOSE_DENSITY = 19400             # *** CHANGE THIS: Density for nose section in kg/m³ (Steel: 7850, Tungsten: 19300, Titanium: 4500) ***
NOSE_THICKNESS = 0.005            # *** CHANGE THIS: Thickness of nose section in meters (10mm default - thicker for heat) ***

# ============================================================================
# AFT SECTION - Variable Thickness (for rear portion of vehicle)
# ============================================================================
USE_VARIABLE_AFT_THICKNESS = True  # *** CHANGE THIS: Set to False for uniform thickness ***
AFT_LENGTH = 0.350                  # *** CHANGE THIS: Length of aft section in meters (20cm from tail) ***
AFT_THICKNESS = 0.003              # *** CHANGE THIS: Thickness of aft section in meters (3mm - thinner/lighter) ***

# ============================================================================
# *** MODIFY THESE PARAMETERS TO ADJUST YOUR DESIGN ***
# ============================================================================

# ----------------------------------------------------------------------------
# WATER RESERVOIR - Modeled as POINT MASS (no geometry, just mass and position)
# ----------------------------------------------------------------------------
WATER_MASS = 7.0    
WATER_DENSITY = 1000        # *** CHANGE THIS: Mass in kilograms (kg) ***
WATER_X_FRACTION = 0.4    # *** CHANGE THIS: Position as fraction of vehicle length (0.0=nose, 1.0=tail) ***
WATER_Y_POS = 0.0          # Lateral position (0 = centerline, +right, -left)
WATER_Z_POS = 0.0         # Vertical position (0 = centerline, +up, -down)

# ----------------------------------------------------------------------------
# PAYLOAD BAY - Modeled as CYLINDER aligned with vehicle axis
# ----------------------------------------------------------------------------
PAYLOAD_RADIUS = 0.032      # *** CHANGE THIS: Cylinder radius in meters ***
PAYLOAD_LENGTH = 0.161      # *** CHANGE THIS: Cylinder length in meters ***
PAYLOAD_DENSITY = 17376.694     # *** CHANGE THIS: Material density in kg/m³ ***
PAYLOAD_X_FRACTION = 0.65  # *** CHANGE THIS: Position as fraction of vehicle length (0.0=nose, 1.0=tail) ***
PAYLOAD_Y_POS = 0.0        # Lateral position (0 = centerline, +right, -left)
PAYLOAD_Z_POS = 0.0        # Vertical position (0 = centerline, +up, -down)
PAYLOAD_AXIS = 'x'         # Cylinder axis: 'x'=along vehicle, 'y'=spanwise, 'z'=vertical

# ============================================================================
# Visualization settings
SHOW_VEHICLE_TRANSPARENT = True
VEHICLE_ALPHA = 0.3
SHOW_CG_MARKER = True

# ============================================================================


class Waverider:
    def __init__(self, stl_fname, shell_thickness=SHELL_THICKNESS, 
                 shell_density=SHELL_MATERIAL_DENSITY):
        """
        Initialize the Waverider vehicle with shell properties
        
        Parameters:
            stl_fname: path to the vehicle STL file
            shell_thickness: thickness of the shell in meters
            shell_density: material density in kg/m³
        """
        self.mesh = vedo.Mesh(stl_fname)
        self.stl_fname = stl_fname
        self.shell_thickness = shell_thickness
        self.shell_density = shell_density
        
        # List of additional masses (ellipsoids, spheres, point masses)
        self.additional_masses = []
        
        # Mass properties (calculated by get_cg())
        self.CG = None  # 3D center of gravity [x, y, z]
        self.mass = None  # total mass in kg
        self.shell_mass = None  # mass of just the shell
        
    def add_cylinder_mass(self, name, radius, length, density, x_pos, y_pos=0, z_pos=0, axis='x'):
        """
        Add a cylindrical mass component (water tank, payload bay, fuel tank)
        Cylinder is aligned with the specified vehicle axis
        
        Parameters:
            name: identifier string for this component
            radius: cylinder radius in meters
            length: cylinder length in meters
            density: material density in kg/m³
            x_pos: x-position of cylinder center (meters from nose)
            y_pos: y-position of cylinder center (meters, +y = right wing)
            z_pos: z-position of cylinder center (meters, +z = up)
            axis: 'x' (along vehicle length), 'y' (spanwise), or 'z' (vertical)
        
        Note: For axis='x', the cylinder extends along the vehicle's longitudinal axis
              The semi-axes are arranged as: [length/2, radius, radius]
        """
        # A cylinder is an ellipsoid with two equal semi-axes
        if axis == 'x':
            # Cylinder along x-axis (vehicle longitudinal)
            semi_axes = [length/2, radius, radius]
        elif axis == 'y':
            # Cylinder along y-axis (spanwise)
            semi_axes = [radius, length/2, radius]
        elif axis == 'z':
            # Cylinder along z-axis (vertical)
            semi_axes = [radius, radius, length/2]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        
        # Calculate volume and mass
        volume = np.pi * radius**2 * length
        mass = volume * density
        
        component = {
            "name": name,
            "type": "cylinder",
            "radius": radius,
            "length": length,
            "axis": axis,
            "semi_axes": semi_axes,  # Store for ellipsoid calculations
            "density": density,
            "mass": mass,
            "volume": volume,
            "pos": np.array([x_pos, y_pos, z_pos]),
            "visualization_mesh": None
        }
        
        self.additional_masses.append(component)
        
        print(f"Added {name}:")
        print(f"  Type: Cylinder (axis-aligned: {axis})")
        print(f"  Radius: {radius:.3f} m, Length: {length:.3f} m")
        print(f"  Volume: {volume*1000:.2f} liters")
        print(f"  Mass: {mass:.2f} kg")
        print(f"  Position: ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}) m")
        
        return component
    
    def add_ellipsoid_mass(self, name, semi_axes, density, x_pos, y_pos=0, z_pos=0):
        """
        Add an ellipsoidal mass component (water reservoir, payload, fuel tank)
        
        Parameters:
            name: identifier string for this component
            semi_axes: [a, b, c] - semi-axis lengths in meters
                       a = along vehicle x-axis (longitudinal)
                       b = along vehicle y-axis (lateral)
                       c = along vehicle z-axis (vertical)
            density: material density in kg/m³
            x_pos: x-position of ellipsoid center (meters from nose)
            y_pos: y-position of ellipsoid center (meters, +y = right wing)
            z_pos: z-position of ellipsoid center (meters, +z = up)
        """
        # Calculate volume and mass
        a, b, c = semi_axes
        volume = (4/3) * np.pi * a * b * c
        mass = volume * density
        
        component = {
            "name": name,
            "type": "ellipsoid",
            "semi_axes": semi_axes,
            "density": density,
            "mass": mass,
            "volume": volume,
            "pos": np.array([x_pos, y_pos, z_pos]),
            "visualization_mesh": None  # Will be created when graphing
        }
        
        self.additional_masses.append(component)
        
        print(f"Added {name}:")
        print(f"  Type: Ellipsoid")
        print(f"  Semi-axes (a,b,c): {semi_axes} m")
        print(f"  Volume: {volume*1000:.2f} liters")
        print(f"  Mass: {mass:.2f} kg")
        print(f"  Position: ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}) m")
        
        return component
    
    def add_sphere_mass(self, name, radius, density, x_pos, y_pos=0, z_pos=0):
        """
        Add a spherical mass component (simplified model)
        
        Parameters:
            name: identifier string for this component
            radius: sphere radius in meters
            density: material density in kg/m³
            x_pos, y_pos, z_pos: position of sphere center
        """
        # A sphere is just an ellipsoid with equal semi-axes
        semi_axes = [radius, radius, radius]
        return self.add_ellipsoid_mass(name, semi_axes, density, x_pos, y_pos, z_pos)
    
    def add_point_mass(self, name, mass, x_pos, y_pos=0, z_pos=0):
        """
        Add a point mass (for components where geometry doesn't matter)
        
        Parameters:
            name: identifier string
            mass: mass in kg
            x_pos, y_pos, z_pos: position coordinates in meters
        """
        component = {
            "name": name,
            "type": "point",
            "mass": mass,
            "pos": np.array([x_pos, y_pos, z_pos]),
            "visualization_mesh": None
        }
        
        self.additional_masses.append(component)
        
        print(f"Added {name}:")
        print(f"  Type: Point mass")
        print(f"  Mass: {mass:.2f} kg")
        print(f"  Position: ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}) m")
        
        return component
    
    def inspect_vehicle_geometry(self):
        """
        Print vehicle dimensions and suggest component positions
        """
        bounds = self.mesh.bounds()
        
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        length = x_max - x_min
        width = y_max - y_min
        height = z_max - z_min
        
        print("\n" + "="*70)
        print("VEHICLE GEOMETRY INSPECTION")
        print("="*70)
        print(f"\nBounding Box:")
        print(f"  X: [{x_min:.4f}, {x_max:.4f}] m  (Length: {length:.4f} m)")
        print(f"  Y: [{y_min:.4f}, {y_max:.4f}] m  (Width: {width:.4f} m)")
        print(f"  Z: [{z_min:.4f}, {z_max:.4f}] m  (Height: {height:.4f} m)")
        
        print(f"\nSuggested Component Positions (centered on Y and Z):")
        print(f"  Front section (0-25%): X = {x_min + 0.125*length:.4f} m")
        print(f"  Forward-mid (25-40%): X = {x_min + 0.325*length:.4f} m")
        print(f"  Mid section (40-60%): X = {x_min + 0.5*length:.4f} m")
        print(f"  Aft-mid (60-75%): X = {x_min + 0.675*length:.4f} m")
        print(f"  Rear section (75-100%): X = {x_min + 0.875*length:.4f} m")
        
        print(f"\nMaximum component dimensions (to fit inside):")
        print(f"  Length (x): ~{length*0.8:.4f} m (80% of vehicle length)")
        print(f"  Width (y): ~{width*0.6:.4f} m (60% of vehicle width)")
        print(f"  Height (z): ~{height*0.6:.4f} m (60% of vehicle height)")
        print("="*70 + "\n")
        
        return bounds
    
    def check_interference(self, component_index=-1):
        """
        Check if a component fits inside the vehicle shell and doesn't overlap others
        
        Parameters:
            component_index: index of component to check (-1 for most recent)
        
        Returns:
            dict with 'fits_in_shell' and 'overlaps' boolean flags
        """
        if len(self.additional_masses) == 0:
            print("No components to check!")
            return None
        
        # Handle negative indexing
        if component_index < 0:
            component_index = len(self.additional_masses) + component_index
        
        component = self.additional_masses[component_index]
        
        # Only check ellipsoids, cylinders, and spheres (point masses have no volume)
        if component["type"] == "point":
            return {"fits_in_shell": True, "overlaps": False}
        
        # Check if component is inside shell (simplified check using bounding box)
        pos = component["pos"]
        semi_axes = component["semi_axes"]
        
        # Get vehicle bounding box
        vehicle_bounds = self.mesh.bounds()  # [xmin, xmax, ymin, ymax, zmin, zmax]
        
        # Check if ellipsoid bounding box is inside vehicle
        component_min = pos - np.array(semi_axes)
        component_max = pos + np.array(semi_axes)
        
        fits_x = (component_min[0] >= vehicle_bounds[0] and component_max[0] <= vehicle_bounds[1])
        fits_y = (component_min[1] >= vehicle_bounds[2] and component_max[1] <= vehicle_bounds[3])
        fits_z = (component_min[2] >= vehicle_bounds[4] and component_max[2] <= vehicle_bounds[5])
        
        fits_in_shell = fits_x and fits_y and fits_z
        
        # Check for overlaps with other components (simplified using bounding spheres)
        overlaps = False
        for i, other in enumerate(self.additional_masses):
            # Skip self-comparison and point masses
            if i == component_index or other["type"] == "point":
                continue
            
            # Distance between centers
            distance = np.linalg.norm(component["pos"] - other["pos"])
            
            # Maximum radii (conservative check)
            r1 = np.max(component["semi_axes"])
            r2 = np.max(other["semi_axes"])
            
            if distance < (r1 + r2):
                overlaps = True
                print(f"WARNING: {component['name']} may overlap with {other['name']}")
        
        if not fits_in_shell:
            print(f"WARNING: {component['name']} may extend outside vehicle shell!")
            print(f"  Vehicle bounds: X[{vehicle_bounds[0]:.3f}, {vehicle_bounds[1]:.3f}], "
                  f"Y[{vehicle_bounds[2]:.3f}, {vehicle_bounds[3]:.3f}], "
                  f"Z[{vehicle_bounds[4]:.3f}, {vehicle_bounds[5]:.3f}]")
            print(f"  Component bounds: X[{component_min[0]:.3f}, {component_max[0]:.3f}], "
                  f"Y[{component_min[1]:.3f}, {component_max[1]:.3f}], "
                  f"Z[{component_min[2]:.3f}, {component_max[2]:.3f}]")
        
        return {"fits_in_shell": fits_in_shell, "overlaps": overlaps}
    
    def graph_vehicle(self, show_cg=SHOW_CG_MARKER):
        """
        Visualize the vehicle shell and all components in 3D
        
        Parameters:
            show_cg: if True, display the center of gravity marker
        """
        meshes = []
        
        # Create vehicle shell mesh (semi-transparent)
        vehicle_mesh = self.mesh.clone()
        if SHOW_VEHICLE_TRANSPARENT:
            vehicle_mesh.alpha(VEHICLE_ALPHA).color('lightblue')
        else:
            vehicle_mesh.color('lightblue')
        meshes.append(vehicle_mesh)
        
        # Create meshes for all components
        for comp in self.additional_masses:
            if comp["type"] == "cylinder":
                # Create actual CYLINDER visualization
                radius = comp["radius"]
                length = comp["length"]
                pos = comp["pos"]
                axis = comp["axis"]
                
                # Determine cylinder orientation based on axis
                if axis == 'x':
                    # Cylinder along x-axis
                    cylinder = vedo.Cylinder(
                        pos=pos,
                        r=radius,
                        height=length,
                        axis=(1, 0, 0),  # Along x
                        res=24
                    ).color('red').alpha(0.8)
                elif axis == 'y':
                    # Cylinder along y-axis
                    cylinder = vedo.Cylinder(
                        pos=pos,
                        r=radius,
                        height=length,
                        axis=(0, 1, 0),  # Along y
                        res=24
                    ).color('red').alpha(0.8)
                elif axis == 'z':
                    # Cylinder along z-axis
                    cylinder = vedo.Cylinder(
                        pos=pos,
                        r=radius,
                        height=length,
                        axis=(0, 0, 1),  # Along z
                        res=24
                    ).color('red').alpha(0.8)
                
                meshes.append(cylinder)
                
                # Add label with dimensions
                label_text = f"{comp['name']}\n(R={radius:.2f}m, L={length:.2f}m)"
                label = vedo.Text3D(
                    label_text,
                    pos=pos + np.array([0, 0, radius + 0.05]),
                    s=0.03,
                    c='black'
                )
                meshes.append(label)
                
            elif comp["type"] == "ellipsoid":
                # Create ellipsoid visualization
                a, b, c = comp["semi_axes"]
                pos = comp["pos"]
                
                # Create ellipsoid using Vedo
                ellipsoid = vedo.Ellipsoid(
                    pos=pos,
                    axis1=[2*a, 0, 0],
                    axis2=[0, 2*b, 0],
                    axis3=[0, 0, 2*c]
                ).color('red').alpha(0.8)
                
                meshes.append(ellipsoid)
                
                # Add label
                label_text = comp["name"]
                label = vedo.Text3D(
                    label_text,
                    pos=pos + np.array([0, 0, np.max(comp["semi_axes"]) + 0.05]),
                    s=0.03,
                    c='black'
                )
                meshes.append(label)
                
            elif comp["type"] == "point":
                # Show point mass as small sphere
                point_marker = vedo.Sphere(pos=comp["pos"], r=0.02).color('yellow')
                meshes.append(point_marker)
                
                label = vedo.Text3D(
                    comp["name"],
                    pos=comp["pos"] + np.array([0, 0, 0.05]),
                    s=0.02,
                    c='black'
                )
                meshes.append(label)
        
        # Show CG marker if calculated
        if show_cg and self.CG is not None:
            cg_marker = vedo.Sphere(pos=self.CG, r=0.03).color('green')
            meshes.append(cg_marker)
            
            cg_label = vedo.Text3D(
                f"CG",
                pos=self.CG + np.array([0, 0, 0.08]),
                s=0.04,
                c='green'
            )
            meshes.append(cg_label)
            
            # Add simple arrows at CG to show axes
            arrow_length = 0.15
            # X-axis arrow (red)
            arrow_x = vedo.Arrow(
                start_pt=self.CG,
                end_pt=self.CG + np.array([arrow_length, 0, 0]),
                c='red'
            )
            meshes.append(arrow_x)
            
            # Y-axis arrow (green)
            arrow_y = vedo.Arrow(
                start_pt=self.CG,
                end_pt=self.CG + np.array([0, arrow_length, 0]),
                c='green'
            )
            meshes.append(arrow_y)
            
            # Z-axis arrow (blue)
            arrow_z = vedo.Arrow(
                start_pt=self.CG,
                end_pt=self.CG + np.array([0, 0, arrow_length]),
                c='blue'
            )
            meshes.append(arrow_z)
        
        # Create plotter with axes
        plt = vedo.Plotter(title="Waverider Vehicle - CG Analysis", axes=1)
        
        print(f'\nDisplaying vehicle visualization...')
        plt.show(meshes, interactive=True)
    
    def get_shell_cg(self):
        """
        Calculate the CG and mass of the vehicle shell alone
        Supports variable density and thickness for nose and aft sections
        
        Returns:
            tuple: (CG_3D, mass) where CG_3D is [x, y, z] in meters
        """
        mesh = self.mesh
        mesh.triangulate()  # Ensure all faces are triangles
        
        points = mesh.points
        face_indices = mesh.cells
        faces = points[face_indices]
        
        # Extract vertices for each triangle
        v0 = faces[:, 0]
        v1 = faces[:, 1]
        v2 = faces[:, 2]
        
        centroids = []
        masses = []
        
        # Get vehicle bounds for nose and aft section calculation
        x_min = self.mesh.bounds()[0]  # Front of vehicle
        x_max = self.mesh.bounds()[1]  # Rear of vehicle
        
        # Variable density and thickness function
        def get_area_density_at_position(centroid):
            """
            Return area density based on x-position
            Supports variable nose density/thickness and aft thickness
            """
            x_pos = centroid[0]
            
            # Priority 1: Check if in NOSE section (front of vehicle)
            if USE_VARIABLE_NOSE_DENSITY and x_pos < (x_min + NOSE_LENGTH):
                # Nose section with special material and thickness
                return NOSE_THICKNESS * NOSE_DENSITY
            
            # Priority 2: Check if in AFT section (rear of vehicle)
            elif USE_VARIABLE_AFT_THICKNESS and x_pos > (x_max - AFT_LENGTH):
                # Aft section with different thickness (same material as body)
                return AFT_THICKNESS * self.shell_density
            
            # Default: Main body section
            else:
                # Rest of vehicle uses standard density and thickness
                return self.shell_thickness * self.shell_density
        
        # For each triangle
        for i in range(len(v0)):
            A = v0[i]
            B = v1[i]
            C = v2[i]
            
            # Centroid of triangle
            centroid = (A + B + C) / 3.0
            centroids.append(centroid)
            
            # Area of triangle (half magnitude of cross product)
            AB = B - A
            AC = C - A
            area = 0.5 * np.linalg.norm(np.cross(AB, AC))
            
            # Mass of this triangle (using position-dependent density and thickness)
            area_density = get_area_density_at_position(centroid)
            mass = area * area_density
            masses.append(mass)
        
        centroids = np.array(centroids)
        masses = np.array(masses)
        
        # Total shell mass
        total_mass = np.sum(masses)
        
        # Shell center of gravity (weighted average of centroids)
        CG = np.zeros(3)
        for i in range(len(centroids)):
            CG += masses[i] * centroids[i]
        CG /= total_mass
        
        return CG, total_mass
    
    def get_cg(self):
        """
        Calculate the full 3D center of gravity including shell and all components
        
        Returns:
            np.array: CG position [x, y, z] in meters
        """
        # Calculate shell CG and mass
        shell_CG, shell_mass = self.get_shell_cg()
        
        self.shell_mass = shell_mass
        self.mass = shell_mass
        self.CG = shell_CG.copy()
        
        print(f"\nShell Properties:")
        print(f"  Main Body Thickness: {self.shell_thickness*1000:.1f} mm")
        print(f"  Main Body Density: {self.shell_density} kg/m³")
        
        # Show nose section info if variable density is enabled
        if USE_VARIABLE_NOSE_DENSITY:
            print(f"\n  NOSE SECTION (Variable Density & Thickness):")
            print(f"    Length: {NOSE_LENGTH*100:.1f} cm ({NOSE_LENGTH:.3f} m)")
            print(f"    Thickness: {NOSE_THICKNESS*1000:.1f} mm")
            print(f"    Density: {NOSE_DENSITY} kg/m³")
            if NOSE_DENSITY == 7850:
                print(f"    Material: Steel")
            elif NOSE_DENSITY == 19300:
                print(f"    Material: Tungsten")
            elif NOSE_DENSITY == 4500:
                print(f"    Material: Titanium")
        
        # Show aft section info if variable thickness is enabled
        if USE_VARIABLE_AFT_THICKNESS:
            print(f"\n  AFT SECTION (Variable Thickness):")
            print(f"    Length: {AFT_LENGTH*100:.1f} cm ({AFT_LENGTH:.3f} m from tail)")
            print(f"    Thickness: {AFT_THICKNESS*1000:.1f} mm")
            print(f"    Density: {self.shell_density} kg/m³ (same as body)")
        
        print(f"\n  Total Shell Mass: {shell_mass:.2f} kg")
        print(f"  Shell CG: ({shell_CG[0]:.4f}, {shell_CG[1]:.4f}, {shell_CG[2]:.4f}) m")
        
        # Add each component to the CG calculation
        if len(self.additional_masses) == 0:
            print(f'\nWARNING: No additional masses added! Consider adding payload, fuel, etc.')
        else:
            print(f"\nAdditional Components:")
            
        for comp in self.additional_masses:
            comp_mass = comp["mass"]
            comp_pos = comp["pos"]
            
            # Update CG using weighted average
            # CG_new = (CG_old * mass_old + pos_component * mass_component) / (mass_old + mass_component)
            self.CG = (self.CG * self.mass + comp_pos * comp_mass) / (self.mass + comp_mass)
            self.mass += comp_mass
            
            print(f"  {comp['name']}: {comp_mass:.2f} kg at ({comp_pos[0]:.3f}, {comp_pos[1]:.3f}, {comp_pos[2]:.3f}) m")
        
        print(f"\nTotal Vehicle Properties:")
        print(f"  Total Mass: {self.mass:.2f} kg")
        print(f"  Final CG: ({self.CG[0]:.4f}, {self.CG[1]:.4f}, {self.CG[2]:.4f}) m")
        print(f"  Shell percentage: {(shell_mass/self.mass)*100:.1f}%")
        
        if len(self.additional_masses) > 0:
            component_mass = self.mass - shell_mass
            print(f"  Component percentage: {(component_mass/self.mass)*100:.1f}%")
        
        return self.CG
    
    def parametric_study(self, parameter_name, values, component_name):
        """
        Perform a parametric study varying one parameter
        
        Parameters:
            parameter_name: 'x_pos', 'y_pos', 'z_pos', 'mass', or 'density'
            values: list of values to sweep
            component_name: name of component to vary
        
        Returns:
            results: dict with 'values', 'cg_x', 'cg_y', 'cg_z', 'total_mass'
        """
        results = {
            'values': [],
            'cg_x': [],
            'cg_y': [],
            'cg_z': [],
            'total_mass': []
        }
        
        # Find the component to vary
        comp_index = None
        for i, comp in enumerate(self.additional_masses):
            if comp['name'] == component_name:
                comp_index = i
                break
        
        if comp_index is None:
            print(f"Error: Component '{component_name}' not found!")
            return None
        
        # Store original value
        original_comp = self.additional_masses[comp_index].copy()
        
        print(f"\nParametric Study: {parameter_name} for {component_name}")
        print("-" * 60)
        
        for value in values:
            # Modify the parameter
            if parameter_name == 'x_pos':
                self.additional_masses[comp_index]['pos'][0] = value
            elif parameter_name == 'y_pos':
                self.additional_masses[comp_index]['pos'][1] = value
            elif parameter_name == 'z_pos':
                self.additional_masses[comp_index]['pos'][2] = value
            elif parameter_name == 'density':
                # Recalculate mass with new density
                old_density = self.additional_masses[comp_index]['density']
                self.additional_masses[comp_index]['density'] = value
                self.additional_masses[comp_index]['mass'] *= (value / old_density)
            elif parameter_name == 'mass':
                self.additional_masses[comp_index]['mass'] = value
            
            # Recalculate CG
            cg = self.get_cg()
            
            results['values'].append(value)
            results['cg_x'].append(cg[0])
            results['cg_y'].append(cg[1])
            results['cg_z'].append(cg[2])
            results['total_mass'].append(self.mass)
        
        # Restore original component
        self.additional_masses[comp_index] = original_comp
        
        return results
    
    def export_mass_properties(self, filename='mass_properties.txt'):
        """
        Export mass properties to a text file
        
        Parameters:
            filename: output file name
        """
        if self.CG is None:
            print("Error: CG not calculated yet! Run get_cg() first.")
            return
        
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("WAVERIDER VEHICLE - MASS PROPERTIES REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Vehicle STL: {self.stl_fname}\n")
            f.write(f"Shell Thickness: {self.shell_thickness*1000:.1f} mm\n")
            f.write(f"Shell Material Density: {self.shell_density} kg/m³\n")
            
            if USE_VARIABLE_NOSE_DENSITY:
                f.write(f"\nNose Section (Variable Density & Thickness):\n")
                f.write(f"  Length: {NOSE_LENGTH*100:.1f} cm\n")
                f.write(f"  Thickness: {NOSE_THICKNESS*1000:.1f} mm\n")
                f.write(f"  Density: {NOSE_DENSITY} kg/m³\n")
            
            if USE_VARIABLE_AFT_THICKNESS:
                f.write(f"\nAft Section (Variable Thickness):\n")
                f.write(f"  Length: {AFT_LENGTH*100:.1f} cm\n")
                f.write(f"  Thickness: {AFT_THICKNESS*1000:.1f} mm\n")
                f.write(f"  Density: {self.shell_density} kg/m³\n")
            
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("SHELL PROPERTIES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mass: {self.shell_mass:.2f} kg\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("ADDITIONAL COMPONENTS\n")
            f.write("-" * 70 + "\n")
            
            for comp in self.additional_masses:
                f.write(f"\n{comp['name']}:\n")
                f.write(f"  Type: {comp['type']}\n")
                f.write(f"  Mass: {comp['mass']:.2f} kg\n")
                f.write(f"  Position: ({comp['pos'][0]:.4f}, {comp['pos'][1]:.4f}, {comp['pos'][2]:.4f}) m\n")
                
                if comp['type'] == 'cylinder':
                    f.write(f"  Radius: {comp['radius']:.3f} m\n")
                    f.write(f"  Length: {comp['length']:.3f} m\n")
                    f.write(f"  Axis: {comp['axis']}\n")
                    f.write(f"  Volume: {comp['volume']*1000:.2f} liters\n")
                    f.write(f"  Density: {comp['density']} kg/m³\n")
                elif comp['type'] == 'ellipsoid':
                    f.write(f"  Semi-axes: ({comp['semi_axes'][0]:.3f}, {comp['semi_axes'][1]:.3f}, {comp['semi_axes'][2]:.3f}) m\n")
                    f.write(f"  Volume: {comp['volume']*1000:.2f} liters\n")
                    f.write(f"  Density: {comp['density']} kg/m³\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("TOTAL VEHICLE PROPERTIES\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Mass: {self.mass:.2f} kg\n")
            f.write(f"Center of Gravity (X, Y, Z): ({self.CG[0]:.4f}, {self.CG[1]:.4f}, {self.CG[2]:.4f}) m\n")
            f.write(f"Shell Mass Percentage: {(self.shell_mass/self.mass)*100:.1f}%\n")
            
            if len(self.additional_masses) > 0:
                component_mass = self.mass - self.shell_mass
                f.write(f"Component Mass Percentage: {(component_mass/self.mass)*100:.1f}%\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"\nMass properties exported to: {filename}")


def example_basic_usage():
    """
    Basic example: Calculate CG with point mass water reservoir and cylindrical payload
    Water is positioned in front of (before) the payload
    """
    print("\n" + "="*70)
    print("EXAMPLE: CG Calculation - Point Mass Water + Cylindrical Payload")
    print("="*70)
    
    # Create vehicle
    vehicle = Waverider(STL_FILE, SHELL_THICKNESS, SHELL_MATERIAL_DENSITY)
    
    # Inspect vehicle geometry and get dimensions
    bounds = vehicle.inspect_vehicle_geometry()
    
    # Extract vehicle dimensions
    x_min, x_max = bounds[0], bounds[1]
    vehicle_length = x_max - x_min
    
    # Calculate actual positions based on fractions of vehicle length
    water_x_pos = x_min + (WATER_X_FRACTION * vehicle_length)
    payload_x_pos = x_min + (PAYLOAD_X_FRACTION * vehicle_length)
    
    # Verify water is in front of payload
    if water_x_pos >= payload_x_pos:
        print("\n" + "!"*70)
        print("WARNING: Water position should be LESS than payload position!")
        print(f"Current: Water at {WATER_X_FRACTION:.1%}, Payload at {PAYLOAD_X_FRACTION:.1%}")
        print("Suggestion: Set WATER_X_FRACTION < PAYLOAD_X_FRACTION")
        print("!"*70 + "\n")
    
    print(f"\n" + "="*70)
    print("COMPONENT POSITIONS")
    print("="*70)
    print(f"Water Reservoir (Point Mass):")
    print(f"  Position: X = {water_x_pos:.4f} m ({WATER_X_FRACTION*100:.1f}% along vehicle)")
    print(f"  Mass: {WATER_MASS:.2f} kg")
    print(f"\nPayload Bay (Cylinder):")
    print(f"  Position: X = {payload_x_pos:.4f} m ({PAYLOAD_X_FRACTION*100:.1f}% along vehicle)")
    print(f"  Radius: {PAYLOAD_RADIUS:.3f} m, Length: {PAYLOAD_LENGTH:.3f} m")
    print(f"  Density: {PAYLOAD_DENSITY} kg/m³")
    print("="*70 + "\n")
    
    # Add water reservoir as POINT MASS (in front of payload)
    vehicle.add_point_mass(
        name="Water Reservoir",
        mass=WATER_MASS,
        x_pos=water_x_pos,
        y_pos=WATER_Y_POS,
        z_pos=WATER_Z_POS
    )
    
    '''
    vehicle.add_cylinder_mass(
        name="Water Reservoir",
        radius=PAYLOAD_RADIUS,
        length=PAYLOAD_LENGTH,
        density=PAYLOAD_DENSITY,
        x_pos=water_x_pos,
        y_pos=WATER_Y_POS,
        z_pos=WATER_Z_POS,
        axis=PAYLOAD_AXIS
    )
    '''

    # Add payload as CYLINDER (behind water)
    vehicle.add_cylinder_mass(
        name="Payload Bay",
        radius=PAYLOAD_RADIUS,
        length=PAYLOAD_LENGTH,
        density=PAYLOAD_DENSITY,
        x_pos=payload_x_pos,
        y_pos=PAYLOAD_Y_POS,
        z_pos=PAYLOAD_Z_POS,
        axis=PAYLOAD_AXIS
    )
    
    # Check for interference
    print("Checking component fit:")
    vehicle.check_interference(0)  # water (first component)
    vehicle.check_interference(1)  # payload (second component)
    
    # Calculate CG
    cg = vehicle.get_cg()
    
    print("\n" + "="*70)
    print("REMINDER: TO MODIFY THE DESIGN")
    print("="*70)
    print("Edit these parameters at the TOP of the script:")
    print("")
    print("  NOSE SECTION (Lines 13-17):")
    print("    USE_VARIABLE_NOSE_DENSITY = True  # Enable/disable")
    print("    NOSE_LENGTH = 0.08        # Length (m)")
    print("    NOSE_DENSITY = 7850       # Density (kg/m³)")
    print("    NOSE_THICKNESS = 0.010    # Thickness (m)")
    print("")
    print("  AFT SECTION (Lines 21-24):")
    print("    USE_VARIABLE_AFT_THICKNESS = True  # Enable/disable")
    print("    AFT_LENGTH = 0.20         # Length (m)")
    print("    AFT_THICKNESS = 0.003     # Thickness (m)")
    print("")
    print("  WATER (Lines 31-35):")
    print("    WATER_MASS = 15.0           # Mass (kg)")
    print("    WATER_X_FRACTION = 0.30     # Position (0.0-1.0)")
    print("")
    print("  PAYLOAD (Lines 38-44):")
    print("    PAYLOAD_RADIUS = 0.12       # Radius (m)")
    print("    PAYLOAD_LENGTH = 0.50       # Length (m)")
    print("    PAYLOAD_DENSITY = 2500      # Density (kg/m³)")
    print("    PAYLOAD_X_FRACTION = 0.70   # Position (0.0-1.0)")
    print("="*70 + "\n")
    
    # Export results
    vehicle.export_mass_properties('mass_properties_basic.txt')
    
    # Visualize
    vehicle.graph_vehicle()
    
    return vehicle


def example_parametric_study():
    """
    Example: Perform parametric study on water reservoir position
    Shows how water position affects overall CG
    """
    print("\n" + "="*70)
    print("EXAMPLE: Parametric Study - Water Position Effect on CG")
    print("="*70)
    
    # Create vehicle
    vehicle = Waverider(STL_FILE, SHELL_THICKNESS, SHELL_MATERIAL_DENSITY)
    
    # Get vehicle dimensions
    bounds = vehicle.inspect_vehicle_geometry()
    x_min, x_max = bounds[0], bounds[1]
    vehicle_length = x_max - x_min
    
    # Calculate positions
    water_x_pos = x_min + (WATER_X_FRACTION * vehicle_length)
    payload_x_pos = x_min + (PAYLOAD_X_FRACTION * vehicle_length)
    
    # Add components
    vehicle.add_point_mass("Water Reservoir", WATER_MASS, 
                          water_x_pos, WATER_Y_POS, WATER_Z_POS)
    vehicle.add_cylinder_mass("Payload Bay", PAYLOAD_RADIUS, PAYLOAD_LENGTH,
                              PAYLOAD_DENSITY, payload_x_pos, PAYLOAD_Y_POS, PAYLOAD_Z_POS, PAYLOAD_AXIS)
    
    # Sweep water position from 15% to 45% of vehicle length
    x_fractions = np.linspace(0.15, 0.45, 7)
    x_positions = x_min + (x_fractions * vehicle_length)
    results = vehicle.parametric_study('x_pos', x_positions, 'Water Reservoir')
    
    print("\nParametric Study Results:")
    print(f"{'Water X-Pos (m)':<20} {'X-Fraction':<15} {'CG X (m)':<15} {'Total Mass (kg)':<15}")
    print("-" * 65)
    for i in range(len(results['values'])):
        x_frac = (results['values'][i] - x_min) / vehicle_length
        print(f"{results['values'][i]:<20.3f} {x_frac:<15.2%} {results['cg_x'][i]:<15.4f} {results['total_mass'][i]:<15.2f}")
    
    return vehicle, results




if __name__ == "__main__":
    # Run the basic example
    print("\n" + "="*70)
    print("Starting CG Analysis...")
    print("="*70)
    print("QUICK START GUIDE:")
    print("")
    print("To modify the design, edit these parameters at the TOP of this file:")
    print("")
    print("  Lines 13-17: NOSE SECTION (Variable Density & Thickness)")
    print("    USE_VARIABLE_NOSE_DENSITY = True  # Enable/disable")
    print("    NOSE_LENGTH = 0.08        # Length in meters (8cm)")
    print("    NOSE_DENSITY = 7850       # Density kg/m³ (Steel)")
    print("    NOSE_THICKNESS = 0.010    # Thickness in meters (10mm)")
    print("")
    print("  Lines 21-24: AFT SECTION (Variable Thickness)")
    print("    USE_VARIABLE_AFT_THICKNESS = True  # Enable/disable")
    print("    AFT_LENGTH = 0.20         # Length in meters (20cm)")
    print("    AFT_THICKNESS = 0.003     # Thickness in meters (3mm)")
    print("")
    print("  Lines 31-35: WATER RESERVOIR (Point Mass)")
    print("    WATER_MASS = 15.0          # Mass in kg")
    print("    WATER_X_FRACTION = 0.30    # Position (0.0=nose, 1.0=tail)")
    print("")
    print("  Lines 38-44: PAYLOAD BAY (Cylinder)")
    print("    PAYLOAD_RADIUS = 0.12      # Radius in meters")
    print("    PAYLOAD_LENGTH = 0.50      # Length in meters")
    print("    PAYLOAD_DENSITY = 2500     # Density in kg/m³")
    print("    PAYLOAD_X_FRACTION = 0.70  # Position (0.0=nose, 1.0=tail)")
    print("")
    print("="*70 + "\n")
    
    # Choose which example to run:
    vehicle = example_basic_usage()
    
    # Uncomment to run other examples:
    # vehicle, results = example_parametric_study()
    