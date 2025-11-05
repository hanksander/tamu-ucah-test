from waverider_generator.generator import waverider as wr
from waverider_generator.plotting_tools import Plot_Base_Plane,Plot_Leading_Edge
import matplotlib.pyplot as plt
from waverider_generator.cad_export import to_CAD

import cadquery as cq
import gmsh
import meshio
import tempfile
import numpy as np
import os
import trimesh
# import pymeshlab

def export_watertight_stl(waverider_cad, stl_path):
    obj_to_export = waverider_cad
    try:
        if hasattr(waverider_cad, "combineSolids"):
            obj_to_export = waverider_cad.combineSolids().clean()
            if hasattr(obj_to_export, "val"):
                obj_to_export = obj_to_export.val()
    except Exception:
        obj_to_export = waverider_cad

    cq.exporters.export(obj_to_export, stl_path)

    tm = trimesh.load_mesh(stl_path, force='mesh')
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tm.dump())

    # Cleanup steps
    for fn in (
        "remove_duplicate_faces",
        "remove_degenerate_faces",
        "remove_unreferenced_vertices",
        "merge_vertices",
    ):
        try:
            getattr(tm, fn)()
        except Exception:
            pass

    # Try to fill small holes if not watertight
    if not tm.is_watertight:
        try:
            tm.fill_holes()   # best-effort
        except Exception:
            pass
        for fn in ("remove_duplicate_faces", "remove_degenerate_faces", "remove_unreferenced_vertices"):
            try:
                getattr(tm, fn)()
            except Exception:
                pass

    tm.export(stl_path)
    return stl_path



#this version uses stls. fails earlier than the one below using steps so commented out, but should theoretically work better
# def generate_surface_mesh(waverider_cad, mesh_size=0.01, basename="waverider", output_dir="."):
#     """
#     Generate surface meshes (.msh for CBaero, .tri for Cart3D) from a CadQuery object.

#     Parameters
#     ----------
#     waverider_cad : cadquery.Workplane
#         The CadQuery object representing the surface geometry.
#     mesh_size : float, optional
#         Target element size for meshing (smaller = finer).
#     basename : str, optional
#         Base name for output files (e.g., 'waverider').
#     output_dir : str, optional
#         Directory to write outputs into.
#     """

#     os.makedirs(output_dir, exist_ok=True)
#     step_path = os.path.join(output_dir, f"{basename}.step")
#     stl_path = os.path.join(output_dir, f"{basename}_msh.stl")
#     msh_path  = os.path.join(output_dir, f"{basename}.msh")
#     tri_path  = os.path.join(output_dir, f"{basename}.tri")

#     # --- 1. Export CadQuery object to STEP ---

#     waverider_cad = waverider_cad.mirror("XY", union=True)  # Ensure both sides for meshing
#     waverider_cad = waverider_cad.combineSolids().clean()
#     # cq.exporters.export(waverider_cad, stl_path)
#     export_watertight_stl(waverider_cad, stl_path)
    
#     assy = cq.Assembly()
#     assy.add(waverider_cad)
#     assy.save(step_path)

#     # --- 2. Initialize gmsh and import geometry ---
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Terminal", 1)
#     gmsh.option.setNumber("General.Verbosity", 99)
#     gmsh.option.setNumber("General.AbortOnError", 1)
#     gmsh.model.add(basename)

#     mesh = trimesh.load_mesh(stl_path)
#     mesh.remove_duplicate_faces()
#     mesh.remove_unreferenced_vertices()
#     mesh.remove_degenerate_faces()
#     mesh.merge_vertices()
#     mesh.remove_infinite_values()
#     mesh.remove_duplicate_faces()
#     mesh.export(stl_path)

#     gmsh.merge(stl_path)
#     gmsh.model.mesh.classifySurfaces(10 * np.pi / 180., False,
#                                      3,
#                                      10 * np.pi / 180)
#     gmsh.model.mesh.createGeometry()

#     s = gmsh.model.getEntities(2)
#     l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
#     gmsh.model.geo.addVolume([l])
#     gmsh.model.geo.synchronize()

#     # these options will rebuild the mesh from the ground up treating the stl just as a geometry
#     gmsh.model.mesh.field.add("Constant", 1)
#     gmsh.model.mesh.field.setNumber(1, "VIn", mesh_size)
#     gmsh.model.mesh.field.setAsBackgroundMesh(1)


#     # --- 3. Set GMSH mesh options ---

#     gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.8*mesh_size)
#     gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.2*mesh_size)
#     gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
#     gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
#     gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-5)
#     gmsh.option.setNumber("Mesh.Algorithm", 5)  # Frontal-Delaunay


#     # --- 4. Generate mesh and write to .msh ---
    
#     #for regeneration
#     gmsh.model.mesh.generate(3)
    
#     #for refinement -- doesn't work well so commenting
#     #gmsh.model.mesh.refine()
    
#     gmsh.write(msh_path)
#     gmsh.finalize()


#     # --- 5. Convert Gmsh mesh to Cart3D .tri ---
#     mesh = meshio.read(msh_path)
#     tri_cells = mesh.get_cells_type("triangle")
#     points = mesh.points.copy()

#     # swap y and z and convert to meters from mm
#     points = points[:, [0, 2, 1]] * 1e-3

#     # swap node order to get outward normals
#     tri_cells = tri_cells[:, [0, 2, 1]]


#     # Cart3D expects 1-based indices and ASCII format
#     n_verts = len(points)
#     n_tris = len(tri_cells)

#     with open(tri_path, "w") as f:
#         f.write(f"{n_verts} {n_tris}\n")
#         for p in points:
#             f.write(f"    {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
#         for tri in tri_cells:
#             f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
#         for i in range(n_tris):
#             f.write("1\n")  # Uniform area weight

#     print(f"[✓] Mesh generation complete.")
#     print(f"    CBaero mesh: {msh_path}")
#     print(f"    Cart3D mesh: {tri_path}")

#     return msh_path, tri_path

def generate_surface_mesh(waverider_cad, mesh_size=0.01, basename="waverider", output_dir="."):
    """
    Generate surface meshes (.msh for CBaero, .tri for Cart3D) from a CadQuery object.

    Parameters
    ----------
    waverider_cad : cadquery.Workplane
        The CadQuery object representing the surface geometry.
    mesh_size : float, optional
        Target element size for meshing (smaller = finer).
    basename : str, optional
        Base name for output files (e.g., 'waverider').
    output_dir : str, optional
        Directory to write outputs into.
    """

    os.makedirs(output_dir, exist_ok=True)
    step_path = os.path.join(output_dir, f"{basename}.step")
    stl_path = os.path.join(output_dir, f"{basename}_msh.stl")
    msh_path  = os.path.join(output_dir, f"{basename}.msh")
    tri_path  = os.path.join(output_dir, f"{basename}.tri")

    # --- 1. Export CadQuery object to STEP ---

    # mirrored = waverider_cad.mirror("XY").translate((0, 0, 1e-6))
    # waverider_cad = waverider_cad.union(mirrored).combineSolids().clean()

    # waverider_cad = waverider_cad.combineSolids().clean()
    # cq.exporters.export(waverider_cad, stl_path)
    export_watertight_stl(waverider_cad, stl_path)
    
    assy = cq.Assembly()
    assy.add(waverider_cad)
    assy.save(step_path)

    # --- 2. Initialize gmsh and import geometry ---
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 99)
    gmsh.option.setNumber("General.AbortOnError", 1)
    gmsh.model.add(basename)

    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # Inside gmsh python api, before meshing:
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    gmsh.model.occ.fragment([(3, s[1]) for s in gmsh.model.getEntities(2)], [])
    gmsh.model.occ.synchronize()


    print("Number of surfaces:", len(gmsh.model.getEntities(2)))
    print("Number of solids:", len(gmsh.model.getEntities(3)))


    # --- 3. Set mesh parameters ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.8*mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.2*mesh_size)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)   # 5 --> delaunay frontal, 8 --> Delauny Quad

    # --- 4. Generate surface mesh (2D only) ---
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    gmsh.finalize()

    
    # --- 5. Convert Gmsh mesh to Cart3D .tri ---
    mesh = meshio.read(msh_path)
    tri_cells = mesh.get_cells_type("triangle")
    points = mesh.points.copy()

    # swap y and z and convert to meters from mm
    points = points[:, [0, 2, 1]] * 1e-3

    # swap node order to get outward normals
    tri_cells = tri_cells[:, [0, 2, 1]]


    # Cart3D expects 1-based indices and ASCII format
    n_verts = len(points)
    n_tris = len(tri_cells)

    with open(tri_path, "w") as f:
        f.write(f"{n_verts} {n_tris}\n")
        for p in points:
            f.write(f"    {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for tri in tri_cells:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        for i in range(n_tris):
            f.write("1\n")  # Uniform area weight

    print(f"[✓] Mesh generation complete.")
    print(f"    CBaero mesh: {msh_path}")
    print(f"    Cart3D mesh: {tri_path}")

    return msh_path, tri_path

if __name__ == "__main__":

    #DEFAULT PARAMETERS
    M_inf=5
    beta=15
    height=1.34
    width=3
    dp=[0.11,0.63,0,0.46]
    n_planes=20
    n_streamwise=10
    delta_streamwise=0.1


    #OUR WAVERIDER PARAMETERS

    # M_inf=7
    # beta=11.3
    # length = 1

    # if (length*np.tan(np.radians(beta)) <= 0.2):
    #     height= length*np.tan(np.radians(beta)) #h = L/tan(beta)
    # else:
    #     height=0.2
        
    # width=0.1
    # dp=[0.0,0.0,0.0,0.0]
    # n_planes=20
    # n_streamwise=10
    # delta_streamwise=0.1

    print("[*] Generating waverider geometry...")
    waverider=wr(M_inf=M_inf, 
                beta=beta,
                height=height,
                width=width,
                dp=dp,
                n_upper_surface=10000,
                n_shockwave=10000,
                n_planes=n_planes,
                n_streamwise=n_streamwise,
                delta_streamise=delta_streamwise)

    print("[*] Plotting waverider geometry...")
    base_plane=Plot_Base_Plane(waverider=waverider,latex=False)
    leading_edge=Plot_Leading_Edge(waverider=waverider,latex=False)
    plt.show()

    print("[*] Exporting CAD and generating meshes...")

    waverider_cad=to_CAD(waverider=waverider,sides='left',export=True,filename='waverider.stl',scale=1000)
    # waverider_cad = waverider_cad.mirror("XY", union=True)  # Ensure both sides for meshing
    # waverider_cad = waverider_cad.combineSolids().clean()

    # all_edges = waverider_cad.edges()


    # # Iterate through the edges and print their start and end points
    # for edge in all_edges.objects: # .objects accesses the underlying OCP objects
    #     print(f"Edge Start Point: {edge.startPoint()}, End Point: {edge.endPoint()}")
    #     try:
    #         waverider_cad = waverider_cad.edges(cq.selectors.NearestToPoint(edge.Center())).fillet(0.001)
    #     except Exception:
    #         print("Fillet failed on edge:", edge, "with exception:", Exception.__name__)


    print("[*] Generating surface meshes...")
    msh_path, tri_path = generate_surface_mesh(waverider_cad, mesh_size=0.1, basename="waverider", output_dir=".")