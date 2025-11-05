import gmsh
import os
import trimesh
import numpy as np

def reparameterize_and_remesh_stl(stl_path, output_prefix="G", cl_scales=(0.2, 0.1, 0.05)):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("reparam")

    print("Cleaning up stl with trimesh...")
    mesh = trimesh.load_mesh(stl_path)
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.merge_vertices()
    mesh.remove_infinite_values()
    mesh.remove_duplicate_faces()
    mesh.export(stl_path)

    print(f"[INFO] Loading STL: {stl_path}")
    gmsh.merge(stl_path)
    #gmsh.model.mesh.closeHole()
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.removeDuplicateElements()
    gmsh.model.mesh.checkMesh(True)
    #gmsh.model.mesh.removeSmallElements(1e-6)


    print("[INFO] Creating topology from STL...")
    gmsh.model.mesh.classifySurfaces((10/180)*np.pi, True, True, (10/180)*np.pi)


    gmsh.model.mesh.createGeometry()

    reparam_msh = f"{output_prefix}_reparam.msh"
    print(f"[INFO] Saving reparameterized mesh to {reparam_msh}")
    gmsh.write(reparam_msh)

    gmsh.finalize()

    # Now sequentially remesh with decreasing characteristic lengths
    for cl in cl_scales:
        print(f"[INFO] Remeshing with clscale={cl} ...")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(reparam_msh)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", cl)
        gmsh.model.mesh.generate(2)

        remeshed_file = f"{output_prefix}_cl{cl:.2f}.msh"
        gmsh.write(remeshed_file)
        gmsh.finalize()

        print(f"[INFO] Saved remeshed file: {remeshed_file}")

    print("[DONE] Reparameterization and remeshing complete.")

if __name__ == "__main__":
    reparameterize_and_remesh_stl("waverider.stl", output_prefix="reparam_waverider", cl_scales=(0.2, 0.1, 0.05))
