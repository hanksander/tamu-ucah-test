# goals:
 # see if payload stl fits inside vehicle
 # calculate CG

import os as os
import numpy as np
import scipy as sp
import trimesh as tm

class OgiveCase:
    def __init__(self, mesh_stl_fname):
        self.vehicle_mesh = tm.load_mesh(f'{mesh_stl_fname}')    
        self.payload_mesh = tm.load_mesh(F'{os.path.dirname(os.path.abspath(__file__))}/payload.stl')

        # snap the payload to the centroid of the vehicle
        c=self.vehicle_mesh.centroid
        to_coincide_centroids = c - self.payload_mesh.centroid
        print(f'moving {to_coincide_centroids} to make centroids coincident')
        self.payload_mesh.apply_translation(to_coincide_centroids)

        self.vehicle_volume = self.vehicle_mesh.volume

        # all masses that are not the vehicle's aeroshell go here for 
        self.additional_masses = [{"fname":"payload.stl",
                                       "mass":9.0, # kgs
                                       "pos":[c[0], c[1], c[2]]
                                       }]


    # grapher (for debugging)
    def graph(self):
        # needs "pyglet<2"
        total_mesh = self.vehicle_mesh.union(self.payload_mesh)
        total_mesh.show()
    
    def payload_fit(self):
        """returns boolean 
            True if the payload fits entirely inside the vehicle
            False if it does not
            """
        combined = self.vehicle_mesh.union(self.payload_mesh)
        return np.isclose(self.vehicle_volume, combined.volume)
    
    def try_fit_payload(self, verbose=False):
        def F(params):
            """function to be minimized. Is the volume of the payload that sticks outside the vehicle
            """
            x, z = params
            # change the absolute location of the payload
            current_centroid = self.payload_mesh.centroid
            target_centroid = np.array([x, current_centroid[1], z])
            tran_vec = target_centroid - current_centroid
            self.payload_mesh.apply_translation(tran_vec)

            # do the join and get the volume
            bool_mesh = self.vehicle_mesh.union(self.payload_mesh)
            return bool_mesh.volume - self.vehicle_volume

        starting_pt = self.vehicle_mesh.centroid
        res = sp.optimize.minimize(F, [starting_pt[0], starting_pt[2]], method="tnc", tol=1e-5) 
        # CG - 8 seconds
        # tnc - <1 second
        # Powell - 51 seconds
        # BFGS - >15 seconds

        if verbose:
            print(f'x, z = {res.x}\n'
                f'successful? {res.success}\n'
                f'statusmmsg: {res.status}\n'
                f'function: {res.fun}')
        is_zero = lambda x: bool(np.abs(x)<5e-5)
        if is_zero(res.fun):
            return (True, res.fun)
        else:
            return (False, res.fun)
            

    # shell CG + payload (requires vedo)
    # def get_cg(self):

    #     """Calculate the CG of the vehicle based on the VSP file and all included additional masses"""
    #     def material_thickness(x,y,z):
    #         """return the estimated thickness of the material. Should be proportional to the aerodynamic heating.
            
    #         One weakness of this is that if the geometry of the wing is more than twice the thickness of the material, then it will return a 
    #         thickness that is not physically possible. This is kind of OK though, because this simply means that IRL we will have to either round the edge
    #         or use a denser material (tung tung tung tung tung tung sten)"""

    #         # get cbAero data and determine the thickness

    #         return 0.01 # 1cm thick all round until we get the cbAero data. Then interpolate between the discreet heating points.
    #     def skin_area_density(x,y,z):
    #         """define the skin density, ie, kg/m2
    #         Is the density of the material times its local thickness"""

    #         # get cbAero data and determine the material
    #         al_density = 2700 # density of aluminum in kg/m3 
    #         steel_density = 7850 # density of steel in kg/m3
    #         tungsten_density = 19300 # denity of tungsten in kg/m3
    #         material_density = steel_density

    #         return material_thickness(x,y,z)*material_density 
    #     def mesh_to_cg(
    #                         axis=0, # of the vehicle's x axis in the stl's coordinate system. 0 for x, 1, for y, 2 for z,
    #                             f = lambda x: np.min(x), # max if the vehicle is oriented with +x toward +x; min if the vehicle is -x toward +x
    #                                 ):
    #         """a mesh object is a series of triangles. 
    #         We will calculate the centroid and size of each of the triangles in order to do calculate the shell-center of area of the craft.
    #         returns the center of area and the surface area"""
    #         import vedo
    #         mesh = vedo.Mesh(rf'{mesh_stl_fname}')    
    #         self.payload_mesh = vedo.Mesh(r'./401/payload.stl')
    #         mesh.triangulate # type:  ignore
    #         points = mesh.points # type: ignore
    #         face_indeces = mesh.cells # type: ignore
    #         faces = points[face_indeces]
    #         v0 = faces[:, 0]
    #         v1 = faces[:, 1]
    #         v2 = faces[:, 2]

    #         centroids = []
    #         masses = []

    #         # for the ith triangle
    #         for i in range(len(v0)):
    #             A = v0[i]
    #             B = v1[i]
    #             C = v2[i]

    #             centroids.append(
    #                 (0.33334)*np.array([A[0] + B[0] + C[0],
    #                                 A[1] + B[1] + C[1],
    #                                 A[2] + B[2] + C[2]])
    #             )
    #             AB = B-A
    #             AC = C-A
    #             masses.append(0.5*np.linalg.norm(np.cross(AB, AC))*skin_area_density(0,0,0))
            
    #         centroids = np.array(centroids)
    #         masses = np.array(masses)

    #         # total mass calculation
    #         mass = np.sum(masses)

    #         # centre of gravity calculation
    #         CG = np.array([0.0, 0.0, 0.0])
    #         for i in range(len(v0)):
    #             CG += masses[i]*centroids[i] 
    #         else:
    #             CG /= mass

    #         vehicle_front = f(centroids[:, axis]) 
    #         # print(f'CG: {CG}; LE location: {min(centroids[axis]), max(centroids[axis])}')
    #         CG_from_front = np.abs(vehicle_front - CG[axis])

    #         # print(f'surface area: {SA}\n'
    #         #       f'centre of gravity: {CG_from_front}')
    #         return CG_from_front, mass

    # # generate mesh for the CG calculation

    #     # calculate Center of area
    #     self.CG, self.mass = mesh_to_cg()

    #     for _m in self.additional_masses:
    #         self.CG = (self.CG*self.mass + _m["mass"]*_m["pos"][0]) / (self.mass+_m["mass"])
    #         self.mass += _m["mass"]

    #     if len(self.additional_masses) == 0:
    #         print(f'WARN: no additional masses added! Need to add at least the payload.')

    #     return self.CG
    
    def add_mass(self, fname, mass, x_pos, y_pos=0, z_pos=0):
        """adds to the list of non-aeroshell bodies to add to the CG and interferance simulation
        Parameters:
            fname is the name of the component's stl file
            mass is the mass of the compoenent in kg
            x_pos is the position of the object (in meters) from the nose of the vehicle
            positive y_pos is toward the right wing
            positive z_pos is the vehicle's up
            """
            
        # adds an additional mass to the the list of massed components
        # the actual geometry of the object only comes into play when evaluting dynamic properties such as moment of inertia
        self.additional_masses.append({"fname":fname,
                                       "mass":mass,
                                       "pos":[x_pos, y_pos, z_pos]
                                       })


def __test__():
    inst = OgiveCase('ogive.stl')

    print(f'beginning case:\n')
    print(f'payload fit?')
    if inst.payload_fit():
        print('yes\n')
    else:
        print('no\n')

    print(f'fitting payload...')
    successful = inst.try_fit_payload(verbose=True)

    if successful:
        print('payload fit successful!\n')
    else:
        print('payload fit failed!\n')
        
    inst.graph()

if __name__ == "__main__":
    __test__()
    
