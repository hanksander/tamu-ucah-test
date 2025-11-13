# goals:
 # see if payload stl fits inside vehicle
 # calculate CG
import vedo
import os
import numpy as np
import scipy as sp

class WaveriderCase:
    def __init__(self, mesh_stl_fname):
        assert os.path.exists(f'cg'), "CWD must be the main directory"
        self.vehicle_mesh = vedo.Mesh(f'cg/{mesh_stl_fname}')    
        self.payload_mesh = vedo.Mesh(r'cg\\payload.stl')

        self._payload_rotation = 0.0 # current position of the cylinder in radians
        c = self.vehicle_mesh.center_of_mass() # default position is the center of mass of the vehicle
        self.payload_mesh.pos(x=c[0], y=c[1], z=c[2])

        # all masses that are not the vehicle's aeroshell go here for 
        self.additional_masses = [{"fname":"payload.stl",
                                       "mass":9.0, # kgs
                                       "pos":[c[0], c[1], c[2]]
                                       }]

    # rotation tracking (not currently used)
    @property
    def payload_rotation(self):
        return self._payload_rotation
    @payload_rotation.setter
    def payload_rotation_setter(self, value):
        # put it back then rotate to new
        self.payload_mesh.rotate_y(-self._payload_rotation + value)
        self._payload_rotation = value

    # grapher (for debugging)
    def graph(self):
        """graphs the payload and vehicle with Vedo"""
        plt = vedo.Plotter(title="Multiple STL files", axes=1)
        print(f'about to show...')
        plt.show([self.payload_mesh, self.vehicle_mesh.alpha(0.5)], interactive=True)

    # interferance checkers
    def __intersection_len__(self):
        """Returns float length of the intersection curve between the payload and the geometry"""
        intersection = self.vehicle_mesh.intersect_with(self.payload_mesh).points
        if len(intersection) in [0, 1]:
            return 0.0
        dist = lambda p1, p2: np.sqrt(np.dot(np.array(p1)-np.array(p2), np.array(p1)-np.array(p2))) # distance between points in n-D space
        length = sum([dist(intersection[i], intersection[i+1]) for i in range(len(intersection)-1)])
        return length
    
    def payload_clears(self):
        """return boolean 
            True the payload fits in its current without intersecting the vehicle
            False if else"""
        if np.isclose(self.__intersection_len__(), 0):
            return True # if there is no intersection curve, then yea it clears!
        else:
            return True
        
    def payload_inside(self):
        """Returns boolean
            True if the centroid of the payload is inside the vehicle"""
        # make a vector from payload centroid to infinity
        # if the vector crosses an odd number of borders then it's inside
        p_CG = self.payload_mesh.center_of_mass()
        ray_intersections = self.vehicle_mesh.intersect_with_line(p_CG, np.array([1000.0, 0.0, 0.0], dtype=float))
        if len(ray_intersections)%2==1:
            return True
        return False
    
    def payload_fit(self):
        """returns boolean 
            True if the payload both clears and is inside the vehicle
            False if the payload does not meet either one of the criteria"""
        if self.payload_clears() and self.payload_inside():
            return True
        return False
    
    def try_fit_payload(self):
        def F(params):
            """function to be minimized
            punishment terms for:
                1) being outside the vehicle
                2) clipping through the vehicle


            returns whether it was able to get it to fit
            """
            x, z = params
            # change the location of the payload
            self.payload_mesh.pos(x=x,y=0,z=z)
            
            interferance = self.__intersection_len__()

            if self.payload_fit():
                exclusion = 0
            else:
                d = self.payload_mesh.center_of_mass() - self.vehicle_mesh.center_of_mass() # type: ignore
                exclusion = np.dot(d, d) + 10
            
            return interferance + exclusion 
        res = sp.optimize.minimize(F, [0.0, 0.0], method="Powell") 
        print(f'x, z = {res.x}\n'
              f'successful? {res.success}\n'
              f'statusmmsg: {res.status}\n'
              f'function: {res.fun}')
        if np.isclose(res.fun, 0.0):
            return True
        else:
            return False
            

    # shell CG + payload
    def get_cg(self):
        """Calculate the CG of the vehicle based on the VSP file and all included additional masses"""
        def material_thickness(x,y,z):
            """return the estimated thickness of the material. Should be proportional to the aerodynamic heating.
            
            One weakness of this is that if the geometry of the wing is more than twice the thickness of the material, then it will return a 
            thickness that is not physically possible. This is kind of OK though, because this simply means that IRL we will have to either round the edge
            or use a denser material (tung tung tung tung tung tung sten)"""

            # get cbAero data and determine the thickness

            return 0.01 # 1cm thick all round until we get the cbAero data. Then interpolate between the discreet heating points.
        def skin_area_density(x,y,z):
            """define the skin density, ie, kg/m2
            Is the density of the material times its local thickness"""

            # get cbAero data and determine the material
            al_density = 2700 # density of aluminum in kg/m3 
            steel_density = 7850 # density of steel in kg/m3
            tungsten_density = 19300 # denity of tungsten in kg/m3
            material_density = steel_density

            return material_thickness(x,y,z)*material_density 
        def mesh_to_cg(
                            axis=0, # of the vehicle's x axis in the stl's coordinate system. 0 for x, 1, for y, 2 for z,
                                f = lambda x: np.min(x), # max if the vehicle is oriented with +x toward +x; min if the vehicle is -x toward +x
                                    ):
            """a mesh object is a series of triangles. 
            We will calculate the centroid and size of each of the triangles in order to do calculate the shell-center of area of the craft.
            returns the center of area and the surface area"""

            mesh = self.vehicle_mesh
            mesh.triangulate # type:  ignore
            points = mesh.points # type: ignore
            face_indeces = mesh.cells # type: ignore
            faces = points[face_indeces]
            v0 = faces[:, 0]
            v1 = faces[:, 1]
            v2 = faces[:, 2]

            centroids = []
            masses = []

            # for the ith triangle
            for i in range(len(v0)):
                A = v0[i]
                B = v1[i]
                C = v2[i]

                centroids.append(
                    (0.33334)*np.array([A[0] + B[0] + C[0],
                                    A[1] + B[1] + C[1],
                                    A[2] + B[2] + C[2]])
                )
                AB = B-A
                AC = C-A
                masses.append(0.5*np.linalg.norm(np.cross(AB, AC))*skin_area_density(0,0,0))
            
            centroids = np.array(centroids)
            masses = np.array(masses)

            # total mass calculation
            mass = np.sum(masses)

            # centre of gravity calculation
            CG = np.array([0.0, 0.0, 0.0])
            for i in range(len(v0)):
                CG += masses[i]*centroids[i] 
            else:
                CG /= mass

            vehicle_front = f(centroids[:, axis]) 
            # print(f'CG: {CG}; LE location: {min(centroids[axis]), max(centroids[axis])}')
            CG_from_front = np.abs(vehicle_front - CG[axis])

            # print(f'surface area: {SA}\n'
            #       f'centre of gravity: {CG_from_front}')
            return CG_from_front, mass

    # generate mesh for the CG calculation

        # calculate Center of area
        self.CG, self.mass = mesh_to_cg()

        for _m in self.additional_masses:
            self.CG = (self.CG*self.mass + _m["mass"]*_m["pos"][0]) / (self.mass+_m["mass"])
            self.mass += _m["mass"]

        if len(self.additional_masses) == 0:
            print(f'WARN: no additional masses added! Need to add at least the payload.')

        return self.CG
    
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
    inst = WaveriderCase('example.stl')
    # inst.__intersection_len__()
    # print(f'payload inside: {inst.payload_inside()}\n'
    #       f'payload interfere: {inst.payload_clears()}\n'
    #       f'payload fit: {inst.payload_fit()}')
    inst.graph()
    inst.try_fit_payload()
    print(f'payload inside: {inst.payload_inside()}\n'
          f'payload clears: {inst.payload_clears()}\n'
          f'payload fit: {inst.payload_fit()}')
    print(inst.get_cg())
    inst.graph()

if __name__ == "__main__":
    __test__()
    
    
