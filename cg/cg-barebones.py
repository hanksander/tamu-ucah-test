import numpy as np
import vedo # it looks


stl = 'test.stl' # path to the STL file
class Waverider:
    def __init__(self, stl_fname):
        self.mesh = vedo.mesh(stl_fname)

        self.additional_masses = [] # list of dictionaries with the data for all the components in the aircraft


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

    def graph_vehicle(self):
        """Plots the meshes of the vehicle and all additional bodies in Vedo"""
        filenames = [self.additional_masses[i]["fname"] for i in range(len(self.additional_masses))]
        positions =  [self.additional_masses[i]["pos"] for i in range(len(self.additional_masses))]
        filenames = list(filenames)
        meshes = [vedo.Mesh(filenames[i]).pos(positions[i]) for i in range(len(filenames))]

        # include aeroshell stl
        self.update_airframe_mesh()
        meshes.append(self.airframe_mesh.alpha(0.5)) # type: ignore


        plt = vedo.Plotter(title="Multiple STL files", axes=1)
        print(f'about to show...')
        plt.show(meshes, interactive=True)

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

            mesh = self.airframe_mesh
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


def test():

    stl = 'test.stl' # path to the STL file
    my_waverider = Waverider(stl)
    my_waverider.add_mass(fname="payload.stl", mass=9.0, x_pos=0.25)         # add the payload mass

    # graph all parts
    my_waverider.graph_vehicle()

    # test CG calculator
    my_waverider.get_cg()
    print(my_waverider.CG)

if __name__ == "__main__":
    test()