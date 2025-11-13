# goals:
 # see if payload stl fits inside vehicle
 # calculate CG
import vedo
import os
import numpy as np
import scipy as sp

class WaveriderCase:
    def __init__(self, mesh_stl_fname):
        self.vehicle_mesh = vedo.Mesh(mesh_stl_fname)    
        self.payload_mesh = vedo.Mesh('payload.stl')
    
        # as an axisymmetric body inside a symmetrical vehicle only two paramaters
        # are needed to characterisze the location of the payload
        self.payload_x
        self.payload_theta
    def __intersection_len__(self):
        """Get the length of the intersection curve between the payload and the geometry"""
        intersection = self.vehicle_mesh.intersect_with(self.payload_mesh)
        if len(intersection) in [0, 1]:
            return 0.0
        dist = lambda p1, p2: np.sqrt(np.cdot(np.array(p1)-np.array(p2), np.array(p1)-np.array(p2))) # distance between points in n-D space
        length = sum([dist(intersection[i], intersection[i+1]) for i in range(len(intersection)-1)])
        return length
    def does_payload_fit(self):
        """return boolean whether the payload fits in its current position or not"""
        if np.isclose(self.__intersection_len__(), 0):
            return True
        else:
            return False
    def try_fit_payload(self):
        def F(x, theta):
            # change the location of the payload
                
            
            return self.x__intersection_len__()
        res = sp.optimize.minimize
            

def __test__():
    inst = WaveriderCase('example.stl')

if __name__ == "__main__":
    __test__()
    
    
