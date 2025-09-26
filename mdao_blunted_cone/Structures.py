# do mass properties here for now
import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops


class ConeShellMass(co.ExplicitSystem):
    """
    Mass of a conical shell with finite thickness and rounded nose.
    """
    phi = input()       # cone half angle [rad]
    r_cone = input()   # base radius [m]
    r_nose = input()    # nose radius [m]
    thickness = input() # shell thickness [m]
    rho = input()       # material density [kg/m^3]

    m = 2 * ops.pi * thickness * (
        -r_nose * (2 * phi - ops.pi) / 2
        + ops.sqrt((r_cone - r_nose * ops.cos(phi))**2 / ops.sin(phi)**2)
    ) * rho

    output.m = m


class ConeShellCG(co.ExplicitSystem):
    """
    Center of gravity of a conical shell with a rounded nose.
    """
    phi = input()
    r_cone = input()
    r_nose = input()

    cg = -(r_cone * ops.cos(phi) + r_nose * ops.sin(phi) - r_nose)**2 / (
        (r_nose * (2 * phi - ops.pi)
         - 2 * ops.sqrt((r_cone - r_nose * ops.cos(phi))**2 / ops.sin(phi)**2))
        * ops.sin(phi)**2
    )

    output.cg = cg


class PayloadCG(co.ExplicitSystem):
    """
    Center of mass location of payload, assuming it sits as far forward
    as possible for stability.
    """
    phi = input()
    r_cone = input()
    r_nose = input()

    payload_radius = 0.0640 / 2  # m
    payload_length = 0.161       # m

    cond1 = r_nose**2 - payload_radius**2 >= 0
    x_payload = ops.if_else(
        (cond1, ops.sqrt(r_nose**2 - payload_radius**2) + r_nose),
        (payload_radius - r_nose * ops.sin(ops.pi / 2 - phi)) / ops.tan(phi) + r_nose * (1 - ops.cos(ops.pi / 2 - phi)),
    )

    cg = x_payload + payload_length / 2

    output.cg = cg


class OverallCG(co.ExplicitSystem):
    """
    Overall CG along the X axis given multiple components.
    NOTE: CG measured from the tip of the nose.
    """
    m_aero = input()  # aeroshell mass
    x_aero = input()  # aeroshell CG

    m_payload = input()  # payload mass
    x_payload = input() # payload CG

    num = m_aero * x_aero + m_payload * x_payload
    den = m_aero + m_payload
    cg_total = num / den

    output.cg = cg_total

if __name__ == "__main__":
    # test mass and cg calculations
    phi = np.deg2rad(14)
    r_cone = 0.2
    r_nose = 0.01

    mass_test = ConeShellMass(
        phi=phi,
        r_cone=r_cone,
        r_nose=r_nose,
        thickness=0.005,
        rho=2700,
    )
    print("\nCone Shell Mass:")
    print(mass_test.output)


'''

class MassElement:
    def __init__(self, m, x, y, z):
        self.m = m
        self.x = x
        self.y = y
        self.z = z
        

def cone_cg(phi, # angle of the cone in radians
            r_cone, # radius of the base of the cone
             r_nose, # radius of the nose
             ):
    """center of gravity of the conical shell with a leading point radius"""
    sin = np.sin
    cos = np.cos
    tan = np.tan
    sqrt = np.sqrt
    pi = 3.14
    cg = -(r_cone*cos(phi) + r_nose*sin(phi) - r_nose)**2/((r_nose*(2*phi - pi) - 2*sqrt((r_cone - r_nose*cos(phi))**2/sin(phi)**2))*sin(phi)**2)
    return cg


def cone_mass(phi, # angle of the cone in radians
            r_cone, # radius of the base of the cone
             r_nose, # radius of the nose
             thickness, # thickness of the aeroshell
              rho # average density of the aeroshell
              ):
    """Calculates the mass assuming a conical shell with a leading point radius and a constant (thin) thickness"""
    pi = np.pi
    sqrt = np.sqrt
    cos = np.cos
    sin = np.sin
    m = 2*pi*thickness*(-r_nose*(2*phi - pi)/2 + sqrt((r_cone - r_nose*cos(phi))**2/sin(phi)**2))*rho
    return m


def payload_location(phi, # angle of the cone in radians
            r_cone, # radius of the base of the cone
             r_nose, # radius of the nose
             ):
    """Calculates the location of the center of mass of the payload, assuming that you want the payload as far forward as you can possibly have it (for maximum stability)"""

    #  defined by the competition - https://hypersonics.tamu.edu/2025-undergraduate-hypersonic-flight-design-competition/
    payload_radius = 0.0640/2
    payload_length = 0.161 

    # if the nose cone radius is big
    if r_nose**2 - payload_radius**2>=0:
        x_payload = np.sqrt(r_nose**2 - payload_radius**2) + r_nose     # the point at which the payload starts
    else:
        x_payload = (payload_radius - r_nose*np.sin(3.14/2-phi))/np.tan(phi) + r_nose*(1 - np.cos(np.pi / 2 - phi))

    return x_payload + payload_length/2


def CG(*elements:MassElement):
    """Finds CG along X axis
    NOTE: The CG is measured from the tip of the nose."""
    # m_i * r_i
    num = np.dot(np.array([i.m for i in elements]),
           np.array([i.x for i in elements]))
    
    # m_i
    den = np.sum(np.array([i.m for i in elements]))
    return num/den


# input parameters (should they be condor.parameter???)
phi = np.deg2rad(14)
r_cone = 0.08
r_nose = 0.0117

# create mass objects
aeroshell = MassElement(m=cone_mass(phi=phi,r_cone=r_cone,r_nose=r_nose, thickness=0.005, rho=7850.0),
                         x=cone_cg(phi=phi,r_cone=r_cone,r_nose=r_nose),
                          y=0,
                           z=0)
payload = MassElement(m=9, x=payload_location(phi=phi,r_cone=r_cone,r_nose=r_nose), y=0, z=0)

# debugging messages
print(f'aeroshell: {aeroshell.m, aeroshell.x}')
print(f'payload: {payload.m, payload.x}')

# ouptut CG
print(f'overall CG: {CG(payload, aeroshell)}')


# to do: 
#   consider effeccts of thickness
#   consider effects of variable thickness (perhaps thicker near the faces experiencing most heating per mass)

'''