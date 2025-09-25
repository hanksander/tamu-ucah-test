import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops


'''
Standard Atmosphere set up
'''
gamma = 1.4
R = 287.058  # J/(kg*K)

# Standard atmosphere (1976) up to 9 km
alt_km = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
alt_m  = alt_km * 1000.0

T_vals = np.array([288.15, 281.65, 275.15, 268.65, 262.15,
                   255.65, 249.15, 242.65, 236.15, 229.65])  # K
P_vals = np.array([101325, 89874, 79495, 70121, 61660,
                   54019, 47181, 41060, 35651, 30858])       # Pa
rho_vals = np.array([1.225, 1.112, 1.007, 0.9093, 0.8194,
                     0.7364, 0.6601, 0.59, 0.5258, 0.4671])  # kg/m^3

# Speed of sound
a_vals = np.sqrt(gamma * R * T_vals)

# Construct the TableLookup
atm_lookup = co.TableLookup(
    xx={"h": alt_m},
    yy={
        "T": T_vals,
        "p": P_vals,
        "rho": rho_vals,
        "a_sound": a_vals,
    },
    degrees=3,
    bcs=(-1, 0),
)
    

class HypersonicBluntedConeAero(co.ExplicitSystem):
    """
    Aerodynamics model for a blunted cone in hypersonic flow.
    co.parameter()s
    ----------
    Rn  : nose radius [m]
    phi : half cone angle [rad]
    L   : length of cone [m]
    h   : geometric altitude [m]
    V   : velocity [m/s]
    alpha : angle of attack [rad]
    Outputs
    -------
    """

    #cone co.parameter()s
    Rn = input()  #nose radius
    phi = input()  #half cone angle
    L = input()  #length of cone

    Rb = L * ops.tan(phi)  #base radius
    zeta = Rn / Rb  #bluntness ratio
    A_ref = ops.pi * Rb**2  #reference area, m^2
    output.A_ref = A_ref
    SA = ops.pi * Rn * (Rn + ops.sqrt(L**2 + Rb**2))  #surface area, m^2 
    #double check SA
    

    #flight conditions
    h = input() #height
    V = input() #velocity
    alpha = input() #angle of attack

    atmos = atm_lookup(h=h)


    rho = atmos.rho
    output.rho = rho
    T_freestream = atmos.T
    a = atmos.a_sound
    M = V / a
    q = 0.5 * rho * V**2
    mu_freestream = 1.7894e-5 * (T_freestream / 273.15)**(3/2) * (273.15 + 110.4)/(T_freestream + 110.4)  #Sutherland's formula for dynamic viscosity, at T atmos
    Re = rho * V * (2 * Rn) / (mu_freestream)  #Reynolds number

    K = 1.84 / 2 # modified newtonian proportionality constant for gamma = 1.4

    Cn = ((K * Rb**2) / A_ref) * ((ops.pi * ops.sin(alpha) * ops.cos(alpha) * ops.cos(phi)**2 * (1 - (zeta**2/2)*ops.cos(phi)**2)))
    Ca = ((K * Rb**2) / A_ref) * ((ops.pi / 2)*(1 - (zeta**2/2)*ops.cos(phi)**2) * (2 * ops.cos(alpha)**2 * ops.sin(phi)**2 + ops.sin(alpha)**2 * ops.cos(phi)**2) + (zeta**2 * ops.cos(alpha)**2 * ops.cos(phi)**2))

    #aerodynamic coefficients
    output.q_dot = 7.207 * rho**0.47 * Rn**(-0.54) * V**3.5   #convective heat rate, W/m^2. See equation 4, https://ntrs.nasa.gov/api/citations/20200002354/downloads/20200002354.pdf


    CDf = (SA*0.664/ops.sqrt(Re))/(0.5 * rho * (V**2)*A_ref) #this is for a low speed laminar flat plate. It needs to be updated. Equation 6.75 from anderson is a better model
    
    CDp = Cn * ops.sin(alpha) + Ca * ops.cos(alpha) #modified newtonian theory https://apps.dtic.mil/sti/tr/pdf/AD0631149.pdf

    output.CD = CDf + CDp

    output.CL = Cn * ops.cos(alpha) - Ca * ops.sin(alpha)

    output.CLalpha = 1.1 #close enough (I have a citation I promise)

if __name__ == "__main__":

    z_test = 4500.0  # altitude in meters
    atm_props = atm_lookup(h=z_test)
    print(f"At {z_test:.0f} m:")
    print(atm_props.output)

    aero_test = HypersonicBluntedConeAero(
        Rn=0.5,
        phi=np.radians(15.0),
        L=3.0,
        h=20000.0,
        V=2000.0,
        alpha=np.radians(5.0),
    )

    print("\nHypersonic Blunted Cone Aerodynamics:")
    print(aero_test.output)
