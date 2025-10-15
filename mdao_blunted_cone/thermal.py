import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops

'''
Thermal model for a blunted cone vehicle.

Classes:
Thermal: dT/dt
StagHeating: energy balance (heating rates)
MinTw: optimize nose radius and emissivity

Sources: NASA Aerothermal Heating Slides
'''


R_air   = 287.058        # J/(kg K)
gamma   = 1.4            # unsure about this one
sigmaSB = 5.670374419e-8 # (Stefan-Boltzmann)
k_SG    = 1.7415e-4      # (W * s^3) / (m^(2.5) * kg^0.5) effectively Sutton-Graves constant


class Thermal(co.ODESystem):
    """
    Equations
    -------
    dT/dt = q_dot * A_ref / (m * cp)

    """
    q_dot = co.input()  # convective heat rate, W/m^2
    A_ref = co.input()  # reference area, m^2
    m = co.input()      # mass, kg
    cp = co.input()     # specific heat, J/(kg*K)
    T0 =   co.input()     # initial temperature, K
    T = co.output()     # temperature, K

    T = q_dot * A_ref / (m * cp)


#flight conditions
M = co.input()   
rho_inf =  co.input()  
T_inf =  co.input()  

#Vehicle params 
k_s = co.input()  # W/(m·K) effective through-thickness conductivity
L_thk = co.input()  # m wall/TPS thickness
T_back = co.input()  # K back-face/structure temp target/limit


class StagHeating(co.AlgebraicSystem):
    # design variables
    Rn  = co.variable(initializer=0.5, lower_bound=0.01, upper_bound=5.0)   # m (nose radius)
    eps = co.variable(initializer=0.85, lower_bound=0.5,  upper_bound=0.98) # emissivity (wall properties)

    # wall temp 
    Tw  = co.variable(initializer=800.0, lower_bound=250.0, upper_bound=3000.0)  # K

    q_conv  = co.output()
    q_rerad = co.output()
    q_cond  = co.output()
    q_net = co.output()

    # freestream conditions
    a_inf = ops.sqrt(gamma * R_air * T_inf)
    V_inf = M * a_inf

    # heating rates for blunted cone vehicle
    co.residual(q_conv  == k_SG * ops.sqrt(rho_inf / Rn) * V_inf**3) #Sutton-Graves law convective heating
    co.residual(q_rerad == eps * sigmaSB * Tw**4) #stefan-boltzmann law radiative cooling
    co.residual(q_cond  == (k_s / L_thk) * (Tw - T_back)) #conduction cooling
  
    co.residual(q_net == q_conv - (q_rerad + q_cond)) #energy balance


class MinTw(co.OptimizationProblem):
    model = StagHeating()
    objective = model.Tw  # Minimize Tw directly
   
    co.constraint(model.Rn >= 0.02)     # geometry lower bound (already in bounds)
    co.constraint(model.eps <= 0.98) 