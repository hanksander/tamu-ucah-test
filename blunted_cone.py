import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops


"""This script optimizes the a blunted cone. It's base is taken from the condor glider example.

    To run you'll need to have condor installed. You can install it with pip:
    pip install condor

    you should also have updated versions of numpy, scipy, matplotlib, and pandas:
    pip install numpy scipy matplotlib pandas --upgrade
"""

#UTILITIES

def flight_path_plot(sims, **plot_kwargs):
    """Plot the flight path of one or more simulations.
    Parameters
    ----------
    sims : list of condor.Simulation
        The simulations to plot.
        plot_kwargs : dict, optional
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6.4, 3.2))
    plt.ylabel("altitude")
    plt.xlabel("range")
    # reverse zorders to show progression of sims more nicely
    zorders = np.linspace(2.1, 2.5, len(sims))[::-1]
    marker = plot_kwargs.pop("marker", "o")

    fig, axs = plt.subplots(nrows=len(sims[0].state.asdict().items()), constrained_layout=True, sharex=True)
    for sim, zorder in zip(sims, zorders):
        for ax, (state_name, state_hist) in zip(axs, sim.state.asdict().items()):
            ax.plot(sim.t, state_hist, marker = marker, zorder=zorder, **plot_kwargs)
            ax.set_ylabel(state_name)
    ax.grid(True)
            
    return ax

# ----------------------------------------------------------- #

# AERODYNAMICS 

class StandardAtmosphere20km(co.ExplicitSystem):
    """
    Standard atmosphere model valid up to 20 km.
    Parameters
    ----------
    h : float
        Geometric altitude [m]. This system is intended for 0 <= h <= 20000.
    Outputs
    -------
    T       : temperature [K]
    P       : pressure [Pa]
    rho     : density [kg/m^3]
    a_sound : speed of sound [m/s]
    """
    h = parameter()

    R = 287.05          # specific gas constant for air [J/(kg*K)]
    g0 = 9.80665        # gravitational acceleration [m/s^2]
    gamma = 1.4         # ratio of specific heats for air

    # sea level
    T0 = 288.15         # sea level temperature [K]
    P0 = 101325.0       # sea level pressure [Pa]
    rho0 = 1.225        # sea level density [kg/m^3]

    # troposphere
    a = -6.5e-3         # temperature lapse rate in troposphere [K/m]
    h_tropopause = 11000.0   # tropopause altitude [m]

    # temperature at tropopause (constant above until 20 km)
    T_tropopause = T0 + a * h_tropopause  # 216.65 K

    # Temperature: linear decrease in troposphere, constant above up to 20 km
    T = variable()
    #T = if else something(h < h_tropopause,
    #              T0 + a * h,           # troposphere
    #              T_tropopause)         # lower stratosphere (isothermal up to 20 km)

    # Pressure:
    P = variable()
    P_trop = P0 * (T_tropopause / T0) ** (-g0 / (a * R))
    #P = co.switch(h < h_tropopause,
    #              P0 * (T / T0) ** (-g0 / (a * R)),
    #              P_trop * ops.exp(-g0 * (h - h_tropopause) / (R * T_tropopause)))

    # Density from ideal gas law
    rho = variable()
    rho = P / (R * T)

    # Speed of sound
    a_sound = variable()
    a_sound = ops.sqrt(gamma * R * T)


class HypersonicBluntedConeAero(co.ExplicitSystem):
    """
    Aerodynamics model for a blunted cone in hypersonic flow.
    Parameters
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

    #cone parameters
    Rn = parameter()  #nose radius
    phi = parameter()  #half cone angle
    L = parameter()  #length of cone

    #flight conditions
    h = parameter() #height
    V = parameter() #velocity
    alpha = parameter() #angle of attack

    atmos = StandardAtmosphere(h=h)

    rho = variable()
    rho = atmos.rho
    T_freestream = variable()
    T_freestream = atmos.T
    a = variable()
    a = atmos.a_sound
    M = variable()
    M = V / a
    q = variable()
    q = 0.5 * rho * V**2
    mu_freestream = variable()
    mu_freestream = 1.7894e-5 * (T_freestream / 273.15)**(3/2) * (273.15 + 110.4)/(T_freestream + 110.4)  #Sutherland's formula for dynamic viscosity, at T atmos
    Re = variable()
    Re = rho * V * (2 * Rn) / (mu)  #Reynolds number

    #aerodynamic coefficients
    q_dot = variable()
    q_dot = 7.207 * rho**0.47 * Rn**(-0.54) * V**3.5   #convective heat rate, W/m^2. See equation 4, https://ntrs.nasa.gov/api/citations/20200002354/downloads/20200002354.pdf

    CDf = variable() # viscous Cd
    CDf = 
    CDp = variable() # pressure Cd
    CDp = 
    CD = variable() # total Cd
    CD = CDf + CDp

    CL = variable()

    Clalpha = variable()
    CLalpha = 


# ----------------------------------------------------------------------- #

# THERMALS 

class Thermal(co.ODESystem):
    """
    Simple thermal model with convective heating.
    Parameters
    ----------
    q_dot : convective heat rate [W/m^2]
    A_ref : reference area [m^2]
    m     : mass [kg]
    cp    : specific heat [J/(kg*K)]
    T0    : initial temperature [K]
    Outputs
    -------
    T : temperature [K]
    ----
    TODO - generalize to multiple materials with different thermal properties"""
    q_dot = parameter()  # convective heat rate, W/m^2
    A_ref = parameter()  # reference area, m^2
    m = parameter()      # mass, kg
    cp = parameter()     # specific heat, J/(kg*K)
    T0 = parameter()     # initial temperature, K

    T = state()          # temperature, K

    dot[T] = q_dot * A_ref / (m * cp)

    initial[T] = T0
# ----------------------------------------------------------------------- #

# Structures

'''
class Structure(co.ExplicitSystem):
'''
# ------------------------------------------------------- #

# Mass Props

'''
class MassProps(co.ExplicitSystem):'''

# ------------------------------------------------------- #

# S&C

'''
class SC(co.ExplicitSystem):'''
# ------------------------------------------------------- #

#Trajectory

#All of this is adapted from the condor glider example, not right rn (and not mine)

class Glider(co.ODESystem):
    r = state()
    h = state()
    gamma = state()
    v = state()

    alpha = modal()

    CL_alpha = parameter()
    CD_0 = parameter()
    CD_i_q = parameter()
    g = parameter()

    CL = CL_alpha * alpha
    CD = CD_0 + CD_i_q * CL**2

    dot[r] = v * ops.cos(gamma)
    dot[h] = v * ops.sin(gamma)
    dot[gamma] = (CL * v**2 - g * ops.cos(gamma)) / v
    dot[v] = -CD * v**2 - g * ops.sin(gamma)

    initial[r] = 0.0
    initial[h] = 1.0
    initial[v] = 15.0
    initial[gamma] = 30 * ops.pi / 180.0

class Land(Glider.Event):
    function = h
    terminate = True

class LandSim(Glider.TrajectoryAnalysis):
    tf = 20.0

class MaxAlt(Glider.Event):
    function = gamma
    max_alt = state()
    update[max_alt] = h

class UpdateAlpha(Glider.Mode):
    condition = 1
    alpha_coeff_0 = parameter()
    alpha_coeff_1 = parameter()
    alpha_coeff_2 = parameter()

    action[alpha] = alpha_coeff_0 + alpha_coeff_1 * t + alpha_coeff_2 * t**2

class AlphaSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.0
    initial[h] = 100.0
    initial[v] = 100.0
    initial[gamma] = 30 * ops.pi / 180.0
    tf = 400.0

    area = trajectory_output(integrand=dot[r] * h)
    max_h = trajectory_output(max_alt)
    max_r = trajectory_output(r)

    class Options:
        state_rtol = 1e-12
        state_atol = 1e-15
        adjoint_rtol = 1e-12
        adjoint_atol = 1e-15


#Optimization

class GlideOpt(co.OptimizationProblem):
    alpha_coeff_0 = variable(
        initializer=0.001,
        lower_bound=-1.0,
        upper_bound=1,
        warm_start=False,
    )

    alpha_coeff_1 = variable(
        initializer=0.001,
        lower_bound=-1,
        upper_bound=1,
        warm_start=False,
    )

    # alpha_coeff_2 = variable(
    #     initializer=0.001,
    #     lower_bound=-1,
    #     upper_bound=1,
    #     warm_start=False,
    # )

    A = 3e-1

    sim = AlphaSim(CL_alpha=0.11 * A, CD_0=0.05 * A, CD_i_q=0.05, g=1.0, 
                   alpha_coeff_0=alpha_coeff_0, alpha_coeff_1=alpha_coeff_1, alpha_coeff_2=0.)#alpha_coeff_2)
    
    objective = sim.max_r

    constrain = sim.max_h <= 110.0
    constrain = sim.v >= 20.0


    class Options:
        exact_hessian = False
        print_level = 0
        tol = 1e-1
        max_iter = 8

# class VelocityExtrema(Glider.Event):
#     function = dot[v]
#     max_v = state()
#     min_v = state()

#     if (dot[dot[v]] >0):
#         update[max_v] = v
#     elif (dot[dot[v]] <0):
#         update[min_v] = v

if __name__ == "__main__":
    A = 3e-1
    Land_Sim = LandSim(CL_alpha=0.11 * A, CD_0=0.05 * A, CD_i_q=0.05, g=1.0)

    Alpha_Sim = AlphaSim(CL_alpha=0.11 * A, CD_0=0.05 * A, CD_i_q=0.05, g=1.0, alpha_coeff_0=0.1, alpha_coeff_1=1.0, alpha_coeff_2=0.0)

    # flight_path_plot([Land_Sim, Alpha_Sim])
    # plt.legend(["No Control", "Alpha Control"])
    # plt.show()

    opt_range = GlideOpt()

    ax = flight_path_plot([opt_range.sim])
    ax.text(
        *(0.05, 0.92),
        f"max range: {opt_range.sim.max_r} ($\\alpha_0={opt_range.alpha_coeff_0}, alpha_1={opt_range.alpha_coeff_1}$",#, alpha_2={opt_range.alpha_coeff_2}$",
        transform=ax.transAxes,
    )
    plt.legend(["Optimized Glide Trajectory"])
    plt.show()
    
