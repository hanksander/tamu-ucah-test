import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops

from aerodynamics import *
from thermal import *
from Structures import *

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

    Rn = parameter()  # nose radius
    phi = parameter()  # half cone angle
    L = parameter()  # length of cone

    aero = HypersonicBluntedConeAero(Rn=Rn, phi=phi, L=L, h=h, V=v, alpha=alpha)
    #mass = ConeShellMass(phi=phi, r_cone=L*ops.tan(phi), r_nose=Rn, thickness=0.0005, rho=1500)
    mass = 25

    g = 9.81  # m/s^2
    #rho = atm_lookup(h=h).rho
    
    rho = aero.rho

    L =  rho* v**2 * aero.A_ref * aero.CL / 2
    D = rho * v**2 * aero.A_ref * aero.CD / 2
    #W = mass.m * g
    W = mass*g

    dot[r] = v * ops.cos(gamma)
    dot[h] = v * ops.sin(gamma)
    #dot[gamma] = (L - W * ops.cos(gamma)) / (v * mass.m)
    #dot[v] = -D/mass.m - g * ops.sin(gamma)
    dot[gamma] = (L - W * ops.cos(gamma)) / (v * mass)
    dot[v] = -D/mass - g * ops.sin(gamma)

    initial[r] = 0.0
    initial[h] = 8000
    initial[v] = atm_lookup(h=initial[h]).a_sound * 7.0
    initial[gamma] = 20 * ops.pi / 180.0  # radians

class Land(Glider.Event):
    function = h
    terminate = True

class LandSim(Glider.TrajectoryAnalysis):
    tf = 20.0

class MaxAlt(Glider.Event):
    # fires every step so it can track the max altitude seen
    function = gamma
    max_alt = state()
    initial[max_alt] = 0
    # if current h > stored max_alt then update, else keep stored
    update[max_alt] = ops.if_else((h > max_alt, h), max_alt)


class MaxRange(Glider.Event):
    # track the maximum horizontal range seen
    function = gamma
    max_r = state()
    initial[max_r] = 0
    update[max_r] = ops.if_else((r > max_r, r), max_r)


# class MaxVel(Glider.Event):
#     # track the maximum velocity seen
#     function = dot[v]
#     max_v = state()
#     initial[max_v] = 0
#     update[max_v] = ops.if_else((v > max_v, v), max_v)


# class MinVel(Glider.Event):
#     # track the minimum velocity seen
#     function = dot[v]
#     min_v = state()
#     initial[min_v] = 0
#     update[min_v] = ops.if_else((v < min_v, v), min_v)


class UpdateAlpha(Glider.Mode):
    condition = 1
    alpha_coeff_0 = parameter()
    alpha_coeff_1 = parameter()
    alpha_coeff_2 = parameter()

    action[alpha] = alpha_coeff_0 + alpha_coeff_1 * t + alpha_coeff_2 * t**2

class AlphaSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.0
    initial[h] = 1000
    initial[v] = atm_lookup(h=initial[h]).a_sound * 7.0
    initial[gamma] = 20 * ops.pi / 180.0  # radians
    tf = 800.0

    area = trajectory_output(integrand=dot[r] * h)
    max_h = trajectory_output(MaxAlt.max_alt)
    max_r = trajectory_output(r)
    # max_r = trajectory_output(MaxRange.max_r)
    # max_v = trajectory_output(MaxVel.max_v)
    # min_v = trajectory_output(MinVel.min_v)

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

    # alpha_coeff_1 = variable(
    #     initializer=0.001,
    #     lower_bound=-1,
    #     upper_bound=1,
    #     warm_start=False,
    # )

    # alpha_coeff_2 = variable(
    #     initializer=0.001,
    #     lower_bound=-1,
    #     upper_bound=1,
    #     warm_start=False,
    # )

    Rn = variable(
        initializer=0.01,
        lower_bound= 0.01,
        upper_bound= 0.1,
        warm_start=False,
    )
    phi = variable(
        initializer=np.radians(10.0),
        lower_bound=np.radians(5.0),
        upper_bound=np.radians(25.0),
        warm_start=False,
    )
    L = variable(
        initializer=0.5,
        lower_bound=0.1,
        upper_bound=1,
        warm_start=False,
    )

    A = 3e-1

    sim = AlphaSim(Rn=Rn, phi=phi, L=L, alpha_coeff_0=alpha_coeff_0, alpha_coeff_1=0., alpha_coeff_2=0.)#alpha_coeff_2)
    
    objective = sim.max_r

    constrain = sim.max_h <= 9000
    
    #constrain(sim.v >= 20.0)



    class Options:
        exact_hessian = False
        print_level = 0
        tol = 1e-5
        max_iter = 10

# class VelocityExtrema(Glider.Event):
#     function = dot[v]
#     max_v = state()
#     min_v = state()

#     if (dot[dot[v]] >0):
#         update[max_v] = v
#     elif (dot[dot[v]] <0):
#         update[min_v] = v

if __name__ == "__main__":
    # A = 3e-1
    # Land_Sim = LandSim(CL_alpha=0.11 * A, CD_0=0.05 * A, CD_i_q=0.05, g=1.0)

    # Alpha_Sim = AlphaSim(CL_alpha=0.11 * A, CD_0=0.05 * A, CD_i_q=0.05, g=1.0, alpha_coeff_0=0.1, alpha_coeff_1=1.0, alpha_coeff_2=0.0)

    # flight_path_plot([Land_Sim, Alpha_Sim])
    # plt.legend(["No Control", "Alpha Control"])
    # plt.show()

    opt_range = GlideOpt()

    print(f"Max Range: {opt_range.sim.max_r} m, at alpha_0 = {opt_range.alpha_coeff_0}, Rn = {opt_range.Rn}, phi = {np.degrees(opt_range.phi)}, L = {opt_range.L}")

    ax = flight_path_plot([opt_range.sim])
    # ax.text(
    #     *(0.05, 0.92),
    #     f"max range: {opt_range.sim.max_r} ($\\alpha_0={opt_range.alpha_coeff_0}, alpha_1={opt_range.alpha_coeff_1}$",#, alpha_2={opt_range.alpha_coeff_2}$",
    #     transform=ax.transAxes,
    # )
    plt.legend(["Optimized Glide Trajectory"])
    plt.show()
    
