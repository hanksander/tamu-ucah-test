import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops

#from aerodynamics import *
#from thermal import *

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

    CL_alpha = parameter()
    CD_0 = parameter()
    CD_i_q = parameter()
    g = parameter()

    CL = 
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

    cl = variable(
        initializer=0.11,
        lower_bound=0.01,
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

    sim = AlphaSim(CL_alpha=cl, CD_0=0.05 * A, CD_i_q=0.05, g=1.0, 
                   alpha_coeff_0=alpha_coeff_0, alpha_coeff_1=alpha_coeff_1, alpha_coeff_2=0.)#alpha_coeff_2)
    
    objective = sim.max_r

    constrain(sim.max_h <= 110.0)
    constrain(sim.v >= 20.0)


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
    
