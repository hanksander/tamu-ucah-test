import condor as co
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from condor.backend import operators as ops

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