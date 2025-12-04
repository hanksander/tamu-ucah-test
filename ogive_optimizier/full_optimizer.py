import random
import math
import time
import copy

from waverider_generator.generator import waverider as wr
from waverider_generator.plotting_tools import Plot_Base_Plane,Plot_Leading_Edge
import matplotlib.pyplot as plt
from waverider_generator.cad_export import to_CAD

import os as os
import numpy as np
import scipy as sp
import trimesh as tm
from fit_optimizer_3 import *

from integrated_traj_test import run_dymos_optimization

from manual_mesh_main import output_waverider_mesh

# --- your functions assumed available ---
def set_up_waverider(params):
    w = wr.Waverider()
    w.set_parameters(
        X1=params['X1'], X2=params['X2'], X3=params['X3'], X4=params['X4'],
        M=params['M'], length=params['length'],
        width=params['width'], height=params['height']
    )

    w.generate_geometry()

    waverider_cad=to_CAD(waverider=waverider,sides='both',export=True,
                         filename='./waverider.stl',scale=1)
    return w

def check_fitting(filename):
    inst = WaveriderCase(rf'{filename}')
    return inst.try_fit_payload(verbose=True)

def compute_aerodatabase(waverider, filename):
    output_waverider_mesh(waverider, filename)

calculate_waverider_range(waverider):
    path = os.getcwd()
    max_range, max_q_dot = run_dymos_optimization(path, plotting=False)
    return max_range, max_q_dot

class Particle:
    def __init__(self, dim, bounds, vel_bounds):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [random.uniform(vb[0], vb[1]) for vb in vel_bounds]
        self.best_pos = self.position[:]
        self.best_cost = float('inf')

def enforce_bounds(x, bounds):
    return [max(b[0] + 1e-12, min(b[1] - 1e-12, xi)) for xi, b in zip(x, bounds)]

def clip_velocity(v, v_bounds):
    return [max(vb[0], min(vb[1], vi)) for vi, vb in zip(v, v_bounds)]

def inequality_satisfied(X1, X2, width, height):
    eps = 1e-12
    if height <= eps or (1 - X1) <= eps:
        return False
    lhs = X2 / ((1 - X1) ** 4)
    rhs = (7.0 / 64.0) * ((width / height) ** 4)
    return lhs <= rhs

def evaluate_particle(pos, penalty_coeff=1e6, verbose=False):
    X1, X2, X3, X4, M, length, width, height = pos

    # ---- PRE-CHECK inequality ----
    if not inequality_satisfied(X1, X2, width, height):
        # return immediate heavy penalty; don't attempt setup
        return 1e9

    params = {
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
        'M': M, 'length': length, 'width': width, 'height': height
    }

    os.mkdir('temp_waverider_cases') if not os.path.exists('temp_waverider_cases') else None
    os.chdir('temp_waverider_cases')
    os.mkdir(f'waverider-{X1}-{X2}-{X3}-{X4}-{M}-{length}-{width}-{height}') if not os.path.exists(f'waverider-{X1}-{X2}-{X3}-{X4}-{M}-{length}-{width}-{height}') else None
    os.chdir(f'waverider-{X1}-{X2}-{X3}-{X4}-{M}-{length}-{width}-{height}')

    # ---- Try waverider setup ----
    try:
        w = set_up_waverider(params)
    except Exception as e:
        if verbose:
            print("set_up_waverider failed:", e)
        return 1e9

    # ---- Check fitting ----
    try:
        fits = check_fitting(w)
    except Exception as e:
        if verbose:
            print("check_fitting failed:", e)
        return 1e9


    # ---- Compute Aero Database ----
    try:
        compute_aerodatabase()
    except Exception as e:
        if verbose:
            print("compute_aerodatabase failed:", e)
            exit()


    # ---- Compute range ----
    try:
        rng, max_q_dot = calculate_waverider_range(w)
        if not (isinstance(rng, (int, float)) and math.isfinite(rng)):
            return 1e9
    except Exception as e:
        if verbose:
            print("calculate_waverider_range failed:", e)
        return 1e9

    # maximize range -> minimize cost
    cost = cost(range, max_q_dot)
    return cost

def pso_optimize(num_particles=40, iterations=200, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)

    # [X1, X2, X3, X4, M, length, width, height]
    dim = 8
    M_max = 10.0

    bounds = [
        (1e-8, 1.0 - 1e-8),  # X1
        (1e-8, 1.0 - 1e-8),  # X2
        (1e-8, 1.0 - 1e-8),  # X3
        (1e-8, 1.0 - 1e-8),  # X4
        (1e-8, M_max),        # M
        (1e-8, 1.0 - 1e-8),   # length
        (1e-8, 0.1 - 1e-8),   # width
        (1e-8, 0.18 - 1e-8)    # height
    ]

    vel_bounds = [(-(hi - lo) * 0.5, (hi - lo) * 0.5) for lo, hi in bounds]

    swarm = [Particle(dim, bounds, vel_bounds) for _ in range(num_particles)]

    gbest_pos = None
    gbest_cost = float('inf')

    # Evaluate initial swarm
    for p in swarm:
        p.position = enforce_bounds(p.position, bounds)
        cost = evaluate_particle(p.position, verbose=verbose)
        p.best_cost = cost
        p.best_pos = p.position[:]
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_pos = p.position[:]

    history = []
    for it in range(iterations):
        for p in swarm:
            # velocity update
            for i in range(dim):
                r1, r2 = random.random(), random.random()
                c1, c2, w_inertia = 1.49445, 1.49445, 0.729
                cognitive = c1 * r1 * (p.best_pos[i] - p.position[i])
                social = c2 * r2 * (gbest_pos[i] - p.position[i])
                p.velocity[i] = w_inertia * p.velocity[i] + cognitive + social
            p.velocity = clip_velocity(p.velocity, vel_bounds)
            p.position = enforce_bounds([pi + vi for pi, vi in zip(p.position, p.velocity)], bounds)

            cost = evaluate_particle(p.position, verbose=verbose)
            if cost < p.best_cost:
                p.best_cost = cost
                p.best_pos = p.position[:]
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = p.position[:]

        history.append(gbest_cost)
        if verbose and it % max(1, iterations // 10) == 0:
            print(f"Iter {it}/{iterations}: best range = {-gbest_cost:.6g}")

    best = {
        'X1': gbest_pos[0], 'X2': gbest_pos[1], 'X3': gbest_pos[2], 'X4': gbest_pos[3],
        'M': gbest_pos[4], 'length': gbest_pos[5],
        'width': gbest_pos[6], 'height': gbest_pos[7]
    }
    return best, -gbest_cost, history


if __name__ == "__main__":
    best, best_range, hist = pso_optimize(num_particles=20, iterations=50, seed=42, verbose=True)
    print("\nBest design:")
    for k, v in best.items():
        print(f"  {k:8s} = {v:.6f}")
    print(f"Best predicted range = {best_range:.6f}")
