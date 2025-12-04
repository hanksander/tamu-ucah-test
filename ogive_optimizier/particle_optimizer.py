import random
import math
import time
import copy
import os
import subprocess
import shutil
import numpy as np
import scipy as sp
import trimesh as tm

from waverider_generator.generator import waverider as wr
from waverider_generator.plotting_tools import Plot_Base_Plane, Plot_Leading_Edge
import matplotlib.pyplot as plt
from waverider_generator.cad_export import to_CAD

from fit_optimizer_3 import *
from integrated_traj_test import run_dymos_optimization
from manual_mesh_main import output_waverider_mesh

from parameter_solver import compute_reference_parameters

# --- Configuration ---
OPTIMIZATION_ROOT = os.getcwd()  # Root optimization directory
CBAERO_SCRIPT = os.path.join(OPTIMIZATION_ROOT, "run_cbaero.sh")  # Path to shell script
PATH_TO_BINS = "/root/401/CBaero/bin"  # Update if needed

# Setup virtual display for GUI applications if DISPLAY not set
USE_XVFB = os.environ.get('DISPLAY') is None
if USE_XVFB:
    try:
        import subprocess
        # Start Xvfb virtual display
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ['DISPLAY'] = ':99'
        print("Started virtual display (Xvfb) on :99")
    except Exception as e:
        print(f"Warning: Could not start Xvfb: {e}")
        print("GUI applications may not work properly")

# --- Helper functions ---
def set_up_waverider(params, case_dir):

    print(params)
    """Generate waverider geometry and export to STL."""
    dp = [params['X1'], params['X2'], params['X3'], params['X4']]

    for i, val in enumerate(dp):
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"dp[{i}] = {val} out of bounds (0, 1)")
        if (val<1e-5):
            dp[i]=0

    print(dp)
    beta = np.rad2deg(np.atan(params['height']/params['length']))
    print(beta)

    waverider=wr(M_inf=params['M'], 
            beta=beta,
            height=params['height'],
            width=params['width'],
            dp=dp,
            n_upper_surface=1000,
            n_shockwave=1000,
            n_planes=20,
            n_streamwise=10,
            delta_streamise=0.02)

    # print("[*] Plotting waverider geometry...")
    # base_plane=Plot_Base_Plane(waverider=waverider,latex=False)
    # leading_edge=Plot_Leading_Edge(waverider=waverider,latex=False)
    # plt.show()

    # Export CAD to the case directory
    stl_path = os.path.join(case_dir, 'waverider.stl')

    if os.path.exists("./waverider.stl"):
        os.remove("./waverider.stl")

    print("[*] Exporting CAD...")

    waverider_cad = to_CAD(waverider=waverider, sides='both', export=True,
                          filename="waverider.stl", scale=1)
    

    os.system(f"mv ./waverider.stl {stl_path}")

    print(f"Exported waverider STL to: {stl_path}")
    return waverider

def check_fitting(filename, case_dir):
    stl_path = os.path.join(case_dir, 'waverider.stl')
    inst = WaveriderCase(stl_path)
    fitting =  inst.try_fit_payload(verbose=True)

    print(f"FITTING: {fitting}")

    return fitting

def compute_aerodatabase(params, case_dir, filename='waverider'):
    """
    Generate mesh and run CBAERO simulation.
    
    Directory structure:
    case_dir/
        waverider.stl
        waverider/  <- CBAERO run directory
            waverider.tri
            waverider.msh
            waverider.cbaero
            ... (CBAERO output files)
    """
    # Create CBAERO subdirectory
    cbaero_dir = os.path.join(case_dir, filename)
    os.makedirs(cbaero_dir, exist_ok=True)

    dp = [params['X1'], params['X2'], params['X3'], params['X4']]

    for i, val in enumerate(dp):
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"dp[{i}] = {val} out of bounds (0, 1)")
        if (val<1e-7):
            dp[i]=0

    print(dp)
    beta = np.rad2deg(np.atan(params['height']/params['length']))
    print(beta)

    print("[*] Generating fine waverider geometry...")

    waverider=wr(M_inf=params['M'], 
            beta=beta,
            height=params['height'],
            width=params['width'],
            dp=dp,
            n_upper_surface=1000,
            n_shockwave=1000,
            n_planes=75,
            n_streamwise=50,
            delta_streamise=0.02)
    
    # Generate triangulated mesh from waverider geometry
    tri_path = os.path.join(cbaero_dir, f'{filename}.tri')
    vertices, triangles, stats = output_waverider_mesh(waverider, tri_path)

    print(f"Computing parameters")
    geom_params = compute_reference_parameters(vertices, triangles)
    print(f"  Sref = {geom_params['Sref']}, cref = {geom_params['cref']}, bref = {geom_params['bref']}")
    sref = geom_params['Sref']
    cref = geom_params['cref']
    bref = geom_params['bref']
    
    # Change to CBAERO directory to run the script
    original_dir = os.getcwd()
    os.chdir(cbaero_dir)
    
    try:
    # Run the CBAERO shell script
        #cmd = ['bash', CBAERO_SCRIPT, filename, str(sref), str(cref), str(bref)]
        # Install: apt-get install expect (includes unbuffer)
        cmd = ['unbuffer', 'bash', CBAERO_SCRIPT, filename, str(sref), str(cref), str(bref)]
        
        # CRITICAL: expect needs a TTY to interact with the GUI
        # Use Popen instead of run() to give it proper terminal access
        log_file = os.path.join(cbaero_dir, 'cbaero_run.log')
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,  # Provide stdin for expect
                text=True,
                env=os.environ.copy()  # Inherit environment including TERM
            )
            
            # Wait for completion with timeout
            try:
                returncode = process.wait(timeout=600)
                
                if returncode != 0:
                    # Read log file for error details
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    raise subprocess.CalledProcessError(returncode, cmd, output=log_content)
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise TimeoutError("CBAERO script exceeded 10 minute timeout")
        
        # Print log for debugging
        with open(log_file, 'r') as f:
            log_content = f.read()
            print(f"CBAERO output:\n{log_content}")
        
        # Verify output files were created
        required_files = [
            f'{filename}.msh',
            f'{filename}.cbaero',
        ]
        
        for req_file in required_files:
            if not os.path.exists(req_file):
                with open(log_file, 'r') as f:
                    error_log = f.read()
                raise FileNotFoundError(
                    f"Expected output file not found: {req_file}\n"
                    f"Script output:\n{error_log}"
                )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"CBAERO script failed with error:\n{e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print("CBAERO script timed out")
        raise
    finally:
        os.chdir(original_dir)


def calculate_waverider_range(waverider, case_dir):
    """Calculate waverider range using trajectory optimization."""
    original_dir = os.getcwd()
    os.chdir(case_dir)
    
    try:
        max_range, max_q_dot = run_dymos_optimization(case_dir, plotting=True)
        return max_range, max_q_dot
    finally:
        os.chdir(original_dir)

def compute_cost(rng, max_q_dot):
    """
    Compute cost function from range and heating rate.
    Modify this based on your optimization objectives.
    """
    # Example: maximize range, penalize high heating
    # Adjust weights as needed
    return -rng + 0.01 * max_q_dot  # Negative range for minimization

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
    """
    Evaluate a particle's fitness.
    
    Directory structure created:
    optimization/
        temp_waverider_cases/
            waverider-X1-X2-X3-X4-M-length-width-height/
                waverider.stl
                waverider/
                    waverider.tri
                    waverider.msh
                    waverider.cbaero
                    ... (CBAERO outputs)
    """
    X1, X2, X3, X4, M, length, width, height = pos

    # ---- PRE-CHECK inequality ----
    if not inequality_satisfied(X1, X2, width, height):
        if verbose:
            print(f"Inequality constraint violated for particle {pos[:4]}")
        return 1e9

    params = {
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
        'M': M, 'length': length, 'width': width, 'height': height
    }

    # Create case directory structure
    temp_dir = os.path.join(OPTIMIZATION_ROOT, 'temp_waverider_cases')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Use shorter directory name to avoid filesystem limits
    case_name = f'waverider-{X1:.4f}-{X2:.4f}-{X3:.4f}-{X4:.4f}-{M:.2f}-{length:.4f}-{width:.4f}-{height:.4f}'
    case_dir = os.path.join(temp_dir, case_name)
    
    # If case already exists (retry), clean it
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    
    os.makedirs(case_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating particle in: {case_name}")
        print(f"{'='*60}")

    # ---- Try waverider setup ----
    try:
        if verbose:
            print("[1/4] Setting up waverider geometry...")
        w = set_up_waverider(params, case_dir)
    except Exception as e:
        if verbose:
            print(f"❌ set_up_waverider failed: {e}")
        return 1e9

    # ---- Check fitting ----
    try:
        if verbose:
            print("[2/4] Checking payload fitting...")
        fitting = check_fitting(w, case_dir)

        fits = fitting[0]
        res = fitting[1]

        if not fits:
            if verbose:
                print("❌ Payload does not fit")
            return res*1e9
    except Exception as e:
        if verbose:
            print(f"❌ check_fitting failed: {e}")
        return 1e9

    # ---- Compute Aero Database ----
    try:
        if verbose:
            print("[3/4] Computing aerodynamic database (running CBAERO)...")
        compute_aerodatabase(params, case_dir, filename='waverider')
    except Exception as e:
        if verbose:
            print(f"❌ compute_aerodatabase failed: {e}")
        # Don't exit, just return penalty
        return 1e9

    # ---- Compute range ----
    try:
        if verbose:
            print("[4/4] Computing trajectory and range...")
        rng, max_q_dot = calculate_waverider_range(w, case_dir)
        if not (isinstance(rng, (int, float)) and math.isfinite(rng)):
            if verbose:
                print(f"❌ Invalid range value: {rng}")
            return 1e9
    except Exception as e:
        if verbose:
            print(f"❌ calculate_waverider_range failed: {e}")
        return 1e9

    # Calculate cost (maximize range -> minimize cost)
    cost = compute_cost(rng, max_q_dot)
    
    if verbose:
        print(f"✓ Evaluation complete: range={rng:.4f}, q_dot={max_q_dot:.4f}, cost={cost:.4f}")
    
    return cost

def pso_optimize(num_particles=40, iterations=200, seed=None, verbose=False, 
                 initial_design=None, perturbation=0.1):
    """
    Run Particle Swarm Optimization.
    
    Args:
        initial_design: Dict with keys [X1, X2, X3, X4, M, length, width, height]
                       If provided, particles will be initialized around this design
        perturbation: Fraction of parameter range to perturb initial design (default 0.1)
    """
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
        (5., M_max),        # M
        (0.5, 1.0 - 1e-8),   # length
        (0.09, 0.1 - 1e-8),   # width
        (0.15, 0.2 - 1e-8)    # height
    ]

    vel_bounds = [(-(hi - lo) * 0.5, (hi - lo) * 0.5) for lo, hi in bounds]

    swarm = [Particle(dim, bounds, vel_bounds) for _ in range(num_particles)]

    # Initialize particles around initial design if provided
    if initial_design is not None:
        print(f"\nInitializing swarm around provided design:")
        print(f"  X1={initial_design['X1']:.6f}, X2={initial_design['X2']:.6f}, "
              f"X3={initial_design['X3']:.6f}, X4={initial_design['X4']:.6f}")
        print(f"  M={initial_design['M']:.2f}, length={initial_design['length']:.6f}, "
              f"width={initial_design['width']:.6f}, height={initial_design['height']:.6f}")
        print(f"  Perturbation: ±{perturbation*100:.1f}% of bounds\n")
        
        base_position = [
            initial_design['X1'],
            initial_design['X2'],
            initial_design['X3'],
            initial_design['X4'],
            initial_design['M'],
            initial_design['length'],
            initial_design['width'],
            initial_design['height']
        ]
        
        # First particle is exactly the initial design
        swarm[0].position = base_position[:]
        
        # Other particles are perturbed versions
        for i, p in enumerate(swarm[1:], 1):
            p.position = []
            for j, (val, (lo, hi)) in enumerate(zip(base_position, bounds)):
                param_range = hi - lo
                noise = random.uniform(-perturbation * param_range, perturbation * param_range)
                perturbed = val + noise
                p.position.append(perturbed)
            p.position = enforce_bounds(p.position, bounds)

    gbest_pos = None
    gbest_cost = float('inf')

    print(f"\n{'='*60}")
    print(f"Starting PSO Optimization")
    print(f"Particles: {num_particles}, Iterations: {iterations}")
    print(f"{'='*60}\n")

    # Evaluate initial swarm
    print("Initializing swarm...")
    for i, p in enumerate(swarm):
        p.position = enforce_bounds(p.position, bounds)
        cost = evaluate_particle(p.position, verbose=verbose)
        p.best_cost = cost
        p.best_pos = p.position[:]
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_pos = p.position[:]
        
        status = "✓ [INITIAL DESIGN]" if i == 0 and initial_design else ""
        print(f"  Initial particle {i+1}/{num_particles}: cost={cost:.6g} {status}")

    history = []
    for it in range(iterations):
        print(f"\n--- Iteration {it+1}/{iterations} ---")
        
        for p in swarm:
            # Velocity update
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
        print(f"Iteration {it+1} complete: best cost = {gbest_cost:.6g}")

    best = {
        'X1': gbest_pos[0], 'X2': gbest_pos[1], 'X3': gbest_pos[2], 'X4': gbest_pos[3],
        'M': gbest_pos[4], 'length': gbest_pos[5],
        'width': gbest_pos[6], 'height': gbest_pos[7]
    }
    return best, -gbest_cost, history


if __name__ == "__main__":
    # Verify shell script exists
    if not os.path.exists(CBAERO_SCRIPT):
        print(f"ERROR: CBAERO shell script not found at: {CBAERO_SCRIPT}")
        print("Please update the CBAERO_SCRIPT path in the configuration section.")
        exit(1)
    
    # Make shell script executable
    os.chmod(CBAERO_SCRIPT, 0o755)
    
    # === TESTING MODE: Initialize around your specific waverider ===
    M_inf = 6.39
    height = 0.1894
    length = 0.5
    width = 0.1
    
    # dp = [X1, X2, X3, X4] from your design parameters
    dp = [0.8235, 0.0, 0.1147, 0.0]
    
    print(f"Test waverider parameters:")
    print(f"  M_inf = {M_inf}")
    print(f"  height = {height}")
    print(f"  length = {length:.6f}")
    print(f"  width = {width}")
    print(f"  dp = {dp}")
    
    initial_design = {
        'X1': dp[0],
        'X2': dp[1],
        'X3': dp[2],
        'X4': dp[3],
        'M': M_inf,
        'length': length,
        'width': width,
        'height': height
    }
    
    # Run optimization starting from this design
    # perturbation=0.05 means particles vary by ±5% of parameter bounds
    # Set perturbation=0.0 to test ONLY the exact design (no variation)
    best, best_range, hist = pso_optimize(
        num_particles=4,      # Fewer particles for testing
        iterations=15,          # Fewer iterations for testing
        seed=42, 
        verbose=True,
        initial_design=initial_design,
        perturbation=0.1      # ±5% variation, or 0.0 for exact design only
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("\nBest design found:")
    for k, v in best.items():
        print(f"  {k:8s} = {v:.6f}")
    print(f"\nBest predicted range = {best_range:.6f}")
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.plot(hist)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('PSO Convergence History')
    plt.grid(True)
    plt.savefig('pso_convergence.png', dpi=150, bbox_inches='tight')
    print("\nConvergence plot saved to: pso_convergence.png")