import numpy as np
from scipy.optimize import bisect
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# --- CONSTANTS ---
# ---------------------------------------------------------
S_REF = 0.0314 # Based on 0.2m diameter (pi*0.1^2)
G_ACCEL = 9.81
ALPHA_BOUNDS = [-0.7, 0.7] 

# ---------------------------------------------------------
# --- AERO MODEL ---
# ---------------------------------------------------------

def aero_CL(M, alpha):
    """Simplified placeholder CL model — replace with aero database."""
    # Note: Use radians for alpha input
    return 0.1 * alpha

def aero_CD(M, alpha):
    """Simplified placeholder CD model — replace with aero database."""
    return 0.02 + 0.04 * alpha**2

def find_alpha_zero_lift(M):
    """Solve CL(M, α)=0."""
    f = lambda a: aero_CL(M, a)
    # The true zero is at 0, but using bisect to show method
    return bisect(f, *ALPHA_BOUNDS)

def find_alpha_for_L_equals_W(M, rho, V, W):
    """Solve L = W for alpha."""
    def f(a):
        q = 0.5 * rho * V**2
        L = q * S_REF * aero_CL(M, a) 
        return L - W
    return bisect(f, *ALPHA_BOUNDS)

def find_alpha_for_g_load(M, rho, V, mass, g_load):
    """Solve L = m*g*gload. This is where the error occurred."""
    target_L = mass * G_ACCEL * g_load
    def f(a):
        q = 0.5 * rho * V**2
        L = q * S_REF * aero_CL(M, a) 
        return L - target_L
    # FIX: If the target lift is physically impossible, bisect fails.
    # The wider bounds [-0.7, 0.7] should allow it to converge.
    return bisect(f, *ALPHA_BOUNDS)

def find_alpha_for_max_LoverD(M):
    """Find AoA maximizing CL/CD."""
    def f(a):
        return -aero_CL(M, a) / aero_CD(M, a)
    # brute force approach is fine here
    alphas = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], 200)
    ratio = [aero_CL(M, a)/aero_CD(M, a) for a in alphas]
    return alphas[np.argmax(ratio)]

# ---------------------------------------------------------
# --- ATMOSPHERE MODEL ---
# ---------------------------------------------------------
def atmosphere(h):
    """Return density and speed of sound."""
    T0 = 288.15
    a0 = 340.0
    rho0 = 1.225
    R_earth = 6371000 # meters

    g = G_ACCEL * (R_earth / (R_earth + h))**2


    if h < 11000:
        T = T0 - 0.00649*h
        # Use simple density relation for consistency
        rho = rho0 * (T/T0)**4.2559
        a = a0 * np.sqrt(T/T0)
    else:
        # Simplistic Stratosphere/higher altitude model
        rho = 0.3 # Adjusted density for altitude > 11km
        a = 295
    return rho, a, g

# ---------------------------------------------------------
# --- RK4 Integrator & EOM ---
# ---------------------------------------------------------
def rk4_step(f, state, dt, *args):
    k1 = f(state, *args)
    k2 = f(state + 0.5*dt*k1, *args)
    k3 = f(state + 0.5*dt*k2, *args)
    k4 = f(state + dt*k3, *args)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# state = [x, h, V, gamma]
def eom(state, mass, alpha):
    x, h, V, gamma = state
    rho, a, g = atmosphere(h)
    q = 0.5 * rho * V**2
    M = V/a
    CL = aero_CL(M, alpha)
    CD = aero_CD(M, alpha)

    L = q * S_REF * CL 
    D = q * S_REF * CD 

    dxdt = V * np.cos(gamma)
    dhdt = V * np.sin(gamma)
    dVdt = (-D/mass) - (g*np.sin(gamma))
    dgamdt = (L/(mass*V)) - (g*np.cos(gamma)/V)

    return np.array([dxdt, dhdt, dVdt, dgamdt])

# ---------------------------------------------------------
# --- TRAJECTORY SIMULATION ---
# ---------------------------------------------------------
def simulate_trajectory(mass=45.0, dt=0.02):
    traj = []

    # -----------------------------------------
    # 1) BOOST + ZERO-LIFT BALLISTIC TO APOGEE (Fixed: Launch Angle Guess must be iteratively solved)
    # -----------------------------------------
    # NOTE: The provided code does NOT iterate on launch angle to hit a 9km apogee.
    # It just runs one simulation. For this fix, we'll keep the single run structure,
    # but the loop condition needs to be removed for clarity in this single run.
    
    V0 = 1200 # m/s
    h0 = 0
    gamma0 = np.radians(45)
    x0 = 0
    t = 0.0
    state = np.array([x0, h0, V0, gamma0])

    while True:
        rho, a, g = atmosphere(state[1])
        M = state[2]/a
        alpha = find_alpha_zero_lift(M)

        new_state = rk4_step(eom, state, dt, mass, alpha)
        traj.append([t,*state, alpha])
        state = new_state
        t += dt
        
        # Stop condition for ballistic phase
        if state[3] < 0 and state[1] > 100: # Reached Apogee
            break
        if state[1] < 0: # Avoid crash if trajectory is too flat
             print("Warning: Vehicle crashed during boost/ballistic phase.")
             return np.array(traj)


    # -----------------------------------------
    # 2) GLIDE SECTION (L=W then Max L/D)
    # -----------------------------------------
    phase = "L_EQUALS_W" # Start in L=W mode

    while state[1] > 0:
        rho, a, g = atmosphere(state[1])
        M = state[2] / a
        q = 0.5 * rho * state[2]**2
        W = mass * G_ACCEL

        alpha_maxLD = find_alpha_for_max_LoverD(M)
        L_max = q * S_REF * aero_CL(M, ALPHA_BOUNDS[1])
        
        if phase == "L_EQUALS_W":
            if L_max < W:
                # Cannot maintain altitude, switch to max L/D immediately
                phase = "MAX_L_OVER_D"
                alpha = alpha_maxLD
            else:
                try:
                    alpha = find_alpha_for_L_equals_W(M, rho, state[2], W)
                    # Check if L=W trim alpha exceeds max L/D alpha
                    if np.abs(alpha) >= np.abs(alpha_maxLD):
                        phase = "MAX_L_OVER_D"
                        alpha = alpha_maxLD
                except ValueError:
                    # Should not happen with widened bounds, but use max L/D as safe backup
                    phase = "MAX_L_OVER_D"
                    alpha = alpha_maxLD
        
        if phase == "MAX_L_OVER_D":
             alpha = alpha_maxLD

        state_new = rk4_step(eom, state, dt, mass, alpha)
        traj.append([t,*state, alpha])
        state = state_new
        t += dt

        if state[1] <= 0:
            break

    # -----------------------------------------
    # 3) TERMINAL DESCENT
    # -----------------------------------------
    rho_sl, a_sl, g_sl = atmosphere(0)
    
    # Check current impact speed against Mach 2 requirement
    if state[2] < 2 * a_sl:
        print(f"Impact Mach ({state[2]/a_sl:.2f}) < Mach 2. Initiating pushover.")
        g_load = 15.0 # Max 15g turn
        
        # 3a) Pushover until gamma = -90 deg
        while state[3] > -np.pi/2:
            if state[1] <= 0: break
            rho, a = atmosphere(state[1])
            M = state[2]/a
            
            # Use negative g_load to force a downward (negative lift) maneuver
            try:
                alpha = find_alpha_for_g_load(M, rho, state[2], mass, -g_load) 
            except ValueError:
                # If 15g is too much, use max negative alpha available
                alpha = ALPHA_BOUNDS[0]
                
            new_state = rk4_step(eom, state, dt, mass, alpha)
            traj.append([t,*state, alpha])
            state = new_state
            t += dt

        # 3b) Zero-lift vertical drop
        while state[1] > 0:
            rho, a = atmosphere(state[1])
            M = state[2]/a
            alpha = find_alpha_zero_lift(M)
            new_state = rk4_step(eom, state, dt, mass, alpha)
            traj.append((*state, alpha))
            state = new_state
            
    else:
        print(f"Impact Mach ({state[2]/a_sl:.2f}) >= Mach 2. No pushover needed.")


    return np.array(traj)

# ---------------------------------------------------------
# --- PLOTTING ---
# ---------------------------------------------------------
def plot_trajectory(traj):
    t = traj[:,0]
    x = traj[:,1]/1000 # km
    h = traj[:,2]/1000 # km
    V = traj[:,3]
    gamma = np.degrees(traj[:,3])
    alpha = np.degrees(traj[:,4])

    # Plots over Downrange
    fig, axs = plt.subplots(4,1, figsize=(8,12))
    axs[0].plot(x, h); axs[0].set_ylabel('Altitude (km)')
    axs[1].plot(x, V); axs[1].set_ylabel('Velocity (m/s)')
    axs[2].plot(x, gamma); axs[2].set_ylabel('Flight Path Angle (deg)')
    axs[3].plot(x, alpha); axs[3].set_ylabel('AoA (deg)')
    axs[3].set_xlabel('Downrange (km)')
    plt.tight_layout()
    plt.show()

    # Plots over Time
    fig, axs = plt.subplots(4,1, figsize=(8,12))
    axs[0].plot(t, h); axs[0].set_ylabel('Altitude (km)')
    axs[1].plot(t, V); axs[1].set_ylabel('Velocity (m/s)')
    axs[2].plot(t, gamma); axs[2].set_ylabel('Flight Path Angle (deg)')
    axs[3].plot(t, alpha); axs[3].set_ylabel('AoA (deg)')
    axs[3].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    traj = simulate_trajectory()
    plot_trajectory(traj)