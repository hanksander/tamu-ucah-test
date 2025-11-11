import openmdao.api as om
import dymos as dm
import numpy as np
import math
import matplotlib.pyplot as plt

# --- 1. Define the Physics Component (Equations of Motion) ---
class RocketBoostEOM(om.ExplicitComponent):
    """
    Computes the differential equations for a 2D atmospheric rocket boost with lift and AOA as control variables
    """
    def initialize(self):
        """
        Define constant parameters for the environment
        """
        self.options.declare('num_nodes', types=int)
        self.R_E = 6378137.0  # Earth radius (m)
        self.g0 = 9.80665    # Gravity at sea level (m/s^2)
        
        # Atmospheric constants (Simplified Isothermal Model)
        self.rho0 = 1.225     # Density at sea level (kg/m^3)
        self.H = 7500.0       # Scale height (m)
        self.T0 = 288.15      # Temperature at sea level (K) - used for speed of sound
        self.R_gas = 287.05   # Specific Gas Constant (J/(kg*K))
        self.gamma_atm = 1.4  # Ratio of specific heats 

    def setup(self):
        """
        Define Inputs and Outputs
        """
        nn = self.options['num_nodes']

        # Inputs (States, Controls, Parameters)
        self.add_input('h', val=np.zeros(nn), units='m', desc='Altitude')
        self.add_input('initV', val=np.zeros(nn), units='m/s', desc=' Initial Velocity')
        self.add_input('gamma', val=np.zeros(nn), units='rad', desc='Flight path angle')
        self.add_input('m', val=np.zeros(nn), units='kg', desc='Vehicle Mass')
        
        # Control Input: Angle of Attack (alpha)
        self.add_input('alpha', val=np.zeros(nn), units='rad', desc='Angle of attack')
        
        # Parameters (defining the vehicle/engine/shape characteristics)
        self.add_input('T', val=4250.0, units='N', desc='Engine Thrust (constant)')
        self.add_input('Isp', val=250.0, units='s', desc='Specific Impulse')
        self.add_input('S', val=0.0124, units='m**2', desc='Reference Area (fixed to 0.0314 m^2)')
        self.add_input('payload_mass', val=9.0, units='kg', desc='Payload mass')
        
        # Shape Parameters (simplified)
        self.add_input('CD_0', val=0.1, desc='Zero-lift drag coefficient (related to nose/friction)')
        # Take this shit outtttt
        self.add_input('CL_alpha', val=0.1, desc='Lift coefficient slope (how much lift per rad of alpha)')

        # Outputs (State Derivatives and intermediate values)
        self.add_output('v', val = np.zeros(nn), units = 'm/s', desc = 'velocity over time')
        self.add_output('h_dot', val=np.zeros(nn), units='m/s', desc='dh/dt: Rate of change of altitude')
        self.add_output('V_dot', val=np.zeros(nn), units='m/s**2', desc='dV/dt: Acceleration')
        self.add_output('gamma_dot', val=np.zeros(nn), units='rad/s', desc='dgamma/dt: Rate of change of flight path angle')
        self.add_output('m_dot', val=np.zeros(nn), units='kg/s', desc='dm/dt: Mass flow rate')
        self.add_output('x_dot', val=np.zeros(nn), units='m/s', desc='dx/dt: Rate of change of horizontal range')
        
        self.add_output('Drag', val=np.zeros(nn), units='N', desc='Total Drag Force')
        self.add_output('Lift', val=np.zeros(nn), units='N', desc='Total Lift Force')
        self.add_output('Mach', val=np.zeros(nn), desc='Mach Number')
        self.add_output('g_load', val=np.zeros(nn), units=None, desc='Load factor (L / (m*g0))')


        # Partials setup
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        """
        Executes physics calculations for every point in the trajectory
        """
        h = inputs['h']
        V = inputs['V']
        gamma = inputs['gamma']
        m = inputs['m']
        alpha = inputs['alpha']
        T = inputs['T']
        S = inputs['S'] 
        CD_0 = inputs['CD_0']
        CL_alpha = inputs['CL_alpha']
        Isp = inputs['Isp']
        g0 = self.g0
        payload = inputs['payload_mass']

        # --- Sub-Models ---

        # 1. Local Gravity & Radius
        r = self.R_E + h
        g = g0 * (self.R_E / r)**2

        # 2. Atmosphere (Density and Speed of Sound)
        Temp = (15.04 - 0.00649 * h) + 273.1
        pressure = 101.29 * (Temp/288.08)**5.256
        rho = self.rho0 * np.exp(-h / self.H)
        a = np.sqrt(self.gamma_atm * self.R_gas * Temp) 
        outputs['Mach'] = V / a

        # 3. Aerodynamics (Shape-dependent forces)
        q = 0.5 * rho * V**2 # Dynamic Pressure
        
        # will get values from aakash (the goat)
        CL = CL_alpha * alpha
        CD = CD_0 + 0.1 * CL**2 # Simplified parabolic drag polar

        Drag = q * S * CD
        Lift = q * S * CL
        outputs['Drag'] = Drag
        outputs['Lift'] = Lift

        # 4. Mass flow rate
        m_dot = -T / (Isp * g0)
        outputs['m_dot'][:] = m_dot 

        # 5. G-Load Calculation
        outputs['g_load'] = Lift / (m * g0)

        # --- Equations of Motion (EOMs) ---
        outputs['h_dot'] = V * np.sin(gamma)
        outputs['V_dot'] = (T * np.cos(alpha) - Drag) / m - g * np.sin(gamma)
        outputs['gamma_dot'] = (T * np.sin(alpha) + Lift) / (m * V) + (V / r - g / V) * np.cos(gamma)
        outputs['x_dot'] = V * np.cos(gamma) # Horizontal range rate
        # outputs['v'] = 


# --- 2. Define the Plotting Function ---
def plot_results(sim_out):
    """
    Generates plots from the Dymos simulation timeseries output.
    """
    time = sim_out.get_val('time')
    h = sim_out.get_val('h')
    x = sim_out.get_val('x')
    V = sim_out.get_val('V')
    Mach = sim_out.get_val('Mach')
    g_load = sim_out.get_val('g_load')
    alpha_deg = np.rad2deg(sim_out.get_val('alpha'))

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.suptitle('Rocket Boost Trajectory Optimization Results', fontsize=16)

    # 1. Altitude vs. Time
    ax = axes[0, 0]
    ax.plot(time, h / 1000.0)
    ax.set_title('Altitude vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)

    # 2. Altitude vs. Range (Trajectory)
    ax = axes[0, 1]
    ax.plot(x / 1000.0, h / 1000.0)
    ax.set_title('Trajectory (Altitude vs. Range)')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)

    # 3. Mach vs. Time
    ax = axes[1, 0]
    ax.plot(time, Mach)
    ax.set_title('Mach Number vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach')
    ax.grid(True)

    # 4. G-Load vs. Time
    ax = axes[1, 1]
    ax.plot(time, g_load)
    ax.set_title('G-Load vs. Time (Maximized by Aerodynamics)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('G-Load (L/W)')
    ax.grid(True)

    # 5. Angle of Attack vs. Time
    ax = axes[2, 0]
    ax.plot(time, alpha_deg)
    ax.set_title('Angle of Attack vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Alpha (deg)')
    ax.grid(True)
    
    # 6. Velocity vs. Time
    ax = axes[2, 1]
    ax.plot(time, V)
    ax.set_title('Velocity vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 3. Setup the Dymos Optimization Problem ---
def run_dymos_optimization():
    # 1. Create the OpenMDAO Problem
    p = om.Problem(model=om.Group())
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['maxiter'] = 1000 
    p.driver.options['tol'] = 1.0E-6
    p.driver.declare_coloring()

    # 2. Initialize the Dymos Phase
    # num_segments set to 20 for faster computation
    phase = dm.Phase(
        ode_class=RocketBoostEOM,
        transcription=dm.GaussLobatto(num_segments=20, order=3)
    )

    # 3. Add the Phase to the Problem
    p.model.add_subsystem('phase0', phase)

    phase.add_timeseries_output('Mach')
    phase.add_timeseries_output('Lift')
    phase.add_timeseries_output('Drag')
    phase.add_timeseries_output('x')      
    phase.add_timeseries_output('g_load') 
    phase.add_timeseries_output('alpha')

    # 4. Set Time (Booster Burn Time)
    phase.set_time_options(fix_initial=True, duration_bounds=(10, 30), duration_ref=20.0)

    # 5. Define States and their Bounds
    # Altitude is free to vary, but constrained by the path constraint (see below)
    phase.add_state('h', units='m', rate_source='h_dot',
                    lower=0.0, upper=9000.0, ref=9000.0, defect_ref=9000.0,
                    fix_initial=True, fix_final=False) 

    phase.add_state('V', units='m/s', rate_source='V_dot',
                    lower=1.0, ref=3000.0, defect_ref=3000.0,
                    fix_initial=True) 

    phase.add_state('gamma', units='rad', rate_source='gamma_dot',
                    lower=np.deg2rad(0), upper=np.pi/2, ref=1.0, defect_ref=1.0,
                    fix_initial=False) 

    phase.add_state('m', units='kg', rate_source='m_dot',
                    lower=10.0, ref=45.0,
                    fix_initial=True, # Initial mass is fixed at 45 kg
                    fix_final=False) 

    phase.add_state('x', units='m', rate_source='x_dot',
                    fix_initial=True, ref=10000.0, defect_ref=10000.0)

    # 6. Define Controls and Boundary Constraints
    phase.add_control('alpha', units='rad', lower=np.deg2rad(0), upper=np.deg2rad(6),
                      opt=True)
    
    # Boundary constraints
    phase.add_boundary_constraint('m', loc='final', lower=19.0)    
    
    # CRITICAL CHANGE: REMOVED G-LOAD CONSTRAINT
    
    # Path constraints
    # Maximum altitude is constrained to 9000m throughout the flight
    phase.add_path_constraint('h', upper=9000.0, ref=9000.0)
    phase.add_path_constraint('Mach', lower=5.0, upper=8.0, ref=0.01) # Tight Mach constraint
    phase.add_path_constraint('alpha', lower=np.deg2rad(0), upper=np.deg2rad(6), ref=0.1)
    
    # 7. Define Parameters (S is fixed at 0.0314 m^2)
    phase.add_parameter('T', units='N', opt=False, val=4250.0)
    phase.add_parameter('Isp', units='s', opt=False, val=250.0)
    phase.add_parameter('S', units='m**2', opt=False, val=0.0124) 
    phase.add_parameter('CD_0', opt=False, val=0.1)
    phase.add_parameter('CL_alpha', opt=False, val=0.1)
    phase.add_parameter('payload_mass', units='kg', val=9.0, opt=False)

    # 8. Set the Objective Function
    phase.add_objective('x', loc='final', scaler=-1.0)

    # 9. Setup the Problem and Set Initial Guesses
    p.setup(check=True)

    # Set initial values
    p.set_val('phase0.t_initial', 0.0)
    p.set_val('phase0.t_duration', 15.0)
    
    # Initial mass fixed at 45 kg
    p.set_val('phase0.states:m', phase.interp(ys=[45.0, 19.0], nodes='state_input'))
    
    # State Guesses 
    p.set_val('phase0.states:h', phase.interp(ys=[0.0, 9000], nodes='state_input')) 
    p.set_val('phase0.states:V', phase.interp(ys=[1.0, 2000], nodes='state_input')) 
    p.set_val('phase0.states:gamma', phase.interp(ys=[np.deg2rad(85), np.deg2rad(5)], nodes='state_input'))
    p.set_val('phase0.states:x', phase.interp(ys=[0.0, 450000.0], nodes='state_input')) 
    
    # Using a moderate alpha guess for stability
    p.set_val('phase0.controls:alpha', np.deg2rad(3.0))
    
    # 10. Run the Optimization
    print("Starting optimization...")
    dm.run_problem(p, simulate=True) 
    print("Optimization finished.")

    # 11. Output the Results
    sim_out = p.model.phase0.timeseries
    
    optimal_angle_rad = sim_out.get_val('gamma')[0].item()
    optimal_angle_deg = np.rad2deg(optimal_angle_rad)
    final_velocity = sim_out.get_val('V')[-1].item()
    final_altitude = sim_out.get_val('h')[-1].item()
    max_range = sim_out.get_val('x')[-1].item() 
    final_g_load = sim_out.get_val('g_load')[-1].item() 
    alpha_profile = sim_out.get_val('alpha')
    alpha_start_deg = np.rad2deg(alpha_profile[0].item())
    alpha_end_deg = np.rad2deg(alpha_profile[-1].item())

    print("\n" + "="*50)
    print(" OPTIMIZATION RESULTS (MAX RANGE, Max H=9km)")
    print("="*50)
    #print(f"Initial Mass: {p.get_val('phase0.states:m')[0].item():.2f} kg")
    print(f"Optimal Initial Pitch Angle (gamma_0): {optimal_angle_deg:.2f} degrees")
    print(f"Final Velocity at Burnout: {final_velocity:.2f} m/s")
    print(f"Final Altitude at Burnout: {final_altitude:.2f} meters")
    print(f"Maximum Range Achieved: {max_range / 1000.0:.2f} km") 
    #print(f"Final G-Load (Maneuverability): {final_g_load:.2f} g") # G-Load is now calculated, not constrained
    #print(f"Angle of Attack (alpha) Profile (start to end): {alpha_start_deg:.2f} deg to {alpha_end_deg:.2f} deg")
    print("="*50 + "\n")
    
    # 12. Plot the results
    plot_results(sim_out)

if __name__ == '__main__':
    run_dymos_optimization()
