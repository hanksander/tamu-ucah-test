
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
    Rb = variable()  #base radius
    Rb = L * ops.tan(phi)  #base radius
    zeta = variable()  #bluntness ratio
    zeta = Rn / Rb  #bluntness ratio
    A_ref = variable()  #reference area
    A_ref = ops.pi * Rb**2  #reference area, m^2
    SA = variable()  #surface area
    SA = ops.pi * Rn * (Rn + ops.sqrt(L**2 + Rb**2))  #surface area, m^2 
    #double check SA
    

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
    CDf = SA*0.664/sqrt(Re) #this is for a low speed laminar flat plate. It needs to be updated. Equation 6.75 from anderson is a better model
    
    CDp = variable() # pressure Cd
    CDp = #modified newtonian theory https://apps.dtic.mil/sti/tr/pdf/AD0631149.pdf
    
    CD = variable() # total Cd
    CD = CDf + CDp

    CL = variable()

    Clalpha = variable()
    CLalpha = 1.1 #close enough (I have a citation I promise)

