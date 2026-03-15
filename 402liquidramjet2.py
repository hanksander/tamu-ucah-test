import numpy as np

gamma = 1.4
R = 287.0
g0 = 9.81

# Fit parabola for Isp(M)
M_pts = np.array([2, 4, 6])
Isp_pts = np.array([1500, 1400, 1000])
a, b, c = np.polyfit(M_pts, Isp_pts, 2)

def isp_from_mach(M):
    return a*M**2 + b*M + c

def isa_atmosphere(h):
    if h < 11000:
        T = 288.15 - 0.0065 * h
        P = 101325 * (T / 288.15)**5.2561
    else:
        T = 216.65
        P = 22632 * np.exp(-9.81 * (h - 11000) / (R * T))
    rho = P / (R * T)
    return T, P, rho

def ramjet_design(Mach, altitude, Cd, diameter, thrust_to_drag,
                  AFR):

    S = np.pi * diameter**2 / 4

    T0, P0, rho0 = isa_atmosphere(altitude)
    a0 = np.sqrt(gamma * R * T0)
    V0 = Mach * a0

    drag = 0.5 * rho0 * V0**2 * Cd * S
    required_thrust = thrust_to_drag * drag

    # Isp from fitted parabola
    Isp = isp_from_mach(Mach)

    # Fuel flow from thrust
    mdot_fuel = required_thrust / (g0 * Isp)

    # Air flow from assumed AFR
    mdot_air = AFR * mdot_fuel
    mdot_total = mdot_air + mdot_fuel

    # Inlet area from freestream capture
    A_inlet = mdot_air / (rho0 * V0)

    inlet_to_S = A_inlet / S
    cp = gamma * R / (gamma - 1)
    burner_eff = 0.98
    LHV = 43e6  # JP-10, J/kg

    f = mdot_fuel / mdot_air
    Tmax = T0 + f * LHV * burner_eff / (cp * (1 + f))
    return {
        "Reference Area S (m^2)": S,
        "Drag (N)": drag,
        "Required Thrust (N)": required_thrust,
        "Mach": Mach,
        "Isp (s)": Isp,
        "Fuel Mass Flow (kg/s)": mdot_fuel,
        "Air Mass Flow (kg/s)": mdot_air,
        "Total Mass Flow (kg/s)": mdot_total,
        "Required Inlet Area (m^2)": A_inlet,
        "Inlet / S Ratio": inlet_to_S,
        "Tmax": Tmax
    }


if __name__ == "__main__":
    result = ramjet_design(
        Mach=4.0,
        altitude=21000,
        Cd=0.3,
        diameter=0.27,
        thrust_to_drag=2,
        AFR=15
    )

    for k, v in result.items():
        print(f"{k}: {v}")