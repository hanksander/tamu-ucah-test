"""Thin wrapper around Ambiance for ICAO standard atmosphere."""
from ambiance import Atmosphere as _Atm


def freestream(altitude_m: float) -> tuple[float, float, float]:
    """
    Return (T0, P0, rho0) at the given geometric altitude.

    Parameters
    ----------
    altitude_m : float   Geometric altitude [m]

    Returns
    -------
    T0   : float   Static temperature [K]
    P0   : float   Static pressure [Pa]
    rho0 : float   Density [kg/m³]
    """
    atm = _Atm(altitude_m)
    return (
        float(atm.temperature[0]),
        float(atm.pressure[0]),
        float(atm.density[0]),
    )
