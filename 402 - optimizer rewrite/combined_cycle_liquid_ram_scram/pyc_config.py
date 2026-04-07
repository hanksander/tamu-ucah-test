"""
pyc_config.py
=============
Constants for the pyCycle dual-mode ram/scramjet.

Fuel note
---------
JP-10 (C10H16) is not in pyCycle's CEA database.  JP-7 is the closest
available species: LHV ~43.5 MJ/kg vs JP-10 43.4 MJ/kg, stoichiometric
FAR ~0.0664.  All thermochemistry is handled by pyCycle's CEA solver.

Units
-----
pyCycle works internally in American engineering units (degR, psia, lbm/s,
lbf, ft).  pyc_run.py accepts SI inputs and reports SI outputs.
"""

# Fuel
FUEL_TYPE     = 'JP-7'    # closest to JP-10 in pyCycle CEA database
F_STOICH_JP7  = 0.0664    # stoichiometric fuel-air mass ratio
LHV_JP7       = 43.5e6    # lower heating value [J/kg]

# Engine geometry
A_CAPTURE = 0.05          # inlet capture area [m^2]

# Inlet
INLET_RAMPS_DEG = [6.0, 8.0, 10.0]   # external-compression ramp angles [deg]

# Efficiencies
ETA_COMBUSTOR        = 0.92   # combustion efficiency
ETA_NOZZLE_CV        = 0.97   # nozzle velocity coefficient (Cv)
ISOLATOR_PT_RECOVERY = 0.97   # isolator total-pressure recovery (both modes)

# Mode transition (default; overridable per-call in pyc_run.analyze)
M_TRANSITION = 5.0

# RAM combustor exit Mach for pyCycle design mode
# Fixed-geometry approximation used across the Mach sweep.
RAM_COMBUSTOR_EXIT_MN = 0.6

# Operating range
M_MIN = 2.0
M_MAX = 8.0
