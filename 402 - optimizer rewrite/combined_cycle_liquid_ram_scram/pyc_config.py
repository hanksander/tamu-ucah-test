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



# Inlet design point — passed to 402inlet2.design_2ramp_shock_matched_inlet().
# Geometry is frozen at import time; all flight-point calls use
# evaluate_fixed_geometry_at_condition() against this design.
INLET_DESIGN_M0                    = 5
INLET_DESIGN_ALT_M                 = 12_000.0
INLET_DESIGN_ALPHA_DEG             = 2.0
INLET_DESIGN_LEADING_EDGE_ANGLE_DEG = 6.0
INLET_DESIGN_MDOT_KGS              = 6.0     # design-point air mass flow [kg/s]
INLET_DESIGN_WIDTH_M               = 0.25    # inlet spanwise width [m]
INLET_FOREBODY_SEP_MARGIN          = 0.20
INLET_RAMP_SEP_MARGIN              = 0.28
INLET_KANTROWITZ_MARGIN            = 0.95
INLET_SHOCK_FOCUS_FACTOR           = 1.25

# Efficiencies
ETA_COMBUSTOR        = 0.92   # combustion efficiency
ETA_NOZZLE_CV        = 0.97   # nozzle velocity coefficient (Cv)
NOZZLE_TYPE          = 'CD'   # nozzle_design.py pyCycle nozzle type: 'CD', 'CV', or 'CD_CV'
ISOLATOR_PT_RECOVERY = 0.97   # isolator total-pressure recovery (both modes)

# Mode transition (default; overridable per-call in pyc_run.analyze)
M_TRANSITION = 5.2

# RAM combustor exit Mach for pyCycle design mode
# Fixed-geometry approximation used across the Mach sweep.
RAM_COMBUSTOR_EXIT_MN = 0.6

# Operating range
M_MIN = 2.0
M_MAX = 8.0
