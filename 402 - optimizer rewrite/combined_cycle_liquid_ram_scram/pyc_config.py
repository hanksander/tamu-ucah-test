"""
pyc_config.py
=============
Constants for the pyCycle ramjet.

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

# Fuel — JP-10 (C10H16, tetrahydrodicyclopentadiene)
# All RAM-cycle thermochemistry is handled by Cantera via JP10Thermo;
# the constants below are used for stoichiometry and heat-release bookkeeping.
FUEL_TYPE      = 'JP-10'
F_STOICH_JP10  = 0.0667    # stoichiometric fuel-air mass ratio
F_STOICH = 0.0667
LHV_JP10       = 43.4e6    # lower heating value [J/kg]




"""
DESIGN PARAMETERS 
=========================================
 INLET_DESIGN_M0

  - Sets the Mach number the inlet geometry is designed around.
  - Higher: stronger design-point compression, smaller throat tendency, worse low-Mach operability.
  - Lower: easier starting/off-design at low Mach, weaker high-Mach design performance.

  INLET_DESIGN_ALT_M

  - Sets design altitude for freestream density, pressure, and temperature during inlet sizing.
  - Higher: thinner air, affects required capture opening and shock thermodynamics.
  - Lower: denser air, changes required opening and mass-flow-based geometry sizing.

  INLET_DESIGN_ALPHA_DEG

  - Sets the angle of attack used during inlet design.
  - Higher: increases effective forebody turn, usually makes shock attachment margins tighter.
  - Lower: more benign design geometry, generally better robustness.

  INLET_DESIGN_LEADING_EDGE_ANGLE_DEG

  - Sets the forebody leading-edge / initial compression angle.
  - Higher: stronger initial compression, more risk of separation/unattached shocks.
  - Lower: gentler compression, safer off design, lower peak recovery.

  INLET_DESIGN_MDOT_KGS

  - Sets required captured air mass flow at the design point.
  - Higher: requires larger capture area / inlet opening.
  - Lower: smaller inlet opening and generally smaller geometry.

  INLET_DESIGN_WIDTH_M

  - Sets spanwise inlet width used to convert area into heights.
  - Higher: same area can be achieved with smaller vertical height.
  - Lower: larger required heights for the same flow area.

  INLET_FOREBODY_SEP_MARGIN

  - Fraction of max attached-shock turn allowed on the forebody.
  - Higher: more aggressive forebody compression, better potential recovery, less margin.
  - Lower: more conservative, more robust, less compression.

  INLET_RAMP_SEP_MARGIN

  - Fraction of max attached-shock turn allowed on ramp 1 and ramp 2.
  - Higher: stronger ramp compression, smaller/tighter geometry, more risk off design.
  - Lower: gentler ramps, smoother off-design behavior, less total compression.

  INLET_KANTROWITZ_MARGIN

  - Fraction of the Kantrowitz contraction limit used to size the throat.
  - Higher: smaller throat, stronger contraction, better design compression, worse starting/off-design tolerance.
  - Lower: larger throat, easier starting, weaker compression, usually more robust.

  INLET_SHOCK_FOCUS_FACTOR

  - Controls where the cowl lip sits relative to the ramp-2 shock focus.
  - Higher: pushes geometry toward tighter shock focusing / capture, can get more fragile.
  - Lower: safer/less aggressive placement, often better off-design robustness.

  DIFFUSER_AREA_RATIO

  - Sets combustor-face area divided by throat area.
  - Higher: more diffuser expansion, more subsonic diffusion, more room for terminal-shock travel, but can create stronger regime
    sensitivity and discontinuities in this model.
  - Lower: more compact diffuser, less shock travel room, usually smoother but less flexible back-pressure accommodation.

  DIFFUSER_HALF_ANGLE_DEG

  - User-set diffuser half-angle if explicit angle sizing is used.
  - Higher: shorter diffuser, stronger geometric divergence, more real-world separation risk.
  - Lower: longer, gentler diffuser.

  DIFFUSER_PHYSICS_EQUIV_HALF_ANGLE_DEG

  - Equivalent half-angle limit used by the physics-based diffuser length sizing.
  - Higher: allows shorter diffuser.
  - Lower: forces longer diffuser for gentler diffusion.

  DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH

  - Minimum diffuser length in throat hydraulic diameters to give the terminal shock room.
  - Higher: longer diffuser, better modeled off-design shock accommodation.
  - Lower: shorter diffuser, less shock travel room, earlier expelled/swallowed transitions.
"""

INLET_DESIGN_M0                    = 5
INLET_DESIGN_ALT_M                 = 18_000.0
INLET_DESIGN_ALPHA_DEG             = 4.0
INLET_DESIGN_LEADING_EDGE_ANGLE_DEG = 4.0
INLET_DESIGN_MDOT_KGS              = 10.0     # design-point air mass flow [kg/s]
INLET_DESIGN_WIDTH_M               = 0.25    # inlet spanwise width [m]
INLET_FOREBODY_SEP_MARGIN          = 0.25
INLET_RAMP_SEP_MARGIN              = 0.25
INLET_KANTROWITZ_MARGIN            = 0.80
INLET_SHOCK_FOCUS_FACTOR           = 1.1

DIFFUSER_AREA_RATIO = 2
DIFFUSER_HALF_ANGLE_DEG = 7.0
DIFFUSER_PHYSICS_EQUIV_HALF_ANGLE_DEG = 2.5
DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH = 3.0


COMBUSTOR_L_STAR_DEFAULT = 1.25

# Efficiencies
ETA_COMBUSTOR        = 0.92   # combustion efficiency
ETA_NOZZLE_CV        = 0.97   # nozzle velocity coefficient (Cv)
NOZZLE_TYPE          = 'CD'   # nozzle_design.py pyCycle nozzle type: 'CD', 'CV', or 'CD_CV'
ISOLATOR_PT_RECOVERY = 0.97   # isolator total-pressure recovery (both modes)

PHI_DEFAULT = 0.5

# RAM combustor exit Mach for pyCycle design mode
# Fixed-geometry approximation used across the Mach sweep.
RAM_COMBUSTOR_EXIT_MN = 0.6

# Operating range
M_MIN = 2.0
M_MAX = 5.0
