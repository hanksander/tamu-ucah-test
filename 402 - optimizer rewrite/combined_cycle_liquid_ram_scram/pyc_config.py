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
MW_JP10        = 136.23    # molecular weight [g/mol]

# Legacy (non-pyCycle) stack defaults.  Kept here so config.py can be removed
# and every constant has one canonical source.
A_CAPTURE                  = 0.1                 # legacy capture area [m^2]
INLET_RAMPS_DEG            = [7.0, 10.0, 13.0]   # legacy 3-ramp deflections
M_TRANSITION               = 5.5                 # ram <-> scram mode boundary
ISOLATOR_PT_RECOVERY_SCRAM = 0.97                # scram isolator Pt ratio




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

#Vehicle forebody angle = 8 degrees.
#total height 0.35 meters.
#total width = 0.45

INLET_DESIGN_M0                    = 4.0
INLET_DESIGN_ALT_M                 = 16_000.0   # mid of 19–21 km envelope
INLET_DESIGN_ALPHA_DEG             = 4        # worst-α design anchor
INLET_DESIGN_LEADING_EDGE_ANGLE_DEG = 8
INLET_DESIGN_MDOT_KGS              = 8     # design-point air mass flow [kg/s]
INLET_DESIGN_WIDTH_M               = 0.3   # inlet spanwise width [m] (hard req)
INLET_FOREBODY_SEP_MARGIN          = 0.6
INLET_RAMP_SEP_MARGIN              = 0.2
INLET_KANTROWITZ_MARGIN            = 0.80
INLET_SHOCK_FOCUS_FACTOR           = 1.25

# Air properties used by 402inlet2's cold-air oblique/normal-shock solver
# (freestream-temperature relations). The thermally-perfect path uses
# gamma_air(T) from NASA-7 polynomials for N2/O2 at the mole fractions below.
AIR_GAMMA_REF = 1.4       # reference cold-air ratio of specific heats
AIR_R         = 287.05    # dry-air specific gas constant [J/(kg·K)]
AIR_X_N2      = 0.79      # air mole fraction N2 (frozen-chemistry shock)
AIR_X_O2      = 0.21      # air mole fraction O2 (frozen-chemistry shock)

# 402inlet2 legacy function-default factors (used by __main__ sweep harness;
# runtime callers override via the INLET_*_MARGIN / SHOCK_FOCUS_FACTOR knobs
# above).
INLET_LEGACY_FOREBODY_SEP_MARGIN = 0.95
INLET_LEGACY_RAMP_SEP_MARGIN     = 0.95
INLET_LEGACY_KANTROWITZ_MARGIN   = 0.95
INLET_LEGACY_SHOCK_FOCUS_FACTOR  = 1.1

# 402inlet2 plotting knobs (visual only — do not affect flow solution)
INLET_SHOCK_EXTENSION_FACTOR = 1.40   # how far shocks are drawn past the inlet
INLET_COWL_EXTENSION_FACTOR  = 1.25   # cowl length scale in plots
INLET_COWL_MIN_LENGTH_M      = 0.1    # floor on drawn cowl length [m]
INLET_CONSTANT_AREA_LENGTH_M = 0.3    # visual-only constant-area section ahead of diffuser [m]

DIFFUSER_AREA_RATIO = 3.5
DIFFUSER_HALF_ANGLE_DEG = 7.0
DIFFUSER_PHYSICS_EQUIV_HALF_ANGLE_DEG = 12
DIFFUSER_MIN_SHOCK_ACCOMMODATION_DH = 1.0


COMBUSTOR_LENGTH_M_DEFAULT = 0.75
COMBUSTOR_WIDTH_M_DEFAULT  = 0.35
NOZZLE_AR                = 9  # nozzle Ae/At committed design knob
NOZZLE_AR_DEFAULT        = NOZZLE_AR   # legacy alias

# Efficiencies
ETA_COMBUSTOR        = 0.85   # combustion efficiency
ETA_NOZZLE_CV        = 0.97   # nozzle velocity coefficient (Cv)
ETA_NOZZLE           = ETA_NOZZLE_CV   # legacy alias
NOZZLE_TYPE          = 'CD'   # nozzle_design.py pyCycle nozzle type: 'CD', 'CV', or 'CD_CV'
ISOLATOR_PT_RECOVERY = 0.90   # isolator total-pressure recovery (both modes)
ETA_DIFFUSER         = 0.95   # subsonic diffuser total-pressure recovery (friction)

PHI_DEFAULT = 0.75

# Upper edge of the φ search bracket used by _solve_phi_envelope when inverting
# the Tt4 / thermal-choke / inlet-expulsion caps. Decoupled from PHI_DEFAULT so
# physical caps remain detectable at any requested φ.
PHI_SEARCH_MAX = 3.0

# RAM combustor exit Mach for pyCycle design mode
# Fixed-geometry approximation used across the Mach sweep.
RAM_COMBUSTOR_EXIT_MN = 0.6

# Operating range
M_MIN = 2.0
M_MAX = 5.0


# Path-constraint thresholds for optimization
M4_MAX         = 0.85
TT4_MAX_K      = 2800.0
Q_MAX_PA       = 120_000.0
PHI_MIN        = 0.30
PHI_MAX        = 0.90
ENGINE_L_MAX_M = 3.8       # hard geometry budget (user requirement)
ENGINE_D_MAX_M = 0.38      # overall frontal cap; combustor chamber is stricter (0.35)
ENGINE_COMBUSTOR_D_MAX_M    = 0.35   # combustion-chamber diameter cap
ENGINE_NOZZLE_EXIT_D_MAX_M  = 0.38   # nozzle exit diameter cap
ENGINE_MIN_THRUST_N         = 6_000.0

# v2 φ-envelope closure: where to place the terminal shock along the
# diffuser capability range [Ps_min, Ps_max].
#   Ps3 = Ps_min + PS3_BIAS · (Ps_max − Ps_min)
# 0.0 → shock at exit (strong, max diffusion, tight margin to unstart).
# 1.0 → shock at throat (weak, least diffusion, safest start margin).
# 0.7 is a conservative design-intent default — weak shock near throat,
# mimicking typical started-mode operation.
PS3_BIAS = 0.75


#0.14 m^3
