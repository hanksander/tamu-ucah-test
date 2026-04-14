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


"""
INLET DESIGN PARAMETERS 
=========================================

1. DESIGN MACH NUMBER (INLET_DESIGN_M0)
   - Choose ~midpoint of operating range (e.g., 4.0–4.5), NOT the maximum.
   - Higher → more compression, smaller throat → harder to start at low Mach.
   - Lower → easier starting but lower recovery at high Mach.

2. KANTROWITZ MARGIN (INLET_KANTROWITZ_MARGIN)
   - Fraction of the self-starting contraction limit. Must be < 1.0.
   - Lower (0.85–0.90) ensures starting at lowest Mach in range.
   - Check: CR_geom < kantrowitz_limit(M_min) after design.

3. DIFFUSER AREA RATIO (DIFFUSER_AREA_RATIO)
   - Ratio of combustor face area to throat area.
   - Larger (2.5–3.0) gives more room for terminal shock to move aft,
     delaying swallowing at high Mach.
   - Too large → long diffuser, risk of separation.

4. DIFFUSER HALF-ANGLE (DIFFUSER_HALF_ANGLE_DEG)
   - Divergence angle of subsonic diffuser walls.
   - Shallow (2.5–3.0°) prevents flow separation, especially at low Mach
     when shock is near throat and pressure gradient is steep.

5. FOREBODY / RAMP SEPARATION MARGINS
   - Fraction of maximum attached-shock turning angle allowed.
   - Conservative (0.15–0.20) avoids unstart at low Mach / high AoA.
   - Lower margins improve recovery but narrow operability.

6. SHOCK FOCUS FACTOR (INLET_SHOCK_FOCUS_FACTOR)
   - Positions cowl lip relative to ramp-2 shock impingement.
   - Lower (1.15–1.20) moves lip forward → safer low-Mach starting,
     slightly reduced capture area.
   - Higher → better mass flow capture at design point.

7. ANGLE OF ATTACK (INLET_DESIGN_ALPHA_DEG)
   - Design at 0° unless cruise AoA is known and fixed.
   - Positive AoA increases effective ramp angles → reduces low-Mach margin.

TYPICAL VALUES FOR MACH 2.5–5.0 RAMJET:
   M0_design       = 4.2
   Kantrowitz_margin = 0.88
   Diffuser_AR     = 2.8
   Diffuser_half_angle = 2.5°
   Forebody_sep    = 0.15
   Ramp_sep        = 0.18
   Shock_focus     = 1.18
   Alpha_design    = 0.0

After changing parameters, re-run design and check:
   - Kantrowitz pass at M_min (CR_geom < CR_k(M_min))
   - Off-design sweep: success=True across range, smooth recovery curve.
"""

INLET_DESIGN_M0                    = 4
INLET_DESIGN_ALT_M                 = 12_000.0
INLET_DESIGN_ALPHA_DEG             = 0.0
INLET_DESIGN_LEADING_EDGE_ANGLE_DEG = 5.0
INLET_DESIGN_MDOT_KGS              = 7.0     # design-point air mass flow [kg/s]
INLET_DESIGN_WIDTH_M               = 0.25    # inlet spanwise width [m]
INLET_FOREBODY_SEP_MARGIN          = 0.15
INLET_RAMP_SEP_MARGIN              = 0.18
INLET_KANTROWITZ_MARGIN            = 0.85
INLET_SHOCK_FOCUS_FACTOR           = 1.18

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
