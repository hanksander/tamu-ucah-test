# ── Fuel: JP-10 (C₁₀H₁₆, tetrahydrodicyclopentadiene) ─────────────────────
LHV_JP10   = 43.4e6     # lower heating value [J/kg]
F_STOICH   = 0.0667     # stoichiometric fuel-air mass ratio
MW_JP10    = 136.23     # molecular weight [g/mol]

# ── Engine geometry (all areas normalised to A_capture = 1 m²) ──────────────
A_CAPTURE  = 0.1        # capture area [m²] — scale thrust results linearly

# ── Inlet ramp angles ────────────────────────────────────────────────────────
# Two-ramp external compression.  Tune these for your design Mach.
INLET_RAMPS_DEG = [7.0, 10.0, 13.0]   # [deg] — weak shocks, low loss at Mach 6-10

# ── Mode transition ──────────────────────────────────────────────────────────
M_TRANSITION = 5.5      # below → ramjet, above → scramjet

# ── Component efficiencies ───────────────────────────────────────────────────
ETA_COMBUSTOR = 0.90    # fraction of LHV that goes into the flow
ETA_NOZZLE    = 0.95    # nozzle kinetic energy efficiency (velocity coefficient)

# ── Isolator loss model (scram mode) ────────────────────────────────────────
ISOLATOR_PT_RECOVERY_SCRAM = 0.97   # Pt3/Pt2 for scramjet isolator shock train
