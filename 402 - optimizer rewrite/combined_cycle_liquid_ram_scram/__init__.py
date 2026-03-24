import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Gas dynamics core ────────────────────────────────────────────────────────
from gas_dynamics import (
    FlowState,
    make_state,
    isentropic_T,
    isentropic_P,
    isentropic_M_from_Pt_P,
    normal_shock,
    oblique_shock,
    beta_from_theta,
    rayleigh_exit,
    pi_milspec,
    kantrowitz_limit,
)

# ── Thermodynamics ───────────────────────────────────────────────────────────
from thermo import JP10Thermo, get_thermo

# ── Atmosphere ───────────────────────────────────────────────────────────────
from atmosphere import freestream

# ── Cycle components ─────────────────────────────────────────────────────────
from inlet     import compute_inlet
from isolator  import compute_isolator
from combustor import compute_combustor
from nozzle    import compute_nozzle

# ── Top-level analysis ───────────────────────────────────────────────────────
from main import analyze, mach_sweep

# ── Config constants (re-exported for convenience) ───────────────────────────
from config import (
    LHV_JP10,
    F_STOICH,
    MW_JP10,
    A_CAPTURE,
    INLET_RAMPS_DEG,
    M_TRANSITION,
    ETA_COMBUSTOR,
    ETA_NOZZLE,
    ISOLATOR_PT_RECOVERY_SCRAM,
)

__all__ = [
    # gas_dynamics
    "FlowState", "make_state",
    "isentropic_T", "isentropic_P", "isentropic_M_from_Pt_P",
    "normal_shock", "oblique_shock", "beta_from_theta",
    "rayleigh_exit", "pi_milspec", "kantrowitz_limit",
    # thermo
    "JP10Thermo", "get_thermo",
    # atmosphere
    "freestream",
    # components
    "compute_inlet", "compute_isolator", "compute_combustor", "compute_nozzle",
    # top-level
    "analyze", "mach_sweep",
    # config
    "LHV_JP10", "F_STOICH", "MW_JP10", "A_CAPTURE", "INLET_RAMPS_DEG",
    "M_TRANSITION", "ETA_COMBUSTOR", "ETA_NOZZLE", "ISOLATOR_PT_RECOVERY_SCRAM",
]

