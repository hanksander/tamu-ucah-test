"""
nozzle_model.py
===============
pyCycle convergent-divergent nozzle model for a hypersonic ramjet.

Cycle elements used
-------------------
  pyc.FlowStart  – prescribes total conditions (Tt, Pt, W, FAR) entering the nozzle
  pyc.Nozzle     – CD nozzle with CEA real-gas thermodynamics and Cfg loss model
  pyc.ExecComp   – post-processing: CF, Cv, area ratio, expansion checks

Flow variable naming (pyCycle convention)
-----------------------------------------
  Fl_I / Fl_O   – inlet / outlet flow station
  tot:T / tot:P – total temperature [°R] / total pressure [psia]
  stat:T / stat:P / stat:MN / stat:V / stat:area
  W             – mass flow [lbm/s]
  FAR           – fuel-to-air ratio (combustion products composition)

Units
-----
  pyCycle uses English units internally.
  All SI equivalents are returned by the post-processor.

Design vs off-design
--------------------
  design=True  → sizes throat area (A*) and exit area (A_exit) from MN target
  design=False → fixed geometry; solver finds actual MN_exit and Fg at new conditions
"""

import openmdao.api as om
import pycycle.api as pyc


# ---------------------------------------------------------------------------
# Thermochemistry configuration
# ---------------------------------------------------------------------------

# Species set for hydrocarbon combustion products (C, H, O, N, Ar).
# For hydrogen fuel: swap to pyc.H2O2_MIX
ELEMENTS = pyc.AIR_FUEL_MIX

# CEA thermochemical tables (JANAF database)
THERMO_DATA = pyc.species_data.janaf


# ---------------------------------------------------------------------------
# Nozzle analysis group
# ---------------------------------------------------------------------------

class RamjetNozzle(pyc.Cycle):
    """
    Standalone CD nozzle analysis group.

    Inputs (set via prob.set_val)
    ------------------------------
    nozzle_in.Tt   – nozzle inlet total temperature [°R]
    nozzle_in.Pt   – nozzle inlet total pressure    [psia]
    nozzle_in.W    – mass flow rate                 [lbm/s]
    nozzle_in.FAR  – fuel-to-air ratio              [-]
    nozzle.MN      – target exit Mach (design only) [-]
    nozzle.Cfg     – gross thrust coefficient        [-]
    nozzle.Ps_exhaust – ambient back-pressure       [psia]

    Key outputs
    -----------
    nozzle.Fg               – gross thrust [lbf]
    nozzle.Fl_O:stat:MN     – exit Mach
    nozzle.Fl_O:stat:V      – exit velocity [ft/s]
    nozzle.Fl_O:stat:area   – exit area [in²]
    nozzle.Fl_O:stat:P      – exit static pressure [psia]
    nozzle.Throat:Fl_O:stat:area – throat area [in²]
    post.AR                 – area ratio A_exit / A_throat
    post.CF                 – thrust coefficient Fg / (Pt * A_throat)
    post.NPR                – nozzle pressure ratio Pt / Ps_exhaust
    post.delta_P            – expansion error Ps_exit - Ps_amb [psia]
    """

    def initialize(self):
        self.options.declare("design", default=True, recordable=False)

    def setup(self):
        design = self.options["design"]

        # ── 1. Flow start: prescribe nozzle inlet conditions ─────────────
        # FlowStart takes (Tt, Pt, W, FAR) and computes the full
        # thermodynamic state at the nozzle inlet.
        self.add_subsystem(
            "nozzle_in",
            pyc.FlowStart(
                thermo_data=THERMO_DATA,
                elements=ELEMENTS,
            ),
        )

        # ── 2. Convergent-divergent nozzle ───────────────────────────────
        # nozzType = 'CD' : convergent-divergent
        # lossCoef = 'Cfg': thrust loss via gross thrust coefficient
        #   Fg_actual = Cfg * Fg_ideal
        # At design   → MN is an input, areas are sized outputs
        # At off-design → areas are fixed inputs, MN is a solved output
        self.add_subsystem(
            "nozzle",
            pyc.Nozzle(
                nozzType="CD",
                lossCoef="Cfg",
                thermo_data=THERMO_DATA,
                elements=ELEMENTS,
                design=design,
            ),
        )

        # ── 3. Post-processing: derived performance metrics ───────────────
        # ExecComp evaluates simple algebraic expressions inside OpenMDAO
        # so derivatives flow through correctly (useful if you later optimize).
        self.add_subsystem(
            "post",
            om.ExecComp(
                [
                    # Area ratio
                    "AR = A_exit / A_throat",
                    # Thrust coefficient: CF = Fg / (Pt_in * A_throat)
                    # A_throat is in in², Pt_in in psia → Fg in lbf (144 in²/ft²)
                    "CF = Fg / (Pt_in * A_throat)",
                    # Nozzle pressure ratio
                    "NPR = Pt_in / Ps_exhaust",
                    # Expansion error: positive = under-expanded, negative = over-expanded
                    "delta_P = Ps_exit - Ps_exhaust",
                    # Velocity coefficient (ratio of actual to isentropic ideal velocity)
                    # Cv is computed from Fg and momentum: Cv ≈ Cfg for small pressure thrust
                    "Cv = Cfg * 1.0",   # simplified; full Cv needs Ps correction
                ],
                A_exit={"units": "inch**2"},
                A_throat={"units": "inch**2"},
                Fg={"units": "lbf"},
                Pt_in={"units": "psia"},
                Ps_exhaust={"units": "psia"},
                Ps_exit={"units": "psia"},
                Cfg={"val": 0.985},
                AR={"units": None},
                CF={"units": None},
                NPR={"units": None},
                delta_P={"units": "psia"},
                Cv={"units": None},
            ),
        )

        # ── Flow connection ───────────────────────────────────────────────
        self.pyc_connect_flow("nozzle_in.Fl_O", "nozzle.Fl_I")

        # ── Post-processing connections ───────────────────────────────────
        self.connect("nozzle.Fl_O:stat:area",           "post.A_exit")
        self.connect("nozzle.Throat:Fl_O:stat:area",    "post.A_throat")
        self.connect("nozzle.Fg",                       "post.Fg")
        self.connect("nozzle_in.Fl_O:tot:P",            "post.Pt_in")
        self.connect("nozzle.Ps_exhaust",               "post.Ps_exhaust")
        self.connect("nozzle.Fl_O:stat:P",              "post.Ps_exit")
        self.connect("nozzle.Cfg",                      "post.Cfg")

        # ── Solvers ───────────────────────────────────────────────────────
        # Newton solver: necessary for the nonlinear CEA thermochemistry.
        # solve_subsystems=True: each subsystem's own solver runs first,
        # giving Newton a better initial residual → faster convergence.
        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        newton.options["atol"]    = 1e-8
        newton.options["rtol"]    = 1e-8
        newton.options["maxiter"] = 50
        newton.options["iprint"]  = 2    # set to -1 to silence

        # DirectSolver: exact LU factorization of the Jacobian.
        # Works well at this model size; swap to PETScKrylov for very large systems.
        self.linear_solver = om.DirectSolver()
