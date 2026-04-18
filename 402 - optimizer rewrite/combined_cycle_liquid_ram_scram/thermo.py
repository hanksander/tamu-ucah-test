"""
JP-10 thermodynamic properties via Cantera chemical equilibrium.
====================================================================
Uses GRI 3.0 as the thermodynamic database for all product species.
JP-10 (C₁₀H₁₆) is represented by initialising the gas mixture to the
analytically correct combustion product composition for each φ, then
calling Cantera's equilibrate('TP') to handle high-T dissociation.

Singleton pattern: call get_thermo() anywhere — the Cantera Solution
object is instantiated once and reused.

Usage
-----
    from thermo import get_thermo
    th = get_thermo()
    print(th.gamma(2500.0, 0.8))         # → ~1.22  (at 1 atm default)
    print(th.gamma(2500.0, 0.8, 5e5))    # → at actual combustor pressure
    print(th.cp(2500.0, 0.8))            # → J/(kg·K)
    print(th.h(2500.0, 0.8))             # sensible enthalpy J/kg_mix
    print(th.h_air(1500.0))              # pure air sensible enthalpy J/kg_air

CHANGES vs original
-------------------
- _set_state() now accepts an optional pressure P [Pa] (default: 1 atm).
  All public methods forward P to _set_state so Cantera equilibrates at
  the actual flow pressure instead of always at 1 atm.  This matters at
  combustor pressures (2–10 atm) where dissociation equilibrium shifts.
- h_air() likewise accepts P.
- All new parameters default to ct.one_atm so existing call sites that
  don't pass pressure continue to work without modification.
"""

import cantera as ct
import numpy as np

# JP-10 stoichiometry: C₁₀H₁₆ + 14 O₂ → 10 CO₂ + 8 H₂O
_C   = 10
_H   = 16
_O2S = _C + _H / 4.0   # = 14 mol O₂ per mol fuel (stoichiometric O₂)

# Dry air: per mole O₂ → N₂:3.76, Ar:0.04435
_AIR = {'N2': 3.76, 'O2': 1.0, 'AR': 0.04435}   # GRI30 uses 'AR' not 'Ar'

_H_REF = 298.15   # reference temperature for sensible enthalpy [K]


def _products_composition(phi: float) -> str:
    """
    Initial product mole fractions for JP-10 combustion at equivalence ratio φ.
    Cantera equilibrate('TP') will refine this for actual temperature.
    """
    n_O2 = _O2S / max(phi, 1e-6)
    n_N2 = n_O2 * _AIR['N2']
    n_Ar = n_O2 * _AIR['AR']

    if phi <= 1.0:
        n_CO2 = float(_C)
        n_H2O = float(_H / 2)
        n_O2x = n_O2 - _O2S          # excess O₂
        return (f'N2:{n_N2:.4f}, O2:{max(n_O2x,0):.4f}, '
                f'CO2:{n_CO2:.4f}, H2O:{n_H2O:.4f}, AR:{n_Ar:.4f}')
    else:
        # Rich: oxygen-atom balance → CO + H₂
        n_O_avail = 2.0 * n_O2
        n_H2O = float(_H / 2)
        n_O_avail -= n_H2O            # O consumed by H₂O
        n_CO2 = max(0.0, n_O_avail - _C)
        n_CO  = max(0.0, _C - n_CO2)
        n_H2  = 0.0                   # all H assumed to H₂O for lean-side start
        return (f'N2:{n_N2:.4f}, CO2:{n_CO2:.4f}, CO:{max(n_CO,1e-6):.4f}, '
                f'H2O:{n_H2O:.4f}, H2:{max(n_H2,1e-6):.4f}, AR:{n_Ar:.4f}')


class JP10Thermo:
    """
    Cantera-backed JP-10 combustion product thermodynamics.

    All properties are returned at chemical equilibrium for the given
    (T, P, φ) — dissociation of CO₂, H₂O, and N₂ at high T is captured,
    and the equilibrium shift with pressure is now correctly handled.

    The sensible enthalpy convention (h = 0 at 298.15 K for each φ)
    matches the LHV-based combustor energy balance in combustor.py.
    """

    def __init__(self, mechanism: str = 'gri30.yaml'):
        self._gas = ct.Solution(mechanism)
        # Cache reference enthalpies per phi (avoids re-equilibrating at 298 K)
        self._h_ref_cache: dict[float, float] = {}
        # Air reference at 1 atm (reference is always at standard conditions)
        self._gas.TPX = _H_REF, ct.one_atm, f"N2:{_AIR['N2']}, O2:{_AIR['O2']}, AR:{_AIR['AR']}"
        self._gas.equilibrate('TP')
        self._h_air_ref = self._gas.enthalpy_mass

    # ── INTERNAL ──────────────────────────────────────────────────────────────

    def _set_state(self, T: float, phi: float, P: float = ct.one_atm):
        """Set Cantera gas to equilibrated products at (T, P, phi).

        Parameters
        ----------
        T   : float  Static temperature [K]
        phi : float  Equivalence ratio
        P   : float  Static pressure [Pa] — default 1 atm.
                     Pass the actual flow pressure for correct dissociation
                     equilibrium (high P suppresses dissociation per Le Chatelier).
        """
        comp = f"N2:{_AIR['N2']}, O2:{_AIR['O2']}, AR:{_AIR['AR']}" if phi < 1e-4 \
               else _products_composition(phi)
        self._gas.TPX = T, P, comp
        self._gas.equilibrate('TP')

    def _h_ref(self, phi: float) -> float:
        """Sensible enthalpy reference at 298.15 K, 1 atm for this φ.
        Always evaluated at standard conditions so h=0 at (298 K, 1 atm).
        """
        key = round(phi, 4)
        if key not in self._h_ref_cache:
            self._set_state(_H_REF, phi, ct.one_atm)
            self._h_ref_cache[key] = self._gas.enthalpy_mass
        return self._h_ref_cache[key]

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def gamma(self, T: float, phi: float, P: float = ct.one_atm) -> float:
        """Ratio of specific heats γ = cp/cv at (T, P, phi)."""
        self._set_state(T, phi, P)
        return self._gas.cp_mass / self._gas.cv_mass

    def cp(self, T: float, phi: float, P: float = ct.one_atm) -> float:
        """Specific heat at constant pressure [J/(kg·K)] at (T, P, phi)."""
        self._set_state(T, phi, P)
        return self._gas.cp_mass

    def R(self, T: float, phi: float, P: float = ct.one_atm) -> float:
        """Specific gas constant [J/(kg·K)] = R_u / MW_mix at (T, P, phi)."""
        self._set_state(T, phi, P)
        return ct.gas_constant / self._gas.mean_molecular_weight

    def h(self, T: float, phi: float, P: float = ct.one_atm) -> float:
        """Sensible enthalpy of combustion products [J/kg_mix] (zero at 298.15 K, 1 atm).

        For the combustor energy balance, pass the combustor static pressure
        so dissociation is evaluated at the correct conditions.
        """
        self._set_state(T, phi, P)
        return self._gas.enthalpy_mass - self._h_ref(phi)

    def h_air(self, T: float, P: float = ct.one_atm) -> float:
        """Sensible enthalpy of pure air [J/kg_air] (zero at 298.15 K, 1 atm)."""
        comp = f"N2:{_AIR['N2']}, O2:{_AIR['O2']}, AR:{_AIR['AR']}"
        self._gas.TPX = T, P, comp
        self._gas.equilibrate('TP')
        return self._gas.enthalpy_mass - self._h_air_ref

    def all_props(self, T: float, phi: float, P: float = ct.one_atm) -> dict:
        """Return γ, cp, R, h, MW in one dict (one Cantera call) at (T, P, phi)."""
        self._set_state(T, phi, P)
        return {
            'T': T, 'phi': phi, 'P': P,
            'gamma': self._gas.cp_mass / self._gas.cv_mass,
            'cp':    self._gas.cp_mass,
            'R':     ct.gas_constant / self._gas.mean_molecular_weight,
            'h':     self._gas.enthalpy_mass - self._h_ref(phi),
            'MW':    self._gas.mean_molecular_weight * 1e3,  # g/mol
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_thermo: JP10Thermo | None = None


def get_thermo() -> JP10Thermo:
    """Return module-level singleton.  Cantera initialised once."""
    global _thermo
    if _thermo is None:
        _thermo = JP10Thermo()
    return _thermo


if __name__ == '__main__':
    th = get_thermo()
    print(f"{'T [K]':>8}  {'γ (1atm)':>10}  {'γ (5atm)':>10}  "
          f"{'cp [J/kgK]':>12}  {'h [MJ/kg]':>11}  {'R [J/kgK]':>11}")
    for T in [1000, 1500, 2000, 2500, 3000]:
        p1 = th.all_props(T, 0.8, ct.one_atm)
        p5 = th.all_props(T, 0.8, 5 * ct.one_atm)
        print(f"{T:>8.0f}  {p1['gamma']:>10.4f}  {p5['gamma']:>10.4f}  "
              f"{p1['cp']:>12.1f}  {p1['h']/1e6:>11.4f}  {p1['R']:>11.2f}")