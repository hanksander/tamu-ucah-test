"""Compatibility package for the combined-cycle engine model.

This package historically re-exported many symbols at import time. That
eager import pattern forces optional heavy dependencies such as Cantera to
load even when callers only need lightweight submodules like ``pyc_config``.

Keep the package import cheap, preserve the legacy package-level API via
lazy attribute loading, and retain the local-directory ``sys.path`` shim so
existing absolute intra-package imports (for example ``from thermo import
get_thermo``) continue to work.
"""

from __future__ import annotations

import importlib
import os
import sys

_PKG_DIR = os.path.dirname(__file__)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_EXPORTS = {
    "FlowState": (".gas_dynamics", "FlowState"),
    "make_state": (".gas_dynamics", "make_state"),
    "isentropic_T": (".gas_dynamics", "isentropic_T"),
    "isentropic_P": (".gas_dynamics", "isentropic_P"),
    "isentropic_M_from_Pt_P": (".gas_dynamics", "isentropic_M_from_Pt_P"),
    "normal_shock": (".gas_dynamics", "normal_shock"),
    "oblique_shock": (".gas_dynamics", "oblique_shock"),
    "beta_from_theta": (".gas_dynamics", "beta_from_theta"),
    "rayleigh_exit": (".gas_dynamics", "rayleigh_exit"),
    "pi_milspec": (".gas_dynamics", "pi_milspec"),
    "kantrowitz_limit": (".gas_dynamics", "kantrowitz_limit"),
    "JP10Thermo": (".thermo", "JP10Thermo"),
    "get_thermo": (".thermo", "get_thermo"),
    "freestream": (".atmosphere", "freestream"),
    "compute_inlet": (".inlet", "compute_inlet"),
    "compute_isolator": (".isolator", "compute_isolator"),
    "compute_combustor": (".combustor", "compute_combustor"),
    "compute_nozzle": (".nozzle", "compute_nozzle"),
    "analyze": (".main", "analyze"),
    "mach_sweep": (".main", "mach_sweep"),
    "pyc_run": (".pyc_run", None),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
