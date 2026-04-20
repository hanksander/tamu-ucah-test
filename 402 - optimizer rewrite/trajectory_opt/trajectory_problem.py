from __future__ import annotations

import os
import sys
import numpy as np
import openmdao.api as om
import dymos as dm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from combined_cycle_liquid_ram_scram.pyc_config import (
    M4_MAX, TT4_MAX_K, Q_MAX_PA,
)
from powered_flight_trajectory_code_v4 import (
    M0 as M0_MASS, M_PROP_BOOST, M_STRUCT, M_FUEL_LFRJ, LFRJ_CRUISE_ALT,
    atmosphere,
)
try:
    from .trajectory_ode import LFRJCruiseODEPath
except ImportError:
    from trajectory_opt.trajectory_ode import LFRJCruiseODEPath

M_POST_BOOST = M0_MASS - M_PROP_BOOST   # mass entering LFRJ cruise (808 kg)

# Cruise envelope the engine is designed to satisfy.
CRUISE_H_LO_M  = 16_000.0
CRUISE_H_HI_M  = 22_000.0
CRUISE_M_LO    = 4.0
CRUISE_M_HI    = 5.0


def build_inner_problem(perf_table, h0_m, V0, phi_init=0.8):
    """Cruise-only inner problem.

    The descent phase has been removed: this engine is being designed to
    operate only in the cruise envelope (M=4-5, alt=15-24 km), so the
    trajectory is a single cruise phase with range as the objective. No
    terminal impact constraint.
    """
    p    = om.Problem(reports=False)
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    sc = 4

    cruise = dm.Phase(ode_class=LFRJCruiseODEPath,
                        transcription=dm.GaussLobatto(num_segments=sc, order=3),
                        ode_init_kwargs={'perf_table': perf_table})
    traj.add_phase('cruise', cruise)
    cruise.set_time_options(fix_initial=True, initial_val=0.0,
                            duration_bounds=(60, 1500), units='s')
    for n, (lb, ub) in {
        'x_range': (0, 5e6),
        'h':       (CRUISE_H_LO_M, CRUISE_H_HI_M),
        'V':       (1150, 1500),
        'gamma':   (-0.3, 0.3),
        'm':       (M_STRUCT, M_POST_BOOST),
    }.items():
        cruise.add_state(n, fix_initial=True, rate_source=f'{n}_dot',
                        lower=lb, upper=ub)
    cruise.add_control('alpha', lower=-5, upper=15, units='deg')
    cruise.add_parameter('phi_cruise', val=phi_init, opt=False,
                        static_target=True)
    cruise.add_boundary_constraint('m', loc='final', lower=M_STRUCT, units='kg')

    # Path constraints — no choking, no unstart, stay in envelope.
    cruise.add_path_constraint('M4',           upper=M4_MAX,    ref=M4_MAX)
    cruise.add_path_constraint('Tt4',          upper=TT4_MAX_K, ref=TT4_MAX_K, units='K')
    cruise.add_path_constraint('unstart_flag', upper=0.5,       ref=1.0)
    cruise.add_path_constraint('Mach',         lower=CRUISE_M_LO, upper=CRUISE_M_HI,
                                                ref=CRUISE_M_HI)

    # Maximise cruise range.
    cruise.add_objective('x_range', loc='final', scaler=-1e-5)

    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
    p.driver.options['maxiter'] = 50
    p.driver.options['tol'] = 1e-3

    """
    p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    p.driver.opt_settings['max_iter'] = 50
    p.driver.opt_settings['tol'] = 1e-3
    p.driver.opt_settings['mu_strategy'] = 'adaptive'
    p.driver.opt_settings['print_level'] = 0          # quiet logs
    p.driver.declare_coloring()
    
    """


    p.setup(force_alloc_complex=False)
    _set_initial_guesses(p, sc, h0_m, V0)
    return p


def _set_initial_guesses(p, sc, h0_m, V0):
    nn_c = sc + 1

    p.set_val('traj.cruise.t_duration',    600.0)
    p.set_val('traj.cruise.states:x_range', np.linspace(0, 1.5e6, nn_c))
    p.set_val('traj.cruise.states:h',       np.full(nn_c, h0_m))
    p.set_val('traj.cruise.states:V',       np.full(nn_c, V0))
    p.set_val('traj.cruise.states:gamma',   np.zeros(nn_c))
    p.set_val('traj.cruise.states:m',       np.linspace(M_POST_BOOST, M_STRUCT + 10, nn_c))

    # Pin the fixed initial state (index 0 of the cruise state vector)
    p.set_val('traj.cruise.states:x_range', 0.0,          indices=[0])
    p.set_val('traj.cruise.states:h',       h0_m,         indices=[0])
    p.set_val('traj.cruise.states:V',       V0,           indices=[0])
    p.set_val('traj.cruise.states:gamma',   0.0,          indices=[0])
    p.set_val('traj.cruise.states:m',       M_POST_BOOST, indices=[0])


def extract_metrics(p):
    return dict(
        range_m=float(p.get_val('traj.cruise.states:x_range').ravel()[-1]),
        fuel_kg=float(M_POST_BOOST - p.get_val('traj.cruise.states:m').ravel()[-1]),
    )


def snapshot_for_warmstart(p):
    return {
        'cruise': {
            't_duration': float(np.asarray(p.get_val('traj.cruise.t_duration')).item()),
            **{f'states:{s}': p.get_val(f'traj.cruise.states:{s}').copy()
                for s in ('x_range', 'h', 'V', 'gamma', 'm')},
            'controls:alpha': p.get_val('traj.cruise.controls:alpha').copy(),
        }
    }


def apply_warmstart(p, snap):
    for ph, d in snap.items():
        for k, v in d.items():
            try: p.set_val(f'traj.{ph}.{k}', v)
            except Exception: pass
