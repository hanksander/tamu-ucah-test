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
    from .trajectory_ode import LFRJCruiseODEPath, DescentODEPath
except ImportError:
    from trajectory_opt.trajectory_ode import LFRJCruiseODEPath, DescentODEPath

M_POST_BOOST = M0_MASS - M_PROP_BOOST   # mass entering LFRJ cruise (808 kg)

def build_inner_problem(perf_table, h0_m, V0, phi_init=0.8):
    p    = om.Problem(reports=False)
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    sc, sd = 4, 4

    # ----- Cruise (starts at Mach 4, post-boost) -----
    cruise = dm.Phase(ode_class=LFRJCruiseODEPath,
                        transcription=dm.GaussLobatto(num_segments=sc, order=3),
                        ode_init_kwargs={'perf_table': perf_table})
    traj.add_phase('cruise', cruise)
    cruise.set_time_options(fix_initial=True, initial_val=0.0,
                            duration_bounds=(60, 1500), units='s')
    for n,(lb,ub) in {'x_range':(0,5e6),'h':(15_000,30_000),
                        'V':(1150,1500),'gamma':(-0.3,0.3),
                        'm':(M_STRUCT, M_POST_BOOST)}.items():
        cruise.add_state(n, fix_initial=True, rate_source=f'{n}_dot',
                        lower=lb, upper=ub)
    cruise.add_control('alpha', lower=-5, upper=15, units='deg')
    cruise.add_parameter('phi_cruise', val=phi_init, opt=False,
                        static_target=True)
    cruise.add_boundary_constraint('m', loc='final', lower=M_STRUCT, units='kg')
    # path constraints (ref= for clean scaling)
    cruise.add_path_constraint('M4',           upper=M4_MAX,    ref=M4_MAX)
    cruise.add_path_constraint('Tt4',          upper=TT4_MAX_K, ref=TT4_MAX_K, units='K')
    cruise.add_path_constraint('unstart_flag', upper=0.5,       ref=1.0)
    cruise.add_path_constraint('Mach',         lower=4.0, upper=5.0, ref=5.0)

    # ----- Descent -----
    descent = dm.Phase(ode_class=DescentODEPath,
                        transcription=dm.GaussLobatto(num_segments=sd, order=3))
    traj.add_phase('descent', descent)
    descent.set_time_options(fix_initial=False, duration_bounds=(20,200), units='s')
    for n,(lb,ub) in {'x_range':(0,5e6),'h':(0,50000),'V':(100,5000),
                        'gamma':(-np.pi,0),'m':(M_STRUCT*0.9,M_POST_BOOST)}.items():
        descent.add_state(n, fix_initial=False, rate_source=f'{n}_dot',
                            lower=lb, upper=ub)
    descent.add_control('alpha', lower=-30, upper=10, units='deg')
    descent.add_boundary_constraint('h',     loc='final', equals=0.,              units='m')
    descent.add_boundary_constraint('V',     loc='final', lower=680.,             units='m/s')
    descent.add_boundary_constraint('gamma', loc='final', upper=np.deg2rad(-80.), units='rad')

    traj.link_phases(['cruise','descent'], vars=['time','x_range','h','V','gamma','m'])

    descent.add_objective('x_range', loc='final', scaler=-1e-5)

    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
    p.driver.options['maxiter'] = 50
    p.driver.options['tol'] = 1e-3
    p.setup(force_alloc_complex=False)
    _set_initial_guesses(p, sc, sd, h0_m, V0)
    return p

def _set_initial_guesses(p, sc, sd, h0_m, V0):
    nn_c, nn_d = sc+1, sd+1

    p.set_val('traj.cruise.t_duration',    600.0)
    p.set_val('traj.cruise.states:x_range',np.linspace(0, 1.5e6, nn_c))
    p.set_val('traj.cruise.states:h',      np.full(nn_c, h0_m))
    p.set_val('traj.cruise.states:V',      np.full(nn_c, V0))
    p.set_val('traj.cruise.states:gamma',  np.zeros(nn_c))
    p.set_val('traj.cruise.states:m',      np.linspace(M_POST_BOOST, M_STRUCT+10, nn_c))

    # Pin the fixed initial state (index 0 of the cruise state vector)
    p.set_val('traj.cruise.states:x_range', 0.0,          indices=[0])
    p.set_val('traj.cruise.states:h',       h0_m,         indices=[0])
    p.set_val('traj.cruise.states:V',       V0,           indices=[0])
    p.set_val('traj.cruise.states:gamma',   0.0,          indices=[0])
    p.set_val('traj.cruise.states:m',       M_POST_BOOST, indices=[0])

    p.set_val('traj.descent.t_duration',   60.0)
    p.set_val('traj.descent.states:h',     np.linspace(h0_m, 0, nn_d))
    p.set_val('traj.descent.states:V',     np.linspace(V0, 900, nn_d))
    p.set_val('traj.descent.states:gamma', np.linspace(-0.1, -1.5, nn_d))

def extract_metrics(p):
    return dict(
        range_m=float(p.get_val('traj.descent.states:x_range').ravel()[-1]),
        fuel_kg=float(M_POST_BOOST - p.get_val('traj.cruise.states:m').ravel()[-1]),
    )

def snapshot_for_warmstart(p):
    keys = ('cruise','descent')
    out = {}
    for ph in keys:
        out[ph] = {
            't_duration': float(np.asarray(p.get_val(f'traj.{ph}.t_duration')).item()),
            **{f'states:{s}': p.get_val(f'traj.{ph}.states:{s}').copy()
                for s in ('x_range','h','V','gamma','m')},
            'controls:alpha': p.get_val(f'traj.{ph}.controls:alpha').copy(),
        }
    return out

def apply_warmstart(p, snap):
    for ph, d in snap.items():
        for k, v in d.items():
            try: p.set_val(f'traj.{ph}.{k}', v)
            except Exception: pass
