"""
pyc_ram_cycle.py
================
RAM mode cycle:  FlightConditions -> Inlet -> Combustor -> Nozzle -> Performance

The inlet total-pressure recovery is now solved implicitly via
DiffuserTerminalShock: it takes the combustor inlet static pressure and
returns the terminal-normal-shock Pt recovery plus the subsonic exit Mach.
Newton closes the Ps_back <-> ram_recovery loop.
"""

import os
import importlib.util

import numpy as np
import openmdao.api as om
import pycycle.api as pyc

from pyc_config import (
    FUEL_TYPE, RAM_COMBUSTOR_EXIT_MN,
    ISOLATOR_PT_RECOVERY, INLET_DESIGN_ALPHA_DEG,
)

# 402inlet2.py — module name begins with a digit, import via importlib
_inlet2_spec = importlib.util.spec_from_file_location(
    'inlet2', os.path.join(os.path.dirname(__file__), '402inlet2.py'))
_inlet2 = importlib.util.module_from_spec(_inlet2_spec)
_inlet2_spec.loader.exec_module(_inlet2)


class DiffuserTerminalShock(om.ExplicitComponent):
    """
    Solves the terminal normal shock position from combustor back pressure.
    """

    def initialize(self):
        self.options.declare('inlet_design')
        self.options.declare('isolator_pt_recovery', default=1.0, types=float)
        self.options.declare('alpha_deg', default=0.0, types=float)

    def setup(self):
        self.add_input('Ps_back', val=5e4, units='Pa')
        self.add_input('M0', val=2.0)
        self.add_input('alt_m', val=10000.0, units='m')
        self.add_output('ram_recovery', val=0.5)
        self.add_output('MN_exit', val=0.3)
        self.add_output('Pt_after_cowl_Pa', val=1e5, units='Pa')
        self.add_output('x_shock', val=0.0)
        self.add_output('unstart_flag', val=0.0)
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        design  = self.options['inlet_design']
        alpha   = self.options['alpha_deg']
        iso_pt  = self.options['isolator_pt_recovery']
        Ps_back = float(inputs['Ps_back'][0])
        M0      = float(inputs['M0'][0])
        alt_m   = float(inputs['alt_m'][0])

        case = _inlet2.evaluate_fixed_geometry_at_condition(
            design, M0=M0, altitude_m=alt_m, alpha_deg=alpha,
            p_back=Ps_back,
        )

        if not case.get('success', False):
            status = case.get('status', 'unknown')
            term   = case.get('terminal', {})
            Pt_ac  = case.get('Pt_after_cowl',
                              term.get('Pt_after_shock', 1.0))
            if status == 'expelled':
                outputs['ram_recovery'] = 0.1 * iso_pt
                outputs['MN_exit']      = 0.2
                outputs['unstart_flag'] = +1.0
            elif status == 'swallowed':
                outputs['ram_recovery'] = (
                    term.get('pt_frac_after_terminal_shock', 0.9)
                    * iso_pt)
                outputs['MN_exit']      = 0.9
                outputs['unstart_flag'] = -1.0
            else:
                outputs['ram_recovery'] = 0.05 * iso_pt
                outputs['MN_exit']      = 0.3
                outputs['unstart_flag'] = +1.0
            outputs['Pt_after_cowl_Pa'] = Pt_ac
            outputs['x_shock']          = 0.0
            return

        pt_frac_total = case['pt_frac_after_terminal_shock'] * iso_pt
        MN_exit       = float(case['M_at_combustor_face'])
        MN_exit = float(np.clip(MN_exit, 0.05, 0.95))

        outputs['ram_recovery']     = float(pt_frac_total)
        outputs['MN_exit']          = MN_exit
        outputs['Pt_after_cowl_Pa'] = float(case['Pt_after_cowl'])
        outputs['x_shock']          = float(case['x_terminal_shock'])
        outputs['unstart_flag']     = 0.0


class RamCycle(pyc.Cycle):

    def setup(self):
        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data']   = pyc.species_data.janaf

        # Elements
        self.add_subsystem('fc',    pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('nozz',  pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('perf',  pyc.Performance(num_nozzles=1, num_burners=1))

        # Terminal-shock diffuser coupled to combustor inlet static pressure.
        # Lazy import to avoid circularity: pyc_run imports RamCycle.
        from pyc_run import _get_inlet_design
        self.add_subsystem(
            'diff',
            DiffuserTerminalShock(
                inlet_design=_get_inlet_design(),
                isolator_pt_recovery=ISOLATOR_PT_RECOVERY,
                alpha_deg=INLET_DESIGN_ALPHA_DEG,
            ),
        )

        # Flow connections
        self.pyc_connect_flow('fc.Fl_O',     'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O',  'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'nozz.Fl_I')

        # Close the Ps_back <-> ram_recovery loop.
        self.connect('burner.Fl_I:stat:P', 'diff.Ps_back')
        self.connect('diff.ram_recovery',  'inlet.ram_recovery')
        self.connect('diff.MN_exit',       'inlet.MN')

        # Scalar connections for Performance
        self.connect('fc.Fl_O:tot:P',     'perf.Pt2')
        self.connect('burner.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram',       'perf.ram_drag')
        self.connect('nozz.Fg',           'perf.Fg_0')
        self.connect('burner.Wfuel',      'perf.Wfuel_0')

        # Ambient static pressure drives nozzle perfect expansion
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        self.set_order(['fc', 'inlet', 'diff', 'burner', 'nozz', 'perf'])
        super().setup()
