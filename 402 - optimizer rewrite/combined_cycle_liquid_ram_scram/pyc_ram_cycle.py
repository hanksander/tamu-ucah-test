"""
pyc_ram_cycle.py
================
RAM mode cycle:  FlightConditions -> Inlet -> Combustor -> Nozzle -> Performance

The inlet total-pressure recovery and combustor-face Mach are pre-solved by
402inlet2's cascade-only subsonic closure and handed to pyc.Inlet via
ram_recovery and MN at analyze() time — no implicit Newton loop inside the
cycle.
"""

import os
import importlib.util

import numpy as np
import openmdao.api as om
import pycycle.api as pyc

from combustor import (
    compute_combustor_variable_rayleigh,
    build_station3_state_from_totals,
)
from pyc_config import (
    ETA_COMBUSTOR, F_STOICH_JP10,
    INLET_DESIGN_ALPHA_DEG,
)
from thermo import get_thermo
from pycycle.element_base import Element

# 402inlet2.py — module name begins with a digit, import via importlib
_inlet2_spec = importlib.util.spec_from_file_location(
    'inlet2', os.path.join(os.path.dirname(__file__), '402inlet2.py'))
_inlet2 = importlib.util.module_from_spec(_inlet2_spec)
_inlet2_spec.loader.exec_module(_inlet2)

class RayleighCombustorCalcs(om.ExplicitComponent):
    """
    RAM-mode combustor using the local Rayleigh-flow combustor model.
    """

    _FD_INPUTS = (
        'Fl_I:tot:T',
        'Fl_I:tot:P',
        'Fl_I:stat:MN',
        'Fl_I:FAR',
        'area_ratio',
    )
    _MASS_DEPENDENT_OUTPUTS = (
        'Fl_O:stat:area',
        'Fl_O:stat:W',
    )

    _FLOW_OUTPUTS = (
        ('Fl_O:tot:h', 'J/kg', 1.0e6),
        ('Fl_O:tot:T', 'K', 1200.0),
        ('Fl_O:tot:P', 'Pa', 2.0e5),
        ('Fl_O:tot:S', 'J/(kg*K)', 1000.0),
        ('Fl_O:tot:rho', 'kg/m**3', 1.0),
        ('Fl_O:tot:gamma', None, 1.3),
        ('Fl_O:tot:Cp', 'J/(kg*K)', 1300.0),
        ('Fl_O:tot:Cv', 'J/(kg*K)', 1000.0),
        ('Fl_O:tot:R', 'J/(kg*K)', 287.0),
        ('Fl_O:stat:h', 'J/kg', 9.0e5),
        ('Fl_O:stat:T', 'K', 1000.0),
        ('Fl_O:stat:P', 'Pa', 1.5e5),
        ('Fl_O:stat:S', 'J/(kg*K)', 1000.0),
        ('Fl_O:stat:rho', 'kg/m**3', 1.0),
        ('Fl_O:stat:gamma', None, 1.3),
        ('Fl_O:stat:Cp', 'J/(kg*K)', 1300.0),
        ('Fl_O:stat:Cv', 'J/(kg*K)', 1000.0),
        ('Fl_O:stat:R', 'J/(kg*K)', 287.0),
        ('Fl_O:stat:V', 'm/s', 250.0),
        ('Fl_O:stat:Vsonic', 'm/s', 400.0),
        ('Fl_O:stat:MN', None, 0.5),
        ('Fl_O:stat:area', 'm**2', 0.05),
        ('Fl_O:stat:W', 'kg/s', 5.0),
    )

    def setup(self):
        self.add_input('Fl_I:tot:h', val=1.0, units='J/kg')
        self.add_input('Fl_I:tot:T', val=1000.0, units='K')
        self.add_input('Fl_I:tot:P', val=2.0e5, units='Pa')
        self.add_input('Fl_I:tot:S', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:tot:rho', val=1.0, units='kg/m**3')
        self.add_input('Fl_I:tot:gamma', val=1.4)
        self.add_input('Fl_I:tot:Cp', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:tot:Cv', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:tot:R', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:tot:composition', val=1.0, shape_by_conn=True)

        self.add_input('Fl_I:stat:h', val=1.0, units='J/kg')
        self.add_input('Fl_I:stat:T', val=1000.0, units='K')
        self.add_input('Fl_I:stat:P', val=2.0e5, units='Pa')
        self.add_input('Fl_I:stat:S', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:stat:rho', val=1.0, units='kg/m**3')
        self.add_input('Fl_I:stat:gamma', val=1.4)
        self.add_input('Fl_I:stat:Cp', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:stat:Cv', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:stat:R', val=1.0, units='J/(kg*K)')
        self.add_input('Fl_I:stat:composition', val=1.0, shape_by_conn=True)
        self.add_input('Fl_I:stat:V', val=1.0, units='m/s')
        self.add_input('Fl_I:stat:Vsonic', val=1.0, units='m/s')
        self.add_input('Fl_I:stat:MN', val=0.3)
        self.add_input('Fl_I:stat:area', val=0.05, units='m**2')
        self.add_input('Fl_I:stat:W', val=5.0, units='kg/s')
        self.add_input('Fl_I:FAR', val=0.02)
        self.add_input('area_ratio', val=1.0)

        for name, units, val in self._FLOW_OUTPUTS:
            if units is None:
                self.add_output(name, val=val)
            else:
                self.add_output(name, val=val, units=units)
        self.add_output('Fl_O:tot:composition', val=1.0, copy_shape='Fl_I:tot:composition')
        self.add_output('Fl_O:stat:composition', val=1.0, copy_shape='Fl_I:stat:composition')
        self.add_output('Wfuel', val=0.1, units='kg/s')
        self.add_output('choked', val=0.0)

        for out_name, _, _ in self._FLOW_OUTPUTS:
            for in_name in self._FD_INPUTS:
                self.declare_partials(out_name, in_name, method='fd')
        for out_name in self._MASS_DEPENDENT_OUTPUTS:
            self.declare_partials(out_name, 'Fl_I:stat:W', method='fd')
        self.declare_partials('Wfuel', 'Fl_I:stat:W', method='fd')
        self.declare_partials('Wfuel', 'Fl_I:FAR', method='fd')
        for in_name in self._FD_INPUTS:
            self.declare_partials('choked', in_name, method='fd')

        # Composition is a pass-through from the inlet state; only declare the
        # matching dependencies instead of a full output-input Cartesian product.
        self.declare_partials('Fl_O:tot:composition', 'Fl_I:tot:composition', method='fd')
        self.declare_partials('Fl_O:stat:composition', 'Fl_I:stat:composition', method='fd')

    @staticmethod
    def _entropy(thermo, T, phi, P):
        thermo._set_state(T, phi, P)
        return thermo._gas.entropy_mass

    @staticmethod
    def _flow_props(thermo, T, phi, P):
        props = thermo.all_props(T, phi, P)
        gamma = props['gamma']
        cp = props['cp']
        gas_r = props['R']
        rho = P / max(gas_r * T, 1.0e-12)
        return {
            'h': props['h'],
            'T': T,
            'P': P,
            'S': RayleighCombustorCalcs._entropy(thermo, T, phi, P),
            'rho': rho,
            'gamma': gamma,
            'Cp': cp,
            'Cv': cp / gamma,
            'R': gas_r,
        }

    def compute(self, inputs, outputs):
        thermo = get_thermo()

        W_air = float(inputs['Fl_I:stat:W'][0])
        Pt3 = float(inputs['Fl_I:tot:P'][0])
        Tt3 = float(inputs['Fl_I:tot:T'][0])
        M3 = float(inputs['Fl_I:stat:MN'][0])
        far = float(inputs['Fl_I:FAR'][0])
        area_ratio = float(inputs['area_ratio'][0])

        state3 = build_station3_state_from_totals(
            Pt3=Pt3,
            Tt3=Tt3,
            M3=M3,
            thermo=thermo,
        )
        # FAR from analyze() is now the true phi*f_stoich (no eta_c baked in),
        # so recover phi by dividing by f_stoich alone. eta_c is applied by
        # compute_combustor to the heat release only.
        phi = far / max(F_STOICH_JP10, 1.0e-12)
        # The current combustor model is intentionally constant-area. OpenMDAO
        # finite-difference passes may perturb `area_ratio` away from 1.0, so
        # ignore that input here until a variable-area combustor model exists.
        _ = area_ratio
        state4, choked = compute_combustor_variable_rayleigh(
            state3,
            phi,
            thermo,
            area_ratio=1.0,
        )

        Wfuel = W_air * far
        Wout = W_air + Wfuel
        a4 = np.sqrt(state4.gamma * state4.R * state4.T)
        V4 = state4.M * a4
        rho4 = state4.P / max(state4.R * state4.T, 1.0e-12)
        area4 = Wout / max(rho4 * V4, 1.0e-12)

        tot = self._flow_props(thermo, state4.Tt, phi, state4.Pt)
        stat = self._flow_props(thermo, state4.T, phi, state4.P)
        stat.update({
            'V': V4,
            'Vsonic': a4,
            'MN': state4.M,
            'area': area4,
            'W': Wout,
        })

        for key, value in tot.items():
            outputs[f'Fl_O:tot:{key}'] = value
        for key, value in stat.items():
            outputs[f'Fl_O:stat:{key}'] = value

        outputs['Fl_O:tot:composition'] = inputs['Fl_I:tot:composition']
        outputs['Fl_O:stat:composition'] = inputs['Fl_I:stat:composition']
        outputs['Wfuel'] = Wfuel
        outputs['choked'] = 1.0 if choked else 0.0


class RayleighCombustor(Element):
    """
    pyCycle-compatible RAM combustor wrapper around RayleighCombustorCalcs.
    """

    def pyc_setup_output_ports(self):
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        self.add_subsystem(
            'rayleigh',
            RayleighCombustorCalcs(),
            promotes=['*'],
        )

        super().setup()


class RamCycle(pyc.Cycle):

    def setup(self):
        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data']   = pyc.species_data.janaf

        # Elements
        self.add_subsystem('fc',    pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('burner', RayleighCombustor())
        self.add_subsystem('nozz',  pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('perf',  pyc.Performance(num_nozzles=1, num_burners=1))

        # Flow connections
        self.pyc_connect_flow('fc.Fl_O',     'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O',  'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'nozz.Fl_I')

        # Scalar connections for Performance
        self.connect('fc.Fl_O:tot:P',     'perf.Pt2')
        self.connect('burner.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram',       'perf.ram_drag')
        self.connect('nozz.Fg',           'perf.Fg_0')
        self.connect('burner.Wfuel',      'perf.Wfuel_0')

        # Ambient static pressure drives nozzle perfect expansion
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        self.set_order(['fc', 'inlet', 'burner', 'nozz', 'perf'])
        super().setup()
