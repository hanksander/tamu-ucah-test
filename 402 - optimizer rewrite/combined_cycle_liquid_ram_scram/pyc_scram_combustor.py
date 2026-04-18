"""
pyc_scram_combustor.py
======================
Custom pyCycle Element for supersonic (Rayleigh) combustion.

Architecture
------------
The standard pyc.Combustor uses constant-pressure heat addition (subsonic
RAM mode).  For SCRAM mode we need constant-area Rayleigh flow where both
the Mach number and static pressure change through the combustor.

The element is structured as four sub-steps:

  1. mix_fuel  (ThermoAdd)
     Mixes JP-7 into the incoming air at the given FAR.
     Outputs: mass_avg_h  (total enthalpy of products, Btu/lbm_mix)
              composition_out  (air+fuel species vector)
              Wout  (total mass flow, lbm/s)

  2. vitiated_flow  (Thermo 'total_hP', fl_name='Fl_tmp')
     Finds Tt4 from enthalpy balance at the entry total pressure Pt3.
     Using Pt3 as the pressure is an approximation (Rayleigh Pt4 < Pt3),
     but the effect on temperature is negligible for ideal-gas-like CEA.
     Outputs: Fl_tmp:T  (= Tt4, degR)

  3. rayleigh  (RayleighCalcs, ExplicitComponent)
     Computes exit Mach M4 and exit total pressure Pt4 from the
     Rayleigh flow relations using M3, Tt4/Tt3, and gamma3.
     Uses finite-difference partials (no gradients needed for analysis).
     Outputs: M_out, Pt4  (psia)

  4. real_flow  (Thermo 'total_TP', fl_name='Fl_O:tot')
     Re-evaluates the total flow state at the correct (Tt4, Pt4).
     Promotes outputs as Fl_O:tot:*.

  5. out_stat  (Thermo 'static_MN', fl_name='Fl_O:stat')
     Isentropic expansion from total to static conditions at M4.
     Promotes outputs as Fl_O:stat:*.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import brentq

import openmdao.api as om
import pycycle.api as pyc

from pycycle.flow_in import FlowIn
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.element_base import Element


# ── Rayleigh flow helper functions ────────────────────────────────────────────

def _rayleigh_Tt_ratio(M, gam):
    """Tt/Tt* for Rayleigh flow at Mach M and ratio of specific heats gam."""
    f = 1.0 + gam * M ** 2
    return 2.0 * (1.0 + gam) * M ** 2 * (1.0 + (gam - 1.0) / 2.0 * M ** 2) / f ** 2


def _rayleigh_Pt_ratio(M, gam):
    """Pt/Pt* for Rayleigh flow at Mach M."""
    f = 1.0 + gam * M ** 2
    return ((1.0 + gam) / f) * (
        (2.0 / (gam + 1.0)) * (1.0 + (gam - 1.0) / 2.0 * M ** 2)
    ) ** (gam / (gam - 1.0))


# ── RayleighCalcs component ───────────────────────────────────────────────────

class RayleighCalcs(om.ExplicitComponent):
    """
    Given combustor-entry Mach M_in, exit total temperature Tt4, entry
    total temperature Tt3, specific heat ratio gamma, and entry total
    pressure Pt3, compute the exit Mach M_out and exit total pressure Pt4
    from Rayleigh flow relations.

    Supersonic branch only (SCRAM combustor).  Clips to M=1 if heat
    addition thermally chokes the flow.

    Partials computed by finite difference (analysis-only use case).
    """

    def setup(self):
        self.add_input('M_in',  val=2.5,  desc='Combustor entry Mach number')
        self.add_input('Tt4',   val=3500., units='degR', desc='Exit total temperature')
        self.add_input('Tt3',   val=2000., units='degR', desc='Entry total temperature')
        self.add_input('gamma', val=1.36,  desc='Ratio of specific heats at entry')
        self.add_input('Pt3',   val=50.0,  units='lbf/inch**2', desc='Entry total pressure')

        self.add_output('M_out',  val=1.8,  lower=0.5,  desc='Exit Mach number')
        self.add_output('Pt4',    val=40.0, lower=0.001, units='lbf/inch**2',
                        desc='Exit total pressure')
        self.add_output('choked', val=0.0,  desc='1 if thermally choked, 0 otherwise')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        M3   = float(inputs['M_in'])
        Tt4  = float(inputs['Tt4'])
        Tt3  = float(inputs['Tt3'])
        gam  = float(inputs['gamma'])
        Pt3  = float(inputs['Pt3'])

        Tt_ratio = Tt4 / max(Tt3, 1.0)

        Tt3_Tts = _rayleigh_Tt_ratio(M3, gam)
        Tt4_Tts = Tt3_Tts * Tt_ratio

        if Tt4_Tts >= 1.0:
            # Thermally choked — clamp at M=1
            M4     = 1.0
            choked = 1.0
        else:
            try:
                M4 = brentq(
                    lambda M: _rayleigh_Tt_ratio(M, gam) - Tt4_Tts,
                    1.0 + 1e-6, 20.0, xtol=1e-8,
                )
                choked = 0.0
            except Exception:
                M4     = 1.0
                choked = 1.0

        Pt4_Pt3 = _rayleigh_Pt_ratio(M4, gam) / _rayleigh_Pt_ratio(M3, gam)

        outputs['M_out']  = M4
        outputs['Pt4']    = Pt3 * Pt4_Pt3
        outputs['choked'] = choked


# ── ScramCombustor Element ────────────────────────────────────────────────────

class ScramCombustor(Element):
    """
    pyCycle Element for supersonic constant-area Rayleigh combustion.

    Flow ports
    ----------
    Fl_I  : inlet  (supersonic, connected from upstream Inlet element)
    Fl_O  : outlet (connected to downstream Nozzle element)

    Design inputs
    -------------
    Fl_I:FAR   fuel-to-air mass ratio (= phi * f_stoich_JP7)

    Design outputs
    --------------
    Wfuel      fuel mass flow [lbm/s]
    """

    def initialize(self):
        self.options.declare('fuel_type', default='JP-7',
                             desc='Fuel species name in pyCycle CEA database.')
        self.options.declare('statics', default=True,
                             desc='If True, compute static flow properties.')

        self.default_des_od_conns = [
            ('Fl_O:stat:area', 'area'),
        ]
        super().initialize()

    def pyc_setup_output_ports(self):
        """Register output-port composition (air + fuel mixture)."""
        thermo_method = self.options['thermo_method']
        thermo_data   = self.options['thermo_data']
        fuel_type     = self.options['fuel_type']

        # Build ThermoAdd to determine output composition — same pattern as
        # pyc.Combustor.pyc_setup_output_ports().
        self.thermo_add_comp = ThermoAdd(
            method=thermo_method,
            mix_mode='reactant',
            thermo_kwargs={
                'spec':               thermo_data,
                'inflow_composition': self.Fl_I_data['Fl_I'],
                'mix_composition':    fuel_type,
            },
        )
        self.copy_flow(self.thermo_add_comp, 'Fl_O')

    def setup(self):
        thermo_method      = self.options['thermo_method']
        thermo_data        = self.options['thermo_data']
        statics            = self.options['statics']
        inflow_composition = self.Fl_I_data['Fl_I']
        air_fuel_comp      = self.Fl_O_data['Fl_O']

        # ── 1. Accept inlet flow ──────────────────────────────────────────────
        in_flow = FlowIn(fl_name='Fl_I')
        self.add_subsystem('in_flow', in_flow,
                           promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # ── 2. Mix fuel → h_mix, composition_out, Wout ───────────────────────
        self.add_subsystem(
            'mix_fuel', self.thermo_add_comp,
            promotes=[
                'Fl_I:stat:W',
                ('mix:ratio', 'Fl_I:FAR'),
                'Fl_I:tot:composition',
                'Fl_I:tot:h',
                ('mix:W', 'Wfuel'),
                'Wout',
            ],
        )

        # ── 3. Enthalpy balance: find Tt4 at entry pressure (Pt3 approx) ─────
        # Use fl_name='Fl_tmp' so outputs don't collide with Fl_O:tot:*.
        vit_flow = Thermo(
            mode='total_hP',
            fl_name='Fl_tmp',
            method=thermo_method,
            thermo_kwargs={'composition': air_fuel_comp, 'spec': thermo_data},
        )
        self.add_subsystem('vitiated_flow', vit_flow)
        self.connect('mix_fuel.mass_avg_h',      'vitiated_flow.h')
        self.connect('mix_fuel.composition_out', 'vitiated_flow.composition')
        self.connect('Fl_I:tot:P',               'vitiated_flow.P')  # Pt3 approx

        # ── 4. Rayleigh exit: M4, Pt4 ─────────────────────────────────────────
        self.add_subsystem('rayleigh', RayleighCalcs())
        self.connect('Fl_I:stat:MN',        'rayleigh.M_in')
        self.connect('vitiated_flow.Fl_tmp:T', 'rayleigh.Tt4')
        self.connect('Fl_I:tot:T',          'rayleigh.Tt3')
        self.connect('Fl_I:tot:gamma',      'rayleigh.gamma')
        self.connect('Fl_I:tot:P',          'rayleigh.Pt3')

        # ── 5. Final total state at correct (Tt4, Pt4) ────────────────────────
        real_flow = Thermo(
            mode='total_TP',
            fl_name='Fl_O:tot',
            method=thermo_method,
            thermo_kwargs={'composition': air_fuel_comp, 'spec': thermo_data},
        )
        self.add_subsystem('real_flow', real_flow,
                           promotes_outputs=['Fl_O:tot:*'])
        self.connect('vitiated_flow.Fl_tmp:T', 'real_flow.T')
        self.connect('rayleigh.Pt4',           'real_flow.P')
        self.connect('mix_fuel.composition_out', 'real_flow.composition')

        # ── 6. Static conditions at exit Mach M4 ─────────────────────────────
        if statics:
            out_stat = Thermo(
                mode='static_MN',
                fl_name='Fl_O:stat',
                method=thermo_method,
                thermo_kwargs={'composition': air_fuel_comp, 'spec': thermo_data},
            )
            self.add_subsystem('out_stat', out_stat,
                               promotes_outputs=['Fl_O:stat:*'])
            self.connect('rayleigh.M_out',         'out_stat.MN')
            self.connect('mix_fuel.composition_out', 'out_stat.composition')
            self.connect('Fl_O:tot:S',             'out_stat.S')
            self.connect('Fl_O:tot:h',             'out_stat.ht')
            self.connect('Fl_O:tot:P',             'out_stat.guess:Pt')
            self.connect('Fl_O:tot:gamma',         'out_stat.guess:gamt')
            self.connect('Wout',                   'out_stat.W')
        else:
            from pycycle.passthrough import PassThrough
            self.add_subsystem(
                'W_passthru',
                PassThrough('Wout', 'Fl_O:stat:W', 1.0, units='lbm/s'),
                promotes=['*'],
            )

        super().setup()