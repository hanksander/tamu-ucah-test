"""
pyc_scram_cycle.py
==================
SCRAM mode cycle:  FlightConditions -> Inlet -> ScramCombustor -> Nozzle -> Performance

The ScramCombustor replaces pyc.Combustor with Rayleigh constant-area heat
addition.  The inlet still uses pyc.Inlet — the oblique-shock recovery is
pre-computed and passed in as ram_recovery, while the supersonic exit Mach is
passed in as inlet.MN.  No normal shock is added (that is the RAM/SCRAM split).
"""

import openmdao.api as om
import pycycle.api as pyc

from pyc_config import FUEL_TYPE
from pyc_scram_combustor import ScramCombustor


class ScramCycle(pyc.Cycle):

    def setup(self):
        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data']   = pyc.species_data.janaf

        # Elements
        self.add_subsystem('fc',    pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('burner', ScramCombustor(fuel_type=FUEL_TYPE))
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