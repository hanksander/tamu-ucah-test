import numpy as np
import openmdao.api as om

# Re-use the trajectory's atm_vec + aero by importing:
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powered_flight_trajectory_code_v4 import (
    atm_vec, aero,
    G0, RE, M_STRUCT, M0 as M0_CONST,
)


class LFRJCruiseODEPath(om.ExplicitComponent):
    """Cruise ODE that emits M4, Tt4, unstart_flag, q for path constraints."""
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('perf_table', recordable=False)   # PerfTable
    def setup(self):
        nn = self.options['num_nodes']
        for n,u in [('x_range','m'),('h','m'),('V','m/s'),
                    ('gamma','rad'),('m','kg'),('alpha','deg')]:
            self.add_input(n, val=np.zeros(nn), units=u)
        self.add_input('phi_cruise', val=0.8)   # static parameter
        for n,u in [('x_range_dot','m/s'),('h_dot','m/s'),('V_dot','m/s**2'),
                    ('gamma_dot','rad/s'),('m_dot','kg/s'),
                    ('M4',None),('Tt4','K'),('unstart_flag',None),
                    ('q','Pa'),('Mach',None)]:
            if u is None: self.add_output(n, val=np.zeros(nn))
            else:         self.add_output(n, val=np.zeros(nn), units=u)
        self.declare_partials('*','*', method='fd')
    def compute(self, I, O):
        tbl = self.options['perf_table']
        h = np.clip(I['h'], 500., 79_000.)
        V = np.clip(I['V'],100., 4_000.)
        gm,m,al = I['gamma'], np.clip(I['m'], M_STRUCT, M0_CONST+10), I['alpha']
        phi = float(np.asarray(I['phi_cruise']).item())
        rho,_,_,a = atm_vec(h); Mn = V/a
        CL,CD,L,D,q = aero(Mn, al, rho, V)
        r = tbl.lookup_batch(Mn, h, phi)
        T     = r['thrust']
        isp   = np.maximum(r['isp'], 1.0)
        mdotf = r['mdot_f']
        M4    = r['M4']
        Tt4   = r['Tt4']
        uns   = r['unstart']
        g  = G0*(RE/(RE+h))**2; ar = np.deg2rad(al)
        O['x_range_dot']=V*np.cos(gm); O['h_dot']=V*np.sin(gm)
        O['V_dot']      =(T*np.cos(ar)-D)/m - g*np.sin(gm)
        O['gamma_dot']  =((T*np.sin(ar)+L)/(m*V) - (g/V-V/(RE+h))*np.cos(gm))
        O['m_dot']      =-mdotf
        O['M4']=M4; O['Tt4']=Tt4; O['unstart_flag']=uns; O['q']=q; O['Mach']=Mn
