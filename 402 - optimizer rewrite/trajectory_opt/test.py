from engine_adapter import PyCycleRamAdapter
from perf_surrogate import PerfTable
from engine_interface import Design

t = PerfTable(Design(), PyCycleRamAdapter()).build()
for M in (2.5, 3.5, 4.5):
    for h in (15000, 20000, 25000):
        r = t.lookup(M, h, 0.8)
        print(f"M={M} h={h}: unstart={r.unstart_flag:.3f}")