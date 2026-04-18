from pyc_run import compute_inlet_conditions, _get_inlet_design
d = _get_inlet_design()
print("design success:", d['success'])
for M in [3.0, 4.5, 5.0, 6.0, 7.0]:
    for mode in ('ram', 'scram'):
        pt, mn = compute_inlet_conditions(M, 20000, mode,
alpha_deg=0.0)
        print(f"M={M} {mode}: ram_recovery={pt:.4e} exit_MN={mn:.3f}")