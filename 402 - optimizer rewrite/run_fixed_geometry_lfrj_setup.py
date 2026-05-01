from __future__ import annotations

import argparse
from pathlib import Path

from trajectory_opt.engine_adapter import PyCycleRamAdapter
from trajectory_opt.fixed_lfrj import (
    CRUISE_ALT_M,
    CRUISE_MACH,
    PHI_CRUISE,
    build_fixed_perf_table,
    fixed_lfrj_design,
    validate_fixed_engine,
    write_fixed_engine_report,
)


def _print_validation(rows: list[dict]):
    print("\nFixed-geometry direct pyCycle validation")
    print("  Mach   alt[km]  status       thrust[kN]   Isp[s]   mdot_f[g/s]   M4     Tt4[K]  unstart")
    print("  " + "-" * 91)
    for r in rows:
        print(
            f"  {r['Mach']:>4.2f}   {r['altitude_m'] / 1000.0:>6.1f}  "
            f"{r['status']:<10s}  {r['thrust_N'] / 1000.0:>10.2f}  "
            f"{r['Isp_s']:>7.0f}  {r['mdot_fuel_kgs'] * 1000.0:>11.2f}  "
            f"{r['M4']:>5.3f}  {r['Tt4_K']:>7.0f}  {r['unstart_flag']:>7.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate and cache the fixed-geometry LFRJ performance model."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the surrogate table even if a matching cache exists",
    )
    parser.add_argument(
        "--output-dir",
        default="fixed_lfrj_outputs",
        help="directory for design, validation, and summary files",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run the direct validation map and skip the surrogate cache build",
    )
    parser.add_argument(
        "--show-solver-output",
        action="store_true",
        help="do not suppress pyCycle/OpenMDAO solver logs",
    )
    args = parser.parse_args()

    design = fixed_lfrj_design()
    engine = PyCycleRamAdapter()

    print("=" * 72)
    print("Fixed-geometry LFRJ setup")
    print("=" * 72)
    print(f"  design digest : {design.digest()}")
    print(f"  engine design : Mach {design.design_M0:.2f} at {design.design_alt_m / 1000.0:.1f} km")
    print(f"  cruise target : Mach {CRUISE_MACH:.2f} at {CRUISE_ALT_M / 1000.0:.1f} km")
    print(f"  cruise phi    : {PHI_CRUISE:.2f}")
    print("\nBuilding fixed geometry...")
    geom = engine.geometry(design)
    print(f"  length        : {geom.get('total_length_m', float('nan')):.3f} m")
    print(f"  max diameter  : {geom.get('max_diameter_m', float('nan')):.3f} m")
    print(f"  max width     : {geom.get('max_width_m', float('nan')):.3f} m")
    print(f"  max height    : {geom.get('max_height_m', float('nan')):.3f} m")

    quiet = not args.show_solver_output
    rows = validate_fixed_engine(design, engine=engine, phi=PHI_CRUISE, quiet=quiet)
    _print_validation(rows)

    table = None
    if args.validate_only:
        print("\nSkipping focused performance table build (--validate-only).")
    else:
        print("\nBuilding focused fixed-geometry performance table...")
        table = build_fixed_perf_table(
            design,
            engine=engine,
            force=args.force,
            quiet=quiet,
        )
        cache_path = table._cache_path()
        print(f"  cache         : {cache_path}")

    summary = write_fixed_engine_report(Path(args.output_dir), design, rows, table)
    print("\nWrote setup outputs")
    for label, path in summary["files"].items():
        print(f"  {label:<10s}: {path}")


if __name__ == "__main__":
    main()
