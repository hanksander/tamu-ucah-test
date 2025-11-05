import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import operator
import pykridge as pk

def parse_cfd_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Locate the function data section
    data_start = None
    for i, line in enumerate(lines):
        if "Function Data:" in line:
            data_start = i + 2
            break

    if data_start is None:
        raise RuntimeError(f"Cannot find 'Function Data:' section in {filepath}")

    # Parse the function data
    data = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        tokens = line.strip().split()
        if len(tokens) != 5:
            continue
        mach, q, alpha, beta, f = map(float, tokens)
        data.append([mach, q, alpha, beta, f])

    df = pd.DataFrame(data, columns=["Mach", "q", "alpha", "beta", "F"])
    return df

def find_files(directory, model_name, param):
    totl_params = ["CDw", "CLw", "CFx", "CFy", "CFz", "CMl", "CMm", "CMx", "CMy", "CMz", "CSw"]
    if param in totl_params:
        pattern = f"{model_name}.{param}.Totl.dat"
    else:
        pattern = f"{model_name}.{param}.dat"
    return glob.glob(os.path.join(directory, pattern))

def extract_geometry_from_dir(dirname, prefix="waverider"):
    """
    Extracts geometry parameters from directory name.
    e.g., waverider_4_2_3 -> {"x1": 4, "x2": 2, "x3": 3}
    """
    parts = dirname.replace(prefix + "_", "").split("_")
    return {f"x{i+1}": int(p) for i, p in enumerate(parts)}

def collect_data_across_dirs(param, model_prefix="waverider"):
    all_data = []

    is_derived = param == "L/D"
    required_params = ["CLw", "CDw"] if is_derived else [param]

    for entry in os.scandir('.'):
        if entry.is_dir() and entry.name.startswith(model_prefix + "_"):
            geom_params = extract_geometry_from_dir(entry.name, prefix=model_prefix)

            data_frames = {}
            for req_param in required_params:
                files = find_files(entry.path, model_prefix, req_param)
                if not files:
                    print(f"Skipping {entry.name}: no file for {req_param}")
                    break
                try:
                    df = parse_cfd_file(files[0])
                    for k, v in geom_params.items():
                        df[k] = v
                    df["config_dir"] = entry.name
                    data_frames[req_param] = df
                except RuntimeError as e:
                    print(f"Error reading {files[0]}: {e}")
                    break
            else:
                # All required parameters were successfully loaded
                if is_derived:
                    df_cl = data_frames["CLw"]
                    df_cd = data_frames["CDw"]
                    # Merge on shared variables
                    merged = pd.merge(
                            df_cl, df_cd,
                            on=["Mach", "q", "alpha", "beta"] + list(geom_params.keys()),
                        suffixes=("_CL", "_CD")
                    )

                    merged["F"] = merged["F_CL"] / merged["F_CD"]

                    # Add config_dir manually (same for whole df)
                    merged["config_dir"] = df_cl["config_dir"].iloc[0]

                    all_data.append(merged[["Mach", "q", "alpha", "beta", "F"] + list(geom_params.keys()) + ["config_dir"]])
                else:
                    all_data.append(data_frames[param])

    if not all_data:
        raise RuntimeError(f"No data found for parameter {param} in any subdirectory.")

    return pd.concat(all_data, ignore_index=True)


def plot_vs(df, x_var, group_vars, y_label):
    if isinstance(group_vars, str):
        group_vars = [group_vars]

    unique_groups = df[group_vars].drop_duplicates()

    plt.figure(figsize=(10, 6))

    for _, group_vals in unique_groups.iterrows():
        # Build a mask for all group_var values
        mask = pd.Series(True, index=df.index)
        label_parts = []
        for var in group_vars:
            val = group_vals[var]
            mask &= df[var] == val
            label_parts.append(f"{var}={val}")
        label = ", ".join(label_parts)

        group_df = df[mask]

        if group_df.empty:
            continue

        grouped = group_df.groupby(x_var)["F"].mean().reset_index()
        plt.plot(grouped[x_var], grouped["F"], label=label)

    plt.xlabel(x_var)
    plt.ylabel(y_label)
    title = f"{y_label} vs {x_var} for groups by {', '.join(group_vars)}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def apply_filters(df, filters):
    if not filters:
        return df

    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt,
    }

    for f in filters:
        # Parse the expression
        for op_str, op_func in ops.items():
            if op_str in f:
                col, val = f.split(op_str)
                col = col.strip()
                val = val.strip()

                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in data")

                try:
                    val = float(val)
                except ValueError:
                    pass  # Leave it as string if not a number

                df = df[op_func(df[col], val)]
                break
        else:
            raise ValueError(f"Invalid filter expression: '{f}'")

    return df


def main():
    parser = argparse.ArgumentParser(description="Plot CFD parameter from multiple configuration directories.")
    parser.add_argument("model", help="Model name (prefix for directories and files, e.g. 'waverider')")
    parser.add_argument("parameter", help="Parameter to plot (e.g. CLw, CDw, L/D, q_dot)")
    parser.add_argument("--x", default="Mach", help="X-axis variable (e.g., Mach, alpha, x1, q)")
    parser.add_argument("--group_by", nargs="+", default=["alpha"],
                    help="Grouping variable(s) for multiple lines (e.g., alpha, x1, Mach alpha)")
    parser.add_argument("--filter", action="append", default=[], help="Filter expression, e.g., x1==1 or Mach>5 (can use multiple)")


    args = parser.parse_args()

    model_prefix = args.model
    param = args.parameter

    print(f"Scanning directories starting with '{model_prefix}_' for parameter '{param}'...")
    df = collect_data_across_dirs(param, model_prefix=model_prefix)

    # Apply filters
    df = apply_filters(df, args.filter)

    # Validate x axis
    if args.x not in df.columns:
        raise ValueError(f"Axis '{args.x}' not found in data columns: {df.columns.tolist()}")

    # Validate group-by columns
    for group_var in args.group_by:
        if group_var not in df.columns:
            raise ValueError(f"Group-by variable '{group_var}' not found in data columns: {df.columns.tolist()}")

    # Plot
    plot_vs(df, args.x, args.group_by, y_label=param)



if __name__ == "__main__":
    main()

