import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import operator
# Removed pykridge usage in favor of sklearn GPR for robustness
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

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
    totl_params = ["CDw", "CLw", "CFx", "CFy", "CFz", "CMl", "CMn", "CMm", "CMx", "CMy", "CMz", "CSw"]
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


# def extract_geometry_from_dir(dirname, prefix="waverider"):
#     """
#     Extracts geometry parameters from directory name.
#     Supports both integer and floating-point geometry identifiers.

#     Examples:
#         waverider_4_2_3     -> {"x1": 4.0, "x2": 2.0, "x3": 3.0}
#         waverider_0.2_1.2_3 -> {"x1": 0.2, "x2": 1.2, "x3": 3.0}
#     """
#     pattern = re.escape(prefix) + r"_(.*)"
#     match = re.match(pattern, dirname)
#     if not match:
#         return {}

#     suffix = match.group(1)
#     parts = suffix.split("_")

#     geom = {}
#     for i, part in enumerate(parts):
#         try:
#             geom[f"x{i+1}"] = float(part)
#         except ValueError:
#             geom[f"x{i+1}"] = np.nan
#     return geom

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


def plot_vs_nointerpolation(df, x_var, group_vars, y_label): #depreciated
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

def build_global_database(model_prefix):
    """
    Build a global merged DataFrame containing CLw, CDw, CMn, MaxQdotTotalQdotConvection
    across all model_prefix_* directories. Train GaussianProcessRegressor models for
    each of these outputs and return (global_df, models, scalers).

    Returns:
        global_df: pandas.DataFrame with columns including Mach,q,alpha,beta,x1...,CLw,CDw,CMn,MaxQdotTotalQdotConvection
        models: dict mapping output_name -> trained sklearn GaussianProcessRegressor
        scalers: dict with feature scaler (scaler for X) and target scalers if needed (not used here)
    """
    # parameter names to collect (these must match file param names)
    target_params = ["CLw", "CDw", "CMn", "MaxQdotTotalQdotConvection"]
    collected = {}

    # Collect data for each target param
    for param in target_params:
        try:
            collected[param] = collect_data_across_dirs(param, model_prefix=model_prefix)
            # rename column 'F' -> param to avoid collisions
            collected[param] = collected[param].rename(columns={"F": param})
        except RuntimeError as e:
            print(f"Warning: could not collect {param}: {e}")
            collected[param] = None

    # Determine geom columns from any available dataset
    geom_cols = []
    for df in collected.values():
        if df is not None:
            # geometry columns are those starting with 'x' per your extract function
            geom_cols = [c for c in df.columns if re.match(r"x\d+", c)]
            break

    # Merge datasets on Mach,q,alpha,beta and geom columns
    keycols = ["Mach", "q", "alpha", "beta"] + geom_cols
    # Start from the first available df
    dfs_to_merge = []
    for param, df in collected.items():
        if df is not None:
            # select relevant cols (may include config_dir)
            cols_needed = keycols + [param]
            # Some dfs might not include all geom cols explicitly; ensure they exist
            for g in geom_cols:
                if g not in df.columns:
                    df[g] = np.nan
            dfs_to_merge.append(df[cols_needed].copy())

    if not dfs_to_merge:
        raise RuntimeError("No parameter data found to build global database.")

    # Perform sequential merges (outer join to keep all points)
    merged = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        merged = pd.merge(merged, df, on=keycols, how="outer")

    # Ensure numeric types
    for col in ["Mach", "q", "alpha", "beta"] + geom_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Optionally drop rows with missing core variables
    merged = merged.dropna(subset=["Mach", "q", "alpha", "beta"])

    # Fill geometry NaNs with median of each geom column (so model can use them)
    for g in geom_cols:
        if merged[g].isna().any():
            median = merged[g].median(skipna=True)
            merged[g] = merged[g].fillna(median)

    # Save global database to CSV
    outcsv = f"global_database_{model_prefix}.csv"
    merged.to_csv(outcsv, index=False)
    print(f"Global database written to: {outcsv}")

    # === Prepare training data ===
    feature_cols = ["Mach", "q", "alpha", "beta"] + geom_cols
    X = merged[feature_cols].values

    # We'll train one GPR per target that exists (non-all-nan)
    models = {}
    scalers = {}

    # standardize features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    scalers["X_scaler"] = X_scaler

    for target in target_params:
        if target not in merged.columns:
            continue
        y = merged[target].values
        # drop rows where target is nan
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            print(f"Skipping training for {target}: not enough valid samples ({mask.sum()})")
            continue
        X_train = X_scaled[mask]
        y_train = y[mask]

        # Simple GPR kernel: constant * Matern + white noise
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        print(f"Training GPR for {target} on {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        gpr.fit(X_train, y_train)
        models[target] = gpr

    return merged, models, scalers, feature_cols

def interpolate(models, scalers, feature_cols, query_df):
    """
    Given trained models and a dataframe of query points (with feature_cols present),
    return query_df with added predicted columns for each model key.

    Args:
        models: dict target_name -> trained sklearn regressor
        scalers: dict containing X_scaler used for features
        feature_cols: list of feature column names expected in query_df
        query_df: pandas.DataFrame containing at least feature_cols

    Returns:
        pandas.DataFrame: copy of query_df with added columns "<target>_pred" (and "<target>_std" if available)
    """
    out = query_df.copy()
    # Ensure numeric and fill missing geom columns with medians if needed
    for c in feature_cols:
        if c not in out.columns:
            # Fill with median from scalers? Here, attempt to use 0
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    Xq = out[feature_cols].values
    Xq_scaled = scalers["X_scaler"].transform(Xq)

    for target, model in models.items():
        # predict with std
        try:
            y_pred, y_std = model.predict(Xq_scaled, return_std=True)
        except TypeError:
            # Some sklearn versions might not support return_std; fall back
            y_pred = model.predict(Xq_scaled)
            y_std = np.full_like(y_pred, np.nan)
        out[f"{target}_pred"] = y_pred
        out[f"{target}_std"] = y_std

    return out


# Small helper to map user requested plotting param to trained model key
_param_synonyms = {
    "q_dot": "MaxQdotTotalQdotConvection",
    "qdot": "MaxQdotTotalQdotConvection",
    "Cm": "CMn",
    "Cl": "CLw",
    "Cd": "CDw",
    # exact keys also map to themselves implicitly
}

def _map_param_to_model_key(param):
    if param in _param_synonyms:
        return _param_synonyms[param]
    return param  # assume user passed the exact key like "CLw"

# --- Updated plot_vs to optionally overlay interpolation ---
def plot_vs(df, x_var, group_vars, y_label, models=None, scalers=None, feature_cols=None):
    if isinstance(group_vars, str):
        group_vars = [group_vars]

    unique_groups = df[group_vars].drop_duplicates()

    plt.figure(figsize=(10, 6))

    # plot data lines
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
        plt.plot(grouped[x_var], grouped["F"], label=label + " (data)")

    # If models provided and one matches the plotted y_label, overlay predictions
    if models is not None and scalers is not None and feature_cols is not None:
        model_key = _map_param_to_model_key(y_label)
        if model_key in models:
            # For each group, construct query points varying x_var across min/max in df
            x_min = df[x_var].min()
            x_max = df[x_var].max()
            x_grid = np.linspace(x_min, x_max, 200)

            for _, group_vals in unique_groups.iterrows():
                # Construct query dataframe
                qdict = {}
                for col in feature_cols:
                    if col == x_var:
                        qdict[col] = x_grid
                    elif col in group_vals.index:
                        # if group var is part of features and available in this group, use it
                        if col in group_vars:
                            qdict[col] = float(group_vals[col])
                        else:
                            # if group var is not present in feature set but we have a fixed column value in group_vals, use that
                            qdict[col] = float(group_vals.get(col, df[col].median()))
                    else:
                        # Use median of that column from df if available, else zero
                        if col in df.columns:
                            qdict[col] = float(df[col].median())
                        else:
                            qdict[col] = 0.0

                query_df = pd.DataFrame(qdict)
                preds = interpolate(models, scalers, feature_cols, query_df)
                plt.plot(x_grid, preds[f"{model_key}_pred"], linestyle="--", label=", ".join([f"{v}" for v in group_vals.values]) + " (interp)")

    plt.xlabel(x_var)
    plt.ylabel(y_label)
    title = f"{y_label} vs {x_var} for groups by {', '.join(group_vars)}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drag_polar(df, group_vars, model_prefix, models=None, scalers=None, feature_cols=None, filters=None):
    """
    Plot drag polar (CDw vs CLw) for given group-by variables.
    - model_prefix: prefix used for directories (e.g., 'waverider')
    - group_vars: grouping variables to separate polars
    - models, scalers, feature_cols: optional trained models for overlay
    - filters: optional list of filter expressions to apply to the merged polar dataset
    """
    print("Plotting drag polar (CDw vs CLw)...")

    if isinstance(group_vars, str):
        group_vars = [group_vars]

    # Collect CLw and CDw from directories with correct model_prefix
    try:
        df_cl = collect_data_across_dirs("CLw", model_prefix=model_prefix)
        df_cd = collect_data_across_dirs("CDw", model_prefix=model_prefix)
    except RuntimeError as e:
        print(f"Error collecting CLw/CDw for polar: {e}")
        return

    if df_cl is None or df_cd is None or df_cl.empty or df_cd.empty:
        print("No CLw or CDw data found; cannot plot drag polar.")
        return

    # Rename the 'F' column to CLw and CDw respectively (collect_data_across_dirs uses 'F')
    df_cl = df_cl.rename(columns={"F": "CLw"})
    df_cd = df_cd.rename(columns={"F": "CDw"})

    # Determine geom cols present in either df
    geom_cols = sorted({c for c in list(df_cl.columns) + list(df_cd.columns) if re.match(r"x\d+", c)})
    keycols = ["Mach", "q", "alpha", "beta"] + geom_cols

    # Ensure geom columns exist in both frames
    for g in geom_cols:
        if g not in df_cl.columns:
            df_cl[g] = np.nan
        if g not in df_cd.columns:
            df_cd[g] = np.nan

    # Merge on keycols (outer to keep everything), then drop rows missing CL or CD
    merged = pd.merge(df_cl[keycols + ["CLw"]], df_cd[keycols + ["CDw"]], on=keycols, how="outer")

    # Apply optional filters (so CLI --filter affects polar)
    if filters:
        merged = apply_filters(merged, filters)

    # Drop rows missing CLw or CDw
    merged = merged.dropna(subset=["CLw", "CDw"])
    if merged.empty:
        print("Merged CL/CD dataset is empty after filtering — nothing to plot.")
        return

    # Use the passed `df` (which was collected for the requested parameter) to derive groups;
    # but prefer grouping values from merged if present
    if not all(var in merged.columns for var in group_vars):
        # If group_vars are not in merged, try to fall back to values from df
        for var in group_vars:
            if var not in merged.columns and var in df.columns:
                merged[var] = df[var].median()  # best effort: use median if no group info

    unique_groups = merged[group_vars].drop_duplicates()

    plt.figure(figsize=(10, 6))

    # Plot all points grouped
    for _, group_vals in unique_groups.iterrows():
        # Build mask on merged for the group
        mask = pd.Series(True, index=merged.index)
        label_parts = []
        for var in group_vars:
            val = group_vals[var]
            mask &= merged[var] == val
            label_parts.append(f"{var}={val}")
        label = ", ".join(label_parts)

        group_df = merged[mask]
        if group_df.empty:
            continue

        # Sort by CL for nicer polars
        group_df = group_df.sort_values("CLw")
        plt.plot(group_df["CLw"], group_df["CDw"], "o", label=label + " (data)")

        # Interpolation overlay if models exist
        # Need models for both CLw and CDw to overlay CD vs CL predictions
        if models is None or scalers is None or feature_cols is None:
            continue
        if "CLw" not in models or "CDw" not in models:
            # maybe synonyms were used; check synonyms mapping
            mk_cl = _map_param_to_model_key("CLw")
            mk_cd = _map_param_to_model_key("CDw")
            if mk_cl not in models or mk_cd not in models:
                # not enough models for overlay
                continue

        # Create query points along CL (but Kriging predicts from state vars; we need to sweep one variable)
        # We'll sweep the x-axis variable CLw by using the actual states from group_df (best-effort)
        # Approach: use the group's existing feature rows (feature_cols) as query points to predict CLw & CDw
        # Then plot predicted CDw vs predicted CLw
        # If feature_cols are missing in merged, fill with medians
        qdf = merged.loc[mask, feature_cols].copy()
        if qdf.empty:
            # fallback: build query grid from median features, varying Mach if group had Mach
            base = {c: merged[c].median() if c in merged.columns else 0.0 for c in feature_cols}
            # attempt a Mach sweep if Mach is in group_vars else use 50 points constant
            if "Mach" in feature_cols:
                mach_vals = np.linspace(merged["Mach"].min(), merged["Mach"].max(), 100)
                qdf = pd.DataFrame([{**base, "Mach": m} for m in mach_vals])
            else:
                qdf = pd.DataFrame([base for _ in range(100)])

        # Ensure numeric and no NaNs
        for c in feature_cols:
            if c not in qdf.columns:
                qdf[c] = merged[c].median() if c in merged.columns else 0.0
            qdf[c] = pd.to_numeric(qdf[c], errors="coerce").fillna(merged[c].median() if c in merged.columns else 0.0)

        # Predict
        preds = interpolate(models, scalers, feature_cols, qdf)

        # Predicted columns named 'CLw_pred' and 'CDw_pred'
        if "CLw_pred" in preds.columns and "CDw_pred" in preds.columns:
            # Sort by predicted CL for line plotting
            sort_idx = np.argsort(preds["CLw_pred"].values)
            plt.plot(preds["CLw_pred"].values[sort_idx], preds["CDw_pred"].values[sort_idx], "--", label=label + " (interp)")

    plt.xlabel("CL")
    plt.ylabel("CD")
    plt.title("Drag Polar (CD vs CL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot CFD parameter from multiple configuration directories.")
    parser.add_argument("model", help="Model name (prefix for directories and files, e.g. 'waverider')")
    parser.add_argument("parameter", help="Parameter to plot (e.g. CLw, CDw, L/D, q_dot)")
    parser.add_argument("--x", default="Mach", help="X-axis variable (e.g., Mach, alpha, x1, q)")
    parser.add_argument("--group_by", nargs="+", default=["alpha"],
                    help="Grouping variable(s) for multiple lines (e.g., alpha, x1, Mach alpha)")
    parser.add_argument("--filter", action="append", default=[], help="Filter expression, e.g., x1==1 or Mach>5")
    parser.add_argument("--polar", action="store_true", help="Plot drag polar (CD vs CL) instead of parameter vs X")

    args = parser.parse_args()

    model_prefix = args.model
    param = args.parameter

    # Collect data for plotting variable
    print(f"Scanning directories starting with '{model_prefix}_' for parameter '{param}'...")
    df = collect_data_across_dirs(param, model_prefix=model_prefix)

    # Apply filters
    df = apply_filters(df, args.filter)

    # Build kriging / GPR models
    global_df, models, scalers, feature_cols = build_global_database(model_prefix)

    # Polar plot or normal plot
    if args.polar:
        plot_drag_polar(df, args.group_by, model_prefix=model_prefix, models=models, scalers=scalers, feature_cols=feature_cols, filters=args.filter)
    else:
        plot_vs(df, args.x, args.group_by, y_label=param, models=models, scalers=scalers, feature_cols=feature_cols)



if __name__ == "__main__":
    main()
