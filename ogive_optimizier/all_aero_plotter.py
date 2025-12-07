import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def find_file(directory, model_name, param):
    totl_params = ["CDw", "CLw", "CFx", "CFy", "CFz", "CMl", "CMn", "CMm", "CMx", "CMy", "CMz", "CSw"]
    if param in totl_params:
        pattern = f"{model_name}.{param}.Totl.dat"
    else:
        pattern = f"{model_name}.{param}.dat"
    
    files = glob.glob(os.path.join(directory, pattern))
    return files[0] if files else None

def load_parameter_data(directory, model_name, param):
    """Load data for a single parameter from the ogive directory."""
    filepath = find_file(directory, model_name, param)
    if not filepath:
        raise RuntimeError(f"Could not find file for parameter {param}")
    
    df = parse_cfd_file(filepath)
    df = df.rename(columns={"F": param})
    return df

def main():
    # Configuration
    directory = "ogive"
    model_name = "ogive"
    mach_numbers = [2, 3, 4, 5, 6, 7, 8]
    
    # Check if directory exists
    if not os.path.exists(directory):
        raise RuntimeError(f"Directory '{directory}' not found in current folder")
    
    print(f"Loading data from '{directory}' directory...")
    
    # Load all required parameters
    try:
        df_cl = load_parameter_data(directory, model_name, "CLw")
        df_cd = load_parameter_data(directory, model_name, "CDw")
        df_cm = load_parameter_data(directory, model_name, "CMn")
        df_cmx = load_parameter_data(directory, model_name, "CMx")
        df_cmy = load_parameter_data(directory, model_name, "CMy")
        df_cmz = load_parameter_data(directory, model_name, "CMz")
        df_qdot = load_parameter_data(directory, model_name, "MaxQdotTotalQdotConvection")
    except RuntimeError as e:
        print(f"Error loading data: {e}")
        return
    
    # Merge all dataframes
    merged = df_cl.copy()
    merged = pd.merge(merged, df_cd[["Mach", "q", "alpha", "beta", "CDw"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    merged = pd.merge(merged, df_cm[["Mach", "q", "alpha", "beta", "CMn"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    merged = pd.merge(merged, df_cmx[["Mach", "q", "alpha", "beta", "CMx"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    merged = pd.merge(merged, df_cmy[["Mach", "q", "alpha", "beta", "CMy"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    merged = pd.merge(merged, df_cmz[["Mach", "q", "alpha", "beta", "CMz"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    merged = pd.merge(merged, df_qdot[["Mach", "q", "alpha", "beta", "MaxQdotTotalQdotConvection"]], 
                      on=["Mach", "q", "alpha", "beta"], how="outer")
    
    # Calculate L/D
    merged["L/D"] = merged["CLw"] / merged["CDw"]
    
    # Filter for specified Mach numbers
    merged = merged[merged["Mach"].isin(mach_numbers)]
    
    print(f"Loaded {len(merged)} data points")
    print(f"Mach numbers found: {sorted(merged['Mach'].unique())}")
    print(f"Alpha range: {merged['alpha'].min()} to {merged['alpha'].max()}")
    
    # Create combined plots - 3x3 grid for 9 plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Ogive Aerodynamic Data - Combined View', fontsize=16)
    
    # Plot 1: CL vs AoA (all Mach numbers on one plot)
    ax = axes[0, 0]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        ax.plot(data["alpha"], data["CLw"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('AoA (deg)')
    ax.set_ylabel('CL')
    ax.set_title('CL vs AoA')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: CD vs CL (all on one plot)
    ax = axes[0, 1]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        ax.plot(data["CLw"], data["CDw"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('CL')
    ax.set_ylabel('CD')
    ax.set_title('Drag Polar (CD vs CL)')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: L/D vs AoA
    ax = axes[0, 2]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        ax.plot(data["alpha"], data["L/D"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('AoA (deg)')
    ax.set_ylabel('L/D')
    ax.set_title('L/D vs AoA')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: CM_cg vs AoA
    ax = axes[1, 0]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        ax.plot(data["alpha"], data["CMn"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('AoA (deg)')
    ax.set_ylabel('CM_cg')
    ax.set_title('CM_cg vs AoA')
    ax.legend()
    ax.grid(True)
    
    # Plot 5: CM_cg vs CL
    ax = axes[1, 1]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        ax.plot(data["CLw"], data["CMn"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('CL')
    ax.set_ylabel('CM_cg')
    ax.set_title('CM_cg vs CL')
    ax.legend()
    ax.grid(True)
    
    # Plot 6: q_dot vs dynamic pressure
    ax = axes[1, 2]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("q")
        ax.plot(data["q"], data["MaxQdotTotalQdotConvection"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('Dynamic Pressure (q)')
    ax.set_ylabel('q_dot')
    ax.set_title('q_dot vs Dynamic Pressure')
    ax.legend()
    ax.grid(True)
    
    # Plot 7: CMx vs CL
    ax = axes[2, 0]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        ax.plot(data["CLw"], data["CMx"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('CL')
    ax.set_ylabel('CMx')
    ax.set_title('CMx vs CL')
    ax.legend()
    ax.grid(True)
    
    # Plot 8: CMy vs CL
    ax = axes[2, 1]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        ax.plot(data["CLw"], data["CMy"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('CL')
    ax.set_ylabel('CMy')
    ax.set_title('CMy vs CL')
    ax.legend()
    ax.grid(True)
    
    # Plot 9: CMz vs CL
    ax = axes[2, 2]
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        ax.plot(data["CLw"], data["CMz"], 'o:', label=f'Mach {mach}')
    ax.set_xlabel('CL')
    ax.set_ylabel('CMz')
    ax.set_title('CMz vs CL')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Now create individual plots
    print("\nGenerating individual plots...")
    
    # Individual Plot 1: CL vs AoA
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        plt.plot(data["alpha"], data["CLw"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('AoA (deg)', fontsize=12)
    plt.ylabel('CL', fontsize=12)
    plt.title('CL vs AoA', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 2: CD vs CL
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        plt.plot(data["CLw"], data["CDw"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('CL', fontsize=12)
    plt.ylabel('CD', fontsize=12)
    plt.title('Drag Polar (CD vs CL)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 3: L/D vs AoA
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        plt.plot(data["alpha"], data["L/D"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('AoA (deg)', fontsize=12)
    plt.ylabel('L/D', fontsize=12)
    plt.title('L/D vs AoA', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 4: CM_cg vs AoA
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("alpha")
        plt.plot(data["alpha"], data["CMn"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('AoA (deg)', fontsize=12)
    plt.ylabel('CM_cg', fontsize=12)
    plt.title('CM_cg vs AoA', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 5: CM_cg vs CL
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        plt.plot(data["CLw"], data["CMn"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('CL', fontsize=12)
    plt.ylabel('CM_cg', fontsize=12)
    plt.title('CM_cg vs CL', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 6: q_dot vs dynamic pressure
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("q")
        plt.plot(data["q"], data["MaxQdotTotalQdotConvection"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('Dynamic Pressure (q)', fontsize=12)
    plt.ylabel('q_dot', fontsize=12)
    plt.title('q_dot vs Dynamic Pressure', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 7: CMx vs CL
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        plt.plot(data["CLw"], data["CMx"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('CL', fontsize=12)
    plt.ylabel('CMx', fontsize=12)
    plt.title('CMx vs CL', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 8: CMy vs CL
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        plt.plot(data["CLw"], data["CMy"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('CL', fontsize=12)
    plt.ylabel('CMy', fontsize=12)
    plt.title('CMy vs CL', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Individual Plot 9: CMz vs CL
    plt.figure(figsize=(10, 6))
    for mach in mach_numbers:
        data = merged[merged["Mach"] == mach].sort_values("CLw")
        plt.plot(data["CLw"], data["CMz"], 'o:', label=f'Mach {mach}', linewidth=2, markersize=6)
    plt.xlabel('CL', fontsize=12)
    plt.ylabel('CMz', fontsize=12)
    plt.title('CMz vs CL', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nAll plotting complete!")

if __name__ == "__main__":
    main()
