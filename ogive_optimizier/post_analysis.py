"""
post_analysis.py - Analyze all solved trajectory cases to identify overlooked high-performance vehicles

This script examines trajectory results across all vehicle configurations to find vehicles that may
perform better with an improved optimizer, focusing on:
- High L/D capability
- High initial energy tolerance (nose radius -> max Mach)
- Initial conditions and constraints
"""

import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
from pathlib import Path

# ====================== NOSE RADIUS TO MAX MACH CORRELATION ======================
def estimate_max_mach_from_nose_radius(nose_radius_m):
    """
    Estimate maximum allowable Mach number based on nose radius.
    Larger nose radius -> better heating distribution -> higher Mach tolerance
    
    Based on typical hypersonic vehicle design rules:
    - Sharp noses (r < 5mm): Limited to Mach 5-6
    - Medium noses (5-15mm): Can handle Mach 6-8
    - Blunt noses (r > 15mm): Can handle Mach 8-10+
    
    This is a simplified model - actual capability depends on materials and duration.
    """
    r_mm = nose_radius_m * 1000  # Convert to mm
    
    if r_mm < 5:
        max_mach = 5.0 + 1.0 * (r_mm / 5.0)  # 5.0-6.0
    elif r_mm < 15:
        max_mach = 6.0 + 2.0 * ((r_mm - 5) / 10.0)  # 6.0-8.0
    else:
        max_mach = 8.0 + 2.0 * min((r_mm - 15) / 35.0, 1.0)  # 8.0-10.0
    
    return max_mach

# ====================== GEOMETRY PARSING ======================
def parse_geometry_from_dirname(dirname):
    """
    Extract geometry parameters from directory name.
    Expected format: ogive-L{length}-R{radius}-nr{nose_radius}-zs{shoulder}-zc{centroid}
    
    Example: ogive-L1.000-R0.050-nr1.0mm-zs0.657-zc-0.035
    """
    geom = {}
    
    # Extract length
    match = re.search(r'L([\d.]+)', dirname)
    if match:
        geom['length'] = float(match.group(1))
    
    # Extract radius
    match = re.search(r'R([\d.]+)', dirname)
    if match:
        geom['radius'] = float(match.group(1))
    
    # Extract nose radius (in mm)
    match = re.search(r'nr([\d.]+)mm', dirname)
    if match:
        geom['nose_radius_mm'] = float(match.group(1))
        geom['nose_radius_m'] = float(match.group(1)) / 1000.0
    
    # Extract shoulder position
    match = re.search(r'zs([-\d.]+)', dirname)
    if match:
        geom['z_shoulder'] = float(match.group(1))
    
    # Extract centroid position
    match = re.search(r'zc([-\d.]+)', dirname)
    if match:
        geom['z_centroid'] = float(match.group(1))
    
    return geom

# ====================== TRAJECTORY FILE PARSING ======================
def parse_trajectory_file(filepath):
    """Parse a .traj file and extract key metrics."""
    try:
        # Read the file, skipping comments
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                # Skip header lines with text
                if 'time' in line.lower() or '(s)' in line:
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 14:  # Ensure we have all columns
                        data.append(values)
                except ValueError:
                    continue
        
        if not data:
            return None
        
        # Convert to numpy array
        data = np.array(data)
        
        # Extract columns (from the .traj format)
        t = data[:, 0]          # time (s)
        v = data[:, 1]          # velocity (m/s)
        M = data[:, 2]          # Mach
        downrange = data[:, 3]  # downrange (m)
        h = data[:, 4]          # altitude (m)
        rho = data[:, 5]        # density (kg/m³)
        L = data[:, 6]          # lift (N)
        D = data[:, 7]          # drag (N)
        q_dot = data[:, 8]      # heat flux (W/m²)
        alpha = data[:, 9]      # angle of attack (deg)
        gamma = data[:, 10]     # flight path angle (deg)
        LD = data[:, 11]        # L/D ratio
        q_dyn = data[:, 12]     # dynamic pressure (Pa)
        mass = data[:, 13]      # mass (kg)
        
        # Calculate key metrics
        metrics = {
            # Initial conditions
            'initial_mach': M[0],
            'initial_altitude': h[0],
            'initial_velocity': v[0],
            'initial_q_dot': q_dot[0],
            'initial_gamma': gamma[0],
            
            # Performance
            'final_range_km': downrange[-1] / 1000.0,
            'max_q_dot': np.max(q_dot),
            'max_altitude': np.max(h),
            'flight_duration': t[-1],
            
            # L/D analysis
            'max_LD': np.max(LD),
            'mean_LD': np.mean(LD[LD > 0]),  # Mean of positive L/D
            'alpha_at_max_LD': alpha[np.argmax(LD)],
            'mach_at_max_LD': M[np.argmax(LD)],
            
            # Energy analysis
            'initial_specific_energy': v[0]**2 / 2 + 9.81 * h[0],  # J/kg
            'final_specific_energy': v[-1]**2 / 2 + 9.81 * h[-1],
            
            # Alpha usage
            'mean_alpha': np.mean(alpha),
            'max_alpha': np.max(alpha),
            'min_alpha': np.min(alpha),
            
            # Trajectory data
            'trajectory': {
                't': t, 'M': M, 'h': h, 'v': v, 'alpha': alpha, 
                'LD': LD, 'q_dot': q_dot, 'downrange': downrange
            }
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

# ====================== MAIN ANALYSIS ======================
def analyze_all_cases(base_dir='.', pattern='ogive-*'):
    """
    Scan all directories matching pattern and analyze their trajectories.
    """
    results = []
    
    # Find all matching directories
    dirs = [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('ogive-')]
    
    print(f"Found {len(dirs)} vehicle configuration directories")
    print("="*80)
    
    for dirname in sorted(dirs):
        dir_path = os.path.join(base_dir, dirname)
        
        # Parse geometry
        geom = parse_geometry_from_dirname(dirname)
        
        # Find trajectory file
        traj_file = os.path.join(dir_path, 'best_trajectory.traj')
        
        if not os.path.exists(traj_file):
            print(f"⊘ {dirname}: No trajectory file found")
            continue
        
        # Parse trajectory
        metrics = parse_trajectory_file(traj_file)
        
        if metrics is None:
            print(f"⊘ {dirname}: Failed to parse trajectory")
            continue
        
        # Estimate maximum Mach capability from nose radius
        if 'nose_radius_m' in geom:
            geom['estimated_max_mach'] = estimate_max_mach_from_nose_radius(geom['nose_radius_m'])
            geom['mach_margin'] = geom['estimated_max_mach'] - metrics['initial_mach']
        else:
            geom['estimated_max_mach'] = np.nan
            geom['mach_margin'] = np.nan
        
        # Combine geometry and metrics
        result = {**geom, **metrics}
        result['dirname'] = dirname
        
        results.append(result)
        
        # Print summary
        print(f"✓ {dirname}")
        print(f"    Geometry: L={geom.get('length', '?'):.3f}m, R={geom.get('radius', '?'):.3f}m, "
              f"nr={geom.get('nose_radius_mm', '?'):.1f}mm")
        print(f"    Performance: Range={metrics['final_range_km']:.1f} km, "
              f"Max L/D={metrics['max_LD']:.2f} @ α={metrics['alpha_at_max_LD']:.1f}°")
        print(f"    Initial: M={metrics['initial_mach']:.1f}, h={metrics['initial_altitude']/1000:.1f} km, "
              f"q̇={metrics['initial_q_dot']/1e6:.2f} MW/m²")
        print(f"    Mach potential: Est. max M={geom.get('estimated_max_mach', np.nan):.1f}, "
              f"Margin={geom.get('mach_margin', np.nan):.1f}")
        print()
    
    return pd.DataFrame(results)

# ====================== RANKING AND IDENTIFICATION ======================
def identify_overlooked_vehicles(df):
    """
    Identify vehicles that may perform better with improved optimization.
    
    Key indicators:
    1. High L/D capability (efficient flight)
    2. Large Mach margin (can start with more energy)
    3. Low initial q_dot (thermal margin)
    4. High specific energy capability
    """
    
    print("\n" + "="*80)
    print("OVERLOOKED VEHICLE ANALYSIS")
    print("="*80)
    
    # Remove trajectories with missing data
    df_clean = df.dropna(subset=['max_LD', 'estimated_max_mach', 'mach_margin'])
    
    if len(df_clean) == 0:
        print("No valid data to analyze")
        return
    
    # Calculate composite scores
    
    # 1. L/D Performance Score (0-100)
    LD_score = 100 * (df_clean['max_LD'] - df_clean['max_LD'].min()) / \
               (df_clean['max_LD'].max() - df_clean['max_LD'].min())
    
    # 2. Energy Potential Score (based on Mach margin, 0-100)
    mach_margin_score = 100 * np.clip(df_clean['mach_margin'] / 3.0, 0, 1)  # 3 Mach margin = 100 points
    
    # 3. Thermal Margin Score (lower initial q_dot is better, 0-100)
    thermal_score = 100 * (1 - (df_clean['initial_q_dot'] - df_clean['initial_q_dot'].min()) / 
                           (df_clean['initial_q_dot'].max() - df_clean['initial_q_dot'].min()))
    
    # 4. Current Performance (normalized range, 0-100)
    range_score = 100 * (df_clean['final_range_km'] - df_clean['final_range_km'].min()) / \
                  (df_clean['final_range_km'].max() - df_clean['final_range_km'].min())
    
    # Composite "potential" score (high L/D + high energy margin + good thermal margin)
    potential_score = 0.4 * LD_score + 0.4 * mach_margin_score + 0.2 * thermal_score
    
    # "Underperformance" indicator (high potential but low current range)
    underperformance = potential_score - range_score
    
    df_clean = df_clean.copy()
    df_clean['LD_score'] = LD_score.values
    df_clean['mach_margin_score'] = mach_margin_score.values
    df_clean['thermal_score'] = thermal_score.values
    df_clean['range_score'] = range_score.values
    df_clean['potential_score'] = potential_score.values
    df_clean['underperformance'] = underperformance.values
    
    # Sort by underperformance (highest = most overlooked)
    df_sorted = df_clean.sort_values('underperformance', ascending=False)
    
    print("\n🔍 TOP 10 POTENTIALLY OVERLOOKED VEHICLES")
    print("   (High potential but underperforming in current optimization)")
    print("-"*80)
    print(f"{'Rank':<5} {'Vehicle':<40} {'Potential':<10} {'Current':<10} {'Gap':<8}")
    print(f"{'':5} {'':40} {'Score':<10} {'Range(km)':<10} {'Score':<8}")
    print("-"*80)
    
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"{i+1:<5} {row['dirname']:<40} {row['potential_score']:>8.1f}  {row['final_range_km']:>8.1f}  {row['underperformance']:>6.1f}")
        print(f"{'':5} L/D={row['max_LD']:.2f} @ α={row['alpha_at_max_LD']:.1f}°, "
              f"M_margin={row['mach_margin']:.1f}, "
              f"q̇_init={row['initial_q_dot']/1e6:.2f} MW/m²")
        print()
    
    print("\n🏆 TOP 10 VEHICLES BY MAX L/D")
    print("-"*80)
    df_by_LD = df_clean.sort_values('max_LD', ascending=False)
    print(f"{'Rank':<5} {'Vehicle':<40} {'Max L/D':<10} {'@ Alpha':<10} {'@ Mach':<10}")
    print("-"*80)
    for i, (idx, row) in enumerate(df_by_LD.head(10).iterrows()):
        print(f"{i+1:<5} {row['dirname']:<40} {row['max_LD']:>8.2f}  {row['alpha_at_max_LD']:>8.1f}°  {row['mach_at_max_LD']:>8.1f}")
    
    print("\n⚡ TOP 10 VEHICLES BY ENERGY POTENTIAL (Mach Margin)")
    print("-"*80)
    df_by_mach = df_clean.sort_values('mach_margin', ascending=False)
    print(f"{'Rank':<5} {'Vehicle':<40} {'Est.MaxM':<10} {'Current M':<10} {'Margin':<10}")
    print("-"*80)
    for i, (idx, row) in enumerate(df_by_mach.head(10).iterrows()):
        print(f"{i+1:<5} {row['dirname']:<40} {row['estimated_max_mach']:>8.1f}  {row['initial_mach']:>8.1f}  {row['mach_margin']:>8.1f}")
    
    print("\n🎯 CURRENT TOP 10 BY RANGE")
    print("-"*80)
    df_by_range = df_clean.sort_values('final_range_km', ascending=False)
    print(f"{'Rank':<5} {'Vehicle':<40} {'Range(km)':<12} {'L/D':<8} {'M_init':<8}")
    print("-"*80)
    for i, (idx, row) in enumerate(df_by_range.head(10).iterrows()):
        print(f"{i+1:<5} {row['dirname']:<40} {row['final_range_km']:>10.1f}  {row['max_LD']:>6.2f}  {row['initial_mach']:>6.1f}")
    
    return df_sorted

# ====================== VISUALIZATION ======================
def plot_vehicle_analysis(df, output_dir='.'):
    """Create visualization plots for vehicle analysis."""
    
    df_clean = df.dropna(subset=['max_LD', 'estimated_max_mach', 'final_range_km'])
    
    if len(df_clean) == 0:
        print("No valid data to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: L/D vs Range
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(df_clean['max_LD'], df_clean['final_range_km'], 
                          c=df_clean['initial_mach'], s=100, alpha=0.6, cmap='viridis')
    ax1.set_xlabel('Maximum L/D', fontsize=11)
    ax1.set_ylabel('Final Range (km)', fontsize=11)
    ax1.set_title('Range vs L/D Performance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Initial Mach', fontsize=9)
    
    # Plot 2: Mach Margin vs Range
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(df_clean['mach_margin'], df_clean['final_range_km'],
                          c=df_clean['max_LD'], s=100, alpha=0.6, cmap='plasma')
    ax2.set_xlabel('Mach Margin (Est. Max - Current)', fontsize=11)
    ax2.set_ylabel('Final Range (km)', fontsize=11)
    ax2.set_title('Range vs Energy Potential', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Max L/D', fontsize=9)
    
    # Plot 3: Nose Radius vs Max Mach Estimate
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(df_clean['nose_radius_mm'], df_clean['estimated_max_mach'],
               s=100, alpha=0.6, color='green')
    ax3.scatter(df_clean['nose_radius_mm'], df_clean['initial_mach'],
               s=100, alpha=0.6, color='red', marker='x', label='Actual Initial Mach')
    ax3.set_xlabel('Nose Radius (mm)', fontsize=11)
    ax3.set_ylabel('Mach Number', fontsize=11)
    ax3.set_title('Nose Radius vs Mach Capability', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: L/D vs Alpha
    ax4 = plt.subplot(2, 3, 4)
    scatter4 = ax4.scatter(df_clean['alpha_at_max_LD'], df_clean['max_LD'],
                          c=df_clean['final_range_km'], s=100, alpha=0.6, cmap='coolwarm')
    ax4.set_xlabel('Alpha at Max L/D (deg)', fontsize=11)
    ax4.set_ylabel('Maximum L/D', fontsize=11)
    ax4.set_title('L/D vs Optimal Angle of Attack', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Range (km)', fontsize=9)
    
    # Plot 5: Heat Flux vs Range
    ax5 = plt.subplot(2, 3, 5)
    scatter5 = ax5.scatter(df_clean['max_q_dot']/1e6, df_clean['final_range_km'],
                          c=df_clean['initial_mach'], s=100, alpha=0.6, cmap='hot')
    ax5.set_xlabel('Max Heat Flux (MW/m²)', fontsize=11)
    ax5.set_ylabel('Final Range (km)', fontsize=11)
    ax5.set_title('Range vs Thermal Loading', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar5 = plt.colorbar(scatter5, ax=ax5)
    cbar5.set_label('Initial Mach', fontsize=9)
    
    # Plot 6: Potential Score vs Current Range
    if 'potential_score' in df_clean.columns:
        ax6 = plt.subplot(2, 3, 6)
        scatter6 = ax6.scatter(df_clean['potential_score'], df_clean['final_range_km'],
                              c=df_clean['underperformance'], s=100, alpha=0.6, cmap='RdYlGn_r')
        ax6.set_xlabel('Potential Score', fontsize=11)
        ax6.set_ylabel('Final Range (km)', fontsize=11)
        ax6.set_title('Potential vs Actual Performance', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        cbar6 = plt.colorbar(scatter6, ax=ax6)
        cbar6.set_label('Underperformance', fontsize=9)
        
        # Add diagonal line (ideal performance)
        range_min, range_max = df_clean['final_range_km'].min(), df_clean['final_range_km'].max()
        ax6.plot([0, 100], [range_min, range_max], 'k--', alpha=0.3, label='Ideal')
        ax6.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'vehicle_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Analysis plots saved to: {output_path}")
    plt.close()

# ====================== EXPORT RECOMMENDATIONS ======================
def export_recommendations(df_sorted, output_dir='.'):
    """Export recommended vehicles for re-optimization to CSV."""
    
    if 'underperformance' not in df_sorted.columns:
        print("Run identify_overlooked_vehicles first")
        return
    
    # Select top candidates
    top_candidates = df_sorted.head(20).copy()
    
    # Select relevant columns
    export_cols = [
        'dirname', 'length', 'radius', 'nose_radius_mm', 'z_shoulder', 'z_centroid',
        'final_range_km', 'max_LD', 'alpha_at_max_LD', 'mach_at_max_LD',
        'initial_mach', 'estimated_max_mach', 'mach_margin',
        'max_q_dot', 'initial_q_dot',
        'potential_score', 'underperformance'
    ]
    
    export_cols = [col for col in export_cols if col in top_candidates.columns]
    
    output_path = os.path.join(output_dir, 'recommended_vehicles_for_reoptimization.csv')
    top_candidates[export_cols].to_csv(output_path, index=False)
    
    print(f"\n💾 Top candidates exported to: {output_path}")

# ====================== MAIN ======================
if __name__ == '__main__':
    print("="*80)
    print("TRAJECTORY POST-ANALYSIS: Identifying Overlooked High-Performance Vehicles")
    print("="*80)
    print()
    
    # Analyze all cases
    df = analyze_all_cases()
    
    if len(df) == 0:
        print("\n❌ No trajectory data found!")
    else:
        print(f"\n✓ Successfully analyzed {len(df)} vehicle configurations")
        
        # Identify overlooked vehicles
        df_sorted = identify_overlooked_vehicles(df)
        
        # Create visualizations
        plot_vehicle_analysis(df_sorted if df_sorted is not None else df)
        
        # Export recommendations
        if df_sorted is not None:
            export_recommendations(df_sorted)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
