import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Read trajectory file
# -----------------------------
traj_file = "old_veh_trajectory.traj"

downrange = []
altitude = []

with open(traj_file, "r") as f:
    lines = f.readlines()

# Skip first two header lines
for line in lines[2:]:
    if not line.strip():
        continue

    clean_line = line.replace("...", "")
    cols = clean_line.split()

    try:
        downrange.append(float(cols[3]))  # meters
        altitude.append(float(cols[4]))   # meters
    except (IndexError, ValueError):
        continue

downrange = np.array(downrange).reshape(-1, 1)
altitude = np.array(altitude).reshape(-1, 1)

# -----------------------------
# 2. Read output_x.csv & output_y.csv
# -----------------------------
output_x = np.loadtxt("output_x.csv", delimiter=",")
output_y = np.loadtxt("output_y.csv", delimiter=",")

output_x = np.atleast_1d(output_x).reshape(-1, 1)
output_y = np.atleast_1d(output_y).reshape(-1, 1)

print(f"Original output_x points: {len(output_x)}")
print(f"Original trajectory points: {len(downrange)}")

# -----------------------------
# 3. Extrapolate ascent phase until altitude matches glide start
# -----------------------------
x_last_ascent = output_x[-1, 0]
y_last_ascent = output_y[-1, 0]
y_first_glide = altitude[0, 0]

print(f"Last ascent altitude: {y_last_ascent/1000:.2f} km")
print(f"First glide altitude: {y_first_glide/1000:.2f} km")

# Use the last few points of ascent to fit a polynomial
n_fit_points = min(10, len(output_x))
x_fit = output_x[-n_fit_points:].flatten()
y_fit = output_y[-n_fit_points:].flatten()

# Fit a polynomial (degree 2 for smooth curve)
poly_coeffs = np.polyfit(x_fit, y_fit, deg=2)
poly_func = np.poly1d(poly_coeffs)

# Extrapolate forward until we reach or exceed the glide altitude
# Start from the last ascent point and step forward
step_size = 1000  # meters per step
x_bridge = [x_last_ascent]
y_bridge = [y_last_ascent]

x_current = x_last_ascent
while True:
    x_current += step_size
    y_current = poly_func(x_current)
    x_bridge.append(x_current)
    y_bridge.append(y_current)
    
    # Stop when we reach or exceed the target altitude
    if y_current >= y_first_glide:
        break
    
    # Safety check to prevent infinite loop
    if len(x_bridge) > 1000:
        print("Warning: extrapolation exceeded 1000 points, stopping")
        break

x_bridge = np.array(x_bridge).reshape(-1, 1)
y_bridge = np.array(y_bridge).reshape(-1, 1)

print(f"Bridge points created: {len(x_bridge)}")
print(f"Bridge end altitude: {y_bridge[-1, 0]/1000:.2f} km")
print(f"Bridge end downrange: {x_bridge[-1, 0]/1000:.2f} km")

# -----------------------------
# 4. Calculate offset to align glide phase with bridge end
# -----------------------------
x_bridge_end = x_bridge[-1, 0]
x_first_glide = downrange[0, 0]

# Offset the glide trajectory so it starts where the bridge ends
x_offset = x_bridge_end - x_first_glide
downrange = downrange + x_offset

print(f"Offset applied: {x_offset/1000:.2f} km")
print(f"Glide now starts at: {downrange[0, 0]/1000:.2f} km")

# -----------------------------
# 5. Combine all data
# -----------------------------
# Combine: ascent (output_x/y) + bridge + glide (downrange/altitude)
downrange_full = np.vstack((output_x, x_bridge[1:], downrange))  # Skip first bridge point (duplicate)
altitude_full = np.vstack((output_y, y_bridge[1:], altitude))

print(f"Total points after combining: {len(downrange_full)}")

# -----------------------------
# 6. Convert units to kilometers
# -----------------------------
downrange_km = downrange_full / 1000.0
altitude_km = altitude_full / 1000.0

# -----------------------------
# 7. Plot downrange vs altitude (single color, dark blue)
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(downrange_km, altitude_km, color='#00008B', linewidth=2)  # Dark blue
plt.xlabel("Downrange (km)", fontsize=12, fontweight='bold')
plt.ylabel("Altitude (km)", fontsize=12, fontweight='bold')
plt.title("Downrange vs Altitude", fontsize=14, fontweight='bold')
plt.grid(True)

# Make tick labels bold
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.show()