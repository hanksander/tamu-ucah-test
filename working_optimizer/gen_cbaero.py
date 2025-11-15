#!/usr/bin/env python3
import sys

if len(sys.argv) < 5:
    print("Usage: gen_cbaero.py <model_name> <Sref> <Cref> <Bref>")
    sys.exit(1)

model_name = sys.argv[1]
Sref = float(sys.argv[2])
Cref = float(sys.argv[3])
Bref = float(sys.argv[4])

content = f"""FileType =     fast
Sref =         {Sref:.6f}
Cref =         {Cref:.6f}
Bref =         {Bref:.6f}
X_cg =         0.000000
Y_cg =         0.000000
Z_cg =         0.000000
scale =        1.000000
flotyp =       0
retm_c =       220.000000
retm_t =       0.000000
strm_line_dt = 0.250000
Planet =       EARTH
Mach Number
7
 2.000  3.000   4.000  5.000 6.000 7.000 8.000
Dynamic Pressure (Bars)
3
 5  10  15
Angle of Attack
7
 -8.0 -4.0 -2.0 0.0 2.0 4.0 8.0
Angle of SideSlip
1
 0.000
Control Surfaces
0
"""

with open(f"{model_name}.cbaero", "w") as f:
    f.write(content)

print(f"Generated {model_name}.cbaero")

