import subprocess
import sys

# packages to install
# Each entry is passed verbatim to `pip install`, so VCS URLs are allowed
# (e.g. pyCycle is not on PyPI under this name and must come from GitHub).
packages = [
    "numpy",
    "scipy",
    "cython",
    "sqlitedict",
    "matplotlib",
    "openmdao",
    "dymos",
    "ambiance",
    "cantera",
    "scikit-learn",
    "pandas",
    "trimesh",
    "shapely",
    "manifold3d",
    # pyCycle (OpenMDAO thermo-cycle library) — required by pyc_run,
    # pyc_ram_cycle, and engine_envelope_test.  Installed from source.
    "git+https://github.com/OpenMDAO/pyCycle.git",
]

print("Installing aerospace packages in virtual environment...")
print(f"Using Python: {sys.executable}")

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

for package in packages:
    print(f"\nInstalling {package}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package])

    if result.returncode == 0:
        print(f"✓ {package} installed successfully")
    else:
        print(f"✗ Failed to install {package}")

print("\nInstallation complete!")