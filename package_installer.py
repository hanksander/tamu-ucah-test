import subprocess
import sys

# packages to install
packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "openmdao",
    "dymos",
    "ambiance",
    "cantera",
    "scikit-learn",
    "pandas",
    "om-pycycle"
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