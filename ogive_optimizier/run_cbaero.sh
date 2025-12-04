#!/bin/bash
# Usage: ./run_cbaero.sh <model_name> [Sref] [Cref] [Bref]
# Example: ./run_cbaero.sh waverider 1.0 1.0 1.0

# --- Configuration & Error Handling ---
set -e         # Exit immediately if a command exits with a non-zero status
set -u         # Treat unset variables as an error
set -o pipefail # The return value of a pipeline is the status of the last command 

# sudo rm -rf /tmp/.X11-unix

PATH_TO_BINS="/root/401/CBaero/bin"

MODEL_NAME="$1"
SREF="${2:-1.0}"
CREF="${3:-1.0}"
BREF="${4:-1.0}"

# 1. Argument Validation
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name> [Sref] [Cref] [Bref]" >&2
    exit 1
fi

# 2. Dependency Check for Xvfb and xdotool
if ! command -v xdotool &> /dev/null || ! command -v Xvfb &> /dev/null; then
    echo "Error: xdotool or Xvfb not found. Install with: apt-get install xdotool xvfb" >&2
    exit 1
fi

echo "=== Running full CBAERO setup for model: $MODEL_NAME ==="

# --- Step 1: Run cart2mesh non-interactively ---
echo "[1/5] Running cart2mesh to generate mesh..."

expect <<EOF
spawn ${PATH_TO_BINS}/cart2mesh ${MODEL_NAME}
expect "Enter triangulated surface filename"
send "${MODEL_NAME}.tri\r"
expect eof
EOF

# Use trap to ensure Xvfb is killed even if a later step fails
Xvfb_PID=""
function cleanup {
    if [ ! -z "$Xvfb_PID" ]; then
        kill $Xvfb_PID 2>/dev/null || true
        unset DISPLAY
    fi
}
trap cleanup EXIT

# Rename the generated mesh
mv -f output.msh "${MODEL_NAME}.msh"
echo "Mesh renamed to ${MODEL_NAME}.msh"

# --- Step 2: Run cbsetup HEADLESS using Xvfb and xdotool ---
# --- Step 2: Run cbsetup HEADLESS using Xvfb and xdotool ---
echo "[2/5] Running cbsetup headless (Bypassing /tmp)..."

# --- Define Custom X Environment ---
# 1. Define custom, safe temporary paths
export DISPLAY=":99"
X_AUTH_FILE="/root/.Xauthority_cbsetup"
X_SOCKET_DIR="/root/x_sockets"
mkdir -p $X_SOCKET_DIR

# 2. Start Xvfb, telling it where to put the socket using the -logfile option
# Note: We must also tell Xvfb not to use the default /tmp/.X11-unix
# We use -listen tcp and -nolisten unix to avoid using the broken /tmp/.X11-unix

# Start virtual display, listening only on TCP and disabling the default UNIX socket
Xvfb $DISPLAY -screen 0 1024x768x24 -nolisten unix -listen tcp &
Xvfb_PID=$!
sleep 1.5 # Wait for Xvfb to start

# Start the lightweight Window Manager (twm) inside Xvfb
twm &
TWM_PID=$!
sleep 1 # Wait for twm to start

# Update cleanup function to kill twm as well
function cleanup {
    if [ ! -z "$TWM_PID" ]; then
        kill $TWM_PID 2>/dev/null || true
    fi
    if [ ! -z "$Xvfb_PID" ]; then
        kill $Xvfb_PID 2>/dev/null || true
        unset DISPLAY
    fi
}
trap cleanup EXIT
# The trap is already set at the start, but we re-define it to include TWM_PID

# Run cbsetup in the virtual environment
stty -ixon -ixoff
${PATH_TO_BINS}/cbsetup ${MODEL_NAME} &
CBSETUP_PID=$!
sleep 3 # Use a slightly longer sleep to ensure cbsetup is ready

# Find the cbsetup window (no --display needed, as DISPLAY is exported)
WINDOW_ID=$(xdotool search --name "cbsetup" | head -1)

# ... (fallback searches) ...

if [ -z "$WINDOW_ID" ]; then
    echo "Error: Could not find cbsetup window. Check logs." >&2
    kill $CBSETUP_PID 2>/dev/null || true
    exit 1
fi

echo "Found cbsetup window on $DISPLAY: $WINDOW_ID"

# Send keys
# We use xdotool windowfocus instead of windowactivate, as it is often more robust
# in minimal environments.
xdotool windowfocus --sync $WINDOW_ID
sleep 0.5

# OLD (Fragile)
# echo "Sending Ctrl+S (Save)..."
# xdotool key --window $WINDOW_ID ctrl+s
# sleep 1
# echo "Sending Ctrl+Q (Quit)..."
# xdotool key --window $WINDOW_ID ctrl+q

# NEW (More Robust)
echo "Sending Ctrl+S (Save)..."
xdotool key --window $WINDOW_ID ctrl+s
sleep 1
echo "Sending close command to window $WINDOW_ID..."
# Use windowclose command, which is often more robust than keypresses for quitting
xdotool windowclose $WINDOW_ID
sleep 0.5

# Wait for cbsetup to finish
wait $CBSETUP_PID 2>/dev/null || true

# Kill TWM and Xvfb (cleanup trap takes care of this, but we reset PIDs here)
TWM_PID=""
Xvfb_PID="" 
echo "cbsetup completed"

# Verification
[ -f "${MODEL_NAME}.bc" ] || echo "Warning: ${MODEL_NAME}.bc not found after cbsetup"

# --- Rest of script continues ---

# --- Step 3: Generate .cbaero input file ---
echo "[3/5] Generating ${MODEL_NAME}.cbaero input file..."
python3 "${PATH_TO_BINS}/gen_cbaero.py" "$MODEL_NAME" "$SREF" "$CREF" "$BREF"

[ -f "${MODEL_NAME}.cbaero" ] || { echo "Error: ${MODEL_NAME}.cbaero was not created" >&2; exit 1; }

# --- Step 4: Run CBAERO ---
echo "[4/5] Running CBAERO with 16 processes..."
"${PATH_TO_BINS}/cbaero" -mp 16 "$MODEL_NAME"

echo "=== CBAERO run complete for model: $MODEL_NAME ==="