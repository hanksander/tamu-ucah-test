#!/bin/bash
# Usage: ./run_cbaero.sh <model_name> [Sref] [Cref] [Bref]
# Example: ./run_cbaero.sh waverider 1.0 1.0 1.0

set -e  # Exit on any error

# --- Configuration ---
PATH_TO_BINS="/root/401/CBaero/bin"

MODEL_NAME="$1"
SREF="${2:-1.0}"
CREF="${3:-1.0}"
BREF="${4:-1.0}"

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name> [Sref] [Cref] [Bref]"
    exit 1
fi

echo "=== Running full CBAERO setup for model: $MODEL_NAME ==="

# --- Step 1: Run cart2mesh interactively with expect ---
echo "[1/5] Running cart2mesh to generate mesh..."

expect <<EOF
spawn ${PATH_TO_BINS}/cart2mesh ${MODEL_NAME}
expect "Enter triangulated surface filename"
send "${MODEL_NAME}.tri\r"
expect eof
EOF

# Rename the generated mesh
if [ ! -f "output.msh" ]; then
    echo "Error: cart2mesh did not create output.msh"
    exit 1
fi

mv output.msh "${MODEL_NAME}.msh"
echo "Mesh renamed to ${MODEL_NAME}.msh"

# --- Step 2: Run cbsetup with xdotool to send keys to GUI window ---
echo "[2/5] Running cbsetup..."

# Check if xdotool is available
if ! command -v xdotool &> /dev/null; then
    echo "Error: xdotool not found. Install with: apt-get install xdotool"
    exit 1
fi

# Start cbsetup in background and get its PID
stty -ixon -ixoff
${PATH_TO_BINS}/cbsetup ${MODEL_NAME} &
CBSETUP_PID=$!

# Wait for the window to appear
sleep 3

# Find the cbsetup window
WINDOW_ID=$(xdotool search --name "cbsetup" | head -1)

if [ -z "$WINDOW_ID" ]; then
    echo "Warning: Could not find cbsetup window, trying with CBSETUP name..."
    WINDOW_ID=$(xdotool search --name "CBSETUP" | head -1)
fi

if [ -z "$WINDOW_ID" ]; then
    echo "Warning: Could not find cbsetup window, trying with class name..."
    WINDOW_ID=$(xdotool search --class "cbsetup" | head -1)
fi

if [ -z "$WINDOW_ID" ]; then
    echo "Error: Could not find cbsetup window"
    kill $CBSETUP_PID 2>/dev/null || true
    exit 1
fi

echo "Found cbsetup window: $WINDOW_ID"

# Activate the window and send keys
xdotool windowactivate --sync $WINDOW_ID
sleep 1

# Send Ctrl+S to save
echo "Sending Ctrl+S..."
xdotool key --window $WINDOW_ID ctrl+s
sleep 2

# Send Ctrl+Q to quit
echo "Sending Ctrl+Q..."
xdotool key --window $WINDOW_ID ctrl+q

# Wait for cbsetup to finish
wait $CBSETUP_PID 2>/dev/null || true

echo "cbsetup completed"

# Verify cbsetup output files were created
if [ ! -f "${MODEL_NAME}.bc" ]; then
    echo "Warning: ${MODEL_NAME}.bc not found after cbsetup"
fi

# --- Step 3: Generate .cbaero input file ---
echo "[3/5] Generating ${MODEL_NAME}.cbaero input file..."
python3 "${PATH_TO_BINS}/gen_cbaero.py" "$MODEL_NAME" "$SREF" "$CREF" "$BREF"

if [ ! -f "${MODEL_NAME}.cbaero" ]; then
    echo "Error: ${MODEL_NAME}.cbaero was not created"
    exit 1
fi

# --- Step 4: Run CBAERO ---
echo "[4/5] Running CBAERO with 8 processes..."
"${PATH_TO_BINS}/cbaero" -mp 8 "$MODEL_NAME"

echo "=== CBAERO run complete for model: $MODEL_NAME ==="
