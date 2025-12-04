#!/bin/bash
# Usage: ./run_cbaero.sh <model_name> [Sref] [Cref] [Bref]
# Example: ./run_cbaero.sh waverider 1.0 1.0 1.0

# Force allocation of a pseudo-TTY
if [ ! -t 0 ]; then
    exec script -q -c "$0 $*" /dev/null
fi

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
mv output.msh "${MODEL_NAME}.msh"
echo "Mesh renamed to ${MODEL_NAME}.msh"

# --- Step 2: Run cbsetup non-interactively ---
echo "[2/5] Running cbsetup..."

stty -ixon -ixoff
expect <<EOF
spawn ${PATH_TO_BINS}/cbsetup ${MODEL_NAME}
# Wait for the prompt and save/quit
expect ">"
sleep 3
send "\x13"
sleep 
send "\x11"
expect eof
EOF

# --- Step 3: Generate .cbaero input file ---
echo "[3/5] Generating ${MODEL_NAME}.cbaero input file..."
python3 "${PATH_TO_BINS}/gen_cbaero.py" "$MODEL_NAME" "$SREF" "$CREF" "$BREF"

# --- Step 4: Run CBAERO ---
echo "[4/5] Running CBAERO with 8 processes..."
"${PATH_TO_BINS}/cbaero" -mp 8 -omp 2 "$MODEL_NAME"

echo "=== CBAERO run complete for model: $MODEL_NAME ==="

