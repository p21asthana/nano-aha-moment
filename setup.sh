#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

# --- Configuration ---
CONDA_ENV_NAME="nano-aha"
REPO_URL="https://github.com/p21asthana/nano-aha-moment.git"
REPO_DIR="nano-aha-moment"
PYTHON_VERSION="3.10" # Specify Python version for conda env

# --- System Updates and Build Tools ---
echo ">>> Updating system and installing Development Tools..."
sudo yum update -y
sudo yum groupinstall "Development Tools" -y

# --- Clone Repository ---
echo ">>> Cloning repository..."
git clone "${REPO_URL}"
cd "${REPO_DIR}"

# --- Create Conda Environment ---
echo ">>> Creating Conda environment: ${CONDA_ENV_NAME}..."
conda create --name "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y

# --- Activate Conda Environment (Requires sourcing conda init) ---
echo ">>> Activating Conda environment..."
# Find conda base path and source init script
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# --- Modify requirements.txt ---
echo ">>> Modifying requirements.txt for compatibility..."
# Check if files exist before modifying
if [ -f requirements.txt ]; then
    # Update torch to 2.6.0 (required by vllm 0.8.4)
    sed -i 's/torch==2.5.1/torch==2.6.0/' requirements.txt
    # Update vllm to 0.8.4 (to fix xgrammar issue and specify compatible version)
    sed -i 's/vllm==0.7.3/vllm==0.8.4/' requirements.txt
    # Update transformers to >=4.51.1 (required by vllm 0.8.4)
    sed -i 's/transformers==4.48.3/transformers>=4.51.1/' requirements.txt
    echo "requirements.txt modified."
else
    echo "ERROR: requirements.txt not found in $(pwd)"
    exit 1
fi

# --- Configure Conda Channels ---
echo ">>> Configuring Conda channels..."
conda config --add channels conda-forge
conda config --set channel_priority strict

# --- Install Compatible Conda Compiler (GCC 11) ---
echo ">>> Installing Conda GCC 11..."
conda install gcc_linux-64=11 gxx_linux-64=11 -c conda-forge -y

# --- Install Ninja Build System ---
echo ">>> Installing Ninja..."
# Ensure pip uses the conda environment's pip
"${CONDA_PREFIX}/bin/pip" install ninja

# --- Install Python Dependencies with Workarounds ---
echo ">>> Installing Python dependencies from requirements.txt..."

# 1. Unset LD_LIBRARY_PATH to avoid conflicts with system CUDA libs
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset."

# 2. Set CC and CXX to use the Conda GCC 11
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
echo "CC set to: ${CC}"
echo "CXX set to: ${CXX}"

# 3. Install requirements using pip from the conda env
"${CONDA_PREFIX}/bin/pip" install -r requirements.txt

# --- Final Verification (Optional) ---
echo ">>> Verifying installation (optional)..."
unset LD_LIBRARY_PATH # Ensure it's still unset for verification
"${CONDA_PREFIX}/bin/python" -c "import torch; import deepspeed; import xformers; import vllm; import flash_attn; print('--- Verification ---'); print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Verification complete.')"

echo ">>> Setup script finished successfully!"
echo ">>> To use the environment, run: conda activate ${CONDA_ENV_NAME}"
echo ">>> Remember to run 'unset LD_LIBRARY_PATH' in your shell before running python scripts."
