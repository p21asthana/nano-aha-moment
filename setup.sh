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

# --- Clone or Enter Repository ---
echo ">>> Ensuring repository directory exists and entering it..."
if [ -d "${REPO_DIR}" ]; then
  echo "Directory ${REPO_DIR} already exists. Entering directory."
  cd "${REPO_DIR}"
  # Optional: You might want to pull the latest changes if it exists
  # echo "Pulling latest changes..."
  # git pull
else
  echo "Cloning repository ${REPO_URL}..."
  git clone "${REPO_URL}"
  cd "${REPO_DIR}"
fi

# --- Create Conda Environment ---
echo ">>> Creating Conda environment: ${CONDA_ENV_NAME}..."
conda create --name "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y

# --- Activate Conda Environment (Requires sourcing conda init) ---
echo ">>> Activating Conda environment..."
# Find conda base path and source init script
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

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

# --- Install Python Dependencies (Two Steps) ---
echo ">>> Installing PyTorch first..."

# 1. Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset for PyTorch install."

# 2. Set CC/CXX (Might not be needed for torch install, but doesn't hurt)
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
echo "CC/CXX set for PyTorch install."

# 3. Install torch==2.6.0 directly (from modified requirements)
# Let pip find the appropriate wheel (omit index-url for now)
# Use the conda env pip explicitly
"${CONDA_PREFIX}/bin/pip" install torch==2.6.0

echo ">>> Installing remaining dependencies..."

# 1. Unset LD_LIBRARY_PATH again (safety measure)
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset for remaining installs."

# 2. Set CC/CXX again (essential for xformers, deepspeed)
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
echo "CC/CXX set for remaining installs."

# 3. Install the rest of the requirements (pip will skip already installed torch)
"${CONDA_PREFIX}/bin/pip" install -r requirements.txt

# --- Final Verification (Optional) ---
echo ">>> Verifying installation (optional)..."
unset LD_LIBRARY_PATH # Ensure it's still unset for verification
"${CONDA_PREFIX}/bin/python" -c "import torch; import deepspeed; import xformers; import vllm; import flash_attn; print('--- Verification ---'); print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Verification complete.')"

echo ">>> Setup script finished successfully!"
echo ">>> To use the environment, run: conda activate ${CONDA_ENV_NAME}"
echo ">>> Remember to run 'unset LD_LIBRARY_PATH' in your shell before running python scripts."
