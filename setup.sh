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
# Define persistent environment path inside SageMaker directory
PERSISTENT_ENV_PATH="/home/ec2-user/SageMaker/envs/${CONDA_ENV_NAME}"

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

# --- Create Conda Environment at Persistent Path ---
echo ">>> Checking/Creating Conda environment at persistent path: ${PERSISTENT_ENV_PATH}..."
# Ensure the parent directory exists
mkdir -p "$(dirname "${PERSISTENT_ENV_PATH}")"

if [ -d "${PERSISTENT_ENV_PATH}" ]; then
    echo "Environment directory ${PERSISTENT_ENV_PATH} already exists. Skipping creation."
    echo "If you want a fresh environment, please remove the directory first:"
    echo "rm -rf ${PERSISTENT_ENV_PATH}"
else
    conda create --prefix "${PERSISTENT_ENV_PATH}" python="${PYTHON_VERSION}" -y
    echo "Environment created at ${PERSISTENT_ENV_PATH}"
fi


# --- Activate Conda Environment (Requires sourcing conda init) ---
echo ">>> Activating Conda environment..."
# Find conda base path and source init script
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Activate using the persistent path
conda activate "${PERSISTENT_ENV_PATH}"

# --- Configure Conda Channels ---
echo ">>> Configuring Conda channels..."
conda config --add channels conda-forge
conda config --set channel_priority strict

# --- Install Compatible Conda Compiler (GCC 11) ---
echo ">>> Installing Conda GCC 11 into ${PERSISTENT_ENV_PATH}..."
# Use conda install with prefix flag if not activated, or direct if activated
conda install --prefix "${PERSISTENT_ENV_PATH}" gcc_linux-64=11 gxx_linux-64=11 -c conda-forge -y

# --- Install Ninja Build System ---
echo ">>> Installing Ninja into ${PERSISTENT_ENV_PATH}..."
# Use explicit path to pip inside the persistent env
"${PERSISTENT_ENV_PATH}/bin/pip" install ninja

# --- Install Python Dependencies (Two Steps) ---
echo ">>> Installing PyTorch first..."

# 1. Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset for PyTorch install."

# 2. Set CC/CXX to use the Conda GCC 11 from the persistent path
export CC="${PERSISTENT_ENV_PATH}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${PERSISTENT_ENV_PATH}/bin/x86_64-conda-linux-gnu-g++"
echo "CC set to: ${CC}"
echo "CXX set to: ${CXX}"

# 3. Install torch==2.6.0 directly (from modified requirements)
# Use explicit path to pip inside the persistent env
"${PERSISTENT_ENV_PATH}/bin/pip" install torch==2.6.0

echo ">>> Installing remaining dependencies..."

# 1. Unset LD_LIBRARY_PATH again (safety measure)
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset for remaining installs."

# 2. Set CC/CXX again (essential for xformers, deepspeed)
export CC="${PERSISTENT_ENV_PATH}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${PERSISTENT_ENV_PATH}/bin/x86_64-conda-linux-gnu-g++"
echo "CC set to: ${CC}"
echo "CXX set to: ${CXX}"

# 3. Install the rest of the requirements (pip will skip already installed torch)
# Use explicit path to pip inside the persistent env
"${PERSISTENT_ENV_PATH}/bin/pip" install -r requirements.txt

# --- Final Verification (Optional) ---
echo ">>> Verifying installation (optional)..."
unset LD_LIBRARY_PATH # Ensure it's still unset for verification
# Use explicit path to python inside the persistent env
"${PERSISTENT_ENV_PATH}/bin/python" -c "import torch; import deepspeed; import xformers; import vllm; import flash_attn; print('--- Verification ---'); print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Verification complete.')"

echo ">>> Setup script finished successfully!"
echo ">>> To use the environment, run: conda activate ${PERSISTENT_ENV_PATH}"
echo ">>> Remember to run 'unset LD_LIBRARY_PATH' in your shell before running python scripts."
