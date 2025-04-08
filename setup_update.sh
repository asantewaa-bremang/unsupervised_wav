#!/bin/bash

# Wav2Vec Unsupervised Pipeline Setup Script
# This script handles all the necessary installations and environment setup
# for running the fairseq wav2vec unsupervised pipeline

set -e                       # Exit on error
set -o pipefail              # Exit if any command in a pipe fails
set -x                       # Print each command for debugging

# ==================== CONFIGURATION ====================
# Set these variables according to your environment

# Main directories
INSTALL_ROOT="$HOME/unsupervised_wav"
FAIRSEQ_ROOT="$INSTALL_ROOT/fairseq"
KENLM_ROOT="$INSTALL_ROOT/kenlm"
VENV_PATH="$INSTALL_ROOT/venv"
KALDI_ROOT="$INSTALL_ROOT/pykaldi/tools/kaldi"
RVADFAST_ROOT="$INSTALL_ROOT/rVADfast"
PYKALDI_ROOT="$INSTALL_ROOT/pykaldi"
FLASHLIGHT_SEQ_ROOT="$INSTALL_ROOT/sequence"


# Python version
PYTHON_VERSION="3.10"  # Options: 3.7, 3.8, 3.9, 3.10

# ==================== HELPER FUNCTIONS ====================

# Log message with timestamp
log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create directories if they don't exist
create_dirs() {
    mkdir -p "$INSTALL_ROOT"
    mkdir -p "$INSTALL_ROOT/logs"
}

get_system_cuda_suffix() {
    if ! command_exists nvcc; then
        log "ERROR: nvcc (NVIDIA CUDA Compiler) not found in PATH. Cannot determine CUDA version for GPU packages."
        exit 1
    fi
    local cuda_version
    cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')

    case "$cuda_version" in
        "12.1") echo "cu121" ;;
        "11.8") echo "cu118" ;;
        "11.7") echo "cu117" ;;
        "11.6") echo "cu116" ;;
        # Add other supported versions PyTorch offers as needed
        *)
            log "ERROR: Detected unsupported CUDA version $cuda_version. PyTorch might not have a pre-built package."
            log "Please check available PyTorch builds for your CUDA version."
            exit 1
            ;;
    esac
}

# ==================== PREREQUISITE CHECKS ====================
check_prerequisites() {
    log "--- Running Prerequisite Checks ---"
    local issues_found=0

    # ... (Keep the nvcc check as before) ...
    # if ! command_exists nvcc; then
    #     # ... (error message) ...
    #     issues_found=1
    # else
    #     # ... (success message) ...
    # fi

    # --- MODIFIED LD_LIBRARY_PATH Check ---
    # Check if the specific problematic path is forced at the beginning
    if [[ "$LD_LIBRARY_PATH" == "/usr/local/cuda/lib64"* ]]; then
        log "[FAIL] Prerequisite Check: LD_LIBRARY_PATH starts with '/usr/local/cuda/lib64'."
        log "       This indicates a system misconfiguration likely caused by manual CUDA/cuDNN setup"
        log "       or the /etc/profile.d/nvidia-env.sh script."
        log "       This WILL cause library conflicts (like the libcudnn error)."
        log ""
        log "       >>> MANUAL FIX REQUIRED <<<"
        log "       1. Edit the system file with: sudoedit /etc/profile.d/nvidia-env.sh"
        log "          (or use: sudo nano /etc/profile.d/nvidia-env.sh)"
        log "       2. Find the line starting with 'export LD_LIBRARY_PATH=...'"
        log "       3. Carefully REMOVE the '/usr/local/cuda/lib64:' part from the beginning of that line."
        log "          Example - Change:"
        log "            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nccl2/lib:..."
        log "          To:"
        log "            export LD_LIBRARY_PATH=/usr/local/nccl2/lib:..."
        log "       4. Save the file and exit the editor."
        log "       5. IMPORTANT: REBOOT your system or fully LOG OUT and LOG BACK IN."
        log "       6. Rerun this setup script AFTER rebooting/re-logging in."
        log ""
        issues_found=1 # Treat this as a fatal error for the script
    else
         log "[PASS] Prerequisite Check: LD_LIBRARY_PATH does not start with '/usr/local/cuda/lib64'."
    fi

    # ... (Keep Conda check and required command checks as before) ...
     # if [[ -n "$CONDA_PREFIX" # ... etc ... ]]; then
     #    # ... (conda info message) ...
     # fi
     # for cmd in git # ... etc ... ; do
     #     if ! command_exists "$cmd"; then
     #        # ... (error message) ...
     #        issues_found=1
     #     fi
     # done

    # --- Final Check ---
    if [ "$issues_found" -ne 0 ]; then
        log "ERROR: Prerequisite checks failed. Please address the [FAIL] items above before running the script again."
        exit 1 # Exit the script
    fi
    log "--- Prerequisite Checks Passed ---"
}

# ==================== SETUP STEPS ====================

# Step 2: Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
     #setting up pyenv to tackle linkage errors, protobuf requires a python environment which is not static 
    curl -fsSL https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
py_version_full=$(python --version 2>&1)
version=$(echo "$py_version_full" | cut -d ' ' -f 2)

# Or combined:
# version=$(python --version 2>&1 | cut -d ' ' -f 2)

echo "Detected Python version: $version"
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install $version
    pyenv local $version
    
    if [ -d "$VENV_PATH" ]; then
        log "Virtual environment already exists at $VENV_PATH"
    else
        python${PYTHON_VERSION} -m venv "$VENV_PATH"
        log "Created virtual environment at $VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip and related tools
    pip install --upgrade pip setuptools wheel

    log "Python virtual environment setup completed."
}

# Step 3: Install PyTorch and related packages
install_pytorch() {
    log "Installing PyTorch and related packages..."

    torch_cuda_suffix=$(get_system_cuda_suffix)
    
    source "$VENV_PATH/bin/activate"
    
   
    
    # For now, we install without checking nvcc (adjust as needed)
    # pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

    # pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/${torch_cuda_suffix}"
    
    # Install other required packages
    pip install "numpy<2" scipy tqdm sentencepiece soundfile librosa editdistance tensorboardX packaging soundfile
    pip install npy-append-array faiss-gpu h5py kaldi-io g2p_en
    sudo apt install zsh
    sudo apt install yq
    python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
    # Optional: omegaconf, hydra-core (if needed by your workflow)
    
    log "PyTorch and related packages installed successfully."
}

# Step 4: Clone and install fairseq
install_fairseq() {
    log "--- Installing fairseq ---"
    log "Activating virtual environment: $VENV_PATH"
     pip install "pip==24.0"
    source "$VENV_PATH/bin/activate"

    cd "$INSTALL_ROOT"

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists. Pulling latest changes..."
        cd "$FAIRSEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest fairseq changes. Continuing with existing version."; }
    else
        log "Cloning fairseq repository..."
        git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_ROOT" \
            || { log "[ERROR] Failed to clone fairseq repository."; exit 1; }
        cd "$FAIRSEQ_ROOT"
    fi

    log "Installing fairseq in editable mode..."
    # Upgrade pip first if necessary (sometimes helps with editable installs)
    # pip install --upgrade pip
    pip install --editable ./ \
        || { log "[ERROR] Failed to install fairseq in editable mode."; exit 1; }

    # Install wav2vec specific requirements if the file exists
    local wav2vec_req_file="$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    if [ -f "$wav2vec_req_file" ]; then
        log "Installing wav2vec specific requirements from $wav2vec_req_file..."
        pip install -r "$wav2vec_req_file" \
            || { log "[WARN] Failed to install some wav2vec requirements. Check $wav2vec_req_file."; }
    else
        log "[INFO] No specific requirements file found at $wav2vec_req_file."
    fi

    log "fairseq installed successfully."
    deactivate
}


# Step 5: Install rVADfast for audio silence removal
install_rVADfast() {
    log "Cloning and installing rVADfast..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"

    if [ -d "$RVADFAST_ROOT" ]; then
        log "rVADfast already exists. Updating..."
        cd "$RVADFAST_ROOT"
        git pull
    else
        log "Cloning rVADfast repository..."
        git clone https://github.com/zhenghuatan/rVADfast.git "$RVADFAST_ROOT"
        cd "$RVADFAST_ROOT"
    fi

   
    mkdir -p "$RVADFAST_ROOT/src"
    
    log "rVADfast installed successfully."
}

# Step 6: Clone and build KenLM
install_kenlm() {
    log "Cloning and building KenLM..."
    cd "$INSTALL_ROOT"

    sudo apt update
    sudo apt install libeigen3-dev

    sudo apt update
    sudo apt install libboost-all-dev

    if [ -d "$KENLM_ROOT" ]; then
        log "KenLM repository already exists."
    else
        log "Cloning KenLM repository..."
        git clone https://github.com/kpu/kenlm.git "$KENLM_ROOT"
    fi
    
    cd "$KENLM_ROOT"
    if [ -d "build" ]; then
        log "KenLM build directory already exists. Skipping build step."
    else  
        mkdir -p build
        cd build
        cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        make -j $(nproc)
    fi
    
    source "$VENV_PATH/bin/activate"
    pip install https://github.com/kpu/kenlm/archive/master.zip
    
    log "KenLM built successfully."
}

# Step 7: Install Flashlight and Flashlight-Sequence
install_flashlight() {
    log "--- Installing Flashlight (Text and Sequence) ---"
    cd "$INSTALL_ROOT"

    sudo apt-get install pybind11-dev

    # Check for nvcc before proceeding with GPU build
    # if ! command_exists nvcc; then
    #     log "[ERROR] nvcc not found. Cannot build Flashlight Sequence with CUDA support."
    #     exit 1
    # fi
    # log "[INFO] Found nvcc. Proceeding with Flashlight Sequence GPU build."

    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # Install flashlight-text (Python-only package)
    log "Installing flashlight-text Python package..."
    pip install flashlight-text \
        || { log "[ERROR] Failed to install flashlight-text."; exit 1; }

    # Clone or update the sequence repository
    if [ -d "$FLASHLIGHT_SEQ_ROOT" ]; then
        log "Flashlight sequence repository already exists. Updating..."
        cd "$FLASHLIGHT_SEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest flashlight sequence changes."; }
    else
        log "Cloning flashlight sequence repository..."
        git clone https://github.com/flashlight/sequence.git "$FLASHLIGHT_SEQ_ROOT" \
            || { log "[ERROR] Failed to clone flashlight sequence."; exit 1; }
        cd "$FLASHLIGHT_SEQ_ROOT"
    fi

    log "Configuring and Building flashlight sequence library WITH Python bindings..."
    # Remove old build directory for a clean state
    rm -rf build
    mkdir build && cd build

    # Configure using CMake - ADD THE PYTHON FLAG!
    # !!! CHECK FLASHLIGHT DOCUMENTATION FOR THE CORRECT PYTHON FLAG NAME !!!
    # Common names: -DFLASHLIGHT_BUILD_PYTHON=ON, -DBUILD_PYTHON=ON, etc.
    # Using -DFLASHLIGHT_BUILD_PYTHON=ON as the most likely candidate
    local flashlight_python_flag="-DFLASHLIGHT_BUILD_PYTHON=ON" # <--- CHECK THIS FLAG!
    log "[INFO] Using CMake flag for Python bindings: $flashlight_python_flag (Verify this is correct!)"

    export USE_CUDA=1 # Set if building for CUDA
    # Explicitly point CMake to the Python executable in the venv for robustness
    local python_executable="$VENV_PATH/bin/python"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE="$python_executable" \
             "$flashlight_python_flag" \
             # Add other required CMake flags here (e.g., -DFLASHLIGHT_BACKEND=CUDA)
             # Check Flashlight docs for required flags! Example: -DFLASHLIGHT_SEQUENCE_BUILD_TESTS=OFF
             # -DCMAKE_PREFIX_PATH might be needed if dependencies aren't found
             # -DCMAKE_PREFIX_PATH="$VENV_PATH" # Example
         # || { log "[ERROR] Flashlight sequence CMake configuration failed."; exit 1; }

    # Build the C++ library AND Python bindings
    log "Building Flashlight sequence (C++ and Python)..."
    cmake --build . --config Release --parallel "$(nproc)" \
        # || { log "[ERROR] Flashlight sequence build failed."; exit 1; }

    # Install the Python Bindings into the ACTIVE virtual environment
    log "Installing Flashlight sequence Python bindings into venv..."
    # This assumes setup.py or similar is generated in the build directory.
    pip install . \
        # || { log "[ERROR] Failed to install Flashlight Python bindings via pip. Check build output and Flashlight docs."; exit 1; }
    log "[PASS] Flashlight Python bindings installed via pip."

    # Optional: Install C++ library system-wide AFTER Python install succeeds
    log "Installing Flashlight sequence C++ library system-wide (optional)..."
    sudo cmake --install . --config Release \
        # || { log "[WARN] Failed to install Flashlight C++ library system-wide (sudo). Python bindings are installed."; }

    cd "$INSTALL_ROOT" # Go back to install root
    log "Flashlight installation steps completed."

    # --- Re-install fairseq AFTER Flashlight bindings are in venv ---
    log "Re-installing fairseq to ensure it picks up Flashlight bindings..."
    install_fairseq # Call the fairseq install function again (it will activate/deactivate venv)

    log "--- Flashlight Installation Finished ---"
    # Final deactivate handled by install_fairseq
}



# Step 8: Install PyKaldi
install_pykaldi() {
    log "Installing pykaldi..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"
    sudo apt update
    sudo apt install pkg-config python3-dev

    pip install numpy pyparsing ninja wheel setuptools cython

    if [ -d "$PYKALDI_ROOT" ]; then
        log "PyKaldi repository already exists. Updating..."
        cd "$PYKALDI_ROOT"
        git pull
    else
        log "Cloning PyKaldi repository..."
        git clone https://github.com/pykaldi/pykaldi.git "$PYKALDI_ROOT"
        cd "$PYKALDI_ROOT"
    fi

    cd "$PYKALDI_ROOT/tools"
    ./check_dependencies.sh
    pip uninstall protobuf
     pip install pyparsing
    source "$VENV_PATH/bin/activate"
    ./install_protobuf.sh
    
    sudo apt update
    sudo apt install -y libprotobuf-dev protobuf-compiler

    
    pip install protobuf #this installs the python version 
   
    source "$VENV_PATH/bin/activate"
    cp -r $PWD/protobuf/include/google $VENV_PATH/include  #this step is to load protobuf headers in roots for easy compliation of pyclif
    ./install_clif.sh

    cd "$PYKALDI_ROOT/tools"
    sudo apt-get install -y python2.7  # Kaldi build system may require Python 2

    sudo ./install_mkl.sh

    sudo apt-get update
    sudo apt-get install sox subversion
    source "$VENV_PATH/bin/activate"
    if [ ! -d "$KALDI_ROOT" ]; then
        log "Cloning and building Kaldi for PyKaldi..."
        ./install_kaldi.sh
    else
        log "Kaldi already exists at $KALDI_ROOT"
    fi

    cd "$PYKALDI_ROOT"

    source "$VENV_PATH/bin/activate"
    python setup.py install

    source "$VENV_PATH/bin/activate"
    sudo pip uninstall tensorboardX
    pip install tensorboardX
    
    log "PyKaldi installed successfully."
}

# Step 9: Download pre-trained wav2vec model
download_pretrained_model() {
    log "Downloading pre-trained wav2vec model..."
    
    mkdir -p "$INSTALL_ROOT/pre-trained"
    cd "$INSTALL_ROOT/pre-trained"
    
    if [ -f "$INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt" ]; then
        log "Pre-trained model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt
    fi
    
    log "Pre-trained model downloaded successfully."
}

# Step 10: Download language identification model
download_languageIdentification_model() {
    log "Downloading language identification model..."
    
    mkdir -p "$INSTALL_ROOT/lid_model"
    cd "$INSTALL_ROOT/lid_model"
    
    if [ -f "$INSTALL_ROOT/lid_model/lid.176.bin" ]; then
        log "LID model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi

    source "$VENV_PATH/bin/activate"
    pip install fasttext
    
    log "Language identification model downloaded successfully."
}

# Step 11: Create a configuration file
create_config_file() {
    log "Creating a configuration file..."
    
    cat > "$INSTALL_ROOT/wav2vec_config.sh" << EOF
#!/bin/bash
# Wav2Vec configuration file
# Source this file to set up environment variables

# Main directories
export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
export KENLM_ROOT="$KENLM_ROOT"
export VENV_PATH="$VENV_PATH"
export KALDI_ROOT="$KALDI_ROOT"
export PYKALDI_ROOT="$PYKALDI_ROOT"
export RVADFAST_ROOT="$RVADFAST_ROOT"

# Add KenLM binaries to PATH
export PATH="\$KENLM_ROOT/build/bin:\$PATH"

# Add rVADfast to PATH
export PATH="\$RVADFAST_ROOT/src:\$PATH"

# Add Kaldi binaries to PATH
export PATH="\$KALDI_ROOT/src/bin:\$KALDI_ROOT/tools/openfst/bin:\$KALDI_ROOT/src/fstbin:\$KALDI_ROOT/src/lmbin:\$PATH"

# Python module paths
export PYTHONPATH="\$FAIRSEQ_ROOT:\$PYKALDI_ROOT:\$PYTHONPATH"

# Function to activate the environment
activate_wav2vec_env() {
    source "\$VENV_PATH/bin/activate"
    echo "Wav2Vec environment activated."
}

export -f activate_wav2vec_env
EOF

    chmod +x "$INSTALL_ROOT/wav2vec_config.sh"
    
    log "Configuration file created at $INSTALL_ROOT/wav2vec_config.sh"
    log "To use, run: source $INSTALL_ROOT/wav2vec_config.sh"
}

# Step 12: Create a simple test script to verify installation
create_test_script() {
    log "Creating a test script..."
    
    cat > "$INSTALL_ROOT/test_installation.py" << EOF
#!/usr/bin/env python3
"""
Test script to verify Wav2Vec installation and dependencies
"""
import os
import importlib

def check_module(name):
    try:
        importlib.import_module(name)
        print(f"✅ {name} installed successfully")
        return True
    except ImportError as e:
        print(f"❌ {name} import failed: {e}")
        return False

def main():
    print("Testing Wav2Vec dependencies installation...")
    modules = [
        "torch", "numpy", "scipy", "tqdm", "sentencepiece", 
        "soundfile", "librosa", "kenlm", "fairseq", "tensorboardX",
        "fasttext", "flashlight_text", "flashlight.sequence.criteria"
    ]
    success = 0
    for module in modules:
        if check_module(module):
            success += 1

    try:
        import torch
        import fairseq
        model_path = os.path.join(os.environ.get("INSTALL_ROOT", ""), "pre-trained", "wav2vec_vox_new.pt")
        if os.path.exists(model_path):
            print(f"✅ Pre-trained model exists at {model_path}")
            model = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Pre-trained model not found at {model_path}")
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
    
    print(f"\nResults: {success}/{len(modules)} dependencies installed successfully")

if __name__ == "__main__":
    main()
EOF

    chmod +x "$INSTALL_ROOT/test_installation.py"
    
    log "Test script created at $INSTALL_ROOT/test_installation.py"
    log "After installation, run: source $INSTALL_ROOT/wav2vec_config.sh && python $INSTALL_ROOT/test_installation.py"
}

# ==================== MAIN EXECUTION ====================
main() {
    log "Starting Wav2Vec environment setup..."
    check_prerequisites
    create_dirs
    
    setup_venv
    install_fairseq
    install_flashlight
    # install_system_deps
    install_pytorch
    install_kenlm
    install_rVADfast
    
    install_pykaldi
    download_pretrained_model
    download_languageIdentification_model
    create_config_file
    create_test_script
    
    log "Setup completed successfully!"
    log "------------------------------------------------------"
    log "To use this environment, run: source $INSTALL_ROOT/wav2vec_config.sh"
    log "To test the installation, run: python $INSTALL_ROOT/test_installation.py"
    log "------------------------------------------------------"
}

# Run the main function
main
