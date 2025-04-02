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
KALDI_ROOT="$INSTALL_ROOT/kaldi"
RVADFAST_ROOT="$INSTALL_ROOT/rVADfast"
PYKALDI_ROOT="$INSTALL_ROOT/pykaldi"

# CUDA version (for PyTorch installation)
CUDA_VERSION="11.7"  # Options: 10.2, 11.3, 11.6, 11.7, etc.

# Python version
PYTHON_VERSION="3.10.16"  # Options: 3.7, 3.8, 3.9, 3.10

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

# ==================== SETUP STEPS ====================

# Step 1: Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."

    if command_exists apt-get; then
        sudo apt-get update

        sudo apt-get install -y \
            build-essential \
            cmake \
            libboost-all-dev \
            libeigen3-dev \
            libatlas-base-dev \
            libfftw3-dev \
            libopenblas-dev \
            python3-pip \
            python3-venv \
            git \
            wget \
            zlib1g-dev \
            automake \
            autoconf \
            libtool \
            subversion \
            sox \
            libsox-dev \
            libsox-fmt-all \
            flac \
            ffmpeg \
            libprotobuf-dev \
            protobuf-compiler \
            bzip2 \
            gfortran \
            libbz2-dev \
            liblzma-dev

    else
        log "ERROR: Unsupported package manager. Please install dependencies manually."
        exit 1
    fi

    log "System dependencies installed successfully."
}

# Step 2: Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
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
    
    source "$VENV_PATH/bin/activate"
    
    # Uncomment the following block if you want CUDA support (ensure nvcc exists)
    # if command_exists nvcc; then
    #     log "CUDA detected. Installing PyTorch with CUDA support..."
    #     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
    # else
    #     log "CUDA not detected. Installing PyTorch for CPU only..."
    #     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    # fi
    
    # For now, we install without checking nvcc (adjust as needed)
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
    
    # Install other required packages
    pip install numpy scipy tqdm sentencepiece soundfile librosa editdistance tensorboardX packaging 
    pip install npy-append-array faiss-gpu h5py kaldi-io
    # Optional: omegaconf, hydra-core (if needed by your workflow)
    
    log "PyTorch and related packages installed successfully."
}

# Step 4: Clone and install fairseq
install_fairseq() {
    log "Cloning and installing fairseq..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"

    #changing my pip version to this 
    
    pip install --upgrade pip==24

# Your other commands here...

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists. Updating..."
        cd "$FAIRSEQ_ROOT"
        git pull
    else
        log "Cloning fairseq repository..."
        git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_ROOT"
        cd "$FAIRSEQ_ROOT"
    fi
    
    # pip install sacrebleu==1.5.1 requests regex sacremoses
    pip install --editable ./
    
    # if [ -f "$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt" ]; then
    #     pip install -r "$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    # fi
    
    cd "$INSTALL_ROOT"
    log "fairseq installed successfully."
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

    pip install numpy scipy soundfile
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
    log "Installing flashlight..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"

    log "Installing flashlight-text..."
    pip install flashlight-text

    if [ -d "$INSTALL_ROOT/sequence" ]; then
        log "Flashlight sequence repository already exists. Updating..."
        cd "$INSTALL_ROOT/sequence"
        git pull
    else
        log "Cloning flashlight sequence repository..."
        git clone https://github.com/flashlight/sequence.git "$INSTALL_ROOT/sequence"
        cd "$INSTALL_ROOT/sequence"
    fi
    
    # pip install .
    export USE_CUDA=1
    cmake -S . -B build
    cmake --build build --parallel
    cd build && cd .. # run test
sudo cmake --install build

    
    log "Flashlight installed successfully."
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
    ./install_protobuf.sh
    
    sudo apt update
    sudo apt install -y libprotobuf-dev protobuf-compiler

    #setting up pyenv to tackle errors 
    curl -fsSL https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.10.16
    pyenv global 3.10.16
    pip install protobuf #this installs the python version 
    pip install 
    ./install_clif.sh

    cd "$PYKALDI_ROOT/tools"
    sudo apt-get install -y python2.7  # Kaldi build system may require Python 2

    sudo ./install_mkl.sh

    if [ ! -d "$KALDI_ROOT" ]; then
        log "Cloning and building Kaldi for PyKaldi..."
        ./install_kaldi.sh
    else
        log "Kaldi already exists at $KALDI_ROOT"
    fi

    cd "$PYKALDI_ROOT"
    python setup.py install
    
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
