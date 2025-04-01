#!/bin/bash

# Wav2Vec Unsupervised Pipeline Setup Script
# This script handles all the necessary installations and environment setup
# for running the fairseq wav2vec unsupervised pipeline

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# ==================== CONFIGURATION ====================
# Set these variables according to your environment

# Main directories
INSTALL_ROOT="$HOME/unsupervised_wav"
FAIRSEQ_ROOT="$INSTALL_ROOT/fairseq"
KENLM_ROOT="$INSTALL_ROOT/kenlm"
VENV_PATH="$INSTALL_ROOT/venv"
KALDI_ROOT="$INSTALL_ROOT/pykaldi/tools/kaldi"
RVADFAST_PATH="$INSTALL_ROOT/rVADfast/src/rVADfast"
PYKALDI_ROOT="$INSTALL_ROOT/pykaldi"


# CUDA version (for PyTorch installation)
CUDA_VERSION="11.7"  # Options: 10.2, 11.3, 11.6, 11.7, etc.



# Python version
PYTHON_VERSION="3.10"  # Options: 3.7, 3.8, 3.9

# ==================== HELPER FUNCTIONS ====================

# Log message with timestamp
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
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
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            libboost-all-dev \
            libeigen3-dev \
            libatlas-base-dev \
            libfftw3-dev \
            libopenblas-dev \
            python${PYTHON_VERSION} \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv \
            pip \
            git \
            wget \
            zlib1g-dev
    
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
    
    # Upgrade pip
    # pip install --upgrade pip
    
    log "Python virtual environment setup completed."
}

# Step 3: Install PyTorch and related packages
install_pytorch() {
    log "Installing PyTorch and related packages..."
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Install PyTorch with CUDA support if available
    if command_exists nvcc; then
        # CUDA is available
        log "CUDA detected. Installing PyTorch with CUDA support..."
        pip install torch 
        
    fi
    
    # Install other required packages
    pip install numpy scipy tqdm sentencepiece soundfile librosa editdistance tensorboardX faiss-gpu npy-append-array kenlm yp pyparsing kaldi-io

    # sudo apt-get install zsh
    log "PyTorch and related packages installed successfully."
}

# Step 4: Clone and install fairseq
install_fairseq() {
    log "Cloning and installing fairseq..."
    cd "$INSTALL_ROOT"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists. Updating..."
        cd "$FAIRSEQ_ROOT"
        git pull
    else
        log "Cloning fairseq repository..."
        git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_ROOT"
        cd "$FAIRSEQ_ROOT"
    fi
    
    # Install fairseq
    pip install --editable ./
    
    # Install additional requirements for wav2vec
    if [ -f "$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt" ]; then
        pip install -r "$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    fi
    
    cd "$INSTALL_ROOT"
    log "fairseq installed successfully."
}

#step : Install RVAD for audio silence removal
install_rVADfast() {
    log "Cloning and installing rVADfast..."
    cd "$INSTALL_ROOT"
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"

    if [ -f $RVADFAST_PATH ]; then
        log "RVADFAST already exists. Skipping download."
    else
        git clone https://github.com/zhenghuatan/rVADfast.git
        if [ -f "$RVADFAST_PATH/requirements.txt" ]; then
            pip install -r "$RVADFAST_PATH/requirements.txt"
        fi
    fi


}

# Step 5: Clone and build KenLM
install_kenlm() {
    log "Cloning and building KenLM..."
    cd "$INSTALL_ROOT"
    if [ -d "$KENLM_ROOT" ]; then
        log "KenLM repository already exists."
    else
        log "Cloning KenLM repository..."
        git clone https://github.com/kpu/kenlm.git "$KENLM_ROOT"
    fi
    
    # Build KenLM
    cd "$KENLM_ROOT"
    if [ -d "build" ]; then
        log "KenLM build directory already exists. Skipping build step."
    else
        mkdir -p build
        cd build
        cmake ..
        make -j 4
    fi
    
    log "KenLM built successfully."
}



#install flashlight
install_flashlight() {

    log " installing flashlight..."
    cd "$INSTALL_ROOT"

    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"

    log " installing flashlight-text..."
    pip install flashlight-text

    log "installing flashlight-sequence..."

    git clone https://github.com/flashlight/sequence
    cd sequence
    pip install .
    
   
    log "flashlight installed successfully."
}


#Install pykald
install_pykaldi()
{
    log " installing pykaldi..."
    cd "$INSTALL_ROOT"

    
    # # Activate virtual environment
    # source "$VENV_PATH/bin/activate"

    # # # #installing important dependencies 
    python3 -m pip install numpy pyparsing ninja wheel setuptools
    python -m pip install --upgrade pip setuptools
    git clone https://github.com/pykaldi/pykaldi.git
    sudo apt-get update
    sudo apt-get install sox

    cd "$INSTALL_ROOT/pykaldi/tools"
    
    ./check_dependencies.sh
    
    #build protobuf 
    ./install_protobuf.sh

    sudo apt-get install libprotobuf-dev protobuf-compiler


    #build clif 
    ./install_clif.sh

    #build kaldi 
    deactivate
    sudo apt-get install python2.7
        source "$VENV_PATH/bin/activate"
        git clone -b pykaldi_02 https://github.com/pykaldi/kaldi.git
        cd kaldi/tools
        git pull
       
        cd ../..
        sudo ./install_mkl.sh
        
        ./install_kaldi.sh
        cd ..

    #build pykaldi and wheel 
    python3 setup.py install
    python3 setup.py bdist_wheel


}

#  Download pre-trained wav2vec model
download_pretrained_model() {
    log "Downloading pre-trained wav2vec model..."
    
    # Create directory for pre-trained models
    mkdir -p "$INSTALL_ROOT/pre-trained"
    cd "$INSTALL_ROOT/pre-trained"
    
    # Check if model already exists
    
    if [ -f "wav2vec_vox_new" ]; then
        log "Pre-trained model already exists. Skipping download."
    else
        # Download pre-trained wav2vec model
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt
    fi
    
    log "Pre-trained model downloaded successfully."
}

download_languageIdentification_model() {
    log "Downloading pre-trained wav2vec model..."
    
    # Create directory for pre-trained models
    mkdir -p "$INSTALL_ROOT/lid_model"
    cd "$INSTALL_ROOT/lid_model"
    
    # Check if model already exists
    
    if [ -f "lid.176.bin" ]; then
        log "Lid model already exists. Skipping download."
    else
        # Download pre-trained wav2vec model
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi
    
    log "LID model downloaded successfully."
}



# Create a configuration file
create_config_file() {
    log "Creating a configuration file..."
    
    cat > "$INSTALL_ROOT/wav2vec_config.sh" << EOF
# Wav2Vec configuration file
# Source this file to set up environment variables

# Main directories
export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
export KENLM_ROOT="$KENLM_ROOT/bin/"
export VENV_PATH="$VENV_PATH"
export KALDI_ROOT="$KALDI_ROOT"
export PYKALDI_ROOT="$PYKALDI_ROOT"

# Add KenLM binaries to PATH
export PATH="\$KENLM_ROOT/build/bin:\$PATH"


# Add rVADfast binaries to PATH (correcting missing `$PATH` reference)
export PATH="\$RVADFAST/rVADfast/src/rVADfast:\$PATH"

# Function to activate the environment
activate_wav2vec_env() {
    source "$VENV_PATH/bin/activate"
    echo "Wav2Vec environment activated."
}

# Export the function
export -f activate_wav2vec_env
EOF
    
    log "Configuration file created at $INSTALL_ROOT/wav2vec_config.sh"
    log "To use, run: source $INSTALL_ROOT/wav2vec_config.sh"
}

# ==================== MAIN EXECUTION ====================

main() {
    create_dirs
    setup_venv
    install_fairseq
    log "Starting Wav2Vec environment setup..."
    
    
    # install_system_deps
    install_rVADfast
    install_pytorch
    install_flashlight
    install_pykaldi
    
    install_kenlm
    download_pretrained_model
    download_languageIdentification_model
    create_config_file
    
    log "Setup completed successfully!"
    log "------------------------------------------------------"
}

# Run the main function
main
