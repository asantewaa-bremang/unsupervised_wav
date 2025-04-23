#!/bin/bash

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

# Main directories
#.... directories to add to root.......
DIR_PATH="$HOME/unsupervised_wav"
DATA_ROOT="$DIR_PATH/data" #find a way to deal with this
FAIRSEQ_ROOT="$DIR_PATH/fairseq"
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"  # Path to KenLM installation
VENV_PATH="$DIR_PATH/venv"    # Path to virtual environment (optional)
KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast"


#fairseq file paths with slight changes made 
SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$DIR_PATH/fairseq/examples/speech_recognition/kaldi/kaldi_initializer.py"


# adding to system paths
TRAIN_DATASETS=$1 #/path/to/unlabelled/train_audio_data 
VAL_DATASETS=$2 #/path/to/unlabelled/validation_audio_data 
UNLABELLED_TEXT=$3 #/path/to/unlabelled_text_file 
NEW_SAMPLE_PCT=0.5
MIN_PHONES=15
NEW_BATCH_SIZE=32

#models 
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin" 
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt"

# Dataset specifics

DATASET_NAME="librispeech"

# Output directories (will be created if they don't exist)
MANIFEST_DIR="$DATA_ROOT/manifests"
NONSIL_AUDIO="$DATA_ROOT/processed_audio/"
MANIFEST_NONSIL_DIR="$DATA_ROOT/manifests_nonsil"
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET_NAME"
RESULTS_DIR="$DATA_ROOT/results/$DATASET_NAME"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME"
LOG_DIR="$DATA_ROOT/logs/$DATASET_NAME"
TEXT_OUTPUT="$DATA_ROOT/text"
GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"
GANS_OUTPUT_WORDS="$DATA_ROOT/transcription_words"
ST_OUTPUT="$DATA_ROOT/selftraining"


# Checkpoint file to track progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"

# ==================== HELPER FUNCTIONS ====================

# Create directories if they don't exist
create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR" "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$GANS_OUTPUT_PHONES" \
             "$TEXT_OUTPUT" "$GANS_OUTPUT_PHONES" "$ST_OUTPUT"
}

# Log message with timestamp
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_DIR/pipeline.log"
}

# Check if a step has been completed
is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Mark a step as completed
mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

# Mark a step as in progress
mark_in_progress() {
    local step="$1"
    # First remove any existing in-progress markers for this step
    if [ -f "$CHECKPOINT_FILE" ]; then
        sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}

# Check if a step is in progress (for recovery after crash)
is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Activate virtual environment if provided
activate_venv() {
    if [ -n "$VENV_PATH" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi
}

setup_path() {
    export HYDRA_FULL_ERROR=1
    export LD_LIBRARY_PATH="${KALDI_ROOT}/src/lib:${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
}


#fixing all occurrences of np.NaN with np.nan because of dependency upgrades
fixing_nan() {
    # Define the target file
    TARGET_FILE="$SPEECHPROCS"
    
    # Check if the file exists
    if [ -f "$TARGET_FILE" ]; then
        echo "Fixing NumPy np.NaN issue in $TARGET_FILE..."
        
        # Use sed to replace all occurrences of np.NaN with np.nan
        sed -i 's/np.NaN/np.nan/g' "$TARGET_FILE"
        
        echo "Fix applied successfully!"
    else
        echo "Error: $TARGET_FILE not found!"
        exit 1
    fi

}

# updating code in SPEECHPROCS to return outputs that are compatible with code change in vads.py
fixing_sflux() {
    TARGET_FILE="$SPEECHPROCS"

    # Check if the file exists
    if [ -f "$TARGET_FILE" ]; then
        echo "Updating sflux() to return two values in $TARGET_FILE..."
        
        # Find and modify the return statement inside sflux()
        sed -i '/def sflux/,/return/ s/^ *return .*/    return s_flatness, n_frames/' "$TARGET_FILE"
        
        # Confirm the fix by printing the modified return statement
        echo "Updated return statement in sflux():"
        grep "return " "$TARGET_FILE"
        
        echo "Fix applied successfully!"
    else
        echo "Error: $TARGET_FILE not found!"
        exit 1
    fi
}

# Update sample_pct in the file 'prepare_audio'-- the variable measures the amount of audio dataset
#to us in generating k-mean clusters 
update_sample_pct()
{

# This regex matches '--sample-pct' followed by any whitespace and a number (integer or decimal)
# and replaces it with '--sample-pct' followed by the new value.
    sed -i.bak -E "s/(--sample-pct[[:space:]]+)[0-9]*\.?[0-9]+/\1${NEW_SAMPLE_PCT}/g" $PREPARE_AUDIO
    echo "Updated '--sample-pct' to ${NEW_SAMPLE_PCT} in 'prepare_audio'. Backup saved as 'prepare_audio.bak'."

}

# Update batch_size in the file 'prepare_audio'
update_batch_size()
{
    sed -i.bak -E "s/(--batch-size[[:space:]]+)[0-9]+/\1${NEW_BATCH_SIZE}/g" $PREPARE_AUDIO
    echo "Updated '--batch-size' to ${NEW_BATCH_SIZE} in 'prepare_audio'. Backup saved as 'prepare_audio.bak'."

}


#update is done in add-self-loop-simple.cc to replace std_endl with "\n" since std_endl is not compatible with pykaldi installation for text preprocessing
replace_std_endl() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        echo "Error: File '$input_file' not found!"
        return 1
    fi

    # Use sed to replace std::endl with \n and save the output
    sed -i 's/std::endl/"\\n"/g' "$input_file"

    echo "Replacement done in '$input_file'"
}

#this is to help updates yaml files needs for sucessful run of certain processes.
add_to_existing_yaml() {
  if [ "$#" -ne 4 ]; then
    echo "Usage: add_to_existing_yaml <config_file> <parent_key> <new_key> <new_value>"
    return 1
  fi

  local config_file="$1"
  local parent_key="$2"
  local new_key="$3"
  local new_value="$4"

  if [ ! -f "$config_file" ]; then
    echo "Error: File '$config_file' not found."
    return 1
  fi


  # Convert input to a valid yq-compatible path
  full_path=".$parent_key.$new_key"


  # Check if the parent key exists
  yq -y "(${full_path}) = ${new_value}" $config_file  > tmp.yaml && mv tmp.yaml $config_file 

#   echo "passed 0"
  if yq "$full_path" "$config_file" >/dev/null 2>&1; then
    # Parent exists, update the file
    # echo "passed 1"
    yq -n  "$full_path = $new_value" "$config_file"

    echo "Updated '$parent_key' with '$new_key: $new_value'."
  else
    # Parent doesn't exist, create it
    yq -n "$full_path = $new_value"  "$config_file"
    echo "Created new parent '$parent_key' and added '$new_key: $new_value'."
  fi

  echo "Configuration file '$config_file' updated successfully."
}

delete_yaml_field() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: delete_yaml_field <config_file> <yaml_path>"
    return 1
  fi

  local config_file="$1"
  local yaml_path="$2"

  if [ ! -f "$config_file" ]; then
    echo "Error: File '$config_file' not found."
    return 1
  fi

  # Use yq to delete the field. The jq filter del() removes the key.
  yq -y "del(${yaml_path})" "$config_file" > tmp.yaml && mv tmp.yaml "$config_file"

  if [ $? -eq 0 ]; then
    echo "Deleted field ${yaml_path} from ${config_file}"
  else
    echo "Failed to update ${config_file}"
    return 1
  fi
}

#script is used to update a yaml file 
update_yaml_config() {
    if [ "$#" -lt 2 ]; then
        echo "Usage: update_yaml_config <config_file> <key=value> [<key=value> ...]"
        return 1
    fi

    local CONFIG_FILE="$1"
    shift  # Remove config file from arguments

    # Run embedded Python script
    python3 - "$CONFIG_FILE" "$@" <<EOF
import yaml
import sys
import os

config_file = sys.argv[1]
updates = dict(arg.split("=", 1) for arg in sys.argv[2:])

if not os.path.exists(config_file):
        print(f"Error: File '{config_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load YAML
with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file) or {}

    # Function to update nested keys
def set_nested_value(data, key_path, value):
        keys = key_path.split(".")
        d = data
        for key in keys[:-1]:  
            d = d.setdefault(key, {})  # Create subkeys if missing
        d[keys[-1]] = value  # Set the final key

    # Apply updates
for key, value in updates.items():
        print(f"Updating: {key} = {value}")
        set_nested_value(yaml_data, key, value)

    # Save updated YAML
with open(config_file, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

print(f"Configuration file '{config_file}' updated successfully.")
EOF
}


# ==================== MAIN STEPS ====================

# Step 1: Create data manifests
create_manifests() {
    local valid_pct="${1:-$VALID_PERCENT}"  # Use provided value or default from config
   
 
    local step_name="create_manifests_${valid_pct//./_}" 
    if is_completed "create_manifests"; then
        log "Skipping manifest creation (already completed)"
        return 0
    fi
    
    log "Creating data manifests..."
    mark_in_progress "create_manifests"
    
    # Adjust this command according to your dataset
    # echo "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py"
    # echo "$DATASETS"
    # echo "$MANIFEST_DIR"
    
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TRAIN_DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent 0.0 #"$valid_pct"

   python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$VAL_DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent 1.0 #"$valid_pct"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "create_manifests"
        log "Manifest creation completed successfully"
    else
        log "ERROR: Manifest creation failed"
        exit 1
    fi
}


# Step 2: create vads files out of the audios 
create_rVADfast() { 
    
    # # fixing certain code errors in the rvads  
    fixing_sflux #this script changes the sflux function to return both ft and n_frames
    fixing_nan # replaces old numpy NaN to modern numpy nan

    if is_completed "create_rVADfast"; then
        log "Skipping audio silence removal (already completed)"
        return 0
    fi
    
    
    log "removing silence from audios"
    mark_in_progress "removing silence from audios"
    python "$DIR_PATH/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"
    python "$DIR_PATH/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/valid.tsv" > "$MANIFEST_DIR/valid.vads"
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "create_rVADfast"
        log "silence removed successfully"
    else
        log "ERROR: silence removal  failed"
        exit 1
    fi
}

# Step 3: Remove silence from audios with vads files 
remove_silence() {

   if is_completed "remove_silence"; then
        log "Skipping audio silence removal1 (already completed)"
        return 0
    fi
    
    
    log "removing silence from audios1"
    mark_in_progress "removing silence from audios1"

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/train.tsv" --vads "$MANIFEST_DIR/train.vads" --out "$NONSIL_AUDIO/train"
    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/valid.tsv" --vads "$MANIFEST_DIR/valid.vads" --out "$NONSIL_AUDIO/val"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "remove_silence"
        log "silence1 removed successfully"
    else
        log "ERROR: silence removal  failed"
        exit 1
    fi

}

#Step 4: create new manifest files for train and validation set with no silence 
create_manifests_nonsil() {
    local valid_pct="${1:-$VALID_PERCENT}"  # Use provided value or default from config
    # echo $valid_pct
    local step_name="create_manifests_${valid_pct//./_}" 
    if is_completed "create_manifests_nonsil"; then
        log "Skipping nonsil manifest creation (already completed)"
        return 0
    fi
    
    log "Creating data manifests..."
    mark_in_progress "create_manifests_nonsil"
    
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/train" \
        --dest "$MANIFEST_NONSIL_DIR" \
        --ext wav \
        --valid-percent 0.0 #"$valid_pct"

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/val" \
        --dest "$MANIFEST_NONSIL_DIR" \
        --ext wav \
        --valid-percent 1.0 #"$valid_pct"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "create_manifests_nonsil"
        log "nonsil Manifest creation completed successfully"
    else
        log "ERROR: nonsil Manifest creation failed"
        exit 1
    fi
}


#Step 5: Prepare audio file
#a. 
prepare_audio() {
export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT=$KALDI_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"

   update_sample_pct #personal scripts added to change sample_pct variable in prepare_audio.sh
   update_batch_size #personal scripts added to change batch_size variable in prepare_audio.sh  


   export KENLM_ROOT="$KENLM_ROOT"


   if is_completed "prepare_audio"; then
        log "Skipping audio preparation (already completed)"
        return 0
    fi
    
    log "audio preparation"
    mark_in_progress "audio preparation"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh" "$MANIFEST_NONSIL_DIR" $CLUSTERING_DIR $MODEL 512 14
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "prepare_audio"
        log "audio preparation successfully"
    else
        log "ERROR: audio preparation  failed"
        exit 1
    fi

}

#======================Text preparation =================================
# unsupervised/wav2vec-U/libri_dataset/librispeech-lm-norm_4k.txt
prepare_text() {
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT=$KALDI_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"

   if is_completed "prepare_text"; then
        log "Skipping text preparation (already completed)"
        return 0
    fi

    log "audio preparation."
    mark_in_progress "audio preparation"
    replace_std_endl $ADD_SELF_LOOP_SIMPLE
    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh" en $UNLABELLED_TEXT $TEXT_OUTPUT $MIN_PHONES G2P $FASTTEXT_LIB_MODEL 0.25

    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "prepare_text"
        log "text preparation successfully"
    else
        log "ERROR: text preparation  failed"
        exit 1
    fi

}
#=========================== GANS training and preparation ==============================
train_gans(){
export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"


   if is_completed "train_gans"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "gans training"
   

update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" task.text_data="$TEXT_OUTPUT/phones/" task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" model.code_penalty=2,4 model.gradient_penalty=1.5 model.smoothness_weight=0.5

add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.discriminator.optimizer lr [0.004]
add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.generator.optimizer lr [0.004]
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.generator.optimizer.amsgrad 
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.discriminator.optimizer.amsgrad

   PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    -m --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
    model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)' \
    2>&1 | tee $RESULTS_DIR/training1.log

   if [ $? -eq 0 ]; then
        mark_completed "train_gans"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}













# ==================== MAIN EXECUTION ====================

main() {
    
    create_dirs #creates directories for storing outputs from the different steps 

    activate_venv  
    setup_path#add kenlm and kaldi to the LD_LIBRARY directory
    
    log "Starting wav2vec unsupervised pipeline for $DATASET"
 
   log "
   it creates a manifest files for the audio dataset
audio format
   "
    create_manifests 0 

      #creates new manifest with silence removed
    
    create_rVADfast # identifies the sequence of silence in an audio 
    remove_silence # removes the silence sequence found by rvad in the audio
    create_manifests_nonsil 0.1


   # Train GANS: 
   #     prepare_audio:  processes the unlabelled audio 
   #     prepare_text:
   #     train_gans:

    prepare_audio 
    prepare_text  
    train_gans


    
    log "Pipeline completed successfully!"
}

# Run the main function
main
