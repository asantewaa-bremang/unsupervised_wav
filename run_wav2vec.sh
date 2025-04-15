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
DATASETS=$1 #/path/to/unlabelled/audio_data 
UNLABELLED_TEXT=$2 #/path/to/unlabelled_text_file 
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
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR"  \
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

setup_env() {
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

# Function to update a file's empty variables with provided values, this is applied in train.sh and the other files in the self training phase of the project
update_file_variables() {
    if [ "$#" -lt 2 ]; then
        echo "Usage: update_file_variables <filename> <var1=value> <var2=value> ..."
        return 1
    fi

    local file="$1"
    shift  # Remove filename from arguments

    # Create a backup of the original file
    # cp "$file" "$file.bak"

    # Loop through remaining arguments (variable=value pairs)
    for arg in "$@"; do
        local var_name=$(echo "$arg" | cut -d '=' -f 1)   # Extract variable name
        local var_value=$(echo "$arg" | cut -d '=' -f 2-) # Extract everything after '='

        # Use `sed` to replace only existing variables in the file
        echo $var_name
        echo $var_value
        local escaped_value=$(echo "$var_value" | sed 's/[\/&]/\\&/g')

        # Use `sed` to replace only existing variables in the file
        sed -i "s|^$var_name=.*$|$var_name=$escaped_value|" "$file"
      
        echo "done"
    done

    echo "Updated $file successfully."
}

#used to comment a specific line in the train.sh script. it prevent processing the text for the train_gt folder
comment_line() {
    # Ensure correct number of arguments
    if [ "$#" -ne 2 ]; then
        echo "Usage: comment_line <file_name> <line_to_comment>"
        return 1
    fi

    file="$1"         # File to modify
    line_to_find="$2" # Line of code to comment out

    # Escape special characters in the input to prevent sed errors
    escaped_line_to_find=$(printf "%s" "$line_to_find" | sed 's/[\/&]/\\&/g')

    # Check if the line exists in the file
    if grep -Fxq "$line_to_find" "$file"; then
        echo "Found line in $file: $line_to_find"

        # Backup the file before modification
        # cp "$file" "$file.bak"

        # Comment out the matching line
        sed -i "s|^$escaped_line_to_find|# $escaped_line_to_find|" "$file"

        echo "Commented out the line in $file"
    else
        echo "Line not found in $file. No changes made."
    fi
}


#this is used to update the evaluation scripts for with the best self-trained model 
get_best_path_pipeline() {
    local input_file="$1"

    # --- Input Validation ---
    if [ -z "$input_file" ]; then
        echo "Usage: get_best_path_pipeline <input_log_file>" >&2
        return 1
    fi
    if [ ! -r "$input_file" ]; then
        echo "Error: File '$input_file' not found or not readable." >&2
        return 1
    fi
    # --- End Input Validation ---

    # 1. Filter lines containing both "INFO:root:" and "wer [...]%"
    #    Use grep -E for extended regex OR chain two greps. Chaining is safer.
    local filtered_lines
    filtered_lines=$(grep 'INFO:root:' "$input_file" | grep 'wer [0-9.]*%' || true)
    # '|| true' prevents script exit if grep finds nothing

    # Check if any lines matched
    if [ -z "$filtered_lines" ]; then
         # echo "Warning: No lines matching required format found." >&2 # Optional warning
         return 0 # Indicate success, but no result found
    fi

    # 2. Transform matching lines to "WER PATH" format using POSIX awk
    #    This awk script extracts WER and path from the already filtered lines.
    local transformed_lines
    transformed_lines=$(echo "$filtered_lines" | awk '
    {
        # Find WER using match() and substr()
        if (match($0, /wer [0-9.]+\%/)) {
             wer_match_str = substr($0, RSTART, RLENGTH); # e.g., "wer 71.71%"
             if (match(wer_match_str, /[0-9.]+/)) {
                 current_wer = substr(wer_match_str, RSTART, RLENGTH);
             } else { next; } # Skip if number extraction fails
        } else { next; } # Skip if WER pattern not found

        # Find Path using match() and substr()
        if (match($0, /INFO:root:[^:]+:/)) {
            path_match_str = substr($0, RSTART, RLENGTH); # e.g., "INFO:root:/path:"
            sub(/^INFO:root:/, "", path_match_str);
            sub(/:$/, "", path_match_str);
            current_path = path_match_str;
        } else { next; } # Skip if path pattern not found

        # Print in "WER PATH" format
        print current_wer, current_path;
    }')

    if [ -z "$transformed_lines" ]; then
        # This might happen if extraction failed even after initial grep
        # echo "Warning: Could not extract WER/Path from matching lines." >&2
        return 0
    fi

    # 3. Sort numerically based on the first field (WER)
    # 4. Take the first line (lowest WER)
    local best_line
    best_line=$(echo "$transformed_lines" | sort -n -k1,1 | head -n 1)

    # 5. Extract the path (everything after the first space)
    #    Using awk is robust for paths with spaces
    local best_path
    best_path=$(echo "$best_line" | awk '{ $1=""; sub(/^ /,""); print }')

    echo "$best_path"
    return 0
}


#this script is there to allow processing of ground truth text in the self training for only validation dataset
update_script_with_condition() {
    local SCRIPT_FILE="$1"
    
    # Create a backup of the original file
    cp "$SCRIPT_FILE" "$SCRIPT_FILE.bak"
    echo "Backup created at ${SCRIPT_FILE}.bak"
    
    # Export the filename so the Python code can access it
    export SCRIPT_FILE
    
    python3 << 'EOF'
import re
import sys
import os

    # Retrieve the target script filename from the environment variable
script_file = os.environ['SCRIPT_FILE']

    # Define a regex pattern that matches the ground truth cp command
    # The pattern is designed to match:
    # cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $data_dir/$x_gt/
pattern = re.compile(r'cp\s+\$data_dir/\$x/\{feats\.scp,cmvn\.scp,utt2spk,spk2utt\}\s+\$data_dir/\$x_gt/')

    # The conditional block to insert
conditional_block = '''if [[ "$x" == "$valid_name" ]]; then
        python local/copy_aligned_text.py < $w2v_dir/$x.$label > $data_dir/$x_gt/text
fi
    '''

    # Read the original script
with open(script_file, 'r') as f:
        lines = f.readlines()

new_lines = []
found = False

for line in lines:
        new_lines.append(line)
        # When the line matches our pattern, insert the conditional block immediately after it
        if pattern.search(line):
            found = True
            # Prevent duplicate insertion by checking if the conditional block already exists
            if not any('if [[ "$x" == "$valid_name" ]]; then' in l for l in lines):
                new_lines.append(conditional_block + "\n")

if not found:
        sys.stderr.write("Pattern not found in file. No modifications done.\n")
        sys.exit(1)

    # Write back the modified script
with open(script_file, 'w') as f:
        f.writelines(new_lines)

print(f"Modification completed. {script_file} updated.")
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
    echo "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py"
    echo "$DATASETS"
    echo "$MANIFEST_DIR"
    
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent "$valid_pct"
    
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

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/train.tsv" --vads "$MANIFEST_DIR/train.vads" --out "$DATA_ROOT/processed_audio"
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
        "$DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent "$valid_pct"
    
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

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh" "$MANIFEST_DIR" $CLUSTERING_DIR $MODEL 512 14
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

#=================Evaluating the GANS =============================================
transcription_gans_viterbi(){

   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

  if is_completed "transcription_gans_viterbi"; then
        log "Skipping gans viterbi transcription  (already completed)"
        return 0
    fi

    log "gans viterbi transcription."
    mark_in_progress "gans viterbi transcription"

#updating parameters viterbi.yaml 
update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml" fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" fairseq.task.text_data="$TEXT_OUTPUT/phones/" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"

#evaluating the GANS models for validation phones
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"


#evaluating the GANS models for validation phones
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_PHONES"

  if [ $? -eq 0 ]; then
        mark_completed "transcription_gans_viterbi"
        log "gans viterbi transcription successfully"
    else
        log "ERROR: gans viterbi transcription failed"
        exit 1
    fi

}

transcription_gans_kaldi(){

   if is_completed "transcription_gans_kaldi"; then
        log "Skipping gans kaldi transcription  (already completed)"
        return 0
    fi

    log "gans kaldi transcription."
    mark_in_progress "gans kaldi transcription"
   # first step is to make a copy of viterbi and name is as kaldi
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml

 
   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
  
update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml" fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt fairseq.task.labels="wrd" w2l_decoder="KALDI" fairseq.dataset.gen_subset=train fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 results_path="$GANS_OUTPUT_WORDS" 


python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_WORDS" 

 #evaluating for validation words
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_WORDS" 


 if [ $? -eq 0 ]; then
        mark_completed "transcription_gans_kaldi"
        log "gans kaldi transcription successfully"
    else
        log "ERROR: gans kaldi transcription failed"
        exit 1
    fi
  

}



#======================self training -====================
#1. first we copy kaldi_st_selftrain folder into the right place 
self_training()
{
   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   if is_completed "self_training"; then
        log "Skipping self training  (already completed)"
        return 0
    fi

    log "self_training."
    mark_in_progress "self_training"

   #very important step  copy 
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/kaldi_self_train $KALDI_ROOT/egs 

    TRAIN_FILE=$KALDI_ROOT/egs/kaldi_self_train/st/train.sh 

    update_file_variables $TRAIN_FILE w2v_dir="$CLUSTERING_DIR" lab_dir=$GANS_OUTPUT_PHONES out_dir=$ST_OUTPUT arpa_lm="$TEXT_OUTPUT/phones/lm.phones.filtered.04.arpa" arpa_lm_bin="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin/" label=phnc

    comment_line $TRAIN_FILE "  python local/copy_aligned_text.py < \$w2v_dir/\$x.\$label > \$data_dir/\$x_gt/text"

    update_script_with_condition $TRAIN_FILE

    cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
    chmod +x $TRAIN_FILE
    $TRAIN_FILE > $KALDI_ROOT/egs/kaldi_self_train/st/results.txt


     if [ $? -eq 0 ]; then
        mark_completed "self_training"
        log "gans self_training successfully"
    else
        log "ERROR: self_training failed"
        exit 1
    fi
  

}



transcription_HMM_phone_eval()
{
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
   
if is_completed "transcription_HMM_phone_eval"; then
        log "Skipping transcription and evaluation of HMM on phones  (already completed)"
        return 0
    fi

    log "transcription and evaluation of HMM on phones."
    mark_in_progress "transcription_HMM_phone_eval"
   
 DECODE_PHONE=$KALDI_ROOT/egs/kaldi_self_train/st/decode_phone.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #created an output to store best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_PHONE out_dir=$ST_OUTPUT dec_lmparam="${ADDR[-1]%.tra.txt}" dec_exp=${ADDR[-4]} dec_script=$KALDI_ROOT/egs/kaldi_self_train/st/decode.sh dec_splits="valid"
chmod +x  $DECODE_PHONE
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 

$DECODE_PHONE

 if [ $? -eq 0 ]; then
        mark_completed "transcription_HMM_phone_eval"
        log "transcription and evaluation of HMM on phones successfully"
    else
        log "ERROR: transcription and evaluation of HMM on phones failed"
        exit 1
    fi
  
}

transcription_HMM_word_eval()
{
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   if is_completed "transcription_HMM_word_eval"; then
        log "Skipping HMM training on word (already completed)"
        return 0
    fi

    log "HMM training on word."
    mark_in_progress "HMM training on word"
   
 DECODE_WORD=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step1.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #results.txt is an output that stores best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_WORD w2v_dir="$CLUSTERING_DIR" out_dir=$ST_OUTPUT lexicon=$TEXT_OUTPUT/lexicon_filtered.lst wrd_arpa_lm=$TEXT_OUTPUT/kenlm.wrd.o40003.arpa wrd_arpa_lm_bin=$TEXT_OUTPUT/kenlm.wrd.o40003.bin dec_exp=${ADDR[-4]} dec_splits="valid" dec_script=steps/decode_fmllr.sh

chmod +x  $DECODE_WORD

cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD > $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt

 if [ $? -eq 0 ]; then
        mark_completed "transcription_HMM_word_eval"
        log "HMM training on word successfully"
    else
        log "ERROR: HMM training on word failed"
        exit 1
    fi
  

}

transcription_HMM_word2_eval()
{
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   if is_completed "transcription_HMM_word2_eval"; then
        log "Skipping transcription and evaluation of HMM on word  (already completed)"
        return 0
    fi

    log "transcription and evaluation of HMM on word."
    mark_in_progress "transcription and evaluation of HMM on word"
   
 DECODE_WORD2=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step2.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt) #created an output to store best hmm results  #not necessary, we can view and place our values 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

dec_splits="train valid"
update_file_variables $DECODE_WORD2 out_dir=$ST_OUTPUT dec_exp=${ADDR[-4]} dec_splits="valid" dec_lmparam="${ADDR[-1]%.tra.txt}"
sed -i 's|\(decode\${dec_suffix}_[^/]*\)/scoring|\1.si/scoring|g' "$DECODE_WORD2"

chmod +x  $DECODE_WORD2
echo "here"
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD2

 if [ $? -eq 0 ]; then
        mark_completed "transcription_HMM_word2_eval"
        log "transcription and evaluation of HMM on word successfully"
    else
        log "ERROR: transcription and evaluation of HMM on word  failed"
        exit 1
    fi
  

}

# ==================== MAIN EXECUTION ====================

main() {
    
    create_dirs #creates directories for storing outputs from the different steps 

    activate_venv  
    setup_env #add kenlm and kaldi to the LD_LIBRARY directory
    
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

#===========================================================Evaluation ===========================================

#the trained checkpoints from train_gans will be stored in a folder called multirun. The checkpoint will be stored in this format 
#multirun --
 #         |
 #         day/month/year --
 #                         |
 #                         time --
 #                                |
 #                                checkpoint_best.pt
 #                                 checkpoint_last.pt
 #therefore it is advisable to manually provide the path to the exact checkpoint to use under the variable $CHECKPOINT_DIR  in the run_wav2vec.sh script
 
# '''
# Transcriptions from the GAN model 
#      transcription_gans_viterbi: outputs phonetic transcription in variable name $GANS_OUTPUT_PHONES
#      transcription_gans_kaldi: outputs word transcription in variable name $GANS_OUTPUT_WORDS
# '''
    transcription_gans_viterbi  #for these we need both train and validation since the train will be used by the HMM
    transcription_gans_kaldi

# '''
#   Does Hidden Markov training on the outputs from the transcription_gans_viterbi and prepare_audio 
#  outputs three HMM 
# '''
    self_training

# '''
# transcribes and evaluates the validation set using the best HMM Model 
# '''
    transcription_HMM_phone_eval
    transcription_HMM_word_eval
    transcription_HMM_word2_eval
    
    log "Pipeline completed successfully!"
}

# Run the main function
main
