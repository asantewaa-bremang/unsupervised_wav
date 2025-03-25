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
DIR_PATH="$HOME/wav2vec_setup2"
DATA_ROOT="$DIR_PATH/data" #find a way to deal with this
FAIRSEQ_ROOT="$DIR_PATH/fairseq"
KENLM_ROOT="$DIR_PATH/kenlm"  # Path to KenLM installation
VENV_PATH="$DIR_PATH/venv"    # Path to virtual environment (optional)
KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast"

SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$DIR_PATH/fairseq/examples/speech_recognition/kaldi/kaldi_initializer.py"


# adding to system paths
DATASETS=/path/to/unlabelled/audio_data #"$HOME/unsupervised/wav2vec-U/libri_dataset/unlabelled_audio"
UNLABELLED_TEXT=/path/to/unlabelled_text_file #"$HOME/unsupervised/wav2vec-U/libri_dataset/librispeech-lm-norm_4k.txt"

MIN_PHONES=15
NEW_BATCH_SIZE=32

#models 
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin" 
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt"


# Dataset specifics

DATASET="librispeech"


# Output directories (will be created if they don't exist)
MANIFEST_DIR="$DATA_ROOT/manifests"
# FEATURES_DIR="$DATA_ROOT/features/$DATASET"
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET"
# PHNDICT_DIR="$DATA_ROOT/phoneme_dictionaries/$DATASET"
# UNITS_DIR="$DATA_ROOT/units/$DATASET"
LM_DIR="$DATA_ROOT/language_models/$DATASET"
RESULTS_DIR="$DATA_ROOT/results/$DATASET"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET"
LOG_DIR="$DATA_ROOT/logs/$DATASET"
TEXT_OUTPUT="$DATA_ROOT/text"
GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"
GANS_OUTPUT_WORDS="$DATA_ROOT/transcription_words"
ST_OUTPUT="$DATA_ROOT/selftraining"
# ST_OUTPUT="$DATA_ROOT/st_phones_st"
# ST_OUTPUT_WORD="$DATA_ROOT/st_word"

# Checkpoint file to track progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"

# ==================== HELPER FUNCTIONS ====================

# Create directories if they don't exist
# create_dirs() {
#     mkdir  "$MANIFEST_DIR"}
create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR"  \
              "$LM_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$GANS_OUTPUT_PHONES" \
             "$TEXT_OUTPUT" "$GANS_OUTPUT_PHONES" "$ST_OUTPUT"
}

# "$FEATURES_DIR"

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
    # Set base directory for the project
    export DIR_PATH="$HOME/wav2vec_setup"

    # Set project-specific environment variables
    export DATA_ROOT="${DIR_PATH}/data"
    export FAIRSEQ_ROOT="${DIR_PATH}/fairseq"
    export KENLM_ROOT="${DIR_PATH}/kenlm"      # Path to KenLM installation
    export VENV_PATH="${DIR_PATH}/venv"        # Path to virtual environment (optional)
    # export KALDI_ROOT="${DIR_PATH}/kaldi"  # Kaldi installation path
    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
    
    export RVAD_ROOT="${DIR_PATH}/rVADfast/src/rVADfast"
    export HYDRA_FULL_ERROR=1

    # Optionally add directories containing executables to PATH.
    # Adjust these subdirectory names if your project layout is different.
    export PATH="${FAIRSEQ_ROOT}/bin:${KENLM_ROOT}/bin:${KALDI_ROOT}/src/bin:${RVAD_ROOT}:$PATH"

    # Add library directories (if needed) to LD_LIBRARY_PATH.
    # For example, Kaldi libraries and KenLM libraries.
    # export LD_LIBRARY_PATH="${KALDI_ROOT}/src/lib:${KENLM_ROOT}/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${KALDI_ROOT}/src/lib:${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"

}

add_path()
{
    export PYTHONPATH="$RVADFAST_PATH:$PYTHONPATH"
    
    # Make it permanent by adding to ~/.bashrc if it's not already there
    if ! grep -q "$RVADFAST_PATH" ~/.bashrc; then
        echo "export PYTHONPATH=\"$RVADFAST_PATH:\$PYTHONPATH\"" >> ~/.bashrc
        echo "Added $RVADFAST_PATH to PYTHONPATH in ~/.bashrc"
    else
        echo "$RVADFAST_PATH is already in ~/.bashrc"
    fi
    
    # Reload ~/.bashrc so changes take effect immediately
    source ~/.bashrc

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


replace_std_endl() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        echo "Error: File '$input_file' not found!"
        return 1
    fi

    # Use sed to replace std::endl with \n and save the output
    sed -i 's/std::endl/\\n/g' "$input_file"

    echo "Replacement done in '$input_file'"
}


update_yaml_config1() {
  if [ "$#" -lt 2 ]; then
    echo "Usage: update_yaml_config <config_file> <key=value> [<key=value> ...]"
    return 1
  fi

  local config_file="$1"
  shift

  if [ ! -f "$config_file" ]; then
    echo "Error: File '$config_file' does not exist."
    return 1
  fi

  # Create a backup of the original file.
  cp "$config_file" "$config_file.bak"
  echo "Backup of '$config_file' saved as '$config_file.bak'."

  # Loop over each key=value pair.
  for update in "$@"; do
    # Split the update into key and value.
    local key="${update%%=*}"
    local value="${update#*=}"

    if [[ "$key" == *.* ]]; then
      # If the key contains a dot, assume the format is parent.child.
      local parent="${key%%.*}"
      local child="${key#*.}"
      # Use sed to update the block under the parent key.
      # This command finds the range from the parent's line until the next non-indented line,
      # then looks for a line starting (with possible spaces) with the child key.
      sed -i -E "/^[[:space:]]*$parent:/,/^[^[:space:]]/ s|^([[:space:]]*$child:[[:space:]]*).*|\1$value|" "$config_file"
      echo "Updated nested key '$key' to '$value'."
    else
      # For a top-level key, match the beginning of the line (with optional spaces) and replace its value.
      sed -i -E "s|^([[:space:]]*$key:[[:space:]]*).*|\1$value|" "$config_file"
      echo "Updated key '$key' to '$value'."
    fi
  done

  echo "Configuration file '$config_file' has been updated."
}


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

#   echo "Processing YAML: $config_file"
#   echo "Parent Key: $parent_key"
#   echo "New Key: $new_key, New Value: $new_value"

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



update_imports() {
    local file="$1"
    
    # Use sed to modify the import statements
    sed -i -E \
        's/from fairseq import checkpoint_utils, progress_bar, tasks, utils/from fairseq.logging import progress_bar\nfrom fairseq import checkpoint_utils, tasks, utils/' "$file"

    echo "Updated imports in $file"
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


#!/bin/bash

# Function to update a file's empty variables with provided values
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
        # sed -i "s/^$var_name=.*$/$var_name=$var_value/" "$file"
        echo "done"
    done

    echo "Updated $file successfully."
}


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
        cp "$file" "$file.bak"

        # Comment out the matching line
        sed -i "s|^$escaped_line_to_find|# $escaped_line_to_find|" "$file"

        echo "Commented out the line in $file"
    else
        echo "Line not found in $file. No changes made."
    fi
}


#!/usr/bin/env bash

rewrite_file() {
    local input_file="$1"
    local temp_file="temp_file.py"
    # local TEMP_FILE="$DIR_PATH/kaldi/egs/kaldi_self_train/st/local/unsup_select_temp.py"

    # Step 1: Remove any "f args.gt_tra:" prefixes from every line.
    sed -i 's/^f args\.gt_tra:[[:space:]]*//' "$input_file"

    # Step 2: Use awk to comment out the block starting with "if args.gt_tra:".
    # It finds a line (ignoring leading spaces) that starts with "if args.gt_tra:" and
    # then comments out that line and every subsequent line that is more indented than it.
    awk '
    BEGIN { in_block=0; base_indent="" }
    {
      # Check if the current line starts the block.
      if (in_block == 0 && $0 ~ /^[[:space:]]*if args\.gt_tra:/) {
        # Capture the leading spaces (indentation).
        match($0, /^([[:space:]]*)if args\.gt_tra:/, arr);
        base_indent = arr[1];
        in_block = 1;
        # Print the line with a "#" inserted after the base indentation.
        print base_indent "#" substr($0, length(base_indent)+1);
        next;
      }
      # If inside the block, comment lines that are further indented than base_indent.
      if (in_block == 1) {
        if ($0 ~ "^" base_indent "[[:space:]]") {
          print base_indent "#" substr($0, length(base_indent)+1);
          next;
        } else {
          # Block ended when a line is not indented further.
          in_block = 0;
        }
      }
      # Print lines outside the block unchanged.
      print;
    }
    # ' "$input_file" >"$temp_file"
    
    mv $temp_file $input_file

   

    # Replace the original file with the updated file.
   
    echo "File '$input_file' has been updated with the commented block."
}

# Example usage:
# rewrite_file "script.py"

get_best_path() {
    local input_file="$1"
    awk '
    BEGIN {
        best_wer = 1e9;  # Initialize with a very high value.
        best_path = "";
    }
    {
        # Extract the WER value using a regex match.
        if (match($0, /wer ([0-9.]+)%/, arr)) {
            wer = arr[1] + 0;  # Convert to a number.
        } else {
            next;
        }
        # Extract the file path from the INFO:root: portion.
        if (match($0, /INFO:root:([^:]+):/, arr2)) {
            path = arr2[1];
        }
        # If the current WER is lower than the best so far, update best.
        if (wer < best_wer) {
            best_wer = wer;
            best_path = path;
        }
    }
    END {
        print best_path;
    }
    ' "$input_file"
}



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




# Function to comment out lines in a file that match a given regex pattern
comment_line_in_file() {
    local file="$1"         # The file to modify
    local pattern="$2"      # The regex pattern to match the line(s) you want to comment

    # Create a backup of the original file
    cp "$file" "$file.bak"
    echo "Backup created at ${file}.bak"

    python3 <<EOF
    import re

    filename = "$file"
    pattern = re.compile(r"$pattern")

    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # If the line matches the pattern and is not already commented out
        if pattern.search(line) and not line.lstrip().startswith("#"):
            new_lines.append("# " + line)
        else:
            new_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(new_lines)

    print(f"Modification completed. {filename} updated.")
EOF
}

# ==================== MAIN STEPS ====================

# Step 1: Create data manifests
create_manifests() {
    local valid_pct="${1:-$VALID_PERCENT}"  # Use provided value or default from config
    # echo $valid_pct
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
    python "$HOME/wav2vec_setup/addition_scripts/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"
    
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
    # python "$HOME/wav2vec_setup/addition_scripts/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"
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
    
    # Adjust this command according to your dataset
    # echo "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py"
    # echo "$DATASETS"
    # echo "$MANIFEST_DIR"
    
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$DATA_ROOT/processed_audio/unlabelled_audio" \
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

   update_sample_pct #personal scripts added to change sample_pct variable in prepare_audio.sh
   update_batch_size #personal scripts added to change batch_size variable in prepare_audio.sh  


   export KENLM_ROOT="$KENLM_ROOT/build/bin"


   if is_completed "prepare_audio"; then
        log "Skipping audio preparation (already completed)"
        return 0
    fi
    
    #delete later
    source "$HOME/myenv1/bin/activate"
    
    log "audio preparation"
    mark_in_progress "audio preparation"

    # python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/train.tsv" --vads "$MANIFEST_DIR/train.vads" --out "$DATA_ROOT/processed_audio"
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
#    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT=$KALDI_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"

   if is_completed "prepare_text"; then
        log "Skipping text preparation (already completed)"
        return 0
    fi
    
    #delete later
    source "$HOME/myenv1/bin/activate"
 
    log "audio preparation"
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
# export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"
   

update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" task.text_data="$TEXT_OUTPUT/phones/" task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" model.code_penalty=2,4 model.gradient_penalty=1.5 model.smoothness_weight=0.5 checkpoint.save_dir=$RESULTS_DIR  

add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.discriminator.optimizer lr [0.004]
add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.generator.optimizer lr [0.004]
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.generator.optimizer.amsgrad 
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.discriminator.optimizer.amsgrad

   PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    -m --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
    model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)' \
    checkpoint.save_dir=$RESULTS_DIR 2>&1 | tee $RESULTS_DIR/training1.log


}

#=================Evaluating the GANS =============================================
transcription_gans_viterbi(){
   activate_venv #will delete later

#    export HYDRA_FULL_ERROR=1
#    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
#    

#updating parameters viterbi.yaml 
update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml" fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" fairseq.task.text_data="$TEXT_OUTPUT/phones/" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"

#evaluating the GANS models for validation phones
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"


#evaluating the GANS models for validation phones
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_PHONES"

}

transcription_gans_kaldi(){
   activate_venv 

   # first step is to make a copy of viterbi and name is as kaldi
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml

 
#    export HYDRA_FULL_ERROR=1
#    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
  



update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml" fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt fairseq.task.labels="wrd" w2l_decoder="KALDI" fairseq.dataset.gen_subset=train fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 results_path="$GANS_OUTPUT_WORDS" 


python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_WORDS" 

 #evaluating for validation words
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$HOME/unsupervised2/gen_phonomes/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_WORDS" 
  

}




#======================self training -====================
#1. first we copy kaldi_st_selftrain folder into the right place 
self_training()
{
#    export HYDRA_FULL_ERROR=1
#    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   setup_env
   
   #very important step  copy 
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/kaldi_self_train $KALDI_ROOT/egs 

    TRAIN_FILE=$KALDI_ROOT/egs/kaldi_self_train/st/train.sh 

    update_file_variables $TRAIN_FILE w2v_dir="$HOME/unsupervised2/gen_phonomes" lab_dir=$GANS_OUTPUT_PHONES out_dir=$ST_OUTPUT_PHONES arpa_lm="$TEXT_OUTPUT/phones/lm.phones.filtered.04.arpa" arpa_lm_bin="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin/" label=phnc

    # comment_line $TRAIN_FILE "  python local/copy_aligned_text.py < \$w2v_dir/\$x.\$label > \$data_dir/\$x_gt/text"

    update_script_with_condition $TRAIN_FILE

    # #for phonemes
    # cp -r $TEXT_OUTPUT/phones/dict.phn.txt $HOME/unsupervised2/gen_phonomes/dict.phnc.txt #WILL HAVE TO CHANGE THIS AND MAKE IT MORE DYNAMIC

    # #for words 
    # cp -r $TEXT_OUTPUT/words.txt $HOME/unsupervised2/gen_phonomes/dict.wrd.txt #i will have to fix this too and make it dynamic


    cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
    chmod +x $TRAIN_FILE
    $TRAIN_FILE > $KALDI_ROOT/egs/kaldi_self_train/st/results.txt

}

#2. add the path names and make changes 

transcription_HMM_phone_eval()
{
#     export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   
 DECODE_PHONE=$KALDI_ROOT/egs/kaldi_self_train/st/decode_phone.sh

 output=$(get_best_path $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #created an output to store best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_PHONE out_dir=$ST_OUTPUT dec_lmparam="${ADDR[-1]%.tra.txt}" dec_exp=${ADDR[-4]} dec_script=$KALDI_ROOT/egs/kaldi_self_train/st/decode.sh dec_splits="valid"
chmod +x  $DECODE_PHONE
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 

$DECODE_PHONE
}

transcription_HMM_word_eval()
{
#     export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
   
 DECODE_WORD=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step1.sh

 output=$(get_best_path $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #results.txt is an output that stores best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_WORD w2v_dir="$HOME/unsupervised2/gen_phonomes" out_dir=$ST_OUTPUT_PHONES lexicon=$TEXT_OUTPUT/lexicon_filtered.lst wrd_arpa_lm=$TEXT_OUTPUT/kenlm.wrd.o40003.arpa wrd_arpa_lm_bin=$TEXT_OUTPUT/kenlm.wrd.o40003.bin dec_exp=${ADDR[-4]} dec_splits="valid" dec_script=steps/decode_fmllr.sh

chmod +x  $DECODE_WORD

cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD > $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt

}

transcription_HMM_word2_eval()
{
#     export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
#    export KENLM_ROOT="$KENLM_ROOT/build/bin"
#    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
   
 DECODE_WORD2=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step2.sh

 output=$(get_best_path $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt) #created an output to store best hmm results  #not necessary, we can view and place our values 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

dec_splits="train valid"
update_file_variables $DECODE_WORD2 out_dir=$ST_OUTPUT_PHONES dec_exp=${ADDR[-4]} dec_splits="valid" dec_lmparam="${ADDR[-1]%.tra.txt}"
sed -i 's|\(decode\${dec_suffix}_[^/]*\)/scoring|\1.si/scoring|g' "$DECODE_WORD2"

chmod +x  $DECODE_WORD2
echo "here"
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD2

}


# ==================== MAIN EXECUTION ====================

main() {
    
    create_dirs
    chmod +x $DIR_PATH/wav2vec_config.sh #this file helps export all required variables 
    $DIR_PATH/wav2vec_config.sh
    activate_venv
    setup_env

    
    log "Starting wav2vec unsupervised pipeline for $DATASET"
 

    # Run all steps in sequence

    create_manifests 0
    
    create_rVADfast
    remove_silence
    
    create_manifests_nonsil 0.1

    prepare_audio
    prepare_text  

    # create our GANS
    train_gans

    transcription_gans_viterbi  #for these we need both train and validation since the train will be used by the HMM
    transcription_gans_kaldi

    self_training

    transcription_HMM_phone_eval
    
    transcription_HMM_word_eval
    transcription_HMM_word2_eval
    log "Pipeline completed successfully!"
}

# Run the main function
main