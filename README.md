# unsupervised_wav
unsupervised_wav is a script compilation which runs the the fairseq wav2vec unsupervised project https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

The scripts work well on a Python Pytorch virtual environment, with cuda 12.1 and torch less than 2.6 

## How to run the project 
the steps provided can be run in a terminal of your machine

1. ### chmod +x setup.sh
2. ### ./setup.sh
   please note the following steps while running ./setup.sh, before setup run's it makes this check LD_LIBRARY_PATH starts with '/usr/local/cuda/lib64'
   A fail in the prerequisite indicates a system misconfiguration likely caused by manual CUDA/cuDNN setup or the /etc/profile.d/nvidia-env.sh script."
         This WILL cause library conflicts (like the libcudnn error) later when interacting with cuda in the code.
   This requires a manual fix: 
        
          >>> MANUAL FIX REQUIRED <<<
        1. Edit the system file with: sudoedit /etc/profile.d/nvidia-env.sh
          (or use: sudo nano /etc/profile.d/nvidia-env.sh)
          (or use: sudo vi /etc/profile.d/nvidia-env.sh)
       2. Find the line starting with 'export LD_LIBRARY_PATH=...'
       3. Carefully REMOVE the '/usr/local/cuda/lib64:' part from the beginning of that line.
         Example - Change:
          export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nccl2/lib:...
         To:
               export LD_LIBRARY_PATH=/usr/local/nccl2/lib:...
       4. Save the file and exit the editor."
       5. IMPORTANT: REBOOT your system or fully LOG OUT and LOG BACK IN."
       6. Rerun this setup script AFTER rebooting/re-logging in."

4. ### chmod +x run_wav2vec.sh
5.  ### ./run_wav2vec.sh "/path/to/audio_dataset" "/path/to/unlabelled/text_dataset"
         
         >>> Please note:<<<
         1. Before you run the run_wav2vec script, you  can manually edit the variable "max_update: " in this file with the path 
          unsupervised_wav/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml

         2. Also, the all audio files must be converted to .wav for a successful script run

    >>> steps before gans training<<<

 Also, if your dataset is less than 2000 audios you need to change the parameters in the file "fairseq/examples/wav2vec/unsupervised/kaldi_self_train/st/train.sh" to ensure a smooth run 
**parameters to in file:** *local/train_subset_lgbeam.sh \
  --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
  --mono_size 2000 --tri1_size 5000 --tri2b_size -1 --tri3b_size -1 \
  --stage 1 --max_stage 3 $data_dir $data_dir/lang $data_dir/lang_test*
  
**changes:** *local/train_subset_lgbeam.sh \
  --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
  --mono_size -1 --tri1_size -1 --tri2b_size -1 --tri3b_size -1 \
  --stage 1 --max_stage 3 $data_dir $data_dir/lang $data_dir/lang_test*

  **We set the mono_size and the tri1_size to -1 to accommodate all our audio data present**

 
The current max_update is high so you can adjust it based on your dataset. 
