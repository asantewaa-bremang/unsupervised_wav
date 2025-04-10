# unsupervised_wav
unsupervised_wav is a script compilation which runs the the fairseq wav2vec unsupervised project https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

The scripts work well on a Python Pytorch virtual environment, with cuda 12.1 and torch less than 2.6 

## How to run the project 
the steps provided can be run in a terminal of your machine

1. chmod +x setup.sh
2. ./setup.sh

3. chmod +x run_wav2vec.sh
4.  ./run_wav2vec.sh "/path/to/audio_dataset" "/path/to/unlabelled/text_dataset"
   
*Please note:* Before you run the run_wav2vec script, you  can manually edit the variable "max_update: " in this file with the path 
 unsupervised_wav/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml

 Also, the all audio files must be converted to .wav for a successful script run

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
