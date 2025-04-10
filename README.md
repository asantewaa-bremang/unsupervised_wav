# unsupervised_wav
unsupervised_wav is a script compilation of which runs the the fairseq wav2vec unsupervised project https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

The scripts work well on a Python Pytorch virtual environment, with cuda 12.1 and torch less than 2.6 

## How to run the project 
the steps provided can be run in a terminal of your machine

1. ./setup.sh

2. ./run_wav2vec.sh "/path/to/audio_dataset" "/path/to/unlabelled/text_dataset"
   
*Please note:* Before you run the run_wav2vec script, you  can manually edit the variable "max_update: " in this file with the path 
 unsupervised_wav/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml
 
The current max_update is high so you can adjust it based on your dataset. 
