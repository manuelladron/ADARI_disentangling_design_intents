#!/bin/sh

aws s3 cp s3://mmmlf20/preprocessed_patches.pkl .
source activate pytorch_p36
pip install transformers
pip install sentencepiece==0.1.91
conda deactivate
git clone https://github.com/manuelladron/ADARI_disentangling_design_intents.git