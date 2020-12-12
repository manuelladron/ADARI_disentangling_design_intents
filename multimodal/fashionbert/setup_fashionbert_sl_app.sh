#!/bin/sh

aws s3 cp s3://mmmlf20/preprocessed_patches.pkl .
source activate pytorch_p36
pip install streamlit
conda deactivate
git clone https://github.com/manuelladron/ADARI_disentangling_design_intents.git

# run file using: streamlit run fashionbert_sl_app.py 