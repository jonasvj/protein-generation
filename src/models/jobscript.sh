#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J wn_1
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s162615@student.dtu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu-%J.err
# -- end of LSF options --

source ~/.virtualenvs/protein-generation/bin/activate
# Wavenet X
#./train_model.py wnx_1_e1000_kwP_emb16_dil6_res8_f4 --epochs 1000 --kw_method permute --embedding_size 16 wavenet --X --n_dilation 6 --res_channels 8 --f_channels 4
# Wavenet
#./train_model.py wn_1_e1000_kwP_emb16_dil6_res8_f4 --epochs 1000 --kw_method permute --embedding_size 16 wavenet --n_dilation 6 --res_channels 8 --f_channels 4
# Transformer
#./train_model.py tf_1_e1000_kwP_emb16_nl4_nh8_hs512 --epochs 1000 --kw_method permute --embedding_size 16 transformer --n_layers 4 --n_heads 8 --hidden_size 512
# Gru
#./train_model.py gru_1_e1000_kwP_emb16_nl4_hs512_do05 --epochs 1000 --kw_method permute --embedding_size 16 gru --n_layers 4 --hidden_size 512 --dropout 0.5
# Lstm
./train_model.py lstm_1_e1000_kwP_emb16_nl4_hs512_do05 --epochs 1000 --kw_method permute --embedding_size 16 lstm --n_layers 4 --hidden_size 512 --dropout 0.5