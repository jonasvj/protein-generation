#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J tf6
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
###BSUB -u s162615@student.dtu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu-%J.err
# -- end of LSF options --

source ~/.virtualenvs/protein-generation/bin/activate

# Transformer
# tf1 - val perplexity: 6.58
#./train_model.py tf1 --embedding_size 32 --learning_rate 4e-3 --mb_size 128 --epochs 300 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 6 --n_heads 16 --hidden_size 2048 --dropout 0.1
# tf2 - val perplexity 378732
#./train_model.py tf2 --embedding_size 32 --learning_rate 4e-3 --mb_size 64 --epochs 300 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 6 --n_heads 16 --hidden_size 2048 --dropout 0.1
# tf3 - val perplexity 76101
#./train_model.py tf3 --embedding_size 64 --learning_rate 4e-3 --mb_size 128 --epochs 300 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 6 --n_heads 16 --hidden_size 2048 --dropout 0.1
# tf4 - 
#./train_model.py tf4 --embedding_size 32 --learning_rate 1e-2 --mb_size 128 --epochs 500 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 6 --n_heads 16 --hidden_size 2048 --dropout 0.1
# tf5 -
#./train_model.py tf5 --embedding_size 32 --learning_rate 1e-2 --mb_size 128 --epochs 500 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 6 --n_heads 32 --hidden_size 2048 --dropout 0.1
# tf6 -
#./train_model.py tf6 --embedding_size 32 --learning_rate 1e-2 --mb_size 128 --epochs 500 --kw_method permute --include_non_insulin --include_reverse transformer --n_layers 8 --n_heads 16 --hidden_size 2048 --dropout 0.1

# Wavenet X
# wnx1 - 7.661763310432434
#./train_model.py wnx1 --embedding_size 32 --learning_rate 3e-4 --mb_size 128 --weight_decay 1e-5 --epochs 300 --kw_method permute --include_non_insulin --include_reverse wavenet --X --n_dilations 6 --kernel_size 32 --residual_channels 64 --dilation_channels 128 --skip_channels 64 --final_channels 32
# wnx2 -
#./train_model.py wnx2 --embedding_size 32 --learning_rate 3e-4 --mb_size 128 --weight_decay 1e-4 --epochs 300 --kw_method permute --include_non_insulin --include_reverse wavenet --X --n_dilations 6 --kernel_size 32 --residual_channels 64 --dilation_channels 128 --skip_channels 64 --final_channels 32
# wnx3 -
#./train_model.py wnx3 --embedding_size 32 --learning_rate 3e-4 --mb_size 128 --weight_decay 1e-4 --epochs 300 --kw_method permute --include_non_insulin --include_reverse wavenet --X --n_dilations 8 --kernel_size 32 --residual_channels 64 --dilation_channels 128 --skip_channels 64 --final_channels 32
# wnx4 
#./train_model.py wnx4 --embedding_size 32 --learning_rate 3e-4 --mb_size 128 --weight_decay 1e-4 --epochs 300 --kw_method permute --include_non_insulin --include_reverse wavenet --X --n_dilations 8 --kernel_size 32 --residual_channels 64 --dilation_channels 128 --skip_channels 64 --final_channels 32