#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J wavenet
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 02:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
###BSUB -u sXXXXXX@student.dtu.dk
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

src/models/train_model.py WaveNet --embedding_size 16 --learning_rate 1e-3 --mb_size 64 --weight_decay 3.5e-3 --epochs 200 --kw_method permute wavenet --X --n_dilations 5 --n_repeats 2 --kernel_size 2 --residual_channels 256 --dilation_channels 256 --skip_channels 128 --final_channels 64