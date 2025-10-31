#!/bin/bash

#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 2-00:00:00
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

source ~/.bashrc
mamba activate hbdesigner
cd $gfn
echo "GPU: $CUDA_VISIBLE_DEVICES"

train_script=/home/hdieckhaus/scripts/HBDesigner/hbdesigner/scripts/train_hbdesigner.py
cd $(dirname $train_script)

python $train_script --config hbdesigner_wout

# python train_hbdesigner.py --use_wandb --config hbdes3