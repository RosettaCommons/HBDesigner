#!/bin/bash

#SBATCH -J run_hbdesigner
#SBATCH -p kuhlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH -t 00:10:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate hbdesigner

run_hbdesigner \
    --pdb 10GS.pdb \
    --n_workers 8 \
    --n_samples 500 \
    --n_res 3 \
    --top_k 5 \
    --symm_chains A,B