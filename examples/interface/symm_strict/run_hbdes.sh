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
module load gcc
conda activate hbdesigner

# Make symmetry file
# ${ROSETTA}/main/source/src/apps/public/symmetry/make_symmdef_file.pl  -r 12 -m NCS -p 5J0K_clean.pdb > 5J0K.symm

run_hbdesigner \
    --pdb 5J0K_clean_symm.pdb \
    --n_workers 8 \
    --n_samples 500 \
    --n_res 2 \
    --symm_chains A,B \
    --symm_file 5J0K.symm
