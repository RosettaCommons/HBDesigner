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
conda activate hbdesigner_public

# Design model
design_cfg=/proj/kuhl_lab/HBDesigner_public/HBDesigner/model_weights/design_020.yaml
design_ckpt=/proj/kuhl_lab/HBDesigner_public/HBDesigner/model_weights/design_020.pt

# Packing model
pack_cfg=/proj/kuhl_lab/HBDesigner_public/HBDesigner/model_weights/pack.yaml
pack_ckpt=/proj/kuhl_lab/HBDesigner_public/HBDesigner/model_weights/pack.pt
script=/proj/kuhl_lab/HBDesigner_public/HBDesigner/hbdesigner/inference/inference_hbdesigner.py

python $script \
    --pdb ../1PGA.pdb \
    --pack_cfg $pack_cfg \
    --pack_ckpt $pack_ckpt \
    --design_cfg $design_cfg \
    --design_ckpt $design_ckpt \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3 \
    --guide_res A3,A26
