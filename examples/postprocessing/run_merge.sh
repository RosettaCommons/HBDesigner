#!/bin/bash

# Generate single networks with HBDesigner
run_hbdesigner \
    --pdb ../interface/1YRK.pdb \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3 \
    --top_k 10 \
    --out_dir ./single_network

# Merge single networks into multi-network designs
python ../../hbdesigner/scripts/merge_networks.py \
    --designs ./single_network \
    --output ./multi_network \
    --max_order 2 \
    --seed 123

