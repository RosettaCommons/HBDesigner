#!/bin/bash

# To omit ARG and LYS from any designs...
run_hbdesigner \
    --pdb ../1PGA.pdb \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3 \
    --omit_AA "R,K"
