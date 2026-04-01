#!/bin/bash

run_hbdesigner \
    --pdb 10GS.pdb \
    --n_workers 8 \
    --n_samples 500 \
    --n_res 3 \
    --symm_chains A,B