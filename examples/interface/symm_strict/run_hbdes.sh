#!/bin/bash

run_hbdesigner \
    --pdb 5J0K_clean_symm.pdb \
    --n_workers 8 \
    --n_samples 500 \
    --n_res 2 \
    --symm_chains A,B \
    --symm_file 5J0K.symm
