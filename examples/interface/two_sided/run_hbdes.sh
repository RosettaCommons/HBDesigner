#!/bin/bash

run_hbdesigner \
    --pdb ../1YRK.pdb \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3 \
    --top_k 5