#!/bin/bash

run_hbdesigner \
    --pdb ../1PGA.pdb \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3 \
    --guide_seq "S,N|Q,T"
