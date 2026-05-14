#!/bin/bash

run_hbdesigner \
    --pdb ../1PGA.pdb \
    --n_workers 8 \
    --n_samples 200 \
    --n_res 3
