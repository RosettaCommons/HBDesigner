#!/bin/bash

# After one-sided design, we want to graft the binder residues onto the original structure to get the target seq back
f=1YRK_HBDes_rank_1.pdb
mkdir -p grafted
base=$(basename $f .pdb)
python ../../hbdesigner/scripts/graft_seq.py \
    --target_pdb $f \
    --ref_pdb ../interface/1YRK.pdb \
    --out_pdb grafted/${base}_grafted.pdb \
    --graft_chains B
