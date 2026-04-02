#!/bin/bash

# This example shows how to run LigandMPNN on HBDesigner outputs while keeping the network residues fixed.
# Note: the key helper script (get_res_sel_multi_json.py) can only be found in the Kuhlman Lab fork of LigandMPNN (https://github.com/Kuhlman-Lab/ligandmpnn/)
source ~/.bashrc
mamba activate ligandmpnn

input=./grafted
ligandmpnn_loc=/proj/kuhl_lab/LigandMPNN

# Get JSON listing files to process
python ${ligandmpnn_loc}/get_pdb_multi_json.py --pdb_dir $input --json_file multi_pdb.json

# Get JSON listing which residues to keep fixed (all except GLY)
python ${ligandmpnn_loc}/get_sel_res_multi_json.py --pdb_dir $input --flip --json_file multi_res.json --sel_restypes "G"

# Run LigandMPNN with sidechain context
python ${ligandmpnn_loc}/run.py \
    --pdb_path_multi ./multi_pdb.json \
    --out_folder ./mpnn_outputs \
    --fixed_residues_multi multi_res.json \
    --checkpoint_ligand_mpnn ${ligandmpnn_loc}/model_params/ligandmpnn_v_32_010_25.pt \
    --temperature 0.1 \
    --number_of_batches 1 \
    --batch_size 1 \
    --checkpoint_path_sc ${ligandmpnn_loc}/model_params/ligandmpnn_sc_v_32_002_16.pt \
    --pack_side_chains 1 \
    --number_of_packs_per_design 1 \
    --ligand_mpnn_use_side_chain_context 1 \
    --pack_with_ligand_context 1 \
    --chains_to_design "A" \
    --repack_everything 0