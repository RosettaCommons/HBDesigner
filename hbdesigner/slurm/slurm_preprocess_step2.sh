
source ~/.bashrc

mamba activate hbdesigner

cd /home/hdieckhaus/scripts/HBDesigner/hbdesigner/scripts

# corresponds to array number - use it to pick chunk
python extract_hbnets.py \
    --data_dir /data/pdb_2021aug02/preprocessed \
    --chunk 1 \
    --chunk_size 1000 \
    --max_net_size 6 \
    --min_net_size 2