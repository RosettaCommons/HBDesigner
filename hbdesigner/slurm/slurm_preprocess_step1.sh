
source ~/.bashrc

mamba activate hbdesigner

cd /home/hdieckhaus/scripts/HBDesigner/hbdesigner/scripts

python preprocess_asmbs.py \
    --data_dir /data/pdb_2021aug02 \
    --out_dir /data/pdb_2021aug02/tmp_preprocessed \
    --num_workers 8 \
    --split test
