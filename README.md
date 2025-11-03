# HBDesigner


### Preprocessing the training dataset

1. Obtain the [ProteinMPNN training dataset](https://github.com/dauparas/ProteinMPNN/tree/main/training)

2. Generate assemblies and idealize sidechains
```
# Repeat this for split=train, valid, and test

python preprocess_asmbs.py \
    --data_dir /data/pdb_2021aug02 \
    --out_dir /data/pdb_2021aug02/preprocessed \
    --num_workers 32 \
    --split test
```
This should take around 5-15 minutes for the full dataset, depending on how many workers (CPUs) you can provide.

3. Extract hydrogen bonding networks from idealized assemblies

This script is meant to be run on a "chunk" of about 1000 files, since it can take a few hours to process these.
To preprocess the full dataset, run this for all 243 "chunks" of 1000 assemblies. This is easiest to run as a SLURM array job if possible:
```

#SBATCH -J curation
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH -t 12:00:00
#SBATCH --array=1-243

source ~/.bashrc
mamba activate hbdesigner

python extract_hbnet.py \
    --data_dir $/data/pdb_2021aug02/preprocessed \
    --chunk $SLURM_ARRAY_TASK_ID \
    --chunk_size 1000
```
Each "chunk" can take up to a few hours to complete.

### Training the sequence design and packing models

```
# Training can take up to 2 days to complete, depending on your GPU/CPU specs

python train_hbdesigner.py --use_wandb
python train_hbpacker.py --use_wandb

```