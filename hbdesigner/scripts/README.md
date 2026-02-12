
### Preprocessing the training dataset

1. Obtain the [ProteinMPNN training dataset](https://github.com/dauparas/ProteinMPNN/tree/main/training)

2. Generate assemblies and idealize sidechains
This should take around 5-15 minutes for the full dataset, depending on how many workers (CPUs) you can provide.
```
# Repeat this for split=train, valid, and test
python preprocess_asmbs.py \
    --data_dir /data/pdb_2021aug02 \
    --out_dir /data/pdb_2021aug02/preprocessed \
    --num_workers 32 \
    --split test
```

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
    --data_dir /data/pdb_2021aug02/preprocessed \
    --chunk $SLURM_ARRAY_TASK_ID \
    --chunk_size 1000
```
Each "chunk" can take up to a few hours to complete.

### Training the sequence design and packing models
The sequence design and packing models are trained separately. Training can take up to 2 days to complete, depending on your GPU/CPU specs. Data and log file paths may need to be updated prior to training. All training config options can be adjusted in `train_hbdesigner.py` and `train_hbpacker.py`.
```
# Sequence design model
python train_hbdesigner.py --use_wandb
# Packing model
python train_hbpacker.py --use_wandb

```
### Evaluating the models
The trained models can be evaluated using the provided eval scripts, which loop through the test set and calculate useful metrics. Validation takes a few seconds per batch. You must provide the config and ckpt files for each trained model.
```
# Design + Packing evaluation
python evaluate_hbdesigner.py \
    --pack_config $pack_cfg \
    --pack_ckpt $pack_ckpt \
    --num_workers 8 \
    --pack_method hbpacker \
    --pack_min \
    --design_config $design_cfg \
    --design_ckpt $design_ckpt

# Packing-only evaluation
python evaluate_packer.py \
    --pack_config $cfg \
    --pack_ckpt $ckpt \
    --num_workers 8 \
    --pack_method hbpacker \
    --pack_min
```

