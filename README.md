# HBDesigner

## Overview 

HBDesigner is an algorithm that designs highly-connected hydrogen bonding networks that fit the requested design constraints onto an input protein backbone.

## Running HBDesigner

### Input files

HBDesigner takes a `.pdb` file as its primary input. It can have a sequence and/or sidechains on it, but they will be removed to create a PolyGLY backbone before use.

- If you want to design intra-chain or monomer networks, you should include only the chain you wish to design.
- If you want to design inter-chain or interface networks, you should include only the chains forming the interface you wish to design.

### Basic input parameters:

These are the minimal input params that you should consider setting for any design run. They have a large impact on performance and runtime.

```
# Path to input PDB file. 
# Needs to be specified for each run.
--pdb /path/to/1ABC.pdb

# Number of residues you want in your completed network(s). 
# HBDesigner is trained to produce networks of 2-6 residues. Larger networks will be harder to design.
--n_res 3

# Path to output directory to save results. 
# If left blank, HBDesigner will use the current working directory.
--out_dir /path/to/output/

# Number of CPUs the script can use for packing. 
# Good values are 8-24, lower values will make it run slower. 
# This should match the --cpus-per-task param if running as a SLURM job.
--n_workers 16

# Number of unique samples to generate before packing/scoring. 
# Good values are 100-500, but higher values are useful if initial runs fail to find any good networks.
--n_samples 200

# Number of top (best scoring, see below section for details) designs to save. 
# Good values are 5-25, depending on your use case.
--top_k 5

# Temperature range to use for sampling procedure. 
# The first value is the initial temperature, the second value is the maximum temperature. This will be ramped up during sampling if needed.
# Good values are 0.1 to 1.0. Lower values will adhere to conditioning better but will be less diverse, so generation may take longer.
--T_range 0.3 1.0
```

### Guidance (conditioning) parameters:

These are extra (optional) params you can use to help guide the model toward making specific types of networks more often.

```
# Restypes you want the model to include in the network. 
# Use 1-letter abbreviations, with "X" representing "any polar".
# Note: the number of residues listed must match --n_res.
--guide_seq STH # this means "make 3-res networks with a SER, a THR, and a HIS."
--guide_seq HX # this means "make 2-res networks with a HIS and ANY polar."
--guide_seq HHX # this means "make 3-res networks with 2x HIS and 1x ANY polar."
--guide_seq XX # this means "make 2-res networks with 2x ANY polar."

# Residues used to calculate centroid for a "guide atom" which tell the model to build its networks nearby to this location in 3D space.
# If enabled, the "guide atom" will be represented as a virtual atom (atomid: V1, resid: ORI) in the output .pdb files.
--guide_res A45,A49,B34

# Enable a hard distance cutoff to prevent the model from straying too far from the guide atom.
# This is usually overkill for most cases, but can be useful if you're really particular about where the network should be placed.
--guide_radius 10. # this disables design for any res with Cb >10A from the guide atom.

# Enable a hard burial cutoff to prevent the model from using surface residues.
--min_burial 4.0 # this disables design for any res with <4.0 weighted sidechain neighbors (too solvent-exposed)

# Select only networks with a certain number of core residues.
--min_core_res 1 # this rejects any networks with <1 core residue present

# Add backbone noise to input model during sampling. This increases sampling diversity but lowers packing success rates.
--bb_noise 0.1 # this adds 0.1 Angstrom noise
```

### Scoring parameters:

These are useful if you have very specific scoring criteria you want to enforce in your output networks (saturation, )

```
# Set max number of buried unsatisfied heavy polar atoms allowed. Lower is more selective.
--max_BUNs 1

# Set max number of buried unsatisfied polar heavy atoms allowed. Lower is more selective.
--max_BUPHs 1

# Set min saturation level allowed. Higher is more selective.
--min_sat 0.7

# Set maximum Rosetta energy considered a successful hydrogen bond. Lower is more selective.
--max_hb_energy -0.5

```

### Symmetry and assembly parameters:

```
# Select certain chains for design. HBDesigner will run on these chains, then paste them back in to the original assembly. 
# By default, HBDesigner will run on all chains provided. If an interface is present, it will focus on interface networks.
--sel_chains A,B

# Symmetrize output networks after design. Useful for designing homooligomer interfaces. 
# For example, if designing a homodimer interface, HBDesigner will attempt to copy any non-clashing networks across the interface.
--symm_chains A,B;C,D # Tie chains A and B together and separately tie chains C and D together.

# Provide 'anchor' residue(s) around which to design a network.

```

### Advanced input parameters:

These params offer granular control over the inner workings of the model. In general, don't touch these unless you know what you're doing.

```
# Path to trained model checkpoint files.
--design_ckpt /path/to/model.pt
--pack_ckpt /path/to/model.pt

# Path to trained model configuration files.
--design_cfg /path/to/config.yaml
--pack_cfg /path/to/config.yaml

# Enable slower (but more accurate) packing by increasing the rotamer library considered by the Rosetta packer.
--slow

# Use an alternative packer other than HBPacker (rosetta or pippack). These have lower success rates so are only useful for benchmarking.
--packer rosetta 

# Change the packing crop radius (in Angstrom). Larger crops run slower but may be slightly more accurate.
--pack_crop 10.0

```

## Output Guide

The sampling script produces two kinds of output:
1) PDB files named "HBDes_rank_N.pdb", where N is the rank calculated by HBDesigner3. Lower ranks are "better".
2) A CSV file named "HBDes_stats.csv", which includes all of the scores and residue IDs of all designed networks for an overall summary. This includes networks that passed the scoring filters but didn't make the --top_k cutoff.

HBDesigner will only output networks that pass all of its scoring filters and meet its definition of 'successful'. A 'successful' design meets the following criteria:
- All network residues must be engaged in at least 1 sc-sc h-bond
- All network residues must form a single continuous network
- No buried unsatisfied heavy polar atoms (BUNs) may be present

After filtering, HBDesigner will rank the remaining networks using various score terms, with the following priority:
1) buried_unsat_Hpol (buried unsatisfied polar H atoms): the fewer the better.
2) saturation: the fraction of total h-bonding "capacity" that is being used across all network residue sc atoms: the higher the better.
3) HBond Score (HB_Score_full): the change in Rosetta energy provided by the designed network, calculated against an identical PolyG backbone: the lower (more negative) the better.

This setup has a few implications:
- It is possible for HBDesigner to return 0 networks, if given a hard enough task and/or few enough tries at it. If this happens, try increasing --n_samples.
- If HBDesigner finds more networks than --top_k allows, it will only return the --top_k "best" according to the ranking scheme. If you want more outputs, increase --top_k.

## Usage Advice

- The larger you set --n_res, the lower the success rate. This means you might want to increase --n_samples when increasing --n_res. Here is a good place to start:
  - --n_res=2, --n_samples=500
  - --n_res=3, --n_samples=1000
  - --n_res=4, --n_samples=2000
  - --n_res=5, --n_samples=10000
  - --n_res=6, --n_samples=10000

- The smaller amino acids, especially the hydroxyls, have higher success rates. This means that, if you don't care what amino acids are in your network, you can get higher success rates using --guide_seq SXX, --guide_seq TXX, etc.


## Postprocessing

We provide a helper script called `merge_networks.py` that attempts to naively combine output networks by checking for sequence overlap and clashes. This is NOT an exhaustive sweep, so it will not return ALL possible networks, but a subset from a rapid sampling procedure.

Usage:
```
python merge_networks.py --designs designs/ --output merged_designs/ --no_duplicates --max_order 5 --min_order 2
```




## Repeating the training and validation experiments

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
# Sequence design model
python train_hbdesigner.py --use_wandb
# Packing model
python train_hbpacker.py --use_wandb

```
Training can take up to 2 days to complete, depending on your GPU/CPU specs.


### Evaluating the models

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

Validation takes a few seconds per batch. You must provide the config and ckpt files for each trained model.