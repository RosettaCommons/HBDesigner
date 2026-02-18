# Quick start: HBDesigner examples for different design scenarios

## Unconditional Monomer Design
This is the simplest possible design case - we want networks on a monomer or interface, but we don't care where or what amino acids are used.

- Unconditional monomer: `monomer/unconditional`

## Interface Design
If provided with an interface, HBDesigner will automatically try to design a network across it. This can be used for either one-sided or two-sided interface design. The one-sided case requires one or more "anchor residue(s)" on the target strand(s).

- Two-sided: `interface/two_sided`
- One-sided: `interface/one_sided`

## Symmetric Design:
HBDesigner has limited support for symmetric design across protein-protein interfaces. To do this, we offer two complementary approaches: "lazy" and "strict" symmetry. "Lazy" symmetry designs asymmetric networks, then tries to symmetrize them across any symmetric chains. This is useful for cases where you don't care if your network itself is symmetric, just that you preserve sequence symmetry across the interface. Lazy symmetry can be run on Cn symmetric assemblies for networks of any size. "Strict" symmetry explicitly designs symmetric networks where all symmetric residues must contribute to the network and interact with symmetric copies of themselves. This is difficult to satisfy unless the designable residues are oriented very close to the plane of symmetry. Strict symmetry for N-wise symmetry must be given a `--n_res` that is divisible by N (e.g., if designing a homotrimer, `--n_res` can be `3,6,etc.`). Symmetric design is still experimental and has not been validated when used in combination with conditioning features.

- Lazy: `interface/symm_lazy`
- Strict: `interface/symm_strict`

## Sequence Conditioning
We can use sequence conditioning to specify which amino acid(s) are used, including partial or ambiguous (e.g., "Either ASN or GLN") specifications. This is specified with a comma-separated list of amino acid groups, as shown below.

- Full sequence conditioning (`S,N,T`): `monomer/sequence_conditioning_full`
- Partial sequence conditioning (`X,N,X`): `monomer/sequence_conditioning_partial`
- Ambiguous sequence conditioning (`S,N|Q,T`): `monomer/sequence_conditioning_ambiguous` 

## Virtual Guide Atom Conditioning
We can use virtual guide atom conditioning to specify an approximate location for our network. For convenience, users can provide a list of residues, and HBDesigner will use the centroid of their C-betas as the guide atom.

- Virtual guide atom conditioning: `monomer/guide_atom_conditioning`

# More details: HBDesigner inference settings

## Input and Output files:
HBDesigner takes a `.pdb` file as its primary input. It can have a sequence and/or sidechains on it, but they will be removed to create a PolyGLY backbone before use.
- If you want to design intra-chain or monomer networks, you should include only the chain you wish to design.
- If you want to design inter-chain or interface networks, you should include only the chains forming the interface you wish to design.

HBDesigner produces two kinds of output:
-  PDB files named "HBDes_rank_N.pdb", where N is the rank calculated by HBDesigner. Lower ranks are "better".
- A CSV file named "HBDes_stats.csv", which includes all of the scores and residue IDs of all designed networks. This includes networks that passed the scoring filters but didn't make the `--top_k` cutoff.

HBDesigner will only output networks that pass all of its scoring filters and meet its definition of 'successful'. A 'successful' design meets the following criteria:
- All network residues must be engaged in at least 1 sidechain-sidechain H-bond
- All network residues must form a single contiguous network
- The network passes the minimum thresholds for saturation, BUHs, BUPHs, etc. (see `Scoring parameters` below)

After filtering, HBDesigner will rank the remaining networks using various score terms, with the following priority:
1) buried_unsat_Hpol (buried unsatisfied polar H atoms): the fewer the better.
2) saturation: the fraction of total h-bonding "capacity" that is being used across all network residue sc atoms: the higher the better.
3) HBond Score (HB_Score_full): the change in Rosetta energy provided by the designed network, calculated against an identical PolyG backbone: the lower (more negative) the better.

This setup has a few implications:
- It is possible for HBDesigner to return 0 networks, if given a hard enough task and/or few enough tries at it. If this happens, try increasing `--n_samples`.
- If HBDesigner finds more networks than `--top_k` allows, it will only return the `--top_k` "best" according to the ranking scheme. If you want more outputs, increase `--top_k`.

## Basic input parameters:
These are the minimal input params that you should consider setting for any design run. They have a large impact on performance and runtime.
```
# Path to input PDB file. Needs to be specified for each run.
--pdb /path/to/1ABC.pdb

# Number of residues you want in your completed network(s). HBDesigner is trained to produce networks of 2-6 residues.
--n_res 3

# Path to output directory to save results. If left blank, HBDesigner will use the current working directory.
--out_dir /path/to/output/

# Number of CPUs the script can use for packing. Good values are 8-24, higher values will run faster, up to a point of saturation. 
# This should match the --cpus-per-task param if running as a SLURM job.
--n_workers 16

# Number of unique samples to generate before packing/scoring. Good values are 100-500, but higher values are useful if initial runs fail to find networks.
--n_samples 200

# Number of top (best scoring, see below section for details) designs to save. Good values are 5-25, depending on your use case.
--top_k 5
```

## Usage Advice
At larger `--n_res`, packing is harder, so you will get fewer good designs per `--n_samples`. This means you might want to increase `--n_samples` when increasing `--n_res`. Here is a good place to start:
```
--n_res=2, --n_samples=100
--n_res=3, --n_samples=200
--n_res=4, --n_samples=500
--n_res=5, --n_samples=500
--n_res=6, --n_samples=1000
```
Smaller amino acids, especially SER and THR, have notably higher success rates. This means that, if you don't care what amino acids are in your network, you can get higher success rates using --guide_seq SXX, --guide_seq TXX, etc.

## Postprocessing
We provide a helper script called `merge_networks.py` that attempts to naively combine output networks by checking for sequence overlap and clashes. This is NOT an exhaustive sweep, so it will not return ALL possible networks, but a subset from a rapid sampling procedure.
```
python merge_networks.py --designs designs/ --output merged_designs/ --no_duplicates --max_order 5 --min_order 2
```

### Conditioning parameters:
These are extra (optional) params you can use to help guide the model toward making specific types of networks more often.
```
# Sequence conditioning tells the model which amino acid types to include in the network.
# Use 1-letter abbreviations, with "X" representing "any polar".
# Note: the number of residues listed must match --n_res.
--guide_seq S,T,H # this means "make 3-res networks with a SER, a THR, and a HIS."
--guide_seq H,X # this means "make 2-res networks with a HIS and ANY polar."
--guide_seq H,H,X # this means "make 3-res networks with 2x HIS and 1x ANY polar."
--guide_seq X,X # this means "make 2-res networks with 2x ANY polar."
--guide_seq S|T,H # this means "make 2-res networks with (either a SER or a THR) and a HIS.

# Virtual guide atom conditioning tells the model where to build its networks.
# To use this, provide a list of residues which will be used to triangulate the virtual guide atom position.
# If enabled, the "guide atom" will be represented as a virtual atom (atomid: V1, resid: ORI) in the output .pdb files.
--guide_res A45,A49,B34

# Enable a hard distance cutoff to prevent the model from straying too far from the guide atom.
# This is usually overkill for most cases, but can be useful if you're really particular about where the network should be placed.
--guide_radius 10. # this disables design for any res with Cb >10A from the guide atom.

# Provide 'anchor' residue(s) around which to design a network.
--anchor_res B5 # Design networks using the residue B5 as an anchor, so all networks must contain this residue.
```

### Other sampling parameters:
These are useful for enforcing sufficient burial and/or network sequence diversity.
```
# Enable a hard burial cutoff to prevent the model from using surface residues.
--min_burial 4.0 # this disables design for any res with <4.0 weighted sidechain neighbors (too solvent-exposed)

# Select only networks with a certain number of core residues.
--min_core_res 1 # this rejects any networks with <1 core residue present

# Temperature range to use for sampling procedure. Good values are 0.1 to 1.0. 
# Lower values will adhere to conditioning better but will be less diverse, so generation may take longer.
--T_range 0.3 1.0

# Set a fixed seed for random number generation. Off by default.
# Rosetta energy values may still vary slightly, but model sampling should be consistent.
--seed 42
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
--symm_chains A,B # Tie chains A and B together.

# Symmetry definition file (from Rosetta), required for 'strict' symmetry (see example above).
--symm_file 5JOK.symm

# Turn off design for one or more chains.
--omit_chains B,C # Turn off chain B and C design options.

# Omit certain amino acids
--omit_AA K,R # don't use LYS or ARG in any networks.
```
