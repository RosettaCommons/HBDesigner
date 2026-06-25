# HBDesigner

## Overview 

HBDesigner is an algorithm that designs highly-connected hydrogen bonding networks that fit the requested design constraints onto an input protein backbone.

## Installation

HBDesigner can be installed using `mamba`, `uv`, or `Pixi`. 
```
# first, clone the repo 
git clone https://github.com/Kuhlman-Lab/HBDesigner.git
cd HBDesigner/
```
To create a virtual environment with `mamba` for use on a GPU or CPU, respectively:
```
# for running on a GPU
mamba env create -f env_gpu.yaml
pip install .

# for running on a CPU
mamba env create -f env_cpu.yaml
pip install .
```
To create a virtual environment with `uv` for use on a GPU or CPU respectively:
```
uv venv --python 3.10
source .venv/bin/activate

# for running on a GPU
uv pip install -e ".[gpu]"

# for running on a GPU with cuda 12.4
uv pip install -e ".[gpu-cu124]"

# for running on a CPU
uv pip install -e ".[cpu]"
```
To create a virtual environment with `Pixi` for use on a GPU or CPU respectively:
```
# for running on a GPU
pixi install

# if installing with pixi on a GPU with CUDA 12.4 and include -e gpu-cu124 in pixi run command
pixi install -e gpu-cu124

# to install with pixi to run on a CPU instead use and include -e cpu in pixi run command
pixi install -e cpu

```

The `Pixi` installation is much faster than `mamba`, but requires slightly more awkward syntax when running `HBDesigner`. See `examples/monomer/run_with_pixi` for an example.

## Using HBDesigner

A detailed guide for running HBDesigner on your protein(s) of interest can be found at `examples/README.md`, along with many example runscripts for common design scenarios.

## Repeating the training and validation experiments

The preprocessed HBDesigner training dataset can be obtained here (TBD). For developers interested in replicating or extending this work, we provide details on preprocessing, training, and validation at `hbdesigner/scripts/README.md`.

## License

The HBDesigner source code and model weights are provided under an MIT license (see `LICENSE` file). However, HBDesigner uses PyRosetta for minimization and scoring. PyRosetta requires a paid license for commercial use but is free for academic use. See [the PyRosetta docs](https://www.pyrosetta.org/home/licensing-pyrosetta) for details.

## Citation

If you find HBDesigner useful for your own work, please use the following citation:
```
@article {Dieckhaus2026.06.08.730848,
	author = {Dieckhaus, Henry and Harvey, Brock T. and Mulikova, Tomiris and Horenstein, Jessica T. and Nicely, Nathan I. and Randolph, Nicholas Z. and Kuhlman, Brian},
	title = {Deep learning based design of buried hydrogen bond networks with HBDesigner},
	elocation-id = {2026.06.08.730848},
	year = {2026},
	doi = {10.64898/2026.06.08.730848},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/06/11/2026.06.08.730848},
	eprint = {https://www.biorxiv.org/content/early/2026/06/11/2026.06.08.730848.full.pdf},
	journal = {bioRxiv}
}
```
