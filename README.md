# HBDesigner

## Overview 

HBDesigner is an algorithm that designs highly-connected hydrogen bonding networks that fit the requested design constraints onto an input protein backbone.

## Installation

HBDesigner can be installed using the provided `install_hbdesigner.sh` script. This script requires the `mamba` package manager, but it can be readily adapted to use other package managers.
```
git clone https://github.com/Kuhlman-Lab/HBDesigner.git
cd HBDesigner/
sh install_hbdesigner.sh
```

## Using HBDesigner

A detailed guide for running HBDesigner on your protein(s) of interest can be found at `examples/README.md`, along with many example runscripts for common design scenarios.

## Repeating the training and validation experiments

The preprocessed HBDesigner training dataset can be obtained here (TBD). For developers interested in replicating or extending this work, we provide details on preprocessing, training, and validation at `hbdesigner/scripts/README.md`.

## License

The HBDesigner source code and model weights are provided under an MIT license (see `LICENSE` file). However, HBDesigner uses PyRosetta for minimization and scoring. PyRosetta requires a paid license for commercial use but is free for academic use. See [the PyRosetta docs](https://www.pyrosetta.org/home/licensing-pyrosetta) for details.

## Citation

If you find HBDesigner useful for your own work, please use the following citation:
```
TBD
```