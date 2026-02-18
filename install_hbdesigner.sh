
#!/bin/bash

source ~/.bashrc

# Make HBDesigner environment
mamba env create -n hbdesigner python=3.10

# Install HBDesigner and dependencies (note this order matters)
mamba activate hbdesigner

# 1. torch 2.8 for CUDA 12.8
pip install torch==2.8

# 2. torch-geometric and scatter (need to install torch first)
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 3. Other dependencies
pip install omegaconf biopython wandb "numpy==1.26.4" scipy networkx pandas pebble

# 4. PyRosetta (install via installer script)
pip install pyrosetta-installer; python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(distributed=True)'

# 5. HBDesigner pip install
pip install -e .

# 6. Optional dependencies only used for development:
# pip install ruff pytest

# 7. Optional dependencies only used for benchmarking:
# pip install hydride biotite
# git clone https://github.com/rlabduke/reduce.git
# follow REDUCE install README