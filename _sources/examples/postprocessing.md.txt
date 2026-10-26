# Postprocessing
We provide a helper script called `merge_networks.py` that attempts to naively combine output networks by checking for sequence overlap and clashes. This is NOT an exhaustive sweep, so it will not return ALL possible networks, but a subset from a rapid sampling procedure. For example usage, see `examples/postprocessing/run_merge.sh`.

When doing one-sided interface design, we may want to graft our output network back onto the wildtype target for downstream modeling. We provide a helper script called `graft_seq.py` to do this. For example usage, see `examples/postprocessing/run_graft.sh`

The most common use for HBDesigner outputs is as input for traditional sequence design. To do this, we use LigandMPNN to design the remaining sequence, keeping the HBDesigner network residues fixed. For an example of this, see `examples/postprocessing/run_mpnn.sh`.

These scripts pull from the resources in the other example directories.