# Unconditional Monomer Design

You will want to perform unconditional monomer design if you: 
- want to create networks on a monomer or interface
- do not have specific amino acids that need to be used to create these networks

Unconditional monomer design only requires:
- `--pdb` (a PDB file of your input structure)
- `--n_res` (The number of residues in the completed network(s))
- `--n_samples` (The number of samples to generate before packing and scoring)
- `--n_workers` (The number of CPUs that can be used for packing)

You can run this example by `cd`-ing into `examples/monomer/unconditional` and running the shell script located there.

For the requested 200 samples, it is typical for only ~10 to 'succeed' based on the packing and scoring procedures. Each of the generated designs should have 3 non-glycine residues, as was specified by `n_res`.