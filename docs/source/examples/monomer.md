# Monomer Design
The types of examples in `monomer` include: 
- [Unconditional Monomer Design](#unconditional_monomer_design)
- [Sequence Conditioning](#sequence_conditioning)
- [Virtual Guide Atom Conditioning](#virtual_guide_atom_conditioning)
- [Omitting Amino Acids](#omitting_amino_acids)
- [Pixi Example](#pixi_example)

All of these examples use the provided 1PGA PDB file. You can find out more about this structure [here](https://www.rcsb.org/structure/1PGA). 

(unconditional_monomer_design)=
## Unconditional Monomer Design
You will want to perform unconditional monomer design if you: 
- want to create networks on a monomer 
- do not have specific amino acids that need to be used to create these networks

Unconditional monomer design only requires:
- `--pdb` (a PDB file of your input structure)
- `--n_res` (The number of residues in the completed network(s))
- `--n_samples` (The number of samples to generate before packing and scoring)
- `--n_workers` (The number of CPUs that can be used for packing)

You can run this example by `cd`-ing into `examples/monomer/unconditional` and running the shell script located there.

For the requested 200 samples, it is typical for only ~10 to 'succeed' based on the packing and scoring procedures. Each of the generated designs should have 3 non-glycine residues, as was specified by `n_res`.

(sequence_conditioning)=
## Sequence Conditioning
Sequence conditioning can be used to specify which amino acid(s) are used to form the hydrogen bonding network, including partial or ambiguous (e.g., "Either ASN or GLN") specifications. This is specified with a comma-separated list of amino acid groups. 

Aside from the arguments discussed above, the only additional argument required for conditional sequence design is `guide_seq`. This option specifies what amino acids can be used when designing the hydrogen bonding network.

Three sequence conditioning examples are provided: 
- `monomer/sequence_conditioning_full`:
    `S,N,T` is passed to `guide_seq` resulting in designs that use those three amino acids specifically.
- `monomer/sequence_conditioning_partial`:
    `X,T,X` is passed to `guide_seq` resulting in designs that have at least one threonine residue in each design. There are no restrictions on what the other two residues can be.
- `monomer/sequence_conditioning_ambiguous`: 
    `S,N|Q,T` is passed to `guide_seq` resulting in designs that have S, N **or** Q, and T in any order. 

(virtual_guide_atom_conditioning)=
## Virtual Guide Atom Conditioning
Guide atoms can be used to specify an approximate location for the hydrogen bonding network. Given a list of residues in the input structure, HBDesigner will place a 'guide atom' in the centroid of the C-betas of these residues. 

You can find an example of this in `monomer/guide_atom_conditioning`. The only additional argument needed, besides those discussed in the [Unconditional Monomer Design section](#unconditional_monomer_design), is `guide_res`.

(omitting_amino_acids)=
## Omitting Amino Acids
The `monomer/omit_AA` example shows how to omit amino acids from hydrogen bonding networks designed by HBDesigner. The addition of the argument `omit-AA` will prevent any of the amino acids specified from the user from being included in the designs. 

(pixi_example)=
## Pixi Example
An example has been included here see how to run HBDesigner with [Pixi](https://pixi.prefix.dev/latest/). The same command line arguments can be used but now 
```bash
pixi run --manifest-path=<path to pyproject.toml>
```
will need to be added before the rest of the command. 

This is only relevant when pixi was used to install HBDesigner. 