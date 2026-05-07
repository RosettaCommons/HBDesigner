# Interface Design

If provided with an interface, HBDesigner will automatically try to design a network across it. This can be used for either one-sided or two-sided interface design. The one-sided case requires one or more "anchor residue(s)" on the target strand(s).

## Symmetric Design
Once more than one chain is involved in the hydrogen bonding network, symmetry can be taken into account. HBDesigner has limited support for symmetric design across protein-protein interfaces. To do this, we offer two complementary approaches: "lazy" and "strict" symmetry:

```{warning}
Symmetric design is still experimental and has not been validated when used in combination with conditioning features. 
```

### Lazy symmetry
In a 'lazy' symmetry scenario, HBDesigner designs asymmetric networks then attempts to symmetrize them across any symmetric chains. This is best for when the network itself doesn't necessarily need to be symmetric, but the sequence symmetry needs to be preserved across the interface.

### Strict symmetry
For 'strict' symmetry, HBDesigner explicitly designs symmetric networks where all symmetric residues must contribute to the network and must interact with the symmetric copies of themselves.

```{note}
For strict symmetry to work well the designable residues must be oriented very close to the plane of symmetry.
```

For a system that is N-wise symmetric, the provided value of `n_res` must be divisible by N. For example, for a homotrimer the valid choices for `n_res` are 3, 6, etc.

---
## Examples

Examples of using HBDesigner to design hydrogen bonding networks for protein interfaces can be found in examples/interface. The following examples use the 1YRK PDB file located in the `interface` folder. You can find its entry on the RCSB [here](https://www.rcsb.org/structure/1YRK). This structure is composed of two chains and a ligand. Chain A is a protein kinase while chain B is a 13-residue peptide. The ligand is acetic acid.

- One-sided: `interface/one_sided`
    - Residue B5 (serine) is selected as an anchor residue - this residue will be part of any designed hydrogen bonding networks
    - Chain B is omitted - HBDesigner cannot choose residues from chain B in this calculation, except for those specified by `anchor_res`
- Two-sided: `interface/two_sided`
    - This example does not have any special options. HBDesigner recognizes that multiple chains are provided, and will automatically design a network between them.

The following examples use PDB files that are located in their individual folders: 
- Lazy Symmetry: `interface/symm_lazy`
    - System of interest: [10GS](https://www.rcsb.org/structure/10GS), a dimer with C2 symmetry
    - The only extra option that needs to be provided for 'lazy' symmetry is `symm_chains` which tells HBDesigner which chains should be symmetrized.
- Strict Symmetry: `interface/symm_strict`
    - System of interest [5J0K](https://www.rcsb.org/structure/5J0K), a dimer with C2 symmetry - the provided structure has been cleaned
    - The [Rosetta](https://github.com/RosettaCommons/rosetta) script, [`make_symmdef_file.pl`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/apps/public/symmetry/make_symmdef_file.pl) was used to create the provided `.symm` file.
        ```{important}
        The use of Rosetta requires a [license](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.md)
        ```
    - The addition of the `symm_chains` argument is required to tell HBDesigner which chains should be symmetrized.