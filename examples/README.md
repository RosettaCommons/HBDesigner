# HBDesigner examples for different design scenarios

## Unconditional Monomer Design
This is the simplest possible design case - we want networks on a monomer or interface, but we don't care where or what amino acids are used.

- Unconditional monomer: `monomer/unconditional`

## Interface Design
If provided with an interface, HBDesigner will automatically try to design a network across it. This can be used for either one-sided or two-sided interface design. The one-sided case requires one or more "anchor residue(s)" on the target strand(s).

- Two-sided: `interface/two_sided`
- One-sided: `interface/one_sided`

## Sequence Conditioning
We can use sequence conditioning to specify which amino acid(s) are used, including partial or ambiguous (e.g., "Either ASN or GLN") specifications. This is specified with a comma-separated list of amino acid groups, as shown below.

- Full sequence conditioning (`S,N,T`): `monomer/sequence_conditioning_full`
- Partial sequence conditioning (`X,N,X`): `monomer/sequence_conditioning_partial`
- Ambiguous sequence conditioning (`S,N|Q,T`): `monomer/sequence_conditioning_ambiguous` 

## Virtual Guide Atom Conditioning
- TODO
