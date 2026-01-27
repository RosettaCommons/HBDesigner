# HBDesigner examples for different design scenarios

## Unconditional
This is the simplest possible design case - we want networks on a monomer or interface, but we don't care where or what amino acids are used.

- Unconditional monomer: `monomer/unconditional`
- Unconditional heterodimer: TBD

## Sequence Conditioning
We can use sequence conditioning to specify which amino acid(s) are used, including partial or ambiguous (e.g., "Either ASN or GLN") specifications. This is specified with a comma-separated list of amino acid groups, as shown below.

- Full sequence conditioning (`S,N,T`): `monomer/sequence_conditioning_full`
- Partial sequence conditioning (`X,N,X`): `monomer/sequence_conditioning_partial`
- Ambiguous sequence conditioning (`S,N|Q,T`): `monomer/sequence_conditioning_ambiguous` 

