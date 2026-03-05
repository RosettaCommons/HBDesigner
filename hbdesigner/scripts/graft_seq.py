import argparse
import os
from hbdesigner.data import residue_constants as rc
from hbdesigner.data.protein import Protein, PDB_CHAIN_IDS


def main(target_pdb, ref_pdb, graft_chains, out_pdb):

    assert os.path.exists(target_pdb), f"Invalid target PDB {target_pdb}"
    assert os.path.exists(ref_pdb), f"Invalid ref PDB {ref_pdb}"

    p_target = Protein.from_pdb_file(target_pdb, discard_Hs=False)
    p_ref = Protein.from_pdb_file(ref_pdb, discard_Hs=False)
    assert p_target.n_res == p_ref.n_res, f"Target file {target_pdb} with {p_target.n_res} res != {p_ref.n_res} res in ref file {ref_pdb}"
    graft_chains = graft_chains.split(",")
    for chain in graft_chains:
        real_chain = PDB_CHAIN_IDS.index(chain)
        # Make sure you don't graft over existing design
        chain_mask = p_target.chain_index == real_chain
        chain_mask *= (p_target.aatype == rc.restype_order["G"])

        p_target.aatype[chain_mask] = p_ref.aatype[chain_mask]
        p_target.atom27_xyz[chain_mask] = p_ref.atom27_xyz[chain_mask]
        p_target.atom27_mask[chain_mask] = p_ref.atom27_mask[chain_mask]

    with open(out_pdb, "w") as fopen:
        fopen.write(p_target.to_pdb(unk_to_gly=True))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Helper script to graft sequence/sidechains from one structure onto another")
    parser.add_argument("--target_pdb", help="Path to the target PDB file")
    parser.add_argument("--ref_pdb", help="Path to the reference PDB file")
    parser.add_argument("--out_pdb", help="Path to save the output PDB file")
    parser.add_argument("--graft_chains", help="Chains to graft sequence/sidechains from")
    main(**vars(parser.parse_args()))