import argparse
import os 
import glob
import random
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

from hbdesigner.data.protein import Protein
import hbdesigner.data.residue_constants as rc


def check_clashes(p_anchor, p_next, threshold=3.0):
    anchor_xyz = np.reshape(p_anchor.atom27_xyz[:, 4:14], (-1, 3)) # [N, 3]
    next_xyz = np.reshape(p_next.atom27_xyz[:, 4:14], (-1, 3)) # [N, 3]
    anchor_xyz_mask = np.reshape(p_anchor.atom27_mask[:, 4:14], (-1, 1))
    next_xyz_mask = np.reshape(p_next.atom27_mask[:, 4:14], (-1, 1))

    pair_dists = cdist(anchor_xyz, next_xyz) # [N, N]
    pair_masks = anchor_xyz_mask * np.transpose(next_xyz_mask) # [N, N]
    clashes = (pair_dists < threshold) * pair_masks
    n_clashes = np.sum(clashes)

    return n_clashes > 0


def check_overlap(anchor_res, next_res):
    return np.sum(anchor_res[:, None] == next_res[None, :]) > 0


def main(designs: str, output: str, max_order: int = 2, min_order: int = 2, no_duplicates: bool = False, threshold: float = 3.0):

    assert os.path.isdir(designs), f"Design directory {designs} does not exist."

    os.makedirs(output, exist_ok=True)

    # Collect all network files
    network_files = glob.glob(os.path.join(designs, "*.pdb"))

    # Shuffle into random order
    random.shuffle(network_files)
    idx_used = []

    # Iterate over files and only consider networks after them in the list
    for idx, anchor_network in tqdm(enumerate(network_files)):
        # Load anchor network
        p_anchor = Protein.from_pdb_file(anchor_network, discard_Hs=False)
        if no_duplicates and idx in idx_used:
            continue

        # Iterate over all networks after the anchor network
        graft_idx = idx + 1
        while True:
            # Break on list end
            if graft_idx >= len(network_files):
                break

            next_network = network_files[graft_idx]
            p_next = Protein.from_pdb_file(next_network, discard_Hs=False)

            anchor_res = np.where(p_anchor.aatype != rc.restype_order["G"])[0]
            next_res = np.where(p_next.aatype != rc.restype_order["G"])[0]

            # Break if clash found
            has_clash = check_clashes(p_anchor.mask(anchor_res), p_next.mask(next_res), threshold=threshold)
            has_clash = has_clash or check_overlap(anchor_res, next_res)
            if not has_clash:
                # If no clash, graft together
                p_anchor.aatype[next_res] = p_next.aatype[next_res]
                p_anchor.atom27_xyz[next_res] = p_next.atom27_xyz[next_res]
                p_anchor.atom27_mask[next_res] = p_next.atom27_mask[next_res]
            else:
                break
            graft_idx += 1

            # Break on max graft order
            if (graft_idx - idx) >= max_order:
                break
        # Check if net passes min graft order
        if (graft_idx - idx) < min_order:
            continue
        # Save the merged network
        else:
            order = graft_idx - idx
            p_hash = round(np.abs(p_anchor.__hash__()) % 1e6)
            print("Saving merged network: ", f"HBDes_merged_{p_hash}_order{order}.pdb")
            output_file = os.path.join(output, f"HBDes_merged_{p_hash}_order{order}.pdb")
            with open(output_file, "w") as f:
                f.write(p_anchor.to_pdb())
        
            if no_duplicates:
                idx_used.extend(range(idx, graft_idx + 1))
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge multiple network files into one.")
    parser.add_argument("--designs", type=str, required=True, help="Design directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max_order", type=int, default=2, help="Maximum grafting order (i.e., number of concurrent networks attempted).")
    parser.add_argument("--min_order", type=int, default=2, help="Minimum grafting order (i.e., number of concurrent networks attempted).")
    parser.add_argument("--no_duplicates", action="store_true", help="If set, will not allow the same network to be grafted multiple times.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Clash threshold distance in Angstroms.")

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    main(args.designs, args.output, args.max_order, args.min_order, args.no_duplicates, args.threshold)