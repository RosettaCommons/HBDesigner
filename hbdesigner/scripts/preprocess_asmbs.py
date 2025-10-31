import argparse
import os
import re
import time
from concurrent.futures import TimeoutError
from functools import partial
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from pebble import ProcessExpired, ProcessPool

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import build_sc_from_chi, calc_sc_dihedrals
from hbdesigner.data.protein import Protein


def load_metadata(location: str, split: str) -> pd.DataFrame:
    """
    Load metadata for a specific split (train/valid/test) from the ProteinMPNN dataset.

    Args:
        location (str): Directory containing the ProteinMPNN metadata files.
        split (str): Which split to load ("train", "valid", or "test").
    Returns:
        pd.DataFrame: Filtered metadata for the specified split.
    """
    # Load metadata and clusters
    meta_file = os.path.join(location, "list.csv")
    meta_data = pd.read_csv(meta_file, index_col=None)

    # Filter by resolution
    RES_CUTOFF = 3.5
    meta_data = meta_data.loc[meta_data["RESOLUTION"] <= RES_CUTOFF]

    # Load valid and test clusters
    with open(os.path.join(location, "valid_clusters.txt"), "r") as fopen:
        valid_clusters = fopen.readlines()
        valid_clusters = set([int(v.removesuffix("\n")) for v in valid_clusters])

    with open(os.path.join(location, "test_clusters.txt"), "r") as fopen:
        test_clusters = fopen.readlines()
        test_clusters = set([int(v.removesuffix("\n")) for v in test_clusters])

    # Filter by split
    orig_size = meta_data.shape[0]
    if split == "test":
        meta_data = meta_data.loc[meta_data["CLUSTER"].isin(test_clusters)]
    elif split == "valid":
        meta_data = meta_data.loc[meta_data["CLUSTER"].isin(valid_clusters)]
    elif split == "train":
        meta_data = meta_data.loc[
            ~meta_data["CLUSTER"].isin(valid_clusters)
            & ~meta_data["CLUSTER"].isin(test_clusters)
        ]
    new_size = meta_data.shape[0]
    pct = round(100 * (new_size / orig_size), 1)

    print(
        f"Metadata for {meta_data.shape[0]} chains ({pct}%) loaded for split {split}..."
    )
    return meta_data


def dict_to_protein(data: Dict) -> Protein:
    """
    Convert from ProteinMPNN dict format to HBDesigner compatible Protein format.

    Args:
        data (Dict): A dictionary containing ProteinMPNN data.
    Returns:
        Protein: A Protein object compatible with HBDesigner.
    """
    atom14_mask = torch.prod(torch.isfinite(data["xyz"]), dim=-1)  # [L, 14]
    data["seq"] = re.sub(r"([^ACDEFGHIKLMNPQRSTVWXY])", "X", data["seq"])
    r_order = {restype: i for i, restype in enumerate(rc.restypes_with_x)}
    aatype = torch.tensor([r_order[aa] for aa in data["seq"]], dtype=torch.int32)
    chain_index = data["idx"]
    chains = torch.unique(chain_index)
    residue_index = []
    for c_idx in chains:
        positions = torch.where(chain_index == c_idx)[0]
        res_idx = torch.arange(positions.numel()) + 1
        residue_index.append(res_idx)
    residue_index = torch.concatenate(residue_index)  # [L]
    b_factors = torch.nan_to_num(data["bfac"])
    atom14_xyz = torch.nan_to_num(data["xyz"])
    return Protein(
        atom27_xyz=atom14_xyz.numpy(),
        atom27_mask=atom14_mask.numpy(),
        aatype=aatype.numpy(),
        residue_index=residue_index.numpy(),
        chain_index=chain_index.numpy(),
        b_factors=b_factors.numpy(),
    )


def preprocess_protein(
    data_entry: Tuple[str, Protein],
    out_dir: str,
) -> bool:
    """
    Preprocess a Protein object by idealizing sidechains and checking for missing sidechains.

    Args:
        data_entry (Tuple[str, str]): A tuple containing the entry id and the Protein.
        out_dir (str): The output directory for processed files

    Returns:
        bool: Whether the preprocessing succeeded or not
    """

    entry_id, protein = data_entry

    # Cap total size for memory reasons
    if protein.n_res > 10_000:
        return False

    # Make sure at least 25% of the backbone exists
    bb_present = np.prod(protein.atom27_mask[:, :4], -1) == 1.0
    if bb_present.sum() / bb_present.shape[0] < 0.25:
        print(
            f"Invalid PDB Assembly {entry_id}: Less than 25% of the residues have all backbone atoms."
        )
        return False

    # Make sure there are at least 10 residues with full backbone
    if bb_present.sum() < 10:
        print(
            f"Invalid PDB Assembly {entry_id}: Less than 10 residues with all backbone atoms."
        )
        return False

    # Rebuild idealized sidechains after decomposing into chi angles
    sc_dihedral, sc_dihedral_mask = calc_sc_dihedrals(
        protein.atom27_xyz[:, :14], protein.aatype
    )
    atom14_xyz, atom14_mask = build_sc_from_chi(
        protein.atom27_xyz[:, :4], protein.aatype, sc_dihedral, sc_dihedral_mask
    )
    protein.atom27_xyz[:, :14] = atom14_xyz
    protein.atom27_mask[:, :14] *= atom14_mask

    # Save protein as a .npz instead of .pdb
    subdir = os.path.join(out_dir, entry_id[1:3])
    out_file = os.path.join(subdir, f"{entry_id}.npz")
    np.savez_compressed(
        out_file,
        aatype=protein.aatype,
        atom27_xyz=protein.atom27_xyz,
        atom27_mask=protein.atom27_mask,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        b_factors=protein.b_factors,
    )
    return True


def _preprocess_pdb(entry: str, data_dir: str, out_dir: str) -> bool:
    """
    Preprocess a single PDB entry by building assemblies and idealizing sidechains.

    Args:
        entry (str): The PDB entry ID.
        data_dir (str): Directory containing the ProteinMPNN dataset and metadata.
        out_dir (str): Output directory that will contain the preprocessed dataset files.

    Returns:
        bool: Whether the preprocessing succeeded or not.

    """
    # Get assembly metadata
    pdb_loc = os.path.join(data_dir, "pdb/")
    PREFIX = os.path.join(pdb_loc, entry[1:3], entry)
    assert os.path.isfile(PREFIX + ".pt")

    # Load all available chains
    mdata = torch.load(PREFIX + ".pt", weights_only=True)
    asmb_ids = mdata["asmb_ids"]
    asmb_chains = mdata["asmb_chains"]

    subdir = os.path.join(out_dir, entry[1:3])
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    flag_total = False

    mdata["asmb_chain_key"] = {}
    # Loop over all assemblies
    for asmb_i in sorted(set(asmb_ids)):
        # Get all transforms for a certain asmb (can be more than one)
        idx = np.where(np.array(asmb_ids) == asmb_i)[0]

        # load relevant chains
        chains = {
            c: torch.load("%s_%s.pt" % (PREFIX, c))
            for i in idx
            for c in asmb_chains[i]
            if c in mdata["chains"]
        }

        # Generate asmb
        asmb = {}
        # This is iterating over transforms - one chain can have multiple transforms
        for k in idx:
            xform = mdata["asmb_xform%d" % k]
            u = xform[:, :3, :3]
            r = xform[:, :3, 3]

            # select chains which k-th xform should be applied to
            s1 = set(mdata["chains"])  # all available chains
            s2 = set(asmb_chains[k].split(","))  # current assembly chains
            chains_k = sorted(s1 & s2)

            for c in chains_k:
                try:
                    xyz = chains[c]["xyz"]
                    xyz_ru = torch.einsum("bij,raj->brai", u, xyz) + r[:, None, None, :]
                    asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
                except KeyError:
                    continue

        # stack all chains in the assembly together
        seq, xyz, idx, masked = "", [], [], []
        seq_list, bfac_list = [], []
        chain_key = {}
        for counter, (k, v) in enumerate(asmb.items()):
            if k[0] not in chain_key.keys():
                chain_key[k[0]] = [counter]
            else:
                chain_key[k[0]].append(counter)
            seq += chains[k[0]]["seq"]
            seq_list.append(chains[k[0]]["seq"])
            bfac_list.append(chains[k[0]]["bfac"])
            xyz.append(v)
            idx.append(torch.full((v.shape[0],), counter))

        data = {
            "seq": seq,  # combined seq
            "xyz": torch.cat(xyz, dim=0),  # combined xyz
            "idx": torch.cat(idx, dim=0),  # chain IDs
            "bfac": torch.cat(bfac_list, dim=0),  # bfactors
            "masked": torch.Tensor(masked).int(),  # empty
            "label": entry,
        }

        # skip any that exceed max PDB chain or residue number
        if torch.unique(data["idx"]).numel() > 62:
            continue
        if len(seq) > 10_000:
            continue

        # Convert data dict to Protein format
        protein = dict_to_protein(data)

        entry_n = (f"{entry}_{asmb_i}", protein)
        code = preprocess_protein(
            data_entry=entry_n,
            out_dir=out_dir,
        )
        flag_total = flag_total or code

        # Save chain conversion for later use
        mdata["asmb_chain_key"][asmb_i] = chain_key

    if flag_total:
        dst = os.path.join(subdir, entry + ".pt")
        torch.save(mdata, dst)

    return flag_total


def main(
    data_dir: str, out_dir: str, num_workers: int = 1, split: str = "test"
) -> None:
    """
    Preprocess the ProteinMPNN dataset by building assemblies, idealizing sidechains.

    Args:
        data_dir (str): Directory containing the ProteinMPNN dataset and metadata.
        out_dir (str): Output directory that will contain the preprocessed dataset files.
        num_workers (int): Number of workers for multiprocessing. Defaults to 1
        split (str): Which split to use for metadata selection.

    Returns:
        None
    """

    df = load_metadata(data_dir, split)
    pdbs = set(df["CHAINID"].str[:4].tolist())
    print(f"Running preprocessing on {len(list(pdbs))} pdbs...")

    # Create the out directory
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_processed = 0

    # Create the worker func.
    worker_func = partial(
        _preprocess_pdb,
        data_dir=data_dir,
        out_dir=out_dir,
    )

    i = 0
    pdbs = list(pdbs)

    # ProcessPool uses a worker timeout argument to avoid hangs
    with ProcessPool(max_workers=num_workers) as p:
        future = p.map(worker_func, pdbs, chunksize=1, timeout=300)
        iterator = future.result()

        while True:
            try:
                _ = next(iterator)
                print("function successful", flush=True)
                n_processed += 1
            except StopIteration:
                break
            except TimeoutError as error:
                print(
                    "function took longer than %d seconds for pdb %s"
                    % (error.args[1], pdbs[i]),
                    flush=True,
                )
            except ProcessExpired as error:
                print(
                    "%s. Exit code: %d for pdb %s" % (error, error.exitcode, pdbs[i]),
                    flush=True,
                )
            except Exception as error:
                print("function raised %s for pdb %s" % (error, pdbs[i]), flush=True)
            finally:
                i += 1

    print(
        f"Succeeded in processing {n_processed}/{len(pdbs)} PDBs in {time.time() - t0:.3f} sec.",
        flush=True,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ProteinMPNN dataset for HBDesigner (step 1)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the ProteinMPNN dataset and metadata.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory that will contain the preprocessed dataset files.",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for multiprocessing. Defaults to 1",
        required=False,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to use for metadata selection.",
        required=False,
    )

    args = parser.parse_args()
    main(**vars(args))
