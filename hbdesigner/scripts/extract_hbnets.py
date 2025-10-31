import argparse
import glob
import os
import time
from copy import deepcopy
from functools import partial

import networkx as nx
import numpy as np
import pyrosetta
from pebble import ProcessExpired, ProcessPool
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

from hbdesigner.data.protein import Protein


pyrosetta.init(
    "-pack_missing_sidechains False"  # leave xtal as-is
    + " -optH_MCA True"  # better optH algorithm
    + " -flip_HNQ True"  # allow tautomer flips
    + " -no_optH False" # allow optH for better bonding
    + " -hbond_disable_bbsc_exclusion_rule True"  # remove hotfix
    + " -mute all"
    + " -ignore_unrecognized_res 1"
)


def detect_asmb(fname: str, max_net_size: int, min_net_size: int) -> bool:
    """
    Detect H-Bond networks in a given assembly.

    Args:
        fname (str): Path to assembly .npz file.
    
    Returns:
        bool: True if detection was successful, False otherwise.
    """
    out_graph = fname.removesuffix(".npz") + ".gml"
    # Overwrite or delete, regardless
    if os.path.exists(out_graph):
        os.remove(out_graph)

    # Load ASMB from disk
    p = np.load(fname)
    p = Protein(
        atom27_xyz=p["atom27_xyz"],
        atom27_mask=p["atom27_mask"],
        aatype=p["aatype"],
        residue_index=p["residue_index"],
        chain_index=p["chain_index"],
        b_factors=p["b_factors"],
    )

    # Load into Pose
    pose = Pose()
    pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))
    print("Chains:", pose.num_chains(), "----\tResidues:", pose.total_residue())

    # Slightly more permissive core (default cutoffs are 5.2 and 2.0)
    core_sel = LayerSelector()
    core_sel.set_cutoffs(core=4.0, surf=2.0)
    core_sel.set_layers(pick_core=True, pick_boundary=False, pick_surface=False)
    core_bool_mask = np.array(core_sel.apply(pose))

    # Collect bonds
    hbondset = pose.get_hbonds(exclude_scb=True, exclude_bsc=True, exclude_bb=True)
    bonds = hbondset.hbonds()
    if len(bonds) < 1:
        return False
    edgelist = []
    for bond in bonds:
        acc, don = bond.acc_res() - 1, bond.don_res() - 1
        edgelist.append((acc, don))

    # Filter graphs by size and burial
    g = nx.from_edgelist(edgelist, create_using=nx.DiGraph())
    cc = nx.weakly_connected_components(g)
    g_new = deepcopy(g)
    node_attrs = {}
    i = 0
    for c in cc:
        sg = g.subgraph(c)
        sg_size = len(sg)
        nodes = list(sg.nodes)
        core_nodes = core_bool_mask[np.array(nodes)]
        # Filter by size
        if (sg_size < min_net_size) or (sg_size > max_net_size):
            g_new.remove_nodes_from(sg.nodes)
        # Filter by burial - must be >50% core to stay
        elif np.sum(core_nodes) <= len(nodes) // 2:
            g_new.remove_nodes_from(sg.nodes)
        else:
            for n in nodes:
                node_attrs[n] = {
                    "PDB_Chain": pose.pdb_info().chain(n + 1),
                    "PDB_Number": pose.pdb_info().number(n + 1),
                    "is_Core": bool(core_bool_mask[n]),
                }
            i += 1

    if g_new.number_of_edges() < 1:
        return False
    # Record PDB chain/numbering for alignment
    nx.set_node_attributes(g_new, node_attrs)

    # Save output graph
    nx.write_gml(g_new, path=out_graph)
    return True


def main(data_dir: str, chunk: int, chunk_size: int, max_net_size: int, min_net_size: int) -> None:
    """
    Run H-Bond network detection on a chunk of preprocessed ProteinMPNN data.

    Args:
        data_dir (str): Directory containing preprocessed .npz files.
        chunk (int): Which chunk of data to handle.
        chunk_size (int): Size to make each chunk.
        max_net_size (int): Max network size (inclusive).
        min_net_size (int): Min network size (inclusive).

    Returns:
        None
    """

    # Grab chunk of sorted asmbs
    pdbs = sorted(glob.glob(data_dir + "/*/*.npz"))
    pdbs = [p for p in pdbs if 'hbnet' not in p]
    pdbs = [pdbs[i : i + chunk_size] for i in range(0, len(pdbs), chunk_size)]
    asmbs = pdbs[chunk - 1]
    print(
        f"Running network detection on chunk {chunk - 1} of {len(pdbs)}: {len(list(asmbs))} pdbs...",
        flush=True,
    )

    t0 = time.time()
    n_processed = 0
    i = 0

    worker_func = partial(
        detect_asmb, 
        max_net_size=max_net_size, 
        min_net_size=min_net_size,
    )
    # ProcessPool uses a worker timeout argument to avoid hangs
    with ProcessPool(max_workers=1) as p:
        future = p.map(worker_func, asmbs, chunksize=1, timeout=300)
        iterator = future.result()

        while True:
            try:
                flag = next(iterator)
                if flag is True:
                    n_processed += 1
                    print(f"Processed {n_processed}", flush=True)
                else:
                    print(
                        "function failed to find valid hbonds for pdb %s" % asmbs[i],
                        flush=True,
                    )
            except StopIteration:
                break
            except TimeoutError as error:
                print(
                    "function took longer than %d seconds for pdb %s"
                    % (error.args[1], asmbs[i]),
                    flush=True,
                )
            except ProcessExpired as error:
                print(
                    "%s. Exit code: %d for pdb %s" % (error, error.exitcode, asmbs[i]),
                    flush=True,
                )
            except Exception as error:
                print("function raised %s for pdb %s" % (error, asmbs[i]), flush=True)
            finally:
                i += 1

    print(
        f"Succeeded in processing {n_processed}/{len(asmbs)} PDBs in {time.time() - t0:.3f} sec.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess ProteinMPNN dataset for HBDesigner (step 2).")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing preprocessed .npz files.",
        required=True,
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Which chunk of data to handle. Defaults to 1.",
        required=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100, help="Size to make each chunk.", 
        required=False,
    )
    parser.add_argument(
        "--max_net_size",
        type=int,
        default=6,
        help="Max network size (inclusive). Defaults to 6.",
        required=False,
    )
    parser.add_argument(
        "--min_net_size",
        type=int,
        default=2,
        help="Min network size (inclusive). Defaults to 2.",
        required=False,
    )

    args = parser.parse_args()
    main(**vars(args))
