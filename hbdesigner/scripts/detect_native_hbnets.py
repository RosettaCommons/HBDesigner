import argparse
import glob
import os
import time
from copy import deepcopy
from pebble import ProcessExpired, ProcessPool
import networkx as nx
import numpy as np
import pyrosetta
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

MIN_NET, MAX_NET = 2, 6


def detect_asmb(fname) -> bool:
    print(fname)
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
        if (sg_size < MIN_NET) or (sg_size > MAX_NET):
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


def main(args):
    """Preprocess full ProteinMPNN dataset"""

    # Grab chunk of sorted asmbs
    pdbs = sorted(glob.glob(args.data_dir + "/*/*.npz"))
    pdbs = [p for p in pdbs if 'hbnet' not in p]
    pdbs = [pdbs[i : i + args.chunks] for i in range(0, len(pdbs), args.chunks)]
    asmbs = pdbs[args.chunk - 1]
    print(
        f"Running network detection on chunk {args.chunk - 1} of {len(pdbs)}: {len(list(asmbs))} pdbs...",
        flush=True,
    )

    t0 = time.time()
    n_processed = 0
    i = 0

    worker_func = detect_asmb
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing preprocessed .npz files.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Which chunk of data to handle. Defaults to 1.",
    )
    parser.add_argument(
        "--chunks", type=int, default=100, help="Size to make each chunk."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=6,
        help="Max network size (inclusive). Defaults to 6.",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=2,
        help="Min network size (inclusive). Defaults to 2.",
    )

    args = parser.parse_args()
    print(args)
    main(args)
