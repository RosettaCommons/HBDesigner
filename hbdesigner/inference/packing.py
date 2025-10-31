from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pyrosetta
import torch_geometric.data as gd

from pyrosetta import Pose, get_fa_scorefxn
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.pack.task.operation import (
    OperateOnResidueSubset,
    PreventRepackingRLT,
)
from pyrosetta.rosetta.core.scoring import (
    ScoreFunction,
    fa_rep,
    hbond_bb_sc,
    hbond_sc,
)
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueNameSelector,
)
from pyrosetta.rosetta.core.select.util import calc_sc_neighbors
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.basic.options import set_real_option, get_real_option

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.hbnet import crop_by_distance, get_satisfaction
from hbdesigner.data.protein import Protein
from hbdesigner.inference.protein_ops import get_network_res


def minimize_and_score_network(
    pack: gd.Data,
    scaffold: Protein,
    minimize: bool = True,
    core_mask: np.ndarray = None,
    max_BUNs: int = 0,
):
    """
    Minimize and score a packed network using PyRosetta.

    Arguments:
        pack (gd.Data): The packed network data.
        scaffold (Protein): The reference scaffold protein to use for scoring.
        minimize (bool): Whether to minimize the pose after packing. Default is True.
        core_mask (np.ndarray): Mask indicating core residues in the scaffold.
        max_BUNs (int): Maximum number of buried unsatisfied heavy polar atoms allowed. Default is 0.

    Returns:
        Dict[str, Any]: A dictionary containing scores and the packed protein.
    """
    full_scaffold = scaffold.copy()

    # Handle if pack is a Protein, not a gd.Data
    if isinstance(pack, Protein):
        pack_knn = pack.pack_knn.numpy()
        scaffold = scaffold.mask(pack_knn)
        scaffold.clear_sequence()
        net_mask = pack.aatype != rc.restype_order["A"]
        scaffold.aatype[net_mask] = pack.aatype[net_mask]
        scaffold.atom27_xyz[net_mask] = pack.atom27_xyz[net_mask]
        scaffold.atom27_mask[net_mask] = pack.atom27_mask[net_mask]
        core_mask = core_mask[pack_knn]
    else:
        # Crop scaffold down to match pack
        pack_knn = pack.pack_knn.numpy()
        scaffold = scaffold.mask(pack_knn)
        scaffold.clear_sequence()
        scaffold.aatype = pack.aatype.numpy()
        scaffold.atom27_xyz[:, :14] = pack.atom14_xyz.numpy()
        scaffold.atom27_mask[:, :14] = pack.atom14_mask.numpy()
        core_mask = core_mask[pack_knn]

    # Import into PyRosetta
    pose = Pose()
    pdbstr = scaffold.to_pdb(unk_to_gly=True)
    pose_from_pdbstring(pose, pdbstr)

    # Run quick-and-dirty connectivity check before expensive minimization
    default_energy = get_real_option("score:hb_max_energy")
    set_real_option("score:hb_max_energy", 10.0)
    is_valid = check_valid_network(pose)
    qad_scores = get_satisfaction(pose, core_mask)
    set_real_option("score:hb_max_energy", default_energy)
    if qad_scores["buried_heavy_unsats"] > max_BUNs or not is_valid:
        return None

    # Minimize, if enabled
    if minimize:
        pose = minimize_network(pose, cartesian=True)

    # Check if network is "valid" aka fully connected
    is_valid = check_valid_network(pose)
    if not is_valid:
        return None

    # Run energy scoring vs polyG backbone
    scores = score_network(pose, scaffold.copy())
    # Run satisfaction/BUNs scoring
    scores.update(get_satisfaction(pose, core_mask))

    net_res = [i for i, aa in enumerate(pose.sequence()) if aa != "G"]
    n_core_res = np.sum(core_mask[net_res])

    # Convert pose to Protein so we can return it
    buffer = pyrosetta.rosetta.std.stringbuf()
    pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
    pdb_block = buffer.str()
    protein = Protein.from_pdb_string(pdb_block, discard_Hs=False, from_rosetta=True)

    # Graft network res back onto original scaffold
    full_scaffold.atom27_xyz[pack_knn] = protein.atom27_xyz
    full_scaffold.atom27_mask[pack_knn] = protein.atom27_mask
    full_scaffold.aatype[:] = rc.restype_order["G"]
    full_scaffold.aatype[pack_knn] = protein.aatype

    protein = full_scaffold.copy()

    net_string = get_network_res(protein)
    n_chains = len(set([ns[0] for ns in net_string.split(":")]))

    scores.update(
        {
            "network": net_string,
            "protein": protein,
            "hash": hash(pdbstr),
            "is_valid": is_valid,
            "n_chains": n_chains,
            "n_core_res": n_core_res,
        }
    )
    return scores


def pack_and_score_network(
    sample: Dict[str, List[int]],
    scaffold: Protein,
    minimize: bool = True,
    pack_crop: float = 10.0,
    core_mask: np.ndarray = None,
    max_BUNs: int = 0,
) -> Dict[str, Any]:
    """
    Pack, minimize, and score a network with PyRosetta.

    Arguments:
        sample (Dict[str, List[int]]): Sample containing network residue indices and sequence.
        scaffold (Protein): The reference scaffold protein to use for scoring.
        minimize (bool): Whether to minimize the pose after packing. Default is True.
        pack_crop (float): Distance threshold for cropping scaffold to network residues. Default is 10.0.
        core_mask (np.ndarray): Mask indicating core residues in the scaffold.
        max_BUNs (int): Maximum number of buried unsatisfied heavy polar atoms. Default is 0.

    Returns:
        Dict[str, Any]: A dictionary containing scores and the packed protein.
    """
    # Thread on network seq
    scaffold.clear_sequence()
    scaffold.aatype[:] = rc.restype_order["G"]
    scaffold.aatype[sample["net_res"]] = sample["seq"]

    # Crop scaffold to only net res and their K-closest-neighbors
    net_pos_orig = np.where(scaffold.aatype != rc.restype_order["G"])[0]

    if pack_crop > 0.0:
        scaffold_cropped, knn = crop_by_distance(scaffold, net_pos_orig, pack_crop)
        core_mask = core_mask[knn]
    else:
        scaffold_cropped = scaffold

    # Import into PyRosetta
    pose = Pose()
    pdbstr = scaffold_cropped.to_pdb(unk_to_gly=True)
    pose_from_pdbstring(pose, pdbstr)

    # Pack if possible
    pose = pack_network(pose, core_mask, minimize, cartesian=True, max_BUNs=max_BUNs)
    if pose is None:
        return None

    # Check if network is "valid" aka fully connected
    is_valid = check_valid_network(pose)
    if not is_valid:
        return None

    # Run energy scoring vs polyG backbone
    scores = score_network(pose, scaffold_cropped.copy())
    # Run satisfaction/BUNs scoring
    scores.update(get_satisfaction(pose, core_mask))
    net_res = [i for i, aa in enumerate(pose.sequence()) if aa != "G"]
    n_core_res = np.sum(core_mask[net_res])

    # Convert pose to Protein so we can return it
    buffer = pyrosetta.rosetta.std.stringbuf()
    pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
    pdb_block = buffer.str()
    protein = Protein.from_pdb_string(pdb_block, discard_Hs=False, from_rosetta=True)

    # Graft network res back onto original scaffold
    scaffold.atom27_xyz[knn] = protein.atom27_xyz
    scaffold.atom27_mask[knn] = protein.atom27_mask

    protein = scaffold.copy()

    # Recalculate net res names after graft for output
    net_string = get_network_res(protein)
    n_chains = len(set([ns[0] for ns in net_string.split(":")]))
    scores.update(
        {
            "network": net_string,
            "protein": protein,
            "hash": hash(pdbstr),
            "n_chains": n_chains,
            "n_core_res": n_core_res,
        }
    )
    return scores


def pack_network(pose: Pose, core_mask: np.ndarray, minimize: bool = False, cartesian: bool = True, max_BUNs: int = 0) -> Pose:
    """
    Packs a Pose using a simplified H-bond score function.

    Arguments:
        pose (Pose): The PyRosetta pose to pack.
        core_mask (np.ndarray): Mask indicating core residues in the scaffold.
        minimize (bool): Whether to minimize the pose after packing. Default is False.
        cartesian (bool): Whether to use Cartesian minimization. Default is True.
        max_BUNs (int): Maximum number of buried unsatisfied heavy polar atoms allowed. Default is 0.

    Returns:
        Pose: The packed and optionally minimized pose.
    """
    # Simple scorefxn using only h-bond and repulsive terms
    # Note: default scorefxn should be symmetry-aware
    scorefxn = ScoreFunction()
    scorefxn.set_weight(fa_rep, 0.55)
    scorefxn.set_weight(hbond_sc, 1.0)

    # Set up packer and task
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())

    # Skip packing for non-motif res
    not_motif_sel = ResidueNameSelector()
    not_motif_sel.set_residue_name3("GLY")

    # prevent non motif residues from repacking/designing
    fix_non_motif = OperateOnResidueSubset(PreventRepackingRLT(), not_motif_sel)
    tf.push_back(fix_non_motif)

    packer = pack_min.PackRotamersMover(scorefxn)
    packer.task_factory(tf)

    # Set up MinMover for postprocessing
    if minimize:
        min_scorefxn = get_fa_scorefxn()

        # Cart min needs adjusted scorefxn
        if cartesian:
            min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.cart_bonded, 0.5)
            min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.pro_close, 0.0)

        movemap = MoveMap()
        movemap.clear()
        movemap.set_bb(False)
        movemap.set_jump(False)
        for res in range(1, pose.total_residue() + 1):
            if pose.residue(res).name3() not in ["ALA", "GLY"]:
                movemap.set_chi(res, True)
                # Cart min uses extra DOFs
                if cartesian:
                    for a_idx in range(
                        pose.residue(res).first_sidechain_atom(),
                        pose.residue(res).nheavyatoms() + 1,
                    ):
                        atom_id = pyrosetta.rosetta.core.id.AtomID(a_idx, res)
                        # Only use sidechain atoms
                        movemap.set(
                            pyrosetta.rosetta.core.id.DOF_ID(
                                atom_id, pyrosetta.rosetta.core.id.DOF_Type.THETA
                            ),
                            True,
                        )
                        movemap.set(
                            pyrosetta.rosetta.core.id.DOF_ID(
                                atom_id, pyrosetta.rosetta.core.id.DOF_Type.D
                            ),
                            True,
                        )

        minmover = MinMover()
        minmover.score_function(min_scorefxn)
        minmover.movemap(movemap)
        minmover.tolerance(1e-6)
        minmover.max_iter(1_000)
        minmover.type("lbfgs_armijo_nonmonotone")
        minmover.cartesian(cartesian)

    packer.apply(pose)

    # Run quick-and-dirty connectivity check before expensive minimization
    default_energy = get_real_option("score:hb_max_energy")
    set_real_option("score:hb_max_energy", 10.0)
    is_valid = check_valid_network(pose)
    qad_scores = get_satisfaction(pose, core_mask)
    set_real_option("score:hb_max_energy", default_energy)
    if qad_scores["buried_heavy_unsats"] > max_BUNs or not is_valid:
        return None

    if minimize:
        minmover.apply(pose)
    return pose


def minimize_network(pose: Pose, cartesian: bool = True) -> Pose:
    """
    Minimizes a Pose.

    Arguments:
        pose (Pose): The PyRosetta pose to minimize.
        cartesian (bool): Whether to use Cartesian minimization. Default is True.

    Returns:
        Pose: The minimized pose.
    """

    # Set up MinMover for postprocessing
    min_scorefxn = get_fa_scorefxn()
    if cartesian:
        min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.cart_bonded, 0.5)
        min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.pro_close, 0.0)

    movemap = MoveMap()
    movemap.clear()
    movemap.set_bb(False)
    movemap.set_jump(False)
    for res in range(1, pose.total_residue() + 1):
        if pose.residue(res).name3() not in ["ALA", "GLY"]:
            movemap.set_chi(res, True)

            if cartesian:
                for a_idx in range(
                    pose.residue(res).first_sidechain_atom(),
                    pose.residue(res).nheavyatoms() + 1,
                ):
                    atom_id = pyrosetta.rosetta.core.id.AtomID(a_idx, res)
                    # Only use sidechain atoms
                    movemap.set(
                        pyrosetta.rosetta.core.id.DOF_ID(
                            atom_id, pyrosetta.rosetta.core.id.DOF_Type.THETA
                        ),
                        True,
                    )
                    movemap.set(
                        pyrosetta.rosetta.core.id.DOF_ID(
                            atom_id, pyrosetta.rosetta.core.id.DOF_Type.D
                        ),
                        True,
                    )

    minmover = MinMover()
    minmover.score_function(min_scorefxn)
    minmover.movemap(movemap)
    minmover.tolerance(1e-6)
    minmover.max_iter(1_000)
    minmover.type("lbfgs_armijo_nonmonotone")
    minmover.cartesian(cartesian)

    minmover.apply(pose)
    return pose


def score_network(pose: Pose, scaffold: Protein) -> Dict[str, float]:
    """
    Score a Pose for network energy and burial.

    Arguments:
        pose (Pose): The PyRosetta pose to score.
        scaffold (Protein): The reference scaffold protein used for scoring.

    Returns:
        Dict[str, float]: A dictionary containing the scores:
            - "HB_Score_full": Full score function score.
            - "HB_Score_hb": Hydrogen bond score.
            - "Avg_Burial": Average burial of sidechain neighbors.
    """
    # Score vs polyG backbone
    polyG_pose = Pose()
    scaffold.clear_sequence()
    pose_from_pdbstring(polyG_pose, scaffold.to_pdb(unk_to_gly=True))
    seq, seq_polyG = pose.sequence(), polyG_pose.sequence()
    net_idx = [i for i, (s, sg) in enumerate(zip(seq, seq_polyG)) if s != sg]
    n_res = len(net_idx)

    scorefxn = ScoreFunction()
    scorefxn.set_weight(fa_rep, 0.55)
    scorefxn.set_weight(hbond_sc, 1.0)
    scorefxn.set_weight(hbond_bb_sc, 1.0)
    energy_hb = (scorefxn(pose) - scorefxn(polyG_pose)) / n_res

    # Get full scorefxn score
    scorefxn = get_fa_scorefxn()
    energy_full = (scorefxn(pose) - scorefxn(polyG_pose)) / n_res

    sc_neighbors = np.array(calc_sc_neighbors(pose))
    avg_sc_neighbors = np.mean(sc_neighbors[net_idx])

    return {
        "HB_Score_full": energy_full,
        "HB_Score_hb": energy_hb,
        "Avg_Burial": avg_sc_neighbors,
    }


def check_valid_network(pose: Pose) -> bool:
    """
    Check if Pose has a valid hydrogen bond network.

    A valid network is defined as:
    - All non-glycine residues have at least one hydrogen bond.
    - The hydrogen bond network is a single connected component.

    Arguments:
        pose (Pose): The PyRosetta pose to check.

    Returns:
        bool: True if the network is valid, False otherwise.
    """
    # Collect sc-sc bonds and check if all res are participating
    hbondset = pose.get_hbonds(exclude_bb=True, exclude_bsc=True, exclude_scb=True)
    hbonds = hbondset.hbonds()
    edge_list = []
    for bond in hbonds:
        acc_res = int(bond.acc_res())
        don_res = int(bond.don_res())
        edge_list.append([acc_res, don_res])

    # Invalid if no h-bonds found
    if len(edge_list) < 1:
        return False

    hb_res = set([x for xs in edge_list for x in xs])

    # Every non-GLY should have an hbond
    for i in range(1, pose.total_residue() + 1):
        rname3 = pose.residue(i).name()[:3]
        if rname3 != "GLY" and rname3 != "VRT":
            if i not in hb_res:
                return False

    # Check that network makes one continuous component
    g = nx.Graph()
    g.add_edges_from(edge_list)
    n_comp = len(list(nx.connected_components(g)))
    return n_comp == 1
