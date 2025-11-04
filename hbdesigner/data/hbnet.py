from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pyrosetta
import time
import subprocess
import torch
import torch_geometric.data as gd
from pyrosetta import Pose, get_fa_scorefxn
from pyrosetta.rosetta.basic.options import set_boolean_option, set_real_option
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.scoring import (
    ScoreFunction,
    fa_rep,
    hbond_bb_sc,
    hbond_sc,
)
from pyrosetta.rosetta.core.select.residue_selector import LayerSelector
from pyrosetta.rosetta.core.select.util import calc_sc_neighbors
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from scipy.spatial.distance import cdist
from torch_scatter import scatter

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import impute_CB
from hbdesigner.data.protein import Protein

REDUCE_EXE = "/spshared/apps/reduce/reduce_src/reduce"
HET_DICT = "/spshared/apps/reduce/reduce_wwPDB_het_dict.txt"


def batch_to_proteins(batch: gd.Batch) -> Tuple[List[Protein], List[gd.Data]]:
    """
    Converts Batch object into a list of Proteins and a list of gd.Data objects.

    Args:
        batch (gd.Batch): Batch returned from HBDesignerDataset.collate.

    Returns:
        List[Protein]: List of Protein objects.
        List[gd.Data]: List of gd.Data objects.
    """
    batch = batch.to("cpu")
    b_list = batch.to_data_list()
    proteins = []
    for b in b_list:
        p = Protein(
            atom27_xyz=b["atom14_xyz"].numpy(),
            atom27_mask=b["atom14_mask"].numpy(),
            aatype=b["aatype"].numpy(),
            residue_index=b["residue_index"].numpy(),
            chain_index=b["c_idx"].numpy(),
            b_factors=np.zeros_like(b["atom14_mask"].numpy()),
        )
        proteins.append(p)

    return proteins, b_list


def get_satisfaction(
    pose: Pose,
    core_mask: np.ndarray = None
) -> Dict[str, float]:
    """
    Direct python port of the find_unsats function from Rosetta HBNet.

    Arguments:
        pose (Pose): Pose that has already been scored with appropriate energy fxn.
        core_mask (np.ndarray, optional): Preexisting core mask for burial determination.

    Returns:
        Dict[str, float]: Satisfaction metric dict. Includes BUNs, BUPHs, and Saturation scores.
    """
    # Get all H-bonds with sc-bb enabled
    hbondset = pose.get_hbonds(exclude_bb=True, exclude_bsc=False, exclude_scb=False)

    # Collect core selector
    if core_mask is None:
        core_sel = LayerSelector()
        core_sel.set_cutoffs(core=4.4, surf=2.0)
        core_sel.set_layers(pick_core=True, pick_boundary=False, pick_surface=False)
        core_mask = np.array(core_sel.apply(pose))
    else:
        assert pose.total_residue() == core_mask.size, "If provided for scoring, core mask size must match pose size."

    hydroxyls_must_donate = False
    seq = pose.sequence()
    total_polar_groups_that_could_hbond, polar_groups_making_hbonds = 0, 0
    num_unsat_Hpol, num_heavy_unsat = 0, 0

    for i in range(1, pose.total_residue() + 1):
        # Get only network res
        if seq[i - 1] != "G":
            res = pose.residue(i)
            # Iterate over heavy atoms in side chain
            for a_idx in range(1, res.nheavyatoms() + 1):
                if not res.atom_is_backbone(a_idx):
                    atm_type = res.atom_type(a_idx)

                    # 1. Check if non-OH donor
                    if (
                        (atm_type.is_donor())
                        and (res.atomic_charge(a_idx) != 0.0)
                        and (atm_type.name() != "OH")
                    ):
                        h_count, h_unsat = 0, 0
                        # One point for each h-atom
                        for hatm in range(
                            res.attached_H_begin(a_idx), res.attached_H_end(a_idx) + 1
                        ):
                            h_count += 1
                            total_polar_groups_that_could_hbond += 1
                            # Check if atom is h-bonding
                            atm_id = pyrosetta.rosetta.core.id.AtomID(
                                atomno_in=hatm, rsd_in=i
                            )
                            if hbondset.nhbonds(atm_id, False) == 0:
                                # If not and if core, penalize it
                                if core_mask[i - 1]:
                                    num_unsat_Hpol += 1
                                    h_unsat += 1
                            else:
                                polar_groups_making_hbonds += 1
                        # Count heavy atom only if all H atoms are unsat
                        if h_unsat == h_count:
                            # Does not need to be buried
                            num_heavy_unsat += 1

                    # 2. Check if any acceptor
                    elif (atm_type.is_acceptor()) and (res.atomic_charge(a_idx) != 0.0):
                        total_polar_groups_that_could_hbond += 1

                        # Check if carbonyl (2x lone pair) or -OH (donor/acceptor)
                        is_sp2 = "SP2_HYBRID" in atm_type.get_all_properties()
                        if res.atom_type(a_idx).name() == "OH" or is_sp2:
                            total_polar_groups_that_could_hbond += 1

                        # Check if atom is h-bonding
                        atm_id = pyrosetta.rosetta.core.id.AtomID(
                            atomno_in=a_idx, rsd_in=i
                        )
                        num_hbonds = hbondset.nhbonds(atm_id, False)
                        polar_groups_making_hbonds += num_hbonds

                        if num_hbonds == 0:
                            # Only penalize if core
                            if core_mask[i - 1]:
                                # If no bond, check if -OH
                                if atm_type.name() == "OH":
                                    hatm = res.attached_H_begin(a_idx)
                                    hatm_id = pyrosetta.rosetta.core.id.AtomID(
                                        atomno_in=hatm, rsd_in=i
                                    )
                                    hatm_hbonds = hbondset.nhbonds(hatm_id, False)
                                    # Only penalize if -OH fails to be both donor/acceptor
                                    if hatm_hbonds == 0:
                                        num_unsat_Hpol += 1
                                        num_heavy_unsat += 1
                                    else:
                                        polar_groups_making_hbonds += 1
                                # All other failed donors are automatically counted
                                else:
                                    num_unsat_Hpol += 1
                                    num_heavy_unsat += 1
                        else:
                            # If at least one bond, check if -OH
                            if atm_type.name() == "OH":
                                hatm = res.attached_H_begin(a_idx)
                                hatm_id = pyrosetta.rosetta.core.id.AtomID(
                                    atomno_in=hatm, rsd_in=i
                                )
                                hatm_hbonds = hbondset.nhbonds(hatm_id, False)
                                # Count -OH donor H-bond here, if present
                                if hatm_hbonds == 0:
                                    if core_mask[i - 1] and hydroxyls_must_donate:
                                        num_unsat_Hpol += 1
                                        num_heavy_unsat += 1
                                else:
                                    polar_groups_making_hbonds += 1

    pct_hb_capacity = (
        float(polar_groups_making_hbonds) / float(total_polar_groups_that_could_hbond)
        if total_polar_groups_that_could_hbond != 0
        else 0.0
    )
    sat_stats = {
        "saturation": pct_hb_capacity,
        "buried_heavy_unsats": num_heavy_unsat,
        "buried_unsat_Hpol": num_unsat_Hpol,
    }
    return sat_stats


def rosetta_hbond_detect(
    p: Protein, max_energy: float = 0.0, optH: bool = True, optH_MCA: bool = True,
) -> List[Tuple[int, int]]:
    """
    Detect Rosetta HBonds based on energy threshold.

    Args:
        p (Protein): Stripped (packed) protein with polyG everywhere except the network.
        max_energy (float): Max energy to consider a 'valid' HBond. Larger is more permissive. Default is 0.0.
        optH (bool): Whether to let Rosetta re-optimize the hydrogens in the input Protein. Default is True.
        optH_MCA (bool): Whether to use more rigorous but slower optH MCA protocol. Default is True.
    """
    t0 = time.time()
    # Configure scoring settings, if not already set
    set_real_option("score:hb_max_energy", max_energy)
    set_boolean_option("packing:no_optH", not (optH))
    set_boolean_option("packing:optH_MCA", optH_MCA)
    set_boolean_option("packing:flip_HNQ", True)
    set_boolean_option("score:hbond_disable_bbsc_exclusion_rule", True)

    # Set up polyG backbone for energy calc
    pose, polyG_pose = Pose(), Pose()
    pair_list = []
    try:
        pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))
        # Make PolyG pose for comparison
        polyG_p = deepcopy(p)
        polyG_p = clear_non_network_res(
            polyG_p, b=None, mask=np.zeros_like(p.aatype), unk="G"
        )

        pose_from_pdbstring(polyG_pose, polyG_p.to_pdb(unk_to_gly=True))

    except RuntimeError:
        print("Rosetta failed to accept PDB!")
        return pair_list

    # Need to apply FA scorefxn to populate HBondSet
    scorefxn = get_fa_scorefxn()
    scorefxn(pose)
    hbondset = pose.get_hbonds(exclude_bb=True, exclude_bsc=True, exclude_scb=True)
    hbonds = hbondset.hbonds()

    # Collect HBonds
    for bond in hbonds:
        # Convert from Pose to Protein numbering
        acc_res = int(bond.acc_res()) - 1
        don_res = int(bond.don_res()) - 1
        pair_list.append((acc_res, don_res))

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

    # Calculate seq-independent burial of network residues
    sc_neighbors = np.array(calc_sc_neighbors(pose))
    avg_sc_neighbors = np.mean(sc_neighbors[net_idx])

    stats = {
        "HBScore_Full": energy_full,
        "HBScore_HBond": energy_hb,
        "Avg_Sc_Neighbors": avg_sc_neighbors,
    }
    # Get Satisfaction and BUNs/BUPHs
    stats.update(get_satisfaction(pose))
    t1 = time.time()
    stats["score_time"] = (t1 - t0)
    return pair_list, stats


def biotite_hbond_detect(
    p: Protein, 
) -> List[Tuple[int, int]]:
    """
    Detect HBonds based on Baker-Hubbard criteria using Biotite.

    Args:
        p (Protein): Stripped (packed) protein with polyG everywhere except the network.
    """
    t0 = time.time()

    # Parse Protein into Biotite structure
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import hbond
    from io import StringIO
    from proteingfn.data.protein import PDB_CHAIN_IDS

    try:
        pdb_str = p.to_pdb()
    except IndexError:
        print("No hydrogens found in Protein, returning empty HBond stats...")
        return [], {
            "HBScore_Full": 0.0,
            "HBScore_HBond": 0.0,
            "Avg_Sc_Neighbors": 0.0,
            "saturation": 0.0,
            "buried_heavy_unsats": 0,
            "buried_unsat_Hpol": 0,
            "score_time": 0.0,
        }

    struct = PDBFile.read(StringIO(pdb_str))
    atom_stack = struct.get_structure()

    from biotite.structure import connect_via_residue_names
    atom_stack.bonds = connect_via_residue_names(atom_stack)

    # Create mask for backbone atoms (including H atoms)
    backbone_atoms = ['N', 'CA', 'C', 'O', 'OXT', 'H', 'HA']
    sidechain_mask = ~np.isin(atom_stack.atom_name, backbone_atoms)
    sidechain_stack = atom_stack[:, sidechain_mask]

    # Get sc-sc bonds only
    # AHD_dist = 2.5 # default
    AHD_dist = 3.2 # more permissive
    # AHD_dist = 4.0 # very permissive
    triplets, mask = hbond(sidechain_stack, donor_elements=["O", "N"], acceptor_elements=["O", "N"], cutoff_dist=AHD_dist)

    pair_list = []
    for donor, _, acceptor in triplets:
        # Convert from Biotite to Protein numbering
        biotite_resid = sidechain_stack.res_id[donor]
        biotite_chainid = PDB_CHAIN_IDS.index(sidechain_stack.chain_id[donor])
        don_res = np.where((biotite_chainid == p.chain_index) * biotite_resid == p.residue_index)[0][0]

        biotite_resid = sidechain_stack.res_id[acceptor]
        biotite_chainid = PDB_CHAIN_IDS.index(sidechain_stack.chain_id[acceptor])
        acc_res = np.where((biotite_chainid == p.chain_index) * biotite_resid == p.residue_index)[0][0]

        # don_res = int(sidechain_stack.res_id[donor]) - 1
        # acc_res = int(sidechain_stack.res_id[acceptor]) - 1
        pair_list.append((acc_res, don_res))

    stats = {
        "HBScore_Full": 0.0,
        "HBScore_HBond": 0.0,
        "Avg_Sc_Neighbors": 0.0,
        "saturation": 0.0,
        "buried_heavy_unsats": 0,
        "buried_unsat_Hpol": 0,
    }
    
    t1 = time.time()
    stats["score_time"] = (t1 - t0)
    return pair_list, stats


def clear_non_network_res(
    p: Protein, b: gd.Data, mask: np.array, unk: str = "A"
) -> Protein:
    """
    Takes a Protein and replaces non-network res with ALA.
    Then uses the NLL mask to add back the network aatypes for packing.

    Args:
        p (Protein): Protein object to be stripped.
        b (gd.Data): Output of HBDesignDataset.featurize(p, g). Must match.
        mask (np.ndarray): Mask of network res in protein.
        unk (str): Which residue to replace non-network res with. Default is A.

    Returns:
        Protein: Protein that is PolyALA except for network res. For these, aatype is kept but sc is not.
    """
    hb_idx = np.where(mask)[0]
    hb_aatype = p.aatype[hb_idx]

    # Need to re-wrap if only one res
    if not isinstance(hb_aatype, np.ndarray):
        hb_aatype = np.array([hb_aatype])

    # Clear seq and sc atoms
    p.clear_sequence()

    # Impute CB for PolyA
    p.aatype[:] = rc.restype_order[unk]
    if unk == "A":
        p.atom27_mask[:, 4] = np.prod(p.atom27_mask[:, :4], axis=-1)
        p.atom27_xyz[:, 4, :] = (
            impute_CB(
                p.atom27_xyz[:, 0, :], p.atom27_xyz[:, 1, :], p.atom27_xyz[:, 2, :]
            )
            * p.atom27_mask[:, 4:5]
        )

    # Insert HBNet residues back in
    for idx, aatype in zip(hb_idx, hb_aatype):
        i, aa = idx, aatype
        p.aatype[i] = aa
        # Edge case - gly has no Cb, parsing will crash
        if aa == rc.restype_order["G"]:
            p.atom27_mask[i, 4] = 0.0
            p.atom27_xyz[i, 4] = 0.0
    return p


def packer_task(p: Protein) -> Protein:
    """
    Rosetta packer routine for a single Protein instance.

    Args:
        p (Protein): Preprocessed Protein obj with non-network residues cleared.
        minimize (bool): Whether to run post-packing minimization.
    Returns:
        Protein: Packed protein retrieved from Rosetta Pose output.
    """
    t0 = time.time()
    # Import into PyRosetta
    pose = Pose()
    pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))

    # Simple scorefxn using only h-bond and repulsive terms
    scorefxn = ScoreFunction()
    scorefxn.set_weight(fa_rep, 0.55)
    scorefxn.set_weight(hbond_sc, 1.0)

    # Set up packer and task just once
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    packer = pack_min.PackRotamersMover(scorefxn)
    packer.task_factory(tf)

    # Apply to pose
    packer.apply(pose)

    # Convert pose to Protein
    buffer = pyrosetta.rosetta.std.stringbuf()
    pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
    pdb_block = buffer.str()
    # Rosetta drops the ghost residues, so we need to pad them back in
    ros_p = Protein.from_pdb_string(pdb_block, discard_Hs=False)
    t1 = time.time()
    ros_p = ros_p.pad(n=p.n_res)
    ros_p.pack_time = t1 - t0
    return ros_p


def minimize_task(p: Protein, cartesian: bool = True) -> Protein:
    """
    Rosetta MinMover routine for a single Protein instance.

    Args:
        p (Protein): Preprocessed Protein obj with non-network residues cleared.
        cartesian (bool): Whether to enable cartesian minimization (slower, but more accurate). Defaults to True.
    Returns:
        Protein: Minimized protein retrieved from Rosetta Pose output.
    """
    t0 = time.time()
    # Import into PyRosetta
    pose = Pose()
    pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))

    # Set up MinMover for postprocessing
    min_scorefxn = get_fa_scorefxn()

    if cartesian:
        min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.cart_bonded, 0.5)
        min_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.pro_close, 0.0)

    movemap = MoveMap()
    movemap.set_bb(False)
    movemap.set_jump(False)
    for res in range(1, pose.total_residue() + 1):
        if pose.residue(res).name3() not in ["ALA", "GLY"]:
            movemap.set_chi(res, True)

            if cartesian:
                for a_idx in range(pose.residue(res).first_sidechain_atom(), pose.residue(res).nheavyatoms() + 1):
                    atom_id = pyrosetta.rosetta.core.id.AtomID(a_idx, res)
                    # Only use sidechain atoms
                    movemap.set(pyrosetta.rosetta.core.id.DOF_ID(atom_id, pyrosetta.rosetta.core.id.DOF_Type.THETA), True)
                    movemap.set(pyrosetta.rosetta.core.id.DOF_ID(atom_id, pyrosetta.rosetta.core.id.DOF_Type.D), True)

    minmover = MinMover()
    minmover.score_function(min_scorefxn)
    minmover.movemap(movemap)
    minmover.tolerance(1e-6)
    minmover.max_iter(1_000)
    minmover.type("lbfgs_armijo_nonmonotone")
    minmover.cartesian(cartesian) # NOTE: this makes it slower but more accurate
    minmover.apply(pose)

    # Convert pose to Protein
    buffer = pyrosetta.rosetta.std.stringbuf()
    pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
    pdb_block = buffer.str()
    # Rosetta drops the ghost residues, so we need to pad them back in
    ros_p = Protein.from_pdb_string(pdb_block, discard_Hs=False, from_rosetta=True)
    t1 = time.time()
    ros_p = ros_p.pad(n=p.n_res)

    if hasattr(p, "pack_time"):
        ros_p.pack_time = p.pack_time + (t1 - t0)
    else:
        ros_p.pack_time = t1 - t0
    return ros_p


def run_hydride(p: Protein) -> Protein:

    # Parse Protein into Biotite structure
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import connect_via_residue_names
    from io import StringIO
    import hydride

    pdb_str = p.to_pdb(unk_to_gly=True)    
    struct = PDBFile.read(StringIO(pdb_str))
    atom_array = struct.get_structure()[0, :]

    # Set up charges and bonds for hydride
    atom_array.bonds = connect_via_residue_names(atom_array)
    charges = hydride.estimate_amino_acid_charges(atom_array, ph=7.0)
    atom_array.set_annotation("charge", charges)

    # Add, then relax with UFF
    atom_array, _ = hydride.add_hydrogen(atom_array)
    atom_array.coords = hydride.relax_hydrogen(atom_array)

    # Convert back to Protein
    output = PDBFile()
    output.set_structure(atom_array)
    buffer = StringIO()
    output.write(buffer)

    p = Protein.from_pdb_string(buffer.getvalue(), discard_Hs=False)
    return p


def run_pdb2pqr(p: Protein) -> Protein:
    """
    Run PDB2PQR on Protein to add hydrogens."""
    import subprocess
    import tempfile
    import os
    from contextlib import contextmanager

    @contextmanager
    def temporary_file(suffix='', dir=None):
        """Context manager for a temporary file that gets deleted."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=dir)
        os.close(fd)  # Close the file descriptor
        try:
            yield path
        finally:
            if os.path.exists(path):
                os.unlink(path)

    temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None
    pdb2pqr_EXE = "/home/hdieckhaus/miniforge3/envs/proteingfn/bin/pdb2pqr"
    with temporary_file(suffix='.pdb', dir=temp_dir) as tmp_input, \
         temporary_file(suffix='.pdb', dir=temp_dir) as tmp_output:
        
        with open(tmp_input, 'w') as f:
            f.write(p.to_pdb(unk_to_gly=True))
        
        pdb2pqr_cmd = f"{pdb2pqr_EXE} --pdb-output {tmp_output} --with-ph 7.0 {tmp_input} {tmp_output}"
        subprocess.run(
            pdb2pqr_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )

        with open(tmp_output, 'r') as f:
            out_pdb = f.read()

    return Protein.from_pdb_string(out_pdb, discard_Hs=False)


def run_reduce(
    pdb_str: str, his: bool = True, flip: bool = True, database: str = HET_DICT, timeout: Optional[int] = None
) -> str:
    """Runs Reduce on a pdb_str

    Args:
        pdb_str (str): The string of the PDB to run Reduce on.
        his (bool, optional): If True, include the -HIS argument for Reduce; otherwise, don't include it. Defaults to True.
        flip (bool, optional): If True, includes the -FLIP argument for Reduce; otherwise, doesn't include it. Defaults to True.
        database (str, optional): Path to the HETATM database for Reduce. Defaults to HET_DICT.
        timeout (int, optional): If provided, sets the timeout value for trying to run Reduce. Defaults to None.
    """
    # Build the command to run Reduce
    reduce_cmd = [REDUCE_EXE, "-q", "-DB", database, "-"]
    if his:
        reduce_cmd += ["-HIS"]
    if flip:
        reduce_cmd += ["-FLIP"]
    pop = subprocess.Popen(
        reduce_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    # Pass the string to Reduce
    if timeout is None:
        out_pdb = pop.communicate(input=pdb_str)[0]
    else:
        # Need a timeout for really large PDBs
        try:
            out_pdb = pop.communicate(input=pdb_str, timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            pop.kill()
            raise subprocess.TimeoutExpired

    return out_pdb

def safe_run(p: Protein, mode: str = "pack"):
    """Keeps run from failing due to isolated errors."""
    try:
        if mode == "pack":
            return packer_task(p)
        elif mode == "minimize-cart":
            return minimize_task(p, True)
        elif mode == "reduce":
            return run_reduce(p, his=True, flip=False)
        elif mode == "hydride":
            return run_hydride(p)
        elif mode == "pdb2pqr":
            return run_pdb2pqr(p)
        else:
            return minimize_task(p, False)
    except Exception:
        print(f"Job distributor failed to run {mode} on {p}, returning unpacked protein.")
        return p


def pack_with_rosetta(
    proteins: Sequence[Protein], n_workers: int = 1, mode: str = "pack"
) -> Sequence[Protein]:
    """
    Uses the Rosetta packer to pack only the residues in the assigned network.

    Args:
        proteins (Sequence[Protein]): List of Proteins to pack.
        n_workers (int): How many workers to run in parallel for packing.
        mode (str): Which routine to run (pack or minimize or minimize-cart).

    Returns:
        Sequence[Protein]: The repacked Protein objects from Rosetta.
    """

    with Pool(n_workers) as p:
        proteins = p.map(
            partial(
                safe_run,
                mode=mode,
            ),
            proteins,
        )
    return proteins


def get_guide_atom(xyz: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Calculate a stochastic guide atom pulled from normal dist centered at centroid of provided points.

    Arguments:
        xyz (np.ndarray): backbone atoms for each network residue [N, 3, 3]
        sigma (float): Sigma of normal distribution from which the guide atom xyz is sampled.

    Returns:
        np.ndarray: guide atom coordinates. [1, 3]
    """

    xyz_Cb = impute_CB(xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :])  # [N, 3]
    xyz_centroid = np.mean(xyz_Cb, axis=0)  # [3]
    return np.random.normal(loc=xyz_centroid, scale=sigma)[None, :]  # [1, 3]


def get_seq_cond(hbnet_res: np.ndarray) -> np.ndarray:
    """
    Generate a sequence conditioning vector for a given residue set.

    Arguments:
        hbnet_res (np.ndarray): Array of aatypes for network of size N with shape [N].

    Returns:
        np.ndarray: Normalized vector of shape [1, 21] ready for input into HBDesigner3Model.

    Notes:
        All non-X (21) res will receive 1 point per occurrence.
        Any X res will get their 1 point distributed across all valid polar residues.

    Examples:
        [2, 3, 3] -> [0, 0, 0.33, 0.66, 0,
                    [0, 0, 0, 0, 0,
                    [0, 0, 0, 0, 0,
                    [0, 0, 0, 0, 0, 0]
        [2, 3, X] -> [0, 0.03, 0.36, 0.36, 0,
                    [0.03, 0.03, 0, 0.03, 0,
                    [0, 0.03, 0, 0, 0,
                    [0.03, 0.03, 0.03, 0.03, 0, 0]
        [X, X] -> [0, 0.09, 0.09, 0.09, 0,
                    [0.09, 0.09, 0, 0.09, 0,
                    [0, 0.09, 0, 0, 0,
                    [0.09, 0.09, 0.09, 0.09, 0, 0]
    """
    unique, counts = np.unique(hbnet_res, return_counts=True)
    aatype_dist = np.zeros(rc.restype_num + 1, dtype=np.float32)
    for u, c in zip(unique, counts):
        # For UNK token, spread prob out across all polar restypes
        if u == 20:
            aatype_dist[rc.restype_hb_idx] += 1.0 / rc.restype_hb_idx.size
        else:
            aatype_dist[u] += c
    aatype_dist /= np.sum(aatype_dist) + 1e-8
    aatype_dist[aatype_dist > 10] = 0.0

    return aatype_dist[None, :]


def calc_seq_rec_batched(
    pred_pos: torch.tensor,
    true_pos: torch.tensor,
    pred_seq: torch.tensor,
    true_seq: torch.tensor,
    aatype_batch: torch.tensor,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Calculates seq and pos recovery on batched predictions.

    Arguments:
        pred_pos (torch.tensor): Predicted position vector of shape [L] for the full batch.
        true_pos (torch.tensor): True position vector of shape [L] for the full batch.
        pred_seq (torch.tensor): Predicted seq vector of shape [L] for the full batch.
        true_seq (torch.tensor): True seq vector of shape [L] for the full batch.
        aatype_batch (torch.tensor): Batch indexing vector of shape [L] for scatter ops.

    Returns:
        torch.tensor: pos recovery of each sample shaped [B].
        torch.tensor: seq recovery of each sample shaped [B].
    """
    # Check that vectors are comparable
    assert (
        pred_pos.numel()
        == true_pos.numel()
        == pred_seq.numel()
        == true_seq.numel()
        == aatype_batch.numel()
    )
    assert pred_pos.sum() == true_pos.sum()
    assert (pred_seq != rc.restype_num).sum() == (true_seq != rc.restype_num).sum()

    # Calculate pos recovery per sample
    mask = true_pos.clone()
    pos_rec_per_samp = scatter(
        pred_pos[mask] == true_pos[mask], aatype_batch[mask], dim=0, reduce="sum"
    )  # [B]
    pos_per_samp = scatter(pred_pos, aatype_batch, dim=0, reduce="sum")  # [B]
    pos_rec = pos_rec_per_samp / (pos_per_samp + 1e-8)  # [B]

    # Calculate seq recovery per sample
    rec_mask = (pred_seq[mask] == true_seq[mask]) * (pred_pos[mask] == true_pos[mask])
    seq_rec_per_samp = scatter(rec_mask, aatype_batch[mask], dim=0, reduce="sum")  # [B]
    seq_rec = seq_rec_per_samp / (pos_per_samp + 1e-8)  # [B]

    return pos_rec, seq_rec


def initialize_rosetta(mode: str = "fast") -> None:
    """
    Initialize Rosetta for packing/scoring use.

    Arguments:
        mode (str): Packing 'mode' to use. If 'slow', will add -ex4 rotamers. Default is 'fast'.
    """
    ros_opts = [
        "-mute all",
        "-ignore_unrecognized_res 1",
        "-ex1 -ex1aro -ex1aro_exposed",
        " -ex2 -ex2aro -ex2aro_exposed",
        "-hbond_disable_bbsc_exclusion_rule True",
        "-pack_missing_sidechains False",
        "-fast_restyping True",
        "-optH_MCA False -flip_HNQ True -no_optH False",
        "-ex3",
    ]
    if mode == "slow":
        ros_opts.extend(["-ex4"])
    pyrosetta.init(" ".join(ros_opts))


def crop_around_network(p: Protein, net_pos: np.ndarray, topk: int = 32) -> Tuple[Protein, np.ndarray]:
    """
    Crops a Protein based on the following logic:
    - Find the centroid of the specified residues (based on Cb atoms)
    - Calculate the topK nearest neighbors of the centroid
    - Crop, keeping the union of the network residues and topK neighbors

    Arguments:
        p (Protein): Input Protein scaffold.
        net_pos (np.ndarray): Array of network residue positions.
        topk (int): How many centroid neighbors to keep. Default is 32.
    
    Returns:
        Protein: Cropped protein output.
        np.ndarray: Array of network + neighbor indices for masking use.
    """
    # Calculate net centroid
    bb_xyz = p.atom27_xyz[:, :4]
    Cb_xyz = impute_CB(bb_xyz[:, 0], bb_xyz[:, 1], bb_xyz[:, 2])  # [L, 3]
    net_centroid = np.mean(Cb_xyz[net_pos], axis=0)[None, :]  # [1, 3]

    # Extract topK nearest residues
    topk = min(topk, p.n_res - 1)
    Cb_dists = np.squeeze(cdist(Cb_xyz, net_centroid), -1)  # [L, 1]
    knn = np.argpartition(Cb_dists, topk)[:topk]

    # Add net res to this subset
    knn = np.concatenate([knn, net_pos])
    knn = np.unique(knn)

    # Sort to avoid scrambling residue order
    knn = np.sort(knn)

    # Crop scaffold down to just this region
    return p.mask(knn), knn


def crop_by_distance(p: Protein, net_pos: np.ndarray, d: float = 5.0) -> Tuple[Protein, np.ndarray]:
    """
    Crop by keeping only residues within a certain distance of the network residues.

    Arguments:
        p (Protein): Input Protein scaffold.
        net_pos (np.ndarray): Array of network positions for indexing.
        d (float): Cb distance cutoff, in Angstrom. Defaults is 5.0.

    Returns:
        Protein: Cropped Protein output.
        np.ndarray: Array of positions used in the crop.
    """

    # Crop down to only residues within a certain distance.
    p.set_neighbor_mask(d)
    net_neighbors = np.sum(p.neighbor_mask[net_pos, :], axis=0) > 0 # [L]
    net_neighbors = np.where(net_neighbors)[0]

    # Crop scaffold down to just this region
    return p.mask(net_neighbors), net_neighbors