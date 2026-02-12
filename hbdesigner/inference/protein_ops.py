import re
from itertools import combinations
from typing import List, Tuple, Union
from copy import deepcopy
import numpy as np
import pandas as pd
import pyrosetta

from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.select.residue_selector import LayerSelector
from scipy.spatial.distance import cdist

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import build_sc_from_chi, calc_sc_dihedrals
from hbdesigner.data.protein import PDB_CHAIN_IDS, Protein


def symmetrize_output(
    row: int,
    df: pd.DataFrame,
    symm_mask: np.ndarray,
    clash_thresh: float = 3.0,
) -> pd.core.series.Series:
    """
    Symmetrizes a packed network using the provided symmetry mask and network information.
    Operates on a single row of an inference results DataFrame.

    Arguments:
        row (int): The index of the row in the DataFrame to symmetrize.
        df (pd.DataFrame): The DataFrame containing network information.
        symm_mask (np.ndarray): A binary mask indicating symmetry mates.
        clash_thresh (float): The distance threshold, in Angstrom, for detecting clashes.

    Returns:
        pd.core.series.Series: The updated DataFrame row with the symmetrized network.
    """
    net = df.iloc[row]

    # Collect network res
    p = deepcopy(net.protein)
    net_mask = p.aatype != rc.restype_order["G"]
    net_res = np.where(net_mask)[0]
    n_res_asym = net_res.size

    for nr in net_res:
        # Copy aatype to symmetry-mates
        symm_mates = np.where(symm_mask[nr])[0]
        p.aatype[symm_mates] = p.aatype[nr]

        # Collect network chi angles
        sc_dihedrals, sc_dihedral_mask = calc_sc_dihedrals(
            p.atom27_xyz[nr], np.array(p.aatype[nr]), return_mask=True
        )
        sc_dihedrals = np.tile(sc_dihedrals, (symm_mates.size, 1))
        sc_dihedral_mask = np.tile(sc_dihedral_mask, (symm_mates.size, 1))

        # Apply chi angles to symmetry mates
        bb_xyz = p.atom27_xyz[symm_mates, :4]
        sc_xyz, sc_mask = build_sc_from_chi(
            bb_xyz, p.aatype[symm_mates], sc_dihedrals, sc_dihedral_mask
        )
        p.atom27_xyz[symm_mates, :14], p.atom27_mask[symm_mates, :14] = sc_xyz, sc_mask

    # Import back into Rosetta to optimize missing hydrogens
    p.atom27_xyz[:, 14:] = 0.0
    p.atom27_mask[:, 14:] = 0.0
    pose = Pose()
    pdbstr = p.to_pdb(unk_to_gly=True)
    pose_from_pdbstring(pose, pdbstr)

    buffer = pyrosetta.rosetta.std.stringbuf()
    pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
    pdb_block = buffer.str()
    p = Protein.from_pdb_string(pdb_block, discard_Hs=False)

    # Check for clashes between symmetry-mates
    symm_mate_res = np.where(p.aatype != rc.restype_order["G"])[0]
    symm_mate_res = np.setdiff1d(symm_mate_res, net_res)
    
    # If network picks overlapping residues on different chains, it can fail to symmetrize
    if (n_res_asym != symm_mate_res.size):
        print("Found quasi-symmetric network before symmetrization. Returning un-symmetrized network.")
        return None

    net_res_xyz = np.reshape(p.atom27_xyz[net_res, 4:14], (-1, 3))  # [N, 3]
    symm_mate_xyz = np.reshape(p.atom27_xyz[symm_mate_res, 4:14], (-1, 3))  # [N, 3]
    net_res_mask = np.reshape(p.atom27_mask[net_res, 4:14], (-1, 1))
    pair_dists = cdist(net_res_xyz, symm_mate_xyz)  # [N, N]
    pair_masks = net_res_mask * np.transpose(net_res_mask)  # [N, N]
    clashes = (pair_dists < clash_thresh) * pair_masks
    n_clashes = np.sum(clashes)

    if n_clashes > 0:
        print("Found clash while symmetrizing. Returning un-symmetrized network.")
        return None
    else:
        net.protein = p
        # Update metadata and scores to be symmetry-aware
        net.network = get_network_res(p)
        symm_factor = np.sum(symm_mask, axis=0)[0]
        for score in [
            "buried_heavy_unsats",
            "buried_unsat_Hpol",
            "HB_Score_full",
            "HB_Score_hb",
        ]:
            net[score] *= symm_factor
        return net


def concat_proteins(p_all: List[Protein], sort: bool = True) -> Protein:
    """
    Concatenate a List of Protein objects into a single Protein object.

    Arguments:
        p_all (List[Protein]): List of Protein objects to concatenate.
        sort (bool): If True, sorts the concatenated protein by chain index. Defaults to True.

    Returns:
        Protein: A single Protein object containing all concatenated data.
    """
    unique_chains = set()
    for p in p_all:
        for p_chain in np.unique(p.chain_index):
            if p_chain in unique_chains:
                raise ValueError(
                    f"Duplicate chain index {PDB_CHAIN_IDS[p_chain]} found across concatenated proteins. Please ensure unique chain indices for each protein."
                )
            else:
                unique_chains.add(p_chain)

    p = Protein(
        atom27_xyz=np.concatenate([p.atom27_xyz for p in p_all], axis=0),
        atom27_mask=np.concatenate([p.atom27_mask for p in p_all], axis=0),
        aatype=np.concatenate([p.aatype for p in p_all], axis=0),
        residue_index=np.concatenate([p.residue_index for p in p_all], axis=0),
        chain_index=np.concatenate([p.chain_index for p in p_all], axis=0),
        b_factors=np.concatenate([p.b_factors for p in p_all], axis=0),
    )
    # Sort by chain ID
    if sort:
        chain_order = np.argsort(p.chain_index, kind="stable")
        p = Protein(
            atom27_xyz=p.atom27_xyz[chain_order],
            atom27_mask=p.atom27_mask[chain_order],
            aatype=p.aatype[chain_order],
            residue_index=p.residue_index[chain_order],
            chain_index=p.chain_index[chain_order],
            b_factors=p.b_factors[chain_order],
        )
    return p


def get_network_res(p: Protein, non_net: str = "G") -> str:
    """
    Detects any non-GLY res in the protein and collects their PDB chain and res IDs.

    Arguments:
        p (Protein): The Protein object to get the network residues from.
        non_net (str): The residue type to exclude from the network. Defaults to "G" (glycine).

    Returns:
        str: A colon-separated string representation of the network residues.
    """

    # Update network resid metadata
    net_pos = np.where(p.aatype != rc.restype_order[non_net])[0]
    net_resid = p.residue_index[net_pos]
    net_chainid = p.chain_index[net_pos]
    net_aatype = p.aatype[net_pos]

    net_string = []
    for chainid, resid, aatype in zip(net_chainid, net_resid, net_aatype):
        chain = PDB_CHAIN_IDS[chainid]
        net_string += [f"{chain}{resid}{rc.restypes[aatype]}"]

    return ":".join(sorted(net_string))


def add_guide_atom(p: Protein, guide_atom_xyz: np.ndarray) -> Protein:
    """
    Adds a guide atom to the Protein as a HETATM entry.

    Arguments:
        p (Protein): The Protein object to which the guide atom will be added.
        guide_atom_xyz (np.ndarray): The XYZ coordinates of the guide atom, shape [3,].

    Returns:
        Protein: The updated Protein object.
    """
    p.hetatm_dict = {
        "atom_name": np.array(["V1"]),
        "element": np.array(["V"]),
        "res_name": np.array(["ORI"]),
        "residue_index": np.array([p.residue_index.max() + 1]),
        "chain_index": np.array([p.chain_index.max() + 1]),
        "atom_xyz": guide_atom_xyz,
    }
    return p


def get_core_mask(scaffold: Protein, core_cutoff: float = 5.2) -> np.ndarray:
    """
    Calculate mask of core residues using Rosetta's LayerSelector.

    Args:
        scaffold (Protein): The protein object containing the residues.
        core_cutoff (float): The layer cutoff value for core residues. Defaults to 5.2 (typical for HBNet).

    Returns:
        np.ndarray: A binary mask of shape [L,] where L is the number of residues in the protein.
                    The mask has ones for core residues and zeros elsewhere.
    """
    pose = Pose()
    pose_from_pdbstring(pose, scaffold.to_pdb(unk_to_gly=True))
    core_sel = LayerSelector()
    core_sel.set_cutoffs(core=core_cutoff, surf=2.0)
    core_sel.set_layers(pick_core=True, pick_boundary=False, pick_surface=False)
    return np.array(core_sel.apply(pose))


def validate_residues(p: Protein, residues: str = None, mode: str = "guide") -> Union[np.ndarray, None]:
    """
    Validates and converts string-formatted residues into np array of positions.
    Also converts from PDB chain/res numbering to absolute numbering.

    Arguments:
        p (Protein): The Protein object to validate the guide res against.
        residues (str): A comma-separated string of guide residues in PDB format, e.g., 'A12,B45'.
        mode (str): The mode of validation. Options are 'guide' and 'fixed'. ".
    Returns:
        np.ndarray: An array of absolute residue indices corresponding to the specified residues.
                    Returns None if residues is None.
    """
    if residues is None:
        if mode == "fixed":
            return np.empty((0,), dtype=np.int64)
        else:
            return None

    residues = residues.split(",")
    abs_res = []
    for res in residues:
        pdb_ch, pdb_num = [item for item in re.split("(\\d+)", res) if item]
        matches = np.where(
            (p.residue_index == int(pdb_num))
            * (p.chain_index == PDB_CHAIN_IDS.index(pdb_ch))
        )[0]
        if matches.size != 1:
            raise ValueError(
                f"Parsing error! {matches.size} matches found for res {res}."
            )
        abs_res.append(matches[0])

    if (mode == "guide") and (len(abs_res) < 2):
        raise ValueError(
            f"Only {len(abs_res)} provided. You must provide at least 2 guide res for centroid calculation."
        )
    elif mode == "fixed":
        abs_res = np.array(abs_res)
        fixed_res_aatypes = p.aatype[abs_res]
        invalid = fixed_res_aatypes[:, None] == rc.restype_non_hb_idx[None, :]
        resnames = [rc.restypes_with_x[idx] for idx in fixed_res_aatypes]
        description = []
        for rn, ri in zip(resnames, residues):
            description.append(ri + "->" + rn)
        description = " : ".join(description)
        if np.sum(invalid) > 0.0:
            raise ValueError(
                f"Fixed residues contain hydrophobic restypes which are not allowed in HBDesigner networks:\t{description} ")

    return np.array(abs_res)


def validate_chains(p: Protein, chains: str = None) -> Union[np.ndarray, None]:
    """
    Validates and converts a comma-separated string of chain IDs into a design mask.

    Arguments:
        p (Protein): The Protein object to validate the guide res against.
        chains (str): A comma-separated string of masked chains in PDB format, e.g., 'A,B,D'.
    Returns:
        np.ndarray: A binary mask of designable positions of shape (p.n_res, ). True where designable, False where not.
    """
    if chains is None:
        return np.ones_like(p.aatype, dtype=bool)

    chains = chains.split(",")
    try:
        chain_ids = [PDB_CHAIN_IDS.index(ch) for ch in chains]
    except ValueError as e:
        raise ValueError(f"Invalid chain ID in input: {e}. Must be one of {PDB_CHAIN_IDS}")
    
    for cid in chain_ids:
        if cid not in p.chain_index:
            raise ValueError(
                f"Selected chain {PDB_CHAIN_IDS[cid]} not in provided PDB with chains {[PDB_CHAIN_IDS[c] for c in np.unique(p.chain_index)]}"
            )

    # True if not in requested chain IDs
    design_mask = np.isin(p.chain_index, chain_ids, invert=True)

    # False if already noted as a fixed residue
    fixed_res_mask = p.aatype != rc.restype_num
    design_mask[fixed_res_mask] = False
    return design_mask


def get_symmetry_mask(p: Protein, symm_chains: str = None) -> np.ndarray:
    """
    Calculate binary symmetry mask based on specified chain symmetries.

    Args:
        p (Protein): The protein object containing the chains.
        symm_chains (str): Semi-colon separated string of comma-separated chain IDs to symmetrize, e.g., 'A,B;C,D'.

    Returns:
        np.ndarray: A binary mask of shape [L, L] where L is the number of residues in the protein.
        The mask has ones for symmetric pairs and zeros elsewhere.
        If no symmetry is specified, returns an identity matrix.

    """
    # Ones mark symmetric pairs
    symmetry_mask = np.eye(p.n_res, p.n_res, dtype=np.float32)  # [L, L]

    # Collect positions for each chain
    chain_pos = {}  # {A: [1, 2, 3,], B: [1, 2, 3]}
    for c in np.unique(p.chain_index):
        chain_pos[PDB_CHAIN_IDS[c]] = np.where(p.chain_index == c)[0]

    n_symm = None
    if symm_chains is not None:
        # Split into different symmetry params
        symm_chains = symm_chains.split(";")
        for sc in symm_chains:
            # For each symmetry, collect positions for each chain
            chains = sc.split(",")
            if n_symm is None:
                n_symm = len(chains)
            else:
                assert len(chains) == n_symm, (
                    "ERROR: Cannot specify two different symmetries in one run!"
                )
            chain_combos = combinations(chains, 2)
            for cc in chain_combos:
                if chain_pos[cc[0]].size != chain_pos[cc[1]].size:
                    raise ValueError(
                        f"ERROR: Chains {cc[0]} and {cc[1]} are different sizes, so they can't be symmetrized!"
                    )
                symmetry_mask[chain_pos[cc[1]], chain_pos[cc[0]]] = 1.0
                symmetry_mask[chain_pos[cc[0]], chain_pos[cc[1]]] = 1.0

    # Check that full complex is symmetrized
    symm_factor = np.sum(symmetry_mask, axis=-1)
    if np.unique(symm_factor).size > 1:
        raise ValueError(
            "ERROR: Cannot do partial symmetry, must include full complex."
        )
    return symmetry_mask


def extract_chains(p: Protein, sel_chains: str = "") -> Tuple[Protein, Protein]:
    """
    Extracts the specified chains from the protein.

    Args:
        p (Protein): The protein object containing the chains.
        sel_chains (str): Comma-separated string of chain IDs to extract, e.g., 'A,B,C'.

    Returns:
        Tuple[Protein, Protein]: A tuple containing:
            1) the protein with only the extracted chains and
            2) the protein with only the non-extracted chains.
    """    
    p_chains = np.unique(p.chain_index)
    p_chains_str = [PDB_CHAIN_IDS[p_ch] for p_ch in p_chains]
    # Get numeric value for each sel chain
    sel_chains = sel_chains.split(",")
    sel_chains = [sc.strip() for sc in sel_chains if sc.strip() != '']
    p_chain_sel = []
    for s_ch in sel_chains:
        s_ch_idx = PDB_CHAIN_IDS.index(s_ch)
        if s_ch_idx not in p_chains:
            raise ValueError(
                f"Selected chain {s_ch} not in provided PDB with chains {p_chains_str}"
            )
        else:
            p_chain_sel.append(s_ch_idx)

    # Split protein into used/unused chains
    p_chain_sel = np.array(p_chain_sel)
    chain_mask = np.sum(p_chain_sel[:, None] == p.chain_index[None, :], axis=0) > 0
    p_used = deepcopy(p).mask(np.where(chain_mask)[0])
    p_unused = deepcopy(p).mask(np.where(~chain_mask)[0])
    return p_used, p_unused
