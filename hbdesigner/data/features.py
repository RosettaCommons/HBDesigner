import copy
from typing import Union, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

import hbdesigner.data.residue_constants as rc
import hbdesigner.data.rigid_utils as ru


Array = Union[np.ndarray, torch.Tensor]


def robust_norm(
    array: Array, axis: int = -1, p: float = 2.0, eps: float = 1e-8
) -> Array:
    """Computes robust p-norm of vectors.

    Args:
        array (Array): Array containing vectors to compute norm.
        axis (int, optional): Axis of array to norm. Defaults to -1.
        p (float, optional): p value for the p-norm to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norm computation. Defaults to 1e-8.

    Returns:
        Array: Norm of axis of array
    """
    assert p >= 1, "p must be greater than or equal to 1"

    if isinstance(array, np.ndarray):
        return (np.sum(np.abs(array) ** p, axis=axis) + eps) ** (1 / p)
    else:
        return (torch.sum(torch.abs(array) ** p, dim=axis) + eps) ** (1 / p)


def robust_normalize(
    array: Array, axis: int = -1, p: float = 2.0, eps: float = 1e-8
) -> Array:
    """Computes robust p-normalization of vectors.

    Args:
        array (Array): Array containing vectors to normalize.
        axis (int, optional): Axis of array to normalize. Defaults to -1.
        p (float, optional): p value for the p-norm to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norma computation. Defaults to 1e-8.

    Returns:
        Array: Normalized array
    """
    assert p >= 1, "p must be greater than or equal to 1"

    if isinstance(array, np.ndarray):
        return array / np.expand_dims(
            robust_norm(array, axis=axis, p=p, eps=eps), axis=axis
        )
    else:
        return array / robust_norm(array, axis=axis, p=p, eps=eps).unsqueeze(axis)


def calc_dihedrals(atom_positions: Array, eps: float = 1e-8) -> Array:
    """Calculates dihedral angles from "window-ed" atom positions.

    Args:
        atom_positions (Array): Array of atom positions with shape: (..., n_atoms, 3).
        eps (float, optional): Epsilon for robust norm computation. Defaults to 1e-8.

    Returns:
        Array: Dihedral angles with shape: (..., n_atoms - 3).
    """

    # Unit vectors
    uvecs = robust_normalize(
        atom_positions[..., 1:, :] - atom_positions[..., :-1, :], eps=eps
    )
    uvec_2 = uvecs[..., :-2, :]
    uvec_1 = uvecs[..., 1:-1, :]
    uvec_0 = uvecs[..., 2:, :]

    # Normals
    if isinstance(atom_positions, np.ndarray):
        nvec_2 = robust_normalize(np.cross(uvec_2, uvec_1, axis=-1), eps=eps)
        nvec_1 = robust_normalize(np.cross(uvec_1, uvec_0, axis=-1), eps=eps)

        # Angle between normals
        cos_dihedral = np.sum(nvec_2 * nvec_1, axis=-1)
        cos_dihedral = np.clip(cos_dihedral, -1 + eps, 1 - eps)
        dihedral = np.sign(np.sum(uvec_2 * nvec_1, axis=-1)) * np.arccos(cos_dihedral)
    else:
        nvec_2 = robust_normalize(torch.cross(uvec_2, uvec_1, dim=-1), eps=eps)
        nvec_1 = robust_normalize(torch.cross(uvec_1, uvec_0, dim=-1), eps=eps)

        # Angle between normals
        cos_dihedral = torch.sum(nvec_2 * nvec_1, dim=-1)
        cos_dihedral = torch.clamp(cos_dihedral, -1 + eps, 1 - eps)
        dihedral = torch.sign(torch.sum(uvec_2 * nvec_1, dim=-1)) * torch.acos(
            cos_dihedral
        )

    return dihedral


def calc_bb_dihedrals(
    atom_positions: Array,
    residue_index: Optional[Array] = None,
    use_pre_omega: bool = True,
    return_mask: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Calculates backbone dihedral angles from atom positions.

    Args:
        atom_positions (Array): Array of atom positions with shape: (n_atoms, 3).
        residue_index (Optional[Array], optional): Array of residue indices.
            Defaults to None.
        use_pre_omega (bool, optional): Compute "pre-omega" dihedral (like AF2)
            instead of omega. Defaults to True.
        return_mask (bool, optional): Return mask for dihedrals. Defaults to
            True.

    Returns:
        Union[Array, Tuple[Array, Array]]: Backbone dihedral angles with shape:
            (n_residues, 3). If return_mask, also returns mask with shape:
            (n_residues, 3)
    """

    # Get backbone coordinates (and reshape). First 3 coordinates are N, CA, C
    bb_atom_positions = atom_positions[:, :3].reshape((3 * atom_positions.shape[0], 3))

    # Get backbone dihedrals
    bb_dihedrals = calc_dihedrals(bb_atom_positions)
    if isinstance(atom_positions, np.ndarray):
        bb_dihedrals = np.pad(
            bb_dihedrals, [(1, 2)], constant_values=0.0
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))

        # Get dihedral mask based on existance of atoms
        bb_atom_dihedral_mask = [
            np.sum(np.prod(bb_atom_positions[i : i + 4] == 0.0, axis=-1)) == 0.0
            for i in range(len(bb_atom_positions) - 3)
        ]
        bb_atom_dihedral_mask = np.array(bb_atom_dihedral_mask, dtype=np.float32)
        bb_atom_dihedral_mask = np.pad(
            bb_atom_dihedral_mask, [(1, 2)], constant_values=0.0
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_atom_dihedral_mask = bb_atom_dihedral_mask.reshape(
            (atom_positions.shape[0], 3)
        )

        # Get mask based on residue_index
        bb_dihedrals_mask = np.ones_like(bb_dihedrals)
        if residue_index is not None:
            assert type(atom_positions) is type(residue_index)
            pre_mask = np.concatenate(
                (
                    [0.0],
                    (residue_index[1:] - 1 == residue_index[:-1]).astype(np.float32),
                ),
                axis=-1,
            )
            post_mask = np.concatenate(
                (
                    (residue_index[:-1] + 1 == residue_index[1:]).astype(np.float32),
                    [0.0],
                ),
                axis=-1,
            )
            bb_dihedrals_mask = np.stack((pre_mask, post_mask, post_mask), axis=-1)

        if use_pre_omega:
            # Move omegas such that they're "pre-omegas" and reorder dihedrals
            bb_dihedrals[:, 2] = np.concatenate(([0.0], bb_dihedrals[:-1, 2]), axis=-1)
            bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
            bb_atom_dihedral_mask[:, 2] = np.concatenate(
                ([0.0], bb_atom_dihedral_mask[:-1, 2]), axis=-1
            )
            bb_atom_dihedral_mask = bb_atom_dihedral_mask[:, [2, 0, 1]]
            bb_dihedrals_mask[:, 2] = np.concatenate(
                ([0.0], bb_dihedrals_mask[:-1, 2]), axis=-1
            )
            bb_dihedrals_mask[:, [0, 1, 2]] = bb_dihedrals_mask[:, [2, 0, 1]]

    else:
        bb_dihedrals = F.pad(
            bb_dihedrals, [1, 2], value=0.0
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))

        # Get dihedral mask based on existance of atoms
        bb_atom_dihedral_mask = [
            torch.sum(torch.prod(bb_atom_positions[i : i + 4] == 0.0, dim=-1)) == 0.0
            for i in range(len(bb_atom_positions) - 3)
        ]
        bb_atom_dihedral_mask = torch.tensor(bb_atom_dihedral_mask).to(torch.float32)
        bb_atom_dihedral_mask = F.pad(
            bb_atom_dihedral_mask, [1, 2], value=0.0
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_atom_dihedral_mask = bb_atom_dihedral_mask.reshape(
            (atom_positions.shape[0], 3)
        )
        bb_atom_dihedral_mask = bb_atom_dihedral_mask.to(atom_positions.device)

        # Get mask based on residue_index
        bb_dihedrals_mask = torch.ones_like(bb_dihedrals)
        if residue_index is not None:
            assert type(atom_positions) is type(residue_index)
            pre_mask = torch.cat(
                (
                    torch.tensor([0.0], device=atom_positions.device),
                    (residue_index[1:] - 1 == residue_index[:-1]).to(torch.float32),
                ),
                dim=-1,
            )
            post_mask = torch.cat(
                (
                    (residue_index[:-1] + 1 == residue_index[1:]).to(torch.float32),
                    torch.tensor([0.0], device=atom_positions.device),
                ),
                dim=-1,
            )
            bb_dihedrals_mask = torch.stack((pre_mask, post_mask, post_mask), axis=-1)

        if use_pre_omega:
            # Move omegas such that they're "pre-omegas" and reorder dihedrals
            bb_dihedrals[:, 2] = torch.cat(
                (
                    torch.tensor([0.0], device=atom_positions.device),
                    bb_dihedrals[:-1, 2],
                ),
                dim=-1,
            )
            bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
            bb_atom_dihedral_mask[:, 2] = torch.cat(
                (
                    torch.tensor([0.0], device=atom_positions.device),
                    bb_atom_dihedral_mask[:-1, 2],
                ),
                dim=-1,
            )
            bb_atom_dihedral_mask = bb_atom_dihedral_mask[:, [2, 0, 1]]
            bb_dihedrals_mask[:, 2] = torch.cat(
                (
                    torch.tensor([0.0], device=atom_positions.device),
                    bb_dihedrals_mask[:-1, 2],
                ),
                dim=-1,
            )
            bb_dihedrals_mask[:, [0, 1, 2]] = bb_dihedrals_mask[:, [2, 0, 1]]

    # Update dihedral_mask and dihedrals
    bb_dihedrals_mask = bb_dihedrals_mask * bb_atom_dihedral_mask
    bb_dihedrals = bb_dihedrals * bb_dihedrals_mask

    if return_mask:
        return bb_dihedrals, bb_dihedrals_mask
    else:
        return bb_dihedrals


def calc_sc_dihedrals(
    atom_positions: Array, aatype: Array, return_mask: bool = True
) -> Union[Array, Tuple[Array, Array]]:
    """Calculates sidechain dihedral angles from atom positions.

    Args:
        atom_positions (Array): Array of atom positions with shape: (n_res,
            n_atoms, 3).
        aatype (Array): Array of amino acid types with shape: (n_res,).
        return_mask (bool, optional): Return mask for dihedrals. Defaults to
            True.

    Returns:
        Union[Array, Tuple[Array, Array]]: Sidechain dihedral angles with shape:
            (n_res, 3). If return_mask, also returns mask with shape: (n_res, 3).
    """

    # Make sure atom_positions and aatype are same class
    assert type(atom_positions) is type(aatype)

    # Get atom indicies for atoms that make up chi angles and chi mask
    if isinstance(atom_positions, np.ndarray):
        chi_atom_indices = np.array(rc.chi_atom_indices_atom14, dtype=np.int32)[aatype]
        chi_mask = np.array(rc.chi_mask_atom14, dtype=np.float32)[aatype]

        # Get coordinates for chi atoms
        chi_atom_positions = np.take_along_axis(
            atom_positions,
            chi_atom_indices[..., None].repeat(3, axis=-1),
            axis=-2,
        )
    else:
        chi_atom_indices = torch.from_numpy(
            np.array(rc.chi_atom_indices_atom14, dtype=np.int32)
        ).to(aatype.device)[aatype]
        chi_mask = torch.from_numpy(np.array(rc.chi_mask_atom14, dtype=np.float32)).to(
            aatype.device
        )[aatype]

        # Get coordinates for chi atoms
        chi_atom_positions = torch.gather(
            atom_positions,
            -2,
            chi_atom_indices[..., None].expand(*chi_atom_indices.shape, 3).long(),
        )

    sc_dihedrals = calc_dihedrals(chi_atom_positions)

    # Mask dihedrals
    if isinstance(atom_positions, np.ndarray):
        # Get dihedral mask based on existance of atoms
        sc_atom_dihedral_mask = [
            np.sum(
                np.prod(chi_atom_positions[..., i : i + 4, :] == 0.0, axis=-1),
                axis=-1,
            )
            == 0.0
            for i in range(chi_atom_positions.shape[-2] - 3)
        ]
        sc_atom_dihedral_mask = np.array(sc_atom_dihedral_mask, dtype=np.float32)
        sc_atom_dihedral_mask = sc_atom_dihedral_mask.T

        # Update dihedral_mask and dihedrals
        sc_dihedrals_mask = chi_mask * sc_atom_dihedral_mask
        sc_dihedrals = sc_dihedrals * sc_dihedrals_mask
    else:
        # Get dihedral mask based on existance of atoms
        sc_atom_dihedral_mask = torch.stack(
            [
                torch.sum(
                    torch.prod(chi_atom_positions[..., i : i + 4, :] == 0.0, dim=-1),
                    dim=-1,
                )
                == 0.0
                for i in range(chi_atom_positions.shape[-2] - 3)
            ],
            dim=-1,
        )
        sc_atom_dihedral_mask = sc_atom_dihedral_mask.to(torch.float32)

        # Update dihedral_mask and dihedrals
        sc_dihedrals_mask = chi_mask * sc_atom_dihedral_mask
        sc_dihedrals = sc_dihedrals * sc_dihedrals_mask

    if return_mask:
        return sc_dihedrals, sc_dihedrals_mask
    else:
        return sc_dihedrals


def impute_CB(N_xyz: Array, CA_xyz: Array, C_xyz: Array) -> Array:
    """Imputes CB coordinates from N, CA, and C coordinates.

    Args:
        N_xyz (Array): Coordinates of N atom with shape: (..., 3).
        CA_xyz (Array): Coordinates of CA atom with shape: (..., 3).
        C_xyz (Array): Coordinates of C atom with shape: (..., 3).

    Returns:
        Array: Imputed CB coordinates with shape: (..., 3).
    """

    # Make sure N_xyz, CA_xyz, and C_xyz are same class
    assert type(N_xyz) is type(CA_xyz) is type(C_xyz)

    # Calculate a, b, c orientation vectors
    b = CA_xyz - N_xyz
    c = C_xyz - CA_xyz
    if isinstance(N_xyz, np.ndarray):
        a = np.cross(b, c, axis=-1)
    else:
        a = torch.cross(b, c, dim=-1)

    # Calculate CB coordinates
    CB_xyz = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA_xyz

    return CB_xyz


def build_sc_from_chi(
    bb_xyz: Array, aatype: Array, chi_angles: Array, chi_angle_mask: Array
) -> Tuple[Array, Array]:
    """Build side chain atoms from backbone atoms and chi angles.

    Args:
        bb_xyz (Array): 3D coordinates of the backbone atoms, shape: (Nres, 4, 3).
        aatype (Array): Amino acid type, shape (Nres,).
        chi_angles (Array): Chi angles in radians, shape: (Nres, 4).
        chi_angle_mask (Array): Mask of which chi angles are present, shape: (Nres, 4).

    Returns:
        Tuple[Array, Array]: Tuple containing 3D coordinates of each residue's atoms, shape: (Nres, 14, 3), and a mask of which atoms are present, shape: (Nres, 14).
    """
    # Make sure bb_xyz, chi_angles, and chi_angle_mask are same class
    assert type(bb_xyz) is type(aatype) is type(chi_angles) is type(chi_angle_mask)

    # Make sure the shapes are expected
    n_res = bb_xyz.shape[0]
    assert bb_xyz.shape == (n_res, 4, 3)
    assert aatype.shape == (n_res,)
    assert chi_angles.shape == (n_res, 4)
    assert chi_angle_mask.shape == (n_res, 4)

    if isinstance(bb_xyz, np.ndarray):
        # For ease, if using numpy, we convert arrays to tensors and then back.
        is_numpy = True
        bb_xyz = torch.from_numpy(bb_xyz)
        chi_angles = torch.from_numpy(chi_angles)
        chi_angle_mask = torch.from_numpy(chi_angle_mask)
    else:
        is_numpy = False

    # Convert chi_angles to sine and cosine.
    chi_angles = torch.stack(
        [torch.sin(chi_angles), torch.cos(chi_angles)],
        dim=-1,
    )

    # Get the default transformations for the chis
    default_4x4 = torch.from_numpy(rc.restype_rigid_group_default_frame[aatype])[:, -4:]
    default_r = ru.Rigid.from_tensor_4x4(default_4x4)

    # Construct and apply updates to the defaults based on chi values
    chi_rots = torch.zeros(default_r.get_rots().get_rot_mats().shape)
    chi_rots[..., 0, 0] = 1
    chi_rots[..., 1, 1] = chi_angles[..., 1]
    chi_rots[..., 1, 2] = -chi_angles[..., 0]
    chi_rots[..., 2, 1:] = chi_angles
    chi_rots = ru.Rigid(ru.Rotation(rot_mats=chi_rots), None)
    chi_frames = default_r.compose(chi_rots)

    # Build transforms for each chi directly from the bb frame
    chi2_frame_to_frame = chi_frames[:, 1]
    chi3_frame_to_frame = chi_frames[:, 2]
    chi4_frame_to_frame = chi_frames[:, 3]

    chi1_frame_to_bb = chi_frames[:, 0]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = ru.Rigid.cat(
        [
            chi1_frame_to_bb.unsqueeze(-1),
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    # Build backbone frame and transform chi frames to global frame.
    bb_frames = ru.Rigid.from_3_points(bb_xyz[:, 0], bb_xyz[:, 1], bb_xyz[:, 2])
    chi_frames_to_global = bb_frames[..., None].compose(all_frames_to_bb)

    # Construct group mask for assigning atoms to chi groups.
    atom14_group_mask = torch.clamp(
        torch.from_numpy(rc.restype_atom14_to_rigid_group[aatype])[:, 5:] - 4, min=0
    )
    atom14_group_mask_oh = nn.functional.one_hot(atom14_group_mask, num_classes=4)

    # Mask transformations appropriately for each atom.
    atoms_to_global = chi_frames_to_global[:, None] * atom14_group_mask_oh
    atoms_to_global = atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # Get the literature positions for each atom.
    lit_xyz = torch.from_numpy(rc.restype_atom14_rigid_group_positions[aatype])[:, 5:]

    # Apply transformations to lit positions to get final positions.
    xyz = atoms_to_global.apply(lit_xyz)

    # Create an appropriate atom mask.
    atom_mask = copy.deepcopy(torch.from_numpy(rc.restype_atom14_mask[aatype])[:, 5:])
    mask_mask = (chi_angle_mask[:, None] * atom14_group_mask_oh).sum(-1)
    atom_mask = mask_mask * atom_mask

    # Apply mask and construct final coordinates.
    xyz = xyz * atom_mask[..., None]
    xyz = torch.cat(
        [bb_xyz, impute_CB(bb_xyz[:, 0], bb_xyz[:, 1], bb_xyz[:, 2]).unsqueeze(1), xyz],
        dim=1,
    )
    atom_mask = torch.cat(
        [torch.from_numpy(rc.restype_atom14_mask[aatype])[:, :5], atom_mask], dim=1
    )
    xyz = xyz * atom_mask[..., None]

    if is_numpy:
        return xyz.numpy(), atom_mask.numpy()
    else:
        return xyz, atom_mask


def get_renamed_coords(atom14_xyz: Array, aatype: Array) -> Array:
    # Rename symmetric atoms
    if isinstance(atom14_xyz, np.ndarray):
        renamed_xyz = copy.deepcopy(atom14_xyz)
    else:
        renamed_xyz = atom14_xyz.clone()

    for restype in rc.residue_atom_renaming_swaps:
        # Get mask based on restype
        restype_idx = rc.restype_order[rc.restype_3to1[restype]]
        restype_mask = aatype == restype_idx

        # Swap atom coordinates for restype
        restype_xyz = renamed_xyz * restype_mask[..., None, None]
        for atom_pair in rc.residue_atom_renaming_swaps[restype]:
            atom1, atom2 = atom_pair
            atom1_idx = rc.restype_name_to_atom14_names[restype].index(atom1)
            atom2_idx = rc.restype_name_to_atom14_names[restype].index(atom2)
            restype_xyz[..., atom1_idx, :] = atom14_xyz[..., atom2_idx, :]
            restype_xyz[..., atom2_idx, :] = atom14_xyz[..., atom1_idx, :]

        # Update full tensor
        if isinstance(atom14_xyz, np.ndarray):
            restype_xyz = np.nan_to_num(restype_xyz) * restype_mask[..., None, None]
        else:
            restype_xyz = torch.nan_to_num(restype_xyz) * restype_mask[..., None, None]
        renamed_xyz = renamed_xyz * ~restype_mask[..., None, None] + restype_xyz

    return renamed_xyz


def masked_mean(
    a: Array,
    mask: Array,
    dim: Optional[Union[int, Tuple[int]]] = None,
    eps: float = 1e-8,
) -> Array:
    if isinstance(a, np.ndarray):
        return np.sum(a * mask, axis=dim) / (eps + np.sum(mask, axis=dim))
    else:
        return torch.sum(a * mask, dim=dim) / (eps + torch.sum(mask, dim=dim))


def scatter_masked_mean(
    a: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    eps: float = 1e-8,
) -> Array:
    mask = mask.expand(*a.shape)
    numer = scatter(a * mask, index, dim)
    denom = scatter(mask + eps, index, dim)
    return numer / denom


def sincos_to_angle(sincos_ang: Array) -> Array:
    """Convert array with sin/cos of an angle to that angle value in radians.

    Args:
        sincos_ang (Array): Array containing sin/cos angle values, [..., 2]

    Returns:
        Array: Angles corresponding to sincos_ang, [...]
    """
    if isinstance(sincos_ang, np.ndarray):
        p = np
    else:
        p = torch
    arcsin_ang = p.arcsin(sincos_ang[..., 0])
    arccos_ang = p.arccos(sincos_ang[..., 1])
    angle = p.where(arcsin_ang >= 0, arccos_ang, -arccos_ang)

    return angle


def normalize_chi(chi_unnorm: torch.Tensor) -> torch.Tensor:
    norm_denom = torch.sqrt(
        torch.clamp(torch.sum(chi_unnorm**2, dim=-1, keepdim=True), min=1e-12)
    )
    chi_norm = chi_unnorm / norm_denom
    return chi_norm
