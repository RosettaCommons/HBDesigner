from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_scatter import scatter, scatter_max
import numpy as np

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import (
    impute_CB, 
    scatter_masked_mean, 
    sincos_to_angle, 
    get_renamed_coords, 
    normalize_chi, 
    masked_mean,
)
from hbdesigner.model.layers import (
    MPNNLayer,
)
import hbdesigner.data.rigid_utils as ru
from hbdesigner.train.config import TrainConfig


class HBPacker(nn.Module):
    """
    Updated class for motif design (HBDesigner) using diffusion-like decoding and backbone-only encoding.

    Accepts sequence and guide atom conditioning, as well as total network size.

    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.model.hbdesigner
        self.max_comps = c.max_res
        self.knn_k = c.knn_k
        self.n_atoms = 14

        # Layers for nodes
        self.bb_dihedral_linear = nn.Linear(6, c.pn_dim)
        self.sc_dihedral_linear = nn.Linear(8, c.pn_dim)
        self.seq_embedder = nn.Embedding(rc.restype_num + 1, c.pn_dim)
        self.node_norm = nn.LayerNorm(c.pn_dim)

        # Layers for edges
        self.rbf_linear = nn.Linear(self.n_atoms**2 * c.num_rbf, c.pe_dim)
        self.seq_sep_linear = nn.Linear(2 * c.max_seq_sep + 2, c.pe_dim)
        self.edge_norm = nn.LayerNorm(c.pe_dim)

        # Message passing layers
        self.mpnn_layers = nn.ModuleList(
            [
                MPNNLayer(
                    c.pn_dim,
                    c.pn_dim,
                    c.pe_dim,
                    c.pe_dim,
                    num_mlp_layers=c.num_mlp_layers,
                    act="relu",
                    dropout=self.cfg.model.dropout,
                )
                for _ in range(c.num_protein_encoder_layers)
            ]
        )


        # Output layers
        self.sc_linear = nn.Linear(c.pn_dim * 2, 8)
        
        # Components for building sidechains
        self._restype_rigid_group_default_frame = None
        self._restype_atom14_to_rigid_group = None
        self._restype_atom14_rigid_group_positions = None
        self._restype_atom14_mask = None
        self._chi_angles_mask = None
        self._chi_pi_periodic = None

    @property
    def restype_rigid_group_default_frame(self) -> torch.Tensor:
        if self._restype_rigid_group_default_frame is None:
            self._restype_rigid_group_default_frame = (
                torch.from_numpy(rc.restype_rigid_group_default_frame)
                .to(torch.float32)
                .to(next(self.parameters()).device)
            )
        return self._restype_rigid_group_default_frame

    @property
    def restype_atom14_to_rigid_group(self) -> torch.Tensor:
        if self._restype_atom14_to_rigid_group is None:
            self._restype_atom14_to_rigid_group = (
                torch.from_numpy(rc.restype_atom14_to_rigid_group)
                .to(torch.long)
                .to(next(self.parameters()).device)
            )
        return self._restype_atom14_to_rigid_group

    @property
    def restype_atom14_rigid_group_positions(self) -> torch.Tensor:
        if self._restype_atom14_rigid_group_positions is None:
            self._restype_atom14_rigid_group_positions = (
                torch.from_numpy(rc.restype_atom14_rigid_group_positions)
                .to(torch.float32)
                .to(next(self.parameters()).device)
            )
        return self._restype_atom14_rigid_group_positions

    @property
    def restype_atom14_mask(self) -> torch.Tensor:
        if self._restype_atom14_mask is None:
            self._restype_atom14_mask = (
                torch.from_numpy(rc.restype_atom14_mask)
                .to(torch.float32)
                .to(next(self.parameters()).device)
            )
        return self._restype_atom14_mask

    @property
    def chi_angles_mask(self) -> torch.Tensor:
        if self._chi_angles_mask is None:
            self._chi_angles_mask = (
                torch.from_numpy(np.array(rc.chi_angles_mask))
                .to(torch.float32)
                .to(next(self.parameters()).device)
            )
        return self._chi_angles_mask

    @property
    def chi_pi_periodic(self) -> torch.Tensor:
        if self._chi_pi_periodic is None:
            self._chi_pi_periodic = (
                torch.from_numpy(np.array(rc.chi_pi_periodic))
                .to(torch.float32)
                .to(next(self.parameters()).device)
            )
        return self._chi_pi_periodic

    def _get_knn_edges(
        self, atom14_xyz: torch.Tensor, aatype_batch: torch.Tensor
    ) -> torch.Tensor:
        """Computes k-nearest neighbor graph for protein based on CA atom positions"""

        # Get CA atom positions
        ca_xyz = atom14_xyz[..., 1, :]

        # Compute k-nearest neighbor graph
        edge_index = gnn.knn_graph(ca_xyz, self.knn_k, batch=aatype_batch, loop=True)

        return edge_index

    def _form_protein_nodes(
        self,
        bb_dihedral: torch.Tensor,
        aatype: torch.Tensor,
        sc_dihedral: torch.Tensor,
    ) -> torch.Tensor:
        """Forms initial nodes for protein"""
        c = self.cfg.model.hbdesigner

        # Initialize nodes
        nodes = bb_dihedral.new_zeros((bb_dihedral.shape[0], c.pn_dim))

        # Sin-cos encoding of backbone dihedrals
        bb_dihedral_sincos = torch.stack(
            [torch.sin(bb_dihedral), torch.cos(bb_dihedral)], dim=-1
        ).view(-1, 6)
        nodes += self.bb_dihedral_linear(bb_dihedral_sincos)

        sc_dihedral_sincos = torch.stack(
            [torch.sin(sc_dihedral), torch.cos(sc_dihedral)], dim=-1
        ).view(-1, 8)
        nodes += self.sc_dihedral_linear(sc_dihedral_sincos)

        # Sequence embedding
        nodes += self.seq_embedder(aatype)  # [L, pn_dim]

        # Normalize initial node features
        nodes = self.node_norm(nodes)
        return nodes

    def _form_protein_edges(
        self,
        atom14_xyz: torch.Tensor,
        atom14_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forms initial edges for protein"""
        c = self.cfg.model.hbdesigner

        # Initialize edges
        edges = atom14_xyz.new_zeros((edge_index.shape[1], c.pe_dim))

        # RBF-encoded pairwise atomic distances
        atom_xyz = atom14_xyz.clone()
        atom_mask = atom14_mask.clone()

        atom_xyz[..., 4, :] = impute_CB(
            atom_xyz[..., 0, :], atom_xyz[..., 1, :], atom_xyz[..., 2, :]
        )
        atom_mask[..., 4] = torch.prod(atom_mask[..., :3], dim=-1)
        pair_dists = torch.cdist(atom_xyz[edge_index[0]], atom_xyz[edge_index[1]])
        pair_dists = pair_dists.view(
            edge_index.shape[1], -1, 1
        )  # (num_edges, 5 ** 2, 1)
        pair_mask = (
            atom_mask[edge_index[0], :, None] * atom_mask[edge_index[1], None, :]
        )  # (num_edges, 5, 5)
        pair_dists = pair_mask.reshape(edge_index.shape[1], -1, 1) * pair_dists
        rbf_mu = torch.linspace(0, 20, c.num_rbf).view(1, 1, -1)  # (1, 1, num_rbf)
        rbf_mu = rbf_mu.to(atom14_xyz.device)
        rbf_sigma = 20 / c.num_rbf
        rbf = torch.exp(-1 * (pair_dists - rbf_mu) ** 2 / rbf_sigma**2)
        rbf = rbf.view(edge_index.shape[1], -1)
        edges += self.rbf_linear(rbf)

        # Sequence separation
        # If on the same chain, use one-hot encoding of sequence separation (up to 32 residues away)
        # If on different chains, mask one-hot encoding and provide extra bit
        dij = residue_index[edge_index[0]] - residue_index[edge_index[1]]
        dij = torch.clamp(dij, -c.max_seq_sep, c.max_seq_sep) + c.max_seq_sep
        dij = F.one_hot(dij.long(), 2 * c.max_seq_sep + 1).float()
        mij = chain_index[edge_index[0]] == chain_index[edge_index[1]]
        mij = mij.float().view(-1, 1)
        dij = mij * dij
        seq_sep = torch.cat([dij, 1 - mij], dim=-1)
        edges += self.seq_sep_linear(seq_sep)

        # Normalize initial edge features
        edges = self.edge_norm(edges)

        return edges

    def get_atom14_xyz_from_chi(
        self,
        aatype: torch.Tensor,
        bb_xyz: torch.Tensor,
        chi_angles_sincos: torch.Tensor,
    ) -> torch.Tensor:
        chi_angles = chi_angles_sincos

        # Get the default transformations for the chis
        default_4x4 = self.restype_rigid_group_default_frame[aatype][:, -4:]
        default_r = ru.Rigid.from_tensor_4x4(default_4x4)

        # Construct and apply updates to the defaults based on chi values
        chi_rots = torch.zeros(default_r.get_rots().get_rot_mats().shape).to(
            aatype.device
        )
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
            self.restype_atom14_to_rigid_group[aatype][:, 5:] - 4, min=0
        )
        atom14_group_mask = torch.clamp(
            self.restype_atom14_to_rigid_group[aatype][:, 5:] - 4, min=0
        )
        atom14_group_mask_oh = nn.functional.one_hot(atom14_group_mask, num_classes=4)

        # Mask transformations appropriately for each atom.
        atoms_to_global = chi_frames_to_global[:, None] * atom14_group_mask_oh
        atoms_to_global = atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

        # Get the literature positions for each atom.
        lit_xyz = self.restype_atom14_rigid_group_positions[aatype][:, 5:]

        # Apply transformations to lit positions to get final positions.
        xyz = atoms_to_global.apply(lit_xyz)

        # Create an appropriate atom mask.
        atom_mask = self.restype_atom14_mask[aatype][:, 5:]
        mask_mask = (self.chi_angles_mask[aatype][:, None] * atom14_group_mask_oh).sum(
            -1
        )
        atom_mask = mask_mask * atom_mask

        # Apply mask and construct final coordinates.
        xyz = xyz * atom_mask[..., None]
        xyz = torch.cat(
            [
                bb_xyz,
                impute_CB(bb_xyz[:, 0], bb_xyz[:, 1], bb_xyz[:, 2]).unsqueeze(1),
                xyz,
            ],
            dim=1,
        )
        atom_mask = torch.cat(
            [self.restype_atom14_mask[aatype][:, :5], atom_mask], dim=1
        )
        xyz = xyz * atom_mask[..., None]

        return xyz, atom_mask

    def forward(self, b: gd.Batch) -> Dict[str, torch.Tensor]:

        c = self.cfg.model.hbdesigner
        if self.training and (c.num_recycles > 0):
            # Recycles are random during training
            n_cycles = torch.randint(
                0, c.num_recycles, (1,)
            ).item()
        else:
            n_cycles = c.num_recycles

        with torch.no_grad():
            for _ in range(n_cycles):
                # Run the forward pass
                results_dict = self._forward(b)

                # Update the batch values
                chi_mask = b.chi_nll_mask > 0
                b.atom14_xyz[chi_mask] = results_dict["pred_atom14_xyz"]
                b.sc_dihedral[chi_mask] = sincos_to_angle(
                    results_dict["pred_chis_norm"]
                )[chi_mask]  # [L, 4]
                b.sc_dihedral = b.sc_dihedral_mask_gt * b.sc_dihedral

        # Run the final iteration with gradients
        results_dict = self._forward(b)
        return results_dict

    def _forward(self, b: gd.Batch) -> Dict[str, torch.Tensor]:

        # Create initial embedding of protein nodes
        protein_nodes = self._form_protein_nodes(
            b.bb_dihedral,
            b.aatype,
            b.sc_dihedral,
        )

        # Get k-nearest neighbor graph for protein
        protein_edge_index = self._get_knn_edges(b.atom14_xyz, b.aatype_batch)

        # Create initial embedding of protein edges
        protein_edges = self._form_protein_edges(
            b.atom14_xyz,
            b.atom14_mask,
            b.residue_index,
            b.chain_index,
            protein_edge_index,
        )

        # Pass through MPNN layers
        for i, layer in enumerate(self.mpnn_layers):

            protein_nodes, protein_edges = layer(
                protein_nodes, protein_edges, protein_edge_index
            )

        # Make chi predictions with ground truth seq context
        seq_embed = self.seq_embedder(b.aatype_gt)

        protein_nodes_seq = torch.cat([protein_nodes, seq_embed], 1)
        chi_preds_unnorm = self.sc_linear(protein_nodes_seq).view(-1, 4, 2) # [L, 4, 2]
        chi_preds_norm = normalize_chi(chi_preds_unnorm)

        # Build sidechain coords now
        mask = b.chi_nll_mask.bool()
        aatype = b.aatype_gt[mask]
        bb_xyz = b.atom14_xyz[mask, :4]
        chi_angles_sincos = chi_preds_norm[mask]
        atom14_xyz, atom14_mask = self.get_atom14_xyz_from_chi(aatype, bb_xyz, chi_angles_sincos)
        results_dict = {
        "pred_chis_unnorm": chi_preds_unnorm,
        "pred_chis_norm": chi_preds_norm,
        "pred_atom14_xyz": atom14_xyz,
        "pred_atom14_mask": atom14_mask,
        }

        return results_dict

    def compute_chi_ae(
        self,
        pred_chi_rad: torch.Tensor,
        aatype: torch.Tensor,
        chi_mask: torch.Tensor,
        true_chi_rad: torch.Tensor,
    ) -> torch.Tensor:
        residue_type_one_hot = F.one_hot(aatype, rc.restype_num + 1)  # [Nres, 21]
        chi_pi_periodic = torch.einsum(
            "ij,jk->ik",
            residue_type_one_hot.to(pred_chi_rad.dtype),
            self.chi_pi_periodic,
        )  # [Nres, 4]

        # Determine alternative locations for pi-periodic chis
        alt_true_chi_rad = true_chi_rad.clone()
        alt_true_chi_rad[(alt_true_chi_rad * chi_pi_periodic) > 0] -= torch.pi
        alt_true_chi_rad[(alt_true_chi_rad * chi_pi_periodic) < 0] += torch.pi

        # Compute angle difference, accounting for periodicity
        chi_diff = true_chi_rad - pred_chi_rad  # [Nres, 4]
        chi_diff[chi_diff > torch.pi] = chi_diff[chi_diff > torch.pi] - 2 * torch.pi
        chi_diff[chi_diff < -torch.pi] = chi_diff[chi_diff < -torch.pi] + 2 * torch.pi

        # Compute alt angle difference, accounting for periodicity
        alt_chi_diff = alt_true_chi_rad - pred_chi_rad  # [Nres, 4]
        alt_chi_diff[alt_chi_diff > torch.pi] = (
            alt_chi_diff[alt_chi_diff > torch.pi] - 2 * torch.pi
        )
        alt_chi_diff[alt_chi_diff < -torch.pi] = (
            alt_chi_diff[alt_chi_diff < -torch.pi] + 2 * torch.pi
        )

        # Compute masked absolute error
        ae = torch.minimum(chi_diff.abs(), alt_chi_diff.abs())  # [Nres, 4]
        ae = chi_mask * ae

        return ae

    def compute_chi_mse(
        self,
        pred_norm_chi_sincos: torch.Tensor,
        pred_unnorm_chi_sincos: torch.Tensor,
        aatype: torch.Tensor,
        chi_mask: torch.Tensor,
        true_chi_sincos: torch.Tensor,
        eps: float = 1e-6,
        chi_angle_norm_weight: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute MSE of sin/cos chis

        Args:
            pred_norm_chi_sincos:
                [N, 4, 2] predicted angles
            pred_unnorm_chi_sincos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            chi_mask:
                [*, N, 4] angle mask
            true_chi_sincos:
                [*, N, 4, 2] ground truth angles
            eps:
                Small epsilon for numerical stability
            chi_angle_norm_weight:
                Weight for chi angle normalization penalty.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The per-residue chi MSE loss and per-residue chi norm loss.
        """

        residue_type_one_hot = F.one_hot(aatype, rc.restype_num + 1)  # [Nres, 21]
        chi_pi_periodic = torch.einsum(
            "ij,jk->ik",
            residue_type_one_hot.to(pred_norm_chi_sincos.dtype),
            self.chi_pi_periodic,
        )  # [Nres, 4]

        shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)  # [Nres, 4, 1]
        true_chi_shifted = shifted_mask * true_chi_sincos  # [Nres, 4, 2]
        sq_chi_error = torch.sum(
            (true_chi_sincos - pred_norm_chi_sincos) ** 2, dim=-1
        )  # [Nres, 4]
        sq_chi_error_shifted = torch.sum(
            (true_chi_shifted - pred_norm_chi_sincos) ** 2, dim=-1
        )  # [Nres, 4]
        sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)  # [Nres, 4]

        # Reweight MSE loss by chi abundance
        if self.cfg.model.hbdesigner.reweight_chi_mse:
            chi_count = torch.sum(chi_mask, dim=0) + 1 # [4]
            chi_count = 1. / (chi_count / (1e-10 + chi_count.sum()))
            chi_count = 6. * (chi_count / (1e-10 + chi_count.sum())) # Rescale to ~4 (native chi mask)
            sq_chi_error = (sq_chi_error * chi_count[None, :])
            sq_chi_error = torch.clamp(sq_chi_error, min=0., max=100.)

        # Chi MSE loss
        sq_chi_loss = masked_mean(sq_chi_error, chi_mask, -1)

        # Angle norm loss
        angle_norm = torch.sqrt(
            torch.sum(pred_unnorm_chi_sincos**2, dim=-1) + eps
        )  # [Nres, 4]
        norm_error = torch.abs(angle_norm - 1.0)  # [Nres, 4]
        angle_norm_loss = masked_mean(norm_error, chi_mask, -1)

        # Combine sq_chi_loss and angle_norm_loss
        chi_mse_loss = sq_chi_loss + chi_angle_norm_weight * angle_norm_loss
        return chi_mse_loss

    def compute_sc_msd(
        self,
        pred_atom14_xyz: torch.Tensor,
        true_atom14_xyz: torch.Tensor,
        atom14_mask: torch.Tensor,
        aatype: torch.Tensor,
    ) -> torch.Tensor:
        # Compute atom deviation based on original coordinates
        atom_deviation = torch.sum(
            torch.square(pred_atom14_xyz - true_atom14_xyz), dim=-1
        )

        # Compute atom deviation based on alternative coordinates
        true_renamed_xyz = get_renamed_coords(true_atom14_xyz, aatype)
        renamed_atom_deviation = torch.sum(
            torch.square(pred_atom14_xyz - true_renamed_xyz), dim=-1
        )

        # Get atom mask including backbone atoms
        atom_mask = atom14_mask.clone()
        atom_mask[..., :4] = 0.0

        # Compute MSD based on original and alternative coordinates
        # msd is the per-residue sidechain MSD
        msd_og = masked_mean(atom_deviation, atom_mask, -1)
        msd_renamed = masked_mean(renamed_atom_deviation, atom_mask, -1)
        msd = torch.minimum(msd_og, msd_renamed)
        return msd

    def compute_sc_clash_loss(
            self, 
            atom14_pred_positions: torch.Tensor,
            atom14_mask: torch.Tensor,
            aatype: torch.Tensor, 
            aatype_batch: torch.Tensor,
            chi_mask: torch.Tensor,
            clash_overlap_tolerance: float = 1.5, # OpenFold value is 1.5
            distance_threshold: float = 14., 
    ) -> torch.Tensor:
        """Uses VdW  radii to find clashes b/w heavy atoms.
        Note: ignores intra-residue clashes and backbone-backbone clashes."""

        # Get needed components from batch.
        restype_atom14_to_atom37 = []
        for rt in rc.restypes:
            atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
            restype_atom14_to_atom37.append(
                [(rc.atom_order[name] if name else 0) for name in atom_names]
            )
        restype_atom14_to_atom37.append([0] * 14)
        restype_atom14_to_atom37 = torch.tensor(
            restype_atom14_to_atom37, 
            dtype=torch.long, 
            device=aatype.device
        )
        residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]

        # Compute the Van der Waals radius for every atom
        # (the first letter of the atom name is the element type).
        # Shape: (*, N, 14).
        atomtype_radius = [
            rc.van_der_waals_radius[name[0]]
            for name in rc.atom_types
        ]
        atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
        atom14_atom_radius = (
            atom14_mask
            * atomtype_radius[residx_atom14_to_atom37]
        )
        
        # Get the basis atom xyz for each residue.
        # shape (*, N, 3)
        eps = 1e-10
        # Using Cb basis atom
        basis_atom_idx = 4 * torch.ones_like(aatype)
        basis_atom_idx[aatype == rc.restype_order["G"]] = 1

        basis_xyz = torch.gather(atom14_pred_positions, -2, basis_atom_idx[..., None, None].expand(*atom14_pred_positions.shape))[:, 0, :]

        # Determine distances based on basis atoms.
        # shape (*, N, N)
        basis_dists = torch.sqrt(
            eps
            + torch.sum(
                (basis_xyz[..., None, :, :] - basis_xyz[..., :, None, :]) ** 2, dim=-1
            )
        )

        # Create the mask for valid residue pairs.
        # shape (*, N, N)
        fp_type = atom14_pred_positions.dtype
        dists_mask = (
            aatype_batch[:, None] == aatype_batch[None, :]
        ).type(fp_type)

        # Mask out all the duplicate entries in the lower triangular matrix.
        # Also mask out the diagonal (same residue pairs)
        dists_mask = dists_mask * torch.triu(dists_mask, diagonal=1)

        # Determine which residues pairs contain at least one network residue.
        dists_mask = dists_mask * chi_mask[..., None]

        # Determine which residue pairs are within the distance threshold.
        # shape (*, N, N)
        dists_lower_bound = distance_threshold * torch.ones_like(dists_mask)
        dists_mask = dists_mask * (basis_dists < dists_lower_bound)
        valid_pairs = torch.where(dists_mask)

        # Get the atom14 coordinates for the valid residue pairs.
        # shape (N_pairs, 14, 3)
        res1_atom14_xyz = atom14_pred_positions[valid_pairs[0]]
        res2_atom14_xyz = atom14_pred_positions[valid_pairs[1]]

        # Get the atomic distances for the valid residue pairs.
        # shape (N_pairs, 14, 14)
        dists = torch.sqrt(
            eps
            + torch.sum(
                (res1_atom14_xyz[..., None, :] - res2_atom14_xyz[..., None, :, :]) ** 2, 
                dim=-1
            )
        )

        # Initialize the mask for the allowed distances.
        # shape (N_pairs, 14, 14)
        dists_mask = torch.ones_like(dists)

        # Backbone-backbone clashes are ignored. CB is included in the backbone.
        bb_bb_mask = torch.zeros_like(dists_mask)
        bb_bb_mask[..., :5, :5] = 1.0
        dists_mask = dists_mask * (1.0 - bb_bb_mask)    

        # Compute the lower bound for the allowed distances.
        # shape (N_pairs, 14, 14)
        dists_lower_bound = dists_mask * (
            atom14_atom_radius[valid_pairs[0]][..., :, None]
            + atom14_atom_radius[valid_pairs[1]][..., None, :]
        )

        # Compute the error.
        # shape (N_pairs, 14, 14)
        dists_to_low_error = dists_mask * F.relu(
            dists_lower_bound - clash_overlap_tolerance - dists
        )

        # Collect into per-sample loss
        dists_to_low_error = torch.sum(dists_to_low_error, dim=(-1, -2))
        clash_loss_batch = scatter(
            dists_to_low_error, 
            index=aatype_batch[valid_pairs[0]], 
            dim=0, 
            reduce="sum"
        )

        return clash_loss_batch

    def compute_sc_orient_loss(
            self, 
            pred_xyz: torch.Tensor,
            true_xyz: torch.Tensor,
            atom_mask: torch.Tensor,
            aatype_batch: torch.Tensor,
            aatype: torch.Tensor,
        ) -> torch.Tensor:
        """Calculates sc relative orientation loss using relative distances b/w the sidechains atoms in each network."""

        eps = 1e-10
        atom_mask[:, :5] = 0. # no backbone

        # Start mask w/res in the same sample
        pair_mask = (aatype_batch[:, None] == aatype_batch[None, :]).to(torch.float32) # [N, N]

        # Drop upper diagonal and self-interactions
        pair_mask = pair_mask * torch.triu(pair_mask, diagonal=1) # [N, N]

        # Collect tuple of residue pairs
        valid_pairs = torch.where(pair_mask) # [N, N]

        # Get the atom14 coordinates for the valid residue pairs.
        # shape (N_pairs, 14, 3)
        res1_atom14_xyz_pred = pred_xyz[valid_pairs[0]]
        res2_atom14_xyz_pred = pred_xyz[valid_pairs[1]]
        pred_dists = torch.sqrt(
            eps
            + torch.sum(
                (res1_atom14_xyz_pred[..., None, :] - res2_atom14_xyz_pred[..., None, :, :]) ** 2, 
                dim=-1
            )
        ) # [N, 14, 14]

        res1_atom14_xyz_true = true_xyz[valid_pairs[0]]
        res2_atom14_xyz_true = true_xyz[valid_pairs[1]]
        true_dists = torch.sqrt(
            eps
            + torch.sum(
                (res1_atom14_xyz_true[..., None, :] - res2_atom14_xyz_true[..., None, :, :]) ** 2, 
                dim=-1
            )
        ) # [N, 14, 14]

        dist_atom_mask = atom_mask[valid_pairs[0]][:, :, None] * atom_mask[valid_pairs[1]][:, None, :] # [N, 14, 14]
        orig_dev = (((pred_dists - true_dists) * dist_atom_mask) ** 2) # [N, 14, 14]

        true_renamed_xyz = get_renamed_coords(true_xyz, aatype)
        res1_atom14_xyz_true_renamed = true_renamed_xyz[valid_pairs[0]]
        res2_atom14_xyz_true_renamed = true_renamed_xyz[valid_pairs[1]]
        true_dists_renamed = torch.sqrt(
            eps
            + torch.sum(
                (res1_atom14_xyz_true_renamed[..., None, :] - res2_atom14_xyz_true_renamed[..., None, :, :]) ** 2, 
                dim=-1
            )
        ) # [N, 14, 14]
        # Get square deviation
        renamed_dev = (((pred_dists - true_dists_renamed) * dist_atom_mask) ** 2) # [N, 14, 14]
        min_dev = torch.min(orig_dev, renamed_dev) # [N, 14, 14]

        # Take masked atom-wise mean of each residue pair
        orient_error = masked_mean(min_dev, mask=dist_atom_mask, dim=(-1, -2)) # [N]
        orient_loss_batch = scatter(
            orient_error, 
            index=aatype_batch[valid_pairs[0]], 
            dim=0, 
            reduce="mean"
        )
        return orient_loss_batch

    @torch.no_grad()
    def _compute_other_metrics(self, b: gd.Batch, results_dict: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training metrics not involved in loss calculation."""
        
        metric_dict = {}

        mask = b.chi_nll_mask.bool()

        # Convert sin/cos angles to radian angles
        pred_chi_rad = sincos_to_angle(results_dict["pred_chis_norm"][mask])
        true_chi_rad = sincos_to_angle(b.chi_sincos_gt[mask])

        # Chi angle absolute error
        chi_mask = self.chi_angles_mask[b.aatype_gt[mask]]
        chi_ae = self.compute_chi_ae(
            pred_chi_rad=pred_chi_rad,
            aatype=b.aatype_gt[mask],
            chi_mask=chi_mask,
            true_chi_rad=true_chi_rad,
        )  # [L, 4]

        for i, chi_error in enumerate(torch.unbind(chi_ae, -1)):
            metric_dict[f"chi_mae_chi_{i + 1}_batch"] = chi_error[chi_mask[:, i].bool()]

        for i, mask_i in enumerate(torch.unbind(chi_mask, -1)):
            metric_dict[f"chi_mask_chi_{i + 1}_batch"] = mask_i

        # Rotamer recovery (within 20 degrees)
        # Compare the AE to the threshold to find correct chi
        rad_thresh = (20 / 180) * torch.pi
        correct_chi = chi_mask * (chi_ae - rad_thresh) < 0.0

        # Determine whether each residue has its rotamer recovered
        metric_dict["rot_rec_batch"] = (correct_chi.sum(-1) == chi_mask.sum(-1)).float()
        metric_dict["rot_rec_mask_batch"] = chi_mask.sum(-1).float()

        # Keep only final pred for reporting
        loss_dict["sc_rmsd_batch"] = loss_dict["sc_msd_batch"].sqrt()
        return metric_dict

    def compute_losses(
        self, b: gd.Batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Obtain predictions
        results_dict = self(b)
        c = self.cfg.model.hbdesigner

        # Compute various losses.
        loss = 0.0
        loss_dict = {}

        # Chi loss terms
        mask = b.chi_nll_mask.bool()
        eps = 1e-6

        # Angle (sin/cos) MSE with norm penalty
        loss_dict["chi_mse_batch"] = self.compute_chi_mse(
                    results_dict["pred_chis_norm"][mask],
                    results_dict["pred_chis_unnorm"][mask],
                    b.aatype_gt[mask],
                    self.chi_angles_mask[b.aatype_gt[mask]],
                    b.chi_sincos_gt[mask],
                    eps,
                    c.chi_norm_weight,
                )
        loss_dict["chi_mse"] = torch.mean(loss_dict["chi_mse_batch"])

        if c.chi_mse_weight > 0:
            loss += c.chi_mse_weight * loss_dict["chi_mse"]

        # Sidechain atomic MSD loss
        loss_dict["sc_msd_batch"] = self.compute_sc_msd(
                    results_dict["pred_atom14_xyz"], 
                    b.atom14_xyz_gt[mask], 
                    b.atom14_mask_gt[mask], 
                    b.aatype_gt[mask]
                )
        loss_dict["sc_msd"] = torch.mean(loss_dict["sc_msd_batch"])

        if c.sc_msd_weight > 0:
            loss += c.sc_msd_weight * loss_dict["sc_msd"]

        # Sidechain atomic vdW clash loss here
        pred_xyz = b.atom14_xyz.clone()
        pred_xyz[mask] = results_dict["pred_atom14_xyz"]
        loss_dict["clash_loss_batch"] = self.compute_sc_clash_loss(
            pred_xyz, 
            b.atom14_mask_gt, 
            b.aatype_gt, 
            b.aatype_batch,
            b.chi_nll_mask,
            clash_overlap_tolerance=0.6, 
            distance_threshold=14.,
            )
        loss_dict["clash_loss"] = torch.mean(loss_dict["clash_loss_batch"])
        if c.sc_clash_weight > 0:
            loss += c.sc_clash_weight * loss_dict["clash_loss"]

        # Sidechain atomic orientation loss
        loss_dict["orient_loss_batch"] = self.compute_sc_orient_loss(
            results_dict["pred_atom14_xyz"], 
            b.atom14_xyz_gt[mask], 
            b.atom14_mask_gt[mask], 
            b.aatype_batch[mask],
            b.aatype[mask],
        )
        loss_dict["orient_loss"] = torch.mean(loss_dict["orient_loss_batch"])
        if c.orient_msd_weight > 0:
            loss += c.orient_msd_weight * loss_dict["orient_loss"]

        # Compute other metrics
        loss_dict.update(self._compute_other_metrics(b, results_dict, loss_dict))
        return loss, loss_dict

    @torch.no_grad()
    def run_pack_recyc(self, 
        b: gd.Batch, 
        n_recycles: int = 0,
        ) -> Dict[str, torch.Tensor]:
        """Currently used by Trainer.pack_batch in hbdes3 mode"""

        # NOTE: do I need to zero these out?
        # Override with native sequence for packing-only
        b.net_res_num[:] = 0
        seq = b.aatype

        b.atom14_xyz[:, 4:] = 0.
        b.atom14_mask[:, 4:] = 0.

        # Zero out sc dihedral
        b.sc_dihedral[:] = 0.
        
        bb_xyz = b.atom14_xyz[:, :4]
        chi_angles_sincos = torch.stack(
            [torch.sin(b.sc_dihedral), torch.cos(b.sc_dihedral)], dim=-1
        ).view(-1, 4, 2)
        xyz_sc, mask_sc = self.get_atom14_xyz_from_chi(seq, bb_xyz, chi_angles_sincos)
        mask = b.chi_nll_mask.bool()

        b.atom14_xyz[mask] = xyz_sc[mask]
        b.atom14_mask[mask] = mask_sc[mask]
        b.sc_dihedral_mask[mask] = self.chi_angles_mask[seq[mask]]

        def get_processed_node_embeddings(b, seq):
            # Create initial embedding of protein nodes
            protein_nodes = self._form_protein_nodes(
                b.bb_dihedral,
                seq,
                b.sc_dihedral,
            )

            # Get k-nearest neighbor graph for protein
            protein_edge_index = self._get_knn_edges(b.atom14_xyz, b.aatype_batch)

            # Create initial embedding of protein edges
            protein_edges = self._form_protein_edges(
                b.atom14_xyz,
                b.atom14_mask,
                b.residue_index,
                b.chain_index,
                protein_edge_index,
            )

            # Pass through MPNN layers
            for i, layer in enumerate(self.mpnn_layers):
                # Message passing
                protein_nodes, protein_edges = layer(
                    protein_nodes, protein_edges, protein_edge_index
                )
            return protein_nodes

        # Make predictions until model predicts each example is done.
        for n_r in range(n_recycles + 1):
        
            protein_nodes = get_processed_node_embeddings(
                b, seq,
            )  # [L, pn_dim]

            chi_res_to_pred_i = b.chi_nll_mask.bool()

            # Get chi predictions
            seq_embed = self.seq_embedder(seq[chi_res_to_pred_i])
            protein_nodes_seq = torch.cat([protein_nodes[chi_res_to_pred_i], seq_embed], 1)

            chi_preds_unnorm = self.sc_linear(protein_nodes_seq).view(-1, 4, 2) # [L, 4, 2]
            chi_preds_norm = normalize_chi(chi_preds_unnorm)

            # Build sidechain coords now
            aatype = b.aatype[chi_res_to_pred_i]
            bb_xyz = b.atom14_xyz[chi_res_to_pred_i, :4]
            chi_angles_sincos = chi_preds_norm
            sc_xyz, sc_mask = self.get_atom14_xyz_from_chi(aatype, bb_xyz, chi_angles_sincos)

            # Update xyz and dihedrals
            b.atom14_xyz[chi_res_to_pred_i] = sc_xyz
            b.atom14_mask[chi_res_to_pred_i] = sc_mask
            b.sc_dihedral[chi_res_to_pred_i] = sincos_to_angle(chi_angles_sincos) * self.chi_angles_mask[aatype]

        # Create the results dictionary
        results_dict = {}
        for i in range(b.num_graphs):
            seq_i = seq[b.aatype_batch == i]
            seq_i_mask = seq_i != rc.restype_num
            results_dict[i] = {"net_res": torch.where(seq_i_mask)[0], "seq": seq_i[seq_i_mask]}

        b.aatype = seq
        b.atom14_xyz[(1 - b.chi_nll_mask).bool(), 4:] = 0.
        b.atom14_mask[(1 - b.chi_nll_mask).bool(), 4:] = 0.
        b.done_mask = b.chi_nll_mask

        return b, results_dict

