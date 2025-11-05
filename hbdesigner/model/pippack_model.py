import logging
import math
from typing import Sequence, Optional, Union, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch_geometric.data as gd

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.rigid_utils import Rigid, Rotation
from hbdesigner.model.layers import MLP, get_act_fxn
from hbdesigner.model.config import ModelConfig
from hbdesigner.data.protein import Protein
from hbdesigner.data.features import (
    build_sc_from_chi,
    calc_bb_dihedrals,
    calc_sc_dihedrals,
)


def make_atom14_masks(S):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=S.device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=S.device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=S.device,
    )
    protein_aatype = S.to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype].long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=S.device)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]

    return (
        residx_atom37_to_atom14,
        residx_atom37_mask,
        residx_atom14_to_atom37,
        residx_atom14_mask,
    )


def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (B, N, 14, 3)
    residx_atom37_to_atom14: torch.Tensor,  # (B, N, 37)
    atom37_atom_exists: torch.Tensor,  # (B, N, 37)
) -> torch.Tensor:  # (B, N, 37, 3)
    """Convert atom14 to atom37 representation."""

    atom37_data = torch.gather(
        atom14_data,
        dim=2,
        index=residx_atom37_to_atom14.unsqueeze(-1).expand(-1, -1, -1, 3).long(),
    )

    atom37_data *= atom37_atom_exists[..., None].float()

    return atom37_data


def get_bb_frames(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor, fixed=True):
    # N, CA, C = [*, L, 3]
    return Rigid.from_3_points(N, CA, C, fixed=fixed)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [...,N,C] at Neighbor indices [...,N,K] => [...,N,K,C]
    is_batched = neighbor_idx.dim() == 3
    n_feat_dims = nodes.dim() - (1 + is_batched)

    # Flatten and expand indices per batch [...,N,K] => [...,NK] => [...,NK,C]
    neighbors_flat = neighbor_idx.view((*neighbor_idx.shape[:-2], -1))
    for _ in range(n_feat_dims):
        neighbors_flat = neighbors_flat.unsqueeze(-1)
    neighbors_flat = neighbors_flat.expand(
        *([-1] * (1 + is_batched)), *nodes.shape[-n_feat_dims:]
    )

    # Gather and re-pack
    neighbor_features = torch.gather(nodes, -n_feat_dims - 1, neighbors_flat)
    neighbor_features = neighbor_features.view(
        list(neighbor_idx.shape) + list(nodes.shape[-n_feat_dims:])
    )
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class MPNNLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        edge_update=False,
        act="relu",
        extra_params=0,
    ):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.edge_update = edge_update

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.W_v = MLP(
            num_hidden + num_in,
            num_hidden + extra_params,
            num_hidden,
            num_layers=3,
            act=act,
        )
        self.dense = MLP(num_hidden, num_hidden * 4, num_hidden, num_layers=2, act=act)

        self.act = get_act_fxn(act)

        if edge_update:
            self.W_e = MLP(
                num_hidden + num_in,
                num_hidden + extra_params,
                num_hidden,
                num_layers=3,
                act=act,
            )
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, E_idx=None, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        if torch.is_tensor(E_idx):
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
        else:
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W_v(h_EV)
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout[0](dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if self.edge_update:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W_e(h_EV)
            h_E = self.norm2(h_E + self.dropout2(h_message))

            return h_V, h_E
        else:
            return h_V


class InvariantPointMessagePassing(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        n_points=8,
        dropout=0.1,
        act="relu",
        edge_update=False,
        position_scale=10.0,
    ):
        super().__init__()

        self.edge_update = edge_update
        self.n_points = n_points
        self.position_scale = position_scale
        self.points_fn_node = nn.Linear(node_dim, n_points * 3)
        if edge_update:
            self.points_fn_edge = nn.Linear(node_dim, n_points * 3)

        # Input to message is: 2*node_dim + edge_dim + 3*3*n_points
        self.node_message_fn = MLP(
            2 * node_dim + edge_dim + 9 * n_points, hidden_dim, hidden_dim, 3, act=act
        )
        if edge_update:
            self.edge_message_fn = MLP(
                2 * node_dim + edge_dim + 9 * n_points,
                hidden_dim,
                hidden_dim,
                3,
                act=act,
            )

        # Dropout and layer norms
        n_layers = 2
        if edge_update:
            n_layers = 4
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Feedforward layers
        self.node_dense = MLP(
            hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act
        )
        if edge_update:
            self.edge_dense = MLP(
                hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act
            )

    def _get_message_input(self, h_V, h_E, E_idx, X, edge=False):
        # Get backbone global frames from N, CA, and C
        bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])
        bb_to_global = bb_to_global.scale_translation(1 / self.position_scale)

        # Generate points in local frame of each node
        if not edge:
            p_local = self.points_fn_node(h_V).reshape(
                (*h_V.shape[:-1], self.n_points, 3)
            )  # [B, L, N, 3]
        else:
            p_local = self.points_fn_edge(h_V).reshape(
                (*h_V.shape[:-1], self.n_points, 3)
            )  # [B, L, N, 3]

        # Project points into global frame
        p_global = bb_to_global[..., None].apply(p_local)  # [B, L, N, 3]
        p_global_expand = p_global.unsqueeze(-3).expand(
            *E_idx.shape, *p_global.shape[-2:]
        )  # [B, L, K, N, 3]

        # Get neighbor points in global frame for each node
        neighbor_idx = E_idx.view((*E_idx.shape[:-2], -1))  # [B, LK]
        neighbor_p_global = torch.gather(
            p_global,
            -3,
            neighbor_idx[..., None, None].expand(*neighbor_idx.shape, self.n_points, 3),
        )
        neighbor_p_global = neighbor_p_global.view(
            *E_idx.shape, self.n_points, 3
        )  # [B, L, K, N, 3]

        # Form message components:
        # 1. Node i's local points
        p_local_expand = p_local.unsqueeze(-3).expand(
            *E_idx.shape, *p_local.shape[-2:]
        )  # [B, L, K, N, 3]

        # 2. Distance between node i's local points and its CA
        p_local_norm = torch.sqrt(
            torch.sum(p_local_expand**2, dim=-1) + 1e-8
        )  # [B, L, K, N]

        # 3. Node j's points in i's local frame
        neighbor_p_local = bb_to_global[..., None, None].invert_apply(
            neighbor_p_global
        )  # [B, L, K, N, 3]

        # 4. Distance between node j's points in i's local frame and i's CA
        neighbor_p_local_norm = torch.sqrt(
            torch.sum(neighbor_p_local**2, dim=-1) + 1e-8
        )  # [B, L, K, N]

        # 5. Distance between node i's global points and node j's global points
        neighbor_p_global_norm = torch.sqrt(
            torch.sum((p_global_expand - neighbor_p_global) ** 2, dim=-1) + 1e-8
        )  # [B, L, K, N]

        # Node message
        node_expand = h_V.unsqueeze(-2).expand(*E_idx.shape, h_V.shape[-1])
        neighbor_edge = cat_neighbors_nodes(h_V, h_E, E_idx)
        message_in = torch.cat(
            (
                node_expand,
                neighbor_edge,
                p_local_expand.view((*E_idx.shape, -1)),
                p_local_norm,
                neighbor_p_local.view((*E_idx.shape, -1)),
                neighbor_p_local_norm,
                neighbor_p_global_norm,
            ),
            dim=-1,
        )

        return message_in

    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        # Get message fn input
        message_in = self._get_message_input(h_V, h_E, E_idx, X)

        # Update nodes
        node_m = self.node_message_fn(message_in)
        if mask_attend is not None:
            node_m = node_m * mask_attend[..., None]
        node_m = torch.mean(node_m, dim=-2)
        h_V = self.norm[0](h_V + self.dropout[0](node_m))
        node_m = self.node_dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](node_m))
        if mask_V is not None:
            h_V = h_V * mask_V[..., None]

        if self.edge_update:
            # Get message fn input
            message_in = self._get_message_input(h_V, h_E, E_idx, X, edge=True)

            # Update edges
            edge_m = self.edge_message_fn(message_in)
            if mask_attend is not None:
                edge_m = edge_m * mask_attend[..., None]
            h_E = self.norm[2](h_E + self.dropout[2](edge_m))
            edge_m = self.edge_dense(h_E)
            h_E = self.norm[3](h_E + self.dropout[3](edge_m))
            if mask_attend is not None:
                h_E = h_E * mask_attend[..., None]

        return h_V, h_E


class IPMP_IPA(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=16,
        n_heads=1,
        n_query_points=4,
        n_value_points=8,
        edge_update=False,
        position_scale=10.0,
        dropout=0.1,
        act="relu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        self.position_scale = position_scale
        self.edge_update = edge_update

        # Linear layers for queries, keys, and values
        self.linear_q = nn.Linear(node_dim, hidden_dim * n_heads)
        self.linear_kv = nn.Linear(node_dim, 2 * hidden_dim * n_heads)

        self.linear_q_points = nn.Linear(node_dim, n_heads * n_query_points * 3)
        self.linear_kv_points = nn.Linear(
            node_dim, n_heads * (n_query_points + n_value_points) * 3
        )

        self.linear_b = nn.Linear(edge_dim, n_heads)

        self.head_weights = nn.Parameter(torch.zeros((n_heads)))
        with torch.no_grad():
            self.head_weights.fill_(0.541324854612918)

        out_dim = n_heads * (edge_dim + hidden_dim + n_value_points * 4)
        self.linear_out = nn.Linear(out_dim, node_dim)

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

        self.node_dense = MLP(node_dim, node_dim * 4, node_dim, num_layers=2, act=act)

        if edge_update:
            # Linear layers for queries, keys, and values
            self.linear_q_e = nn.Linear(node_dim, hidden_dim * n_heads)
            self.linear_kv_e = nn.Linear(node_dim, 2 * hidden_dim * n_heads)

            self.linear_q_points_e = nn.Linear(node_dim, n_heads * n_query_points * 3)
            self.linear_kv_points_e = nn.Linear(
                node_dim, n_heads * (n_query_points + n_value_points) * 3
            )

            self.linear_b_e = nn.Linear(edge_dim, n_heads)

            self.head_weights_e = nn.Parameter(torch.zeros((n_heads)))
            with torch.no_grad():
                self.head_weights_e.fill_(0.541324854612918)

            out_dim = n_heads * (edge_dim + hidden_dim + n_value_points * 4)
            self.linear_out_e = nn.Linear(out_dim, edge_dim)

            self.dropout_e = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
            self.norm_e = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

            self.edge_dense = MLP(
                edge_dim, edge_dim * 4, edge_dim, num_layers=2, act=act
            )

    def _get_node_update(self, h_V, h_E, E_idx, X, mask_attend=None):
        # Get backbone global frames from N, CA, and C
        scaled_X = X / self.position_scale
        bb_to_global = get_bb_frames(
            scaled_X[..., 0, :], scaled_X[..., 1, :], scaled_X[..., 2, :]
        )

        # Generate queries, keys, and values from nodes
        q = self.linear_q(h_V)  # [*, N_res, H * C]
        q = q.view(q.shape[:-1] + (self.n_heads, -1))  # [*, N_res, H, C]

        kv = self.linear_kv(h_V)  # [*, N_res, 2 * H * C]
        kv = kv.view(kv.shape[:-1] + (self.n_heads, -1))  # [*, N_res, H, 2 * C]
        k, v = torch.split(kv, self.hidden_dim, dim=-1)  # 2 [*, N_res, H, C]

        # Generate query, key, and value points from nodes
        q_pts = self.linear_q_points(h_V)  # [*, N_res, H * P_q * 3]
        q_pts = torch.split(
            q_pts, q_pts.shape[-1] // 3, dim=-1
        )  # 3 [*, N_res, H * P_q]
        q_pts = torch.stack(q_pts, dim=-1)  # [*, N_res, H * P_q, 3]
        q_pts = bb_to_global[..., None].apply(q_pts)  # [*, N_res, H * P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.n_heads, self.n_query_points, 3)
        )  # [*, N_res, H, P_q, 3]

        kv_pts = self.linear_kv_points(h_V)  # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = torch.split(
            kv_pts, kv_pts.shape[-1] // 3, dim=-1
        )  # 3 [*, N_res, H * (P_q + P_v)]
        kv_pts = torch.stack(kv_pts, dim=-1)  # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = bb_to_global[..., None].apply(kv_pts)  # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.n_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.n_query_points, self.n_value_points], dim=-2
        )  # [*, N_res, H, P_q, 3], [*, N_res, H, P_v, 3]

        # Compute attention bias
        b = self.linear_b(h_E)  # [*, N_res, K, H]

        # Compute attention weight
        a = torch.einsum("...ihc,...ijhc->...ijh", q, gather_nodes(k, E_idx))
        a *= math.sqrt(1.0 / (3 * self.hidden_dim))
        a += math.sqrt(1.0 / 3) * b  # [*, N_res, K, H]

        pt_att = q_pts.unsqueeze(-4) - gather_nodes(
            k_pts, E_idx
        )  # [*, N_res, K, H, P_q, 3]
        pt_att = torch.sum(pt_att**2, dim=-1)  # [*, N_res, K, H, P_q]

        head_weights = F.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )  # [*, 1, 1, H, 1]
        pt_att = (
            math.sqrt(1.0 / (3 * (self.n_query_points * 9.0 / 2)))
            * head_weights
            * pt_att
        )  # [*, N_res, K, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1) * -0.5  # [*, N_res, K, H]

        if mask_attend is not None:
            att_mask = 1e5 * (mask_attend - 1)
        else:
            att_mask = torch.zeros_like(E_idx)

        a = a + pt_att + att_mask[..., None]  # [*, N_res, K, H]
        a = F.softmax(a, dim=-2)  # [*, N_res, K, H]

        # Compute update
        # [*, N_res, H, C_hidden]
        o = torch.einsum("...ijh,...ijhc->...ihc", a, gather_nodes(v, E_idx))
        o = o.view(*o.shape[:-2], -1)

        o_pt = torch.einsum("...ijh,...ijhpx->...ihpx", a, gather_nodes(v_pts, E_idx))
        o_pt = bb_to_global[..., None, None].invert_apply(o_pt)  # [*, N_res, H, P_v, 3]
        o_pt_norm = torch.sqrt(torch.sum(o_pt**2, dim=-1) + 1e-8).view(
            *o_pt.shape[:-3], -1
        )
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        o_pair = torch.einsum("...ijh,...ijc->...ihc", a, h_E)  # [*, N_res, H, C_z]
        o_pair = o_pair.view(*o_pair.shape[:-2], -1)

        # Compute node update
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1)
        )

        return s

    def _get_edge_update(self, h_V, h_E, E_idx, X, mask_attend=None):
        # Get backbone global frames from N, CA, and C
        scaled_X = X / self.position_scale
        bb_to_global = get_bb_frames(
            scaled_X[..., 0, :], scaled_X[..., 1, :], scaled_X[..., 2, :]
        )

        # Generate queries, keys, and values from nodes
        q = self.linear_q_e(h_V)  # [*, N_res, H * C]
        q = q.view(q.shape[:-1] + (self.n_heads, -1))  # [*, N_res, H, C]

        kv = self.linear_kv_e(h_V)  # [*, N_res, 2 * H * C]
        kv = kv.view(kv.shape[:-1] + (self.n_heads, -1))  # [*, N_res, H, 2 * C]
        k, v = torch.split(kv, self.hidden_dim, dim=-1)  # 2 [*, N_res, H, C]

        # Generate query, key, and value points from nodes
        q_pts = self.linear_q_points_e(h_V)  # [*, N_res, H * P_q * 3]
        q_pts = torch.split(
            q_pts, q_pts.shape[-1] // 3, dim=-1
        )  # 3 [*, N_res, H * P_q]
        q_pts = torch.stack(q_pts, dim=-1)  # [*, N_res, H * P_q, 3]
        q_pts = bb_to_global[..., None].apply(q_pts)  # [*, N_res, H * P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.n_heads, self.n_query_points, 3)
        )  # [*, N_res, H, P_q, 3]

        kv_pts = self.linear_kv_points_e(h_V)  # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = torch.split(
            kv_pts, kv_pts.shape[-1] // 3, dim=-1
        )  # 3 [*, N_res, H * (P_q + P_v)]
        kv_pts = torch.stack(kv_pts, dim=-1)  # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = bb_to_global[..., None].apply(kv_pts)  # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.n_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.n_query_points, self.n_value_points], dim=-2
        )  # [*, N_res, H, P_q, 3], [*, N_res, H, P_v, 3]

        # Compute attention bias
        b = self.linear_b_e(h_E)  # [*, N_res, K, H]

        # Compute attention weight
        a = torch.einsum("...ihc,...ijhc->...ijh", q, gather_nodes(k, E_idx))
        a *= math.sqrt(1.0 / (3 * self.hidden_dim))
        a += math.sqrt(1.0 / 3) * b  # [*, N_res, K, H]

        pt_att = q_pts.unsqueeze(-4) - gather_nodes(
            k_pts, E_idx
        )  # [*, N_res, K, H, P_q, 3]
        pt_att = torch.sum(pt_att**2, dim=-1)  # [*, N_res, K, H, P_q]

        head_weights = F.softplus(self.head_weights_e).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )  # [*, 1, 1, H, 1]
        pt_att = (
            math.sqrt(1.0 / (3 * (self.n_query_points * 9.0 / 2)))
            * head_weights
            * pt_att
        )  # [*, N_res, K, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1) * -0.5  # [*, N_res, K, H]

        if mask_attend is not None:
            att_mask = 1e5 * (mask_attend - 1)
        else:
            att_mask = torch.zeros_like(E_idx)

        a = a + pt_att + att_mask[..., None]  # [*, N_res, K, H]
        a = F.softmax(a, dim=-2)  # [*, N_res, K, H]

        # Compute update
        # [*, N_res, K, H, C_hidden]
        o = torch.einsum("...ijh,...ijhc->...ijhc", a, gather_nodes(v, E_idx))
        o = o.view(*o.shape[:-2], -1)

        o_pt = torch.einsum("...ijh,...ijhpx->...ijhpx", a, gather_nodes(v_pts, E_idx))
        o_pt = bb_to_global[..., None, None, None].invert_apply(
            o_pt
        )  # [*, N_res, K, H, P_v, 3]
        o_pt_norm = torch.sqrt(torch.sum(o_pt**2, dim=-1) + 1e-8).view(
            *o_pt.shape[:-3], -1
        )
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        o_pair = torch.einsum("...ijh,...ijc->...ijhc", a, h_E)  # [*, N_res, K, H, C_z]
        o_pair = o_pair.view(*o_pair.shape[:-2], -1)

        # Compute edge update
        s = self.linear_out_e(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1)
        )

        return s

    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        s = self._get_node_update(h_V, h_E, E_idx, X, mask_attend)
        h_V = self.norm[0](h_V + self.dropout[0](s))
        node_m = self.node_dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](node_m))

        if mask_V is not None:
            h_V = h_V * mask_V[..., None]

        if self.edge_update:
            s = self._get_edge_update(h_V, h_E, E_idx, X, mask_attend)
            if mask_attend is not None:
                s = s * mask_attend[..., None]
            h_E = self.norm_e[0](h_E + self.dropout_e[0](s))
            edge_m = self.edge_dense(h_E)
            h_E = self.norm_e[1](h_E + self.dropout_e[1](edge_m))
            if mask_attend is not None:
                h_E = h_E * mask_attend[..., None]

        return h_V, h_E


class PositionalEncodings(nn.Module):
    def __init__(
        self,
        num_embeddings,
        period_range=[2, 1000],
        max_relative_feature=32,
        af2_relpos=False,
    ):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range
        self.max_relative_feature = max_relative_feature
        self.af2_relpos = af2_relpos

    def _transformer_encoding(self, E_idx):
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view(
            (1, -1, 1)
        )
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(
                0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device
            )
            * -(np.log(10000.0) / self.num_embeddings)
        )

        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        return E

    def _af2_encoding(self, E_idx, residue_index=None):
        # i-j
        if residue_index is not None:
            offset = residue_index[..., None] - residue_index[..., None, :]
            offset = torch.gather(offset, -1, E_idx)
        else:
            N_nodes = E_idx.size(1)
            ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view(
                (1, -1, 1)
            )
            offset = E_idx.float() - ii

        relpos = torch.clip(
            offset.long() + self.max_relative_feature, 0, 2 * self.max_relative_feature
        )
        relpos = F.one_hot(relpos, 2 * self.max_relative_feature + 1)

        return relpos

    def forward(self, E_idx, residue_index=None):
        if self.af2_relpos:
            E = self._af2_encoding(E_idx, residue_index)
        else:
            E = self._transformer_encoding(E_idx)

        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        dropout=0.1,
        af2_relpos=True,
        mask_distances=False,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.mask_distances = mask_distances

        if af2_relpos:
            num_positional_embeddings = 65

        # Feature dimensions
        node_in = 21 + 3 * 2
        edge_in = num_positional_embeddings + (14**2) * num_rbf

        # Positional encoding
        self.embeddings = PositionalEncodings(
            num_positional_embeddings, af2_relpos=af2_relpos
        )
        self.dropout = nn.Dropout(dropout)

        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        """Pairwise euclidean distances"""
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + 2 * (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(self.top_k, X.shape[-2]), dim=-1, largest=False
        )
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _impute_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        return Cb

    def _atomic_distances(self, X, E_idx, X_mask):
        RBF_all = []
        for i in range(X.shape[-2]):
            for j in range(X.shape[-2]):
                rbf = self._get_rbf(X[..., i, :], X[..., j, :], E_idx)
                if self.mask_distances:
                    X_mask_j = gather_nodes(X_mask[..., j, None], E_idx)
                    rbf = rbf * X_mask[..., i, None, None] * X_mask_j
                RBF_all.append(rbf)

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all

    def forward(self, X, S, BB_D, mask, residue_index=None, X_mask=None):
        """Featurize coordinates as an attributed graph"""

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        _, E_idx, _ = self._dist(X_ca, mask)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx, residue_index)

        # Pairwise bb atomic distances
        Ca_xyz = X[:, :, 1, :]
        N_xyz = X[:, :, 0, :]
        C_xyz = X[:, :, 2, :]
        O_xyz = X[:, :, 3, :]
        Cb = self._impute_CB(N_xyz, Ca_xyz, C_xyz)
        sc_atoms = X[..., 5:, :]
        X2 = torch.stack((N_xyz, Ca_xyz, C_xyz, O_xyz, Cb), dim=-2)
        X2 = torch.cat((X2, sc_atoms), dim=-2)
        if X_mask is None:
            X_mask = torch.ones_like(X2[..., 0])
        else:
            Cb_mask = (torch.prod(X_mask[:, :, :4], dim=-1) != 0.0).float()
            X_mask[:, :, 4] = Cb_mask
        RBF_all = self._atomic_distances(X2, E_idx, X_mask)

        E = torch.cat((E_positional, RBF_all), -1)
        Vs = []
        # One-hot encoded sequence
        Vs.append(F.one_hot(S, num_classes=21).float())

        # Sin/cos encoded backbone dihedrals
        Vs.append(BB_D.view(*BB_D.shape[:-2], -1))

        # Embed nodes
        V = torch.cat(Vs, dim=-1)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)

        # Embed edges
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return V, E, E_idx, X


def get_atom14_coords(X, S, BB_D, SC_D):
    # Convert angles to sin/cos
    BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
    SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)

    # Get backbone global frames from N, CA, and C
    bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])

    # Concatenate all angles
    angle_agglo = torch.cat([BB_D_sincos, SC_D_sincos], dim=-2)  # [B, L, 7, 2]

    # Get norm of angles
    norm_denom = torch.sqrt(
        torch.clamp(torch.sum(angle_agglo**2, dim=-1, keepdim=True), min=1e-12)
    )

    # Normalize
    normalized_angles = angle_agglo / norm_denom

    # Make default frames
    default_frames = torch.tensor(
        rc.restype_rigid_group_default_frame,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make group ids
    group_idx = torch.tensor(
        rc.restype_atom14_to_rigid_group, device=X.device, requires_grad=False
    )

    # Make atom mask
    atom_mask = torch.tensor(
        rc.restype_atom14_mask,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make literature positions
    lit_positions = torch.tensor(
        rc.restype_atom14_rigid_group_positions,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make all global frames
    all_frames_to_global = torsion_angles_to_frames(
        bb_to_global, normalized_angles, S, default_frames
    )

    # Predict coordinates
    pred_xyz = frames_and_literature_positions_to_atom14_pos(
        all_frames_to_global, S, default_frames, group_idx, atom_mask, lit_positions
    )

    # Replace backbone atoms with input coordinates
    pred_xyz[..., :4, :] = X[..., :4, :]

    return pred_xyz


class PIPPack(nn.Module):
    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_mpnn_layers: int = 3,
        k_neighbors: int = 30,
        augment_eps: float = 0.0,
        use_ipmp: bool = False,
        use_ipmp_ipa: bool = False,
        n_points: Optional[int] = None,
        dropout: float = 0.1,
        act: str = "relu",
        predict_bin_chi: bool = True,
        n_chi_bins: int = 72,
        predict_offset: bool = True,
        position_scale: float = 1.0,
        recycle_strategy: str = "mode",
        recycle_SC_D_sc: bool = False,
        recycle_SC_D_probs: bool = False,
        recycle_X: bool = True,
        mask_distances: bool = False,
        loss: Optional[Dict[str, Union[float, bool]]] = {
            "chi_nll_loss_weight": 1.0,
            "chi_mse_loss_weight": 1.0,
            "offset_mse_loss_weight": 1.0,
        },
    ) -> None:
        """Graph labeling network"""
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.recycle_strategy = recycle_strategy
        self.recycle_SC_D_sc = recycle_SC_D_sc
        self.recycle_SC_D_probs = recycle_SC_D_probs
        self.recycle_X = recycle_X
        self.loss = loss
        self.log = logging.getLogger("PIPPack")

        # Featurization layers
        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout,
            mask_distances=mask_distances,
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Sequence embedding layer
        self.W_seq = nn.Embedding(21, hidden_dim)

        # Recycling embedding layers
        if recycle_SC_D_sc:
            self.W_recycle_SC_D_sc = nn.Linear(8, hidden_dim)
        if recycle_SC_D_probs:
            self.W_recycle_SC_D_probs = nn.Linear(4 * (n_chi_bins + 1), hidden_dim)

        # Recycling embedding layers
        if recycle_SC_D_sc:
            self.W_recycle_SC_D_sc = nn.Linear(8, hidden_dim)
        if recycle_SC_D_probs:
            self.W_recycle_SC_D_probs = nn.Linear(4 * (n_chi_bins + 1), hidden_dim)

        # MPNN layers
        self.use_ipmp = use_ipmp
        self.use_ipmp_ipa = use_ipmp_ipa
        if use_ipmp:
            self.mpnn_layers = nn.ModuleList(
                [
                    InvariantPointMessagePassing(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        n_points,
                        dropout,
                        act=act,
                        edge_update=True,
                        position_scale=position_scale,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )
        elif use_ipmp_ipa:
            self.mpnn_layers = nn.ModuleList(
                [
                    IPMP_IPA(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        edge_update=True,
                        dropout=dropout,
                        act=act,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )
        else:
            self.mpnn_layers = nn.ModuleList(
                [
                    MPNNLayer(
                        hidden_dim,
                        hidden_dim * 2,
                        dropout=dropout,
                        edge_update=True,
                        act=act,
                        scale=k_neighbors,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )

        # Output layers
        self.predict_bin_chi = predict_bin_chi
        self.n_chi_bins = n_chi_bins
        out_dim = 8 if not predict_bin_chi else (n_chi_bins + 1) * 4
        self.W_out_chi = MLP(hidden_dim * 2, hidden_dim, out_dim, 3, act=act)

        # Offset prediction
        self.predict_offset = predict_offset
        if predict_offset:
            self.offset_layer = nn.Linear(node_features, 4)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _chi_prediction_from_probs(
        self, chi_probs, chi_bin_offset=None, strategy="mode"
    ):
        # One-hot encode predicted chi bin
        if strategy == "mode":
            chi_bin = torch.argmax(chi_probs, dim=-1)
        elif strategy == "sample":
            chi_bin = (
                torch.multinomial(
                    chi_probs.view(-1, chi_probs.shape[-1]), num_samples=1
                )
                .squeeze(-1)
                .view(*chi_probs.shape[:-1])
            )
        chi_bin_one_hot = F.one_hot(chi_bin, num_classes=self.n_chi_bins + 1)

        # Determine actual chi value from bin
        chi_bin_rad = torch.cat(
            (
                torch.arange(
                    -torch.pi,
                    torch.pi,
                    2 * torch.pi / self.n_chi_bins,
                    device=chi_bin.device,
                ),
                torch.tensor([0]).to(device=chi_bin.device),
            )
        )
        pred_chi_bin = torch.sum(
            chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1
        )

        # Add bin offset
        if self.predict_offset and chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / self.n_chi_bins) * torch.rand(
                chi_bin.shape, device=chi_bin.device
            )
        sampled_chi = pred_chi_bin + bin_sample_update

        return sampled_chi

    @property
    def metric_names(self) -> Sequence[str]:
        metrics = [
            "rotamer recovery",
            "rmsd",
        ]

        if self.predict_bin_chi:
            metrics.append("chi nll loss")
            if self.predict_offset:
                metrics.append("offset mse loss")
        else:
            metrics.append("chi mse loss")

        return metrics

    @property
    def monitor_metric(self) -> str:
        if self.predict_bin_chi:
            return "val chi nll loss mean"
        else:
            return "val chi mse loss mean"

    def forward(self, batch, n_recycle=0):
        # Add empty previous prediction
        prevs = {
            "pred_X": torch.zeros_like(batch.X),
            "pred_X_mask": torch.concatenate(
                (
                    torch.ones_like(batch.X_mask[..., :5]),
                    torch.zeros_like(batch.X_mask[..., 5:]),
                ),
                -1,
            ),
            "pred_SC_D": torch.zeros_like(batch.SC_D),
            "pred_SC_D_probs": torch.zeros(
                (*batch.S.shape, 4, self.n_chi_bins + 1), device=batch.S.device
            ),
        }

        with torch.no_grad():
            # Loop over all recycle iterations
            for _ in range(n_recycle):
                outputs = self.single_forward(batch, prevs)

                # Create coordinates for prediction
                if self.predict_bin_chi:
                    chi_pred = self._chi_prediction_from_probs(
                        outputs["chi_probs"],
                        outputs.get("chi_bin_offset", None),
                        strategy=self.recycle_strategy,
                    )
                else:
                    chi_pred = outputs["norm_chi"]
                    chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
                aatype_chi_mask = torch.tensor(
                    rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device
                )[batch.S]
                chi_pred = aatype_chi_mask * chi_pred
                atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

                # Update previous predictions
                if self.recycle_X:
                    prevs["pred_X"] = atom14_xyz
                    prevs["pred_X_mask"] = (atom14_xyz.sum(-1) != 0).float()
                prevs["pred_SC_D"] = chi_pred
                prevs["pred_SC_D_probs"] = outputs.get("chi_probs", None)

        # Final prediction
        outputs = self.single_forward(batch, prevs)

        # Create coordinates for prediction
        if self.predict_bin_chi:
            chi_pred = self._chi_prediction_from_probs(
                outputs["chi_probs"], outputs.get("chi_bin_offset", None)
            )
        else:
            chi_pred = outputs["norm_chi"]
            chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
        aatype_chi_mask = torch.tensor(
            rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device
        )[batch.S]
        chi_pred = aatype_chi_mask * chi_pred
        atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

        # Add final predictions to outputs
        outputs["final_SC_D"] = chi_pred
        outputs["final_X"] = atom14_xyz
        outputs["final_X_mask"] = (atom14_xyz.sum(-1) != 0).float()

        return outputs

    def single_forward(self, batch, prevs):
        """Graph-conditioned sequence model"""
        # Unpack batch
        X = torch.cat((batch.X[..., :4, :], prevs["pred_X"][..., 4:, :]), dim=-2)
        S = batch.S
        BB_D = batch.BB_D_sincos
        mask = batch.residue_mask
        residue_index = batch.residue_index

        # Embed initial features
        V, E, E_idx, X = self.features(
            X, S, BB_D, mask, residue_index, prevs["pred_X_mask"]
        )
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack(
                (torch.sin(prevs["pred_SC_D"]), torch.cos(prevs["pred_SC_D"])), dim=-1
            )
            h_V = h_V + self.W_recycle_SC_D_sc(
                pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1)
            )
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(
                prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1)
            )

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack(
                (torch.sin(prevs["pred_SC_D"]), torch.cos(prevs["pred_SC_D"])), dim=-1
            )
            h_V = h_V + self.W_recycle_SC_D_sc(
                pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1)
            )
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(
                prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1)
            )

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.mpnn_layers:
            if torch.is_grad_enabled():
                if self.use_ipmp or self.use_ipmp_ipa:
                    h_V, h_E = checkpoint(
                        layer,
                        h_V,
                        h_E,
                        E_idx,
                        X,
                        mask,
                        mask_attend,
                        use_reentrant=False,
                    )
                else:
                    h_V, h_E = checkpoint(
                        layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False
                    )
            else:
                if self.use_ipmp or self.use_ipmp_ipa:
                    h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)
                else:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        outputs = {}
        # One-hot encoded sequence for node features
        h_S = self.W_seq(S)
        h_VS = torch.cat((h_V, h_S), -1)
        if not self.predict_bin_chi:
            unnorm_chi = self.W_out_chi(h_VS)
            unnorm_chi = unnorm_chi.view(X.shape[0], X.shape[1], 4, 2)

            # Normalize chi outputs
            norm_denom = torch.sqrt(
                torch.clamp(torch.sum(unnorm_chi**2, dim=-1, keepdim=True), min=1e-12)
            )
            norm_chi = unnorm_chi / norm_denom
            outputs["unnorm_chi"] = unnorm_chi
            outputs["norm_chi"] = norm_chi
        else:
            CH_logits = self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1)
            chi_log_probs = F.log_softmax(CH_logits, dim=-1)
            chi_probs = F.softmax(CH_logits, dim=-1)
            outputs["chi_log_probs"] = chi_log_probs
            outputs["chi_probs"] = chi_probs
            outputs["chi_logits"] = CH_logits

        if self.predict_offset:
            offset = (2 * torch.pi / self.n_chi_bins) * torch.sigmoid(
                self.offset_layer(h_V)
            )
            outputs["chi_bin_offset"] = offset

        return outputs

    def sample(self, batch, temperature=1.0, n_recycle=0):
        # Add empty previous prediction
        prevs = {
            "pred_X": torch.zeros_like(batch.X),
            "pred_X_mask": torch.concatenate(
                (
                    torch.ones_like(batch.X_mask[..., :5]),
                    torch.zeros_like(batch.X_mask[..., 5:]),
                ),
                -1,
            ),
            "pred_SC_D": torch.zeros_like(batch.SC_D),
            "pred_SC_D_probs": torch.zeros(
                (*batch.S.shape, 4, self.n_chi_bins + 1), device=batch.S.device
            ),
        }

        with torch.no_grad():
            # Loop over all recycle iterations
            for _ in range(n_recycle):
                sample_out = self.single_sample(batch, prevs, temperature)

                # Create coordinates for prediction
                if self.predict_bin_chi:
                    chi_pred = self._chi_prediction_from_probs(
                        sample_out["chi_probs"],
                        sample_out["chi_bin_offset"],
                        strategy=self.recycle_strategy,
                    )
                else:
                    chi_pred = sample_out["norm_chi"]
                    chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
                aatype_chi_mask = torch.tensor(
                    rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device
                )[batch.S]
                chi_pred = aatype_chi_mask * chi_pred
                atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

                # Update previous predictions
                if self.recycle_X:
                    prevs["pred_X"] = atom14_xyz
                    prevs["pred_X_mask"] = (atom14_xyz.sum(-1) != 0).float()
                prevs["pred_SC_D"] = chi_pred
                prevs["pred_SC_D_probs"] = sample_out.get("chi_probs", None)

            # Final prediction
            sample_out = self.single_sample(batch, prevs, temperature)

            # Create coordinates for prediction
            if self.predict_bin_chi:
                chi_pred = self._chi_prediction_from_probs(
                    sample_out["chi_probs"], sample_out["chi_bin_offset"]
                )
            else:
                chi_pred = sample_out["norm_chi"]
                chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
            aatype_chi_mask = torch.tensor(
                rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device
            )[batch.S]
            chi_pred = aatype_chi_mask * chi_pred
            atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

            # Add final predictions to outputs
            sample_out["final_SC_D"] = chi_pred
            sample_out["final_X"] = atom14_xyz
            sample_out["final_X_mask"] = (atom14_xyz.sum(-1) != 0).float()

        return sample_out

    def single_sample(self, batch, prevs, temperature=1.0):
        """Autoregressive decoding of a model"""
        # Unpack batch
        X = torch.cat((batch.X[..., :4, :], prevs["pred_X"][..., 4:, :]), dim=-2)
        S = batch.S
        BB_D = batch.BB_D_sincos
        mask = batch.residue_mask
        residue_index = batch.residue_index

        # Prepare node and edge embeddings
        V, E, E_idx, X = self.features(
            X, S, BB_D, mask, residue_index, prevs["pred_X_mask"]
        )
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack(
                (torch.sin(prevs["pred_SC_D"]), torch.cos(prevs["pred_SC_D"])), dim=-1
            )
            h_V = h_V + self.W_recycle_SC_D_sc(
                pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1)
            )
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(
                prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1)
            )

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack(
                (torch.sin(prevs["pred_SC_D"]), torch.cos(prevs["pred_SC_D"])), dim=-1
            )
            h_V = h_V + self.W_recycle_SC_D_sc(
                pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1)
            )
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(
                prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1)
            )

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.mpnn_layers:
            if self.use_ipmp or self.use_ipmp_ipa:
                h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_seq(S)
        h_VS = torch.cat((h_V, h_S), dim=-1)

        # Chi prediction
        if not self.predict_bin_chi:
            chi_mask = torch.tensor(
                rc.chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], device=X.device
            )[S].unsqueeze(-1)
            unnorm_chi = self.W_out_chi(h_VS)
            unnorm_chi = unnorm_chi.view(X.shape[0], X.shape[1], 4, 2)

            # Normalize chi outputs
            norm_denom = torch.sqrt(
                torch.clamp(torch.sum(unnorm_chi**2, dim=-1, keepdim=True), min=1e-12)
            )
            norm_chi = unnorm_chi / norm_denom

            # Mask the chi outputs
            unnorm_chi = chi_mask * unnorm_chi
            norm_chi = chi_mask * norm_chi
        else:
            chi_mask = torch.tensor(
                rc.chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], device=X.device
            )[S].unsqueeze(-1)
            h_VS = torch.cat([h_V, h_S], dim=-1)
            if temperature > 0.0:
                CH_logits = (
                    self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1)
                    / temperature
                )
                chi_probs = F.softmax(CH_logits, dim=-1)
                CH = (
                    torch.multinomial(chi_probs.view(-1, CH_logits.shape[-1]), 1)
                    .view(
                        CH_logits.shape[0], CH_logits.shape[1], CH_logits.shape[2], -1
                    )
                    .squeeze(-1)
                )
            else:
                CH_logits = self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1)
                chi_probs = F.softmax(CH_logits, dim=-1)
                CH = torch.argmax(chi_probs, dim=-1)

        if self.predict_offset:
            offset = (2 * torch.pi / self.n_chi_bins) * torch.sigmoid(
                self.offset_layer(h_V)
            )

        output = {
            "norm_chi": norm_chi if not self.predict_bin_chi else None,
            "unnorm_chi": unnorm_chi if not self.predict_bin_chi else None,
            "chi_bin": CH if self.predict_bin_chi else None,
            "chi_probs": chi_probs if self.predict_bin_chi else None,
            "chi_bin_offset": offset if self.predict_offset else None,
            "chi_logits": CH_logits if self.predict_bin_chi else None,
        }

        return output


class PIPPackFineTune(PIPPack):
    def __init__(self, gumbel_temp=1.0, **kwargs):
        self.gumbel_temp = gumbel_temp
        super().__init__(**kwargs)

    @property
    def metric_names(self) -> Sequence[str]:
        metrics = ["rotamer recovery", "rmsd", "clash loss", "proline loss"]

        if self.predict_bin_chi:
            metrics.append("chi nll loss")
            if self.predict_offset:
                metrics.append("offset mse loss")
        else:
            metrics.append("chi mse loss")

        return metrics

    def _gumbel_sample_from_logits(self, chi_logits, chi_bin_offset=None):
        # Sample from Gumbel-Softmax distribution
        gumbel_chi_bin = F.gumbel_softmax(chi_logits, self.gumbel_temp, hard=True)

        # Determine actual chi value from bin
        chi_bin_rad = torch.cat(
            (
                torch.arange(
                    -torch.pi,
                    torch.pi,
                    2 * torch.pi / self.n_chi_bins,
                    device=chi_logits.device,
                ),
                torch.tensor([0]).to(device=chi_logits.device),
            )
        )
        pred_chi_bin = torch.sum(
            chi_bin_rad.view(*([1] * (len(chi_logits.shape) - 1)), -1) * gumbel_chi_bin,
            dim=-1,
        )

        # Add bin offset
        if self.predict_offset and chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / self.n_chi_bins) * torch.rand(
                chi_logits.shape, device=chi_logits.device
            )
        sampled_chi = pred_chi_bin + bin_sample_update

        return sampled_chi

    def forward(self, batch, n_recycle=0):
        outputs = super().forward(batch, n_recycle)

        # Add a gumbel sample to outputs
        gumbel_sample = self._gumbel_sample_from_logits(
            outputs["chi_logits"], outputs.get("chi_bin_offset", None)
        )
        aatype_chi_mask = torch.tensor(
            rc.chi_mask_atom14, dtype=torch.float32, device=gumbel_sample.device
        )[batch.S]
        chi_pred = aatype_chi_mask * gumbel_sample

        atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)
        outputs["gumbel_SC_D"] = chi_pred
        outputs["gumbel_X"] = atom14_xyz

        return outputs


# ---------- Utils for running PIPPack on Protein objects --------- #


def load_PIPPack(cfg: ModelConfig) -> Optional[PIPPackFineTune]:
    c = cfg.pippack

    if c.ckpt != "":
        sidechain_model = PIPPackFineTune(
            use_ipmp=c.use_ipmp,
            n_points=c.n_points,
            recycle_SC_D_sc=c.recycle_SC_D_sc,
            mask_distances=c.mask_distances,
            dropout=0.0,
        )
        ckpt = torch.load(c.ckpt, map_location="cpu")
        sidechain_model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError("No PIPPack checkpoint specified!")

    # PIPPack can be run on CPU or GPU depending on memory needs
    sidechain_model.to(c.device)
    sidechain_model.eval()
    return sidechain_model


class MPDict(dict):
    """Dict class able to move keys to/from CPU/GPU.
    Required to use PIPPack with multiprocessing."""

    def to(self, device):
        for key, value in self.items():
            self[key] = self[key].to(device)
        return self


def batch_to_PIPPack_batch(g: gd.Batch, k_neighbors: int) -> gd.Data:
    dev = g.x.device

    # Deconstruct batch back into data list
    data_list = g.to_data_list()

    # Determine shapes needed for batched tensors
    batch_dim = g.num_graphs
    if "full_aatype" in g:
        res_dim = max([d["full_aatype"].shape[0] for d in data_list])
    else:
        res_dim = max([d.num_nodes for d in data_list])
    if res_dim < k_neighbors:
        res_dim = k_neighbors  # Make sure to have full knn graph

    # Initialize necessary tensors with size and default value
    X = torch.zeros((batch_dim, res_dim, 14, 3), device=dev)
    X_mask = torch.zeros((batch_dim, res_dim, 14), device=dev)
    BB_D = torch.zeros((batch_dim, res_dim, 3), device=dev)
    BB_D_sincos = torch.zeros((batch_dim, res_dim, 3, 2), device=dev)
    SC_D = torch.zeros((batch_dim, res_dim, 4), device=dev)
    S = rc.restype_num * torch.ones((batch_dim, res_dim), device=dev).long()
    residue_mask = torch.zeros((batch_dim, res_dim), device=dev)
    residue_index = torch.zeros((batch_dim, res_dim), device=dev)

    # Loop over each Data in the data_list and update appropriate slice of tensors
    for i, d in enumerate(data_list):
        if "full_aatype" in g:
            n_res = d.full_aatype.shape[0]

            # Create arrays but overwrite any changes in the crop
            S[i, :n_res] = d.full_aatype
            S[i, :n_res][d.crop_mask] = d.aatype
            X[i, :n_res] = d.full_atom14_xyz
            X[i, :n_res][d.crop_mask] = d.atom14_xyz
            X_mask[i, :n_res] = d.full_atom14_mask
            X_mask[i, :n_res][d.crop_mask] = d.atom14_mask

            residue_mask[i, :n_res] = (
                torch.prod(X_mask[i, :n_res][..., :4], dim=-1) == 1.0
            )
            residue_index[i, :n_res] = d.full_residue_index

            bb_dihedral, _ = calc_bb_dihedrals(X[i, :n_res], d.full_residue_index)
            BB_D[i, :n_res] = bb_dihedral
            BB_D_sincos[i, :n_res] = torch.stack(
                [torch.sin(bb_dihedral), torch.cos(bb_dihedral)], dim=-1
            )

            sc_dihedral, _ = calc_sc_dihedrals(X[i, :n_res], S[i, :n_res].long())
            SC_D[i, :n_res] = sc_dihedral
        else:
            n_res = d.aatype.shape[0]
            X[i, :n_res] = d.atom14_xyz
            X_mask[i, :n_res] = d.atom14_mask
            BB_D[i, :n_res] = d.bb_dihedral
            BB_D_sincos[i, :n_res] = torch.stack(
                [torch.sin(d.bb_dihedral), torch.cos(d.bb_dihedral)], dim=-1
            )
            SC_D[i, :n_res] = d.sc_dihedral
            S[i, :n_res] = d.aatype
            residue_mask[i, :n_res] = torch.prod(d.atom14_mask[..., :4], dim=-1) == 1.0
            residue_index[i, :n_res] = d.residue_index

    # Create the PIPPack batch object
    pippack_batch = gd.Data(
        S=S.long(),
        X=X,
        X_mask=X_mask,
        residue_index=residue_index,
        residue_mask=residue_mask,
        BB_D=BB_D,
        BB_D_sincos=BB_D_sincos,
        SC_D=SC_D,
    )

    return pippack_batch


def get_sidechain_logits(
    pippack: PIPPackFineTune, b: gd.Batch, recycles: int = 0
) -> torch.Tensor:
    nchi = pippack.n_chi_bins
    if pippack is not None:
        pippack_batch = batch_to_PIPPack_batch(b, pippack.k_neighbors)
        results = pippack.forward(pippack_batch, n_recycle=recycles)
        pippack_logits = results["chi_logits"]
    else:
        max_res = max([(b.aatype_batch == i).sum() for i in range(b.num_graphs)])
        pippack_logits = torch.ones(
            (b.num_graphs, max_res, 4, nchi + 1),
            dtype=torch.float32,
            device=b.x.device,
        )
    return pippack_logits


def apply_logits_to_proteins(
    proteins: List[Protein], logits: torch.Tensor, resample: bool = False
) -> List[Protein]:
    """Takes PIPPack logits and applies them to list of Protein objects"""
    # PIPPack has extra chi bin
    nchi = logits.shape[-1] - 1

    # Outer loop is over individual objects (batch dim)
    for obj_idx in range(len(proteins)):
        current_protein = proteins[obj_idx]
        # Don't apply PIPPack's logits to A, G, or X
        res_sel = np.logical_and(
            current_protein.aatype != rc.restype_order["A"],
            current_protein.aatype != rc.restype_order["G"],
        )
        res_sel = np.logical_and(res_sel, current_protein.aatype != rc.restype_num)
        res_idx = np.where(res_sel)[0]

        # Grab xyz and aatype from scaffold
        bb_xyz = current_protein.atom27_xyz[res_idx, :4, :]
        aatype = current_protein.aatype[res_idx]

        cur_logits = logits[obj_idx]

        # Get chi angles from cur_logits [L, 4, nchi + 1]
        chi_logits = cur_logits[res_idx, :, :nchi]  # [4, nchi]
        chi_probs = F.softmax(chi_logits, dim=-1)  # [4, nchi]
        _, chi_bins = torch.max(chi_probs, dim=-1)  # [4, ]
        chi_bins = chi_bins.detach().cpu().numpy()
        chi_angles = (chi_bins + 0.5) * 2 * np.pi / nchi - np.pi  # [4, ]
        # TODO: add offset from last pred column

        # Get chi mask from aatype
        chi_angle_mask = np.array(rc.chi_angles_mask)[aatype]

        # Retrieve full side chain xyz and mask
        atom14_xyz, atom14_mask = build_sc_from_chi(
            bb_xyz, aatype, chi_angles, chi_angle_mask
        )

        # Apply predicted side chains to scaffold atom positions and mask
        current_protein.atom27_xyz[res_idx, :14, :] = atom14_xyz
        current_protein.atom27_mask[res_idx, :14] = atom14_mask

        if resample:
            # Get the protein components.
            protein = {
                "S": torch.from_numpy(current_protein.aatype).long(),
                "X": torch.from_numpy(current_protein.atom27_xyz[:, :14]).float(),
                "X_mask": torch.from_numpy(current_protein.atom27_mask[:, :14]).float(),
                "BB_D": torch.from_numpy(
                    calc_bb_dihedrals(
                        current_protein.atom27_xyz[:, :14],
                        current_protein.residue_index,
                        return_mask=False,
                    )
                ).float(),
                "residue_index": torch.from_numpy(current_protein.residue_index).long(),
                "residue_mask": torch.from_numpy(
                    np.ones_like(current_protein.aatype)
                ).float(),
            }
            for k, v in protein.items():
                protein[k] = v.to(cur_logits.device)
            protein["chi_logits"] = cur_logits

            # Perform resampling
            resample_xyz, _, chi_angles = resample_loop(protein, protein["X"])
            current_protein.atom27_xyz[:, :14] = resample_xyz.cpu().numpy()
            chi_angles = chi_angles.cpu().numpy()

        proteins[obj_idx] = current_protein

    return proteins


def local_interresidue_sc_clash_loss(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clash_overlap_tolerance: float,  # OpenFold value is 1.5
    distance_threshold: float = 14.0,
    basis_atom: str = "CB",
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations resulting from sidechains.

    Note: This ignores intra-residue clashes and backbone-backbone clashes.
    """

    # Get needed components from batch.
    aatype = batch["S"].squeeze().clone()
    restype_atom14_to_atom37 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37, dtype=torch.long, device=aatype.device
    )
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    atom14_atom_exists = batch["X_mask"].squeeze().clone()
    residue_index = batch["residue_index"].squeeze().clone().long()
    residue_mask = batch["residue_mask"].squeeze().clone()
    atom14_pred_positions = atom14_pred_positions.squeeze().clone()

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [rc.van_der_waals_radius[name[0]] for name in rc.atom_types]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = atom14_atom_exists * atomtype_radius[residx_atom14_to_atom37]

    # Get the basis atom xyz for each residue.
    # shape (N, 3)
    if basis_atom == "CB":
        basis_atom_idx = 4 * torch.ones_like(aatype)
        basis_atom_idx[aatype == rc.restype_order["G"]] = 1
    else:
        basis_atom_idx = rc.atom_order[basis_atom] * torch.ones_like(aatype)
    basis_xyz = torch.gather(
        atom14_pred_positions,
        1,
        basis_atom_idx[..., None, None].expand(*atom14_pred_positions.shape),
    )[:, 0, :]

    # Determine distances based on basis atoms.
    # shape (N, N)
    basis_dists = torch.sqrt(
        eps + torch.sum((basis_xyz[None, :, :] - basis_xyz[:, None, :]) ** 2, dim=-1)
    )

    # Create the mask for valid residue pairs.
    # shape (N, N)
    fp_type = atom14_pred_positions.dtype
    dists_mask = (residue_mask[:, None] * residue_mask[None, :]).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (same residue pairs)
    dists_mask = dists_mask * (residue_index[:, None] < residue_index[None, :])

    # Determine which residue pairs are within the distance threshold.
    # shape (N, N)
    dists_lower_bound = distance_threshold * torch.ones_like(dists_mask)
    dists_mask = dists_mask * (basis_dists < dists_lower_bound)
    valid_pairs = torch.where(dists_mask)

    # Get the atom14 coordinates for the valid residue pairs.
    # shape (N_pairs, 14, 3)
    res1_atom14_xyz = atom14_pred_positions.squeeze().clone()[valid_pairs[0]]
    res2_atom14_xyz = atom14_pred_positions.squeeze().clone()[valid_pairs[1]]

    # Get the atomic distances for the valid residue pairs.
    # shape (N_pairs, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (res1_atom14_xyz[..., None, :] - res2_atom14_xyz[..., None, :, :]) ** 2,
            dim=-1,
        )
    )

    # Initialize the mask for the allowed distances.
    # shape (N_pairs, 14, 14)
    dists_mask = torch.ones_like(dists)

    # Backbone-backbone clashes are ignored. CB is included in the backbone.
    bb_bb_mask = torch.zeros_like(dists_mask)
    bb_bb_mask[..., :5, :5] = 1.0
    dists_mask = dists_mask * (1.0 - bb_bb_mask)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = aatype.new_tensor(cys_sg_idx)
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14)
    cys_res1 = aatype[valid_pairs[0]] == rc.restype_order["C"]
    cys_res2 = aatype[valid_pairs[1]] == rc.restype_order["C"]
    cys_mask = torch.logical_and(cys_res1, cys_res2)
    disulfide_bonds = cys_mask[..., None, None] * (
        cys_sg_one_hot[None, :, None] * cys_sg_one_hot[None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Mask interactions between side chain and backbone when atoms are separated by less than 4 bonds.
    # For all residues, ignore Cb_i - N_i+1 and C_i - Cb_i+1.
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14).type(fp_type)
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14).type(fp_type)
    cb_one_hot = F.one_hot(residue_index.new_tensor(4), num_classes=14).type(fp_type)
    neighbor_mask = (residue_index[valid_pairs[0]] + 1) == residue_index[valid_pairs[1]]
    cb_n_dists = (
        neighbor_mask[..., None, None]
        * cb_one_hot[None, :, None]
        * n_one_hot[None, None, :]
    )
    c_cb_dists = (
        neighbor_mask[..., None, None]
        * c_one_hot[None, :, None]
        * cb_one_hot[None, None, :]
    )
    dists_mask = dists_mask * (1.0 - cb_n_dists) * (1.0 - c_cb_dists)

    # For PRO at i+1, also ignore
    # C_i - Cg_i+1, C_i - Cd_i+1, O_i - Cd_i+1, and Ca_i - Cd_i+1.
    ca_one_hot = F.one_hot(residue_index.new_tensor(1), num_classes=14).type(fp_type)
    o_one_hot = F.one_hot(residue_index.new_tensor(3), num_classes=14).type(fp_type)
    pro = rc.restype_name_to_atom14_names["PRO"]
    pro_cg_idx = pro.index("CG")
    pro_cg_idx = residue_index.new_tensor(pro_cg_idx)
    pro_cg_one_hot = F.one_hot(pro_cg_idx, num_classes=14).type(fp_type)
    pro_cd_idx = pro.index("CD")
    pro_cd_idx = residue_index.new_tensor(pro_cd_idx)
    pro_cd_one_hot = F.one_hot(pro_cd_idx, num_classes=14).type(fp_type)
    pro_res2 = aatype[valid_pairs[1]] == rc.restype_order["P"]
    pro_neighbor_mask = pro_res2 * neighbor_mask  # [N_pairs]
    c_pro_cg_dists = (
        pro_neighbor_mask[..., None, None]
        * c_one_hot[None, :, None]
        * pro_cg_one_hot[None, None, :]
    )
    c_pro_cd_dists = (
        pro_neighbor_mask[..., None, None]
        * c_one_hot[None, :, None]
        * pro_cd_one_hot[None, None, :]
    )
    o_pro_cd_dists = (
        pro_neighbor_mask[..., None, None]
        * o_one_hot[None, :, None]
        * pro_cd_one_hot[None, None, :]
    )
    ca_pro_cd_dists = (
        pro_neighbor_mask[..., None, None]
        * ca_one_hot[None, :, None]
        * pro_cd_one_hot[None, None, :]
    )
    dists_mask = (
        dists_mask
        * (1.0 - c_pro_cg_dists)
        * (1.0 - c_pro_cd_dists)
        * (1.0 - o_pro_cd_dists)
        * (1.0 - ca_pro_cd_dists)
    )

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

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (eps + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.zeros_like(atom14_atom_exists)
    per_atom_loss_sum = per_atom_loss_sum.index_add(
        0, valid_pairs[0], torch.sum(dists_to_low_error, dim=2)
    )
    per_atom_loss_sum = per_atom_loss_sum.index_add(
        0, valid_pairs[1], torch.sum(dists_to_low_error, dim=1)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = (per_atom_loss_sum > 0.0).long()

    clash_info = {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

    return clash_info


def find_clashing_residues(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clash_overlap_tolerance: float = 0.6,
) -> torch.Tensor:
    # NOTE: This assumes that batch has only 1 protein in it.

    # Find the clashing atoms and energy
    clash_info = local_interresidue_sc_clash_loss(
        batch, atom14_pred_positions, clash_overlap_tolerance
    )
    atom_clash_mask = clash_info["per_atom_clash_mask"].squeeze()
    clash_energy = clash_info["mean_loss"].squeeze()

    # Get residue indices of residues that have at least one clashing atom
    clashing_residues = torch.unique(torch.where(atom_clash_mask)[0])
    return clashing_residues, clash_energy


def resample_clashes(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clashing_indices: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get X, S, BB_D, and chi_logits of clashing residues
    resampled_X = batch["X"].squeeze()[clashing_indices]
    resampled_S = batch["S"].squeeze()[clashing_indices]
    resampled_BB_D = batch["BB_D"].squeeze()[clashing_indices]
    resampled_chi_logits = batch["chi_logits"].squeeze()[clashing_indices]

    # Get resampled chi values
    if temperature > 0.0:
        logits = resampled_chi_logits / temperature
        chi_probs = F.softmax(logits, -1)
        chi_bin = (
            torch.multinomial(chi_probs.view(-1, logits.shape[-1]), 1)
            .view(*logits.shape[:2], -1)
            .squeeze(-1)
        )
    else:
        chi_bin = torch.argmax(F.softmax(resampled_chi_logits, -1), dim=-1)
    chi_bin_one_hot = F.one_hot(chi_bin, num_classes=resampled_chi_logits.shape[-1])
    chi_bin_rad = torch.cat(
        (
            torch.arange(
                -torch.pi,
                torch.pi,
                2 * torch.pi / (resampled_chi_logits.shape[-1] - 1),
                device=chi_bin.device,
            ),
            torch.tensor([0]).to(device=chi_bin.device),
        )
    )
    pred_chi_bin = torch.sum(
        chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1
    )
    chi_bin_offset = batch.get("chi_bin_offset", None)
    if chi_bin_offset is not None:
        bin_sample_update = chi_bin_offset.squeeze()[clashing_indices]
    else:
        # If None, set to middle of bin
        bin_sample_update = (2 * torch.pi / (resampled_chi_logits.shape[-1] - 1)) * 0.5
    chi_pred = pred_chi_bin + bin_sample_update

    # Construct resampled atom14 coordinates
    aatype_chi_mask = torch.tensor(
        rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device
    )[resampled_S]
    chi_pred = aatype_chi_mask * chi_pred
    resampled_atom14_xyz = get_atom14_coords(
        resampled_X, resampled_S, resampled_BB_D, chi_pred
    )

    # Update coordinate tensor
    resampled_coords = atom14_pred_positions.clone().squeeze()
    resampled_coords[clashing_indices] = resampled_atom14_xyz

    return resampled_coords, chi_pred


def resample_loop(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    sample_temp: float = 0.5,
    clash_overlap_tolerance: float = 0.6,
    max_iters: int = 10,
    metropolis_temp: float = 5e-6,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Find clashing residues and energy
    clashing_residues, clash_energy = find_clashing_residues(
        batch, atom14_pred_positions, clash_overlap_tolerance
    )
    if verbose:
        print(
            "Number of initial clashing residues to resample:", len(clashing_residues)
        )
        print("Initial clash energy:", clash_energy)

    # Resample clashing residues
    resampled_coords = atom14_pred_positions.clone()
    resampled_energy = clash_energy.clone()
    resampled_chis = calc_sc_dihedrals(atom14_pred_positions, batch["S"], False)
    resampled_iter = -1
    temp = sample_temp
    for i in range(max_iters):
        # If there are no violations, break
        if resampled_energy == 0.0:
            break

        # Resample clashes
        if i % 10 == 0:
            temp += 0.1
        temp_coords, temp_chi = resample_clashes(
            batch, resampled_coords, clashing_residues, temp
        )

        # Find new clashing residues and energy
        clashing_residues, clash_energy = find_clashing_residues(
            batch, temp_coords, clash_overlap_tolerance
        )
        if verbose:
            print(
                f"Number of clashing residues to resample (iter {i}):",
                len(clashing_residues),
            )
            print(f"Clash energy (iter {i}):", clash_energy)

        # Update bests based on Metropolis Criterion
        if clash_energy < resampled_energy:
            resampled_coords = temp_coords.clone()
            resampled_energy = clash_energy.clone()
            resampled_chis = temp_chi.clone()
            resampled_iter = i
        else:
            if torch.rand(1, device=atom14_pred_positions.device) < torch.exp(
                -(clash_energy - resampled_energy) / metropolis_temp
            ):
                if verbose:
                    print("Metropolis criterion accepted.")
                resampled_coords = temp_coords.clone()
                resampled_energy = clash_energy.clone()
                resampled_chis = temp_chi.clone()
                resampled_iter = i

    if verbose:
        print("Final energy:", resampled_energy)
        print("Final iteration:", resampled_iter)

    return resampled_coords, resampled_energy, resampled_chis
