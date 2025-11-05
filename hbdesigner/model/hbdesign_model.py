from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_scatter import scatter, scatter_max

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import (
    impute_CB,
    scatter_masked_mean,
)
from hbdesigner.model.layers import (
    MPNNLayer,
)
from hbdesigner.train.config import TrainConfig


class HBDesigner(nn.Module):
    """
    Sequence design model for HBDesigner. 
    
    Predicts positions and amino acids for putative network residues given an empty backbone.

    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.model.hbdesigner
        self.max_comps = c.max_res
        self.knn_k = c.knn_k
        self.n_atoms = 5

        # Layers for nodes
        self.bb_dihedral_linear = nn.Linear(6, c.pn_dim)
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

        # Decoding step conditioning
        self.net_res_num_range = c.max_res + 1
        self.net_res_num_linear = nn.Linear(self.net_res_num_range, c.pg_dim)
        self.cond2res_linears = nn.ModuleList(
            [
                nn.Linear(c.pn_dim + c.pg_dim, c.pn_dim)
                for _ in range(c.num_protein_encoder_layers)
            ]
        )

        # Guide atom conditioning
        self.guide_atom_rbf_linear = nn.Linear(c.num_rbf, c.pg_dim)
        self.guide_atom_rbf_norm = nn.LayerNorm(c.pg_dim)
        self.guide2res_linears = nn.ModuleList(
            [
                nn.Linear(c.pn_dim + c.pg_dim, c.pn_dim)
                for _ in range(c.num_protein_encoder_layers)
            ]
        )

        # Amino acid type conditioning
        self.seq_dist_linear = nn.Linear(rc.restype_num + 1, c.pn_dim)
        self.seq2res_linears = nn.ModuleList(
            [
                nn.Linear(c.pn_dim * 2, c.pn_dim)
                for _ in range(c.num_protein_encoder_layers)
            ]
        )

        # Output layers
        self.seq_layer = nn.Linear(c.pn_dim, rc.restype_num + 1)
        self.net_res_layer = nn.Linear(c.pn_dim, 1)

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
        atom_xyz = atom14_xyz[:, :5].clone()
        atom_mask = atom14_mask[:, :5].clone()

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

    def _form_guide_atom_feats(
        self,
        atom14_xyz: torch.Tensor,
        atom14_mask: torch.Tensor,
        guide_atom_xyz: torch.Tensor,
        aatype_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate RBFs between each Cb and the guide atom (node feature)."""
        c = self.cfg.model.hbdesigner

        # Impute Cb
        atom_xyz = atom14_xyz[..., :5, :]
        atom_mask = atom14_mask[..., :5]
        atom_xyz[..., 4, :] = impute_CB(
            atom_xyz[..., 0, :], atom_xyz[..., 1, :], atom_xyz[..., 2, :]
        )
        atom_mask[..., 4] = torch.prod(atom_mask[..., :3], dim=-1)
        cdists = torch.cdist(atom_xyz[..., 4, :], guide_atom_xyz)  # [L, B]
        cdists = cdists * atom_mask[..., 4:]

        # Mask out cross-sample distances
        sample_mask = F.one_hot(aatype_batch, cdists.shape[-1]).float()  # [L, B]
        cdists = (cdists * sample_mask).sum(-1)[:, None]  # [L, 1]
        rbf_mu = torch.linspace(0, 20, c.num_rbf)  # [N_RBF]
        rbf_mu = rbf_mu.to(atom14_xyz.device)
        rbf_sigma = 20 / c.num_rbf
        rbf = torch.exp(-1 * (cdists - rbf_mu) ** 2 / rbf_sigma**2)  # [L, N_RBF]

        # Featurize and norm
        nodes = self.guide_atom_rbf_linear(rbf)  # [L, pg_dim]
        nodes = self.guide_atom_rbf_norm(nodes)  # [L, pg_dim]
        return nodes

    def _get_conditioning_info(self, b: gd.Batch) -> Dict[str, torch.Tensor]:
        """Process conditioning information based on specified probabilities."""
        cond_info = {}
        c = self.cfg.model.hbdesigner

        # Encode number of network residues remaining
        net_res_num = F.one_hot(b.net_res_num, self.net_res_num_range).float()  # [B, N]
        net_res_num_cond = self.net_res_num_linear(net_res_num)  # [B, pg_dim]
        cond_info["net_res_num_nodes"] = net_res_num_cond.repeat_interleave(
            b.batch2res_repeats, dim=0
        )  # [L, pg_dim]

        # Encode and mask guide atom node features
        guide_atom_nodes = self._form_guide_atom_feats(
            b.atom14_xyz, b.atom14_mask, b.guide_atom_xyz, b.aatype_batch
        )  # [L, pg_dim]
        guide_atom_pct = torch.full_like(
            b.batch2res_repeats, fill_value=c.guide_atom_pct, dtype=torch.float32
        )  # [B]
        guide_atom_mask = torch.bernoulli(guide_atom_pct)  # [B]
        guide_atom_mask = guide_atom_mask.repeat_interleave(
            b.batch2res_repeats, dim=0
        )  # [L]
        cond_info["guide_atom_nodes"] = (
            guide_atom_nodes * guide_atom_mask[:, None]
        )  # [L, pg_dim]

        # Encode expected network sequence distribution
        seq_dist_nodes = self.seq_dist_linear(b.aatype_cond)  # [B, pn_dim]
        seq_dist_nodes = seq_dist_nodes.repeat_interleave(
            b.batch2res_repeats, dim=0
        )  # [L, pn_dim]
        cond_info["seq_dist_nodes"] = seq_dist_nodes

        return cond_info

    def forward(self, b: gd.Batch) -> Dict[str, torch.Tensor]:
        # Create initial embedding of protein nodes
        protein_nodes = self._form_protein_nodes(
            b.bb_dihedral,
            b.aatype,
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

        # Gather conditioning info
        cond_info = self._get_conditioning_info(b)

        # Pass through MPNN layers
        for i, layer in enumerate(self.mpnn_layers):
            # Residue count conditioning
            protein_nodes = protein_nodes + self.cond2res_linears[i](
                torch.cat(
                    [
                        protein_nodes,
                        cond_info["net_res_num_nodes"],
                    ],
                    dim=-1,
                )
            )
            # Guide atom conditioning
            protein_nodes = protein_nodes + self.guide2res_linears[i](
                torch.cat(
                    [
                        protein_nodes,
                        cond_info["guide_atom_nodes"],
                    ],
                    dim=-1,
                )
            )
            # Sequence conditioning
            protein_nodes = protein_nodes + self.seq2res_linears[i](
                torch.cat(
                    [
                        protein_nodes,
                        cond_info["seq_dist_nodes"],
                    ],
                    dim=-1,
                )
            )

            # Message passing
            protein_nodes, protein_edges = layer(
                protein_nodes, protein_edges, protein_edge_index
            )

        # Make position predictions
        net_res_logits = self.net_res_layer(protein_nodes)

        # Make sequence predictions
        seq_logits = self.seq_layer(protein_nodes)

        # Form results dictionary
        results_dict = {
            "net_res_logits": net_res_logits,
            "seq_logits": seq_logits,
        }

        return results_dict

    def compute_net_res_nll(
        self,
        nll_mask: torch.Tensor,  # [L,]
        done_mask: torch.Tensor,  # [L,]
        aatype_batch: torch.Tensor,  # [L,]
        net_res_logits: torch.Tensor,  # [L, 1]
    ) -> torch.Tensor:
        # Need to compute log_probs for each res and for stop
        # log_probs = log_softmax(logits)
        #           = logit - logsumexp(logits)
        #           = logit - logsumexp(logits - max(logits)) + max(logits)

        # Squeeze the inputs
        net_res_logits = net_res_logits.squeeze(1)

        # Determine max of logits per sample but make sure already predicted residues are not
        # affecting the max computation
        adj_logits = net_res_logits.clone()
        adj_logits[done_mask.bool()] = -torch.inf  # [L,]
        max_logit_batch = scatter_max(adj_logits, aatype_batch, dim=0)[0]  # [B,]

        # Subtract max_logit from each set of logits
        adj_net_res_logits = net_res_logits - max_logit_batch[aatype_batch]  # [L,]

        # Compute max_logit + logsumexp of adj_logits but make sure already predicted residues are not
        # affecting the logsumexp computation
        sumexp = scatter(
            (1 - done_mask) * adj_net_res_logits.exp(), aatype_batch, dim=0
        )  # [B,]
        logsumexp = sumexp.log() + max_logit_batch  # [B,]

        # Compute log_probs
        net_res_log_probs = net_res_logits - logsumexp[aatype_batch]  # [L,]

        # Create the ground-truth probability vectors.
        p_net_res = (
            nll_mask / scatter(nll_mask, aatype_batch, dim=0)[aatype_batch]
        )  # [L,]

        # Compute nll_loss = - (p * log_probs)
        net_res_nll_loss = -(p_net_res * net_res_log_probs)  # [L,]

        # Compute overall nll_loss
        nll_loss_batch = scatter_masked_mean(
            net_res_nll_loss,
            aatype_batch,
            nll_mask,
            dim=0,
        )  # [B,]
        return torch.nan_to_num(nll_loss_batch)

    def compute_seq_nll(
        self,
        aatype: torch.Tensor,
        aatype_batch: torch.Tensor,
        nll_mask: torch.Tensor,
        seq_logits: torch.Tensor,
    ) -> torch.Tensor:
        residue_type_one_hot = F.one_hot(aatype, rc.restype_num + 1).float()  # [L, 20]

        # Do label smoothing, if enabled
        residue_type_one_hot += self.cfg.model.hbdesigner.nll_smoothing / float(
            residue_type_one_hot.size(-1)
        )
        residue_type_one_hot /= residue_type_one_hot.sum(-1, keepdim=True)

        seq_log_probs = F.log_softmax(seq_logits, 1)  # [L, 20]

        if self.cfg.model.hbdesigner.loss_type == "focal":
            # Focal loss upweights loss from less confident preds and vice versa
            log_probs_t = (residue_type_one_hot * seq_log_probs).sum(1)
            probs_t = torch.exp(log_probs_t)
            gamma = self.cfg.model.hbdesigner.focal_gamma
            nll = -1 * ((1 - probs_t) ** gamma) * log_probs_t
        else:
            nll = -1 * (residue_type_one_hot * seq_log_probs).sum(1)  # [L,]

        seq_nll = scatter_masked_mean(
            nll,
            aatype_batch,
            nll_mask,
            dim=0,
        )  # [B,]

        return torch.nan_to_num(seq_nll)

    @torch.no_grad()
    def _compute_other_metrics(
        self, loss_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute training metrics not involved in loss calculation."""

        metric_dict = {}

        # Perplexity for each pred task
        metric_dict["net_res_perp_batch"] = torch.clamp(
            torch.exp(loss_dict["net_res_nll_batch"]), max=100.0
        )
        metric_dict["net_res_perp"] = torch.clamp(
            torch.exp(loss_dict["net_res_nll"]), max=100.0
        )
        metric_dict["seq_perp_batch"] = torch.clamp(
            torch.exp(loss_dict["seq_nll_batch"]), max=100.0
        )
        metric_dict["seq_perp"] = torch.clamp(
            torch.exp(loss_dict["seq_nll"]), max=100.0
        )

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

        # Network residue prediction
        loss_dict["net_res_nll_batch"] = self.compute_net_res_nll(
            b.nll_mask,
            b.done_mask,
            b.aatype_batch,
            results_dict["net_res_logits"],
        )
        loss_dict["net_res_nll"] = torch.mean(loss_dict["net_res_nll_batch"])

        if c.net_res_nll_weight > 0:
            loss += c.net_res_nll_weight * loss_dict["net_res_nll"]

        # Sequence prediction
        loss_dict["seq_nll_batch"] = self.compute_seq_nll(
            b.aatype_gt,
            b.aatype_batch,
            b.nll_mask,
            results_dict["seq_logits"],
        )
        loss_dict["seq_nll"] = torch.mean(loss_dict["seq_nll_batch"])

        if c.seq_nll_weight > 0:
            loss += c.seq_nll_weight * loss_dict["seq_nll"]

        # Compute other metrics
        loss_dict.update(self._compute_other_metrics(loss_dict))
        return loss, loss_dict

    def sample_res(
        self,
        net_res_logits: torch.Tensor,
        aatype_batch: torch.Tensor,
        nll_mask: torch.Tensor,
        temp: float = 1.0,
    ) -> torch.Tensor:
        # Uniform noise
        net_res_u = torch.rand(
            net_res_logits.shape, device=aatype_batch.device
        )  # [L, 1]

        # Gumbel noise
        net_res_gumbel = net_res_logits / temp - (-net_res_u.log()).log()  # [L, 1]

        # Mask out visible residues
        net_res_gumbel[nll_mask.bool()] = -torch.inf

        # Take max for each sample
        _, net_res_argmax = scatter_max(net_res_gumbel, aatype_batch, 0)  # [B, 1]

        # Squeeze the inputs
        net_res_logits = net_res_logits.squeeze(1)

        # Determine max of logits per sample but make sure already predicted residues are not
        # affecting the max computation
        adj_logits = net_res_logits.clone()
        adj_logits[nll_mask.bool()] = -torch.inf  # [L,]
        max_logit_batch = scatter_max(adj_logits, aatype_batch, dim=0)[0]  # [B,]

        # Subtract max_logit from each set of logits
        adj_net_res_logits = net_res_logits - max_logit_batch[aatype_batch]  # [L,]

        # Compute max_logit + logsumexp of adj_logits but make sure already predicted residues are not
        # affecting the logsumexp computation
        sumexp = scatter(
            (1 - nll_mask) * adj_net_res_logits.exp(), aatype_batch, dim=0
        )  # [B,]
        logsumexp = sumexp.log() + max_logit_batch  # [B,]

        # Compute log_probs
        net_res_log_probs = net_res_logits - logsumexp[aatype_batch]  # [L,]
        net_res_probs = torch.exp(net_res_log_probs)

        net_res_probs = net_res_probs[net_res_argmax].squeeze(-1)  # [B]
        return net_res_argmax, net_res_probs

    def sample_seq(
        self,
        seq_logits: torch.Tensor,
        temp: float = 1.0,
    ) -> torch.Tensor:
        # Uniform noise
        seq_u = torch.rand(seq_logits.shape, device=seq_logits.device)  # [L, 21]

        # Gumbel noise
        seq_gumbel = seq_logits / temp - (-seq_u.log()).log()  # [L, 21]

        # Prohibit nonpolars
        nonpolars = torch.tensor(rc.restype_non_hb_idx).to(seq_gumbel.device)
        seq_gumbel[:, nonpolars] = -torch.inf

        # Get argmax of last dim
        seq_argmax = torch.argmax(seq_gumbel, dim=-1)

        # Collect pred seq prob
        res_mask = torch.tensor(
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            dtype=torch.bool,
        ).to(seq_logits.device)
        seq_logits = seq_logits[:, res_mask]
        seq_log_probs = F.log_softmax(seq_logits, 1)  # [L, 20]
        residue_type_one_hot = F.one_hot(
            seq_argmax, rc.restype_num + 1
        ).float()  # [L, 20]
        residue_type_one_hot = residue_type_one_hot[:, res_mask]
        seq_log_probs = (residue_type_one_hot * seq_log_probs).sum(1)  # [L,]
        seq_probs = torch.exp(seq_log_probs)
        return seq_argmax, seq_probs

    @torch.no_grad()
    def sample(
        self,
        b: gd.Batch,
        res_sample_temp: float = 0.1,
        seq_sample_temp: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Currently used by Trainer.sample_batch in seq design mode"""

        # Clear all seq and sidechain info
        b.aatype[:] = rc.restype_num
        b.atom14_xyz[:, 4:, :] = 0.0
        b.atom14_mask[:, 4:] = 0.0

        # Revert masks back to start of decoding
        net_sizes = (
            scatter(b.nll_mask, b.aatype_batch, dim=0)
            + scatter(b.done_mask, b.aatype_batch, dim=0)
        ).long()
        b.nll_mask[:] = 0
        b.done_mask[:] = 0
        b.net_res_num = net_sizes

        # Create initial sequence
        seq = (rc.restype_num * torch.ones_like(b.aatype)).to(torch.long)  # [B]
        net_res_probs = torch.zeros_like(b.aatype).to(torch.float32)  # [B]
        seq_probs = torch.zeros_like(b.aatype).to(torch.float32)  # [B]

        # Calculate allowed counts of each aatype based on cond info
        # aatype_cond_counts tracks valid aatypes
        # real_counts tracks strict/ambiguous restype accounting
        aatype_cond_counts = b.aatype_cond * b.net_res_num[:, None]  # [B, 21]
        polars = torch.tensor(rc.restype_hb_idx).to(b.x.device)
        real_counts = torch.zeros_like(aatype_cond_counts)
        real_counts[aatype_cond_counts >= 1] += aatype_cond_counts[
            aatype_cond_counts >= 1
        ].round()
        aatype_cond_counts[aatype_cond_counts >= 1] -= real_counts[
            aatype_cond_counts >= 1
        ]
        n_unk = torch.sum(aatype_cond_counts, dim=-1)
        aatype_cond_counts = torch.clone(real_counts)
        aatype_cond_counts[:, rc.restype_hb_idx] += n_unk[:, None]
        real_counts[:, -1] += n_unk

        # Define main function to embed and process the protein for each step
        def get_processed_node_embeddings(b, seq, cond_info):
            # Create initial embedding of protein nodes
            protein_nodes = self._form_protein_nodes(
                b.bb_dihedral,
                seq,
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
                # Add conditioning info to protein nodes
                protein_nodes = protein_nodes + self.cond2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["net_res_num_nodes"].repeat_interleave(
                                b.batch2res_repeats, dim=0
                            ),
                        ],
                        dim=-1,
                    )
                )

                # Guide atom conditioning
                protein_nodes = protein_nodes + self.guide2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["guide_atom_nodes"],
                        ],
                        dim=-1,
                    )
                )

                # Guide sequence conditioning
                protein_nodes = protein_nodes + self.seq2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["seq_dist_nodes"],
                        ],
                        dim=-1,
                    )
                )

                # Message passing
                protein_nodes, protein_edges = layer(
                    protein_nodes, protein_edges, protein_edge_index
                )
            return protein_nodes

        cond_info = self._get_conditioning_info(b)

        # Make predictions until model predicts each example is done.
        done = [False] * b.num_graphs
        while sum(done) < b.num_graphs:
            # Generate net res condition for current setting
            net_res_num = F.one_hot(
                b.net_res_num, self.net_res_num_range
            ).float()  # [B, N]
            net_res_num_cond = self.net_res_num_linear(net_res_num)  # [B, pg_dim]
            cond_info["net_res_num_nodes"] = net_res_num_cond

            # Get processed node embeddings for current step
            protein_nodes = get_processed_node_embeddings(
                b,
                seq,
                cond_info,
            )  # [L, pn_dim]

            # If graph is already done, don't decode its nodes
            net_res_mask = b.net_res_num > 0

            # Make res prediction
            net_res_logits = self.net_res_layer(protein_nodes)
            net_res, net_res_p = self.sample_res(
                net_res_logits, b.aatype_batch, b.done_mask, res_sample_temp
            )
            net_res = net_res.squeeze(-1)
            net_res = net_res[net_res_mask]
            net_res_probs[net_res] = net_res_p[net_res_mask]

            # Make seq prediction
            seq_logits = self.seq_layer(protein_nodes[net_res])

            # For any samples w/seq cond, mask out all other logits to ensure correct aatype chosen
            aatypes_not_allowed = (aatype_cond_counts <= 0.0).to(bool)
            seq_logits[aatypes_not_allowed[net_res_mask]] = -torch.inf

            seq_pred, seq_p = self.sample_seq(
                seq_logits,
                seq_sample_temp,
            )

            # Update seq and done_mask
            seq[net_res] = seq_pred
            b.done_mask[net_res] = 1.0
            seq_probs[net_res] = seq_p

            # One less res to predict
            b.net_res_num -= 1
            b.net_res_num = torch.clamp(b.net_res_num, min=0)

            # Update res-to-predict vector for each graph
            graphs_changed = b.aatype_batch[net_res]
            for i in range(b.num_graphs):
                # Skip finished graphs
                if net_res_mask[i]:
                    r = seq[net_res][graphs_changed == i].item()
                    # if not a "strict" requirement, remove one UNK token
                    if real_counts[i, r] < 1.0:
                        aatype_cond_counts[i, polars] -= 1
                        real_counts[i, -1] -= 1
                    # else, remove just the restype token
                    else:
                        aatype_cond_counts[i, r] -= 1
                        real_counts[i, r] -= 1

            # Update seq dist feature with new conditioning
            b.aatype_cond = aatype_cond_counts / (b.net_res_num[:, None] + 1e-8)
            b.aatype_cond /= torch.sum(b.aatype_cond, axis=-1)[:, None] + 1e-8
            seq_dist_nodes = self.seq_dist_linear(b.aatype_cond)  # [B, pn_dim]
            seq_dist_nodes = seq_dist_nodes.repeat_interleave(
                b.batch2res_repeats, dim=0
            )  # [L, pn_dim]

            is_stop = b.net_res_num < 1
            # Handle samples that need to stop
            if is_stop.sum() > 0:
                stop_idx = torch.where(is_stop)[0]
                for i in stop_idx:
                    done[i] = True

        # Create the results dictionary
        results_dict = {}
        for i in range(b.num_graphs):
            seq_i = seq[b.aatype_batch == i]
            seq_i_mask = seq_i != rc.restype_num
            results_dict[i] = {
                "net_res": torch.where(seq_i_mask)[0],
                "seq": seq_i[seq_i_mask],
                "net_res_probs": net_res_probs[b.aatype_batch == i][seq_i_mask],
                "seq_probs": seq_probs[b.aatype_batch == i][seq_i_mask],
            }
        return results_dict

    @torch.no_grad()
    def sample_new(
        self,
        b: gd.Batch,
        res_sample_temp: float = 0.1,
        seq_sample_temp: float = 0.1,
        bb_noise: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Currently used by inference_hbdesigner
        Intended as sample fxn for actual inference operation.

        - Does less housekeeping than original sample fxn.
        - Assumes batch is already configured (seq empty, network sizes set, etc)."""

        if bb_noise > 0.0:
            b.atom14_xyz = b.atom14_xyz + (bb_noise * torch.randn_like(b.atom14_xyz))
            b.atom14_xyz = b.atom14_xyz * b.atom14_mask[..., None]

        # Define starting seq
        seq = b.aatype_masked

        # For confidence prediction
        net_res_probs = torch.zeros_like(b.aatype).to(torch.float32)  # [B]
        seq_probs = torch.zeros_like(b.aatype).to(torch.float32)  # [B]

        # Collect aatype cond counts for record keeping
        aatype_cond_counts = b.aatype_cond * b.net_res_num[:, None]  # [B, 21]
        real_counts = torch.zeros_like(aatype_cond_counts)
        real_counts[aatype_cond_counts >= 1] += aatype_cond_counts[
            aatype_cond_counts >= 1
        ].round()
        aatype_cond_counts[aatype_cond_counts >= 1] -= real_counts[
            aatype_cond_counts >= 1
        ]
        n_unk = torch.sum(aatype_cond_counts, dim=-1)
        aatype_cond_counts = torch.clone(real_counts)
        aatype_cond_counts[:, rc.restype_hb_idx] += n_unk[:, None]
        real_counts[:, -1] += n_unk
        results_dict = {}

        # Define main function to embed and process the protein for each step
        def get_processed_node_embeddings(b, seq, cond_info):
            # Create initial embedding of protein nodes
            protein_nodes = self._form_protein_nodes(
                b.bb_dihedral,
                seq,
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
                # Add conditioning info to protein nodes
                protein_nodes = protein_nodes + self.cond2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["net_res_num_nodes"].repeat_interleave(
                                b.batch2res_repeats, dim=0
                            ),
                        ],
                        dim=-1,
                    )
                )

                # Guide atom conditioning
                protein_nodes = protein_nodes + self.guide2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["guide_atom_nodes"],
                        ],
                        dim=-1,
                    )
                )

                # Guide sequence conditioning
                protein_nodes = protein_nodes + self.seq2res_linears[i](
                    torch.cat(
                        [
                            protein_nodes,
                            cond_info["seq_dist_nodes"],
                        ],
                        dim=-1,
                    )
                )

                # Message passing
                protein_nodes, protein_edges = layer(
                    protein_nodes, protein_edges, protein_edge_index
                )
            return protein_nodes

        cond_info = self._get_conditioning_info(b)

        # Make predictions until model predicts each example is done.
        done = [False] * b.num_graphs
        while sum(done) < b.num_graphs:
            # Generate net res condition for current setting
            net_res_num = F.one_hot(
                b.net_res_num, self.net_res_num_range
            ).float()  # [B, N]
            net_res_num_cond = self.net_res_num_linear(net_res_num)  # [B, pg_dim]
            cond_info["net_res_num_nodes"] = net_res_num_cond

            # Get processed node embeddings for current step
            protein_nodes = get_processed_node_embeddings(
                b,
                seq,
                cond_info,
            )  # [L, pn_dim]

            # If graph is already done, don't decode its nodes
            net_res_mask = b.net_res_num > 0

            # Make res prediction
            net_res_logits = self.net_res_layer(protein_nodes)
            # Mask out fixed positions
            net_res_logits[~b.des_mask[..., None]] = -torch.inf

            # Sample positions
            net_res, net_res_p = self.sample_res(
                net_res_logits,
                b.aatype_batch,
                b.done_mask,
                res_sample_temp,
            )
            net_res = net_res.squeeze(-1)
            net_res = net_res[net_res_mask]
            net_res_probs[net_res] = net_res_p[net_res_mask]

            # Make seq prediction
            seq_logits = self.seq_layer(protein_nodes[net_res])

            # For any samples w/seq cond, mask out all other logits to ensure correct aatype chosen
            aatypes_not_allowed = (aatype_cond_counts <= 1e-4).to(bool)
            seq_logits[aatypes_not_allowed[net_res_mask]] = -torch.inf

            seq_pred, seq_p = self.sample_seq(
                seq_logits,
                seq_sample_temp,
            )

            # Update seq and done_mask
            seq[net_res] = seq_pred
            b.done_mask[net_res] = 1.0
            seq_probs[net_res] = seq_p

            # One less res to predict
            b.net_res_num -= 1  # only 1 if asymmtric
            b.net_res_num = torch.clamp(b.net_res_num, min=0)

            # Update res-to-predict vector for each graph
            graphs_changed = b.aatype_batch[net_res]
            for i in range(b.num_graphs):
                # Skip finished graphs
                if net_res_mask[i]:
                    r_all = seq[net_res][graphs_changed == i]
                    for r in r_all:
                        # Check if ambiguous or strict residue
                        if real_counts[i, r] < 1.0:
                            aatype_cond_counts[i, rc.restype_hb_idx] -= 1
                            real_counts[i, -1] -= 1
                        else:
                            aatype_cond_counts[i, r] -= 1
                            real_counts[i, r] -= 1

            # Update seq dist feature with new conditioning
            b.aatype_cond = aatype_cond_counts / (b.net_res_num[:, None] + 1e-8)

            seq_dist_nodes = self.seq_dist_linear(b.aatype_cond)  # [B, pn_dim]
            seq_dist_nodes = seq_dist_nodes.repeat_interleave(
                b.batch2res_repeats, dim=0
            )  # [L, pn_dim]

            is_stop = b.net_res_num < 1
            # Handle samples that need to stop
            if is_stop.sum() > 0:
                stop_idx = torch.where(is_stop)[0]
                for i in stop_idx:
                    done[i] = True

        # Create the results dictionary
        for i in range(b.num_graphs):
            seq_i = seq[b.aatype_batch == i]
            seq_i_mask = seq_i != rc.restype_num
            results_dict[i] = {
                "net_res": torch.where(seq_i_mask)[0],
                "seq": seq_i[seq_i_mask],
                "net_res_probs": net_res_probs[b.aatype_batch == i][seq_i_mask],
                "seq_probs": seq_probs[b.aatype_batch == i][seq_i_mask],
            }

        return results_dict


def load_HBDesigner(
    cfg: TrainConfig,
    ckpt: str,
    device: str = "cuda",
) -> HBDesigner:
    # Load pre-trained weights
    ckpt = torch.load(ckpt, map_location="cpu")

    model = HBDesigner(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    return model
