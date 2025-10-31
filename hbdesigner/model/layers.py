from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_scatter import scatter
from torch.utils.checkpoint import checkpoint

import proteingfn.data.residue_constants as rc
from proteingfn.data.features import impute_CB
from proteingfn.data.rigid_utils import Rigid


def get_act_fxn(act: str) -> Callable:
    if act == "relu":
        return F.relu
    elif act == "gelu":
        return F.gelu
    elif act == "elu":
        return F.elu
    elif act == "selu":
        return F.selu
    elif act == "celu":
        return F.celu
    elif act == "leaky_relu":
        return F.leaky_relu
    elif act == "prelu":
        return F.prelu
    elif act == "silu":
        return F.silu
    elif act == "sigmoid":
        return nn.Sigmoid()


class MLP(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_inter: int,
        num_out: int,
        num_layers: int,
        act: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)
        self.W_inter = nn.ModuleList(
            [nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)]
        )
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)

        # Activation function
        self.act = get_act_fxn(act)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Embed inputs with input layer
        X = self.act(self.W_in(X))

        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))

        # Get output from output layer
        X = self.W_out(X)

        return X


class IPMPConv(gnn.MessagePassing):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        e_out_dim: int,
        num_mlp_layers: int = 3,
        num_points: int = 8,
        act: str = "relu",
    ) -> None:
        super().__init__(aggr="mean")

        self.num_points = num_points
        self.act = get_act_fxn(act)

        # Point and message functions
        self.point_fn = nn.Linear(n_in_dim, num_points * 3)
        self.node_message_func = MLP(
            2 * n_in_dim
            + e_in_dim
            + 2 * num_points * 3
            + 2 * num_points
            + num_points**2,
            n_out_dim,
            n_out_dim,
            num_mlp_layers,
            act=act,
        )
        self.edge_message_func = MLP(
            2 * n_in_dim
            + e_in_dim
            + 2 * num_points * 3
            + 2 * num_points
            + num_points**2,
            e_out_dim,
            e_out_dim,
            num_mlp_layers,
            act=act,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        bb_frames: Rigid,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute node and edge messages
        node_message = self.propagate(edge_index, n=nodes, F=bb_frames, edge_attr=edges)
        edge_message = self.edge_updater(
            edge_index, n=nodes, F=bb_frames, edge_attr=edges
        )

        return node_message, edge_message

    def _get_message_in(
        self,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        F_i: Rigid,
        F_j: Rigid,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Get node i's local points and their norms
        p_local_i = self.point_fn(n_i).view(-1, self.num_points, 3)
        p_local_i_norm = torch.sqrt(torch.sum(p_local_i**2, dim=-1) + 1e-6)

        # Get node j's local points in i's frame and their norms
        p_local_j = self.point_fn(n_j).view(-1, self.num_points, 3)
        p_local_j = F_i[0].invert_apply(F_j[0].apply(p_local_j))
        p_local_j_norm = torch.sqrt(torch.sum(p_local_j**2, dim=-1) + 1e-6)

        # Get the distances between node i's local points and node j's local points in i's frame
        p_local_dists = torch.sqrt(
            torch.sum(
                (p_local_i[..., None, :] - p_local_j[..., None, :, :]) ** 2, dim=-1
            )
            + 1e-6
        )

        # Compute message
        message_in = torch.cat(
            [
                n_i,
                n_j,
                edge_attr,
                p_local_i.view(-1, self.num_points * 3),
                p_local_i_norm,
                p_local_j.view(-1, self.num_points * 3),
                p_local_j_norm,
                p_local_dists.view(-1, self.num_points**2),
            ],
            dim=-1,
        )

        return message_in

    def message(
        self,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        F_i: Rigid,
        F_j: Rigid,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Compute node message
        message_in = self._get_message_in(n_i, n_j, F_i, F_j, edge_attr)
        message = self.node_message_func(message_in)

        return message

    def edge_update(
        self,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        F_i: Rigid,
        F_j: Rigid,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Compute edge message
        message_in = self._get_message_in(n_i, n_j, F_i, F_j, edge_attr)
        message = self.edge_message_func(message_in)

        return message


class IPMPLayer(nn.Module):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        e_out_dim: int,
        num_mlp_layers: int = 3,
        num_points: int = 8,
        act: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = IPMPConv(
            n_in_dim, n_out_dim, e_in_dim, e_out_dim, num_mlp_layers, num_points, act
        )
        self.node_norm = nn.ModuleList([nn.LayerNorm(n_out_dim) for _ in range(2)])
        self.node_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.node_ff = MLP(n_out_dim, n_out_dim, n_out_dim, num_mlp_layers, act=act)
        self.edge_norm = nn.ModuleList([nn.LayerNorm(e_out_dim) for _ in range(2)])
        self.edge_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.edge_ff = MLP(e_out_dim, e_out_dim, e_out_dim, num_mlp_layers, act=act)

    def forward(
        self,
        nodes: torch.Tensor,
        bb_frames: Rigid,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute node and edge messages
        node_message, edge_message = self.conv(nodes, bb_frames, edges, edge_index)

        # Update node and edge features
        nodes = self.node_norm[0](nodes + self.node_dropout[0](node_message))
        nodes = self.node_norm[1](nodes + self.node_dropout[1](self.node_ff(nodes)))

        edges = self.edge_norm[0](edges + self.edge_dropout[0](edge_message))
        edges = self.edge_norm[1](edges + self.edge_dropout[1](self.edge_ff(edges)))

        return nodes, edges


class MPNNConv(gnn.MessagePassing):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        e_out_dim: int,
        num_mlp_layers: int = 3,
        act: str = "relu",
    ) -> None:
        super().__init__(aggr="mean")

        self.act = get_act_fxn(act)

        # Message functions
        self.node_message_func = MLP(
            2 * n_in_dim + e_in_dim, n_out_dim, n_out_dim, num_mlp_layers, act=act
        )
        self.edge_message_func = MLP(
            2 * n_in_dim + e_in_dim, e_out_dim, e_out_dim, num_mlp_layers, act=act
        )

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute node and edge messages
        node_message = self.propagate(edge_index, n=nodes, edge_attr=edges)
        edge_message = self.edge_updater(edge_index, n=nodes, edge_attr=edges)

        return node_message, edge_message

    def message(
        self, n_i: torch.Tensor, n_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Compute node message
        message_in = torch.cat([n_i, n_j, edge_attr], dim=-1)
        message = self.node_message_func(message_in)

        return message

    def edge_update(
        self, n_i: torch.Tensor, n_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Compute edge message
        message_in = torch.cat([n_i, n_j, edge_attr], dim=-1)
        message = self.edge_message_func(message_in)

        return message


class MPNNLayer(nn.Module):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        e_out_dim: int,
        num_mlp_layers: int = 3,
        act: str = "relu",
        dropout: float = 0.1,
        mlp_inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Set default value to input dimension
        if mlp_inner_dim is None:
            mlp_inner_dim = n_in_dim

        self.conv = MPNNConv(
            n_in_dim, n_out_dim, e_in_dim, e_out_dim, num_mlp_layers, act
        )
        self.node_norm = nn.ModuleList([nn.LayerNorm(n_out_dim) for _ in range(2)])
        self.node_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.node_ff = MLP(n_out_dim, mlp_inner_dim, n_out_dim, num_mlp_layers, act=act)
        self.edge_norm = nn.ModuleList([nn.LayerNorm(e_out_dim) for _ in range(2)])
        self.edge_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.edge_ff = MLP(e_out_dim, mlp_inner_dim, e_out_dim, num_mlp_layers, act=act)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        # Compute node and edge messages
        node_message, edge_message = self.conv(nodes, edges, edge_index)

        # Update node and edge features
        nodes = self.node_norm[0](nodes + self.node_dropout[0](node_message))
        nodes = self.node_norm[1](nodes + self.node_dropout[1](self.node_ff(nodes)))

        edges = self.edge_norm[0](edges + self.edge_dropout[0](edge_message))
        edges = self.edge_norm[1](edges + self.edge_dropout[1](self.edge_ff(edges)))

        return nodes, edges


class GraphCommunicationLayer(nn.Module):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        g_in_dim: int,
        g_out_dim: int,
        num_mlp_layers: int = 3,
        act: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Communication layers
        self.node_to_graph = nn.Linear(n_in_dim + g_in_dim, g_out_dim)
        self.graph_to_node = nn.Linear(n_in_dim + g_in_dim, n_out_dim)

        # Transition layers
        self.node_norm = nn.ModuleList([nn.LayerNorm(n_out_dim) for _ in range(2)])
        self.node_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.node_ff = MLP(n_out_dim, n_out_dim, n_out_dim, num_mlp_layers, act=act)
        self.graph_norm = nn.ModuleList([nn.LayerNorm(g_out_dim) for _ in range(2)])
        self.graph_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.graph_ff = MLP(g_out_dim, g_out_dim, g_out_dim, num_mlp_layers, act=act)

    def _get_graph_edges(
        self, nodes: torch.Tensor, graph_nodes: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        _, node_counts = torch.unique(batch, return_counts=True)
        graph_edges = graph_nodes.repeat_interleave(node_counts, dim=0)
        graph_edges = torch.cat([nodes, graph_edges], dim=-1)

        return graph_edges

    def forward(
        self, nodes: torch.Tensor, graph_nodes: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build edge information for graph communication
        graph_edges = self._get_graph_edges(nodes, graph_nodes, batch)

        # Update graph features
        graph_message = self.node_to_graph(graph_edges)
        graph_message = scatter(graph_message, batch, dim=0, reduce="mean")
        graph_nodes = self.graph_norm[0](
            graph_nodes + self.graph_dropout[0](graph_message)
        )
        graph_nodes = self.graph_norm[1](
            graph_nodes + self.graph_dropout[1](self.graph_ff(graph_nodes))
        )

        # Update node features
        graph_edges = self._get_graph_edges(nodes, graph_nodes, batch)
        node_message = self.graph_to_node(graph_edges)
        nodes = self.node_norm[0](nodes + self.node_dropout[0](node_message))
        nodes = self.node_norm[1](nodes + self.node_dropout[1](self.node_ff(nodes)))

        return nodes, graph_nodes


class ProteinEncoder(nn.Module):
    def __init__(
        self,
        n_hidden_dim: int,
        e_hidden_dim: int,
        g_hidden_dim: int,
        knn_k: int = 30,
        max_seq_sep: int = 32,
        num_rbf: int = 16,
        num_mpnn_layers: int = 3,
        num_mlp_layers: int = 3,
        num_ipmp_points: int = 8,
        dropout: float = 0.1,
        use_imputed_CB_for_edge_rbfs: bool = True,
    ) -> None:
        super().__init__()

        # Module attributes
        self.n_hidden_dim = n_hidden_dim
        self.e_hidden_dim = e_hidden_dim
        g_in_dim = n_hidden_dim
        self.knn_k = knn_k
        self.max_seq_sep = max_seq_sep
        self.num_rbf = num_rbf
        self.n_restype = rc.restype_num + 1  # includes UNK token
        self.use_imputed_CB_for_edge_rbfs = use_imputed_CB_for_edge_rbfs

        # Node feature layers
        self.bb_dihedral_linear = nn.Linear(6, n_hidden_dim)
        self.sc_dihedral_linear = nn.Linear(8, n_hidden_dim)
        self.seq_linear = nn.Linear(self.n_restype, n_hidden_dim)
        self.node_norm = nn.LayerNorm(n_hidden_dim)
        self.des_res_linear = nn.Linear(2, n_hidden_dim)

        # Edge feature layers
        self.rbf_linear = nn.Linear(14**2 * self.num_rbf, e_hidden_dim)
        self.seq_sep_linear = nn.Linear(2 * self.max_seq_sep + 2, e_hidden_dim)
        self.edge_norm = nn.LayerNorm(e_hidden_dim)

        # Graph feature layers
        self.graph_init = nn.Linear(g_in_dim, g_hidden_dim)
        self.graph_norm = nn.LayerNorm(g_hidden_dim)

        self.graph_comms = nn.ModuleList(
            [
                GraphCommunicationLayer(
                    n_hidden_dim,
                    n_hidden_dim,
                    g_hidden_dim,
                    g_hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    act="relu",
                    dropout=dropout,
                )
                for _ in range(num_mpnn_layers + 1)
            ]
        )
        self.ipmp_layers = nn.ModuleList(
            [
                IPMPLayer(
                    n_hidden_dim,
                    n_hidden_dim,
                    e_hidden_dim,
                    e_hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    num_points=num_ipmp_points,
                    act="relu",
                    dropout=dropout,
                )
                for _ in range(num_mpnn_layers)
            ]
        )

    def get_knn_edges(self, batch: gd.Batch) -> torch.Tensor:
        """Computes k-nearest neighbor graph for protein based on CA atom positions"""

        # Get CA atom positions
        ca_xyz = batch.atom14_xyz[..., 1, :]

        # Compute k-nearest neighbor graph
        edge_index = gnn.knn_graph(ca_xyz, self.knn_k, batch=batch.batch, loop=True)

        return edge_index

    def form_nodes(
        self, batch: gd.Batch, scaffold: bool = True, fix_res: bool = False
    ) -> torch.Tensor:
        """Forms initial nodes for protein"""

        # Initialize nodes
        nodes = batch.bb_dihedral.new_zeros(
            (batch.bb_dihedral.shape[0], self.n_hidden_dim)
        )

        # Sin-cos encoding of backbone dihedrals
        bb_dihedral_sincos = torch.stack(
            [torch.sin(batch.bb_dihedral), torch.cos(batch.bb_dihedral)], dim=-1
        ).view(-1, 6)
        nodes += self.bb_dihedral_linear(bb_dihedral_sincos)

        # Designable residue encoding
        if fix_res:
            des_res = F.one_hot(batch.designable_res_mask, 2).float()
            nodes += self.des_res_linear(des_res)

        if not scaffold:
            # Sin-cos encoding of sidechain dihedrals
            sc_dihedral_sincos = torch.stack(
                [torch.sin(batch.sc_dihedral), torch.cos(batch.sc_dihedral)], dim=-1
            ).view(-1, 8)
            nodes += self.sc_dihedral_linear(sc_dihedral_sincos)

            # One-hot encoding of sequence (includes UNK token)
            seq_onehot = F.one_hot(batch.aatype, 21).float()
            nodes += self.seq_linear(seq_onehot)

        # Normalize initial node features
        nodes = self.node_norm(nodes)

        return nodes

    def form_edges(
        self, batch: gd.Batch, edge_index: torch.Tensor, scaffold: bool = True
    ) -> torch.Tensor:
        """Forms initial edges for protein"""

        # Initialize edges
        edges = batch.bb_dihedral.new_zeros((edge_index.shape[1], self.e_hidden_dim))

        # RBF-encoded pairwise atomic distances
        if scaffold:
            atom_xyz = torch.cat(
                [
                    batch.atom14_xyz[..., :4, :],
                    torch.zeros_like(batch.atom14_xyz[..., 4:, :]),
                ],
                dim=-2,
            )
            atom_mask = torch.cat(
                [
                    batch.atom14_mask[..., :4],
                    torch.zeros_like(batch.atom14_mask[..., 4:]),
                ],
                dim=-1,
            )
        else:
            atom_xyz = batch.atom14_xyz
            atom_mask = batch.atom14_mask

        if self.use_imputed_CB_for_edge_rbfs:
            atom_xyz[..., 5, :] = impute_CB(
                atom_xyz[..., 0, :], atom_xyz[..., 1, :], atom_xyz[..., 2, :]
            )
            atom_mask[..., 5] = torch.prod(atom_mask[..., :3], dim=-1)

        pair_dists = torch.cdist(atom_xyz[edge_index[0]], atom_xyz[edge_index[1]])
        pair_dists = pair_dists.view(
            edge_index.shape[1], -1, 1
        )  # (num_edges, 14 ** 2, 1)
        pair_mask = (
            atom_mask[edge_index[0], :, None] * atom_mask[edge_index[1], None, :]
        )  # (num_edges, 14, 14)
        pair_dists = pair_mask.reshape(edge_index.shape[1], -1, 1) * pair_dists
        rbf_mu = torch.linspace(0, 20, self.num_rbf).view(1, 1, -1)  # (1, 1, num_rbf)
        rbf_mu = rbf_mu.to(batch.x.device)
        rbf_sigma = 20 / self.num_rbf
        rbf = torch.exp(-1 * (pair_dists - rbf_mu) ** 2 / rbf_sigma**2)
        rbf = rbf.view(edge_index.shape[1], -1)
        edges += self.rbf_linear(rbf)

        # Sequence separation
        # If on the same chain, use one-hot encoding of sequence separation (up to 32 residues away)
        # If on different chains, mask one-hot encoding and provide extra bit
        dij = batch.residue_index[edge_index[0]] - batch.residue_index[edge_index[1]]
        dij = torch.clamp(dij, -self.max_seq_sep, self.max_seq_sep) + self.max_seq_sep
        dij = F.one_hot(dij.long(), 2 * self.max_seq_sep + 1).float()
        mij = batch.chain_index[edge_index[0]] == batch.chain_index[edge_index[1]]
        mij = mij.float().view(-1, 1)
        dij = mij * dij
        seq_sep = torch.cat([dij, 1 - mij], dim=-1)
        edges += self.seq_sep_linear(seq_sep)

        # Normalize initial edge features
        edges = self.edge_norm(edges)

        return edges

    def form_graph_nodes(self, nodes: torch.Tensor, batch: gd.Batch) -> torch.Tensor:
        """Forms initial graph nodes for protein"""

        # Create graph-level virtual node
        graph_nodes = gnn.global_max_pool(nodes, batch.batch)
        graph_nodes = self.graph_init(graph_nodes)
        graph_nodes = self.graph_norm(graph_nodes)

        return graph_nodes

    def forward(
        self, batch: gd.Batch, scaffold: bool = True, fix_res: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create initial embedding of protein nodes
        nodes = self.form_nodes(batch, scaffold, fix_res)

        # Get k-nearest neighbor graph for protein
        edge_index = self.get_knn_edges(batch)

        # Create initial embedding of protein edges
        edges = self.form_edges(batch, edge_index, scaffold)

        # Create graph-level virtual node
        graph_nodes = self.form_graph_nodes(nodes, batch)

        # Pass through global graph communication layers and IPMP layers
        nodes, graph_nodes = self.graph_comms[0](nodes, graph_nodes, batch.batch)
        for i, layer in enumerate(self.ipmp_layers):
            nodes, edges = layer(nodes, batch.bb_frames, edges, edge_index)
            nodes, graph_nodes = self.graph_comms[i + 1](
                nodes, graph_nodes, batch.batch
            )

        return nodes, edges, graph_nodes, edge_index



class ProteinDecoder(nn.Module):
    def __init__(
        self,
        n_hidden_dim: int,
        e_hidden_dim: int,
        g_hidden_dim: int,
        num_mpnn_layers: int = 3,
        num_mlp_layers: int = 3,
        num_ipmp_points: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.graph_comms = nn.ModuleList(
            [
                GraphCommunicationLayer(
                    n_hidden_dim,
                    n_hidden_dim,
                    g_hidden_dim,
                    g_hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    act="relu",
                    dropout=dropout,
                )
                for _ in range(num_mpnn_layers + 1)
            ]
        )
        self.ipmp_layers = nn.ModuleList(
            [
                IPMPLayer(
                    n_hidden_dim,
                    n_hidden_dim,
                    e_hidden_dim,
                    e_hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    num_points=num_ipmp_points,
                    act="relu",
                    dropout=dropout,
                )
                for _ in range(num_mpnn_layers)
            ]
        )

    def forward(
        self,
        batch: gd.Batch,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        graph_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass through global graph communication layers and IPMP layers
        nodes, graph_nodes = self.graph_comms[0](nodes, graph_nodes, batch.batch)
        for i, layer in enumerate(self.ipmp_layers):
            nodes, edges = layer(nodes, batch.bb_frames, edges, edge_index)
            nodes, graph_nodes = self.graph_comms[i + 1](
                nodes, graph_nodes, batch.batch
            )

        return nodes, edges, graph_nodes


class AutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        n_hidden_dim: int,
        e_hidden_dim: int,
        g_hidden_dim: int,
        knn_k: int = 30,
        max_seq_sep: int = 32,
        num_rbf: int = 16,
        num_mpnn_layers: int = 3,
        num_mlp_layers: int = 3,
        num_ipmp_points: int = 8,
        dropout: float = 0.1,
        mlp_inner_dim: int = 128,
        use_ipmp: bool = False,
        grad_ckpt: bool = False,
    ) -> None:
        super().__init__()

        # Module attributes
        self.n_hidden_dim = n_hidden_dim
        self.e_hidden_dim = e_hidden_dim
        self.knn_k = knn_k
        self.max_seq_sep = max_seq_sep
        self.num_rbf = num_rbf
        self.n_restype = rc.restype_num + 1  # includes UNK token
        self.use_ipmp = use_ipmp
        self.grad_ckpt = grad_ckpt

        # Node feature layers
        self.seq_linear = nn.Linear(self.n_restype, n_hidden_dim)
        self.seq_norm = nn.LayerNorm(n_hidden_dim)

        # Specialized autoregressive IPMP layer for decoding
        self.ipmp_layers = nn.ModuleList(
            [
                AutoregressiveIPMPLayer(
                    n_hidden_dim,
                    n_hidden_dim,
                    e_hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    num_points=num_ipmp_points,
                    act="relu",
                    dropout=dropout,
                    mlp_inner_dim=mlp_inner_dim,
                    use_ipmp=use_ipmp,
                )
                for _ in range(num_mpnn_layers)
            ]
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch: gd.Batch,
    ):
        # Get seq node embedding
        seq_nodes = F.one_hot(batch.aatype, num_classes=self.n_restype).to(
            torch.float32
        )
        seq_nodes = self.seq_norm(self.seq_linear(seq_nodes))
        dec_order = batch.decoding_order[:, None]

        # Re-use encoder node reps
        enc_nodes = torch.clone(nodes)
        # Update through IPMP using autoreg masking
        for i, layer in enumerate(self.ipmp_layers):
            if torch.is_grad_enabled() and self.grad_ckpt:
                nodes = checkpoint(
                    layer,
                    nodes,
                    batch.bb_frames,
                    edges,
                    edge_index,
                    seq_nodes,
                    dec_order,
                    enc_nodes,
                    use_reentrant=False,
                )
            else:
                nodes = layer(
                    nodes,
                    batch.bb_frames,
                    edges,
                    edge_index,
                    seq_nodes,
                    dec_order,
                    enc_nodes,
                )
        return nodes


class AutoregressiveIPMPLayer(nn.Module):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        num_mlp_layers: int = 3,
        num_points: int = 8,
        act: str = "relu",
        dropout: float = 0.1,
        mlp_inner_dim: int = 128,
        use_ipmp: bool = False,
    ) -> None:
        super().__init__()

        self.conv = AutoregressiveIPMPConv(
            n_in_dim,
            n_out_dim,
            e_in_dim,
            num_mlp_layers,
            num_points,
            act,
            use_ipmp,
        )
        self.node_norm = nn.ModuleList([nn.LayerNorm(n_out_dim) for _ in range(2)])
        self.node_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.node_ff = MLP(n_out_dim, mlp_inner_dim, n_out_dim, num_mlp_layers, act=act)

    def forward(
        self,
        nodes: torch.Tensor,
        bb_frames: Rigid,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        seq_nodes: torch.Tensor,
        dec_order: torch.Tensor,
        enc_nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute node and edge messages
        node_message = self.conv(
            nodes, bb_frames, edges, edge_index, seq_nodes, dec_order, enc_nodes
        )
        # Update node and edge features
        nodes = self.node_norm[0](nodes + self.node_dropout[0](node_message))
        nodes = self.node_norm[1](nodes + self.node_dropout[1](self.node_ff(nodes)))
        return nodes


class AutoregressiveIPMPConv(gnn.MessagePassing):
    def __init__(
        self,
        n_in_dim: int,
        n_out_dim: int,
        e_in_dim: int,
        num_mlp_layers: int = 3,
        num_points: int = 8,
        act: str = "relu",
        use_ipmp: bool = False,
    ) -> None:
        super().__init__(aggr="mean")

        self.num_points = num_points
        self.act = get_act_fxn(act)
        self.use_ipmp = use_ipmp

        # Point and message functions
        self.point_fn = nn.Linear(n_in_dim, num_points * 3)
        msg_size = 3 * n_in_dim + e_in_dim
        if self.use_ipmp:
            msg_size += (2 * num_points * 3) + (2 * num_points) + (num_points**2)
        self.node_message_func = MLP(
            msg_size,
            n_out_dim,
            n_out_dim,
            num_mlp_layers,
            act=act,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        bb_frames: Rigid,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        seq_nodes: torch.Tensor,
        dec_order: torch.Tensor,
        enc_nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute node and edge messages
        node_message = self.propagate(
            edge_index,
            n=nodes,
            F=bb_frames,
            S=seq_nodes,
            D=dec_order,
            en=enc_nodes,
            edge_attr=edges,
        )
        return node_message

    def _get_message_in(
        self,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        F_i: Rigid,
        F_j: Rigid,
        S_i: torch.Tensor,
        S_j: torch.Tensor,
        D_i: torch.Tensor,
        D_j: torch.Tensor,
        en_i: torch.Tensor,
        en_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # i is the target node (edge_index[1])
        # j is the source node (edge_index[0])
        # message is propagated j -> i
        # If i decoded after j, allow i to see S_j
        D_mask = (D_i > D_j).float()
        S_j = S_j * D_mask
        node_j = n_j * D_mask + (1 - D_mask) * en_j

        if self.use_ipmp:
            # Get node i's local points and their norms
            p_local_i = self.point_fn(n_i).view(-1, self.num_points, 3)
            p_local_i_norm = torch.sqrt(torch.sum(p_local_i**2, dim=-1) + 1e-6)

            # Get node j's local points in i's frame and their norms
            p_local_j = self.point_fn(node_j).view(-1, self.num_points, 3)
            p_local_j = F_i[0].invert_apply(F_j[0].apply(p_local_j))
            p_local_j_norm = torch.sqrt(torch.sum(p_local_j**2, dim=-1) + 1e-6)

            # Get the distances between node i's local points and node j's local points in i's frame
            p_local_dists = torch.sqrt(
                torch.sum(
                    (p_local_i[..., None, :] - p_local_j[..., None, :, :]) ** 2, dim=-1
                )
                + 1e-6
            )

        # Compute message
        message_in = [n_i, node_j, edge_attr, S_j]

        if self.use_ipmp:
            message_in += [
                p_local_i.view(-1, self.num_points * 3),
                p_local_i_norm,
                p_local_j.view(-1, self.num_points * 3),
                p_local_j_norm,
                p_local_dists.view(-1, self.num_points**2),
            ]

        message_in = torch.cat(
            message_in,
            dim=-1,
        )

        return message_in

    def message(
        self,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        F_i: Rigid,
        F_j: Rigid,
        S_i: torch.Tensor,
        S_j: torch.Tensor,
        D_i: torch.Tensor,
        D_j: torch.Tensor,
        en_i: torch.Tensor,
        en_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Compute node message
        message_in = self._get_message_in(
            n_i, n_j, F_i, F_j, S_i, S_j, D_i, D_j, en_i, en_j, edge_attr
        )
        message = self.node_message_func(message_in)

        return message


class ProteinMPNNEncoder(nn.Module):
    def __init__(
        self,
        n_hidden_dim: int,
        g_hidden_dim: int,
        e_hidden_dim: int,
        knn_k: int = 30,
        max_seq_sep: int = 32,
        num_rbf: int = 16,
        num_mpnn_layers: int = 3,
        num_mlp_layers: int = 3,
        num_ipmp_points: int = 8,
        dropout: float = 0.1,
        mlp_inner_dim: int = 128,
        use_ipmp: bool = False,
        use_graph_comms: bool = False,
        grad_ckpt: bool = False,
        use_bb_dih: bool = False,
        bb_noise: float = 0.0,
    ) -> None:
        super().__init__()

        # Module attributes
        self.n_hidden_dim = n_hidden_dim
        g_in_dim = n_hidden_dim
        self.e_hidden_dim = e_hidden_dim
        self.knn_k = knn_k
        self.max_seq_sep = max_seq_sep
        self.num_rbf = num_rbf
        self.use_ipmp = use_ipmp
        self.use_graph_comms = use_graph_comms
        self.grad_ckpt = grad_ckpt
        self.use_bb_dih = use_bb_dih

        # Node feature layers
        if self.use_bb_dih:
            self.bb_dihedral_linear = nn.Linear(6, n_hidden_dim)
            self.node_norm = nn.LayerNorm(n_hidden_dim)

        # Edge feature layers
        self.rbf_linear = nn.Linear(5**2 * self.num_rbf, e_hidden_dim)
        self.seq_sep_linear = nn.Linear(2 * self.max_seq_sep + 2, e_hidden_dim)
        self.edge_norm = nn.LayerNorm(e_hidden_dim)

        if self.use_ipmp:
            self.mpnn_layers = nn.ModuleList(
                [
                    IPMPLayer(
                        n_hidden_dim,
                        n_hidden_dim,
                        e_hidden_dim,
                        e_hidden_dim,
                        num_mlp_layers=num_mlp_layers,
                        num_points=num_ipmp_points,
                        act="relu",
                        dropout=dropout,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )
        else:
            self.mpnn_layers = nn.ModuleList(
                [
                    MPNNLayer(
                        n_hidden_dim,
                        n_hidden_dim,
                        e_hidden_dim,
                        e_hidden_dim,
                        num_mlp_layers=num_mlp_layers,
                        act="relu",
                        dropout=dropout,
                        mlp_inner_dim=mlp_inner_dim,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )

        if self.use_graph_comms:
            # Graph feature layers
            self.graph_init = nn.Linear(g_in_dim, g_hidden_dim)
            self.graph_norm = nn.LayerNorm(g_hidden_dim)

            self.graph_comms = nn.ModuleList(
                [
                    GraphCommunicationLayer(
                        n_hidden_dim,
                        n_hidden_dim,
                        g_hidden_dim,
                        g_hidden_dim,
                        num_mlp_layers=num_mlp_layers,
                        act="relu",
                        dropout=dropout,
                    )
                    for _ in range(num_mpnn_layers)
                ]
            )

    def get_knn_edges(self, batch: gd.Batch) -> torch.Tensor:
        """Computes k-nearest neighbor graph for protein based on CA atom positions"""

        # Get CA atom positions
        ca_xyz = batch.atom14_xyz[..., 1, :]

        # Compute k-nearest neighbor graph
        edge_index = gnn.knn_graph(ca_xyz, self.knn_k, batch=batch.batch, loop=True)

        return edge_index

    def form_nodes(
        self,
        batch: gd.Batch,
    ) -> torch.Tensor:
        """Forms initial nodes for protein"""

        # Initialize nodes
        nodes = batch.bb_dihedral.new_zeros(
            (batch.bb_dihedral.shape[0], self.n_hidden_dim)
        )

        # Add bb dihedrals, if using
        if self.use_bb_dih:
            bb_dihedral_sincos = torch.stack(
                [torch.sin(batch.bb_dihedral), torch.cos(batch.bb_dihedral)], dim=-1
            ).view(-1, 6)
            nodes += self.bb_dihedral_linear(bb_dihedral_sincos)
            nodes = self.node_norm(nodes)

        return nodes

    def form_edges(
        self,
        batch: gd.Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forms initial edges for protein"""

        # Initialize edges
        edges = batch.bb_dihedral.new_zeros((edge_index.shape[1], self.e_hidden_dim))

        # RBF-encoded pairwise atomic distances
        atom_xyz = batch.atom14_xyz[..., :5, :]
        atom_mask = batch.atom14_mask[..., :5]
        atom_xyz[..., 4, :] = impute_CB(
            atom_xyz[..., 0, :], atom_xyz[..., 1, :], atom_xyz[..., 2, :]
        )
        atom_mask[..., 4] = 1.0

        pair_dists = torch.cdist(atom_xyz[edge_index[0]], atom_xyz[edge_index[1]])
        pair_dists = pair_dists.view(
            edge_index.shape[1], -1, 1
        )  # (num_edges, 5 ** 2, 1)
        pair_mask = (
            atom_mask[edge_index[0], :, None] * atom_mask[edge_index[1], None, :]
        )  # (num_edges, 5, 5)
        pair_dists = pair_mask.reshape(edge_index.shape[1], -1, 1) * pair_dists
        rbf_mu = torch.linspace(2, 22, self.num_rbf).view(1, 1, -1)  # (1, 1, num_rbf)
        rbf_mu = rbf_mu.to(batch.x.device)
        rbf_sigma = 20 / self.num_rbf
        rbf = torch.exp(-1 * (pair_dists - rbf_mu) ** 2 / rbf_sigma**2)
        rbf = rbf.view(edge_index.shape[1], -1)
        edges += self.rbf_linear(rbf)

        # Sequence separation
        # If on the same chain, use one-hot encoding of sequence separation (up to 32 residues away)
        # If on different chains, mask one-hot encoding and provide extra bit
        dij = batch.residue_index[edge_index[0]] - batch.residue_index[edge_index[1]]
        dij = torch.clamp(dij, -self.max_seq_sep, self.max_seq_sep) + self.max_seq_sep
        dij = F.one_hot(dij.long(), 2 * self.max_seq_sep + 1).float()
        mij = batch.chain_index[edge_index[0]] == batch.chain_index[edge_index[1]]
        mij = mij.float().view(-1, 1)
        dij = mij * dij
        seq_sep = torch.cat([dij, 1 - mij], dim=-1)
        edges += self.seq_sep_linear(seq_sep)

        # Normalize initial edge features
        edges = self.edge_norm(edges)

        return edges

    def form_graph_nodes(self, nodes: torch.Tensor, batch: gd.Batch) -> torch.Tensor:
        """Forms initial graph nodes for protein"""

        # Create graph-level virtual node
        graph_nodes = gnn.global_max_pool(nodes, batch.batch)
        graph_nodes = self.graph_init(graph_nodes)
        graph_nodes = self.graph_norm(graph_nodes)

        return graph_nodes

    def forward(
        self,
        batch: gd.Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create initial embedding of protein nodes
        nodes = self.form_nodes(batch)

        # Get k-nearest neighbor graph for protein
        edge_index = self.get_knn_edges(batch)

        # Create initial embedding of protein edges
        edges = self.form_edges(batch, edge_index)

        if self.use_graph_comms:
            # Create graph-level virtual node
            graph_nodes = self.form_graph_nodes(nodes, batch)

        # Pass through MPNN layers
        for i, layer in enumerate(self.mpnn_layers):
            # Implemented gradient checkpointing
            if torch.is_grad_enabled() and self.grad_ckpt:
                if self.use_ipmp:
                    nodes, edges = checkpoint(
                        layer,
                        nodes,
                        batch.bb_frames,
                        edges,
                        edge_index,
                        use_reentrant=False,
                    )
                else:
                    nodes, edges = checkpoint(
                        layer, nodes, edges, edge_index, use_reentrant=False
                    )
                if self.use_graph_comms:
                    nodes, graph_nodes = checkpoint(
                        self.graph_comms[i],
                        nodes,
                        graph_nodes,
                        batch.batch,
                        use_reentrant=False,
                    )
            else:
                if self.use_ipmp:
                    nodes, edges = layer(nodes, batch.bb_frames, edges, edge_index)
                else:
                    nodes, edges = layer(nodes, edges, edge_index)
                if self.use_graph_comms:
                    nodes, graph_nodes = self.graph_comms[i](
                        nodes, graph_nodes, batch.batch
                    )
        return nodes, edges, None, edge_index
