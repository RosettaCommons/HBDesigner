from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


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
