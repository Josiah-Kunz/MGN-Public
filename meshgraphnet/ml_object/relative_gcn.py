import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class RelativeGCN(nn.Module):
    """
    Graph Convolutional Network that uses relative differences between neighbors.
    
    Combines absolute node features with neighbor-relative gradients to capture
    field variations. Useful for any problem where spatial/feature gradients matter
    (stress fields, temperature distributions, flow fields, etc.).
    
    Args:
        in_ch (int): Number of input features per node (e.g., 3 for x, y, load)
        hidden (int): Number of hidden units per layer
        num_layers (int): Total number of convolution layers (so 3 = input, conv, output)
    
    Example:
        # Features: x, y, load
        model = RelativeGCN(in_ch=3, hidden=64, num_layers=4)
        x = torch.tensor([[0.5, 0.2, 5000], [1.0, 0.3, 5000], ...])
        edge_index = torch.tensor([[0, 1, ...], [1, 2, ...]])
        predictions = model(x, edge_index)
    """
    def __init__(self, in_ch, hidden=64, num_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RelativeGCNConv(in_ch, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(RelativeGCNConv(hidden, hidden))
        self.convs.append(RelativeGCNConv(hidden, 1))

    def forward(self, x, edge_indices):
        """
        Forward pass through all convolution layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_indices: Graph connectivity [2, num_edges]
        
        Returns:
            Predictions for each node [num_nodes]
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_indices).relu()
        return self.convs[-1](x, edge_indices).squeeze()


class RelativeGCNConv(MessagePassing):
    """
    Graph convolution layer that uses both absolute and relative neighbor features.
    
    Processes two types of information:
    1. Absolute neighbor features (what neighbors have)
    2. Relative differences (how neighbors differ from current node)
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Sum messages from all neighbors

        # Transform absolute neighbor features
        self.lin_neighbor = nn.Linear(in_channels, out_channels)

        # Transform relative differences (gradients)
        self.lin_relative = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        
        # Consider our own features as well
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Degree normalization (otherwise highly-connected nodes dominate)
        """
        Message from E (degree=6) → A (degree=2): norm = 1/√(6×2) ≈ 0.29
        Message from A (degree=2) → E (degree=6): norm = 1/√(2×6) ≈ 0.29 (symmetric!)
        Message from E (degree=6) → E (degree=6): norm = 1/√(6×6) ≈ 0.17 (self-loop damped)
        """
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_i, x_j, norm):
        """
        Compute messages from neighbors. PyTorch Geometric provides:
        
        x_i: [num_edges, features] - features of CURRENT/TARGET nodes (receiving)
        x_j: [num_edges, features] - features of NEIGHBOR/SOURCE nodes (sending)
    
        Example for one edge (neighbor node 5 → current node 3):
            Each node has (x, y, load)
            x_i = [2.5, 3.1, 5000]  # Current node 3
            x_j = [2.7, 3.0, 5000]  # Neighbor node 5
        """

        diff = x_j - x_i  # [0.2, -0.1, 0] - difference in x, y, load

        """
        lin_neighbor: Learns from absolute neighbor features
            "What are my neighbor's properties?"
            Input: x_j = [x_neighbor, y_neighbor, load_neighbor]
            Learns: "High load neighbors -> high stress"
        
        lin_relative: Learns from relative differences (gradients)
            "How different am I from my neighbor?"
            Input: diff = [Δx, Δy, Δload]
            Learns: "Large load gradient -> high stress concentration"
        
        Combined message:
            message = lin_neighbor(x_j) + lin_relative(diff)
                    = W1 @ [2.7, 3.0, 5000] + W2 @ [0.2, -0.1, 0]
                    = [info about neighbor] + [info about gradient]
        """

        msg = self.lin_neighbor(x_j) + self.lin_relative(diff)
        return norm.view(-1,1) * msg