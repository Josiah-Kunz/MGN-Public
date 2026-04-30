import torch.nn as nn
from torch_geometric.nn import GCNConv

class BasicGCN(nn.Module):
    """
    Graph convolution network (to be used inside GNN). 
    
    Example: 
        in_ch = 2  # x and y coordinates
        x = torch.tensor([
            [0.5, 0.2],   # Node 0: x=0.5, y=0.2
            [1.0, 0.3],   # Node 1: x=1.0, y=0.3
            [0.7, 0.8],   # Node 2: x=0.7, y=0.8
            ...
        ])
    """
    def __init__(self, in_ch, hidden, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, 1))

    def forward(self, x, edge_indices):
        """
        Does message passing for each convolution layer, resulting in a weighted normalized sum.
        For all except the last convolution, it also applies the activation function (relu in this case) as a measure 
        of "how active" the neuron is.
        :param x: 
        :param edge_indices: An array of [to] and [from] node indices. Example for a triangle:
        
            # 0→1, 1→2, 2→0
            edge_index = [[0, 1, 2],    # source
                        [1, 2, 0]]      # target
                        
        Then the new x is normalized, aggregated, and weighed as (simplified):
        
            for i in range(num_edges):
                current_node = edge_index[0, i]
                neighbor = edge_index[1, i]
                aggregated[target] += normalize(x[current_node]-x[neighbor])
            
            # Then transform
            x_new = weights @ aggregated
        :return: 
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_indices).relu()
        return self.convs[-1](x, edge_indices).squeeze()