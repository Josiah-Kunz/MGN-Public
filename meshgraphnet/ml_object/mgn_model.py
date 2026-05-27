import torch
import torch.nn as nn

class MGNModel(nn.Module):
    """
    The actual PyTorch model for MGN.
    
    Architecture:
    1. Node type → learned embedding → encoder → hidden state
    2. Edge features [dx, dy, length] → encoder → hidden state  
    3. Message passing: combine neighbor nodes + edge features
    4. Decode node states → predictions
    
    Key insight: NO absolute positions! Only:
    - Node types (what boundary condition)
    - Edge features (relative positions between neighbors)
    """

    def __init__(
            self,
            num_node_types: int,
            embedding_dim: int,
            edge_feature_dim: int,
            hidden_channels: int,
            num_layers: int,
            global_feature_dim: int = 0,
    ):
        super().__init__()

        # Node type embedding: "fixed" → [0.23, -0.15, ...]
        self.node_embedding = nn.Embedding(num_node_types, embedding_dim)

        # Edge feature encoder: [dx, dy, length] → hidden
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Node encoder: embedding → hidden
        # It's already encoded, so no need to relu
        self.node_encoder = nn.Linear(embedding_dim, hidden_channels)

        # Global encoder (if we have global features)
        self.has_globals = global_feature_dim > 0
        if self.has_globals:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_feature_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.node_update_layers = nn.ModuleList()

        for _ in range(num_layers):
            
            # Message function: source_node + edge_features → message
            self.message_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )

            # Node update: node + aggregated + global → new node
            update_input_dim = hidden_channels * 2  # node + aggregated
            if self.has_globals:
                update_input_dim += hidden_channels  # + global

            self.node_update_layers.append(
                nn.Sequential(
                    nn.Linear(update_input_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )

        # Output decoder: hidden → prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, node_types, edge_index, edge_attr, global_features=None):
        """
        Forward pass.
        
        Args:
            node_types: [num_nodes] - integer node type IDs
            edge_index: [2, num_edges] - source/target node indices
            edge_attr: [num_edges, 3] - edge features [dx, dy, length]
            global_features: [num_nodes, global_dim] - same value broadcast to all nodes
                             OR [1, global_dim] and we broadcast internally
        
        Returns:
            [num_nodes] - predicted values
        """
        
        num_nodes = node_types.size(0)
        src, dst = edge_index

        # Encode nodes: type_id → embedding → hidden
        x = self.node_encoder(self.node_embedding(node_types))  # [num_nodes, hidden]

        # Encode edges: [dx, dy, length] → hidden
        e = self.edge_encoder(edge_attr)  # [num_edges, hidden]

        # Encode globals (broadcast to all nodes)
        if self.has_globals and global_features is not None:
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
            if global_features.size(0) == 1:
                global_features = global_features.expand(num_nodes, -1)
            g = self.global_encoder(global_features)  # [num_nodes, hidden]
        else:
            g = None

        # Message passing layers
        for msg_layer, update_layer in zip(self.message_layers, self.node_update_layers):
            # Compute messages from neighbors
            src_features = x[src]  # [num_edges, hidden]
            messages = msg_layer(torch.cat([src_features, e], dim=1))  # [num_edges, hidden]

            # Aggregate messages at each node (sum), but disable autocast in case of mixed precision
            # (scatter_add likes everything to be float32)
            with torch.cuda.amp.autocast(enabled=False):
                aggregated = torch.zeros_like(x).float()
                aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages.float())

            # Update node states (possibly with global context)
            if g is not None:
                x = update_layer(torch.cat([x, aggregated, g], dim=1))
            else:
                x = update_layer(torch.cat([x, aggregated], dim=1))

        # Decode to predictions
        out = self.decoder(x).squeeze(-1)  # [num_nodes]

        return out