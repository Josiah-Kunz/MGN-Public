import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Type
from copy import deepcopy
import os

# Supported GCNs
from .basic_gcn import BasicGCN
from .relative_gcn import RelativeGCN


class GNN:
    """
    GNN wrapper with sklearn-like interface.
    
    Args:
        conv_class: Convolution architecture to use (BasicGCN, RelativeGCN, or custom)
        edges_csv: Path to edge file (single graph mode)
        hidden_channels: Hidden layer size
        num_layers: Total number of convolution layers
        epochs: Maximum training epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience (-1 for no early stopping)
        transductive: True for mask-based split, False for separate graphs, None for auto-detect
    
    Modes:
    - transductive=True: Single graph, mask-based train/test split
    - transductive=False: Separate train/test graphs (multiple files)
    
    Example:
        # Use BasicGCN
        gnn = GNN(conv_class=BasicGCN, hidden_channels=64, num_layers=4)
        
        # Use RelativeGCN
        gnn = GNN(conv_class=RelativeGCN, hidden_channels=128, num_layers=6)
    """

    def __init__(self, edges_csv=None, hidden_channels=64, num_layers=4, epochs=2000,
                 learning_rate=0.01, patience=-1, transductive=None, conv_class: Type[nn.Module] = BasicGCN):
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.transductive = transductive
        self.patience = patience
        self.conv_class = conv_class

        if patience < 0:
            self.patience = float("inf")

        self._model = None
        self._train_edge_index = None
        self._test_edge_index = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

        # For transductive mode (single graph with masks)
        self._X = None
        self._y = None
        self._train_mask = None
        self._test_mask = None

        # For inductive mode (separate graphs)
        self._X_train = None
        self._y_train = None

        # See if we can use gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GNN using: {self.device}")

        # Load edges if provided (single file mode)
        if edges_csv is not None:
            self._train_edge_index = self._load_edge_file(edges_csv)
            self._test_edge_index = self._train_edge_index

    def _load_edge_file(self, edges_csv):
        """Load a single edge file and return edge_index tensor."""
        edges_df = pd.read_csv(edges_csv)
        edge_index = torch.tensor(
            np.array([edges_df['node_i'].values, edges_df['node_j'].values]),
            dtype=torch.long
        )
        return torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def _load_and_combine_edges(self, edge_files: List[str], node_offsets: List[int]):
        """Load multiple edge files and combine into one graph."""
        all_sources = []
        all_targets = []

        for edge_file, offset in zip(edge_files, node_offsets):
            edges_df = pd.read_csv(edge_file)
            all_sources.append(edges_df['node_i'].values + offset)
            all_targets.append(edges_df['node_j'].values + offset)

        sources = np.concatenate(all_sources)
        targets = np.concatenate(all_targets)

        edge_index = torch.tensor(np.array([sources, targets]), dtype=torch.long)
        return torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def set_edge_files(self, train_edge_files: List[str], test_edge_files: List[str],
                       train_node_counts: List[int] = None, test_node_counts: List[int] = None):
        """Set edge files for train and test graphs (inductive mode)."""
        self._train_edge_files = train_edge_files
        self._test_edge_files = test_edge_files
        self._train_node_counts = train_node_counts
        self._test_node_counts = test_node_counts

    def fit(self, X, y, train_ratio=0.8):
        """Train the GNN."""
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).squeeze().to(self.device)

        if self.transductive:
            # Store full data
            self._X = X
            self._y = y

            # Create train/test masks
            n_nodes = X.shape[0]
            perm = torch.randperm(n_nodes)
            n_train = int(n_nodes * train_ratio)
            self._train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            self._train_mask[perm[:n_train]] = True
            self._test_mask = ~self._train_mask
        else:
            # Store train data only
            self._X_train = X
            self._y_train = y

            # Handle edge indices for multiple files
            if hasattr(self, '_train_edge_files') and self._train_edge_files is not None:
                if self._train_node_counts is None:
                    self._train_node_counts = [len(X) // len(self._train_edge_files)] * len(self._train_edge_files)

                offsets = [0]
                for count in self._train_node_counts[:-1]:
                    offsets.append(offsets[-1] + count)

                self._train_edge_index = self._load_and_combine_edges(self._train_edge_files, offsets)

        self._train_edge_index = self._train_edge_index.to(self.device)

        # Normalize
        self._x_mean = X.mean(dim=0)
        self._x_std = X.std(dim=0)
        X_norm = (X - self._x_mean) / (self._x_std + 1e-8)

        self._y_mean = y.mean()
        self._y_std = y.std()
        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)

        # Build model using specified convolution class
        self._model = self.conv_class(X.shape[1], self.hidden_channels, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Train
        self._model.train()
        pbar = tqdm(range(self.epochs), desc=f"Training GNN ({'transductive' if self.transductive else 'inductive'})")
        best_loss = None
        best_state = None
        patience_counter = 0
        self.train_losses = []

        for epoch in pbar:
            optimizer.zero_grad()
            pred = self._model(X_norm, self._train_edge_index)

            if self.transductive:
                loss = loss_fn(pred[self._train_mask], y_norm[self._train_mask])
            else:
                loss = loss_fn(pred, y_norm)

            loss.backward()
            optimizer.step()
            self.train_losses.append(loss.item())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Early stopping
            if best_loss is None:
                pct_improvement = 0
            elif best_loss > 0:
                pct_improvement = (best_loss - loss.item()) / best_loss
            else:
                pct_improvement = None

            if pct_improvement > 0.001 or best_loss is None:
                best_loss = loss.item()
                best_state = deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def predict(self, X=None, edge_index=None):
        """Predict stress values."""
        if self._model is None:
            raise ValueError("Must call fit() first")

        if self.transductive:
            # Always use full graph
            X_to_use = self._X
            edge_index = self._train_edge_index
        else:
            if X is None:
                X_to_use = self._X_train
                edge_index = self._train_edge_index
            else:
                X_to_use = torch.tensor(X, dtype=torch.float).to(self.device)

                if edge_index is None and hasattr(self, '_test_edge_files') and self._test_edge_files is not None:
                    if self._test_node_counts is None:
                        self._test_node_counts = [len(X) // len(self._test_edge_files)] * len(self._test_edge_files)

                    offsets = [0]
                    for count in self._test_node_counts[:-1]:
                        offsets.append(offsets[-1] + count)

                    edge_index = self._load_and_combine_edges(self._test_edge_files, offsets).to(self.device)
                elif edge_index is None:
                    edge_index = self._train_edge_index

        X_norm = (X_to_use - self._x_mean) / (self._x_std + 1e-8)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_norm, edge_index)
            pred = pred * self._y_std + self._y_mean

        return pred.cpu().numpy()

    def score(self, X, y):
        """Return R² score."""
        y_pred = self.predict(X)
        y = np.array(y).squeeze()

        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    def train_score(self):
        """Return R² on training data."""
        if self.transductive:
            y_pred = self.predict()
            y_pred = y_pred[self._train_mask.cpu().numpy()]
            y_actual = self._y[self._train_mask].cpu().numpy()
        else:
            y_pred = self.predict()
            y_actual = self._y_train.cpu().numpy()

        ss_res = ((y_actual - y_pred) ** 2).sum()
        ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    def save(self, filepath):
        """
        Usage: 
            gnn = GNN(conv_class=RelativeGCN)
            ml.train(model=gnn, model_name="Relative GCN")
            gnn.save('models/rgcn.pt')
        :param filepath: 
        :return: 
        """
        if self._model is None:
            raise ValueError("Must call fit() first before saving")
    
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
        save_dict = {
            'model_state_dict': self._model.state_dict(),
            'conv_class_name': self.conv_class.__name__,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'x_mean': self._x_mean.cpu(),
            'x_std': self._x_std.cpu(),
            'y_mean': self._y_mean.cpu(),
            'y_std': self._y_std.cpu(),
            'transductive': self.transductive,
            'train_losses': self.train_losses,
        }
    
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load a trained GNN model.
        
        Args:
            filepath: Path to saved model file
            device: Device to load model on (None = auto-detect)
        
        Returns:
            GNN: Loaded model ready for prediction
        
        Example:
            gnn = GNN.load('models/my_model.pt')
            predictions = gnn.predict(X_test)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        checkpoint = torch.load(filepath, map_location=device)
    
        # Get convolution class
        conv_class_name = checkpoint['conv_class_name']
        if conv_class_name == 'BasicGCN':
            from .basic_gcn import BasicGCN
            conv_class = BasicGCN
        elif conv_class_name == 'RelativeGCN':
            from .relative_gcn import RelativeGCN
            conv_class = RelativeGCN
        else:
            raise ValueError(f"Unknown convolution class: {conv_class_name}")
    
        # Create new GNN instance
        gnn = cls(
            hidden_channels=checkpoint['hidden_channels'],
            num_layers=checkpoint['num_layers'],
            learning_rate=checkpoint['learning_rate'],
            transductive=checkpoint['transductive'],
            conv_class=conv_class
        )

        # Apply device
        gnn.device = device
    
        # Rebuild model
        in_channels = checkpoint['x_mean'].shape[0]
        gnn._model = conv_class(in_channels, gnn.hidden_channels, gnn.num_layers).to(device)
        gnn._model.load_state_dict(checkpoint['model_state_dict'])
        gnn._model.eval()
    
        # Restore normalization
        gnn._x_mean = checkpoint['x_mean'].to(device)
        gnn._x_std = checkpoint['x_std'].to(device)
        gnn._y_mean = checkpoint['y_mean'].to(device)
        gnn._y_std = checkpoint['y_std'].to(device)
        gnn.train_losses = checkpoint['train_losses']
    
        print(f"Model loaded from {filepath}")
        return gnn