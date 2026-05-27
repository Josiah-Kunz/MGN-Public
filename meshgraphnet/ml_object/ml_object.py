"""
Machine Learning object for FEM result analysis.

Authors: Josiah Kunz, Claude
"""

from .gnn import GNN
from .mgn import MGN
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import pickle # saving/loading models
import torch # for loading edges


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    r2: float
    rmse: float
    mae: float

    def __str__(self):
        return f"R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata."""
    model: Any
    name: str
    objectives: List[str]
    train_r2: float
    features: List[str] = None
    coefficients: Optional[np.ndarray] = None
    intercept: Optional[float] = None

    def predict(self, df=None, fem=None):
        """Make predictions on new data."""
        if isinstance(self.model, MGN):
            if fem is None:
                raise ValueError("MGN requires fem argument")
            return self.model.predict(fem)
    
        X = df[self.features].values
        return self.model.predict(X)


@dataclass
class MLObject:
    """
    Machine Learning wrapper for FEM result analysis.
    
    Can use either:
    - data: single file, auto-split into train/test
    - train_data + test_data: explicit train/test files
    
    For GNN, edge files are auto-derived as {file}.edges.csv unless
    train_edges/test_edges are explicitly specified.
    """

    # CSV-based for sklearn and GNNs
    # Either use data (auto-split) OR train_data/test_data (explicit)
    data: str = None
    train_data: Union[str, List[str]] = None
    test_data: Union[str, List[str]] = None
    features: List[str] = field(default_factory=list)

    # Optional explicit edge files for GNN
    train_edges: Union[str, List[str]] = None
    test_edges: Union[str, List[str]] = None

    # FEM-based (MGN)
    train_fem: Union['FEMObject', List['FEMObject']] = None
    test_fem: Union['FEMObject', List['FEMObject']] = None

    # Common
    objectives: List[str] = field(default_factory=list)
    name: str = None
    train_ratio: float = 0.8

    # Internal state
    _df: pd.DataFrame = field(default=None, init=False, repr=False)
    _df_train: pd.DataFrame = field(default=None, init=False, repr=False)
    _df_test: pd.DataFrame = field(default=None, init=False, repr=False)
    _trained_model: TrainedModel = field(default=None, init=False, repr=False)
    _test_metrics: ModelMetrics = field(default=None, init=False, repr=False)
    _comparison_results: Dict = field(default_factory=dict, init=False, repr=False)
    _fem_mode: bool = field(default=False, init=False, repr=False)

    # Edge file tracking for GNN
    _train_edge_files: List[str] = field(default=None, init=False, repr=False)
    _test_edge_files: List[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Validate: use one or the other
        has_csv = self.train_data is not None or self.data is not None
        has_fem = self.train_fem is not None
        self._fem_mode = has_fem
        if has_csv and has_fem:
            raise ValueError("Use train_data (CSV) OR train_fem (FEMObject), not both")
        if not has_csv and not has_fem:
            raise ValueError("Must provide train_data (CSV) OR train_fem (FEMObject)")

        # Validate single objective (for now)
        if len(self.objectives) != 1:
            raise NotImplementedError(
                "Multi-objective prediction not yet supported. "
                "Use a single objective, e.g., objectives=['von_mises'], for now."
            )

        # Using FEM data?
        if has_fem:
            self._load_from_fem()
            return
        
        # Validate: either data OR train_data/test_data
        if self.data is not None and (self.train_data is not None or self.test_data is not None):
            raise ValueError("Use either 'data' OR 'train_data'/'test_data', not both")

        if self.data is None and (self.train_data is None or self.test_data is None):
            raise ValueError("Must provide either 'data' OR both 'train_data' and 'test_data'")

        if self.data is not None:
            # Single file mode: auto-split
            self._load_single_file()
        else:
            # Explicit train/test mode
            self._load_train_test_files()

        # Validate columns
        missing_features = [f for f in self.features if f not in self._df.columns]
        missing_objectives = [o for o in self.objectives if o not in self._df.columns]

        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        if missing_objectives:
            raise ValueError(f"Objectives not found in data: {missing_objectives}")

    def _load_single_file(self):
        """Load single data file and auto-split."""
        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise FileNotFoundError(f"Data file not found: {self.data}")
            self._df = pd.read_csv(self.data)
            if self.name is None:
                self.name = os.path.splitext(os.path.basename(self.data))[0]

            # Edge file for GNN (single file)
            self._train_edge_files = [self.data.replace('.csv', '.edges.csv')]
            self._test_edge_files = self._train_edge_files  # Same graph for train/test

        elif isinstance(self.data, pd.DataFrame):
            self._df = self.data.copy()
            if self.name is None:
                self.name = "ML Analysis"
            self._train_edge_files = None
            self._test_edge_files = None
        else:
            raise TypeError("data must be a file path or DataFrame")

        # Split data
        self._df_train, self._df_test = train_test_split(
            self._df,
            train_size=self.train_ratio,
            random_state=42
        )

    def _load_train_test_files(self):
        """Load explicit train/test files."""
        # Normalize to lists
        if isinstance(self.train_data, str):
            self.train_data = [self.train_data]
        if isinstance(self.test_data, str):
            self.test_data = [self.test_data]

        # Load training data
        train_dfs = []
        for i, f in enumerate(self.train_data):
            if not os.path.exists(f):
                raise FileNotFoundError(f"Train data file not found: {f}")
            df = pd.read_csv(f)
            df['_source_file'] = f
            df['_source_idx'] = range(len(df))
            df['_file_id'] = i
            train_dfs.append(df)
        self._df_train = pd.concat(train_dfs, ignore_index=True)

        # Load test data
        test_dfs = []
        for i, f in enumerate(self.test_data):
            if not os.path.exists(f):
                raise FileNotFoundError(f"Test data file not found: {f}")
            df = pd.read_csv(f)
            df['_source_file'] = f
            df['_source_idx'] = range(len(df))
            df['_file_id'] = i
            test_dfs.append(df)
        self._df_test = pd.concat(test_dfs, ignore_index=True)

        # Combined df for compatibility
        self._df = pd.concat([self._df_train, self._df_test], ignore_index=True)

        # Set name
        if self.name is None:
            self.name = "ML Analysis"

        # Derive edge files if not specified
        if self.train_edges is None:
            self._train_edge_files = [f.replace('.csv', '.edges.csv') for f in self.train_data]
        elif isinstance(self.train_edges, str):
            self._train_edge_files = [self.train_edges]
        else:
            self._train_edge_files = self.train_edges

        if self.test_edges is None:
            self._test_edge_files = [f.replace('.csv', '.edges.csv') for f in self.test_data]
        elif isinstance(self.test_edges, str):
            self._test_edge_files = [self.test_edges]
        else:
            self._test_edge_files = self.test_edges

    def _load_from_fem(self):
        """Load training/test data from FEMObject(s)."""
        
        # Normalize to lists
        if not isinstance(self.train_fem, list):
            self.train_fem = [self.train_fem]
        if self.test_fem is not None and not isinstance(self.test_fem, list):
            self.test_fem = [self.test_fem]

        # Validate all FEMs are solved
        for fem in self.train_fem:
            if fem._displacement is None:
                raise ValueError(f"FEM '{fem.name}' must be solved before using as training data")

        if self.test_fem:
            for fem in self.test_fem:
                if fem._displacement is None:
                    raise ValueError(f"FEM '{fem.name}' must be solved before using as test data")

        # Build DataFrames from FEM results
        train_dfs = []
        for i, fem in enumerate(self.train_fem):
            df = self._fem_to_dataframe(fem)
            df['_source_idx'] = range(len(df))
            df['_file_id'] = i
            train_dfs.append(df)

        self._df_train = pd.concat(train_dfs, ignore_index=True)

        if self.test_fem:
            test_dfs = []
            for i, fem in enumerate(self.test_fem):
                df = self._fem_to_dataframe(fem)
                df['_source_idx'] = range(len(df))
                df['_file_id'] = i
                test_dfs.append(df)
            self._df_test = pd.concat(test_dfs, ignore_index=True)
        else:
            # No test FEM - use train_ratio split
            self._df_train, self._df_test = train_test_split(
                self._df_train, train_size=self.train_ratio, random_state=42
            )

        # Combined for compatibility
        self._df = pd.concat([self._df_train, self._df_test], ignore_index=True)

        # Set name
        if self.name is None:
            self.name = self.train_fem[0].name or "FEM Analysis"

        # Store edge indices directly (no CSV files needed!)
        self._train_edge_indices = [fem.edge_index for fem in self.train_fem]
        self._test_edge_indices = [fem.edge_index for fem in self.test_fem] if self.test_fem else None

    def _fem_to_dataframe(self, fem: 'FEMObject') -> pd.DataFrame:
        """Convert FEMObject results to DataFrame."""
        coords = fem.coordinates
        dim = coords.shape[1]

        data = {
            'x': coords[:, 0],
            'y': coords[:, 1],
        }
        if dim == 3:
            data['z'] = coords[:, 2]

        # Add von Mises
        # Other objectives currently not supported (because they're untracked)
        data['von_mises'] = fem.von_mises

        # Add metadata as columns (e.g., load value)
        for key, value in fem.metadata.items():
            # Clean up key name (remove units for column name)
            clean_key = key.split('(')[0].strip()
            data[clean_key] = value

        return pd.DataFrame(data)

    def _split_data(self, log=False):
        """Split data into train/test sets."""
        self._df_train, self._df_test = train_test_split(
            self._df,
            train_size=self.train_ratio,
            random_state=42
        )
        if log:
            print(f"Data split: {len(self._df_train)} train, {len(self._df_test)} test")

    def summary(self):
        """Print summary of the ML setup."""
        print(f"\n{'='*60}")
        print(f"ML Analysis: {self.name}")
        print(f"{'='*60}")
        print(f"\nData:")
        print(f"  Total samples: {len(self._df)}")
        print(f"  Train samples: {len(self._df_train)}")
        print(f"  Test samples: {len(self._df_test)}")
        print(f"\nFeatures ({len(self.features)}):")
        for f in self.features:
            print(f"  - {f}")
        print(f"\nObjectives ({len(self.objectives)}):")
        for o in self.objectives:
            print(f"  - {o}")

        if self._trained_model is not None:
            print(f"\nTrained Model: {self._trained_model.name}")
            print(f"  Train R²: {self._trained_model.train_r2:.4f}")

        if self._test_metrics is not None:
            print(f"  Test metrics: {self._test_metrics}")

        print(f"{'='*60}\n")

    def analyze_features(self, plot=True, r2_threshold=0.5, log=False):
        """
        Analyze feature importance using correlation with objectives.
        
        Parameters:
        -----------
        plot : bool
            Whether to show plots
        r2_threshold : float
            Threshold for "important" features
            
        Returns:
        --------
        dict : Feature scores
        """
        if log:
            print(f"\n{'='*60}")
            print("Feature Importance Analysis")
            print(f"{'='*60}")

        feature_scores = {}

        for obj in self.objectives:
            y = self._df_train[obj].values

            for feat in self.features:
                X = self._df_train[[feat]].values
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                feature_scores[feat] = r2

        # Sort by importance
        if log:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"\nFeature R² scores:")
            for feat, score in sorted_features:
                marker = "✓" if score >= r2_threshold else " "
                print(f"  {marker} {feat}: {score:.4f}")
    
            important = [f for f, s in sorted_features if s >= r2_threshold]
            print(f"\nImportant features (R² >= {r2_threshold}): {important}")

        if plot:
            self._plot_feature_importance(feature_scores)

        return feature_scores

    def _plot_feature_importance(self, feature_scores):
        """Plot feature importance bar chart."""
        features = list(feature_scores.keys())
        scores = list(feature_scores.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, scores, color='steelblue')
        ax.set_xlabel('R² Score')
        ax.set_title(f'{self.name} - Feature Importance')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def train(self, model=None, model_name=None, auto_select=False, log=False, **kwargs):
        """Train a model on the data."""
        
        if auto_select:
            model, model_name = self._select_best_model()
        elif model is None:
            model = LinearRegression()
            model_name = "Linear Regression"

        # Validate model compatibility: can it take FEMs or should it be csv-only?
        if self._fem_mode and not isinstance(model, MGN):
            raise ValueError(
                "FEM data (train_fem/test_fem) only works with MGN. "
                "Use train_data/test_data (CSV) for sklearn models and GNN."
            )
    
        if not self._fem_mode and isinstance(model, MGN):
            raise ValueError(
                "MGN requires FEM data (train_fem/test_fem). "
                "Use train_fem=[fem1, fem2, ...] instead of train_data."
            )
    
        if model_name is None:
            model_name = model.__class__.__name__
    
        if log:
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

        if isinstance(model, MGN):
            # MGN uses FEM objects directly
            y_train = self._df_train[self.objectives[0]].values
    
            # Pass FEM context to MGN
            model.fit(self.train_fem, y_train, **kwargs)
    
            train_r2 = model.train_score()
    
        elif isinstance(model, GNN):
            
            # Transductive/inductive not explicitly set, so it depends on number of inputs (this is the default)
            if model.transductive is None:
                has_multiple_files = (self.train_data is not None) and (len(self.train_data) > 1)
                model.transductive = not has_multiple_files
                if log:
                    mode = "transductive" if model.transductive else "inductive"
                    print(f"Auto-detected mode: {mode} learning")
            
            if model.transductive:
                # Single file, mask-based split
                X_all = self._df[self.features].values
                y_all = self._df[self.objectives].values.ravel()
                model.fit(X_all, y_all, train_ratio=self.train_ratio)
            else:
                # Multiple files, separate graphs
                X_train = self._df_train[self.features].values
                y_train = self._df_train[self.objectives].values.ravel()
        
                if '_file_id' in self._df_train.columns:
                    train_node_counts = self._df_train.groupby('_file_id').size().tolist()
                    test_node_counts = self._df_test.groupby('_file_id').size().tolist()
                    model.set_edge_files(self._train_edge_files, self._test_edge_files,
                                         train_node_counts, test_node_counts)
        
                model.fit(X_train, y_train)
        
            train_r2 = model.train_score()
        else:
            X_train = self._df_train[self.features].values
            y_train = self._df_train[self.objectives].values.ravel()
            model.fit(X_train, y_train)
    
            y_pred_train = model.predict(X_train)
            train_r2 = r2_score(y_train, y_pred_train)
    
        # Get coefficients if available
        coefficients = getattr(model, 'coef_', None)
        intercept = getattr(model, 'intercept_', None)
    
        self._trained_model = TrainedModel(
            model=model,
            name=model_name,
            features=self.features,
            objectives=self.objectives,
            train_r2=train_r2,
            coefficients=coefficients,
            intercept=intercept,
        )
    
        if log:
            print(f"  Train R²: {train_r2:.4f}")
            if coefficients is not None:
                print(f"  Coefficients: {coefficients}")
            if intercept is not None:
                print(f"  Intercept: {intercept:.4f}")
    
        return self._trained_model

    def apply_pretrained(self, model, model_name=None, edges_file=None):
        """
        Use a pre-trained model without training.
        
        Args:
            model: Pre-trained model (e.g., loaded GNN)
            model_name: Name for the model
            edges_file: Path to edges file (optional, auto-detected if None)
        
        Example:
            gnn = GNN.load('models/my_model.pt')
            ml.apply_pretrained(gnn, "Pretrained GNN")
            ml.evaluate_on_unseen_data()
        """
        if model_name is None:
            model_name = model.__class__.__name__
    
        # MGN: just needs FEM at predict time, no setup needed
        if isinstance(model, MGN):
            if not self._fem_mode:
                raise ValueError(
                    "MGN requires FEM data (train_fem/test_fem), not CSV files."
                )
            # Nothing else to set up - FEM passed at predict time
    
        elif isinstance(model, GNN):
            # Warn if using data= with GNN (causes train/test split issues)
            if self.data is not None:
                print("⚠️  Warning: Using 'data=' with pretrained graph network will split data 80/20.")
                print("   This causes edge index mismatches!")
                print("   Use 'train_data=[file], test_data=[file]' instead for full dataset.")
                raise ValueError(
                    "Cannot use 'data=' with pretrained graph network. "
                    "Use 'train_data=[file], test_data=[file]' to evaluate on full dataset."
                )
    
            # Auto-detect edges file if not provided
            if edges_file is None:
                if hasattr(self, 'test_data') and self.test_data is not None:
                    if isinstance(self.test_data, str):
                        edges_file = self.test_data.replace('.csv', '.edges.csv')
                    elif isinstance(self.test_data, list) and len(self.test_data) > 0:
                        edges_file = self.test_data[0].replace('.csv', '.edges.csv')
                elif hasattr(self, 'data') and isinstance(self.data, str):
                    edges_file = self.data.replace('.csv', '.edges.csv')
    
            # Load and set edges
            if edges_file and os.path.exists(edges_file):
                edges_df = pd.read_csv(edges_file)
                edge_index = torch.tensor(
                    np.array([edges_df['node_i'].values, edges_df['node_j'].values]),
                    dtype=torch.long
                )
                # Add reverse edges (undirected graph)
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
                # Set the loaded edges directly
                model._train_edge_index = edge_index.to(model.device)
                model._test_edge_index = edge_index.to(model.device)
            else:
                print("⚠️  Warning: No edges file found, GNN predictions may fail")
    
        # Get train R² if possible
        train_r2 = None
        if hasattr(model, 'train_score'):
            try:
                train_r2 = model.train_score()
            except:
                pass
    
        self._trained_model = TrainedModel(
            model=model,
            name=model_name,
            features=self.features,
            objectives=self.objectives,
            train_r2=train_r2,
            coefficients=getattr(model, 'coef_', None),
            intercept=getattr(model, 'intercept_', None)
        )
    
        return self._trained_model

    def _select_best_model(self):
        """
        Try multiple models and return the best one based on R². This is different than get_best_model() since that
        retrieves already-trained models.
        """
        candidates = self._get_default_models()
    
        X_train = self._df_train[self.features].values
        y_train = self._df_train[self.objectives].values.ravel()
        X_test = self._df_test[self.features].values
        y_test = self._df_test[self.objectives].values.ravel()
    
        best_score = -float('inf')
        best_model = None
        best_name = None
    
        print("Auto-selecting best model...")
    
        for model, name in candidates:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"  {name}: R² = {score:.4f}")
    
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
    
        print(f"Selected: {best_name} (R² = {best_score:.4f})")
    
        # Return a fresh instance to retrain on full training data
        for model, name in candidates:
            if name == best_name:
                return model.__class__(**model.get_params()), best_name

    def evaluate_on_unseen_data(self, log=False):
        """
        Evaluate trained model on test data.
        
        Returns:
        --------
        ModelMetrics
        """
        if self._trained_model is None:
            raise ValueError("Must call train() first")

        if log:
            print(f"\n{'='*60}")
            print(f"Evaluating: {self._trained_model.name}")
            print(f"{'='*60}")

        y_test, y_pred = self._get_test_predictions()

        self._test_metrics = ModelMetrics(
            r2=r2_score(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mae=mean_absolute_error(y_test, y_pred)
        )

        # Baseline comparison
        baseline_pred = np.full_like(y_test, y_test.mean())
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
        improvement = (1 - self._test_metrics.rmse / baseline_rmse) * 100

        if log:
            print(f"  Test R²: {self._test_metrics.r2:.4f}")
            print(f"  Test RMSE: {self._test_metrics.rmse:.4f}")
            print(f"  Test MAE: {self._test_metrics.mae:.4f}")
            print(f"  Baseline RMSE: {baseline_rmse:.4f}")
            print(f"  Improvement over baseline: {improvement:.1f}%")

        # Check for overfitting
        train_r2 = self._trained_model.train_r2 # Will be None if this is a pretrained model
        test_r2 = self._test_metrics.r2
        if train_r2 is not None and abs(train_r2 - test_r2) > 0.1 and log:
            print(f"  ⚠️  Warning: Possible overfitting (Train R²={train_r2:.4f}, Test R²={test_r2:.4f})")

        return self._test_metrics

    def plot_predictions(self, show=True, save_path=None):
        """
        Plot predicted vs actual values.
        
        Parameters:
        -----------
        show : bool
            Whether to display the plot
        save_path : str, optional
            Path to save the figure
        """
        if self._trained_model is None:
            raise ValueError("Must call train() first")
    
        y_test, y_pred = self._get_test_predictions()
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
        # Check if MGN - color by node type
        if isinstance(self._trained_model.model, MGN):
            mgn = self._trained_model.model
            node_types, _, _, _ = mgn._fem_to_graph(self.test_fem if isinstance(self.test_fem, list) else [self.test_fem])
    
            # Map to colors
            unique_types = sorted(set(node_types))
            node_type_colors = {
                'interior': '#CCCCCC',      # Gray (boring, most common)
                'fixed': '#2196F3',         # Blue (constrained)
                'applied_load': '#FF5722',  # Orange (where force applied)
                'free': '#4CAF50',          # Green (unconstrained boundary)
                'hole': '#9C27B0',          # Purple (stress concentrator!)
                'corner': '#FFEB3B',        # Yellow (special points)
            }
            
            # Use custom colors, fallback to tab10 for unknown types
            fallback_cmap = plt.cm.get_cmap('tab10', 10)
            fallback_idx = 0
            type_to_color = {}
            for t in unique_types:
                if t in node_type_colors:
                    type_to_color[t] = node_type_colors[t]
                else:
                    type_to_color[t] = fallback_cmap(fallback_idx)
                    fallback_idx += 1
    
            # Scatter with colors
            ax1 = axes[0]
            for node_type in unique_types:
                mask = [t == node_type for t in node_types]
                ax1.scatter(
                    y_test[mask], y_pred[mask],
                    alpha=0.5, s=10,
                    color=type_to_color[node_type],
                    label=node_type
                )
            ax1.legend(loc='upper left', fontsize=8)
    
            # Residuals with colors
            ax2 = axes[1]
            residuals = y_pred - y_test
            for node_type in unique_types:
                mask = [t == node_type for t in node_types]
                ax2.scatter(
                    y_test[mask], residuals[mask],
                    alpha=0.5, s=10,
                    color=type_to_color[node_type],
                    label=node_type
                )
        else:
            # Default: no coloring
            ax1 = axes[0]
            ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
    
            ax2 = axes[1]
            residuals = y_pred - y_test
            ax2.scatter(y_test, residuals, alpha=0.5, s=10)
    
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
        ax1.set_xlabel(f'Actual {self.objectives[0]}')
        ax1.set_ylabel(f'Predicted {self.objectives[0]}')
        ax1.set_title(f'{self._trained_model.name}: Predicted vs Actual')
        ax1.grid(True, alpha=0.3)
    
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel(f'Actual {self.objectives[0]}')
        ax2.set_ylabel('Residual (Predicted - Actual)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
    
        if self._test_metrics:
            fig.suptitle(f'{self.name} - {self._trained_model.name} (R²={self._test_metrics.r2:.4f})',
                         fontsize=14, fontweight='bold')
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
    
        if show:
            plt.show()
        else:
            return fig

    def plot_feature_distributions(self, show=True):
        """Plot distribution of features in train vs test sets."""
        
        # Feature analysis for GNNs is inappropriate---the whole point is that graphs don't just look at regressions
        # of individual features
        if isinstance(self._trained_model.model, (GNN, MGN)):
            raise ValueError("Feature analysis looks at regressions, which doesn't really work for graph networks.")
        
        n_plots = len(self.features) + len(self.objectives)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.array(axes).flatten()

        all_cols = self.features + self.objectives

        for i, col in enumerate(all_cols):
            ax = axes[i]
            ax.hist(self._df_train[col], bins=30, alpha=0.6, label='Train',
                    density=False, color='blue', edgecolor='black')  # Remove density=True
            ax.hist(self._df_test[col], bins=30, alpha=0.6, label='Test',
                    density=False, color='orange', edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title(col)

        # Hide unused axes
        for i in range(len(all_cols), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{self.name} - Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if show:
            plt.show()
        else:
            return fig

    def plot_loss(self, show=True, log_y=True):
        """
        Plot training loss history (for models that support it).
        
        Note: Only works with iterative models like GradientBoostingRegressor,
        not with closed-form solutions like LinearRegression.
        """
        if self._trained_model is None:
            raise ValueError("Must call train() first")
    
        model = self._trained_model.model
    
        # Check if model has training history
        if hasattr(model, 'train_score_'):
            # GradientBoostingRegressor
            scores = model.train_score_
            xlabel = 'Iteration'
        elif hasattr(model, 'loss_curve_'):
            # MLPRegressor
            scores = model.loss_curve_
            xlabel = 'Iteration'
        elif hasattr(model, 'oob_improvement_'):
            # RandomForest with oob_score=True
            scores = model.oob_improvement_
            xlabel = 'Iteration'
        elif hasattr(model, 'train_losses'):
            # GNN
            scores = model.train_losses
            xlabel = 'Epoch'
        else:
            print(f"Model '{self._trained_model.name}' doesn't track training loss.")
            print("Loss curves available for: GradientBoostingRegressor, MLPRegressor")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(scores, 'b-', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss / Score')
        ax.set_title(f'{self._trained_model.name} - Training History')
        ax.grid(True, alpha=0.3)
        
        # Semilog scale?
        if log_y: ax.set_yscale("log")
    
        plt.tight_layout()
    
        if show:
            plt.show()
        else:
            return fig

    def plot_ml_vs_fem(self, mesh_object, show=True, save_path=None, colormap='viridis'):
        if self._trained_model is None:
            raise ValueError("Must call train() first")

        # MGN: no features needed, just objective
        if isinstance(self._trained_model.model, MGN):
            y_actual = self._df_test[self.objectives[0]].values
            y_pred = self._trained_model.model.predict(self.test_fem)
    
        # GNN inductive
        elif isinstance(self._trained_model.model, GNN) and not self._trained_model.model.transductive:
            X_plot = self._df_test[self.features].values
            y_actual = self._df_test[self.objectives].values.ravel()
            y_pred = self._trained_model.model.predict(X_plot)
    
        # Transductive GNN or sklearn
        else:
            X_plot = self._df[self.features].values
            y_actual = self._df[self.objectives].values.ravel()
            y_pred = self._trained_model.model.predict(X_plot)

        title = f'{self.name} - {self._trained_model.name} (R² = {self._test_metrics.r2:.4f})' if self._test_metrics else None

        return mesh_object.plot_field_comparison(
            y_actual, y_pred,
            title=title, label=self.objectives[0],
            show=show, save_path=save_path, colormap=colormap
        )

    def compare_models(self, models=None, plot=True, save_dir=None, log=True):
        """
        Compare multiple models.
        
        Parameters:
        -----------
        models : list of tuples, optional
            List of (model, name) tuples. Default: common regression models
        plot : bool
            Whether to show comparison plot
        save_dir : str, optional
            Directory to save individual model plots
            
        Returns:
        --------
        dict : Results for each model
        """
        if models is None:
            models = self._get_default_models()

        if log:
            print(f"\n{'='*60}")
            print("Model Comparison")
            print(f"{'='*60}")

        results = {}

        for model, name in models:
            self.train(model, name)
            metrics = self.evaluate_on_unseen_data()

            results[name] = {
                'model': self._trained_model,
                'train_r2': self._trained_model.train_r2,
                'test_r2': metrics.r2,
                'test_rmse': metrics.rmse,
                'test_mae': metrics.mae
            }

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = name.lower().replace(' ', '_') + '.png'
                save_path = os.path.join(save_dir, filename)
                self.plot_predictions(show=False, save_path=save_path)

        self._comparison_results = results

        # Print summary
        if log:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12}")
            print('-'*60)
            for name, res in results.items():
                print(f"{name:<25} {res['train_r2']:<12.4f} {res['test_r2']:<12.4f} {res['test_rmse']:<12.4f}")

            # Best model
            best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
            print(f"\nBest model: {best_name} (Test R²={results[best_name]['test_r2']:.4f})")

        if plot:
            self._plot_model_comparison(results)

        return results

    def _plot_model_comparison(self, results):
        """Plot model comparison bar chart."""
        names = list(results.keys())
        train_r2 = [results[n]['train_r2'] for n in names]
        test_r2 = [results[n]['test_r2'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', color='steelblue')
        bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', color='darkorange')

        ax.set_ylabel('R² Score')
        ax.set_title(f'{self.name} - Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def get_best_model(self):
        """Get the best model from comparison results."""
        if not self._comparison_results:
            raise ValueError("Must call compare_models() first")

        best_name = max(self._comparison_results.keys(),
                        key=lambda k: self._comparison_results[k]['test_r2'])
        return self._comparison_results[best_name]['model']

    def get_trained_model_name(self, lowercase=False):
        raw_name = self._trained_model.name
        if lowercase: raw_name = raw_name.lower()
        return raw_name.replace(" ", "_")

    def save_model(self, filepath):
        """Save trained model to file."""

        if self._trained_model is None:
            raise ValueError("Must call train() first")

        with open(filepath, 'wb') as f:
            pickle.dump(self._trained_model, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model from file."""
        

        with open(filepath, 'rb') as f:
            self._trained_model = pickle.load(f)

        print(f"Model loaded from {filepath}: {self._trained_model.name}")
        return self._trained_model

    def _get_default_models(self):
        """Default models for comparison and auto-selection."""
        return [
            (LinearRegression(), "Linear Regression"),
            (Ridge(alpha=1.0), "Ridge Regression"),
            (Lasso(alpha=0.1), "Lasso Regression"),
            (DecisionTreeRegressor(max_depth=10), "Decision Tree"),
            (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"),
            (GradientBoostingRegressor(random_state=42), "Gradient Boosting"),
            (KNeighborsRegressor(), "K-Neighbors"),
        ]

    def _get_test_predictions(self):
        """Get y_test and y_pred for the test set."""

        # MGN: uses internal FEM context, not features
        if isinstance(self._trained_model.model, MGN):
            y_test = self._df_test[self.objectives[0]].values
            y_pred = self._trained_model.model.predict(self.test_fem)
            return y_test, y_pred
    
        # GNN: still uses features, but has two different modes
        elif isinstance(self._trained_model.model, GNN):
            gnn = self._trained_model.model
    
            if gnn.transductive:
                y_pred_all = gnn.predict()
                y_all = self._df[self.objectives].values.ravel()
                test_mask = gnn._test_mask.cpu().numpy()
                return y_all[test_mask], y_pred_all[test_mask]
            else:
                X_test = self._df_test[self.features].values
                y_test = self._df_test[self.objectives].values.ravel()
                y_pred = gnn.predict(X_test)
                return y_test, y_pred
    
        # sklearn models are the simplest (just use features)
        else:
            X_test = self._df_test[self.features].values
            y_test = self._df_test[self.objectives].values.ravel()
            y_pred = self._trained_model.model.predict(X_test)
            return y_test, y_pred