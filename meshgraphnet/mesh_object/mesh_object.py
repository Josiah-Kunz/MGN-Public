from ..common_utils.common_utils import format_name
import dolfin                           # dolfin meshes are required because they integrate into fenics
import gmsh                             # a nice free automesher
import matplotlib.pyplot as plt         # for visualizing the mesh
import meshio                           # reads gmsh and writes dolfin
import numpy as np                      # linear algebra during deep inspection
import os                               # useful for things like checking if directories exist
from ..units.unit_system import Units   # for plotting and showing the mesh's lengths
from ..units.unit_system import UNIT_REGISTRY   # For converting units


class MeshObject:
    """
    Workflow: 
        STEP file 
          ↓ (gmsh.merge)
        gmsh.model (in-memory)
          ↓ (gmsh.write to .msh)
        .msh file on disk
          ↓ (meshio.read)
        meshio Mesh object
          ↓ (meshio.write to .xdmf)
        .xdmf file on disk
          ↓ (dolfin.XDMFFile.read)
        dolfin.Mesh object
    """

    # =============================================================================
    # Public Functions
    # =============================================================================
    
    def __init__(self, file, mesh_size, units=Units.SI, name = None, cache_dir=".mesh_cache", force_regenerate=False, force_2d = False, convert_to = None):

        # Initialize geometry attributes to None
        self.extents = None
        self.extents_sorted = None
        self.bounds = None
        self.max_extent = None
        self.aspect_ratio = None
        self.major_plane = None
        self.axis1_name = None
        self.axis2_name = None
        self.axis1_idx = None
        self.axis2_idx = None
        self.dim = None
        self.neighbor_counts = None
        
        # Save new properties
        self.src_file = file
        self.mesh_size = mesh_size
        self.unit_system = units
        self.units = units.length
        self._generate_cache_paths(cache_dir, force_2d)
        
        # Default name can be derived from the file name
        if name is not None:
            self.name = name
        else:
            base_name = os.path.splitext(os.path.basename(file))[0]
            self.name = format_name(base_name, False)
        
        # Get (or create a new) dolfin mesh
        self.dolfin_mesh = None
        self._ensure_dolfin_mesh(force_regenerate, force_2d)

        # Convert units if requested
        if convert_to is not None:
            self.convert_units(convert_to)

        # Assign the geometry from cache (otherwise, it's assigned on generation from the meshio mesh)
        if self.extents is None:
            self._assign_geometry()

    def visualize(self, show=True, alpha=1, title="__auto__", color=False):

        # Do the plot
        if self.dim == 2:
            ax = self._plot_2d(alpha, color=color)
        else:
            ax = self._plot_3d(alpha, color=color)

        # Set title
        if title == "__auto__":
            if show:
                ax.set_title(f"{self.name} Mesh Visualization")
        elif title is not None:
            ax.set_title(title)
            
        # Show it or return it
        if show:
            plt.tight_layout()
            plt.show()
        else:
            return plt.gcf()

    def summary(self):
        print(f"\n{'='*60}")
        print(f"Mesh Summary: {self.name}")
        print(f"{'='*60}")
    
        # File information
        print(f"\nSource:")
        print(f"  File: {self.src_file}")
        print(f"  Cache: {self.dolfin_cache_path}")
        print(f"  Mesh size parameter: {self.mesh_size} {self.units}")
    
        # Mesh statistics
        print(f"\nMesh Statistics:")
        print(f"  Dimension: {self.dim}D")
        print(f"  Vertices: {self.dolfin_mesh.num_vertices()}")
        print(f"  Cells: {self.dolfin_mesh.num_cells()}")
        print(f"  Edges: {self.dolfin_mesh.num_edges()}")
    
        # Coordinate ranges - use stored values
        print(f"\nCoordinate Ranges ({self.units}):")
        for axis in ['x', 'y', 'z']:
            if self.extents[axis] > 0:
                print(f"  {axis.upper()}: [{self.bounds[axis][0]:.3f}, {self.bounds[axis][1]:.3f}] (extent: {self.extents[axis]:.3f})")
    
        # Mesh density - use stored max_extent
        elements_per_unit = self.dolfin_mesh.num_cells() / self.max_extent
        print(f"\nMesh Density:")
        print(f"  Elements per unit length: ~{elements_per_unit:.2f}")
    
        print(f"{'='*60}\n")

    def analyze(self):
        """
        Perform detailed mesh quality analysis with visualizations.
        
        Analyzes and plots:
        - Element aspect ratios (equilateral = 1, stretched > 1)
        - Element sizes/areas
        - Mesh quality metrics
        - Quality distribution histograms
        """
    
        coords = self.dolfin_mesh.coordinates()
        cells = self.dolfin_mesh.cells()
    
        print(f"\n{'='*60}")
        print(f"Deep Mesh Inspection: {self.name}")
        print(f"{'='*60}\n")
    
        # Calculate element quality metrics
        aspect_ratios = []
        element_sizes = []
    
        for cell in cells:
            # Get vertices of this element
            if self.dim == 2:
                # Triangle
                p0, p1, p2 = coords[cell[0]], coords[cell[1]], coords[cell[2]]
    
                # Edge lengths
                L1 = np.linalg.norm(p1 - p0)
                L2 = np.linalg.norm(p2 - p1)
                L3 = np.linalg.norm(p0 - p2)
    
                # Aspect ratio (max edge / min edge)
                max_edge = max(L1, L2, L3)
                min_edge = min(L1, L2, L3)
                aspect_ratio = max_edge / min_edge if min_edge > 0 else np.inf
    
                # Element size (area using cross product)
                area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                element_sizes.append(area)
    
            else:  # 3D tetrahedra
                p0, p1, p2, p3 = coords[cell[0]], coords[cell[1]], coords[cell[2]], coords[cell[3]]
    
                # Edge lengths (6 edges in a tetrahedron)
                edges = [
                    np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0),
                    np.linalg.norm(p3 - p0), np.linalg.norm(p2 - p1),
                    np.linalg.norm(p3 - p1), np.linalg.norm(p3 - p2)
                ]
    
                max_edge = max(edges)
                min_edge = min(edges)
                aspect_ratio = max_edge / min_edge if min_edge > 0 else np.inf
    
                # Element size (volume)
                volume = abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
                element_sizes.append(volume)
    
            aspect_ratios.append(aspect_ratio)
    
        aspect_ratios = np.array(aspect_ratios)
        element_sizes = np.array(element_sizes)
    
        # Print statistics
        print("Element Quality Metrics:")
        print(f"  Aspect Ratio:")
        print(f"    Mean: {np.mean(aspect_ratios):.3f}")
        print(f"    Min: {np.min(aspect_ratios):.3f} (best = 1.0)")
        print(f"    Max: {np.max(aspect_ratios):.3f}")
        print(f"    Std Dev: {np.std(aspect_ratios):.3f}")
    
        size_label = "Area" if self.dim == 2 else "Volume"
        print(f"\n  Element {size_label} {self.units}^{self.dim}:")
        print(f"    Mean: {np.mean(element_sizes):.3f}")
        print(f"    Min: {np.min(element_sizes):.3f}")
        print(f"    Max: {np.max(element_sizes):.3f}")
        print(f"    Std Dev: {np.std(element_sizes):.3f}")
    
        # Quality assessment
        print(f"\nQuality Assessment:")
        good_elements = np.sum(aspect_ratios < 2.0)
        fair_elements = np.sum((aspect_ratios >= 2.0) & (aspect_ratios < 5.0))
        poor_elements = np.sum(aspect_ratios >= 5.0)
    
        print(f"  Good (AR < 2.0): {good_elements} ({100*good_elements/len(cells):.1f}%)")
        print(f"  Fair (2.0 ≤ AR < 5.0): {fair_elements} ({100*fair_elements/len(cells):.1f}%)")
        print(f"  Poor (AR ≥ 5.0): {poor_elements} ({100*poor_elements/len(cells):.1f}%)")
    
        if poor_elements > 0.1 * len(cells):
            print(f"  ⚠️  Warning: >10% of elements have poor aspect ratio")
        else:
            print(f"  ✓ Mesh quality is acceptable")
    
        print(f"{'='*60}\n")
    
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # Plot 1: Aspect ratio histogram
        ax1 = axes[0, 0]
        ax1.hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(2.0, color='orange', linestyle='--', label='Fair threshold')
        ax1.axvline(5.0, color='red', linestyle='--', label='Poor threshold')
        ax1.set_xlabel('Aspect Ratio')
        ax1.set_ylabel('Count')
        ax1.set_title('Element Aspect Ratio Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Plot 2: Element size histogram
        ax2 = axes[0, 1]
        ax2.hist(element_sizes, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel(f'Element {size_label} ({self.units}$^{self.dim}$)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Element {size_label} Distribution')
        ax2.grid(True, alpha=0.3)
    
        # Plot 3: Mesh visualization
        ax3 = axes[1, 0]
        if self.dim == 2:
            coord1 = coords[:, self.axis1_idx]
            coord2 = coords[:, self.axis2_idx]
            ax3.triplot(coord1, coord2, cells, 'k-', linewidth=0.3)
            ax3.set_xlabel(f"{self.axis1_name} ({self.units})")
            ax3.set_ylabel(f"{self.axis2_name} ({self.units})")
            ax3.set_aspect('equal')
        else:
            ax3 = plt.subplot(2, 2, 3, projection='3d')
            ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
            ax3.set_xlabel(f"x ({self.units})")
            ax3.set_ylabel(f"y ({self.units})")
            ax3.set_zlabel(f"z ({self.units})")
        ax3.set_title('Mesh Geometry')
        ax3.grid(True, alpha=0.3)
    
        # Plot 4: Quality distribution by location (color-coded)
        ax4 = axes[1, 1]
        if self.dim == 2:
            coord1 = coords[:, self.axis1_idx]
            coord2 = coords[:, self.axis2_idx]
    
            # Calculate aspect ratio at each vertex (average of connected cells)
            vertex_quality = np.zeros(len(coords))
            vertex_count = np.zeros(len(coords))
            for i, cell in enumerate(cells):
                for vertex_idx in cell:
                    vertex_quality[vertex_idx] += aspect_ratios[i]
                    vertex_count[vertex_idx] += 1
            vertex_quality /= np.maximum(vertex_count, 1)
    
            scatter = ax4.tricontourf(coord1, coord2, cells, vertex_quality, levels=20, cmap='RdYlGn_r')
            plt.colorbar(scatter, ax=ax4, label='Aspect Ratio')
            ax4.set_xlabel(f"{self.axis1_name} ({self.units})")
            ax4.set_ylabel(f"{self.axis2_name} ({self.units})")
            ax4.set_aspect('equal')
        else:
            # For 3D, just show text summary
            ax4.text(0.5, 0.5, 'Quality map\navailable for 2D only',
                     ha='center', va='center', fontsize=14)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
    
        ax4.set_title('Element Quality Distribution')
    
        plt.suptitle(f'{self.name} - Detailed Mesh Quality Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def convert_units(self, target_units):
        """
        Convert mesh coordinates to target unit system.
        
        Parameters:
        -----------
        target_units : UnitSystem
            Target unit system (e.g., Units.SI, Units.US)
        """

        # Get scale factor
        source_unit = self.unit_system.get_length_unit()
        target_unit = target_units.get_length_unit()
        scale = (1 * source_unit).to(target_unit).magnitude
    
        # Scale mesh coordinates in place
        coords = self.dolfin_mesh.coordinates()
        coords[:] *= scale
    
        # Update unit system
        self.unit_system = target_units
        self.units = target_units.length
        self.mesh_size *= scale
    
        # Clear any cached geometry and recalculate
        if self.extents is not None:
            self._assign_geometry()

    # In MeshObject class

    def plot_field_comparison(self, y_actual, y_pred, title=None, label='Value',
                              show=True, save_path=None, colormap='viridis'):
        """
        Plot comparison of two fields on this mesh.
        
        Parameters
        ----------
        y_actual : array
            Reference values (e.g., FEM result)
        y_pred : array
            Comparison values (e.g., ML prediction)
        title : str, optional
        label : str
            Colorbar label
        show : bool
        save_path : str, optional
        colormap : str
        """
        error = y_pred - y_actual
    
        coords = self.dolfin_mesh.coordinates()
        cells = self.dolfin_mesh.cells()
    
        vmin = min(y_actual.min(), y_pred.min())
        vmax = max(y_actual.max(), y_pred.max())
        max_err = max(abs(error.min()), abs(error.max()))
    
        shrink = 1 / self.aspect_ratio if self.aspect_ratio > 1 else 1
    
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)
    
        # Plot 1: Actual
        tc1 = ax1.tripcolor(coords[:, 0], coords[:, 1], cells, y_actual,
                            cmap=colormap, vmin=vmin, vmax=vmax, shading='flat')
        ax1.set_title('Actual')
        ax1.set_aspect('equal')
        ax1.set_xlabel(f'{self.axis1_name} ({self.units})')
        ax1.set_ylabel(f'{self.axis2_name} ({self.units})')
        plt.colorbar(tc1, ax=ax1, label=label, shrink=shrink)
    
        # Plot 2: Predicted
        tc2 = ax2.tripcolor(coords[:, 0], coords[:, 1], cells, y_pred,
                            cmap=colormap, vmin=vmin, vmax=vmax, shading='flat')
        ax2.set_title('Predicted')
        ax2.set_aspect('equal')
        ax2.set_xlabel(f'{self.axis1_name} ({self.units})')
        ax2.set_ylabel(f'{self.axis2_name} ({self.units})')
        plt.colorbar(tc2, ax=ax2, label=label, shrink=shrink)
    
        # Plot 3: Error
        tc3 = ax3.tripcolor(coords[:, 0], coords[:, 1], cells, error,
                            cmap='RdBu', vmin=-max_err, vmax=max_err, shading='flat')
        ax3.set_title('Error (Predicted - Actual)')
        ax3.set_aspect('equal')
        ax3.set_xlabel(f'{self.axis1_name} ({self.units})')
        ax3.set_ylabel(f'{self.axis2_name} ({self.units})')
        plt.colorbar(tc3, ax=ax3, label=label, shrink=shrink)
    
        if title:
            plt.suptitle(title)
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
    
        if show:
            plt.show()
        else:
            return fig

    # =============================================================================
    # Private functions - Cache management
    # =============================================================================

    def _generate_cache_paths(self, cache_dir=".mesh_cache", force_2d=False):
        """Generate cache file paths based on input file, mesh size, and 2D flag."""

        os.makedirs(cache_dir, exist_ok=True)
    
        base_name = os.path.splitext(os.path.basename(self.src_file))[0]
    
        # Include 2D flag in filename
        suffix = "_2d" if force_2d else ""
        output_name = f"{base_name}_{self.mesh_size}{suffix}"
    
        self.gmsh_cache_path = os.path.join(cache_dir, f"{output_name}.msh")
        self.dolfin_cache_path = os.path.join(cache_dir, f"{output_name}.xdmf")
    
    def _ensure_dolfin_mesh(self, force_regenerate=False, force_2d=False):
        
        # Is this even necessary?
        if not force_regenerate and self.dolfin_mesh is not None:
            return
    
        # Try to load from cache
        if self._cache_exists() and not force_regenerate:
            self.dolfin_mesh = self._load_cached_dolfin_mesh()
        else:
            # Create new
            self._generate_gmsh_file(force_2d)
            self._convert_gmsh_file_to_dolfin_file()
            self.dolfin_mesh = self._load_cached_dolfin_mesh()

        # Neighbors list
        self._count_neighbors()

    def _assign_geometry(self, points=None):
        """
        Assigns geometry quantities from point coordinates.
        
        Parameters:
            points: numpy array of coordinates. If None, uses dolfin_mesh.coordinates()
        
        Sets:
            bounds: dict with min/max for each axis
            extents: dict with range for each axis
            extents_sorted: list of (axis_name, extent) sorted by size descending
            max_extent: largest extent value
            axis1_name, axis2_name: names of the two largest dimensions
            axis1_idx, axis2_idx: indices of the two largest dimensions
        """
        
        # Get coordinates from appropriate source
        if points is None:
            if self.dolfin_mesh is None:
                raise RuntimeError("Cannot assign geometry: no points provided and dolfin_mesh is None")
            points = self.dolfin_mesh.coordinates()
    
        # Calculate bounds
        self.bounds = {
            'x': (points[:, 0].min(), points[:, 0].max()),
            'y': (points[:, 1].min(), points[:, 1].max()),
        }
        if points.shape[1] > 2:
            self.bounds['z'] = (points[:, 2].min(), points[:, 2].max())
        else:
            self.bounds['z'] = (0.0, 0.0)
    
        # Calculate extents
        self.extents = {
            'x': self.bounds['x'][1] - self.bounds['x'][0],
            'y': self.bounds['y'][1] - self.bounds['y'][0],
            'z': self.bounds['z'][1] - self.bounds['z'][0],
        }
    
        # Sort by size (largest first)
        self.extents_sorted = sorted(self.extents.items(), key=lambda x: x[1], reverse=True)
        self.max_extent = self.extents_sorted[0][1]
        self.min_extent = next(e for _, e in reversed(self.extents_sorted) if e > 0)
        
        # Get aspect ratio
        self.aspect_ratio = self.max_extent / self.extents_sorted[1][1] if self.extents_sorted[1][1] > 0 else 1
    
        # Assign major plane axes
        self.axis1_name = self.extents_sorted[0][0]
        self.axis2_name = self.extents_sorted[1][0]
    
        axis_indices = {'x': 0, 'y': 1, 'z': 2}
        self.axis1_idx = axis_indices[self.axis1_name]
        self.axis2_idx = axis_indices[self.axis2_name]

    def _count_neighbors(self):
        """Count number of neighbors for each node based on mesh connectivity."""
        cells = self.dolfin_mesh.cells()
        num_nodes = len(self.dolfin_mesh.coordinates())

        # Build neighbor set for each node
        neighbors = [set() for _ in range(num_nodes)]

        for cell in cells:
            # Each node in the cell is connected to every other node in the cell
            for i, node_i in enumerate(cell):
                for j, node_j in enumerate(cell):
                    if i != j:
                        neighbors[node_i].add(node_j)

        # Count neighbors - they're in the same order as coords
        self.neighbor_counts = np.array([len(n) for n in neighbors])
    
    def _get_major_plane_from_gmsh(self):
        
        # Find the best surface
        self.major_plane = None
        best_score = -1
        for surface in gmsh.model.getEntities(2):

            # Get how much area this surface takes up
            # (Okay, we're using squares for np.abs, but it's still proportional!)
            bbox = gmsh.model.getBoundingBox(surface[0], surface[1])
            x = abs(bbox[3] - bbox[0])
            y = abs(bbox[4] - bbox[1])
            z = abs(bbox[5] - bbox[2])

            # Score = area 
            score = max(x*y, y*z, z*x)

            if score > best_score:
                best_score = score
                self.major_plane = surface
    
    def _cache_exists(self):
        """Check if all required cache files exist."""
        return os.path.exists(self.dolfin_cache_path) and os.path.exists(self.gmsh_cache_path)
    
    
    def _load_cached_dolfin_mesh(self):
        """Load mesh from cached XDMF file."""
        
        # Get the mesh
        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(self.dolfin_cache_path) as infile:
            infile.read(mesh)

        # Detect the dimension based on the cells
        cell_type = mesh.ufl_cell()
        if cell_type is not None and hasattr(cell_type, 'geometric_dimension'):
            self.dim = cell_type.geometric_dimension()
        else:
            raise ValueError(f"Could not determine the mesh dimension for {self.name} at {self.src_file}.")
            
        return mesh

    # =============================================================================
    # Private functions - Mesh generation and loading
    # =============================================================================

    def _generate_gmsh_file(self, force_2d=False):
        """
        Generate mesh from STEP file using gmsh and return the gmsh model.
        
        Parameters:
        -----------
        input_file : str
            Path to STEP file
        mesh_size : float
            Target element size
            
        Returns:
        --------
        gmsh.model
            The gmsh model object with generated mesh
            
        Note: Does NOT call gmsh.finalize() - caller is responsible for cleanup
        """
        if not gmsh.isInitialized():
            gmsh.initialize()
        
        gmsh.model.add("model")
        gmsh.merge(self.src_file)
        gmsh.model.occ.synchronize()

        # Determine native dimensionality
        # This could get overwritten if we're forcing 2d
        if len(gmsh.model.getEntities(3)) > 0:
            self.dim = 3
        elif len(gmsh.model.getEntities(2)) > 0:
            self.dim = 2
        else:
            self.dim = 1

       # 2d only?
        if force_2d:
            self._force_2d()
        
        # Set mesh size for all points
        points = gmsh.model.getEntities(0)
        for point in points:
            gmsh.model.mesh.setSize([point], self.mesh_size)
        
        # Generate mesh with the correct dimension
        gmsh.model.mesh.generate(self.dim)
        
        # Save
        gmsh.write(self.gmsh_cache_path)
        gmsh.finalize()

    def _force_2d(self):

        # Guard
        if self.major_plane is None:
            self._get_major_plane_from_gmsh()
            if self.major_plane is None:
                raise ValueError(f"Attempting to force {self.src_file} to be 2D, but it has no surfaces!")
        
        # Remove all volumes
        for vol in gmsh.model.getEntities(3):
            gmsh.model.removeEntities([vol], recursive=False)
            
        # Set surface
        gmsh.model.addPhysicalGroup(2, [self.major_plane[1]], tag=1)
        gmsh.model.setPhysicalName(2, 1, "MainSurface")
        self.dim = 2

    def _convert_gmsh_file_to_dolfin_file(self):
        """
        Convert gmsh .msh file to XDMF format for FEniCS.
        """

        # Get the gmsh mesh in meshio form
        meshio_mesh = meshio.read(self.gmsh_cache_path)
        points = meshio_mesh.points
        self._assign_geometry(points)
    
        # Detect dimension from available cell types
        if "tetra" in meshio_mesh.cells_dict:
            cells = {"tetra": meshio_mesh.cells_dict["tetra"]}
            self.dim = 3
        elif "triangle" in meshio_mesh.cells_dict:
            cells = {"triangle": meshio_mesh.cells_dict["triangle"]}
            self.dim = 2
        elif "line" in meshio_mesh.cells_dict:
            cells = {"line": meshio_mesh.cells_dict["line"]}
            self.dim = 1
        else:
            available = list(meshio_mesh.cells_dict.keys())
            raise ValueError(f"No supported cell types found. Available: {available}")
    
        # If 2D mesh, project to 2D coordinates (remove thin dimension)
        if self.dim == 2 and points.shape[1] == 3:
            points = points[:, [self.axis1_idx, self.axis2_idx]]
    
        # Create meshio Mesh with only the cells we want
        mesh_data = meshio.Mesh(points=points, cells=cells)
    
        # Write to XDMF
        meshio.write(self.dolfin_cache_path, mesh_data)

    # =============================================================================
    # Private functions - Visualization
    # =============================================================================

    def _plot_2d(self, alpha=1, color=False):
        """Plot 2D mesh."""
        fig, ax = plt.subplots(figsize=(10, 8))
    
        coords = self.dolfin_mesh.coordinates()
        cells = self.dolfin_mesh.cells()
    
        # Always draw mesh edges
        ax.triplot(coords[:, 0], coords[:, 1], cells, 'k-', alpha=0.3, linewidth=0.5)

        # Color nodes by number of neighbors
        if color:
    
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=self.neighbor_counts,
                cmap='viridis',
                s=5,
                edgecolors=None,
                linewidths=0.5,
                zorder=5,
                alpha=alpha
            )
    
            plt.colorbar(scatter, ax=ax, label='Neighbor count')
    
        ax.set_aspect('equal')
        ax.set_xlabel(f'{self.axis1_name} ({self.units})')
        ax.set_ylabel(f'{self.axis2_name} ({self.units})')
    
        return ax

    def _plot_3d(self, alpha=1, color=False):
        """Plot 3D mesh."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    
        coords = self.dolfin_mesh.coordinates()
        cells = self.dolfin_mesh.cells()
    
        if color:
    
            # Scatter plot nodes colored by neighbor count
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=self.neighbor_counts,
                cmap='viridis',
                s=20,
                alpha=alpha,
                edgecolors='black',
                linewidths=0.3
            )
    
            plt.colorbar(scatter, ax=ax, label='Neighbor count', shrink=0.6)
    
            # Optionally draw edges (can be slow for large meshes)
            if len(cells) < 5000:  # Only for smaller meshes
                edges = set()
                for cell in cells:
                    for i in range(len(cell)):
                        for j in range(i + 1, len(cell)):
                            edge = tuple(sorted([cell[i], cell[j]]))
                            edges.add(edge)
    
                for n1, n2 in edges:
                    ax.plot3D(
                        [coords[n1, 0], coords[n2, 0]],
                        [coords[n1, 1], coords[n2, 1]],
                        [coords[n1, 2], coords[n2, 2]],
                        'k-', alpha=0.1, linewidth=0.3
                    )
        else:
            # Original: just scatter the nodes
            ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c='steelblue', s=10, alpha=alpha
            )
    
        ax.set_xlabel(f'{self.axis1_name} ({self.units})')
        ax.set_ylabel(f'{self.axis2_name} ({self.units})')
        ax.set_zlabel(f'{self.axis3_name} ({self.units})')
    
        return ax
    
    @staticmethod
    def _calculate_figsize(x, y, max_size=15):
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        aspect_ratio = x_range / y_range
    
        # Wide
        if aspect_ratio > 1.25:
            return (max_size, max_size / aspect_ratio * 5)
        # Tall
        else:
            return (max_size * aspect_ratio, max_size)