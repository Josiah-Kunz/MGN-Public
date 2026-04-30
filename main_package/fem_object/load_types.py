
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Union
import dolfin
import numpy as np

# Common functions

# For surface and edge laziness
BOUNDARY_LOCATION_MAP = {
    'left': (0, 'min'), 'right': (0, 'max'),
    'bottom': (1, 'min'), 'top': (1, 'max'),
    'back': (2, 'min'), 'front': (2, 'max'),
    'x_min': (0, 'min'), 'x_max': (0, 'max'),
    'y_min': (1, 'min'), 'y_max': (1, 'max'),
    'z_min': (2, 'min'), 'z_max': (2, 'max'),
}

def plot_arrows(ax, points, direction, mesh_object, arrow_scale, color, label):
    """
    Plot arrows at given points.
    
    Parameters:
    -----------
    ax : matplotlib axes
    points : ndarray
        Nx2 or Nx3 array of arrow start points
    direction : array-like
        Normalized direction vector (2D or 3D)
    mesh_object : MeshObject
    arrow_scale : float
        Arrow length as fraction of max extent
    color : str
    label : str
    
    Returns:
    --------
    float : arrow length in data units
    """
    arrow_length = mesh_object.min_extent * arrow_scale

    if mesh_object.dim == 2:
        X = points[:, 0]
        Y = points[:, 1]
        U = np.full_like(X, direction[0] * arrow_length)
        V = np.full_like(Y, direction[1] * arrow_length)

        ax.quiver(X, Y, U, V, color=color, alpha=0.7,
                  scale=1.0,
                  scale_units='xy',
                  angles='xy',
                  label=label)
    else:
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        U = np.full_like(X, direction[0] * arrow_length)
        V = np.full_like(Y, direction[1] * arrow_length)
        W = np.full_like(Z, direction[2] * arrow_length)

        ax.quiver(X, Y, Z, U, V, W, color=color, alpha=0.7,
                  length=arrow_length,
                  normalize=True,
                  label=label)

    return arrow_length

@dataclass
class VolumeLoad:
    """
    Volume load (body load) applied throughout a volume.
    
    Typical uses: gravity, centrifugal forces, thermal expansion
    Units: N/m³ (force per unit volume)
    
    Parameters:
    -----------
    load_per_volume : tuple
        Load vector per unit volume (fx, fy, fz) in N/m³
    name : str, optional
        Custom name for this load (e.g., "Gravity", "Inertia")
    
    Example:
    --------
    # Gravity on aluminum (ρ=2700 kg/m³, g=9.81 m/s²)
    gravity = VolumeLoad((0, 0, -2700 * 9.81), name="Gravity")
    """
    load_per_volume: Tuple[float, float, float]
    name: Optional[str] = None

    def to_fenics(self, mesh_object):
        """Convert to FEniCS dolfin.Constant."""
        dim = mesh_object.dim
        return dolfin.Constant(self.load_per_volume[:dim])

    def get_display_name(self):
        """Get display name for this load."""
        return self.name if self.name is not None else "Volume Load"

    def get_magnitude(self, mesh_object):
        extents = mesh_object.extents_sorted
        if mesh_object.dim == 3:
            volume = extents[0][1] * extents[1][1] * extents[2][1]
        else:
            volume = extents[0][1] * extents[1][1]
        return np.linalg.norm(self.load_per_volume) * volume

    # VolumeLoad
    def visualize(self, ax, mesh_object, arrow_spacing=None, num_arrows=8, arrow_scale=1):
        coords = mesh_object.dolfin_mesh.coordinates()
        extents = mesh_object.extents
    
        # Default spacing: ~8 arrows along the longest dimension
        if arrow_spacing is None:
            arrow_spacing = max(extents['x'], extents['y']) / num_arrows
    
        num_x = max(3, int(extents['x'] / arrow_spacing) + 1)
        num_y = max(3, int(extents['y'] / arrow_spacing) + 1)
    
        x_sample = np.linspace(coords[:, 0].min(), coords[:, 0].max(), num_x)
        y_sample = np.linspace(coords[:, 1].min(), coords[:, 1].max(), num_y)
        X, Y = np.meshgrid(x_sample, y_sample)
    
        points = np.column_stack([X.ravel(), Y.ravel()])

        # Note: load_magnitude is different from magnitude since magnitude includes volume
        load_magnitude = np.linalg.norm(self.load_per_volume)
        if load_magnitude == 0:
            return 0
        
        direction = np.array(self.load_per_volume) / load_magnitude
    
        return plot_arrows(ax, points, direction, mesh_object, arrow_scale,
                           color='blue', label=f'Volume load ({self.get_display_name()})')

    def convert_units(self, target_system, from_system):
        """Convert force per volume."""
        if callable(self.load_per_volume):
            original_func = self.load_per_volume
            from_force = from_system.get_force_unit()
            from_length = from_system.get_length_unit()
            to_force = target_system.get_force_unit()
            to_length = target_system.get_length_unit()
    
            # Force/volume = Force/Length³
            scale = ((1 * from_force / from_length**3).to(to_force / to_length**3)).magnitude
    
            def wrapped(x):
                result = original_func(x)
                if isinstance(result, (list, tuple)):
                    return tuple(r * scale for r in result)
                return result * scale
    
            self.load_per_volume = wrapped
        else:
            from_force = from_system.get_force_unit()
            from_length = from_system.get_length_unit()
            to_force = target_system.get_force_unit()
            to_length = target_system.get_length_unit()
    
            scale = ((1 * from_force / from_length**3).to(to_force / to_length**3)).magnitude
            self.load_per_volume = tuple(v * scale for v in self.load_per_volume)


@dataclass
class SurfaceLoad:
    """
    Surface load (pressure, traction) applied to a boundary surface.
    """
    location: object
    force: Optional[Tuple[float, float, float]] = None
    pressure: Optional[Tuple[float, float, float]] = None
    name: Optional[str] = None

    _area: float = field(default=None, init=False, repr=False)
    _cached_mesh_id: int = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if (self.force is None) == (self.pressure is None):
            raise ValueError("Must specify exactly one of 'force' or 'pressure'")

    def get_display_name(self):
        if self.name:
            return self.name
        if isinstance(self.location, str):
            return self.location
        return 'Custom Surface Load'

    def _get_boundary_function(self, mesh, mesh_object):
        if callable(self.location):
            location_func = self.location

            class CustomBoundary(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return location_func(x, on_boundary)

            return CustomBoundary()

        if isinstance(self.location, str):
            if self.location not in BOUNDARY_LOCATION_MAP:
                raise ValueError(
                    f"Unknown location '{self.location}'. "
                    f"Valid options: {list(BOUNDARY_LOCATION_MAP.keys())}"
                )

            axis_idx, min_max = BOUNDARY_LOCATION_MAP[self.location]
            coords = mesh.coordinates()

            if axis_idx >= coords.shape[1]:
                raise ValueError(f"Mesh is {coords.shape[1]}D, can't use '{self.location}'")
            
            value = coords[:, axis_idx].min() if min_max == 'min' else coords[:, axis_idx].max()

            # The tolerance is 1% of the smallest (nonzero) extent
            extents = [coords[:, i].max() - coords[:, i].min() for i in range(coords.shape[1])]
            min_extent = min(e for e in extents if e > 0)
            tol = min_extent * 0.01  

            class AxisBoundary(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and dolfin.near(x[axis_idx], value, tol)

            return AxisBoundary()

        raise TypeError(f"location must be str or callable, got {type(self.location)}")

    def get_boundary_marker(self, mesh, mesh_object):
        boundary_func = self._get_boundary_function(mesh, mesh_object)

        boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)

        marker_id = 1
        boundary_func.mark(boundaries, marker_id)

        return boundaries, marker_id

    def get_area(self, mesh_object):
        mesh = mesh_object.dolfin_mesh
        mesh_id = id(mesh)

        if self._area is not None and self._cached_mesh_id == mesh_id:
            return self._area

        boundaries, marker_id = self.get_boundary_marker(mesh, mesh_object)
        ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundaries)
        self._area = dolfin.assemble(dolfin.Constant(1.0) * ds(marker_id))
        self._cached_mesh_id = mesh_id

        return self._area

    def get_magnitude(self, mesh_object):
        if self.force is not None:
            return np.linalg.norm(self.force)
        return np.linalg.norm(self.pressure) * self.get_area(mesh_object)

    def calculate_pressure(self, mesh_object):
        dim = mesh_object.dim

        if self.pressure is not None:
            return dolfin.Constant(self.pressure[:dim]), None
    
        area = self.get_area(mesh_object)
    
        if area == 0:
            raise ValueError(f"Boundary '{self.get_display_name()}' has zero area")
    
        pressure_vector = tuple(f / area for f in self.force[:dim])
        return dolfin.Constant(pressure_vector), area

    def visualize(self, ax, mesh_object, arrow_scale=0.025, max_arrows=50):
        mesh = mesh_object.dolfin_mesh
        boundaries, marker_id = self.get_boundary_marker(mesh, mesh_object)
    
        # Get boundary facet midpoints
        mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
        boundary_points = []
    
        for facet in dolfin.facets(mesh):
            if boundaries[facet] == marker_id:
                boundary_points.append(facet.midpoint().array()[:mesh_object.dim])
    
        if not boundary_points:
            print(f"Warning: No boundary facets found for '{self.get_display_name()}'")
            return 0
    
        boundary_points = np.array(boundary_points)
    
        # Subsample if too many points
        if len(boundary_points) > max_arrows:
            indices = np.linspace(0, len(boundary_points) - 1, max_arrows, dtype=int)
            boundary_points = boundary_points[indices]
    
        # Get load direction
        load = self.pressure if self.pressure is not None else self.force
        load_magnitude = np.linalg.norm(load)
        if load_magnitude == 0:
            return 0
        
        direction = np.array(load) / load_magnitude
    
        return plot_arrows(ax, boundary_points, direction, mesh_object, arrow_scale,
                           color='red', label=f'Surface load ({self.get_display_name()})')

    def convert_units(self, target_system, from_system):
        """Convert force or pressure."""
        if self.force is not None:
            # Convert force
            from_force = from_system.get_force_unit()
            to_force = target_system.get_force_unit()
            scale = (1 * from_force).to(to_force).magnitude
    
            if callable(self.force):
                original_func = self.force
                self.force = lambda x: tuple(f * scale for f in original_func(x))
            else:
                self.force = tuple(f * scale for f in self.force)
    
        if self.pressure is not None:
            # Convert pressure (force/area)
            from_stress = from_system.get_stress_unit()
            to_stress = target_system.get_stress_unit()
            scale = (1 * from_stress).to(to_stress).magnitude
    
            if callable(self.pressure):
                original_func = self.pressure
                self.pressure = lambda x: tuple(p * scale for p in original_func(x))
            else:
                self.pressure = tuple(p * scale for p in self.pressure)

@dataclass
class EdgeLoad:
    """
    Edge load applied along an edge.
    
    Parameters:
    -----------
    location : str or callable
        Either a string like 'left', 'top', etc., or a function(x, on_boundary) that identifies the edge
    load_per_length : tuple or float
        Load vector per unit length (fx, fy, fz) in N/m or total force in N
    name : str, optional
        Custom name for this load (e.g., "Weld Line", "Contact Edge")
    """
    location: Union[str, Callable]
    load_per_length: object
    name: Optional[str] = None

    _edge_length: float = field(default=None, init=False, repr=False)
    _cached_mesh_id: int = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not callable(self.location) and not isinstance(self.location, str):
            raise TypeError("location must be a string or callable function")

        if isinstance(self.load_per_length, (int, float)):
            self._is_total_force = True
        elif isinstance(self.load_per_length, tuple) and len(self.load_per_length) == 3:
            self._is_total_force = False
        else:
            raise ValueError("load_per_length must be a scalar (total force) or 3-tuple (force per length)")

    def get_display_name(self):
        if self.name:
            return self.name
        if isinstance(self.location, str):
            return self.location
        return "Edge Load"

    def _get_edge_function(self, mesh, mesh_object):
        """Get the SubDomain function for this edge."""
        
        # Cannot use in 2D
        if mesh_object.dim < 3:
            raise ValueError(f"EdgeLoad can only be used in 3D. Try using an SurfaceLoad instead.")


        if callable(self.location):
            location_func = self.location

            class CustomEdge(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return location_func(x, on_boundary)

            return CustomEdge()

        if isinstance(self.location, str):
            if self.location not in BOUNDARY_LOCATION_MAP:
                raise ValueError(
                    f"Unknown location '{self.location}'. "
                    f"Valid options: {list(BOUNDARY_LOCATION_MAP.keys())}"
                )

            axis_idx, min_max = BOUNDARY_LOCATION_MAP[self.location]
            coords = mesh.coordinates()

            if axis_idx >= coords.shape[1]:
                raise ValueError(f"Mesh is {coords.shape[1]}D, can't use '{self.location}'")

            value = coords[:, axis_idx].min() if min_max == 'min' else coords[:, axis_idx].max()

            # The tolerance is 1% of the smallest (nonzero) extent
            extents = [coords[:, i].max() - coords[:, i].min() for i in range(coords.shape[1])]
            min_extent = min(e for e in extents if e > 0)
            tol = min_extent * 0.01

            class AxisEdge(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and dolfin.near(x[axis_idx], value, tol)

            return AxisEdge()

        raise TypeError(f"location must be str or callable, got {type(self.location)}")

    def get_edge_marker(self, mesh, mesh_object):
        dim = mesh.topology().dim()

        if dim not in [2, 3]:
            raise ValueError(f"Unsupported mesh dimension: {dim}")
    
        edges = dolfin.MeshFunction("size_t", mesh, 1)
        edges.set_all(0)
    
        edge_func = self._get_edge_function(mesh, mesh_object)
        marker_id = 1 
        edge_func.mark(edges, marker_id) 
    
        return edges, marker_id

    def get_edge_length(self, mesh_object):
        mesh = mesh_object.dolfin_mesh
        mesh_id = id(mesh)

        if self._edge_length is not None and self._cached_mesh_id == mesh_id:
            return self._edge_length

        edges, marker_id = self.get_edge_marker(mesh, mesh_object)
        ds_edge = dolfin.Measure("ds", domain=mesh, subdomain_data=edges)
        self._edge_length = dolfin.assemble(dolfin.Constant(1.0) * ds_edge(marker_id))
        self._cached_mesh_id = mesh_id

        return self._edge_length

    def get_magnitude(self, mesh_object):
        if self._is_total_force:
            return abs(self.load_per_length)
        return np.linalg.norm(self.load_per_length) * self.get_edge_length(mesh_object)

    def calculate_load(self, mesh_object):
        edge_length = self.get_edge_length(mesh_object)
        dim = mesh_object.dim
    
        if edge_length == 0:
            raise ValueError("Edge has zero length! Check your edge definition.")
    
        if self._is_total_force:
            raise NotImplementedError("Total force on edge needs direction specification")
    
        load_vector = self.load_per_length[:dim]
    
        return dolfin.Constant(load_vector), edge_length

    def visualize(self, ax, mesh_object, arrow_scale=0.025, max_arrows=50):
        mesh = mesh_object.dolfin_mesh
        edges, marker_id = self.get_edge_marker(mesh, mesh_object)
    
        # Get edge midpoints
        mesh.init(1)  # Initialize edges
        edge_points = []
    
        for edge in dolfin.edges(mesh):
            if edges[edge] == marker_id:
                edge_points.append(edge.midpoint().array()[:mesh_object.dim])
    
        if not edge_points:
            print(f"Warning: No edges found for '{self.get_display_name()}'")
            return 0
    
        edge_points = np.array(edge_points)
    
        # Subsample if too many points
        if len(edge_points) > max_arrows:
            indices = np.linspace(0, len(edge_points) - 1, max_arrows, dtype=int)
            edge_points = edge_points[indices]
    
        # Get load direction
        if self._is_total_force:
            print(f"Warning: Cannot visualize total force edge load without direction")
            return 0
    
        load_magnitude = np.linalg.norm(self.load_per_length)
        if load_magnitude == 0:
            return 0
    
        direction = np.array(self.load_per_length) / load_magnitude
    
        return plot_arrows(ax, edge_points, direction, mesh_object, arrow_scale,
                           color='orange', label=f'Edge load ({self.get_display_name()})')

    def convert_units(self, target_system, from_system):
        """Convert force per length."""
        from_force = from_system.get_force_unit()
        from_length = from_system.get_length_unit()
        to_force = target_system.get_force_unit()
        to_length = target_system.get_length_unit()
    
        scale = ((1 * from_force / from_length).to(to_force / to_length)).magnitude
    
        if callable(self.load_per_length):
            original_func = self.load_per_length
            self.load_per_length = lambda x: tuple(l * scale for l in original_func(x))
        else:
            self.load_per_length = tuple(l * scale for l in self.load_per_length)

@dataclass
class PointLoad:
    """
    Point load (concentrated load) applied at a specific location.
    
    Parameters:
    -----------
    location : tuple or callable
        Point coordinates (x, y, z) or function
    force : tuple
        Force vector (Fx, Fy, Fz) in N
    tolerance : float, optional
        Spatial tolerance for finding the point (default: 1e-6)
    name : str, optional
        Custom name for this load (e.g., "Bolt Load", "Pin Force")
    """
    location: object
    force: Tuple[float, float, float]
    tolerance: float = 1e-6
    name: Optional[str] = None

    def get_display_name(self):
        """Get display name for this load."""
        return self.name if self.name is not None else "Point Load"

    def __post_init__(self):
        """Validate inputs."""
        if not isinstance(self.force, tuple) or len(self.force) != 3:
            raise ValueError("force must be a 3-tuple (Fx, Fy, Fz)")

        if isinstance(self.location, tuple):
            if len(self.location) not in [2, 3]:
                raise ValueError("location tuple must be (x, y) or (x, y, z)")
        elif not callable(self.location):
            raise TypeError("location must be tuple or callable")

    def get_point_marker(self, mesh):
        """
        Find the mesh node closest to the specified location.
        
        Returns:
        --------
        int : node index
        """
        coords = mesh.coordinates()

        if isinstance(self.location, tuple):
            # Find dolfin.nearest node to coordinates
            target = np.array(self.location)
            distances = np.linalg.norm(coords - target, axis=1)
            node_idx = np.argmin(distances)

            if distances[node_idx] > self.tolerance:
                print(f"  WARNING: Nearest node is {distances[node_idx]:.6e} m from target location")

            print(f"  Point load at node {node_idx}: ({coords[node_idx, 0]:.6f}, {coords[node_idx, 1]:.6f}, {coords[node_idx, 2]:.6f})")

        elif callable(self.location):
            # Find nodes matching the callable
            matching_nodes = []
            for i, x in enumerate(coords):
                # Check if on boundary (simplified - may need refinement)
                if self.location(x, True):
                    matching_nodes.append(i)

            if len(matching_nodes) == 0:
                raise ValueError("No nodes match the specified location function")
            elif len(matching_nodes) > 1:
                print(f"  WARNING: Multiple nodes ({len(matching_nodes)}) match location. Using first.")

            node_idx = matching_nodes[0]
            print(f"  Point load at node {node_idx}: ({coords[node_idx, 0]:.6f}, {coords[node_idx, 1]:.6f}, {coords[node_idx, 2]:.6f})")

        return node_idx

    def to_fenics(self):
        """Convert to FEniCS point source."""
        # Point loads require special handling in FEniCS
        # Typically done via PointSource or by modifying the RHS vector
        raise NotImplementedError(
            "PointLoad requires special FEniCS handling. "
            "Consider using a small SurfaceLoad instead to avoid stress singularities."
        )

    def get_magnitude(self, mesh_object=None):
        return np.linalg.norm(self.force) 
    
    def visualize(self, ax, mesh_object, arrow_scale=0.025):
        mesh = mesh_object.dolfin_mesh
        coords = mesh.coordinates()
    
        # Get point location
        if isinstance(self.location, tuple):
            point = np.array(self.location[:mesh_object.dim])
        else:
            node_idx = self.get_point_marker(mesh)
            point = coords[node_idx][:mesh_object.dim]
    
        point = point.reshape(1, -1)  # Make 2D array for plot_arrows
    
        # Get force direction
        load_magnitude = np.linalg.norm(self.force)
        if load_magnitude == 0:
            return 0
    
        direction = np.array(self.force) / load_magnitude
    
        return plot_arrows(ax, point, direction, mesh_object, arrow_scale,
                           color='green', label=f'Point load ({self.get_display_name()})')

    def convert_units(self, target_system, from_system):
        """Convert force."""
        from_force = from_system.get_force_unit()
        to_force = target_system.get_force_unit()
        scale = (1 * from_force).to(to_force).magnitude
    
        self.force = tuple(f * scale for f in self.force)