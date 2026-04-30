from dataclasses import dataclass
from typing import Optional, Tuple
import dolfin
import numpy as np


@dataclass
class FixedBoundary:
    """
    Fixed (Dirichlet) boundary condition - prescribed displacement.
    
    Parameters:
    -----------
    location : str or callable
        Boundary location ('left', 'right', 'top', 'bottom', 'front', 'back',
        'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max') or custom function
    value : tuple or callable
        Displacement values (ux, uy, uz) or function(x) returning displacement
    components : tuple of int, optional
        Which components to constrain (0=x, 1=y, 2=z). Default: all components.
    name : str, optional
    tolerance : float, optional
        Relative tolerance for boundary detection (default: 0.001)
    
    Examples:
    ---------
    # Fixed support
    FixedBoundary('left', value=(0, 0, 0), name="Fixed")
    
    # Roller (y free)
    FixedBoundary('left', value=(0, 0), components=(0, 2), name="Roller")
    
    # Custom location
    FixedBoundary(
        location=lambda x, on_boundary: on_boundary and x[0] < 100,
        value=(0, 0, 0),
        name="Partial fix"
    )
    """
    location: object
    value: object = (0, 0, 0)
    components: Optional[Tuple[int, ...]] = None
    name: Optional[str] = None
    tolerance: float = 0.001

    LOCATION_MAP = {
        'left': (0, 'min'), 'right': (0, 'max'),
        'bottom': (1, 'min'), 'top': (1, 'max'),
        'back': (2, 'min'), 'front': (2, 'max'),
        'x_min': (0, 'min'), 'x_max': (0, 'max'),
        'y_min': (1, 'min'), 'y_max': (1, 'max'),
        'z_min': (2, 'min'), 'z_max': (2, 'max'),
    }

    def __post_init__(self):
        # Validate location
        if not callable(self.location) and not isinstance(self.location, str):
            raise TypeError("location must be str or callable")

        if isinstance(self.location, str) and self.location not in self.LOCATION_MAP:
            raise ValueError(
                f"Unknown location '{self.location}'. "
                f"Valid options: {list(self.LOCATION_MAP.keys())}"
            )

    def get_display_name(self):
        if self.name:
            return self.name
        if isinstance(self.location, str):
            return f"BC ({self.location})"
        return "Custom BC"

    def _get_boundary_function(self, mesh):
        """Convert location to a dolfin SubDomain."""
        if callable(self.location):
            location_func = self.location

            class CustomBoundary(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return location_func(x, on_boundary)

            return CustomBoundary()

        if isinstance(self.location, str):
            axis_idx, min_max = self.LOCATION_MAP[self.location]
            coords = mesh.coordinates()

            if axis_idx >= coords.shape[1]:
                raise ValueError(f"Mesh is {coords.shape[1]}D, can't use '{self.location}'")

            value = coords[:, axis_idx].min() if min_max == 'min' else coords[:, axis_idx].max()

            # Use smallest extent for tolerance
            extents = [coords[:, i].max() - coords[:, i].min() for i in range(coords.shape[1])]
            min_extent = min(e for e in extents if e > 0)
            tol = min_extent * self.tolerance

            class AxisBoundary(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and dolfin.near(x[axis_idx], value, tol)

            return AxisBoundary()

    def to_fenics(self, V, mesh_object):
        """
        Convert to FEniCS DirichletBC.
        
        Parameters:
        -----------
        V : dolfin.FunctionSpace
            The function space (usually VectorFunctionSpace for elasticity)
        mesh_object : MeshObject
            The mesh object (need for dimension)
            
        Returns:
        --------
        list of dolfin.DirichletBC
        """
        mesh = mesh_object.dolfin_mesh
        dim = mesh_object.dim
        boundary = self._get_boundary_function(mesh)
    
        # Trim value to mesh dimension
        if callable(self.value):
            bc_value = self.value
        else:
            bc_value = dolfin.Constant(self.value[:dim])
    
        # If no components specified, apply to all
        if self.components is None:
            return [dolfin.DirichletBC(V, bc_value, boundary)]
    
        # Apply to specific components (filter out components >= dim)
        bcs = []
        for comp in self.components:
            if comp >= dim:
                continue  # Skip z-component for 2D mesh
    
            if callable(self.value):
                comp_value = lambda x, c=comp: self.value(x)[c]
            else:
                comp_value = dolfin.Constant(self.value[comp] if len(self.value) > comp else 0)
    
            bcs.append(dolfin.DirichletBC(V.sub(comp), comp_value, boundary))
    
        return bcs

    def draw_on_ax(self, ax, mesh_object, color='purple', alpha=0.5):
        """
        Visualize the boundary condition on a plot.
        
        Parameters:
        -----------
        ax : matplotlib axes
        mesh_object : MeshObject
        color : str
        alpha : float
        """
        mesh = mesh_object.dolfin_mesh
        boundary = self._get_boundary_function(mesh)

        # Mark boundary facets
        boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        boundary.mark(boundaries, 1)

        # Get boundary facet midpoints
        mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
        boundary_points = []

        for facet in dolfin.facets(mesh):
            if boundaries[facet] == 1:
                boundary_points.append(facet.midpoint().array()[:mesh_object.dim])

        if not boundary_points:
            print(f"Warning: No boundary facets found for '{self.get_display_name()}'")
            return

        boundary_points = np.array(boundary_points)

        # Plot boundary points/line
        if mesh_object.dim == 2:
            ax.scatter(boundary_points[:, 0], boundary_points[:, 1],
                       c=color, s=20, alpha=alpha, marker='s',
                       label=self.get_display_name())
        else:
            ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                       c=color, s=20, alpha=alpha, marker='s',
                       label=self.get_display_name())

    def convert_units(self, target_system, from_system):
        """Convert displacement values to target unit system."""

        if callable(self.value):
            # Wrap the function with unit conversion
            original_func = self.value
            from_length = from_system.get_length_unit()
            to_length = target_system.get_length_unit()
            scale_in = (1 * to_length).to(from_length).magnitude
            scale_out = (1 * from_length).to(to_length).magnitude
    
            def wrapped(x):
                # Convert coordinates to original units
                x_original = tuple(xi * scale_in for xi in x)
                # Call original function
                result = original_func(x_original)
                # Convert result to target units
                if isinstance(result, (list, tuple)):
                    return tuple(ri * scale_out for ri in result)
                else:
                    return result * scale_out
    
            self.value = wrapped
            return
    
        # Non-callable: direct conversion
        from_length = from_system.get_length_unit()
        to_length = target_system.get_length_unit()
        scale = (1 * from_length).to(to_length).magnitude
    
        self.value = tuple(v * scale for v in self.value)