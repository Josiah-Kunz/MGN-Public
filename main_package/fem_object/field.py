"""
Physical field with values and units.

Authors: Josiah Kunz, Claude
"""

from dataclasses import dataclass
import numpy as np
from ..units.unit_system import UNIT_REGISTRY

@dataclass
class Field:
    """
    Represents a physical field with values and units using Pint.
    
    Attributes:
    -----------
    quantity : pint.Quantity
        Physical quantity (values with units)
    
    Usage:
    ------
    from field import Field
    from unit_system import UNIT_REGISTRY, Units
    
    displacement = Field(u_values * UNIT_REGISTRY.mm)
    stress = Field(sigma_values * UNIT_REGISTRY.MPa)
    
    # Or using from_values
    displacement = Field.from_values(u_values, 'mm')
    stress = Field.from_values(sigma_values, Units.SI_MM.stress)
    """
    quantity: object  # pint.Quantity

    # =============================================================================
    # Properties
    # =============================================================================

    @property
    def values(self):
        """Get the magnitude (values without units)."""
        return self.quantity.magnitude

    @property
    def unit(self):
        """Get the Pint unit."""
        return self.quantity.units

    @property
    def unit_string(self):
        """Get compact unit string."""
        return f"{self.unit:~}"

    @property
    def shape(self):
        """Shape of the field values."""
        return self.values.shape

    @property
    def ndim(self):
        """Number of dimensions of the field values."""
        return self.values.ndim

    # =============================================================================
    # Public functions
    # =============================================================================

    @classmethod
    def from_values(cls, values, unit_string):
        """
        Constructor from numpy array and unit string.
        
        Usage:
        
            # Basic data
            forces = [100, 200, 300, 400, 500]
            force_field = Field.from_values(forces, 'kN')
            
            # FEniCS
            u_values = u.compute_vertex_values(mesh)
            u_field = Field.from_values(u_values, 'mm')
        
        Parameters:
        -----------
        values : array-like
            Field values
        unit_string : str
            Unit string (e.g., 'mm', 'MPa', 'kN')
            
        Returns:
        --------
        Field
            Field object with quantity
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        unit = UNIT_REGISTRY.parse_units(unit_string)
        quantity = values * unit
        return cls(quantity)

    def to(self, target_unit):
        """
        Convert to different unit.
        
        Parameters:
        -----------
        target_unit : str or pint.Unit
            Target unit to convert to
            
        Returns:
        --------
        Field
            New Field with converted units
        """
        converted = self.quantity.to(target_unit)
        return Field(converted)

    def to_compact(self):
        """
        Convert to most compact SI prefix automatically.
        
        Chooses prefix such that the maximum value is between 1 and 1000.
        
        Returns:
        --------
        Field
            New Field with compact units
        """
    
        # Find the magnitude range to determine appropriate prefix
        max_val = np.max(np.abs(self.values))
    
        if max_val == 0:
            return self  # No conversion needed for zero
    
        # Check if max_val is already in a good range [1, 1000)
        if 1 <= max_val < 1000:
            return self  # Already in good range, no conversion needed
    
        # Determine appropriate scale factor
        # We want max_val to be in range [1, 1000)
        log_val = np.log10(max_val)
    
        # Round to nearest multiple of 3 (SI prefixes are every 1000)
        exponent = int(np.floor(log_val / 3) * 3)
    
        # Clamp to reasonable range
        exponent = max(-12, min(12, exponent))  # pico to tera
    
        # Calculate the scale factor we need to apply
        scale_factor = 10 ** exponent
    
        # Map exponent to SI prefix
        prefix_map = {
            12: 'T',
            9: 'G',
            6: 'M',
            3: 'k',
            0: '',
            -3: 'm',
            -6: 'μ',
            -9: 'n',
            -12: 'p',
        }
    
        prefix = prefix_map.get(exponent, '')
    
        if not prefix:
            return self  # No prefix needed
    
        # Get current unit string
        current_unit_str = f"{self.unit:~}"
    
        # Try to apply prefix
        try:
            # Check if unit already has a prefix
            has_prefix = len(current_unit_str) > 1 and any(current_unit_str.startswith(p) for p in ['T', 'G', 'M', 'k', 'm', 'μ', 'n', 'p'])
    
            if has_prefix:
                # Already has a prefix, return as-is
                return self
    
            # Build target unit string
            target_unit_str = f"{prefix}{current_unit_str}"
            target_unit = UNIT_REGISTRY.parse_units(target_unit_str)
    
            # Manually scale the values
            scaled_values = self.values / scale_factor
            
            # Create new quantity with scaled values and new unit
            new_quantity = scaled_values * target_unit
            return Field(new_quantity)
    
        except Exception as e:
            print(f"Warning: Could not compact units: {e}")
            # If anything fails, return original
            return self

    def extract_scalar(self, component=None):
        """
        Extract scalar values from a Field that may be scalar, vector, or tensor.
        
        Parameters:
        -----------
        component : int or tuple, optional
            For vector fields: int (e.g., 0 for x-component)
            For tensor fields: tuple (e.g., (0, 0) for σ_xx)
            If None:
                - Scalar: returns values as-is
                - Vector: computes magnitude
                - Tensor: raises error
        
        Returns:
        --------
        Field
            New Field object with scalar (1D) values
            
        Raises:
        -------
        ValueError
            If field dimensionality is unsupported or component is incorrectly specified
        
        Examples:
        ---------
        # Scalar field - returns as-is
        vm_scalar = von_mises_field.extract_scalar()
        
        # Vector field - compute magnitude
        u_mag = displacement_field.extract_scalar()
        
        # Vector field - specific component
        u_x = displacement_field.extract_scalar(component=0)
        
        # Tensor field - specific component
        sigma_xx = stress_field.extract_scalar(component=(0, 0))
        """
    
        if self.values.ndim == 1:
            # Scalar field (n_vertices,) - return as-is
            return self
    
        elif self.values.ndim == 2:
            # Vector field (n_vertices, n_dims)
            if component is None:
                # Default: compute magnitude
                magnitude = np.linalg.norm(self.values, axis=1)
                return Field.from_values(magnitude, self.unit_string)
            else:
                if not isinstance(component, int):
                    raise ValueError(
                        f"For vector field, component must be an int (0, 1, or 2), got {component}"
                    )
                return Field.from_values(self.values[:, component], self.unit_string)
    
        elif self.values.ndim == 3:
            # Tensor field (n_vertices, n_dims, n_dims)
            if component is None:
                raise ValueError(
                    f"Field is a tensor with shape {self.values.shape}. "
                    "Specify which component to extract using component=(i, j), "
                    "e.g., component=(0, 0) for σ_xx."
                )
            if not isinstance(component, tuple) or len(component) != 2:
                raise ValueError("For tensor field, component must be a tuple of (i, j), e.g., (0, 0)")
    
            i, j = component
            return Field.from_values(self.values[:, i, j], self.unit_string)
    
        else:
            raise ValueError(
                f"Unsupported field shape: {self.values.shape}. "
                f"Expected 1D (scalar), 2D (vector), or 3D (tensor)."
            )

    # =============================================================================
    # Less interesting functions
    # =============================================================================
    def __repr__(self):
        max_val = np.max(self.values)
        return f"Field(shape={self.shape}, unit={self.unit_string}, max={max_val:.3e})"

    def __str__(self):
        return f"{self.quantity}"

    def __len__(self):
        """Number of nodes/elements."""
        return len(self.values)

    def __getitem__(self, key):
        """Allow indexing into the values."""
        return self.quantity[key]

    def max(self):
        """Maximum value in the field (returns Quantity)."""
        return np.max(self.quantity)

    def min(self):
        """Minimum value in the field (returns Quantity)."""
        return np.min(self.quantity)

    def mean(self):
        """Mean value of the field (returns Quantity)."""
        return np.mean(self.quantity)