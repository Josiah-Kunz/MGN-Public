"""
Unit system handling for FEM simulations.

Authors: Josiah Kunz, Claude
"""

from dataclasses import dataclass
from pint import UnitRegistry

# Global unit registry
UNIT_REGISTRY = UnitRegistry()
UNIT_REGISTRY.setup_matplotlib(True)

@dataclass
class UnitSystem:
    """
    Define consistent unit system for FEM simulation.
    
    Attributes:
    -----------
    length : str
        Length unit (e.g., 'mm', 'm', 'in', 'ft')
    mass : str
        Mass unit (e.g., 'kg', 'g', 'lb')
    time : str
        Time unit (e.g., 's', 'ms')
    force : str (computed)
        Force unit (e.g., 'N', 'lbf')
    stress : str (computed)
        Stress/pressure unit (e.g., 'Pa', 'MPa', 'psi')
    acceleration : str (computed)
        Acceleration unit (e.g., 'm/s²', 'mm/s²')
    density : str (computed)
        Density unit (e.g., 'kg/m³', 'kg/mm³')
    """
    length: str
    mass: str
    time: str

    def __post_init__(self):
        """Compute derived units and validate with Pint."""
        self.force = self._compute_force_unit()
        self.stress = self._compute_stress_unit()
        self.acceleration = self._compute_acceleration_unit()
        self.density = self._compute_density_unit()

        self._validate_units()

    def _validate_units(self):
        """Validate that all units can be parsed by Pint."""
        try:
            UNIT_REGISTRY.parse_units(self.length)
            UNIT_REGISTRY.parse_units(self.mass)
            UNIT_REGISTRY.parse_units(self.time)
            UNIT_REGISTRY.parse_units(self.force)
            UNIT_REGISTRY.parse_units(self.stress)
            UNIT_REGISTRY.parse_units(self.acceleration)
            UNIT_REGISTRY.parse_units(self.density)
        except Exception as e:
            raise ValueError(f"Invalid unit in UnitSystem: {e}")

    def _compute_force_unit(self):
        """Derive force unit from mass, length, time."""
        if self.mass == 'kg' and self.length == 'm' and self.time == 's':
            return 'N'
        elif self.mass == 'kg' and self.length == 'mm' and self.time == 's':
            return 'N'
        elif self.mass == 'g' and self.length == 'cm' and self.time == 's':
            return 'dyn'
        elif self.mass == 'lb' and self.length == 'ft' and self.time == 's':
            return 'lbf'
        elif self.mass == 'lb' and self.length == 'in' and self.time == 's':
            return 'lbf'
        else:
            try:
                m = UNIT_REGISTRY.parse_units(self.mass)
                l = UNIT_REGISTRY.parse_units(self.length)
                t = UNIT_REGISTRY.parse_units(self.time)
                force = (m * l / t**2).to_compact()
                return f"{force.units:~}"
            except:
                return f"{self.mass}·{self.length}/{self.time}²"

    def _compute_stress_unit(self):
        """Derive stress unit from force and length."""
        if self.force == 'N' and self.length == 'm':
            return 'Pa'
        elif self.force == 'N' and self.length == 'mm':
            return 'MPa'
        elif self.force == 'lbf' and self.length == 'in':
            return 'psi'
        elif self.force == 'lbf' and self.length == 'ft':
            return 'pound_force/foot**2'
        else:
            try:
                f = UNIT_REGISTRY.parse_units(self.force)
                l = UNIT_REGISTRY.parse_units(self.length)
                stress = (f / l**2).to_compact()
                return f"{stress.units:~}"
            except:
                return f"{self.force}/{self.length}²"

    def _compute_acceleration_unit(self):
        """Derive acceleration unit from length and time."""
        try:
            l = UNIT_REGISTRY.parse_units(self.length)
            t = UNIT_REGISTRY.parse_units(self.time)
            accel = (l / t**2)
            return f"{accel:~}"
        except:
            return f"{self.length}/{self.time}²"

    def _compute_density_unit(self):
        """Derive density unit from mass and length."""
        try:
            m = UNIT_REGISTRY.parse_units(self.mass)
            l = UNIT_REGISTRY.parse_units(self.length)
            dens = (m / l**3)
            return f"{dens:~}"
        except:
            return f"{self.mass}/{self.length}³"

    def get_length_unit(self):
        """Get Pint unit for length."""
        return UNIT_REGISTRY.parse_units(self.length)

    def get_mass_unit(self):
        """Get Pint unit for mass."""
        return UNIT_REGISTRY.parse_units(self.mass)

    def get_time_unit(self):
        """Get Pint unit for time."""
        return UNIT_REGISTRY.parse_units(self.time)

    def get_force_unit(self):
        """Get Pint unit for force."""
        return UNIT_REGISTRY.parse_units(self.force)

    def get_stress_unit(self):
        """Get Pint unit for stress."""
        return UNIT_REGISTRY.parse_units(self.stress)

    def get_acceleration_unit(self):
        """Get Pint unit for acceleration."""
        return UNIT_REGISTRY.parse_units(self.acceleration)

    def get_density_unit(self):
        """Get Pint unit for density."""
        return UNIT_REGISTRY.parse_units(self.density)

    def __str__(self):
        return (f"UnitSystem({self.length}, {self.mass}, {self.time}) → "
                f"Force: {self.force}, Stress: {self.stress}, "
                f"Acceleration: {self.acceleration}, Density: {self.density}")


class Units:
    """
    Namespace for common unit system presets. If you'd rather create your own, use UnitSystem(length, mass, time) 
    
    Usage:
    ------
    from fem_utils import Units
    
    result = StructuralResult(
        displacement=Field.from_values(u, Units.SI_MM.length),
        stress=Field.from_values(sigma, Units.SI_MM.stress),
        von_mises=Field.from_values(vm, Units.SI_MM.stress)
    )
    
    Available presets:
    ------------------
    SI : SI base units (m, kg, s) → Pa
    SI_M : alias for SI
    SI_MM : Engineering SI (mm, kg, s) → MPa
    CGS : Centimeter-gram-second (cm, g, s) → dyn/cm²
    US : US customary with feet (ft, lb, s) → psf
    IMPERIAL : alias for US
    US_IN : Engineering imperial with inches (in, lb, s) → psi
    IMPERIAL_IN : alias for US_IN
    """

    SI = UnitSystem('m', 'kg', 's')            # SI base: Pa
    SI_M = UnitSystem('m', 'kg', 's')          # Alias for SI
    SI_MM = UnitSystem('mm', 'kg', 's')        # Engineering SI: MPa
    CGS = UnitSystem('cm', 'g', 's')           # CGS: dyn/cm²
    US = UnitSystem('ft', 'lb', 's')           # Imperial: lbf/ft²
    IMPERIAL = UnitSystem('ft', 'lb', 's')     # Alias for US
    US_IN = UnitSystem('in', 'lb', 's')        # Engineering Imperial: psi
    IMPERIAL_IN = UnitSystem('in', 'lb', 's')  # Alias for US_IN