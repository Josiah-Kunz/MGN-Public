from ..units import *

from dataclasses import dataclass
import numpy as np

@dataclass
class Material:
    """
    Material properties for linear elasticity.
    
    Parameters:
    -----------
    E : float
        Young's modulus (in units system's pressure unit)
    nu : float
        Poisson's ratio (dimensionless)
    rho : float, optional
        Density (in units system's density unit)
    units : UnitSystem
        Unit system for the material properties
    name : str, optional
    """
    E: float
    nu: float
    units: object
    rho: float = None
    name: str = None

    def get_E(self, target_units):
        """Get Young's modulus converted to target unit system."""
        E_quantity = self.E * self.units.get_stress_unit()
        return E_quantity.to(target_units.get_stress_unit()).magnitude

    def get_rho(self, target_units):
        """Get density converted to target unit system."""
        if self.rho is None:
            return None
        rho_quantity = self.rho * self.units.get_density_unit()
        return rho_quantity.to(target_units.get_density_unit()).magnitude

    def get_gravity_load(self, target_units, g=9.81, direction=(0, -1, 0)):
        """
        Get volume load for gravity in target units.
        
        Parameters:
        -----------
        target_units : UnitSystem
            Target unit system for the load
        g : float
            Gravitational acceleration in m/s² (default: 9.81)
        direction : tuple
            Direction vector for gravity (default: -Y)
        
        Returns:
        --------
        tuple : load_per_volume in target units (force/length³)
        """
        if self.rho is None:
            raise ValueError("Density (rho) required for gravity load")

        # Get rho in target units
        rho = self.get_rho(target_units)

        # Convert g from m/s² to target acceleration units
        g_quantity = g * UNIT_REGISTRY.parse_units('m/s²')
        g_target = g_quantity.to(target_units.get_acceleration_unit()).magnitude

        # load = rho * g (force per volume)
        load_magnitude = rho * g_target

        # Normalize direction
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        return tuple(load_magnitude * d for d in direction)

    @classmethod
    def steel(cls, units=None):
        """Steel: E=200 GPa, nu=0.3, rho=7850 kg/m³"""
        if units is None:
            units = Units.SI
        return cls(E=200e9, nu=0.3, rho=7850, units=Units.SI, name="Steel")

    @classmethod
    def aluminum(cls, units=None):
        """Aluminum: E=70 GPa, nu=0.33, rho=2700 kg/m³"""
        if units is None:
            units = Units.SI
        return cls(E=70e9, nu=0.33, rho=2700, units=Units.SI, name="Aluminum")

    @classmethod
    def titanium(cls, units=None):
        """Titanium: E=110 GPa, nu=0.34, rho=4500 kg/m³"""
        if units is None:
            units = Units.SI
        return cls(E=110e9, nu=0.34, rho=4500, units=Units.SI, name="Titanium")

    @classmethod
    def copper(cls, units=None):
        """Copper: E=120 GPa, nu=0.34, rho=8900 kg/m³"""
        if units is None:
            units = Units.SI
        return cls(E=120e9, nu=0.34, rho=8900, units=Units.SI, name="Copper")