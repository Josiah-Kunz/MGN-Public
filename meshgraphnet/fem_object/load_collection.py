"""
Load definitions for FEM simulations.

Authors: Josiah Kunz, Claude
"""

import numpy as np
import matplotlib.pyplot as plt

from .load_types import *
from ..units.unit_system import Units


class LoadCollection:
    """
    Collection of loads to apply in FEM simulation.
    
    Usage:
    ------
    loads = LoadCollection(Units.US)
    loads.add(VolumeLoad((0, 0, -26500)))  # Gravity
    loads.add(SurfaceLoad('right', force=(50000, 0, 0)))  # Applied force
    
    result = fem_structural(mesh, V, bc, E, nu, loads=loads)
    """
    def __init__(self, units=None):
        self.volume_loads = []
        self.surface_loads = []
        self.edge_loads = []
        self.point_loads = []
        self.loads = [] # All loads
        self.units = units
        if units is None: self.units = Units.SI

    # Make this thang iterable
    def __iter__(self):
        return iter(self.loads)

    def add(self, load):
        """Add a load to the collection."""
        if isinstance(load, VolumeLoad):
            self.volume_loads.append(load)
        elif isinstance(load, SurfaceLoad):
            self.surface_loads.append(load)
        elif isinstance(load, EdgeLoad):
            self.edge_loads.append(load)
        elif isinstance(load, PointLoad):
            self.point_loads.append(load)
        else:
            raise TypeError(f"Unknown load type: {type(load)}")
        
        self.loads.append(load)

    def has_surface_loads(self):
        """Check if any surface loads are defined."""
        return len(self.surface_loads) > 0

    def has_edge_loads(self):
        """Check if any edge loads are defined."""
        return len(self.edge_loads) > 0

    def has_point_loads(self):
        """Check if any point loads are defined."""
        return len(self.point_loads) > 0

    def get_volume_load_sum(self):
        """Sum all volume loads into a single FEniCS dolfin.Constant."""
        if not self.volume_loads:
            return None

        total = np.array([0.0, 0.0, 0.0])
        for vl in self.volume_loads:
            total += np.array(vl.load_per_volume)

        return dolfin.Constant(tuple(total))

    def summary(self):
        """Print summary of all loads."""
        print(f"\n{'='*60}")
        print("Load Collection Summary")
        print(f"{'='*60}")
    
        # Get unit strings (they're already strings!)
        force_unit = self.units.force
        length_unit = self.units.length
        stress_unit = self.units.stress
        volume_load_unit = f"{force_unit}/{length_unit}³"
    
        print(f"Volume loads: {len(self.volume_loads)}")
        for idx, vol_load in enumerate(self.volume_loads):
            print(f"  {idx+1}. {vol_load.get_display_name()}: {vol_load.load_per_volume} {volume_load_unit}")
    
        print(f"\nSurface loads: {len(self.surface_loads)}")
        for idx, surf_load in enumerate(self.surface_loads):
            name = surf_load.get_display_name()
            if surf_load.force is not None:
                print(f"  {idx+1}. {name}: Force = {surf_load.force} {force_unit}")
            else:
                print(f"  {idx+1}. {name}: Pressure = {surf_load.pressure} {stress_unit}")
    
        print(f"\nEdge loads: {len(self.edge_loads)}")
        print(f"Point loads: {len(self.point_loads)}")
        print(f"{'='*60}\n")

    def max_magnitude(self, mesh_object):
        magnitude = 0
        for load in self.loads:
            magnitude = max(magnitude, load.get_magnitude(mesh_object))
        return magnitude

    def visualize(self, mesh_object, mesh_alpha=0.5, arrow_scale=0.5, legend=True, title="__auto__"):
        """
        Visualize all loads on a single mesh plot.
        
        Shows all loads simultaneously with color coding:
        - Volume loads: arrows throughout mesh (blue)
        - Surface loads: highlighted boundaries (red)
        - Edge loads: highlighted edges (orange)
        - Point loads: marked points (green)
        """

        fig = mesh_object.visualize(False, alpha=mesh_alpha)
        ax = fig.axes[0]
        max_magnitude = self.max_magnitude(mesh_object)
        max_arrow_length = 0
    
        # Plot all loads
        for load in self.loads:
            scale = arrow_scale * load.get_magnitude(mesh_object) / max_magnitude
            max_arrow_length = max(max_arrow_length, load.visualize(ax, mesh_object, arrow_scale=scale))
    
        # Make sure the arrows don't get cut off
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - max_arrow_length, xlim[1] + max_arrow_length)
        ax.set_ylim(ylim[0] - max_arrow_length, ylim[1] + max_arrow_length)
    
        if mesh_object.dim == 3:
            zlim = ax.get_zlim()
            ax.set_zlim(zlim[0] - max_arrow_length, zlim[1] + max_arrow_length)
            
        # Title could be auto-generated or custom or none
        if title == "__auto__":
            ax.set_title(f"{mesh_object.name} Load Visualization")
        elif title is not None:
            ax.set_title(title)
    
        if legend: ax.legend()
        plt.tight_layout()
        plt.show()

    def convert_units(self, target_system):
        """Convert all loads to target unit system."""
        for load in self.loads:
            load.convert_units(target_system, self.units)
    
        # Update collection's unit system
        self.units = target_system