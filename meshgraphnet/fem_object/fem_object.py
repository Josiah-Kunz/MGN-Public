from ..units.unit_system import UNIT_REGISTRY, Units

import torch # for edge indices
from dataclasses import dataclass, field
from typing import List, Optional
import dolfin
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


@dataclass
class FEMObject:
    """
    Complete FEM problem combining mesh, loads, and boundary conditions.
    
    Parameters:
    -----------
    mesh : MeshObject
        The mesh object
    loads : LoadCollection, optional
        Collection of loads (volume, surface, edge, point)
    boundaries : list of FixedBoundary, optional
        List of boundary conditions
    material : Material, optional
        Material properties (E, nu, rho)
    name : str, optional
        Name for this problem
    metadata : dict, optional
        Additional columns to save (e.g., {'load (lbf/in)': 5000, 'temperature (K)': 300})
    
    Examples:
    ---------
    fem = FEMObject(
        mesh=mesh,
        loads=loads,
        boundaries=[FixedBoundary('left', value=(0, 0, 0))],
        material=Material(E=200e9, nu=0.3),
        name="Cantilever beam"
    )
    
    fem.visualize()  # Show mesh, loads, BCs
    fem.solve()      # Run FEniCS solver
    fem.plot_displacement()
    fem.plot_stress()
    """
    mesh: object
    loads: object = None
    boundaries: List = field(default_factory=list)
    material: object = None
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Computed variables (not passed on initialization)
    edge_index: torch.Tensor = field(default=None, init=False, repr=False)

    # Solution fields (populated after solve)
    _von_mises: np.ndarray = field(default=None, init=False, repr=False)
    _displacement: object = field(default=None, init=False, repr=False)
    _stress: object = field(default=None, init=False, repr=False)
    _strain: object = field(default=None, init=False, repr=False)
    _V: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        
        # Auto-name
        if self.name is None:
            self.name = self.mesh.name
        
        # Initialize edge connectivity if not already done
        mesh = self.mesh.dolfin_mesh
        mesh.init(1)

        # Dolfin already tracks the edges without duplicates, but we also need the vertices
        edges = []
        for edge in dolfin.edges(mesh):
            vertices = edge.entities(0)
            edges.append([vertices[0], vertices[1]])

        # Store as tensor (undirected)
        edges = np.array(edges)
        edge_index = torch.from_numpy(edges.T.astype(np.int64))
        self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    @property
    def von_mises(self) -> np.ndarray:
        """Get von Mises stress at vertices (after solve)."""
        if self._displacement is None:
            raise ValueError("Must call solve() first")
        if self._von_mises is None:
            self._von_mises = self._get_stress_at_vertices('von_mises') 
        return self._von_mises
    
    @property
    def coordinates(self) -> np.ndarray:
        """Get node coordinates."""
        return self.mesh.dolfin_mesh.coordinates()
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in mesh."""
        return self.mesh.dolfin_mesh.num_vertices()

    def visualize_setup(self, show=True, mesh_alpha=0.3, arrow_scale=0.5):
        """
        Visualize the complete FEM setup: mesh, loads, and boundaries.
        """
        fig = self.mesh.visualize(show=False, alpha=mesh_alpha)
        ax = fig.axes[0]

        cmap = plt.cm.tab10
        color_idx = 0

        # Plot loads
        if self.loads is not None:
            max_magnitude = self.loads.max_magnitude(self.mesh)
            if max_magnitude > 0:
                for load in self.loads.loads:
                    mag = load.get_magnitude(self.mesh)
                    scale = arrow_scale * mag / max_magnitude if max_magnitude > 0 else arrow_scale
                    load.visualize(ax, self.mesh, arrow_scale=scale)

        # Plot boundaries
        for boundary in self.boundaries:
            boundary.draw_on_ax(ax, self.mesh, color=cmap(color_idx))
            color_idx += 1

        ax.legend()
        ax.set_title(f"{self.name} - FEM Setup")
        plt.tight_layout()

        if show:
            plt.show()
        else:
            return fig

    def solve(self, element_degree=1):
        if self.material is None:
            raise ValueError("Material must be defined before solving")

        # Convert everything to SI temporarily, then convert it back later
        # This lets the user have 1500 lbf on a 10 meter long cantilever
        og_mesh_units, og_load_units = self._convert(Units.SI, Units.SI)

        dolfin_mesh = self.mesh.dolfin_mesh
        dim = self.mesh.dim
    
        # Get material properties in mesh's unit system
        E = self.material.get_E(self.mesh.unit_system)
        nu = self.material.nu  # Dimensionless

        # Create function space
        self._V = dolfin.VectorFunctionSpace(dolfin_mesh, "Lagrange", element_degree)

        # Trial and test functions
        u = dolfin.TrialFunction(self._V)
        v = dolfin.TestFunction(self._V)

        # Lame parameters
        mu = E / (2 * (1 + nu))
        if dim == 2:
            # Plane stress
            lmbda = E * nu / (1 - nu**2)
        else:
            # 3D
            lmbda = E * nu / ((1 + nu) * (1 - 2*nu))

        # Strain and stress
        def epsilon(u):
            return 0.5 * (dolfin.grad(u) + dolfin.grad(u).T)

        def sigma(u):
            return lmbda * dolfin.div(u) * dolfin.Identity(dim) + 2 * mu * epsilon(u)

        # Bilinear form
        a = dolfin.inner(sigma(u), epsilon(v)) * dolfin.dx

        # Linear form (loads)
        L = dolfin.dot(dolfin.Constant([0] * dim), v) * dolfin.dx  # Start with zero

        if self.loads is not None:
            # Volume loads
            for vol_load in self.loads.volume_loads:
                f = vol_load.to_fenics(self.mesh)
                L += dolfin.dot(f, v) * dolfin.dx

            # Surface loads
            for surf_load in self.loads.surface_loads:
                pressure, _ = surf_load.calculate_pressure(self.mesh)
                boundaries, marker_id = surf_load.get_boundary_marker(dolfin_mesh, self.mesh)
                ds = dolfin.Measure("ds", domain=dolfin_mesh, subdomain_data=boundaries)
                L += dolfin.dot(pressure, v) * ds(marker_id)

            # Edge loads
            for edge_load in self.loads.edge_loads:
                load_vector, _ = edge_load.calculate_load(self.mesh)
                edges, marker_id = edge_load.get_edge_marker(dolfin_mesh, self.mesh)
                ds = dolfin.Measure("ds", domain=dolfin_mesh, subdomain_data=edges)
                L += dolfin.dot(load_vector, v) * ds(marker_id)
            
            # Point loads not supported =/

        # Boundary conditions
        bcs = []
        for boundary in self.boundaries:
            bcs.extend(boundary.to_fenics(self._V, self.mesh)) 

        if not bcs:
            raise ValueError("No boundary conditions defined - problem is underconstrained")
        
        # Solve
        self._displacement = dolfin.Function(self._V)
        dolfin.solve(a == L, self._displacement, bcs)
        
        # Convert back
        self._convert(og_mesh_units, og_load_units)

        # All done!
        return self._displacement

    def _convert(self, new_mesh_units, new_load_units):
        
        # Cache
        og_mesh_units = self.mesh.unit_system   # Also doubles as boundary units
        og_load_units = None
        if self.loads is not None:
            og_load_units = self.loads.units

        # Convert mesh
        self.mesh.convert_units(new_mesh_units)

        # Convert loads
        if self.loads is not None:
            self.loads.convert_units(new_load_units)

        # Convert boundaries (if they have displacement values)
        for boundary in self.boundaries:
            if hasattr(boundary, 'convert_units'):
                boundary.convert_units(new_mesh_units, og_mesh_units)

        # Convert displacement (self-tracking)
        if self._displacement is not None:
            from_length = og_mesh_units.get_length_unit()
            to_length = new_mesh_units.get_length_unit()
            scale = (1 * from_length).to(to_length).magnitude
            self._displacement.vector()[:] *= scale

        # Kick the cache back
        return og_mesh_units, og_load_units

    def plot_displacement(self, component=None, show=True, colormap='viridis'):
        """
        Plot displacement field.
        
        Parameters:
        -----------
        component : int, optional
            0=x, 1=y, 2=z. None for magnitude.
        show : bool
        colormap : str
        """
        if self._displacement is None:
            raise ValueError("Must call solve() first")
    
        mesh = self.mesh.dolfin_mesh
    
        if component is None:
            disp_mag = dolfin.project(
                dolfin.sqrt(dolfin.dot(self._displacement, self._displacement)),
                dolfin.FunctionSpace(mesh, "Lagrange", 1)
            )
            title = "Displacement Magnitude"
        else:
            disp_mag = self._displacement.sub(component, deepcopy=True)
            comp_names = ['X', 'Y', 'Z']
            title = f"Displacement {comp_names[component]}"
    
        # Get compact display unit
        max_val = np.max(np.abs(disp_mag.vector()[:]))
        quantity = max_val * self.mesh.unit_system.get_length_unit()
        compact = quantity.to_compact()
        display_unit = compact.units
        scale = (1 * self.mesh.unit_system.get_length_unit()).to(display_unit).magnitude
    
        fig = plt.figure(figsize=(12, 8))
        p = dolfin.plot(disp_mag, cmap=colormap)
    
        cbar = plt.colorbar(p)
        cbar.set_label(f"Displacement ({display_unit:~})")
    
        # Rescale colorbar ticks
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t * scale:.3g}" for t in ticks])
    
        plt.title(f"{self.name} - {title}")
        plt.xlabel(f"{self.mesh.axis1_name} ({self.mesh.units})")
        plt.ylabel(f"{self.mesh.axis2_name} ({self.mesh.units})")
        plt.tight_layout()
    
        if show:
            plt.show()
        else:
            return fig

    def plot_deformed(self, scale=1.0, show=True):
        """
        Plot deformed mesh.
        
        Parameters:
        -----------
        scale : float
            Displacement scale factor for visualization
        show : bool
        """
        if self._displacement is None:
            raise ValueError("Must call solve() first")

        mesh = self.mesh.dolfin_mesh
        coords = mesh.coordinates().copy()

        # Get displacement at nodes
        V_cg1 = dolfin.VectorFunctionSpace(mesh, "CG", 1)
        u_cg1 = dolfin.project(self._displacement, V_cg1)

        # Deform coordinates
        disp_array = u_cg1.compute_vertex_values(mesh)
        dim = self.mesh.dim
        n_verts = mesh.num_vertices()

        deformed_coords = coords.copy()
        for i in range(dim):
            deformed_coords[:, i] += scale * disp_array[i * n_verts:(i + 1) * n_verts]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cells = mesh.cells()

        # Original
        axes[0].triplot(coords[:, 0], coords[:, 1], cells, 'b-', linewidth=0.3, alpha=0.5)
        axes[0].set_title("Original")
        axes[0].set_aspect('equal')
        axes[0].set_xlabel(f"{self.mesh.axis1_name} ({self.mesh.units})")
        axes[0].set_ylabel(f"{self.mesh.axis2_name} ({self.mesh.units})")

        # Deformed
        axes[1].triplot(deformed_coords[:, 0], deformed_coords[:, 1], cells, 'r-', linewidth=0.3, alpha=0.5)
        axes[1].set_title(f"Deformed (scale={scale}x)")
        axes[1].set_aspect('equal')
        axes[1].set_xlabel(f"{self.mesh.axis1_name} ({self.mesh.units})")
        axes[1].set_ylabel(f"{self.mesh.axis2_name} ({self.mesh.units})")

        plt.suptitle(f"{self.name} - Deformation")
        plt.tight_layout()

        if show:
            plt.show()
        else:
            return fig

    def plot_stress(self, component='von_mises', show=True, colormap='viridis'):
        """
        Plot stress field.
        
        Parameters:
        -----------
        component : str
            'von_mises', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        show : bool
        colormap : str (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        """
        if self._displacement is None:
            raise ValueError("Must call solve() first")

        mesh = self.mesh.dolfin_mesh
    
        stress_expr = self._get_stress_component(component)
        V_scalar = dolfin.FunctionSpace(mesh, "DG", 0)
        stress_field = dolfin.project(stress_expr, V_scalar)
    
        title = "Von Mises Stress" if component == 'von_mises' else f"Stress {component.upper()}"
    
        # Get compact display unit
        max_val = np.max(np.abs(stress_field.vector()[:]))
        quantity = max_val * self.mesh.unit_system.get_stress_unit()
        compact = quantity.to_compact()
        display_unit = compact.units
        scale = (1 * self.mesh.unit_system.get_stress_unit()).to(display_unit).magnitude
    
        fig = plt.figure(figsize=(12, 8))
        p = dolfin.plot(stress_field, cmap=colormap)
    
        cbar = plt.colorbar(p)
        cbar.set_label(f"Stress ({display_unit:~})")
    
        # Rescale colorbar ticks
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t * scale:.3g}" for t in ticks])
    
        plt.title(f"{self.name} - {title}")
        plt.xlabel(f"{self.mesh.axis1_name} ({self.mesh.units})")
        plt.ylabel(f"{self.mesh.axis2_name} ({self.mesh.units})")
        plt.tight_layout()
    
        if show:
            plt.show()
        else:
            return fig

    def get_max_displacement(self):
        """Return maximum displacement magnitude."""
        if self._displacement is None:
            raise ValueError("Must call solve() first")
        return np.abs(self._displacement.vector()[:]).max()

    def get_displacement_at(self, point):
        """Get displacement at a specific point."""
        if self._displacement is None:
            raise ValueError("Must call solve() first")
        return self._displacement(point)

    def save_results(self, output_file="results.pvd"):
        """
        Save results to file - format determined by extension.
        
        Supports:
        - ParaView formats: .pvd, .vtu, .xdmf, .xml
        - CSV format: .csv (for data analysis/curve fitting)
        
        Parameters:
        -----------
        output_file : str
            Output filename with extension
        """
        if self._displacement is None:
            raise ValueError("Must call solve() first")
    
        extension = os.path.splitext(output_file)[1].lower()
        if extension == "":
            extension = ".pvd"
            output_file += ".pvd"
    
        paraview_formats = [".pvd", ".vtu", ".xdmf", ".xml"]
    
        if extension in paraview_formats:
            self._save_results_paraview(output_file)
        elif extension == ".csv":
            self._save_results_csv(output_file)
        else:
            raise ValueError(f"Unsupported format: {extension}. Use {paraview_formats} or .csv")
        
        # Save mesh edges for GNN
        if output_file.endswith(".csv"):
            edges_file = output_file.replace(".csv", ".edges.csv")
            self._save_mesh_edges(edges_file)

    def _save_results_csv(self, output_file):
        """Save as CSV for data analysis with compact unit labels."""

        mesh = self.mesh.dolfin_mesh
        coords = mesh.coordinates()
        dim = self.mesh.dim
        n_verts = mesh.num_vertices()
    
        # Project displacement to vertices
        V_cg1 = dolfin.VectorFunctionSpace(mesh, "CG", 1)
        u_cg1 = dolfin.project(self._displacement, V_cg1)
    
        u_array = u_cg1.compute_vertex_values(mesh)
    
        u_values = np.zeros((n_verts, dim))
        for i in range(dim):
            u_values[:, i] = u_array[i * n_verts:(i + 1) * n_verts]
    
        u_mag = np.linalg.norm(u_values, axis=1)
    
        # Get stress components
        vm_values = self._get_stress_at_vertices('von_mises')
        s_xx = self._get_stress_at_vertices('xx')
        s_yy = self._get_stress_at_vertices('yy')
        s_xy = self._get_stress_at_vertices('xy')
        if dim == 3:
            s_zz = self._get_stress_at_vertices('zz')
            s_xz = self._get_stress_at_vertices('xz')
            s_yz = self._get_stress_at_vertices('yz')
    
        # Determine compact units based on mesh coordinates and von_mises stress only
        length_unit = self.mesh.units
        stress_unit = self.mesh.unit_system.stress
    
        coord_data = [coords[:, i] for i in range(dim)]
        _, length_unit_compact = self._compact_group(coord_data, length_unit)
    
        _, stress_unit_compact = self._compact_group([vm_values], stress_unit)
    
        # Calculate scale factors
        length_scale = (1 * UNIT_REGISTRY.parse_units(length_unit)).to(length_unit_compact).magnitude
        stress_scale = (1 * UNIT_REGISTRY.parse_units(stress_unit)).to(stress_unit_compact).magnitude
    
        # Build column data structure
        columns = []
    
        # Coordinates
        coord_names = ['x', 'y', 'z']
        for i in range(dim):
            columns.append({
                'name': coord_names[i],
                'data': coords[:, i] * length_scale,
                'unit': length_unit_compact
            })
    
        # Displacements
        for i in range(dim):
            columns.append({
                'name': f'u_{coord_names[i]}',
                'data': u_values[:, i] * length_scale,
                'unit': length_unit_compact
            })
    
        columns.append({
            'name': 'u_magnitude',
            'data': u_mag * length_scale,
            'unit': length_unit_compact
        })
    
        # Stresses
        stress_components = [
            ('von_mises', vm_values),
            ('stress_xx', s_xx),
            ('stress_yy', s_yy),
        ]
    
        if dim == 3:
            stress_components.extend([
                ('stress_zz', s_zz),
                ('stress_xy', s_xy),
                ('stress_xz', s_xz),
                ('stress_yz', s_yz),
            ])
        else:
            stress_components.append(('stress_xy', s_xy))
    
        for name, data in stress_components:
            columns.append({
                'name': name,
                'data': data * stress_scale,
                'unit': stress_unit_compact
            })
    
        # Build header
        header = [f"{col['name']} ({col['unit']})" for col in columns]
    
        # Add metadata columns
        for key in self.metadata.keys():
            header.append(key)
            columns.append({
                'name': key,
                'data': np.full(n_verts, self.metadata[key]),
                'unit': None
            })
    
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(n_verts):
                row = [col['data'][i] for col in columns]
                writer.writerow(row)
    
        print(f"Results saved to {output_file} (CSV format, {n_verts} points)")
        print(f"  Units: length={length_unit_compact}, stress={stress_unit_compact}")

    def _compact_group(self, values_list, unit_str):
        max_val = max(np.abs(v).max() for v in values_list)
        if max_val == 0:
            return values_list, unit_str

        unit = UNIT_REGISTRY.parse_units(unit_str)
        quantity = max_val * unit
        compact_quantity = quantity.to_compact()
        new_unit = f"{compact_quantity.units:~}"

        if new_unit == unit_str:
            return values_list, unit_str

        scale = (1 * unit).to(compact_quantity.units).magnitude
        return [v * scale for v in values_list], new_unit

    def _save_results_paraview(self, output_file):
        """Save for ParaView visualization."""
        mesh = self.mesh.dolfin_mesh
    
        # Displacement
        u_out = dolfin.Function(self._V, name="Displacement")
        u_out.assign(self._displacement)
    
        # Von Mises stress
        V_scalar = dolfin.FunctionSpace(mesh, "DG", 0)
        vm = self._get_stress_component('von_mises')
        vm_out = dolfin.project(vm, V_scalar)
        vm_out.rename("VonMises", "VonMises")
    
        # Save
        file = dolfin.File(output_file)
        file << u_out
        file << vm_out
    
        print(f"Results saved to {output_file} (ParaView format)")
        print(f"  Units: length={self.mesh.units}, stress={self.mesh.unit_system.stress}")

    def _save_mesh_edges(self, output_file):
        """Save mesh connectivity as edge list."""
        mesh = self.mesh.dolfin_mesh

        # Initialize edge connectivity if not already done
        mesh.init(1)  
    
        # Dolfin already tracks the edges without duplicates, but we also need the vertices
        edges = []
        for edge in dolfin.edges(mesh):
            vertices = edge.entities(0)
            edges.append([vertices[0], vertices[1]])
    
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_i', 'node_j'])
            for e in edges:
                writer.writerow(e)
    
        print(f"Mesh edges saved to {output_file} ({len(edges)} edges)")

    def _get_stress_tensor(self):
        """Compute stress tensor from displacement."""
        if self._displacement is None:
            raise ValueError("Must call solve() first")
    
        dim = self.mesh.dim
        E = self.material.get_E(self.mesh.unit_system)
        nu = self.material.nu
    
        mu = E / (2 * (1 + nu))
        if dim == 2:
            lmbda = E * nu / (1 - nu**2)
        else:
            lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
    
        eps = 0.5 * (dolfin.grad(self._displacement) + dolfin.grad(self._displacement).T)
        sig = lmbda * dolfin.div(self._displacement) * dolfin.Identity(dim) + 2 * mu * eps
    
        return sig
    
    
    def _get_stress_component(self, component):
        """Get a stress component as a dolfin expression.
        
        Parameters:
        -----------
        component : str
            'von_mises', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        """
        sig = self._get_stress_tensor()
        dim = self.mesh.dim
    
        if component == 'von_mises':
            if dim == 2:
                return dolfin.sqrt(
                    sig[0, 0]**2 + sig[1, 1]**2 - sig[0, 0]*sig[1, 1] + 3*sig[0, 1]**2
                )
            else:
                return dolfin.sqrt(
                    0.5 * ((sig[0, 0] - sig[1, 1])**2 +
                           (sig[1, 1] - sig[2, 2])**2 +
                           (sig[2, 2] - sig[0, 0])**2 +
                           6 * (sig[0, 1]**2 + sig[1, 2]**2 + sig[0, 2]**2))
                )
        else:
            comp_map = {
                'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
                'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)
            }
            if component not in comp_map:
                raise ValueError(f"Unknown component '{component}'. Use: von_mises, xx, yy, zz, xy, xz, yz")
            i, j = comp_map[component]
            return sig[i, j]

    def _get_stress_at_vertices(self, component):
        """Get stress component values at mesh vertices."""
        mesh = self.mesh.dolfin_mesh
        stress_expr = self._get_stress_component(component)
    
        # Project to DG0 (element-wise constant). This is done over CG1 since there is a sympy/CG1 bug between 
        # FEniCS + newer sympy incompatibility. The workaround we did (average DG0 to vertices manually) gives 
        # essentially the same result without triggering it.
        V_dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
        stress_dg0 = dolfin.project(stress_expr, V_dg0)
    
        # Average cell values to vertices manually
        cell_values = stress_dg0.vector()[:]
        n_verts = mesh.num_vertices()
        vertex_values = np.zeros(n_verts)
        vertex_counts = np.zeros(n_verts)
    
        for cell in dolfin.cells(mesh):
            cell_idx = cell.index()
            for vertex in cell.entities(0):
                vertex_values[vertex] += cell_values[cell_idx]
                vertex_counts[vertex] += 1
    
        vertex_values /= vertex_counts
        return vertex_values