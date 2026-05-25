"""
Load trained MGN and test on unseen geometries.
"""

from main_package import *
import os

# Settings
TEST_GEOMETRIES = [

    # Previously trained

#    "plate-with-hole_8in.stp",
#    "plate-with-hole_8in_square.stp",
#    "plate-with-hole_8in_ellipse.stp",
#    "plate-with-hole_4in.stp",
#    "plate-with-hole_8in_double.stp",
#    "plate-with-hole_8in_offset-right.stp",
#    "plate-with-hole_8in_offset-right_1in.stp",
#    "plate-with-hole_4in_double.stp",
#    "plate-with-hole_1in.stp",
#    "plate-without-hole.stp",
#    "plate-with-hole_3holes.stp",
    
    # NEW!

    "plate-with-hole_8in_hex.stp",
    "plate-with-hole_8in_tri.stp",
    "plate-with-hole_8in_j.stp",
    "plate-with-hole_4in_j.stp",
    "plate-with-hole_8in_figure8.stp",
    "plate-with-hole_8in_track.stp",
    "plate-with-hole_8in_track_sideways.stp",

]

MESH_SIZE = 25  # mm
TEST_LOAD = 12000  # psi
MODEL_PATH = "results/multi_geometry_mgn.pt"

def main():
    # Load trained model
    print(f"Loading model from {MODEL_PATH}...")
    mgn = MGN.load(MODEL_PATH)

    # Test on each geometry
    print("\n" + "=" * 50)
    print("Evaluating on test geometries:")
    print("=" * 50)

    for geom_file in TEST_GEOMETRIES:
        print(f"\n--- {geom_file} ---")

        # Create mesh and FEM
        geometry_path = os.path.join("geometries", geom_file)
        mesh = MeshObject(geometry_path, MESH_SIZE, units=Units.SI_MM, force_2d=True, convert_to=Units.US_IN, force_regenerate=True)
        print(f"Nodes: {len(mesh.dolfin_mesh.coordinates())}")

        fem = create_fem(mesh, TEST_LOAD, geom_file)
        fem.solve()
        #mgn.visualize_mesh(fem)

        # Score and visualize
        r2 = mgn.score(fem, fem.von_mises)
        print(f"R² = {r2:.4f}")

        # Plot comparison
        mgn.plot_ml_vs_fem(fem, show=True)


def create_fem(mesh, load, geom_name):
    """Create FEM object for a given mesh and load."""
    material = Material.steel()

    loads = LoadCollection(units=Units.US_IN)
    loads.add(SurfaceLoad(
        location=lambda x, on_boundary: on_boundary and x[0] > mesh.bounds["x"][1] - 1e-6,
        pressure=(load, 0, 0),
        name="Applied Load"
    ))

    fem = FEMObject(
        mesh=mesh,
        loads=loads,
        boundaries=[FixedBoundary('left', value=(0, 0, 0), name="Fixed")],
        material=material,
        name=geom_name.replace('.stp', '').replace('-', '_'),
        metadata={"load": load},
    )
    return fem


if __name__ == "__main__":
    main()