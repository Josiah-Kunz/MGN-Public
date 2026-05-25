"""
Example of applying a mesh graph network (MGN).

MGN uses FEM context directly (boundaries, loads, mesh topology) rather than CSV files.
This allows it to learn geometry-agnostic physics that can transfer to new geometries.
"""

from main_package import *
import os
import numpy as np

# Settings

TRAIN_GEOMETRIES = [
    "plate-with-hole_8in.stp",
    "plate-with-hole_8in_square.stp",
    "plate-with-hole_8in_ellipse.stp",
    "plate-with-hole_4in.stp",
    "plate-with-hole_8in_double.stp",
    "plate-with-hole_8in_offset-right.stp",
    "plate-with-hole_8in_offset-right_1in.stp",
]
TEST_GEOMETRY = "plate-with-hole_8in_hex.stp"

MESH_SIZE = 25 # mm

num_loads = 20
TRAIN_LOADS = (
        np.linspace(1000, 4500, num_loads//2, dtype=int).tolist() +
        np.linspace(5500, 10000, num_loads//2, dtype=int).tolist()
)
TEST_LOAD = 5000

def main():

    # Create visualizations
    mgn = MGN(
        hidden_channels=64,
        num_layers=20,
        global_features=["load"],
        learning_rate=1e-4,
        epochs=5000,
    )

    # Create training FEMs (all geometries x all loads)
    train_fems = []
    for geom_file in TRAIN_GEOMETRIES:
        geometry_path = os.path.join("geometries", geom_file)
        mesh = MeshObject(geometry_path, MESH_SIZE, units=Units.SI_MM, force_2d=True, convert_to=Units.US_IN)
        print(f"\nGeometry: {geom_file} ({len(mesh.dolfin_mesh.coordinates())} nodes)")

        for load in TRAIN_LOADS:
            fem = create_fem(mesh, load, geom_file)
            fem.solve()
            train_fems.append(fem)

        print(f"\tSolved {len(TRAIN_LOADS)} load cases")

    print(f"\nTotal training FEMs: {len(train_fems)}")

    # Create test FEM
    test_geometry_path = os.path.join("geometries", TEST_GEOMETRY)
    test_mesh = MeshObject(test_geometry_path, MESH_SIZE, units=Units.SI_MM, force_2d=True, convert_to=Units.US_IN)
    test_fem = create_fem(test_mesh, TEST_LOAD, TEST_GEOMETRY)
    test_fem.solve()
    print(f"\nTest geometry: {TEST_GEOMETRY} ({len(test_mesh.dolfin_mesh.coordinates())} nodes)")

    # Define ML wrapper
    ml = MLObject(
        train_fem=train_fems,
        test_fem=test_fem,
        objectives=["von_mises"],
        name="Multi-Geometry MGN"
    )

    # Train MGN
    mgn_filename = "results/multi_geometry_mgn.pt"
    ml.train(model=mgn, model_name="MGN", batch_size=1)
    mgn.save(mgn_filename)

    # Evaluate on unseen geometry and unseen load
    ml.evaluate_on_unseen_data()
    ml.plot_predictions()
    ml.plot_loss()
    ml.plot_ml_vs_fem(test_mesh)

    # Test on each training geometry with unseen load
    print("\n" + "="*50)
    print("Evaluating on training geometries with unseen load:")
    print("="*50)
    for geom_file in TRAIN_GEOMETRIES:
        geometry_path = os.path.join("geometries", geom_file)
        mesh = MeshObject(geometry_path, MESH_SIZE, units=Units.SI_MM, force_2d=True, convert_to=Units.US_IN)
        fem = create_fem(mesh, TEST_LOAD, geom_file)  # Unseen load
        fem.solve()
        r2 = mgn.score(fem, fem.von_mises)
        print(f"  {geom_file}: R² = {r2:.4f}")

def create_fem(mesh, load, geom_name):
    """Create FEM object for a given mesh and load (but does *not* solve it)."""
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