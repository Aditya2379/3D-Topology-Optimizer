import trimesh
import numpy as np

def create_input_block():
    print("Generating input_beam.stl (30x10x10 block)...")
    # Create a simple box of dimensions 30x10x10
    mesh = trimesh.creation.box(extents=(30, 10, 10))
    # Shift so bottom left corner is at (0,0,0)
    mesh.apply_translation((15, 5, 5))
    mesh.export('input_beam.stl')
    print("Created input_beam.stl successfully!")

if __name__ == "__main__":
    create_input_block()
