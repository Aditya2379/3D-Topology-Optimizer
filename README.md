# 3D Topology Optimizer

A robust, pure-Python Computational Solid Mechanics engine designed to bridge the gap between mathematical structural analysis and Additive Manufacturing. 

This tool performs true **CAD-to-CAD Topology Optimization**. It ingests arbitrary 3D models (`.stl`), mathematically carves away inefficient material using Finite Element Analysis (FEA), and exports a brand new, highly optimized, watertight 3D mesh ready for 3D printing.

## Features
- **Dynamic Voxelization**: Automatically imports native `.stl` files and discretizes them into accurate volumetric 3D Boolean grids.
- **3D FEA Engine**: Computes complex structural strain energy using 8-node hexahedral element stiffness matrices.
- **SIMP Algorithm**: Iteratively removes material using the Solid Isotropic Material with Penalization method to minimize compliance (maximize rigidity) at a given volume fraction.
- **Sensitivity Filtering**: Employs Gaussian density blurring to eliminate "checkerboard" artifacts and enforce minimum length scales.
- **Marching Cubes CAD Export**: Extracts cohesive, organic surface meshes from the optimized density fields and exports them back to `.stl` format.

## Installation

Ensure you have Python 3.8+ installed. The engine relies on heavily optimized numerical and geometry libraries.

```bash
pip install numpy scipy scikit-image trimesh
```

## Usage

1. Place your starting CAD file in the root directory (the default script expects `input_beam.stl`).
2. Run the optimization engine:

```bash
python topopt_3d.py
```

3. The solver will output convergence metrics to the terminal. Once the target volume fraction is reached, it will export the finalized structure to `optimized_beam.stl`.

*(Note: 3D Finite Element Analysis involves solving massive sparse matrices. High-resolution voxel grids will exponentially increase RAM usage and convergence time).*

## How It Works
The engine fixes the leftmost physical extremity of the provided CAD geometry to a rigid boundary and applies a concentrated downward force to the rightmost extremity. It then seeks to remove a specified percentage (e.g., 70%) of the material while maintaining structural integrity. The resulting shapes are organic, bone-like structures characteristic of generative design.
