"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Diffused Fields - A Python package for diffusion-based methods on manifolds.

This package implements diffusion algorithms on geometric manifolds including
point clouds, meshes, and other geometric structures. It provides both
traditional diffusion solvers and walk-on-spheres methods.

The package is organized into focused submodules:
- manifold: Geometric structures (Pointcloud, Mesh, Grid, Manifold)
- diffusion: Diffusion algorithms and solvers (scalar, vector, walk-on-spheres)

Copyright (c) 2024 Idiap Research Institute
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

# Version information
__version__ = "0.1.0"
__author__ = "Cem Bilaloglu"
__email__ = "cem.bilaloglu@idiap.ch"
__license__ = "MIT"

# Import from organized submodules with error handling
_MANIFOLD_IMPORTS = []
_DIFFUSION_IMPORTS = []
_UTILS_IMPORTS = []

# Manifold submodule - geometric structures
try:
    from .manifold import (
        Manifold, Plane, Line, Sphere, Pointcloud, Mesh, Grid
    )
    _MANIFOLD_IMPORTS = [
        "Manifold", "Plane", "Line", "Sphere", "Pointcloud", "Mesh", "Grid"
    ]
except ImportError as e:
    print(f"Warning: Manifold submodule import failed: {e}")

# Solver/Field/Problem - diffusion algorithms and solvers
try:
    from .field import Field, QuaternionField, ScalarField, VectorField
    from .problem import DiffusionProblem, Problem
    from .solver import LaplacianSolver, Solver, WalkOnSpheresSolver
    _DIFFUSION_IMPORTS = [
        "Field", "ScalarField", "VectorField", "QuaternionField",
        "Problem", "DiffusionProblem",
        "Solver", "LaplacianSolver", "WalkOnSpheresSolver",
    ]

except ImportError as e:
    print(f"Warning: Solver/Field/Problem submodule import failed: {e}")

# Utils submodule - utility functions
try:
    from .utils import (
        find_endpoints_via_diffusion,
        find_endpoints_via_extremal_projection,
    )
    _UTILS_IMPORTS = [
        "find_endpoints_via_diffusion",
        "find_endpoints_via_extremal_projection",
    ]
except ImportError as e:
    print(f"Note: Utils submodule not available: {e}")

# Combine all successful imports
__all__ = _MANIFOLD_IMPORTS + _DIFFUSION_IMPORTS + _UTILS_IMPORTS

# Provide easy access to submodules
from . import field, manifold, problem, solver

# Optional submodules (may fail)
try:
    from . import utils
except ImportError:
    pass

# No automatic print statements on import - keep the console clean