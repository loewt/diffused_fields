"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Diffusion algorithms and solvers.

This module contains all diffusion-related algorithms:
- Point cloud diffusion solvers (scalar, quaternion)
- Walk-on-spheres algorithms

Dependencies: numpy, scipy, robust_laplacian, potpourri3d, gafropy
"""

# Base diffusion classes
from .pointcloud_scalar_diffusion import DiffusionSolver, PointcloudScalarDiffusion
from .pointcloud_quaternion_diffusion import PointcloudQuaternionDiffusion
from .walk_on_spheres import WalkOnSpheresDiffusion

__all__ = [
    'DiffusionSolver',
    'PointcloudScalarDiffusion',
    'PointcloudQuaternionDiffusion',
    'WalkOnSpheresDiffusion',
]