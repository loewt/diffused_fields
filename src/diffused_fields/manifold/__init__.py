"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Geometric structures and manifold operations.

This module contains the core geometric data structures:
- Manifold: Base geometric manifold class
- Pointcloud: Point cloud data structure and operations
- Mesh: Triangle mesh data structure and operations  
- Grid: Grid utilities and voxel operations

Dependencies: numpy, scipy, trimesh, scikit-learn, potpourri3d
"""

# Core geometric structures
from .manifold import Manifold, Plane, Line, Sphere
from .pointcloud import Pointcloud
from .mesh import Mesh
from .grid import Grid

__all__ = [
    'Manifold', 'Plane', 'Line', 'Sphere',
    'Pointcloud', 'Mesh', 'Grid'
]