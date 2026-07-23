"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Utility functions for the diffused fields library.
"""

from .keypoint_detection import (
    find_endpoints_via_diffusion,
    find_endpoints_via_extremal_projection,
)

__all__ = [
    'find_endpoints_via_diffusion',
    'find_endpoints_via_extremal_projection',
]
