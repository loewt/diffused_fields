"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Automatic keypoint detection utilities for action primitives.

This module provides functions to automatically detect keypoints on objects,
such as endpoints for elongated objects like bananas.
"""

import numpy as np
from diffused_fields.diffusion import PointcloudScalarDiffusion


def find_endpoints_via_diffusion(
    pcloud,
    num_candidates=10,
    temperature_threshold=0.1,
    min_distance_ratio=0.5
):
    """
    Find two endpoints of an elongated object using two-stage scalar diffusion.

    Strategy:
    1. Run diffusion from a central point to find the first endpoint (coldest point)
    2. Run diffusion from the first endpoint to find the second endpoint (coldest point on opposite side)

    Args:
        pcloud: Pointcloud object
        num_candidates: Number of cold points to consider as candidates (unused in current implementation)
        temperature_threshold: Relative threshold for considering a point "cold" (unused)
        min_distance_ratio: Minimum ratio of endpoint distance to object diameter

    Returns:
        tuple: (endpoint1_idx, endpoint2_idx) - indices of the two endpoints
    """
    # Get boundary points to start from
    if not hasattr(pcloud, 'is_boundary_arr'):
        pcloud.get_boundary()
    boundary_vertices = np.where(pcloud.is_boundary_arr)[0]

    # If no boundary detected or all points are boundary, use all vertices
    if len(boundary_vertices) == 0 or len(boundary_vertices) == len(pcloud.vertices):
        print("No explicit boundary detected, using all vertices")
        boundary_vertices = np.arange(len(pcloud.vertices))

    # Choose a point near the "middle" of the object as heat source
    # Use the centroid-closest point
    centroid = np.mean(pcloud.vertices, axis=0)
    distances_to_centroid = np.linalg.norm(
        pcloud.vertices[boundary_vertices] - centroid, axis=1
    )
    middle_point_idx = boundary_vertices[np.argmin(distances_to_centroid)]

    # STAGE 1: Run scalar diffusion from middle point to find first endpoint
    print("Stage 1: Finding first endpoint...")
    scalar_diffusion_1 = PointcloudScalarDiffusion(
        pcloud,
        diffusion_scalar=100.0
    )
    scalar_diffusion_1.source_vertices = [middle_point_idx]
    scalar_diffusion_1.get_local_bases()

    # Get the scalar field (temperature values)
    scalar_field_1 = scalar_diffusion_1.ut

    # Normalize to [0, 1]
    scalar_min = scalar_field_1.min()
    scalar_max = scalar_field_1.max()
    normalized_field_1 = (scalar_field_1 - scalar_min) / (scalar_max - scalar_min)

    # Find coldest point (first endpoint)
    boundary_scalars_1 = normalized_field_1[boundary_vertices]
    coldest_idx_1 = np.argmin(boundary_scalars_1)
    endpoint1 = boundary_vertices[coldest_idx_1]

    print(f"First endpoint found at vertex {endpoint1}, temperature: {normalized_field_1[endpoint1]:.4f}")

    # STAGE 2: Run diffusion from first endpoint to find second endpoint
    print("Stage 2: Finding second endpoint from first endpoint...")
    scalar_diffusion_2 = PointcloudScalarDiffusion(
        pcloud,
        diffusion_scalar=100.0
    )
    scalar_diffusion_2.source_vertices = [endpoint1]
    scalar_diffusion_2.get_local_bases()

    # Get the scalar field from second diffusion
    scalar_field_2 = scalar_diffusion_2.ut

    # Normalize to [0, 1]
    scalar_min_2 = scalar_field_2.min()
    scalar_max_2 = scalar_field_2.max()
    normalized_field_2 = (scalar_field_2 - scalar_min_2) / (scalar_max_2 - scalar_min_2)

    # Find coldest point (second endpoint, should be on opposite side)
    boundary_scalars_2 = normalized_field_2[boundary_vertices]
    coldest_idx_2 = np.argmin(boundary_scalars_2)
    endpoint2 = boundary_vertices[coldest_idx_2]

    print(f"Second endpoint found at vertex {endpoint2}, temperature: {normalized_field_2[endpoint2]:.4f}")

    # Verify the endpoints are reasonable
    distance = np.linalg.norm(pcloud.vertices[endpoint1] - pcloud.vertices[endpoint2])
    object_diameter = np.linalg.norm(
        pcloud.vertices.max(axis=0) - pcloud.vertices.min(axis=0)
    )

    print(f"Distance between endpoints: {distance:.4f}")
    print(f"Object diameter: {object_diameter:.4f}")
    print(f"Ratio: {distance/object_diameter:.2%}")

    if distance < min_distance_ratio * object_diameter:
        print(
            f"Warning: Detected endpoints are close ({distance:.3f} vs "
            f"diameter {object_diameter:.3f}). Consider adjusting parameters."
        )

    return (endpoint1, endpoint2)


def find_endpoints_via_extremal_projection(pcloud, axis=None):
    """
    Find endpoints by projecting onto principal axis.

    Args:
        pcloud: Pointcloud object
        axis: Optional 3D vector defining the projection axis.
              If None, uses PCA to find principal axis.

    Returns:
        tuple: (endpoint1_idx, endpoint2_idx) - indices of the two endpoints
    """
    if axis is None:
        # Use PCA to find principal axis
        vertices_centered = pcloud.vertices - np.mean(pcloud.vertices, axis=0)
        cov_matrix = np.cov(vertices_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Principal axis is the eigenvector with largest eigenvalue
        axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Project all vertices onto this axis
    projections = np.dot(pcloud.vertices, axis)

    # Get boundary points
    if not hasattr(pcloud, 'is_boundary_arr'):
        pcloud.get_boundary()
    boundary_vertices = np.where(pcloud.is_boundary_arr)[0]

    # Find boundary points with min and max projections
    boundary_projections = projections[boundary_vertices]
    min_idx = boundary_vertices[np.argmin(boundary_projections)]
    max_idx = boundary_vertices[np.argmax(boundary_projections)]

    return (min_idx, max_idx)
