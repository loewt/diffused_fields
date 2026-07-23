"""
Copyright (c) 2026 Tobias Löw

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Lightweight geometry backend replacing Open3D.

Provides file IO (via trimesh), a minimal point cloud container, oriented
bounding boxes, and the handful of point-cloud/mesh processing algorithms
(voxel downsampling, DBSCAN clustering, normal estimation + consistent
orientation, oriented bounding boxes) that this package used to obtain from
Open3D. Kept intentionally narrow: only what `manifold/pointcloud.py` and
`manifold/mesh.py` actually call.
"""




import itertools
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
def _extract_vertex_colors(visual, num_vertices):
    colors = getattr(visual, "vertex_colors", None)
    if colors is None or len(colors) == 0:
        return np.zeros((0, 3))
    colors = np.asarray(colors, dtype=np.float64)[:, :3]
    if colors.max() > 1.0:
        colors = colors / 255.0
    return colors


def read_point_cloud(filepath):
    """Load a point cloud file. Returns (points, colors) float64 arrays."""
    loaded = trimesh.load(filepath, process=False)
    if isinstance(loaded, trimesh.points.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = _extract_vertex_colors(loaded, len(points))
    elif isinstance(loaded, trimesh.Trimesh):
        # Point-cloud-only PLY files can be parsed as a degenerate mesh
        # (vertices with no faces); fall back to just using the vertices.
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = _extract_vertex_colors(loaded.visual, len(points))
    else:
        raise ValueError(f"Unsupported point cloud file: {filepath}")
    return points, colors


def read_triangle_mesh(filepath):
    """Load a triangle mesh file. Returns (vertices, faces, vertex_colors)."""
    mesh = trimesh.load(filepath, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = _extract_vertex_colors(mesh.visual, len(vertices))
    return vertices, faces, colors


def compute_vertex_normals(vertices, faces):
    """Area-weighted vertex normals, matching Open3D's convention."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.vertex_normals, dtype=np.float64)


class OrientedBoundingBox:
    """Minimal stand-in for open3d.geometry.OrientedBoundingBox."""

    def __init__(self, center, extent, rotation):
        self.center = np.asarray(center, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.R = np.asarray(rotation, dtype=np.float64)

    @classmethod
    def from_points(cls, points):
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 3:
            center = points.mean(axis=0) if len(points) else np.zeros(3)
            return cls(center=center, extent=np.zeros(3), rotation=np.eye(3))

        # world_to_box: 4x4 transform mapping world points into the
        # axis-aligned frame of the (near-minimal) oriented bounding box.
        world_to_box, extent = trimesh.bounds.oriented_bounds(points)
        box_to_world = np.linalg.inv(world_to_box)
        rotation = box_to_world[:3, :3]
        center = box_to_world[:3, 3]
        return cls(center=center, extent=extent, rotation=rotation)

    def get_center(self):
        return self.center

    def scale(self, scale, center=None):
        if center is None:
            center = self.center
        center = np.asarray(center, dtype=np.float64)
        self.center = center + scale * (self.center - center)
        self.extent = self.extent * scale
        return self

    def get_box_points(self):
        signs = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))
        local_corners = signs * self.extent
        return local_corners @ self.R.T + self.center


def voxel_down_sample(points, colors, voxel_size):
    """Average points (and colors) that fall in the same voxel bin."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    num_voxels = unique_index.shape[0]

    counts = np.bincount(inverse, minlength=num_voxels).astype(np.float64)
    down_points = np.zeros((num_voxels, 3))
    for axis in range(3):
        down_points[:, axis] = (
            np.bincount(inverse, weights=points[:, axis], minlength=num_voxels)
            / counts
        )

    if colors.size:
        down_colors = np.zeros((num_voxels, 3))
        for axis in range(3):
            down_colors[:, axis] = (
                np.bincount(
                    inverse, weights=colors[:, axis], minlength=num_voxels)
                / counts
            )
    else:
        down_colors = np.zeros((0, 3))

    return down_points, down_colors


def cluster_dbscan(points, eps, min_points):
    """DBSCAN labels for `points`; -1 marks noise, matching Open3D's convention."""
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    return DBSCAN(eps=eps, min_samples=min_points).fit(points).labels_


def estimate_normals(points, knn):
    """Per-point normals via PCA of the k-nearest-neighbor local covariance."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3))
    k = min(knn, n)
    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=k)
    if k == 1:
        neighbor_idx = neighbor_idx[:, None]

    neighborhoods = points[neighbor_idx]  # (n, k, 3)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered) / k
    _, eigenvectors = np.linalg.eigh(covariances)  # ascending eigenvalue order
    return eigenvectors[:, :, 0]  # eigenvector of the smallest eigenvalue


def orient_normals_consistent_tangent_plane(points, normals, k):
    """
    Propagate a consistent normal orientation across the point cloud.

    Reimplements the Hoppe et al. (1992) approach used by Open3D /
    PCL: build a k-NN graph weighted by 1 - |n_i . n_j|, take its minimum
    spanning forest, then flip normals along a breadth-first traversal of
    each tree so neighboring normals agree in sign. Since the weighting
    ignores sign, this only fixes *consistency*, not absolute direction
    (e.g. inward vs. outward) -- same caveat as Open3D's implementation.
    """
    n = len(points)
    if n <= 1:
        return normals.copy()
    k = min(k, n - 1)

    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=k + 1)  # includes self at column 0
    neighbor_idx = neighbor_idx[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = neighbor_idx.ravel()
    alignment = np.abs(np.einsum("ij,ij->i", normals[rows], normals[cols]))
    weights = 1.0 - alignment

    graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(graph)

    oriented = normals.copy()
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        order, predecessors = breadth_first_order(
            mst, i_start=start, directed=False, return_predecessors=True
        )
        visited[order] = True
        for node in order[1:]:
            parent = predecessors[node]
            if np.dot(oriented[node], oriented[parent]) < 0:
                oriented[node] = -oriented[node]

    return oriented
