"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from . import _geometry_backend as geometry_backend
from ._rotor_interop import as_gafropy_rotor
from .grid import Grid
from .manifold import Manifold, Sphere


class _TriangleMeshData:
    """Minimal triangle-mesh container replacing open3d.geometry.TriangleMesh."""

    def __init__(self, vertices=None, faces=None, vertex_colors=None):
        self.vertices = np.zeros((0, 3)) if vertices is None else np.asarray(
            vertices, dtype=np.float64
        )
        self.triangles = np.zeros((0, 3), dtype=np.int64) if faces is None else np.asarray(
            faces, dtype=np.int64
        )
        self.vertex_colors = np.zeros((0, 3)) if vertex_colors is None else np.asarray(
            vertex_colors, dtype=np.float64
        )

    def get_center(self):
        return self.vertices.mean(axis=0)

    def scale(self, scale, center):
        self.vertices = center + scale * (self.vertices - center)
        return self

    def rotate(self, rotation_matrix, center):
        self.vertices = (self.vertices - center) @ rotation_matrix.T + center
        return self

    def translate(self, translation, relative=True):
        translation = np.asarray(translation, dtype=np.float64)
        if relative:
            self.vertices = self.vertices + translation
        else:
            self.vertices = self.vertices - self.get_center() + translation
        return self

    def compute_vertex_normals(self):
        self.vertex_normals = geometry_backend.compute_vertex_normals(
            self.vertices, self.triangles
        )
        return self.vertex_normals

    def get_oriented_bounding_box(self):
        return geometry_backend.OrientedBoundingBox.from_points(self.vertices)


# child class
class Mesh(Manifold):
    def __init__(
        self,
        vertices=None,
        faces=None,
        filename=None,
        voxel_size=None,
        scale=None,
        translation=None,
        rotation=None,
        normal_orientation=None,
        file_directory=None,
        center_position=None,
        center_vertex=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            type=type(self), scale=scale, translation=translation, rotation=rotation
        )

        # No default directory: pass file_directory= explicitly, or pass a
        # relative/absolute path directly via filename=.
        self.file_directory = file_directory or ""
        self.voxel_size = voxel_size
        self.normal_orientation = normal_orientation

        # Construct the point cloud
        # ==============================================================================
        object_name = "default"
        if (
            vertices is not None and faces is not None
        ):  # initialize from vertices and faces
            self.mesh = _TriangleMeshData(vertices, faces)
        elif filename is not None:  # initialize from file
            object_name = filename.split(".")[0]
            filepath = self.file_directory + filename
            # print(f"Reading mesh from {filepath}")
            file_vertices, file_faces, file_colors = geometry_backend.read_triangle_mesh(
                filepath
            )
            self.mesh = _TriangleMeshData(file_vertices, file_faces, file_colors)
            # Extract vertices and faces as NumPy arrays
            vertices = np.asarray(self.mesh.vertices)  # Nx3 numpy array
            faces = np.asarray(self.mesh.triangles)  # Mx3 numpy array

        self.object_name = object_name
        self._set_default_parameters()

        # Transform the point cloud
        # ==============================================================================
        self.mesh.scale(self.scale, center=self.mesh.get_center())
        self.mesh.rotate(self.rotation.as_matrix(), center=self.mesh.get_center())
        self.mesh.translate(self.translation, relative=False)

        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

        if center_position is None:
            if center_vertex is not None:
                self.center_position = self.vertices[center_vertex]
            else:
                self.center_position = self.mesh.get_center()
        else:
            self.center_position = center_position

        self.center_offset = self.mesh.get_center() - self.center_position

        self.colors = np.asarray(self.mesh.vertex_colors)
        if self.colors.size == 0:  # pointcloud have no color
            self.colors = np.zeros_like(self.vertices)

    def translate_center(self, position):
        target_pos = position + self.center_offset
        self.mesh.translate(target_pos, relative=False)
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

    def get_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.kd_tree = cKDTree(self.vertices)
        # self.knn_graph = knn_graph(self.vertices, self.num_neighbors)
        return self.kd_tree

    def get_boundary_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.boundary_kd_tree = cKDTree(self.vertices[self.is_boundary_arr])
        # self.knn_graph = knn_graph(self.vertices, self.num_neighbors)
        return self.boundary_kd_tree

    def _set_default_parameters(self):
        """
        Fill in scale/translation/rotation/normal_orientation with fixed
        defaults for any that weren't passed to the constructor.

        These are runtime constructor parameters, not read from a config
        file. The identity rotation is built directly (not via Euler
        angles) -- pass `rotation=` explicitly (e.g. a gafropy Rotor, or
        gafropy.numpy.Rotor.from_euler_angles(...) if Euler angles are
        genuinely the representation you have) if you need something else.
        """
        if self.scale is None:
            self.scale = 1.0
        if self.translation is None:
            self.translation = np.array([0.0, 0.0, 0.0])
        if self.rotation is None:
            self.rotation = R.identity()
        if self.normal_orientation is None:
            self.normal_orientation = 1

    def get_center(self):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        self.center_point = np.mean(self.vertices, axis=0)
        _, center_vertex = self.kd_tree.query(np.array([self.center_point]), k=1)
        self.center_vertex = center_vertex[0]
        return self.center_point, self.center_vertex

    def get_closest_points(self, points):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        distances, indices = self.kd_tree.query(points, k=1)
        return distances, indices

    def get_bounding_box(self, scale=1.0):
        self.oriented_bounding_box = self.mesh.get_oriented_bounding_box()
        self.oriented_bounding_box.scale(
            scale, center=self.oriented_bounding_box.get_center()
        )
        self.oriented_bounding_box_corners = np.asarray(
            self.oriented_bounding_box.get_box_points()
        )
        return self.oriented_bounding_box

    def get_bounding_sphere(self):
        if not hasattr(self, "center_point"):
            self.get_center()
        if not hasattr(self, "oriented_bounding_box"):
            self.get_bounding_box(scale=1.5)
        sphere_center = self.center_point

        enclosing_sphere_radius = np.max(
            np.linalg.norm(
                self.oriented_bounding_box_corners - self.oriented_bounding_box.center,
                axis=1,
            )
        )
        self.bounding_sphere = Sphere(
            radius=enclosing_sphere_radius, center=sphere_center
        )
        return self.bounding_sphere

    def get_bounding_box_grid(self, bounding_box_scalar=2.0, nb_points=5):
        self.get_bounding_box(bounding_box_scalar)

        min_vals = np.min(self.oriented_bounding_box_corners, axis=0)
        max_vals = np.max(self.oriented_bounding_box_corners, axis=0)

        x_min, y_min, z_min = min_vals
        x_max, y_max, z_max = max_vals
        grid = Grid(
            Nx=nb_points,
            Ny=nb_points,
            Nz=nb_points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        return grid

    def get_normals(self, num_neighbors=30):
        if not hasattr(self, "center_vertex"):
            self.get_center()
        self.mesh.compute_vertex_normals()
        normals = np.asarray(self.mesh.vertex_normals)
        self.normals = normals * self.normal_orientation
        return self.normals

    def get_signed_distance(self, position):
        distance, point_index = self.kd_tree.query(position, k=1)
        projected_point = self.vertices[point_index]
        projected_normal = self.normals[point_index]

        sign = np.sign(np.dot(position - projected_point, projected_normal))
        signed_distance = distance * sign
        return signed_distance, point_index

    def correct_distance_smooth(
        self, position, distance_target, epsilon=5e-4, max_iterations=15, max_error=1e-1
    ):
        position = np.float32(position)
        for i in range(max_iterations):
            signed_distance, point_index = self.get_signed_distance(position)
            error = distance_target - signed_distance
            if np.abs(error) < epsilon:
                return position, signed_distance, point_index
            elif np.abs(error) > max_error:
                error_sign = np.sign(error)
                error = error_sign * max_error

            local_normal = self.normals[point_index]
            correction = error * 0.2 * local_normal
            position += correction
        return position, signed_distance, point_index

    def get_local_basis(self):
        if not hasattr(self, "normals"):
            self.get_normals()
        self.tangent_vectors_u = np.column_stack(
            [self.normals[:, 1], -self.normals[:, 0], np.zeros(len(self.vertices))]
        )
        self.tangent_vectors_v = np.cross(self.normals, self.tangent_vectors_u)

    def get_k_edges(self, num_neighbors=3):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        self.d_kdtree, idx = self.kd_tree.query(self.vertices, k=num_neighbors)

        idx = idx[:, 1:]
        point_numbers = np.arange(len(self.vertices))
        point_numbers = np.repeat(point_numbers, num_neighbors - 1)
        idx_flatten = idx.flatten()
        edges = np.vstack((point_numbers, idx_flatten)).T

        return edges

    def get_mean_edge_length(self):
        edges = self.get_k_edges()
        edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        self.mean_edge_length = np.mean(edge_lengths)

        print(f"Mean edge length: {self.mean_edge_length*1e3:.1f} mm")
        return self.mean_edge_length

    def get_boundary(self, max_neighbors=30, angle_threshold=np.pi / 2):
        def is_boundary(
            vertex,
            neighbor_vertices,
            tangent_vector_u,
            tangent_vector_v,
            angle_threshold=3 * np.pi / 2,
        ):
            angles = np.zeros(len(neighbor_vertices))
            for j in range(len(neighbor_vertices)):
                neighbor = neighbor_vertices[j]
                delta = neighbor - vertex
                angles[j] = np.arctan2(
                    np.dot(delta, tangent_vector_u), np.dot(delta, tangent_vector_v)
                )

            angles = np.sort(angles)
            diff = np.diff(angles)
            diff = np.append(diff, 2 * np.pi - angles[-1] + angles[0])
            if np.max(diff) > angle_threshold:
                return True
            return False

        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        if not hasattr(self, "tangent_vectors_u"):
            self.get_local_basis()
        is_boundary_arr = np.zeros(len(self.vertices)).astype(bool)
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            distances, indices = self.kd_tree.query(vertex, k=max_neighbors)
            neighbor_vertices = self.vertices[indices[0:]]
            is_boundary_arr[i] = is_boundary(
                vertex,
                neighbor_vertices,
                self.tangent_vectors_u[i],
                self.tangent_vectors_v[i],
                angle_threshold,
            )
        self.is_boundary_arr = is_boundary_arr
        return is_boundary_arr

    def get_boundary_normals(self):
        if not hasattr(self, "is_boundary_arr"):
            self.get_boundary()
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()

        boundary_vertices = self.vertices[self.is_boundary_arr]
        boundary_normal_vectors = np.zeros_like(boundary_vertices)

        for i, vertex in enumerate(boundary_vertices):
            distances, indices = self.kd_tree.query(vertex, k=10)
            neighbor_vertices = self.vertices[indices[1:]]

            inward_direction = np.mean(neighbor_vertices - vertex, axis=0)
            inward_direction /= np.linalg.norm(inward_direction)

            tangent_vector_u = self.tangent_vectors_u[self.is_boundary_arr][i]
            tangent_vector_v = self.tangent_vectors_v[self.is_boundary_arr][i]

            projected_u = np.dot(inward_direction, tangent_vector_u) * tangent_vector_u
            projected_v = np.dot(inward_direction, tangent_vector_v) * tangent_vector_v
            tangent_projection = projected_u + projected_v

            tangent_projection /= np.linalg.norm(tangent_projection)
            boundary_normal_vectors[i] = tangent_projection
        self.boundary_normals = boundary_normal_vectors
        self.boundary_tangents = np.cross(
            boundary_normal_vectors, self.normals[self.is_boundary_arr]
        )
        return boundary_normal_vectors

    def get_bases_from_tangent_vector_and_normal(self, tangent_vector):
        if not hasattr(self, "normals"):
            self.get_normals()

        # Ensure normals are normalized
        normal_norms = np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        normal_norms[normal_norms == 0] = 1  # Avoid division by zero
        normals = self.normals / normal_norms

        # Normalize tangent vector safely
        tangent_norms = np.linalg.norm(tangent_vector, axis=1)[:, np.newaxis]
        zero_tangent_mask = tangent_norms.flatten() == 0
        tangent_norms[tangent_norms == 0] = 1  # Avoid division by zero
        tangent_vector = tangent_vector / tangent_norms

        # For vertices with zero tangent vector, create arbitrary orthogonal vector
        if np.any(zero_tangent_mask):
            # Create a default tangent vector orthogonal to normal
            default_tangent = np.zeros_like(tangent_vector)
            # Use [1,0,0] unless normal is parallel to x-axis, then use [0,1,0]
            for i in np.where(zero_tangent_mask)[0]:
                n = normals[i]
                if abs(n[0]) < 0.9:  # not parallel to x-axis
                    default_tangent[i] = [1, 0, 0]
                else:  # parallel to x-axis, use y-axis
                    default_tangent[i] = [0, 1, 0]
                # Make it orthogonal to normal
                default_tangent[i] = (
                    default_tangent[i] - np.dot(default_tangent[i], n) * n
                )
                default_tangent[i] /= np.linalg.norm(default_tangent[i])
            tangent_vector[zero_tangent_mask] = default_tangent[zero_tangent_mask]

        # Create orthogonal y-vector using cross product
        y_vector = np.cross(normals, tangent_vector)

        # Normalize y_vector safely
        y_norms = np.linalg.norm(y_vector, axis=1)[:, np.newaxis]
        y_norms[y_norms == 0] = 1  # Avoid division by zero
        y_vector = y_vector / y_norms

        # Recompute tangent vector to ensure orthogonality (Gram-Schmidt)
        tangent_vector = np.cross(y_vector, normals)

        # Normalize the recomputed tangent vector
        tangent_norms = np.linalg.norm(tangent_vector, axis=1)[:, np.newaxis]
        tangent_norms[tangent_norms == 0] = 1  # Avoid division by zero
        tangent_vector = tangent_vector / tangent_norms

        if np.any(np.isnan(tangent_vector)):
            print("ERROR! NaN in tangent vector")
        if np.any(np.isnan(y_vector)):
            print("ERROR! NaN in y vector")
        if np.any(np.isnan(normals)):
            print("ERROR! NaN in normals")

        # Create right-handed coordinate system: [tangent, y, normal]
        local_bases = np.stack([tangent_vector, y_vector, normals], axis=2)
        self.local_bases = local_bases

    def get_local_bases_rotor(self):
        """Return `self.local_bases` as a gafropy.numpy.Rotor batch.

        `local_bases` itself is populated by a diffusion pass, not by this
        class on its own -- raises if nothing has computed it yet.
        """
        if not hasattr(self, "local_bases"):
            raise AttributeError(
                "local_bases is not available yet; run a diffusion pass "
                "(e.g. get_bases_from_tangent_vector_and_normal, or a "
                "PointcloudQuaternionDiffusion/WalkOnSpheresDiffusion pass) first."
            )
        return as_gafropy_rotor(self.local_bases)
