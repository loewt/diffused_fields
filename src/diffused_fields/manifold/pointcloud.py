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


class _PointCloudData:
    """Minimal point-cloud container replacing open3d.geometry.PointCloud."""

    def __init__(self, points=None, colors=None):
        self.points = np.zeros((0, 3)) if points is None else np.asarray(
            points, dtype=np.float64
        )
        self.colors = np.zeros((0, 3)) if colors is None else np.asarray(
            colors, dtype=np.float64
        )

    def get_center(self):
        return self.points.mean(axis=0)

    def scale(self, scale, center):
        self.points = center + scale * (self.points - center)
        return self

    def rotate(self, rotation_matrix, center):
        self.points = (self.points - center) @ rotation_matrix.T + center
        return self

    def translate(self, translation, relative=True):
        translation = np.asarray(translation, dtype=np.float64)
        if relative:
            self.points = self.points + translation
        else:
            self.points = self.points - self.get_center() + translation
        return self

    def voxel_down_sample(self, voxel_size):
        points, colors = geometry_backend.voxel_down_sample(
            self.points, self.colors, voxel_size
        )
        return _PointCloudData(points, colors)

    def select_by_index(self, indices):
        colors = self.colors[indices] if self.colors.size else self.colors
        return _PointCloudData(self.points[indices], colors)

    def cluster_dbscan(self, eps, min_points):
        return geometry_backend.cluster_dbscan(self.points, eps, min_points)

    def estimate_normals(self, knn=20):
        self.normals = geometry_backend.estimate_normals(self.points, knn)
        return self.normals

    def orient_normals_consistent_tangent_plane(self, k=20):
        self.normals = geometry_backend.orient_normals_consistent_tangent_plane(
            self.points, self.normals, k
        )
        return self.normals

    def get_oriented_bounding_box(self):
        return geometry_backend.OrientedBoundingBox.from_points(self.points)


class Pointcloud(Manifold):
    def __init__(
        self,
        vertices=None,
        colors=None,
        filename=None,
        voxel_size=None,
        scale=None,
        translation=None,
        rotation=None,
        normal_orientation=None,
        file_directory=None,
        is_cluster_points=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            type=type(self), scale=scale, translation=translation, rotation=rotation
        )

        # No default directory: pass file_directory= explicitly, or pass a
        # relative/absolute path directly via filename=.
        self.file_directory = file_directory or ""
        self.is_cluster_points = is_cluster_points
        self.voxel_size = voxel_size
        self.normal_orientation = normal_orientation

        # Construct the point cloud
        # ==============================================================================
        object_name = "default"
        if vertices is not None:
            pcd_tmp = _PointCloudData(vertices)
            self.num_vertices_original = vertices.shape[0]
        elif filename is not None:  # initialize from file
            object_name = filename.split(".")[0]
            filepath = self.file_directory + filename
            # print(f"Reading point cloud from {filepath}")
            file_points, file_colors = geometry_backend.read_point_cloud(filepath)
            pcd_tmp = _PointCloudData(file_points, file_colors)
            vertices = file_points
            self.num_vertices_original = vertices.shape[0]
        else:  # initialize with an empty point cloud
            pcd_tmp = _PointCloudData()
            self.num_vertices_original = 0
        self.object_name = object_name
        if colors is not None:
            pcd_tmp.colors = np.asarray(colors, dtype=np.float64)
        self.pcd_tmp = pcd_tmp
        self._set_default_parameters()
        self.transform(pcd_tmp)

    def transform(self, pcd_tmp):
        # Transform the point cloud
        # ==============================================================================

        pcd_tmp.scale(self.scale, center=pcd_tmp.get_center())
        pcd_tmp.rotate(self.rotation.as_matrix(), center=pcd_tmp.get_center())
        pcd_tmp.translate(self.translation, relative=True)

        # Downsample the point cloud
        # ==============================================================================
        if self.voxel_size is None:
            pcd = pcd_tmp
        else:
            pcd = pcd_tmp.voxel_down_sample(voxel_size=self.voxel_size)  # downsample
            num_vertices_downsampled = len(pcd.points)
            # print(
            #     f"Original Point cloud with {self.num_vertices_original}"
            #     + f" points is downsampled with voxel size {self.voxel_size}"
            #     + f"\n resulted in {num_vertices_downsampled} points"
            # )
        self.pcd = pcd

        self.vertices = np.asarray(self.pcd.points)
        if self.is_cluster_points:
            self.cluster_points()
        self.colors = np.asarray(self.pcd.colors)
        if self.colors.size == 0:  # pointcloud have no color
            self.colors = np.zeros((len(self.vertices), 3))

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
        # Find the center point of the point cloud
        self.center_point = np.mean(
            self.vertices, axis=0
        )  # this is not necessarily on the pointcloud

        _, center_vertex = self.kd_tree.query(
            np.array([self.center_point]), k=1
        )  # this is on the pointcloud
        self.center_vertex = center_vertex[0]
        return self.center_point, self.center_vertex

    def get_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.kd_tree = cKDTree(self.vertices)
        return self.kd_tree

    def get_boundary_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.boundary_kd_tree = cKDTree(self.vertices[self.is_boundary_arr])
        return self.boundary_kd_tree

    def get_rotations(self, boundary_points):
        if not hasattr(self, "diffused_rotations"):
            print("Diffused rotations are not available.")
        boundary_vertices = boundary_points[:, 0].astype(
            int
        )  # this the way that I handle in wos for efficiency
        print(boundary_vertices)
        return self.diffused_rotations[boundary_vertices]

    def get_closest_points(self, points):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        distances, indices = self.kd_tree.query(points, k=1)
        return distances, indices

    def get_bounding_box(self, scale=1.0):
        self.oriented_bounding_box = self.pcd.get_oriented_bounding_box()
        self.oriented_bounding_box.scale(
            scale, center=self.oriented_bounding_box.get_center()
        )

        # Get the corner points of the OBB to compute the radius of the enclosing sphere
        self.oriented_bounding_box_corners = np.asarray(
            self.oriented_bounding_box.get_box_points()
        )
        return self.oriented_bounding_box

    def get_bases_from_tangent_vector_and_normal(self, tangent_vector, eps=1e-9):
        """
        Compute orthonormal local bases from tangent vectors and surface normals.

        Uses Gram-Schmidt orthonormalization to ensure proper orthonormal frames:
        1. Normalize tangent (keep tangent direction exact)
        2. Project normal onto plane perpendicular to tangent (Gram-Schmidt)
        3. Normalize normal
        4. Compute y = normal × tangent
        5. Normalize y

        Args:
            tangent_vector: (N, 3) array of tangent directions
            eps: Small epsilon for numerical stability
        """
        if not hasattr(self, "normals"):
            self.get_normals()
        if np.any(np.isnan(tangent_vector)):
            print("ERROR! NaN in vector in diffuse_rotations")

        # Work with copies to avoid modifying original data
        normals = self.normals.astype(float).copy()
        tangent_vector = tangent_vector.astype(float).copy()

        # 1. Normalize tangent (keep tangent direction exact)
        tangent_norm = np.linalg.norm(tangent_vector, axis=1, keepdims=True)
        tangent_norm = np.clip(tangent_norm, eps, None)
        tangent_vector /= tangent_norm

        # 2. Make normal orthogonal to tangent (Gram-Schmidt)
        proj_on_tangent = np.sum(normals * tangent_vector, axis=1, keepdims=True)
        normals = normals - proj_on_tangent * tangent_vector

        # 3. Normalize normal
        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normal_norm = np.clip(normal_norm, eps, None)
        normals /= normal_norm

        # 4. Compute y direction as n × t
        y_vector = np.cross(normals, tangent_vector)

        # 5. Normalize y
        y_norm = np.linalg.norm(y_vector, axis=1, keepdims=True)
        y_norm = np.clip(y_norm, eps, None)
        y_vector /= y_norm

        # 6. Stack into rotation matrices: [t, y, n]
        self.local_bases = np.stack([tangent_vector, y_vector, normals], axis=2)

    def get_local_bases_rotor(self):
        """Return `self.local_bases` as a gafropy.numpy.Rotor batch.

        `local_bases` itself is populated by a diffusion pass (e.g.
        PointcloudQuaternionDiffusion or WalkOnSpheresDiffusion), not by
        this class on its own -- raises if nothing has computed it yet.
        """
        if not hasattr(self, "local_bases"):
            raise AttributeError(
                "local_bases is not available yet; run a diffusion pass "
                "(e.g. get_bases_from_tangent_vector_and_normal, or a "
                "PointcloudQuaternionDiffusion/WalkOnSpheresDiffusion pass) first."
            )
        return as_gafropy_rotor(self.local_bases)

    def get_bounding_sphere(self):
        if not hasattr(self, "center_point"):
            self.get_center()
        if not hasattr(self, "oriented_bounding_box"):
            self.get_bounding_box(scale=1.5)
        sphere_center = self.center_point

        # Compute the distance from the center to the farthest corner (this is the radius
        #  of the enclosing sphere)
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

    def surround_object_with_sphere(self, radius_scalar=2.0):
        if not hasattr(self, "bounding_sphere"):
            self.get_bounding_sphere()

        sphere_pcloud = Pointcloud(
            filename="sphere_1k_uniform.ply",
            voxel_size=None,
            scale=self.bounding_sphere.radius * radius_scalar,
            translation=self.bounding_sphere.center,
        )

        combined_vertices = np.concatenate(
            (self.vertices, sphere_pcloud.vertices), axis=0
        )
        self.combined_pcloud = Pointcloud(vertices=combined_vertices)
        self.combined_pcloud.get_kd_tree()

        self.sphere_pcloud = sphere_pcloud

    def surround_object_with_hemisphere(self, radius_scalar=2.0):
        if not hasattr(self, "bounding_sphere"):
            self.get_bounding_sphere()

        sphere_pcloud = Pointcloud(
            filename="sphere_1k_uniform_top_hemisphere.ply",
            voxel_size=None,
            scale=self.bounding_sphere.radius * radius_scalar,
            translation=self.bounding_sphere.center,
        )

        combined_vertices = np.concatenate(
            (self.vertices, sphere_pcloud.vertices), axis=0
        )
        self.combined_pcloud = Pointcloud(vertices=combined_vertices)

        self.sphere_pcloud = sphere_pcloud

    def get_bounding_box_grid(self, bounding_box_scalar=2.0, nb_points=5):
        # if not hasattr(self, "oriented_bounding_box"):

        self.get_bounding_box(bounding_box_scalar)

        # Find the min and max for x, y, z
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

    def get_signed_distance(self, position):
        distance, point_index = self.kd_tree.query(position, k=1)
        projected_point = self.vertices[point_index]
        projected_normal = self.normals[point_index]

        # if the direction from the point to its projection to the surface
        # is in the same direction with the normal, the distance is positive
        sign = np.sign(np.dot(position - projected_point, projected_normal))
        signed_distance = distance * sign
        return signed_distance, point_index

    def correct_distance_smooth(
        self,
        position,
        distance_target,
        epsilon=1e-3,
        max_iterations=15,
        max_error=1e-1,
        k_neighbors=5,
    ):
        """
        Project position to maintain target distance from surface.

        Improved to prevent oscillation and reduce sensitivity to local variations:
        - Uses k-nearest neighbors to compute smoothed distance and normal
        - Damped correction with oscillation detection
        - Exponentially decreasing gain schedule

        Args:
            epsilon: Convergence tolerance (default: 1e-3, increased from 5e-4 for stability)
            k_neighbors: Number of neighbors to use for smoothing (default: 15, increased from 5)
        """
        position = np.float32(position)
        prev_error = None
        gain = 0.3  # Initial gain (reduced from 0.4 for stability)
        min_gain = 0.05  # Minimum gain to prevent stalling

        for i in range(max_iterations):
            # Query k nearest neighbors for smoothing
            distances, indices = self.kd_tree.query(np.array([position]), k=k_neighbors)
            distances = distances[0]
            indices = indices[0]

            # Compute signed distances to all k neighbors
            signed_distances = []
            normals = []
            for j, idx in enumerate(indices):
                # Get closest point on surface
                closest_point = self.vertices[idx]
                normal = self.normals[idx]

                # Compute signed distance
                vec_to_position = position - closest_point
                distance = np.linalg.norm(vec_to_position)
                sign = np.sign(np.dot(vec_to_position, normal))
                signed_distances.append(distance * sign)
                normals.append(normal)

            # Weighted average: closer points have more influence
            weights = 1.0 / (distances + 1e-6)  # Inverse distance weighting
            weights = weights / np.sum(weights)  # Normalize

            # Smoothed signed distance and normal
            signed_distance = np.sum(np.array(signed_distances) * weights)
            smoothed_normal = np.sum(np.array(normals) * weights[:, np.newaxis], axis=0)
            smoothed_normal = smoothed_normal / np.linalg.norm(smoothed_normal)

            # Use closest point index for return value
            point_index = indices[0]

            error = distance_target - signed_distance

            # Check convergence
            if np.abs(error) < epsilon:
                return position, signed_distance, point_index

            # Clamp large errors
            if np.abs(error) > max_error:
                error = np.sign(error) * max_error

            # Detect oscillation: if error sign flips, we're oscillating
            if prev_error is not None and np.sign(error) != np.sign(prev_error):
                gain *= 0.5  # Halve gain when oscillation detected
                gain = max(gain, min_gain)  # Don't go below minimum

            # Exponential decay of gain over iterations
            iteration_factor = 0.95**i

            # Use smoothed normal for correction direction
            effective_gain = gain * iteration_factor
            correction = error * effective_gain * smoothed_normal
            position += correction

            prev_error = error

        # print(f"Could not correct the distance after {max_iterations} iterations")
        return position, signed_distance, point_index

    def correct_distance_smooth_old(
        self, position, distance_target, epsilon=5e-4, max_iterations=15, max_error=1e-1
    ):
        """
        Project position to maintain target distance from surface.
        Original simple version without k-neighbor smoothing (kept for reference).
        """
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

    def cluster_points(self, distance_threshold=1e-2, min_points=10):
        # Apply DBSCAN clustering (eps is the distance threshold, min_points is the
        # minimum number of points in a cluster)
        labels = np.array(
            self.pcd.cluster_dbscan(eps=distance_threshold, min_points=min_points)
        )

        # Number of clusters (ignoring noise points, labeled as -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # print(f"Clustering points...{num_clusters}")
        if num_clusters > 0:
            unique_labels, num_points = np.unique(labels, return_counts=True)
            desired_label = unique_labels[np.argmax(num_points)]
            # print(f"Number of clusters: {num_clusters}")
            # print(
            #     f"Cluster {desired_label} has {num_points[np.argmax(num_points)]} points"
            # )
            if desired_label not in labels:
                pass  # print(f"Cluster {desired_label} not found in the point cloud.")
            else:
                # Filter points corresponding to cluster 1
                indices_cluster = np.where(labels == desired_label)[0]

                # Select points corresponding to cluster 1
                self.pcd = self.pcd.select_by_index(indices_cluster)
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)

    def get_normals(self, num_neighbors=20):
        if not hasattr(self, "center_vertex"):
            self.get_center()
        self.pcd.estimate_normals(knn=num_neighbors)
        # force all the normals point to the same direction: outwards from the object

        self.pcd.orient_normals_consistent_tangent_plane(k=num_neighbors)

        normals = np.asarray(self.pcd.normals)
        # outward = self.center_vertex - self.center_point
        # sign = np.sign(np.dot(outward, normals[self.center_vertex]))
        # normals *= sign
        self.normals = -normals * self.normal_orientation
        return self.normals

    def get_local_basis(self):
        if not hasattr(self, "normals"):
            self.get_normals()
        # set u vector as a vector orthogonal to the normal
        self.tangent_vectors_u = np.column_stack(
            [self.normals[:, 1], -self.normals[:, 0], np.zeros(len(self.vertices))]
        )
        # ordering matters for the right handedness
        self.tangent_vectors_v = np.cross(self.normals, self.tangent_vectors_u)

    def get_k_edges(self, num_neighbors=3):
        """
        Calculate the edges between points based on the KD-tree.

        Parameters:
        -----------
        metric: numpy.ndarray
            The metric to calculate the edges on.
        num_neighbours: int
            The number of nearest neighbors to consider.

        Returns:
        --------
        numpy.ndarray
            The edges between points.
        """
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        # Query the num_neighbor neighbourhoods for each point of the selected feature
        # space to each point
        self.d_kdtree, idx = self.kd_tree.query(self.vertices, k=num_neighbors)

        # Remove the first point in the neighborhood as this is just the
        # queried point itself
        idx = idx[:, 1:]

        # Create the edges array between all the points and their closest
        # neighbours
        point_numbers = np.arange(len(self.vertices))
        # Repeat each point in the point numbers array the number of closest
        # neighbours -> 1,2,3,4... becomes 1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4...
        point_numbers = np.repeat(point_numbers, num_neighbors - 1)
        # Flatten  the neighbour indices array -> from [1,3,10,14], [4,7,17,23]
        # , ... becomes [1,3,10,4,7,17,23,...]
        idx_flatten = idx.flatten()
        # Create the edges array by combining the two other ones as a vertical
        # stack and transposing them to get the input that LineSet requires
        edges = np.vstack((point_numbers, idx_flatten)).T

        return edges

    def get_mean_edge_length(self):
        """
        Calculate the mean edge length of a point cloud.

        Parameters:
        -----------
        vertices : numpy.ndarray
            The vertices of the point cloud.

        Returns:
        --------
        float
            The mean edge length of the point cloud.
        """
        # if self.voxel_size is None:
        #     edges = self.get_k_edges()
        #     edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        #     edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        #     self.mean_edge_length = np.mean(edge_lengths)
        # else:
        #     self.mean_edge_length = self.voxel_size

        edges = self.get_k_edges()
        edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        self.mean_edge_length = np.mean(edge_lengths)

        # print(f"Mean edge length: {self.mean_edge_length*1e3:.1f} mm")
        return self.mean_edge_length

    def get_boundary(
        self,
        max_neighbors=30,
        angle_threshold=np.pi / 2,
    ):
        def is_boundary(
            vertex,
            neighbor_vertices,
            tangent_vector_u,
            tangent_vector_v,
            angle_threshold=3 * np.pi / 2,
        ):
            """
            Given a point and its neighbors, this function checks if the point is on
            the boundary
            Inspired by the implementation of the boundaryEstimation() function of PCL
            https://pointclouds.org/documentation/boundary_8hpp_source.html
            """
            # Compute the angles between the point and its neighbors
            # in the tangent plane
            angles = np.zeros(len(neighbor_vertices))
            for j in range(len(neighbor_vertices)):
                neighbor = neighbor_vertices[j]
                delta = neighbor - vertex
                angles[j] = np.arctan2(
                    np.dot(delta, tangent_vector_u), np.dot(delta, tangent_vector_v)
                )

            # Sort the angles and get the maximum difference
            angles = np.sort(angles)
            diff = np.zeros_like(angles)
            diff = np.diff(angles)
            # Get the angle difference between the last and the first
            diff[-1] = 2 * np.pi - angles[-1] + angles[0]
            # Check the angle condition for boundary
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
            # exclude the current vertex from the search
            # [_, idx, _] = self.kd_tree.search_knn_vector_3d(vertex, max_neighbors)
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
        """
        Computes normal vectors at the boundary vertices that point inside the point
        cloud.

        Returns:
        - boundary_vertices: np.ndarray, the coordinates of the boundary vertices.
        - boundary_normal_vectors: np.ndarray, the normal vectors at boundary points
        pointing inward.
        """
        if not hasattr(self, "is_boundary_arr"):
            self.get_boundary()
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()

        # Step 1: Get boundary vertices
        boundary_vertices = self.vertices[self.is_boundary_arr]

        # Step 2: Compute normal vectors at the boundary points
        boundary_normal_vectors = np.zeros_like(boundary_vertices)

        for i, vertex in enumerate(boundary_vertices):
            # Query all neighbors (not just boundary vertices) for normal estimation
            distances, indices = self.kd_tree.query(vertex, k=10)
            neighbor_vertices = self.vertices[indices[1:]]  # Exclude the vertex itself

            # Step 3: Compute inward-pointing vector by averaging direction to
            # neighboring points
            inward_direction = np.mean(neighbor_vertices - vertex, axis=0)
            inward_direction /= np.linalg.norm(inward_direction)  # Normalize

            # Optionally, project onto tangent plane if you want the vector in the plane
            # of the boundary stripe
            tangent_vector_u = self.tangent_vectors_u[self.is_boundary_arr][i]
            tangent_vector_v = self.tangent_vectors_v[self.is_boundary_arr][i]

            # Project the inward direction onto the tangent vectors
            projected_u = np.dot(inward_direction, tangent_vector_u) * tangent_vector_u
            projected_v = np.dot(inward_direction, tangent_vector_v) * tangent_vector_v
            tangent_projection = projected_u + projected_v

            tangent_projection /= np.linalg.norm(
                tangent_projection
            )  # Normalize the projection

            # Store the normal vector (the inward direction)
            boundary_normal_vectors[i] = tangent_projection
        self.boundary_normals = boundary_normal_vectors
        self.boundary_tangents = np.cross(
            boundary_normal_vectors, self.normals[self.is_boundary_arr]
        )

        return boundary_normal_vectors

    def add_gaussian_noise(self, noise_std=0.002, seed=None):
        """
        Add Gaussian noise to the point cloud vertices.

        Args:
            noise_std (float): Standard deviation of the noise
            seed (int, optional): Random seed for reproducibility

        Returns:
            np.ndarray: The original vertices before noise was added
        """
        if seed is not None:
            np.random.seed(seed)

        # Store original vertices for potential restoration
        original_vertices = np.copy(self.vertices)

        # Generate and apply noise
        noise = np.random.normal(scale=noise_std, size=self.vertices.shape)
        self.vertices = self.vertices + noise

        # Update the point cloud
        self.pcd.points = np.asarray(self.vertices, dtype=np.float64)

        # Recalculate normals since vertices changed
        self.get_normals()

        return original_vertices

    def apply_scaling(self, scale_factors):
        """
        Apply anisotropic scaling to the point cloud.

        Args:
            scale_factors (array-like): Scaling factors for [x, y, z] axes

        Returns:
            np.ndarray: The original vertices before scaling was applied
        """
        # Store original vertices
        original_vertices = np.copy(self.vertices)

        # Apply scaling
        scale_factors = np.array(scale_factors)
        self.vertices = self.vertices * scale_factors

        # Note: Normal recalculation and spatial structures (KD-tree) are invalidated.
        # Call get_normals() and get_kd_tree() manually after all deformations.

        return original_vertices

    def create_holes(
        self, num_holes=10, hole_radius=0.005, seed=None, preserve_keypoints=None
    ):
        """
        Create holes in the point cloud by removing points within circular regions.

        Args:
            num_holes (int): Number of holes to create
            hole_radius (float): Radius of each hole
            seed (int, optional): Random seed for reproducibility
            preserve_keypoints (list, optional): Vertex indices to preserve from removal

        Returns:
            tuple: (original_vertices, hole_centers, removed_indices)
        """
        if seed is not None:
            np.random.seed(seed)

        # Store original state
        original_vertices = np.copy(self.vertices)
        points = np.asarray(self.vertices)

        # Select random hole centers from the point cloud
        hole_centers = points[
            np.random.choice(len(points), size=num_holes, replace=False)
        ]

        # Build a mask to keep points outside all holes
        keep_mask = np.ones(len(points), dtype=bool)

        # Create holes
        for center in hole_centers:
            distances = np.linalg.norm(points - center, axis=1)
            hole_mask = distances > hole_radius

            # If we have keypoints to preserve, don't remove them
            if preserve_keypoints is not None:
                for keypoint_idx in preserve_keypoints:
                    if keypoint_idx < len(hole_mask):
                        hole_mask[keypoint_idx] = True

            keep_mask &= hole_mask

        # Get indices of removed points
        removed_indices = np.where(~keep_mask)[0]

        # Apply the mask to keep only points outside holes
        reduced_points = points[keep_mask]

        # Update the point cloud
        self.vertices = reduced_points
        self.pcd.points = np.asarray(reduced_points, dtype=np.float64)

        # Recalculate connectivity and normals since vertices changed
        self.get_normals()

        # Recalculate clustering since topology changed
        if hasattr(self, "is_boundary_arr"):
            self.cluster_points()

        return original_vertices, hole_centers, removed_indices

    def update_keypoints_after_holes(self, original_keypoint_positions):
        """
        Update keypoint indices after holes have been created by finding
        closest points to the original keypoint positions.

        Args:
            original_keypoint_positions (np.ndarray): Original 3D positions of keypoints

        Returns:
            tuple: (new_source_vertices, new_start_vertex) - updated indices
        """
        # Find closest points to original keypoints in the modified point cloud
        _, new_indices = self.get_closest_points(original_keypoint_positions)

        if len(original_keypoint_positions) == 3:
            # Assume format: [start_position, source1_position, source2_position]
            return [new_indices[1], new_indices[2]], new_indices[0]
        elif len(original_keypoint_positions) == 2:
            # Assume format: [source1_position, source2_position]
            return new_indices, None
        else:
            return new_indices, None

    def apply_bend(self, bend_axis=2, curvature=0.01):
        """
        Bend the point cloud along the specified axis.

        Args:
            bend_axis (int): Axis along which to apply the bend (0=x, 1=y, 2=z)
            curvature (float): Amount of curvature, positive or negative
        """
        points = np.asarray(self.vertices)
        bent = points.copy()
        other_axes = [i for i in range(3) if i != bend_axis]

        coord = bent[:, bend_axis]
        for ax in other_axes:
            bent[:, ax] += curvature * coord**2  # quadratic bend

        # Update vertices
        self.vertices = bent

        # Note: Normal recalculation and spatial structures (KD-tree) are invalidated.
        # Call get_normals() and get_kd_tree() manually after all deformations.

    def apply_bulge(self, amount=0.05, seed=None):
        """
        Apply random bulging deformation to the point cloud.

        Args:
            amount (float): Maximum amount of bulging displacement
            seed (int, optional): Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)

        points = np.asarray(self.vertices)
        center = points.mean(axis=0)
        directions = points - center
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        directions /= norm
        r = np.random.uniform(-amount, amount, size=(points.shape[0], 1))

        bulged = points + directions * r

        # Update vertices
        self.vertices = bulged

        # Update point cloud
        if hasattr(self, "pcd") and self.pcd is not None:
            self.pcd.points = np.asarray(bulged, dtype=np.float64)
            # Recalculate normals after deformation
            self.pcd.estimate_normals()
            self.normals = np.asarray(self.pcd.normals)

    def apply_twist(self, axis=2, twist_strength=2.0):
        """
        Apply twisting deformation around the specified axis.

        Args:
            axis (int): Axis around which to twist (0=x, 1=y, 2=z)
            twist_strength (float): Strength of the twist
        """
        points = np.asarray(self.vertices)
        twisted = points.copy()

        # Get coordinate along twist axis
        coord = twisted[:, axis]
        angle = twist_strength * coord  # twist amount depends on coordinate

        # Apply rotation in the plane perpendicular to twist axis
        if axis == 0:  # Twist around x-axis (rotate in yz plane)
            y, z = twisted[:, 1], twisted[:, 2]
            twisted[:, 1] = y * np.cos(angle) - z * np.sin(angle)
            twisted[:, 2] = y * np.sin(angle) + z * np.cos(angle)
        elif axis == 1:  # Twist around y-axis (rotate in xz plane)
            x, z = twisted[:, 0], twisted[:, 2]
            twisted[:, 0] = x * np.cos(angle) - z * np.sin(angle)
            twisted[:, 2] = x * np.sin(angle) + z * np.cos(angle)
        else:  # axis == 2, twist around z-axis (rotate in xy plane)
            x, y = twisted[:, 0], twisted[:, 1]
            twisted[:, 0] = x * np.cos(angle) - y * np.sin(angle)
            twisted[:, 1] = x * np.sin(angle) + y * np.cos(angle)

        # Update vertices
        self.vertices = twisted

        # Note: Normal recalculation and spatial structures (KD-tree) are invalidated.
        # Call get_normals() and get_kd_tree() manually after all deformations.
