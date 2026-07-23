"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import time

import gafropy.numpy as gn
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ..manifold._rotor_interop import as_gafropy_rotor, as_scipy_rotation
from ..manifold.mesh import Mesh
from ..manifold.pointcloud import Pointcloud


def compute_mean_rotor(rotors, weights=None):
    """
    Compute the mean rotor from a batch of rotor samples. Based on the algorithm
    described in: Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y.
    (2006). Averaging Quaternions. Journal of Guidance, Control, and Dynamics,
    29(4), 1193-1197. https://doi.org/9.2514/1.28949

    This function calculates the mean rotor using a weighted average approach. If
    no weights are provided, it assumes equal weighting for all rotors. The
    averaging itself operates on the rotors' quaternion coefficients -- gafro has
    no native averaging primitive, so this is Markley's eigen-decomposition
    method, unchanged from before, just reading from and returning a
    gafropy.numpy.Rotor instead of a raw quaternion array.

    Parameters: rotors: A batch of shape (N,) -- a gafropy.numpy.Rotor, a
    gafropy.Rotor, a scipy Rotation, or an (N, 3, 3) / (N, 4) matrix/quaternion
    array (anything accepted by diffused_fields.manifold.as_gafropy_rotor).
    weights (array-like, optional): An array of shape (N,) representing the weights for
    each rotor.
                                    If not provided, equal weights are assumed.

    Returns: gafropy.numpy.Rotor: A shape-() rotor representing the mean.

    Raises:
    ValueError: If the number of weights does not match the number of rotors.

    Notes:
    - The mean rotor is normalized before being returned, with its scalar part
    ensured to be non-negative.
    """
    quaternions = as_gafropy_rotor(rotors).to_quaternion()
    num_quaternions = quaternions.shape[0]

    # Ensure quaternions are normalized
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]

    # Use equal weights if none are provided
    if weights is None:
        weights = np.ones(num_quaternions)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != num_quaternions:
            raise ValueError("Number of weights must match number of rotors")

    # Build the symmetric accumulator matrix (weighted sum of outer products)
    A = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Extract the eigenvector corresponding to the largest eigenvalue
    max_index = np.argmax(eigenvalues)
    mean_quaternion = eigenvectors[:, max_index]

    # Normalize the mean quaternion
    mean_quaternion /= np.linalg.norm(mean_quaternion)

    # Ensure the scalar part is non-negative
    if mean_quaternion[3] < 0:
        mean_quaternion *= -1

    # mean_quaternion is [x, y, z, w] (matches Rotor.to_quaternion()'s
    # convention, since that's what built `quaternions` above); gafropy's
    # own from_quaternion takes [w, x, y, z] instead, so reorder here.
    return gn.Rotor.from_quaternion(mean_quaternion[[3, 0, 1, 2]])


class WalkOnSpheresDiffusion(object):
    def __init__(
        self,
        boundaries=[],
        batch_size=512,
        max_iterations=24,
        primitive=False,
        convergence_threshold=1e-3,
        divergence_threshold=10,
    ):
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.primitive = primitive
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold
        self.boundaries = boundaries

    def batch_iteration(self, points, status_arr):
        """
        Perform a batch iteration of the Walk on Spheres algorithm
        for points still walking.

        Args:
            points (numpy.ndarray): Array of batch points.
            status_arr (numpy.ndarray): Array to store the status of each point.

        Returns:
            tuple: A tuple containing the updated batch points, boundary values,
            and status array.
        """
        # Identify the points that are still walking
        walking_indices = np.where(status_arr == "walking")[0]  # Get the actual indices

        # Filter the points that are still walking
        walking_points = points[walking_indices]

        for i, boundary in enumerate(self.boundaries):
            # Get distances and indices for the subset of points
            distances, indices = boundary.get_closest_points(walking_points)

            # Update the min_distances array for the walking points only where the new
            # distances are smaller
            update_mask = distances < self.min_distances[walking_indices]
            to_update = walking_indices[update_mask]
            self.min_distances[to_update] = distances[update_mask]

            # Update the min_indices for the walking points where
            # the distances are smaller
            if isinstance(
                boundary, (Pointcloud, Mesh)
            ):  # for pointcloud/mesh indices are 1D
                self.min_indices[to_update, 0] = indices[update_mask]
            else:  # for primitives indices are 3D points
                self.min_indices[to_update] = indices[update_mask, :]

            # Update the boundary index if the minimum distance is attained
            self.min_boundary[to_update] = i

        # Check for points that have converged or diverged
        converged = self.min_distances < self.convergence_threshold
        status_arr[converged] = "converged"

        diverged = self.min_distances > self.divergence_threshold
        status_arr[diverged] = "diverged"

        # Sample random direction for the points still walking
        direction = batch_sample_random_direction(len(points))
        points[walking_indices] += (
            self.min_distances[walking_indices, None] * direction[walking_indices]
        )

        return points, self.min_indices, status_arr

    def batch_solve(self, points):
        # Initialize status array and boundary vertex array
        status_arr = np.full(points.shape[0], "walking", dtype=object)
        boundary_vertex_arr = np.zeros(len(points), dtype=int)

        # Initialize arrays for min_distances, min_indices, and min_iteration
        self.min_distances = np.full((points.shape[0],), np.inf)
        # self.min_indices = np.full((points.shape[0], 3), -1)
        self.min_indices = np.zeros((points.shape[0], 3))
        self.min_boundary = np.full((points.shape[0],), -1)

        # Main loop
        for i in range(1, self.max_iterations):
            # Perform batch iteration only for points that are still walking
            points, boundary_vertex_arr, status_arr = self.batch_iteration(
                points, status_arr
            )

            # Stop if all points have converged or terminated
            if np.all(status_arr != "walking"):
                break

        # Mark points that are still walking after max_iterations as "diverged"
        status_arr[status_arr == "walking"] = "diverged"

        return boundary_vertex_arr, status_arr

    def diffuse_vectors(self, points, vector_type="normal"):
        """
        Diffuse vectors using walk-on-spheres method.

        Args:
            points: Query points for diffusion
            vector_type: Type of vector to diffuse
                - "normal": Use boundary.normals
                - "tangent_x": Use boundary.local_bases[:, :, 0] (first tangent)
                - "tangent_y": Use boundary.local_bases[:, :, 1] (second tangent)
                - "tangent_z": Use boundary.local_bases[:, :, 2] (normal/third axis)

        Returns:
            diffused_vector: Average diffused vector
            boundary_vertex_arr: Indices of converged boundary vertices
            status_arr: Status of each walk
        """
        # get where we landed on the point cloud
        boundary_vertex_arr, status_arr = self.batch_solve(points)
        converged_indices = status_arr == "converged"

        all_vectors = []
        for i, boundary in enumerate(self.boundaries):
            current_boundary_points = self.min_boundary == i
            boundary_indices = np.logical_and(
                converged_indices, current_boundary_points
            )
            if isinstance(
                boundary, (Pointcloud, Mesh)
            ):  # for pointcloud/mesh indices are 1D
                converged_boundary_indices = self.min_indices[
                    boundary_indices, 0
                ].astype(int)
            else:  # for primitives indices are 3D points
                converged_boundary_indices = self.min_indices[boundary_indices, :]
            if len(converged_boundary_indices) == 0:
                continue

            # Extract the requested vector type
            if vector_type == "normal":
                vectors = boundary.normals[converged_boundary_indices]
            elif vector_type == "tangent_x":
                vectors = boundary.local_bases[converged_boundary_indices, :, 0]
            elif vector_type == "tangent_y":
                vectors = boundary.local_bases[converged_boundary_indices, :, 1]
            elif vector_type == "tangent_z":
                vectors = boundary.local_bases[converged_boundary_indices, :, 2]
            else:
                raise ValueError(
                    f"Unknown vector_type: {vector_type}. Use 'normal', 'tangent_x', 'tangent_y', or 'tangent_z'"
                )

            all_vectors.append(vectors)
        all_vectors = np.concatenate(all_vectors, axis=0)

        diffused_vector = np.mean(all_vectors, axis=0)
        # Normalize the diffused vector to unit length
        norm = np.linalg.norm(diffused_vector)
        if norm > 1e-10:  # Avoid division by zero
            diffused_vector = diffused_vector / norm
        return (
            diffused_vector,
            boundary_vertex_arr[status_arr == "converged"],
            status_arr,
        )

    def diffuse_normal_vectors(self, points):
        """Legacy wrapper for backward compatibility."""
        return self.diffuse_vectors(points, vector_type="normal")

    def diffuse_tangent_vectors(self, points):
        """Legacy wrapper for backward compatibility."""
        return self.diffuse_vectors(points, vector_type="tangent_x")

    def diffuse_scalars(self, points):
        """
        Diffuse scalar values using walk-on-spheres method.

        Args:
            points: Query points for diffusion

        Returns:
            diffused_scalar: Average scalar value from boundary
            boundary_vertex_arr: Indices of converged boundary vertices
            status_arr: Status of each walk
        """
        # get where we landed on the boundaries
        boundary_vertex_arr, status_arr = self.batch_solve(points)
        converged_indices = status_arr == "converged"

        all_scalars = []
        for i, boundary in enumerate(self.boundaries):
            current_boundary_points = self.min_boundary == i
            boundary_indices = np.logical_and(
                converged_indices, current_boundary_points
            )
            if isinstance(
                boundary, (Pointcloud, Mesh)
            ):  # for pointcloud/mesh indices are 1D
                converged_boundary_indices = self.min_indices[
                    boundary_indices, 0
                ].astype(int)
            else:  # for primitives indices are 3D points
                converged_boundary_indices = self.min_indices[boundary_indices, :]
            if len(converged_boundary_indices) == 0:
                continue

            # Extract scalar values from the boundary
            if hasattr(boundary, "u0"):
                scalars = boundary.u0[converged_boundary_indices]
            else:
                # If no scalar values defined, use zeros
                scalars = np.zeros(len(converged_boundary_indices))

            all_scalars.append(scalars)

        if len(all_scalars) > 0:
            all_scalars = np.concatenate(all_scalars, axis=0)
            diffused_scalar = np.mean(all_scalars)
        else:
            diffused_scalar = 0.0

        return (
            diffused_scalar,
            boundary_vertex_arr[status_arr == "converged"],
            status_arr,
        )

    def _diffuse_mean_rotor(self, points):
        """
        Core of orientation diffusion: run WoS, collect the local basis each
        converged walker landed on as a rotor, and average them.

        Shared by `diffuse_rotor` (gafro-native output) and `diffuse_rotations`
        (matrix output, kept for backward compatibility) so the boundary
        bookkeeping isn't duplicated between them.

        Returns (mean_rotor, converged_boundary_vertices, status_arr), where
        mean_rotor is a shape-() gafropy.numpy.Rotor.
        """
        # get where we landed on the point cloud
        boundary_vertex_arr, status_arr = self.batch_solve(points)

        # grab the value at the pcloud
        converged_indices = status_arr == "converged"
        all_rotor_coeffs = []
        for i, boundary in enumerate(self.boundaries):
            # points that we landed on the i-th boundary
            current_boundary_points = self.min_boundary == i
            # ensure we converged to these points and not prematurely finished there
            boundary_indices = np.logical_and(
                converged_indices, current_boundary_points
            )
            if isinstance(
                boundary, (Pointcloud, Mesh)
            ):  # for pointcloud/mesh indices are 1D
                converged_boundary_indices = self.min_indices[boundary_indices, 0]
            else:  # for primitives indices are 3D points
                converged_boundary_indices = self.min_indices[boundary_indices, :]
            if len(converged_boundary_indices) == 0:
                # print("No converged indices for boundary", i)
                continue

            if isinstance(
                boundary, (Pointcloud, Mesh)
            ):  # for pointcloud/mesh indices are 1D
                converged_boundary_indices = converged_boundary_indices.astype(int)
                local_bases = boundary.local_bases[converged_boundary_indices]
            else:  # for primitives indices are 3D points
                local_bases = boundary.get_local_bases(converged_boundary_indices)
            # gafro's Rotor.from_matrix is scalar-only (no batched numpy
            # binding yet), so the matrix -> rotor hop for a whole batch of
            # local bases still goes through scipy's vectorized conversion.
            all_rotor_coeffs.append(as_gafropy_rotor(local_bases).coeffs)

        all_rotors = gn.Rotor(np.concatenate(all_rotor_coeffs, axis=0))
        mean_rotor = compute_mean_rotor(all_rotors)

        return mean_rotor, boundary_vertex_arr[status_arr == "converged"], status_arr

    def diffuse_rotor(self, points):
        """
        Gafro-native counterpart of `diffuse_rotations`: same computation,
        returns a gafropy.numpy.Rotor (shape ()) instead of a rotation matrix.
        """
        return self._diffuse_mean_rotor(points)

    def diffuse_rotations(self, points):
        mean_rotor, boundary_vertices, status_arr = self._diffuse_mean_rotor(points)
        local_basis = as_scipy_rotation(mean_rotor).as_matrix()
        return local_basis, boundary_vertices, status_arr

    # def solve_vector_diffusion_on_grid(self, grid):
    #     # Estimate the value of the Laplace's equation for all vertices of the grid
    #     # using the Walk on Spheres method
    #     # =============================================================================
    #     grid_normals = np.zeros((len(grid.vertices), 3))
    #     for vertex in range(len(grid.vertices)):
    #         # Get the batch of points for parallel computation of the diffusion
    #         batch_points = self.get_batch_from_point(grid.vertices[vertex])
    #         grid_normals[vertex, ...], _, _ = self.diffuse_normal_vectors(batch_points)
    #         if vertex % 100 == 0:  # print progress
    #             print(f"Processing vertex {vertex} out of {len(grid.vertices)}")

    def get_batch_from_point(self, x):
        # every sample starts on the current poisition
        return np.tile(x, (self.batch_size, 1))

    def debug_walk_on_spheres(status_arr):
        nb_elements = status_arr.size

        print(
            f""" converged in {np.sum(status_arr=="converged")/nb_elements*100:.2f}"""
            + f""" converged survivor in {np.sum(
                status_arr=="converged_survivor")/nb_elements*100:.2f}"""
            # + f" terminated in {np.sum(s"terminated")/nb_elements*100:.2f}"
            # + f" diverged in {np.sum("diverged")/nb_elements*100:.2f}"
            # + f" walking in {np.sum("walking")/nb_elements*100:.2f}"
            # + f" survivor in {np.sum("survivor")/nb_elements*100:.2f}"
        )

    def trajectory_rollout(
        self,
        initial_position,
        steps,
        distance_target=0.0,
        step_size=0.03,
        project=False,
        axis_sequence=None,
        direction_mappings=None,
    ):
        """
        Roll out a trajectory following a sequence of local frame directions.

        Args:
            initial_position: Starting 3D position
            steps: Number of steps per axis direction
            distance_target: Target distance (unused, kept for compatibility)
            step_size: Step size for each movement
            project: Whether to project onto surface
            axis_sequence: List of axis directions to follow (e.g., ["+z", "+x", "+y"])
            direction_mappings: Dict mapping axis strings to [axis_index, sign] pairs

        Returns:
            positions: Array of 3D positions along trajectory
            local_bases: Array of local coordinate frames along trajectory
        """
        # Default axis sequence if not provided
        if axis_sequence is None:
            axis_sequence = ["+z", "+x", "+y"]

        # Default direction mappings if not provided
        if direction_mappings is None:
            direction_mappings = {
                "+x": [0, 1],  # x-axis, positive direction
                "-x": [0, -1],  # x-axis, negative direction
                "+y": [1, 1],  # y-axis, positive direction
                "-y": [1, -1],  # y-axis, negative direction
                "+z": [2, 1],  # z-axis, positive direction
                "-z": [2, -1],  # z-axis, negative direction
            }

        batch_points = self.get_batch_from_point(initial_position)
        initial_basis, _, _ = self.diffuse_rotations(batch_points)
        initial_rotation = R.from_matrix(initial_basis)

        positions = [initial_position]
        local_bases = [initial_basis]

        position = initial_position
        local_basis = initial_rotation.as_matrix()

        for axis in axis_sequence:
            # Validate axis is in direction mappings
            if axis not in direction_mappings:
                raise ValueError(
                    f"Unknown axis '{axis}'. Available axes: {list(direction_mappings.keys())}"
                )

            for i in range(steps):
                # Get the sign and direction of the axis
                direction_info = direction_mappings[axis]
                axis_index = direction_info[0]  # Which axis (0=x, 1=y, 2=z)
                sign = direction_info[1]  # Direction (+1 or -1)

                # Move in the direction of the axis
                next_position = position + sign * step_size * local_basis[:, axis_index]

                if project:
                    _, indices = self.boundaries[0].get_closest_points(next_position)
                    next_local_basis = self.boundaries[0].local_bases[indices, ...]
                else:
                    batch_points = self.get_batch_from_point(next_position)
                    next_local_basis, _, _ = self.diffuse_rotations(batch_points)

                next_local_rotation = R.from_matrix(next_local_basis)
                position = next_position
                local_basis = next_local_rotation.as_matrix()

                positions.append(position)
                local_bases.append(local_basis)

        positions = np.array(positions)
        local_bases = np.array(local_bases)

        return positions, local_bases

    # def trajectory_rollout(trajectory_steps = 10):

    def diffuse_orientations_on_grid(self, grid):

        # Estimate the value of the Laplace's equation for all vertices of the grid
        # using the Walk on Spheres method
        # =============================================================================
        grid.local_bases = np.zeros((len(grid.vertices), 3, 3))

        start_time = time.time()
        for vertex in tqdm(
            range(len(grid.vertices)), desc="Diffusing orientations", unit="vertex"
        ):
            # Get the batch of points for parallel computation of the diffusion
            batch_points = self.get_batch_from_point(grid.vertices[vertex])
            grid.local_bases[vertex, :, :], boundary_vertex_arr, status_arr = (
                self.diffuse_rotations(batch_points)
            )
        print(
            f"Process finished --- {(time.time() - start_time)/len(grid.vertices)} seconds ---"
        )

    def diffuse_scalars_on_grid(self, grid):
        """
        Estimate scalar values on grid vertices using Walk-on-Spheres method.

        Args:
            grid: Grid object with vertices attribute

        Returns:
            Array of scalar values at grid vertices
        """
        grid.scalars = np.zeros(len(grid.vertices))
        start_time = time.time()
        for vertex in tqdm(
            range(len(grid.vertices)), desc="Diffusing scalars", unit="vertex"
        ):
            # Get the batch of points for parallel computation of the diffusion
            batch_points = self.get_batch_from_point(grid.vertices[vertex])
            grid.scalars[vertex], boundary_vertex_arr, status_arr = (
                self.diffuse_scalars(batch_points)
            )
        print(
            f"Process finished --- {(time.time() - start_time)/len(grid.vertices)} seconds ---"
        )
        return grid.scalars

    def project_orientations_on_grid(self, grid):

        # Estimate the value of the Laplace's equation for all vertices of the grid
        # using the Walk on Spheres method
        # =============================================================================
        grid.local_bases = np.zeros((len(grid.vertices), 3, 3))

        start_time = time.time()
        for vertex in range(len(grid.vertices)):
            print(vertex)
            _, projected_vertex = self.boundaries[0].get_closest_points(
                grid.vertices[vertex]
            )
            # Get the batch of points for parallel computation of the diffusion
            grid.local_bases[vertex, :, :] = self.boundaries[0].local_bases[
                projected_vertex, ...
            ]

    def local_step(self, x, direction, sign, step_size):
        # Get batch points for diffusion
        batch_points = self.get_batch_from_point(x)

        # Compute local bases using diffusion
        local_basis, _, _ = self.diffuse_rotations(batch_points)

        # Update the next position
        next_x = x + (local_basis[:, direction] * step_size * sign)
        return next_x, local_basis

    def move_multistep(
        self,
        num_steps,
        x0,
        direction,
        sign,
        step_size=0.001,
        distance_to_surface=0.001,
        project=False,
        terminal_condition=None,
    ):
        x_next = x0
        trajectory = []
        trajectory_bases = []
        for _ in range(num_steps):
            x_next, local_basis = self.local_step(x_next, direction, sign, step_size)
            if project:
                x_next, _, _ = self.boundaries[0].correct_distance_smooth(
                    x_next, distance_to_surface
                )
            if terminal_condition is not None:
                if terminal_condition(x_next, local_basis):
                    break

            trajectory.append(x_next)
            trajectory_bases.append(local_basis)
        trajectory = np.asarray(trajectory)
        trajectory_bases = np.asarray(trajectory_bases)
        return trajectory, trajectory_bases

    def diffuse_vectors_on_grid(self, grid):
        # Estimate the value of the Laplace's equation for all vertices of the grid
        # using the Walk on Spheres method
        # =============================================================================
        grid.vectors = np.zeros((len(grid.vertices), 3))
        start_time = time.time()
        for vertex in tqdm(
            range(len(grid.vertices)), desc="Diffusing vectors", unit="vertex"
        ):
            # Get the batch of points for parallel computation of the diffusion
            batch_points = self.get_batch_from_point(grid.vertices[vertex])
            grid.vectors[vertex, ...], boundary_vertex_arr, status_arr = (
                self.diffuse_vectors(batch_points, vector_type="normal")
            )
        print(
            f"Process finished --- {(time.time() - start_time)/len(grid.vertices)} seconds ---"
        )
        return grid.vectors


def batch_sample_random_direction(batch_size):
    """
    Samples random direction vectors in R^3 for a given batch size.

    This function generates `batch_size` number of random direction vectors
    in 3-dimensional space (R^3). It does so by sampling from a Gaussian
    distribution and then normalizing the vectors to have a unit norm.

    Args:
        batch_size (int): The number of random direction vectors to sample.

    Returns:
        np.ndarray: A (batch_size, 3) array where each row is a unit vector
                    representing a random direction in 3-dimensional space.
    """
    direction = np.random.randn(batch_size, 3)
    return direction / np.linalg.norm(direction, axis=1)[:, None]


def walk_on_spheres_primitive():
    # Step 1: Compute vectors from each point to the sphere center
    vectors_to_center = points - self.sphere_center  # Nx3 matrix

    # Step 2: Normalize the vectors to get direction
    distances_to_center = np.linalg.norm(
        vectors_to_center, axis=1, keepdims=True
    )  # Nx1 vector of distances
    directions_to_sphere = (
        vectors_to_center / distances_to_center
    )  # Normalize the vectors (Nx3 matrix)

    # Step 3: Compute the closest points on the sphere
    closest_points_on_primitive = (
        self.sphere_center + directions_to_sphere * self.sphere_radius
    )  # Nx3 matrix

    # Step 4: Compute the distances from each point to the closest point on the sphere
    sphere_distances = np.linalg.norm(
        points - closest_points_on_primitive, axis=1
    )  # Nx1 vector of distances
    converged_primitive = sphere_distances < self.epsilon
    status_arr[converged_primitive] = "converged_primitive"
