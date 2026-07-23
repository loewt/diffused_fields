"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from ._rotor_interop import as_scipy_rotation


def extract_plane(vertices, axis="x", value=0.0, tol=1e-3):
    """
    Extracts a 2D plane from a 3D Grid instance at a fixed coordinate value.

    Parameters:
    - grid (Grid): An instance of your Grid class (must be 3D).
    - axis (str): Axis to fix ('x', 'y', or 'z').
    - value (float): Coordinate value at which to slice.
    - tol (float): Tolerance for floating point comparisons.

    Returns:
    - np.ndarray: Array of shape (N_plane, 3) containing the sliced plane points.
    """

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

    idx = axis_to_index[axis]
    mask = np.abs(vertices[:, idx] - value) < tol
    plane = vertices[mask]

    return plane


class Manifold:

    def __init__(
        self,
        type=None,
        scale=None,
        translation=None,
        rotation=None,
    ):
        self.type = type
        self.scale = scale
        self.translation = translation
        # Accept a plain rotation matrix, a scipy Rotation, or a gafropy
        # Rotor (scalar or gafropy.numpy.Rotor) here; normalized to a scipy
        # Rotation so the rest of the codebase's `.as_matrix()`/`.as_euler()`
        # calls keep working unchanged.
        self.rotation = None if rotation is None else as_scipy_rotation(rotation)


class Plane(Manifold):
    def __init__(
        self,
        normal=(0, 0, 1),
        point=(0, 0, 0),
        scale=None,
        translation=None,
        rotation=None,
        angle=0.0,
    ):
        """
        Initialize a plane defined by a normal vector and a point on the plane.
        """
        super().__init__(
            type="Plane", scale=scale, translation=translation, rotation=rotation
        )

        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(
            self.normal
        )  # ensure it's a unit vector
        self.point = np.array(point, dtype=float)
        self.angle = angle

    def get_closest_points(self, points):
        """
        Compute closest points on the plane for given query points.

        Args:
            points (np.ndarray): Nx3 array of 3D query points.

        Returns:
            distances (np.ndarray): Euclidean distances to the plane.
            directions (np.ndarray): Nx3 array of direction vectors from the point to its projection.
        """
        # Vector from a fixed point on the plane to the query points
        vecs = points - self.point

        # Signed distance along the normal
        distances_signed = np.dot(vecs, self.normal)  # (N,)
        distances = np.abs(distances_signed)

        # Closest points on the plane
        closest_points = points - np.outer(distances_signed, self.normal)

        # Direction vectors from query to closest points
        directions = closest_points - points  # Note: pointing inward

        return distances, directions

    def get_local_bases(self, points_on_plane):
        """
        Return an orthonormal basis (tangent1, tangent2, normal) at each point,
        with an optional in-plane rotation by `angle` (in radians).

        Args:
            points_on_plane (np.ndarray): Nx3 points on the plane.
            angle (float): Angle (in radians) to rotate the in-plane tangent vectors
                        around the normal vector.

        Returns:
            basis (np.ndarray): Nx3x3 array where each [i,:,:] is the 3x3 local basis at point i.
        """
        N = points_on_plane.shape[0]

        # The normal is the same everywhere for a flat plane
        normal_unit = self.normal[None, :].repeat(N, axis=0)

        # Create an arbitrary vector not parallel to the normal
        arbitrary = np.array([0.0, 1.0, 0.0])
        if np.allclose(arbitrary, self.normal):
            arbitrary = np.array([1.0, 0.0, 0.0])

        # First tangent: orthogonal to the normal
        tangent1 = np.cross(self.normal, arbitrary)
        tangent1 /= np.linalg.norm(tangent1)

        # Second tangent: orthogonal to both
        tangent2 = np.cross(self.normal, tangent1)

        # Apply rotation in the tangent plane
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        tangent1_rot = cos_a * tangent1 + sin_a * tangent2
        tangent2_rot = -sin_a * tangent1 + cos_a * tangent2

        # Repeat for each point
        tangent1s = np.tile(tangent1_rot, (N, 1))
        tangent2s = np.tile(tangent2_rot, (N, 1))

        # Stack the basis vectors
        basis = np.stack((tangent1s, tangent2s, normal_unit), axis=2)  # Nx3x3

        return basis

    # def get_local_bases(self, points_on_plane):
    #     """
    #     Return an orthonormal basis (tangent1, tangent2, normal) at each point.

    #     Args:
    #         points_on_plane (np.ndarray): Nx3 points on the plane.

    #     Returns:
    #         basis (np.ndarray): Nx3x3 array where each [i,:,:] is the 3x3 local basis at point i.
    #     """
    #     N = points_on_plane.shape[0]

    #     # The normal is the same everywhere for a flat plane
    #     normal_unit = self.normal[None, :].repeat(N, axis=0)

    #     # Create an arbitrary vector not parallel to the normal
    #     arbitrary = np.array([1.0, 0.0, 0.0])
    #     if np.allclose(arbitrary, self.normal):
    #         arbitrary = np.array([0.0, 1.0, 0.0])

    #     # First tangent: orthogonal to the normal
    #     tangent1 = -np.cross(self.normal, arbitrary)
    #     tangent1 = tangent1 / np.linalg.norm(tangent1)

    #     # Second tangent: orthogonal to both
    #     tangent2 = np.cross(self.normal, tangent1)

    #     # Repeat for each point
    #     tangent1s = np.tile(tangent1, (N, 1))
    #     tangent2s = np.tile(tangent2, (N, 1))

    #     # Stack the basis vectors
    #     basis = np.stack((tangent1s, tangent2s, normal_unit), axis=2)  # Nx3x3

    #     return basis

    def sample_points(self, center=None, width=1.0, height=1.0, resolution=10):
        """
        Uniformly sample a grid of points on the plane for visualization.

        Args:
            center (np.ndarray or tuple): Center of the sampling grid. Defaults to self.point.
            width (float): Total width of the sampled rectangle.
            height (float): Total height of the sampled rectangle.
            resolution (int): Number of points per axis (resolution x resolution grid).

        Returns:
            np.ndarray: (resolution**2, 3) array of sampled 3D points on the plane.
        """
        if center is None:
            center = self.point
        else:
            center = np.array(center)

        # Orthonormal tangent basis vectors
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(arbitrary, self.normal):
            arbitrary = np.array([0.0, 1.0, 0.0])

        tangent1 = np.cross(self.normal, arbitrary)
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(self.normal, tangent1)

        # Grid coordinates in tangent plane
        lin = np.linspace(-0.5, 0.5, resolution)
        uu, vv = np.meshgrid(lin * width, lin * height)
        uu = uu.flatten()
        vv = vv.flatten()

        # Sample points: center + u * tangent1 + v * tangent2
        points = (
            center[None, :]
            + uu[:, None] * tangent1[None, :]
            + vv[:, None] * tangent2[None, :]
        )
        return points


class Line(Manifold):
    def __init__(
        self,
        direction=(1, 0, 0),
        point=(0, 0, 0),
        scale=None,
        translation=None,
        rotation=None,
    ):
        """
        Initialize a line defined by a direction vector and a point on the line.
        """
        super().__init__(
            type="Line", scale=scale, translation=translation, rotation=rotation
        )

        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # Ensure it's a unit vector
        self.point = np.array(point, dtype=float)

    def get_closest_points(self, points):
        """
        Compute closest points on the line for given query points.

        Args:
            points (np.ndarray): Nx3 array of 3D query points.

        Returns:
            distances (np.ndarray): Euclidean distances to the line.
            directions (np.ndarray): Nx3 direction vectors from the point to its projection.
        """
        vecs = points - self.point  # Nx3
        projections = np.dot(vecs, self.direction)[:, None] * self.direction  # Nx3
        closest_points = self.point + projections
        directions = closest_points - points
        distances = np.linalg.norm(directions, axis=1)

        return distances, directions

    def get_local_bases(self, points_on_line):
        """
        Return an orthonormal basis (tangent, normal1, normal2) at each point on the line.

        Args:
            points_on_line (np.ndarray): Nx3 points on the line.

        Returns:
            basis (np.ndarray): Nx3x3 array where each [i,:,:] is the 3x3 local basis at point i.
        """
        N = points_on_line.shape[0]
        tangent_unit = self.direction[None, :].repeat(N, axis=0)

        # Find two orthonormal vectors perpendicular to the direction
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(arbitrary, self.direction):
            arbitrary = np.array([0.0, 1.0, 0.0])

        normal1 = np.cross(self.direction, arbitrary)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.cross(self.direction, normal1)

        normal1s = np.tile(normal1, (N, 1))
        normal2s = np.tile(normal2, (N, 1))

        basis = np.stack((tangent_unit, normal1s, normal2s), axis=2)  # Nx3x3

        return basis

    def sample_points(self, center=None, length=1.0, resolution=10):
        """
        Sample points uniformly along the line segment centered at `center`.

        Args:
            center (np.ndarray or tuple): Center of the sampled segment. Defaults to self.point.
            length (float): Total length of the sampled segment.
            resolution (int): Number of points to sample.

        Returns:
            np.ndarray: (resolution, 3) array of sampled 3D points along the line.
        """
        if center is None:
            center = self.point
        else:
            center = np.array(center)

        lin = np.linspace(-0.5, 0.5, resolution) * length
        points = center[None, :] + lin[:, None] * self.direction[None, :]

        return points


class Sphere(Manifold):
    def __init__(
        self, radius=1.0, center=(0, 0, 0), scale=None, translation=None, rotation=None
    ):
        # Initialize the base Manifold class
        super().__init__(
            type="Sphere", scale=scale, translation=translation, rotation=rotation
        )

        # Sphere-specific attributes
        self.radius = radius
        self.center = np.array(center)

    def apply_scale(self):
        """Apply scaling to the radius."""
        if self.scale is not None:
            self.radius *= self.scale

    def get_closest_points(self, points):
        """
        Compute closest points on the sphere surface for given query points.

        Args:
            query_points (np.ndarray): Nx3 array of 3D points.

        Returns:
            distances (np.ndarray): Euclidean distances to the sphere surface.
            indices (np.ndarray): Nx2 array of (theta, phi) values in radians.
        """
        # Vector from sphere center to query points
        vecs = points - self.center

        # Normalize vectors to lie on the sphere surface
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        directions = vecs / norms

        # Closest points lie on the sphere surface along the direction vectors
        closest_points = self.center + self.radius * directions

        # Compute distances from query points to closest points
        distances = np.linalg.norm(points - closest_points, axis=1)

        # Convert direction vectors to spherical coordinates (theta, phi)
        # theta: azimuthal angle in [0, 2pi], phi: polar angle in [0, pi]
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        theta = np.arctan2(y, x) % (2 * np.pi)
        phi = np.arccos(z)

        # indices = np.stack([theta, phi], axis=1)

        indices = np.stack([x, y, z], axis=1)

        return distances, indices

    # def get_local_bases(self, points_on_sphere):
    #     """
    #     Computes the vector field for a batch of points on the surface of the sphere.
    #     The field emanates from the south pole (0, 0, -radius) and sinks into the north
    #     pole (0, 0, +radius).

    #     :self.sphere_center: The center of the sphere (3D vector)
    #     :self.sphere_radius: The radius of the sphere (scalar)
    #     :self.batch_points_on_sphere: A batch of points on the surface of the sphere
    #     (Nx3 matrix)
    #     :return: An Nx3 matrix representing the vector field for each point in world
    #     coordinates.
    #     """
    #     self.get_south_north_poles()
    #     # self.get_left_right_poles()
    #     # Step 1: Compute the radial vectors from the south pole to each point on the sphere
    #     radial_vectors = points_on_sphere - self.south_pole  # Nx3 matrix

    #     # Step 2: Compute the normal vectors (center to each point on the sphere)
    #     # points towards the center this is our convention
    #     normal_vectors = -(points_on_sphere - self.center)  # Nx3 matrix

    #     # Normalize the normal vectors (get the direction for each point)
    #     normal_unit_vectors = normal_vectors / np.linalg.norm(
    #         normal_vectors, axis=1, keepdims=True
    #     )  # Nx3 matrix

    #     # Step 3: Project the radial vectors onto the tangent plane at each point (i.e., remove the normal component)
    #     dot_products = np.sum(
    #         radial_vectors * normal_unit_vectors, axis=1, keepdims=True
    #     )  # Nx1 vector of dot products
    #     tangent_vectors = -(
    #         radial_vectors - dot_products * normal_unit_vectors
    #     )  # Nx3 matrix (tangent vectors)

    #     # Step 4: Normalize the tangent vectors to get unit vectors
    #     tangent_unit_vectors = tangent_vectors / np.linalg.norm(
    #         tangent_vectors, axis=1, keepdims=True
    #     )  # Nx3 matrix
    #     tangent_unit_vectors_2 = np.cross(normal_unit_vectors, tangent_unit_vectors)

    #     sphere_basis = np.stack(
    #         (tangent_unit_vectors_2, -tangent_unit_vectors, normal_unit_vectors),
    #         axis=2,
    #     )
    #     return sphere_basis

    def get_local_bases(self, points_on_sphere):
        """
        Computes the vector field for a batch of points on the surface of the sphere.
        The field emanates from the south pole (0, 0, -radius) and sinks into the north
        pole (0, 0, +radius).

        :self.sphere_center: The center of the sphere (3D vector)
        :self.sphere_radius: The radius of the sphere (scalar)
        :self.batch_points_on_sphere: A batch of points on the surface of the sphere
        (Nx3 matrix)
        :return: An Nx3 matrix representing the vector field for each point in world
        coordinates.
        """
        self.get_left_right_poles()

        # self.get_south_north_poles()
        # Step 1: Compute the radial vectors from the south pole to each point on the sphere
        radial_vectors = points_on_sphere - self.left_pole  # Nx3 matrix
        # radial_vectors = points_on_sphere - self.south_pole  # Nx3 matrix

        # Step 2: Compute the normal vectors (center to each point on the sphere)
        normal_vectors = points_on_sphere - self.center  # Nx3 matrix

        # Normalize the normal vectors (get the direction for each point)
        normal_unit_vectors = -normal_vectors / np.linalg.norm(
            normal_vectors, axis=1, keepdims=True
        )  # Nx3 matrix

        # Step 3: Project the radial vectors onto the tangent plane at each point (i.e., remove the normal component)
        dot_products = np.sum(
            radial_vectors * normal_unit_vectors, axis=1, keepdims=True
        )  # Nx1 vector of dot products
        tangent_vectors = -(
            radial_vectors - dot_products * normal_unit_vectors
        )  # Nx3 matrix (tangent vectors)

        # Step 4: Normalize the tangent vectors to get unit vectors
        tangent_unit_vectors = tangent_vectors / np.linalg.norm(
            tangent_vectors, axis=1, keepdims=True
        )  # Nx3 matrix
        tangent_unit_vectors = np.cross(normal_unit_vectors, tangent_unit_vectors)
        tangent_unit_vectors_2 = np.cross(normal_unit_vectors, tangent_unit_vectors)

        sphere_basis = np.stack(
            (tangent_unit_vectors, tangent_unit_vectors_2, normal_unit_vectors),
            axis=2,
        )

        return sphere_basis

    def get_south_north_poles(self):
        # Define the south pole and north pole in world coordinates
        self.south_pole = self.center + np.array([0.0, 0.0, -self.radius])
        self.north_pole = self.center + np.array([0.0, 0.0, +self.radius])

    def get_left_right_poles(self):

        # self.left_pole = self.center + np.array([0.0, -self.radius, 0.0])
        # self.right_pole = self.center + np.array([0.0, +self.radius, 0.0])

        self.left_pole = self.center + np.array([-self.radius, 0.0, 0.0])
        self.right_pole = self.center + np.array([+self.radius, 0.0, 0.0])
        return [self.right_pole, self.left_pole]

    def apply_translation(self):
        """Translate the sphere's center."""
        if self.translation is not None:
            self.center += np.array(self.translation)

    def __repr__(self):
        return (
            f"Sphere(radius={self.radius}, center={self.center.tolist()}, "
            f"scale={self.scale}, translation={self.translation}, rotation={self.rotation})"
        )

    def get_visualization_vertices(self, num_vertices=1000):
        """
        Uniformly sample vertices on the surface of the sphere.

        Args:
            num_vertices (int): Number of points to sample.

        Returns:
            np.ndarray: Array of shape (num_vertices, 3) with 3D coordinates.
        """
        indices = np.arange(0, num_vertices)
        phi = np.arccos(1 - 2 * (indices + 0.5) / num_vertices)  # polar angle
        theta = np.pi * (1 + 5**0.5) * indices  # golden angle increment

        # Convert spherical to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Stack and transform points
        points = np.stack([x, y, z], axis=1)
        points = self.radius * points + self.center

        return points


def import_system_matrix(
    grid,
    obj_name,
    method,
    dt,
    directory,
):
    """
    For large grids, it is more efficient to save the system matrix and load it.
    `directory` is where the cached matrix is read from/written to.
    """
    filename = f"{directory}/grid_{grid.Nx}_{obj_name}_{method}.npy"
    try:

        # Try to load the laplacian from file

        # Object name matters because the scale is different?
        invA = np.load(filename, allow_pickle=True)
        print("invA loaded from file.")

    except FileNotFoundError:
        print("File not found. Creating the laplacian matrix...")
        # If the file doesn't exist, preprocess the laplacian and save it
        L = laplacian_3d_matrix(grid)
        if method == "laplace":
            invA = sla.inv(-L)

        elif method == "diffusion":
            A = get_system_matrix_implicit_heat(L, dt=dt)
            invA = sla.inv(A)

        import pickle

        with open(filename, "wb") as f:
            pickle.dump(invA, f, protocol=4)
    return invA


def laplacian_3d_matrix(grid):
    """
    TODO: I am not sure whether we use the sparsity structure
    to the best extent that we can

    Generate a sparse Laplacian matrix for a 3D grid.

    Parameters:
    - Nx (int): Number of grid points along the x-axis.
    - Ny (int): Number of grid points along the y-axis.
    - Nz (int): Number of grid points along the z-axis.
    - h (float): Uniform grid spacing.

    Returns:
    - laplacian (scipy.sparse.csr_matrix): Sparse Laplacian matrix.

    The Laplacian matrix is generated using the finite difference method.
    It represents the discretized Laplace operator for a 3D grid.
    The matrix is returned in Compressed Sparse Row (CSR) format for efficient storage and computation.
    """

    N = grid.Nx * grid.Ny * grid.Nz
    data = []
    rows = []
    cols = []

    def index(x, y, z):
        """
        Calculates the index of a point in a 3D grid.

        Parameters:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        z (int): The z-coordinate of the point.

        Returns:
        int: The calculated index of the point in the grid.
        """
        return x * (grid.Ny * grid.Nz) + y * grid.Nz + z

    for x in range(grid.Nx):
        for y in range(grid.Ny):
            for z in range(grid.Nz):
                i = index(x, y, z)
                rows.append(i)
                cols.append(i)
                data.append(-6 / grid.h**2)  # Center point

                if x > 0:
                    rows.append(i)
                    cols.append(index(x - 1, y, z))
                    data.append(1 / grid.h**2)
                if x < grid.Nx - 1:
                    rows.append(i)
                    cols.append(index(x + 1, y, z))
                    data.append(1 / grid.h**2)
                if y > 0:
                    rows.append(i)
                    cols.append(index(x, y - 1, z))
                    data.append(1 / grid.h**2)
                if y < grid.Ny - 1:
                    rows.append(i)
                    cols.append(index(x, y + 1, z))
                    data.append(1 / grid.h**2)
                if z > 0:
                    rows.append(i)
                    cols.append(index(x, y, z - 1))
                    data.append(1 / grid.h**2)
                if z < grid.Nz - 1:
                    rows.append(i)
                    cols.append(index(x, y, z + 1))
                    data.append(1 / grid.h**2)

    laplacian = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return laplacian
