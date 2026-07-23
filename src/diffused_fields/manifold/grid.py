"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .manifold import Manifold

# from diffusion_utils import *


class Grid(Manifold):
    def __init__(
        self,
        Nx=10,
        Ny=None,
        Nz=None,
        x_min=0,
        x_max=1,
        y_min=None,
        y_max=None,
        z_min=None,
        z_max=None,
        scale=None,
        translation=None,
        rotation=None,
    ):
        """
        Initializes a 2D or 3D grid object with specified dimensions and limits.

        Parameters:
        - Nx (int): Number of grid points in the x-direction (default: 20).
        - Ny (int): Number of grid points in the y-direction (default: None, which sets Ny = Nx).
        - Nz (int): Number of grid points in the z-direction (default: None for 2D grids).
        - x_min (float): Minimum value of x-coordinate (default: 0).
        - x_max (float): Maximum value of x-coordinate (default: 1).
        - y_min (float): Minimum value of y-coordinate (default: None, which sets y_min = x_min).
        - y_max (float): Maximum value of y-coordinate (default: None, which sets y_max = x_max).
        - z_min (float): Minimum value of z-coordinate for 3D grids (default: None).
        - z_max (float): Maximum value of z-coordinate for 3D grids (default: None).

        Attributes:
        - Nx, Ny, Nz: Number of grid points in each direction.
        - N: Total number of grid points (Nx * Ny * Nz or Nx * Ny for 2D grids).
        - vertices: Array of grid point coordinates.
        - implicit_shape: Shape of the grid values for interpolation.
        - h: Uniform spacing between grid points.
        """

        super().__init__(
            type=type(self), scale=scale, translation=translation, rotation=rotation
        )

        # Set grid dimensions
        self.Nx = Nx
        self.Ny = Ny if Ny is not None else Nx
        self.Nz = Nz if Nz is not None else Nx

        # Determine if it's a 2D or 3D grid based on Nz
        self.is_3d = Nz is not None

        # Set axis limits
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min if y_min is not None else x_min, (
            y_max if y_max is not None else x_max
        )
        if self.is_3d:
            self.z_min, self.z_max = z_min if z_min is not None else x_min, (
                z_max if z_max is not None else x_max
            )

        # Calculate spacings
        self.hx = (self.x_max - self.x_min) / self.Nx
        self.hy = (self.y_max - self.y_min) / self.Ny
        if self.is_3d:
            self.hz = (self.z_max - self.z_min) / self.Nz

        # Create grid points
        x_vec = np.linspace(self.x_min, self.x_max, self.Nx)
        y_vec = np.linspace(self.y_min, self.y_max, self.Ny)

        if self.is_3d:
            z_vec = np.linspace(self.z_min, self.z_max, self.Nz)
            xx, yy, zz = np.meshgrid(x_vec, y_vec, z_vec, indexing="ij")
            self.vertices = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
            self.implicit_shape = [self.Nx, self.Ny, self.Nz, 3]
            self.N = self.Nx * self.Ny * self.Nz
            if not (self.hx == self.hy == self.hz):
                # print(
                #     "Warning!: Non-uniform grid spacing is not supported for Laplacian calculation."
                # )
                pass
            else:
                self.h = self.hx
        else:
            xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
            self.vertices = np.vstack([xx.flatten(), yy.flatten()]).T
            self.implicit_shape = [self.Nx, self.Ny, 2]
            self.N = self.Nx * self.Ny
            if self.hx != self.hy:
                # print(
                #     "Warning!: Non-uniform grid spacing is not supported for Laplacian calculation."
                # )
                pass
            else:
                self.h = self.hx

        # Store grid points
        self.points = [x_vec, y_vec] if not self.is_3d else [x_vec, y_vec, z_vec]

    def __repr__(self):
        grid_type = "3D" if self.is_3d else "2D"
        return f"<{grid_type} Grid: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz if self.is_3d else 'N/A'}, Total Points={self.N}>"

    def extract_yz_plane(self, x_value, tol=1e-2):
        """
        Extracts the yz-plane at a fixed x-value from a 3D Grid instance.

        Parameters:
        - grid (Grid): An instance of your Grid class (must be 3D).
        - x_value (float): The x-coordinate at which to slice the yz-plane.
        - tol (float): Tolerance for matching x-values due to floating point error.

        Returns:
        - np.ndarray: Array of shape (Ny * Nz, 3) containing the yz-plane points.
        """
        if not self.is_3d:
            raise ValueError("Grid must be 3D to extract a yz-plane.")

        # Boolean mask for vertices with x == x_value (within a small tolerance)
        mask = np.abs(self.vertices[:, 0] - x_value) < tol
        yz_plane = self.vertices[mask]

        return yz_plane

    def get_center(self):
        """
        Calculate the center coordinates of the grid.

        Returns:
        - numpy.ndarray: Array containing [x_mid, y_mid, z_mid] (or [x_mid, y_mid] for 2D grids)
        """
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2

        if self.is_3d:
            z_mid = (self.z_min + self.z_max) / 2
            self.center = np.array([x_mid, y_mid, z_mid])
        else:
            self.center = np.array([x_mid, y_mid])

        return self.center

    def compute_rotation_around_global_x_smoothness(self, orientation_matrices):
        """
        Compute smoothness metric based on rotation around global X-axis.

        For each local frame:
        1. Project local X-axis onto global X-axis direction
        2. Use global Y direction as 0° reference around global X-axis
        3. Measure how much each frame is rotated around global X compared to neighbors

        Parameters:
        - orientation_matrices (np.ndarray): Array of shape (N, 3, 3) containing orientation matrices

        Returns:
        - tuple: (avg_angular_deviation, std_angular_deviation, angular_deviations, per_point_deviations)
        """
        # Ensure we have a proper grid structure for neighbor finding
        if not self.is_3d:
            # For 2D grids, treat as XY plane
            unique_x = np.unique(self.vertices[:, 0])
            unique_y = np.unique(self.vertices[:, 1])
            grid_height = len(unique_x)
            grid_width = len(unique_y)
        else:
            # For 3D grids on a plane (e.g., XZ plane), determine the structure
            # This assumes vertices were extracted from a plane
            unique_coords_1 = np.unique(self.vertices[:, 1])  # Y or Z coordinate
            unique_coords_2 = np.unique(self.vertices[:, 2])  # Z coordinate
            grid_height = len(unique_coords_1)
            grid_width = len(unique_coords_2)

        # Global reference directions
        global_x = np.array([1, 0, 0])
        global_y = np.array([0, 1, 0])

        total_angular_deviation = 0
        neighbor_count = 0
        angular_deviations = []
        per_point_deviations = np.zeros(len(self.vertices))

        for i in range(len(self.vertices)):
            # Convert linear index to 2D grid coordinates
            row = i // grid_width
            col = i % grid_width

            # Get current frame's local X-axis
            current_local_x = orientation_matrices[i, :, 0]

            # Project local X onto global X direction and get the sign/alignment
            x_alignment = np.dot(current_local_x, global_x)

            # Get the component of local X that's perpendicular to global X (in YZ plane)
            local_x_perp = current_local_x - x_alignment * global_x

            # If the perpendicular component is too small, the frame is aligned with global X
            if np.linalg.norm(local_x_perp) < 1e-6:
                current_angle = 0.0
            else:
                # Normalize the perpendicular component
                local_x_perp = local_x_perp / np.linalg.norm(local_x_perp)

                # Compute angle from global Y direction around global X axis
                # Use atan2 to get full 360° range
                y_component = np.dot(local_x_perp, global_y)
                z_component = np.dot(local_x_perp, np.array([0, 0, 1]))
                current_angle = np.arctan2(z_component, y_component)

            # Check neighbors (right and down)
            neighbors = []
            if col < grid_width - 1:  # Right neighbor
                neighbors.append(i + 1)
            if row < grid_height - 1:  # Down neighbor
                neighbors.append(i + grid_width)

            point_total_deviation = 0.0
            point_neighbor_count = 0

            for neighbor_idx in neighbors:
                # Get neighbor frame's local X-axis
                neighbor_local_x = orientation_matrices[neighbor_idx, :, 0]

                # Project neighbor local X onto global X direction
                neighbor_x_alignment = np.dot(neighbor_local_x, global_x)

                # Get perpendicular component
                neighbor_x_perp = neighbor_local_x - neighbor_x_alignment * global_x

                if np.linalg.norm(neighbor_x_perp) < 1e-6:
                    neighbor_angle = 0.0
                else:
                    neighbor_x_perp = neighbor_x_perp / np.linalg.norm(neighbor_x_perp)
                    neighbor_y_component = np.dot(neighbor_x_perp, global_y)
                    neighbor_z_component = np.dot(neighbor_x_perp, np.array([0, 0, 1]))
                    neighbor_angle = np.arctan2(
                        neighbor_z_component, neighbor_y_component
                    )

                # Compute angular difference (handle wraparound)
                angle_diff = abs(current_angle - neighbor_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                total_angular_deviation += angle_diff
                angular_deviations.append(angle_diff)
                neighbor_count += 1

                point_total_deviation += angle_diff
                point_neighbor_count += 1

            # Store average deviation for this grid point
            per_point_deviations[i] = (
                point_total_deviation / point_neighbor_count
                if point_neighbor_count > 0
                else 0.0
            )

        avg_angular_deviation = (
            total_angular_deviation / neighbor_count if neighbor_count > 0 else 0
        )
        std_angular_deviation = (
            np.std(angular_deviations) if len(angular_deviations) > 0 else 0
        )

        return (
            avg_angular_deviation,
            std_angular_deviation,
            angular_deviations,
            per_point_deviations,
        )

    def unconstrained_vector_diffusion(self, source_vectors):
        vf = self.diffuse_vector_directions(source_vectors)

        u0 = np.zeros(len(self.vertices))
        for i, vertex in enumerate(self.source_vertices):
            u0[vertex] = np.linalg.norm(source_vectors[i, :])
        uf = self.solve_scalar_diffusion(u0)

        phi_0 = np.zeros(len(self.vertices))
        for i, vertex in enumerate(self.source_vertices):
            phi_0[vertex] = 1
        phi_f = self.solve_scalar_diffusion(phi_0)
        # print(f"phi_f: {phi_f}")
        # print(f"uf: {uf}")

        # Normalize the vector field
        vf = vf / np.linalg.norm(vf, axis=1)[:, None]
        vf = vf * uf[:, None] / phi_f[:, None]
        self.u0, self.uf, self.phi_0, self.phi_f, self.vf = u0, uf, phi_0, phi_f, vf

        # Debug
        # ==============================================================================
        # print(
        #     f"Magnitude error after diffusion:\n{vf[source_vertices_grid,:]-set_vals}"
        # )
        # print(
        #     f"Magnitude error after diffusion:\n{np.linalg.norm(vf[source_vertices_grid,:],axis=1)-np.linalg.norm(set_vals,axis=1)}"
        # )

    def diffuse_vector_directions(self, source_vectors):
        # In the Euclidean space we can diffuse the scalar components independently
        # then combine to a vector field
        diffused_vector_field = np.zeros(
            (len(self.vertices), len(source_vectors[0]))
        )  # Final vector field
        for i in range(len(source_vectors[0])):
            # Consider each component of the vector field separately as a scalar field
            u0 = np.zeros(len(self.vertices))
            for j, vertex in enumerate(self.source_vertices):
                u0[vertex] = source_vectors[j, i]  # mind the i,j ordering
                diffused_vector_field[:, i] = self.solve_scalar_diffusion(u0)
        return diffused_vector_field

    def preprocess(self, method):
        self.method = method
        self.L = self.laplacian_3d_matrix()
        if self.method == "inv":
            # System matrix for solving heat equation with implicit Euler
            self.invA = sla.inv(-self.L)
        elif self.method == "cg" or self.method == "sp":
            self.A = self.get_system_matrix_implicit_heat(self.L, dt=1 * self.h**2)

    def solve_scalar_diffusion(self, u0):
        if not hasattr(self, "L"):
            self.preprocess()
            print("Error: Laplacian matrix L is not defined.")
        # Solve the linear system
        if self.method == "cg":
            return solve_conjugate_gradient(A=self.A, b=u0)
        elif self.method == "sp":
            return sla.spsolve(A=self.A, b=u0)
        elif self.method == "inv":
            return self.invA @ u0

    def laplacian_3d_matrix(self):
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

        N = self.Nx * self.Ny * self.Nz
        data = []
        rows = []
        cols = []

        def index(x, y, z):
            """
            Calculates the index of a point in a 3D self.

            Parameters:
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.
            z (int): The z-coordinate of the point.

            Returns:
            int: The calculated index of the point in the self.
            """
            return x * (self.Ny * self.Nz) + y * self.Nz + z

        for x in range(self.Nx):
            for y in range(self.Ny):
                for z in range(self.Nz):
                    i = index(x, y, z)
                    rows.append(i)
                    cols.append(i)
                    data.append(-6 / self.h**2)  # Center point

                    if x > 0:
                        rows.append(i)
                        cols.append(index(x - 1, y, z))
                        data.append(1 / self.h**2)
                    if x < self.Nx - 1:
                        rows.append(i)
                        cols.append(index(x + 1, y, z))
                        data.append(1 / self.h**2)
                    if y > 0:
                        rows.append(i)
                        cols.append(index(x, y - 1, z))
                        data.append(1 / self.h**2)
                    if y < self.Ny - 1:
                        rows.append(i)
                        cols.append(index(x, y + 1, z))
                        data.append(1 / self.h**2)
                    if z > 0:
                        rows.append(i)
                        cols.append(index(x, y, z - 1))
                        data.append(1 / self.h**2)
                    if z < self.Nz - 1:
                        rows.append(i)
                        cols.append(index(x, y, z + 1))
                        data.append(1 / self.h**2)

        laplacian = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        return laplacian


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


def find_closest_point_on_grid(grid_vertices, point):
    """
    Finds the index of the closest point in the grid_vertices array to the given point.

    Parameters:
    grid_vertices (numpy.ndarray): Array of grid vertices.
    point (numpy.ndarray): The point to find the closest point to.

    Returns:
    int: The index of the closest point in the grid_vertices array.
    """
    dists = np.linalg.norm(grid_vertices - point, axis=1)
    min_idx = np.argmin(dists)
    return min_idx
