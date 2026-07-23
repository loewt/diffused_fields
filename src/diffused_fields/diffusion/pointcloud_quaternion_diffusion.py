"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import gafropy as ga
import gafropy.numpy as gn
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..manifold._rotor_interop import as_gafropy_rotor
from .pointcloud_scalar_diffusion import PointcloudScalarDiffusion

# ========== Quaternion helpers ==========
#
# Note: this module's own "quat_*" helpers use gafro's math library
# convention [w, x, y, z] (scalar first), NOT gafropy.Rotor's [x, y, z, w]
# (scipy/Eigen convention). `_as_wxyz_quaternions`/`_rotor_to_wxyz` below
# are the explicit bridge between the two -- a plain index permutation, kept
# local to this module rather than in the shared _rotor_interop helpers
# since those are built around the [x, y, z, w] convention throughout.


def _as_wxyz_quaternions(Q):
    """
    Coerce Q to an (N, 4) array of [w, x, y, z] quaternions (this module's
    convention). Accepts the existing list/array-of-quaternions form, a
    gafropy.numpy.Rotor batch, or a list of gafropy.Rotor instances.
    """
    if isinstance(Q, gn.Rotor):
        quat_xyzw = np.atleast_2d(Q.to_quaternion())
        return quat_xyzw[..., [3, 0, 1, 2]]
    if isinstance(Q, (list, tuple)) and len(Q) > 0 and isinstance(Q[0], ga.Rotor):
        quat_xyzw = np.array([q.to_quaternion() for q in Q])
        return quat_xyzw[..., [3, 0, 1, 2]]
    return np.asarray(Q, dtype=np.float64)


def _rotor_to_wxyz(rotor):
    """Convert a gafropy.numpy.Rotor batch to this module's [w, x, y, z] convention."""
    quat_xyzw = rotor.to_quaternion()
    return quat_xyzw[..., [3, 0, 1, 2]]


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle_deg):
    ang = np.deg2rad(angle_deg)
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = axis / n
    c = np.cos(ang / 2.0)
    s = np.sin(ang / 2.0)
    return quat_normalize(np.array([c, *(u * s)]))


def quat_dot(q1, q2):
    return float(np.dot(q1, q2))


def quat_log(q, eps=1e-12):
    """Stable principal log: returns a 3 vector (pure quaternion)."""
    q = quat_normalize(q)
    w, x, y, z = q
    v = np.array([x, y, z], dtype=float)
    vn = np.linalg.norm(v)
    if vn < eps:
        return np.zeros(3)
    phi = np.arctan2(vn, w)  # in [0, pi)
    return v * (phi / vn)


def quat_exp(v, eps=1e-12):
    """Exponential map from a 3 vector to a unit quaternion."""
    v = np.asarray(v, dtype=float)
    phi = np.linalg.norm(v)
    if phi < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = v / phi
    return np.array([np.cos(phi), *(u * np.sin(phi))])


def expquat_2_rotated_frame(pure_quaternions):
    """
    Convert pure quaternions (N x 3 array) to rotation matrices (N x 3 x 3 array).
    Each 3-vector is converted to a quaternion via quat_exp, then to a rotation matrix.
    """
    pure_quaternions = np.asarray(pure_quaternions)
    if pure_quaternions.ndim == 1:
        pure_quaternions = pure_quaternions.reshape(1, -1)

    N = pure_quaternions.shape[0]
    bases = np.zeros((N, 3, 3))

    for i in range(N):
        quat = quat_exp(pure_quaternions[i])
        # Convert quaternion to rotation matrix using scipy
        r = R.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        )  # scipy uses [x,y,z,w] format
        bases[i] = r.as_matrix()

    return bases


def expquat_2_rotor(pure_quaternions):
    """
    Convert pure quaternions (N x 3 array) to a gafropy.numpy.Rotor batch.

    Gafro-native sibling of `expquat_2_rotated_frame`: reuses its existing,
    already-correct matrix output rather than re-deriving the exp map in
    gafro's own bivector-log convention (which differs from this module's).
    """
    return as_gafropy_rotor(expquat_2_rotated_frame(pure_quaternions))


def get_quat_between_vectors(v1, v2):
    """
    Get the quaternion that rotates v1 to v2.
    Both vectors should be unit vectors or will be normalized.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute rotation axis and angle
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # Handle parallel or anti-parallel vectors
    if axis_norm < 1e-12:
        dot = np.dot(v1, v2)
        if dot > 0:  # Same direction
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:  # Opposite direction - find perpendicular axis
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, [1, 0, 0])
            else:
                axis = np.cross(v1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return np.array([0.0, axis[0], axis[1], axis[2]])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # Convert to quaternion
    half_angle = angle / 2.0
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))])


# ========== Sign search on S^3 ==========


def best_sign_assignment(quats):
    """
    Try all 2^{N-1} sign patterns relative to q0.
    Maximize the minimum absolute dot among all pairs.
    """
    Q = [quat_normalize(q) for q in quats]
    N = len(Q)
    if N <= 1:
        return Q

    best = None
    best_min_dot = -np.inf
    for mask in range(1 << (N - 1)):
        cand = [Q[0].copy()]
        for i in range(1, N):
            flip = -1.0 if (mask & (1 << (i - 1))) else 1.0
            cand.append(flip * Q[i])
        # score this assignment
        m = +np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = abs(quat_dot(cand[i], cand[j]))
                if d < m:
                    m = d
        if m > best_min_dot:
            best_min_dot = m
            best = [c.copy() for c in cand]
    return best


# ========== Your entry points ==========


def quats_from_z_angles(z_deg_list):
    """Make unit quaternions for pure z rotations."""
    return [quat_from_axis_angle([0, 0, 1], deg) for deg in z_deg_list]


def pure_quaternions_for_dirichlet(Q, antipode_tol_deg=179.9):
    """
    Input: list of unit quaternions Q (N x 4 array-like, [w, x, y, z]), or
        anything _as_wxyz_quaternions accepts (a gafropy.numpy.Rotor batch,
        or a list of gafropy.Rotor instances).
    Output: list of pure quaternion 3 vectors for Dirichlet data
    """
    Q = _as_wxyz_quaternions(Q)

    # 2) global sign choice on S^3
    Q = best_sign_assignment(Q)

    # 3) optional guard for near pi pairs
    def rel_angle_deg(q1, q2):
        d = abs(quat_dot(q1, q2))
        d = np.clip(d, -1.0, 1.0)
        return np.degrees(2.0 * np.arccos(d))

    worst = 0.0
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            worst = max(worst, rel_angle_deg(Q[i], Q[j]))
    if worst >= antipode_tol_deg:
        print("Warning: a boundary pair is near one hundred eighty degrees.")

    # 4) logs for Dirichlet data
    return [quat_log(q) for q in Q], Q  # returns logs and the lifted quats


# child class
class PointcloudQuaternionDiffusion(PointcloudScalarDiffusion):
    def __init__(
        self,
        pcloud,
        diffusion_scalar=100,
        # method="laplace",
        method="LU",
        num_eigen=None,
        num_integration_steps=1,
    ):

        super().__init__(
            pcloud,
            diffusion_scalar=diffusion_scalar,
            method=method,
            num_eigen=num_eigen,
            num_integration_steps=num_integration_steps,
        )

        self.pcloud = pcloud

    def set_source_quaternion(self, source_vertices, pure_quaternion):
        self.source_vertices = source_vertices
        self.source_pure_quaternions = pure_quaternion

    def set_source_direction(self, source_vertices, direction):
        self.source_vertices = source_vertices
        self.directions = direction

    def diffuse_key_direction_as_tangent_field(self, direction, vertex=None):
        """
        Vector-diffuse a single key direction from one source vertex across
        the whole pointcloud, as a raw tangent-direction field -- not a full
        rotation. Deliberately skips the quaternion Lie-algebra encoding
        (set_source_direction/set_pure_quaternions_from_directions): that
        machinery is for combining/averaging multiple, possibly disagreeing
        keypoint orientations, and for a single source it just adds a log/exp
        round trip. Diffusing the raw vector directly with
        set_source_quaternion + diffuse_quaternions gives the same magnitude-
        corrected vector-heat-method result more directly.

        direction is projected onto the tangent plane at the source vertex
        (via that vertex's estimated normal) before being used as the
        diffusion seed -- so the seed itself starts genuinely tangential
        there, rather than relying solely on the final per-vertex
        Gram-Schmidt step (Pointcloud.get_bases_from_tangent_vector_and_normal)
        to clean up whatever normal-direction component the raw direction
        happened to carry into the diffusion.

        The result is meant to be combined with the pointcloud's own local
        surface normals (e.g. via Pointcloud.get_bases_from_tangent_vector_and_normal)
        rather than exponentiated into a rotation on its own -- diffusing a
        full orientation field this way ignores the true local surface
        normal entirely, so the result is nearly uniform except very close to
        the source (that's the ambient Lie-algebra diffusion homogenizing,
        not the surface's actual curvature).
        """
        if vertex is None:
            vertex = self.get_farthest_from_boundary_vertex()
        if not hasattr(self.pcloud, "normals"):
            self.pcloud.get_normals()

        direction = np.asarray(direction, dtype=float)
        normal = self.pcloud.normals[vertex]
        tangent_direction = direction - np.dot(direction, normal) * normal
        tangent_norm = np.linalg.norm(tangent_direction)
        if tangent_norm < 1e-9:
            raise ValueError(
                "diffuse_key_direction_as_tangent_field: key_direction is "
                "parallel to the surface normal at the source vertex -- no "
                "tangential component left to diffuse"
            )
        direction = tangent_direction / tangent_norm
        self.set_source_quaternion([vertex], direction[np.newaxis, :])
        self.diffuse_quaternions()
        return self.diffused_pure_quaternions

    def diffuse_quaternions(self):
        if not hasattr(self, "source_pure_quaternions"):
            self.get_random_source_vertices()
        # In the Euclidean space we can diffuse the scalar components independently
        # then combine to a vector field
        vf = np.zeros((len(self.pcloud.vertices), 3))  # Final vector field
        for i in range(3):
            # Consider each component of the vector field separately as a scalar field
            u0 = np.zeros(len(self.pcloud.vertices))
            for j, vertex in enumerate(self.source_vertices):
                # mind the i,j ordering
                u0[vertex] = self.source_pure_quaternions[j, i]
            vf[:, i] = self.integrate_diffusion(u0)

        # Note that the vector diffusion finds the correct direction but changes
        # the magnitude of the vectors. Diffuse the magnitudes seperately to
        # recover the correct magnitudes after diffusion
        u0 = np.zeros(len(self.pcloud.vertices))
        for i, vertex in enumerate(self.source_vertices):
            u0[vertex] = np.linalg.norm(self.source_pure_quaternions[i, :])
        uf = self.integrate_diffusion(u0)

        phi_0 = np.zeros(len(self.pcloud.vertices))
        for i, vertex in enumerate(self.source_vertices):
            phi_0[vertex] = 1
        # Solve the linear system
        phi_f = self.integrate_diffusion(phi_0)

        # Normalize the vector field
        vf = vf / np.linalg.norm(vf, axis=1)[:, None]
        vf = vf * uf[:, None] / phi_f[:, None]
        self.diffused_pure_quaternions = vf

    def steady_state_diffuse_quaternions(self):
        if not hasattr(self, "source_pure_quaternions"):
            self.get_random_source_vertices()
        self.pcloud.boundary_points = self.source_vertices

        vf = np.zeros((len(self.pcloud.vertices), 3))

        for i in range(3):
            u0 = np.zeros(len(self.pcloud.vertices))

            for j, vertex in enumerate(self.source_vertices):
                u0[vertex] = self.source_pure_quaternions[j, i]

            vf[:, i] = self.integrate_diffusion(u0)

        self.diffused_pure_quaternions = vf

    def get_diffused_rotors(self):
        if not hasattr(self, "diffused_pure_quaternions"):
            raise AttributeError(
                "diffused_pure_quaternions is not available yet; call "
                "diffuse_quaternions() or steady_state_diffuse_quaternions() first."
            )
        return expquat_2_rotor(self.diffused_pure_quaternions)

    def set_pure_quaternions_from_directions(self):
        x_vector = np.array([1, 0, 0])  # global x axis
        source_pure_quaternions = np.zeros_like(self.directions)
        for i in range(len(self.directions)):
            quaternion = get_quat_between_vectors(
                x_vector, self.directions[i, :])
            source_pure_quaternions[i, :] = quat_log(
                quaternion
            )
        self.source_pure_quaternions = source_pure_quaternions
