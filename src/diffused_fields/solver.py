import numpy as np
import potpourri3d as pp3d
import robust_laplacian
from pcdiff import build_grad_div, estimate_basis, knn_graph
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .field import QuaternionField, ScalarField, VectorField
from .manifold._rotor_interop import as_gafropy_rotor


def farthest_from_boundary_vertex(manifold):
    if not hasattr(manifold, "is_boundary_arr"):
        manifold.get_boundary()
    boundary_vertices = list(np.where(manifold.is_boundary_arr)[0])
    if not boundary_vertices:
        raise ValueError("farthest_from_boundary_vertex: manifold has no boundary")
    geodesic = pp3d.PointCloudHeatSolver(manifold.vertices).compute_distance_multisource(boundary_vertices)
    return int(np.argmax(geodesic))


def compute_mean_rotor(rotors, weights=None):
    quaternions = as_gafropy_rotor(rotors).to_quaternion()
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]

    n = quaternions.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights)

    accumulator = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean_quaternion = eigenvectors[:, np.argmax(eigenvalues)]
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    if mean_quaternion[3] < 0:
        mean_quaternion *= -1

    import gafropy.numpy as gn
    return gn.Rotor.from_quaternion(mean_quaternion[[3, 0, 1, 2]])


def batch_sample_random_direction(batch_size):
    direction = np.random.randn(batch_size, 3)
    return direction / np.linalg.norm(direction, axis=1)[:, None]


class Solver:
    def solve_scalar(self, problem):
        raise NotImplementedError

    def solve_vector(self, problem):
        raise NotImplementedError

    def solve_quaternion(self, problem):
        raise NotImplementedError

    def evaluate_scalar_at(self, problem, position):
        raise NotImplementedError

    def evaluate_vector_at(self, problem, position):
        raise NotImplementedError

    def evaluate_quaternion_at(self, problem, position):
        raise NotImplementedError


class LaplacianSolver(Solver):
    def __init__(self, diffusion_scalar=1.0):
        self.diffusion_scalar = diffusion_scalar
        self._laplacian = {}
        self._factorization = {}
        self._gradient_operator = {}

    def _laplacian_matrices(self, manifold):
        key = id(manifold)
        if key not in self._laplacian:
            self._laplacian[key] = robust_laplacian.point_cloud_laplacian(manifold.vertices)
        return self._laplacian[key]

    def _factorized_system(self, manifold):
        key = id(manifold)
        if key not in self._factorization:
            C, M = self._laplacian_matrices(manifold)
            if not hasattr(manifold, "mean_edge_length"):
                manifold.get_mean_edge_length()
            dt = self.diffusion_scalar * manifold.mean_edge_length**2
            self._factorization[key] = splu(csc_matrix(M + dt * C))
        return self._factorization[key]

    def _integrate(self, manifold, u0):
        _, M = self._laplacian_matrices(manifold)
        return self._factorized_system(manifold).solve(M @ u0)

    def _gradient_operator_for(self, manifold):
        key = id(manifold)
        if key not in self._gradient_operator:
            edge_index = knn_graph(manifold.vertices, 20)
            normal, x_basis, y_basis = estimate_basis(manifold.vertices, edge_index, k=20)
            grad, _ = build_grad_div(manifold.vertices, normal, x_basis, y_basis, edge_index)
            self._gradient_operator[key] = (grad, x_basis, y_basis)
        return self._gradient_operator[key]

    def _tangent_direction(self, problem):
        manifold = problem.manifold
        ut = self._integrate(manifold, problem.initial_values())
        grad, x_basis, y_basis = self._gradient_operator_for(manifold)
        gradient = (grad @ ut).reshape(-1, 2)
        tangent = gradient[:, 0:1] * x_basis + gradient[:, 1:] * y_basis
        return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)

    def solve_scalar(self, problem):
        values = self._integrate(problem.manifold, problem.initial_values())
        return ScalarField(problem, self, values=values)

    def solve_vector(self, problem):
        manifold = problem.manifold
        u0 = problem.initial_values(dim=3)
        vf = np.stack([self._integrate(manifold, u0[:, i]) for i in range(3)], axis=1)

        magnitude_u0 = np.zeros(len(manifold.vertices))
        magnitude_u0[problem.source_vertices] = np.linalg.norm(np.atleast_2d(u0[problem.source_vertices]), axis=1)
        magnitude = self._integrate(manifold, magnitude_u0)

        indicator_u0 = np.zeros(len(manifold.vertices))
        indicator_u0[problem.source_vertices] = 1.0
        indicator = self._integrate(manifold, indicator_u0)

        vf = vf / np.linalg.norm(vf, axis=1, keepdims=True)
        vf = vf * (magnitude / indicator)[:, None]
        return VectorField(problem, self, values=vf)

    def solve_quaternion(self, problem):
        manifold = problem.manifold
        tangent = self._tangent_direction(problem)
        if not hasattr(manifold, "normals"):
            manifold.get_normals()
        manifold.get_bases_from_tangent_vector_and_normal(tangent)
        return QuaternionField(problem, self, values=as_gafropy_rotor(manifold.local_bases))

    def evaluate_scalar_at(self, problem, position):
        return self.solve_scalar(problem)(position)

    def evaluate_vector_at(self, problem, position):
        return self.solve_vector(problem)(position)

    def evaluate_quaternion_at(self, problem, position):
        return self.solve_quaternion(problem)(position)


class WalkOnSpheresSolver(Solver):
    def __init__(self, batch_size=512, max_iterations=24, convergence_threshold=1e-3, divergence_threshold=10.0):
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold

    def _walk(self, manifold, points):
        status = np.full(points.shape[0], "walking", dtype=object)
        min_distances = np.full(points.shape[0], np.inf)
        min_indices = np.zeros(points.shape[0], dtype=int)

        for _ in range(1, self.max_iterations):
            walking = np.where(status == "walking")[0]
            if len(walking) == 0:
                break
            distances, indices = manifold.get_closest_points(points[walking])
            update = distances < min_distances[walking]
            to_update = walking[update]
            min_distances[to_update] = distances[update]
            min_indices[to_update] = indices[update]

            status[min_distances < self.convergence_threshold] = "converged"
            status[min_distances > self.divergence_threshold] = "diverged"

            direction = batch_sample_random_direction(len(points))
            points[walking] += min_distances[walking, None] * direction[walking]

        status[status == "walking"] = "diverged"
        return min_indices, status

    def _batch_from(self, position):
        return np.tile(position, (self.batch_size, 1)).astype(float)

    def solve_scalar(self, problem):
        return ScalarField(problem, self)

    def solve_vector(self, problem):
        return VectorField(problem, self)

    def solve_quaternion(self, problem):
        return QuaternionField(problem, self)

    def _converged_indices(self, problem, position):
        indices, status = self._walk(problem.manifold, self._batch_from(position))
        converged = indices[status == "converged"]
        if len(converged) == 0:
            raise ValueError("WalkOnSpheresSolver: no walkers converged from this position")
        return converged

    def evaluate_scalar_at(self, problem, position):
        converged = self._converged_indices(problem, position)
        u0 = problem.initial_values()
        return float(np.mean(u0[converged]))

    def evaluate_vector_at(self, problem, position):
        converged = self._converged_indices(problem, position)
        vectors = problem.manifold.normals[converged]
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-10 else mean

    def evaluate_quaternion_at(self, problem, position):
        converged = self._converged_indices(problem, position)
        rotors = as_gafropy_rotor(problem.manifold.local_bases[converged])
        return compute_mean_rotor(rotors)
