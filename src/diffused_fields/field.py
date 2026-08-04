import numpy as np


class Field:
    def __init__(self, problem, solver, values=None):
        self.problem = problem
        self.solver = solver
        self.values = values

    @property
    def manifold(self):
        return self.problem.manifold

    def _nearest_value(self, position):
        _, index = self.manifold.get_closest_points(np.atleast_2d(position))
        return self.values[index[0]]


class ScalarField(Field):
    def __call__(self, position):
        if self.values is not None:
            return self._nearest_value(position)
        return self.solver.evaluate_scalar_at(self.problem, position)


class VectorField(Field):
    def __call__(self, position):
        if self.values is not None:
            return self._nearest_value(position)
        return self.solver.evaluate_vector_at(self.problem, position)


class QuaternionField(Field):
    def __call__(self, position):
        if self.values is not None:
            return self._nearest_value(position)
        return self.solver.evaluate_quaternion_at(self.problem, position)
