import numpy as np


class Problem:
    def __init__(self, manifold):
        self.manifold = manifold


class DiffusionProblem(Problem):
    def __init__(self, manifold, source_vertices=None, source_values=None, diffusion_time=1.0):
        super().__init__(manifold)
        self.source_vertices = None if source_vertices is None else np.asarray(source_vertices)
        self.source_values = None if source_values is None else np.asarray(source_values)
        self.diffusion_time = diffusion_time

    def initial_values(self, dim=1):
        if self.source_vertices is None:
            raise ValueError("DiffusionProblem.initial_values: source_vertices not set")
        values = self.source_values
        if values is None:
            values = np.ones((len(self.source_vertices), dim)) if dim > 1 else np.ones(len(self.source_vertices))
        u0 = np.zeros((len(self.manifold.vertices), dim)) if dim > 1 else np.zeros(len(self.manifold.vertices))
        u0[self.source_vertices] = values
        return u0
