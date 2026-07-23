"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Coercion helpers so orientation-bearing constructor arguments, attributes,
and return values can be passed/read as either plain numpy rotation
matrices, scipy Rotation objects, or gafropy Rotor objects, without every
call site needing its own isinstance checks.

Two directions:
- `as_scipy_rotation`: normalizes whatever is passed *in* (matrix,
  gafropy.Rotor, gafropy.numpy.Rotor, or an existing scipy Rotation) into
  a scipy Rotation, so the rest of the codebase keeps using the
  scipy-based methods (`.as_matrix()`, `.as_euler()`) it already relies on.
- `as_gafropy_rotor`: converts whatever comes *out* of the geometry
  (matrices, batches of matrices, scipy Rotation) into a
  `gafropy.numpy.Rotor`, so callers who want the gafro-native type can get
  one without hand-rolling the quaternion hop themselves.

Note gafropy.Rotor's own quaternion methods are asymmetric in convention:
`.from_quaternion(w, x, y, z)` takes Eigen's constructor order (scalar
first), while `.to_quaternion()` returns `[x, y, z, w]` (scipy's array
order) -- and scipy's `.as_quat()` also returns `[x, y, z, w]`. So
`as_scipy_rotation` (which only ever calls `.to_quaternion()`) needs no
reordering, but `as_gafropy_rotor` (which calls `.from_quaternion()`) does.

Matrix inputs go through `gafropy.numpy.Rotor.from_matrix` directly (both
single and batched) -- scipy is not involved in that path at all anymore.
`as_scipy_rotation` still exists for the reverse direction (producing a
scipy Rotation for the rest of the codebase's `.as_matrix()`/`.as_euler()`
calls) and for accepting a scipy Rotation as an alternate input.
"""

import gafropy as ga
import gafropy.numpy as gn
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

_XYZW_TO_WXYZ = [3, 0, 1, 2]


def as_scipy_rotation(value):
    """Coerce a rotation-like value to a scipy Rotation.

    Accepts a scipy Rotation (returned as-is), a gafropy.Rotor, a
    gafropy.numpy.Rotor of shape (), or a (3, 3) / (N, 3, 3) matrix.
    """
    if isinstance(value, ScipyRotation):
        return value
    if isinstance(value, gn.Rotor):
        return ScipyRotation.from_quat(value.to_quaternion())
    if isinstance(value, ga.Rotor):
        return ScipyRotation.from_quat(value.to_quaternion())
    return ScipyRotation.from_matrix(np.asarray(value, dtype=np.float64))


def as_gafropy_rotor(value):
    """Coerce a rotation-like value to a gafropy.numpy.Rotor.

    Accepts an existing gafropy.numpy.Rotor (returned as-is), a single
    gafropy.Rotor, a scipy Rotation, or a (3, 3) / (N, 3, 3) matrix
    (or anything array-like coercible to one).
    """
    if isinstance(value, gn.Rotor):
        return value
    if isinstance(value, ga.Rotor):
        return gn.Rotor(value.to_array())
    if isinstance(value, ScipyRotation):
        return gn.Rotor.from_quaternion(value.as_quat()[..., _XYZW_TO_WXYZ])
    # gafropy.numpy.Rotor.from_matrix handles both a single (3, 3) matrix
    # and a batch (N, 3, 3) natively (Shepperd's method, vectorized) --
    # no scipy needed for either case.
    return gn.Rotor.from_matrix(np.asarray(value, dtype=np.float64))
