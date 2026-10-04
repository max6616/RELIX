"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

def orthogonalization(cell):
    """Upper-triangular direct basis, matching the CCTBX convention."""
    a, b, c, alpha, beta, gamma = np.asarray(cell, dtype=np.float64)
    if not np.isfinite(cell).all() or min(a, b, c) <= 0:
        raise ValueError("invalid_cell")
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sg = np.sin(np.deg2rad(gamma))
    if sg <= 0:
        raise ValueError("invalid_cell_angles")
    yz = (ca - cb * cg) / sg
    zz = 1.0 - cb * cb - yz * yz
    if zz <= 0:
        raise ValueError("nonpositive_cell_volume")
    return np.array([[a, b * cg, c * cb], [0., b * sg, c * yz],
                     [0., 0., c * np.sqrt(zz)]])

def reciprocal_basis(cell):
    return np.linalg.inv(orthogonalization(cell)).T

def proper_rotation(matrix):
    left, _, right = np.linalg.svd(matrix)
    left[:, -1] *= np.linalg.det(left @ right)
    return left @ right

def lex_less(a, b):
    return ((a[..., 0] < b[..., 0]) |
            ((a[..., 0] == b[..., 0]) & (a[..., 1] < b[..., 1])) |
            ((a[..., 0] == b[..., 0]) & (a[..., 1] == b[..., 1]) & (a[..., 2] < b[..., 2])))

def family(h, actions):
    """Lexicographically smallest proper-group/Friedel orbit; order retained."""
    h = np.asarray(h, dtype=np.int64)
    result = h.copy()
    for A in np.asarray(actions, dtype=np.int64):
        transformed = h @ A.T
        for candidate in (transformed, -transformed):
            mask = lex_less(candidate, result)
            result = np.where(mask[..., None], candidate, result)
    return result
