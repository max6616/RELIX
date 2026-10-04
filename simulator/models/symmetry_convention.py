"""Diffraction orientation gauge for this simulator's explicit Friedel approximation.

Use the actual CIF setting, rather than a point-group symbol's reference axes.
The returned fractional operation R and Cartesian operation P=O R O^-1 obey
U' = U P and h' = R^T h, preserving U B h for B=O^-T.
"""
import numpy as np

CONVENTION = "laue-fz-cif-setting-v1"


def proper_laue_operations(crystal_symmetry):
    O = np.asarray(crystal_symmetry.unit_cell().orthogonalization_matrix()).reshape(3, 3)
    inverse = np.linalg.inv(O)
    fractional = {}
    for op in crystal_symmetry.space_group():
        R = np.asarray(op.r().as_double()).reshape(3, 3)
        # The simulator explicitly duplicates Friedel intensities, so include -R.
        for sign in (1, -1):
            candidate = sign * R
            if np.linalg.det(candidate) > .5:
                fractional[tuple(np.rint(candidate).astype(int).flat)] = candidate
    records = []
    for key in sorted(fractional):
        R = fractional[key]
        P = O @ R @ inverse
        if np.max(np.abs(P.T @ P - np.eye(3))) > 1e-6:
            raise ValueError("CIF symmetry operation is incompatible with its unit-cell metric")
        records.append(dict(cartesian=P.tolist(), fractional=R.astype(int).tolist()))
    return records


def canonicalize_orientation(U, operations):
    U = np.asarray(U, dtype=float)
    candidates = []
    for operation in operations:
        P = np.asarray(operation["cartesian"])
        candidate = U @ P
        key = (round(float(np.linalg.norm(candidate - np.eye(3))), 12),
               tuple(np.round(candidate.ravel(), 12)))
        candidates.append((key, candidate, np.asarray(operation["fractional"], dtype=int)))
    if not candidates:
        raise ValueError("No proper Laue operations")
    _, selected, fractional = min(candidates, key=lambda item: item[0])
    return selected, fractional.T
