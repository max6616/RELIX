"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import itertools

import math

from fractions import Fraction

import numpy as np

from scipy.spatial import cKDTree

CONVENTION = 'primitive-niggli-lattice-closest-identity-v1'

def reciprocal_observations(raw, geometry):
    """Scattering vector / 2π in Å⁻¹, derotated to the scan's zero-angle frame."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Observations must be a nonempty finite N x 3 array of x_px, y_px, angle_deg")
    for key, shape in (("center_px", (2,)), ("pixel_mm", (2,)), ("rotation_axis", (3,))):
        value = np.asarray(geometry[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid calibration {key}")
    if (not np.isfinite([geometry["distance_mm"], geometry["wavelength_A"]]).all()
            or geometry["distance_mm"] <= 0 or geometry["wavelength_A"] <= 0
            or np.min(geometry["pixel_mm"]) <= 0 or np.linalg.norm(geometry["rotation_axis"]) <= 0):
        raise ValueError("Invalid detector calibration")
    ray = np.column_stack([(raw[:, :2] - geometry["center_px"]) * geometry["pixel_mm"],
                           np.full(len(raw), geometry["distance_mm"])])
    direction = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    q = (direction - np.array([0., 0., 1.])) / geometry["wavelength_A"]
    axis = np.asarray(geometry["rotation_axis"]) / np.linalg.norm(geometry["rotation_axis"])
    phi = np.deg2rad(-raw[:, 2])[:, None]
    return (q * np.cos(phi) + np.cross(axis, q) * np.sin(phi)
            + axis * (q @ axis)[:, None] * (1 - np.cos(phi)))

def observed_lattice_vectors(q, count=24):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) < 12 or not np.isfinite(q).all():
        raise ValueError('Need twelve finite reciprocal observations')
    _, ids = np.unique(np.rint(q/1e-4).astype(np.int64), axis=0, return_index=True)
    distinct = q[ids]
    tree = cKDTree(distinct)
    for k in (32, 96, 256):
        _, neighbors = tree.query(distinct, k=min(k, len(distinct)))
        delta = (distinct[neighbors[:, 1:]] - distinct[:, None]).reshape(-1, 3)
        lengths = np.linalg.norm(delta, axis=1)
        delta = delta[lengths > 1e-4]
        signs = np.sign(delta[np.arange(len(delta)), np.abs(delta).argmax(1)])
        delta *= signs[:, None]
        keys, inverse = np.unique(np.rint(delta/1e-4).astype(np.int64), axis=0, return_inverse=True)
        sums = np.zeros((len(keys), 3)); np.add.at(sums, inverse, delta)
        means = sums / np.bincount(inverse)[:, None]
        lengths = np.linalg.norm(means, axis=1)
        order = np.argsort(lengths, kind='stable')
        vectors = []; units = []
        for index in order:
            unit = means[index]/lengths[index]
            if units and np.min(np.linalg.norm(np.array(units)-unit, axis=1)) < .001:
                continue
            vectors.append(means[index]); units.append(unit)
            if len(vectors) >= count and np.linalg.matrix_rank(vectors, tol=1e-4) == 3:
                break
        if len(vectors) >= 3 and np.linalg.matrix_rank(vectors, tol=1e-4) == 3:
            break
    else:
        # A very anisotropic lattice can have hundreds of neighbors in one
        # plane. Add the shortest observed global differences that supply the
        # missing span, instead of pretending the scan itself is two-dimensional.
        global_difference=distinct-distinct[0]
        global_length=np.linalg.norm(global_difference,axis=1)
        for _ in range(3):
            array=np.asarray(vectors)
            _,singular,right=np.linalg.svd(array,full_matrices=False)
            scale=np.linalg.norm(array,axis=1).min()
            rank=int((singular>scale*.001).sum())
            if rank==3:break
            residual=np.linalg.norm(global_difference@right[rank:].T,axis=1)
            available=np.flatnonzero(residual>scale*.01)
            if not len(available):raise ValueError('Observed differences do not span three dimensions')
            index=available[np.argmin(global_length[available])]
            vectors.append(global_difference[index])
        if np.linalg.matrix_rank(vectors,tol=scale*.001)<3:
            raise ValueError('Observed differences do not span three dimensions')
    vectors = np.asarray(vectors)
    spacing = np.linalg.norm(vectors, axis=1).min()
    if len(vectors)>count and np.linalg.matrix_rank(vectors[:count],tol=spacing*.001)<3:
        anchors=[]
        for i in range(len(vectors)):
            if np.linalg.matrix_rank(vectors[anchors+[i]],tol=spacing*.001)>len(anchors):
                anchors.append(i)
            if len(anchors)==3:break
        selected=list(range(count))
        for anchor in anchors:
            if anchor not in selected:
                replace=next(i for i in reversed(range(count)) if selected[i] not in anchors)
                selected[replace]=anchor
        vectors=vectors[selected]
    # Repeat observed vectors only if a very small scan supplies fewer directions.
    vectors = vectors[np.arange(count) % len(vectors)]
    if np.linalg.matrix_rank(vectors,tol=spacing*.001)<3:
        raise ValueError('The retained local vectors do not span three dimensions')
    return np.concatenate([vectors, -vectors])/spacing, float(spacing)

def canonical_primitive(crystal_symmetry, reciprocal_matrix, max_delta=.001):
    """Return one right-handed primitive Niggli basis and its exact HKL map.

    reciprocal_matrix maps column HKL to q (without 2*pi). The metric's proper
    lattice symmetry resolves equivalent orientations; it can exceed the actual
    structure symmetry, so this changes the structural setting as well as HKL.
    """
    from cctbx.sgtbx import lattice_symmetry
    source = np.asarray(reciprocal_matrix, dtype=float)
    cb = crystal_symmetry.change_of_basis_op_to_niggli_cell()
    reduced = crystal_symmetry.change_basis(cb)
    transform = np.asarray(cb.c_inv().r().as_double()).reshape(3, 3).T
    if np.linalg.det(source @ np.linalg.inv(transform)) < 0:
        transform = -transform
    A = source @ np.linalg.inv(transform)
    O = np.asarray(reduced.unit_cell().orthogonalization_matrix()).reshape(3, 3)
    U = A @ O.T
    if np.max(np.abs(U.T @ U - np.eye(3))) > 1e-4 or np.linalg.det(U) < .999:
        raise ValueError('Reciprocal matrix does not match the supplied unit-cell metric')
    operations = lattice_symmetry.group(reduced.unit_cell(), max_delta=max_delta)
    candidates = []
    for operation in operations:
        R = np.asarray(operation.r().as_double()).reshape(3, 3)
        if np.linalg.det(R) < .5:
            continue
        P = O @ R @ np.linalg.inv(O)
        if np.max(np.abs(P.T @ P - np.eye(3))) > 1e-4:
            raise ValueError('Approximate metric operation exceeds the numerical tolerance')
        candidate = U @ P
        key = (-round(float(np.trace(candidate)), 10), tuple(np.round(candidate.flat, 10)))
        candidates.append((key, candidate, R.T @ transform))
    _, orientation, total = min(candidates, key=lambda value: value[0])
    matrix = total @ np.linalg.inv(source)
    return dict(convention=CONVENTION, cell=np.array(reduced.unit_cell().parameters()),
                orientation=orientation, indexing_matrix=matrix, hkl_transform=total,
                fractional_site_transform=np.linalg.inv(total).T,
                proper_metric_operations=len(candidates), cctbx_niggli_change_of_basis=str(cb))

def cell_from_indexing_matrix(matrix):
    lengths=np.linalg.norm(matrix,axis=1)
    cosines=np.array([matrix[1]@matrix[2]/lengths[1]/lengths[2],
                      matrix[0]@matrix[2]/lengths[0]/lengths[2],matrix[0]@matrix[1]/lengths[0]/lengths[1]])
    return np.r_[lengths,np.rad2deg(np.arccos(np.clip(cosines,-1,1)))]

def observed_primitive_matrix(q):
    """Return an integer-generating reciprocal basis using only measured q.

    Start from three short independent observed differences. If a retained q is
    in a missing rational coset, extend the lattice by a Hermite normal form of
    [d I | d c], where c is that reflection in the current basis. Least squares
    finally removes centroid rounding error, using assigned integers only.
    """
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form
    q=np.asarray(q,dtype=float)
    vectors,spacing=observed_lattice_vectors(q)
    vectors=vectors[:len(vectors)//2]*spacing
    candidates=[]
    for indices in itertools.combinations(range(len(vectors)),3):
        basis=vectors[list(indices)].T
        volume=np.linalg.det(basis)
        relative=abs(volume)/np.prod(np.linalg.norm(basis,axis=0))
        if relative>.025:
            candidates.append((abs(volume),np.linalg.cond(basis),indices))
    if not candidates:raise ValueError('No well-spanning observed difference basis')
    smallest=min(c[0] for c in candidates)
    _,_,indices=min((c for c in candidates if c[0]<smallest*1.001),key=lambda c:c[1])
    basis=vectors[list(indices)].T
    if np.linalg.det(basis)<0:basis[:,0]*=-1
    extensions=[]
    order=np.argsort(np.linalg.norm(q,axis=1),kind='stable')
    for iteration in range(12):
        continuous=np.linalg.solve(basis,q.T).T
        error=np.max(np.abs(continuous-np.rint(continuous)),axis=1)
        if error.max()<.003:break
        chosen=None
        for index in order[error[order]>.003][:128]:
            fractions=[Fraction(float(c)).limit_denominator(96) for c in continuous[index]]
            denominator=math.lcm(*(f.denominator for f in fractions))
            approximation=np.array([float(f) for f in fractions])
            if 1<denominator<=256 and np.max(np.abs(approximation-continuous[index]))<1e-4:
                chosen=(index,denominator,approximation);break
        if chosen is None:raise ValueError('Observed vectors do not admit a stable small-denominator lattice extension')
        index,denominator,approximation=chosen
        integer=np.column_stack([np.eye(3,dtype=np.int64)*denominator,np.rint(approximation*denominator).astype(np.int64)])
        hnf=np.array(hermite_normal_form(Matrix(integer.tolist()))).astype(float)
        basis=basis@hnf/denominator
        extensions.append(dict(denominator=denominator,relative_volume=float(np.linalg.det(hnf)/denominator**3)))
    else:raise ValueError('Integer lattice did not stabilize within twelve extensions')
    assigned=np.rint(np.linalg.solve(basis,q.T).T)
    matrix=np.linalg.lstsq(q,assigned,rcond=None)[0].T
    final=q@matrix.T
    if np.max(np.abs(final-np.rint(final)))>.003:raise ValueError('Final lattice fit does not explain every observation')
    if np.linalg.det(matrix)<=0:raise ValueError('Fitted lattice lost handedness')
    return matrix,dict(extensions=extensions,integer_fit_max_error=float(np.max(np.abs(final-np.rint(final)))))

def index_integer_lattice(q):
    from cctbx import crystal
    matrix,details=observed_primitive_matrix(q)
    symmetry=crystal.symmetry(unit_cell=tuple(cell_from_indexing_matrix(matrix)),space_group_symbol='P 1')
    canonical=canonical_primitive(symmetry,np.linalg.inv(matrix))
    matrix=canonical['indexing_matrix'];continuous=q@matrix.T;integer=np.rint(continuous).astype(np.int64)
    return dict(hkl=integer,continuous_hkl=continuous,indexing_matrix=matrix,cell=canonical['cell'],
                fractional_hkl_rms=np.array(np.sqrt(np.mean((continuous-integer)**2))),
                observed_integer_inlier_fraction=np.array((np.max(np.abs(continuous-integer),axis=1)<.1).mean()),
                row_count=np.array(len(q))),details
