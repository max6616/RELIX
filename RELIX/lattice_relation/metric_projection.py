"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from cctbx import uctbx, sgtbx

from cctbx.sgtbx import lattice_symmetry

from lattice_aligned.candidates import decode_basis

from multitask_indexer.geometry import proper_rotation

from .full_inference import bravais_from_space_group

def decode_measured_basis(basis):
    """Retry numerical reduction tolerances without changing the lattice."""
    from cctbx.uctbx.reduction_base import iteration_limit_exceeded
    attempts = []
    for epsilon in (1e-4, 1e-5, 1e-6, 1e-8):
        try:
            result = decode_basis(basis, niggli_epsilon=epsilon)
            result['reduction_epsilon'] = epsilon
            result['reduction_retries'] = attempts
            return result
        except iteration_limit_exceeded as exc:
            attempts.append(dict(epsilon=epsilon, error=str(exc)))
    raise iteration_limit_exceeded('All measured-basis reduction tolerances exhausted: '+str(attempts))

def project_measured_symmetry(basis, space_group):
    frame = decode_measured_basis(basis)
    reduced = np.linalg.inv(frame['matrix'])
    gram = reduced.T @ reduced
    ev, vec = np.linalg.eigh(gram)
    whitener = (vec/np.sqrt(ev)) @ vec.T
    family = bravais_from_space_group(space_group)
    proposals = []
    seen = set()
    # Le Page groups act in this measured reciprocal frame, including the
    # alternative integer bases near a Niggli boundary. Network family selects
    # which of these finite metric hypotheses is eligible for projection.
    for delta in (.001, .003, .01, .03, .1, .3, 1.):
        group = lattice_symmetry.group(uctbx.unit_cell(tuple(frame['cell'])), max_delta=delta)
        key = tuple(sorted(str(op) for op in group))
        if key in seen:continue
        seen.add(key)
        number = sgtbx.space_group_info(group=group).type().number()
        if bravais_from_space_group(number) != family:
            continue
        rotations = np.stack([np.asarray(op.r().as_double()).reshape(3, 3) for op in group])
        actions = np.linalg.inv(rotations).transpose(0, 2, 1)
        metric = (actions.transpose(0, 2, 1) @ gram @ actions).mean(0)
        strain = float(np.max(abs(np.linalg.eigvalsh(whitener @ (metric-gram) @ whitener))))
        proposals.append((len(group), strain, delta, metric))
    if not proposals:
        frame['prediction_projection'] = dict(space_group=int(space_group), family=family,
            metric_strain=0., applied=False, reason='no_measured_metric_group_matches_neural_family')
        return frame
    order, strain, delta, metric = min(proposals, key=lambda p:(-p[0], p[1]))
    target = np.linalg.cholesky(metric).T
    orientation = proper_rotation(reduced @ np.linalg.inv(target))
    result = decode_measured_basis(orientation @ target)
    result['prediction_projection'] = dict(space_group=int(space_group), family=family, metric_strain=strain,
        applied=True, max_delta=delta, group_order=order)
    return result
