"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from lattice_relation.consensus_inference import gaussian_consensus

from robustness.primitive_readout import observed_primitive

def posterior_primitive(basis, q, scale, threshold=.99, minimum_points=12):
    basis, q = np.asarray(basis, float), np.asarray(q, float)
    original = gaussian_consensus(q, basis, scale)
    fraction, variance = original[1], (original[2] * scale) ** 2
    floating = q @ np.linalg.inv(basis).T
    indices = np.rint(floating)
    distance2 = np.square(q - indices @ basis.T).sum(1)
    peak = np.log(abs(np.linalg.det(basis))) - 1.5 * np.log(2 * np.pi * variance) - distance2 / (2 * variance)
    signal = np.log(fraction) + peak
    responsibility = np.exp(signal - np.logaddexp(np.log1p(-fraction), signal))
    keep = (responsibility >= threshold) & (abs(floating - indices).max(1) < .1)
    meta = dict(threshold=threshold, retained=int(keep.sum()), points=len(q),
                initial_score=original[0], initial_fraction=fraction, initial_sigma_relative=original[2])
    if keep.sum() < minimum_points:
        return basis.copy(), meta | dict(applied=False, reason='insufficient_high_posterior_points')
    candidate, primitive = observed_primitive(basis, q, keep)
    meta.update(primitive)
    if not primitive.get('applied'):
        return basis.copy(), meta
    updated = gaussian_consensus(q, candidate, scale)
    meta.update(proposed_score=updated[0], proposed_fraction=updated[1], proposed_sigma_relative=updated[2])
    if updated[0] <= original[0] + 1e-10:
        return basis.copy(), meta | dict(applied=False, reason='no_likelihood_improvement')
    return candidate, meta
