"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from scipy.spatial import cKDTree

def dense_relation_tokens(q, intensity, count=512, anchors=1024, neighbors=24, bin_fraction=.03):
    q = np.asarray(q, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if len(q) < 12 or not np.isfinite(q).all() or not np.isfinite(intensity).all():
        raise ValueError('insufficient_or_invalid_observations')
    order = np.lexsort((intensity, q[:, 2], q[:, 1], q[:, 0], np.linalg.norm(q, axis=1)))
    selected_anchors = order[np.linspace(0, len(order)-1, min(anchors, len(order))).astype(int)]
    distance, neighbor = cKDTree(q).query(q[selected_anchors], k=min(neighbors+1, len(q)))
    scale = max(float(np.median(distance[:, min(5, distance.shape[1]-1)])), 1e-5)
    source = np.repeat(selected_anchors, neighbor.shape[1]-1)
    target = neighbor[:, 1:].reshape(-1)
    delta = q[target]-q[source]
    length = np.linalg.norm(delta, axis=1)
    valid = length > max(.02*scale, 1e-6)
    source, target, delta, length = [v[valid] for v in (source, target, delta, length)]
    sign = np.sign(delta[np.arange(len(delta)), np.abs(delta).argmax(1)])
    delta *= sign[:, None]
    if not len(delta):
        raise ValueError('insufficient_relations')
    # Preserve a concrete observed pair per occupied bin. Frequency controls
    # sampling only; it is not a crystal classifier or a lattice selector.
    keys, first, inverse, counts = np.unique(np.rint(delta/(bin_fraction*scale)).astype(np.int64),
        axis=0, return_index=True, return_inverse=True, return_counts=True)
    ranked = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], length[first], -counts))
    radial_count = min(count//4, len(q))
    relation_count = min(count-radial_count, len(ranked))
    indices = first[ranked[:relation_count]]
    support = counts[inverse[indices]].astype(float)
    source, target, delta, sign = [v[indices] for v in (source, target, delta, sign)]
    radial = order[np.linspace(0, len(order)-1, radial_count).astype(int)]
    radial_delta = q[radial].copy()
    radial_sign = np.sign(radial_delta[np.arange(len(radial)), np.abs(radial_delta).argmax(1)])
    radial_sign[radial_sign == 0] = 1
    radial_delta *= radial_sign[:, None]
    source = np.r_[source, np.full(len(radial), -1)]
    target = np.r_[target, radial]
    delta = np.concatenate([delta, radial_delta])
    sign = np.r_[sign, radial_sign]
    support = np.r_[support, np.ones(len(radial))]
    a = np.where(source[:, None] >= 0, q[np.maximum(source, 0)], 0.)
    b = q[target]
    radius = max(float(np.quantile(np.linalg.norm(q, axis=1), .9)), 1e-5)
    ia = np.where(source >= 0, intensity[np.maximum(source, 0)], 0.)
    ib = intensity[target]
    features = np.c_[delta/scale, a/radius, b/radius, ia, ib,
        np.log1p(np.linalg.norm(delta, axis=1)/scale), np.full(len(delta), np.log(scale)),
        np.full(len(delta), np.log(radius)), np.full(len(delta), np.log1p(len(q))), source < 0].astype(np.float32)
    n = len(delta)
    return dict(features=np.pad(features, ((0, count-n), (0, 0))),
        vectors=np.pad(delta.astype(np.float32), ((0, count-n), (0, 0))), mask=np.arange(count) < n,
        source=np.pad(source, (0, count-n), constant_values=-1),
        target=np.pad(target, (0, count-n), constant_values=-1), sign=np.pad(sign, (0, count-n)),
        support=np.pad(support, (0, count-n)), scale=scale,
        sampling_stats=dict(anchors=len(selected_anchors), pairs=len(inverse), bins=len(keys),
            repeated_bins=int((counts > 1).sum()), token_relations=relation_count, token_radials=radial_count))
