"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from scipy.spatial import cKDTree

def relation_tokens(q, intensity, count=512):
    q = np.asarray(q, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if len(q) < 12 or not np.isfinite(q).all() or not np.isfinite(intensity).all():
        raise ValueError('insufficient_or_invalid_observations')
    # Deterministic coverage and order independence. No reference values enter.
    order = np.lexsort((intensity, q[:, 2], q[:, 1], q[:, 0], np.linalg.norm(q, axis=1)))
    anchors = order[np.linspace(0, len(order)-1, min(128, len(order))).astype(int)]
    distance, neighbor = cKDTree(q).query(q[anchors], k=min(17, len(q)))
    scale = float(np.median(distance[:, min(5, distance.shape[1]-1)]))
    scale = max(scale, 1e-5)
    source = np.repeat(anchors, neighbor.shape[1]-1)
    target = neighbor[:, 1:].reshape(-1)
    # Include measured origin-to-reflection vectors, to retain weak generators.
    radial = order[np.linspace(0, len(order)-1, min(128, len(order))).astype(int)]
    source = np.r_[source, np.full(len(radial), -1)]
    target = np.r_[target, radial]
    a = np.where(source[:, None] >= 0, q[np.maximum(source, 0)], 0.)
    b = q[target]
    delta = b-a
    norm = np.linalg.norm(delta, axis=1)
    keep = norm > max(scale*.02, 1e-6)
    source, target, a, b, delta, norm = [v[keep] for v in (source, target, a, b, delta, norm)]
    sign = np.sign(delta[np.arange(len(delta)), np.abs(delta).argmax(1)])
    delta *= sign[:, None]
    # Uniform coverage of the raw relation list; no frequency vote or basis bank.
    if len(delta) < 3:
        raise ValueError('insufficient_relations')
    ix = np.linspace(0, len(delta)-1, min(count, len(delta))).astype(int)
    source, target, a, b, delta, norm, sign = [v[ix] for v in (source, target, a, b, delta, norm, sign)]
    radius = max(float(np.quantile(np.linalg.norm(q, axis=1), .9)), 1e-5)
    ia = np.where(source >= 0, intensity[np.maximum(source, 0)], 0.)
    ib = intensity[target]
    features = np.c_[delta/scale, a/radius, b/radius, ia, ib,
                     np.log1p(norm/scale), np.full(len(ix), np.log(scale)),
                     np.full(len(ix), np.log(radius)), np.full(len(ix), np.log1p(len(q))),
                     source < 0].astype(np.float32)
    n = len(features)
    mask = np.arange(count) < n
    features = np.pad(features, ((0, count-n), (0, 0)))
    delta = np.pad(delta, ((0, count-n), (0, 0)))
    return dict(features=features, vectors=delta.astype(np.float32), mask=mask,
                source=np.pad(source, (0, count-n), constant_values=-1),
                target=np.pad(target, (0, count-n), constant_values=-1),
                sign=np.pad(sign, (0, count-n)), scale=scale)
