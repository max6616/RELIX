"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from lattice_aligned.features import catalogue, intensity_statistics

from .orbit_features import describe_orbit, orbit_catalogue, transform_descriptor

PROFILE_TOKENS = 24

PROFILE_WIDTH = 40

def describe_profiles(q, intensity, geometry, *, min_per_shell=None):
    h = np.rint(np.asarray(q) @ geometry['matrix'].T).astype(np.int64)
    unique, inv, counts = np.unique(h, axis=0, return_inverse=True, return_counts=True)
    raw = np.expm1(np.asarray(intensity, float)*np.log(1001))/1000
    merged = np.bincount(inv, weights=raw)/counts
    uq = np.zeros((len(unique), 3)); np.add.at(uq, inv, q); uq /= counts[:, None]
    radius = np.linalg.norm(uq, axis=1)
    e, _ = intensity_statistics(radius, merged, min_per_shell=min_per_shell)
    shell = np.searchsorted(np.quantile(radius, np.linspace(0, 1, 11)[1:-1]), radius, side='right')
    masks = [np.ones(len(e), bool)] + [shell == i for i in range(10)]
    masks += [unique @ normal == 0 for normal in catalogue()[2]]
    profiles = np.zeros((PROFILE_TOKENS, PROFILE_WIDTH), np.float32)
    for i, mask in enumerate(masks):
        values = e[mask]
        if not len(values): continue
        logs = np.log1p(values)
        profiles[i] = np.r_[np.quantile(logs, np.linspace(.01, .99, 31)),
            np.log1p(len(values))/10, mask.mean(), np.log1p(values.mean()),
            np.log1p(np.mean(values**2)), np.log1p(np.mean(values**3)),
            logs.mean(), logs.std(), (values < .5).mean(), (values > 2).mean()]
    assert np.isfinite(profiles).all()
    return profiles

def describe_profile_orbit(q, intensity, geometry):
    return np.r_[describe_orbit(q, intensity, geometry), describe_profiles(q, intensity, geometry).ravel()].astype(np.float32)

@lru_cache(None)
def profile_permutations():
    normals = catalogue()[2]
    lookup = {tuple(n):i for i,n in enumerate(normals)}
    result = []
    for t in orbit_catalogue()[0]:
        indices = list(range(11))
        for normal in normals:
            old = t.T @ normal
            old *= np.sign(old[np.flatnonzero(old)[0]])
            indices.append(11+lookup[tuple(old)])
        result.append(indices)
    return np.asarray(result)

def transform_profile_descriptor(descriptor, index):
    import torch
    base = descriptor[:, :-PROFILE_TOKENS*PROFILE_WIDTH]
    profile = descriptor[:, -PROFILE_TOKENS*PROFILE_WIDTH:].reshape(-1, PROFILE_TOKENS, PROFILE_WIDTH)
    indices = torch.as_tensor(profile_permutations(), device=descriptor.device)[index]
    profile = profile.gather(1, indices[..., None].expand(-1, -1, PROFILE_WIDTH))
    return torch.cat([transform_descriptor(base, index), profile.flatten(1)], -1)
