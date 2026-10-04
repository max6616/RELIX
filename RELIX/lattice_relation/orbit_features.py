"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

from itertools import permutations, product

import numpy as np

from lattice_aligned.features import catalogue, describe, intensity_statistics

from multitask_indexer.geometry import reciprocal_basis

@lru_cache(None)
def orbit_catalogue():
    transforms = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            t = np.eye(3, dtype=np.int64)[list(perm)]*np.array(signs)[:, None]
            if round(np.linalg.det(t)) == 1:
                transforms.append(t)
    transforms = np.asarray(transforms)
    old = catalogue()[1]
    actions = np.unique(np.concatenate([t[None]@old@t.T[None] for t in transforms]), axis=0)
    return transforms, actions

def describe_orbit(q, intensity, geometry):
    original = describe(q, intensity, geometry=geometry)
    base = original['descriptor']
    actions = orbit_catalogue()[1]
    old = {tuple(a.ravel()):v for a,v in zip(catalogue()[1], base[1735:].reshape(-1, 6))}
    h = original['hkl']
    unique, inv, counts = np.unique(h, axis=0, return_inverse=True, return_counts=True)
    raw = np.expm1(np.asarray(intensity, float)*np.log(1001))/1000
    merged = np.bincount(inv, weights=raw)/counts
    uq = np.zeros((len(unique), 3)); np.add.at(uq, inv, q); uq /= counts[:, None]
    e, _ = intensity_statistics(np.linalg.norm(uq, axis=1), merged)
    B = reciprocal_basis(geometry['cell']); metric = B.T@B
    span = int(max(abs(unique).max(initial=0), 1)); shift = 8*span+16; radix = 2*shift+1
    def keys(x):
        return ((x[:, 0]+shift)*radix+x[:, 1]+shift)*radix+x[:, 2]+shift
    key = keys(unique); order = np.argsort(key); key = key[order]
    values = []
    for action in actions:
        k = tuple(action.ravel())
        if k in old:
            values.append(old[k]); continue
        target = keys(unique@action.T); found = np.searchsorted(key, target)
        clipped = np.minimum(found, len(key)-1)
        valid = (found < len(key)) & (key[clipped] == target)
        pair = order[clipped[valid]]
        a, b = e[valid], e[pair]
        metric_error = float(np.max(abs(action.T@metric@action-metric))/np.max(abs(metric)))
        if len(a) >= 3:
            residual = float(np.sum(abs(a-b))/max(np.sum(abs(a)+abs(b)), 1e-12))
            correlation = float(np.mean((a-a.mean())*(b-b.mean()))/max(a.std()*b.std(), 1e-12))
            difference = abs(np.log1p(a)-np.log1p(b))
            values.append([np.log1p(metric_error), valid.mean(), correlation, residual, float(difference.mean()), float(np.quantile(difference, .9))])
        else:
            values.append([np.log1p(metric_error), valid.mean(), 0., 1., 1., 1.])
    return np.r_[base[:1735], np.asarray(values).ravel()].astype(np.float32)

@lru_cache(None)
def descriptor_permutations():
    transforms, actions = orbit_catalogue()
    normals = catalogue()[2]
    action_lookup = {tuple(a.ravel()):i for i,a in enumerate(actions)}
    normal_lookup = {tuple(n):i for i,n in enumerate(normals)}
    columns = 1735+6*len(actions)
    indices = []
    for t in transforms:
        ix = np.arange(columns)
        cursor = 75
        for modulus in (2, 3, 4, 6):
            coordinates = np.array(list(product(range(modulus), repeat=3)))
            old = (coordinates@t) % modulus
            old_index = old@np.array([modulus**2, modulus, 1])
            for field in range(3):
                start = cursor+field*modulus**3
                ix[start:start+modulus**3] = start+old_index
            cursor += 3*modulus**3
        for new_zone, normal in enumerate(normals):
            old_normal = t.T@normal
            old_normal *= np.sign(old_normal[np.flatnonzero(old_normal)[0]])
            old_zone = normal_lookup[tuple(old_normal)]
            new_start, old_start = 1020+55*new_zone, 1020+55*old_zone
            ix[new_start] = old_start
            offset = 1
            for modulus in (2, 3, 4):
                for new_axis in range(3):
                    old_axis = int(np.flatnonzero(t[new_axis])[0]); sign = int(t[new_axis, old_axis])
                    for field in range(2):
                        dst = new_start+offset+(new_axis*2+field)*modulus
                        src = old_start+offset+(old_axis*2+field)*modulus
                        ix[dst:dst+modulus] = src+(sign*np.arange(modulus)) % modulus
                offset += 6*modulus
        for i, action in enumerate(actions):
            j = action_lookup[tuple((t.T@action@t).ravel())]
            ix[1735+6*i:1735+6*(i+1)] = np.arange(1735+6*j, 1735+6*(j+1))
        indices.append(ix)
    return np.asarray(indices)

def transform_descriptor(descriptor, transform_index):
    """Torch implementation; each row gets its own proper signed permutation."""
    import torch
    x = descriptor
    transforms = torch.as_tensor(orbit_catalogue()[0], device=x.device, dtype=x.dtype)[transform_index]
    gather = torch.as_tensor(descriptor_permutations(), device=x.device)[transform_index]
    result = x.gather(1, gather)
    length = x[:, :3].exp()
    direct = torch.diag_embed(length.square())
    for i, j, column in ((0, 1, 5), (0, 2, 4), (1, 2, 3)):
        direct[:, i, j] = direct[:, j, i] = length[:, i]*length[:, j]*x[:, column]
    direct = transforms@direct@transforms.transpose(-1, -2)
    new_length = direct.diagonal(dim1=-2, dim2=-1).sqrt()
    result[:, :3] = new_length.log()
    for column, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
        result[:, column] = direct[:, i, j]/(new_length[:, i]*new_length[:, j])
    return result
