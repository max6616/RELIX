"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

from itertools import product

from xrdt_workspace import WorkspacePath as Path

import json

import numpy as np

from .reference_geometry import index_integer_lattice

from multitask_indexer.geometry import reciprocal_basis, orthogonalization, proper_rotation, family

@lru_cache(None)
def catalogue():
    data = json.loads((Path(__file__).parent/'assets/registry.json').read_text())
    templates = sorted(data['templates'], key=lambda t:t['id'])
    actions = np.unique(np.concatenate([np.asarray(t['actions']) for t in templates]), axis=0)
    normals = np.array([v for v in product((-1,0,1), repeat=3)
                        if any(v) and next(x for x in v if x) > 0], dtype=np.int64)
    return templates, actions, normals

def recover(q):
    prediction, details = index_integer_lattice(q)
    M = prediction['indexing_matrix']
    cell = prediction['cell']
    B = reciprocal_basis(cell)
    G = B.T@B
    templates, _, _ = catalogue()
    errors=[]
    for t in templates:
        actions=np.asarray(t['actions'])
        residual=np.max(np.abs(actions.transpose(0,2,1)@G@actions-G))/np.max(abs(G))
        errors.append(float(residual))
    available=[i for i,e in enumerate(errors) if e < 2e-5]
    ti=min(available, key=lambda i:(-len(templates[i]['actions']),errors[i],templates[i]['id']))
    template=templates[ti]
    U=proper_rotation(np.linalg.inv(M)@orthogonalization(cell).T)
    A=np.asarray(template['actions'])
    rotations=B@np.linalg.inv(A)@np.linalg.inv(B)
    candidates=U@rotations
    k=min(range(len(A)),key=lambda i:(-round(float(np.trace(candidates[i])),12),tuple(np.round(candidates[i].ravel(),12))))
    M=A[k]@M
    return dict(matrix=M,cell=cell,orientation=proper_rotation(candidates[k]),template=ti,
                template_id=template['id'],actions=A,template_residual=errors[ti],details=details)

def intensity_statistics(radius, intensity, *, min_per_shell=None):
    # Keep the historical ten-shell representation unless explicitly requested.
    # Quantile bins with one observation divide that value by itself.
    bins=10 if min_per_shell is None else min(10,max(1,len(radius)//min_per_shell))
    edges=np.quantile(radius,np.linspace(0,1,bins+1)[1:-1])
    shell=np.searchsorted(edges,radius,side='right')
    counts=np.bincount(shell,minlength=bins)
    means=np.bincount(shell,weights=intensity,minlength=bins)/np.maximum(counts,1)
    e=intensity/np.maximum(means[shell],1e-12)
    v=(e-e.mean())/max(e.std(),1e-8)
    ordered=np.sort(e);n=len(e)
    stats=np.r_[np.mean(abs(e-1)),e.std(),[(e<=t).mean() for t in (.25,.5,1,2,3,5)],
                2*np.sum(np.arange(1,n+1)*ordered)/max(n*ordered.sum(),1e-12)-(n+1)/n,
                np.mean(v**3),np.mean(v**4)-3,np.quantile(e,[.1,.25,.5,.75,.9,.99])]
    return e,stats

def describe(q, cached_intensity, point_count=512, *, geometry=None):
    q=np.asarray(q,dtype=np.float64)
    I=np.expm1(np.asarray(cached_intensity,dtype=np.float64)*np.log(1001))/1000
    geom=recover(q) if geometry is None else geometry
    h=np.rint(q@geom['matrix'].T).astype(np.int64)
    templates,actions,normals=catalogue()
    unique,inv,counts=np.unique(h,axis=0,return_inverse=True,return_counts=True)
    intensity=np.bincount(inv,weights=I)/counts
    uq=np.zeros((len(unique),3));np.add.at(uq,inv,q);uq/=counts[:,None]
    radius=np.linalg.norm(uq,axis=1)
    e,stats=intensity_statistics(radius,intensity)
    cell=geom['cell'];B=reciprocal_basis(cell);G=B.T@B
    base=[np.log(cell[:3]),np.cos(np.deg2rad(cell[3:])),
          [np.log(np.linalg.det(orthogonalization(cell))),np.log1p(len(q)),np.log1p(len(unique)),geom['template_residual']],
          np.eye(len(templates))[geom['template']],np.quantile(radius,[0,.1,.25,.5,.75,.9,1]),stats]
    # Complete residue occupancy, retaining absences as well as intensity.
    residue=[]
    for modulus in (2,3,4,6):
        bucket=((unique%modulus)*np.array([modulus**2,modulus,1])).sum(1)
        n=modulus**3
        c=np.bincount(bucket,minlength=n)
        residue.extend([c/max(len(unique),1),np.bincount(bucket,weights=e,minlength=n)/np.maximum(c,1),
                        np.bincount(bucket,weights=np.log1p(e),minlength=n)/np.maximum(c,1)])
    # Zone-specific absence patterns survive removal of conventional centering.
    zones=[]
    for normal in normals:
        zone=(unique@normal==0)
        z=unique[zone];ze=e[zone]
        zones.append(np.array([zone.mean()]))
        for modulus in (2,3,4):
            for axis in range(3):
                c=np.bincount(z[:,axis]%modulus,minlength=modulus)
                zones.extend([c/max(len(z),1),np.bincount(z[:,axis]%modulus,weights=ze,minlength=modulus)/np.maximum(c,1)])
    # Reflection-by-reflection intensity agreement for every catalogue operation.
    span=int(max(abs(unique).max(initial=0),1));shift=8*span+16;base_key=2*shift+1
    def keys(x):return ((x[:,0]+shift)*base_key+x[:,1]+shift)*base_key+x[:,2]+shift
    key=keys(unique);sort=np.argsort(key);key=key[sort]
    agreements=[]
    for action in actions:
        transformed=unique@action.T
        tk=keys(transformed);idx=np.searchsorted(key,tk);valid=idx<len(key)
        clipped=np.minimum(idx,len(key)-1);valid&=key[clipped]==tk
        pair=sort[clipped[valid]]
        a=e[valid];b=e[pair]
        metric=float(np.max(abs(action.T@G@action-G))/np.max(abs(G)))
        if len(a)>=3:
            ra=float(np.sum(abs(a-b))/max(np.sum(abs(a)+abs(b)),1e-12))
            correlation=float(np.mean((a-a.mean())*(b-b.mean()))/max(a.std()*b.std(),1e-12))
            ld=abs(np.log1p(a)-np.log1p(b))
            agreements.append([np.log1p(metric),valid.mean(),correlation,ra,float(np.mean(ld)),float(np.quantile(ld,.9))])
        else:agreements.append([np.log1p(metric),valid.mean(),0.,1.,1.,1.])
    desc=np.concatenate([np.asarray(v).ravel() for v in base+residue+zones+[np.asarray(agreements)]]).astype(np.float32)
    # A fixed coverage sample for separate raw-q versus lattice-aligned learned
    # point encoders. All observations contributed to the descriptor above.
    order=np.lexsort((unique[:,2],unique[:,1],unique[:,0],radius))
    ix=order[np.linspace(0,len(order)-1,point_count).astype(int)]
    common=np.c_[np.log1p(e[ix]),np.log1p(intensity[ix]*1000),radius[ix],np.log1p(counts[ix])]
    raw=np.c_[uq[ix],common].astype(np.float32)
    aligned=np.c_[unique[ix]/10.,common].astype(np.float32)
    if not np.isfinite(desc).all():raise FloatingPointError('nonfinite_observed_descriptor')
    return dict(descriptor=desc,raw_points=raw,aligned_points=aligned,geometry=geom,hkl=h,
                families=family(h,geom['actions']))
