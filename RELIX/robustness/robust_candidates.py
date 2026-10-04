"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from itertools import combinations

from time import perf_counter

import numpy as np

from scipy.spatial import cKDTree

def translation_modes(q, count=24, anchors=2048):
    q=np.asarray(q,dtype=float)
    if len(q)<12:
        raise ValueError('Need twelve reciprocal observations')
    # Only remove essentially exact duplicates, then avoid using the nearest
    # distance as the scale: repeated reflections can have tiny noisy separations.
    _,ix=np.unique(np.rint(q/1e-6).astype(np.int64),axis=0,return_index=True)
    points=q[ix]
    radial=np.lexsort((points[:,2],points[:,1],points[:,0],np.linalg.norm(points,axis=1)))
    probe=points[radial[np.linspace(0,len(points)-1,min(anchors,len(points))).astype(int)]]
    tree=cKDTree(points);dist,neighbors=tree.query(probe,k=min(25,len(points)))
    scale=float(np.median(dist[:,min(5,dist.shape[1]-1)]))
    if scale<=1e-8:
        raise ValueError('degenerate_observation_scale')
    delta=(points[neighbors[:,1:]]-probe[:,None]).reshape(-1,3)
    lengths=np.linalg.norm(delta,axis=1)
    delta=delta[lengths>scale*.03]
    delta*=np.sign(delta[np.arange(len(delta)),abs(delta).argmax(1)])[:,None]
    width=.02*scale
    keys,inv,counts=np.unique(np.rint(delta/width).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    sums=np.zeros((len(keys),3));np.add.at(sums,inv,delta)
    centers=sums/counts[:,None]
    center_tree=cKDTree(centers)
    seeds=np.argsort(-counts,kind='stable')[:min(512,len(counts))]
    vectors=[];support=[]
    for index in seeds:
        if vectors and np.min(np.linalg.norm(np.asarray(vectors)-centers[index],axis=1))<width*3:
            continue
        ids=center_tree.query_ball_point(centers[index],r=width*2)
        weight=counts[ids].sum();v=sums[ids].sum(0)/weight
        if vectors and np.min(np.linalg.norm(np.asarray(vectors)-v,axis=1))<width*3:
            continue
        vectors.append(v);support.append(weight)
        if len(vectors)>=96:
            break
    vectors=np.asarray(vectors);support=np.asarray(support)
    lengths=np.linalg.norm(vectors,axis=1)
    # Frequent translations are shared by many reflections; isolated false
    # spots produce short differences with little repeated support.
    eligible=np.flatnonzero(support>=max(3.,float(support.max())*.06))
    order=eligible[np.argsort(lengths[eligible],kind='stable')]
    selected=list(order[:count])
    if len(selected)<3 or np.linalg.matrix_rank(vectors[selected],tol=scale*.005)<3:
        for i in np.argsort(-support):
            if i not in selected:selected.append(int(i))
            if len(selected)>=3 and np.linalg.matrix_rank(vectors[selected],tol=scale*.005)==3:break
    if len(selected)<3 or np.linalg.matrix_rank(vectors[selected],tol=scale*.005)<3:
        raise ValueError('translation_modes_do_not_span')
    # Bound complexity while preserving a third independent direction.
    if len(selected)>count:
        anchors=[]
        for i in selected:
            if np.linalg.matrix_rank(vectors[anchors+[i]],tol=scale*.005)>len(anchors):anchors.append(i)
            if len(anchors)==3:break
        selected=anchors+[i for i in selected if i not in anchors][:count-len(anchors)]
    return vectors[selected],support[selected]/len(probe),scale,dict(anchors=len(probe),pairs=len(delta),histogram_bins=len(keys))

def build_candidates(q, count=192, point_probe=256):
    start=perf_counter();q=np.asarray(q,dtype=float)
    vectors,support,scale,stats=translation_modes(q)
    stats['translation_seconds']=perf_counter()-start
    triples=np.array(list(combinations(range(len(vectors)),3)),dtype=np.int64)
    basis=vectors[triples].transpose(0,2,1)
    volume=np.linalg.det(basis);length=np.linalg.norm(basis,axis=1)
    quality=abs(volume)/np.prod(length,axis=1)
    keep=quality>.02;basis=basis[keep];volume=volume[keep];quality=quality[keep];triples=triples[keep]
    if not len(basis):raise ValueError('no_spanning_vote_candidates')
    basis[volume<0,:,0]*=-1;volume=abs(volume)
    radial=np.lexsort((q[:,2],q[:,1],q[:,0],np.linalg.norm(q,axis=1)))
    probe=q[radial[np.linspace(0,len(q)-1,min(point_probe,len(q))).astype(int)]]
    inverse=np.linalg.inv(basis)
    continuous=np.einsum('pj,cij->cpi',probe,inverse)
    error=abs(continuous-np.rint(continuous));maximum=error.max(2)
    fractions=np.stack([(maximum<t).mean(1) for t in (.01,.03,.06,.1,.15,.25)],1)
    # Broad coverage in the retained bank, using only observable consensus.
    # Prefer the coarsest basis among nearly equally supported lattices.
    consensus=fractions[:,3]
    top=np.argsort(-consensus,kind='stable')
    first=list(top[:min(count//2,len(top))])
    remaining=np.array([i for i in np.argsort(volume) if i not in set(first)])
    if len(remaining):
        first+=remaining[np.linspace(0,len(remaining)-1,min(count-len(first),len(remaining))).astype(int)].tolist()
    first=np.array(first);mask=np.arange(count)<len(first);first=first[np.arange(count)%len(first)]
    basis=basis[first];volume=volume[first];quality=quality[first];triples=triples[first]
    fractions=fractions[first];maximum=maximum[first];continuous=continuous[first]
    length=np.linalg.norm(basis,axis=1);unit=basis/length[:,None]
    cosine=np.stack([(unit[:,:,i]*unit[:,:,j]).sum(1) for i,j in ((0,1),(0,2),(1,2))],1)
    features=np.c_[np.log(length/scale),cosine,np.log(volume/scale**3),np.log(volume/volume.min()),
                   np.log(np.linalg.cond(basis)),quality,
                   np.quantile(maximum,[.1,.25,.5,.75,.9],axis=1).T,fractions,
                   np.log1p(abs(continuous).mean((1,2))),np.log1p(abs(continuous).max((1,2))),
                   support[triples],np.full(count,np.log(scale)),np.full(count,np.log1p(len(q)))]
    best=fractions[mask,3].max()
    available=np.flatnonzero(mask&(fractions[:,3]>=best-.015))
    chosen=int(available[np.argmax(volume[available])])
    stats.update(total_triples=len(keep),valid_triples=int(keep.sum()),generation_seconds=perf_counter()-start)
    return dict(features=features.astype(np.float32),basis=basis,mask=mask,volume=volume,
                consensus_index=chosen,stats=stats)
