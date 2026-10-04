"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from lattice_aligned.features import catalogue

LOCAL_TOKENS=50

LOCAL_WIDTH=40

def describe_local_intensity(q,intensity,geometry):
    q=np.asarray(q,float);h=np.rint(q@geometry['matrix'].T).astype(int)
    unique,inverse,counts=np.unique(h,axis=0,return_inverse=True,return_counts=True)
    raw=np.expm1(np.asarray(intensity)*np.log(1001))/1000;I=np.bincount(inverse,weights=raw)/counts
    uq=np.zeros((len(unique),3));np.add.at(uq,inverse,q);uq/=counts[:,None]
    radius=np.linalg.norm(uq,axis=1);mean=max(float(I.mean()),1e-12);I=I/mean
    span=int(abs(unique).max())+8;radix=2*span+1
    def keys(v):return ((v[:,0]+span)*radix+v[:,1]+span)*radix+v[:,2]+span
    key=keys(unique);order=np.argsort(key);ordered=key[order]
    profiles=[];all_a=[];all_b=[];all_radius=[];all_distance=[]
    def statistics(a,b,distance):
        if not len(a):return np.zeros(LOCAL_WIDTH,np.float32)
        ratio=abs(a-b)/np.maximum(a+b,1e-12);logs=abs(np.log1p(a)-np.log1p(b));la=np.log1p(a);lb=np.log1p(b)
        center=(la.mean()+lb.mean())/2
        correlation=float(2*np.mean((la-center)*(lb-center))/max(np.mean((la-center)**2)+np.mean((lb-center)**2),1e-12)) if len(a)>=3 else 0.
        return np.r_[np.quantile(ratio,np.linspace(.01,.99,31)),np.log1p(len(a))/10,ratio.mean(),np.mean(ratio**2),logs.mean(),logs.std(),correlation,np.log1p(np.mean(distance)),np.log1p(np.mean(np.sqrt(a*b))),np.mean((a==0)|(b==0))].astype(np.float32)
    for step in (1,2,3):
        for direction in catalogue()[2]:
            shifted=keys(unique+step*direction);found=np.searchsorted(ordered,shifted);clipped=np.minimum(found,len(ordered)-1);valid=(found<len(ordered))&(ordered[clipped]==shifted);other=order[clipped]
            # Equal reciprocal lengths include exact crystallographic mates;
            # these cannot serve as unrelated local intensity pairs.
            valid&=abs(radius-radius[other])>1e-6*np.maximum(radius,radius[other]).clip(1e-12)
            valid&=(I+I[other])>0;ix=np.flatnonzero(valid);j=other[ix]
            distance=np.linalg.norm(uq[ix]-uq[j],axis=1)/max(float(np.median(radius)),1e-12)
            profiles.append(statistics(I[ix],I[j],distance));all_a.append(I[ix]);all_b.append(I[j]);all_radius.append((radius[ix]+radius[j])/2);all_distance.append(distance)
    a=np.concatenate(all_a);b=np.concatenate(all_b);r=np.concatenate(all_radius);distance=np.concatenate(all_distance)
    if len(r):
        shell=np.searchsorted(np.quantile(r,np.linspace(0,1,11)[1:-1]),r,side='right')
        pooled=[statistics(a,b,distance)]+[statistics(a[shell==i],b[shell==i],distance[shell==i]) for i in range(10)]
    else:pooled=[np.zeros(LOCAL_WIDTH,np.float32) for _ in range(11)]
    result=np.asarray(pooled+profiles,np.float32);assert result.shape==(LOCAL_TOKENS,LOCAL_WIDTH) and np.isfinite(result).all();return result
