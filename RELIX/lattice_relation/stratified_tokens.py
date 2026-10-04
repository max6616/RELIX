"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from .data import relation_tokens

def stratified_relation_tokens(q,intensity,count=512,radial_fraction=.25):
    q=np.asarray(q,float);intensity=np.asarray(intensity,float)
    if not 0<radial_fraction<1:raise ValueError('invalid_radial_fraction')
    # The original tokenizer has at most 128*16 local pairs and 128 radial
    # vectors before downsampling. Asking for the whole list preserves it.
    original=relation_tokens(q,intensity,2176);scale=original['scale']
    pairs=np.flatnonzero(original['mask']&(original['source']>=0))
    order=np.lexsort((intensity,q[:,2],q[:,1],q[:,0],np.linalg.norm(q,axis=1)))
    order=order[np.linalg.norm(q[order],axis=1)>max(scale*.02,1e-6)]
    nradial=min(int(round(count*radial_fraction)),len(order));npairs=min(count-nradial,len(pairs))
    chosen=pairs[np.linspace(0,len(pairs)-1,npairs).astype(int)] if npairs else np.array([],int)
    radial=order[np.linspace(0,len(order)-1,nradial).astype(int)]
    vectors=q[radial].copy();sign=np.sign(vectors[np.arange(len(vectors)),abs(vectors).argmax(1)]);vectors*=sign[:,None]
    radius=max(float(np.quantile(np.linalg.norm(q,axis=1),.9)),1e-5)
    features=np.c_[vectors/scale,np.zeros_like(vectors),q[radial]/radius,np.zeros(nradial),intensity[radial],np.log1p(np.linalg.norm(vectors,axis=1)/scale),np.full(nradial,np.log(scale)),np.full(nradial,np.log(radius)),np.full(nradial,np.log1p(len(q))),np.ones(nradial)]
    f=np.r_[original['features'][chosen],features].astype(np.float32);v=np.r_[original['vectors'][chosen],vectors].astype(np.float32);source=np.r_[original['source'][chosen],np.full(nradial,-1)];target=np.r_[original['target'][chosen],radial];sign=np.r_[original['sign'][chosen],sign];n=len(f)
    return dict(features=np.pad(f,((0,count-n),(0,0))),vectors=np.pad(v,((0,count-n),(0,0))),mask=np.arange(count)<n,source=np.pad(source,(0,count-n),constant_values=-1),target=np.pad(target,(0,count-n),constant_values=-1),sign=np.pad(sign,(0,count-n)),scale=scale,sampling_stats=dict(local_pairs=npairs,radial=nradial,radial_fraction=radial_fraction))
