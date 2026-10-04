"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from lattice_aligned.features import intensity_statistics

GRID_LIMIT=8

GRID_SIZE=2*GRID_LIMIT+1

GRID_CHANNELS=3

def describe_reflection_grid(q,intensity,geometry):
    q=np.asarray(q,float);h=np.rint(q@geometry['matrix'].T).astype(np.int64)
    unique,inverse,counts=np.unique(h,axis=0,return_inverse=True,return_counts=True)
    raw=np.expm1(np.asarray(intensity,float)*np.log(1001))/1000
    merged=np.bincount(inverse,weights=raw)/counts;positions=np.zeros((len(unique),3));np.add.at(positions,inverse,q);positions/=counts[:,None]
    e,_=intensity_statistics(np.linalg.norm(positions,axis=1),merged)
    keep=(abs(unique)<=GRID_LIMIT).all(1);grid=np.zeros((GRID_CHANNELS,GRID_SIZE,GRID_SIZE,GRID_SIZE),np.float32)
    if keep.any():
        x,y,z=(unique[keep]+GRID_LIMIT).T
        grid[0,x,y,z]=1
        grid[1,x,y,z]=np.log1p(np.maximum(e[keep],0))
        grid[2,x,y,z]=np.log1p(np.maximum(merged[keep],0)/max(float(merged.mean()),1e-12))
    meta=np.array([np.log1p(len(unique))/10,float(keep.mean()),float((merged>0).mean()),np.log1p(abs(unique).max(initial=0))/5],np.float32)
    assert np.isfinite(grid).all() and np.isfinite(meta).all()
    return grid,meta
