"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import itertools

import numpy as np

from .inference import refine_q

from .consensus_inference import gaussian_consensus

from .metric_projection import decode_measured_basis

def index_transforms(indices=(2, 3)):
    yield np.eye(3), 'identity'
    for index in indices:
        for a in range(1, index+1):
            for b in range(1, index+1):
                if index % (a*b): continue
                c = index//(a*b)
                for x,y,z in itertools.product(range(a),range(a),range(b)):
                    T = np.array([[a,x,y],[0,b,z],[0,0,c]],float)
                    # Dualizing column-HNF sublattices enumerates every superlattice.
                    yield np.linalg.inv(T.T), f'expand-{index}'
                    yield T, f'contract-{index}'

def polish_candidates(q, initial, *, indices=(2,3), steps=24):
    q=np.asarray(q,float);initial=np.asarray(initial,float)
    order=np.argsort(np.linalg.norm(q,axis=1));scale=abs(np.linalg.det(initial))**(1/3)
    candidates=[];failures=[];solves=0
    def record(B,tag):
        B=np.linalg.inv(decode_measured_basis(B)['matrix'])
        score,fraction,sigma=gaussian_consensus(q,B,scale)
        candidates.append(dict(basis=B,score=score,fraction=fraction,sigma=sigma,tag=tag))
    record(initial,'original')
    for T,label in index_transforms(indices):
        for mode in (('all','low-first','small-first') if label=='identity' else ('all','low-first')):
            try:
                B=np.linalg.inv(decode_measured_basis(initial@T)['matrix'])
                schedule={'all':(1.,),'low-first':(.15,.3,.5,.75,1.),'small-first':(.02,.05,.15,.4,1.)}[mode]
                for proportion in schedule:
                    rows=order[:max(48,int(len(q)*proportion))]
                    B,_,_,history=refine_q(q[rows],B,steps,1e-7)
                    solves+=len(history)
                record(B,label+'-'+mode)
            except (ValueError,np.linalg.LinAlgError,RuntimeError,AssertionError) as exc:
                failures.append(type(exc).__name__)
    best=max(candidates,key=lambda c:c['score'])
    best['polish_linear_solves']=solves
    return best,candidates,failures
