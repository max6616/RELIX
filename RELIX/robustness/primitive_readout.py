"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from sympy import Matrix

from sympy.matrices.normalforms import hermite_normal_form

def observed_primitive(basis,q,accepted=None):
    basis=np.asarray(basis,float);q=np.asarray(q,float);fraction=q@np.linalg.inv(basis).T;h=np.rint(fraction).astype(np.int64)
    consistent=abs(fraction-h).max(1)<.1
    if accepted is not None:consistent &= np.asarray(accepted,bool)
    if consistent.sum()<3:return basis,dict(applied=False,reason='insufficient_consistent_points',integer_hnf_calls=0)
    change=np.asarray(hermite_normal_form(Matrix(np.unique(h[consistent],axis=0).T)),dtype=np.int64)
    if change.shape!=(3,3):return basis,dict(applied=False,reason='rank_deficient_indices',integer_hnf_calls=1)
    index=int(round(np.linalg.det(change)))
    if index<1:raise ValueError('nonpositive_observed_integer_index')
    return basis@change,dict(applied=index>1,observed_index=index,change=change.tolist(),consistent_points=int(consistent.sum()),integer_hnf_calls=1)
