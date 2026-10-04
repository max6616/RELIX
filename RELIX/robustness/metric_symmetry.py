"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from cctbx import uctbx

from cctbx.sgtbx import lattice_symmetry

from lattice_aligned.reference_geometry import cell_from_indexing_matrix

from lattice_aligned.candidates import decode_basis

from multitask_indexer.geometry import proper_rotation

def adaptive_decode(basis,q,significance=54.):
    C=np.asarray(basis,dtype=float);q=np.asarray(q,dtype=float)
    uc=uctbx.unit_cell(tuple(cell_from_indexing_matrix(np.linalg.inv(C))))
    cb=uc.change_of_basis_op_to_niggli_cell(relative_epsilon=1e-9)
    transform=np.array(cb.c_inv().r().as_double()).reshape(3,3).T
    C=C@np.linalg.inv(transform)
    if np.linalg.det(C)<0:C=-C
    cell=uctbx.unit_cell(tuple(cell_from_indexing_matrix(np.linalg.inv(C))))
    G=C.T@C;fractional=q@np.linalg.inv(C).T;h=np.rint(fractional)
    phase=abs(fractional-h).max(1);residual=q-h@C.T
    cutoff=float(np.clip(6*np.quantile(phase,.25),.02,.2))
    w=np.clip(1-(phase/cutoff)**2,0,1)**2
    selected=w>.25
    if selected.sum()<12:raise ValueError('too_few_consistent_points_for_metric_symmetry')
    # Median 3D isotropic-Gaussian radius is about 1.538 times component sigma.
    # The floor accounts for float32 reciprocal coordinates.
    sigma=max(float(np.median(np.linalg.norm(residual[selected],axis=1)))/1.538,
              1e-7*float(np.median(np.linalg.norm(q,axis=1))),1e-10)
    proposals=[];seen=set()
    for delta in [.001,.003,.01,.03,.1,.3,1.]:
        group=lattice_symmetry.group(cell,max_delta=delta)
        key=tuple(sorted(str(op) for op in group))
        if key in seen:continue
        seen.add(key)
        R=np.stack([np.array(op.r().as_double()).reshape(3,3) for op in group])
        actions=np.linalg.inv(R).transpose(0,2,1)
        projected=(actions.transpose(0,2,1)@G@actions).mean(0)
        B=np.linalg.cholesky(projected).T
        U=proper_rotation((q*w[:,None]).T@(h@B.T))
        candidate=U@B
        displacement=h@(candidate-C).T
        statistic=float(np.sum(w[:,None]*displacement**2)/sigma**2)
        proposals.append(dict(order=len(group),delta=delta,statistic=statistic,basis=candidate))
    accepted=[p for p in proposals if p['statistic']<=significance]
    if accepted:
        best=min(accepted,key=lambda p:(-p['order'],p['statistic']))
        result=decode_basis(best['basis'],niggli_epsilon=1e-4)
    else:
        best=dict(order=1,delta=0.,statistic=0.)
        result=decode_basis(C,niggli_epsilon=1e-4)
    result['projection']=dict(method='finite_lepage_covariance_v1',sigma=sigma,effective_points=float(w.sum()),
                              significance=significance,group_order=best['order'],statistic=best['statistic'],
                              proposals=[{k:v for k,v in p.items() if k!='basis'} for p in proposals])
    return result
