"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from .reference_geometry import observed_lattice_vectors,cell_from_indexing_matrix,canonical_primitive

from .features import catalogue

from multitask_indexer.geometry import reciprocal_basis,orthogonalization,proper_rotation

class _NiggliTolerance:
    """Use the unit-cell API: crystal.symmetry's wrapper ignores this argument."""
    def __init__(self,symmetry,epsilon):self.symmetry,self.epsilon=symmetry,epsilon
    def __getattr__(self,name):return getattr(self.symmetry,name)
    def change_of_basis_op_to_niggli_cell(self):
        return self.symmetry.unit_cell().change_of_basis_op_to_niggli_cell(relative_epsilon=self.epsilon)

def decode_basis(basis,niggli_epsilon=None):
    from cctbx import crystal
    basis=np.asarray(basis,dtype=np.float64);M=np.linalg.inv(basis)
    cell=cell_from_indexing_matrix(M)
    sym=crystal.symmetry(unit_cell=tuple(cell),space_group_symbol='P 1')
    if niggli_epsilon is not None:sym=_NiggliTolerance(sym,niggli_epsilon)
    p=canonical_primitive(sym,basis)
    M=p['indexing_matrix'];cell=p['cell'];B=reciprocal_basis(cell);G=B.T@B
    templates,_,_=catalogue();errors=[]
    for t in templates:
        A=np.asarray(t['actions']);errors.append(float(np.max(abs(A.transpose(0,2,1)@G@A-G))/np.max(abs(G))))
    available=[i for i,e in enumerate(errors) if e<2e-5]
    ti=min(available,key=lambda i:(-len(templates[i]['actions']),errors[i],templates[i]['id']))
    t=templates[ti];A=np.asarray(t['actions']);U=proper_rotation(np.linalg.inv(M)@orthogonalization(cell).T)
    rotations=U@(B@np.linalg.inv(A)@np.linalg.inv(B))
    k=min(range(len(A)),key=lambda i:(-round(float(np.trace(rotations[i])),12),tuple(np.round(rotations[i].ravel(),12))))
    return {'matrix':A[k]@M,'cell':cell,'orientation':proper_rotation(rotations[k]),'template':ti,
            'template_id':t['id'],'actions':A,'template_residual':errors[ti],'details':{'method':'learned_basis_selection_no_integer_lattice_solver'}}
