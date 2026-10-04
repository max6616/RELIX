"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import time

import numpy as np

from cctbx.array_family import flex

from .consensus_inference import gaussian_consensus

from .data import relation_tokens

from .positive_reflection_evidence import metric_settings

from .positive_reflection_marginals import preserve_point_group

from .direct_contradiction_readout import retain_uncontradicted

from .full_inference import bravais_from_space_group

def readout(original, mapping):
    start=time.perf_counter()
    result=dict(original)
    try:
        q=np.asarray(original['q'],float);intensity=np.asarray(original['normalized_intensity'],float);basis=np.asarray(original['basis'],float)
        fractional=q@np.linalg.inv(basis).T;h=np.rint(fractional).astype(np.int64)
        keep=(np.max(abs(fractional-h),axis=1)<.1)&np.any(h!=0,axis=1)
        if keep.sum()<12:raise ValueError('insufficient_integer_consistent_positive_evidence')
        scale=relation_tokens(q,intensity)['scale'];_,fraction,relative_sigma=gaussian_consensus(q,basis,scale)
        variance=(relative_sigma*scale)**2;residual2=np.square(q-h@basis.T).sum(1)
        log_signal=np.log(fraction)+np.log(max(abs(np.linalg.det(basis)),1e-30))-1.5*np.log(2*np.pi*variance)-residual2/(2*variance)
        responsibility=np.exp(log_signal-np.logaddexp(np.log1p(-fraction),log_signal))
        raw=np.maximum(np.expm1(intensity*np.log(1001)),0.)
        current=raw/max(float(raw.max()),1e-30);p80=np.clip(raw/max(float(np.quantile(raw,.8)),1e-30),0.,1.)
        h=h[keep];first=np.argmax(h!=0,axis=1);h=h*np.sign(h[np.arange(len(h)),first])[:,None]
        unique,inverse=np.unique(h,axis=0,return_inverse=True)
        weights=[]
        for row_weight in (responsibility*current,responsibility*p80):
            merged=np.zeros(len(unique));np.maximum.at(merged,inverse,row_weight[keep]);weights.append(-np.log1p(-np.clip(merged,0.,1.-1e-12)))
        # Build Python tuples and the C++ index array once for every setting.
        indices=flex.miller_index([tuple(map(int,h)) for h in unique])
        settings=metric_settings(basis,max_delta=1.)
        penalties=np.zeros((2,230))
        for number,variants in settings.items():
            masks=np.stack([np.asarray(row['group'].is_sys_absent(indices),bool) for row in variants])
            for j,weight in enumerate(weights):penalties[j,number-1]=float((masks@weight).min())
        prior=np.asarray(original['space_group_probability'],float)
        current=preserve_point_group(prior,penalties[0]);proposed=preserve_point_group(prior,penalties[1])
        probability,decision=retain_uncontradicted(current,proposed,penalties[1],mapping)
        group=int(probability.argmax())+1;marginal=np.bincount(mapping,weights=probability,minlength=122);chosen=int(marginal.argmax())
        result.update(space_group_probability=probability,space_group=group,diffraction_class_probability=marginal,diffraction_class=chosen,
            space_group_implied_class=int(mapping[group-1]),diffraction_class_members=np.flatnonzero(mapping==chosen)+1,
            neural_lattice_family=bravais_from_space_group(group),positive_reflection_neural_prior=prior.copy())
        result['budget_symmetry']=dict(applied=True,settings_groups=len(settings),unique_indices=len(unique),decision=decision)
    except Exception as exc:
        result=dict(original);result['budget_symmetry']=dict(applied=False,error=repr(exc))
    result['budget_symmetry']['seconds']=time.perf_counter()-start
    return result
