"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from cctbx import sgtbx

@lru_cache(None)
def joint_mapping():
    keys=[]
    for number in range(1,231):
        group=sgtbx.space_group_info(number=number).group()
        keys.append((group.build_derived_patterson_group().type().number(),group.point_group_type()))
    lookup={key:i for i,key in enumerate(sorted(set(keys)))}
    return np.asarray([lookup[key] for key in keys])

def preserve_point_group(probability,penalty):
    probability=np.asarray(probability,float);penalty=np.asarray(penalty,float)
    assert probability.shape==penalty.shape==(230,)
    mapping=joint_mapping();result=probability.copy()
    for family in np.unique(mapping):
        mask=mapping==family;prior=probability[mask];mass=prior.sum()
        if mass<=0:continue
        positive=prior>0;score=np.full(len(prior),-np.inf)
        score[positive]=np.log(prior[positive])-penalty[mask][positive]
        value=np.exp(score-score.max());result[mask]=mass*value/value.sum()
    assert np.allclose(np.bincount(mapping,weights=result),np.bincount(mapping,weights=probability),atol=1e-14,rtol=0.)
    return result
