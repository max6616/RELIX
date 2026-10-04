"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from types import MethodType

import torch

from .profile_features import transform_profile_descriptor

from .reflection_set import transform_reflection_set

def profile_forward(self, descriptor):
    if not self.orbit:
        return self.single(descriptor)
    batch=len(descriptor)
    index=torch.arange(24,device=descriptor.device).repeat_interleave(batch)
    transformed=transform_profile_descriptor(descriptor.repeat(24,1),index)
    probability=self.single(transformed).exp().reshape(24,batch,230)
    result=torch.zeros((batch,230),device=descriptor.device)
    for frame in range(24):result+=probability[frame]
    return (result/24).clamp_min(1e-30).log()

def reflection_forward(self,tokens,parent):
    batch=len(tokens)
    index=torch.arange(24,device=tokens.device).repeat_interleave(batch)
    transformed=transform_reflection_set(tokens.repeat(24,1,1),index)
    probability=self.single(transformed,parent.repeat(24,1)).exp().reshape(24,batch,230)
    result=torch.zeros_like(parent,dtype=torch.float32)
    for frame in range(24):result+=probability[frame]
    return (result/24).clamp_min(1e-30).log()

def attach(postprocessor):
    from .profile_symmetry import ProfileSymmetryTransformer
    from .reflection_set import ReflectionSetSpaceGroup
    assert type(postprocessor.model.symmetry) is ProfileSymmetryTransformer
    assert type(postprocessor.expert) is ReflectionSetSpaceGroup
    postprocessor.model.symmetry.forward=MethodType(profile_forward,postprocessor.model.symmetry)
    postprocessor.expert.forward=MethodType(reflection_forward,postprocessor.expert)
