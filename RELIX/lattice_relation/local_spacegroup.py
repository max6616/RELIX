"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import torch

def compose(parent,learned,sg_to_laue,strength):
    membership=torch.nn.functional.one_hot(sg_to_laue).to(parent.dtype)
    mass=parent@membership;new_mass=learned@membership
    refined=learned/new_mass[:,sg_to_laue].clamp_min(1e-30)*mass[:,sg_to_laue]
    result=(1-strength)*parent+strength*refined
    return result.clamp_min(1e-30).log().log_softmax(-1)
