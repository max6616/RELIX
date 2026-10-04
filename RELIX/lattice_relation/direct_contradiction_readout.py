"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

def retain_uncontradicted(current,proposed,penalties,mapping):
    """Keep a coherent parent probability vector when a changed winner has no evidence against it.

    Fixed family-mass normalization can concentrate a competing family's mass
    even when the old winner is entirely compatible with observed reflections.
    Do not change that SG or diffraction-class decision merely through such
    concentration. No threshold is fitted: zero minimum penalty means at least one
    compatible setting has no effective observed contradiction.
    """
    current=np.asarray(current,float);proposed=np.asarray(proposed,float)
    penalties=np.asarray(penalties,float);mapping=np.asarray(mapping,int)
    old_sg=int(current.argmax());new_sg=int(proposed.argmax())
    old_class=int(np.bincount(mapping,weights=current,minlength=122).argmax())
    new_class=int(np.bincount(mapping,weights=proposed,minlength=122).argmax())
    old_class_penalty=float(penalties[mapping==old_class].min())
    info=dict(previous_space_group=old_sg+1,proposed_space_group=new_sg+1,
        previous_class=old_class,proposed_class=new_class,
        previous_sg_penalty=float(penalties[old_sg]),previous_class_minimum_penalty=old_class_penalty)
    if new_sg!=old_sg and penalties[old_sg]<=0.:
        return current.copy(),dict(info,applied=False,reason='previous_sg_not_contradicted')
    if new_class!=old_class and old_class_penalty<=0.:
        return current.copy(),dict(info,applied=False,reason='previous_class_not_contradicted')
    return proposed.copy(),dict(info,applied=True,reason='changed_decisions_have_direct_contradiction')
