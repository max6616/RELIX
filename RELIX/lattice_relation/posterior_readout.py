"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from .consensus_inference import gaussian_consensus

@torch.inference_mode()
def finish_projection_comparison(model, q, intensity, basis, scale):
    original_flag = model.project_geometry
    candidates, errors = [], []
    try:
        for project in dict.fromkeys((original_flag, False)):
            model.project_geometry = project
            try:
                result = model._finish(q, intensity,
                         dict(basis=np.asarray(basis).copy(), q=q, network_forwards=0, linear_solves=0,
                              calibration_solves=0, solve_count_complete=False, geometry_source='posterior_primitive'))
                quality = gaussian_consensus(result['q'], result['basis'], scale)
                if not np.isfinite(quality[0]):
                    raise ValueError('nonfinite_final_likelihood')
                candidates.append(dict(project_geometry=bool(project), score=quality[0], result=result))
            except Exception as exc:
                errors.append(dict(project_geometry=bool(project), error=repr(exc)))
    finally:
        model.project_geometry = original_flag
    if not candidates:
        raise ValueError('all_posterior_readout_variants_failed:' + repr(errors))
    selected = max(range(len(candidates)), key=lambda j: candidates[j]['score'])
    result = candidates[selected]['result']
    result['projection_readout_comparison'] = dict(selected_project_geometry=candidates[selected]['project_geometry'],
            candidates=[dict(project_geometry=c['project_geometry'], score=c['score'], space_group=int(c['result']['space_group'])) for c in candidates], errors=errors)
    return result
