"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from .data import relation_tokens

from .model import RelationTransformer

def refine_q(q, basis, steps=2, tolerance=0.):
    C = np.asarray(basis, float).copy()
    if np.linalg.det(C) < 0:
        C[:, 0] *= -1
    history = []
    for _ in range(steps):
        continuous = q @ np.linalg.inv(C).T
        h = np.rint(continuous)
        residual = abs(continuous-h).max(1)
        cutoff = float(np.clip(6*np.quantile(residual, .25), .02, .2))
        w = np.maximum(1-(residual/cutoff)**2, 0.)**2
        if (w > .05).sum() < 12:
            raise ValueError('insufficient_consensus_for_fixed_q_refinement')
        updated = np.linalg.lstsq(h*np.sqrt(w[:, None]), q*np.sqrt(w[:, None]), rcond=1e-8)[0].T
        change = float(np.linalg.norm(updated-C)/max(np.linalg.norm(C), 1e-12))
        C = updated
        history.append(dict(retained=int((w > .05).sum()), cutoff=cutoff, relative_change=change))
        if tolerance > 0 and change < tolerance:
            break
    h = np.rint(q @ np.linalg.inv(C).T).astype(np.int64)
    phase = abs(q @ np.linalg.inv(C).T-h).max(1)
    return C, h, phase, history

class RelationIndexer:
    def __init__(self, model, mean, std, count=512, device='cpu'):
        self.model, self.mean, self.std, self.count, self.device = model, mean, std, count, device
        self.refinement_steps, self.refinement_tolerance = 2, 0.

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get('model_kind') == 'equivariant_consensus':
            from .consensus_inference import ConsensusIndexer
            return ConsensusIndexer.load(path, device)
        model = RelationTransformer(**state['config']).to(device).eval()
        model.load_state_dict(state['model'])
        return cls(model, state['mean'], state['std'], state['tokens'], device)

    @torch.inference_mode()
    def predict(self, q, intensity):
        q = np.asarray(q, float)
        token = relation_tokens(q, intensity, self.count)
        features = np.clip((token['features']-self.mean)/self.std, -12, 12).astype(np.float32)
        features[~token['mask']] = 0
        encoded, chosen = self.model(torch.as_tensor(features[None], device=self.device),
                                     torch.as_tensor(token['vectors'][None]/token['scale'], device=self.device),
                                     torch.as_tensor(token['mask'][None], device=self.device))
        chosen = chosen[0].cpu().numpy()
        C = token['vectors'][chosen].T.astype(float)
        if abs(np.linalg.det(C))/max(np.prod(np.linalg.norm(C, axis=0)), 1e-20) < .005:
            raise ValueError('neural_basis_degenerate')
        C, h, residual, history = refine_q(q, C, steps=self.refinement_steps, tolerance=self.refinement_tolerance)
        return dict(basis=C, hkl=h, residual=residual, accepted=residual < .1, chosen=chosen,
                    space_group=int(encoded['sg'][0].argmax())+1 if 'sg' in encoded else 0,
                    template=int(encoded['template'][0].argmax()) if 'template' in encoded else -1,
                    correction_history=history, network_forwards=1, linear_solves=len(history))
