"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from torch import nn

def candidate_features(q, token, encoded, proposals):
    context = encoded['context'][0].detach().float().cpu().numpy()
    latent = encoded['tokens'][0].detach().float().cpu().numpy()
    predicted_volume = float(encoded['volume'][0])
    scale = token['scale']; rows = []
    for p in proposals:
        basis = p['basis']; frac = q @ np.linalg.inv(basis).T; h = np.rint(frac)
        error = abs(frac-h).max(1); inlier = error < .1
        distance = np.linalg.norm(q-h@basis.T, axis=1)/scale
        singular = np.linalg.svd(basis/scale, compute_uv=False)
        logvolume = np.log(max(abs(np.linalg.det(basis))/scale**3, 1e-15))
        values = [p['score'], p['support'], p['neural_score'], p['mixture_fraction'] or 0., p['sigma'] or 0.,
                  logvolume, predicted_volume, logvolume-predicted_volume, np.log1p(len(q))/10, inlier.mean()]
        values.extend(np.log(singular.clip(1e-10)))
        values.extend(np.quantile(error, [0, .1, .25, .5, .75, .9, 1]))
        values.extend(np.log1p(np.quantile(distance, [.1, .25, .5, .75, .9, 1])))
        hh = h[inlier].astype(np.int64)
        for modulus in (2, 3, 4):
            bucket = (hh % modulus) @ np.array([modulus**2, modulus, 1])
            counts = np.bincount(bucket, minlength=modulus**3).astype(float)
            prob = counts/max(counts.sum(), 1)
            values.extend([prob.max(), float((counts == 0).mean()), float(-(prob*np.log(prob.clip(1e-12))).sum()/np.log(modulus**3))])
        selected = latent[p['chosen']].mean(0)
        rows.append(np.r_[values, context, selected])
    result = np.asarray(rows, np.float32)
    assert np.isfinite(result).all()
    return result

class CandidateRanker(nn.Module):
    def __init__(self, mean, std, width=192, strength=1.):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-4))
        self.network = nn.Sequential(nn.Linear(len(mean), width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(.1),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))
        self.width = width; self.strength = strength

    def forward(self, x):
        return self.network(((x-self.mean)/self.std).clamp(-12, 12)).squeeze(-1)

    @torch.inference_mode()
    def rank(self, q, token, encoded, proposals):
        features = candidate_features(q, token, encoded, proposals)
        device = self.mean.device
        logits = self(torch.as_tensor(features, device=device)).cpu().numpy()
        return features[:, 0]+self.strength*logits

    @classmethod
    def load(cls, path, device='cpu', strength=1.):
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(state['mean'], state['std'], state['width'], strength).to(device)
        model.load_state_dict(state['model']); model.eval()
        return model
