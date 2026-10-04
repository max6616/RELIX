"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from torch import nn

from .orbit_symmetry import OrbitSymmetryTransformer

from .profile_features import PROFILE_TOKENS, PROFILE_WIDTH, transform_profile_descriptor

class ProfileSymmetryTransformer(nn.Module):
    descriptor_kind = 'profile_orbit'

    def __init__(self, base_mean, base_std, base_config, profile_mean, profile_std,
                 width=128, layers=3, orbit=True):
        super().__init__()
        self.config = dict(base_config=base_config, width=width, layers=layers, orbit=orbit)
        self.base = OrbitSymmetryTransformer(base_mean, base_std, **base_config)
        self.register_buffer('mapping', self.base.mapping.detach().clone())
        self.register_buffer('membership', self.base.membership.detach().clone())
        self.register_buffer('profile_mean', torch.as_tensor(np.asarray(profile_mean), dtype=torch.float32))
        self.register_buffer('profile_std', torch.as_tensor(np.asarray(profile_std), dtype=torch.float32).clamp_min(1e-4))
        self.embed = nn.Sequential(nn.Linear(PROFILE_WIDTH, width), nn.LayerNorm(width), nn.GELU())
        self.position = nn.Parameter(torch.randn(1, PROFILE_TOKENS, width)*.02)
        block = nn.TransformerEncoderLayer(width, 4, width*3, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(9, width)
        self.fusion = nn.Sequential(nn.Linear(width*3, width*2), nn.GELU(), nn.Dropout(.1), nn.Linear(width*2, 230))
        nn.init.zeros_(self.fusion[-1].weight); nn.init.zeros_(self.fusion[-1].bias)
        self.orbit = orbit
        self.frame_evaluations = 24 if orbit else 1

    def single(self, descriptor):
        base = descriptor[:, :-PROFILE_TOKENS*PROFILE_WIDTH]
        profile = descriptor[:, -PROFILE_TOKENS*PROFILE_WIDTH:].reshape(-1, PROFILE_TOKENS, PROFILE_WIDTH)
        x = ((profile-self.profile_mean)/self.profile_std).clamp(-12, 12)
        missing = profile[:, :, 31] == 0
        latent = self.norm(self.encoder(self.embed(x)+self.position, src_key_padding_mask=missing))
        weights = (~missing).to(latent.dtype)
        pooled = (latent*weights[..., None]).sum(1)/weights.sum(1, keepdim=True).clamp_min(1)
        shape = base[:, :9].clone()
        log_length = shape[:, :3].mean(1, keepdim=True)
        shape[:, :3] -= log_length
        shape[:, 6:7] -= 3*log_length
        shape[:, 7:9] /= 10.
        correction = self.fusion(torch.cat([latent[:, 0], pooled, self.context(shape)], -1)).float()
        return (self.base.single(base)+correction).log_softmax(-1)

    def forward(self, descriptor):
        if not self.orbit: return self.single(descriptor)
        result = torch.zeros((len(descriptor), 230), device=descriptor.device)
        for i in range(24):
            index = torch.full((len(descriptor),), i, dtype=torch.long, device=descriptor.device)
            result += self.single(transform_profile_descriptor(descriptor, index)).exp()
        return (result/24).clamp_min(1e-30).log()

    def checkpoint(self):
        return dict(model_kind='profile_symmetry', model=self.state_dict(), config=self.config,
            base_mean=self.base.mean.cpu().numpy(), base_std=self.base.std.cpu().numpy(),
            profile_mean=self.profile_mean.cpu().numpy(), profile_std=self.profile_std.cpu().numpy())

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(state['base_mean'], state['base_std'], profile_mean=state['profile_mean'],
                    profile_std=state['profile_std'], **state['config']).to(device)
        model.load_state_dict(state['model']); model.eval()
        return model
