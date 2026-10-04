"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from torch import nn

from lattice_aligned.features import catalogue

class OperationTransformer(nn.Module):
    def __init__(self, mean, std, width=96, layers=2, metric_encoding="standard"):
        super().__init__()
        self.config = dict(width=width, layers=layers, metric_encoding=metric_encoding)
        self.metric_encoding = metric_encoding
        self.register_buffer('mean', torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(np.asarray(std), dtype=torch.float32))
        _, actions, normals = catalogue()
        self.register_buffer('actions', torch.as_tensor(actions, dtype=torch.float32))
        self.register_buffer('normals', torch.as_tensor(normals, dtype=torch.float32))
        self.global_embed = nn.Linear(33, width)
        self.residue = nn.ModuleList([nn.Linear(3*m**3, width) for m in (2, 3, 4, 6)])
        self.zone = nn.Linear(55, width)
        self.zone_position = nn.Linear(3, width, bias=False)
        self.operation = nn.Linear(6, width)
        self.operation_position = nn.Linear(9, width, bias=False)
        self.type_embed = nn.Parameter(torch.randn(1, 103, width)*.02)
        layer = nn.TransformerEncoderLayer(width, 4, width*3, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width*2, 230)

    def forward(self, descriptor):
        # Discard indices 9:51: categorical template and its hard residual.
        x = ((descriptor-self.mean)/self.std).clamp(-12, 12)
        if self.metric_encoding == "log_relative":
            # Relative metric errors span exact symmetry through order-one
            # incompatibility. Resolve the .001 measurement scale explicitly;
            # standard deviation over all incompatible operations hides it.
            x = x.clone()
            x[:, 1735::6] = torch.log1p(torch.expm1(descriptor[:, 1735::6]).clamp_min(0)/.001)/3.
        global_x = torch.cat([x[:, :9], x[:, 51:75]], -1)
        tokens = [self.global_embed(global_x)[:, None]]
        cursor = 75
        for modulus, layer in zip((2, 3, 4, 6), self.residue):
            columns = 3*modulus**3
            tokens.append(layer(x[:, cursor:cursor+columns])[:, None])
            cursor += columns
        assert cursor == 1020
        tokens.append(self.zone(x[:, 1020:1735].reshape(-1, 13, 55)) + self.zone_position(self.normals)[None])
        tokens.append(self.operation(x[:, 1735:].reshape(-1, 85, 6)) + self.operation_position(self.actions.flatten(1))[None])
        latent = torch.cat(tokens, 1) + self.type_embed
        latent = self.norm(self.encoder(latent))
        return self.head(torch.cat([latent[:, 0], latent[:, 1:].mean(1)], -1))
