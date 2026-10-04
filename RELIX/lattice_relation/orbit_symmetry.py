"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from xrdt_workspace import workspace_root, frozen_source_digest, canonical_relative, source_digest_matches

import json

import numpy as np

import torch

from torch import nn

from lattice_aligned.features import catalogue

from .orbit_features import orbit_catalogue, transform_descriptor

def normalize_descriptor_scale(descriptor):
    """Remove global length units; space-group identity depends on shape."""
    value = descriptor.clone()
    log_length = descriptor[:, :3].mean(1, keepdim=True)
    value[:, :3] -= log_length
    value[:, 6:7] -= 3*log_length
    value[:, 51:58] *= log_length.exp()
    return value

class OrbitSymmetryTransformer(nn.Module):
    descriptor_kind = 'proper_orbit'

    def __init__(self, mean, std, width=128, layers=4, orbit=True, metric_family=False, metric_fusion='conditional', scale_invariant=False, reference_log_lengths=None):
        super().__init__()
        if metric_fusion not in ('conditional', 'product'):
            raise ValueError('unknown_metric_fusion:'+metric_fusion)
        if scale_invariant and reference_log_lengths is not None:
            raise ValueError('choose_scale_normalization_or_marginalization')
        self.config = dict(width=width, layers=layers, orbit=orbit, metric_family=metric_family, metric_fusion=metric_fusion, scale_invariant=scale_invariant,
                           reference_log_lengths=reference_log_lengths)
        self.orbit = orbit
        self.metric_family = metric_family
        self.metric_fusion = metric_fusion
        self.scale_invariant = scale_invariant
        self.reference_log_lengths = reference_log_lengths
        self.frame_evaluations = (24 if orbit else 1)*(len(reference_log_lengths) if reference_log_lengths else 1)
        self.register_buffer('mean', torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(np.asarray(std), dtype=torch.float32))
        self.register_buffer('actions', torch.as_tensor(orbit_catalogue()[1], dtype=torch.float32))
        self.register_buffer('normals', torch.as_tensor(catalogue()[2], dtype=torch.float32))
        mapping = json.loads((workspace_root(__file__)/'lattice_aligned/assets/symmetry_122.json').read_text())['sg_to_class']
        self.register_buffer('mapping', torch.tensor(mapping, dtype=torch.long))
        self.register_buffer('membership', torch.tensor(np.eye(122)[mapping], dtype=torch.float32))
        self.global_embed = nn.Linear(33, width)
        self.residue = nn.ModuleList([nn.Linear(3*m**3, width) for m in (2, 3, 4, 6)])
        self.zone = nn.Linear(55, width)
        self.zone_position = nn.Linear(3, width, bias=False)
        self.operation = nn.Linear(6, width)
        self.operation_position = nn.Linear(9, width, bias=False)
        self.type_embed = nn.Parameter(torch.randn(1, 18+len(self.actions), width)*.02)
        block = nn.TransformerEncoderLayer(width, 4, width*3, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.fine = nn.Linear(width*2, 230)
        self.coarse = nn.Linear(width*2, 122)
        if metric_family:
            from .full_inference import bravais_from_space_group
            families = sorted({bravais_from_space_group(i) for i in range(1, 231)})
            family_index = [families.index(bravais_from_space_group(i)) for i in range(1, 231)]
            self.register_buffer('family_index', torch.tensor(family_index, dtype=torch.long))
            self.register_buffer('family_membership', torch.tensor(np.eye(len(families))[family_index], dtype=torch.float32))
            self.metric_head = nn.Sequential(nn.Linear(6+len(self.actions), 192), nn.LayerNorm(192), nn.GELU(),
                                             nn.Linear(192, 192), nn.LayerNorm(192), nn.GELU(), nn.Linear(192, len(families)))
            if metric_fusion == 'product':
                self.log_metric_strength = nn.Parameter(torch.tensor(-.7))

    def metric_log_probability(self, descriptor):
        lengths = descriptor[:, :3]-descriptor[:, :3].mean(1, keepdim=True)
        operations = torch.log1p(torch.expm1(descriptor[:, 1735::6]).clamp_min(0)/.001)/3.
        metric_input = torch.cat([lengths, descriptor[:, 3:6], operations], -1)
        return self.metric_head(metric_input).float().log_softmax(-1)

    def single(self, descriptor):
        if self.scale_invariant:
            descriptor = normalize_descriptor_scale(descriptor)
        x = ((descriptor-self.mean)/self.std).clamp(-12, 12)
        x = x.clone()
        x[:, 1735::6] = torch.log1p(torch.expm1(descriptor[:, 1735::6]).clamp_min(0)/.001)/3.
        tokens = [self.global_embed(torch.cat([x[:, :9], x[:, 51:75]], -1))[:, None]]
        cursor = 75
        for modulus, layer in zip((2, 3, 4, 6), self.residue):
            length = 3*modulus**3
            tokens.append(layer(x[:, cursor:cursor+length])[:, None]); cursor += length
        tokens.append(self.zone(x[:, 1020:1735].reshape(-1, 13, 55))+self.zone_position(self.normals)[None])
        tokens.append(self.operation(x[:, 1735:].reshape(-1, len(self.actions), 6))+self.operation_position(self.actions.flatten(1))[None])
        latent = self.norm(self.encoder(torch.cat(tokens, 1)+self.type_embed))
        pooled = torch.cat([latent[:, 0], latent[:, 1:].mean(1)], -1)
        fine = self.fine(pooled).float().log_softmax(-1)
        coarse = self.coarse(pooled).float().log_softmax(-1)
        denominator = (fine.exp()@self.membership).clamp_min(1e-30).log()
        probability = fine-denominator[:, self.mapping]+coarse[:, self.mapping]
        if self.metric_family:
            # Geometry determines the distribution over lattice families.
            # Reflection/intensity evidence decides the conditional SG within
            # each family. This is a learned probability, with no hard gate.
            family = self.metric_log_probability(descriptor)
            if self.metric_fusion == 'product':
                strength = self.log_metric_strength.exp().clamp_max(8.)
                probability = (probability+strength*family[:, self.family_index]).log_softmax(-1)
            else:
                mass = (probability.exp()@self.family_membership).clamp_min(1e-30).log()
                probability = probability-mass[:, self.family_index]+family[:, self.family_index]
        return probability

    def forward(self, descriptor):
        if self.reference_log_lengths is not None:
            # Integrate out absolute size using fixed training-derived sizes.
            # Shape, reciprocal-radius products, occupancies and intensities
            # remain unchanged. Every input uses the same sizes and weights.
            shape = normalize_descriptor_scale(descriptor)
            views = []
            for length in self.reference_log_lengths:
                value = shape.clone()
                value[:, :3] += length
                value[:, 6] += 3*length
                value[:, 51:58] *= float(np.exp(-length))
                views.append(value)
            expanded = torch.stack(views, 1).flatten(0, 1)
            if self.orbit:
                transforms = torch.arange(24, device=descriptor.device).repeat(len(expanded))
                expanded = transform_descriptor(expanded.repeat_interleave(24, 0), transforms)
            probability = torch.cat([self.single(v).exp() for v in expanded.split(128)])
            return probability.reshape(len(descriptor), self.frame_evaluations, 230).mean(1).clamp_min(1e-30).log()
        if not self.orbit:
            return self.single(descriptor)
        if len(descriptor) <= 4:
            # Batch the 24 equivalent views for serving. LayerNorm and attention
            # operate independently per view, so this preserves the group mean
            # while avoiding 24 small accelerator launches.
            expanded = descriptor.repeat_interleave(24, dim=0)
            transforms = torch.arange(24, device=descriptor.device).repeat(len(descriptor))
            value = self.single(transform_descriptor(expanded, transforms)).exp()
            return value.reshape(len(descriptor), 24, 230).mean(1).clamp_min(1e-30).log()
        total = torch.zeros((len(descriptor), 230), device=descriptor.device)
        for i in range(24):
            transform = torch.full((len(descriptor),), i, device=descriptor.device, dtype=torch.long)
            total += self.single(transform_descriptor(descriptor, transform)).exp()
        return (total/24).clamp_min(1e-30).log()
