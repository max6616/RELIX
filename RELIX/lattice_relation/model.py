"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import torch

from torch import nn

class GeometricAttention(nn.Module):
    """Local attention over observed displacement vectors, with learned weights."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.position = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.weight = nn.Sequential(nn.Linear(width, width//4), nn.GELU(), nn.Linear(width//4, 1))
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))

    def forward(self, x, relative, neighbors, neighbor_mask):
        z = self.norm(x)
        batch = torch.arange(len(x), device=x.device)[:, None, None]
        key, value = self.key(z)[batch, neighbors], self.value(z)[batch, neighbors]
        position = self.position(torch.cat([relative, relative.norm(dim=-1, keepdim=True)], -1))
        logits = self.weight(self.query(z)[:, :, None]-key+position).squeeze(-1)
        weight = logits.masked_fill(~neighbor_mask, -1e4).softmax(-1)
        x = x + self.out(((value+position)*weight[..., None]).sum(2))
        return x + self.ffn(x)

class RelationTransformer(nn.Module):
    def __init__(self, columns=16, width=96, latents=32, layers=2, local_relations=False, mlp_relations=False,
                 geometry_intensity=True, auxiliary_symmetry=True):
        super().__init__()
        if local_relations and mlp_relations:
            raise ValueError('choose_one_encoder')
        self.config = dict(columns=columns, width=width, latents=latents, layers=layers, local_relations=local_relations,
                           mlp_relations=mlp_relations, geometry_intensity=geometry_intensity, auxiliary_symmetry=auxiliary_symmetry)
        self.local_relations = local_relations
        self.mlp_relations = mlp_relations
        self.geometry_intensity = geometry_intensity
        self.auxiliary_symmetry = auxiliary_symmetry
        self.embed = nn.Sequential(nn.Linear(columns, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        if local_relations or mlp_relations:
            if local_relations:
                self.local = nn.ModuleList([GeometricAttention(width) for _ in range(layers)])
            else:
                self.local = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width)) for _ in range(layers)])
            self.pool = nn.Linear(width*2, width)
        else:
            self.latents = nn.Parameter(torch.randn(1, latents, width)*.02)
            self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
            layer = nn.TransformerEncoderLayer(width, 4, width*3, dropout=0., batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.back = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.pointer = nn.Sequential(nn.Linear(width*3+10, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.volume = nn.Linear(width, 1)
        if auxiliary_symmetry:
            self.space_group = nn.Linear(width, 230)
            self.template = nn.Linear(width, 41)

    def encode(self, features, mask, vectors=None):
        if not self.geometry_intensity:
            features = torch.cat([features[..., :9], torch.zeros_like(features[..., 9:11]), features[..., 11:]], -1)
        e = self.embed(features)
        if self.local_relations or self.mlp_relations:
            if vectors is None:
                raise ValueError('local_attention_requires_measured_vectors')
            if self.local_relations:
                distance = torch.cdist(vectors.float(), vectors.float()).masked_fill(~mask[:, None], 1e6)
                neighbors = distance.topk(min(16, vectors.shape[1]), largest=False).indices
                row = torch.arange(len(e), device=e.device)[:, None, None]
                relative = vectors[:, :, None]-vectors[row, neighbors]
                neighbor_mask = mask[row, neighbors]
                for layer in self.local:
                    e = layer(e, relative, neighbors, neighbor_mask)
            else:
                for layer in self.local:
                    e = e + layer(e)
            e = self.norm(e)
            mean = (e*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
            maximum = e.masked_fill(~mask[..., None], -1e4).amax(1)
            context = self.pool(torch.cat([mean, maximum], -1))
        else:
            latent = self.latents.expand(len(e), -1, -1)
            latent = latent + self.cross(latent, e, e, key_padding_mask=~mask, need_weights=False)[0]
            latent = self.norm(self.encoder(latent))
            context = latent.mean(1)
            e = self.norm(e + self.back(e, latent, latent, need_weights=False)[0])
        result = dict(tokens=e, context=context, volume=self.volume(context).squeeze(-1))
        if self.auxiliary_symmetry:
            result.update(sg=self.space_group(context), template=self.template(context))
        return result

    def scores(self, encoded, vectors, mask, chosen):
        e, context = encoded['tokens'], encoded['context']
        batch, count, _ = e.shape
        row = torch.arange(batch, device=e.device)
        selected = torch.zeros_like(context)
        v1 = torch.zeros((batch, 3), device=e.device, dtype=vectors.dtype)
        v2 = torch.zeros_like(v1)
        if chosen:
            selected = torch.stack([e[row, j] for j in chosen]).mean(0)
            v1 = vectors[row, chosen[0]]
        if len(chosen) > 1:
            v2 = vectors[row, chosen[1]]
        norm = vectors.norm(dim=-1).clamp_min(1e-6)
        n1, n2 = v1.norm(dim=-1).clamp_min(1e-6), v2.norm(dim=-1).clamp_min(1e-6)
        dot1, dot2 = (vectors*v1[:, None]).sum(-1), (vectors*v2[:, None]).sum(-1)
        det = (vectors*torch.linalg.cross(v1, v2)[:, None]).sum(-1)
        geo = torch.stack([norm.log(), dot1/(norm*n1[:, None]), dot2/(norm*n2[:, None]),
                           det.abs().clamp_min(1e-6).log(), dot1, dot2,
                           n1[:, None].expand_as(norm).log(), n2[:, None].expand_as(norm).log(),
                           encoded['volume'][:, None].expand_as(norm), torch.full_like(norm, len(chosen))], -1)
        joined = torch.cat([e, context[:, None].expand_as(e), selected[:, None].expand_as(e), geo], -1)
        score = self.pointer(joined).squeeze(-1).masked_fill(~mask, -1e4)
        for j in chosen:
            score = score.scatter(1, j[:, None], -1e4)
        return score

    def forward(self, features, vectors, mask):
        encoded = self.encode(features, mask, vectors)
        chosen = []
        for _ in range(3):
            chosen.append(self.scores(encoded, vectors, mask, chosen).argmax(1))
        return encoded, torch.stack(chosen, 1)
