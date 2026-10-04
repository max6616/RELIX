"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

from .model import RelationTransformer

def invariant_features(features, vectors, scale_invariant=False):
    a, b = features[..., 3:6], features[..., 6:9]
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    n = vectors.norm(dim=-1).clamp_min(1e-8)
    cosine = (a*b).sum(-1)/(na*nb).clamp_min(1e-8)
    # All quantities are unchanged by a common orthogonal rotation.
    length_scale = features[..., 12]-features[..., 13] if scale_invariant else features[..., 12]
    radius = torch.zeros_like(length_scale) if scale_invariant else features[..., 13]
    return torch.stack([n.log(), torch.log1p(n), na, nb, cosine,
                        (a-b).norm(dim=-1), length_scale, radius,
                        features[..., 14]/10., features[..., 15]], -1)

class InvariantRelationBlock(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width*3)
        self.geometry = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, heads))
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*3), nn.GELU(), nn.Linear(width*3, width))

    def forward(self, x, edges, neighbors, neighbor_mask):
        query, key, value = self.qkv(self.norm(x)).chunk(3, -1)
        query, key, value = [t.reshape(*t.shape[:-1], self.heads, -1).transpose(1, 2) for t in (query, key, value)]
        bias = self.geometry(edges).permute(0, 3, 1, 2).masked_fill(~neighbor_mask[:, None], -1e4)
        update = F.scaled_dot_product_attention(query, key, value, attn_mask=bias).transpose(1, 2).flatten(-2)
        x = x+self.out(update)
        return x+self.ff(x)

class ConsensusRelationTransformer(RelationTransformer):
    def __init__(self, width=128, layers=4, neighbors=32, consensus=True, scale_invariant=False):
        nn.Module.__init__(self)
        self.config = dict(width=width, layers=layers, neighbors=neighbors, consensus=consensus, scale_invariant=scale_invariant)
        self.neighbors, self.consensus = neighbors, consensus
        self.scale_invariant = scale_invariant
        self.auxiliary_symmetry = False
        self.embed = nn.Sequential(nn.Linear(10, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([InvariantRelationBlock(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.pool = nn.Linear(width*2, width)
        self.affinity_query = nn.Linear(width, 32)
        self.affinity_key = nn.Linear(width, 32)
        self.affinity_geometry = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        self.validity = nn.Linear(width, 1)
        self.volume = nn.Linear(width, 1)
        self.pointer = nn.Sequential(nn.Linear(width*3+10, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        # Learn how strongly spatial closeness should enter relation affinity.
        self.log_precision = nn.Parameter(torch.tensor(5.))

    def encode(self, features, mask, vectors=None):
        if vectors is None:
            raise ValueError('measured_vectors_required')
        x = self.embed(invariant_features(features, vectors, self.scale_invariant))
        # Full attention avoids arbitrary top-k truncation of the many equal
        # distances in a lattice, and captures long-range generator relations.
        neighbors = torch.arange(vectors.shape[1], device=x.device)[None, None].expand(len(x), vectors.shape[1], -1)
        row = torch.arange(len(x), device=x.device)[:, None, None]
        vj = vectors[row, neighbors]
        vi = vectors[:, :, None]
        dot_local = (vi*vj).sum(-1)
        signs = torch.where(dot_local < 0, -1., 1.)
        aligned = vj*signs[..., None]
        displacement = aligned-vi
        distance = displacement.norm(dim=-1)
        ni, nj = vi.norm(dim=-1).clamp_min(1e-8), vj.norm(dim=-1).clamp_min(1e-8)
        edges = torch.stack([distance, distance.square().clamp_max(100), torch.log1p(distance),
                             dot_local.abs()/(ni*nj), (nj/ni).log(), nj.log()], -1)
        neighbor_mask = mask[:, None].expand(-1, vectors.shape[1], -1)
        for block in self.blocks:
            x = block(x, edges, neighbors, neighbor_mask)
        x = self.norm(x)
        mean = (x*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~mask[..., None], -1e4).amax(1)
        context = self.pool(torch.cat([mean, maximum], -1))
        logits = self.affinity_query(x)@self.affinity_key(x).transpose(-1, -2)/32**.5
        logits = logits+self.affinity_geometry(edges).squeeze(-1)-self.log_precision.exp().clamp_max(1e4)*distance.square()
        weight = logits.masked_fill(~neighbor_mask, -1e4).softmax(-1)
        corrected = vectors+(weight[..., None]*displacement).sum(2) if self.consensus else vectors
        return dict(tokens=x, context=context, volume=self.volume(context).squeeze(-1),
                    vectors=corrected, affinity_logits=logits, neighbors=neighbors,
                    validity=self.validity(x).squeeze(-1), neighbor_mask=neighbor_mask)

    def scores(self, encoded, vectors, mask, chosen):
        # Sign choice of individual relation tokens has no semantic meaning.
        # Use absolute dot products in the pointer to preserve that symmetry.
        e, context, vectors = encoded['tokens'], encoded['context'], encoded['vectors']
        row = torch.arange(len(e), device=e.device)
        selected = torch.zeros_like(context)
        v1 = torch.zeros((len(e), 3), device=e.device, dtype=vectors.dtype)
        v2 = torch.zeros_like(v1)
        if chosen:
            selected = torch.stack([e[row, j] for j in chosen]).mean(0)
            v1 = vectors[row, chosen[0]]
        if len(chosen) > 1:
            v2 = vectors[row, chosen[1]]
        norm = vectors.norm(dim=-1).clamp_min(1e-6)
        n1, n2 = v1.norm(dim=-1).clamp_min(1e-6), v2.norm(dim=-1).clamp_min(1e-6)
        dot1, dot2 = (vectors*v1[:, None]).sum(-1).abs(), (vectors*v2[:, None]).sum(-1).abs()
        det = (vectors*torch.linalg.cross(v1, v2)[:, None]).sum(-1).abs()
        geo = torch.stack([norm.log(), dot1/(norm*n1[:, None]), dot2/(norm*n2[:, None]),
                           det.clamp_min(1e-6).log(), dot1, dot2,
                           n1[:, None].expand_as(norm).log(), n2[:, None].expand_as(norm).log(),
                           encoded['volume'][:, None].expand_as(norm), torch.full_like(norm, len(chosen))], -1)
        joined = torch.cat([e, context[:, None].expand_as(e), selected[:, None].expand_as(e), geo], -1)
        score = self.pointer(joined).squeeze(-1).masked_fill(~mask, -1e4)
        for j in chosen:
            score = score.scatter(1, j[:, None], -1e4)
        return score
