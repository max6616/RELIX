"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from torch import nn

from .orbit_features import orbit_catalogue

from lattice_aligned.features import intensity_statistics

SET_TOKENS=512

SET_WIDTH=8

def describe_reflection_set(q,intensity,frame,count=SET_TOKENS):
    """h,k,l, radial scale, two intensity scales, multiplicity, observed mask.

    Only observed rows enter. Missing reflections are never encoded as zero
    intensity. Duplicate indices are averaged before deterministic sampling.
    """
    q=np.asarray(q,float);h=np.rint(q@frame['matrix'].T).astype(np.int64)
    unique,inverse,counts=np.unique(h,axis=0,return_inverse=True,return_counts=True)
    raw=np.expm1(np.asarray(intensity,float)*np.log(1001))/1000
    merged=np.bincount(inverse,weights=raw)/counts
    uq=np.zeros((len(unique),3));np.add.at(uq,inverse,q);uq/=counts[:,None]
    radius=np.linalg.norm(uq,axis=1)
    e,_=intensity_statistics(radius,merged,min_per_shell=32)
    median_radius=max(float(np.median(radius)),1e-12)
    positive=merged[merged>0];median_i=max(float(np.median(positive)) if len(positive) else 1.,1e-12)
    tokens=np.c_[unique,np.log1p(radius/median_radius),np.log1p(e),np.log1p(merged/median_i),np.log1p(counts),np.ones(len(unique))]
    if len(tokens)>count:
        # Radius/intensity strata retain weak observations and the full measured
        # range; the frozen parent continues to use every observed reflection.
        radial=np.searchsorted(np.quantile(radius,np.linspace(0,1,17)[1:-1]),radius,side='right')
        bright=np.searchsorted(np.quantile(merged,np.linspace(0,1,5)[1:-1]),merged,side='right')
        bins=radial*4+bright
        # Sorting inside a stratum is independent of input row order.
        order=np.lexsort((unique[:,2],unique[:,1],unique[:,0],radius,merged,bins))
        groups=[order[bins[order]==k] for k in range(64)]
        chosen=[];cursor=0
        while len(chosen)<count:
            for g in groups:
                if cursor<len(g):chosen.append(int(g[cursor]))
                if len(chosen)==count:break
            cursor+=1
        tokens=tokens[np.asarray(chosen)]
    out=np.zeros((count,SET_WIDTH),np.float32);out[:len(tokens)]=tokens
    assert np.isfinite(out).all() and (out[:,-1]>0).any()
    return out

def transform_reflection_set(tokens,index):
    transforms=torch.as_tensor(orbit_catalogue()[0],device=tokens.device,dtype=tokens.dtype)[index]
    out=tokens.clone();out[:,:,:3]=tokens[:,:,:3]@transforms.transpose(-1,-2)
    return out

class ReflectionSetSpaceGroup(nn.Module):
    """Learned queries attend to individual reflection evidence in linear cost."""
    def __init__(self,width=128,layers=3,queries=32):
        super().__init__();self.config=dict(width=width,layers=layers,queries=queries)
        self.embed=nn.Sequential(nn.Linear(3+24+4,width),nn.LayerNorm(width),nn.GELU(),nn.Linear(width,width))
        self.queries=nn.Parameter(torch.randn(1,queries,width)*.02)
        self.cross=nn.ModuleList([nn.MultiheadAttention(width,4,dropout=.1,batch_first=True) for _ in range(layers)])
        self.norm1=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(width,width*3),nn.GELU(),nn.Dropout(.1),nn.Linear(width*3,width)) for _ in range(layers)])
        self.norm2=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.context=nn.Sequential(nn.Linear(230,width),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width*3,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,230))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        from cctbx import sgtbx
        laue=[sgtbx.space_group_info(number=i).group().laue_group_type() for i in range(1,231)]
        labels=sorted(set(laue));self.register_buffer('sg_to_laue',torch.tensor([labels.index(s) for s in laue]))

    def single(self,tokens,parent):
        h=tokens[:,:,:3];mod=torch.tensor([2,3,4,6],device=h.device,dtype=h.dtype)
        phase=2*torch.pi*h[:,:,:,None]/mod
        fourier=torch.cat([phase.cos().flatten(2),phase.sin().flatten(2)],-1)
        signed=h.sign()*h.abs().log1p()/4
        x=self.embed(torch.cat([signed,fourier,tokens[:,:,3:7].clamp(0,12)],-1))
        missing=tokens[:,:,-1]==0
        if missing.all(1).any():raise ValueError('empty_reflection_set')
        latent=self.queries.expand(len(tokens),-1,-1)
        for cross,n1,ff,n2 in zip(self.cross,self.norm1,self.ff,self.norm2):
            latent=n1(latent+cross(latent,x,x,key_padding_mask=missing,need_weights=False)[0])
            latent=n2(latent+ff(latent))
        context=self.context(parent.clamp_min(1e-30).log().clamp(-12,0))
        residual=self.head(torch.cat([latent.mean(1),latent.max(1).values,context],-1)).float()
        return (parent.clamp_min(1e-30).log()+residual).log_softmax(-1)

    def forward(self,tokens,parent):
        total=torch.zeros_like(parent,dtype=torch.float32)
        for frame in range(24):
            index=torch.full((len(tokens),),frame,device=tokens.device,dtype=torch.long)
            total+=self.single(transform_reflection_set(tokens,index),parent).exp()
        return (total/24).clamp_min(1e-30).log()

    def checkpoint(self):
        return dict(model_kind='reflection_set_spacegroup',model=self.state_dict(),config=self.config)

    @classmethod
    def load(cls,path,device='cpu'):
        state=torch.load(path,map_location=device,weights_only=False)
        model=cls(**state['config']).to(device);model.load_state_dict(state['model']);return model.eval()
