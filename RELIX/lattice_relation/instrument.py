"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import copy

import numpy as np

def unit(value):
    value=np.asarray(value,dtype=float)
    return value/np.linalg.norm(value,axis=-1,keepdims=True)

def map_observations(xy,phi,panel_ids,metadata,photometry=False):
    """Return reciprocal points, optionally LP and sensor efficiency factors.

    xy is observed pixel position; phi is instrument-derived rotation in radians.
    Panel axes/origins are resolved laboratory coordinates. Only simple and
    parallax-corrected planar pixels are accepted by the exporter.
    """
    s1=np.empty((len(xy),3),dtype=float)
    qe=np.ones(len(xy),dtype=float)
    for index,panel in enumerate(metadata['panels']):
        rows=np.asarray(panel_ids)==index
        if not rows.any():continue
        fast,slow,origin=[np.asarray(panel[key],dtype=float) for key in ('fast','slow','origin')]
        mm=np.asarray(xy[rows],dtype=float)*panel['pixel_mm']
        direction=unit(origin+mm[:,0,None]*fast+mm[:,1,None]*slow)
        normal=unit(np.cross(fast,slow))
        normal*=1 if origin@normal>=0 else -1
        if panel['strategy']=='ParallaxCorrectedPxMmStrategy':
            depth=panel['thickness_mm']/(direction@normal)
            mu=panel['mu_per_mm']
            path=1/mu-(depth+1/mu)*np.exp(-mu*depth)
            mm-=np.c_[direction@fast,direction@slow]*path[:,None]
            direction=unit(origin+mm[:,0,None]*fast+mm[:,1,None]*slow)
        elif panel['strategy']!='SimplePxMmStrategy':
            raise ValueError('unsupported_pixel_strategy:'+panel['strategy'])
        s1[rows]=direction/metadata['wavelength_A']
        if photometry and panel['mu_per_mm']>0:
            qe[rows]=-np.expm1(-panel['mu_per_mm']*panel['thickness_mm']/abs(direction@normal))
    s0=np.asarray(metadata['s0'],dtype=float)
    q=(s1-s0)@np.linalg.inv(np.asarray(metadata['setting_rotation'])).T
    axis=unit(metadata['rotation_axis_datum'])
    angle=-np.asarray(phi)[:,None]
    q=q*np.cos(angle)+np.cross(axis,q)*np.sin(angle)+axis*(q@axis)[:,None]*(1-np.cos(angle))
    q=q@np.linalg.inv(np.asarray(metadata['fixed_rotation'])).T
    if not photometry:return q
    direction=unit(s1)
    lab_axis=np.asarray(metadata['rotation_axis_lab'])
    lorentz=abs(direction@np.cross(lab_axis,unit(s0))) if metadata['rotation_scan'] else np.ones(len(q))
    pf=metadata['polarization_fraction'];pn=np.asarray(metadata['polarization_normal'])
    polarization=(1-2*pf)*(1-(direction@pn)**2)+pf*(1+(direction@unit(s0))**2)
    return q,lorentz/polarization,qe

def shift_instrument(metadata,translation=None,beam_tangent=None):
    result=copy.deepcopy(metadata)
    if translation is not None:
        for panel in result['panels']:panel['origin']=(np.asarray(panel['origin'])+translation).tolist()
    if beam_tangent is not None:
        initial=unit(result['s0']);tangent=np.eye(3)[:2]-initial[:2,None]*initial[None]
        result['s0']=(unit(initial+np.asarray(beam_tangent)@tangent)/result['wavelength_A']).tolist()
    return result

def shift_beam_in_spindle_plane(metadata,angle,spindle=None):
    """One beam rotation, matching DIALS' default fixed Mu1 and wavelength."""
    result=copy.deepcopy(metadata);direction=unit(result['s0'])
    spindle=result['rotation_axis_lab'] if spindle is None else spindle
    axis=unit(np.cross(direction,spindle))
    shifted=direction*np.cos(angle)+np.cross(axis,direction)*np.sin(angle)
    result['s0']=(shifted/result['wavelength_A']).tolist()
    normal=np.asarray(result['polarization_normal'])
    result['polarization_normal']=(normal*np.cos(angle)+np.cross(axis,normal)*np.sin(angle)+axis*(axis@normal)*(1-np.cos(angle))).tolist()
    return result

def refine_instrument(xy,phi,panel_ids,metadata,basis,steps=2,free_beam=True,beam_mode='free'):
    """Fixed robust least-squares updates of lattice and instrument metadata."""
    if beam_mode not in ('free','native_plane'):raise ValueError('unknown_beam_mode:'+beam_mode)
    C=np.array(basis,dtype=float,copy=True);g=copy.deepcopy(metadata)
    history=[];evaluations=0
    for _ in range(steps):
        q=map_observations(xy,phi,panel_ids,g);evaluations+=1
        fraction=q@np.linalg.inv(C).T;h=np.rint(fraction);phase=abs(fraction-h).max(1)
        cutoff=float(np.clip(6*np.quantile(phase,.25),.02,.2))
        weights=np.clip(1-(phase/cutoff)**2,0,1)**2
        if (weights>.05).sum()<12:raise ValueError('too_few_consistent_observations')
        beam_parameters=(1 if beam_mode=='native_plane' else 2) if free_beam else 0
        cols=12+beam_parameters
        jacobian=np.zeros((len(q),3,cols))
        for axis in range(3):jacobian[:,axis,3*axis:3*axis+3]=-h
        for axis in range(3):
            delta=np.eye(3)[axis]*1e-3
            plus=map_observations(xy,phi,panel_ids,shift_instrument(g,translation=delta))
            minus=map_observations(xy,phi,panel_ids,shift_instrument(g,translation=-delta))
            jacobian[:,:,9+axis]=(plus-minus)/2e-3;evaluations+=2
        if free_beam:
            for axis in range(beam_parameters):
                if beam_mode=='native_plane':
                    pg=shift_beam_in_spindle_plane(g,1e-6);mg=shift_beam_in_spindle_plane(g,-1e-6)
                else:
                    delta=np.eye(2)[axis]*1e-6
                    pg=shift_instrument(g,beam_tangent=delta);mg=shift_instrument(g,beam_tangent=-delta)
                plus=map_observations(xy,phi,panel_ids,pg)
                minus=map_observations(xy,phi,panel_ids,mg)
                jacobian[:,:,12+axis]=(plus-minus)/2e-6;evaluations+=2
        A=(jacobian*np.sqrt(weights)[:,None,None]).reshape(-1,cols)
        b=-((q-h@C.T)*np.sqrt(weights)[:,None]).reshape(-1)
        scale=np.linalg.norm(A,axis=0).clip(1e-12)
        change=np.linalg.lstsq(A/scale,b,rcond=1e-8)[0]/scale
        panel=g['panels'][0];distance=abs(np.asarray(panel['origin'])@np.cross(panel['fast'],panel['slow']))
        amount=min(1.,.2*distance/max(np.linalg.norm(change[9:12]),1e-12))
        if free_beam:amount=min(amount,.05/max(np.linalg.norm(change[12:]),1e-12))
        change*=amount;C+=change[:9].reshape(3,3)
        g=shift_instrument(g,translation=change[9:12],beam_tangent=change[12:] if free_beam and beam_mode=='free' else None)
        if free_beam and beam_mode=='native_plane':g=shift_beam_in_spindle_plane(g,float(change[12]))
        if np.linalg.det(C)<=0:raise ValueError('handedness_lost')
        history.append(dict(used=int((weights>.05).sum()),cutoff=cutoff,
                            translation_mm=change[9:12].tolist(),beam_tangent_step=change[12:].tolist(),step_fraction=amount))
    return C,g,dict(linear_solves=steps,geometry_evaluations=evaluations,history=history)
