"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import copy

import json

import numpy as np

from .instrument import map_observations,shift_instrument,shift_beam_in_spindle_plane

def rotate_detector(metadata,vector):
    result=copy.deepcopy(metadata);vector=np.asarray(vector,float);angle=float(np.linalg.norm(vector))
    if angle==0:return result
    x,y,z=vector/angle;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    R=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    anchor=np.mean([p['origin'] for p in result['panels']],axis=0)
    for p in result['panels']:
        p['origin']=(anchor+R@(np.asarray(p['origin'])-anchor)).tolist()
        p['fast']=(R@p['fast']).tolist();p['slow']=(R@p['slow']).tolist()
    return result

def mapped(scans,metadata):
    return np.concatenate([map_observations(s['xy_px'],s['phi_rad'],s['panel'],g) for s,g in zip(scans,metadata)])

def shift_beam_plane(metadata,angle,spindle):
    return shift_beam_in_spindle_plane(metadata,angle,spindle)

def reciprocal_whitener(scan,metadata):
    """Propagate observed pixel/rotation variances into reciprocal coordinates."""
    variances=np.c_[scan['xy_variance_px'],scan['phi_variance_rad']]
    derivatives=[]
    for axis in range(3):
        step=1e-3 if axis<2 else 1e-6
        plus=scan['xy_px'].copy();minus=plus.copy()
        phi_plus=scan['phi_rad'].copy();phi_minus=phi_plus.copy()
        if axis<2:plus[:,axis]+=step;minus[:,axis]-=step
        else:phi_plus+=step;phi_minus-=step
        qp=map_observations(plus,phi_plus,scan['panel'],metadata)
        qm=map_observations(minus,phi_minus,scan['panel'],metadata)
        derivatives.append((qp-qm)/(2*step))
    derivative=np.stack(derivatives,axis=-1)
    covariance=(derivative*variances[:,None,:])@derivative.transpose(0,2,1)
    eigen,vectors=np.linalg.eigh(covariance)
    floor=np.maximum(eigen[:,-1:]*1e-6,1e-16);eigen=np.maximum(eigen,floor)
    return (vectors/np.sqrt(eigen)[:,None,:])@vectors.transpose(0,2,1)

def refine_multi_instrument(scans,metadata,basis,steps=2,free_beam=True,detector_rotation=True,centroid_covariance=False,robust_covariance=False,native_beam_plane=False):
    if len(scans)!=len(metadata) or not len(scans):raise ValueError('one_instrument_per_nonempty_scan_required')
    # Equal supplied detector poses share one correction. Distinct detector
    # settings receive separate pose corrections and share the crystal/beam.
    for g in metadata[1:]:
        if g['s0']!=metadata[0]['s0'] or g['wavelength_A']!=metadata[0]['wavelength_A']:
            raise ValueError('joint_calibration_requires_shared_initial_beam')
    keys=[json.dumps(g['panels'],sort_keys=True) for g in metadata]
    unique=list(dict.fromkeys(keys));groups=[unique.index(k) for k in keys]
    C=np.asarray(basis,float).copy();gs=copy.deepcopy(metadata);history=[];evaluations=0
    for _ in range(steps):
        q=mapped(scans,gs);evaluations+=len(scans)
        continuous=q@np.linalg.inv(C).T;h=np.rint(continuous);phase=abs(continuous-h).max(1)
        cutoff=float(np.clip(6*np.quantile(phase,.25),.02,.2));weight=np.clip(1-(phase/cutoff)**2,0,1)**2
        if (weight>.05).sum()<12:raise ValueError('too_few_consistent_joint_observations')
        residual_vector=q-h@C.T
        parameters=[]
        for group in range(len(unique)):
            parameters += [('translation',i,1e-3,group) for i in range(3)]
            if detector_rotation:parameters += [('rotation',i,1e-6,group) for i in range(3)]
        if free_beam:
            parameters += [('beam_native',0,1e-6,-1)] if native_beam_plane else [('beam',i,1e-6,-1) for i in range(2)]
        cols=9+len(parameters);J=np.zeros((len(q),3,cols))
        for i in range(3):J[:,i,3*i:3*i+3]=-h
        for j,(kind,axis,epsilon,group) in enumerate(parameters):
            delta=np.eye(1 if kind=='beam_native' else 2 if kind=='beam' else 3)[axis]*epsilon
            def perturb(g,value):
                if kind=='rotation':return rotate_detector(g,value)
                if kind=='beam_native':return shift_beam_plane(g,float(value[0]),metadata[0]['rotation_axis_lab'])
                return shift_instrument(g,translation=value if kind=='translation' else None,beam_tangent=value if kind=='beam' else None)
            plus=mapped(scans,[perturb(g,delta) if group in (-1,groups[k]) else g for k,g in enumerate(gs)])
            minus=mapped(scans,[perturb(g,-delta) if group in (-1,groups[k]) else g for k,g in enumerate(gs)]);evaluations+=2*len(scans)
            J[:,:,9+j]=(plus-minus)/(2*epsilon)
        residual_for_fit=residual_vector
        if centroid_covariance:
            whiten=np.concatenate([reciprocal_whitener(s,g) for s,g in zip(scans,gs)])
            evaluations+=6*len(scans)
            J=whiten@J;residual_for_fit=(whiten@residual_vector[:,:,None])[:,:,0]
            if robust_covariance:
                standardized=np.linalg.norm(residual_for_fit,axis=1)
                standardized_cutoff=max(4.685,float(6*np.quantile(standardized,.25)))
                weight*=np.clip(1-(standardized/standardized_cutoff)**2,0,1)**2
                if (weight>.05).sum()<12:raise ValueError('too_few_covariance_consistent_observations')
        A=(J*np.sqrt(weight)[:,None,None]).reshape(-1,cols);b=-(residual_for_fit*np.sqrt(weight)[:,None]).reshape(-1)
        scale=np.linalg.norm(A,axis=0).clip(1e-12);solution,residual,rank,singular=np.linalg.lstsq(A/scale,b,rcond=1e-8);change=solution/scale
        updates={(kind,group):np.zeros(1 if kind=='beam_native' else 2 if kind=='beam' else 3) for kind,_,_,group in parameters}
        for v,(kind,axis,_,group) in zip(change[9:],parameters):updates[kind,group][axis]=v
        amount=1.
        for (kind,group),value in updates.items():
            panel=gs[groups.index(group) if group>=0 else 0]['panels'][0]
            distance=abs(np.asarray(panel['origin'])@np.cross(panel['fast'],panel['slow']))
            bound=.2*distance if kind=='translation' else .05
            amount=min(amount,bound/max(np.linalg.norm(value),1e-12))
        change*=amount
        for k in updates:updates[k]*=amount
        C+=change[:9].reshape(3,3)
        gs=[shift_instrument(rotate_detector(g,updates.get(('rotation',groups[k]),np.zeros(3))),updates['translation',groups[k]],updates.get(('beam',-1))) for k,g in enumerate(gs)]
        if ('beam_native',-1) in updates:gs=[shift_beam_plane(g,float(updates['beam_native',-1][0]),metadata[0]['rotation_axis_lab']) for g in gs]
        if np.linalg.det(C)<=0:raise ValueError('joint_refinement_handedness_lost')
        history.append(dict(used=int((weight>.05).sum()),cutoff=cutoff,step_fraction=amount,rank=int(rank),columns=cols,
            weighted_rms=float(np.sqrt(np.average(np.sum(residual_vector**2,axis=1),weights=weight))),
            updates={str(k):v.tolist() for k,v in updates.items()}))
    return C,gs,dict(linear_solves=steps,geometry_evaluations=evaluations,history=history,detector_groups=groups)
