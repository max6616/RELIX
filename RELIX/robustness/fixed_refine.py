"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from lattice_aligned.reference_geometry import reciprocal_observations

def calibration_jacobian(raw, geometry):
    """dq / d(center_x_px, center_y_px, distance_mm), including de-rotation."""
    g=geometry
    ray=np.column_stack([(raw[:,:2]-g['center_px'])*g['pixel_mm'],np.full(len(raw),g['distance_mm'])])
    norm=np.linalg.norm(ray,axis=1);unit=ray/norm[:,None]
    jac=(np.eye(3)[None]-unit[:,:,None]*unit[:,None,:])/norm[:,None,None]
    jac=jac@np.diag([-g['pixel_mm'][0],-g['pixel_mm'][1],1.])/g['wavelength_A']
    axis=np.asarray(g['rotation_axis'])/np.linalg.norm(g['rotation_axis'])
    phi=np.deg2rad(-raw[:,2])[:,None]
    result=[]
    for j in range(3):
        v=jac[:,:,j]
        result.append(v*np.cos(phi)+np.cross(axis,v)*np.sin(phi)+axis*(v@axis)[:,None]*(1-np.cos(phi)))
    return np.stack(result,axis=-1)

def correct(raw, geometry, basis, steps=2, geometry_free=True, adaptive_weights=False, work=None):
    """Exactly `steps` robust linearized solves; no learned component or truth input.

    Wavelength, beam direction, and rotation-axis reference remain fixed.
    Scale cannot be inferred independently of wavelength from spot geometry.
    """
    g={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in geometry.items()}
    C=np.array(basis,dtype=float,copy=True)
    history=[]
    for _ in range(steps):
        q=reciprocal_observations(raw[:,:3],g)
        continuous=q@np.linalg.inv(C).T;h=np.rint(continuous)
        phase=np.max(abs(continuous-h),axis=1)
        cutoff=float(np.clip(6*np.quantile(phase,.25),.02,.2)) if adaptive_weights else .2
        weights=np.clip(1-(phase/cutoff)**2,0,1)**2
        if (weights>.05).sum()<12:raise ValueError('insufficient_consensus_for_fixed_refinement')
        residual=q-h@C.T
        cols=12 if geometry_free else 9
        jac=np.zeros((len(raw),3,cols))
        for i in range(3):jac[:,i,i*3:i*3+3]=-h
        if geometry_free:jac[:,:,9:]=calibration_jacobian(raw,g)
        weighted=(jac*np.sqrt(weights)[:,None,None]).reshape(-1,cols)
        rhs=(residual*np.sqrt(weights)[:,None]).reshape(-1)
        scale=np.linalg.norm(weighted,axis=0).clip(1e-12)
        if work is not None:work['linear_solves']=work.get('linear_solves',0)+1
        change=-np.linalg.lstsq(weighted/scale,rhs,rcond=1e-8)[0]/scale
        fraction=1.
        if geometry_free:
            fraction=min(1.,.2*g['distance_mm']/max(abs(change[11]),1e-12),100/max(np.linalg.norm(change[9:11]),1e-12))
        change*=fraction
        C+=change[:9].reshape(3,3)
        if geometry_free:
            g['center_px']+=change[9:11];g['distance_mm']+=change[11]
        if g['distance_mm']<=0 or np.linalg.det(C)<=0:raise ValueError('invalid_fixed_refinement_update')
        history.append(dict(retained=int((weights>.05).sum()),step_fraction=float(fraction),cutoff=cutoff,
                            weighted_rms=float(np.sqrt(np.average(np.sum(residual**2,axis=1),weights=weights)))))
    return C,g,dict(linear_solves=steps,geometry_free=geometry_free,history=history)
