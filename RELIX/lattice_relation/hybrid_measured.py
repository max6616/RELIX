"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from cctbx.uctbx.reduction_base import iteration_limit_exceeded

from .measured_inference import MeasuredRelationIndexer

from .classical_control import ClassicalRelationControl

from .consensus_inference import gaussian_consensus

from .data import relation_tokens

class HybridMeasuredRelationIndexer(MeasuredRelationIndexer):
    def _classical(self, q, intensity):
        result = ClassicalRelationControl().predict(q, intensity)
        result.update(chosen_tokens=0, expanded=False, solve_count_complete=True,
                      geometry_source='classical', classical_attempted=True)
        return result

    def _run(self, q, intensity, refine=None):
        original_q, original_i = q, intensity
        try:
            primary = self._geometry(q, intensity)
            primary.update(geometry_source='neural', classical_attempted=False)
        except (ValueError, np.linalg.LinAlgError, iteration_limit_exceeded) as exc:
            primary = self._classical(q, intensity)
            primary.update(network_forwards=2, solve_count_complete=False,
                           neural_geometry_error=str(exc))
        if refine is not None:q, intensity, primary = refine(primary)
        if primary['geometry_source']=='neural' and primary.get('expanded',False):
            primary['classical_attempted'] = True
            secondary = None
            try:
                secondary = self._classical(original_q, original_i)
                if refine is not None:other_q, other_i, secondary = refine(secondary)
                else:other_q, other_i = original_q, original_i
                scale = relation_tokens(q, intensity, self.small_tokens)['scale']
                other_scale = relation_tokens(other_q, other_i, self.small_tokens)['scale']
                neural_quality = gaussian_consensus(q, primary['basis'], scale)[0]
                classical_quality = gaussian_consensus(other_q, secondary['basis'], other_scale)[0]
                choose_classical = classical_quality > neural_quality+1e-6
                work = {key:int(primary.get(key,0))+int(secondary.get(key,0))
                        for key in ('network_forwards','linear_solves','calibration_solves','calibration_geometry_evaluations')}
                complete = primary.get('solve_count_complete',False) and secondary.get('solve_count_complete',False)
                expanded = primary['expanded']
                if choose_classical:q, intensity, primary = other_q, other_i, secondary
                primary.update(work, expanded=expanded, classical_attempted=True,
                    solve_count_complete=complete, neural_candidate_quality=neural_quality,
                    classical_candidate_quality=classical_quality)
            except (ValueError, np.linalg.LinAlgError, iteration_limit_exceeded) as exc:
                primary['classical_error'] = str(exc)
                primary['solve_count_complete'] = False
                if secondary is not None:
                    for key in ('linear_solves','calibration_solves','calibration_geometry_evaluations'):
                        primary[key]=primary.get(key,0)+secondary.get(key,0)
        primary['q'] = q
        return self._finish(q, intensity, primary)

    @torch.inference_mode()
    def predict(self, q, intensity):
        return self._run(np.asarray(q), np.asarray(intensity))

    @torch.inference_mode()
    def predict_measured(self, raw, geometry):
        from robustness.observations import measured_features
        from robustness.fixed_refine import correct
        original = measured_features(raw, geometry)
        def refine(pred):
            x, calibrated = original, geometry
            work = dict(linear_solves=0)
            if self.calibration_steps:
                try:
                    basis, calibrated, result = correct(raw, geometry, pred['basis'],
                        steps=self.calibration_steps, geometry_free=True, adaptive_weights=True, work=work)
                    pred['basis']=basis;pred['calibration_history']=result['history']
                    x=measured_features(raw, calibrated)
                except (ValueError,np.linalg.LinAlgError) as exc:pred['calibration_error']=str(exc)
            pred['linear_solves']+=work['linear_solves'];pred['calibration_solves']=work['linear_solves']
            pred['calibrated_geometry']=calibrated
            return x[:,:3],x[:,3],pred
        return self._run(original[:,:3], original[:,3], refine)

    @torch.inference_mode()
    def predict_instrument(self, xy, phi, panel_ids, raw_intensity, metadata,
                           view='raw', free_beam=True, variance=None, symmetry_min_isigma=None):
        from .instrument import map_observations, refine_instrument
        if view not in ('raw','lp_qe'):raise ValueError('unknown_photometry_view:'+view)
        if symmetry_min_isigma is not None:raise ValueError('hybrid_significance_filter_not_enabled')
        def observations(instrument):
            q,lp,qe=map_observations(xy,phi,panel_ids,instrument,True)
            intensity=np.maximum(raw_intensity,0.)
            if view=='lp_qe':intensity=intensity*lp/np.maximum(qe,1e-12)
            return q,np.log1p(1000*intensity/max(float(intensity.max()),1e-12))/np.log(1001)
        original_q, original_i = observations(metadata)
        def refine(pred):
            q,intensity,calibrated=original_q,original_i,metadata
            pred['calibration_solves']=0
            if self.calibration_steps:
                try:
                    basis,calibrated,work=self._calibrate_instrument(xy,phi,panel_ids,
                        metadata,pred['basis'],free_beam)
                    pred['basis']=basis;pred['linear_solves']+=work['linear_solves']
                    pred['calibration_solves']=work['linear_solves'];pred['calibration_history']=work['history']
                    pred['calibration_geometry_evaluations']=work['geometry_evaluations']
                    q,intensity=observations(calibrated)
                except (ValueError,np.linalg.LinAlgError) as exc:
                    pred['calibration_error']=str(exc);pred['solve_count_complete']=False
            pred['calibrated_geometry']=calibrated
            return q,intensity,pred
        return self._run(original_q,original_i,refine)
