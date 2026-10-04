"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from .consensus_inference import gaussian_consensus

from .data import relation_tokens

from .full_inference import FullRelationIndexer, bravais_from_space_group

from .metric_projection import project_measured_symmetry, decode_measured_basis

class MeasuredRelationIndexer(FullRelationIndexer):
    def __init__(self, geometry, symmetry, device='cpu', project_geometry=False,
                 small_tokens=512, large_tokens=2048, expansion_fraction=.8,
                 calibration_steps=2):
        super().__init__(geometry, symmetry, device, project_geometry)
        self.small_tokens = small_tokens
        self.large_tokens = large_tokens
        self.expansion_fraction = expansion_fraction
        self.calibration_steps = calibration_steps
        self.symmetry_metric_frame = False
        self.native_beam_plane = False
        self.rigid_instrument = False
        self.symmetry_mapping = symmetry.mapping.detach().cpu().numpy() if hasattr(symmetry, 'mapping') else None

    def _geometry(self, q, intensity):
        scale = relation_tokens(q, intensity, self.small_tokens)['scale']
        candidates, errors = [], []
        forwards = solves = 0
        old_count = self.geometry.count
        try:
            for count in (self.small_tokens, self.large_tokens):
                self.geometry.count = count
                forwards += getattr(self.geometry, 'network_evaluations', 1)
                try:
                    pred = self.geometry.predict(q, intensity)
                    solves += pred['linear_solves']
                    quality = gaussian_consensus(q, pred['basis'], scale)
                    candidates.append((quality[0], count, quality, pred))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    errors.append(dict(tokens=count, error=str(exc)))
                if not getattr(self.geometry, 'adaptive_expansion', True):
                    break
                if candidates and (count == self.large_tokens or
                                   candidates[-1][2][1] >= self.expansion_fraction):
                    break
        finally:
            self.geometry.count = old_count
        if not candidates:
            raise ValueError('both_neural_relation_budgets_failed:'+str(errors))
        _, count, quality, pred = max(candidates, key=lambda item:item[0])
        pred.update(network_forwards=forwards, linear_solves=solves,
                    chosen_tokens=count, expanded=forwards > 1,
                    geometry_quality=np.asarray(quality), geometry_errors=errors,
                    # Failed proposal work cannot be reconstructed from an exception.
                    solve_count_complete=not bool(errors))
        return pred

    def _calibrate_instrument(self,xy,phi,panel_ids,metadata,basis,free_beam):
        if self.rigid_instrument:
            from .instrument_refinement import refine_multi_instrument
            scan=dict(xy_px=xy,phi_rad=phi,panel=panel_ids)
            C,gs,work=refine_multi_instrument([scan],[metadata],basis,
                self.calibration_steps,free_beam,detector_rotation=True,
                native_beam_plane=self.native_beam_plane)
            return C,gs[0],work
        from .instrument import refine_instrument
        return refine_instrument(xy,phi,panel_ids,metadata,basis,
            self.calibration_steps,free_beam,
            beam_mode='native_plane' if self.native_beam_plane else 'free')

    def _finish(self, q, intensity, pred):
        unprojected = decode_measured_basis(pred['basis'])
        pred['reduction_epsilon'] = unprojected['reduction_epsilon']
        pred['reduction_retries'] = unprojected['reduction_retries']
        reduced_h = np.asarray(q) @ unprojected['matrix'].T
        keep = abs(reduced_h-np.rint(reduced_h)).max(1) < .1
        if keep.sum() < 12:
            raise ValueError('insufficient_integer_consistent_points_for_symmetry')
        feature_frame = unprojected
        if self.symmetry_metric_frame:
            from robustness.metric_symmetry import adaptive_decode
            try:
                feature_frame = adaptive_decode(pred['basis'], q)
                pred['symmetry_frame_projection'] = feature_frame['projection']
            except (ValueError, np.linalg.LinAlgError) as exc:
                pred['symmetry_frame_error'] = str(exc)
        if getattr(self.symmetry, 'descriptor_kind', None) == 'reflection_set_profile':
            from .profile_features import describe_profile_orbit
            from .reflection_set import describe_reflection_set
            descriptor = np.r_[describe_profile_orbit(q[keep], intensity[keep], feature_frame),
                               describe_reflection_set(q[keep], intensity[keep], feature_frame).ravel()]
        elif getattr(self.symmetry, 'descriptor_kind', None) in ('adaptive_extra_profile','fixed_extra_profile'):
            from .profile_features import describe_profile_orbit, describe_profiles
            minimum = 32 if self.symmetry.descriptor_kind == 'adaptive_extra_profile' else None
            descriptor = np.r_[describe_profile_orbit(q[keep], intensity[keep], feature_frame),
                               describe_profiles(q[keep], intensity[keep], feature_frame, min_per_shell=minimum).ravel()]
        elif getattr(self.symmetry, 'descriptor_kind', None) == 'reflection_grid_profile':
            from .profile_features import describe_profile_orbit
            from .reflection_grid import describe_reflection_grid
            grid, meta = describe_reflection_grid(q[keep], intensity[keep], feature_frame)
            descriptor = np.r_[describe_profile_orbit(q[keep], intensity[keep], feature_frame), grid.ravel(), meta]
        elif getattr(self.symmetry, 'descriptor_kind', None) == 'local_intensity_profile':
            from .profile_features import describe_profile_orbit
            from .local_intensity import describe_local_intensity
            descriptor = np.r_[describe_profile_orbit(q[keep], intensity[keep], feature_frame),
                               describe_local_intensity(q[keep], intensity[keep], feature_frame).ravel()]
        elif getattr(self.symmetry, 'descriptor_kind', None) == 'profile_orbit':
            from .profile_features import describe_profile_orbit
            descriptor = describe_profile_orbit(q[keep], intensity[keep], feature_frame)
        elif getattr(self.symmetry, 'descriptor_kind', None) == 'proper_orbit':
            from .orbit_features import describe_orbit
            descriptor = describe_orbit(q[keep], intensity[keep], feature_frame)
        else:
            from lattice_aligned.features import describe
            descriptor = describe(q[keep], intensity[keep], geometry=feature_frame)['descriptor']
        logits = self.symmetry(torch.as_tensor(descriptor[None], device=self.device))
        probability = logits[0].softmax(0).cpu().numpy()
        space_group = int(probability.argmax())+1
        if self.symmetry_mapping is not None:
            class_probability=np.bincount(self.symmetry_mapping,weights=probability,minlength=122)
            predicted_class=int(class_probability.argmax())
            pred.update(diffraction_class=predicted_class,diffraction_class_probability=class_probability,
                        space_group_implied_class=int(self.symmetry_mapping[space_group-1]),
                        diffraction_class_members=np.flatnonzero(self.symmetry_mapping==predicted_class)+1)
        final = project_measured_symmetry(pred['basis'], space_group) if self.project_geometry else unprojected
        continuous = np.asarray(q) @ final['matrix'].T
        hkl = np.rint(continuous).astype(np.int64)
        residual = abs(continuous-hkl).max(1)
        before_symmetry = pred['basis'].copy()
        pred.update(cell=final['cell'], orientation=final['orientation'],
                    symmetry_actions=final['actions'], basis=np.linalg.inv(final['matrix']),
                    hkl=hkl, residual=residual, accepted=residual < .1,
                    space_group=space_group, template=final['template'],
                    metric_family=final['template_id'].split('-')[0],
                    neural_lattice_family=bravais_from_space_group(space_group),
                    space_group_probability=probability,
                    neural_basis_before_symmetry=before_symmetry,
                    metric_projection_strain=final.get('prediction_projection', {}).get('metric_strain', 0.))
        pred['metric_projection_info']=final.get('prediction_projection', {})
        pred['network_forwards'] += getattr(self.symmetry, 'frame_evaluations',
                                           24 if getattr(self.symmetry, 'orbit', False) else 1)
        if getattr(self, 'polish_ranker', None) is not None and not pred.get('post_polished', False):
            from .lattice_polish import polish_candidates
            original = pred['basis'].copy()
            _, candidates, errors = polish_candidates(q, original)
            raw_intensity = np.expm1(np.asarray(intensity)*np.log(1001))
            ranking = self.polish_ranker.rank(q, raw_intensity, original, candidates)
            pred['network_forwards'] += 1
            selected = int(np.argmax(ranking))
            best = candidates[selected]
            solves = next((p.get('polish_linear_solves') for p in candidates if 'polish_linear_solves' in p), 0)
            pred.update(post_polished=True, polish_candidates=len(candidates), polish_choice=best['tag'],
                        polish_score=float(ranking[selected]), polish_initial_score=float(ranking[0]),
                        polish_errors=len(errors), polish_linear_solves=solves)
            pred['linear_solves'] += solves
            pred['solve_count_complete'] = pred.get('solve_count_complete', False) and not bool(errors)
            if selected != 0:
                pred['basis'] = best['basis']
                return self._finish(q, intensity, pred)
        return pred

    @torch.inference_mode()
    def predict(self, q, intensity):
        q, intensity = np.asarray(q), np.asarray(intensity)
        return self._finish(q, intensity, self._geometry(q, intensity))

    @torch.inference_mode()
    def predict_measured(self, raw, geometry):
        """Planar detector, beam along z, supplied wavelength and rotation axis.

        Arbitrary DIALS detectors require their native measurement mapper. This
        interface follows the simulator's public instrument convention exactly.
        """
        from robustness.observations import measured_features
        from robustness.fixed_refine import correct
        x = measured_features(raw, geometry)
        pred = self._geometry(x[:, :3], x[:, 3])
        calibrated_geometry = geometry
        work = dict(linear_solves=0)
        if self.calibration_steps:
            try:
                basis, calibrated_geometry, result = correct(
                    raw, geometry, pred['basis'], steps=self.calibration_steps,
                    geometry_free=True, adaptive_weights=True, work=work)
                pred['basis'] = basis
                pred['calibration_history'] = result['history']
                x = measured_features(raw, calibrated_geometry)
            except (ValueError, np.linalg.LinAlgError) as exc:
                pred['calibration_error'] = str(exc)
        pred['linear_solves'] += work['linear_solves']
        pred['calibration_solves'] = work['linear_solves']
        pred['calibrated_geometry'] = calibrated_geometry
        pred['q'] = x[:, :3]
        return self._finish(x[:, :3], x[:, 3], pred)

    @torch.inference_mode()
    def predict_instrument(self, xy, phi, panel_ids, raw_intensity, metadata,
                           view='raw', free_beam=True, variance=None,
                           symmetry_min_isigma=None):
        """Full native-instrument inference on exported measurement-only inputs."""
        from .instrument import map_observations, refine_instrument
        if view not in ('raw', 'lp_qe'):
            raise ValueError('unknown_photometry_view:'+view)
        def observations(instrument):
            q, lp, qe = map_observations(xy, phi, panel_ids, instrument, True)
            intensity = np.maximum(raw_intensity, 0.)
            if view == 'lp_qe':
                intensity = intensity*lp/np.maximum(qe, 1e-12)
            normalized = np.log1p(1000*intensity/max(float(intensity.max()), 1e-12))/np.log(1001)
            return q, normalized
        q, intensity = observations(metadata)
        pred = self._geometry(q, intensity)
        calibrated = metadata
        pred['calibration_solves'] = 0
        if self.calibration_steps:
            try:
                C, calibrated, work = self._calibrate_instrument(xy, phi, panel_ids,
                              metadata, pred['basis'], free_beam)
                pred['basis'] = C
                pred['linear_solves'] += work['linear_solves']
                pred['calibration_solves'] = work['linear_solves']
                pred['calibration_history'] = work['history']
                pred['calibration_geometry_evaluations'] = work['geometry_evaluations']
                q, intensity = observations(calibrated)
            except (ValueError, np.linalg.LinAlgError) as exc:
                pred['calibration_error'] = str(exc)
                pred['solve_count_complete'] = False
        pred['calibrated_geometry'] = calibrated
        pred['q'] = q
        if symmetry_min_isigma is not None:
            if variance is None:raise ValueError('variance_required_for_significance_filter')
            selected=np.asarray(raw_intensity)/np.sqrt(np.maximum(variance,1e-20))>symmetry_min_isigma
            # The significance rule affects symmetry evidence only. Geometry
            # and final indices always use all supplied measured points.
            temporary=self._finish(q[selected],intensity[selected],pred)
            continuous=q@np.linalg.inv(temporary['basis']).T
            hkl=np.rint(continuous).astype(np.int64);residual=abs(continuous-hkl).max(1)
            temporary.update(hkl=hkl,residual=residual,accepted=residual<.1,
                             symmetry_significance_points=int(selected.sum()))
            return temporary
        return self._finish(q,intensity,pred)
