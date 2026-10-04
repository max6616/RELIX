"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from .consensus_inference import gaussian_consensus

from .data import relation_tokens

from .intensity_recording import record_finish_intensity

from .posterior_readout import finish_projection_comparison

from .profile_features import describe_profile_orbit

from .reflection_set import ReflectionSetSpaceGroup, describe_reflection_set

from .local_spacegroup import compose

from .metric_projection import decode_measured_basis

from .full_inference import bravais_from_space_group

from robustness.posterior_primitive import posterior_primitive

class ContinuousPostprocessor:
    """Two readouts with unchanged search proposals and explicit work scope."""
    def __init__(self, measured_model, expert, strength=.75, threshold=.99):
        assert measured_model.symmetry.descriptor_kind == 'profile_orbit'
        self.model = record_finish_intensity(measured_model)
        self.expert = expert.eval()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.mapping = measured_model.symmetry_mapping
        self.laue = expert.sg_to_laue.detach().cpu().numpy()

    @torch.inference_mode()
    def fixed_probability(self, q, intensity, basis, original_probability):
        frame = decode_measured_basis(basis)
        fractional = np.asarray(q) @ frame['matrix'].T
        keep = np.abs(fractional - np.rint(fractional)).max(1) < .1
        if keep.sum() < 12:
            raise ValueError('insufficient_integer_consistent_points_for_symmetry')
        descriptor = describe_profile_orbit(q[keep], intensity[keep], frame)
        tokens = describe_reflection_set(q[keep], intensity[keep], frame)
        parent = self.model.symmetry(torch.as_tensor(descriptor[None], device=self.model.device)).float().exp()
        learned = self.expert(torch.as_tensor(tokens[None], device=self.model.device), parent).float().exp()
        refined = compose(parent, learned, self.expert.sg_to_laue, self.strength).exp()[0].cpu().numpy().astype(float)
        original_probability = np.asarray(original_probability, float)
        old_mass = np.bincount(self.laue, weights=original_probability)
        new_mass = np.bincount(self.laue, weights=refined)
        probability = refined / np.maximum(new_mass[self.laue], 1e-30) * old_mass[self.laue]
        return probability / probability.sum()

    @torch.inference_mode()
    def apply(self, result, initial_q=None, initial_intensity=None):
        original = result
        q = np.asarray(result['q'])
        intensity = np.asarray(result['normalized_intensity'])
        if intensity.shape != (len(q),):
            raise ValueError('recorded_intensity_and_geometry_row_count_mismatch')
        info = dict(applied=False, reason='original_search_not_triggered')
        if result.get('global_search', {}).get('triggered', False):
            if initial_q is None:
                raise ValueError('initial_measurements_required_for_triggered_search')
            if initial_intensity is None:
                initial_intensity = np.ones(len(initial_q))
            scale = relation_tokens(initial_q, initial_intensity)['scale']
            basis, info = posterior_primitive(result['basis'], q, scale, self.threshold)
            if info['applied']:
                try:
                    proposed = finish_projection_comparison(self.model, q, intensity, basis, scale)
                    info['readout_comparison'] = proposed['projection_readout_comparison']
                    score = gaussian_consensus(proposed['q'], proposed['basis'], scale)[0]
                    info['final_score_after_readout'] = score
                    if score > info['initial_score'] + 1e-10:
                        # Keep original search/instrument provenance; renew the full geometry readout.
                        result = dict(original)
                        retained_work = {'network_forwards', 'linear_solves', 'calibration_solves', 'solve_count_complete'}
                        result.update({key: value for key, value in proposed.items() if key not in retained_work})
                        result['geometry_source'] = 'posterior_primitive'
                    else:
                        info.update(applied=False, reason='no_likelihood_improvement_after_readout')
                except Exception as exc:
                    info.update(applied=False, reason='renewed_readout_failed', error=repr(exc))
        result = dict(result)
        result['posterior_readout'] = info
        result['geometry_symmetry_readout_sg'] = int(result['space_group'])
        try:
            probability = self.fixed_probability(np.asarray(result['q']), np.asarray(result['normalized_intensity']),
                                                 result['basis'], result['space_group_probability'])
            group = int(probability.argmax()) + 1
            class_probability = np.bincount(self.mapping, weights=probability, minlength=122)
            selected_class = int(class_probability.argmax())
            result.update(space_group=group, space_group_probability=probability,
                          diffraction_class=selected_class, diffraction_class_probability=class_probability,
                          space_group_implied_class=int(self.mapping[group - 1]),
                          diffraction_class_members=np.flatnonzero(self.mapping == selected_class) + 1,
                          neural_lattice_family=bravais_from_space_group(group),
                          fixed_symmetry_readout=dict(applied=True, variant='reflection_conditional', geometry_unchanged=True))
        except Exception as exc:
            result['fixed_symmetry_readout'] = dict(applied=False, error=repr(exc), geometry_unchanged=True)
        result['postprocessor_work_scope'] = 'Existing search counters exclude posterior candidate readouts and the final conditional symmetry pass; total postprocessor work is not reconstructed from selected-path counters.'
        result['network_forward_count_complete'] = False
        result['solve_count_complete'] = False
        return result
