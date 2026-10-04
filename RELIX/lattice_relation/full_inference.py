"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import torch

from cctbx import uctbx, sgtbx

from lattice_aligned.features import catalogue, describe

from lattice_aligned.candidates import decode_basis

from lattice_aligned.reference_geometry import cell_from_indexing_matrix

from multitask_indexer.geometry import proper_rotation

from .inference import RelationIndexer

from .symmetry_model import OperationTransformer

@lru_cache(None)
def bravais_from_space_group(number):
    group = sgtbx.space_group_info(number=int(number)).group()
    family = dict(Triclinic='a', Monoclinic='m', Orthorhombic='o', Tetragonal='t', Trigonal='h', Hexagonal='h', Cubic='c')[group.crystal_system()]
    centering = group.conventional_centring_type_symbol()
    if centering in ('A', 'B'):
        centering = 'C'
    return family+centering

def project_predicted_symmetry(basis, space_group):
    """Enforce the network's lattice family, choosing its closest metric setting.

    This does not select the space group or apply a covariance acceptance gate.
    The class is the unmodified network argmax. No reference is accepted here.
    """
    cell = cell_from_indexing_matrix(np.linalg.inv(basis))
    uc = uctbx.unit_cell(tuple(cell))
    cb = uc.change_of_basis_op_to_niggli_cell(relative_epsilon=1e-9)
    transform = np.array(cb.c_inv().r().as_double()).reshape(3, 3).T
    reduced = basis @ np.linalg.inv(transform)
    if np.linalg.det(reduced) < 0:
        reduced = -reduced
    gram = reduced.T @ reduced
    ev, vec = np.linalg.eigh(gram)
    whitener = (vec/np.sqrt(ev)) @ vec.T
    family = bravais_from_space_group(space_group)
    proposals = []
    for t in catalogue()[0]:
        if t['id'].split('-')[0] != family:
            continue
        actions = np.asarray(t['actions'])
        metric = (actions.transpose(0, 2, 1) @ gram @ actions).mean(0)
        strain = float(np.max(abs(np.linalg.eigvalsh(whitener @ (metric-gram) @ whitener))))
        proposals.append((strain, t['id'], metric))
    if not proposals:
        raise ValueError('unsupported_predicted_lattice_family:'+family)
    strain, _, metric = min(proposals, key=lambda p:p[:2])
    target = np.linalg.cholesky(metric).T
    orientation = proper_rotation(reduced @ np.linalg.inv(target))
    result = decode_basis(orientation @ target, niggli_epsilon=1e-4)
    result['prediction_projection'] = dict(space_group=int(space_group), family=family, metric_strain=strain)
    return result

class FullRelationIndexer:
    def __init__(self, geometry, symmetry, device='cpu', project_geometry=True):
        self.geometry, self.symmetry, self.device = geometry, symmetry, device
        self.project_geometry = project_geometry

    @classmethod
    def load(cls, geometry_path, symmetry_path, device='cpu'):
        geometry = RelationIndexer.load(geometry_path, device)
        state = torch.load(symmetry_path, map_location=device, weights_only=False)
        if state.get('model_kind') == 'orbit_symmetry':
            from .orbit_symmetry import OrbitSymmetryTransformer
            model = OrbitSymmetryTransformer(state['mean'], state['std'], **state['config']).to(device).eval()
        else:
            model = OperationTransformer(state['mean'], state['std'], **state['config']).to(device).eval()
        model.load_state_dict(state['model'])
        return cls(geometry, model, device)

    @torch.inference_mode()
    def predict(self, q, intensity):
        pred = self.geometry.predict(q, intensity)
        unprojected = decode_basis(pred['basis'], niggli_epsilon=1e-4)
        # Apply the inlier threshold in the same reduced frame that supplies
        # symmetry features. A max-component threshold in the neural generator
        # frame would otherwise depend on its arbitrary integer basis choice.
        reduced_h = np.asarray(q)@unprojected['matrix'].T
        keep = abs(reduced_h-np.rint(reduced_h)).max(1) < .1
        if keep.sum() < 12:
            raise ValueError('insufficient_integer_consistent_points_for_symmetry')
        if getattr(self.symmetry, 'descriptor_kind', None) == 'profile_orbit':
            from .profile_features import describe_profile_orbit
            descriptor = describe_profile_orbit(q[keep], intensity[keep], unprojected)
        elif getattr(self.symmetry, 'descriptor_kind', None) == 'proper_orbit':
            from .orbit_features import describe_orbit
            descriptor = describe_orbit(q[keep], intensity[keep], unprojected)
        else:
            descriptor = describe(q[keep], intensity[keep], geometry=unprojected)['descriptor']
        logits = self.symmetry(torch.as_tensor(descriptor[None], device=self.device))
        probability = logits[0].softmax(0).cpu().numpy()
        space_group = int(probability.argmax())+1
        final = project_predicted_symmetry(pred['basis'], space_group) if self.project_geometry else unprojected
        continuous = np.asarray(q) @ final['matrix'].T
        hkl = np.rint(continuous).astype(np.int64)
        residual = abs(continuous-hkl).max(1)
        pred.update(cell=final['cell'], orientation=final['orientation'], symmetry_actions=final['actions'], basis=np.linalg.inv(final['matrix']), hkl=hkl, residual=residual, accepted=residual < .1,
                    space_group=space_group, template=final['template'], space_group_probability=probability,
                    network_forwards=1+getattr(self.symmetry, 'frame_evaluations', 24 if getattr(self.symmetry, 'orbit', False) else 1),
                    metric_projection_strain=final.get('prediction_projection', {}).get('metric_strain', 0.),
                    neural_basis_before_symmetry=pred['basis'].copy())
        return pred
