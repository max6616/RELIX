"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from .consensus_inference import ConsensusIndexer, gaussian_consensus

from .consensus_model import ConsensusRelationTransformer

from .data import relation_tokens

from .robust_tokens import dense_relation_tokens

from .inference import refine_q

class RobustRelationIndexer(ConsensusIndexer):
    def __init__(self, model, count=512, device='cpu', sampling='dense', first_candidates=1):
        super().__init__(model, count, device)
        self.sampling = sampling
        self.radial_fraction = .5
        self.first_candidates = first_candidates

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=False)
        model = ConsensusRelationTransformer(**state['config']).to(device).eval()
        model.load_state_dict(state['model'])
        result=cls(model, state['tokens'], device, state.get('sampling', 'uniform'))
        result.radial_fraction=state.get('radial_fraction',.5)
        return result

    @torch.inference_mode()
    def predict(self, q, intensity):
        q = np.asarray(q, dtype=float)
        sampler = dense_relation_tokens if self.sampling == 'dense' else relation_tokens
        if self.sampling == 'stratified':
            from .stratified_tokens import stratified_relation_tokens
            token = stratified_relation_tokens(q, intensity, self.count, self.radial_fraction)
        else:
            token = sampler(q, intensity, self.count)
        features = torch.as_tensor(token['features'][None], device=self.device)
        vectors = torch.as_tensor(token['vectors'][None]/token['scale'], device=self.device)
        mask = torch.as_tensor(token['mask'][None], device=self.device)
        encoded, selected = self.model(features, vectors, mask)
        corrected = encoded['vectors'][0].cpu().numpy()*token['scale']
        if self.first_candidates == 1:
            if self.hypotheses > 1 or self.prefixes > 1:
                result = self.structured_readout(q, token, encoded, selected, corrected)
            else:
                chosen = selected[0].cpu().numpy()
                basis = corrected[chosen].T.astype(float)
                if abs(np.linalg.det(basis))/max(np.prod(np.linalg.norm(basis, axis=0)), 1e-20) < .005:
                    raise ValueError('neural_consensus_basis_degenerate')
                basis, hkl, residual, history = refine_q(q, basis, self.refinement_steps, self.refinement_tolerance)
                result = dict(basis=basis, hkl=hkl, residual=residual, accepted=residual < .1,
                    chosen=chosen, space_group=0, template=-1, correction_history=history,
                    network_forwards=1, linear_solves=len(history))
        else:
            logits = self.model.scores(encoded, vectors, mask, [])[0]
            firsts = self.distinct_topk(logits, corrected/token['scale'], self.first_candidates)
            predictions, errors = [], []
            solves = 0
            for first in firsts:
                second = self.model.scores(encoded, vectors, mask, [first.reshape(1)]).argmax(1)
                third = self.model.scores(encoded, vectors, mask, [first.reshape(1), second]).argmax(1)
                selected = torch.stack([first.reshape(1), second, third], dim=1)
                try:
                    pred = self.structured_readout(q, token, encoded, selected, corrected)
                    solves += pred['linear_solves']
                    quality = (pred['candidate_rank_score'] if getattr(self, 'candidate_ranker', None) is not None
                               else gaussian_consensus(q, pred['basis'], token['scale'])[0])
                    predictions.append((quality, pred))
                except ValueError as exc:
                    errors.append(str(exc))
            if not predictions:
                raise ValueError('no_first_generator_continuation_succeeded:'+str(errors))
            result = max(predictions, key=lambda x:x[0])[1]
            result.update(linear_solves=solves, first_candidates=len(firsts), first_candidate_errors=errors)
        result['sampling'] = self.sampling
        return result
