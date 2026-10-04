"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

import torch

from cctbx.uctbx.reduction_base import iteration_limit_exceeded

from .data import relation_tokens

from .inference import refine_q

from .consensus_model import ConsensusRelationTransformer

def gaussian_consensus(q, basis, scale):
    """Likelihood ratio of lattice peaks plus uniform background.

    A lattice peak's probability mass is proportional to reciprocal-cell
    volume. This density factor prevents a finer lattice from winning merely
    by fitting more points. Sigma and the inlier fraction are fitted from this
    observation alone. The ratio is invariant under a common q/basis scale.
    """
    h = np.rint(q@np.linalg.inv(basis).T)
    d2 = np.square(q-h@basis.T).sum(1)
    volume = max(abs(np.linalg.det(basis)), 1e-30)
    low, high = (scale*1e-4)**2, (scale*.2)**2
    variance = float(np.clip(np.quantile(d2, .2)/3, low, high))
    fraction = .8
    for _ in range(8):
        peak = np.log(volume)-1.5*np.log(2*np.pi*variance)-d2/(2*variance)
        signal = np.log(fraction)+peak
        total = np.logaddexp(np.log1p(-fraction), signal)
        weight = np.exp(signal-total)
        fraction = float(np.clip(weight.mean(), .01, .99))
        variance = float(np.clip(np.sum(weight*d2)/max(3*weight.sum(), 1e-20), low, high))
    peak = np.log(volume)-1.5*np.log(2*np.pi*variance)-d2/(2*variance)
    score = float(np.logaddexp(np.log1p(-fraction), np.log(fraction)+peak).mean())
    return score, fraction, float(np.sqrt(variance)/scale)

class ConsensusIndexer:
    def __init__(self, model, count=512, device='cpu'):
        self.model, self.count, self.device = model, count, device
        self.refinement_steps, self.refinement_tolerance = 32, 1e-6
        self.hypotheses = 1
        self.prefixes = 1
        self.diversity = 0.
        self.scoring = 'support'

    def distinct_topk(self, scores, vectors, count):
        """One beam entry per resolved antipodal relation direction/length.

        Vectors use the tokenizer's local-distance unit. At nonzero resolution,
        repeated estimates of one relation cannot occupy the entire beam.
        """
        if self.diversity <= 0:
            return scores.topk(count).indices
        values = scores.detach().cpu().numpy()
        order = np.argsort(-values, kind='stable')
        keep = []
        for index in order:
            if not np.isfinite(values[index]) or values[index] <= -9000:
                continue
            if keep:
                previous = vectors[keep]
                distance = np.minimum(np.linalg.norm(previous-vectors[index], axis=1),
                                      np.linalg.norm(previous+vectors[index], axis=1))
                if distance.min() <= self.diversity:
                    continue
            keep.append(int(index))
            if len(keep) == count:
                break
        return torch.tensor(keep, device=scores.device, dtype=torch.long)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=False)
        model = ConsensusRelationTransformer(**state['config']).to(device).eval()
        model.load_state_dict(state['model'])
        return cls(model, state['tokens'], device)

    @torch.inference_mode()
    def predict(self, q, intensity):
        q = np.asarray(q, float)
        token = relation_tokens(q, intensity, self.count)
        encoded, selected = self.model(torch.as_tensor(token['features'][None], device=self.device),
                                       torch.as_tensor(token['vectors'][None]/token['scale'], device=self.device),
                                       torch.as_tensor(token['mask'][None], device=self.device))
        chosen = selected[0].cpu().numpy()
        vectors = encoded['vectors'][0].cpu().numpy()*token['scale']
        if self.hypotheses > 1 or self.prefixes > 1:
            return self.structured_readout(q, token, encoded, selected, vectors)
        basis = vectors[chosen].T.astype(float)
        if abs(np.linalg.det(basis))/max(np.prod(np.linalg.norm(basis, axis=0)), 1e-20) < .005:
            raise ValueError('neural_consensus_basis_degenerate')
        basis, hkl, residual, history = refine_q(q, basis, self.refinement_steps, self.refinement_tolerance)
        return dict(basis=basis, hkl=hkl, residual=residual, accepted=residual < .1,
                    chosen=chosen, space_group=0, template=-1, correction_history=history,
                    network_forwards=1, linear_solves=len(history))

    def structured_readout(self, q, token, encoded, selected, vectors, scores=None):
        """Keep bounded neural continuations and score the complete scan.

        The encoder runs once. Candidates are network-ranked continuations of
        its first generator, with a configurable beam for the second and third.
        No classical lattice-candidate bank is constructed.
        Integer agreement is evaluated in a reduced basis, so arbitrary long
        generator choices do not change the physical residual scale.
        """
        from lattice_aligned.candidates import decode_basis
        mask = torch.as_tensor(token['mask'][None], device=self.device)
        normalized = torch.as_tensor(token['vectors'][None]/token['scale'], device=self.device)
        consensus_vectors = vectors/token['scale']
        if self.prefixes > 1:
            second_scores = self.model.scores(encoded, normalized, mask, [selected[:, 0]])[0]
            seconds = self.distinct_topk(second_scores, consensus_vectors, min(self.prefixes, int(mask.sum())-1))
        else:
            seconds = selected[0, 1:2]
        count = min(self.hypotheses, int(mask.sum()))
        alternatives = []
        for second in seconds:
            third_scores = scores if scores is not None and self.prefixes == 1 else self.model.scores(encoded, normalized, mask, [selected[:, 0], second.reshape(1)])[0]
            for third in self.distinct_topk(third_scores, consensus_vectors, count):
                neural_score = float(third_scores[third]) if self.prefixes == 1 else float(second_scores.log_softmax(0)[second]+third_scores.log_softmax(0)[third])
                alternatives.append((int(second), int(third), neural_score))
        proposals, solves, rejected = [], 0, []
        for second, third, neural_score in alternatives:
            chosen = np.array([int(selected[0, 0]), second, third])
            basis = vectors[chosen].T.astype(float)
            quality = abs(np.linalg.det(basis))/max(np.prod(np.linalg.norm(basis, axis=0)), 1e-20)
            if quality < .005:
                continue
            try:
                basis = np.linalg.inv(decode_basis(basis, niggli_epsilon=1e-4)['matrix'])
                basis, hkl, residual, warmup = refine_q(q, basis, steps=2)
                solves += len(warmup)
                frame = decode_basis(basis, niggli_epsilon=1e-4)
                fractional = q@frame['matrix'].T
                error = abs(fractional-np.rint(fractional)).max(1)
                support = float(np.exp(-(error/.1)**2).mean())
                candidate_basis = np.linalg.inv(frame['matrix'])
                likelihood, fraction, sigma = gaussian_consensus(q, candidate_basis, token['scale']) if self.scoring == 'likelihood' else (support, None, None)
                proposals.append(dict(basis=candidate_basis, chosen=chosen,
                                      support=support, score=likelihood, mixture_fraction=fraction, sigma=sigma,
                                      neural_score=neural_score, warmup=warmup))
            except (ValueError, np.linalg.LinAlgError, iteration_limit_exceeded, AssertionError) as exc:
                rejected.append(type(exc).__name__+': '+str(exc))
                continue
        if not proposals:
            raise ValueError('no_neural_continuation_admits_integer_consensus')
        if getattr(self, 'candidate_observer', None) is not None:
            self.candidate_observer(q, token, encoded, proposals)
        if getattr(self, 'candidate_ranker', None) is not None:
            learned = self.candidate_ranker.rank(q, token, encoded, proposals)
            for proposal, value in zip(proposals, learned):
                proposal['rank_score'] = float(value)
        best = max(proposals, key=lambda p:(p['score'], p['neural_score']))
        if getattr(self, 'candidate_ranker', None) is not None:
            best = max(proposals, key=lambda p:(p['rank_score'], p['score']))
        basis, hkl, residual, history = refine_q(q, best['basis'], self.refinement_steps, self.refinement_tolerance)
        return dict(basis=basis, hkl=hkl, residual=residual, accepted=residual < .1,
                    chosen=best['chosen'], space_group=0, template=-1, correction_history=best['warmup']+history,
                    network_forwards=1, linear_solves=solves+len(history), neural_proposals=len(alternatives),
                    viable_proposals=len(proposals), proposal_support=np.array([p['support'] for p in proposals]),
                    proposal_score=np.array([p['score'] for p in proposals]), proposal_rejections=np.array(rejected, dtype=str),
                    scoring=self.scoring,
                    proposal_neural_score=np.array([p['neural_score'] for p in proposals]),
                    candidate_rank_score=best.get('rank_score', best['score']))
