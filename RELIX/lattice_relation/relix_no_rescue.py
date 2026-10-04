"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import copy

from xrdt_workspace import WorkspacePath as Path

import signal

import time

import numpy as np

import torch

from .consensus_inference import gaussian_consensus

from .data import relation_tokens

from robustness.observations import measured_features

class SearchBudgetExpired(BaseException):
    """Bypass optional legacy Exception handlers and unwind to the last complete result."""

class BudgetedIndexer:
    def __init__(self, model, settings, root):
        self.model, self.settings, self.root = model, settings, Path(root)
        self.base = model.postprocessor.model
        self.base.geometry.first_candidates = settings['first_candidates']
        self.base.polish_ranker = None
        self.last_result = None
        if settings.get("shared_symmetry_evidence", False):
            from .positive_reflection_evidence import reference_templates
            reference_templates()
        if settings.get("batched_symmetry", False):
            from .batched_symmetry_orbits import attach
            attach(model.postprocessor)

    def _commit(self, result):
        self.last_result = copy.deepcopy(result)
        callback = getattr(self, "on_candidate", None)
        if callback is not None:
            callback(self.last_result)

    def _quality(self, result, scale):
        return gaussian_consensus(result['q'], result['basis'], scale)

    def _readout(self, result):
        result = self.model.postprocessor.apply(result)
        # Each successful stage is a complete prediction, including probabilities.
        self._commit(result)
        if self.settings.get('shared_symmetry_evidence', False):
            from .budgeted_symmetry import readout
            result = readout(result, self.model.postprocessor.mapping)
        else:
            result, _ = self.model.finish(result, result['space_group_probability'])
        self._commit(result)
        return result

    @torch.inference_mode()
    def predict_measured(self, raw, geometry):
        start = time.perf_counter()
        deadline = start + self.settings['budget_seconds']
        self.last_result = None
        stages = []
        exhausted = False
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            raise RuntimeError('budgeted indexer requires ownership of the request timer')
        def expire(*_):
            raise SearchBudgetExpired()
        def stamp(name, since):
            stages.append({'stage': name, 'seconds': time.perf_counter()-since})
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, self.settings['budget_seconds'])
        try:
            x = measured_features(raw, geometry)
            scale = relation_tokens(x[:,:3], x[:,3], self.base.small_tokens)['scale']
            stage = time.perf_counter()
            try:
                result = self.base.predict_measured(raw, geometry)
                self._commit(result)
                result = self._readout(result)
            except (ValueError, np.linalg.LinAlgError):
                result = None
            stamp('fast_hybrid_and_readout', stage)
            quality = self._quality(result, scale) if result is not None else (-np.inf, 0., np.inf)
            good = (result is not None and int(np.sum(result['accepted'])) >= 12
                    and quality[2] <= self.settings['stop_sigma']
                    and quality[1] >= self.settings['stop_fraction'])
        except SearchBudgetExpired:
            exhausted = True
        except Exception as exc:
            if self.last_result is None:
                raise
            stages.append({"stage": "optional_candidate_error", "error": repr(exc)})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        if self.last_result is None:
            raise ValueError('budget_exhausted_without_complete_candidate' if exhausted else 'no_complete_candidate')
        result = self.last_result
        result['search_budget'] = dict(seconds=time.perf_counter()-start,
            limit_seconds=self.settings['budget_seconds'], exhausted=exhausted,
            stages=stages, selected_source=result.get('geometry_source'),
            retained_complete_candidate=True)
        return result
