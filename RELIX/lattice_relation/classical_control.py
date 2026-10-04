"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from robustness.robust_candidates import build_candidates

from .metric_projection import decode_measured_basis

from .inference import refine_q

class ClassicalRelationControl:
    # A deterministic candidate bank is built once; token expansion is irrelevant.
    network_evaluations = 0
    adaptive_expansion = False

    def __init__(self):
        self.count = 512

    def predict(self, q, intensity):
        q = np.asarray(q, float)
        pack = build_candidates(q)
        chosen = pack['consensus_index']
        basis = np.linalg.inv(decode_measured_basis(pack['basis'][chosen])['matrix'])
        basis, hkl, residual, history = refine_q(q, basis, 32, 1e-6)
        return dict(basis=basis, hkl=hkl, residual=residual, accepted=residual < .1,
                    chosen=chosen, candidate_stats=pack['stats'],
                    correction_history=history, network_forwards=0,
                    linear_solves=len(history), geometry_kind='classical_translation_vote')
