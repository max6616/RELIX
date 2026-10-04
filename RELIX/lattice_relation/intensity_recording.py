"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

def record_finish_intensity(model):
    if getattr(model, '_records_finish_intensity', False):
        return model
    original = model._finish

    def finish(q, intensity, prediction):
        result = original(q, intensity, prediction)
        result['normalized_intensity'] = np.asarray(intensity).copy()
        return result

    model._finish = finish
    model._records_finish_intensity = True
    return model
