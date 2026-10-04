"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

import numpy as np

from lattice_aligned.reference_geometry import reciprocal_observations

def measured_features(raw, geometry):
    """Same measured-only transformation as the frozen CSV inference entrypoint."""
    q = reciprocal_observations(raw[:, :3], geometry)
    intensity = raw[:, 3].copy()
    if geometry['apply_lp_correction']:
        # Rotation around this axis leaves q dot axis and |q| invariant.
        axis = geometry['rotation_axis'] / np.linalg.norm(geometry['rotation_axis'])
        norm = np.linalg.norm(q, axis=1)
        theta2 = 2 * np.arcsin(np.clip(geometry['wavelength_A'] * norm / 2, -1, 1))
        cosalpha = np.clip(q @ axis / np.maximum(norm, 1e-12), -1, 1)
        sinalpha = np.maximum(np.sqrt(1 - cosalpha**2), 1e-12)
        sintheta = np.sin(theta2)
        sintheta = np.where(abs(sintheta) < 1e-12, 1e-12, sintheta)
        lp = (1 + np.cos(theta2)**2) / 2 / (sintheta * sinalpha)
        intensity /= lp
    intensity = np.log1p(1000 * intensity / max(float(intensity.max()), 1e-12)) / np.log(1001)
    return np.column_stack([q, intensity]).astype(np.float32)
