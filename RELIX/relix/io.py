"""Read only measured fields from the simulator's integrated-reflection CSV."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def read_measurements(path, scan=0, crystal=None):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            raw = source["raw"].copy()
            geometry = json.loads(str(source["geometry_json"]))
        for name in ("center_px", "pixel_mm", "rotation_axis"):
            geometry[name] = np.asarray(geometry[name], dtype=float)
        return raw, geometry
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            candidates = [m for m in archive.getmembers()
                          if m.isfile() and m.name.endswith(f"/combined_matrix_{scan:04d}.csv")
                          and (crystal is None or m.name.split("/")[0] == str(crystal))]
            if len(candidates) != 1:
                raise ValueError("Expected one matching integrated scan in the archive")
            content = archive.extractfile(candidates[0]).read().decode("utf-8")
    else:
        content = path.read_text(encoding="utf-8")
    metadata = {}
    for line in content.splitlines():
        if line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
    if metadata.get("Output mode") != "integrated":
        raise ValueError("Use an integrated-reflection CSV")
    columns = [s.strip() for s in metadata["Columns"].split(",")]
    indices = [columns.index(k) for k in ("x_px", "y_px", "angle", "I_LP")]
    raw = np.loadtxt(io.StringIO(content), delimiter=",", usecols=indices, ndmin=2)
    geometry = dict(
        distance_mm=float(metadata["direct_dist (mm)"]),
        center_px=np.array([float(metadata["center_x (px)"]), float(metadata["center_y (px)"])]),
        pixel_mm=np.array([float(metadata["pixel_x (um)"]), float(metadata["pixel_y (um)"])]) / 1000,
        wavelength_A=float(metadata["wavelength (A)"]),
        rotation_axis=np.array(json.loads(metadata["rotation_axis"])),
        apply_lp_correction=metadata.get("apply_lp_correction", "True").lower() == "true")
    if raw.shape[1] != 4 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Invalid measured reflection array")
    return raw, geometry


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value
