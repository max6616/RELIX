"""Run one measurement through the unified RELIX model."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="RELIX.pt")
    p.add_argument("--input", type=Path, required=True, help="CSV, per-crystal .tar.gz, or measurement NPZ")
    p.add_argument("--scan", type=int, default=0, help="0-based scan within a per-crystal archive")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from . import load
    from .io import read_measurements, plain
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw, geometry = read_measurements(args.input, args.scan)
    model = load(args.model, args.device)
    result = model.predict_measured(raw, geometry)
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ("q", "basis", "hkl", "accepted", "cell", "orientation", "space_group",
              "space_group_probability", "diffraction_class", "diffraction_class_probability")
    np.savez_compressed(args.output / "indexing.npz",
                        **{k: np.asarray(result[k]) for k in fields if k in result})
    summary = {k: plain(result[k]) for k in
               ("cell", "space_group", "diffraction_class", "search_budget") if k in result}
    summary["reflections"] = len(raw)
    summary["accepted_reflections"] = int(np.asarray(result["accepted"]).sum())
    (args.output / "indexing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
