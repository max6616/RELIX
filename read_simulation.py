#!/usr/bin/env python3
"""Inspect a measured input, or export one scan from an original simulation archive."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "RELIX"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--scan", type=int, default=0)
    p.add_argument("--output", type=Path, help="Measurement-only NPZ output (for CSV or raw archive input)")
    args = p.parse_args()
    import numpy as np
    if args.input.suffix == ".npz" and args.output is None:
        with np.load(args.input, allow_pickle=False) as d:
            print(json.dumps({k: {"shape": list(d[k].shape), "dtype": str(d[k].dtype)} for k in d.files}, indent=2))
            if "offsets" in d.files:
                start, stop = d["offsets"][args.scan:args.scan + 2]
                print(f"scan={args.scan} reflections={int(stop-start)}")
        return
    from relix.io import read_measurements, plain
    raw, geometry = read_measurements(args.input, args.scan)
    print(json.dumps({"reflections": len(raw), "geometry": plain(geometry)}, indent=2))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, raw=raw, geometry_json=np.array(json.dumps(plain(geometry))))


if __name__ == "__main__":
    main()
