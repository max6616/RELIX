#!/usr/bin/env python3
"""Generate integrated diffraction scans from one CIF with the frozen simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--source-index", type=int, default=0,
                   help="SeedSequence source index; archived indices are in crystals.jsonl")
    args = p.parse_args()
    if args.count < 1 or args.source_index < 0:
        p.error("count must be positive and source-index must be nonnegative")
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root / "simulator/scripts"), str(root / "simulator")]
    from generate_ml_dataset_parallel import process_cif
    cif = args.cif.resolve()
    if not cif.is_file():
        raise FileNotFoundError(cif)
    args.output.mkdir(parents=True, exist_ok=False)
    task = dict(cif_path=str(cif), cif_stem=cif.stem, cif_index=args.source_index,
                master_seed=args.seed, legacy_sample_seeds=None, rng_version="seedsequence-v1",
                n_per_cif=args.count, base_config={}, use_fz=True, intensity_threshold=1e-4,
                output_mode="integrated", show_crystal_info=True, export_primitive_labels=True,
                decimals=8, random_scan_start=True, resume=False,
                code_version=hashlib.sha256((root / "RELIX_PACKAGE.json").read_bytes()).hexdigest(),
                output_dir=str(args.output.resolve()))
    result = process_cif(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["fatal"] or result["failures"] or result["n_ok"] != args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
