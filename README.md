# RELIX

RELIX indexes integrated single-crystal diffraction observations. Its single
`RELIX.pt` checkpoint contains the fixed inference configuration and all learned
components. The Python loader assembles the complete RELIX inference pipeline.

This release contains the inference algorithm, diffraction simulator, small
input examples, and commands to generate, read and index observations.
Downloads and dataset links are listed in the [v1.0.0 release](https://github.com/max6616/RELIX/releases/tag/v1.0.0).

`RELIX/` contains the model architecture and indexing algorithm; `simulator/`
contains the diffraction simulator; `examples/` contains small example inputs.
Source code is distributed through this repository. The simulation test
dataset is available on [Hugging Face](https://doi.org/10.57967/hf/10757).

## Downloads

The unified model is an attachment to the [v1.0.0 release](https://github.com/max6616/RELIX/releases/tag/v1.0.0). Download it before running inference:

```sh
curl -fL https://github.com/max6616/RELIX/releases/download/v1.0.0/RELIX.pt -o RELIX.pt
```

## Quick start

The tested platform is Linux x86-64. CPU inference requires Python, NumPy, SciPy,
PyTorch and CCTBX. `environment-inference.yml` describes the dependencies.
The inference deadline uses POSIX timers and runs on the main Python thread.

```sh
conda env create -f environment-inference.yml
conda activate relix
python run_relix.py --model RELIX.pt --input examples/measured_scan.npz --output output/example
```

The command writes `indexing.npz` (per-reflection integer indices and acceptance,
lattice basis, cell, orientation, and symmetry probabilities) and `indexing.json`
(a readable summary). The default device is CPU. CUDA is selectable with
`--device cuda` when a compatible PyTorch/CUDA installation is available.
The fixed default prediction deadline is 4.8 seconds; a slower computer can
return an earlier complete result or a timeout. Model loading is separate from
this prediction deadline.

For optional command installation from this directory:

```sh
python -m pip install --no-deps --no-build-isolation -e .
relix --model RELIX.pt --input examples/integrated_scan.csv --output output/csv
```

Python interface:

```python
import torch
import relix
from relix.io import read_measurements

torch.set_num_threads(1)
model = relix.load("/path/to/RELIX.pt", device="cpu")
observations, instrument = read_measurements("examples/integrated_scan.csv")
prediction = model.predict_measured(observations, instrument)
```

Only detector x/y, rotation angle, intensity and the supplied instrument geometry
enter this interface. See `DATA_FORMAT.md` for units and output conventions.

## Generate a simulated scan

The simulator has a separate environment because it uses DIALS/dxtbx and pyFAI.
`environment-simulator.yml` lists its dependencies. Its scientific source is the
version used to create the supplied primary corpus.

```sh
conda env create -f environment-simulator.yml
conda activate relix-simulator
python generate_simulation.py --cif examples/example.cif --count 10 --seed 20260913 --source-index 81804 --output generated/example
```

The supplied example CIF has archived source index 81804. For new structures,
choose a stable nonnegative source index and retain it with the master seed.
The generated CSVs have simulator reference labels in separate columns and
sidecars. Run one generated scan through RELIX:

```sh
conda activate relix
python run_relix.py --model RELIX.pt --input generated/example/combined_matrix_0000.csv --output output/generated
```

The original batch simulator entrypoint is also supplied in
`simulator/scripts/generate_ml_dataset_parallel.py`; its `--help` describes
generation from a CIF directory. The simulator produces integrated reflection tables; the intensity
definition is documented in `DATA_FORMAT.md`.

## Use the generated corpus

The full primary corpus has fixed train, val and test splits of 800,000,
100,000 and 100,000 scans, respectively. The first data release provides the
**test split: 10,000 source crystals and 100,000 scans**. Train and val are
organized separately for subsequent release. The split assignments are fixed
by crystal group, with ten scans per crystal.

The [release page](https://github.com/max6616/RELIX/releases/tag/v1.0.0) links to the [published test dataset](https://doi.org/10.57967/hf/10757). Download
both `RELIX-test-NNN.tar` volumes and the metadata ZIP. Follow its README
to extract `test/raw/` and `test/metadata/`. `DATA_INDEX.csv` maps crystal IDs
to volumes; `test.txt` lists the fixed scan identifiers.

```sh
python run_relix.py --model RELIX.pt --input /path/to/data/test/raw/2232561.tar.gz --scan 0 --output output/corpus
python read_simulation.py /path/to/data/test/raw/2232561.tar.gz --scan 0
```

Each raw archive includes integrated reflection tables, instrument metadata,
source CIF and reference labels. The reader can inspect a scan without loading
the model or export its measured input as an NPZ with `--output input.npz`.

## Package identity

`FILES.json` records paths, sizes and SHA-256 checksums. The single checkpoint
retains the original learned tensors; optimizer states and training logs are
omitted. The public interface is `relix.load` / `run_relix.py`.

The example CIF is from the archived Materials Project export, source-file ID 1.
Its file hash is recorded in `FILES.json`. Attribution and third-party
terms are listed in `THIRD_PARTY_NOTICES.md`.

## License and citation

RELIX and the supplied simulator are licensed under MIT; see `LICENSE`.
The released model parameters and simulation data use CC BY 4.0.
Source CIFs and dependencies retain their own attribution and license terms.
Please cite the software version in `CITATION.cff` and the test dataset
using `DATASET-CITATION.bib` or the BibTeX below.

## Dataset citation

The published test split contains 10,000 source crystals and 100,000 simulated scans. DOI: [10.57967/hf/10757](https://doi.org/10.57967/hf/10757); registered revision `0ab8d55`; CC BY 4.0.

```bibtex
@misc{relix_test_dataset_2026,
  author = {Zhang, Zhao and Dong, Zheng and Geng, Zhi and Chen, Rongchao and Dong, Xinlong and Wang, Changbo and Zhang, Yi and He, Gaoqi},
  title = {{RELIX} (Revision 0ab8d55)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {Hugging Face},
  doi = {10.57967/hf/10757},
  url = {https://doi.org/10.57967/hf/10757}
}
```
