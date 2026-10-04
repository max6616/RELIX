# Third-party notices

## Materials Project

The example CIF and source structures for the simulation corpus originate from
the archived [Materials Project](https://materialsproject.org/) CIF export.
Materials Project data are distributed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The simulator generates diffraction observations and reference labels from these
structures. Local crystal/file identifiers and SHA-256 values are retained in
the dataset metadata; they are not asserted to be Materials Project entry IDs.
The original database release and download date were not recorded.

## CCTBX crystallographic tables

The supplied symmetry lookup data use CCTBX crystallographic information,
including `cctbx/sgtbx/sys_abs_equiv.py`. Their scientific source and construction
are recorded in `RELIX/lattice_aligned/assets/symmetry_122.json`.
The CCTBX copyright and license are reproduced in [licenses/CCTBX.txt](licenses/CCTBX.txt).
Upstream source: https://github.com/cctbx/cctbx_project/blob/master/LICENSE.txt

## Runtime dependencies

NumPy, SciPy, PyTorch, SymPy, CCTBX, DIALS, dxtbx, pyFAI, orix and PyYAML are
installed separately and retain their own licenses. Their implementations and
binaries are not included in this source snapshot.
