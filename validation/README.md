# Validation

Each script in this folder checks one area of CytePool against an independent reference: the package that defined the method, or a high-precision computation. A script generates its inputs from fixed random seeds, runs the reference and CytePool on the same inputs, prints a table of the agreement for every check, and exits with an error when a check falls outside its tolerance.

With `--write-fixtures`, a script also saves a small set of inputs and reference outputs under `tests/data/`. The test suite compares CytePool against those files, so it runs in seconds and needs neither R nor the reference Python packages.

## Tolerances

| Tier | Applies to | Bound |
| --- | --- | --- |
| exact | closed-form results | relative difference at most 1e-10 |
| close | iterative fits | relative difference at most 1e-6 |
| calibrated | Monte Carlo methods | within simulation error, with the same calls |

Each script prints the tolerance of every check; some checks are stricter than their tier, and a few use a bound derived for their case.

## Requirements

- Python 3.13 or newer with numpy and numba. Some scripts also need scipy, mpmath, statsmodels or zarr; each script lists what it imports at the top.
- R 4.4 or newer with the Bioconductor and CRAN packages a script names. Set `RSCRIPT` to the `Rscript` executable when it is not on `PATH`.
- `interferon_stimulated_pbmcs.py` runs on published data: on its first run it downloads 77 MB from GEO (series GSE96583) into a cache directory outside the repository, and it checks the SHA-256 of every file before reading it. Its docstring says where the files go and how to choose another directory.
- `activity.py` needs a connection too: CytePool's loaders download CollecTRI (4.3 MB) and progeny (10.6 MB) once into `~/.cache/cytepool/networks`, checking their SHA-256, and decoupler's loaders and the checks of the published digests read Zenodo, Bioconductor and OmniPath.

Run a script from the repository root, against the working tree:

```bash
PYTHONPATH=. python validation/<script>.py
PYTHONPATH=. python validation/<script>.py --write-fixtures
```

## How reference packages are used

Reference packages run as separate programs that produce outputs to compare against. Methods are implemented from their papers and documentation; the source code of a package under a copyleft licence is not read, and what is adapted from permissively licensed packages, such as fgsea and decoupler, `THIRD_PARTY_NOTICES.md` records.
