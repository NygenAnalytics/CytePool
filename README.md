# CytePool

CytePool is a Python library for statistics on aggregated omics data: pseudobulk
profiles, built by pooling the cells of each sample and cell type, and ordinary bulk
RNA-seq. Its aim is to bring the sample-level methods that are standard in R into one
package with one data model. The status table below separates what is implemented from
what is planned.

**Status:** early development. Nothing is released yet.

## What is implemented

| Area | Implemented | Checked against |
| --- | --- | --- |
| Pseudobulk cubes | `Cube` (additive layers per sample, group and feature; merging groups; pooling samples; lazy layers, read as a selection needs them), `from_cells` (dense or CSR counts), `from_bulk` | scanpy, decoupler |
| Provenance and fingerprints | A record of each cube's source (the caller's record, kept unchanged), of the settings that built it and of what each operation kept, merged or pooled, with the fingerprint of the cube it was applied to; `Cube.history()` reads it back. A content fingerprint (SHA-256, documented encoding) of the tables, cell numbers, normalization and layers | a reimplementation of the documented encoding, zarr-python |
| Cube files | `write_cube`, `read_cube` (a subset of Zarr v3, with orthogonal selections; a read opens only the chunks that hold the selected features; gzip, and zstd on Python 3.14 or newer; stored digests, checked when a file is opened and when a read covers a whole group) | zarr-python |
| Normalization and filtering | TMM, upper-quartile and RLE factors, DESeq2 size factors, CPM, log-CPM, `filter_by_expression` | edgeR, DESeq2 |
| Designs | Model matrices from R formulas, named contrasts, design checks | R, limma, lme4 |
| Linear models | Ordinary, weighted and generalised least squares for every gene (in parallel when each gene needs its own fit), contrasts (named contrasts are matched to the coefficients by name) | limma |
| Precision weights for counts | `voom`: log-CPM values and observation weights from the mean-variance trend, whose LOWESS span is chosen from the number of genes as in limma, for a weighted linear model (limma-voom) | limma |
| Empirical Bayes moderation | Moderated t and F statistics, B-statistics, TREAT, with constant, trend and robust priors | limma |
| Multiple testing | Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg, Benjamini-Yekutieli | R |
| Differential expression | `limma_voom` and `limma_trend`: limma-voom and limma-trend on every group (cell type) of a cube, with the design matched to the cube by sample id, samples with too few cells left out, rank-deficient groups fitted on their column space with the contrasts they can estimate, and a provenance record for every result | limma, edgeR |
| Distributions and smoothers | Tails of the normal, t, F, chi-squared, gamma, beta and hypergeometric distributions, and quantiles of the first four; LOWESS, and limma's rule for its span; natural spline bases | mpmath, R, limma |

## Planned

These are not implemented yet:

- Negative binomial models (edgeR quasi-likelihood, DESeq2-style tests) and dispersion
  estimation
- Mixed models (dream-style fits, variance partitioning). A design can describe random
  effects, and its checks examine them, but no function estimates them yet.
  `fit_linear_model` fits one random intercept with a known correlation, given `block=`
  that groups the samples as the intercept does and `correlation=` (which CytePool does
  not estimate), and refuses a design whose random effects it would otherwise drop; its
  message says what can be fitted instead.
- Gene-set tests and pathway or transcription-factor activity
- Composition tests
- Ligand-receptor scoring and differential interactions
- Dose-response fits, benchmark doses and synergy
- Meta-analysis across studies

## Known limits

- `from_cells` needs the cell-level matrix in memory and returns dense layers:
  32 bytes per group x sample x feature with all five layers, 4 with counts only. For
  30 groups, 100 samples and 30,000 genes that is 2.9 GB, or 360 MB for counts only.
  Ask for the layers you need.
- A linear model fitted with observation weights or missing values stores a covariance
  matrix for every gene: 8 x genes x coefficients² bytes, so 16 MB for 20,000 genes and
  10 coefficients, and 1.6 GB at 100 coefficients.
- Such fits, in which each gene needs its own fit, run on numba's threads, as
  `from_cells` does. With numba's OpenMP threading layer, a process that has run a
  parallel routine cannot fork workers that run one again: start them with the
  "spawn" or "forkserver" method of `multiprocessing`. With one thread
  (`n_threads=1`), these fits run without a parallel region.
- A chunk of a cube file holds up to 4,096 features of every sample of one group, and a
  read decompresses every chunk that holds a selected feature. A few neighbouring
  features are therefore quick to read, but features scattered over the file, or a few
  samples of every feature, decompress most chunks of each selected group: 50 genes
  drawn at random from 30,000 usually fall in all eight chunks of a group. A read holds
  about what it returns plus one chunk, and with zstd also the decoder's window (2 MiB
  at the default level).
- Every read that covers a whole group of a cube file, in any order, is hashed and
  checked against the digest stored with the file. Reading every group of a gzip
  file of 20 groups x 100 samples x 30,000 genes took 23% longer (SHA-256 ran at
  2.3 GB/s with the processor's SHA instructions; without them it runs at about
  0.5 GB/s). A selection of some samples or features reads part of a group, which is
  checked only by `Cube.verify_fingerprint()`. The fingerprint of a lazy cube derived
  from a cube file reads the groups whose values the operation changed and that were
  not read whole before, and an operation applied to such a cube records the
  fingerprint of its input only if it was computed. The first operation applied to a
  cube in memory hashes its layers to record its fingerprint (about 1 s per GB).
- A lazy layer other than those of `read_cube` is read with `oindex` when it has one, as
  zarr-python arrays do, and otherwise one whole group at a time, as HDF5 datasets
  are. The cube copies what such a layer returns, so a read holds it twice for a moment.
- Integer layers and cell numbers hold whole numbers up to 2**63 - 1, the largest int64.
  A library size, merged group or pooled sample whose sum would exceed it raises
  `OverflowError` instead of wrapping around. Normalization works in float64, which is
  exact for counts up to 2**53.
- Empirical Bayes moderation uses limma's moment estimator of the prior (limma's
  `legacy=TRUE`). Since limma 3.61.8, limma uses a different estimator by default when the
  residual degrees of freedom differ between genes.
- Contrast standard errors and moderated F statistics are exact by default. limma's
  approximations are available as an option and can differ when weights vary between
  genes.
- `voom` normalizes by library-size factors only, and fits its trend without prior
  weights or a block correlation; `voomWithQualityWeights` and `voomLmFit` are not
  implemented. It needs a design of full column rank (limma drops aliased columns;
  `limma_voom` fits a group's column space), and it raises an error where limma returns
  values without meaning: for a sample whose library size is zero, or a trend that is
  not positive where weights are read.
- `voom` chooses the span of its trend from the number of genes, as limma does by
  default since limma 3.66, and limma 3.60 to 3.65 do with `adaptive.span = TRUE`.
  Their default, and that of earlier versions, is a span of 0.5: pass `span=0.5` to
  reproduce it.
- `limma_voom` and `limma_trend` fit fixed effects only: a design with random effects
  is refused, and there is no `duplicateCorrelation`. Normalization factors are TMM or
  none. In a group whose design is rank deficient they test every contrast the
  group's samples can estimate, including those that weigh an aliased column, which
  limma's `contrasts.fit` refuses, and they report the contrasts that cannot be
  estimated, which limma either refuses or, when they weigh no aliased column,
  computes with a changed meaning.
- numpy masked arrays, and array-likes that convert to one such as netCDF4 variables,
  are refused with a `TypeError` that names the argument, because numpy drops the
  mask when it converts one and the masked entries would be read as data. Where an
  argument accepts missing values, mark them with NaN in numbers
  (`numpy.ma.filled(values.astype(float), numpy.nan)`) and None in text
  (`values.tolist()`); otherwise remove them first. pandas objects work as before.

## Principles

- **Samples are the unit of replication.** Cells are pooled into samples before any
  test, and a cube records how many cells each pooled profile holds.
- **Two required dependencies:** numpy and numba.
- **Checked against the reference implementations.** Every method is compared with
  the package that defined it. The comparisons live in [`validation/`](validation/),
  and their results are reported with each change.
- **Implemented from the literature.** Methods whose reference packages carry
  copyleft licenses are written from their papers and documentation, never from their
  source code, so CytePool can stay BSD-licensed.

## Documentation

The documentation in [`docs/`](docs/) has a quick start, an API reference and a
[Validation](docs/source/validation.md) page that reports, for every validation
script, the reference, the measured differences, the tolerances and the outcome.
The site is not published yet; build it with
`uv sync --extra test --group dev --group docs` and `make -C docs html` (Node.js 18,
20, or 22 and newer), as
[the installation page](docs/source/installation.md#build-this-documentation)
describes.

## Development

```bash
uv sync --extra test --group dev
uv run pytest
uv run ruff check cytepool tests validation
uv run ruff format --check cytepool tests validation
uv run mypy cytepool
```

## License

BSD 3-Clause. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
