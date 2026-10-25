---
description: CytePool brings the sample-level statistics of limma, edgeR and DESeq2 to Python for pseudobulk profiles and bulk RNA-seq, with numpy and numba as its only required dependencies.
---

# CytePool

**Sample-level statistics for pseudobulk and bulk omics data, in Python.**

CytePool brings the methods that are standard in R for aggregated omics data into one
Python package with one data model: pseudobulk profiles, built by pooling the cells of
each sample and cell type, and ordinary bulk RNA-seq. Its only required dependencies
are numpy and numba. Every method is compared with an independent reference, usually
the package that defined it, and the [Validation](validation.md) page reports how
closely they agree.

[Quick start](quickstart.ipynb) · [Differential expression per cell type](differential_expression.ipynb) ·
[Installation](installation.md) · [Validation](validation.md) ·
[API reference](/api/index.html)

:::{note} Early development
Nothing is released yet, and the interface may still change.
:::

## What is implemented

| Area | API | Checked against |
| --- | --- | --- |
| Pseudobulk cubes | [`Cube`, `from_cells`, `from_bulk`](/api/cubes.html) | scanpy, decoupler |
| Cube files | [`write_cube`, `read_cube`](/api/cube_files.html) | zarr-python |
| AnnData and MuData | [`to_anndata`, `from_anndata`, `to_mudata`, `from_mudata`](/api/anndata.html) | scanpy, decoupler, LIANA |
| Normalization and filtering | [TMM, upper-quartile and RLE factors, DESeq2 size factors, CPM, log-CPM, `filter_by_expression`](/api/normalization.html) | edgeR, DESeq2 |
| Designs | [Model matrices from R formulas, named contrasts, design checks](/api/designs.html) | R, limma, lme4 |
| Linear models | [Ordinary, weighted and generalised least squares for every gene, contrasts](/api/linear_models.html) | limma |
| Precision weights for counts | [`voom`](/api/voom.html) | limma |
| Empirical Bayes moderation | [Moderated t and F statistics, B-statistics, TREAT](/api/empirical_bayes.html) | limma |
| Multiple testing | [Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg, Benjamini-Yekutieli](/api/multiple_testing.html) | R |
| Differential expression | [`limma_voom`, `limma_trend` on every cell type of a cube](/api/differential_expression.html) | limma, edgeR |
| Distributions and smoothers | [Tails of the normal, t, F, chi-squared, gamma, beta and hypergeometric distributions, quantiles of the first four](/api/distributions.html); [LOWESS, natural spline bases](/api/smoothing.html) | mpmath, R |

The [README](https://github.com/NygenAnalytics/CytePool#planned) lists what is planned
(negative binomial and mixed models, gene-set and composition tests, and more) and the
known limits of what exists.

## Principles

- **Samples are the unit of replication.** Cells are pooled into samples before any
  test, and a cube records how many cells each pooled profile holds.
- **Two required dependencies:** numpy and numba. Numerical routines are compiled by
  numba the first time they run and cached on disk, so the first call of a function
  in a new environment is slower than the next.
- **Checked against the reference implementations.** Each area has a script in
  [`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation)
  that runs the reference package as a separate program on the same inputs; the
  [Validation](validation.md) page reports the results.
- **Implemented from the literature.** Methods whose reference packages carry
  copyleft licenses are written from their papers and documentation, never from their
  source code, so CytePool can stay BSD-licensed.

## Help and contributing

[Report a bug or ask a question](https://github.com/NygenAnalytics/CytePool/issues) on
GitHub. The [README](https://github.com/NygenAnalytics/CytePool#development) lists the
checks a change must pass, and [](installation.md) shows how to build this site.
