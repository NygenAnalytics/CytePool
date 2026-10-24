---
description: How closely each method in CytePool agrees with the tool that defined it, from the tables the validation scripts printed on 2026-10-24.
---

(validation)=
# Validation

Every area of CytePool has a script in
[`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation) that
compares it with an independent reference: the package that defined the method, run
as a black box, or a high-precision computation. A script generates its inputs from
fixed random seeds, runs the reference as a separate program on the same inputs,
exchanging data through files in a temporary directory (float64 values as binary, so
they compare bit for bit), runs CytePool, prints a table with one row per check, and
exits with an error when any check falls outside its tolerance.

This page summarises the tables the scripts printed when they were run on
**2026-10-24**, against the package in this revision of the repository, with the
reference versions given in each section. Every script exited without a failure. Each
row below is a family of checks: it gives the number of printed checks it covers and
the largest difference among them. Run a script to see every check.

## Summary

| Script | Area | Reference | Checks | Outcomes | Run time | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| [`cube.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube.py) | Pseudobulk cubes | scanpy 1.12.4 | 20 | 19 pass, 1 information | 14.3 s | 926 MiB |
| [`pool_cells.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pool_cells.py) | Pooling cells into cubes | scanpy 1.12.4, decoupler 2.2.0 | 27 | 27 pass | 47.1 s | 1,751 MiB |
| [`cube_files.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_files.py) | Cube files | zarr-python 3.4.1 | 7 | 7 pass | 6.3 s | 179 MiB |
| [`normalization.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/normalization.py) | Normalization and filtering | edgeR 4.4.0, DESeq2 1.46.0 | 50 | 46 pass, 3 deliberate differences, 1 control | 72.9 s | 745 MiB |
| [`design.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/design.py) | Designs and contrasts | R 4.4.3, limma 3.62.1, lme4 2.0.6 | 27 | 26 pass, 1 information | 4.3 s | 221 MiB |
| [`linear_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/linear_models.py) | Linear models | limma 3.62.1 | 218 | 200 pass, 18 information | 5.5 s | 176 MiB |
| [`voom.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/voom.py) | Precision weights for counts | limma 3.62.1, edgeR 4.4.0 | 80 | 75 pass, 5 deliberate differences | 8.3 s | 289 MiB |
| [`empirical_bayes.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/empirical_bayes.py) | Empirical Bayes moderation | limma 3.62.1 | 73 | 73 pass | 14.7 s | 306 MiB |
| [`multiple_testing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/multiple_testing.py) | Multiple testing | R 4.4.3 | 15 | 15 pass | 2 min 56 s | 202 MiB |
| [`distributions.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/distributions.py) | Distributions and special functions | mpmath 1.3.0 | 46 | 46 pass | 6 min 39 s | 221 MiB |
| [`smoothing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/smoothing.py) | LOWESS and spline bases | R 4.4.3 | 90 | 87 pass, 1 information, 2 deliberate differences | 1.5 s | 175 MiB |
| **All** | | | **653** | **621 pass, none failed; 21 information, 10 deliberate differences, 1 control** | | |

Run time is the wall time of the whole script and peak memory the largest resident
set of any of its processes, reference programs included, both measured with GNU
`time` on the machine described under [](#validation-timings).

## How to read the tables

Each check is printed with its tolerance. Most tolerances follow three tiers:

| Tier | Applies to | Bound |
| --- | --- | --- |
| exact | closed-form results | relative difference at most 1e-10 |
| close | iterative fits | relative difference at most 1e-6 |
| calibrated | Monte Carlo methods | within simulation error, with the same calls |

Some checks are stricter, such as the special functions (1e-15 to 1e-12) and counts
or labels that must be identical (tolerance 0), and some use a bound derived for
their case: the variances from raw moments in `cube.py`, and the leverages of badly
scaled designs (1e-8, absolute). Differences are relative unless a row says
otherwise; scripts that compare statistics near zero scale them as their sections
describe. Besides **pass**, a printed check can be:

- **information**: a value reported for context, without a tolerance, such as how far
  limma's approximate standard errors are from exact ones;
- **deliberate difference**: an input where CytePool behaves differently from the
  reference on purpose, usually raising an error where the reference returns values
  without meaning; the row records what each program does;
- **control**: a check that must exceed the tolerance, to show that the comparison
  can tell nearly identical rules apart.

Sections also list deliberate differences that passing checks confirm, such as
formulas that CytePool rejects by design; the summary counts as deliberate
differences only the checks that the scripts print as such. No check is calibrated
yet: every method implemented so far is deterministic.

## Pseudobulk cubes

[`validation/cube.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube.py)
builds a cube from 28,946 simulated cells x 2,000 genes (density 0.210, 8 samples x
10 cell types, with an empty sample x cell type entry and entries with one and two
cells; seed 20261022) and compares its statistics, and those of `Cube.regroup` (10 to
4 groups) and `Cube.collapse_samples` (8 to 4 donors), with `scanpy.get.aggregate` run
on the same cells. scanpy normalizes the cells itself (`normalize_total`, then
`log1p`), which also checks the definitions of the normalized layers. Reference:
scanpy 1.12.4, anndata 0.13.4, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, Python
3.13.12.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Counts, detected cells, cells per entry and library sizes, for the cube, the regrouped and the collapsed cube | `sum`, `count_nonzero`, `n_obs_aggregated` | 0 (absolute) | 0 | 3 pass |
| Means of log-normalized and of normalized values | `mean` | 5.8e-15 | 1e-10 | 6 pass |
| Variances of log-normalized values (entries with two or more cells) | `var`, `dof=1` | 9.4e-13 | 1e-10 | 3 pass |
| Fractions of cells detecting each gene | `count_nonzero / n_obs_aggregated` | 0 | 1e-10 | 3 pass |
| The empty and the single-cell entry | no row in scanpy; variance NaN | NaN where expected | n/a | 1 pass |
| Variance from raw moments at mean-to-SD ratios r of 10, 800 and 80,000 | exact rational arithmetic | 1e-14, 5.6e-11, 1.3e-6 | 1.4e-12, 9.1e-9, 9.1e-5 | 3 pass |
| The same at r = 8e6 | exact rational arithmetic | 0.014 | n/a | 1 information |

The cube stores sums, so its variances come from raw moments, `sum_sq - sum**2 / n`.
Their relative error is about eps (1 + r²) for float64 machine epsilon eps; the
tolerance is a worst-case bound, 64 eps (1 + r²). At r = 8e6 that bound exceeds 0.5,
so the last row only shows where raw moments stop being useful. Log-normalized
expression has small r and stays in the exact tier.

## Pooling cells into cubes

[`validation/pool_cells.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pool_cells.py)
pools 29,068 simulated cells x 2,000 genes (12,097,859 stored counts, density 0.208,
8 samples x 10 cell types, cells in random order, with an empty entry, entries with
one and two cells, and genes never detected; seed 20261023) with `from_cells`, once
as integer counts stored as a float32 CSR matrix and once as non-integer counts, and
compares the cube with `scanpy.get.aggregate` (after scanpy normalized the cells in
float64) and with `decoupler.pp.pseudobulk`. Reference: scanpy 1.12.4 and decoupler
2.2.0, each with anndata 0.13.4, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, Python
3.13.12; CytePool with numpy 2.5.4 and numba 0.68.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Data type of the counts layer: uint32 for whole sums below 2³², float64 for fractions | the documented rule | uint32 and float64 | as the rule | 2 pass |
| Counts, detected cells and cells per entry | scanpy `sum`, `count_nonzero`, `n_obs_aggregated` | 0 | 0 (1e-10 for non-integer sums) | 4 pass |
| Means and variances of log-normalized values, means of normalized values | scanpy | 1.4e-11 | 1e-10 | 6 pass |
| The empty and the single-cell entry | scanpy | zero and NaN where expected | n/a | 2 pass |
| Sums, cells, library sizes and detected fractions | decoupler `pseudobulk` | 4.1e-16 | 0 for sums and cells of integer counts, else 1e-10 | 6 pass |
| Detected fraction of the entry without cells | decoupler (0 by convention) | NaN in the cube, 0 in decoupler | n/a | 2 pass |
| Per-sample sums, and `from_bulk` of them | decoupler without groups | 0 | 0 | 2 pass |
| The same counts as an unsorted CSR matrix with repeated and zero entries, as a dense array, and pooled with one thread | `from_cells` on the canonical CSR matrix | identical cube | identical | 3 pass |

For an entry without cells, the cube reports the fraction of cells detecting a gene
as NaN (zero cells out of zero), where decoupler reports 0. The script also times
`from_cells`; see [](#validation-timings).

## Cube files

[`validation/cube_files.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_files.py)
checks the Zarr v3 cube files against zarr-python, an independent implementation.
Reference: zarr-python 3.4.1, numcodecs 0.17.1, numpy 2.5.4, Python 3.13.12. CytePool
ran on Python 3.14.4 with numpy 2.5.4, so that the zstd cases ran too: CytePool reads
and writes zstd with the `compression.zstd` module of Python 3.14. On Python 3.13 the
script skips those cases.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| zarr-python opens cubes CytePool wrote (5,000 x 12 x 6, 2,000 x 40 x 1, 0 x 0 x 2 and 7 x 4 x 3 genes x samples x groups; gzip, uncompressed and zstd) and finds the same tables, cell numbers, layers and attributes | zarr-python | 0 values and attributes differ | 0 | 4 pass |
| CytePool reads 84 arrays zarr-python wrote: every core data type, partial edge chunks, both byte orders, gzip, zstd, gzip then zstd and no compression, and the default (with either separator) and v2 chunk key encodings | zarr-python | 0 elements differ | 0 | 1 pass |
| Chunks of those arrays left unstored because they equal the fill value | zarr-python | 295 of 840 chunks read as the fill value | more than 0 | 1 pass |
| `read_cube` of copies that zarr-python made node by node, with other chunks, big-endian bytes and gzip then zstd | zarr-python | 0 values differ | 0 | 1 pass |

## Normalization and filtering

[`validation/normalization.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/normalization.py)
compares `cytepool.normalization` with edgeR and DESeq2 on eight count matrices from
fixed seeds, of 6 to 40 samples and 200 to 2,000 features (dense, sparse, very sparse
with most upper quartiles zero, unequal depths, many ties, large groups, and two of
non-integer counts), on 300 small matrices with many ties for TMM's default reference
sample and its trimming boundaries, 100 of them chosen because a rule that rounds
differently picks another reference, and on hand-built edge cases. Reference: R 4.4.3,
edgeR 4.4.0, DESeq2 1.46.0; numpy 2.5.4.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| TMM factors: 32 option sets, and the default and unweighted reference on the 300 small matrices | edgeR `calcNormFactors` | 0; 6,738 of 6,738 values identical | 1e-10 | 3 pass |
| Upper-quartile factors, p = 0.75 and 0.9, and on the 300 small matrices | edgeR | 0; all 1,505 values identical, and 29 cases undefined in both | 1e-10 | 3 pass |
| RLE factors, with and without given library sizes | edgeR | 0; all 204 values identical, and 2 cases undefined in both | 1e-10 | 2 pass |
| DESeq2 size factors, ratio and poscounts, and poscounts on the 300 small matrices | DESeq2 `estimateSizeFactors` | 0; all 1,629 values identical, and 1 case undefined in both | 1e-10 | 3 pass |
| poscounts factors before their scaling to geometric mean one | DESeq2 `estimateSizeFactorsForMatrix` | 1e-15 (spread of the ratio) | 1e-10 | 1 pass |
| CPM, with normalization factors and with given library sizes | edgeR `cpm` | 0; 670,800 of 670,800 identical | 1e-10 | 3 pass |
| log2 CPM with prior counts 0.5, 2 and 5, with normalization factors and with given library sizes | edgeR `cpm(log = TRUE)` | 0 (absolute); 1,118,000 of 1,118,000 identical | 1e-10 | 5 pass |
| `filterByExpr` by group, with other cut-offs, with designs with and without an intercept, without group or design, and with normalization factors | edgeR `filterByExpr` | 0 of 88,200 features differ | 0 | 7 pass |
| Edge cases of TMM and the upper quartile: empty trimmed set, no shared features, one sample, identical profiles, ties at the trimming and A cut-offs, the rules for the default reference | edgeR | 0; 25 of 25 identical | 1e-10 | 11 pass |
| Edge cases of `filterByExpr`: counts at the CPM cut-off, all-zero and nearly collinear designs, sample thresholds a rounding error apart | edgeR | 0 of 23 features differ | 0 | 7 pass |
| Control: four rules for TMM's default reference that differ from edgeR's only in the last bits | edgeR | 0.97; factors change in 105 of the 300 small matrices | must exceed 1e-10 | 1 control |
| A negative count | edgeR rejects it too | both raise | n/a | 1 pass |
| Deliberate differences, listed below | edgeR | CytePool raises, edgeR returns values | n/a | 3 deliberate differences |

The functions repeat R's floating-point operations wherever an operation decides a
discrete outcome, such as which sample is TMM's reference or which genes a cut-off
keeps, so factors and filters agree to the last bit. The control shows that the test
matrices are sensitive to such details: rules that are equal in exact arithmetic but
round differently change the factors of 105 matrices. A case is undefined in both
when edgeR or DESeq2 returns NaN factors and CytePool raises an error, as for an
upper quartile of zero. Deliberate differences:

- A sample whose counts are all zero: edgeR's TMM with `refColumn` gives it factor 1;
  CytePool raises, since such a sample has no library size to normalize.
- An upper quartile of zero: edgeR returns NaN factors; CytePool raises.
- Both `group` and `design` given to `filterByExpr`: edgeR ignores the design;
  CytePool raises.

## Designs and contrasts

[`validation/design.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/design.py)
compares `cytepool.design` with R, limma and lme4 on a table of 24 samples: factors
with 2 to 4 levels, including mixed-case labels, labels with spaces and a
non-syntactic column name (`cell type`), a boolean, covariates that are multiples of
1/16, an integer covariate, and 10 badly scaled or nearly collinear covariates; seed
20261023. Reference: R 4.4.3, limma 3.62.1, lme4 2.0.6; R's session collation
C.UTF-8.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Model matrices of main effects, numeric covariates, factor:factor, factor:numeric and numeric:numeric interactions, no intercept, nesting (`/`, `%in%`), powers, `-` removal, booleans, missing margins and badly scaled covariates, each under 3 level orders | R `model.matrix` | 342 of 342 identical in column names, `assign`, term labels and every entry | all identical | 12 pass |
| 300 random formulas from the full grammar | R `model.matrix` | 300 of 300 identical | all identical | 1 pass |
| Default level order, code points, without levels given | R with `LC_COLLATE=C` | 4 of 4 identical | all identical | 1 pass |
| The same against R's default order in its session locale | R | 1 of 4 identical | n/a | 1 information |
| Formulas R also rejects, and formulas CytePool rejects by design | R `model.matrix` | 6 of 6 and 8 of 8 rejected | all rejected | 2 pass |
| `terms()` on 282 grammar cases without data (35 rejected by design, 11 by both) | R `terms` | 236 of 236 identical | all identical | 1 pass |
| Rank of 646 matrices, and the aliased columns of the 346 that are rank-deficient | R `qr()` (tolerance 1e-7, pivoting as `lm.fit` does) | 646 of 646 and 346 of 346 identical | all identical | 2 pass |
| Leverages of 604 matrices, and of 42 with badly scaled covariates | R `rowSums(qr.Q(qr(X))[, 1:rank]^2)` | 1e-13 and 6.3e-10 (absolute) | 1e-10 and 1e-8 | 2 pass |
| Estimability of 1,038 contrasts on the 346 rank-deficient designs: `X'u` estimable, random and aliased-column contrasts not | contrasts drawn by R | 1,038 of 1,038 right | all right | 1 pass |
| Random-effect terms of 21 specifications: grouping factors and effect columns | lme4 `lFormula` | 21 of 21 identical | all identical | 1 pass |
| `make_contrasts` on 13 linear expressions, and their names | limma `makeContrasts` | 0; names identical | 1e-10 | 2 pass |
| Expressions that are not linear: a constant, a product, a ratio, a power | limma `makeContrasts` | 4 of 4 rejected | all rejected | 1 pass |

Deliberate differences:

- Default level order is by code point in every locale, as R's is with
  `LC_COLLATE=C`. R's own default follows the session's collation, and agreed in 1 of
  the 4 formulas here.
- Formulas that R accepts and CytePool rejects: a `-` that removes nothing (`~ a - b`,
  `~ x - z`, `~ a + (-b)`), `^` of a single term (`~ x^2`, which R reads as `x`), a
  fractional power (`~ (a + b)^2.5`), and function calls (`~ log(z)`, `~ I(x^2)`,
  `~ factor(w)`), which CytePool's formulas do not support; transform the column in
  the sample table instead.
- Contrast expressions that limma turns into weights without meaning are rejected:
  limma gives `groupB - 1` the weights (-1, 0, -1, -1, -1), `groupA*groupB` (0, 0, 0,
  0, 0), `groupA/groupB` (Inf, 0, NaN, NaN, NaN) and `groupA^2` (1, 0, 0, 0, 0).

## Linear models

[`validation/linear_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/linear_models.py)
compares `fit_linear_model` with limma's `lmFit` and `contrasts_fit` with
`contrasts.fit` on 15 cases of 2,000 genes and 6 to 20 samples from fixed seeds: two
groups; groups with a covariate; covariates on a large scale; sample weights;
observation weights; group means with observation weights; about 5% missing values,
with and without sample weights, including genes that lose rank; a block correlation,
alone and with sample weights, observation weights or missing values; zero observation
weights;
a zero sample weight; and large-scale covariates with weights and missing values.
Reference: R 4.4.3, limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Coefficients, relative to the larger of their absolute value and standard error | `lmFit` | 9.7e-13 | 1e-10 | 15 pass |
| `stdev.unscaled`, `sigma` and `Amean` | `lmFit` | 2.1e-13 | 1e-10 | 45 pass |
| `df.residual` | `lmFit` | 0 genes differ | 0 | 15 pass |
| `cov.coefficients`, relative to products of standard deviations | `lmFit` | 1.8e-14 | 1e-10 | 15 pass |
| `contrasts_fit(method="limma")`: coefficients, unscaled standard errors and covariance | `contrasts.fit` | 9.7e-13 | 1e-10 | 45 pass |
| Named contrasts whose coefficients are listed in another order (a `Contrasts` over the reversed columns): coefficients and unscaled standard errors | `contrasts.fit` of `makeContrasts` in limma's order | 9.7e-13 | 1e-10 | 30 pass |
| `contrasts_fit(method="exact")`: coefficients and unscaled standard errors | `lmFit` on a design whose first coefficient is the contrast | 1.3e-12 | 1e-10 | 30 pass |
| Exact contrasts that are finite although not estimable | `lmFit` with the columns reversed | 0 entries | 0 | 5 pass |
| Coefficients limma reports although the data do not determine them, and estimable contrasts of such coefficients, which stay NaN | `lmFit` | up to 6 and 2 entries in a case | n/a | 10 information |
| limma's contrast standard errors over the exact ones | `contrasts.fit` against a refit | 0.56 to 2.55; up to 46% of entries off by more than 5% | n/a | 8 information |

Deliberate differences:

- Weights that name their coefficients, such as a `Contrasts` object, are matched to the
  fit's coefficients by name. limma's `contrasts.fit` takes the rows by position and
  only warns when the names differ (found by probing).
- A coefficient that a gene's observed values cannot estimate, because missing values
  or zero weights leave its column collinear with others, is NaN. limma drops the
  later of collinear columns and reports the rest, which can give a value for a
  coefficient the data do not determine. The script confirms that the NaN entries are
  exactly the coefficients whose limma value changes when the columns of the design
  are reversed.
- Contrast standard errors are exact by default: each comes from the gene's own
  covariance matrix. limma's `contrasts.fit` combines a gene's standard errors with
  the correlations of the coefficients in the design without weights. That is exact
  when one factorisation of the design serves every gene (no missing values, and no
  weights or only sample weights) and for contrasts of uncorrelated coefficients,
  such as group means; otherwise it is approximate, and here its values ranged from
  0.56 to 2.55 times the exact ones. `method="limma"` reproduces it.

With `--benchmark`, the script times both packages on 20,000 genes x 200 samples; see
[](#validation-timings).

## Precision weights for counts

[`validation/voom.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/voom.py)
compares `voom` with limma's `voom` on negative binomial counts from fixed seeds, 2 to
20,000 genes and 4 to 40 samples, in 19 cases: two groups, four groups with a
covariate, group means, a covariate without an intercept, one residual degree of
freedom, normalization factors (edgeR's `calcNormFactors` in R, `tmm_factors` here)
with and without given library sizes, two very small libraries, libraries from 1e3 to
1e8 reads, an intercept-only design, genes with the same counts and blocks of genes
with a single read (ties in the trend), rarefied counts, non-integer counts, spans of
0.3 and 1, a design without replication and two genes. Designs come from the same
sample table through R's `model.matrix` and `Design`, contrasts through
`makeContrasts` and named contrasts. Reference: R 4.4.3, limma 3.62.1, edgeR 4.4.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| log2 CPM `E`, and library sizes times normalization factors | `voom` | 0 | 1e-10 | 19 pass |
| Weights | `voom` | 5.2e-13 | 1e-10 | 18 pass |
| Trend: the points and LOWESS line saved with `save.plot = TRUE` | `voom` | 3.7e-14 | 1e-10 | 17 pass |
| Moderated t and p after `lmFit` with the weights, `contrasts.fit` and `eBayes` | limma | 5.0e-13 | 1e-10 | 17 pass |
| Rarefied counts: weights, trend, moderated t and p | limma | 2.2e-8, 8.5e-8, 8.9e-9 | 1e-6 | 3 pass |
| A design without replication: the warning, its weights being compared above | `voom` | both warn | n/a | 1 pass |
| Deliberate differences, listed below | `voom` | 4 where limma returns values and CytePool raises, 1 where limma fails and CytePool returns values | n/a | 5 deliberate differences |

Differences are relative, except for E, the trend's x and t, which are scaled by
max(\|reference\|, 1); the trend's y, scaled by max(\|reference\|, median of the
reference), because near zero both are rounding noise; and p-values, scaled by
max(p, 1e-300) max(1, \|log p\|). E was identical to limma's to the last bit in 19 of
19 cases, and so was the trend's x in 18 of 18. Rarefied counts have equal library
sizes, so a gene with the same count in every sample has a residual standard
deviation that is pure rounding noise in either program; those points of the trend,
and through them the weights, differ in the close tier. Deliberate differences:

- A sample without counts: limma gives it the log2 CPM of a count of 0.5 in a library
  of one read; CytePool raises.
- A trend that is not positive where weights are read from it: limma takes its
  inverse fourth power anyway; CytePool raises.
- Every gene in the trend with the same x: limma fails; CytePool uses a constant
  curve.
- A design with an aliased column: limma drops it; CytePool raises, as
  `fit_linear_model` needs a design of full column rank.
- A span above 1: limma accepts it; CytePool requires a span in (0, 1], as limma
  documents.

## Empirical Bayes moderation

[`validation/empirical_bayes.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/empirical_bayes.py)
compares `moderated_statistics`, `treat`, `squeeze_variances`, `fit_f_distribution`
and `fit_f_distribution_robust` with limma's `eBayes`, `treat`, `squeezeVar`,
`fitFDist` and `fitFDistRobustly`, all with `legacy = TRUE`. Linear model cases (2,000
genes, 6 to 40 samples) are fitted by `lmFit` in R and `fit_linear_model` here, then
moderated with a constant prior, a trend, robustly, and with both; they include
several contrasts, some linearly dependent, hypervariable genes, an infinite prior
df, missing values, observation weights, an all-zero contrast and as few as two genes.
Variance cases call the estimators directly on 2 to 2,000 variances, with and without
a covariate, few distinct or infinite covariate values, zero, missing and unusable
variances, planted outliers with equal and unequal df, and infinite robust prior df.
TREAT tests a fold change of 1.2. Reference: R 4.4.3, limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| t, p, B, F and its p-value, priors, posterior variances, total df and `var.prior`, with constant and trend priors | `eBayes`, `treat` | 6.8e-13 | 1e-10 | 16 pass |
| The same with robust priors | `eBayes`, `treat` | 5.1e-8 | 1e-6 | 7 pass |
| Missing values and observation weights, constant and trend priors | `eBayes`, `treat`; the exact F against limma's F of group means | 1.8e-13 | 1e-10 | 3 pass |
| Missing values and observation weights, robust priors | as above | 3.0e-11 | 1e-6 | 3 pass |
| Every component of `squeezeVar` and `fitFDist` | `squeezeVar`, `fitFDist` | 3.1e-13 | 1e-10 | 24 pass |
| Every component of `fitFDistRobustly`, with two choices of Winsorized tails | `squeezeVar`, `fitFDistRobustly` | 8.4e-8 | 1e-6 | 20 pass |

Each row gives the largest difference over all outputs of a case. Differences are
relative, except for t, B, F and outlier probabilities, scaled by
max(\|reference\|, 1), and p-values, scaled by max(p, 1e-300) max(1, \|log p\|):
about the error in log p relative to \|log p\|. The robust estimator is in the close
tier because limma solves its Winsorized moment equation only to a tolerance.
Deliberate differences, none of which is a failing check:

- The prior is estimated by limma's moment estimator, its `legacy = TRUE`. From limma
  3.62.0 (3.61.8 in development), `eBayes` uses another estimator by default when the
  residual degrees of freedom differ between genes (`fitFDistUnequalDF1`); when they
  are all equal, its default is the moment estimator too.
- The moderated F-statistic of a fit with missing values or observation weights is
  exact by default; `f_statistic="limma"` gives limma's approximation, which the
  `eBayes` rows compare. The exact F has no limma counterpart, so it is checked
  against limma's F of the same data coded as group means, whose coefficients are
  uncorrelated for every gene.
- `treat` reports a t-statistic of NaN for a coefficient that is NaN, and for an
  all-zero contrast tested with a zero threshold, where limma reports 0.

## Multiple testing

[`validation/multiple_testing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/multiple_testing.py)
runs every method of `adjust_p_values` (Holm, Hochberg, Hommel, Bonferroni,
Benjamini-Hochberg, Benjamini-Yekutieli and none) on each case and compares it with
R's `p.adjust`. Each check covers all seven methods. Reference: R 4.4.3,
`stats::p.adjust`.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| 2,000 uniform values; rounded values with ties, 0 and 1; 10% NaN; 10% exactly 0 or 1; n = 2,000 for 500 values with ties and NaN | `p.adjust` | 0 | 1e-10 | 5 pass |
| All 9,330 vectors of length 1 to 5 over {0, 0.01, 0.03, 0.5, 1, NaN}, with the default n and with n = 6 | `p.adjust` | 0 | 1e-10 | 2 pass |
| A 2,000 x 4 matrix adjusted by column, 0 to 50% NaN | `p.adjust` | 0 | 1e-10 | 1 pass |
| 50 random families of 2 to 2,000 values, with the default n and with n = 2,500 | `p.adjust` | 1.4e-16 | 1e-10 | 2 pass |
| 100,000 values, 10% duplicated | `p.adjust` | 0 | 1e-10 | 1 pass |
| The small cases written as test fixtures | `p.adjust` | 0 | 1e-10 | 4 pass |

Of the 290,224 p-values each method adjusted, all but 5 Hommel values were identical
to R's to the last bit.

## Distributions and special functions

[`validation/distributions.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/distributions.py)
checks every public function of `cytepool.stats.special` and
`cytepool.stats.distributions` on 19,493 points against mpmath evaluated at the exact
binary values of the inputs, with 50 significant digits (70 for intermediate steps);
quantile references are roots of the 70-digit tail probabilities. The grids reach
probabilities of 1e-300 (and log probabilities far below), shapes from 1e-3 to 1e6
(incomplete gamma shapes to 1e18 above the center), degrees of freedom from 0.5 to
1e12 and infinity, hypergeometric populations up to 1e13, batches of 2,000 moderated t
and quasi-likelihood F statistics, and cases where textbook methods lose accuracy.
The incomplete beta reference is DLMF 8.17.22 in 70-digit arithmetic, and the
hypergeometric reference sums exact ratios in fixed-point integer arithmetic.
Reference: mpmath 1.3.0, Python 3.13.12, numpy 2.5.4.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Log-gamma (x > 0, x < 0), digamma (x > 0, x < 0), trigamma, inverse trigamma, log-beta | mpmath | 9.99e-16 (digamma, x < 0) | 1e-15, or 2e-15 for digamma at x < 0 and log-beta | 7 pass |
| Regularized incomplete gamma and beta functions, their complements and logarithms | mpmath | 2.44e-13 | 1e-12 | 8 pass |
| Normal: cdf, sf, logcdf, logsf, ppf, isf | mpmath | 6.22e-16 | 1e-12 | 6 pass |
| t: cdf, sf, logsf, ppf, isf, and sf and logsf of 2,000 moderated t-statistics | mpmath | 1.95e-13 | 1e-12 | 7 pass |
| F: cdf, sf, logsf, ppf, isf, and sf and logsf of 2,000 quasi-likelihood F statistics | mpmath | 2.12e-13 | 1e-12 | 7 pass |
| Chi-squared: cdf, sf, logsf, ppf, isf | mpmath | 7.38e-14 | 1e-12 | 5 pass |
| Gamma and beta: cdf and sf | mpmath | 2.45e-13 | 1e-12 | 4 pass |
| Hypergeometric: sf and logsf | exact sums | 2.28e-13 | 1e-12 | 2 pass |

Errors are relative for values of at least 1e-300, relative errors of the logarithm
for the log functions, and relative to max(1, \|f\|) for log-gamma, log-beta and
digamma at negative x. The script checks its references too: the continued fraction
for the incomplete beta agrees with mpmath's own `betainc` to 6.6e-66 on the 1,554
points where both shapes are at most 1e3, and the log-gamma form of the first
hypergeometric probability agrees with exact binomial coefficients to 3.0e-66 on the
421 points with populations up to 1e6. For information it also evaluates scipy 1.18.1
on the same points. scipy is more accurate on some checks, such as the beta cdf and
sf (5.68e-15 and 9.99e-16, against 2.72e-14 and 8.58e-14 for CytePool) and the tails
of the 2,000 moderated t and quasi-likelihood F statistics (at most 3.45e-15, against
at most 1.25e-14). In the far tails of others it loses accuracy or returns values
that are not finite: its largest errors are 1.05 for `betainc`, 1.00 for the t
distribution's cdf and 0.193 for `gammainc` (1.94e-13, 1.95e-13 and 1.26e-13 for
CytePool), and it returns non-finite values at up to 493 points of a check
(`f_isf`).

## LOWESS and spline bases

[`validation/smoothing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/smoothing.py)
compares `lowess` with R's `stats::lowess` and `natural_spline_basis` with
`splines::ns`, on inputs from fixed seeds. LOWESS fits are compared relative to the
largest absolute fitted value (close tier). Spline bases are compared through their
column spaces: the least squares fits of random responses on either basis (exact
tier), their ranks and their rows of NaN. Reference: R 4.4.3.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| LOWESS of mean-variance trends of 2,000 to 20,000 genes, as limma and voom smooth them | `lowess` | 0 | 1e-6 | 5 pass |
| Spans from 0.1 to 1.7, 0 to 4 robustness iterations, delta of 0, the default, larger and beyond the range | `lowess` | 0 | 1e-6 | 13 pass |
| Ties: integer x, pairs, a tie block wider than the neighbourhood, ties at both ends | `lowess` | 0 | 1e-6 | 8 pass |
| 1 to 5 points | `lowess` | 0 | 1e-6 | 6 pass |
| Heavy tails, constant x or y, zero median absolute residual, neighbourhoods whose weights are all zero, spans whose product with n is just below an integer, points at the cut-offs of the tricube and bisquare weights and of the local constant fit | `lowess` | 0 | 1e-6 | 14 pass |
| x or y on large or small scales | `lowess` | 0 | 1e-6 | 3 pass |
| Fitted values identical to R's to the last bit | `lowess` | 51,905 of 51,905 | n/a | 1 information |
| Spline bases: df with and without an intercept, explicit and unsorted knots, boundary knots inside and beyond the data | `splines::ns` | 2.7e-15 | 1e-10 | 15 pass |
| Repeated knots, ties, few distinct values (rank-deficient bases), quantile knots that fall on a boundary | `splines::ns` | 1.7e-15 | 1e-10 | 15 pass |
| NaN in x, and covariates of 2,000 genes | `splines::ns` | 2.7e-15 | 1e-10 | 4 pass |
| Inputs for which R fails, where CytePool must fail too | `splines::ns` | 0 mismatches | 0 | 4 pass |
| Deliberate differences, listed below | `splines::ns` | R returns a basis, CytePool raises | n/a | 2 deliberate differences |

Deliberate differences: where R's `ns` warns and increases a `df` below 1 plus the
intercept, and where it accepts an interior knot on the lower boundary knot, returning
a basis with a degenerate column, `natural_spline_basis` raises an error.

(validation-timings)=
## Timings

The scripts ran one after another on a laptop with an Intel Core Ultra 7 265U, under
Linux 6.18 in WSL2 with 8 logical CPUs and 31 GiB of memory, with numba, OpenMP,
OpenBLAS and MKL limited to 2 threads (`NUMBA_NUM_THREADS=2` and so on). The machine
was shared with other work, so timings are indicative.

`from_cells` on the 29,068 cells x 2,000 genes of `pool_cells.py`, best of three after
a first run, building all five layers:

| Run | Seconds |
| --- | --- |
| `from_cells`, 2 threads | 0.159 |
| `from_cells`, 1 thread | 0.235 |
| `from_cells`, dense float32 input | 0.400 |
| `from_cells`, first call (compiles unless cached) | 3.747 |
| scanpy `normalize_total`, `log1p` and 3 `aggregate` calls (float32), warm | 0.322 |
| decoupler `pseudobulk` (sums only), warm | 0.831 |

The peak resident memory of that whole run was 1,751 MiB, in its largest process.

`linear_models.py --benchmark` on 20,000 genes x 200 samples (limma's time as R
measured it, CytePool's the median of three runs after a warm-up), with a peak
resident memory of 479 MiB in the largest process of the run:

| Scenario | limma `lmFit` (s) | CytePool (s) |
| --- | --- | --- |
| Ordinary least squares | 0.22 | 0.028 |
| Observation weights | 1.15 | 0.368 |
| 5% missing values | 1.00 | 0.308 |
| Block correlation | 0.15 | 0.110 |
| Block correlation and observation weights | 11.50 | 0.386 |

`adjust_p_values` with Hommel's method took 0.006 s on the 100,000 p-values of
`multiple_testing.py`. In `distributions.py`, mpmath took 395 s to compute the
references and CytePool 3.6 s to evaluate them, including compilation.

(reproduce-validation)=
## Reproducing this page

Run each script from the repository root, against the working tree, with `PYTHONPATH=.`
and a Python that has numpy, numba, scipy and mpmath; `RSCRIPT` names the `Rscript` of
the R environment, and the other variables name Python environments with the
reference packages. The runs on this page used:

```bash
REF=/path/to/reference
export RSCRIPT=$REF/r/bin/Rscript PYTHONPATH=.
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python validation/multiple_testing.py
python validation/smoothing.py
python validation/normalization.py
python validation/design.py
python validation/linear_models.py
python validation/linear_models.py --benchmark
python validation/empirical_bayes.py
python validation/voom.py
python validation/distributions.py
REFERENCE_PYTHON=$REF/python-scanpy/bin/python python validation/cube.py
SCANPY_PYTHON=$REF/python-scanpy/bin/python \
    DECOUPLER_PYTHON=$REF/python/bin/python python validation/pool_cells.py
REFERENCE_PYTHON=$REF/python/bin/python $REF/python3.14/bin/python \
    validation/cube_files.py
```

Here `python` is `$REF/python/bin/python`, the environment below. With
`--write-fixtures`, a script also rewrites the small inputs and reference outputs
under `tests/data/` that the test suite compares against.

The R environment holds R 4.4.3 with limma 3.62.1, edgeR 4.4.0 and DESeq2 1.46.0
(Bioconductor 3.20) and lme4 2.0.6 (CRAN). It was installed from conda-forge and
bioconda:

```bash
conda create -p $REF/r -c conda-forge -c bioconda r-base=4.4.3 \
    bioconductor-limma=3.62.1 bioconductor-edger=4.4.0 bioconductor-deseq2=1.46.0 \
    r-lme4=2.0_6
```

The Python environments use Python 3.13.12. scanpy has its own environment, and
`cube_files.py` ran CytePool on Python 3.14.4 for the zstd cases:

```bash
uv venv $REF/python --python 3.13.12
uv pip install --python $REF/python/bin/python numpy==2.5.4 numba==0.68.0 \
    scipy==1.18.1 mpmath==1.3.0 zarr==3.4.1 numcodecs==0.17.1 pandas==3.1.0 \
    anndata==0.13.4 decoupler==2.2.0
uv venv $REF/python-scanpy --python 3.13.12
uv pip install --python $REF/python-scanpy/bin/python scanpy==1.12.4 \
    numpy==2.5.4 numba==0.68.0 scipy==1.18.1 pandas==3.1.0 anndata==0.13.4
uv venv $REF/python3.14 --python 3.14.4
uv pip install --python $REF/python3.14/bin/python numpy==2.5.4 numba==0.68.0
```

The reference packages run only as separate programs that produce outputs to compare
against. Their source code is not read or copied: CytePool implements the methods from
their papers and documentation, and conventions that the documentation leaves open
were found by running the packages on chosen inputs.
