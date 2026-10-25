---
description: How closely each method in CytePool agrees with the tool that defined it, from the tables the validation scripts printed on 2026-10-24 and 2026-10-25.
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

This page summarises the tables the scripts printed when they were run against the
package in this revision of the repository: on **2026-10-25** for the six scripts on
cubes and their files (`cube.py`, `pool_cells.py`, `cube_files.py`, `cube_reads.py`,
`cube_fingerprints.py` and `anndata_conversion.py`), for `linear_models.py`, `voom.py`,
`sample_quality_weights.py`, `differential_expression.py`,
`differential_expression_options.py`, `interferon_stimulated_pbmcs.py`,
`gene_set_tests.py` and `composition_tests.py`, and for `design.py` and
`empirical_bayes.py`, which printed the same tables again after changes to the code
they check; and on **2026-10-24** for the others, with the
reference versions given in each section. Every script exited without a failure.
Each row below is a family of checks: it gives the number of printed checks it
covers and the largest difference among them. Run a script to see every check.

## Summary

| Script | Area | Reference | Checks | Outcomes | Run time | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| [`cube.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube.py) | Pseudobulk cubes | scanpy 1.12.4 | 20 | 19 pass, 1 information | 12.5 s | 926 MiB |
| [`pool_cells.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pool_cells.py) | Pooling cells into cubes | scanpy 1.12.4, decoupler 2.2.0 | 27 | 27 pass | 30.2 s | 1,725 MiB |
| [`cube_files.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_files.py) | Cube files | zarr-python 3.4.1 | 7 | 7 pass | 3.1 s | 146 MiB |
| [`cube_reads.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_reads.py) | Reads of selected features | zarr-python 3.4.1 | 5 | 5 pass | 1 min 46 s | 500 MiB |
| [`cube_fingerprints.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_fingerprints.py) | Cube fingerprints | the documented encoding, with zarr-python 3.4.1 | 7 | 7 pass | 4.4 s | 122 MiB |
| [`anndata_conversion.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/anndata_conversion.py) | AnnData and MuData | scanpy 1.12.4, decoupler 2.2.0, LIANA 1.10.0, anndata 0.13.4, mudata 0.4.1 | 18 | 18 pass | 1 min 5 s | 1,483 MiB |
| [`normalization.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/normalization.py) | Normalization and filtering | edgeR 4.4.0, DESeq2 1.46.0 | 50 | 46 pass, 3 deliberate differences, 1 control | 72.9 s | 745 MiB |
| [`design.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/design.py) | Designs and contrasts | R 4.4.3, limma 3.62.1, lme4 2.0.6 | 27 | 26 pass, 1 information | 4.3 s | 221 MiB |
| [`linear_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/linear_models.py) | Linear models | limma 3.62.1 | 218 | 200 pass, 18 information | 4.5 s | 177 MiB |
| [`voom.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/voom.py) | Precision weights for counts | limma 3.62.1, edgeR 4.4.0 | 118 | 111 pass, 7 deliberate differences | 15.3 s | 310 MiB |
| [`sample_quality_weights.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/sample_quality_weights.py) | Sample quality weights | limma 3.62.1, edgeR 4.4.0 | 89 | 83 pass, 6 deliberate differences | 12.1 s | 206 MiB |
| [`empirical_bayes.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/empirical_bayes.py) | Empirical Bayes moderation | limma 3.62.1 | 73 | 73 pass | 14.7 s | 306 MiB |
| [`gene_set_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/gene_set_tests.py) | Gene-set tests | limma 3.62.1 | 93 | 79 pass, 8 deliberate differences, 6 information | 6.9 s | 278 MiB |
| [`multiple_testing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/multiple_testing.py) | Multiple testing | R 4.4.3 | 15 | 15 pass | 2 min 56 s | 202 MiB |
| [`differential_expression.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression.py) | Differential expression | limma 3.62.1, edgeR 4.4.0 | 32 | 28 pass, 4 deliberate differences | 1.8 s | 189 MiB |
| [`differential_expression_options.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression_options.py) | Options for pseudobulk differential expression | limma 3.62.1, edgeR 4.4.0 | 66 | 66 pass | 5.0 s | 182 MiB |
| [`interferon_stimulated_pbmcs.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/interferon_stimulated_pbmcs.py) | Differential expression on published data | limma 3.62.1, edgeR 4.4.0 | 41 | 17 pass, 24 information | 15.2 s | 617 MiB |
| [`composition_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/composition_tests.py) | Cell-type composition | speckle 1.6.0, limma 3.62.1 | 32 | 25 pass, 7 deliberate differences | 13.1 s | 860 MiB |
| [`distributions.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/distributions.py) | Distributions and special functions | mpmath 1.3.0 | 46 | 46 pass | 6 min 39 s | 221 MiB |
| [`smoothing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/smoothing.py) | LOWESS and spline bases | R 4.4.3 | 90 | 87 pass, 1 information, 2 deliberate differences | 1.5 s | 175 MiB |
| **All** | | | **1,074** | **985 pass, none failed; 51 information, 37 deliberate differences, 1 control** | | |

Run time is the wall time of the whole script and peak memory the largest resident
set of any of its processes, reference programs included, both measured with GNU
`time` on the machine described under [](#validation-timings). For
`interferon_stimulated_pbmcs.py`, the run time is with its data downloaded already,
and the peak memory is the largest that the script measured itself: it restarts the
peak of its own process before it runs `limma_voom`, which hides the earlier, higher
peak from GNU `time`.

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

## Reads of selected features

[`validation/cube_reads.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_reads.py)
checks that reads of a selection take only what they need and return what zarr-python
reads. zarr-python writes an int32 array of 3 x 5 x 11 in chunks of 2 x 2 x 4 (gzip,
fill value 7, one chunk left unstored), and CytePool writes a cube file of 20 groups x
100 samples x 30,000 genes, with uint32 counts and float64 `log_sum` (gzip level 4,
seed 20261024). zarr-python's `Array.oindex` then reads 7 selections of each array and
6 selections of the cube: 50 neighbouring genes, 50 and 2,000 genes at random, 50
genes out of order in 2 groups, 10 samples of every gene, and one whole group.
Reference: zarr-python 3.4.1, numpy 2.5.4, Python 3.13.12.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| `ZarrArray.oindex`, 21 selections of the small array and of the two layers of the cube file: positions in any order, repeated and negative, masks, slices, integers, an empty selection, and the chunk that is not stored | zarr-python `Array.oindex` | 0 elements differ | 0 | 1 pass |
| Every layer of the 6 selections of the cube, read with `Cube.select` from the cube `read_cube` opened | zarr-python `Array.oindex` | 0 elements differ | 0 | 1 pass |
| Chunk files each selection opens | the chunks that hold a selected feature of a selected group | 0 selections open other files | 0 | 1 pass |
| Merged groups and pooled samples of the 50 genes at random: counts, and `log_sum` | sums of what zarr-python read | 0 and 4.3e-16 | 0 and 1e-12 | 2 pass |

Every chunk that holds a selected element is opened once, and no other: 40 chunk
files for 50 neighbouring genes of the 20 groups, where reading whole groups opens all
320. The script also measures each read; see [](#validation-timings).

## Cube fingerprints

No other tool computes the fingerprint of a cube, so
[`validation/cube_fingerprints.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_fingerprints.py)
checks it against a second implementation, written from the encoding that
`cytepool.cube.fingerprints` documents: a separate program reads cube files with
zarr-python and recomputes every digest and the fingerprint with numpy and `hashlib`,
without importing CytePool. Its encoder differs from CytePool's where it can: it finds
NaN with `numpy.isnan` instead of comparing bits, splits complex values with `.real`
and `.imag`, and decodes text columns from their codes and levels. The cubes (seed
20261025) have a table column of every type a cube's tables hold (text, bool, int8 to
int64, uint8 to uint64, float16 to float64, complex64, complex128, datetime and
timedelta), missing text, NaT, NaN, complex values with one NaN part, -0.0 in a table
and in `log_sum`, a uint64 column beyond the int64 range, a normalization holding
every kind of JSON value, uint16 and int16 layers, a group without cells, a bulk cube
without cell numbers or normalization, and a cube with empty axes. Reference:
zarr-python 3.4.1, numpy 2.5.4, Python 3.13.12.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Digests `write_cube` stores with 5 cube files (3,000 x 12 x 6 genes x samples x groups and smaller), 74 parts | recomputed from the documentation | 0 parts differ | 0 | 1 pass |
| `Cube.fingerprint()` of the 5 cubes in memory | recomputed from the documentation | 0 fingerprints differ | 0 | 1 pass |
| Copies zarr-python made with other chunks, big-endian bytes, gzip level 1 or no compression: recomputed, and checked by `verify_fingerprint()` | the original fingerprint | 0 fingerprints differ | 0 | 1 pass |
| The 3,000-gene cube with int64 layers instead of uint16 and int16 | the fingerprint of the narrow cube | 0 fingerprints differ | 0 | 1 pass |
| A count, a `log_sum` value moved by one unit in the last place, a sample id, a gene symbol, a cell number and the target sum, each changed in a copy | recomputed from the documentation | 6 of 6 change the fingerprint | all | 1 pass |
| -0.0 for 0.0, and NaN with another payload, in sample columns | recomputed from the documentation | 0 of 2 change it | none | 1 pass |
| CytePool on those 8 copies: `read_cube` refuses the changed tables, cell numbers and normalization, each with the message for that part; a changed layer group raises when it is read, whole or with its samples pooled, and in `verify_fingerprint()`, while the other groups read; and the 2 equal copies read and verify | the changes made | 8 of 8 as expected | all | 1 pass |

Before this run, the script was run against CytePool with five deliberate mistakes:
-0.0 left as it is, integer layers hashed in their own width, the lengths of text
written after the text, every float of the normalization hashed as a float64, and
reads of whole groups in another order (pooled samples) left unchecked. The first four
each made five checks fail, and the last one the check of the changed files.

## AnnData and MuData

[`validation/anndata_conversion.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/anndata_conversion.py)
simulates 18,914 cells x 1,500 genes (8 samples x 10 cell types, with an empty sample
x cell type entry, entries with one and two cells, and genes never detected; seed
20261025), with a condition for each sample and a lineage for each cell type, and
runs scanpy, decoupler and LIANA on them, each in its own environment. CytePool
imports what each tool made and compares the cube with the one `from_cells` builds
from the same cells; it then exports cubes and reads them back. Reference: scanpy
1.12.4 and decoupler 2.2.0, each with anndata 0.13.4 and pandas 3.1.0; LIANA 1.10.0
with decoupler 2.2.0, mudata 0.4.1, anndata 0.13.4 and pandas 2.3.3 (LIANA requires
pandas 2); Python 3.13.12. CytePool ran with numpy 2.5.4, numba 0.68.0, anndata
0.13.4, mudata 0.4.1 and pandas 3.1.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| `from_anndata` of `scanpy.get.aggregate` (`sum` and `count_nonzero` of the counts): counts, detected cells, cell numbers, and the fingerprint | `from_cells` | 0; identical fingerprint | 0 | 2 pass |
| The same with scanpy's sums of log-normalized values, of their squares and of normalized values | `from_cells` | 6.4e-16 | 1e-10 | 1 pass |
| `from_anndata` of `decoupler.pp.pseudobulk`: counts, detected cells from its fractions times its cell numbers, cell numbers, and the condition and lineage read from its `obs`; and after `decoupler.pp.filter_by_expr` kept 1,355 of the 1,500 genes | `from_cells`, the same genes, and the simulated metadata | 0; identical tables and fingerprint | 0 | 3 pass |
| `from_mudata` of LIANA's `adata_to_views`, with the cells' crosstab as cell numbers: counts, detected cells, cell numbers, and the fingerprint | `from_cells` | 0; identical fingerprint | 0 | 2 pass |
| scanpy's means and variances; decoupler with `mode="mean"` and `mode="median"`; LIANA's views with `mode="median"`, and without cell numbers | refused, each with its reason | 5 of 5 refused | all | 5 pass |
| `to_anndata` against scanpy's rows and sums, `to_mudata` against LIANA's modalities, and scanpy aggregating the export by condition against `Cube.collapse_samples` | scanpy, LIANA | 0; identical | 0 | 3 pass |
| Round trips of 4 cubes (every layer; merged, pooled and selected; bulk; tables of every column type, with columns without any value, names anndata does not write as they are, and a sample and a group without cells) through AnnData and MuData, in memory and through `.h5ad`, `.zarr` and `.h5mu` files | the exported cube | 24 of 24 fingerprints kept | all | 1 pass |
| The same files, read and written again by anndata and mudata under pandas 2 | the exported cube | 12 of 12 fingerprints kept | all | 1 pass |

A cube's layers are sums, so an import takes only what is a sum or gives one exactly.
Integer layers match exactly: scanpy and decoupler store sums of whole numbers as
float64, which CytePool turns back into the integers that `from_cells` gives, and
decoupler's fractions of detecting cells times its cell numbers are whole numbers to
within the precision of the fractions, so rounding recovers the counts exactly.
Means, variances and medians are refused; every tool that gives them gives sums too.
The refusals do not depend on the tool saying what it computed, which neither
decoupler nor LIANA records: decoupler computes `psbulk_counts` from the summed
counts whatever its mode (found by probing), so means add up to it divided by the
cells of their row, and a sum is positive exactly where a cell detects the feature,
which medians are not. `X` may add up to less than `psbulk_counts`, as after
`filter_by_expr`. LIANA 1.10 keeps cell numbers only under the column name of
decoupler 1, so its views carry none, and an entry without cells cannot be told from
a missing profile: the import refuses them until cell numbers are given. decoupler
gives its row without cells the metadata of another row (found by probing); CytePool
reads a column as sample or cell-type metadata from the rows with cells.

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
20,000 genes and 4 to 40 samples, in 26 cases: two groups, four groups with a
covariate, group means, a covariate without an intercept, one residual degree of
freedom, normalization factors (edgeR's `calcNormFactors` in R, `tmm_factors` here)
with and without given library sizes, two very small libraries, libraries from 1e3 to
1e8 reads, an intercept-only design, genes with the same counts and blocks of genes
with a single read (ties in the trend), rarefied counts, non-integer counts, spans of
0.3 and 1, a design without replication and two genes. These 19 cases fix the span
(0.5 unless named), as limma's `adaptive.span = FALSE` does. In the other 7, both
programs choose it from the number of genes, as `adaptive.span = TRUE` does (limma's
default since 3.66, and CytePool's): 2, 30, 51 and 120 genes, about a fifth of the
last three without counts, the group means and four-group cases (10,000 and 20,000
genes), and the design without replication. Designs come from the same sample table through R's
`model.matrix` and `Design`, contrasts through `makeContrasts` and named contrasts.
The span rule, `choose_lowess_span`, is compared with limma's `chooseLowessSpan` at
466 settings: limma's own (`small.n = 50`, `min.span = 0.3`, `power = 1/3`) for 0 to
1e9 points, and 216 others with `small.n` of 0, 1 and 10, `min.span` from 0.05 to 1
and `power` from 0 to 1. Reference: R 4.4.3, limma 3.62.1, edgeR 4.4.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| log2 CPM `E`, and library sizes times normalization factors | `voom` | 0 | 1e-10 | 26 pass |
| Span chosen from the number of genes, which limma returns with `adaptive.span = TRUE` | `voom` | 0 | 1e-10 | 7 pass |
| Weights | `voom` | 5.2e-13 | 1e-10 | 25 pass |
| Trend: the points and LOWESS line saved with `save.plot = TRUE` | `voom` | 3.7e-14 | 1e-10 | 23 pass |
| Moderated t and p after `lmFit` with the weights, `contrasts.fit` and `eBayes` | limma | 5.0e-13 | 1e-10 | 23 pass |
| Rarefied counts: weights, trend, moderated t and p | limma | 2.2e-8, 8.5e-8, 8.9e-9 | 1e-6 | 3 pass |
| Designs without replication: the warning, their weights being compared above | `voom` | both warn | n/a | 2 pass |
| Span rule: limma's settings for 0 to 1e9 points, and the other settings | `chooseLowessSpan` | 0 | 1e-10 | 2 pass |
| Deliberate differences, listed below | `voom`, `chooseLowessSpan` | 4 where limma returns values and CytePool raises, 1 where limma fails and CytePool returns values; 2 for the span rule | n/a | 7 deliberate differences |

Differences are relative, except for E, the trend's x and t, which are scaled by
max(\|reference\|, 1); the trend's y, scaled by max(\|reference\|, median of the
reference), because near zero both are rounding noise; and p-values, scaled by
max(p, 1e-300) max(1, \|log p\|). E was identical to limma's to the last bit in 26 of
26 cases, and so were the trend's x in 24 of 24, the chosen span in the 6 cases where
limma returns one (without replication it returns none, and CytePool's is NaN), and
the span rule at all 466 settings. limma's `voom` chooses the span from every gene,
those without counts included (`nrow(counts)` on its help page): in the case of 51
genes, of which 38 enter the trend, both choose 0.9954, where 38 genes would give 1.
Rarefied counts have equal library sizes, so a gene with the same count in every
sample has a residual standard deviation that is pure rounding noise in either
program; those points of the trend, and through them the weights, differ in the close
tier. Deliberate differences:

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
- The span rule for 0 points with a `min.span` of 1 or a `small.n` of 0: limma's
  formula takes 0 times infinity or 0 / 0 and returns NaN; `choose_lowess_span`
  returns 1, the span limma's help page gives for up to `small.n` points.
- The span rule with arguments outside the ranges `choose_lowess_span` accepts (n of
  -1, `min.span` of 0 or 1.5, `power` of -1 or 2): limma computes its formula, capped
  at 1, which gives NaN for a negative n, and spans below 1 for fewer than `small.n`
  points for a `min.span` above 1 or a negative `power`; CytePool raises. limma
  documents a `power` between 0 and 1, and `min.span` is the span for many points, a
  fraction in (0, 1].

## Sample quality weights

[`validation/sample_quality_weights.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/sample_quality_weights.py)
compares `sample_quality_weights` with limma's `arrayWeights`, `voom` with prior
weights with limma's `voom(weights = ...)`, and `voom_with_quality_weights` with
`voomWithQualityWeights`, on data simulated from fixed seeds, in 37 cases. The sample
weights are estimated from normal log-expression values with one or two samples two
to four times noisier, and from voom's log2 CPM with its weights: 4 to 12 samples and
2,000 or 20,000 genes; two groups, an intercept, and three groups with a covariate;
REML and the gene-by-gene algorithm; a weight per sample, per group (`var.group`) and
from a covariate (`var.design`); `prior.n` of 0, 10 and 50; observation weights;
missing values, with genes left with one residual degree of freedom or none of
residual variance; data on which limma's REML stops early; one gene, and no
replication. voom gets sample, observation and
feature weights, observation weights with zeros, and a span chosen from the number of
genes. voom with quality weights runs both algorithms, with the span fixed and chosen,
weights per group and from log library size, TMM factors with given library sizes and
contrasts, two noisy pseudobulk samples among 20,000 genes, and no replication.
Designs come from the same sample table through R's `model.matrix` and `Design`,
contrasts through `makeContrasts` and named contrasts. Reference: R 4.4.3, limma
3.62.1, edgeR 4.4.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Sample weights | `arrayWeights` | 8.3e-15 | 1e-6 | 24 pass |
| Reports: the genes with missing values that REML leaves out (the same number), and an early stop of REML | `arrayWeights` | both report | n/a | 2 pass |
| voom with prior weights: E | `voom` | 0 | 1e-10 | 5 pass |
| voom with prior weights: the weights times the prior weights, the trend saved with `save.plot = TRUE`, and the chosen span | `voom` | 1.2e-13 | 1e-10 | 11 pass |
| voom with prior weights: moderated t and p after `lmFit` with the weights and `eBayes` | limma | 3.2e-13 | 1e-10 | 5 pass |
| voom with quality weights: E and the chosen span | `voomWithQualityWeights` | 0 | 1e-10 | 11 pass |
| voom with quality weights: sample weights | `voomWithQualityWeights` | 7.2e-15 | 1e-6 | 8 pass |
| voom with quality weights: weights | `voomWithQualityWeights` | 9.8e-14 | 1e-6 | 8 pass |
| voom with quality weights: moderated t and p after `lmFit`, `contrasts.fit` and `eBayes` | limma | 1.1e-13 | 1e-6 | 7 pass |
| No replication: both report it | `voomWithQualityWeights` | both | n/a | 1 pass |
| A gene whose prior weights are all zero: the weights of the other genes against limma's without that gene | `voom` | 3.0e-14 | 1e-10 | 1 pass |
| Deliberate differences, listed below | `arrayWeights`, `voom` | 3 where limma returns values and CytePool raises, 1 where limma fails and CytePool returns values, 2 where both return values that differ | n/a | 6 deliberate differences |

Differences are relative, except for E, the trend's x and t, scaled by
max(\|reference\|, 1); the trend's y, scaled by max(\|reference\|, its median); and
p-values, scaled by max(p, 1e-300) max(1, \|log p\|). The sample weights are iterative
estimates, in the close tier, as is everything computed from them. They agree to
rounding because CytePool repeats limma's iterations and stopping rule, found by
running limma as a black box: the starting point, the prior's terms in the score and
the information, the exact REML information of Smyth (2002), and the criterion that
stops the iterations, including the early stop where it rises. The weights that limma's `voom` returns with prior weights are
the trend's alone; CytePool's are those times the prior weights, which the comparison
multiplies in. `eBayes` runs with `legacy = TRUE`, CytePool's estimator of the prior:
zero weights make the residual degrees of freedom unequal, where limma's default
estimator differs. Deliberate differences:

- `var.design` and `var.group` together: limma uses `var.group`; CytePool refuses the
  two.
- A negative `prior.n`: limma accepts it; CytePool requires at least zero.
- REML when one gene has every value observed: limma fails; CytePool gives weights of
  one, with a warning, as limma does with one gene.
- REML with `prior.n = 0` and a sample that carries no information about its
  variance (the only sample of a group): limma returns weights that rounding error
  decides; CytePool raises, as the information matrix is singular.
- A gene whose prior weights are all zero: limma passes its missing residual standard
  deviation to R's `lowess`, which returns a curve without meaning; CytePool leaves the
  gene out of the trend, and the other genes get the weights that limma gives them
  without the gene (the row above).
- Zero prior weights that leave a gene's coefficients inestimable: limma returns
  missing weights for the gene; CytePool's are those of the trend at its fitted values
  over the samples of positive weight, and zero where the prior weight is.

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

## Gene-set tests

[`validation/gene_set_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/gene_set_tests.py)
compares `camera` and `fry` with limma's `camera` and `fry` on log-expression of 1,500
genes simulated from fixed seeds, with gene-specific variances from a scaled inverse
chi-squared prior around a mean-variance trend. The gene sets change up, down or in
both directions, or not at all, share random sample effects (inter-gene correlation)
with or without a change, overlap others, hold one, two or five genes, or 40% of the
genes. The cases are three groups with the contrast given as a design column, also
with only sets of one or more than d genes (so that fry's adjusted mixed p-values can
be compared); three groups in interleaved order with observation weights and the
contrast given as weights (a named contrast); counts with voom weights from `voom`
(limma gets the same weights matrix), with the contrast both ways; a paired design with
sample weights; residual variances that are all equal, so that the prior df is
infinite; 16 samples with sets of 2 to 30 unchanged genes and three sets with changed
genes; a gene the design fits exactly and ten identical genes; and 8 genes, fewer than
the residual df. Designs are `Design` objects with named contrasts and genes are named
by id in CytePool; R gets the design matrices and the genes' positions. camera runs with
the default inter-gene correlation of 0.01, with it estimated for each set, with
negative correlations allowed, with ranks, with a trend and robustly; fry with each
standardization, and with a trend and robustly. limma's camera has no robust option: its
reference is `cameraPR` or `rankSumTestWithCorrelation` on the z-scores of the
moderated t-statistics of `eBayes(robust = TRUE)`, limma's `zscoreT(approx = TRUE,
method = "hill")`, with the design reparametrised by `contrastAsCoef`; with an estimated
correlation, the parametric p-value of `cameraPR` is converted to the df of the
estimate. Reference: R 4.4.3, limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| camera: numbers of genes, correlations, directions, p-values and FDR, with a given or estimated correlation, ranks and a trend | `camera` | 1.5e-14 | 1e-10 | 35 pass |
| camera with robust moderation | `cameraPR`, `rankSumTestWithCorrelation` | 1.7e-12 | 1e-6 | 12 pass |
| fry with `standardize` `"none"` and `"residual_sd"`: numbers of genes, directions, p-values and FDR, mixed p-values of sets of one or more than d genes, and their FDR | `fry` | 5.2e-13 | 1e-10 | 15 pass |
| fry with `"posterior_sd"`, with and without trend or robust moderation | `fry` | 4.4e-5 | 1e-3 | 15 pass |
| fry with `"posterior_sd"` and an infinite prior df, where both standardize as `"none"` | `fry` | 3.3e-14 | 1e-10 | 1 pass |
| CytePool's mixed p-values of the sets of 2 to 14 genes against 200,000 random rotations of their effects | simulation | 0.025 (absolute) | 0.1 | 1 pass |
| fry's mixed p-values of sets of 2 to d genes | `fry` | 0.88 (absolute) | n/a | 8 deliberate differences |
| The rotation simulation: limma's mixed p-values of the sets of 2 to 14 genes; both programs' of the sets of 15 to 30 genes; both programs' of the three sets with changed genes (relative) | simulation | 0.59, 0.030, 0.99 | n/a | 3 information |
| Samples sorted by group, observation weights and the contrast as weights: camera, fry with `"residual_sd"`, and the genes whose last pivot R's `qr()` signs by rounding noise | `camera`, `fry`, `qr` | 0.012, 0.48; 115 of 1,500 genes | n/a | 3 information |

Correlations are compared as differences, p-values relative to max(p, 1e-300) max(1,
\|log p\|) as in `empirical_bayes.py` (limma's p-values above 1 count as 1), and a
direction or a number of genes that differs counts as an infinite difference. Robust
moderation is in the close tier because limma solves the robust prior df only to a
tolerance. With `"posterior_sd"`, fry moderates each gene's mean square of its d + 1
effects without the largest, divided by its expected value under the null hypothesis,
as a variance on 0.92 d degrees of freedom, as limma does (found by probing); limma's
expected value is about 7.3e-5 (d + 1) / d above the exact one that CytePool computes,
so directional p-values differ by up to 4.9e-5 and mixed p-values by up to 5.7e-4 of
their value, which the tolerance of 1e-3 allows. The rotation check uses sets without
a change, whose p-values a simulation resolves: the beta approximation that both
programs use for the mixed p-value is off by up to 0.03 at such p-values for sets of
more than d genes, where the programs agree, and in the tail it can be much too small
for both: sets with changed genes and rotation p-values of 3.2e-3 and about 6e-5 get
6.3e-4 and 2.9e-7.

In each gene's QR decomposition, CytePool treats a pivot that is zero up to rounding
as zero, which R's `qr()` signs as positive. With the samples sorted by group and a
contrast between groups given as weights, every gene's last pivot is such a zero.
limma computes most of them exactly, but its arithmetic leaves rounding noise in a few
percent of genes, which changes their residual bases, and camera and fry do so for
different genes: fry's are the 115 of the 1,500 whose last pivot R's `qr()` signs by
noise. camera's correlations of sets then differ by up to 0.012, and fry's p-values
with `"residual_sd"` by up to a factor of 2. limma's results there change with the
last bits of the weights, CytePool's do not; in every other case the two agree.

Deliberate differences:

- fry's mixed p-value of a set of 2 to d genes. Its statistic, the sum of the set's
  squared standardized contrast effects, is r'(E'E)r under a random rotation r of the
  d + 1 effects, which both programs approximate by a beta distribution with its first
  two moments. CytePool uses all d + 1 dimensions, as for larger sets; limma uses only
  the set's m nonzero eigenvalues of E'E as if r had m dimensions (found by probing).
  Against 200,000 random rotations, CytePool's mixed p-values of the sets of 2 to 14
  genes are within 0.025 and limma's off by up to 0.59.
- A p-value above 1 from camera's rank test (its continuity correction can put both
  tails above 1/2): limma reports twice the smaller tail, CytePool 1. A gene listed
  twice in a set: limma counts it twice, CytePool once. A set without genes, or holding
  every gene in camera: NaN p-values in CytePool, left out of the other sets'
  adjustment; limma reports the direction "Up", in camera's rank test the p-value 2,
  which it adjusts with the others, and in fry a mixed p-value of 0 for an empty set.
  The tests check these.

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

## Differential expression

[`validation/differential_expression.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression.py)
simulates cells with a donor structure (seed 20261025), pools them into cubes with
`from_cells`, and runs `limma_voom` and `limma_trend` on every cell type; R runs the
workflow of the limma and edgeR User's Guides on the same pseudobulk counts, group by
group: it leaves out the samples with fewer than 10 cells, keeps the rows of the
remaining samples of a design built by `model.matrix` on the full sample table, and
runs `filterByExpr`, `calcNormFactors`, `voom` (or `cpm(log = TRUE, prior.count = 3)`
for limma-trend), `lmFit`, `contrasts.fit` one contrast at a time, `eBayes` and
`topTable(sort.by = "none")`. voom chooses its span from the number of genes tested
in each group, in both programs (`adaptive.span = TRUE` in R, limma's default since
3.66), except in the case that fixes it at 0.5 (`span = 0.5, adaptive.span = FALSE`);
limma's voom reports the span it chose (found by probing), and each group's span is
compared. The cases are a paired design, `~ donor + condition` (6 donors x 2
conditions, 4 cell types, 1,000 genes), by limma-voom and limma-trend, and an
unpaired design with a covariate, `~ condition + age` (10 donors, 3 cell types), by
limma-voom with robust moderation and a span of 0.5, and by limma-trend without
filtering or normalization factors. The paired design tests the stimulation and
three donor comparisons: `D2_vs_D1`, `D2_vs_D6` and `D3_vs_D2`. Some sample x cell
type entries have fewer than 10 cells; in the NK cells both samples of donor D1, the
reference level, are left out, which makes the column `donorD6` a combination of the
others, and in the DC cells four samples remain, too few for any residual degrees of
freedom.
Reference: R 4.4.3, limma 3.62.1, edgeR 4.4.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Samples used, features tested, coefficients that cannot be estimated, contrasts tested (other than the deliberate differences), voom's span, residual and prior df, in the 12 fitted groups | `filterByExpr`, `voom`, `lmFit`, `contrasts.fit`, `eBayes` | 0 mismatches | 0 | 12 pass |
| logFC, AveExpr, t, P.Value, adj.P.Val and B of every contrast both programs compute | `topTable` | 6.3e-13 | 1e-10 | 9 pass |
| The same with robust moderation | `topTable` | 1.0e-13 | 1e-6 | 3 pass |
| `D2_vs_D6` in the NK cells, which limma refuses, against limma's values for `D2_vs_D1`, which there estimate the same difference | `topTable` | 2.8e-13 | 1e-10 | 2 pass |
| DC cells, without residual degrees of freedom: both programs refuse every contrast | `eBayes` | both refuse | n/a | 2 pass |
| Deliberate differences, listed below | `contrasts.fit` | 2 contrasts limma refuses and CytePool computes, 2 limma computes and CytePool reports as not estimable | n/a | 4 deliberate differences |

Differences are scaled by max(\|reference\|, 1), and p-values by
max(p, 1e-300) max(1, \|log p\|). limma-voom ran with
`contrast_standard_errors="limma"`, except in the robust case, whose contrasts are
single coefficients, for which the default exact standard errors are limma's. In
the NK cells, `D3_vs_D2` combines two of the columns the group keeps, and agrees with
limma's approximate standard errors too. The robust case keeps the close tier because
limma solves the robust prior only to a tolerance; here it agreed to 1e-13.
Deliberate differences, in the NK cells of both paired cases:

- `D2_vs_D6` (`donorD2 - donorD6`) is estimable without donor D1, but it weighs the
  aliased column `donorD6`, and limma's `contrasts.fit` stops with "trying to take
  contrast of non-estimable coefficient". CytePool computes it.
- `D2_vs_D1` (`donorD2`, donor D2 against D1) cannot be estimated without donor D1's
  samples. limma computes it from the kept columns, where `donorD2` means donor D2
  against donor D6, the values of the third row of the table; CytePool reports it as
  not estimable.

## Options for pseudobulk differential expression

[`validation/differential_expression_options.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression_options.py)
checks the options of `limma_voom` and `limma_trend` for pseudobulk data against
limma and edgeR. It simulates cells from 6 donors, each sampled before and after a
stimulation (seed 20261027), in 3 cell types with 1,000 genes, pools them with
`from_cells`, and fits `~ donor + condition` with two contrasts, `stim_vs_ctrl` and
`D3_vs_D2`. One sample is an outlier in every cell type, two are sequenced shallowly
in one cell type each, and in the NK cells both samples of donor D1, the reference
level, have fewer than 10 cells, which makes the column `donorD6` a combination of the
others. Six analyses each set some of the options, and R runs the workflow of the
section above with the same options:

- sample quality weights (`quality_weights`): limma's `voomWithQualityWeights`, gene by
  gene and by REML, for limma-voom; `arrayWeights` on the log2 CPM, by REML and gene
  by gene, then `lmFit` with them as sample weights, for limma-trend;
- prior weights from cell numbers (`prior_weights="n_cells"`): R counts the cells of
  each sample and cell type from the cells' labels, not from the cube, and uses them
  as prior weights: `voom(weights = n)`, whose weights times n enter `lmFit`, as
  voom's help page says its weights modify prior weights; independently, edgeR's
  `voomLmFit(prior.weights = n)`, which forms that product itself (found by
  probing); and `lmFit(weights = n)` for limma-trend;
- a minimum total count per sample (`min_counts=30000`, in three analyses): R leaves
  out the samples whose column sum is below it, here the two shallow samples, besides
  those with fewer than 10 cells;
- p-values adjusted across cell types (`adjust_across_groups`, in four analyses):
  `p.adjust(method = "BH")` of the `P.Value` of the three cell types stacked, one
  contrast at a time.

Reference: R 4.4.3, limma 3.62.1, edgeR 4.4.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Samples used, features tested, voom's span, residual and prior df, in the 18 fits (6 analyses x 3 cell types) | `filterByExpr`, `voom`, `lmFit`, `eBayes` | 0 mismatches | 0 | 18 pass |
| Sample quality weights | `voomWithQualityWeights`, `arrayWeights` | 2.7e-11 | 1e-6 | 12 pass |
| Prior weights against the number of cells R counts from the cells | `table` | 0 | 1e-10 | 6 pass |
| logFC, AveExpr, t, P.Value, adj.P.Val and B of both contrasts, with quality weights | `topTable` | 1.1e-12 | 1e-6 | 12 pass |
| The same with prior weights | `topTable` | 1.7e-12 | 1e-10 | 6 pass |
| The same, limma-voom with prior weights, against edgeR | `voomLmFit` | 1.8e-13 | 1e-10 | 3 pass |
| P-values adjusted across the cell types, with quality weights | `p.adjust` | 5.1e-13 | 1e-6 | 4 pass |
| The same with prior weights | `p.adjust` | 7.6e-13 | 1e-10 | 4 pass |
| Prior weights together with estimated sample weights: both programs refuse | `voomLmFit` | both refuse | n/a | 1 pass |

Differences are relative for the weights; the statistics are scaled as in the section
above. Everything computed from sample quality weights is in the close tier, as they
are iterative estimates; the rest is exact. limma-voom ran with
`contrast_standard_errors="limma"`, since `D3_vs_D2` combines two coefficients. limma's
quality weights on the rank-deficient design of the NK cells are those on its kept
columns (found by probing), as CytePool's are. `voomLmFit` treats some genes with zero
counts differently from `voom` and `lmFit`: it lowers their residual df, as its help
page says, and can move their average log count in the trend (found by probing). In
these data that happens only where `min_counts` leaves a donor with one sample, so
`voomLmFit` is compared in the analysis of limma-voom with prior weights that does not
set `min_counts`. limma's `voomWithQualityWeights` takes no prior weights, and edgeR's
`voomLmFit` stops with "Can't specify prior.weights and estimate sample weights";
CytePool refuses the two together as well.

## Differential expression on published data

[`validation/interferon_stimulated_pbmcs.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/interferon_stimulated_pbmcs.py)
runs the whole path from cells to differential-expression tables on published
single-cell data, and compares it with limma on the same pseudobulk counts. The data
are batch 2 of GEO series
[GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) (Kang et al.
2018, *Nature Biotechnology* 36, 89-94): peripheral blood mononuclear cells (PBMCs) of
8 lupus patients, each split into an untreated culture and one stimulated with
interferon-beta for 6 hours, sequenced with 10x Genomics, the cells assigned to their
donors by genotype. The script downloads the three files it needs (77 MB) into a cache
directory outside the repository and checks their SHA-256 checksums. It reads them as
data, without extracting the archive, and stops if a fact it relies on fails: the
dimensions of the two count matrices, the gene identifiers, the number of cells of
each type, and the rule that joins each cell to the published cell table by its
condition and barcode (every cell matches one row and every row one cell; nothing is
joined by position).

`from_cells` pools the 24,673 singlets that have a cell type into a cube of 16 samples
(8 donors, control and stimulated) x 8 cell types x 35,635 genes, keyed by Ensembl id
with symbols, and its counts are checked against sums of the cells that scipy computes
independently. `limma_voom` then tests stimulated against control in every cell type,
with the paired design `~ donor + condition` and its defaults: samples with fewer than
10 cells (`min_cells`) are left out, then come `filterByExpr`, TMM and voom, whose
LOWESS span is chosen from the number of genes tested by limma's `chooseLowessSpan`
rule, as limma's voom does by default since limma 3.66. R runs the limma User's Guide
workflow on the counts, samples and cell numbers exported from the cube: the same
`min_cells` rule, then `filterByExpr`, `calcNormFactors`, `voom`, `lmFit`,
`contrasts.fit`, `eBayes` and `topTable(sort.by = "none")`, all with R's defaults
except two settings that differ between limma versions: voom gets the span that the
cell type's record in the CytePool result gives, with `adaptive.span = FALSE`, and
`eBayes` gets `legacy = TRUE`. R also computes `chooseLowessSpan` of the number of
genes tested, the span its voom would choose itself, and that span must be the same.
Reference: R 4.4.3, limma 3.62.1, edgeR 4.4.0; CytePool ran with Python 3.13.12,
numpy 2.5.4 and numba 0.68.0, and scipy 1.18.1 read the count matrices.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Counts of the 16 samples x 8 cell types x 35,635 genes of the cube | sums of the cells, by scipy | 0 counts differ | 0 | 1 pass |
| Samples used, genes tested, aliased columns, residual and prior df, and voom's span, in the 8 cell types | `filterByExpr`, `lmFit`, `eBayes`, `chooseLowessSpan` | 0 mismatches | 0 | 8 pass |
| logFC, AveExpr, t, P.Value, adj.P.Val and B of every gene tested in the 8 cell types | `topTable` | 6.6e-13 | 1e-10 | 8 pass |
| Per cell type: genes tested and called, the interferon response, the negative control (tables below) | MSigDB, none | n/a | n/a | 24 information |

Differences are scaled as in the section above. The dendritic cells are fitted without
2 of the 16 samples, and the megakaryocytes without 4, among them both samples of donor
107: that donor's column of the design is then zero, and both programs leave it out of
the fit.

The recovery of a pre-specified list of interferon-stimulated genes is measured with
MSigDB's Hallmark
[interferon-alpha response](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE.html)
gene set (97 genes, MSigDB v2026.2.Hs; Liberzon et al. 2015, *Cell Systems* 1,
417-425; CC BY 4.0), matched to the data by symbol; two of its genes have earlier
symbols in the data, FAM46A for TENT5A and WARS for WARS1. Up and down are calls at a
5% false discovery rate:

| Cell type | Cells | Samples used (left out) | Genes tested | voom span | Up | Down | Hallmark genes tested | Up | Down | Median log2 fold change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B cells | 2,573 | 16 (0) | 2,768 | 0.484 | 498 | 399 | 76 | 73 | 2 | 2.73 |
| CD14+ monocytes | 5,385 | 16 (0) | 5,125 | 0.450 | 1,619 | 1,549 | 92 | 89 | 3 | 3.26 |
| CD4 T cells | 10,389 | 16 (0) | 6,295 | 0.440 | 991 | 826 | 86 | 82 | 1 | 2.41 |
| CD8 T cells | 2,042 | 16 (0) | 1,453 | 0.528 | 233 | 147 | 55 | 54 | 0 | 2.51 |
| Dendritic cells | 432 | 14 (2) | 2,281 | 0.496 | 454 | 285 | 76 | 68 | 0 | 3.27 |
| FCGR3A+ monocytes | 1,599 | 16 (0) | 2,947 | 0.480 | 814 | 663 | 84 | 82 | 1 | 2.42 |
| Megakaryocytes | 260 | 12 (4) | 704 | 0.590 | 63 | 16 | 48 | 35 | 0 | 2.48 |
| NK cells | 1,993 | 16 (0) | 2,292 | 0.496 | 374 | 300 | 68 | 66 | 0 | 2.13 |

In the six cell types with all 16 samples, 95% to 98% of the Hallmark genes tested are
called up, with median log2 fold changes of 2.1 to 3.3, that is 4 to 10 times the
expression of the controls; in the dendritic cells and megakaryocytes, with fewer cells
and samples, 89% and 73% are. The only Hallmark genes called down are IL4R, in four
cell types, CD74 in the B cells, and PROCR and IFI30 in the CD14+ monocytes.

The response to interferon is so large that recovering it says little about error
control; a negative control, which needs no reference, does. In each cell type, the
control samples that its fit used are split into two halves in every possible way,
each split once, and every split is fitted with `~ split`, so that a call compares
donors, not conditions: 35 splits of 4 against 4 samples, or 10 splits of 3 against 3
in the dendritic cells and megakaryocytes, where 2 control samples have fewer than 10
cells. If the donors did not differ, the Benjamini-Hochberg procedure at a 5% false
discovery rate would make any call in an analysis with a probability of at most about
5%, so in about 12 of these 230 analyses or fewer.

| Cell type | Splits | Splits with calls | Calls | Most in one split | Genes called in more than one split |
| --- | --- | --- | --- | --- | --- |
| B cells | 35 | 4 | 4 | 1 | none |
| CD14+ monocytes | 35 | 1 | 6 | 6 | none |
| CD4 T cells | 35 | 7 | 123 | 44 | OASL, LMNA, DDIT4, ANXA2R (3 splits); YIPF5, FBLN5, RGS1, TPRG1, CD27, C16orf54, SCML4, NUCB2, SUN2, PASK, FASLG, FAM46C (2) |
| CD8 T cells | 35 | 5 | 13 | 6 | TMSB10 (2) |
| Dendritic cells | 10 | 2 | 2 | 1 | none |
| FCGR3A+ monocytes | 35 | 2 | 5 | 3 | none |
| Megakaryocytes | 10 | 0 | 0 | 0 | none |
| NK cells | 35 | 2 | 3 | 2 | none |

23 of the 230 analyses made a call, 156 calls in all, and most of them follow
differences between the donors. Of the five splits that put donors 1244, 1256 and 1488
in one group with a fourth donor, the three in which that donor is 1016, 1039 or 107
make 117 calls in the CD4 T cells, among them genes of regulatory and memory T cells
(TIGIT, TNFRSF4, TNFRSF18, FASLG, LMNA, RGS1), which suggests that the CD4 T cells of
these donors differ in composition. The split of donors 101, 107, 1244 and 1488 against
the others calls interferon-stimulated genes (ISG15, IFI6, IFIT1, IFIT3, LY6E, IFITM3)
in the monocytes and the CD4 T cells of the untreated cultures, consistent with lupus
patients differing in interferon activity. Apart from those three CD4 T splits, no
analysis makes more than 6 calls. The donors appear to be all female, so sex does not
divide them: XIST is counted in all 16 samples, and none of the Y-chromosome genes
RPS4Y1, DDX3Y, KDM5D, UTY and EIF1AY in any.

The script times each part on the machine described under [](#validation-timings),
with numba, OpenMP, OpenBLAS and MKL limited to 2 threads, the data downloaded already
and numba's cache of compiled routines filled:

| Part | Seconds |
| --- | --- |
| Reading and checking the files, and keeping the 24,673 singlets with a cell type | 1.30 |
| `from_cells`: 24,673 cells x 35,635 genes (14.2 million stored counts) into the cube | 0.24 |
| `limma_voom` on the 8 cell types, first call in the process | 0.32 |
| `limma_voom` on the 8 cell types, second call | 0.26 |
| R: the limma workflow on the 8 cell types, from the counts in memory | 1.42 |
| The negative control: `limma_voom` on the 230 splits, one cell type each | 3.61 |

With an empty numba cache, as after installing or changing the package, the first call
of `limma_voom` compiles its routines too, and took 17 s in a run with
`NUMBA_CACHE_DIR` set to an empty directory; later processes load them from the cache.
The Python process held at most 617 MiB while it read the files and pooled the
cells. Once the cell-level matrices were released it held 232 MiB, and while
`limma_voom` ran at most 252 MiB, so the analysis of the 8 cell types added
20 MiB to a process that held the cube; from then to the end, with the negative
control and the export to R, it held at most 321 MiB. R peaked at 227 MiB.

## Cell-type composition

[`validation/composition_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/composition_tests.py)
compares `propeller` with speckle's `propeller`, `propeller.ttest` and
`propeller.anova`. It simulates the number of cells of every sample in every cell type
from a Dirichlet-multinomial model (seed 20261025), holds them in cubes, and gives
speckle one label per cell; `propeller.ttest` and `propeller.anova` get
`getTransformedProps` and design matrices from `model.matrix`. Every case runs with
the logit and the arcsine square-root transform, and with speckle's defaults (robust
moderation, no trend) unless a row says otherwise. The cases are two groups of 5
samples with 1,000 to 5,000 cells and 12 cell types, three of which change, also with
`robust = FALSE` and `trend = TRUE`; three groups of 4 samples, by speckle's F-test;
three groups of 5 samples and a covariate (age), `~ 0 + group + age`, with the
contrasts B - A and (A + B)/2 - C, the F-test of `group` (`propeller.anova` with
`coef = 1:3`), the slope of age and A + B, and `~ group + age` with the contrast
`groupB`; a pairing of 6 donors in two conditions, `~ 0 + condition + donor` and
`~ donor + condition`; 8 samples with 6 to 32 cells, in which most cell types are
absent from some samples and two from all of them, in a cube that also holds a cell
type and a sample without cells; two cell types, which limma does not moderate; and
samples that are copies of one another within each condition. Reference: R 4.4.3,
speckle 1.6.0, limma 3.62.1.

| Check family | Reference | Largest difference: proportions, statistics | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Two groups (t-test) and three groups (F-test), 12 cell types: `BaselineProp`, `PropMean`, `PropRatio`, `Tstatistic` or `Fstatistic`, `P.Value` and `FDR` | `propeller` | 1.7e-15, 3.9e-9 | 1e-10, 1e-6 | 4 pass |
| Two groups with `robust = FALSE` and with `trend = TRUE` | `propeller` | 5.4e-16, 6.7e-15 | 1e-10, 1e-10 and 1e-6 | 2 pass |
| B - A and (A + B)/2 - C with a covariate, and B - A as `groupB` in `~ group + age` | `propeller.ttest` | 6.1e-16, 9.4e-15 | 1e-10, 1e-6 | 6 pass |
| F-test of `group` with a covariate | `propeller.anova` | 4.9e-16, 3.6e-15 | 1e-10, 1e-6 | 2 pass |
| The slope of the covariate, which speckle refuses | limma `eBayes` on speckle's transformed proportions | none reported, 3.0e-15 | 1e-6 | 2 pass |
| A pairing, in group means and in treatment coding | `propeller.ttest` | 4.7e-16, 7.7e-15 | 1e-10, 1e-6 | 4 pass |
| Few cells and absent cell types; the sample without cells, with the arcsine transform | `propeller`, `propeller.ttest` | 3.7e-16, 3.5e-15 | 1e-10 | 3 pass |
| Two cell types | `propeller` | 2.3e-16, 3.3e-15 | 1e-10 | 2 pass |
| Deliberate differences, listed below | `propeller.ttest`, `propeller` | 2 contrasts speckle refuses, 2 ratios of contrasts that compare nothing, 2 unused levels speckle keeps, 1 data set CytePool refuses | n/a | 7 deliberate differences |

Each row compares every column that speckle reports, matched by cell type, and the
columns themselves must match: the mean proportions of the levels a contrast
compares, and no others. Proportions and ratios are compared relative to speckle's
values, in the exact tier; statistics are scaled by max(\|reference\|, 1), and
p-values and FDRs by max(p, 1e-300) max(1, \|log p\|), in the exact tier, or in the
close tier with robust moderation of 10 cell types or more, whose prior limma solves
only to a tolerance. The largest difference, 3.9e-9, is in such a case; every other is
below 1e-14. Deliberate differences:

- speckle's `PropMean` are the coefficients of the proportions regressed on the
  columns of the design matrix that a contrast weighs, and its `PropRatio` their
  product raised to the contrast's weights (found by probing). Where those columns
  indicate groups of samples, as in the group-means designs above, they are the
  groups' mean proportions, and CytePool's values agree. Otherwise they are not. For
  A + B, which compares nothing, speckle reports the product of the two means as a
  ratio; CytePool reports no mean proportions and no ratio, and its t-statistics,
  p-values and FDRs agree with speckle's. For the slope of the covariate,
  `propeller.ttest` stops with "dim(X) must have a positive length", as it does for a
  contrast of a single column such as `groupB` in `~ group + age`. CytePool finds the
  levels a contrast compares from the design's coding: it tests the slope, without
  mean proportions or ratio, and its statistics agree with limma's `eBayes` on
  speckle's transformed proportions; `groupB` in `~ group + age` and `conditionstim`
  in `~ donor + condition` agree with speckle's values for the same comparisons in
  its group-means designs.
- A cell type without cells: from cell labels speckle never sees it, and CytePool,
  which leaves it out, agrees with it. From a factor of cell labels with an unused
  level, speckle tests that level, and with the logit transform its half cells change
  the transformed proportions of every other cell type (a scaled difference of 0.23
  here).
- A sample without cells, as an unused level of a factor of sample labels: with the
  logit transform speckle gives each of its I cell types the proportion 1/I and fits
  them as data (a scaled difference of 0.74 here); with the arcsine transform its
  proportions are NaN and limma leaves it out, as CytePool does with either transform.
- Samples that are copies of one another within each condition: the design fits the
  transformed proportions of every cell type exactly, so no variance can be estimated.
  speckle tests the rounding errors, with t-statistics up to 3.4e15 here; CytePool
  raises an error.

CytePool computes the transforms as log((x + 0.5) / (N - x + 0.5 (I - 1))) and
arctan2(sqrt(x), sqrt(N - x)), equal to speckle's formulas but without the
cancellation in 1 - q and the loss of digits of arcsin near 1; on these data the two
agree to rounding.

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
| `from_cells`, 2 threads | 0.092 |
| `from_cells`, 1 thread | 0.171 |
| `from_cells`, dense float32 input | 0.254 |
| `from_cells`, first call (compiles unless cached) | 0.379 |
| scanpy `normalize_total`, `log1p` and 3 `aggregate` calls (float32), warm | 0.282 |
| decoupler `pseudobulk` (sums only), warm | 0.583 |

The peak resident memory of that whole run was 1,725 MiB, in its largest process.

`anndata_conversion.py` timed each conversion of a cube of counts and detected cells
of 20 cell types x 50 samples x 20,000 genes (one call; `to_anndata` includes
computing the fingerprint, which the cube then keeps):

| Conversion | Seconds |
| --- | --- |
| `to_anndata` | 0.278 |
| `from_anndata` | 0.087 |
| `to_mudata` | 0.729 |
| `from_mudata` | 0.251 |

`linear_models.py --benchmark` on 20,000 genes x 200 samples (limma's time as R
measured it, CytePool's the median of three runs after a warm-up), with a peak
resident memory of 479 MiB in the largest process of the run. The scenarios with
observation weights or missing values fit each gene on its own, on 2 threads here:

| Scenario | limma `lmFit` (s) | CytePool (s) |
| --- | --- | --- |
| Ordinary least squares | 0.21 | 0.032 |
| Observation weights | 0.97 | 0.173 |
| 5% missing values | 0.73 | 0.170 |
| Block correlation | 0.13 | 0.090 |
| Block correlation and observation weights | 10.81 | 0.194 |

`adjust_p_values` with Hommel's method took 0.006 s on the 100,000 p-values of
`multiple_testing.py`. In `distributions.py`, mpmath took 395 s to compute the
references and CytePool 3.6 s to evaluate them, including compilation.

`cube_reads.py` read every layer of each selection of its cube file into memory with
`to_memory()`. The file holds 720 MB of values in chunks of one group, every sample
and 4,096 genes, 507 MB once compressed. Peak memory is the most that Python's
`tracemalloc` traced during a read, and time the best of three reads after a first
one:

| Selection | Chunk files opened | Read from disk (MB) | Decompressed (MB) | Result (MB) | Peak memory (MB) | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 50 neighbouring genes | 40 | 69.2 | 98.3 | 1.20 | 5.6 | 0.77 |
| 50 genes at random | 320 | 507.1 | 786.4 | 1.20 | 5.8 | 4.55 |
| 2,000 genes at random | 320 | 507.1 | 786.4 | 48.00 | 55.8 | 3.02 |
| 50 genes at random, 2 groups | 32 | 51.3 | 78.6 | 0.12 | 4.7 | 0.35 |
| 10 samples, every gene | 320 | 507.1 | 786.4 | 72.00 | 81.3 | 3.18 |
| one whole group | 16 | 25.0 | 39.3 | 36.00 | 40.5 | 0.20 |

A read holds what it returns and about one decompressed chunk (3.3 MB of `log_sum`).
Genes at random lie in every chunk, so they decompress the whole file, and so does a
selection of samples, since every chunk holds every sample. A read that covers a whole
group is hashed, to check it against the digest stored with the file, and a selection's
groups are hashed as they are read, for the selection's own fingerprint.

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
python validation/gene_set_tests.py
python validation/voom.py
python validation/sample_quality_weights.py
python validation/differential_expression.py
python validation/differential_expression_options.py
# Downloads 77 MB from GEO on its first run, into ~/.cache/cytepool-validation/GSE96583
# unless --data or GSE96583_DATA names another directory outside the repository.
python validation/interferon_stimulated_pbmcs.py
RSCRIPT=$REF/r-speckle/bin/Rscript python validation/composition_tests.py
python validation/distributions.py
REFERENCE_PYTHON=$REF/python-scanpy/bin/python python validation/cube.py
SCANPY_PYTHON=$REF/python-scanpy/bin/python \
    DECOUPLER_PYTHON=$REF/python/bin/python python validation/pool_cells.py
REFERENCE_PYTHON=$REF/python/bin/python $REF/python3.14/bin/python \
    validation/cube_files.py
python validation/cube_reads.py
python validation/cube_fingerprints.py
SCANPY_PYTHON=$REF/python-scanpy/bin/python DECOUPLER_PYTHON=$REF/python/bin/python \
    LIANA_PYTHON=$REF/python-liana/bin/python \
    $REF/python-anndata/bin/python validation/anndata_conversion.py
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

speckle 1.6.0 (Bioconductor 3.20), which `composition_tests.py` runs, has an R
environment of its own, with R 4.4.3, limma 3.62.1 and jsonlite 2.0.0:

```bash
conda create -p $REF/r-speckle -c conda-forge -c bioconda r-base=4.4.3 \
    bioconductor-limma=3.62.1 bioconductor-speckle=1.6.0 r-jsonlite=2.0.0
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

`anndata_conversion.py` runs CytePool in an environment with mudata, and LIANA in
one of its own, since LIANA requires pandas 2:

```bash
uv venv $REF/python-anndata --python 3.13.12
uv pip install --python $REF/python-anndata/bin/python numpy==2.5.4 numba==0.68.0 \
    scipy==1.18.1 pandas==3.1.0 anndata==0.13.4 mudata==0.4.1 zarr==3.4.1 \
    h5py==3.16.0
uv venv $REF/python-liana --python 3.13.12
uv pip install --python $REF/python-liana/bin/python liana==1.10.0 decoupler==2.2.0
```

The reference packages run only as separate programs that produce outputs to compare
against. Their source code is not read or copied: CytePool implements the methods from
their papers and documentation, and conventions that the documentation leaves open
were found by running the packages on chosen inputs.
