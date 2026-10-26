<h1><img src="https://docs.nygen.io/assets/brand/logo/cytepool-dark.png" alt="" width="96" align="right"> CytePool</h1>

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
| AnnData and MuData | `to_anndata`, `from_anndata`, `to_mudata`, `from_mudata` (optional extras `anndata` and `mudata`): cubes to and from the long AnnData of `scanpy.get.aggregate` and `decoupler.pp.pseudobulk` and the MuData of LIANA's `adata_to_views`, whose layouts are recognised; an import takes sums and counts of cells, refuses means, and gives a cube exported and read back with the same fingerprint | scanpy, decoupler, LIANA, anndata, mudata |
| Normalization and filtering | TMM, upper-quartile and RLE factors, DESeq2 size factors, CPM, log-CPM, `filter_by_expression` | edgeR, DESeq2 |
| Designs | Model matrices from R formulas, named contrasts, design checks | R, limma, lme4 |
| Linear models | Ordinary, weighted and generalised least squares for every gene (in parallel when each gene needs its own fit), contrasts (named contrasts are matched to the coefficients by name) | limma |
| Mixed models | `fit_mixed_model`: linear mixed models with random intercepts of one or more grouping factors (nested or crossed), with observation weights, fitted to every gene by REML or maximum likelihood, with exact zeros for variance components at the boundary, fitted values with the predicted random effects, and approximate residual degrees of freedom; `fixed_effect_tests`: t-tests of contrasts with Satterthwaite's degrees of freedom, as lmerTest computes them for dream | lme4, lmerTest, variancePartition |
| Precision weights for counts | `voom`: log-CPM values and observation weights from the mean-variance trend, whose LOWESS span is chosen from the number of genes as in limma, with prior weights for samples, observations or genes, for a weighted linear model (limma-voom); `voom_with_quality_weights`, which combines them with sample quality weights | limma |
| dream | `voom_with_dream_weights`: voom's precision weights from the mean-variance trend of every gene's mixed model, fitted by maximum likelihood, as variancePartition's `voomWithDreamWeights`; `moderated_fixed_effect_tests`: Satterthwaite t-tests of a mixed model moderated by empirical Bayes as variancePartition's `eBayes` moderates `dream` fits, with `topTable`'s z-statistic | variancePartition |
| Sample quality weights | `sample_quality_weights`: a weight for each sample, or each group of samples, from how closely its values follow the linear model, by REML or the gene-by-gene update, as limma's `arrayWeights` | limma |
| Negative binomial models | `fit_negative_binomial_model`: a negative binomial log-linear model for every gene, with known dispersions, as edgeR's `glmFit` (one-way layouts group by group, other designs by Newton's method with Levenberg damping, in parallel), with log fold changes shrunk by prior counts; exact unit deviances and deviances | edgeR, mpmath |
| Quasi-likelihood F-tests | `fit_quasi_likelihood` and `quasi_likelihood_f_test`: edgeR 4's `glmQLFit` and `glmQLFTest` with their defaults: the common negative binomial dispersion of the most highly expressed genes by Cox-Reid adjusted profile likelihood (`common_dispersion`), deviances adjusted for small counts with exact moments of the unit deviance (`negative_binomial_unit_deviance_moments`), the average quasi-dispersion, limma's prior for unequal degrees of freedom with a trend in the average log CPM (`average_log_cpm`) or robust, prior weights and offsets, and F-tests of coefficients and named contrasts with edgeR's columns (logFC, logCPM, F, PValue, FDR) | edgeR, limma, mpmath |
| DESeq2 | `fit_deseq2`, `deseq2_wald_test`, `deseq2_likelihood_ratio_test` and `deseq2_dispersions`: DESeq2's analysis with its defaults, from gene-wise dispersions (Cox-Reid adjusted profile likelihood), the parametric trend and the maximum a posteriori dispersions to its fits of every gene, Wald tests of coefficients and contrasts, and likelihood ratio tests of a reduced design (all the levels of a factor at once, a time course), with the columns of `results()`, its independent filtering and its Cook's distance flags | DESeq2, PyDESeq2 |
| Empirical Bayes moderation | Moderated t and F statistics, B-statistics, TREAT, with constant, trend and robust priors, estimated as limma estimates them by default: by the method of moments, or, when the residual degrees of freedom differ between genes, by weighted moments and profile likelihood (limma's `fitFDistUnequalDF1`); either can be chosen | limma |
| Gene-set tests | `camera` (competitive, allowing for the correlation between the genes of a set, given or estimated per set; parametric or rank-based) and `fry` (self-contained, directional and mixed p-values), for one contrast of a linear model, with voom or other precision weights and trend or robust moderation; the rotation tests `roast` and `mroast` (self-contained, with the mean, floor-mean, mean-50 and mean-square set statistics, contribution weights per gene or per set, and the proportions of active genes) and `romer` (competitive, by the ranks of the genes), with rotations from a seeded generator; gene sets named by gene id or position | limma |
| Multiple testing | Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg, Benjamini-Yekutieli | R |
| Differential expression | `limma_voom` and `limma_trend`: limma-voom and limma-trend on every group (cell type) of a cube, and `dream_voom`: dream, with the design's random intercepts, such as a donor's, in every group's fits, with the design matched to the cube by sample id, samples with too few cells left out, rank-deficient groups fitted on their column space with the contrasts they can estimate, and a provenance record for every result. Options for pseudobulk data, off by default: sample quality weights, prior weights from the number of cells, a minimum total count per sample, and p-values adjusted across cell types | limma, edgeR, variancePartition; on simulated data and on published single-cell data |
| Meta-analysis | `meta_analysis`: for every gene, the fixed-effect (inverse-variance) or random-effects estimate of a common effect from the effects and standard errors of several studies, with the between-study variance by DerSimonian-Laird, REML or Paule-Mandel, Cochran's Q, I² and H², z-tests or the Knapp-Hartung adjustment, and genes measured in some studies only; `differential_expression_meta_analysis`: the same for the `limma_voom` or `limma_trend` results of several cubes, matched by cell type, contrast and gene id, with p-values adjusted over the genes and a provenance record | metafor |
| Composition tests | `propeller`: differences in cell-type proportions between conditions, from the cell numbers of a cube or a table of them, with the logit or arcsine square-root transform, a design matched by sample id, t-tests of its contrasts and the F-test of a factor, robust empirical Bayes moderation, speckle's columns (baseline and mean proportions, proportion ratio, statistic, p-value, FDR) and a provenance record | speckle, limma |
| Figures | `cytepool.plotting` (optional extra `plot`, matplotlib): the plots of limma, edgeR, DESeq2, speckle and metafor from CytePool's results, in one call each: MDS plots of pseudobulk samples (`plotMDS`), the cells pooled into each sample and cell type, cell-type composition and `propeller`'s results, voom's mean-variance trend, residual variances and their prior (`plotSA`), sample quality weights, MD and volcano plots, p-value histograms, the genes up and down in each cell type, barcode plots and summaries of gene-set tests, DESeq2's dispersions (`plotDispEsts`), edgeR's quasi-dispersions (`plotQLDisp`) and forest plots of meta-analyses | limma, edgeR, DESeq2, metafor, speckle |
| Distributions and smoothers | Tails of the normal, t, F, chi-squared, gamma, beta and hypergeometric distributions, and quantiles of the first four; LOWESS, and limma's rule for its span; natural spline bases | mpmath, R, limma |

## Planned

These are not implemented yet:

- Likelihood ratio tests and gene-wise dispersions for negative binomial models
  (edgeR's `glmLRT` and `estimateDisp`), edgeR's quasi-likelihood F-tests on every
  group of a cube, and their `legacy = TRUE` methods
- DESeq2's shrunken log fold changes (`lfcShrink` with apeglm, ashr or normal priors),
  its replacement of outlier counts, and its local and glmGamPoi dispersion trends
- The rest of dream (variancePartition): Kenward-Roger degrees of freedom (dream's
  default for 20 samples or fewer), F-tests of several contrasts at once, random
  slopes, and variance partitioning. dream with random intercepts and Satterthwaite's
  degrees of freedom is implemented (see the table above); `fit_linear_model` refuses
  a design whose random effects it would drop, and its message points to
  `fit_mixed_model`.
- Pathway or transcription-factor activity
- Composition tests with precision weights or random effects (crumblr)
- Ligand-receptor scoring and differential interactions
- Dose-response fits, benchmark doses and synergy
- Meta-regression (meta-analysis with study-level covariates) and multivariate
  meta-analysis of correlated effects

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
- A mixed model stores, for every gene, the covariance matrix of the coefficients and
  its derivative with respect to each variance parameter, for Satterthwaite's degrees
  of freedom: about 8 x genes x (components + 2) x coefficients² bytes, so 23 MB for
  20,000 genes, two random intercepts and 6 coefficients. `fit_mixed_model` fits random
  intercepts only, by REML or maximum likelihood, to complete data with positive
  weights: missing values or a zero weight make it raise an error, as no gene is
  fitted to fewer samples. Its fitted values take as much memory as the values. It also
  refuses a design whose fixed and random effects together span every sample, which
  lme4 fits. A gene whose values the fixed and random effects reproduce, such as one
  constant within every donor, has no REML estimate and is returned unfitted (NaN).
  Its genes run on Python threads, in compiled code that releases the GIL, so the
  forking limit above does not apply to it.
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
- With residual degrees of freedom that differ between genes, robust moderation
  (`robust=True`) uses limma's likelihood estimator, as limma does, but leaves the
  genes without residual degrees of freedom out of its test for outliers. limma 3.62.1
  counts them in that test as zero variances, which become outliers and change which
  other genes count as outliers: in a simulation where 2% of the genes had no residual
  degrees of freedom, its robust prior degrees of freedom were 11.5% larger with a
  constant prior and 6% to 10% larger with a trend. Where every gene has residual
  degrees of freedom, the two agree. `fit_f_distribution_likelihood` also keeps prior
  weights in its second robust fit, where limma's drops them.
- Contrast standard errors and moderated F statistics are exact by default. limma's
  approximations are available as an option and can differ when weights vary between
  genes.
- `voom` normalizes by library-size factors only, and fits its trend without a block
  correlation; `voomLmFit` is not implemented. `voom` and `voom_with_quality_weights`
  need a design of full column rank (limma drops aliased columns; `limma_voom` fits a
  group's column space), and raise an error where limma returns values without
  meaning: for a sample whose library size is zero, or a trend that is not positive
  where weights are read. With prior weights, `voom` returns the trend's weights
  times the prior weights, which a fit needs; limma returns the trend's alone.
- In `limma_voom` and `limma_trend`, sample quality weights give each sample a weight
  of its own; there is no variance model for groups of samples (limma's `var.group`
  and `var.design`). Prior weights are the number of cells of each sample and cannot
  be combined with quality weights, which edgeR's `voomLmFit` refuses too. In a
  design that pairs two samples per donor, both samples of a donor get the same
  quality weight from limma-trend and similar ones from limma-voom, as their
  residuals are equal and opposite.
- `sample_quality_weights` stops REML where limma does, so that its weights are
  limma's: within about half a percent of the REML estimate without prior weights, and
  up to about 15% from it with prior weights (as for the voom weights of
  `voom_with_quality_weights` with `method="reml"`), where limma stops after one or
  two iterations, and further where the iterations oscillate and limma stops early. A
  smaller `tolerance` converges further where the iterations converge. The
  gene-by-gene algorithm, limma's default for quality weights, depends on the order of
  the genes, as limma's does, and its time grows with the cube of the number of
  samples: 0.2 to 0.3 s for 20,000 genes and 40 samples, 2.5 to 3 s for 100 samples.
- `voom` chooses the span of its trend from the number of genes, as limma does by
  default since limma 3.66, and limma 3.60 to 3.65 do with `adaptive.span = TRUE`.
  Their default, and that of earlier versions, is a span of 0.5: pass `span=0.5` to
  reproduce it.
- `camera` and `fry` need every value observed, with positive weights, and fit fixed
  effects only (no block correlation); `fry` has no gene contribution weights (limma's
  `gene.weights`). `fry`'s mixed p-value of a set of 2 to d genes (d the residual
  degrees of freedom) differs from limma's on purpose: it uses all d + 1 dimensions of
  the rotation, where limma's uses only the set's and can be far from the rotation
  p-value. For larger sets the two agree, and their beta approximation can be much too
  small far in the tail. `camera` also offers robust moderation, which limma's does
  not.
- `fit_negative_binomial_model` takes the dispersions as given; CytePool estimates
  only a common dispersion (`common_dispersion`). Where a gene's maximum likelihood
  estimate is infinite, as when every count of a group is zero, its fitted values tend
  to zero and its coefficients are large but finite: -1e8 for such a group of a
  one-way layout, as in edgeR. Its deviances are exact, and differ from edgeR's, which
  adds 1e-8 to counts and means and approximates the deviance for dispersions below
  1e-4 and where the dispersion times the mean reaches 1e6. Where weights differ
  between samples by a factor of more than about 1e10, the coefficients that only the
  samples of small weight determine lose accuracy (about 1e-16 times that factor), and
  beyond about 1e14 they are not determined.
- `roast`, `mroast` and `romer` draw their rotations from a numpy generator, so their
  p-values agree with limma's within Monte Carlo error, and exactly when given the
  rotations limma draws; `mroast` tests every set with the same rotations. Like `camera`
  and `fry` they need every value observed and fit fixed effects only (no block
  correlation), and `romer` takes sample weights only, as limma's does. A rotated
  statistic equal to the observed one counts as at least as extreme, as Wu et al.
  (2010) define the p-value; limma's `roast` counts only larger ones, which gives about
  1/2 instead of 1 where a set has no gene on one side of zero (`"floormean"`,
  `"msq"`). With contribution weights, a gene of weight zero leaves the set, and
  `"msq"` and the mixed `"floormean"` statistic weigh each gene by the size of its
  weight, as the paper does; limma's `"msq"` p-values differ for weights of both signs
  (and change with the scale of the weights beyond 1,000 rotations), and its
  `"floormean"` takes a negative weight as evidence against the set. `romer` shrinks
  genes whose log-odds exceed 709.8, which limma's shrinkage turns into NaN. Exact
  z-scores (limma's `approx.zscore = FALSE`) and given prior parameters (`var.prior`,
  `df.prior`) are not offered.
- `fit_quasi_likelihood` follows edgeR 4.4's default method only (`legacy = FALSE`),
  as edgeR runs it on a matrix of counts; a given dispersion should come from a
  global model (common or trended), as edgeR's help page says. edgeR's method for a
  DGEList agrees, except when the DGEList holds a trended dispersion from
  `estimateDisp`: it then takes one dispersion from that trend in place of the
  Cox-Reid estimate. The moments of the unit deviance are exact, where edgeR's
  approximations are off by up to about 1e-3 at dispersions up to 0.2 and 1e-2 at a
  dispersion of 1, so that adjusted deviances, quasi-dispersions and F-statistics
  differ from edgeR's by up to about 3e-3. The robust prior leaves genes without
  residual degrees of freedom out of the outlier tests, where limma counts them as
  zero variances. With offsets per observation, edgeR's `aveLogCPM` ignores the
  offsets of the first gene, and so differs from CytePool's there.
- `fit_negative_binomial_model` takes the dispersions as given; CytePool does not
  estimate them yet. Where a gene's maximum likelihood estimate is infinite, as when
  every count of a group is zero, its fitted values tend to zero and its coefficients
  are large but finite: -1e8 for such a group of a one-way layout, as in edgeR. Its
  deviances are exact, and differ from edgeR's, which adds 1e-8 to counts and means
  and approximates the deviance for dispersions below 1e-4 and where the dispersion
  times the mean reaches 1e6. Where weights differ between samples by a factor of more
  than about 1e10, the coefficients that only the samples of small weight determine
  lose accuracy (about 1e-16 times that factor), and beyond about 1e14 they are not
  determined.
- The DESeq2 functions give the results of `DESeq(..., minReplicatesForReplace =
  Inf)`: outlier counts are not replaced, which DESeq2 does by default for samples
  whose design row occurs seven or more times. Each dispersion is the maximum of its
  objective, where DESeq2's line searches stop when an iteration gains less than 1e-6
  and can stall where they start, but for genes whose rough dispersion is at the lower
  bound, which keep it as in DESeq2; and the fits converge, where DESeq2's stop when
  the deviance changes by less than 1e-8 of it. On the validation's data DESeq2's
  final dispersions differ by medians of 1e-3 to 4e-2 on the log scale, and its
  p-values by medians of 3e-4 to 8e-3, relative, with 7 of 1,297 genes called
  differently at a false discovery rate of 0.1. Where DESeq2's parametric trend fails
  it substitutes a local regression; CytePool uses the mean of the gene-wise
  estimates, with a warning. With three or fewer residual degrees of freedom the
  prior's variance is the maximum likelihood estimate, where DESeq2 compares densities
  of simulated residuals: 10% apart in the validation. For Cook's distance a design
  matrix given without its `Design` compares two groups when its rows take two
  distinct values of zeros and ones, as a factor does in DESeq2 and a numeric
  covariate does not. Observation weights, normalization factors that differ between
  genes, thresholds on the fold change (`lfcThreshold`) and t-distributed Wald tests
  (`useT`) are not implemented.
- `limma_voom` and `limma_trend` fit fixed effects only: a design with random effects
  is refused (`dream_voom` fits random intercepts), and there is no
  `duplicateCorrelation`. Normalization factors are TMM or
  none. In a group whose design is rank deficient they test every contrast the
  group's samples can estimate, including those that weigh an aliased column, which
  limma's `contrasts.fit` refuses, and they report the contrasts that cannot be
  estimated, which limma either refuses or, when they weigh no aliased column,
  computes with a changed meaning.
- `dream_voom` and `moderated_fixed_effect_tests` use Satterthwaite's degrees of
  freedom, as `dream(ddf = "Satterthwaite")`; dream's default, `ddf = "adaptive"`,
  uses Kenward-Roger's for 20 samples or fewer, which are not available yet, nor are
  sample quality weights for dream. As variancePartition, they moderate the residual
  variance of an observation of each gene's average weight, which keeps the
  mean-variance trend that voom's weights remove; `trend=True` lets the prior follow
  it. `voom_with_dream_weights` chooses its span from the number of genes, as `voom`
  does; `voomWithDreamWeights` uses 0.5 (`span=0.5`). A gene whose values the fixed
  and random effects reproduce is left out of the trend and untested, where
  variancePartition keeps it with a residual standard deviation near zero.
- `from_anndata` and `from_mudata` take only sums and counts of cells: means,
  variances and medians, which `scanpy.get.aggregate` and `decoupler.pp.pseudobulk`
  can give, are refused, and counts are checked against the cells that detect each
  feature. A combination of sample and cell type without a row needs cell numbers to
  become an entry without cells, and a row removed after pooling (decoupler's
  `filter_samples`, LIANA's `min_cells`) becomes one too, so its cells are missing
  when groups are merged or samples pooled: import before filtering. LIANA's
  `adata_to_views` keeps no cell numbers with decoupler 2, so its views need them
  given, for example as a crosstab of the cells' labels.
- `to_anndata` and `to_mudata` leave `X` empty and write every layer under its own
  name. anndata cannot write datetime or timedelta columns, text columns without any
  value, or column names with `/`, so the exports write them as ISO 8601 text,
  integers, NaN and names with `_`; the record in `uns` restores them on import. A
  MuData file cannot store a modality whose name holds `/` or is `.` or `..`. The
  tests of these
  conversions are skipped on Python 3.15, for which h5py and numcodecs, which
  anndata needs, have no wheels yet.
- `propeller` fits fixed effects only: a design with random effects is refused, and
  there is no `duplicateCorrelation`. It reads whole numbers of cells, not
  proportions with assumed totals (speckle's `convertDataToList`), and its F-test
  compares all the levels of a factor; to compare some of them, test their contrasts,
  whose p-values use limma's degrees of freedom for t-tests, capped at the residual
  degrees of freedom of all cell types, where those of F-tests are not. It leaves out
  a cell type or a sample without cells, which speckle keeps when it is an unused
  level of a factor of labels; it tests contrasts that speckle refuses, such as the
  slope of a covariate, and gives no mean proportions or ratio for a contrast that
  compares no levels, where speckle reports regression coefficients; and it refuses
  data that the design fits exactly, such as copies of samples.
- `meta_analysis` fits models without moderators and takes the studies as independent,
  with known sampling variances; `differential_expression_meta_analysis` takes limma's
  moderated standard errors as such, so a gene that one study tested, kept with
  `min_studies=1`, gets a z-test whose p-value is smaller than the study's own
  t-test's. Its REML estimate is the largest maximum of the restricted likelihood:
  where that likelihood has two maxima, as when an imprecise study lies far from
  precise ones that agree, metafor's Fisher scoring from the Hedges estimate can stop
  at the lower one or at zero (in 2 of the 9 such features of one validation case),
  and on some features it does not converge without `stepadj`. Its REML and
  Paule-Mandel estimates converge to rounding, beyond metafor's default stopping
  rules, and it reports I² as a proportion, where metafor reports a percentage. It
  refuses standard errors that are zero or infinite; NaN marks a study that did not
  measure a gene.
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

The documentation is published at <https://docs.nygen.io/CytePool/>. It has a quick
start and guides, pages on how the statistics work, an API reference, a
[capability matrix](docs/source/reference/capabilities.md) and a
[Validation](docs/source/validation.md) page that reports, for every validation
script, the reference, the measured differences, the tolerances and the outcome. Its
sources are in [`docs/`](docs/); build it with
`uv sync --extra test --group dev --group docs` and `make -C docs html` with Node.js
20 or newer (CI uses Node.js 24), as
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
