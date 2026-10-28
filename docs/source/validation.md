---
description: How closely each method in CytePool agrees with the tool that defined it, from the tables the validation scripts printed on 2026-10-24, 2026-10-25 and 2026-10-26.
---

(validation)=
# Validation

Every area of CytePool has a script in
[`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation) that
compares it with an independent reference: the package that defined the method, run
as a black box, or a high-precision computation. A script generates its inputs from
fixed random seeds, or reads published data, runs the reference as a separate program
on the same inputs,
exchanging data through files in a temporary directory (float64 values as binary, so
they compare bit for bit), runs CytePool, prints a table with one row per check, and
exits with an error when any check falls outside its tolerance.

This page summarizes the tables the scripts printed when they were run against the
package in this revision of the repository: on **2026-10-26** for `plotting.py`,
`variance_partitioning.py`, `gene_set_files_and_ora.py`, `kenward_roger_and_f_tests.py`,
`gsea.py`, `cube_collection.py`, `pathway_scores.py`,
`combined_p_values_and_replication.py`, `activity.py`, `sumrank.py`,
`pca_and_dot_plots.py` and `distributions.py`, and for `dream.py`, which printed the
same table again after changes to the code it checks; on
**2026-10-25** for the six scripts on
cubes and their files (`cube.py`, `pool_cells.py`, `cube_files.py`, `cube_reads.py`,
`cube_fingerprints.py` and `anndata_conversion.py`), for `linear_models.py`,
`mixed_models.py`, `voom.py`, `empirical_bayes.py`, `sample_quality_weights.py`,
`negative_binomial_fits.py`, `quasi_likelihood.py`, `deseq2_tests.py`,
`differential_expression.py`, `differential_expression_options.py`,
`interferon_stimulated_pbmcs.py`, `gene_set_tests.py`, `rotation_gene_set_tests.py`,
`composition_tests.py` and `meta_analysis.py`, and for `design.py`, which printed the
same table again after changes to the code it checks; and on **2026-10-24** for the
others, with the
reference versions given in each section. Every script exited without a failure.
In the table of each section, a row is a family of checks: it gives the number of
printed checks it covers and the largest difference among them. Run a script to see
every check.

## Summary

Each method is listed with the functions that implement it, the reference it is checked
against and its version, the tier with the bounds its checks use, and the section that
gives every check. The sections are grouped as the chapters are, and each opens with the
checks its script printed, their outcomes, its run time and its peak memory.

| Method | Functions | Checked against | Tier and bound | Section |
| --- | --- | --- | --- | --- |
| Pseudobulk profiles pooled from cells | {api}`from_cells` | scanpy 1.12.4, decoupler 2.2.0 | exact: identical sums of whole counts, 1e-10 otherwise | [Pooling cells into cubes](#validation-pool-cells) |
| Bulk counts as a cube | {api}`from_bulk` | decoupler 2.2.0 | exact: identical | [Pooling cells into cubes](#validation-pool-cells) |
| Merging groups and pooling samples; means, variances and detection | {api}`Cube` | scanpy 1.12.4 | exact: identical counts, 1e-10; variances from raw moments to a bound derived for each case | [Pseudobulk cubes](#validation-cube) |
| Cube files | {api}`write_cube`, {api}`read_cube` | zarr-python 3.4.1 | exact: identical | [Cube files](#validation-cube-files) |
| Reads of selected features from cube files | {api}`read_cube`, {api}`Cube.select` | zarr-python 3.4.1 | exact: identical, 1e-12 for merged groups and pooled samples | [Reads of selected features](#validation-cube-reads) |
| Fingerprints of cubes | {api}`Cube.fingerprint`, {api}`Cube.digests` | the documented encoding, with zarr-python 3.4.1 | exact: identical digests | [Cube fingerprints](#validation-cube-fingerprints) |
| AnnData and MuData layouts of scanpy, decoupler and LIANA | {api}`to_anndata`, {api}`from_anndata`, {api}`to_mudata`, {api}`from_mudata` | scanpy 1.12.4, decoupler 2.2.0, LIANA 1.10.0, anndata 0.13.4, mudata 0.4.1 | exact: identical, 1e-10 for sums of normalized values | [Conversions to AnnData and MuData](#validation-anndata-conversion) |
| Model matrices from R formulas, named contrasts, design checks | {api}`Design`, {api}`model_matrix`, {api}`make_contrasts`, {api}`Design.check` | R 4.4.3, limma 3.62.1, lme4 2.0.6 | exact: identical, 1e-10; 1e-8 (absolute) for the leverages of badly scaled designs | [Designs and contrasts](#validation-design) |
| Linear models of every gene and their contrasts (limma's `lmFit`, `contrasts.fit`) | {api}`fit_linear_model`, {api}`contrasts_fit` | limma 3.62.1 | exact: 1e-10 | [Linear models](#validation-linear-models) |
| Empirical Bayes moderation, moderated t and F, B-statistics, TREAT | {api}`moderated_statistics`, {api}`treat`, {api}`squeeze_variances`, {api}`fit_f_distribution`, {api}`fit_f_distribution_robust`, {api}`fit_f_distribution_likelihood` | limma 3.62.1 | exact: 1e-10; close: 1e-6 for the robust and likelihood priors | [Empirical Bayes moderation](#validation-empirical-bayes) |
| Adjustment for multiple testing (R's `p.adjust`) | {api}`adjust_p_values` | R 4.4.3 | exact: 1e-10 | [Multiple testing](#validation-multiple-testing) |
| TMM, upper-quartile and RLE factors; CPM and log-CPM; `filterByExpr` | {api}`tmm_factors`, {api}`upper_quartile_factors`, {api}`rle_factors`, {api}`cpm`, {api}`log2_cpm`, {api}`filter_by_expression` | edgeR 4.4.0 | exact: 1e-10, identical genes kept | [Normalization and filtering](#validation-normalization) |
| DESeq2's median-of-ratios size factors | {api}`median_ratio_size_factors` | DESeq2 1.46.0 | exact: 1e-10 | [Normalization and filtering](#validation-normalization) |
| voom's precision weights and limma's span rule | {api}`voom`, {api}`choose_lowess_span` | limma 3.62.1, edgeR 4.4.0 | exact: 1e-10; close: 1e-6 on rarefied counts | [Precision weights for counts](#validation-voom) |
| Sample quality weights (`arrayWeights`, `voomWithQualityWeights`) | {api}`sample_quality_weights`, {api}`voom_with_quality_weights` | limma 3.62.1, edgeR 4.4.0 | close: 1e-6; exact: 1e-10 for voom with prior weights, and for the log-CPM values and the span with quality weights | [Sample quality weights and prior weights](#validation-sample-quality-weights) |
| limma-voom and limma-trend on every cell type of a cube | {api}`limma_voom`, {api}`limma_trend` | limma 3.62.1, edgeR 4.4.0 | exact: 1e-10; close: 1e-6 with robust moderation | [limma-voom and limma-trend on cubes](#validation-differential-expression) |
| Options for pseudobulk data: quality weights, weights from cell numbers, a minimum count, adjustment across cell types | {api}`limma_voom`, {api}`limma_trend` | limma 3.62.1, edgeR 4.4.0 | exact: 1e-10; close: 1e-6 with quality weights | [Options for pseudobulk differential expression](#validation-differential-expression-options) |
| limma-voom on published single-cell data (Kang et al. 2018) | {api}`from_cells`, {api}`limma_voom` | limma 3.62.1, edgeR 4.4.0 | exact: identical counts, 1e-10 | [Differential expression on published data](#validation-interferon-stimulated-pbmcs) |
| Negative binomial models of every gene (edgeR's `glmFit`), deviances | {api}`fit_negative_binomial_model`, {api}`negative_binomial_deviance`, {api}`negative_binomial_unit_deviance` | edgeR 4.4.0, mpmath 1.3.0 | close: 1e-6; 1e-13 for unit deviances against mpmath | [Negative binomial models](#validation-negative-binomial-fits) |
| Quasi-likelihood F-tests (edgeR 4's `glmQLFit`, `glmQLFTest`) | {api}`fit_quasi_likelihood`, {api}`quasi_likelihood_f_test`, {api}`common_dispersion`, {api}`average_log2_cpm` | edgeR 4.4.0, limma 3.62.1, mpmath 1.3.0 | close: 1e-6 where both programs compute the same quantity; up to 2e-2 where edgeR approximates the moments of the unit deviance, which CytePool computes exactly, and where edgeR's searches and fits stop short | [Quasi-likelihood F-tests](#validation-quasi-likelihood) |
| DESeq2's dispersions, Wald tests and likelihood ratio tests | {api}`fit_deseq2`, {api}`deseq2_dispersions`, {api}`deseq2_wald_test`, {api}`deseq2_likelihood_ratio_test` | DESeq2 1.46.0, PyDESeq2 0.5.4, mpmath 1.3.0 | close: 1e-6, each stage given DESeq2's own inputs; 1e-5 for fitted values, Cook's distances and Wald tests, as DESeq2's fits stop up to 1.5e-6 from their fixed point; 1e-9 for deviances and likelihood ratio tests; exact: 1e-10 for size factors, the trend and the variance of the prior | [DESeq2's dispersions and tests](#validation-deseq2-tests) |
| Mixed models with random intercepts by REML, Satterthwaite's t-tests (maximum likelihood fits are checked with dream) | {api}`fit_mixed_model`, {api}`fixed_effect_tests` | lme4 2.0.6, lmerTest 3.2.1 | close: 1e-6; 1e-5 for Satterthwaite's degrees of freedom and p-values and for fits with large variance ratios; exact: 1e-10 for the REML criterion and at given estimates | [Mixed models](#validation-mixed-models) |
| Kenward-Roger's t-tests and F-tests, Satterthwaite's F-tests (pbkrtest's `vcovAdj` and `KRmodcomp`, lmerTest's `contestMD`); dream's choice of degrees of freedom and its F-tests | {api}`fixed_effect_tests`, {api}`fixed_effect_f_test`, {api}`dream_voom`, {api}`moderated_fixed_effect_tests` | pbkrtest 0.5.3, lmerTest 3.1-3, lme4 1.1-38, variancePartition 1.36.2, limma 3.62.1 | exact: 1e-10 at given estimates; close: 1e-6 at the optimum and for dream; 1e-5 for the degrees of freedom and p-values of Satterthwaite's F-tests | [Kenward-Roger's tests and F-tests](#validation-kenward-roger-and-f-tests) |
| dream on every cell type of a cube | {api}`dream_voom`, {api}`voom_with_dream_weights`, {api}`moderated_fixed_effect_tests` | variancePartition 1.36.2, limma 3.62.1, lme4 1.1-38, lmerTest 3.1-3 | close: 1e-6; 1e-5 for Satterthwaite's degrees of freedom and the p-values and z-statistics that use them; exact: 1e-10 for the likelihoods at given estimates | [dream's weights, fits and tests](#validation-dream) |
| Variance partitioning (`fitExtractVarPartModel`) on every cell type of a cube, and its figures (`plotVarPart`, `plotPercentBars`) | {api}`variance_fractions`, {api}`variance_partitioning`, {api}`plot_variance_fractions`, {api}`plot_feature_variance_fractions` | variancePartition 1.36.2, lme4 1.1-38, ggplot2 4.0.3 | close: 1e-6; exact: 1e-10 at given estimates, for linear models and for the figures; 1e-3 for the widths of the violins | [Variance partitioning](#validation-variance-partitioning) |
| Cell-type composition (speckle's propeller) | {api}`propeller` | speckle 1.6.0, limma 3.62.1 | exact: 1e-10; close: 1e-6 for the robust prior | [Cell-type composition](#validation-composition-tests) |
| Gene-set tests camera and fry | {api}`camera`, {api}`fry` | limma 3.62.1 | exact: 1e-10; close: 1e-6 with robust moderation; 1e-3 for fry's default standardization, which differs on purpose; 0.1 (absolute) for fry's mixed p-values against random rotations | [Gene-set tests](#validation-gene-set-tests) |
| Rotation gene-set tests roast, mroast and romer | {api}`roast`, {api}`mroast`, {api}`romer` | limma 3.62.1 | calibrated: within 4 standard deviations of the Monte Carlo difference, plus 2 / (B + 1) for B rotations; exact: 1e-12 given limma's rotations; 0.1 (absolute) for fry's mixed p-values against the rotation test | [Rotation gene-set tests](#validation-rotation-gene-set-tests) |
| Gene-set files (GMT) read and written; collections matched to an analysis (limma's `ids2indices`) and filtered by size | {api}`read_gmt`, {api}`write_gmt`, {api}`GeneSets` | R 4.4.3, limma 3.62.1, decoupler 2.2.0 | exact: identical, 1e-10 | [Gene-set files and over-representation](#validation-gene-set-files-and-ora) |
| Over-representation analysis (limma's `kegga`) of the genes up, down or either way, and its dot plot | {api}`over_representation`, {api}`plot_over_representation` | limma 3.62.1, R 4.4.3 | exact: 1e-10 | [Gene-set files and over-representation](#validation-gene-set-files-and-ora) |
| Gene set enrichment analysis of ranked genes (fgsea's `fgseaMultilevel` and `fgseaSimple`), with its enrichment plot and table (`plotEnrichment`, `plotGseaTable`) | {api}`gsea`, {api}`plot_enrichment`, {api}`plot_gsea_table` | fgsea 1.32.2 | exact: 1e-10 given fgsea's seeds, and for enrichment scores, leading edges and figures; calibrated: within 4 standard errors over 20 runs in which each program draws its own random sets | [Gene set enrichment analysis](#validation-gsea) |
| Scores of gene sets in every sample (GSVA's `gsva` with `gsvaParam`, `ssgseaParam`, `zscoreParam` and `plageParam`), and their heatmap | {api}`gsva`, {api}`ssgsea`, {api}`combined_z_scores`, {api}`plage`, {api}`plot_pathway_scores` | GSVA 2.0.0 | exact: 1e-10, PLAGE up to sign | [Pathway scores](#validation-pathway-scores) |
| Activities of regulators and pathways (decoupler's `mt.ulm`, `mt.mlm` and `mt.waggr`) from samples, a cube's cell type or contrasts, the networks CollecTRI and PROGENy downloaded with their checksums and licences, and the bar plot of the top sources (`pl.barplot`) | {api}`ulm`, {api}`mlm`, {api}`weighted_aggregate`, {api}`collectri`, {api}`progeny`, {api}`plot_top_activities` | decoupler 2.2.0, progeny 1.28.0 | exact: 1e-10, normalized scores given decoupler's permutations; 1.2e-7 for ULM's adjusted p-values, which decoupler keeps in single precision | [Pathway and regulator activity](#validation-activity) |
| Tests of pathway scores and activities with a design (limma's `lmFit`, `contrasts.fit` and `eBayes` on scores) | {api}`limma` | the linear models and moderation of this page: `tests/test_score_tests.py` checks that the results are theirs | exact: identical | none of its own; [Linear models](#validation-linear-models), [Empirical Bayes moderation](#validation-empirical-bayes) |
| Several studies' cubes aligned by sample, gene and cell type, with what each study measured; stacked for a joint model; each study's results lined up gene by gene | {api}`CubeCollection`, {api}`StudyResults.table` | anndata 0.13.4, pandas 3.1.0, limma 3.62.1, edgeR 4.4.0 | exact: identical, 1e-10 | [Studies aligned in a collection](#validation-cube-collection) |
| Meta-analysis by fixed and random effects (metafor's `rma.uni`) | {api}`meta_analysis`, {api}`fit_meta_analysis` | metafor 5.0.1 | exact: 1e-10; close: 1e-6 for REML and Paule-Mandel | [Meta-analysis](#validation-meta-analysis) |
| Combined p-values of several studies (Fisher, Stouffer with weights, Tippett, Wilkinson, Pearson, Mudholkar and George, Simes), one-sided or two-sided with the directions of the effects; partial conjunctions, an effect in at least u studies | {api}`combine_p_values`, {api}`partial_conjunction` | scipy 1.18.1, metap 1.1, mpmath 1.3.0 | exact: 1e-10 | [Combined p-values and replication](#validation-combined-p-values-and-replication) |
| The proportion of true nulls, pi0, and pi1 (qvalue's `pi0est`, its smoother and bootstrap) | {api}`proportion_of_true_nulls` | qvalue 2.38.0 | exact: 1e-10 | [Combined p-values and replication](#validation-combined-p-values-and-replication) |
| Replication across the studies of a collection: combined and partial conjunction p-values, sign agreement, heterogeneity, pi1 between studies, genes found in k studies, and its figures | {api}`replication`, {api}`plot_study_agreement`, {api}`plot_replicated_genes` | metap 1.1, qvalue 2.38.0, metafor 5.0.1 | exact: 1e-10; close: 1e-6 for REML | [Combined p-values and replication](#validation-combined-p-values-and-replication) |
| SumRank (Nakatsuka et al. 2025): every gene's best ranks summed across studies, with Irwin-Hall p-values and permutations that re-run each study's analysis; prediction of cases in held-out studies by genes chosen from the others, by SumRank and by meta-analysis; their figures | {api}`sumrank`, {api}`fit_sumrank`, {api}`held_out_prediction`, {api}`plot_sumrank`, {api}`plot_held_out_auc` | the authors' code (commit 46392f66), R 4.4.3, mpmath 1.3.0, scipy 1.18.1, scikit-learn 1.9.1 | exact: 1e-12; calibrated: within 4 standard errors for p-values from random permutations | [SumRank and held-out prediction](#validation-sumrank) |
| Figures of limma, edgeR, DESeq2, speckle and metafor (MDS, voom's trend, plotSA, MD and volcano plots, barcode plots, dispersions, composition, forest plots) | {api}`cytepool.plotting` | limma 3.62.1, edgeR 4.4.0, DESeq2 1.46.0, speckle 1.6.0, metafor 5.0.1 | exact: 1e-10 for what is drawn; close: 1e-6 where an iterative fit enters, as for sample quality weights, plotSA's prior and REML | [Figures of limma, edgeR, DESeq2, speckle and metafor](#validation-plotting) |
| Principal components of samples (DESeq2's `plotPCA`), of a matrix or a cell type of a cube, and their figure | {api}`principal_components`, {api}`plot_pca` | DESeq2 1.46.0, edgeR 4.4.0 | exact: 1e-10, up to the sign of each component | [Principal components and dot plots](#validation-pca-and-dot-plots) |
| Dot plots of the fraction of cells that express each gene and their mean expression, per cell type, from a cube (scanpy's `dotplot`) | {api}`plot_dotplot` | scanpy 1.12.4 | exact: 1e-10 | [Principal components and dot plots](#validation-pca-and-dot-plots) |
| Tails and quantiles of distributions, special functions | {api}`cytepool.stats.distributions`, {api}`cytepool.stats.special` | mpmath 1.3.0 | exact: 1e-15 to 1e-12 | [Distributions and special functions](#validation-distributions) |
| LOWESS, natural spline bases | {api}`lowess`, {api}`natural_spline_basis` | R 4.4.3 | close: 1e-6 for LOWESS; exact: 1e-10 for spline bases | [LOWESS and spline bases](#validation-smoothing) |

In all, the scripts printed 2,397 checks: 2,065 pass, none failed; 186 information, 144
deliberate differences, 2 controls.

Run time is the wall time of the whole script and peak memory the largest resident
set of any of its processes, reference programs included, both measured with GNU
`time` on the machine described under [](#validation-timings). For
`interferon_stimulated_pbmcs.py`, the run time is with its data downloaded already,
and the peak memory is the largest that the script measured itself: it restarts the
peak of its own process before it runs `limma_voom`, which hides the earlier, higher
peak from GNU `time`.

(reading-the-tables)=
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
scaled designs (1e-8, absolute). Differences are relative unless a section says
otherwise. Two scalings recur, for values that can be near zero: a statistic, such as
a t-statistic or a log fold change, is compared as its difference divided by
max(\|reference\|, 1), and a p-value as its difference divided by
max(p, 1e-300) max(1, \|log p\|), which is about the error in log p relative to
\|log p\|. The sections say which values they scale so. Besides **pass**, a printed
check can be:

- **information**: a value reported for context, without a tolerance, such as how far
  limma's approximate standard errors are from exact ones;
- **deliberate difference**: an input where CytePool behaves differently from the
  reference on purpose, usually raising an error where the reference returns values
  without meaning; the row records what each program does;
- **control**: a check that must exceed the tolerance, to show that the comparison
  can tell nearly identical rules apart.

Sections also list deliberate differences that passing checks confirm, such as
formulas that CytePool rejects by design; the summary counts as deliberate
differences only the checks that the scripts print as such. The calibrated checks are
those of `rotation_gene_set_tests.py`, whose tests draw random rotations, of `gsea.py`
where each program draws its own random sets, and of `sumrank.py` for p-values from
random permutations; the other checks compare deterministic results.

## Cubes and files

(validation-cube)=
### Pseudobulk cubes

The script printed 20 checks (19 pass and 1 information) in 12.5 s, with a peak memory
of 926 MiB. [Pseudobulk profiles and cubes](statistics/cubes.md#exact-regrouping) explains
what it checks.

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

(validation-pool-cells)=
### Pooling cells into cubes

The script printed 27 checks (27 pass) in 30.2 s, with a peak memory of 1,725 MiB.
[Pseudobulk profiles and cubes](statistics/cubes.md#cube-layers) explains what it
checks.

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

(validation-cube-files)=
### Cube files

The script printed 7 checks (7 pass) in 3.1 s, with a peak memory of 146 MiB. [Cube
files and provenance records](reference/cube_files.md#cube-file-format) explains what
it checks.

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

(validation-cube-reads)=
### Reads of selected features

The script printed 5 checks (5 pass) in 1 min 46 s, with a peak memory of 500 MiB. [Cube
files and provenance records](reference/cube_files.md#cube-file-reading-and-writing)
explains what it checks.

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

(validation-cube-fingerprints)=
### Cube fingerprints

The script printed 7 checks (7 pass) in 4.4 s, with a peak memory of 122 MiB. [Cube
files and provenance records](reference/cube_files.md#fingerprint-encoding) explains
what it checks.

No other tool computes the fingerprint of a cube, so
[`validation/cube_fingerprints.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_fingerprints.py)
checks it against a second implementation, written from the encoding that [Cube files
and provenance records](#fingerprint-encoding) specifies: a separate program reads cube
files with
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

(validation-anndata-conversion)=
### Conversions to AnnData and MuData

The script printed 18 checks (18 pass) in 1 min 5 s, with a peak memory of 1,483 MiB.
[Pseudobulk profiles and cubes](statistics/cubes.md#profiles-from-other-tools) explains
what it checks, and [Work with AnnData and MuData](guides/anndata_and_mudata.ipynb)
shows the conversions.

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

## Designs and linear models

(validation-design)=
### Designs and contrasts

The script printed 27 checks (26 pass and 1 information) in 4.3 s, with a peak memory of
221 MiB. [Designs, linear models and
contrasts](statistics/linear_models.md#designs-and-formulas) explains what it checks.

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

(validation-linear-models)=
### Linear models

The script printed 218 checks (200 pass and 18 information) in 4.5 s, with a peak memory
of 177 MiB. [Designs, linear models and
contrasts](statistics/linear_models.md#one-model-per-gene) explains what it checks.

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
  when one factorization of the design serves every gene (no missing values, and no
  weights or only positive sample weights) and for contrasts of uncorrelated coefficients,
  such as group means; otherwise it is approximate, and here its values ranged from
  0.56 to 2.55 times the exact ones. `method="limma"` reproduces it.

With `--benchmark`, the script times both packages on 20,000 genes x 200 samples; see
[](#validation-timings).

(validation-empirical-bayes)=
### Empirical Bayes moderation

The script printed 150 checks (146 pass and 4 deliberate differences) in 4.8 s, with a
peak memory of 195 MiB. [Empirical Bayes
moderation](statistics/empirical_bayes.md#empirical-bayes-model) explains what it
checks.

[`validation/empirical_bayes.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/empirical_bayes.py)
compares `moderated_statistics`, `treat`, `squeeze_variances`, `fit_f_distribution`,
`fit_f_distribution_robust` and `fit_f_distribution_likelihood` with limma's `eBayes`,
`treat`, `squeezeVar`, `fitFDist`, `fitFDistRobustly` and `fitFDistUnequalDF1`, with
limma's `legacy` argument TRUE (the method of moments, `prior_estimator="moments"`),
FALSE (the likelihood estimator, `"likelihood"`) or NULL (limma's default,
`prior_estimator=None`, which takes the likelihood estimator when the positive residual
degrees of freedom differ; genes without residual degrees of freedom do not count). Linear model cases (2,000 genes, 6 to 40 samples) are fitted by `lmFit`
in R and `fit_linear_model` here, then moderated with a constant prior, a trend,
robustly, and with both; they include several contrasts, some linearly dependent,
hypervariable genes, an infinite prior df, missing values, observation weights (some of
them zero), genes without residual df, an all-zero contrast, and 1 to 25 genes, which
with one or two genes are not moderated (some of them all zero). Variance cases call
the estimators directly on 2 to 6,000 variances, with and without a covariate, few
distinct or infinite covariate values, zero, missing and unusable variances, fractional
df from 0.05, prior weights, planted outliers with equal and unequal df, variances that
vary less or more than any prior df explains, infinite robust prior df, three to five
trended variances, two variances above zero among zeros, and covariates on scales of
1e-5 and 1e-9. The weighted LOWESS of the likelihood estimator's trend is also compared
with limma's `weightedLowess`. TREAT tests a fold change of 1.2. Reference: R 4.4.3,
limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| t, p, B, F and its p-value, priors, posterior variances, total df and `var.prior`, with constant and trend priors, `legacy = TRUE` | `eBayes`, `treat` | 6.8e-13 | 1e-10 | 16 pass |
| The same with robust priors | `eBayes`, `treat` | 5.1e-8 | 1e-6 | 7 pass |
| One or two genes, not moderated, some of them all zero: NaN t-statistics and an infinite `var.prior` where limma gives them, with every option set | `eBayes`, `treat` | 1.3e-14 | 1e-10, robust 1e-6 | 12 pass |
| Missing values and observation weights, constant and trend priors, `legacy = TRUE` | `eBayes`, `treat`; the exact F against limma's F of group means | 1.8e-13 | 1e-10 | 3 pass |
| Missing values and observation weights, robust priors, `legacy = TRUE` | as above | 3.0e-11 | 1e-6 | 3 pass |
| Missing values, zero observation weights and genes without residual df, by default (the likelihood estimator), constant and trend priors | `eBayes`, `treat`, `legacy = NULL` | 1.4e-12 | 1e-6 | 6 pass |
| The same with robust priors, every gene with residual df | as above | 2.1e-12 | 1e-6 | 5 pass |
| The likelihood estimator on equal residual df | `eBayes`, `treat`, `legacy = FALSE` | 2.8e-9 | 1e-6 | 3 pass |
| Every component of `squeezeVar` and `fitFDist` | `squeezeVar`, `fitFDist` | 3.1e-13 | 1e-10 | 24 pass |
| Every component of `fitFDistRobustly`, with two choices of Winsorized tails | `squeezeVar`, `fitFDistRobustly` | 8.4e-8 | 1e-6 | 20 pass |
| Every component of `fitFDistUnequalDF1` and `squeezeVar`, with a constant prior and a trend, prior weights, fractional df, zero, missing and unusable variances | `squeezeVar`, `fitFDistUnequalDF1`, `legacy = FALSE` | 2.7e-12 | 1e-6 | 18 pass |
| The same with `robust = TRUE`, every variance usable and no prior weights | as above | 4.6e-12 | 1e-6 | 8 pass |
| Trends through three to five variances (a weighted straight line through at most four), two variances above zero among zeros (a constant prior, not robust), and covariates on scales of 1e-5 and 1e-9; constant and robust | as above | 2.6e-13 | 1e-6 | 13 pass |
| The weighted LOWESS of the trend: ties, windows narrower than 1e-7, windows whose x barely vary, two distant clusters, the trend's weights | `weightedLowess` | 1.1e-15 | 1e-10 | 5 pass |
| Robust likelihood fits with unusable variances or genes without residual df, against limma's fits of the usable features alone (the controls of the deliberate difference below) | `fitFDistUnequalDF1`, `eBayes` | 1.4e-12 | 1e-6 | 3 pass |
| Robust likelihood fits with genes without residual df, unusable variances or prior weights | `eBayes`, `squeezeVar`, `fitFDistUnequalDF1` | 0.39 (B-statistics), 0.086 (prior variance) | n/a | 4 deliberate differences |

Each row gives the largest difference over all outputs of a case. Differences are
relative, except for t, B, F and outlier probabilities and p-values, scaled as
[above](#reading-the-tables). NaN and infinite values must be in the
same places, and a prior given as one value must be one value in both. The robust
moment estimator is in the close tier because limma solves its Winsorized moment
equation only to a tolerance. The likelihood estimator is in the close tier too: both
programs search the prior df by the same steps of Brent's method, which stop at a
tolerance, and agree to about 1e-12; the largest difference, 2.8e-9, is where the
likelihood barely changes near the largest prior df searched, so that rounding can
move the steps. Beyond these checks, with residual df of 1e7 and more limma's
likelihood loses precision: its prior df differed from CytePool's by up to 1e-5 at 1e7
and 5e-2 at 1e12, where CytePool's agreed with the likelihood computed to 60 digits.
Deliberate differences:

- The robust likelihood estimator leaves the genes without a usable variance out of its
  test for outliers. limma 3.62.1 counts them there, a missing variance as a zero one,
  which becomes an outlier and lowers the adjusted tail probabilities of the other
  genes. With 40 of 2,000 genes without residual df, limma's robust prior df were 11.5%
  larger with a constant prior, where limma also gave 45 genes smaller prior df as
  outliers and CytePool gives every gene the same, and 6% to 10% larger with a trend.
  The control rows show that without a trend CytePool's fit equals limma's of the usable
  features alone. `fit_f_distribution_likelihood` also keeps prior weights in its second
  robust fit, which limma's drops.
- `fit_f_distribution_likelihood` raises an error where fewer than two usable variances
  are above zero, where limma returns NA. With exactly two above zero both fit a
  constant prior, but limma returns NaN when some other variances are not usable,
  which CytePool leaves out. It moves infinite covariates one unit beyond the finite
  range, as `fit_f_distribution` and limma's `fitFDist` do, where `fitFDistUnequalDF1`
  returns NaN.
- The moderated F-statistic of a fit with missing values or observation weights is
  exact by default; `f_statistic="limma"` gives limma's approximation, which the
  `eBayes` rows compare. The exact F has no limma counterpart, so it is checked
  against limma's F of the same data coded as group means, whose coefficients are
  uncorrelated for every gene.
- `treat` reports a t-statistic of NaN for a coefficient that is NaN, and for an
  all-zero contrast tested with a zero threshold, where limma reports 0.

(validation-multiple-testing)=
### Multiple testing

The script printed 15 checks (15 pass) in 2 min 56 s, with a peak memory of 202 MiB.
[Multiple testing](statistics/multiple_testing.md#error-rates) explains what it checks.

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

(validation-normalization)=
### Normalization and filtering

The script printed 50 checks (46 pass, 3 deliberate differences and 1 control) in 72.9
s, with a peak memory of 745 MiB. [Normalization and
filtering](statistics/normalization_and_filtering.md#normalization-and-size-factors)
explains what it checks.

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

(validation-voom)=
### Precision weights for counts

The script printed 118 checks (111 pass and 7 deliberate differences) in 15.3 s, with a
peak memory of 310 MiB. [limma-voom and limma-trend](statistics/limma_voom.md#voom)
explains what it checks.

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

Differences are relative, except for E, the trend's x, t and p-values, scaled as
[above](#reading-the-tables), and the trend's y, scaled by max(\|reference\|, median of the
reference), because near zero these values are rounding noise. E was identical to limma's to the last bit in 26 of
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

(validation-sample-quality-weights)=
### Sample quality weights and prior weights

The script printed 89 checks (83 pass and 6 deliberate differences) in 12.1 s, with a
peak memory of 206 MiB. [limma-voom and
limma-trend](statistics/limma_voom.md#sample-quality-weights) explains what it checks.

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

Differences are relative, except for E, the trend's x, t and p-values, scaled as
[above](#reading-the-tables), and the trend's y, scaled by max(\|reference\|, its median). The
sample weights are iterative
estimates, in the close tier, as is everything computed from them. They agree to
rounding because CytePool repeats limma's iterations and stopping rule, found by
running limma as a black box: the starting point, the prior's terms in the score and
the information, the exact REML information of Smyth (2002), and the criterion that
stops the iterations, including the early stop where it rises. The weights that limma's `voom` returns with prior weights are
the trend's alone; CytePool's are those times the prior weights, which the comparison
multiplies in. `eBayes` runs with its default estimator of the prior, which
CytePool's default follows: where zero weights make the residual degrees of freedom
unequal, that is the likelihood estimator. Deliberate differences:

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

(validation-differential-expression)=
### limma-voom and limma-trend on cubes

The script printed 32 checks (28 pass and 4 deliberate differences) in 1.8 s, with a
peak memory of 189 MiB. [limma-voom and
limma-trend](statistics/limma_voom.md#limma-voom-steps) explains what it checks.

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
| logFC, AveExpr, t, P.Value, adj.P.Val and B of every contrast both programs compute, and the moderated standard error of logFC (`sqrt(s2.post) * stdev.unscaled`) | `topTable`, `eBayes` | 6.3e-13 | 1e-10 | 9 pass |
| The same with robust moderation | `topTable` | 1.0e-13 | 1e-6 | 3 pass |
| `D2_vs_D6` in the NK cells, which limma refuses, against limma's values for `D2_vs_D1`, which there estimate the same difference | `topTable` | 2.8e-13 | 1e-10 | 2 pass |
| DC cells, without residual degrees of freedom: both programs refuse every contrast | `eBayes` | both refuse | n/a | 2 pass |
| Deliberate differences, listed below | `contrasts.fit` | 2 contrasts limma refuses and CytePool computes, 2 limma computes and CytePool reports as not estimable | n/a | 4 deliberate differences |

Differences are scaled as [above](#reading-the-tables), values as statistics and p-values as
p-values. limma-voom ran with
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

(validation-differential-expression-options)=
### Options for pseudobulk differential expression

The script printed 66 checks (66 pass) in 5.0 s, with a peak memory of 182 MiB.
[limma-voom and limma-trend](statistics/limma_voom.md#limma-voom-defaults) explains what
it checks.

[`validation/differential_expression_options.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression_options.py)
checks the options of `limma_voom` and `limma_trend` for pseudobulk data against
limma and edgeR. It simulates cells from 6 donors, each sampled before and after a
stimulation (seed 20261027), in 3 cell types with 1,000 genes, pools them with
`from_cells`, and fits `~ donor + condition` with two contrasts, `stim_vs_ctrl` and
`D3_vs_D2`. One sample is an outlier in every cell type, two are sequenced shallowly
in one cell type each, and in the NK cells both samples of donor D1, the reference
level, have fewer than 10 cells, which makes the column `donorD6` a combination of the
others. Six analyses each set some of the options, and R runs the workflow of [limma-voom and limma-trend on
cubes](#validation-differential-expression) with the same options:

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

Differences are relative for the weights; the statistics and p-values are scaled as
[above](#reading-the-tables). Everything computed from sample quality weights is in the close tier, as they
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

(validation-interferon-stimulated-pbmcs)=
### Differential expression on published data

The script printed 41 checks (17 pass and 24 information) in 15.2 s, with a peak memory
of 617 MiB. [limma-voom and limma-trend](statistics/limma_voom.md#limma-voom-steps) explains
what it checks.

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

Differences are scaled as [above](#reading-the-tables), values as statistics and p-values as
p-values. The dendritic cells are fitted without
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

(validation-negative-binomial-fits)=
### Negative binomial models

The script printed 121 checks (78 pass, 30 information and 13 deliberate differences) in
14.2 s, with a peak memory of 187 MiB. [edgeR and
DESeq2](statistics/count_models.md#negative-binomial-model) explains what it checks.

[`validation/negative_binomial_fits.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/negative_binomial_fits.py)
compares `fit_negative_binomial_model` with edgeR's `glmFit`,
`negative_binomial_deviance` with `nbinomDeviance`, and
`negative_binomial_unit_deviance` with `nbinomUnitDeviance` and with mpmath, on
negative binomial counts simulated from fixed seeds: 2,000 genes and 6 to 16 samples in
15 cases, with genes without counts and groups whose counts are all zero planted in
most. Six are one-way layouts (two groups, group means with TMM factors, an intercept
alone, one coefficient per sample, sample weights, and dispersions, offsets and weights
per observation), which edgeR fits group by group to its last bits. Nine are other
designs: groups and a covariate with trended dispersions, a paired design with TMM
factors, dispersions, offsets and weights per observation, sample weights, the Poisson
model, sparse counts of about one read per gene and sample, a covariate such as a date
next to an intercept, means near 1e8, and a dispersion of 2e-5. edgeR's fits of these
stop short of the maximum, so they are compared with the maximum that Newton's method
with the observed information, written in R for the script, reaches from edgeR's fit,
and with `mglmLevenberg(tol = 1e-14)`, a check where it converges and information where
it does not. The unit deviance is compared at 5,373 points, counts and means from 1e-30
to 1e30 and dispersions from 0 to 1e12, with mpmath at 150 digits. Reference: R 4.4.3,
edgeR 4.4.0, mpmath 1.3.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| One-way layouts: coefficients and fitted values, shrunk coefficients | `glmFit` | 3.0e-11 | 1e-6 | 12 pass |
| One-way layouts: deviance, and `nbinomDeviance` at edgeR's fitted values | `glmFit`, `nbinomDeviance` | 2.3e-7 | 1e-6 | 6 pass |
| Residual df, and fits that did not converge | `glmFit` | 0 | 0 | 15 pass |
| Other designs: coefficients and fitted values | Newton's method in R | 1.8e-7 | 1e-6 | 9 pass |
| Other designs where edgeR converges: coefficients and fitted values, shrunk coefficients | `mglmLevenberg(tol = 1e-14)` | 3.1e-7 | 1e-6 | 8 pass |
| Other designs: deviance at the maximum, and at edgeR's fitted values | `nbinomDeviance` | 3.2e-7 | 1e-6 | 8 pass |
| Genes whose maximum lies at infinity: fitted values that tend to zero, below 1e-6 in both; the others and the deviance | `mglmLevenberg(tol = 1e-14)` | 3.9e-7, 8.5e-7 | 1e-6 | 15 pass |
| Unit deviance | mpmath; `nbinomUnitDeviance` where edgeR is accurate | 7.8e-15; 3.0e-7 | 1e-13; 1e-6 | 2 pass |
| Refused inputs: an aliased column, a zero weight, a negative dispersion | `glmFit` | both raise | n/a | 3 pass |
| Where edgeR stops short: its distance from the maximum | `glmFit`, `mglmLevenberg` | up to 1.6 for `glmFit` (a date), 2.3e-3 for `mglmLevenberg` | n/a | 19 information |
| `glmFit` deviances that are not those of its own fitted values | `glmFit`, `nbinomDeviance` | 5 of 2,000 genes, with dispersions and weights per observation | n/a | 9 information |
| Seconds for 2,000 genes, group means and a paired design | `glmFit` | CytePool 0.007 and 0.059, edgeR 0.008 and 0.062 | n/a | 2 information |
| Deliberate differences, listed below | `glmFit`, `nbinomDeviance`, `nbinomUnitDeviance` | 9 cases of genes without counts; 2 rows for a dispersion of 2e-5; 2 for edgeR's unit deviance | n/a | 13 deliberate differences |

Coefficients are scaled as statistics are [above](#reading-the-tables), fitted values relative to
the reference, and deviances relative to the larger of the reference and the number of
samples. A gene whose maximum lies at infinity has coefficients that depend on where
each program stops, so for such genes only the fitted values that tend to zero (below
1e-6 in both programs), the others and the deviance are compared. edgeR's Levenberg
fits stop up to about 3e-4 from the maximum in `glmFit` (1e-2 with dispersions and
weights per observation, and 1.6 with a date, where fitted values that should tend to
zero stay near 1e2), and about 2e-7 in `mglmLevenberg(tol = 1e-14)`, but 2e-3 with
dispersions and weights per observation, 2e-6 with sparse counts, 1e-5 with a date or
large counts, and 2e-3 with a dispersion of 2e-5. Deliberate differences:

- A gene without counts, in a design with an intercept: edgeR's Levenberg fit returns
  NA coefficients; CytePool returns those of the log mean -1e8, as edgeR's one-way fit
  does, with fitted values and deviance of zero.
- The deviance is exact. edgeR adds 1e-8 to each count and mean, which lowers the unit
  deviance of a zero count by up to about 5e-7. Below a dispersion of 1e-4 it
  approximates the unit deviance, with an error that grows with the dispersion times
  the mean, from about 1e-6 at 1e-3 to factors from about 1; with the dispersion of
  2e-5, its deviances, and the fits of genes whose maximum lies at infinity, differ by
  up to 0.97 and 1.2e-3. From a dispersion times the mean of 1e6 it uses the limit of
  the gamma distribution, which gives zero counts 1.6 to 2.6 times their deviance.

(validation-quasi-likelihood)=
### Quasi-likelihood F-tests

The script printed 223 checks (210 pass and 13 deliberate differences) in 5.1 s, with a
peak memory of 186 MiB. [edgeR and DESeq2](statistics/count_models.md#quasi-likelihood)
explains what it checks.

[`validation/quasi_likelihood.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/quasi_likelihood.py)
compares `fit_quasi_likelihood` and `quasi_likelihood_f_test` with edgeR's `glmQLFit`
and `glmQLFTest` (edgeR 4.4's default, `legacy = FALSE`), `common_dispersion` with
`estimateGLMCommonDisp` and `dispCoxReid`, `average_log2_cpm` with `aveLogCPM`, and
`negative_binomial_unit_deviance_moments` with sums at 30 digits in mpmath and the
gamma limit. The counts are simulated from fixed seeds, with a dispersion for each
gene, in 14 cases of 1,000 genes and 8 or 9 samples, with genes without counts and
groups whose counts are all zero planted in most: two groups with TMM factors; three
groups and a covariate with several contrasts and a joint test; a paired design;
robust priors with and without genes that have no counts; small counts; a dispersion
given below the true one, so that the average quasi-dispersion is well above 1; prior
weights; trended dispersions; no trend; a trend in another covariate; offsets per
observation; Poisson counts; and a test that both programs refuse. R runs with its
address space capped at 8 GB. Reference: R 4.4.3, edgeR 4.4.0, limma 3.62.1, mpmath
1.3.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Share of genes the dispersion comes from | `glmQLFit` | 0 | 0 | 9 pass |
| NB dispersion | `glmQLFit`; `dispCoxReid(tol = 1e-12)` on the same genes | 8.7e-6; 8.9e-7 (6.5e-6 for designs that are not one-way layouts) | 1e-4; 1e-6 (1e-5) | 18 pass |
| Common dispersion, with prior weights and a systematic subset | `estimateGLMCommonDisp`; `dispCoxReid(tol = 1e-12)` | 8.1e-6; 4.6e-7 | 1e-4; 1e-6 | 5 pass |
| Average log CPM in four settings | `aveLogCPM` | 3.3e-12 | 1e-6 | 1 pass |
| Moments of the unit deviance at 41 points, and at a mean of 1e14 | mpmath at 30 digits; the gamma limit | 1.7e-14 | 1e-11 | 2 pass |
| Average quasi-dispersion | `glmQLFit` | 2.1e-3 | 1e-2 | 14 pass |
| Adjusted deviances and residual df | `glmQLFit` | 3.3e-3 | 5e-3 | 14 pass |
| Prior and posterior quasi-dispersions; prior df | `glmQLFit` | 1.4e-3 where the average quasi-dispersions agree, 5.6e-3 where they differ; 9.8e-3 | 5e-3, 1e-2; 2e-2 | 26 pass |
| The prior and posterior that `squeeze_variances` gives edgeR's own adjusted deviances | `glmQLFit`, `squeezeVar` | 4.7e-8 | 1e-6 | 13 pass |
| Leverages, those of fitted values that tend to zero included | `glmQLFit` | 2.8e-7 for one-way layouts at the same working dispersion; 8.7e-5 | 1e-6; 1e-3 | 14 pass |
| logCPM | `glmQLFTest` | 2.8e-12 at a given dispersion; 2.7e-7 at an estimated one | 1e-6; 1e-5 | 16 pass |
| logFC | `glmQLFTest` | 4.7e-7 for one-way layouts at the same working dispersion; 2.0e-4 | 1e-6; 1e-3 | 16 pass |
| Likelihood ratio statistic (F times the posterior quasi-dispersion and its df) | `glmQLFTest` | 8.5e-10 at the same dispersion; 8.9e-6 at dispersions estimated by both; 4.5e-3 where the averages differ | 1e-6; 1e-4; 1e-2 | 16 pass |
| F; PValue and FDR; df.total | `glmQLFTest`, `topTags` | 2.6e-3; 3.9e-3 of \|log10 p\|; 9.8e-3 | 5e-3; 5e-3; 2e-2 | 45 pass |
| A test that leaves the null model without a coefficient | `glmQLFTest` | both raise | n/a | 1 pass |
| Deliberate differences, listed below | `glmQLFit`, `glmQLFTest`, `squeezeVar` | 6 rows of the robust prior with genes without counts; the first gene's logCPM with offsets per observation; 6 rows of edgeR's moments | n/a | 13 deliberate differences |

Three differences set the tolerances, and the script isolates each. edgeR approximates
the mean and variance of the unit deviance: the scale s implied by its unit matrices
is off by up to 7.6e-4, 1.2e-3 and 1.3e-2 at dispersions of 0.01, 0.2 and 1, and the
degrees of freedom k by up to 2.0e-4, 1.3e-4 and 4.0e-3, where CytePool's sums are
exact. The adjusted deviances, quasi-dispersions and F-statistics therefore agree to
about 3e-3, the average quasi-dispersion to 2.1e-3, and the prior df, which maximizes
a likelihood that is flat at large values, to 9.8e-3. The prior and posterior that
`squeeze_variances` computes from edgeR's own adjusted deviances agree with edgeR's to
4.7e-8, so the moments are the only cause. Where the average quasi-dispersions differ,
so do the working dispersions of the fits, and with them the quasi-dispersions,
leverages, log fold changes and likelihood ratios, by about 1e-4 to 6e-3. edgeR's
search for the Cox-Reid dispersion stops within its tolerance of 1e-5, and
`dispCoxReid(tol = 1e-12)` on the same genes agrees with CytePool to 8.9e-7. edgeR's
Levenberg fits stop short of the maximum (see [Negative binomial models](#validation-negative-binomial-fits)), so log
fold changes of designs that are not one-way layouts agree to about 2e-4, and their
fitted values that tend to zero stop between about 1e-8 and 1e-7, far above
CytePool's; both report the
leverages of the limit for these. Deliberate differences:

- The robust prior with genes that have no counts: limma counts their missing
  quasi-dispersions as zero variances in its tests for outliers, which lowers the
  outlier probabilities of the others and, here, changes which genes are outliers;
  CytePool tests only the genes with residual degrees of freedom, as its
  `fit_f_distribution_likelihood` does (see [Empirical Bayes moderation](#validation-empirical-bayes)). Without
  such genes the two agree.
- With offsets per observation, edgeR's `aveLogCPM` gives the first gene a value that
  does not depend on its offsets, minus infinity for a gene without counts. The case
  gives the first gene no counts, a dispersion and no trend, so that the value enters
  nothing else, and every other gene is compared.
- The moments of the unit deviance, as above: CytePool's are exact.

(validation-deseq2-tests)=
### DESeq2's dispersions and tests

The script printed 45 checks (28 pass, 11 information and 6 deliberate differences) in 2
min 16 s, with a peak memory of 814 MiB. [edgeR and
DESeq2](statistics/count_models.md#deseq2) explains what it checks.

[`validation/deseq2_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/deseq2_tests.py)
compares `deseq2_dispersions`, `fit_deseq2`, `deseq2_wald_test` and
`deseq2_likelihood_ratio_test` with DESeq2's `estimateDispersions`, `nbinomWaldTest`,
`nbinomLRT` and `results`, and with PyDESeq2, on negative binomial counts simulated
from fixed seeds: 600 genes and 4 to 12 samples in 9 cases, with 2% of the genes
without counts. The cases are two groups; four groups, tested by the likelihood ratio
test of the factor and by a Wald test of two levels that 15 genes have no counts in;
two groups and a covariate; a paired design of six donors; small counts; two samples
per group, which leave two residual degrees of freedom; an outlier sample; dispersions
that rise with the mean; and nearly Poisson counts, 69% of whose gene-wise estimates
are at the lower bound. Counts with an outlier in most genes, at a dispersion of 0.05,
test the rules by which Cook's distance removes p-values under four designs: two groups
of four, and groups of two and five as a factor, as means without an intercept and as
a numeric covariate. DESeq2 runs with its defaults, except that it does not replace outlier counts:
`DESeq(dds, minReplicatesForReplace = Inf)` (the default is 7), with `fitType = "mean"` where its parametric trend
fails (the small counts and the rising dispersions), and each stage of CytePool is
compared given DESeq2's own inputs to it: the dispersions with their maxima, found in R
by the script from a grid of the Cox-Reid adjusted log-likelihood and the root of its
derivative; the trend and the prior with DESeq2's; the fits with DESeq2's at
`betaTol = 1e-14`, polished by Newton's method in R; and the tests with `results()` at
those fits, with its defaults, with `cooksCutoff = 0.5` and without independent
filtering. Each printed row pools the cases it applies to and names the worst.
Reference: R 4.4.3, DESeq2 1.46.0, PyDESeq2 0.5.4, mpmath 1.3.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Design matrices; size factors, means and variances of the normalized counts | `model.matrix`; DESeq2 | 0; 5.9e-16 | 0; 1e-10 | 2 pass |
| Means of the likelihood | `estimateDispersionsGeneEst`; for the covariate and paired designs, DESeq2's fit at the rough dispersions, polished | 3.6e-14 | 1e-6 | 1 pass |
| Gene-wise dispersions of at least 1e-6 | maxima in R | 3.8e-9 | 1e-6 | 1 pass |
| Gene-wise dispersions below 1e-6 (49 genes), and genes whose rough dispersion is at the lower bound (1,935), which keep 1e-8 as in DESeq2 | maxima in R; DESeq2 | 0 genes differ | 0 | 2 pass |
| The parametric trend from DESeq2's gene-wise estimates (7 cases); its failure where DESeq2's fails, and the mean trend there (2 cases) | DESeq2 | 1.4e-15; 0 cases differ; 0 | 1e-10; 0; 1e-10 | 3 pass |
| Variance of the log residuals, and of the prior with more than three residual degrees of freedom | DESeq2 | 1.3e-16 | 1e-10 | 2 pass |
| Variance of the prior with two residual degrees of freedom, maximum likelihood | integration on a grid | 2.3e-8 | 1e-6 | 1 pass |
| Maximum a posteriori dispersions | maxima in R | 2.5e-12 | 1e-6 | 1 pass |
| Dispersion outliers (33 genes), and the final dispersions they choose | DESeq2 | 0 genes differ | 0 | 1 pass |
| Coefficients and standard errors | DESeq2 at `betaTol = 1e-14`, polished in R | 3.2e-15 | 1e-6 | 1 pass |
| Deviances | DESeq2 at `betaTol = 1e-14` | 1.2e-11 | 1e-9 | 1 pass |
| Fitted values, hat values, Cook's distances; the log2 fold changes and standard errors of every test | DESeq2 at `betaTol = 1e-14` | 2.2e-6; 1.4e-6 | 1e-5 | 2 pass |
| Wald tests (9): statistics, p-values, adjusted p-values | `results()` at `betaTol = 1e-14` | 1.8e-6 | 1e-5 | 1 pass |
| Likelihood ratio tests (3): statistics, p-values, adjusted p-values | `results()` at `betaTol = 1e-14` | 3.2e-11 | 1e-9 | 1 pass |
| Genes flagged by Cook's distance (27 genes), filtered out, and the filter's quantile, in every test; genes flagged at `cooksCutoff = 0.5` (862), and the adjusted p-values there and without independent filtering | `results()` at `betaTol = 1e-14` | 0 differences; 1.9e-6 | 0; 1e-5 | 3 pass |
| Independent filtering on 20 statistics, some of whose curves of rejections reach the fallbacks to 90% and 80% of their fitted maximum | `results(filter = ...)` | 0 | 0; 1e-10 | 2 pass |
| Cook's distances at given dispersions under four designs: genes flagged (79 genes, 9 more above the cutoff but spared in a comparison of two groups), and the largest distances | DESeq2 at `betaTol = 1e-14` | 0 genes differ; 4.0e-14 | 0; 1e-5 | 2 pass |
| Derivative in log(alpha) of a count's log-likelihood, 192 points | mpmath, 50 digits | 2.7e-15 | 1e-12 | 1 pass |
| DESeq2's line searches, its fits at its defaults, its means of the likelihood for the covariate and paired designs, its trend where the parametric fit fails, and its prior with two residual degrees of freedom | DESeq2 defaults | listed below | bounds listed below | 6 deliberate differences |
| Genes spared in a comparison of two groups; DESeq2's coefficients at `betaTol = 1e-14`, from the polished fits; the largest differences of DESeq2's gene-wise line searches | DESeq2 | 9 genes; up to 1.5e-6; up to 13 on the log scale | n/a | 3 information |
| The analysis from the counts: final dispersions, p-values, statistics, genes called | DESeq2 defaults | medians 1.2e-3 to 3.8e-2 (dispersions, log scale) and 3.0e-4 to 7.6e-3 (p-values); 7 of 1,297 genes called differently at 0.1 | n/a | 4 information |
| The analysis from the counts | PyDESeq2 | medians up to 5.8e-7 (gene-wise dispersions, log scale), 4.7e-5 to 0.21 (final dispersions) and 1.7e-5 to 4.2e-2 (p-values); 5 of 281 genes called differently | n/a | 4 information |

Dispersions are compared relative to the reference, coefficients and statistics are
scaled as statistics are [above](#reading-the-tables), Cook's and hat values relative
to the larger of the reference and 1, as only values near the cutoff matter, and
p-values are compared on the log scale, relative to max(\|log p\|, 1), as an error e
in a statistic z moves log p by about z^2 e. Below a
gene-wise dispersion of 1e-6 the adjusted likelihood is flat in the log dispersion,
whose derivative is proportional to the dispersion, so there the check is that both
sides place the gene below 1e-6. DESeq2 stops its fits when the deviance changes by
less than `betaTol` of it; for genes of large dispersion, where its Fisher scoring
converges slowly, its coefficients are then up to 1.5e-6 from the fixed point even at
`betaTol = 1e-14` (an information row), which bounds the tolerance of what is computed
from them, but for the deviances and the likelihood ratio statistics, which are
stationary there. Rows that count genes fail when they cover none. Deliberate
differences, each within a bound that the script checks:

- DESeq2 maximizes the dispersions by a line search that stops when an iteration gains
  less than 1e-6, and that can stall where it starts, returning its starting value:
  its gene-wise estimates are a median of up to 5.5e-5 from the maximum on the log
  scale (bound 1e-3), and up to 13 in a few genes, and its maximum a posteriori
  estimates up to 7.7e-3 from theirs, relative (bound 0.1). CytePool computes the
  maxima. A gene whose rough dispersion is at the lower bound keeps it, as in DESeq2,
  even where the maximum lies far above.
- DESeq2's fits at its defaults stop up to 1.9e-3 from the fixed point (log2
  coefficients, scaled as statistics; bound 1e-2), and its means of the
  likelihood for designs that are not one-way layouts, from such a fit at the rough
  dispersions, up to 7.5e-4 (bound 1e-2). CytePool's fits converge.
- Where DESeq2's parametric trend fails, for the small counts and the rising
  dispersions, DESeq2 substitutes a local regression, which CytePool does not have;
  CytePool uses the mean trend, with a warning. From DESeq2's gene-wise estimates,
  CytePool's parametric trend fails where DESeq2's does and agrees with it to rounding
  where it does not: each gamma fit is R's `glm()` iteration, and the genes fitted are
  those whose gene-wise estimate is above 1e-6, as in DESeq2, which the nearly Poisson
  counts test.
- With two residual degrees of freedom DESeq2 matches densities of simulated
  residuals over a grid, and CytePool maximizes the likelihood of the same model:
  9.5% apart here (bound 0.2).

From the counts, the final dispersions differ from DESeq2's by a median of 1.2e-3 to
3.8e-2 on the log scale (the largest with two residual degrees of freedom, from the
prior), p-values by a median of 3.0e-4 to 7.6e-3, relative, and statistics by up to
3.4e-2 relative, 7 of 1,297 genes being called differently at a false discovery rate
of 0.1. PyDESeq2's gene-wise dispersions agree with CytePool's to medians of up to
5.8e-7 on the log scale; its final dispersions differ by medians of 4.7e-5 to 0.21, the
largest for the small counts, where DESeq2's and CytePool's trend is the mean of the
gene-wise estimates.

(validation-mixed-models)=
### Mixed models

The script printed 135 checks (102 pass, 29 information and 4 deliberate differences) in
5 min 28 s, with a peak memory of 330 MiB. [Mixed models and
dream](statistics/mixed_models.md#mixed-model-reml) explains what it checks.

[`validation/mixed_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/mixed_models.py)
compares `fit_mixed_model` with lme4's `lmer` (REML) and `fixed_effect_tests` with
lmerTest's Satterthwaite t-tests (`summary` for every coefficient, `contest1D` for a
contrast), gene by gene, in 9 cases from fixed seeds: one random intercept (8 donors x
3 samples, a condition within donors and an age per donor); unbalanced groups (12
donors with 1 to 6 samples); two crossed random intercepts (10 donors x 4 samples over
5 batches); two nested random intercepts (`(1 | donor/sample)`, 6 donors x 3 samples x
2 replicates, with sample labels that repeat across donors); a design simulated
without donor variance, so that most fits lie on the boundary; observation weights
like voom's, with one and with two random intercepts; the crossed design with random
intercepts up to 30 times the residual standard deviation; and 2,000 genes of 16
donors x 4 samples with observation weights. The other cases have 100 genes each. The
standard deviation of each random intercept is drawn per gene, and is zero for a fifth
of the genes. Reference: R 4.4.3, lme4 2.0.6, lmerTest 3.2.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| REML criterion at the optimum | lme4 `REMLcrit` | 3.1e-13 | 1e-10 | 9 pass |
| Genes whose criterion is above lme4's by more than 1e-10, relative | lme4 `REMLcrit` | 0 genes | 0 | 9 pass |
| Coefficients, relative to the larger of their absolute value and standard error, and standard errors | lme4 `fixef`, `vcov` | 2.3e-7 | 1e-6 | 8 pass |
| Variance components, relative to the larger of themselves and the residual variance, and the residual variance | lme4 `VarCorr`, `sigma` | 1.7e-7 | 1e-6 | 8 pass |
| The same two families where the variance ratios are large | lme4 `fixef`, `vcov`, `VarCorr`, `sigma` | 1.1e-6 and 2.3e-6 | 1e-5 | 2 pass |
| The theta of the gene with large ratios that differs most from lme4 | the optimum with 30 digits (mpmath) | 3.4e-13 | 1e-10 | 1 pass |
| Genes where lme4 puts a variance at zero and CytePool does not | lme4 `getME(fit, "theta")` | 0 genes | 0 | 9 pass |
| REML criterion at CytePool's estimates | lme4 `REMLcrit` | 3.6e-13 | 1e-10 | 9 pass |
| Coefficients, standard errors and t-statistics (relative to max(\|t\|, 1)) at CytePool's estimates | lme4 `fixef`, `vcov`; lmerTest `summary` | 1.6e-11 | 1e-10 | 9 pass |
| Satterthwaite df at CytePool's estimates | lmerTest `summary` | 1.3e-6 | 1e-5 | 9 pass |
| p-values at CytePool's estimates | lmerTest `summary` | 7.3e-7 | 1e-5 | 9 pass |
| A contrast: estimate, standard error and t-statistic | lmerTest `contest1D` | 1.6e-11 | 1e-10 | 9 pass |
| A contrast: df and p-value | lmerTest `contest1D` | 6.6e-7 | 1e-5 | 9 pass |
| df of the genes that differ most from lmerTest, with one and with two random intercepts | the definition with 30 digits (mpmath) | 9.7e-16 and 2.2e-11 | 1e-10 | 2 pass |
| Fits with a variance at zero; Newton iterations; lme4's largest theta where CytePool's variance is zero | CytePool; lme4 `getME` | 12% to 54% of the genes of a case; 3.4 to 6.6 on average, at most 9; 4.4e-7 | n/a | 9 information |
| Satterthwaite df at lme4's own optimum | lmerTest `summary` | 3.5e-6 | n/a | 9 information |
| lme4 with its default tolerances: variance components; df | lme4 `VarCorr`; lmerTest `summary` | 1.0e-3; 8.5e-4 | n/a | 8 information |
| lmerTest's df and lme4's theta at the genes checked with 30 digits | the definition with 30 digits (mpmath) | 1.3e-6 and 4.9e-7; 1.3e-6 | n/a | 3 information |
| Deliberate differences, listed below | lme4 `lmer` | 3 inputs lme4 fits and CytePool refuses, 1 that lme4 fits with a warning and CytePool leaves unfitted | n/a | 4 deliberate differences |

The script keeps two questions apart. The optimum is compared with lme4 fitted from its
own start, with the tolerances of its default optimizer (BOBYQA from nloptr) tightened
to `xtol_abs = xtol_rel = 1e-14` and `ftol_abs = ftol_rel = 1e-16`; with
`xtol_abs = 1e-12` and `ftol_abs = 1e-14` alone, some of lme4's fits stopped with a
criterion 3e-7 above the minimum. The computations at given estimates are compared
with lme4 and lmerTest evaluated once at CytePool's estimates, which the script passes
as lme4's starting values with a single evaluation allowed (`maxeval = 1`, and
`restart_edge = FALSE` so that a variance of zero stays zero). There the criterion,
the coefficients, their standard errors and the t-statistics are closed forms, while
lmerTest computes Satterthwaite's degrees of freedom from numerical derivatives.
p-values are scaled as [above](#reading-the-tables).

Two bounds are looser than the tiers. The REML deviance computed with 30 significant
digits (mpmath, from dense matrices) shows which program is exact in each:

- Satterthwaite's degrees of freedom and the p-values that use them, 1e-5. lmerTest
  differentiates the deviance numerically in theta, on which it depends only through
  theta^2, so at an interior optimum its curvature in theta is about 4 theta^2 times
  that in theta^2, and the numerical Hessian loses relative precision like 1 / theta^2
  where a variance is small. At the genes whose df differ most from lmerTest's, with
  one and with two random intercepts, CytePool's df agree with the definition to
  9.7e-16 and 2.2e-11, and lmerTest's to 1.3e-6 and 4.9e-7. The second gene has large
  variance ratios, where CytePool's factorization of the second grouping factor loses
  digits (see the module's documentation).
- The estimates against lme4's own optimum where the variance ratios are large, 1e-5.
  There the criterion is nearly flat in theta, and lme4's optimizer, which stops on
  changes in the criterion, places theta to about 1e-6 only: at the gene that differs
  most, CytePool's theta agree with the optimum computed with 30 digits to 3.4e-13,
  and lme4's to 1.3e-6.

The information rows count the fits with a variance at zero, which `fit_mixed_model`
reports in `boundary` and lme4 approaches without always reaching (its theta there was
at most 4.4e-7), and the Newton iterations per fit. They also give the df at lme4's
own optimum, which inherit its precision where a variance is small, and the fits of
lme4 with its default tolerances, whose variance components were up to 1.0e-3 of the
residual variance from the optimum and whose df up to 8.5e-4 from those at the
optimum.

Deliberate differences:

- A zero weight is refused, as is any weight that is negative or not finite. lme4
  fits the data, and its REML criterion is then infinite.
- A missing value is refused: this version fits complete data only. lme4 drops the
  sample (its default `na.action`), here fitting 23 of 24 samples.
- Values constant within every donor, which the fixed and random effects reproduce
  exactly, have no REML optimum: the criterion decreases without bound as the
  residual variance goes to zero. `fit_mixed_model` returns such a feature unfitted
  (NaN, not converged). lme4 warns that the model failed to converge or is nearly
  unidentifiable, and returns a residual standard deviation of 1.1e-5 times that of
  the values.
- A design whose fixed and random effects together span every sample (6 samples, 3
  donors and 3 batches in a cycle, and a covariate) reproduces every feature
  exactly, and `fit_mixed_model` refuses it. Its criterion can nonetheless have a
  minimum, and lme4 fits it (REML criterion 14.62).

With `--benchmark`, the script times both packages on 20,000 genes x 120 samples; see
[](#validation-timings).

(validation-kenward-roger-and-f-tests)=
### Kenward-Roger's tests and F-tests

The script printed 143 checks (115 pass, 16 information and 12 deliberate differences)
in 3 min 59 s, with a peak memory of 390 MiB. [Mixed models and
dream](statistics/mixed_models.md#kenward-roger) explains what it checks.

[`validation/kenward_roger_and_f_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/kenward_roger_and_f_tests.py)
compares `fixed_effect_tests(df_method="kenward_roger")` with lmerTest's `summary` and
`contest1D` (`ddf = "Kenward-Roger"`) and pbkrtest's `vcovAdj`, and
`fixed_effect_f_test` with pbkrtest's `KRmodcomp` and lmerTest's `contestMD` (by
Kenward-Roger and by Satterthwaite), gene by gene, in 9 cases of 24 genes from fixed
seeds: a time course of three time points within 6 donors with an age per donor, with
the F-tests of two contrasts and of three that span two dimensions; unbalanced donors
with 1 to 5 samples; crossed donor and batch intercepts; nested intercepts
(`(1 | donor/sample)`); a design simulated without donor variance, so that 15 of the 24
fits lie on the boundary; and observation weights like voom's, with one and with two
random intercepts, each taken both ways: every observation at the average weight,
against lme4 fits whose weights are divided by their mean, as dream fits them, and the
precision weights, against pbkrtest on the same model without weights, its values,
fixed effects and random effects multiplied by the square roots of the weights, which is
the weighted model. It then compares dream with variancePartition's `dream` with
`ddf = "Kenward-Roger"`, `"adaptive"` and `"Satterthwaite"`, `eBayes` and `topTable`,
on negative binomial counts of 40 genes in studies of 12, 18, 19, 20 and 24 samples (a
time course of three time points and an age), including `topTable`'s F-test of time,
and `dream_voom` with its defaults on two cubes: cell types left with 12 and with 24
samples, and one of 15 samples whose age, 32 years on average and about 1 apart, makes
the adjusted covariance matrices ill conditioned. Reference: R 4.4.3, variancePartition
1.36.2, limma 3.62.1, lme4 1.1-38, lmerTest 3.1-3, pbkrtest 0.5.3 and minqa 1.2.8.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Adjusted covariance matrix of the coefficients, relative to each gene's largest entry | pbkrtest `vcovAdj` | 1.4e-13 | 1e-10 | 9 pass |
| Kenward-Roger t-tests of every coefficient and a contrast: estimates (relative to the larger of their absolute value and standard error), standard errors and t-statistics (relative to max(\|t\|, 1)) | lmerTest `summary`, `contest1D` | 1.7e-12 | 1e-10 | 9 pass |
| Their df and p-values | lmerTest `summary`, `contest1D` | 1.8e-12 | 1e-10 | 9 pass |
| Kenward-Roger F-tests: F, numerator and denominator df, scaling and p-value, and lmerTest's F and df | pbkrtest `KRmodcomp`, lmerTest `contestMD` | 8.2e-13 | 1e-10 | 13 pass |
| Satterthwaite F-statistics | lmerTest `contestMD` | 8.2e-13 | 1e-10 | 13 pass |
| Satterthwaite F-tests: denominator df and p-values | lmerTest `contestMD` | 1.9e-7 | 1e-5 | 13 pass |
| Kenward-Roger standard errors and df at lme4's own optimum | lmerTest `summary` | 1.2e-7 | 1e-6 | 9 pass |
| dream: precision weights | `voomWithDreamWeights` | 2.1e-8 | 1e-6 | 5 pass |
| dream with each `ddf`: standard errors and df of the tests | `dream` | 8.0e-8 | 1e-6 | 9 pass |
| Moderated t-statistics, B-statistics and z-statistics (relative to max(\|x\|, 1)), total df and p-values | `eBayes`, `topTable` | 5.0e-8 | 1e-6 | 9 pass |
| Moderated F-statistics of time | `topTable` | 4.1e-8 | 1e-6 | 11 pass |
| `dream_voom`: samples used and features tested | the rules of `dream_voom` | 0 differences | 0 | 3 pass |
| `dream_voom` with its defaults: t, B and z-statistics, df and p-values of every contrast | `dream`, `eBayes`, `topTable` | 3.3e-7 | 1e-6 | 3 pass |
| Fits with a variance at zero; the two treatments of the weights; the default against Satterthwaite's method | CytePool | below | n/a | 16 information |
| Deliberate differences, listed below | variancePartition | 12 inputs | n/a | 12 deliberate differences |

At CytePool's estimates, which the script passes to lme4 as starting values with one
evaluation allowed, as for [mixed models](#validation-mixed-models), pbkrtest computes
Kenward-Roger's quantities from dense matrices without numerical derivatives, so
they are compared to the exact tier; lmerTest's Satterthwaite F-tests differentiate the
deviance numerically, compared to 1e-5. At lme4's own optimum, with Powell's BOBYQA and
its trust region shrunk to 1e-10 as in the section on dream, and in dream's fits, which
the script runs with that optimizer, the estimates agree to the close tier. P-values are
scaled as [above](#reading-the-tables).

The checks confirm what probing pbkrtest, lmerTest and variancePartition as black boxes
found, and the documentation of `cytepool.stats.mixed_models` and
`cytepool.stats.dream` lists: pbkrtest takes the inverse of the expected information
of REML, includes a variance component estimated at zero, which 15 fits of the
boundary case have, and builds the covariance matrix of the observations without lme4's
prior weights while it takes the covariance matrix of the coefficients from the
weighted fit; lmerTest's `contest1D` and `summary` use the adjusted standard errors with
pbkrtest's degrees of freedom; `dream` uses Kenward-Roger's method with the adjusted
standard errors for 19 samples and Satterthwaite's for 20 under `ddf = "adaptive"`, and
`topTable`'s F-statistic is limma's, from the moderated t-statistics and each gene's
covariance matrix of the coefficients, the adjusted one under Kenward-Roger's
method. In the ill-conditioned cell type `dream` tested 24 genes by Satterthwaite's
method because R's estimate of the condition number of their adjusted covariance
exceeded 1e6, and `dream_voom` the same 24, from the exact condition number.

The information rows give the size of the two treatments of the weights. Against the
precision weights, every observation at the average weight gave standard errors 0.976
times as large at the median (0.51 to 1.05) and df 0.91 times (0.26 to 3.7) with one
random intercept, and 0.977 (0.81 to 1.02) and 0.89 (0.27 to 3.4) with two; its standard
errors were below the unadjusted ones for 61% and 33% of the genes, the precision
weights' for none. And they give how dream's default changes the results of small
studies against Satterthwaite's method, with which `dream_voom` tested them before:
with 12 samples, p-values 1.12 times as large at the median (0.97 to 1.66 for 90% of the
genes), with standard errors 1.06 times as large at the median and smaller for 20% of
the genes; with 18 samples, 1.00 (0.91 to 1.07), and standard errors 1.00, smaller for
48%; in the cube's cell type with 12 samples, 0.99 (0.73 to 1.12), with as many genes
below an adjusted p-value of 0.05 (12) as before.

Deliberate differences:

- The p-values of `topTable`'s F-tests: dream computes those of every gene on the total
  degrees of freedom of the first gene for the first coefficient listed, which the
  script reproduces to 3e-14 in the 11 F-tests; CytePool gives each gene its own (see
  `cytepool.stats.dream`). Its p-values were 0.14 to 11.5 times dream's.
- Which genes fall back to Satterthwaite's method: dream decides from R's estimate of
  the condition number (`kappa`), CytePool from the exact one; they can differ within
  about 2% of 1e6. Here they decided alike for every gene tested, and the tests agree
  to 3.3e-7.

dream also computes `F.std` with the number of coefficients where they are linearly
dependent, and CytePool with their rank, the numerator degrees of freedom of the F-test;
the F-tests here have independent coefficients.

(validation-dream)=
### dream's weights, fits and tests

The script printed 63 checks (53 pass, 5 information and 5 deliberate differences) in
30.5 s, with a peak memory of 390 MiB. [Mixed models and
dream](statistics/mixed_models.md#dream) explains what it checks.

[`validation/dream.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/dream.py)
compares `voom_with_dream_weights`, `moderated_fixed_effect_tests` and `dream_voom`,
and the maximum likelihood fits, fitted values and approximate residual degrees of
freedom of `fit_mixed_model` they come from, with variancePartition's
`voomWithDreamWeights`, `dream(ddf = "Satterthwaite")`, `eBayes` and `topTable`
(`dream_voom` with `df_method="satterthwaite"`; [Kenward-Roger's tests and
F-tests](#validation-kenward-roger-and-f-tests) checks its default, the other choices of degrees of freedom and the F-tests), on
negative binomial counts of 60 genes from fixed seeds (20261025 to 20261029): a random
donor intercept with a condition, an age and the contrast `conditionstim - age` (10
donors x 3 samples, with two hypervariable genes), moderated also with `robust = TRUE`
and `trend = TRUE`; crossed donor and batch intercepts (8 donors x 4 samples over 3
batches) with sample prior weights; unbalanced donors with 1 to 4 samples, with prior
weights for every observation; and `dream_voom` on three cell types of a cube of 8
donors x 2 conditions, with prior weights from the numbers of cells, the samples with
fewer than 10 cells left out (both samples of a donor and one of another in one cell
type, and all but one sample of each of three donors in the third), and the moderation
also with `trend = TRUE`. Reference: R 4.4.3, variancePartition 1.36.2,
limma 3.62.1, lme4 1.1-38, lmerTest 3.1-3, pbkrtest 0.5.3, Matrix 1.7-4, RcppEigen
0.3.4.0.2 and minqa 1.2.8. dream is compared on lme4 1.1-38, while the
[mixed models](#validation-mixed-models) are compared on lme4 2.0.6: variancePartition 1.36.2's
`dream` fails for every gene with lme4 2.0, which reworked lme4's random-effect terms.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Precision weights and the square roots of the residual SDs in the trend; with prior weights for samples or observations, also the trend's weights alone | `voomWithDreamWeights` | 2.1e-8 | 1e-6 | 3 pass |
| Coefficients and a contrast, relative to the larger of their absolute value and standard error, and standard errors | `dream` | 6.2e-8 | 1e-6 | 3 pass |
| Satterthwaite df | `dream` | 2.2e-7 | 1e-5 | 3 pass |
| Residual SD of an observation of average weight, and approximate residual df | `dream` `sigma`, `rdf` | 2.8e-8 | 1e-6 | 3 pass |
| Prior df and variance, posterior variances (constant, robust and trend priors) | `eBayes` | 1.6e-8 | 1e-6 | 5 pass |
| Moderated t-statistics and B-statistics, relative to max(\|x\|, 1) | `eBayes` | 1.2e-7 | 1e-6 | 5 pass |
| Total df, p-values, adjusted p-values and z-statistics | `eBayes`, `topTable` | 7.9e-8 | 1e-5 | 5 pass |
| Maximum likelihood and REML at CytePool's estimates: log-likelihood, residual variance, approximate residual df, and fitted values relative to their largest value | lme4 `logLik`, `sigma`, `fitted`; `rdf.merMod` | 5.9e-15 | 1e-10 | 4 pass |
| Satterthwaite df there, of maximum likelihood and REML fits | lmerTest `summary` | 5.8e-7 | 1e-5 | 4 pass |
| Maximum likelihood optimum: log-likelihood and residual variance | lme4 `logLik`, `sigma` | 2.5e-8 | 1e-6 | 2 pass |
| Genes whose log-likelihood is below lme4's maximum by more than 1e-10, relative | lme4 `logLik` | 0 genes | 0 | 2 pass |
| `dream_voom`, in two cell types: samples used and features tested | the rules of `limma_voom` | 0 differences | 0 | 2 pass |
| `dream_voom`, with constant and trend priors: logFC and AveExpr | `topTable` | 1.5e-8 | 1e-6 | 4 pass |
| `dream_voom`: t and B | `eBayes` | 2.7e-8 | 1e-6 | 4 pass |
| `dream_voom`: df, P.Value, adj.P.Val and z.std | `eBayes`, `topTable` | 1.1e-7 | 1e-5 | 4 pass |
| Prior df (median and smallest) and the estimator of the prior | CytePool | 8.4 to 15.2; robust 5.7 to 12.0; likelihood | n/a | 5 information |
| Deliberate differences, listed below | variancePartition | 5 inputs | n/a | 5 deliberate differences |

variancePartition fits every gene with lme4, whose optimizer stops at about 1e-6 of
the optimum with its default tolerances. The script runs it with Powell's BOBYQA from
minqa (lme4's `"bobyqa"`) with the trust region shrunk to 1e-10, which lme4's
`optimizer` argument accepts by name as a function that reports convergence: at such
tolerances lme4 warns for some genes, and variancePartition turns the warning into an
error and drops the gene. BOBYQA from nloptr at tolerances near machine precision
ended at worse points inside `voomWithDreamWeights` for crossed random effects, so it
is not used. Estimates are then compared to the close tier, and Satterthwaite's
degrees of freedom, with the p-values and z-statistics that use them, to 1e-5, as in
the section on mixed models: lmerTest differentiates the deviance numerically. At
CytePool's estimates, lme4 evaluated once (`maxeval = 1`) gives closed forms, compared
to the exact tier. P-values are scaled as [above](#reading-the-tables). In the cube the features tested and the TMM factors are CytePool's
(`filter_by_expression` and `tmm_factors`, which [Normalization and
filtering](#validation-normalization) checks against edgeR), passed to R with the samples kept by the rule of `limma_voom`; R gets the
effective library sizes and the numbers of cells as prior weights, with
`rescaleWeightsAfter = TRUE`.

The conventions of variancePartition that its documentation leaves open, found by
probing it and listed in the documentation of `cytepool.stats.dream`, are all exercised
here: `voomWithDreamWeights` fits the mixed models by maximum likelihood and reads the
trend at the fitted values with the predicted random effects; it divides each gene's
prior weights by their mean, and by default returns the trend's weights alone; `dream`
fits each gene with its weights divided by their mean, so that `sigma` is the residual
standard deviation of an observation of average weight; `eBayes` moderates these
variances on `rdf` with limma's default estimator (here the likelihood estimator, as
the `rdf` differ), and the p-values use Satterthwaite's df plus the prior's, without
limma's cap. With `robust = TRUE` the two hypervariable genes got prior df of 5.7,
against 12.0 for the others.

Deliberate differences:

- The default span: `voom_with_dream_weights` chooses it from the number of genes, as
  `voom` does (0.959 for 60 genes), and `voomWithDreamWeights` uses 0.5; with the
  defaults, the weights differ by up to 26%. `span=0.5` reproduces variancePartition.
- A z-statistic beyond p = 1e-300: `topTable` computes `z.std` from the p-value
  floored at 1e-300, so it gives -37.07 for a contrast scaled up 1e13 times, where
  CytePool's z-statistic, from the logarithm of the tail probability, is -44.59.
- A gene that the fixed and random effects reproduce, constant within every donor:
  `voomWithDreamWeights` keeps it in the trend with the square root of a residual SD
  of 3.7e-4, and `dream` keeps it; CytePool leaves it out of the trend and returns its
  tests as NaN.
- A design without random effects: `dream` fits it with `lmFit` (class `MArrayLM`);
  `dream_voom` refuses it and points to `limma_voom`.
- A cell type left with one sample per donor: `voomWithDreamWeights` stops ("Initial
  model failed"); `dream_voom` reports the status `random_effects_not_estimable` and
  goes on with the other cell types.
- The degrees of freedom of `topTable`'s F-tests, and the condition number that decides
  which genes fall back to Satterthwaite's method: [Kenward-Roger's tests and
  F-tests](#validation-kenward-roger-and-f-tests) checks both.

(validation-variance-partitioning)=
### Variance partitioning

The script printed 31 checks (22 pass, 6 information and 3 deliberate differences) in
63.1 s, with a peak memory of 394 MiB. [Mixed models and
dream](statistics/mixed_models.md#variance-fractions) explains what it checks.

[`validation/variance_partitioning.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/variance_partitioning.py)
compares `variance_fractions` and `variance_partitioning`, and the figures
`plot_variance_fractions` and `plot_feature_variance_fractions`, with variancePartition's
`fitExtractVarPartModel`, `calcVarPart`, `plotVarPart` and `plotPercentBars`, on data
from fixed seeds (20261101 to 20261108), 60 genes each: one random donor intercept (10
donors x 3 samples, a fifth of the genes without a donor effect); random donor, batch
and condition intercepts with a donor's age and a sample's RNA integrity as fixed
effects and precision weights for every observation (8 donors x 4 samples), also
fitted by REML; unbalanced donors with 1 to 4 samples and sample weights; nested
donor and tissue intercepts (6 donors x 2 tissues x 2 replicates); a linear model of
the age, the RNA integrity, the batch and the condition with the same weights (which
variancePartition fits with `lm`); negative binomial counts of 80 genes with the
weights of `voomWithDreamWeights(span = 0.5)`; and `variance_partitioning` on three
cell types of a cube of 8 donors x 2 conditions, with prior weights from the numbers
of cells, three samples left out of one cell type for having fewer than 10 cells, and
one sample per donor in the third. Reference: R 4.4.3, variancePartition 1.36.2,
lme4 1.1-38, minqa 1.2.8 and ggplot2 4.0.3, in the R environment of `dream.py`.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Fractions and variances at the optimum, relative to the gene's total variance: one, several, unbalanced and nested random effects, and voom's weights | `fitExtractVarPartModel`, `calcVarPart` | 4.4e-8 | 1e-6 | 5 pass |
| The same, fitted by REML | `fitExtractVarPartModel(REML = TRUE)` | 2.0e-7 | 1e-6 | 1 pass |
| A linear model with factors and observation weights | `fitExtractVarPartModel` (`lm`) | 3.8e-15 | 1e-10 | 1 pass |
| Variances at CytePool's estimates | lme4 (`maxeval = 1`), `calcVarPart` | 1.9e-13 | 1e-10 | 5 pass |
| Random effects estimated at zero: CytePool's exact zeros and lme4's below 1e-10 | `fitExtractVarPartModel` | equal in every case (19 to 91) | n/a | 6 information |
| `variance_partitioning`, in two cell types: samples used and features | the rules of `dream_voom` | 0 differences | 0 | 2 pass |
| `variance_partitioning`: fractions | `voomWithDreamWeights`, `fitExtractVarPartModel` | 3.4e-8 | 1e-6 | 2 pass |
| Figures: the variables and their order, the quartiles, whiskers and outliers of the violins, and the segments of the bars, in percent over 100 | `plotVarPart(sortCols())`, `plotPercentBars` | 5.3e-16 | 1e-10 | 3 pass |
| The violins' half-widths at R's points, relative to the widest | `plotVarPart` | 5.9e-4 | 1e-3 | 1 pass |
| Designs both refuse: a factor among the fixed effects of a mixed model, and a formula without an intercept | `fitExtractVarPartModel` | 0 differences | 0 | 2 pass |
| Deliberate differences, listed below | variancePartition | 3 inputs | n/a | 3 deliberate differences |

lme4 runs with Powell's BOBYQA with its trust region shrunk to 1e-10, as in `dream.py`,
so the fractions at the optimum are compared to the close tier. Differences are taken
relative to each gene's total variance, the scale of a fraction: lme4 leaves a variance
component at the boundary at about 1e-16 of the total where CytePool holds it at zero,
and the two found the same components at zero in every case. At CytePool's estimates,
lme4 evaluated once and `calcVarPart` give closed forms, compared to the exact tier,
as are linear models, which have closed forms. The figures are drawn by R from
CytePool's fractions, so they compare the drawing alone. The violins agree to 1e-3 of
their widest point: ggplot2 takes R's `density()`, which bins the values on a grid of
512 points and convolves them by the fast Fourier transform, as its help page
describes, and differs from the kernel density estimate that CytePool draws by up to
5.9e-4 here. ggplot2 stacks a segment of no width anywhere in its bar, so only the
segments with a width are compared by position.

The conventions of variancePartition that its documentation leaves open, found by
probing it and listed in the documentation of `cytepool.stats.variance_fractions`, are
exercised here: the variance of a fixed effect is the weighted variance of its fitted
contribution, with the denominator of a maximum likelihood variance; in a mixed model
the terms share the variance of all the fitted fixed effects in proportion to their
own, and in a linear model each term has its own; each gene's precision weights are
divided by their mean. The two covariates of the second case have a correlation of
-0.13 over its samples, and without the sharing its fractions would differ by up to
0.034.

Deliberate differences:

- A gene constant within every donor: variancePartition reports the fractions of
  lme4's fit with a residual variance near zero, a donor fraction of 1.0000; CytePool
  leaves the gene unfitted, with NaN fractions, as its mixed models have no maximum
  likelihood estimate for such a gene.
- REML without random effects: variancePartition fits `lm` and takes the residual sum
  of squares over n whatever `REML` says, so its fractions do not change; CytePool
  takes REML's residual variance, over n - p, which changed a fraction by up to 0.052.
- A cell type with one sample per donor: `voomWithDreamWeights` stops ("Initial model
  failed"); `variance_partitioning` reports the status `random_effects_not_estimable`
  and goes on with the other cell types.

## Composition tests

(validation-composition-tests)=
### Cell-type composition

The script printed 32 checks (25 pass and 7 deliberate differences) in 13.1 s, with a
peak memory of 860 MiB. [Composition tests](statistics/composition.md#propeller-steps) explains what it
checks.

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
values, in the exact tier; statistics, p-values and FDRs are scaled as [above](#reading-the-tables),
in the exact tier, or in the
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

## Gene sets and pathways

(validation-gene-set-tests)=
### Gene-set tests

The script printed 93 checks (79 pass, 8 deliberate differences and 6 information) in
6.9 s, with a peak memory of 278 MiB. [Gene-set tests](statistics/gene_set_tests.md#camera)
explains what it checks.

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
method = "hill")`, with the design reparametrized by `contrastAsCoef`; with an estimated
correlation, the parametric p-value of `cameraPR` is converted to the df of the
estimate. Reference: R 4.4.3, limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| camera: numbers of genes, correlations, directions, p-values and FDR, with a given or estimated correlation, ranks and a trend | `camera` | 1.3e-14 | 1e-10 | 35 pass |
| camera with robust moderation | `cameraPR`, `rankSumTestWithCorrelation` | 1.7e-12 | 1e-6 | 12 pass |
| fry with `standardize` `"none"` and `"residual_sd"`: numbers of genes, directions, p-values and FDR, mixed p-values of sets of one or more than d genes, and their FDR | `fry` | 5.2e-13 | 1e-10 | 15 pass |
| fry with `"posterior_sd"`, with and without trend or robust moderation | `fry` | 4.4e-5 | 1e-3 | 15 pass |
| fry with `"posterior_sd"` and an infinite prior df, where both standardize as `"none"` | `fry` | 3.3e-14 | 1e-10 | 1 pass |
| CytePool's mixed p-values of the sets of 2 to 14 genes against 200,000 random rotations of their effects | simulation | 0.025 (absolute) | 0.1 | 1 pass |
| fry's mixed p-values of sets of 2 to d genes | `fry` | 0.88 (absolute) | n/a | 8 deliberate differences |
| The rotation simulation: limma's mixed p-values of the sets of 2 to 14 genes; both programs' of the sets of 15 to 30 genes; both programs' of the three sets with changed genes (relative) | simulation | 0.59, 0.030, 0.99 | n/a | 3 information |
| Samples sorted by group, observation weights and the contrast as weights: camera, fry with `"residual_sd"`, and the genes whose last pivot R's `qr()` signs by rounding noise | `camera`, `fry`, `qr` | 0.012, 0.48; 115 of 1,500 genes | n/a | 3 information |

Correlations are compared as differences, p-values scaled as [above](#reading-the-tables)
(limma's p-values above 1 count as 1), and a
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

(validation-rotation-gene-set-tests)=
### Rotation gene-set tests

The script printed 103 checks (81 pass, 20 deliberate differences and 2 information) in
20.9 s, with a peak memory of 190 MiB. [Gene-set tests](statistics/gene_set_tests.md#roast)
explains what it checks.

[`validation/rotation_gene_set_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/rotation_gene_set_tests.py)
compares `roast`, `mroast` and `romer` with limma's functions of the same names on
log-expression of 1,500 genes simulated from fixed seeds, with the gene variances and
gene sets of `gene_set_tests.py` and three sets with contribution weights: a signature
whose genes go up or down, with weights +1 and -1; a set with weights of different
sizes; and a set with half of its weights zero. The cases are three groups with the
contrast given as a design column; interleaved groups with observation weights and a
named contrast given as weights; the same with a trended prior estimated robustly;
counts with voom weights; a paired design with sample weights; equal residual
variances (an infinite prior df), also with effects so large that limma's log-odds of
23 genes exceed 709.8; and contribution weights for every gene, drawn from -2, -1, 0.5,
1 and 3. The exact checks give CytePool the rotations limma drew: a generator that
replays the standard normal draws R gives limma after `set.seed`, in the order limma
uses them (found by probing). limma tests each set with the same seed, which is how
CytePool's `mroast` shares its rotations between sets. The calibrated checks let each
program draw its own rotations, and allow a difference of 4 sd(p_1 - p_2) + 2 / (B + 1)
for B rotations each, with the sd from the binomial variance of each estimate at their
mean. Differences of p-values are absolute: each p-value counts rotations, so a
difference of one count is at least 1 / (2B + 1). Reference: R 4.4.3, limma 3.62.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| roast given limma's rotations: the down, up, two-sided and mixed p-values, active proportions and numbers of genes of every set, for every set statistic and case | `roast` | 5.0e-16 | 1e-12 | 19 pass |
| mroast given the same rotations for every set: two-sided p-values, and directions (mismatched sets) | `mroast` | 0; 0 sets | 1e-12; 0 | 38 pass |
| romer given limma's rotations: up, down and mixed p-values and numbers of genes, for every set statistic, with and without shrinkage, with sample weights and with an infinite prior df | `romer` | 0 | 1e-12 | 7 pass |
| limma's FDR and FDR.Mixed from its p-values by mroast's rule; numbers of genes and active proportions with each program's own rotations | `mroast` | 4.4e-16 | 1e-12 | 9 pass |
| Each program's own rotations: mroast (mean, msq, mean50; B = 1,999) and romer (mean, mean50; B = 999) | `mroast`, `romer` | 0.54 of the bound | 1 | 5 pass |
| mroast's mean statistic with 20,000 rotations against fry, its limit, with equal variances | `fry` | 0.34 of the bound | 1 | 1 pass |
| CytePool's fry, mixed p-values of the sets of 2 to 6 genes, against the msq rotation test with 20,000 rotations | `mroast` | 0.015 (absolute) | 0.1 | 1 pass |
| CytePool's msq p-values with every contribution weight multiplied by 10 (B = 1,999) | `roast` | 0 | 1e-12 | 1 pass |
| limma's fry, mixed p-values of the sets of 2 to 6 genes, against the same | `mroast` | 0.45 (absolute) | n/a | 1 information |
| Genes whose log-odds in limma exceed 709.8 (very large effects) | `romer` | 23 genes | n/a | 1 information |
| p-values where the observed floor-mean or mean-square statistic is the smallest it can be | `roast` | 0.50 (absolute) | n/a | 6 deliberate differences |
| floormean's mixed p-values of sets with negative weights; msq's up, down and two-sided p-values of sets with weights of both signs | `roast` | 0.65; 0.33 (absolute) | n/a | 6 deliberate differences |
| limma's msq p-values with every weight multiplied by 10 (B = 1,999) | `roast` | 0.45 (absolute change) | n/a | 1 deliberate difference |
| The set with weights of zero: its number of genes, and its mean50 p-values | `roast` | 8 against 16 genes; 0.039 | n/a | 6 deliberate differences |
| romer with shrinkage where limma's posterior probabilities are NaN | `romer` | 0.15 (absolute) | n/a | 1 deliberate difference |

The z-scores of the moderated t-statistics are Bailey's (1980) transformation on
min(d_0 + d, 10^4) degrees of freedom; romer's shrinkage of each gene's contrast effect
computes its log-odds on d_0 + d degrees of freedom, without the limit, and with the
proportion of true null hypotheses of limma's `propTrueNull(method = "lfdr")`. These
conventions, the set statistics' definitions where limma differs from the paper (the
mean-50 statistic averages the ceiling(m / 2) largest values, and romer's floor-mean
mixed statistic floors \|t\| at 1), and the order of limma's random draws were found
by probing, and the exact checks confirm them, with an infinite prior df too. Without
shrinkage, romer agrees with limma exactly on the data with very large effects, which
shows that the difference with shrinkage comes from limma's NaN posterior
probabilities. The fry checks give the reference that `gene_set_tests.py` approximated
by simulation: the mixed p-values of fry for sets of 2 to d genes, where CytePool uses
all d + 1 dimensions of the rotation and limma only the set's, against the rotation
test of the same statistic, which `mroast`'s msq statistic is when all genes have the
same residual variance (up to the limit on the df of the z-scores).

Deliberate differences:

- roast and mroast count a rotated statistic equal to the observed one as at least as
  extreme, as Wu et al. (2010) and Phipson and Smyth (2010) define the p-value and as
  limma's `romer` does; limma's `roast` counts only larger ones. Where the observed
  floor-mean or mean-square statistic is the smallest it can be (a set without a gene
  on one side of zero, or, for the floor-mean mixed statistic, without a gene whose
  \|z\| exceeds 0.674), CytePool's p-value is 1 and limma's about 1/2. The script
  checks that CytePool's p-value is 1 in each such case.
- With contribution weights a_g, the mean-square statistic weighs each gene's z^2 by
  \|a_g\|, as Wu et al. (2010) define it; limma's up, down and two-sided p-values differ
  from it for sets with weights of both signs, and with more than 1,000 rotations its
  p-values change when every weight is multiplied by the same number (here by up to
  0.45 for a factor of 10). The floor-mean mixed statistic weighs each gene by
  \|a_g\|, where limma takes a negative weight as evidence against the set; the script
  checks that CytePool's mixed p-value is limma's with the absolute weights.
- A gene of contribution weight zero leaves the set: it counts neither in the number
  of genes nor among the values that the mean-50 statistic averages. limma keeps it in
  both, and leaves it out of the active proportions, as CytePool does.
- romer's posterior probability that a gene is differentially expressed is
  1 / (1 + exp(-lods)). limma's is NaN beyond log-odds of 709.8 (\|t\| above about 38
  with an infinite prior df), which makes the gene's shrunk contrast effect NaN and
  changes the p-values of every set (found by probing).
- A set without genes, or whose genes all have weight zero, has NaN p-values and no
  direction in CytePool and is left out of the other sets' adjustment; limma reports
  "down" and NA, or the smallest possible p-value when all weights are zero. A gene
  listed twice counts once in CytePool, twice in limma. mroast tests every set with the
  same rotations, where limma draws new ones for each set. The tests check these.

(validation-gene-set-files-and-ora)=
### Gene-set files and over-representation

The script printed 34 checks (29 pass and 5 deliberate differences) in 4.7 s, with a
peak memory of 297 MiB. [Over-representation and
GSEA](statistics/over_representation_and_gsea.md#over-representation) explains what it
checks.

[`validation/gene_set_files_and_ora.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/gene_set_files_and_ora.py)
checks the reading and writing of GMT files, the matching of gene sets to an analysis,
and over-representation analysis. Four GMT files of 60 sets are simulated from fixed
seeds in the styles of MSigDB (names in capitals, links as descriptions, symbols),
Reactome (names with spaces and punctuation, lines ending in a carriage return and a
line feed, tabs at the ends of lines), WikiPathways (names joined by %, Entrez ids, a
byte-order mark, blank lines) and the GMT exports of the Gene Ontology (descriptions
"na" or empty, genes listed twice, a set without genes, a name outside ASCII,
compressed by gzip). `read_gmt` reads each file and `write_gmt` writes it back; R reads
both files by hand (`readLines`, fields split at tabs), and decoupler's `read_gmt`
reads the MSigDB-style file. A file of 60 sets of symbols is matched to the 2,000 genes
of an analysis by `GeneSets.restrict_to` and by limma's `ids2indices`, the sets of 15
to 200 genes are kept, and `camera` tests them on simulated log-expression, given the
sets that CytePool read, against limma's `camera` given those that R read.

`over_representation` runs on simulated results of 4,000 genes, 8% up and 6% down at
an FDR of 5%, with a block of tied p-values, and 305 sets of up to 600 genes, a fifth
of each set outside the results: sets enriched for genes up, down or both, or for none,
two identical sets, a set that lists genes twice, a set of every gene, a set outside
the universe, and a set with two features of one symbol, one up and one down. Its table
is compared with limma's `kegga` of a fit with the same p-values and log fold changes,
which selects the genes up and down itself, and with kegga's default method given the
genes of each direction and the universe: by feature id; by symbol, which 160 features
share in pairs; by symbol where 131 features lack one; with a universe of 60% of the
genes, 100 genes of the sets outside the results and 50 genes of no set; with 416 genes
without a p-value; and for a list of 230 genes, 30 of them outside the universe and
some listed twice. The adjusted p-values are compared with `p.adjust` of kegga's
p-values, the order of the sets with `topKEGG`'s, the p-values also with R's `phyper`
on a grid of 247 points that reaches 1e-300 and underflow, and the dot plot's sets,
dots and colours with kegga's counts and adjusted p-values. Reference: R 4.4.3, limma
3.62.1, decoupler 2.2.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| GMT files in four styles: the names, descriptions and genes of every set, read, written and read again | R, by hand | 0 sets differ | 0 | 4 pass |
| The MSigDB-style file: names and genes | decoupler `read_gmt` | 0 sets differ | 0 | 1 pass |
| `restrict_to`: the sets kept and the positions of their genes | `ids2indices` | identical | 0 | 1 pass |
| `filter_by_size(15, 200)`, then `camera` of the 34 sets kept | `camera` | 3.1e-15 | 1e-10 | 1 pass |
| P(X >= k) at 247 points | `phyper` | 2.1e-12 | 1e-10 | 1 pass |
| Genes up and down that kegga selects from a fit, by feature id and by shared symbols: numbers of genes, overlaps, p-values, adjusted p-values and the order of the sets | `kegga` of a fit, `p.adjust`, `topKEGG` | 6.2e-14 | 1e-10 | 4 pass |
| Genes up, down and either way, by feature id, by shared or missing symbols, with a universe given and without p-values for some genes, and a list of genes: the same, and the numbers selected, the size of the universe and the expected overlaps | `kegga`, `p.adjust`, `topKEGG` | 9.7e-14 | 1e-10 | 16 pass |
| The dot plot: the 20 sets shown and their order, the area and colour of its 60 dots | `kegga` | identical | 0 | 1 pass |
| Features without a symbol, which kegga of a fit leaves out of its adjustment | `kegga` of a fit | 1.9 (relative) | n/a | 2 deliberate differences |
| Genes without a p-value, which kegga of a fit keeps in the universe | `kegga` of a fit | 5.3 (relative) | n/a | 2 deliberate differences |
| Sets without genes in the universe: rows with NaN p-values here, none in kegga's table | `kegga` | 1 to 23 sets per case | n/a | 1 deliberate difference |

p-values, adjusted p-values and expected overlaps are compared relative to max(p,
1e-300); numbers of genes, overlaps, names and orders must be identical. kegga selects
the genes of a fit whose adjusted p-value is below its `FDR`, strictly, up or down by
the sign of the coefficient; it counts a gene id on several features once, in both
lists when one feature goes up and another down; it leaves out features without an id;
and `topKEGG` orders sets by p-value, ties by the number of genes. These conventions
were found by probing, and the checks by symbol confirm them. Half of the features
without a symbol are changed genes, so that kegga's adjustment without them moves
other genes across the cutoff and the difference shows.

Deliberate differences:

- A set without genes in the universe has a row with NaN p-values in CytePool, left
  out of the adjustment of the others; kegga leaves the set out of its table.
- Genes without a p-value, such as those DESeq2's independent filtering leaves out:
  kegga's method for a fit keeps them in the universe, where CytePool, which takes them
  as not tested, leaves them out. Given the universe of every gene, CytePool gives
  kegga's values; the tests check this.
- Features without an id, such as a symbol: kegga's method for a fit leaves them out
  of its Benjamini-Hochberg adjustment, so that the genes it selects depend on which
  features have a symbol. CytePool takes the adjusted p-values of the results, the
  genes the analysis called significant. With the p-values adjusted over the features
  with a symbol, CytePool gives kegga's values; the tests check this.

(validation-gsea)=
### Gene set enrichment analysis

The script printed 24 checks (23 pass and 1 deliberate difference) in 7 min 3 s, with a
peak memory of 323 MiB. [Over-representation and
GSEA](statistics/over_representation_and_gsea.md#gsea) explains what it checks.

[`validation/gsea.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/gsea.py)
checks gene set enrichment analysis and its figures against fgsea. Statistics of 4,000
genes are simulated from fixed seeds like moderated t-statistics, 6% of the genes up and
5% down, with 60 genes tied at 1.25 and 30 at zero, with 90 gene sets of 1 to 1,000
genes, a fifth of each outside the ranking: sets enriched for genes up, down or both, or
for none, a set of every gene, which neither program tests, a set of one gene, half of
the tied genes, a set that lists genes twice and two identical sets. `gsea` and fgsea's
`fgsea` run with the same options: fgsea's defaults (its multilevel method, 1,000
permutations, 101 sets per level, eps 1e-50); gseaParam 0 and 0.5; scoreType "neg";
scoreType "pos" on -log10 p-values, which are all positive; sets of 15 to 500 genes,
with eps 0, 51 sets per level and 2,000 permutations; permutations alone (fgsea's
`nperm`, 2,500 in three chunks); statistics mostly positive, with 6 sets spread evenly
along the ranking, a set of a few genes near its top and others spread along it, whose
p-value is about 0.05, and 3 sets of 25 genes from its bottom, whose negative scores
fewer than 10 random sets of their size reach, so that no set goes to the multilevel
method; and one set alone.

The R script records the seeds that fgsea draws where it hands them to its compiled code
(`trace` of `fgseaSimpleImpl` and `multilevelImpl`), and `gsea` draws the same through
`rng`; the two programs then draw the same random sets, and must agree exactly. fgsea
draws the random sets of a set tested alone with R's `sample.int`, which cannot be
replayed: that case is exact for the enrichment score and leading edge, and calibrated
for the rest, with seeds of their own (R's 1 to 20 and numpy's 1 to 20): the mean
log10 p-value and the mean normalized score of each set over 20 runs must agree within
4 standard errors of their difference, for 36 sets and for the set alone. The figures:
what `plot_enrichment` draws against `plotEnrichmentData`, for three sets with gseaParam
1 and 0.5, and the bars and numbers of `plot_gsea_table` against the ticks of
`plotEnrichmentData`, scaled as `plotGseaTable` scales them, and its numbers as it
writes them. Reference: R 4.4.3, fgsea 1.32.2.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| The sets tested, their sizes, leading edges and enrichment scores, in all 9 cases | `fgsea` | 0 | 1e-10 | 9 pass |
| Normalized scores, p-values, log2 errors and adjusted p-values, given fgsea's seeds, in 7 cases | `fgsea` | 2.2e-13 | 1e-10 | 7 pass |
| The same of the mostly positive statistics, without the adjusted p-values | `fgsea` | 4.2e-16 | 1e-10 | 1 pass |
| Mean log10 p-values and mean normalized scores over 20 runs, seeds of their own, for 36 sets and one set alone | `fgsea` | 2.4 standard errors | 4 | 4 pass |
| `plot_enrichment`: the walk, the ticks and the extremes of 6 plots | `plotEnrichmentData` | 1.7e-14 | 1e-10 | 1 pass |
| `plot_gsea_table`: the bars and numbers of 10 sets | `plotEnrichmentData`, `plotGseaTable` | 1.1e-13 | 1e-10 | 1 pass |
| Adjusted p-values of the mostly positive statistics | `fgsea` | 0.16 (relative) | n/a | 1 deliberate difference |

Numbers are compared relative to their absolute value, or 1e-300, and NaN must stand
where fgsea gives NA; sets, sizes and leading edges must be identical. Given fgsea's
seeds, the random scores and the counts of the permutations are the same to the last
bit; the multilevel p-values differ in the last bits, through the digamma function of
each program.

Deliberate difference:

- The adjusted p-values of the mostly positive statistics, where some sets have too few
  random scores of their sign and no set goes to the multilevel method: fgsea adjusts
  the p-values over the sets before it removes those that have some, but fewer than 10,
  random scores of their sign, as though they had been tested, so that its adjusted
  p-values of the sets it reports come out smaller or larger; CytePool adjusts over the
  sets with a p-value. The tests check that CytePool's are Benjamini and Hochberg's of
  the p-values it reports.

(validation-pathway-scores)=
### Pathway scores

The script printed 24 checks (19 pass, 3 deliberate differences and 2 information) in
10.7 s, with a peak memory of 949 MiB. [Pathway scores per
sample](statistics/pathway_scores.md#gsva) explains what it checks.

[`validation/pathway_scores.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pathway_scores.py)
checks the scores of gene sets in every sample, and their heatmap, against GSVA 2.0.0,
whose `gsva()` runs with a `gsvaParam`, `ssgseaParam`, `zscoreParam` or `plageParam`,
and the CytePool function with the same options. The inputs are simulated from fixed
seeds: log-expression values of 2,000 genes in 24 samples, 200 genes higher in the
second 12, with two constant genes, a gene equal to another, a gene of whole numbers
and a gene with an outlier sample; negative binomial counts of the same genes in 16
samples, stored as integers and as decimals; Poisson counts with means from 0.3 to 6
(low counts); 400 genes in 210 samples; the log-expression values rounded to one
decimal; the same values with 2% of those of the first 12 samples missing; and a cube of
one cell type with the counts, whose samples have 3 to 60 cells. The 66 gene sets have
1 to 300 genes, a fifth of each outside the data: random sets, a set of one gene, one
with the constant genes, one that lists genes twice, two identical sets and a set of
every gene, which is left out for ssGSEA but in one case. Reference: R 4.4.3, GSVA
2.0.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| GSVA's method: the Gaussian kernel, tau 0.5 with maxDiff FALSE, tau 2 with absRanking, no kernel, sizes 10 to 100, counts stored as integers (the Poisson kernel), as decimals (the Gaussian kernel) and 210 samples (no kernel), in 8 cases | `gsva` with `gsvaParam` | 0 | 1e-10 | 8 pass |
| ssGSEA: the defaults, alpha 1 without normalization, alpha 0 on tied values, in 3 cases | `ssgseaParam` | 1.8e-15 | 1e-10 | 3 pass |
| Combined z-scores | `zscoreParam` | 2.7e-15 | 1e-10 | 1 pass |
| PLAGE, the sets of two genes or more, up to sign | `plageParam` | 8.7e-14 | 1e-10 | 1 pass |
| Missing values: GSVA's method in the samples without any, ssGSEA and combined z-scores | `gsvaParam` and `ssgseaParam` with `use = "na.rm"`, `zscoreParam` | 2.0e-15 | 1e-10 | 3 pass |
| A cube's cell type: its log-CPM and its counts | `gsva` with `gsvaParam` | 0 | 1e-10 | 2 pass |
| `plot_pathway_scores`: the 1,584 cells of the standardized heatmap, the samples grouped | GSVA, R's `heatmap(scale = "row")` | 3.2e-16 | 1e-10 | 1 pass |
| PLAGE: the share of sets whose sign is GSVA's | `plageParam` | 0.42 | n/a | 1 information |
| GSVA's method with the Poisson kernel on low counts | `gsvaParam` | 2.2e-3 | n/a | 1 information |
| GSVA's method in the samples with missing values; PLAGE's sets of one gene; ssGSEA with a set of every gene | `gsvaParam`, `plageParam`, `ssgseaParam` | 1.6 | n/a | 3 deliberate differences |

Differences are relative to the largest absolute score of a case, NaN must stand where
GSVA gives NA, and the sets scored must be GSVA's, in its order. The scores depend on
the ranks of the genes in each sample, which come from comparisons of their kernel
estimates; CytePool evaluates the normal distribution function from a table at steps
of 0.001, as GSVA does, and sums the estimates over the samples in their order, so that
the ranks, and with them the scores, are GSVA's on continuous values and on counts of
the sizes of the cases above. With low counts, many genes have kernel estimates that
are equal in exact arithmetic, such as genes with the same counts in different
samples, and GSVA orders them by the last bit of R's Poisson probabilities, which
CytePool computes as accurately but not always to the same bit: the scores of the sets
that hold them then differ by up to 2.2e-3 of the largest score. PLAGE's sign is that
of R's singular value decomposition, which agrees with CytePool's convention for 42% of
the sets.

Deliberate differences:

- GSVA's method in samples with missing values: with `use = "na.rm"`, GSVA's walk
  falls by 1/(p - k) at a sample's observed genes outside a set, p counting the genes
  missing in the sample too, so that it ends above zero, and it scores the set of every
  gene there; CytePool's walk falls by 1/(p_j - k_j), with p_j and k_j counting the
  genes observed in the sample, and ends at zero (largest difference 4.8e-2, where both
  give a score). The samples without missing values agree exactly.
- PLAGE's sign: GSVA returns the sign that R's singular value decomposition gives,
  which depends on the linear algebra library; CytePool makes a set's scores rise with
  the mean standardized expression of its genes. The scores are compared up to sign,
  and the information row gives the share of sets whose signs agree.
- PLAGE's sets of one gene: GSVA gives 1 in every sample; CytePool gives the gene's
  standardized values, scaled to length one, its first right singular vector (largest
  difference 1.6).
- ssGSEA with a set of every gene: GSVA stops ("Cannot calculate normalizing factor for
  the enrichment scores in ssGSEA, likely due to NA values in the input expression
  data"); CytePool gives that set NaN and normalizes the other scores by their range.

(validation-activity)=
### Pathway and regulator activity

The script printed 89 checks (82 pass, 5 deliberate differences and 2 information) in 2
min 39 s, with a peak memory of 987 MiB. [Activities from
networks](statistics/activities.md#ulm) explains what it checks.

[`validation/activity.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/activity.py)
checks the activities of regulators and pathways, their bar plot and the networks
against decoupler 2.2.0 and progeny 1.28.0. CytePool's loaders download the networks
into its cache. CollecTRI's file and progeny's package are compared with the MD5
digests and sizes their sources publish (Zenodo's record, Bioconductor's `PACKAGES`
and the server's `Content-Length`); `collectri(licence="academic")` with decoupler's
`op.collectri`; the commercial level with the licence purposes that OmniPath's
resources service lists, applied to the file by the script; and the weights and
p-values of `progeny` with progeny's `model_human_full` and `model_mouse_full`, and its
`top` with `getModel` (100 human genes; 500 mouse genes, with ties). The inputs of the
activities are simulated from fixed seeds: log-expression values of 6 samples over
CollecTRI's 6,693 targets and 2,000 other genes, the targets of 6 regulators shifted in
the last 3 samples, with 3 genes zero in every sample; a profile of t-statistics; the
values with 3% missing in two samples, which decoupler, refusing missing values,
scores one profile at a time over its observed genes; 6 samples over the targets of
PROGENy's top 500 genes and 1,000 others; a cube of one cell type, scored as its
log-CPM; and limma_voom's moderated t-statistics in two cell types that test different
genes. `ulm`, `mlm` and `weighted_aggregate` run as decoupler's `mt.ulm`, `mt.mlm` and
`mt.waggr` with `tmin` 5, or 3 for a network of weights of one sign. Reference:
decoupler 2.2.0 with scipy 1.18.1, and progeny 1.28.0 on R 4.4.3.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| ULM's t-statistics and slopes, in 10 cases | `mt.ulm` | 2.3e-13 | 1e-10 | 20 pass |
| ULM's p-values, and their Benjamini-Hochberg adjustment, in 10 cases | `mt.ulm`; scipy's `false_discovery_control` of its p-values | 4.4e-11 | 1e-10 | 20 pass |
| ULM's adjusted p-values, in single precision | `mt.ulm` | 0 | 1.2e-7 | 10 pass |
| MLM's t-statistics and coefficients, in 6 cases | `mt.mlm` | 3.2e-14 | 1e-10 | 12 pass |
| MLM's p-values | scipy's `t.sf` of `mt.mlm`'s t-statistics | 3.3e-12 | 1e-10 | 6 pass |
| Weighted means and sums, and the normalized scores given decoupler's permutations (100 and 1,000) | `mt.waggr` | 1.1e-14 | 1e-10 | 4 pass |
| Weighted sums of PROGENy's top 100 genes | progeny's `progeny(scale = FALSE)` | 6.5e-16 | 1e-10 | 1 pass |
| `plot_top_activities`: the 15 bars of a sample, their sources and lengths | `pl.barplot` | 4.8e-14 | 1e-10 | 1 pass |
| CollecTRI at the academic level; at the commercial level; PROGENy's weights and p-values, human and mouse; its top 100 human and 500 mouse genes | `op.collectri`, OmniPath's resources, progeny's models and `getModel` | 0 | 0 | 6 pass |
| The files downloaded: sizes and MD5 digests | Zenodo, Bioconductor | 0 | 0 | 2 pass |
| decoupler's `op.collectri` at its levels; PROGENy through OmniPath | `op.collectri`, `op.progeny` | n/a | n/a | 2 information |
| The weighted aggregate's p-values; a sample of zeros; MLM with two sources alike; CollecTRI's interactions without a reference | `mt.waggr`, `mt.ulm`, `mt.mlm`, `op.collectri` | 0.97 | n/a | 5 deliberate differences |

Scores and estimates are compared relative to the largest absolute value of a case,
p-values relative to each value, and NaN must stand where the reference has none; the
networks and the bars must be identical. decoupler keeps its adjusted p-values in
single precision, so CytePool's are rounded to it first. decoupler computes MLM's
p-values as one minus the distribution function, which loses their digits below about
1e-12, so CytePool's are compared with scipy's upper tail at decoupler's t-statistics.
The normalized scores are compared given decoupler's permutations: the genes in the
order in which decoupler shuffles them, and its seed, so that both draw the same
permutations. decoupler 2.2.0 accepts a licence for CollecTRI but returns the whole
file at every level. Its `op.progeny` reads PROGENy from OmniPath's annotations, which
name genes by their current symbols, leave out genes OmniPath does not map to a
protein, and merge some genes: of its 62,421 edges with a p-value below 0.05, where the
progeny package has 69,827, 59,617 are named alike, and 59,599 of them have the same
weight and p-value.

Deliberate differences:

- The weighted aggregate's p-values (largest difference 0.97 with 100 permutations,
  2.0e-3 with 1,000): decoupler counts the permutations whose absolute score exceeds
  the absolute observed score and then folds that share as it would fold a one-sided
  one, so that a score near zero gets a p-value near 2 / n for n permutations; CytePool
  doubles the smaller of the two one-sided p-values (1 + b) / (1 + n).
- A sample whose values are all zero: decoupler leaves it out of its result; CytePool
  gives it NaN scores. The other samples agree exactly.
- MLM with two sources of the same targets and weights: decoupler inverts a singular
  matrix and gives NaN t-statistics for both (12 in 6 samples); CytePool refuses,
  naming the source it cannot tell apart from the other.
- CollecTRI's 185 interactions whose field of references is empty: decoupler's loader
  drops them, through pandas' `dropna`; CytePool keeps them, as the file holds them and
  the paper counts them (43,175 interactions).
- MLM's adjusted p-values: decoupler reports MLM's p-values unadjusted, under the name
  of its adjusted p-values; CytePool adjusts them by Benjamini and Hochberg's method
  over the sources of each profile, as for the other methods, so the script compares
  MLM's p-values before adjustment only.

## Several studies

(validation-cube-collection)=
### Studies aligned in a collection

The script printed 15 checks (14 pass and 1 control) in 26.7 s, with a peak memory of
495 MiB. [Aligning studies](statistics/aligning_studies.md#aligning-samples) explains what it
checks.

[`validation/cube_collection.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_collection.py)
simulates the cells of three studies of one stimulation (seed 20261026) and pools each
with `from_cells`: `lupus`, 8 samples of 4 donors, each control and stimulated, 4 cell
types with Cell Ontology ids and 700 genes; `covid`, 10 samples of 5 donors whose
sample and donor ids repeat those of `lupus`, 5 cell types of which two share an
ontology id and merge, and 750 genes, 40 of them filled with zeros and marked as not
measured; and `atlas`, 6 unpaired samples with integer donor ids, 5 cell types without
ontology ids, aligned by a mapping that leaves one out, and 950 genes. The collection
aligns the cell types by ontology id and writes donor ids `study::donor`. anndata and
pandas run as a separate program: each study's cube is exported with `to_anndata`, the
genes marked as not measured are dropped, the rows of each sample and aligned cell type
are summed, and the studies are joined with `anndata.concat` (outer and inner joins,
labelled by study); pandas joins the sample tables. R runs limma-voom on each study's
own counts, summing the cell types that share an aligned id, and on the stacked counts
of `lupus` and `covid` with `~ donor + condition`, by the steps of
`differential_expression.py`. Reference: anndata 0.13.4, pandas 3.1.0, numpy 2.5.4,
Python 3.13.12; R 4.4.3, limma 3.62.1, edgeR 4.4.0. CytePool ran with numpy 2.5.4 and
numba 0.68.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| The sample table: ids `study::sample_id`, donors `study::donor`, the union of the columns, ages that are integers in two studies and floats in one as float64, and missing values where a study lacks a column | `pandas.concat` | 0 mismatches | 0 | 1 pass |
| The genes, as a set and in the order in which the studies first list them, and which study measured which | `anndata.concat`, outer join (NaN where a study lacks a gene), `pandas.unique` | identical; 0 mismatches | 0 | 2 pass |
| Every layer of each study's aligned cube: merged cell types, measured genes | the outer join's values | 0 | 1e-10 | 1 pass |
| `stack()`: genes in order, cell types and samples; every layer; the cells of each entry, and an entry without cells | `anndata.concat`, inner join, kept to the cell types of every study | 0 | 0 and 1e-10 | 3 pass |
| `limma_voom` on each study's aligned cube, lined up by `StudyResults.table`: logFC, AveExpr, t, P.Value, adj.P.Val, B and the standard error, in 12 cell types of 3 studies | limma, study by study | 3.8e-13 | 1e-10 | 3 pass |
| The status of every gene of every study: tested, measured and left out, or not measured | limma's tables and each study's measured genes | 0 mismatches | 0 | 3 pass |
| `limma_voom` on the stack of `lupus` and `covid`, `~ donor + condition` with donors `study::donor` | limma on the stacked counts | 1.4e-13 | 1e-10 | 1 pass |
| The same with the donor ids as the studies give them | limma on the stacked counts, donors `study::donor` | 1.9 (log2 fold changes) | must exceed 1e-10 | 1 control |

limma ran on each study's own counts, with the genes filled with zeros, and its tables
equal those of the aligned cubes, which leave those genes out: genes without counts
change no library size, expression filter or normalization factor, so a gene that a
study did not measure can be marked missing without changing any other gene's result.
The control shows why donor ids are written with their study: taken for one person,
donor D1 of two studies changes the fit of every gene.

(validation-meta-analysis)=
### Meta-analysis

The script printed 75 checks (52 pass, 20 information and 3 deliberate differences) in
59.5 s, with a peak memory of 256 MiB. [Meta-analysis](statistics/meta_analysis.md#meta-analysis-model)
explains what it checks.

[`validation/meta_analysis.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/meta_analysis.py)
compares `fit_meta_analysis` with metafor's `rma.uni`, run feature by feature in R on the
same simulated effects and standard errors (seed 20261028), for the fixed-effect model
(`method = "EE"`) and the DerSimonian-Laird, REML and Paule-Mandel estimators of the
between-study variance, each with z-tests and with the Knapp-Hartung adjustment
(`test = "knha"`). The ten cases have 120 features each: 3 studies without and with
large heterogeneity; 30 studies without and with heterogeneity; 2 studies; 5 studies,
one with a standard error 1,000 times the others'; 4 studies, one imprecise and far
from the others; 8 studies with 30% of the values missing (NaN in the effect, the
standard error or both), three features measured by one study and two by none;
features of 4 studies whose between-study variance is at zero although the Hedges
estimate, from which metafor's REML starts, is positive; and 5 studies whose standard
errors are 1e-9 to 4e-9 for effects of spread 1. metafor runs its iterative estimators
to convergence, REML with `control = list(threshold = 1e-13, maxiter = 10000)` and
Paule-Mandel with `control = list(tol = 1e-14, maxiter = 10000)`, and, for
information, with their defaults. Then three cohorts of 5, 4 and 6 donors, each sampled
before and after a stimulation (seed 20261029), are simulated as pseudobulk counts:
the first has T, B and NK cells and 400 genes, the second too few NK cells to fit them,
10% fewer genes, 20 genes of its own and its genes in shuffled order, the third no NK
cells and 10% fewer genes, and 30% of the changes differ between cohorts. `limma_voom`
tests the stimulation in every cohort; R merges the cohorts' tables by gene id, runs
`rma.uni` on every gene that enough cohorts tested and adjusts the p-values with
`p.adjust(method = "BH")`, and `meta_analysis` is compared with its results. Reference:
R 4.4.3, metafor 5.0.1, jsonlite 2.0.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Fixed effect and DerSimonian-Laird, with z-tests and Knapp-Hartung, in the 10 cases: estimate, standard error, statistic, p-value, confidence limits, tau², Q and its p-value, I², H² and the number of studies | `rma.uni` | 2.3e-13 | 1e-10 | 20 pass |
| REML and Paule-Mandel, the same | `rma.uni` | 3.4e-10 | 1e-6 | 20 pass |
| REML where the restricted likelihood has a higher maximum than metafor's estimate | `rma.uni`, `logLik` | 2 of 120 features, restricted log-likelihood higher by up to 0.023 | n/a | 1 deliberate difference |
| REML and Paule-Mandel against metafor's default control | `rma.uni` | 6.8e-2 | n/a | 20 information |
| Meta-analyses of the cohorts' tables by cell type: REML and z-tests, DerSimonian-Laird and Knapp-Hartung, Paule-Mandel and z-tests (genes of 2 cohorts or more), and the fixed effect (every gene) | `rma.uni`, `p.adjust` | 6.0e-11 | 1e-10 (fixed, DerSimonian-Laird), 1e-6 | 9 pass |
| One study with the Knapp-Hartung adjustment | `rma.uni` | both give the z-test, metafor with a warning | 1e-10 | 1 pass |
| A Paule-Mandel estimate above metafor's search interval: against metafor with `tau2.max = 1e6`, and with its default, where it stops | `rma.uni` | 1.7e-15 | 1e-6 | 1 pass, 1 deliberate difference |
| Zero and infinite standard errors, which CytePool refuses | `rma.uni` | metafor stops for an infinite one, and for a zero one with the fixed effect, DerSimonian-Laird and Paule-Mandel; with REML it fits without Q, I² and H² | n/a | 1 pass, 1 deliberate difference |

Differences are scaled as [above](#reading-the-tables), values as statistics and p-values as
p-values; I² is compared in percent, as metafor reports it
(CytePool reports a proportion), and the number of studies must be equal. A row covers
both tests of one case and method. The REML rows' largest differences, near 1e-10, are
in I²: metafor stops when tau² changes by less than its threshold of 1e-13, which I² in
percent multiplies by 100 / (tau² + s²), s² the typical within-study variance; tau²
itself agrees to 3.3e-13. In one feature of 30 studies metafor's Fisher scoring does
not converge, oscillating, and the remedy its help page suggests, `stepadj = 0.5`,
converges to CytePool's value. With its defaults metafor stops REML when tau² changes by
less than 1e-5 and Paule-Mandel at `uniroot`'s tolerance, about 1.2e-4 on tau²: its
estimates then differ by up to 3e-5 in tau², and the largest scaled difference, 6.8e-2,
is in I² in percent. Deliberate differences:

- The REML estimate is the largest maximum of the restricted likelihood. Where an
  imprecise study lies far from precise ones that agree, the likelihood can have two
  maxima, as it has in 9 of the 120 features of that case. metafor's Fisher scoring from
  the Hedges estimate converges to one of them, and it sets tau² to zero, with the
  warning that Fisher scoring "may have gotten stuck at a local maximum", where the
  likelihood is higher at zero (found by probing). In 2 of those 9 features the other
  maximum is higher, by up to 0.023 in restricted log-likelihood, as metafor's own
  `logLik` at the two estimates shows; they are left out of that case's REML row.
- metafor's Paule-Mandel search stops with an error ("Value of 'tau2.max' too low")
  when the root lies above max(100, 10 mad(y)²): for effects of -40, 0, 0, 0 and 40,
  whose median absolute deviation is 0, that bound is 100 and the root is 799. CytePool
  has no such limit.
- A zero standard error: metafor's fixed-effect, DerSimonian-Laird and Paule-Mandel
  fits stop, its REML fit returns estimates without Q, I² and H²; CytePool refuses it,
  and an infinite one, which metafor refuses too.

Where the Hedges estimate is positive but the restricted likelihood is largest at zero,
CytePool's REML estimate is zero; metafor's step halving approaches zero and stops near
its threshold, at about 1e-6 by default. With standard errors 1e-8 of the spread of the
effects, the two programs agree to 1.5e-14.

(validation-combined-p-values-and-replication)=
### Combined p-values and replication

The script printed 54 checks (47 pass and 7 deliberate differences) in 67.4 s, with a
peak memory of 444 MiB. [Replication](statistics/replication.md#combined-p-values) explains what it
checks.

[`validation/combined_p_values_and_replication.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/combined_p_values_and_replication.py)
checks combined p-values, partial conjunctions, the proportion of true nulls and the
replication of studies. p-values of 2, 5 and 20 studies for 300, 400 and 200 features
(seed 20261102), a third of the features with effects, none, 15% and 5% missing, with
the signs of the effects and the square roots of sample sizes as weights, are combined
by every method and compared with scipy's `combine_pvalues`, feature by feature on the
p-values each has, and with metap's `sumlog`, `sumz` (with the weights), `minimump`,
`wilkinsonp` (r = 2), `maximump` and `logitp`, and Simes's combination as the smallest
Benjamini-Hochberg p-value of `p.adjust`. With directions, R combines the one-sided
p-values of metap's `two2one` for each direction and doubles the smaller; `two2one`
rounds 1 - p / 2 to a double, which loses the digits of p / 2 for small p, so those
rows hold the features whose p-values are all at least 1e-5, and 13 features of
p-values from 5e-324 to 1 - 1e-15 are compared with mpmath at 400 digits instead. The
partial conjunction p-values, for u = 1 to 3, are computed in R from their definition
[@benjamini2008]: `sumlog`, `sumz` or Simes's combination of the n - u + 1 largest
p-values. `proportion_of_true_nulls` is compared with qvalue's `pi0est` on five sets of
60 to 20,000 p-values, one of them rounded to 0.01 so that p-values equal values of
lambda, with its defaults, `smooth.df = 5`, `smooth.log.pi0`, `pi0.method =
"bootstrap"`, one lambda and three other grids. Then three cohorts of 6, 4 and 8 donors,
before and after a stimulation that changes 15% of 700 genes in two cell types, alike in
every cohort up to 50% more or less, each measuring part of the genes, are aligned in a
`CubeCollection` and tested by `limma_voom`; `replication` runs on both cell types
with four settings, among them qvalue's defaults for pi1
(`null_proportion={}`) and pi1 by default (lambda = 0.5), and R computes every column
from the cohorts' tables with metap, qvalue, `p.adjust` and metafor's `rma.uni` (REML
with `control = list(threshold = 1e-13, maxiter = 10000)`, or DerSimonian-Laird). The
figures are compared with what R computes for them. Reference: scipy 1.18.1, mpmath
1.3.0; R 4.4.3, metap 1.1, qvalue 2.38.0, jsonlite 2.0.0; metafor 5.0.1.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Fisher, Pearson, Tippett, Stouffer with weights, Mudholkar and George: statistics and p-values of 2, 5 and 20 studies | `combine_pvalues` | 7.1e-15 | 1e-10 | 15 pass |
| Fisher, Stouffer, Tippett, Wilkinson, the maximum, the logit and Simes, as given and with directions, in the 3 cases | metap, `p.adjust` | 1.2e-12 | 1e-10 | 6 pass |
| Partial conjunctions, u = 1 to 3, by Simes, Fisher and Stouffer, with and without directions | the definition, with metap and `p.adjust` | 7.8e-13 | 1e-10 | 6 pass |
| 13 features of p-values from 5e-324 to 1 - 1e-15, every method, with and without directions | mpmath, 400 digits | 2.0e-14 | 1e-10 | 1 pass |
| pi0: defaults, `smooth.df = 5`, `smooth.log.pi0`, bootstrap, one lambda and three grids, on 4 or 5 sets each | `pi0est` | 1.5e-12 | 1e-10 | 8 pass |
| pi0 with a missing p-value; options both refuse (the log smoother where no p-value reaches a lambda, two values of lambda, lambda = 1) | `pi0est` | 6.7e-14 | 1e-10 | 4 pass |
| Replications, genes: counts, directions, sign agreement, combined and partial conjunction p-values, adjusted | metap, `p.adjust` | 3.1e-15 | 1e-10 | 1 pass |
| Replications, heterogeneity: Q, its p-value, I² and tau², by REML and DerSimonian-Laird | `rma.uni` | 2.0e-12 | 1e-6 (REML), 1e-10 | 2 pass |
| Replications, pairs of studies: genes found, sign agreement, pi1 where qvalue gives it; genes found in at least k studies | `pi0est`, R | 0 | 1e-10, 0 mismatches | 2 pass |
| pi1 by default (`null_proportion` left out): pi0 at lambda = 0.5, 9 values where qvalue gives one | `pi0est(lambda = 0.5)` | 0 | 1e-10 | 1 pass |
| `plot_study_agreement` and `plot_replicated_genes`: the values and bars drawn | `pi0est`, R | 0 | 1e-10, 0 mismatches | 2 pass |
| metap at a p-value of 0 and for one study; pi0est where no p-value reaches the largest lambda (smoother, bootstrap), where its estimate is at most 0, for pi1 of 23 pairs of studies, and for 96 values of lambda | metap, `pi0est` | listed below | n/a | 7 deliberate differences |

Differences are scaled as [above](#reading-the-tables), values as statistics and p-values as
p-values; counts and directions must be equal. qvalue's
smoother is R's `smooth.spline`, which CytePool follows step by step in its search for
the smoothing parameter: the estimates agree to about 1e-12. Deliberate differences:

- metap leaves out a p-value of 0 (with the warning "Some studies omitted") and stops
  with fewer than two p-values; CytePool combines 0 as the limit it is, which makes
  Fisher's p-value 0, and gives a single study its own p-value.
- qvalue stops with an error where no p-value reaches the largest lambda, with the
  smoother ("missing or infinite values in inputs are not allowed") and the bootstrap
  alike; CytePool smooths the zero, and gives the count no variance in the bootstrap.
  This is common for pi1 between studies, whose genes have small p-values: of the 48
  pairs of the replications, qvalue stopped for all 12 with its defaults, 7 of 12 with
  the bootstrap and 4 of 24 with lambda = 0.5, where its estimate was 0.
- qvalue stops where its estimate is at most 0; CytePool gives 0.
- With more than 49 values of lambda, R's `smooth.spline` keeps a knot at some of them
  only; CytePool keeps one at every lambda. For 96 values (0, 0.01, ..., 0.95) the
  estimates differed by 2.6e-5, relative.

(validation-sumrank)=
### SumRank and held-out prediction

The script printed 23 checks (13 pass, 9 information and 1 deliberate difference) in
44.9 s, with a peak memory of 399 MiB. [Replication](statistics/replication.md#sum-of-ranks)
explains what it checks.

[`validation/sumrank.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/sumrank.py)
runs the authors' R code of SumRank, under the Apache-2.0 licence, as they publish it:
`SumRank.R`, `CalibratePValueswithPermutations.R` and `MakingManhattanPlots.R` of their
repository at commit 46392f66, downloaded and checked by their SHA-256. `SumRank.R` loads the unifed package
only for `dirwin.hall`, the Irwin-Hall density, which it evaluates at each gene's capped
sum; the script replaces `dirwin.hall` by the identity, so that SumRank's tables hold
the sums, and the calibration, which compares -log10 of these values, orders the genes
as it would. The data (seed 20261030) are five studies of cases and controls, of 10 to
20 samples, with cell types T and B (one study without B), 300 genes (one study without
20 of them), 30 genes changed in every study and 8 changed 8-fold in one study each,
analysed by `limma_voom`; tables of 4 studies of 60 genes whose p-values take 6 values,
many tied; and three studies of 6 samples, 3 cases each, whose 20 assignments of the
labels each are all analysed. Reference: R 4.4.3, the authors' code, mpmath 1.3.0, scipy
1.18.1, scikit-learn 1.9.1, numpy 2.5.4 and numba 0.68.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| The Irwin-Hall distribution function on a grid of 1 to 200 studies (118 values), and the Irwin-Hall p-values of `fit_sumrank` for 3, 10 and 30 studies (1,199 genes) | mpmath, 200 digits | 6.2e-15 | 1e-12 | 2 pass |
| The SumRank statistics of 5 studies, T and B, up and down, summing every study, 0.65 and 0.5 of them; of 4 studies with tied p-values | the authors' SumRank | 4.2e-15 | 1e-12 | 2 pass |
| Permutation p-values of 40 permutations of the condition, up and down: (1 + the null statistics at most the gene's) / (1 + N) | the authors' null statistics | 0 | 0 | 1 pass |
| The authors' calibration, between the null statistics strictly below the gene's and those at most it, divided by N | `CalibratePValueswithPermutations` | 0 genes outside | 0 | 1 pass |
| The authors' calibrated p-values, against CytePool's | `CalibratePValueswithPermutations` | 0.14 | n/a | 1 deliberate difference |
| Permutation p-values of 1,000 random permutations, 3 and 2 of 3 studies summed | the exact distribution of every assignment of the labels, 8,000 combinations | 0.49 standard errors | 4 standard errors | 2 pass |
| The held-out score of every sample and the AUC of cases against controls, 5 studies, 2 methods | scipy's `rankdata` with UCell's formula; scikit-learn's `roc_auc_score` | 0 | 1e-12 | 2 pass |
| The genes SumRank chose for each held-out study, T and B, up and down | the authors' SumRank on the other studies | 0 of 18 lists differ | 0 | 1 pass |
| The Manhattan plot of `plot_sumrank`: every gene's -log10 p-value in alphabetical order, the threshold line and the genes named, for p-values of 20 permutations summing every study | the authors' `MakeManhattanPlot`, with `plot`, `abline` and `text` recording what it draws | 0 | 1e-12 | 2 pass |
| Calibration, what each method finds, held-out AUC (simulated studies) | the simulated changes | below | n/a | 9 information |

The sums of the authors' code are floating-point sums of each gene's ranks in its own
order, written with 15 digits, so that equal sums can differ in their last digits; the
script compares them as the integers q of q / (2 (G - 1)), G the genes ranked, which
every sum of normalized ranks is. Its calibration then counts the null statistics
strictly below a gene's and some of those equal to it, as the rounding decides, divided
by N; CytePool counts every null statistic at most the gene's, plus one, divided by
1 + N, so that no p-value is 0 and a gene at the cap has 1, which is the deliberate
difference of 0.14. The authors' code stops when it sums the rank of one study (its
`rowSums` takes a matrix), so the held-out runs sum 0.7 of the other studies, 2 of 3 or
4. The standard error of a permutation p-value is taken as sqrt(p (1 - p) / B) for B
permutations, which bounds it whatever the correlation of the genes of a permutation.
`MakeManhattanPlot` draws its line 1e-5 above the highest gene that does not pass;
`plot_sumrank` draws it at that gene.

The information rows, on six simulated studies of 16 samples and 1,200 genes: without
changes, the Irwin-Hall p-values summing every study fell at or below 0.05 and 0.01 in
4.8% and 0.79% of 1,400 tests, and the permutation p-values (40 permutations) in 4.4%
and 0.79%; summing the best 3 of 6, the Irwin-Hall p-values fell below them in 37% and
12%, and the permutation p-values in 4.4% and 0.50%. With 60 genes changed by a log2
fold change of 0.5 in every study and 30 changed 8-fold in one study each, SumRank with
40 permutations found 35 of the 60 at an adjusted p-value below 0.05 and none of the 30,
the meta-analysis with REML and z-tests 35 and none, and with the Knapp-Hartung
adjustment none of either, as its t-test on 5 degrees of freedom gives no p-value small
enough for the adjustment over the genes. Chosen by SumRank, 30 genes up and 30 down
gave a held-out AUC of 1 in every study; chosen by the meta-analysis with Knapp-Hartung,
0.977 on average (0.906 to 1).

## Figures

(validation-plotting)=
### Figures of limma, edgeR, DESeq2, speckle and metafor

The script printed 26 checks (23 pass, 2 information and 1 deliberate difference) in
25.4 s, with a peak memory of 810 MiB. The API reference of {api}`cytepool.plotting`
documents what it checks.

[`validation/plotting.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/plotting.py)
draws each figure of `cytepool.plotting` that follows an R function with CytePool and
with R on the same data, and compares what the two drew: R's base graphics as the
recorded display list of the plot holds them (`recordPlot()`, every primitive drawn
with its coordinates), speckle's ggplot as its built layer data, and CytePool's as the
data of the matplotlib artists. The data (seed 20261101) are counts of 300 genes in 4
control and 4 stimulated samples, 2 genes of which have no counts and 4 an outlying
count; three studies of 4, 5 and 6 donors, tested by `limma_voom` and meta-analysed,
for the forest plots; and the cell numbers of 5 cell types in 12 samples for the
composition. limma-voom runs in R as the limma User's Guide runs it, and voom with
`adaptive.span = TRUE`, as CytePool chooses the span. Reference: R 4.4.3, limma
3.62.1, edgeR 4.4.0, DESeq2 1.46.0, metafor 5.0.1, speckle 1.6.0.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| MDS plots: leading log2 fold changes over the top 50 and the top 500 genes, the 100 genes of largest variance (dimensions 2 and 3), and the samples of a cube after `filterByExpr` and TMM, two of which have fewer than 10 cells; coordinates up to sign, and the percentages of the axis labels | `plotMDS` | 5.4e-15 | 1e-10 | 4 pass |
| voom's mean-variance trend: the points and the LOWESS line | `voom(save.plot = TRUE)` | 6.9e-15 | 1e-10 | 1 pass |
| Sample quality weights | `voomWithQualityWeights(plot = TRUE)` | 3.9e-15 | 1e-6 | 1 pass |
| Residual variances, with a trend and robust moderation (14 outliers marked) and with a constant prior: points, outliers, prior | `plotSA` | 2.6e-9 | 1e-6 | 2 pass |
| MD plot (genes up, down and not significant at 5% FDR), volcano plot and the 10 genes it labels, and p-value histogram, of limma-voom | `plotMD`, `volcanoplot`, `hist` | 2.1e-14; counts identical | 1e-10; 0 | 3 pass |
| Barcode plots: the bars and the enrichment worm, with spans 0.45 and 1 | `barcodeplot` | 4.1e-15 | 1e-10 | 2 pass |
| Dispersions (4 outliers) and quasi-dispersions drawn from DESeq2's and edgeR's own estimates | `plotDispEsts`, `plotQLDisp` | 4.7e-15 | 1e-10 | 2 pass |
| The same, from CytePool's fits of the counts | `plotDispEsts`, `plotQLDisp` | 2.8e-2; 3.3e-4 | n/a | 2 information |
| Quasi-dispersions of the 2 genes without residual degrees of freedom | `plotQLDisp` | edgeR draws them at 0; CytePool leaves them out | n/a | 1 deliberate difference |
| Forest plots of 3 genes, one of them tested by two studies, by REML with z-tests and by DerSimonian-Laird with the Knapp-Hartung adjustment: the studies' intervals, and the pooled estimate and interval | `forest`, `rma.uni` | 4.5e-15 | 1e-6 (REML), 1e-10 | 6 pass |
| Composition: the bar segment of every sample and cell type; proportions by condition, their means and adjusted p-values | `plotCellTypeProps`, `propeller` | 5.0e-16 | 1e-10; 1e-6 | 2 pass |

Every difference, p-values included, is scaled as a statistic is [above](#reading-the-tables). R
writes numbers with 15 significant
digits, which the exact rows reach. The residual variances agree to 2.6e-9 because
limma solves the prior's degrees of freedom to a tolerance (see [Empirical Bayes
moderation](#validation-empirical-bayes)). After robust moderation, `plotSA` marks the variances whose
two-sided p-value as a variance of the prior, adjusted by Benjamini and Hochberg, is
below 0.5 (found by probing), and both mark the same 14. CytePool's dispersions differ
from DESeq2's where DESeq2's line search stops short of the maximum, and its
quasi-dispersions from edgeR's by the exact moments of the unit deviance (see [DESeq2's
dispersions and tests](#validation-deseq2-tests) and [Quasi-likelihood
F-tests](#validation-quasi-likelihood)), so these figures are
compared drawn from R's own estimates, and from CytePool's fits for information.
metafor clips its diamond at the axis limits, which a pooled interval of two studies
with the Knapp-Hartung adjustment exceeds here, so the pooled estimate and interval
are compared with those it writes beside the diamond, from the fit. The deliberate
difference: a gene without residual degrees of freedom, such as one without counts,
has no quasi-dispersion (its deviance and its degrees of freedom are 0); edgeR draws it
at 0, and CytePool leaves it out.

(validation-pca-and-dot-plots)=
### Principal components and dot plots

The script printed 13 checks (11 pass, 1 deliberate difference and 1 information) in
19.8 s, with a peak memory of 671 MiB. [Normalization and
filtering](statistics/normalization_and_filtering.md#samples-at-a-glance), with
[Pseudobulk profiles and cubes](statistics/cubes.md#dot-plot), explains what it checks.

[`validation/pca_and_dot_plots.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pca_and_dot_plots.py)
compares `principal_components` and `plot_pca` with DESeq2's `plotPCA`, run as a black
box with `returnData = TRUE` and drawn, on the same values, and `plot_dotplot` with
scanpy's `dotplot` on the cells a cube was pooled from. The principal components (seed
20261103) are those of the log2 counts per million of negative binomial counts of 300
genes in 12 samples of three conditions and two batches, with the 100 genes of largest
variance and with 500, more than there are; of a matrix whose genes 19 to 22 have equal
variance, with the 20 of largest variance; and of a cube of the counts, two of whose
samples have 5 cells, which R prepares as the edgeR User's Guide does (`filterByExpr`,
`normLibSizes`, `cpm(log = TRUE, prior.count = 2)`). The cells of the dot plots (seed
20261104) are 4 cell types in 6 samples, one sample without cells of one type, with
counts of 40 genes; scanpy normalizes them (`normalize_total`, `log1p`) and
`from_cells` pools them. Reference: R 4.4.3, DESeq2 1.46.0, edgeR 4.4.0, ggplot2
4.0.3; scanpy 1.12.4, anndata 0.13.4, matplotlib 3.11.2.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Principal components: the coordinates up to sign and the shares of variance, with the 100 genes of largest variance (components 1 and 2, and 3 and 4) and with 500; genes of equal variance; a cube's cell type prepared by edgeR | `plotPCA(returnData = TRUE)` | 1.1e-14 | 1e-10 | 3 pass |
| The figure: the points up to sign, the axis labels, equal scales, the samples coloured by condition and batch | `plotPCA` | 6.9e-15 | 1e-10 | 1 pass |
| Two samples: the first component; DESeq2 returns a second of rounding noise (share 9.5e-31), CytePool none | `plotPCA(returnData = TRUE)` | 8.9e-16 | n/a | 1 deliberate difference |
| Components whose sign differs from R's | `plotPCA(returnData = TRUE)` | 2 of 10 | n/a | 1 information |
| Dot plots: the areas and the fractions they encode, and the values the colours show, plain, scaled by gene or by group and over the expressing cells only, for every sample and for 3 samples and 3 cell types in a chosen order, and for genes in groups; the names of the rows and columns, and the key of the areas | `scanpy.pl.dotplot` | 1.0e-15 | 1e-10 | 6 pass |
| The brackets of genes given in groups: their columns and names | `scanpy.pl.dotplot` | identical | 0 | 1 pass |

Every difference is scaled as a statistic is [above](#reading-the-tables). R's `prcomp`, whose
coordinates `plotPCA` returns, leaves the sign of each component arbitrary; CytePool
makes each component's coordinate of largest absolute value positive, so the
coordinates are compared after each
component takes the sign that matches R's. What DESeq2's help page leaves open was
found by probing: the genes are ranked by their variance, genes of equal variance keep
their order (the first are kept), `ntop` beyond the number of genes takes every gene,
the shares are those of all the components (`percentVar`), the axes are labelled with
them rounded to a whole percent, and the figure has equal scales. The dot plots
compare the areas scanpy draws and the values its colours show (`dot_color_df`), and
its fractions (`dot_size_df`) with those that CytePool's areas encode.

Deliberate difference: with two samples, DESeq2's `plotPCA` returns, and draws, a
second component of rounding noise; CytePool returns one component, since the centred
values of n samples have at most n - 1 dimensions, and refuses to draw a second.

## Building blocks on arrays

(validation-distributions)=
### Distributions and special functions

The script printed 51 checks (51 pass) in 12 min 28 s, with a peak memory of 174 MiB.
The API reference of {api}`cytepool.stats.distributions` documents what it checks.

[`validation/distributions.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/distributions.py)
checks every public function of `cytepool.stats.special` and
`cytepool.stats.distributions` on 20,375 points against mpmath evaluated at the exact
binary values of the inputs, with 50 significant digits (70 for intermediate steps);
quantile references are roots of the 70-digit tail probabilities. The grids reach
probabilities of 1e-300 (and log probabilities far below), shapes from 1e-3 to 1e6
(incomplete gamma shapes to 1e18 above the center), degrees of freedom from 0.5 to
1e12 and infinity, hypergeometric populations up to 1e13, batches of 2,000 moderated t
and quasi-likelihood F statistics, and cases where textbook methods lose accuracy. For
the incomplete beta function they also hold the edges of the double range: every
combination of the shapes 5e-324 (the smallest subnormal), 1e-300, 1e-8, 0.5,
1 - 2^-53, 1, 2, 30, 1e8, 1e300 and 1.8e308 (the largest double) with the arguments 0,
5e-324, 1e-300, 1e-8, 0.5, 1 - 2^-53 and 1, and 35 points within 1.5 standard
deviations of the center of shapes from 1e10 to 1e20. The incomplete beta reference
is DLMF 8.17.22 in 70-digit arithmetic, at the edges of the double range in 1,200 to
2,400 bits, and near the center of large shapes, where the fraction converges too
slowly, the beta density integrated numerically; the hypergeometric reference sums
exact ratios in fixed-point integer arithmetic. Reference: mpmath 1.3.0, Python
3.13.12, numpy 2.5.4.

| Check family | Reference | Largest difference | Tolerance | Outcome |
| --- | --- | --- | --- | --- |
| Log-gamma (x > 0, x < 0), digamma (x > 0, x < 0), trigamma, inverse trigamma, log-beta | mpmath | 9.99e-16 (digamma, x < 0) | 1e-15, or 2e-15 for digamma at x < 0 and log-beta | 7 pass |
| Regularized incomplete gamma and beta functions, their complements and logarithms | mpmath | 2.44e-13 | 1e-12 | 8 pass |
| The regularized incomplete beta function, its complement and their logarithms at the edges of the double range and near the center of large shapes | mpmath | 2.04e-13 | 1e-12 | 4 pass |
| The time per value of the incomplete beta function at every combination of 0, 5e-324, 1e-300, 1e-8, 0.5, 1 - 2^-53, 1, 2, 30, 1e8, 1e300 and 1.8e308 as shapes and argument (1,728, those outside the domain included), the best of three calls | timed | 2.1e-4 s (median 2.5e-5 s) | 1e-2 s | 1 pass |
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

(validation-smoothing)=
### LOWESS and spline bases

The script printed 90 checks (87 pass, 1 information and 2 deliberate differences) in
1.5 s, with a peak memory of 175 MiB. [limma-voom and
limma-trend](statistics/limma_voom.md#voom-span) explains what it checks.

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

`mixed_models.py --benchmark`, run on 2026-10-28, on 20,000 genes with observation
weights and the fixed effects `~ condition + age` (3 conditions), at 120 samples with three structures of
random effects and at 18 samples, a study small enough for dream's default to use
Kenward-Roger's method, with one random intercept: CytePool on one thread, the fits the
median of three runs and the tests one run, after a first call that compiled them, and
lmerTest's `lmer` and `summary` (Satterthwaite's df for every coefficient, with lme4's
default tolerances) on 100 of the genes, after one fit that was not timed. The peak
resident memory of the run was 729 MiB.

| Random effects | Samples | `fit_mixed_model` (s) | Satterthwaite's tests (s) | Kenward-Roger's tests (s) | lmerTest, per gene (s) | lmerTest, 20,000 genes at that rate (min) |
| --- | --- | --- | --- | --- | --- | --- |
| One random intercept: 40 donors x 3 samples | 120 | 0.49 | 0.06 | 8.49 | 0.026 | 8.8 |
| Two crossed: 40 donors, 6 batches | 120 | 1.18 | 0.07 | 9.70 | 0.033 | 11.1 |
| Two nested: 20 donors x 3 samples x 2 replicates | 120 | 3.03 | 0.08 | 9.17 | 0.029 | 9.7 |
| One random intercept: 6 donors x 3 samples | 18 | 0.20 | 0.06 | 0.15 | 0.021 | 7.1 |

Kenward-Roger's tests cost O(n³) operations per gene for n samples: in the small study
they took 0.15 s, against 0.06 s for Satterthwaite's, and at 120 samples more than 100
times as long as Satterthwaite's.

On fgsea's example data, 12,000 genes and 586 sets of 15 to 500 genes, with its
defaults, fgsea's `fgsea` took 2.9 s and `gsea` 4.1 s, both on one thread, after a
first call of `gsea` that compiled it; with plain permutations (10,000), fgsea took
1.9 s and `gsea` 1.5 s, on 2 threads.

`pathway_scores.py --benchmark` scores 500 sets of 15 to 500 genes in 50 samples of
15,000 genes, log-expression values or, for the Poisson kernel, negative binomial
counts. GSVA ran on one thread, as R measured it; CytePool on 2 threads, the best of
three runs after a first call that compiled it. The peak resident memory of the run was
952 MiB.

| Method | GSVA (s) | CytePool (s) |
| --- | --- | --- |
| GSVA, Gaussian kernel | 4.13 | 0.83 |
| GSVA, Poisson kernel | 17.61 | 7.32 |
| ssGSEA | 2.21 | 0.22 |
| Combined z-scores | 0.67 | 0.15 |
| PLAGE | 1.95 | 0.96 |

`sumrank.py --benchmark` permutes the condition of 10 studies of 40 samples and 10,000
genes 10 times: each permutation and study took 0.21 s with `limma_voom` and 0.14 s with
`limma_trend`, the analysis that a permutation runs again, on 2 threads, with a peak
resident memory of 567 MiB for the run.

`adjust_p_values` with Hommel's method took 0.006 s on the 100,000 p-values of
`multiple_testing.py`. In `distributions.py`, mpmath took 746 s to compute the
references, 128 s of them at the edges of the double range, and CytePool 1.8 s to
evaluate them, including compilation; the incomplete beta function took at most
2.1e-4 s for any value at those edges, with the time of a call from Python.

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
selection of samples, since every chunk of this file holds every sample. A read that
covers a whole group is hashed, to check it against the digest stored with the file,
and a selection's groups are hashed as they are read, for the selection's own
fingerprint.

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
# lme4 with lmerTest, in the R environment of its own described below.
RSCRIPT=$REF/r-mixed/bin/Rscript python validation/mixed_models.py
RSCRIPT=$REF/r-mixed/bin/Rscript python validation/mixed_models.py --benchmark
python validation/empirical_bayes.py
python validation/gene_set_tests.py
python validation/rotation_gene_set_tests.py
python validation/gene_set_files_and_ora.py
RSCRIPT=$REF/r-gsea/bin/Rscript python validation/gsea.py
RSCRIPT=$REF/r-gsea/bin/Rscript python validation/pathway_scores.py
RSCRIPT=$REF/r-gsea/bin/Rscript python validation/pathway_scores.py --benchmark
# Downloads CollecTRI and progeny into ~/.cache/cytepool/networks on its first run.
RSCRIPT=$REF/r-progeny/bin/Rscript python validation/activity.py
python validation/voom.py
python validation/sample_quality_weights.py
python validation/negative_binomial_fits.py
python validation/quasi_likelihood.py
python validation/deseq2_tests.py
python validation/differential_expression.py
python validation/differential_expression_options.py
# Downloads 77 MB from GEO on its first run, into ~/.cache/cytepool-validation/GSE96583
# unless --data or GSE96583_DATA names another directory outside the repository.
python validation/interferon_stimulated_pbmcs.py
RSCRIPT=$REF/r-speckle/bin/Rscript python validation/composition_tests.py
RSCRIPT=$REF/r-metafor/bin/Rscript python validation/meta_analysis.py
RSCRIPT=$REF/r-meta/bin/Rscript METAFOR_RSCRIPT=$REF/r-metafor/bin/Rscript \
    python validation/combined_p_values_and_replication.py
RSCRIPT=$REF/r/bin/Rscript METAFOR_RSCRIPT=$REF/r-metafor/bin/Rscript \
    SPECKLE_RSCRIPT=$REF/r-speckle/bin/Rscript python validation/plotting.py
SCANPY_PYTHON=$REF/python-scanpy/bin/python python validation/pca_and_dot_plots.py
# variancePartition, in an R environment of its own with lme4 1.1-38 (below).
RSCRIPT=$REF/r-dream/bin/Rscript python validation/dream.py
RSCRIPT=$REF/r-dream/bin/Rscript python validation/kenward_roger_and_f_tests.py
RSCRIPT=$REF/r-dream/bin/Rscript python validation/variance_partitioning.py
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
REFERENCE_PYTHON=$REF/python-scanpy/bin/python python validation/cube_collection.py
# Downloads the authors' SumRank code (13 kB) on its first run, into
# ~/.cache/cytepool-validation/sumrank unless --code or SUMRANK_CODE names another.
python validation/sumrank.py
python validation/sumrank.py --benchmark
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

`mixed_models.py` needs lmerTest, which runs on the same lme4 2.0.6 in an R
environment of its own: R 4.4.3, lme4 2.0.6, lmerTest 3.2.1, numDeriv 2016.8.1.1 and
jsonlite 2.0.0, from conda-forge:

```bash
conda create -p $REF/r-mixed -c conda-forge r-base=4.4.3 r-lme4=2.0_6 \
    r-lmertest=3.2_1 r-numderiv=2016.8_1.1 r-jsonlite=2.0.0
```

metap 1.1 (CRAN) and qvalue 2.38.0 (Bioconductor 3.20), which
`combined_p_values_and_replication.py` runs, have an R environment of their own, with R
4.4.3 and jsonlite 2.0.0, from conda-forge and bioconda:

```bash
conda create -p $REF/r-meta -c conda-forge -c bioconda r-base=4.4.3 r-metap=1.1 \
    bioconductor-qvalue=2.38.0 r-jsonlite=2.0.0
```

metafor 5.0.1 (CRAN), which `meta_analysis.py` and
`combined_p_values_and_replication.py` run, has one too, with R 4.4.3 and jsonlite
2.0.0:

```bash
conda create -p $REF/r-metafor -c conda-forge r-base=4.4.3 r-metafor=5.0.1 \
    r-jsonlite=2.0.0
```

`dream.py` runs variancePartition 1.36.2 (Bioconductor 3.20) with limma 3.62.1, in an
R environment of its own on lme4 1.1-38, as variancePartition 1.36.2's `dream` fails
for every gene with lme4 2.0 ("Unable to extract deviance function from model fit"):
R 4.4.3, lme4 1.1-38, lmerTest 3.1-3, pbkrtest 0.5.3, Matrix 1.7-4, RcppEigen
0.3.4.0.2, minqa 1.2.8, nloptr 2.2.1, BiocParallel 1.40.0, jsonlite 2.0.0 and ggplot2
4.0.3 (which variancePartition's figures use), from conda-forge and bioconda. lme4 1.1-35.5's conda build crashed in `lmer` with these
versions of Matrix and RcppEigen:

```bash
conda create -p $REF/r-dream -c conda-forge -c bioconda r-base=4.4.3 \
    bioconductor-variancepartition=1.36.2 bioconductor-limma=3.62.1 r-lme4=1.1_38 \
    r-lmertest=3.1_3 r-pbkrtest=0.5.3 r-matrix=1.7_4 r-rcppeigen=0.3.4.0.2 \
    r-minqa=1.2.8 r-nloptr=2.2.1 r-jsonlite=2.0.0
```

fgsea 1.32.2 (Bioconductor 3.20), which `gsea.py` runs, and GSVA 2.0.0 (Bioconductor
3.20), which `pathway_scores.py` runs, have an R environment of their own, with R 4.4.3, limma 3.62.1, GSVA 2.0.0, BiocParallel 1.40.0, data.table 1.18.6.1 and
jsonlite 2.0.0, from conda-forge and bioconda:

```bash
conda create -p $REF/r-gsea -c conda-forge -c bioconda r-base=4.4.3 \
    bioconductor-fgsea=1.32.2 bioconductor-gsva=2.0.0 bioconductor-limma=3.62.1 \
    r-data.table=1.18.6.1 r-jsonlite=2.0.0
```

progeny 1.28.0 (Bioconductor 3.20), which `activity.py` runs for PROGENy's weights, has
one too, with R 4.4.3 and jsonlite 2.0.0:

```bash
conda create -p $REF/r-progeny -c conda-forge -c bioconda r-base=4.4.3 \
    bioconductor-progeny=1.28.0 r-jsonlite=2.0.0
```

The Python environments use Python 3.13.12. scanpy has its own environment, and
`cube_files.py` ran CytePool on Python 3.14.4 for the zstd cases:

```bash
uv venv $REF/python --python 3.13.12
uv pip install --python $REF/python/bin/python numpy==2.5.4 numba==0.68.0 \
    scipy==1.18.1 mpmath==1.3.0 zarr==3.4.1 numcodecs==0.17.1 pandas==3.1.0 \
    anndata==0.13.4 decoupler==2.2.0 matplotlib==3.11.2
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
against. CytePool implements the methods from their papers and documentation, and
conventions that the documentation leaves open were found by running the packages on
chosen inputs; the source of a package under a copyleft licence is never read, and
what CytePool adapts or follows from permissively licensed ones (fgsea, decoupler,
the authors' code of SumRank, and the conventions of scanpy's dot plot that its
documentation leaves open) `THIRD_PARTY_NOTICES.md` records.
