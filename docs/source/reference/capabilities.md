---
description: Every method CytePool implements, with its function, the reference it is checked against and its version, the tolerance tier, and the validation script; then what is not implemented yet.
---

# Capability matrix

Every method below is compared with an independent reference, usually the package that
defined it, run as a separate program on the same inputs. The tier says how closely:
**exact**, a relative difference of at most 1e-10, for closed-form results; **close**,
at most 1e-6, for iterative fits; **calibrated**, within simulation error with the same
calls, for methods that draw random rotations. Where CytePool differs from the reference
on purpose, the [Validation](../validation.md) page reports each difference with its
size; its sections give every check.

## Implemented

| Method | CytePool | Checked against | Tier | Script |
| --- | --- | --- | --- | --- |
| Pseudobulk profiles pooled from cells | {api}`from_cells` | scanpy 1.12.4, decoupler 2.2.0 | exact | [`pool_cells.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pool_cells.py) |
| Bulk counts as a cube | {api}`from_bulk` | decoupler 2.2.0 | exact | [`pool_cells.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/pool_cells.py) |
| Selecting, merging groups, pooling samples; means, variances, detection | {api}`Cube` | scanpy 1.12.4 | exact | [`cube.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube.py) |
| Cube files | {api}`write_cube`, {api}`read_cube` | zarr-python 3.4.1 | exact | [`cube_files.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_files.py) |
| Reads of selected features from cube files | {api}`read_cube`, {api}`Cube.select` | zarr-python 3.4.1 | exact | [`cube_reads.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_reads.py) |
| Fingerprints of cubes | {api}`Cube.fingerprint`, {api}`Cube.digests` | the documented encoding, with zarr-python 3.4.1 | exact | [`cube_fingerprints.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/cube_fingerprints.py) |
| AnnData and MuData layouts of scanpy, decoupler and LIANA | {api}`to_anndata`, {api}`from_anndata`, {api}`to_mudata`, {api}`from_mudata` | scanpy 1.12.4, decoupler 2.2.0, LIANA 1.10.0, anndata 0.13.4, mudata 0.4.1 | exact | [`anndata_conversion.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/anndata_conversion.py) |
| TMM, upper-quartile and RLE factors; CPM and log-CPM; `filterByExpr` | {api}`tmm_factors`, {api}`upper_quartile_factors`, {api}`rle_factors`, {api}`cpm`, {api}`log_cpm`, {api}`filter_by_expression` | edgeR 4.4.0 | exact | [`normalization.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/normalization.py) |
| DESeq2's median-of-ratios size factors | {api}`median_ratio_size_factors` | DESeq2 1.46.0 | exact | [`normalization.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/normalization.py) |
| Model matrices from R formulas, named contrasts, design checks | {api}`Design`, {api}`model_matrix`, {api}`make_contrasts`, {api}`Design.check` | R 4.4.3, limma 3.62.1, lme4 2.0.6 | exact | [`design.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/design.py) |
| Linear models of every gene, contrasts (limma's `lmFit`, `contrasts.fit`) | {api}`fit_linear_model`, {api}`contrasts_fit` | limma 3.62.1 | exact | [`linear_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/linear_models.py) |
| Mixed models with random intercepts by REML or ML, Satterthwaite's t-tests | {api}`fit_mixed_model`, {api}`fixed_effect_tests` | lme4 2.0.6, lmerTest 3.2.1 | close | [`mixed_models.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/mixed_models.py) |
| voom precision weights, limma's span rule | {api}`voom`, {api}`choose_lowess_span` | limma 3.62.1, edgeR 4.4.0 | exact; close on rarefied counts | [`voom.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/voom.py) |
| Sample quality weights (`arrayWeights`, `voomWithQualityWeights`) | {api}`sample_quality_weights`, {api}`voom_with_quality_weights` | limma 3.62.1, edgeR 4.4.0 | close | [`sample_quality_weights.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/sample_quality_weights.py) |
| Negative binomial models of every gene (edgeR's `glmFit`), deviances | {api}`fit_negative_binomial_model`, {api}`negative_binomial_deviance`, {api}`negative_binomial_unit_deviance` | edgeR 4.4.0, mpmath 1.3.0 | close | [`negative_binomial_fits.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/negative_binomial_fits.py) |
| Quasi-likelihood F-tests (edgeR 4's `glmQLFit`, `glmQLFTest`) | {api}`fit_quasi_likelihood`, {api}`quasi_likelihood_f_test`, {api}`common_dispersion`, {api}`average_log_cpm` | edgeR 4.4.0, limma 3.62.1, mpmath 1.3.0 | close | [`quasi_likelihood.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/quasi_likelihood.py) |
| DESeq2's dispersions, Wald tests and likelihood ratio tests | {api}`fit_deseq2`, {api}`deseq2_dispersions`, {api}`deseq2_wald_test`, {api}`deseq2_likelihood_ratio_test` | DESeq2 1.46.0, PyDESeq2 0.5.4 | close | [`deseq2_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/deseq2_tests.py) |
| Empirical Bayes moderation, moderated t and F, B-statistics, TREAT | {api}`moderated_statistics`, {api}`treat`, {api}`squeeze_variances`, {api}`fit_f_distribution`, {api}`fit_f_distribution_robust`, {api}`fit_f_distribution_likelihood` | limma 3.62.1 | exact; close for the robust and likelihood priors | [`empirical_bayes.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/empirical_bayes.py) |
| Gene-set tests camera and fry | {api}`camera`, {api}`fry` | limma 3.62.1 | exact; close with robust moderation; 1e-3 for fry's default standardization, which differs on purpose | [`gene_set_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/gene_set_tests.py) |
| Rotation gene-set tests roast, mroast and romer | {api}`roast`, {api}`mroast`, {api}`romer` | limma 3.62.1 | calibrated; exact given limma's rotations | [`rotation_gene_set_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/rotation_gene_set_tests.py) |
| Adjustment for multiple testing (R's `p.adjust`) | {api}`adjust_p_values` | R 4.4.3 | exact | [`multiple_testing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/multiple_testing.py) |
| limma-voom and limma-trend on every cell type of a cube | {api}`limma_voom`, {api}`limma_trend` | limma 3.62.1, edgeR 4.4.0 | exact; close with robust moderation | [`differential_expression.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression.py) |
| Options for pseudobulk data: quality weights, weights from cell numbers, a minimum count, adjustment across cell types | {api}`limma_voom`, {api}`limma_trend` | limma 3.62.1, edgeR 4.4.0 | exact; close with quality weights | [`differential_expression_options.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/differential_expression_options.py) |
| dream on every cell type of a cube | {api}`dream_voom`, {api}`voom_with_dream_weights`, {api}`moderated_fixed_effect_tests` | variancePartition 1.36.2, limma 3.62.1, lme4 1.1-38, lmerTest 3.1-3 | close | [`dream.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/dream.py) |
| Variance partitioning (`fitExtractVarPartModel`), on every cell type of a cube, and its figures (`plotVarPart`, `plotPercentBars`) | {api}`variance_fractions`, {api}`variance_partitioning`, {api}`plot_variance_fractions`, {api}`plot_gene_variance_fractions` | variancePartition 1.36.2, lme4 1.1-38, ggplot2 4.0.3 | close; exact at given estimates and for linear models | [`variance_partitioning.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/variance_partitioning.py) |
| limma-voom on published single-cell data (Kang et al. 2018) | {api}`from_cells`, {api}`limma_voom` | limma 3.62.1, edgeR 4.4.0 | exact | [`interferon_stimulated_pbmcs.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/interferon_stimulated_pbmcs.py) |
| Cell-type composition (speckle's propeller) | {api}`propeller` | speckle 1.6.0, limma 3.62.1 | exact; close for the robust prior | [`composition_tests.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/composition_tests.py) |
| Meta-analysis by fixed and random effects (metafor's `rma.uni`) | {api}`meta_analysis`, {api}`differential_expression_meta_analysis` | metafor 5.0.1 | exact; close for REML and Paule-Mandel | [`meta_analysis.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/meta_analysis.py) |
| Figures of limma, edgeR, DESeq2, speckle and metafor (MDS, voom's trend, plotSA, MD and volcano plots, barcode plots, dispersions, composition, forest plots) | {api}`cytepool.plotting` | limma 3.62.1, edgeR 4.4.0, DESeq2 1.46.0, speckle 1.6.0, metafor 5.0.1 | exact for what is drawn; close for plotSA's prior | [`plotting.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/plotting.py) |
| Tails and quantiles of distributions, special functions | {api}`cytepool.stats.distributions`, {api}`cytepool.stats.special` | mpmath 1.3.0 | exact (1e-15 to 1e-12) | [`distributions.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/distributions.py) |
| LOWESS, natural spline bases | {api}`lowess`, {api}`natural_spline_basis` | R 4.4.3 | close for LOWESS; exact for spline bases | [`smoothing.py`](https://github.com/NygenAnalytics/CytePool/blob/master/validation/smoothing.py) |

## Not implemented yet

These methods come up in the questions that studies ask ([Overview](../index.md)) and
have no CytePool function yet:

- random slopes, Kenward-Roger's degrees of freedom and F-tests of several contrasts in
  mixed models;
- composition tests with precision weights or random effects (crumblr), and Bayesian
  count models of composition;
- pathway and transcription-factor activity scores;
- ligand-receptor scoring and differential cell-cell interactions;
- dose-response fits, benchmark doses and drug synergy;
- factorization of samples across cell types;
- shrinkage of effects across conditions (mashr) and meta-regression;
- edgeR's quasi-likelihood F-tests and DESeq2's tests on every cell type of a cube, and
  DESeq2's shrunken fold changes.
