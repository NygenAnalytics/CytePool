---
description: Which genes change between conditions in each cell type, tested by limma-voom or limma-trend on the pseudobulk profiles of every cell type of a cube.
---

# limma-voom and limma-trend

This chapter answers the question that [Normalization and filtering](normalization_and_filtering.md) prepared the counts for: which genes change between conditions in each cell type? {api}`limma_voom` answers it by default. It runs limma-voom [@law2014; @ritchie2015] on every cell type of a cube, one cell type at a time, and {api}`limma_trend` runs limma-trend the same way. [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) shows both on published data.

(limma-voom-steps)=
## What runs in each cell type

Each cell type is fitted on its own, with its own samples, genes, normalization factors, mean-variance trend and prior, in the steps of the RNA-seq workflows that the limma and edgeR User's Guides describe:

1. **Samples.** The samples with too few cells or counts in the cell type are left out ([Samples with few cells or counts](#samples-with-few-cells)). The cell type's design matrix keeps the rows of the full design matrix that belong to the remaining samples, so every column keeps its meaning.
2. **Genes.** edgeR's `filterByExpr`, given that design matrix, keeps the genes with enough counts, and the library sizes become the totals over the genes kept ([Filtering by expression](#filtering-by-expression)).
3. **Normalization.** TMM factors scale the library sizes ([Normalization factors and size factors](#normalization-and-size-factors)).
4. **voom.** voom turns the counts into log-CPM values with a precision weight for each value ([below](#voom)), and a weighted linear model is fitted to every gene ([Designs, linear models and contrasts](linear_models.md)).
5. **Tests.** Each contrast is estimated from the fit ([Contrasts](#contrasts)), each gene's residual variance is moderated by empirical Bayes ([Empirical Bayes moderation](empirical_bayes.md)), sets of contrasts are tested together where asked ([F-tests of several contrasts](#limma-f-tests)), and the p-values of each contrast and F-test are adjusted by Benjamini and Hochberg's method over the genes of the cell type ([Multiple testing](#adjusting-across-cell-types)).

In R these are `filterByExpr` and `calcNormFactors` of edgeR, then `voom`, `lmFit`, `contrasts.fit`, `eBayes` and `topTable` of limma, and the results agree with theirs ([Checks and further reading](#limma-voom-checks)).

With fewer samples, the design matrix of a cell type can lose rank, for instance when all the samples of one donor were left out. The cell type is then fitted on the columns kept by R's `qr()`, and each contrast is tested where its samples can estimate it; the others are reported ([Designs that cannot answer the question](#estimable-contrasts)). A cell type that cannot be fitted at all, such as one left without residual degrees of freedom, gets the reason in the result's summary ({api}`DifferentialExpressionResult`), and the other cell types are fitted as usual.

(voom)=
## voom

Log counts are noisier at low counts than at high ones, and a linear model that weighs every observation alike gives the noisy ones too much say. voom estimates the mean-variance trend of the log counts and gives every observation a precision weight from it, so that the weighted linear model and the empirical Bayes moderation of limma apply to counts [@law2014].

For counts $r_{gi}$ of gene $g$ in sample $i$ and effective library sizes $R_i$ (library sizes times normalization factors), voom works with log-CPM values, log2 counts per million:

$$
y_{gi} = \log_2 \frac{r_{gi} + 0.5}{R_i + 1}\, 10^6 .
$$

The design is fitted to every gene by least squares, which gives fitted values $\hat\mu_{gi}$ and residual standard deviations $s_g$. The square roots $s_g^{1/2}$ are smoothed by LOWESS ({api}`lowess`; @cleveland1979) against each gene's average log count, $x_g = \bar y_g + \overline{\log_2(R_i + 1)} - \log_2 10^6$, and each observation gets the weight

$$
w_{gi} = \operatorname{lo}(\lambda_{gi})^{-4}, \qquad \lambda_{gi} = \log_2\!\left(2^{\hat\mu_{gi}} (R_i + 1) / 10^6\right),
$$

the inverse of the variance that the trend predicts at its fitted log count. {api}`voom` returns the log-CPM values, the weights and the trend in a {api}`VoomResult`.

(voom-span)=
### The span

Each local fit of the LOWESS uses a share of the genes, the span. By default the span is chosen from the number of genes $n$ by limma's rule ({api}`choose_lowess_span`): 1 up to 50 genes, and

$$
\text{span} = 0.3 + 0.7\,(50 / n)^{1/3}
$$

beyond, so 0.65 at 400 genes and 0.395 at 20,000: the part of the span above 0.3 falls in proportion to $n^{-1/3}$, as the width of a histogram's bins does in the rules of @scott1979 and @freedman1981. limma's `voom` chooses the span this way by default since limma 3.66; its default before was 0.5, which `span=0.5` reproduces. In a cube, each cell type's span follows the number of its genes that pass the filter, and the cell type's record keeps it: in the data of @kang2018 the spans run from 0.44 in the CD4 T cells (6,295 genes) to 0.59 in the megakaryocytes (704 genes).

(limma-trend)=
## limma-trend

limma-trend uses no precision weights. Instead, the empirical Bayes prior for the variances follows the mean-variance trend. {api}`limma_trend` takes the steps above, but the counts become log-CPM values with a prior count of 3, as the limma User's Guide computes them for limma-trend; every gene is fitted by unweighted least squares, or with sample weights when they are asked for; and the prior variance follows the genes' average log-CPM (`eBayes(trend = TRUE)`; [Estimating the prior](#estimating-the-prior)). The standard errors of contrasts are then exact, as limma's are, since every gene has the same weights.

voom applies the trend to each observation, with its sample's library size; limma-trend applies it to each gene, and assumes that library sizes are not very different. limma's documentation says that voom should be more powerful when library sizes are very different, and that otherwise the two methods should give similar results (voom's help page; @law2014). The limma User's Guide (sections 15.4 and 15.5) says that limma-trend usually works well when the largest library size is not more than about three times the smallest. Pseudobulk library sizes grow with the number of cells pooled, which can differ severalfold between samples, so {api}`limma_voom` is the default.

(prior-weights-from-cells)=
## Prior weights from the number of cells

A pseudobulk profile sums the counts of its cells, so its log-CPM values estimate the mean expression of those cells, whose variance falls as one over their number: the precision of a profile is proportional to its number of cells, as the dreamlet paper argues [@hoffman2026a]. `prior_weights="n_cells"` makes the number of cells of each sample in the cell type its prior weight, as dreamlet does. In {api}`limma_voom` they weight the fits from which voom estimates its trend, and multiply its precision weights; in {api}`limma_trend` they are sample weights of the linear model. Only their ratios within a cell type count, so weights ten times larger change nothing. The cube must record its cell numbers, as one from {api}`from_cells` does.

(sample-quality-weights)=
## Sample quality weights

Some samples follow the model less closely than others: pooled from few cells, sequenced less deeply, or taken from an outlying donor. Removing them loses power, and keeping them adds noise. Sample quality weights down-weight them instead [@ritchie2006; @liu2015]. For pseudobulk data, a limma author recommends sample weights, estimated by edgeR's `voomLmFit`, in preference to a covariate of quality such as RNA integrity [@smyth2025]. For gene $g$ and sample $j$, with prior weights $w_{gj}$, the model gives the mean and the variance

$$
E(y_{gj}) = x_j' b_g, \qquad \operatorname{var}(y_{gj}) = \sigma_g^2\, e^{u_j} / w_{gj},
$$

where the log variance factors $u = Z\gamma$ are common to all genes and sum to zero. The sample quality weights are $e^{-u_j}$, whose product is one. {api}`sample_quality_weights` estimates $\gamma$ by restricted maximum likelihood (REML; @smyth2002) or gene by gene [@ritchie2006], as limma's `arrayWeights` does, with a prior worth 10 genes at equal weights (`prior_n`) that keeps the weights near one when the evidence is weak.

{api}`voom_with_quality_weights` combines them with voom as limma's `voomWithQualityWeights` does: voom; sample quality weights from its log-CPM values and weights; voom again, with those as prior weights; sample quality weights again, from the new trend's weights; and the new trend's weights times the new sample quality weights. `quality_weights=True` runs it in every cell type of {api}`limma_voom`, gene by gene, as `voomWithQualityWeights` does by default. {api}`limma_trend` estimates the weights from its log-CPM values by REML, which `arrayWeights` chooses for data without prior weights or missing values, and fits them as sample weights. The result's sample table keeps each sample's weight. Where the weights cannot be estimated, for example when fewer than two residual degrees of freedom remain, each is one, and a warning names the cell type.

In a design that pairs two samples per donor, such as `~ donor + condition`, the two samples of a donor have equal and opposite residuals (up to voom's precision weights), so an outlying sample pulls its partner's weight down with it: limma-trend gives both samples the same weight, and limma-voom gives them similar ones.

(limma-f-tests)=
## F-tests of several contrasts

Does a gene change at any dose, or anywhere along a curve of dose? A contrast asks about one difference, while a dose factor or a spline of dose has several coefficients to test together. `f_tests` tests a set of contrasts at once by limma's moderated F-statistic ([Moderated F-statistics](empirical_bayes.md#moderated-f-statistics)), as limma's `topTable(coef = ...)` does: `contrasts=["dose_s1", "dose_s2", "dose_s3"], f_tests={"dose": ["dose_s1", "dose_s2", "dose_s3"]}` tests three columns of the design at once, as a contrast named after a design column tests that column. Each cell type runs the F-tests whose contrasts its samples can all estimate, and the result lists the others with the reason, as it does for contrasts. A cell type that lost a dose group, for example because the highest dose killed its cells, runs an F-test of the doses that remain when one is named in the same call.

One F-test per gene controls the false discovery rate of the question "does the gene respond at any dose?". The genes that any single contrast calls do not: each contrast's list is controlled on its own, and their union is not. A significant F-test says that the gene responds, and the log fold changes of its contrasts, in the same table, say at which doses. {api}`DifferentialExpressionResult` lists the table's columns; the numerator degrees of freedom of each F-test, the number of linearly independent contrasts, are in the cell type's entry of `result.provenance["groups"]` (`f_test_df1`).

Two designs answer the dose question:

- **Every dose against the vehicle.** With the dose as a factor whose reference level is the vehicle, the contrasts are the dose columns, and their F-test asks whether a gene's mean differs from the vehicle's at any dose. With the dose alone in the design, this F-test is a one-way ANOVA of the dose groups with moderated variances, the moderated counterpart of BMDExpress's ANOVA prefilter. A contrast that combines others, such as one dose against another, does not change the default test, since the numerator degrees of freedom count the linearly independent contrasts; with `contrast_standard_errors="limma"`, limma's correlation under voom's weights does change it.
- **A natural spline of dose.** The columns of {api}`natural_spline_basis`, computed from the doses of every sample and added to the sample table with `cube.samples.with_columns`, enter the design as covariates, `~ dose_s1 + dose_s2 + dose_s3`, and their F-test asks whether expression changes anywhere along a smooth curve of dose, as the limma User's Guide tests a spline of time (section 9.6.2). The curve is smooth in the values given, and the knots lie at their quantiles. Doses spaced by a constant factor are better given as logarithms, which also keep the columns well conditioned when the doses span many decades; the vehicle's dose of 0 has no logarithm, so it needs a finite value first, such as one dose step below the lowest dose, which then becomes the lower boundary knot. A spline of `df` columns needs at least `df + 1` distinct doses, the vehicle included. The columns' coefficients, in the table's `log2_fold_change_dose_s1` and the like, mean nothing one by one and change with the basis: read the F-test, and take fold changes from contrasts of dose groups.

An F-test reads the cell type's moderated fit, so it needs no second fit. Its denominator degrees of freedom (`df`) are each gene's residual plus prior degrees of freedom, as limma's `eBayes` gives them. Where the genes' residual variances scatter about the prior no more than sampling alone would make them, the prior degrees of freedom are infinite, and so is `df`: the F-statistic is then referred to a chi-squared distribution on the numerator degrees of freedom, divided by them. Under voom's weights the default F-statistic is exact, from each gene's own correlation of the contrasts ([Moderated F-statistics](empirical_bayes.md#moderated-f-statistics)).

(limma-voom-defaults)=
## Defaults and choices

The defaults follow limma's workflow for RNA-seq, with two additions: samples with fewer than 10 cells in a cell type are left out, and contrasts get exact standard errors ([Contrasts](#contrasts)).

- **voom or trend.** {api}`limma_voom` by default; {api}`limma_trend` when library sizes lie within about threefold of each other ([limma-trend](#limma-trend)).
- **Samples, genes and normalization.** `min_cells`, `min_counts`, `expression_filter` and `normalization` set the samples, genes and library sizes of each cell type's fit, and `prior_count` the prior count of limma-trend, as [Normalization and filtering](#normalization-defaults) gives them.
- **The span**, of {api}`limma_voom`: chosen in each cell type by default ([The span](#voom-span)); `span=0.5` reproduces limma's voom before limma 3.66.
- **Sample weights.** `quality_weights=True` estimates a weight for each sample from the data, and a mapping changes the settings of the estimate, such as `{"method": "reml"}`; `prior_weights="n_cells"` takes the weights from the numbers of cells. The two are alternatives: limma's `voomWithQualityWeights` takes no prior weights, and edgeR's `voomLmFit` refuses prior weights together with estimated sample weights, as {api}`limma_voom` and {api}`limma_trend` do.
- **Hypervariable genes.** `robust=True` estimates the prior robustly, so that genes with outlying variances do not distort it [@phipson2016] ([Estimating the prior](#estimating-the-prior)).
- **Contrasts of several coefficients.** Their standard errors are exact by default; in {api}`limma_voom`, `contrast_standard_errors="limma"` reproduces limma's approximation ([Contrasts](#contrasts)). In limma-trend the two agree.
- **Several contrasts at once.** `f_tests` names sets of contrasts to test together, by F-tests that are exact under voom's weights by default ([F-tests of several contrasts](#limma-f-tests)). `contrast_standard_errors="limma"` gives limma's F-statistics, which under voom's weights differ for any F-test of several contrasts, also when each contrast is a single coefficient.
- **Many cell types.** p-values are adjusted within each cell type and contrast; `adjust_across_groups=True` adds a column adjusted over every cell type ([Multiple testing](#adjusting-across-cell-types)).

(limma-voom-limits)=
## Limits and differences

- **Fixed effects only.** A design with random effects is refused. Several samples of a donor are paired in the design (`~ donor + condition`), or the donor gets a random effect with {api}`dream_voom` ([Mixed models and dream](mixed_models.md)). limma's `duplicateCorrelation` and voom's block correlation are not implemented.
- **No voomLmFit.** limma's documentation now recommends edgeR's `voomLmFit` over `voom`, in particular to estimate a correlation within blocks or for sparse counts with many zeros (voom's help page); `voomLmFit` is not implemented.
- **Normalization.** Normalization factors are TMM factors or none; voom's microarray-style normalization of log-CPM values (`normalize.method`) is not offered.
- **Groups of samples.** {api}`limma_voom` and {api}`limma_trend` give each sample a weight of its own. limma's variance models for groups of samples (`var.group`, `var.design`) are offered by {api}`sample_quality_weights` and {api}`voom_with_quality_weights`, on one matrix.
- **Rank-deficient cell types.** Every contrast that a cell type's samples can estimate is tested, and so is every F-test of such contrasts, where limma's `contrasts.fit` refuses some contrasts and changes the meaning of others ([Designs that cannot answer the question](#estimable-contrasts)) and limma's `eBayes` computes no F-statistic once the design has lost a column (found by probing).
- **F-tests.** Under voom's weights the default F-statistic is exact, where limma's uses the correlation of the design without weights ([Moderated F-statistics](empirical_bayes.md#moderated-f-statistics)). Each gene's F-test takes its own denominator degrees of freedom, as `eBayes`'s `F.p.value` does; after a robust fit whose first gene has infinite prior degrees of freedom, limma's `topTable(coef = ...)` of some of a fit's columns refers every gene to infinite degrees of freedom instead (found by probing limma 3.62.1).
- **Values without meaning are refused.** {api}`voom` raises an error where limma's voom returns values without meaning: for a sample whose library size is zero, or where the trend is not positive at a fitted value.
- **Designs of full rank.** {api}`voom` needs a design of full column rank, where limma's drops the aliased columns; {api}`limma_voom` fits each cell type on the columns it keeps.
- **Prior weights in voom's weights.** With prior weights, {api}`voom` returns the product of the trend's weights and the prior weights, the weights that a fit needs; limma's `voom` returns only the trend's weights, which its help page says modify the prior weights.
- **Where REML stops.** {api}`sample_quality_weights` stops REML where limma stops, which can be short of the maximum of the likelihood, so that its weights are limma's; a smaller `tolerance` converges further. The gene-by-gene algorithm depends on the order of the genes, as limma's does.

(limma-voom-checks)=
## Checks and further reading

Each step is compared with limma 3.62.1 and edgeR 4.4.0, run as separate programs on the same data:

- voom and its span: exact (1e-10), and close (1e-6) on rarefied counts ([Validation](#validation-voom)).
- Sample quality weights, alone and with voom: close (1e-6), the tier of iterative estimates ([Validation](#validation-sample-quality-weights)).
- {api}`limma_voom` and {api}`limma_trend` on simulated cubes, with rank-deficient cell types: exact, and close with robust moderation ([Validation](#validation-differential-expression)); with sample quality weights, prior weights, `min_counts` and adjustment across cell types: exact, and close with sample quality weights ([Validation](#validation-differential-expression-options)).
- F-tests: exact against limma's `topTable(coef = ...)` and `classifyTestsF`; limma-voom's default F-statistic exact against nested weighted least-squares fits in R; a natural spline exact against R's `ns()`; close with robust moderation ([Validation](#validation-differential-expression)).
- {api}`limma_voom` on the published data of @kang2018, from the cells to the tables of 8 cell types: exact ([Validation](#validation-interferon-stimulated-pbmcs)).

Further reading:

- @law2014 introduced voom, whose precision weights let limma's linear models and empirical Bayes moderation analyse RNA-seq counts.
- @ritchie2015 described limma and its workflows for RNA-seq and microarray data.
- @ritchie2006 introduced quality weights for the arrays of a microarray experiment, and @liu2015 combined them with voom's weights for RNA-seq, which improved power.
- @kang2018 pooled cells of 8 lupus patients, split into interferon-beta stimulated and untreated cultures, and assigned cells to donors by their genotypes; the documentation uses these data ([Datasets](../reference/datasets.md)).
- @perez2022 profiled more than 1.2 million blood cells of 162 lupus cases and 99 controls, and found interferon-stimulated genes raised in monocytes and fewer naive CD4 T cells.

The next chapters model what limma-voom does not: the counts themselves ([edgeR and DESeq2](count_models.md)), and several samples of each donor ([Mixed models and dream](mixed_models.md)).
