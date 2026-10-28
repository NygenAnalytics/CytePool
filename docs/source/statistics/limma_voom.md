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
5. **Tests.** Each contrast is estimated from the fit ([Contrasts](#contrasts)), each gene's residual variance is moderated by empirical Bayes ([Empirical Bayes moderation](empirical_bayes.md)), and the p-values of each contrast are adjusted by Benjamini and Hochberg's method over the genes of the cell type ([Multiple testing](#adjusting-across-cell-types)).

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

(limma-voom-defaults)=
## Defaults and choices

The defaults follow limma's workflow for RNA-seq, with two additions: samples with fewer than 10 cells in a cell type are left out, and contrasts get exact standard errors ([Contrasts](#contrasts)).

- **voom or trend.** {api}`limma_voom` by default; {api}`limma_trend` when library sizes lie within about threefold of each other ([limma-trend](#limma-trend)).
- **Samples, genes and normalization.** `min_cells`, `min_counts`, `expression_filter` and `normalization` set the samples, genes and library sizes of each cell type's fit, and `prior_count` the prior count of limma-trend, as [Normalization and filtering](#normalization-defaults) gives them.
- **The span**, of {api}`limma_voom`: chosen in each cell type by default ([The span](#voom-span)); `span=0.5` reproduces limma's voom before limma 3.66.
- **Sample weights.** `quality_weights=True` estimates a weight for each sample from the data, and a mapping changes the settings of the estimate, such as `{"method": "reml"}`; `prior_weights="n_cells"` takes the weights from the numbers of cells. The two are alternatives: limma's `voomWithQualityWeights` takes no prior weights, and edgeR's `voomLmFit` refuses prior weights together with estimated sample weights, as {api}`limma_voom` and {api}`limma_trend` do.
- **Hypervariable genes.** `robust=True` estimates the prior robustly, so that genes with outlying variances do not distort it [@phipson2016] ([Estimating the prior](#estimating-the-prior)).
- **Contrasts of several coefficients.** Their standard errors are exact by default; in {api}`limma_voom`, `contrast_standard_errors="limma"` reproduces limma's approximation ([Contrasts](#contrasts)). In limma-trend the two agree.
- **Many cell types.** p-values are adjusted within each cell type and contrast; `adjust_across_groups=True` adds a column adjusted over every cell type ([Multiple testing](#adjusting-across-cell-types)).

(limma-voom-limits)=
## Limits and differences

- **Fixed effects only.** A design with random effects is refused. Several samples of a donor are paired in the design (`~ donor + condition`), or the donor gets a random effect with {api}`dream_voom` ([Mixed models and dream](mixed_models.md)). limma's `duplicateCorrelation` and voom's block correlation are not implemented.
- **No voomLmFit.** limma's documentation now recommends edgeR's `voomLmFit` over `voom`, in particular to estimate a correlation within blocks or for sparse counts with many zeros (voom's help page); `voomLmFit` is not implemented.
- **Normalization.** Normalization factors are TMM factors or none; voom's microarray-style normalization of log-CPM values (`normalize.method`) is not offered.
- **Groups of samples.** {api}`limma_voom` and {api}`limma_trend` give each sample a weight of its own. limma's variance models for groups of samples (`var.group`, `var.design`) are offered by {api}`sample_quality_weights` and {api}`voom_with_quality_weights`, on one matrix.
- **Rank-deficient cell types.** Every contrast that a cell type's samples can estimate is tested, where limma's `contrasts.fit` refuses some and changes the meaning of others ([Designs that cannot answer the question](#estimable-contrasts)).
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
- {api}`limma_voom` on the published data of @kang2018, from the cells to the tables of 8 cell types: exact ([Validation](#validation-interferon-stimulated-pbmcs)).

Further reading:

- @law2014 introduced voom, whose precision weights let limma's linear models and empirical Bayes moderation analyse RNA-seq counts.
- @ritchie2015 described limma and its workflows for RNA-seq and microarray data.
- @ritchie2006 introduced quality weights for the arrays of a microarray experiment, and @liu2015 combined them with voom's weights for RNA-seq, which improved power.
- @kang2018 pooled cells of 8 lupus patients, split into interferon-beta stimulated and untreated cultures, and assigned cells to donors by their genotypes; the documentation uses these data ([Datasets](../reference/datasets.md)).
- @perez2022 profiled more than 1.2 million blood cells of 162 lupus cases and 99 controls, and found interferon-stimulated genes raised in monocytes and fewer naive CD4 T cells.

The next chapters model what limma-voom does not: the counts themselves ([edgeR and DESeq2](count_models.md)), and several samples of each donor ([Mixed models and dream](mixed_models.md)).
