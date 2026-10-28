---
description: How counts are put on a common scale across samples, and which genes and samples enter a test, with normalization factors, log-CPM, filtering by expression and the samples left out.
---

# Normalization and filtering

[Multiple testing](multiple_testing.md) closed the foundations. Counts need two steps before any test. Samples are sequenced to different depths and hold different mixes of transcripts, so their counts are put on a common scale; and genes and samples with too few counts carry no evidence, so they are left out. In every cell type, {api}`limma_voom` scales the libraries by TMM factors ({api}`tmm_factors`) and keeps the genes that edgeR's rule passes ({api}`filter_by_expression`), and so do limma-trend and dream. [Normalization, filtering and sample quality](../guides/normalization_and_quality.ipynb) takes these steps one at a time on published data.

(library-sizes)=
## Library sizes and composition

The library size of a sample is its total count, and a pseudobulk profile's grows with the number of cells pooled into it; {api}`Cube.library_sizes` gives it for every sample and group. The count of a gene grows with the library size, and dividing by it removes the depth of sequencing.

It does not remove composition. When a few genes take a larger share of one library, as when a stimulation raises a set of interferon-stimulated genes, every other gene takes a smaller share of it, and looks lower there although its expression did not change. The normalization factors below correct for composition, on the assumption that most genes do not change between the samples. The analyses take the library sizes over the genes that pass the expression filter ([below](#filtering-by-expression)), as edgeR's `keep.lib.sizes = FALSE` does.

(normalization-and-size-factors)=
## Normalization factors and size factors

edgeR keeps the library sizes $N_i$ and multiplies each by a normalization factor $f_i$, into the effective library size $N_i f_i$; the factors are scaled to a geometric mean of one. Three rules give them, as edgeR's `normLibSizes` (`calcNormFactors`) computes them:

- **TMM**, the trimmed mean of M values ({api}`tmm_factors`; @robinson2010a), the default. Each sample $k$ is compared with a reference sample $r$ over the genes counted in both. With proportions $p_{gi} = y_{gi} / N_i$, a gene has the log ratio $M_g = \log_2 (p_{gk} / p_{gr})$ and the average log proportion $A_g = \tfrac12 \log_2 (p_{gk}\, p_{gr})$. The genes in the lowest and highest 30% of $M$, or in the lowest and highest 5% of $A$, are trimmed, and $\log_2 f_k$ averages the $M$ values left, each weighted by its inverse variance $1 / v_g$, with $v_g = (N_k - y_{gk}) / (N_k y_{gk}) + (N_r - y_{gr}) / (N_r y_{gr})$. The reference is the sample whose ratio of upper quartile to library size is nearest the mean of these ratios.
- **The upper quartile** ({api}`upper_quartile_factors`; @bullard2010): the 75% quantile of each sample's counts, over the genes counted in some sample, divided by its library size.
- **RLE**, the relative log expression ({api}`rle_factors`; @anders2010): the median ratio of a sample's counts to each gene's geometric mean over the samples, taken over the genes counted in every sample and divided by the library size.

DESeq2 replaces the library sizes by size factors $s_i$, by which it divides the counts into its normalized counts: {api}`median_ratio_size_factors` gives the median of each sample's ratios to the genes' geometric means, over the genes counted in every sample [@anders2010], and `method="poscounts"` its form for sparse counts, in which genes with zeros take part.

TMM is the default of edgeR's `calcNormFactors` and of the limma and edgeR workflows for RNA-seq. The trimming leaves out the genes that change most, so that a minority of strongly changed genes, like those of the stimulation above, does not move the factors. TMM, RLE and DESeq2's size factors assume that most genes do not change between samples; the upper quartile assumes that one quantile of the counts does not, and corrects less where a response changes many genes, as the guide shows.

(log-cpm)=
## Counts per million and log-CPM

Counts per million (CPM) divide each count by its effective library size, times a million: $10^6\, y_{gi} / (N_i f_i)$ ({api}`cpm`). Log-CPM values are their log2, after a small count is added so that zero counts have finite logs. edgeR adds a prior count $c$ that is an average count, scaled by library size: sample $i$ gets $c_i = c\, R_i / \bar R$, with $R_i$ the effective library sizes and $\bar R$ their mean, added to every count and twice to its library size ({api}`log2_cpm`):

$$
\log_2 \frac{y_{gi} + c_i}{R_i + 2 c_i}\, 10^6 .
$$

The prior count sets the floor of the log values and damps the variance of small counts. Each method adds the count that its reference tool adds:

- **voom** ({api}`voom`, and so {api}`limma_voom` and {api}`dream_voom`) adds 0.5 to every count and 1 to every library size ([voom](#voom));
- **limma-trend** ({api}`limma_trend`) adds an average count of 3 ([limma-trend](#limma-trend));
- **the MDS plot and the principal components** of a cube add an average count of 2, edgeR's default, which the edgeR User's Guide uses for its MDS plot ([below](#samples-at-a-glance)), and so do the pathway scores and activities of a cube's cell type ([The values scored](#pathway-score-values));
- **edgeR's average log-CPM** ({api}`average_log2_cpm`), which ranks the genes and carries the trends of the quasi-likelihood fit, adds an average count of 2 too ([edgeR's quasi-likelihood F-tests](#quasi-likelihood)).

(filtering-by-expression)=
## Filtering by expression

A gene with a few counts in a few samples cannot show a change that a test could find, and it costs a test in the adjustment for multiple testing. Its log-CPM values also mislead: they sit near the floor that the prior count sets, so their spread is small for want of counts, not because they are precise. Such genes bend voom's mean-variance trend down at its low end, where voom would then give noisy observations the largest weights; the guide shows the trend with and without the filter.

{api}`filter_by_expression` keeps a gene by edgeR's `filterByExpr` rule [@chen2016]: its CPM must reach $10 / \operatorname{median}(R) \cdot 10^6$, the CPM of 10 counts at the median effective library size, in at least $n$ samples, and its counts summed over all samples must reach 15. $n$ is the size of the smallest group of samples; with a design it is $1 / \max_i h_i$, the smallest inverse leverage of the design with an intercept added, which equals the smallest group's size for a one-way design. Beyond 10 samples, $n$ is relaxed to $10 + 0.7\,(n - 10)$. `min_count`, `min_total_count`, `large_n` and `min_prop` change these settings.

{api}`limma_voom`, {api}`limma_trend` and {api}`dream_voom` filter each cell type with its own design, as the User's Guides of edgeR and limma do. Only the fixed effects of a design set $n$. A paired design such as `~ donor + condition` gives every sample a high leverage, so a gene needs counts in fewer samples than with the condition alone, and two designs of the same samples then test different genes. `expression_filter={"sample_groups": "condition"}` groups the samples by a column of the design's sample table instead of by the design, so that different designs of the same samples test the same genes.

(samples-with-few-cells)=
## Samples with few cells or counts

A profile pooled from a handful of cells is noisy: it estimates the mean expression of the sample's cells of that type from few of them. In each cell type, {api}`limma_voom`, {api}`limma_trend` and {api}`dream_voom` fit only the samples that have at least `min_cells` cells there (10 by default; a cube that does not record cell numbers skips this rule), some counts, and at least `min_counts` counts over every gene of the cube (0 by default), so that a sample with exactly `min_cells` cells stays in, as in decoupler's `filter_samples`. The scverse best practices keep samples with at least 10 cells and 1,000 counts [@heumos2023], which `min_counts=1000` adds to `min_cells`. The design matrix of a cell type is then the rows of the design matrix for the samples left ([What runs in each cell type](#limma-voom-steps)), and the result records which samples each fit used and why the others were left out ({api}`DifferentialExpressionResult`). {api}`plot_cell_numbers` shows the entries of a cube that have fewer than `min_cells` cells.

Leaving a sample out loses its information. Two alternatives keep every sample and weight it instead: prior weights from its number of cells ([Prior weights from the number of cells](#prior-weights-from-cells)) and sample quality weights estimated from the data ([Sample quality weights](#sample-quality-weights)).

(samples-at-a-glance)=
## Samples at a glance

Two figures place the samples of one cell type before any test, so that a sample far from its group, or a variable that the design leaves out, shows. From a cube, both take the cell type's log-CPM values, with TMM factors and an average count of 2 ([above](#log-cpm)), as the edgeR User's Guide prepares them for its MDS plot: without the samples that have too few cells (`min_cells`, 10 by default) or no counts, and over the genes that pass the filter with all samples in one group.

- **The MDS plot** ({api}`plot_mds`), limma's `plotMDS` [@ritchie2015], measures the distance between two samples by their leading log2 fold change, which is the root mean square of the 500 log2 fold changes of largest size between the two, chosen afresh for every pair. Classical scaling then places the samples so that their distances on the plot approximate these [@gower1966].
- **The principal components** ({api}`principal_components`, drawn by {api}`plot_pca`) are those that DESeq2's `plotPCA` computes [@pearson1901; @love2014]. The 500 genes of largest variance over the samples, each centred on its mean, make up the matrix $X = U S V^\top$ of samples by genes, whose singular value decomposition gives the samples' coordinates $U S$ and the genes' loadings $V$; each component explains its squared singular value's share of their sum. The result, a {api}`PrincipalComponents`, keeps the loadings, which say which genes a component is made of.

The two measure distances differently: MDS compares each pair of samples by the genes that differ most between those two, and PCA compares every sample by the same 500 genes. `gene_selection="common"` makes the MDS plot a principal component plot. The sign of a component, or of an MDS dimension, is arbitrary: both make the coordinate of largest absolute value positive (of coordinates within a relative 1e-9 of it, the first), so that neither the order of the genes nor the sign of the values changes the result.

(normalization-defaults)=
## Defaults and choices

- **Normalization.** TMM factors by default; `normalization=None` in {api}`limma_voom`, {api}`limma_trend` and {api}`dream_voom` uses library sizes alone. On a matrix of counts, {api}`upper_quartile_factors`, {api}`rle_factors` and {api}`median_ratio_size_factors` give the alternatives. When a response changes many genes, check that the factors make sense, as the guide does.
- **The genes tested.** `expression_filter=True` applies edgeR's rule with each cell type's design; a mapping changes its settings, or names the column whose values group the samples (`"sample_groups"`); `False` tests every gene.
- **The samples.** `min_cells` (10) and `min_counts` (0) set which samples enter each cell type's fit; `min_counts=1000` adds the scverse threshold of counts.
- **The prior count**, of {api}`log2_cpm` and {api}`limma_trend`: 2 and 3 by default, as above.
- **Looking at the samples.** `top`, 500 by default as in limma and DESeq2, sets how many genes each distance of {api}`plot_mds` or each component of {api}`principal_components` uses.

(normalization-limits)=
## Limits and differences

- **Factors that differ between genes.** Normalization factors and size factors are one per sample. DESeq2's normalization factors for every gene and sample are not implemented ([Not implemented yet](#not-implemented-yet)).
- **Refusals where edgeR returns values.** A sample whose counts are all zero has no library size to normalize, and {api}`tmm_factors` raises an error where edgeR's TMM with a given reference sample returns a factor of 1; an upper quartile of zero raises an error where edgeR returns NaN factors; and {api}`filter_by_expression` given both sample groups and a design raises an error, where edgeR ignores the design.
- **Two samples.** {api}`principal_components` returns one component for two samples, since the centred values of $n$ samples have at most $n - 1$ dimensions; DESeq2's `plotPCA` returns a second, of rounding noise.

(normalization-checks)=
## Checks and further reading

The factors, CPM and the filter are compared with edgeR 4.4.0 and DESeq2 1.46.0 on dense, sparse and tied counts: exact (1e-10), and the filter keeps the same genes ([Validation](#validation-normalization)). The functions repeat R's floating-point operations wherever one decides a discrete outcome, such as TMM's reference sample or a cut-off of the filter, so that on the validation's inputs they agree to the last bit. The principal components are compared with DESeq2 1.46.0's `plotPCA` ([Validation](#validation-pca-and-dot-plots)), and the MDS plot with limma 3.62.1's `plotMDS` ([Validation](#validation-plotting)): exact, up to the sign of each component.

Further reading:

- @robinson2010a introduced TMM normalization and showed that composition, a few highly expressed genes taking reads from the others, biases comparisons that scale by library size alone.
- @anders2010 introduced the median-of-ratios size factors with DESeq.
- @bullard2010 evaluated normalization and testing methods for mRNA-seq and proposed scaling by the upper quartile of the counts.
- @chen2016 described edgeR's quasi-likelihood pipeline from reads to pathways, with the rule for filtering genes by expression that `filterByExpr` applies.
- @heumos2023 reviewed best practices for single-cell analysis, with thresholds of cells and counts for the samples of a pseudobulk analysis.

[limma-voom and limma-trend](limma_voom.md), the next chapter, tests every cell type.
