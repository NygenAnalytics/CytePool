---
description: How CytePool adjusts p-values for testing thousands of genes, the family-wise and false discovery rate methods of R's p.adjust, and adjustment within or across cell types.
---

# Multiple testing

Testing 20,000 genes at a p-value of 0.05 calls about 1,000 of them even if none
changed. An adjustment for multiple testing turns each p-value into one that controls an
error rate over the family of tests. {api}`adjust_p_values` computes the adjustments of
R's `p.adjust`. Every differential-expression result adjusts its p-values by the method
of Benjamini and Hochberg over the genes of one cell type and contrast, as limma's
`topTable` does; {api}`deseq2_wald_test` and {api}`deseq2_likelihood_ratio_test` adjust
over the genes that pass DESeq2's independent filter on the mean count [@bourgon2010].

## The methods

With $n$ tests in a family and the sorted p-values $p_{(1)} \le \dots \le p_{(n)}$, the
adjusted value of $p_{(i)}$ is, capped at 1:

| Method | Controls | Adjusted p-value |
| --- | --- | --- |
| `"bonferroni"` | family-wise error rate | $n\, p_{(i)}$ |
| `"holm"` [@holm1979] | family-wise error rate | $\max_{j \le i} (n - j + 1)\, p_{(j)}$ |
| `"hochberg"` [@hochberg1988] | family-wise error rate | $\min_{j \ge i} (n - j + 1)\, p_{(j)}$ |
| `"hommel"` [@hommel1988] | family-wise error rate | the largest Simes p-value [@simes1986] over the sets of tests that contain test $i$ |
| `"bh"` [@benjamini1995] | false discovery rate | $\min_{j \ge i} n\, p_{(j)} / j$ |
| `"by"` [@benjamini2001] | false discovery rate under any dependence | the `"bh"` value times $\sum_{k=1}^{n} 1/k$ |

The family-wise error rate is the chance of any false call; the false discovery rate is
the expected share of false calls among all calls. Genome-wide screens usually use the
false discovery rate: a list of 200 genes at an adjusted p-value of 0.05 is expected to
hold at most about 10 false ones. Bonferroni's and Holm's methods hold whatever the
dependence between the tests; Benjamini and Yekutieli's method controls the false
discovery rate under any dependence, at a cost in power [@benjamini2001].

## Adjusting within or across cell types

{api}`limma_voom`, {api}`limma_trend` and {api}`dream_voom` adjust over the genes of one
cell type and contrast, so a call is judged within its cell type. With
`adjust_across_groups=True` they add a column adjusted over every gene of every cell
type for the same contrast, one family per contrast. In simulations, the adjustment over
all cell types lowered both the false discovery rate and the power, and so was very
conservative [@crowell2020]. The adjustment across cell types is not larger for every
test: a family with more small p-values can lower an adjusted value, as [Genes that
change in each cell type](../guides/genes_per_cell_type.ipynb#adjusting-within-or-across-cell-types)
shows on published data.

## Other families

Gene-set tests adjust over the sets ({api}`camera`, {api}`fry`, {api}`mroast`);
{api}`romer`'s p-values are not adjusted, as in limma, and {api}`adjust_p_values`
adjusts them. A composition test adjusts over the cell types, and a meta-analysis over
the genes of each cell type and contrast.

{api}`proportion_of_true_nulls` estimates the share of the tests without an effect,
$\pi_0$, as qvalue's `pi0est` does [@storey2002]; the
[replication of studies](meta_analysis.md#the-proportion-of-true-nulls-and-pi1) reads
$\pi_1 = 1 - \pi_0$ of one study's p-values among the genes another finds.
