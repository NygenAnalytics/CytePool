---
description: How the p-values of thousands of tests become a list of calls with a known error rate, within or across cell types, and how many of the tests have an effect at all.
---

# Multiple testing

[Empirical Bayes moderation](empirical_bayes.md) gave every gene a p-value for each contrast. Testing 20,000 genes at a p-value of 0.05 calls about 1,000 of them even if none changed, so the p-values are adjusted for the number of tests, to control an error rate over the family of tests. {api}`adjust_p_values` computes the adjustments of R's `p.adjust`, by the method of Benjamini and Hochberg by default, as limma's `topTable` adjusts them. [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) adjusts within and across cell types on published data.

(error-rates)=
## Error rates and their methods

The family-wise error rate is the chance of any false call; the false discovery rate is the expected share of false calls among all calls. For $n$ tests in a family, with p-values sorted as $p_{(1)} \le \dots \le p_{(n)}$, each method adjusts $p_{(i)}$ to the value below, capped at 1:

| Method | Controls | Adjusted p-value |
| --- | --- | --- |
| `"bonferroni"` | family-wise error rate | $n\, p_{(i)}$ |
| `"holm"` [@holm1979] | family-wise error rate | $\max_{j \le i} (n - j + 1)\, p_{(j)}$ |
| `"hochberg"` [@hochberg1988] | family-wise error rate | $\min_{j \ge i} (n - j + 1)\, p_{(j)}$ |
| `"hommel"` [@hommel1988] | family-wise error rate | the largest Simes p-value [@simes1986] over the sets of tests that contain test $i$ |
| `"bh"` [@benjamini1995] | false discovery rate | $\min_{j \ge i} n\, p_{(j)} / j$ |
| `"by"` [@benjamini2001] | false discovery rate under any dependence | the `"bh"` value times $\sum_{k=1}^{n} 1/k$ |

A test is called when its adjusted p-value is below the chosen rate. Genome-wide screens usually control the false discovery rate: calling the genes whose adjusted p-values are below 0.05 keeps the expected share of false calls at 5% or less, about 10 in a list of
200. Bonferroni's and Holm's methods
hold whatever the dependence between the tests. Benjamini and Hochberg's method controls the false discovery rate for independent tests and for tests with positive dependence, and Benjamini and Yekutieli's under any dependence, at a cost in power [@benjamini2001].

(adjusting-across-cell-types)=
## Adjusting within or across cell types

{api}`limma_voom`, {api}`limma_trend` and {api}`dream_voom` adjust over the genes of one cell type and contrast, so that a call is judged within its cell type. With `adjust_across_groups=True` they add a column adjusted over every gene of every cell type for the same contrast, one family per contrast. Adjust within each cell type to rank its genes, and across cell types when the study's list of genes, of every cell type, is the result. In simulations, the adjustment over all cell types lowered both the false discovery rate and the power, and so was very conservative [@crowell2020]. It is not larger for every test: a family with more small p-values can lower an adjusted value, as [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb#guide-adjusting-across-cell-types) shows on published data.

(other-families)=
## Other families

Gene-set tests adjust over the sets ({api}`camera`, {api}`fry`, {api}`mroast`); the p-values of {api}`romer` are not adjusted, as in limma, and {api}`adjust_p_values` adjusts them. A composition test adjusts over the cell types, and a meta-analysis over the genes of each cell type and contrast. {api}`deseq2_wald_test` and {api}`deseq2_likelihood_ratio_test` adjust over the genes that pass DESeq2's independent filter on the mean count, which raises the number of calls at the same false discovery rate [@bourgon2010].

(proportion-of-true-nulls)=
## The proportion of true nulls

Adjusted p-values say which tests to call; the proportion of the tests without an effect, $\pi_0$, says how many effects there are in all, called or not. The p-values of tests without an effect are uniform, so among $m$ tests the p-values at least $\lambda$ hold about $\pi_0 m (1 - \lambda)$ of them, and

$$
\hat\pi_0(\lambda) = \frac{\#\{p \ge \lambda\}}{m (1 - \lambda)}
$$

estimates $\pi_0$ [@storey2002], with a bias that falls, and a variance that grows, as $\lambda$ rises. {api}`proportion_of_true_nulls` gives it as qvalue's `pi0est` does, with its defaults and options, and returns a {api}`TrueNullProportion`: at one $\lambda$; smoothed over $\lambda = 0.05, 0.10, \dots, 0.95$ by a cubic smoothing spline with 3 degrees of freedom and read at the largest $\lambda$, the default [@storey2003]; or at the $\lambda$ whose bootstrap estimate of mean squared error is smallest [@storey2004]. The estimate is capped at 1. Its complement, $\pi_1 = 1 - \pi_0$, is the share of the tests with an effect: the p-values of one study among the genes that another finds give it as a rate of replication ([How well one study replicates another](replication.md#pi1-between-studies)).

(multiple-testing-defaults)=
## Defaults and choices

- **The method.** `"bh"`, where R's `p.adjust` defaults to `"holm"`: a screen of many genes asks for a list with few false calls, which the false discovery rate measures. `"holm"` suits a few tests chosen in advance, when any false call matters.
- **The family.** Each cell type and contrast by default; `adjust_across_groups=True` adds a family of every cell type ([above](#adjusting-across-cell-types)). {api}`adjust_p_values` adjusts all the p-values it is given as one family, or each slice along `axis`, such as each contrast of a matrix of genes by contrasts.
- **Families larger than observed.** `n` declares a family larger than the p-values given, as R's `p.adjust` does: the missing p-values count as larger than every one given for Bonferroni's and Holm's methods, and as 1 for the others.
- **Missing p-values**, marked NaN, stay NaN and do not count as tests.
- **The estimate of $\pi_0$.** The smoother over qvalue's grid of $\lambda$ by default. One value of `lambdas`, such as 0.5, suits a few dozen p-values, on which the smoother, which reads the largest $\lambda$, often gives an estimate of 0 or 1; `method="bootstrap"`, `smooth_df` and `smooth_log` are qvalue's options.

(multiple-testing-limits)=
## Limits and differences

{api}`adjust_p_values` follows R's `p.adjust`. {api}`proportion_of_true_nulls` follows qvalue 2.38.0, also where qvalue's documentation leaves details open, such as counting the p-values equal to $\lambda$, and differs from it on purpose in these:

- **No p-value at the largest $\lambda$.** qvalue stops with an error, with the smoother and the bootstrap alike. CytePool smooths the count of zero as it is, and gives it no variance in the bootstrap; the smoother of logarithms, which a count of zero leaves without a value, refuses it. This is common among the genes that one study finds, whose p-values in another study are small.
- **An estimate at most 0.** qvalue stops; CytePool gives 0.
- **More than 49 values of $\lambda$.** R's smoother then places knots at only some of them; CytePool places one at every $\lambda$. Repeated values of $\lambda$ are refused.

(multiple-testing-checks)=
## Checks and further reading

- {api}`adjust_p_values`, every method on every case: R 4.4.3's `p.adjust`, exact (1e-10) ([Validation](#validation-multiple-testing)).
- {api}`proportion_of_true_nulls`, with qvalue's defaults and options: qvalue 2.38.0's `pi0est`, exact (1e-10) ([Validation](#validation-combined-p-values-and-replication)).

Further reading:

- @holm1979 introduced a sequentially rejective procedure that controls the family-wise error rate and rejects at least as much as Bonferroni's.
- @simes1986 introduced an improved Bonferroni test of a set of hypotheses, and @hochberg1988 and @hommel1988 procedures that build on it.
- @benjamini1995 introduced the false discovery rate and the procedure that controls it, and @benjamini2001 showed that it holds under positive dependence and gave a procedure for any dependence.
- @storey2002 estimated the proportion of true nulls from the p-values above a threshold, and @storey2003 introduced the q-value for genome-wide studies, with that estimate smoothed over thresholds.

The foundations are in place: a cube of profiles, a design fitted to every gene, moderated tests and their adjustment. [Normalization and filtering](normalization_and_filtering.md) begins the part on differential expression, which applies them to counts.
