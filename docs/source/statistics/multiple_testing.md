---
description: How thousands of p-values become calls with a known error rate, within or across cell types, which of several contrasts of a gene differ, and how many tests have an effect.
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

(correlated-tests)=
## Several contrasts of one gene

A dose series asks, for each gene, which doses differ from the vehicle, with one error rate for the gene's comparisons. Testing each dose against the vehicle on its own and multiplying its p-value by the number of doses (Bonferroni) ignores that the contrasts share the vehicle, which makes their t-statistics correlated: about 0.5 between two doses of groups the size of the vehicle's. An F-test of the contrasts, as `f_tests` runs it, says whether any dose differs, not which. {api}`multiple_contrast_test` tests every contrast of every gene and adjusts each p-value for the gene's contrasts by the single-step max-t method [@hothorn2008], and returns a {api}`MultipleContrastTest`.

For a gene with estimates $b$, such as the means of its dose groups or the coefficients of a fit, and their covariance $V$, contrast $c_j$ has the t-statistic $t_j = c_j^\top b / \sqrt{c_j^\top V c_j}$, and two contrasts are correlated by

$$
\rho_{jl} = \frac{c_j^\top V c_l}{\sqrt{c_j^\top V c_j \; c_l^\top V c_l}}.
$$

The single-step adjusted p-value of contrast $j$ is the chance that the largest of the gene's t-statistics is at least as extreme as $t_j$: $P(\max_l |T_l| \ge |t_j|)$ two-sided, with $T$ multivariate t on the gene's degrees of freedom and correlation $\rho$. Calling the contrasts whose adjusted p-values are below $\alpha$ keeps the chance of any false call among the gene's contrasts at $\alpha$. How much this gains over Bonferroni depends on the degrees of freedom: for three contrasts correlated by 0.5, Bonferroni's p-values are 20% larger than the single-step ones at every magnitude with 5 degrees of freedom (42% with eight contrasts), while for z-tests the gap falls from 3.0% at a p-value of 1.4e-3 to 0.11% at 1.1e-7 and 0.01% at 2.4e-10. The gain is largest at the small moderated degrees of freedom that 2 to 4 replicates per dose give.

{api}`MultipleContrastTest.table` gives two levels. Without a contrast, each gene's test as a whole: its most extreme contrast, whose adjusted p-value is the gene's max-t p-value, and `adjusted_p_value`, Benjamini and Hochberg's adjustment of these over the genes, which calls the genes that respond. With a contrast, its estimate, standard error, t-statistic and `p_value`, adjusted for the gene's contrasts, unlike the p-values of other CytePool tables. To call doses across genes, screen the genes by `adjusted_p_value` first and read the doses of the genes called, the order of the two-stage methods that control the false discovery rate of the genes, such as stageR's [@vandenberge2017].

(dunnett-and-williams-contrasts)=
### Dunnett's and Williams-type contrasts

| Family | Contrasts | Alternative | Function |
| --- | --- | --- | --- |
| Dunnett's [@dunnett1955] | each dose minus the vehicle | two-sided, or one-sided when the direction is known | {api}`dunnett_contrasts` |
| Williams-type [@bretz2006] | the mean of the $j$ highest doses, weighted by their group sizes, minus the vehicle, for $j = 1, \dots, K-1$ | one-sided, in the expected direction | {api}`williams_contrasts` |

Williams-type contrasts suit responses that grow in dose order: the largest of their t-statistics says how many of the top doses have changed, and pooling the doses gains power when several have. They differ from Williams' trend test, which compares each dose with the vehicle after fitting means that change in dose order, and which @bretz2006 extends. Both builders return a {api}`Contrasts` whose columns are the groups, matched by name to the estimates.

(multivariate-t-probabilities)=
### Multivariate t probabilities

{api}`multivariate_t_cdf` computes the probability that every coordinate of a multivariate t vector stays within its limit, and the complement, the adjusted p-value, as a {api}`MultivariateTProbability`; {api}`multivariate_t_quantile` computes the critical value of a max-t test, as a {api}`MultivariateTQuantile`. The complement is a sum over the coordinates of the chance that each is the first to exceed its limit; each term is computed by conditioning on its exceedance, with randomized lattice points [@genz1992; @genz2002], which keeps the relative error bounded however small the p-value [@owen2019]. Two contrasts, as the vehicle and two doses give, are computed without random numbers. The API reference of {api}`cytepool.stats.multivariate_t` gives the method.

Each p-value carries `p_value_error`, its standard error, at most `relative_error` of it (1% by default) at every magnitude. That matters for calls: at a false discovery rate of 5%, the cut-off for the top gene of 20,000 is $0.05/20{,}000 = 2.5 \times 10^{-6}$, which mvtnorm's default absolute error of 1e-3 cannot resolve. A call changes with the seed only for p-values within a few errors of its cut-off, about 2% of it at the default.

The example tests each dose against the vehicle in every gene of a fit with one coefficient per dose group, whose names follow the dose order given by `levels`:

```python
from cytepool.design import Design
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.multiple_contrasts import dunnett_contrasts, multiple_contrast_test

levels = {"dose": ["vehicle", "low", "mid", "high"]}
design = Design(samples, "~ 0 + dose", levels=levels)
fit = fit_linear_model(log_expression, design, weights=weights)
moderated = moderated_statistics(fit)
covariances = (
    moderated.posterior_variance[:, None, None] * fit.cov_coefficients_per_feature
)
test = multiple_contrast_test(
    fit.coefficients,
    covariances,
    dunnett_contrasts(fit.coefficient_names),
    df=moderated.df_total,
    estimate_names=fit.coefficient_names,
    rng=1,
)
test.table()  # each gene's test as a whole
test.table("dosehigh - dosevehicle")  # the highest dose in every gene
```

The contrasts are matched to `estimate_names` by name; the first group of `dunnett_contrasts`, here `dosevehicle`, is the control. A fit without weights or missing values keeps one covariance matrix for every gene, `fit.cov_coefficients`, and `fit.cov_coefficients_per_feature` is None; the covariances are then `moderated.posterior_variance[:, None, None] * fit.cov_coefficients`.

(multiple-testing-defaults)=
## Defaults and choices

- **The method.** `"bh"`, where R's `p.adjust` defaults to `"holm"`: a screen of many genes asks for a list with few false calls, which the false discovery rate measures. `"holm"` suits a few tests chosen in advance, when any false call matters.
- **The family.** Each cell type and contrast by default; `adjust_across_groups=True` adds a family of every cell type ([above](#adjusting-across-cell-types)). {api}`adjust_p_values` adjusts all the p-values it is given as one family, or each slice along `axis`, such as each contrast of a matrix of genes by contrasts.
- **Families larger than observed.** `n` declares a family larger than the p-values given, as R's `p.adjust` does: the missing p-values count as larger than every one given for Bonferroni's and Holm's methods, and as 1 for the others.
- **Missing p-values**, marked NaN, stay NaN and do not count as tests.
- **The contrasts of a gene.** {api}`dunnett_contrasts` for each dose against the vehicle; {api}`williams_contrasts` with `alternative="greater"` or `"less"` when the response grows in dose order; an F-test (`f_tests`) when any change matters and which one does not.
- **The degrees of freedom.** One value per gene, the moderated `df_total` of a fit. The multivariate t takes one value per gene, so where each contrast has its own, as in dream's tests, the smallest of a gene's is conservative. `np.inf` gives z-tests.
- **The accuracy of p-values.** `relative_error=0.01`, a standard error of at most 1% of each p-value; the cost grows as its inverse square, and smaller values suit p-values near a cut-off.
- **The estimate of $\pi_0$.** The smoother over qvalue's grid of $\lambda$ by default. One value of `lambdas`, such as 0.5, suits a few dozen p-values, on which the smoother, which reads the largest $\lambda$, often gives an estimate of 0 or 1; `method="bootstrap"`, `smooth_df` and `smooth_log` are qvalue's options.

(multiple-testing-limits)=
## Limits and differences

{api}`adjust_p_values` follows R's `p.adjust`. {api}`proportion_of_true_nulls` follows qvalue 2.38.0, also where qvalue's documentation leaves details open, such as counting the p-values equal to $\lambda$, and differs from it on purpose in these:

- **No p-value at the largest $\lambda$.** qvalue stops with an error, with the smoother and the bootstrap alike. CytePool smooths the count of zero as it is, and gives it no variance in the bootstrap; the smoother of logarithms, which a count of zero leaves without a value, refuses it. This is common among the genes that one study finds, whose p-values in another study are small.
- **An estimate at most 0.** qvalue stops; CytePool gives 0.
- **More than 49 values of $\lambda$.** R's smoother then places knots at only some of them; CytePool places one at every $\lambda$. Repeated values of $\lambda$ are refused.

{api}`multiple_contrast_test` follows multcomp's `glht` with `parm()`, and differs from it in these:

- **Single-step adjustment only.** No step-down adjustments and no simultaneous confidence intervals; {api}`multivariate_t_quantile` gives the critical value that an interval needs.
- **Degrees of freedom that are not integers**, such as moderated ones, are accepted, where mvtnorm refuses them; `df` has no default, where multcomp's 0 gives z-tests.
- **Names of Williams-type contrasts** say which doses are pooled, such as "mid to high - vehicle", where multcomp writes "C 2"; with two groups they are Dunnett's one contrast, which multcomp refuses.

(multiple-testing-checks)=
## Checks and further reading

- {api}`adjust_p_values`, every method on every case: R 4.4.3's `p.adjust`, exact (1e-10) ([Validation](#validation-multiple-testing)).
- {api}`proportion_of_true_nulls`, with qvalue's defaults and options: qvalue 2.38.0's `pi0est`, exact (1e-10) ([Validation](#validation-combined-p-values-and-replication)).
- {api}`multivariate_t_cdf` and {api}`multivariate_t_quantile`: mpmath at 30 digits for two coordinates and rank-two correlations, Dunnett's integral for product correlations into the tail, mvtnorm 1.4-2 and scipy 1.18.1; {api}`dunnett_contrasts`, {api}`williams_contrasts` and {api}`multiple_contrast_test`: multcomp 1.4-32 ([Validation](#validation-multiple-contrast-tests)).

Further reading:

- @holm1979 introduced a sequentially rejective procedure that controls the family-wise error rate and rejects at least as much as Bonferroni's.
- @simes1986 introduced an improved Bonferroni test of a set of hypotheses, and @hochberg1988 and @hommel1988 procedures that build on it.
- @benjamini1995 introduced the false discovery rate and the procedure that controls it, and @benjamini2001 showed that it holds under positive dependence and gave a procedure for any dependence.
- @storey2002 estimated the proportion of true nulls from the p-values above a threshold, and @storey2003 introduced the q-value for genome-wide studies, with that estimate smoothed over thresholds.
- @dunnett1955 introduced the comparison of several treatments with a control at one error rate, @bretz2006 the Williams-type contrasts of unbalanced designs, and @hothorn2008 single-step tests of general linear hypotheses of any parametric model.
- @genz2002 compared methods for multivariate t probabilities, and @owen2019 showed why conditioning on each rare event keeps the relative error of a union bounded.

The foundations are in place: a cube of profiles, a design fitted to every gene, moderated tests and their adjustment. [Normalization and filtering](normalization_and_filtering.md) begins the part on differential expression, which applies them to counts.
