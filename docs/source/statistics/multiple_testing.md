---
description: How thousands of p-values become calls with a known error rate, within or across cell types, which contrasts of a gene differ, also across genes, and how many tests have an effect.
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

The single-step adjusted p-value of contrast $j$ is the chance that the largest of the gene's t-statistics is at least as extreme as $t_j$: $P(\max_l |T_l| \ge |t_j|)$ two-sided, with $T$ multivariate t on the gene's degrees of freedom and correlation $\rho$. Calling the contrasts whose adjusted p-values are below $\alpha$ keeps the chance of any false call among the gene's contrasts at most $\alpha$. Bonferroni's p-values are larger. The table gives how much larger, as a share of the single-step p-value, for Dunnett's contrasts of equal groups (correlated by 0.5) at single-step p-values of 1e-2, 1e-4 and 1e-6, from Dunnett's integral:

| Degrees of freedom | 3 contrasts | 8 contrasts |
| --- | --- | --- |
| 5 | 26%, 23%, 22% | 73%, 66%, 65% |
| 10 | 16%, 10%, 9% | 41%, 29%, 25% |
| 20 | 11%, 5%, 3% | 27%, 13%, 9% |
| 40 | 9%, 3%, 1% | 21%, 7%, 3% |
| Normal (z-tests) | 7%, 1%, 0.2% | 15%, 3%, 0.6% |

The gap falls as the degrees of freedom grow and as the p-values shrink. The degrees of freedom of a moderated test, `df_total` of the fit, add the prior's to the residual ones: in the example below, 3 replicates of 5 groups leave 10 residual degrees of freedom, and the median `df_total` is 32.5.

The example tests each dose against the vehicle in every gene of one cell type of a simulated dose series ([Simulated dose series with known curves](dose_response.md#simulated-dose-series)). The design has one coefficient per dose group, `~ 0 + dose_group`, and {api}`limma_voom` fits the Dunnett contrasts of the groups; {api}`multiple_contrast_test` takes the moderated fit and tests its contrasts as they are, with their covariance in each gene and the fit's degrees of freedom:

```python
from cytepool import Design, limma_voom, simulate_dose_series
from cytepool.stats.multiple_contrasts import dunnett_contrasts, multiple_contrast_test

cube, truth = simulate_dose_series([0, 1, 3, 10, 30], replicates=3, n_genes=2000, rng=3)
design = Design(cube.samples, "~ 0 + dose_group")
result = limma_voom(cube, design, contrasts=dunnett_contrasts(design.columns))
cell_type = result.fit("cell_type_1")
test = multiple_contrast_test(cell_type.moderated, rng=1)
test.table(feature_ids=cell_type.features["feature_id"]).sort("p_value")
# The first rows, some of the columns:
# feature_id  contrast                     estimate  standard_error      t  p_value  adjusted_p_value
# gene_0344   dose_groupd4 - dose_groupd0      2.47            0.27   9.15  6.5e-10           7.0e-07
# gene_1499   dose_groupd4 - dose_groupd0      2.11            0.24   8.69  2.2e-09           1.2e-06
# gene_1844   dose_groupd4 - dose_groupd0     -2.50            0.31  -7.98  1.5e-08           5.4e-06
```

Each row is a gene's test as a whole: `contrast` is its most extreme contrast, here the highest dose, with its estimate and standard error; `p_value` is that contrast's p-value adjusted for the gene's four contrasts, the gene's max-t p-value; and `adjusted_p_value` is Benjamini and Hochberg's adjustment of these over the genes, below 0.05 for 57 of the 1,092 genes tested. The first group of {api}`dunnett_contrasts`, here `dose_groupd0`, is the control. In a design with an intercept, such as `~ dose_group`, the coefficients of the doses are already differences from the vehicle: name them as the contrasts of the fit, as the [bulk RNA-seq guide](../guides/bulk_rna_seq.ipynb) does with its two drugs, so that the intercept stays out of each gene's family. Group means and their covariance work too, as the example of {api}`multiple_contrast_test` shows for a one-way design. A fit whose contrasts carry limma's approximate standard errors (`contrast_standard_errors="limma"`) holds no covariance of its contrasts and is refused.

{api}`MultipleContrastTest.table` gives two levels. Without a contrast, each gene's test as a whole, as above, which calls the genes that respond. With a contrast, its estimate, standard error, t-statistic and `p_value`, adjusted for the gene's contrasts, unlike the p-values of other CytePool tables. To call doses across genes, screen the genes by `adjusted_p_value` at a false discovery rate $\alpha$ first; then, in the $R$ genes called of the $G$ tested, call the contrasts whose `p_value` is below $\alpha R / G$. This is the confirmation level of stageR's two-stage rule, which keeps the overall false discovery rate, the share of called genes with any false call, a gene without a change or a dose named wrongly, at $\alpha$ [@vandenberge2017]. Each called gene has at least one such contrast, its most extreme, since Benjamini and Hochberg's rule calls a gene whose p-value is below $\alpha R / G$. [Calling contrasts across genes](#stage-wise-testing) gives these calls as adjusted p-values, and those of stageR's default, Holm's step-down procedure within each gene.

(dunnett-and-williams-contrasts)=
### Dunnett's and Williams-type contrasts

| Family | Contrasts | Alternative | Function |
| --- | --- | --- | --- |
| Dunnett's [@dunnett1955] | each dose minus the vehicle | two-sided, or one-sided when the direction is known | {api}`dunnett_contrasts` |
| Williams-type [@bretz2006] | the mean of the $j$ highest doses, weighted by their group sizes, minus the vehicle, for $j = 1, \dots, K-1$ | two-sided for genes, whose direction is not known; one-sided when it is | {api}`williams_contrasts` |

Williams-type contrasts suit responses that grow in dose order: pooling the top doses gains power when several have changed. A two-sided test covers rises and falls at one error rate, where testing `"greater"` and `"less"` separately and joining the calls doubles it. A significant Williams-type contrast means that the doses it pools differ from the vehicle on average, not that each of them does, and the most extreme contrast is the pooling that best matches the observed change, not an estimate of how many doses changed: with 3 replicates and every dose changed in proportion to its rank (0, 1, 2 and 3 residual standard deviations), the highest dose alone had the largest t-statistic in 60% of 2,000 simulated genes, and the pooling of all three doses in 2%. Which doses changed is the question of Dunnett's contrasts. Williams-type contrasts differ from Williams' trend test [@williams1971], which compares each dose with the vehicle after fitting means that change in dose order, and which @bretz2006 extends. Both builders return a {api}`Contrasts` whose columns are the groups, matched by name to the estimates; {api}`williams_contrasts` takes the size of each group, in dose order with the vehicle first. {api}`optimal_contrasts` builds a third family, the contrast that best detects each of a few candidate shapes of response, for the MCP step of MCP-Mod [@bretz2005; @pinheiro2014] ([Trends, doses against the vehicle and candidate shapes](dose_response.md#dose-response-contrasts)).

(multivariate-t-probabilities)=
### Multivariate t probabilities

{api}`multivariate_t_cdf` computes the probability that every coordinate of a multivariate t vector stays within its limit, and the complement, the adjusted p-value, as a {api}`MultivariateTProbability`; {api}`multivariate_t_quantile` computes the critical value of a max-t test, as a {api}`MultivariateTQuantile`. The complement is a sum over the coordinates of the chance that each is the first to exceed its limit; each term is computed by conditioning on its exceedance, with randomized lattice points [@genz1992; @genz2002], which keeps the relative error bounded however small the p-value [@owen2019]. Two contrasts, as the vehicle and two doses give, are computed without random numbers. The API reference of {api}`cytepool.stats.multivariate_t` gives the method.

Each p-value carries `p_value_error`, its standard error, at most `relative_error` of it (1% by default) at every magnitude. That matters for calls: at a false discovery rate of 5%, the cut-off for the top gene of 20,000 is $0.05/20{,}000 = 2.5 \times 10^{-6}$, which mvtnorm's default absolute error of 1e-3 cannot resolve. A call changes with the seed only for p-values within a few errors of its cut-off, about 2% of it at the default. A tenfold smaller `relative_error` took about 12 times as long in the benchmark ([Timings](#validation-timings)).

Simultaneous confidence intervals of a gene's contrasts take the critical value of its max-t test, from the correlation of its contrasts that the test keeps:

```python
from cytepool.stats.multivariate_t import multivariate_t_quantile

critical = multivariate_t_quantile(
    0.95, test.correlation, test.df, two_sided=True, rng=2
)
half_width = critical.quantile[:, None] * test.standard_error
lower, upper = test.estimate - half_width, test.estimate + half_width
```

For each gene, all its intervals hold their contrasts with a probability of 95%. For group means, these intervals agree with scipy's `dunnett` to within its own Monte Carlo error.

(stage-wise-testing)=
### Calling contrasts across genes

{api}`stage_wise_adjusted_p_values` gives the calls of stage-wise testing as adjusted p-values, as stageR's `stageWiseAdjustment` computes them [@heller2009; @vandenberge2017]. The screen, a test of every hypothesis of a gene at once such as an F-test or a max-t test, calls the genes whose Benjamini-Hochberg adjusted p-value is at most $\alpha$: $R$ of the $G$ genes it tested. In each gene called, its $m$ p-values, sorted $p_{(1)} \le \dots \le p_{(m)}$, are adjusted by Holm's step-down procedure [@holm1979] at the level $R\alpha / G$ that the screen leaves, which gives the stage-wise adjusted p-value

$$
\tilde p_{(j)} = \min\left(1, \frac{G}{R} \max_{i \le j} c_i \, p_{(i)}\right), \qquad c = (m - 1, m - 1, m - 2, \dots, 1).
$$

The first multiplier is $m - 1$, not Holm's $m$: the screen has already rejected that all $m$ hypotheses hold, so at least one is false, which Shaffer's refinement of Holm's procedure uses [@shaffer1986]. With one hypothesis left the multiplier is 1, Bonferroni, where stageR's 0 would call the hypothesis on the screen alone. A contrast is called when $\tilde p \le \alpha$, and a gene the screen does not call gets no values. The calls control the overall false discovery rate (OFDR) at $\alpha$, the expected share of the genes called with at least one false call, the screen's or a contrast's, proved under independence between genes [@heller2009] and expected wherever Benjamini-Hochberg controls the false discovery rate [@vandenberge2017]. The share of false calls among all gene-contrast calls is not controlled, since a gene counts once however many of its contrasts are called. The values hold only at the $\alpha$ given, as $R$ depends on it.

`method="holm"`, the default, takes unadjusted p-values, such as the moderated t-tests of each dose against the vehicle; `method="none"` takes p-values already adjusted within each gene, such as the `p_value` of each contrast of a {api}`MultipleContrastTest`, which gives the rule above. Missing p-values leave their family: a gene without a screen p-value leaves $G$, and a missing contrast leaves its gene's $m$. Shaffer's first step then needs the screen to have tested the gene's other contrasts only; when it also tested those missing here, adjust each gene's p-values by Holm's procedure first, `adjust_p_values(p_values, "holm", axis=1)`, and pass them with `method="none"`. {api}`DoseResponseResult.lowest_effective_doses` calls each dose of a dose series this way, with the spline test as screen ([From which dose a gene responds](dose_response.md#lowest-effective-dose)).

(multiple-testing-defaults)=
## Defaults and choices

- **The method.** `"bh"`, where R's `p.adjust` defaults to `"holm"`: a screen of many genes asks for a list with few false calls, which the false discovery rate measures. `"holm"` suits a few tests chosen in advance, when any false call matters.
- **The family.** Each cell type and contrast by default; `adjust_across_groups=True` adds a family of every cell type ([above](#adjusting-across-cell-types)). {api}`adjust_p_values` adjusts all the p-values it is given as one family, or each slice along `axis`, such as each contrast of a matrix of genes by contrasts.
- **Families larger than observed.** `n` declares a family larger than the p-values given, as R's `p.adjust` does: the missing p-values count as larger than every one given for Bonferroni's and Holm's methods, and as 1 for the others.
- **Missing p-values**, marked NaN, stay NaN and do not count as tests.
- **The contrasts of a gene.** {api}`dunnett_contrasts` for each dose against the vehicle; {api}`williams_contrasts` when the response grows in dose order, two-sided for genes, whose direction is not known, which tests rises and falls at one error rate, and one-sided (`alternative="greater"` or `"less"`) when the direction is known; an F-test (`f_tests`) when any change matters and which one does not.
- **Calls of contrasts across genes.** {api}`stage_wise_adjusted_p_values` with `method="holm"` on unadjusted p-values and `alpha=0.05`, stageR's defaults; `method="none"` for p-values already adjusted within each gene, such as the single-step ones.
- **The degrees of freedom.** A {api}`ModeratedFit` gives its `df_total`; with estimates, one value per gene. The multivariate t takes one value per gene, so where each contrast has its own, as in dream's tests, the smallest of a gene's is conservative. `np.inf` gives z-tests.
- **The accuracy of p-values.** `relative_error=0.01`, a standard error of at most 1% of each p-value. A tenfold smaller value took about 12 times as long in the benchmark ([Timings](#validation-timings)), and suits p-values near a cut-off.
- **The estimate of $\pi_0$.** The smoother over qvalue's grid of $\lambda$ by default. One value of `lambdas`, such as 0.5, suits a few dozen p-values, on which the smoother, which reads the largest $\lambda$, often gives an estimate of 0 or 1; `method="bootstrap"`, `smooth_df` and `smooth_log` are qvalue's options.

(multiple-testing-limits)=
## Limits and differences

{api}`adjust_p_values` follows R's `p.adjust`. {api}`proportion_of_true_nulls` follows qvalue 2.38.0, also where qvalue's documentation leaves details open, such as counting the p-values equal to $\lambda$, and differs from it on purpose in these:

- **No p-value at the largest $\lambda$.** qvalue stops with an error, with the smoother and the bootstrap alike. CytePool smooths the count of zero as it is, and gives it no variance in the bootstrap; the smoother of logarithms, which a count of zero leaves without a value, refuses it. This is common among the genes that one study finds, whose p-values in another study are small.
- **An estimate at most 0.** qvalue stops; CytePool gives 0.
- **More than 49 values of $\lambda$.** R's smoother then places knots at only some of them; CytePool places one at every $\lambda$. Repeated values of $\lambda$ are refused.

{api}`multiple_contrast_test` follows multcomp's `glht` with `parm()`, and differs from it in these:

- **Single-step adjustment only.** No step-down adjustments; simultaneous confidence intervals come from {api}`multivariate_t_quantile` and the correlation the test keeps ([above](#multivariate-t-probabilities)), where multcomp's `confint` computes them.
- **Degrees of freedom that are not integers**, such as moderated ones, are accepted, where mvtnorm refuses them; `df` has no default, where multcomp's 0 gives z-tests.
- **Names of Williams-type contrasts** say which doses are pooled, such as "mid to high - vehicle", where multcomp writes "C 2"; with two groups they are Dunnett's one contrast, which multcomp refuses.

{api}`stage_wise_adjusted_p_values` follows stageR's `stageWiseAdjustment` and differs from it in one case: a called gene with one hypothesis left, of one or of several, gets the multiplier 1, Bonferroni at the screen's level, which holds whatever the screen tested; stageR's `"holm"` gives it 0, calling it on the screen alone, or stops when the gene had two hypotheses or one ([Validation](#validation-dose-contrasts)).

(multiple-testing-checks)=
## Checks and further reading

- {api}`adjust_p_values`, every method on every case: R 4.4.3's `p.adjust`, exact (1e-10) ([Validation](#validation-multiple-testing)).
- {api}`proportion_of_true_nulls`, with qvalue's defaults and options: qvalue 2.38.0's `pi0est`, exact (1e-10) ([Validation](#validation-combined-p-values-and-replication)).
- {api}`multivariate_t_cdf` and {api}`multivariate_t_quantile`: mpmath at 30 digits for two coordinates and rank-two correlations, Dunnett's integral for product correlations into the tail, with one limit for every coordinate or a limit per coordinate, ALOE [@owen2019] for correlations without a product structure down to complements of 1e-10, mvtnorm 1.4-2 and scipy 1.18.1; {api}`dunnett_contrasts`, {api}`williams_contrasts` and {api}`multiple_contrast_test`: multcomp 1.4-32 ([Validation](#validation-multiple-contrast-tests)).
- {api}`optimal_contrasts`: DoseFinding 1.4-2's `optContr` and `MCTtest`; {api}`stage_wise_adjusted_p_values`: stageR 1.28.0's `stageWiseAdjustment` and the closed test enumerated in R, exact ([Validation](#validation-dose-contrasts)).

Further reading:

- @holm1979 introduced a sequentially rejective procedure that controls the family-wise error rate and rejects at least as much as Bonferroni's.
- @simes1986 introduced an improved Bonferroni test of a set of hypotheses, and @hochberg1988 and @hommel1988 procedures that build on it.
- @benjamini1995 introduced the false discovery rate and the procedure that controls it, and @benjamini2001 showed that it holds under positive dependence and gave a procedure for any dependence.
- @storey2002 estimated the proportion of true nulls from the p-values above a threshold, and @storey2003 introduced the q-value for genome-wide studies, with that estimate smoothed over thresholds.
- @dunnett1955 introduced the comparison of several treatments with a control at one error rate, @williams1971 a test of doses whose means change in dose order against a zero-dose control, @bretz2006 the Williams-type contrasts of unbalanced designs, and @hothorn2008 single-step tests of general linear hypotheses of any parametric model.
- @heller2009 introduced the two-stage procedure that controls the overall false discovery rate of gene sets, and @vandenberge2017 stageR's two-stage rule, which screens the genes and then confirms the hypotheses of each gene called; @shaffer1986 introduced the step that a rejected global hypothesis allows Holm's procedure.
- @genz2002 compared methods for multivariate t probabilities, and @owen2019 showed why conditioning on each rare event keeps the relative error of a union bounded.

The foundations are in place: a cube of profiles, a design fitted to every gene, moderated tests and their adjustment. [Normalization and filtering](normalization_and_filtering.md) begins the part on differential expression, which applies them to counts.
