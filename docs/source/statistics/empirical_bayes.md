---
description: Why each gene's residual variance is shrunk towards a value shared by all genes, how that prior is estimated, and the moderated t, F and B statistics and TREAT that follow.
---

# Empirical Bayes moderation

[Designs, linear models and contrasts](linear_models.md) gave every gene an estimate of each contrast and a residual variance $s_g^2$. With a few samples, $s_g^2$ is a poor estimate: some genes look precise by chance and get large t-statistics. Empirical Bayes moderation borrows strength from all the genes, by shrinking each variance towards a value estimated from them [@smyth2004]. {api}`moderated_statistics` moderates a fit as limma's `eBayes` does, and so do {api}`limma_voom`, {api}`limma_trend`, {api}`dream_voom` and {api}`propeller` in their tests. [Normalization, filtering and sample quality](../guides/normalization_and_quality.ipynb) shows the moderated variances of one cell type.

(empirical-bayes-model)=
## The model

The residual variance of gene $g$ on $d_g$ degrees of freedom, and its true variance $\sigma_g^2$, follow

$$
s_g^2 \mid \sigma_g^2 \sim \frac{\sigma_g^2\, \chi^2_{d_g}}{d_g}, \qquad \frac{1}{\sigma_g^2} \sim \frac{\chi^2_{d_0}}{d_0 s_0^2},
$$

so that $s_g^2$ follows $s_0^2$ times an $F(d_g, d_0)$ distribution. The prior $(s_0^2, d_0)$ is estimated from all the genes, and each gene's variance becomes its posterior value

$$
\tilde s_g^2 = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g}.
$$

The moderated t-statistic $t_g = \hat\beta_g / (u_g \tilde s_g)$, for a coefficient or contrast $\hat\beta_g$ with unscaled standard error $u_g$, follows a t distribution on $d_0 + d_g$ degrees of freedom when the coefficient is zero. These are capped, as in limma, at the total of all the genes' residual degrees of freedom, and the two-sided p-value comes from the tail of that distribution ({api}`t_sf`), on fractional degrees of freedom too. A large $d_0$ means that the genes have similar variances and the prior counts for much; $d_0 = 0$ gives the ordinary t-test, and an infinite $d_0$ gives every gene the prior variance. {api}`moderated_statistics` returns the statistics in a {api}`ModeratedFit`, whose {api}`ModeratedFit.table` has the columns of limma's `topTable`, and {api}`squeeze_variances` moderates any set of variances ({api}`SqueezedVariances`).

(estimating-the-prior)=
## Estimating the prior

The prior is estimated from the variances of all the genes, by one of two estimators (`prior_estimator`):

- **The method of moments** ({api}`fit_f_distribution`, limma's `fitFDist`) matches the mean and variance of $\log s_g^2$, corrected for their sampling distribution, to those of the scaled F distribution [@smyth2004, section 6.2], and returns a {api}`FDistributionFit`. When the variances vary no more than sampling explains, $d_0$ is infinite.
- **Weighted moments and profile likelihood** ({api}`fit_f_distribution_likelihood`, limma's `fitFDistUnequalDF1`) estimate $s_0^2$ from a weighted mean of the log variances, in which a variance counts less the fewer degrees of freedom it has, and $d_0$ by maximizing the likelihood of the variances [@chen2025] ({api}`LikelihoodFDistributionFit`). They suit degrees of freedom that differ between genes, or are small or fractional, as with missing values, zero weights, mixed models or edgeR's quasi-likelihood.

By default, as in limma since version 3.61.8 (`legacy = NULL`), the likelihood estimator is used when the genes that have residual degrees of freedom do not all have the same number of them, and the method of moments otherwise; genes without residual degrees of freedom do not count.

With a covariate, the prior variance follows a trend. limma-trend gives it each gene's mean log-expression (`trend=True`): $s_0^2$ then becomes a natural cubic spline in it for the method of moments ({api}`natural_spline_basis`; @phipson2016, section 10.1), a LOWESS trend for its robust form below, and a weighted LOWESS trend for the likelihood estimator.

Genes with outlying variances, such as hypervariable genes, widen the spread of the variances, and so lower $d_0$ for every gene. With `robust=True` they do not
[@phipson2016]: the prior is estimated so that they count less, and they get smaller
prior degrees of freedom of their own. With the method of moments, the log variances are Winsorized at their 5% and 90% quantiles ({api}`fit_f_distribution_robust`, which returns a {api}`RobustFDistributionFit`); fewer than 10 usable variances are too few to Winsorize at these quantiles, and the robust estimate is then the ordinary one, as in limma. With the likelihood estimator, which robust moderation uses when the degrees of freedom differ, nothing is Winsorized: the outlying genes get smaller prior weights, and the prior is fitted again, as limma does.

(moderated-f-statistics)=
## Moderated F-statistics

The moderated F-statistic tests that all the coefficients of a gene's fit, or all its contrasts, are zero: $F = t' R^{+} t / r$, with $t$ the moderated t-statistics, $R$ their correlation matrix and $r$ its rank, on $r$ and $d_0 + d_g$ degrees of freedom. When the fit holds a correlation matrix for each gene, as it does with missing values or observation weights, CytePool uses each gene's own, which gives the correct F-statistic [@smyth2004, section 7]; limma uses the correlation of the design without weights for every gene, which `f_statistic="limma"` reproduces. The two agree when one factorization serves every gene. Unlike those of the t-tests, the denominator degrees of freedom of the F-tests are not capped, as in limma.

(b-statistic)=
## The B-statistic

The B-statistic is the log-odds that a coefficient is not zero [@loennstedt2002], in the form of @smyth2004:

$$
B = \log\frac{p}{1 - p} + \frac{1}{2}\log r
+ \frac{1 + d}{2}\log\frac{t^2 + d}{t^2 r + d},
$$

where $p$ is the assumed proportion of coefficients that are not zero (0.01 by default), $d$ the degrees of freedom of the moderated t-statistic, $r = v / (v + v_0)$, $v$ the squared unscaled standard error, and $v_0$, estimated from the largest t-statistics, the prior variance of the coefficients that are not zero. When every gene has the same degrees of freedom and unscaled standard error, as in a fit without weights or missing values and with one prior for all genes, B orders the genes as $|t|$ does.

(treat-test)=
## Testing against a fold-change threshold

A small p-value says that a fold change is not zero, not that it is large. TREAT tests $H_0: |\beta| \le \tau$ against $|\beta| > \tau$ [@mccarthy2009]. Its p-value is the largest that any $\beta$ of the null interval allows, which the ends $\pm\tau$ give: for a moderated t-statistic $t$ with standard error $\mathrm{se}$, and $\delta = \tau / \mathrm{se}$,

$$
p = P(T > |t| - \delta) + P(T > |t| + \delta),
$$

where $T$ has the moderated t-statistic's degrees of freedom. {api}`treat` returns a {api}`TreatFit`, whose t-statistics are shrunk towards zero by the threshold, $\operatorname{sign}(t) \max(|t| - \delta, 0)$. Filtering significant genes by their estimated fold change does not test the threshold; TREAT does, and @mccarthy2009 found that it improves on the false discovery rate of existing methods when the size of the change is taken into account. Thresholds are typically fold changes of 1.1, 1.2 or 1.5, and since TREAT asks for evidence that a change exceeds the threshold, an estimated fold change must be clearly above it to be significant.

(empirical-bayes-defaults)=
## Defaults and choices

- **The estimator.** limma's default rule (`prior_estimator=None`), above; `"moments"` or `"likelihood"` fixes one, as limma's `legacy = TRUE` or `FALSE` does.
- **A trend.** None by default. `trend=True` lets the prior variance follow the mean log-expression, as limma-trend needs; voom's precision weights model the trend already, so limma-voom moderates without one ([limma-trend](limma_voom.md#limma-trend)).
- **Robust moderation.** Off by default, as in limma's `eBayes`; `robust=True` when a plot of the residual variances against expression shows outlying genes, as limma's `plotSA` does ({api}`plot_residual_variances`). {api}`propeller` moderates robustly by default, as speckle does.
- **F-statistics.** Exact by default; `f_statistic="limma"` gives limma's ([Moderated F-statistics](#moderated-f-statistics)).
- **The B-statistic.** `proportion=0.01`, and `stdev_coef_lim=(0.1, 4)` to bound the standard deviation of a coefficient that is not zero, as in limma.
- **The threshold of TREAT.** A fold change of 1.2 by default (`fc`), as in limma, or a log2 fold change (`lfc`); the threshold applies to the absolute log2 fold change, so a fold change and its inverse give the same test.

(empirical-bayes-limits)=
## Limits and differences

- **Robust moderation with unusable variances.** The robust likelihood estimator leaves the genes without a usable variance, such as those without residual degrees of freedom, out of its test for outliers. limma 3.62.1 counts a missing variance there as a zero one, which becomes an outlier and changes which other genes count as outliers. CytePool's estimator also keeps the prior weights in its second fit, where limma's drops them. Without unusable variances or prior weights, the two agree ([Validation](#validation-empirical-bayes) gives the sizes).
- **Too few variances.** {api}`fit_f_distribution_likelihood` needs at least two usable variances above zero, and raises an error otherwise, where limma returns NA. Where exactly two are above zero, both fit a constant prior, but limma returns NaN when other variances are not usable, which CytePool leaves out. With one or two genes nothing is moderated, and each gene keeps its own variance, as limma does since version 3.61.12.
- **Infinite covariates.** An infinite covariate is placed one unit beyond the largest or smallest finite one, as limma's `fitFDist` places it, where its `fitFDistUnequalDF1` returns NaN.
- **TREAT's t-statistic** is NaN where the coefficient is NaN, or where a contrast of zeros meets a threshold of zero; limma reports 0 for both.
- **Many degrees of freedom.** With residual degrees of freedom of ten million and more, limma's likelihood loses precision, and its prior degrees of freedom move away from those of the likelihood computed to 60 digits, which CytePool's agree with ([Validation](#validation-empirical-bayes)).

(empirical-bayes-checks)=
## Checks and further reading

Every estimator and statistic is compared with limma 3.62.1, run as a separate program on the same fits: exact (1e-10), and close (1e-6) for the robust prior, which limma solves only to a tolerance, and for the likelihood prior, whose search both programs stop at a tolerance ([Validation](#validation-empirical-bayes)).

Further reading:

- @smyth2004 introduced the hierarchical model of gene-wise variances and the moderated t-statistic, for microarray experiments with few arrays.
- @loennstedt2002 introduced the B-statistic, the log-odds of differential expression, for replicated microarray data.
- @phipson2016 estimated the prior robustly, which protects against hypervariable genes and improves the power to detect differential expression.
- @mccarthy2009 introduced TREAT, which tests whether a fold change exceeds a threshold.
- @chen2025 described edgeR 4, whose estimate of the prior of variances on unequal or fractional degrees of freedom {api}`fit_f_distribution_likelihood` follows.

[Multiple testing](multiple_testing.md) turns the p-values of thousands of genes into a list of calls with a known error rate.
