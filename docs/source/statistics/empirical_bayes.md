---
description: Empirical Bayes moderation of gene-wise variances as limma computes it, its prior estimators, moderated t and F statistics, the B-statistic, and TREAT's test against a fold-change threshold.
---

# Empirical Bayes moderation and TREAT

With a few samples, each gene's residual variance $s_g^2$ is a poor estimate: some genes
look precise by chance and give large t-statistics. Empirical Bayes moderation borrows
strength from all genes by shrinking each variance towards a common value [@smyth2004].
Every test of a linear model in CytePool moderates this way: {api}`moderated_statistics`
and {api}`treat` on a fit, and the differential-expression, gene-set and composition
functions inside.

## The model

The residual variance of gene $g$ on $d_g$ degrees of freedom, and the true variance
$\sigma_g^2$, follow

$$
s_g^2 \mid \sigma_g^2 \sim \frac{\sigma_g^2\, \chi^2_{d_g}}{d_g}, \qquad
\frac{1}{\sigma_g^2} \sim \frac{\chi^2_{d_0}}{d_0 s_0^2},
$$

so that $s_g^2$ follows $s_0^2$ times an $F(d_g, d_0)$ distribution. The prior
$(s_0^2, d_0)$ is estimated from all genes, and each variance is replaced by its
posterior value

$$
\tilde s_g^2 = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g}.
$$

The moderated t-statistic $t_g = \hat\beta_g / (u_g \tilde s_g)$, with $u_g$ the
unscaled standard error of the coefficient or contrast, follows a t distribution on
$d_0 + d_g$ degrees of freedom when the coefficient is zero; its two-sided p-value comes
from the tail of that distribution ({api}`t_sf`), on fractional degrees of freedom too.
Large $d_0$ means the genes have similar variances and the prior counts for much;
$d_0 = 0$ is the ordinary t-test. {api}`moderated_statistics` returns them in a
{api}`ModeratedFit`, and {api}`squeeze_variances` applies the same moderation to any set
of variances ({api}`SqueezedVariances`).

## Estimating the prior

- **Method of moments** ({api}`fit_f_distribution`, limma's `fitFDist`): matches the
  mean and variance of $\log s_g^2$, corrected for its sampling distribution, to those
  of the scaled F distribution [@smyth2004, section 6.2], and returns a
  {api}`FDistributionFit`. With a covariate, as in limma-trend, $s_0^2$ follows a
  natural cubic spline ({api}`natural_spline_basis`) in the mean log-expression of the
  genes [@phipson2016, section 10.1].
- **Robust** ({api}`fit_f_distribution_robust`, `robust=True`): Winsorizes the log
  variances so that hypervariable genes do not lower $d_0$, and gives outlying genes a
  smaller prior df of their own [@phipson2016] ({api}`RobustFDistributionFit`).
- **Weighted moments and profile likelihood** ({api}`fit_f_distribution_likelihood`,
  limma's `fitFDistUnequalDF1`): for degrees of freedom that differ between genes or are
  fractional, as with missing values, zero weights, mixed models or the quasi-likelihood
  of edgeR 4 [@chen2025] ({api}`LikelihoodFDistributionFit`).

By default the likelihood estimator is used when the positive degrees of freedom differ
between genes and the method of moments otherwise, as in limma since version 3.61.8
(`legacy = NULL`).

## Moderated F and the B-statistic

The moderated F-statistic tests that all coefficients of a gene are zero:
$F = t' R^{+} t / r$, with $R$ the correlation matrix of the coefficients and $r$ its
rank. CytePool uses each gene's own correlation matrix when the fit has one (missing
values or observation weights), which is the correct F-statistic [@smyth2004, section
7]; limma uses the correlation of the unweighted design for every gene.

The B-statistic is the log-odds that a coefficient is non-zero [@loennstedt2002], in the
form of @smyth2004:

$$
B = \log\frac{p}{1 - p} + \frac{1}{2}\log r
+ \frac{1 + d}{2}\log\frac{t^2 + d}{t^2 r + d},
$$

with $p$ the assumed proportion of non-zero coefficients (0.01 by default),
$d = d_0 + d_g$, $r = v / (v + v_0)$, $v$ the squared unscaled standard error and $v_0$
the prior variance of non-zero coefficients, estimated from the largest t-statistics.
When every gene has the same degrees of freedom and unscaled standard error, as in a fit
without weights or missing values, B orders the genes as $|t|$ does.

## Testing against a fold-change threshold

A small p-value says a fold change is not zero, not that it is large. TREAT tests
$H_0: |\beta| \le \tau$ against $|\beta| > \tau$ [@mccarthy2009]. With the moderated
t-statistic $t$, its standard error $\mathrm{se}$ and $\delta = \tau / \mathrm{se}$, the
p-value is the worst case over the null interval,

$$
p = P(T > |t| - \delta) + P(T > |t| + \delta),
$$

with $T$ on $d_0 + d_g$ degrees of freedom. {api}`treat` returns a {api}`TreatFit` and
takes the threshold as a log2 fold change (`lfc`) or a fold change (`fc`, 1.2 by
default, as in limma). Filtering significant genes by their estimated fold change does
not test the threshold; TREAT does, and @mccarthy2009 found that it improves on the
false discovery rate of existing methods when the size of the change is taken into
account.
