---
description: How CytePool turns a sample table and an R formula into a design matrix, fits one linear model to every gene, and tests named contrasts, with the checks that catch designs that cannot answer the question.
---

# Designs, linear models and contrasts

## The design

A {api}`Design` joins the sample table, the fixed effects as an R formula, optional
random effects in lme4's notation and named contrasts. The formula follows the notation
of @wilkinson1973 as R reads it [@chambers1992]: `~ condition + batch` adds terms, `a:b`
is an interaction, `a * b` is `a + b + a:b`, and `~ 0 + group` has no intercept.
{api}`model_matrix` codes it as R's `model.matrix` does: a numeric column is a
covariate, a text or boolean column is a factor coded by treatment contrasts, indicators
of every level but the first.

Columns are named as R names them: the intercept `(Intercept)` ({api}`INTERCEPT`), a
level after its factor (`conditionstim`), interactions joined by `:`, and names that are
not syntactic in R between backticks ({api}`quote_name`). {api}`parse_formula` expands a
formula into its terms ({api}`FormulaTerms`), {api}`parse_random_effects` reads random
effects ({api}`RandomEffect`), and {api}`model_matrix` returns the matrix with its
column names ({api}`ModelMatrix`).

The first level of a factor is the reference, and every coefficient of that factor is a
difference from it. Without `levels`, the levels of a text column are sorted by code
point, which is R's order only in the C locale; pass
`levels={"condition": ["ctrl", "stim"]}` whenever the reference matters, which is almost
always.

## One model for every gene

Every gene $g$ of a samples x genes matrix of log-expression values is fitted with the
same design matrix $X$ (limma's `lmFit`; @smyth2004):

$$
E(y_g) = X a_g, \qquad \operatorname{var}(y_g) = \sigma_g^2\, W_g^{-1/2} R\, W_g^{-1/2},
$$

where $W_g$ holds the gene's precision weights (from voom, or sample weights) and $R$ is
the identity, or the correlation matrix of a known correlation between the samples of a
block [@smyth2005]. {api}`fit_linear_model` returns a {api}`LinearModelFit` with, for
every gene, the coefficients $\hat a_g$, the residual variance $s_g^2$ on $d_g$ residual
degrees of freedom, and the unscaled standard errors $u_{gk}$, so that the standard
error of coefficient $k$ is $u_{gk} s_g$. A gene with missing values, or with weights of
its own, is fitted to its observed samples by a QR decomposition of its own
[@golub2013]; missing values and zero weights lower its $d_g$.

## Contrasts

A contrast is a combination $c$ of the coefficients, such as the difference between two
groups. {api}`make_contrasts` reads named expressions such as `"groupB - groupA"` or
`"(groupB + groupC)/2 - groupA"` written in the design's column names, as limma's
`makeContrasts` does [@ritchie2015], into {api}`Contrasts`, a mapping of contrast names
to weights; {api}`Contrasts.select` takes some of them by name, and
{api}`Design.with_contrasts` returns the design with other contrasts.
{api}`contrasts_fit` re-expresses each gene's fit: the estimate is $c' \hat a_g$ and its
unscaled variance $c' V_g c$, with $V_g$ the gene's unscaled covariance matrix.
Contrasts are matched to the coefficients by name, never by position.

With weights that differ between genes, limma's `contrasts.fit` approximates the
standard errors of contrasts of correlated coefficients from the covariance of the
unweighted design. CytePool computes them exactly by default; `method="limma"`
reproduces the approximation.

## Designs that cannot answer the question

A contrast is estimable when it is a combination of the rows of $X$, that is, when it is
orthogonal to the null space of $X$ [@searle1971]. In a design of full rank every
contrast is. Rank deficiency arises when a column is a combination of others, decided as
R's `qr()` decides it: for example when two factors group the samples alike, or, within
one cell type, when every sample of a donor was left out. {api}`limma_voom` then fits
the column space of the group's design and tests only the contrasts its samples can
estimate, reporting the others.

{api}`Design.check` reports such problems before anything is fitted, as a
{api}`DesignReport` of {api}`DesignIssue` entries ({api}`DesignReport.kinds` lists their
kinds), each with what to do: aliased columns, no residual degrees of freedom,
confounded factors, empty levels, levels with one sample, samples fitted exactly
(leverage 1, the diagonal of the hat matrix; @hoaglin1978), numeric codes that look like
batches, labels that look like missing values, contrasts that cannot be estimated, and
random effects whose variances cannot be estimated.
