---
description: Competitive and self-contained gene-set tests on a linear model, as limma's camera, fry, roast, mroast and romer compute them, and how to choose between them.
---

# Gene-set tests

A gene-set test asks a question of a pathway or a signature as a whole, for one contrast
of the linear model fitted to every gene. Two questions differ [@goeman2007]:

- **Self-contained**: are the genes of the set differentially expressed at all?
  {api}`fry`, {api}`roast` and {api}`mroast`.
- **Competitive**: are they more differentially expressed than the other genes?
  {api}`camera` and {api}`romer`.

A self-contained test finds a set significant when its genes respond, however many other
genes respond too: in a response that touches thousands of genes, many sets are. A
competitive test asks whether a set stands out from the other genes, and so suits
ranking many sets, such as a pathway collection. Every test here works on the linear
model, so it allows any design and contrast and does not permute samples; all but romer
take voom's precision weights.

## The effects of each gene

All five tests start from the same decomposition [@wu2010, section 4.1]. The design is
reparametrized so that the contrast is its last coefficient, and the QR decomposition of
that design turns each gene's data into a contrast effect $u_0$ and $d$ residual effects
$u_1, \dots, u_d$, independent with variance $\sigma^2$ under the model. The contrast
effect over the residual standard deviation is the gene's t-statistic.

## camera

Genes are correlated, so a set's genes are not independent evidence. camera turns each
gene's moderated t-statistic into a z-score and compares the mean z-score of the set's
$m$ genes with that of the other genes by a two-sample t-test, with the variance of the
set's mean inflated by

$$
\mathrm{VIF} = 1 + (m - 1)\rho,
$$

for an inter-gene correlation $\rho$ [@wu2012]. The default $\rho = 0.01$ is limma's;
`inter_gene_correlation=None` estimates it for each set from its residual effects.
`use_ranks=True` gives the rank version, a correlation-adjusted Wilcoxon-Mann-Whitney
test.

## fry

fry averages each gene's standardized effects over the set into a vector $b$ of $d + 1$
values. Under the null hypothesis a random rotation of the residual space leaves the
data's distribution unchanged, so

$$
T = \frac{b_0}{\sqrt{\sum_{k=1}^{d} b_k^2 / d}}
$$

follows a t distribution on $d$ degrees of freedom: the p-value that roast with the mean
statistic would give with infinitely many rotations [@giner2016]. fry is fast enough for
thousands of sets and gives a directional and a mixed p-value for each.

## roast and mroast

roast draws random rotations of the residual space and compares a set statistic with its
values under them [@wu2010; @langsrud2005]. The statistic summarizes the genes'
z-scores: their mean (the default), which detects sets in which most genes change in the
same direction, or a floored mean, the mean of the top half or the mean square, which
detect sets in which fewer genes change, the mean square the fewest. With $b$ of $B$
rotations at least as extreme as the observed statistic, the p-value is

$$
p = \frac{b + 1}{B + 1}
$$

[@phipson2010], so it is never zero; each rotation also gives its mirror image for the
up and down alternatives. roast reports the proportion of the set's genes that are
active in each direction. {api}`mroast` tests many sets with the same rotations and
adjusts their p-values. Rotations come from a seeded numpy generator, so a seed gives
the same p-values every time; they agree with limma's within Monte Carlo error.

## romer

romer ranks every gene's moderated t-statistic among all genes and compares a set's mean
rank with its values when every gene's rotated statistics are ranked again
[@majewski2010]. Like gene set enrichment analysis [@subramanian2005] it pits the sets
against each other, but by rotating the residual space instead of permuting samples, so
it works for any contrast with residual degrees of freedom. Before rotating, each gene's
contrast effect is shrunk by its posterior probability of being differentially
expressed, so that strongly responding genes do not inflate the rotated statistics.

## Choices

- **Competitive or self-contained.** Rank a collection with camera; ask about one set
  chosen in advance, such as an interferon signature in an interferon stimulation, with
  fry or roast.
- **Correlation.** camera's small default correlation ranks co-regulated sets higher; an
  estimated correlation controls the error rate strictly for each set.
- **Rotations.** roast's smallest p-value is $1/(B+1)$; use enough rotations for the
  number of sets adjusted for.
- **Weights.** Pass a voom result to use its precision weights; romer takes sample
  weights only.
