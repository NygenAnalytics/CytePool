---
description: Whether the genes of a pathway respond to a contrast, tested on the linear model of every gene by limma's camera, fry, roast, mroast and romer.
---

# Gene-set tests

The chapters so far asked which genes change ([limma-voom and limma-trend](limma_voom.md))
and which cell types expand or contract ([Composition tests](composition.md)). This part
asks the question of a pathway or a signature as a whole: do the genes of a set respond
to a contrast? {api}`camera` answers it by default for a collection of sets: it asks
whether the genes of each set change more than the other genes, and allows for the
correlation between them [@wu2012]. {api}`fry`, {api}`roast` and {api}`mroast` ask whether
the genes of a set change at all, and {api}`romer` compares the ranks of a set's genes
with those of the others. All five test one contrast of the linear model that
{api}`limma_voom` or {api}`limma_trend` fitted to a cell type. [Gene sets and
pathways](../guides/gene_sets.ipynb) runs camera, fry and roast on published data.

(gene-set-collections)=
## Gene-set collections

A gene set names genes that share a function or a response, such as a pathway, the
targets of a transcription factor or a signature. Collections such as MSigDB
[@liberzon2011; @liberzon2015], Reactome and WikiPathways distribute them as GMT files,
and so do the Gene Ontology's GMT exports, such as MSigDB's: one set per line, a name,
a description and the genes, in fields separated by tabs [@subramanian2005; @gsea2026].
{api}`read_gmt` reads such a file into a {api}`GeneSets`, a mapping from set names to
gene ids that every function of this part takes as its gene sets, and {api}`write_gmt`
writes one back. CytePool bundles no collection, since each provider states its own
terms. A collection records the name, version and licence that you give when you read
it, with the name and SHA-256 digest of the file, so that a result can say which sets it
used.

Sets name genes by symbol, by Entrez id or by Ensembl id, and an analysis must name them
the same way: the tests read a result's ids from a column of its features (`id_column`,
such as `"symbol"`). {api}`GeneSets.restrict_to` matches a collection to the genes that
an analysis measured, as limma's `ids2indices` does, and its error when nothing matches
is the sign that the two name genes differently. {api}`GeneSets.filter_by_size` then
keeps sets by their size among the genes measured, and {api}`GeneSets.select` by name.
The GSEA program skips sets of under 15 or over 500 genes by default [@gsea2026]; the
gene-set tests, GSEA and the pathway scores of CytePool test sets of every size unless
told otherwise, as fgsea and GSVA do.

(competitive-and-self-contained)=
## Competitive or self-contained

Two questions can be asked of a set's genes [@goeman2007]. A *self-contained* test asks
whether they respond to the contrast at all: {api}`fry`, {api}`roast` and {api}`mroast`
ask it. A *competitive* test asks whether they respond more than the other genes:
{api}`camera` and {api}`romer` ask it, as do the over-representation analysis and the
gene set enrichment analysis of the [next chapter](over_representation_and_gsea.md). In
a response that touches thousands of genes, the genes of many sets respond, and a
self-contained test finds each of those sets significant; a competitive test asks which
sets respond more than the other genes, and so ranks a collection. A self-contained test
suits a set chosen in advance, such as an interferon signature in an interferon
stimulation.

(correlated-genes)=
## Genes that respond together

The genes of a set are seldom independent: co-regulated genes rise and fall together
from sample to sample. A test that takes them as independent draws, such as one that
compares a set with random sets of genes, counts one shared fluctuation many times, and
its p-values are too small for sets whose genes are correlated [@goeman2007; @wu2012].
The tests of this chapter allow for it in two ways. camera inflates the variance of a
set's mean statistic by a factor that grows with the correlation between its genes. fry,
roast, mroast and romer compare a set's statistic with its values under rotations of the
residual space, which keep the correlation between genes as it is.

(gene-effects)=
## The effects of each gene

The five tests start from the same decomposition [@wu2010, section 4.1]. The design, of n
samples and p coefficients, is reparametrized to make the contrast its last
coefficient, and the QR decomposition of that design turns the values of a gene into a
contrast effect $u_0$ and $d = n - p$ residual effects $u_1, \dots, u_d$. Under the
model they are independent, each with the gene's variance $\sigma^2$, and $u_0$ over the
residual standard deviation is the gene's t-statistic. With precision weights, each
gene's values and design are first multiplied by the square roots of its weights.

Under the null hypothesis $u_0$ has mean zero too, so the $d + 1$ effects of a gene are
independent with equal variances, and a rotation of them leaves their distribution
unchanged. One rotation applied to the effects of every gene then leaves the joint
distribution of the data unchanged, the correlation between genes included [@wu2010;
@langsrud2005]. Rotations take the place of permutations of the samples, which a small
experiment has too few of and a design with several factors or covariates does not
allow: they need only residual degrees of freedom.

A result of {api}`limma_voom` or {api}`limma_trend` is tested on the fit of one cell type,
which {api}`DifferentialExpressionResult.fit` computes again: its log-CPM values and
precision weights, the design rows of the samples fitted, the contrast's weights on the
columns kept, and the moderation the result used. Values with a design and a contrast
are tested the same way.

(camera)=
## camera

camera turns each gene's moderated t-statistic into a z-score and compares the mean
z-score of the set's $m$ genes with that of the other genes by a two-sample t-test, with
the variance of the set's mean inflated by

$$
\mathrm{VIF} = 1 + (m - 1)\rho,
$$

where $\rho$ is the correlation between the set's genes [@wu2012]. The default
$\rho = 0.01$ is limma's. `inter_gene_correlation=None` estimates $\rho$ for each set
from the residual effects of its genes, each divided by the gene's residual standard
deviation: the sum of squares of their mean over the set, times $m/d$, estimates the
variance inflation factor. `use_ranks=True` gives the rank version, a
Wilcoxon-Mann-Whitney test of the set's z-scores against the others, with the variance
of the rank sum adjusted for the correlation [@barry2008; @wu2012]. The p-value is
two-sided, and the direction says whether the set's genes went up or down.

(fry)=
## fry

fry asks whether a set's genes respond in one direction, for the directional p-value, or
each in its own direction, for the mixed p-value [@giner2016]. It divides each gene's
$d + 1$ effects by an estimate of the gene's standard deviation and averages them over
the set's genes into a vector $b$. Under the null hypothesis a random rotation turns $b$
uniformly over its sphere, so

$$
T = \frac{b_0}{\sqrt{\sum_{k=1}^{d} b_k^2 / d}}
$$

follows a t distribution on $d$ degrees of freedom. When the prior degrees of freedom are
very large, as when every gene has the same residual variance, this is the p-value that
roast with the mean statistic gives with infinitely many rotations; otherwise fry
approximates it. The mixed statistic sums the squared standardized contrast effects of
the set's genes, and its p-value under rotation is approximated by the beta distribution
whose mean and variance match its own; for a set of one gene it equals the gene's
two-sided t-test p-value.

By default (`standardize="posterior_sd"`) each gene is divided by a posterior standard
deviation: the mean square of its $d + 1$ effects with the largest left out, so that a
gene's own strong response does not inflate it, moderated by the prior of [empirical
Bayes moderation](empirical_bayes.md). `"residual_sd"` divides by the root of that mean
square, unmoderated, and `"none"` leaves the effects as they are, which suits genes of
equal variance. fry draws no rotations, so it is fast enough for thousands of
sets, and its p-values are not limited by a number of rotations.

(roast)=
## roast and mroast

roast draws random rotations of the residual space and sets the observed set statistic
against the statistics of the rotated data [@wu2010; @langsrud2005]. Each gene's
moderated t-statistic becomes a z-score, and the statistic summarizes the z-scores of the
set's genes, for the alternatives up, down and mixed:

- `"mean"`, the default, their mean, which makes a set significant when most of its genes
  change in the same direction;
- `"floormean"`, the mean with values on the wrong side of zero, or for the mixed
  alternative small values, raised to a floor, and `"mean50"`, the mean of the larger
  half, which detect sets in which as few as a quarter of the genes change;
- `"msq"`, the mean square, which detects smaller shares still when their changes are
  large, but which a few genes can drive.

If $b$ of the $B$ rotated statistics are at least as extreme as the observed one, the
p-value is

$$
p = \frac{b + 1}{B + 1}
$$

[@phipson2010], so it is never zero. The two-sided p-value asks whether the set's genes
went one way, up or down, and the mixed p-value whether they changed in either direction
gene by gene. Each rotation also gives its mirror image for the up and down
alternatives, so their p-values count $2B$ rotations: p-values reach down to
$1/(B + 1)$ for the two-sided and mixed alternatives and $1/(2B + 1)$ for up or down,
0.0005 and 0.00025 with the default 1,999 rotations. roast also reports the proportion
of the set's genes that are active in each direction, with z-scores beyond $\sqrt 2$ in
size [@wu2010]. Contribution weights of opposite signs mark genes expected to change in
opposite directions, as in a signature of genes up and genes down, and larger weights
count more (`gene_weights`, or a mapping of gene to weight as the set).

{api}`mroast` tests many sets with the same rotations and adjusts their p-values for the
number of sets by Benjamini and Hochberg's method, which it applies to mid-p-values
[@routledge1994], since they suit false discovery rates, as limma does by default. The
rotations come from a numpy generator: a seed (`rng`) gives the same p-values every
time.

(romer)=
## romer

romer is a competitive test by ranks [@majewski2010]. It ranks every gene's moderated
t-statistic among all genes, ranking $t$ for the up alternative, $-t$ for down and
$|t|$ for mixed, and compares a set's mean rank with the mean ranks that the rotations
give, each rotation ranking every gene again; the mean, the floor mean or the mean of the
larger half summarizes the ranks, as for roast. Like gene set enrichment analysis it
compares a set's genes with the other genes, but its null distribution comes from
rotations of the residual space, not from random sets of genes. Before the rotations,
romer shrinks each gene's contrast effect according to the gene's posterior probability
of differential expression, so that the genes that respond strongly do not inflate the
rotated statistics. Its p-values are $(b + 1)/(B + 1)$, at least 0.0001 with the default
9,999 rotations, and are not adjusted for the number of sets, as in limma.

(gene-set-tests-choices)=
## Defaults and choices

- **Competitive or self-contained.** {api}`camera` ranks a collection by default; for one
  set chosen in advance, {api}`fry` or {api}`roast` asks whether it responds at all
  ([Competitive or self-contained](#competitive-and-self-contained)).
- **The correlation.** camera's small default correlation, 0.01, ranks highly
  co-regulated sets, which are often the interpretable ones, higher; a correlation
  estimated for each set (`inter_gene_correlation=None`) controls the error rate
  strictly for every set, as limma's documentation says of the two. Negative
  correlations count as zero unless `allow_negative_correlation=True`.
- **fry or roast.** fry approximates the directional p-value of mroast's mean statistic
  without rotations, so it suits many sets. Its mixed p-value comes from an
  approximation, and roast's from the rotations themselves.
- **The set statistic.** `set_statistic="mean"`, limma's default, asks that most genes
  change together; `"floormean"`, `"mean50"` and `"msq"` find sets in which fewer genes
  change.
- **Rotations.** roast and mroast draw 1,999 rotations by default and romer 9,999, as
  limma does. Use enough of them for the number of sets that the p-values are adjusted
  for: limma's documentation recommends at least 9,999 for publication.
- **Weights.** A result of limma-voom is tested with its precision weights. romer takes
  sample weights only, as limma's does, so it tests a result of {api}`limma_trend`.
- **The moderation.** camera, fry, roast and mroast moderate the variances as the result
  did: with the trend of limma-trend, and robustly when the result was ([Estimating the
  prior](#estimating-the-prior)); `trend` and `robust` change it. romer always moderates
  them without a trend and not robustly, as limma's romer, which offers neither.
- **Gene ids.** `id_column` names the column of a result's features that holds the ids
  the sets use, `"feature_id"` by default.

(gene-set-tests-limits)=
## Limits and differences

- **Fixed effects only.** The tests fit a linear model of fixed effects: a design with
  random effects is refused, there is no block correlation, and a result of
  {api}`dream_voom` is refused, as its fits are mixed models. Test a result of
  {api}`limma_voom` with the donor as a fixed effect instead. Every value must be
  observed, with a positive weight, so that every gene has the same residual degrees of
  freedom.
- **Not implemented.** fry's gene contribution weights (limma's `gene.weights`), and exact
  z-scores and given priors in the rotation tests (`approx.zscore = FALSE`, `var.prior`,
  `df.prior`) ([Not implemented yet](#not-implemented-yet)).
- **Rotations.** CytePool draws its rotations from numpy, and limma from R, so their
  p-values agree within Monte Carlo error; given limma's rotations, CytePool returns
  limma's p-values. mroast tests every set with the same rotations, where limma draws new
  ones for each set; each p-value is a rotation p-value either way.
- **fry's mixed p-value of small sets.** For a set of 2 to $d$ genes, limma approximates
  the rotation p-value with the set's own dimensions only, which can be far from it;
  CytePool uses all $d + 1$ dimensions, as for larger sets. For larger sets the two
  agree, and the approximation that both use can be much too small far in the tail.
- **fry's posterior standard deviations.** limma's expected value of a gene's mean square
  without its largest effect is slightly above the exact value that CytePool uses, which
  moves fry's p-values by a small fraction of their value.
- **Equal statistics in roast.** roast and mroast count a rotated statistic that equals
  the observed one as at least as extreme, as Wu et al. and Phipson and Smyth define the
  p-value, and as limma's romer counts it; limma's roast counts only larger ones. The two
  differ only where a floor-mean or mean-square statistic is the smallest it can be: a
  set with no gene on one side of zero has a one-sided p-value of 1 here and about 1/2
  in limma, and a set whose z-scores all lie within $\pm 0.674$ has a mixed floor-mean
  p-value of 1 here.
- **Contribution weights.** A gene whose weight is zero is dropped from the set, and the
  mean square and the mixed floor mean weigh each gene by the size of its weight, as the
  paper does. limma keeps a gene of weight zero in the set; its mean-square p-values
  change, beyond 1,000 rotations, when every weight is multiplied by the same number, and
  with weights of both signs its up, down and two-sided ones depart from the paper's; and
  its mixed floor mean takes a negative weight as evidence against the set.
- **Sets without meaning.** A set without genes, or that holds every gene in camera, has
  no p-value and no direction, and the adjustment of the others leaves it out; limma
  reports a direction for it, and in some cases a p-value. A gene that a set lists twice
  counts once, where limma counts it twice, and camera's rank test reports 1 where its
  continuity correction puts both tails above 1/2 and limma reports a p-value above 1.
- **Values that overflow.** romer's shrinkage stays finite for genes with very large
  effects, where limma's becomes NaN beyond a log-odds of 709.8 (an absolute t above
  about 38 with an infinite prior df) and changes the p-values of every set, and the
  z-scores of the rotation tests stay finite for t-statistics whose square overflows.
- **Bases of the residual space.** With observation weights each gene's residual effects
  have a basis of their own, on which camera's estimated correlations and fry's
  standardized statistics depend. Where a pivot of the decomposition is zero but for
  rounding, limma leaves rounding noise in a few percent of genes, a different few in
  camera and in fry, so that its basis there, and with it its results, can change with
  the last bits of the data; CytePool's does not.
- **Robust camera.** camera takes `robust`, which limma's camera does not offer.

(gene-set-tests-checks)=
## Checks and further reading

Each test is compared with limma 3.62.1, run as a separate program on the same data:

- camera and fry: exact (1e-10); camera with robust moderation close (1e-6); fry's
  default standardization, with or without a trend or robust moderation, within 1e-3, as
  it differs on purpose; fry's mixed p-values within 0.1 of a simulation of the rotation
  test ([Validation](#validation-gene-set-tests)).
- roast, mroast and romer: exact (1e-12) when given the rotations that limma drew,
  mroast against limma run on each set from the same seed, and calibrated, within Monte
  Carlo error, when each program draws its own
  ([Validation](#validation-rotation-gene-set-tests)).
- GMT files read and written, and sets matched to an analysis as limma's `ids2indices`
  matches them: identical, against R 4.4.3, decoupler 2.2.0's `read_gmt` and limma
  3.62.1 ([Validation](#validation-gene-set-files-and-ora)).

Further reading:

- @goeman2007 set out the competitive and self-contained questions that a gene-set test
  can ask, and the problems of tests that take genes as independent.
- @wu2012 showed that gene set tests that permute genes are very sensitive to the
  correlation between genes, and introduced camera, which allows for it.
- @wu2010 introduced ROAST, which rotates the residuals of a linear model instead of
  permuting samples, so that it applies to almost any design.
- @langsrud2005 introduced rotation tests for multivariate linear models.
- @giner2016 presented fry, which approximates roast's test of the mean without
  rotations.
- @majewski2010 described romer, the rotation test of the mean ranks of a set's genes.

[Over-representation and GSEA](over_representation_and_gsea.md), the next chapter,
needs no linear model.
