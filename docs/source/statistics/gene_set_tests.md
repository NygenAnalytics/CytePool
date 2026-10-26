---
description: Competitive and self-contained gene-set tests on a linear model, as limma's camera, fry, roast, mroast and romer compute them; gene set enrichment analysis of ranked genes, as fgsea computes it; over-representation analysis of the genes called significant, as limma's kegga computes it; scores of gene sets in every sample by GSVA, ssGSEA, combined z-scores and PLAGE, as the GSVA package computes them; gene sets read from GMT files; and how to choose between the tests.
---

# Gene-set tests

A gene-set test asks a question of a pathway or a signature as a whole, for one contrast
of the linear model fitted to every gene. Two questions differ [@goeman2007]:

- **Self-contained**: are the genes of the set differentially expressed at all?
  {api}`fry`, {api}`roast` and {api}`mroast`.
- **Competitive**: are they more differentially expressed than the other genes?
  {api}`camera` and {api}`romer`; {api}`gsea`, which asks it of a ranking of all genes;
  and {api}`over_representation`, which asks it of the genes called significant.

A self-contained test finds a set significant when its genes respond, however many other
genes respond too: in a response that touches thousands of genes, many sets are. A
competitive test asks whether a set stands out from the other genes, and so suits
ranking many sets, such as a pathway collection. The five tests of limma work on the
linear model, so they allow any design and contrast and do not permute samples; all but
romer take voom's precision weights. Gene set enrichment analysis needs only a statistic
of every gene, and over-representation analysis only the genes that an analysis called
significant.

## The effects of each gene

The five tests on the linear model start from the same decomposition [@wu2010, section 4.1]. The design is
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

## Gene-set files

Gene sets come from collections such as MSigDB [@liberzon2011; @liberzon2015],
Reactome, WikiPathways and the Gene Ontology, which distribute them as GMT files: one
set per line, with its name, a description and its genes, separated by tabs
[@subramanian2005; @gsea2026]. {api}`read_gmt` reads such a file into a
{api}`GeneSets`, a mapping of set name to gene ids, which every function on this page
takes as its gene sets. CytePool bundles no collection: each provider states its own
terms. The collection records the source, version and licence that you give when you
read it, with the name of the file and the SHA-256 digest of its bytes, so that a
result can say which sets it used; {api}`write_gmt` writes a collection back.

Sets name genes by symbol, by Entrez id or by Ensembl id, and an analysis must name
them the same way. {api}`GeneSets.restrict_to` keeps the genes of each set that an
analysis measured, drops the sets left without genes, as limma's `ids2indices` does,
and refuses ids that match no gene at all. {api}`GeneSets.filter_by_size` then keeps
the sets of a range of sizes, counted within the analysis: GSEA leaves out sets of
fewer than 15 or more than 500 genes by default, since two or three genes can make a
small set significant [@gsea2026]. {api}`GeneSets.select` takes sets by name.

## Over-representation analysis

Over-representation analysis asks whether the genes called significant hold more of a
set's genes than chance would give [@tavazoie1999]. Of a universe of $N$ genes, $n$
are selected; a set has $K$ genes in the universe, and $k$ of them are selected. Were
the $n$ genes drawn at random from the universe, $k$ would follow the hypergeometric
distribution, and the p-value is the chance of an overlap at least as large:

$$
P(X \ge k) = \sum_{j=k}^{\min(K, n)}
\frac{\binom{K}{j} \binom{N - K}{n - j}}{\binom{N}{n}},
$$

the one-sided p-value of Fisher's exact test of the table of genes in and out of the
set, selected or not [@fisher1922; @rivals2007]. The expected overlap is $K n / N$.
{api}`over_representation` computes it as limma's `kegga` and `goana` do [@young2010],
for the genes that go up, those that go down and those that change either way in a
group and contrast of a differential-expression result or in a table of results, or
for a list of genes; the adjusted p-values are Benjamini and Hochberg's over the sets
of each list. {api}`plot_over_representation` draws the top sets as dots for each
list, as clusterProfiler compares lists of genes [@yu2012].

The universe is the genes that could have been selected, by default the genes tested.
The genes of the genome, or of all the sets, include genes that were never expressed
or tested, and make the p-values too small [@timmons2015; @wijesooriya2022]. Genes of
the universe that are in no set stay in it, as in `kegga`.

The test takes genes, not samples, as the units drawn, independently of each other.
The genes of a set are often correlated, as co-regulated genes are, so its p-values
are too small for sets that respond together [@goeman2007], the problem that camera
solves for a linear model [@wu2012]. The result also depends on the cutoff
that selects the genes, and ignores how much each one changes. In RNA-seq data, long
and highly expressed genes are called significant more often, which favours sets of
such genes [@young2010]; the adjustment that `kegga` and `goana` offer for this is not
implemented.

## Gene set enrichment analysis

Gene set enrichment analysis (GSEA) ranks all genes by a statistic, such as the moderated
t-statistic of a contrast, and asks whether the genes of a set gather at the top or at
the bottom of the ranking [@subramanian2005]. A walk down the ranking of $N$ genes rises
at each of the $m$ genes of the set by its share of the set's weight and falls at each
other gene by $1/(N - m)$; after the $i$-th gene of the set, at rank $p_i$, it stands at

$$
\mathrm{RS}_i = \frac{\sum_{j \le i} |r_j|^q}{\sum_{j = 1}^{m} |r_j|^q}
- \frac{p_i - i}{N - m},
$$

with $r_j$ the statistics of the set's genes and $q$ the exponent: 1 for the weighted
score of GSEA, 0 for the unweighted one of @mootha2003. The enrichment score is the
walk's largest deviation from zero, positive when the set's genes gather at the top and
negative when they gather at the bottom, and the leading edge is the set's genes that
the walk passes before its extreme, which drive the score. {api}`gsea` computes them as
fgsea does [@korotkevich2021], from a group and contrast of a differential-expression
result, a table of results or a mapping of gene id to statistic. Genes with equal
statistics keep the order in which they are given, as in fgsea, so that the score of a
set whose genes tie with others depends on that order; a warning says how many tie.

The scores of random sets of the same size give the null distribution, so that the test
is competitive, like camera's, and it permutes genes, not samples. Like
over-representation analysis it treats genes as independent: the genes of a set that
respond together make its p-values too small [@goeman2007; @wu2012]. Each score is
divided by the mean of the random scores of its sign, so that sets of different sizes
compare, and the p-value is the share of the random scores of its sign that reach it.
With $n$ random sets it is at least $1/(n + 1)$; fgsea's multilevel method, the default,
reaches far smaller ones [@korotkevich2021]. It draws random sets of the size of the set,
keeps at each level the half with the higher scores and moves them by a Markov chain that
keeps their scores above the level's median, so that each level about halves the
probability, until the median reaches the observed score; `log2_error` gives the expected
standard deviation of $\log_2 p$.

The random sets come from seeds that `rng` draws, through the Mersenne Twister
[@matsumoto1998] seeded as C++ seeds it, as fgsea draws them: given fgsea's seeds,
{api}`gsea` gives fgsea's results exactly, and a seed gives the same results every time.
{api}`plot_enrichment` draws the walk of one set, as fgsea's `plotEnrichment` does, and
{api}`plot_gsea_table` the top sets with their genes on the ranking, as its
`plotGseaTable` does.

## Pathway scores per sample

The tests above ask one question of a contrast. A pathway score instead turns the
expression of a set's genes in each sample into one number, so that pathways can be
compared across samples, drawn as a heatmap, correlated with other measurements, and
tested with the same designs and contrasts as genes [@haenzelmann2013].
{api}`gsva`, {api}`ssgsea`, {api}`combined_z_scores` and {api}`plage` compute the four
methods of the GSVA package, from a samples x genes matrix or a cell type of a cube,
whose log-CPM after TMM normalization (or counts, for GSVA's Poisson kernel) they score
over the genes that `filterByExpr` keeps. They return a {api}`PathwayScores`, samples x
sets, with the samples' table and the genes each set used; {api}`PathwayScores.table`
lists the scores one per row.

**GSVA** first puts each gene on a common scale: the probability of a value at most as
high under the gene's distribution across the $n$ samples, estimated by a kernel,

$$
F_{ij} = \frac{1}{n} \sum_{k=1}^{n} \Phi\!\left(\frac{x_{ij} - x_{ik}}{h_i}\right),
$$

with the bandwidth $h_i$ a quarter of the gene's standard deviation, for continuous
values such as log-CPM, or the Poisson probability $P(Y \le x_{ij})$ with mean
$x_{ik} + 1/2$ for counts. Each sample ranks its $p$ genes by $F_{ij}$, and each gene
weighs $|p/2 - r_{ij}|$, so that the genes at both ends of the ranking weigh most. A walk
down the ranking rises at the genes of a set and falls at the others, as in gene set
enrichment analysis, and the score is the walk's largest deviation above zero minus its
largest deviation below: high for a set whose genes are higher than usual in the sample,
low for one whose genes are lower, and between $-1$ and $1$. GSVA estimates $\Phi$ from a
table of its values at steps of 0.001, and {api}`gsva` does the same, so that genes are
ranked as in GSVA.

**ssGSEA** ranks the genes of each sample by their values and sums the same walk over
all genes rather than taking its extreme, with the genes of the set weighted by their
ranks to the power $\alpha = 1/4$ [@barbie2009]. By default every score is divided by the
range of all the scores, over sets and samples, so that a score depends on the other
sets and samples scored with it. **Combined z-scores** standardize each gene across
samples and add the $k$ z-scores of a set, divided by $\sqrt{k}$ [@lee2008]. **PLAGE**
takes the standardized values of a set's genes as a genes x samples matrix and scores
the samples by its first right singular vector, the direction along which the genes vary
most together [@tomfohr2005]; its sign is arbitrary, and CytePool chooses the one that
makes the scores rise with the set's mean standardized expression.

ssGSEA ranks the values themselves, so that a set of highly expressed genes scores high
in every sample. GSVA first sets each gene against its own distribution across the
samples, so that a set scores high where its genes are higher than usual, compared with
the other genes of the sample. Combined z-scores and PLAGE also set each gene against
itself across samples, but do not compare it with other genes. GSVA's guide notes that
the scores of larger sets vary less; to test them, fit the design to the scores and let
the prior variance follow the set sizes, as its guide does with limma-trend:
`moderated_statistics(fit_linear_model(scores.scores, Design(scores.samples, "~
condition")), trend=scores.n_genes)`. {api}`plot_pathway_scores` draws the scores as a
heatmap, with the samples grouped by a variable.

The scores agree with GSVA's to rounding, except where CytePool differs on purpose:
PLAGE's sign, GSVA's walk in samples with missing values, and inputs on which GSVA
stops with an error. A gene's rank is decided by comparing its kernel estimate with
those of the other genes, and genes with equal estimates in exact arithmetic, such as
genes with the same low counts in different samples, can be ordered by the last bit of
R's probabilities: their ranks, and the scores of their sets, can then differ slightly
from GSVA's.

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
- **Over-representation or camera.** Over-representation analysis needs only the
  genes called significant, so it takes the results of any method and lists from
  elsewhere. camera uses every gene's statistic and allows for the correlation between
  genes, so with a linear model at hand it is the sounder competitive test.
- **GSEA or camera.** GSEA needs only a statistic of every gene, from any method, and
  uses the whole ranking rather than a cutoff; like over-representation analysis it
  takes genes as independent, which camera does not. Set `min_size` and `max_size`, as
  GSEA's 15 and 500, to leave out sets too small to rely on and too large to mean much.
- **Scores or tests.** Pathway scores serve analyses beyond one contrast: heatmaps,
  correlations and models with any design. For one contrast, camera, fry and GSEA test
  the genes' own statistics, where scores first reduce each set to one number per
  sample. GSVA and ssGSEA compare a set's genes with the other genes of each sample;
  combined z-scores and PLAGE follow each set across samples.
