---
description: Competitive and self-contained gene-set tests on a linear model, as limma's camera, fry, roast, mroast and romer compute them; over-representation analysis of the genes called significant, as limma's kegga computes it; gene sets read from GMT files; and how to choose between the tests.
---

# Gene-set tests

A gene-set test asks a question of a pathway or a signature as a whole, for one contrast
of the linear model fitted to every gene. Two questions differ [@goeman2007]:

- **Self-contained**: are the genes of the set differentially expressed at all?
  {api}`fry`, {api}`roast` and {api}`mroast`.
- **Competitive**: are they more differentially expressed than the other genes?
  {api}`camera` and {api}`romer`, and {api}`over_representation`, which asks it of the
  genes called significant.

A self-contained test finds a set significant when its genes respond, however many other
genes respond too: in a response that touches thousands of genes, many sets are. A
competitive test asks whether a set stands out from the other genes, and so suits
ranking many sets, such as a pathway collection. The five tests of limma work on the
linear model, so they allow any design and contrast and do not permute samples; all but
romer take voom's precision weights. Over-representation analysis needs only the genes
that an analysis called significant.

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
