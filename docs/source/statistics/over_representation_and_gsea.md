---
description: Whether a pathway's genes are over-represented among the genes that change, or gather at the top or bottom of a ranking of all genes, by over-representation analysis and GSEA.
---

# Over-representation and GSEA

The tests of [Gene-set tests](gene_set_tests.md) read the linear model of every gene.
Often there is less to go on: a list of the genes that an analysis called significant,
or one statistic for every gene, from any method. This chapter tests gene sets from
those. {api}`gsea` is the default along a ranking: it asks whether each set's genes sit
near the top or the bottom when all genes are ordered by a statistic, as fgsea computes
it [@subramanian2005; @korotkevich2021]. {api}`over_representation` asks whether a set
has more of its genes among those called significant than chance would give, as limma's
`kegga` computes it [@young2010]. [Gene sets and
pathways](../guides/gene_sets.ipynb) runs both on published data.

(over-representation)=
## Over-representation analysis

Say that $n$ genes of a universe of $N$ are selected, such as the genes called
significant, and that a set has $K$ genes in the universe, $k$ of them selected. Were the
$n$ genes drawn at random from the universe, $k$ would be hypergeometric with mean
$K n / N$, the overlap expected by chance, and the p-value is the chance of an overlap
at least as large [@tavazoie1999]:

$$
P(X \ge k) = \sum_{j=k}^{\min(K, n)}
\frac{\binom{K}{j} \binom{N - K}{n - j}}{\binom{N}{n}},
$$

which is also the p-value of Fisher's exact test, one-sided, for the table of the genes in
and out of the set, selected or not [@fisher1922; @rivals2007].

{api}`over_representation` selects the genes of a cell type and contrast of a
differential-expression result, or of any table of results, whose adjusted p-value is
below `alpha`. Like `kegga` and `goana` on a fit of limma, it tests the genes up and the
genes down as two lists, and adds both together as a third. A list of genes from
elsewhere is tested as it is, against a universe that you give. The p-values of each list
are adjusted over its sets by Benjamini and Hochberg's method [@benjamini1995], and
{api}`plot_over_representation` draws the sets with the smallest p-values in any list,
with a dot for each list, as clusterProfiler compares lists of genes [@yu2012].

(over-representation-universe)=
### The universe

The universe holds every gene that the analysis could have selected, which by default
means every gene with a p-value. The genes of the genome, or of all the sets, include
genes that were never expressed, or were filtered out before the tests, and so could not
have been selected; they make the p-values too small [@timmons2015; @wijesooriya2022].
Genes outside a universe that you give are left out, even when selected or in a set, and
genes of the universe that belong to no set still count, as in `kegga`.

(gsea)=
## Gene set enrichment analysis

Gene set enrichment analysis (GSEA) orders all genes by a statistic, such as a contrast's
moderated t-statistic, and asks whether a set's genes crowd towards either end of the
ranking [@subramanian2005]. It needs no cutoff, and the statistic can come from any
method. A walk down the ranking of $N$ genes steps up at each of the set's $m$ genes and
down at every other gene; after the $i$-th gene of the set, at rank $p_i$, it stands at

$$
\mathrm{RS}_i = \frac{\sum_{j \le i} |r_j|^q}{\sum_{j = 1}^{m} |r_j|^q}
- \frac{p_i - i}{N - m},
$$

with $r_j$ the statistics of the set's genes and $q$ the `exponent`: 1 for the weighted
score of GSEA, 0 for the unweighted one of @mootha2003. The enrichment score is the
walk's furthest point from zero: positive when the set's genes gather at the top and
negative when they gather at the bottom. The leading edge is the genes that drive the
score: the set's genes up to the walk's extreme, counted from the top for a positive
score and from the bottom for a negative one.

{api}`gsea` ranks the statistics of a cell type and contrast of a differential-expression
result, of a table of results, or of gene ids mapped to statistics, and tests each set
with at least one gene in the ranking; a set of every ranked gene is not tested.
{api}`plot_enrichment` draws the walk of one set, as fgsea's `plotEnrichment` does, and
{api}`plot_gsea_table` the top sets with their genes on the ranking, as its
`plotGseaTable` does.

(gsea-p-values)=
### p-values from random sets

The scores of random sets of the same size give the null distribution, so that the test
is competitive, like camera's, and it permutes genes, not samples. Dividing a score by
the mean random score of its sign gives the normalized enrichment score, which lets sets
of different sizes compare, and the p-value is the share of the random scores of its sign
that reach it: with $n$ random scores of that sign, at least $1/(n + 1)$. fgsea's
multilevel method, the default, reaches far smaller p-values [@korotkevich2021]. It draws
random sets of the size of the set, keeps at each level the half with the higher scores
and moves them by a Markov chain that keeps their scores above the level's median, so
that each level about halves the probability, until the median reaches the observed
score; `log2_error` estimates the standard deviation of $\log_2 p$. With the multilevel
method, a set with under 10 random scores of its sign gets no p-value, as in fgsea.

The random sets come from seeds that `rng` draws, through the Mersenne Twister
[@matsumoto1998] seeded as C++ seeds it, as fgsea draws them: a seed gives the same
results every time, and given fgsea's seeds, {api}`gsea` gives fgsea's results exactly.
CytePool's routines are adapted from fgsea's, under its MIT licence ([Related
software](../reference/related_software.md)).

(over-representation-and-gsea-defaults)=
## Defaults and choices

- **GSEA or over-representation.** {api}`gsea` uses the whole ranking, without a cutoff.
  {api}`over_representation` needs only the genes called significant, so it takes the
  results of any method and lists from elsewhere. With a linear model at hand,
  {api}`camera` uses every gene's statistic and allows for the correlation between
  genes, so it is the sounder competitive test ([camera](#camera)).
- **The statistic.** gsea ranks genes by the moderated t-statistic of a result
  (`statistic_column="t"`); any column of a table, such as the log fold change, can rank
  them instead.
- **The weight of each gene.** `exponent=1` weighs the genes of a set by the size of
  their statistics, as GSEA does; `exponent=0` weighs them alike.
- **One end of the ranking.** `score_type="std"` takes the larger deviation of the walk
  in either direction; `"pos"` and `"neg"` look only at the top or the bottom, as for
  statistics that are all positive, such as $-\log_{10} p$.
- **p-values.** The multilevel method by default, after 1,000 permutations, with 101 sets
  at each level and p-values down to `eps`, 1e-50; `method="permutation"` takes the
  permutation p-values alone, as fgsea's `fgseaSimple` does.
- **Set sizes.** gsea tests sets of every size, as fgsea does; `min_size=15` and
  `max_size=500` reproduce the GSEA program's defaults ([Gene-set
  collections](#gene-set-collections)).
- **The cutoff.** over_representation selects genes with an adjusted p-value below
  `alpha`, 0.05, read from `p_value_column`, which may name another adjustment, such as
  `"adjusted_p_value_across_groups"`. Without directions, `fold_change_column=None`, the
  selected genes are tested as one list.

(over-representation-and-gsea-limits)=
## Limits and differences

- **Genes taken as independent.** Both tests take genes, not samples, as the units drawn,
  independently of each other, so the genes of a set that respond together make its
  p-values too small ([Genes that respond together](#correlated-genes)).
- **A cutoff and a bias.** Over-representation analysis depends on the cutoff that
  selected the genes and ignores how much each one changes. In RNA-seq data, long and
  highly expressed genes are called significant more often, which favours sets of such
  genes [@young2010]; the adjustment for it that `kegga` and `goana` offer (`trend`) is
  not implemented.
- **Tied statistics.** As in fgsea, tied genes are ranked in the order given, so a set's
  score can depend on that order; a warning counts the ties.
- **Differences from kegga.** A set with no genes in the universe gets a row without a
  p-value; `kegga` drops it from its table. For a fit of limma, `kegga` counts genes
  without a p-value in its universe, and adjusts the p-values of only the genes that have
  an id, so that which genes it selects depends on which genes have an id, such as a
  symbol. CytePool takes a gene without a p-value as not tested, and the adjusted p-values
  of the results as they are, the genes that the analysis called significant. `kegga` and
  `goana` report no adjusted p-values of the sets.
- **Differences from fgsea.** CytePool adjusts the p-values of the sets that have one.
  When no set goes to the multilevel method, fgsea adjusts them as though the sets with
  too few random scores of their sign had been tested too, which changes the adjusted
  p-values of the sets it reports. A set tested alone draws its random sets with fgsea's
  own sampler, as several sets do, where fgsea uses R's `sample.int`; random samples of
  half of the genes or more are shuffled by the Fisher-Yates algorithm
  [@durstenfeld1964], where fgsea calls C++'s `std::shuffle`, an algorithm that each C++
  library chooses for itself. The results agree within Monte Carlo error. Genes without
  an id or a statistic are left out, where fgsea refuses them, and the table lists the
  sets in the order given, where fgsea sorts them by name.

(over-representation-and-gsea-checks)=
## Checks and further reading

- Over-representation analysis, with its dot plot, is compared with limma 3.62.1's
  `kegga` and R 4.4.3's `phyper`: exact (1e-10)
  ([Validation](#validation-gene-set-files-and-ora)).
- Gene set enrichment analysis and its figures are compared with fgsea 1.32.2: exact
  (1e-10) given fgsea's seeds, and calibrated, within 4 standard errors over 20 runs, when
  each program draws its own random sets ([Validation](#validation-gsea)).

Further reading:

- @subramanian2005 described gene set enrichment analysis with its weighted score, the
  one that `gsea` computes by default; @mootha2003 had introduced it with the unweighted
  score.
- @korotkevich2021 introduced fgsea, whose multilevel method estimates p-values far
  smaller than permutations can.
- @timmons2015 and @wijesooriya2022 showed that over-representation analyses depend on
  the universe of genes they test against: a universe of every gene of the genome,
  rather than of the genes measured, makes sets look significant that are not.
- @young2010 showed that RNA-seq calls long genes significant more often, and adjusted
  the test of gene categories for it.

[Pathway scores per sample](pathway_scores.md), the next chapter, gives every sample a
score of each gene set.
