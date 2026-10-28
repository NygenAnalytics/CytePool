---
description: How active a pathway is in each sample, as one score per sample from GSVA, ssGSEA, combined z-scores or PLAGE, and how to test those scores with a design.
---

# Pathway scores per sample

The tests of [Gene-set tests](gene_set_tests.md) and [Over-representation and
GSEA](over_representation_and_gsea.md) ask one question of one contrast. A pathway score
instead turns the expression of a set's genes in each sample into one number, so that
pathways can be compared across samples, drawn as a heatmap, correlated with other
measurements and tested with any design [@haenzelmann2013]. By default {api}`gsva`
computes the scores by gene set variation analysis, and {api}`ssgsea`,
{api}`combined_z_scores` and {api}`plage` compute the other three methods of the GSVA
package. [Pathway and regulator activity](../guides/pathway_activity.ipynb) scores
published data with GSVA and tests the scores.

(pathway-score-values)=
## The values scored

Each method scores a samples x genes matrix of expression values, or a cell type of a
cube. A cell type gives its samples with enough cells (`min_cells`, 10 by default, as for
{api}`limma_voom`; [Samples with few cells or counts](#samples-with-few-cells)) and its
genes that pass edgeR's filter ([Filtering by expression](#filtering-by-expression)), as
log-CPM values ([log-CPM](#log-cpm)) or, with `values="counts"`, as counts: the values
that GSVA's guide scores. NaN marks a missing value.

Sets with fewer than `min_size` or more than `max_size` genes among those scored are left
out. The result, a {api}`PathwayScores`, holds the scores of the samples by sets, with the
table of the samples and the genes that each set used, and {api}`PathwayScores.table`
lists the scores one per row; {api}`plot_pathway_scores` draws them as a heatmap, with
the samples grouped by a variable.

(gsva)=
## GSVA

GSVA first puts each gene on a common scale. A kernel estimate of the gene's
distribution over the $n$ samples gives each value the probability that the gene's value
is no higher,

$$
F_{ij} = \frac{1}{n} \sum_{k=1}^{n} \Phi\!\left(\frac{x_{ij} - x_{ik}}{h_i}\right)
$$

for gene $i$ in sample $j$, with a bandwidth $h_i$ of one quarter of gene $i$'s standard
deviation, for continuous values such as log-CPM. For counts the kernel is the Poisson
probability $P(Y \le x_{ij})$ with mean $x_{ik} + 1/2$, and without a kernel $F_{ij}$ is
the share of the samples whose value is at most $x_{ij}$. Each sample ranks its $p$ genes
by $F_{ij}$ and gives the gene of rank $r_{ij}$ the weight $|p/2 - r_{ij}|$, so that the
genes at both ends of the ranking weigh most. A walk down the ranking rises at the genes
of a set and falls at the others, as in [gene set enrichment analysis](#gsea), and the
score is the walk's largest deviation above zero minus its largest deviation below: high
for a set whose genes are higher than usual in the sample, low for one whose genes are
lower, and between $-1$ and $1$.

(ssgsea)=
## ssGSEA

Single-sample GSEA ranks the genes of each sample by their values and sums the same walk
over all genes rather than taking its extreme, with the genes of the set weighted by
their ranks to the power $\alpha = 1/4$ [@barbie2009]. By default the scores are then
scaled by their range over every set and sample, as @barbie2009 scaled them, so that a
score depends on the other sets and samples scored with it.

(combined-z-scores)=
## Combined z-scores

Each gene is standardized across the samples, and a set scores the sum of its $k$ genes'
z-scores over $\sqrt{k}$: a standard normal score if its genes are independent and
standard normal [@lee2008].

(plage)=
## PLAGE

PLAGE standardizes the genes of a set and scores the samples by the first right singular
vector of their genes x samples matrix: the samples' coordinates on the direction along
which the set's genes vary most together [@tomfohr2005]. Across the samples, a set's
scores have mean zero and length one. The vector could as well point the other way;
CytePool points it where the scores rise with the set's mean standardized expression.

(how-pathway-scores-differ)=
## How the scores differ

ssGSEA ranks the values themselves, so that a set of highly expressed genes scores high
in every sample. GSVA first sets each gene against its own distribution across the
samples, so that a set scores high where its genes are higher than usual, compared with
the other genes of the sample. Combined z-scores and PLAGE also set each gene against
itself across the samples, but do not compare it with other genes: combined z-scores add
the genes of a set alike, and PLAGE follows the pattern along which they vary most
together. GSVA and ssGSEA thus ask how a set ranks within a sample, and combined z-scores
and PLAGE how a sample stands among the others.

(testing-scores)=
## Testing scores with a design

The scores of a set differ between conditions as the expression of a gene does, so they
are tested as genes are, with limma's linear models, contrasts, moderated t-tests and
adjusted p-values, as GSVA's guide does [@haenzelmann2013; @smyth2004]. {api}`limma`
takes these steps in one call, with the {api}`Design` built for the cube. It fits the
samples that have scores, so a sample that the scores leave out, such as one with too few
cells, is not fitted either, and a design that the remaining samples no longer determine
is fitted as {api}`limma_voom` fits a cell type ([Designs that cannot answer the
question](#estimable-contrasts)). GSVA's guide notes that the scores of larger sets vary
less, and lets the prior variance of the moderation follow the set sizes:
`limma(scores, design, trend=scores.n_genes)` does the same. The activities of the [next
chapter](activities.md) are tested alike.

(pathway-scores-defaults)=
## Defaults and choices

- **The method.** {api}`gsva` by default, which compares a set's genes with the other
  genes of a sample after setting each gene against its own distribution; {api}`ssgsea`
  compares the values themselves; {api}`combined_z_scores` and {api}`plage` follow each
  set across the samples without comparing it with other genes ([How the scores
  differ](#how-pathway-scores-differ)).
- **The kernel.** `kernel="auto"`, GSVA's default, chooses the Poisson kernel when the
  values are stored as integers, such as counts, the Gaussian kernel for other values,
  and no kernel from 200 samples on (`min_samples_without_kernel`); `"gaussian"`,
  `"poisson"` and `"none"` choose one.
- **The score of GSVA.** `score_type="difference"`, GSVA's default, subtracts the walk's
  largest deviation below zero from its largest above, so that a set scores high where
  its genes are high and low where they are low; `"kuiper"` adds the two, to score a set
  whose genes sit at both ends of the ranking, and `"max_deviation"` takes the larger of
  them, with its sign.
- **The weights of the ranks.** `exponent` is GSVA's tau, 1, and ssGSEA's alpha, 1/4.
- **ssGSEA's normalization.** `normalize=True` scales the scores by their range;
  `normalize=False` keeps each score independent of the others.
- **The values of a cube.** log-CPM values by default, and `values="counts"` for GSVA's
  Poisson kernel; `min_cells` and `expression_filter` choose the samples and genes ([The
  values scored](#pathway-score-values)).
- **Set sizes.** Every size by default, as in GSVA; `min_size` and `max_size` limit them.
- **Tests.** {api}`limma` with the cube's design, `trend=scores.n_genes` to let the prior
  follow the set sizes, and `robust=True` for a robust prior.

(pathway-scores-limits)=
## Limits and differences

- **The cost of GSVA.** GSVA estimates each gene's distribution from every pair of its
  samples, so its time grows with the square of the number of samples, and the Poisson
  kernel costs more than the Gaussian ([Timings](#validation-timings)).
- **PLAGE's sign.** CytePool's scores rise with the set's mean standardized expression.
  GSVA keeps the sign that its singular value decomposition happens to return, which
  varies with the linear algebra library, so the two agree up to sign. For a set of one
  gene, GSVA scores 1 in every sample and CytePool the gene's standardized values, over
  their length.
- **Missing values.** {api}`gsva` works on the observed values, as GSVA does with
  `use = "na.rm"`, but a sample's walk falls only at its observed genes outside the set
  and so ends at zero, where GSVA's also counts the missing genes and ends above zero.
  GSVA's default, `use = "everything"`, gives scores without meaning, and GSVA stops on a
  sample without any value, which here gets no scores. ssGSEA and combined z-scores
  treat missing values as GSVA does, and PLAGE refuses them, as GSVA does; each
  function's documentation says how.
- **Inputs on which GSVA stops.** In ssGSEA, a set of every gene gets no score, and the
  other scores are normalized as usual. Values stored as integers, from 200 samples on,
  are scored without a kernel by `kernel="auto"`, as GSVA's documentation says; GSVA
  2.0.0 stops on both. The Poisson kernel refuses values that are not counts, where GSVA
  stops on negative values and scores fractional ones.
- **Ranks decided by the last bit.** Genes whose kernel estimates are equal in exact
  arithmetic, such as genes with the same low counts in different samples, are ordered
  by the last bit of the normal and Poisson probabilities, which CytePool computes
  accurately but not always to the same bit as R. Such genes can take each other's
  ranks, and the scores of their sets then differ slightly from GSVA's.
- **Exact standard errors.** CytePool's {api}`limma` computes the standard errors of
  contrasts exactly; they differ from those of limma's `contrasts.fit` only with missing
  scores, or weights that differ between scores ([Contrasts](#contrasts)).

(pathway-scores-checks)=
## Checks and further reading

The four methods and their heatmap are compared with GSVA 2.0.0, run as a separate
program on the same data: exact (1e-10), PLAGE up to sign
([Validation](#validation-pathway-scores)). {api}`limma` has no section of its own:
`tests/test_score_tests.py` checks that its results are those of
{api}`fit_linear_model`, {api}`contrasts_fit` and {api}`moderated_statistics`, which are
compared with limma 3.62.1 ([Validation](#validation-linear-models) and
[Validation](#validation-empirical-bayes)).

Further reading:

- @haenzelmann2013 introduced GSVA, which scores gene sets in every sample of microarray
  and RNA-seq data.
- @barbie2009 introduced single-sample GSEA, in a screen of the genes that cancers with
  KRAS mutations depend on.
- @lee2008 scored a pathway's activity in each sample by the combined z-scores of its
  genes, and chose for each pathway the genes that best tell the conditions apart.
- @tomfohr2005 scored pathways in each sample by the singular value decomposition of
  their genes' expression.

[Activities from networks](activities.md), the next chapter, scores regulators and
pathways rather than gene sets.
