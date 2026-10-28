---
description: Which changes replicate across studies, by SumRank, prediction in held-out studies, combined p-values, partial conjunctions and the agreement between pairs of studies.
---

# Replication

[Meta-analysis](meta_analysis.md) combined the studies' estimates into one per gene.
Whether a change replicates is another question: that the
gene changes in study after study, not that the studies together find it. {api}`sumrank`
asks it by default: each study ranks the genes, and every gene's ranks are summed across
the studies [@nakatsuka2025]. [Replication across
studies](../guides/replication_across_studies.ipynb) runs it on six simulated cohorts, with
the other methods of this chapter.

(sum-of-ranks)=
## SumRank

A meta-analysis weighs each study by its precision, so that one or two precise studies
can carry a gene; the random-effects model widens the interval when the studies disagree
([Defaults and choices](meta_analysis.md#meta-analysis-defaults)). SumRank asks instead
that a gene rank high in every study, or, with fewer ranks summed, in most, so that one
study alone cannot carry a gene, however small its p-value there. In the paper that
introduced it, its genes predicted case status in studies left out better than those of a
random-effects meta-analysis [@nakatsuka2025]. {api}`sumrank` computes it for one cell
type and contrast of the studies' results, a {api}`StudyResults`, and returns a
{api}`SumRankResult`, whose table ({api}`SumRankResult.features`) gives each gene's
statistics and p-values, up and down; {api}`fit_sumrank` computes it from arrays of the
studies' signed statistics, into a {api}`SumRankFit`.

Within each study, the genes that every study tested are ranked by their signed
$-\log_{10}$ p-value, the gene most up first, $r = 1$, and the gene most down last,
$r = G$; tied genes share their average rank. The normalized ranks
$u = (r - 1) / (G - 1)$ run from 0 to 1. A gene's statistic adds up its best $m$ of its
$k$ normalized ranks, one from each of the $m$ studies that rank it highest, and is capped
at their mean under the null:

$$
S_g = \min\Big(\sum_{j=1}^{m} u_{g(j)},\ \frac{m}{2}\Big), \qquad
m = \lfloor p\,k \rfloor,
$$

with $u_{g(1)} \le \dots \le u_{g(k)}$ and the proportion $p$ of the studies
(`proportion_of_studies`), or $m$ given itself (`n_studies_summed`). By default every study
is summed, $m = k$. The paper summed every study when there were few, 3 to 6, and chose $p$
by cross-validation when there were many: 0.65 of 17 studies of Alzheimer's disease and
0.55 of 16 of COVID-19. Genes down have the statistic of the ranks $1 - u$. A gene missing
from some study's results, because the study did not measure it or its expression filter
left it out, is not ranked, as in the paper, which ranked the genes that every study
measured: a rank says where a gene stands among the genes ranked with it.

If each study ranked the genes at random, the normalized ranks would be independent
uniform variables, and $S_g$ below the cap would follow the Irwin-Hall distribution of a
sum of $m$ of them [@irwin1927; @hall1927], whose distribution function gives the
Irwin-Hall p-value. With every study summed this distribution holds, but for the
discreteness of the ranks, which is small among many genes. The sum of the best $m$ of $k$
ranks is smaller than a sum of $m$ ranks, so with $m < k$ these p-values are much too
small, and {api}`sumrank` warns when it gives them: in six simulated studies without
changes, 4.8% of the p-values fell at or below 0.05 with every study summed, and 37% with
the best three of six ([Validation](#validation-sumrank)).

The paper takes its p-values from permutations instead. With `n_permutations`,
{api}`sumrank` permutes the variable whose levels the contrast compares (or `permute`) in
each study's design, among the samples of its fit of the cell type, from a seeded
`numpy.random.Generator`, and runs the study's {api}`limma_voom` or {api}`limma_trend`
again, with the design and the settings that its result records. Pooled over the genes and
the permutations, the permuted statistics make one null distribution of $N$ values,
against which a gene of statistic $s$ has the p-value

$$
p = \frac{1 + \#\{\text{null statistics} \le s\}}{1 + N},
$$

never zero [@phipson2010]. This pooling gives small p-values from few permutations: 100
permutations of 10,000 genes reach $10^{-6}$. Each permutation runs every study's analysis
once, for one cell type ([Timings](#validation-timings) gives the cost). {api}`plot_sumrank`
draws the p-values as a Manhattan plot, a point per gene in alphabetical order, as
SumRank's authors draw them.

(held-out-prediction)=
## Prediction in held-out studies

Genes that replicate should tell cases from controls in a study that had no part in
choosing them. {api}`held_out_prediction` leaves each study of a {api}`StudyResults` out in
turn and chooses, in each cell type, `n_features` genes up and as many down from the other
studies' results (50 by default), by SumRank and by the random-effects meta-analysis with
REML and the Knapp-Hartung adjustment, the paper's comparison. It scores each of the
held-out study's cases and controls, the two levels that the contrast compares, and reports
the area under the ROC curve of cases against controls [@hanley1982], by study and method,
in a {api}`HeldOutPrediction` ({api}`HeldOutPrediction.auc`).

A sample's score builds on UCell's rank score [@andreatta2021]: in each cell type, how high
the sample's profile ranks the chosen genes among all its genes, rescaled to [0, 1] across
the study's samples, with the genes up adding to the score and the genes down taking from
it; the cell types' scores are summed, as the authors' code sums them. A sample without a
profile in a cell type takes the mean of the others' scaled scores there. The genes chosen
and the scores are in {api}`HeldOutPrediction.features` and
{api}`HeldOutPrediction.scores`, and {api}`plot_held_out_auc` draws the AUC of every study
by method. An AUC of 0.5 says that the score does not separate cases from controls, and
below 0.5 that the genes predict the wrong direction.

(combined-p-values)=
## Combined p-values

Where the studies give p-values rather than comparable effects, they can be combined. For
every gene that $k$ studies tested, with p-values $p_1, \dots, p_k$, {api}`combine_p_values`
tests the hypothesis that none of these studies has an effect, by the methods of scipy's
`combine_pvalues` and the R package metap, and returns a {api}`CombinedPValues`. With
$p_{(1)} \le \dots \le p_{(k)}$ in order:

| Method | Statistic | Its distribution without effects |
| --- | --- | --- |
| `"fisher"` [@fisher1932] | $X = -2 \sum \log p_i$ | $\chi^2_{2k}$, upper tail |
| `"stouffer"` [@stouffer1949; @liptak1958] | $Z = \sum w_i z_i / \sqrt{\sum w_i^2}$, $z_i = \Phi^{-1}(1 - p_i)$ | standard normal, upper tail |
| `"tippett"` [@tippett1931] | $p_{(1)}$ | $1 - (1 - p_{(1)})^k$ |
| `"wilkinson"` [@wilkinson1951] | $p_{(r)}$ | beta with parameters $r$ and $k - r + 1$ |
| `"pearson"` [@pearson1933] | $X = -2 \sum \log (1 - p_i)$ | $\chi^2_{2k}$, lower tail |
| `"mudholkar_george"` [@george1983] | $\sum \log((1 - p_i)/p_i)$, scaled | Student's t on $5k + 4$ degrees of freedom |
| `"simes"` [@simes1986] | $\min_j k\, p_{(j)} / j$ | the p-value itself |

Fisher's and Tippett's methods respond to the strongest study; Stouffer's, the logit and
Pearson's to evidence that the studies share. Stouffer's method weighs the studies equally
unless given `weights`, usually the square roots of their sample sizes [@zaykin2011]. A
small p-value of Wilkinson's maximum ($r = k$) says that every study has some evidence, not
that every study has an effect, which the partial conjunction below tests. A gene that some
studies did not test is combined over the others, and `n_studies` says how many; one study
gives its own p-value, except with the logit, which approximates its statistic's
distribution even then. p-values of exactly 0 or 1 are taken as the limits that the
methods reach.

Each method expects one-sided p-values of one direction of effect, so that an effect the
other way gives a p-value near 1. Two-sided p-values can be combined as they are, but they
then ask whether some study has an effect either way, and opposite effects do not cancel.
Given the signs of the studies' effects in `directions`, such as their log2 fold changes,
{api}`combine_p_values` makes each two-sided p-value one-sided for each direction, $p/2$
where the effect points that way and $1 - p/2$ where it points the other, as metap's
`two2one` does; it combines each direction and doubles the smaller result, up to 1
[@owen2009]; for Stouffer's method, that is the two-sided p-value of $Z$. Each method uses
$p/2$ as it is, never as $1 - (1 - p/2)$, so that p-values near 1e-300 keep their digits.

(partial-conjunction)=
## An effect in at least u studies

One study's effect makes the hypothesis of no effect in any study false, so a small
combined p-value does not say that an effect replicates. {api}`partial_conjunction` tests
the hypothesis that fewer than $u$ of the $n$ studies that tested a gene have an effect
[@benjamini2008]. Up to $u - 1$ studies with effects could explain the $u - 1$ smallest
p-values, so the test sets them aside and combines the others, $p_{(u)}, \dots, p_{(n)}$:
by Simes's method, $\min_{i \ge u} (n - u + 1)\, p_{(i)} / (i - u + 1)$, the default, or by
Fisher's or Stouffer's (unweighted). With $u = 1$ this is the test of no effect in any
study; with $u = n$ every method returns the largest p-value, $p_{(n)}$. With
`directions`, the $u$ effects must point the same way. Benjamini and Hochberg's adjustment
of these p-values over the genes controls the false discovery rate of the claims that a
gene has an effect in at least $u$ studies [@benjamini2008]. It gains this at a cost in
power: the change must be clear in $u$ studies, each on its own.

(pi1-between-studies)=
## How well one study replicates another

A gene that one study finds and another does not may still change in the other, by too
little for it to find. Take the genes that the discovery study finds, and their p-values
in the replication study: one minus Storey's estimate of the proportion of these p-values
without an effect, $\pi_1 = 1 - \pi_0$ ([The proportion of true
nulls](multiple_testing.md#proportion-of-true-nulls)), estimates the share of the genes
that change in the replication study, whether it finds them or not [@storey2003]. It is not
symmetric: it reads the genes of one study in the p-values of the other. The sign
agreement, the share of those genes whose log2 fold changes point the same way in both
studies, says whether the changes agree in direction.

By default $\pi_0$ is estimated at $\lambda = 0.5$, which suits the few dozen p-values of
one study's findings better than qvalue's smoother ([Defaults and
choices](multiple_testing.md#multiple-testing-defaults)); `null_proportion={}` asks for the
smoother.

(replication-tables)=
## The replication tables

{api}`replication` reads the studies' results, a {api}`StudyResults`, for one cell type
and contrast, and returns a {api}`ReplicationResult` with four tables, each of which
answers one question:

- `features`: does each gene replicate? For every gene: how many studies tested it and how
  many find it, with an adjusted p-value below `alpha`, up and down; its combined p-value,
  by Stouffer's method with each study weighted by the square root of its number of samples
  and signed by its log2 fold change; the p-value of the partial conjunction test that at
  least `at_least` studies, two by default, have an effect in one direction, by Simes's
  method; both adjusted over the genes; the combined direction and the share of the
  studies that agree with it; and Cochran's $Q$, $I^2$ and $\tau^2$ of its log2 fold
  changes ([Tests and heterogeneity](#heterogeneity)).
- `study_pairs`: how well does each study replicate each other? For every ordered pair,
  $\pi_1$ and the sign agreement in the second study of the genes that the first finds,
  and how many of them the second finds too.
- `found_in`: how many genes do at least $k$ studies find, for every $k$, and in one
  direction?
- `studies`: each study's number of samples and weight, and how many genes it tested and
  finds.

{api}`plot_study_agreement` shows $\pi_1$ or the sign agreement for every pair of studies
as a heatmap, and {api}`plot_replicated_genes` the number of genes found by each number of
studies.

(replication-defaults)=
## Defaults and choices

- **Which genes replicate.** {api}`sumrank` by default, which asks that a gene rank high in
  study after study and weighs every study alike; {api}`partial_conjunction` to claim a change
  in at least $u$ studies, with a false discovery rate over the genes; a
  [meta-analysis](meta_analysis.md) to estimate each gene's change with its uncertainty.
- **SumRank's ranks and p-values.** Every study summed, with Irwin-Hall p-values, by
  default; fewer summed (`n_studies_summed` or `proportion_of_studies`) needs permutations
  (`n_permutations`). Held-out prediction for a few proportions of studies summed shows which
  predicts best, as the paper chose its proportions.
- **Which combination of p-values.** Fisher's method finds genes that any study changes;
  Stouffer's, weighted by the square roots of the studies' sizes and the default of
  {api}`replication`, rewards evidence that the studies share. Neither says that a change
  replicates, which {api}`partial_conjunction` tests (`at_least`, 2, and
  `conjunction_method`, Simes's by default).
- **$\pi_1$.** $\pi_0$ at $\lambda = 0.5$ by default; `null_proportion` chooses qvalue's
  smoother (`{}`) or its bootstrap (`{"method": "bootstrap"}`) for studies that find many
  genes.
- **Held-out prediction.** `n_features` (50) sets how many genes up and down are chosen in
  each cell type, and `groups` the cell types scored.

(replication-limits)=
## Limits and differences

- **Independent studies.** Combined p-values and partial conjunctions take the studies as
  independent, with p-values uniform without effects; for discrete tests they are
  conservative. {api}`replication` reads limma's two-sided p-values with the signs of the
  log2 fold changes, so it needs a contrast, not an F-test.
- **Not implemented.** The r-values of a primary study and its follow-up [@heller2014],
  metap's methods of Edgington, Lancaster, the inverse t, the mean z and vote counting,
  weights for the studies of SumRank, and permutations of designs with random effects
  ([Not implemented yet](#not-implemented-yet)). The proportion of studies that SumRank
  sums for held-out prediction is given, not chosen: a loop over the proportions chooses
  it, as the guide does.
- **SumRank's p-values.** The Irwin-Hall p-value is the lower tail of the distribution. The
  paper calls its Irwin-Hall p-values two-sided, which would double them, and its authors'
  code takes the Irwin-Hall density at the statistic (`dirwin.hall` of the R package
  unifed), which is not a probability; both order the genes as the statistic does, so the
  permutation p-values do not depend on it. Permutation p-values count ties as at least as
  extreme and add one, so that none is zero, where the authors' code takes the share of
  null statistics strictly below a gene's, 0 for a gene below them all. The authors' code
  also forms each gene's sum in floating point, in the gene's own order, so that equal
  statistics can compare as unequal, where here they are exact; it stops when it sums one
  study; and it takes $\lfloor p\,k \rfloor$ in floating point, 56 for 0.57 of 100
  studies, where CytePool rounds $p\,k$ first and sums 57.
- **SumRank's permutations.** Each study's labels are shuffled among the samples of its own
  fit, which keeps its numbers of cases and controls; the paper permutes them among all the
  individuals of a dataset, for all its cell types at once, and draws again until every
  cell type has at least two cases and two controls.
- **Held-out prediction.** The paper fits a logistic regression of case status on the score
  in the other studies, and takes the AUC of its predictions in the held-out study from
  pROC, which by default compares the medians of the cases and controls and takes the
  direction in which the cases score higher [@robin2011]. A regression on one score is
  monotone in it, so that AUC is the score's own in the direction pROC chooses: a score
  that ranks controls above cases also scores above 0.5. Here the cases must score higher,
  as genes up minus genes down should make them, and no regression is needed. The profiles
  are summed counts, which rank a sample's genes as their mean over its cells does; the
  paper ranked Seurat's average of the cells' normalized expression, which weighs each
  cell's counts by its depth.
- **Combined p-values.** metap leaves out p-values of 0, and of 1 for `sumz` and `logitp`,
  with the warning "some studies omitted", and needs two studies. CytePool takes p-values
  of 0 and 1 as limits, and one study gives its own p-value, except with the logit.

(replication-checks)=
## Checks and further reading

- Combined p-values and partial conjunctions are compared with scipy 1.18.1, metap 1.1 and
  400-digit arithmetic, the proportion of true nulls with qvalue 2.38.0, and the
  replication tables with metap, qvalue and metafor 5.0.1: exact (1e-10; close, 1e-6, for
  REML) ([Validation](#validation-combined-p-values-and-replication)).
- SumRank and held-out prediction are compared with the authors' code (Apache-2.0, commit
  46392f66), with 200-digit arithmetic for the Irwin-Hall distribution, and with scipy
  1.18.1 and scikit-learn 1.9.1 for the scores and AUCs: exact (1e-12), and calibrated for
  p-values from random permutations ([Validation](#validation-sumrank)).

Further reading:

- @nakatsuka2025 introduced SumRank, a meta-analysis of ranks across studies, whose genes
  predicted disease in studies left out better than those of a random-effects
  meta-analysis.
- @benjamini2008 introduced partial conjunction tests, which claim an effect in at least
  $u$ of $n$ studies.
- @phipson2010 showed that p-values from random permutations should never be zero, and
  how to compute them.

This is the last chapter. [Choosing a method](choosing_a_method.md) maps every question to
its method, and the [Validation](../validation.md) page gives how closely each agrees with
the tool that defined it.
