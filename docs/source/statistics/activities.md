---
description: How active transcription factors and pathways are in each sample or contrast, read from the signed targets of a network such as CollecTRI or PROGENy.
---

# Activities from networks

[Pathway scores per sample](pathway_scores.md) take the genes of a set without signs. A transcription factor, though, raises some of its targets and lowers others, and a signalling pathway leaves a footprint of genes that respond to it, some up and some down. A network gives each such source signed weights on its targets, and its activity is read from them: high where the genes it raises are high and those it lowers are low. {api}`ulm` scores every source of a network in every sample or contrast by default, as decoupler computes it [@badiaimompel2022], and {api}`mlm` and {api}`weighted_aggregate` compute two more of decoupler's methods. [Pathway and regulator activity](../guides/pathway_activity.ipynb) scores the regulators and pathways of published data.

(regulator-networks)=
## Networks of regulators and pathways

A {api}`Network` gives each source, a transcription factor or a pathway, its target genes, each with a weight: positive where the source raises the gene and negative where it lowers it. Two networks are built in:

- {api}`collectri`, CollecTRI's signed regulons of human transcription factors, gathered from twelve resources of curated literature and text mining [@muellerdott2023]: a weight of 1 for a target that a factor activates and -1 for one that it represses.
- {api}`progeny`, PROGENy's footprints of 14 signalling pathways: the z-scores with which genes responded to the perturbation of each pathway in hundreds of human experiments [@schubert2018], transferred to mouse [@holland2020b], from the R package progeny.

Both download their files on first use from where they were published, check them against pinned SHA-256 digests and keep them in a cache, so that a machine without a connection can use a file copied there. Each resource of CollecTRI keeps its own terms, which OmniPath records [@tuerei2021]. The default level, `licence="commercial"`, keeps the interactions that a resource listed for commercial use supports, and `licence="academic"` adds those that only a resource listed for academic use supports ([Datasets](../reference/datasets.md#networks) gives the terms and the counts). Any network given as a table of `source`, `target` and `weight`, or as gene sets, whose genes weigh 1, works the same way, and {api}`Network.select` keeps some of its sources.

(activity-profiles)=
## Profiles and targets

A profile is one vector of values over genes: a sample's expression, or a contrast's statistics. The activity functions score

- a samples x genes matrix, one profile per sample;
- a cell type of a cube, one profile per sample, read as the pathway scores read it ([The values scored](#pathway-score-values));
- a differential-expression result, one profile per cell type and contrast, by default of its moderated t-statistics, the statistics that decoupler's guides score; or one profile from a table of results, or from gene ids mapped to statistics.

Each profile uses the genes that have a value in it, NaN marking a missing one, so that a profile of a result keeps the genes its cell type tested. In sparse data, genes that are zero in every profile were not detected, and they are left out, as in decoupler. A source is scored in a profile when at least `min_targets` of its targets, matched by their ids, are among the profile's genes. The result, an {api}`Activities`, holds the scores of the profiles by sources, with their estimates (slopes, coefficients, or weighted means or sums), their p-values and the p-values adjusted by Benjamini and Hochberg's method over the sources of each profile [@benjamini1995]; {api}`Activities.table` lists them one per row. {api}`plot_activities` draws the scores as a heatmap, and {api}`plot_top_activities` draws a bar for each of the sources that score highest in absolute value in one profile, as decoupler's `pl.barplot` does.

(ulm)=
## ULM

The univariate linear model fits, for each source, a line through the values $y_g$ of the profile's $n$ genes against the source's weights $x_g$, which are zero at the genes outside its targets,

$$
y_g = \beta_0 + \beta_1 x_g + \varepsilon_g,
$$

and scores the source by the slope's t-statistic, $t = \hat\beta_1 / \mathrm{se}(\hat\beta_1) = r \sqrt{(n - 2) / (1 - r^2)}$ for the correlation $r$ between $y$ and $x$, tested on $n - 2$ degrees of freedom. Each source is tested against all the other genes, those of other sources included.

(mlm)=
## MLM

The multivariate linear model fits all the sources of a profile at once, with an intercept,

$$
y_g = \beta_0 + \sum_s \beta_s x_{gs} + \varepsilon_g,
$$

so that each score, the t-statistic of $\beta_s$ with $n - m - 1$ residual degrees of freedom for $m$ sources, measures a source beyond what the others explain: where two regulators share targets, ULM credits both with their response and MLM splits it between them. MLM needs at least two more genes than sources, and sources whose weights are not linear combinations of the weights of others; it refuses those that it cannot tell apart, by name.

(weighted-aggregate)=
## The weighted aggregate

The weighted aggregate scores a source by the weighted mean of its targets' values, $\sum_g w_g y_g / \sum_g |w_g|$; without the denominator, it is their weighted sum, PROGENy's original score [@schubert2018]. With random permutations of the genes, the score becomes $(\mathrm{ES} - \text{mean}) / \text{sd}$ over the permutations' scores, and its two-sided p-value is twice the smaller one-sided one, $(1 + b)/(1 + n)$ for the $b$ of $n$ permutations that score at least, or at most, as high [@phipson2010].

(activities-or-scores)=
## Activities or scores

Activities use the signs and sizes of a network's weights, so that a regulator's repressed targets count against it, where [pathway scores](pathway_scores.md) take the genes of a set without signs; {api}`Network.to_gene_sets` gives each source's targets as a set, without their signs, for the pathway scores and the gene-set tests. Scored on the statistics of a contrast, activities say which regulators drive a response, in each cell type. Scored on the samples, they can be compared, correlated with other measurements and tested with a design, as pathway scores are: {api}`limma` takes the activities of samples with the cube's design ([Testing scores with a design](#testing-scores)). The activities of contrasts carry their own p-values, from each source's test across genes, and {api}`limma` refuses them.

(activities-defaults)=
## Defaults and choices

- **The method.** {api}`ulm` by default, decoupler's default for regulators and pathways alike; simple linear models were among the best at predicting perturbed regulators in decoupler's benchmark [@badiaimompel2022]. {api}`mlm` separates sources whose targets overlap ([MLM](#mlm)). {api}`weighted_aggregate` gives the weighted mean or sum of the targets, with permutation p-values.
- **A contrast or the samples.** Score a result's statistics, the moderated t-statistic by default (`statistic_column="t"`), to see which regulators drive a response; score a cube's cell type to compare its samples or to test them with a design.
- **The networks.** `collectri()` keeps the interactions for commercial use, and `collectri(licence="academic")` all of them. `progeny()` keeps the genes of each pathway whose p-value is below 0.05, as decoupler does (`max_p_value`), and `progeny(top=100)` the 100 with the smallest p-values, as progeny's `getModel` does by default; `organism="mouse"` gives the mouse model.
- **Targets.** `min_targets=5`, decoupler's `tmin`.
- **Permutations.** The weighted aggregate draws 1,000 permutations from `rng`, as many as decoupler; `n_permutations=0` gives the weighted means or sums alone, and `statistic="sum"` the sums.

(activities-limits)=
## Limits and differences

- **MLM's memory.** For each set of profiles with the same genes, MLM holds a matrix of genes by sources and its decomposition, so its memory grows with the number of genes times the number of sources ({api}`mlm` gives the size).
- **PROGENy's gene names.** PROGENy names genes by the symbols of progeny 1.28.0, from Bioconductor 3.20, so data named by current symbols miss the genes renamed since; decoupler takes PROGENy from OmniPath, which gives some of them their current symbols.
- **No VIPER.** decoupler's VIPER is not offered ([Not implemented yet](#not-implemented-yet) says why).
- **MLM's adjusted p-values.** They are Benjamini and Hochberg's, as for the other methods; decoupler leaves MLM's p-values unadjusted and reports them as adjusted.
- **Precision.** p-values are computed in double precision, from the upper tail. decoupler stores its adjusted p-values as single-precision numbers, and takes MLM's p-values as one minus the distribution function, so that small ones lose their digits and the smallest become 0.
- **The weighted aggregate's p-values.** CytePool doubles the smaller one-sided permutation p-value. decoupler takes the share of permutations whose absolute score exceeds the observed one, a two-sided p-value already, and folds it again as if it were one-sided, so that a score near zero, which most permutations exceed, gets a p-value near $2/n$ for $n$ permutations. When every permutation scores alike, the normalized score is NaN, where decoupler's is infinite, or 0 for a score of 0.
- **ULM's t-statistic.** It is exact; decoupler adds 2.2e-16 to $1 - r$ and $1 + r$, which changes the score only where $|r|$ is within about $10^{-6}$ of 1.
- **Missing values and empty profiles.** Each profile uses the genes observed in it, and a profile whose values are all zero has no scores; decoupler refuses missing values and leaves such a profile out. Where MLM cannot tell sources apart, CytePool names them, where decoupler gives them NaN t-statistics.
- **CollecTRI's file.** The 185 interactions without references are kept, since the file holds them and the paper counts them; decoupler's loader drops them.

(activities-checks)=
## Checks and further reading

The activities, their bar plot and the networks are compared with decoupler 2.2.0 and progeny 1.28.0: exact (1e-10), the normalized scores given decoupler's permutations, and within 1.2e-7 for ULM's adjusted p-values, which decoupler keeps in single precision. The networks hold every interaction of decoupler's CollecTRI, and the 185 that its loader drops; CollecTRI's licence levels follow OmniPath's records, and PROGENy's weights are progeny's ([Validation](#validation-activity)). The tests of activities with a design are those of pathway scores ([Checks and further reading](#pathway-scores-checks)).

Further reading:

- @holland2020 found that tools for pathway and transcription-factor activity built for bulk data, with curated gene sets, work on single-cell data, and that their results depend more on the gene sets than on the statistic.
- @badiaimompel2022 collected activity methods in one framework, decoupleR, and benchmarked them on perturbation experiments.
- @muellerdott2023 assembled CollecTRI, signed regulons of human transcription factors that expand the coverage of earlier collections.
- @schubert2018 introduced PROGENy, which infers the activity of signalling pathways from the genes that respond to their perturbation rather than from the pathways' members.

[Aligning studies](aligning_studies.md) begins the last part of the book, on several studies.
