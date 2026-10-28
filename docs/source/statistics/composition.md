---
description: Whether a condition changes the share of each cell type in the samples, tested by propeller on the cell numbers of a cube.
---

# Composition tests

[Mixed models and dream](mixed_models.md) closed the chapters on which genes change in
each cell type. A condition can also change which cells a sample holds, when a cell type
expands or contracts. A cube records how many cells of each type every sample holds
({api}`Cube.n_cells`), and {api}`propeller` tests whether the share of each cell type
changes with the condition, by speckle's method [@phipson2022]. Proportions
estimated from single-cell data are variable, so the test compares samples, the
biological replicates, as the tests of genes do. [Cell-type
composition](../guides/composition.ipynb) runs it on a simulated study whose answer is
known.

(propeller-steps)=
## The steps

1. **Proportions.** Sample $j$, with $x_{ij}$ cells of type $i$ and $N_j$ cells in all,
   has the proportion $p_{ij} = x_{ij} / N_j$ of type $i$.
2. **Transform.** The logit, $\log(q_{ij} / (1 - q_{ij}))$, of the proportions after
   half a cell is added to every count,
   $q_{ij} = (x_{ij} + 0.5) / \sum_i (x_{ij} + 0.5)$, so that a cell type absent from a
   sample has a finite value; or the arcsine square root, $\arcsin\sqrt{p_{ij}}$.
3. **Linear model.** The transformed proportions of every cell type are fitted with the
   design, matched to the samples by `sample_id` ([Designs, linear models and
   contrasts](linear_models.md)).
4. **Moderation.** The residual variances of the cell types are moderated by empirical
   Bayes, robustly by default ([below](#composition-moderation)).
5. **Tests.** Each contrast is tested by its moderated t-statistic, as speckle's
   `propeller.ttest` does. `f_tests` names a factor, or several, each tested on its own:
   a moderated F-statistic asks whether all its levels share one mean, as
   `propeller.anova` does. The p-values of each test are adjusted by the method of
   Benjamini and Hochberg over the cell types.

The data are a cube that records its cell numbers, or a table of the cell numbers of
every sample and cell type, such as a cross-tabulation of the labels of the cells.

(composition-moderation)=
## Moderation across cell types

A study has a few cell types to borrow strength from, where a test of genes has
thousands, so step 4 rests on few variances. propeller estimates their prior robustly
by default, as speckle calls limma's `eBayes(robust = TRUE)`: with 10 cell types or
more, a cell type whose transformed proportions vary much more than the others' gets
smaller prior degrees of freedom of its own, and does not lower those of the others
[@phipson2016]. Fewer cell types are too few for the robust estimate, which is then the
ordinary one, and two cell types are not moderated at all, as in limma ([Estimating
the prior](#estimating-the-prior)). With `trend=True`, the prior variance is a trend in
each cell type's mean transformed proportion.

(composition-result)=
## The result

The result, a {api}`CompositionResult`, holds a table for every contrast and F-test
({api}`CompositionResult.table`), with speckle's descriptive columns: each cell type's
baseline proportion over the samples used and, for a contrast between the levels of a
factor, the cell type's mean proportion at each level the contrast weighs and their
ratio, such as $m_\text{stim} / m_\text{ctrl}$ for stimulated against control. A
contrast compares levels when its weights, summing to zero, fall on the levels of one
factor of the formula ({api}`CompositionResult` gives the rule): `conditionstim` in
`~ donor + condition` compares stimulated with control, and
`(groupA + groupB)/2 - groupC` in `~ 0 + group` three groups, but `groupA + groupB`
compares none, and neither does the slope of a covariate, so their tables have no such
columns. The summary says which cell types were left out, and why, the samples table
which samples were, and the proportions table gives each proportion with its
transformed value.

(composition-defaults)=
## Defaults and choices

- **The denominator.** Proportions are taken over the cell types of the data, so a
  change in one cell type moves the shares of all the others. To test the composition
  within one lineage, such as the T cell subtypes, select its cell types first
  ({api}`Cube.select`); then a shift in another lineage does not move them.
- **The transform.** The logit, speckle's default, or the arcsine square root
  (`transform="asin"`).
- **Robust moderation.** On by default, as in speckle; it changes the prior only with
  10 cell types or more ([above](#composition-moderation)), and `robust=False` turns it
  off.
- **The design.** Pair the samples of a donor when each donor gives one in every
  condition, as in `~ donor + condition`. speckle's test of two groups is
  `~ 0 + group` with the contrast of the first level against the second, its test of
  more groups `f_tests="group"`, and its tests with a covariate those of
  `~ 0 + group + age`.
- **Samples and cell types without cells.** A sample without cells has no proportions,
  and a cell type that has no cells in the samples used was not observed; both are left
  out.

(composition-limits)=
## Limits and differences

- **Fixed effects only.** A design with random effects is refused, and there is no
  `duplicateCorrelation`: a donor sampled in every condition enters the formula as a
  fixed effect. Precision-weighted mixed models of composition (crumblr; @hoffman2026b)
  and Bayesian count models, such as scCODA [@buttner2021] and sccomp [@mangiola2023],
  are not implemented ([Not implemented yet](choosing_a_method.md#not-implemented-yet)).
- **Cell numbers.** propeller reads whole numbers of cells; it does not take
  proportions with assumed totals, which speckle's `convertDataToList` accepts.
- **F-tests of all the levels.** An F-test compares all the levels of its factor; to
  compare some of them, test their contrasts. The t-tests of contrasts take limma's
  degrees of freedom, which cannot exceed the total residual degrees of freedom of the
  cell types, and the F-tests leave theirs uncapped, as limma does ([Moderated
  F-statistics](empirical_bayes.md#moderated-f-statistics)).
- **Unused levels.** Given cell labels as a factor with a level that no cell has,
  speckle keeps that cell type and tests it, where CytePool leaves it out; with the logit
  transform, the half cell added to it shifts the transformed proportions of all the
  other cell types. A level of the sample labels without cells is kept too: with the logit
  transform each cell type of that sample gets the proportion one over the number of
  cell types, a value that enters the fit as data, and with the arcsine transform limma
  leaves the sample out, as CytePool does with either.
- **Descriptive columns from the design's coding.** speckle regresses the proportions on
  the columns of the design that a contrast weighs and reports the coefficients as mean
  proportions. They are means only where those columns indicate groups of samples: for
  `(Intercept) + groupB` in `~ group + age` they are the mean of the samples outside B
  and a difference of means, and for `groupA + groupB`, which compares nothing, the
  ratio is a product of two means. CytePool reads the levels that a contrast compares
  from the design's coding ([The result](#composition-result)), so that any
  parametrization of one comparison gives one table, and it tests the contrasts of a
  single column that speckle refuses, such as the slope of a covariate.
- **Data that the design fits exactly.** When the samples of each condition are copies
  of one another, say, no variance can be estimated, and CytePool raises an error where
  speckle returns statistics of the rounding errors, or NaN.

(composition-checks)=
## Checks and further reading

{api}`propeller` is compared with the three functions of speckle 1.6.0 (`propeller`,
`propeller.ttest`, `propeller.anova`) and limma 3.62.1, run as a separate program on the
same cell numbers: exact (1e-10), and close (1e-6) for the robust prior of 10 cell types
or more, which limma solves only to a tolerance
([Validation](#validation-composition-tests)).

Further reading:

- @phipson2022 introduced propeller, which tests transformed cell-type proportions with
  linear models and empirical Bayes moderation across replicates.
- @buttner2021 introduced scCODA, a Bayesian model of compositional single-cell data.
- @mangiola2023 introduced sccomp, which models composition and its variability
  together, and found that the variability of proportions depends on their mean in every
  technology examined.
- @hoffman2026b introduced crumblr, precision-weighted mixed models of cell-type
  composition that test several levels of the cell lineage at once.
- @dann2022 introduced Milo, which tests differential abundance in neighbourhoods of a
  k-nearest neighbour graph instead of clusters.
- @yi2024 benchmarked six differential abundance methods for condition-specific cell
  states, and gave recommendations by dataset.
- @stephenson2021 profiled over 780,000 blood cells of 130 COVID-19 patients and related
  the expansion of cell populations to disease severity.

[Gene-set tests](gene_set_tests.md) begins the part that turns the results for genes
into results for pathways and regulators.
