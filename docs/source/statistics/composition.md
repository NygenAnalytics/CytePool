---
description: How propeller tests changes in cell-type proportions from the cell numbers of a cube, with transformed proportions, a linear model and empirical Bayes moderation across cell types.
---

# Composition tests

A condition can change which cells are present as well as what they express. A cube
records how many cells of each type every sample holds ({api}`Cube.n_cells`), and
{api}`propeller` tests whether the share of each cell type differs between conditions,
as speckle's `propeller` does [@phipson2022]. Proportions estimated from single-cell
data are variable, so the test compares samples, the biological replicates, not cells.
[Cell-type composition](../guides/composition.ipynb) runs it on a simulated study.

## The steps

1. **Proportions.** Sample $j$, with $x_{ij}$ cells of type $i$ and $N_j$ cells in all,
   has the proportion $p_{ij} = x_{ij} / N_j$ of type $i$.
2. **Transform.** The logit, $\log(q_{ij} / (1 - q_{ij}))$, of the proportions after
   half a cell is added to every count,
   $q_{ij} = (x_{ij} + 0.5) / \sum_i (x_{ij} + 0.5)$, so that a type absent from a
   sample has a finite value; or the arcsine square root, $\arcsin\sqrt{p_{ij}}$.
3. **Linear model.** The transformed proportions of every cell type are fitted with the
   design ([Designs, linear models and contrasts](linear_models.md)).
4. **Moderation.** The residual variances of the cell types are moderated by empirical
   Bayes, robustly by default, as speckle does ([Empirical Bayes
   moderation](empirical_bayes.md); @phipson2016).
5. **Tests.** Each contrast is tested by its moderated t-statistic; `f_tests` names a
   factor, or several, whose levels are tested together by a moderated F-statistic.
   p-values are adjusted by the method of Benjamini and Hochberg over the cell types.

The result, a {api}`CompositionResult`, holds a table for every contrast and F-test
({api}`CompositionResult.table`), with speckle's descriptive columns: the baseline
proportion of every cell type, the mean proportion in each level of the factor a
contrast compares, and their ratio. Its summary says which cell types and samples were
left out, and why.

## Choices

- **The denominator.** Proportions are taken over the cell types of the data. To test
  the composition within one lineage, such as the T cell subtypes, select its cell types
  first ({api}`Cube.select`); then a shift in another lineage does not move them.
- **The transform.** The logit, speckle's default, or the arcsine square root
  (`transform="asin"`).
- **Samples and cell types without cells.** A sample without cells has no proportions
  and is left out; a cell type without cells in any sample used is left out, as it was
  not observed.

## Limits

propeller fits fixed effects only: a design with random effects is refused, so a donor
sampled in every condition enters the formula as a fixed effect, as in
`~ donor + condition`. Precision-weighted mixed models of composition (crumblr;
@hoffman2026b) and Bayesian count models, such as scCODA [@buttner2021] and sccomp
[@mangiola2023], are not implemented.
