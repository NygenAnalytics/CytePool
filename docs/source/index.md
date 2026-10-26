---
description: CytePool brings the sample-level statistics of limma, edgeR, DESeq2, dream, propeller and metafor to Python for pseudobulk and bulk omics data, with numpy and numba as its only required dependencies.
---

# CytePool

**Sample-level statistics for pseudobulk and bulk omics data, in Python.**

Studies that compare conditions, donors or treatments ask questions about samples:
cells of one donor are not independent replicates, and a test that treats them as
such finds differences that other donors do not share [@squair2021; @zimmerman2021].
CytePool pools the cells of each sample and cell type into a *cube* of pseudobulk
profiles, which can be saved, shared and traced back to the cells, and tests them with
the methods that are standard in R: limma, edgeR, DESeq2, dream, propeller and
metafor. Its only required dependencies are numpy and numba, and every method is
compared with the package that defined it ([Validation](validation.md)).

[Installation](installation.md) · [Quick start](quickstart.ipynb) ·
[Differential expression per cell type](differential_expression.ipynb) ·
[Choosing a method](statistics/choosing_a_method.md) ·
[Capability matrix](reference/capabilities.md) · [API reference](/api/index.html)

:::{note} Early development
Nothing is released yet, and the interface may still change.
:::

## The questions CytePool answers

Studies of many samples, across immune and brain cohorts, population genetics,
pathways, interactions, pharmacology and atlases, come back to ten questions. This is
what CytePool offers for each today.

| Question | What CytePool offers | Where to read |
| --- | --- | --- |
| Which genes change in each cell type? | limma-voom, limma-trend and dream on every cell type of a cube; edgeR's quasi-likelihood tests and DESeq2 on one matrix at a time | [Differential expression per cell type](differential_expression.ipynb), [Bulk RNA-seq](guides/bulk_rna_seq.ipynb), [Choosing a method](statistics/choosing_a_method.md) |
| Which cell types expand or contract? | propeller. Not yet: precision-weighted mixed models or count models of composition | [Composition tests](statistics/composition.md) |
| Which pathways and regulators respond? | camera, fry, roast, mroast and romer on any contrast; gene set enrichment analysis (GSEA) of ranked genes; over-representation analysis of the genes that change; scores of pathways in every sample (GSVA, ssGSEA, combined z-scores, PLAGE); activities of transcription factors and pathways from the networks CollecTRI and PROGENy (decoupler's ULM, MLM and weighted aggregate); gene sets read from GMT files | [Gene-set tests](statistics/gene_set_tests.md) |
| What drives the variation between donors, and what must be adjusted for? | Design checks for confounded factors, sample quality weights, mixed models that estimate a variance per random effect for every gene, and variance partitioning: the share of each gene's variance that each variable explains, in every cell type | [Designs, linear models and contrasts](statistics/linear_models.md), [Mixed models, REML and dream](statistics/mixed_models.md) |
| Does it replicate, and how should studies be combined? | Studies aligned by sample, gene and cell type, each analysed on its own and lined up gene by gene; meta-analysis of every gene across studies, with the between-study variance, Q and I²; combined p-values, the p-value that a change replicates in at least u studies, sign agreement, pi1 between studies and the genes found in k studies; SumRank, which asks that a gene rank high in most studies, and the prediction of cases in held-out studies. Not yet: shrinkage across studies | [Meta-analysis](statistics/meta_analysis.md), [The cube and its additive layers](statistics/cubes.md) |
| Can patients be stratified from their profiles? | Not yet; cubes convert to AnnData and MuData for other tools | [Capability matrix](reference/capabilities.md) |
| Which cell-cell interactions change? | Not yet | [Related software](reference/related_software.md) |
| Do effects depend on genotype, time, sex or context? | Interactions and F-tests in designs, likelihood ratio tests, random intercepts for repeated measures, with F-tests and small-sample (Kenward-Roger) tests in mixed models. Not yet: random slopes. QTL mapping is left to dedicated tools | [Designs, linear models and contrasts](statistics/linear_models.md), [Count models](statistics/count_models.md), [Mixed models, REML and dream](statistics/mixed_models.md) |
| How do cells respond to drugs and doses? | A dose as a factor or a covariate of a design. Not yet: dose-response fits, benchmark doses, synergy | [Capability matrix](reference/capabilities.md) |
| Can public studies be pooled into one cube? | The cubes of several studies stacked into one, of the genes and cell types that every study has, with sample and donor ids kept apart by study; cubes import the pseudobulk layouts of scanpy, decoupler and LIANA | [The cube and its additive layers](statistics/cubes.md), [Export a cube from another tool](guides/export_cubes.ipynb) |

The [Literature](reference/literature.md) page collects studies behind these questions,
and the [Capability matrix](reference/capabilities.md) lists every method with the
reference it is checked against, and what is not implemented yet.

## Principles

- **Samples are the unit of replication.** Cells are pooled into samples before any
  test, and a cube records how many cells each profile holds.
- **Traceable results.** A cube's fingerprint names its values, its provenance record
  says how it was made, and every result keeps both.
- **Checked against the reference implementations.** Each area has a script in
  [`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation)
  that runs the reference package as a separate program on the same inputs.
- **Implemented from the literature.** Methods whose reference packages carry copyleft
  licences are written from their papers and documentation, never from their source
  code, so that CytePool can stay BSD-licensed.
- **Two required dependencies,** numpy and numba. Numerical routines are compiled the
  first time they run and cached on disk.

## Help and contributing

[Report a bug or ask a question](https://github.com/NygenAnalytics/CytePool/issues) on
GitHub. The [README](https://github.com/NygenAnalytics/CytePool#development) lists the
checks a change must pass, and [](installation.md) shows how to build this site.
