---
description: Test which genes, cell types and pathways change between conditions in pseudobulk and bulk omics data, with the sample-level statistics of limma, edgeR, DESeq2, dream, propeller and metafor, in Python.
---

# CytePool

**Sample-level statistics for pseudobulk and bulk omics data, in Python.**

Studies that compare conditions, donors or treatments ask questions about samples:
cells of one donor are not independent replicates, and a test that treats them as
such finds differences that other donors do not share [@squair2021; @zimmerman2021].
CytePool pools the cells of each sample and cell type into a *cube* of pseudobulk
profiles, which can be saved and shared, and tests them with the methods that are
standard in R: limma, edgeR, DESeq2, dream, propeller and metafor. Its only required
dependencies are numpy and numba, and every method is checked against an independent
reference, usually the package that defined it ([Validation](validation.md)).

Start with [Installation](installation.md) and the [Quick start](quickstart.ipynb),
which goes from cells to results in a few minutes. [Why pseudobulk](why_pseudobulk.ipynb)
shows why the tests compare samples, [Choosing a method](statistics/choosing_a_method.md)
which function answers which question and when to choose another, and the [API
reference](/api/index.html) documents every function.

:::{note} Early development
Nothing is released yet, and the interface may still change.
:::

## The questions CytePool answers

Each question has a function that answers it by default, and a guide that runs it on
data. [Choosing a method](statistics/choosing_a_method.md) gives the alternatives for
each, and what is not implemented yet.

| Question | Start with | Guide |
| --- | --- | --- |
| Which genes change in each cell type? | {api}`limma_voom` | [Genes that change in each cell type](guides/genes_per_cell_type.ipynb) |
| Which genes change in bulk RNA-seq? | {api}`limma_voom` on a cube from {api}`from_bulk` | [Bulk RNA-seq](guides/bulk_rna_seq.ipynb) |
| Which genes change, with several samples of each donor? | {api}`dream_voom` | [Repeated measures and donors](guides/repeated_measures.ipynb) |
| What drives the variation between samples? | {api}`variance_partitioning` | [Repeated measures and donors](guides/repeated_measures.ipynb) |
| Which cell types expand or contract? | {api}`propeller` | [Cell-type composition](guides/composition.ipynb) |
| Do the genes of a pathway respond to a contrast? | {api}`camera` | [Gene sets and pathways](guides/gene_sets.ipynb) |
| Do the genes of a pathway gather along a ranking? | {api}`gsea` | [Gene sets and pathways](guides/gene_sets.ipynb) |
| How active is a pathway in each sample? | {api}`gsva` | [Pathway and regulator activity](guides/pathway_activity.ipynb) |
| Which regulators and pathways are active? | {api}`ulm` | [Pathway and regulator activity](guides/pathway_activity.ipynb) |
| What do several studies say together? | {api}`meta_analysis` | [Align and combine studies](guides/combine_studies.ipynb) |
| Which changes replicate across studies? | {api}`sumrank` | [Replication across studies](guides/replication_across_studies.ipynb) |

Grouping patients by their profiles, changes in cell-cell interactions and the response
to drugs and doses are not answered yet ([Not implemented
yet](statistics/choosing_a_method.md#not-implemented-yet)). Cubes convert to AnnData and
MuData for other tools ([Work with AnnData and MuData](guides/anndata_and_mudata.ipynb)).

## Principles

- **Samples are the unit of replication.** Cells are pooled into samples before any
  test, and a cube records how many cells each profile holds.
- **Traceable results.** A cube's fingerprint names its values, its provenance record
  says how it was made, and the results of the analyses of a cube, such as those of
  `limma_voom` and `propeller`, keep both.
- **Checked against the reference implementations.** Each area has a script in
  [`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation)
  that runs an independent reference, usually the package that defined the method, as a
  separate program on the same inputs.
- **Implemented from the literature.** CytePool is written from the papers and
  documentation of the methods, and stays BSD-licensed ([Related
  software](reference/related_software.md)).
- **Two required dependencies,** numpy and numba. Numerical routines are compiled the
  first time they run and cached on disk.

## Help and contributing

[Report a bug or ask a question](https://github.com/NygenAnalytics/CytePool/issues) on
GitHub. The [README](https://github.com/NygenAnalytics/CytePool#development) lists the
checks a change must pass, and [](installation.md) shows how to build this site.
