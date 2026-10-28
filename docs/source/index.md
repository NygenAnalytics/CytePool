---
description: Test which genes, cell types and pathways change between conditions in pseudobulk and bulk omics data, with the sample-level statistics of limma, edgeR, DESeq2, dream, propeller and metafor, in Python.
---

# CytePool

**Test which genes, cell types and pathways change between conditions in single-cell and bulk omics data.**

Studies that compare conditions, donors or treatments ask questions about samples. Cells from one donor are not independent replicates, and a test that treats cells as replicates finds differences that other donors do not share [@squair2021; @zimmerman2021]. CytePool pools each sample and cell type into a *cube* of pseudobulk profiles, and runs limma, edgeR, DESeq2, dream, variancePartition, propeller, fgsea, GSVA, decoupler, metafor and SumRank in Python on numpy arrays.

CytePool requires only numpy and numba, and each method is validated against the reference R or Python package ([Validation](validation.md)). Start with [Installation](installation.md) and the [Quick start](quickstart.ipynb), or browse [Why pseudobulk](why_pseudobulk.ipynb), [Choosing a method](statistics/choosing_a_method.md) and the [API reference](/api/index.html).

## The questions CytePool answers

[Choosing a method](statistics/choosing_a_method.md) compares the default function and alternatives for each question, and lists what is not implemented yet.

| Question | Start with | Alternatives | R and Python equivalents | Guide |
| --- | --- | --- | --- | --- |
| Pool cells or import profiles | {api}`from_cells`, {api}`from_bulk` | {api}`from_anndata`, {api}`from_mudata`, {api}`write_cube`, {api}`read_cube` | scanpy `get.aggregate`, decoupler `pp.pseudobulk`, LIANA `adata_to_views` | [Pool cells into pseudobulk profiles](guides/pool_cells.ipynb), [Work with AnnData and MuData](guides/anndata_and_mudata.ipynb) |
| Which genes change in each cell type? | {api}`limma_voom` | {api}`limma_trend`, {api}`filter_by_expression`, {api}`tmm_factors` | limma `voom`, `lmFit`, `eBayes`, `topTable`; edgeR `filterByExpr`, `calcNormFactors` | [Genes that change in each cell type](guides/genes_per_cell_type.ipynb) |
| Which genes change in bulk RNA-seq? | {api}`limma_voom` on {api}`from_bulk` | {api}`fit_quasi_likelihood`, {api}`quasi_likelihood_f_test`, {api}`fit_deseq2`, {api}`deseq2_wald_test`, {api}`deseq2_likelihood_ratio_test`, {api}`treat` | edgeR `glmQLFit`, `glmQLFTest`; DESeq2 `DESeq`, `results`; limma `treat` | [Bulk RNA-seq](guides/bulk_rna_seq.ipynb) |
| Which genes change, with several samples of each donor? | {api}`dream_voom` | {api}`fit_mixed_model`, {api}`fixed_effect_tests`, {api}`fixed_effect_f_test` | variancePartition `dream`, `voomWithDreamWeights`; lme4 `lmer`, lmerTest, pbkrtest | [Repeated measures and donors](guides/repeated_measures.ipynb) |
| What drives the variation between samples? | {api}`variance_partitioning` | {api}`principal_components`, {api}`plot_mds`, {api}`plot_pca` | variancePartition `fitExtractVarPartModel`; DESeq2 `plotPCA`; limma `plotMDS` | [Repeated measures and donors](guides/repeated_measures.ipynb), [Normalization, filtering and sample quality](guides/normalization_and_quality.ipynb) |
| Which cell types expand or contract? | {api}`propeller` | {api}`plot_composition` | speckle `propeller` | [Cell-type composition](guides/composition.ipynb) |
| Do the genes of a pathway respond to a contrast? | {api}`camera` | {api}`fry`, {api}`roast`, {api}`mroast`, {api}`romer`, {api}`over_representation` | limma `camera`, `fry`, `roast`, `mroast`, `romer`, `kegga` | [Gene sets and pathways](guides/gene_sets.ipynb) |
| Do the genes of a pathway gather along a ranking? | {api}`gsea` | {api}`plot_enrichment`, {api}`plot_gsea_table` | fgsea `fgseaMultilevel`, `plotEnrichment`, `plotGseaTable` | [Gene sets and pathways](guides/gene_sets.ipynb) |
| How active is a pathway in each sample? | {api}`gsva` | {api}`ssgsea`, {api}`combined_z_scores`, {api}`plage`, {api}`limma` | GSVA `gsva`, `ssgseaParam`, `zscoreParam`, `plageParam` | [Pathway and regulator activity](guides/pathway_activity.ipynb) |
| Which regulators and pathways are active? | {api}`ulm` | {api}`mlm`, {api}`weighted_aggregate`, {api}`collectri`, {api}`progeny`, {api}`limma` | decoupler `mt.ulm`, `mt.mlm`, `mt.waggr`; CollecTRI, PROGENy | [Pathway and regulator activity](guides/pathway_activity.ipynb) |
| What do several studies say together? | {api}`meta_analysis` | {api}`CubeCollection`, {api}`analyse_each_study`, {api}`combine_p_values` | metafor `rma.uni`; metap | [Align and combine studies](guides/combine_studies.ipynb) |
| Which changes replicate across studies? | {api}`sumrank` | {api}`replication`, {api}`partial_conjunction`, {api}`proportion_of_true_nulls`, {api}`held_out_prediction` | SumRank; qvalue `pi0est` | [Replication across studies](guides/replication_across_studies.ipynb) |

## Principles

- **Samples are the unit of replication.** CytePool pools cells into sample-level profiles before testing, and records the cell count of each profile.
- **Traceable results.** Cubes store a SHA-256 content fingerprint and a provenance record, and result objects from functions such as `limma_voom` and `propeller` retain both.
- **Validated against the reference packages.** Validation scripts in [`validation/`](https://github.com/NygenAnalytics/CytePool/tree/master/validation) compare CytePool outputs against the reference R and Python packages on shared inputs.
- **Implemented from the literature.** Methods are implemented from published papers and package documentation under the BSD 3-Clause license ([Related software](reference/related_software.md)).
- **Two required dependencies,** numpy and numba. Numerical routines compile on first use and cache machine code on disk.

## Help and contributing

[Report a bug or ask a question](https://github.com/NygenAnalytics/CytePool/issues) on GitHub. The [README](https://github.com/NygenAnalytics/CytePool#development) lists the development checks, and [](installation.md) covers building the documentation.
