<h1><img src="https://docs.nygen.io/assets/brand/logo/cytepool-dark.png" alt="" width="96" align="right"> CytePool</h1>

<p align="left">
  <a href="https://github.com/NygenAnalytics/CytePool/actions/workflows/tests.yml"><img src="https://github.com/NygenAnalytics/CytePool/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://docs.nygen.io/CytePool/"><img src="https://img.shields.io/github/actions/workflow/status/NygenAnalytics/CytePool/pages.yml?label=Documentation" alt="Documentation"></a>
  <a href="https://pypi.org/project/cytepool"><img src="https://img.shields.io/pypi/v/cytepool.svg?color=4c72b0" alt="PyPI"></a>
</p>

**Test which genes, cell types and pathways change between conditions in single-cell and bulk RNA-seq.**

Cells from one donor are not independent replicates. CytePool pools each sample and cell type into a cube of pseudobulk profiles, and runs limma, edgeR, DESeq2, dream, variancePartition, propeller, fgsea, GSVA, decoupler, metafor and SumRank in Python on numpy arrays.

## Install

Requires Python 3.13 or newer:

```bash
pip install cytepool
```

For AnnData, MuData and plotting, install `pip install "cytepool[mudata,plot]"`. See [Installation](https://docs.nygen.io/CytePool/installation).

## Example

Pool cells into a cube, then test a stimulation across cell types with limma-voom, pairing samples by donor:

```python
from cytepool import Design, camera, from_cells, limma_voom, read_gmt

cube = from_cells(
    counts,  # cells x genes, e.g. adata.X
    sample_labels=cell_sample,  # e.g. adata.obs["sample"]
    group_labels=cell_type,  # e.g. adata.obs["cell_type"]
    samples=samples,  # sample_id, donor, condition
    features=genes,  # feature_id and symbol, or adata.var
)
design = Design(
    cube.samples,
    "~ donor + condition",
    levels={"condition": ["ctrl", "stim"]},
    contrasts={"stim_vs_ctrl": "conditionstim"},
)
result = limma_voom(cube, design)
table = result.table("Monocytes", "stim_vs_ctrl")  # topTable columns

# Test pathway response in monocytes from a GMT file.
hallmark = read_gmt("h.all.v2026.2.Hs.symbols.gmt")
pathways = camera(result, hallmark, group="Monocytes", id_column="symbol")
```

If you already pooled profiles in scanpy, decoupler or LIANA, import them with `from_anndata` or `from_mudata`. See the [Quick start](https://docs.nygen.io/CytePool/quickstart) and [Gene sets and pathways](https://docs.nygen.io/CytePool/guides/gene-sets).

## What CytePool answers

| Question | Start with | Other CytePool functions | R and Python equivalents |
| --- | --- | --- | --- |
| Pool cells or import profiles | `from_cells`, `from_bulk` | `from_anndata`, `from_mudata`, `write_cube`, `read_cube` | scanpy `get.aggregate`, decoupler `pp.pseudobulk`, LIANA `adata_to_views` |
| Which genes change in each cell type? | `limma_voom` | `limma_trend`, `filter_by_expression`, `tmm_factors` | limma `voom`, `lmFit`, `eBayes`, `topTable`; edgeR `filterByExpr`, `calcNormFactors` |
| Which genes change in bulk RNA-seq? | `limma_voom` on `from_bulk` | `fit_quasi_likelihood`, `quasi_likelihood_f_test`, `fit_deseq2`, `deseq2_wald_test`, `deseq2_likelihood_ratio_test`, `treat` | edgeR `glmQLFit`, `glmQLFTest`; DESeq2 `DESeq`, `results`; limma `treat` |
| Which genes change, with several samples of each donor? | `dream_voom` | `fit_mixed_model`, `fixed_effect_tests`, `fixed_effect_f_test` | variancePartition `dream`, `voomWithDreamWeights`; lme4 `lmer`, lmerTest, pbkrtest |
| What drives the variation between samples? | `variance_partitioning` | `principal_components`, `plot_mds`, `plot_pca` | variancePartition `fitExtractVarPartModel`; DESeq2 `plotPCA`; limma `plotMDS` |
| Which cell types expand or contract? | `propeller` | `plot_composition` | speckle `propeller` |
| Do the genes of a pathway respond to a contrast? | `camera` | `fry`, `roast`, `mroast`, `romer`, `over_representation` | limma `camera`, `fry`, `roast`, `mroast`, `romer`, `kegga` |
| Do the genes of a pathway gather along a ranking? | `gsea` | `plot_enrichment`, `plot_gsea_table` | fgsea `fgseaMultilevel`, `plotEnrichment`, `plotGseaTable` |
| How active is a pathway in each sample? | `gsva` | `ssgsea`, `combined_z_scores`, `plage`, `limma` | GSVA `gsva`, `ssgseaParam`, `zscoreParam`, `plageParam` |
| Which regulators and pathways are active? | `ulm` | `mlm`, `weighted_aggregate`, `collectri`, `progeny`, `limma` | decoupler `mt.ulm`, `mt.mlm`, `mt.waggr`; CollecTRI, PROGENy |
| What do several studies say together? | `meta_analysis` | `CubeCollection`, `analyse_each_study`, `combine_p_values` | metafor `rma.uni`; metap |
| Which changes replicate across studies? | `sumrank` | `replication`, `partial_conjunction`, `proportion_of_true_nulls`, `held_out_prediction` | SumRank; qvalue `pi0est` |

[Choosing a method](https://docs.nygen.io/CytePool/statistics/choosing-a-method) compares the alternatives for each question, and [Validation](https://docs.nygen.io/CytePool/validation) benchmarks CytePool against the reference packages.

## Documentation

The documentation is at <https://docs.nygen.io/CytePool/>.

## Development

```bash
uv sync --extra test --group dev --group docs
uv run pytest
uv run ruff check cytepool tests validation docs
uv run ruff format --check cytepool tests validation docs
uv run mypy cytepool
make -C docs api
```

## License

BSD 3-Clause. See [LICENSE](https://github.com/NygenAnalytics/CytePool/blob/master/LICENSE) and [THIRD_PARTY_NOTICES.md](https://github.com/NygenAnalytics/CytePool/blob/master/THIRD_PARTY_NOTICES.md).
