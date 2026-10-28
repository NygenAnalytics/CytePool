<h1><img src="https://docs.nygen.io/assets/brand/logo/cytepool-dark.png" alt="" width="96" align="right"> CytePool</h1>

Test which genes, cell types and pathways change between conditions in studies of many
samples, from single-cell or bulk RNA-seq data. Cells of one donor are not independent
replicates, so CytePool pools the cells of each sample and cell type into a cube of
pseudobulk profiles and tests the samples with the methods that are standard in R:
limma, edgeR, DESeq2, dream, propeller and metafor. It runs them in Python, with numpy
and numba as its only required dependencies, and every method is checked against an
independent reference, usually the package that defined it.

**Status:** early development. Nothing is released yet.

## Install

CytePool needs Python 3.13 or newer:

```bash
pip install "git+https://github.com/NygenAnalytics/CytePool.git"
```

The extras `anndata`, `mudata` and `plot` add conversions to AnnData and MuData and the
figures; [Installation](https://docs.nygen.io/CytePool/installation) gives the details.

## Example

Pool the cells into a cube, then test a stimulation in every cell type with
limma-voom, with a design that pairs the samples of each donor:

```python
from cytepool import Design, camera, from_cells, limma_voom, read_gmt

# counts: cells x genes; cell_sample and cell_type: the labels of each cell;
# samples: a table with sample_id, donor and condition; genes: with feature_id
# and symbol.
cube = from_cells(
    counts,
    sample_labels=cell_sample,
    group_labels=cell_type,
    samples=samples,
    features=genes,
)
design = Design(
    cube.samples,
    "~ donor + condition",
    levels={"condition": ["ctrl", "stim"]},
    contrasts={"stim_vs_ctrl": "conditionstim"},
)
result = limma_voom(cube, design)
table = result.table("Monocytes", "stim_vs_ctrl")  # topTable's columns

# Which pathways respond in the monocytes, from a GMT file of gene sets.
hallmark = read_gmt("h.all.v2026.2.Hs.symbols.gmt")
pathways = camera(result, hallmark, group="Monocytes", id_column="symbol")
```

The result keeps the cube's fingerprint, so that it can be traced to the profiles it
was computed from. The [Quick start](https://docs.nygen.io/CytePool/quickstart) pools
and tests a simulated study this way, and [Gene sets and
pathways](https://docs.nygen.io/CytePool/guides/gene-sets) tests pathways.

## What CytePool answers

| Question | Start with |
| --- | --- |
| Which genes change in each cell type? | `limma_voom` |
| Which genes change in bulk RNA-seq? | `limma_voom` on a cube from `from_bulk` |
| Which genes change, with several samples of each donor? | `dream_voom` |
| What drives the variation between samples? | `variance_partitioning` |
| Which cell types expand or contract? | `propeller` |
| Do the genes of a pathway respond to a contrast? | `camera` |
| Do the genes of a pathway gather along a ranking? | `gsea` |
| How active is a pathway in each sample? | `gsva` |
| Which regulators and pathways are active? | `ulm` |
| What do several studies say together? | `meta_analysis` |
| Which changes replicate across studies? | `sumrank` |

[Choosing a method](https://docs.nygen.io/CytePool/statistics/choosing-a-method) gives
the alternatives for each question and what is not implemented yet, and
[Validation](https://docs.nygen.io/CytePool/validation) how closely each method agrees
with its reference.

## Documentation

The documentation is at <https://docs.nygen.io/CytePool/>: a quick start, guides to
building cubes, comparing conditions and combining studies, chapters on how each method
works, the API reference and the Validation page. Its sources are in
[`docs/`](https://github.com/NygenAnalytics/CytePool/tree/master/docs), and
[Installation](https://docs.nygen.io/CytePool/installation#build-the-documentation)
shows how to build it.

## Development

```bash
uv sync --extra test --group dev --group docs
uv run pytest
uv run ruff check cytepool tests validation docs
uv run ruff format --check cytepool tests validation docs
uv run mypy cytepool
make -C docs api
```

[`AGENTS.md`](https://github.com/NygenAnalytics/CytePool/blob/master/AGENTS.md) gives
the conventions of the code, its tests and its documentation.

## License

BSD 3-Clause. See
[LICENSE](https://github.com/NygenAnalytics/CytePool/blob/master/LICENSE) and
[THIRD_PARTY_NOTICES.md](https://github.com/NygenAnalytics/CytePool/blob/master/THIRD_PARTY_NOTICES.md).
