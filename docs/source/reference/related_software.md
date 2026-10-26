---
description: The R and Python tools for sample-level statistics on pseudobulk and bulk data, what each does, its licence, and how CytePool relates to it.
---

# Related software

CytePool brings the sample-level methods of several R packages into one Python package,
reads the pseudobulk layouts of the scverse tools, and works beside other tools that
answer neighbouring questions. Licences and versions were read from each project's CRAN
or Bioconductor index entry, PyPI page or repository on 26 October 2026.

## Reimplemented and validated against

CytePool implements these methods from their papers and documentation, and runs the
packages as separate programs to compare results ([Validation](../validation.md)). The
validation uses the versions the Validation page lists, which can be older than the
current releases below.

| Tool | What it does | Licence | Current release |
| --- | --- | --- | --- |
| [limma](https://bioconductor.org/packages/limma) | Linear models with empirical Bayes moderation; voom, sample quality weights, gene-set tests [@ritchie2015] | GPL (>= 2) | Bioconductor 3.68.5 |
| [edgeR](https://bioconductor.org/packages/edgeR) | Negative binomial models, quasi-likelihood F-tests, TMM normalization, expression filtering [@chen2025] | GPL (>= 2) | Bioconductor 4.10.7 |
| [DESeq2](https://bioconductor.org/packages/DESeq2) | Negative binomial models with shrunken dispersions; Wald and likelihood ratio tests [@love2014] | LGPL (>= 3) | Bioconductor 1.52.0 |
| [variancePartition](https://bioconductor.org/packages/variancePartition) | Linear mixed models of expression: variance partitioning and dream [@hoffman2016; @hoffman2021] | GPL-2 | Bioconductor 1.42.0 |
| [speckle](https://bioconductor.org/packages/speckle) | propeller tests of cell-type proportions [@phipson2022] | GPL-3 | Bioconductor 1.12.0 |
| [metafor](https://cran.r-project.org/package=metafor) | Meta-analysis [@viechtbauer2010] | GPL (>= 2) | CRAN 5.2-1 |
| [lme4](https://cran.r-project.org/package=lme4) and [lmerTest](https://cran.r-project.org/package=lmerTest) | Linear mixed models; Satterthwaite's and Kenward-Roger's tests [@bates2015; @kuznetsova2017] | GPL (>= 2) | CRAN 2.1-0 and 3.2-1 |
| [fgsea](https://bioconductor.org/packages/fgsea) | Fast preranked gene set enrichment analysis, with p-values by multilevel splitting [@korotkevich2021]; CytePool's `gsea` adapts its algorithms, as `THIRD_PARTY_NOTICES.md` records | MIT | Bioconductor 1.38.0 |
| [GSVA](https://bioconductor.org/packages/GSVA) | Scores of gene sets in every sample: GSVA, ssGSEA, combined z-scores and PLAGE [@haenzelmann2013] | Artistic-2.0 (GPL (>= 2) up to version 2.0) | Bioconductor 2.6.14 |
| [PyDESeq2](https://github.com/scverse/PyDESeq2) | A Python implementation of DESeq2 [@muzellec2023]; CytePool also adapts some of its constants, as `THIRD_PARTY_NOTICES.md` records | MIT | PyPI 0.5.4 |

Because limma, edgeR, DESeq2, variancePartition, speckle, metafor, lme4, lmerTest and
GSVA are under copyleft licences, CytePool is written from their papers and
documentation and never from their source code, so that it can be distributed under the
BSD 3-Clause license.

## Read and written by CytePool

CytePool reads the pseudobulk layouts these tools write and writes them back
({api}`from_anndata`, {api}`to_anndata`, {api}`from_mudata`, {api}`to_mudata`), and its
cube files are Zarr v3 stores.

| Tool | What it does | Licence | Current release |
| --- | --- | --- | --- |
| [scanpy](https://github.com/scverse/scanpy) | Single-cell analysis; `get.aggregate` sums cells into pseudobulk profiles [@wolf2018] | BSD 3-Clause | PyPI 1.12.4 |
| [decoupler](https://github.com/scverse/decoupler) | Activity scores of pathways and regulators; `pp.pseudobulk` [@badiaimompel2022] | BSD 3-Clause | PyPI 2.3.0 |
| [LIANA+](https://github.com/saezlab/liana-py) | Cell-cell communication; `adata_to_views` makes a MuData of pseudobulk views [@dimitrov2024] | BSD 3-Clause | PyPI 1.10.0 |
| [anndata](https://github.com/scverse/anndata) and [mudata](https://github.com/scverse/mudata) | Annotated data matrices and multimodal containers [@virshup2024; @bredikhin2022] | BSD 3-Clause | PyPI 0.13.4 and 0.4.1 |
| [zarr-python](https://github.com/zarr-developers/zarr-python) | The Python implementation of Zarr; opens cube files | MIT | PyPI 3.4.1 |

## Beside CytePool

These tools answer neighbouring questions, or the same ones by other models. CytePool
neither uses nor reads them.

| Tool | What it does | Licence | Current release |
| --- | --- | --- | --- |
| [dreamlet](https://bioconductor.org/packages/dreamlet) | Precision-weighted mixed models on the pseudobulk profiles of every cell type [@hoffman2026a] | Artistic-2.0 | Bioconductor 1.10.0 |
| [muscat](https://bioconductor.org/packages/muscat) | Differential state analysis of multi-sample, multi-condition single-cell data, with pseudobulk and mixed models [@crowell2020] | GPL-3 | Bioconductor 1.26.0 |
| [crumblr](https://bioconductor.org/packages/crumblr) | Precision-weighted mixed models of cell-type composition [@hoffman2026b] | Artistic-2.0 | Bioconductor 1.4.5 |
| [sccomp](https://bioconductor.org/packages/sccomp) | Bayesian tests of composition and its variability [@mangiola2023] | GPL-3 | Bioconductor 2.4.0 |
| [miloR](https://bioconductor.org/packages/miloR) | Differential abundance in neighbourhoods of a k-nearest neighbour graph [@dann2022] | GPL-3 | Bioconductor 2.8.1 |
| [glmGamPoi](https://bioconductor.org/packages/glmGamPoi) | Gamma-Poisson generalized linear models for single-cell counts | GPL-3 | Bioconductor 1.24.0 |
| [pertpy](https://github.com/scverse/pertpy) | Perturbation analysis in the scverse ecosystem, with composition and differential expression tools [@heumos2026] | MIT | PyPI 1.4.0 |
| [statsmodels](https://github.com/statsmodels/statsmodels) | Statistical models in Python, including generalized linear and mixed models | BSD 3-Clause | PyPI 0.15.0 |
| [GSEApy](https://github.com/zqfang/GSEApy) | Gene set enrichment analysis in Python | BSD 3-Clause | PyPI 1.3.1 |
| [scvi-tools](https://github.com/scverse/scvi-tools) | Probabilistic models of single-cell omics data | BSD 3-Clause | PyPI 1.5.1 |

Independent Python ports of the copyleft R packages exist too. CytePool neither uses nor
reads them.

| Tool | What it does | Licence | Current release |
| --- | --- | --- | --- |
| [pylimma](https://github.com/john-mulvey/pylimma) | A Python port of limma | GPL-3.0-or-later | PyPI 0.2.0 |
| [python-limma](https://github.com/omicverse/py-limma) | A Python port of limma | LGPL-3.0-or-later | PyPI 0.1.0 |
| [edgePython](https://pypi.org/project/edgepython/) | A Python port of edgeR | GPL-3.0-or-later | PyPI 0.5.0 |
| [InMoose](https://github.com/epigenelabs/inmoose) | Python ports of limma, edgeR, DESeq2 and ComBat, among other tools | GPL-3.0-or-later | PyPI 0.9.1 |
| [dreampy](https://github.com/sbwells22/dreampy) | A Python implementation of the dreamlet framework | MIT, as its repository states | GitHub |
