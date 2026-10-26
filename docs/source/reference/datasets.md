---
description: The datasets the documentation's pages read and the networks CytePool downloads, where each comes from, its licence and citation, and how to prepare them on another machine.
---

# Datasets

The pages that run code read shared datasets by file name, such as
`read_cube("simulated_bulk.cube.zarr")`. A script in the repository,
[`docs/datasets.py`](https://github.com/NygenAnalytics/CytePool/blob/master/docs/datasets.py),
makes them deterministically into a cache outside the repository, downloads what they
need with checksums, and checks each against the fingerprint of every cube and the
digest of every other file it records. From the repository root:

```bash
uv run --group docs python docs/datasets.py list
uv run --group docs python docs/datasets.py prepare simulated_pbmc
uv run --group docs python docs/datasets.py verify
```

The cache is `CYTEPOOL_DOCS_DATA`, else `cytepool-docs` under `XDG_CACHE_HOME` or
`~/.cache`. Copy or link the files a page names into your working directory to run its
code.

| Dataset | What it holds | Source | Licence | Cite |
| --- | --- | --- | --- | --- |
| `simulated_pbmc` | 12 donors, each sampled before and after a stimulation; about 20,000 cells of six cell types over 2,000 genes, with planted changes in every cell type, a response program shared by all, a rise in the share of monocytes, and dendritic cells too rare in some samples; the cube of all five layers and the cells with the true effects | simulated by the script, seed 20261026 | part of CytePool, BSD 3-Clause | |
| `simulated_bulk` | 3 treatments x 4 replicates in two batches, 10,000 genes; drug A changes 500 genes, drug B 300; one noisier sample; the cube and the true effects | simulated by the script, seed 20261027 | part of CytePool, BSD 3-Clause | |
| `simulated_cohorts` | 6 case-control cohorts of one disease, 12 to 60 donors each, with the pseudobulk counts of four blood cell types over 2,000 genes; changes in every cohort, 100 of them in the monocytes, and 10 monocyte genes changed in each cohort alone; donor ids that repeat across cohorts, genes that two cohorts did not measure, and cell types named in each cohort's own words with their Cell Ontology ids; a cube per cohort and the true effects | simulated by the script, seed 20261028 | part of CytePool, BSD 3-Clause | |
| `kang2018` | Peripheral blood cells of 8 lupus patients, each split into an interferon-beta stimulated and an untreated culture: 24,673 cells pooled into 16 samples x 8 cell types x 35,635 genes (counts and detecting cells); a 77 MB download | [GEO GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583), batch 2 | GEO's terms: NCBI places no restrictions on the use of GEO data, though submitters may claim rights in theirs ([disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html)) | @kang2018 |
| `hallmark_interferon_alpha` | The 97 genes of the Hallmark interferon-alpha response set (M5911, MSigDB v2026.2.Hs) | [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE) | Creative Commons Attribution 4.0 ([terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp)) | @liberzon2015 |

The downloads of `kang2018` are the files that the validation script
`validation/interferon_stimulated_pbmcs.py` downloads, with the same checksums and in
the same directory, so a machine fetches them once; its cube is the one that script
tests. A notebook page names its datasets in its metadata; `make -C docs execute`
prepares them, links them into the directory the page runs in, and records the
fingerprint of each file it was given, which the build checks against the registry.

## Networks

The activity functions of {mod}`cytepool.activity` read networks that {api}`collectri`
and {api}`progeny` download on first use into `cytepool/networks` under
`XDG_CACHE_HOME` or `~/.cache` (or the directory given as `cache_dir`), each checked
against its size and SHA-256 digest; no file of them is in the repository.

| Network | What it holds | Source | Licence | Cite |
| --- | --- | --- | --- | --- |
| CollecTRI | 43,175 signed interactions of 1,186 human transcription factors with 6,693 target genes, gathered from twelve resources; the file `CollecTRI_regulons.csv`, 4,345,649 bytes, SHA-256 `4473c9189dd53dacc80297709ad1452dda1086a1cc2185f9a56146c261668701` | [Zenodo record 8192729](https://zenodo.org/records/8192729), version 2.0 | Creative Commons Attribution 4.0 for the record; each resource keeps its own terms, below | @muellerdott2023 |
| PROGENy | The weights (z-scores) and p-values of 22,479 human and 17,426 mouse genes in 14 signalling pathways; the R package `progeny_1.28.0.tar.gz`, 10,642,873 bytes, SHA-256 `2c41531ef6d088047e2830ccc5741aa0f28f48fb51d4c164cc8331e19b925d7b` | [progeny 1.28.0](https://bioconductor.org/packages/3.20/bioc/html/progeny.html), Bioconductor 3.20 | Apache License 2.0 | @schubert2018; mouse @holland2020b |

CollecTRI's resources and their terms, as OmniPath's records of resources list them
[@tuerei2021]: Creative Commons Attribution 4.0 for ExTRI, NTNU Curated, TFactS,
SIGNOR, IntAct, CytReg, GOA and Pavlidis2021; Creative Commons Attribution-ShareAlike
4.0 for TRRUST; the GNU Lesser General Public License 3.0 for HTRIdb; an informal
licence for DoRothEA's level A; and the Academic Free License 3.0 for GEREDB, which
OmniPath lists for academic use. `collectri(licence="commercial")`, the default, keeps
the 39,896 interactions that at least one resource listed for commercial use supports;
`collectri(licence="academic")` keeps all 43,175, including the 3,279 that GEREDB alone
supports. Each level is subject to the terms of the resources it keeps, such as
attribution; the providers' own statements of their terms prevail over these
records.
