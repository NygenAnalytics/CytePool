# Third-party notices

CytePool is distributed under the BSD 3-Clause license (see [LICENSE](LICENSE)).
This file records material adapted from other permissively licensed projects, and the
source and license of every dataset used by the tests or the validation scripts.

## Adapted material

None.

## Test and validation data

The files under `tests/data/` are written by the scripts in `validation/`. Except for
the dataset below, their inputs are generated with fixed random seeds. Reference values
in those files come from the packages each script names.

### GEO series GSE96583 (Kang et al. 2018)

`validation/interferon_stimulated_pbmcs.py` downloads `GSE96583_RAW.tar`,
`GSE96583_batch2.genes.tsv.gz` and `GSE96583_batch2.total.tsne.df.tsv.gz` from the Gene
Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) into a
cache outside the repository, and `tests/data/interferon_stimulated_pbmcs/limma.npz`
holds pseudobulk counts derived from them.

Kang HM, Subramaniam M, Targ S, et al. (2018). Multiplexed droplet single-cell
RNA-sequencing using natural genetic variation. Nature Biotechnology 36, 89-94.

Terms, from NCBI's GEO disclaimer (https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html):
"NCBI places no restrictions on the use or distribution of the GEO data. However, some
submitters may claim patent, copyright, or other intellectual property rights in all or
a portion of the data they have submitted. NCBI is not in a position to assess the
validity of such claims, and therefore cannot provide comment or unrestricted permission
concerning the use, copying, or distribution of the information contained in GEO."

### MSigDB Hallmark gene set HALLMARK_INTERFERON_ALPHA_RESPONSE

The gene symbols of this set (M5911) appear in
`validation/interferon_stimulated_pbmcs.py`.

Liberzon A, et al. (2015). The Molecular Signatures Database hallmark gene set
collection. Cell Systems 1, 417-425.

The contents of MSigDB are copyright (c) 2004-2025 Broad Institute, Inc., Massachusetts
Institute of Technology, and Regents of the University of California, and are
distributed under the Creative Commons Attribution 4.0 International License
(https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).
