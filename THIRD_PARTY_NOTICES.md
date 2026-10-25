# Third-party notices

CytePool is distributed under the BSD 3-Clause license (see [LICENSE](LICENSE)).
This file records material adapted from other permissively licensed projects, and the
source and license of every dataset used by the tests or the validation scripts.

## Adapted material

### PyDESeq2

`cytepool/stats/deseq2.py` takes from PyDESeq2 0.5.4 (https://github.com/owkin/PyDESeq2,
`pydeseq2/utils.py`) the constants and structure of the robust dispersion behind Cook's
distances (`trimmed_cell_variance`, `trimmed_variance`, `robust_method_of_moments_disp`:
trimmed fractions 1/3, 1/4 and 1/8 with scales 2.04, 1.86 and 1.51 by group size, the
maximum over design cells, and the floor 0.04), and its rough dispersions follow
`fit_rough_dispersions` and `fit_moments_dispersions`. PyDESeq2 is distributed under
the MIT License:

```text
MIT License

Copyright (c) 2022 Owkin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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
