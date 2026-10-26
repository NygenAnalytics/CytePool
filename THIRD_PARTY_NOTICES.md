# Third-party notices

CytePool is distributed under the BSD 3-Clause license (see [LICENSE](LICENSE)).
This file records material adapted from other permissively licensed projects, and the
source and license of every dataset used by the tests, the validation scripts or the
documentation. Methods, formulas and coefficients taken from papers, books and
specifications are cited in the `References` section of the module that implements
them, not here.

## Adapted material

### PyDESeq2

`cytepool/stats/deseq2.py` takes from PyDESeq2 0.5.4 (https://github.com/owkin/PyDESeq2,
`pydeseq2/utils.py`) the constants and structure of the robust dispersion behind Cook's
distances (`trimmed_cell_variance`, `trimmed_variance`, `robust_method_of_moments_disp`:
trimmed fractions 1/3, 1/4 and 1/8 with scales 2.04, 1.86 and 1.51 by group size, the
maximum over design cells, and the floor 0.04), and its rough dispersions follow
`fit_rough_dispersions` and `fit_moments_dispersions`. Where the parametric dispersion
trend does not converge with positive coefficients, it uses the mean trend with a
warning, as `DeseqDataSet._fit_parametric_dispersion_trend` (`pydeseq2/dds.py`) does.
PyDESeq2 is described in Muzellec B, Telenczuk M, Cabeli V, Andreux M (2023). PyDESeq2:
a python package for bulk RNA-seq differential expression analysis. Bioinformatics 39,
btad547. doi:10.1093/bioinformatics/btad547. It is distributed under the MIT License:

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

### fgsea

`cytepool/gsea.py` ports the algorithms of fgsea 1.32.2
(https://bioconductor.org/packages/fgsea, https://github.com/ctlab/fgsea), keeping their
structure, constants and order of operations, so that given fgsea's seeds it gives
fgsea's results: from `src/fastGSEA.cpp`, the block prefix sums (`SegmentTree`), the
scores of all the first genes of a random sample from the upper convex hulls of their
walk (`gseaStats1`, `calcGseaStatCumulative`, with the hull tolerance 1e-13 and the floor
of the statistics, the smaller of 1e-5 and the smallest nonzero one, divided by 1024) and
the counts of `calcGseaStatCumulativeBatch`; from `src/fgseaMultilevelSupplement.cpp`,
`src/fgseaMultilevel.cpp`, `src/esCalculation.cpp` and `src/util.cpp`, the multilevel
method (`EsRuler`: `extend`, `duplicateSamples`, `perturbate` with its chunks, its
remembered gene and its swaps of a tenth of a sample, `getPvalue`,
`calcLogCorrection`), `calcES`, `calcPositiveES`, the sampler of distinct genes
(`combination`, with its 100 draws per gene), the random integers of `uid_wrapper`,
`betaMeanLog` and `fgseaMultilevelCpp`; and from `R/fgsea.R`, `R/fgseaMultilevel.R` and
`R/util.R`, the enrichment score and leading edge of `calcGseaStat`, the normalized
scores and p-values of `fgseaSimpleImpl`, the chunks of permutations of `fgseaSimple`,
and the rules of `fgseaMultilevel` (sets with fewer than 10 random scores of their sign,
the choice of the multilevel method by its expected error, `multilevelError`, the
denominator of its p-values and `eps`). `cytepool/_ranked_genes.py` computes the walk of
`calcGseaStat`, and `cytepool/plotting.py` draws `plotEnrichment` from the data of
`plotEnrichmentData` and the rows of `plotGseaTable` with its numbers written as its
`valueToExpExpression` writes them. `validation/gsea.py` states fgseaMultilevel's choice
of the multilevel method in R, to compare CytePool's with it. fgsea is described in
Korotkevich G, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A (2021). Fast gene
set enrichment analysis. bioRxiv 060012, version 3. doi:10.1101/060012. It is
distributed under the MIT License (its `DESCRIPTION` says "MIT + file LICENCE"), as its
`LICENCE` file states:

```text
YEAR: 2016-2019
COPYRIGHT HOLDER: Alexey Sergushichev

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

### numba

`cytepool/_jit.py` places the cache of its compiled routines with subclasses of numba's
cache classes (`numba/core/caching.py`), and its `PackageLocator` follows numba 0.68.0's
`UserProvidedCacheLocator` and `_SourceFileBackedLocatorMixin`: the same attributes
(`_py_file`, `_lineno`, `_cache_path`), the line number as disambiguator, and a
`from_function` that returns no locator when the cache directory cannot be made. numba
(https://github.com/numba/numba) is distributed under the BSD 2-Clause License:

```text
Copyright (c) 2012, Anaconda, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Test, validation and documentation data

The files under `tests/data/` are written by the scripts in `validation/`. Except for
the dataset below, their inputs are generated with fixed random seeds. Reference values
in those files come from the packages each script names.

The pages of the documentation (`docs/source/`) read the datasets that `docs/datasets.py`
prepares in a cache outside the repository: two simulated with fixed random seeds by
that script, and the two below, GEO series GSE96583 and the MSigDB Hallmark
interferon-alpha response set. No file of these datasets is in the repository; the
script records the source, licence, citation and checksum of each.

### GEO series GSE96583 (Kang et al. 2018)

`validation/interferon_stimulated_pbmcs.py` downloads `GSE96583_RAW.tar`,
`GSE96583_batch2.genes.tsv.gz` and `GSE96583_batch2.total.tsne.df.tsv.gz` from the Gene
Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) into a
cache outside the repository, and `tests/data/interferon_stimulated_pbmcs/limma.npz`
holds pseudobulk counts derived from them. `docs/datasets.py` reads the same files, with
the same checksums and from the same cache, and pools them into the cube that the
documentation's pages read (`kang2018`), which it writes to its own cache outside the
repository.

Kang HM, Subramaniam M, Targ S, Nguyen M, Maliskova L, McCarthy E, Wan E, Wong S,
Byrnes L, Lanata CM, Gate RE, Mostafavi S, Marson A, Zaitlen N, Criswell LA, Ye CJ
(2018). Multiplexed droplet single-cell RNA-sequencing using natural genetic variation.
Nature Biotechnology 36, 89-94. doi:10.1038/nbt.4042

Terms, from NCBI's GEO disclaimer (https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html):
"NCBI places no restrictions on the use or distribution of the GEO data. However, some
submitters may claim patent, copyright, or other intellectual property rights in all or
a portion of the data they have submitted. NCBI is not in a position to assess the
validity of such claims, and therefore cannot provide comment or unrestricted permission
concerning the use, copying, or distribution of the information contained in GEO."

### MSigDB Hallmark gene set HALLMARK_INTERFERON_ALPHA_RESPONSE

The 97 gene symbols of this set (M5911, MSigDB release v2026.2.Hs, from
https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE)
appear in `validation/interferon_stimulated_pbmcs.py` and in `docs/datasets.py`, which
writes them for the documentation's pages (`hallmark_interferon_alpha`).

Liberzon A, Birger C, Thorvaldsdottir H, Ghandi M, Mesirov JP, Tamayo P (2015). The
Molecular Signatures Database hallmark gene set collection. Cell Systems 1, 417-425.
doi:10.1016/j.cels.2015.12.004

The contents of this gene set are protected by copyright (c) 2004-2026 Broad Institute,
Inc., Massachusetts Institute of Technology, and Regents of the University of
California, as its page states. MSigDB versions v6.0 to v7.5.1, and v2022.1 and above,
are distributed under the Creative Commons Attribution 4.0 International License
(https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).
