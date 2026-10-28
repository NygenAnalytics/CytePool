# Third-party notices

CytePool is distributed under the BSD 3-Clause license (see [LICENSE](LICENSE)).
This file records material adapted from other permissively licensed projects, and the
source and license of every dataset used by the tests, the validation scripts or the
documentation. Methods, formulas and coefficients taken from papers, books and
specifications are cited in the `References` section of the module that implements
them, not here.

## Adapted material

### MyST book theme

- Source: <https://github.com/jupyter-book/myst-theme>
- Copyright (c) 2023 Executable Books.
- License: MIT
- Used in: `docs/build_docs.py` adjusts the documentation theme so navigation sections
  are expanded in the initial page and remain expanded when browser code starts.

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

`cytepool/gene_set_enrichment.py` ports the algorithms of fgsea 1.32.2
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

### decoupler

`cytepool/activity.py` follows decoupler 2.2.0 (https://github.com/scverse/decoupler) in
how it scores the sources of a network, so that given the same inputs it gives
decoupler's activities: from `mt/_ulm.py`, the regression of each profile over all of
its genes on a source's weights, zero for the genes that are not its targets, and the
t-statistic of the slope from their correlation; from `mt/_mlm.py`, the one model of
all the sources with an intercept and its t-statistics on n - m - 1 degrees of
freedom; from `mt/_waggr.py`, the weighted mean (`_wmean`) and sum (`_wsum`) and the
normalized score of `_perm`, the score less the mean of the permuted scores over their
standard deviation with n - 1; from `mt/_gsea.py`, the permutations of `_ridx`, a
tiled range of the genes shuffled row by row by numpy's generator from the seed; from
`pp/net.py`, the sources left out by `prune` with fewer than `tmin` targets among the
genes; from `pp/data.py`, the features left out by `_validate_mat` that are zero in
every observation; and from `mt/_run.py`, the Benjamini-Hochberg adjustment over the
sources of each observation. `cytepool/networks.py` follows `op/_collectri.py` in the
address of CollecTRI's file and in how it writes the resources and references of each
interaction, and `op/_progeny.py` in keeping PROGENy's genes with a p-value below 0.05.
`plot_top_activities` in `cytepool/plotting.py` chooses and orders its bars as
`pl/_barplot.py` does. Nothing is taken from `mt/_viper.py`. decoupler is described in
Badia-i-Mompel P, Velez Santiago J, Braunger J, Geiss C, Dimitrov D, Mueller-Dott S,
Taus P, Dugourd A, Holland CH, Ramirez Flores RO, Saez-Rodriguez J (2022). decoupleR:
ensemble of computational methods to infer biological activities from omics data.
Bioinformatics Advances 2, vbac016. doi:10.1093/bioadv/vbac016. It is distributed
under the BSD 3-Clause License:

```text
BSD 3-Clause License

Copyright (c) 2025, scverse®
Copyright (c) 2025, Pau Badia i Mompel, Saez lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### SumRank

`cytepool/stats/sumrank.py`, `cytepool/sumrank_across_studies.py` and
`cytepool/held_out_studies.py` implement SumRank and its held-out prediction from
Nakatsuka et al. (2025), following conventions of the authors'
code (https://github.com/nathan-nakatsuka/scRNA_Reproducibility, commit 46392f66;
`SumRank.R`, `CalibratePValueswithPermutations.R`, `PermutingCaseControlStatus.R`,
`UCell_AUC_code.R`): ties ranked by their average rank, floor(p k) of the studies summed
and the cap at half their number, a dataset without the cell type left out, the null
pooled over genes and permutations, and, for held-out scores, each cell type's scores
scaled to [0, 1], a missing cell type at their mean and the scores of genes up minus
those of genes down; `cytepool/plotting.py` draws `plot_sumrank` as their
`MakeManhattanPlot` (`MakingManhattanPlots.R`) draws it. No code is copied.
`validation/sumrank.py` downloads `SumRank.R`, `CalibratePValueswithPermutations.R` and
`MakingManhattanPlots.R` of that commit and runs them. Nakatsuka et al. (2025) is:
Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R. Improving
reproducibility of differentially expressed genes in single-cell transcriptomic studies
of neurodegenerative diseases through meta-analysis. Nature Communications 16, 7436.
doi:10.1038/s41467-025-62579-z. The code is distributed under the Apache License,
Version 2.0 (Copyright 2024 Nathan Nakatsuka; http://www.apache.org/licenses/LICENSE-2.0).

### scanpy

`plot_dotplot` in `cytepool/plotting.py` follows three conventions of scanpy 1.12.4's
dot plot (https://github.com/scverse/scanpy, `src/scanpy/plotting/_dotplot.py`) that its
documentation leaves open: the largest fraction rounded up to a tenth, the steps of the
key of the dots' areas (0.2, 0.1 or 0.05), and a scaled row of equal values drawn as 0.
No code is copied. scanpy is described in Wolf FA, Angerer P, Theis FJ (2018). SCANPY:
large-scale single-cell gene expression data analysis. Genome Biology 19, 15.
doi:10.1186/s13059-017-1382-0. It is distributed under the BSD 3-Clause License:

```text
BSD 3-Clause License

Copyright (c) 2025 scverse®
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
prepares in a cache outside the repository: three simulated with fixed random seeds by
that script, and the two below, GEO series GSE96583 and the MSigDB Hallmark
interferon-alpha response set. No file of these datasets is in the repository; the
script records the source, licence, citation and checksum of each. Two guides also
download the networks CollecTRI and PROGENy, as their sections below say.

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

### CollecTRI (Mueller-Dott et al. 2023)

`collectri()` in `cytepool/networks.py`, called by `validation/activity.py` and by the
guides `docs/source/guides/gene_sets.ipynb` and
`docs/source/guides/pathway_activity.ipynb`, downloads
`CollecTRI_regulons.csv`, version 2.0 of Zenodo record 8192729
(https://zenodo.org/records/8192729, doi:10.5281/zenodo.8192729), into a cache outside
the repository. No row of it is in the repository: `tests/data/activity` holds rows in
its format whose data are simulated. The record is distributed under the Creative
Commons Attribution 4.0 International License; the resources that CollecTRI gathers
keep their own terms, which OmniPath lists (https://omnipathdb.org/resources, read on
26 October 2026): Creative Commons Attribution 4.0 for ExTRI, NTNU Curated, TFactS,
SIGNOR, IntAct, CytReg, GOA and Pavlidis2021, Creative Commons
Attribution-ShareAlike 4.0 for TRRUST, the GNU Lesser General Public License 3.0 for
HTRIdb, an informal licence for DoRothEA's level A, and the Academic Free License 3.0
for GEREDB, which OmniPath lists for academic use. `collectri(licence="commercial")`,
the default, leaves out the interactions that GEREDB alone supports.

Mueller-Dott S, Tsirvouli E, Vazquez M, Ramirez Flores RO, Badia-i-Mompel P, Fallegger
R, Tuerei D, Laegreid A, Saez-Rodriguez J (2023). Expanding the coverage of regulons
from high-confidence prior knowledge for accurate estimation of transcription factor
activities. Nucleic Acids Research 51, 10934-10949. doi:10.1093/nar/gkad841

### PROGENy (Schubert et al. 2018; Holland, Szalai and Saez-Rodriguez 2020)

`progeny()` in `cytepool/networks.py`, called by `validation/activity.py` and by the
guide `docs/source/guides/pathway_activity.ipynb`, downloads the R
package progeny 1.28.0 from Bioconductor 3.20
(https://bioconductor.org/packages/3.20/bioc/src/contrib/progeny_1.28.0.tar.gz) into a
cache outside the repository and reads its data frames `model_human_full` and
`model_mouse_full`. No value of them is in the repository: `tests/data/activity` holds
a package in its layout whose data are simulated. progeny, its data included, is
distributed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

Schubert M, Klinger B, Kluenemann M, Sieber A, Uhlitz F, Sauer S, Garnett MJ,
Bluethgen N, Saez-Rodriguez J (2018). Perturbation-response genes reveal signaling
footprints in cancer gene expression. Nature Communications 9, 20.
doi:10.1038/s41467-017-02391-6

Holland CH, Szalai B, Saez-Rodriguez J (2020). Transfer of regulatory knowledge from
human to mouse for functional genomics analysis. Biochimica et Biophysica Acta (BBA) -
Gene Regulatory Mechanisms 1863, 194431. doi:10.1016/j.bbagrm.2019.194431
