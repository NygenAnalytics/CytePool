---
description: What a CytePool cube holds, why every layer is a sum over cells, how merging cell types and pooling samples stay exact, the principal components of its samples and the dot plot of its cells, and how the cubes of several studies are aligned.
---

# The cube and its additive layers

A cube holds one pseudobulk profile for every sample and group (usually a cell type):
the sums, over the cells of that sample and group, of each feature's counts and of a few
statistics of its normalized values. {api}`from_cells` builds it from a cells x features
count matrix and the sample and group label of every cell; {api}`from_bulk` wraps a
samples x features matrix as a cube with one group.

## The layers

For sample $j$, group $g$ and feature $f$, let $C_{jg}$ be the cells of that sample and
group, $y_{cf}$ the count of cell $c$, $L_c$ its library size (its total count, or a
given value) and $T$ the target sum (10,000 by default). Each cell's normalized value is
$x_{cf} = y_{cf} T / L_c$, and the layers are

$$
\begin{aligned}
\text{counts}_{gjf} &= \sum_{c \in C_{jg}} y_{cf}, &
\text{detected}_{gjf} &= \sum_{c \in C_{jg}} [y_{cf} > 0], \\
\text{linear\_sum}_{gjf} &= \sum_{c \in C_{jg}} x_{cf}, &
\text{log\_sum}_{gjf} &= \sum_{c \in C_{jg}} \log(1 + x_{cf}), \\
\text{log\_sum\_sq}_{gjf} &= \sum_{c \in C_{jg}} \log(1 + x_{cf})^2, &
n_{jg} &= |C_{jg}|.
\end{aligned}
$$

Layers are stored as groups x samples x features, so that one group is a samples x
features matrix, the shape a per-group model fits. The counts layer is the one every
cube needs; the others are optional. Bulk cubes have counts only and no cell numbers.

{api}`Cube.layer` returns a layer, for one group or all. From the sums come the
quantities of every entry: {api}`Cube.mean` (the mean normalized value, on the log or
the linear scale), {api}`Cube.variance` (the sample variance of the log values,
$(\text{log\_sum\_sq} - \text{log\_sum}^2 / n) / (n - 1)$),
{api}`Cube.fraction_detected` (detected over $n$) and {api}`Cube.library_sizes` (the
counts summed over features).

## Exact regrouping

Every layer is a sum over cells, so the profile of a union of cells is the sum of the
profiles of its parts. Merging two cell types into one, or pooling the samples of a
donor, needs no cells:

$$
\text{layer}(C_{jg} \cup C_{jh}) = \text{layer}(C_{jg}) + \text{layer}(C_{jh}).
$$

{api}`Cube.regroup` merges groups, {api}`Cube.collapse_samples` pools the samples that
share a value of a sample column, and {api}`Cube.select` keeps some samples, groups and
features. Integer layers are summed exactly, and a sum that would exceed the largest
64-bit integer raises an error instead of wrapping around. Means and variances, which
are not sums, cannot be merged this way; that is why the cube keeps sums and derives
them.

## Samples and cells at a glance

Two figures show a cube's data before any test. {api}`principal_components` places the
samples of one cell type on the principal components of their most variable genes
[@pearson1901], as DESeq2's `plotPCA` does [@love2014]. The samples are prepared as
for an MDS plot: those with fewer than 10 cells are left out, genes are kept by
{api}`filter_by_expression`, and the values are log-CPM with TMM factors and an average
count of 2 added. The `top` genes of largest variance over the samples (500 by
default), each centred on its mean, form a samples x genes matrix $X = U S V^\top$;
the samples' coordinates are $U S$, each component's share of the variance is its
squared singular value over their sum, and the result, a {api}`PrincipalComponents`,
keeps the loadings $V$ of the genes. The sign of a component is arbitrary; the
coordinate of largest absolute value is made positive. `plot_pca` draws two
components, labelled with their shares as `plotPCA` labels them.

The dot plot shows, for chosen genes and cell types, the fraction of the cells that
express each gene and their mean expression, as scanpy draws it [@wolf2018]. A cube
holds what it needs without the cells. Because every layer is a sum, the fraction and
the mean over the cells of all the samples pooled,

$$
f_{gf} = \frac{\sum_j \text{detected}_{gjf}}{\sum_j n_{jg}}, \qquad
m_{gf} = \frac{\sum_j \text{log\_sum}_{gjf}}{\sum_j n_{jg}},
$$

are those computed from the cells themselves, and so is the mean over the cells that
express the gene, $\sum_j \text{log\_sum}_{gjf} / \sum_j \text{detected}_{gjf}$,
since $\log(1 + 0) = 0$ for the others. {api}`Cube.fraction_detected` and {api}`Cube.mean`
give the same quantities per sample. `plot_dotplot` draws them, pooled over every sample
or some, with scanpy's scaling of the means by gene or by cell type and its groups of
genes.

## What a cube carries

- **Tables.** Samples, groups and features each have a table ({api}`Table`) with a
  unique text id (`sample_id`, `group_id`, `feature_id`) and any other columns, such as
  donor and condition. Designs are built from the sample table and matched to the cube
  by `sample_id`, never by position.
- **A fingerprint.** {api}`Cube.fingerprint` is a SHA-256 digest of the tables, cell
  numbers, normalization and every layer [@nist2015]. Two cubes with the same values
  have the same fingerprint however they were made, so a result can name the cube it
  came from. Its byte-level encoding is documented, so another program can compute it
  ([Cube files and provenance records](../reference/cube_files.md)).
- **A provenance record.** {api}`Cube.provenance` says where the data came from (a
  record the caller gives as `source`), how the cube was built, and every operation
  since, each with the fingerprint of the cube it was applied to. {api}`Cube.history`
  prints it in plain terms.

## Cubes larger than memory

{api}`write_cube` saves a cube as a Zarr v3 store [@miles2025], in chunks of one group,
every sample and up to 4,096 features, and {api}`read_cube` opens it without reading the
layers: an analysis then reads one group at a time, and only the chunks that hold the
features it needs. The differential-expression functions keep one group in memory at a
time. A layer can also be any array-like with `shape`, `dtype` and indexing, such as a
numpy memmap or an HDF5 or Zarr array, read as needed; {api}`Cube.to_memory` reads every
lazy layer at once.

{api}`to_anndata` and {api}`to_mudata` convert a cube to the pseudobulk layouts of
scverse, and {api}`from_anndata` and {api}`from_mudata` read them back.

## Several studies

Results of one study often do not replicate in the next [@nakatsuka2025], and studies
are compared or combined only once their samples, genes and cell types are aligned.
Aligned by hand, they go wrong in two ways: donor ids repeat across studies, and genes
that a study did not measure are read as zeros. {api}`CubeCollection` aligns the cubes
of several studies and keeps each as it was given:

- **Samples** are named `study::sample_id`, and so are the values of the columns of ids
  that `within_study` names, such as donors: donor `D1` of one study is not donor `D1`
  of another.
- **Genes** are aligned by `feature_id`, such as Ensembl ids, and
  {api}`CubeCollection.measured` says which study measured which: a gene that a study's
  cube lacks, or marks as not measured in its feature column `measured`, is missing in
  that study, not zero.
- **Cell types** are aligned by their Cell Ontology ids [@diehl2016] when every cube
  has them, by group id otherwise, or by a mapping for each study. Groups of one study
  that share an id are merged, exactly, since every layer is a sum.

Each study is then analysed on its own: {api}`CubeCollection.aligned_cube` gives its
cube with the aligned cell types and the genes it measured, and with its own samples and
library sizes, and an analysis such as {api}`limma_voom` with the study's own design
gives results keyed by the aligned cell types. {api}`align_study_results` lines them up
gene by gene, and {api}`differential_expression_meta_analysis` combines them
([Meta-analysis](meta_analysis.md)). For a joint model with the study as a covariate,
{api}`CubeCollection.stack` puts the studies into one cube with the genes that every
study measured and the cell types that every study has: a cube holds no missing values,
so it refuses the others rather than fill them with zeros. The collection's provenance
names each study's cube by its fingerprint, with the alignment, and so does the
provenance of a stacked cube.
