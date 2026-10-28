---
description: What a cube of pseudobulk profiles holds, why each of its layers is a sum over cells, and what it records so that a result can be traced to its data.
---

# Pseudobulk profiles and cubes

[Choosing a method](choosing_a_method.md) lists the questions a study asks of its
samples. Every method that answers them starts from a cube, which holds, for every
sample, group (usually a cell type) and feature (usually a gene), sums over the sample's
cells of that group ([Why pseudobulk](../why_pseudobulk.ipynb) says why the tests
compare samples). {api}`from_cells` builds a cube from the counts of the cells and the
sample and group of each cell, and {api}`from_bulk` wraps a samples x features matrix of
bulk counts as a cube with one group. [Pool cells
into pseudobulk profiles](../guides/pool_cells.ipynb) builds one from a simulated study.

(cube-layers)=
## The layers

For sample $j$, group $g$ and feature $f$, let $C_{jg}$ be the cells of that sample and
group, $y_{cf}$ the count of cell $c$, and $L_c$ the cell's library size: its total
count, or a value given as `library_sizes`. Each cell's counts are scaled to a target
sum $T$, 10,000 by default, so that its normalized values are $x_{cf} = y_{cf} T / L_c$,
and the layers, with the number of cells of each profile, are

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

{api}`Cube.n_cells` gives the numbers of cells. Layers are stored as groups x samples x
features, so that one group is a samples x features matrix, the shape that the model of
one cell type is fitted to, and {api}`Cube.layer` returns a layer for one group or for
every group. The counts are the one layer that every cube needs, and the other layers
describe the cells. A cube from {api}`from_bulk` has counts only, and no cell numbers.

(exact-regrouping)=
## Exact regrouping

Every layer is a sum over cells, so the profile of a union of cells is the sum of the
profiles of its parts:

$$
\text{layer}(C_{jg} \cup C_{jh}) = \text{layer}(C_{jg}) + \text{layer}(C_{jh}).
$$

Merging two cell types into one, or pooling the samples of a donor, therefore needs no
cells, and gives the profiles that pooling the cells would have given.
{api}`Cube.regroup` merges groups, {api}`Cube.collapse_samples` pools the samples that
share the values of sample columns, such as the tissues of one donor,
{api}`Cube.with_columns` adds a column to pool by, and {api}`Cube.select` keeps some
samples, groups and features. A mean or a variance cannot be merged this way, which is
why a cube keeps sums and derives the rest from them. Integer layers are summed exactly,
in 64-bit integers.

(what-the-sums-give)=
## What the sums give

The sums give the statistics that describe the cells of each profile, with $n$ its
number of cells: {api}`Cube.mean`, the mean normalized value on the log or the linear
scale, $\text{log\_sum} / n$ or $\text{linear\_sum} / n$; {api}`Cube.variance`, the
sample variance of the log values,
$(\text{log\_sum\_sq} - \text{log\_sum}^2 / n) / (n - 1)$;
{api}`Cube.fraction_detected`, the share of the cells with a positive count,
$\text{detected} / n$; and {api}`Cube.library_sizes`, the counts of each profile summed
over the features.

(dot-plot)=
### The dot plot

A dot plot shows, for chosen genes and cell types, the fraction of the cells that
express each gene and their mean expression, as scanpy draws it [@wolf2018]. Because
every layer is a sum, the fraction and the mean over the cells of all the samples
pooled,

$$
f_{gf} = \frac{\sum_j \text{detected}_{gjf}}{\sum_j n_{jg}}, \qquad
m_{gf} = \frac{\sum_j \text{log\_sum}_{gjf}}{\sum_j n_{jg}},
$$

are those computed from the cells themselves, and so is the mean over only the cells
that express it, $\sum_j \text{log\_sum}_{gjf} / \sum_j \text{detected}_{gjf}$, since
$\log(1 + 0) = 0$ for the others. A cube therefore draws the dot plot without its cells:
{api}`plot_dotplot` pools every sample or some, and scales the means by gene or by cell
type and draws groups of genes, as scanpy does.

(what-a-cube-records)=
## What a cube records

Besides its layers, a cube keeps what is needed to trace a result to its data.

- **Tables.** Samples, groups and features each have a table ({api}`Table`), with a
  unique text id (`sample_id`, `group_id` and `feature_id`) and any other columns: each
  sample's donor and condition, each group's Cell Ontology id, by which the cell types
  of several studies are aligned ([Aligning studies](aligning_studies.md)), or each
  gene's symbol. A design takes its variables from the sample table, and is matched to
  the cube's samples by `sample_id` ([The design](linear_models.md#designs-and-formulas)).
- **A fingerprint.** {api}`Cube.fingerprint` is a SHA-256 digest [@nist2015] of
  everything an analysis reads (the tables, the cell numbers, the normalization of the
  normalized layers, and every layer) but not of the provenance: cubes that hold the
  same values share a fingerprint, whatever their history, and a result, such as those
  of {api}`limma_voom` and {api}`propeller`, names by it the cube it came from. The
  fingerprint is built from one digest for each table, for the cell numbers, for the
  normalization and for each group of each layer ({api}`Cube.digests`). A cube file
  stores the digests, so that one group can be checked when it is read, without reading
  the others, where a single digest of the whole cube would need every layer read to
  check any part of it. The encoding is specified byte by byte, so that another program
  can compute a fingerprint ([Fingerprints](../reference/cube_files.md#fingerprint-encoding)).
- **A provenance record.** {api}`Cube.provenance` says how the values were made, under
  three keys: `source`, where the data came from, as the caller recorded it;
  `construction`, the function that built the cube and its settings; and `operations`,
  one entry per later operation, oldest first, each naming its input by fingerprint
  when that is known without reading lazy layers ([Provenance
  records](../reference/cube_files.md#provenance-keys) says when). A chain of operations
  can then be checked link by link, back to the cube it started from.
  {api}`Cube.history` gives the record in plain terms, as text to print.

(profiles-from-other-tools)=
## Profiles pooled by other tools

scanpy's `get.aggregate` and decoupler's `pp.pseudobulk` pool cells into an AnnData
with one row per sample and cell type, and LIANA into a MuData with one modality per
cell type [@virshup2024; @bredikhin2022]. {api}`from_anndata` and {api}`from_mudata`
read these layouts into a cube, and {api}`to_anndata` and {api}`to_mudata` write a cube
in them, with a record that lets an import restore the same cube, down to its
fingerprint. Since a cube's layers are sums, an import takes only what is a sum or gives
one exactly: summed counts, the number of cells that detect each feature, and sums of
normalized values. Means, variances and medians are refused: a variance or a median
gives no sum, and a mean times the number of cells gives the sum exactly only where the
sum is a whole number, while the tools that give means give sums too. [Work with
AnnData and MuData](../guides/anndata_and_mudata.ipynb) converts cubes both ways.

(cubes-larger-than-memory)=
## Cubes larger than memory

{api}`write_cube` saves a cube as a Zarr v3 store [@miles2025], in chunks of one group
and up to 4,096 features, and {api}`read_cube` opens it without reading its layers: a
group is read when an analysis needs it, and of that group only the chunks of the
features it selects. The analyses of every cell type hold one cell type in memory at a
time, so they run on cubes larger than memory, also on a cube whose layers are other
array-likes read as needed, such as a numpy memmap or an HDF5 dataset ({api}`Cube`).
[Cube files and provenance records](../reference/cube_files.md) specifies the file.

(cubes-defaults)=
## Defaults and choices

- **Layers.** {api}`from_cells` computes all five. The tests need only the counts and
  the cell numbers, and the dot plot also `detected` and `log_sum`; a cube of counts
  alone takes an eighth of the memory of all five layers (`layers=["counts"]`).
- **Normalization of the cells.** Each cell is scaled to `target_sum`, 10,000, by its
  total count, and its log values are $\log(1 + x)$, as scanpy's `normalize_total` and
  `log1p` compute them. `library_sizes` gives other
  totals, such as those over every gene when the counts hold only some genes. Only the
  normalized layers depend on it; the counts, and the tests, do not.
- **Samples and groups.** By default the distinct labels, sorted. A sample table
  (`samples=`) or group table (`groups=`) sets their order, carries the variables of
  the design, and gives a sample without cells a profile of zeros.
- **Rare cell types.** A cell type with few cells in most samples gives noisy
  profiles; {api}`Cube.regroup` merges it into its lineage, exactly, as [Pool cells into
  pseudobulk profiles](../guides/pool_cells.ipynb) does.
- **Files.** {api}`write_cube` compresses with gzip at level 4 by default;
  `compression="zstd"` uses zstd, on Python 3.14 and newer, and `compression=None`
  stores the values as they are. {api}`Cube.to_memory` reads every lazy layer at once,
  for analyses that read the same groups many times.

(cubes-limits)=
## Limits and differences

- **Memory.** {api}`from_cells` needs the counts of the cells in memory, and returns
  dense layers: 32 bytes per sample, group and feature with all five layers.
- **Whole numbers.** Integer layers and cell numbers hold whole numbers up to
  $2^{63} - 1$, the largest signed 64-bit integer, and a sum beyond it raises an error
  instead of wrapping around. Normalization works in float64, which is exact for counts
  up to $2^{53}$.
- **Variances from sums.** {api}`Cube.variance` computes the variance from raw moments,
  which lose digits when the values of a profile are nearly constant and far from zero.
  Log-normalized expression is far from that case ([Validation](#validation-cube)
  gives the bound).
- **Profiles without cells.** The fraction detected of a profile without cells is NaN,
  zero cells out of zero, where decoupler reports 0.
- **Scattered reads.** A read decompresses every chunk that holds a selected feature.
  A few neighbouring genes are therefore quick to read, but genes scattered over a file
  decompress nearly every chunk of each group, and a few samples of every gene every
  chunk ([Timings](#validation-timings)).
- **What a read checks.** Reading a whole group of a cube file checks it against the
  group's digest; reading some samples or features of a group does not, and
  {api}`Cube.verify_fingerprint` checks those too. A cube in memory is hashed once, by
  the first operation applied to it ({api}`Cube.fingerprint` gives the cost).
- **Other array-likes.** A lazy layer of another array-like is read with its
  orthogonal indexing when it has one, as zarr-python's arrays do, and otherwise a whole
  group per read, and the cube copies what it returns. Such an array can change,
  so its fingerprint is computed anew on every call, and the operations applied to it
  record no input fingerprint.
- **Profiles filtered before an import.** Profiles that another tool removed after
  pooling, such as by a filter of samples with few cells, are missing from merged
  groups and pooled samples: import the profiles first, and leave such samples out in
  the analysis.

(cubes-checks)=
## Checks and further reading

Each part of a cube is compared with an independent implementation, run as a separate
program on the same data:

- The layers, merged groups and pooled samples, and the means, variances and fractions
  they give: scanpy 1.12.4, exact (identical counts, 1e-10 otherwise)
  ([Validation](#validation-cube)).
- {api}`from_cells` and {api}`from_bulk`: scanpy 1.12.4 and decoupler 2.2.0, exact
  ([Validation](#validation-pool-cells)).
- Cube files, and reads of selected features: zarr-python 3.4.1, identical, and 1e-12
  for merged groups and pooled samples that a read gives
  ([Validation](#validation-cube-files) and [Validation](#validation-cube-reads)).
- Fingerprints: a second implementation written from the specification, which reads
  cube files with zarr-python 3.4.1, identical digests
  ([Validation](#validation-cube-fingerprints)).
- Conversions to and from AnnData and MuData: scanpy 1.12.4, decoupler 2.2.0, LIANA
  1.10.0, anndata 0.13.4 and mudata 0.4.1, exact
  ([Validation](#validation-anndata-conversion)).
- The dot plot: scanpy 1.12.4, exact ([Validation](#validation-pca-and-dot-plots)).

Further reading:

- @wolf2018 introduced scanpy, the toolkit for single-cell analysis whose aggregation
  of cells and dot plot the cube is compared with.
- @virshup2024 described anndata, the annotated data matrices in which scverse tools
  keep cells and their pseudobulk profiles.
- @miles2025 specify version 3 of Zarr, the format of cube files, which any Zarr v3
  reader can open.

[Designs, linear models and contrasts](linear_models.md) turns the questions asked of
these profiles into one model for every gene.
