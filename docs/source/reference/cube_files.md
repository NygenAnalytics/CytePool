---
description: The cube file format (a Zarr v3 group), the encoding of cube fingerprints, and the keys of provenance records, including what a tool that exports cubes should record.
---

# Cube files and provenance records

A cube file stores a cube so that any Zarr v3 reader can open it, and so that a reader
can tell whether it holds what was written. Its fingerprint names the cube's values, and
its provenance record says how the cube was made. This page describes all three, for
programs that read or write cubes; [Export a cube from another
tool](../guides/export_cubes.ipynb) shows the steps in code.

## The file

{api}`write_cube` writes a directory, by convention `<name>.cube.zarr`, holding a Zarr
v3 group [@miles2025]; {api}`read_cube` opens it. Format `cytepool.cube`, version 1:

```text
<name>.cube.zarr/
  zarr.json      the root group, with the attributes below
  samples/       one 1-D array per sample column
  groups/        one 1-D array per group column
  features/      one 1-D array per feature column
  n_cells        integers (n_samples, n_groups); absent from bulk cubes
  layers/
    counts       (n_groups, n_samples, n_features)
    ...          detected, log_sum, log_sum_sq and linear_sum, when present
```

| Root attribute | Value |
| --- | --- |
| `format` | `"cytepool.cube"` |
| `format_version` | `1`; a reader rejects other versions |
| `axes` | `["groups", "samples", "features"]`, the axes of every layer |
| `layers` | the layer names, in the order counts, detected, log_sum, log_sum_sq, linear_sum |
| `has_n_cells` | whether `n_cells` is present |
| `normalization` | how the normalized layers were computed, or null |
| `provenance` | the provenance record (below) |
| `digests` | the digests of the parts and the fingerprint (below) |

Other attributes are ignored, so a publisher may add, say, a title.

**Tables.** Each column is a 1-D array with `dimension_names` set to its axis. Text is
stored as int32 codes with the attribute `levels` (the distinct values in order of first
appearance; -1 marks a missing value); datetimes and timedeltas as int64 counts of their
unit; booleans and numbers as their own type. The group's attributes `columns` and
`arrays` give the column names and the array of each.

**Layers.** Chunks hold one group, every sample and up to 4,096 features, so one group,
or a few features of it, is read without the others. Every array uses the `bytes` codec
(little endian) followed by gzip (the default; @deutsch1996b), zstd (Python 3.14 or
newer; @collet2021) or no compression [@zarr2023; @maitinshepard2025]. Chunks of zeros
are not stored.

**Reading and writing.** Reading is strict: the format, every attribute, every array's
metadata and every value is checked, and the tables, cell numbers and normalization are
checked against their digests when the file is opened, and each group of a layer
whenever a read covers all of it. {api}`Cube.verify_fingerprint` checks everything.
Writing is atomic: the store is written to a hidden directory beside the target and
renamed into place when it is complete.

## Fingerprints

{api}`Cube.fingerprint` identifies the values an analysis reads: the sample, group and
feature tables, the cell numbers, the normalization and every layer. It leaves out the
provenance, so two cubes with the same values have the same fingerprint however they
were made, and a result can name the cube it was computed from.

It is built from digests, each the SHA-256 [@nist2015] of a byte string, written as
`sha256:` and 64 hexadecimal digits: one per table, one for the cell numbers, one for
the normalization, and one for each group of each layer, so that one group can be
checked without reading the others. Numbers are little endian; integer layers are hashed
as int64 and float layers as float64, with -0.0 written as 0.0. The fingerprint is the
digest of a text manifest that lists them:

```text
cytepool cube fingerprint 1
shape <groups> <samples> <features>
samples <digest>
groups <digest>
features <digest>
n_cells <digest, or none without cell numbers>
normalization <digest, or none without normalization>
layer <name> <int64 or float64> <group position> <digest>
```

with a `layer` line for every group of every layer. The `1` of the first line is the
version of the encoding ({api}`FINGERPRINT_VERSION`). {api}`Cube.digests` returns the
digests in the form the file stores them, and the module documentation of
`cytepool.cube.fingerprints` gives the encoding of every part byte by byte, so that
another program can compute the fingerprint. The encoding was checked against an
independent implementation ([Validation](../validation.md#cube-fingerprints)).

## Provenance records

{api}`Cube.provenance` is a JSON object with three keys that CytePool writes;
{api}`Cube.history` and {api}`describe_provenance` print it in plain terms.

| Key | What it records |
| --- | --- |
| `source` | Where the data came from, as the caller gives it to {api}`from_cells`, {api}`from_bulk`, {api}`from_anndata`, {api}`from_mudata` or `Cube(provenance={"source": ...})`. Stored as given and never interpreted; its references are reported as unresolved, since following them needs the tool that wrote them. |
| `construction` | How the cube was built: the function, the CytePool version and its settings, such as the layers computed, the target sum of the normalization and where the sample, group and feature ids came from. A cube that {api}`CubeCollection.stack` built records each study's cube by its fingerprint, with that cube's provenance and its groups under their aligned ids. |
| `operations` | Every operation that derived the cube, oldest first: `select`, `regroup` or `collapse_samples`, with what it kept, merged or pooled, and `input`, the fingerprint of the cube it was applied to. |

Because each operation names its input by fingerprint, a chain can be checked link by
link: apply each operation to the cube with its input fingerprint, and compare the
result's fingerprint with the next record's input. The results of an analysis, such as
{api}`limma_voom`'s, keep the fingerprint of the cube they were computed from, the
digests it is built from and its provenance record.

## What an exporter records

A tool that pools cells into a cube, such as CyteArc or another single-cell analysis
platform exporting its pseudobulk profiles, builds a {api}`Cube` from its sums, with
lazy layers if they are large, and passes its own record as `source`. CytePool keeps
whatever the record holds; these fields make a cube traceable to the cells it came from:

| Field | Example |
| --- | --- |
| the tool and its version | `{"tool": "example-exporter", "version": "0.1"}` |
| the dataset's identity | an accession or a dataset id, with the checksum or version of the data read |
| the frozen cell selection | a reference to the exporter's record of which cells entered each sample and group, such as a stored membership and its digest |
| the sample and group columns | which cell annotations defined the samples (for example donor and condition) and the groups (the cell-type annotation and its version) |
| the settings | filters, the counts layer used, the normalization of the summed values |

Cell memberships stay in the exporter's files; the cube records how to find them.
