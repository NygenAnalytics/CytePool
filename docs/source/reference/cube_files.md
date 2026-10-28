---
description: The specification of cube files (a Zarr v3 group), of the encoding of cube fingerprints, of provenance records and of the record an AnnData or MuData export keeps, for programs that read or write cubes.
---

# Cube files and provenance records

A cube file stores a cube so that any Zarr v3 reader can open it, and so that a reader can tell whether it holds what was written. Its fingerprint names the cube's values, and its provenance record says how the cube was made. This page specifies all three, and the record that an AnnData or MuData export keeps, for programs that read or write cubes. [Export a cube from another tool](../guides/export_cubes.ipynb) shows the steps in code, and [Save, share and trace cubes](../guides/save_and_trace_cubes.ipynb) what a reader can check with them.

(cube-file-format)=
## The file

{api}`write_cube` writes a directory, by convention `<name>.cube.zarr`, holding a Zarr v3 group as the Zarr core specification defines it [@miles2025], which any Zarr v3 reader can open; {api}`read_cube` opens it. Below, a "Zarr group" is a node of the store, and a "group" one of the cube's groups, usually cell types. Format `cytepool.cube`, version 1:

```text
<name>.cube.zarr/
  zarr.json      the root Zarr group; its attributes are listed below
  samples/       a Zarr group with one 1-D array per sample column
  groups/        a Zarr group with one 1-D array per group column
  features/      a Zarr group with one 1-D array per feature column
  n_cells        integers (n_samples, n_groups); absent from bulk cubes
  layers/        a Zarr group with one array per layer
    counts       (n_groups, n_samples, n_features)
    ...          detected, log_sum, log_sum_sq and linear_sum, when present
```

| Root attribute | Value |
| --- | --- |
| `format` | `"cytepool.cube"` |
| `format_version` | `1`. A reader rejects other versions; any change to the layout increments the version |
| `axes` | `["groups", "samples", "features"]`, the axes of every layer, in order |
| `layers` | the layer names, in the order `counts`, `detected`, `log_sum`, `log_sum_sq`, `linear_sum` |
| `has_n_cells` | whether the `n_cells` array is present: false for bulk cubes, where cell numbers are unknown |
| `normalization` | how the normalized layers were computed, as a mapping, or null |
| `provenance` | the provenance record ([below](#provenance-keys)) |
| `digests` | the digests of the cube's tables, cell numbers, normalization and of each group of each layer, and the fingerprint they make ([below](#digest-record)). {api}`write_cube` computes them while it writes the layers. A file written before CytePool stored digests has none; it is read all the same, and its fingerprint is computed from its values when it is asked for |

Other attributes are ignored, so a publisher may add, say, a title.

(cube-file-tables)=
### Tables

Each table's Zarr group holds one 1-D array per column, with `dimension_names` set to the table's axis (`["samples"]`, `["groups"]` or `["features"]`), in chunks of up to 2²⁰ values. Its attributes give the columns in table order: `columns` lists the column names and `arrays` the name of each column's array. An array takes its column's name when that name is portable: ASCII letters, digits, `_`, `.` and `-`, at most 64 characters, starting with a letter, a digit or `_` but not with `__`, not ending with `.`, without a Windows device name before its first `.` (`con`, `prn`, `aux`, `nul`, `com0` to `com9`, `lpt0` to `lpt9`, in any case), not `zarr.json` in any case, not of the form `column_<n>` in any case, and not equal to another column's name when case is ignored. Otherwise the array is named `column_<position>`, counting from 0. Columns are stored as:

- text: int32 codes with the attribute `levels`, the distinct values in order of first appearance; code `i` stands for `levels[i]` and `-1` for a missing value;
- datetimes and timedeltas: int64 counts of their unit, with the attribute `numpy_dtype` (for example `"datetime64[s]"`); NaT is -2⁶³, as in numpy;
- booleans and numbers: their own data type.

The sample table has a column `sample_id` of unique, non-empty text ids; the group table a column `group_id` of such ids, and may have the text columns `cell_ontology_id` and `parent`; the feature table a column `feature_id` of such ids, and may have a text column `symbol` and a boolean column `measured`.

(cube-file-layers)=
### Layers and cell numbers

Layers have `dimension_names` `["groups", "samples", "features"]` and chunks of one group, every sample and up to 4,096 features, so that reading one group reads only its own chunks, and reading some features of it only the chunks that hold them. A chunk holds fewer samples only where it would otherwise exceed 256 MiB (2²⁸ bytes): it then holds as many samples as fit, 2²⁸ bytes divided by the bytes of its features, so 8,192 samples of 4,096 float64 values. `counts` is of an integer type or float64 (as for counts that are not whole numbers); `detected` is of an integer type; `log_sum`, `log_sum_sq` and `linear_sum` are float64.

`n_cells` is int32, or int64 if a value does not fit, with `dimension_names` `["samples", "groups"]`, in chunks of every group and up to 2²⁰ values.

Every array has the default chunk key encoding with `/`, a fill value of zero (`false` for bool) and the `bytes` codec, little endian (without an `endian` for types of one byte), followed by `gzip` (the default, at level 4 unless `compression_level` gives another), `zstd` (at level 3 unless given another, with checksums; Python 3.14 or newer) or no compression: the `bytes` and `gzip` codecs of the Zarr specifications [@zarr2023] and the `zstd` codec of its extensions [@maitinshepard2025], which store gzip streams [@deutsch1996b] and Zstandard frames [@collet2021]. Chunks that hold only zeros are not stored, as Zarr allows, and read as zeros. The part of Zarr v3 that CytePool reads and writes is listed in the API reference of {api}`cytepool.io.zarr`; zarr-python reads every cube file.

(cube-file-reading-and-writing)=
### Reading and writing

Reading is strict: the format name and version, every attribute above, every node's metadata, dimension names, data types and text codes are checked, and so are the values of every layer group when the cube reads it. The tables, cell numbers and normalization are checked against their stored digests when the file is opened, and each group of a layer whenever a read covers all of it, in any order, so a part changed after the file was written is reported when it is read whole, not read as if it were the cube the file held. A read of some samples or features of a group is not checked against the digest; {api}`Cube.verify_fingerprint` checks every group, also those. Digests describe values, not their encoding: a copy with other chunks, byte order or compression keeps them. A store replaced while {api}`read_cube` reads it raises an error instead of mixing the tables of one cube with the layers of another, and so does a lazy layer whose file was replaced after it was opened.

Writing is atomic: {api}`write_cube` writes the store into a new hidden directory next to the target, `.<directory>.partial-<random>` (the target's name and 8 hexadecimal digits, such as `.pbmc.cube.zarr.partial-3f9a0c21`), writes the root `zarr.json` last, and renames the directory to the target only when everything is written. A write that fails or is interrupted removes the partial directory and leaves the target as it was. With `overwrite=True`, which replaces only a cube file or an empty directory, the old store is renamed aside to `.<directory>.replaced-<random>`, the new one renamed into place and the old one deleted; if the new store cannot be moved into place, the old one is put back. A process killed outright can leave either hidden directory behind (between the two renames, the old store survives only under its hidden name). A partial directory has no root `zarr.json` until every array is written, so it does not open as a Zarr group; a process killed after that and before the rename leaves a complete store under the hidden name. Files are not flushed to disk with fsync, so the guarantee covers failures of the writing process, not of the operating system.

(fingerprint-encoding)=
## Fingerprints

{api}`Cube.fingerprint` identifies the values an analysis reads: the sample, group and feature tables (ids included), the cells pooled into each entry (`n_cells`), the normalization that defines the normalized layers, and every layer. It leaves out the provenance, which records how the cube was made, so two cubes with the same values have the same fingerprint however they were made, and a result can name the cube it was computed from by its fingerprint. {api}`Cube.digests` returns the digests it is built from.

The fingerprint is built from the digests of the cube's parts: each table, the cell numbers, the normalization, and each group of each layer, so that one group can be checked without reading the others. Encoding version 1 ({api}`FINGERPRINT_VERSION`) is as follows. The encoding was checked against an independent implementation written from this description ([Validation](#validation-cube-fingerprints)).

**Digests.** A digest is the SHA-256 hash [@nist2015] of a byte string, written as `sha256:` and 64 lowercase hexadecimal digits. Numbers in byte strings are little endian: `u64` and `i64` are unsigned and signed 8-byte integers. Text is UTF-8 [@yergeau2003], written as its length in bytes (`u64`) followed by its bytes; a lone surrogate, which a Python string can hold but a cube file cannot, is written in its 3-byte form, as Python's `surrogatepass` error handler writes it. Strings are taken as they are, without Unicode normalization.

**Tables** (samples, groups and features): the number of rows and of columns (`u64` each), then for each column, in table order, its name (text), its type (text) and its values:

- text, type `text`: the length in bytes of each value (`i64`, -1 for a missing value), then the bytes of the values present, one after another;
- `bool`: one byte per value, 0 or 1;
- `int8` to `int64` and `uint8` to `uint64`: the values in that type;
- `float16`, `float32` and `float64`: the values in that type, the binary formats of IEEE 754 [@ieee2019], with -0.0 written as 0.0 and every NaN as the quiet NaN (`0x7E00`, `0x7FC00000` or `0x7FF8000000000000`);
- `complex64` and `complex128`: the real and imaginary parts of each value, as floats of half the size written as above; a value with a NaN part is missing, and both its parts are written as the quiet NaN;
- datetimes and timedeltas, of type `datetime64[unit]` or `timedelta64[unit]` (numpy's name, such as `datetime64[ms]`): the values as `i64`, NaT being -2⁶³.

The type is numpy's name for the column's dtype. These are the column types cube files store, and the only ones a cube's tables accept. Two tables have the same digest exactly when {api}`Table.equals` finds them equal: the same names in the same order, the same types, and equal values, with missing values equal.

**Cell numbers:** the number of samples and of groups (`u64` each), then `n_cells` as `i64`, sample by sample.

**Normalization:** the mapping as a JSON value [@bray2017], each value written as a tag byte and a payload: `n` for null; `f` and `t` for false and true; `i` and the decimal digits of the number as text (with a leading `-` when negative) for a number with a whole value, such as 10000 or 10000.0; `d` and its 8 bytes as a float64 for any other number; `s` and the text for a string; `a`, the number of items (`u64`) and the items for an array; `o`, the number of members (`u64`) and, in the order of the UTF-8 bytes of their keys, each key (text) and value for an object. A number is thus taken by its value, however a JSON writer spells it.

**Layers:** one digest for each group of each layer, of the group's samples x features block: the number of samples and of features (`u64` each), then the values sample by sample, as `i64` for an integer layer and as float64, with -0.0 written as 0.0, for a float layer. An integer layer holds the same numbers whatever the width of its dtype (a cube accepts integers from 0 to 2⁶³ - 1, and merging groups or pooling samples makes them int64), so a uint32 group and an int64 group with the same values have the same digest.

**Manifest:** UTF-8 text with one line per part, each ending with a line feed (`\n`):

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

with a `layer` line for each group of each layer: layers in the order counts, detected, log_sum, log_sum_sq, linear_sum, and groups in the cube's order, counted from 0. The fingerprint is the digest of the manifest.

(digest-record)=
### The digest record

{api}`Cube.digests` returns, and a cube file stores as its root attribute `digests`, a JSON object with everything the manifest lists (the order of its members does not matter):

```text
{"version": 1, "shape": [<groups>, <samples>, <features>],
 "samples": <digest>, "groups": <digest>, "features": <digest>,
 "n_cells": <digest or null>, "normalization": <digest or null>,
 "layers": {<name>: {"type": "int64" or "float64",
                     "groups": [<digest of each group>]}},
 "fingerprint": <digest of the manifest>}
```

(provenance-keys)=
## Provenance records

{api}`Cube.provenance` is a JSON object whose keys below are written by the functions and methods that build and derive cubes. Any other key that a caller gives (`Cube(provenance=...)`) is kept as it is. {api}`Cube.history` and {api}`describe_provenance` give a record in plain terms.

**`source`:** the caller's record of where the data came from, given to {api}`from_cells`, {api}`from_bulk`, {api}`from_anndata` or {api}`from_mudata` as `source`: for example a dataset identity, references to the exporter's artifacts (such as frozen cell memberships, which stay in the exporter's files) and the exporter's settings. CytePool stores it as JSON, as it was given, does not interpret it, and reports its references as unresolved: following them needs the tool that wrote them. Null when no record was given.

**`construction`:** how the cube was built: `function` (`"from_cells"`, `"from_bulk"`, `"from_anndata"`, `"from_mudata"` or `"CubeCollection.stack"`), `cytepool_version`, and the function's settings. {api}`from_cells` records:

- `n_cells`: the number of cells pooled (the cells of each sample and group are the cube's `n_cells`);
- `layers`: the layers computed;
- `target_sum`: the total that normalization scales every cell to;
- `library_sizes`: `"cell_totals"` (the total count of each cell) or `"given"` (the `library_sizes` argument), as in the normalization;
- `samples` and `groups`: `{"order": ..., "labels": ...}`, where `order` is `"sample_table"` or `"group_table"` (the rows of the sample or group table given) or `"sorted_labels"` (the distinct labels, text sorted by code point and integers numerically), and `labels` is `"text"` or `"integers"` (integer labels became their decimal text);
- `features`: where the feature ids came from: `"table"`, `"ids"` (a sequence), `"data_frame_columns"` or `"positions"` (`feature_0`, `feature_1`, ...).

{api}`from_bulk` records `samples` and `features`, where their ids came from: `"table"`, `"ids"`, `"data_frame_index"` (samples) or `"data_frame_columns"` (features), or `"positions"`.

{api}`from_anndata` and {api}`from_mudata`, reading data that CytePool did not export, record:

- `layout`: the layout that CytePool recognized in the data, which chose the sources that `layers` did not name: `"scanpy.get.aggregate"`, `"decoupler.pp.pseudobulk"` or `"liana.multi.adata_to_views"`, or `"given"` for data of no layout it recognizes;
- `rows`: the number of profiles read, and for {api}`from_mudata` `modalities`, the modality names;
- `layers`: each cube layer with its source, `"X"` or a layer name;
- `n_cells`: where the cell numbers came from: an `obs` column, `"table"` (a table of samples x groups), `"uns['psbulk_stats']"` (LIANA's statistics) or null (none);
- `samples` and `groups`, as {api}`from_cells` records them;
- `obs_columns`: the metadata read into the tables, null when the tables were given. {api}`from_anndata` gives the `obs` columns placed in the sample table and in the group table, and those left out, as `{"samples": [...], "groups": [...], "left_out": [...]}`, and records the label columns as `sample_column` and `group_column`; {api}`from_mudata` lists the columns of the MuData's `obs` placed in the sample table.

{api}`CubeCollection.stack`, which stacks the cubes of several studies, records:

- `studies`: for each study, by its name and in the order stacked, the cube that the collection was given, by its `fingerprint`, with that cube's own `provenance`, and `groups`: each group stacked, by its aligned id, with the ids of the study's groups that it sums;
- `align_groups_by`: the group column whose values aligned the groups;
- `group_names`, when the collection was given them: each aligned id with the name that became its group's id;
- `within_study`: the sample columns whose values were written `study::value`, as the sample ids are;
- `layers`: the layers stacked.

**`operations`:** the operations that derived the cube, oldest first. Each is an object with `operation`, the method, and `input`, the fingerprint of the cube it was applied to, so that a chain of operations can be checked link by link: apply each operation to the cube with its input fingerprint, and compare the result's fingerprint with the next record's input, or with the cube's own fingerprint after the last.

- `select`: `samples`, `groups` and `features`, the ids kept, in the order of the result, or null for an axis that was given as None.
- `regroup`: `membership`, the id of each merged or renamed group with the ids of the groups it sums, in the input's order. The groups not listed are kept as they are.
- `collapse_samples`: `by`, the sample column, a list of them, or null when every sample was pooled into one, and `membership`, the id of each new sample with the ids of the samples it sums, in the input's order.
- `with_columns`: `samples`, `groups` and `features`, each with the names of the columns `added` and those `replaced` on that axis, or null for an axis that was given as None. The values are those of the cube's tables, which its fingerprint covers.
- `from_anndata` and `from_mudata`: the import of an AnnData or MuData that CytePool exported, whose provenance the cube continues. `input` is the fingerprint that the export stored, which the cube's own fingerprint equals unless the data were changed in between, and `source` the caller's record, or null.

`input` is recorded when it is known without reading lazy layer values. That is so for a cube in memory: the first operation applied to it hashes its layers, once. It is so for a cube read from a cube file, which stores its digests, and for a lazy cube derived from one when every digest it needs is known: kept unchanged from the file, or kept from whole reads of its groups, as {api}`Cube.fingerprint` makes them. It is null for a lazy cube derived from another whose groups were not read whole, such as a cube file regrouped and then collapsed: call {api}`Cube.fingerprint` on the intermediate cube before deriving from it to record its fingerprint, or check the operations after the last recorded fingerprint together. It is always null for a cube whose lazy layers are other array-likes (a memmap or an HDF5 dataset), which can change: their fingerprint is computed anew on every call and never kept.

The results of an analysis, such as {api}`limma_voom`'s, keep the fingerprint of the cube they were computed from, the digests it is built from and its provenance record.

(anndata-record)=
## The record of an AnnData or MuData export

{api}`to_anndata` and {api}`to_mudata` keep a record in `uns["cytepool"]` (of the AnnData, or of the MuData), from which {api}`from_anndata` and {api}`from_mudata` give back the cube, with the same fingerprint unless the data were changed in between. Only text and integers are stored, which every anndata writer keeps:

| Key | Value |
| --- | --- |
| `format` | `"cytepool.cube.anndata"` or `"cytepool.cube.mudata"` |
| `format_version` | `1` |
| `fingerprint` | the cube's fingerprint |
| `provenance` | the provenance record, as JSON text |
| `normalization` | the normalization, as JSON text |
| `layers` | JSON text: each layer's name with its numpy type, such as `{"counts": "uint32"}` |
| `n_cells` | JSON text: whether the cube has cell numbers |
| `samples`, `groups` | JSON text: the whole sample or group table, so that samples and groups without a row are kept, as `{"columns": [{"name": ..., "type": ..., "values": [...]}]}`; the type is `text` or numpy's name, datetimes and timedeltas are int64 counts of their unit, NaN and infinities are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`, and complex values pairs of numbers |
| `features` | JSON text: for each column of the feature table, its `name`, the `var` column that holds it (`column`, null for `feature_id`, the index of `var`) and its `type` |

(what-an-exporter-records)=
## What an exporter records

A tool that pools cells into a cube, such as CyteArc or another single-cell analysis platform exporting its pseudobulk profiles, builds a {api}`Cube` from its sums, with lazy layers if they are large, and passes its own record as the `source` of the provenance, `Cube(..., provenance={"source": record})`. CytePool keeps whatever the record holds; these fields make a cube traceable to the cells it came from:

| Field | Example |
| --- | --- |
| the tool and its version | `{"tool": "example-exporter", "version": "0.1"}` |
| the dataset's identity | an accession or a dataset id, with the checksum or version of the data read |
| the frozen cell selection | a reference to the exporter's record of which cells entered each sample and group, such as a stored membership and its digest |
| the sample and group columns | which cell annotations defined the samples (for example donor and condition) and the groups (the cell-type annotation and its version) |
| the settings | filters, the counts layer used, the normalization of the summed values |

Cell memberships stay in the exporter's files; the cube records how to find them.
