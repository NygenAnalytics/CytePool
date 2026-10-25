---
myst:
  html_meta:
    description: The CytePool Python API for pseudobulk cubes, their conversion to and from AnnData and MuData, designs, normalization, linear models, voom, sample quality weights, empirical Bayes moderation, multiple testing, differential expression and distributions.
---

# CytePool API reference

Return to the [guides and validation](../).

Arrays of counts and expression values are samples x features: rows are samples,
columns are features such as genes. Where an argument accepts missing values, NaN
marks them in numbers and None in text. numpy masked arrays, and array-likes that
convert to one, are refused with a `TypeError`, because numpy drops the mask when it
converts one and the masked entries would be read as data; the message names the
argument and says how to mark its missing values instead. The
[quick start](../quickstart/) shows how the pieces fit together in a
differential-expression analysis.

```{toctree}
:maxdepth: 1

cubes
cube_files
anndata
designs
normalization
linear_models
voom
sample_weights
empirical_bayes
multiple_testing
differential_expression
distributions
smoothing
```
