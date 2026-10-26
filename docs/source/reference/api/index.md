---
myst:
  html_meta:
    description: The CytePool Python API for pseudobulk cubes, their conversion to and from AnnData and MuData, designs, normalization, linear models, mixed models, voom, dream, sample quality weights, negative binomial models, quasi-likelihood F-tests, DESeq2's dispersions and tests, empirical Bayes moderation, gene-set tests (including rotation tests), pathway scores, multiple testing, differential expression, cell-type composition, meta-analysis, combined p-values and replication across studies and distributions.
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
mixed_models
voom
dream
sample_weights
negative_binomial_models
quasi_likelihood
deseq2
empirical_bayes
gene_sets
gene_set_tests
rotation_gene_set_tests
over_representation
gsea
pathway_scores
multiple_testing
combined_p_values
differential_expression
variance_partitioning
composition
meta_analysis
replication
plotting
distributions
smoothing
```
