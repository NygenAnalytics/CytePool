---
myst:
  html_meta:
    description: The CytePool Python API for pseudobulk cubes, their conversion to and from AnnData and MuData, designs, normalization, linear models, mixed models, voom, dream, sample quality weights, negative binomial models, quasi-likelihood F-tests, DESeq2's dispersions and tests, empirical Bayes moderation, gene-set tests (including rotation tests), pathway scores, networks of regulators and pathways and their activities, tests of scores and activities with a design, multiple testing, differential expression, principal components of samples, cell-type composition, the results of several studies, meta-analysis, combined p-values and replication across studies, SumRank and held-out prediction, and distributions.
---

# CytePool API reference

Return to the [guides and validation](../).

`import cytepool` gives the functions and classes an analysis starts from, such as
`cytepool.limma_voom` and `cytepool.Cube`, and the pages below document them by topic.
The building blocks that work on arrays are in their modules: `cytepool.stats.voom`
and the other modules of `cytepool.stats`, `cytepool.normalization`, the parts of a
design in `cytepool.design`, and the figures in `cytepool.plotting`.

Arrays of counts and expression values are samples x features: rows are samples,
columns are features such as genes. Where an argument accepts missing values, NaN
marks them in numbers and None in text. numpy masked arrays, and array-likes that
convert to one, are refused with a `TypeError`, because numpy drops the mask when it
converts one and the masked entries would be read as data; the message names the
argument and says how to mark its missing values instead. The
[quick start](../quickstart/) shows how the pieces fit together in a
differential-expression analysis.

```{eval-rst}
.. py:data:: __version__
   :module: cytepool

   The version of CytePool that is installed, which every result's provenance
   records.
```

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
networks
activity
score_tests
multiple_testing
combined_p_values
differential_expression
variance_partitioning
principal_components
composition
meta_analysis
study_results
replication
sumrank
held_out_prediction
plotting
distributions
smoothing
```
