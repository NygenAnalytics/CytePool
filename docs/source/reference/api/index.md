---
myst:
  html_meta:
    description: The CytePool Python API for pseudobulk cubes, their conversion to and from AnnData and MuData, designs, normalization, linear models, mixed models, voom, dream, sample quality weights, negative binomial models, quasi-likelihood F-tests, DESeq2's dispersions and tests, empirical Bayes moderation, gene-set tests (including rotation tests), pathway scores, networks of regulators and pathways and their activities, tests of scores and activities with a design, multiple testing, differential expression, principal components of samples, cell-type composition, the results of several studies, meta-analysis, combined p-values and replication across studies, SumRank and held-out prediction, and distributions.
---

# CytePool API reference

Return to the [guides and validation](../).

`import cytepool` exposes the main functions and classes, such as `cytepool.limma_voom`, `cytepool.dream_voom`, `cytepool.propeller`, `cytepool.camera` and `cytepool.Cube`:

- **Top-level workflows (`cytepool`):** pooling cells (`from_cells`, `from_bulk`), converting AnnData and MuData (`from_anndata`, `from_mudata`), designs (`Design`), differential expression (`limma_voom`, `limma_trend`, `dream_voom`), composition (`propeller`), variance partitioning (`variance_partitioning`, `principal_components`), pathways and activities (`camera`, `fry`, `roast`, `mroast`, `romer`, `gsea`, `over_representation`, `gsva`, `ssgsea`, `ulm`, `mlm`), and multi-study analysis (`CubeCollection`, `analyse_each_study`, `meta_analysis`, `sumrank`, `replication`, `held_out_prediction`).
- **Array building blocks (`cytepool.stats`, `cytepool.normalization`, `cytepool.design`):** linear and mixed models, empirical Bayes moderation, voom, sample quality weights, negative binomial and quasi-likelihood fits, DESeq2, TMM and size-factor normalization, formula parsing, and multiple testing.
- **Figures (`cytepool.plotting`):** volcano, MD, MDS, PCA, voom trend, dispersion, barcode, GSEA, dot plot, composition, variance fraction and forest plots (requires the `plot` extra).

See the [quick start](../quickstart/) for an end-to-end differential-expression example.

**Input conventions:** Count and expression arrays have shape `samples x features` (rows are samples, columns are features such as genes). Missing values are represented by `NaN` in numeric arrays and `None` in string arrays. numpy masked arrays are rejected with a `TypeError`, because converting them to standard arrays silently drops the mask; the error message names the argument and explains how to pass missing values instead.

```{eval-rst}
.. py:data:: __version__
   :module: cytepool

   The version of CytePool that is installed, which every result's provenance
   records.
```

```{toctree}
:caption: Cubes and files
:maxdepth: 1

cubes
cube_files
anndata
```

```{toctree}
:caption: Designs and linear models
:maxdepth: 1

designs
linear_models
empirical_bayes
multiple_testing
```

```{toctree}
:caption: Differential expression
:maxdepth: 1

differential_expression
normalization
principal_components
voom
sample_weights
negative_binomial_models
quasi_likelihood
deseq2
mixed_models
dream
variance_partitioning
```

```{toctree}
:caption: Composition
:maxdepth: 1

composition
```

```{toctree}
:caption: Gene sets and pathways
:maxdepth: 1

gene_sets
gene_set_tests
rotation_gene_set_tests
over_representation
gsea
pathway_scores
networks
activity
score_tests
```

```{toctree}
:caption: Several studies
:maxdepth: 1

study_results
meta_analysis
combined_p_values
replication
sumrank
held_out_prediction
```

```{toctree}
:caption: Figures
:maxdepth: 1

plotting
```

```{toctree}
:caption: Building blocks on arrays
:maxdepth: 1

distributions
smoothing
```
