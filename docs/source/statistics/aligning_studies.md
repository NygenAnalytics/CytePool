---
description: How the cubes of several studies are aligned by sample, gene and cell type before their results are compared, combined or modelled together.
---

# Aligning studies

[Activities from networks](activities.md) completed the analysis of one study: its genes,
cell types, pathways and regulators. A finding counts for more when other studies of the
same question reproduce it, and the chapters of this part compare and combine studies.
Studies can be compared only once their samples, genes and cell types are aligned, and
{api}`CubeCollection` aligns them, keeping each study's cube as it was given. [Align and
combine studies](../guides/combine_studies.ipynb) aligns six simulated cohorts and analyses
each.

(why-findings-fail-to-replicate)=
## Why findings fail to replicate

Results of one study often do not hold in others: of the genes called differentially
expressed in Alzheimer's disease in one dataset, over 85% were not reproduced in any of 16
others [@nakatsuka2025]. Some of these are changes that the other studies were too small
to find; others belong to one study, to its donors, its protocol or a confounder of its own.
Telling the two apart takes several studies analysed alike.

Aligned by hand, studies go wrong in two ways: donor ids repeat across studies, so that
donor `D1` of one study is taken for donor `D1` of another, and genes that a study did not
measure are read as zeros. Each study also names its cell types in its own words.

(aligning-samples)=
## Samples and donors

A sample id means something only inside its study, so {api}`CubeCollection` names every
sample `study::sample_id`, and does the same for the columns named in `within_study`, such
as donor ids: donor `D1` of two studies is then two people, not one, in any design built
on the collection. The collection's sample table ({api}`CubeCollection.samples`) joins the
studies' sample tables, with missing values where a study lacks a column; the class's
documentation says how the types of a column are combined, and a column of text in one
study and of numbers in another is refused.

(aligning-genes)=
## Genes

Genes are matched by `feature_id` exactly as written, so every study must use ids of one
kind, such as Ensembl gene ids, without versions: `ENSG00000141510.17` and
`ENSG00000141510` differ. Some tools give every study the same genes and fill the genes
that a study lacks with zeros, so a cube can mark such genes False in a feature column
`measured`: a study measured the genes that its cube has and does not mark so. The
collection keeps every gene that some study measured, and {api}`CubeCollection.measured`
records which study measured which, so that an unmeasured gene is missing for that study
rather than zero.

A missing gene changes nothing for the other genes. A gene without counts changes no
library size, expression filter or normalization factor, so each study's results on the
genes it measured are those of its cube as given.

(aligning-cell-types)=
## Cell types

Cell types are aligned by their Cell Ontology ids [@diehl2016], the column
`cell_ontology_id` of the group tables, when every study's cube has it, and by
`group_id` otherwise; `align_groups_by` names another column. `group_mapping` gives some
groups of a study an aligned id of their own, or None to leave a group out, and
`group_names` names the aligned groups for the tables and figures that follow, such as
CD14 monocytes for CL:0001054. Two groups of one study that receive the same aligned id are
summed into one, exactly, since every layer of a cube is a sum ([Exact
regrouping](cubes.md#exact-regrouping)): naive and memory CD4 T cells of one study,
mapped to CL:0000624, become the CD4 T cells of the others.
{api}`CubeCollection.group_members` shows which of a study's groups went into each aligned
group.

(one-analysis-per-study)=
## One analysis per study

Each study is analysed on its own. {api}`CubeCollection.aligned_cube` returns a study's
cube in the collection's terms, with its cell types renamed and merged as aligned and only
the genes it measured, but with the study's own samples and sample ids, so that a design
built from the study's sample table fits it. {api}`analyse_each_study` builds that design
for every study, with one formula, its levels and contrasts, and runs {api}`limma_voom`,
{api}`limma_trend` or {api}`dream_voom` on each aligned cube. Each study keeps its own
design, normalization, expression filter and variances.

The {api}`StudyResults` it returns keeps the results with their collection.
{api}`StudyResults.table` lines up the results of one cell type and contrast gene by gene,
with a row for every study and gene, and a status that gives the reason when a study lacks
a result for a gene: it did not measure the gene (`not_measured`), measured it and left it
out, as its expression filter does (`not_tested`), or did not analyse the cell type or the
contrast (`not_analysed`). These are different findings, and a missing result is never
read as a zero. {api}`StudyResults.matrix` puts one column of every study, such as the
p-values, into a studies x genes matrix. The [meta-analysis](meta_analysis.md) and the
[replication](replication.md) of the next chapters take these results.

(stacked-studies)=
## A joint model of stacked studies

{api}`CubeCollection.stack` builds one cube of all the studies' samples for a joint model,
whose sample table names each sample's study. A cube holds no missing values, so the
stacked cube keeps the genes that every study measured and the cell types that every study
has, and refuses the others rather than fill them with zeros. Each study keeps its own
counts and library sizes, and a normalized layer is stacked only from studies normalized
alike. The design `~ study + condition` gives every gene a baseline in each study and one
difference between the conditions, shared by all; with several samples per donor,
`within_study` keeps the donors of different studies apart.

A joint model suits studies made alike, with one protocol and platform, that are too small
to estimate their variances alone. It estimates one difference for every study, so a change
in one study enters the estimate for all, and it tests only the genes that every study
measured.

(aligning-studies-defaults)=
## Defaults and choices

- **Cell types.** Aligned by `cell_ontology_id` when every study has it, by `group_id`
  otherwise, or by the column `align_groups_by` names; `group_mapping` and `group_names`
  map and name them.
- **Ids of donors and other columns.** `within_study` names the sample columns, such as
  donor ids, that are local to each study.
- **The analysis of each study.** {api}`limma_voom` by default in {api}`analyse_each_study`;
  its other options, such as `groups` or `expression_filter`, apply to every study.
- **Combine results or stack studies.** A meta-analysis of each study's results, the
  default, needs only the results, keeps the genes that some studies did not measure, and
  measures how far the studies disagree ([Meta-analysis](meta_analysis.md)); a joint model of
  the stacked studies suits studies made alike, as above.

(aligning-studies-limits)=
## Limits and differences

- **Equal ids only.** Cell types are aligned by equal values of a column. Rolling them up
  to a common ancestor of the Cell Ontology needs the ontology as a resource and is not
  implemented: give such merges in `group_mapping` ([Not implemented
  yet](#not-implemented-yet)).
- **Gene ids as given.** Ids are not converted between kinds or versions.
- **A stack in memory.** {api}`CubeCollection.stack` returns a cube in memory, of the genes
  and cell types common to every study; of a lazy study cube, only those are read. The
  collection itself reads no layer.

(aligning-studies-checks)=
## Checks and further reading

The alignment of three simulated studies, with repeated donor ids, merged cell types,
genes filled with zeros and marked as not measured, is compared with anndata 0.13.4's
`concat` and pandas 3.1.0, and limma-voom on each aligned study and on the stack with limma
3.62.1 and edgeR 4.4.0 run on the studies' own counts: exact, with identical samples, genes
and layers, and limma's tables within 3.8e-13 ([Validation](#validation-cube-collection)).
A control shows why donor ids are written with their study: taken for one person, donor
`D1` of two studies changes the fit of every gene.

Further reading:

- @diehl2016 described the Cell Ontology, whose ids name cell types alike across
  studies.
- @czi2025 described CZ CELLxGENE Discover, a curated corpus of over 93 million unique
  cells with standardized metadata.
- @sikkema2023 integrated 49 datasets into the Human Lung Cell Atlas, over 2.4 million
  cells from 486 individuals.

[Meta-analysis](meta_analysis.md), the next chapter, combines the results of the
studies.
