---
description: Which CytePool function answers each question about samples, the default and when to choose another, and how the chapters that explain the methods follow from one another.
---

# Choosing a method

Use the table below to pick a starting function, see its R or Python equivalent, and find when to switch to an alternative. Every method operates on sample-level profiles rather than individual cells ([Why pseudobulk](../why_pseudobulk.ipynb)), starting from a cube built with {api}`from_cells` or {api}`from_bulk`.

| Question | Start with | R and Python equivalents | Choose instead | Chapter and guide |
| --- | --- | --- | --- | --- |
| Which genes change in each cell type? | {api}`limma_voom` | limma `voom`, `lmFit`, `eBayes`, `topTable`; edgeR `filterByExpr`, `calcNormFactors` | {api}`limma_trend` when library sizes lie within threefold; {api}`dream_voom` for repeated measures | [limma-voom and limma-trend](limma_voom.md), [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) |
| Which genes change in bulk RNA-seq? | {api}`limma_voom` on {api}`from_bulk` | edgeR `glmQLFit`, `glmQLFTest`; DESeq2 `DESeq`, `results`; limma `treat` | {api}`fit_quasi_likelihood` + {api}`quasi_likelihood_f_test`, or {api}`fit_deseq2` + {api}`deseq2_wald_test` / {api}`deseq2_likelihood_ratio_test` | [edgeR and DESeq2](count_models.md), [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb) |
| Which treatments or doses differ from the control, with one error rate per gene? | {api}`multiple_contrast_test` with {api}`dunnett_contrasts` | multcomp `glht`, `contrMat`; mvtnorm `pmvt`, `qmvt` | {api}`williams_contrasts` when the response grows in dose order, two-sided for genes; `f_tests` for any change without saying which | [Multiple testing](multiple_testing.md#correlated-tests), [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb) |
| Which genes change, with several samples of each donor? | {api}`dream_voom` | variancePartition `dream`, `voomWithDreamWeights`; lme4 `lmer`, lmerTest, pbkrtest | {api}`limma_voom` with `~ donor + condition` for within-donor questions, or on {api}`Cube.collapse_samples` between few donors | [Mixed models and dream](mixed_models.md), [Repeated measures and donors](../guides/repeated_measures.ipynb) |
| What drives the variation between samples? | {api}`variance_partitioning` | variancePartition `fitExtractVarPartModel`; DESeq2 `plotPCA`; limma `plotMDS` | {api}`principal_components`, {api}`plot_pca` and {api}`plot_mds` to inspect samples first | [Mixed models and dream](mixed_models.md), [Normalization and filtering](normalization_and_filtering.md) |
| Which cell types expand or contract? | {api}`propeller` | speckle `propeller` | select one lineage with `Cube.select` to test proportions within that lineage; {api}`plot_composition` | [Composition tests](composition.md), [Cell-type composition](../guides/composition.ipynb) |
| Does a dose series support tests in each cell type? | {api}`check_dose_design` | limma `nonEstimable`, `is.fullrank`; R `qr()` | {api}`Design.check` for the problems of any design | [Dose response](dose_response.md) |
| Which genes respond to dose in each cell type? | {api}`dose_response` | limma `lmFit` on the dose groups with `contrasts.fit` and `topTable`; DRomics `itemselect`; dreamlet; muscat `pbDS` | `gate={"test": "dose_groups"}` for the dose groups alone, or `gate={"fold_change": 1.5}` for changes beyond a size, to decide which genes count as responding; {api}`dream_voom` as `analysis` for repeated measures; {api}`plot_dose_response` for the curves | [Dose response](dose_response.md), [Dose response per cell type](../guides/dose_response.ipynb) |
| Does a response follow one of a few expected shapes? | {api}`dose_response` with `tests="mcp"` and {api}`CandidateShapes` | DoseFinding `Mods`, `guesst`, `optContr`, `MCTtest` | `"spline"` for a response of any shape; `"williams"` for responses in dose order; `"dunnett"` for each dose against the vehicle | [Dose response](dose_response.md#dose-response-contrasts), [Dose response per cell type](../guides/dose_response.ipynb) |
| Could the genes a dose series calls come from noise? | {api}`permutation_check` | a loop of limma's `lmFit` and `eBayes` over permuted dose labels | a vehicle-only series placed in mock dose groups by its design, run through {api}`dose_response` and the check alike; `gate={"fold_change": 1.5}`, which passes fewer null genes, when only changes beyond a size matter | [Dose response](dose_response.md#dose-permutation-check), [Dose response per cell type](../guides/dose_response.ipynb) |
| What shape does each response take? | {api}`dose_response` with `models` | BMDS, pybmds or BMDExpress; DRomics `drcfit`; drc `drm` with `LL.4()` | {api}`fit_dose_curves` for values outside a cube; `models="ll4"` for log-logistic curves, `"dromics"` for responses that rise and then fall; `variance="weighted"` or `"moderated"` for the precision of the cell type's fit | [Dose response](dose_response.md#dose-response-model-curves), [Dose response per cell type](../guides/dose_response.ipynb) |
| From which dose does each gene respond? | {api}`dose_response` with `models` and `bmr` | BMDS, pybmds or BMDExpress; DRomics `bmdcalc`; limma's contrasts of each dose with stageR's `stageWiseAdjustment` | {api}`DoseResponseResult.lowest_effective_doses` for the lowest dose called at an error rate, without a curve; {api}`benchmark_doses` for curves outside a cube; `flags` for the checks of BMDS, NTP, ETAP and EFSA; an absolute `bmr` to compare cell types | [Potency and points of departure](potency.md), [Dose response](dose_response.md#lowest-effective-dose), [Dose response per cell type](../guides/dose_response.ipynb) |
| What would NTP's or ETAP's workflow find in each cell type? | {api}`dose_response` with `rules` | BMDExpress 2 with the documents' settings | `selection="bmds_2026"` for BMDS's recommended model; {api}`williams_trend_test` for Williams' trend test on any values | [Potency and points of departure](potency.md#rule-sets) |
| Does a signature respond to dose, in the direction expected? | {api}`dose_response` on the scores of {api}`gsva` or {api}`ulm`, with `tests=("mcp",)` and `direction` | GSVA `gsva`; limma `lmFit`, `eBayes`; DoseFinding `MCTtest`, `fitMod` | {api}`dose_response` on the cube for the genes; {api}`limma` for two conditions; `direction=None` for a change either way | [Dose response](dose_response.md#dose-response-signatures), [Dose response per cell type](../guides/dose_response.ipynb) |
| Do the genes of a pathway respond to a contrast? | {api}`camera` | limma `camera`, `fry`, `roast`, `mroast`, `romer`, `kegga` | {api}`fry`, {api}`roast` or {api}`mroast` for self-contained tests; {api}`romer`; {api}`over_representation` for gene lists | [Gene-set tests](gene_set_tests.md), [Gene sets and pathways](../guides/gene_sets.ipynb) |
| Do the genes of a pathway gather along a ranking? | {api}`gsea` | fgsea `fgseaMultilevel`, `fgseaSimple` | {api}`camera` when the linear model is available; {api}`plot_enrichment`, {api}`plot_gsea_table` | [Over-representation and GSEA](over_representation_and_gsea.md), [Gene sets and pathways](../guides/gene_sets.ipynb) |
| How active is a pathway in each sample? | {api}`gsva` | GSVA `gsva`, `ssgseaParam`, `zscoreParam`, `plageParam` | {api}`ssgsea`, {api}`combined_z_scores`, {api}`plage`; test scores with {api}`limma` | [Pathway scores per sample](pathway_scores.md), [Pathway and regulator activity](../guides/pathway_activity.ipynb) |
| Which regulators and pathways are active? | {api}`ulm` | decoupler `mt.ulm`, `mt.mlm`, `mt.waggr`; CollecTRI, PROGENy | {api}`mlm` for overlapping targets; {api}`weighted_aggregate` | [Activities from networks](activities.md), [Pathway and regulator activity](../guides/pathway_activity.ipynb) |
| What do several studies say together? | {api}`meta_analysis` | metafor `rma.uni`; metap | {api}`combine_p_values` when only p-values are comparable; {api}`CubeCollection.stack` for small uniform studies | [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md), [Align and combine studies](../guides/combine_studies.ipynb) |
| Which changes replicate across studies? | {api}`sumrank` | SumRank; qvalue `pi0est` | {api}`replication` and {api}`partial_conjunction`; {api}`held_out_prediction` | [Replication](replication.md), [Replication across studies](../guides/replication_across_studies.ipynb) |

## Which genes change in each cell type?

Use {api}`limma_voom` to test contrasts across every cell type in a cube using voom precision weights [@law2014]. To test several contrasts at once, such as every dose against the vehicle or the columns of a spline of dose, name them in `f_tests` ([F-tests of several contrasts](limma_voom.md#limma-f-tests)), and use {api}`multiple_contrast_test` to say which of them differ ([below](#contrasts-against-the-control)); {api}`DoseAxis` gives the columns of a spline of dose, with the vehicle in a group of its own ([Dose axes and the vehicle](dose_response.md#dose-axes)). Switch to:

- {api}`limma_trend` when library sizes vary by less than about threefold;
- {api}`dream_voom` when donors contribute multiple samples ([below](#several-samples-of-each-donor)).

See [limma-voom and limma-trend](limma_voom.md) for the methods, and [Genes that change in each cell type](../guides/genes_per_cell_type.ipynb) for a worked example.

## Which genes change in bulk RNA-seq?

Pass a bulk count matrix to {api}`from_bulk` and run {api}`limma_voom`, with `f_tests` to test several coefficients at once. Switch to:

- edgeR's quasi-likelihood F-tests ({api}`fit_quasi_likelihood` and {api}`quasi_likelihood_f_test`) or DESeq2 ({api}`fit_deseq2` and {api}`deseq2_wald_test`) to fit negative binomial models directly to counts [@chen2025; @love2014];
- {api}`quasi_likelihood_f_test` or {api}`deseq2_likelihood_ratio_test` to test multiple coefficients at once (such as all levels of a factor or time course);
- the step-by-step limma functions ({api}`voom`, {api}`fit_linear_model`, {api}`contrasts_fit`, {api}`moderated_statistics`), or {api}`treat` to test against a minimum log-fold-change threshold [@mccarthy2009].

edgeR and DESeq2 operate on a single count matrix (or one cell type extracted from a cube). See [edgeR and DESeq2](count_models.md) and [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb).

(contrasts-against-the-control)=
## Which treatments or doses differ from the control, with one error rate per gene?

Use {api}`multiple_contrast_test` with {api}`dunnett_contrasts` to test each treatment or dose against the control in every gene, from a moderated fit or from estimates and their covariances, with p-values adjusted for the gene's contrasts by the single-step max-t method [@hothorn2008]; the contrasts' correlation through the shared control makes it less conservative than Bonferroni's adjustment. Switch to:

- {api}`williams_contrasts` when the response grows in dose order [@bretz2006]: two-sided for genes, whose direction is not known, and one-sided (`alternative="greater"` or `"less"`) when it is known;
- `f_tests` of {api}`limma_voom` when any change matters and which one does not;
- {api}`dose_response` with `tests="dunnett"` or `"williams"` for a dose series, in every cell type of a cube ([Trends, doses against the vehicle and candidate shapes](dose_response.md#dose-response-contrasts)).

See [Several contrasts of one gene](multiple_testing.md#correlated-tests) and [Bulk RNA-seq](../guides/bulk_rna_seq.ipynb).

(several-samples-of-each-donor)=
## Which genes change, with several samples of each donor?

Use {api}`dream_voom` to fit a random intercept per donor, which lets you test both within-donor changes and between-donor variables such as disease or sex [@hoffman2021]. Switch to:

- {api}`limma_voom` with donor as a fixed effect (`~ donor + condition`) when all comparisons are within donor (such as stimulated versus control);
- {api}`limma_voom` on donor-collapsed profiles ({api}`Cube.collapse_samples`) when testing a between-donor variable with very few donors per group (where dream can inflate false positives at n = 3 per group).

See [Mixed models and dream](mixed_models.md) and [Repeated measures and donors](../guides/repeated_measures.ipynb).

## What drives the variation between samples?

Use {api}`variance_partitioning` to estimate the fraction of each gene's variance explained by each variable in the design [@hoffman2016]. Use {api}`principal_components` ({api}`plot_pca`, {api}`plot_mds`) to inspect sample separation first. See [Mixed models and dream](mixed_models.md), [Normalization and filtering](normalization_and_filtering.md), [Repeated measures and donors](../guides/repeated_measures.ipynb) and [Normalization, filtering and sample quality](../guides/normalization_and_quality.ipynb).

## Which cell types expand or contract?

Use {api}`propeller` ({api}`plot_composition`) to test whether cell-type proportions change between conditions [@phipson2022]. See [Composition tests](composition.md) and [Cell-type composition](../guides/composition.ipynb).

## Does a dose series support tests in each cell type?

Use {api}`check_dose_design` before fitting a dose series: it reports, for every cell type, the samples each dose group keeps after the filter of {api}`limma_voom`, the dose groups that the design's other terms or a processing batch make impossible to compare with the vehicle, and whether the cell type supports tests, only curves that describe the data, or no dose analysis [@lun2017]. With `rules="etap_2024"` it lists the design minimums of EPA's Transcriptomic Assessment Products, each met or not [@usepa2024]. {api}`Design.check` finds the problems of any design. See [Dose response](dose_response.md).

## Which genes respond to dose in each cell type?

Use {api}`dose_response` on a dose series: in every cell type that supports tests, it tests whether each gene responds along a natural spline of dose, the default gate that decides which genes count as responding, whether the response bends away from a straight line, its linear trend along the dose axis, and whether any dose group differs from the vehicle, as contrasts of one moderated fit of the dose groups [@law2014], with FDR within and across cell types and the fitted curve with a 95% band ({api}`plot_dose_response`). It runs {api}`check_dose_design` first. Switch to:

- `analysis=dream_voom` with random intercepts in the design for several samples of each donor or animal;
- `gate={"test": "dose_groups"}` for the dose groups alone, the moderated counterpart of the ANOVA that the National Toxicology Program runs as a gate of the data set [@ntp2018], with `tests="dose_groups"` to leave out the other tests; in series where no gene responds it passed a gene in more cell types than the default gate, 7.3% to 7.5% against 6.8% to 6.9% with 3 replicates per dose;
- `gate={"fold_change": 1.5}` when only changes beyond a size count, tested by TREAT at each dose [@mccarthy2009] rather than cut from the estimated changes;
- {api}`limma_voom` with columns from {api}`DoseAxis` for a model that `dose_response` does not build, such as a spline that differs by another term, dose by sex, or a spline on knots of your own; covariates such as sex go in the design that `dose_response` takes.

See [Dose response](dose_response.md) and [Dose response per cell type](../guides/dose_response.ipynb).

## Does a response follow one of a few expected shapes?

Use {api}`dose_response` with `tests="mcp"` when a few shapes of response are expected: the MCP step, the multiple comparison procedure of MCP-Mod (multiple comparison procedure and modelling), turns each candidate shape into the contrast of the dose groups that best detects it, and tests the largest of their t-statistics against the multivariate t distribution [@bretz2005; @pinheiro2014]. {api}`CandidateShapes.spread` places Emax, sigmoid Emax, exponential and beta shapes over the doses by default, and {api}`CandidateShapes` takes shapes of your own. Switch to:

- `"spline"`, the default gate, for a response of any shape; `gate={"test": "mcp"}` lets the MCP step decide which genes count as responding;
- `"williams"` for responses that rise or fall in dose order, the Williams-type test [@bretz2006];
- `"dunnett"` to ask whether any dose differs from the vehicle and which differs most, with one error rate for the gene's doses [@dunnett1955]; {api}`multiple_contrast_test` on the cell type's fit gives each dose's own adjusted p-value, and {api}`DoseResponseResult.lowest_effective_doses` calls the doses across genes.

See [Trends, doses against the vehicle and candidate shapes](dose_response.md#dose-response-contrasts).

## Could the genes a dose series calls come from noise?

Use {api}`permutation_check` on a result of {api}`dose_response`: in each cell type it permutes the dose labels among the samples that the design treats alike, runs the analysis again on each permuted labelling, and tests by a permutation p-value whether the true doses pass more genes through the gate than permuted ones [@phipson2010], beside how often the gate passes genes with the doses permuted. In vehicle-only data placed in 4 mock dose groups of 6 samples, BMDExpress's default prefilter gave a transcriptomic point of departure in 93% of runs [@lee2024]. Switch to:

- a vehicle-only series that its design places in mock dose groups, run through {api}`dose_response` and the check alike, to measure the gate's error at a design before a study uses it;
- `gate={"fold_change": 1.5}` in {api}`dose_response` when only changes beyond a size matter: the threshold enters each dose's test by TREAT [@mccarthy2009], where a cut on the estimated changes controls no error rate, and in series where no gene responds it passed a gene in at most 1.1% of cell types with 2 replicates per dose, against 10% to 16% for the default gate;
- more replicates, when the smallest p-value the check can give lies above the level: the vehicle and two doses in duplicate allow 90 assignments of the doses, and the default gate, with the vehicle in a group of its own, no p-value below 0.067.

See [Could the calls come from noise?](dose_response.md#dose-permutation-check).

## What shape does each response take?

Use `models` in {api}`dose_response` to fit a set of named curves to every gene of each cell type: `"epa"`, the eight models of the National Toxicology Program's and ETAP's analyses in BMDS's forms and bounds [@ntp2018; @usepa2024; @usepa2026], by default under constant variance, as BMDS fits a gene's samples. Each gene gets each model's parameters, AICc, Akaike weight, lack-of-fit test and largest scaled residual, and its best model ({api}`DoseResponseResult.curve_fits`, `table(g, "curves")`). Switch to:

- `models="ll4"` for drc's four-parameter log-logistic curve [@ritz2015], and `models="dromics"` for DRomics' models, whose Gauss-probit curves rise and then fall, with the dose group that drives each turn [@larras2018];
- `variance="weighted"` or `"moderated"` to fit the dose-group means of the cell type's moderated fit with their covariance, which carries voom's weights and the design's other terms, as a design with terms besides dose needs;
- {api}`fit_dose_curves` for a matrix of values outside a cube, such as bulk log2 CPM.

See [Curves of named models](dose_response.md#dose-response-model-curves).

(from-which-dose)=
## From which dose does each gene respond?

Use `models` and `bmr` in {api}`dose_response`: each gene's best curve gives its benchmark dose, the dose at which the curve departs from its level at dose 0 by the benchmark response, with one-sided 95% profile-likelihood bounds, as BMDS and BMDExpress compute them for NTP's and EPA's analyses [@usepa2012; @ntp2018; @usepa2024]. A benchmark dose can lie between the doses tested, and needs a curve and a stated change: no BMR is set by default, and the 1 SD benchmark dose is always added beside the others. See [Potency and points of departure](potency.md). Switch to:

- {api}`DoseResponseResult.lowest_effective_doses` for the lowest tested dose that differs from the vehicle, at an error rate and without a curve: in the genes that pass the gate, the spline test at 5% FDR by default, each dose is called against the vehicle by Holm's step-down procedure at the level the gate leaves, as stageR does, and the calls control the overall false discovery rate of the genes [@vandenberge2017]; {api}`stage_wise_adjusted_p_values` gives every dose called ([From which dose a gene responds](dose_response.md#lowest-effective-dose));
- `flags` for the checks of BMDS, NTP 2018, ETAP 2024 or EFSA 2022 beside each benchmark dose;
- an absolute `bmr`, such as `BenchmarkResponse.absolute(np.log2(1.5))`, to compare cell types, whose SDs differ with their cells per profile;
- {api}`benchmark_doses` for curves fitted outside a cube with {api}`fit_dose_curves`.

(ntp-and-etap-workflows)=
## What would NTP's or ETAP's workflow find in each cell type?

Use `rules="ntp_2018"` or `rules="etap_2024"` in {api}`dose_response` to run a document's workflow in every cell type, as BMDExpress 2.30.520 runs it: log2(CPM + 1) values, the ANOVA pre-modelling test, Williams' trend test with a 1.5-fold change, the eight models of BMDS 2's programs, the document's BMR and model choice, test 4 and its acceptance criteria [@ntp2018; @usepa2024; @phillips2019]. {api}`DoseResponseResult.rules` gives each setting with its source and what it found, and `table(g, "rule_set")` each gene's steps. See [NTP's and ETAP's rule sets](potency.md#rule-sets). Switch to:

- `models`, `bmr` and `flags` for CytePool's own curves and checks, which read the cell type's moderated fit;
- `selection="bmds_2026"` for BMDS's recommended model, which neither document uses ([BMDS's recommended model](potency.md#bmds-recommended-model));
- {api}`williams_trend_test` for Williams' trend test on any samples x genes values ([Williams' trend test as BMDExpress runs it](dose_response.md#williams-trend-test)).

## Does a signature respond to dose, in the direction expected?

Score the signature in each sample with {api}`gsva`, or a regulator's activity with {api}`ulm`, and pass the scores of one cell type to {api}`dose_response` in place of the cube: every test, table and curve runs on them, fitted as {api}`limma` fits scores, by limma's linear model and empirical Bayes moderation across the scores [@smyth2004; @haenzelmann2013]. A pre-specified signature tested alone is an ordinary linear model, as DoseFinding's `MCTtest` fits one endpoint; `tests=("mcp",)` with `direction="down"` or `"up"` runs the MCP step one-sided, as `MCTtest` does by default, and @furie2025 ran MCP-Mod on such a score. Switch to:

- `direction=None`, the default, for a change either way, with the F-tests;
- `trend=scores.n_genes` for many sets of different sizes, as [Signatures along dose](dose_response.md#dose-response-signatures) says;
- {api}`dose_response` on the cube for the genes themselves, and {api}`limma` to compare two conditions.

See [Signatures along dose](dose_response.md#dose-response-signatures) and [Tests in one direction](dose_response.md#one-sided-dose-tests).

## Do the genes of a pathway respond to a contrast?

Use {api}`camera` to test whether a gene set's genes change more than the remaining genes, accounting for inter-gene correlation [@wu2012]. Switch to:

- {api}`roast`, {api}`mroast` or {api}`fry` for self-contained tests of whether a set's genes change at all (roast for one set, mroast for multiple sets sharing rotations, fry for a fast analytic approximation across many sets);
- {api}`romer` for a rank-based rotation test on {api}`limma_trend` results;
- {api}`over_representation` for an unranked list of significant genes.

See [Gene-set tests](gene_set_tests.md), [Over-representation and GSEA](over_representation_and_gsea.md) and [Gene sets and pathways](../guides/gene_sets.ipynb).

## Do the genes of a pathway gather at the top or bottom of a ranking?

Use {api}`gsea` (matching fgsea) to test whether a gene set's genes are enriched at the top or bottom of a ranked gene list [@subramanian2005; @korotkevich2021]. `gsea` takes a per-gene ranking statistic from any differential-expression method. See [Over-representation and GSEA](over_representation_and_gsea.md) and [Gene sets and pathways](../guides/gene_sets.ipynb).

## How active is a pathway in each sample?

Use {api}`gsva` to compute a per-sample activity score for every gene set [@haenzelmann2013]. Switch to {api}`ssgsea`, {api}`combined_z_scores` or {api}`plage`, then test the scores across conditions with {api}`limma`. See [Pathway scores per sample](pathway_scores.md) and [Pathway and regulator activity](../guides/pathway_activity.ipynb).

## Which regulators and pathways are active?

Use {api}`ulm` (matching decoupler) to score transcription factors or pathways in each sample or contrast from signed target weights [@badiaimompel2022], using networks from {api}`collectri` or {api}`progeny`. Switch to {api}`mlm` when regulators share many targets, or {api}`weighted_aggregate` for permutation-tested weighted sums or means. See [Activities from networks](activities.md) and [Pathway and regulator activity](../guides/pathway_activity.ipynb).

## What do several studies say together?

Use {api}`meta_analysis` to combine per-gene effect estimates across studies with a random-effects model and estimate between-study heterogeneity [@viechtbauer2010]. Switch to:

- {api}`combine_p_values` when studies only provide p-values rather than comparable effect sizes;
- a joint model on stacked studies ({api}`CubeCollection.stack`) when small studies share the same protocol and platform.

See [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md), [Replication](replication.md#combined-p-values), [Align and combine studies](../guides/combine_studies.ipynb) and [Replication across studies](../guides/replication_across_studies.ipynb).

## Which changes replicate across studies?

Use {api}`sumrank` to rank genes within each study and test the sum of ranks across studies for up- and down-regulated genes [@nakatsuka2025]. Switch to:

- {api}`replication` for cross-study concordance and {api}`partial_conjunction` to test whether a gene changes in at least u studies;
- {api}`held_out_prediction` to test whether genes selected in training studies classify samples in a held-out study.

See [Replication](replication.md) and [Replication across studies](../guides/replication_across_studies.ipynb).

(not-implemented-yet)=
## Not implemented yet

Each chapter's *Limits and differences* section documents how CytePool differs from the reference packages:

- **Dose response:** MCP-Mod's modelling step, which fits the shapes that its test step finds and estimates target doses (as DoseFinding's `fitMod` and `TD`); EC50s with their intervals (as in drc); curves of `dream_voom` fits; random intercepts and sample weights in the dose tests of scores, and scores given as one value per sample without their genes; one-sided F-tests, which test order-restricted means; and benchmark doses of curves averaged over models and their bootstrap bounds (as in DRomics' `bmdboot`). {api}`dose_response` tests every cell type along a spline of dose, its departure from a line, its slope and its dose groups, with curves and 95% bands ([Which genes respond to dose](dose_response.md#dose-response-tests)), fits curves of named models, also to cell types that support only description ([Curves of named models](dose_response.md#dose-response-model-curves)), and tests by Dunnett's, Williams-type and MCP tests, with each gene's lowest effective dose ([Trends, doses against the vehicle and candidate shapes](dose_response.md#dose-response-contrasts)), behind a gate at a stated false discovery rate that {api}`permutation_check` checks against permuted doses ([Which genes count as responding](dose_response.md#dose-response-gate)), in one direction when one is expected ([Tests in one direction](dose_response.md#one-sided-dose-tests)), and on the scores of signatures ([Signatures along dose](dose_response.md#dose-response-signatures)); Dunnett's and Williams-type multiple contrast tests also run on any estimates and their covariances through {api}`multiple_contrast_test` ([Several contrasts of one gene](multiple_testing.md#correlated-tests)); {api}`check_dose_design` checks a dose series and ETAP's design minimums ([Checking a dose design](dose_response.md#dose-design-check)); {api}`dose_response` with `bmr` gives each gene's benchmark dose with profile-likelihood bounds and the checks of BMDS, NTP, ETAP and EFSA ([Potency and points of departure](potency.md)), and with `rules` runs NTP's and ETAP's workflows as BMDExpress does ([NTP's and ETAP's rule sets](potency.md#rule-sets)); and {api}`simulate_dose_series` draws dose series from known curves, with their true EC50s and benchmark doses, to see how a method does at a design before it is run ([Simulated dose series with known curves](dose_response.md#simulated-dose-series)).
- **Differential expression:** edgeR and DESeq2 run on one matrix at a time rather than all cell types of a cube; edgeR's `voomLmFit`, `glmLRT` and gene-wise `estimateDisp`; DESeq2's `lfcShrink`, outlier replacement, `glmGamPoi` and local dispersion trends, and gene-specific normalization factors; limma's `duplicateCorrelation` and block correlations in voom.
- **Mixed models and composition:** random slopes `(tissue + 0 | donor)`, generalized linear mixed models, sample quality weights in dream, and bootstrap intervals (`varPartConfInf`); precision-weighted mixed models of composition (crumblr, @hoffman2026b) and Bayesian count models (scCODA, @buttner2021; sccomp, @mangiola2023).
- **Gene sets and activities:** gene-set tests on {api}`dream_voom` fits or with block correlations; gene length or abundance trends in over-representation (`kegga`, `goana`); gene weights in fry and exact z-scores in roast; decoupler's VIPER (whose reference implementation is licensed for academic use only).
- **Several studies:** meta-regression, multivariate meta-analysis, and effect shrinkage across conditions (mash, @urbut2019); r-values (@heller2014) and additional metap methods; study weights and random-effect permutations in SumRank; automatic ontology roll-up across the Cell Ontology.

(how-the-chapters-follow)=
## How the chapters follow each other

The chapters after this page explain the methods in an order where each builds on the ones before:

- **Foundations** cover what every method shares: [Pseudobulk profiles and cubes](cubes.md), the data every test reads; [Designs, linear models and contrasts](linear_models.md), the model fitted to every gene; [Empirical Bayes moderation](empirical_bayes.md), which shrinks each gene's variance towards a value estimated from all the genes; and [Multiple testing](multiple_testing.md), which turns thousands of p-values into a list of calls.
- **Differential expression** applies them to counts: [Normalization and filtering](normalization_and_filtering.md) prepares the counts, [limma-voom and limma-trend](limma_voom.md) tests every cell type, [edgeR and DESeq2](count_models.md) model the counts themselves, and [Mixed models and dream](mixed_models.md) allow for several samples of each donor.
- [Composition tests](composition.md) ask whether the shares of the cell types change.
- [Dose response](dose_response.md) places doses on the axis of a curve, checks whether a dose series supports tests in each cell type, tests which genes respond to dose with curves and bands, trends and candidate shapes, fits curves of named models, decides which genes count as responding with a gate whose calls a permutation check tests against permuted doses, finds the lowest dose from which each gene responds, tests signatures along dose and in one direction, and simulates dose series from known curves; [Potency and points of departure](potency.md) reads from each curve the dose at which its response reaches a stated change, its benchmark dose, with bounds and the checks of regulatory guidance.
- **Gene sets and pathways** turn the results for genes into results for pathways and regulators: [Gene-set tests](gene_set_tests.md), [Over-representation and GSEA](over_representation_and_gsea.md), [Pathway scores per sample](pathway_scores.md) and [Activities from networks](activities.md).
- **Several studies** compare and combine the results of studies: [Aligning studies](aligning_studies.md), [Meta-analysis](meta_analysis.md) and [Replication](replication.md).

The chapters begin with [Pseudobulk profiles and cubes](cubes.md).
