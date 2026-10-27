---
description: How to cite CytePool, and which papers to cite for each of its methods.
---

# How to cite

CytePool has no paper yet. Cite the software with the version you ran, which
`cytepool.__version__` gives and every result's provenance record keeps:

> CytePool, version `<version>`, https://github.com/NygenAnalytics/CytePool

Cite also the papers that define the methods you used. CytePool reimplements them, and
the credit for the methods belongs to their authors.

| Functions | Cite |
| --- | --- |
| {api}`limma_voom`, {api}`voom` | voom [@law2014]; limma [@ritchie2015]; empirical Bayes moderation [@smyth2004] |
| {api}`limma_trend` | limma-trend [@law2014]; limma [@ritchie2015; @smyth2004] |
| `robust=True` in the functions above | [@phipson2016] |
| `quality_weights=True`, {api}`sample_quality_weights`, {api}`voom_with_quality_weights` | [@ritchie2006; @liu2015] |
| `prior_weights="n_cells"` | dreamlet [@hoffman2026a] |
| {api}`dream_voom`, {api}`voom_with_dream_weights`, {api}`moderated_fixed_effect_tests` | dream [@hoffman2021] |
| {api}`fit_mixed_model`, {api}`fixed_effect_tests`, {api}`fixed_effect_f_test` | lme4 [@bates2015]; Satterthwaite's tests as lmerTest computes them [@kuznetsova2017]; Kenward and Roger's [@kenward1997] as pbkrtest computes them [@halekoh2014] |
| {api}`treat` | [@mccarthy2009] |
| {api}`fit_quasi_likelihood`, {api}`quasi_likelihood_f_test` | edgeR 4 [@chen2025]; quasi-likelihood [@lund2012] |
| {api}`fit_negative_binomial_model`, {api}`common_dispersion` | [@mccarthy2012] |
| {api}`fit_deseq2`, {api}`deseq2_wald_test`, {api}`deseq2_likelihood_ratio_test` | DESeq2 [@love2014] |
| {api}`tmm_factors` | [@robinson2010a] |
| {api}`median_ratio_size_factors`, {api}`rle_factors` | [@anders2010] |
| {api}`upper_quartile_factors` | [@bullard2010] |
| {api}`filter_by_expression` | [@chen2016] |
| {api}`camera` | [@wu2012] |
| {api}`fry` | [@giner2016] |
| {api}`roast`, {api}`mroast` | [@wu2010] |
| {api}`romer` | [@majewski2010] |
| {api}`over_representation` | limma's `kegga` and `goana` [@young2010]; the hypergeometric test [@tavazoie1999] |
| {api}`gsea` | fgsea [@korotkevich2021]; GSEA [@subramanian2005] |
| {api}`ulm`, {api}`mlm`, {api}`weighted_aggregate` | decoupler [@badiaimompel2022], and the network you used: CollecTRI [@muellerdott2023] or PROGENy [@schubert2018], for mouse [@holland2020b] |
| {api}`gsva`, {api}`ssgsea`, {api}`combined_z_scores`, {api}`plage` | the GSVA package [@haenzelmann2013], and the paper of the method you chose: GSVA [@haenzelmann2013], ssGSEA [@barbie2009], combined z-scores [@lee2008] or PLAGE [@tomfohr2005] |
| {api}`read_gmt` | the collection you read, as its provider asks; for MSigDB, [@subramanian2005; @liberzon2011] |
| {api}`propeller` | [@phipson2022] |
| {api}`meta_analysis`, {api}`fit_meta_analysis` | metafor [@viechtbauer2010]; the estimator of the between-study variance you chose: REML [@viechtbauer2005], DerSimonian-Laird [@dersimonian1986] or Paule-Mandel [@paule1982]; Knapp-Hartung [@knapp2003] |
| {api}`adjust_p_values` | the method you chose: [@benjamini1995], [@benjamini2001], [@holm1979], [@hochberg1988] or [@hommel1988] |

Each module's documentation lists, under References, every work it implements; the
[Bibliography](bibliography.md) collects them. For the data of the documentation, cite
@kang2018 and, for the Hallmark gene set, @liberzon2015 ([Datasets](datasets.md)).
