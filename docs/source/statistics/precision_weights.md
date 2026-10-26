---
description: How voom turns counts into log counts per million with a precision weight for every observation, how its span is chosen, and how sample quality weights and the number of cells pooled enter the weights.
---

# Precision weights

Log counts are noisier at low counts than at high ones, and a linear model that treats
every observation alike gives the noisy ones too much say. voom estimates the
mean-variance trend of the log counts and gives every observation a precision weight
from it, so that the weighted linear model and the empirical Bayes moderation of limma
apply to counts [@law2014].

## voom

For counts $r_{gi}$ of gene $g$ in sample $i$ and effective library sizes $R_i$ (library
sizes times normalization factors), voom works with the log2 counts per million

$$
y_{gi} = \log_2 \frac{r_{gi} + 0.5}{R_i + 1}\, 10^6 .
$$

The design is fitted to every gene by least squares, which gives fitted values
$\hat\mu_{gi}$ and residual standard deviations $s_g$. The square roots $s_g^{1/2}$ are
smoothed by LOWESS ({api}`lowess`; @cleveland1979) against each gene's average log
count, $x_g = \bar y_g + \overline{\log_2(R_i + 1)} - \log_2 10^6$, and each observation
gets the weight

$$
w_{gi} = \operatorname{lo}(\lambda_{gi})^{-4}, \qquad
\lambda_{gi} = \log_2\!\left(2^{\hat\mu_{gi}} (R_i + 1) / 10^6\right),
$$

the inverse of the variance that the trend predicts at its fitted log count. {api}`voom`
returns the log2 counts per million, the weights and the trend in a {api}`VoomResult`; a
weighted fit ({api}`fit_linear_model` with `weights`) and {api}`moderated_statistics`
complete limma-voom, which {api}`limma_voom` runs on every cell type of a cube.

**The span.** The LOWESS uses a share of the genes, the span, in each local fit. By
default it is chosen from the number of genes $n$, as limma's `voom` chooses it since
limma 3.66: 1 up to 50 genes, and

$$
\text{span} = 0.3 + 0.7\,(50 / n)^{1/3}
$$

beyond, so 0.65 at 400 genes and 0.395 at 20,000 ({api}`choose_lowess_span`). Before
limma 3.66 the default was 0.5; pass `span=0.5` to reproduce it. In a cube, each cell
type's span follows the number of its genes that pass the filter, and its record keeps
the span used.

## Prior weights and the number of cells

voom takes prior weights for samples, observations or genes: the trend is fitted with
them, and the weights returned are the trend's weights times the prior weights. A
pseudobulk profile sums the counts of its cells, so its log counts per million estimate
the mean expression of those cells, whose variance falls as one over their number: the
precision of a profile is proportional to its number of cells, as the dreamlet paper
argues [@hoffman2026a]. `prior_weights="n_cells"` in {api}`limma_voom` passes each
sample's number of cells as its prior weight.

## Sample quality weights

Some samples follow the model less closely than others: pooled from few cells, sequenced
less deeply, or taken from an outlying donor. Removing them loses power; keeping them
adds noise. Sample quality weights down-weight them instead [@ritchie2006; @liu2015].
The model, for gene $g$ and sample $j$ with prior weights $w_{gj}$, is

$$
E(y_{gj}) = x_j' b_g, \qquad \operatorname{var}(y_{gj}) = \sigma_g^2\, e^{u_j} / w_{gj},
$$

with the log variance factors $u = Z\gamma$ shared by every gene and summing to zero.
The sample weights are $e^{-u_j}$, whose product is one. {api}`sample_quality_weights`
estimates $\gamma$ by residual maximum likelihood [@smyth2002] or gene by gene, as
limma's `arrayWeights` does, with a prior worth 10 genes at equal weights that keeps the
weights near one when the evidence is weak.

{api}`voom_with_quality_weights` combines them with voom as limma's
`voomWithQualityWeights` does: voom; sample weights from its log counts per million and
weights; voom again with those sample weights as prior weights; sample weights again;
and the product of the new trend's weights and the sample weights.
`quality_weights=True` in {api}`limma_voom` runs it in every cell type. The sample
weights and the weights from the number of cells are alternatives: limma's
`voomWithQualityWeights` takes no prior weights, and CytePool refuses the two together.

In a design that pairs two samples per donor, both samples of a donor have equal and
opposite residuals, so an outlying sample pulls its partner's weight down with it.
