"""Activities of transcription factors and pathways, inferred from the expression of
their target genes, as decoupler computes them: the univariate and multivariate
linear models (ULM, MLM) and the weighted aggregate.

A network (:class:`~cytepool.networks.Network`), such as CollecTRI's regulons or
PROGENy's pathway signatures (:mod:`cytepool.networks`), gives each source, a
transcription factor or a pathway, a signed weight on each of its target genes. A
source is active in a profile, the expression of a sample or the statistics of a
contrast, when its targets are high where their weights are positive and low where
they are negative. Each function scores every source in every profile, as decoupler
2.2.0's ``mt.ulm``, ``mt.mlm`` and ``mt.waggr`` do (Badia-i-Mompel et al. 2022), and
returns an :class:`Activities`, profiles x sources, which the linear models take as
they take pathway scores, and which :func:`~cytepool.plotting.plot_activities` draws
as a heatmap and :func:`~cytepool.plotting.plot_top_activities` as bars.

The profiles:
    - A samples x genes array, one profile per row, with one gene id per column.
    - A group (cell type) of a :class:`~cytepool.cube.Cube`: its samples with at
      least ``min_cells`` cells and some counts, over the genes that edgeR's
      ``filterByExpr`` keeps, as log2 counts per million after TMM normalization,
      the values that :func:`~cytepool.pathway_scores.gsva` scores.
    - The statistics of a
      :class:`~cytepool.differential_expression.DifferentialExpressionResult`: one
      profile for every group and contrast tested, by default the moderated
      t-statistics, as decoupler's guides score the statistics of a contrast; or of a
      table of results, a pandas Series or a mapping of gene id to statistic, one
      profile.

    Each profile's genes are those with an id and a value: NaN marks a missing value,
    and a gene that a group did not test has none in that group's profiles. Genes
    whose values are zero in every profile are left out, as decoupler leaves them
    out: in sparse data they are genes that were not detected.

The network's targets:
    Targets are matched to the genes' ids exactly. An edge whose weight is zero is no
    edge. A source with fewer than ``min_targets`` targets among a profile's genes
    has no score there (NaN), and a source with fewer in every profile is left out,
    as decoupler's ``tmin`` leaves it out; ``n_targets`` gives each source's targets
    among each profile's genes.

ULM (:func:`ulm`):
    For each source and profile, a linear model of the profile's n genes on the
    source's weights, zero for the genes that are not its targets:

        y_g = b_0 + b_1 x_g + e_g,

    fitted by least squares. The activity score is the t-statistic of the slope,
    t = b_1 / se(b_1) = r sqrt((n - 2) / (1 - r^2)), with r the correlation of y and
    x, and its p-value is two-sided, from Student's t distribution on n - 2 degrees
    of freedom. ``estimates`` are the slopes b_1.

MLM (:func:`mlm`):
    One linear model of each profile's genes on the weights of all its sources at
    once, with an intercept, y_g = b_0 + sum_s b_s x_gs + e_g, so that each source's
    score, the t-statistic of b_s on n - m - 1 degrees of freedom for m sources, is
    that of its targets beyond what the other sources explain. The model is fitted by
    a QR decomposition. Sources whose weights over the genes are linear combinations
    of the weights of other sources, such as two sources with the same targets and
    weights, cannot be told apart, and are refused: the message names them.

Weighted aggregate (:func:`weighted_aggregate`):
    The weighted mean of the values of a source's targets, sum_g w_g y_g /
    sum_g |w_g|, or their weighted sum, sum_g w_g y_g, the score of the original
    PROGENy method (Schubert et al. 2018). With ``n_permutations`` random
    permutations of the genes, each permutation giving every gene the value of
    another, as decoupler draws them (its ``seed`` is ``rng``), the score is
    normalized by the scores of the permutations, (ES - mean) / sd, the standard
    deviation with n - 1 in its denominator, and the p-value doubles the smaller
    of the two one-sided permutation p-values, (1 + b) / (1 + n_permutations), with b
    the permutations whose score is at least, or at most, the observed one (Phipson
    and Smyth 2010). ``estimates`` keep the weighted means or sums.

Adjusted p-values:
    ``adjusted_p_values`` are Benjamini and Hochberg's (1995) over the sources of each
    profile, as decoupler adjusts them.

Deliberate differences from decoupler 2.2.0:
    - decoupler reports MLM's p-values without adjusting them, under the name of its
      adjusted p-values; here they are ``p_values``, and ``adjusted_p_values`` are
      adjusted as for the other methods.
    - decoupler keeps adjusted p-values in single precision, and computes MLM's
      p-values as one minus the distribution function, which gives 0 below about
      1e-16; here both are in double precision, from the upper tail.
    - decoupler counts the permutations of the weighted aggregate whose absolute
      score exceeds the absolute observed score, and then folds that share as it
      would fold a one-sided one, so that a score near zero, which most permutations
      exceed, gets a p-value near 2 / n_permutations; here the p-value doubles the
      smaller one-sided p-value. When every permutation scores alike, decoupler's
      normalized score is infinite, or 0 for a score of 0; here it is NaN.
    - decoupler adds 2.2e-16 to 1 - r and 1 + r in ULM's t-statistic; here it is
      exact, which changes the score only for |r| within about 1e-6 of 1.
    - decoupler refuses missing values and leaves out samples whose values are all
      zero; here each profile uses its observed genes, and a profile without genes
      has NaN scores. Where MLM cannot separate sources, decoupler stops with a
      singular matrix; here the message names them.

References:
    Badia-i-Mompel P, Velez Santiago J, Braunger J, Geiss C, Dimitrov D, Mueller-Dott S,
    Taus P, Dugourd A, Holland CH, Ramirez Flores RO, Saez-Rodriguez J (2022).
    decoupleR: ensemble of computational methods to infer biological activities from
    omics data. Bioinformatics Advances 2, vbac016. doi:10.1093/bioadv/vbac016

    Schubert M, Klinger B, Kluenemann M, Sieber A, Uhlitz F, Sauer S, Garnett MJ,
    Bluethgen N, Saez-Rodriguez J (2018). Perturbation-response genes reveal
    signaling footprints in cancer gene expression. Nature Communications 9, 20.
    doi:10.1038/s41467-017-02391-6

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Welford BP (1962). Note on a method for calculating corrected sums of squares and
    products. Technometrics 4, 419-420. doi:10.1080/00401706.1962.10490022
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numba import prange

from ._analysis_records import (
    keep_provenance,
    provenance_of,
    provenance_text,
    restore_provenance,
)
from ._columns import is_data_frame, text_column
from ._expression_values import MIN_CELLS, expression_values
from ._jit import by_chunks, jit, thread_limit
from ._linear_algebra import RANK_TOLERANCE
from ._masked_arrays import read_only, reject_masked
from ._messages import listing, quoted
from ._options import (
    checked_choice,
    integral_number,
    positive_integer,
    random_generator,
)
from ._ranked_genes import ranked_genes
from .cube import Cube
from .differential_expression import DifferentialExpressionResult
from .networks import Network
from .stats.distributions import t_sf
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = ["Activities", "mlm", "ulm", "weighted_aggregate"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_MIN_TARGETS = 5  # decoupler's tmin
_N_PERMUTATIONS = 1000  # decoupler's times
_STATISTICS = ("mean", "sum")
_BLOCK = 128  # permutations drawn and scored at a time
_PROFILE_ID = "profile_id"


@dataclass(frozen=True, eq=False, repr=False)
class Activities:
    """Activities of the sources of a network in every profile.

    Rows of the arrays are profiles, in the order of ``profiles``, and columns are
    sources, in the order of ``network.sources``, so that
    :func:`~cytepool.stats.linear_models.fit_linear_model` takes ``scores`` as its
    values, with a design built from ``profiles``. The arrays are read-only.

    Attributes:
        scores: ``(n_profiles, n_sources)`` activity scores: the t-statistics of ULM
            and MLM, and of the weighted aggregate the scores normalized by
            permutations, or, without them, the weighted means or sums. NaN where a
            source has fewer than ``min_targets`` targets among a profile's genes.
        estimates: The slopes (ULM), coefficients (MLM) or weighted means or sums
            (weighted aggregate).
        p_values: Two-sided p-values: of the t-tests (ULM, MLM), or of the
            permutations (weighted aggregate; NaN without them).
        adjusted_p_values: Benjamini and Hochberg's, over the sources of each
            profile.
        n_targets: ``(n_profiles, n_sources)`` int64: the targets of each source
            among each profile's genes.
        profiles: One row per profile: ``profile_id`` and, for a cube's group, the
            columns of the cube's sample table, or, for a differential-expression
            result, ``group_id`` and ``contrast``.
        network: The sources scored, with their targets among the genes and the
            record of the network given.
        method: ``"ulm"``, ``"mlm"`` or ``"weighted_aggregate"``.
        settings: The options of the method and how the profiles were read.
    """

    scores: FloatArray
    estimates: FloatArray
    p_values: FloatArray
    adjusted_p_values: FloatArray
    n_targets: IntArray
    profiles: Table
    network: Network
    method: str
    settings: Mapping[str, Any]

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # The settings are a read-only view, which pickle cannot write: a dict
        # travels in their place, and the provenance kept outside the fields with it.
        return (
            _restored_activities,
            (
                self.scores,
                self.estimates,
                self.p_values,
                self.adjusted_p_values,
                self.n_targets,
                self.profiles,
                self.network,
                self.method,
                dict(self.settings),
                provenance_text(self),
            ),
        )

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources scored, in the order of the columns."""
        return self.network.sources

    @property
    def provenance(self) -> dict[str, Any] | None:
        """A JSON-serialisable record of what produced the activities of a cube's
        group, as a new dict: the method, the CytePool, numpy, numba and Python
        versions, the cube's fingerprint, the digests it is built from and its
        provenance (``"cube"``), the layer read and the settings; None for
        activities of values given as an array, or of statistics, whose result
        keeps its own record. The cube's record costs what it costs for
        :func:`~cytepool.differential_expression.limma_voom` ("Provenance")."""
        return provenance_of(self)

    def table(self) -> Table:
        """Return the activities as a table with one row per source and profile.

        Returns:
            A table with ``source``, ``profile_id``, ``n_targets``, ``score``,
            ``estimate``, ``p_value`` and ``adjusted_p_value``, the rows of each
            source together, in the order of ``sources``, and its profiles in the
            order of ``profiles``.
        """
        n_profiles, n_sources = self.scores.shape
        ids = self.profiles[_PROFILE_ID].tolist()
        return Table(
            {
                "source": text_column([source for source in self.sources for _ in ids]),
                "profile_id": text_column(ids * n_sources),
                "n_targets": self.n_targets.T.ravel().copy(),
                "score": self.scores.T.ravel().copy(),
                "estimate": self.estimates.T.ravel().copy(),
                "p_value": self.p_values.T.ravel().copy(),
                "adjusted_p_value": self.adjusted_p_values.T.ravel().copy(),
            }
        )

    def __repr__(self) -> str:
        n_profiles, n_sources = self.scores.shape
        return (
            f"Activities(method={self.method!r}, n_profiles={n_profiles}, "
            f"n_sources={n_sources})"
        )


def _restored_activities(
    scores: FloatArray,
    estimates: FloatArray,
    p_values: FloatArray,
    adjusted_p_values: FloatArray,
    n_targets: IntArray,
    profiles: Table,
    network: Network,
    method: str,
    settings: dict[str, Any],
    provenance: str | None,
) -> Activities:
    """Activities unpickled: read-only again, with their provenance."""
    restored = Activities(
        read_only(np.asarray(scores)),
        read_only(np.asarray(estimates)),
        read_only(np.asarray(p_values)),
        read_only(np.asarray(adjusted_p_values)),
        read_only(np.asarray(n_targets)),
        profiles,
        network,
        method,
        MappingProxyType(settings),
    )
    return restore_provenance(restored, provenance)


class _Profiles(NamedTuple):
    """The values scored, profiles x genes, with their genes and profiles."""

    values: FloatArray  # NaN where a profile has no value of a gene
    gene_ids: list[str | None]
    profiles: Table
    settings: dict[str, Any]
    cube: Cube | None = None  # the cube whose group was read


class _Edges(NamedTuple):
    """The network's edges among the genes kept: those of each source after another,
    by the position of their gene among the genes kept."""

    starts: IntArray  # where each source's edges start, and where the last ends
    genes: IntArray
    weights: FloatArray
    rows: IntArray  # the rows of the network's table


def ulm(
    data: Any,
    network: Any,
    *,
    gene_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    contrast: str | None = None,
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    min_targets: int = _MIN_TARGETS,
    n_threads: int | None = None,
) -> Activities:
    """Score the sources of a network in every profile by univariate linear models
    (ULM).

    As decoupler's ``mt.ulm`` (Badia-i-Mompel et al. 2022): for each source, the
    values of a profile's genes are regressed on the source's weights, zero for the
    genes that are not its targets, and the score is the t-statistic of the slope.
    The module docstring gives the profiles, the matching of targets and the
    deliberate differences from decoupler.

    Args:
        data: ``(n_profiles, n_genes)`` values, such as log-expression values of
            samples or statistics of contrasts, NaN where missing; a
            :class:`~cytepool.cube.Cube`, whose ``group`` is scored; a
            :class:`~cytepool.differential_expression.DifferentialExpressionResult`,
            whose statistics of every group and contrast are scored; or a table of
            results (a ``Table`` or pandas DataFrame), a pandas Series indexed by
            gene id or a mapping of gene id to statistic, one profile. A samples x
            genes DataFrame, as decoupler takes it, is given as
            ``frame.to_numpy()`` with ``gene_ids`` and ``sample_ids``.
        network: A :class:`~cytepool.networks.Network`, such as
            :func:`~cytepool.networks.collectri` or
            :func:`~cytepool.networks.progeny` returns, or what ``Network`` takes: a
            table with the columns ``source``, ``target`` and ``weight``, or a
            mapping of source to targets.
        gene_ids: For an array, the genes' ids, one per column (None, NaN or an
            empty string for a gene without one); for a cube, the name of a column of
            its features table, ``"feature_id"`` by default or ``"symbol"``, or one
            id per feature.
        sample_ids: For an array, one id per row; by default ``"1"``, ``"2"``, ...
        group: For a cube, the group (cell type) to score; it may be left out when
            the cube has one. For a differential-expression result, the group whose
            statistics are scored; by default every group.
        contrast: For a differential-expression result, the contrast whose
            statistics are scored; by default every contrast.
        min_cells: For a cube, the fewest cells a sample needs in the group to be
            scored, at least 0, as for
            :func:`~cytepool.differential_expression.limma_voom`.
        expression_filter: For a cube, whether to score only the genes that
            :func:`~cytepool.normalization.filter_by_expression` keeps, with the
            samples as one group; or a mapping of its settings, which may name a
            column of the cube's samples whose values group them, such as
            ``{"sample_groups": "condition"}``, as
            :func:`~cytepool.differential_expression.limma_voom` takes it.
        id_column: For a differential-expression result or a table, the column of
            gene ids, such as ``"feature_id"`` or ``"symbol"``.
        statistic_column: For a differential-expression result or a table, the
            column of statistics, such as ``"t"`` or ``"log2_fold_change"``.
        min_targets: The fewest targets among a profile's genes for a source to be
            scored, at least 1 (decoupler's ``tmin``).
        n_threads: The most threads to use; None uses numba's current setting. The
            result does not depend on it.

    Returns:
        The activities, with the t-statistics as ``scores`` and the slopes as
        ``estimates``.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range or
            applies to another kind of data, a table lacks a column, or no source has
            ``min_targets`` targets among the genes.
        KeyError: If the cube has no such group or features column, or the result
            has no such group or contrast.

    Example:
        >>> import numpy as np
        >>> from cytepool.networks import Network
        >>> rng = np.random.default_rng(1)
        >>> genes = [f"g{j}" for j in range(40)]
        >>> y = rng.normal(size=(2, 40))
        >>> y[1, :6] += 3.0  # the targets of A are up in the second sample
        >>> network = Network({"A": genes[:6], "B": genes[10:16]})
        >>> ulm(y, network, gene_ids=genes).scores.round(2).tolist()
        [[0.71, -0.1], [7.94, -0.85]]
    """
    profiles, graph, edges, smallest = _prepared(
        data,
        network,
        gene_ids,
        sample_ids,
        group,
        contrast,
        min_cells,
        expression_filter,
        id_column,
        statistic_column,
        min_targets,
    )
    values = profiles.values
    n_profiles = values.shape[0]
    n_sources = edges.starts.size - 1
    slopes = np.full((n_profiles, n_sources), np.nan)
    t_values = np.full((n_profiles, n_sources), np.nan)
    counts = np.zeros((n_profiles, n_sources), dtype=np.int64)
    with thread_limit(n_threads):
        by_chunks(
            _ulm_parallel,
            _ulm_serial,
            (values, edges.starts, edges.genes, edges.weights, smallest, slopes,
             t_values, counts),
            n_profiles,
        )  # fmt: skip
    df = np.sum(~np.isnan(values), axis=1).astype(np.float64) - 2.0
    return _activities(
        "ulm",
        t_values,
        slopes,
        _two_sided(t_values, df),
        counts,
        profiles,
        graph,
        edges,
        {"min_targets": smallest},
    )


def mlm(
    data: Any,
    network: Any,
    *,
    gene_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    contrast: str | None = None,
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    min_targets: int = _MIN_TARGETS,
) -> Activities:
    """Score the sources of a network in every profile by one multivariate linear
    model (MLM).

    As decoupler's ``mt.mlm`` (Badia-i-Mompel et al. 2022): the values of a
    profile's genes are regressed on the weights of all the sources at once, with an
    intercept, and each source's score is the t-statistic of its coefficient, which
    measures its targets beyond what the other sources explain. The module docstring
    gives the profiles, the matching of targets and the deliberate differences from
    decoupler.

    Args:
        data: The profiles, as for :func:`ulm`.
        network: The network, as for :func:`ulm`.
        gene_ids: For an array or a cube, as for :func:`ulm`.
        sample_ids: For an array, as for :func:`ulm`.
        group: For a cube or a differential-expression result, as for :func:`ulm`.
        contrast: For a differential-expression result, as for :func:`ulm`.
        min_cells: For a cube, as for :func:`ulm`.
        expression_filter: For a cube, as for :func:`ulm`.
        id_column: For a differential-expression result or a table, as for
            :func:`ulm`.
        statistic_column: For a differential-expression result or a table, as for
            :func:`ulm`.
        min_targets: The fewest targets among a profile's genes for a source to be
            in its model, at least 1 (decoupler's ``tmin``).

    Returns:
        The activities, with the t-statistics as ``scores`` and the coefficients as
        ``estimates``.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range or
            applies to another kind of data, a table lacks a column, no source has
            ``min_targets`` targets among the genes, a profile has no more genes than
            sources, or the weights of some sources are linear combinations of
            those of others.
        KeyError: If the cube has no such group or features column, or the result
            has no such group or contrast.
    """
    profiles, graph, edges, smallest = _prepared(
        data,
        network,
        gene_ids,
        sample_ids,
        group,
        contrast,
        min_cells,
        expression_filter,
        id_column,
        statistic_column,
        min_targets,
    )
    values = profiles.values
    n_profiles = values.shape[0]
    n_sources = edges.starts.size - 1
    coefficients = np.full((n_profiles, n_sources), np.nan)
    t_values = np.full((n_profiles, n_sources), np.nan)
    p_values = np.full((n_profiles, n_sources), np.nan)
    counts = np.zeros((n_profiles, n_sources), dtype=np.int64)
    observed = ~np.isnan(values)
    patterns: dict[bytes, list[int]] = {}
    for row in range(n_profiles):
        patterns.setdefault(np.packbits(observed[row]).tobytes(), []).append(row)
    names = list(graph.sources)
    for rows in patterns.values():
        _fit_pattern(
            values, observed, rows, edges, smallest, names, profiles.profiles,
            coefficients, t_values, p_values, counts,
        )  # fmt: skip
    return _activities(
        "mlm",
        t_values,
        coefficients,
        p_values,
        counts,
        profiles,
        graph,
        edges,
        {"min_targets": smallest},
    )


def weighted_aggregate(
    data: Any,
    network: Any,
    *,
    statistic: str = "mean",
    n_permutations: int = _N_PERMUTATIONS,
    rng: np.random.Generator | int | None = None,
    gene_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    contrast: str | None = None,
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    min_targets: int = _MIN_TARGETS,
    n_threads: int | None = None,
) -> Activities:
    """Score the sources of a network in every profile by the weighted mean or sum of
    their targets' values.

    As decoupler's ``mt.waggr`` (Badia-i-Mompel et al. 2022): the weighted mean of
    the values of a source's targets, sum w y / sum |w|, or their weighted sum,
    sum w y, which is how PROGENy scores its pathways (Schubert et al. 2018); with
    permutations of the genes, normalized by the scores of the permutations, with a
    permutation p-value. The module docstring gives the permutations, the p-values
    and the deliberate differences from decoupler.

    Args:
        data: The profiles, as for :func:`ulm`.
        network: The network, as for :func:`ulm`.
        statistic: ``"mean"``, the weighted mean (decoupler's ``"wmean"``), or
            ``"sum"``, the weighted sum (``"wsum"``).
        n_permutations: The random permutations of the genes, 0 for none or at least
            2 (decoupler's ``times``).
        rng: A ``numpy.random.Generator`` to draw the permutations from, an integer
            seed for one (decoupler's ``seed``, 42 by default there), or None for a
            fresh, unseeded generator.
        gene_ids: For an array or a cube, as for :func:`ulm`.
        sample_ids: For an array, as for :func:`ulm`.
        group: For a cube or a differential-expression result, as for :func:`ulm`.
        contrast: For a differential-expression result, as for :func:`ulm`.
        min_cells: For a cube, as for :func:`ulm`.
        expression_filter: For a cube, as for :func:`ulm`.
        id_column: For a differential-expression result or a table, as for
            :func:`ulm`.
        statistic_column: For a differential-expression result or a table, as for
            :func:`ulm`.
        min_targets: The fewest targets among a profile's genes for a source to be
            scored, at least 1 (decoupler's ``tmin``).
        n_threads: The most threads to use; None uses numba's current setting. The
            result does not depend on it.

    Returns:
        The activities, with the normalized scores as ``scores`` (the weighted means
        or sums without permutations) and the weighted means or sums as
        ``estimates``.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range or
            applies to another kind of data, a table lacks a column, or no source has
            ``min_targets`` targets among the genes.
        KeyError: If the cube has no such group or features column, or the result
            has no such group or contrast.
    """
    kind = checked_choice(statistic, "statistic", _STATISTICS)
    reject_masked(n_permutations, "n_permutations")
    count = integral_number(n_permutations, "n_permutations")
    if count < 0 or count == 1:
        raise ValueError(
            f"n_permutations must be 0, for no permutations, or at least 2, got {count}"
        )
    generator = random_generator(rng)
    profiles, graph, edges, smallest = _prepared(
        data,
        network,
        gene_ids,
        sample_ids,
        group,
        contrast,
        min_cells,
        expression_filter,
        id_column,
        statistic_column,
        min_targets,
    )
    values = profiles.values
    n_profiles, n_genes = values.shape
    n_sources = edges.starts.size - 1
    mean = kind == "mean"
    estimates = np.full((n_profiles, n_sources), np.nan)
    counts = np.zeros((n_profiles, n_sources), dtype=np.int64)
    with thread_limit(n_threads):
        identity = np.arange(n_genes, dtype=np.int64)[np.newaxis, :]
        _aggregate(values, edges, identity, mean, smallest, estimates, counts)
        scores = estimates.copy()
        p_values = np.full((n_profiles, n_sources), np.nan)
        if count:
            scores, p_values = _permuted(
                values, edges, mean, smallest, estimates, count, generator
            )
    settings = {
        "statistic": kind,
        "n_permutations": count,
        "min_targets": smallest,
    }
    return _activities(
        "weighted_aggregate",
        scores,
        estimates,
        p_values,
        counts,
        profiles,
        graph,
        edges,
        settings,
    )


# The profiles and the network.


def _prepared(
    data: Any,
    network: Any,
    gene_ids: Any,
    sample_ids: Any,
    group: Any,
    contrast: Any,
    min_cells: Any,
    expression_filter: Any,
    id_column: Any,
    statistic_column: Any,
    min_targets: Any,
) -> tuple[_Profiles, Network, _Edges, int]:
    """The profiles over the genes kept, the network, its edges among those genes,
    and the fewest targets a source needs."""
    reject_masked(min_targets, "min_targets")
    smallest = positive_integer(min_targets, "min_targets")
    graph = network if isinstance(network, Network) else _network(network)
    profiles = _profiles(
        data,
        gene_ids,
        sample_ids,
        group,
        contrast,
        min_cells,
        expression_filter,
        id_column,
        statistic_column,
    )
    values = profiles.values
    observed = ~np.isnan(values)
    named = np.array([gene is not None for gene in profiles.gene_ids], dtype=bool)
    kept = np.flatnonzero(named & np.any(observed & (values != 0.0), axis=0))
    ids = [profiles.gene_ids[j] for j in kept.tolist()]
    edges = _matched_edges(graph, ids, profiles.gene_ids)
    kept_profiles = _Profiles(
        np.ascontiguousarray(values[:, kept]),
        list(ids),
        profiles.profiles,
        profiles.settings,
        profiles.cube,
    )
    return kept_profiles, graph, edges, smallest


def _network(network: Any) -> Network:
    """A network given as a table or a mapping, which Network checks."""
    if isinstance(network, Table | Mapping) or is_data_frame(network):
        return Network(network)
    reject_masked(network, "network")
    raise TypeError(
        "network must be a Network, such as collectri() or progeny() returns, a "
        "table with the columns 'source', 'target' and 'weight', or a mapping of "
        f"source to targets; got {type(network).__name__}"
    )


def _matched_edges(
    network: Network, ids: list[str | None], all_ids: list[str | None]
) -> _Edges:
    """The edges of every source of the network whose target is among the genes kept
    and whose weight is not zero, by source in the network's order."""
    position = {gene: j for j, gene in enumerate(ids)}
    edges = network.edges
    sources = edges["source"].tolist()
    targets = edges["target"].tolist()
    weights = np.asarray(edges["weight"], dtype=np.float64)
    by_source: dict[str, list[int]] = {source: [] for source in network.sources}
    for row, (source, target) in enumerate(zip(sources, targets, strict=True)):
        if target in position and weights[row] != 0.0:
            by_source[source].append(row)
    rows = [row for source in network.sources for row in by_source[source]]
    if not rows:
        examples = list(dict.fromkeys(targets))[:3]
        genes = [gene for gene in all_ids if gene is not None][:3]
        raise ValueError(
            "no target of the network is among the genes; the network and the data "
            "must name genes the same way (for example both by symbol, or both by "
            f"Ensembl id): the network names genes such as {quoted(examples)}, the "
            f"data genes such as {quoted(genes)}"
        )
    starts = np.zeros(len(network.sources) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(by_source[source]) for source in network.sources])
    taken = np.array(rows, dtype=np.int64)
    return _Edges(
        starts,
        np.array([position[targets[row]] for row in rows], dtype=np.int64),
        np.ascontiguousarray(weights[taken]),
        taken,
    )


def _profiles(
    data: Any,
    gene_ids: Any,
    sample_ids: Any,
    group: Any,
    contrast: Any,
    min_cells: Any,
    expression_filter: Any,
    id_column: Any,
    statistic_column: Any,
) -> _Profiles:
    """The profiles of an array, a cube's group or a set of statistics."""
    statistics = (
        isinstance(data, DifferentialExpressionResult | Table | Mapping)
        or is_data_frame(data)
        or _is_series(data)
    )
    if statistics:
        for name, value, default in (
            ("gene_ids", gene_ids, gene_ids is None),
            ("sample_ids", sample_ids, sample_ids is None),
            ("min_cells", min_cells, _is_default_cells(min_cells)),
            ("expression_filter", expression_filter, expression_filter is True),
        ):
            if not default:
                reject_masked(value, name)
                hint = ""
                if name == "gene_ids":
                    hint = "; name the column of gene ids with id_column"
                raise ValueError(
                    f"{name} applies to an array or a cube, and data is a "
                    f"{type(data).__name__}{hint}"
                )
        return _statistics(data, group, contrast, id_column, statistic_column)
    for name, given, expected in (
        ("contrast", contrast, None),
        ("id_column", id_column, "feature_id"),
        ("statistic_column", statistic_column, "t"),
    ):
        if expected is None:
            same = given is None
        else:
            same = isinstance(given, str) and given == expected
        if not same:
            reject_masked(given, name)
            raise ValueError(
                f"{name} applies to the statistics of a differential-expression "
                f"result or a table, and data is a {type(data).__name__}"
            )
    expression = expression_values(
        data,
        gene_ids,
        sample_ids,
        group,
        "log_cpm",
        min_cells,
        expression_filter,
        matched_to="the network's targets",
        other_forms=(
            ", a DifferentialExpressionResult, a table of statistics or a mapping of "
            "gene id to statistic"
        ),
    )
    samples = expression.samples
    if _PROFILE_ID in samples:
        raise ValueError(
            f"the sample table has a column {_PROFILE_ID!r}, the name the activities "
            "give the ids of their profiles; rename it"
        )
    ids = samples["sample_id"]
    table = Table({_PROFILE_ID: ids, **samples.to_dict()})
    return _Profiles(
        expression.values,
        expression.gene_ids,
        table,
        dict(expression.settings),
        expression.cube,
    )


def _statistics(
    data: Any, group: Any, contrast: Any, id_column: Any, statistic_column: Any
) -> _Profiles:
    """The statistics of every group and contrast of a result, or of a table or
    mapping, one profile each, over the union of their genes."""
    ranked = []
    if isinstance(data, DifferentialExpressionResult):
        groups = list(data.groups)
        contrasts = list(data.contrasts.names)
        for value, name, options in (
            (group, "group", groups),
            (contrast, "contrast", contrasts),
        ):
            if value is not None:
                if not isinstance(value, str):
                    reject_masked(value, name)
                    raise TypeError(
                        f"{name} must be a string, got {type(value).__name__}"
                    )
                if value not in options:
                    raise KeyError(
                        f"the result has no {name} {value!r}; its {name}s are "
                        f"{quoted(options)}"
                    )
        pairs = [
            (g, c)
            for g in groups
            for c in contrasts
            if (g, c) in data.tables
            and (group is None or g == group)
            and (contrast is None or c == contrast)
        ]
        if not pairs:
            raise ValueError(
                "the result has no table of "
                + (f"group {group!r}" if group is not None else "any group")
                + " and "
                + (f"contrast {contrast!r}" if contrast is not None else "any contrast")
            )
        for g, c in pairs:
            ranked.append(
                ranked_genes(
                    data,
                    group=g,
                    contrast=c,
                    id_column=id_column,
                    statistic_column=statistic_column,
                )
            )
        if len(contrasts) == 1 or contrast is not None:
            ids = [g for g, _ in pairs]
        elif len(groups) == 1 or group is not None:
            ids = [c for _, c in pairs]
        else:
            ids = [f"{g}: {c}" for g, c in pairs]
        table = Table(
            {
                _PROFILE_ID: text_column(ids),
                "group_id": text_column([g for g, _ in pairs]),
                "contrast": text_column([c for _, c in pairs]),
            }
        )
    else:
        if (isinstance(data, Table) or is_data_frame(data)) and not (
            _has_column(data, id_column) or _has_column(data, statistic_column)
        ):
            raise ValueError(
                f"data is read as a table of results, with a column of gene ids "
                f"({id_column!r}) and one of statistics ({statistic_column!r}), and "
                "it has neither. To score a samples x genes table, give its values "
                "with their ids: data=frame.to_numpy(), gene_ids=list(frame.columns), "
                "sample_ids=list(frame.index)"
            )
        ranked.append(
            ranked_genes(
                data,
                group=group,
                contrast=contrast,
                id_column=id_column,
                statistic_column=statistic_column,
            )
        )
        if isinstance(data, Table) or is_data_frame(data):
            label = str(statistic_column)
        elif _is_series(data) and data.name is not None:
            label = str(data.name)
        else:
            label = "statistic"
        table = Table({_PROFILE_ID: text_column([label])})
    genes = list(dict.fromkeys(gene for found in ranked for gene in found.ids))
    position = {gene: j for j, gene in enumerate(genes)}
    values = np.full((len(ranked), len(genes)), np.nan)
    for row, found in enumerate(ranked):
        columns = np.array([position[gene] for gene in found.ids], dtype=np.intp)
        values[row, columns] = found.values
    settings = {"id_column": id_column, "statistic_column": statistic_column}
    return _Profiles(values, list(genes), table, settings)


def _has_column(data: Any, name: Any) -> bool:
    """Whether a Table or DataFrame has a column, or a DataFrame an index, of this
    name."""
    if not isinstance(name, str):
        return False
    if name in list(data.columns):
        return True
    return is_data_frame(data) and name in list(data.index.names)


def _is_default_cells(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == MIN_CELLS


def _is_series(value: object) -> bool:
    """Whether ``value`` is a pandas Series, recognised without importing pandas."""
    return all(
        hasattr(value, name) for name in ("index", "to_numpy", "name", "dtype")
    ) and not hasattr(value, "columns")


# The result.


def _activities(
    method: str,
    scores: FloatArray,
    estimates: FloatArray,
    p_values: FloatArray,
    counts: IntArray,
    profiles: _Profiles,
    network: Network,
    edges: _Edges,
    options: dict[str, Any],
) -> Activities:
    """The record of the sources scored in at least one profile."""
    scored = np.flatnonzero(np.any(counts >= options["min_targets"], axis=0))
    if scored.size == 0:
        most = int(counts.max()) if counts.size else 0
        raise ValueError(
            f"no source of the network has {options['min_targets']} targets among "
            f"the genes of a profile (the most is {most}); lower min_targets, or "
            "check that the network and the data name genes the same way"
        )
    rows = np.concatenate(
        [edges.rows[edges.starts[s] : edges.starts[s + 1]] for s in scored.tolist()]
    )
    kept = Network(
        network.edges.take(rows),
        name=network.name,
        version=network.version,
        url=network.url,
        licence=network.licence,
        citation=network.citation,
        file=network.file,
        sha256=network.sha256,
    )
    p = p_values[:, scored]
    adjusted = adjust_p_values(p, axis=1) if p.size else p.copy()
    settings = {**options, **profiles.settings}
    result = Activities(
        read_only(np.ascontiguousarray(scores[:, scored])),
        read_only(np.ascontiguousarray(estimates[:, scored])),
        read_only(np.ascontiguousarray(p)),
        read_only(np.ascontiguousarray(adjusted)),
        read_only(np.ascontiguousarray(counts[:, scored])),
        profiles.profiles,
        kept,
        method,
        MappingProxyType(settings),
    )
    if profiles.cube is not None:
        keep_provenance(result, method, profiles.cube, settings)
    return result


def _two_sided(t_values: FloatArray, df: FloatArray) -> FloatArray:
    """Two-sided p-values of t-statistics on ``df`` degrees of freedom per row."""
    df_full = np.broadcast_to(df[:, np.newaxis], t_values.shape)
    usable = ~np.isnan(t_values) & (df_full > 0.0)
    p = np.full(t_values.shape, np.nan)
    if usable.any():
        tails = np.asarray(t_sf(np.abs(t_values[usable]), df_full[usable]))
        p[usable] = np.minimum(2.0 * tails, 1.0)
    return p


# ULM.


@jit(parallel=True)
def _ulm_parallel(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    min_targets: int,
    slopes: FloatArray,
    t_values: FloatArray,
    counts: IntArray,
    n_runs: int,
) -> None:
    n_profiles = values.shape[0]
    for run in prange(n_runs):
        for p in range(run * n_profiles // n_runs, (run + 1) * n_profiles // n_runs):
            _ulm_profile(
                values[p], starts, genes, weights, min_targets, slopes[p],
                t_values[p], counts[p],
            )  # fmt: skip


@jit
def _ulm_serial(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    min_targets: int,
    slopes: FloatArray,
    t_values: FloatArray,
    counts: IntArray,
) -> None:
    for p in range(values.shape[0]):
        _ulm_profile(
            values[p], starts, genes, weights, min_targets, slopes[p], t_values[p],
            counts[p],
        )  # fmt: skip


@jit(error_model="numpy")
def _ulm_profile(
    y: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    min_targets: int,
    slopes: FloatArray,
    t_values: FloatArray,
    counts: IntArray,
) -> None:
    """The slope and its t-statistic of every source in one profile.

    Over the n observed genes, with x the source's weights (zero off its targets):
    Sxx = sum (x - xbar)^2, from the targets and the n - k other genes, and
    Sxy = sum (x - xbar)(y - ybar) = sum_targets w (y - ybar) - xbar sum (y - ybar),
    whose last sum is zero but for rounding.
    """
    n = 0
    total = 0.0
    for g in range(y.size):
        if not math.isnan(y[g]):
            n += 1
            total += y[g]
    if n == 0:
        for s in range(starts.size - 1):
            counts[s] = 0
        return
    mean = total / n
    syy = 0.0
    residual = 0.0
    for g in range(y.size):
        if not math.isnan(y[g]):
            deviation = y[g] - mean
            syy += deviation * deviation
            residual += deviation
    df = n - 2
    for s in range(starts.size - 1):
        k = 0
        weight_sum = 0.0
        for e in range(starts[s], starts[s + 1]):
            if not math.isnan(y[genes[e]]):
                k += 1
                weight_sum += weights[e]
        counts[s] = k
        if k < min_targets or df < 1:
            continue
        x_mean = weight_sum / n
        sxx = (n - k) * x_mean * x_mean
        sxy = -x_mean * residual
        for e in range(starts[s], starts[s + 1]):
            value = y[genes[e]]
            if not math.isnan(value):
                deviation = weights[e] - x_mean
                sxx += deviation * deviation
                sxy += weights[e] * (value - mean)
        if not sxx > 0.0:
            continue
        slope = sxy / sxx
        slopes[s] = slope
        rss = max(syy - slope * sxy, 0.0)
        if rss > 0.0:
            t_values[s] = slope / math.sqrt(rss / (df * sxx))
        elif slope != 0.0:
            t_values[s] = math.copysign(math.inf, slope)


# MLM.


def _fit_pattern(
    values: FloatArray,
    observed: npt.NDArray[np.bool_],
    rows: list[int],
    edges: _Edges,
    min_targets: int,
    names: list[str],
    profiles: Table,
    coefficients: FloatArray,
    t_values: FloatArray,
    p_values: FloatArray,
    counts: IntArray,
) -> None:
    """Fit MLM to the profiles that observe the same genes."""
    genes = np.flatnonzero(observed[rows[0]])
    inside = np.concatenate(([0], np.cumsum(observed[rows[0]][edges.genes])))
    per_source = inside[edges.starts[1:]] - inside[edges.starts[:-1]]
    counts[rows] = per_source
    scored = np.flatnonzero(per_source >= min_targets)
    n = genes.size
    if scored.size == 0 or n == 0:
        return
    m = scored.size
    df = n - m - 1
    ids = [str(profiles[_PROFILE_ID][row]) for row in rows]
    if df < 1:
        raise ValueError(
            f"MLM fits {m} sources and an intercept to the {n} genes of profile(s) "
            f"{quoted(ids)}, which leaves no residual degrees of freedom; it needs "
            "more genes than sources: use ulm, or a network with fewer sources"
        )
    position = np.full(observed.shape[1], -1, dtype=np.int64)
    position[genes] = np.arange(n)
    design = np.zeros((n, m + 1))
    design[:, 0] = 1.0
    for column, s in enumerate(scored.tolist(), start=1):
        span = slice(edges.starts[s], edges.starts[s + 1])
        places = position[edges.genes[span]]
        present = places >= 0
        design[places[present], column] = edges.weights[span][present]
    q, r = np.linalg.qr(design)
    norms = np.linalg.norm(design, axis=0)
    aliased = np.flatnonzero(np.abs(np.diag(r)) <= RANK_TOLERANCE * norms)
    if aliased.size:
        listed = [names[scored[j - 1]] for j in aliased.tolist() if j > 0]
        raise ValueError(
            f"MLM cannot tell the sources {listing([repr(s) for s in listed])} apart "
            "from the other sources: over the genes of profile(s) "
            f"{quoted(ids)}, their weights are linear combinations of the weights of "
            "sources before them in the network. Leave them out of the network "
            "(Network.select), or use ulm, which scores each source alone"
        )
    y = values[np.ix_(rows, genes)].T
    effects = q.T @ y
    estimates = np.linalg.solve(r, effects)
    residuals = y - q @ effects
    sigma2 = np.einsum("ij,ij->j", residuals, residuals) / df
    inverse = np.linalg.solve(r, np.eye(m + 1))
    unscaled = np.einsum("ij,ij->i", inverse, inverse)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = estimates / np.sqrt(np.outer(unscaled, sigma2))
    coefficients[np.ix_(rows, scored)] = estimates[1:].T
    t_values[np.ix_(rows, scored)] = t[1:].T
    p_values[np.ix_(rows, scored)] = _two_sided(
        np.ascontiguousarray(t[1:].T), np.full(len(rows), float(df))
    )


# The weighted aggregate.


def _aggregate(
    values: FloatArray,
    edges: _Edges,
    order: IntArray,
    mean: bool,
    min_targets: int,
    out: FloatArray,
    counts: IntArray,
) -> None:
    """The weighted means or sums of the targets' values, the genes taking the
    values of ``order[0]``, the identity for the observed scores."""
    by_chunks(
        _aggregate_parallel,
        _aggregate_serial,
        (values, edges.starts, edges.genes, edges.weights, order[0], mean,
         min_targets, out, counts),
        values.shape[0],
    )  # fmt: skip


@jit(parallel=True)
def _aggregate_parallel(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    order: IntArray,
    mean: bool,
    min_targets: int,
    out: FloatArray,
    counts: IntArray,
    n_runs: int,
) -> None:
    n_profiles = values.shape[0]
    for run in prange(n_runs):
        for p in range(run * n_profiles // n_runs, (run + 1) * n_profiles // n_runs):
            _aggregate_profile(
                values[p], starts, genes, weights, order, mean, min_targets, out[p],
                counts[p],
            )  # fmt: skip


@jit
def _aggregate_serial(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    order: IntArray,
    mean: bool,
    min_targets: int,
    out: FloatArray,
    counts: IntArray,
) -> None:
    for p in range(values.shape[0]):
        _aggregate_profile(
            values[p], starts, genes, weights, order, mean, min_targets, out[p],
            counts[p],
        )  # fmt: skip


@jit
def _aggregate_profile(
    y: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    order: IntArray,
    mean: bool,
    min_targets: int,
    out: FloatArray,
    counts: IntArray,
) -> None:
    for s in range(starts.size - 1):
        k = 0
        total = 0.0
        size = 0.0
        for e in range(starts[s], starts[s + 1]):
            if not math.isnan(y[genes[e]]):
                k += 1
                value = y[order[genes[e]]]
                if not math.isnan(value):
                    total += weights[e] * value
                    size += abs(weights[e])
        counts[s] = k
        if k >= min_targets and size > 0.0:
            out[s] = total / size if mean else total


def _permuted(
    values: FloatArray,
    edges: _Edges,
    mean: bool,
    min_targets: int,
    estimates: FloatArray,
    n_permutations: int,
    generator: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """The scores normalized by those of random permutations of the genes, and the
    permutation p-values.

    Each permutation shuffles the positions of the genes, as decoupler's ``_ridx``
    shuffles each row of a tiled range with numpy's ``Generator.shuffle``, so that the
    same seed gives decoupler's permutations.
    """
    n_profiles, n_genes = values.shape
    n_sources = edges.starts.size - 1
    # For each profile and source: permutations counted, their running mean and sum
    # of squared deviations (Welford 1962), and those at least and at most as high.
    seen = np.zeros((n_profiles, n_sources))
    running = np.zeros((n_profiles, n_sources))
    squares = np.zeros((n_profiles, n_sources))
    higher = np.zeros((n_profiles, n_sources))
    lower = np.zeros((n_profiles, n_sources))
    done = 0
    while done < n_permutations:
        size = min(_BLOCK, n_permutations - done)
        block = np.tile(np.arange(n_genes, dtype=np.int64), (size, 1))
        for row in block:
            generator.shuffle(row)
        by_chunks(
            _permutations_parallel,
            _permutations_serial,
            (values, edges.starts, edges.genes, edges.weights, block, mean,
             min_targets, estimates, seen, running, squares, higher, lower),
            n_profiles,
        )  # fmt: skip
        done += size
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(squares / (seen - 1.0))
        scores = np.where(sd > 0.0, (estimates - running) / sd, np.nan)
        p_values = np.minimum(
            1.0, 2.0 * (1.0 + np.minimum(higher, lower)) / (1.0 + seen)
        )
    p_values[np.isnan(estimates)] = np.nan
    scores[np.isnan(estimates)] = np.nan
    return scores, p_values


@jit(parallel=True)
def _permutations_parallel(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    block: IntArray,
    mean: bool,
    min_targets: int,
    estimates: FloatArray,
    seen: FloatArray,
    running: FloatArray,
    squares: FloatArray,
    higher: FloatArray,
    lower: FloatArray,
    n_runs: int,
) -> None:
    n_profiles = values.shape[0]
    for run in prange(n_runs):
        for p in range(run * n_profiles // n_runs, (run + 1) * n_profiles // n_runs):
            _permutations_profile(
                values[p], starts, genes, weights, block, mean, min_targets,
                estimates[p], seen[p], running[p], squares[p], higher[p], lower[p],
            )  # fmt: skip


@jit
def _permutations_serial(
    values: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    block: IntArray,
    mean: bool,
    min_targets: int,
    estimates: FloatArray,
    seen: FloatArray,
    running: FloatArray,
    squares: FloatArray,
    higher: FloatArray,
    lower: FloatArray,
) -> None:
    for p in range(values.shape[0]):
        _permutations_profile(
            values[p], starts, genes, weights, block, mean, min_targets,
            estimates[p], seen[p], running[p], squares[p], higher[p], lower[p],
        )  # fmt: skip


@jit
def _permutations_profile(
    y: FloatArray,
    starts: IntArray,
    genes: IntArray,
    weights: FloatArray,
    block: IntArray,
    mean: bool,
    min_targets: int,
    estimates: FloatArray,
    seen: FloatArray,
    running: FloatArray,
    squares: FloatArray,
    higher: FloatArray,
    lower: FloatArray,
) -> None:
    """Add a block of permutations to the counts of one profile's sources."""
    for b in range(block.shape[0]):
        order = block[b]
        for s in range(starts.size - 1):
            observed = estimates[s]
            if math.isnan(observed):
                continue
            total = 0.0
            size = 0.0
            for e in range(starts[s], starts[s + 1]):
                if not math.isnan(y[genes[e]]):
                    value = y[order[genes[e]]]
                    if not math.isnan(value):
                        total += weights[e] * value
                        size += abs(weights[e])
            if not size > 0.0:
                continue
            score = total / size if mean else total
            seen[s] += 1.0
            delta = score - running[s]
            running[s] += delta / seen[s]
            squares[s] += delta * (score - running[s])
            if score >= observed:
                higher[s] += 1.0
            if score <= observed:
                lower[s] += 1.0
