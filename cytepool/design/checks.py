"""Checks of a design for the problems that silently ruin an analysis.

``Design.check()`` returns a ``DesignReport``. Errors are problems that make some
coefficient, contrast or variance impossible to estimate; warnings point at designs
that can be fitted but deserve a second look. Each message says what is wrong and
how to fix it.

Errors:
    empty_design: the formula has no columns.
    duplicate_columns: two columns have the same name.
    rank_deficient: some columns are linear combinations of earlier ones (aliased),
        so their coefficients cannot be estimated. A column is aliased when its part
        orthogonal to the earlier columns has a norm at most 1e-7 times its own
        norm, the rule of the pivoted QR decomposition that R's ``lm.fit`` and
        limma's ``lmFit`` use, so the same columns are named.
    no_residual_df: as many independent columns as samples, so no variance can be
        estimated.
    constant_variable: a variable has the same value in every sample and so makes
        a column aliased (a warning when it does not, as in ``~ 0 + x``).
    empty_level: a factor level has no samples, which gives a column of zeros.
    confounded: two factors (main effects) group the samples so that one is
        constant within every level of the other, or a covariate is constant within
        every level of a factor, so their effects cannot be separated.
    one_sample_per_level: a factor gives every sample its own level.
    contrast_not_estimable: a contrast depends on effects the design cannot
        separate.
    random_single_group: a random-effect grouping with one group.
    random_unreplicated: a random effect with at least as many effects (groups
        times effect columns) as samples, lme4's rule, which includes one sample
        per group.
    random_confounded: the fixed effects already separate the groups of a random
        effect.
    random_duplicate: two random effects with the same effects and groupings that
        split the samples identically.

Warnings:
    single_sample_level: a factor level with a single sample, so no replicate.
    unreplicated_sample: a sample that the model fits exactly (leverage 1), so it
        tells nothing about the residual variance and the effects it determines
        rest on it alone.
    high_leverage: a sample with leverage above 0.99, nearly fitted exactly,
        because of a covariate value far beyond the others (beyond them by more
        than their range) or because the design has few residual degrees of
        freedom.
    numeric_codes: a numeric covariate whose values are a few consecutive whole
        numbers, each shared by several samples, which is how codes for batches or
        plates look; as a number it enters the model as one slope.
    missing_label: a factor level such as "" or "NA" that looks like a missing
        value, which then forms a group of its own.
    similar_levels: factor levels that differ only in case or surrounding spaces,
        which usually split one group in two.
"""

import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._linear_algebra import RANK_TOLERANCE, column_space
from ..table import Table
from .contrasts import Contrasts
from .formula import RandomEffect
from .model_matrix import INTERCEPT, ModelMatrix, _Covariate, _Factor

if TYPE_CHECKING:
    from .design import Design

__all__ = ["DesignIssue", "DesignReport"]

ESTIMABILITY_TOLERANCE = 1e-6
"""How far the terms of an estimable contrast may fail to cancel.

For each aliased column, the relation it satisfies gives a null vector ``nu`` of the
design, and a contrast ``c`` is estimable when ``c . nu = 0`` for every relation.
The terms ``c_k * nu_k`` of that product do not depend on the units of the
covariates, and for an estimable contrast they cancel: ``|sum| <= 1e-6 * sum of
absolute values``. A contrast in the row space of a nearly aliased design leaves at
most about ``RANK_TOLERANCE`` of its terms, while a contrast that needs an aliased
coefficient leaves an uncancelled term, however small its weight on that column.
"""

HIGH_LEVERAGE = 0.99
"""Leverage above which a sample is reported as nearly fitted exactly."""

_EXACT_LEVERAGE = 1 - 1e-10  # leverages above this are 1 up to rounding
_EXACT_ALIAS = 1e-12  # relative remainders below this are rounding, not data
_ROUNDING = (
    1e-10  # relation coefficients this small relative to the largest are rounding
)
_LISTED = 6  # items listed in a message before "and n more"
_MISSING_LABELS = frozenset({"", "na", "n/a", "nan", "null", "<na>", "nat", "?"})


@dataclass(frozen=True)
class DesignIssue:
    """One finding of ``Design.check``.

    Attributes:
        kind: A short code for the kind of problem, such as ``"rank_deficient"`` or
            ``"confounded"`` (see ``cytepool.design.checks``).
        message: What is wrong and how to fix it.
    """

    kind: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class DesignReport:
    """The result of ``Design.check``: errors, warnings and a summary of the rank.

    Attributes:
        errors: Problems that make a coefficient, contrast or variance impossible
            to estimate.
        warnings: Designs that can be fitted but deserve a second look.
        n_samples: Number of samples.
        n_columns: Number of design columns.
        rank: Rank of the design matrix.
        aliased: Columns that are linear combinations of earlier columns, in column
            order: the columns whose coefficients R's ``lm.fit`` (and limma's
            ``lmFit``) reports as NA, found by the same pivoting rule.
        leverage: The leverage of each sample, the diagonal of the hat matrix
            ``X (X'X)^- X'``: 1 for a sample the model fits exactly.
    """

    errors: tuple[DesignIssue, ...]
    warnings: tuple[DesignIssue, ...]
    n_samples: int
    n_columns: int
    rank: int
    aliased: tuple[str, ...]
    leverage: tuple[float, ...]

    @property
    def residual_df(self) -> int:
        """Residual degrees of freedom, the number of samples minus the rank."""
        return self.n_samples - self.rank

    @property
    def ok(self) -> bool:
        """Whether the check found no errors."""
        return not self.errors

    def kinds(self) -> tuple[str, ...]:
        """The kinds of all findings, errors first."""
        return tuple(issue.kind for issue in self.errors + self.warnings)

    def __str__(self) -> str:
        summary = (
            f"Design check: {_count(self.n_samples, 'sample')}, "
            f"{_count(self.n_columns, 'column')}, rank {self.rank}, "
            f"{_count(self.residual_df, 'residual degree')} of freedom"
        )
        if not self.errors and not self.warnings:
            return summary + "; no problems found."
        lines = [summary + "."]
        for title, issues in (("error", self.errors), ("warning", self.warnings)):
            if not issues:
                continue
            lines.append(f"{_count(len(issues), title)}:")
            for number, issue in enumerate(issues, start=1):
                lines.append(
                    textwrap.fill(
                        f"{number}. [{issue.kind}] {issue.message}",
                        width=88,
                        initial_indent="  ",
                        subsequent_indent="     ",
                        break_on_hyphens=False,
                        break_long_words=False,
                    )
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Aliasing:
    kept: tuple[int, ...]
    aliased: tuple[int, ...]
    relations: dict[int, np.ndarray]  # aliased column -> weights on the kept columns
    scaled_relations: dict[int, np.ndarray]  # the same with every column scaled
    remainders: dict[int, float]  # aliased column -> relative norm of its remainder
    basis: np.ndarray  # orthonormal basis of the column space, (n_samples, rank)
    scales: np.ndarray  # largest absolute value of each column (1 for a zero one)
    norms: np.ndarray  # norm of each scaled column

    @property
    def kept_norms(self) -> np.ndarray:
        return self.norms[list(self.kept)]


def check_design(design: "Design") -> DesignReport:
    """Run every check on a design; see ``Design.check``."""
    model = design.model_matrix
    samples = design.samples
    names = _sample_names(samples)
    errors: list[DesignIssue] = []
    warnings: list[DesignIssue] = []
    n, p = model.shape
    if p == 0:
        errors.append(
            DesignIssue(
                "empty_design",
                f"the formula {model.formula!r} gives no columns (no terms and no "
                "intercept); add terms or the intercept",
            )
        )
    duplicates = _duplicates(model.columns)
    if duplicates:
        errors.append(
            DesignIssue(
                "duplicate_columns",
                "more than one column is named "
                + _join(repr(name) for name in duplicates)
                + "; coefficients and contrasts refer to columns by name, so rename "
                "a variable or a level so that every column name is unique",
            )
        )
    aliasing = _aliasing(model.matrix)
    rank = len(aliasing.kept)
    if aliasing.aliased:
        errors.append(_rank_message(model, aliasing))
    if p > 0 and n - rank == 0:
        errors.append(
            DesignIssue(
                "no_residual_df",
                f"the design leaves no residual degrees of freedom: "
                f"{_count(n, 'sample')} and rank {rank}, so the model fits every "
                "sample exactly and no variance can be estimated or tested; remove "
                "terms from the formula or add samples",
            )
        )
    variables = design._variables
    single_sample = _variable_checks(
        model, variables, aliasing, names, errors, warnings
    )
    leverage = np.clip(np.sum(aliasing.basis**2, axis=1), 0.0, 1.0)
    if n - rank > 0:
        _leverage_check(variables, leverage, n - rank, names, single_sample, warnings)
    _contrast_checks(design, model, aliasing, errors)
    _random_checks(design, aliasing, errors)
    return DesignReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_samples=n,
        n_columns=p,
        rank=rank,
        aliased=tuple(model.columns[j] for j in aliasing.aliased),
        leverage=tuple(leverage.tolist()),
    )


def _aliasing(matrix: np.ndarray) -> _Aliasing:
    """Find the columns that are combinations of earlier columns, and how.

    ``column_space`` decides which columns are aliased, as R's ``qr()`` does; each
    aliased column is then expressed in the kept columns before it, on the scale of
    the original columns and with every column scaled to a largest absolute value
    of one.
    """
    space = column_space(matrix)
    scaled = matrix / space.scales
    kept = list(space.kept)
    relations, scaled_relations = {}, {}
    for j in space.aliased:
        weights = np.zeros(len(kept))
        if kept:
            weights = np.linalg.lstsq(scaled[:, kept], scaled[:, j], rcond=None)[0]
        scaled_relations[j] = weights
        with np.errstate(over="ignore"):  # only columns 1e300 apart overflow here
            relations[j] = weights * space.scales[j] / space.scales[kept]
    return _Aliasing(
        space.kept,
        space.aliased,
        relations,
        scaled_relations,
        {j: float(space.remainders[j]) for j in space.aliased},
        space.basis,
        space.scales,
        np.linalg.norm(scaled, axis=0),
    )


def _relation(model: ModelMatrix, aliasing: _Aliasing, j: int) -> str:
    """Describe how an aliased column depends on the kept columns."""
    columns = model.columns
    if not np.any(model.matrix[:, j]):
        return f"{columns[j]} is zero in every sample"
    text = f"{columns[j]} = " + _combination(
        aliasing.relations[j], [columns[k] for k in aliasing.kept]
    )
    remainder = aliasing.remainders[j]
    if remainder > _EXACT_ALIAS:
        column = model.matrix[:, j]
        spread = float(np.max(column) - np.min(column))
        covariates = [name for name, level in model.coding[j] if level is None]
        if covariates and spread <= 1e-6 * float(np.max(np.abs(column))):
            text += (
                f" up to {remainder:.2g} of its size: {_join(covariates)} hardly "
                "varies compared with its size, which R's QR also treats as aliased; "
                "centering it (subtracting its mean) makes it estimable"
            )
        else:
            text += (
                f" up to {remainder:.2g} of its size, which R's QR also treats as "
                "aliased: the data can hardly tell these columns apart, so remove or "
                "combine the nearly collinear variables"
            )
    return text


def _rank_message(model: ModelMatrix, aliasing: _Aliasing) -> DesignIssue:
    relations = [_relation(model, aliasing, j) for j in aliasing.aliased[:_LISTED]]
    more = len(aliasing.aliased) - len(relations)
    count = len(aliasing.aliased)
    text = (
        f"the design matrix has {_count(model.shape[1], 'column')} but rank "
        f"{len(aliasing.kept)}, so {_count(count, 'coefficient')} "
        f"{'is' if count == 1 else 'are'} not estimable. Aliased columns: "
        + "; ".join(relations)
        + (f"; and {more} more" if more else "")
        + ". The columns in each relation cannot be estimated separately; remove "
        "or recode the variables behind them (see the other findings), or drop "
        "terms from the formula"
    )
    return DesignIssue("rank_deficient", text)


def _combination(weights: np.ndarray, names: Sequence[str]) -> str:
    scale = float(np.max(np.abs(weights), initial=0.0))
    parts = []
    for weight, name in zip(weights.tolist(), names, strict=True):
        if abs(weight) <= 1e-8 * scale:
            continue
        sign = "-" if weight < 0 else "+"
        size = abs(weight)
        rounded = round(size)
        if abs(size - rounded) <= 1e-9 * max(1.0, size):
            size = rounded
        text = name if size == 1 else f"{size:.6g}*{name}"
        parts.append((sign, text))
    if not parts:
        return "0"
    first = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    return first + "".join(f" {sign} {text}" for sign, text in parts[1:])


def _variable_checks(
    model: ModelMatrix,
    variables: dict[str, _Factor | _Covariate],
    aliasing: _Aliasing,
    names: list[str],
    errors: list[DesignIssue],
    warnings: list[DesignIssue],
) -> set[int]:
    """Check each variable and each pair of main effects; return samples named."""
    main = [term[0] for term in _term_variables(model) if len(term) == 1]
    aliased_columns = set(aliasing.aliased)
    constant: set[str] = set()
    per_sample: set[str] = set()
    single_sample: set[int] = set()
    for name, variable in variables.items():
        makes_aliased = any(
            j in aliased_columns
            for j, coding in enumerate(model.coding)
            if any(part[0] == name for part in coding)
        )
        if isinstance(variable, _Covariate):
            distinct = np.unique(variable.values)
            if distinct.size == 1:
                constant.add(name)
                if makes_aliased:
                    errors.append(
                        DesignIssue(
                            "constant_variable",
                            f"{name} has the same value ({distinct[0]:g}) in every "
                            "sample, so its effect cannot be estimated; remove it "
                            "from the formula",
                        )
                    )
                else:
                    warnings.append(
                        DesignIssue(
                            "constant_variable",
                            f"{name} has the same value ({distinct[0]:g}) in every "
                            "sample, so its coefficients only rescale the means "
                            "they multiply and are not effects of "
                            f"{name}; remove it from the formula unless that is "
                            "intended",
                        )
                    )
            else:
                _numeric_codes_check(name, variable.values, distinct, warnings)
            continue
        counts = np.bincount(variable.codes, minlength=len(variable.levels))
        observed = [level for level, count in zip(variable.levels, counts) if count]
        if len(observed) == 1:
            # A factor has at least two levels, so the column of a level without
            # samples is zero and aliased: always an error.
            constant.add(name)
            errors.append(
                DesignIssue(
                    "constant_variable",
                    f"{name} is {observed[0]!r} in every sample, so it cannot have an "
                    "effect; remove it from the formula",
                )
            )
            continue
        empty = [level for level, count in zip(variable.levels, counts) if not count]
        if empty:
            zero_columns = [
                column
                for column, coding in zip(model.columns, model.coding, strict=True)
                if any(part == (name, level) for part in coding for level in empty)
            ]
            errors.append(
                DesignIssue(
                    "empty_level",
                    f"{name} has no samples at level{'' if len(empty) == 1 else 's'} "
                    + _join(repr(level) for level in empty)
                    + (
                        ", so "
                        + _join(zero_columns)
                        + " "
                        + ("is" if len(zero_columns) == 1 else "are")
                        + " zero in every sample"
                        if zero_columns
                        else ""
                    )
                    + "; remove unused levels from levels= (for a pandas "
                    "categorical, drop unused categories)",
                )
            )
        _label_checks(name, observed, warnings)
        if np.all(counts[counts > 0] == 1):
            per_sample.add(name)
            errors.append(
                DesignIssue(
                    "one_sample_per_level",
                    f"{name} has a different level for every sample, so it explains "
                    "each sample on its own and leaves nothing to estimate the other "
                    "effects or the residual variance with; remove it from the "
                    "formula. To account for repeated samples of a donor or batch, "
                    "the right model has that grouping as a random effect, such as "
                    "random='(1 | g)' for a grouping column g; "
                    + _random_intercept_alternatives(None),
                )
            )
            continue
        lonely = [k for k, count in enumerate(counts) if count == 1]
        if lonely:
            samples = [int(np.flatnonzero(variable.codes == k)[0]) for k in lonely]
            single_sample.update(samples)
            described = [
                f"{variable.levels[k]!r} ({names[sample]})"
                for k, sample in zip(lonely[:_LISTED], samples, strict=False)
            ]
            more = len(lonely) - len(described)
            warnings.append(
                DesignIssue(
                    "single_sample_level",
                    f"{name} has a single sample at level"
                    f"{'' if len(lonely) == 1 else 's'} "
                    + _join(described)
                    + (f" and {more} more" if more else "")
                    + "; effects of such a level rest on one sample with no "
                    "replicate. Add replicates, merge the level with a similar one, "
                    "or remove the sample",
                )
            )
    factors = [name for name in main if isinstance(variables[name], _Factor)]
    covariates = [name for name in main if isinstance(variables[name], _Covariate)]
    usable = [name for name in factors if name not in constant | per_sample]
    reported: set[frozenset[str]] = set()
    for first in usable:
        for second in usable:
            if first == second or frozenset((first, second)) in reported:
                continue
            outer, inner = variables[first], variables[second]
            assert isinstance(outer, _Factor) and isinstance(inner, _Factor)
            if not _constant_within(inner.codes, outer.codes):
                continue
            both = _constant_within(outer.codes, inner.codes)
            reported.add(frozenset((first, second)))
            errors.append(_confounded_factors(first, second, outer, inner, both))
    for factor in usable:
        grouping = variables[factor]
        assert isinstance(grouping, _Factor)
        for covariate in covariates:
            if covariate in constant:
                continue
            values = variables[covariate]
            assert isinstance(values, _Covariate)
            if _constant_within(values.values, grouping.codes):
                errors.append(
                    DesignIssue(
                        "confounded",
                        f"{covariate} has a single value within every level of "
                        f"{factor}, so the {covariate} slope cannot be separated from "
                        f"the {factor} effects; remove one of them from the formula",
                    )
                )
    return single_sample


def _label_checks(name: str, observed: list[str], warnings: list[DesignIssue]) -> None:
    """Warn about labels that look missing or that differ only in case or spaces."""
    suspicious = [
        level for level in observed if level.strip().lower() in _MISSING_LABELS
    ]
    if suspicious:
        warnings.append(
            DesignIssue(
                "missing_label",
                f"{name} has the level{'' if len(suspicious) == 1 else 's'} "
                + _join(repr(level) for level in suspicious)
                + ", which looks like a missing value and forms a group of its own; "
                "if these samples lack a label, fill it in or remove them (a real "
                "group with this name needs no change)",
            )
        )
    groups: dict[str, list[str]] = {}
    for level in observed:
        groups.setdefault(level.strip().casefold(), []).append(level)
    similar = [members for members in groups.values() if len(members) > 1]
    if similar:
        listed = "; ".join(_join(repr(level) for level in group) for group in similar)
        warnings.append(
            DesignIssue(
                "similar_levels",
                f"{name} has levels that differ only in case or surrounding spaces "
                f"({listed}), and each is a group of its own; if they mean the same "
                "group, clean the labels before building the design",
            )
        )


def _confounded_factors(
    first: str, second: str, outer: _Factor, inner: _Factor, both: bool
) -> DesignIssue:
    pairs: dict[str, list[str]] = {}
    for code_outer, code_inner in sorted(
        set(zip(outer.codes.tolist(), inner.codes.tolist()))
    ):
        pairs.setdefault(inner.levels[code_inner], []).append(outer.levels[code_outer])
    listing = "; ".join(
        f"{second} {level!r}: {first} " + _join(repr(value) for value in values)
        for level, values in list(pairs.items())[:_LISTED]
    )
    if both:
        return DesignIssue(
            "confounded",
            f"{first} and {second} group the samples identically ({listing}), so "
            "their effects cannot be separated by any analysis; keep one of them in "
            "the formula, knowing that its effect is also the effect of the other",
        )
    return DesignIssue(
        "confounded",
        f"every level of {first} holds a single level of {second} ({listing}), so "
        f"the effect of {second} cannot be separated from the {first} effects. Remove "
        f"{first} from the formula; if {first} groups repeated samples (donors, "
        f"batches), the right model has it as a random effect, random='(1 | {first})'; "
        + _random_intercept_alternatives(first),
    )


def _random_intercept_alternatives(group: str | None) -> str:
    """How a random intercept can be fitted: by REML, or with a known correlation."""
    if group is None:
        return (
            "fit_mixed_model estimates such a random intercept by REML; "
            "fit_linear_model fits one whose correlation is known, given block= (the "
            "group of each sample) and correlation=; and, when the conditions "
            "compared differ between groups, not within them, Cube.collapse_samples "
            "pools the repeated samples of each group into one"
        )
    return (
        "fit_mixed_model estimates this random intercept by REML; fit_linear_model "
        f"fits it with a known correlation, given block= (the {group} of each "
        f"sample) and correlation=; or pool the samples of each {group} with "
        f"Cube.collapse_samples({group!r})"
    )


def _numeric_codes_check(
    name: str, values: np.ndarray, distinct: np.ndarray, warnings: list[DesignIssue]
) -> None:
    """Warn about whole numbers 1, 2, 3, ... that look like batch or plate codes."""
    if distinct.size < 3 or not np.all(distinct == np.round(distinct)):
        return
    if distinct[-1] - distinct[0] + 1 != distinct.size:
        return  # not consecutive, such as ages or doses
    counts = np.unique(values, return_counts=True)[1]
    if np.any(counts < 2):
        return
    warnings.append(
        DesignIssue(
            "numeric_codes",
            f"{name} is numeric with the consecutive whole values "
            f"{distinct[0]:g} to {distinct[-1]:g}, each shared by several samples, "
            "which is how codes for batches or plates look; as a number it enters "
            "the model as one slope (a linear trend). If these numbers label groups, "
            "convert the column to text so that it becomes a factor",
        )
    )


def _leverage_check(
    variables: dict[str, _Factor | _Covariate],
    leverage: np.ndarray,
    residual_df: int,
    names: list[str],
    already_named: set[int],
    warnings: list[DesignIssue],
) -> None:
    exact = [
        int(sample)
        for sample in np.flatnonzero(leverage > _EXACT_LEVERAGE)
        if int(sample) not in already_named
    ]
    if exact:
        factors = [k for k, value in variables.items() if isinstance(value, _Factor)]
        described = []
        for sample in exact[:_LISTED]:
            labels = []
            for name in factors:
                factor = variables[name]
                assert isinstance(factor, _Factor)
                labels.append(f"{name} = {factor.levels[factor.codes[sample]]}")
            described.append(
                names[sample] + (f" ({', '.join(labels)})" if labels else "")
            )
        more = len(exact) - len(described)
        plural = len(exact) > 1
        warnings.append(
            DesignIssue(
                "unreplicated_sample",
                f"the model fits sample{'s' if plural else ''} "
                + _join(described)
                + (f" and {more} more" if more else "")
                + " exactly (leverage 1), so "
                + ("they tell" if plural else "it tells")
                + " nothing about the residual variance and the effects "
                + (
                    "they determine rest on them"
                    if plural
                    else "it determines rest on it"
                )
                + " alone. This happens when a level, or a combination of levels in an "
                "interaction, has too few samples for the terms that involve it; add "
                "replicates or simplify the formula",
            )
        )
    high = np.flatnonzero((leverage > HIGH_LEVERAGE) & (leverage <= _EXACT_LEVERAGE))
    if not high.size:
        return
    covariates = {
        name: value.values
        for name, value in variables.items()
        if isinstance(value, _Covariate)
    }
    extreme: list[str] = []
    crowded: list[str] = []
    for sample in high.tolist():
        values = [
            f"{name} = {values[sample]:g}, the others {np.delete(values, sample).min():g} "
            f"to {np.delete(values, sample).max():g}"
            for name, values in covariates.items()
            if _far_outside(values, sample)
        ]
        text = f"{names[sample]} (leverage {_leverage_text(leverage[sample])}"
        (extreme if values else crowded).append(
            text + "".join(f"; {v}" for v in values) + ")"
        )
    if extreme:
        warnings.append(
            DesignIssue(
                "high_leverage",
                "the model nearly fits "
                + _listing(extreme)
                + " exactly because of an extreme covariate value, so such a sample "
                "decides that covariate's slope almost alone and barely informs the "
                "residual variance. Check the value, or transform the covariate (for "
                "example to a log scale)",
            )
        )
    if crowded:
        warnings.append(
            DesignIssue(
                "high_leverage",
                "the model nearly fits "
                + _listing(crowded)
                + f" exactly: with {_count(residual_df, 'residual degree')} of "
                "freedom, such a sample largely decides some coefficients and barely "
                "informs the residual variance. Add samples or simplify the formula",
            )
        )


def _far_outside(values: np.ndarray, sample: int) -> bool:
    """Whether a sample's value lies beyond the others by more than their range."""
    others = np.delete(values, sample)
    if not others.size:
        return False
    low, high = float(others.min()), float(others.max())
    spread = high - low
    return bool(values[sample] < low - spread or values[sample] > high + spread)


def _listing(items: list[str]) -> str:
    more = len(items) - _LISTED
    return _join(items[:_LISTED]) + (f" and {more} more" if more > 0 else "")


def _contrast_checks(
    design: "Design", model: ModelMatrix, aliasing: _Aliasing, errors: list[DesignIssue]
) -> None:
    # The design's contrasts weigh its own columns in order: make_contrasts
    # matched them to the model's columns by name.
    contrasts = design.contrasts
    for position, relations in _contrast_violations(model, aliasing, contrasts).items():
        errors.append(
            DesignIssue(
                "contrast_not_estimable",
                f"contrast {contrasts.names[position]!r} is not estimable: it depends "
                "on effects that the design cannot separate ("
                + "; ".join(relations)
                + "). Fix the design first, or choose a contrast that does not "
                "separate the columns of these relations",
            )
        )


def _contrast_violations(
    model: ModelMatrix, aliasing: _Aliasing, contrasts: Contrasts
) -> dict[int, list[str]]:
    """The contrasts that the rows of a model matrix cannot estimate.

    A contrast c is estimable when it is orthogonal to the null space of the
    matrix, which the relations of the aliased columns span: c must weigh each
    aliased column as the kept columns it equals imply (see
    ``ESTIMABILITY_TOLERANCE``).

    Args:
        model: The model matrix; its ``matrix`` may be some of a design's rows.
        aliasing: The aliasing of ``model.matrix``, from ``_aliasing``.
        contrasts: Contrasts over the columns of ``model``, in their order.

    Returns:
        The position of each contrast that is not estimable, with the relations it
        violates as text (the first ``_LISTED`` of them).
    """
    if not len(contrasts) or not aliasing.aliased:
        return {}
    p = model.shape[1]
    nulls = [_null_vector(aliasing, j, p) for j in aliasing.aliased]
    violations = {}
    for position in range(len(contrasts)):
        weights = contrasts.matrix[:, position]
        violated = [
            index
            for index, null in enumerate(nulls)
            if _uncancelled(null, weights, aliasing.scales) > ESTIMABILITY_TOLERANCE
        ]
        if violated:
            violations[position] = [
                _relation(model, aliasing, aliasing.aliased[index])
                for index in violated[:_LISTED]
            ]
    return violations


def _null_vector(aliasing: _Aliasing, j: int, p: int) -> np.ndarray:
    """The null vector of an aliased column, in scaled coordinates.

    A relation is known only to within ``RANK_TOLERANCE`` of the aliased column's
    norm, so a kept column whose term in it is smaller than that has no part in it:
    its coefficient is rounding, or a fit to the remainder of a nearly aliased
    column. Such coefficients are set to zero, so that a contrast on that column is
    not taken for one that needs the relation.
    """
    kept = list(aliasing.kept)
    coefficients = aliasing.scaled_relations[j]
    sizes = np.abs(coefficients) * aliasing.kept_norms
    coefficients = np.where(
        sizes > RANK_TOLERANCE * aliasing.norms[j], coefficients, 0.0
    )
    vector = np.zeros(p)
    vector[j] = 1.0
    vector[kept] = -coefficients
    return vector


def _uncancelled(null: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> float:
    """The part of a contrast's terms on a relation that does not cancel.

    The terms ``c_k * n_k / s_k`` (weight, scaled null vector, column scale) are
    the terms of the contrast's product with the null vector in original units, the
    same in any units. Returns ``|sum| / sum of absolute values``, computed on a log
    scale so that no term overflows or underflows; 0 when the contrast has no weight
    on the columns of the relation.
    """
    active = (null != 0) & (weights != 0)
    if not active.any():
        return 0.0
    logs = (
        np.log(np.abs(null[active]))
        + np.log(np.abs(weights[active]))
        - np.log(scales[active])
    )
    terms = np.sign(null[active] * weights[active]) * np.exp(logs - logs.max())
    return float(abs(terms.sum()) / np.abs(terms).sum())


def _random_checks(
    design: "Design", aliasing: _Aliasing, errors: list[DesignIssue]
) -> None:
    n = design.n_samples
    parts = design._random_parts
    for effect, codes, effects in parts:
        group = effect.group
        n_groups, n_effects = int(codes.max()) + 1, effects.shape[1]
        if n_groups == 1:
            errors.append(
                DesignIssue(
                    "random_single_group",
                    f"the random effect {effect} has a single group (every sample "
                    f"has the same {group}), so its variance cannot be estimated; "
                    "remove it",
                )
            )
            continue
        # lme4 makes one term per column of a double-bar term.
        per_term = n_effects if effect.correlated else 1
        if n <= n_groups * per_term:
            errors.append(_unreplicated(effect, n, n_groups, per_term))
            continue
        absorbed = [
            name
            for name, column in zip(effects.columns, effects.matrix.T, strict=True)
            if _absorbed(column, codes, aliasing.basis)
        ]
        if absorbed:
            errors.append(_random_confounded(effect, absorbed, len(effects.columns)))
    for i, (first, first_codes, first_effects) in enumerate(parts):
        for second, second_codes, second_effects in parts[i + 1 :]:
            if _same_partition(first_codes, second_codes) and np.array_equal(
                first_effects.matrix, second_effects.matrix
            ):
                errors.append(
                    DesignIssue(
                        "random_duplicate",
                        f"the random effects {first} and {second} group the samples "
                        "identically, so their variances cannot be told apart; keep "
                        "one of them",
                    )
                )


def _unreplicated(
    effect: RandomEffect, n: int, n_groups: int, per_term: int
) -> DesignIssue:
    group = effect.group
    if per_term == 1:
        message = (
            f"every group of the random effect {effect} has a single sample, so the "
            f"{group} variance cannot be separated from the residual variance; remove "
            f"the random effect (it needs repeated samples per {group})"
        )
    else:
        message = (
            f"the random effect {effect} has {per_term} effects in each of "
            f"{n_groups} groups, {n_groups * per_term} random effects in all, but only "
            f"{_count(n, 'sample')}, so its variances cannot be separated from the "
            "residual variance (lme4 refuses this model); use a simpler random effect "
            f"such as (1 | {group}) or more samples per {group}"
        )
    return DesignIssue("random_unreplicated", message)


def _random_confounded(
    effect: RandomEffect, absorbed: list[str], n_effects: int
) -> DesignIssue:
    group = effect.group
    parts = [
        "random intercept" if name == INTERCEPT else f"random slope of {name}"
        for name in absorbed
    ]
    if len(absorbed) == n_effects:
        what = f"every effect of the random effect {effect}"
        fix = "drop the random effect"
    else:
        what = f"the {_join(parts)} of the random effect {effect}"
        fix = (
            "drop " + ("it" if len(parts) == 1 else "them") + " from the random effect"
        )
        if INTERCEPT in absorbed:
            fix += " (write 0 + on its left side to drop the intercept)"
    return DesignIssue(
        "random_confounded",
        f"the fixed effects already separate every group for {what} ({group} "
        "itself, or a variable that groups the samples the same way, is in the fixed "
        f"formula), so {'its variance' if len(absorbed) == 1 else 'their variances'} "
        f"cannot be estimated; remove {group} from the fixed formula or {fix}",
    )


def _absorbed(column: np.ndarray, codes: np.ndarray, basis: np.ndarray) -> bool:
    """Whether a random-effect column, within every group, is a fixed column."""
    for group in range(int(codes.max()) + 1):
        vector = np.where(codes == group, column, 0.0)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            continue
        remainder = vector - basis @ (basis.T @ vector)
        if float(np.linalg.norm(remainder)) > RANK_TOLERANCE * norm:
            return False
    return bool(np.any(column))


def _same_partition(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two group codings split the samples into the same groups."""
    pairs = len(set(zip(first.tolist(), second.tolist())))
    return pairs == len(set(first.tolist())) == len(set(second.tolist()))


def _term_variables(model: ModelMatrix) -> list[tuple[str, ...]]:
    """The variables of each term, recovered from the column coding."""
    terms: dict[int, tuple[str, ...]] = {}
    for index, coding in zip(model.assign, model.coding, strict=True):
        if index and index not in terms:
            terms[index] = tuple(name for name, _ in coding)
    return [terms[index] for index in sorted(terms)]


def _constant_within(values: np.ndarray, groups: np.ndarray) -> bool:
    """Whether ``values`` is constant within every group."""
    order = np.lexsort((values, groups))
    same_group = groups[order][1:] == groups[order][:-1]
    same_value = values[order][1:] == values[order][:-1]
    return bool(np.all(same_value[same_group]))


def _duplicates(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for column in columns:
        if column in seen and column not in repeated:
            repeated.append(column)
        seen.add(column)
    return repeated


def _sample_names(samples: Table) -> list[str]:
    if "sample_id" in samples and samples["sample_id"].dtype == object:
        return [str(value) for value in samples["sample_id"].tolist()]
    return [f"row {position}" for position in range(len(samples))]


def _leverage_text(value: float) -> str:
    """Write a leverage so that a value just below 1 does not read as 1."""
    return f"1 - {1 - value:.2g}" if value > 0.9999 else f"{value:.4f}"


def _count(number: int, noun: str) -> str:
    """Write a count with its noun, as in "1 sample" and "2 samples"."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _join(items: Iterable[object]) -> str:
    """Join items as "a, b and c"."""
    values = [str(item) for item in items]
    if len(values) <= 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]
