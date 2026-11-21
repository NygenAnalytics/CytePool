"""Analyses run again from what their results record, as permutation tests run them.

A result of :func:`~cytepool.limma_voom`, :func:`~cytepool.limma_trend` or
:func:`~cytepool.dream_voom` records the settings its analysis checked. A permutation
test runs the analysis again on permuted labels with the options read back from that
record, so that each run is the analysis of the result.
"""

import inspect
from collections.abc import Callable, Collection, Mapping
from typing import Any


def recorded_options(
    analysis: Callable[..., Any],
    settings: Mapping[str, Any],
    *,
    set_by_caller: Collection[str] = (),
) -> dict[str, Any]:
    """The keyword-only options of an analysis, read back from the settings that its
    result records, as the analysis takes them.

    An expression filter turned off is recorded as None and given back as False. The
    options in ``set_by_caller``, such as the groups or the contrasts of the run, are
    left out, for the caller to give.
    """
    parameters = inspect.signature(analysis).parameters
    options = {
        name: settings[name]
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in set_by_caller
        and name in settings
    }
    if options.get("expression_filter", True) is None:
        options["expression_filter"] = False  # recorded as None when it was off
    return options
