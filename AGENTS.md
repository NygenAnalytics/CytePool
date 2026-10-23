# Repository instructions

These instructions apply to the whole repository.

## Layout

- Package code lives in `cytepool/`, tests in `tests/`, reference comparisons in
  `validation/`, and the test data those comparisons write in `tests/data/`.
- Name modules, functions and files after what they compute, in terms a statistician or
  biologist would search for. Avoid project-internal labels.

## Conventions

- Python 3.13 or newer. Use `uv` for every Python command and dependency change.
- The package depends only on numpy and numba. Optional integrations go behind extras
  and are imported where they are used.
- Use built-in generics and `X | None`. Do not add `typing.List`, `typing.Optional`, or
  `from __future__ import annotations`.
- Avoid em dashes in public prose, docstrings and comments.
- Validate arguments at the public boundary and raise errors that say what was wrong and
  what is expected.

## Numerical routines

- Compile numerical loops with `cytepool._jit.jit`. It caches machine code in a directory
  keyed by the package source, so a change to any routine invalidates every cache.
- Never call `numba.get_num_threads()` inside compiled code; numba then stops caching
  the function.
- Write loops over contiguous memory, and keep accumulators in float64.

## Licensing

- Do not copy, translate or port code from GPL, LGPL, AGPL, Artistic, MPL or
  non-commercial projects, or from projects without a license, and do not read their
  source to match their implementation. Implement methods from papers and
  documentation, and validate against outputs produced by running the upstream tool.
- Record material adapted from permissively licensed projects, and the source and
  license of every dataset used by tests or validation, in `THIRD_PARTY_NOTICES.md`.

## Tests and validation

- Tests must be fast and must check results against something independent of the code
  under test: reference outputs saved by a validation script, values derived by hand,
  or invariants that a wrong implementation would break. A test that repeats the
  implementation's formula proves nothing.
- Every public function has a check in `validation/` against its reference
  implementation. Each validation script is self-contained, prints its comparison
  table, and writes the small fixtures the tests use with `--write-fixtures`.

## Checks

```bash
uv sync --extra test --group dev
uv run pytest
uv run ruff check cytepool tests validation
uv run ruff format --check cytepool tests validation
uv run mypy cytepool
```
