# Repository instructions

These instructions apply to the whole repository.

## Layout

- Package code is in `cytepool/`, tests in `tests/`, reference comparisons in
  `validation/`, and the test data those comparisons write in `tests/data/`.
- Name modules, functions and files after what they compute, in terms a statistician or
  biologist would search for. Avoid project-internal labels.

## Conventions

- Python 3.13 or newer. Use `uv` for every Python command and dependency change.
- The package depends only on numpy and numba. Optional integrations go behind extras
  and are imported where they are used.
- Use built-in generics and `X | None`. Do not add `typing.List`, `typing.Optional`, or
  `from __future__ import annotations`.
- Validate arguments at the public boundary and raise errors that say what was wrong and
  what is expected.

## Writing

These rules apply to everything people read: pages, the README, docstrings, error
messages, figure labels, commit messages and pull-request descriptions.

- Start with the problem, explain the solution, then show its value. Lead with the
  reader's task: "Test which genes change..." rather than "A library for testing...".
- Give numbers context, round them where more digits add nothing, and keep the
  precision that statistics need. Keep caveats in supporting detail: an opening stays
  accurate, but conditions, versions and limits do not dominate it.
- Use plain words, short sentences and one term for one thing, and explain a
  statistic where the reader meets it.
- No em dashes, and no en dashes used as punctuation; hyphenated words and terms such
  as p-value are fine. No all-caps or labelled headings, no decorative numbering, no
  centred dots between words.
- No stock metaphors, such as "X lives in Y", "the X behind it" or "under the hood",
  and no promotional claims: say what something does or where it is stored.

## Documentation

- Each fact has one home; other places give it in a clause and link to it. The
  chapters of "Understand the methods" (`docs/source/statistics/`) explain the methods;
  the Validation page gives the agreement with the reference tools, with the numbers
  the validation scripts print; docstrings define behaviour; the guides show tasks on
  data; and "Choosing a method" lists the questions, their defaults and what is not
  implemented yet.
- A change that adds, removes or limits a capability updates the summary of the
  Validation page and "Choosing a method".
- Link a section by its label, `(name)=`, rather than by its heading, and give a
  docstring the address of the chapter section that explains it.

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
uv sync --extra test --group dev --group docs
uv run pytest
uv run ruff check cytepool tests validation docs
uv run ruff format --check cytepool tests validation docs
uv run mypy cytepool
make -C docs api
```
