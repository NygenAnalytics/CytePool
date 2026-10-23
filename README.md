# CytePool

CytePool is a Python library for statistics on aggregated omics data: pseudobulk
profiles, built by pooling the cells of each sample and cell type, and ordinary bulk
RNA-seq. It brings the sample-level methods that are standard in R, such as linear
models with empirical Bayes moderation, negative binomial models, mixed models,
gene-set tests, composition tests, dose-response fits and meta-analysis, into one
package with one data model.

**Status:** early development. Nothing is released yet.

## Principles

- **Samples are the unit of replication.** Cells are pooled into samples before any
  test, and the number of cells behind every pooled profile travels with the results.
- **Two required dependencies:** numpy and numba.
- **Checked against the reference implementations.** Every method is compared with
  the package that defined it. The comparisons live in [`validation/`](validation/),
  and their results are reported with each change.
- **Implemented from the literature.** Methods whose reference packages carry
  copyleft licenses are written from their papers and documentation, never from their
  source code, so CytePool can stay BSD-licensed.

## Development

```bash
uv sync --extra test --group dev
uv run pytest
uv run ruff check cytepool tests validation
uv run ruff format --check cytepool tests validation
uv run mypy cytepool
```

## License

BSD 3-Clause. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
