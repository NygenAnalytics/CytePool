---
description: Install CytePool from PyPI or its repository with Python 3.13 or newer, set up a development environment, build this documentation, and make a release.
---

(installation)=
# Installation

Install CytePool with uv or pip on Python 3.13 or newer. numpy and numba are CytePool's only required dependencies and are installed automatically.

## With uv

[uv](https://docs.astral.sh/uv/) creates an isolated virtual environment:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install cytepool
python -c "import cytepool; print(cytepool.__version__)"
```

## With pip

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install cytepool
python -c "import cytepool; print(cytepool.__version__)"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Optional integrations

Install optional integrations with CytePool's extras:

- `anndata` or `mudata`: converts cubes to and from AnnData and MuData ({api}`to_anndata`, {api}`from_anndata`, {api}`to_mudata`, {api}`from_mudata`); `mudata` installs `anndata` too.
- `plot`: installs matplotlib for the figures in `cytepool.plotting`.

```bash
uv pip install "cytepool[mudata,plot]"
```

CytePool imports these packages only when needed, and tells you which extra to install if one is missing.

## First calls and threads

numba compiles numerical routines the first time they run, which takes a few seconds per routine. The compiled code is cached on disk and reused across sessions (under `NUMBA_CACHE_DIR`, `$XDG_CACHE_HOME/cytepool`, `~/.cache/cytepool` or the temporary directory).

Parallel routines (`from_cells`, weighted linear models, negative binomial and DESeq2 fits, rotation gene-set tests, GSEA, pathway scores, activities, and mixed models) use numba's thread pool, and give identical results regardless of thread count. To limit threads on a shared machine, set `NUMBA_NUM_THREADS` before starting Python, or pass `n_threads` to functions that accept a thread count.

:::{note} Multiprocessing with OpenMP
With numba's OpenMP threading layer, a process that has run a parallel routine cannot `fork` worker processes that run one again: start `multiprocessing` workers with the `"spawn"` or `"forkserver"` start method instead, or run with one thread (`n_threads=1` or `numba.set_num_threads(1)`).
:::

## Contributing and development

Clone the repository and install the package with its test and development tools:

```bash
git clone https://github.com/NygenAnalytics/CytePool.git
cd CytePool
uv sync --extra test --group dev
uv run pytest
```

On Python 3.15, h5py and numcodecs do not have wheels yet, so the `test` extra omits anndata and mudata and skips their conversion tests. See the [README](https://github.com/NygenAnalytics/CytePool#development) for the lint, format and type checks. The validation scripts compare CytePool against R and the reference Python packages, which must be installed separately; see [reproducing the validation](#reproduce-validation).

(build-the-documentation)=
### Build this documentation

The site is built with [MyST](https://mystmd.org) and the API reference with Sphinx. MyST needs [Node.js](https://nodejs.org) 20 or newer. CI uses Node.js 24. From the repository root:

```bash
uv sync --extra test --group dev --group docs
make -C docs html
```

The first build downloads MyST's book theme, at the version pinned in `docs/source/myst.yml`, so it needs network access; later builds reuse it. The site is written to `docs/build/site/_build/html/`; open its `index.html` through a local web server, such as `python -m http.server -d docs/build/site/_build/html`. `make -C docs api` builds only the API reference, into `docs/build/api/`.

Building never runs code. Pages with code are notebooks (`.ipynb`) saved with their outputs. After changing the code of a notebook, execute it again and commit it with its new outputs:

```bash
make -C docs execute PAGES="quickstart"
```

This runs the notebook in a fresh kernel and records in its metadata when it ran, how long it took, a fingerprint of its code cells and the version of CytePool that ran it. The build, and the tests in `tests/test_documentation.py`, fail when a notebook's code no longer matches its fingerprint or a saved cell has an error. Prose can change without executing again. Each figure a notebook shows needs a description, in the code cell's `figure_descriptions` metadata, which becomes its alternative text.

A change to the package can change what a notebook prints without changing its code. `make -C docs check-outputs PAGES="quickstart"` runs a notebook without saving it and fails when its printed text differs from the saved outputs, other than in the version of CytePool, which changes with every commit; the documentation workflow runs it for the pages that need no downloaded data. Execute such a page again after changing the methods it uses.

To add a page, put a Markdown file or a notebook under `docs/source/` and add it to the `toc` in `docs/source/myst.yml`; execute it if it is a notebook. To document a new public module, add a page for it under `docs/source/reference/api/` with an `automodule` directive, and list the page in that directory's `index.md`.

Pages link to the API reference with the `{api}` role, as in ``{api}`limma_voom` `` or ``{api}`Cube.select` ``: the build replaces it with a link to the entry, and stops on a name that the API reference does not document or that could be several entries. Each API entry lists the pages that use and explain it. Pages cite works as `[@law2014]`, with keys of `docs/source/references.bib`; an unknown key stops the build, and the [Bibliography](reference/bibliography.md) cites every entry. A notebook that reads shared data names its datasets in its metadata `datasets` ([Datasets](reference/datasets.md)); `make -C docs execute` prepares them. `uv run --group docs python docs/coverage_report.py` lists the public names that no page uses or explains yet.

(make-a-release)=
### Make a release

To release a version, check that the Tests run of the commit on master passed, then draft a new release on GitHub, create its tag with that commit as the target, and publish the release. The tag is the version, or a release candidate such as `0.2.0rc1`, which PyPI lists as a pre-release. Publishing the release starts the [release workflow](https://github.com/NygenAnalytics/CytePool/blob/master/.github/workflows/publish.yml), which builds CytePool from the commit that the tag names and uploads it to PyPI once all of these pass:

- the latest run of the Tests workflow for that commit, started by a push or by hand (a pull request's run tests a merge commit instead), passed;
- the documentation builds, as the Docs workflow builds it;
- the wheel and the source distribution, built from a clean checkout of the commit with the build tools pinned in `.github/build-constraints.txt`, name the version of the tag, declare what `pyproject.toml` declares, and hold the package and no other file of the repository (`tests/validate_release_artifacts.py`);
- the wheel, installed in a clean environment on Linux with Python 3.13, 3.14 and 3.15 and on Windows and macOS with Python 3.13, pools a small dataset into a cube, gives limma's tables from `limma_voom`, and loads its compiled routines from the cache in a second run (`tests/smoke_test_wheel.py`).

When the `pypi` environment has required reviewers, one of them approves the upload after the checks. When a check fails, nothing is uploaded: fix the cause on master, delete the release and its tag, and publish the release again with the fixed commit as the target. Running the failed workflow again checks the same commit, which helps only when the failure came from outside it, such as a network error.

`uv run pytest` builds the two files and checks them as the workflow does. To run the smoke test on a wheel you build:

```bash
uv build --wheel --clear --out-dir dist/
uv run --no-project python tests/smoke_test_wheel.py dist/cytepool-*.whl
```

#### Trusted publishing setup

The repository uses PyPI trusted publishing and a protected GitHub environment:

1. On PyPI, at <https://pypi.org/manage/account/publishing/>, the publisher is configured for GitHub with the project name `cytepool`, the owner `NygenAnalytics`, the repository `CytePool`, the workflow `publish.yml` and the environment `pypi`.
2. On GitHub, in the repository's settings, the environment `pypi` lists the required reviewers and restricts *Deployment branches and tags* to the tag rule `*`, so that only a run for a tag can upload.
