---
description: Install CytePool from its repository with Python 3.13 or newer, set up a development environment, and build this documentation.
---

(installation)=
# Installation

CytePool needs Python 3.13 or newer. Its only required dependencies, numpy and numba,
are installed with it. Nothing is released on PyPI yet, so install it from the
repository.

## With uv

[uv](https://docs.astral.sh/uv/) creates an isolated environment, so CytePool does not
change the system Python:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "git+https://github.com/NygenAnalytics/CytePool.git"
python -c "import cytepool; print(cytepool.__version__)"
```

## With pip

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/NygenAnalytics/CytePool.git"
python -c "import cytepool; print(cytepool.__version__)"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Optional integrations

Converting cubes to and from AnnData and MuData (`cytepool.io.anndata`) needs anndata,
and mudata for MuData. Install them with CytePool's `anndata` or `mudata` extra:

```bash
uv pip install "cytepool[mudata] @ git+https://github.com/NygenAnalytics/CytePool.git"
```

The `mudata` extra installs anndata too. CytePool imports them only when a
conversion runs, and says which extra to install when one is missing.

## First calls and threads

Numerical routines are compiled by numba the first time they run, which takes up to
a few seconds per routine; the machine code is then cached on disk and reused by later
sessions. The cache is a `cytepool` directory under `NUMBA_CACHE_DIR` when that is set,
else under the user cache directory (`$XDG_CACHE_HOME` or `~/.cache`), else under the
temporary directory. Any change to the installed CytePool source starts a new cache.

Parallel routines use numba's threads: `from_cells`, the fits of linear models in
which each gene needs its own fit (with observation weights, as from voom, or missing
values), which `limma_voom` runs, and the fits of negative binomial models. Set
`NUMBA_NUM_THREADS` before Python starts to cap them, for example on a shared machine,
or pass `n_threads` to a function that takes it; results do not depend on the number
of threads. With numba's OpenMP threading layer, a process that has run a parallel
routine cannot fork worker processes that run one again: start workers with the
"spawn" or "forkserver" method of `multiprocessing` instead. With one thread
(`n_threads=1`, or `numba.set_num_threads(1)`), the fits of linear and negative
binomial models run without a parallel region, so forked workers and several Python
threads can run them at once; `from_cells` always starts one.

## Development setup

Clone the repository and install the package with its test and development tools:

```bash
git clone https://github.com/NygenAnalytics/CytePool.git
cd CytePool
uv sync --extra test --group dev
uv run pytest
```

The [README](https://github.com/NygenAnalytics/CytePool#development) lists the lint,
format and type checks a change must pass. The validation scripts, which compare
CytePool with R and the reference Python packages, need those packages installed
separately; see [reproducing the validation](#reproduce-validation).

(build-the-documentation)=
## Build this documentation

The site is built with [MyST](https://mystmd.org) and the API reference with Sphinx.
MyST needs [Node.js](https://nodejs.org) 18, 20, or 22 and newer; the pinned MyST
release is the last that runs on Node.js 18. From the repository root:

```bash
uv sync --extra test --group dev --group docs
make -C docs html
```

The first build downloads MyST's book theme, at the version pinned in
`docs/source/myst.yml`, so it needs network access; later builds reuse it. The site is
written to `docs/build/site/_build/html/`; open its `index.html` through a local web
server, such as `python -m http.server -d docs/build/site/_build/html`.
`make -C docs api` builds only the API reference, into `docs/build/api/`.

Building never runs code. Pages with code are notebooks (`.ipynb`) saved with their
outputs. After changing the code of a notebook, execute it again and commit it with
its new outputs:

```bash
make -C docs execute PAGES="quickstart"
```

This runs the notebook in a fresh kernel and records in its metadata when it ran, how
long it took, a fingerprint of its code cells and the version of CytePool that ran it.
The build, and the tests in
`tests/test_documentation.py`, fail when a notebook's code no longer matches its
fingerprint or a saved cell has an error. Prose can change without executing again.
Each figure a notebook shows needs a description, in the code cell's
`figure_descriptions` metadata, which becomes its alternative text.

A change to the package can change what a notebook prints without changing its code.
`make -C docs check-outputs PAGES="quickstart"` runs a notebook without saving it and
fails when its printed text differs from the saved outputs, other than in the version of
CytePool, which changes with every commit; the documentation workflow runs it for the
pages that need no downloaded data. Execute such a page again after changing the
methods it uses.

To add a page, put a Markdown file or a notebook under `docs/source/` and add it to the
`toc` in `docs/source/myst.yml`; execute it if it is a notebook. To document a new
public module, add a page for it under `docs/source/reference/api/` with an
`automodule` directive, and list the page in that directory's `index.md`.

Pages link to the API reference with the `{api}` role, as in ``{api}`limma_voom` `` or
``{api}`Cube.select` ``: the build replaces it with a link to the entry, and stops on a
name that the API reference does not document or that could be several entries. Each
API entry lists the pages that use and explain it. Pages cite works as `[@law2014]`,
with keys of `docs/source/references.bib`; an unknown key stops the build, and the
[Bibliography](reference/bibliography.md) cites every entry. A notebook that reads shared
data names its datasets in its metadata `datasets`
([Datasets](reference/datasets.md)); `make -C docs execute` prepares them.
`uv run --group docs python docs/coverage_report.py` lists the public names that no page
uses or explains yet.
