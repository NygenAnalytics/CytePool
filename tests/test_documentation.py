"""Checks on the documentation sources in docs/: pages, notebooks, links, API,
validation, the bibliography, the datasets and the coverage of the public names.

The checks read the sources and do not build the site. Reading the table of contents
needs PyYAML, a documentation dependency (``uv sync --group docs``); the tests that need
it are skipped without it. The checks of the bibliography read the References sections
of the package's module docstrings, from the source files.
"""

import ast
import copy
import gzip
import importlib
import importlib.util
import inspect
import io
import json
import pkgutil
import re
import sys
import textwrap
import tokenize
import urllib.parse
from collections import defaultdict
from collections.abc import Callable, Iterator
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest

import cytepool

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"
API_SOURCE = SOURCE / "reference" / "api"
REPOSITORY = "https://github.com/NygenAnalytics/CytePool/"
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
DIRECTIVE = re.compile(r"^\.\. (automodule|autofunction|autoclass|py:data)::\s+(\S+)")
OPTION = re.compile(r"^\s+:([\w-]+):\s*(.*)$")
LABEL = re.compile(r"^\((\S+)\)=$", re.M)

# The scripts of docs/ import page_index by name; appended, so that it never takes
# the place of an installed package.
sys.path.append(str(ROOT / "docs"))
import page_index  # noqa: E402


def _load_build_docs() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_docs", ROOT / "docs" / "build_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_docs = _load_build_docs()


def _site_pages() -> list[Path]:
    """The MyST pages: Markdown and notebooks outside the Sphinx API sources."""
    return sorted(
        path
        for path in SOURCE.rglob("*")
        if path.suffix in {".md", ".ipynb"}
        and not path.is_relative_to(API_SOURCE)
        and not {"_build", ".ipynb_checkpoints"} & set(path.relative_to(SOURCE).parts)
    )


def _notebooks() -> list[Path]:
    return [path for path in _site_pages() if path.suffix == ".ipynb"]


def _text(path: Path) -> str:
    """The prose of a page: a Markdown file, or the Markdown cells of a notebook."""
    if path.suffix != ".ipynb":
        return path.read_text()
    cells = json.loads(path.read_text())["cells"]
    return "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )


def _toc_files(entries: list[dict[str, Any]]) -> Iterator[str]:
    for entry in entries:
        if "file" in entry:
            yield entry["file"]
        yield from _toc_files(entry.get("children", []))


# What the API reference must document.


def _public_names(module: ModuleType) -> list[str]:
    """``__all__``, or else the public functions and classes a module defines."""
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and value.__module__ == module.__name__
    ]


def _modules() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Every module whose name has no private component, with its package."""
    for info in pkgutil.walk_packages(cytepool.__path__, "cytepool."):
        if not any(part.startswith("_") for part in info.name.split(".")):
            module = importlib.import_module(info.name)
            yield module, importlib.import_module(info.name.rpartition(".")[0])


def _is_implementation(module: ModuleType, package: ModuleType) -> bool:
    """Whether a package re-exports names of a module, which implements it."""
    exported = set(getattr(package, "__all__", ()))
    return package is not cytepool and bool(set(_public_names(module)) & exported)


def _public_modules() -> set[str]:
    """The modules users import from: modules with public names that do not just
    implement their package. The top level holds only ``__version__``."""
    return {
        module.__name__
        for module, package in _modules()
        if _public_names(module) and not _is_implementation(module, package)
    }


def _required_objects() -> set[str]:
    """Every public name, as the API reference must name it: the names of the public
    modules, and names an implementation module exports but its package does not."""
    public = _public_modules()
    required = set()
    for module, package in _modules():
        exported = set(getattr(package, "__all__", ()))
        implementation = _is_implementation(module, package)
        for name in _public_names(module):
            if module.__name__ in public or (implementation and name not in exported):
                required.add(f"{module.__name__}.{name}")
    return required


# What the API reference documents, named as Sphinx names it.


def _directives(page: Path) -> Iterator[tuple[str, str, dict[str, str]]]:
    """The autodoc and Python directives of an API page, with their options."""
    lines = page.read_text().splitlines()
    for number, line in enumerate(lines):
        if match := DIRECTIVE.match(line):
            options = {}
            for following in lines[number + 1 :]:
                if not (option := OPTION.match(following)):
                    break
                options[option[1]] = option[2]
            yield match[1], match[2], options


def _members(name: str, value: Any) -> set[str]:
    """What autodoc documents of a function or class: the object, if it has a
    docstring, and the methods and properties of a class that have one, other
    than the field aliases of named tuples, which docs/source/reference/api/conf.py
    leaves out."""
    if not (inspect.isfunction(value) or inspect.isclass(value)):
        return set()
    found = {name} if inspect.getdoc(value) else set()
    if inspect.isclass(value):
        for attribute, member in vars(value).items():
            routine = inspect.isfunction(member) or isinstance(
                member, property | classmethod | staticmethod
            )
            doc = inspect.getdoc(member) if routine else None
            alias = (doc or "").startswith("Alias for field number")
            if not attribute.startswith("_") and doc and not alias:
                found.add(f"{name}.{attribute}")
    return found


def _documented_objects() -> dict[str, set[str]]:
    """The full names of the objects each API page documents.

    ``automodule`` documents the module's public names that have docstrings and the
    methods and properties of its classes. ``autofunction`` and ``autoclass`` take
    full names. ``py:data`` names an object in the module that the page documents
    last, or in its ``:module:`` option.
    """
    pages = {}
    for page in API_SOURCE.glob("*.md"):
        documented: set[str] = set()
        current = None
        for kind, name, options in _directives(page):
            if kind == "automodule":
                current = name
                module = importlib.import_module(name)
                documented.add(name)
                for member in _public_names(module):
                    documented |= _members(f"{name}.{member}", getattr(module, member))
            elif kind.startswith("auto"):
                module_name, _, attribute = name.rpartition(".")
                value = getattr(importlib.import_module(module_name), attribute)
                documented |= _members(name, value)
            else:
                prefix = options.get("module", current)
                documented.add(f"{prefix}.{name}" if prefix else name)
        pages[page.name] = documented
    return pages


def test_every_page_in_the_table_of_contents_exists() -> None:
    yaml = pytest.importorskip("yaml")
    project = yaml.safe_load((SOURCE / "myst.yml").read_text())["project"]
    listed = list(_toc_files(project["toc"]))

    assert len(listed) == len(set(listed)), "a page is listed twice"
    assert [page for page in listed if not (SOURCE / page).is_file()] == []
    # Every page is reachable: none is left out of the table of contents.
    pages = {path.relative_to(SOURCE).as_posix() for path in _site_pages()}
    assert pages == set(listed)
    assert {"index.md", "quickstart.ipynb", "validation.md"} <= set(listed)


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_every_notebook_is_executed_unchanged_and_without_errors(path: Path) -> None:
    notebook = json.loads(path.read_text())

    build_docs.check_execution(notebook, path.name)
    build_docs.check_figures(notebook, path.name)
    seconds = notebook["metadata"]["execution_seconds"]
    assert isinstance(seconds, int | float) and seconds >= 0
    # A cell need not print anything, but a page whose outputs were all stripped
    # shows no results.
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert any(cell["outputs"] for cell in code)


def _executed_notebook() -> dict[str, Any]:
    """A small notebook as execution saves it: a figure, and printed output."""
    figure = {"image/png": "iVBORw0KGgo=", "text/plain": "<Figure size 1x1>"}
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "# Example"},
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {"figure_descriptions": ["A rising line."]},
                "source": "plt.plot([0, 1])",
                "outputs": [{"output_type": "display_data", "data": figure}],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "source": ["x = 1\n", "print(x)"],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "1\n"}],
            },
        ],
        "metadata": {"executed_at": "2026-10-24T16:47:14+00:00"},
    }
    notebook["metadata"]["executed_code_sha256"] = build_docs.code_fingerprint(notebook)
    return notebook


def _code_cell(notebook: dict[str, Any], index: int) -> dict[str, Any]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"][index]


def _change_code(notebook: dict[str, Any]) -> None:
    _code_cell(notebook, 1)["source"].append("\nprint(2)")


def _add_error(notebook: dict[str, Any]) -> None:
    error = {"output_type": "error", "ename": "E", "evalue": "", "traceback": []}
    _code_cell(notebook, 1)["outputs"].append(error)


def _unexecute_cell(notebook: dict[str, Any]) -> None:
    _code_cell(notebook, 1)["execution_count"] = None


def _drop_timestamp(notebook: dict[str, Any]) -> None:
    del notebook["metadata"]["executed_at"]


def _drop_fingerprint(notebook: dict[str, Any]) -> None:
    del notebook["metadata"]["executed_code_sha256"]


def _drop_description(notebook: dict[str, Any]) -> None:
    _code_cell(notebook, 0)["metadata"]["figure_descriptions"] = []


def _blank_description(notebook: dict[str, Any]) -> None:
    _code_cell(notebook, 0)["metadata"]["figure_descriptions"] = [" "]


def _description_not_a_list(notebook: dict[str, Any]) -> None:
    # A string of one character has the length of a list of one description.
    _code_cell(notebook, 0)["metadata"]["figure_descriptions"] = "A"


def _stale_dataset(notebook: dict[str, Any]) -> None:
    # Executed with a cube whose fingerprint the registry no longer records.
    notebook["metadata"]["datasets"] = ["simulated_bulk"]
    files = dict(_datasets().DATASETS["simulated_bulk"].files)
    files["simulated_bulk.cube.zarr"] = "sha256:" + "0" * 64
    notebook["metadata"]["executed_datasets"] = {"simulated_bulk": files}


def _unregistered_dataset(notebook: dict[str, Any]) -> None:
    notebook["metadata"]["datasets"] = ["no_such_dataset"]


EXECUTION = "check_execution"
FIGURES = "check_figures"


@pytest.mark.parametrize(
    ("change", "check", "message"),
    [
        (_change_code, EXECUTION, "changed after it was executed"),
        (_add_error, EXECUTION, "failed when it was executed"),
        (_unexecute_cell, EXECUTION, "has not been executed"),
        (_drop_timestamp, EXECUTION, "has never been executed"),
        (_drop_fingerprint, EXECUTION, "records no fingerprint"),
        (_stale_dataset, EXECUTION, "other data than the datasets it names"),
        (_unregistered_dataset, EXECUTION, "not registered"),
        (_drop_description, FIGURES, "figure_descriptions"),
        (_blank_description, FIGURES, "figure_descriptions"),
        (_description_not_a_list, FIGURES, "figure_descriptions"),
    ],
)
def test_notebook_checks_reject_stale_failed_partial_or_undescribed_notebooks(
    change: Callable[[dict[str, Any]], None], check: str, message: str
) -> None:
    notebook = _executed_notebook()
    build_docs.check_execution(copy.deepcopy(notebook), "example.ipynb")
    build_docs.check_figures(copy.deepcopy(notebook), "example.ipynb")
    change(notebook)

    with pytest.raises(ValueError, match=message):
        getattr(build_docs, check)(notebook, "example.ipynb")


def test_the_table_of_contents_links_below_the_base_url() -> None:
    entries = [
        {"file": "index.md"},
        {
            "title": "Reference",
            "children": [
                {"url": "/api/index.html", "open_in_same_tab": True},
                {"url": "https://github.com/NygenAnalytics/CytePool/issues"},
                {"url": "//example.org/page.html"},
                {"url": "page.html"},
            ],
        },
    ]
    original = copy.deepcopy(entries)

    result = build_docs.toc_with_base_url(entries, "/CytePool")

    assert result[0] == {"file": "index.md"}
    assert result[1]["children"] == [
        {"url": "/CytePool/api/index.html", "open_in_same_tab": True},
        {"url": "https://github.com/NygenAnalytics/CytePool/issues"},
        {"url": "//example.org/page.html"},
        {"url": "page.html"},
    ]
    assert entries == original


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "/api/index.html"),
        ("/", "/api/index.html"),
        ("/CytePool", "/CytePool/api/index.html"),
        ("/CytePool/", "/CytePool/api/index.html"),
    ],
)
def test_the_staged_site_links_to_the_api_reference_below_the_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
    expected: str,
) -> None:
    yaml = pytest.importorskip("yaml")
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("# Example\n")
    toc = [
        {"file": "index.md"},
        {"title": "Reference", "children": [{"url": "/api/index.html"}]},
    ]
    (source / "myst.yml").write_text(
        yaml.safe_dump({"version": 1, "project": {"toc": toc}})
    )
    monkeypatch.setattr(build_docs, "SOURCE", source)
    monkeypatch.setattr(build_docs, "BUILD", tmp_path / "build")
    monkeypatch.setattr(build_docs, "SITE", tmp_path / "build" / "site")
    if base_url is None:
        monkeypatch.delenv("BASE_URL", raising=False)
    else:
        monkeypatch.setenv("BASE_URL", base_url)

    site = build_docs.stage_site()

    staged = yaml.safe_load((site / "myst.yml").read_text())
    assert staged["project"]["toc"][1]["children"] == [{"url": expected}]


def test_prose_can_change_without_executing_again() -> None:
    notebook = _executed_notebook()
    fingerprint = build_docs.code_fingerprint(notebook)
    notebook["cells"][0]["source"] = "# Rewritten prose"

    assert build_docs.code_fingerprint(notebook) == fingerprint
    build_docs.check_execution(notebook, "example.ipynb")
    # Code stored as one string fingerprints as the same code stored as a list of
    # lines, the form in which notebook files keep it.
    _code_cell(notebook, 1)["source"] = "x = 1\nprint(x)"
    assert build_docs.code_fingerprint(notebook) == fingerprint
    _code_cell(notebook, 1)["source"] = "x = 1\nprint(x) "
    assert build_docs.code_fingerprint(notebook) != fingerprint


def test_output_checks_ignore_the_versions_of_cytepool_a_page_prints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runner imports nbclient, a documentation dependency.
    pytest.importorskip("nbclient")
    monkeypatch.syspath_prepend(str(ROOT / "docs"))
    runner = importlib.import_module("execute_notebooks")
    saved = [
        "Built by from_cells (CytePool 0.1.dev24+gabc) from 9 cells\n",
        "t = 4.2\n",
    ]
    fresh = [
        "Built by from_cells (CytePool 0.1.dev31+gdef) from 9 cells\n",
        "t = 4.2\n",
    ]

    # A page run by another development version prints the same text otherwise.
    assert runner.differences(saved, fresh, "0.1.dev24+gabc", "0.1.dev31+gdef") == []
    # Any other change is reported, cell by cell.
    changed = [fresh[0], "t = 4.3\n"]
    reported = runner.differences(saved, changed, "0.1.dev24+gabc", "0.1.dev31+gdef")
    assert [line.split(":")[0] for line in reported] == ["code cell 2"]
    # A version named after "CytePool " is left out, whichever version ran the page:
    # a dataset prepared earlier was built by another one.
    built_earlier = ["Built by from_cells (CytePool 0.1.dev18+g123) from 9 cells\n"]
    now = ["Built by from_cells (CytePool 0.1.dev31+gdef) from 9 cells\n"]
    assert (
        runner.differences(built_earlier, now, "0.1.dev24+gabc", "0.1.dev31+gdef") == []
    )
    assert runner.differences(built_earlier, now, None, "0.1.dev31+gdef") == []
    # A version printed on its own is compared, replaced only by the recorded one.
    alone, alone_now = ["0.1.dev24+gabc\n"], ["0.1.dev31+gdef\n"]
    assert (
        runner.differences(alone, alone_now, "0.1.dev24+gabc", "0.1.dev31+gdef") == []
    )
    assert len(runner.differences(alone, alone_now, None, "0.1.dev31+gdef")) == 1


def test_every_public_module_is_in_the_api_reference() -> None:
    documented = {
        name
        for page in API_SOURCE.glob("*.md")
        for kind, name, _ in _directives(page)
        if kind == "automodule"
    }
    public = _public_modules()

    assert public == documented
    # The definition finds what a user imports from, and not the files that
    # implement a package.
    assert {
        "cytepool.cube",
        "cytepool.design",
        "cytepool.io.zarr",
        "cytepool.stats.voom",
        "cytepool.table",
    } <= public
    assert (
        not {"cytepool.cube.cube", "cytepool.design.formula", "cytepool.stats"} & public
    )


def test_every_public_name_is_documented_under_its_own_name() -> None:
    documented = set().union(*_documented_objects().values())

    assert sorted(_required_objects() - documented) == []
    # Constants and names that a package does not re-export need directives of
    # their own; these are the ones that exist today.
    assert {"cytepool.design.INTERCEPT", "cytepool.design.formula.quote_name"} <= (
        _required_objects()
    )


def test_every_api_page_is_in_the_api_index() -> None:
    index = (API_SOURCE / "index.md").read_text()
    toctree = re.search(r"```\{toctree\}\n(.*?)```", index, re.S)
    assert toctree is not None
    listed = [
        line.strip()
        for line in toctree[1].splitlines()
        if line.strip() and not line.startswith(":")
    ]

    assert sorted(listed) == sorted(
        path.stem for path in API_SOURCE.glob("*.md") if path.stem != "index"
    )


def test_every_validation_script_is_listed_on_the_validation_page() -> None:
    page = (SOURCE / "validation.md").read_text()
    linked = re.findall(
        re.escape(REPOSITORY) + r"blob/master/validation/(\w+\.py)\)", page
    )
    scripts = {path.name for path in (ROOT / "validation").glob("*.py")}

    assert set(linked) == scripts
    # Each script has a row in the summary table and a section of its own.
    for script in scripts:
        assert linked.count(script) >= 2, script


def _anchors(page: Path) -> set[str]:
    """The labels of a page and the identifiers of its headings."""
    text = _text(page)
    headings = re.findall(r"^#+\s+(.+)$", text, re.M)
    slugs = {
        re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") for heading in headings
    }
    return set(LABEL.findall(text)) | slugs


def _link_problem(
    page: Path, target: str, labels: set[str], api: dict[str, set[str]]
) -> str | None:
    """Why a link of a page does not resolve, or None when it does."""
    if target.startswith("mailto:"):
        return None
    if target.startswith(REPOSITORY):
        path = re.split(r"[#?]", target.removeprefix(REPOSITORY))[0]
        branch = re.match(r"(?:blob|tree)/([^/]+)/(.*)", path)
        if branch and branch[1] != "master":
            return f"links to branch {branch[1]}; link to master"
        if branch and not (ROOT / branch[2]).exists():
            return f"no {branch[2]} in the repository"
        return None
    if target.startswith(("http://", "https://")):
        return None
    if target.startswith("/api/"):
        file, _, anchor = target.removeprefix("/api/").partition("#")
        api_page = file.removesuffix(".html") + ".md"
        if not file.endswith(".html") or api_page not in api:
            return f"no API page {file}"
        if anchor and anchor not in api[api_page]:
            return f"{anchor} is not documented on {file}"
        return None
    if target.startswith("#"):
        return None if target[1:] in labels else f"no label {target[1:]}"
    file, _, anchor = target.partition("#")
    if not (page.parent / file).is_file():
        return f"no page {file}"
    if anchor and anchor not in _anchors(page.parent / file):
        return f"no label or heading {anchor} in {file}"
    return None


def test_links_between_pages_resolve() -> None:
    # The site build ignores unresolved links (docs/source/myst.yml), since it cannot
    # see the API reference and should not depend on websites.
    pages = {path: _text(path) for path in _site_pages()}
    labels = {label for text in pages.values() for label in LABEL.findall(text)}
    api = _documented_objects()
    problems = [
        f"{page.name}: {target}: {problem}"
        for page, text in pages.items()
        for target in MARKDOWN_LINK.findall(text)
        if (problem := _link_problem(page, target, labels, api))
    ]

    assert problems == []
    assert {"validation", "reproduce-validation", "quickstart"} <= labels


def test_link_check_finds_broken_links() -> None:
    page = SOURCE / "index.md"
    api = _documented_objects()
    broken = [
        "missing.md",
        "installation.md#no-such-heading",
        "#no-such-label",
        "/api/missing.html",
        "/api/voom.html#cytepool.stats.voom.missing_function",
        # Reachable from a documented module, but not documented under that name.
        "/api/voom.html#cytepool.stats.voom.np",
        "/api/cubes.html#cytepool.cube.cube.Cube",
        "/api/designs.html#cytepool.design.cytepool.design.INTERCEPT",
        REPOSITORY + "blob/master/validation/missing.py",
        REPOSITORY + "blob/main/validation/voom.py",
    ]
    working = [
        "installation.md#build-this-documentation",
        "/api/voom.html#cytepool.stats.voom.voom",
        "/api/designs.html#cytepool.design.Design.check",
        "/api/designs.html#cytepool.design.INTERCEPT",
        REPOSITORY + "blob/master/validation/cube.py#L10",
    ]

    assert [t for t in broken if not _link_problem(page, t, {"validation"}, api)] == []
    assert [t for t in working if _link_problem(page, t, set(), api)] == []


# The references of the package's modules.

PACKAGE = ROOT / "cytepool"
# The modules that implement no published method, algorithm, notation, specification
# or file format, and so have no References section, with the reason.
NO_PUBLISHED_WORK = {
    "cytepool": "the package's namespace; its modules cite their sources",
    "cytepool._analysis_records": "assembles CytePool's records of an analysis",
    "cytepool._columns": "names the columns of result tables",
    "cytepool._gene_ids": "reads lists of gene ids",
    "cytepool._jit": "compiles and caches routines with numba; THIRD_PARTY_NOTICES.md "
    "records what its cache locator follows of numba",
    "cytepool._masked_arrays": "refuses numpy masked arrays at the public boundary",
    "cytepool._messages": "formats lists of ids and names in error messages",
    "cytepool._options": "checks flags and choices among names",
    "cytepool._version": "holds the version that the build writes",
    "cytepool.cube": "the package's namespace; its modules cite their sources",
    "cytepool.cube._axes": "reads the axis tables, cell numbers and cell labels of "
    "CytePool's own cubes",
    "cytepool.cube._provenance_records": "keeps CytePool's own provenance records "
    "of cubes as JSON text, written by the standard library",
    "cytepool.cube.cube": "CytePool's own structure of sums per sample, group and "
    "feature; its digests are those of cytepool.cube.fingerprints",
    "cytepool.cube.provenance": "CytePool's own record of how a cube was made, "
    "written as JSON by the standard library",
    "cytepool.design": "the package's namespace; its modules cite their sources",
    "cytepool.io": "the package's namespace; its modules cite their sources",
    "cytepool.io._files": "names local files and notices their changes with the "
    "standard library",
    "cytepool.stats": "the package's namespace; its modules cite their sources",
    "cytepool.table": "CytePool's own table of named, equal-length columns",
}
# An entry: "Authors (year). Title. Journal Volume, pages. doi:10....".
REFERENCE = re.compile(
    r"^(?P<authors>[^()]+?) \((?P<year>\d{4})[a-z]?\)\. (?P<rest>\S.*)$"
)
DOI = re.compile(r"doi:(10\.\d{4,9}/\S+)")
# A month before a year, as in "October 2026", dates a statement; it cites no work.
MONTHS = {
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}  # fmt: skip
# A work cited in the text by an author and a year, such as "Smyth (2004", "Wu et al.
# 2010" or "Hill's (1970"; the surname just before the year is the one looked up.
CITATION = re.compile(
    r"\b(?P<name>[A-Z][a-z][\w-]*)(?:'s)?(?: et al\.)? \(?(?P<year>(?:1[6-9]|20)\d\d)\b"
)


def _reference_section(doc: str) -> tuple[list[str] | None, str]:
    """The entries of a docstring's References section, each reflowed to one line,
    and the docstring without the section; None for a docstring without one."""
    lines = doc.splitlines()
    if "References:" not in lines:
        return None, doc
    start = lines.index("References:")
    end = next(
        (
            number
            for number in range(start + 1, len(lines))
            if lines[number] and not lines[number].startswith(" ")
        ),
        len(lines),
    )
    body = "\n".join(lines[start + 1 : end])
    entries = [" ".join(part.split()) for part in re.split(r"\n\s*\n", body)]
    return [entry for entry in entries if entry], "\n".join(lines[:start] + lines[end:])


def _parse_references(source: str) -> tuple[list[str] | None, list[str]]:
    """The reference entries of a module's source (None without a section), and its
    other prose: the rest of its docstring, the docstrings of its functions, classes
    and constants, and its comments, each block reflowed to one line."""
    tree = ast.parse(source)
    entries, rest = _reference_section(ast.get_docstring(tree, clean=True) or "")
    texts = [rest]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node is not tree.body[0]
        ):
            texts.append(node.value.value)
    # Comments on consecutive lines are one block of text.
    previous_row = -2
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comment = token.string.lstrip("#")
            if token.start[0] == previous_row + 1:
                texts[-1] += " " + comment
            else:
                texts.append(comment)
            previous_row = token.start[0]
    return entries, [" ".join(text.split()) for text in texts if text.strip()]


@cache
def _module_references() -> dict[str, tuple[list[str] | None, list[str]]]:
    """Every module of the package, by its import name: its reference entries and
    its other prose, read from the source without importing the module."""
    modules = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = _parse_references(path.read_text())
    return modules


def _style_problem(entry: str) -> str | None:
    """Why an entry is not in the style "Authors (year). Title. ... doi:10....", or
    None when it is."""
    if entry.startswith("- ") or not REFERENCE.match(entry):
        return "does not start with the authors and the year in parentheses"
    dois = DOI.findall(entry)
    if len(dois) > 1 or (dois and not entry.endswith(dois[0])):
        return "has a DOI that does not end it"
    return None


def _citation_problems(
    name: str, entries: list[str] | None, texts: list[str]
) -> list[str]:
    """The works that a module's prose cites by author and year but its References
    section does not list."""
    listed = [
        (match["authors"], match["year"])
        for entry in entries or []
        if (match := REFERENCE.match(entry))
    ]
    cited = {
        (m["name"], m["year"])
        for text in texts
        for m in CITATION.finditer(text)
        if m["name"] not in MONTHS
    }
    return [
        f"{name} cites {surname} {year}, which its References do not list"
        for surname, year in sorted(cited)
        if not any(
            year == listed_year and re.search(rf"\b{re.escape(surname)}\b", authors)
            for authors, listed_year in listed
        )
    ]


def _variants(entries: list[str]) -> list[list[str]]:
    """Works cited in different words: entries with the same DOI, or with the same
    first author, year and title (up to the first period), that differ."""
    groups = defaultdict(set)
    for entry in entries:
        if match := REFERENCE.match(entry):
            first_author = match["authors"].split(",")[0]
            title = re.sub(r"\W+", " ", match["rest"].split(". ")[0].casefold()).strip()
            groups["work", first_author, match["year"], title].add(entry)
        for doi in DOI.findall(entry):
            groups["doi", doi.casefold()].add(entry)
    differing = {tuple(sorted(texts)) for texts in groups.values() if len(texts) > 1}
    return [list(texts) for texts in sorted(differing)]


def test_every_module_cites_its_sources_or_says_why_it_has_none() -> None:
    modules = _module_references()
    without = {name for name, (entries, _) in modules.items() if entries is None}

    assert sorted(without - set(NO_PUBLISHED_WORK)) == []
    # Every module listed exists, has no References section and has a reason.
    assert sorted(set(NO_PUBLISHED_WORK) - without) == []
    assert all(reason.strip() for reason in NO_PUBLISHED_WORK.values())


def test_every_reference_names_its_year_in_parentheses() -> None:
    entries = [
        (name, entry)
        for name, (found, _) in _module_references().items()
        for entry in found or []
    ]
    problems = [
        f"{name}: {entry} {problem}"
        for name, entry in entries
        if (problem := _style_problem(entry))
    ]

    assert problems == []
    # The sections were read: the package cites well over a hundred works.
    assert len({entry for _, entry in entries}) > 100


def test_a_work_is_cited_in_the_same_words_everywhere() -> None:
    entries = [
        entry for found, _ in _module_references().values() for entry in found or []
    ]

    assert _variants(entries) == []


def test_every_work_a_module_cites_is_in_its_references() -> None:
    problems = [
        problem
        for name, (entries, texts) in _module_references().items()
        for problem in _citation_problems(name, entries, texts)
    ]

    assert problems == []


def test_reference_checks_find_what_is_wrong() -> None:
    source = textwrap.dedent(
        '''
        """Moderated tests.

        The variances are moderated as Smyth (2004) and Wu et al. (2010) describe.

        References:
            Smyth GK (2004). Linear models and empirical Bayes methods. Statistical
            Applications in Genetics and Molecular Biology 3, article 3.
            doi:10.2202/1544-6115.1027

            - Hill GW (1970). ACM Algorithm 395: Student's t-distribution.

            Wu D, Smyth GK (2010) ROAST. Bioinformatics 26, 2176-2182.
        """

        # Workers take segments by the longest-processing-time rule (Graham
        # 1969), as measured in October 2026.


        def f():
            """Fitted as lme4 fits it (Bates et al. 2015)."""
        '''
    )
    entries, texts = _parse_references(source)

    assert entries is not None and len(entries) == 3
    assert [_style_problem(entry) is None for entry in entries] == [True, False, False]
    # Wu (2010) is cited, but its entry has no period after the year; Graham's
    # citation spans two comment lines.
    assert _citation_problems("example", entries, texts) == [
        "example cites Bates 2015, which its References do not list",
        "example cites Graham 1969, which its References do not list",
        "example cites Wu 2010, which its References do not list",
    ]
    assert _parse_references('"""No sources."""\n') == (None, ["No sources."])
    # A DOI ends its entry; the same work is cited in the same words everywhere.
    assert (
        _style_problem("Smyth GK (2004). Title. doi:10.2202/1. Journal 3.") is not None
    )
    same_doi = [
        "Smyth GK (2004). Linear models. SAGMB 3, 3. doi:10.2202/1544-6115.1027",
        "Smyth GK (2004). Linear models. SAGMB 3(1):3. doi:10.2202/1544-6115.1027",
    ]
    same_work = [
        "Searle SR (1971). Linear Models. Wiley.",
        "Searle SR (1971). Linear Models.",
    ]
    assert _variants(same_doi) == [sorted(same_doi)]
    assert _variants(same_work) == [sorted(same_work)]
    assert _variants(same_doi[:1] + same_work[:1]) == []


# The navigation.


def _toc() -> list[dict[str, Any]]:
    yaml = pytest.importorskip("yaml")
    toc: list[dict[str, Any]] = yaml.safe_load((SOURCE / "myst.yml").read_text())[
        "project"
    ]["toc"]
    return toc


def test_the_navigation_has_the_agreed_sections() -> None:
    overview, *sections = _toc()

    # MyST needs a file first: the Overview.
    assert overview == {"file": "index.md", "title": "Overview"}
    assert [entry.get("title") for entry in sections] == list(page_index.SECTIONS)
    # Every section is a group of pages, and holds at least one.
    assert all(entry.get("children") for entry in sections)
    assert all(
        any("file" in child for child in entry["children"]) for entry in sections
    )


def test_every_page_has_a_kind_and_the_lists_of_kinds_name_pages() -> None:
    files = {file for file, _, _ in page_index.toc_pages({"project": {"toc": _toc()}})}

    assert sorted(page_index.INDEX_PAGES - files) == []
    assert sorted(page_index.EXPLANATION_PAGES - files) == []
    kinds = {page.file: page.kind for page in page_index.pages()}
    assert kinds["statistics/choosing_a_method.md"] == page_index.EXPLANATION
    assert kinds["reference/cube_files.md"] == page_index.EXPLANATION
    assert kinds["reference/capabilities.md"] == page_index.INDEX
    assert kinds["quickstart.ipynb"] == page_index.GUIDE


@pytest.mark.parametrize(
    ("file", "url"),
    [
        ("index.md", "/"),
        ("installation.md", "/installation"),
        ("why_pseudobulk.ipynb", "/why-pseudobulk"),
        ("statistics/empirical_bayes.md", "/statistics/empirical-bayes"),
        ("guides/Bulk_RNA.seq.ipynb", "/guides/bulk-rna-seq"),
    ],
)
def test_page_urls_follow_the_rule_of_myst(file: str, url: str) -> None:
    # MyST 1.8.1 serves why_pseudobulk.ipynb at /why-pseudobulk and,
    # with folders, stats/a_page.md at /stats/a-page; build_docs.py checks every page
    # against the built site's myst.xref.json.
    assert page_index.page_url(file) == url


def test_the_build_finds_pages_that_the_site_does_not_have(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    pages = page_index.pages()
    references = [{"kind": "page", "url": page.url} for page in pages]
    xref = tmp_path / "myst.xref.json"
    xref.write_text(json.dumps({"references": references}))
    build_docs.check_page_urls(xref)
    # Under a base URL the site's URLs carry it.
    prefixed = [{"kind": "page", "url": "/CytePool" + page.url} for page in pages]
    xref.write_text(json.dumps({"references": prefixed}))
    build_docs.check_page_urls(xref, "/CytePool")

    xref.write_text(json.dumps({"references": references[1:]}))
    with pytest.raises(ValueError, match=re.escape(pages[0].file)):
        build_docs.check_page_urls(xref)


# Links from the pages to the API reference, and back.


def test_the_api_entries_are_the_documented_objects() -> None:
    entries = page_index.api_entries()
    pages = _documented_objects()

    assert set(entries) == set().union(*pages.values())
    for name, entry in entries.items():
        assert name in pages[entry.page.replace(".html", ".md")]


def test_every_api_role_resolves_to_one_entry() -> None:
    entries = page_index.api_entries()
    problems = []
    for path in _site_pages():
        prose, _ = page_index.read_page(path)
        for role in page_index.roles_in(prose):
            try:
                page_index.resolve(role, entries)
            except ValueError as error:
                problems.append(f"{path.relative_to(SOURCE)}: {error}")

    assert problems == []


def _entries() -> dict[str, Any]:
    entry = page_index.ApiEntry
    return {
        "pkg.tools": entry("pkg.tools", "tools.html", "module"),
        "pkg.tools.fit": entry("pkg.tools.fit", "tools.html", "function"),
        "pkg.tools.Table": entry("pkg.tools.Table", "tools.html", "class"),
        "pkg.tools.Table.select": entry(
            "pkg.tools.Table.select", "tools.html", "method"
        ),
        "pkg.tools.Table.fit": entry("pkg.tools.Table.fit", "tools.html", "method"),
        "pkg.cubes.Cube.select": entry("pkg.cubes.Cube.select", "cubes.html", "method"),
        "pkg.cubes.Cube.shape": entry("pkg.cubes.Cube.shape", "cubes.html", "property"),
    }


@pytest.mark.parametrize(
    ("name", "resolved"),
    [
        ("pkg.tools.fit", "pkg.tools.fit"),
        # A unique last part.
        ("Cube.select", "pkg.cubes.Cube.select"),
        ("shape", "pkg.cubes.Cube.shape"),
        ("~pkg.tools.Table", "pkg.tools.Table"),
        # Without a dot, the function rather than the method of the same name.
        ("fit", "pkg.tools.fit"),
        ("Table.fit", "pkg.tools.Table.fit"),
        # A module by its full name only.
        ("pkg.tools", "pkg.tools"),
    ],
)
def test_a_role_resolves_its_name(name: str, resolved: str) -> None:
    assert page_index.resolve(name, _entries()).name == resolved


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (
            "select",
            "ambiguous: it could be pkg.cubes.Cube.select, pkg.tools.Table.select",
        ),
        ("tools", "not documented"),
        ("missing", "not documented"),
    ],
)
def test_a_role_that_names_nothing_or_several_entries_is_refused(
    name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        page_index.resolve(name, _entries())


def test_roles_become_links_outside_code() -> None:
    text = textwrap.dedent(
        """\
        Fit with {api}`fit`, select with {api}`the method <Cube.select>` or
        {api}`~pkg.tools.Table`; ``{api}`fit` `` shows the role itself.

        ```python
        x = "{api}`fit`"
        ```

        ```{note}
        A note may link {api}`Table.select`.
        ```
        """
    )

    rewritten = page_index.rewrite_roles(text, _entries())

    assert rewritten == textwrap.dedent(
        """\
        Fit with [`fit`](/api/tools.html#pkg.tools.fit), select with [the method](/api/cubes.html#pkg.cubes.Cube.select) or
        [`Table`](/api/tools.html#pkg.tools.Table); ``{api}`fit` `` shows the role itself.

        ```python
        x = "{api}`fit`"
        ```

        ```{note}
        A note may link [`Table.select`](/api/tools.html#pkg.tools.Table.select).
        ```
        """
    )
    prose, code = page_index.split_markdown(text)
    assert page_index.roles_in(prose) == [
        "fit",
        "Cube.select",
        "~pkg.tools.Table",
        "Table.select",
    ]
    assert code == 'x = "{api}`fit`"\n'


def test_the_staged_site_links_roles_to_the_api_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml = pytest.importorskip("yaml")
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("# Example\n\nSee {api}`limma_voom`.\n")
    toc = [{"file": "index.md"}]
    (source / "myst.yml").write_text(
        yaml.safe_dump({"version": 1, "project": {"toc": toc}})
    )
    monkeypatch.setattr(build_docs, "SOURCE", source)
    monkeypatch.setattr(build_docs, "BUILD", tmp_path / "build")
    monkeypatch.setattr(build_docs, "SITE", tmp_path / "build" / "site")
    monkeypatch.delenv("BASE_URL", raising=False)

    site = build_docs.stage_site()

    assert (site / "index.md").read_text() == (
        "# Example\n\nSee [`limma_voom`]("
        "/api/differential_expression.html#cytepool.differential_expression.limma_voom).\n"
    )
    (source / "index.md").write_text("# Example\n\nSee {api}`select`.\n")
    with pytest.raises(ValueError, match="'select' is ambiguous"):
        build_docs.stage_site()


def test_the_code_of_a_page_uses_what_it_imports_from_cytepool() -> None:
    code = textwrap.dedent(
        """\
        %time import numpy as np
        from cytepool.cube.pool import from_cells
        import cytepool.stats.voom as weights
        from cytepool._jit import jit
        from cytepool.design import Design as D
        result = weights.voom(counts, D(samples, "~ group"))
        """
    )

    uses = page_index.code_uses(code, page_index.api_entries())

    # A name imported from the module that implements it finds its documented name;
    # private names and modules are not uses.
    assert uses == {
        "cytepool.cube.from_cells",
        "cytepool.stats.voom.voom",
        "cytepool.design.Design",
    }


def test_api_entries_link_back_to_the_pages_that_use_and_explain_them() -> None:
    pytest.importorskip("yaml")
    links = page_index.links()
    used = {
        page.file
        for page in links.used_in["cytepool.differential_expression.limma_voom"]
    }
    explained = {
        page.file
        for page in links.explained_in["cytepool.differential_expression.limma_voom"]
    }

    assert "quickstart.ipynb" in used  # its code imports it
    assert "guides/bulk_rna_seq.ipynb" in used
    assert "statistics/choosing_a_method.md" in explained
    # Index pages link to many names and give no links back.
    index = {page.file for page in page_index.pages() if page.kind == page_index.INDEX}
    for pages in (*links.used_in.values(), *links.explained_in.values()):
        assert not index & {page.file for page in pages}
    lines = page_index.backlink_lines(
        "cytepool.differential_expression.limma_voom", links
    )
    assert lines[0] == ""
    assert lines[1].startswith("**Used in:** `Quick start <../quickstart/>`__")
    assert "`Choosing a method <../statistics/choosing-a-method/>`__" in lines[3]
    assert page_index.backlink_lines("cytepool.stats.special.lgamma", links) == []


# The bibliography.

BIBLIOGRAPHY = SOURCE / "references.bib"
BIB_ENTRY = re.compile(
    r"^@(?P<kind>\w+)\{(?P<key>[^,\s]+),\n(?P<body>.*?)\n\}$", re.M | re.S
)
BIB_FIELD = re.compile(r"^  (?P<name>\w+) = \{(?P<value>.*)\},$", re.M)


@cache
def _bibliography() -> dict[str, dict[str, str]]:
    """The entries of references.bib by key, each a mapping of field to value."""
    text = BIBLIOGRAPHY.read_text()
    entries: dict[str, dict[str, str]] = {}
    for match in BIB_ENTRY.finditer(text):
        assert match["key"] not in entries, f"{match['key']} is in references.bib twice"
        fields = dict(BIB_FIELD.findall(match["body"]))
        fields["@kind"] = match["kind"]
        entries[match["key"]] = fields
    # Every entry was read: as many as the file starts.
    assert len(entries) == text.count("\n@") + text.startswith("@")
    return entries


def test_the_site_build_reads_the_bibliography_and_stops_on_unknown_keys() -> None:
    yaml = pytest.importorskip("yaml")
    project = yaml.safe_load((SOURCE / "myst.yml").read_text())["project"]

    assert project["bibliography"] == ["references.bib"]
    assert {"rule": "reference-target-resolves", "severity": "error"} in project[
        "error_rules"
    ]


def test_every_bibliography_entry_is_complete() -> None:
    entries = _bibliography()
    problems = []
    dois: dict[str, str] = {}
    for key, fields in entries.items():
        year = fields.get("year", "")
        if (
            not re.fullmatch(r"[a-z]+(\d{4})[a-z]?", key)
            or key[-5:-1] != year
            and key[-4:] != year
        ):
            problems.append(f"{key}: the key is not a name and the year {year}")
        if not ({"author", "editor"} & set(fields) and "title" in fields):
            problems.append(f"{key}: no author, editor or title")
        if fields["@kind"] == "article" and "journal" not in fields:
            problems.append(f"{key}: an article without a journal")
        if "doi" in fields:
            doi = urllib.parse.unquote(fields["doi"]).lower()
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
                problems.append(f"{key}: {fields['doi']} is not a DOI")
            if doi in dois:
                problems.append(f"{key} and {dois[doi]} have the same DOI")
            dois[doi] = key

    assert problems == []
    assert len(entries) > 150


def test_every_citation_of_the_pages_is_in_the_bibliography() -> None:
    keys = set(_bibliography())
    unknown = [
        f"{path.relative_to(SOURCE)}: {key}"
        for path in _site_pages()
        for key in page_index.citations(page_index.read_page(path)[0])
        if key not in keys
    ]

    assert unknown == []


def test_the_bibliography_page_cites_every_entry() -> None:
    prose, _ = page_index.read_page(SOURCE / "reference" / "bibliography.md")

    assert sorted(set(_bibliography()) - set(page_index.citations(prose))) == []


def test_every_doi_the_modules_cite_is_in_the_bibliography_with_its_year() -> None:
    by_doi = {
        urllib.parse.unquote(fields["doi"]).lower(): (key, fields["year"])
        for key, fields in _bibliography().items()
        if "doi" in fields
    }
    problems = []
    for name, (entries, _) in _module_references().items():
        for entry in entries or []:
            match = REFERENCE.match(entry)
            for doi in DOI.findall(entry):
                found = by_doi.get(doi.lower())
                if found is None:
                    problems.append(f"{name}: doi:{doi} is not in references.bib")
                elif match and found[1] != match["year"]:
                    problems.append(f"{name}: doi:{doi} is {found[0]} of {found[1]}")

    assert problems == []
    # The modules cite about a hundred works with DOIs.
    assert len(by_doi) > 100


def test_citations_are_read_outside_code_links_and_mail_addresses() -> None:
    prose = (
        "As shown [@squair2021; @murphy2022, figure 1] and by @zimmerman2021. Write to "
        "help@example.org, or [the list](mailto:list@example.org); `@notakey` and "
        "{api}`limma_voom` are code, and {cite:p}`law2014` a role."
    )

    assert page_index.citations(prose) == [
        "law2014",
        "squair2021",
        "murphy2022",
        "zimmerman2021",
    ]


# The datasets.


def _datasets() -> ModuleType:
    return page_index.docs_module("datasets")


def _validation_constants() -> dict[str, Any]:
    """The constants of the validation script of GSE96583, read without running it."""
    tree = ast.parse(
        (ROOT / "validation" / "interferon_stimulated_pbmcs.py").read_text()
    )
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                if isinstance(node.value, ast.Call):  # "... ".split()
                    call = node.value
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "split"
                    ):
                        found[node.targets[0].id] = ast.literal_eval(
                            call.func.value
                        ).split()
    return found


def test_every_dataset_records_its_source_licence_citation_and_contents() -> None:
    datasets = _datasets()
    keys = set(_bibliography())

    assert set(datasets.DATASETS) == {
        "simulated_pbmc",
        "simulated_bulk",
        "simulated_cohorts",
        "kang2018",
        "hallmark_interferon_alpha",
    }
    for dataset_id, dataset in datasets.DATASETS.items():
        assert dataset.id == dataset_id
        assert (
            dataset.title and dataset.description and dataset.source and dataset.licence
        )
        assert dataset.citation is None or dataset.citation in keys
        # What it does not simulate, it downloads or copies from a publication.
        assert dataset.citation is not None or "simulated" in dataset.source
        assert dataset.files
        assert all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", f) for f in dataset.files.values()
        )
        for name, download in dataset.downloads.items():
            assert download.url.startswith("https://") and download.url.endswith(name)
            assert download.size > 0 and re.fullmatch(r"[0-9a-f]{64}", download.sha256)


def test_the_downloads_of_kang2018_are_those_of_the_validation_script() -> None:
    datasets = _datasets()
    validation = _validation_constants()

    kang = datasets.DATASETS["kang2018"]
    assert {
        name: (download.size, download.sha256)
        for name, download in kang.downloads.items()
    } == validation["FILES"]
    assert all(
        d.url == validation["SOURCE"] + name for name, d in kang.downloads.items()
    )
    assert datasets.GEO_MATRICES == validation["MATRICES"]
    assert datasets.GEO_GENES == validation["N_GENES"]
    assert datasets.GEO_CELL_TYPES == validation["CELL_TYPES"]
    assert datasets.INTERFERON_ALPHA_RESPONSE == validation["INTERFERON_ALPHA_RESPONSE"]
    assert datasets.EARLIER_SYMBOLS == validation["EARLIER_SYMBOLS"]
    assert len(set(datasets.INTERFERON_ALPHA_RESPONSE)) == 97


def test_the_notices_and_the_datasets_page_name_every_dataset() -> None:
    datasets = _datasets()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    page = (SOURCE / "reference" / "datasets.md").read_text()

    for dataset_id, dataset in datasets.DATASETS.items():
        row = next(
            line for line in page.splitlines() if line.startswith(f"| `{dataset_id}`")
        )
        if dataset.citation:
            assert f"@{dataset.citation}" in row
            # Data the documentation does not make itself are in the notices.
            assert f"`{dataset_id}`" in notices
    assert "docs/datasets.py" in notices
    assert "use no dataset" not in notices


def test_the_cache_must_be_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = _datasets()
    monkeypatch.setenv("CYTEPOOL_DOCS_DATA", str(tmp_path / "cache"))
    assert datasets.data_directory() == tmp_path / "cache"

    monkeypatch.setenv("CYTEPOOL_DOCS_DATA", str(ROOT / "docs" / "data"))
    with pytest.raises(SystemExit, match="inside the repository"):
        datasets.data_directory()


def test_the_digest_of_arrays_ignores_how_the_archive_was_written(
    tmp_path: Path,
) -> None:
    datasets = _datasets()
    arrays = {"counts": np.arange(12).reshape(3, 4), "ids": np.array(["a", "b", "c"])}
    np.savez(tmp_path / "plain.npz", **arrays)
    np.savez_compressed(tmp_path / "compressed.npz", **dict(reversed(arrays.items())))
    np.savez(
        tmp_path / "changed.npz",
        counts=arrays["counts"] + (arrays["counts"] == 5),
        ids=arrays["ids"],
    )
    np.savez(
        tmp_path / "reshaped.npz",
        counts=arrays["counts"].reshape(4, 3),
        ids=arrays["ids"],
    )

    digest = datasets.digest(tmp_path / "plain.npz")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert datasets.digest(tmp_path / "compressed.npz") == digest
    assert datasets.digest(tmp_path / "changed.npz") != digest
    assert datasets.digest(tmp_path / "reshaped.npz") != digest


def test_a_dataset_is_made_once_and_checked_against_the_registry(
    tmp_path: Path,
) -> None:
    datasets = _datasets()
    builds = []

    def build(downloads: Path, output: Path) -> None:
        builds.append(output)
        (output / "values.json").write_text('{"x": 1}\n')

    expected = "sha256:" + __import__("hashlib").sha256(b'{"x": 1}\n').hexdigest()
    dataset = datasets.Dataset(
        id="example",
        title="Example",
        description="one value",
        source="simulated",
        licence="BSD",
        citation=None,
        files={"values.json": expected},
        build=build,
    )
    directory = tmp_path / "cache"

    assert datasets.check(dataset, directory) == [
        "values.json is missing; run: python docs/datasets.py prepare example"
    ]
    assert datasets.prepare(dataset, directory) == []
    assert datasets.prepare(dataset, directory) == []
    assert builds == [directory]  # the second call finds it prepared
    (directory / "values.json").write_text('{"x": 2}\n')
    assert datasets.check(dataset, directory)[0].startswith("values.json has sha256:")


def test_counts_are_read_from_matrix_market_files_without_scipy(tmp_path: Path) -> None:
    datasets = _datasets()
    # 4 genes x 3 cells, genes x cells and counted from 1, as GEO stores them.
    entries = [(1, 1, 5), (3, 1, 2), (2, 2, 7), (4, 3, 1), (1, 3, 3)]
    text = "%%MatrixMarket matrix coordinate real general\n%\n4 3 5\n" + "".join(
        f"{g} {c} {v}\n" for g, c, v in entries
    )
    path = tmp_path / "counts.mtx.gz"
    path.write_bytes(gzip.compress(text.encode()))
    expected = np.zeros((3, 4), dtype=np.int64)  # cells x genes
    for g, c, v in entries:
        expected[c - 1, g - 1] = v

    with path.open("rb") as raw:
        shape, read = datasets._matrix_market(raw)
    cells = datasets._cells_from_entries(read, 3, 4)
    dense = np.zeros((3, 4), dtype=np.int64)
    for row in range(3):
        start, end = cells["indptr"][row], cells["indptr"][row + 1]
        dense[row, cells["indices"][start:end]] = cells["data"][start:end]

    assert shape == (4, 3)
    np.testing.assert_array_equal(dense, expected)
    kept = datasets._take_rows(cells, np.array([2, 0]))
    stacked = datasets._stack([kept, datasets._take_rows(cells, np.array([1]))], 4)
    rebuilt = np.zeros((3, 4), dtype=np.int64)
    for row in range(3):
        start, end = stacked["indptr"][row], stacked["indptr"][row + 1]
        rebuilt[row, stacked["indices"][start:end]] = stacked["data"][start:end]
    np.testing.assert_array_equal(rebuilt, expected[[2, 0, 1]])
    assert stacked["shape"] == (3, 4)


# The coverage of the public names.


def _coverage_report() -> ModuleType:
    return page_index.docs_module("coverage_report")


def test_every_public_name_is_covered_or_listed_with_a_reason() -> None:
    pytest.importorskip("yaml")
    report = _coverage_report().report()

    assert report.uncovered == []
    assert report.stale == []
    assert report.empty == []
    # Most names earn a place in a page.
    assert len(report.pages) > len(report.exempt) + len(report.planned)


def test_the_public_names_are_those_of_the_api_rule_and_their_methods() -> None:
    names = _coverage_report().public_names()
    required = _required_objects()

    assert required <= names
    for name in names - required:
        parent, _, attribute = name.rpartition(".")
        assert parent in required and inspect.isclass(
            getattr(
                importlib.import_module(parent.rpartition(".")[0]),
                parent.rpartition(".")[2],
            )
        ), name
    assert {"cytepool.cube.Cube.select", "cytepool.design.Design.check"} <= names
    # Properties are not methods.
    assert "cytepool.cube.Cube.n_cells" not in names


def test_the_coverage_report_finds_names_listed_wrongly(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    coverage = _coverage_report()
    exempt, planned = coverage.read_exemptions()
    path = tmp_path / "exemptions.toml"

    def write(entries: dict[str, str], more: str = "") -> None:
        lines = ["[exempt]"] + [
            f'"{name}" = "{reason}"' for name, reason in entries.items()
        ]
        path.write_text("\n".join(lines + ["", "[planned]", more]) + "\n")

    write(
        exempt
        | {
            "cytepool.differential_expression.limma_voom": "covered",
            "cytepool.nothing": "x",
        }
    )
    report = coverage.report(path)
    assert report.stale == [
        "cytepool.differential_expression.limma_voom",
        "cytepool.nothing",
    ]
    first = next(iter(exempt))
    write({**exempt, first: " "})
    assert coverage.report(path).empty == [first]
    write({name: reason for name, reason in exempt.items() if name != first})
    assert coverage.report(path).uncovered == sorted([first, *planned])
    path.write_text("[ignored]\n")
    with pytest.raises(ValueError, match="expected exempt and planned"):
        coverage.read_exemptions(path)


# The capability matrix.


def _table_rows(text: str, header: str) -> list[list[str]]:
    """The rows of the Markdown table that starts with ``header``, as cells."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _script(cell: str) -> str:
    """The validation script a table cell links to."""
    match = re.search(r"`(\w+\.py)`", cell)
    assert match is not None, cell
    return match[1]


def test_the_capability_matrix_agrees_with_the_validation_summary() -> None:
    summary = {
        _script(row[0]): row[2]
        for row in _table_rows((SOURCE / "validation.md").read_text(), "| Script |")
        if "`" in row[0]
    }
    matrix = _table_rows(
        (SOURCE / "reference" / "capabilities.md").read_text(), "| Method |"
    )
    problems = []
    for method, _, checked, tier, script_cell in matrix:
        script = _script(script_cell)
        if script not in summary:
            problems.append(f"{method}: {script} is not on the Validation page")
            continue
        # Every reference and version named is one the script was run against.
        for tool in re.findall(r"[\w-]+ \d[\w.-]*\d", checked):
            if tool not in summary[script]:
                problems.append(f"{method}: {tool} is not among {summary[script]}")
        if not re.match(r"(exact|close|calibrated)\b", tier):
            problems.append(f"{method}: the tier {tier!r} is not a tier")

    assert problems == []
    # Every script of the Validation page has a row.
    assert sorted(set(summary) - {_script(row[4]) for row in matrix}) == []


@pytest.mark.parametrize("base_url", ["", "/CytePool"])
def test_api_pages_join_the_staged_guide_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    yaml = pytest.importorskip("yaml")

    api = tmp_path / "docs" / "build" / "api"
    api.mkdir(parents=True)
    (api / "index.html").write_text(
        '<nav class="docs-api-tree"><ul><li>'
        '<a href="cubes.html"><span>Cubes API reference</span></a>'
        '</li><li><a href="import_export.html">Import &amp; export</a>'
        "</li></ul></nav>"
    )
    site = tmp_path / "site"
    site.mkdir()
    config = {
        "project": {
            "toc": [
                {"file": "index.md"},
                {
                    "title": "Reference",
                    "children": [
                        {"url": f"{base_url}/api/index.html", "title": "API reference"}
                    ],
                },
            ]
        }
    }
    (site / "myst.yml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(build_docs, "DOCS", api.parents[1])
    monkeypatch.setenv("BASE_URL", base_url)

    build_docs.include_api_navigation(site)

    toc = yaml.safe_load((site / "myst.yml").read_text())["project"]["toc"]
    assert toc[0] == {"file": "index.md"}
    assert toc[1]["children"][0]["children"] == [
        {
            "title": "Cubes API reference",
            "url": f"{base_url}/api/cubes.html",
            "open_in_same_tab": True,
        },
        {
            "title": "Import & export",
            "url": f"{base_url}/api/import_export.html",
            "open_in_same_tab": True,
        },
    ]


@pytest.mark.parametrize("base_url", ["", "/CytePool"])
def test_api_pages_keep_the_guide_header_navigation_and_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    pytest.importorskip("yaml")

    monkeypatch.setenv("BASE_URL", base_url)
    html = tmp_path / "html"
    api = html / "api"
    api.mkdir(parents=True)
    header = (
        '<nav class="myst-top-nav-bar"><div>'
        '<button class="myst-top-nav-menu-button flex">Menu</button>'
        f'<a class="myst-home-link" href="{base_url}/"><img src="logo.svg"/>CytePool</a>'
        '</div><button aria-expanded="false" data-state="closed" class="myst-search-bar flex myst-search-bar-disabled">Search'
        "<script>window.headerScript = true;</script></button>"
        '<button class="myst-theme-button">Theme</button></nav>'
    )
    navigation = (
        '<nav class="myst-primary-sidebar-toc"><div class="myst-toc">'
        f'<a class="myst-toc-item myst-toc-item-exact font-bold p-2 my-1" aria-current="page" href="{base_url}/">Overview</a>'
        f'<a href="{base_url}/tutorials/remote-stores/">Remote stores</a>'
        f'<a href="{base_url}/api/index.html">API reference</a>'
        f'<a href="{base_url}/api/cubes.html">Cubes</a>'
        "</div></nav>"
    )
    (html / "index.html").write_text(
        f"<html><head></head><body>{header}{navigation}</body></html>"
    )
    article = '<article><h1 id="cubes">Cubes</h1><a href="cubes.html#method">Method</a></article>'
    for name in ("index.html", "cubes.html", "search.html", "genindex.html"):
        (api / name).write_text(
            '<html><head></head><body class="docs-api"><header>'
            '<nav class="docs-header-placeholder"></nav></header>'
            '<div class="myst-primary-sidebar-nav docs-api-navigation">'
            '<nav class="docs-api-tree"><ul><li>Original API menu</li></ul></nav>'
            f"</div>{article}</body></html>"
        )
    (api / "objects.inv").write_bytes(b"existing Sphinx inventory")
    static = api / "_static"
    static.mkdir()
    (static / "webpack-macros.html").write_text("{% macro assets() %}{% endmacro %}")

    build_docs.integrate_api_navigation(html)

    for page in api.glob("*.html"):
        contents = page.read_text()
        copied_header = build_docs.html_fragment(contents, "myst-top-nav-bar")
        copied_navigation = build_docs.html_fragment(
            contents, "myst-primary-sidebar-toc"
        )
        assert 'class="myst-search-bar search-button__button"' in copied_header
        assert (
            'class="myst-top-nav-menu-button sidebar-toggle primary-toggle"'
            in copied_header
        )
        assert "myst-search-bar-disabled" not in copied_header
        assert 'aria-label="Search documentation"' in copied_header
        assert 'aria-controls="pst-search-dialog"' in copied_header
        assert "aria-expanded" not in copied_header
        assert "data-state" not in copied_header
        assert "<script" not in copied_header
        assert "Original API menu" not in contents
        assert "font-bold" not in copied_navigation
        assert "myst-toc-item-exact" not in copied_navigation
        assert "p-2" not in copied_navigation
        assert "my-1" not in copied_navigation
        assert re.findall(r'href="([^"]+)"', copied_navigation) == re.findall(
            r'href="([^"]+)"', navigation
        )
        selected = re.findall(r'<a\b[^>]*aria-current="page"[^>]*>', copied_navigation)
        if page.name in {"index.html", "cubes.html"}:
            assert len(selected) == 1
            assert f'href="{base_url}/api/{page.name}"' in selected[0]
        else:
            assert not selected
        assert article in contents
        assert contents.count('src="https://docs.nygen.io/assets/brand/docs.js"') == 1
        assert "nygen-docs-sidebar:" in contents
        assert "fonts/inter/latin.woff2" in contents
    assert (api / "objects.inv").read_bytes() == b"existing Sphinx inventory"
    assert (
        static / "webpack-macros.html"
    ).read_text() == "{% macro assets() %}{% endmacro %}"


def test_html_fragment_rejects_missing_or_duplicate_navigation() -> None:
    pytest.importorskip("yaml")

    for contents in (
        "<html></html>",
        '<nav class="menu"></nav><nav class="menu"></nav>',
    ):
        with pytest.raises(ValueError, match="Expected one complete .menu element"):
            build_docs.html_fragment(contents, "menu")


@pytest.mark.parametrize(
    ("route", "title", "expected"),
    [
        ("/", "CytePool - CytePool", "CytePool documentation"),
        ("/quickstart/", "Quick start - CytePool", "Quick start - CytePool"),
        (
            "/api/index.html",
            "CytePool Python API reference — CytePool API reference",
            "CytePool Python API reference",
        ),
        (
            "/api/cytepool.html",
            "CytePool API reference — CytePool API reference",
            "CytePool Python API",
        ),
    ],
)
def test_page_metadata_preserves_authored_content_and_completes_link_previews(
    route: str, title: str, expected: str
) -> None:

    class Metadata(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: list[dict[str, str | None]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag == "meta":
                self.tags.append(dict(attrs))

    body = "<body><h1>Keep this heading</h1><svg><title>Keep SVG title</title></svg></body>"
    source = (
        f"<html><head><title>{title}</title>"
        '<meta name="description" content="Read RNA &amp; ATAC &quot;together&quot;."/>'
        '<meta property="og:title" content="Stale title"/>'
        '<meta name="twitter:card" content="summary_large_image"/>'
        '<meta name="robots" content="noindex,follow"/>'
        f"</head>{body}</html>"
    )
    canonical = build_docs.PUBLIC_URL + route

    actual = build_docs.update_page_metadata(source, canonical)

    head = actual.split("</head>")[0]
    parser = Metadata()
    parser.feed(head)
    keys = [tag.get("property", tag.get("name")) for tag in parser.tags]
    assert len(keys) == len(set(keys))
    metadata = dict(zip(keys, (tag["content"] for tag in parser.tags), strict=True))
    assert re.findall(r"<title>(.*?)</title>", head) == [expected]
    assert metadata["description"] == 'Read RNA & ATAC "together".'
    assert metadata["og:description"] == metadata["description"]
    assert metadata["og:title"] == expected
    assert metadata["og:type"] == "website"
    assert metadata["og:url"] == canonical
    assert (
        metadata["og:image"]
        == "https://docs.nygen.io/assets/brand/logo/cytepool-dark.png"
    )
    assert metadata["og:image:type"] == "image/png"
    assert metadata["og:image:width"] == metadata["og:image:height"] == "512"
    assert metadata["og:image:alt"] == "CytePool logo"
    assert metadata["twitter:card"] == "summary"
    assert metadata["twitter:image"] == metadata["og:image"]
    assert metadata["twitter:image:alt"] == metadata["og:image:alt"]
    assert metadata["robots"] == "noindex,follow"
    assert actual.endswith(body + "</html>")
    assert build_docs.update_page_metadata(actual, canonical) == actual


def test_page_metadata_rejects_missing_authored_description() -> None:

    with pytest.raises(ValueError, match="Expected a title and description"):
        build_docs.update_page_metadata(
            "<html><head><title>Guide</title></head><body></body></html>",
            build_docs.PUBLIC_URL + "/guide/",
        )


@pytest.mark.parametrize("authored", [False, True])
def test_staging_adds_introductory_descriptions_without_changing_source_or_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authored: bool
) -> None:
    yaml = pytest.importorskip("yaml")
    source = tmp_path / "source"
    source.mkdir()
    prose = "# Example\n\nCompare **sample groups** with [tables](index.md).\n\n## Results\n"
    if authored:
        prose = "---\ndescription: Keep the authored summary.\n---\n\n" + prose
    notebook = _executed_notebook()
    notebook["cells"][0]["source"] = prose
    saved = json.dumps(notebook)
    (source / "example.ipynb").write_text(saved)
    (source / "myst.yml").write_text(
        yaml.safe_dump({"project": {"toc": [{"file": "example.ipynb"}]}})
    )
    monkeypatch.setattr(build_docs, "SOURCE", source)
    monkeypatch.setattr(build_docs, "BUILD", tmp_path / "build")
    monkeypatch.setattr(build_docs, "SITE", tmp_path / "build" / "site")

    staged = json.loads((build_docs.stage_site() / "example.ipynb").read_text())

    frontmatter = re.match(r"\A---\n(.*?)\n---\n", staged["cells"][0]["source"], re.S)
    assert frontmatter
    assert yaml.safe_load(frontmatter[1])["description"] == (
        "Keep the authored summary."
        if authored
        else "Compare sample groups with tables."
    )
    assert build_docs.code_fingerprint(staged) == build_docs.code_fingerprint(notebook)
    assert (source / "example.ipynb").read_text() == saved
    assert prose.split("# Example", 1)[1] in staged["cells"][0]["source"]


@pytest.mark.parametrize("base_url", ["", "/CytePool"])
def test_discovery_covers_guides_and_authored_api_with_one_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    html = tmp_path / "html"
    (html / "api" / "_sources").mkdir(parents=True)
    body = '<body><h1>Unchanged title</h1><a id="symbol">API symbol</a></body>'
    routes = ("/", "/guide/", "/api/index.html", "/api/cubes.html")
    for route in routes:
        path = html / route.lstrip("/")
        if route.endswith("/"):
            path /= "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "<html><head><title>CytePool - CytePool</title>"
            '<meta name="description" content="An authored description."/>'
            '<link rel="canonical" href="https://old.example/"/>'
            '<link href="https://duplicate.example/" rel="canonical"/>'
            f"</head>{body}</html>"
        )
        if route.startswith("/api/"):
            (html / "api" / "_sources" / f"{path.stem}.md").write_text("# API\n")
    auxiliary = "<html><head></head><body>Useful index or search</body></html>"
    for name in ("search.html", "genindex.html", "py-modindex.html"):
        (html / "api" / name).write_text(auxiliary)
    references = [
        {"kind": "page", "url": f"{base_url}/", "data": "/index.json"},
        {"kind": "page", "url": f"{base_url}/guide", "data": "/guide.json"},
        {"kind": "page", "url": f"{base_url}/guide/", "data": "/guide.json"},
    ]
    (html / "myst.xref.json").write_text(json.dumps({"references": references}))
    monkeypatch.setenv("BASE_URL", base_url)

    build_docs.write_discovery_metadata(html)
    build_docs.write_discovery_metadata(html)

    sitemap = ElementTree.parse(html / "sitemap.xml")
    locations = [node.text for node in sitemap.findall(".//{*}loc")]
    assert sorted(locations) == sorted(
        build_docs.PUBLIC_URL + route for route in routes
    )
    for route in routes:
        path = html / route.lstrip("/")
        if route.endswith("/"):
            path /= "index.html"
        contents = path.read_text()
        assert re.findall(r'<link rel="canonical" href="([^"]+)"', contents) == [
            build_docs.PUBLIC_URL + route
        ]
        assert contents.endswith(body + "</html>")
        assert 'content="An authored description."' in contents
        assert (
            'content="https://docs.nygen.io/assets/brand/logo/cytepool-dark.png"'
            in contents
        )
    for name in ("search.html", "genindex.html", "py-modindex.html"):
        assert (html / "api" / name).read_text() == auxiliary
    assert (
        (html / "robots.txt")
        .read_text()
        .endswith("Sitemap: https://docs.nygen.io/CytePool/sitemap.xml\n")
    )
