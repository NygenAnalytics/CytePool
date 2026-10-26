"""Checks on the documentation sources in docs/: pages, notebooks, links, API, validation.

The checks read the sources and do not build the site. Reading the table of contents
needs PyYAML, a documentation dependency (``uv sync --group docs``); that test is
skipped without it. The last checks read the References sections of the package's
module docstrings, from the source files.
"""

import ast
import copy
import importlib
import importlib.util
import inspect
import io
import json
import pkgutil
import re
import textwrap
import tokenize
from collections import defaultdict
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

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


def test_output_checks_ignore_only_the_version_of_cytepool_that_ran_a_page(
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
    # Without a recorded version, a different version is a difference.
    assert len(runner.differences(saved, fresh, None, "0.1.dev31+gdef")) == 1


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
    "cytepool._jit": "compiles and caches routines with numba; THIRD_PARTY_NOTICES.md "
    "records what its cache locator follows of numba",
    "cytepool._masked_arrays": "refuses numpy masked arrays at the public boundary",
    "cytepool._messages": "formats lists of ids and names in error messages",
    "cytepool._options": "checks flags and choices among names",
    "cytepool._version": "holds the version that the build writes",
    "cytepool.cube": "the package's namespace; its modules cite their sources",
    "cytepool.cube._axes": "reads the axis tables, cell numbers and cell labels of "
    "CytePool's own cubes",
    "cytepool.cube.cube": "CytePool's own structure of sums per sample, group and "
    "feature; its digests are those of cytepool.cube.fingerprints",
    "cytepool.cube.provenance": "CytePool's own record of how a cube was made, "
    "written as JSON by the standard library",
    "cytepool.design": "the package's namespace; its modules cite their sources",
    "cytepool.io": "the package's namespace; its modules cite their sources",
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
