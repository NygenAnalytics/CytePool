"""Build the documentation site from saved notebook outputs, without running code.

The MyST sources in ``docs/source`` are copied to ``docs/build/site``, where every
notebook is checked (executed, unchanged since, free of errors) and its figures get
their descriptions as alternative text, and every ``{api}`` role of a page becomes a
link to the API entry it names (``page_index.py``). ``myst build --html`` then
renders the site, and the Sphinx API reference, built beforehand into
``docs/build/api``, is copied into it under ``api/``, where the site's links to
``/api/...`` point. The API entries link back to the pages that use and explain them;
the build checks that every such page is where the links expect it.

A site served below a path, as docs.nygen.io/CytePool/ is, is built with that path in
``BASE_URL`` (``BASE_URL=/CytePool``). MyST puts it before the links of the pages, and
``stage_site`` puts it before the links of the table of contents, which MyST leaves.
"""

import hashlib
import json
import os
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any

import page_index

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
BUILD = DOCS / "build"
SITE = BUILD / "site"
API = BUILD / "api"


def code_fingerprint(notebook: dict[str, Any]) -> str:
    """Return the SHA-256 of a notebook's code cells, which execution records.

    Only code is fingerprinted, so prose can change without a new execution.
    """
    sources = [
        _text(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    encoded = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_execution(notebook: dict[str, Any], page: str) -> None:
    """Raise ValueError unless the saved outputs come from the notebook's current code.

    The notebook must record when it was executed and the fingerprint of the code it
    ran, that fingerprint must match its code now, the datasets it ran with must be
    those it names (metadata ``datasets``) as ``docs/datasets.py`` registers them,
    every code cell must have run, and no output may be an error.
    """
    metadata = notebook.get("metadata", {})
    if not metadata.get("executed_at"):
        raise ValueError(f"{page} has never been executed; run {_execute(page)}")
    if not metadata.get("executed_code_sha256"):
        raise ValueError(
            f"{page} records no fingerprint of its code; run {_execute(page)}"
        )
    if metadata["executed_code_sha256"] != code_fingerprint(notebook):
        raise ValueError(
            f"the code of {page} changed after it was executed; run {_execute(page)}"
        )
    declared = metadata.get("datasets", [])
    if declared or "executed_datasets" in metadata:
        datasets = page_index.docs_module("datasets")
        unknown = sorted(set(declared) - set(datasets.DATASETS))
        if unknown:
            raise ValueError(f"{page} names datasets {unknown} that are not registered")
        registered = {name: dict(datasets.DATASETS[name].files) for name in declared}
        if metadata.get("executed_datasets") != registered:
            raise ValueError(
                f"{page} was executed with other data than the datasets it names, as "
                f"registered in docs/datasets.py; run {_execute(page)}"
            )
    for number, cell in enumerate(_code_cells(notebook), start=1):
        if any(output["output_type"] == "error" for output in cell.get("outputs", [])):
            raise ValueError(
                f"code cell {number} of {page} failed when it was executed"
            )
        count = cell.get("execution_count")
        if _text(cell["source"]).strip() and (type(count) is not int or count < 1):
            raise ValueError(f"code cell {number} of {page} has not been executed")


def check_figures(notebook: dict[str, Any], page: str) -> None:
    """Raise ValueError unless every figure has a description for its alternative text.

    A code cell lists one description per figure it shows, in order, in its metadata
    ``figure_descriptions``.
    """
    for number, cell in enumerate(_code_cells(notebook), start=1):
        figures = _figures(cell)
        descriptions = cell.get("metadata", {}).get("figure_descriptions", [])
        if (
            not isinstance(descriptions, list)
            or len(descriptions) != len(figures)
            or not all(isinstance(text, str) and text.strip() for text in descriptions)
        ):
            raise ValueError(
                f"code cell {number} of {page} shows {len(figures)} figure(s); give "
                "each a description in the list metadata.figure_descriptions"
            )


def toc_with_base_url(
    entries: list[dict[str, Any]], base_url: str
) -> list[dict[str, Any]]:
    """Return a copy of a MyST table of contents with ``base_url`` before every link
    to a path of the site, such as ``/api/index.html``.

    Links to other sites, links that name a host (``//...``) and relative links are
    kept as they are.
    """
    result = []
    for entry in entries:
        entry = dict(entry)
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("/") and not url.startswith("//"):
            entry["url"] = base_url + url
        if "children" in entry:
            entry["children"] = toc_with_base_url(entry["children"], base_url)
        result.append(entry)
    return result


def toc_files(entries: list[dict[str, Any]]) -> list[str]:
    """Return the files of a MyST table of contents, depth first."""
    files = []
    for entry in entries:
        if "file" in entry:
            files.append(entry["file"])
        files.extend(toc_files(entry.get("children", [])))
    return files


def inventory_names(path: Path) -> set[str]:
    """The Python names of a Sphinx inventory (``objects.inv``)."""
    header = path.read_bytes().split(b"\n", 4)
    if len(header) < 5 or not header[0].startswith(b"# Sphinx inventory version 2"):
        raise ValueError(f"{path} is not a Sphinx inventory of version 2")
    lines = zlib.decompress(header[4]).decode("utf-8").splitlines()
    return {
        line.split(" ")[0]
        for line in lines
        if len(line.split(" ")) > 1 and line.split(" ")[1].startswith("py:")
    }


def rewrite_api_roles(site: Path, files: list[str]) -> None:
    """Replace the ``{api}`` roles of the staged pages by links to the API reference.

    Raises:
        ValueError: If a role names nothing that the API reference documents, names
            more than one entry, or names an entry that the built inventory of the
            API reference (``docs/build/api/objects.inv``), when there is one, lacks.
    """
    entries = page_index.api_entries()
    inventory = API / "objects.inv"
    built = inventory_names(inventory) if inventory.is_file() else None
    problems = []

    def link(match: Any, file: str) -> str:
        if not match["role"]:
            return str(match[0])  # inline code, kept as it is
        _, name = page_index.role_parts(match["role"])
        try:
            entry = page_index.resolve(name, entries)
        except ValueError as error:
            problems.append(f"{file}: {{api}}`{match['role']}`: {error}")
            return str(match[0])
        if built is not None and entry.name not in built:
            problems.append(f"{file}: {entry.name} is missing from {inventory}")
        return page_index.role_link(match["role"], entries)

    def rewrite(text: str, file: str) -> str:
        return page_index.map_prose(
            text,
            lambda prose: page_index.ROLE_OR_CODE.sub(lambda m: link(m, file), prose),
        )

    for file in files:
        path = site / file
        if file.endswith(".ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    cell["source"] = rewrite(_text(cell["source"]), file)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        else:
            path.write_text(rewrite(path.read_text(), file))
    if problems:
        raise ValueError("Links to the API reference:\n" + "\n".join(problems))


def check_page_urls(xref: Path, base_url: str = "") -> None:
    """Raise ValueError unless every page of the table of contents is at the URL that
    the links of the API reference to it use, according to the site's
    ``myst.xref.json``."""
    built = {
        reference["url"].removeprefix(base_url) or "/"
        for reference in json.loads(xref.read_text())["references"]
        if reference.get("kind") == "page"
    }
    missing = [
        f"{page.file}: {page.url}"
        for page in page_index.pages(SOURCE)
        if page.url not in built
    ]
    if missing:
        raise ValueError(
            "The API reference links to these pages at URLs the site does not have:\n"
            + "\n".join(missing)
        )


def stage_site() -> Path:
    """Copy the sources to the build directory and prepare them for MyST."""
    # A documentation dependency, imported here so that the tests can use the
    # notebook checks above without it.
    import yaml

    config = yaml.safe_load((SOURCE / "myst.yml").read_text())
    # Keep the theme MyST downloaded, so that later builds need no network.
    themes, kept = SITE / "_build" / "templates", BUILD / "templates"
    if themes.is_dir():
        shutil.rmtree(kept, ignore_errors=True)
        themes.rename(kept)
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(
        SOURCE,
        SITE,
        ignore=shutil.ignore_patterns("_build", "__pycache__", ".ipynb_checkpoints"),
    )
    if kept.is_dir():
        themes.parent.mkdir()
        kept.rename(themes)
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    if base_url:
        config["project"]["toc"] = toc_with_base_url(config["project"]["toc"], base_url)
        (SITE / "myst.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    files = toc_files(config["project"]["toc"])
    for page in files:
        if not page.endswith(".ipynb"):
            continue
        path = SITE / page
        notebook = json.loads(path.read_text())
        check_execution(notebook, page)
        check_figures(notebook, page)
        for cell in _code_cells(notebook):
            descriptions = cell.get("metadata", {}).get("figure_descriptions", [])
            for output, description in zip(_figures(cell), descriptions, strict=True):
                output["data"]["text/plain"] = description
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    rewrite_api_roles(SITE, files)
    return SITE


def main() -> None:
    if not (API / "index.html").is_file():
        raise SystemExit(f"Build the API reference into {API} first: make -C docs api")
    try:
        site = stage_site()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    subprocess.run(["myst", "build", "--html", "--strict"], cwd=site, check=True)
    html = site / "_build" / "html"
    try:
        check_page_urls(
            html / "myst.xref.json", os.environ.get("BASE_URL", "").rstrip("/")
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    shutil.copytree(
        API,
        html / "api",
        ignore=shutil.ignore_patterns(".buildinfo", ".doctrees"),
        dirs_exist_ok=True,
    )
    print(f"Documentation written to {html}")


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _execute(page: str) -> str:
    return f'make -C docs execute PAGES="{page}"'


def _figures(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        output
        for output in cell.get("outputs", [])
        if any(kind.startswith("image/") for kind in output.get("data", {}))
    ]


def _text(source: str | list[str]) -> str:
    return source if isinstance(source, str) else "".join(source)


if __name__ == "__main__":
    main()
