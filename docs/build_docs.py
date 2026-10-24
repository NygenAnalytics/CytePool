"""Build the documentation site from saved notebook outputs, without running code.

The MyST sources in ``docs/source`` are copied to ``docs/build/site``, where every
notebook is checked (executed, unchanged since, free of errors) and its figures get
their descriptions as alternative text. ``myst build --html`` then renders the site,
and the Sphinx API reference, built beforehand into ``docs/build/api``, is copied
into it under ``api/``, where the site's links to ``/api/...`` point.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    ran, that fingerprint must match its code now, every code cell must have run, and
    no output may be an error.
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


def toc_files(entries: list[dict[str, Any]]) -> list[str]:
    """Return the files of a MyST table of contents, depth first."""
    files = []
    for entry in entries:
        if "file" in entry:
            files.append(entry["file"])
        files.extend(toc_files(entry.get("children", [])))
    return files


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
    for page in toc_files(config["project"]["toc"]):
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
    return SITE


def main() -> None:
    if not (API / "index.html").is_file():
        raise SystemExit(f"Build the API reference into {API} first: make -C docs api")
    site = stage_site()
    subprocess.run(["myst", "build", "--html", "--strict"], cwd=site, check=True)
    shutil.copytree(
        API,
        site / "_build" / "html" / "api",
        ignore=shutil.ignore_patterns(".buildinfo", ".doctrees"),
        dirs_exist_ok=True,
    )
    print(f"Documentation written to {site / '_build' / 'html'}")


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
