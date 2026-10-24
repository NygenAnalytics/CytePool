"""Execute documentation notebooks and save their outputs in place.

Each notebook runs from a fresh kernel in a temporary working directory. On success it
is saved with its outputs and three metadata entries: ``executed_at`` (UTC, when the
run finished), ``execution_seconds`` (its wall time) and ``executed_code_sha256`` (the
fingerprint of the code cells it ran, which the site build and the tests compare with
the code). A failing cell raises and leaves the saved notebook as it was.

With ``--check``, notebooks run the same way but are not saved: the command fails when
the text they print differs from the saved outputs, as it does when a change to the
package changed their results. Figures and standard error are not compared.

Usage, from the repository root::

    uv run --group docs python docs/execute_notebooks.py quickstart
    make -C docs execute PAGES="quickstart"
    make -C docs check-outputs PAGES="quickstart"

Pages are named relative to ``docs/source``, with or without ``.ipynb``.
"""

import argparse
import copy
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from build_docs import SOURCE, check_execution, check_figures, code_fingerprint

TIMEOUT_SECONDS = 600


def notebook_path(page: str) -> Path:
    """Return the source notebook of a page name such as ``quickstart``."""
    path = (SOURCE / page).with_suffix(".ipynb").resolve()
    if not path.is_relative_to(SOURCE.resolve()) or not path.is_file():
        raise SystemExit(f"no notebook {page!r} under {SOURCE}")
    return path


def run(notebook: nbformat.NotebookNode) -> float:
    """Clear a notebook's outputs and execute it in place; return its wall time."""
    for key in ("executed_at", "execution_seconds", "executed_code_sha256"):
        notebook.metadata.pop(key, None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    with tempfile.TemporaryDirectory(prefix="cytepool-docs-") as directory:
        client = NotebookClient(
            notebook,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": directory}},
        )
        start = time.perf_counter()
        client.execute()
        return time.perf_counter() - start


def execute(path: Path) -> float:
    """Execute one notebook and save it; return its wall time in seconds."""
    source = path.read_text()
    notebook = nbformat.reads(source, as_version=4)
    seconds = run(notebook)
    notebook.metadata["executed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    notebook.metadata["execution_seconds"] = round(seconds, 1)
    notebook.metadata["executed_code_sha256"] = code_fingerprint(notebook)
    relative = path.relative_to(SOURCE).as_posix()
    check_execution(notebook, relative)
    if path.read_text() != source:
        raise SystemExit(f"{relative} changed while it ran; not overwriting it")
    temporary = path.with_name(f".{path.name}.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    try:
        check_figures(notebook, relative)
    except ValueError as error:
        print(f"saved, but the site build needs this first: {error}")
    return seconds


def printed_text(notebook: nbformat.NotebookNode) -> list[str]:
    """The text each code cell printed or showed, other than figures and stderr."""
    texts = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        parts = []
        for output in cell.outputs:
            if output.output_type == "stream" and output.name == "stdout":
                parts.append(output.text)
            elif output.output_type in {"display_data", "execute_result"}:
                if not any(kind.startswith("image/") for kind in output.data):
                    parts.append(output.data.get("text/plain", ""))
        texts.append("".join(parts))
    return texts


def check(path: Path) -> list[str]:
    """Execute a copy of a notebook; return the code cells whose text differs."""
    saved = nbformat.read(path, as_version=4)
    check_execution(saved, path.relative_to(SOURCE).as_posix())
    fresh = copy.deepcopy(saved)
    run(fresh)
    return [
        f"code cell {number}: saved {old!r}, now {new!r}"
        for number, (old, new) in enumerate(
            zip(printed_text(saved), printed_text(fresh), strict=True), start=1
        )
        if old != new
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pages", nargs="+", help="notebooks to execute")
    parser.add_argument(
        "--check",
        action="store_true",
        help="execute without saving, and fail if the printed text changed",
    )
    arguments = parser.parse_args()
    failures = []
    for path in dict.fromkeys(notebook_path(page) for page in arguments.pages):
        relative = path.relative_to(SOURCE)
        if arguments.check:
            differences = check(path)
            failures += [f"{relative}, {difference}" for difference in differences]
            print(
                f"{relative}: {'outputs differ' if differences else 'outputs current'}"
            )
        else:
            print(f"executed {relative} in {execute(path):.1f} s")
    if failures:
        raise SystemExit(
            "Outputs no longer match; execute these pages again:\n"
            + "\n".join(failures)
        )


if __name__ == "__main__":
    main()
