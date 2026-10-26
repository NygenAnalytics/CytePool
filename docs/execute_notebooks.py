"""Execute documentation notebooks and save their outputs in place.

Each notebook runs from a fresh kernel in a temporary working directory. A notebook
that reads shared data names the datasets in its metadata ``datasets`` (ids of
``docs/datasets.py``); they are prepared in the cache first, if they are not there
yet, and their files are linked into the working directory, where the notebook opens
them by name, as ``read_cube("kang2018.cube.zarr")``. On success the notebook is
saved with its outputs and five metadata entries: ``executed_at`` (UTC, when the run
finished), ``execution_seconds`` (its wall time), ``executed_code_sha256`` (the
fingerprint of the code cells it ran, which the site build and the tests compare with
the code), ``executed_with`` (the version of CytePool that ran it) and
``executed_datasets`` (the fingerprint of every dataset file it was given, which the
tests compare with the registry). A failing cell raises and leaves the saved notebook
as it was.

With ``--check``, notebooks run the same way but are not saved: the command fails when
the text they print differs from the saved outputs, as it does when a change to the
package changed their results. Figures and standard error are not compared, and
neither is CytePool's own version, which changes with every commit of a development
version and which a page can print: the saved text is compared with the version that
ran it replaced by the version running now, and a version written after "CytePool ",
as a cube's history writes the version that built it, is left out of both. A dataset
prepared before the page ran was built by another version, and on CI every dataset is
built again by the version being checked.

Usage, from the repository root::

    uv run --group docs python docs/execute_notebooks.py quickstart
    make -C docs execute PAGES="quickstart"
    make -C docs check-outputs PAGES="quickstart"

Pages are named relative to ``docs/source``, with or without ``.ipynb``.
"""

import argparse
import copy
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import cytepool
import page_index
from build_docs import SOURCE, check_execution, check_figures, code_fingerprint

datasets = page_index.docs_module("datasets")

TIMEOUT_SECONDS = 600


def notebook_path(page: str) -> Path:
    """Return the source notebook of a page name such as ``quickstart``."""
    path = (SOURCE / page).with_suffix(".ipynb").resolve()
    if not path.is_relative_to(SOURCE.resolve()) or not path.is_file():
        raise SystemExit(f"no notebook {page!r} under {SOURCE}")
    return path


def provide_datasets(
    notebook: nbformat.NotebookNode, directory: Path
) -> dict[str, dict[str, str]]:
    """Prepare the datasets a notebook names and link their files into the directory
    it runs in; return the fingerprint of each file, by dataset."""
    declared = notebook.metadata.get("datasets", [])
    unknown = sorted(set(declared) - set(datasets.DATASETS))
    if unknown:
        raise SystemExit(
            f"the notebook names datasets {unknown}, which docs/datasets.py does not "
            f"register; choose from {sorted(datasets.DATASETS)}"
        )
    cache = datasets.data_directory()
    provided = {}
    for dataset_id in declared:
        dataset = datasets.DATASETS[dataset_id]
        problems = datasets.prepare(dataset, cache)
        if problems:
            raise SystemExit(f"{dataset_id}: " + "; ".join(problems))
        for name in dataset.files:
            (directory / name).symlink_to(
                cache / name, target_is_directory=(cache / name).is_dir()
            )
        provided[dataset_id] = dict(dataset.files)
    return provided


def run(notebook: nbformat.NotebookNode) -> float:
    """Clear a notebook's outputs and execute it in place; return its wall time."""
    for key in (
        "executed_at",
        "execution_seconds",
        "executed_code_sha256",
        "executed_with",
        "executed_datasets",
    ):
        notebook.metadata.pop(key, None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    with tempfile.TemporaryDirectory(prefix="cytepool-docs-") as directory:
        provided = provide_datasets(notebook, Path(directory))
        client = NotebookClient(
            notebook,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": directory}},
        )
        start = time.perf_counter()
        client.execute()
        seconds = time.perf_counter() - start
    if provided:
        notebook.metadata["executed_datasets"] = provided
    return seconds


def execute(path: Path) -> float:
    """Execute one notebook and save it; return its wall time in seconds."""
    source = path.read_text()
    notebook = nbformat.reads(source, as_version=4)
    seconds = run(notebook)
    notebook.metadata["executed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    notebook.metadata["execution_seconds"] = round(seconds, 1)
    notebook.metadata["executed_code_sha256"] = code_fingerprint(notebook)
    notebook.metadata["executed_with"] = cytepool.__version__
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


# A version of CytePool where printed text names it, as in "CytePool 0.1.dev42+g6991e63f8".
_NAMED_VERSION = re.compile(
    r"(?<=CytePool )\d+(?:\.\d+)*(?:(?:a|b|rc|\.dev|\.post)\d+)*(?:\+[0-9A-Za-z.]+)?"
)


def differences(
    saved: list[str], fresh: list[str], saved_version: str | None, version: str
) -> list[str]:
    """The code cells whose printed text differs, other than in CytePool's version.

    ``saved_version`` is the version that ran the saved notebook (None if it was
    not recorded) and ``version`` the one that ran it again. A version written after
    "CytePool " is left out of both texts.
    """

    def compared(text: str) -> str:
        return _NAMED_VERSION.sub("<version>", text)

    return [
        f"code cell {number}: saved {old!r}, now {new!r}"
        for number, (old, new) in enumerate(zip(saved, fresh, strict=True), start=1)
        if compared(old.replace(saved_version, version) if saved_version else old)
        != compared(new)
    ]


def check(path: Path) -> list[str]:
    """Execute a copy of a notebook; return the code cells whose text differs."""
    saved = nbformat.read(path, as_version=4)
    check_execution(saved, path.relative_to(SOURCE).as_posix())
    fresh = copy.deepcopy(saved)
    run(fresh)
    return differences(
        printed_text(saved),
        printed_text(fresh),
        saved.metadata.get("executed_with"),
        cytepool.__version__,
    )


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
