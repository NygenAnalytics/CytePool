"""Install a built wheel in a clean environment and run a small analysis with it.

The release workflow, ``.github/workflows/publish.yml``, runs this with every Python
and on every platform it tests a release on::

    uv run --no-project python tests/smoke_test_wheel.py dist/cytepool-*.whl

It creates an environment with ``uv venv`` that holds only the wheel and the
dependencies it declares, resolved as a user's installation resolves them, and runs
the analysis twice with that environment's interpreter, in isolated mode from a
scratch folder, so that the package is imported from the environment, never from this
checkout or ``PYTHONPATH``. The analysis pools the cells of
``tests/data/differential_expression`` into a cube, tests the contrasts of a paired
design by ``limma_voom`` and compares the tables of the T cells with limma's. The
first run must compile the package's routines and cache them; the second must load
them all from the cache and compile none. Installing needs network access, or a uv
cache that holds the dependencies (with ``UV_OFFLINE=1``).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "differential_expression"
CONTRASTS = {
    "stim_vs_ctrl": "conditionstim",
    "D2_vs_D1": "donorD2",
    "D2_vs_D4": "donorD2 - donorD4",
}
# The rows of limma's tables in limma.npz.
COLUMNS = (
    "log2_fold_change",
    "average_log2_cpm",
    "t",
    "p_value",
    "adjusted_p_value",
    "b_statistic",
    "standard_error",
)


def analyse() -> None:
    """Test the fixture's contrasts by limma_voom and compare the T cells' with limma's.

    Raises:
        AssertionError: If a table holds other features than limma's, or a value
            differs from limma's by more than 1e-6.
    """
    import numpy as np

    from cytepool.cube import from_cells
    from cytepool.design import Design
    from cytepool.differential_expression import limma_voom

    with np.load(DATA / "cells.npz") as data:
        cells = {key: data[key] for key in data.files}
    with np.load(DATA / "limma.npz") as data:
        limma = {key: data[key] for key in data.files}
    sample_table = {
        key.removeprefix("sample_table "): values
        for key, values in cells.items()
        if key.startswith("sample_table ")
    }
    cube = from_cells(
        cells["counts"],
        samples=cells["samples"],
        groups=cells["groups"],
        sample_table=sample_table,
        layers="counts",
    )
    design = Design(
        cube.samples,
        "~ donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts=CONTRASTS,
    )
    result = limma_voom(cube, design, contrast_standard_errors="limma")
    tested = cube.features["feature_id"][limma["voom T keep"]].tolist()
    for contrast in CONTRASTS:
        table = result.table("T", contrast)
        assert table["feature_id"].tolist() == tested, contrast
        for row, column in enumerate(COLUMNS):
            # tests/test_differential_expression.py holds these to 1e-10 of limma's;
            # other processors and math libraries may round differently, and a broken
            # build or installation differs by far more than 1e-6.
            np.testing.assert_allclose(
                table[column],
                limma[f"voom T {contrast}"][row],
                rtol=1e-6,
                atol=0 if column.endswith("p_value") else 1e-6,
                err_msg=f"{contrast}: {column}",
            )


def run_installed(version: str, from_cache: bool) -> None:
    """Run the analysis with the installed package, and check how it compiled.

    Args:
        version: The version of the wheel, which the package must report.
        from_cache: Whether the routines must all load from the cache, which the
            first run fills; otherwise they must all compile.
    """
    import numba
    import numpy as np
    from numba.core.dispatcher import Dispatcher

    import cytepool

    location = Path(cytepool.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    assert location.is_relative_to(environment), (
        f"cytepool was imported from {location}, not from {environment}"
    )
    assert cytepool.__version__ == version, (
        f"the installed cytepool reports version {cytepool.__version__}, the wheel "
        f"{version}"
    )
    analyse()
    routines = {
        id(value): value
        for name, module in list(sys.modules.items())
        if name.partition(".")[0] == "cytepool"
        for value in vars(module).values()
        if isinstance(value, Dispatcher)
    }.values()
    loaded = sum(sum(routine.stats.cache_hits.values()) for routine in routines)
    compiled = sum(sum(routine.stats.cache_misses.values()) for routine in routines)
    if from_cache:
        assert loaded > 0 and compiled == 0, (
            f"{compiled} routines compiled again, {loaded} loaded from the cache"
        )
    else:
        assert compiled > 0 and loaded == 0, (
            f"{compiled} routines compiled, {loaded} loaded from a cache that "
            "should be empty"
        )
    print(
        f"cytepool {version} with numpy {np.__version__} and numba "
        f"{numba.__version__}: limma's tables; {compiled} routines compiled, "
        f"{loaded} loaded from the cache"
    )


def smoke_test(wheel: Path) -> None:
    """Install a wheel in a clean environment and run the analysis there twice."""
    version = wheel.name.split("-")[1]
    with tempfile.TemporaryDirectory(
        prefix="cytepool-wheel-", ignore_cleanup_errors=True
    ) as scratch:
        root = Path(scratch)
        environment = root / "environment"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)], check=True
        )
        if os.name == "nt":
            python = environment / "Scripts" / "python.exe"
        else:
            python = environment / "bin" / "python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)], check=True
        )
        # A cache of its own, so that the first run compiles everything.
        env = os.environ | {"NUMBA_CACHE_DIR": str(root / "numba-cache")}
        for from_cache in (False, True):
            command = [str(python), "-I", str(Path(__file__).resolve())]
            command += ["--installed", version] + ["--from-cache"] * from_cache
            subprocess.run(command, cwd=root, env=env, check=True)
    print(f"{wheel.name} installs and runs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install a wheel in a clean environment and run an analysis."
    )
    parser.add_argument("wheel", type=Path, nargs="?", help="the wheel to test")
    # The runs inside the environment.
    parser.add_argument("--installed", metavar="VERSION", help=argparse.SUPPRESS)
    parser.add_argument("--from-cache", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.installed is not None:
        run_installed(args.installed, args.from_cache)
    elif args.wheel is not None:
        smoke_test(args.wheel.resolve())
    else:
        parser.error("give the wheel to test")


if __name__ == "__main__":
    main()
