"""Which validation scripts check code that has changed since their recorded run.

The Validation page reports the numbers that each script in ``validation/`` printed in
its last run. The scripts need R and other reference tools, so they run by hand, not in
the tests, and the package keeps changing between runs. ``validation_runs.toml``
records, for each script, a digest of the code that its last run checked. This helper
computes the digests of the working tree and lists the scripts whose code has changed
since their recorded run, or that have no record, so that reviewers know which scripts
to run again.

The code a script checks is the script and every module of CytePool that it imports,
directly or through other modules, read from the source files without importing them:

- Every ``import`` and ``from`` statement of a file counts, those inside functions
  included, and relative imports are resolved against the file's package.
- A name is read in the module that defines it: ``from cytepool.stats.voom import voom``
  reaches ``cytepool/stats/voom.py``, and a name that a package gathers, such as
  ``from cytepool import limma_voom``, is followed through the ``from`` import that
  binds it in the package's ``__init__.py``. The ``__init__.py`` files are neither
  checked nor followed: they only gather names, and the top level's sets
  ``__version__``, which results record but do not depend on.
- What this reading cannot see stops the helper with an error that names the file and
  line: a package of CytePool imported as a name, such as ``import cytepool as cp``,
  whose uses would have to be read; a star import; a name that a package's
  ``__init__.py`` does not import from a module; a module that does not exist; a
  relative import in a script; and a string in a script that imports CytePool, as code
  that the script runs in another process does.

A file's code is its syntax tree without the statements that are only a string, such
as docstrings, for which Python runs nothing, written out by ``ast.unparse``. Comments,
docstrings and layout do not count; every string the code uses, such as an error
message or R code, does. The digest is SHA-256 over the path and the code of each
checked file, cut to 12 hexadecimal digits. It needs no history, so a record stays
valid through rebases and squash merges that leave the checked code as it was.

Usage, from the repository root::

    uv run python docs/changed_since_run.py

The helper prints one line for each script whose code has changed or that has no
record, and then exits with status 1; otherwise it prints one line and exits with 0.
"""

import ast
import hashlib
import importlib.util
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = Path(__file__).resolve().parent / "validation_runs.toml"
PACKAGE = "cytepool"
DIGEST = re.compile(r"[0-9a-f]{12}")
# A line of a string that imports CytePool, as code run in another process has.
IMPORT_IN_STRING = re.compile(rf"^\s*(from|import)\s+{PACKAGE}\b", re.M)


# The code of a file. The functions that read a source are cached by its bytes, since
# most modules are read once for each script that reaches them.


def _is_text(statement: ast.AST) -> bool:
    """Whether a statement is only a string, as a docstring is."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


@cache
def normalized_code(source: bytes) -> str:
    """The code of a Python source without comments, docstrings or layout: its syntax
    tree without the statements that are only a string, written out by
    ``ast.unparse``. A body left empty becomes ``pass``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if isinstance(statements, list) and any(map(_is_text, statements)):
                kept = [s for s in statements if not _is_text(s)]
                empty = [ast.Pass()] if field == "body" and node is not tree else []
                setattr(node, field, kept or empty)
    return ast.unparse(tree)


def code_digest(files: Iterable[Path], root: Path) -> str:
    """A digest of the code of some files: SHA-256 over the path of each file, relative
    to ``root``, and the SHA-256 of its code, in the order of the paths, cut to 12
    hexadecimal digits."""
    digest = hashlib.sha256()
    for path, file in sorted(
        (file.relative_to(root).as_posix(), file) for file in files
    ):
        code = hashlib.sha256(normalized_code(file.read_bytes()).encode()).hexdigest()
        digest.update(f"{path}\0{code}\n".encode())
    return digest.hexdigest()[:12]


@cache
def _import_statements(source: bytes) -> tuple[ast.Import | ast.ImportFrom, ...]:
    """Every ``import`` and ``from`` statement of a source, those inside functions and
    other blocks included. The statements are shared, so they are only read."""
    return tuple(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
    )


# Where the names of CytePool are defined, read from the package's files.


def _in_package(name: str) -> bool:
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _where(file: Path, root: Path, line: int) -> str:
    return f"{file.relative_to(root).as_posix()}, line {line}"


def _module_file(root: Path, module: str) -> Path | None:
    """The file of a module of CytePool, ``a/b/__init__.py`` for a package or else
    ``a/b.py``, as Python looks for them; None when there is none."""
    path = root.joinpath(*module.split("."))
    for file in (path / "__init__.py", path.with_suffix(".py")):
        if file.is_file():
            return file
    return None


def _is_package(root: Path, module: str) -> bool:
    file = _module_file(root, module)
    return file is not None and file.name == "__init__.py"


def _imported_from(node: ast.ImportFrom, package: str | None, where: str) -> str:
    """The module a ``from`` statement imports from, with relative imports resolved
    against ``package``, which is None for a script."""
    try:
        return importlib.util.resolve_name(
            "." * node.level + (node.module or ""), package
        )
    except ImportError as error:
        raise ValueError(f"{where}: {error}") from error


def _binding(root: Path, package: str, name: str) -> str | None:
    """The dotted name that the last ``from`` import at the top level of a package's
    ``__init__.py`` binds ``name`` to; None for a name from outside CytePool, or for
    a special name, such as ``__version__`` or ``__file__``, which holds no code.

    Raises:
        ValueError: If no ``from`` import binds the name.
    """
    init = root.joinpath(*package.split("."), "__init__.py")
    source = None
    for statement in ast.parse(init.read_bytes(), filename=str(init)).body:
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if (alias.asname or alias.name) == name:
                    where = _where(init, root, statement.lineno)
                    base = _imported_from(statement, package, where)
                    source = f"{base}.{alias.name}"
    if source is None and not (name.startswith("__") and name.endswith("__")):
        raise ValueError(
            f"{package} has no name {name} that its __init__.py imports from a module; "
            "check the name, and define code in a module rather than in __init__.py"
        )
    return source if source is not None and _in_package(source) else None


def _defining_module(root: Path, module: str, name: str) -> str | None:
    """The module that defines ``name`` of ``module``: the submodule of that name, if
    there is one, else ``module`` itself, or, for a package, the module that its
    ``__init__.py`` imports the name from, followed until a module defines it; None
    for a name from outside CytePool, or a special name, such as ``__version__``.

    Raises:
        ValueError: If a module does not exist, or a package does not import the name.
    """
    if _module_file(root, module) is None:
        raise ValueError(f"no module {module}")
    if _module_file(root, f"{module}.{name}") is not None:
        return f"{module}.{name}"
    if not _is_package(root, module):
        return module
    source = _binding(root, module, name)
    return None if source is None else _defining_module(root, *source.rsplit(".", 1))


def _package_as_name(where: str, package: str) -> ValueError:
    return ValueError(
        f"{where}: the package {package} is imported as a name, whose uses this "
        f"reading does not follow; import the names the code uses, such as "
        f"from {package} import name"
    )


# The modules a file imports.


def _imports(file: Path, root: Path, package: str | None) -> set[str]:
    """The modules of CytePool that the code of a file imports, or imports names from.
    ``package`` is the package of a module, for its relative imports, and None for a
    script.

    Raises:
        ValueError: If the code imports in a way that this reading cannot follow.
    """
    modules = set()
    for node in _import_statements(file.read_bytes()):
        where = _where(file, root, node.lineno)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _in_package(alias.name):
                    continue
                if _module_file(root, alias.name) is None:
                    raise ValueError(f"{where}: no module {alias.name}")
                # `import cytepool.stats.voom` binds the name cytepool, as Python does.
                bound = alias.name if alias.asname else PACKAGE
                if _is_package(root, bound):
                    raise _package_as_name(where, bound)
                modules.add(alias.name)
            continue
        base = _imported_from(node, package, where)
        if not _in_package(base):
            continue
        for alias in node.names:
            if alias.name == "*":
                raise ValueError(
                    f"{where}: a star import from {base}; "
                    "import the names it uses one by one"
                )
            try:
                module = _defining_module(root, base, alias.name)
            except ValueError as error:
                raise ValueError(f"{where}: {error}") from error
            if module is None:
                continue
            if _is_package(root, module):
                raise _package_as_name(where, module)
            modules.add(module)
    return modules


def _refuse_imports_in_strings(script: Path, root: Path) -> None:
    """Stop at a string of a script with a line that imports CytePool, unless the
    string is a statement of its own, as a docstring is: code that the script runs in
    another process, which this reading cannot follow."""
    tree = ast.parse(script.read_bytes(), filename=str(script))
    text = {id(node.value) for node in ast.walk(tree) if _is_text(node)}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in text
            and IMPORT_IN_STRING.search(node.value)
        ):
            raise ValueError(
                f"{_where(script, root, node.lineno)}: a string imports {PACKAGE}, "
                "which the helper cannot follow; to run CytePool in another process, "
                "run the script itself there, with an argument"
            )


def checked_files(script: Path, root: Path) -> list[Path]:
    """The files whose code a validation script checks: the script and every module of
    CytePool it imports, directly or through other modules, in the order of their
    paths.

    Raises:
        ValueError: If the script or a module it reaches imports in a way that this
            reading cannot follow.
    """
    _refuse_imports_in_strings(script, root)
    files, seen = {script}, set()
    queue = list(_imports(script, root, package=None))
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        file = _module_file(root, module)
        assert file is not None, module  # _imports returns modules that exist
        files.add(file)
        queue.extend(_imports(file, root, module.rpartition(".")[0]))
    return sorted(files)


def script_digests(root: Path) -> dict[str, str]:
    """The digest of the code that each script in ``validation/`` checks, by the
    script's file name."""
    return {
        script.name: code_digest(checked_files(script, root), root)
        for script in sorted((root / "validation").glob("*.py"))
    }


# The records.

FORM = (
    '"<script>.py" = "<digest>", with a digest of 12 lowercase hexadecimal digits, '
    'such as "0f3a9c21b7d4"'
)


def recorded_digests(path: Path) -> dict[str, str]:
    """The digest of the code that each script's recorded run checked, by the script's
    file name, from a file of lines such as ``"voom.py" = "0f3a9c21b7d4"``.

    Raises:
        ValueError: If the file holds anything else.
    """
    try:
        records = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"{path.name}: {error}; write each record as {FORM}"
        ) from error
    for script, digest in records.items():
        if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
            raise ValueError(
                f"{path.name}: {script} = {digest!r}; write each record as {FORM}"
            )
    return records


@dataclass(frozen=True)
class Change:
    """A script whose checked code differs from its record, or that has none."""

    script: str
    recorded: str | None  # None when no run is recorded
    current: str


def changed_since_run(
    recorded: dict[str, str], current: dict[str, str]
) -> list[Change]:
    """The scripts whose current digest differs from their recorded one, or that have no
    record, in the order of their names.

    Raises:
        ValueError: If a run is recorded for a script that does not exist.
    """
    missing = sorted(set(recorded) - set(current))
    if missing:
        raise ValueError(
            f"runs are recorded for scripts that do not exist: {', '.join(missing)}; "
            f"remove their lines from {RECORDS.name}"
        )
    return [
        Change(script, recorded.get(script), digest)
        for script, digest in sorted(current.items())
        if recorded.get(script) != digest
    ]


def main(root: Path = ROOT, records: Path = RECORDS) -> int:
    changes = changed_since_run(recorded_digests(records), script_digests(root))
    for change in changes:
        if change.recorded is None:
            print(f"validation/{change.script}: no recorded run (now {change.current})")
        else:
            print(
                f"validation/{change.script}: the code it checks has changed since its "
                f"recorded run (recorded {change.recorded}, now {change.current})"
            )
    if not changes:
        print(
            "No script in validation/ checks code that has changed since its "
            "recorded run."
        )
    return 1 if changes else 0


if __name__ == "__main__":  # pragma: no cover - the command line, which runs main()
    sys.exit(main())
