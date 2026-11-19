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
  ``from cytepool import limma_voom``, is followed through the import that binds it in
  the package's ``__init__.py``. The ``__init__.py`` files are neither checked nor
  followed: they only gather names, and the top level's sets ``__version__``, which
  results record but do not depend on.
- A package imported as a name, such as ``import cytepool as cp``, is read through the
  attribute chains on that name, such as ``cp.limma_voom``.
- What this reading cannot see stops the helper with an error that names the file: a
  star import from CytePool, a package name used other than in an attribute chain, a
  name or module that does not exist, a relative import in a script, and a string in a
  script that imports CytePool, as code that the script runs in another process does.

A file's code is its syntax tree without docstrings, written out by ``ast.unparse``.
Comments, docstrings and layout do not count; every other string, such as an error
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
import os
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
SCOPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
Scope = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


# The code of a file. The functions that read a source are cached by its bytes, since
# most modules are read once for each script that reaches them.


def _with_docstrings(tree: ast.AST) -> list[Scope]:
    """The modules, classes and functions of a tree whose body opens with a docstring:
    a string constant as the first statement, as Python defines it."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, SCOPES)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ]


@cache
def code_identity(source: bytes) -> str:
    """The code of a Python source without comments, docstrings or layout: its syntax
    tree without docstrings, written out by ``ast.unparse``. A body that held only a
    docstring becomes ``pass``."""
    tree = ast.parse(source)
    for node in _with_docstrings(tree):
        node.body = node.body[1:] or (
            [] if isinstance(node, ast.Module) else [ast.Pass()]
        )
    return ast.unparse(tree)


def code_digest(files: Iterable[Path], root: Path) -> str:
    """A digest of the code of some files: SHA-256 over the path of each file, relative
    to ``root``, and the SHA-256 of its code, in the order of the paths, cut to 12
    hexadecimal digits."""
    digest = hashlib.sha256()
    for path, file in sorted(
        (file.relative_to(root).as_posix(), file) for file in files
    ):
        code = hashlib.sha256(code_identity(file.read_bytes()).encode()).hexdigest()
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


def _module_file(root: Path, module: str) -> Path | None:
    """The file of a module of CytePool, ``a/b.py``, or ``a/b/__init__.py`` for a
    package; None when there is none. Names are compared exactly, as imports compare
    them, also on file systems that ignore case."""
    *folders, name = module.split(".")
    folder = root.joinpath(*folders)
    entries = set(os.listdir(folder)) if folder.is_dir() else set()
    if f"{name}.py" in entries:
        return folder / f"{name}.py"
    if name in entries and (folder / name / "__init__.py").is_file():
        return folder / name / "__init__.py"
    return None


def _is_package(root: Path, module: str) -> bool:
    file = _module_file(root, module)
    return file is not None and file.name == "__init__.py"


def _imported_from(node: ast.ImportFrom, package: str | None, file: Path) -> str:
    """The module a ``from`` statement imports from, with relative imports resolved
    against ``package``, which is None for a script."""
    if not node.level:
        return node.module or ""
    if package is None:
        raise ValueError(
            f"{file}, line {node.lineno}: a relative import in a script; import from "
            f"{PACKAGE} by the full name of the module"
        )
    try:
        return importlib.util.resolve_name(
            "." * node.level + (node.module or ""), package
        )
    except ImportError as error:
        raise ValueError(f"{file}, line {node.lineno}: {error}") from error


def _binding(init: Path, package: str, name: str) -> str | None:
    """The dotted name that a package's ``__init__.py`` binds ``name`` to, or None when
    the ``__init__.py`` defines the name itself, binds it from outside CytePool, or the
    name is an attribute of every module, such as ``__file__``. The last binding at
    the top level of the file counts, as when Python runs it.

    Raises:
        ValueError: If the ``__init__.py`` does not bind the name.
    """
    bound, source = name.startswith("__") and name.endswith("__"), None
    for statement in ast.parse(init.read_bytes(), filename=str(init)).body:
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if (alias.asname or alias.name) == name:
                    base = _imported_from(statement, package, init)
                    bound, source = True, f"{base}.{alias.name}"
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname == name:
                    bound, source = True, alias.name
                elif not alias.asname and alias.name.partition(".")[0] == name:
                    bound, source = True, name
        elif isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            if statement.name == name:
                bound, source = True, None
        elif isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if any(
                isinstance(node, ast.Name) and node.id == name
                for target in targets
                for node in ast.walk(target)
            ):
                bound, source = True, None
    if not bound:
        raise ValueError(f"{init} does not bind {name}, which {package}.{name} names")
    return source if source is not None and _in_package(source) else None


def _defining_module(root: Path, name: str) -> str | None:
    """The module that a dotted name of CytePool reaches: the module the name names, or
    else the module file that defines it, following the bindings of ``__init__.py``
    files; None for a name that a package's ``__init__.py`` defines itself, such as
    ``__version__``. Attributes after the defining module, as in ``Design.check``, are
    ignored. The name starts with the name of a module, as the callers check.

    Raises:
        ValueError: If a package's ``__init__.py`` does not bind the name it is asked.
    """
    parts = name.split(".")
    # The longest start of the name that names a module.
    end = max(
        end
        for end in range(1, len(parts) + 1)
        if _module_file(root, ".".join(parts[:end])) is not None
    )
    module = ".".join(parts[:end])
    if end == len(parts) or not _is_package(root, module):
        return module
    init = _module_file(root, module)
    assert init is not None
    source = _binding(init, module, parts[end])
    return None if source is None else _defining_module(root, source)


def _reached(root: Path, name: str, file: Path, line: int) -> str | None:
    """The module that a name a file uses reaches, with the file and the line of the
    use in the errors."""
    try:
        return _defining_module(root, name)
    except ValueError as error:
        raise ValueError(f"{file}, line {line}: {error}") from error


# The modules a file imports.


def _attribute_chains(
    tree: ast.AST, packages: dict[str, str], root: Path, file: Path
) -> set[str]:
    """The modules that attribute chains on names bound to packages reach, such as
    ``cp.limma_voom`` after ``import cytepool as cp``.

    Raises:
        ValueError: If a name bound to a package is used other than in an attribute
            chain, since the names it reaches cannot then be read.
    """
    modules, bases = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain, value = [], node
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id in packages:
                bases.add(id(value))
                name = ".".join([packages[value.id], *reversed(chain)])
                if (module := _reached(root, name, file, node.lineno)) is not None:
                    modules.add(module)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in packages
            and isinstance(node.ctx, ast.Load)
            and id(node) not in bases
        ):
            raise ValueError(
                f"{file}, line {node.lineno}: the package {packages[node.id]} is used "
                f"as a value; use its names through attributes, such as "
                f"{node.id}.name, or import them by name"
            )
    return modules


def _imports(file: Path, root: Path, package: str | None) -> set[str]:
    """The modules of CytePool that the code of a file imports or reads names from.
    ``package`` is the package of a module, for its relative imports, and None for a
    script.

    Raises:
        ValueError: If the code imports in a way that this reading cannot follow.
    """
    source = file.read_bytes()
    modules: set[str] = set()
    packages: dict[str, str] = {}  # names bound to a package, with the package
    for node in _import_statements(source):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _in_package(alias.name):
                    continue
                if _module_file(root, alias.name) is None:
                    raise ValueError(
                        f"{file}, line {node.lineno}: no module {alias.name}"
                    )
                modules.add(alias.name)
                # `import cytepool.stats.voom` binds the name cytepool, as Python does.
                bound = alias.asname or PACKAGE
                target = alias.name if alias.asname else PACKAGE
                if _is_package(root, target):
                    packages[bound] = target
            continue
        base = _imported_from(node, package, file)
        if not _in_package(base):
            continue
        if _module_file(root, base) is None:
            raise ValueError(f"{file}, line {node.lineno}: no module {base}")
        for alias in node.names:
            if alias.name == "*":
                raise ValueError(
                    f"{file}, line {node.lineno}: a star import from {base}; "
                    "import the names it uses one by one"
                )
            module = _reached(root, f"{base}.{alias.name}", file, node.lineno)
            if module is None:
                continue
            modules.add(module)
            if _is_package(root, module):
                packages[alias.asname or alias.name] = module
    if packages:
        modules |= _attribute_chains(ast.parse(source), packages, root, file)
    return modules


def _refuse_imports_in_strings(script: Path) -> None:
    """Stop at a string of a script, other than a docstring, with a line that imports
    CytePool: code that the script runs in another process, which this reading cannot
    follow."""
    tree = ast.parse(script.read_bytes(), filename=str(script))
    docstrings = {
        id(part) for node in _with_docstrings(tree) for part in ast.walk(node.body[0])
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and IMPORT_IN_STRING.search(node.value)
        ):
            raise ValueError(
                f"{script}, line {node.lineno}: a string imports {PACKAGE}, which the "
                "helper cannot follow; to run CytePool in another process, run the "
                "script itself there, with an argument"
            )


def checked_files(script: Path, root: Path) -> list[Path]:
    """The files whose code a validation script checks: the script and every module of
    CytePool it imports, directly or through other modules, other than the
    ``__init__.py`` files, in the order of their paths.

    Raises:
        ValueError: If the script or a module it reaches imports in a way that this
            reading cannot follow.
    """
    _refuse_imports_in_strings(script)
    files, seen = {script}, set()
    queue = list(_imports(script, root, package=None))
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        file = _module_file(root, module)
        assert file is not None, module  # _imports returns modules that exist
        if file.name == "__init__.py":
            continue
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


def recorded_digests(path: Path) -> dict[str, str]:
    """The digest of the code that each script's recorded run checked, by the script's
    file name, from a file of lines such as ``"voom.py" = "0f3a9c21b7d4"``.

    Raises:
        ValueError: If a digest is not 12 lowercase hexadecimal digits.
    """
    records = tomllib.loads(path.read_text())
    for script, digest in records.items():
        if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
            raise ValueError(
                f"{path}: {script} = {digest!r}; write each record as "
                '"<script>.py" = "<digest>", with a digest of 12 lowercase '
                'hexadecimal digits, such as "0f3a9c21b7d4"'
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
            "remove their records"
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


if __name__ == "__main__":
    sys.exit(main())
