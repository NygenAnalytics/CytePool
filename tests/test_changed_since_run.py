"""Tests of docs/changed_since_run.py, which lists the validation scripts whose checked
code has changed since their recorded run.

The packages here are small trees of files named ``cytepool`` in a temporary folder:
the helper reads source files and never imports them. Each fixture reaches its
expected files only through the rule it tests.
"""

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "docs"))
import page_index  # noqa: E402

helper = page_index.docs_module("changed_since_run")


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


def _checked(root: Path, files: dict[str, str]) -> list[str]:
    """The checked files of validation/check.py in a tree of ``files``, as paths."""
    _write(root, {"cytepool/__init__.py": "", **files})
    script = root / "validation" / "check.py"
    return [p.relative_to(root).as_posix() for p in helper.checked_files(script, root)]


# The code of a file.


def _same_code(first: str, second: str) -> bool:
    identity = helper.code_identity
    return identity(textwrap.dedent(first).encode()) == identity(
        textwrap.dedent(second).encode()
    )


def test_code_ignores_comments_docstrings_and_layout() -> None:
    code = textwrap.dedent('''\
        """A module."""

        import numpy as np


        class Fit:
            """A fit."""

            def mean(self, values):
                """The mean."""
                return np.mean(values)  # of every value


        async def fetch(url, timeout=10.0):
            """Fetch a page."""
            if timeout <= 0:
                raise ValueError("timeout must be positive")
            return await get(url, timeout=timeout)
        ''')
    same = '''\
        """A module, documented at length.

        With a second paragraph.
        """
        # A comment above the imports.
        import numpy as np
        class Fit:
            """Another description."""
            def mean(self, values):
                return np.mean(
                    values,
                )
        async def fetch(url, timeout=(10.0)):
            "One line."

            if timeout <= 0:  # nothing to wait for
                raise ValueError(
                    "timeout must be positive"
                )
            return await get(url, timeout=timeout)
        '''

    assert _same_code(code, same)
    # A changed message, number, operator or import is a change of code.
    assert not _same_code(code, code.replace("must be positive", "must exceed 0"))
    assert not _same_code(code, code.replace("10.0", "10.5"))
    assert not _same_code(code, code.replace("<=", "<"))
    assert not _same_code(
        code, code.replace("import numpy as np", "import numpy as np\nimport scipy")
    )


def test_a_body_of_only_a_docstring_reads_as_pass() -> None:
    # Both bodies do nothing; ast.unparse fails on an empty body.
    assert _same_code('def f():\n    "Nothing."\n', "def f():\n    pass\n")
    assert _same_code('class C:\n    "Nothing."\n', "class C:\n    pass\n")
    assert _same_code('"""Only a docstring."""\n', "")
    assert not _same_code('def f():\n    "Nothing."\n', "def f():\n    return None\n")


def test_strings_that_are_not_docstrings_count() -> None:
    # Python takes as a docstring only a str constant that opens a body: an f-string,
    # bytes, or a later string are code.
    assert not _same_code('def f(x):\n    f"{x}"\n', "def f(x):\n    pass\n")
    assert not _same_code('def f():\n    b"bytes"\n', "def f():\n    pass\n")
    assert not _same_code('x = 1\n"A note."\n', "x = 1\n")
    assert not _same_code(
        'def f():\n    "Doc."\n    "Second."\n', 'def f():\n    "Doc."\n'
    )


SOURCE = '''\
"""A fixed source of the constructs CytePool uses, to pin ast.unparse."""

from __future__ import annotations

import numpy as np
from ._jit import jit as compile_loop


@compile_loop(cache=True)
def total(values: np.ndarray, *, scale: float = 1e-3, **options) -> float:
    """Sum the values."""
    if not (0 < scale <= 1 and values.ndim in (1, 2)):
        raise ValueError(f"scale must be in (0, 1], got {scale!r:>8}")
    result = 0.0
    for i, value in enumerate(values[::-1, ...]):
        result += value * scale if value == value else -0.0
    while (count := len(options)) > 0x10:
        options.popitem()
    return result ** 2 // 3 % 7 @ np.eye(2) | count & ~1 ^ 2 << 1 >> 1


class Fit[T](dict[str, T]):
    """A fit."""

    type Alias = list[T]
    labels: tuple[str, ...] = ()

    async def rows(self):
        async with self.lock as lock, open("x") as f:
            yield {key: [v async for v in value] for key, value in self.items()}

    def match(self, item):
        match item:
            case {"name": str(name), **rest} if rest:
                return name, rest
            case [first, *_] | (first,):
                return lambda x, /, y=first: (x, y)
        try:
            pass
        except* (ValueError, TypeError) as group:
            raise RuntimeError("no match") from group
        finally:
            del item
        global CACHE
        return {*self} - {None}, not self, b"\\x00", r"\\d+", 1j, ...
'''


def test_the_code_of_a_fixed_source_keeps_its_saved_digest(tmp_path: Path) -> None:
    # Saved under Python 3.13 and checked under 3.14 and 3.15: a change of
    # ast.unparse in a new Python fails here, before it changes every record.
    _write(tmp_path, {"cytepool/fit.py": SOURCE})
    identity = helper.code_identity(SOURCE.encode())

    assert '"""' not in identity and "Sum the values" not in identity
    assert (
        helper.code_digest([tmp_path / "cytepool/fit.py"], tmp_path) == "94662e984101"
    )


# The files a script checks.


def test_a_name_is_read_in_the_module_that_defines_it(tmp_path: Path) -> None:
    files = {
        "cytepool/a.py": "def f():\n    return 1\n",
        "cytepool/b.py": "def g():\n    return 2\n",
        "validation/check.py": "from cytepool.a import f\n",
    }

    assert _checked(tmp_path, files) == ["cytepool/a.py", "validation/check.py"]


def test_a_name_a_package_gathers_is_followed_to_its_module(tmp_path: Path) -> None:
    # K is a constant, which has no __module__; a reading that stopped at the
    # __init__.py that gathers it would miss sub/x.py.
    files = {
        "cytepool/sub/__init__.py": "from .x import K\n",
        "cytepool/sub/x.py": "K = 3\n",
        "validation/check.py": "from cytepool.sub import K\n",
    }

    assert _checked(tmp_path, files) == ["cytepool/sub/x.py", "validation/check.py"]


def test_names_are_followed_through_several_packages(tmp_path: Path) -> None:
    files = {
        "cytepool/__init__.py": "from .sub import g\n",
        "cytepool/sub/__init__.py": "from .y import g as g\n",
        "cytepool/sub/y.py": "def g():\n    return 4\n",
        "validation/check.py": "from cytepool import g\n",
    }

    assert _checked(tmp_path, files) == ["cytepool/sub/y.py", "validation/check.py"]


def test_relative_imports_are_resolved_by_their_level(tmp_path: Path) -> None:
    # sub/top.py and n.py are decoys that a wrong level would reach instead.
    files = {
        "cytepool/sub/__init__.py": "",
        "cytepool/sub/m.py": "from .n import h\nfrom ..top import t\n",
        "cytepool/sub/n.py": "h = 1\n",
        "cytepool/top.py": "t = 2\n",
        "cytepool/sub/top.py": "t = 3\n",
        "cytepool/n.py": "h = 4\n",
        "validation/check.py": "from cytepool.sub.m import h\n",
    }

    assert _checked(tmp_path, files) == [
        "cytepool/sub/m.py",
        "cytepool/sub/n.py",
        "cytepool/top.py",
        "validation/check.py",
    ]


def test_imports_inside_functions_and_blocks_count(tmp_path: Path) -> None:
    files = {
        "cytepool/a.py": """\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from .types import Kind


            def f():
                from cytepool.c import g

                return g()
            """,
        "cytepool/c.py": "def g():\n    return 1\n",
        "cytepool/types.py": "Kind = int\n",
        "validation/check.py": "def main():\n    from cytepool.a import f\n",
    }

    assert _checked(tmp_path, files) == [
        "cytepool/a.py",
        "cytepool/c.py",
        "cytepool/types.py",
        "validation/check.py",
    ]


def test_a_cycle_of_imports_ends(tmp_path: Path) -> None:
    files = {
        "cytepool/p.py": "from .q import qq\n\ndef pp():\n    return qq()\n",
        "cytepool/q.py": "from .p import pp\n\ndef qq():\n    return 1\n",
        "validation/check.py": "from cytepool.p import pp\n",
    }

    assert _checked(tmp_path, files) == [
        "cytepool/p.py",
        "cytepool/q.py",
        "validation/check.py",
    ]


def test_init_files_are_neither_checked_nor_followed(tmp_path: Path) -> None:
    # The script reaches the package's __init__.py, which imports e.py for nothing
    # the script uses.
    files = {
        "cytepool/__init__.py": "from .a import f\nfrom .e import E\n__version__ = '1'\n",
        "cytepool/a.py": "def f():\n    return 1\n",
        "cytepool/e.py": "E = 2\n",
        "validation/check.py": "import cytepool\n\nprint(cytepool.f(), cytepool.__version__)\n",
    }

    assert _checked(tmp_path, files) == ["cytepool/a.py", "validation/check.py"]


def test_packages_imported_as_names_are_read_through_attributes(tmp_path: Path) -> None:
    package = {
        "cytepool/__init__.py": "from .a import f\n",
        "cytepool/a.py": "def f():\n    return 1\n",
        "cytepool/sub/__init__.py": "",
        "cytepool/sub/x.py": "def go():\n    return 2\n",
        "cytepool/sub/w.py": "def stay():\n    return 3\n",
    }
    scripts = {
        "import cytepool as cp\ncp.f()\n": ["cytepool/a.py"],
        # The dotted import binds the name cytepool, as Python does.
        "import cytepool.sub.x\ncytepool.sub.x.go()\n": ["cytepool/sub/x.py"],
        "import cytepool.sub.x\ncytepool.f()\n": ["cytepool/a.py", "cytepool/sub/x.py"],
        "from cytepool import sub\nsub.w.stay()\n": ["cytepool/sub/w.py"],
        "import cytepool.sub as s\nprint(s.x.go)\n": ["cytepool/sub/x.py"],
    }

    for script, expected in scripts.items():
        found = _checked(tmp_path, {**package, "validation/check.py": script})
        assert found == [*expected, "validation/check.py"], script


def test_a_module_imported_as_a_name_is_checked_whole(tmp_path: Path) -> None:
    # As validation/normalization.py calls functions of the module by their names.
    files = {
        "cytepool/m.py": "def tmm():\n    return 1\n",
        "validation/check.py": "import cytepool.m as m\n\ngetattr(m, 'tmm')()\n",
    }

    assert _checked(tmp_path, files) == ["cytepool/m.py", "validation/check.py"]


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (
            {"validation/check.py": 'CODE = """\nfrom cytepool import f\n"""\n'},
            "check.py, line 1: a string imports cytepool",
        ),
        (
            {"validation/check.py": 'run(f"""\n    import cytepool\n    {x}""")\n'},
            "a string imports cytepool",
        ),
        (
            {
                "cytepool/a.py": "from .b import *\n",
                "cytepool/b.py": "x = 1\n",
                "validation/check.py": "from cytepool.a import x\n",
            },
            "a.py, line 1: a star import from cytepool.b",
        ),
        (
            {"validation/check.py": "from cytepool.missing import f\n"},
            "check.py, line 1: no module cytepool.missing",
        ),
        (
            {"validation/check.py": "import cytepool.missing\n"},
            "check.py, line 1: no module cytepool.missing",
        ),
        (
            {"validation/check.py": "from cytepool import nothing\n"},
            "check.py, line 1: .*does not bind nothing, which cytepool.nothing names",
        ),
        (
            {"validation/check.py": "from . import helpers\n"},
            "check.py, line 1: a relative import in a script",
        ),
        (
            {
                "cytepool/a.py": "from ...x import y\n",
                "validation/check.py": "from cytepool.a import y\n",
            },
            "a.py, line 1: attempted relative import beyond top-level package",
        ),
        (
            {
                "cytepool/__init__.py": "from .a import f\n",
                "cytepool/a.py": "f = 1\n",
                "validation/check.py": "import cytepool as cp\n\ngetattr(cp, 'f')\n",
            },
            "check.py, line 3: the package cytepool is used as a value",
        ),
    ],
)
def test_code_the_helper_cannot_follow_is_refused(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _checked(tmp_path, files)


def test_a_docstring_may_show_imports_of_cytepool(tmp_path: Path) -> None:
    files = {
        "cytepool/a.py": "def f():\n    return 1\n",
        "validation/check.py": '"""Run it:\n\n    from cytepool.a import f\n"""\n\nfrom cytepool.a import f\n',
    }

    assert _checked(tmp_path, files) == ["cytepool/a.py", "validation/check.py"]


def test_the_digest_follows_only_the_checked_code(tmp_path: Path) -> None:
    files = {
        "cytepool/__init__.py": "from .b import f\n",
        "cytepool/b.py": '"""B."""\n\nfrom .c import g\n\ndef f():\n    return g()\n',
        "cytepool/c.py": "def g():\n    return 1\n",
        "cytepool/e.py": "def unused():\n    return 0\n",
        "validation/check.py": "from cytepool import f\n\nprint(f())\n",
    }
    _write(tmp_path, files)
    script = tmp_path / "validation" / "check.py"

    def digest() -> str:
        return helper.code_digest(helper.checked_files(script, tmp_path), tmp_path)

    first = digest()
    _write(tmp_path, {"cytepool/e.py": "def unused():\n    return 1\n"})
    _write(tmp_path, {"cytepool/b.py": files["cytepool/b.py"].replace("B.", "Bee.")})
    assert digest() == first
    _write(tmp_path, {"cytepool/c.py": "def g():\n    return 2\n"})
    assert digest() != first
    changed = digest()
    _write(
        tmp_path, {"validation/check.py": "from cytepool import f\n\nprint(f(), 1)\n"}
    )
    assert digest() != changed
    # The same code under another path: only the unchecked __init__.py names it.
    moved = digest()
    (tmp_path / "cytepool" / "b.py").rename(tmp_path / "cytepool" / "b2.py")
    _write(tmp_path, {"cytepool/__init__.py": "from .b2 import f\n"})
    assert digest() != moved


def test_the_scripts_check_the_package_modules_they_import() -> None:
    def files(script: str) -> set[str]:
        found = helper.checked_files(ROOT / "validation" / script, ROOT)
        return {path.relative_to(ROOT).as_posix() for path in found}

    for script in sorted((ROOT / "validation").glob("*.py")):
        checked = files(script.name)
        assert f"validation/{script.name}" in checked
        package = checked - {f"validation/{script.name}"}
        assert package and all(name.startswith("cytepool/") for name in package)
        assert not any(
            name.endswith(("__init__.py", "_version.py")) for name in package
        )
    # Read from the scripts' imports by hand.
    assert "cytepool/cube/pool.py" in files("pool_cells.py")
    assert "cytepool/plotting.py" not in files("pool_cells.py")
    assert "cytepool/stats/multiple_testing.py" in files("multiple_testing.py")
    assert "cytepool/cube/cube.py" not in files("multiple_testing.py")
    assert "cytepool/plotting.py" in files("plotting.py")
    # normalization.py calls the module's functions through getattr.
    assert "cytepool/normalization.py" in files("normalization.py")
    assert "cytepool/stats/smoothing.py" in files("smoothing.py")


# The records.


def test_recorded_digests_read_the_file_and_refuse_malformed_digests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runs.toml"
    path.write_text(
        '# Records.\n"voom.py" = "0f3a9c21b7d4"\n"dream.py" = "6e02d1c9a8b5"\n'
    )

    assert helper.recorded_digests(path) == {
        "voom.py": "0f3a9c21b7d4",
        "dream.py": "6e02d1c9a8b5",
    }
    for line in [
        '"voom.py" = "0f3a9c21b7d"',  # 11 digits
        '"voom.py" = "0F3A9C21B7D4"',  # upper case
        '"voom.py" = "0f3a9c21b7dz"',  # not hexadecimal
        '"voom.py" = 123456789012',  # not a string
        'voom.py = "0f3a9c21b7d4"',  # a dotted key, read as a table
    ]:
        path.write_text(line + "\n")
        with pytest.raises(ValueError, match="12 lowercase hexadecimal digits"):
            helper.recorded_digests(path)


def test_changed_since_run_lists_changed_and_unrecorded_scripts() -> None:
    current = {
        "s1.py": "111111111111",
        "s2.py": "222222222222",
        "s3.py": "333333333333",
    }
    recorded = {"s1.py": "111111111111", "s2.py": "000000000000"}

    assert helper.changed_since_run(recorded, current) == [
        helper.Change("s2.py", "000000000000", "222222222222"),
        helper.Change("s3.py", None, "333333333333"),
    ]
    assert helper.changed_since_run(current, current) == []
    with pytest.raises(ValueError, match="scripts that do not exist: gone.py"):
        helper.changed_since_run({**recorded, "gone.py": "444444444444"}, current)


def test_main_prints_each_change_and_sets_the_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path,
        {
            "cytepool/__init__.py": "",
            "cytepool/a.py": "def f():\n    return 1\n",
            "validation/s1.py": "from cytepool.a import f\n",
            "validation/s2.py": "from cytepool.a import f\n\nprint(f())\n",
            "validation/s3.py": "print(3)\n",
        },
    )
    digests = helper.script_digests(tmp_path)
    records = tmp_path / "runs.toml"
    records.write_text(f'"s1.py" = "{digests["s1.py"]}"\n"s2.py" = "000000000000"\n')

    assert helper.main(tmp_path, records) == 1
    assert capsys.readouterr().out.splitlines() == [
        "validation/s2.py: the code it checks has changed since its recorded run "
        f"(recorded 000000000000, now {digests['s2.py']})",
        f"validation/s3.py: no recorded run (now {digests['s3.py']})",
    ]
    records.write_text("".join(f'"{s}" = "{d}"\n' for s, d in digests.items()))
    assert helper.main(tmp_path, records) == 0
    assert capsys.readouterr().out == (
        "No script in validation/ checks code that has changed since its recorded run.\n"
    )
