"""Tests for the checks that the release workflow runs before it publishes.

The wheel and the source distribution are built once from this source tree, into a
temporary folder, by the command and the pinned build tools of the release workflow,
with the version that setuptools-scm gives the current commit (without a local part,
as a release tag gives none). ``validate_release_artifacts`` must accept them, and
refuse copies of them with each defect it looks for, naming that defect and nothing
else. The analysis of ``smoke_test_wheel`` runs here on this source tree, so that a
change that breaks it fails before a release does; the release workflow runs it with
the installed wheel.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.utils import parse_wheel_filename
from smoke_test_wheel import analyse
from validate_release_artifacts import (
    ROOT,
    ReleaseArtifactError,
    validate_release_artifacts,
)

Change = Callable[[dict[str, bytes]], None]


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """The wheel and the source distribution of this tree, and their version."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("building the release files needs uv, which is not on PATH")
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("building the release files needs git and a git checkout")
    out = tmp_path_factory.mktemp("dist")
    command = [
        uv,
        "build",
        "--quiet",
        "--python",
        sys.executable,
        "--out-dir",
        str(out),
    ]
    command += ["--build-constraints", str(ROOT / ".github" / "build-constraints.txt")]
    # As the release workflow builds, but from a checkout that may have changes.
    env = os.environ | {
        "SETUPTOOLS_SCM_OVERRIDES_FOR_CYTEPOOL": '{local_scheme = "no-local-version"}'
    }
    build = subprocess.run(
        [*command, str(ROOT)], env=env, capture_output=True, text=True
    )
    assert build.returncode == 0, build.stderr
    (wheel,) = out.glob("*.whl")
    (sdist,) = out.glob("*.tar.gz")
    return wheel, sdist, str(parse_wheel_filename(wheel.name)[1])


def _files(artifact: Path) -> dict[str, bytes]:
    """The files of a wheel or source distribution, by their path in the release."""
    if artifact.name.endswith(".whl"):
        with ZipFile(artifact) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    folder = artifact.name.removesuffix(".tar.gz") + "/"
    files = {}
    with tarfile.open(artifact) as archive:
        for member in archive.getmembers():
            if (data := archive.extractfile(member)) is not None:
                files[member.name.removeprefix(folder)] = data.read()
    return files


def _write(artifact: Path, files: dict[str, bytes]) -> Path:
    if artifact.name.endswith(".whl"):
        with ZipFile(artifact, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return artifact
    folder = artifact.name.removesuffix(".tar.gz") + "/"
    with tarfile.open(artifact, "w:gz", compresslevel=1) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(folder + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return artifact


def _problems(artifacts: list[Path], release_tag: str) -> list[str]:
    with pytest.raises(ReleaseArtifactError) as raised:
        validate_release_artifacts(artifacts, release_tag)
    return str(raised.value).splitlines()


def _add(name: str) -> Change:
    return lambda files: files.update({name: b"VALUE = 1\n"})


def _remove(name: str) -> Change:
    def change(files: dict[str, bytes]) -> None:
        del files[name]

    return change


def _stale_version_file(files: dict[str, bytes]) -> None:
    text = files["cytepool/_version.py"].decode()
    stale = re.sub(
        r"(?m)^__version__ = version = .*$", "__version__ = version = '0.0.1'", text
    )
    assert stale != text
    files["cytepool/_version.py"] = stale.encode()


def _replace(name: str, old: str, new: str) -> Change:
    """Replace text in the file ``name``, or in the one file in a folder of that name."""

    def change(files: dict[str, bytes]) -> None:
        if name in files:
            path = name
        else:
            (path,) = [path for path in files if path.endswith(f"/{name}")]
        text = files[path].decode()
        assert old in text
        files[path] = text.replace(old, new, 1).encode()

    return change


def _metadata(old: str, new: str) -> Change:
    """Replace text in the core metadata of a wheel or a source distribution."""

    def change(files: dict[str, bytes]) -> None:
        _replace("PKG-INFO" if "PKG-INFO" in files else "METADATA", old, new)(files)

    return change


def test_the_release_files_of_this_tree_can_be_published(
    release: tuple[Path, Path, str],
) -> None:
    wheel, sdist, version = release

    assert validate_release_artifacts([wheel, sdist], version) == [
        f"{wheel.name}: cytepool {version}, ready to publish",
        f"{sdist.name}: cytepool {version}, ready to publish",
    ]
    # What a release holds and leaves out, read from the files themselves.
    wheel_files, sdist_files = _files(wheel), _files(sdist)
    assert {name.split("/")[0] for name in wheel_files} == {
        "cytepool",
        f"cytepool-{version}.dist-info",
    }
    assert {name.split("/")[0] for name in sdist_files} == {
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "cytepool",
        "cytepool.egg-info",
        "pyproject.toml",
        "setup.cfg",
    }
    for files in (wheel_files, sdist_files):
        assert {"cytepool/_version.py", "cytepool/stats/voom.py"} <= files.keys()


@pytest.mark.parametrize(
    ("artifact", "change", "expected"),
    [
        pytest.param(
            "wheel",
            _add("tests/test_cube.py"),
            ["tests/test_cube.py does not belong in a release"],
            id="tests-in-the-wheel",
        ),
        pytest.param(
            "sdist",
            _add("docs/source/index.md"),
            ["docs/source/index.md does not belong in a release"],
            id="docs-in-the-sdist",
        ),
        pytest.param(
            "wheel",
            _remove("cytepool/stats/voom.py"),
            ["cytepool/stats/voom.py is missing"],
            id="a-module-missing",
        ),
        pytest.param(
            "wheel",
            _remove("cytepool/_version.py"),
            ["cytepool/_version.py is missing"],
            id="no-version-file",
        ),
        pytest.param(
            "sdist",
            _remove("LICENSE"),
            ["LICENSE is missing"],
            id="no-licence",
        ),
        pytest.param(
            "sdist",
            _stale_version_file,
            ["cytepool/_version.py names version 0.0.1, expected {version}"],
            id="a-stale-version-file",
        ),
        pytest.param(
            "wheel",
            _replace("cytepool/table.py", '"""', '"""Edited. '),
            ["cytepool/table.py differs from the source tree"],
            id="an-edited-module",
        ),
        pytest.param(
            "wheel",
            _replace("WHEEL", "Root-Is-Purelib: true", "Root-Is-Purelib: false"),
            [
                "cytepool-{version}.dist-info/WHEEL does not describe a pure Python "
                "wheel (Root-Is-Purelib: true, Tag: py3-none-any)"
            ],
            id="not-pure-python",
        ),
        pytest.param(
            "wheel",
            _metadata(
                "Requires-Dist: numpy>=2.1\n",
                "Requires-Dist: numpy>=2.1\nRequires-Dist: pandas>=2.2\n",
            ),
            [
                "requires pandas>=2.2, which pyproject.toml does not",
                "requires numba, numpy, pandas without an extra; only numba and "
                "numpy may be",
            ],
            id="another-requirement",
        ),
        pytest.param(
            "sdist",
            _metadata(
                'Provides-Extra: plot\nRequires-Dist: matplotlib>=3.9; extra == "plot"\n',
                "",
            ),
            [
                'does not require matplotlib>=3.9; extra == "plot", as pyproject.toml '
                "does",
                "Provides-Extra is ['anndata', 'mudata', 'test'], expected "
                "['anndata', 'mudata', 'plot', 'test']",
            ],
            id="an-extra-missing",
        ),
        pytest.param(
            "wheel",
            _metadata(
                'mudata!=0.4.0,>=0.3.1; python_version < "3.15" and extra == "test"',
                'mudata!=0.4.0,>=0.3.1; extra == "test"',
            ),
            [
                'requires mudata!=0.4.0,>=0.3.1; extra == "test", which '
                "pyproject.toml does not",
                'does not require mudata!=0.4.0,>=0.3.1; python_version < "3.15" and '
                'extra == "test", as pyproject.toml does',
            ],
            id="a-marker-missing",
        ),
        pytest.param(
            "wheel",
            _metadata("License-Expression: BSD-3-Clause", "License-Expression: MIT"),
            ["License-Expression is MIT, expected BSD-3-Clause"],
            id="another-licence",
        ),
        pytest.param(
            "sdist",
            _metadata("Requires-Python: >=3.13", "Requires-Python: >=3.12"),
            ["Requires-Python is >=3.12, expected >=3.13"],
            id="other-python-versions",
        ),
        pytest.param(
            "wheel",
            _metadata(
                "Test which genes, cell types and pathways change",
                "Test which genes, cell types and pathways differ",
            ),
            ["the description is not README.md"],
            id="another-description",
        ),
    ],
)
def test_a_defect_of_a_release_file_is_named(
    release: tuple[Path, Path, str],
    tmp_path: Path,
    artifact: str,
    change: Change,
    expected: list[str],
) -> None:
    wheel, sdist, version = release
    original = wheel if artifact == "wheel" else sdist
    files = _files(original)
    change(files)
    defective = _write(tmp_path / original.name, files)
    artifacts = [defective, sdist] if artifact == "wheel" else [wheel, defective]

    assert _problems(artifacts, version) == [
        f"{original.name}: {problem.format(version=version)}" for problem in expected
    ]


def test_the_version_must_be_that_of_the_release_tag(
    release: tuple[Path, Path, str],
) -> None:
    wheel, sdist, version = release

    assert _problems([wheel, sdist], "v99.0") == [
        f"{artifact.name}: {problem}"
        for artifact in (wheel, sdist)
        for problem in (
            f"the file name names version {version}, expected 99.0",
            f"cytepool/_version.py names version {version}, expected 99.0",
            f"Version is {version}, expected 99.0",
        )
    ]


def test_a_release_is_one_pure_wheel_and_one_source_distribution(
    release: tuple[Path, Path, str], tmp_path: Path
) -> None:
    wheel, sdist, version = release
    notes = tmp_path / "notes.txt"
    notes.write_text("")
    for_one_platform = tmp_path / wheel.name.replace(
        "py3-none-any", "cp313-cp313-win_amd64"
    )
    shutil.copy(wheel, for_one_platform)

    assert _problems([wheel, wheel, notes], version) == [
        "notes.txt is neither a wheel nor a source distribution",
        "a release has one wheel, found 2",
        "a release has one source distribution, found 0",
    ]
    assert _problems([for_one_platform, sdist], version) == [
        f"{for_one_platform.name}: is built for cp313-cp313-win_amd64, expected "
        "py3-none-any"
    ]


@pytest.mark.parametrize(
    ("release_tag", "message"),
    [
        ("v-next", "the release tag 'v-next' is not a version, such as 0.1.0"),
        (
            "0.1.0+nygen",
            "the release tag '0.1.0+nygen' has a local version part (+nygen), which "
            "PyPI refuses",
        ),
    ],
)
def test_a_release_tag_must_be_a_version_without_a_local_part(
    release_tag: str, message: str
) -> None:
    with pytest.raises(ReleaseArtifactError, match=re.escape(message)):
        validate_release_artifacts([], release_tag)


def test_the_analysis_of_the_wheel_smoke_test_gives_limmas_tables() -> None:
    analyse()
