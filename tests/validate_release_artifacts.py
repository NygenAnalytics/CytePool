"""Check the wheel and the source distribution of a release before they are published.

The release workflow, ``.github/workflows/publish.yml``, runs this on what it built::

    uv run --no-project --with packaging==26.3 \\
        python tests/validate_release_artifacts.py --release-tag 0.1.0 dist/*

A release has one wheel, for every platform (``py3-none-any``), and one source
distribution. Both must name the distribution ``cytepool`` and the version of the
release tag, which has no local part, and declare what ``pyproject.toml`` declares:
the Python versions, the requirements and the extras (only numpy and numba are
required without an extra), the licence and its files, and ``README.md`` as the
description. Both must hold every module of the package as it is in the source tree,
and ``cytepool/_version.py`` naming the version. Besides those, the wheel holds its
metadata and the licence files, and the source distribution the files that build the
package and those that setuptools writes; any other file, such as one of ``tests/`` or
``docs/``, is refused. Every problem found is printed, and the exit status is then 1.
"""

import argparse
import ast
import sys
import tarfile
import tomllib
from email.parser import Parser
from email.policy import compat32
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "cytepool"
VERSION_FILE = f"{PACKAGE}/_version.py"
# The package depends only on numpy and numba.
REQUIRED = ["numba", "numpy"]
PURE_PYTHON = Tag("py3", "none", "any")
# The files of the source tree in a source distribution, besides the package's modules
# and the licence files.
BUILD_FILES = ("MANIFEST.in", "README.md", "pyproject.toml")
# What setuptools writes into a source distribution; a name ending in "/" is a folder.
SDIST_GENERATED = ("PKG-INFO", "setup.cfg", f"{PACKAGE}.egg-info/")


class ReleaseArtifactError(ValueError):
    """The files of a release cannot be published as they are."""


def release_version(release_tag: str) -> Version:
    """Return the version that a release tag names.

    Raises:
        ReleaseArtifactError: If the tag is not a version, or has a local part, which
            PyPI refuses.
    """
    try:
        version = Version(release_tag)
    except InvalidVersion:
        raise ReleaseArtifactError(
            f"the release tag {release_tag!r} is not a version, such as 0.1.0 or "
            "0.2.0rc1"
        ) from None
    if version.local is not None:
        raise ReleaseArtifactError(
            f"the release tag {release_tag!r} has a local version part "
            f"(+{version.local}), which PyPI refuses"
        )
    return version


def validate_release_artifacts(
    artifacts: list[Path], release_tag: str, root: Path = ROOT
) -> list[str]:
    """Check the files of a release against its tag and the source tree they come from.

    Args:
        artifacts: The files to publish: one wheel and one source distribution.
        release_tag: The tag of the release, which names its version.
        root: The source tree that the files were built from.

    Returns:
        A line for each file, saying that it can be published.

    Raises:
        ReleaseArtifactError: Naming every problem found, one per line.
    """
    version = release_version(release_tag)
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    wheels = [path for path in artifacts if path.name.endswith(".whl")]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    problems = [
        f"{path.name} is neither a wheel nor a source distribution"
        for path in artifacts
        if path not in wheels and path not in sdists
    ]
    for kind, found in (("wheel", wheels), ("source distribution", sdists)):
        if len(found) != 1:
            problems.append(f"a release has one {kind}, found {len(found)}")
    for path in wheels:
        for problem in _wheel_problems(path, version, project, root):
            problems.append(f"{path.name}: {problem}")
    for path in sdists:
        for problem in _sdist_problems(path, version, project, root):
            problems.append(f"{path.name}: {problem}")
    if problems:
        raise ReleaseArtifactError("\n".join(problems))
    return [f"{path.name}: {PACKAGE} {version}, ready to publish" for path in artifacts]


def _wheel_problems(
    path: Path, version: Version, project: dict[str, Any], root: Path
) -> list[str]:
    try:
        name, file_version, _, tags = parse_wheel_filename(path.name)
    except ValueError as error:
        return [str(error)]
    problems = _file_name_problems(name, file_version, version)
    if tags != {PURE_PYTHON}:
        built_for = ", ".join(sorted(map(str, tags)))
        problems.append(f"is built for {built_for}, expected {PURE_PYTHON}")
    with ZipFile(path) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    folders = {name.split("/")[0] for name in files}
    dist_infos = sorted(folder for folder in folders if folder.endswith(".dist-info"))
    if len(dist_infos) != 1:
        return problems + [f"has {len(dist_infos)} .dist-info folders, expected one"]
    dist_info = dist_infos[0]
    sources = _modules(root) | {
        f"{dist_info}/licenses/{name}": source
        for name, source in _licence_files(project, root)
    }
    problems += _content_problems(files, sources, (f"{dist_info}/",), version)
    wheel = Parser(policy=compat32).parsestr(
        files.get(f"{dist_info}/WHEEL", b"").decode("utf-8")
    )
    if wheel.get("Root-Is-Purelib") != "true" or wheel.get_all("Tag") != [
        str(PURE_PYTHON)
    ]:
        problems.append(
            f"{dist_info}/WHEEL does not describe a pure Python wheel "
            f"(Root-Is-Purelib: true, Tag: {PURE_PYTHON})"
        )
    metadata = files.get(f"{dist_info}/METADATA")
    if metadata is None:
        return problems + [f"{dist_info}/METADATA is missing"]
    return problems + _metadata_problems(metadata, version, project, root)


def _sdist_problems(
    path: Path, version: Version, project: dict[str, Any], root: Path
) -> list[str]:
    try:
        name, file_version = parse_sdist_filename(path.name)
    except ValueError as error:
        return [str(error)]
    problems = _file_name_problems(name, file_version, version)
    folder = path.name.removesuffix(".tar.gz") + "/"
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            data = archive.extractfile(member) if member.isfile() else None
            if data is None or not member.name.startswith(folder):
                problems.append(f"{member.name} is not a file in {folder}")
                continue
            files[member.name.removeprefix(folder)] = data.read()
    sources = _modules(root) | dict(_licence_files(project, root))
    sources |= {name: root / name for name in BUILD_FILES}
    problems += _content_problems(files, sources, SDIST_GENERATED, version)
    metadata = files.get("PKG-INFO")
    if metadata is None:
        return problems + ["PKG-INFO is missing"]
    return problems + _metadata_problems(metadata, version, project, root)


def _file_name_problems(
    name: str, file_version: Version, version: Version
) -> list[str]:
    problems = []
    if name != PACKAGE:
        problems.append(f"the file name names {name}, expected {PACKAGE}")
    if file_version != version:
        problems.append(
            f"the file name names version {file_version}, expected {version}"
        )
    return problems


def _modules(root: Path) -> dict[str, Path]:
    """The package's modules in the source tree, by their path in an artifact."""
    package = root / PACKAGE
    return {
        f"{PACKAGE}/{path.relative_to(package).as_posix()}": path
        for path in sorted(package.rglob("*.py"))
        if path.relative_to(package).as_posix() != "_version.py"
    }


def _licence_files(project: dict[str, Any], root: Path) -> list[tuple[str, Path]]:
    """The licence files of pyproject.toml, with their path in the source tree."""
    return sorted(
        (path.relative_to(root).as_posix(), path)
        for pattern in project["license-files"]
        for path in root.glob(pattern)
    )


def _content_problems(
    files: dict[str, bytes],
    sources: dict[str, Path],
    generated: tuple[str, ...],
    version: Version,
) -> list[str]:
    """Problems with the files of an artifact.

    Each file must be a copy of its source in the source tree, the version file, or a
    file that the build writes (``generated``); every source and the version file must
    be there.
    """
    problems = []
    for name, data in sorted(files.items()):
        if name in sources:
            if data != sources[name].read_bytes():
                problems.append(f"{name} differs from the source tree")
        elif name == VERSION_FILE:
            found = _assigned_version(data)
            try:
                matches = found is not None and Version(found) == version
            except InvalidVersion:
                matches = False
            if not matches:
                named = "no version" if found is None else f"version {found}"
                problems.append(f"{name} names {named}, expected {version}")
        elif not any(
            name == entry or (entry.endswith("/") and name.startswith(entry))
            for entry in generated
        ):
            problems.append(f"{name} does not belong in a release")
    missing = (sources.keys() | {VERSION_FILE}) - files.keys()
    return problems + [f"{name} is missing" for name in sorted(missing)]


def _assigned_version(source: bytes) -> str | None:
    """The version that a version file assigns to ``__version__``."""
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        ):
            return str(node.value.value)
    return None


def _metadata_problems(
    data: bytes, version: Version, project: dict[str, Any], root: Path
) -> list[str]:
    """Problems with an artifact's core metadata, against pyproject.toml."""
    metadata = Parser(policy=compat32).parsestr(data.decode("utf-8"))
    problems = []

    def check(field: str, found: object, expected: object) -> None:
        if found != expected:
            problems.append(f"{field} is {found}, expected {expected}")

    check("Name", canonicalize_name(metadata.get("Name", "")), PACKAGE)
    try:
        check("Version", Version(metadata.get("Version", "")), version)
    except InvalidVersion:
        problems.append(f"Version is {metadata.get('Version')}, which is not a version")
    check(
        "Requires-Python",
        SpecifierSet(metadata.get("Requires-Python", "")),
        SpecifierSet(project["requires-python"]),
    )
    check("License-Expression", metadata.get("License-Expression"), project["license"])
    check(
        "License-File",
        sorted(metadata.get_all("License-File", [])),
        [name for name, _ in _licence_files(project, root)],
    )
    readme = project["readme"]
    check(
        "Description-Content-Type",
        metadata.get("Description-Content-Type"),
        readme["content-type"],
    )
    if metadata.get_payload() != (root / readme["file"]).read_text(encoding="utf-8"):
        problems.append(f"the description is not {readme['file']}")
    requirements = [Requirement(line) for line in metadata.get_all("Requires-Dist", [])]
    found = {str(requirement) for requirement in requirements}
    declared = _declared_requirements(project)
    for requirement in sorted(found - declared):
        problems.append(f"requires {requirement}, which pyproject.toml does not")
    for requirement in sorted(declared - found):
        problems.append(f"does not require {requirement}, as pyproject.toml does")
    required = sorted(
        {canonicalize_name(r.name) for r in requirements if r.marker is None}
    )
    if required != REQUIRED:
        problems.append(
            f"requires {', '.join(required)} without an extra; only "
            f"{' and '.join(REQUIRED)} may be"
        )
    check(
        "Provides-Extra",
        sorted(
            canonicalize_name(extra) for extra in metadata.get_all("Provides-Extra", [])
        ),
        sorted(canonicalize_name(extra) for extra in project["optional-dependencies"]),
    )
    return problems


def _declared_requirements(project: dict[str, Any]) -> set[str]:
    """The requirements of pyproject.toml, written as core metadata writes them."""
    declared = {str(Requirement(text)) for text in project["dependencies"]}
    for extra, texts in project["optional-dependencies"].items():
        for text in texts:
            requirement = Requirement(text)
            condition = f'extra == "{canonicalize_name(extra)}"'
            marker = requirement.marker
            requirement.marker = Marker(
                f"({marker}) and {condition}" if marker else condition
            )
            declared.add(str(requirement))
    return declared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the wheel and the source distribution of a release."
    )
    parser.add_argument(
        "--release-tag", required=True, help="the tag of the release, such as 0.1.0"
    )
    parser.add_argument(
        "artifacts", nargs="+", type=Path, help="the wheel and the source distribution"
    )
    args = parser.parse_args()
    try:
        for line in validate_release_artifacts(args.artifacts, args.release_tag):
            print(line)
    except ReleaseArtifactError as error:
        sys.exit(f"The release cannot be published:\n{error}")


if __name__ == "__main__":
    main()
