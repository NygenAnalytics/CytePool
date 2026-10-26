"""Which public names of CytePool the documentation uses or explains, and which not.

(Named so that it never takes the place of the ``coverage`` package, which numba
imports when it is installed.)

Every public name must earn a place in a page, or be listed with the reason it needs
none. The public names are those the API reference must document (a module's
``__all__``, or else the public functions and classes it defines, as
``tests/test_documentation.py`` defines them) and the public methods of their
classes. A name is covered when a guide or tutorial uses it (its code imports it, or
its prose links to it) or an explanation links to it (``page_index.py``); index
pages, such as the capability matrix and the glossary, do not count.

``coverage_exemptions.toml`` lists the names that need no page, each with its reason
(``[exempt]``), and the names a page that is planned will cover, each with that page
(``[planned]``). The tests fail on a name that is neither covered nor listed, on a
listed name that is not public or is covered already, and on an empty reason.

Usage, from the repository root::

    uv run --group docs python docs/coverage_report.py
"""

import importlib
import inspect
import pkgutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import page_index

import cytepool

EXEMPTIONS = Path(__file__).resolve().parent / "coverage_exemptions.toml"


def _modules() -> list[tuple[ModuleType, ModuleType]]:
    """Every module whose name has no private component, with its package."""
    found = []
    for info in pkgutil.walk_packages(cytepool.__path__, "cytepool."):
        if not any(part.startswith("_") for part in info.name.split(".")):
            module = importlib.import_module(info.name)
            found.append(
                (module, importlib.import_module(info.name.rpartition(".")[0]))
            )
    return found


def public_names() -> set[str]:
    """Every public name: what the API reference must document, and the public
    methods of the public classes."""
    modules = _modules()

    def implements(module: ModuleType, package: ModuleType) -> bool:
        exported = set(getattr(package, "__all__", ()))
        return package is not cytepool and bool(
            set(page_index.public_names(module)) & exported
        )

    public_modules = {
        module.__name__
        for module, package in modules
        if page_index.public_names(module) and not implements(module, package)
    }
    names = set()
    for module, package in modules:
        exported = set(getattr(package, "__all__", ()))
        for name in page_index.public_names(module):
            if module.__name__ in public_modules or (
                implements(module, package) and name not in exported
            ):
                names.add(f"{module.__name__}.{name}")
    for name in list(names):
        value = getattr(
            importlib.import_module(name.rpartition(".")[0]), name.rpartition(".")[2]
        )
        if inspect.isclass(value):
            for attribute, member in vars(value).items():
                method = inspect.isfunction(member) or isinstance(
                    member, classmethod | staticmethod
                )
                if method and not attribute.startswith("_"):
                    names.add(f"{name}.{attribute}")
    return names


@dataclass(frozen=True)
class Report:
    """Where each public name is used or explained, and what is not covered."""

    pages: dict[str, list[str]]  # name -> the files of the pages that cover it
    exempt: dict[str, str]  # name -> reason
    planned: dict[str, str]  # name -> the page that will cover it
    uncovered: list[str]  # neither covered nor listed
    stale: list[str]  # listed, but not public or covered already
    empty: list[str]  # listed without a reason


def read_exemptions(path: Path = EXEMPTIONS) -> tuple[dict[str, str], dict[str, str]]:
    """The exempt names with their reasons, and the planned names with their pages."""
    data = tomllib.loads(path.read_text())
    unknown = set(data) - {"exempt", "planned"}
    if unknown:
        raise ValueError(
            f"{path} has tables {sorted(unknown)}; expected exempt and planned"
        )
    return dict(data.get("exempt", {})), dict(data.get("planned", {}))


def report(path: Path = EXEMPTIONS) -> Report:
    links = page_index.links()
    covered: dict[str, list[str]] = {}
    for found in (links.used_in, links.explained_in):
        for name, pages in found.items():
            covered.setdefault(name, []).extend(page.file for page in pages)
    exempt, planned = read_exemptions(path)
    names = public_names()
    listed = {**exempt, **planned}
    return Report(
        pages={
            name: sorted(set(covered[name]))
            for name in sorted(names)
            if name in covered
        },
        exempt=exempt,
        planned=planned,
        uncovered=sorted(names - set(covered) - set(listed)),
        stale=sorted(name for name in listed if name not in names or name in covered),
        empty=sorted(
            name for name, reason in listed.items() if not str(reason).strip()
        ),
    )


def main() -> None:
    result = report()
    names = public_names()
    print(
        f"{len(names)} public names: {len(result.pages)} used or explained in a page, "
        f"{len(result.exempt)} exempt, {len(result.planned)} planned.\n"
    )
    for name, pages in result.pages.items():
        print(f"{name}: {', '.join(pages)}")
    if result.planned:
        print("\nOpen items, which a planned page will cover:")
        for name, page in sorted(result.planned.items()):
            print(f"  {name}: {page}")
    print("\nExempt:")
    for name, reason in sorted(result.exempt.items()):
        print(f"  {name}: {reason}")
    problems = [
        *(f"not covered and not listed: {name}" for name in result.uncovered),
        *(
            f"listed, but not public or covered already: {name}"
            for name in result.stale
        ),
        *(f"listed without a reason: {name}" for name in result.empty),
    ]
    if problems:
        print("\n" + "\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
