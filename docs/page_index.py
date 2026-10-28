"""An index of the documentation's pages and of the API entries they use or explain.

The pages of the site (``docs/source``, listed in ``myst.yml``) and the Sphinx API
reference (``docs/source/reference/api``) link to each other through this index:

- Pages link to an API entry with the ``{api}`` role, writing a public name as it is
  documented or by a unique part of it: ``{api}`limma_voom``` or
  ``{api}`Cube.select``` (an object whose documented name ends with the part, such
  as ``cytepool.cube.Cube.select``), or ``{api}`text <name>``` for other link text.
  A module needs its full name. ``build_docs.py`` rewrites every role into a link
  to the API page and anchor that document the name, and stops on a name that is
  unknown or ambiguous.
- The API reference lists under each entry the pages that use it and the pages that
  explain it (``docs/source/reference/api/conf.py``). A guide or tutorial uses a
  name when its code imports it from CytePool, or when its prose links to it with
  the role or a link to the API reference; an explanation (the chapters of
  "Understand the methods" and the reference of cube files) explains the names its
  prose links to. Index pages, which list or point to many names (the overview,
  installation, and the reference pages other than cube files), give no such links.
- ``coverage_report.py`` lists the public names that no guide, tutorial or
  explanation uses or explains.

Citations are written ``[@key]`` or ``@key`` and resolve in ``references.bib``.

The documented names are read from the API sources, as Sphinx documents them:
``automodule`` documents a module and its public names (its ``__all__``, or else the
public functions and classes it defines) that have docstrings, and the methods and
properties of its classes that have docstrings; with ``:no-members:`` it documents none
of its names, and with ``:no-index:`` the module itself is no entry, as for the
description of a module whose names are documented at the top level;
``autofunction`` and ``autoclass`` document one object; ``py:data`` documents a
constant under the module the page documents last, or its ``:module:`` option.
"""

import ast
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
API_SOURCE = SOURCE / "reference" / "api"

# The top-level groups of pages in the navigation, in order (myst.yml). The API
# reference follows them, as a link whose pages the site build adds.
SECTIONS = (
    "Start here",
    "Build and share cubes",
    "Compare conditions",
    "Combine studies",
    "Understand the methods",
    "Reference",
)
API_REFERENCE = "API reference"
# Pages whose prose explains the names it links to: their links become "Explained
# in" on the API entries.
EXPLANATION_SECTIONS = frozenset({"Understand the methods"})
EXPLANATION_PAGES = frozenset({"reference/cube_files.md"})
# Pages that list or point to many names without using or explaining them: their
# links give no backlinks and do not count as coverage.
INDEX_PAGES = frozenset(
    {
        "index.md",
        "installation.md",
        "validation.md",
        "reference/datasets.md",
        "reference/related_software.md",
        "reference/bibliography.md",
        "reference/glossary.md",
        "reference/citing.md",
    }
)
GUIDE, EXPLANATION, INDEX = "guide", "explanation", "index"

# A role, written {api}`name` or {api}`text <name>`, or inline code, inside which a
# role is text: Markdown is read in that order, so that ``{api}`name` `` shows the
# role as code and is not rewritten.
ROLE = re.compile(r"\{api\}`(?P<role>[^`]+)`")
ROLE_OR_CODE = re.compile(
    r"\{api\}`(?P<role>[^`]+)`|(?P<ticks>`+)(?:(?!(?P=ticks)).)+?(?P=ticks)", re.S
)
ROLE_WITH_TEXT = re.compile(r"^(?P<text>.*\S)\s*<(?P<name>[^<>]+)>$", re.S)
API_LINK = re.compile(r"\]\(/api/([\w.-]+\.html)#([\w.-]+)\)")
CITATION = re.compile(r"(?<![\w.@/\\])-?@([A-Za-z][\w:.#$%&+?<>~/-]*[\w])")
CITE_ROLE = re.compile(r"\{cite(?::\w+)?\}`([^`]+)`")
FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n?$")
# Fences of code; the content of other fences, such as notes and figures, is prose.
PYTHON_FENCE = re.compile(
    r"^\s*(\{code(?:-block|-cell)?\}\s*)?(python|py|ipython3?)\s*$", re.I
)
CODE_FENCE = re.compile(
    r"^\s*(\{code(?:-block|-cell)?\}\s*)?(bash|sh|shell|console|text|json|yaml|toml|r|"
    r"python|py|ipython3?)\s*$",
    re.I,
)
PROSE, PYTHON, CODE, FENCE_LINE = "prose", "python", "code", "fence"


@dataclass(frozen=True)
class ApiEntry:
    """A name the API reference documents, with the page and anchor of its entry."""

    name: str
    page: str  # the HTML page of the API reference, such as "voom.html"
    kind: str  # "module", "function", "class", "method", "property" or "data"

    @property
    def url(self) -> str:
        anchor = f"module-{self.name}" if self.kind == "module" else self.name
        return f"/api/{self.page}#{anchor}"


@dataclass(frozen=True)
class Page:
    """A page of the site: where it is, what kind it is, and what it uses and cites."""

    file: str  # relative to docs/source, as myst.yml lists it
    title: str
    section: str
    url: str  # its path on the site, such as "/statistics/empirical-bayes"
    kind: str  # GUIDE, EXPLANATION or INDEX
    prose: str
    code: str
    roles: tuple[str, ...] = field(default=())  # names as the {api} roles write them
    api_links: tuple[str, ...] = field(default=())  # anchors of links to /api/
    citations: tuple[str, ...] = field(default=())


def docs_module(name: str) -> ModuleType:
    """A script of ``docs/``, such as ``datasets``, imported by its path under a name
    of its own, so that it never takes the place of an installed package of the same
    name, such as Hugging Face's ``datasets``."""
    key = f"_cytepool_docs_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, DOCS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


# Reading the API reference.


def public_names(module: Any) -> list[str]:
    """``__all__``, or else the public functions and classes a module defines."""
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and value.__module__ == module.__name__
    ]


def _members(name: str, value: Any) -> dict[str, str]:
    """What autodoc documents of a function or class, by name, with its kind: the
    object, if it has a docstring, and the methods and properties of a class that
    have one, other than the field aliases of named tuples, which conf.py leaves out.
    """
    if not (inspect.isfunction(value) or inspect.isclass(value)):
        return {}
    found = {}
    if inspect.getdoc(value):
        found[name] = "class" if inspect.isclass(value) else "function"
    if inspect.isclass(value):
        for attribute, member in vars(value).items():
            routine = inspect.isfunction(member) or isinstance(
                member, property | classmethod | staticmethod
            )
            doc = inspect.getdoc(member) if routine else None
            alias = (doc or "").startswith("Alias for field number")
            if not attribute.startswith("_") and doc and not alias:
                kind = "property" if isinstance(member, property) else "method"
                found[f"{name}.{attribute}"] = kind
    return found


_DIRECTIVE = re.compile(r"^\.\. (automodule|autofunction|autoclass|py:data)::\s+(\S+)")
_OPTION = re.compile(r"^\s+:([\w-]+):\s*(.*)$")


def directives(page: Path) -> Iterator[tuple[str, str, dict[str, str]]]:
    """The autodoc and Python directives of an API page, with their options."""
    lines = page.read_text().splitlines()
    for number, line in enumerate(lines):
        if match := _DIRECTIVE.match(line):
            options = {}
            for following in lines[number + 1 :]:
                if not (option := _OPTION.match(following)):
                    break
                options[option[1]] = option[2]
            yield match[1], match[2], options


@cache
def api_entries(api_source: Path = API_SOURCE) -> dict[str, ApiEntry]:
    """Every name the API reference documents, with its page and anchor."""
    entries: dict[str, ApiEntry] = {}
    for source in sorted(api_source.glob("*.md")):
        page = source.with_suffix(".html").name
        documented: dict[str, str] = {}
        current = None
        for kind, name, options in directives(source):
            excluded = {
                part.strip() for part in options.get("exclude-members", "").split(",")
            }
            if kind == "automodule":
                current = name
                module = importlib.import_module(name)
                if "no-index" not in options:
                    documented[name] = "module"
                if "no-members" in options:
                    continue
                for member in public_names(module):
                    if member not in excluded:
                        documented |= _members(
                            f"{name}.{member}", getattr(module, member)
                        )
            elif kind.startswith("auto"):
                module_name, _, attribute = name.rpartition(".")
                value = getattr(importlib.import_module(module_name), attribute)
                documented |= _members(name, value)
            else:
                prefix = options.get("module", current)
                documented[f"{prefix}.{name}" if prefix else name] = "data"
        for name, kind in documented.items():
            entries.setdefault(name, ApiEntry(name, page, kind))
    return entries


def resolve(name: str, entries: dict[str, ApiEntry]) -> ApiEntry:
    """The API entry a role names: its documented name, or a unique last part of one.

    A name without a dot names a function, class or constant when one has it, such
    as ``model_matrix``; methods and properties are named with their class, such as
    ``Cube.select``, unless their name is unique. A module needs its full name.

    Raises:
        ValueError: If no entry or more than one entry has that name.
    """
    name = name.strip().removeprefix("~")
    if name in entries:
        return entries[name]
    found = [
        entry
        for full, entry in entries.items()
        if entry.kind != "module" and full.endswith("." + name)
    ]
    top_level = [
        entry for entry in found if entry.kind in ("function", "class", "data")
    ]
    if "." not in name and top_level:
        found = top_level
    if not found:
        raise ValueError(f"{name!r} is not documented in the API reference")
    if len(found) > 1:
        names = ", ".join(sorted(entry.name for entry in found))
        raise ValueError(
            f"{name!r} is ambiguous: it could be {names}; write more of it"
        )
    return found[0]


def role_parts(content: str) -> tuple[str | None, str]:
    """The link text, or None for the name itself, and the name of a role."""
    if match := ROLE_WITH_TEXT.match(content):
        return match["text"], match["name"].strip()
    return None, content.strip()


def role_link(content: str, entries: dict[str, ApiEntry]) -> str:
    """The Markdown link that a role ``{api}`content``` stands for."""
    text, name = role_parts(content)
    entry = resolve(name, entries)
    if text is None:
        shown = name.removeprefix("~")
        text = f"`{shown.rpartition('.')[2] if name.startswith('~') else shown}`"
    return f"[{text}]({entry.url})"


def rewrite_roles(text: str, entries: dict[str, ApiEntry]) -> str:
    """A Markdown text with every ``{api}`` role of its prose replaced by its link to
    the API reference; roles in code and inline code stay as they are."""

    def rewrite(prose: str) -> str:
        return ROLE_OR_CODE.sub(
            lambda m: role_link(m["role"], entries) if m["role"] else m[0], prose
        )

    return map_prose(text, rewrite)


# Reading the pages.


def page_url(file: str) -> str:
    """The path of a page on the site, as MyST forms it with ``folders: true``: the
    folders and the file name without its suffix, lower case, with characters other
    than letters, digits and hyphens as hyphens; ``index.md`` at the root is ``/``."""
    if file == "index.md":
        return "/"
    parts = Path(file).with_suffix("").parts
    slugs = [
        re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", part.lower())) for part in parts
    ]
    return "/" + "/".join(slug.strip("-") for slug in slugs)


def toc_pages(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The file, title and top-level section of every page of a MyST table of
    contents, in order; pages at the top level are in the section of their title."""
    found = []

    def visit(entries: list[dict[str, Any]], section: str | None) -> None:
        for entry in entries:
            here = section or entry.get("title", "")
            if "file" in entry:
                found.append((entry["file"], entry.get("title", ""), here))
            visit(entry.get("children", []), here)

    visit(config["project"]["toc"], None)
    return found


def markdown_pieces(text: str) -> list[tuple[str, str]]:
    """The consecutive pieces of a Markdown text, each with its kind: a fenced block
    of Python, a fenced block of other code, a fence line of a directive such as a
    note, or prose, which includes the content of directives. Joined, the pieces give
    the text back. Fence lines are pieces of their own, so that backticks of a fence
    are never read as inline code."""
    pieces: list[tuple[str, str]] = []
    code_fence: str | None = None
    directives: list[str] = []  # the fences of the directives open around a line

    def closes(line: str, fence: str) -> bool:
        stripped = line.strip()
        return stripped.startswith(fence[0] * len(fence)) and not stripped.strip(
            fence[0]
        )

    for line in text.splitlines(keepends=True):
        if code_fence is not None:
            pieces[-1] = (pieces[-1][0], pieces[-1][1] + line)
            if closes(line, code_fence):
                code_fence = None
            continue
        if directives and closes(line, directives[-1]):
            directives.pop()
            pieces.append((FENCE_LINE, line))
            continue
        if match := FENCE.match(line):
            info = match["info"]
            if PYTHON_FENCE.match(info) or CODE_FENCE.match(info):
                code_fence = match["fence"]
                pieces.append((PYTHON if PYTHON_FENCE.match(info) else CODE, line))
            else:
                directives.append(match["fence"])
                pieces.append((FENCE_LINE, line))
            continue
        if pieces and pieces[-1][0] == PROSE:
            pieces[-1] = (PROSE, pieces[-1][1] + line)
        else:
            pieces.append((PROSE, line))
    return pieces


def map_prose(text: str, function: Any) -> str:
    """A Markdown text with ``function`` applied to its prose and its code kept."""
    return "".join(
        function(piece) if kind == PROSE else piece
        for kind, piece in markdown_pieces(text)
    )


def split_markdown(text: str) -> tuple[str, str]:
    """The prose and the Python code of a Markdown text."""
    prose, code = [], []
    for kind, piece in markdown_pieces(text):
        if kind == PROSE:
            prose.append(piece)
        elif kind == PYTHON:
            lines = piece.splitlines(keepends=True)
            code.append("".join(lines[1:-1] if len(lines) > 1 else []))
    return "".join(prose), "\n".join(code)


def roles_in(prose: str) -> list[str]:
    """The names of the ``{api}`` roles of some prose, outside inline code."""
    return [role_parts(m["role"])[1] for m in ROLE_OR_CODE.finditer(prose) if m["role"]]


def read_page(path: Path) -> tuple[str, str]:
    """The prose and the Python code of a Markdown page or a notebook."""
    if path.suffix == ".ipynb":
        import json

        prose, code = [], []
        for cell in json.loads(path.read_text())["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                cell_prose, cell_code = split_markdown(text)
                prose.append(cell_prose)
            elif cell["cell_type"] == "code":
                code.append(text)
        return "\n\n".join(prose), "\n\n".join(code)
    return split_markdown(path.read_text())


def citations(prose: str) -> list[str]:
    """The citation keys of some prose, in order: ``[@key]``, ``@key`` and
    ``{cite}`key```, outside inline code and roles."""
    keys = []
    for match in CITE_ROLE.finditer(prose):
        keys += [key.strip() for key in match[1].split(",")]
    plain = ROLE_OR_CODE.sub(" ", CITE_ROLE.sub(" ", prose))
    plain = re.sub(r"\]\([^)]*\)", "]", plain)  # link targets, such as e-mail links
    keys += CITATION.findall(plain)
    return keys


def _title(path: Path, prose: str) -> str:
    match = re.search(r"^#\s+(.+)$", prose, re.M)
    return match[1].strip() if match else path.stem


def page_kind(file: str, section: str) -> str:
    if file in INDEX_PAGES:
        return INDEX
    if file in EXPLANATION_PAGES or section in EXPLANATION_SECTIONS:
        return EXPLANATION
    return GUIDE


@cache
def pages(source: Path = SOURCE) -> tuple[Page, ...]:
    """Every page of the table of contents, in its order."""
    import yaml

    config = yaml.safe_load((source / "myst.yml").read_text())
    found = []
    for file, title, section in toc_pages(config):
        prose, code = read_page(source / file)
        found.append(
            Page(
                file=file,
                title=title or _title(source / file, prose),
                section=section,
                url=page_url(file),
                kind=page_kind(file, section),
                prose=prose,
                code=code,
                roles=tuple(roles_in(prose)),
                api_links=tuple(match[2] for match in API_LINK.finditer(prose)),
                citations=tuple(citations(prose)),
            )
        )
    return tuple(found)


# What the code of a page uses.


def _code_tree(code: str) -> ast.Module:
    """The syntax tree of a page's code, without IPython's magics and shell lines."""
    lines = [
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in code.splitlines()
    ]
    return ast.parse("\n".join(lines))


def imported_names(code: str) -> set[str]:
    """The CytePool objects a page's code imports and uses, by the module and name it
    imports them from, such as ``cytepool.cube.from_cells``."""
    tree = _code_tree(code)
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "cytepool"
        ):
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                aliases[alias.asname or alias.name] = full
                used.add(full)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cytepool"):
                    aliases[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = []
            value: ast.expr = node
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id in aliases:
                used.add(".".join([aliases[value.id], *reversed(chain)]))
    return used


def documented_object_names(entries: dict[str, ApiEntry]) -> dict[int, str]:
    """The documented name of each function and class, by the identity of the object,
    so that a name imported from another module finds its entry."""
    found: dict[int, str] = {}
    for name, entry in entries.items():
        if entry.kind not in ("function", "class"):
            continue
        value = _lookup(name)
        if value is not None:
            found.setdefault(id(value), name)
    return found


def _lookup(name: str) -> Any:
    """The object a dotted name refers to, or None."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for part in parts[split:]:
                value = getattr(value, part)
        except AttributeError:
            return None
        return value
    return None


def code_uses(code: str, entries: dict[str, ApiEntry]) -> set[str]:
    """The documented names that a page's code imports from CytePool."""
    by_identity = documented_object_names(entries)
    found = set()
    for name in imported_names(code):
        if name in entries and entries[name].kind != "module":
            found.add(name)
            continue
        value = _lookup(name)
        if value is not None and id(value) in by_identity:
            found.add(by_identity[id(value)])
    return found


@dataclass(frozen=True)
class Links:
    """The pages that use and that explain each documented name."""

    used_in: dict[str, list[Page]]
    explained_in: dict[str, list[Page]]


@cache
def links(source: Path = SOURCE, api_source: Path = API_SOURCE) -> Links:
    """The pages that use and explain each name, from guides, tutorials and
    explanations; index pages give none.

    Raises:
        ValueError: If a role or a link to the API reference names nothing that is
            documented, or a role is ambiguous.
    """
    entries = api_entries(api_source)
    anchors = {entry.url.partition("#")[2]: name for name, entry in entries.items()}
    used_in: dict[str, list[Page]] = {}
    explained_in: dict[str, list[Page]] = {}
    for page in pages(source):
        names = set()
        for role in page.roles:
            try:
                names.add(resolve(role, entries).name)
            except ValueError as error:
                raise ValueError(f"{page.file}: {{api}}`{role}`: {error}") from None
        for anchor in page.api_links:
            if anchor not in anchors:
                raise ValueError(
                    f"{page.file} links to the API anchor {anchor}, which no API page "
                    "documents"
                )
            names.add(anchors[anchor])
        if page.kind == GUIDE:
            names |= code_uses(page.code, entries)
            target = used_in
        elif page.kind == EXPLANATION:
            target = explained_in
        else:
            continue
        for name in sorted(names):
            target.setdefault(name, []).append(page)
    return Links(used_in, explained_in)


def backlink_lines(name: str, links: Links) -> list[str]:
    """reStructuredText lines that link an API entry to the pages that use and explain
    it, relative to the API reference, which the site serves under ``api/``."""
    lines = []
    for label, found in (
        ("Used in", links.used_in),
        ("Explained in", links.explained_in),
    ):
        pages_of_name = found.get(name, [])
        if pages_of_name:
            targets = ", ".join(
                f"`{page.title} <..{page.url.rstrip('/')}/>`__"
                for page in pages_of_name
            )
            lines += ["", f"**{label}:** {targets}"]
    return lines
