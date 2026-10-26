"""Sphinx configuration of the API reference; ``make -C docs api`` builds it."""

import re
import sys
from pathlib import Path
from typing import Any

# Document the working tree, also when another copy of cytepool is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
# The index of the site's pages, which says where each entry is used and explained.
sys.path.append(str(Path(__file__).resolve().parents[3]))

import page_index  # noqa: E402

project = "CytePool API"
copyright = "2026, Nygen Analytics AB"
author = "Nygen Analytics AB"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
master_doc = "index"
exclude_patterns = ["_build"]
language = "en"

# The docstrings describe the types and shapes of arguments in words, so the
# signatures show names and defaults only.
autodoc_typehints = "none"
# Show defaults as the source writes them, such as ``1 / 3``, not as their values.
autodoc_preserve_defaults = True
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# "Methods:" in module docstrings describes numerical methods, not class methods,
# and "Layers:" lists the layers of a cube as "Args:" lists arguments.
napoleon_custom_sections = ["Methods", ("Layers", "params_style")]
# List the attributes of a class as fields, so that a documented attribute is not
# described twice when autodoc also documents it as a property, and without the
# types of their annotations, which the descriptions give in words.
napoleon_use_ivar = True
napoleon_attr_annotations = False
myst_all_links_external = True

html_theme = "sphinx_book_theme"
html_title = "CytePool API reference"
html_theme_options = {
    "repository_url": "https://github.com/NygenAnalytics/CytePool",
    "path_to_docs": "docs/source/reference/api",
    "use_repository_button": True,
    "show_navbar_depth": 2,
    "navigation_with_keys": False,
}


def _skip_tuple_fields(
    app: Any, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool | None:
    """Leave out the generated docstrings of named-tuple fields.

    The class docstrings already describe each field under Attributes.
    """
    if (getattr(obj, "__doc__", None) or "").startswith("Alias for field number"):
        return True
    return None


def _escape_vertical_bars(
    app: Any, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """Keep absolute values such as |t| as written.

    Docstrings write |x| for an absolute value, which reStructuredText reads as a
    substitution. Bars are escaped outside inline literals, literal blocks and
    doctests, where an escape would show.
    """
    literal_indent = None
    in_doctest = False
    for number, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if literal_indent is not None and (not stripped or indent > literal_indent):
            continue
        literal_indent = None
        in_doctest = bool(stripped) and (in_doctest or stripped.startswith(">>>"))
        if in_doctest:
            continue
        parts = re.split(r"(``.*?``)", line)
        lines[number] = "".join(
            part if part.startswith("``") else part.replace("|", r"\|")
            for part in parts
        )
        if stripped.endswith("::"):
            literal_indent = indent


def _link_pages(
    app: Any, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """Add links to the pages of the site that use and explain an entry.

    A page uses or explains the names that its code imports and that its prose links
    to (``docs/page_index.py``); the links are relative, as the site serves this
    reference under ``api/``.
    """
    lines += page_index.backlink_lines(name, page_index.links())


def setup(app: Any) -> None:
    app.connect("autodoc-skip-member", _skip_tuple_fields)
    # Before napoleon (priority 500) turns the docstrings into reStructuredText.
    app.connect("autodoc-process-docstring", _escape_vertical_bars, priority=400)
    # After napoleon, so that the links follow the docstring's last section.
    app.connect("autodoc-process-docstring", _link_pages, priority=600)
