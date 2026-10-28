"""Build the documentation site from saved notebook outputs, without running code.

The MyST sources in ``docs/source`` are copied to ``docs/build/site``, where every
notebook is checked (executed, unchanged since, free of errors) and its figures get
their descriptions as alternative text, and every ``{api}`` role of a page becomes a
link to the API entry it names (``page_index.py``). ``myst build --html`` then
renders the site, and the Sphinx API reference, built beforehand into
``docs/build/api``, is copied into it under ``api/``, where the site's links to
``/api/...`` point. The API entries link back to the pages that use and explain them;
the build checks that every such page is where the links expect it.

A site served below a path, as docs.nygen.io/CytePool/ is, is built with that path in
``BASE_URL`` (``BASE_URL=/CytePool``). MyST puts it before the links of the pages, and
``stage_site`` puts it before the links of the table of contents, which MyST leaves.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import zlib
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import page_index

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
BUILD = DOCS / "build"
SITE = BUILD / "site"
API = BUILD / "api"
PUBLIC_URL = "https://docs.nygen.io/CytePool"


def code_fingerprint(notebook: dict[str, Any]) -> str:
    """Return the SHA-256 of a notebook's code cells, which execution records.

    Only code is fingerprinted, so prose can change without a new execution.
    """
    sources = [
        _text(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    encoded = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_execution(notebook: dict[str, Any], page: str) -> None:
    """Raise ValueError unless the saved outputs come from the notebook's current code.

    The notebook must record when it was executed and the fingerprint of the code it
    ran, that fingerprint must match its code now, the datasets it ran with must be
    those it names (metadata ``datasets``) as ``docs/datasets.py`` registers them,
    every code cell must have run, and no output may be an error.
    """
    metadata = notebook.get("metadata", {})
    if not metadata.get("executed_at"):
        raise ValueError(f"{page} has never been executed; run {_execute(page)}")
    if not metadata.get("executed_code_sha256"):
        raise ValueError(
            f"{page} records no fingerprint of its code; run {_execute(page)}"
        )
    if metadata["executed_code_sha256"] != code_fingerprint(notebook):
        raise ValueError(
            f"the code of {page} changed after it was executed; run {_execute(page)}"
        )
    declared = metadata.get("datasets", [])
    if declared or "executed_datasets" in metadata:
        datasets = page_index.docs_module("datasets")
        unknown = sorted(set(declared) - set(datasets.DATASETS))
        if unknown:
            raise ValueError(f"{page} names datasets {unknown} that are not registered")
        registered = {name: dict(datasets.DATASETS[name].files) for name in declared}
        if metadata.get("executed_datasets") != registered:
            raise ValueError(
                f"{page} was executed with other data than the datasets it names, as "
                f"registered in docs/datasets.py; run {_execute(page)}"
            )
    for number, cell in enumerate(_code_cells(notebook), start=1):
        if any(output["output_type"] == "error" for output in cell.get("outputs", [])):
            raise ValueError(
                f"code cell {number} of {page} failed when it was executed"
            )
        count = cell.get("execution_count")
        if _text(cell["source"]).strip() and (type(count) is not int or count < 1):
            raise ValueError(f"code cell {number} of {page} has not been executed")


def check_figures(notebook: dict[str, Any], page: str) -> None:
    """Raise ValueError unless every figure has a description for its alternative text.

    A code cell lists one description per figure it shows, in order, in its metadata
    ``figure_descriptions``.
    """
    for number, cell in enumerate(_code_cells(notebook), start=1):
        figures = _figures(cell)
        descriptions = cell.get("metadata", {}).get("figure_descriptions", [])
        if (
            not isinstance(descriptions, list)
            or len(descriptions) != len(figures)
            or not all(isinstance(text, str) and text.strip() for text in descriptions)
        ):
            raise ValueError(
                f"code cell {number} of {page} shows {len(figures)} figure(s); give "
                "each a description in the list metadata.figure_descriptions"
            )


def toc_with_base_url(
    entries: list[dict[str, Any]], base_url: str
) -> list[dict[str, Any]]:
    """Return a copy of a MyST table of contents with ``base_url`` before every link
    to a path of the site, such as ``/api/index.html``.

    Links to other sites, links that name a host (``//...``) and relative links are
    kept as they are.
    """
    result = []
    for entry in entries:
        entry = dict(entry)
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("/") and not url.startswith("//"):
            entry["url"] = base_url + url
        if "children" in entry:
            entry["children"] = toc_with_base_url(entry["children"], base_url)
        result.append(entry)
    return result


def toc_files(entries: list[dict[str, Any]]) -> list[str]:
    """Return the files of a MyST table of contents, depth first."""
    files = []
    for entry in entries:
        if "file" in entry:
            files.append(entry["file"])
        files.extend(toc_files(entry.get("children", [])))
    return files


def inventory_names(path: Path) -> set[str]:
    """The Python names of a Sphinx inventory (``objects.inv``)."""
    header = path.read_bytes().split(b"\n", 4)
    if len(header) < 5 or not header[0].startswith(b"# Sphinx inventory version 2"):
        raise ValueError(f"{path} is not a Sphinx inventory of version 2")
    lines = zlib.decompress(header[4]).decode("utf-8").splitlines()
    return {
        line.split(" ")[0]
        for line in lines
        if len(line.split(" ")) > 1 and line.split(" ")[1].startswith("py:")
    }


def rewrite_api_roles(site: Path, files: list[str]) -> None:
    """Replace the ``{api}`` roles of the staged pages by links to the API reference.

    Raises:
        ValueError: If a role names nothing that the API reference documents, names
            more than one entry, or names an entry that the built inventory of the
            API reference (``docs/build/api/objects.inv``), when there is one, lacks.
    """
    entries = page_index.api_entries()
    inventory = API / "objects.inv"
    built = inventory_names(inventory) if inventory.is_file() else None
    problems = []

    def link(match: Any, file: str) -> str:
        if not match["role"]:
            return str(match[0])  # inline code, kept as it is
        _, name = page_index.role_parts(match["role"])
        try:
            entry = page_index.resolve(name, entries)
        except ValueError as error:
            problems.append(f"{file}: {{api}}`{match['role']}`: {error}")
            return str(match[0])
        if built is not None and entry.name not in built:
            problems.append(f"{file}: {entry.name} is missing from {inventory}")
        return page_index.role_link(match["role"], entries)

    def rewrite(text: str, file: str) -> str:
        return page_index.map_prose(
            text,
            lambda prose: page_index.ROLE_OR_CODE.sub(lambda m: link(m, file), prose),
        )

    for file in files:
        path = site / file
        if file.endswith(".ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    cell["source"] = rewrite(_text(cell["source"]), file)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        else:
            path.write_text(rewrite(path.read_text(), file))
    if problems:
        raise ValueError("Links to the API reference:\n" + "\n".join(problems))


def check_page_urls(xref: Path, base_url: str = "") -> None:
    """Raise ValueError unless every page of the table of contents is at the URL that
    the links of the API reference to it use, according to the site's
    ``myst.xref.json``."""
    built = {
        reference["url"].removeprefix(base_url) or "/"
        for reference in json.loads(xref.read_text())["references"]
        if reference.get("kind") == "page"
    }
    missing = [
        f"{page.file}: {page.url}"
        for page in page_index.pages(SOURCE)
        if page.url not in built
    ]
    if missing:
        raise ValueError(
            "The API reference links to these pages at URLs the site does not have:\n"
            + "\n".join(missing)
        )


def stage_site() -> Path:
    """Copy the sources to the build directory and prepare them for MyST."""
    # A documentation dependency, imported here so that the tests can use the
    # notebook checks above without it.
    import yaml

    config = yaml.safe_load((SOURCE / "myst.yml").read_text())
    # Keep the theme MyST downloaded, so that later builds need no network.
    themes, kept = SITE / "_build" / "templates", BUILD / "templates"
    if themes.is_dir():
        shutil.rmtree(kept, ignore_errors=True)
        themes.rename(kept)
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(
        SOURCE,
        SITE,
        ignore=shutil.ignore_patterns("_build", "__pycache__", ".ipynb_checkpoints"),
    )
    if kept.is_dir():
        themes.parent.mkdir()
        kept.rename(themes)
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    if base_url:
        config["project"]["toc"] = toc_with_base_url(config["project"]["toc"], base_url)
        (SITE / "myst.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    files = toc_files(config["project"]["toc"])
    for page in files:
        if not page.endswith(".ipynb"):
            continue
        path = SITE / page
        notebook = json.loads(path.read_text())
        check_execution(notebook, page)
        check_figures(notebook, page)
        markdown = next(
            cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        prose = _text(markdown["source"])
        frontmatter = re.match(r"\A---\n(.*?)\n---\n", prose, re.DOTALL)
        metadata = (yaml.safe_load(frontmatter[1]) or {}) if frontmatter else {}
        if not metadata.get("description"):
            body = prose[frontmatter.end() :] if frontmatter else prose
            introduction = re.sub(r"(?m)^(?:#+\s.*|\([^)]+\)=)\s*$", "", body).strip()
            summary = re.split(r"\n\s*\n", introduction, maxsplit=1)[0]
            summary = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", summary)
            summary = re.sub(r"\[@[^\]]+\]|\{[\w:]+\}", "", summary)
            summary = " ".join(summary.translate(str.maketrans("", "", "`*")).split())
            summary = re.sub(r"\s+([,.;:!?])", r"\1", summary)
            if not summary:
                raise ValueError(f"No introductory paragraph in {page}")
            sentences = re.split(r"(?<=[.!?])\s+", summary)
            description = sentences.pop(0)
            for sentence in sentences:
                if len(description) + len(sentence) + 1 > 300:
                    break
                description += " " + sentence
            if len(description) > 300:
                description = description[:297].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
            metadata["description"] = description
            markdown["source"] = (
                "---\n"
                + yaml.safe_dump(metadata, sort_keys=False)
                + "---\n\n"
                + body.lstrip()
            )
        for cell in _code_cells(notebook):
            descriptions = cell.get("metadata", {}).get("figure_descriptions", [])
            for output, description in zip(_figures(cell), descriptions, strict=True):
                output["data"]["text/plain"] = description
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    rewrite_api_roles(SITE, files)
    return SITE


def navigation_head_html() -> str:
    return """<link rel="preload" as="font" href="https://docs.nygen.io/assets/brand/fonts/inter/latin.woff2" type="font/woff2" crossorigin/>
<script>
(() => {
  const observer = new MutationObserver(() => {
    if (!document.querySelector('main')) return;
    observer.disconnect();
    const sidebar = document.querySelector('.myst-primary-sidebar-nav');
    const home = document.querySelector('.myst-home-link');
    if (sidebar?.clientHeight && home) {
      const saved = sessionStorage.getItem('nygen-docs-sidebar:' + home.pathname);
      if (saved !== null) sidebar.scrollTop = Number(saved);
    }
  });
  observer.observe(document.documentElement, {childList: true, subtree: true});
})();
</script>
<script defer src="https://docs.nygen.io/assets/brand/docs.js"></script>"""


def html_fragment(contents: str, class_name: str) -> str:
    class Fragment(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.lines = [0, *(match.end() for match in re.finditer("\n", contents))]
            self.depth = 0
            self.start = 0
            self.matches: list[str] = []

        def position(self) -> int:
            line, column = self.getpos()
            return self.lines[line - 1] + column

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if self.depth:
                self.depth += tag not in {
                    "area",
                    "base",
                    "br",
                    "col",
                    "embed",
                    "hr",
                    "img",
                    "input",
                    "link",
                    "meta",
                    "param",
                    "source",
                    "track",
                    "wbr",
                }
            elif class_name in (dict(attrs).get("class") or "").split():
                self.start = self.position()
                self.depth = 1

        def handle_startendtag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            pass

        def handle_endtag(self, tag: str) -> None:
            if not self.depth:
                return
            self.depth -= 1
            if not self.depth:
                end = contents.index(">", self.position()) + 1
                self.matches.append(contents[self.start : end])

    parser = Fragment()
    parser.feed(contents)
    if parser.depth or len(parser.matches) != 1:
        raise ValueError(f"Expected one complete .{class_name} element")
    return parser.matches[0]


def include_api_navigation(site: Path) -> None:
    """Give the API reference entry of the staged table of contents the pages of the
    API index as its children, in the groups of the index's captioned tables of
    contents; pages listed before any caption come first, ungrouped."""
    import yaml

    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    source = DOCS / "build" / "api" / "index.html"
    navigation = html_fragment(source.read_text(), "docs-api-tree")
    links: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = links
    for caption, href, title in re.findall(
        r'<span\b[^>]*\bclass="caption-text"[^>]*>(.*?)</span>'
        r'|<a\b[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>',
        navigation,
        re.DOTALL,
    ):
        if caption:
            group = []
            links.append({"title": _plain_text(caption), "children": group})
            continue
        href = unescape(href)
        if urlsplit(href).scheme or href.startswith("//") or not href.endswith(".html"):
            raise ValueError(f"Expected an API page link in {source}: {href}")
        group.append(
            {
                "title": _plain_text(title),
                "url": urljoin(f"{base_url}/api/", href),
                "open_in_same_tab": True,
            }
        )
    empty = [entry["title"] for entry in links if entry.get("children") == []]
    if empty:
        raise ValueError(f"API navigation groups without pages in {source}: {empty}")
    if not links:
        raise ValueError(f"No API navigation links in {source}")
    config = yaml.safe_load((site / "myst.yml").read_text())
    entries = list(config["project"]["toc"])
    matches = []
    while entries:
        entry = entries.pop()
        if entry.get("url") == f"{base_url}/api/index.html":
            matches.append(entry)
        entries.extend(entry.get("children", []))
    if len(matches) != 1:
        raise ValueError(f"Expected one API reference entry in {site / 'myst.yml'}")
    matches[0]["children"] = links
    (site / "myst.yml").write_text(yaml.safe_dump(config, sort_keys=False))


def integrate_api_navigation(html: Path) -> None:
    def static_classes(match: re.Match[str]) -> str:
        classes = [
            token
            for token in match[1].split()
            if (
                token.startswith("myst-")
                or token in {"collapsible-content", "sr-only", "hide-mac", "show-mac"}
            )
            and token not in {"myst-toc-item-exact", "myst-search-bar-disabled"}
        ]
        if "myst-search-bar" in classes:
            classes.append("search-button__button")
        if "myst-top-nav-menu-button" in classes:
            classes.extend(("sidebar-toggle", "primary-toggle"))
        return f'class="{" ".join(classes)}"'

    home = (html / "index.html").read_text()
    header = html_fragment(home, "myst-top-nav-bar")
    header = re.sub(r"<script\b[^>]*>.*?</script>", "", header, flags=re.DOTALL)
    header = re.sub(r'\bclass="([^"]*)"', static_classes, header)
    header, count = re.subn(
        r'<button\b[^>]*\bclass="[^"]*\bmyst-search-bar\b[^"]*"[^>]*>',
        lambda match: (
            re.sub(
                r' (?:aria-expanded|data-state|aria-label|aria-controls)="[^"]*"',
                "",
                match[0][:-1],
            )
            + ' aria-label="Search documentation" aria-controls="pst-search-dialog">'
        ),
        header,
    )
    if count != 1:
        raise ValueError("Expected one MyST search button in the guide header")
    navigation = html_fragment(home, "myst-primary-sidebar-toc")
    navigation = re.sub(r' aria-current="[^"]*"', "", navigation)
    navigation = re.sub(r'\bclass="([^"]*)"', static_classes, navigation)
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    for page in (html / "api").glob("*.html"):
        contents = page.read_text()
        route = f"{base_url}/{page.relative_to(html).as_posix()}"

        def mark_current(match: re.Match[str]) -> str:
            tag = match[0]
            href = re.search(r'\bhref="([^"]+)"', tag)
            return (
                tag[:-1] + ' aria-current="page">'
                if href and unescape(href[1]) == route
                else tag
            )

        current_navigation = re.sub(r"<a\b[^>]*>", mark_current, navigation)
        contents = contents.replace(
            html_fragment(contents, "docs-header-placeholder"), header
        )
        contents = contents.replace(
            html_fragment(contents, "docs-api-tree"), current_navigation
        )
        if contents.count("</head>") != 1:
            raise ValueError(f"Expected one HTML head in {page}")
        page.write_text(
            contents.replace("</head>", navigation_head_html() + "</head>", 1)
        )


def link_shared_brand_assets(html: Path) -> None:
    import yaml

    source_options = yaml.safe_load((SOURCE / "myst.yml").read_text())["site"][
        "options"
    ]
    config = json.loads((html / "config.json").read_text())
    urls = {
        config["options"][key]: source_options[key]
        for key in ("logo", "logo_dark", "favicon", "style")
    }
    nodes = [config["parts"]["footer"]["mdast"]]
    symbols = []
    while nodes:
        node = nodes.pop()
        if node.get("type") == "image" and node.get("class") == "brand-symbol":
            symbols.append(node["url"])
        nodes.extend(node.get("children", []))
    if len(symbols) != 1:
        raise ValueError(f"Expected one Nygen symbol in {html / 'config.json'}")
    urls[symbols[0]] = "https://docs.nygen.io/assets/brand/logo/nygen-symbol.svg"
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    for page in html.rglob("*"):
        if page.suffix not in {".html", ".json"}:
            continue
        contents = page.read_text()
        for local, public in urls.items():
            contents = contents.replace(f"{base_url}/build{local}", public)
            contents = contents.replace(local, public)
        for name, public in (
            ("favicon.ico", source_options["favicon"]),
            ("myst-theme.css", source_options["style"]),
        ):
            contents = contents.replace(f"{base_url}/{name}", public)
        page.write_text(contents)
    roots = list((html / "build").glob("root-*.js"))
    if len(roots) != 1:
        raise ValueError(f"Expected one generated theme root in {html / 'build'}")
    contents = roots[0].read_text()
    for name, public in (
        ("favicon.ico", source_options["favicon"]),
        ("myst-theme.css", source_options["style"]),
    ):
        pattern = r'href:`\$\{[\w$]+\|\|""\}/' + re.escape(name) + r"`"
        contents, count = re.subn(pattern, f"href:{json.dumps(public)}", contents)
        if count != 1:
            raise ValueError(f"Expected one {name} link in {roots[0]}")
    roots[0].write_text(contents)
    renames = {
        roots[0].name: f"root-{hashlib.sha256(contents.encode()).hexdigest()[:12]}.js"
    }
    manifests = list((html / "build").glob("manifest-*.js"))
    if len(manifests) != 1:
        raise ValueError(f"Expected one generated manifest in {html / 'build'}")
    contents = manifests[0].read_text().replace(roots[0].name, renames[roots[0].name])
    renames[manifests[0].name] = (
        f"manifest-{hashlib.sha256(contents.encode()).hexdigest()[:12]}.js"
    )
    for page in html.rglob("*"):
        if page.suffix not in {".html", ".json", ".js"}:
            continue
        contents = page.read_text()
        updated = contents
        for old, new in renames.items():
            updated = updated.replace(old, new)
        if updated != contents:
            page.write_text(updated)
    for old, new in renames.items():
        (html / "build" / old).rename(html / "build" / new)
    for local in urls:
        (html / "build" / local.lstrip("/")).unlink()
    (html / "favicon.ico").unlink()
    (html / "myst-theme.css").unlink()


def prepare_site_template(site: Path) -> None:
    import yaml

    template_url = yaml.safe_load((site / "myst.yml").read_text())["site"]["template"]
    template = (
        site
        / "_build"
        / "templates"
        / "site"
        / hashlib.sha256(template_url.encode()).hexdigest()
    )
    if not (template / "template.yml").is_file():
        subprocess.run(
            ["myst", "templates", "download", "--site", template_url],
            cwd=site,
            check=True,
        )
    replacements = {
        "build/index.js": [
            (
                'Pje.default.useState(i);(0,Pje.useEffect)(()=>{o.state==="idle"'
                "&&s(tzt([t],e,r).includes(t.id))}",
                'Pje.default.useState(!0);(0,Pje.useEffect)(()=>{o.state==="idle"&&s(!0)}',
            ),
            (
                'title:o!=null&&o.title?`${o.title}${a?` - ${a}`:""}`:a',
                'title:o!=null&&o.title?o.title===a?`${a} documentation`:`${o.title}${a?` - ${a}`:""}`:a',
            ),
        ],
        "public/build/_shared/chunk-LS4ALTVG.js": [
            (
                '_0.default.useState(o);(0,_0.useEffect)(()=>{i.state==="idle"'
                "&&a(DS([t],e,r).includes(t.id))}",
                '_0.default.useState(!0);(0,_0.useEffect)(()=>{i.state==="idle"&&a(!0)}',
            ),
        ],
        "public/build/_shared/chunk-WGW5MSUV.js": [
            (
                'title:m!=null&&m.title?`${m.title}${g?` - ${g}`:""}`:g',
                'title:m!=null&&m.title?m.title===g?`${g} documentation`:`${m.title}${g?` - ${g}`:""}`:g',
            ),
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(replacements, sort_keys=True).encode()
    ).hexdigest()[:12]
    for filename, rules in replacements.items():
        path = template / filename
        if not path.is_file() and filename.startswith("public/"):
            matches = list(path.parent.glob(f"{path.stem}-*.js"))
            if len(matches) != 1:
                raise ValueError(f"Expected one cached theme module for {path}")
            path = matches[0]
        for original, replacement in rules:
            content = path.read_text()
            if content.count(original) == 1 and replacement not in content:
                path.write_text(content.replace(original, replacement))
            elif original in content or content.count(replacement) != 1:
                raise ValueError(f"Unexpected theme component in {path}")
    modules = list((template / "public" / "build").rglob("*.js"))
    filenames = {
        path.name: f"{path.stem}-{fingerprint}.js"
        for path in modules
        if not path.stem.endswith(f"-{fingerprint}")
    }
    if not filenames:
        return
    if len({path.name for path in modules}) != len(modules):
        raise ValueError(f"Duplicate browser module filenames in {template}")
    references = re.compile(r"(?<![\w$().\[\]-])[\w$().\[\]-]+\.js\b")
    for path in [template / "build" / "index.js", *modules]:
        path.write_text(
            references.sub(
                lambda match: filenames.get(match[0], match[0]), path.read_text()
            )
        )
    for path in modules:
        if path.name in filenames:
            path.rename(path.with_name(filenames[path.name]))


def update_page_metadata(contents: str, canonical: str) -> str:
    class Description(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[str] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            attributes = dict(attrs)
            if tag == "meta" and attributes.get("name") == "description":
                self.values.append(attributes.get("content") or "")

    if contents.count("</head>") != 1:
        raise ValueError(f"Expected one HTML head for {canonical}")
    head, body = contents.split("</head>", 1)
    titles = re.findall(r"<title>(.*?)</title>", head, re.DOTALL)
    description = Description()
    description.feed(head)
    if len(titles) != 1 or len(description.values) != 1 or not description.values[0]:
        raise ValueError(f"Expected a title and description for {canonical}")
    title = unescape(titles[0])
    tool = PUBLIC_URL.rsplit("/", 1)[-1]
    parts = re.split(r" (?:-|—) ", title)
    if len(parts) == 2 and (
        parts[0] == parts[1]
        or canonical == f"{PUBLIC_URL}/api/index.html"
        and parts[1] == f"{tool} API reference"
    ):
        title = parts[0]
    if canonical == f"{PUBLIC_URL}/" and title == tool:
        title = f"{tool} documentation"
    if (
        canonical == f"{PUBLIC_URL}/api/{tool.lower()}.html"
        and title == f"{tool} API reference"
    ):
        title = f"{tool} Python API"
    image = f"https://docs.nygen.io/assets/brand/logo/{tool.lower()}-dark.png"
    metadata = {
        "og:title": title,
        "og:description": description.values[0],
        "og:type": "website",
        "og:url": canonical,
        "og:image": image,
        "og:image:type": "image/png",
        "og:image:width": "512",
        "og:image:height": "512",
        "og:image:alt": f"{tool} logo",
        "twitter:card": "summary",
        "twitter:image": image,
        "twitter:image:alt": f"{tool} logo",
    }

    def remove_managed(match: re.Match[str]) -> str:
        key = re.search(r"""\b(?:name|property)=["']([^"']+)["']""", match[0])
        return "" if key and key[1] in metadata else match[0]

    head = re.sub(r"<meta\b[^>]*>", remove_managed, head)
    head = re.sub(
        r"<title>.*?</title>",
        lambda _: f"<title>{escape(title)}</title>",
        head,
        count=1,
        flags=re.DOTALL,
    )
    for name, value in metadata.items():
        attribute = "property" if name.startswith("og:") else "name"
        head += f'<meta {attribute}="{name}" content="{escape(value, quote=True)}"/>'
    return head + "</head>" + body


def write_discovery_metadata(html: Path) -> None:
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    references = json.loads((html / "myst.xref.json").read_text())["references"]
    routes = {
        reference["url"].removeprefix(base_url).rstrip("/") + "/"
        for reference in references
        if reference["kind"] == "page"
    }
    if not routes:
        raise ValueError(f"No documentation pages in {html / 'myst.xref.json'}")
    sources = sorted((html / "api" / "_sources").glob("*.md"))
    if not sources:
        raise ValueError(f"No API source pages in {html / 'api' / '_sources'}")
    routes.update(f"/api/{source.stem}.html" for source in sources)
    sitemap = ElementTree.Element(
        "urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
    )
    for route in sorted(routes):
        page = html / route.lstrip("/")
        if route.endswith("/"):
            page /= "index.html"
        canonical = f"{PUBLIC_URL}{route}"
        contents = update_page_metadata(page.read_text(), canonical)
        head, body = contents.split("</head>", 1)
        head = re.sub(r"""<link\b(?=[^>]*\brel=["']canonical["'])[^>]*>""", "", head)
        head += f'<link rel="canonical" href="{canonical}"/>'
        page.write_text(head + "</head>" + body)
        entry = ElementTree.SubElement(sitemap, "url")
        ElementTree.SubElement(entry, "loc").text = canonical
    ElementTree.indent(sitemap)
    ElementTree.ElementTree(sitemap).write(
        html / "sitemap.xml", encoding="utf-8", xml_declaration=True
    )
    (html / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {PUBLIC_URL}/sitemap.xml\n"
    )


def main() -> None:
    if not (API / "index.html").is_file():
        raise SystemExit(f"Build the API reference into {API} first: make -C docs api")
    try:
        site = stage_site()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    include_api_navigation(site)
    prepare_site_template(site)
    subprocess.run(["myst", "build", "--html", "--strict"], cwd=site, check=True)
    html = site / "_build" / "html"
    try:
        check_page_urls(
            html / "myst.xref.json", os.environ.get("BASE_URL", "").rstrip("/")
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    head = navigation_head_html()
    for page in html.rglob("*.html"):
        contents = page.read_text()
        if contents.count("</head>") != 1:
            raise ValueError(f"Expected one HTML head in {page}")
        page.write_text(contents.replace("</head>", head + "</head>", 1))
    shutil.copytree(
        API,
        html / "api",
        ignore=shutil.ignore_patterns(".buildinfo", ".doctrees"),
        dirs_exist_ok=True,
    )
    link_shared_brand_assets(html)
    integrate_api_navigation(html)
    write_discovery_metadata(html)
    print(f"Documentation written to {html}")


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _execute(page: str) -> str:
    return f'make -C docs execute PAGES="{page}"'


def _figures(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        output
        for output in cell.get("outputs", [])
        if any(kind.startswith("image/") for kind in output.get("data", {}))
    ]


def _plain_text(html: str) -> str:
    """The text of an HTML fragment, without tags, entities or runs of spaces."""
    return " ".join(unescape(re.sub(r"<[^>]+>", "", html)).split())


def _text(source: str | list[str]) -> str:
    return source if isinstance(source, str) else "".join(source)


if __name__ == "__main__":
    main()
