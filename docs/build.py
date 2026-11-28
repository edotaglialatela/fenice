"""Build the Fenice HTML documentation.

    python docs/build.py            # build into docs/site/ (gitignored)
    python docs/build.py --open     # build and open docs/site/index.html

The published copy is built by .github/workflows/docs.yml and served by GitHub Pages.

Pages are Markdown files in ``docs/source/`` with a small front-matter block::

    ---
    title: Riemann solvers, limiters, MUSCL
    nav: Schemes                 # short label in the page list
    section: API reference       # Start here | User guide | Methods | Test cases | Verification | API reference
    order: 30                    # position within the section
    header: Schemes.hpp          # optional: C++ header documented on this page
    ---

Optional keys: ``layout: home`` with ``tagline:`` (the landing page); on a case page ``group:`` (one of
``GROUPS``), ``thumb: img/NAME.png`` (add ``cover`` to fill the tile, a GIF gives a still frame) and ``summary:``
(one sentence, also the page description) put it in the gallery.

Directives inside a page:

* ``{{widget:NAME}}``  inlines ``docs/widgets/NAME.html`` (interactive explorers)
* ``{{api}}``          the API reference extracted from ``header`` (appended at
                       the end when the directive is absent)
* ``{{root}}``         relative path to the docs root (for images and links)
* ``{{defaults:CASE}}`` table of a test case's default parameters, read from the
                       built solver (``fenice --case CASE --print-case-defaults``)
* ``{{switches}}``     table of the FENICE_* environment switches, read from the
                       built solver (``fenice --print-switches``)
* ``{{gallery}}``      thumbnail grid of the case pages, one heading per group
* ``{{showcase}}``     the ``SHOWCASE`` case pages as a mosaic (landing page)

Math is written as ``$...$`` / ``$$...$$`` and rendered by MathJax.  The API
reference is parsed from the ``//`` comments above each declaration in
``cpp/include/fv/*.hpp``, so it always matches the code.  The look and the
behaviour (search, page list, copy buttons) are ``_templates/site.css`` and
``_templates/site.js``, copied next to the pages.

Requires the ``markdown2`` package; with ``Pillow`` the gallery gets small
thumbnails (else it shows the full images).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import subprocess

import markdown2

import fenice

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
SOURCE = DOCS / "source"
HEADERS = ROOT / "cpp" / "include" / "fv"
TEMPLATES = DOCS / "_templates"
SITE = DOCS / "site"   # the generated site (gitignored; published to GitHub Pages by CI)
SECTIONS = ["Start here", "User guide", "Methods", "Immersed boundary", "Test cases", "Verification", "API reference"]

EXTRAS = {"code-friendly": None, "cuddled-lists": None, "fenced-code-blocks": {"cssclass": "pdoc-code codehilite"},
          "footnotes": None, "header-ids": {"mixed": True, "reset-count": True}, "markdown-in-html": None,
          "strike": None, "tables": None, "toc": {"depth": 3}}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@dataclass
class Page:
    src: Path
    meta: dict
    body: str
    out: Path = None
    html: str = ""
    toc: str = ""
    api: list = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.meta.get("title", self.src.stem)

    @property
    def rel_root(self) -> str:
        depth = len(self.out.relative_to(SITE).parts) - 1
        return "../" * depth


def load_pages() -> list[Page]:
    pages = []
    for src in sorted(SOURCE.rglob("*.md")):
        text = src.read_text(encoding="utf-8")
        meta, body = {}, text
        m = re.match(r"---\n(.*?)\n---\n", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.split("#")[0].strip()
            body = text[m.end():]
        rel = src.relative_to(SOURCE).with_suffix(".html")
        pages.append(Page(src, meta, body, SITE / rel))
    return pages


def protect_math(text: str):
    """Swap $$..$$ / $..$ spans for placeholders so Markdown leaves them alone."""
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"\x00MATH{len(store) - 1}\x00"
    text = re.sub(r"\$\$.+?\$\$", keep, text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, text)
    return text, store


def restore_math(text: str, store: list) -> str:
    return re.sub(r"\x00MATH(\d+)\x00", lambda m: html.escape(store[int(m.group(1))], quote=False), text)


_DETAILS = re.compile(
    r'<details markdown="1">\n<summary>(.*?)</summary>\n\n(.*?)\n\n</details>', re.S)


def render_markdown(text: str):
    # A collapsible block: its body is Markdown, rendered on its own.
    blocks = []

    def keep_details(m):
        blocks.append((m.group(1), render_markdown(m.group(2))[0]))
        return f"\x00DETAILS{len(blocks) - 1}\x00"
    text = _DETAILS.sub(keep_details, text)
    text, store = protect_math(text)
    out = markdown2.markdown(text, extras=EXTRAS)
    html_out = str(out)
    for n, (summary, body) in enumerate(blocks):
        html_out = html_out.replace(
            f"<p>\x00DETAILS{n}\x00</p>",
            f"<details>\n<summary>{summary}</summary>\n{body}</details>")
    return restore_math(html_out, store), restore_math(out.toc_html or "", store)


# ---------------------------------------------------------------------------
# C++ header API extraction
# ---------------------------------------------------------------------------

@dataclass
class Decl:
    kind: str           # class | struct | enum | function | variable | alias
    name: str
    signature: str
    doc: str
    members: list = field(default_factory=list)


_IDENT = r"[A-Za-z_]\w*"


def _clean_comment(lines: list[str]) -> str:
    return " ".join(l.strip() for l in lines).strip()


def _statement(lines: list[str], i: int):
    """Collect a declaration starting at line i; returns (text, next_index, has_body)."""
    buf, depth_paren = [], 0
    while i < len(lines):
        line = re.sub(r"//.*", "", lines[i]).rstrip()
        buf.append(line.strip())
        depth_paren += line.count("(") - line.count(")")
        i += 1
        if depth_paren <= 0 and (line.endswith(";") or "{" in line):
            text = " ".join(buf)
            if "{" in text:
                # skip a body (inline function / enum / class handled by caller)
                depth = text.count("{") - text.count("}")
                while depth > 0 and i < len(lines):
                    l2 = re.sub(r"//.*", "", lines[i])
                    depth += l2.count("{") - l2.count("}")
                    i += 1
                return text, i, True
            return text, i, False
    return " ".join(buf), i, False


def _trailing_comment(line: str) -> str:
    m = re.search(r"//\s*(.*)$", line)
    return m.group(1).strip() if m else ""


def _func_name(sig: str) -> str | None:
    if "operator()" in sig:
        return "operator()"
    m = re.search(rf"(~?{_IDENT}|operator\S+)\s*\(", sig)
    return m.group(1) if m else None


def _var_names(sig: str) -> str:
    """'double xmin = 0, xmax = 0' -> 'xmin, xmax' (top-level commas only)."""
    parts, depth, cur = [], 0, ""
    for ch in sig:
        depth += ch in "<({[" and 1 or 0
        depth -= ch in ">)}]" and 1 or 0
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    names = [(re.findall(_IDENT, part.split("=")[0]) or ["?"])[-1] for part in parts]
    return ", ".join(names)


def _skip_member(sig: str) -> bool:
    return (sig.startswith(("using ", "friend ")) or "= default" in sig or "= delete" in sig
            or re.match(r"(virtual\s+)?~", sig) is not None)


def _member_kind(sig: str):
    head = sig.split("{")[0]
    if "(" in head.split("=")[0]:
        return "function"
    return "variable"


def parse_header(path: Path) -> list[Decl]:
    lines = path.read_text(encoding="utf-8").splitlines()
    decls, pending, i = [], [], 0
    in_ns = False
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if s.startswith("#") or not s:
            pending = [] if not s else pending
            i += 1
            continue
        if s.startswith("//"):
            pending.append(s[2:].strip())
            i += 1
            continue
        if s.startswith("namespace fv"):
            in_ns, pending = True, []
            i += 1
            continue
        if not in_ns or s.startswith("}"):
            pending = []
            i += 1
            continue
        doc = _clean_comment(pending) or _trailing_comment(raw)
        pending = []
        if re.match(rf"(class|struct)\s+{_IDENT}\s*;", s):   # forward declaration
            i += 1
            continue
        m = re.match(rf"(class|struct)\s+({_IDENT})\b([^;]*)$", s)
        if m and not s.endswith(";"):
            kind, name = m.group(1), m.group(2)
            decl = Decl(kind, name, s.rstrip("{").strip(), doc)
            i += 1
            access = "public" if kind == "struct" else "private"
            mpending = []
            while i < len(lines) and not lines[i].strip().startswith("};"):
                l = lines[i].strip()
                if l in ("public:", "private:", "protected:"):
                    access, mpending = l[:-1], []
                    i += 1
                    continue
                if not l:
                    mpending = []
                    i += 1
                    continue
                if l.startswith("//"):
                    mpending.append(l[2:].strip())
                    i += 1
                    continue
                if l.startswith("#"):
                    i += 1
                    continue
                trailing = _trailing_comment(lines[i])
                text, i, _ = _statement(lines, i)
                if access == "public":
                    mk = _member_kind(text)
                    sig = text.split("{")[0] if mk == "function" else text.split("{")[0]
                    sig = re.sub(r"\s+", " ", sig).strip().rstrip(";").strip()
                    sig = re.sub(r"\)\s*:\s*[A-Za-z_]\w*\s*\(.*$", ")", sig)   # constructor initialiser list
                    if re.search(r"=\s*$", sig):           # "x = {..." initialiser cut by the brace split
                        sig = sig.rstrip("= ").strip()
                    doc = _clean_comment(mpending) or trailing
                    if not doc and re.search(r"\boverride\b", sig):
                        doc = "Overrides the base-class method."
                    name = _func_name(sig) if mk == "function" else _var_names(sig)
                    if not _skip_member(sig):
                        decl.members.append(Decl(mk, name or "?", sig, doc))
                mpending = []
            decls.append(decl)
            i += 1
            continue
        m = re.match(rf"enum class\s+({_IDENT})", s)
        if m:
            text, i, _ = _statement(lines, i)
            values = re.search(r"\{(.*)\}", text)
            decls.append(Decl("enum", m.group(1), f"enum class {m.group(1)} {{ {values.group(1).strip() if values else ''} }}", doc))
            continue
        text, i, has_body = _statement(lines, i)
        sig = re.sub(r"\s+", " ", text.split("{")[0] if has_body else text).strip().rstrip(";").strip()
        if sig.startswith("using "):
            decls.append(Decl("alias", sig.split()[1], sig, doc))
        elif _member_kind(sig) == "function" and not sig.startswith(("constexpr", "extern")):
            decls.append(Decl("function", _func_name(sig) or "?", sig, doc))
        else:
            decls.append(Decl("variable", _var_names(sig), sig, doc))
    return decls


def _doc_html(text: str) -> str:
    if not text:
        return ""
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return f'<div class="docstring"><p>{t}</p></div>'


def _sig_html(d: Decl) -> str:
    sig = html.escape(d.signature, quote=False)
    name = html.escape(d.name)
    if d.kind in ("class", "struct", "enum"):
        rest = sig.split(name, 1)[1] if name in sig else ""
        return (f'<span class="def">{d.kind}</span> <span class="name">{name}</span>'
                f'<span class="signature pdoc-code condensed">{rest}</span>')
    if d.kind == "variable" and "," in d.name:
        return f'<span class="name">{sig}</span>'
    if name and name in sig:
        pre, post = sig.split(name, 1) if d.kind == "function" else sig.rsplit(name, 1)
        return (f'<span class="def">{pre.strip()}</span> <span class="name">{name}</span>'
                f'<span class="signature pdoc-code condensed">{post}</span>')
    return f'<span class="name">{sig}</span>'


def api_html(header: str):
    decls = parse_header(HEADERS / header)
    parts = [f'<h2 id="api-reference">API reference <code>fv/{header}</code></h2>']
    seen = set()

    def anchor(n):
        n = re.sub(r"[^\w.\-]+", "-", n.split(",")[0].strip()).strip("-")   # "Mesh.nx, ny" -> "Mesh.nx"
        a, k = n, 2
        while a in seen:
            a, k = f"{n}-{k}", k + 1
        seen.add(a)
        return a
    nav = []
    for d in decls:
        aid = anchor(d.name)
        kind_cls = "class" if d.kind in ("class", "struct", "enum") else d.kind
        parts.append(f'<section id="{aid}"><div class="attr {kind_cls}">{_sig_html(d)}</div>'
                     f'<a class="headerlink" href="#{aid}"></a>{_doc_html(d.doc)}')
        sub = []
        for mbr in d.members:
            mid = anchor(f"{d.name}.{mbr.name}")
            parts.append(f'<div id="{mid}" class="classattr"><div class="attr {mbr.kind}">{_sig_html(mbr)}</div>'
                         f'<a class="headerlink" href="#{mid}"></a>{_doc_html(mbr.doc)}</div>')
            sub.append((mid, mbr.name, mbr.kind))
        parts.append("</section>")
        nav.append((aid, d.name, kind_cls, sub))
    return "\n".join(parts), nav


# ---------------------------------------------------------------------------
# Site frame: top bar, page tree, on-this-page column, gallery
# ---------------------------------------------------------------------------

REPO = "https://github.com/edotaglialatela/fenice"
# Top-bar tabs: label, the page sections it covers, the page it opens.
TABS = [("Guide", ("Start here", "User guide"), "guide/getting-started.html"),
        ("Methods", ("Methods", "Immersed boundary"), "modules/overview.html"),
        ("Gallery", ("Test cases",), "cases/index.html"),
        ("Verification", ("Verification",), "guide/testing.html"),
        ("API", ("API reference",), "api/index.html")]
# Gallery groups (the ``group:`` of a case page), in display order.
GROUPS = ["Shocks and Riemann problems", "Vortices, instabilities and turbulence", "Viscous flows with walls",
          "Bodies in the flow", "Code verification"]
# Case pages shown on the home page ({{showcase}}), by file name; the first is shown large.
SHOWCASE = ["cylinder", "taylor-green-3d", "cylinder-supersonic", "sphere", "airfoil-pitch", "cavity"]
THUMB = (720, 450)   # gallery thumbnail size (16:10)

ICON_MENU = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>')
ICON_GITHUB = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 '
               '7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28'
               '-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64'
               '-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27'
               '.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87'
               ' 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42'
               '-3.58-8-8-8z"/></svg>')

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<meta name="description" content="{description}">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#10242c">
<link rel="icon" href="{root}img/brand/fenice.ico" sizes="any">
<link rel="icon" href="{root}img/brand/fenice-icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="{root}img/brand/fenice-icon-256.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap">
<link rel="stylesheet" href="{root}site.css">
<script>window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }} }};</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="{root}search_index.js" defer></script>
<script src="{root}site.js" defer></script>
</head>
"""

TOPBAR = """<a class="skip" href="#content">Skip to content</a>
<header class="topbar">
  {menu}
  <a class="brand" href="{root}index.html"><picture><source media="(prefers-color-scheme: dark)" srcset="{root}img/brand/fenice-logo-on-dark.svg"><img src="{root}img/brand/fenice-logo.svg" alt="Fenice" width="96" height="28"></picture></a>
  <nav class="tabs" aria-label="Sections">{tabs}</nav>
  <div class="search" role="search">
    <input type="search" id="fv-search" placeholder="Search" aria-label="Search the documentation" autocomplete="off" aria-controls="fv-results" aria-expanded="false">
    <kbd aria-hidden="true">/</kbd>
    <div id="fv-results" class="search-results" role="listbox" aria-label="Search results" hidden></div>
  </div>
  <a class="icon-link" href="{repo}" aria-label="Fenice on GitHub">{github}</a>
</header>
"""

DOC = """<body data-root="{root}">
{topbar}
<div class="frame">
<nav class="sidebar" id="sidebar" aria-label="All pages">
{tree}
</nav>
<main id="content" class="doc">
{crumbs}
<article class="doc-article">
<section class="module-info">
<h1 class="page-title">{heading}</h1>
{toc_inline}
<div class="docstring">
{body}
</div>
</section>
{api}
</article>
{pager}
{footer}
</main>
<aside class="toc" aria-label="On this page">
{toc}
</aside>
</div>
</body>
</html>
"""

HOME = """<body class="home" data-root="{root}">
{topbar}
<section class="hero" aria-label="Fenice">
<div class="hero-inner">
<div class="hero-text">
<h1 class="hero-title"><img src="{root}img/brand/fenice-logo-on-dark.svg" alt="Fenice" width="328" height="96" fetchpriority="high"></h1>
<p class="hero-tagline">{tagline}</p>
<p class="hero-actions"><a class="btn btn-primary" href="{root}guide/getting-started.html">Get started</a>
<a class="btn btn-quiet" href="{root}cases/index.html">Browse the gallery</a>
<a class="btn btn-quiet" href="{repo}">Source on GitHub</a></p>
<p class="hero-meta">C++17 &middot; OpenMP and MPI &middot; 2-D and 3-D &middot; Windows, Linux and macOS &middot; MIT licence</p>
</div>
</div>
</section>
<main id="content" class="home-main">
<div class="docstring">
{body}
</div>
{footer}
</main>
</body>
</html>
"""


def ordered(pages):
    """The pages of the tree, section by section, each section by ``order``."""
    return sorted((p for p in pages if p.meta.get("section") in SECTIONS),
                  key=lambda p: (SECTIONS.index(p.meta["section"]), int(p.meta.get("order", 99))))


def link(cur, p) -> str:
    return cur.rel_root + p.out.relative_to(SITE).as_posix()


def label(p) -> str:
    return html.escape(p.meta.get("nav", p.title))


def topbar(cur, menu: bool) -> str:
    sec = None if cur.meta.get("layout") == "home" else cur.meta.get("section")
    here = ' aria-current="true"'
    tabs = "".join(f'<a href="{cur.rel_root}{url}"{here if sec in secs else ""}>{name}</a>' for name, secs, url in TABS)
    btn = ('<button type="button" class="menu-btn" aria-controls="sidebar" aria-expanded="false" '
           f'aria-label="Show the list of pages">{ICON_MENU}</button>') if menu else ""
    return TOPBAR.format(menu=btn, root=cur.rel_root, tabs=tabs, repo=REPO, github=ICON_GITHUB)


def tree(pages, cur) -> str:
    out, seq, here = [], ordered(pages), ' aria-current="page"'
    for sec in SECTIONS:
        group = [p for p in seq if p.meta["section"] == sec]
        if not group:
            continue
        items = "".join(f'<li><a href="{link(cur, p)}"{here if p is cur else ""}>{label(p)}</a></li>' for p in group)
        is_open = " open" if cur.meta.get("section") == sec else ""
        out.append(f"<details{is_open}><summary>{sec}</summary><ul>{items}</ul></details>")
    return "\n".join(out)


def crumbs(pages, cur) -> str:
    sec = cur.meta.get("section")
    first = next((p for p in ordered(pages) if p.meta.get("section") == sec), None)
    if first is None:
        return ""
    sec_html = html.escape(sec) if first is cur else f'<a href="{link(cur, first)}">{html.escape(sec)}</a>'
    return (f'<nav class="crumbs" aria-label="Breadcrumb"><ol><li><a href="{cur.rel_root}index.html">Fenice</a></li>'
            f'<li>{sec_html}</li><li aria-current="page">{label(cur)}</li></ol></nav>')


def pager(pages, cur) -> str:
    seq = ordered(pages)
    if cur not in seq:
        return ""
    i = seq.index(cur)
    prev = seq[i - 1] if i > 0 else None
    nxt = seq[i + 1] if i + 1 < len(seq) else None
    parts = [f'<a class="prev" href="{link(cur, prev)}" rel="prev"><span class="pager-dir">Previous</span>{label(prev)}</a>'
             if prev else "<span></span>"]
    if nxt:
        parts.append(f'<a class="next" href="{link(cur, nxt)}" rel="next"><span class="pager-dir">Next</span>{label(nxt)}</a>')
    return f'<nav class="pager" aria-label="Previous and next page">{"".join(parts)}</nav>'


def footer(cur) -> str:
    src = cur.src.relative_to(ROOT).as_posix()
    header = cur.meta.get("header")
    hdr = f' &middot; <a href="{REPO}/blob/main/cpp/include/fv/{header}">fv/{header}</a>' if header else ""
    return (f'<footer class="site-foot"><p><a href="{REPO}/blob/main/{src}">Source of this page</a>{hdr}</p>'
            f'<p>Fenice &middot; MIT licence &middot; <a href="{REPO}">Repository</a></p></footer>')


def nav_api(nav) -> str:
    if not nav:
        return ""
    items = []
    for aid, name, kind, sub in nav:
        inner = "".join(f'<li><a class="{k}" href="#{mid}">{html.escape(n)}</a></li>' for mid, n, k in sub)
        items.append(f'<li><a class="{kind}" href="#{aid}">{html.escape(name)}</a>'
                     + (f"<ul>{inner}</ul>" if inner else "") + "</li>")
    return '<h2 class="api-title">API</h2><ul class="api-nav">' + "".join(items) + "</ul>"


def thumbnail(spec: str):
    """``img/x.gif cover`` -> (path under the site, width, height, fit). A small WebP still (a late frame of an
    animation) when Pillow is installed, else the original image."""
    parts = spec.split()
    src, fit = parts[0], ("cover" if "cover" in parts[1:] else "contain")
    path = DOCS / src
    if not path.exists():
        sys.exit(f"thumbnail not found: {src}")
    try:
        from PIL import Image
    except ImportError:
        return src, None, None, fit
    rel = f"img/thumbs/{path.stem}.{fit}.webp"
    dst = SITE / rel
    if not dst.exists() or dst.stat().st_mtime < path.stat().st_mtime:
        im = Image.open(path)
        frames = getattr(im, "n_frames", 1)
        if frames > 1:
            im.seek(frames * 2 // 3)   # a late frame: the developed flow
        im = im.convert("RGB")
        w, h = THUMB
        if fit == "cover":   # centre crop to the tile's aspect ratio
            scale = max(w / im.width, h / im.height)
            cw, ch = round(w / scale), round(h / scale)
            x0, y0 = (im.width - cw) // 2, (im.height - ch) // 2
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
        else:
            im.thumbnail((w, h), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "WEBP", quality=82, method=6)
    with Image.open(dst) as im:
        return rel, im.width, im.height, fit


def tile(cur, p) -> str:
    src, w, h, fit = thumbnail(p.meta["thumb"])
    size = f' width="{w}" height="{h}"' if w else ""
    return (f'<a class="tile fit-{fit}" href="{link(cur, p)}"><span class="tile-img"><img src="{cur.rel_root}{src}" '
            f'alt="" loading="lazy" decoding="async"{size}></span><span class="tile-name">{label(p)}</span>'
            f'<span class="tile-sum">{html.escape(p.meta.get("summary", ""))}</span></a>')


def gallery_md(pages, cur) -> str:
    """{{gallery}}: every page with a ``thumb:``, one heading per group (Markdown headings, so they enter the contents)."""
    cases = [p for p in ordered(pages) if p.meta.get("thumb")]
    stray = [p.src.name for p in cases if p.meta.get("group") not in GROUPS]
    if stray:
        sys.exit("gallery: unknown or missing group in " + ", ".join(stray))
    out = []
    for g in GROUPS:
        group = [p for p in cases if p.meta["group"] == g]
        if group:
            out.append(f'## {g}\n\n<div class="tiles">{"".join(tile(cur, p) for p in group)}</div>\n')
    return "\n".join(out)


def showcase_md(pages, cur) -> str:
    """{{showcase}}: the SHOWCASE case pages as a mosaic."""
    by_name = {p.src.stem: p for p in pages if p.meta.get("thumb")}
    return f'\n<div class="showcase">{"".join(tile(cur, by_name[s]) for s in SHOWCASE)}</div>\n'


def finish_html(text: str) -> str:
    """Heading permalinks, scrollable tables and lazy images on a rendered body (before the widgets go in)."""
    text = re.sub(r'<(h[234]) id="([^"]+)">(.*?)</\1>',
                  lambda m: (f'<{m[1]} id="{m[2]}">{m[3]}<a class="headerlink" href="#{m[2]}" '
                             f'aria-label="Link to this section"></a></{m[1]}>'), text, flags=re.S)
    text = text.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return re.sub(r"<img (?![^>]*\bloading=)", '<img loading="lazy" decoding="async" ', text)


def case_defaults_md(case: str) -> str:
    """Markdown table of the case's own defaults, straight from the solver."""
    try:
        solver = fenice.exe()
    except FileNotFoundError:
        return f"*(Build the solver to list the defaults of {case}: `.\\cpp\\build.ps1`)*"
    out = subprocess.run([solver, "--case", case, "--print-case-defaults"], capture_output=True, text=True,
                         check=True).stdout
    rows = [line.split(" = ", 1) for line in out.splitlines() if " = " in line]
    table = "| Parameter | Default |\n|---|---|\n" + "\n".join(f"| `{k}` | `{v}` |" for k, v in rows)
    return (f'<details markdown="1">\n<summary>Default parameters: {case} ({len(rows)} values)</summary>\n\n'
            f"{table}\n\n</details>")


def switches_md() -> str:
    """Markdown table of the FENICE_* environment switches, straight from the solver (``fenice --print-switches``)."""
    try:
        solver = fenice.exe()
    except FileNotFoundError:
        return "*(Build the solver to list the switches: `.\\cpp\\build.ps1`)*"
    out = subprocess.run([solver, "--print-switches", "--all"], capture_output=True, text=True, check=True).stdout
    rows = re.findall(r"^  (FENICE_\w+)  \((\w+), read by the (\w+), for the (\w+)\)\n      (.+)$", out, re.M)
    def table(aud):
        return "| Switch | Value | Read by | Effect |\n|---|---|---|---|\n" + "\n".join(
            f"| `{n}` | {kind} | {who} | {effect} |" for n, kind, who, a, effect in rows if a == aud)
    return (table("user") + "\n\n### Developer switches\n\nThese only select a slower or instrumented alternative; the defaults are the "
            "fastest paths, so a normal run sets none of them.\n\n" + table("developer"))


def check_parameters_documented() -> None:
    """Every parameter of the solver's table (``fenice --print-parameters``) must appear in guide/parameters.md."""
    try:
        solver = fenice.exe()
    except FileNotFoundError:
        print("note: solver not built, parameter table not checked against guide/parameters.md")
        return
    out = subprocess.run([solver, "--print-parameters"], capture_output=True, text=True, check=True).stdout
    keys = [line.split("	", 1)[0] for line in out.splitlines() if line]
    page = (SOURCE / "guide" / "parameters.md").read_text(encoding="utf-8")
    missing = [k for k in keys if f"`{k}`" not in page]
    if missing:
        sys.exit("guide/parameters.md does not document: " + ", ".join(missing))


def build(open_browser: bool = False) -> None:
    check_parameters_documented()
    pages = load_pages()
    search = []

    for p in pages:
        body = p.body.replace("{{root}}", p.rel_root)
        for case in re.findall(r"\{\{defaults:([^}]+)\}\}", body):
            body = body.replace(f"{{{{defaults:{case}}}}}", case_defaults_md(case.strip()))
        if "{{switches}}" in body:
            body = body.replace("{{switches}}", switches_md())
        if "{{gallery}}" in body:
            body = body.replace("{{gallery}}", gallery_md(pages, p))
        if "{{showcase}}" in body:
            body = body.replace("{{showcase}}", showcase_md(pages, p))
        widgets = re.findall(r"\{\{widget:(\w+)\}\}", body)
        for k, w in enumerate(widgets):   # injected after Markdown (raw HTML/JS must not be touched)
            body = body.replace(f"{{{{widget:{w}}}}}", f"\n\n\x00WIDGET{k}\x00\n\n", 1)
        api, nav = ("", [])
        if p.meta.get("header"):
            api, nav = api_html(p.meta["header"])
        split = "{{api}}" in body
        html_body, toc = render_markdown(body.replace("{{api}}", "\x00API\x00") if split else body)
        if split:
            html_body = re.sub(r"(<p>)?\x00API\x00(</p>)?", lambda _: "</div></section>" + api + '<section><div class="docstring">', html_body)
            api = ""

        url = p.out.relative_to(SITE).as_posix()
        plain = re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", html_body)   # widget code is not content
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", plain))).strip()
        search.append({"title": p.title, "url": url, "heading": "", "text": plain[:400]})
        for hid, htext in re.findall(r'<h[23] id="([^"]+)">(.*?)</h[23]>', html_body):
            search.append({"title": p.title, "url": f"{url}#{hid}", "heading": html.unescape(re.sub(r"<[^>]+>", "", htext)), "text": ""})
        for aid, name, _, sub in nav:
            search.append({"title": p.title, "url": f"{url}#{aid}", "heading": name, "text": ""})
            search += [{"title": p.title, "url": f"{url}#{mid}", "heading": f"{name}.{n}", "text": ""} for mid, n, _ in sub]

        html_body = finish_html(html_body)
        for k, w in enumerate(widgets):
            widget = (DOCS / "widgets" / f"{w}.html").read_text(encoding="utf-8")
            html_body = re.sub(rf"(<p>)?\x00WIDGET{k}\x00(</p>)?", lambda _: f'<div class="fv-widget">\n{widget}\n</div>', html_body)
        p.html, p.toc, p.api = html_body, toc, nav

        description = p.meta.get("summary") or p.meta.get("tagline") or plain[:155].rsplit(" ", 1)[0]
        home = p.meta.get("layout") == "home"
        title = "Fenice: a finite-volume solver for compressible flow" if home else f"{p.title} · Fenice"
        head = HEAD.format(title=html.escape(title), description=html.escape(description), root=p.rel_root)
        if home:
            page = HOME.format(root=p.rel_root, topbar=topbar(p, menu=False), tagline=html.escape(p.meta.get("tagline", "")),
                               repo=REPO, body=html_body, footer=footer(p))
        else:
            toc_side = (f"<h2>On this page</h2>{toc}" if toc else "") + nav_api(nav)
            toc_inline = f'<details class="toc-inline"><summary>On this page</summary>{toc}</details>' if toc else ""
            page = DOC.format(root=p.rel_root, topbar=topbar(p, menu=True), tree=tree(pages, p), crumbs=crumbs(pages, p),
                              heading=html.escape(p.title), toc_inline=toc_inline, body=html_body, api=api,
                              pager=pager(pages, p), footer=footer(p), toc=toc_side)
        p.out.parent.mkdir(parents=True, exist_ok=True)
        p.out.write_text(head + page, encoding="utf-8")

    (SITE / "search_index.js").write_text("window.FV_SEARCH = " + json.dumps(search) + ";\n", encoding="utf-8")
    shutil.copytree(DOCS / "img", SITE / "img", dirs_exist_ok=True)
    for name in ("site.css", "site.js"):
        shutil.copyfile(TEMPLATES / name, SITE / name)
    (SITE / ".nojekyll").write_text("", encoding="utf-8")   # GitHub Pages: serve the files as they are
    print(f"Built {len(pages)} pages into {SITE}")
    if open_browser:
        webbrowser.open((SITE / "index.html").as_uri())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the Fenice documentation.")
    ap.add_argument("--open", action="store_true", help="open index.html after building")
    build(ap.parse_args().open)
