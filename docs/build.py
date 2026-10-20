"""Build the FV-Solver HTML documentation.

    python docs/build.py            # build into docs/
    python docs/build.py --open     # build and open index.html

Pages are Markdown files in ``docs/source/`` with a small front-matter block::

    ---
    title: Riemann solvers, limiters, MUSCL
    nav: Schemes                 # short label in the page list
    section: Modules             # Guide | Modules | Test cases
    order: 30                    # position within the section
    header: Schemes.hpp          # optional: C++ header documented on this page
    ---

Directives inside a page:

* ``{{widget:NAME}}``  inlines ``docs/widgets/NAME.html`` (interactive explorers)
* ``{{api}}``          the API reference extracted from ``header`` (appended at
                       the end when the directive is absent)
* ``{{root}}``         relative path to the docs root (for images and links)
* ``{{defaults:CASE}}`` table of a test case's default parameters, read from the
                       built solver (``fvsolver --case CASE --print-case-defaults``)

Math is written as ``$...$`` / ``$$...$$`` and rendered by MathJax.  The API
reference is parsed from the ``//`` comments above each declaration in
``cpp/include/fv/*.hpp``, so it always matches the code.  The look reuses
pdoc's stylesheet (``_templates/pdoc_base.css``) plus ``_templates/custom.css``.

Requires the ``markdown2`` package (installed with pdoc).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import subprocess

import markdown2

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
SOURCE = DOCS / "source"
HEADERS = ROOT / "cpp" / "include" / "fv"
SOLVER = ROOT / "cpp" / "build" / ("fvsolver.exe" if __import__("os").name == "nt" else "fvsolver")
TEMPLATES = DOCS / "_templates"
SECTIONS = ["Guide", "Modules", "Test cases"]

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
        depth = len(self.out.relative_to(DOCS).parts) - 1
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
        pages.append(Page(src, meta, body, DOCS / rel))
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


def render_markdown(text: str):
    text, store = protect_math(text)
    out = markdown2.markdown(text, extras=EXTRAS)
    return restore_math(str(out), store), restore_math(out.toc_html or "", store)


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
# Rendering
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - FV-Solver</title>
<style>{base_css}</style>
<style>{custom_css}</style>
<style>
  .fv-pagelist h3 {{ margin: 1.2em 0 .3em; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted, #51606f); }}
  .fv-search-results li {{ margin: .4em 0; }}
  .fv-search-results small {{ color: var(--muted, #51606f); }}
  main.pdoc figure {{ margin: 1.2em 0; }}
  main.pdoc figure img {{ max-width: 100%; }}
  main.pdoc figcaption {{ font-size: .9em; color: var(--muted, #51606f); }}
</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }} }};</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="{root}search_index.js"></script>
</head>
<body>
<nav class="pdoc">
  <label id="navtoggle" for="togglestate" class="pdoc-button"><svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'><path stroke-linecap='round' stroke="currentColor" stroke-miterlimit='10' stroke-width='2' d='M4 7h22M4 15h22M4 23h22'/></svg></label>
  <input id="togglestate" type="checkbox" aria-hidden="true" tabindex="-1">
  <div>
    {back}
    <input type="search" id="fv-search" placeholder="Search..." role="searchbox" aria-label="search">
    {contents}
    {api_nav}
    {pagelist}
  </div>
</nav>
<main class="pdoc">
  <section class="module-info">
    <h1 class="modulename">{heading}</h1>
    <div class="docstring">
{body}
    </div>
  </section>
{api}
</main>
<script>
(function () {{
  const box = document.getElementById("fv-search"), main = document.querySelector("main.pdoc"), original = main.innerHTML;
  const esc = s => s.replace(/[&<>"]/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}})[c]);
  box.addEventListener("input", () => {{
    const q = box.value.trim().toLowerCase();
    if (!q) {{ main.innerHTML = original; if (window.MathJax && MathJax.typeset) MathJax.typeset(); return; }}
    const terms = q.split(/\\s+/);
    const hits = (window.FV_SEARCH || []).filter(e => terms.every(t => (e.title + " " + e.heading + " " + e.text).toLowerCase().includes(t))).slice(0, 60);
    main.innerHTML = "<h1>Search results</h1>" + (hits.length ? "<ul class='fv-search-results'>" + hits.map(e =>
      `<li><a href="{root}${{e.url}}">${{esc(e.heading || e.title)}}</a> <small>${{esc(e.title)}}</small><br><small>${{esc(e.text.slice(0, 160))}}</small></li>`).join("") + "</ul>" : "<p>No results.</p>");
  }});
}})();
</script>
</body>
</html>
"""


def nav_api(nav) -> str:
    if not nav:
        return ""
    items = []
    for aid, name, kind, sub in nav:
        inner = "".join(f'<li><a class="{k}" href="#{mid}">{html.escape(n)}</a></li>' for mid, n, k in sub)
        items.append(f'<li><a class="{kind}" href="#{aid}">{html.escape(name)}</a>'
                     + (f'<ul class="memberlist">{inner}</ul>' if inner else "") + "</li>")
    return '<h2>API Documentation</h2><ul class="memberlist">' + "".join(items) + "</ul>"


def page_list(pages, current) -> str:
    out = ['<div class="fv-pagelist"><h2>Pages</h2>']
    for sec in SECTIONS:
        group = sorted((p for p in pages if p.meta.get("section") == sec), key=lambda p: int(p.meta.get("order", 99)))
        if not group:
            continue
        out.append(f"<h3>{sec}</h3><ul>")
        for p in group:
            href = current.rel_root + p.out.relative_to(DOCS).as_posix()
            label = html.escape(p.meta.get("nav", p.title))
            out.append(f'<li><a href="{href}">{"<strong>" + label + "</strong>" if p is current else label}</a></li>')
        out.append("</ul>")
    return "".join(out) + "</div>"


def case_defaults_md(case: str) -> str:
    """Markdown table of the case's own defaults, straight from the solver."""
    if not SOLVER.is_file():
        return f"*(Build the solver to list the defaults of {case}: `.\\cpp\\build.ps1`)*"
    out = subprocess.run([str(SOLVER), "--case", case, "--print-case-defaults"], capture_output=True, text=True,
                         check=True).stdout
    rows = [line.split(" = ", 1) for line in out.splitlines() if " = " in line]
    return "| Parameter | Default |\n|---|---|\n" + "\n".join(f"| `{k}` | `{v}` |" for k, v in rows)


def build(open_browser: bool = False) -> None:
    base_css = (TEMPLATES / "pdoc_base.css").read_text(encoding="utf-8")
    custom_css = (TEMPLATES / "custom.css").read_text(encoding="utf-8")
    pages = load_pages()
    search = []

    for p in pages:
        body = p.body.replace("{{root}}", p.rel_root)
        for case in re.findall(r"\{\{defaults:([^}]+)\}\}", body):
            body = body.replace(f"{{{{defaults:{case}}}}}", case_defaults_md(case.strip()))
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
        for k, w in enumerate(widgets):
            widget = (DOCS / "widgets" / f"{w}.html").read_text(encoding="utf-8")
            html_body = re.sub(rf"(<p>)?\x00WIDGET{k}\x00(</p>)?", lambda _: f'<div class="fv-widget">\n{widget}\n</div>', html_body)
        p.html, p.toc, p.api = html_body, toc, nav
        back = "" if p.out == DOCS / "index.html" else (
            f'<a class="pdoc-button module-list-button" href="{p.rel_root}index.html">&#8592;&nbsp;FV-Solver</a>')
        contents = f"<h2>Contents</h2>{toc}" if toc else ""
        p.out.parent.mkdir(parents=True, exist_ok=True)
        p.out.write_text(PAGE.format(title=html.escape(p.title), heading=html.escape(p.title), base_css=base_css,
                                     custom_css=custom_css, root=p.rel_root, back=back, contents=contents,
                                     api_nav=nav_api(nav), pagelist=page_list(pages, p), body=html_body, api=api),
                         encoding="utf-8")

        url = p.out.relative_to(DOCS).as_posix()
        plain = re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", html_body)   # widget code is not content
        plain = re.sub(r"<[^>]+>", " ", plain)
        search.append({"title": p.title, "url": url, "heading": "", "text": re.sub(r"\s+", " ", plain)[:400]})
        for hid, htext in re.findall(r'<h[23] id="([^"]+)">(.*?)</h[23]>', html_body):
            search.append({"title": p.title, "url": f"{url}#{hid}", "heading": re.sub(r"<[^>]+>", "", htext), "text": ""})
        for aid, name, _, sub in nav:
            search.append({"title": p.title, "url": f"{url}#{aid}", "heading": name, "text": ""})
            search += [{"title": p.title, "url": f"{url}#{mid}", "heading": f"{name}.{n}", "text": ""} for mid, n, _ in sub]

    (DOCS / "search_index.js").write_text("window.FV_SEARCH = " + json.dumps(search) + ";\n", encoding="utf-8")
    print(f"Built {len(pages)} pages into {DOCS}")
    if open_browser:
        webbrowser.open((DOCS / "index.html").as_uri())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the FV-Solver documentation.")
    ap.add_argument("--open", action="store_true", help="open index.html after building")
    build(ap.parse_args().open)
