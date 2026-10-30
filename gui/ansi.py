"""ANSI colour codes of the solver's output (no Qt): `strip` for parsing, `spans` to render.

The solver colours its banner with SGR codes (24-bit or 256-colour foreground, bold, dim, reset); everything else is plain text.
"""
from __future__ import annotations

import re

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_ANY = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_BASIC = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5"]
_BRIGHT = ["#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff"]


def strip(text: str) -> str:
    """The text without any escape sequence."""
    return _ANY.sub("", text)


def _xterm256(n: int) -> str:
    if n < 8:
        return _BASIC[n]
    if n < 16:
        return _BRIGHT[n - 8]
    if n < 232:
        n -= 16
        level = lambda v: 0 if v == 0 else 55 + 40 * v   # noqa: E731 - the xterm cube: 0, 95, 135, 175, 215, 255
        return "#%02x%02x%02x" % (level(n // 36), level(n // 6 % 6), level(n % 6))
    g = 8 + 10 * (n - 232)
    return "#%02x%02x%02x" % (g, g, g)


def spans(text: str) -> list[tuple[str, dict]]:
    """[(text, style)] with style keys 'color' (#rrggbb), 'bold', 'dim'; escape sequences other than colours are dropped."""
    out: list[tuple[str, dict]] = []
    style: dict = {}
    pos = 0
    for m in _ANY.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], dict(style)))
        pos = m.end()
        sgr = _SGR.fullmatch(m.group(0))
        if not sgr:
            continue
        codes = [int(c) if c else 0 for c in sgr.group(1).split(";")] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                style = {}
            elif c == 1:
                style["bold"] = True
            elif c == 2:
                style["dim"] = True
            elif c == 22:
                style.pop("bold", None)
                style.pop("dim", None)
            elif 30 <= c <= 37:
                style["color"] = _BASIC[c - 30]
            elif 90 <= c <= 97:
                style["color"] = _BRIGHT[c - 90]
            elif c == 39:
                style.pop("color", None)
            elif c == 38 and i + 1 < len(codes):
                if codes[i + 1] == 2 and i + 4 < len(codes):
                    style["color"] = "#%02x%02x%02x" % tuple(min(max(v, 0), 255) for v in codes[i + 2:i + 5])
                    i += 4
                elif codes[i + 1] == 5 and i + 2 < len(codes):
                    style["color"] = _xterm256(codes[i + 2])
                    i += 2
            i += 1
    if pos < len(text):
        out.append((text[pos:], dict(style)))
    return out
