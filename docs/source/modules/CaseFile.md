---
title: CaseFile - case.toml reader and writer
nav: CaseFile
section: Methods
order: 105
---

> **In short.** The reader and writer of `case.toml`, the file that defines a case. You will rarely need this page: the file format
> is described on [Writing a test case]({{root}}guide/writing-cases.html). It is here for the details of how the file is parsed and saved.

`load_case_file` parses a `case.toml` with [toml++](https://github.com/marzer/tomlplusplus).
It checks every section, key, value type, region and expression, and reports problems with
the file, line and a "did you mean" hint. `save_case_file` writes the canonical layout used
by `--save-case`. `evaluate_initial` builds the initial field: a background state (or a
profile's field), then each region in order, with expressions evaluated by
[TinyExpr](https://github.com/codeplea/tinyexpr). The file format itself is described on
[Writing a test case]({{root}}guide/writing-cases.html).

Numbers keep their exact value through load and save: a TOML float is stored as the
shortest decimal that reads back to the same double. Numbers in initial states are never
passed through the expression engine, so an initial field defined in a case file is bit
for bit the same as the equivalent C++ loop.

---

*Declarations: [`fv/CaseFile.hpp`]({{root}}api/CaseFile.html).*
