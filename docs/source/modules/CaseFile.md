---
title: CaseFile - case.toml reader and writer
nav: CaseFile
section: Modules
order: 105
header: CaseFile.hpp
---

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

{{api}}
