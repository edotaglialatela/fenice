# Third-party code

Vendored unmodified, so the solver builds with no downloads or package manager.

| Library | Version | License | Used for |
|---|---|---|---|
| [toml++](https://github.com/marzer/tomlplusplus) (`tomlplusplus/toml.hpp`) | v3.4.0 | MIT | Reading case files (`cases/*/case.toml`) |
| [TinyExpr](https://github.com/codeplea/tinyexpr) (`tinyexpr/`) | commit `c3b2f32eee61762f4c9d89c2c08cf34556a4a780` (2026-09-21) | zlib | Analytic expressions in case-file initial conditions |

To update one, replace its files with the new release, update the version here, and rerun
the unit tests and `python cpp/tests/regression.py --mpi 4`.
