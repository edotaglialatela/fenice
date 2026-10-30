---
title: Environment switches
nav: Switches
section: Guide
order: 60
---

A few environment variables change how the solver runs without changing a case file: they select a code path for comparison
(A/B timing, bisecting a change), turn on a diagnostic, or point to a folder. They are all `FENICE_*`; `fenice --print-switches`
prints the same table. A flag is on when it is set to anything but an empty string or `0`; the solver reads each one through a
single table (`cpp/include/fv/Switches.hpp`), which rejects a name that is not in it.

{{switches}}

Example (PowerShell, one run with the scalar kernels):

```powershell
$env:FENICE_NO_SIMD = "1"; build\release\fenice.exe --case "Sod Shock Tube"; Remove-Item Env:FENICE_NO_SIMD
```

Besides these, the program honours the usual `OMP_NUM_THREADS` (OpenMP threads, when `--threads` is not given), `NO_COLOR` and
`FORCE_COLOR` (the colours of the banner), and, on Windows, `MSMPI_BIN` (where `mpiexec` is).

The switches that select an alternative implementation (`FENICE_NO_SIMD`, `FENICE_FULL_REBUILD`, `FENICE_REPLICATED_REMESH`,
`FENICE_REPLICATED_TOPOLOGY`) never change the results: the unit tests, `cases/validation/amr/amr_mpi.py --check` and the golden
dumps run the code with and without them and compare the states bit for bit.
