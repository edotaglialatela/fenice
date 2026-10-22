"""Validation uncertainty (ASME V&V 20, Roache 2009 ch. 11) of the benchmarks that have a reference value.

    python cases/validation/benchmarks/vv20.py             # results/validation/vv20/
    python cases/validation/benchmarks/vv20.py --docs      # also docs/img/vv20.png

For every benchmark quantity of gci.py with a reference value D (the cavity extrema against Ghia, Ghia & Shin
(1982), the 3-D Taylor-Green peak dissipation against the spectral DNS) it takes the fine-grid solution as S, the
comparison error E = S - D and the validation uncertainty U_val = sqrt(U_num^2 + U_input^2 + U_D^2) at about 95 %:

* U_num is the fine-grid GCI of the quantity times |S|; a quantity whose grid triplet gives no GCI is 'not
  assessable' and has no U_val: no other estimate is substituted;
* U_input = 0: the inputs (Reynolds number, lid speed, domain) are exact, a strong-model assumption;
* U_D = 0: Ghia's tables and the DNS value come without an uncertainty. This is a strict assumption, and it makes
  U_val a lower bound: a benchmark computed by another code is a gray area for validation (Roache 2009, sec. 10.10).

|E| <= U_val: the modelling error is not larger than the noise level; |E| >= 3 U_val: the modelling error is
evident; in between: indeterminate (the factor 3 is a choice made here). Figure: vv20.png (E / |D| with the interval
+- U_val / |D|); table: results/validation/vv20/vv20_table.md.
"""
import sys
from pathlib import Path

import numpy as np

from fenice import study
from fenice.vv20 import from_gci

sys.path.insert(0, str(Path(__file__).resolve().parent))   # gci.py, in the same folder
import gci  # noqa: E402

STUDY = "vv20"


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    plt = study.use_style()
    args.cavity_grids = [32, 64, 128]
    rows = [r for r in gci.cavity_rows(args) + gci.tg3d_rows(args)
            if r.reference is not None and not r.study.startswith("cavity, 1st")]
    results = []
    for r in rows:
        v = from_gci(f"{r.study}: {r.name}", r.values[-1], r.reference, r.main)
        results.append((r, v))
        if v.assessable:
            print(f"{v.name.replace('$', '')[:62]:62s} S {v.S:+.4g}  D {v.D:+.4g}  E {v.E:+.2e}  U_num {v.U_num:.2e}  "
                  f"U_val {v.U_val:.2e}  |E|/U_val {v.ratio:.2f}  {v.verdict}")
        else:
            print(f"{v.name.replace('$', '')[:62]:62s} S {v.S:+.4g}  D {v.D:+.4g}  E {v.E:+.2e}  {v.verdict}")

    lines = ["| quantity | $S$ | $D$ | $E = S - D$ | $U_{num}$ | $U_{val}$ | $|E|/U_{val}$ | reading |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r, v in results:
        lines.append(f"| {r.study}: {r.name} | {v.S:.4g} | {v.D:.4g} | {v.E:+.2e} | "
                     f"{'-' if v.U_num is None else format(v.U_num, '.2e')} | {'-' if v.U_val is None else format(v.U_val, '.2e')} | "
                     f"{'-' if v.ratio is None else format(v.ratio, '.2f')} | {v.verdict} |")
    (study.study_dir(STUDY) / "vv20_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(results) + 1.8))
    ys = np.arange(len(results))[::-1]
    for y, (r, v) in zip(ys, results):
        rel = v.E / abs(v.D) * 100
        if v.assessable:
            ax.errorbar([rel], [y], xerr=[v.U_val / abs(v.D) * 100], fmt="o", color="C0", capsize=4, lw=2)
        else:
            ax.plot([rel], [y], "o", mfc="white", mec="C3", ms=8)
            ax.annotate("not assessable", (rel, y), xytext=(8, 4), textcoords="offset points", fontsize=8, color="C3")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys, [f"{r.name}\n{r.study}" for r, _ in results], fontsize=8)
    ax.set_xlabel(r"comparison error $E / |D|$ [%], interval $\pm U_{val}$")
    ax.set_title("Validation uncertainty (V&V 20): the modelling error lies in $E \\pm U_{val}$", fontsize=11)
    fig.tight_layout()
    study.save(fig, STUDY, "vv20.png", args.docs)


if __name__ == "__main__":
    main()
