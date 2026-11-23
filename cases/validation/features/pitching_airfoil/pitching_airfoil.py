"""Pitching NACA 0012 at Re = 1000: post-processing of the runs of ``run_workstation.sh``.

    python pitching_airfoil.py dt DIR                        # time per step of a short run (used by the script to plan the frames)
    python pitching_airfoil.py post RESULTS [--reference polar.csv]   # statistics, gates, figures, report.md; exit code 1 if a gate fails
    python pitching_airfoil.py frames RUN_B --runs runs.tsv  # vorticity frames and a GIF from the snapshots of the fast pitch-up

The runs (``RESULTS/runs.tsv``): A (slow, 0 -> 6 degrees) at the fine and the coarser level, B (fast, 0 -> 15 degrees), and the static airfoil at several
angles on the same mesh. What is checked, with the bands fixed BEFORE the runs (smoke runs, ``*_quick``, report them and always exit 0):

  G1  symmetry          the static airfoil at 0 degrees has |mean Cl| < 0.005
  G2  the rest phase    until the rotation starts the pitching runs are the static 0-degree run: max |dCl| < 0.01, max |dCd| < 0.005
  G3  the moving wall   after the hold, run A has the mean Cl and Cd of the STATIC airfoil at its final angle, each within 3 %
  G4  critical angle    static 6 degrees steady, static 10 degrees unsteady (the literature puts the transition at about 8 degrees at Re = 1000:
                        Kurtulus, J. Fluids Struct. 2015 / Phys. Fluids 2019; the box here has slip walls at +-3 chords and Ma = 0.2)
  G5  mesh              the final Cl and Cd of run A change by less than 3 % from the coarser level to the fine one

Reported, not gated: the lag of the lift behind the static polar during the slow ramp, the lift overshoot of the fast pitch-up, and (with --reference)
the difference of the static means from a reference polar.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import fenice


# ----------------------------------------------------------------------------------------------------------------------------- runs
class Run:
    def __init__(self, row, root):
        self.tag, self.kind, self.alpha, self.omega, self.t_from, self.t_until, self.level, self.t_end = row[:8]
        self.alpha, self.omega = float(self.alpha), float(self.omega)
        self.t_from, self.t_until, self.level, self.t_end = float(self.t_from), float(self.t_until), int(self.level), float(self.t_end)
        self.dir = Path(root) / self.tag
        self._m = None

    def monitor(self):
        if self._m is None:
            folder = next((d for d in self.dir.iterdir() if d.is_dir() and any(d.glob("*_monitor.csv"))), self.dir)
            self._m = fenice.read_monitor(folder)
        return self._m

    def angle(self, t):
        """Incidence in degrees at time t (nose-up positive)."""
        t = np.asarray(t, dtype=float)
        if self.kind == "static":
            return np.full(t.shape, self.alpha)
        return self.alpha * np.clip((t - self.t_from) / (self.t_until - self.t_from), 0.0, 1.0)


def load_runs(root):
    rows = [line.rstrip("\n").split("\t") for line in open(Path(root) / "runs.tsv") if line.strip()]
    return {r[0]: Run(r, root) for r in rows}


def window_stats(t, y, t0, t1):
    w = (t >= t0) & (t <= t1)
    y = y[w]
    return float(y.mean()), float(0.5 * (y.max() - y.min())), int(w.sum())


# ------------------------------------------------------------------------------------------------------------------------------- dt
def cmd_dt(a):
    folder = next((d for d in Path(a.dir).rglob("*_monitor.csv")), None)
    d = fenice.read_monitor(folder.parent)
    print(f"{d['t'][-1] / d['step'][-1]:.6e}")


# ----------------------------------------------------------------------------------------------------------------------------- post
def cmd_post(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(a.root)
    quick = root.name.endswith("_quick")
    runs = load_runs(root)
    levels = sorted({r.level for r in runs.values()})
    L = levels[-1]
    statics = sorted((r for r in runs.values() if r.kind == "static" and r.level == L), key=lambda r: r.alpha)
    A = runs.get(f"A_L{L}")
    A_c = runs.get(f"A_L{L - 1}")
    B = runs.get(f"B_L{L}")
    out = {}

    def win(r):   # the window of the means of a static run: its last third, at most 12 time units
        t1 = r.t_end
        return t1 - min(12.0, t1 / 3.0), t1

    # statics
    stat = {}
    for r in statics:
        m = r.monitor()
        t0, t1 = win(r)
        cl, acl, n = window_stats(m["t"], m["Cl"], t0, t1)
        cd, acd, _ = window_stats(m["t"], m["Cd"], t0, t1)
        cm, _, _ = window_stats(m["t"], m["Cm"], t0, t1)
        unsteady = acl > 0.01 and acl > 0.02 * abs(cl)
        stat[r.alpha] = dict(Cl=cl, Cl_amp=acl, Cd=cd, Cd_amp=acd, Cm=cm, unsteady=bool(unsteady), window=[t0, t1], samples=n)
    out["static"] = {str(k): v for k, v in stat.items()}

    gates = []

    def gate(name, ok, text):
        gates.append((name, bool(ok), text))

    if 0.0 in stat:
        gate("G1 symmetry", abs(stat[0.0]["Cl"]) < 0.005, f"static 0 degrees: mean Cl = {stat[0.0]['Cl']:+.2e} (band 5e-3)")
    if 0.0 in stat and A is not None and B is not None:
        s0 = runs[f"S0_L{L}"].monitor()
        worst_l = worst_d = 0.0
        for r in (A, B):
            m = r.monitor()
            w = m["t"] < r.t_from
            cl0 = np.interp(m["t"][w], s0["t"], s0["Cl"])
            cd0 = np.interp(m["t"][w], s0["t"], s0["Cd"])
            worst_l = max(worst_l, float(np.max(np.abs(m["Cl"][w] - cl0))))
            worst_d = max(worst_d, float(np.max(np.abs(m["Cd"][w] - cd0))))
        gate("G2 rest phase", worst_l < 0.01 and worst_d < 0.005, f"pitching runs before the rotation vs static 0 degrees: max dCl = {worst_l:.2e} (band 1e-2), max dCd = {worst_d:.2e} (band 5e-3)")
    if A is not None and A.alpha in stat:
        m = A.monitor()
        t1 = A.t_end
        t0 = t1 - min(10.0, 0.5 * (t1 - A.t_until))
        cl, _, n = window_stats(m["t"], m["Cl"], t0, t1)
        cd, _, _ = window_stats(m["t"], m["Cd"], t0, t1)
        s = stat[A.alpha]
        el, ed = abs(cl / s["Cl"] - 1.0), abs(cd / s["Cd"] - 1.0)
        out["A_final"] = dict(Cl=cl, Cd=cd, window=[t0, t1], static_Cl=s["Cl"], static_Cd=s["Cd"])
        gate("G3 moving wall", el < 0.03 and ed < 0.03,
             f"run A after the hold (t = {t0:.1f}..{t1:.1f}): Cl = {cl:.4f} vs static {s['Cl']:.4f} ({100 * el:.2f} %), Cd = {cd:.4f} vs static {s['Cd']:.4f} ({100 * ed:.2f} %), bands 3 %")
    if 6.0 in stat and 10.0 in stat:
        gate("G4 critical angle", (not stat[6.0]["unsteady"]) and stat[10.0]["unsteady"],
             f"6 degrees: Cl amplitude {stat[6.0]['Cl_amp']:.4f} ({'unsteady' if stat[6.0]['unsteady'] else 'steady'}); 10 degrees: {stat[10.0]['Cl_amp']:.4f} ({'unsteady' if stat[10.0]['unsteady'] else 'steady'}); literature: transition about 8 degrees")
    if A is not None and A_c is not None:
        def final(r):
            m = r.monitor()
            t1 = r.t_end
            t0 = t1 - min(10.0, 0.5 * (t1 - r.t_until))
            return window_stats(m["t"], m["Cl"], t0, t1)[0], window_stats(m["t"], m["Cd"], t0, t1)[0]
        (clf, cdf), (clc, cdc) = final(A), final(A_c)
        el, ed = abs(clf / clc - 1.0), abs(cdf / cdc - 1.0)
        out["mesh"] = dict(fine=[clf, cdf], coarse=[clc, cdc])
        gate("G5 mesh", el < 0.03 and ed < 0.03, f"run A, level {L - 1} -> {L}: Cl {clc:.4f} -> {clf:.4f} ({100 * el:.2f} %), Cd {cdc:.4f} -> {cdf:.4f} ({100 * ed:.2f} %), bands 3 %")

    # informational: lag during the ramp of A, overshoot of B
    info = []
    if A is not None and len(stat) >= 2:
        m = A.monitor()
        al = np.array(sorted(stat))
        clp = np.array([stat[x]["Cl"] for x in al])
        w = (m["t"] >= A.t_from) & (m["t"] <= A.t_until)
        quasi = np.interp(A.angle(m["t"][w]), al, clp)
        d = m["Cl"][w] - quasi
        info.append(f"A, during the ramp: Cl - static polar Cl(alpha(t)): mean {d.mean():+.4f}, at the end of the ramp {d[-1]:+.4f} (the unsteady lag; static values interpolated linearly between {al.min():g} and {al.max():g} degrees)")
    if B is not None and B.alpha in stat:
        m = B.monitor()
        w = m["t"] >= B.t_from
        i = int(np.argmax(m["Cl"][w]))
        s = stat[B.alpha]
        info.append(f"B: peak Cl = {m['Cl'][w][i]:.3f} at t = {m['t'][w][i]:.2f} ({m['t'][w][i] - B.t_from:+.2f} after the start of the rotation, {m['t'][w][i] - B.t_until:+.2f} after its end); static {B.alpha:g} degrees: mean {s['Cl']:.3f}, amplitude {s['Cl_amp']:.3f}; peak Cd = {m['Cd'][w].max():.3f} (static mean {s['Cd']:.3f})")
    ref_lines = []
    if a.reference and Path(a.reference).is_file():
        ref = np.genfromtxt(a.reference, delimiter=",", names=True)
        for x, s in sorted(stat.items()):
            cl_r = float(np.interp(x, ref["alpha_deg"], ref["Cl"]))
            cd_r = float(np.interp(x, ref["alpha_deg"], ref["Cd"]))
            ref_lines.append(f"| {x:g} | {s['Cl']:.4f} | {cl_r:.4f} | {100 * (s['Cl'] / cl_r - 1) if cl_r else float('nan'):+.1f} % | {s['Cd']:.4f} | {cd_r:.4f} | {100 * (s['Cd'] / cd_r - 1):+.1f} % |")

    # figures
    def style(ax, xl, yl):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)

    if stat:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        al = np.array(sorted(stat))
        for k, key in enumerate(("Cl", "Cd", "Cm")):
            ax[k].errorbar(al, [stat[x][key] for x in al], yerr=[stat[x].get(key + "_amp", 0.0) for x in al], fmt="o-", capsize=3, label=f"static, level {L} (mean, bars: oscillation amplitude)")
            style(ax[k], "angle of attack [deg]", key)
            ax[k].axvline(8.0, color="gray", ls=":", label="critical angle in the literature (about 8 deg)")
        if A is not None and "A_final" in out:
            ax[0].plot([A.alpha], [out["A_final"]["Cl"]], "r^", label="run A after the hold")
            ax[1].plot([A.alpha], [out["A_final"]["Cd"]], "r^")
        ax[0].legend(fontsize=7)
        fig.suptitle("NACA 0012, Re = 1000: static polar")
        fig.tight_layout()
        fig.savefig(root / "polar.png", dpi=140)
        plt.close(fig)
    if A is not None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        for r, ls in ((A_c, "--"), (A, "-")):
            if r is None:
                continue
            m = r.monitor()
            ax[0, 0].plot(m["t"], r.angle(m["t"]), ls, label=f"level {r.level}")
            ax[0, 1].plot(m["t"], m["Cl"], ls, label=f"A, level {r.level}")
            ax[1, 0].plot(m["t"], m["Cd"], ls, label=f"A, level {r.level}")
            if r is A:
                ax[1, 1].plot(r.angle(m["t"]), m["Cl"], "-", label="run A (ramp and hold)")
        if A.alpha in stat:
            for k, key in ((0, "Cl"), (1, "Cd")):
                axx = ax[0, 1] if key == "Cl" else ax[1, 0]
                axx.axhline(stat[A.alpha][key], color="k", ls=":", label=f"static {A.alpha:g} deg")
        if stat:
            al = np.array(sorted(stat))
            ax[1, 1].plot(al, [stat[x]["Cl"] for x in al], "ko:", label="static polar")
        style(ax[0, 0], "t", "angle of attack [deg]")
        style(ax[0, 1], "t", "Cl")
        style(ax[1, 0], "t", "Cd")
        style(ax[1, 1], "angle of attack [deg]", "Cl")
        for x in ax.flat:
            x.legend(fontsize=7)
        fig.suptitle(f"A: slow pitch-up 0 -> {A.alpha:g} deg, then hold")
        fig.tight_layout()
        fig.savefig(root / "pitch_A.png", dpi=140)
        plt.close(fig)
    if B is not None:
        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        m = B.monitor()
        ax[0].plot(m["t"], B.angle(m["t"]))
        ax[1].plot(m["t"], m["Cl"])
        ax[2].plot(m["t"], m["Cd"])
        ax[3].plot(m["t"], m["Cm"])
        if B.alpha in stat:
            s = stat[B.alpha]
            for x, key in ((ax[1], "Cl"), (ax[2], "Cd"), (ax[3], "Cm")):
                x.axhline(s[key], color="k", ls=":", label=f"static {B.alpha:g} deg: mean")
            ax[1].legend(fontsize=7)
        for x, lab in zip(ax, ("angle [deg]", "Cl", "Cd", "Cm (quarter chord)")):
            x.set_ylabel(lab)
            x.grid(alpha=0.3)
        ax[-1].set_xlabel("t")
        fig.suptitle(f"B: fast pitch-up 0 -> {B.alpha:g} deg (t = {B.t_from:g} .. {B.t_until:g}), then hold")
        fig.tight_layout()
        fig.savefig(root / "pitch_B.png", dpi=140)
        plt.close(fig)

    # report
    lines = ["# Pitching NACA 0012, Re = 1000 (Ma 0.2, AMR, body at the finest level)", ""]
    if quick:
        lines += ["**Smoke run** (`QUICK=1`): coarse mesh and short times; the gates are reported but not meaningful.", ""]
    lines += ["| run | kind | angle [deg] | omega [rad/t] | rotation | level | t_end |", "|---|---|---|---|---|---|---|"]
    for r in runs.values():
        rot = "-" if r.kind == "static" else f"{r.t_from:g} .. {r.t_until:g}"
        lines.append(f"| {r.tag} | {r.kind} | {r.alpha:g} | {r.omega:.5f} | {rot} | {r.level} | {r.t_end:g} |")
    lines += ["", "## Static polar (means over the last time units; amplitude = half the peak-to-peak)", "", "| angle | Cl | Cl amplitude | Cd | Cd amplitude | Cm | flow |", "|---|---|---|---|---|---|---|"]
    for x in sorted(stat):
        s = stat[x]
        lines.append(f"| {x:g} | {s['Cl']:.4f} | {s['Cl_amp']:.4f} | {s['Cd']:.4f} | {s['Cd_amp']:.4f} | {s['Cm']:+.4f} | {'unsteady' if s['unsteady'] else 'steady'} |")
    lines += ["", "## Gates", ""]
    for name, ok, text in gates:
        lines.append(f"* **{name}: {'PASS' if ok else 'FAIL'}**: {text}")
    if info:
        lines += ["", "## Not gated", ""] + [f"* {t}" for t in info]
    if ref_lines:
        lines += ["", "## Against the reference polar", "", "| angle | Cl | Cl ref | diff | Cd | Cd ref | diff |", "|---|---|---|---|---|---|---|"] + ref_lines
    lines += ["", "Figures: `polar.png`, `pitch_A.png`, `pitch_B.png`; movie `pitch_B_vorticity.gif` (`.mp4` if ffmpeg is installed).", "",
              "Caveats: slip walls at +-3 chords (blockage 1/6) and Ma = 0.2 differ from the free-air incompressible references; the angle at the start of the rotation is exactly 0; "
              "the rotation starts and stops impulsively (constant rate).", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(out | {"gates": [[n, ok, t] for n, ok, t in gates]}, indent=1), encoding="utf-8")
    print("\n".join(lines))
    failed = [n for n, ok, _ in gates if not ok]
    print(f"\n{'ALL GATES PASS' if not failed else 'GATES FAILED: ' + ', '.join(failed)}" + ("  (smoke run: not meaningful)" if quick else ""))
    sys.exit(0 if quick or not failed else 1)


# --------------------------------------------------------------------------------------------------------------------------- frames
def cmd_frames(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    from fenice.geometry import naca4, place_airfoil, polygon_contains

    root = Path(a.run).parent
    runs = load_runs(root)
    run = runs[Path(a.run).name]
    pvd = next(Path(a.run).rglob("*.pvd"), None)
    if pvd is None:
        print("no snapshots (.pvd) found: no frames")
        return
    snaps = fenice.read_pvd(pvd)
    frames = root / "frames"
    frames.mkdir(exist_ok=True)
    xw, yw, px = (1.5, 8.5), (2.0, 4.0), 0.004
    X, Y = np.meshgrid(np.arange(xw[0], xw[1], px) + 0.5 * px, np.arange(yw[0], yw[1], px) + 0.5 * px)
    unit = naca4("0012")
    t_start = max(0.0, run.t_from - 1.0)
    done = []
    for t, path in snaps:
        if t < t_start:
            continue
        d = fenice.read_vtu(path)
        cx, cy = 0.5 * (d["x0"] + d["x1"]), 0.5 * (d["y0"] + d["y1"])
        near = (cx > xw[0] - 0.1) & (cx < xw[1] + 0.1) & (cy > yw[0] - 0.1) & (cy < yw[1] + 0.1)
        tree = cKDTree(np.column_stack([cx[near], cy[near]]))
        _, idx = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
        u = d["Velocity"][near, 0][idx].reshape(X.shape)
        v = d["Velocity"][near, 1][idx].reshape(X.shape)
        u, v = gaussian_filter(u, 2.0), gaussian_filter(v, 2.0)   # the leaves are piecewise constant: a light smoothing keeps the cell edges out of the vorticity
        omega = np.gradient(v, px, axis=1) - np.gradient(u, px, axis=0)
        body = polygon_contains(place_airfoil(unit, 1.0, float(run.angle(t)), 3.0, 3.0), X, Y)
        omega = np.ma.array(omega, mask=body)
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.imshow(omega, origin="lower", extent=(xw[0], xw[1], yw[0], yw[1]), cmap="RdBu_r", vmin=-15, vmax=15, interpolation="nearest")
        ax.contourf(X, Y, body.astype(float), levels=[0.5, 1.5], colors="k")
        ax.set_title(f"vorticity, NACA 0012 pitching to {run.alpha:g} deg, Re = 1000     t = {t:.2f}   alpha = {float(run.angle(t)):.1f} deg", fontsize=9)
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")
        fig.tight_layout()
        name = frames / f"frame_{len(done):04d}.png"
        fig.savefig(name, dpi=100)
        plt.close(fig)
        done.append(name)
        if not a.keep_vtu:
            for f in Path(path).parent.glob(Path(path).stem + "*"):   # the snapshot is rendered: drop it (a .pvtu lists pieces next to it)
                if f.suffix in (".vtu", ".pvtu"):
                    f.unlink(missing_ok=True)
    if done:
        ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in done]
        ims[0].save(root / "pitch_B_vorticity.gif", save_all=True, append_images=ims[1:], duration=120, loop=0)
        print(f"{len(done)} frames in {frames}, GIF {root / 'pitch_B_vorticity.gif'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dt")
    p.add_argument("dir")
    p.set_defaults(f=cmd_dt)
    p = sub.add_parser("post")
    p.add_argument("root")
    p.add_argument("--reference")
    p.set_defaults(f=cmd_post)
    p = sub.add_parser("frames")
    p.add_argument("run")
    p.add_argument("--runs")
    p.add_argument("--keep-vtu", type=int, default=0)
    p.set_defaults(f=cmd_frames)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
