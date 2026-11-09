"""Phase Q0 of the 3-D octree AMR: the measurements taken before any 3-D AMR code (no engine change).

    python cases/validation/amr/octree_q0.py run        # the uniform fields the estimate needs (sphere Re 20 / 100 at 20 cells per D)
    python cases/validation/amr/octree_q0.py estimate   # octree leaf counts from the uniform fields (sphere, 3-D Taylor-Green 96^3)
    python cases/validation/amr/octree_q0.py halo       # halo fraction of per-level Morton blocks, 2-D against 3-D
    python cases/validation/amr/octree_q0.py cost       # 3-D uniform cost per cell and step, peak memory
    python cases/validation/amr/octree_q0.py bench      # 2-D baseline of bench_cellflux (interleaved repeats, min and spread)

Everything goes to results/validation/amr_octree_q0/ (summary_<stage>.json). Set FENICE_EXE to the binary of record (Q0 keeps a
copy in results/validation/amr_octree_q0/bin/ with its commit) so later phases compare against the same build.

The leaf-count estimate reproduces the 2-D engine's rules in three dimensions, offline, on the uniform finest-level solution (as the
first study of the patch AMR did, but with the engine's indicator instead of a Lohner one):
  * the cells of level l are the conservative means of the finest cells under them (2^dim children per parent);
  * xi = 1 where a face neighbour of the same level differs by >= amr_criterion (0.1) in density or in speed |m|/rho (relative to
    min(|a|, |b|), floors 1e-2 / 0.1, CellIndicator::evaluate); then amr_diffusion (2) iterations adding diffusion_coeff (0.1) to both
    cells of a face where one cell is above 0.05; refine where xi >= amr_refine_threshold (0.15);
  * every level of the hierarchy is evaluated on the whole restricted grid (the engine sees a coarser leaf's value where a level is
    absent: a second-order difference in where the band ends, not in its size);
  * the body band: the finest-level cells whose centre is inside the body, dilated by `margin` cells (2 for the mask, 6 = the ghost
    wall's ghost_reach), keep the finest level;
  * 2:1 balance across all 3^dim - 1 neighbours (the ripple of CellMesh::adapt), from the finest level down.
Reported: leaves per level, leaves / uniform cells, and the work ratio sum_l N_l 2^l / (N_uniform 2^M) (level l advances 2^l times
per coarse step; the uniform mesh at the finest resolution takes 2^M steps of the finest dt).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fenice
from fenice import study

STUDY = "amr_octree_q0"
OUT = study.study_dir(STUDY)

# ---- the sphere (the case of features/ibm_sphere.py, Re as a parameter) -------------------------------------------------------
MACH, LX, LY, LZ, D, CX, CY, CZ = 0.3, 6.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.5
SPHERE = """[case]
name = "Sphere Re{re}"
description = "sphere in a channel, Re = {re}, Mach 0.3"
[mesh]
Lx = {Lx}
Ly = {Ly}
Lz = {Lz}
nx = {nx}
ny = {ny}
nz = {nz}
[physics]
use_viscous = true
mu = {mu}
use_bodyforce = false
[boundaries]
bc_left = "vel_inlet"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"
bc_front = "wall"
bc_back = "wall"
u_in = 1.0
rho_in = 1.0
p_out = {p}
[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"
reconstruction = "muscl"
muscl_limiter = "Van Leer"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.5
[run]
t_end = {t_end}
save_interval = 1000000000
monitor_interval = 50
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = {R}
ibm_cx = {cx}
ibm_cy = {cy}
ibm_cz = {cz}
[initial]
state = {{ rho = 1.0, u = 1.0, v = 0.0, p = {p}, w = 0.0 }}
"""


def sphere_case(re, n, t_end=30.0):
    c = 1.0 / MACH
    return study.write_case(STUDY, f"sphere_re{re}_{n}", SPHERE.format(
        re=re, Lx=LX, Ly=LY, Lz=LZ, nx=int(LX * n), ny=int(LY * n), nz=int(LZ * n), mu=D / re, p=c * c / 1.4, t_end=t_end,
        R=0.5 * D, cx=CX, cy=CY, cz=CZ))


def sphere_run_dir(re, n=20):
    """Re 20 at 20 cells per D is the ghost_20 run of features/ibm_sphere.py; Re 100 is run here."""
    if re == 20:
        return fenice.RESULTS / "validation" / "ibm_sphere" / f"ghost_{n}"
    return OUT / f"sphere_re{re}_{n}"


def cmd_run(args):
    for re in args.re:
        if re == 20 and (sphere_run_dir(20) / "final.npz").exists():
            print("sphere Re 20: reusing results/validation/ibm_sphere/ghost_20")
            continue
        t0 = time.time()
        d = study.run_cached(STUDY, f"sphere_re{re}_20", sphere_case(re, 20), "ibm_method=ghost", threads=args.threads,
                             force=args.force)
        print(f"sphere Re {re}: {d} ({time.time() - t0:.0f} s)")


# ---- fields -------------------------------------------------------------------------------------------------------------------
def sphere_field(re, n=20):
    """(rho, m) on the finest grid, arrays indexed [k, j, i]; m has the three momentum components last."""
    z = np.load(sphere_run_dir(re, n) / "final.npz")
    nx, ny, nz = int(LX * n), int(LY * n), int(LZ * n)
    U = np.asarray(z["U"]).reshape(nz, ny, nx, -1)
    return U[..., 0], np.stack([U[..., 1], U[..., 2], U[..., 4]], axis=-1)


def sphere_solid(shape, h):
    nz, ny, nx = shape
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    zz = (np.arange(nz) + 0.5) * h
    Z, Y, X = np.meshgrid(zz, y, x, indexing="ij")
    return (X - CX) ** 2 + (Y - CY) ** 2 + (Z - CZ) ** 2 <= (0.5 * D) ** 2


def tg_snapshots(n=96):
    case = fenice.RESULTS / "validation" / "taylor_green_3d" / f"n{n}"
    return fenice.read_pvd(study.case_output(case))


def tg_field(path):
    f = fenice.read_vtr(path)
    rho, vel = f["Density"], f["Velocity"]
    return rho, rho[..., None] * vel


# ---- the indicator and the tree -----------------------------------------------------------------------------------------------
def restrict(a, dim):
    """Conservative mean of the 2^dim children (the order of the sum does not matter for a count)."""
    s = a.shape
    if dim == 3:
        b = a.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2, *s[3:])
        return b.mean(axis=(1, 3, 5))
    b = a.reshape(s[0] // 2, 2, s[1] // 2, 2, *s[2:])
    return b.mean(axis=(1, 3))


def shifted(a, axis, d, periodic):
    """a shifted by d along `axis` (value of the neighbour at +d) and a validity mask (False past a non-periodic edge)."""
    b = np.roll(a, -d, axis=axis)
    ok = np.ones(a.shape[:len(periodic)], dtype=bool)
    if not periodic[axis]:
        idx = [slice(None)] * len(periodic)
        idx[axis] = slice(-d, None) if d > 0 else slice(0, -d)
        ok[tuple(idx)] = False
    return b, ok


def level_xi(rho, mom, periodic, criterion=0.1, diffusion=2, coeff=0.1, trigger=0.05, floor=1e-2, vfloor=0.1):
    """xi of every cell of one level (all cells present), CellIndicator::evaluate on a structured grid."""
    dim = rho.ndim
    speed = np.sqrt(np.sum(mom * mom, axis=-1)) / rho

    def test(a, b, fl):
        vmin = np.minimum(np.abs(a), np.abs(b))
        vmin = np.where(vmin < floor, fl, vmin)
        return np.abs(b - a) / vmin >= criterion

    hit = np.zeros(rho.shape, dtype=bool)
    nbr_ok = []
    for ax in range(dim):
        for d in (-1, 1):
            r2, ok = shifted(rho, ax, d, periodic)
            s2, _ = shifted(speed, ax, d, periodic)
            hit |= ok & (test(rho, r2, floor) | test(speed, s2, vfloor))
            nbr_ok.append((ax, d, ok))
    xi = hit.astype(float)
    for _ in range(diffusion):
        add = np.zeros_like(xi)
        above = xi > trigger
        for ax, d, ok in nbr_ok:
            a2, _ = shifted(above, ax, d, periodic)
            add += np.where(ok & (above | a2), coeff, 0.0)
        xi = xi + add
    return xi


def dilate(mask, r, periodic):
    """Chebyshev dilation by r cells (the 3^dim - 1 neighbours, r rings), wrapped on periodic axes."""
    out = mask.copy()
    for ax in range(mask.ndim):
        acc = out.copy()
        for d in range(1, r + 1):
            for s in (d, -d):
                b = np.roll(out, s, axis=ax)
                if not periodic[ax]:
                    idx = [slice(None)] * mask.ndim
                    idx[ax] = slice(0, s) if s > 0 else slice(s, None)
                    b[tuple(idx)] = False
                acc |= b
        out = acc
    return out


def dilate_stencil(mask, periodic):
    """The cells a residual reads: +-1, +-2 along each axis (MUSCL) and the edge cells (+-1, +-1) of two axes (viscous cross terms)."""
    dim = mask.ndim
    offs = []
    for ax in range(dim):
        for d in (-2, -1, 1, 2):
            o = [0] * dim
            o[ax] = d
            offs.append(o)
    for a in range(dim):
        for b in range(a + 1, dim):
            for da in (-1, 1):
                for db in (-1, 1):
                    o = [0] * dim
                    o[a], o[b] = da, db
                    offs.append(o)
    out = mask.copy()
    for o in offs:
        b = mask
        for ax, s in enumerate(o):
            if s == 0:
                continue
            b = np.roll(b, s, axis=ax)
            if not periodic[ax]:
                idx = [slice(None)] * dim
                idx[ax] = slice(0, s) if s > 0 else slice(s, None)
                b = b.copy()
                b[tuple(idx)] = False
        out |= b
    return out


def coarsen_any(mask, dim):
    s = mask.shape
    if dim == 3:
        return mask.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).any(axis=(1, 3, 5))
    return mask.reshape(s[0] // 2, 2, s[1] // 2, 2).any(axis=(1, 3))


def refine_bool(mask, dim):
    for ax in range(dim):
        mask = np.repeat(mask, 2, axis=ax)
    return mask


def build_tree(rho, mom, M, periodic, solid=None, margin=2, coeff=0.1, balance=True, criterion=0.1):
    """The split flags of every level (split[l] on the level-l grid: the cell has children) and the leaves per level."""
    dim = rho.ndim
    levels = [(rho, mom)]
    for _ in range(M):
        r, m = levels[-1]
        levels.append((restrict(r, dim), restrict(m, dim)))
    levels = levels[::-1]   # levels[l] = level l, l = 0 coarsest
    want = [level_xi(r, m, periodic, criterion=criterion, coeff=coeff) >= 0.15 for r, m in levels]
    if solid is not None:
        band = dilate(solid, margin, [False] * dim)
        b = band
        for l in range(M, -1, -1):
            want[l] |= b
            b = coarsen_any(b, dim) if l > 0 else b
    split = [None] * (M + 1)
    exists = np.ones(levels[0][0].shape, dtype=bool)
    for l in range(M):   # top-down: a cell splits if it exists and wants it
        split[l] = exists & want[l]
        exists = refine_bool(split[l], dim)
    split[M] = np.zeros(levels[M][0].shape, dtype=bool)
    if balance:   # 2:1 over every neighbour: a split cell of level l needs its neighbours of level l, i.e. their parents split
        for l in range(M - 1, 0, -1):
            need = coarsen_any(dilate(split[l], 1, periodic), dim)
            split[l - 1] |= need
            # a cell whose parent is newly split exists now; nothing else changes on level l
    leaves, exists = [], np.ones(levels[0][0].shape, dtype=bool)
    for l in range(M + 1):
        leaves.append(int(np.count_nonzero(exists & ~split[l])))
        if l < M:
            exists = refine_bool(split[l], dim)
    return split, leaves


def summarise(leaves, n_uniform, M):
    n = sum(leaves)
    work = sum(c * 2 ** l for l, c in enumerate(leaves))
    return {"leaves": leaves, "total": n, "uniform": n_uniform, "leaf_ratio": n / n_uniform,
            "work_ratio": work / (n_uniform * 2 ** M)}


def cmd_estimate(args):
    out = {"sphere": {}, "taylor_green": {}, "sphere_2d_slice": {}}
    for re in args.re:
        rho, mom = sphere_field(re)
        h = LX / rho.shape[2]
        solid = sphere_solid(rho.shape, h)
        per = [False, False, False]
        res = {}
        for M in (1, 2):
            for margin in (2, 6):
                for coeff in (0.1, 0.1 * 4 / 6):
                    for bal in (True, False):
                        _, lv = build_tree(rho, mom, M, per, solid, margin, coeff, bal)
                        key = f"M{M}_band{margin}_coeff{coeff:.3f}_{'bal26' if bal else 'nobal'}"
                        res[key] = summarise(lv, rho.size, M)
            print(f"sphere Re {re}, M {M}: " + "  ".join(
                f"{k.split('_', 1)[1]}: {v['leaf_ratio']:.3f}/{v['work_ratio']:.3f}" for k, v in res.items() if k.startswith(f"M{M}_")))
        out["sphere"][f"re{re}"] = res
        # the same geometry in 2-D: the mid-plane z = cz as a cylinder-like quadtree (2-D vs 3-D at equal resolution)
        kz = rho.shape[0] // 2
        r2, m2 = rho[kz], mom[kz][..., :2]
        s2 = solid[kz]
        res2 = {}
        for M in (1, 2):
            _, lv = build_tree(r2, m2, M, [False, False], s2, 6, 0.1, True)
            res2[f"M{M}_band6"] = summarise(lv, r2.size, M)
        out["sphere_2d_slice"][f"re{re}"] = res2
    shots = tg_snapshots()[::args.tg_stride]
    fields = [(ts, *tg_field(path)) for ts, path in shots]
    for M, crit in ((1, 0.1), (2, 0.1), (2, 0.3)):   # 0.3: is it the indicator's sensitivity or the flow that fills the box?
        series = []
        for ts, rho, mom in fields:
            _, lv = build_tree(rho, mom, M, [True, True, True], criterion=crit)
            series.append({"t": ts, **summarise(lv, rho.size, M)})
        lr = np.array([s["leaf_ratio"] for s in series])
        wr = np.array([s["work_ratio"] for s in series])
        t = np.array([s["t"] for s in series])
        k, key = int(np.argmax(lr)), f"M{M}_crit{crit}"
        out["taylor_green"][key] = {"series": series, "peak_leaf_ratio": float(lr[k]), "t_peak": float(t[k]),
                                    "min_leaf_ratio": float(lr.min()), "t_min": float(t[int(np.argmin(lr))]),
                                    "mean_leaf_ratio": float(np.trapezoid(lr, t) / (t[-1] - t[0])),
                                    "mean_work_ratio": float(np.trapezoid(wr, t) / (t[-1] - t[0]))}
        o = out["taylor_green"][key]
        print(f"TG 96^3 M {M} criterion {crit}: leaf ratio peak {lr[k]:.3f} at t = {t[k]:.1f}, min {o['min_leaf_ratio']:.3f} at "
              f"t = {o['t_min']:.1f}, time mean {o['mean_leaf_ratio']:.3f}, work mean {o['mean_work_ratio']:.3f}")
    fenice.write_summary(STUDY, out, folder=OUT / "estimate")
    return out


# ---- halo fraction ------------------------------------------------------------------------------------------------------------
def part1by2(x):
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    for s, m in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F), (4, 0x10C30C30C30C30C3),
                 (2, 0x1249249249249249)):
        x = (x | (x << np.uint64(s))) & np.uint64(m)
    return x


def part1by1(x):
    x = x.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    for s, m in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
                 (1, 0x5555555555555555)):
        x = (x | (x << np.uint64(s))) & np.uint64(m)
    return x


def morton(idx):
    """Morton keys of integer coordinates idx (n, dim) given as (i, j[, k])."""
    if idx.shape[1] == 3:
        return part1by2(idx[:, 0]) | (part1by2(idx[:, 1]) << np.uint64(1)) | (part1by2(idx[:, 2]) << np.uint64(2))
    return part1by1(idx[:, 0]) | (part1by1(idx[:, 1]) << np.uint64(1))


def halo_fraction(split, M, periodic, ranks, depths):
    """Per-level Morton blocks of equal leaf count on `ranks`; for each depth r the leaves of other ranks within r rings (of the
    owner's level, all 3^dim - 1 directions) of a rank's leaves. Returns {r: (max over ranks, mean over ranks)} of halo / owned."""
    dim = split[0].ndim
    # leaf id on the finest grid, and the owner of every leaf
    exists = np.ones(split[0].shape, dtype=bool)
    fine_id = None
    owner_parts, nleaf = [], 0
    level_leaves = []
    for l in range(M + 1):
        leaf = exists & ~split[l]
        coords = np.argwhere(leaf)            # rows (k, j, i) or (j, i)
        keys = morton(coords[:, ::-1])
        order = np.argsort(keys, kind="stable")
        coords = coords[order]
        n = len(coords)
        ids = np.full(leaf.shape, -1, dtype=np.int64)
        ids[tuple(coords.T)] = nleaf + np.arange(n)
        own = (np.arange(n) * ranks) // max(n, 1)
        owner_parts.append(own)
        level_leaves.append((l, coords, nleaf, n))
        up = ids
        for _ in range(M - l):
            up = refine_bool_ids(up, dim)
        fine_id = up if fine_id is None else np.where(up >= 0, up, fine_id)
        nleaf += n
        if l < M:
            exists = refine_bool(split[l], dim)
    owner = np.concatenate(owner_parts)
    res = {}
    for r in depths:
        frac = []
        for p in range(ranks):
            halo = set()
            owned = int(np.count_nonzero(owner == p))
            for l, coords, base, n in level_leaves:
                sel = coords[owner[base:base + n] == p]
                if len(sel) == 0:
                    continue
                g = np.zeros(split[l].shape, dtype=bool)
                g[tuple(sel.T)] = True
                g = dilate_stencil(g, periodic) if r == "stencil" else dilate(g, r, periodic)
                for _ in range(M - l):
                    g = refine_bool(g, dim)
                ids = np.unique(fine_id[g])
                halo.update(ids[owner[ids] != p].tolist())
            frac.append(len(halo) / max(owned, 1))
        res[str(r)] = (float(np.max(frac)), float(np.mean(frac)))
    return res, nleaf


def refine_bool_ids(a, dim):
    for ax in range(dim):
        a = np.repeat(a, 2, axis=ax)
    return a


def split_from_leaf_meta(path, nx, ny, M):
    """The split flags of a 2-D AMR dump (leaf_meta rows x0, x1, y0, y1, level), on grids [j, i] of every level."""
    z = np.load(path)
    meta = np.asarray(z["leaf_meta"])
    x0, y0 = meta[:, 0].min(), meta[:, 2].min()
    Lx, Ly = meta[:, 1].max() - x0, meta[:, 3].max() - y0
    split = [np.zeros((ny << l, nx << l), dtype=bool) for l in range(M + 1)]
    lv = meta[:, 4].astype(int)
    for l in range(1, M + 1):   # a leaf of level l marks its ancestors on levels 0 .. l-1 as split
        sel = meta[lv == l]
        i = np.floor((sel[:, 0] - x0) / Lx * (nx << l) + 0.5).astype(int)
        j = np.floor((sel[:, 2] - y0) / Ly * (ny << l) + 0.5).astype(int)
        for t in range(l):
            split[t][j >> (l - t), i >> (l - t)] = True
    return split, [int(np.count_nonzero(lv == l)) for l in range(M + 1)]


def cmd_halo(args):
    out = {}
    if args.airfoil:   # the real 2-D partner: the 720 x 360 airfoil after 50 steps (golden dump), max_level 2
        split, lv = split_from_leaf_meta(args.airfoil, 720, 360, 2)
        out["airfoil_720_2d"] = {"leaves": lv}
        for ranks in args.ranks:
            res, n = halo_fraction(split, 2, [False, False], ranks, ("stencil", 2, 3))
            out["airfoil_720_2d"][f"ranks{ranks}"] = {"per_rank": n // ranks, "halo": res}
            print(f"airfoil 720 2-D ({n} leaves {lv}) on {ranks} ranks ({n // ranks} per rank): "
                  + "  ".join(f"r={r}: max {a:.3f} mean {b:.3f}" for r, (a, b) in res.items()), flush=True)
    # compact proxies: one level, n = 131 072 leaves per rank, in 2-D and 3-D (the estimate 2 dim r n^((dim-1)/dim))
    for dim, shape, ranks in ((2, (2048, 2048), 32), (3, (128, 128, 128), 16)):
        split = [np.zeros(shape, dtype=bool)]
        res, n = halo_fraction(split, 0, [False] * dim, ranks, ("stencil", 2, 3))
        out[f"uniform_{dim}d"] = {"leaves": n, "ranks": ranks, "per_rank": n // ranks, "halo": res}
        print(f"uniform {dim}-D, {n // ranks} leaves per rank: " + "  ".join(f"r={r}: max {a:.3f} mean {b:.3f}" for r, (a, b) in res.items()))
    # the estimated sphere trees (3-D) and their mid-plane (2-D), M = 2, ghost band
    rho, mom = sphere_field(args.re[0])
    h = LX / rho.shape[2]
    solid = sphere_solid(rho.shape, h)
    split3, lv3 = build_tree(rho, mom, 2, [False] * 3, solid, 6)
    kz = rho.shape[0] // 2
    split2, lv2 = build_tree(rho[kz], mom[kz][..., :2], 2, [False] * 2, solid[kz], 6)
    for name, split, lv in (("sphere_3d", split3, lv3), ("sphere_slice_2d", split2, lv2)):
        dim = split[0].ndim
        out[name] = {"leaves": lv}
        for ranks in args.ranks:
            res, n = halo_fraction(split, 2, [False] * dim, ranks, ("stencil", 2, 3))
            out[name][f"ranks{ranks}"] = {"per_rank": n // ranks, "halo": res}
            print(f"{name} ({n} leaves) on {ranks} ranks ({n // ranks} per rank): "
                  + "  ".join(f"r={r}: max {a:.3f} mean {b:.3f}" for r, (a, b) in res.items()))
    fenice.write_summary(STUDY, out, folder=OUT / "halo")


# ---- 3-D uniform cost ---------------------------------------------------------------------------------------------------------
def timed_run(cmd, env=None):
    """Wall time and peak working set (MB) of one solver run."""
    import psutil
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ps = psutil.Process(p.pid)
    peak = 0
    while p.poll() is None:
        try:
            peak = max(peak, ps.memory_info().peak_wset)
        except psutil.Error:
            pass
        time.sleep(0.05)
    out = p.stdout.read()
    if p.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{out[-3000:]}")
    return time.perf_counter() - t0, peak / 2 ** 20, out


def cmd_cost(args):
    exe = fenice.exe()
    cases = {
        "sphere_re100_20": (str(sphere_case(100, 20)), ["ibm_method=ghost"], 120 * 60 * 60),
        "sphere_re100_20_mask": (str(sphere_case(100, 20)), ["ibm_method=mask"], 120 * 60 * 60),
        "taylor_green_64": ("Taylor-Green Vortex 3D", ["nx=64", "ny=64", "nz=64"], 64 ** 3),
        "taylor_green_96": ("Taylor-Green Vortex 3D", ["nx=96", "ny=96", "nz=96"], 96 ** 3),
    }
    out = {}
    for name, (case, sets, cells) in cases.items():
        row = {}
        for threads in args.threads:
            times = []
            for steps in (args.steps, 2 * args.steps):   # the difference removes the setup
                best = None
                for _ in range(args.repeat):
                    cmd = [str(exe), "--case", case, "--no-logo", "--threads", str(threads), "--set", "use_mpi=false",
                           "--set", f"output_root={OUT / 'cost_runs'}", "--set", "save_interval=1000000000",
                           "--set", "monitor_interval=0", "--set", f"max_steps={steps}", "--set", "t_end=1e9"]
                    for s in sets:
                        cmd += ["--set", s]
                    wall, peak, _ = timed_run(cmd)
                    best = (wall, peak) if best is None or wall < best[0] else best
                times.append(best)
            per = (times[1][0] - times[0][0]) / args.steps
            row[threads] = {"s_per_step": per, "ns_per_cell_step": per / cells * 1e9, "peak_MB": times[1][1],
                            "bytes_per_cell": times[1][1] * 2 ** 20 / cells, "setup_s": times[0][0] - args.steps * per}
            print(f"{name:22s} {threads:2d} threads: {per * 1e3:8.1f} ms/step  {per / cells * 1e9:7.1f} ns/cell/step  "
                  f"peak {times[1][1]:7.0f} MB ({times[1][1] * 2 ** 20 / cells:.0f} B/cell)", flush=True)
        out[name] = {"cells": cells, "threads": row}
    fenice.write_summary(STUDY, out, folder=OUT / "cost")


# ---- 2-D bench_cellflux baseline ----------------------------------------------------------------------------------------------
def cmd_bench(args):
    exe = Path(os.environ.get("FENICE_BENCH_EXE", Path(fenice.exe()).with_name("bench_cellflux.exe")))
    out = {"exe": str(exe), "args": args.bench_args, "repeat": args.repeat, "runs": {}}
    for rep in range(args.repeat):   # interleaved: thread counts alternate within a repeat
        for threads in args.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(threads))
            r = subprocess.run([str(exe), *args.bench_args], capture_output=True, text=True, env=env)
            if r.returncode:
                sys.exit(r.stdout + r.stderr)
            out["runs"].setdefault(str(threads), []).append(r.stdout)
            print(f"repeat {rep} threads {threads} done", flush=True)
    (OUT / "bench").mkdir(exist_ok=True)
    with open(OUT / "bench" / "bench_raw.json", "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run", "estimate", "halo", "cost", "bench"])
    ap.add_argument("--re", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--threads", type=int, nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tg-stride", type=int, default=2, help="every n-th Taylor-Green snapshot")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--airfoil", help="halo: a 2-D AMR dump with leaf_meta (golden.py's airfoil_720_t1.npz) as the 2-D partner")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--bench-args", nargs="+", default=["768", "384", "20"])
    args = ap.parse_args()
    if args.stage == "run":
        args.threads = (args.threads or [16])[0]
        args.re = [r for r in args.re if r != 20]
        return cmd_run(args)
    if args.stage == "cost":
        args.threads = args.threads or [1, 8, 16]
    if args.stage == "bench":
        args.threads = args.threads or [1, 8]
    return {"estimate": cmd_estimate, "halo": cmd_halo, "cost": cmd_cost, "bench": cmd_bench}[args.stage](args)


if __name__ == "__main__":
    main()
