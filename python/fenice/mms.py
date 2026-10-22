"""Manufactured solutions of the compressible Euler / Navier-Stokes equations (Roache 2009, ch. 3).

A smooth periodic solution U_m(x, y, z) is chosen, pushed through the equations the solver discretises, and the
residual that remains is the source term S that makes U_m an exact (steady) solution:

    dU/dt + div(F - F_v) = S,     S = div(F(U_m) - F_v(U_m)),

with U = [rho, rho u, rho v, rho E, rho w] (the slot order of the solver), the ideal gas rho E = p/(gamma-1) +
rho |u|^2 / 2, the Stokes stress tau_ij = mu (u_i,j + u_j,i) - 2/3 mu delta_ij div u and the Fourier heat flux
q = -k grad T with T = p / rho and k = mu gamma / (Pr (gamma - 1)) (read off cpp/src/Physics.cpp). The solution is
non-separable (different wave numbers and phases in every variable, the sine and cosine arguments mix x, y, z), so
that every derivative term of the truncation error is exercised: a pure product solution would let cross terms
vanish. The source is emitted as TinyExpr strings (only + - * / pow sin cos and numbers), the same expressions the
solver evaluates at the cell centres (source_rho, source_mx, ... in the case file).

    from fenice import mms
    sol = mms.Solution(dim=2, viscous=True)
    sol.write_cases(directory)            # cases/manufactured_solution*/case.toml
    rho, u, v, w, p = sol.exact()(x, y, z)   # numpy arrays

Needs sympy (``python prerequisites.py``). ``python -m fenice.mms --check`` verifies that the case files of the
repository are what this generator writes.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import sympy as sp
except ImportError as exc:   # pragma: no cover
    raise ImportError("fenice.mms needs sympy (pip install sympy)") from exc

x, y, z = sp.symbols("x y z", real=True)
SLOTS = ("source_rho", "source_mx", "source_my", "source_E", "source_mz")   # order of the state [rho, mx, my, E, mz]
K = 2 * sp.pi   # wave number of the unit domain [0, 1]^d
U_WALL = "0.1"   # velocity of the moving bottom wall of the "walls" solution


def _r(value: str) -> sp.Rational:
    """An exact decimal (0.2 is the rational 1/5, not the nearest double)."""
    return sp.Rational(value)


@dataclass
class Solution:
    """The manufactured solution and its source terms.

    dim: 2 or 3; viscous: Navier-Stokes (the source contains the viscous terms) or Euler; mu, pr, gamma: the
    constants of the equations (they are written into the case file and must match the solver's parameters).
    kind: "periodic" (the default), or a 2-D solution with non-periodic sides whose variables are even or odd
    about the boundary exactly as the solver's ghost cells assume, so that the ghost state IS the extension of
    the solution across the boundary and the boundary condition itself is verified:

    * "walls": x periodic; bottom MovingWall (u = U_WALL, v = 0), top Wall (slip: v = 0). rho, p even about
      y = 0 and y = 1, v odd about both, u - U_WALL odd about y = 0 and u even about y = 1;
    * "outflow": y periodic; left NoSlip, right Transmissive. rho, p even about x = 0 and x = 1, u and v odd about
      x = 0 and even about x = 1 (zero normal gradient).

    The symmetry is checked numerically at construction (check_symmetry)."""
    dim: int = 2
    viscous: bool = True
    mu: float = 0.02
    pr: float = 0.72
    gamma: float = 1.4
    kind: str = "periodic"

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.kind not in ("periodic", "walls", "outflow"):
            raise ValueError("kind must be periodic, walls or outflow")
        if self.kind != "periodic" and self.dim != 2:
            raise ValueError("the non-periodic solutions are 2-D")
        k = K
        if self.kind == "walls":
            c = sp.cos(sp.pi * y)   # even about y = 0 and y = 1
            self.rho = 1 + _r("0.2") * sp.sin(k * x + _r("0.4") + _r("0.5") * c) * (_r("0.7") + _r("0.3") * c)
            self.u = _r(U_WALL) + sp.sin(sp.pi * y / 2) * (_r("0.3") * sp.sin(k * x + _r("0.5") + _r("0.6") * c)
                                                           + _r("0.1") * sp.cos(2 * k * x + _r("0.3") * c))
            self.v = sp.sin(sp.pi * y) * (-_r("0.25") * sp.cos(k * x + _r("0.2") + _r("0.4") * c)
                                          + _r("0.1") * sp.sin(2 * k * x + _r("0.9") - _r("0.3") * c))
            self.w = sp.Integer(0)
            self.p = 1 + _r("0.2") * sp.cos(k * x + _r("0.6") - _r("0.5") * c) + _r("0.1") * c
            self.check_symmetry()
        elif self.kind == "outflow":
            c = sp.cos(sp.pi * x)   # even about x = 0 and x = 1
            s2 = sp.sin(sp.pi * x / 2)   # odd about x = 0, even about x = 1
            self.rho = 1 + _r("0.2") * sp.sin(k * y + _r("0.4") + _r("0.5") * c) * (_r("0.7") + _r("0.3") * c)
            self.u = s2 * (_r("0.3") * sp.sin(k * y + _r("0.5") + _r("0.6") * c) + _r("0.1") * sp.cos(2 * k * y + _r("0.3") * c))
            self.v = s2 * (-_r("0.25") * sp.cos(k * y + _r("0.2") + _r("0.4") * c)
                           + _r("0.1") * sp.sin(2 * k * y + _r("0.9") - _r("0.3") * c))
            self.w = sp.Integer(0)
            self.p = 1 + _r("0.2") * sp.cos(k * y + _r("0.6") - _r("0.5") * c) + _r("0.1") * c
            self.check_symmetry()
        elif self.dim == 2:
            self.rho = 1 + _r("0.2") * sp.sin(k * x + _r("0.4")) * sp.cos(2 * k * y - _r("0.3"))
            self.u = _r("0.3") * sp.sin(k * (x + y) + _r("0.5")) + _r("0.1") * sp.cos(2 * k * x)
            self.v = -_r("0.25") * sp.cos(k * (x - y) + _r("0.2")) + _r("0.1") * sp.sin(2 * k * y + _r("0.9"))
            self.w = sp.Integer(0)
            self.p = 1 + _r("0.2") * sp.cos(k * x - 2 * k * y + _r("0.6")) + _r("0.1") * sp.sin(k * y + _r("0.3"))
        else:
            self.rho = (1 + _r("0.2") * sp.sin(k * x + _r("0.4")) * sp.cos(2 * k * y - _r("0.3"))) * (
                1 + _r("0.15") * sp.sin(k * z + _r("0.1")))
            self.u = _r("0.3") * sp.sin(k * (x + y) + _r("0.5")) + _r("0.1") * sp.cos(2 * k * x) \
                + _r("0.1") * sp.cos(k * (x + z) + _r("0.4"))
            self.v = -_r("0.25") * sp.cos(k * (x - y) + _r("0.2")) + _r("0.1") * sp.sin(2 * k * y + _r("0.9")) \
                + _r("0.1") * sp.sin(k * (y - z) + _r("0.2"))
            self.w = _r("0.2") * sp.sin(k * (y + z) + _r("0.3")) + _r("0.1") * sp.cos(k * x - k * z + _r("0.7"))
            self.p = 1 + _r("0.2") * sp.cos(k * x - 2 * k * y + _r("0.6")) + _r("0.1") * sp.sin(k * y + _r("0.3")) \
                + _r("0.1") * sp.cos(k * (x + y + z))

    def check_symmetry(self, tol=1e-12):
        """The parities the boundary conditions of `kind` need, checked at random points (raises AssertionError)."""
        rng = np.random.default_rng(7)
        f = sp.lambdify((x, y), [self.rho, self.u, self.v, self.p], "numpy")
        t, s = rng.random(20), rng.random(20) * 0.4   # tangential coordinate, distance from the boundary

        def ev(a, b):
            return [np.broadcast_to(np.asarray(q, float), t.shape) for q in f(a, b)]

        if self.kind == "walls":
            lo, mid, hi, hi_m = ev(t, -s), ev(t, s), ev(t, 1 + s), ev(t, 1 - s)
            checks = {"rho even at y=0": lo[0] - mid[0], "p even at y=0": lo[3] - mid[3], "v odd at y=0": lo[2] + mid[2],
                      "u - U_WALL odd at y=0": (lo[1] - float(U_WALL)) + (mid[1] - float(U_WALL)),
                      "rho even at y=1": hi[0] - hi_m[0], "p even at y=1": hi[3] - hi_m[3], "v odd at y=1": hi[2] + hi_m[2],
                      "u even at y=1": hi[1] - hi_m[1]}
        else:
            lo, mid, hi, hi_m = ev(-s, t), ev(s, t), ev(1 + s, t), ev(1 - s, t)
            checks = {"rho even at x=0": lo[0] - mid[0], "p even at x=0": lo[3] - mid[3], "u odd at x=0": lo[1] + mid[1],
                      "v odd at x=0": lo[2] + mid[2], "rho even at x=1": hi[0] - hi_m[0], "p even at x=1": hi[3] - hi_m[3],
                      "u even at x=1": hi[1] - hi_m[1], "v even at x=1": hi[2] - hi_m[2]}
        for what, d in checks.items():
            assert np.max(np.abs(d)) < tol, f"manufactured solution '{self.kind}' is not symmetric: {what} ({np.max(np.abs(d)):.2e})"

    # ------------------------------------------------------------------ the equations
    def conservative(self):
        """[rho, rho u, rho v, rho E, rho w]."""
        rho, u, v, w, p = self.rho, self.u, self.v, self.w, self.p
        E = p / (sp.Float(self.gamma) - 1) + rho * (u ** 2 + v ** 2 + w ** 2) / 2
        return [rho, rho * u, rho * v, E, rho * w]

    def source(self) -> dict[str, sp.Expr]:
        """S = div(F - F_v) of the manufactured solution, one expression per slot (SLOTS)."""
        rho, u, v, w, p = self.rho, self.u, self.v, self.w, self.p
        U = self.conservative()
        E = U[3]
        vel = [u, v, w]
        coords = [x, y, z][:self.dim]
        # convective fluxes F_d = [rho u_d, rho u u_d + p delta, rho v u_d + p delta, (E + p) u_d, rho w u_d + p delta]
        flux = []
        for d in range(self.dim):
            ud = vel[d]
            f = [rho * ud, rho * u * ud + (p if d == 0 else 0), rho * v * ud + (p if d == 1 else 0),
                 (E + p) * ud, rho * w * ud + (p if d == 2 else 0)]
            flux.append(f)
        if self.viscous:
            mu = sp.Float(self.mu)
            gamma = sp.Float(self.gamma)
            k_cond = mu * gamma / (sp.Float(self.pr) * (gamma - 1))
            T = p / rho
            div = sum(sp.diff(vel[i], coords[i]) for i in range(self.dim))
            for d in range(self.dim):
                # viscous flux through the face normal to d: [0, tau_xd, tau_yd, u_i tau_id + k dT/dx_d, tau_zd]
                tau = [mu * (sp.diff(vel[i], coords[d]) + (sp.diff(vel[d], coords[i]) if i < self.dim else 0))
                       - (sp.Rational(2, 3) * mu * div if i == d else 0) for i in range(3)]
                fv = [0, tau[0], tau[1], sum(vel[i] * tau[i] for i in range(3)) + k_cond * sp.diff(T, coords[d]), tau[2]]
                flux[d] = [flux[d][q] - fv[q] for q in range(5)]
        S = [sum(sp.diff(flux[d][q], coords[d]) for d in range(self.dim)) for q in range(5)]
        return dict(zip(SLOTS, S))

    # ------------------------------------------------------------------ evaluation
    def exact(self):
        """f(x, y, z) -> (rho, u, v, w, p) as numpy arrays (z may be 0 in 2-D)."""
        f = sp.lambdify((x, y, z), [self.rho, self.u, self.v, self.w, self.p], "numpy")
        return lambda X, Y, Z=0.0: tuple(np.broadcast_to(np.asarray(a, float), np.broadcast(X, Y, Z).shape)
                                         for a in f(X, Y, Z))

    def source_functions(self):
        """{slot: f(x, y, z)} of the source terms, for checking what the solver computes."""
        src = self.source()
        return {k: sp.lambdify((x, y, z), e, "numpy") for k, e in src.items()}

    # ------------------------------------------------------------------ case files
    def primitive_strings(self) -> dict[str, str]:
        out = {"rho": tinyexpr(self.rho), "u": tinyexpr(self.u), "v": tinyexpr(self.v), "p": tinyexpr(self.p)}
        if self.dim == 3:
            out["w"] = tinyexpr(self.w)
        return out

    def source_strings(self) -> dict[str, str]:
        return {k: tinyexpr(e) for k, e in self.source().items() if not (self.dim == 2 and k == "source_mz")}

    def name(self) -> str:
        return ("Manufactured Solution" + {"periodic": "", "walls": " Walls", "outflow": " Outflow"}[self.kind] +
                ("" if self.viscous else " Euler") + (" 3D" if self.dim == 3 else ""))

    def kind_label(self) -> str:
        return {"periodic": "periodic", "walls": "moving and slip walls", "outflow": "no-slip wall and outflow"}[self.kind]

    def boundaries(self) -> dict[str, str]:
        """bc_left ... bc_top of the case file."""
        if self.kind == "walls":
            return {"bc_left": "periodic", "bc_right": "periodic", "bc_bottom": "moving_wall", "bc_top": "wall"}
        if self.kind == "outflow":
            return {"bc_left": "no_slip", "bc_right": "transmissive", "bc_bottom": "periodic", "bc_top": "periodic"}
        out = {k: "periodic" for k in ("bc_left", "bc_right", "bc_bottom", "bc_top")}
        if self.dim == 3:
            out.update(bc_front="periodic", bc_back="periodic")
        return out

    def interior_mask(self, n, margin=2):
        """Boolean n x n array (2-D), False within `margin` cells of a non-periodic boundary."""
        m = np.ones((n, n), bool)
        if self.kind == "walls":
            m[:margin, :] = m[-margin:, :] = False
        elif self.kind == "outflow":
            m[:, :margin] = m[:, -margin:] = False
        return m

    def folder(self) -> str:
        return self.name().lower().replace(" ", "_")

    def case_toml(self) -> str:
        """The text of cases/<folder>/case.toml."""
        order = {("periodic", 2, True): 100, ("periodic", 2, False): 101, ("periodic", 3, True): 102,
                 ("walls", 2, True): 103, ("outflow", 2, True): 104}.get((self.kind, self.dim, self.viscous), 105)
        n = 32 if self.dim == 2 else 16
        lines = [
            f"# Manufactured solution of the {'Navier-Stokes' if self.viscous else 'Euler'} equations, {self.dim}-D, "
            + {"periodic": "periodic unit", "walls": "moving wall below, slip wall above, periodic in x,",
               "outflow": "no-slip wall on the left, transmissive on the right, periodic in y,"}[self.kind],
            "# domain (Roache 2009, ch. 3). GENERATED by `python -m fenice.mms --write`: do not edit by hand. The source",
            "# terms S = div(F - F_v) of the exact solution below are evaluated at the cell centres, so the run should",
            "# stay close to the initial state; the error against it falls at the order of the scheme.",
            "[case]",
            f'name = "{self.name()}"',
            f'description = "Code verification by manufactured solution, {self.kind_label()}, source terms"',
            f"order = {order}",
            f"requires_viscous = {'true' if self.viscous else 'false'}",
            *(["requires_3d = true"] if self.dim == 3 else []),
            "",
            "[mesh]",
            "Lx = 1.0",
            "Ly = 1.0",
            *(["Lz = 1.0"] if self.dim == 3 else []),
            f"nx = {n}",
            f"ny = {n}",
            *([f"nz = {n}"] if self.dim == 3 else []),
            "",
            "[thermo]",
            f"gamma = {self.gamma}",
            "",
            "[physics]",
            f"use_viscous = {'true' if self.viscous else 'false'}",
            f"mu = {self.mu}",
            f"pr = {self.pr}",
            "",
            "[boundaries]",
            *[f'{k} = "{v}"' for k, v in self.boundaries().items()],
            *([f"u_in = {U_WALL}"] if self.kind == "walls" else []),
            "",
            "[numerics]",
            'riemann_solver = "HLLC"',
            "use_muscl = true",
            'muscl_limiter = "Van Leer"',
            'time_scheme = "SSP Runge-Kutta 2"',
            "cfl = 0.5",
            "",
            "[run]",
            "t_end = 0.5",
            "save_interval = 100000",
            "",
            "[source]",
            *[f'{k} = "{v}"' for k, v in self.source_strings().items()],
            "",
            "[initial]",
            "state = { " + ", ".join(f'{k} = "{v}"' for k, v in self.primitive_strings().items()) + " }",
            "",
        ]
        return "\n".join(lines)

    def write_cases(self, cases_dir) -> Path:
        d = Path(cases_dir) / self.folder()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "case.toml"
        path.write_bytes(self.case_toml().encode("utf8"))   # bytes: always LF, whatever the platform
        return path


# ---------------------------------------------------------------------------- TinyExpr
def tinyexpr(e: sp.Expr) -> str:
    """A sympy expression as a TinyExpr string: numbers, x y z, + - * /, pow(a, b), sin, cos, exp, sqrt. Every
    operand is parenthesised, so TinyExpr's precedence rules cannot matter."""
    if e.is_Number:
        v = float(e)
        s = repr(v)
        return f"({s})" if v < 0 else s
    if e is x or e is y or e is z:
        return str(e)
    if e == sp.pi:
        return repr(float(sp.pi))
    if e.is_Add:
        return "(" + "+".join(tinyexpr(a) for a in e.args) + ")"
    if e.is_Mul:
        return "(" + "*".join(tinyexpr(a) for a in e.args) + ")"
    if e.is_Pow:
        b, p = e.args
        if p == sp.Rational(1, 2):
            return f"sqrt({tinyexpr(b)})"
        return f"pow({tinyexpr(b)},{tinyexpr(sp.Float(p))})"
    if isinstance(e, (sp.sin, sp.cos, sp.exp)):
        return f"{type(e).__name__}({tinyexpr(e.args[0])})"
    raise ValueError(f"cannot write {e} for TinyExpr")


def default_solutions() -> list[Solution]:
    """The three cases of the repository."""
    return [Solution(2, True), Solution(2, False), Solution(3, True), Solution(kind="walls"), Solution(kind="outflow")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the case files into the cases/ folder")
    ap.add_argument("--check", action="store_true", help="check that the case files are what the generator writes")
    ap.add_argument("--cases", default=None, help="the cases folder (default: <repository>/cases)")
    args = ap.parse_args(argv)
    from .solver import ROOT
    cases = Path(args.cases) if args.cases else ROOT / "cases"
    bad = 0
    for sol in default_solutions():
        text = sol.case_toml().encode("utf8")
        path = cases / sol.folder() / "case.toml"
        n = sum(len(v) for v in sol.source_strings().values())
        if args.write:
            sol.write_cases(cases)
            print(f"wrote {path} ({n} characters of source terms)")
        elif args.check:
            ok = path.exists() and path.read_bytes().replace(b"\r\n", b"\n") == text   # git may check out CRLF
            print(f"{'OK ' if ok else 'DIFFERENT'} {path}")
            bad += 0 if ok else 1
        else:
            print(f"{sol.name():30s} source terms: {n} characters")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
