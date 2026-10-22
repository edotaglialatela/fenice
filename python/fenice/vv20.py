"""Validation uncertainty after ASME V&V 20, as presented in Roache (2009), chapter 11.

A validation compares the simulation value S of a quantity with the data value D. The comparison error is

    E = S - D                                                                  (11.4.1)

and it is the sum of the modelling error and of the errors of the numerics, of the input parameters and of the
data: E = d_model + d_num + d_input - d_D. The validation uncertainty combines the uncertainties of the three last
terms, if they are independent,

    U_val = sqrt(U_num^2 + U_input^2 + U_D^2),                                 (11.9.1b)

and the modelling error then falls in the interval E +- U_val, with about 95 % coverage if every term is a 95 %
(expanded) uncertainty. For the numerics U_num,95 is the Grid Convergence Index of the fine grid itself (11.9.1.4,
see fenice.convergence). A data uncertainty given as a standard uncertainty u_D is expanded with the coverage
factor k = 2 (U_D = k u_D); a range [lo, hi] of values with no stated distribution is a rectangular distribution of
half-width A, whose standard uncertainty is A / sqrt(3) (11.8.2).

The reading of the result (11.10): if |E| <= U_val the modelling error is of the order of, or below, the noise level
set by the numerical, input and data uncertainties and tuning the model would not be justified; if |E| >> U_val the
modelling error is about E and the model is to blame. What counts as '>>' is a choice, set here to
|E| >= `strong` U_val (3 by default); in between, the result is reported as indeterminate. A quantity without a
valid GCI has no U_num: it is *not assessable*, and no other estimate is substituted.

    >>> v = Validation("u", S=-0.3277, D=-0.3273, U_num=0.0019)
    >>> round(v.E, 4), round(v.U_val, 4), v.verdict
    (-0.0004, 0.0019, 'within the noise level')

Run ``python -m fenice.vv20`` for the self-test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def uniform_standard_uncertainty(half_width: float) -> float:
    """Standard uncertainty of a rectangular distribution of half-width A: A / sqrt(3)."""
    return half_width / math.sqrt(3.0)


def triangular_standard_uncertainty(half_width: float) -> float:
    """Standard uncertainty of a symmetric triangular distribution of half-width A: A / sqrt(6)."""
    return half_width / math.sqrt(6.0)


def from_range(lo: float, hi: float, shape: str = "uniform") -> tuple[float, float]:
    """(D, u_D) of a data value known as a range [lo, hi]: the midpoint and the standard uncertainty."""
    a = 0.5 * (hi - lo)
    u = uniform_standard_uncertainty(a) if shape == "uniform" else triangular_standard_uncertainty(a)
    return 0.5 * (lo + hi), u


@dataclass
class Validation:
    """One validation comparison; all uncertainties are absolute and 95 % expanded unless stated.

    U_num: numerical uncertainty (the GCI of the fine grid times |S|), None if it cannot be assessed;
    U_input: input-parameter uncertainty (0: the strong-model assumption, every input exactly known);
    u_D: data *standard* uncertainty (expanded with `k`); U_D95 overrides it with a stated 95 % uncertainty."""
    name: str
    S: float
    D: float
    U_num: float | None = None
    U_input: float = 0.0
    u_D: float = 0.0
    U_D95: float | None = None
    k: float = 2.0
    strong: float = 3.0
    notes: list = field(default_factory=list)

    @property
    def E(self) -> float:
        return self.S - self.D

    @property
    def U_D(self) -> float:
        return self.U_D95 if self.U_D95 is not None else self.k * self.u_D

    @property
    def assessable(self) -> bool:
        return self.U_num is not None

    @property
    def U_val(self) -> float | None:
        if not self.assessable:
            return None
        return math.sqrt(self.U_num ** 2 + self.U_input ** 2 + self.U_D ** 2)

    @property
    def ratio(self) -> float | None:
        """|E| / U_val."""
        return None if not self.assessable or self.U_val == 0.0 else abs(self.E) / self.U_val

    @property
    def interval(self) -> tuple[float, float] | None:
        """The interval E +- U_val of the modelling error."""
        return None if not self.assessable else (self.E - self.U_val, self.E + self.U_val)

    @property
    def verdict(self) -> str:
        if not self.assessable:
            return "not assessable (no numerical uncertainty)"
        if abs(self.E) <= self.U_val:
            return "within the noise level"
        if abs(self.E) >= self.strong * self.U_val:
            return "modelling error evident"
        return "indeterminate"


def from_gci(name: str, S: float, D: float, gci, **kw) -> Validation:
    """A comparison whose numerical uncertainty is the fine-grid GCI (a fenice.convergence.GCI of a grid triplet,
    relative to the fine solution S); a triplet without a GCI makes the result 'not assessable'."""
    U_num = None if gci is None or gci.gci_fine is None else abs(gci.gci_fine) * abs(S)
    v = Validation(name, S, D, U_num=U_num, **kw)
    if gci is not None:
        v.notes.extend(gci.notes)
    return v


def selftest():
    # the five measurements of the book's example (11.9.1.1): population standard deviation 0.0894
    vals = [0.9, 0.9, 1.0, 1.1, 1.1]
    mean = sum(vals) / 5
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / 5)
    assert abs(sd - 0.08944) < 1e-4
    D, u = from_range(1.33, 1.35)
    assert abs(D - 1.34) < 1e-12 and abs(u - 0.01 / math.sqrt(3)) < 1e-12
    v = Validation("x", S=1.0, D=0.9, U_num=0.05, u_D=0.02)          # U_D = 0.04: U_val = sqrt(0.05^2 + 0.04^2)
    assert abs(v.U_val - math.sqrt(0.0025 + 0.0016)) < 1e-12
    assert v.verdict == "indeterminate" and abs(v.E - 0.1) < 1e-12   # 0.1 vs 0.064: between 1 and 3
    assert Validation("x", 1.0, 0.99, U_num=0.05).verdict == "within the noise level"
    assert Validation("x", 1.0, 0.5, U_num=0.05).verdict == "modelling error evident"
    assert Validation("x", 1.0, 0.5).verdict.startswith("not assessable")
    assert from_gci("x", 1.0, 0.9, None).U_val is None
    print("fenice.vv20 self-test: OK")


if __name__ == "__main__":
    selftest()
