"""Norm-based Trotter step-count bounds for any product formula.

simcount's ``ana`` and ``min`` bounds (``bounds.py``; Childs, Maslov, Nam,
Ross, Su 2018, SI App. F, Props. F.3/F.4 and Defs. 1/2) are written for
Suzuki's S_2k only. This module states the same argument for an arbitrary
product formula of order p, so the extension formulas (M&O, Blanes & Moan,
Morales) get a bound of the same kind. It is our generalisation, not a
published bound; nothing here is used by simcount.

The argument (our reconstruction of App. F's method; the paper omits the
proof of F.4)
-----------------------------------------------------------------------
One step of size tau is a product of exponentials exp(-i w_j tau c_j P_j).
Let s = sum_j |w_j tau c_j| be its *norm budget*. Expanding the product in a
Taylor series, the terms of total degree >= p+1 have norm at most
s^(p+1)/(p+1)! * e^s (sum_{k>p} s^k/k! <= s^(p+1)/(p+1)! sum_j s^j/j!). The
same holds for exp(-i tau H) with ||H|| tau <= sum_j |c_j| tau <= s (each
term's weights sum to 1, so s is at least the 1-norm of H times tau). An
order-p formula matches exp(-i tau H) up to degree p, so

    ||S(tau) - exp(-i tau H)|| <= c_p s^(p+1) e^s,   c_p = 2/(p+1)!.     (*)

Over r steps (triangle inequality, unitaries), with s = L t / r:

    r c_p (L t)^(p+1) / r^(p+1) e^(L t / r) <= eps.

``min`` is the smallest r satisfying this (searched as simcount does);
``ana`` is the closed form obtained by imposing r >= L t, so e^(L t/r) <= e:
r = ceil(max(L t, L t (e c_p L t / eps)^(1/p))).

Two choices, both explicit arguments
------------------------------------
budget -- how L (the budget rate, s = L tau) is measured:
  "count": L = (number of sweeps) * m * Lambda. This is the counting used by
      simcount for Suzuki: S_2k has 2 * 5^(k-1) sweeps through the m terms
      (Berry et al. 2007, Lemma 1: |p_k|, |1 - 4 p_k| < 1), each exponential
      bounded by Lambda tau; Lie-Trotter has 1. For a ramp formula with q
      cycles there are 2q sweeps. Valid only if every |w_j| <= 1, which is
      checked (via exact <= count) and raises otherwise. With Suzuki this
      reproduces ``bounds.minimised`` / ``bounds.analytic`` exactly.
  "exact": L = sum_j |w_j c_j| over the unmerged schedule of one unit step.
      The tightest budget the same argument allows; it uses the real weights
      and the real coefficients (so it depends on the seed's fields). Not
      simcount's.
constant -- c_p:
  "childs": 1 for p = 1, 1/3 for p >= 2, the constants simcount uses
      (Childs et al. 2018 Prop. F.3 and F.4). For p >= 2 this is >= 2/(p+1)!,
      so it is valid but loose for p > 2.
  "taylor": 2/(p+1)! from (*). Equals "childs" at p = 1, 2.

Conventions: hbar = 1, t is the total simulation time, eps the spectral-norm
error of the whole evolution, Lambda = max_j |c_j| (``H.max_coeff``), m the
number of terms (``H.n_terms``).
"""

from __future__ import annotations

import math

from .bounds import E, _exp, _safe, exponential_search
from .hamiltonian import Hamiltonian
from .weighted import FormulaSpec

BUDGETS = ("count", "exact")
CONSTANTS = ("childs", "taylor")
KINDS = ("min", "ana")


def sweeps(spec: FormulaSpec) -> int:
    """Number of passes through the m terms counted per step: 1 for
    Lie-Trotter, 2q (q forward + q backward ramps) for a symmetric formula.
    For Suzuki S_2k, q = 5^(k-1), giving simcount's 2 * 5^(k-1)."""
    if spec.order == 1:
        if spec.q != 1:
            raise ValueError(f"first-order formula with q = {spec.q}: sweep count undefined")
        return 1
    if spec.order % 2:
        raise ValueError(f"odd order {spec.order} > 1: not a symmetric formula, sweep count undefined")
    return 2 * spec.q


_RATE_CACHE: dict[tuple, float] = {}


def exact_rate(spec: FormulaSpec, H: Hamiltonian) -> float:
    """sum_j |w_j c_j| over one unmerged step of unit size: the norm budget
    per unit time (s = exact_rate * tau). Cached per (formula, terms): a ramp
    schedule at n = 500 takes ~10 s to build (stage_groups is quadratic in m)."""
    # keyed by the spec object itself (its schedule function compares by identity, and
    # the dict keeps it alive so the identity cannot be reused), never by its name
    key = (spec, tuple((t.coeff, t.word) for t in H.terms))
    if key not in _RATE_CACHE:
        _RATE_CACHE[key] = math.fsum(abs(time * H.terms[j].coeff) for j, time in spec.schedule(H, 1.0))
    return _RATE_CACHE[key]


def count_rate(spec: FormulaSpec, H: Hamiltonian) -> float:
    """sweeps * m * Lambda, after checking it really bounds the exact budget
    (i.e. the counting argument applies to this formula)."""
    rate = sweeps(spec) * H.n_terms * H.max_coeff
    ex = exact_rate(spec, H)
    if ex > rate * (1 + 1e-12):
        raise ValueError(f"{spec.name}: exact budget {ex} exceeds the counted {rate}; "
                         "some |weight| > 1, the counting argument does not apply")
    return rate


def c_p(order: int, constant: str) -> float:
    """The constant in (*) for a whole-evolution bound."""
    if constant not in CONSTANTS:
        raise ValueError(f"constant must be one of {CONSTANTS}, got {constant!r}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if constant == "taylor":
        return 2 / math.factorial(order + 1)
    return 1.0 if order == 1 else 1 / 3


def _lmt(spec: FormulaSpec, H: Hamiltonian, t: float, budget: str, kind: str) -> float:
    """L t, with the 'count' arithmetic written in simcount's operand order
    (PF.hs via bounds.py) so Suzuki values agree to the last bit."""
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}, got {budget!r}")
    if budget == "exact":
        return exact_rate(spec, H) * t
    count_rate(spec, H)  # raises if the counting argument does not apply
    m, l = H.n_terms, H.max_coeff
    if spec.order == 1:
        return m * t * l if kind == "ana" else l * t * m
    q = float(spec.q)  # 5.0 ** (k - 1) for Suzuki, exact in a double
    return 2 * m * t * l * q if kind == "ana" else 2 * l * t * m * q


def steps(spec: FormulaSpec, H: Hamiltonian, t: float, eps: float, *, kind: str = "min",
          budget: str = "count", constant: str = "childs") -> int:
    """Trotter steps r so that the norm bound (*) on the whole evolution is <= eps.

    kind "min": smallest r satisfying r c_p (Lt/r)^(p+1) e^(Lt/r) <= eps
    (simcount's exponential + binary search). kind "ana": the closed form.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not (t > 0 and 0 < eps < 1):
        raise ValueError(f"need t > 0 and 0 < eps < 1, got t={t}, eps={eps}")
    p = spec.order
    c = c_p(p, constant)
    lmt = _lmt(spec, H, t, budget, kind)

    if kind == "ana":
        if p == 1:
            b = (E * lmt**2) / eps if constant == "childs" else (E * c * lmt**2) / eps
            return math.ceil(max(lmt, b))
        if constant == "childs":
            b = ((E * lmt) / (3 * eps)) ** (1 / p)  # simcount's expression, c = 1/3
        else:
            b = ((E * c * lmt) / eps) ** (1 / p)
        return math.ceil(max(lmt, lmt * b))

    if p == 1:
        def cond(r):
            y = (lmt**2) / r if constant == "childs" else c * (lmt**2) / r
            return y * _exp(lmt / r) <= eps
    else:
        def cond(r):
            if constant == "childs":
                y = (lmt ** (p + 1)) / (3 * (r**p))  # simcount's expression
            else:
                y = c * (lmt ** (p + 1)) / (r**p)
            return y * _exp(lmt / r) <= eps
    return exponential_search(_safe(cond))


def step_error_bound(spec: FormulaSpec, H: Hamiltonian, tau: float, *, budget: str = "count",
                     constant: str = "childs") -> float:
    """Right-hand side of (*) for one step of size tau: c_p s^(p+1) e^s."""
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}, got {budget!r}")
    rate = count_rate(spec, H) if budget == "count" else exact_rate(spec, H)
    s = rate * tau
    return c_p(spec.order, constant) * s ** (spec.order + 1) * _exp(s)
