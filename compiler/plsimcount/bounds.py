"""Trotter step-count bounds, ported one-to-one from simcount's ``PF.hs``.

Every bound is a pure function ``(hamiltonian, t, eps) -> r`` so that the
choice of bound is orthogonal to the choice of product formula and to the
gate-level compilation. Arithmetic is written in the same order as the
Haskell source so the resulting ``r`` values agree exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .hamiltonian import Hamiltonian

E = math.exp(1)


def _exp(x: float) -> float:
    """``exp`` that saturates to +inf like a Haskell Double instead of raising."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


# ---------------------------------------------------------------- searching

def exponential_search(condition: Callable[[int], bool]) -> int:
    """``exponentialSearch`` from Definitions.hs: find the smallest power of
    two satisfying ``condition``, then binary-search between half of it and
    it. ``condition`` must be monotone in r."""
    ub = 1
    while not condition(ub):
        ub *= 2
    lb = ub // 2

    def binary(lb, ub):
        if ub == lb + 1:
            if condition(lb):
                return lb
            if condition(ub):
                return ub
            raise RuntimeError("no bound found")
        mid = lb + (ub - lb) // 2
        return binary(lb, mid) if condition(mid) else binary(mid, ub)

    return binary(lb, ub)


def _safe(condition):
    """r = 0 evaluates to +inf in Haskell; treat it as failing here."""
    return lambda r: r > 0 and condition(r)


# ---------------------------------------------------------------- analytic

def analytic(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    m, l = H.n_terms, H.max_coeff
    if order == 1:
        a = m * t * l
        b = (E * a**2) / eps
        return math.ceil(max(a, b))
    k = order // 2
    a = 2 * m * t * l * (5 ** (k - 1))
    b = ((E * a) / (3 * eps)) ** (1 / order)
    return math.ceil(max(a, a * b))


# ---------------------------------------------------------------- minimised

def minimised(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    m, l = H.n_terms, H.max_coeff
    if order == 1:
        def cond(r):
            y = ((l * t * m) ** 2) / r
            z = _exp((l * t * m) / r)
            return y * z <= eps
    else:
        k = order // 2
        lmt = 2 * l * t * m * (5.0 ** (k - 1))

        def cond(r):
            y = (lmt ** (order + 1)) / (3 * (r**order))
            z = _exp(lmt / r)
            return y * z <= eps

    return exponential_search(_safe(cond))


# ---------------------------------------------------------------- commutator

def _com1(n):
    return {1: 0, 2: 4}.get(n, 10 * n)


def _com2(n):
    return 194 if n == 3 else 40 * n**2 - 58 * n


def _com4(n):
    return {3: 23073564672, 4: 94192316416, 5: 278878851840}.get(
        n, 1280000000 * n**4 - 7701760000 * n**3 + 23685120000 * n**2 - 30224677632 * n
    )


def commutator(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    """Commutator bounds for the 1D nearest-neighbour Heisenberg model.

    The constants com1/com2/com4 were computed by Childs et al. for that
    model with simcount's term ordering; they are *not* valid for arbitrary
    Hamiltonians.
    """
    n, m, l = H.n_qubits, H.n_terms, H.max_coeff
    mlt = m * l * t

    if order == 1:
        def cond(r):
            x = (l * t) ** 2 / r
            y = mlt**3 / (3 * r**2)
            return x * _com1(n) + y * _exp(mlt / r) <= eps
    elif order == 2:
        def cond(r):
            x = (l * t) ** 3 / (r**2)
            y = 4 * mlt**4 / (3 * r**3)
            return x * _com2(n) + y * _exp((2 * mlt) / r) <= eps
    elif order == 4:
        p = 0.414490771794376
        fp1 = 4 * p - 1

        def cond(r):
            x = ((fp1 / 2) * l * t) ** 5 / (120 * r**4)
            y = 2 * (5 * fp1 * mlt) ** 6 / (720 * r**5)
            return x * _com4(n) + y * _exp((5 * fp1 * mlt) / r) <= eps
    else:
        raise ValueError("commutator bound is only defined for orders 1, 2, 4")

    return exponential_search(_safe(cond))


# ---------------------------------------------------------------- empirical

_EMPIRICAL = {
    1: (2417.223485784009, 1.964012729442229),
    2: (39.469590792617986, 1.882643504940215),
    4: (4.034530593241404, 1.554526451252546),
    6: (1.788782171117122, 1.311258229512891),
    8: (1.144579960605601, 1.140694313868143),
}


def empirical(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    """Fitted to numerics for the Heisenberg model with t = n, eps = 1e-3.
    Ignores ``t`` and ``eps`` by construction, as in simcount."""
    if order not in _EMPIRICAL:
        raise ValueError("no empirical bound for this order")
    a, b = _EMPIRICAL[order]
    return math.ceil(a * (H.n_qubits**b))


# ------------------------------------------------------------ commutator scaling

def trotter_error_theory(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    """Instance-specific nested-commutator bounds from Childs et al.,
    *Theory of Trotter Error with Commutator Scaling* (2021). See ``tte.py``.
    Not part of simcount; valid for any Pauli Hamiltonian (orders 1, 2; order
    4 for two or three commuting groups)."""
    from . import tte

    return tte.steps(order, H, t, eps)


BOUNDS = {
    "ana": analytic,
    "min": minimised,
    "com": commutator,
    "emp": empirical,
    "tte": trotter_error_theory,
}


def parse_algo(name: str) -> tuple[int, str]:
    """``'pf4com' -> (4, 'com')``."""
    if not name.startswith("pf"):
        raise ValueError(f"only product-formula algorithms are supported, got {name!r}")
    order, bound = int(name[2:-3]), name[-3:]
    if bound not in BOUNDS:
        raise ValueError(f"unknown bound {bound!r}")
    return order, bound


def steps(algo: str, H: Hamiltonian, t: float, eps: float) -> int:
    order, bound = parse_algo(algo)
    return BOUNDS[bound](order, H, t, eps)
