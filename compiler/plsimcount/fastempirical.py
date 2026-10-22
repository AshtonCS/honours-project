"""Brute-force Trotter numbers, fast enough for n = 10 at order 8.

The smallest r from which the error stays within eps,

    || S(t/r')^r' - exp(-iHt) ||_2  <=  eps   for every r' >= r   (spectral norm)

checked up to r + ``window`` (see "Definition of r" below), for one specific
Hamiltonian instance. It is exact for that instance (the
error is computed, not estimated), and it is not a bound: it says nothing
about other fields or sizes. ``empirical.py`` is kept as the slow reference;
the tests check that both return the same r.

What makes it fast (none of it changes the unitary; see the tests):

1. Pauli action. exp(-i a P) M = cos(a) M - i sin(a) P M, and P M is a row
   permutation with signs: O(N^2) per exponential instead of a dense expm
   plus a matrix product, O(N^3). Measured 64x at n = 10.
2. Diagonal blocks. A commuting block whose terms are all Z/I (simcount's
   ZZ bonds + fields) is one diagonal phase, applied in one pass.
3. Palindrome halving. A symmetric formula's step is F_1 ... F_k M F_k ... F_1
   in commuting blocks. If every Pauli word has an even number of Ys, each
   factor is a symmetric matrix, so one half is the transpose of the
   other: S = V^T M V. Checked, not assumed; falls back to the full product.
4. Order-predicted search. In the scaling region the error falls like r^-p,
   so two evaluations predict the boundary and a check at r and r - 1 pins
   it. Far fewer step builds than doubling-then-bisection.

The result records the errors at r and r - 1 so a reader can see how close
the boundary was (rounding can only matter when one of them is within ~1e-13
of eps).

Definition of r (chosen by the author, 2026-10-21). Before the error reaches
its r^-p regime it need not fall monotonically: for M&O q = 14 on simcount's
n = 6 seed-1 instance at eps = 1e-3 the errors at r = 13..17 are 1.37e-3,
9.91e-4, 1.55e-3, 8.85e-4, 2.29e-4. "Smallest passing r" would give 14, a
dip that fails again at 15; this module gives 16, the smallest r after which
every larger r passes. "Every larger r" cannot be checked to infinity: it is
checked for r + 1 .. r + ``window``, and beyond that rests on the error being
monotone in the r^-p regime. ``empirical.steps`` (the slow reference) is a
plain bisection and makes no such check; the two agree whenever the error is
monotone near the boundary.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .hamiltonian import Hamiltonian
from .weighted import Schedule, commuting_blocks

# The scaling-region test: below this error the r^-p law is used to predict.
# 0.5 is far above eps (1e-3) and well below the ~2 where the spectral error
# saturates; it only steers the search, never the answer.
SCALING_REGION = 0.5

# How far above the returned r every r must also pass (see "Definition of r").
# ASSUMPTION: 10, the window used in the 2026-10-21 audit of pf-thesis, which
# found one non-monotone case (failure 1 step above a dip); no case is known
# that a larger window would change (unconfirmed).
WINDOW = 10


class _Term:
    """One Pauli term, precomputed for fast application on N x N matrices."""

    def __init__(self, word: str, coeff: float):
        n = len(word)
        N = 2 ** n
        y = np.arange(N)
        flip = 0
        phase = np.ones(N, dtype=complex)
        self.n_y = 0
        for q, p in enumerate(word):  # q = 0 is the leftmost kron factor (most significant bit)
            bit = 1 << (n - 1 - q)
            set_ = (y & bit) != 0
            if p == "X":
                flip |= bit
            elif p == "Y":
                flip |= bit
                phase *= np.where(set_, 1j, -1j)
                self.n_y += 1
            elif p == "Z":
                phase *= np.where(set_, -1.0, 1.0)
        self.coeff = coeff
        self.perm = y ^ flip
        self.phase = phase
        self.diagonal = flip == 0  # Z/I only: P is diag(phase), phase = +-1

    def exp_apply(self, angle: float, M: np.ndarray) -> np.ndarray:
        """exp(-i angle P) M."""
        return np.cos(angle) * M - 1j * np.sin(angle) * (self.phase[:, None] * M[self.perm])


class FastEvolution:
    """Exact and Trotterised unitaries for one Hamiltonian."""

    def __init__(self, H: Hamiltonian):
        self.H = H
        self.N = 2 ** H.n_qubits
        self.terms = [_Term(t.word, t.coeff) for t in H.terms]
        self.all_symmetric = all(tm.n_y % 2 == 0 for tm in self.terms)
        E, V = np.linalg.eigh(H.matrix())
        self._E, self._V = E, V

    # -------------------------------------------------------------- exact
    def exact(self, t: float) -> np.ndarray:
        return (self._V * np.exp(-1j * self._E * t)) @ self._V.conj().T

    # -------------------------------------------------------------- blocks
    def _apply_block(self, block: dict[int, float], M: np.ndarray) -> np.ndarray:
        if all(self.terms[j].diagonal for j in block):
            ang = np.zeros(self.N)
            for j, time in block.items():
                tm = self.terms[j]
                ang += time * tm.coeff * tm.phase.real
            return np.exp(-1j * ang)[:, None] * M
        for j, time in block.items():
            tm = self.terms[j]
            M = tm.exp_apply(time * tm.coeff, M)
        return M

    def _product(self, blocks: list[dict[int, float]], M: np.ndarray | None = None) -> np.ndarray:
        """B_k ... B_2 B_1 @ M for blocks [B_1, ..., B_k]: the first block is
        applied first, as in a circuit (``empirical.Evolution.step`` and
        ``circuit.build_step`` use the same convention)."""
        if M is None:
            M = np.eye(self.N, dtype=complex)
        for b in blocks:
            M = self._apply_block(b, M)
        return M

    @staticmethod
    def _is_palindrome(blocks: list[dict[int, float]]) -> bool:
        k = len(blocks)
        for i in range(k // 2):
            a, b = blocks[i], blocks[k - 1 - i]
            if a.keys() != b.keys():
                return False
            for j in a:
                if not math.isclose(a[j], b[j], rel_tol=1e-12, abs_tol=1e-15):
                    return False
        return True

    def step(self, schedule: Schedule, use_palindrome: bool = True) -> np.ndarray:
        """Unitary of one step; the schedule's first entry acts first.

        Palindrome: with blocks B_1..B_K and B_k = B_(K+1-k), the step is
        B_1 ... B_h [mid] B_h ... B_1. Let V = B_h ... B_1 (B_1 acting first).
        If every factor is a symmetric matrix, B_1 ... B_h = V^T, so
        S = V^T [mid] V.
        """
        blocks = commuting_blocks(self.H, schedule)
        if use_palindrome and self.all_symmetric and len(blocks) > 2 and self._is_palindrome(blocks):
            h = len(blocks) // 2
            V = self._product(blocks[:h])
            inner = self._apply_block(blocks[h], V) if len(blocks) % 2 else V
            return V.T @ inner
        return self._product(blocks)

    # -------------------------------------------------------------- error
    def error(self, schedule_fn, t: float, r: int) -> float:
        S = self.step(schedule_fn(t / r))
        return float(np.linalg.norm(np.linalg.matrix_power(S, r) - self.exact(t), 2))


@dataclass
class SearchResult:
    r: int
    err_r: float          # error at r (<= eps)
    err_r_minus_1: float  # error at r - 1 (> eps), inf if r == 1
    window: int           # r + 1 .. r + window were also checked to pass
    dips: list[int]       # passing r below the answer (dips that fail again above)
    evaluations: int
    seconds: float
    trace: list[tuple[int, float]] = field(default_factory=list)


def steps(evolution: FastEvolution, schedule_fn, order: int, t: float, eps: float,
          start: int = 1, max_evals: int = 200, window: int = WINDOW) -> SearchResult:
    """Smallest r such that r, r + 1, ..., r + ``window`` all have error <= eps,
    for the formula ``schedule_fn(H, d)``; r - 1 fails.

    ``start`` is only a first guess (e.g. the previous size's r). The answer
    does not depend on it unless a failure lies more than ``window`` above a
    passing run, which the module docstring discusses.
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    H = evolution.H
    cache: dict[int, float] = {}
    t0 = time.perf_counter()

    def err(r: int) -> float:
        if r not in cache:
            if len(cache) >= max_evals:
                raise RuntimeError(f"search did not converge in {max_evals} evaluations: {sorted(cache.items())}")
            cache[r] = evolution.error(lambda d: schedule_fn(H, d), t, r)
        return cache[r]

    lo, hi = 0, None  # lo: largest failing r found (0 = none), hi: smallest passing r above lo
    r = max(1, int(start))
    while True:
        # Stage 1: bracket a boundary lo | lo + 1 with lo failing and lo + 1 passing.
        e = err(r)
        if e <= eps:
            if r > lo:
                hi = r if hi is None else min(hi, r)
        else:
            if r > lo:
                lo = r
                if hi is not None and hi <= lo:
                    hi = None
        if hi is not None and hi - lo <= 1:
            # Stage 2: the boundary only counts if nothing in the window above fails.
            failing_above = [k for k in range(hi + 1, hi + window + 1) if err(k) > eps]
            if not failing_above:
                break
            lo = max(failing_above)
            passing_above = [k for k, v in cache.items() if k > lo and v <= eps]
            hi = min(passing_above) if passing_above else None
            if hi is not None and hi - lo <= 1:
                continue  # re-check the window above the new boundary
            r = lo + 1
            continue
        if hi is None:  # still failing: go up
            nxt = math.ceil(r * (e / eps) ** (1 / order) * 1.01) if e < SCALING_REGION else 2 * r
            nxt = max(nxt, r + 1, lo + 1)
        elif lo == 0:  # passing, nothing failing known: predict downwards
            nxt = math.floor(r * (e / eps) ** (1 / order)) if e > 0 else r // 2
            nxt = min(max(nxt, 1), hi - 1)
        else:  # bracketed: interpolate log(err) against log(r), else bisect
            e_lo, e_hi = cache[lo], cache[hi]
            if e_lo < SCALING_REGION and e_lo > e_hi > 0:
                s = (math.log(e_lo) - math.log(e_hi)) / (math.log(hi) - math.log(lo))
                nxt = round(lo * (e_lo / eps) ** (1 / s))
            else:
                nxt = (lo + hi) // 2
            nxt = min(max(nxt, lo + 1), hi - 1)
        if nxt in cache:  # no progress from the model: bisect
            nxt = (lo + hi) // 2 if hi is not None else 2 * r
            if nxt in cache:
                nxt = next(k for k in range(lo + 1, hi) if k not in cache)
        r = nxt

    dips = sorted(k for k, v in cache.items() if k < hi and v <= eps)
    return SearchResult(hi, cache[hi], cache.get(hi - 1, math.inf) if hi > 1 else math.inf,
                        window, dips, len(cache), time.perf_counter() - t0, sorted(cache.items()))
