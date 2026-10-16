"""Trotterised vs exact time evolution and the paper's error measure.

Port of upstream ``Code/Time_Evolution/src/Trotter.cpp`` (operator method) and
``Exact.cpp``.

Step operator, paper Eq. (7) with Lambda local stages:

    S(h) = R_up(c_1 h) R_down(d_1 h) R_up(c_2 h) ... R_up(c_q h) R_down(d_q h)

read as a matrix product left to right (upstream builds it with
``step_op *= factor``).  Every factor is exp(-i w h H_stage) for a local stage
of ``heisenberg.Chain``, so S(h) approximates U(h) = exp(-i h H): real time,
hbar = 1.

Two orderings of the stages inside a ramp (paper Sec. 3.1):

* ``"global"`` -- S^(3), Eq. (27): all X bonds, all Y bonds, all Z bonds.
  R_up:   X_0 .. X_{L-1}, Y_0 .. Y_{L-1}, Z_0 .. Z_{L-1}
  R_down: Z_{L-1} .. Z_0, Y_{L-1} .. Y_0, X_{L-1} .. X_0
* ``"local"`` -- S^(3L), Eq. (26): bond by bond.
  R_up:   X_0, Y_0, Z_0, X_1, Y_1, Z_1, ...
  R_down: ..., Z_1, Y_1, X_1, Z_0, Y_0, X_0

Upstream selects these with a boolean ``group``: ``group = true`` runs the
*local* branch and ``group = false`` the *global* one.  The comments in
upstream ``Trotter.h`` and ``InOut.cpp`` say the opposite ("local = 0, global =
1"); the code, not the comment, is what this module and ``reference.py`` follow.

Error, paper Eq. (25):  Delta = || U(t) - S(h)^{N_t} ||_F / sqrt(N),  N = 2^L.
"""

from __future__ import annotations

import numpy as np

from .heisenberg import Chain, LocalStage
from .schemes import Scheme

GROUPINGS = ("global", "local")


def ramp(chain: Chain, grouping: str, direction: str) -> list[LocalStage]:
    """The local stages of one ramp, in matrix-product order (left to right)."""
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")
    dirs = [d for d in "XYZ" if d in chain.stages]
    up: list[LocalStage] = []
    if grouping == "global":
        for d in dirs:
            for stage in chain.stages[d]:
                up.append(stage)
    else:
        for i in range(chain.L):
            for d in dirs:
                up.append(chain.stages[d][i])
    if direction == "up":
        return up
    if direction == "down":
        return up[::-1]
    raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")


def step_factors(chain: Chain, scheme: Scheme, h: float, grouping: str) -> list[tuple[float, LocalStage]]:
    """S(h) as a list of (time weight, stage), leftmost factor first.

    The factor for (w, stage) is prod_{(P, coef) in stage.terms} exp(-i w coef P),
    terms in listed order (upstream keeps that order in both ramp directions;
    the terms of one stage commute, so it does not affect the product).
    """
    c, d = scheme.ramps()
    up = ramp(chain, grouping, "up")
    down = ramp(chain, grouping, "down")
    out: list[tuple[float, LocalStage]] = []
    for i in range(scheme.q):
        out.extend((c[i] * h, s) for s in up)
        out.extend((d[i] * h, s) for s in down)
    return out


def step_operator(chain: Chain, scheme: Scheme, h: float, grouping: str) -> np.ndarray:
    """Dense S(h), 2^L x 2^L.

    S = F_1 F_2 ... F_m is accumulated as F_1 (F_2 (... (F_m I))), i.e. the
    factors are applied to the identity from the left in reverse order.  Same
    product as upstream's left-to-right accumulation; only rounding differs.
    """
    N = 2 ** chain.L
    M = np.eye(N, dtype=complex)
    for w, stage in reversed(step_factors(chain, scheme, h, grouping)):
        for p, coef in reversed(stage.terms):  # F = E_1 E_2 -> apply E_2 first
            M = p.exp_apply(w * coef, M)
    return M


class ExactEvolution:
    """U(t) = exp(-i H t) by diagonalising H once (upstream ``Exact``)."""

    def __init__(self, chain: Chain) -> None:
        H = chain.hamiltonian()
        if not np.allclose(H, H.conj().T, atol=1e-14):
            raise ValueError("H is not Hermitian")
        self.energies, self.vectors = np.linalg.eigh(H)

    def U(self, t: float) -> np.ndarray:
        phases = np.exp(-1j * self.energies * t)
        return (self.vectors * phases) @ self.vectors.conj().T


def frobenius_error(U: np.ndarray, V: np.ndarray) -> float:
    """Paper Eq. (25): ||U - V||_F / sqrt(N).  Ranges over [0, 2] for unitaries."""
    return float(np.linalg.norm(U - V) / np.sqrt(U.shape[0]))


def error_per_step(chain: Chain, scheme: Scheme, t: float, n_steps: int, grouping: str,
                   exact: ExactEvolution | None = None) -> np.ndarray:
    """Delta after each of the n_steps steps of size h = t / n_steps.

    Mirrors upstream ``error_estimate`` with ``method = operator``: entry k is
    the error of S(h)^(k+1) against U(h (k+1)).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    exact = exact or ExactEvolution(chain)
    h = t / n_steps
    S = step_operator(chain, scheme, h, grouping)
    V = np.eye(S.shape[0], dtype=complex)
    out = np.empty(n_steps)
    for k in range(n_steps):
        V = V @ S
        out[k] = frobenius_error(exact.U(h * (k + 1)), V)
    return out


def error_at(chain: Chain, scheme: Scheme, t: float, n_steps: int, grouping: str,
             exact: ExactEvolution | None = None) -> float:
    """Delta at the final time only, using S^n_steps by repeated squaring."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    exact = exact or ExactEvolution(chain)
    S = step_operator(chain, scheme, t / n_steps, grouping)
    return frobenius_error(exact.U(t), np.linalg.matrix_power(S, n_steps))
