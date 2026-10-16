"""The paper's checkable claims, each as a function returning plain rows.

Every function here is deterministic: the only inputs are the copied scheme
data, the paper's own parameters, and upstream's recorded seed-42 fields.

Paper parameters used (arXiv:2601.18756v3, Sec. 3.1, not chosen here):
    XXZ model, J = 1, h_i ~ U[-0.1, 0.1] via upstream's seed 42, periodic,
    t = 10;  Fig. 5: L = 6;  Fig. 6: cost q N_t = 500, global grouping.

One choice is not stated in the paper: how N_t is chosen when q does not
divide the cost (Fig. 6).  It is fixed from the figure itself, not assumed:
N_t = round(500 / q) matches every Fig. 6 marker, while floor and ceil miss
by 0.01-0.07 decades (30-500x the digitisation precision) -- see
``tests/test_paper.py::test_fig6_rounding_rule``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from . import bch
from . import evolution as E
from . import heisenberg as Hm
from . import schemes as S
from .digitise import FIG5_FIXTURE, FIG6_FIXTURE
from .reference import FIELDS_FIXTURE

T_PAPER = 10.0  # evolution time t of Figs. 4-6


def paper_fields(L: int) -> list[float]:
    """Upstream's hmag = random for chain length L (seed 42, libstdc++, double).

    libc++ gives the same values to 1 ulp (both are recorded in the fixture).
    """
    data = json.loads(FIELDS_FIXTURE.read_text())
    return data["libstdc++"][str(L)]


def paper_chain(L: int) -> Hm.Chain:
    return Hm.chain(L, "XXZ", paper_fields(L), 1.0, periodic=True)


# ------------------------------------------------------------- theory claims

@dataclass
class ErrRow:
    source: str
    order: int
    q: int
    n: int
    ours_err2: float
    upstream_err2: float

    @property
    def rel(self) -> float:
        return abs(self.ours_err2 - self.upstream_err2) / self.upstream_err2


def published_errors() -> list[ErrRow]:
    """Err_n^2 for every scheme in Recommended_schemes, ours vs upstream, n >= order.

    (n < order are constraint residues; upstream reports ~1e-70 from quad
    arithmetic, double gives ~1e-32; compared separately in the tests.)
    """
    rows = []
    for path, s in S.all_recommended():
        co = bch.error_coefficients(s)
        for n, e2 in s.reference:
            if n >= s.order:
                rows.append(ErrRow(f"{path.parent.name}/{path.name}", s.order, s.q, n,
                                   float(np.sum(co[n + 1] ** 2)), e2))
    return rows


TABLE4 = [  # (label, scheme factory, paper Eff_n, paper x-bar), arXiv v3 Table 4
    ("n=4 Blanes & Moan (q=6)", lambda: S.blanes_moan(6), 10.216, 0.753),
    ("n=4 Highest Eff (q=6)", lambda: S.recommended(4, 6)[0], 10.831, 0.863),
    ("n=4 Recommended (q=6)", S.recommended_q6, 10.530, 0.720),
    ("n=6 Blanes & Moan (q=10)", lambda: S.blanes_moan(10), 0.370, 1.453),
    ("n=6 Highest Eff (q=14)", S.highest_q14, 0.762, 1.256),
    ("n=6 Recommended (q=14)", S.recommended_q14, 0.736, 0.784),
]


def table4() -> list[tuple[str, float, float, float, float]]:
    """(label, ours Eff, paper Eff, ours x-bar, paper x-bar)."""
    out = []
    for label, make, eff, xbar in TABLE4:
        s = make()
        out.append((label, bch.efficiency(s), eff, s.distance(), xbar))
    return out


# --------------------------------------------------------------- Fig. 5 / 6

def fig5_comparison(panel: str) -> list[dict]:
    """Our Delta at every digitised Fig. 5 point of the reproducible schemes.

    Returns one dict per scheme with arrays cost, paper, ours.  Points below
    the plotted range (1.2e-11) are dropped -- they are clipped in the figure.
    """
    data = json.loads(FIG5_FIXTURE.read_text())
    by_q = {s.q: s for s in S.figure5_schemes()}
    ch = paper_chain(6)
    exact = E.ExactEvolution(ch)
    out = []
    for curve in data["curves"]:
        if curve["panel"] != panel or curve["q"] not in by_q:
            continue
        s = by_q[curve["q"]]
        cost, paper, ours = [], [], []
        for c, err in curve["points"]:
            if err < 1.2e-11:
                continue
            nt = round(c / s.q)
            cost.append(s.q * nt)
            paper.append(err)
            ours.append(E.error_at(ch, s, T_PAPER, nt, panel, exact))
        out.append(dict(label=curve["label"], q=s.q, order=s.order, cost=np.array(cost),
                        paper=np.array(paper), ours=np.array(ours)))
    return out


def fig6_comparison(L_values: list[int], rounding: str = "round") -> list[dict]:
    """Our Delta at cost 500 (global grouping) vs the digitised Fig. 6 markers."""
    rnd = {"round": round, "floor": math.floor, "ceil": math.ceil}[rounding]
    data = json.loads(FIG6_FIXTURE.read_text())
    by_q = {s.q: s for s in S.figure5_schemes()}
    rows = []
    for L in L_values:
        ch = paper_chain(L)
        exact = E.ExactEvolution(ch)
        for series in data["series"]:
            if series["q"] not in by_q:
                continue
            s = by_q[series["q"]]
            paper = dict((Lp, e) for Lp, e in series["points"])[L]
            ours = E.error_at(ch, s, T_PAPER, rnd(data["cost"] / s.q), "global", exact)
            rows.append(dict(L=L, label=series["label"], q=s.q, paper=paper, ours=ours))
    return rows


def ranking_at_costs(grouping: str, costs: list[int], L: int = 6) -> list[dict]:
    """Paper Sec. 3.1 claim: the q=14 scheme beats every historical n <= 6
    scheme "at each cost".  For each target cost, every scheme runs at
    N_t = round(cost / q) and its Delta is reported with its actual cost."""
    ch = paper_chain(L)
    exact = E.ExactEvolution(ch)
    rows = []
    for cost in costs:
        row = {"cost": cost}
        for s in S.figure5_schemes():
            nt = max(1, round(cost / s.q))
            row[s.q] = (s.q * nt, E.error_at(ch, s, T_PAPER, nt, grouping, exact))
        rows.append(row)
    return rows


# ------------------------------------------------- Morales et al. reproduction

# Morales et al. (arXiv:2210.15817v2), Table "tab:order8-constant": average
# spectral-norm error constant chi (delta(t) = chi t^9) and their cost score
# M chi^(1/8), for the order-8 formulas we have weights for.
MORALES_TABLE8 = {  # label: (chi, M chi^(1/8))
    "S8m2": (4.8e-9, 11.4),
    "KL8s15": (5.9e-6, 3.33),
    "Y8m10": (4.9e-8, 2.56),
    "Y8m10b": (5.4e-7, 3.45),
}


def random_pair(rng: np.random.Generator, dim: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Two random Hermitian matrices of spectral norm 1.

    ASSUMPTION: Gaussian (GUE-like) entries, then scaled to spectral norm 1
    (unconfirmed). Morales et al. say only "random Hermitian matrices of
    dimension 6x6 and norm 1" (Sec. VI.1) without the distribution.
    """
    out = []
    for _ in range(2):
        X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        Hm = (X + X.conj().T) / 2
        out.append(Hm / np.linalg.norm(Hm, 2))
    return out[0], out[1]


def random_matrix_constant(scheme: S.Scheme, n_samples: int, seed: int, t: float = 1.0,
                           order: int = 8) -> float:
    """Geometric mean over random pairs of chi = ||S(t) - e^{-i t (A+B)}||_2 / t^(order+1).

    ASSUMPTION: t = 1.0 (unconfirmed). Morales et al. check the t^9 scaling
    between two unstated times t1, t2 and take chi = delta / t^9. At t = 1 the
    measured slope is ~9.0 for Y8m10 but ~8.5 for KL8s15, so formulas with
    large weights are not fully in the small-t regime there.
    """
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(n_samples):
        A, B = random_pair(rng)
        Ea, Va = np.linalg.eigh(A)
        Eb, Vb = np.linalg.eigh(B)
        Eh, Vh = np.linalg.eigh(A + B)
        U = np.eye(A.shape[0], dtype=complex)
        for op, w in scheme.stages():
            E, V = (Ea, Va) if op == "A" else (Eb, Vb)
            U = U @ ((V * np.exp(-1j * w * t * E)) @ V.conj().T)
        exact = (Vh * np.exp(-1j * t * Eh)) @ Vh.conj().T
        logs.append(np.log(np.linalg.norm(U - exact, 2) / t ** (order + 1)))
    return float(np.exp(np.mean(logs)))


def morales_reproduction(n_samples: int = 2000, seed: int = 2026) -> list[tuple[str, int, float, float | None]]:
    """(label, M, chi ours, chi paper) for the order-8 formulas, incl. two with no published chi."""
    from . import morales as Mo
    from plsimcount.weighted import suzuki_scheme
    rows = [("S8m2", suzuki_scheme(8)), ("KL8s15", Mo.m7()), ("Y8m10", Mo.y8m10()), ("Y8m10b", Mo.y8m10b()),
            ("Morales m=8 (unpublished chi)", Mo.m8()), ("Malezic q=21 (unpublished chi)", S.recommended(8, 21)[0])]
    out = []
    for label, s in rows:
        chi = random_matrix_constant(s, n_samples, seed)
        out.append((label, s.q, chi, MORALES_TABLE8.get(label, (None,))[0]))
    return out
