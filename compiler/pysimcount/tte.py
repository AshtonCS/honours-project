"""Trotter error bounds from Childs, Su, Tran, Wiebe & Zhu,
*Theory of Trotter Error with Commutator Scaling* (PRX 11, 011020, 2021;
arXiv:1912.08854).

Implemented, with the paper's numbering:

* Proposition 15 -- tight bound for the first-order Lie-Trotter formula,
* Proposition 16 -- tight bound for the second-order Suzuki formula,
* Proposition J.1 / J.2 -- fourth-order Suzuki formula with two / three
  grouped summands (Table 2 coefficients; heuristic, no tightness proof).

All are evaluated for the *actual* Hamiltonian instance by computing the
nested commutators symbolically in the Pauli basis. Norms are bounded the
way the paper does for lattice models: the innermost operand is split into
its individual terms (triangle inequality) and each resulting local
operator's spectral norm is computed exactly on its support. ``exact=True``
computes the full spectral norm instead (small systems only).

Conventions (Section 5.1 of the paper): the product formula is written with
H_1 applied *first* / outermost and H_Gamma last / innermost, which is the
order in which simcount's step applies its commuting groups (XX, YY, ZZ+Z).

Pauli words are held as ``(x, z)`` bit masks meaning the operator
``prod_q X_q^{x_q} Z_q^{z_q}`` (so a Y carries a factor of i into the
coefficient); products and commutators are then XORs and pop-counts.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .hamiltonian import Hamiltonian, Term

Word = tuple[int, int]
Sentence = dict[Word, complex]

# ------------------------------------------------------------------ Pauli algebra

def _term_word(t: Term) -> tuple[Word, complex]:
    x = z = 0
    phase = 1
    for q, p in enumerate(t.word):
        if p == "X":
            x |= 1 << q
        elif p == "Z":
            z |= 1 << q
        elif p == "Y":
            x |= 1 << q
            z |= 1 << q
            phase *= 1j  # Y = i X Z
    return (x, z), t.coeff * phase


def _sentence(terms: list[Term]) -> Sentence:
    s: Sentence = defaultdict(complex)
    for t in terms:
        w, c = _term_word(t)
        s[w] += c
    return dict(s)


def _popcount(v: int) -> int:
    return bin(v).count("1")


def _commute(a: Term, b: Term) -> bool:
    (xa, za), _ = _term_word(a)
    (xb, zb), _ = _term_word(b)
    return (_popcount(za & xb) + _popcount(zb & xa)) % 2 == 0


def commutator(A: Sentence, B: Sentence) -> Sentence:
    """[A, B] for sentences of (x, z) words."""
    out: Sentence = defaultdict(complex)
    for (xa, za), ca in A.items():
        sa = xa | za
        for (xb, zb), cb in B.items():
            if not (sa & (xb | zb)):
                continue  # disjoint support: commutes
            s1 = _popcount(za & xb)
            s2 = _popcount(zb & xa)
            if (s1 + s2) % 2 == 0:
                continue  # commutes
            sign = -1 if s1 % 2 else 1
            out[(xa ^ xb, za ^ zb)] += 2 * sign * ca * cb
    if not out:
        return {}
    # Prune only cancellation noise. The cutoff must be *relative*: an
    # absolute one silently deletes every term of a Hamiltonian with small
    # couplings, which would make the bound too small rather than too large.
    cutoff = 1e-14 * max(abs(c) for c in out.values())
    return {w: c for w, c in out.items() if abs(c) > cutoff}


def _parity(v: np.ndarray) -> np.ndarray:
    """Parity of the pop-count of each entry (entries < 2**32)."""
    v = v.copy()
    for shift in (16, 8, 4, 2, 1):
        v ^= v >> shift
    return v & 1


def _matrix(S: Sentence):
    """Sparse matrix of a sentence on its own support.

    A word X^x Z^z maps |i> to (-1)^{popcount(z & i)} |i ^ x>, so every word
    contributes exactly one entry per column."""
    from scipy.sparse import coo_matrix

    support = 0
    for x, z in S:
        support |= x | z
    qubits = [q for q in range(support.bit_length()) if support >> q & 1]
    pos = {q: k for k, q in enumerate(qubits)}
    dim = 2 ** len(qubits)
    idx = np.arange(dim, dtype=np.int64)
    rows, cols, data = [], [], []
    for (x, z), c in S.items():
        xs = sum(1 << pos[q] for q in qubits if x >> q & 1)
        zs = sum(1 << pos[q] for q in qubits if z >> q & 1)
        rows.append(idx ^ xs)
        cols.append(idx)
        data.append(c * (1 - 2 * _parity(idx & zs)))
    M = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(dim, dim), dtype=complex)
    return M.tocsr()


def norm(S: Sentence) -> float:
    """Spectral norm of a sentence.

    Nested commutators of Hermitian operators are Hermitian or
    anti-Hermitian, so the norm is the largest |eigenvalue| of M or iM.
    Small supports use a dense eigensolver; larger ones try Lanczos first
    and fall back to the dense solver if ARPACK does not converge."""
    if not S:
        return 0.0
    if len(S) == 1:
        return float(abs(next(iter(S.values()))))
    M = _matrix(S)
    scale = abs(M.data).max()
    if abs((M - M.getH()).data).max(initial=0.0) <= 1e-12 * scale:
        Mh = M
    elif abs((M + M.getH()).data).max(initial=0.0) <= 1e-12 * scale:
        Mh = 1j * M
    else:  # not expected for nested commutators of Hermitian terms
        return float(np.linalg.norm(M.toarray(), 2))
    if Mh.shape[0] <= 512:
        return float(np.abs(np.linalg.eigvalsh(Mh.toarray())).max())
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    try:
        return float(abs(eigsh(Mh, k=1, which="LM", return_eigenvectors=False, tol=1e-10)[0]))
    except ArpackNoConvergence:
        return float(np.abs(np.linalg.eigvalsh(Mh.toarray())).max())


def groups(H: Hamiltonian) -> list[list[Term]]:
    """Greedily group consecutive mutually commuting terms. Exponentials
    within a group factorise exactly, so the formula is S_p over these
    grouped summands (the paper's 'X-Y-Z ordering' for the Heisenberg chain)."""
    out: list[list[Term]] = []
    for t in H.terms:
        if out and all(_commute(t, u) for u in out[-1]):
            out[-1].append(t)
        else:
            out.append([t])
    return out


# ------------------------------------------------------------------ nested norms

def _sum(sentences: list[Sentence]) -> Sentence:
    out: Sentence = defaultdict(complex)
    for s in sentences:
        for w, c in s.items():
            out[w] += c
    return dict(out)


def nested_sum(Hs: list[Sentence], inner: tuple[int, int], entries: dict[tuple[int, ...], float],
               exact: bool = False) -> float:
    """sum_e c_e || [H_{e[-1]}, ... [H_{e[0]}, [H_l, H_m]]] ||  for the
    innermost pair ``inner = (l, m)`` and outer index tuples ``e`` listed
    from the inside out. Shared prefixes are computed once.

    Without ``exact`` the innermost pair is split term by term on whichever
    side has fewer terms (norm unchanged by antisymmetry), as in the
    paper's lattice analyses.
    """
    l, m = inner
    A, B = Hs[l], Hs[m]

    def rec(S: Sentence, ents: dict[tuple[int, ...], float]) -> float:
        if not S:
            return 0.0
        total = 0.0
        by_next: dict[int | None, dict] = defaultdict(dict)
        for e, c in ents.items():
            by_next[e[0] if e else None][e[1:] if e else ()] = c
        for g, sub in by_next.items():
            if g is None:
                total += sum(sub.values()) * norm(S)
            else:
                total += rec(commutator(Hs[g], S), sub)
        return total

    if exact:
        return rec(commutator(A, B), entries)

    split, other = (A, B) if len(A) <= len(B) else (B, A)
    return sum(rec(commutator(other, {w: c}), entries) for w, c in split.items())


# ------------------------------------------------------------------ the bounds

def first_order_bound(Hs: list[Sentence], t: float, exact=False) -> float:
    """Proposition 15:  t^2/2 sum_g || [sum_{g'>g} H_g', H_g] ||."""
    total = 0.0
    for g in range(len(Hs)):
        rest = _sum(Hs[g + 1:])
        if rest:
            total += nested_sum(Hs + [rest], (len(Hs), g), {(): 1.0}, exact)
    return t**2 / 2 * total


def second_order_bound(Hs: list[Sentence], t: float, exact=False) -> float:
    """Proposition 16:  t^3/12 sum_g ||[R_g,[R_g,H_g]]|| + t^3/24 sum_g ||[H_g,[H_g,R_g]]||
    with R_g = sum_{g'>g} H_g'."""
    a = b = 0.0
    G = len(Hs)
    for g in range(G):
        rest = _sum(Hs[g + 1:])
        if not rest:
            continue
        ext = Hs + [rest]
        a += nested_sum(ext, (G, g), {(G,): 1.0}, exact)
        b += nested_sum(ext, (g, G), {(g,): 1.0}, exact)
    return t**3 / 12 * a + t**3 / 24 * b


# Proposition J.1: two summands, A = H_1 (outer), B = H_2 (middle). Keys are
# the nesting [k1,[k2,[k3,[k4,k5]]]] read left to right.
_J1 = {
    "AAABA": 0.0047, "AABBA": 0.0057, "ABABA": 0.0046, "ABBBA": 0.0074,
    "BAABA": 0.0097, "BABBA": 0.0097, "BBABA": 0.0173, "BBBBA": 0.0284,
}

# Proposition J.2, Table 2: c_{i,j,k,l,m} for [H_i,[H_j,[H_k,[H_l,H_m]]]],
# listed for (l, m) in (2,1), (3,1), (3,2) with (i, j, k) in row order.
_J2_ROWS = [
    (0.0047, 0.0047, 0.0043), (0.0057, 0.0057, 0.0057), (0.0057, 0.0057, 0.0057),
    (0.0046, 0.0046, 0.0035), (0.0074, 0.0070, 0.0062), (0.0082, 0.0082, 0.0082),
    (0.0046, 0.0046, 0.0035), (0.0070, 0.0058, 0.0046), (0.0082, 0.0074, 0.0074),
    (0.0150, 0.0150, 0.0141), (0.0161, 0.0161, 0.0161), (0.0161, 0.0161, 0.0161),
    (0.0239, 0.0239, 0.0212), (0.0315, 0.0306, 0.0290), (0.0303, 0.0303, 0.0303),
    (0.0179, 0.0179, 0.0153), (0.0232, 0.0206, 0.0179), (0.0259, 0.0241, 0.0241),
    (0.0204, 0.0204, 0.0186), (0.0225, 0.0225, 0.0217), (0.0225, 0.0225, 0.0225),
    (0.0423, 0.0423, 0.0377), (0.0585, 0.0571, 0.0537), (0.0502, 0.0502, 0.0502),
    (0.0423, 0.0423, 0.0377), (0.0681, 0.0641, 0.0601), (0.0648, 0.0621, 0.0628),
]


def _table(n_summands: int) -> dict[tuple[int, int, int, int, int], float]:
    if n_summands == 2:
        idx = {"A": 0, "B": 1}
        return {tuple(idx[ch] for ch in key): c for key, c in _J1.items()}
    if n_summands == 3:
        table = {}
        ijk = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
        for row, (i, j, k) in zip(_J2_ROWS, ijk):
            for c, (l, m) in zip(row, [(1, 0), (2, 0), (2, 1)]):
                table[(i, j, k, l, m)] = c
        return table
    raise ValueError("fourth-order bound is tabulated for 2 or 3 grouped summands only")


def fourth_order_bound(Hs: list[Sentence], t: float, exact=False) -> float:
    """Proposition J.1 (two summands) or J.2 (three summands)."""
    by_inner: dict[tuple[int, int], dict] = defaultdict(dict)
    for (i, j, k, l, m), c in _table(len(Hs)).items():
        by_inner[(l, m)][(k, j, i)] = c  # outer layers from the inside out
    return t**5 * sum(nested_sum(Hs, inner, ents, exact) for inner, ents in by_inner.items())


_BOUNDS = {1: first_order_bound, 2: second_order_bound, 4: fourth_order_bound}
_ALPHA_CACHE: dict = {}


def alpha(H: Hamiltonian, order: int, exact: bool = False) -> float:
    """Prefactor such that one step of time tau has error <= alpha * tau^(p+1)."""
    key = (tuple(H.terms), order, exact)
    if key not in _ALPHA_CACHE:
        Hs = [_sentence(g) for g in groups(H)]
        _ALPHA_CACHE[key] = _BOUNDS[order](Hs, 1.0, exact)
    return _ALPHA_CACHE[key]


def step_error(H: Hamiltonian, order: int, tau: float, exact: bool = False) -> float:
    return alpha(H, order, exact) * tau ** (order + 1)


def steps(order: int, H: Hamiltonian, t: float, eps: float, exact: bool = False) -> int:
    """Smallest r with r * error(t/r) <= eps. Closed form since the step error
    is a pure power law: r = ceil((alpha t^(p+1) / eps)^(1/p))."""
    if order not in _BOUNDS:
        raise ValueError("Theory-of-Trotter-error bounds implemented for orders 1, 2, 4")
    return max(1, math.ceil((alpha(H, order, exact) * t ** (order + 1) / eps) ** (1 / order)))
