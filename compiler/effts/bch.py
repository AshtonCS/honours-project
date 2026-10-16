"""BCH error coefficients of a two-operator scheme, computed independently.

Upstream computes the coefficients alpha, beta, gamma_k, delta_k, epsilon_k
with hand-derived recursive formulae (paper Sec. 2.2, Eqs. 15-19; ~5000
generated lines in ``NCoefficients.cpp``).  This module does not port those
formulae.  It computes the same numbers a different way:

1. Work in the free associative algebra on two letters {A, B}, truncated at
   word length D = 9 (enough for Err_8, whose commutators have degree 9).
2. Multiply out the scheme's product of exponentials (paper Eq. 8) in that
   algebra, with h = 1.  Each exponential is exp(w X) = sum_k w^k X^k / k!.
3. Take the logarithm, log(1 + X) = sum_k (-1)^{k+1} X^k / k.  Because the
   product has constant term 1 and is truncated, this series is finite.
4. Split the logarithm by degree.  Degree d holds the coefficient of h^d in
   the exponent of Eq. (10), i.e. O_d.  Expand each upstream basis
   commutator of degree d (``basis.py``) into words and solve the linear
   system for its coordinates.

So the two implementations share only the definition of the basis.  If they
agree on Err_n for every published scheme, both the recursion and the
published coefficients are confirmed.

Representation.  A word of length d is an integer in [0, 2^d) written in
binary with the first letter as the most significant bit, A = 0, B = 1.
So "AAB" = 0b001 = 1.  Concatenating u (length i) and v (length j) gives
u * 2^j + v, which is exactly the flattened index of ``np.outer(x, y)``.
An algebra element is a list ``x`` with ``x[d]`` the length-2^d coefficient
vector of its degree-d part.

Floating point is double throughout.  Upstream works in long double / quad,
which matters only for the constraint residues (their Err^2 ~ 1e-72); for the
leading-order errors the comparison is at ~1e-10 relative (see tests).
"""

from __future__ import annotations

import math

import numpy as np

from .basis import BASIS
from .schemes import Scheme

MAX_DEGREE = 9  # Err_8 needs degree 9 (the F_k); nothing higher is supported.

Element = list[np.ndarray]


def zero(max_degree: int = MAX_DEGREE) -> Element:
    """The zero element, degrees 0..max_degree."""
    return [np.zeros(2 ** d) for d in range(max_degree + 1)]


def one(max_degree: int = MAX_DEGREE) -> Element:
    """The unit element (empty word has coefficient 1)."""
    x = zero(max_degree)
    x[0][0] = 1.0
    return x


def letter(name: str, max_degree: int = MAX_DEGREE) -> Element:
    """The single letter A or B as an element of degree 1."""
    x = zero(max_degree)
    x[1][{"A": 0, "B": 1}[name]] = 1.0
    return x


def add(x: Element, y: Element, scale: float = 1.0) -> Element:
    """x + scale * y."""
    return [xd + scale * yd for xd, yd in zip(x, y)]


def mul(x: Element, y: Element) -> Element:
    """Product x y, truncated at the common maximum degree.

    Degree d of the product collects every split d = i + j of a degree-i word
    of x followed by a degree-j word of y.
    """
    D = len(x) - 1
    z = zero(D)
    for d in range(D + 1):
        for i in range(d + 1):
            j = d - i
            if not x[i].any() or not y[j].any():
                continue
            z[d] += np.outer(x[i], y[j]).ravel()
    return z


def exp_letter(name: str, weight: float, max_degree: int = MAX_DEGREE) -> Element:
    """exp(weight * X) for a single letter X: coefficient weight^k / k! on X^k."""
    x = zero(max_degree)
    index_all = {"A": lambda k: 0, "B": lambda k: 2 ** k - 1}[name]  # AAA..A or BBB..B
    for k in range(max_degree + 1):
        x[k][index_all(k)] = weight ** k / math.factorial(k)
    return x


def log(p: Element) -> Element:
    """log(p) for p with constant term exactly 1, by the truncated series."""
    if abs(p[0][0] - 1.0) > 1e-14:
        raise ValueError(f"log needs constant term 1, got {p[0][0]}")
    D = len(p) - 1
    x = [pd.copy() for pd in p]
    x[0][0] = 0.0  # x = p - 1, which has no degree-0 part
    result = zero(D)
    power = one(D)
    for k in range(1, D + 1):  # x^k has minimum degree k, so k <= D suffices
        power = mul(power, x)
        result = add(result, power, (-1) ** (k + 1) / k)
    return result


def commutator(x: Element, y: Element) -> Element:
    """[x, y] = x y - y x."""
    return add(mul(x, y), mul(y, x), -1.0)


def right_nested(word: str, max_degree: int = MAX_DEGREE) -> np.ndarray:
    """Expand [X1 X2 ... Xk] = [X1, [X2, [..., [X(k-1), Xk]]]] into words.

    Returns the degree-k coefficient vector.  Example: "AB" -> AB - BA.
    """
    e = letter(word[-1], max_degree)
    for ch in reversed(word[:-1]):
        e = commutator(letter(ch, max_degree), e)
    return e[len(word)]


def basis_matrix(degree: int) -> np.ndarray:
    """Columns = upstream basis commutators of this degree, expanded into words."""
    if degree not in BASIS:
        raise ValueError(f"no basis for degree {degree}; have {sorted(BASIS)}")
    cols = [right_nested(w) for w, _ in BASIS[degree]]
    return np.column_stack(cols)


_BASIS_CACHE: dict[int, np.ndarray] = {}


def coordinates(vec: np.ndarray, degree: int, tol: float = 1e-10) -> np.ndarray:
    """Coordinates of a degree-``degree`` Lie element in the upstream basis.

    Least-squares solve, then a hard check that the residual is ~0: the input
    must actually be a Lie element (it is, for the log of a product of
    exponentials, by the BCH theorem), and the basis must span the space.
    Raises otherwise rather than returning a best fit.
    """
    if degree not in _BASIS_CACHE:
        _BASIS_CACHE[degree] = basis_matrix(degree)
    M = _BASIS_CACHE[degree]
    coef, *_ = np.linalg.lstsq(M, vec, rcond=None)
    resid = np.linalg.norm(M @ coef - vec)
    scale = max(np.linalg.norm(vec), 1.0)
    if resid > tol * scale:
        raise ValueError(f"degree-{degree} element not in span of basis: residual {resid:.3e}")
    return coef


def scheme_log(scheme: Scheme, max_degree: int = MAX_DEGREE) -> Element:
    """log of the Eq. (8) product with h = 1, truncated at ``max_degree``."""
    p = one(max_degree)
    for op, w in scheme.stages():  # leftmost factor first
        p = mul(p, exp_letter(op, w, max_degree))
    return log(p)


def error_coefficients(scheme: Scheme, max_degree: int = MAX_DEGREE) -> dict[int, np.ndarray]:
    """BCH coefficients of the scheme, keyed by degree.

    Degree 1: (nu - 1, sigma - 1) of Eq. (12), coefficients of A and B.
    Degree 3: (alpha, beta), Eq. (13).  Degree 5: gamma_1..6, Eq. (14).
    Degree 7: delta_1..18.  Degree 9: epsilon_1..56.
    Even degrees vanish for a symmetric scheme; this is checked.
    """
    lg = scheme_log(scheme, max_degree)
    out = {1: np.array([lg[1][0] - 1.0, lg[1][1] - 1.0])}
    for d in range(2, max_degree + 1):
        if d % 2 == 0:
            if np.linalg.norm(lg[d]) > 1e-10:
                raise ValueError(f"symmetric scheme has nonzero degree-{d} term: {np.linalg.norm(lg[d]):.3e}")
            continue
        out[d] = coordinates(lg[d], d)
    return out


def err(scheme: Scheme, n: int) -> float:
    """Err_n of paper Eq. (20): Euclidean norm of the degree-(n+1) coefficients.

    n is an even order 2, 4, 6, 8.  For a scheme of order n this is its
    leading-order error; for orders below the scheme's own it is a
    constraint residue and should be ~0.
    """
    if n not in (2, 4, 6, 8):
        raise ValueError(f"n must be 2, 4, 6 or 8, got {n}")
    return float(np.linalg.norm(error_coefficients(scheme, n + 1)[n + 1]))


def efficiency(scheme: Scheme, n: int | None = None) -> float:
    """Eff_n = 1 / (q^n Err_n), paper Eq. (22).  Defaults to n = scheme.order."""
    n = scheme.order if n is None else n
    return 1.0 / (scheme.q ** n * err(scheme, n))
