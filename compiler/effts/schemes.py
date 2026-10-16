"""Two-operator Trotter-Suzuki schemes of Malezic & Ostmeyer, and their notations.

Paper: M. Malezic and J. Ostmeyer, "Efficient Trotter-Suzuki Schemes for
Long-time Quantum Dynamics", arXiv:2601.18756 (v3).  Equation numbers below
refer to v3.

A scheme approximates exp(h (A + B)) by a palindromic product.  Three
notations for the same scheme appear in the paper and the upstream code:

1. **File notation** -- the ``Vector a`` / ``Vector b`` lines in the upstream
   ``schemes_q#.out`` files and the ``a_vec`` / ``b_vec`` keys of their input
   files.  Half a palindrome, listed from the centre outward, and every entry
   except the central one is *twice* the stage parameter it stands for
   (README-Schemes.md: "multiplied by 1/2 to those found in Section 2.2").
   The central entry is not doubled: paper Eq. (10) writes it as 2 a_1.
2. **Stage notation**, paper Eq. (8):
       e^{a_1 h A} e^{b_1 h B} e^{a_2 h A} ... e^{b_q h B} e^{a_{q+1} h A},
   with q + 1 A-stages and q B-stages, sum(a) = sum(b) = 1.
3. **Ramp notation**, paper Eqs. (7) and (9): q forward ramps with weights c_i
   and q backward ramps with weights d_i, sum(c) = sum(d) = 1/2.  This is the
   form that generalises to Lambda > 2 operators.

q is the number of cycles (one forward plus one backward ramp); the cost of a
step is proportional to q (the paper's computational cost is q * N_t).

The file -> stage conversion follows ``Trotter::to_standard`` in
``Code/Time_Evolution/src/Trotter.cpp``; the stage -> ramp conversion is Eq. (9)
and reproduces ``Trotter::to_ramp``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Scheme:
    """A symmetric two-operator scheme in stage notation (paper Eq. 8).

    a: the q + 1 A-stage weights a_1 .. a_{q+1} (fractions of the step h).
    b: the q B-stage weights b_1 .. b_q.
    order: the order n claimed for the scheme (S(h) = U(h) + O(h^{n+1})).
        Recorded, not checked here -- ``bch.py`` and the tests check it.
    name: free-text label.
    reference: the upstream Err_n^2 values for this scheme, keyed by n, when
        the scheme came from an upstream ``.out`` file (else empty).  Used only
        for comparison, never as input.
    reference_distance: the upstream "Distance from the origin" value, if any.
    """

    a: tuple[float, ...]
    b: tuple[float, ...]
    order: int
    name: str = ""
    reference: tuple[tuple[int, float], ...] = ()
    reference_distance: float | None = None

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b) + 1:
            raise ValueError(f"need len(a) == len(b) + 1, got {len(self.a)}, {len(self.b)}")
        if len(self.b) < 1:
            raise ValueError("a scheme needs at least one B stage")
        # Consistency of Eq. (12): nu = sigma = 1.  1e-12 is far above the
        # rounding of 18-36 digit inputs and far below any real defect.
        if abs(sum(self.a) - 1.0) > 1e-12 or abs(sum(self.b) - 1.0) > 1e-12:
            raise ValueError(f"stage weights must each sum to 1: {sum(self.a)}, {sum(self.b)}")
        if not np.allclose(self.a, self.a[::-1], atol=1e-14, rtol=0) or \
           not np.allclose(self.b, self.b[::-1], atol=1e-14, rtol=0):
            raise ValueError("only symmetric (palindromic) schemes are supported")

    @property
    def q(self) -> int:
        """Number of cycles (paper's q)."""
        return len(self.b)

    def stages(self) -> list[tuple[str, float]]:
        """The Eq. (8) product as a list of (operator, weight), leftmost first."""
        out: list[tuple[str, float]] = []
        for i in range(self.q):
            out.append(("A", self.a[i]))
            out.append(("B", self.b[i]))
        out.append(("A", self.a[self.q]))
        return out

    def ramps(self) -> tuple[np.ndarray, np.ndarray]:
        """Ramp parameters (c, d) of paper Eq. (9).

        c_1 = a_1,  d_i = b_i - c_i,  c_{i+1} = a_{i+1} - d_i.
        The recursion leaves one equation unused, a_{q+1} = d_q, which holds
        because sum(a) = sum(b); it is checked rather than assumed.
        """
        q = self.q
        c = np.zeros(q)
        d = np.zeros(q)
        for i in range(q):
            c[i] = self.a[0] if i == 0 else self.a[i] - d[i - 1]
            d[i] = self.b[i] - c[i]
        if abs(d[q - 1] - self.a[q]) > 1e-12:
            raise ValueError(f"ramp recursion inconsistent: d_q = {d[q - 1]}, a_(q+1) = {self.a[q]}")
        return c, d

    def distance(self) -> float:
        """Distance from the origin x-bar, paper Eq. (23).

        x-bar^2 = 2 sum_i (c_i - 1/(2q))^2.  The factor 2 counts the d_i, which
        equal the c_i in reverse for a symmetric scheme (d_i = c_{q+1-i},
        Tables 2-3).  The origin c_i = d_i = 1/(2q) is the equal-ramp scheme.

        Note: Sec. 2.3 of the paper calls the origin "c_i = d_i = 1/q"; with
        sum(c) = 1/2 that cannot be the same point, and 1/(2q) is what Eq. (23)
        uses and what reproduces the upstream x values (see tests).
        """
        c, d = self.ramps()
        if not np.allclose(d, c[::-1], atol=1e-12, rtol=0):
            raise ValueError("Eq. (23) assumes d_i = c_{q+1-i}")
        q = self.q
        total = 0.0
        for ci in c:
            total += (ci - 1.0 / (2 * q)) ** 2
        return float(np.sqrt(2.0 * total))


def from_file_notation(a_file: list[float], b_file: list[float], order: int, name: str = "",
                       reference: tuple[tuple[int, float], ...] = (),
                       reference_distance: float | None = None) -> Scheme:
    """Build a Scheme from upstream file notation (``Vector a`` / ``Vector b``).

    Mirrors ``Trotter::to_standard``.  For even q the central stage is an A
    (a_file[0], not halved) and there are q/2 + 1 a-entries, q/2 b-entries.
    For odd q the central stage is a B (b_file[0], not halved) and there are
    (q + 1)/2 entries of each.  All other entries are halved.

    Worked example, Forest & Ruth (q = 3, file a = [-0.3512, 1.3512],
    b = [-1.7024, 2.7024]):  A stages 0.6756, -0.1756, -0.1756, 0.6756;
    B stages 1.3512, -1.7024, 1.3512.
    """
    na, nb = len(a_file), len(b_file)
    if na == nb + 1:  # even q, central A
        q = 2 * nb
        half_a = [a_file[k] / 2 for k in range(na - 1, 0, -1)] + [a_file[0]]
        half_b = [b_file[k] / 2 for k in range(nb - 1, -1, -1)]
        a = half_a + half_a[-2::-1]
        b = half_b + half_b[::-1]
    elif na == nb:  # odd q, central B
        q = 2 * nb - 1
        half_a = [a_file[k] / 2 for k in range(na - 1, -1, -1)]
        half_b = [b_file[k] / 2 for k in range(nb - 1, 0, -1)] + [b_file[0]]
        a = half_a + half_a[::-1]
        b = half_b + half_b[-2::-1]
    else:
        raise ValueError(f"file notation needs len(a) in {{len(b), len(b)+1}}, got {na}, {nb}")
    assert len(b) == q and len(a) == q + 1
    return Scheme(tuple(a), tuple(b), order, name, reference, reference_distance)


def to_file_notation(scheme: Scheme) -> tuple[list[float], list[float]]:
    """Inverse of ``from_file_notation`` (used to write upstream input files)."""
    q = scheme.q
    if q % 2 == 0:
        a_file = [scheme.a[q // 2]] + [2 * scheme.a[k] for k in range(q // 2 - 1, -1, -1)]
        b_file = [2 * scheme.b[k] for k in range(q // 2 - 1, -1, -1)]
    else:
        m = (q - 1) // 2
        a_file = [2 * scheme.a[k] for k in range(m, -1, -1)]
        b_file = [scheme.b[m]] + [2 * scheme.b[k] for k in range(m - 1, -1, -1)]
    return a_file, b_file


# ---------------------------------------------------------------- .out parser

_ORDER_RE = re.compile(r"Order:\s*(\d+),\s*No\. cycles:\s*(\d+)")
_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([-+0-9.eE]+|Infinity|0\.0)\s*\|\s*([-+0-9.eE]+|Infinity)\s*\|")
_DIST_RE = re.compile(r"Distance from the origin: x = ([-+0-9.eE]+)")


def parse_out_file(path: Path) -> list[Scheme]:
    """Parse every scheme in an upstream ``schemes_q#.out`` file.

    The description lines become ``name`` (first line only, trimmed).  The
    Error^2 column is kept as ``reference``; the Efficiency column is not
    stored because it is a function of Error^2 (Eff_n = 1 / (q^n Err_n),
    paper Eq. 22) and is recomputed.

    Raises if a block is missing its vectors or order line -- no silent skips.
    """
    text = Path(path).read_text()
    blocks = text.split("Scheme evaluation results:")[1:]
    if not blocks:
        raise ValueError(f"no scheme blocks in {path}")
    schemes: list[Scheme] = []
    for block in blocks:
        m = _ORDER_RE.search(block)
        if not m:
            raise ValueError(f"no order line in a block of {path}")
        order, q = int(m.group(1)), int(m.group(2))
        a_line = re.search(r"Vector a:(.*)", block)
        b_line = re.search(r"Vector b:(.*)", block)
        if not a_line or not b_line:
            raise ValueError(f"missing Vector a/b in {path}")
        a_file = [float(x) for x in a_line.group(1).split(",")]
        b_file = [float(x) for x in b_line.group(1).split(",")]
        lines = block.splitlines()
        b_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Vector b:"))
        desc = lines[b_idx + 2].strip() if len(lines) > b_idx + 2 else ""
        rows: list[tuple[int, float]] = []
        for ln in lines:
            r = _ROW_RE.match(ln.strip())
            if r:
                rows.append((int(r.group(1)), float(r.group(2))))
        d = _DIST_RE.search(block)
        s = from_file_notation(a_file, b_file, order, desc, tuple(rows),
                               float(d.group(1)) if d else None)
        if s.q != q:
            raise ValueError(f"{path}: header says q={q}, vectors give q={s.q}")
        schemes.append(s)
    return schemes


def recommended(order: int, q: int) -> list[Scheme]:
    """All schemes in the upstream Recommended_schemes file for (order, q)."""
    path = DATA_DIR / "recommended" / f"order_n{order}" / f"schemes_q{q}.out"
    if not path.exists():
        raise FileNotFoundError(f"no recommended schemes for order {order}, q {q}: {path}")
    return parse_out_file(path)


def all_recommended() -> list[tuple[Path, Scheme]]:
    """Every scheme in every copied Recommended_schemes file, with its source."""
    out = []
    for path in sorted((DATA_DIR / "recommended").glob("order_n*/schemes_q*.out")):
        for s in parse_out_file(path):
            out.append((path, s))
    return out


# ------------------------------------------------------- named schemes (Fig 5)

def leapfrog() -> Scheme:
    """Leapfrog / Strang splitting, q = 1: e^{hA/2} e^{hB} e^{hA/2}."""
    return Scheme((0.5, 0.5), (1.0,), 2, "Leapfrog")


def _first(order: int, q: int, startswith: str) -> Scheme:
    for s in recommended(order, q):
        if s.name.startswith(startswith):
            return s
    raise LookupError(f"no scheme starting {startswith!r} in order {order}, q {q}")


def forest_ruth() -> Scheme:
    """Forest & Ruth, q = 3, order 4 (upstream order_n4/cycles_q3)."""
    return _first(4, 3, "The Forest & Ruth scheme")


def yoshida6() -> Scheme:
    """Yoshida's 6th-order scheme, q = 7 (upstream order_n6/cycles_q7)."""
    return _first(6, 7, "Yoshida's 6th order scheme")


def recommended_q6() -> Scheme:
    """The recommended order-4 scheme at q = 6 (paper Table 2), from the repo.

    Identified as the "2nd highest efficiency" entry, which is how Table 2's
    caption describes it.  Its ramp parameters agree with Table 2 to ~1e-10,
    not to the 18 printed digits -- see ``tests/test_schemes.py``.
    """
    return _first(4, 6, "2nd highest efficiency scheme")


def recommended_q14() -> Scheme:
    """The recommended order-6 scheme at q = 14 (paper Table 3), from the repo.

    Identified as the "2nd highest efficiency" entry (Table 3's caption).
    Its ramp parameters differ from Table 3 by up to ~5e-7 -- a real
    disagreement between the repository and the paper, recorded in
    ``tests/test_schemes.py`` and ``table3_scheme`` below.
    """
    return _first(6, 14, "2nd highest efficiency scheme")


# Paper Tables 2 and 3, c_i = d_{q+1-i}, copied from the arXiv v3 HTML.
TABLE2_C = (0.074082572180463262, 0.232923088374338803, 0.296820560634668408,
            0.122086989386933251, -0.350153632343424469, 0.124240421767020743)
TABLE3_C = (0.037251326545569924, 0.120600278793781562, 0.266062994460763541,
            0.163668553338143183, 0.071316838327437583, 0.058117508592333414,
            0.188707697234255120, -0.200016005078878524, 0.074145714537530386,
            0.087345801243357893, 0.044234977360777830, -0.230821838291030424,
            -0.237197828922049295, 0.056583981858007803)


def from_ramps(c: tuple[float, ...], order: int, name: str = "") -> Scheme:
    """Scheme from symmetric ramp parameters c_i (with d_i = c_{q+1-i}).

    Inverse of Eq. (9): a_1 = c_1, b_i = c_i + d_i, a_{i+1} = d_i + c_{i+1},
    a_{q+1} = d_q.

    The paper prints only 18 digits and the stage sums are then checked by
    ``Scheme`` to 1e-12, so a table with a typo in one entry raises here.
    """
    q = len(c)
    d = c[::-1]
    a = [c[0]] + [d[i] + c[i + 1] for i in range(q - 1)] + [d[q - 1]]
    b = [c[i] + d[i] for i in range(q)]
    return Scheme(tuple(a), tuple(b), order, name)


def table_scheme(table: int) -> Scheme:
    """The scheme exactly as printed in paper Table 2 (q=6) or Table 3 (q=14)."""
    if table == 2:
        return from_ramps(TABLE2_C, 4, "Paper Table 2 (q=6)")
    if table == 3:
        return from_ramps(TABLE3_C, 6, "Paper Table 3 (q=14)")
    raise ValueError("table must be 2 or 3")


# Blanes & Moan schemes are not in the upstream scheme archive; upstream uses
# them only as input files for the harmonic-oscillator tests.  Values copied
# verbatim from those files (file notation):
#   Code/Time_Evolution/Input_files/Errors/Harmonic/tests/template_q6_Blanes6.in
#   Code/Time_Evolution/Input_files/Errors/Harmonic/tests/template_q10_Blanes10.in
# The original source is S. Blanes and P. C. Moan, J. Comput. Appl. Math. 142
# (2002) 313 (the paper's Ref. [4]); I have NOT checked these values against
# that paper -- the comparison below is against the upstream copy only.
_BLANES_MOAN = {
    6: ([0.21937695575349960000, -0.08413016071543900000, 0.70634581209954810000, 0.15840739286239140001],
        [0.86867333313291200003, -0.28770354635963599999, 0.41903021322672399999], 4),
    10: ([-0.72552555850868970000, 1.08392135690155999995, -0.37610970763913799999,
          0.09015977958879540000, 0.82702860085668800002, 0.10052552880078440000],
         [-0.03280917880723600000, 0.86533280515635000000, 0.13461520938437000000,
          -0.26477173153556799999, 0.29763289580208399999], 6),
}


def blanes_moan(q: int) -> Scheme:
    """Blanes & Moan: q = 6 (order 4) or q = 10 (order 6), as copied upstream."""
    if q not in _BLANES_MOAN:
        raise ValueError(f"Blanes & Moan available for q in {sorted(_BLANES_MOAN)}, got {q}")
    a, b, n = _BLANES_MOAN[q]
    return from_file_notation(a, b, n, f"Blanes & Moan (q={q})")


def figure5_schemes() -> list[Scheme]:
    """The schemes of paper Fig. 5 that can be reproduced from upstream data.

    Missing: Morales et al. (q = 17, order 8; q = 33, order 10).  Their
    coefficients are not in the upstream repository and I have not
    transcribed them from Morales et al. (2022).
    """
    return [leapfrog(), forest_ruth(), recommended_q6(), yoshida6(),
            blanes_moan(10), recommended_q14()]

# Paper Table 4, "Highest Eff_n" column at n = 6: the most efficient q = 14
# scheme in the upstream archive.  It is not in Recommended_schemes; it is the
# first entry (and the maximum Eff_6 among all 4750 entries) of
#   Code/Trotter-Suzuki_schemes/Schemes/Trotter-Suzuki/All_schemes/double/
#   order_n6/cycles_q14/schemes_q14.out
# copied verbatim (file notation, 16 digits).  The 4.2 MB archive is not
# vendored.
_HIGHEST_Q14 = (
    [0.2153736392748491, 0.7077494047084163, -1.2620251951557697, 0.0400514723199147, 0.8600732886612621, 0.3187448128450009, -0.0399978659193973, 0.1600304432657239],
    [0.4357973406884966, -0.0140826575929769, 0.0903766592237020, -0.1158516661646498, 0.3220652753565244, 0.3618798814607640, -0.0801848329718604],
)


def highest_q14() -> Scheme:
    """The highest-Eff_6 scheme at q = 14 (paper Table 4, middle column)."""
    a, b = _HIGHEST_Q14
    return from_file_notation(a, b, 6, "Highest Eff_6 (q=14), upstream archive")
