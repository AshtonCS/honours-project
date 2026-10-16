"""Morales et al.'s unprocessed order-8 formulas -- weights GENERATED, do not edit.

M. E. S. Morales, P. C. S. Costa, G. Pantaleoni, D. K. Burgarth, Y. R. Sanders
and D. W. Berry, "Selection and improvement of product formulae for best
performance of quantum simulation", arXiv:2210.15817v2 (published in Quant.
Inf. Comput. 25, 2025).

Structure (their Sec. II.2, Eq. 15, "Yoshida's ansatz"): a symmetric chain of
second-order (Leapfrog / Strang) steps,

    S(t) = S2(w_m t) ... S2(w_1 t) S2(w_0 t) S2(w_1 t) ... S2(w_m t),
    w_0 = 1 - 2 (w_1 + ... + w_m)                 (their Sec. II.2),

so 2m + 1 Leapfrog steps. In ramp form (Maležič & Ostmeyer, Eq. 9) each
S2(w t) is one cycle with c_i = d_i = w_i / 2: a special case of the ramp
form with symmetric cycles. Unprocessed: no kernel/processor (their Sec. II.3).

Weights extracted by regex (no hand transcription) from the arXiv LaTeX
source, file improved.tex, sha256 9fa28cd99c1eb7da83c9c145c633bfcb32774682a9922ea35fccd651183729eb:

* Y8M10:  Table "tab:order8-solm10", left column, "Best 8th order for
  spectral-norm error" -- labelled Y8m10 in their Table 8 (M = 21, average
  spectral-norm constant 4.9e-8, best unprocessed order 8 on that measure).
* Y8M10B: same table, right column, eigenvalue-error optimised (Y8m10b).
* M8:     Table "tab:order8-sol", right column, best order 8 with m = 8
  (M = 17); presumably the "Morales et al. (q=17)" scheme in Maležič &
  Ostmeyer's Figs. 5-6 (not confirmed against their coefficients).
* M7:     same table, left column (M = 15); Morales et al. identify it with
  Kahan & Li's s15odr8.

Order 8 of every set is checked in tests/test_effts_morales.py through the
independent BCH computation (bch.py): a single mistyped digit fails it.
"""

from __future__ import annotations

from .schemes import Scheme, from_ramps

Y8M10 = (
    "0.59358060400850625863514059265224",
    "-0.46916012347004197296293264921328",
    "0.2743566425898467907228242878146",
    "0.17193879484656773059919074965377",
    "0.23439874482541384415430578747541",
    "-0.48616424480326193899617759997914",
    "0.49617367388114660354871757044906",
    "-0.32660218948439130114501815323814",
    "0.23271679349369857679445410270557",
    "0.098249557414708533273471906180643",
)

Y8M10B = (
    "0.10467636532245895252340732579853",
    "-0.57896999331780988041471955125778",
    "0.57503350160061785946141563279891",
    "0.12231011868707029786561397542663",
    "0.27793149999039524816733903301747",
    "-0.37349605088056728482635987352576",
    "0.11575566589480463220616543972403",
    "0.1464645610975800618712569230326",
    "-0.39443578322284085764474498594073",
    "0.44370228726021218923197141183196",
)

M8 = (
    "0.29137384767986663096528500968049",
    "0.26020394234904150277316667709864",
    "0.18669648149540687549831902999911",
    "-0.40049110428180105319963667975074",
    "0.15982762208609923217390166127256",
    "-0.38400573301491401473462588779099",
    "0.56148845266356446893590729572808",
    "0.12783360986284110837857554950443",
)

M7 = (
    "0.315293092396766596632056663811",
    "0.33462491824529818378495797988218",
    "0.2990641813036559238444635406886",
    "-0.57386247111608226665638772663554",
    "0.19075471029623837995387625645037",
    "-0.40910082580003159399730009589356",
    "0.74167036435061295344822780178381",
)


def composition(weights: tuple[str, ...], order: int, name: str) -> Scheme:
    """Scheme for S2(w_m) ... S2(w_1) S2(w_0) S2(w_1) ... S2(w_m), w_0 = 1 - 2 sum w_j.

    Each S2(w t) is one cycle with ramp weights c = d = w/2; the sequence is a
    palindrome, so ``from_ramps`` (which sets d_i = c_(q+1-i)) gives c_i = d_i.
    """
    w = [float(x) for x in weights]           # w_1 .. w_m
    w0 = 1.0 - 2.0 * sum(w)
    seq = list(reversed(w)) + [w0] + w        # w_m .. w_1, w_0, w_1 .. w_m
    return from_ramps(tuple(x / 2 for x in seq), order, name)


def y8m10() -> Scheme:
    """Morales et al. Y8m10: order 8, 21 Leapfrog steps, spectral-norm optimised."""
    return composition(Y8M10, 8, "Morales Y8m10")


def y8m10b() -> Scheme:
    """Morales et al. Y8m10b: order 8, 21 steps, eigenvalue-error optimised."""
    return composition(Y8M10B, 8, "Morales Y8m10b")


def m8() -> Scheme:
    """Morales et al. best order 8 with m = 8: 17 Leapfrog steps."""
    return composition(M8, 8, "Morales m=8 (17 steps)")


def m7() -> Scheme:
    """Morales et al. best order 8 with m = 7 (= Kahan & Li s15odr8): 15 steps."""
    return composition(M7, 8, "Morales m=7 / Kahan-Li s15odr8")
