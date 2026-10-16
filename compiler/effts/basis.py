"""Commutator basis of Malezic & Ostmeyer, degrees 3, 5, 7, 9 -- GENERATED, do not edit.

Extracted mechanically (regex, no hand transcription) from
``Code/Recursive_formulae/Recursive_formulae.tex``, "Basis" section, of
github.com/MarkoMalezic/efficient-trotterizations at commit 1a5a16839dd32369f8a4db06b2568bfe087e335a.

Each entry is ``(word, tilde)``:

* ``word`` -- right-nested commutator in their notation,
  [X1 X2 ... Xk] = [X1, [X2, [..., [X(k-1), Xk]]]].  "AAB" is [A,[A,B]] = C_1.
* ``tilde`` -- the coefficient of that commutator in the symmetric BCH formula
  log(e^{A/2} e^{B} e^{A/2}) as printed in the .tex (alpha~, beta~, gamma~_k,
  delta~_k, epsilon~_k).  Not used by the error computation; kept so the test
  suite can check that ``bch.py`` reproduces them, which pins down the reading
  of the bracket notation and the sign convention.

BASIS[d] is the basis of the degree-d part of the free Lie algebra on {A, B}
(dimensions 2, 6, 18, 56).  Err_n of a scheme of order n is the Euclidean
norm of its BCH coefficients at degree n + 1 in this basis (paper Eq. 20).
"""

from __future__ import annotations

from fractions import Fraction

BASIS: dict[int, list[tuple[str, Fraction]]] = {
    3: [  # C_1 .. C_2
        ("AAB", Fraction("-1/24")),
        ("BBA", Fraction("1/12")),
    ],
    5: [  # D_1 .. D_6
        ("AAAAB", Fraction("7/5760")),
        ("AABAB", Fraction("1/480")),
        ("BAAAB", Fraction("1/360")),
        ("ABBBA", Fraction("1/360")),
        ("BBABA", Fraction("-1/120")),
        ("BBBBA", Fraction("-1/720")),
    ],
    7: [  # E_1 .. E_18
        ("AAAAAAB", Fraction("-31/967680")),
        ("AABAAAB", Fraction("-19/80640")),
        ("ABAAAAB", Fraction("23/161280")),
        ("BAAAAAB", Fraction("-1/10080")),
        ("AABABAB", Fraction("-1/10080")),
        ("ABAABAB", Fraction("-1/20160")),
        ("ABBAAAB", Fraction("13/60480")),
        ("BABAAAB", Fraction("-1/5040")),
        ("BAAABAB", Fraction("-1/3360")),
        ("ABBBABA", Fraction("-19/40320")),
        ("ABABBBA", Fraction("1/6720")),
        ("BAABBBA", Fraction("-1/7560")),
        ("BABBABA", Fraction("1/1008")),
        ("BBABABA", Fraction("-1/10080")),
        ("ABBBBBA", Fraction("-1/10080")),
        ("BABBBBA", Fraction("1/10080")),
        ("BBABBBA", Fraction("1/5040")),
        ("BBBBBBA", Fraction("1/30240")),
    ],
    9: [  # F_1 .. F_56
        ("AAAAAAAAB", Fraction("127/154828800")),
        ("AAABAAAAB", Fraction("11/774144")),
        ("AABAAAAAB", Fraction("-41/3225600")),
        ("ABAAAAAAB", Fraction("17/9676800")),
        ("BAAAAAAAB", Fraction("1/302400")),
        ("AAABBAAAB", Fraction("1/290304")),
        ("AABABAAAB", Fraction("-1/48384")),
        ("AABAAABAB", Fraction("1/26880")),
        ("ABAABAAAB", Fraction("-1/13824")),
        ("ABABAAAAB", Fraction("1/15360")),
        ("ABBAAAAAB", Fraction("-23/1612800")),
        ("BAAABAAAB", Fraction("211/4838400")),
        ("BAABAAAAB", Fraction("-37/1209600")),
        ("BABAAAAAB", Fraction("7/691200")),
        ("AABAABBAB", Fraction("47/1209600")),
        ("AABABBAAB", Fraction("-281/2419200")),
        ("AABBABAAB", Fraction("163/1612800")),
        ("ABAABABAB", Fraction("73/1612800")),
        ("ABABAABAB", Fraction("-43/2419200")),
        ("ABABBAAAB", Fraction("-79/1209600")),
        ("ABBABAAAB", Fraction("-1/193536")),
        ("ABBAAABAB", Fraction("-13/3225600")),
        ("BAAABABAB", Fraction("-1/25200")),
        ("BAABAABAB", Fraction("1/11200")),
        ("BAABBAAAB", Fraction("-1/75600")),
        ("BABABAAAB", Fraction("1/12600")),
        ("BABAAABAB", Fraction("-1/25200")),
        ("BBABAAAAB", Fraction("-1/67200")),
        ("AABABBBBA", Fraction("-1/67200")),
        ("ABABBBABA", Fraction("-31/403200")),
        ("ABABABBBA", Fraction("11/201600")),
        ("ABBAABBBA", Fraction("-1/1209600")),
        ("ABBABBABA", Fraction("17/134400")),
        ("ABBBABABA", Fraction("-1/19200")),
        ("BAABBBABA", Fraction("31/604800")),
        ("BAABABBBA", Fraction("-1/60480")),
        ("BABAABBBA", Fraction("-1/50400")),
        ("BABABBABA", Fraction("-1/5400")),
        ("BABBABABA", Fraction("1/12600")),
        ("BBAABABBA", Fraction("1/12600")),
        ("BBABAABBA", Fraction("-11/151200")),
        ("BBABBAABA", Fraction("1/100800")),
        ("ABABBBBBA", Fraction("-29/1209600")),
        ("ABBABBBBA", Fraction("83/1209600")),
        ("ABBBABBBA", Fraction("-31/1209600")),
        ("BAABBBBBA", Fraction("1/50400")),
        ("BABABBBBA", Fraction("-1/10080")),
        ("BABBABBBA", Fraction("1/30240")),
        ("BBABBBABA", Fraction("-1/20160")),
        ("BBABABBBA", Fraction("1/15120")),
        ("BBBAABBBA", Fraction("-1/90720")),
        ("ABBBBBBBA", Fraction("1/302400")),
        ("BABBBBBBA", Fraction("-1/75600")),
        ("BBABBBBBA", Fraction("1/50400")),
        ("BBBABBBBA", Fraction("-1/60480")),
        ("BBBBBBBBA", Fraction("-1/1209600")),
    ],
}
