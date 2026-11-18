"""RQ4 figure: the optimiser's CNOT factor beyond merging, against cycles per step q.

    python -m pysimcount.figoptq results/np-mean/rq4-optimiser-table.json [--n 11]

Reads the record written by ``pfopttable`` (nothing is recomputed) and plots,
for every formula of order >= 2 at one measured n, the optimiser CNOT factor

    opt_cnot = CNOT_total(merged) / CNOT_total(Light-optimised)

(geometric mean over seeds) against q, the formula's cycles per step
(``FormulaSpec.q``; Suzuki S_2k has 5^(k-1)), with the layer-count model

    opt_cnot(q) = (4q + 1) / (4q)

drawn as a curve. The model, for the X-Y-Z grouping (three stages,
XX | YY | ZZ+Z, ``weighted`` module docstring): call a "layer" one two-qubit
Pauli type on every bond (2n CNOT with simcount's template). A cycle is a
forward ramp XX YY ZZ and a backward ramp ZZ YY XX, so 6 layers unmerged.
``weighted.merge`` joins the two ZZ layers in each cycle (q of them) and the
XX layers where consecutive cycles meet (q - 1), leaving 4q + 1 layers per
step. The optimiser's only further CNOT saving is the XX layer shared across
a step boundary, leaving 4q. The model is derived, not fitted: ``check``
raises unless every per-seed factor equals it to 1e-12, so the figure cannot
silently plot data that disagree with the curve.

Lie-Trotter S1 is excluded: it is not palindromic (XX YY ZZ, then the next
step starts again at XX), so no layer repeats and its factor is 1, outside
the model.

Writes ``<record dir>/figures/rq4-optimiser-vs-q.tex``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pffigure
from .pfcompare import formula_registry, provenance

TOL = 1e-12


def optimiser_factor(q: int) -> float:
    """Model CNOT factor merged -> optimised for q palindromic cycles per step.

    q: cycles per step (dimensionless, q >= 1). Valid for the three-stage
    X-Y-Z grouping with simcount's gate template; see the module docstring.
    Raises for q < 1."""
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    return (4 * q + 1) / (4 * q)


def check(rows: list[dict], registry: dict, n: int) -> list[dict]:
    """The rows at size n for formulas of order >= 2, each verified against the model.

    Raises if n is absent, if a scheme is unknown, or if any per-seed factor
    differs from ``optimiser_factor(q)`` by more than TOL."""
    picked = [r for r in rows if r["n"] == n and registry[r["scheme"]].order >= 2]
    if not picked:
        raise ValueError(f"no rows at n = {n}")
    for r in picked:
        q = registry[r["scheme"]].q
        model = optimiser_factor(q)
        for seed, v in r["opt_cnot"]["per_seed"].items():
            if abs(v - model) > TOL:
                raise ArithmeticError(f"{r['scheme']} n={n} seed {seed}: optimiser CNOT factor {v!r} "
                                      f"!= model (4q+1)/4q = {model!r} at q = {q}")
    return picked


def figure_tex(rows: list[dict], registry: dict, header: list[str]) -> str:
    """semilogx: model curve plus one marker per formula, chapter colours/markers.

    Formulas sharing a q (hence the same point) are drawn as nested markers:
    the first filled, later ones open and larger, so none is hidden."""
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{semilogxaxis}[",
        "    width=10cm, height=7cm,",
        "    xlabel={Cycles per step $q$}, ylabel={Optimiser CNOT factor},",
        "    xmin=0.7, xmax=200, ymin=0.99, ymax=1.27,",
        "    xtick={1,5,10,25,50,125}, xticklabels={1,5,10,25,50,125},",
        "    ymajorgrids=true, xmajorgrids=true,",
        "    legend pos=outer north east, legend cell align=left, legend style={font=\\footnotesize},",
        "  ]",
        "    \\addplot[black, thick, domain=1:125, samples=200] {(4*x + 1)/(4*x)};",
        "    \\addlegendentry{$(4q+1)/4q$}"]
    seen: dict[int, int] = {}
    # Order by q then by registry order, so coincident points nest predictably.
    order = list(registry)
    for r in sorted(rows, key=lambda r: (registry[r["scheme"]].q, order.index(r["scheme"]))):
        f = registry[r["scheme"]]
        colour = pffigure.COLOUR_OVERRIDE.get(r["scheme"], pffigure.FAMILY_COLOUR.get(f.family))
        if colour is None or f.order not in pffigure.ORDER_MARK:
            raise ValueError(f"no pffigure style for {r['scheme']}")
        k = seen.get(f.q, 0)
        seen[f.q] = k + 1
        size = 2.2 + 1.4 * k
        fill = colour if k == 0 else "none"
        out += [f"    \\addplot[only marks, {colour}, mark={pffigure.ORDER_MARK[f.order]}, mark size={size:.1f}pt, "
                f"mark options={{solid, fill={fill}}}] coordinates {{({f.q},{r['opt_cnot']['geomean']:.6f})}};",
                f"    \\addlegendentry{{{pffigure.latex_escape(f.name)}}}"]
    out += ["  \\end{semilogxaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path, help="pfopttable JSON record (rq4-optimiser-table.json)")
    ap.add_argument("--n", type=int, default=None, help="measured size (default: the record's table_n)")
    args = ap.parse_args(argv)
    rec = json.loads(args.record.read_text())
    n = rec["table_n"] if args.n is None else args.n
    registry = formula_registry()
    rows = check(rec["rows"], registry, n)
    prov = provenance()
    header = ["Generated by pysimcount.figoptq -- do not edit; regenerate instead.",
              f"source: {args.record.name} (pfopttable), n = {n}, seeds {rec['seeds']}, cz, measured r, merged steps",
              "y = optimiser CNOT factor merged -> Light-optimised, geometric mean over seeds (marginal per step x r)",
              f"every per-seed factor checked equal to (4q+1)/4q within {TOL:g}; Lie-Trotter S1 excluded (not palindromic)",
              f"figure made at git_commit={prov.get('git_commit')} git_dirty={prov.get('git_dirty')}"]
    out = args.record.parent / "figures" / "rq4-optimiser-vs-q.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(figure_tex(rows, registry, header))
    print(f"wrote {out} ({len(rows)} formulas, n = {n}; all per-seed factors match (4q+1)/4q)")


if __name__ == "__main__":
    main()
