"""Readable tables from a ``pfcompare`` run.

    python -m pysimcount.pfreport ../results/pf-comparison/results.json

Writes ``report.md`` next to the results and prints it. Every number comes
from ``results.json``; nothing is re-run. Rows from the power-law fit are
always marked "fit".

Prediction check: the paper's efficiency Eff_n = 1 / (q^n Err_n) (Eq. 22)
predicts that, at the same accuracy, scheme X needs about
(Eff_X / Eff_Suzuki)^(1/n) times fewer exponentials than Suzuki. The number
of exponentials per run is the rotation count (one Rz per exponential), so
the measured gain is Rz_total(Suzuki) / Rz_total(X).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAIRS = [  # (order, Suzuki id, challenger ids)
    (4, "suzuki4", ["mo-q6"]),
    (6, "suzuki6", ["mo-q14", "bm-q10"]),
    (8, "suzuki8", ["mo-o8-q21", "morales-y8m10", "morales-m8"]),
]


def _eff_prediction(order: int, suzuki_id: str, other_id: str) -> float:
    from effts import bch
    from pysimcount import weighted
    from pysimcount.pfcompare import _effts
    S = _effts()
    scheme = {"mo-q6": S.recommended_q6, "mo-q14": S.recommended_q14, "bm-q10": lambda: S.blanes_moan(10),
              "mo-o8-q21": lambda: S.recommended(8, 21)[0],
              "morales-y8m10": lambda: __import__("effts.morales", fromlist=["y8m10"]).y8m10(),
              "morales-m8": lambda: __import__("effts.morales", fromlist=["m8"]).m8()}[other_id]()
    suz = weighted.suzuki_scheme(order)
    return (bch.efficiency(scheme) / bch.efficiency(suz)) ** (1 / order)


def _get(rows, **kw):
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    if len(out) != 1:
        raise LookupError(f"{len(out)} rows match {kw}")
    return out[0]


def _fmt(x):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.3g}"
    return f"{x:,}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return "\n".join(out)


def report(path: Path) -> str:
    data = json.loads(path.read_text())
    rows, meta = data["rows"], data["meta"]
    sizes = sorted({r["n"] for r in rows if r["instance"] == "simcount" and r["r_source"] == "measured"})
    fit_sizes = sorted({r["n"] for r in rows if r["r_source"] == "fit"})
    gbs = meta["config"]["gatebases"]
    names = {r["scheme"]: r for r in rows}
    md = [f"# {meta['config']['name']}", "",
          f"Source: `{path.name}`, git `{meta['provenance']['git_commit'][:10]}`"
          f"{' (dirty tree)' if meta['provenance']['git_dirty'] else ''}, sources sha256 "
          f"`{meta['provenance']['source_sha256'][:12]}`, started {meta['provenance']['started']}.", "",
          "r is measured by brute force for every formula (exact for each instance, not a bound). "
          "Totals are per-step counts × r. 'merged' combines repeated exponentials inside each "
          "commuting block (same unitary, fewer gates), applied identically to every formula.", ""]

    # 1. headline: total cost per order, merged
    for gb in gbs:
        key = "t_total" if gb == "ct" else "rz_total"
        what = "T gates" if gb == "ct" else "rotations (= exponentials)"
        md += [f"## Headline, `{gb}`: total {what}, merged", ""]
        hdr = ["n"]
        for order, suz, others in PAIRS:
            hdr += [f"S{order}"] + [names[o]["scheme"] for o in others] + [f"gain {o}" for o in others]
        body = []
        for n in sizes + fit_sizes:
            src = "measured" if n in sizes else "fit"
            line = [f"{n}" + (" (fit)" if src == "fit" else "")]
            for order, suz, others in PAIRS:
                s = _get(rows, instance="simcount", n=n, scheme=suz, gatebase=gb, merged=True, r_source=src)[key]
                vals = [_get(rows, instance="simcount", n=n, scheme=o, gatebase=gb, merged=True, r_source=src)[key] for o in others]
                line += [s] + vals + [f"{s / v:.2f}×" for v in vals]
            body.append(line)
        md += [table(hdr, body), ""]

    # 2. prediction vs measurement
    md += ["## Does Eff predict the gain? (`cz`, merged, rotations)", "",
           "Predicted = (Eff ratio)^(1/n) from the paper's Eq. 22. Measured = Suzuki rotations ÷ scheme rotations.", ""]
    body = []
    for order, suz, others in PAIRS:
        for o in others:
            pred = _eff_prediction(order, suz, o)
            meas = [_get(rows, instance="simcount", n=n, scheme=suz, gatebase="cz", merged=True, r_source="measured")["rz_total"]
                    / _get(rows, instance="simcount", n=n, scheme=o, gatebase="cz", merged=True, r_source="measured")["rz_total"]
                    for n in sizes]
            body.append([f"order {order}: {o} vs {suz}", f"{pred:.2f}×"] + [f"{m:.2f}×" for m in meas])
    md += [table(["pair", "predicted"] + [f"n={n}" for n in sizes], body), ""]

    # 3. r per scheme
    for gb in gbs:
        md += [f"## Trotter number r, `{gb}`", ""]
        keys = meta["only"] or meta["config"]["schemes"]
        body = []
        for k in keys:
            line = [k]
            for n in sizes + fit_sizes:
                src = "measured" if n in sizes else "fit"
                line.append(_get(rows, instance="simcount", n=n, scheme=k, gatebase=gb, merged=True, r_source=src)["r"])
            body.append(line)
        md += [table(["scheme"] + [f"n={n}" for n in sizes] + [f"n={n} (fit)" for n in fit_sizes], body), ""]
        if gb == "cz":
            emp = [[k] + [_get(rows, instance="simcount", n=n, scheme=k, gatebase="cz", merged=True, r_source="measured")["emp_reference_r"]
                          for n in sizes] for k in keys if k.startswith("suzuki")]
            md += ["simcount's fitted `emp` r for the same Suzuki rows (reference only; it ignores t and eps):", "",
                   table(["scheme"] + [f"n={n}" for n in sizes], emp), ""]

    # 4. merging effect
    n_big = max(sizes)
    md += [f"## Effect of merging at n = {n_big} (`cz`): unmerged ÷ merged", ""]
    body = []
    for k in meta["only"] or meta["config"]["schemes"]:
        u = _get(rows, instance="simcount", n=n_big, scheme=k, gatebase="cz", merged=False, r_source="measured")
        m = _get(rows, instance="simcount", n=n_big, scheme=k, gatebase="cz", merged=True, r_source="measured")
        body.append([k, f"{u['rz_total'] / m['rz_total']:.3f}×", f"{u['cnot_total'] / m['cnot_total']:.3f}×",
                     f"{u['gates_total'] / m['gates_total']:.3f}×"])
    md += [table(["scheme", "rotations", "CNOT", "all gates"], body), ""]

    # 5. fits
    if "fits" in meta:
        md += ["## Power-law fits r = a·n^b (used only for the 'fit' rows)", ""]
        body = [[k, f"{f['a']:.3g}", f"{f['b']:.3f}", ", ".join(f"{n}:{r}" for n, r in f["points"])]
                for k, f in meta["fits"].items()]
        md += [table(["scheme/gatebase", "a", "b", "points (n:r)"], body), ""]

    # 6. paper instance
    prow = [r for r in rows if r["instance"] == "paper" and r["merged"]]
    if prow:
        n, t = prow[0]["n"], prow[0]["t"]
        md += [f"## The paper's instance (fields ±0.1, seed 42), n = {n}, t = {t:g}, merged", ""]
        body = [[r["scheme"], r["gatebase"], r["r"], r["rz_total"], r["gates_total"]] for r in prow]
        md += [table(["scheme", "gatebase", "r", "rotations", "all gates"], body), ""]
    return "\n".join(md)


def main(argv=None):
    path = Path((argv or sys.argv[1:])[0])
    md = report(path)
    (path.parent / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
