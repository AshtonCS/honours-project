"""Hold-out check of the power-law extrapolation of r used by ``pfcompare``.

    python -m plsimcount.pffit results/pf-fitcheck/config.json

``pfcompare`` extrapolates the Trotter number to n = 13, 20 with

    r(n) = ceil(a n^b),   (a, b) = least squares on log r = log a + b log n

fitted to measured r (``pfcompare.fit_power_law``). This module asks how good
that rule is on sizes where r *was* measured: fit on a subset of sizes, predict
a larger held-out size with the same rule, and compare with the measured r.

    relative error = (r_predicted - r_measured) / r_measured

Positive = the fit over-predicts r (and so over-predicts every total gate
count, which is per-step count x r). No new computation: it reads the measured
rows of an existing ``pfcompare`` results.json. Every number is a property of
that one instance (seed, t = n, eps); nothing here is an error model for other
instances.

Gate totals: for ``cz`` the per-step counts do not depend on r, so the relative
error of any cz total equals that of r (up to the ceiling). For ``ct`` the
per-step T count depends weakly on r through eps_rot; that dependence is not
examined here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .pfcompare import fit_power_law, provenance, write_environment


@dataclass
class Check:
    """One hold-out prediction for one (scheme, gatebase)."""
    scheme: str
    family: str
    order: int
    gatebase: str
    fit_sizes: tuple[int, ...]   # sizes n the power law is fitted on
    target: int                  # held-out size n
    a: float                     # r = a n^b
    b: float
    r_predicted: int             # ceil(a target^b), pfcompare's rule
    r_measured: int
    rel_error: float             # (r_predicted - r_measured) / r_measured


def measured_r(rows: list[dict]) -> dict[tuple[str, str], dict[int, int]]:
    """{(scheme, gatebase): {n: r}} from the measured simcount rows.

    r does not depend on ``merged`` (merging only changes gates per step), so
    each (scheme, gatebase, n) appears twice; the two must agree or this raises.
    """
    out: dict[tuple[str, str], dict[int, int]] = {}
    for row in rows:
        if row["instance"] != "simcount" or row["r_source"] != "measured":
            continue
        per = out.setdefault((row["scheme"], row["gatebase"]), {})
        n, r = int(row["n"]), int(row["r"])
        if n in per and per[n] != r:
            raise ValueError(f"two measured r for {row['scheme']}/{row['gatebase']} n={n}: {per[n]} and {r}")
        per[n] = r
    return out


def predict(fit_sizes: list[int], rs: list[int], target: int) -> tuple[float, float, int]:
    """(a, b, ceil(a target^b)): pfcompare's extrapolation rule, unchanged."""
    if len(fit_sizes) < 2:
        raise ValueError("a power law needs at least two sizes")
    if target in fit_sizes:
        raise ValueError(f"target n={target} is one of the fit sizes {fit_sizes}; not a hold-out")
    a, b = fit_power_law(fit_sizes, rs)
    return a, b, math.ceil(a * target ** b)


def run_checks(rows: list[dict], checks: list[dict]) -> list[Check]:
    """Every (fit sizes -> target) check in ``checks`` for every measured
    (scheme, gatebase). Raises if a needed size was not measured."""
    table = measured_r(rows)
    meta = {}
    for row in rows:
        meta[row["scheme"]] = (row["family"], int(row["order"]))
    out: list[Check] = []
    for (scheme, gb), per in table.items():
        for chk in checks:
            fit_sizes = [int(n) for n in chk["fit"]]
            target = int(chk["predict"])
            missing = [n for n in fit_sizes + [target] if n not in per]
            if missing:
                raise ValueError(f"{scheme}/{gb}: no measured r at n={missing}")
            a, b, r_pred = predict(fit_sizes, [per[n] for n in fit_sizes], target)
            r_meas = per[target]
            family, order = meta[scheme]
            out.append(Check(scheme, family, order, gb, tuple(fit_sizes), target, a, b, r_pred, r_meas,
                             (r_pred - r_meas) / r_meas))
    return out


# ------------------------------------------------------------------ output

def markdown_table(results: list[Check], scheme_order: list[str]) -> str:
    """One row per (scheme, gatebase), one column group per check."""
    keys = []
    for c in results:
        k = (c.fit_sizes, c.target)
        if k not in keys:
            keys.append(k)
    head = "| scheme | order | gate set |"
    sep = "|---|---:|---|"
    for fs, tg in keys:
        lbl = "{" + ",".join(map(str, fs)) + "}" + f"->{tg}"
        head += f" {lbl}: pred / meas (rel. err) | b |"
        sep += "---:|---:|"
    lines = [head, sep]
    by = {(c.scheme, c.gatebase, c.fit_sizes, c.target): c for c in results}
    gbs = []
    for c in results:
        if c.gatebase not in gbs:
            gbs.append(c.gatebase)
    for gb in gbs:
        for s in scheme_order:
            cells = []
            order = None
            for fs, tg in keys:
                c = by.get((s, gb, fs, tg))
                if c is None:
                    cells += ["", ""]
                    continue
                order = c.order
                cells.append(f"{c.r_predicted:,} / {c.r_measured:,} ({100 * c.rel_error:+.1f}%)")
                cells.append(f"{c.b:.2f}")
            if order is not None:
                lines.append(f"| {s} | {order} | {gb} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(results: list[Check]) -> str:
    """Per check: max |rel. error| and the signed range, over all schemes and gate sets."""
    lines = ["| check | gate set | min rel. err | max rel. err | max abs rel. err | worst scheme |",
             "|---|---|---:|---:|---:|---|"]
    groups: dict[tuple, list[Check]] = {}
    for c in results:
        groups.setdefault((c.fit_sizes, c.target, c.gatebase), []).append(c)
    for (fs, tg, gb), cs in groups.items():
        errs = [c.rel_error for c in cs]
        worst = max(cs, key=lambda c: abs(c.rel_error))
        lbl = "{" + ",".join(map(str, fs)) + "}" + f"->{tg}"
        lines.append(f"| {lbl} | {gb} | {100 * min(errs):+.1f}% | {100 * max(errs):+.1f}% | "
                     f"{100 * abs(worst.rel_error):.1f}% | {worst.scheme} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out_dir = args.config.parent
    source = (out_dir / cfg["source"]).resolve()
    data = json.loads(source.read_text())
    results = run_checks(data["rows"], cfg["checks"])
    meta = {"config": cfg, "provenance": provenance(),
            "source_results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_provenance": data["meta"]["provenance"]}
    write_environment(out_dir)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": [asdict(c) for c in results]},
                                                     indent=1))
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(results[0])))
        w.writeheader()
        for c in results:
            d = asdict(c)
            d["fit_sizes"] = " ".join(map(str, c.fit_sizes))
            w.writerow(d)
    order = data["meta"]["config"]["schemes"]
    md = ["# Hold-out check of r = ceil(a n^b)", "",
          f"Source: `{cfg['source']}` (sha256 {meta['source_results_sha256'][:12]}...).",
          "rel. err = (predicted - measured) / measured; positive = over-prediction.", "",
          "## Summary", "", summary(results), "", "## Every scheme", "", markdown_table(results, order), ""]
    (out_dir / "table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
