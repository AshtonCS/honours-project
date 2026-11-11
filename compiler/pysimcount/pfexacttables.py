"""Thesis tables for the exact-reference validation (``pfexact`` runs np-exact and np-exact-local).

    python -m pysimcount.pfexacttables results/np-exact/results.json \\
        results/np-exact-local/results.json --out results/np-exact/thesis-tables.tex

Five tables, every number read from the two results files:

exact-summary    one row per check: what it compares, the criterion, cases, passes.
exact-commuting  Part A: commuting ZZ + Z chain against the closed-form unitary.
exact-order      Part B (i) compilation, Part B (ii) global order, Part C local
                 order, side by side per formula; failures marked.
exact-s8         S8's raw single-step errors and pairwise local orders over d,
                 the evidence behind its two failures (no selection rule applied:
                 every grid point from d = 1 down to where the floor sets in).
exact-controls   the four negative controls, each of which must fail.

Fitted "order" is reported throughout as an estimate of p: the global slope of
log error against log r with the sign flipped (Part B), or the local slope of
log error against log d minus one (Part C).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .pfcompare import provenance

NAMES = {"lie-trotter": "Lie-Trotter S1", "suzuki2": "S2", "suzuki4": "S4", "suzuki6": "S6", "suzuki8": "S8",
         "mo-q6": "M\\&O $q=6$", "mo-q14": "M\\&O $q=14$", "bm-q10": "Blanes \\& Moan $q=10$",
         "mo-o8-q21": "M\\&O $q=21$ (order 8)", "morales-y8m10": "Morales Y8m10", "morales-m8": "Morales $m=8$"}
S8_D_MIN = 0.044  # S8 table: show d down to the first two points on the rounding floor


def sci(x: float, digits: int = 1) -> str:
    """$a.b\\times10^{e}$ for LaTeX."""
    if x == 0:
        return "$0$"
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    if digits == 0 and round(m) == 1:
        return f"$10^{{{e}}}$"
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def thousands(k: int) -> str:
    """1400 -> 1{,}400 (the chapter's style)."""
    return f"{k:,}".replace(",", "{,}")


def _table(label: str, caption: str, body: list[str]) -> list[str]:
    return ["\\begin{table}[htbp]", "  \\centering", "  \\small", f"  \\caption{{{caption}}}",
            f"  \\label{{tab:{label}}}", "  \\begin{fitwidth}", *body, "  \\end{fitwidth}", "\\end{table}", ""]


# ------------------------------------------------------------------ tables

def summary_table(ex: dict, loc: dict) -> list[str]:
    A = ex["commuting"]
    fits = ex["noncommuting"]["fits"]
    lf = loc["fits"]
    sizes_a = ex["meta"]["config"]["commuting"]["sizes"]
    rs_a = ex["meta"]["config"]["commuting"]["r"]
    sizes_b = ex["meta"]["config"]["noncommuting"]["sizes"]
    n_ctrl = len(ex["controls"]) + len(loc["controls"])
    ctrl_failed = (int(not ex["controls"]["A1"]["pass"]) + int(not ex["controls"]["B1"]["order_pass"])
                   + sum(int(not c["pass"]) for c in loc["controls"].values()))
    tol = ex["meta"]["config"]["criteria"]["tol_per_exponential"]
    st = ex["meta"]["config"]["criteria"]["slope_tol"]

    def ns(sizes: list[int]) -> str:
        return ", ".join(map(str, sizes))

    rows = [
        ("Exactness, commuting $H$", f"ZZ$+$Z, $n = {ns(sizes_a)}$, $r = {ns(rs_a)}$", "closed form",
         f"error $\\le {sci(tol, 0)[1:-1]}$ per exponential", len(A), sum(x["pass"] for x in A)),
        ("Compilation", f"Heisenberg, $n = {ns(sizes_b)}$, all $r$", "exact exponentials",
         f"$\\|G-F\\| \\le {sci(tol, 0)[1:-1]}$ per exponential", len(fits), sum(x["compile_pass"] for x in fits)),
        ("Global order", f"Heisenberg, $n = {ns(sizes_b)}$", "exact diagonalisation",
         f"$|\\hat p - p| \\le {st}$", len(fits), sum(x["order_pass"] for x in fits)),
        ("Local order", f"Heisenberg, $n = {ns(loc['meta']['config']['local']['sizes'])}$", "exact diagonalisation",
         f"$|\\hat p - p| \\le {loc['meta']['config']['local']['slope_tol']}$", len(lf), sum(x["pass"] for x in lf)),
    ]
    body = ["\\begin{tabular}{llllrr}", "\\hline",
            "Check & system & reference & pass criterion & cases & passed \\\\", "\\hline"]
    for name, sysd, ref, crit, n, p in rows:
        body.append(f"{name} & {sysd} & {ref} & {crit} & {n} & {p} \\\\")
    body += ["\\hline",
             f"Negative controls & broken formulas & as above & must fail & {n_ctrl} & "
             f"{ctrl_failed} failed \\\\", "\\hline", "\\end{tabular}"]
    return _table("exact-summary", "Checks of the compiled circuits against exactly computed unitaries", body)


def commuting_table(ex: dict) -> list[str]:
    cfg = ex["meta"]["config"]
    sizes = cfg["commuting"]["sizes"]
    r_show = min(cfg["commuting"]["r"])
    A = ex["commuting"]
    n_cases = len(sizes) * len(cfg["commuting"]["r"])
    body = ["\\begin{tabular}{lr" + "r" * len(sizes) + "rr}", "\\hline",
            f"Formula & exponentials & \\multicolumn{{{len(sizes)}}}{{c}}{{error at $r = {r_show}$}} & "
            "max error per & passed \\\\",
            f" & per step ($n={sizes[-1]}$) & " + " & ".join(f"$n = {n}$" for n in sizes)
            + " & exponential & \\\\", "\\hline"]
    for k in cfg["schemes"]:
        rows = [x for x in A if x["scheme"] == k]
        at = {x["n"]: x for x in rows if x["r"] == r_show}
        worst = max(x["error_per_exp"] for x in rows)
        passed = sum(x["pass"] for x in rows)
        body.append(f"{NAMES[k]} & {thousands(at[sizes[-1]]['exps_per_step'])} & "
                    + " & ".join(sci(at[n]["error"]) for n in sizes)
                    + f" & {sci(worst)} & {passed}/{n_cases} \\\\")
    tol = cfg["criteria"]["tol_per_exponential"]
    body += ["\\hline",
             f"\\multicolumn{{{4 + len(sizes)}}}{{l}}{{\\footnotesize Spectral norm, no global-phase alignment; "
             f"pass: error $\\le {sci(tol, 0)[1:-1]}$ per exponential applied, over $n = {', '.join(map(str, sizes))}$ "
             f"and $r = {', '.join(map(str, cfg['commuting']['r']))}$.}} \\\\",
             "\\end{tabular}"]
    return _table("exact-commuting",
                  "Commuting chain ($J_x = J_y = 0$): compiled circuit against the closed-form unitary", body)


def order_table(ex: dict, loc: dict) -> list[str]:
    cfg = ex["meta"]["config"]
    sizes = cfg["noncommuting"]["sizes"]
    lsizes = loc["meta"]["config"]["local"]["sizes"]
    g = {(x["scheme"], x["n"]): x for x in ex["noncommuting"]["fits"]}
    lo = {(x["scheme"], x["n"]): x for x in loc["fits"]}

    def cell(val: float | None, ok: bool) -> str:
        s = "--" if val is None else f"{val:.2f}"
        return s if ok else f"{s}$^\\dagger$"

    body = ["\\begin{tabular}{lrr" + "r" * len(sizes) + "r" * len(lsizes) + "}", "\\hline",
            f"Formula & $p$ & max $\\|G-F\\|$ & \\multicolumn{{{len(sizes)}}}{{c}}{{global $\\hat p$}} & "
            f"\\multicolumn{{{len(lsizes)}}}{{c}}{{local $\\hat p$}} \\\\",
            " & & per step & " + " & ".join(f"$n={n}$" for n in sizes) + " & "
            + " & ".join(f"$n={n}$" for n in lsizes) + " \\\\", "\\hline"]
    for k in cfg["schemes"]:
        p = g[(k, sizes[0])]["order"]
        diff = max(g[(k, n)]["max_step_diff"] for n in sizes)
        gl = [cell(g[(k, n)]["slope"], g[(k, n)]["order_pass"]) for n in sizes]
        lc = [cell(lo[(k, n)]["fitted_order"], lo[(k, n)]["pass"]) for n in lsizes]
        body.append(f"{NAMES[k]} & {p} & {sci(diff)} & " + " & ".join(gl) + " & " + " & ".join(lc) + " \\\\")
    c = cfg["criteria"]
    lc_ = loc["meta"]["config"]["local"]
    ncol = 3 + len(sizes) + len(lsizes)
    body += ["\\hline",
             f"\\multicolumn{{{ncol}}}{{p{{14cm}}}}{{\\footnotesize $G$: compiled step; $F$: product of exact "
             f"exponentials. Global $\\hat p$: least-squares slope of $\\log\\|G^r - U\\|$ against $\\log r$, "
             f"points with error in $[{sci(c['slope_window'][0], 0)[1:-1]}, {sci(c['slope_window'][1], 0)[1:-1]}]$, "
             f"at least {c['min_points']}. Local $\\hat p$: slope of $\\log\\|G(d) - e^{{-iHd}}\\|$ against $\\log d$ "
             f"minus one, over the {lc_['n_fit']} smallest $d$ with error at least {lc_['floor_margin']:g} times "
             f"${sci(c['tol_per_exponential'], 0)[1:-1]}$ per exponential. Pass: $|\\hat p - p| \\le {c['slope_tol']}$. "
             f"$^\\dagger$ fails the criterion.}} \\\\",
             "\\end{tabular}"]
    return _table("exact-order", "Compilation error and fitted order against exact diagonalisation", body)


def s8_table(loc: dict, key: str = "suzuki8") -> list[str]:
    sizes = loc["meta"]["config"]["local"]["sizes"]
    pts = {n: sorted((x for x in loc["points"] if x["scheme"] == key and x["n"] == n), key=lambda x: -x["d"])
           for n in sizes}
    ds = [x["d"] for x in pts[sizes[0]] if x["d"] >= S8_D_MIN]
    body = ["\\begin{tabular}{r" + "rr" * len(sizes) + "}", "\\hline",
            "$d$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$n = {n}$}}" for n in sizes) + " \\\\",
            " & " + " & ".join("error & local order to next" for _ in sizes) + " \\\\", "\\hline"]
    for i, d in enumerate(ds):
        cells = []
        for n in sizes:
            a = pts[n][i]
            cells.append(sci(a["error"]))
            if i + 1 < len(ds):
                b = pts[n][i + 1]
                cells.append(f"{math.log(a['error'] / b['error']) / math.log(a['d'] / b['d']) - 1:.2f}")
            else:
                cells.append("")
        body.append(f"{d:.4f} & " + " & ".join(cells) + " \\\\")
    body += ["\\hline",
             f"\\multicolumn{{{1 + 2 * len(sizes)}}}{{p{{11cm}}}}{{\\footnotesize Single-step error "
             f"$\\|G(d) - e^{{-iHd}}\\|$; local order between consecutive $d$ is "
             "$\\log(e_i/e_{i+1})/\\log(d_i/d_{i+1}) - 1$, expected $p = 8$ as $d \\to 0$. "
             "Every grid point is shown down to the onset of the rounding floor; the smaller $d$ on the "
             "grid all sit on the floor.} \\\\", "\\end{tabular}"]
    return _table("exact-s8", "Suzuki S8: single-step error against step size", body)


def controls_table(ex: dict, loc: dict) -> list[str]:
    a1, b1 = ex["controls"]["A1"], ex["controls"]["B1"]
    c1, c2 = list(loc["controls"].values())
    pa = ex["meta"]["config"]["controls"]["perturb"]
    pc = loc["meta"]["config"]["local"]["perturb"]

    def res(passed: bool) -> str:
        return "passed (bad)" if passed else "failed, as required"

    rows = [
        (f"S2, every time $\\times(1 + {sci(pa, 0)[1:-1]})$", "commuting exactness",
         f"error {sci(a1['error'])}", f"$\\le {sci(a1['tol'])[1:-1]}$", res(a1["pass"])),
        ("Lie-Trotter, claimed order 2", "global order", f"$\\hat p = {b1['slope']:.2f}$",
         "$|\\hat p - 2| \\le 0.3$", res(b1["order_pass"])),
        ("Lie-Trotter, claimed order 2", "local order", f"$\\hat p = {c1['fitted_order']:.2f}$",
         "$|\\hat p - 2| \\le 0.3$", res(c1["pass"])),
        (f"S4, every time $\\times(1 + {sci(pc, 0)[1:-1]})$", "local order", f"$\\hat p = {c2['fitted_order']:.2f}$",
         "$|\\hat p - 4| \\le 0.3$", res(c2["pass"])),
    ]
    body = ["\\begin{tabular}{lllll}", "\\hline", "Deliberately broken formula & check & measured & criterion & result \\\\",
            "\\hline"]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body += ["\\hline", "\\end{tabular}"]
    return _table("exact-controls", f"Negative controls at $n = {a1['n']}$: each check must reject a broken formula",
                  body)


def compact_table(ex: dict, loc: dict) -> list[str]:
    """One table for a short acknowledgement in the Validation section: each
    check, cases passed, and the worst case or the failures by name."""
    A = ex["commuting"]
    fits = ex["noncommuting"]["fits"]
    lf = loc["fits"]
    crit = ex["meta"]["config"]["criteria"]

    def failed(rows: list[dict], flag: str, val: str) -> str:
        bad = [x for x in rows if not x[flag]]
        return "; ".join(f"{NAMES[x['scheme']]} $n={x['n']}$ ($\\hat p = {x[val]:.2f}$)" for x in bad)

    worst_exp = max(x["error_per_exp"] for x in A)
    worst_gf = max(x["max_step_diff"] for x in fits)
    s8 = sorted((x for x in loc["points"] if x["scheme"] == "suzuki8" and x["n"] == 4), key=lambda x: -x["d"])
    pair = []
    for a, b in zip(s8, s8[1:]):
        pair.append(math.log(a["error"] / b["error"]) / math.log(a["d"] / b["d"]) - 1)
    s8_peak = max(pair)
    n_ctrl = len(ex["controls"]) + len(loc["controls"])
    ctrl_failed = (int(not ex["controls"]["A1"]["pass"]) + int(not ex["controls"]["B1"]["order_pass"])
                   + sum(int(not c["pass"]) for c in loc["controls"].values()))
    rows = [
        ("Commuting $H$: circuit is exact", "closed form", f"{sum(x['pass'] for x in A)}/{len(A)}",
         f"max error {sci(worst_exp)} per exponential (rounding)"),
        ("Compiled step = exact exponentials", "exact exponentials",
         f"{sum(x['compile_pass'] for x in fits)}/{len(fits)}", f"max $\\|G-F\\|$ {sci(worst_gf)}"),
        ("Error $\\propto r^{-p}$ (global order)", "exact diagonalisation",
         f"{sum(x['order_pass'] for x in fits)}/{len(fits)}", "fail: " + failed(fits, "order_pass", "slope")),
        ("Error $\\propto d^{p+1}$ (local order)", "exact diagonalisation",
         f"{sum(x['pass'] for x in lf)}/{len(lf)}",
         "fail: " + failed(lf, "pass", "fitted_order") + f"; pairwise order reaches {s8_peak:.2f} "
         "before the rounding floor"),
        ("Broken formulas rejected", "as above", f"{ctrl_failed}/{n_ctrl}", "all four controls fail, as required"),
    ]
    body = ["\\begin{tabular}{llrp{7.5cm}}", "\\hline", "Check & reference & passed & worst case \\\\", "\\hline"]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body += ["\\hline",
             f"\\multicolumn{{4}}{{p{{15cm}}}}{{\\footnotesize All eleven formulas; Heisenberg chain (NumPy seed "
             f"{ex['meta']['config']['seed']}, $t = n$), $n = 4, 6$; commuting case $J_x = J_y = 0$, $n = 4, 6, 8$. "
             f"Order passes if $|\\hat p - p| \\le {crit['slope_tol']}$; criteria fixed before each run. The local "
             "check was added after the global one failed, which is reported unchanged.} \\\\",
             "\\end{tabular}"]
    return _table("exact-compact", "Compiled circuits against exactly computed unitaries", body)


def tables_tex(ex: dict, loc: dict) -> str:
    prov = provenance()
    head = [f"% Generated by pysimcount.pfexacttables; git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}; "
            "do not edit, regenerate instead.",
            f"% Sources: np-exact (git_dirty={ex['meta']['provenance']['git_dirty']}), "
            f"np-exact-local (git_dirty={loc['meta']['provenance']['git_dirty']}).",
            "% Needs the fitwidth environment defined at the top of chapter 4.", ""]
    out = head + summary_table(ex, loc) + commuting_table(ex) + order_table(ex, loc) + s8_table(loc) \
        + controls_table(ex, loc)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exact", type=Path, help="np-exact results.json")
    ap.add_argument("local", type=Path, help="np-exact-local results.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ex = json.loads(args.exact.read_text())
    loc = json.loads(args.local.read_text())
    if "local" in ex["meta"]["config"] or "local" not in loc["meta"]["config"]:
        raise ValueError("expected the np-exact results first and the np-exact-local results second")
    args.out.write_text(tables_tex(ex, loc))
    compact = args.out.with_name(args.out.stem + "-compact.tex")
    compact.write_text("\n".join(tables_tex(ex, loc).splitlines()[:4] + compact_table(ex, loc)))
    print(f"wrote {args.out} and {compact}")


if __name__ == "__main__":
    main()
