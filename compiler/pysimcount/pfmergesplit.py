"""Where Childs et al.'s optimiser saving comes from: exponential merging vs the optimiser.

    python -m pysimcount.pfmergesplit results/childs-fig13-split/config.json [--render]

Childs et al. (2018, arXiv:1711.10980v1, SI I.3 and Fig. 13) report that
their optimiser cuts the CNOT count of the higher-order Suzuki circuits by
about 33%. Their input is simcount's *unmerged* step. For one Suzuki
formula at one n, this module compiles that step three ways and counts gates
per Trotter step:

    U  unmerged step, simcount template (Childs et al.'s "before")
    M  the same step after ``weighted.merge`` (adjacent exponentials of the
       same commuting block combined; this project's baseline)
    O  the Light-optimiser reconstruction of U, per-step marginal
       opt(3 steps) - opt(2 steps) (``pffig13.optimise_step``; Childs et al.'s
       "after" without the one-off edge term)

and splits the optimiser's saving U - O into

    (U - M) / U   removed by merging alone (no optimiser needed)
    (M - O) / U   removed only by the optimiser (the XX layer shared across a
                  step boundary, which ``merge`` cannot see inside one step)

It also optimises M (marginal O_M) and raises if O_M != O, so the split does
not depend on which input the optimiser is given.

Validation: for the orders with published sample circuits (4, 6; empirical
bound, Eq. 147 r) the published pre-optimisation counts must equal r x U
(else it raises), and the published saving 1 - after/before is drawn beside
ours. The published totals include the step-edge term, so they differ from
the marginal split by O(1/r); the table shows both.

Counts do not depend on the fields (test_pffig13 checks this on two seeds),
and with periodic boundaries every count is a multiple of n. Every case is
cached under ``cases/`` (delete it to recompute).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import bounds, pffigure
from .nam import samples
from .pfcompare import formula_registry, heisenberg, numpy_fields, provenance, write_environment
from .pffig13 import EPS, ORDER_NAMES, PUBLISHED, optimise_step
from .weighted import build_step_from_schedule, merge

METRICS = ("CNOT", "Rz")
TICK = {1: "$S_1$", 2: "$S_2$", 4: "$S_4$", 6: "$S_6$", 8: "$S_8$"}


def run_case(scheme: str, n: int, seed: int, verify_max_n: int) -> dict:
    """U, M, O and O_M per step for one Suzuki formula at size n."""
    spec = formula_registry()[scheme]
    if spec.family != "suzuki":
        raise ValueError(f"{scheme}: Childs et al.'s comparison covers the Suzuki formulas only")
    # ASSUMPTION: fields from NumPy seed ``seed``; counts do not depend on them
    # (test_pffig13 checks two seeds). (unconfirmed for every n)
    H = heisenberg(n, numpy_fields(n, seed))
    r = bounds.empirical(spec.order, H, float(n), EPS)  # Eq. 147, as in the replica
    sched = spec.schedule(H, float(n) / r)
    unmerged = build_step_from_schedule(H, sched, "simcount")
    merged = build_step_from_schedule(H, merge(H, sched), "simcount")
    o_u = optimise_step(unmerged, n, r, verify_max_n, f"{scheme} n={n} unmerged")
    o_m = optimise_step(merged, n, r, verify_max_n, f"{scheme} n={n} merged")
    case = {"scheme": scheme, "order": spec.order, "n": n, "seed": seed, "r": r,
            "U": o_u["before_per_step"], "M": o_m["before_per_step"],
            "O": o_u["marginal_per_step"], "O_M": o_m["marginal_per_step"],
            "edge": o_u["edge"], "edge_M": o_m["edge"],
            "verified": o_u["verified"] and o_m["verified"],
            "verify_state_err": max(o_u["verify_state_err"] or 0.0, o_m["verify_state_err"] or 0.0),
            "optimiser_seconds": o_u["optimiser_seconds"] + o_m["optimiser_seconds"]}
    for m in METRICS:
        if case["O_M"][m] != case["O"][m]:
            raise ValueError(f"{scheme} n={n}: optimising the merged step gives {m} {case['O_M'][m]}/step, "
                             f"the unmerged step {case['O'][m]}/step")
    case["split"] = {m: {"merging": (case["U"][m] - case["M"][m]) / case["U"][m],
                         "optimiser_only": (case["M"][m] - case["O"][m]) / case["U"][m]} for m in METRICS}
    if spec.order in PUBLISHED and n in samples.SIZES:
        pair = samples.load(PUBLISHED[spec.order], n)
        pre, post = pair.pre.total_counts(), pair.post.total_counts()
        pub = {"before": {m: pre.get(m, 0) for m in METRICS}, "after": {m: post.get(m, 0) for m in METRICS}}
        for m in METRICS:
            if pub["before"][m] != r * case["U"][m]:
                raise ValueError(f"{PUBLISHED[spec.order]} n={n}: published {m} before {pub['before'][m]:,} "
                                 f"!= r x U = {r * case['U'][m]:,}")
        case["published"] = pub
        case["published_saving"] = {m: 1 - pub["after"][m] / pub["before"][m] for m in METRICS}
        case["our_total_saving"] = {m: 1 - (r * case["O"][m] + case["edge"][m]) / (r * case["U"][m])
                                    for m in METRICS}
    return case


def table_md(cases: list[dict], metric: str) -> str:
    lines = [f"| formula | n | r | U/n | M/n | O/n | merging | optimiser only | total (marginal) | "
             "ours, total at r | published, total at r |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        s, n = c["split"][metric], c["n"]
        ours = f"{c['our_total_saving'][metric]:.2%}" if "published" in c else "--"
        pub = f"{c['published_saving'][metric]:.2%}" if "published" in c else "--"
        lines.append(f"| {ORDER_NAMES[c['order']]} | {n} | {c['r']:,} | {c['U'][metric] / n:g} | "
                     f"{c['M'][metric] / n:g} | {c['O'][metric] / n:g} | {s['merging']:.1%} | "
                     f"{s['optimiser_only']:.1%} | {s['merging'] + s['optimiser_only']:.1%} | {ours} | {pub} |")
    return "\n".join(lines)


def figure_tex(cases: list[dict], metric: str, header: list[str]) -> str:
    """Stacked bars per order: merging (dark) + optimiser only (light), as a
    percentage of the unmerged step; published savings as open diamonds."""
    ticks = ",".join(f"S{c['order']}" for c in cases)
    label = {"CNOT": "CNOT", "Rz": "$R_z$"}[metric]
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{axis}[", "    width=10cm, height=7cm, ybar stacked, bar width=18pt,",
        f"    symbolic x coords={{{ticks}}}, xtick=data,",
        "    xticklabels={" + ",".join(TICK[c["order"]] for c in cases) + "},",
        f"    xlabel={{Suzuki formula}}, ylabel={{{label} saving (\\% of unmerged step)}},",
        "    ymin=0, ymax=40, ymajorgrids=true, enlarge x limits=0.15,",
        "    legend pos=outer north east, legend cell align=left, legend style={font=\\footnotesize},",
        "  ]",
        "    \\addplot[fill=red!70, draw=red!50!black, area legend] coordinates {"]
    out += [f"      (S{c['order']},{100 * c['split'][metric]['merging']:.4f})" for c in cases]
    out += ["    };", "    \\addlegendentry{merging repeated exponentials}",
            "    \\addplot[fill=red!20, draw=red!50!black, area legend] coordinates {"]
    out += [f"      (S{c['order']},{100 * c['split'][metric]['optimiser_only']:.4f})" for c in cases]
    out += ["    };", "    \\addlegendentry{optimiser only (step boundary)}"]
    pubs = [c for c in cases if "published" in c]
    for c in pubs:
        # an explicit diamond path: the 'diamond' node shape needs a TikZ library the preamble lacks
        out.append(f"    \\filldraw[draw=black, fill=white] (axis cs:S{c['order']},"
                   f"{100 * c['published_saving'][metric]:.4f}) ++(0pt,2.8pt) -- ++(2.2pt,-2.8pt) "
                   "-- ++(-2.2pt,-2.8pt) -- ++(-2.2pt,2.8pt) -- cycle;")
    if pubs:
        out += ["    \\addlegendimage{line legend, only marks, mark=diamond*, mark size=2.2pt, "
                "mark options={draw=black, fill=white}}",
                "    \\addlegendentry{simcount published circuits, $n = " + str(pubs[0]["n"]) + "$}"]
    if metric == "CNOT":
        out += [f"    \\draw[dashed, black] (rel axis cs:0,{100 / 3 / 40:.6f}) -- (rel axis cs:1,{100 / 3 / 40:.6f});",
                "    \\addlegendimage{line legend, dashed, black}", "    \\addlegendentry{33\\% (Childs et al.)}"]
    out += ["  \\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out = args.config.parent / cfg.get("out", ".")
    n = cfg["n"]
    if n not in samples.SIZES:
        raise ValueError(f"n must be a published sample size {samples.SIZES}, got {n}")
    cache = out / "cases"
    cache.mkdir(parents=True, exist_ok=True)
    cases = []
    for scheme in cfg["schemes"]:
        f = cache / f"{scheme}-n{n}.json"
        c = json.loads(f.read_text()) if f.exists() else None
        if c is not None and (c["seed"], c.get("verify_max_n")) != (cfg["seed"], cfg["verify_max_n"]):
            raise ValueError(f"cached {f.name} was made with another seed or verify_max_n; delete cases/ to rerun")
        if c is None:
            c = run_case(scheme, n, cfg["seed"], cfg["verify_max_n"]) | {"verify_max_n": cfg["verify_max_n"]}
            f.write_text(json.dumps(c, indent=1) + "\n")
        cases.append(c)
        s = c["split"]["CNOT"]
        print(f"{scheme:12s} CNOT/n U {c['U']['CNOT'] / n:g} M {c['M']['CNOT'] / n:g} O {c['O']['CNOT'] / n:g}  "
              f"merging {s['merging']:.1%} + optimiser {s['optimiser_only']:.1%}"
              + (f"  published {c['published_saving']['CNOT']:.2%}" if "published" in c else ""), flush=True)
    cases.sort(key=lambda c: c["order"])
    prov = provenance()
    meta = {"config": cfg, "provenance": prov}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=1) + "\n")
    write_environment(out)
    md = [f"# Optimiser saving split into merging and optimiser-only, n = {n}\n",
          f"Written by `python -m pysimcount.pfmergesplit {args.config}`. Per step: U unmerged (simcount template), "
          "M after `weighted.merge`, O Light reconstruction of U (marginal). Percentages of U. "
          "O_M (optimised merged step) equals O for every row (checked). 'total at r' includes the edge term.\n"]
    for m in METRICS:
        md += [f"## {m}\n", table_md(cases, m), ""]
    (out / "table.md").write_text("\n".join(md))
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    names = []
    for m in METRICS:
        name = f"merge-split-{m.lower()}"
        header = ["Generated by pysimcount.pfmergesplit -- do not edit; regenerate instead.",
                  f"figure: {name} -- Childs et al. 2018 optimiser saving ({m}) split into exponential merging "
                  "and optimiser-only, Suzuki formulas",
                  f"metric={m.lower()} gatebase=cz; n = {n}, seed {cfg['seed']}; r_method=emp (Eq. 147, as in Fig. 13); "
                  "per-step counts, % of the unmerged simcount step",
                  f"optimised = Light reconstruction, marginal per step; verified n <= {cfg['verify_max_n']}",
                  "formulas: " + ", ".join(formula_registry()[c["scheme"]].name for c in cases),
                  f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
        (figs / f"{name}.tex").write_text(figure_tex(cases, m, header))
        names.append(name)
    (figs / "all.tex").write_text(pffigure.PREAMBLE + "".join(
        f"\\section*{{\\texttt{{{s}.tex}}}}\n\\input{{{s}.tex}}\n\\clearpage\n" for s in names) + "\\end{document}\n")
    if args.render:
        print(f"rendered {pffigure.render(figs)}")


if __name__ == "__main__":
    main()
