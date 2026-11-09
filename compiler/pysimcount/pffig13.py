"""Replica of Childs et al. (2018) Fig. 13 (left): CNOT count against n,
before and after circuit optimisation, Suzuki orders 1-8, empirical bound.

    python -m pysimcount.pffig13 results/childs-fig13/config.json [--render]

Source: Childs, Maslov, Nam, Ross, Su, "Toward the first quantum simulation
with quantum speedup", arXiv:1711.10980v1, SI I.3 and Fig. 13 (p. 57).
Their plot shows, for the first- to eighth-order product formulas with the
empirical bound, the CNOT count over Clifford+Rz before (diamonds) and after
(circles) the Nam et al. optimiser, at the sizes of simcount's published
sample circuits, n = 13, 16, 20, 25, 32, 40, 50, 63, 79, 100.

How each point is made, deliberately on their footing rather than ours:

* r is simcount's ``emp`` bound, i.e. Childs et al. Eq. 147
  (``bounds.empirical``), not this project's measured r. It depends on n
  only, so the plotted counts do not depend on the random fields.
* The step is the *unmerged* step with simcount's gate template, which is
  what simcount exports and what their optimiser was given.
* Before = CNOT per step x r. This is checked to equal the published
  pre-optimisation circuit exactly, for pf4emp and pf6emp at every size
  (``nam.samples``); a mismatch raises.
* After = our reconstruction of the Nam et al. Light optimiser
  (``pysimcount.nam``; see nam/README.md) on chains of 2 and 3 steps:
  marginal = opt(3) - opt(2), edge = opt(2) - 2 x marginal, and the
  optimised r-step circuit is taken as r x marginal + edge. This differs from
  ``pfoptimised``, which reports r x marginal and drops the edge term: here
  the totals are compared with published circuits that contain their edges.
  The published optimised circuits for pf4emp and pf6emp are compared and the
  ratio reported, not enforced (the reconstruction is known to recover less
  of the Rz and total reduction than the original).
* Soundness: for n <= ``verify_max_n`` every optimised chain is checked
  against the unoptimised chain on random states (``nam.verify.equivalent``).
  Larger chains are optimised but not verified, and the table says so.

Every (order, n) case is cached under ``cases/`` so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import bounds, resources
from .circuit import simulate
from .nam import Circuit, bridge, optimise, samples, verify
from .pfcompare import formula_registry, heisenberg, numpy_fields, provenance, write_environment

METRICS = ("CNOT", "Rz", "total")
ORDER_NAMES = {1: "First order", 2: "Second order", 4: "Fourth order", 6: "Sixth order", 8: "Eighth order"}
# Childs et al. Fig. 13: red outline on every marker, fill fading from solid (first order)
# to open (eighth order).
ORDER_SHADE = {1: "red", 2: "red!75", 4: "red!45", 6: "red!20", 8: "white"}
# y ranges read off Childs et al. Fig. 13 (arXiv:1711.10980v1 p. 57), so the two can be set side by side.
YRANGE = {"CNOT": (3e4, 5e10), "Rz": (3e4, 2.5e10)}
PUBLISHED = {4: "pf4emp", 6: "pf6emp"}  # orders with published sample circuits (empirical bound)
EPS = 1e-3  # Childs et al. Eq. 147 is fitted at eps = 1e-3; bounds.empirical ignores eps anyway


def _counts(c: Circuit) -> dict[str, int]:
    k = c.counts()
    return {"CNOT": k.get("CNOT", 0), "Rz": k.get("Rz", 0), "total": sum(k.values())}


def optimise_step(tape, n: int, r: int, verify_max_n: int, label: str) -> dict:
    """Before/after counts of r repetitions of one step tape (cz).

    Before = per-step count x r. After = r x marginal + edge, with
    marginal = opt(3 steps) - opt(2 steps) and edge = opt(2) - 2 x marginal
    (see the module docstring). Chains are checked on random states for
    n <= verify_max_n; an optimiser that changes the unitary raises."""
    step = bridge.from_tape(tape, n)
    before = _counts(step)
    s = resources.summary(resources.count(tape))
    if (before["CNOT"], before["Rz"], before["total"]) != (s["CNOT"], s["Rz"], s["Total excl. omega"]):
        raise ValueError(f"nam circuit counts {before} disagree with the tape's {s}")
    t0 = time.perf_counter()
    opt, chains = {}, {}
    for k in (2, 3):
        chain = Circuit(list(step.gates) * k, n)
        res = optimise(chain)
        opt[k] = _counts(res.circuit)
        chains[k] = (chain, res.circuit)
    seconds = round(time.perf_counter() - t0, 1)
    marginal = {m: opt[3][m] - opt[2][m] for m in METRICS}
    edge = {m: opt[2][m] - 2 * marginal[m] for m in METRICS}
    verified, worst = n <= verify_max_n, None
    if verified:
        worst = 0.0
        for k, (a, b) in chains.items():
            ok, err = verify.equivalent(a, b)
            if not ok:
                raise ArithmeticError(f"optimiser changed the unitary ({label}, {k} steps, err {err:.2e})")
            worst = max(worst, err)
    return {"r": r, "before_per_step": before, "marginal_per_step": marginal, "edge": edge,
            "before_total": {m: before[m] * r for m in METRICS},
            "after_total": {m: marginal[m] * r + edge[m] for m in METRICS},
            "optimiser_seconds": seconds, "verified": verified, "verify_state_err": worst}


def run_case(order: int, n: int, seed: int, verify_max_n: int) -> dict:
    """Before/after counts for one (order, n)."""
    # ASSUMPTION: the fields are NumPy seed ``seed``; neither r (Eq. 147
    # depends on n only) nor the gate counts depend on them, which
    # test_pffig13 checks on two seeds. (unconfirmed for every n)
    H = heisenberg(n, numpy_fields(n, seed))
    algo = f"pf{order}emp"
    sim = simulate(H, algo, float(n), EPS)  # unmerged step, simcount template
    if sim.r != bounds.empirical(order, H, float(n), EPS):
        raise ValueError(f"{algo} n={n}: r {sim.r} is not the Eq. 147 value")
    case = {"order": order, "n": n, "seed": seed, "fields": [float(h) for h in numpy_fields(n, seed)],
            **optimise_step(sim.step, n, sim.r, verify_max_n, f"{algo} n={n}")}
    if order in PUBLISHED:
        pair = samples.load(PUBLISHED[order], n)
        pre, post = pair.pre.total_counts(), pair.post.total_counts()
        pub = {"before": {"CNOT": pre.get("CNOT", 0), "Rz": pre.get("Rz", 0), "total": sum(pre.values())},
               "after": {"CNOT": post.get("CNOT", 0), "Rz": post.get("Rz", 0), "total": sum(post.values())}}
        for m in ("CNOT", "Rz"):
            if case["before_total"][m] != pub["before"][m]:
                raise ValueError(f"{PUBLISHED[order]} n={n}: our unoptimised {m} {case['before_total'][m]:,} "
                                 f"!= published {pub['before'][m]:,}")
        case["published"] = pub
    return case


def table_md(cases: list[dict], metric: str) -> str:
    lines = [f"| order | n | r | {metric}/step before | after (marginal) | edge | total before | total after | "
             "reduction | published before | published after | ours / published after | verified |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in cases:
        b, a = c["before_total"][metric], c["after_total"][metric]
        p = c.get("published")
        pb = f"{p['before'][metric]:,}" if p else "--"
        pa = f"{p['after'][metric]:,}" if p else "--"
        ratio = f"{a / p['after'][metric]:.4f}" if p and p["after"][metric] else "--"
        lines.append(f"| {c['order']} | {c['n']} | {c['r']:,} | {c['before_per_step'][metric]:,} | "
                     f"{c['marginal_per_step'][metric]:,} | {c['edge'][metric]:,} | {b:,} | {a:,} | "
                     f"{1 - a / b:.1%} | {pb} | {pa} | {ratio} | {'yes' if c['verified'] else 'no'} |")
    return "\n".join(lines)


def figure_tex(cases: list[dict], header: list[str], metric: str = "CNOT") -> str:
    """Childs et al. Fig. 13 layout: diamonds before, circles after, one shade per order."""
    ylabel = {"CNOT": "CNOT gate count", "Rz": "$R_z$ gate count"}[metric]
    ymin, ymax = YRANGE[metric]
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm, height=8.3cm,",
        "    xlabel={System size}, " f"ylabel={{{ylabel}}},", "    xmin=10, xmax=110,",
        f"    ymin={ymin:g}, ymax={ymax:g},",
        "    xtick={10,20,30,50,70,100}, xticklabels={10,20,30,50,70,100},",
        "    ymajorgrids=true, yminorgrids=true,", "    legend pos=north west, legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for order in sorted({c["order"] for c in cases}):
        ser = sorted((c["n"], c["before_total"][metric], c["after_total"][metric]) for c in cases if c["order"] == order)
        shade = ORDER_SHADE[order]
        out.append(f"    \\addplot[only marks, mark=*, mark size=2pt, draw=red, fill={shade}] coordinates {{")
        out += [f"      ({n},{a})" for n, _, a in ser]
        out += ["    };", f"    \\addlegendentry{{{ORDER_NAMES[order]}}}"]
        out.append(f"    \\addplot[only marks, mark=diamond*, mark size=2pt, draw=red, fill={shade}, "
                   "forget plot] coordinates {")
        out += [f"      ({n},{b})" for n, b, _ in ser]
        out += ["    };"]
    # As in Childs et al., the legend names orders only; diamonds = before, circles = after
    # is left to the caption.
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out = args.config.parent
    sizes, orders = cfg["sizes"], cfg["orders"]
    if sizes != list(samples.SIZES):
        raise ValueError(f"sizes must be the published sample sizes {samples.SIZES}, got {sizes}")
    unknown = [o for o in orders if o not in ORDER_NAMES]
    if unknown:
        raise ValueError(f"orders {unknown} have no Eq. 147 fit")
    meta = {"config": cfg, "provenance": provenance()}
    write_environment(out)
    cache = out / "cases"
    cache.mkdir(exist_ok=True)
    cases = []
    # Cheap cases first, so a partial run already covers every order at small n.
    for n in sizes:
        for order in orders:
            f = cache / f"pf{order}emp-n{n}.json"
            if f.exists():
                case = json.loads(f.read_text())
            else:
                case = run_case(order, n, cfg["seed"], cfg["verify_max_n"])
                f.write_text(json.dumps(case, indent=1) + "\n")
            cases.append(case)
            print(f"pf{order}emp n={n:3d} r={case['r']:>9,} CNOT {case['before_total']['CNOT']:>14,} -> "
                  f"{case['after_total']['CNOT']:>14,}  ({case['optimiser_seconds']}s, "
                  f"verified={case['verified']})", flush=True)
    cases.sort(key=lambda c: (c["order"], c["n"]))
    (out / "results.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=1) + "\n")
    prov = meta["provenance"]
    md = ["# Childs et al. (2018) Fig. 13 replica: before/after Nam Light optimisation", "",
          f"Written by `python -m pysimcount.pffig13 {args.config}`. r = Eq. 147 (simcount `emp`), unmerged "
          f"simcount step, cz. After = our Light reconstruction, r x marginal + edge. 'published' = simcount's "
          f"sample circuits (pf4emp, pf6emp). Verified = random-state equivalence check run (n <= "
          f"{cfg['verify_max_n']}).", ""]
    for m in METRICS:
        md += [f"## {m}", "", table_md(cases, m), ""]
    (out / "table.md").write_text("\n".join(md))
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    for metric, stem in (("CNOT", "fig13-cnot"), ("Rz", "fig13-rz")):
        header = ["Generated by pysimcount.pffig13 -- do not edit; regenerate instead.",
                  f"figure: {stem} -- Replica of Childs et al. 2018 Fig. 13 "
                  f"{'left' if metric == 'CNOT' else 'right'}: {metric} before/after optimisation, Suzuki orders, "
                  "empirical bound (Eq. 147)",
                  f"metric={metric.lower()} gatebase=cz unmerged (simcount template); r_method=emp: r = Eq. 147 "
                  "(Childs et al.'s power-law fit on n = 5-12, extrapolated to these n), not our measured r",
                  "formulas: " + ", ".join(formula_registry()[k].name for k in
                                          ("lie-trotter", "suzuki2", "suzuki4", "suzuki6", "suzuki8")),
                  f"after = pysimcount.nam Light reconstruction, r x marginal + edge; verified for n <= "
                  f"{cfg['verify_max_n']} only",
                  f"before equals simcount's published pre-optimisation circuits (pf4emp, pf6emp) at every n",
                  f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
        (fig / f"{stem}.tex").write_text(figure_tex(cases, header, metric))
    from .pffigure import PREAMBLE
    (fig / "all.tex").write_text(PREAMBLE + "".join(
        f"\\section*{{\\texttt{{{s}.tex}}}}\n\\input{{{s}.tex}}\n\\clearpage\n" for s in ("fig13-cnot", "fig13-rz"))
        + "\\end{document}\n")
    print("\n".join(md))
    if args.render:
        from .pffigure import render
        print(f"rendered {render(fig)}")


if __name__ == "__main__":
    main()
