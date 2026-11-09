"""Extension of the Childs et al. (2018) Fig. 13 replica: every product
formula in this project, before and after circuit optimisation, CNOT and Rz,
at the same sizes n = 13, 16, 20, 25, 32, 40, 50, 63, 79, 100.

    python -m pysimcount.pffig13ext results/childs-fig13-extended/config.json [--render]

The replica (``pffig13``) takes r from Childs et al. Eq. 147, which exists
for the Suzuki formulas only. Here every formula, Suzuki included, takes r
from this project's measured r instead, so all points share one r source
(decided by the user, 2026-10-29):

* r(n) = ceil(a n^b), with (a, b) the least-squares power law
  (``pfcompare.fit_power_law``) through the *mean* measured r over the
  config's seeds at the config's ``fit_sizes``. Measured r is the
  fastempirical window rule at eps = 1e-3, t = n (np-main rows,
  ``r_source == "measured"``, cz). Every plotted size is an extrapolation
  beyond the largest fitted n.
* The step is the unmerged step (simcount template) of the formula's own
  schedule, as in the replica; before/after are ``pffig13.optimise_step``,
  so the optimiser, the edge term and the n <= verify_max_n check are
  identical to the replica's.
* For the Suzuki formulas the table also gives the Eq. 147 r, so the effect
  of the r source can be read off against the replica.

Markers follow Fig. 13 (diamonds before, circles after); colour names the
formula. Every (scheme, n) case is cached under ``cases/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from . import bounds, pffigure
from .pfcompare import fit_power_law, formula_registry, heisenberg, numpy_fields, provenance, write_environment
from .pffig13 import EPS, METRICS, YRANGE, optimise_step
from .nam import samples
from .weighted import build_step_from_schedule

# Draw colour and fill per formula. Suzuki keeps the replica's reds (red
# outline, fill fading with order); the others get one hue each, filled at 40%.
STYLE = {
    "lie-trotter": ("red", "red"), "suzuki2": ("red", "red!75"), "suzuki4": ("red", "red!45"),
    "suzuki6": ("red", "red!20"), "suzuki8": ("red", "white"),
    "mo-q6": ("blue", "blue!40"), "mo-q14": ("cyan!70!black", "cyan!40"),
    "mo-o8-q21": ("blue!50!black", "blue!50!black!40"), "bm-q10": ("green!50!black", "green!50!black!40"),
    "morales-y8m10": ("orange", "orange!40"), "morales-m8": ("violet", "violet!40"),
}


def mean_measured_r(run_dirs: list[Path], schemes: list[str], fit_sizes: list[int],
                    seeds: list[int]) -> dict[str, dict[int, float]]:
    """{scheme: {n: mean measured r over seeds}} at the fit sizes.

    Reads every run directory's results.csv (rows with r_source == measured,
    gatebase cz; r does not depend on merging). Raises if a seed is missing
    at a fit size, or if two files disagree on one (scheme, n, seed)."""
    r: dict[tuple[str, int, int], int] = {}
    for d in run_dirs:
        seed = int(d.name.removeprefix("seed"))
        with open(d / "results.csv") as f:
            for row in csv.DictReader(f):
                if row["r_source"] != "measured" or row["gatebase"] != "cz" or row["scheme"] not in schemes:
                    continue
                key = (row["scheme"], int(row["n"]), seed)
                if key in r and r[key] != int(row["r"]):
                    raise ValueError(f"conflicting measured r for {key}: {r[key]} vs {row['r']} ({d})")
                r[key] = int(row["r"])
    out: dict[str, dict[int, float]] = {}
    for scheme in schemes:
        out[scheme] = {}
        for n in fit_sizes:
            missing = [s for s in seeds if (scheme, n, s) not in r]
            if missing:
                raise ValueError(f"{scheme} n={n}: no measured r for seeds {missing}")
            out[scheme][n] = math.fsum(r[(scheme, n, s)] for s in seeds) / len(seeds)
    return out


def run_case(scheme: str, n: int, r: int, seed: int, verify_max_n: int) -> dict:
    """Before/after counts for one (scheme, n) at the given r."""
    spec = formula_registry()[scheme]
    # ASSUMPTION: the fields are NumPy seed ``seed``; the counts do not depend
    # on them (checked in test_pffig13ext on two seeds). (unconfirmed for every n)
    H = heisenberg(n, numpy_fields(n, seed))
    tape = build_step_from_schedule(H, spec.schedule(H, float(n) / r), "simcount")  # unmerged
    case = {"scheme": scheme, "name": spec.name, "order": spec.order, "n": n, "seed": seed,
            **optimise_step(tape, n, r, verify_max_n, f"{scheme} n={n}")}
    if spec.family == "suzuki":
        case["r_eq147"] = bounds.empirical(spec.order, H, float(n), EPS)
    return case


def table_md(cases: list[dict], metric: str) -> str:
    lines = [f"| formula | n | r (our fit) | r (Eq. 147) | {metric}/step before | after (marginal) | edge | "
             "total before | total after | reduction | verified |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in cases:
        b, a = c["before_total"][metric], c["after_total"][metric]
        eq = f"{c['r_eq147']:,}" if "r_eq147" in c else "--"
        lines.append(f"| {pffigure.latex_escape(c['name'])} | {c['n']} | {c['r']:,} | {eq} | "
                     f"{c['before_per_step'][metric]:,} | {c['marginal_per_step'][metric]:,} | "
                     f"{c['edge'][metric]:,} | {b:,} | {a:,} | {1 - a / b:.1%} | "
                     f"{'yes' if c['verified'] else 'no'} |")
    return "\n".join(lines)


def y_range(cases: list[dict], metric: str) -> tuple[float, float]:
    """The replica's y range when the data fit inside it (so the panels line
    up side by side), otherwise widened to whole decades."""
    lo, hi = YRANGE[metric]
    vals = [c[k][metric] for c in cases for k in ("before_total", "after_total")]
    if min(vals) < lo:
        lo = 10 ** math.floor(math.log10(min(vals)))
    if max(vals) > hi:
        hi = 10 ** math.ceil(math.log10(max(vals)))
    return lo, hi


def figure_tex(cases: list[dict], schemes: list[str], header: list[str], metric: str = "CNOT",
               lines: bool = False, show: str = "both") -> str:
    """Fig. 13 layout (diamonds before, circles after), one colour per formula,
    legend outside so it covers no data.

    lines=False: markers only, y range shared with the replica (``y_range``).
    lines=True: a solid line through each formula's after points and a dashed
    line through its before points, y autoscaled, and a two-entry style key.
    show='before' / 'after' (lines only): just that series per formula, as a
    solid line with Fig. 13's marker (diamonds before, circles after)."""
    if show not in ("both", "before", "after") or (show != "both" and not lines):
        raise ValueError(f"show={show!r} needs lines=True and one of both/before/after")
    ylabel = {"CNOT": "CNOT gate count", "Rz": "$R_z$ gate count"}[metric]
    yopts = []
    if not lines:
        ymin, ymax = y_range(cases, metric)
        yopts = [f"    ymin={ymin:g}, ymax={ymax:g},"]
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm, height=8.3cm,",
        "    xlabel={System size}, " f"ylabel={{{ylabel}}},", "    xmin=10, xmax=110,", *yopts,
        "    xtick={10,20,30,50,70,100}, xticklabels={10,20,30,50,70,100},",
        "    ymajorgrids=true, yminorgrids=true,", "    legend pos=outer north east, legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    after_style = "solid" if lines else "only marks"
    before_style = "dashed" if lines else "only marks"
    solid_marks = ", mark options={solid}" if lines else ""  # keep markers undashed on dashed lines
    for scheme in schemes:
        ser = sorted((c["n"], c["before_total"][metric], c["after_total"][metric])
                     for c in cases if c["scheme"] == scheme)
        if not ser:
            continue
        draw, fill = STYLE[scheme]
        name = pffigure.latex_escape(next(c["name"] for c in cases if c["scheme"] == scheme))
        if show != "both":
            mark, k = ("diamond*", 1) if show == "before" else ("*", 2)
            out.append(f"    \\addplot[solid, mark={mark}, mark size=2pt, draw={draw}, fill={fill}] coordinates {{")
            out += [f"      ({p[0]},{p[k]})" for p in ser]
            out += ["    };", f"    \\addlegendentry{{{name}}}"]
            continue
        out.append(f"    \\addplot[{after_style}, mark=*, mark size=2pt, draw={draw}, fill={fill}"
                   f"{solid_marks}] coordinates {{")
        out += [f"      ({n},{a})" for n, _, a in ser]
        out += ["    };", f"    \\addlegendentry{{{name}}}"]
        out.append(f"    \\addplot[{before_style}, mark=diamond*, mark size=2pt, draw={draw}, fill={fill}"
                   f"{solid_marks}, forget plot] coordinates {{")
        out += [f"      ({n},{b})" for n, b, _ in ser]
        out += ["    };"]
    if lines and show == "both":
        out += ["    \\addlegendimage{solid, black, mark=*, mark options={solid, fill=white}}",
                "    \\addlegendentry{after optimisation}",
                "    \\addlegendimage{dashed, black, mark=diamond*, mark options={solid, fill=white}}",
                "    \\addlegendentry{before optimisation}"]
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
    sizes, schemes, seeds, fit_sizes = cfg["sizes"], cfg["schemes"], cfg["seeds"], cfg["fit_sizes"]
    if sizes != list(samples.SIZES):
        raise ValueError(f"sizes must be the published sample sizes {samples.SIZES}, got {sizes}")
    unknown = [s for s in schemes if s not in STYLE or s not in formula_registry()]
    if unknown:
        raise ValueError(f"schemes {unknown} are not in the registry or have no style")
    if min(sizes) <= max(fit_sizes):
        raise ValueError("plotted sizes must all lie beyond the fit window")
    run_dirs = [(out / p.format(seed=s)).resolve() for p in cfg["runs"] for s in seeds]
    means = mean_measured_r(run_dirs, schemes, fit_sizes, seeds)
    fits = {}
    for scheme in schemes:
        a, b = fit_power_law(fit_sizes, [means[scheme][n] for n in fit_sizes])
        fits[scheme] = {"a": a, "b": b, "mean_r": {str(n): means[scheme][n] for n in fit_sizes}}
        print(f"{scheme:15s} r = {a:.4g} n^{b:.4f}  (mean over {len(seeds)} seeds, n = {fit_sizes})", flush=True)
    meta = {"config": cfg, "fits": fits, "provenance": provenance()}
    write_environment(out)
    cache = out / "cases"
    cache.mkdir(exist_ok=True)
    cases = []
    for n in sizes:  # cheap cases first, so a partial run covers every formula at small n
        for scheme in schemes:
            r = math.ceil(fits[scheme]["a"] * n ** fits[scheme]["b"])  # pfcompare's extrapolation rule
            f = cache / f"{scheme}-n{n}.json"
            case = json.loads(f.read_text()) if f.exists() else None
            if case is not None and case["r"] != r:
                raise ValueError(f"cached {f.name} has r={case['r']}, the fit now gives {r}; delete cases/ to rerun")
            if case is None:
                case = run_case(scheme, n, r, cfg["seed"], cfg["verify_max_n"])
                f.write_text(json.dumps(case, indent=1) + "\n")
            cases.append(case)
            print(f"{scheme:15s} n={n:3d} r={r:>9,} CNOT {case['before_total']['CNOT']:>14,} -> "
                  f"{case['after_total']['CNOT']:>14,}  ({case['optimiser_seconds']}s, "
                  f"verified={case['verified']})", flush=True)
    cases.sort(key=lambda c: (schemes.index(c["scheme"]), c["n"]))
    (out / "results.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=1) + "\n")
    prov = meta["provenance"]
    fit_txt = f"r = ceil(a n^b) fitted to measured r, mean over seeds {seeds}, n = {fit_sizes}"
    md = [f"# Fig. 13 extended: every formula, before/after optimisation, r from our measured-r fit\n",
          f"Written by `python -m pysimcount.pffig13ext {args.config}`. {fit_txt}. Unmerged simcount step, cz. "
          "After = our Light reconstruction, r x marginal + edge (as in childs-fig13). "
          f"Verified = random-state equivalence check run (n <= {cfg['verify_max_n']}).\n",
          "## Fits\n", "| formula | a | b |", "|---|---:|---:|"]
    md += [f"| {s} | {fits[s]['a']:.4g} | {fits[s]['b']:.4f} |" for s in schemes]
    for m in METRICS:
        md += ["", f"## {m}", "", table_md(cases, m)]
    (out / "table.md").write_text("\n".join(md) + "\n")
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    names = []
    for metric, side in (("CNOT", "left"), ("Rz", "right")):
        name = f"fig13ext-{metric.lower()}"
        header = ["Generated by pysimcount.pffig13ext -- do not edit; regenerate instead.",
                  f"figure: {name} -- Childs et al. 2018 Fig. 13 {side}, extended to every formula; "
                  "diamonds before, circles after optimisation",
                  f"metric={metric.lower()} gatebase=cz unmerged; r_method=measured-fit: {fit_txt}; "
                  "every point extrapolated",
                  f"optimised = Light reconstruction r x marginal + edge; verified n <= {cfg['verify_max_n']}",
                  f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
        (figs / f"{name}.tex").write_text(figure_tex(cases, schemes, header, metric))
        names.append(name)
    # Line-graph variants with some formulas left out (config "variants"), from
    # the same cases: nothing is recomputed.
    for v in cfg.get("variants", []):
        bad = [d for d in v["drop"] if d not in schemes]
        if bad or v["metric"] not in ("CNOT", "Rz"):
            raise ValueError(f"variant {v['name']}: unknown drop {bad} or metric {v['metric']!r}")
        kept = [s for s in schemes if s not in v["drop"]]
        dropped = ", ".join(formula_registry()[d].name for d in v["drop"])
        show = v.get("show", "both")
        what = {"both": "solid lines after, dashed lines before optimisation",
                "before": "before optimisation only (diamonds)", "after": "after optimisation only (circles)"}[show]
        header = ["Generated by pysimcount.pffig13ext -- do not edit; regenerate instead.",
                  f"figure: {v['name']} -- Fig. 13 extended, {v['metric']}, without {dropped}; {what}",
                  f"metric={v['metric'].lower()} gatebase=cz unmerged; r_method=measured-fit: {fit_txt}; "
                  "every point extrapolated; y axis autoscaled (not the replica's range)",
                  f"optimised = Light reconstruction r x marginal + edge; verified n <= {cfg['verify_max_n']}",
                  f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
        (figs / f"{v['name']}.tex").write_text(figure_tex(cases, kept, header, v["metric"], lines=True, show=show))
        names.append(v["name"])
    (figs / "all.tex").write_text(pffigure.PREAMBLE + "".join(
        f"\\section*{{\\texttt{{{s}.tex}}}}\n\\input{{{s}.tex}}\n\\clearpage\n" for s in names)
        + "\\end{document}\n")
    if args.render:
        print(f"rendered {pffigure.render(figs)}")


if __name__ == "__main__":
    main()
