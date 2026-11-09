"""T-count figures for the RQ1 T slide, from existing Clifford+T runs (nothing
is simulated or synthesised here).

    python -m pysimcount.pftcount results/np-tcount/config.json [--render]

Reads the ``ct`` rows (merged, NumPy instances) of the run directories named
in the config and writes, beside the config:

* ``figures/tcount-vs-n.tex``: mean T-count over the seeds against n for the
  configured formulas, bars = min..max over seeds. Colour = order, dashed =
  Suzuki, solid = optimised. Filled markers: measured r. Open markers beyond
  the dashed vertical line: r from each seed's power-law fit in that run
  (``r_source == "fit"``), with the circuit built and synthesised at that r
  by ``pfcompare`` -- so the per-rotation synthesis cost is computed, not
  modelled.
* ``figures/t-per-rotation.tex``: T gates per Rz rotation against
  log2(1/eps_rot), one point per synthesised circuit (every formula, seed, n
  and r source), with the least-squares line. eps_rot = eps_synth / (r R),
  where R is rotations per step (synthesis.py splits the synthesis budget
  evenly over all r R rotations), so T per rotation = t_total / (r R) and
  r R = eps_synth / eps_rot; that this is an integer multiple of r is checked.
* ``table.md``, ``results.json``: the numbers behind both, and the T ratio of
  each configured (Suzuki, optimised) pair at the largest measured n.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from . import pffigure
from .pfcompare import formula_registry, provenance, write_environment

ORDER_COLOUR = {1: "gray", 2: "red", 4: "orange", 6: "blue", 8: "green!50!black"}
OPT_MARKS = ("*", "triangle*", "diamond*")  # optimised formulas sharing an order get these in turn
SUZUKI_MARK = "square*"


def t_per_rotation(t_total: int, r: int, eps_rot: float, eps_synth: float) -> tuple[float, int]:
    """(T gates per rotation, rotations per step R) for one synthesised circuit.

    r R = eps_synth / eps_rot (the synthesis budget is split evenly over all
    r R rotations). Raises if that is not an integer multiple of r."""
    total_rot = eps_synth / eps_rot
    R = round(total_rot / r)
    if R < 1 or abs(total_rot - r * R) > 1e-6 * total_rot:
        raise ValueError(f"eps_synth/eps_rot = {total_rot} is not a multiple of r = {r}")
    return t_total / (r * R), R


def load_ct(run_dirs: list[Path]) -> dict[tuple[str, int, int, str], dict]:
    """{(scheme, n, seed, r_source): row} for merged ct rows on NumPy instances.
    Raises if two files give different T for the same key."""
    out: dict[tuple[str, int, int, str], dict] = {}
    for d in run_dirs:
        seed = int(d.name.removeprefix("seed"))
        with open(d / "results.csv") as f:
            for row in csv.DictReader(f):
                if row["gatebase"] != "ct" or row["merged"] != "True" or row["instance"] != "numpy":
                    continue
                key = (row["scheme"], int(row["n"]), seed, row["r_source"])
                if key in out and out[key]["t_total"] != row["t_total"]:
                    raise ValueError(f"conflicting ct rows for {key} ({d})")
                out[key] = row
    return out


def series_stats(rows: dict, scheme: str, source: str, seeds: list[int]) -> dict[int, dict]:
    """{n: {mean, min, max, mean_r}} of t_total over the seeds, for every n at
    which this scheme has a row with this r_source on every seed (an n present
    on only some seeds raises)."""
    ns = sorted({k[1] for k in rows if k[0] == scheme and k[3] == source})
    out = {}
    for n in ns:
        missing = [s for s in seeds if (scheme, n, s, source) not in rows]
        if missing:
            raise ValueError(f"{scheme} n={n} {source}: no row for seeds {missing}")
        t = [int(rows[(scheme, n, s, source)]["t_total"]) for s in seeds]
        r = [int(rows[(scheme, n, s, source)]["r"]) for s in seeds]
        out[n] = {"mean": math.fsum(t) / len(t), "min": min(t), "max": max(t), "mean_r": math.fsum(r) / len(r)}
    return out


def line_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a x + b."""
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def _styles(schemes: list[str]) -> dict[str, tuple[str, str, str]]:
    """{scheme: (colour, line style, mark)}; raises if two would look the same."""
    reg = formula_registry()
    used: dict[int, int] = {}
    out, looks = {}, {}
    for s in schemes:
        f = reg[s]
        colour = ORDER_COLOUR[f.order]
        if f.family == "suzuki":
            style, mark = "dashed", SUZUKI_MARK
        else:
            k = used.get(f.order, 0)
            used[f.order] = k + 1
            if k >= len(OPT_MARKS):
                raise ValueError(f"more than {len(OPT_MARKS)} optimised formulas of order {f.order}")
            style, mark = "solid", OPT_MARKS[k]
        if (colour, style, mark) in looks:
            raise ValueError(f"{s} would look identical to {looks[(colour, style, mark)]}")
        looks[(colour, style, mark)] = s
        out[s] = (colour, style, mark)
    return out


def _coord(n: int, st: dict) -> str:
    m = st["mean"]
    return f"      ({n},{m:.0f}) += (0,{st['max'] - m:.0f}) -= (0,{m - st['min']:.0f})"


def tcount_figure_tex(stats: dict[str, dict[str, dict[int, dict]]], schemes: list[str], header: list[str]) -> str:
    """T-count vs n: measured series (legend) then, if present, the fitted-r
    continuation from the last measured n (forget plot, open markers)."""
    reg = formula_registry()
    styles = _styles(schemes)
    ys = [v for s in schemes for part in stats[s].values() for st in part.values() for v in (st["min"], st["max"])]
    ymin, ymax = 10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))
    ns = sorted({n for s in schemes for part in stats[s].values() for n in part})
    last_measured = max(n for s in schemes for n in stats[s]["measured"])
    cut = None
    if any(stats[s]["fit"] for s in schemes):
        first_fit = min(n for s in schemes for n in stats[s]["fit"])
        cut = math.sqrt(last_measured * first_fit)  # midway on the log axis
    ticks = ",".join(map(str, ns))
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm, height=8cm,",
        "    xlabel={System size $n$}, ylabel={$T$ count},",
        f"    xmin={ns[0] * 0.92:g}, xmax={ns[-1] * 1.1:g}, ymin={ymin:g}, ymax={ymax:g},",
        f"    xtick={{{ticks}}}, xticklabels={{{ticks}}},",
        "    ymajorgrids=true, legend pos=outer north east, legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for s in schemes:
        colour, style, mark = styles[s]
        meas, fit = stats[s]["measured"], stats[s]["fit"]
        bars = f"error bars/.cd, y dir=both, y explicit, error bar style={{{colour}, solid}}"
        out.append(f"    \\addplot[{colour}, {style}, mark={mark}, mark size=2.2pt, "
                   f"mark options={{solid, fill={colour}}}, {bars}] coordinates {{")
        out += [_coord(n, meas[n]) for n in sorted(meas)]
        out += ["    };", f"    \\addlegendentry{{{pffigure.latex_escape(reg[s].name)}}}"]
        if fit:
            join = {"mean": meas[last_measured]["mean"], "min": meas[last_measured]["mean"],
                    "max": meas[last_measured]["mean"]}  # line starts at the last measured mean, no bar
            pts = [(last_measured, join)] + [(n, fit[n]) for n in sorted(fit)]
            idx = ",".join(str(i) for i in range(2, len(pts) + 1))
            out.append(f"    \\addplot[{colour}, densely dotted, mark={mark}, mark size=2.2pt, mark indices={{{idx}}}, "
                       f"mark options={{solid, fill=white}}, forget plot, "
                       f"error bars/.cd, y dir=both, y explicit, error bar style={{{colour}, dashed}}] coordinates {{")
            out += [_coord(n, st) for n, st in pts]
            out += ["    };"]
    if cut is not None:
        out += [f"    \\draw[gray, densely dashed] (axis cs:{cut:.3f},{ymin:g}) -- (axis cs:{cut:.3f},{ymax:g});",
                f"    \\node[gray, font=\\scriptsize, anchor=north west] at (axis cs:{cut:.3f},{ymax:g}) {{fitted $r$}};"]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def rotation_figure_tex(points: list[tuple[float, float, bool]], a: float, b: float, header: list[str]) -> str:
    """T per rotation vs log2(1/eps_rot); filled = measured r, open = fitted or bound r."""
    xs = [p[0] for p in points]
    lo, hi = math.floor(min(xs)), math.ceil(max(xs))
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{axis}[", "    width=10cm, height=7cm,",
        "    xlabel={$\\log_2(1/\\varepsilon_{\\mathrm{rot}})$}, ylabel={$T$ gates per rotation},",
        f"    xmin={lo}, xmax={hi},", "    ymajorgrids=true, legend pos=north west, legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for measured, label, fill in ((True, "measured $r$", "black"), (False, "fitted or bound $r$", "white")):
        pts = [p for p in points if p[2] == measured]
        if not pts:
            continue
        out.append(f"    \\addplot[only marks, black, mark=*, mark size=1.1pt, mark options={{fill={fill}}}] coordinates {{")
        out += [f"      ({x:.4f},{y:.3f})" for x, y, _ in sorted(pts)]
        out += ["    };", f"    \\addlegendentry{{{label}}}"]
    out += [f"    \\addplot[red, thick, domain={lo}:{hi}, samples=2] {{{a:.6f}*x + {b:.6f}}};",
            f"    \\addlegendentry{{${a:.2f}\\,\\log_2(1/\\varepsilon_{{\\mathrm{{rot}}}}) {b:+.1f}$}}",
            "  \\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out = args.config.parent
    seeds, schemes = cfg["seeds"], cfg["schemes"]
    reg = formula_registry()
    needed = list(dict.fromkeys(schemes + [k for pair in cfg.get("pairs", []) for k in pair]))
    unknown = [s for s in needed if s not in reg]
    if unknown:
        raise ValueError(f"unknown schemes {unknown}")
    run_dirs = [(out / p.format(seed=s)).resolve() for p in cfg["runs"] for s in seeds]
    rows = load_ct(run_dirs)
    # The fitted-r rows come from each run's own extrapolation config; record it.
    fit_cfgs = {json.dumps(json.loads((d / "config.json").read_text()).get("extrapolate"), sort_keys=True)
                for d in run_dirs}
    fit_cfgs.discard("null")
    if len(fit_cfgs) > 1:
        raise ValueError(f"runs disagree on the extrapolation config: {fit_cfgs}")
    fit_cfg = json.loads(fit_cfgs.pop()) if fit_cfgs else None

    stats = {s: {"measured": series_stats(rows, s, "measured", seeds), "fit": series_stats(rows, s, "fit", seeds)}
             for s in needed}

    points, records = [], []
    for (scheme, n, seed, source), row in sorted(rows.items()):
        if not row["eps_rot"] or not row["eps_synth"]:
            continue
        eps_rot, eps_synth = float(row["eps_rot"]), float(row["eps_synth"])
        tpr, R = t_per_rotation(int(row["t_total"]), int(row["r"]), eps_rot, eps_synth)
        points.append((math.log2(1 / eps_rot), tpr, source == "measured"))
        records.append({"scheme": scheme, "n": n, "seed": seed, "r_source": source, "r": int(row["r"]), "R": R,
                        "log2_inv_eps_rot": math.log2(1 / eps_rot), "t_per_rotation": tpr})
    a, b = line_fit([p[0] for p in points], [p[1] for p in points])
    resid = max(abs(y - (a * x + b)) for x, y, _ in points)
    ratio = [y / x for x, y, _ in points]

    n_ref = max(n for s in schemes for n in stats[s]["measured"])
    pairs = []
    for suz, opt in cfg.get("pairs", []):
        ts, to = stats[suz]["measured"][n_ref]["mean"], stats[opt]["measured"][n_ref]["mean"]
        pairs.append({"suzuki": suz, "optimised": opt, "n": n_ref, "t_suzuki": ts, "t_optimised": to, "ratio": ts / to})

    prov = provenance()
    write_environment(out)
    (out / "results.json").write_text(json.dumps(
        {"meta": {"config": cfg, "extrapolate": fit_cfg, "provenance": prov},
         "stats": {s: {k: {str(n): v for n, v in part.items()} for k, part in d.items()} for s, d in stats.items()},
         "t_per_rotation_fit": {"a": a, "b": b, "max_abs_residual": resid, "points": len(points),
                                "t_per_rotation_over_log2": [min(ratio), max(ratio)]},
         "pairs": pairs, "rotation_points": records}, indent=1) + "\n")

    md = ["# T-count for the RQ1 T slide", "",
          f"Written by `python -m pysimcount.pftcount {args.config}`. Mean over seeds {seeds}, merged, ct "
          f"(eps_trotter = eps_synth = 5e-4). Measured r at every n in the runs; fitted r from each seed's "
          f"own power-law fit ({fit_cfg}).", "",
          "## T per rotation", "",
          f"Least squares over {len(points)} synthesised circuits: T/rotation = {a:.3f} log2(1/eps_rot) {b:+.2f} "
          f"(max |residual| {resid:.2f}); T/rotation / log2(1/eps_rot) in [{min(ratio):.3f}, {max(ratio):.3f}].", "",
          f"## Equal-order T ratios at n = {n_ref} (measured)", "",
          "| Suzuki | optimised | T Suzuki | T optimised | ratio |", "|---|---|---:|---:|---:|"]
    md += [f"| {p['suzuki']} | {p['optimised']} | {p['t_suzuki']:,.0f} | {p['t_optimised']:,.0f} | {p['ratio']:.2f}x |"
           for p in pairs]
    md += ["", "## Mean T-count", "", "| formula | " + " | ".join(
        f"n={n}{'' if src == 'measured' else ' (fit)'}" for src in ("measured", "fit")
        for n in sorted(stats[schemes[0]][src])) + " |"]
    md.append("|---|" + "---:|" * sum(len(stats[schemes[0]][src]) for src in ("measured", "fit")))
    for s in schemes:
        cells = [f"{stats[s][src][n]['mean']:,.0f}" for src in ("measured", "fit") for n in sorted(stats[s][src])]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(md) + "\n")

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    git = f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"
    fit_txt = (f"open markers: r from each seed's power-law fit on n = {fit_cfg['fit_sizes']}, "
               "circuit synthesised at that r (extrapolated)") if fit_cfg else "no fitted sizes"
    head_t = ["Generated by pysimcount.pftcount -- do not edit; regenerate instead.",
              "figure: tcount-vs-n -- T-count vs n, Suzuki (dashed) vs optimised (solid), colour = order",
              f"metric=t gatebase=ct merged; mean over seeds {seeds}, bars = min..max; measured r (filled); {fit_txt}",
              "formulas: " + ", ".join(reg[s].name for s in schemes), git]
    head_r = ["Generated by pysimcount.pftcount -- do not edit; regenerate instead.",
              f"figure: t-per-rotation -- T gates per rotation vs log2(1/eps_rot), {len(points)} synthesised circuits",
              f"metric=t gatebase=ct merged; every formula, seed, n and r source in the runs (seeds {seeds}); "
              f"fit {a:.3f} x {b:+.2f}", git]
    (figs / "tcount-vs-n.tex").write_text(tcount_figure_tex(stats, schemes, head_t))
    (figs / "t-per-rotation.tex").write_text(rotation_figure_tex(points, a, b, head_r))
    (figs / "all.tex").write_text(pffigure.PREAMBLE + "".join(
        f"\\section*{{\\texttt{{{s}.tex}}}}\n\\input{{{s}.tex}}\n\\clearpage\n" for s in ("tcount-vs-n", "t-per-rotation"))
        + "\\end{document}\n")
    print("\n".join(md))
    if args.render:
        print(f"rendered {pffigure.render(figs)}")


if __name__ == "__main__":
    main()
