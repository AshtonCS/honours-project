"""X-Y-Z vs two-group grouping at the measured sizes: r beside CNOT count, one line per formula and grouping.

    python -m pysimcount.figgroupline results/np-childs-format/grouping-lines.json [--render]

Two panels on the same x-axis (measured even n only; two-group needs even n on
the periodic chain), same formulas, same styles:

(a) measured Trotter number r     (b) CNOT count of the whole circuit, r x per step

Colour and marker follow the formula (pfchilds's style); X-Y-Z is a solid line,
two-group a dotted line, both with filled markers. Open markers and dashed
lines are not used: in the RQ1 figures they mean "extrapolated", and nothing
here is. Plotted value = arithmetic mean over seeds (the 2026-10-27 choice for
counts and r), bars = min-max over seeds. Each panel states the range over
the shown formulas of (X-Y-Z mean) / (two-group mean) at the largest n.

Both groupings are read from the same runs (``runs``; the np-grouping seeds,
paired with the gains table). Raises unless their X-Y-Z rows reproduce the
RQ1 record ``xyz_record`` and their two-group rows the record
``twogroup_record`` (mean, min, max and mean r at every plotted n), so the
figure agrees with both figures it stands beside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import pfchilds
from .figdrop import DERIVED_TAG
from .pfcompare import formula_registry, provenance
from .pfseeds import load_runs

GROUPINGS = ("xyz", "twogroup-even-out")
LINE = {"xyz": "solid", "twogroup-even-out": "densely dotted"}


# ------------------------------------------------------------------ the numbers

def _rows(runs: dict[int, dict], grouping: str) -> dict[tuple[str, int], dict[int, dict]]:
    """{(scheme, n): {seed: merged row}} for one grouping of np-grouping-style runs."""
    out: dict[tuple[str, int], dict[int, dict]] = {}
    for seed, d in runs.items():
        for row in d["rows"]:
            if row["variant"] != grouping or not row["merged"]:
                continue
            per = out.setdefault((row["scheme"], int(row["n"])), {})
            if seed in per:
                raise ValueError(f"seed {seed}: two {grouping} rows for {row['scheme']} n={row['n']}")
            if int(row["cnot_total"]) != int(row["cnot_per_step"]) * int(row["r"]):
                raise ValueError(f"seed {seed} {grouping} {row['scheme']} n={row['n']}: CNOT total != per step x r")
            per[seed] = row
    return out


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": math.fsum(values) / len(values), "min": min(values), "max": max(values)}


def series(runs: dict[int, dict], schemes: list[str], sizes: list[int]) -> dict[str, dict[str, list[dict]]]:
    """{grouping: {scheme: [{n, r: {mean,min,max}, cnot: {mean,min,max}}, ...]}}."""
    seeds = sorted(runs)
    out: dict[str, dict[str, list[dict]]] = {}
    for g in GROUPINGS:
        rows = _rows(runs, g)
        out[g] = {}
        for k in schemes:
            pts = []
            for n in sizes:
                per = rows.get((k, n), {})
                if sorted(per) != seeds:
                    raise ValueError(f"{g} {k} n={n}: seeds {sorted(per)}, expected {seeds}")
                pts.append({"n": n, "r": _summary([int(per[s]["r"]) for s in seeds]),
                            "cnot": _summary([int(per[s]["cnot_total"]) for s in seeds])})
            out[g][k] = pts
    return out


def check_against_record(pts: dict[str, list[dict]], rec: dict, what: str) -> None:
    """Raise unless every plotted point equals the record's measured point."""
    for k, ps in pts.items():
        mine = {p["n"]: p for p in rec["series"][k]["measured"]}
        for p in ps:
            q = mine.get(p["n"])
            if q is None:
                raise ValueError(f"{what}: {k} n={p['n']} not in the record")
            c = p["cnot"]
            if (c["min"], c["max"]) != (q["min"], q["max"]) or abs(c["mean"] - q["mean"]) > 1e-6 \
                    or abs(p["r"]["mean"] - q["mean_r"]) > 1e-9:
                raise ValueError(f"{what}: {k} n={p['n']} differs from the record")


def ratio_range(data: dict, metric: str, n: int) -> tuple[float, float]:
    """min, max over formulas of mean X-Y-Z / mean two-group at n."""
    ratios = []
    for k in data["xyz"]:
        x = next(p for p in data["xyz"][k] if p["n"] == n)[metric]["mean"]
        t = next(p for p in data["twogroup-even-out"][k] if p["n"] == n)[metric]["mean"]
        ratios.append(x / t)
    return min(ratios), max(ratios)


# ------------------------------------------------------------------ tikz

def _check_range(data: dict, metric: str, ymin: float, ymax: float) -> None:
    for g, by in data.items():
        for k, ps in by.items():
            for p in ps:
                if p[metric]["min"] < ymin or p[metric]["max"] > ymax:
                    raise ValueError(f"{g} {k} n={p['n']}: {metric} outside y-range {ymin:g}..{ymax:g}")


def _ticks(name: str, values: list[float]) -> str:
    text = ",".join(f"{v:g}" for v in values)
    return f"{name}tick={{{text}}}, {name}ticklabels={{{text}}}, {name}minorticks=false,"


def panel(data: dict, metric: str, cfg: dict, pcfg: dict, x_cm: float, legend_x: float | None,
          registry: dict, labels: dict) -> list[str]:
    _check_range(data, metric, pcfg["ymin"], pcfg["ymax"])
    lo, hi = ratio_range(data, metric, max(cfg["sizes"]))
    out = ["  \\begin{loglogaxis}[",
           "    scale only axis,",
           f"    at={{({x_cm}cm,0cm)}}, anchor=south west,",
           f"    width={cfg['width_cm']}cm, height={cfg['height_cm']}cm,",
           f"    title={{{pcfg['title']}}}, title style={{font=\\small}},",
           "    xlabel={System size $n$},",
           f"    ylabel={{{pcfg['ylabel']}}},",
           f"    {_ticks('x', cfg['sizes'])}",
           f"    {_ticks('y', pcfg['ytick'])}",
           f"    xmin={cfg['xmin']:g}, xmax={cfg['xmax']:g}, ymin={pcfg['ymin']:g}, ymax={pcfg['ymax']:g},",
           "    ymajorgrids=true, grid style={black!12}, tick label style={font=\\footnotesize},"]
    if legend_x is not None:
        out += [f"    legend style={{at={{({legend_x:.4f},-0.2)}}, anchor=north, draw=none, font=\\footnotesize,",
                f"      legend columns={cfg['legend_columns']}, /tikz/every even column/.append style={{column sep=8pt}}}},",
                "    legend cell align=left,"]
    out.append("  ]")
    for g in GROUPINGS:
        for k, ps in data[g].items():
            label, colour, mark = pfchilds._style(k, registry)
            forget = "" if (g == "xyz" and legend_x is not None) else ", forget plot"
            out.append(f"    % {k} {g}: mean over seeds, bars min..max over seeds")
            out.append(f"    \\addplot[{colour}, {LINE[g]}, line width=0.9pt, mark={mark}, mark size={cfg['mark_size']}, "
                       f"mark options={{solid, fill={colour}}}{forget}, error bars/.cd, y dir=both, y explicit, "
                       f"error bar style={{solid, line width=0.4pt}}] coordinates {{")
            for p in ps:
                v = p[metric]
                out.append(f"      ({p['n']},{v['mean']:.6g}) += (0,{v['max'] - v['mean']:.6g}) -= (0,{v['mean'] - v['min']:.6g})")
            out.append("    };")
            if not forget:
                out.append(f"    \\addlegendentry{{{labels.get(k, label)}}}")
    if legend_x is not None:
        out += ["    \\addlegendimage{black, solid, line width=0.9pt, mark=*, mark options={solid, fill=black}}",
                "    \\addlegendentry{X-Y-Z grouping}",
                "    \\addlegendimage{black, densely dotted, line width=0.9pt, mark=*, mark options={solid, fill=black}}",
                "    \\addlegendentry{two-group (even-out)}"]
    out.append(f"    \\node[anchor=north west, font=\\footnotesize, fill=white, inner sep=2pt, align=left] "
               f"at (rel axis cs:0.03,0.97) {{X-Y-Z / two-group at $n={max(cfg['sizes'])}$:\\\\"
               f"{lo:.2f}--{hi:.2f}$\\times$ (ratio of means)}};")
    out.append("  \\end{loglogaxis}")
    return out


def figure_tex(data: dict, cfg: dict, header: list[str], registry: dict) -> str:
    w, gap = cfg["width_cm"], cfg["gap_cm"]
    # legend sits in panel (b) but is centred under both panels: (centre - x0) / w in (b)'s axis units
    legend_x = ((2 * w + gap) / 2 - (w + gap)) / w
    out = header + ["\\begin{tikzpicture}"]
    out += panel(data, "r", cfg, cfg["r_panel"], 0, None, registry, cfg.get("labels", {}))
    out += panel(data, "cnot", cfg, cfg["cnot_panel"], w + gap, legend_x, registry, cfg.get("labels", {}))
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ driver

def run(cfg: dict, here: Path, render: bool) -> Path:
    registry = formula_registry()
    agg = here / cfg["runs"]
    runs = load_runs([agg.parent / p for p in json.loads(agg.read_text())["runs"]])
    data = series(runs, cfg["schemes"], cfg["sizes"])
    shas = []
    for g, key in (("xyz", "xyz_record"), ("twogroup-even-out", "twogroup_record")):
        text = (here / cfg[key]).read_text()
        rec = json.loads(text)
        if sorted(rec["seeds"]) != sorted(runs):
            raise ValueError(f"{cfg[key]} has seeds {rec['seeds']}, the runs {sorted(runs)}")
        check_against_record(data[g], rec, cfg[key])
        shas.append(f"{cfg[key]} sha256={hashlib.sha256(text.encode()).hexdigest()}")
    prov = provenance()
    n = max(cfg["sizes"])
    header = ["% Generated by pysimcount.figgroupline -- do not edit; regenerate instead.",
              f"% figure: {cfg['id']} -- X-Y-Z vs two-group (even-out): r and CNOT count at measured n",
              f"% source: {cfg['runs']} (seeds {sorted(runs)}); merged, cz, simcount template",
              f"% shown: {', '.join(cfg['schemes'])}; n = {cfg['sizes']}",
              "% plotted: mean over seeds; bars min..max over seeds; all r measured",
              f"% r ratio at n={n} (mean X-Y-Z / mean two-group): {ratio_range(data, 'r', n)}",
              f"% CNOT ratio at n={n}: {ratio_range(data, 'cnot', n)}",
              f"{DERIVED_TAG} {cfg['runs']}; points equal {'; '.join(shas)} (checked) by pysimcount.figgroupline",
              f"% figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
    if prov["git_dirty"]:
        header.append("% WARNING: made from uncommitted code; the commit above does not reproduce it.")
    path = here / f"{cfg['id']}.tex"
    path.write_text(figure_tex(data, cfg, header, registry))
    print(f"  wrote {path}")
    if render:
        pdf, _ = pfchilds.render_tex(path, here / "rendered")
        print(f"  rendered {pdf}")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    run(json.loads(args.config.read_text()), args.config.parent, args.render)


if __name__ == "__main__":
    main()
