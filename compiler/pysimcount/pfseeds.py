"""Aggregate ``pfcompare`` runs that differ only in the seed (simcount or NumPy instances).

    python -m pysimcount.pfseeds results/pf-seeds/aggregate.json --render

Each seed is a different random-field instance of the same model (periodic
Heisenberg chain, J = 1, fields from simcount's Haskell RNG or from
``pfcompare.numpy_fields``), so the spread across seeds is
instance-to-instance variation at fixed n, t = n and eps. The min-max range
is shown, and the median is the central value (robust to costs that differ by
factors; for an even number of seeds it is the mean of the middle two).

Comparisons between formulas are *paired*: a gain or a win is computed on
each seed (both formulas on the same instance) and only then summarised.
A ratio of two medians would compare different instances.

What it writes (into the config's ``out`` directory):

* ``spread.csv`` / ``spread.md``: per (scheme, gate set, n, metric) the value
  for every seed, median, min, max and max/min, for r and for each gate total;
* ``rankings.md``: every pair of formulas whose cost order at some n differs
  between the reference seed and another seed (strict reversals only);
* ``reproduction.md``: the reference seed's rows compared field by field
  with a reference results file (the seed-1 rows must reproduce pf-thesis);
* ``gains.md``: per-seed gain of every non-Suzuki formula over the Suzuki
  formula of the same order (exponentials, CNOT, T), median and range;
* ``wins.md``: per (metric, n), how many seeds each formula is cheapest on;
* ``bounds.md``: per-seed bound r / measured r for the bound rows, median and
  range (only if the runs have bound rows);
* ``figures/merged/<metric>/``: pgfplots figures in ``pffigure``'s A and B
  families (measured r only), median with min-max error bars, styles from
  ``pffigure``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from . import pffigure
from .pfcompare import provenance, write_environment

# Columns compared when checking that a rerun reproduces an earlier run. Only
# the wall-clock time of the search may differ.
NONDETERMINISTIC = {"search_seconds"}

# metric name -> (column, gatebase); "r" is reported for both gate sets.
SPREAD_METRICS: list[tuple[str, str, str]] = [
    ("r", "r", "cz"), ("r", "r", "ct"),
    ("cnot", "cnot_total", "cz"), ("total", "gates_total", "cz"), ("t", "t_total", "ct"),
]


@dataclass
class Agg:
    """One (scheme, gatebase, n, metric) across seeds."""
    scheme: str
    order: int
    gatebase: str
    n: int
    r_source: str          # "measured" or "fit"
    metric: str
    values: dict[int, float]  # seed -> value
    median: float
    lo: float
    hi: float

    @property
    def ratio(self) -> float:
        """max / min across seeds (1 = no spread)."""
        return self.hi / self.lo


# ------------------------------------------------------------------ loading

def load_runs(paths: list[Path]) -> dict[int, dict]:
    """{seed: results.json content}. Raises if two runs share a seed, if the
    configs differ in anything but seed/name/description, or if a run was made
    from uncommitted code."""
    runs: dict[int, dict] = {}
    base = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        cfg = dict(d["meta"]["config"])
        seed = int(cfg.pop("seed"))
        for k in ("name", "description"):
            cfg.pop(k, None)
        if seed in runs:
            raise ValueError(f"two runs for seed {seed}")
        if base is None:
            base = cfg
        elif cfg != base:
            raise ValueError(f"{p}: config differs from the first run in more than the seed")
        if d["meta"]["provenance"].get("git_dirty"):
            raise ValueError(f"{p} was made from uncommitted code (git_dirty)")
        runs[seed] = d
    return runs


# ------------------------------------------------------------------ spread

def _instance(run: dict) -> str:
    """The instance label of a run's own rows (not the paper instance)."""
    return run["meta"]["config"].get("instances", "simcount")

def aggregate(runs: dict[int, dict], metric: str, column: str, gatebase: str, merged: bool = True) -> list[Agg]:
    """Median/min/max of ``column`` across seeds for every (scheme, n).

    Uses measured and fit rows (bound rows are not part of this comparison).
    Raises if a seed is missing a point another seed has, or has it twice.
    """
    table: dict[tuple[str, int, str], dict[int, float]] = {}
    orders: dict[str, int] = {}
    for seed, d in runs.items():
        for row in d["rows"]:
            if (row["instance"] != _instance(d) or row["gatebase"] != gatebase or bool(row["merged"]) != merged
                    or row["r_source"] not in ("measured", "fit")):
                continue
            key = (row["scheme"], int(row["n"]), row["r_source"])
            per = table.setdefault(key, {})
            if seed in per:
                raise ValueError(f"seed {seed}: two rows for {key}")
            per[seed] = float(row[column])
            orders[row["scheme"]] = int(row["order"])
    out: list[Agg] = []
    seeds = sorted(runs)
    for (scheme, n, src), per in sorted(table.items()):
        if sorted(per) != seeds:
            raise ValueError(f"{scheme} n={n} ({src}): seeds {sorted(per)}, expected {seeds}")
        vals = [per[s] for s in seeds]
        out.append(Agg(scheme, orders[scheme], gatebase, n, src, metric, dict(per),
                       statistics.median(vals), min(vals), max(vals)))
    return out


# ------------------------------------------------------------------ rankings

@dataclass
class Flip:
    """Formulas a and b are in opposite cost order for ``seed`` and the reference."""
    metric: str
    gatebase: str
    n: int
    r_source: str
    a: str             # cheaper on the reference seed
    b: str
    order_a: int
    order_b: int
    seed: int
    ref_a: float
    ref_b: float
    seed_a: float
    seed_b: float


def ranking_flips(aggs: list[Agg], reference: int) -> list[Flip]:
    """Every strict reversal of a pairwise cost order relative to ``reference``.

    For each n, each pair (a, b) with value_ref(a) < value_ref(b), each other
    seed s with value_s(a) > value_s(b) is a flip. Ties on either side are not
    flips.
    """
    by_n: dict[tuple[int, str], list[Agg]] = {}
    for x in aggs:
        by_n.setdefault((x.n, x.r_source), []).append(x)
    out: list[Flip] = []
    for (n, src), xs in sorted(by_n.items()):
        for i in range(len(xs)):
            for j in range(len(xs)):
                a, b = xs[i], xs[j]
                if a.values[reference] >= b.values[reference]:
                    continue
                for s in sorted(a.values):
                    if s != reference and a.values[s] > b.values[s]:
                        out.append(Flip(a.metric, a.gatebase, n, src, a.scheme, b.scheme, a.order, b.order, s,
                                        a.values[reference], b.values[reference], a.values[s], b.values[s]))
    return out


def best_formula(aggs: list[Agg], seed: int) -> dict[tuple[int, str], str]:
    """{(n, r_source): cheapest scheme} for one seed (ties: first in sort order)."""
    best: dict[tuple[int, str], tuple[float, str]] = {}
    for x in aggs:
        k = (x.n, x.r_source)
        v = (x.values[seed], x.scheme)
        if k not in best or v < best[k]:
            best[k] = v
    return {k: v[1] for k, v in best.items()}


# ------------------------------------------------------------------ paired comparisons

GAIN_METRICS: list[tuple[str, str, str]] = [
    ("exponentials", "rz_total", "cz"), ("cnot", "cnot_total", "cz"), ("t", "t_total", "ct")]


def same_order_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """(suzuki, other) for every non-Suzuki formula whose order has a Suzuki formula."""
    fam, order = {}, {}
    for r in rows:
        fam[r["scheme"]], order[r["scheme"]] = r["family"], int(r["order"])
    suz = {order[k]: k for k in fam if fam[k] == "suzuki"}
    return [(suz[order[k]], k) for k in fam if fam[k] != "suzuki" and order[k] in suz]


def _values(run: dict, column: str, gatebase: str, sources=("measured", "fit")) -> dict[tuple[str, int, str], float]:
    out = {}
    for row in run["rows"]:
        if (row["instance"] == _instance(run) and row["gatebase"] == gatebase and row["merged"]
                and row["r_source"] in sources):
            out[(row["scheme"], int(row["n"]), row["r_source"])] = float(row[column])
    return out


def paired_gains(runs: dict[int, dict]) -> list[dict]:
    """Per (metric, pair, n, r_source): gain = Suzuki cost / other cost on each
    seed, then median, min and max over seeds."""
    seeds = sorted(runs)
    pairs = same_order_pairs(runs[seeds[0]]["rows"])
    out = []
    for metric, column, gb in GAIN_METRICS:
        vals = {s: _values(runs[s], column, gb) for s in seeds}
        keys = sorted({(n, src) for (_, n, src) in vals[seeds[0]]})
        for base, alt in pairs:
            for n, src in keys:
                g = {s: vals[s][(base, n, src)] / vals[s][(alt, n, src)] for s in seeds}
                gs = list(g.values())
                out.append({"metric": metric, "base": base, "alt": alt, "n": n, "r_source": src, "per_seed": g,
                            "median": statistics.median(gs), "min": min(gs), "max": max(gs)})
    return out


def win_counts(runs: dict[int, dict]) -> list[dict]:
    """Per (metric, n, r_source): {scheme: number of seeds on which it is cheapest}."""
    seeds = sorted(runs)
    out = []
    for metric, column, gb in [("cnot", "cnot_total", "cz"), ("t", "t_total", "ct"), ("total", "gates_total", "cz")]:
        vals = {s: _values(runs[s], column, gb) for s in seeds}
        for n, src in sorted({(n, src) for (_, n, src) in vals[seeds[0]]}):
            wins: dict[str, int] = {}
            for s in seeds:
                cand = {k: v for (k, nn, ss), v in vals[s].items() if nn == n and ss == src}
                best = min(cand, key=lambda k: (cand[k], k))
                wins[best] = wins.get(best, 0) + 1
            out.append({"metric": metric, "n": n, "r_source": src, "wins": wins})
    return out


def bound_ratios(runs: dict[int, dict]) -> list[dict]:
    """Per (scheme, bound, n): bound r / measured (or fitted) r on each seed, summarised."""
    seeds = sorted(runs)
    meas = {s: _values(runs[s], "r", "cz") for s in seeds}
    bnd = {s: {} for s in seeds}
    for s in seeds:
        for row in runs[s]["rows"]:
            if (row["instance"] == _instance(runs[s]) and row["gatebase"] == "cz" and row["merged"]
                    and row["r_source"] in ("ana", "min", "com", "emp", "tte")):
                bnd[s][(row["scheme"], int(row["n"]), row["r_source"])] = float(row["r"])
    out = []
    for (scheme, n, b) in sorted(bnd[seeds[0]]):
        ref = (scheme, n, "measured") if (scheme, n, "measured") in meas[seeds[0]] else (scheme, n, "fit")
        if ref not in meas[seeds[0]]:
            continue
        rat = {s: bnd[s][(scheme, n, b)] / meas[s][ref] for s in seeds}
        v = list(rat.values())
        out.append({"scheme": scheme, "bound": b, "n": n, "denominator": ref[2], "per_seed": rat,
                    "median": statistics.median(v), "min": min(v), "max": max(v)})
    return out


def gains_markdown(gains: list[dict]) -> str:
    lines = ["| metric | pair | n | r from | median gain | min | max |", "|---|---|---:|---|---:|---:|---:|"]
    for g in gains:
        lines.append(f"| {g['metric']} | {g['alt']} vs {g['base']} | {g['n']} | {g['r_source']} | "
                     f"{g['median']:.2f}× | {g['min']:.2f}× | {g['max']:.2f}× |")
    return "\n".join(lines)


def wins_markdown(wins: list[dict], n_seeds: int) -> str:
    lines = [f"| metric | n | r from | cheapest formula: seeds won (of {n_seeds}) |", "|---|---:|---|---|"]
    for w in wins:
        cells = ", ".join(f"{k} {v}" for k, v in sorted(w["wins"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {w['metric']} | {w['n']} | {w['r_source']} | {cells} |")
    return "\n".join(lines)


def bounds_markdown(rat: list[dict]) -> str:
    lines = ["| formula | bound | n | denominator | median bound/measured | min | max |",
             "|---|---|---:|---|---:|---:|---:|"]
    for x in rat:
        lines.append(f"| {x['scheme']} | {x['bound']} | {x['n']} | {x['denominator']} | {x['median']:,.2f} | "
                     f"{x['min']:,.2f} | {x['max']:,.2f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ reproduction

def compare_rows(rows_a: list[dict], rows_b: list[dict], sources: tuple[str, ...],
                 instances: tuple[str, ...] | None = None) -> tuple[int, list[str]]:
    """Field-by-field comparison of the rows with r_source in ``sources``
    (and instance in ``instances``, if given).

    Rows are matched on (instance, n, scheme, gatebase, r_source, merged).
    Returns (number of rows compared, list of differences); missing rows on
    either side count as differences. ``search_seconds`` (wall-clock) is not
    compared.
    """
    def index(rows):
        out = {}
        for r in rows:
            if r["r_source"] not in sources or (instances is not None and r["instance"] not in instances):
                continue
            k = (r["instance"], int(r["n"]), r["scheme"], r["gatebase"], r["r_source"], bool(r["merged"]))
            if k in out:
                raise ValueError(f"duplicate row {k}")
            out[k] = r
        return out
    ia, ib = index(rows_a), index(rows_b)
    diffs = []
    for k in sorted(set(ia) | set(ib), key=str):
        if k not in ia or k not in ib:
            diffs.append(f"{k}: only in {'second' if k not in ia else 'first'}")
            continue
        for f in ia[k]:
            if f in NONDETERMINISTIC:
                continue
            if ia[k][f] != ib[k].get(f):
                diffs.append(f"{k}: {f} {ia[k][f]!r} != {ib[k].get(f)!r}")
    return len(set(ia) & set(ib)), diffs


# ------------------------------------------------------------------ figures

def _errbar_tex(series: list[pffigure.Series], errs: list[dict[tuple[int, str], tuple[float, float]]], ylabel: str,
                header: list[str]) -> str:
    """pffigure.figure_tex with asymmetric y error bars (median -> min, max).
    Fitted (extrapolated) points are drawn open, as in pffigure, with a
    legend key distinguishing computed from fitted r."""
    sizes = sorted({n for s in series for n, _ in s.measured + s.fit})
    ticks = ",".join(str(n) for n in sizes)
    out = [f"% {line}" for line in header]
    out += [
        "\\begin{tikzpicture}",
        "  \\begin{loglogaxis}[",
        "    width=10cm,",
        "    xlabel={System size $n$},",
        f"    ylabel={{{ylabel}}},",
        f"    xtick={{{ticks}}},",
        f"    xticklabels={{{ticks}}},",
        "    ymajorgrids=true,",
        "    legend pos=outer north east,",
        "    legend cell align=left,",
        "    legend style={font=\\footnotesize},",
        "  ]",
    ]
    for s, e in zip(series, errs):
        out.append(f"    \\addplot[only marks, {s.colour}, mark={s.mark}, mark size=2.4pt, "
                   f"mark options={{solid, fill={s.colour}}}, error bars/.cd, y dir=both, y explicit, "
                   f"error bar style={{{s.colour}}}] coordinates {{")
        for n, y in s.measured:
            lo, hi = e[(n, "measured")]
            out.append(f"      ({n},{y:.0f}) += (0,{hi - y:.0f}) -= (0,{y - lo:.0f})")
        out += ["    };", f"    \\addlegendentry{{{s.label}}}"]
        if s.fit:
            out.append(f"    \\addplot[only marks, {s.colour}, mark={s.mark}, mark size=2.4pt, "
                       f"mark options={{solid, fill=white}}, forget plot, error bars/.cd, y dir=both, y explicit, "
                       f"error bar style={{{s.colour}, dashed}}] coordinates {{")
            for n, y in s.fit:
                lo, hi = e[(n, "fit")]
                out.append(f"      ({n},{y:.0f}) += (0,{hi - y:.0f}) -= (0,{y - lo:.0f})")
            out.append("    };")
    if any(s.fit for s in series):
        out += ["    \\addlegendimage{only marks, black, mark=*}",
                "    \\addlegendentry{measured $r$}",
                "    \\addlegendimage{only marks, black, mark=*, mark options={fill=white}}",
                "    \\addlegendentry{fitted $r$ (extrapolated)}"]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def write_figures(runs: dict[int, dict], root: Path, merged: bool = True) -> Path:
    """A-measured-by-order and B-* figures for every pffigure metric: median
    across seeds, min-max error bars; fitted r (extrapolated) drawn open."""
    from .pfcompare import formula_registry
    registry = formula_registry()
    seeds = sorted(runs)
    pages, index = [], ["# Figures (median over seeds, error bars min-max)", "",
                        f"seeds {seeds}; filled = measured r, open = fitted r (power law, extrapolated).", ""]
    for metric, (column, gatebase, ylabel) in pffigure.METRICS.items():
        aggs = aggregate(runs, metric, column, gatebase, merged)
        data: dict[tuple[str, str], dict[str, list]] = {}
        errs: dict[str, dict[tuple[int, str], tuple[float, float]]] = {}
        for x in aggs:
            data.setdefault((x.scheme, "measured"), {"measured": [], "fit": []})[x.r_source].append((x.n, x.median))
            errs.setdefault(x.scheme, {})[(x.n, x.r_source)] = (x.lo, x.hi)
        specs = [sp for sp in pffigure.standard_figures(set(data), registry)
                 if sp.name == "A-measured-by-order" or sp.name.startswith("B-")]
        (root / metric).mkdir(parents=True, exist_ok=True)
        index += [f"## {metric} ({gatebase})", ""]
        for sp in specs:
            series = pffigure.series_for(sp, data, registry)
            header = ["Generated by pysimcount.pfseeds -- do not edit; regenerate instead.",
                      f"figure: {sp.name} -- {sp.description}; median over seeds {seeds}, bars = min..max",
                      f"metric={metric} gatebase={gatebase} {'merged' if merged else 'unmerged'}"]
            for s in seeds:
                p = runs[s]["meta"]["provenance"]
                header.append(f"seed {s}: git_commit={p['git_commit']} git_dirty={p['git_dirty']}")
            tex = _errbar_tex(series, [errs[sc] for sc, _ in sp.members], ylabel, header)
            rel = f"{metric}/{sp.name}.tex"
            (root / rel).write_text(tex)
            desc = f"{sp.description}; median over seeds, bars min-max"
            index.append(f"- `{rel}` -- {desc}")
            pages.append(f"\\section*{{\\texttt{{{rel}}}}}\n{pffigure.latex_escape(desc)}\\par\\medskip\n"
                         f"\\input{{{rel}}}\n\\clearpage\n")
        index.append("")
    (root / "all.tex").write_text(pffigure.PREAMBLE + "\n".join(pages) + "\\end{document}\n")
    (root / "INDEX.md").write_text("\n".join(index))
    return root


# ------------------------------------------------------------------ reports

def spread_markdown(aggs: list[Agg], scheme_order: list[str], seeds: list[int]) -> str:
    lines = ["| metric | gate set | scheme | n | r from | " + " | ".join(f"seed {s}" for s in seeds)
             + " | median | max/min |",
             "|---|---|---|---:|---|" + "---:|" * (len(seeds) + 2)]
    rank = {s: i for i, s in enumerate(scheme_order)}
    for x in sorted(aggs, key=lambda x: (x.metric, x.gatebase, rank[x.scheme], x.n)):
        lines.append(f"| {x.metric} | {x.gatebase} | {x.scheme} | {x.n} | {x.r_source} | "
                     + " | ".join(f"{x.values[s]:,.0f}" for s in seeds)
                     + f" | {x.median:,.0f} | {x.ratio:.2f} |")
    return "\n".join(lines)


def spread_summary(aggs: list[Agg], scheme_order: list[str]) -> str:
    """Per (metric, gate set, scheme): the largest max/min over measured n and
    over fit n."""
    lines = ["| metric | gate set | scheme | max/min over seeds, measured n (at n) | max/min, fit n (at n) |",
             "|---|---|---|---|---|"]
    groups: dict[tuple[str, str, str], list[Agg]] = {}
    for x in aggs:
        groups.setdefault((x.metric, x.gatebase, x.scheme), []).append(x)
    rank = {s: i for i, s in enumerate(scheme_order)}
    for (m, gb, sc) in sorted(groups, key=lambda k: (k[0], k[1], rank[k[2]])):
        cells = []
        for src in ("measured", "fit"):
            xs = [x for x in groups[(m, gb, sc)] if x.r_source == src]
            if xs:
                w = max(xs, key=lambda x: x.ratio)
                cells.append(f"{w.ratio:.2f} (n={w.n})")
            else:
                cells.append("")
        lines.append(f"| {m} | {gb} | {sc} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def flips_markdown(flips: list[Flip], reference: int) -> str:
    if not flips:
        return f"No pairwise cost order differs from seed {reference} for any seed."
    lines = [f"| metric | gate set | n | r from | cheaper on seed {reference} | other | orders | seed | "
             f"seed {reference}: a / b | seed: a / b |",
             "|---|---|---:|---|---|---|---|---:|---|---|"]
    for f in flips:
        lines.append(f"| {f.metric} | {f.gatebase} | {f.n} | {f.r_source} | {f.a} | {f.b} | {f.order_a}/{f.order_b} | "
                     f"{f.seed} | {f.ref_a:,.0f} / {f.ref_b:,.0f} | {f.seed_a:,.0f} / {f.seed_b:,.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true", help="compile figures/merged/all.pdf with tectonic")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs([here / p for p in cfg["runs"]])
    seeds = sorted(runs)
    ref = int(cfg["reference_seed"])
    order = runs[ref]["meta"]["config"]["schemes"]

    all_aggs: list[Agg] = []
    flips: list[Flip] = []
    best_lines = ["| metric | gate set | n | r from | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                  "|---|---|---:|---|" + "---|" * len(seeds)]
    for metric, column, gb in SPREAD_METRICS:
        aggs = aggregate(runs, metric, column, gb, merged=True)
        all_aggs += aggs
        if metric != "r":
            flips += ranking_flips(aggs, ref)
            bests = {s: best_formula(aggs, s) for s in seeds}
            for k in sorted(bests[ref]):
                best_lines.append(f"| {metric} | {gb} | {k[0]} | {k[1]} | " + " | ".join(bests[s][k] for s in seeds) + " |")

    with open(out / "spread.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "gatebase", "scheme", "order", "n", "r_source"] + [f"seed{s}" for s in seeds]
                   + ["median", "min", "max", "max_over_min"])
        for x in all_aggs:
            w.writerow([x.metric, x.gatebase, x.scheme, x.order, x.n, x.r_source] + [x.values[s] for s in seeds]
                       + [x.median, x.lo, x.hi, x.ratio])
    (out / "spread.md").write_text("\n".join([
        f"# Spread across seeds {seeds} (merged counts)", "",
        "## Largest max/min per formula", "", spread_summary(all_aggs, order), "",
        "## Every point", "", spread_markdown(all_aggs, order, seeds), ""]))

    equal = [f for f in flips if f.order_a == f.order_b]
    (out / "rankings.md").write_text("\n".join([
        f"# Cost-order reversals relative to seed {ref} (merged counts)", "",
        "A row means: on the reference seed formula a is cheaper than b at this n; on the listed seed "
        "it is more expensive. Ties are not counted.", "",
        f"{len(flips)} reversals in total; {len(equal)} between formulas of equal order.", "",
        "## Cheapest formula per seed", "", "\n".join(best_lines), "",
        "## Equal-order reversals", "", flips_markdown(equal, ref), "",
        "## All reversals", "", flips_markdown(flips, ref), ""]))

    gains = paired_gains(runs)
    (out / "gains.md").write_text("\n".join([
        f"# Gain over the same-order Suzuki formula, per seed (seeds {seeds}, merged)", "",
        "gain = Suzuki cost / other cost on the same instance; > 1 favours the non-Suzuki formula.", "",
        gains_markdown(gains), ""]))
    wins = win_counts(runs)
    (out / "wins.md").write_text("\n".join([
        f"# Cheapest formula: number of seeds won (seeds {seeds}, merged)", "", wins_markdown(wins, len(seeds)), ""]))
    rat = bound_ratios(runs)
    if rat:
        (out / "bounds.md").write_text("\n".join([
            f"# Bound r / measured r, per seed (seeds {seeds}, cz)", "",
            "Denominator 'fit' marks extrapolated measured r.", "", bounds_markdown(rat), ""]))
    (out / "paired.json").write_text(json.dumps({"gains": gains, "wins": wins, "bound_ratios": rat}, indent=1))

    rep = []
    if "reproduce" in cfg:
        refdata = json.loads((here / cfg["reproduce"]["results"]).read_text())
        sources = tuple(cfg["reproduce"]["r_sources"])
        instances = tuple(cfg["reproduce"]["instances"]) if "instances" in cfg["reproduce"] else None
        n_cmp, diffs = compare_rows(runs[ref]["rows"], refdata["rows"], sources, instances)
        rep = [f"# Seed {ref} vs `{cfg['reproduce']['results']}`", "",
               f"Rows with r_source in {list(sources)}, matched on (instance, n, scheme, gate set, r_source, "
               f"merged); every field except {sorted(NONDETERMINISTIC)} compared.", "",
               f"Instances compared: {list(instances) if instances else 'all'}.", "",
               f"{n_cmp} rows matched; {len(diffs)} differences.", ""] + [f"- {d}" for d in diffs]
        (out / "reproduction.md").write_text("\n".join(rep) + "\n")

    meta = {"config": cfg, "provenance": provenance(),
            "runs": {s: runs[s]["meta"]["provenance"] for s in seeds}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_environment(out)
    root = write_figures(runs, out / "figures" / "merged")
    print("\n".join(rep))
    print(f"{len(flips)} reversals ({len(equal)} equal order); outputs in {out}")
    if args.render:
        print(f"rendered {pffigure.render(root)}")


if __name__ == "__main__":
    main()
