"""Gate counts against system size in the format of Childs et al. (2018), Figs. 3 and 10.

    python -m pysimcount.pfchilds steps   results/np-childs-format/config.json
    python -m pysimcount.pfchilds figures results/np-childs-format/config.json [--render]

Childs et al. (2018), SI F.3 (p. 38): r measured for n = 5-12, *averaged over
five random field draws*, then one power law r = a n^b fitted per order; the
gate-count figures for n = 13-500 (Figs. 3, 10, 11) are that fitted r times
the per-step count. This module does the same for the NumPy seed studies:

r       At every measured n, the arithmetic mean of the measured r over seeds;
        ``pfcompare.fit_power_law`` on the config's ``fit_sizes``; extrapolated
        r = ceil(a n^b) (pfcompare's rule). Each seed is also fitted on its
        own, and the min-max of those per-seed curves is drawn as error bars.
        Deviations from the paper: 10 seeds (paper: 5), fit on n = 5-10
        (paper: 5-12; n = 9, 11, 12 not measured), r from
        ``fastempirical``'s window definition (paper: plain bisection).

counts  Clifford+Rz per-step counts do not depend on the field values (paper,
        App. I p. 53; checked here on every measured n and seed), and on the
        periodic chain they are affine in n. ``steps`` builds the step at the
        config's ``count_sizes``, checks that the counts are *exactly* affine
        (integer arithmetic on all sizes, not a least-squares fit), and checks
        the affine prediction against a direct build at ``check_sizes``. The
        large-n counts are then c(n) = alpha n + beta, exact for the
        unoptimised and two-group steps as far as the checks reach.
        Optimised steps (Nam Light reconstruction, ``pfoptimised``) are built
        and statevector-verified at ``optimised_sizes`` only; beyond them the
        affine form is not checked (labelled so in every figure).

values  Measured n: mean over seeds of the run's own totals. Extrapolated n:
        c(n) x r_fit(n) from the mean-r fit; bars from the per-seed fits.

pruning Each figure drops up to ``remove`` formulas, worst first by the
        geometric mean of the plotted values over every plotted n, but never
        a formula that is among the ``protect_top`` cheapest at *any* plotted
        n, and never below ``protect_top`` formulas. The ranking table is
        written next to each figure.

Outputs (config ``out``): ``step-counts.json`` (``steps``); per figure
``<id>.tex`` (one tikzpicture), ``<id>.json`` (attributes, values, ranking),
``<id>-ranking.md``; ``all.tex``; with ``--render`` per-figure PDFs and PNGs
in ``rendered/``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from . import pffigure, resources, twogroup
from .pfcompare import fit_power_law, formula_registry, instance_hamiltonian, provenance, step_counts
from .pfseeds import load_runs

# metric -> (np-main / np-grouping total column, per-step column, optimiser metric name, y label)
METRICS: dict[str, tuple[str, str, str, str]] = {
    "total": ("gates_total", "gates_per_step", "total", "Total gate count (Clifford+$R_z$)"),
    "cnot": ("cnot_total", "cnot_per_step", "CNOT", "CNOT count (Clifford+$R_z$)"),
}
SUMMARY_TO_METRIC = {"total": "Total excl. omega", "cnot": "CNOT"}
OPT_TO_METRIC = {"total": "total", "cnot": "CNOT"}

R_LABEL = "r from power-law fit to mean measured r (n = {lo}--{hi})"
OPT_HEADER = ("n > {hi}: optimised per-step count from linear fit (n = {lo}--{hi}); "
              "circuit not built or verified")
OPT_LEGEND = "Optimised count linear in n, fitted on n = {lo}--{hi}"


# ------------------------------------------------------------------ small pieces

def affine_exact(points: list[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    """(alpha, beta) with c = alpha n + beta through *every* point, in exact
    rational arithmetic. Raises if the points are not exactly affine or fewer
    than three are given (two points always fit a line)."""
    if len(points) < 3:
        raise ValueError("need at least three sizes to check an affine count")
    (n0, c0), (n1, c1) = points[0], points[-1]
    alpha = Fraction(c1 - c0, n1 - n0)
    beta = Fraction(c0) - alpha * n0
    for n, c in points:
        if alpha * n + beta != c:
            raise ValueError(f"counts not affine in n: {points}")
    return alpha, beta


def affine_eval(alpha: Fraction, beta: Fraction, n: int) -> int:
    v = alpha * n + beta
    if v.denominator != 1:
        raise ValueError(f"affine count {alpha} n + {beta} is not an integer at n = {n}")
    return int(v)


def r_fit_ceil(a: float, b: float, n: int) -> int:
    """pfcompare's extrapolation rule: ceil(a n^b)."""
    return math.ceil(a * n**b)


def geomean(values: list[float]) -> float:
    for v in values:
        if not v > 0:
            raise ValueError(f"geometric mean needs positive values, got {v}")
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def prune(curves: dict[str, dict[int, float]], remove: int, protect_top: int) -> tuple[list[str], list[dict]]:
    """(kept series ids, ranking records). ``curves``: id -> {n: plotted value}.

    Worst first by geometric mean over all plotted n; a series among the
    ``protect_top`` cheapest at any n is never removed; at least
    ``protect_top`` series are kept. Raises if the curves do not share their n.
    """
    ids = list(curves)
    sizes = sorted(curves[ids[0]])
    for k in ids:
        if sorted(curves[k]) != sizes:
            raise ValueError(f"{k} is plotted at {sorted(curves[k])}, {ids[0]} at {sizes}")
    best_rank: dict[str, int] = {k: len(ids) for k in ids}
    for n in sizes:
        # ties broken by id so the ranking is deterministic
        order = sorted(ids, key=lambda k: (curves[k][n], k))
        for rank, k in enumerate(order, start=1):
            best_rank[k] = min(best_rank[k], rank)
    gm = {k: geomean([curves[k][n] for n in sizes]) for k in ids}
    by_gm = sorted(ids, key=lambda k: (gm[k], k))
    removed: list[str] = []
    reason: dict[str, str] = {}
    for k in reversed(by_gm):
        if len(removed) == remove or len(ids) - len(removed) <= protect_top:
            break
        if best_rank[k] <= protect_top:
            reason[k] = f"kept: rank {best_rank[k]} at some n (protected)"
            continue
        removed.append(k)
        reason[k] = "removed"
    records = []
    for i, k in enumerate(by_gm, start=1):
        records.append({"series": k, "geomean": gm[k], "geomean_rank": i, "best_rank_any_n": best_rank[k],
                        "removed": k in removed, "note": reason.get(k, "kept")})
    kept = [k for k in ids if k not in removed]
    # the two cheapest at every n must survive -- checked again, independently
    for n in sizes:
        top = sorted(ids, key=lambda k: (curves[k][n], k))[:protect_top]
        for k in top:
            if k not in kept:
                raise AssertionError(f"pruning removed {k}, one of the {protect_top} cheapest at n = {n}")
    return kept, records


# ------------------------------------------------------------------ stage 1: per-step counts

def _registry_spec(key: str):
    reg = formula_registry()
    if key not in reg:
        raise ValueError(f"unknown scheme {key!r}")
    return reg[key]


def build_counts(kind: str, scheme: str, n: int, r: int) -> dict[str, int]:
    """Per-step {metric: count} of one step of size t/r, t = n, on NumPy seed 1's
    instance at n (counts do not depend on the fields; see module docstring).

    kind: 'xyz' (merged, simcount template -- np-main's merged rows),
    'twogroup-even-out' / 'twogroup-odd-out' (merged -- np-grouping's rows),
    'optimised' (Nam Light marginal per step on the merged xyz step, with
    pfoptimised's statevector soundness check)."""
    from .pfoptimised import optimise_case

    spec = _registry_spec(scheme)
    H = instance_hamiltonian({"instances": "numpy", "seed": 1}, n)
    t = float(n)
    if kind == "xyz":
        s = resources.summary(step_counts(H, spec, t, r, True, "simcount"))
        return {m: int(s[SUMMARY_TO_METRIC[m]]) for m in METRICS}
    if kind.startswith("twogroup-"):
        if n % 2:
            raise ValueError(f"two-group needs even n on the periodic chain, got {n}")
        tape = twogroup.build_step_schedule(H, twogroup.formula_group_schedule(spec, t / r, True), kind.split("-")[1])
        s = resources.summary(resources.count(tape))
        return {m: int(s[SUMMARY_TO_METRIC[m]]) for m in METRICS}
    if kind == "optimised":
        c = optimise_case(H, spec, n, r)
        return {m: int(c["marginal"][OPT_TO_METRIC[m]]) for m in METRICS}
    raise ValueError(f"unknown kind {kind!r}")


def _series_list(cfg: dict) -> list[tuple[str, str]]:
    """Every (kind, scheme) the figures need per-step counts for."""
    out = []
    for fig in cfg["figures"]:
        for scheme in fig["schemes"]:
            k = (fig["kind"], scheme)
            if k not in out:
                out.append(k)
    return out


def run_steps(cfg: dict, out: Path, log=print) -> dict:
    """Build per-step counts at every needed size; cache in step-counts.json
    (resumable: entries already present are not rebuilt)."""
    path = out / "step-counts.json"
    cache = json.loads(path.read_text()) if path.exists() else {"entries": {}}
    entries = cache["entries"]
    for kind, scheme in _series_list(cfg):
        if kind == "optimised":
            sizes = list(cfg["optimised_sizes"])
        else:
            sizes = list(cfg["count_sizes"]) + list(cfg["check_sizes"])
        for n in sizes:
            key = f"{kind}|{scheme}|{n}"
            if key in entries:
                continue
            # r only sets the rotation angles; 10 steps of size t/10 is a
            # representative step (counts are r-independent; checked in
            # ``figures`` against every measured row)
            t0 = time.perf_counter()
            c = build_counts(kind, scheme, n, 10)
            entries[key] = {"kind": kind, "scheme": scheme, "n": n, "r": 10, "counts": c,
                            "seconds": round(time.perf_counter() - t0, 2)}
            cache["provenance"] = provenance()
            path.write_text(json.dumps(cache, indent=1))
            log(f"  {kind:18s} {scheme:14s} n={n:3d}  {c}  ({entries[key]['seconds']} s)")
    return cache


def affine_models(cfg: dict, cache: dict) -> dict[tuple[str, str, str], dict]:
    """{(kind, scheme, metric): {alpha, beta, fit_sizes, checked_sizes}}.
    Raises if any count is not exactly affine on the fit sizes, or if a check
    size disagrees with the affine prediction."""
    models = {}
    entries = cache["entries"]
    for kind, scheme in _series_list(cfg):
        fit_sizes = cfg["optimised_sizes"] if kind == "optimised" else cfg["count_sizes"]
        checks = [] if kind == "optimised" else cfg["check_sizes"]
        for m in METRICS:
            pts = [(n, entries[f"{kind}|{scheme}|{n}"]["counts"][m]) for n in fit_sizes]
            alpha, beta = affine_exact(pts)
            for n in checks:
                direct = entries[f"{kind}|{scheme}|{n}"]["counts"][m]
                if affine_eval(alpha, beta, n) != direct:
                    raise ValueError(f"{kind} {scheme} {m}: affine prediction {affine_eval(alpha, beta, n)} "
                                     f"!= direct build {direct} at n = {n}")
            models[(kind, scheme, m)] = {"alpha": alpha, "beta": beta, "fit_sizes": list(fit_sizes),
                                         "checked_sizes": list(checks)}
    return models


# ------------------------------------------------------------------ stage 2: data per figure

# run settings that change a number; an extra run must match the aggregate's runs in all of them
RUN_SETTINGS = ("instances", "template", "synth_backend", "gatebases")


def load_run_set(here: Path, spec: str | dict, keys: tuple[str, ...] = ()) -> dict[int, dict]:
    """The runs a figure's ``runs`` names, relative to ``here``.

    A string: an aggregate.json; ``pfseeds.load_runs`` on its run list (refuses
    uncommitted code and configs differing in more than the seed).

    A dict ``{"aggregate": ..., "extra": [".../seed{seed}", ...], "dirty_exception":
    {"commit": ..., "reason": ...}}``: the aggregate's runs, plus each extra run
    directory's rows appended to the same seed's rows (np-main-n9 / -n11 add sizes
    that np-main lacks). Raises if an extra run is missing for a seed, names
    another seed, differs in any of ``RUN_SETTINGS``, repeats a row the seed
    already has, or was made from uncommitted code at any commit other than
    ``dirty_exception["commit"]`` -- an exception to pfseeds's rule that must be
    stated, with its reason, in the config. Each extra run is recorded under
    ``run["extra_provenance"]``."""
    if isinstance(spec, str):
        agg = json.loads((here / spec).read_text())
        return load_runs([(here / spec).parent / p for p in agg["runs"]], keys)
    runs = load_run_set(here, spec["aggregate"], keys)
    exc = spec.get("dirty_exception")
    for pattern in spec["extra"]:
        for seed, d in runs.items():
            path = here / pattern.format(seed=seed) / "results.json"
            x = json.loads(path.read_text())
            prov, cfg = x["meta"]["provenance"], x["meta"]["config"]
            if int(cfg["seed"]) != seed:
                raise ValueError(f"{path}: seed {cfg['seed']}, expected {seed}")
            if prov["git_dirty"] and (exc is None or not prov["git_commit"].startswith(exc["commit"])):
                raise ValueError(f"{path} was made from uncommitted code at {prov['git_commit']}, "
                                 "and the config states no exception for that commit")
            for k in RUN_SETTINGS:
                if cfg.get(k) != d["meta"]["config"].get(k):
                    raise ValueError(f"{path}: {k} = {cfg.get(k)!r}, the aggregate's runs {d['meta']['config'].get(k)!r}")
            have = {(r["scheme"], int(r["n"]), r["gatebase"], r["merged"], r["r_source"]) for r in d["rows"]}
            for r in x["rows"]:
                if (r["scheme"], int(r["n"]), r["gatebase"], r["merged"], r["r_source"]) in have:
                    raise ValueError(f"{path}: {r['scheme']} n={r['n']} {r['gatebase']} already in seed {seed}'s runs")
            d["rows"] = d["rows"] + x["rows"]
            fields = d["meta"].setdefault("fields", {})
            for size, values in x["meta"].get("fields", {}).items():
                if size in fields and fields[size] != values:
                    raise ValueError(f"{path}: fields at n={size} differ from seed {seed}'s runs")
                fields[size] = values
            d.setdefault("extra_provenance", []).append(
                {"path": str(path.relative_to(here)) if path.is_relative_to(here) else str(path),
                 "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"]})
    return runs


def _run_source(spec: str | dict) -> str:
    if isinstance(spec, str):
        return spec
    return f"{spec['aggregate']} + {', '.join(spec['extra'])}"


def _measured_rows(kind: str, runs: dict[int, dict]) -> dict[tuple[str, int], dict[int, dict]]:
    """{(scheme, n): {seed: row}} of the rows a figure of this kind plots."""
    out: dict[tuple[str, int], dict[int, dict]] = {}
    for seed, d in runs.items():
        for row in d["rows"]:
            if kind == "xyz":
                if not (row["r_source"] == "measured" and row["gatebase"] == "cz" and row["merged"]):
                    continue
            elif kind.startswith("twogroup-"):
                if not (row["variant"] == kind and row["merged"]):
                    continue
            elif kind == "optimised":
                if row["metric"] != "CNOT":  # one row per (scheme, n) for r; totals read per metric below
                    continue
            per = out.setdefault((row["scheme"], int(row["n"])), {})
            if seed in per:
                raise ValueError(f"seed {seed}: two rows for {row['scheme']} n={row['n']}")
            per[seed] = row
    return out


def _opt_totals(runs: dict[int, dict]) -> dict[tuple[str, int, str], dict[int, tuple[int, int]]]:
    """{(scheme, n, metric): {seed: (marginal_per_step, marginal_total)}} from pfoptimised rows."""
    out: dict[tuple[str, int, str], dict[int, tuple[int, int]]] = {}
    for seed, d in runs.items():
        for row in d["rows"]:
            for m, name in OPT_TO_METRIC.items():
                if row["metric"] == name:
                    out.setdefault((row["scheme"], int(row["n"]), m), {})[seed] = (
                        int(row["marginal_per_step"]), int(row["marginal_total"]))
    return out


def figure_data(fig: dict, cfg: dict, runs: dict[int, dict], models: dict) -> dict:
    """Per series: measured points (mean, min, max over seeds), fit
    parameters, and extrapolated points (value, per-seed-fit min, max)."""
    kind, metric = fig["kind"], fig["metric"]
    total_col, step_col = METRICS[metric][0], METRICS[metric][1]
    seeds = sorted(runs)
    rows = _measured_rows(kind, runs)
    opt = _opt_totals(runs) if kind == "optimised" else {}
    fit_sizes = list(fig["fit_sizes"])
    series = {}
    for scheme in fig["schemes"]:
        sizes = sorted(n for (sc, n) in rows if sc == scheme)
        if not sizes:
            raise ValueError(f"{fig['id']}: no measured rows for {scheme}")
        model = models[(kind, scheme, metric)]
        r_by_n: dict[int, dict[int, int]] = {}
        measured = []
        for n in sizes:
            per = rows[(scheme, n)]
            if sorted(per) != seeds:
                raise ValueError(f"{fig['id']}: {scheme} n={n} has seeds {sorted(per)}, expected {seeds}")
            r_by_n[n] = {s: int(per[s]["r"]) for s in seeds}
            totals, steps = [], set()
            for s in seeds:
                if kind == "optimised":
                    step, tot = opt[(scheme, n, metric)][s]
                else:
                    step, tot = int(per[s][step_col]), int(per[s][total_col])
                if tot != step * r_by_n[n][s]:
                    raise ValueError(f"{fig['id']}: {scheme} n={n} seed {s}: total {tot} != per step x r")
                steps.add(step)
                totals.append(tot)
            if len(steps) != 1:
                raise ValueError(f"{fig['id']}: {scheme} n={n}: per-step count differs across seeds {sorted(steps)}")
            step = steps.pop()
            if n in model["fit_sizes"] + model["checked_sizes"] or n >= min(model["fit_sizes"]):
                if affine_eval(model["alpha"], model["beta"], n) != step:
                    raise ValueError(f"{fig['id']}: {scheme} n={n}: run's per-step {metric} {step} != "
                                     f"affine model {affine_eval(model['alpha'], model['beta'], n)}")
            measured.append({"n": n, "mean": math.fsum(totals) / len(totals), "min": min(totals),
                             "max": max(totals), "mean_r": math.fsum(r_by_n[n].values()) / len(seeds)})
        missing = [n for n in fit_sizes if n not in r_by_n]
        if missing:
            raise ValueError(f"{fig['id']}: {scheme} has no measured r at fit sizes {missing}")
        mean_r = [math.fsum(r_by_n[n].values()) / len(seeds) for n in fit_sizes]
        a, b = fit_power_law(fit_sizes, mean_r)
        per_seed_fit = {s: fit_power_law(fit_sizes, [r_by_n[n][s] for n in fit_sizes]) for s in seeds}
        extrap = []
        for n in fig["extrapolate_sizes"]:
            c = affine_eval(model["alpha"], model["beta"], n)
            r = r_fit_ceil(a, b, n)
            band = [c * r_fit_ceil(*per_seed_fit[s], n) for s in seeds]
            extrap.append({"n": n, "r": r, "per_step": c, "value": c * r, "min": min(band), "max": max(band)})
        series[scheme] = {"measured": measured, "fit": {"a": a, "b": b, "sizes": fit_sizes},
                          "per_seed_fit": {str(s): list(v) for s, v in per_seed_fit.items()},
                          "per_step_model": {"alpha": str(model["alpha"]), "beta": str(model["beta"]),
                                             "fit_sizes": model["fit_sizes"],
                                             "checked_sizes": model["checked_sizes"]},
                          "extrapolated": extrap}
    return series


# ------------------------------------------------------------------ stage 2: tikz

def _style(scheme: str, registry: dict) -> tuple[str, str, str]:
    f = registry[scheme]
    colour = pffigure.COLOUR_OVERRIDE.get(scheme, pffigure.FAMILY_COLOUR.get(f.family))
    if colour is None or f.order not in pffigure.ORDER_MARK:
        raise ValueError(f"no pffigure style for {scheme}")
    return pffigure.latex_escape(f.name), colour, pffigure.ORDER_MARK[f.order]


def figure_tex(fig: dict, series: dict, kept: list[str], header: list[str], registry: dict) -> str:
    """One loglog tikzpicture: filled markers + solid line = measured (mean
    over seeds), open markers + dashed line = extrapolated; error bars
    min-max over seeds (per-seed fits when extrapolated)."""
    looks: dict[tuple[str, str], str] = {}
    out = [f"% {line}" for line in header]
    out += [
        "\\begin{tikzpicture}",
        "  \\begin{loglogaxis}[",
        "    width=11cm, height=8cm,",
        "    xlabel={System size $n$},",
        f"    ylabel={{{METRICS[fig['metric']][3]}}},",
        "    xtick={5,10,30,100,300}, xticklabels={5,10,30,100,300},",
        f"    xmin=3.5, xmax={max(fig['extrapolate_sizes']) * 1.2:g},",
        "    ymajorgrids=true,",
        "    legend pos=outer north east,",
        "    legend cell align=left,",
        "    legend style={font=\\footnotesize},",
        "  ]",
    ]
    for scheme in kept:
        label, colour, mark = _style(scheme, registry)
        if (colour, mark) in looks:
            raise ValueError(f"{scheme} would look identical to {looks[(colour, mark)]} in {fig['id']}")
        looks[(colour, mark)] = scheme
        s = series[scheme]
        out.append(f"    % {scheme}: measured n, mean over seeds, bars min..max over seeds")
        out.append(f"    \\addplot[{colour}, solid, mark={mark}, mark size=2.2pt, mark options={{solid, fill={colour}}}, "
                   f"error bars/.cd, y dir=both, y explicit, error bar style={{solid}}] coordinates {{")
        for p in s["measured"]:
            out.append(f"      ({p['n']},{p['mean']:.6g}) += (0,{p['max'] - p['mean']:.6g}) -= (0,{p['mean'] - p['min']:.6g})")
        out += ["    };", f"    \\addlegendentry{{{label}}}"]
        out.append(f"    % {scheme}: extrapolated, r = ceil({s['fit']['a']:.6g} n^{s['fit']['b']:.6g}), "
                   f"per step = {s['per_step_model']['alpha']} n + {s['per_step_model']['beta']}; "
                   "bars min..max of per-seed fits")
        out.append(f"    \\addplot[{colour}, dashed, mark={mark}, mark size=2.2pt, mark options={{solid, fill=white}}, "
                   f"forget plot, error bars/.cd, y dir=both, y explicit, error bar style={{solid}}] coordinates {{")
        last = s["measured"][-1]
        out.append(f"      ({last['n']},{last['mean']:.6g})")  # joins the dashed line to the last measured point
        for p in s["extrapolated"]:
            out.append(f"      ({p['n']},{p['value']:.6g}) += (0,{p['max'] - p['value']:.6g}) -= (0,{p['value'] - p['min']:.6g})")
        out.append("    };")
    lo, hi = min(fig["fit_sizes"]), max(fig["fit_sizes"])
    out += ["    \\addlegendimage{black, solid, mark=*, mark options={solid, fill=black}}",
            "    \\addlegendentry{measured $r$ (mean over seeds)}",
            "    \\addlegendimage{black, dashed, mark=*, mark options={solid, fill=white}}",
            f"    \\addlegendentry{{{R_LABEL.format(lo=lo, hi=hi).replace(' r ', ' $r$ ').replace('r from', '$r$ from')}}}"]
    if fig["kind"] == "optimised":
        olo, ohi = fig["optimised_sizes"][0], fig["optimised_sizes"][-1]
        out += ["    \\addlegendimage{empty legend}",
                f"    \\addlegendentry{{{OPT_LEGEND.format(lo=olo, hi=ohi)}}}"]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def ranking_md(fig: dict, records: list[dict], series: dict) -> str:
    sizes = [p["n"] for p in next(iter(series.values()))["measured"]] + list(fig["extrapolate_sizes"])
    lines = [f"# Ranking for `{fig['id']}`", "",
             f"Geometric mean of the plotted {fig['metric']} values over n = {sizes}; best rank at any plotted n "
             f"(1 = cheapest). Up to {fig['remove']} removed, worst first, never one ranked <= "
             f"{fig['protect_top']} at any n.", "",
             "| geo-mean rank | formula | geometric mean | best rank at any n | removed | note |",
             "|---:|---|---:|---:|---|---|"]
    for r in records:
        lines.append(f"| {r['geomean_rank']} | {r['series']} | {r['geomean']:.4g} | {r['best_rank_any_n']} | "
                     f"{'yes' if r['removed'] else 'no'} | {r['note']} |")
    return "\n".join(lines) + "\n"


def _plotted(series: dict) -> dict[str, dict[int, float]]:
    out = {}
    for k, s in series.items():
        v = {p["n"]: p["mean"] for p in s["measured"]}
        v.update({p["n"]: p["value"] for p in s["extrapolated"]})
        out[k] = v
    return out


def run_figures(cfg: dict, here: Path, out: Path, render: bool, only: str | None = None) -> list[dict]:
    """Every figure of the config (or those whose id starts with ``only``)."""
    if only:
        cfg = {**cfg, "figures": [f for f in cfg["figures"] if f["id"].startswith(only)]}
        if not cfg["figures"]:
            raise ValueError(f"no figure id starts with {only!r}")
    registry = formula_registry()
    cache = json.loads((out / "step-counts.json").read_text())
    models = affine_models(cfg, cache)
    prov = provenance()
    runs_cache: dict[str, dict[int, dict]] = {}
    pages, records = [], []
    for fig in cfg["figures"]:
        fig = {**cfg["defaults"], "optimised_sizes": cfg["optimised_sizes"], **fig}
        spec = fig["runs"]
        src = _run_source(spec)
        if src not in runs_cache:
            keys = ("source",) if fig["kind"] == "optimised" else ()
            runs_cache[src] = load_run_set(here, spec, keys)
        runs = runs_cache[src]
        series = figure_data(fig, cfg, runs, models)
        kept, ranking = prune(_plotted(series), fig["remove"], fig["protect_top"])
        seeds = sorted(runs)
        lo, hi = min(fig["fit_sizes"]), max(fig["fit_sizes"])
        header = [
            "Generated by pysimcount.pfchilds -- do not edit; regenerate instead.",
            f"figure: {fig['id']} -- {fig['title']}",
            f"source: {src} (seeds {seeds}); run commits: "
            + ", ".join(sorted({runs[s]['meta']['provenance']['git_commit'] for s in seeds}
                               | {e["git_commit"] for s in seeds for e in runs[s].get("extra_provenance", [])})),
            f"metric={fig['metric']} kind={fig['kind']} merged (cz, simcount template)",
            "measured n: mean over seeds of the run's totals; bars min..max over seeds",
            R_LABEL.format(lo=lo, hi=hi) + "; r = ceil(a n^b); bars = min..max of per-seed fits",
            "per-step counts exactly affine in n (checked on "
            + (f"n = {fig['optimised_sizes']}" if fig["kind"] == "optimised"
               else f"n = {cfg['count_sizes']}, direct build at n = {cfg['check_sizes']}") + ")",
            f"shown: {', '.join(kept)}",
            f"removed (see {fig['id']}-ranking.md): {', '.join(k for k in fig['schemes'] if k not in kept) or 'none'}",
            f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}",
        ]
        if fig["kind"] == "optimised":
            header.insert(7, OPT_HEADER.format(lo=fig["optimised_sizes"][0], hi=fig["optimised_sizes"][-1]))
        if prov["git_dirty"]:
            header.append("WARNING: made from uncommitted code; the commit above does not reproduce it.")
        (out / f"{fig['id']}.tex").write_text(figure_tex(fig, series, kept, header, registry))
        (out / f"{fig['id']}-ranking.md").write_text(ranking_md(fig, ranking, series))
        rec = {"id": fig["id"], "title": fig["title"], "experiment": fig["experiment"], "kind": fig["kind"],
               "metric": fig["metric"], "gate_set": "Clifford+Rz (cz)", "r_method": "measured + power-law fit",
               "seeds": seeds, "fit_sizes": fig["fit_sizes"], "extrapolate_sizes": fig["extrapolate_sizes"],
               "shown": kept, "removed": [k for k in fig["schemes"] if k not in kept], "ranking": ranking,
               "header": header, "series": series}
        (out / f"{fig['id']}.json").write_text(json.dumps(rec, indent=1))
        records.append(rec)
        pages.append(f"\\section*{{\\texttt{{{pffigure.latex_escape(fig['id'])}}}}}\n"
                     f"{pffigure.latex_escape(fig['title'])}\\par\\medskip\n\\input{{{fig['id']}.tex}}\n\\clearpage\n")
        print(f"  {fig['id']:40s} shown {len(kept)}, removed {rec['removed']}")
    (out / "all.tex").write_text(pffigure.PREAMBLE + "\n".join(pages) + "\\end{document}\n")
    if render:
        render_each(out, [r["id"] for r in records])
    return records


STANDALONE = """\\documentclass[border=4pt]{standalone}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\begin{document}
\\input{%s}
\\end{document}
"""


def render_tex(tex: Path, outdir: Path) -> tuple[Path, Path]:
    """Compile one tikzpicture file to ``outdir/<stem>.pdf`` (standalone,
    tectonic) and a 1400-px-wide ``.png`` preview (macOS Quick Look,
    ``qlmanage``). Raises if either tool is missing or fails."""
    tect, ql = shutil.which("tectonic"), shutil.which("qlmanage")
    if tect is None or ql is None:
        raise RuntimeError("needs tectonic and qlmanage on PATH")
    outdir.mkdir(parents=True, exist_ok=True)
    wrapper = outdir / f"{tex.stem}.wrapper.tex"
    wrapper.write_text(STANDALONE % tex.resolve().as_posix())
    subprocess.run([tect, "--chatter", "minimal", wrapper.name], cwd=outdir, check=True)
    pdf = outdir / f"{tex.stem}.pdf"
    (outdir / f"{tex.stem}.wrapper.pdf").replace(pdf)
    wrapper.unlink()
    subprocess.run([ql, "-t", "-s", "1400", "-o", outdir.as_posix(), pdf.as_posix()], check=True, capture_output=True)
    png = outdir / f"{tex.stem}.png"
    made = outdir / f"{tex.stem}.pdf.png"
    if not made.exists():
        raise RuntimeError(f"qlmanage made no preview for {pdf}")
    made.replace(png)
    return pdf, png


def render_each(out: Path, ids: list[str]) -> None:
    for i in ids:
        pdf, _ = render_tex(out / f"{i}.tex", out / "rendered")
        print(f"  rendered {pdf}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["steps", "figures"])
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--only", help="figures stage: only figure ids starting with this prefix")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "steps":
        run_steps(cfg, out)
        affine_models(cfg, json.loads((out / "step-counts.json").read_text()))
        print("per-step counts affine and checked")
    else:
        run_figures(cfg, here, out, args.render, args.only)


if __name__ == "__main__":
    main()
