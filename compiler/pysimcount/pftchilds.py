"""T-count against system size in the format of pfchilds, by synthesising every extrapolated step.

    python -m pysimcount.pftchilds counts results/np-childs-format/t-config.json
    python -m pysimcount.pftchilds figure results/np-childs-format/t-config.json [--render]

pfchilds extrapolates a Clifford+Rz count as c(n) x r_fit(n), with c(n) the
per-step count, exactly affine in n and independent of r and of the seed. A
Clifford+T per-step count has neither property. Every rotation is synthesised
to

    eps_rot = eps_synth / (R r)        (synthesis.synthesise_step; R = rotations per step)

so a larger r asks for more precise rotations, and each costs more T gates
(about 3 log2(1/eps_rot), Ross & Selinger, arXiv:1403.2975); and the angles
are the seed's own field values. So this module builds and synthesises one
step per (formula, seed, n) instead of extrapolating a per-step count:

r       The ``ct`` rows of the runs (``pfchilds.load_run_set``, so the config
        can add np-main-n9 / -n11 to np-main): measured r at eps_trotter = 5e-4
        (the ct split, pfcompare.GATEBASES), merged. Each seed is fitted on
        its own (``pfcompare.fit_power_law`` on ``fit_sizes``), r_s(n) =
        ceil(a_s n^b_s) (pfcompare's rule). Nothing is re-measured.
step    ``pfcompare.step_counts`` (merged, the runs' template) on the seed's
        NumPy instance at n, t = n, step size t / r_s(n); then
        ``pfcompare.synthesised_counts`` with eps_rot = eps_synth / (R r_s(n))
        and the runs' backend. T total = T per step x r_s(n). Every
        synthesised word is checked against eps_rot (synthesis.synthesise).
check   Before any extrapolated size, the largest measured n is recomputed
        this way at each seed's measured r and must equal the run's T total
        exactly; and the fields of the instance must equal those recorded in
        the run.
values  Measured n: mean over seeds of the runs' T totals, bars min-max.
        Extrapolated n: mean over seeds of the per-seed T totals, bars
        min-max. DEVIATION from pfchilds, whose extrapolated point is the
        mean-r fit times c(n): here no single c(n) exists, and synthesising
        a further step at the mean-r fit on some seed's angles would mix
        seeds. The record stores every per-seed value (``per_seed``), so
        figleader ranks from them directly.

The formula set is copied from a pfchilds record (``like``), so the T figure
shows the same formulas as the CNOT figure; its pruning is not re-run.

Outputs, beside the config: the cache (``cache``; one entry per formula,
seed and n, written as each finishes, so an interrupted run resumes),
``<id>.json`` (record), ``<id>.tex`` (pfchilds-style figure).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import pfchilds, resources
from .pfcompare import (GATEBASES, fit_power_law, formula_registry, instance_hamiltonian, numpy_fields,
                        provenance, step_counts, synthesised_counts)

GATEBASE = "ct"


# ------------------------------------------------------------------ the numbers

def ct_rows(runs: dict[int, dict], schemes: list[str]) -> dict[tuple[str, int], dict[int, dict]]:
    """{(scheme, n): {seed: row}} of the measured, merged ct rows."""
    out: dict[tuple[str, int], dict[int, dict]] = {}
    for seed, d in runs.items():
        for row in d["rows"]:
            if not (row["gatebase"] == GATEBASE and row["r_source"] == "measured" and row["merged"]
                    and row["scheme"] in schemes):
                continue
            per = out.setdefault((row["scheme"], int(row["n"])), {})
            if seed in per:
                raise ValueError(f"seed {seed}: two ct rows for {row['scheme']} n={row['n']}")
            if int(row["t_total"]) != int(row["t_per_step"]) * int(row["r"]):
                raise ValueError(f"seed {seed} {row['scheme']} n={row['n']}: T total != T per step x r")
            per[seed] = row
    return out


def per_seed_fits(rows: dict, scheme: str, seeds: list[int], fit_sizes: list[int]) -> dict[int, tuple[float, float]]:
    """{seed: (a, b)}: each seed's measured ct r on ``fit_sizes``, fitted on its own."""
    out = {}
    for s in seeds:
        rs = []
        for n in fit_sizes:
            if s not in rows.get((scheme, n), {}):
                raise ValueError(f"{scheme} seed {s}: no measured ct r at n={n}")
            rs.append(int(rows[(scheme, n)][s]["r"]))
        a, b = fit_power_law(fit_sizes, rs)
        out[s] = (float(a), float(b))
    return out


def synthesised_step(scheme: str, seed: int, n: int, r: int, template: str, backend: str) -> dict:
    """One merged step at step size n / r on the seed's NumPy instance, synthesised
    with eps_rot = eps_synth / (rotations per step x r). Top level, so worker
    processes can run it."""
    t0 = time.perf_counter()
    eps_synth = GATEBASES[GATEBASE][1]
    H = instance_hamiltonian({"instances": "numpy", "seed": seed}, n)
    counts = step_counts(H, formula_registry()[scheme], float(n), r, True, template)
    rot = resources.rotations(counts)
    eps_rot = eps_synth / (rot * r)
    s = resources.summary(synthesised_counts(counts, eps_rot, backend))
    return {"scheme": scheme, "seed": seed, "n": n, "r": r, "rotations_per_step": rot, "eps_rot": eps_rot,
            "t_per_step": int(s["T"]), "t_total": int(s["T"]) * r, "seconds": round(time.perf_counter() - t0, 1)}


def check_against_runs(rows: dict, runs: dict[int, dict], schemes: list[str], seeds: list[int], n: int,
                       template: str, backend: str, workers: int, log=print) -> None:
    """Recompute n at each seed's measured r; must equal the run's T total exactly."""
    for s in seeds:
        if [float(h) for h in runs[s]["meta"]["fields"][str(n)]] != numpy_fields(n, s):
            raise ValueError(f"seed {s} n={n}: run's fields are not numpy_fields({n}, {s})")
    jobs = [(k, s, n, int(rows[(k, n)][s]["r"])) for k in schemes for s in seeds]
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(synthesised_step, *j, template, backend): j for j in jobs}
        for f in as_completed(futures):
            got = f.result()
            want = int(rows[(got["scheme"], n)][got["seed"]]["t_total"])
            if got["t_total"] != want:
                raise ValueError(f"check failed: {got['scheme']} seed {got['seed']} n={n}: "
                                 f"recomputed T {got['t_total']} != run's {want}")
    log(f"  check passed: {len(jobs)} steps at n={n} reproduce the runs' T totals exactly")


def _key(scheme: str, seed: int, n: int) -> str:
    return f"{scheme}|{seed}|{n}"


def run_counts(cfg: dict, here: Path, log=print) -> dict:
    like = json.loads((here / cfg["like"]).read_text())
    schemes, seeds = like["shown"], like["seeds"]
    runs = pfchilds.load_run_set(here, cfg["runs"])
    if sorted(runs) != seeds:
        raise ValueError(f"runs have seeds {sorted(runs)}, {cfg['like']} has {seeds}")
    run_cfgs = {(runs[s]["meta"]["config"]["template"], runs[s]["meta"]["config"]["synth_backend"]) for s in seeds}
    if len(run_cfgs) != 1:
        raise ValueError(f"runs differ in template / synthesis backend: {run_cfgs}")
    template, backend = run_cfgs.pop()
    rows = ct_rows(runs, schemes)
    workers = cfg["workers"]
    cache_path = here / cfg["cache"]
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"entries": {}}
    if not cache.get("checked"):
        check_against_runs(rows, runs, schemes, seeds, max(n for (_, n) in rows), template, backend, workers, log)
        cache["checked"] = True
    fits = {k: per_seed_fits(rows, k, seeds, cfg["fit_sizes"]) for k in schemes}
    jobs = []
    for n in sorted(cfg["extrapolate_sizes"], reverse=True):  # largest first: the long jobs start early
        for k in schemes:
            for s in seeds:
                r = pfchilds.r_fit_ceil(*fits[k][s], n)
                old = cache["entries"].get(_key(k, s, n))
                if old is not None:
                    if old["r"] != r:
                        raise ValueError(f"cache entry {_key(k, s, n)} has r={old['r']}, the fit now gives {r}")
                    continue
                jobs.append((k, s, n, r))
    log(f"  {len(jobs)} steps to synthesise ({len(cache['entries'])} cached), {workers} workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(synthesised_step, *j, template, backend) for j in jobs]
        for i, f in enumerate(as_completed(futures), start=1):
            e = f.result()
            cache["entries"][_key(e["scheme"], e["seed"], e["n"])] = e
            cache["provenance"] = provenance()
            cache_path.write_text(json.dumps(cache, indent=1))
            log(f"  [{i}/{len(jobs)} {time.perf_counter() - t0:7.0f}s] {e['scheme']:14s} seed {e['seed']:2d} "
                f"n={e['n']:3d} r={e['r']:>6} T={e['t_total']:.4g} ({e['seconds']} s)")
    return cache


# ------------------------------------------------------------------ record and figure

def build_record(cfg: dict, here: Path) -> dict:
    like = json.loads((here / cfg["like"]).read_text())
    schemes, seeds = like["shown"], like["seeds"]
    runs = pfchilds.load_run_set(here, cfg["runs"])
    rows = ct_rows(runs, schemes)
    cache = json.loads((here / cfg["cache"]).read_text())
    if not cache.get("checked"):
        raise ValueError("the cache has not passed the check against the runs; run `counts` first")
    series = {}
    for k in schemes:
        fits = per_seed_fits(rows, k, seeds, cfg["fit_sizes"])
        measured = []
        for n in sorted(n for (sc, n) in rows if sc == k):
            per = rows[(k, n)]
            if sorted(per) != seeds:
                raise ValueError(f"{k} n={n}: seeds {sorted(per)}, expected {seeds}")
            totals = {s: int(per[s]["t_total"]) for s in seeds}
            v = list(totals.values())
            measured.append({"n": n, "mean": math.fsum(v) / len(v), "min": min(v), "max": max(v),
                             "mean_r": math.fsum(int(per[s]["r"]) for s in seeds) / len(seeds),
                             "per_seed": {str(s): t for s, t in totals.items()}})
        extrap = []
        for n in cfg["extrapolate_sizes"]:
            per = {}
            for s in seeds:
                e = cache["entries"].get(_key(k, s, n))
                if e is None:
                    raise ValueError(f"no synthesised step for {k} seed {s} n={n}; run `counts`")
                if e["r"] != pfchilds.r_fit_ceil(*fits[s], n):
                    raise ValueError(f"cache entry {_key(k, s, n)} does not match the current fit")
                per[s] = e
            v = [per[s]["t_total"] for s in seeds]
            extrap.append({"n": n, "value": math.fsum(v) / len(v), "min": min(v), "max": max(v),
                           "per_seed": {str(s): per[s]["t_total"] for s in seeds},
                           "r_per_seed": {str(s): per[s]["r"] for s in seeds},
                           "eps_rot_per_seed": {str(s): per[s]["eps_rot"] for s in seeds},
                           "t_per_rotation_mean": math.fsum(per[s]["t_per_step"] / per[s]["rotations_per_step"]
                                                            for s in seeds) / len(seeds)})
        series[k] = {"measured": measured, "per_seed_fit": {str(s): list(f) for s, f in fits.items()},
                     "extrapolated": extrap}
    prov = provenance()
    lo, hi = min(cfg["fit_sizes"]), max(cfg["fit_sizes"])
    header = ["Generated by pysimcount.pftchilds -- do not edit; regenerate instead.",
              f"figure: {cfg['id']} -- {cfg['title']}",
              f"source: {pfchilds._run_source(cfg['runs'])} (seeds {seeds}); formulas as in {cfg['like']}",
              f"metric=t kind=xyz merged ({GATEBASE}: eps_trotter = {GATEBASES[GATEBASE][0]:g}, "
              f"eps_synth = {GATEBASES[GATEBASE][1]:g}, simcount template)",
              "measured n: mean over seeds of the run's T totals; bars min..max over seeds",
              f"extrapolated n: each seed's r = ceil(a_s n^b_s) fitted on n = {lo}--{hi}; one step built and "
              "synthesised per seed at that r; point = mean over seeds, bars min..max",
              f"shown: {', '.join(schemes)}",
              f"removed (as in {cfg['like']}): {', '.join(like['removed']) or 'none'}",
              f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
    if prov["git_dirty"]:
        header.append("WARNING: made from uncommitted code; the commit above does not reproduce it.")
    return {"id": cfg["id"], "title": cfg["title"], "kind": "xyz", "metric": "t", "gate_set": "Clifford+T (ct)",
            "r_method": "measured + per-seed power-law fit; every extrapolated step synthesised",
            "per_seed_stored": True, "seeds": seeds, "fit_sizes": cfg["fit_sizes"],
            "extrapolate_sizes": cfg["extrapolate_sizes"],
            "line_style": cfg.get("line_style", "extrapolated-dashed"),
            "shown": schemes, "removed": like["removed"],
            "header": header, "series": series}


def figure_tex(rec: dict, registry: dict) -> str:
    """pfchilds.figure_tex's layout: filled markers = measured, open = extrapolated;
    line dash per ``rec["line_style"]`` (``pfchilds.LINE_STYLES``)."""
    line_style = rec.get("line_style", "extrapolated-dashed")
    lo, hi = min(rec["fit_sizes"]), max(rec["fit_sizes"])
    out = [f"% {line}" for line in rec["header"]]
    out += ["\\begin{tikzpicture}", "  \\begin{loglogaxis}[",
            "    width=11cm, height=8cm,",
            "    xlabel={System size $n$},",
            "    ylabel={T count (Clifford+T)},",
            "    xtick={5,10,30,100,300}, xticklabels={5,10,30,100,300},",
            f"    xmin=3.5, xmax={max(rec['extrapolate_sizes']) * 1.2:g},",
            "    ymajorgrids=true,",
            "    legend pos=outer north east,",
            "    legend cell align=left,",
            "    legend style={font=\\footnotesize},",
            "  ]"]
    for k in rec["shown"]:
        label, colour, mark = pfchilds._style(k, registry)
        s = rec["series"][k]
        dash_m, dash_x, extra = pfchilds.line_dashes(k, registry, line_style, len(s["extrapolated"]))
        out.append(f"    % {k}: measured n, mean over seeds, bars min..max over seeds")
        out.append(f"    \\addplot[{colour}, {dash_m}, mark={mark}, mark size=2.2pt, mark options={{solid, fill={colour}}}, "
                   f"error bars/.cd, y dir=both, y explicit, error bar style={{solid}}] coordinates {{")
        for p in s["measured"]:
            out.append(f"      ({p['n']},{p['mean']:.6g}) += (0,{p['max'] - p['mean']:.6g}) -= (0,{p['mean'] - p['min']:.6g})")
        out += ["    };", f"    \\addlegendentry{{{label}}}"]
        out.append(f"    % {k}: extrapolated, per-seed fitted r, every step synthesised; mean over seeds, bars min..max")
        out.append(f"    \\addplot[{colour}, {dash_x}, mark={mark}, mark size=2.2pt, mark options={{solid, fill=white}}, "
                   f"{extra}forget plot, error bars/.cd, y dir=both, y explicit, error bar style={{solid}}] coordinates {{")
        last = s["measured"][-1]
        out.append(f"      ({last['n']},{last['mean']:.6g})")
        for p in s["extrapolated"]:
            out.append(f"      ({p['n']},{p['value']:.6g}) += (0,{p['max'] - p['value']:.6g}) -= (0,{p['value'] - p['min']:.6g})")
        out.append("    };")
    out += pfchilds.legend_key(line_style, "measured $r$ (mean over seeds)",
                               f"extrapolated: per-seed power-law fits of $r$ (n = {lo}--{hi}), steps synthesised")
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def run_figure(cfg: dict, here: Path, render: bool) -> dict:
    rec = build_record(cfg, here)
    (here / f"{cfg['id']}.json").write_text(json.dumps(rec, indent=1))
    tex = here / f"{cfg['id']}.tex"
    tex.write_text(figure_tex(rec, formula_registry()))
    print(f"  wrote {tex}")
    if render:
        pdf, _ = pfchilds.render_tex(tex, here / "rendered")
        print(f"  rendered {pdf}")
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "figure"])
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if args.stage == "counts":
        run_counts(cfg, args.config.parent)
    else:
        run_figure(cfg, args.config.parent, args.render)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
    main()
