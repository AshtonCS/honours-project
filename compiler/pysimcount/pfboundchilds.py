"""Gate counts against n with r from a norm bound (``genbound``), in the pfchilds / figleader format.

    python -m pysimcount.pfboundchilds cnot    results/np-bound-min/config.json [--render]
    python -m pysimcount.pfboundchilds tcounts results/np-bound-min/config.json
    python -m pysimcount.pfboundchilds t       results/np-bound-min/config.json [--render]

The measured-r figures (pfchilds, pftchilds, figleader) take r from the error
search at n <= 11 and from a power-law fit beyond. Here r comes from a bound
instead, so it is *computed* at every plotted n, 4 to 500: nothing is
measured and nothing is extrapolated. Every point is drawn filled.

r       ``genbound.steps(spec, H_seed(n), t = n, eps_trotter, **cfg["bound"])``
        on each NumPy seed's instance. eps_trotter from ``GATEBASES``: cz
        1e-3 (CNOT), ct 5e-4 (T). With budget "count" the bound depends only
        on m = 4n, Lambda = 1, t and eps, so r is the same on every seed;
        it is computed per seed anyway and the record says whether it varied.
CNOT    c(n) x r, c(n) the exact affine per-step count from pfchilds's
        step-counts cache (checked there at n = 10, 20, 40 and by a direct
        build at n = 100; ``check_affine_against_runs`` also checks it
        against every np-main run at n = 4-11).
T       One merged step per (formula, seed, n), synthesised with
        eps_rot = eps_synth / (R r) by ``pftchilds.synthesised_step`` (the
        same function as the measured-r T figures), cached and resumable.
        The formula set is the CNOT figure's (pruning not re-run), as in
        pftchilds.
pruning pfchilds.prune with the config's ``remove`` / ``protect_top``.

Per metric this writes ``<id>.json`` (a record with ``per_seed_stored``, so
figleader ranks from it), ``<id>.tex`` (all n, one plot per formula),
``<id>-with-zoom.tex`` (figleader's zoom layout) and ``<id>-n<N>-dots.tex``
(figleader's dot plot), with ``--render`` also PDFs/PNGs in ``rendered/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import figleader, genbound, pfchilds, pftchilds
from .compare import GATEBASES
from .pfcompare import formula_registry, instance_hamiltonian, provenance

BOUND_KEYS = ("kind", "budget", "constant")


# ------------------------------------------------------------------ r

def bound_label(bound: dict) -> str:
    """Short text for headers and legends."""
    for k in BOUND_KEYS:
        if k not in bound:
            raise ValueError(f"config bound needs {BOUND_KEYS}, missing {k!r}")
    if (bound["kind"], bound["budget"], bound["constant"]) == ("min", "count", "childs"):
        return "minimised norm bound (Childs et al. 2018, Def. 2; extended to every formula)"
    return f"norm bound kind={bound['kind']} budget={bound['budget']} constant={bound['constant']}"


GATEBASES_USED = ("cz", "ct")


def bound_r(scheme: str, seed: int, n: int, bound: dict) -> dict[str, int]:
    """{gatebase: r} for one formula on one seed's instance at n, t = n, eps = each
    gatebase's Trotter share. Top level, so worker processes can run it."""
    spec = formula_registry()[scheme]
    H = instance_hamiltonian({"instances": "numpy", "seed": seed}, n)
    return {g: genbound.steps(spec, H, float(n), GATEBASES[g][0], kind=bound["kind"],
                              budget=bound["budget"], constant=bound["constant"]) for g in GATEBASES_USED}


def r_table(cfg: dict, here: Path, log=print) -> dict[str, dict[str, dict[int, dict[int, int]]]]:
    """{gatebase: {scheme: {n: {seed: r}}}} for every configured formula, seed and size.

    Cached in ``r-table.json`` beside the config, keyed by the bound settings;
    a cache made with other settings, seeds, sizes or formulas raises rather
    than being silently reused or overwritten."""
    path = here / "r-table.json"
    want = {"bound": cfg["bound"], "seeds": cfg["seeds"], "sizes": cfg["sizes"], "schemes": cfg["schemes"],
            "eps_trotter": {g: GATEBASES[g][0] for g in GATEBASES_USED}}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["settings"] != want:
            raise ValueError(f"{path} was made with {cached['settings']}, config now gives {want}; move it aside")
        raw = cached["r"]
    else:
        jobs = [(k, s, n) for n in sorted(cfg["sizes"], reverse=True) for k in cfg["schemes"] for s in cfg["seeds"]]
        raw = {}
        t0 = time.perf_counter()
        with ProcessPoolExecutor(cfg.get("workers", 6)) as pool:
            futures = {pool.submit(bound_r, k, s, n, cfg["bound"]): (k, s, n) for k, s, n in jobs}
            for f in as_completed(futures):
                k, s, n = futures[f]
                raw[_key(k, s, n)] = f.result()
        log(f"  r for {len(jobs)} (formula, seed, n) computed in {time.perf_counter() - t0:.0f} s")
        path.write_text(json.dumps({"settings": want, "provenance": provenance(), "r": raw}, indent=1))
    return {g: {k: {n: {s: raw[_key(k, s, n)][g] for s in cfg["seeds"]} for n in cfg["sizes"]}
                for k in cfg["schemes"]} for g in GATEBASES_USED}


# ------------------------------------------------------------------ CNOT

def affine_cnot(cfg: dict, here: Path) -> dict[str, dict]:
    """{scheme: affine CNOT model} from pfchilds's checked step-counts cache."""
    pc = json.loads((here / cfg["step_counts"]["config"]).read_text())
    cache = json.loads((here / cfg["step_counts"]["cache"]).read_text())
    models = pfchilds.affine_models(pc, cache)
    out = {}
    for k in cfg["schemes"]:
        if ("xyz", k, "cnot") not in models:
            raise ValueError(f"no affine CNOT model for {k} in {cfg['step_counts']['cache']}")
        out[k] = models[("xyz", k, "cnot")]
    return out


def check_affine_against_runs(cfg: dict, here: Path, models: dict[str, dict]) -> int:
    """The affine per-step CNOT count must equal every merged cz run row's (n = 4-11)."""
    runs = pfchilds.load_run_set(here, cfg["runs"])
    checked = 0
    for seed, d in runs.items():
        for row in d["rows"]:
            if not (row["gatebase"] == "cz" and row["merged"] and row["r_source"] == "measured"
                    and row["scheme"] in models):
                continue
            m = models[row["scheme"]]
            pred = pfchilds.affine_eval(m["alpha"], m["beta"], int(row["n"]))
            if pred != int(row["cnot_per_step"]):
                raise ValueError(f"seed {seed} {row['scheme']} n={row['n']}: affine CNOT/step {pred} "
                                 f"!= run's {row['cnot_per_step']}")
            checked += 1
    if checked == 0:
        raise ValueError("no run rows to check the affine model against")
    return checked


def point(n: int, per_seed: dict[int, int], r_per_seed: dict[int, int], **extra) -> dict:
    v = list(per_seed.values())
    return {"n": n, "mean": math.fsum(v) / len(v), "min": min(v), "max": max(v),
            "per_seed": {str(s): x for s, x in per_seed.items()},
            "r_per_seed": {str(s): x for s, x in r_per_seed.items()}, **extra}


def cnot_series(cfg: dict, here: Path) -> tuple[dict, int]:
    models = affine_cnot(cfg, here)
    n_checked = check_affine_against_runs(cfg, here, models)
    rs = r_table(cfg, here)["cz"]
    series = {}
    for k in cfg["schemes"]:
        pts = []
        for n in cfg["sizes"]:
            c = pfchilds.affine_eval(models[k]["alpha"], models[k]["beta"], n)
            pts.append(point(n, {s: c * r for s, r in rs[k][n].items()}, rs[k][n], per_step=c))
        series[k] = {"measured": pts, "extrapolated": [],
                     "per_step_model": {"alpha": str(models[k]["alpha"]), "beta": str(models[k]["beta"])}}
    return series, n_checked


# ------------------------------------------------------------------ T

def _key(scheme: str, seed: int, n: int) -> str:
    return f"{scheme}|{seed}|{n}"


def run_tcounts(cfg: dict, here: Path, log=print) -> dict:
    """Synthesise one step per (shown formula, seed, n) at the bound's ct r; resumable."""
    rec = json.loads((here / f"{cfg['cnot']['id']}.json").read_text())
    schemes = rec["shown"]
    tc = cfg["t"]
    path = here / tc["cache"]
    cache = json.loads(path.read_text()) if path.exists() else {"entries": {}}
    rct = r_table(cfg, here)["ct"]
    jobs = []
    for n in sorted(cfg["sizes"], reverse=True):  # largest first: the long jobs start early
        for k in schemes:
            for s in cfg["seeds"]:
                r = rct[k][n][s]
                old = cache["entries"].get(_key(k, s, n))
                if old is not None:
                    if old["r"] != r:
                        raise ValueError(f"cache entry {_key(k, s, n)} has r={old['r']}, the bound gives {r}")
                    continue
                jobs.append((k, s, n, r))
    log(f"  {len(jobs)} steps to synthesise ({len(cache['entries'])} cached), {tc['workers']} workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(tc["workers"]) as pool:
        futures = [pool.submit(pftchilds.synthesised_step, *j, tc["template"], tc["backend"]) for j in jobs]
        for i, f in enumerate(as_completed(futures), start=1):
            e = f.result()
            cache["entries"][_key(e["scheme"], e["seed"], e["n"])] = e
            cache["bound"] = cfg["bound"]
            cache["provenance"] = provenance()
            path.write_text(json.dumps(cache, indent=1))
            log(f"  [{i}/{len(jobs)} {time.perf_counter() - t0:7.0f}s] {e['scheme']:14s} seed {e['seed']:2d} "
                f"n={e['n']:3d} r={e['r']:>12} eps_rot={e['eps_rot']:.2e} T={e['t_total']:.4g} ({e['seconds']} s)")
    return cache


def t_series(cfg: dict, here: Path, schemes: list[str]) -> dict:
    cache = json.loads((here / cfg["t"]["cache"]).read_text())
    if cache.get("bound") != cfg["bound"]:
        raise ValueError(f"T cache was made with bound {cache.get('bound')}, config has {cfg['bound']}")
    rct = r_table(cfg, here)["ct"]
    series = {}
    for k in schemes:
        pts = []
        for n in cfg["sizes"]:
            per, rs, eps = {}, {}, {}
            for s in cfg["seeds"]:
                e = cache["entries"].get(_key(k, s, n))
                if e is None:
                    raise ValueError(f"no synthesised step for {k} seed {s} n={n}; run `tcounts`")
                if e["r"] != rct[k][n][s]:
                    raise ValueError(f"cache entry {_key(k, s, n)} does not match the bound's r")
                per[s], rs[s], eps[s] = e["t_total"], e["r"], e["eps_rot"]
            t_per_rot = math.fsum(cache["entries"][_key(k, s, n)]["t_per_step"]
                                  / cache["entries"][_key(k, s, n)]["rotations_per_step"] for s in cfg["seeds"])
            pts.append(point(n, per, rs, eps_rot_per_seed={str(s): x for s, x in eps.items()},
                             t_per_rotation_mean=t_per_rot / len(cfg["seeds"])))
        series[k] = {"measured": pts, "extrapolated": []}
    return series


# ------------------------------------------------------------------ record and tikz

def make_record(cfg: dict, mcfg: dict, metric: str, gate_set: str, series: dict, kept: list[str],
                ranking: list[dict] | None, header: list[str]) -> dict:
    r_varies = any(len(set(p["r_per_seed"].values())) > 1 for s in series.values() for p in s["measured"])
    return {"id": mcfg["id"], "title": mcfg["title"], "kind": "xyz", "metric": metric, "gate_set": gate_set,
            "r_method": f"bound: {bound_label(cfg['bound'])}", "bound": cfg["bound"],
            "per_seed_stored": True, "seeds": cfg["seeds"], "sizes": cfg["sizes"],
            "fit_sizes": [], "extrapolate_sizes": [], "r_varies_over_seeds": r_varies,
            "line_style": cfg["line_style"], "shown": kept,
            "removed": [k for k in series if k not in kept] if ranking is not None else mcfg.get("removed", []),
            "ranking": ranking, "header": header,
            "series": {k: series[k] for k in kept}}


def bound_key(cfg: dict) -> list[str]:
    """The one black key entry: every point is a computed bound."""
    return ["    \\addlegendimage{only marks, black, mark=*, mark options={solid, fill=black}}",
            "    \\addlegendentry{$r$ from the bound, computed at every $n$}"]


def figure_tex(rec: dict, mcfg: dict, registry: dict) -> str:
    """All n on one log-log axis, one plot per formula, filled markers; bars
    min..max over seeds; line dash per ``rec['line_style']``."""
    out = [f"% {line}" for line in rec["header"]]
    out += ["\\begin{tikzpicture}", "  \\begin{loglogaxis}[",
            "    width=11cm, height=8cm,",
            "    xlabel={System size $n$},",
            f"    ylabel={{{mcfg['ylabel']}}},",
            "    xtick={5,10,30,100,300}, xticklabels={5,10,30,100,300},",
            f"    xmin=3.5, xmax={max(rec['sizes']) * 1.2:g},",
            "    ymajorgrids=true,",
            "    legend pos=outer north east,",
            "    legend cell align=left,",
            "    legend style={font=\\footnotesize},",
            "  ]"]
    looks: dict[tuple[str, str], str] = {}
    for k in rec["shown"]:
        label, colour, mark = pfchilds._style(k, registry)
        if (colour, mark) in looks:
            raise ValueError(f"{k} would look identical to {looks[(colour, mark)]}")
        looks[(colour, mark)] = k
        dash, _, _ = pfchilds.line_dashes(k, registry, rec["line_style"], 0)
        out.append(f"    % {k}: r from the bound at every n; point = mean over seeds, bars min..max over seeds")
        out.append(f"    \\addplot[{colour}, {dash}, mark={mark}, mark size=2.2pt, mark options={{solid, fill={colour}}}, "
                   f"error bars/.cd, y dir=both, y explicit, error bar style={{solid}}] coordinates {{")
        for p in rec["series"][k]["measured"]:
            out.append(f"      ({p['n']},{p['mean']:.6g}) += (0,{p['max'] - p['mean']:.6g}) -= (0,{p['mean'] - p['min']:.6g})")
        out += ["    };", f"    \\addlegendentry{{{label}}}"]
    out += bound_key(rec)
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def _replace_key(text: str, rec: dict) -> str:
    """figleader writes pfchilds's measured/extrapolated key; swap in the bound key."""
    old = "\n".join(pfchilds.legend_key(rec["line_style"], "measured $r$",
                                        f"extrapolated (fit on {figleader.figratio._fit_note(rec)})"))
    if text.count(old) != 1:
        raise ValueError("figleader's key not found exactly once; cannot relabel")
    return text.replace(old, "\n".join(bound_key(rec)))


def write_figures(cfg: dict, here: Path, rec: dict, mcfg: dict, render: bool) -> list[Path]:
    registry = formula_registry()
    out: list[Path] = []
    rec_path = here / f"{rec['id']}.json"
    rec_text = json.dumps(rec, indent=1)
    rec_path.write_text(rec_text)
    tex_path = here / f"{rec['id']}.tex"
    tex_text = figure_tex(rec, mcfg, registry)
    tex_path.write_text(tex_text)
    out += [rec_path, tex_path]

    # figleader's helpers need a fit window for a key we then replace; any pair of sizes will do
    shim = {**rec, "fit_sizes": [min(rec["sizes"]), max(rec["sizes"])]}
    tex_sha = hashlib.sha256(tex_text.encode()).hexdigest()
    rec_sha = hashlib.sha256(rec_text.encode()).hexdigest()
    z = mcfg["zoom"]["panel"]
    zoom = figleader.side_by_side_zoom_tex(tex_text, shim, mcfg["zoom"], figleader.header_lines(
        rec, tex_path.name, tex_sha, f"bound curves (no error bars) with n = {z['xmin']:g}..{z['xmax']:g} enlarged",
        [f"r from {bound_label(cfg['bound'])}", "left axis: coordinates copied unchanged from the source .tex",
         "right axis: the same coordinates and error bars, clipped to the window"]), registry)
    zoom = _replace_key(zoom, shim)
    n = mcfg["n"]
    rows = figleader.leaderboard(rec, {}, n)
    dots = figleader.dots_tex(shim, rows, n, mcfg["dots"], figleader.header_lines(
        rec, rec_path.name, rec_sha, f"cost at n = {n}, one row per formula",
        [f"n = {n}; r from {bound_label(cfg['bound'])}; x = mean over seeds, bars min..max over seeds",
         "cheapest on = seeds on which the formula is cheapest; rank range = min..max of its rank over seeds"]),
        registry)
    for stem, text in ((f"{rec['id']}-with-zoom", zoom), (f"{rec['id']}-n{n}-dots", dots)):
        p = here / f"{stem}.tex"
        p.write_text(text)
        out.append(p)
    (here / f"{rec['id']}-leader.json").write_text(json.dumps({"source": rec_path.name, "source_sha256": rec_sha,
                                                               "n": n, "rows": rows}, indent=1))
    for p in out:
        print(f"  wrote {p}")
    if render:
        for p in out[1:]:
            pdf, _ = pfchilds.render_tex(p, here / "rendered")
            print(f"  rendered {pdf}")
    return out


def _header(cfg: dict, mcfg: dict, what: str, kept: list[str], removed: list[str], extra: list[str]) -> list[str]:
    prov = provenance()
    h = ["Generated by pysimcount.pfboundchilds -- do not edit; regenerate instead.",
         f"figure: {mcfg['id']} -- {mcfg['title']}",
         f"r from {bound_label(cfg['bound'])}; bound settings {json.dumps(cfg['bound'])}",
         f"NumPy seeds {cfg['seeds']}; t = n; sizes {cfg['sizes']}; {what}",
         *extra,
         f"shown: {', '.join(kept)}",
         f"removed: {', '.join(removed) or 'none'}",
         f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
    if prov["git_dirty"]:
        h.append("WARNING: made from uncommitted code; the commit above does not reproduce it.")
    return h


def run_cnot(cfg: dict, here: Path, render: bool) -> dict:
    mcfg = cfg["cnot"]
    series, n_checked = cnot_series(cfg, here)
    plotted = {k: {p["n"]: p["mean"] for p in s["measured"]} for k, s in series.items()}
    kept, ranking = pfchilds.prune(plotted, cfg["remove"], cfg["protect_top"])
    removed = [k for k in cfg["schemes"] if k not in kept]
    header = _header(cfg, mcfg, f"cz: eps_trotter = {GATEBASES['cz'][0]:g}, merged, simcount template", kept, removed,
                     [f"CNOT = exact affine per-step count x r; affine model checked against {n_checked} run rows (n = 4-11)",
                      f"pruning: up to {cfg['remove']} removed, never one of the {cfg['protect_top']} cheapest at any n"])
    rec = make_record(cfg, mcfg, "cnot", "Clifford+Rz (cz)", series, kept, ranking, header)
    write_figures(cfg, here, rec, mcfg, render)
    print(f"  shown {kept}; removed {removed}; r varies over seeds: {rec['r_varies_over_seeds']}")
    return rec


def run_t(cfg: dict, here: Path, render: bool) -> dict:
    mcfg = cfg["t"]
    like = json.loads((here / f"{cfg['cnot']['id']}.json").read_text())
    kept = like["shown"]
    series = t_series(cfg, here, kept)
    header = _header(cfg, mcfg, f"ct: eps_trotter = {GATEBASES['ct'][0]:g}, eps_synth = {GATEBASES['ct'][1]:g}, "
                     f"merged, {mcfg['template']} template, {mcfg['backend']}", kept, like["removed"],
                     ["one step built and synthesised per formula, seed and n, eps_rot = eps_synth / (rotations x r)",
                      f"formulas as in {cfg['cnot']['id']}.json (pruning not re-run)"])
    rec = make_record(cfg, {**mcfg, "removed": like["removed"]}, "t", "Clifford+T (ct)", series, kept, None, header)
    write_figures(cfg, here, rec, mcfg, render)
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["cnot", "tcounts", "t"])
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    if args.stage == "cnot":
        run_cnot(cfg, here, args.render)
    elif args.stage == "tcounts":
        run_tcounts(cfg, here)
    else:
        run_t(cfg, here, args.render)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
    main()
