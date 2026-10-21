"""Nam-optimised cz gate counts for a subset of pfcompare's formulas.

    python -m plsimcount.pfoptimised results/pf-optimised/config.json [--render]

For each formula and size: r is the *measured* r of an existing pfcompare
run (``source``, the pf-thesis results), the step is the merged schedule
compiled with simcount's template (cz, exactly the pf-thesis merged step),
and the step is optimised with the Nam et al. Light optimiser
(``plsimcount.nam``; see ``nam/README.md``).

Per step, three numbers are reported for CNOT, Rz and total gates:

* ``unopt``     the step as compiled (equals pf-thesis's merged per-step row);
* ``isolated``  Light on one step alone -- cannot cancel across step
                boundaries, so it overstates the optimised per-step cost;
* ``marginal``  Light on 3 consecutive steps minus Light on 2: the cost of one
                more step inside a long chain. This is the nam/README's
                "marginal per block", the number it compares with the
                published optimised core ``C`` of ``L . C^m . R``.

Totals: ``marginal x r`` (and ``unopt x r``). This drops the one-off edge
correction opt(2 steps) - 2 x marginal, which is recorded per row
(``edge_*``) so the reader can see it is small next to the totals.

Caveat (nam/README): this reconstruction recovers ~85% of the published
Light reduction on simcount's pf4 circuits (2.28x vs 2.68x). Optimised counts
here are therefore expected to sit *above* what the original optimiser would
give; they are not a reproduction of it.

Soundness: every optimised chain is checked against the unoptimised chain with
``nam.verify.equivalent`` (random states, up to global phase); for n <= 6 the
full unitaries are compared as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import resources
from .nam import Circuit, bridge, optimise, verify
from .pfcompare import fixture_hamiltonian, formula_registry, provenance, write_environment
from .weighted import build_step_from_schedule, merge

METRICS = ("CNOT", "Rz", "total")
FULL_UNITARY_MAX_N = 6


def _counts(c: Circuit) -> dict[str, int]:
    k = c.counts()
    return {"CNOT": k.get("CNOT", 0), "Rz": k.get("Rz", 0), "total": len(c)}


def _unitary_distance_up_to_phase(a: Circuit, b: Circuit) -> float:
    U, V = a.matrix(), b.matrix()
    ph = np.vdot(V.ravel(), U.ravel())
    return float(np.linalg.norm(U - (ph / abs(ph)) * V, 2))


def optimise_case(H, spec, n: int, r: int) -> dict:
    """Unoptimised, isolated and marginal counts for one (formula, n, r)."""
    sched = merge(H, spec.schedule(H, float(n) / r))
    tape = build_step_from_schedule(H, sched, "simcount")
    step = bridge.from_tape(tape, n)
    s = resources.summary(resources.count(tape))
    unopt = _counts(step)
    if (unopt["CNOT"], unopt["Rz"], unopt["total"]) != (s["CNOT"], s["Rz"], s["Total excl. omega"]):
        raise ValueError(f"nam circuit counts {unopt} disagree with the tape's {s}")
    out: dict = {"unopt": unopt}
    t0 = time.perf_counter()
    opt = {}
    for k in (1, 2, 3):
        chain = Circuit(list(step.gates) * k, n)
        res = optimise(chain)
        opt[k] = (chain, res.circuit)
        out[f"opt{k}"] = _counts(res.circuit)
    out["seconds"] = round(time.perf_counter() - t0, 2)
    out["isolated"] = out["opt1"]
    # ASSUMPTION: the optimised total is marginal x r (nam/README's per-block
    # quantity scaled by r); the constant edge term opt(2) - 2 marginal is
    # dropped from the total but recorded per row. (unconfirmed)
    out["marginal"] = {m: out["opt3"][m] - out["opt2"][m] for m in METRICS}
    out["edge"] = {m: out["opt2"][m] - 2 * out["marginal"][m] for m in METRICS}
    # soundness of every optimised chain
    worst = 0.0
    for k, (before, after) in opt.items():
        ok, err = verify.equivalent(before, after)
        if not ok:
            raise ArithmeticError(f"optimiser changed the unitary ({k} steps, err {err:.2e})")
        worst = max(worst, err)
    out["verify_state_err"] = worst
    out["verify_unitary_err"] = None
    if n <= FULL_UNITARY_MAX_N:
        d = max(_unitary_distance_up_to_phase(b, a) for b, a in opt.values())
        if d > 1e-9:
            raise ArithmeticError(f"optimiser changed the unitary (full matrix, err {d:.2e})")
        out["verify_unitary_err"] = d
    return out


FIELDS = ["scheme", "family", "order", "n", "r", "metric", "unopt_per_step", "isolated_per_step",
          "marginal_per_step", "edge", "unopt_total", "marginal_total", "reduction_factor",
          "verify_state_err", "verify_unitary_err", "seconds"]


def measured_r_from(source: dict, seed: int) -> dict[tuple[str, int], int]:
    """{(scheme, n): r} from the cz measured rows of a pfcompare results file."""
    if int(source["meta"]["config"]["seed"]) != seed:
        raise ValueError(f"source is seed {source['meta']['config']['seed']}, config asks for {seed}")
    out: dict[tuple[str, int], int] = {}
    for row in source["rows"]:
        if row["instance"] == "simcount" and row["r_source"] == "measured" and row["gatebase"] == "cz":
            out[(row["scheme"], int(row["n"]))] = int(row["r"])
    return out


def run(cfg: dict, source: dict, log=print) -> list[dict]:
    if cfg.get("gatebase", "cz") != "cz":
        raise ValueError("only cz: the optimiser works on Clifford+Rz, before any synthesis")
    reg = formula_registry()
    rs = measured_r_from(source, cfg["seed"])
    rows = []
    for key in cfg["schemes"]:
        spec = reg[key]
        for n in cfg["sizes"]:
            if (key, n) not in rs:
                raise ValueError(f"no measured cz r for {key} at n={n} in the source")
            r = rs[(key, n)]
            H = fixture_hamiltonian(n, cfg["seed"])
            c = optimise_case(H, spec, n, r)
            for m in METRICS:
                rows.append({
                    "scheme": key, "family": spec.family, "order": spec.order, "n": n, "r": r, "metric": m,
                    "unopt_per_step": c["unopt"][m], "isolated_per_step": c["isolated"][m],
                    "marginal_per_step": c["marginal"][m], "edge": c["edge"][m],
                    "unopt_total": c["unopt"][m] * r, "marginal_total": c["marginal"][m] * r,
                    "reduction_factor": c["unopt"][m] / c["marginal"][m] if c["marginal"][m] else None,
                    "verify_state_err": c["verify_state_err"], "verify_unitary_err": c["verify_unitary_err"],
                    "seconds": c["seconds"],
                })
            log(f"  {key:10s} n={n:2d} r={r:>6,}  CNOT/step {c['unopt']['CNOT']:>6,} -> isolated "
                f"{c['isolated']['CNOT']:>6,}, marginal {c['marginal']['CNOT']:>6,}  ({c['seconds']:.1f}s)")
    return rows


# ------------------------------------------------------------------ outputs

def table_md(rows: list[dict], metric: str) -> str:
    lines = [f"| scheme | n | r | {metric}/step unopt | isolated | marginal | edge | total unopt | "
             "total optimised (marginal x r) | factor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        if x["metric"] != metric:
            continue
        lines.append(f"| {x['scheme']} | {x['n']} | {x['r']:,} | {x['unopt_per_step']:,} | {x['isolated_per_step']:,} | "
                     f"{x['marginal_per_step']:,} | {x['edge']:,} | {x['unopt_total']:,} | {x['marginal_total']:,} | "
                     f"{x['reduction_factor']:.2f} |")
    return "\n".join(lines)


def figure_tex(rows: list[dict], header: list[str]) -> str:
    from .pffigure import COLOUR_OVERRIDE, FAMILY_COLOUR, ORDER_MARK, latex_escape
    reg = formula_registry()
    pts = [x for x in rows if x["metric"] == "CNOT"]
    schemes = list(dict.fromkeys(x["scheme"] for x in pts))
    sizes = sorted({x["n"] for x in pts})
    ticks = ",".join(map(str, sizes))
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm,", "    xlabel={System size $n$},",
        "    ylabel={CNOT count},", f"    xtick={{{ticks}}},", f"    xticklabels={{{ticks}}},",
        "    ymajorgrids=true,", "    legend pos=outer north east,", "    legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for s in schemes:
        f = reg[s]
        colour = COLOUR_OVERRIDE.get(s, FAMILY_COLOUR[f.family])
        mark = ORDER_MARK[f.order]
        ser = sorted((x["n"], x["marginal_total"], x["unopt_total"]) for x in pts if x["scheme"] == s)
        out.append(f"    \\addplot[{colour}, mark={mark}, mark size=2.4pt, mark options={{solid, fill={colour}}}] "
                   "coordinates {")
        out += [f"      ({n},{y:.0f})" for n, y, _ in ser]
        out += ["    };", f"    \\addlegendentry{{{latex_escape(f.name)}, optimised}}"]
        out.append(f"    \\addplot[{colour}!45, dashed, mark={mark}, mark size=1.6pt, "
                   f"mark options={{solid, fill={colour}!45}}] coordinates {{")
        out += [f"      ({n},{u:.0f})" for n, _, u in ser]
        out += ["    };", f"    \\addlegendentry{{{latex_escape(f.name)}, unoptimised}}"]
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
    src_path = (out / cfg["source"]).resolve()
    source = json.loads(src_path.read_text())
    import hashlib
    meta = {"config": cfg, "provenance": provenance(),
            "source_results_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest()}
    write_environment(out)
    t0 = time.perf_counter()
    rows = run(cfg, source)
    meta["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = ["# Nam Light optimised cz counts (merged steps, measured r from the source run)", ""]
    for m in METRICS:
        md += [f"## {m}", "", table_md(rows, m), ""]
    (out / "table.md").write_text("\n".join(md))
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    prov = meta["provenance"]
    header = ["Generated by plsimcount.pfoptimised -- do not edit; regenerate instead.",
              f"CNOT total, cz, seed {cfg['seed']}, merged, measured r; optimised = Nam Light marginal per step x r",
              f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
    (fig / "cnot-optimised.tex").write_text(figure_tex(rows, header))
    from .pffigure import PREAMBLE
    (fig / "all.tex").write_text(PREAMBLE + "\\section*{\\texttt{cnot-optimised.tex}}\n"
                                 "CNOT count, Nam Light (marginal per step $\\times r$) and unoptimised\\par\\medskip\n"
                                 "\\input{cnot-optimised.tex}\n\\end{document}\n")
    print("\n".join(md))
    if args.render:
        from .pffigure import render
        print(f"rendered {render(fig)}")


if __name__ == "__main__":
    main()
