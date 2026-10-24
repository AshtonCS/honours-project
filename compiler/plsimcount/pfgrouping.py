"""Grouping comparison for Suzuki formulas on simcount's Heisenberg instances.

    python -m plsimcount.pfgrouping results/pf-grouping/config.json [--render]

Four ways of turning the same Hamiltonian and the same Suzuki recursion into
a circuit (cz gate set, measured r, t = n, eps_trotter = 1e-3, even n only):

(a) ``xyz``      simcount's term order, XX | YY | ZZ+Z, every term its own
                 Pauli exponential (the pf-thesis baseline);
(b) ``evenodd``  the same terms reordered even bonds | odd bonds | fields
                 (``evenodd.even_odd``), still term by term;
(c) ``blocked``  the (b) ordering with each bond's XX, YY, ZZ compiled as one
                 3-CNOT block (``blocks.build_step_blocked``). Same unitary as
                 (b), so r(c) = r(b) by construction (exactness is tested in
                 ``tests/test_blocks.py``); only gates per step change;
(d) ``twogroup`` two groups, even bonds with the fields folded in (E') and
                 odd bonds (O), every bond one exact two-qubit block
                 (``twogroup.build_step``). Run for both choices of the outer
                 group (``outer`` = even or odd); a different formula from (b),
                 so a different r.

How r is found: the smallest r with ||S(t/r)^r - e^{-iHt}||_2 <= eps, by the
same search as pfcompare (``fastempirical.steps``). For (a)/(b) the step is
built term by term by ``fastempirical.FastEvolution``; for (d) each group
exponential is exact, built from the group's eigendecomposition
(``GroupEvolution``), because the terms inside an E' block do not commute.

What is counted: one step's tape with simcount's gate template (cz, no
synthesis), *unmerged* for all four, so no variant gets a merge the others
do not (blocks.py's reasoning). (a) and (b) are also reported merged
(``weighted.merge``), the convention of the pf-thesis figures. Totals are
per-step x r.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import blocks, evenodd, fastempirical, formula, resources, twogroup
from .compare import GATEBASES
from .hamiltonian import Hamiltonian
from .pfcompare import fixture_hamiltonian, formula_registry, instance_fields, instance_hamiltonian, provenance, \
    write_environment
from .weighted import build_step_from_schedule, merge

VARIANTS = ("xyz", "evenodd", "blocked", "twogroup-even-out", "twogroup-odd-out")


class GroupEvolution:
    """Exact grouped product-formula steps, for ``fastempirical.steps``.

    A schedule here is a list of (group index, time) as built by
    ``twogroup.group_schedule``; group g is exponentiated exactly as
    V_g diag(exp(-i time E_g)) V_g^dagger from H_g = V_g diag(E_g) V_g^dagger.
    Same quantity as ``twogroup.steps_empirical`` (dense expm), faster.
    """

    def __init__(self, H: Hamiltonian, outer: str):
        self.H = H
        self.outer = outer
        self._eig = []
        for M in twogroup._group_matrices(H, outer):
            E, V = np.linalg.eigh(M)
            self._eig.append((E, V))
        E, V = np.linalg.eigh(H.matrix())
        self._E, self._V = E, V

    def exact(self, t: float) -> np.ndarray:
        return (self._V * np.exp(-1j * self._E * t)) @ self._V.conj().T

    def step(self, sched: list[tuple[int, float]]) -> np.ndarray:
        """Unitary of one step; the first entry acts first."""
        N = 2 ** self.H.n_qubits
        U = np.eye(N, dtype=complex)
        for g, tau in sched:
            E, V = self._eig[g]
            U = (V * np.exp(-1j * tau * E)) @ (V.conj().T @ U)
        return U

    def error(self, schedule_fn, t: float, r: int) -> float:
        S = self.step(schedule_fn(t / r))
        return float(np.linalg.norm(np.linalg.matrix_power(S, r) - self.exact(t), 2))


def measured_r(H_xyz: Hamiltonian, variant: str, order: int, t: float, eps: float, start: int = 1):
    """fastempirical.SearchResult for one variant. 'blocked' is not searched
    (it is the 'evenodd' unitary); asking for it raises."""
    if variant == "xyz":
        evol = fastempirical.FastEvolution(H_xyz)
        return fastempirical.steps(evol, lambda H, d: formula.schedule(order, H, d), order, t, eps, start=start)
    if variant == "evenodd":
        H_eo = evenodd.even_odd(H_xyz)
        evol = fastempirical.FastEvolution(H_eo)
        return fastempirical.steps(evol, lambda H, d: formula.schedule(order, H, d), order, t, eps, start=start)
    if variant.startswith("twogroup-"):
        outer = variant.split("-")[1]
        evol = GroupEvolution(H_xyz, outer)
        return fastempirical.steps(evol, lambda H, d: twogroup.group_schedule(order, d), order, t, eps, start=start)
    raise ValueError(f"no r search for variant {variant!r}")


def step_tape(H_xyz: Hamiltonian, variant: str, order: int, d: float, merged: bool):
    """One step's tape (cz, simcount template). ``merged`` only for xyz/evenodd."""
    if merged and variant not in ("xyz", "evenodd"):
        raise ValueError(f"merging is only defined for the term-by-term variants, not {variant!r}")
    if variant in ("xyz", "evenodd"):
        H = H_xyz if variant == "xyz" else evenodd.even_odd(H_xyz)
        sched = formula.schedule(order, H, d)
        if merged:
            sched = merge(H, sched)
        return build_step_from_schedule(H, sched, "simcount")
    if variant == "blocked":
        return blocks.build_step_blocked(evenodd.even_odd(H_xyz), order, d)[0]
    if variant.startswith("twogroup-"):
        return twogroup.build_step(H_xyz, order, d, variant.split("-")[1])
    raise ValueError(f"unknown variant {variant!r}")


FIELDS = ["n", "t", "seed", "order", "variant", "eps_trotter", "r", "r_from", "err_at_r", "err_at_r_minus_1",
          "search_seconds", "merged", "cnot_per_step", "rz_per_step", "gates_per_step",
          "cnot_total", "rz_total", "gates_total"]


def run(cfg: dict, log=print) -> list[dict]:
    if cfg.get("gatebase", "cz") != "cz":
        raise ValueError("only the cz gate set: block rotations are not routed through synthesis")
    eps = GATEBASES["cz"][0]
    rows: list[dict] = []
    for order in cfg["orders"]:
        hints: dict[str, int] = {}
        for n in cfg["sizes"]:
            H = fixture_hamiltonian(n, cfg["seed"])
            t = float(n)
            r_of: dict[str, tuple[int, object]] = {}
            for v in VARIANTS:
                if v == "blocked":
                    r_of[v] = (r_of["evenodd"][0], r_of["evenodd"][1])
                    src = "evenodd"
                else:
                    s = measured_r(H, v, order, t, eps, start=hints.get(v, 1))
                    hints[v] = s.r
                    r_of[v] = (s.r, s)
                    src = "measured"
                r, s = r_of[v]
                for merged in ((False, True) if v in ("xyz", "evenodd") else (False,)):
                    c = resources.summary(resources.count(step_tape(H, v, order, t / r, merged)))
                    rows.append({
                        "n": n, "t": t, "seed": cfg["seed"], "order": order, "variant": v, "eps_trotter": eps,
                        "r": r, "r_from": src, "err_at_r": s.err_r, "err_at_r_minus_1": s.err_r_minus_1,
                        "search_seconds": round(s.seconds, 2), "merged": merged,
                        "cnot_per_step": c["CNOT"], "rz_per_step": c["Rz"], "gates_per_step": c["Total excl. omega"],
                        "cnot_total": c["CNOT"] * r, "rz_total": c["Rz"] * r, "gates_total": c["Total excl. omega"] * r,
                    })
                log(f"  n={n:2d} S{order} {v:18s} r={r:>6,}  CNOT/step={rows[-1]['cnot_per_step']:>6,}  "
                    f"CNOT total={rows[-1]['cnot_total']:>10,}")
    return rows


# ------------------------------------------------------------------ any formula (config key "schemes")

SCHEME_VARIANTS = ("xyz", "twogroup-even-out", "twogroup-odd-out")

FIELDS_SCHEMES = ["instance", "seed", "n", "t", "scheme", "family", "order", "variant", "eps_trotter", "r",
                  "err_at_r", "err_at_r_minus_1", "search_dips", "search_seconds", "merged",
                  "cnot_per_step", "rz_per_step", "gates_per_step", "cnot_total", "rz_total", "gates_total"]


def run_schemes(cfg: dict, meta: dict, log=print) -> list[dict]:
    """X-Y-Z vs two groups for any registry formula (``twogroup.formula_group_schedule``).

    r is measured per (formula, variant, n) with the same search as pfcompare.
    Counts are reported unmerged and merged for every variant: merged means
    ``weighted.merge`` for X-Y-Z (the pf-thesis convention) and, for two
    groups, consecutive applications of one group combined into one (exact).
    Instances as pfcompare (``instances``: simcount fixtures or NumPy fields),
    fields recorded in ``meta["fields"]``.
    """
    if cfg.get("gatebase", "cz") != "cz":
        raise ValueError("only the cz gate set: block rotations are not routed through synthesis")
    variants = cfg["variants"]
    bad = [v for v in variants if v not in SCHEME_VARIANTS]
    if bad:
        raise ValueError(f"unknown variants {bad}; known: {SCHEME_VARIANTS}")
    registry = formula_registry()
    unknown = [k for k in cfg["schemes"] if k not in registry]
    if unknown:
        raise ValueError(f"unknown scheme ids {unknown}")
    eps = GATEBASES["cz"][0]
    inst = cfg.get("instances", "simcount")
    rows: list[dict] = []
    hints: dict[tuple[str, str], int] = {}
    for n in cfg["sizes"]:
        H = instance_hamiltonian(cfg, n)
        meta["fields"][str(n)] = instance_fields(H)
        t = float(n)
        evols = {}
        for v in variants:
            evols[v] = fastempirical.FastEvolution(H) if v == "xyz" else GroupEvolution(H, v.split("-")[1])
        for key in cfg["schemes"]:
            spec = registry[key]
            for v in variants:
                if v == "xyz":
                    fn = spec.schedule
                else:
                    fn = lambda H_, d, spec=spec: twogroup.formula_group_schedule(spec, d)  # noqa: E731
                s = fastempirical.steps(evols[v], fn, spec.order, t, eps, start=hints.get((key, v), 1))
                hints[(key, v)] = r = s.r
                for merged in (False, True):
                    if v == "xyz":
                        sched = spec.schedule(H, t / r)
                        tape = build_step_from_schedule(H, merge(H, sched) if merged else sched, "simcount")
                    else:
                        tape = twogroup.build_step_schedule(
                            H, twogroup.formula_group_schedule(spec, t / r, merged), v.split("-")[1])
                    c = resources.summary(resources.count(tape))
                    rows.append({
                        "instance": inst, "seed": cfg["seed"], "n": n, "t": t, "scheme": key, "family": spec.family,
                        "order": spec.order, "variant": v, "eps_trotter": eps, "r": r,
                        "err_at_r": s.err_r, "err_at_r_minus_1": s.err_r_minus_1, "search_dips": s.dips,
                        "search_seconds": round(s.seconds, 2), "merged": merged,
                        "cnot_per_step": c["CNOT"], "rz_per_step": c["Rz"], "gates_per_step": c["Total excl. omega"],
                        "cnot_total": c["CNOT"] * r, "rz_total": c["Rz"] * r, "gates_total": c["Total excl. omega"] * r,
                    })
                log(f"  n={n:2d} {spec.name:30s} {v:18s} r={r:>8,}  merged CNOT total={rows[-1]['cnot_total']:>12,}")
    return rows


def table_md_schemes(rows: list[dict]) -> str:
    lines = ["| scheme | n | variant | r | CNOT/step | CNOT total | merged CNOT/step | merged CNOT total | merged gates total |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    by = {(r["scheme"], r["n"], r["variant"], r["merged"]): r for r in rows}
    for key in dict.fromkeys(r["scheme"] for r in rows):
        for n in sorted({r["n"] for r in rows}):
            for v in SCHEME_VARIANTS:
                x, m = by.get((key, n, v, False)), by.get((key, n, v, True))
                if x is None:
                    continue
                lines.append(f"| {key} | {n} | {v} | {x['r']:,} | {x['cnot_per_step']:,} | {x['cnot_total']:,} | "
                             f"{m['cnot_per_step']:,} | {m['cnot_total']:,} | {m['gates_total']:,} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ figure

STYLE = {  # variant -> (label, colour, mark)
    "xyz": ("(a) X-Y-Z, term by term", "red", "*"),
    "evenodd": ("(b) even/odd, term by term", "blue", "square*"),
    "blocked": ("(c) even/odd, 3-CNOT bond blocks", "green!50!black", "triangle*"),
    "twogroup-even-out": ("(d) two groups, E$'$ outer", "orange", "diamond*"),
    "twogroup-odd-out": ("(d) two groups, O outer", "violet", "pentagon*"),
}


def figure_tex(rows: list[dict], order: int, column: str, ylabel: str, header: list[str]) -> str:
    pts = [r for r in rows if r["order"] == order and not r["merged"]]
    sizes = sorted({r["n"] for r in pts})
    ticks = ",".join(map(str, sizes))
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm,", "    xlabel={System size $n$},",
        f"    ylabel={{{ylabel}}},", f"    xtick={{{ticks}}},", f"    xticklabels={{{ticks}}},",
        "    ymajorgrids=true,", "    legend pos=outer north east,", "    legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for v in VARIANTS:
        label, colour, mark = STYLE[v]
        ser = sorted((r["n"], r[column]) for r in pts if r["variant"] == v)
        if not ser:
            continue
        out.append(f"    \\addplot[{colour}, mark={mark}, mark size=2.4pt, mark options={{solid, fill={colour}}}] "
                   "coordinates {")
        out += [f"      ({n},{y:.0f})" for n, y in ser]
        out += ["    };", f"    \\addlegendentry{{{label}}}"]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def write_figures(rows: list[dict], meta: dict, root: Path) -> Path:
    from .pffigure import PREAMBLE
    root.mkdir(parents=True, exist_ok=True)
    pages = []
    prov = meta["provenance"]
    for order in sorted({r["order"] for r in rows}):
        for column, ylabel, tag in (("cnot_total", "CNOT count", "cnot"), ("gates_total", "Total gate count (Clifford+$R_z$)", "total")):
            header = ["Generated by plsimcount.pfgrouping -- do not edit; regenerate instead.",
                      f"Suzuki S{order}, {tag}, unmerged, cz, seed {meta['config']['seed']}, t = n, measured r",
                      f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
            rel = f"{tag}-S{order}-by-grouping.tex"
            (root / rel).write_text(figure_tex(rows, order, column, ylabel, header))
            pages.append(f"\\section*{{\\texttt{{{rel}}}}}\nSuzuki $S_{order}$: {tag} count by grouping "
                         f"(unmerged, measured $r$)\\par\\medskip\n\\input{{{rel}}}\n\\clearpage\n")
    (root / "all.tex").write_text(PREAMBLE + "\n".join(pages) + "\\end{document}\n")
    return root


def table_md(rows: list[dict]) -> str:
    lines = ["| order | n | variant | r | CNOT/step | Rz/step | gates/step | CNOT total | gates total | "
             "merged CNOT/step | merged gates total |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by = {(r["order"], r["n"], r["variant"], r["merged"]): r for r in rows}
    for order in sorted({r["order"] for r in rows}):
        for n in sorted({r["n"] for r in rows}):
            for v in VARIANTS:
                x = by.get((order, n, v, False))
                if x is None:
                    continue
                m = by.get((order, n, v, True))
                mc = f"{m['cnot_per_step']:,}" if m else "--"
                mg = f"{m['gates_total']:,}" if m else "--"
                lines.append(f"| {order} | {n} | {v} | {x['r']:,} | {x['cnot_per_step']:,} | {x['rz_per_step']:,} | "
                             f"{x['gates_per_step']:,} | {x['cnot_total']:,} | {x['gates_total']:,} | {mc} | {mg} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out = args.config.parent
    meta = {"config": cfg, "provenance": provenance(), "fields": {}}
    write_environment(out)
    t0 = time.perf_counter()
    if "schemes" in cfg:
        rows = run_schemes(cfg, meta)
        meta["seconds"] = round(time.perf_counter() - t0, 1)
        (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        with open(out / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS_SCHEMES)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        (out / "table.md").write_text(table_md_schemes(rows) + "\n")
        print(table_md_schemes(rows))
        return
    rows = run(cfg)
    meta["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "table.md").write_text(table_md(rows) + "\n")
    root = write_figures(rows, meta, out / "figures")
    print(table_md(rows))
    if args.render:
        from .pffigure import render
        print(f"rendered {render(root)}")


if __name__ == "__main__":
    main()
