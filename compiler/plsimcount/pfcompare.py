"""Suzuki vs Maležič–Ostmeyer product formulas on simcount's Heisenberg instances.

    python -m plsimcount.pfcompare ../results/pf-comparison/config.json

Everything the run depends on is in the config file; the results land next
to it (``results.csv``, ``results.json``) together with the git commit, a
dirty flag and a hash of every source file used, so each number can be
traced back to the code and settings that produced it. Nothing is random:
Hamiltonians come from the committed simcount fixtures (seed in the config).

How each row is made
--------------------
1. r: the smallest Trotter number whose *measured* spectral error at time t
   is <= eps_trotter (``fastempirical.steps``). The same search for every
   formula. It is exact for this instance and not a bound.
2. One step is compiled with simcount's gate template, unmerged and merged
   (``weighted.merge``); totals are per-step counts x r, as in simcount.
3. ct: every rotation is synthesised to eps_rot = eps_synth / (rotations
   per step x r), as ``synthesis.synthesise_step`` does. Counts are
   accumulated per distinct angle instead of materialising the whole
   Clifford+T circuit (identical result, checked in the tests).

Rows with ``r_source == "fit"`` are extrapolations: r = ceil(a n^b) fitted to
the measured sizes. They are labelled as such and are not measurements.

Rows with ``r_source`` in ``ana | min | com | emp | tte`` (config section
``bound_rows``) take r from that Trotter bound (``bounds.steps``) instead of
measuring it, everything else being identical. The bounds are Suzuki-specific
(simcount's ``PF.hs``; tte = Childs et al. 2021, Props. 15, 16, J.1/J.2), so
only ``family == "suzuki"`` formulas are accepted, and a (bound, order) pair
the bound does not define (``BOUND_ORDERS``) is skipped and recorded in
``meta["skipped"]``, never silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import bounds, fastempirical, formula, resources, synthesis
from .compare import GATEBASES
from .hamiltonian import Hamiltonian, heisenberg, parse_simcount
from .simcount import FIXTURES
from .weighted import FormulaSpec, build_step_from_schedule, merge, suzuki_formula, weighted_formula

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ formulas

def _effts():
    from effts import schemes  # sibling package in compiler/
    return schemes


def formula_registry() -> dict[str, FormulaSpec]:
    """Config ids -> formulas. Scheme data come from compiler/effts (validated
    against the paper and the upstream repository)."""
    S = _effts()
    from effts import morales as M
    return {
        # simcount's pf1: every term once per step, in Hamiltonian order.
        "lie-trotter": FormulaSpec("Lie-Trotter S1", "suzuki", 1, 1,
                                   lambda H, d: formula.first_order(H, d)),
        "suzuki2": suzuki_formula(2),
        "suzuki4": suzuki_formula(4),
        "suzuki6": suzuki_formula(6),
        "suzuki8": suzuki_formula(8),
        "mo-q6": weighted_formula(S.recommended_q6(), "M&O q=6 (rec., Tab. 2)", "paper"),
        "mo-q14": weighted_formula(S.recommended_q14(), "M&O q=14 (rec., Tab. 3)", "paper"),
        "bm-q10": weighted_formula(S.blanes_moan(10), "Blanes & Moan q=10", "historical"),
        "mo-o8-q21": weighted_formula(S.recommended(8, 21)[0], "M&O q=21 order 8 (exploratory)", "paper"),
        # Morales et al. (arXiv:2210.15817) unprocessed order-8 chains of Leapfrog steps,
        # weights from effts/morales.py (extracted from the arXiv LaTeX source).
        "morales-y8m10": weighted_formula(M.y8m10(), "Morales Y8m10 (21 steps)", "historical"),
        "morales-m8": weighted_formula(M.m8(), "Morales m=8 (17 steps)", "historical"),
    }


# Orders for which each bound is defined (bounds.py): com and tte stop at 4,
# emp is a fit for orders 1-8, ana/min take any order simcount supports.
BOUND_ORDERS: dict[str, tuple[int, ...]] = {
    "ana": (1, 2, 4, 6, 8),
    "min": (1, 2, 4, 6, 8),
    "com": (1, 2, 4),
    "emp": (1, 2, 4, 6, 8),
    "tte": (1, 2, 4),
}


# ------------------------------------------------------------------ instances

def fixture_hamiltonian(n: int, seed: int) -> Hamiltonian:
    """simcount's Heisenberg instance for (n, seed), read from any committed fixture.

    Raises if no fixture exists (generating one needs Docker: run
    ``python -m plsimcount compare -n N -a pf2emp -g cz -m SEED`` once).
    """
    files = sorted(FIXTURES.glob(f"n{n}_*_m{seed}.txt"))
    if not files:
        raise FileNotFoundError(f"no simcount fixture for n={n}, seed={seed} in {FIXTURES}")
    Hs = {tuple((t.coeff, t.word) for t in parse_simcount(f.read_text()).terms) for f in files}
    if len(Hs) != 1:
        raise ValueError(f"fixtures for n={n}, seed={seed} disagree on the Hamiltonian")
    return parse_simcount(files[0].read_text())


def paper_hamiltonian(n: int) -> Hamiltonian:
    """The paper's XXZ instance (J = 1, upstream seed-42 fields in [-0.1, 0.1])
    in simcount's term layout. Same model; only the fields differ."""
    from effts.experiments import paper_fields
    return heisenberg(n, paper_fields(n))


# ------------------------------------------------------------------ counting

def _word_counts(word: str) -> Counter:
    return Counter(resources.label(op) for op in synthesis.word_to_ops(word, 0))


def synthesised_counts(counts: dict[str, int], eps_rot: float, backend: str) -> dict[str, int]:
    """Clifford+T counts of a Clifford+Rz step, rotation by distinct angle.

    Equals ``resources.count(synthesis.synthesise_tape(tape, eps_rot))``
    without building the Clifford+T tape.
    """
    out: Counter = Counter()
    for lbl, c in counts.items():
        v = resources.rot_value(lbl)
        if v is None:
            out[lbl] += c
            continue
        word = synthesis.synthesise(2 * v, eps_rot, False, backend)  # label holds theta/2
        for k, m in _word_counts(word).items():
            out[k] += m * c
    return dict(out)


def step_counts(H: Hamiltonian, spec: FormulaSpec, t: float, r: int, merged: bool,
                template: str) -> dict[str, int]:
    sched = spec.schedule(H, t / r)
    if merged:
        sched = merge(H, sched)
    return resources.count(build_step_from_schedule(H, sched, template))


# ------------------------------------------------------------------ provenance

def provenance() -> dict:
    root = HERE.parent
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            return f"unavailable: {e}"
    files = sorted(list((root / "plsimcount").glob("*.py")) + list((root / "effts").glob("*.py"))
                   + list((root / "effts" / "data").rglob("*.out")))
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "plsimcount", "effts")),
        "source_sha256": h.hexdigest(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


# ------------------------------------------------------------------ run

FIELDS = ["instance", "n", "t", "scheme", "family", "order", "q", "gatebase", "eps_trotter", "eps_synth",
          "r_source", "r", "err_at_r", "err_at_r_minus_1", "search_evals", "search_seconds", "emp_reference_r",
          "merged", "rz_per_step", "cnot_per_step", "t_per_step", "gates_per_step", "eps_rot",
          "rz_total", "cnot_total", "t_total", "gates_total"]


def _row(instance, n, t, key, spec, gatebase, r_source, r, search, merged, counts, eps_rot):
    eps_trotter, eps_synth = GATEBASES[gatebase]
    s = resources.summary(counts)
    emp = None
    if spec.family == "suzuki" and r_source == "measured":
        # simcount's fitted emp r, for reference only: it ignores t and eps.
        emp = bounds.empirical(spec.order, _dummy(n), t, eps_trotter)
    return {
        "instance": instance, "n": n, "t": t, "scheme": key, "family": spec.family, "order": spec.order, "q": spec.q,
        "gatebase": gatebase, "eps_trotter": eps_trotter, "eps_synth": eps_synth, "r_source": r_source, "r": r,
        "err_at_r": search.err_r if search else None,
        "err_at_r_minus_1": search.err_r_minus_1 if search else None,
        "search_evals": search.evaluations if search else None,
        "search_seconds": round(search.seconds, 2) if search else None,
        "emp_reference_r": emp, "merged": merged,
        "rz_per_step": s["Rz"], "cnot_per_step": s["CNOT"], "t_per_step": s["T"], "gates_per_step": s["Total excl. omega"],
        "eps_rot": eps_rot,
        "rz_total": s["Rz"] * r, "cnot_total": s["CNOT"] * r, "t_total": s["T"] * r, "gates_total": s["Total excl. omega"] * r,
    }


class _dummy:
    """bounds.empirical only reads n_qubits (it is a fitted a*n^b in n)."""
    def __init__(self, n):
        self.n_qubits = n


def run_instance(rows, instance, H, n, t, keys, registry, gatebases, cfg, r_source="measured",
                 r_given=None, hints=None, log=print):
    template, backend = cfg.get("template", "simcount"), cfg.get("synth_backend", "pygridsynth")
    evol = fastempirical.FastEvolution(H) if r_source == "measured" else None
    for key in keys:
        spec = registry[key]
        for gb in gatebases:
            eps_trotter, eps_synth = GATEBASES[gb]
            search = None
            if r_source == "measured":
                start = (hints or {}).get((key, gb), 1)
                search = fastempirical.steps(evol, spec.schedule, spec.order, t, eps_trotter, start=start)
                r = search.r
                if hints is not None:
                    hints[(key, gb)] = r
            else:
                r = r_given[(key, gb)]
            for merged in (False, True):
                counts = step_counts(H, spec, t, r, merged, template)
                eps_rot = None
                if eps_synth is not None:
                    eps_rot = eps_synth / (resources.rotations(counts) * r)
                    counts = synthesised_counts(counts, eps_rot, backend)
                rows.append(_row(instance, n, t, key, spec, gb, r_source, r, search, merged, counts, eps_rot))
            last = rows[-1]
            log(f"  {instance:8s} n={n:2d} {gb:3s} {spec.name:32s} r={r:>8,}"
                + (f" ({search.evaluations} evals, {search.seconds:6.1f}s)" if search else f" ({r_source})")
                + f"  merged gates={last['gates_total']:.3e}" + (f"  T={last['t_total']:.3e}" if eps_synth else ""))


def fit_power_law(ns, rs):
    """Least squares on log r = log a + b log n."""
    b, loga = np.polyfit(np.log(ns), np.log(rs), 1)
    return math.exp(loga), b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--only", nargs="*", help="restrict to these scheme ids (for quick checks; still recorded)")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out_dir = args.config.parent
    registry = formula_registry()
    keys = args.only or cfg["schemes"]
    unknown = [k for k in keys if k not in registry]
    if unknown:
        raise ValueError(f"unknown scheme ids {unknown}; known: {sorted(registry)}")

    meta = {"config": cfg, "only": args.only, "provenance": provenance()}
    rows: list[dict] = []
    hints: dict = {}
    t_start = time.perf_counter()

    # 1. simcount instances, measured r
    print(f"== simcount instances (seed {cfg['seed']}), t = n ==")
    for n in cfg["sizes"]:
        H = fixture_hamiltonian(n, cfg["seed"])
        run_instance(rows, "simcount", H, n, float(n), keys, registry, cfg["gatebases"], cfg, hints=hints)
        _write(out_dir, rows, meta)

    # 2. the paper's instance, measured r
    p = cfg.get("paper_instance")
    if p:
        print(f"== paper instance (fields +-0.1, seed 42), n = {p['n']}, t = {p['t']} ==")
        run_instance(rows, "paper", paper_hamiltonian(p["n"]), p["n"], float(p["t"]), keys, registry,
                     p["gatebases"], cfg)
        _write(out_dir, rows, meta)

    # 3. extrapolation: r from a power-law fit, gate counts on the real larger instances
    x = cfg.get("extrapolate")
    if x:
        print(f"== extrapolation: r = ceil(a n^b) fitted on n in {x['fit_sizes']} ==")
        fits, r_given = {}, {}
        for key in keys:
            for gb in cfg["gatebases"]:
                pts = sorted({(rw["n"], rw["r"]) for rw in rows if rw["instance"] == "simcount"
                              and rw["scheme"] == key and rw["gatebase"] == gb and rw["n"] in x["fit_sizes"]})
                if len(pts) < 2:
                    raise ValueError(f"need at least two measured sizes to fit {key}/{gb}")
                a, b = fit_power_law([q for q, _ in pts], [r for _, r in pts])
                fits[f"{key}/{gb}"] = {"a": a, "b": b, "points": pts}
        meta["fits"] = fits
        for n in x["sizes"]:
            H = fixture_hamiltonian(n, cfg["seed"])
            for key in keys:
                for gb in cfg["gatebases"]:
                    f = fits[f"{key}/{gb}"]
                    r_given[(key, gb)] = math.ceil(f["a"] * n ** f["b"])
            run_instance(rows, "simcount", H, n, float(n), keys, registry, cfg["gatebases"], cfg,
                         r_source="fit", r_given=r_given)
            _write(out_dir, rows, meta)

    # 4. r from a Trotter bound instead of measured (Suzuki formulas only)
    b = cfg.get("bound_rows")
    if b:
        print(f"== bound rows: {b['bounds']} for {b['schemes']} ==")
        meta["skipped"] = []
        for key in b["schemes"]:
            if registry[key].family != "suzuki":
                raise ValueError(f"bound rows need a Suzuki formula (the bounds are Suzuki-specific), got {key!r}")
        for n in b["sizes"]:
            H = fixture_hamiltonian(n, cfg["seed"])
            for bound in b["bounds"]:
                keys_b = []
                for key in b["schemes"]:
                    order = registry[key].order
                    if order in BOUND_ORDERS[bound]:
                        keys_b.append(key)
                    elif n == b["sizes"][0]:
                        meta["skipped"].append(f"{bound} not defined for order {order} ({key})")
                        print(f"  skipped: {bound} not defined for order {order} ({key})")
                r_given = {}
                for key in keys_b:
                    for gb in cfg["gatebases"]:
                        algo = f"pf{registry[key].order}{bound}"
                        r_given[(key, gb)] = bounds.steps(algo, H, float(n), GATEBASES[gb][0])
                run_instance(rows, "simcount", H, n, float(n), keys_b, registry, cfg["gatebases"], cfg,
                             r_source=bound, r_given=r_given)
                _write(out_dir, rows, meta)

    meta["seconds"] = round(time.perf_counter() - t_start, 1)
    _write(out_dir, rows, meta)
    print(f"done in {meta['seconds']:.0f}s -> {out_dir / 'results.csv'}")


def _write(out_dir: Path, rows: list[dict], meta: dict) -> None:
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
