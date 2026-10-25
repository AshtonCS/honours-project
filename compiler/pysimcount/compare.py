"""Side-by-side comparison of the PennyLane pipeline against simcount."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from . import optimise as _optimise
from . import resources, simcount, synthesis
from .circuit import Simulation, simulate
from .hamiltonian import Hamiltonian

# simcount's process_options in Main.hs: (eps_trotter, eps_synth)
GATEBASES = {
    "cz": (0.001, None),
    "cz2ct": (0.0005, None),
    "ct": (0.0005, 0.0005),
}


@dataclass
class PennyLaneResult:
    sim: Simulation
    eps_synth: float | None
    eps_rot: float | None
    step_counts: dict[str, int]  # Clifford+T if synthesised, else Clifford+Rz
    timings: dict[str, float]
    backend: str = "pygridsynth"

    @property
    def aggregated(self) -> dict[str, int]:
        return resources.scale(self.step_counts, self.sim.r)


def run_pennylane(H: Hamiltonian, algo: str, gatebase: str, t: float | None = None,
                  r: int | None = None, zrots: int | None = None, template: str = "simcount",
                  up_to_phase: bool = False, backend: str = "pygridsynth") -> PennyLaneResult:
    eps_trotter, eps_synth = GATEBASES[gatebase]
    t = float(H.n_qubits) if t is None else t  # Main.hs: t = size
    timings = {}

    t0 = time.perf_counter()
    sim = simulate(H, algo, t, eps_trotter, r=r, template=template)
    timings["build step"] = time.perf_counter() - t0

    eps_rot = None
    step = sim.step
    if eps_synth is not None:
        t0 = time.perf_counter()
        step, eps_rot = synthesis.synthesise_step(sim, eps_synth, zrots, up_to_phase, backend)
        timings["synthesis"] = time.perf_counter() - t0

    return PennyLaneResult(sim, eps_synth, eps_rot, resources.count(step), timings, backend)


# ------------------------------------------------------------------ tables

def _match_labels(a: dict[str, int], b: dict[str, int], rtol=1e-9) -> list[tuple[str, str | None]]:
    """Pair up labels across the two dictionaries, matching rotation labels
    by numeric value rather than by string."""
    pairs, used = [], set()
    for la in sorted(a, key=resources._sort_key):
        va = resources.rot_value(la)
        match = None
        if va is None:
            match = la if la in b else None
        else:
            for lb in b:
                vb = resources.rot_value(lb)
                if vb is not None and lb not in used and math.isclose(va, vb, rel_tol=rtol, abs_tol=0):
                    match = lb
                    break
        if match is not None:
            used.add(match)
        pairs.append((la, match))
    for lb in sorted(b, key=resources._sort_key):
        if lb not in used:
            pairs.append((None, lb))
    return pairs


def _fmt(v):
    return "-" if v is None else f"{v:,}"


def table(rows: list[tuple[str, int | None, int | None]], left="simcount", right="pennylane",
          marks: bool = True) -> str:
    w = max([len(r[0]) for r in rows] + [10])
    out = [f"{'':<{w}}  {left:>14}  {right:>14}  {'Δ':>10}  {'%':>7}"]
    for name, a, b in rows:
        if a is None or b is None:
            delta, pct, ok = "", "", "?"
        else:
            delta = f"{b - a:+,}"
            pct = f"{100 * (b - a) / a:+.1f}%" if a else ""
            ok = "✓" if a == b else "✗"
        out.append(f"{name:<{w}}  {_fmt(a):>14}  {_fmt(b):>14}  {delta:>10}  {pct:>7}  {ok if marks else ''}")
    return "\n".join(out)


def gate_table(a: dict[str, int], b: dict[str, int], **kw) -> str:
    rows = []
    for la, lb in _match_labels(a, b):
        name = la if la is not None else lb
        rows.append((name, a.get(la) if la else None, b.get(lb) if lb else None))
    rows.append(("Total gates", resources.total(a), resources.total(b)))
    return table(rows, **kw)


def summary_table(a: dict[str, int], b: dict[str, int], **kw) -> str:
    sa, sb = resources.summary(a), resources.summary(b)
    return table([(k, sa[k], sb[k]) for k in sa], **kw)


def report(ref: simcount.SimcountResult, pl: PennyLaneResult, optimised: dict[str, int] | None = None,
           k: int | None = None) -> str:
    sim = pl.sim
    eps_trotter, eps_synth = GATEBASES[ref.gatebase]
    # In ct mode simcount's output has no Rz left; the count it budgeted over is -z.
    ref_zrots = ref.zrots if eps_synth else resources.rotations(ref.aggregated)
    lines = [
        f"=== n={ref.n}  {ref.algo}  -g {ref.gatebase}  seed={ref.seed}"
        f"  (t={sim.t:g}, eps_trotter={eps_trotter:g}"
        + (f", eps_synth={eps_synth:g}" if eps_synth else "") + ") ===",
        f"Hamiltonian: {sim.H.n_terms} terms from simcount's seed {ref.seed}, "
        f"second-order split = {sim.H.split}, template = {sim.template}"
        + (f", synthesis = {pl.backend}" if pl.eps_rot is not None else ""),
        "",
        table([
            ("r (Trotter steps)", ref.r, sim.r),
            ("gates / step", resources.total(ref.per_step), resources.total(pl.step_counts)),
            ("Rz / step (before synthesis)", ref_zrots // ref.r if ref_zrots else None, sim.n_rz_per_step),
            ("zrots (Rz x r)", ref_zrots, sim.zrots),
        ]),
    ]
    if pl.eps_rot is not None:
        lines += [f"per-rotation synthesis epsilon: {pl.eps_rot:.4e}"
                  f"  (simcount: -z {ref.zrots}  ->  {eps_synth / ref.zrots:.4e})" if ref.zrots else ""]
    lines += ["", "--- per step ---", gate_table(ref.per_step, pl.step_counts),
              "", "--- aggregated (step x r) ---", gate_table(ref.aggregated, pl.aggregated),
              "", "--- summary ---", summary_table(ref.aggregated, pl.aggregated)]
    if optimised is not None:
        # Optimisation acts on the Clifford+Rz step, so compare against the
        # unsynthesised baseline even in ct mode.
        baseline = resources.scale(resources.count(sim.step), sim.r)
        lines += ["", f"--- pennylane optimised (blocks of {k} step{'s' if k != 1 else ''}) vs Clifford+Rz baseline ---",
                  summary_table(baseline, optimised, left="baseline", right=f"optimised k={k}", marks=False)]
    lines += ["", "timings: " + ", ".join(f"{n} {s:.2f}s" for n, s in pl.timings.items())]
    return "\n".join(lines)


def compare(n: int, algo: str, gatebase: str = "cz", seed: int = 1, zrots: int | None = None,
            template: str = "simcount", optimise_k: int | None = None, use_cache: bool = True,
            up_to_phase: bool = False, backend: str = "pygridsynth",
            ) -> tuple[simcount.SimcountResult, PennyLaneResult, str]:
    """Run both implementations on simcount's Hamiltonian and render a report.

    For ``-g ct`` simcount needs the number of rotations up front (``-z``);
    when not given it is taken from the PennyLane cz2ct count, as the README
    instructs.
    """
    if gatebase == "ct" and zrots is None:
        pre = simcount.run(n, algo, "cz2ct", seed, use_cache=use_cache)
        zrots = resources.rotations(pre.aggregated)
    ref = simcount.run(n, algo, gatebase, seed, zrots, use_cache=use_cache)
    pl = run_pennylane(ref.hamiltonian, algo, gatebase, zrots=zrots, template=template,
                       up_to_phase=up_to_phase, backend=backend)
    opt = _optimise.optimise(pl.sim, optimise_k) if optimise_k else None
    return ref, pl, report(ref, pl, opt, optimise_k)
