"""Accuracy of the compiled circuits against exactly computed unitaries.

    python -m pysimcount.pfexact ../results/np-exact/config.json

No simcount is involved: every reference here is computed classically and
exactly, so this checks pysimcount against the physics rather than against
the reference implementation. Two parts, plus negative controls.

Part A -- commuting Hamiltonian, closed-form answer
    H = Jz sum_<i,j> Z_i Z_j + sum_i h_i Z_i  (Jx = Jy = 0; the XX and YY terms
    are kept with coefficient 0, so the circuit has the same structure as in
    the main runs and their basis changes must cancel exactly).
    Every term is diagonal, so all terms commute and every product formula is
    exact at any r. The reference is written down without expm or
    diagonalisation: |z> picks up the phase exp(-i t E(z)), with
    E(z) = Jz sum_<i,j> z_i z_j + sum_i h_i z_i and z_i = +1 for |0>, -1 for |1>.
    Measured: || (G(t/r))^r - U_exact ||_2, G = gate-level step unitary.
    Pass: error <= tol_per_exponential x (exponentials applied), i.e. no more
    than double-precision rounding accumulated over the circuit.
    Catches: R_z sign / factor-of-2 errors, qubit-index errors (the h_i are
    non-uniform), basis changes that fail to cancel, stage weights that do not
    sum to one step per term. Blind to: term order and higher-order
    coefficient errors (Lie-Trotter passes as well as S8).

Part B -- full Heisenberg chain, exact diagonalisation
    The simcount instance (J = 1, periodic for n >= 3) with NumPy fields.
    Reference U = V exp(-i t Lambda) V^dagger from numpy.linalg.eigh.
    (i) Compilation adds no error: || G(d) - F(d) ||_2 per step, where F is the
        product of the exact term exponentials exp(-i c theta P) =
        cos(c theta) I - i sin(c theta) P (exact because P^2 = I).
        Pass: <= tol_per_exponential x exponentials per step.
    (ii) Order: the error ||G^r - U|| behaves as C r^(-p) once r is large
        enough, the leading-order error scaling of an order-p formula. The
        slope is the least-squares slope of log error against log r, using
        only points with error inside ``slope_window`` (above it: not yet
        asymptotic; below it: near the rounding floor). Reported as -slope.
        Pass: at least ``min_points`` points in the window and
        |(-slope) - p| <= slope_tol.

Negative controls (n = smallest Part A size)
    A1: Suzuki S2 with every exponential's time scaled by (1 + perturb) -- the
        weights no longer sum to one step, so Part A must fail.
    B1: Lie-Trotter (order 1) checked as if it were order 2 -- the slope check
        must fail.
    A control that passes means the criterion cannot discriminate, and the run
    raises rather than report the main results as a demonstration.

The pass criteria (tolerances, window, min_points) are set in the config
before the run and recorded with the results; they are not tuned afterwards.

Part C -- local (single-step) order, a config with key ``local``
    Added 2026-11-11 after the first np-exact run, whose Part B (ii) failed
    for S8 (n = 4, 6) and M&O q=14 (n = 6). Diagnosis: over r steps the
    gate-level rounding grows as ~2e-16 x r x exponentials and reached the
    window for S8, and for small r the global error of high-order formulas is
    not yet a straight line on a log-log plot. Part C avoids both: it measures
    one step, e(d) = || G(d) - exp(-i d H) ||_2, whose leading behaviour for an
    order-p formula is C d^(p+1) as d -> 0 (the local error; the global error
    t/d steps of it gives the r^-p of Part B).
    Floor: f = tol_per_exponential x exponentials per step (Part A's rounding
    tolerance). A point is usable if e(d) >= floor_margin x f.
    Fit: least squares of log e on log d over the ``n_fit`` smallest usable d
    (smallest d = most asymptotic data available). Fitted order = slope - 1.
    Pass: n_fit usable points exist and |slope - (p+1)| <= slope_tol.
    Controls (must fail): C1 Lie-Trotter claimed as order 2; C2 Suzuki S4 with
    every time scaled by (1 + perturb), whose local error becomes ~ perturb x d
    at small d, i.e. local order 1.
    Runs instead of Parts A and B (``run_local``); a config without ``local``
    runs exactly as before.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from .hamiltonian import Hamiltonian, heisenberg
from .pfcompare import formula_registry, numpy_fields, provenance, write_environment
from .weighted import FormulaSpec, build_step_from_schedule

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ------------------------------------------------------------------ references

def chain_bonds(n: int) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of simcount's chain: periodic for n >= 3, open
    for n = 2 (``Definitions.hs``, as ``hamiltonian.heisenberg``). Written out
    again here so that Part A's reference does not read the Hamiltonian object
    it is checking."""
    if n < 2:
        raise ValueError(f"a chain needs at least 2 sites, got {n}")
    bonds = [(i, i + 1) for i in range(n - 1)]
    if n >= 3:
        bonds.append((0, n - 1))
    return bonds


def zz_hamiltonian(n: int, fields: list[float], Jz: float = 1.0) -> Hamiltonian:
    """Part A Hamiltonian: simcount's term list with Jx = Jy = 0.

    Coefficients are dimensionless with hbar = 1, H = Jz sum ZZ + sum h_i Z_i
    (no factor 1/2, as in simcount). The 2n zero-coefficient XX/YY terms are
    kept, so each compiles to a basis change around R_z(0)."""
    return heisenberg(n, fields, coupling=(0.0, 0.0, Jz))


def diagonal_reference(n: int, fields: list[float], t: float, Jz: float = 1.0) -> np.ndarray:
    """exp(-i t H) for the Part A Hamiltonian, as the vector of its diagonal.

    Basis order follows PennyLane's ``wire_order=range(n)``: wire 0 is the
    most significant bit, so the k-th entry is the bit string of k read left
    to right as wires 0..n-1. Bit b on wire q gives z_q = 1 - 2b."""
    if len(fields) != n:
        raise ValueError(f"expected {n} field values, got {len(fields)}")
    bonds = chain_bonds(n)
    phases = []
    for bits in itertools.product((0, 1), repeat=n):
        z = [1 - 2 * b for b in bits]
        energy = 0.0
        for i, j in bonds:
            energy += Jz * z[i] * z[j]
        for q in range(n):
            energy += fields[q] * z[q]
        phases.append(np.exp(-1j * t * energy))
    return np.array(phases)


def exact_unitary(H: Hamiltonian, t: float) -> np.ndarray:
    """exp(-i t H) by exact diagonalisation (H Hermitian, hbar = 1)."""
    lam, V = np.linalg.eigh(H.matrix())
    return V @ np.diag(np.exp(-1j * t * lam)) @ V.conj().T


def pauli_matrix(word: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for p in word:
        m = np.kron(m, _PAULI[p])
    return m


def pauli_exponential(coeff: float, word: str, time: float) -> np.ndarray:
    """exp(-i time coeff P) = cos(time coeff) I - i sin(time coeff) P, exact for
    a Pauli word P (P^2 = I)."""
    theta = time * coeff
    P = pauli_matrix(word)
    return np.cos(theta) * np.eye(P.shape[0]) - 1j * np.sin(theta) * P


# ------------------------------------------------------------------ step unitaries

def circuit_step(H: Hamiltonian, schedule: list[tuple[int, float]]) -> np.ndarray:
    """Gate-level unitary of one step: simcount's gate template, multiplied out
    by PennyLane. This is the circuit whose gates the main runs count."""
    tape = build_step_from_schedule(H, schedule)
    return qml.matrix(tape, wire_order=range(H.n_qubits))


def formula_step(H: Hamiltonian, schedule: list[tuple[int, float]]) -> np.ndarray:
    """Product of the exact term exponentials, in schedule order (first applied
    on the right)."""
    U = np.eye(2**H.n_qubits, dtype=complex)
    for j, time in schedule:
        term = H.terms[j]
        U = pauli_exponential(term.coeff, term.word, time) @ U
    return U


def spectral_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, 2))


# ------------------------------------------------------------------ Part A

def commuting_row(n: int, fields: list[float], t: float, r: int, key: str, spec: FormulaSpec,
                  tol_per_exp: float, Jz: float = 1.0, time_scale: float = 1.0) -> dict:
    """One Part A measurement. ``time_scale`` != 1 is only for the negative
    control: it multiplies every exponential's time."""
    if r < 1:
        raise ValueError(f"r must be a positive integer, got {r}")
    H = zz_hamiltonian(n, fields, Jz)
    d = t / r
    schedule = [(j, time_scale * x) for j, x in spec.schedule(H, d)]
    G = circuit_step(H, schedule)
    U_ref = np.diag(diagonal_reference(n, fields, t, Jz))
    err = spectral_norm(np.linalg.matrix_power(G, r) - U_ref)
    # first-order consistency: each term's times over one step should sum to d
    weight_sum = np.zeros(H.n_terms)
    for j, x in schedule:
        weight_sum[j] += x
    exps_total = r * len(schedule)
    tol = tol_per_exp * exps_total
    return {"n": n, "t": t, "r": r, "scheme": key, "name": spec.name, "order": spec.order,
            "exps_per_step": len(schedule), "exps_total": exps_total,
            "error": err, "error_per_exp": err / exps_total,
            "max_weight_sum_dev": float(np.max(np.abs(weight_sum - d))),
            "tol": tol, "pass": bool(err <= tol)}


# ------------------------------------------------------------------ Part B

def order_slope(rs: list[int], errors: list[float], window: tuple[float, float]) -> tuple[float | None, list[int]]:
    """-(least-squares slope of log error against log r) over the points with
    window[0] <= error <= window[1]; returns (slope or None, r values used)."""
    lo, hi = window
    used_r, used_e = [], []
    for r, e in zip(rs, errors):
        if lo <= e <= hi:
            used_r.append(r)
            used_e.append(e)
    if len(used_r) < 2:
        return None, used_r
    b, _ = np.polyfit(np.log(used_r), np.log(used_e), 1)
    return float(-b), used_r


def noncommuting_points(H: Hamiltonian, t: float, key: str, spec: FormulaSpec, r_grid: list[int],
                        tol_per_exp: float) -> list[dict]:
    U_ref = exact_unitary(H, t)
    out = []
    for r in r_grid:
        schedule = spec.schedule(H, t / r)
        G = circuit_step(H, schedule)
        F = formula_step(H, schedule)
        step_diff = spectral_norm(G - F)
        tol = tol_per_exp * len(schedule)
        out.append({"n": H.n_qubits, "t": t, "r": r, "scheme": key, "name": spec.name, "order": spec.order,
                    "exps_per_step": len(schedule),
                    "error_gate": spectral_norm(np.linalg.matrix_power(G, r) - U_ref),
                    "error_formula": spectral_norm(np.linalg.matrix_power(F, r) - U_ref),
                    "step_diff": step_diff, "step_tol": tol, "step_pass": bool(step_diff <= tol)})
    return out


def order_fit(points: list[dict], claimed_order: int, window: tuple[float, float], slope_tol: float,
              min_points: int) -> dict:
    """Slope check for one (n, scheme) from its points, against ``claimed_order``
    (the formula's order, except in negative control B1)."""
    pts = sorted(points, key=lambda x: x["r"])
    rs = [x["r"] for x in pts]
    errs = [x["error_gate"] for x in pts]
    slope, used = order_slope(rs, errs, window)
    pair = []
    for i in range(len(pts) - 1):
        pair.append(float(np.log(errs[i] / errs[i + 1]) / np.log(rs[i + 1] / rs[i])))
    enough = len(used) >= min_points
    ok = enough and slope is not None and abs(slope - claimed_order) <= slope_tol
    reason = "" if ok else ("fewer than %d points in window" % min_points if not enough
                            else f"|slope - p| = {abs(slope - claimed_order):.2f} > {slope_tol}")
    return {"n": pts[0]["n"], "scheme": pts[0]["scheme"], "name": pts[0]["name"], "order": claimed_order,
            "r_grid": rs, "errors": errs, "pair_slopes": pair, "r_used": used, "slope": slope,
            "compile_pass": all(x["step_pass"] for x in pts),
            "max_step_diff": max(x["step_diff"] for x in pts),
            "order_pass": bool(ok), "reason": reason}


# ------------------------------------------------------------------ Part C

def local_points(H: Hamiltonian, key: str, spec: FormulaSpec, d_grid: list[float], tol_per_exp: float,
                 time_scale: float = 1.0) -> list[dict]:
    """Single-step errors || G(d) - exp(-i d H) || for each d. ``time_scale``
    != 1 only for control C2."""
    lam, V = np.linalg.eigh(H.matrix())
    out = []
    for d in d_grid:
        if not d > 0:
            raise ValueError(f"step sizes must be positive, got {d}")
        schedule = [(j, time_scale * x) for j, x in spec.schedule(H, d)]
        G = circuit_step(H, schedule)
        U = V @ np.diag(np.exp(-1j * d * lam)) @ V.conj().T
        out.append({"n": H.n_qubits, "d": d, "scheme": key, "name": spec.name, "order": spec.order,
                    "exps_per_step": len(schedule), "error": spectral_norm(G - U),
                    "floor": tol_per_exp * len(schedule)})
    return out


def local_fit(points: list[dict], claimed_order: int, floor_margin: float, n_fit: int,
              slope_tol: float) -> dict:
    pts = sorted(points, key=lambda x: x["d"])  # ascending d
    usable = [x for x in pts if x["error"] >= floor_margin * x["floor"]]
    used = usable[:n_fit]
    slope = None
    if len(used) >= 2:
        b, _ = np.polyfit(np.log([x["d"] for x in used]), np.log([x["error"] for x in used]), 1)
        slope = float(b)
    want = claimed_order + 1
    ok = len(used) >= n_fit and slope is not None and abs(slope - want) <= slope_tol
    if ok:
        reason = ""
    elif len(used) < n_fit:
        reason = f"only {len(used)} usable points (need {n_fit})"
    else:
        reason = f"|slope - (p+1)| = {abs(slope - want):.2f} > {slope_tol}"
    return {"n": pts[0]["n"], "scheme": pts[0]["scheme"], "name": pts[0]["name"], "order": claimed_order,
            "exps_per_step": pts[0]["exps_per_step"], "d_used": [x["d"] for x in used],
            "errors_used": [x["error"] for x in used], "slope": slope,
            "fitted_order": None if slope is None else slope - 1.0, "pass": bool(ok), "reason": reason}


def tables_local_md(res: dict) -> str:
    cfg = res["meta"]["config"]
    c = cfg["local"]
    out = [f"# {cfg['name']}: local (single-step) order against exact diagonalisation", "",
           cfg["description"], "",
           f"Criteria (fixed before the run): usable if error >= {c['floor_margin']:g} x "
           f"{cfg['criteria']['tol_per_exponential']:g} x exponentials/step; fit the {c['n_fit']} smallest "
           f"usable d; pass if |slope - (p+1)| <= {c['slope_tol']}.", "",
           "| formula | p | n | exps/step | d used | error range | slope (expect p+1) | fitted order | result |",
           "|---|---:|---:|---:|---|---|---:|---:|---|"]
    for k in cfg["schemes"]:
        for f in sorted((x for x in res["fits"] if x["scheme"] == k), key=lambda x: x["n"]):
            ds = ", ".join(f"{d:.3g}" for d in f["d_used"]) or "--"
            er = (f"{min(f['errors_used']):.1e}–{max(f['errors_used']):.1e}" if f["errors_used"] else "--")
            s = "--" if f["slope"] is None else f"{f['slope']:.2f}"
            o = "--" if f["fitted_order"] is None else f"{f['fitted_order']:.2f}"
            out.append(f"| {f['name']} | {f['order']} | {f['n']} | {f['exps_per_step']:,} | {ds} | {er} | {s} | "
                       f"{o} | {'PASS' if f['pass'] else 'FAIL: ' + f['reason']} |")
    out += ["", "## Negative controls (must FAIL)", "", "| control | slope | expected slope | result |",
            "|---|---:|---:|---|"]
    for label, f in res["controls"].items():
        s = "--" if f["slope"] is None else f"{f['slope']:.2f}"
        out.append(f"| {label} | {s} | {f['order'] + 1} | {'PASS (bad)' if f['pass'] else 'FAIL (expected)'} |")
    out += ["", f"Wall clock: {res['meta']['seconds']:.0f} s.", ""]
    return "\n".join(out)


def run_local(cfg: dict, out_dir: Path | None = None, verbose: bool = True) -> dict:
    registry = formula_registry()
    for key in ("name", "description", "seed", "schemes", "local", "criteria"):
        if key not in cfg:
            raise ValueError(f"config is missing {key!r}")
    unknown = [k for k in cfg["schemes"] if k not in registry]
    if unknown:
        raise ValueError(f"unknown scheme ids {unknown}; known: {sorted(registry)}")
    c = cfg["local"]
    for n in c["sizes"]:
        if not 2 <= n <= 8:
            raise ValueError(f"sizes must be in 2..8 for dense unitaries, got {n}")
    tol_exp = float(cfg["criteria"]["tol_per_exponential"])
    margin, n_fit, slope_tol = float(c["floor_margin"]), int(c["n_fit"]), float(c["slope_tol"])
    d_grid = [float(d) for d in c["d_grid"]]
    seed = int(cfg["seed"])
    t_start = time.perf_counter()
    meta = {"config": cfg, "provenance": provenance(), "fields": {}}
    if out_dir is not None:
        write_environment(out_dir)
    points, fits = [], []
    for n in c["sizes"]:
        f = numpy_fields(n, seed)
        meta["fields"][str(n)] = f
        H = heisenberg(n, f)
        for k in cfg["schemes"]:
            spec = registry[k]
            pts = local_points(H, k, spec, d_grid, tol_exp)
            points += pts
            fit = local_fit(pts, spec.order, margin, n_fit, slope_tol)
            fits.append(fit)
            if verbose:
                s = "--" if fit["slope"] is None else f"{fit['slope']:.2f}"
                print(f"C n={n} {k:14s} p={spec.order} slope={s} (expect {spec.order + 1}) "
                      f"{'PASS' if fit['pass'] else 'FAIL ' + fit['reason']}", flush=True)
    n0 = min(c["sizes"])
    H0 = heisenberg(n0, numpy_fields(n0, seed))
    c1 = local_fit(local_points(H0, "lie-trotter", registry["lie-trotter"], d_grid, tol_exp),
                   2, margin, n_fit, slope_tol)
    c2 = local_fit(local_points(H0, "suzuki4", registry["suzuki4"], d_grid, tol_exp,
                                time_scale=1.0 + float(c["perturb"])), 4, margin, n_fit, slope_tol)
    controls = {f"C1: Lie-Trotter claimed order 2, n={n0}": c1,
                f"C2: S4, times x (1 + {float(c['perturb']):g}), n={n0}": c2}
    meta["seconds"] = round(time.perf_counter() - t_start, 1)
    res = {"meta": meta, "points": points, "fits": fits, "controls": controls}
    if out_dir is not None:
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
        (out_dir / "table.md").write_text(tables_local_md(res))
    if c1["pass"] or c2["pass"]:
        raise RuntimeError("a negative control passed: the local-order criterion cannot discriminate "
                           "(results written, but flagged)")
    return res


# ------------------------------------------------------------------ tables

def _e(x: float | None) -> str:
    return "--" if x is None else f"{x:.1e}"


def tables_md(res: dict) -> str:
    cfg = res["meta"]["config"]
    crit = cfg["criteria"]
    A, fits, ctrl = res["commuting"], res["noncommuting"]["fits"], res["controls"]
    keys = cfg["schemes"]
    sizes_a = cfg["commuting"]["sizes"]
    out = [f"# {cfg['name']}: accuracy against exact unitaries", "",
           cfg["description"], "",
           f"Instance: NumPy fields, seed {cfg['seed']}, t = n. Criteria (fixed before the run): "
           f"tol = {crit['tol_per_exponential']:g} per exponential; slope window "
           f"[{crit['slope_window'][0]:g}, {crit['slope_window'][1]:g}], |slope - p| <= {crit['slope_tol']}, "
           f">= {crit['min_points']} points.", ""]
    for r in cfg["commuting"]["r"]:
        out += [f"## Part A: commuting ZZ + Z, error vs closed form, r = {r}", "",
                "| formula | exps/step (n=%d) | " % sizes_a[-1]
                + " | ".join(f"error n={n}" for n in sizes_a) + f" | error/exp (n={sizes_a[-1]}) | pass |",
                "|---|---:|" + "---:|" * len(sizes_a) + "---:|---|"]
        for k in keys:
            rows = {x["n"]: x for x in A if x["scheme"] == k and x["r"] == r}
            last = rows[sizes_a[-1]]
            ok = all(rows[n]["pass"] for n in sizes_a)
            out.append(f"| {last['name']} | {last['exps_per_step']:,} | "
                       + " | ".join(_e(rows[n]["error"]) for n in sizes_a)
                       + f" | {last['error_per_exp']:.1e} | {'PASS' if ok else 'FAIL'} |")
        out.append("")
    out += ["## Part B: full Heisenberg chain vs exact diagonalisation", "",
            "| formula | p | n | max ‖G−F‖/step | compile | r in window | fitted order | order |",
            "|---|---:|---:|---:|---|---|---:|---|"]
    for k in keys:
        for f in sorted((x for x in fits if x["scheme"] == k), key=lambda x: x["n"]):
            s = "--" if f["slope"] is None else f"{f['slope']:.2f}"
            out.append(f"| {f['name']} | {f['order']} | {f['n']} | {f['max_step_diff']:.1e} | "
                       f"{'PASS' if f['compile_pass'] else 'FAIL'} | {', '.join(map(str, f['r_used']))} | {s} | "
                       f"{'PASS' if f['order_pass'] else 'FAIL: ' + f['reason']} |")
    out += ["", "## Negative controls (must FAIL)", "",
            "| control | measured | criterion | result |", "|---|---|---|---|"]
    a1 = ctrl["A1"]
    out.append(f"| A1: S2, times x (1 + {cfg['controls']['perturb']:g}), n={a1['n']}, r={a1['r']} | "
               f"error {a1['error']:.1e} | <= {a1['tol']:.1e} | {'PASS (bad)' if a1['pass'] else 'FAIL (expected)'} |")
    b1 = ctrl["B1"]
    s = "--" if b1["slope"] is None else f"{b1['slope']:.2f}"
    out.append(f"| B1: Lie-Trotter claimed order 2, n={b1['n']} | fitted order {s} | |slope − 2| <= "
               f"{crit['slope_tol']} | {'PASS (bad)' if b1['order_pass'] else 'FAIL (expected)'} |")
    out += ["", f"Wall clock: {res['meta']['seconds']:.0f} s.", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ run

def _check_config(cfg: dict, registry: dict) -> None:
    for key in ("name", "description", "seed", "schemes", "commuting", "noncommuting", "criteria", "controls"):
        if key not in cfg:
            raise ValueError(f"config is missing {key!r}")
    unknown = [k for k in cfg["schemes"] if k not in registry]
    if unknown:
        raise ValueError(f"unknown scheme ids {unknown}; known: {sorted(registry)}")
    for n in cfg["commuting"]["sizes"] + cfg["noncommuting"]["sizes"]:
        if not 2 <= n <= 8:
            # dense 2^n x 2^n gate-level matrices; n = 8 already takes minutes for S8
            raise ValueError(f"sizes must be in 2..8 for dense unitaries, got {n}")
    grids = cfg["noncommuting"]["r_grid"]
    for k in cfg["schemes"]:
        p = str(registry[k].order)
        if p not in grids and "default" not in grids:
            raise ValueError(f"no r grid for order {p} ({k}) and no 'default'")


def r_grid_for(cfg: dict, order: int) -> list[int]:
    grids = cfg["noncommuting"]["r_grid"]
    return list(grids.get(str(order), grids.get("default")))


def run(cfg: dict, out_dir: Path | None = None, verbose: bool = True) -> dict:
    registry = formula_registry()
    _check_config(cfg, registry)
    crit = cfg["criteria"]
    tol_exp = float(crit["tol_per_exponential"])
    window = (float(crit["slope_window"][0]), float(crit["slope_window"][1]))
    keys = cfg["schemes"]
    seed = int(cfg["seed"])
    t_start = time.perf_counter()
    meta = {"config": cfg, "provenance": provenance(), "fields": {}}
    if out_dir is not None:
        write_environment(out_dir)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def fields_of(n: int) -> list[float]:
        f = numpy_fields(n, seed)
        meta["fields"][str(n)] = f
        return f

    # Part A
    A = []
    Jz = float(cfg["commuting"].get("Jz", 1.0))
    for n in cfg["commuting"]["sizes"]:
        f = fields_of(n)
        for r in cfg["commuting"]["r"]:
            for k in keys:
                row = commuting_row(n, f, float(n), r, k, registry[k], tol_exp, Jz)
                A.append(row)
                log(f"A n={n} r={r} {k:14s} err={row['error']:.2e} {'PASS' if row['pass'] else 'FAIL'}")

    # Part B
    points, fits = [], []
    for n in cfg["noncommuting"]["sizes"]:
        H = heisenberg(n, fields_of(n))
        for k in keys:
            spec = registry[k]
            pts = noncommuting_points(H, float(n), k, spec, r_grid_for(cfg, spec.order), tol_exp)
            points += pts
            fit = order_fit(pts, spec.order, window, float(crit["slope_tol"]), int(crit["min_points"]))
            fits.append(fit)
            s = "--" if fit["slope"] is None else f"{fit['slope']:.2f}"
            log(f"B n={n} {k:14s} p={spec.order} fitted={s} on r={fit['r_used']} "
                f"compile={'PASS' if fit['compile_pass'] else 'FAIL'} order={'PASS' if fit['order_pass'] else 'FAIL'}")

    # negative controls, at the smallest Part A size
    n0 = min(cfg["commuting"]["sizes"])
    c = cfg["controls"]
    a1 = commuting_row(n0, fields_of(n0), float(n0), int(c["r"]), "suzuki2", registry["suzuki2"], tol_exp, Jz,
                       time_scale=1.0 + float(c["perturb"]))
    H0 = heisenberg(n0, fields_of(n0))
    lt = registry["lie-trotter"]
    b1_pts = noncommuting_points(H0, float(n0), "lie-trotter", lt, r_grid_for(cfg, 1), tol_exp)
    b1 = order_fit(b1_pts, 2, window, float(crit["slope_tol"]), int(crit["min_points"]))
    controls = {"A1": a1, "B1": b1}
    log(f"control A1 {'PASS (bad)' if a1['pass'] else 'FAIL (expected)'}; "
        f"control B1 {'PASS (bad)' if b1['order_pass'] else 'FAIL (expected)'}")

    meta["seconds"] = round(time.perf_counter() - t_start, 1)
    res = {"meta": meta, "commuting": A, "noncommuting": {"points": points, "fits": fits}, "controls": controls}
    if out_dir is not None:
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
        (out_dir / "table.md").write_text(tables_md(res))
    if a1["pass"] or b1["order_pass"]:
        raise RuntimeError("a negative control passed: the criteria cannot discriminate, so the "
                           "results above do not demonstrate accuracy (written, but flagged)")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if "local" in cfg:
        print(tables_local_md(run_local(cfg, out_dir=args.config.parent)))
        return
    res = run(cfg, out_dir=args.config.parent)
    print(tables_md(res))


if __name__ == "__main__":
    main()
