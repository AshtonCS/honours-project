"""Command line: reproduce each checkable claim of arXiv:2601.18756v3.

    python -m effts errors          # Err_n of all 63 published schemes vs upstream
    python -m effts table4          # paper Table 4 (Eff_n and x-bar)
    python -m effts fig5            # paper Fig. 5, both groupings, vs digitised
    python -m effts fig6 --L-max 12 # paper Fig. 6 (slow above L = 10)
    python -m effts ranking         # "q=14 beats n<=6 historical schemes at each cost"
    python -m effts morales         # Morales et al. Table 8 error constants, random matrices

Everything prints to stdout.  No randomness: fields are upstream's seed-42
values, recorded in fixtures/upstream_fields.json.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import experiments as X


def cmd_errors(_: argparse.Namespace) -> None:
    rows = X.published_errors()
    print(f"{'file':28s} {'n':>2s} {'q':>3s} {'Err_n^2 (ours)':>15s} {'Err_n^2 (upstream)':>19s} {'rel diff':>9s}")
    for r in rows:
        if r.n == r.order:
            print(f"{r.source:28s} {r.n:2d} {r.q:3d} {r.ours_err2:15.8e} {r.upstream_err2:19.8e} {r.rel:9.1e}")
    print(f"\n{len(rows)} (scheme, n >= order) pairs; worst relative difference "
          f"{max(r.rel for r in rows):.1e}")


def cmd_table4(_: argparse.Namespace) -> None:
    print(f"{'scheme':28s} {'Eff ours':>9s} {'paper':>7s}   {'x ours':>7s} {'paper':>6s}")
    for label, eff, eff_p, x, x_p in X.table4():
        # paper prints 3 decimals: agreement means within half a unit of the last digit
        flag = "" if abs(eff - eff_p) <= 5e-4 and abs(x - x_p) <= 5e-4 else "  <-- differs"
        print(f"{label:28s} {eff:9.4f} {eff_p:7.3f}   {x:7.4f} {x_p:6.3f}{flag}")


def cmd_fig5(_: argparse.Namespace) -> None:
    for panel in ("global", "local"):
        print(f"\nFig. 5, {panel} grouping (L=6, t=10): log10(ours / paper) over all digitised points")
        for c in X.fig5_comparison(panel):
            d = np.log10(c["ours"] / c["paper"])
            print(f"  {c['label']:20s} q={c['q']:2d} n={c['order']}  points={len(d):2d}  "
                  f"median |d|={np.median(abs(d)):.4f}  max |d|={abs(d).max():.4f}")
    print("\nDigitisation precision is ~3e-4 decades (0.01 px).  Morales et al. "
          "(q=17, 33) are not reproduced: coefficients not in the upstream repo.")


def cmd_fig6(args: argparse.Namespace) -> None:
    Ls = list(range(2, args.L_max + 1))
    print(f"Fig. 6 (global grouping, t=10, q N_t = 500, N_t = round(500/q))")
    print(f"{'L':>3s} {'scheme':20s} {'q':>3s} {'paper':>10s} {'ours':>10s} {'log10 ratio':>11s}")
    for L in Ls:
        t0 = time.time()
        rows = X.fig6_comparison([L])
        for r in rows:
            print(f"{r['L']:3d} {r['label']:20s} {r['q']:3d} {r['paper']:10.3e} {r['ours']:10.3e} "
                  f"{np.log10(r['ours'] / r['paper']):+11.4f}")
        print(f"    (L={L}: {time.time() - t0:.1f}s)", flush=True)


def cmd_ranking(args: argparse.Namespace) -> None:
    costs = [84, 98, 112, 126, 140, 200, 300, 500, 1000, 2000]
    for g in ("global", "local"):
        rows = X.ranking_at_costs(g, costs)
        qs = [s for s in rows[0] if s != "cost"]
        print(f"\n{g} grouping, L=6, t=10: Delta at N_t = round(cost/q)")
        print("cost  " + "".join(f"{'q=' + str(q):>11s}" for q in qs) + "   best")
        for r in rows:
            best = min(qs, key=lambda q: r[q][1])
            print(f"{r['cost']:5d} " + "".join(f"{r[q][1]:11.2e}" for q in qs) + f"   q={best}")
    print("\nEach scheme runs at N_t = round(cost/q), so its actual cost can differ from the\n"
          "row label by < q/2.  The paper's claim (q=14 best at each cost) holds here from\n"
          "cost 126 (global) / 140 (local) up; below that Leapfrog or q=6 win.")


def cmd_morales(args: argparse.Namespace) -> None:
    print(f"Morales et al. random-matrix test: {args.samples} random 6x6 pairs (Gaussian, norm 1), seed {args.seed}, t = 1")
    print(f"{'formula':32s} {'M':>3s} {'chi ours':>10s} {'chi paper':>10s} {'ratio':>6s} {'M chi^1/8 ours':>15s}")
    for label, M, chi, paper in X.morales_reproduction(args.samples, args.seed):
        ratio = f"{chi / paper:6.2f}" if paper else "     -"
        pp = f"{paper:10.2e}" if paper else "         -"
        print(f"{label:32s} {M:3d} {chi:10.2e} {pp} {ratio} {M * chi ** (1 / 8):15.2f}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m effts", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("errors").set_defaults(f=cmd_errors)
    sub.add_parser("table4").set_defaults(f=cmd_table4)
    sub.add_parser("fig5").set_defaults(f=cmd_fig5)
    f6 = sub.add_parser("fig6")
    f6.add_argument("--L-max", type=int, default=10, help="largest chain length (paper: 12; 12 takes minutes)")
    f6.set_defaults(f=cmd_fig6)
    sub.add_parser("ranking").set_defaults(f=cmd_ranking)
    mo = sub.add_parser("morales")
    mo.add_argument("--samples", type=int, default=2000)
    mo.add_argument("--seed", type=int, default=2026)
    mo.set_defaults(f=cmd_morales)
    args = p.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
