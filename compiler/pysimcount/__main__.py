"""Command line interface mirroring simcount's ``Main`` options.

    python -m pysimcount compare -n 4 -a pf2com -g cz          # both, side by side
    python -m pysimcount run -n 4 -a pf2com -g cz --fields ... # PennyLane only
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def _common(p):
    p.add_argument("-n", "--size", type=int, default=4, help="system size (default 4)")
    p.add_argument("-a", "--algorithm", default="pf2com",
                   help="pf<order><bound>: order 1/2/4/6/8, bound ana|min|com|emp|tte "
                        "(tte and orders 6/8 are PennyLane-only; simcount has no tte)")
    p.add_argument("-g", "--gatebase", default="cz", choices=["cz", "cz2ct", "ct"])
    p.add_argument("-z", "--zrots", type=int, default=None, help="rotations to budget synthesis over (ct)")
    p.add_argument("-m", "--seed", type=int, default=1, help="simcount random seed for the Hamiltonian")
    p.add_argument("--template", default="simcount", choices=["simcount", "pennylane"],
                   help="gate template for Pauli exponentials")
    p.add_argument("--up-to-phase", action="store_true", help="synthesise rotations up to global phase")
    p.add_argument("--synth", default="pygridsynth", choices=["pygridsynth", "quipper"],
                   help="gridsynth implementation for -g ct (quipper = reference binary in the container)")
    p.add_argument("--optimise", type=int, metavar="K", default=None,
                   help="also report counts after optimising blocks of K steps")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysimcount")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="run simcount (Docker/fixture) and PennyLane side by side")
    _common(c)
    c.add_argument("--no-cache", action="store_true", help="re-run simcount even if a fixture exists")

    r = sub.add_parser("run", help="PennyLane only; print Quipper-style gate counts")
    _common(r)
    r.add_argument("--fields", type=float, nargs="+", help="Z-field coefficients (n values)")
    r.add_argument("--hamiltonian", help="simcount's printed 'Hamiltonian: [...]' line")
    r.add_argument("-t", "--time", type=float, default=None, help="simulation time (default: n)")
    r.add_argument("-r", "--steps", type=int, default=None, help="override the Trotter step count")

    b = sub.add_parser("bounds", help="compare Trotter numbers r across bounds for one instance")
    b.add_argument("-n", "--size", type=int, default=4)
    b.add_argument("-m", "--seed", type=int, default=1)
    b.add_argument("-g", "--gatebase", default="cz", choices=["cz", "cz2ct", "ct"])
    b.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4])
    b.add_argument("--empirical", action="store_true", help="also brute-force the true minimal r (n <= ~10)")

    args = ap.parse_args(argv)

    if args.cmd == "bounds":
        return _bounds_table(args)

    from . import compare, resources
    from .hamiltonian import heisenberg, parse_simcount

    if args.cmd == "compare":
        _, _, text = compare.compare(
            args.size, args.algorithm, args.gatebase, args.seed, args.zrots, args.template,
            args.optimise, use_cache=not args.no_cache, up_to_phase=args.up_to_phase,
            backend=args.synth,
        )
        print(text)
        return 0

    if args.hamiltonian:
        H = parse_simcount(args.hamiltonian)
    elif args.fields:
        H = heisenberg(args.size, args.fields)
    else:
        # Fall back to the recorded simcount Hamiltonian for this seed if present.
        from . import simcount
        try:
            # any recorded algorithm prints the same seed-1 Hamiltonian
            H = simcount.run(args.size, "pf2com", "cz", args.seed).hamiltonian
        except Exception as e:  # noqa: BLE001
            sys.exit(f"need --fields or --hamiltonian (no simcount fixture available: {e})")

    pl = compare.run_pennylane(H, args.algorithm, args.gatebase, args.time, args.steps,
                               args.zrots, args.template, args.up_to_phase, args.synth)
    sim = pl.sim
    print(f"Hamiltonian: {H.n_terms} terms, split={H.split}")
    print(f"{sim.r}: Subroutine: S  (t={sim.t:g}, eps_trotter={sim.eps_trotter:g}"
          + (f", eps_rot={pl.eps_rot:.4e}" if pl.eps_rot else "") + ")\n")
    print(resources.format_gatecount(pl.step_counts, 'Subroutine: "S"'))
    print()
    print(resources.format_gatecount(pl.aggregated))
    if args.optimise:
        from . import optimise
        print()
        print(resources.format_gatecount(optimise.optimise(sim, args.optimise),
                                         f"Optimised (blocks of {args.optimise} steps):"))
    return 0


def _bounds_table(args):
    from . import bounds, compare, resources, simcount
    from .circuit import simulate

    H = simcount.run(args.size, "pf2com", "cz", args.seed).hamiltonian
    t = float(args.size)
    eps = compare.GATEBASES[args.gatebase][0]
    names = ["ana", "min", "com", "emp", "tte"] + (["empirical"] if args.empirical else [])
    print(f"=== n={args.size}  seed={args.seed}  t={t:g}  eps_trotter={eps:g}  (Trotter number r / total gates) ===")
    print(f"{'order':>5}  " + "".join(f"{nm:>26}" for nm in names))
    for order in args.orders:
        row = f"{order:>5}  "
        for nm in names:
            try:
                if nm == "empirical":
                    from . import empirical
                    r = empirical.steps(order, H, t, eps)
                else:
                    r = bounds.BOUNDS[nm](order, H, t, eps)
                sim = simulate(H, f"pf{order}com", t, eps, r=r)
                g = resources.total(resources.count(sim.step)) * r
                row += f"{f'{r:,} / {g:.3g}':>26}"
            except Exception as e:  # noqa: BLE001
                row += f"{'n/a':>26}"
        print(row)
    print("\n'tte' = Theory of Trotter Error (Childs et al. 2021), Props 15/16/J.2; "
          "'empirical' = brute-force minimal r.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
