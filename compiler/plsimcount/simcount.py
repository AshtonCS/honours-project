"""Run the reference Haskell/Quipper implementation and parse its output.

simcount lives in a Docker container (``simcount-new``) with the compiled
binary at ``/tmp/simcount-main``. Every run is cached as a text fixture
under ``fixtures/`` so the comparison and the test-suite work without
Docker once a case has been recorded.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .hamiltonian import Hamiltonian, parse_simcount

CONTAINER = "simcount-new"
BINARY = "/tmp/simcount-main"
FIXTURES = Path(__file__).parent / "fixtures" / "simcount"


@dataclass
class SimcountResult:
    n: int
    algo: str
    gatebase: str
    seed: int
    zrots: int | None
    hamiltonian: Hamiltonian
    r: int
    aggregated: dict[str, int]
    subroutines: dict[str, dict[str, int]] = field(default_factory=dict)
    raw: str = ""

    @property
    def per_step(self) -> dict[str, int]:
        """Everything in simcount is step x r, so divide back exactly."""
        out = {}
        for lbl, c in self.aggregated.items():
            if c % self.r:
                raise ValueError(f"{lbl}: {c} is not a multiple of r={self.r}")
            out[lbl] = c // self.r
        return out


def fixture_path(n, algo, gatebase, seed, zrots=None) -> Path:
    z = f"_z{zrots}" if zrots else ""
    return FIXTURES / f"n{n}_{algo}_{gatebase}{z}_m{seed}.txt"


def run(n: int, algo: str, gatebase: str = "cz", seed: int = 1, zrots: int | None = None,
        use_cache: bool = True, timeout: int = 3600) -> SimcountResult:
    path = fixture_path(n, algo, gatebase, seed, zrots)
    if use_cache and path.exists():
        return parse(path.read_text(), n, algo, gatebase, seed, zrots)

    cmd = f"cd /tmp && {BINARY} -n {n} -a {algo} -g {gatebase} -f gatecount -m {seed}"
    if zrots:
        cmd += f" -z {zrots}"
    proc = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"simcount failed:\n{proc.stderr}")
    text = f"$ {cmd}\n{proc.stderr}{proc.stdout}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return parse(text, n, algo, gatebase, seed, zrots)


_LINE = re.compile(r'^\s*(\d+): "(.+?), arity \d+"')
_CALL = re.compile(r"^\s*(\d+): Subroutine: (\S+),")


def parse(text: str, n, algo, gatebase, seed, zrots=None) -> SimcountResult:
    H = parse_simcount(text)

    # Top-level circuit: "<r>: Subroutine: S, shape ..." before any header.
    r = None
    for line in text.splitlines():
        if line.startswith("Subroutine:") or line.startswith("Aggregated"):
            break
        m = _CALL.match(line)
        if m and m.group(2) in ("S", "S1"):
            r = int(m.group(1))
    if r is None:
        raise ValueError("could not find the number of Trotter steps in simcount output")

    subroutines, aggregated, current = {}, {}, None
    for line in text.splitlines():
        if line.startswith('Subroutine: "'):
            current = subroutines.setdefault(line.split('"')[1], {})
        elif line.startswith("Aggregated gate count:"):
            current = aggregated
        elif current is not None and (m := _LINE.match(line)):
            current[m.group(2)] = int(m.group(1))

    return SimcountResult(n, algo, gatebase, seed, zrots, H, r, aggregated, subroutines, text)
