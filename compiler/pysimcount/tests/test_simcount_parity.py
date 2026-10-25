"""Gate-for-gate parity with njross/simcount.

Every recorded fixture under ``fixtures/simcount`` is replayed: the
PennyLane pipeline is run on the Hamiltonian simcount printed, and r plus
every aggregated gate count (rotations matched by angle value) must agree
exactly. New fixtures are recorded with ``python -m pysimcount compare``.
"""

import re
import warnings

import pytest

warnings.filterwarnings("ignore")

from pysimcount import compare, resources, simcount  # noqa: E402

_NAME = re.compile(r"n(\d+)_(pf\w+?)_(cz2ct|cz|ct)(?:_z(\d+))?_m(\d+)\.txt")


def _cases():
    for path in sorted(simcount.FIXTURES.glob("*.txt")):
        m = _NAME.fullmatch(path.name)
        if m:
            n, algo, g, z, seed = m.groups()
            yield pytest.param(int(n), algo, g, int(z) if z else None, int(seed), id=path.stem)


@pytest.mark.parametrize("n,algo,gatebase,zrots,seed", list(_cases()))
def test_parity(n, algo, gatebase, zrots, seed):
    if gatebase == "ct":
        pytest.skip("Clifford+T parity needs the Quipper backend; run `compare --synth quipper`")
    ref = simcount.run(n, algo, gatebase, seed, zrots)
    pl = compare.run_pennylane(ref.hamiltonian, algo, gatebase, zrots=zrots)

    assert pl.sim.r == ref.r
    assert resources.total(pl.step_counts) == resources.total(ref.per_step)
    for la, lb in compare._match_labels(ref.aggregated, pl.aggregated):
        assert la is not None and lb is not None, f"unmatched gate label {la or lb}"
        assert ref.aggregated[la] == pl.aggregated[lb], la
