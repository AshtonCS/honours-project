"""Tests for the merging / optimiser split of Childs et al.'s saving (pysimcount.pfmergesplit)."""

import warnings

import pytest

warnings.filterwarnings("ignore")

from pysimcount import pfmergesplit  # noqa: E402
from pysimcount.nam import samples  # noqa: E402

N = 6


def test_s2_split_by_hand():
    # Unmerged S2 step (periodic, simcount template): forward and backward
    # sweeps over XX, YY, ZZ, 2 CNOTs per bond each -> 12n. merge combines
    # the two ZZ layers in the middle (the fields between them commute) -> 10n.
    # The optimiser also merges the XX layer shared by consecutive steps -> 8n.
    c = pfmergesplit.run_case("suzuki2", N, seed=1, verify_max_n=N)
    assert (c["U"]["CNOT"], c["M"]["CNOT"], c["O"]["CNOT"]) == (12 * N, 10 * N, 8 * N)
    assert c["split"]["CNOT"]["merging"] == pytest.approx(1 / 6)
    assert c["split"]["CNOT"]["optimiser_only"] == pytest.approx(1 / 6)
    assert c["verified"]


def test_s4_split_by_hand():
    # S4 = 5 S2 cycles: unmerged 5 x 12n = 60n. merge leaves (8q + 2)n = 42n
    # (q = 5); the optimiser removes only the boundary XX layer -> 40n. So
    # merging alone gives 18/60 = 30% of the 33.3%.
    c = pfmergesplit.run_case("suzuki4", N, seed=1, verify_max_n=0)
    assert (c["U"]["CNOT"], c["M"]["CNOT"], c["O"]["CNOT"]) == (60 * N, 42 * N, 40 * N)
    assert c["split"]["CNOT"]["merging"] == pytest.approx(0.30)
    assert "published" not in c  # n = 6 has no sample circuit


def test_first_order_has_no_saving():
    # Childs et al. SI I.3: "The first-order PF circuits do not admit any
    # optimization in the cnot counts." Rz is unchanged too; only
    # single-qubit Clifford gates go (so "total" is not compared).
    c = pfmergesplit.run_case("lie-trotter", N, seed=1, verify_max_n=0)
    for m in ("CNOT", "Rz"):
        assert c["U"][m] == c["M"][m] == c["O"][m] == {"CNOT": 6, "Rz": 4}[m] * N


def test_non_suzuki_raises():
    with pytest.raises(ValueError, match="Suzuki"):
        pfmergesplit.run_case("mo-q14", N, seed=1, verify_max_n=0)


@pytest.mark.skipif(not samples.path_for("pf4emp", 13, "preoptim").exists(), reason="simcount samples not on disk")
def test_s4_published_saving():
    # Published pf4emp n=13: 170,040 CNOT before, 113,386 after -> 33.32%.
    c = pfmergesplit.run_case("suzuki4", 13, seed=1, verify_max_n=0)
    assert c["published"]["before"]["CNOT"] == 170_040 == c["r"] * c["U"]["CNOT"]
    assert c["published_saving"]["CNOT"] == pytest.approx(1 - 113_386 / 170_040)
