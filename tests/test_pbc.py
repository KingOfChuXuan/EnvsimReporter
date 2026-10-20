# Copyright JiangXuan
# Update Date: 2026-10-20
"""Self-contained tests for the periodic minimum image and normalization.

Run from the repository root:

    python tests/test_pbc.py
"""

from __future__ import annotations

import itertools
import pathlib
import sys
import tempfile
import warnings

import numpy as np
from openmm.unit import nanometer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from EnvsimReporter import (  # noqa: E402
    EnvsimReporter,
    _box_widths,
    _pairs_bruteforce,
    _pairs_triclinic,
    compute_environment_similarity,
    more_than_count,
)


def _random_reference(rng: np.random.Generator, n: int, radius: float = 0.28) -> np.ndarray:
    vectors = rng.normal(size=(n, 3))
    vectors *= radius / np.linalg.norm(vectors, axis=1).mean()
    return vectors


def _check_box(name: str, box: np.ndarray, rng: np.random.Generator) -> None:
    frac = rng.random((120, 3))
    pos = frac @ box
    refs = [_random_reference(rng, 4) for _ in range(4)]

    auto = compute_environment_similarity(
        pos, box, refs, sigma=0.05, cutoff=3.0, lam=100.0, neighbor_method="auto"
    )
    brute = compute_environment_similarity(
        pos, box, refs, sigma=0.05, cutoff=3.0, lam=100.0, neighbor_method="bruteforce"
    )
    diff = float(np.abs(auto - brute).max())
    print(f"{name}: max |auto - bruteforce| = {diff:.3e}")
    assert diff < 1.0e-12, (name, diff)


def test_minimum_image() -> None:
    rng = np.random.default_rng(20261020)

    orthorhombic = np.diag([2.35, 2.20, 2.69])
    _check_box("orthorhombic", orthorhombic, rng)

    # A generic lower-triangular triclinic box; widths are all ~2 nm.
    triclinic = np.array(
        [
            [2.34, 0.00, 0.00],
            [0.35, 2.20, 0.00],
            [0.25, 0.40, 2.15],
        ]
    )
    _check_box("triclinic", triclinic, rng)


def test_reference_normalization() -> None:
    rng = np.random.default_rng(7)
    box = np.diag([2.5, 2.5, 2.5])
    pos = rng.random((30, 3)) @ box

    refs_equal = [_random_reference(rng, 4) for _ in range(3)]
    per_ref = compute_environment_similarity(
        pos, box, refs_equal, normalization="per_reference"
    )
    plumed = compute_environment_similarity(
        pos, box, refs_equal, normalization="plumed"
    )
    assert np.allclose(per_ref, plumed, atol=1.0e-12)

    # Different reference sizes: per-reference normalization is the correct
    # choice and stays silent; the PLUMED-compatibility mode warns because it
    # applies the first template's size to every template.
    refs_mixed = [_random_reference(rng, 4), _random_reference(rng, 5)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = compute_environment_similarity(
            pos, box, refs_mixed, normalization="per_reference"
        )
    assert not caught, [str(w.message) for w in caught]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = compute_environment_similarity(
            pos, box, refs_mixed, normalization="plumed"
        )
    assert any("plumed" in str(w.message) for w in caught)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b)
    print("normalization: equal-size references agree; mixed sizes warn only for 'plumed'")


def test_more_than_count() -> None:
    # A value exactly at r0 contributes 0.5.
    assert abs(more_than_count(np.array([0.65]), 0.65) - 0.5) < 1.0e-12
    # A value of zero contributes nothing.
    assert abs(more_than_count(np.zeros(10), 0.65)) < 1.0e-12
    # Values above r0 still contribute less than 1 (the switching function
    # decays slowly up to r_max = 4.217 * r0).
    one = more_than_count(np.array([1.0]), 0.65)
    assert 0.8 < one < 1.0

    # No artificial truncation at 4.217*r0: the switch keeps decaying.
    r0 = 0.65
    large = 5.0 * r0
    expected = 1.0 - (1.0 - (large / r0) ** 8) / (1.0 - (large / r0) ** 16)
    got = more_than_count(np.array([large]), r0)
    assert abs(got - expected) < 1.0e-12 and got < 1.0
    try:
        more_than_count(np.array([0.5]), 0.65, nn=16, mm=8)
    except ValueError:
        pass
    else:
        raise AssertionError("nn >= mm must be rejected")
    for bad_r0 in (np.nan, 0.0):
        try:
            more_than_count(np.array([0.5]), bad_r0)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite/non-positive r0 must be rejected")
    print(f"more_than_count: k=0.65 -> 0.5, k=1.0 -> {one:.4f}")




def test_skewed_box_minimum_image() -> None:
    # Regression for a box where component-wise fractional rounding is not the
    # minimum image.  The brute-force fallback must still find this pair.
    box = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.8, 0.2, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    frac = np.array([[0.0, 0.0, 0.5], [0.49, 0.49, 0.5]])
    pos = frac @ box

    # r_cut > smallest width: 27 images are not applicable, brute force must
    # still find the pair.
    ci_b, cj_b, _ = _pairs_bruteforce(pos, box, 0.25)
    assert len(ci_b) == 1 and int(ci_b[0]) == 0 and int(cj_b[0]) == 1

    # A boundary case with r_cut < smallest width: the ghost and brute-force
    # paths must agree.
    box2 = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.2, 1.6, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    pos2 = np.array([[0.0, 0.0, 0.5], [0.49, 0.49, 0.5]]) @ box2
    ci_b2, cj_b2, _ = _pairs_bruteforce(pos2, box2, 1.0)
    ci_t2, cj_t2, _ = _pairs_triclinic(pos2, box2, 1.0)
    assert len(ci_b2) == 1 and len(ci_t2) == 1
    assert (int(ci_b2[0]), int(cj_b2[0])) == (int(ci_t2[0]), int(cj_t2[0]))
    print("skewed box: brute-force and ghost-image minimum images agree")


def test_per_atom_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        ref = root / "ref1.pdb"
        ref.write_text(
            "ATOM      1    O MOL     1       2.760   0.000   0.000"
            "  1.00  0.00           O\n",
            encoding="utf-8",
        )

        calib = EnvsimReporter(
            file=root / "calib.dat",
            interval=100,
            references=[ref],
            r0=None,
            per_atom_file=False,
        )
        assert calib._atom_path is not None
        assert calib._atom_path.endswith("calib_env.dat")
        calib.close()

        prod = EnvsimReporter(
            file=root / "prod.dat",
            interval=100,
            references=[ref],
            r0=0.65,
            per_atom_file=False,
        )
        assert prod._atom_path is None
        prod.close()

        try:
            EnvsimReporter(
                file=None,
                interval=100,
                references=[ref],
                r0=0.65,
                per_atom_file=False,
                time_unit=nanometer,
            )
        except TypeError:
            pass
        else:
            raise AssertionError("time_unit=nanometer must be rejected")

        # Invalid per_atom_file must fail before the existing summary is opened
        # and truncated.
        keep = root / "keep.dat"
        keep.write_text("KEEP\n", encoding="utf-8")
        try:
            EnvsimReporter(
                file=keep,
                interval=100,
                references=[ref],
                r0=0.65,
                per_atom_file=1,
            )
        except TypeError:
            pass
        else:
            raise AssertionError("per_atom_file=1 must be rejected")
        assert keep.read_text(encoding="utf-8") == "KEEP\n"

        auto_named = EnvsimReporter(
            file=root / "named.dat",
            interval=100,
            references=[ref],
            r0=0.65,
            per_atom_file=True,
        )
        assert auto_named._atom_path is not None
        assert auto_named._atom_path.endswith("named_env.dat")
        auto_named.close()
        print("per-atom policy: calibration forces env.N; production is summary-only")




def _manual_single_reference(
    pos: np.ndarray,
    box: np.ndarray,
    ref: np.ndarray,
    sigma: float,
    r_cut: float,
) -> np.ndarray:
    """Direct double loop with explicit periodic images; no vector tricks."""
    n = len(pos)
    inv_box = np.linalg.inv(box)
    n_shift = max(1, int(np.ceil(r_cut / float(_box_widths(box).min()))))
    shifts = list(itertools.product(range(-n_shift, n_shift + 1), repeat=3))
    env = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            frac = (pos[j] - pos[i]) @ inv_box
            base = np.round(frac)
            best = np.inf
            best_d = None
            for shift in shifts:
                d = (frac - base + shift) @ box
                r2 = float(d @ d)
                if r2 < best:
                    best = r2
                    best_d = d
            if best <= r_cut * r_cut:
                env[i] += np.exp(-np.sum((best_d - ref) ** 2) / (4 * sigma**2)) / len(ref)
    return env


def test_kernel_against_manual_loop() -> None:
    pos = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.35, 0.10, 0.10],
            [0.35, 0.25, 0.10],
        ]
    )
    box = np.diag([2.0, 2.0, 2.0])
    ref = np.array([[0.20, 0.0, 0.0]])
    sigma = 0.10
    env = compute_environment_similarity(
        pos,
        box,
        [ref],
        sigma=sigma,
        cutoff=3.0,
        neighbor_method="bruteforce",
    )
    manual = _manual_single_reference(pos, box, ref, sigma, 0.20 + 3.0 * sigma)
    assert np.allclose(env, manual, atol=1.0e-12)
    assert abs(env[0] - env[1]) > 1.0e-6
    print("kernel: matches hand-computed double loop; forward != backward")




def test_nonperiodic_box_rejected() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ref = np.array([[0.2, 0.0, 0.0]])
    try:
        compute_environment_similarity(pos, np.zeros((3, 3)), [ref])
    except ValueError:
        pass
    else:
        raise AssertionError("a zero-volume box must be rejected")
    print("box: zero-volume/non-periodic box is rejected")


if __name__ == "__main__":
    test_minimum_image()
    test_reference_normalization()
    test_more_than_count()
    test_skewed_box_minimum_image()
    test_per_atom_policy()
    test_kernel_against_manual_loop()
    test_nonperiodic_box_rejected()
    print("all PBC/normalization tests passed")
