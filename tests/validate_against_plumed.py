# Copyright JiangXuan
# Update Date: 2026-10-20
"""Replay existing DCD trajectories and compare with PLUMED's colvar output.

This acceptance test reads DCD trajectories with MDAnalysis and compares the
reporter's kernel with PLUMED's colvar output.

Run from the repository root:

    python tests/validate_against_plumed.py --data-root /path/to/phase_diagram

The data root is the directory that contains the `Ih/` and `II/` folders used
by the datasets below.  Without `--data-root`, the test skips.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from EnvsimReporter import (  # noqa: E402
    compute_environment_similarity,
    more_than_count,
    read_reference_pdb,
)


def _frame_row(times: np.ndarray, t_ps: float) -> int:
    idx = int(np.argmin(np.abs(times - t_ps)))
    if abs(float(times[idx]) - t_ps) > 1.0e-3:
        raise AssertionError(f"no colvar row at t={t_ps} ps (closest {times[idx]})")
    return idx


def _check_dataset(
    name: str,
    pdb: pathlib.Path,
    dcd: pathlib.Path,
    colvar: pathlib.Path,
    refs: list[pathlib.Path],
    r0: float | None,
    has_mean_column: bool,
) -> tuple[float | None, float | None, float | None]:
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_vectors

    references = [read_reference_pdb(path) * 0.1 for path in refs]
    universe = mda.Universe(str(pdb), str(dcd))
    oxygen = universe.select_atoms("name O")
    data = np.loadtxt(colvar)
    times = data[:, 0]

    max_env: float | None = None
    max_mean: float | None = None
    max_nice: float | None = None
    for frame in range(len(universe.trajectory)):
        universe.trajectory[frame]
        positions = oxygen.positions.copy() * 0.1  # Angstrom -> nm
        box = triclinic_vectors(universe.trajectory.ts.dimensions) * 0.1
        env = compute_environment_similarity(
            positions,
            box,
            references,
            sigma=0.05,
            cutoff=3.0,
            lam=100.0,
            normalization="plumed",
        )
        row = _frame_row(times, float(universe.trajectory.time))
        if has_mean_column:
            value = abs(float(env.mean()) - data[row, 1])
            max_mean = value if max_mean is None else max(max_mean, value)
            if r0 is not None:
                value = abs(more_than_count(env, r0) - data[row, 2])
                max_nice = value if max_nice is None else max(max_nice, value)
        else:
            value = float(np.abs(env - data[row, 1:]).max())
            max_env = value if max_env is None else max(max_env, value)

    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.3e}"

    print(
        f"{name:10s}  max|d env|={_fmt(max_env)}  "
        f"max|d mean|={_fmt(max_mean)}  max|d n_ice|={_fmt(max_nice)}"
    )
    return max_env, max_mean, max_nice


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        help="directory containing the Ih/ and II/ test datasets",
    )
    args = parser.parse_args()
    if args.data_root is None:
        print("no --data-root supplied; skipping the DCD replay test")
        return 0

    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        print("MDAnalysis not installed; skipping the DCD replay test")
        return 0

    phase_dir = args.data_root
    datasets = [
        (
            "Ih ice",
            phase_dir / "Ih/prepair/ice_Ih_432.pdb",
            phase_dir / "Ih/prepair/N_ice_test/trj_ice.dcd",
            phase_dir / "Ih/prepair/N_ice_test/colvar_ice.out",
            [phase_dir / f"Ih/prepair/ref{i}.pdb" for i in range(1, 5)],
            0.65,
            True,
        ),
        (
            "Ih water",
            phase_dir / "Ih/prepair/wat432.pdb",
            phase_dir / "Ih/prepair/CV_test/trj_wat.dcd",
            phase_dir / "Ih/prepair/CV_test/colvar_wat.out",
            [phase_dir / f"Ih/prepair/ref{i}.pdb" for i in range(1, 5)],
            None,
            False,
        ),
        (
            "II ice",
            phase_dir / "II/prepair/ice_II_432.pdb",
            phase_dir / "II/prepair/N_ice_test/trj_ice.dcd",
            phase_dir / "II/prepair/N_ice_test/colvar_ice.out",
            [phase_dir / f"II/prepair/ref{i}.pdb" for i in range(1, 13)],
            0.69,
            True,
        ),
    ]

    missing = []
    for item in datasets:
        for path in [item[1], item[2], item[3], *item[4]]:
            if not pathlib.Path(path).exists():
                missing.append(str(path))
    if missing:
        print("missing test data; skipping the DCD replay test:")
        for path in missing:
            print("  ", path)
        return 0

    failures = []
    for name, pdb, dcd, colvar, refs, r0, has_mean in datasets:
        try:
            max_env, max_mean, max_nice = _check_dataset(
                name, pdb, dcd, colvar, refs, r0, has_mean
            )
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            continue
        if (
            (max_env is not None and max_env > 1.0e-5)
            or (max_mean is not None and max_mean > 1.0e-5)
            or (max_nice is not None and max_nice > 1.0e-4)
        ):
            failures.append(
                f"{name}: tolerance exceeded "
                f"(env {max_env}, mean {max_mean}, n_ice {max_nice})"
            )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  -", failure)
        return 1
    print("\nPASS: all datasets agree with PLUMED within 1e-5 (env/mean) and 1e-4 (n_ice)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
