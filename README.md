# EnvsimReporter

An OpenMM reporter for PLUMED's `ENVIRONMENTSIMILARITY` collective variable.
It evaluates the CV on report frames and writes the per-atom similarity, its
mean, and the PLUMED `MORE_THAN` soft count `n_ice` used by the water
phase-diagram workflow.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Parameters](#parameters)
- [PLUMED input mapping](#plumed-input-mapping)
- [Reference PDB files](#reference-pdb-files)
- [Normalization](#normalization)
- [Output files](#output-files)
- [Performance](#performance)
- [Validation](#validation)
- [Notes](#notes)
- [Repository layout](#repository-layout)
- [References](#references)
- [License](#license)
- [Citation](#citation)

## Features

- Reproduces PLUMED's `ENVIRONMENTSIMILARITY` for water oxygen atoms, with
  `CRYSTAL_STRUCTURE=CUSTOM` and multiple `REFERENCE_i` templates.
- Reproduces the `MORE_THAN={RATIONAL R_0=... NN=8 MM=16}` soft count `n_ice`.
- Evaluates on report frames; the MD loop keeps its normal speed.
- Supports orthorhombic and triclinic periodic boxes.
- Uses boundary ghost particles for triclinic neighbour search.
- User-controlled `interval`, `r0`, reference environments and output paths.

## Requirements

- Python 3.9 or newer
- NumPy
- SciPy
- OpenMM 8.2 or newer (tested with 8.5.2)

A ready-to-use conda environment is provided in `environment.yml`.  OpenMM is
distributed through conda-forge.  The optional DCD replay test additionally uses
MDAnalysis.

## Installation

This repository is a standalone module.

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate envsim
```

Then copy `EnvsimReporter.py` next to your run script, or add this repository to
`PYTHONPATH`:

```python
import sys
sys.path.insert(0, '/path/to/EnvsimReporter')

from EnvsimReporter import EnvsimReporter
```

## Quick start

### 1. Calibration: determine `R_0`

`R_0` depends on the phase, the reference environments and the water model.
Calibrate it for each phase and box.  Run a short pure-ice and pure-water
simulation with `r0=None`; the reporter writes a per-atom file:

```python
from EnvsimReporter import EnvsimReporter

reporter = EnvsimReporter(
    file='calib.dat',
    interval=100,                 # short interval for calibration
    references=['ref1.pdb', 'ref2.pdb', 'ref3.pdb', 'ref4.pdb'],
    sigma=0.05,                   # nm, same as PLUMED SIGMA
    r0=None,                      # calibration mode
)
simulation.reporters.append(reporter)
```

This writes:

- `calib.dat`: `#Step  Time(ns)  Temperature(K)  env_mean`
- `calib_env.dat`: `time env.1 env.2 ... env.N`

Read the per-atom file with:

```python
import pandas as pd

df = pd.read_table('calib_env.dat', sep=' ')
values = df.iloc[:, 1:].to_numpy()   # shape (n_frames, n_oxygen)
```

Plot the ice and water distributions and choose `R_0`, for example the value
maximizing Youden's J or a value between the two distributions.  `n_ice` uses
the `RATIONAL` switching function: a molecule exactly at `R_0` contributes 0.5,
and a typical ice molecule (`env ≈ 0.88` with `R_0 = 0.65`) contributes ≈0.92.
Pure ice therefore gives `n_ice ≈ 0.91 N`.

### 2. Production run

```python
reporter = EnvsimReporter(
    file='env.dat',
    interval=1000,                # production: 1000 steps is enough
    references=['ref1.pdb', 'ref2.pdb', 'ref3.pdb', 'ref4.pdb'],
    sigma=0.05,
    r0=0.65,                      # calibrated for this phase/box
    nn=8, mm=16,
)
simulation.reporters.append(reporter)
```

`env.dat`:

```
#Step   Time(ns)   Temperature(K)   env_mean   n_ice
1000    1.000000   273.00           0.880697   393.284100
2000    2.000000   273.00           0.875012   387.920300
```

Example from a 432-oxygen box.

Early stopping is handled by the driver:

```python
if reporter.n_ice <= L or reporter.n_ice >= U:
    break

# also available after each report:
reporter.env_mean   # float
reporter.env        # numpy array of per-oxygen values (last frame)
```

For per-atom values during production, pass
`per_atom_file='env_per_atom.dat'`.  With `r0` given, the default
(`per_atom_file=False`) writes the summary file.  In calibration mode
(`r0=None`) the reporter writes `env.N` regardless of this flag.

To extend existing files after a restart, pass `append=True`.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `file` | required | main summary file; `None` = values available as attributes |
| `interval` | required | report interval in MD steps |
| `references` | required | list of reference PDB files (displacement vectors, Å) |
| `sigma` | `0.05` | PLUMED `SIGMA`, in **nm** (0.05 nm = 0.5 Å) |
| `cutoff` | `3.0` | PLUMED `CUTOFF`: `r_cut = max|r0| + cutoff*sigma` |
| `lam` | `100.0` | PLUMED `LAMBDA` soft-max parameter |
| `r0` | `None` | PLUMED `MORE_THAN` `R_0`; `None` runs in calibration mode |
| `nn`, `mm` | `8`, `16` | PLUMED `MORE_THAN={RATIONAL ...}` (`mm=0` means `2*nn`) |
| `group` | `None` | 0-based atom indices; default = O atoms of water residues |
| `water_names` | `('HOH','WAT','SOL','TIP3','TIP3P','SPC','SPCE')` | residue names for automatic O selection |
| `per_atom_file` | `False` | `False`: summary file when `r0` is given, `env.N` when `r0=None`; `True`: automatic name; path: write there |
| `append` | `False` | extend existing files |
| `time_unit` | `nanosecond` | `openmm.unit.Unit` for the time column, e.g. `nanosecond` or `picosecond` |
| `normalization` | `'per_reference'` | `'per_reference'` or `'plumed'` (first-reference convention) |

### `interval`

- Calibration: `100` (or smaller) for enough frames quickly.
- Production: `1000` for a freeze/melt decision; `500` is a good middle ground.
  At 100 ns/day, a 1280-oxygen system with 10 references costs ≈1.4% at
  `interval=1000` and ≈2.7% at `interval=500` (see [Performance](#performance)).

### `group`

By default the reporter selects all oxygen atoms whose residue name is in
`water_names`, using the element symbol when available and the atom name
otherwise.  For an unusual topology, pass the exact 0-based OpenMM particle
indices:

```python
group = [atom.index for atom in topology.atoms()
         if atom.residue.name == 'HOH' and atom.element.symbol == 'O']
reporter = EnvsimReporter(..., group=group)
```

The same group supplies centres and neighbours, matching `SPECIES=...` in
PLUMED.

## PLUMED input mapping

| PLUMED | reporter |
|---|---|
| `SPECIES=1-1296:3` | automatic water O, or `group=[...]` (0-based) |
| `SIGMA=0.05` | `sigma=0.05` (nm) |
| `CRYSTAL_STRUCTURE=CUSTOM` + `REFERENCE_i=refi.pdb` | `references=['ref1.pdb', ...]` |
| `CUTOFF=3` (default) | `cutoff=3.0` |
| `LAMBDA=100` (default) | `lam=100.0` |
| `MORE_THAN={RATIONAL R_0=0.65 NN=8 MM=16}` | `r0=0.65, nn=8, mm=16` |
| `MEAN` | `env_mean` column |
| `PRINT STRIDE=100` | `interval=100` |
| `PRINT STRIDE=1000` | `interval=1000` |

For example, this PLUMED input:

```plumed
env: ENVIRONMENTSIMILARITY ...
   SPECIES=1-1296:3
   SIGMA=0.05
   CRYSTAL_STRUCTURE=CUSTOM
   REFERENCE_1=ref1.pdb
   REFERENCE_2=ref2.pdb
   REFERENCE_3=ref3.pdb
   REFERENCE_4=ref4.pdb
   MORE_THAN={RATIONAL R_0=0.65 NN=8 MM=16}
   MEAN
...
PRINT ARG=env.mean,env.morethan FILE=colvar STRIDE=100
```

corresponds to:

```python
EnvsimReporter(
    file='env.dat', interval=100,
    references=['ref1.pdb', 'ref2.pdb', 'ref3.pdb', 'ref4.pdb'],
    sigma=0.05, cutoff=3.0, lam=100.0,
    r0=0.65, nn=8, mm=16,
)
```

## Reference PDB files

Each `ref*.pdb` contains the displacement vectors from the central atom to its
neighbours, in Ångström:

```
MODEL     1
ATOM      1    O MOL     1      -0.079  -2.758   0.011  1.00  0.00           O
ATOM      2    O MOL     1       1.311   0.935  -2.284  1.00  0.00           O
ATOM      3    O MOL     1      -2.591   0.943   0.087  1.00  0.00           O
ATOM      4    O MOL     1       1.310   0.916   2.248  1.00  0.00           O
ENDMDL
```

The parser reads fixed PDB columns 31–54 and uses the first `MODEL`.
Duplicate serial numbers are allowed.

## Normalization

Each reference template is normalized by its own atom count.  The
`normalization='plumed'` option mirrors PLUMED's implementation, which divides
every template by the first template's size.  With equal-size templates, the
two choices give identical values.

## Output files

### Summary file

Tab-separated, one row per report:

```
#Step	Time(ns)	Temperature(K)	env_mean	n_ice
```

- `Time(ns)`: default `nanosecond`; use `time_unit=picosecond` to match a
  PLUMED `colvar` file directly.
- `Temperature(K)`: from `integrator.getTemperature()`, written as `nan` when
  unavailable.
- `n_ice`: written when `r0` is given.
- Volume can be recorded with `StateDataReporter(volume=True)`.

### Per-atom file

Space-separated:

```
time env.1 env.2 ... env.N
0.000005 0.951399 0.878237 ...
```

- The header is plain text, so `pandas.read_table(..., sep=' ')` gives columns
  that match it.
- `env.i` follows the order of the selected atoms.
- The automatic file is `env_per_atom.dat` when `file=None`, `r0=None` and
  `per_atom_file=False`.

## Performance

Measured on random liquid configurations at 33.4 molecules/nm³, single process,
with `scipy.spatial.cKDTree`:

| O atoms | references | ms/frame |
|---:|---:|---:|
| 432 | 4 | ~2.5 |
| 1280 | 4 | ~5.8 |
| 1280 | 10 | ~11.9 |
| 8000 | 4 | ~27 |

At 100 ns/day (≈1.16 ps/s):

| system | `interval=100` | `interval=500` | `interval=1000` |
|---|---:|---:|---:|
| 432 O, 4 ref | ~2.8% | ~0.6% | ~0.3% |
| 1280 O, 4 ref | ~6.7% | ~1.3% | ~0.7% |
| 1280 O, 10 ref | ~13.7% | ~2.7% | ~1.4% |

Use `interval=1000` in production and `interval=100` for calibration.  For
8000-oxygen systems, use `interval>=1000` (or `5000`).

Triclinic cells use boundary ghost particles for the neighbour search.

The reporter reads positions from the `State` and computes on the CPU.

## Validation

`tests/validate_against_plumed.py` replays existing DCD trajectories and
compares the reporter with PLUMED's `colvar` files:

| system | box | references | `max|Δ env|` | `max|Δ env_mean|` | `max|Δ n_ice|` |
|---|---|---|---|---|
| Ih ice, 10 frames | orthorhombic | 4 × 4 atoms | — | 5.1e-7 | 3.8e-5 |
| Ih water, 10 frames | orthorhombic | 4 × 4 atoms | 1.6e-6 | — | — |
| II ice, 10 frames | **triclinic** | 12 × 5 atoms | — | 4.4e-7 | 1.3e-5 |

The agreement is within the precision of the DCD coordinates and the six-decimal
`colvar` output.  `tests/test_pbc.py` also checks the orthorhombic and triclinic
neighbour searches against a brute-force `O(N^2)` implementation, the
forward/backward kernel against a hand-computed double loop, and the
normalization/switch boundary cases.

Run the self-contained tests:

```bash
python tests/test_pbc.py
```

The DCD replay test is optional and needs your own trajectory data; see
`tests/validate_against_plumed.py` for the expected layout.  It skips
automatically when the data or MDAnalysis is unavailable:

```bash
python tests/validate_against_plumed.py
```

## Notes

1. **`n_ice` uses the RATIONAL switch.**  A value exactly at `R_0` contributes
   0.5; values above `R_0` contribute less than 1 and approach 1 asymptotically.
2. **The CV is orientation-dependent.**  The kernel uses the vector components
   `(x, y, z)`; the four Ih references cover four orientations.
3. **Neighbour cutoff.**  `r_cut = max|r0| + 3σ`; for Ih σ=0.05 nm this is
   ≈0.43 nm, for II ≈0.47 nm.  In an orthorhombic box, keep `r_cut` below half
   the smallest box length.
4. **`SPECIES` supplies both centres and neighbours.**  Only O atoms enter the
   kernel.
5. **Time unit.**  Output uses `nanosecond` by default; PLUMED's `colvar` is in
   `ps`.  Pass `time_unit=picosecond` (`from openmm.unit import picosecond`)
   to compare files directly.
6. **Group indices are 0-based** (OpenMM convention).
7. **Periodic systems.**  The reporter requires valid periodic box vectors.
8. **Reference PDB parsing.**  The parser reads fixed columns directly.
9. **Early stopping.**  Read `reporter.n_ice` in the driver and stop when it
   crosses your threshold.

## Repository layout

```text
EnvsimReporter.py                 reporter implementation
tests/test_pbc.py                 self-contained neighbour/kernel tests
tests/validate_against_plumed.py  optional DCD replay test
environment.yml                   conda environment
CITATION.cff                      citation metadata
LICENSE                           MIT license
.github/workflows/tests.yml       GitHub Actions unit tests
```

## References

1. P. M. Piaggi and M. Parrinello, "Calculation of phase diagrams in the
   multithermal-multibaric ensemble", *J. Chem. Phys.* **150**, 244119 (2019).
   <https://doi.org/10.1063/1.5102104>
2. G. A. Tribello, M. Bonomi, D. Branduardi, C. Camilloni and G. Bussi,
   "PLUMED 2: New feathers for an old bird", *Comput. Phys. Commun.* **185**,
   604–613 (2014). <https://doi.org/10.1016/j.cpc.2013.09.018>
3. P. Eastman et al., "OpenMM 7: Rapid development of high performance
   algorithms for molecular dynamics", *PLOS Comput. Biol.* **13**, e1005659
   (2017). <https://doi.org/10.1371/journal.pcbi.1005659>

## License

MIT.  See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the ENVIRONMENTSIMILARITY paper and this
repository.  Metadata is available in [CITATION.cff](CITATION.cff).
