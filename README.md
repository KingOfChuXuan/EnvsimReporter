# EnvsimReporter

OpenMM reporter that reproduces PLUMED's `ENVIRONMENTSIMILARITY` collective variable
and the `MORE_THAN` soft count `n_ice`.  The values are evaluated on report frames,
so the MD loop keeps its normal speed.  It is a monitoring tool for the water
phase-diagram workflow: use it to decide whether a coexistence box is freezing or
melting.

- `env.N` / `env_mean`: PLUMED `ENVIRONMENTSIMILARITY` (per-atom and mean).
- `n_ice`: PLUMED `MORE_THAN={RATIONAL R_0=... NN=8 MM=16}` soft count.
- `interval`, `r0`, references and output paths are all user-controlled.

The implementation follows PLUMED 2.10
`src/envsim/EnvironmentSimilarity.cpp`; the formulas and the soft-count behaviour
were checked against the existing `phase_diagram` trajectories (see
[Validation](#validation)).

---

## Installation

Dependencies:

```bash
pip install numpy scipy openmm
```

Place `EnvsimReporter.py` anywhere on `PYTHONPATH`, or copy it next to your run
script.

---

## Quick start

### 1. Calibration run: determine `R_0`

`R_0` depends on the phase, the reference environments and the water model.
Calibrate it for each phase and box.  Run a **short** pure-ice and pure-water
simulation with `r0=None`; the reporter then writes a per-atom file by default:

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
- `calib_env.dat`: `time env.1 env.2 ... env.N` (one row per frame)

The per-atom file has a plain text header (`time env.1 ... env.N`) so the columns
line up with the header.  Read it with:

```python
import pandas as pd
df = pd.read_table('calib_env.dat', sep=' ')
values = df.iloc[:, 1:].to_numpy()   # shape (n_frames, n_oxygen)
```

Plot the ice and water distributions and choose `R_0` (e.g. the value maximizing
Youden's J, or a value between the two distributions).  The PLUMED `n_ice` uses
the `RATIONAL` switching function: a molecule exactly at `R_0` contributes 0.5,
and a typical ice molecule (`env ≈ 0.88` with `R_0 = 0.65`) contributes ≈0.92.
Consequently, pure ice gives `n_ice ≈ 0.91 N` and pure water gives a small
baseline.

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
(`per_atom_file=False`) produces the summary file.  In calibration mode
(`r0=None`) the reporter always writes `env.N`, regardless of this flag.

To extend existing files after a restart, pass `append=True`.

---

## Parameters

| parameter | default | meaning |
|---|---|---|
| `file` | required | main summary file; `None` = values available as attributes |
| `interval` | required | report interval in MD steps; user-controlled |
| `references` | required | list of reference PDB files (displacement vectors, Å) |
| `sigma` | `0.05` | PLUMED `SIGMA`, in **nm** (0.05 nm = 0.5 Å) |
| `cutoff` | `3.0` | PLUMED `CUTOFF`: `r_cut = max|r0| + cutoff*sigma` |
| `lam` | `100.0` | PLUMED `LAMBDA` soft-max parameter |
| `r0` | `None` | PLUMED `MORE_THAN` `R_0`; `None` runs in calibration mode and writes per-atom `env.N` |
| `nn`, `mm` | `8`, `16` | PLUMED `MORE_THAN={RATIONAL ...}` (mm=0 means 2*nn) |
| `group` | `None` | 0-based atom indices; default = O atoms of water residues |
| `water_names` | `('HOH','WAT','SOL','TIP3','TIP3P','SPC','SPCE')` | residue names for automatic O selection |
| `per_atom_file` | `False` | `False`: summary file when `r0` is given, forced `env.N` when `r0=None`; `True`: automatic name; path: write there |
| `append` | `False` | extend existing files |
| `time_unit` | `nanosecond` | `openmm.unit.Unit` for the time column, e.g. `nanosecond` or `picosecond` |
| `normalization` | `'per_reference'` | `'per_reference'` (correct) or `'plumed'` (PLUMED's first-reference convention) |

### `interval`

- Calibration: `100` (or smaller) to get enough frames quickly.
- Production: `1000` is enough for a freeze/melt decision; `500` is a good middle
  ground.  At 100 ns/day, a 1280-oxygen system with 10 references costs ≈1.4% at
  `interval=1000` and ≈2.7% at `interval=500` (see [Performance](#performance)).

### `group`

By default the reporter selects all oxygen atoms whose residue name is in
`water_names` (using the element symbol when available, otherwise the atom name).
For an unusual topology, pass the exact **0-based** OpenMM particle indices:

```python
group = [atom.index for atom in topology.atoms()
         if atom.residue.name == 'HOH' and atom.element.symbol == 'O']
reporter = EnvsimReporter(..., group=group)
```

The kernel is single-species: the same group supplies centres and neighbours,
matching `SPECIES=...` in PLUMED.

---

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

---

## Reference PDB files

Each `ref*.pdb` contains the **displacement vectors from the central atom to its
neighbours**, in Ångström, e.g.:

```
MODEL     1
ATOM      1    O MOL     1      -0.079  -2.758   0.011  1.00  0.00           O
ATOM      2    O MOL     1       1.311   0.935  -2.284  1.00  0.00           O
ATOM      3    O MOL     1      -2.591   0.943   0.087  1.00  0.00           O
ATOM      4    O MOL     1       1.310   0.916   2.248  1.00  0.00           O
ENDMDL
```

The parser reads fixed PDB columns 31–54 directly and uses the first `MODEL`.
OpenMM's PDB reader collapses atoms that share a serial number, which would turn
the 4-atom `ref1.pdb` into a 1-atom reference.

---

## Normalization of multiple reference environments

The single-template kernel is normalized by the number of atoms `n` in the
reference environment, so that `k(χ0) = 1` (paper Eq. 6).  For several templates,
the paper generalizes to

```
k_X(χ) = (1/λ) log Σ_l exp(λ k_l(χ))
```

and the consistent choice is to normalize **each** `k_l` by its own `n_l`, so that
each template satisfies `k_l(χ_l) = 1`.

PLUMED's implementation divides *all* environments by the size of the **first**
reference (`str_natoms = environments[0].size()`), which implicitly assumes that
all reference environments have the same number of atoms.  In the water phase
diagrams all `ref*.pdb` of a given phase have the same number of atoms (Ih: 4,
II: 5, ...), so the two choices are identical.

This reporter defaults to `normalization='per_reference'` (the mathematically
correct generalization).  To mirror PLUMED's code exactly — for example when
comparing against an existing `colvar` file — pass `normalization='plumed'`.
For equal-size references, which is the case for the current water phases, the
default already produces identical numbers; the choice matters only when
template sizes differ.  In that mixed-size case, `'per_reference'` stays silent
and `'plumed'` emits a warning, since the latter applies the first template's
size to every template.  The validation below uses `'plumed'` to compare
directly with the PLUMED output.

---

## Output files

### Summary file

Tab-separated, one row per report:

```
#Step	Time(ns)	Temperature(K)	env_mean	n_ice
```

- `Time(ns)`: default `nanosecond`; pass `time_unit=picosecond` to match PLUMED colvar exactly.
  If the State has no time, the reporter falls back to `step * dt`; if that also
  fails, it writes `nan`.
- `Temperature(K)`: from `integrator.getTemperature()`; if unavailable, `nan`.
- `n_ice`: written when `r0` is given.
- The first row is written at the first report interval (step = `interval`); PLUMED's
  `colvar` can include a step-0 row.
- Volume: add `StateDataReporter(volume=True)` to the simulation.

### Per-atom file

Space-separated:

```
time env.1 env.2 ... env.N
0.000005 0.951399 0.878237 ...
```

- The header is plain text, so `pandas.read_table(..., sep=' ')` gives columns
  that match it exactly.
- `env.i` follows the order of the selected atoms (the same order as `reporter._ids`
  / the `group` you passed).  For the default selection this is the order of the
  oxygen atoms in the topology.
- Written when `r0 is None` (automatic / forced) or when `per_atom_file` is `True` or a path.
- With `file=None`, `r0=None` and `per_atom_file=False`, the automatic name is
  `env_per_atom.dat` in the current directory.

---

## Performance

Measured with the corrected kernel (both directions per pair, required because
the CV is orientation-dependent) on random liquid configurations at
33.4 molecules/nm³, single process, `scipy.spatial.cKDTree`:

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
8000-oxygen systems, use `interval>=1000` (or `5000`) to keep the overhead small.
Triclinic cells use boundary ghost particles: only atoms near a cell face get
periodic images, and the primary atoms plus their ghosts share one KD-tree.
When `r_cut` is not smaller than the smallest box width, the reporter falls back
to an O(N²) search that explicitly enumerates periodic images.

The State request contains positions; the reporter computes the values on the CPU
and leaves forces, energies and velocities in the simulation.

---

## Validation

`tests/validate_against_plumed.py` replays the existing DCD trajectories and
compares with the PLUMED `colvar` files.  The replay test uses MDAnalysis to read
the DCD files.

| system | box | references | `max|Δ env|` | `max|Δ env_mean|` | `max|Δ n_ice|` |
|---|---|---|---|---|
| Ih ice, 10 frames | orthorhombic | 4 × 4 atoms | — | 5.1e-7 | 3.8e-5 |
| Ih water, 10 frames | orthorhombic | 4 × 4 atoms | 1.6e-6 | — | — |
| II ice, 10 frames | **triclinic** | 12 × 5 atoms | — | 4.4e-7 | 1.3e-5 |

The deviations come from the DCD single-precision coordinates and the 6-decimal
`colvar` output; the algorithm itself agrees.  `tests/test_pbc.py` also checks the
orthorhombic and triclinic neighbour search against the brute-force `O(N²)`
implementation (max difference ≈2e-15) and the normalization/switch edge cases.

Run:

```bash
python tests/test_pbc.py
python tests/validate_against_plumed.py
```

---

## Notes and pitfalls

1. **`n_ice` uses the RATIONAL switch.**  The phase-diagram thresholds calibrated
   with PLUMED (`60/640`, etc.) assume this definition.  A value exactly at `R_0`
   contributes 0.5; values above `R_0` contribute less than 1 and approach 1
   asymptotically (`r -> infinity`).
2. **The CV is orientation-dependent.**  The kernel uses the vector components
   `(x, y, z)`; the four Ih references cover four orientations, so keep all of
   them.
3. **Neighbour cutoff.**  `r_cut = max|r0| + 3σ`; for Ih σ=0.05 nm this is
   ≈0.43 nm, for II ≈0.47 nm.  In an orthorhombic box, keep `r_cut` below half
   the smallest box length; the reporter warns if multiple periodic images can
   enter the cutoff.
4. **`SPECIES` supplies both centres and neighbours.**  Only O atoms enter the
   kernel.
5. **Time unit.**  Output uses `nanosecond` by default; PLUMED's `colvar` is in
   `ps`.  Pass `time_unit=picosecond` (`from openmm.unit import picosecond`) to
   compare files directly.
6. **Group indices are 0-based** (OpenMM convention).  PLUMED's `SPECIES=1-1296:3`
   is 1-based.
7. **Periodic systems.**  The reporter requires valid periodic box vectors.
8. **Reference PDB parsing.**  The parser reads fixed columns directly; see the
   note above about duplicate serial numbers.
9. **Early stopping.**  Read `reporter.n_ice` in the driver and stop when it
   crosses your threshold.
10. **Volume.**  Add `StateDataReporter(volume=True)`; both reporters share the
    same State request.
11. **PLUMED semantics.**  `n_ice` is PLUMED's `MORE_THAN` soft count,
    `Σ(1 - switch)`.  When comparing with paper Eq. (7), use the same sign
    convention as PLUMED for thresholds.
