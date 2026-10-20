# Copyright JiangXuan
# Update Date: 2026-10-20
"""OpenMM reporter for PLUMED's ``ENVIRONMENTSIMILARITY`` collective variable.

The reporter reproduces the per-atom environment-similarity kernel of PLUMED 2.10
(``src/envsim/EnvironmentSimilarity.cpp``).  The values are evaluated on report
frames, so the MD loop keeps its normal speed.

The CV implemented here is

    k_i = (1/n) * sum_j sum_l exp(-|r_j - r_l^0|^2 / (4 sigma^2))

for every central atom ``i``, where ``r_j`` runs over the neighbours of ``i``
within ``r_cut = max|r^0| + CUTOFF * sigma`` and ``r_l^0`` runs over the atoms of
the reference environment.  If more than one reference environment is supplied,
the values are combined with the soft-max

    k_i = (1/lambda) * log(sum_m exp(lambda * k_i^(m))).

If ``r0`` is given, the PLUMED ``MORE_THAN={RATIONAL R_0=r0 NN=nn MM=mm}`` soft
count is also written as ``n_ice``.  This is the quantity used by the phase-diagram
workflow.

Typical use::

    from EnvsimReporter import EnvsimReporter

    reporter = EnvsimReporter(
        file='env.dat',
        interval=1000,
        references=['ref1.pdb', 'ref2.pdb', 'ref3.pdb', 'ref4.pdb'],
        sigma=0.05,      # nm, same as PLUMED SIGMA
        r0=0.65,         # None: calibration mode, writes per-atom env.N
        nn=8, mm=16,
    )
    simulation.reporters.append(reporter)

Reference PDB files must contain the displacement vectors from the central atom
to the neighbours, in Angstrom, exactly as PLUMED expects them.
"""

from __future__ import annotations

import itertools
import os
import warnings
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from openmm.unit import Unit, kelvin, nanometer, nanosecond, second

__all__ = [
    "EnvsimReporter",
    "compute_environment_similarity",
    "more_than_count",
    "read_reference_pdb",
]

_ANGSTROM_TO_NM = 0.1
_DEFAULT_WATER_NAMES = ("HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE")
_DEFAULT_OXYGEN_NAMES = ("O", "OW", "OH2", "O1", "OT")


def read_reference_pdb(path: str | os.PathLike) -> np.ndarray:
    """Read one reference environment from a PLUMED-style PDB file.

    PLUMED reference PDB files contain the *displacement vectors* from the
    central atom to its neighbours, in Angstrom.  The first model is read.
    The parser reads fixed PDB columns directly; ``openmm.app.PDBFile`` collapses
    atoms that share a serial number, which silently turns a 4-atom reference
    file into a 1-atom one.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 3)`` in Angstrom.
    """
    coords: list[tuple[float, float, float]] = []
    in_model = False
    model_seen = False
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for lineno, line in enumerate(handle, 1):
            record = line[:6].strip().upper()
            if record == "MODEL":
                if model_seen:
                    break
                model_seen = True
                in_model = True
                continue
            if record == "ENDMDL":
                if in_model:
                    break
                continue
            if record not in ("ATOM", "HETATM"):
                continue
            if model_seen and not in_model:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: cannot parse PDB coordinates "
                    f"(expected fixed columns 31-54)"
                ) from exc
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"{path}: no ATOM/HETATM records found")
    arr = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: non-finite coordinates")
    return arr


def _is_orthorhombic(box: np.ndarray, atol: float = 1.0e-8) -> bool:
    return bool(
        np.allclose(box, np.diag(np.diag(box)), rtol=0.0, atol=atol)
    )


def _box_widths(box: np.ndarray) -> np.ndarray:
    """Perpendicular widths of the three pairs of box faces, in nm."""
    a, b, c = box
    volume = abs(float(np.dot(a, np.cross(b, c))))
    if volume <= 0.0:
        raise ValueError("periodic box vectors are singular (non-positive volume)")
    widths = np.array(
        [
            volume / np.linalg.norm(np.cross(b, c)),
            volume / np.linalg.norm(np.cross(c, a)),
            volume / np.linalg.norm(np.cross(a, b)),
        ],
        dtype=np.float64,
    )
    if np.any(widths <= 0.0):
        raise ValueError("periodic box vectors are degenerate")
    return widths


def _pairs_orthorhombic(
    positions: np.ndarray, box: np.ndarray, r_cut: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour pairs for an orthorhombic box, using a KD-tree with PBC."""
    n = len(positions)
    lengths = np.diag(box).astype(np.float64, copy=True)
    if np.any(lengths <= 0.0):
        raise ValueError("orthorhombic box lengths must be positive")

    wrapped = positions % lengths
    tree = cKDTree(wrapped, boxsize=lengths)
    neighbours = tree.query_ball_point(
        wrapped, r_cut, workers=-1, return_sorted=False
    )

    counts = np.fromiter((len(x) for x in neighbours), dtype=np.int64, count=n)
    ci = np.repeat(np.arange(n, dtype=np.int64), counts)
    if counts.sum():
        cj = np.concatenate(
            [np.asarray(x, dtype=np.int64) for x in neighbours]
        )
    else:
        cj = np.empty(0, dtype=np.int64)

    keep = ci < cj
    ci, cj = ci[keep], cj[keep]

    d = positions[cj] - positions[ci]
    d -= lengths * np.round(d / lengths)
    return ci, cj, d


def _pairs_triclinic(
    positions: np.ndarray, box: np.ndarray, r_cut: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour pairs for a general (triclinic) box.

    The minimum image is generated by replicating the fractional coordinates in
    the 27 neighbouring cells and querying a KD-tree.  This is exact whenever
    ``r_cut`` is smaller than the smallest perpendicular box width.
    """
    n = len(positions)
    inv_box = np.linalg.inv(box)
    frac = positions @ inv_box
    frac -= np.floor(frac)  # fractional coordinates in [0, 1)

    shifts = np.array(
        list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64
    )
    images = (frac[None, :, :] + shifts[:, None, :]) @ box  # (27, n, 3)
    flat_images = images.reshape(-1, 3)
    image_ids = np.tile(np.arange(n, dtype=np.int64), len(shifts))
    centres = frac @ box

    tree = cKDTree(flat_images)
    neighbours = tree.query_ball_point(
        centres, r_cut, workers=-1, return_sorted=False
    )

    counts = np.fromiter((len(x) for x in neighbours), dtype=np.int64, count=n)
    ci = np.repeat(np.arange(n, dtype=np.int64), counts)
    if counts.sum():
        ck = np.concatenate(
            [np.asarray(x, dtype=np.int64) for x in neighbours]
        )
    else:
        ck = np.empty(0, dtype=np.int64)

    cj = image_ids[ck]
    d = flat_images[ck] - centres[ci]
    r2 = np.einsum("ij,ij->i", d, d)

    keep = (r2 < r_cut * r_cut) & (cj != ci) & (ci < cj)
    ci, cj, d, r2 = ci[keep], cj[keep], d[keep], r2[keep]

    # A pair can appear through several periodic images.  Keep the closest one.
    if len(ci):
        key = ci.astype(np.int64) * n + cj
        order = np.lexsort((r2, key))
        key_sorted = key[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = key_sorted[1:] != key_sorted[:-1]
        selected = order[first]
        ci, cj, d = ci[selected], cj[selected], d[selected]
    return ci, cj, d


def _pairs_bruteforce(
    positions: np.ndarray, box: np.ndarray, r_cut: float, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference O(N^2) implementation, used as a fallback and in tests.

    The minimum image is found by explicitly searching periodic shifts, not by
    rounding fractional coordinates component-wise: the latter is exact only
    for orthorhombic boxes.
    """
    n = len(positions)
    inv_box = np.linalg.inv(box)
    widths = _box_widths(box)
    n_shift = max(1, int(np.ceil(r_cut / float(widths.min()))))
    shifts = np.array(
        list(itertools.product(range(-n_shift, n_shift + 1), repeat=3)),
        dtype=np.float64,
    )
    if n_shift > 2:
        warnings.warn(
            "brute-force fallback searches a large number of periodic images; "
            "the box is very small compared with the neighbour cutoff",
            stacklevel=2,
        )

    ci_parts: list[np.ndarray] = []
    cj_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    cutoff2 = r_cut * r_cut
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d0 = positions[None, :, :] - positions[start:stop, None, :]
        frac = d0 @ inv_box
        base = np.round(frac)
        relative = frac - base
        best_r2 = np.full(relative.shape[:2], np.inf, dtype=np.float64)
        best_d = np.zeros_like(d0)
        for shift in shifts:
            candidate = (relative + shift) @ box
            r2 = np.einsum("...i,...i->...", candidate, candidate)
            improve = r2 < best_r2
            best_r2[improve] = r2[improve]
            best_d[improve] = candidate[improve]

        ii, jj = np.nonzero(best_r2 < cutoff2)
        ii = ii + start
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        ci_parts.append(ii.astype(np.int64))
        cj_parts.append(jj.astype(np.int64))
        d_parts.append(best_d[ii - start, jj])
    if ci_parts:
        return (
            np.concatenate(ci_parts),
            np.concatenate(cj_parts),
            np.concatenate(d_parts),
        )
    empty = np.empty(0, dtype=np.int64)
    return empty, empty, np.empty((0, 3), dtype=np.float64)


def compute_environment_similarity(
    positions: np.ndarray,
    box_vectors: np.ndarray,
    references: Sequence[np.ndarray],
    sigma: float = 0.05,
    cutoff: float = 3.0,
    lam: float = 100.0,
    normalization: str = "per_reference",
    neighbor_method: str = "auto",
) -> np.ndarray:
    """Compute the per-atom ENVIRONMENTSIMILARITY kernel.

    Parameters
    ----------
    positions : (N, 3) array
        Central/neighbour atom positions in nm.
    box_vectors : (3, 3) array
        Rows are the three periodic box vectors, in nm.
    references : list of (n_m, 3) arrays
        Reference environments (displacement vectors from the central atom), in nm.
    sigma : float
        Gaussian broadening, in nm (PLUMED ``SIGMA``).
    cutoff : float
        Neighbour cutoff beyond the largest reference distance, in units of
        ``sigma`` (PLUMED ``CUTOFF``).
    lam : float
        Soft-max parameter for multiple references (PLUMED ``LAMBDA``).
    normalization : {'per_reference', 'plumed'}
        ``per_reference`` divides each reference by its own atom count, which is
        the mathematically consistent generalization of the paper and gives
        ``k(chi_l) = 1`` for every template.  ``plumed`` reproduces PLUMED's
        implementation, which divides every reference by the atom count of the
        *first* reference; the two are identical when all templates have the
        same number of atoms (the normal case).
    neighbor_method : {'auto', 'orthorhombic', 'triclinic', 'bruteforce'}
        Neighbour-search implementation.  ``auto`` picks the fastest safe one.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N,)`` with the similarity of each central atom.
    """
    pos = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box_vectors, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if box.shape != (3, 3):
        raise ValueError("box_vectors must have shape (3, 3)")
    if len(pos) == 0:
        raise ValueError("positions is empty")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions contain non-finite values")
    if not np.all(np.isfinite(box)):
        raise ValueError("box_vectors contain non-finite values")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite number")
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be a non-negative finite number")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a positive finite number")
    if normalization not in ("per_reference", "plumed"):
        raise ValueError("normalization must be 'per_reference' or 'plumed'")

    refs = [np.asarray(ref, dtype=np.float64) for ref in references]
    if not refs:
        raise ValueError("at least one reference environment is required")
    for ref in refs:
        if ref.ndim != 2 or ref.shape[1] != 3 or len(ref) == 0:
            raise ValueError("each reference must have shape (n, 3) with n > 0")
        if not np.all(np.isfinite(ref)):
            raise ValueError("reference environments contain non-finite values")

    sizes = np.array([len(ref) for ref in refs], dtype=np.float64)
    sizes_equal = bool(np.all(sizes == sizes[0]))
    if sizes_equal:
        # Common case: one normalization factor for every template.  This is
        # identical to PLUMED's first-reference convention.
        norm_div = np.full(len(refs), sizes[0], dtype=np.float64)
    elif normalization == "plumed":
        warnings.warn(
            "normalization='plumed' divides every template by the first "
            "reference size, while the reference sizes differ.  The templates "
            "are then not individually self-normalized; use "
            "normalization='per_reference' for the self-normalized kernel.",
            stacklevel=2,
        )
        norm_div = np.full(len(refs), sizes[0], dtype=np.float64)
    else:
        norm_div = sizes
    inv_norm = 1.0 / norm_div

    max_ref_norm = max(float(np.linalg.norm(ref, axis=1).max()) for ref in refs)
    r_cut = max_ref_norm + cutoff * sigma

    method = neighbor_method
    if method == "auto":
        if _is_orthorhombic(box):
            method = "orthorhombic"
        elif r_cut < float(_box_widths(box).min()):
            method = "triclinic"
        else:
            warnings.warn(
                "neighbour cutoff is comparable to the box width; falling back "
                "to the O(N^2) neighbour search",
                stacklevel=2,
            )
            method = "bruteforce"

    if method == "orthorhombic":
        if not _is_orthorhombic(box):
            raise ValueError(
                "neighbor_method='orthorhombic' requires an orthorhombic box"
            )
        ci, cj, d = _pairs_orthorhombic(pos, box, r_cut)
    elif method == "triclinic":
        if r_cut >= float(_box_widths(box).min()):
            warnings.warn(
                "the 27-image triclinic search requires r_cut < the smallest "
                "box width; falling back to the O(N^2) search",
                stacklevel=2,
            )
            method = "bruteforce"
            ci, cj, d = _pairs_bruteforce(pos, box, r_cut)
        else:
            ci, cj, d = _pairs_triclinic(pos, box, r_cut)
    elif method == "bruteforce":
        ci, cj, d = _pairs_bruteforce(pos, box, r_cut)
    else:
        raise ValueError(
            "neighbor_method must be 'auto', 'orthorhombic', 'triclinic' "
            "or 'bruteforce'"
        )

    n = len(pos)
    inv4s2 = 1.0 / (4.0 * sigma * sigma)
    r2 = np.einsum("ij,ij->i", d, d)
    values = np.zeros((len(refs), n), dtype=np.float64)

    for m, ref in enumerate(refs):
        ref_norm2 = np.einsum("ij,ij->i", ref, ref)
        # The kernel is orientation-dependent: the environment of atom j sees
        # -d.  Compute both directions for every i<j pair.
        dot = d @ ref.T
        d2_forward = r2[:, None] + ref_norm2[None, :] - 2.0 * dot
        d2_backward = r2[:, None] + ref_norm2[None, :] + 2.0 * dot
        np.maximum(d2_forward, 0.0, out=d2_forward)
        np.maximum(d2_backward, 0.0, out=d2_backward)
        kernel_forward = np.exp(-d2_forward * inv4s2).sum(axis=1) * inv_norm[m]
        kernel_backward = np.exp(-d2_backward * inv4s2).sum(axis=1) * inv_norm[m]
        values[m] = np.bincount(ci, weights=kernel_forward, minlength=n)
        values[m] += np.bincount(cj, weights=kernel_backward, minlength=n)

    if len(refs) == 1:
        return values[0]

    vmax = values.max(axis=0)
    return vmax + np.log(np.exp(lam * (values - vmax)).sum(axis=0)) / lam


def more_than_count(
    values: np.ndarray, r0: float, nn: int = 8, mm: int = 16
) -> float:
    """PLUMED ``MORE_THAN={RATIONAL R_0=r0 NN=nn MM=mm}`` soft count.

    The switching function is

        s(r) = (1 - r^nn) / (1 - r^mm),  r < r_max
        s(r) = 0,                        r >= r_max

    with ``r = value / r0`` and ``r_max = (1e-5)**(1/(nn-mm))``.  The returned
    value is ``sum_i (1 - s(value_i / r0))``.  Note that ``s(1) = nn/mm = 0.5``,
    so a value exactly at ``r0`` contributes 0.5, and values above ``r0`` still
    contribute less than 1 (``s`` decays slowly up to ``r_max``).
    """
    if r0 is None or r0 <= 0.0:
        raise ValueError("r0 must be a positive number")
    if mm == 0:
        mm = 2 * nn
    if nn <= 0 or mm <= 0:
        raise ValueError("nn and mm must be positive (mm=0 means 2*nn)")
    if nn == mm:
        raise ValueError("nn and mm must be different")

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain non-finite entries")
    r_max = (1.0e-5) ** (1.0 / (nn - mm))
    r = values / r0
    s = np.zeros_like(r)
    mask = r < r_max
    rr = r[mask]
    near_one = np.abs(rr - 1.0) < 1.0e-12
    result = np.empty_like(rr)
    result[near_one] = nn / mm
    if np.any(~near_one):
        rr2 = rr[~near_one]
        result[~near_one] = (1.0 - rr2**nn) / (1.0 - rr2**mm)
    s[mask] = result
    return float(np.sum(1.0 - s))


class EnvsimReporter:
    """OpenMM reporter that writes the ENVIRONMENTSIMILARITY CV.

    Parameters
    ----------
    file : str, path-like or None
        Main output file.  If ``None``, the values are available as reporter
        attributes after each report.
    interval : int
        Report interval in MD steps.  This is user-controlled: use a small
        value (e.g. 100) while calibrating ``r0``, and a larger value (e.g.
        500-1000) in production.
    references : sequence of str/path-like
        Reference PDB files (displacement vectors in Angstrom).
    sigma : float
        PLUMED ``SIGMA`` in nm.  Default 0.05 nm = 0.5 Angstrom.
    cutoff : float
        PLUMED ``CUTOFF``: neighbour cutoff is
        ``max|r^0| + cutoff * sigma``.
    lam : float
        PLUMED ``LAMBDA`` soft-max parameter (default 100).
    r0 : float or None
        PLUMED ``MORE_THAN`` threshold ``R_0``.  ``None`` (default) runs in
        calibration mode: the reporter writes a per-atom ``env.N`` file so the
        user can inspect the distribution and choose ``r0``.
    nn, mm : int
        PLUMED ``MORE_THAN={RATIONAL ... NN=nn MM=mm}``.  ``mm=0`` means 2*nn.
    group : sequence of int or None
        Zero-based atom indices to use as centres and neighbours.  If ``None``,
        all oxygen atoms of water residues are selected automatically.
    water_names : sequence of str
        Residue names used by the automatic oxygen selection.
    per_atom_file : str, path-like or False
        Per-atom output file.  ``False`` (default): the summary file only when
        ``r0`` is given; in calibration mode (``r0=None``) the reporter always
        writes a per-atom file.  Pass a path to choose its location.
    append : bool
        Extend existing files; headers are written for new files.
    time_unit : openmm.unit.Unit
        Time unit of the output files.  Default ``nanosecond``.
    normalization : {'per_reference', 'plumed'}
        Normalization of multiple reference environments; see
        :func:`compute_environment_similarity`.
    """

    def __init__(
        self,
        file: str | os.PathLike | None,
        interval: int,
        references: Sequence[str | os.PathLike],
        sigma: float = 0.05,
        cutoff: float = 3.0,
        lam: float = 100.0,
        r0: float | None = None,
        nn: int = 8,
        mm: int = 16,
        group: Sequence[int] | None = None,
        water_names: Sequence[str] | str = _DEFAULT_WATER_NAMES,
        per_atom_file: str | os.PathLike | bool = False,
        append: bool = False,
        time_unit: Unit = nanosecond,
        normalization: str = "per_reference",
    ) -> None:
        if int(interval) <= 0:
            raise ValueError("interval must be a positive integer")
        if isinstance(references, (str, os.PathLike)):
            references = [references]
        references = list(references)
        if not references:
            raise ValueError("at least one reference PDB is required")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be a positive finite number")
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a non-negative finite number")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lam must be a positive finite number")
        if r0 is not None and (not np.isfinite(r0) or r0 <= 0.0):
            raise ValueError("r0 must be a positive finite number or None")
        if mm == 0:
            mm = 2 * nn
        if nn <= 0 or mm <= 0 or nn == mm:
            raise ValueError("nn and mm must be positive and different")
        if not isinstance(time_unit, Unit) or not time_unit.is_compatible(second):
            raise TypeError(
                "time_unit must be an openmm.unit time unit, e.g. nanosecond "
                "or picosecond"
            )
        if normalization not in ("per_reference", "plumed"):
            raise ValueError("normalization must be 'per_reference' or 'plumed'")

        self.interval = int(interval)
        self.sigma = float(sigma)
        self.cutoff = float(cutoff)
        self.lam = float(lam)
        self.r0 = None if r0 is None else float(r0)
        self.nn = int(nn)
        self.mm = int(mm)
        self.time_unit = time_unit
        self.normalization = normalization
        self.append = bool(append)

        self._refs = [read_reference_pdb(path) * _ANGSTROM_TO_NM for path in references]

        if group is None:
            self._group = None
        else:
            self._group = np.asarray(group, dtype=np.int64)
            if self._group.ndim != 1 or len(self._group) == 0:
                raise ValueError("group must be a non-empty 1-D sequence of atom indices")
            if np.any(self._group < 0):
                raise ValueError("group indices must be non-negative (0-based)")
        if isinstance(water_names, str):
            water_names = (water_names,)
        self.water_names = tuple(water_names)
        self._oxygen_names = _DEFAULT_OXYGEN_NAMES

        self._ids: np.ndarray | None = None
        self.env: np.ndarray | None = None
        self.env_mean: float | None = None
        self.n_ice: float | None = None

        # Per-atom output path.  Calibration mode (r0 is None) always writes
        # env.N; otherwise it is written only when a path is supplied.
        if per_atom_file is None:
            per_atom_file = False
        if per_atom_file is True:
            raise TypeError("per_atom_file must be a path or False")
        if self.r0 is None and per_atom_file is False:
            if file is not None:
                root, ext = os.path.splitext(os.fspath(file))
                atom_path = root + "_env" + (ext or ".dat")
            else:
                atom_path = "env_per_atom.dat"
        elif per_atom_file is False:
            atom_path = None
        else:
            atom_path = os.fspath(per_atom_file)

        if atom_path is not None and file is not None:
            if os.path.abspath(atom_path) == os.path.abspath(os.fspath(file)):
                raise ValueError("file and per_atom_file must be different paths")

        self._atom_path = atom_path
        self._out = None
        self._atom_out = None
        columns = [
            "#Step",
            f"Time({self.time_unit.get_symbol()})",
            "Temperature(K)",
            "env_mean",
        ]
        if self.r0 is not None:
            columns.append("n_ice")

        # Open the per-atom file first: if it fails, the summary file is not
        # truncated yet.  If the summary open fails, close the per-atom handle.
        try:
            if atom_path is not None:
                self._atom_out = open(atom_path, "a" if self.append else "w")
            if file is not None:
                file_path = os.fspath(file)
                self._out = open(file_path, "a" if self.append else "w")
                if not self.append or os.path.getsize(file_path) == 0:
                    self._out.write("\t".join(columns) + "\n")
                    self._out.flush()
        except Exception:
            if self._atom_out is not None:
                self._atom_out.close()
                self._atom_out = None
            if self._out is not None:
                self._out.close()
                self._out = None
            raise

    # ------------------------------------------------------------------
    # OpenMM reporter interface
    # ------------------------------------------------------------------
    def describeNextReport(self, simulation) -> dict:
        """Return the number of steps until the next report and required data."""
        steps = self.interval - simulation.currentStep % self.interval
        return {"steps": steps, "periodic": True, "include": ["positions"]}

    def report(self, simulation, state) -> None:
        """Compute and write the ENVIRONMENTSIMILARITY values for this frame."""
        if self._ids is None:
            self._ids = self._select_ids(simulation.topology)
            if self._atom_out is not None and (
                not self.append or os.path.getsize(self._atom_path) == 0
            ):
                header = "time " + " ".join(
                    f"env.{i + 1}" for i in range(len(self._ids))
                )
                self._atom_out.write(header + "\n")
                self._atom_out.flush()

        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(nanometer),
            dtype=np.float64,
        )[self._ids]
        box_vectors = np.asarray(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer),
            dtype=np.float64,
        )

        env = compute_environment_similarity(
            positions,
            box_vectors,
            self._refs,
            sigma=self.sigma,
            cutoff=self.cutoff,
            lam=self.lam,
            normalization=self.normalization,
        )

        self.env = env
        self.env_mean = float(env.mean())
        n_ice = None if self.r0 is None else more_than_count(env, self.r0, self.nn, self.mm)
        self.n_ice = n_ice

        time_out = state.getTime().value_in_unit(self.time_unit)
        temperature = self._get_temperature(simulation)

        if self._out is not None:
            parts = [
                str(simulation.currentStep),
                f"{time_out:.6f}",
                "\\" if temperature is None else f"{temperature:.2f}",
                f"{self.env_mean:.6f}",
            ]
            if n_ice is not None:
                parts.append(f"{n_ice:.6f}")
            self._out.write("\t".join(parts) + "\n")
            self._out.flush()

        if self._atom_out is not None:
            self._atom_out.write(
                f"{time_out:.6f} " + " ".join(f"{v:.6f}" for v in env) + "\n"
            )
            self._atom_out.flush()

    def close(self) -> None:
        """Flush and close all output files."""
        for attr in ("_out", "_atom_out"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _select_ids(self, topology) -> np.ndarray:
        if self._group is not None:
            ids = self._group.copy()
            n_atoms = topology.getNumAtoms()
            if np.any(ids >= n_atoms):
                raise ValueError("group contains atom indices outside the topology")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("group contains duplicate atom indices")
            return ids

        ids: list[int] = []
        for atom in topology.atoms():
            if atom.residue.name not in self.water_names:
                continue
            element = getattr(atom, "element", None)
            if element is not None:
                if element.symbol != "O":
                    continue
            elif atom.name not in self._oxygen_names:
                continue
            ids.append(atom.index)
        if not ids:
            raise ValueError(
                "no water oxygen atoms found; set group=[...] explicitly "
                "or adjust water_names"
            )
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _get_temperature(simulation) -> float | None:
        getter = getattr(simulation.integrator, "getTemperature", None)
        if getter is not None:
            try:
                return float(getter().value_in_unit(kelvin))
            except Exception:
                pass
        for name in simulation.context.getParameters():
            if "Temperature" in name:
                try:
                    return float(simulation.context.getParameter(name))
                except Exception:
                    break
        return None
