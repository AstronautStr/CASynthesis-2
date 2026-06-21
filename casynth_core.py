#!/usr/bin/env python3
"""
casynth_core.py — shared shape→spectrum mapping library for CASynth.

Single source of truth for the mapping contract

    map_*(patch: ndarray[PATCH_SIZE, PATCH_SIZE], f0: float, n: int)
        -> (freqs: ndarray[n], amps: ndarray[n])

    - patch : binary square window of ONE object's cells (extract()).
    - f0    : carrier frequency in Hz.
    - n     : number of partial slots to return (the CALLER supplies its own
              constant -- the listening bench uses K_MAX, the live Laplace
              prototype uses MAX_MODES_PER_OBJ).  Making n an explicit parameter
              keeps the algorithm single-sourced while letting each app pick how
              many partials it sounds; the divergence is visible at the call site
              instead of baked into two copies of the function.
    - freqs : partial frequencies in Hz; harmonic mappings return
              f0 * arange(1, n+1).  Laplacian returns inharmonic frequencies
              derived from sqrt(eigenvalue).
    - amps  : amplitudes in [0, 1], normalised.

This module is intentionally dependency-light (numpy + scipy only, NO pygame /
audio / UI) so it imports and unit-tests in isolation.  The audio ENGINE
(render_chunk / render_chunk_laplacian / SlotPool) is NOT here -- it diverged by
purpose between bench and prototype (see memory/decisions.md).

Extracted 1:1 from mapping_bench.py; the Laplacian path matches the (newer,
researcher-fixed) gol_life_synth_laplacian.py copy.  Both apps import from here.
"""

import numpy as np
from scipy import ndimage, linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian

# ── Shared constants ──────────────────────────────────────────────────────────
SR = 44100                 # audio sample rate (used only for the anti-alias guard)
PATCH_SIZE = 8             # shape extraction window (PATCH_SIZE×PATCH_SIZE)
N_PARTIALS_DEFAULT = 20    # default partial count (== gol_life_synth.K_MAX)

_GUARD = 0.45 * SR         # anti-alias guard frequency (Hz)


# ── Shape extraction ──────────────────────────────────────────────────────────

def extract(grid, size=PATCH_SIZE):
    """Return a size×size patch centered on the centroid of live cells.

    Bounds come from grid.shape, so this works for any input grid (the general
    form from the prototype; the bench passes a GRID_SZ×GRID_SZ grid and gets
    identical bounds).
    """
    live = np.argwhere(grid > 0)
    if len(live) == 0:
        return np.zeros((size, size), np.uint8)
    rc = live.mean(axis=0).round().astype(int)
    h = size // 2
    patch = np.zeros((size, size), np.uint8)
    rows, cols = grid.shape
    for dr in range(-h, h):
        for dc in range(-h, h):
            r, c = int(rc[0]) + dr, int(rc[1]) + dc
            if 0 <= r < rows and 0 <= c < cols:
                patch[dr + h, dc + h] = grid[r, c]
    return patch


# ── Precomputed structures (on PATCH_SIZE) ────────────────────────────────────
_W2 = PATCH_SIZE * PATCH_SIZE   # 64

# radial sort index for FFT (skip DC at index 0)
_ys8, _xs8 = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]
_r2_8 = (_ys8 - PATCH_SIZE // 2) ** 2 + (_xs8 - PATCH_SIZE // 2) ** 2
_fft_order = np.argsort(_r2_8.flatten())  # [0]=DC, [1..]=increasing freq

# Walsh-Hadamard PATCH_SIZE matrix for 2D transform: H8 @ patch @ H8.T
# Normalised so that H8 @ H8.T = I (orthonormal).
_H8 = linalg.hadamard(PATCH_SIZE).astype(float) / PATCH_SIZE


def _row_sequency(H):
    """Compute sequency (number of sign changes) for each row of H."""
    seq = np.zeros(H.shape[0], dtype=int)
    for i in range(H.shape[0]):
        row = H[i]
        seq[i] = int(np.sum(row[:-1] * row[1:] < 0))
    return seq


_SEQ8 = _row_sequency(_H8)
# 2D sequency for each (i,j) coefficient = _SEQ8[i] + _SEQ8[j]
_ij_seq = np.array([[_SEQ8[i] + _SEQ8[j] for j in range(PATCH_SIZE)]
                    for i in range(PATCH_SIZE)])
# Sort 2D coefficients by combined sequency (the "radial sort" analog).
# DC coefficient (i=0, j=0, seq=0) is first; skip it when indexing harmonics.
_walsh_order = np.argsort(_ij_seq.flatten(), kind='stable')  # index [0]=DC

# Fixed random projection matrix.  Built at N_PARTIALS_DEFAULT rows; map_random
# slices [:n].  numpy fills row-major, so the first n rows of this draw equal a
# fresh rng(42).standard_normal((n, _W2)) for any n <= N_PARTIALS_DEFAULT --
# output is unchanged from the previous (K_MAX, _W2) matrix.
_R_MAT = np.random.default_rng(42).standard_normal((N_PARTIALS_DEFAULT, _W2))


# ── Mapping functions ─────────────────────────────────────────────────────────

def _norm(v):
    mx = v.max()
    return v / (mx + 1e-9)


def map_fft2d(patch, f0, n=N_PARTIALS_DEFAULT):
    F = np.abs(np.fft.fftshift(np.fft.fft2(patch.astype(float))))
    flat = F.flatten()[_fft_order]
    amps = _norm(flat[1:n + 1])
    freqs = f0 * np.arange(1, n + 1)
    return freqs, amps


def map_walsh(patch, f0, n=N_PARTIALS_DEFAULT):
    """2D Walsh-Hadamard transform: C = H8 @ patch @ H8.T.
    Coefficients are sorted by combined 2D sequency (sum of row and column
    sequency), analogous to radial frequency ordering in FFT.
    Low sequency (coarse spatial patterns) -> low harmonics.
    DC coefficient (index 0) is skipped."""
    C = _H8 @ patch.astype(float) @ _H8.T    # 2D WHT, shape (8,8)
    flat = np.abs(C).flatten()[_walsh_order]  # sort by sequency
    amps = _norm(flat[1:n + 1])               # skip DC at index 0
    freqs = f0 * np.arange(1, n + 1)
    return freqs, amps


def map_random(patch, f0, n=N_PARTIALS_DEFAULT):
    v = np.abs(_R_MAT[:n] @ patch.flatten().astype(float))
    amps = _norm(v)
    freqs = f0 * np.arange(1, n + 1)
    return freqs, amps


def _select_modes(n_modes, n, spread):
    """Pick which of n_modes sorted modes feed the n partial slots.

    spread=0.0 -> the n LOWEST modes consecutively (indices 0..n-1) -- the
    historical behaviour.  spread=1.0 -> n modes decimated across the WHOLE
    spectrum (linspace 0..n_modes-1), reaching the bright upper resonances.
    Intermediate spread linearly interpolates between the two index sets.

    Index 0 is ALWAYS the first selected mode, so the lowest sounding partial
    stays normalised to f0 regardless of spread (pitch must not drift -- see
    decisions.md 2026-06-16).  Returns a strictly increasing int index array
    of length <= n, all entries < n_modes.
    """
    if n_modes <= n or spread <= 0.0:
        return np.arange(min(n, n_modes))
    consec = np.arange(n, dtype=float)               # spread = 0
    spread_idx = np.linspace(0.0, n_modes - 1, n)    # spread = 1
    raw = (1.0 - spread) * consec + spread * spread_idx
    sel = np.round(raw).astype(int)
    sel[0] = 0                                        # lowest stays = f0
    for i in range(1, n):                             # force strictly increasing
        if sel[i] <= sel[i - 1]:
            sel[i] = sel[i - 1] + 1
    return sel[sel < n_modes]


def map_laplacian(patch, f0, n=N_PARTIALS_DEFAULT, spread=0.0, alpha=1.0, shape=0.0,
                  harm=0.0):
    """Graph Laplacian eigenvalues -> inharmonic partial frequencies via sqrt(lambda).
    Lowest selected mode is normalized to f0; amplitudes follow 1/i**alpha rolloff.

    spread : 0.0 = n lowest modes (dark); 1.0 = modes decimated across the whole
             spectrum (bright upper resonances).  Lowest selected mode is always
             f0 (pitch anchored to the carrier -- see decisions.md 2026-06-16).
    alpha  : amplitude rolloff exponent 1/i**alpha.  alpha=1 -> historical 1/i;
             alpha->0 -> flat / brighter; alpha>1 -> steeper / darker.
    shape  : 0.0 = rolloff only (bit-for-bit historical); 1.0 = amplitudes from
             |<e, phi_k>| projection of edge-excitation onto each eigenvector.
             Linear blend at intermediate values; frequencies are never touched.
             (decisions.md 2026-06-18 -- Step B.)
    harm   : 0.0 = inharmonic metal (bit-for-bit); 1.0 = fully quantised to the
             nearest integer harmonic of f0.  Linear blend: each mode's frequency
             ratio r_k = freq_k/f0 is pulled toward round(r_k) by `harm`.
             The shape (which harmonics are occupied) comes from the Laplacian --
             NOT reassigned to 1..n.  Amplitudes are never touched.
             (decisions.md 2026-06-21.)

    Defaults (spread=0, alpha=1, shape=0, harm=0) reproduce the original output
    bit-for-bit."""
    live = list(map(tuple, np.argwhere(patch > 0)))
    cnt = len(live)
    if cnt < 2:
        return np.zeros(n), np.zeros(n)
    pos = {p: i for i, p in enumerate(live)}
    ri, ci_ = [], []
    for r, c in live:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nb = (r + dr, c + dc)
            if nb in pos:
                ri.append(pos[(r, c)])
                ci_.append(pos[nb])
    if not ri:
        return np.zeros(n), np.zeros(n)
    A = csr_matrix((np.ones(len(ri)), (ri, ci_)), shape=(cnt, cnt))
    L = sparse_laplacian(A).toarray().astype(float)

    # sqrt(lambda) is proportional to resonant mode frequency (membrane analogy).
    # shape>0 needs eigenvectors; shape==0 uses the faster eigvalsh-only path.
    if shape > 0.0:
        eigs, vecs = np.linalg.eigh(L)
        nonzero_mask = eigs > 1e-6
        nonzero_idx = np.where(nonzero_mask)[0]
        nonzero_sq = np.sqrt(np.maximum(eigs[nonzero_mask], 0.0))
    else:
        eigs = np.linalg.eigvalsh(L)
        nonzero_sq = np.sqrt(np.maximum(eigs[eigs > 1e-6], 0.0))

    if len(nonzero_sq) == 0:
        return np.zeros(n), np.zeros(n)

    # Normalize: lowest mode -> f0; others proportionally higher
    scale = f0 / nonzero_sq[0]
    mode_freqs = nonzero_sq * scale

    # Anti-alias guard -- track column indices when eigenvectors are needed
    if shape > 0.0:
        guard_mask = mode_freqs < _GUARD
        survived_idx = nonzero_idx[guard_mask]
        mode_freqs = mode_freqs[guard_mask]
    else:
        mode_freqs = mode_freqs[mode_freqs < _GUARD]

    if len(mode_freqs) == 0:
        return np.zeros(n), np.zeros(n)

    # Choose which modes sound (spread spans low->whole-spectrum); index 0 stays f0
    sel = _select_modes(len(mode_freqs), n, spread)
    num = len(sel)
    freqs = np.zeros(n)
    freqs[:num] = mode_freqs[sel]

    # Harmonic quantisation: pull frequency ratios toward nearest integer multiple
    # of f0.  harm=0 -> no change (bit-for-bit); harm=1 -> fully quantised.
    # Only the ratios are blended; r_0 = 1 always -> round(1) = 1 -> f0 preserved.
    if harm > 0.0 and num > 0:
        r = freqs[:num] / f0
        freqs[:num] = f0 * ((1.0 - harm) * r + harm * np.round(r))

    # Amplitudes
    amps = np.zeros(n)
    if num > 0:
        rolloff = 1.0 / np.arange(1, num + 1) ** alpha
        rolloff /= rolloff.max()   # rolloff[0] = 1.0 always

        if shape > 0.0:
            # Edge excitation e_i = deg_i = diagonal of L (graph-intrinsic,
            # invariant to position/rotation/reflection -- decisions.md 2026-06-18).
            e = L.diagonal()
            # Columns of vecs that survived both masks and the mode selection
            final_idx = survived_idx[sel]
            proj = np.array([abs(float(np.dot(e, vecs[:, j]))) for j in final_idx])
            mx_proj = proj.max()
            if mx_proj > 1e-9:
                proj /= mx_proj
            else:
                proj = rolloff.copy()   # regular graph: all projections zero -> fallback
            blended = (1.0 - shape) * rolloff + shape * proj
            mx_blend = blended.max()
            if mx_blend > 1e-9:
                blended /= mx_blend
            amps[:num] = blended
        else:
            amps[:num] = rolloff
    return freqs, amps


def map_granulo(patch, f0, n=N_PARTIALS_DEFAULT):
    """Granulometry: morphological opening at increasing radii measures energy at each scale.
    Large scale (coarse structure) -> low harmonics; small scale (fine detail) -> high harmonics.
    Reversed so that jagged shapes sound brighter."""
    amps = np.zeros(n)
    prev = float(patch.sum())
    if prev == 0:
        return f0 * np.arange(1, n + 1), amps
    for k in range(n):
        radius = k + 1
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        selem = (y * y + x * x <= radius * radius)
        opened = ndimage.binary_opening(patch.astype(bool), structure=selem)
        cur = float(opened.sum())
        amps[k] = prev - cur   # energy at scale k+1
        if cur < 1:
            break
        prev = cur
    amps = _norm(amps)
    amps = amps[::-1].copy()  # small scale (index 0) -> high harmonic; large scale -> low harmonic
    freqs = f0 * np.arange(1, n + 1)
    return freqs, amps


# ── Engine registry ───────────────────────────────────────────────────────────
# Single source of truth for the set of sound engines and their tunable
# attributes, shared by any app that offers an engine selector (gol_synth.py).
# Pure data (no pygame / no UI formatting) so it stays in the dependency-light
# library; the UI derives sliders + display formatting from these specs.
#
# Each engine: id, human label, the map_* function (contract form -> (freqs,amps)),
# and `params` = its tunable ENGINE attributes (NOT synth-wide knobs like release).
# Param spec tuple: (arg, label, lo, hi, integer, default)
#   arg     : keyword name passed to fn -- 'n' (partial count), 'spread', 'alpha'.
#   integer : True -> value snapped/stored as int (partial count).
#   default : starting value; an app remembers per-engine values across switches.
# The harmonic mappings expose only the partial count; Laplacian adds spread/alpha.
ENGINES = [
    dict(id='laplacian', label='Laplace', fn=map_laplacian, params=[
        ('n',      'part',   1,   20,  True,  12),
        ('spread', 'spread', 0.0, 1.0, False, 0.0),
        ('alpha',  'alpha',  0.0, 2.0, False, 1.0),
        ('shape',  'shape',  0.0, 1.0, False, 0.0),
        ('harm',   'harm',   0.0, 1.0, False, 0.0)]),
    dict(id='fft2d',   label='FFT',     fn=map_fft2d,   params=[('n', 'part', 1, 20, True, 16)]),
    dict(id='walsh',   label='Walsh',   fn=map_walsh,   params=[('n', 'part', 1, 20, True, 16)]),
    dict(id='random',  label='Random',  fn=map_random,  params=[('n', 'part', 1, 20, True, 16)]),
    dict(id='granulo', label='Granulo', fn=map_granulo, params=[('n', 'part', 1, 20, True, 16)]),
]

ENGINE_BY_ID = {e['id']: e for e in ENGINES}
