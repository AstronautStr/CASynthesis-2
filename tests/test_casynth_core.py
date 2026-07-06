#!/usr/bin/env python3
"""
Unit tests for casynth_core — the shared shape→spectrum mapping library.

These lock the DESIGN INTENT of the mappings (memory/decisions.md,
memory/questions.md P0-2) so a refactor can't silently change a mapping's
meaning before anyone listens.  Pure numpy/scipy — no pygame / audio.

Run either way (no test-framework dependency required):
    python tests/test_casynth_core.py     # plain assert runner
    pytest tests/                          # auto-discovers test_* functions
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casynth_core as c
import casynth_engine as eng
from casynth_engine import _crop_like_extract
from casynth_session import _exc_for_frames
from casynth_tuning import (pair_dissonance, dissonance_curve,
                             scale_minima, snap_ratio,
                             TUNE_CURVE_STEPS, TUNE_SKIP_BELOW)

F0 = 261.0
GUARD = 0.45 * c.SR
ALL_MAPPINGS = [c.map_fft2d, c.map_walsh, c.map_random,
                c.map_laplacian, c.map_granulo]


def _blinker_patch():
    """3-in-a-row horizontal (path graph of 3 nodes), centred with margin."""
    p = np.zeros((8, 8), np.uint8)
    p[3, 2] = p[3, 3] = p[3, 4] = 1
    return p


def _ell_patch():
    """An asymmetric L so rotation/reflection are non-trivial."""
    p = np.zeros((8, 8), np.uint8)
    p[2, 2] = p[3, 2] = p[4, 2] = p[4, 3] = p[4, 4] = 1
    return p


def _dense_patch():
    """A solid block -> many Laplacian modes (>> 8), so spread/truncation differ."""
    return np.ones((8, 8), np.uint8)


def _tight_small_ell():
    """Tight (3×3) bbox of an L-shape; fits well within 8×8."""
    p = np.zeros((3, 3), np.uint8)
    p[0, 0] = p[1, 0] = p[2, 0] = p[2, 1] = p[2, 2] = 1
    return p


def _large_ell_patch():
    """L-shape in a 12×12 tight bbox; both arms extend beyond 8×8."""
    p = np.zeros((12, 12), np.uint8)
    p[:, 0] = 1    # vertical arm, col 0, rows 0-11
    p[11, :] = 1   # horizontal arm, row 11, cols 0-11
    return p


def _huge_dense():
    """Dense 20×20 block = 400 live cells > MAX_LAPLACIAN_NODES=256."""
    return np.ones((20, 20), np.uint8)


# ── Contract (all mappings) ───────────────────────────────────────────────────

def test_contract_shapes_and_ranges():
    """Every map_* returns (freqs, amps) of length n; amps in [0,1]; freqs in
    [0, guard)."""
    for n in (8, 20):
        for mfn in ALL_MAPPINGS:
            freqs, amps = mfn(_ell_patch(), F0, n)
            assert freqs.shape == (n,), f"{mfn.__name__} freqs len {freqs.shape} != {n}"
            assert amps.shape == (n,), f"{mfn.__name__} amps len {amps.shape} != {n}"
            assert np.all(amps >= -1e-9) and np.all(amps <= 1.0 + 1e-9), \
                f"{mfn.__name__} amps out of [0,1]: {amps}"
            nz = freqs[freqs > 0]
            assert np.all(nz < GUARD), f"{mfn.__name__} freq above anti-alias guard"
            assert np.all(freqs >= 0), f"{mfn.__name__} negative freq"


def test_empty_patch_is_silent():
    """A patch with <2 live cells yields all-zero spectra (no crash, no sound)."""
    empty = np.zeros((8, 8), np.uint8)
    for mfn in ALL_MAPPINGS:
        freqs, amps = mfn(empty, F0, 8)
        assert np.all(amps == 0.0), f"{mfn.__name__} not silent on empty patch"


# ── Laplacian design invariants (questions.md P0-2) ───────────────────────────

def test_laplacian_lowest_mode_is_f0():
    """Lowest non-zero mode normalised to carrier f0 (pitch anchor)."""
    freqs, _ = c.map_laplacian(_ell_patch(), F0, 8)
    nz = np.sort(freqs[freqs > 0])
    assert abs(nz[0] - F0) < 1e-6, f"lowest mode {nz[0]} != f0 {F0}"


def test_laplacian_uses_sqrt_lambda_not_lambda():
    """Blinker = path graph P3, Laplacian eigenvalues {0,1,3}.  Mode freqs use
    sqrt(lambda): second mode = sqrt(3)*f0 (~452 Hz), NOT 3*f0 (~783 Hz)."""
    freqs, _ = c.map_laplacian(_blinker_patch(), F0, 8)
    nz = np.sort(freqs[freqs > 0])
    assert len(nz) == 2, f"expected 2 modes for P3, got {len(nz)}"
    assert abs(nz[0] - F0) < 1e-6
    assert abs(nz[1] - np.sqrt(3.0) * F0) < 1e-3, \
        f"second mode {nz[1]} != sqrt(3)*f0 {np.sqrt(3.0)*F0} (lambda not sqrt'd?)"


def test_laplacian_invariant_to_rotation_reflection_translation():
    """Graph of a shape is isomorphic under rotation/reflection/translation ->
    identical (sorted) mode frequency set."""
    base = _ell_patch()
    fb, _ = c.map_laplacian(base, F0, 8)
    fb = np.sort(fb)
    for name, q in [("rot90", np.rot90(base)),
                    ("rot180", np.rot90(base, 2)),
                    ("fliplr", np.fliplr(base)),
                    ("flipud", np.flipud(base)),
                    ("roll", np.roll(np.roll(base, 1, 0), -1, 1))]:
        fq = np.sort(c.map_laplacian(np.ascontiguousarray(q), F0, 8)[0])
        assert np.allclose(fb, fq, atol=1e-6), \
            f"laplacian not invariant under {name}"


def test_laplacian_amps_nonincreasing_rolloff():
    """Amplitudes follow a 1/i rolloff (not growing toward high modes)."""
    _, amps = c.map_laplacian(_ell_patch(), F0, 8)
    nz = amps[amps > 0]
    assert np.all(np.diff(nz) <= 1e-9), f"laplacian amps not non-increasing: {nz}"


# ── Live knobs: spread / alpha (decisions.md 2026-06-16) ──────────────────────

def test_laplacian_knob_defaults_backward_compatible():
    """Explicit spread=0, alpha=1 == the historical (default) call -- so the
    listening bench, which omits the knobs, is bit-for-bit unchanged."""
    for p in (_blinker_patch(), _ell_patch(), _dense_patch()):
        f_def, a_def = c.map_laplacian(p, F0, 8)
        f_exp, a_exp = c.map_laplacian(p, F0, 8, spread=0.0, alpha=1.0)
        assert np.array_equal(f_def, f_exp) and np.array_equal(a_def, a_exp), \
            "spread=0/alpha=1 not identical to default"


def test_laplacian_n_is_truncation_only():
    """Lower n only truncates the tail: with spread=0 the first 8 freqs at n=8
    equal the first 8 at n=20 (locks the 20-vs-8 decision -- same algorithm,
    different tail length)."""
    p = _dense_patch()
    f8 = c.map_laplacian(p, F0, 8)[0]
    f20 = c.map_laplacian(p, F0, 20)[0]
    assert np.allclose(f8, f20[:8], atol=1e-9), "n changed the spectrum, not just length"


def test_laplacian_spread_keeps_lowest_at_f0_and_reaches_higher():
    """Spread must not move the lowest sounding mode off f0 (pitch anchor), but
    must reach higher resonances than spread=0 on a many-moded shape."""
    p = _dense_patch()
    f_lo = c.map_laplacian(p, F0, 8, spread=0.0)[0]
    f_hi = c.map_laplacian(p, F0, 8, spread=1.0)[0]
    lo_lo = np.sort(f_lo[f_lo > 0])[0]
    lo_hi = np.sort(f_hi[f_hi > 0])[0]
    assert abs(lo_lo - F0) < 1e-6 and abs(lo_hi - F0) < 1e-6, "spread moved lowest mode"
    assert f_hi.max() > f_lo.max() + 1e-6, "spread=1 did not reach higher modes"


def test_laplacian_alpha_controls_brightness():
    """alpha=0 -> flat amps (bright); larger alpha -> steeper rolloff (darker),
    without changing the frequencies."""
    p = _dense_patch()
    f1, a_flat = c.map_laplacian(p, F0, 8, alpha=0.0)
    f2, a_mid = c.map_laplacian(p, F0, 8, alpha=1.0)
    f3, a_steep = c.map_laplacian(p, F0, 8, alpha=2.0)
    assert np.allclose(f1, f2) and np.allclose(f2, f3), "alpha changed frequencies"
    nz_flat = a_flat[a_flat > 0]
    assert np.allclose(nz_flat, 1.0), f"alpha=0 not flat: {nz_flat}"
    # steeper alpha -> smaller amplitude on the 2nd partial relative to the 1st
    assert a_steep[1] < a_mid[1] < a_flat[1], "alpha rolloff not monotonic in steepness"


# ── Step B: shape parameter (decisions.md 2026-06-18) ────────────────────────

def test_laplacian_shape_does_not_affect_freqs():
    """shape never changes any partial frequency (criterion B).
    At any shape value freqs must be identical to the shape=0 call with same
    spread so that pitch is never disturbed."""
    for p in (_blinker_patch(), _ell_patch(), _dense_patch()):
        f0_res, _ = c.map_laplacian(p, F0, 8, shape=0.0)
        f1_res, _ = c.map_laplacian(p, F0, 8, shape=1.0)
        assert np.allclose(f0_res, f1_res, atol=1e-6), \
            f"shape changed frequencies: {f0_res} vs {f1_res}"


def test_laplacian_shape_blinker_exact_amplitudes():
    """Criterion C: index-alignment check on blinker (P3, eigenvalues 0,1,3).
    Edge excitation e=[1,2,1] is symmetric; lambda=1 eigenvector [1,0,-1] is
    anti-symmetric -> projection = 0; lambda=3 eigenvector [1,-2,1] -> 2/sqrt(6).
    At shape=1: amps[:2] must be [0.0, 1.0] (not [1.0, 0.0] -- wrong column order
    would satisfy weaker checks but swap the physical meaning).
    Zeros are correct for symmetric shapes (decisions.md 2026-06-18, questions.md C)."""
    freqs, amps = c.map_laplacian(_blinker_patch(), F0, 8, shape=1.0)
    assert np.all(np.isfinite(amps)), f"non-finite amps at shape=1: {amps}"
    assert not np.any(np.isnan(amps)), f"NaN in amps at shape=1: {amps}"
    nz_freqs = freqs[freqs > 0]
    assert len(nz_freqs) == 2, f"blinker should have 2 mode freqs, got {nz_freqs}"
    assert abs(amps.max() - 1.0) < 1e-9, f"amps not normalised to max=1: {amps}"
    assert np.allclose(amps[:2], [0.0, 1.0], atol=1e-6), \
        f"amps[:2] wrong (index mis-alignment?): {amps[:2]}, expected [0.0, 1.0]"


def test_laplacian_shape_amplitude_invariant_to_rotation():
    """Criterion D: at shape=1 the sorted amplitude vector is invariant under
    rotation/reflection/translation of an asymmetric shape (L-patch).
    Edge excitation e_i=deg_i is graph-intrinsic -> |<e,phi_k>| transforms as a
    scalar under isomorphism of the underlying graph."""
    base = _ell_patch()
    _, a_base = c.map_laplacian(base, F0, 8, shape=1.0)
    a_sorted = np.sort(a_base)
    for name, q in [("rot90",  np.rot90(base)),
                    ("rot180", np.rot90(base, 2)),
                    ("fliplr", np.fliplr(base)),
                    ("flipud", np.flipud(base)),
                    ("roll",   np.roll(np.roll(base, 1, 0), -1, 1))]:
        _, aq = c.map_laplacian(np.ascontiguousarray(q), F0, 8, shape=1.0)
        assert np.allclose(a_sorted, np.sort(aq), atol=1e-6), \
            f"shape=1 amps not invariant under {name}: {np.sort(aq)} vs {a_sorted}"


# ── Step C: harm parameter (decisions.md 2026-06-21) ─────────────────────────

def test_laplacian_harm_zero_is_bitexact():
    """harm=0 (default) must be bit-for-bit identical to the baseline call."""
    for p in (_blinker_patch(), _ell_patch(), _dense_patch()):
        f_base, a_base = c.map_laplacian(p, F0, 8)
        f_harm, a_harm = c.map_laplacian(p, F0, 8, harm=0.0)
        assert np.array_equal(f_base, f_harm) and np.array_equal(a_base, a_harm), \
            "harm=0 not bit-for-bit with baseline"


def test_laplacian_harm_one_integer_multiples():
    """harm=1 -> every non-zero frequency must be an integer multiple of f0."""
    for p in (_ell_patch(), _dense_patch()):
        freqs, _ = c.map_laplacian(p, F0, 8, harm=1.0)
        nz = freqs[freqs > 0]
        ratios = nz / F0
        assert np.allclose(ratios, np.round(ratios), atol=1e-6), \
            f"harm=1 produced non-integer ratios: {ratios}"


def test_laplacian_harm_lowest_mode_stays_f0():
    """harm never moves the lowest sounding mode off f0 (r_0=1, round(1)=1)."""
    for p in (_ell_patch(), _dense_patch()):
        for h in (0.0, 0.5, 1.0):
            freqs, _ = c.map_laplacian(p, F0, 8, harm=h)
            nz = np.sort(freqs[freqs > 0])
            assert abs(nz[0] - F0) < 1e-6, \
                f"harm={h} moved lowest mode to {nz[0]}, expected {F0}"


def test_laplacian_harm_does_not_affect_amps():
    """harm only touches frequencies; amplitudes must be identical to harm=0."""
    for p in (_ell_patch(), _dense_patch()):
        _, a_base = c.map_laplacian(p, F0, 8, harm=0.0)
        for h in (0.5, 1.0):
            _, a_h = c.map_laplacian(p, F0, 8, harm=h)
            assert np.allclose(a_base, a_h, atol=1e-9), \
                f"harm={h} changed amplitudes: {a_h} vs {a_base}"


def test_laplacian_harm_blend_monotonic():
    """Intermediate harm values move each mode ratio toward round(r) monotonically:
    |r(harm) - round(r)| decreases as harm increases (0 -> 0.5 -> 1)."""
    p = _ell_patch()
    f0_r, _ = c.map_laplacian(p, F0, 8, harm=0.0)
    f05_r, _ = c.map_laplacian(p, F0, 8, harm=0.5)
    f1_r, _ = c.map_laplacian(p, F0, 8, harm=1.0)
    nz = f0_r > 0
    r0 = f0_r[nz] / F0
    r05 = f05_r[nz] / F0
    r1 = f1_r[nz] / F0
    rnd = np.round(r0)
    dist0 = np.abs(r0 - rnd)
    dist05 = np.abs(r05 - rnd)
    dist1 = np.abs(r1 - rnd)
    assert np.all(dist05 <= dist0 + 1e-9), "harm=0.5 moved ratios further from integer"
    assert np.all(dist1 <= dist05 + 1e-9), "harm=1.0 moved ratios further than harm=0.5"


# ── fullshape mode (decisions.md 2026-06-22) ─────────────────────────────────

def test_laplacian_fullshape_false_is_bitexact():
    """A: fullshape=False (default) is bit-for-bit with the baseline call on all
    existing fixtures -- no regression."""
    for p in (_blinker_patch(), _ell_patch(), _dense_patch()):
        f_base, a_base = c.map_laplacian(p, F0, 8)
        f_fs, a_fs = c.map_laplacian(p, F0, 8, fullshape=False)
        assert np.array_equal(f_base, f_fs) and np.array_equal(a_base, a_fs), \
            "fullshape=False not bit-for-bit with default"


def test_laplacian_fullshape_small_matches_crop():
    """B: for a shape that fits within 8×8 the full-shape path yields the same
    sorted spectrum as the historical extract+crop path.
    Proves full-shape is 'stop cropping', not a new algorithm."""
    for tight in (_tight_small_ell(), _blinker_patch()[:1, 2:5]):
        crop = c.extract(tight, 8)
        f_crop = np.sort(c.map_laplacian(crop, F0, 8)[0])
        f_full = np.sort(c.map_laplacian(tight, F0, 8, fullshape=True)[0])
        assert np.allclose(f_crop, f_full, atol=1e-6), \
            f"fullshape on small shape differs from crop: {f_full} vs {f_crop}"


def test_laplacian_fullshape_large_differs_from_crop():
    """C: for a shape larger than 8×8 the full-shape spectrum differs from the
    8×8 crop (proving the periphery is now audible)."""
    p = _large_ell_patch()
    crop = c.extract(p, 8)
    f_crop = np.sort(c.map_laplacian(crop, F0, 8)[0])
    f_full = np.sort(c.map_laplacian(p, F0, 8, fullshape=True)[0])
    assert not np.allclose(f_crop, f_full, atol=1e-6), \
        "fullshape on large shape should differ from 8×8 crop but did not"


def test_laplacian_fullshape_invariant_to_rotation():
    """D: sorted spectrum is invariant to rot/flip/zero-padding of the mask."""
    p = _large_ell_patch()
    f_base = np.sort(c.map_laplacian(p, F0, 8, fullshape=True)[0])
    for name, q in [("rot90",  np.rot90(p)),
                    ("rot180", np.rot90(p, 2)),
                    ("fliplr", np.fliplr(p)),
                    ("flipud", np.flipud(p))]:
        fq = np.sort(c.map_laplacian(np.ascontiguousarray(q), F0, 8, fullshape=True)[0])
        assert np.allclose(f_base, fq, atol=1e-6), \
            f"fullshape not invariant under {name}"
    padded = np.pad(p, 2, mode='constant')
    fp = np.sort(c.map_laplacian(padded, F0, 8, fullshape=True)[0])
    assert np.allclose(f_base, fp, atol=1e-6), "fullshape not invariant under zero-padding"


def test_laplacian_fullshape_f0_anchor():
    """E: lowest sounding partial = f0 in full-shape mode at any harm value."""
    for p in (_tight_small_ell(), _large_ell_patch()):
        for h in (0.0, 0.5, 1.0):
            freqs, _ = c.map_laplacian(p, F0, 8, fullshape=True, harm=h)
            nz = np.sort(freqs[freqs > 0])
            assert len(nz) > 0, "fullshape produced silent output"
            assert abs(nz[0] - F0) < 1e-6, \
                f"fullshape f0 anchor failed at harm={h}: {nz[0]} != {F0}"


def test_laplacian_fullshape_node_ceiling():
    """F: dense 20×20 patch (400 nodes > MAX=256) does not crash and is
    deterministic -- the node ceiling is enforced and reproducible."""
    p = _huge_dense()
    f1, a1 = c.map_laplacian(p, F0, 8, fullshape=True)
    f2, a2 = c.map_laplacian(p, F0, 8, fullshape=True)
    assert np.array_equal(f1, f2) and np.array_equal(a1, a2), \
        "fullshape ceiling not deterministic"
    assert np.all(np.isfinite(f1)) and np.all(np.isfinite(a1)), \
        "fullshape ceiling produced non-finite output"
    assert np.all(f1[f1 > 0] < GUARD), "fullshape ceiling freq above anti-alias guard"


# ── FFT / Random characteristics ──────────────────────────────────────────────

def test_fft_translation_invariant():
    """|FFT| is invariant under circular translation of the shape."""
    p = _ell_patch()
    a0 = c.map_fft2d(p, F0, 20)[1]
    a1 = c.map_fft2d(np.roll(np.roll(p, 2, 0), 3, 1), F0, 20)[1]
    assert np.allclose(a0, a1, atol=1e-9), "map_fft2d not translation-invariant"


def test_random_deterministic():
    """Fixed-seed random projection is reproducible call-to-call."""
    p = _ell_patch()
    a0 = c.map_random(p, F0, 20)[1]
    a1 = c.map_random(p, F0, 20)[1]
    assert np.array_equal(a0, a1), "map_random not deterministic"


def test_random_slice_consistency():
    """map_random(n=8) equals the first 8 rows' result of map_random(n=20)
    (the projection matrix is sliced, not re-seeded)."""
    p = _ell_patch()
    a20 = c.map_random(p, F0, 20)
    a8 = c.map_random(p, F0, 8)
    # Same projection rows -> same raw magnitudes; normalisation differs only if
    # the max falls outside the first 8 rows, so compare the raw dot products.
    raw20 = np.abs(c._R_MAT[:20] @ p.flatten().astype(float))
    raw8 = np.abs(c._R_MAT[:8] @ p.flatten().astype(float))
    assert np.array_equal(raw20[:8], raw8), "random projection rows not consistent"


# ── Engine registry (gol_synth.py selector) ──────────────────────────────────

def test_engine_registry_integrity():
    """Every ENGINES entry has a unique id, a callable map_*, and param specs
    whose default sits inside [lo, hi] -- so the gol_synth selector can build a
    knob from each spec and call the engine with its defaults without surprises."""
    ids = [e['id'] for e in c.ENGINES]
    assert len(ids) == len(set(ids)), f"duplicate engine ids: {ids}"
    assert c.ENGINE_BY_ID == {e['id']: e for e in c.ENGINES}, "ENGINE_BY_ID out of sync"
    for e in c.ENGINES:
        assert callable(e['fn']), f"{e['id']} fn not callable"
        args = [p[0] for p in e['params']]
        assert 'n' in args, f"{e['id']} missing partial-count param 'n'"
        for (arg, label, lo, hi, integer, default) in e['params']:
            assert lo <= default <= hi, f"{e['id']}.{arg} default {default} outside [{lo},{hi}]"
            assert isinstance(label, str) and label, f"{e['id']}.{arg} bad label"


def test_engine_registry_call_with_defaults():
    """Calling each engine with its default params yields a finite (freqs, amps)
    pair of equal length n -- locks the registry against signature drift."""
    for e in c.ENGINES:
        kwargs = {p[0]: p[5] for p in e['params']}
        freqs, amps = e['fn'](_ell_patch(), F0, **kwargs)
        n = kwargs['n']
        assert freqs.shape == (n,) and amps.shape == (n,), \
            f"{e['id']} returned wrong length at defaults"
        assert np.all(np.isfinite(freqs)) and np.all(np.isfinite(amps)), \
            f"{e['id']} produced non-finite output"


# ── extract() ─────────────────────────────────────────────────────────────────

def test_extract_size_and_centering():
    """extract returns a size×size patch with the shape's mass near the centre."""
    g = np.zeros((20, 20), np.uint8)
    g[10, 10] = g[10, 11] = g[11, 10] = 1
    patch = c.extract(g, 8)
    assert patch.shape == (8, 8)
    assert patch.sum() == 3, "extract lost/added live cells"
    ys, xs = np.where(patch > 0)
    # centroid should sit near the patch centre (window centred on mass centroid)
    assert 2 <= ys.mean() <= 5 and 2 <= xs.mean() <= 5, "extract not centred"


# ── events_field (casynth_engine, decisions.md 2026-07-05) ───────────────────

def _blinker_prev():
    """3×3 blinker: horizontal phase {(1,0),(1,1),(1,2)}."""
    g = np.zeros((3, 3), np.uint8)
    g[1, 0] = g[1, 1] = g[1, 2] = 1
    return g


def _blinker_new():
    """3×3 blinker: vertical phase {(0,1),(1,1),(2,1)}."""
    g = np.zeros((3, 3), np.uint8)
    g[0, 1] = g[1, 1] = g[2, 1] = 1
    return g


def test_events_field_blinker():
    """Criterion F: exact values for the blinker horizontal→vertical transition.

    births {(0,1),(2,1)} → 1.0; wound edge from deaths {(1,0),(1,2)} covers all
    three live cells → each gets W_WOUND=0.7.
    Expected: (0,1)=1.7, (1,1)=0.7, (2,1)=1.7  (sums by linear superposition).
    """
    prev = _blinker_prev()
    new  = _blinker_new()
    exc  = eng.events_field(prev, new)

    assert exc.shape == (3, 3), f"wrong shape {exc.shape}"
    assert abs(exc[0, 1] - 1.7) < 1e-9, f"(0,1) should be 1.7, got {exc[0,1]}"
    assert abs(exc[1, 1] - 0.7) < 1e-9, f"(1,1) should be 0.7, got {exc[1,1]}"
    assert abs(exc[2, 1] - 1.7) < 1e-9, f"(2,1) should be 1.7, got {exc[2,1]}"
    # Cells with no events should be 0
    assert abs(exc[0, 0]) < 1e-9 and abs(exc[0, 2]) < 1e-9, "non-event cells non-zero"
    assert abs(exc[2, 0]) < 1e-9 and abs(exc[2, 2]) < 1e-9, "non-event cells non-zero"


def test_events_field_empty():
    """No births and no deaths -> all-zero output; no crash."""
    g = _blinker_prev()
    exc = eng.events_field(g, g)
    assert np.all(exc == 0.0), "static field should produce zero exc"


def test_events_field_birth_only():
    """Only births (prev empty) -> birth cells = 1.0, no wound."""
    prev = np.zeros((5, 5), np.uint8)
    new  = np.zeros((5, 5), np.uint8)
    new[2, 2] = 1
    exc = eng.events_field(prev, new)
    assert abs(exc[2, 2] - 1.0) < 1e-9
    assert exc.sum() == 1.0  # only that cell


def test_events_field_wound_only():
    """Only deaths: no births -> wound cells = W_WOUND, death cells=0 (not alive)."""
    prev = np.zeros((5, 5), np.uint8)
    prev[2, 2] = 1
    new  = np.zeros((5, 5), np.uint8)
    exc = eng.events_field(prev, new)
    assert abs(exc[2, 2]) < 1e-9, "dead cell should not be wound (not live-in-new)"
    assert exc.sum() < 1e-9, "no live cells in new -> no wound cells possible"


# ── dyn parameter (decisions.md 2026-07-05) ───────────────────────────────────

def _make_exc_for(patch, scale=1.0):
    """Create a simple non-zero exc array aligned with patch (arbitrary values)."""
    exc = np.zeros_like(patch, dtype=float)
    live = np.argwhere(patch > 0)
    for k, (r, c) in enumerate(live):
        exc[r, c] = (k + 1) * scale   # 1,2,3,... at live cells
    return exc


def test_laplacian_dyn_zero_bitexact():
    """Criterion A: dyn=0 (with or without exc) is bit-for-bit with the baseline."""
    for p in (_ell_patch(), _dense_patch()):
        f_base, a_base = c.map_laplacian(p, F0, 8, shape=0.5)
        exc = _make_exc_for(p)
        # Explicit dyn=0, exc provided -> must not change output
        f_dyn0, a_dyn0 = c.map_laplacian(p, F0, 8, shape=0.5, dyn=0.0, exc=exc)
        assert np.array_equal(f_base, f_dyn0) and np.array_equal(a_base, a_dyn0), \
            "dyn=0 with exc is not bit-for-bit with baseline"
        # Also: dyn=0 without exc
        f_dyn0b, a_dyn0b = c.map_laplacian(p, F0, 8, shape=0.5, dyn=0.0)
        assert np.array_equal(f_base, f_dyn0b) and np.array_equal(a_base, a_dyn0b), \
            "dyn=0 no-exc is not bit-for-bit with baseline"


def test_laplacian_dyn_fallback_no_exc():
    """Criterion B: dyn=1 with exc=None -> same amps as dyn=0 (deg path, atol 1e-9)."""
    for p in (_ell_patch(), _dense_patch()):
        _, a_deg  = c.map_laplacian(p, F0, 8, shape=0.7, dyn=0.0)
        _, a_fall = c.map_laplacian(p, F0, 8, shape=0.7, dyn=1.0, exc=None)
        assert np.allclose(a_deg, a_fall, atol=1e-9), \
            f"dyn=1 exc=None not equal to deg path: max diff {np.abs(a_deg-a_fall).max()}"


def test_laplacian_dyn_fallback_zero_exc():
    """Criterion B: dyn=1 with exc≡0 -> same amps as dyn=0 (deg path, atol 1e-9).
    Unit-normalisation of the deg vector cancels through max-normalisation of
    the projections, giving the same relative amplitudes."""
    for p in (_ell_patch(), _dense_patch()):
        exc_zero = np.zeros_like(p, dtype=float)
        _, a_deg  = c.map_laplacian(p, F0, 8, shape=0.7, dyn=0.0)
        _, a_fall = c.map_laplacian(p, F0, 8, shape=0.7, dyn=1.0, exc=exc_zero)
        assert np.allclose(a_deg, a_fall, atol=1e-9), \
            f"dyn=1 exc≡0 not equal to deg path: max diff {np.abs(a_deg-a_fall).max()}"


def test_laplacian_dyn_scale_invariant():
    """Criterion D: exc and 5·exc produce identical amplitudes (unit-norm cancels scale)."""
    p = _ell_patch()
    exc = _make_exc_for(p, scale=1.0)
    exc5 = exc * 5.0
    _, a1 = c.map_laplacian(p, F0, 8, shape=0.8, dyn=0.6, exc=exc)
    _, a5 = c.map_laplacian(p, F0, 8, shape=0.8, dyn=0.6, exc=exc5)
    assert np.allclose(a1, a5, atol=1e-9), \
        f"dyn scale invariance failed: max diff {np.abs(a1-a5).max()}"


def test_laplacian_dyn_does_not_affect_freqs():
    """dyn only affects amplitudes; frequencies are unchanged at any dyn value."""
    p = _ell_patch()
    exc = _make_exc_for(p)
    f0_res, _ = c.map_laplacian(p, F0, 8, shape=0.5, dyn=0.0)
    f1_res, _ = c.map_laplacian(p, F0, 8, shape=0.5, dyn=1.0, exc=exc)
    assert np.allclose(f0_res, f1_res, atol=1e-6), \
        f"dyn changed frequencies: {f0_res} vs {f1_res}"


def test_laplacian_dyn_symmetry_antisymmetric_mode():
    """Criterion C: dyn modulates which eigenmodes are audible based on excitation.

    Uses the blinker (P3 path graph): nodes A-B-C, deg = [1,2,1].
    lambda=1 eigenvector phi = [1,0,-1]/sqrt(2) is anti-symmetric.
    lambda=3 eigenvector psi = [1,-2,1]/sqrt(6) is symmetric.

    With dyn=0 (deg excitation): <deg, phi> = 0 -> lambda=1 mode is SILENT,
    lambda=3 mode is LOUD.  This is already tested in test_laplacian_shape_blinker_exact_amplitudes.

    Key dyn invariant: asymmetric exc opens the anti-symmetric lambda=1 mode
    that was silent under dyn=0 deg.  amps ordering FLIPS:
      dyn=0: amps = [0, 1.0] (lambda=1 silent, lambda=3 loud)
      dyn=1, exc=[1,0,0]: amps = [max, <max] (lambda=1 now loudest)

    Note: symmetric exc = [w,w,w] is proportional to the DC mode (orthogonal
    to all sounding lambda>0 modes), so mx_proj=0 triggers the rolloff fallback --
    this is tested separately in test_laplacian_dyn_fallback_zero_exc."""
    blinker = _blinker_patch()

    # dyn=0: lambda=1 mode (amps[0]) should be silent
    _, a_deg = c.map_laplacian(blinker, F0, 8, shape=1.0, dyn=0.0)
    assert abs(a_deg[0]) < 1e-6, \
        f"dyn=0 deg path should silence lambda=1 mode, got a[0]={a_deg[0]}"
    assert abs(a_deg[1] - 1.0) < 1e-6, \
        f"dyn=0 deg path: lambda=3 mode should be loudest, got a[1]={a_deg[1]}"

    # Asymmetric exc: only the left cell excited
    # e_ev=[1,0,0]: <e,phi>= 1/sqrt(2) != 0 -> lambda=1 now audible
    exc_asym = np.zeros_like(blinker, dtype=float)
    exc_asym[3, 2] = 1.0          # only the left node (node index 0 in row-major)
    _, a_asym = c.map_laplacian(blinker, F0, 8, shape=1.0, dyn=1.0, exc=exc_asym)
    # lambda=1 mode should now be the loudest (amps[0] = 1.0 by normalization)
    assert a_asym[0] > 0.5, \
        f"asymmetric exc should open lambda=1 mode, got a[0]={a_asym[0]}"
    # And the amps ordering flips: lambda=1 louder than lambda=3
    assert a_asym[0] > a_asym[1], \
        f"amps ordering should flip: a[0]={a_asym[0]} should > a[1]={a_asym[1]}"


def test_laplacian_dyn_alignment_rotation():
    """Criterion E: (mask, exc) rotated/reflected TOGETHER -> same sorted amps.
    Tests both fullshape=True and fullshape=False paths."""
    p = _ell_patch()
    exc = _make_exc_for(p)

    for fullshape in (False, True):
        _, a_base = c.map_laplacian(p, F0, 8, shape=0.8, dyn=0.7,
                                    exc=exc, fullshape=fullshape)
        a_sorted = np.sort(a_base)

        for name, (q, eq) in [
                ("rot90",  (np.rot90(p),    np.rot90(exc))),
                ("rot180", (np.rot90(p, 2), np.rot90(exc, 2))),
                ("fliplr", (np.fliplr(p),   np.fliplr(exc))),
                ("flipud", (np.flipud(p),   np.flipud(exc)))]:
            _, a_q = c.map_laplacian(np.ascontiguousarray(q), F0, 8,
                                     shape=0.8, dyn=0.7,
                                     exc=np.ascontiguousarray(eq),
                                     fullshape=fullshape)
            assert np.allclose(a_sorted, np.sort(a_q), atol=1e-6), \
                f"dyn alignment not invariant under {name} (fullshape={fullshape})"


def test_laplacian_dyn_alignment_fullshape_large():
    """Criterion E (fullshape, large shape > MAX_LAPLACIAN_NODES path):
    decimation [::step,::step] is applied to BOTH patch and exc simultaneously.

    Tests the IMPLEMENTATION invariant: manual pre-decimation with the same step
    gives a bit-for-bit identical result to letting map_laplacian do it internally.
    This proves that exc stays geometrically aligned with patch after decimation
    (decisions.md 2026-07-05, geometric alignment requirement).

    Note: rotation invariance of ARBITRARY exc on a dense block cannot hold after
    lattice decimation (rot90(X)[::s,::s] != rot90(X[::s,::s]) for non-constant X);
    the invariant tested here is implementation consistency, not rotation covariance."""
    import math
    p = _huge_dense()   # 20×20 = 400 > MAX_LAPLACIAN_NODES=256 -> step=ceil(sqrt(400/256))=2
    rng = np.random.default_rng(7)
    exc_full = rng.random(p.shape) * p   # random exc only on live cells

    # Compute the same step the function would use
    cnt_pre = int((p > 0).sum())  # 400
    step = int(math.ceil(math.sqrt(cnt_pre / c.MAX_LAPLACIAN_NODES)))  # 2

    # Pre-decimate manually (same as what map_laplacian does internally)
    p_dec   = np.ascontiguousarray(p[::step, ::step])
    exc_dec = np.ascontiguousarray(exc_full[::step, ::step])

    # Auto-decimate path: pass full-size inputs, fullshape=True -> internal decimation
    _, a_auto = c.map_laplacian(np.ascontiguousarray(p), F0, 8,
                                shape=0.6, dyn=0.5,
                                exc=np.ascontiguousarray(exc_full),
                                fullshape=True)
    # Manual-decimate path: pass pre-decimated inputs
    # (p_dec has <= MAX_LAPLACIAN_NODES nodes so no re-decimation; fullshape=True still ok)
    _, a_manual = c.map_laplacian(p_dec, F0, 8,
                                  shape=0.6, dyn=0.5,
                                  exc=exc_dec,
                                  fullshape=True)
    assert np.allclose(a_auto, a_manual, atol=1e-9), \
        f"auto-decimate != manual-decimate: max diff {np.abs(a_auto - a_manual).max()}"


def test_laplacian_dyn_analyse_no_leak():
    """Criterion E (analyse level): exc on object B's cells that fall INSIDE object A's
    bounding box must NOT affect object A's voice (masking by sub is correct).

    Setup: A is a large L-shape (8 cells, bbox rows [5,10] × cols [5,7]).  B is a
    3-cell vertical bar (8,7)-(10,7) — inside A's bbox but 8-disconnected from A
    (minimum column gap = 2, beyond 8-neighbour range).  exc is non-zero only on B's
    top cell (asymmetric: A sees zero exc everywhere; B sees asymmetric exc).

    Masking works because: exc_bbox_A = exc[A_bbox] * sub_A.  Sub_A is 0 at B's cell
    positions (B's cells are not A's cells) so B's excitation is zeroed.  This is a
    structural guarantee of the implementation; the test locks it against refactoring.

    Verified properties:
    - A's amps (larger object, sorted first) UNCHANGED vs no-exc case (atol 1e-9).
    - B's amps CHANGED vs no-exc case (asymmetric exc opens the anti-symmetric mode).
    """
    # A: 8-cell L-shape.  Vertical arm rows [5,10] col 5; horizontal top (5,6),(5,7).
    # Bbox: rows [5,10], cols [5,7].  8 cells, listed for clarity:
    A_cells = [(5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(5,6),(5,7)]
    # B: 3-cell vertical bar inside A's bbox, 8-disconnected (col gap = 2 from A).
    # Column distance from A's leftmost B-adjacent cell (8,5) to (8,7) is 2 > 1.
    B_cells = [(8,7),(9,7),(10,7)]

    grid = np.zeros((eng.GRID_H, eng.GRID_W), np.uint8)
    for r, c in A_cells:
        grid[r, c] = 1
    for r, c in B_cells:
        grid[r, c] = 1

    # exc: only B's top cell has a non-zero value (asymmetric -> affects mode ratios).
    exc_b_only = np.zeros_like(grid, dtype=float)
    exc_b_only[8, 7] = 10.0   # B's topmost cell only; A's cells all zero

    f0_hz = 261.0
    params_dyn = {'n': 8, 'spread': 0.0, 'alpha': 1.0,
                  'shape': 0.8, 'harm': 0.0, 'fullshape': False, 'dyn': 0.5}

    _, voices_no_exc, _ = eng.analyse(grid, f0_hz, 'laplacian',
                                       dict(params_dyn), exc=None)
    _, voices_exc_b,  _ = eng.analyse(grid, f0_hz, 'laplacian',
                                       dict(params_dyn), exc=exc_b_only)

    assert len(voices_no_exc) == 2 and len(voices_exc_b) == 2, \
        "expected exactly 2 voices"
    # A has 8 cells, B has 3 → A is voices[0] (sorted by area descending).
    a_A_no  = voices_no_exc[0]['amps']
    a_A_exc = voices_exc_b[0]['amps']
    a_B_no  = voices_no_exc[1]['amps']
    a_B_exc = voices_exc_b[1]['amps']

    # A must be unaffected (its exc_bbox is zero-masked by sub_A at B's positions).
    assert np.allclose(a_A_no, a_A_exc, atol=1e-9), \
        f"A's amps changed despite exc being only on B: max diff " \
        f"{np.abs(a_A_no - a_A_exc).max():.2e}"

    # B must change (asymmetric exc opens the anti-symmetric mode on B's P3 graph).
    assert not np.allclose(a_B_no, a_B_exc, atol=1e-4), \
        f"B's amps unchanged despite large asymmetric exc; " \
        f"max diff {np.abs(a_B_no - a_B_exc).max():.2e}"


def test_laplacian_dyn_symmetric_pair_real_projection():
    """FIX-C (criterion C part 2): symmetric pair exc {endpoints only} silences the
    anti-symmetric mode via real projection (mx_proj > 0 → NO rollback fallback).

    Blinker = P3: nodes (3,2),(3,3),(3,4); phi_1=[1,0,-1]/sqrt(2) anti-symmetric.
    exc = endpoints only: e_ev = [1, 0, 1] → e_hat_ev = [1,0,1]/sqrt(2).

    Projection:
      <e_hat_ev, phi_1> = (1*1 + 0*0 + 1*(-1))/2 = 0         → amps[0] = 0
      <e_hat_ev, phi_2> = (1+0+1)/(sqrt(2)*sqrt(6)) = 1/sqrt(3) → amps[1] = 1.0
    mx_proj = 1/sqrt(3) > 0 → this is NOT the rollback path. amps[:2] = [0, 1.0].
    """
    blinker = _blinker_patch()
    exc_sym_pair = np.zeros_like(blinker, dtype=float)
    exc_sym_pair[3, 2] = 1.0   # endpoint node 0
    exc_sym_pair[3, 4] = 1.0   # endpoint node 2; center (3,3) = 0
    _, a = c.map_laplacian(blinker, F0, 8, shape=1.0, dyn=1.0, exc=exc_sym_pair)
    assert abs(a[0]) < 1e-9, \
        f"symmetric pair exc must silence lambda=1 mode (real projection, not fallback): " \
        f"a[0]={a[0]:.6f}"
    assert abs(a[1] - 1.0) < 1e-9, \
        f"symmetric pair exc: lambda=3 mode should be loudest (1.0): a[1]={a[1]:.6f}"


def test_laplacian_dyn_uniform_exc_rollback():
    """FIX-C: uniform exc [w,w,w] on blinker is proportional to DC mode phi_0 =
    [1,1,1]/sqrt(3), which is ORTHOGONAL to all sounding modes (lambda>0).
    All projections = 0 → mx_proj = 0 → ROLLBACK to rolloff (NOT the same as deg path).

    Numeric check: amps[:2] = rolloff = [1.0, 0.5] (alpha=1, 2 modes).
    This is DISTINCT from the symmetric pair case (which has mx_proj > 0)
    and from dyn=0 (which gives amps[:2]=[0, 1.0] via deg projection).
    """
    blinker = _blinker_patch()
    exc_uniform = np.zeros_like(blinker, dtype=float)
    exc_uniform[3, 2] = exc_uniform[3, 3] = exc_uniform[3, 4] = 1.0
    _, a_uni = c.map_laplacian(blinker, F0, 8, shape=1.0, dyn=1.0, exc=exc_uniform)
    # All sounding projections are 0 → fallback to rolloff
    assert abs(a_uni[0] - 1.0) < 1e-9, \
        f"uniform exc -> rollback -> amps[0]=1.0 (rolloff), got {a_uni[0]:.6f}"
    assert abs(a_uni[1] - 0.5) < 1e-9, \
        f"uniform exc -> rollback -> amps[1]=0.5 (rolloff 1/2), got {a_uni[1]:.6f}"

    # Uniform-exc rollback differs from dyn=0 (deg path gives amps=[0,1])
    _, a_deg = c.map_laplacian(blinker, F0, 8, shape=1.0, dyn=0.0)
    assert not np.allclose(a_uni, a_deg, atol=1e-6), \
        "uniform rollback and deg path should differ on P3"


# ── _crop_like_extract↔extract parity (FIX-D) ────────────────────────────────

def test_crop_like_extract_matches_extract():
    """FIX-D: _crop_like_extract(exc, sub) and extract(sub) use the SAME centroid
    and window, so each live cell in extract(sub) maps back to the correct exc value.

    For shapes that fit entirely within the PATCH_SIZE window:
      _crop_like_extract(exc, sub)[extract(sub)>0] == exc[sub>0]
    (row-major order of boolean indexing is preserved by identical window geometry).

    This is the geometric alignment invariant that makes exc[patch>0] correct after
    passing _crop_like_extract output through map_laplacian (decisions.md 2026-07-05).
    """
    PATCH_SIZE = c.PATCH_SIZE

    for sub in (_blinker_patch(), _ell_patch()):
        # Assign unique float values to each live cell (index 1, 2, 3, ...)
        exc = np.zeros_like(sub, dtype=float)
        for k, (r, co) in enumerate(np.argwhere(sub > 0)):
            exc[r, co] = float(k + 1)

        patch_extracted = c.extract(sub, PATCH_SIZE)
        exc_cropped     = _crop_like_extract(exc, sub, PATCH_SIZE)

        # Boolean-index both in row-major order; values must match element-by-element
        vals_from_cropped = exc_cropped[patch_extracted > 0]
        vals_from_exc     = exc[sub > 0]
        assert np.allclose(vals_from_cropped, vals_from_exc, atol=1e-12), \
            f"_crop_like_extract mismatch on {sub.sum()}-cell shape: " \
            f"{vals_from_cropped} vs {vals_from_exc}"


# ── _exc_for_frames live-cycle parity (FIX-B) ────────────────────────────────

def test_exc_for_frames_live_cycle_parity():
    """FIX-B: _exc_for_frames with step_prevs (FIX-A schema) reproduces the EXACT
    exc_field the live synth would have computed, including:
      1. Button-triggered step recorded with true prev (not just auto-steps).
      2. First step: true initial grid used as prev (not zeros).
      3. Manual edit between steps: the edited grid is the true prev for the next step.

    Also verifies that the legacy path (step_prevs=None) DIFFERS for scenarios 1-3,
    proving the legacy fallback was inaccurate for those cases.
    """
    rng = np.random.default_rng(42)

    # Simulate a session with 3 steps and manual edits
    g0 = (rng.random((eng.GRID_H, eng.GRID_W)) < 0.3).astype(np.uint8)  # initial draw
    grid_1   = eng.step(g0)
    # Manual edit between steps 1 and 2: flip a few cells
    grid_1e = grid_1.copy()
    grid_1e[5, 5:10] ^= 1       # toggle some cells (edit)
    grid_2   = eng.step(grid_1e)
    # Random fill between steps 2 and 3
    grid_r   = (rng.random((eng.GRID_H, eng.GRID_W)) < 0.25).astype(np.uint8)
    grid_3   = eng.step(grid_r)

    step_gens  = np.array([1, 2, 3], dtype=int)
    step_grids = np.stack([grid_1, grid_2, grid_3]).astype(np.uint8)
    step_prevs = np.stack([g0, grid_1e, grid_r]).astype(np.uint8)
    # 5 frames: two at gen 1, one at gen 2, two at gen 3
    frame_gens = np.array([1, 1, 2, 3, 3], dtype=int)

    # Expected exc for each frame (direct live-cycle computation)
    exc_gen1 = eng.events_field(g0,      grid_1)   # true prev = g0
    exc_gen2 = eng.events_field(grid_1e, grid_2)   # true prev = edited grid
    exc_gen3 = eng.events_field(grid_r,  grid_3)   # true prev = random grid
    expected = [exc_gen1, exc_gen1, exc_gen2, exc_gen3, exc_gen3]

    # New path (FIX-A step_prevs)
    result = _exc_for_frames(step_gens, step_grids, step_prevs, frame_gens)
    assert len(result) == len(frame_gens), "wrong number of frames in result"
    for i, (got, want) in enumerate(zip(result, expected)):
        assert got is not None, f"frame {i} exc is None unexpectedly"
        assert np.allclose(got, want, atol=1e-12), \
            f"frame {i} (gen={frame_gens[i]}): max diff {np.abs(got-want).max():.2e}"

    # Legacy path (step_prevs=None) gives DIFFERENT results for frames at gen 1 and gen 2
    legacy = _exc_for_frames(step_gens, step_grids, None, frame_gens)
    # Gen 1: legacy uses zeros as prev (step_idx=0 → zeros); new path uses g0
    legacy_exc_gen1 = eng.events_field(np.zeros_like(g0), grid_1)
    assert np.allclose(legacy[0], legacy_exc_gen1, atol=1e-12), \
        "legacy path for gen 1 should use zeros as prev"
    # When g0 has live cells that survive, legacy gen-1 exc differs from new path
    assert not np.allclose(legacy[0], result[0], atol=1e-9), \
        "legacy and new path should differ for first step when g0 has surviving cells"
    # Gen 2: legacy uses grid_1 as prev (not grid_1e)
    legacy_exc_gen2 = eng.events_field(grid_1, grid_2)
    assert np.allclose(legacy[2], legacy_exc_gen2, atol=1e-12), \
        "legacy path for gen 2 should use preceding step grid (grid_1) as prev"
    # Legacy and new path differ for gen 2 when grid_1 ≠ grid_1e
    assert not np.allclose(legacy[2], result[2], atol=1e-9), \
        "legacy and new paths should differ for gen 2 when grid was edited between steps"


def test_exc_for_frames_clear_scenario():
    """FIX-F: serial-path handles clear() (state['gen'] resets to 0 → non-monotonic gens).

    After clear() the user draws a new field and steps again.  step_gens becomes
    non-monotonic (e.g. [1,2,1,2]).  np.searchsorted on a non-sorted array silently
    returns wrong indices → gen-path corrupts exc reconstruction for the pre-clear
    half.  Serial-path (frames col 5 = n_steps_done at frame time) is unambiguous:

      serial=2 → k=1 → events_field(grid_1, grid_2)   [pre-clear]
      serial=4 → k=3 → events_field(grid_3, grid_4)   [post-clear]

    Also verifies that gen-path diverges (documents WHY serial is required).
    """
    rng = np.random.default_rng(99)

    # Before clear: 2 steps at gen=1, gen=2
    g0     = (rng.random((eng.GRID_H, eng.GRID_W)) < 0.3).astype(np.uint8)
    grid_1 = eng.step(g0)        # serial=1, gen=1
    grid_2 = eng.step(grid_1)    # serial=2, gen=2

    # clear() → state['gen']=0; user draws new field, 2 more steps at gen=1,2
    g0p    = (rng.random((eng.GRID_H, eng.GRID_W)) < 0.3).astype(np.uint8)
    grid_3 = eng.step(g0p)       # serial=3, gen=1 (same gen numbers!)
    grid_4 = eng.step(grid_3)    # serial=4, gen=2

    step_gens  = np.array([1, 2, 1, 2], dtype=int)      # non-monotonic!
    step_grids = np.stack([grid_1, grid_2, grid_3, grid_4]).astype(np.uint8)
    step_prevs = np.stack([g0, grid_1, g0p, grid_3]).astype(np.uint8)

    # All 4 frames happen to be at gen=2 → gen-path cannot distinguish the halves.
    # frame_serials correctly discriminates pre-clear (serial=2) and post-clear (serial=4).
    frame_gens    = np.array([2, 2, 2, 2], dtype=int)
    frame_serials = np.array([2, 2, 4, 4], dtype=int)

    exc_pre  = eng.events_field(grid_1, grid_2)   # k=1: pre-clear second step
    exc_post = eng.events_field(grid_3, grid_4)   # k=3: post-clear second step
    expected = [exc_pre, exc_pre, exc_post, exc_post]

    # Serial-path: exact reconstruction for both pre- and post-clear frames
    result = _exc_for_frames(step_gens, step_grids, step_prevs,
                             frame_gens, frame_serials=frame_serials)
    assert len(result) == 4
    for i, (got, want) in enumerate(zip(result, expected)):
        assert got is not None, f"serial-path: frame {i} exc is None"
        assert np.allclose(got, want, atol=1e-12), \
            f"serial-path: frame {i} max diff {np.abs(got - want).max():.2e}"

    # Gen-path: searchsorted([1,2,1,2], 2, side='right')-1 = 3 for ALL frames
    # → pre-clear frames get exc_post instead of exc_pre (silently wrong).
    legacy = _exc_for_frames(step_gens, step_grids, step_prevs, frame_gens)
    assert not np.allclose(legacy[0], result[0], atol=1e-9), \
        "gen-path should give wrong exc for pre-clear frames on non-monotonic step_gens"


def test_laplacian_dyn_registry_integrity():
    """dyn param in Laplacian registry: spec valid, default in [lo, hi]."""
    lap = c.ENGINE_BY_ID['laplacian']
    dyn_spec = next((p for p in lap['params'] if p[0] == 'dyn'), None)
    assert dyn_spec is not None, "laplacian registry missing 'dyn' param"
    arg, label, lo, hi, integer, default = dyn_spec
    assert lo <= default <= hi, f"dyn default {default} outside [{lo},{hi}]"
    assert not integer, "dyn should be float (not integer)"
    assert lo == 0.0 and hi == 1.0, f"dyn range should be [0,1], got [{lo},{hi}]"

    # Calling with all defaults (dyn=0.0) must be bit-for-bit with no-dyn call
    kwargs_full = {p[0]: p[5] for p in lap['params']}
    f_full, a_full = lap['fn'](_ell_patch(), F0, **kwargs_full)
    kwargs_nodyn = {p[0]: p[5] for p in lap['params'] if p[0] != 'dyn'}
    f_nd, a_nd = lap['fn'](_ell_patch(), F0, **kwargs_nodyn)
    assert np.array_equal(f_full, f_nd) and np.array_equal(a_full, a_nd), \
        "registry defaults with dyn=0 not bit-for-bit with omitted-dyn call"


# ── Sethares / casynth_tuning tests ──────────────────────────────────────────

def _t_tetromino_spectrum(f0=261.0):
    """T-tetromino (2x3) spectrum via map_laplacian(fullshape=True).
    Eigenvalues {0,2,4,4} → 3 sounding modes [f0, f0*sqrt(2), f0*sqrt(2)].
    Returns (freqs, amps) for the non-zero partials only."""
    t_tet = np.zeros((2, 3), np.uint8)
    t_tet[0, 1] = 1
    t_tet[1, 0] = t_tet[1, 1] = t_tet[1, 2] = 1
    freqs, amps = c.map_laplacian(t_tet, f0, n=6, fullshape=True)
    mask = amps > 0
    return freqs[mask][:3], amps[mask][:3]


def test_tune_pair_dissonance_js_parity():
    """Criterion B: pair_dissonance matches JS linalg.js:130-134 exactly.

    Tests both the formula constants (0.24/0.021/19/3.5/5.75) and the
    min/abs pattern against a hand-computed reference that duplicates the
    JS formulation literally.  Zero-df returns 0, commutative.
    """
    # Hand-computed reference (duplicates JS formula, same constants)
    f1, a1, f2, a2 = 220.0, 1.0, 330.0, 1.0
    fmin = min(f1, f2); df = abs(f2 - f1)
    s = 0.24 / (0.021 * fmin + 19)
    expected = min(a1, a2) * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
    got = pair_dissonance(f1, a1, f2, a2)
    assert abs(got - expected) < 1e-15, \
        f"JS parity: expected {expected!r} got {got!r}"

    # Unison pair: df=0 → exp(0)-exp(0)=0
    d_unison = pair_dissonance(440.0, 0.7, 440.0, 0.7)
    assert abs(d_unison) < 1e-15, f"unison pair not zero: {d_unison}"

    # Commutative: (f1,a1,f2,a2) == (f2,a2,f1,a1)
    assert abs(pair_dissonance(220.0, 0.5, 880.0, 0.8)
               - pair_dissonance(880.0, 0.8, 220.0, 0.5)) < 1e-15, \
        "pair_dissonance not commutative"

    # Amplitude scaling: min(a1,a2) factor; d(f1, 0.5, f2, 1.0) == 0.5 * d(f1, 1.0, f2, 1.0)
    d1 = pair_dissonance(300.0, 0.5, 500.0, 1.0)
    d2 = pair_dissonance(300.0, 1.0, 500.0, 1.0)
    assert abs(d1 - 0.5 * d2) < 1e-15, \
        f"amplitude-min scaling: d*0.5={0.5*d2:.12g} got {d1:.12g}"


def test_tune_dissonance_curve_harmonic_sanity():
    """Criterion C: dissonance_curve of a 6-partial harmonic spectrum has local
    minima within 5 cents of the JI octave (1200c), fifth (702c), and fourth (498c).

    Uses the standard Sethares demo spectrum: freqs=f0*[1..6], amps=0.88**k.
    """
    f0 = 261.0
    freqs = f0 * np.arange(1, 7, dtype=float)
    amps  = 0.88 ** np.arange(6, dtype=float)
    ratios, curve = dissonance_curve(freqs, amps)
    mins = scale_minima(curve, ratios)
    assert len(mins) > 0, "no minima found in harmonic dissonance curve"
    cents = 1200.0 * np.log2(mins)

    for target, name in ((1200.0, "octave"), (702.0, "fifth"), (498.0, "fourth")):
        nearest = float(min(cents, key=lambda c_: abs(c_ - target)))
        assert abs(nearest - target) < 5.0, \
            f"no minimum within 5c of {name} ({target}c): nearest={nearest:.2f}c"

    # Curve shape: values at grid boundaries (r=1.0, r=2.1) exceed interior minimum
    assert curve.min() < curve[0], "curve min should be below left boundary"


def test_tune_dissonance_curve_t_tetromino_regression():
    """Criterion C (regression anchor): T-tetromino dissonance curve has exactly
    one minimum near 600c (sqrt(2) ratio), anchored to 600.49c +/-0.5c.

    T-tetromino Laplacian eigenvalues: {0, 2, 4, 4}.
    Sounding modes: [f0, f0*sqrt(2), f0*sqrt(2)] (first non-zero lambda = 2).
    The dominant cross-pair (f0*sqrt(2), r*f0) vanishes at r=sqrt(2) (unison),
    creating a consonance pocket near 600 cents.

    Regression anchor: 600.4918c (computed 2026-07-06 with TUNE_CURVE_STEPS=1300).
    Any change to the formula constants, curve resolution, or eigenvalue computation
    that moves this value by more than 0.5c will fail this test.
    """
    T_TET_MIN_CENTS_ANCHOR = 600.4918   # exact anchor, do not change without review
    T_TET_TOLERANCE        = 0.5        # cents

    ft, at = _t_tetromino_spectrum(f0=261.0)
    assert len(ft) == 3, f"expected 3 T-tet sounding modes, got {len(ft)}"

    ratios, curve = dissonance_curve(ft, at)
    mins = scale_minima(curve, ratios)
    assert len(mins) > 0, "no minima in T-tetromino dissonance curve"

    cents = 1200.0 * np.log2(mins)
    nearest_600 = float(min(cents, key=lambda c_: abs(c_ - 600.0)))
    assert abs(nearest_600 - T_TET_MIN_CENTS_ANCHOR) < T_TET_TOLERANCE, \
        (f"T-tetromino minimum: expected {T_TET_MIN_CENTS_ANCHOR:.4f}c "
         f"(anchor), got {nearest_600:.4f}c "
         f"(diff {abs(nearest_600-T_TET_MIN_CENTS_ANCHOR):.4f}c > "
         f"{T_TET_TOLERANCE}c tolerance)")


def test_tune_snap_ratio_exact():
    """Criterion D: tune=1 snaps r_raw exactly to the nearest minimum."""
    minima = np.array([1.2599, 1.5, 1.6818])  # major third, fifth, minor sixth
    for r_raw in (1.26, 1.4, 1.6, 1.65):
        r_snapped = snap_ratio(r_raw, minima, 1.0)
        # Must equal one of the minima exactly (within float tolerance)
        dists = np.abs(minima - r_snapped)
        assert dists.min() < 1e-12, \
            f"snap(r={r_raw}, tune=1) = {r_snapped} not equal to any minimum"

    # When r_raw is already AT a minimum, snap leaves it unchanged
    for r_min in minima:
        assert abs(snap_ratio(float(r_min), minima, 1.0) - float(r_min)) < 1e-12, \
            f"snap of exact minimum {r_min} not idempotent"


def test_tune_snap_ratio_midpoint():
    """Criterion D: tune=0.5 gives geometric midpoint in cent space.

    r_snapped = r_raw * (r_nearest / r_raw)^0.5 = sqrt(r_raw * r_nearest)
    In cents: c_snapped = (c_raw + c_nearest) / 2  (arithmetic midpoint in cents).
    """
    minima  = np.array([1.5])  # one minimum (fifth, ~702c)
    r_raw   = 1.26             # major third ~400c
    r_snap  = snap_ratio(r_raw, minima, 0.5)
    # Geometric mean (= midpoint in log/cent space)
    expected = r_raw * (1.5 / r_raw) ** 0.5
    assert abs(r_snap - expected) < 1e-12, \
        f"snap(0.5) midpoint: expected {expected:.12g} got {r_snap:.12g}"

    # In cents: half-way between ~400c and 702c → ~551c
    cents_snap     = 1200.0 * np.log2(r_snap)
    cents_raw      = 1200.0 * np.log2(r_raw)
    cents_nearest  = 1200.0 * np.log2(1.5)
    cents_expected = (cents_raw + cents_nearest) / 2.0
    assert abs(cents_snap - cents_expected) < 0.001, \
        f"snap(0.5) cent midpoint: expected {cents_expected:.4f} got {cents_snap:.4f}"


def test_tune_snap_ratio_monotone():
    """Criterion D: increasing tune monotonically approaches the minimum.

    For a fixed r_raw and a single minimum, snap should move strictly closer
    to the minimum as tune increases from 0 to 1.
    """
    minima = np.array([1.5])
    r_raw  = 1.26
    cents_raw  = 1200.0 * np.log2(r_raw)
    cents_min  = 1200.0 * np.log2(1.5)
    tune_vals  = [0.0, 0.25, 0.5, 0.75, 1.0]
    snapped    = [1200.0 * np.log2(snap_ratio(r_raw, minima, t)) for t in tune_vals]

    # All snapped values must lie between raw and minimum (inclusive)
    lo, hi = min(cents_raw, cents_min), max(cents_raw, cents_min)
    for t, c_ in zip(tune_vals, snapped):
        assert lo - 1e-9 <= c_ <= hi + 1e-9, \
            f"tune={t}: snap {c_:.4f}c outside [{lo:.2f}, {hi:.2f}]c"

    # Strict monotone: each step is closer to the minimum than the previous
    dist_to_min = [abs(c_ - cents_min) for c_ in snapped]
    for i in range(1, len(dist_to_min)):
        assert dist_to_min[i] <= dist_to_min[i - 1] + 1e-9, \
            f"not monotone: dist at tune={tune_vals[i]} ({dist_to_min[i]:.6f}) " \
            f"> dist at tune={tune_vals[i-1]} ({dist_to_min[i-1]:.6f})"


def test_tune_snap_ratio_octave_fold():
    """Criterion E: octave fold maps multi-octave intervals to their interval class.

    r_raw=3.0 (octave+fifth): k=1, fold to 1.5, snap to fifth, undo → 3.0.
    r_raw=0.75 (fourth below): k=-1, fold to 1.5 (fourth above), snap, undo → 0.75.
    r_raw=2.0 (exact octave): k=1, r_folded=1.0 < TUNE_SKIP_BELOW → no snap, return 2.0.
    """
    minima = np.array([1.5])   # only minimum is the fifth

    # Octave + fifth snaps to twice the fifth (interval class: fifth)
    r_oct_fifth = snap_ratio(3.0, minima, 1.0)
    assert abs(r_oct_fifth - 3.0) < 1e-12, \
        f"r=3.0 (oct+fifth) should snap to 3.0 (fifth*2), got {r_oct_fifth}"

    # Fourth below = 2/3 ≈ 0.667, folds to 4/3 but only [1.5] min available
    # Actually r_raw=0.75 = 3/4: k=-1, r_folded=1.5, snaps to 1.5, undo = 0.75
    r_fourth_down = snap_ratio(0.75, minima, 1.0)
    assert abs(r_fourth_down - 0.75) < 1e-12, \
        f"r=0.75 (fourth-below) should snap to 0.75 (fifth-below), got {r_fourth_down}"

    # Exact octave: r_folded=1.0 < TUNE_SKIP_BELOW → guard fires, return 2.0 unchanged
    r_exact_oct = snap_ratio(2.0, minima, 1.0)
    assert abs(r_exact_oct - 2.0) < 1e-12, \
        f"r=2.0 (exact octave) should be returned unchanged, got {r_exact_oct}"

    # Three octaves + fifth: r=12.0 → k=3, r_folded=1.5, snap to 1.5*8=12.0
    r_multi = snap_ratio(12.0, minima, 1.0)
    assert abs(r_multi - 12.0) < 1e-10, \
        f"r=12.0 (3oct+fifth) should snap to 12.0, got {r_multi}"


def test_tune_snap_ratio_transparent():
    """Criterion F: no snap when tune=0 or minima is empty."""
    minima = np.array([1.3, 1.5, 1.68])
    for r in (1.1, 1.26, 1.5, 1.7, 2.05):
        # tune=0: transparent regardless of minima
        assert abs(snap_ratio(r, minima, 0.0) - r) < 1e-15, \
            f"tune=0 not transparent at r={r}"
        # empty minima: transparent regardless of tune
        assert abs(snap_ratio(r, np.array([]), 1.0) - r) < 1e-15, \
            f"empty minima not transparent at r={r}"


def test_tune_zero_bitexact():
    """Criterion A: tune=0 → snap_ratio returns r_raw exactly (bit-level).

    This is the unit-level guarantee that underlies the full audio bit-exact
    criterion A: when tune=0, _live_voices falls back to midi_to_freq(note)
    and no dissonance computation occurs.  The snap function is the gate.
    """
    import math
    test_ratios = [0.5, 1.0, 1.26, 1.5, 1.99, 2.0, 3.14159, 4.0]
    minima = np.array([1.1, 1.25, 1.5, 1.7, 2.0])
    for r in test_ratios:
        got = snap_ratio(r, minima, 0.0)
        # float-identical: same bits, not just close
        assert got == r, \
            f"tune=0 snap changed r={r} to {got} (expected bit-exact equality)"
        # Also test with empty minima (should be same code path)
        got_empty = snap_ratio(r, np.array([]), 0.0)
        assert got_empty == r, \
            f"tune=0 empty-minima snap changed r={r} to {got_empty}"


# ── Runner (works without pytest) ─────────────────────────────────────────────

def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:           # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
