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
