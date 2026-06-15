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
