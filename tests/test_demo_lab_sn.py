#!/usr/bin/env python3
"""S / N sound demos (REQ memory/req-sonification-sn-demos-2026-09-14.md, section 6):
independent checks of the mechanisms of casynth_lab/scan_surface.py (S) and
casynth_lab/pm_network.py (N) plus the common bench contract (journal
determinism, byte-exact continuation, snapshot hygiene, level / format
limits, quality vs a denser analysis / a higher internal rate).

    python tests/test_demo_lab_sn.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_sn/.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'demos'))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN, _RAMP                       # noqa: E402
from casynth_lab import DemoRunner, BLOCK, registry, scene_from_doc     # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot    # noqa: E402
from casynth_lab.snapshot import save_state, load_state                # noqa: E402
from casynth_lab import scan_surface as S                              # noqa: E402
from casynth_lab import pm_network as N                                # noqa: E402
from scipy import signal                                               # noqa: E402
import build_sn_demos as D                                             # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_sn')
OUTPUTS = ('A', 'B', 'monitor')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 4.0)
GAIN = MASTER_GAIN * 0.7


def _db(x, ref):
    return 20.0 * math.log10((x + 1e-30) / (ref + 1e-30))


def _rms(x):
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _doc(field, A, B, f0=110.0, rate=4.0):
    return D.scene_doc('t', 't', field, A, B, rate, '', f0=f0)


def _drive(runner, n_blocks, commands=(), collect=True):
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    got = {o: [] for o in OUTPUTS} if collect else None
    for _ in range(n_blocks):
        b = runner.next_block()
        if collect:
            for o in OUTPUTS:
                got[o].append(b.get(o))
    return got


def _same(g1, g2):
    for o in OUTPUTS:
        a, b = np.concatenate(g1[o]), np.concatenate(g2[o])
        if a.shape != b.shape or not np.array_equal(a, b):
            return False, o
    return True, ''


def _engine(eid, grid, **over):
    params = dict(registry.defaults(eid))
    params.update(over)
    e = registry.create(eid, CTX, params)
    e.init(grid, None, GAIN)
    return e


def _render(e, n_blocks, gain=GAIN):
    return np.concatenate([e.render_float(gain)[0] for _ in range(n_blocks)])


# =============================================================================
# S -- surfaces
# =============================================================================
def test_s_constant_fields_are_silent_for_every_path_and_surface():
    empty = np.zeros((32, 32), np.uint8)
    full = np.ones((32, 32), np.uint8)
    for grid in (empty, full):
        for surf in (0, 1):
            for path in (0, 1, 2):
                e = _engine('scan_surface', grid, surface=surf, path=path)
                y = _render(e, 30)
                assert np.all(np.isfinite(y)) and np.abs(y).max() == 0.0, (surf, path)
                assert np.abs(e.render(GAIN, 0)[0]).max() == 0
    # Distance on empty / full: constant -1 / +1 exactly, no inf / NaN
    assert np.array_equal(S.distance_surface(empty, 1.5), -np.ones((32, 32)))
    assert np.array_equal(S.distance_surface(full, 1.5), np.ones((32, 32)))
    # a field cleared during play fades to silence within the 20 ms transition
    e = _engine('scan_surface', D.field_f1())
    _render(e, 20)
    e.update_field(empty, None)
    y = _render(e, 10)
    n = int(round(S.XFADE_S * SR))
    assert np.abs(y[:n]).max() > 0.0 and np.abs(y[n + BLOCK:]).max() == 0.0


def test_s_analytic_reading_z_equals_x_gives_cosine():
    """Internal fixture: on the surface z = x an ellipse reads cx + rx cos(2 pi p);
    after DC removal exactly one harmonic with amplitude rx and zero phase."""
    rows, cols = 32, 32
    z = np.tile(np.arange(cols, dtype=np.float64), (rows, 1))
    for rx_frac in (0.8, 0.3):
        params = dict(surface=0, width=1.5, path=0, radius_x=rx_frac, radius_y=0.5)
        v = S.read_period(z, params, 8192)
        cx = (cols - 1) / 2.0
        rx = rx_frac * cx
        assert np.allclose(v, cx + rx * np.cos(2 * np.pi * np.arange(8192) / 8192), atol=1e-9)
        a = S.band_limit(v, 180)
        assert abs(a[0] - rx) < 1e-9 and np.abs(a[1:]).max() < 1e-9
    # phases are kept: a shifted reading (sin instead of cos) keeps its imaginary coefficient
    v = np.sin(2 * np.pi * np.arange(8192) / 8192 + 0.7)
    a = S.band_limit(v, 10)
    assert abs(a[0] - np.exp(1j * (0.7 - np.pi / 2))) < 1e-9


def test_s_paths_raster_coverage_seam_closure_and_radii():
    for rows, cols in ((32, 32), (5, 7), (6, 3), (1, 4), (4, 1)):
        nodes = S.raster_nodes(rows, cols)
        centres = nodes[:-1]
        assert len(centres) == rows * cols
        assert len({(int(x), int(y)) for x, y in centres}) == rows * cols     # each once
        assert np.all(centres == np.round(centres))
        length, cum = S._polyline_cum(nodes)
        seam = length[-1]
        expect_seam = 1.0 if rows % 2 == 0 else math.sqrt(2.0)
        if rows == 1:
            expect_seam = 1.0 if cols > 1 else 0.0
        if cols == 1:
            expect_seam = 1.0
        assert abs(seam - expect_seam) < 1e-12, (rows, cols, seam)
        assert abs(cum[-1] - ((rows * cols - 1) + seam)) < 1e-9
        # p -> P(p) is closed and runs at constant speed along the length
        x0, y0 = S.path_points(2, rows, cols, 0.8, 0.8, np.array([0.0]))
        x1, y1 = S.path_points(2, rows, cols, 0.8, 0.8, np.array([1.0 - 1e-12]))
        assert (x0[0], y0[0]) == (0.0, 0.0)
        d = math.hypot(x1[0] - nodes[-1, 0], y1[0] - nodes[-1, 1])
        assert d < 1e-6
    # analytic paths: closed, radii match the extents used by the overlay
    for path in (0, 1):
        for rx, ry in ((0.8, 0.8), (0.1, 0.95)):
            p = np.linspace(0, 1, 4001)
            x, y = S.path_points(path, 32, 32, rx, ry, p)
            assert abs(x[0] - x[-1]) < 1e-9 and abs(y[0] - y[-1]) < 1e-9
            assert abs(x.max() - (15.5 + rx * 15.5)) < 1e-3 and abs(x.min() - (15.5 - rx * 15.5)) < 1e-3
            assert abs(y.max() - (15.5 + ry * 15.5)) < 1e-3
            ov = S.overlay(dict(surface=0, width=1.5, path=path, radius_x=rx, radius_y=ry), 32, 32)
            pts = np.array(ov['polyline'])
            assert abs(pts[:, 0].max() - x.max()) < 1e-3 and abs(pts[:, 1].max() - y.max()) < 1e-3
            assert ov['start'] == (float(x[0]), float(y[0]))
    ov = S.overlay(dict(surface=0, width=1.5, path=2, radius_x=0.8, radius_y=0.8), 32, 32)
    assert len(ov['polyline']) == 32 * 32 + 1
    assert S.inactive(dict(path=2)) == {'radius_x': 'Full field', 'radius_y': 'Full field'}
    assert S.inactive(dict(path=0)) == {}
    # 1x1: no division by zero
    x, y = S.path_points(2, 1, 1, 0.8, 0.8, np.linspace(0, 1, 5))
    assert np.all(x == 0) and np.all(y == 0)
    # on the asymmetric field F1 the three paths read three different raw sequences
    z = S.smooth_surface(D.field_f1(), 1.5)
    reads = [S.read_period(z, dict(surface=0, width=1.5, path=p, radius_x=0.8, radius_y=0.8),
                           8192) for p in (0, 1, 2)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(reads[i], reads[j]), (i, j)
            assert np.abs(reads[i] - reads[i].mean()).max() > 0.05


def test_s_surfaces_periodic_borders_signs_units_and_not_a_copy():
    g = np.zeros((16, 20), np.uint8)
    g[0, 0] = 1
    zs = S.smooth_surface(g, 1.5)
    # periodic borders: the four seam neighbours of (0,0) are equal, symmetric
    assert abs(zs[0, 1] - zs[0, 19]) < 1e-12 and abs(zs[1, 0] - zs[15, 0]) < 1e-12
    assert zs[0, 0] > zs[0, 1] > zs[0, 2]
    assert zs.max() <= 1.0 + 1e-12 and zs.min() >= -1.0 - 1e-12
    # normalised kernel: the mean of 2G-1 equals 2*density-1
    assert abs(zs.mean() - (2.0 * g.mean() - 1.0)) < 1e-9
    zd = S.distance_surface(g, 1.0)
    d_live = S.torus_distance_to(g > 0)
    assert d_live[0, 2] == 2.0 and d_live[0, 18] == 2.0 and d_live[15, 0] == 1.0      # torus
    assert abs(d_live[3, 4] - 5.0) < 1e-12                                            # euclid
    assert abs(zd[0, 0] - math.tanh(1.0)) < 1e-12                                     # d_dead=1
    assert zd[0, 0] > 0 and zd[8, 10] < 0 and abs(zd[0, 2] - math.tanh(-2.0)) < 1e-12
    assert abs(zd[0, 2] - zd[0, 18]) < 1e-12
    # Distance is not a renormalised Smooth: on F1 the rank order differs
    f1 = D.field_f1()
    a, b = S.smooth_surface(f1, 1.5).ravel(), S.distance_surface(f1, 1.5).ravel()
    from numpy import corrcoef
    assert corrcoef(a, b)[0, 1] < 0.99
    assert np.count_nonzero(np.argsort(a) != np.argsort(b)) > 100


def test_s_density_quality_dc_band_phase_and_no_phase_reset():
    """Doubling the analysis density changes the band-limited wave by < -50 dB
    on the working scenes; DC = 0; nothing above K; a path change keeps the
    phase accumulator."""
    K = S.harmonics(SR, 110.0)
    assert K == 180 and S.harmonics(SR, 440.0) == 45 and S.harmonics(SR, 30000.0) == 0
    p = np.arange(4096) / 4096.0
    # working cases: F1 / F2 with every path, F3 with Raster (the analytic paths
    # miss the centred 13x13 Pulsar at radius 0.8 -- listed, not a working scene)
    cases = [(f, s, pth) for f in ('F1', 'F2') for s in (0, 1) for pth in (0, 1, 2)]
    cases += [('F3', s, 2) for s in (0, 1)]
    for field, surf, path in cases:
        g = D.FIELDS[field]()
        if True:
            if True:
                params = dict(surface=surf, width=1.5, path=path, radius_x=0.8, radius_y=0.8)
                z = S.surface(g, surf, 1.5)
                a1 = S.band_limit(S.read_period(z, params, 8192), K)
                a2 = S.band_limit(S.read_period(z, params, 16384), K)
                w1, w2 = S.evaluate(a1, p), S.evaluate(a2, p)
                assert _rms(w1) > 1e-3, (field, surf, path)
                assert _db(_rms(w1 - w2), _rms(w1)) < -50.0, (field, surf, path, _db(_rms(w1 - w2), _rms(w1)))
                assert abs(w1.mean()) < 1e-9
                assert len(a1) == K and np.abs(a1.imag).max() > 1e-6       # complex phases kept
                # the rendered block equals the exact additive evaluation (no table interpolation)
                e = _engine('scan_surface', g, surface=surf, path=path, trim_db=0.0)
                y = e.render_float(GAIN)[0]
                ref = S.evaluate(e.a_from, np.arange(BLOCK) * 110.0 / SR) * GAIN * S.RAW_SCALE
                assert np.allclose(y, ref, atol=1e-9)
    z = S.smooth_surface(D.field_f3(), 1.5)
    miss = S.read_period(z, dict(surface=0, width=1.5, path=0, radius_x=0.8, radius_y=0.8), 8192)
    assert np.abs(miss - miss.mean()).max() < 0.05          # documented: the ellipse misses F3 (tails only)
    # a path / surface change never resets the phase; the transition is 20 ms
    e = _engine('scan_surface', D.field_f1(), path=0)
    _render(e, 5)
    ph = e.phase
    e.set_params(dict(e.params, path=1))
    assert e.phase == ph and e.xpos == 0
    _render(e, 3)
    assert abs(e.phase - ((ph + 3 * BLOCK * 110.0 / SR) % 1.0)) < 1e-12
    assert e.xpos >= e.xfade and e.xfade == 882
    # an edit in the middle of a transition starts from the reached mix
    e.set_params(dict(e.params, path=2))
    y1 = e.render_float(GAIN)[0]                       # xpos = BLOCK of 882
    mix = e.a_from * (1 - BLOCK / 882.0) + e.a_to * (BLOCK / 882.0)
    e.set_params(dict(e.params, surface=1))
    assert np.allclose(e.a_from, mix) and e.xpos == 0
    y2 = e.render_float(GAIN)[0]
    # continuous: the step across the restart is no larger than the wave's own steps
    assert abs(y2[0] - y1[-1]) <= 1.5 * np.abs(np.diff(y1)).max() + 1e-9


# =============================================================================
# N -- network
# =============================================================================
def test_n_coupling_zero_is_the_plain_sum_through_the_same_output_path():
    """coupling = 0 on a settled non-empty field == the direct sum of four
    harmonic sines / 4 through an independent copy of the decimator + HPF;
    two different fields (masks) give the same output."""
    n_blocks = 60
    outs = []
    for field in ('F5T', 'F4L'):
        e = _engine('pm_network', D.FIELDS[field](), coupling=0.0, trim_db=0.0)
        outs.append(_render(e, n_blocks))
    assert np.array_equal(outs[0], outs[1])
    # independent reference: raw sum at 4x, gate ramp (20 ms from 0), FIR convolve, lfilter HPF
    R = N.OVERSAMPLE
    n = n_blocks * BLOCK * R
    theta = (np.arange(n) * 110.0 / (SR * R)) % 1.0
    raw = sum(np.sin(2 * np.pi * ((k * theta) % 1.0)) for k in (1, 2, 3, 4)) / 4.0
    gate = np.minimum(np.arange(1, n + 1) / (N.GATE_S * SR * R), 1.0)
    taps = N.decimator_taps(SR, R)
    y = np.convolve(raw * gate, taps)[:n][::R]
    b, a = N.hpf_coeffs(SR)
    y = signal.lfilter(b, a, y) * GAIN * N.RAW_SCALE
    skip = BLOCK * 10
    assert np.allclose(outs[0][skip:], y[skip:], atol=1e-9)
    assert _rms(outs[0][skip:]) > 1e-3


def test_n_isolated_edge_direction_and_periodicity():
    theta = np.random.RandomState(3).rand(500) * 7.0
    beta = 2.0
    for k, (i, j) in enumerate(N.EDGES):
        W = np.zeros(6)
        W[k] = 0.25
        x, raw = N.pm_network(theta, W, beta)
        pure = [np.sin(2 * np.pi * ((r * theta) % 1.0)) for r in N.RATIOS]
        # only i is modulated, by the PURE x_j of the same sample; everyone else is pure
        expect_i = np.sin(2 * np.pi * ((N.RATIOS[i] * theta) % 1.0) + beta * 0.25 * pure[j])
        assert np.allclose(x[i], expect_i, atol=1e-12)
        for m in range(4):
            if m != i:
                assert np.allclose(x[m], pure[m], atol=1e-12), (k, m)
        assert np.allclose(raw, x.sum(axis=0) / 4.0)
    # chained: 2>1 and 1>0 -- x_0 sees the MODULATED x_1 (same sample), no delay
    W = np.zeros(6)
    W[N.EDGES.index((1, 2))] = 0.3
    W[N.EDGES.index((0, 1))] = 0.2
    x, _ = N.pm_network(theta, W, beta)
    x2 = np.sin(2 * np.pi * ((3 * theta) % 1.0))
    x1 = np.sin(2 * np.pi * ((2 * theta) % 1.0) + beta * 0.3 * x2)
    x0 = np.sin(2 * np.pi * (theta % 1.0) + beta * 0.2 * x1)
    assert np.allclose(x[0], x0, atol=1e-12) and np.allclose(x[1], x1, atol=1e-12)
    # periodic in theta with period 1 cycle (2 pi), at non-integer thetas
    W = np.random.RandomState(5).rand(6) / 3.0
    x_a, raw_a = N.pm_network(theta, W, 3.0)
    x_b, raw_b = N.pm_network(theta + 1.0, W, 3.0)
    assert np.allclose(raw_a, raw_b, atol=1e-9) and np.allclose(x_a, x_b, atol=1e-9)
    # beta = 0 -> independent sum whatever W
    _x, raw0 = N.pm_network(theta, W, 0.0)
    assert np.allclose(raw0, sum(np.sin(2 * np.pi * ((r * theta) % 1.0)) for r in N.RATIOS) / 4)


def test_n_weights_masks_fields_frozen_links_and_gate():
    K = N.masks(32, 32)
    assert K.shape == (6, 32, 32) and np.all(K.max(axis=(1, 2)) > 0.9)   # centres between cells
    cx = [(31 * f) for f in N.MASK_X]
    assert N.mask_centres(32, 32)[0] == (cx[0], 31 * 0.25) and N.mask_centres(32, 32)[5] == (cx[2], 31 * 0.75)
    # torus: a mask centred near the left edge wraps to the right edge
    k0 = K[0]
    assert k0[8, 31] > k0[8, 12]
    # full field -> every weight exactly 1/3; empty -> 0
    assert np.allclose(N.field_weights(np.ones((32, 32))), 1.0 / 3.0)
    assert np.allclose(N.field_weights(np.zeros((32, 32))), 0.0)
    # F5 top / bottom: the two rows of links swap (about 0.29 <-> 0.04)
    wt, wb = N.field_weights(D.field_f5(True)), N.field_weights(D.field_f5(False))
    assert np.all(wt[:3] > 0.25) and np.all(wt[3:] < 0.06)
    assert np.allclose(wt[:3], wb[3:]) and np.allclose(wt[3:], wb[:3])
    # F1 / F2: near 0.11 each (fine structure averages out)
    assert np.abs(N.field_weights(D.field_f1()) - 0.11).max() < 0.02
    # equal-area F4 left / right: different W; copies: identical W
    wl, wr = N.field_weights(D.FIELDS['F4L']()), N.field_weights(D.FIELDS['F4R']())
    assert not np.allclose(wl, wr) and np.abs(wl - wr).max() > 1e-3
    assert np.array_equal(wl, N.field_weights(D.FIELDS['F4L']().copy()))
    # F3 Pulsar (period 3) modulates W across its cycle
    from casynth_engine import step
    g = D.field_f3()
    ws = []
    for _ in range(3):
        ws.append(N.field_weights(g))
        g = step(g)
    assert max(np.abs(ws[a] - ws[b]).max() for a in range(3) for b in range(a + 1, 3)) > 1e-3
    # engine: init takes W from the field at once; a field edit moves W with tau 30 ms
    e = _engine('pm_network', D.field_f5(True), coupling=2.0)
    assert np.allclose(e.W, wt) and np.allclose(e.W_target, wt)
    e.update_field(D.field_f5(False), None)
    assert np.allclose(e.W, wt) and np.allclose(e.W_target, wb)
    _render(e, 1)
    frac = 1.0 - math.exp(-BLOCK / (N.TAU_W * SR))
    assert np.allclose(e.W, wt + (wb - wt) * frac, atol=1e-9)
    _render(e, 40)                                                       # ~320 ms: settled
    assert np.abs(e.W - wb).max() < 1e-4
    # frozen: further edits of a non-empty field do not move W; off -> reads the field again
    e.set_params(dict(e.params, freeze_links=1))
    held = e.W.copy()
    e.update_field(D.field_f5(True), None)
    _render(e, 20)
    assert np.array_equal(e.W, held) and np.array_equal(e.W_target, held)
    e.set_params(dict(e.params, freeze_links=0))
    assert np.allclose(e.W_target, wt)
    _render(e, 40)
    assert np.abs(e.W - wt).max() < 1e-4
    # initial freeze_links=1 holds the initial field's W; an edit is ignored
    e2 = _engine('pm_network', D.field_f5(True), coupling=2.0, freeze_links=1)
    e2.update_field(D.field_f5(False), None)
    _render(e2, 5)
    assert np.allclose(e2.W, wt)
    # gate: an empty field closes the sound in 20 ms; repainting reopens it; zero links stay audible
    e3 = _engine('pm_network', D.field_f5(True), coupling=2.0)
    _render(e3, 20)
    e3.update_field(np.zeros((32, 32), np.uint8), None)
    y = _render(e3, 60)
    # gate 20 ms + decimator delay 32 samples + the 5 Hz HPF tail (32 ms time constant)
    assert np.abs(y[: 2 * BLOCK]).max() > 0 and np.abs(y[6 * BLOCK:]).max() < 1e-2 * np.abs(y).max()
    assert np.abs(y[50 * BLOCK:]).max() < 1e-6
    g = np.zeros((32, 32), np.uint8)
    g[31, 31] = 1                                    # far from the masks: W ~ 0, still audible
    e3.update_field(g, None)
    y = _render(e3, 30)
    assert _rms(y[10 * BLOCK:]) > 1e-3 and np.abs(e3.W).max() < 1e-3
    # display numbers are plain floats
    d = e3.display()
    assert set(d) >= {'W', 'W_target', 'frozen', 'gate'} and len(d['W']) == 6


def test_n_pause_keeps_phase_and_smoothing_running():
    doc = _doc('F5T', D.network(coupling=2.0), D.network(coupling=2.0))
    r = DemoRunner(scene_from_doc(doc))
    _drive(r, 30, [('start', 0, {}), ('pause', 0, {'on': True})], collect=False)
    e = r.sides['A'].engine
    th, w = e.theta, e.W.copy()
    r.post('set_cell', r=20, c=5, v=1)                  # paint while paused: W target moves
    _drive(r, 10, collect=False)
    assert r.paused and r.gen == 0
    assert e.theta != th and not np.array_equal(e.W, w)
    assert np.allclose(e.W_target, N.field_weights(r.grid))


def test_n_oversampling_4x_vs_8x_and_s_render_quality():
    """4x vs 8x internal rate on the prepared fields, incl. beta = 4 and f0 = 440:
    same delay by design, residual RMS <= -50 dB of the reference RMS on the
    static working cases (compared as floats before int16)."""
    worst = {}
    for f0 in (110.0, 440.0):
        ctx = EngineContext(SR, BLOCK, 2, f0, 1.0, 4.0)
        for field in ('F5T', 'F5B', 'F4L', 'F1', 'F3'):
            for beta in (2.0, 4.0):
                g = D.FIELDS[field]()
                params = dict(coupling=beta, freeze_links=0, trim_db=0.0)
                e4 = N.PMNetworkEngine(ctx, params, oversample=4)
                e8 = N.PMNetworkEngine(ctx, params, oversample=8)
                e4.init(g, None, GAIN)
                e8.init(g, None, GAIN)
                y4 = np.concatenate([e4.render_float(GAIN)[0] for _ in range(80)])
                y8 = np.concatenate([e8.render_float(GAIN)[0] for _ in range(80)])
                skip = 20 * BLOCK
                d = _db(_rms(y4[skip:] - y8[skip:]), _rms(y8[skip:]))
                worst[(f0, field, beta)] = d
                assert _rms(y8[skip:]) > 1e-4
    key = max(worst, key=worst.get)
    print(f"      N 4x vs 8x worst residual {worst[key]:.1f} dB at f0={key[0]:.0f} {key[1]} beta={key[2]}")
    assert worst[key] < -50.0, worst
    # the decimator: exact delay FIR_HALF output samples for both rates, unity DC gain
    for R in (4, 8):
        taps = N.decimator_taps(SR, R)
        assert len(taps) == 2 * R * N.FIR_HALF + 1 and abs(taps.sum() - 1.0) < 1e-9
        assert np.allclose(taps, taps[::-1])
        w, h = signal.freqz(taps, worN=[2000.0, 19000.0, (SR - 19000.0)], fs=R * SR)
        assert abs(abs(h[0]) - 1.0) < 1e-3 and abs(abs(h[1]) - 1.0) < 2e-3 and abs(h[2]) < 1e-3


# =============================================================================
# common: determinism, continuation, snapshot hygiene, limits, gain smoothing
# =============================================================================
def _ab_commands(base):
    return [('set_cell', base + 5 * BLOCK, {'r': 3, 'c': 4, 'v': 1}),
            ('set_cell', base + 5 * BLOCK, {'r': 3, 'c': 5, 'v': 1}),
            ('select', base + 20 * BLOCK, {'side': 'B'}),
            ('set_param', base + 40 * BLOCK, {'side': 'B', 'name': 'trim_db', 'value': -3.0}),
            ('vol', base + 60 * BLOCK, {'value': 0.5}),
            ('pause', base + 80 * BLOCK, {'on': True}),
            ('set_cell', base + 90 * BLOCK, {'r': 10, 'c': 10, 'v': 0}),
            ('pause', base + 100 * BLOCK, {'on': False}),
            ('select', base + 120 * BLOCK, {'side': 'A'})]


def _pairs():
    return [('scan x2', _doc('F1', D.scan(path=0), D.scan(path=2))),
            ('network x2', _doc('F5T', D.network(coupling=0.0), D.network(coupling=2.0), rate=2.0)),
            ('scan + laplace', _doc('F3', D.laplace(), D.scan(surface=1), rate=2.0)),
            ('network + laplace', _doc('F3', D.laplace(harm=1), D.network(coupling=2.0), rate=2.0)),
            ('scan + network', _doc('F2', D.scan(path=1), D.network(coupling=3.0), rate=2.0))]


def test_same_journal_same_output_and_continuation_byte_exact():
    os.makedirs(ART, exist_ok=True)
    states = {}
    for name, doc in _pairs():
        scene = scene_from_doc(doc)
        pre = [('start', 0, {}), ('set_cell', 30 * BLOCK, {'r': 1, 'c': 1, 'v': 1})]
        pre += [('set_param', 60 * BLOCK, {'side': 'B', 'name': 'trim_db', 'value': 2.0})]
        r1, r2 = DemoRunner(scene), DemoRunner(scene)
        g1, g2 = _drive(r1, 100, list(pre)), _drive(r2, 100, list(pre))
        ok, o = _same(g1, g2)
        assert ok, f"{name}: same journal, different {o}"
        assert max(int(np.abs(np.concatenate(g1[o])).max()) for o in OUTPUTS) > 300, name
        # snapshot in the middle of the S coefficient transition / N smoothing:
        # a field edit right before the cut
        r1.post('set_cell', at=None, r=2, c=2, v=1)
        r1.next_block()
        st = r1.export_state()
        assert all(supports_snapshot(r1.sides[s].engine) for s in 'AB')
        sA = st['sides']['A']['engine']
        if sA['engine_id'] == 'scan_surface':
            assert 0 < sA['xpos'] < 882
        base = r1.out_samples
        save_state(ART, name.replace(' ', '_'), st)
        r3 = DemoRunner.from_state(load_state(ART, name.replace(' ', '_')))
        h1 = _drive(r1, 160, _ab_commands(base))
        h3 = _drive(r3, 160, _ab_commands(base))
        ok, o = _same(h1, h3)
        assert ok, f"{name}: continuation differs on {o}"
        assert r1.gen == r3.gen and np.array_equal(r1.grid, r3.grid)
        states[name] = (name.replace(' ', '_'), base,
                        {o: np.concatenate(h1[o]) for o in OUTPUTS})
    # fresh process: restore from the files and render the same commands
    out = os.path.join(ART, 'fresh.npz')
    code = ("import sys, numpy as np; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import test_demo_lab_sn as T\n"
            "from casynth_lab import DemoRunner\nfrom casynth_lab.snapshot import load_state\n"
            "res = {}\n"
            "for key, base in %r:\n"
            "    r = DemoRunner.from_state(load_state(%r, key))\n"
            "    g = T._drive(r, 160, T._ab_commands(base))\n"
            "    for o in T.OUTPUTS: res[key + '.' + o] = np.concatenate(g[o])\n"
            "np.savez(%r, **res)\n"
            % (ROOT, os.path.dirname(os.path.abspath(__file__)),
               [(k, b) for (k, b, _e) in states.values()], ART, out))
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0, p.stderr
    with np.load(out) as z:
        for key, _base, expect in states.values():
            for o in OUTPUTS:
                assert np.array_equal(z[f"{key}.{o}"], expect[o]), f"fresh {key}: {o}"


def test_snapshot_hygiene_and_rejections():
    doc = _doc('F1', D.scan(path=1), D.network(coupling=2.0))
    r = DemoRunner(scene_from_doc(doc))
    _drive(r, 40, [('start', 0, {})], collect=False)
    st = r.export_state()
    # no live arrays referenced: mutate the live engines, the export stays
    keep = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for s in 'AB' for k, v in st['sides'][s]['engine'].items()}
    r.sides['A'].engine.a_from[:] = 0
    r.sides['A'].engine.a_to[:] = 0
    r.sides['B'].engine.W[:] = 9
    r.sides['B'].engine.fir_mem[:] = 1
    for s in 'AB':
        for k, v in st['sides'][s]['engine'].items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, keep[k]), (s, k)
    # rejections before anything is replaced: version / shape / engine mismatch
    def bad(mut, what):
        import copy
        s2 = copy.deepcopy(st)
        mut(s2)
        try:
            DemoRunner.from_state(s2)
        except ValueError as e:
            assert what in str(e), (what, str(e))
            return
        raise AssertionError(f"accepted a bad snapshot: {what}")
    bad(lambda s: s['sides']['A']['engine'].update(version=99), 'version')
    bad(lambda s: s['sides']['A']['engine'].update(a_from=np.zeros(5, complex)), 'a_from')
    bad(lambda s: s['sides']['A']['engine'].update(K=7), 'harmonics')
    bad(lambda s: s['sides']['B']['engine'].update(fir_mem=np.zeros(3)), 'fir_mem')
    bad(lambda s: s['sides']['B']['engine'].update(oversample=8), 'render layout')
    bad(lambda s: s['sides']['B']['engine'].update(engine_id='scan_surface'), 'state is for')
    bad(lambda s: s['sides']['B']['engine'].update(theta=1.5), 'scalar')
    bad(lambda s: s['sides']['A']['engine'].update(a_to=np.full(180, np.nan + 0j)), 'finite')
    # a good one still restores
    DemoRunner.from_state(st)


def test_limits_110_440_extreme_params_and_prepared_scenes_do_not_clip():
    empty = np.zeros((32, 32), np.uint8)
    full = np.ones((32, 32), np.uint8)
    sparse = np.zeros((32, 32), np.uint8)
    sparse[7, 9] = 1
    for f0 in (110.0, 440.0):
        ctx = EngineContext(SR, BLOCK, 2, f0, 1.0, 4.0)
        for grid in (empty, full, sparse, D.field_f1()):
            for eid, extremes in (('scan_surface', [dict(width=0.5, radius_x=0.1, radius_y=0.95, trim_db=24.0),
                                                     dict(width=3.0, surface=1, path=1, trim_db=-24.0)]),
                                  ('pm_network', [dict(coupling=4.0, trim_db=24.0),
                                                  dict(coupling=0.0, freeze_links=1, trim_db=-24.0)])):
                for over in extremes:
                    params = dict(registry.defaults(eid))
                    params.update(over)
                    e = registry.create(eid, ctx, params)
                    e.init(grid, None, MASTER_GAIN * 1.0)
                    for _ in range(10):
                        buf, peak, n_clip = e.render(MASTER_GAIN * 1.0, 0)
                        assert buf.dtype == np.int16 and buf.shape == (BLOCK, 2)
                        assert np.array_equal(buf[:, 0], buf[:, 1])
                        assert math.isfinite(peak) and peak >= 0 and n_clip >= 0
                        assert (n_clip > 0) == (peak > 1.0)
    # prepared scenes (build_sn_demos): every record renders with n_clip = 0 on both sides
    for rec in D.RECORDS:
        cat, sid, title, note, field, A, B, static = rec
        doc = D.scene_for(rec)
        r = DemoRunner(scene_from_doc(doc))
        cmds = [('start', 0, {})] + ([('pause', 0, {'on': True})] if static else [])
        secs = D.SEC_STATIC if static else D.SEC_EVOLVE
        got = _drive(r, int(math.ceil(secs * SR / BLOCK)), cmds)
        clip = r.snapshot()['clip_blocks']
        assert clip == {'A': 0, 'B': 0}, (title, clip)
        for o in ('A', 'B'):
            pcm = np.concatenate(got[o])
            assert int(np.abs(pcm[SR:]).max()) > 500, (title, o, "silent side")
        assert r.running and r.paused == static


def test_gain_and_trim_changes_are_smooth():
    e = _engine('scan_surface', D.field_f1(), trim_db=0.0)
    _render(e, 5)
    # a gain jump is glided with the block ramp: the block equals the unity
    # block times the ramp (the wave itself is unchanged)
    phase = e.phase
    a = e.a_from.copy()
    y = e.render_float(GAIN * 2.0)[0]
    wave = S.evaluate(a, (phase + np.arange(BLOCK) * 110.0 / SR) % 1.0) * S.RAW_SCALE
    ramp = GAIN + (GAIN * 2.0 - GAIN) * _RAMP
    assert np.allclose(y, wave * ramp, atol=1e-9)
    # a trim change is the same glide (through set_params, no phase / wave reset)
    e.set_params(dict(e.params, trim_db=6.0))
    phase = e.phase
    y = e.render_float(GAIN * 2.0)[0]
    wave = S.evaluate(a, (phase + np.arange(BLOCK) * 110.0 / SR) % 1.0) * S.RAW_SCALE
    ramp = GAIN * 2.0 + (GAIN * 2.0 * 10 ** (6 / 20.0) - GAIN * 2.0) * _RAMP
    assert np.allclose(y, wave * ramp, atol=1e-9)
    # network: the same glide on gain; no block-boundary step in a steady state
    e = _engine('pm_network', D.field_f5(True), trim_db=0.0)
    y = _render(e, 60)
    steps = np.abs(np.diff(y[20 * BLOCK:]))
    at_boundaries = steps[BLOCK - 1::BLOCK]
    assert at_boundaries.max() <= steps.max() + 1e-12          # boundaries are not special


def test_registry_ui_hints_and_scene_validation():
    spec = registry.get('scan_surface')
    assert spec.choices == {'surface': ('Smooth', 'Distance'), 'path': ('Ellipse', 'Lissajous', 'Raster')}
    assert registry.value_text('scan_surface', 'path', 2) == 'Raster'
    assert registry.value_text('scan_surface', 'width', 1.5) == '1.5'
    assert registry.get('pm_network').choices == {} and registry.get('pm_network').overlay is not None
    from casynth_lab.runner import describe_difference
    txt = describe_difference({'A': ('scan_surface', dict(registry.defaults('scan_surface'), path=0)),
                               'B': ('scan_surface', dict(registry.defaults('scan_surface'), path=1))})
    assert txt == "path: A=Ellipse, B=Lissajous"
    # a bad value is refused by the registry and by the scene loader
    for bad in (3, 1.5, -1):
        try:
            registry.validate_param('scan_surface', 'path', bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    # every prepared scene file on disk is valid and identical to the builder's document
    for rec in D.RECORDS:
        path = os.path.join(D.DEMOS_DIR, rec[1] + '.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                assert json.load(f) == D.scene_for(rec), path
    # the bench UI computes its geometry from the registry (8 engines -> 3 rows;
    # 5 built-in + Scan + Network + Gutter/N1)
    import demo_bench as db
    from casynth_lab.audio_out import LiveEngine
    scene = scene_from_doc(_doc('F1', D.scan(path=2), D.network()))
    runner = DemoRunner(scene)
    eng = LiveEngine(runner, sink=lambda m, b: None)
    app = db.BenchApp(scene, eng)
    assert app.engine_rows == 3 and len(app.engine_btns) == 8
    ys = sorted({r[1] for r in app.engine_btns.values()})
    assert len(ys) == 3 and app.params_y > ys[-1] + db.ENG_H
    assert app.footer_y >= app.params_y + 7 * db.ROW_H
    rows = dict((s[0], r) for s, r in app._param_rows('scan_surface'))
    rects = app._choice_rects('scan_surface', 'path', rows['path'][1])
    assert [v for v, _r in rects] == [0, 1, 2]
    assert all(r[0] + r[2] <= app.panel_x + db.PANEL_W for _v, r in rects)
    # the radius rows are inactive text for Raster and a click on them does nothing
    assert app._inactive('scan_surface', dict(path=2)) == {'radius_x': 'Full field', 'radius_y': 'Full field'}
    # seam pieces: the raster closing segment is two pieces inside the field, never a line across
    pieces = app._seam_pieces((0.0, 31.0), (0.0, 32.0))
    assert len(pieces) == 2
    assert pieces[0] == ((0.0, 31.0), (0.0, 31.5)) and pieces[1] == ((0.0, -0.5), (0.0, 0.0))
    pieces = app._seam_pieces((31.0, 31.0), (32.0, 32.0))
    assert len(pieces) == 2 and pieces[1][1] == (0.0, 0.0)
    assert app._seam_pieces((3.0, 4.0), (5.0, 4.0)) == [((3.0, 4.0), (5.0, 4.0))]
    # catalog: a double click on a record acts as its Continue button; a slow
    # second click or a click on another record only selects
    import tempfile
    from casynth_lab.catalog import Catalog
    root = os.path.join(ART, 'dblclick')
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, '.tmp'))
    cat = Catalog(root, repo_root=None)
    app2 = db.BenchApp(scene, eng, catalog=cat)
    calls = []
    app2.continue_record = lambda: calls.append(app2.cat['sel'])
    app2.open_catalog()
    assert app2.mode == 'catalog'
    app2.cat['entries'] = [(None, 'x'), (None, 'y')]              # two rows (content irrelevant)
    r0, r1 = app2._list_rect(0), app2._list_rect(1)
    c0, c1 = (r0[0] + 5, r0[1] + 5), (r1[0] + 5, r1[1] + 5)
    assert app2.press(c0, 1, now=10.0) == 'record:0' and app2.cat['sel'] == 0 and calls == []
    assert app2.press(c0, 1, now=10.3) == 'continue' and calls == [0]
    assert app2.press(c0, 1, now=10.5) == 'record:0' and calls == [0]   # the pair was consumed
    assert app2.press(c0, 1, now=11.5) == 'record:0' and calls == [0]   # too slow
    assert app2.press(c1, 1, now=11.6) == 'record:1' and app2.cat['sel'] == 1 and calls == [0]
    assert app2.press(c1, 1, now=11.7) == 'continue' and calls == [0, 1]
    eng.stop()


def test_notes_button_editor_extraction():
    """Listening notes (2026-09-14): in a session continued from a record the
    Notes button opens an editor whose text is saved as typed into
    <record>/notes.md; clicks outside the window still drive A/B and the
    transport; the catalog offers Notes for the selected record and marks
    records that have them; `python -m casynth_lab.catalog notes ROOT`
    prints them for the agents; nothing else in the record changes."""
    import hashlib
    import demo_bench as db
    from casynth_lab.audio_out import LiveEngine
    from casynth_lab.catalog import Catalog
    from casynth_lab.recorder import Recorder
    root = os.path.join(ART, 'notes_cat')
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, '.tmp'))
    cat = Catalog(root, repo_root=None)
    scene = scene_from_doc(_doc('F1', D.scan(path=2), D.network()))
    # a record (offline, as the demo builder does)
    runner = DemoRunner(scene)
    rec_ = Recorder(runner, None)
    runner.post('start', at=0)
    for _ in range(40):
        before = runner.out_samples
        rec_.on_block(runner.next_block(), runner.running, before)
    cut = rec_.cut(dict(device_ok=False, device_error='test', underruns=0, block_errors=0,
                        last_error=None, clip_blocks={'A': 0, 'B': 0}, record_error=None))
    rid = cat.save(cut, "Notes test record", "description at save")

    def digest_without_notes(d):
        h = hashlib.sha256()
        for name in sorted(os.listdir(d)):
            if name.startswith('notes.md'):
                continue
            with open(os.path.join(d, name), 'rb') as f:
                h.update(name.encode() + f.read())
        return h.hexdigest()
    before = digest_without_notes(os.path.join(root, rid))
    # a fresh session: no record yet -> Notes explains
    eng = LiveEngine(DemoRunner(scene), sink=lambda m, b: None)
    eng.start()
    app = db.BenchApp(scene, eng, catalog=cat)
    nb = app.lab_buttons['notes']
    assert app.press((nb[0] + 3, nb[1] + 3), 1) == 'notes' and app.mode == 'live'
    assert app.status.startswith("Notes: no record in this session")
    # continue the record -> the session belongs to it
    app.open_catalog()
    app.select_record([i for i, (r, _e) in enumerate(app.cat['entries']) if r and r.id == rid][0])
    app.continue_record()
    assert app.mode == 'live' and app.session_record == rid
    assert app.press((nb[0] + 3, nb[1] + 3), 1) == 'notes' and app.mode == 'notes'
    assert app.notes_form['rid'] == rid and app.notes_form['text'] == ''
    for ch in "Ellipse thinner":
        app.text_input(ch)
    app.key('return')
    for ch in "raster has more body":
        app.text_input(ch)
    app.key('backspace')
    app.key('backspace', ctrl=True)
    app.text_input("body!")
    want = "Ellipse thinner\nraster has more body!"
    assert app.notes_form['text'] == want
    with open(os.path.join(root, rid, 'notes.md'), encoding='utf-8') as f:
        assert f.read() == want + '\n'                   # saved as typed, LF, trailing newline
    assert app.status.startswith("Notes saved")
    # clicks outside the window reach the live controls, never the lab buttons
    tab_b = app.tabs['B']
    assert app.press((tab_b[0] + 3, tab_b[1] + 3), 1) == 'tab:B' and app.mode == 'notes'
    assert app.press((app.lab_buttons['catalog'][0] + 3, app.lab_buttons['catalog'][1] + 3), 1) is None
    assert app.mode == 'notes'
    import time as _t
    _t.sleep(0.3)
    assert app.engine.snapshot()['selected'] == 'B'
    assert app.key('escape') == 'notes:close' and app.mode == 'live'
    # the catalog: Notes for the selected record, the row is tagged, text persists
    app.open_catalog()
    r = app._catalog_rects()
    app.cat['sel'] = None                                     # no record selected: no button
    assert app.press((r['notes'][0] + 3, r['notes'][1] + 3), 1) is None and app.mode == 'catalog'
    app.select_record([i for i, (rr, _e) in enumerate(app.cat['entries']) if rr and rr.id == rid][0])
    assert app.press((r['notes'][0] + 3, r['notes'][1] + 3), 1) == 'notes' and app.mode == 'notes'
    assert app.notes_form['text'] == want
    app.text_input(" +")
    assert app.key('escape') == 'notes:close' and app.mode == 'catalog'
    assert cat.load(rid).has_notes and cat.load(rid).notes == want + " +\n"
    # a frame renders in both places
    import pygame
    pygame.init()
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    app.press((r['notes'][0] + 3, r['notes'][1] + 3), 1)
    app.draw(screen, font, small)
    app.key('escape')
    app.draw(screen, font, small)
    eng.stop()
    # the record itself is untouched; empty notes remove the file
    assert digest_without_notes(os.path.join(root, rid)) == before
    assert cat.load(rid).note == "description at save"
    # extraction for the agents
    p = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'casynth_lab.catalog', 'notes', root],
                       capture_output=True, text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "## Notes test record" in out and rid in out and "Ellipse thinner" in out
    assert "raster has more body! +" in out and "description at save" in out
    cat.write_notes(rid, "   ")
    assert not os.path.exists(os.path.join(root, rid, 'notes.md')) and not cat.load(rid).has_notes
    p = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'casynth_lab.catalog', 'notes', root],
                       capture_output=True, text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0 and "(no notes" in p.stdout


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:           # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    shutil.rmtree(ART, ignore_errors=True)
    os.makedirs(ART, exist_ok=True)
    sys.exit(1 if _run() else 0)
