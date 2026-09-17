#!/usr/bin/env python3
"""N1 gates (REQ memory/req-network-ca-n1-2026-09-15.md, "Checks before hand-over"):
the live kernel of casynth_lab/gutter_field.py against the slow model
(demos/network_reference_n0/gutter_network.py, scalar mode) and the scalar
node port (Java-verified), the review-R1 route, the field -> frequency
mapping against the fixtures, freeze / depth invariance, journal determinism,
byte-exact continuation, limits, the real-time budget and the bench hooks.

    python tests/test_gutter_field_n1.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_n1_tests/.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN                                    # noqa: E402
from casynth_engine import step as gol_step                                   # noqa: E402
from casynth_lab import DemoRunner, BLOCK, registry, scene_from_doc, load_scene  # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot          # noqa: E402
from casynth_lab.snapshot import save_state, load_state                      # noqa: E402
from casynth_lab import gutter_field as GF                                   # noqa: E402
from casynth_lab.gutter_field_n1_config import CONFIG                        # noqa: E402
from demos.network_reference_n0 import gutter_network as gn                  # noqa: E402
from demos.network_reference_n0.gutter_node import GutterOsc                 # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_n1_tests')
FIXTURES = os.path.join(ROOT, 'memory', 'research', 'network-n1-fixtures-2026-09-15.json')
SCENE = os.path.join(ROOT, 'demos', 'network_n1_blinkers.json')
OUTPUTS = ('A', 'B', 'monitor')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 2.0)
GAIN = MASTER_GAIN * 0.7
DEFAULTS = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)


def _inline_fixtures():
    """The fields of the fixtures file (memory/research, Researcher's), rebuilt here so the
    gate is self-contained: four blinkers in columns 4 / 12 / 20 / 28 -- vertical (rows
    14..16), horizontal (row 15, the next generation), moved (vertical shifted by 8 rows)."""
    cols = (4, 12, 20, 28)
    vertical = [[r, c] for c in cols for r in (14, 15, 16)]
    horizontal = [[15, c + d] for c in cols for d in (-1, 0, 1)]
    moved = [[r + 8, c] for r, c in vertical]

    def fld(cells, counts):
        n = np.array(counts, np.float64)
        u = n / (n + 2.0)
        return dict(cells=cells, counts=list(counts), u=u.tolist(),
                    ratio=np.power(2.0, 2.0 * u - 1.0).tolist())
    return dict(model_version='gutter_field_n1_v1',
                fields=dict(vertical=fld(vertical, [2.0] * 4 + [1.0] * 4),
                            horizontal=fld(horizontal, [3.0] * 4 + [0.0] * 4),
                            moved=fld(moved, [0.0] * 4 + [3.0] * 4)),
                edit_schedule=[dict(seconds=4 * k, field=f)
                               for k, f in enumerate(('vertical', 'moved', 'vertical', 'moved', 'vertical'))])


def _fx():
    """The fixtures file when present (then it must agree with the inline copy), else the
    inline copy."""
    inline = _inline_fixtures()
    if not os.path.isfile(FIXTURES):
        return inline
    with open(FIXTURES, encoding='utf-8') as f:
        fx = json.load(f)
    for name, fld in inline['fields'].items():
        assert sorted(map(tuple, fx['fields'][name]['cells'])) == sorted(map(tuple, fld['cells'])), name
        assert fx['fields'][name]['counts'] == fld['counts'] and fx['fields'][name]['u'] == fld['u'], name
        assert np.allclose(fx['fields'][name]['ratio'], fld['ratio'], rtol=0, atol=1e-12), name
    assert fx['edit_schedule'] == inline['edit_schedule'] and fx['model_version'] == inline['model_version']
    return fx


def _grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def _field(name):
    return _grid(_fx()['fields'][name]['cells'])


def _engine(grid, config=None, **over):
    params = dict(DEFAULTS)
    params.update(over)
    e = GF.GutterFieldEngine(CTX, params, config=config)
    e.init(grid, None, GAIN)
    return e


def _model(grid, interaction=127, scale=1.0, depth=1.0):
    """The slow model with the SAME messages the engine sends at init."""
    cfg = GF.n1_config()
    cfg['interaction_raw'] = int(interaction)
    net = gn.GutterNetwork(cfg)
    _model_tune(net, grid, scale, depth)
    return net


def _model_tune(net, grid, scale=1.0, depth=1.0, u=None):
    u = GF.field_u(GF.region_counts(grid)) if u is None else u
    ratio = GF.ratio_of(u, scale, depth)
    base = np.array(net.cfg['filters_hz'], np.float64)
    for i in range(8):
        net.set_filters(i, base[i] * ratio[i])


def _raw(e, n_blocks):
    return np.concatenate([e.raw_block() for _ in range(n_blocks)])


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


def _seqsum(values):
    """Index-order accumulation from 0.0 (what the kernel and the scalar model do;
    Python's own sum() has used compensated summation since 3.12, numpy's is pairwise)."""
    acc = 0.0
    for v in values:
        acc += float(v)
    return acc


def _same(g1, g2):
    for o in OUTPUTS:
        a, b = np.concatenate(g1[o]), np.concatenate(g2[o])
        if a.shape != b.shape or not np.array_equal(a, b):
            return False, o
    return True, ''


# =============================================================================
# 1. configuration, mappings, field -> control (fixtures)
# =============================================================================
def test_config_equals_fixtures_and_mappings_equal_n0():
    fx = _fx()
    if 'reference_config' in fx:                     # the Researcher's file is present
        ref = dict(fx['reference_config'])
        ref['output_routes'] = fx['output_routes']
        ref['post_math'] = 'scalar'
        ref['model_version'] = fx['model_version']
        assert CONFIG == ref, "casynth_lab/gutter_field_n1_config.py drifted from the fixtures"
    else:
        print("  (fixtures file absent: config compared to the inline fields only)")
    assert np.array(CONFIG['filters_hz']).shape == (8, 24) and CONFIG['sr'] == 44100.0
    assert CONFIG['matrix_delay_samples'] + CONFIG['extra_feedback_samples'] == 2064
    assert CONFIG['svf_res'] == 0.01 and CONFIG['damp_raw'] == [138] * 8 and CONFIG['interaction_raw'] == 127
    assert CONFIG['output_routes']['R'][5] == 0 and sum(CONFIG['output_routes']['L']) == 8
    assert GF.MODEL_VERSION == 'gutter_field_n1_v1'
    for raw in range(0, 257):
        for a, b in ((GF.map_gain, gn.map_gain), (GF.map_damp, gn.map_damp), (GF.map_mod, gn.map_mod),
                     (GF.map_rate, gn.map_rate), (GF.map_q, gn.map_q), (GF.map_soften, gn.map_soften),
                     (GF.map_interaction, gn.map_interaction)):
            assert a(raw) == b(raw)
    assert GF.map_interaction(127) == (127 / 256.0) ** 2 * 5.0
    # region geometry (REQ): 2 x 4, floor bounds, i = 4 r + c
    assert GF.region_bounds(32, 32) == [(0, 16, 0, 8), (0, 16, 8, 16), (0, 16, 16, 24), (0, 16, 24, 32),
                                        (16, 32, 0, 8), (16, 32, 8, 16), (16, 32, 16, 24), (16, 32, 24, 32)]
    assert GF.region_bounds(7, 10) == [(0, 3, 0, 2), (0, 3, 2, 5), (0, 3, 5, 7), (0, 3, 7, 10),
                                       (3, 7, 0, 2), (3, 7, 2, 5), (3, 7, 5, 7), (3, 7, 7, 10)]
    for name, fld in fx['fields'].items():
        g = _field(name)
        counts = GF.region_counts(g)
        assert counts.tolist() == fld['counts'], name
        u = GF.field_u(counts)
        assert u.tolist() == fld['u'], name
        ratio = GF.ratio_of(u, 1.0, 1.0)
        assert np.allclose(ratio, fld['ratio'], rtol=0, atol=1e-12), name
        e = _engine(g)
        assert np.array_equal(e.u_held, u) and np.array_equal(e.counts, counts)
        f = GF.bank_freqs(np.array(CONFIG['filters_hz']), ratio)
        assert np.array_equal(e.ffreq, f) and f.max() <= 19000.0
        assert np.array_equal(f, f.astype(np.float32).astype(np.float64))     # float32 messages
        assert np.array_equal(e.ffreq / e.base, e.ffreq / e.base)            # no accumulation:
        e.update_field(g, None)                                              # re-applying the same
        assert np.array_equal(e.ffreq, f) and np.array_equal(e.a0, GF.calc_coeffs(f, e.fQ, SR)[0])
    # scale is a plain multiplier; depth 0 makes the field irrelevant; the clip acts
    u = GF.field_u(np.array([0, 1, 2, 5, 10, 30, 100, 1000.0]))
    assert np.allclose(GF.ratio_of(u, 1.0, 0.0), 1.0) and np.allclose(GF.ratio_of(u, 0.7, 0.0), 0.7)
    assert np.allclose(GF.ratio_of(u, 2.0, 1.0), 2.0 ** np.clip(1.0 + 2 * u - 1, -1, 1))
    assert GF.ratio_of([0.0], 0.5, 1.0)[0] == 0.5 and GF.ratio_of([1.0], 2.0, 1.0)[0] == 2.0
    assert GF.bank_freqs([[5514.620971073786]], [4.0])[0, 0] == 19000.0


# =============================================================================
# 2. the live kernel == the slow model (bit-exact) under every kind of intervention
# =============================================================================
def test_kernel_equals_slow_model_bit_exact_static_evolution_edits_params():
    g = _field('vertical')
    e = _engine(g)
    net = _model(g)
    assert np.array_equal(net.ffreq, e.ffreq) and np.array_equal(net.a0, e.a0) \
        and np.array_equal(net.b1, e.b1) and np.array_equal(net.b2, e.b2)
    # a) static field, links ramping in, well past the 2064-sample matrix delay
    n = 12
    fast, (slow, _) = _raw(e, n), net.render(n * BLOCK)
    assert np.abs(fast).max() > 0.05 and np.isfinite(fast).all()
    assert np.array_equal(fast, slow), "static field differs"
    # b) evolution: two generations, the frequencies retuned at the step
    for _ in range(2):
        g = gol_step(g)
        e.update_field(g, None)
        _model_tune(net, g)
        fast, (slow, _) = _raw(e, 4), net.render(4 * BLOCK)
        assert np.array_equal(fast, slow), "evolution differs"
    # c) links 127 -> 200 (50 ms ramp) with the ramp cut by a block boundary
    e.set_params(dict(DEFAULTS, interaction=200))
    net.set_slider('interaction', 200)
    fast, (slow, _) = _raw(e, 9), net.render(9 * BLOCK)
    assert np.array_equal(fast, slow), "interaction ramp differs"
    # d) repeated edits: the field moved by 8 rows and back, no restart
    for name in ('moved', 'vertical', 'moved'):
        g = _field(name)
        e.update_field(g, None)
        _model_tune(net, g)
        fast, (slow, _) = _raw(e, 4), net.render(4 * BLOCK)
        assert np.array_equal(fast, slow), f"edit {name} differs"
    # e) freeze on + scale 0.7 + depth 0.4: manual parameters act on the HELD vector,
    #    a field edit while frozen changes nothing
    held = e.u_held.copy()
    e.set_params(dict(DEFAULTS, interaction=200, freeze_ca=1, scale=0.7, depth=0.4))
    _model_tune(net, None, 0.7, 0.4, u=held)
    e.update_field(_field('horizontal'), None)
    assert np.array_equal(e.u_held, held) and not np.array_equal(e.u_field, held)
    fast, (slow, _) = _raw(e, 4), net.render(4 * BLOCK)
    assert np.array_equal(fast, slow), "frozen + manual differs"
    # f) freeze off: the current field is taken
    e.set_params(dict(DEFAULTS, interaction=200, freeze_ca=0, scale=0.7, depth=0.4))
    _model_tune(net, _field('horizontal'), 0.7, 0.4)
    fast, (slow, _) = _raw(e, 4), net.render(4 * BLOCK)
    assert np.array_equal(fast, slow), "unfreeze differs"
    assert int(e.resets.sum()) == int(net.resets.sum()) == 0
    # g) the compiled kernel == the same function interpreted by Python (no contraction)
    st = e.export_state()
    e2 = GF.GutterFieldEngine(CTX, e.params)
    e2.restore_state(_field('horizontal'), None, st)
    outA, outB = np.zeros((300, 2)), np.zeros((300, 2))
    e._kernel(300, outA)
    fn = getattr(GF._render, 'py_func', GF._render)
    fn(300, outB, e2.node, e2.resets, e2.px1, e2.px2, e2.py1, e2.py2, e2.a0, e2.a1, e2.a2, e2.b1,
       e2.b2, e2.count, e2.lp_a0, e2.lp_b1, e2.hp_ratio, e2.hp_cut, e2.ramps, e2.ramp_left, e2.G,
       e2.G_tgt, e2.G_inc, e2.ints, e2.ring, e2.consts, e2.panL, e2.panR, e2.routeL, e2.routeR,
       e2._scratch)
    assert np.array_equal(outA, outB) and np.array_equal(e.node, e2.node)


def test_kernel_node_equals_scalar_java_port_in_lockstep():
    """Sample by sample: the kernel, the slow model and the scalar node port (verified
    bit-exact against the original gutterOsc.class on a JVM) share every node state."""
    g = _field('vertical')
    e = _engine(g)
    net = _model(g)
    cfg = net.cfg
    oscs = []
    for j in range(8):
        osc = GutterOsc()
        osc.setLowpass(gn.map_soften(cfg['soften_raw'][j]))
        osc.setHighpass(cfg['highpass_hz'])
        osc.filters(cfg['filter_count'][j])
        for i, f in enumerate(e.ffreq[j]):
            osc.setFreqN(i, f)
        for i in range(24):
            osc.setQN(i, gn.map_q(cfg['q_raw'][j]))
        osc.setDistortionMethod(cfg['dist_method'])
        osc.dspsetup(SR)
        oscs.append(osc)
    out = np.zeros((1, 2))
    n = 2600                                        # past the matrix delay: links act
    for k in range(n):
        e._kernel(1, out)
        L, R, o = net.step()
        assert out[0, 0] == _seqsum(L) and out[0, 1] == _seqsum(R), k
        gm, w, c, dt, gain = net.last_inlets
        for j in range(8):
            o0, _ = oscs[j].perform_sample(float(gm[j]), float(w), float(c[j]), float(dt[j]),
                                           float(gain[j]), 0.0)
            assert oscs[j].duffX == e.node[GF.DUFFX, j] == net.duffX[j], (k, j)
            assert oscs[j].duffY == e.node[GF.DUFFY, j] and oscs[j].t == e.node[GF.T, j]
            assert oscs[j].finalY == e.node[GF.FINALY, j]
    assert np.abs(e.ring).max() > 0.0 and np.abs(net.ring).max() > 0.0
    assert np.array_equal(e.ring, net.ring)


# =============================================================================
# 3. review R1: node 6 has no right route, in the master sum AND at the matrix input;
#    a single directed link arrives exactly after the matrix delay
# =============================================================================
def test_r6_route_off_in_master_and_matrix_and_link_arrival():
    for D in (3, 2064):
        cfg = GF.n1_config()
        cfg['matrix_delay_samples'] = D
        cfg['extra_feedback_samples'] = 0
        cfg['matrix'] = np.zeros((8, 8)).tolist()
        cfg['matrix'][5][3] = 0.75                       # only node 6 -> node 4
        cfg['mod_raw'] = [0] * 8                         # no forcing: only node 6 is excited
        cfg['ramp_ms'] = {k: 0.0 for k in cfg['ramp_ms']}
        for routes, label in ((cfg['output_routes'], 'n1'), ({'L': [1] * 8, 'R': [1] * 8}, 'n0')):
            c2 = dict(cfg, output_routes=routes)
            e = GF.GutterFieldEngine(CTX, dict(DEFAULTS, interaction=128, depth=0.0), config=c2)
            e.init(np.zeros((32, 32), np.uint8), None, 0.0)
            e.node[GF.DUFFX, 5] = 0.3
            net = gn.GutterNetwork(dict(c2, interaction_raw=128))
            for s, v in (('mod', 0), ('damp', 138), ('rate', 39), ('gain', 162), ('interaction', 128)):
                net.set_slider(s, v, ramp=False)
            net.duffX[5] = 0.3
            _model_tune(net, np.zeros((32, 32)), 1.0, 0.0)
            out = np.zeros((1, 2))
            first = None
            hist = []
            for k in range(D + 40):
                e._kernel(1, out)
                L, R, o = net.step()
                assert out[0, 0] == _seqsum(L) and out[0, 1] == _seqsum(R)
                if label == 'n1':
                    assert R[5] == 0.0 and out[0, 1] == 0.0, "node 6 must not reach the right master"
                else:
                    assert R[5] != 0.0 or o[5] == 0.0
                assert L[5] == o[5] * net.panL[5]
                m_in5 = 0.5 * (L[5] + R[5])
                assert e.ring[3, k % (D + 1)] == m_in5 * 0.75 == net.ring[3, k % (D + 1)]
                hist.append(m_in5 * 0.75)
                c = net.last_inlets[2]
                expect = np.float32(min(max(gn.map_damp(138) + (hist[k - D] * gn.map_interaction(128)
                                                                 if k >= D else 0.0), 0.0001), 1.0))
                assert c[3] == float(expect), (label, D, k)
                if first is None and k >= 1 and c[3] != np.float32(gn.map_damp(138)):
                    first = k
            assert first == D, (label, D, first)
            assert np.abs(out).max() >= 0.0 and np.abs(hist).max() > 0.0
    # the N1 routes make the two masters different only through node 6: with node 6's
    # left route also off, L and R of the N0 and N1 versions coincide except for node 6
    e_n1 = _engine(_field('vertical'))
    cfg0 = GF.n1_config()
    cfg0['output_routes'] = {'L': [1] * 8, 'R': [1] * 8}
    e_n0 = _engine(_field('vertical'), config=cfg0)
    a, b = _raw(e_n1, 3), _raw(e_n0, 3)
    assert np.array_equal(a[:, 0], b[:, 0]) and not np.array_equal(a[:, 1], b[:, 1])


# =============================================================================
# 4. field control at the engine level: depth 0 / Freeze CA / empty field / init frozen
# =============================================================================
def test_depth_zero_and_freeze_make_edits_inert_empty_field_continues():
    g = _field('vertical')
    for over in (dict(depth=0.0), dict(freeze_ca=1)):
        e1, e2 = _engine(g, **over), _engine(g, **over)
        _raw(e1, 3); _raw(e2, 3)
        coef = (e1.a0.copy(), e1.b1.copy(), e1.b2.copy())
        e2.update_field(_field('moved'), None)
        e2.update_field(np.ones((32, 32), np.uint8), None)
        e2.update_field(np.zeros((32, 32), np.uint8), None)
        assert np.array_equal(e2.a0, coef[0]) and np.array_equal(e2.b1, coef[1]) \
            and np.array_equal(e2.b2, coef[2]), over
        assert np.array_equal(_raw(e1, 6), _raw(e2, 6)), over
    # init with Freeze CA on: the held vector is the initial field's
    e = _engine(_field('moved'), freeze_ca=1)
    assert e.frozen and np.array_equal(e.u_held, GF.field_u(GF.region_counts(_field('moved'))))
    e.update_field(_field('vertical'), None)
    assert np.array_equal(e.u_held, GF.field_u(GF.region_counts(_field('moved'))))
    assert np.array_equal(e.u_field, GF.field_u(GF.region_counts(_field('vertical'))))
    # off -> the current field is taken at once
    e.set_params(dict(DEFAULTS, freeze_ca=0))
    assert np.array_equal(e.u_held, e.u_field)
    # on again during play holds what is applied now, and scale / depth still act on it
    e.set_params(dict(DEFAULTS, freeze_ca=1))
    held = e.u_held.copy()
    e.update_field(np.ones((32, 32), np.uint8), None)
    e.set_params(dict(DEFAULTS, freeze_ca=1, scale=2.0))
    assert np.array_equal(e.u_held, held)
    assert np.allclose(e.ratio, GF.ratio_of(held, 2.0, 1.0))
    # empty field: n = 0 everywhere, the sound continues from the current state (no reset)
    e = _engine(g)
    before = _raw(e, 3)
    e.update_field(np.zeros((32, 32), np.uint8), None)
    assert e.counts.tolist() == [0.0] * 8 and e.u_held.tolist() == [0.0] * 8
    assert np.allclose(e.ratio, 0.5)
    after = _raw(e, 3)
    assert np.isfinite(after).all() and np.abs(after).max() > 0.0 and int(e.resets.sum()) == 0
    assert np.abs(before).max() > 0.0
    # display is plain numbers
    d = e.display()
    assert d['counts'] == [0] * 8 and d['frozen'] is False and d['model'] == GF.MODEL_VERSION
    json.dumps(d)
    # depth 0 -> every ratio equals scale, whatever the field
    e = _engine(np.ones((32, 32), np.uint8), depth=0.0, scale=1.5)
    assert np.allclose(e.ratio, 1.5)


# =============================================================================
# 5. bench contract on the prepared scene: journal determinism, continuation, hygiene
# =============================================================================
def _scene():
    return load_scene(SCENE)


def _ab_commands(base):
    return [('set_cell', base + 5 * BLOCK, {'r': 3, 'c': 4, 'v': 1}),
            ('set_cell', base + 5 * BLOCK, {'r': 3, 'c': 5, 'v': 1}),
            ('select', base + 20 * BLOCK, {'side': 'B'}),
            ('set_param', base + 40 * BLOCK, {'side': 'A', 'name': 'interaction', 'value': 200}),
            ('set_param', base + 41 * BLOCK, {'side': 'B', 'name': 'scale', 'value': 0.8}),
            ('vol', base + 60 * BLOCK, {'value': 0.5}),
            ('pause', base + 80 * BLOCK, {'on': True}),
            ('set_cell', base + 90 * BLOCK, {'r': 14, 'c': 4, 'v': 0}),
            ('set_param', base + 95 * BLOCK, {'side': 'A', 'name': 'freeze_ca', 'value': 1}),
            ('pause', base + 100 * BLOCK, {'on': False}),
            ('set_param', base + 110 * BLOCK, {'side': 'A', 'name': 'freeze_ca', 'value': 0}),
            ('select', base + 120 * BLOCK, {'side': 'A'})]


def test_same_journal_same_output_and_continuation_byte_exact():
    os.makedirs(ART, exist_ok=True)
    scene = _scene()
    pre = [('start', 0, {}), ('set_cell', 30 * BLOCK, {'r': 1, 'c': 1, 'v': 1}),
           ('set_param', 60 * BLOCK, {'side': 'B', 'name': 'depth', 'value': 0.5})]
    r1, r2 = DemoRunner(scene), DemoRunner(scene)
    g1, g2 = _drive(r1, 300, list(pre)), _drive(r2, 300, list(pre))      # > 2 generations
    ok, o = _same(g1, g2)
    assert ok, f"same journal, different {o}"
    assert r1.gen >= 2
    assert max(int(np.abs(np.concatenate(g1[o])).max()) for o in OUTPUTS) > 3000
    # cut in the middle of the 50 ms links ramp + right after an edit
    r1.post('set_param', at=None, side='A', name='interaction', value=180)
    r1.next_block()
    r1.post('set_cell', at=None, r=2, c=2, v=1)
    r1.next_block()
    st = r1.export_state()
    assert all(supports_snapshot(r1.sides[s].engine) for s in 'AB')
    sA = st['sides']['A']['engine']
    assert 0 < int(sA['ramp_left'][GF.R_GINT]) < int(round(0.05 * SR))
    assert sA['model_version'] == GF.MODEL_VERSION and sA['ring'].shape == (8, 2065)
    for key in ('node', 'ring', 'ramps', 'a0', 'b2', 'ffreq', 'u_held', 'px1', 'py2', 'ints'):
        assert isinstance(sA[key], np.ndarray), key
    base = r1.out_samples
    save_state(ART, 'n1_state', st)
    r3 = DemoRunner.from_state(load_state(ART, 'n1_state'))
    h1 = _drive(r1, 200, _ab_commands(base))
    h3 = _drive(r3, 200, _ab_commands(base))
    ok, o = _same(h1, h3)
    assert ok, f"continuation differs on {o}"
    assert r1.gen == r3.gen and np.array_equal(r1.grid, r3.grid)
    # fresh process: restore from the files and render the same commands
    out = os.path.join(ART, 'n1_fresh.npz')
    code = ("import sys, numpy as np; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import test_gutter_field_n1 as T\n"
            "from casynth_lab import DemoRunner\nfrom casynth_lab.snapshot import load_state\n"
            "r = DemoRunner.from_state(load_state(%r, 'n1_state'))\n"
            "g = T._drive(r, 200, T._ab_commands(%d))\n"
            "np.savez(%r, **{o: np.concatenate(g[o]) for o in T.OUTPUTS})\n"
            % (ROOT, os.path.dirname(os.path.abspath(__file__)), ART, base, out))
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    z = np.load(out)
    for o in OUTPUTS:
        assert np.array_equal(z[o], np.concatenate(h1[o])), f"fresh process differs on {o}"
    # the exported state does not alias the live engine
    st2 = r1.export_state()
    keep = {k: v.copy() for k, v in st2['sides']['A']['engine'].items() if isinstance(v, np.ndarray)}
    r1.sides['A'].engine.ring[:] = 9.0
    r1.sides['A'].engine.node[:] = 1.0
    for k, v in keep.items():
        assert np.array_equal(st2['sides']['A']['engine'][k], v), k
    # rejections before anything is replaced
    import copy

    def bad(mut, what):
        s2 = copy.deepcopy(st)
        mut(s2)
        try:
            DemoRunner.from_state(s2)
        except ValueError as e:
            assert what in str(e), (what, str(e))
            return
        raise AssertionError(f"accepted a bad snapshot: {what}")
    bad(lambda s: s['sides']['A']['engine'].update(version=99), 'version')
    bad(lambda s: s['sides']['A']['engine'].update(model_version='gutter_field_n1_v0'), 'model version')
    bad(lambda s: s['sides']['A']['engine'].update(ring=np.zeros((8, 2001))), 'ring')
    bad(lambda s: s['sides']['A']['engine'].update(delay=2000), 'matrix delay')
    bad(lambda s: s['sides']['B']['engine'].update(engine_id='pm_network'), 'state is for')
    bad(lambda s: s['sides']['B']['engine'].update(node=np.full((GF.NS, 8), np.nan)), 'finite')
    bad(lambda s: s['sides']['B']['engine'].update(gain_prev=float('inf')), 'scalar')
    bad(lambda s: s['sides']['B']['engine']['params'].pop('depth'), 'params')
    DemoRunner.from_state(st)


# =============================================================================
# 6. limits: extreme knobs x fields, no NaN / Inf, the clip counter, the prepared scene
# =============================================================================
def test_limits_extremes_no_nan_clip_counter_and_prepared_scene():
    empty = np.zeros((32, 32), np.uint8)
    full = np.ones((32, 32), np.uint8)
    sparse = np.zeros((32, 32), np.uint8)
    sparse[7, 9] = 1
    resets = {}
    for grid_name, grid in (('empty', empty), ('full', full), ('sparse', sparse), ('vertical', _field('vertical'))):
        for over in (dict(scale=0.5, depth=1.0, interaction=0), dict(scale=2.0, depth=1.0, interaction=256),
                     dict(scale=2.0, depth=0.0, interaction=127), dict(scale=0.5, depth=0.0, interaction=256),
                     dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=1)):
            e = _engine(grid, **over)
            peaks = []
            for _ in range(40):
                buf, peak, n_clip = e.render(GAIN, 0)
                assert buf.dtype == np.int16 and buf.shape == (BLOCK, 2) and np.isfinite(peak)
                assert (n_clip > 0) == (peak > 1.0)
                peaks.append(peak)
            assert np.isfinite(e.node).all() and np.isfinite(e.ring).all()
            assert np.isfinite(e.ffreq).all() and e.ffreq.max() <= 19000.0
            resets[(grid_name, tuple(sorted(over.items())))] = int(e.resets.sum())
            assert max(peaks) < 1.0, (grid_name, over, max(peaks))
    assert sum(resets.values()) == 0, resets
    # the clip counter works: an absurd gain clips and is counted, values are clipped to int16
    e = _engine(_field('vertical'))
    _raw(e, 30)
    buf, peak, n_clip = e.render(GAIN * 100.0, 0)
    assert peak > 1.0 and n_clip > 0 and int(np.abs(buf).max()) == 32767
    assert n_clip == int(np.count_nonzero(np.abs(buf) == 32767))
    buf, peak, n_clip = e.render(0.0, 0)                 # the ramp block still clips early on
    assert np.abs(buf[-1]).max() == 0 and n_clip < BLOCK * 2
    buf, peak, n_clip = e.render(0.0, 0)
    assert np.abs(buf).max() == 0 and n_clip == 0 and peak == 0.0
    # the prepared scene: both sides sound, no clipping, identical first blocks (same start)
    scene = _scene()
    assert scene.variants['A'][0] == scene.variants['B'][0] == 'gutter_field'
    assert scene.variants['A'][1] == dict(DEFAULTS) and scene.variants['B'][1] == dict(DEFAULTS, freeze_ca=1)
    assert scene.rate_hz == 2.0 and (scene.rows, scene.cols) == (32, 32) and len(scene.cells) == 12
    assert sorted(scene.cells) == sorted((r, c) for r, c in _fx()['fields']['vertical']['cells'])
    r = DemoRunner(scene)
    g = _drive(r, 130, [('start', 0, {})])          # > 1 s: two generations
    for o in ('A', 'B'):
        pcm = np.concatenate(g[o])
        assert np.abs(pcm).max() > 3000, o
    assert r.snapshot()['clip_blocks'] == {'A': 0, 'B': 0}
    a, b = np.concatenate(g['A']), np.concatenate(g['B'])
    first_step = int(SR / scene.rate_hz) // BLOCK * BLOCK
    assert np.array_equal(a[:first_step], b[:first_step])       # identical until the first step
    assert not np.array_equal(a, b)                              # then A follows the field
    assert r.gen >= 2 and GF.region_counts(r.grid).sum() == 12   # blinkers keep 12 cells
    with open(os.path.join(ROOT, 'run_network_n1.bat'), encoding='ascii') as f:
        bat = f.read()
    # The public launcher now opens the corrected hypothesis set; the old scene
    # above remains a diagnostic fixture for the original engine.
    assert 'n1h_rhythm.json' in bat and '--catalog' in bat and 'network_n1_hypotheses_2026_09_16_r2' in bat
    assert '--live' not in bat                    # listening goes through the catalog screen


# =============================================================================
# 7. real-time budget: both sides of the scene per block, well under 352 / 44100 s
# =============================================================================
def test_realtime_budget_two_sides():
    scene = _scene()
    r = DemoRunner(scene)
    _drive(r, 30, [('start', 0, {})], collect=False)              # warm-up
    ts = []
    for i in range(300):
        if i % 25 == 0:
            r.post('set_cell', at=None, r=(i // 25) % 32, c=(i // 3) % 32, v=1)
        if i == 150:
            r.post('set_param', at=None, side='A', name='interaction', value=200)
        t0 = time.perf_counter()
        r.next_block()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    budget = BLOCK / SR * 1e3
    assert np.mean(ts) < budget / 2, f"mean {np.mean(ts):.2f} ms per block (budget {budget:.2f})"
    assert np.percentile(ts, 95) < budget, f"p95 {np.percentile(ts, 95):.2f} ms"
    assert GF.HAVE_NUMBA, "the live bench needs numba (requirements.txt)"


# =============================================================================
# 8. bench hooks: registry, overlay, display panel, headless draw
# =============================================================================
def test_registry_overlay_display_and_headless_draw():
    spec = registry.get('gutter_field')
    assert spec.label == 'Gutter' and [p[0] for p in spec.params] == ['scale', 'depth', 'interaction', 'freeze_ca']
    assert spec.spec_of('interaction')[4] and spec.spec_of('freeze_ca')[2:5] == (0, 1, True)
    assert registry.defaults('gutter_field') == DEFAULTS
    for name, bad in (('scale', 0.4), ('scale', 2.5), ('depth', -0.1), ('interaction', 257),
                      ('interaction', 1.5), ('freeze_ca', 2)):
        try:
            registry.validate_param('gutter_field', name, bad)
            raise AssertionError((name, bad))
        except ValueError:
            pass
    ov = spec.overlay(DEFAULTS, 32, 32)
    assert len(ov['lines']) == 4 and len(ov['labels']) == 8 and ov['text']
    assert ((7.5, -0.5), (7.5, 31.5)) in ov['lines'] and ((-0.5, 15.5), (31.5, 15.5)) in ov['lines']
    assert [t for _x, _y, t in ov['labels']] == [str(i) for i in range(8)]
    assert ov['labels'][0][:2] == (3.5, 7.5) and ov['labels'][7][:2] == (27.5, 23.5)
    from casynth_lab.runner import describe_difference
    assert describe_difference({'A': ('gutter_field', dict(DEFAULTS)),
                                'B': ('gutter_field', dict(DEFAULTS, freeze_ca=1))}) == "freeze_ca: A=0, B=1"
    import pygame
    import demo_bench as db
    from casynth_lab.audio_out import LiveEngine
    scene = _scene()
    runner = DemoRunner(scene)
    eng = LiveEngine(runner, sink=lambda m, b: None)
    pygame.init()
    app = db.BenchApp(scene, eng)
    assert 'gutter_field' in app.engine_btns and app.engine_rows == 4   # 4 engines per row since 2026-09-17
    data = app._overlay_runs('gutter_field', DEFAULTS)
    assert len(data['lines']) == 4 and len(data['labels']) == 8 and data['runs'] == []
    screen = pygame.Surface((app.width, app.height))
    font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    try:
        app.draw(screen, font, small)
        snap = eng.snapshot()
        disp = snap['display']['A']
        assert disp['u'] == _fx()['fields']['vertical']['u'] and disp['frozen'] is False
        assert snap['display']['B']['frozen'] is True
        rows = dict((s[0], r) for s, r in app._param_rows('gutter_field'))
        sx, sy, sw, sh = rows['freeze_ca']
        assert app.press((sx + 5, sy + 3), 1) == 'param:freeze_ca'
        app.release()
        t0 = time.time()
        while time.time() - t0 < 5 and not eng.snapshot()['sides']['A'][1]['freeze_ca']:
            time.sleep(0.02)
        assert eng.snapshot()['sides']['A'][1]['freeze_ca'] == 1
        app.draw(screen, font, small)
    finally:
        eng.stop()
        pygame.quit()


# =============================================================================
# 9. the prepared listening material: scene files == builder documents, a record
#    built offline replays exactly and continues, the N0-routes control differs on the right
# =============================================================================
def test_listening_catalog_builder_scenes_records_and_routes_control():
    sys.path.insert(0, os.path.join(ROOT, 'demos'))
    import build_n1_demos as B
    from casynth_lab.catalog import Catalog
    assert [r[0] for r in B.RECORDS] == ['network_n1_blinkers', 'n1_edits', 'n1_depth', 'n1_links', 'n1_routes']
    for rec in B.RECORDS:
        path = os.path.join(B.DEMOS_DIR, rec[0] + '.json')
        with open(path, encoding='utf-8') as f:
            assert json.load(f) == B.scene_for(rec), path
        scene_from_doc(B.scene_for(rec))
    assert sorted(map(tuple, B.field_vertical())) == sorted(map(tuple, _fx()['fields']['vertical']['cells']))
    assert sorted(map(tuple, B.field_vertical(8))) == sorted(map(tuple, _fx()['fields']['moved']['cells']))
    cmds = B.edit_commands()
    ats = sorted({at for _k, at, _a in cmds if _k == 'set_cell'})
    assert ats == [int(round(4 * k * SR)) for k in range(1, 5)] and cmds[0][0] == 'pause'
    # the N0-routes control: same engine class, N0 routing, its own model version; the left
    # master is identical to N1, the right differs; the snapshot refuses the other id
    e1, e0 = _engine(_field('vertical')), None
    params = dict(DEFAULTS)
    e0 = registry.create('gutter_field_n0r', CTX, params)
    e0.init(_field('vertical'), None, GAIN)
    assert e0.model_version == GF.MODEL_VERSION_N0R and e0.routeR.tolist() == [1.0] * 8
    a, b = _raw(e1, 3), _raw(e0, 3)
    assert np.array_equal(a[:, 0], b[:, 0]) and not np.array_equal(a[:, 1], b[:, 1])
    st = e0.export_state()
    assert st['engine_id'] == 'gutter_field_n0r'
    try:
        e1.restore_state(_field('vertical'), None, st)
        raise AssertionError("N1 engine accepted an N0-routes state")
    except ValueError as err:
        assert 'state is for' in str(err)
    # a short offline record in a scratch catalog: saved, replays exactly, continues
    root = os.path.join(ART, 'n1_catalog')
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, '.tmp'))
    cat = Catalog(root, repo_root=None)
    rec = B.RECORDS[1]                                    # the edit schedule (commands)
    rid, snap = B.record_offline(cat, B.scene_for(rec), 4.5, B.commands_for('edits'), rec[1], rec[2])
    assert snap['paused'] and snap['running'] and snap['gen'] == 0
    r = cat.load(rid)
    assert r.title == rec[1] and cat.ids() == [rid]
    res = cat.replay(rid, yield_cpu=False)
    assert res.status == 'match', (res.status, res.reason)
    runner, state, rec2 = cat.continue_runner(rid)
    assert rec2.id == rid and GF.region_counts(runner.grid).tolist() == [0.0] * 4 + [3.0] * 4
    _drive(runner, 5, [], collect=False)
    # an existing catalog with records is refused, an empty one (only .tmp) accepted
    try:
        B.build(root)
        raise AssertionError("built into a catalog that holds records")
    except SystemExit as err:
        assert 'already holds records' in str(err)


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  PASS  {t.__name__}  ({time.time() - t0:.1f} s)")
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
