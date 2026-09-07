#!/usr/bin/env python3
"""S1 + S2 acceptance tests for the demo bench (casynth_lab + demo_bench).

S1: repeatability, live-vs-offline event equality, clocks/pause/reset/stop,
silence, stereo, scene validation, headless UI smoke, no-device state.
S2: five engines from the registry, per-side param memory, A=B synchrony,
side independence, live/offline with side/param/engine changes, monitor
crossfade, v2 validation, CLI --side.

    python tests/test_demo_lab.py      # stdlib runner, no pytest needed
"""
import json
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, VOL_DEFAULT                          # noqa: E402
from casynth_core import ENGINES                                    # noqa: E402
from casynth_lab import (load_scene, SceneError, DemoRunner, BLOCK,  # noqa: E402
                         render_offline, describe_difference, engine_defaults)
from casynth_lab.scene import validate                              # noqa: E402
from casynth_lab.runner import XFADE_SAMPLES                        # noqa: E402
from casynth_lab.audio_out import LiveEngine                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demos", "laplace_basic.json")
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

SCENE = load_scene(DEMO)
SCENE_AB = load_scene(DEMO_AB)
ALL = ('A', 'B', 'monitor')


def _sec(s):
    return int(round(s * SR))


# The S1 event scenario (times in output samples): start, pause for 1 s,
# edit one cell during the pause, resume, change the level.
SCENARIO = [
    ('start', 0, {}),
    ('pause', _sec(1.0), {'on': True}),
    ('set_cell', _sec(1.5), {'r': 3, 'c': 3, 'v': 1}),
    ('pause', _sec(2.0), {'on': False}),
    ('vol', _sec(2.5), {'value': 0.4}),
]
SCEN_SECONDS = 4.0

# The S2 scenario: side switch, harm on B, engine change on B, cell edit, pause.
SCENARIO_AB = [
    ('start', 0, {}),
    ('select', _sec(0.8), {'side': 'B'}),
    ('set_param', _sec(1.4), {'side': 'B', 'name': 'harm', 'value': 0.5}),
    ('set_engine', _sec(2.0), {'side': 'B', 'engine_id': 'fft2d'}),
    ('set_cell', _sec(2.4), {'r': 3, 'c': 3, 'v': 1}),
    ('pause', _sec(2.8), {'on': True}),
    ('select', _sec(3.2), {'side': 'A'}),
    ('pause', _sec(3.6), {'on': False}),
]
SCEN_AB_SECONDS = 4.5


def _live_render(scene, commands, seconds, batches=1, sleep_between=0.0):
    """Run a scenario through LiveEngine's command queue with a test sink
    paced at real time (like a device), posting commands in `batches` groups
    (= different 'UI update rates'; every group lands before its events' times).
    Returns ({'A','B','monitor'} -> pcm, runner)."""
    n = _sec(seconds)
    runner = DemoRunner(scene)
    got = {o: [] for o in ALL}
    got_n = [0]
    done = threading.Event()
    t0 = [None]

    def sink(mon, blk):
        if t0[0] is None:
            t0[0] = time.perf_counter()
        if got_n[0] < n:
            got['monitor'].append(mon)
            got['A'].append(blk.A)
            got['B'].append(blk.B)
            got_n[0] += len(mon)
            if got_n[0] >= n:
                done.set()
        ahead = got_n[0] / SR - (time.perf_counter() - t0[0])
        if ahead > 0.04:
            time.sleep(ahead - 0.04)

    eng = LiveEngine(runner, sink=sink)
    groups = [commands[i::batches] for i in range(batches)]
    for kind, at, args in groups[0]:
        eng.post(kind, at=at, **args)
    eng.start()
    try:
        for g in groups[1:]:
            if sleep_between:
                time.sleep(sleep_between)
            for kind, at, args in g:
                eng.post(kind, at=at, **args)
        assert done.wait(timeout=90), "live render timed out"
    finally:
        eng.stop()
    return {o: np.concatenate(got[o], axis=0)[:n] for o in ALL}, runner


def _events(runner):
    return [(t, k, a) for (t, s, k, a) in runner.journal if k != 'step']


# =============================================================================
# scene
# =============================================================================
def test_scene_loads_and_matches_spec():
    assert SCENE.rows == 32 and SCENE.cols == 32
    assert sorted(SCENE.cells) == sorted([(14, 15), (14, 16), (15, 14), (15, 15), (16, 15)])
    assert SCENE.rate_hz == 4 and SCENE.engine_id == 'laplacian' and SCENE.f0_hz == 220
    assert SCENE.engine_params == dict(n=12, spread=0, alpha=1, shape=0, harm=0,
                                       fullshape=1, dyn=0)
    assert SCENE.initial_grid().sum() == 5
    # v1 loads as A = B
    assert SCENE.variants['A'] == SCENE.variants['B'] and SCENE.initial_side == 'A'


def test_scene_v2_loads():
    s = SCENE_AB
    assert s.format == 2 and s.initial_side == 'A' and s.listen
    assert s.variants['A'][0] == 'laplacian' and s.variants['B'][0] == 'laplacian'
    assert s.variants['A'][1]['harm'] == 0 and s.variants['B'][1]['harm'] == 1
    assert s.rows == SCENE.rows and s.cells == SCENE.cells and s.f0_hz == SCENE.f0_hz


def _doc(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _expect_error(path, mutate, needle):
    d = _doc(path)
    mutate(d)
    try:
        validate(d)
    except SceneError as e:
        assert needle in str(e), f"message {e!r} lacks {needle!r}"
        return
    raise AssertionError(f"no SceneError for {needle}")


def test_scene_rejects_unknown_engine():
    _expect_error(DEMO, lambda d: d.update(engine_id='nope'), "unknown engine_id")


def test_scene_rejects_unknown_and_missing_params():
    _expect_error(DEMO, lambda d: d['engine_params'].update(zzz=1), "unknown engine_params")
    _expect_error(DEMO, lambda d: d['engine_params'].pop('harm'), "missing engine_params")


def test_scene_rejects_bad_cells_and_bad_json():
    _expect_error(DEMO, lambda d: d['cells'].append([32, 0]), "outside")
    _expect_error(DEMO, lambda d: d['cells'].append([1]), "must be [row, col]")
    bad = os.path.join(ART, "_bad_scene.json")
    with open(bad, 'w') as f:
        f.write("{not json")
    try:
        load_scene(bad)
    except SceneError as e:
        assert "invalid JSON" in str(e)
    else:
        raise AssertionError("bad JSON accepted")
    try:
        load_scene(os.path.join(ART, "_missing.json"))
    except SceneError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("missing file accepted")


def test_scene_v2_rejects_bad_variants():
    _expect_error(DEMO_AB, lambda d: d.pop('variants'), "missing key 'variants'")
    _expect_error(DEMO_AB, lambda d: d['variants'].pop('B'), "variants.B missing")
    _expect_error(DEMO_AB, lambda d: d['variants'].update(C={}), "unknown variants")
    _expect_error(DEMO_AB, lambda d: d['variants']['B'].update(engine_id='zzz'),
                  "unknown variants.B.engine_id")
    _expect_error(DEMO_AB, lambda d: d['variants']['B']['engine_params'].update(harm=2),
                  "variants.B.engine_params.harm=2 outside")
    _expect_error(DEMO_AB, lambda d: d['variants']['A']['engine_params'].pop('n'),
                  "missing variants.A.engine_params")
    _expect_error(DEMO_AB, lambda d: d.update(initial_side='C'), "initial_side")
    _expect_error(DEMO_AB, lambda d: d.update(format=3), "'format' must be")


# =============================================================================
# repeatability / v1 compatibility
# =============================================================================
def test_offline_repeat_and_reset_identical_8s():
    a, ra = render_offline(SCENE, 8.0)
    b, _ = render_offline(SCENE, 8.0)
    assert a.shape == (_sec(8.0), 2) and a.dtype == np.int16
    assert np.array_equal(a, b), "two renders differ"
    assert not np.isnan(a.astype(float)).any()
    assert ra.snapshot()['clip_blocks'] == {'A': 0, 'B': 0}
    assert np.abs(a).max() < 32767
    c, _ = render_offline(SCENE, 8.0, commands=[('reset', None, {})], runner=ra)
    assert np.array_equal(a, c), "render after reset differs"
    assert ra.gen == 31


def test_v1_render_preserved_against_s1_reference():
    """The S1 offline PCM of laplace_basic must survive the A/B refactor.
    Reference = sha256 of the 8 s render recorded at S1 acceptance."""
    import hashlib
    a, _ = render_offline(SCENE, 8.0)
    h = hashlib.sha256(a.tobytes()).hexdigest()[:16]
    ref_path = os.path.join(ROOT, "tests", "golden", "demo_lab_s1_ref.sha256")
    with open(ref_path) as f:
        ref = f.read().strip()
    assert h == ref, f"laplace_basic PCM changed: {h} != {ref}"


def test_stereo_identical_channels_and_nonsilent():
    a, _ = render_offline(SCENE, 2.0)
    assert a.ndim == 2 and a.shape[1] == 2
    assert np.array_equal(a[:, 0], a[:, 1])
    assert np.abs(a).max() > 100, "control scene is silent"


# =============================================================================
# S1 events: live queue vs offline
# =============================================================================
def test_live_equals_offline_on_scenario():
    off, r_off = render_offline(SCENE, SCEN_SECONDS, commands=SCENARIO, output='monitor')
    live, r_live = _live_render(SCENE, SCENARIO, SCEN_SECONDS)
    assert np.array_equal(off, live['monitor']), "live PCM != offline PCM"
    assert np.array_equal(r_off.grid, r_live.grid), "final fields differ"
    assert _events(r_off) == _events(r_live), "journals differ"
    assert any(k == 'set_cell' for (_t, k, _a) in _events(r_off))


def test_ui_update_rate_does_not_change_result():
    ref, _ = render_offline(SCENE, SCEN_SECONDS, commands=SCENARIO, output='monitor')
    slow, _ = _live_render(SCENE, SCENARIO, SCEN_SECONDS, batches=3, sleep_between=0.05)
    assert np.array_equal(ref, slow['monitor'])


def test_events_apply_at_first_block_boundary_not_earlier():
    r = DemoRunner(SCENE)
    at = 1000
    r.post('start', at=0)
    r.post('vol', at=at, value=0.2)
    for _ in range(10):
        r.next_block()
    t_vol = [t for (t, s, k, a) in r.journal if k == 'vol'][0]
    assert t_vol >= at and t_vol - at < BLOCK and t_vol % BLOCK == 0, t_vol
    order = [(s, k) for (t, s, k, a) in r.journal if k != 'step']
    assert order == [(1, 'start'), (2, 'vol')]


# =============================================================================
# clocks / pause / reset / stop
# =============================================================================
def test_tempo_from_sample_count_within_one_block():
    _, r = render_offline(SCENE, 8.0)
    steps = [t for (t, s, k, a) in r.journal if k == 'step']
    assert len(steps) == 31
    for i, t in enumerate(steps):
        nominal = (i + 1) * SR / SCENE.rate_hz
        assert 0 <= t - nominal < BLOCK, (i, t, nominal)


def test_pause_freezes_ca_and_does_not_catch_up():
    cmds = [('start', 0, {}), ('pause', _sec(1.1), {'on': True}),
            ('pause', _sec(3.1), {'on': False})]
    pcm, r = render_offline(SCENE, 4.1, commands=cmds)
    steps = [t for (t, s, k, a) in r.journal if k == 'step']
    in_pause = [t for t in steps if _sec(1.1) < t <= _sec(3.1)]
    assert not in_pause, in_pause
    assert r.gen == 8, r.gen
    after = [t for t in steps if t > _sec(3.1)]
    assert len(after) == 4, after
    assert 0.1 * SR < after[0] - _sec(3.1) < 0.2 * SR, after[0]
    seg = pcm[_sec(1.5):_sec(2.5)]
    assert np.abs(seg).max() > 100          # sound kept running during the pause


def test_start_is_silent_and_frozen_until_started():
    r = DemoRunner(SCENE)
    for _ in range(200):
        b = r.next_block()
        assert not b.monitor.any() and not b.A.any() and not b.B.any()
    assert r.gen == 0 and r.ca_samples == 0 and r.t_samples == 0


def test_reset_clears_tails_phases_and_queue():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(600):
        r.next_block()
    sa = r.sides['A']
    assert sa.amp_cur.max() > 0 and sa.phase.any()
    r.post('vol', at=r.out_samples + 10 * BLOCK, value=0.1)      # future -> dropped
    r.post('reset')
    r.next_block()
    assert r.gen == 0 and r.ca_samples == BLOCK and r.running and not r.paused
    assert not r._pending, "queue not cleared by reset"
    assert np.array_equal(r.grid, SCENE.initial_grid())
    r2 = DemoRunner(SCENE)
    r2.post('start', at=0)
    for _ in range(600):
        r2.next_block()
    r2.post('reset')
    b_reset = r2.next_block()
    fresh = DemoRunner(SCENE)
    fresh.post('start', at=0)
    assert np.array_equal(b_reset.monitor, fresh.next_block().monitor)
    assert r.vol == VOL_DEFAULT


def test_stop_returns_to_initial_silent_scene():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(600):
        r.next_block()
    assert r.running and r.gen > 0
    r.post('vol', at=r.out_samples + 10 * BLOCK, value=0.1)      # future -> dropped
    r.post('stop')
    b = r.next_block()
    assert not r.running and r.gen == 0 and r.ca_samples == 0 and not b.monitor.any()
    assert not r._pending
    assert np.array_equal(r.grid, SCENE.initial_grid())
    r.post('start')
    fresh = DemoRunner(SCENE)
    fresh.post('start', at=0)
    for _ in range(50):
        assert np.array_equal(r.next_block().monitor, fresh.next_block().monitor)


def test_stop_and_restart_keep_settings_but_reset_both_sides():
    r = DemoRunner(SCENE_AB)
    r.post('start', at=0)
    r.post('select', side='B')
    r.post('set_param', side='B', name='alpha', value=0.5)
    r.post('set_engine', side='A', engine_id='walsh')
    r.post('vol', value=0.3)
    for _ in range(300):
        r.next_block()
    for kind in ('stop', 'reset'):
        r.post(kind)
        r.next_block()
        s = r.snapshot()
        assert s['selected'] == 'B' and s['vol'] == 0.3
        assert s['sides']['A'][0] == 'walsh' and s['sides']['B'][1]['alpha'] == 0.5
        assert s['gen'] == 0 and s['running'] == (kind == 'reset')
        if kind == 'stop':                      # silent: no phases/tails survive
            for side in r.sides.values():
                assert side.amp_cur.max() == 0 and not side.phase.any()
        assert not r._pending


def test_empty_field_goes_silent_after_tails():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(50):
        r.next_block()
    for (rr, cc) in np.argwhere(r.grid > 0):
        r.post('set_cell', r=int(rr), c=int(cc), v=0)
    tail = [r.next_block() for _ in range(400)]         # ~3.2 s
    assert r.grid.sum() == 0
    assert not tail[-1].monitor.any(), "empty field still sounds"
    assert r.sides['A'].amp_cur.max() < 1e-4


def test_volume_change_is_smoothed():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(100):
        r.next_block()
    r.post('vol', value=0.0)
    b = r.next_block().monitor.astype(float)
    assert abs(b[0, 0]) > 0 or abs(b[1, 0]) > 0
    assert b[-1, 0] == 0 and b[-1, 1] == 0


# =============================================================================
# S2: engines / params
# =============================================================================
def test_every_engine_runs_on_the_demo_scene():
    for e in ENGINES:
        r = DemoRunner(SCENE_AB)
        r.post('set_engine', side='B', engine_id=e['id'])
        r.post('start', at=0)
        pcm, r = render_offline(SCENE_AB, 2.0, commands=[], runner=r, output='B')
        assert r.sides['B'].engine_id == e['id']
        if e['id'] != 'laplacian':
            assert r.sides['B'].params == engine_defaults(e['id'])
        assert np.abs(pcm).max() > 100, f"{e['id']} silent"
        assert not np.isnan(pcm.astype(float)).any()
        assert r.sides['B'].clip_blocks == 0, f"{e['id']} clips"


def test_param_memory_per_side_and_engine():
    r = DemoRunner(SCENE_AB)
    r.post('set_param', side='B', name='harm', value=0.25)
    r.post('set_engine', side='B', engine_id='fft2d')
    r.next_block()
    assert r.sides['B'].params == engine_defaults('fft2d')       # first pick = defaults
    r.post('set_param', side='B', name='n', value=7)
    r.post('set_engine', side='B', engine_id='laplacian')
    r.next_block()
    assert r.sides['B'].params['harm'] == 0.25                  # previous values return
    assert r.sides['A'].params['harm'] == 0                      # A untouched
    r.post('set_engine', side='B', engine_id='fft2d')
    r.next_block()
    assert r.sides['B'].params['n'] == 7
    r.post('set_engine', side='A', engine_id='fft2d')
    r.next_block()
    assert r.sides['A'].params['n'] == 16                        # memory is per SIDE too
    assert describe_difference(r.side_settings()) == "n: A=16, B=7"
    r.post('set_engine', side='A', engine_id='walsh')
    r.next_block()
    assert describe_difference(r.side_settings()) == "A: Walsh  /  B: FFT"


def test_bad_commands_are_rejected_before_any_state_change():
    r = DemoRunner(SCENE_AB)
    r.post('start', at=0)
    for _ in range(20):
        r.next_block()
    before = (r.side_settings(), r.selected, len(r._pending), r.out_samples)
    bad = [
        ('set_param', dict(side='B', name='harm', value=2.0)),
        ('set_param', dict(side='B', name='harm', value=float('nan'))),
        ('set_param', dict(side='B', name='harm', value=float('inf'))),
        ('set_param', dict(side='B', name='harm', value="0.5")),
        ('set_param', dict(side='B', name='n', value=2.5)),
        ('set_param', dict(side='B', name='zzz', value=0.5)),
        ('set_param', dict(side='C', name='harm', value=0.5)),
        ('set_engine', dict(side='A', engine_id='nope')),
        ('select', dict(side='X')),
        ('vol', dict(value=float('nan'))),
        ('bogus', dict()),
    ]
    for kind, args in bad:
        try:
            r.post(kind, **args)
        except ValueError as e:
            assert str(e), "empty error message"
        else:
            raise AssertionError(f"{kind} {args} accepted")
    assert (r.side_settings(), r.selected, len(r._pending), r.out_samples) == before
    # a queued engine change makes the following param check use the new engine
    r.post('set_engine', side='B', engine_id='fft2d')
    try:
        r.post('set_param', side='B', name='harm', value=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("harm accepted for a pending FFT side")


# =============================================================================
# S2: synchrony / independence
# =============================================================================
def test_ab_equal_when_settings_equal_and_switching_changes_nothing():
    cmds = [('start', 0, {}), ('select', _sec(0.7), {'side': 'B'}),
            ('set_cell', _sec(1.0), {'r': 2, 'c': 2, 'v': 1}),
            ('select', _sec(1.5), {'side': 'A'}), ('select', _sec(1.6), {'side': 'B'})]
    res, r = render_offline(SCENE, 3.0, commands=cmds, output=ALL)
    assert np.array_equal(res['A'], res['B']), "A != B with equal settings"
    ref, r0 = render_offline(SCENE, 3.0, output='A',
                             commands=[('start', 0, {}),
                                       ('set_cell', _sec(1.0), {'r': 2, 'c': 2, 'v': 1})])
    assert np.array_equal(res['A'], ref), "switching changed the raw side PCM"
    assert np.array_equal(r.grid, r0.grid) and r.gen == r0.gen
    assert np.array_equal(res['monitor'], res['A']), "monitor of equal sides != raw"


def test_editing_b_does_not_change_raw_a():
    ref, _ = render_offline(SCENE_AB, 3.0, output='A')
    cmds = [('start', 0, {}),
            ('set_param', _sec(0.5), {'side': 'B', 'name': 'harm', 'value': 0.3}),
            ('set_engine', _sec(1.0), {'side': 'B', 'engine_id': 'granulo'}),
            ('set_param', _sec(1.5), {'side': 'B', 'name': 'n', 'value': 5}),
            ('select', _sec(2.0), {'side': 'B'})]
    res, r = render_offline(SCENE_AB, 3.0, commands=cmds, output=ALL)
    assert np.array_equal(res['A'], ref)
    assert not np.array_equal(res['A'], res['B'])
    assert r.sides['A'].params['harm'] == 0 and r.sides['B'].params['n'] == 5


def test_demo_ab_sides_differ_and_are_nonsilent():
    res, r = render_offline(SCENE_AB, 8.0, output=ALL)
    assert np.abs(res['A']).max() > 100 and np.abs(res['B']).max() > 100
    diff = np.abs(res['A'].astype(int) - res['B'].astype(int)).max()
    assert diff > 100, f"A/B numerically indistinguishable (max diff {diff})"
    assert r.snapshot()['clip_blocks'] == {'A': 0, 'B': 0}
    assert describe_difference(r.side_settings()) == "harm: A=0, B=1"


def test_param_change_applies_at_block_boundary_without_reset():
    r = DemoRunner(SCENE_AB)
    r.post('start', at=0)
    for _ in range(100):
        r.next_block()
    gen, ca = r.gen, r.ca_samples
    ph_a, ph_b = r.sides['A'].phase.copy(), r.sides['B'].phase.copy()
    r.post('set_param', side='B', name='harm', value=0.5)
    r.next_block()
    assert r.gen == gen and r.ca_samples == ca + BLOCK
    assert not np.array_equal(r.sides['B'].phase, ph_b)
    assert not np.array_equal(r.sides['A'].phase, ph_a)
    assert r.sides['B'].params['harm'] == 0.5
    # engine change re-initialises ONLY that side's audio memory
    amp_a = r.sides['A'].amp_cur.copy()
    r.post('set_engine', side='B', engine_id='walsh')
    r.next_block()
    assert r.sides['A'].amp_cur.max() > 0 and r.gen == gen
    assert np.allclose(r.sides['A'].amp_cur, amp_a, atol=0.05)


def test_monitor_crossfade_sums_to_one_and_keeps_block_count():
    cmds = [('start', 0, {}), ('select', _sec(1.0), {'side': 'B'})]
    res, r = render_offline(SCENE_AB, 2.0, commands=cmds, output=ALL)
    t_sel = [t for (t, k, a) in _events(r) if k == 'select'][0]
    assert np.array_equal(res['monitor'][:t_sel], res['A'][:t_sel])
    end = t_sel + XFADE_SAMPLES
    assert np.array_equal(res['monitor'][end + BLOCK:], res['B'][end + BLOCK:])
    a = res['A'][t_sel:end].astype(float)
    b = res['B'][t_sel:end].astype(float)
    x = (np.arange(XFADE_SAMPLES) + 1) / XFADE_SAMPLES
    expect = np.rint(a * (1 - x)[:, None] + b * x[:, None])
    assert np.abs(res['monitor'][t_sel:end] - expect).max() <= 1
    assert len(res['monitor']) == _sec(2.0)


def test_live_equals_offline_on_ab_scenario():
    off, r_off = render_offline(SCENE_AB, SCEN_AB_SECONDS, commands=SCENARIO_AB, output=ALL)
    live, r_live = _live_render(SCENE_AB, SCENARIO_AB, SCEN_AB_SECONDS, batches=2,
                                sleep_between=0.05)
    for o in ALL:
        assert np.array_equal(off[o], live[o]), f"live {o} != offline {o}"
    assert _events(r_off) == _events(r_live)
    assert np.array_equal(r_off.grid, r_live.grid) and r_off.gen == r_live.gen
    ev = _events(r_off)
    assert [k for (_t, k, _a) in ev] == [k for (k, _at, _a) in SCENARIO_AB]
    for (t, k, a), (_k, at, _a) in zip(ev, SCENARIO_AB):
        assert at <= t < at + BLOCK, (k, at, t)
    assert r_off.sides['B'].engine_id == 'fft2d' and r_off.sides['A'].engine_id == 'laplacian'
    assert r_off.selected == 'A'
    again, _ = render_offline(SCENE_AB, SCEN_AB_SECONDS, commands=SCENARIO_AB, output=ALL)
    for o in ALL:
        assert np.array_equal(off[o], again[o])


# =============================================================================
# interface (headless smoke)
# =============================================================================
def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_ui_headless_smoke_buttons_and_painting():
    import pygame
    import demo_bench as db
    runner = DemoRunner(SCENE_AB)
    got = []
    eng = LiveEngine(runner, sink=lambda mon, blk: got.append(mon))
    pygame.init()
    app = db.BenchApp(SCENE_AB, eng)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    try:
        app.draw(screen, font, small)
        snap = eng.snapshot()
        assert not snap['running'] and snap['gen'] == 0 and snap['selected'] == 'A'

        rect = app.buttons['start'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'start'
        assert _wait(lambda: eng.snapshot()['gen'] >= 2, 30), "no evolution after Start"

        # A/B tab: listen + edit side B, field keeps going
        g_before = eng.snapshot()['gen']
        rect = app.tabs['B']
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'tab:B'
        assert _wait(lambda: eng.snapshot()['selected'] == 'B')
        assert eng.snapshot()['gen'] >= g_before

        # harm knob on B (slider), A unchanged
        rows = dict((spec[0], r) for spec, r in app._param_rows('laplacian'))
        sx, sy, sw, sh = rows['harm']
        assert app.press((sx + sw // 2, sy + 3), 1) == 'param:harm'
        app.release()
        assert _wait(lambda: abs(eng.snapshot()['sides']['B'][1]['harm'] - 0.5) < 0.02)
        assert eng.snapshot()['sides']['A'][1]['harm'] == 0
        # toggle (fullshape) flips 1 -> 0
        sx, sy, sw, sh = rows['fullshape']
        assert app.press((sx + 5, sy + 3), 1) == 'param:fullshape'
        app.release()
        assert _wait(lambda: eng.snapshot()['sides']['B'][1]['fullshape'] == 0)
        # rejected value shows a message, state intact
        assert app.set_param('harm', 5.0) is False and "outside" in app.message
        assert abs(eng.snapshot()['sides']['B'][1]['harm'] - 0.5) < 0.02

        # engine FFT on B -> its params appear
        rect = app.engine_btns['fft2d']
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'engine:fft2d'
        assert _wait(lambda: eng.snapshot()['sides']['B'][0] == 'fft2d')
        assert [s[0] for s, _ in app._param_rows('fft2d')] == ['n']
        assert eng.snapshot()['sides']['A'][0] == 'laplacian'
        app.draw(screen, font, small)

        rect = app.buttons['pause'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'pause'
        assert _wait(lambda: eng.snapshot()['paused'])
        g0 = eng.snapshot()['gen']
        fx, fy = app.field_x, app.field_y
        assert app.press((fx + 1 * db.CELL + 2, fy + 1 * db.CELL + 2), 1) == 'paint'
        app.drag((fx + 2 * db.CELL + 2, fy + 1 * db.CELL + 2))
        app.release()
        assert _wait(lambda: eng.snapshot()['grid'][1, 1] == 1 and eng.snapshot()['grid'][1, 2] == 1)
        assert app.press((fx + 1 * db.CELL + 2, fy + 1 * db.CELL + 2), 3) == 'paint'
        app.release()
        assert _wait(lambda: eng.snapshot()['grid'][1, 1] == 0)
        n_before = len(got)
        time.sleep(0.2)
        assert len(got) > n_before, "audio stopped during pause"
        assert eng.snapshot()['gen'] == g0, "field evolved while paused"

        rect = app.buttons['start'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'stop'
        assert _wait(lambda: (not eng.snapshot()['running'] and eng.snapshot()['gen'] == 0
                              and eng.snapshot()['grid'].sum() == 5))
        s = eng.snapshot()
        assert s['selected'] == 'B' and s['sides']['B'][0] == 'fft2d'   # settings kept
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'start'
        assert _wait(lambda: eng.snapshot()['running'])

        rect = app.buttons['reset'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'reset'
        assert _wait(lambda: (eng.snapshot()['gen'] == 0 and eng.snapshot()['running']
                              and not eng.snapshot()['paused']))
        assert eng.snapshot()['grid'].sum() == 5
        assert _wait(lambda: eng.snapshot()['gen'] >= 1, 30), "no evolution after Restart"

        vx, vy, vw, vh = app.vol_rect
        assert app.press((vx + vw // 2, vy + 3), 1) == 'vol'
        app.release()
        assert _wait(lambda: abs(eng.snapshot()['vol'] - 0.5) < 0.02)
        app.draw(screen, font, small)
        pygame.image.save(screen, os.path.join(ART, "_demo_bench_smoke.png"))
    finally:
        eng.stop()
        pygame.quit()


def test_no_audio_device_state_is_visible():
    def broken(_cb):
        raise RuntimeError("Error querying device -1")
    runner = DemoRunner(SCENE)
    eng = LiveEngine(runner, output_factory=broken)
    eng.start()
    try:
        assert not eng.device_ok
        assert "NO AUDIO DEVICE" in eng.status_text()
        eng.post('start')
        assert _wait(lambda: eng.snapshot()['gen'] >= 1, 10), "scene frozen without device"
    finally:
        eng.stop()


def test_device_stream_opened_stereo():
    opened = {}

    class _Fake:
        def stop(self): pass
        def close(self): pass

    def factory(cb):
        import inspect
        from casynth_lab import audio_out
        src = inspect.getsource(audio_out._default_output_factory)
        assert "channels=CHANNELS" in src and audio_out.CHANNELS == 2
        opened['ok'] = True
        return _Fake()

    eng = LiveEngine(DemoRunner(SCENE), output_factory=factory)
    eng.start()
    try:
        assert eng.device_ok and opened.get('ok')
        assert eng.status_text().startswith("OK")
        out = np.zeros((512, 2), np.int16)
        eng.post('start')
        time.sleep(0.1)
        eng._audio_cb(out, 512, None, None)
        assert out.shape == (512, 2)
    finally:
        eng.stop()


def test_cli_render_all_outputs_and_errors():
    import subprocess
    import wave
    py = sys.executable
    pcms = {}
    for side in ALL:
        out = os.path.join(ART, f"_s2_cli_{side}.wav")
        r = subprocess.run([py, os.path.join(ROOT, "demo_bench.py"), "--demo", DEMO_AB,
                            "--render", out, "--seconds", "1", "--side", side],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"({side})" in r.stdout
        with wave.open(out) as w:
            assert w.getnchannels() == 2 and w.getframerate() == SR
            assert w.getnframes() == _sec(1.0)
            pcms[side] = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)
    assert np.array_equal(pcms['A'], pcms['monitor'])
    assert not np.array_equal(pcms['A'], pcms['B'])
    r = subprocess.run([py, os.path.join(ROOT, "demo_bench.py"), "--demo",
                        os.path.join(ART, "_missing.json")], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 2 and "not found" in r.stdout


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
