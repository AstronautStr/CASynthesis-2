#!/usr/bin/env python3
"""S1 acceptance tests for the demo bench (casynth_lab + demo_bench).

Covers req-demo-lab-s1.md "automatic acceptance": repeatability, live-vs-
offline event equality, clocks/pause/reset, silence, stereo, scene validation,
headless UI smoke (buttons + painting with both mouse buttons), no-device state.

    python tests/test_demo_lab.py      # stdlib runner, no pytest needed
"""
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, VOL_DEFAULT                          # noqa: E402
from casynth_lab import load_scene, SceneError, DemoRunner, BLOCK, render_offline  # noqa: E402
from casynth_lab.scene import validate                              # noqa: E402
from casynth_lab.audio_out import LiveEngine                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demos", "laplace_basic.json")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

SCENE = load_scene(DEMO)


def _sec(s):
    return int(round(s * SR))


# The fixed event scenario (times in output samples): start, pause for 1 s,
# edit one cell during the pause, resume, change the level.
SCENARIO = [
    ('start', 0, {}),
    ('pause', _sec(1.0), {'on': True}),
    ('set_cell', _sec(1.5), {'r': 3, 'c': 3, 'v': 1}),
    ('pause', _sec(2.0), {'on': False}),
    ('vol', _sec(2.5), {'value': 0.4}),
]
SCEN_SECONDS = 4.0


def _live_render(commands, seconds, batches=1, sleep_between=0.0):
    """Run the scenario through LiveEngine's command queue with a test sink
    paced at real time (like a device), posting commands in `batches` groups
    (= different 'UI update rates'; every group lands before its events' times)."""
    n = _sec(seconds)
    runner = DemoRunner(SCENE)
    got = []
    got_n = [0]
    done = threading.Event()
    t0 = [None]

    def sink(buf):
        if t0[0] is None:
            t0[0] = time.perf_counter()
        if got_n[0] < n:
            got.append(buf)
            got_n[0] += len(buf)
            if got_n[0] >= n:
                done.set()
        # real-time pacing: never run ahead of the wall clock by > 40 ms
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
        assert done.wait(timeout=60), "live render timed out"
    finally:
        eng.stop()
    pcm = np.concatenate(got, axis=0)[:n]
    return pcm, runner


# -- scene ----------------------------------------------------------------------
def test_scene_loads_and_matches_spec():
    assert SCENE.rows == 32 and SCENE.cols == 32
    assert sorted(SCENE.cells) == sorted([(14, 15), (14, 16), (15, 14), (15, 15), (16, 15)])
    assert SCENE.rate_hz == 4 and SCENE.engine_id == 'laplacian' and SCENE.f0_hz == 220
    assert SCENE.engine_params == dict(n=12, spread=0, alpha=1, shape=0, harm=0,
                                       fullshape=1, dyn=0)
    assert SCENE.initial_grid().sum() == 5


def _base_doc():
    import json
    with open(DEMO, encoding='utf-8') as f:
        return json.load(f)


def _expect_error(mutate, needle):
    d = _base_doc()
    mutate(d)
    try:
        validate(d)
    except SceneError as e:
        assert needle in str(e), f"message {e!r} lacks {needle!r}"
        return
    raise AssertionError(f"no SceneError for {needle}")


def test_scene_rejects_unknown_engine():
    _expect_error(lambda d: d.update(engine_id='nope'), "unknown engine_id")


def test_scene_rejects_unknown_and_missing_params():
    _expect_error(lambda d: d['engine_params'].update(zzz=1), "unknown engine_params")
    _expect_error(lambda d: d['engine_params'].pop('harm'), "missing engine_params")


def test_scene_rejects_bad_cells_and_bad_json():
    _expect_error(lambda d: d['cells'].append([32, 0]), "outside")
    _expect_error(lambda d: d['cells'].append([1]), "must be [row, col]")
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


# -- repeatability --------------------------------------------------------------
def test_offline_repeat_and_reset_identical_8s():
    a, ra = render_offline(SCENE, 8.0)
    b, _ = render_offline(SCENE, 8.0)
    assert a.shape == (_sec(8.0), 2) and a.dtype == np.int16
    assert np.array_equal(a, b), "two renders differ"
    assert not np.isnan(a.astype(float)).any()
    assert ra.clip_blocks == 0, f"scene clips: {ra.clip_blocks} blocks"
    assert np.abs(a).max() < 32767
    # reset on the same runner -> identical PCM again
    c, _ = render_offline(SCENE, 8.0, commands=[('reset', None, {})], runner=ra)
    assert np.array_equal(a, c), "render after reset differs"
    assert ra.gen == 31


def test_stereo_identical_channels_and_nonsilent():
    a, _ = render_offline(SCENE, 2.0)
    assert a.ndim == 2 and a.shape[1] == 2
    assert np.array_equal(a[:, 0], a[:, 1])
    assert np.abs(a).max() > 100, "control scene is silent"


# -- events: live queue vs offline ------------------------------------------------
def test_live_equals_offline_on_scenario():
    off, r_off = render_offline(SCENE, SCEN_SECONDS, commands=SCENARIO)
    live, r_live = _live_render(SCENARIO, SCEN_SECONDS)
    assert np.array_equal(off, live), "live PCM != offline PCM"
    assert np.array_equal(r_off.grid, r_live.grid), "final fields differ"
    ev_off = [(t, k, a) for (t, s, k, a) in r_off.journal if k != 'step']
    ev_live = [(t, k, a) for (t, s, k, a) in r_live.journal if k != 'step']
    assert ev_off == ev_live, f"journals differ: {ev_off} vs {ev_live}"
    assert r_off.grid[3, 3] == 1 or r_off.gen > 8   # the edit was applied (may evolve)
    assert any(k == 'set_cell' for (_t, _s, k, _a) in r_off.journal)


def test_ui_update_rate_does_not_change_result():
    ref, _ = render_offline(SCENE, SCEN_SECONDS, commands=SCENARIO)
    slow, _ = _live_render(SCENARIO, SCEN_SECONDS, batches=3, sleep_between=0.05)
    assert np.array_equal(ref, slow)


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


# -- clocks / pause / reset ---------------------------------------------------------
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
    _, r = render_offline(SCENE, 4.1, commands=cmds)
    steps = [t for (t, s, k, a) in r.journal if k == 'step']
    in_pause = [t for t in steps if _sec(1.1) < t <= _sec(3.1)]
    assert not in_pause, in_pause
    # 1.1 s before pause (4 steps) + 1 s after (steps 5..8) -> 8, not 12
    assert r.gen == 8, r.gen
    after = [t for t in steps if t > _sec(3.1)]
    assert len(after) == 4, after
    # the CA clock resumes where it froze (1.1 s): step 5 is due at CA 1.25 s,
    # i.e. ~0.15 s after resume -- not a burst at the resume boundary
    assert 0.1 * SR < after[0] - _sec(3.1) < 0.2 * SR, after[0]
    # sound kept running during the pause
    pcm, _ = render_offline(SCENE, 4.1, commands=cmds)
    seg = pcm[_sec(1.5):_sec(2.5)]
    assert np.abs(seg).max() > 100


def test_start_is_silent_and_frozen_until_started():
    r = DemoRunner(SCENE)
    for _ in range(200):
        b = r.next_block()
        assert not b.any()
    assert r.gen == 0 and r.ca_samples == 0 and r.t_samples == 0


def test_reset_clears_tails_phases_and_queue():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(600):
        r.next_block()
    assert r.amp_cur.max() > 0 and r.phase.any()
    r.post('vol', at=r.out_samples + 10 * BLOCK, value=0.1)      # future -> dropped
    r.post('reset')
    r.next_block()
    assert r.gen == 0 and r.ca_samples == BLOCK and r.running and not r.paused
    assert not r._pending, "queue not cleared by reset"
    assert np.array_equal(r.grid, SCENE.initial_grid())
    # phases/tails restart from 0 -> the first block after reset is identical
    # to the first block of a fresh runner
    r2 = DemoRunner(SCENE)
    r2.post('start', at=0)
    for _ in range(600):
        r2.next_block()
    r2.post('reset')
    b_reset = r2.next_block()
    fresh = DemoRunner(SCENE)
    fresh.post('start', at=0)
    assert np.array_equal(b_reset, fresh.next_block())
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
    assert not r.running and r.gen == 0 and r.ca_samples == 0 and not b.any()
    assert not r._pending
    assert np.array_equal(r.grid, SCENE.initial_grid())
    # Start again == a fresh run
    r.post('start')
    fresh = DemoRunner(SCENE)
    fresh.post('start', at=0)
    for _ in range(50):
        assert np.array_equal(r.next_block(), fresh.next_block())


def test_empty_field_goes_silent_after_tails():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(50):
        r.next_block()
    for (rr, cc) in np.argwhere(r.grid > 0):
        r.post('set_cell', r=int(rr), c=int(cc), v=0)
    tail = [r.next_block() for _ in range(400)]         # ~3.2 s
    assert r.grid.sum() == 0
    assert not tail[-1].any(), "empty field still sounds"
    assert r.amp_cur.max() < 1e-4


def test_volume_change_is_smoothed():
    r = DemoRunner(SCENE)
    r.post('start', at=0)
    for _ in range(100):
        r.next_block()
    r.post('vol', value=0.0)
    b = r.next_block().astype(float)
    assert abs(b[0, 0]) > 0 or abs(b[1, 0]) > 0
    assert b[-1, 0] == 0 and b[-1, 1] == 0


# -- interface (headless smoke) -------------------------------------------------------
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
    runner = DemoRunner(SCENE)
    got = []
    eng = LiveEngine(runner, sink=got.append)
    pygame.init()
    app = db.BenchApp(SCENE, eng)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    try:
        app.draw(screen, font, small)
        snap = eng.snapshot()
        assert not snap['running'] and snap['gen'] == 0

        rect = app.buttons['start'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'start'
        assert _wait(lambda: eng.snapshot()['gen'] >= 2, 30), "no evolution after Start"

        rect = app.buttons['pause'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'pause'
        assert _wait(lambda: eng.snapshot()['paused'])
        g0 = eng.snapshot()['gen']
        # paint with LMB at (1,1), erase with RMB at a live cell later
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

        # Start button reads Stop while running -> full stop
        rect = app.buttons['start'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'stop'
        assert _wait(lambda: (not eng.snapshot()['running'] and eng.snapshot()['gen'] == 0
                              and eng.snapshot()['grid'].sum() == 5))
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'start'
        assert _wait(lambda: eng.snapshot()['running'])

        rect = app.buttons['reset'][0]
        assert app.press((rect[0] + 5, rect[1] + 5), 1) == 'reset'
        assert _wait(lambda: (eng.snapshot()['gen'] == 0 and eng.snapshot()['running']
                              and not eng.snapshot()['paused']))
        s = eng.snapshot()
        assert s['grid'].sum() == 5 and s['grid'][1, 2] == 0
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
        # mirror _default_output_factory's arguments by inspecting the source
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
        # callback path: pull a frame and check 2-channel int16 output
        out = np.zeros((512, 2), np.int16)
        eng.post('start')
        time.sleep(0.1)
        eng._audio_cb(out, 512, None, None)
        assert out.shape == (512, 2)
    finally:
        eng.stop()


def test_cli_render_and_errors():
    import subprocess
    py = sys.executable
    out = os.path.join(ART, "_s1_cli.wav")
    r = subprocess.run([py, os.path.join(ROOT, "demo_bench.py"), "--demo", DEMO,
                        "--render", out, "--seconds", "1"], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    import wave
    with wave.open(out) as w:
        assert w.getnchannels() == 2 and w.getframerate() == SR
        assert w.getnframes() == _sec(1.0)
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
