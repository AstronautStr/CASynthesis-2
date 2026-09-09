#!/usr/bin/env python3
"""S4 acceptance tests: local experiment catalog, save, playback data, replay.

    python tests/test_demo_lab_s4.py      # stdlib runner
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, VOL_DEFAULT                           # noqa: E402
from casynth_lab import load_scene, DemoRunner, BLOCK, registry      # noqa: E402
from casynth_lab.audio_out import LiveEngine                         # noqa: E402
from casynth_lab.catalog import Catalog, CatalogError, read_wav, bench_scene  # noqa: E402
from casynth_lab.engine_api import SoundEngine                       # noqa: E402
from casynth_lab.registry import EngineSpec, register, unregister    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
ART = os.path.join(ROOT, "artifacts")
CAT_ROOT = os.path.join(ART, "_catalog_test")
OUTPUTS = ('A', 'B', 'monitor')


def _sec(s):
    return int(round(s * SR))


def _fresh_catalog(name):
    root = os.path.join(CAT_ROOT, name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, '.tmp'), exist_ok=True)
    return Catalog(root)


def _wait(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


# The fixed S4 experiment: settings before Start, painting, B changes, volume,
# A/B, pause, Stop/Start, Restart, copy, factory.
EXPERIMENT = [
    ('set_param', 0, {'side': 'B', 'name': 'harm', 'value': 0.3}),
    ('select', 0, {'side': 'B'}),
    ('start', _sec(0.3), {}),
    ('set_cell', _sec(0.6), {'r': 2, 'c': 2, 'v': 1}),
    ('set_cell', _sec(0.6), {'r': 2, 'c': 3, 'v': 1}),
    ('set_engine', _sec(0.9), {'side': 'B', 'engine_id': 'walsh'}),
    ('set_param', _sec(1.1), {'side': 'B', 'name': 'n', 'value': 9}),
    ('vol', _sec(1.3), {'value': 0.5}),
    ('select', _sec(1.5), {'side': 'A'}),
    ('pause', _sec(1.7), {'on': True}),
    ('set_cell', _sec(1.8), {'r': 5, 'c': 5, 'v': 1}),
    ('pause', _sec(2.0), {'on': False}),
    ('stop', _sec(2.2), {}),
    ('start', _sec(2.5), {}),
    ('reset', _sec(2.8), {}),
    ('copy_side', _sec(3.0), {'src': 'B', 'dst': 'A'}),
    ('factory', _sec(3.3), {}),
    ('select', _sec(3.5), {'side': 'B'}),
]
EXP_END = _sec(3.8)


class _Session:
    """A live session with a test sink that records everything it receives."""

    def __init__(self, catalog, scene=None, paced=False):
        self.scene = scene or load_scene(DEMO_AB)
        self.runner = DemoRunner(self.scene)
        self.got = {o: [] for o in OUTPUTS}
        self.first_running = None
        self.n = 0
        self.paced = paced
        self._t0 = None
        self.engine = LiveEngine(self.runner, sink=self._sink, record_root=catalog.tmp_root)

    def _sink(self, mon, blk):
        snap = self.engine.snapshot()
        if snap['running'] and self.first_running is None:
            self.first_running = snap['out_samples'] - BLOCK
        if self.first_running is not None and blk is not None:
            for o in OUTPUTS:
                self.got[o].append(blk.get(o))
        self.n += BLOCK
        if self.paced:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            ahead = self.n / SR - (time.perf_counter() - self._t0)
            if ahead > 0.04:
                time.sleep(ahead - 0.04)

    def run_until(self, out_sample):
        assert _wait(lambda: self.engine.snapshot()['out_samples'] >= out_sample, 60)

    def pcm(self, o, start, end):
        arr = np.concatenate(self.got[o], axis=0)
        s = start - self.first_running
        return arr[s:s + (end - start)]


def _run_experiment(catalog, commands=EXPERIMENT, until=EXP_END):
    """Drive a paced session like the UI does: each command is posted shortly
    before its time (Stop/Restart clear whatever is queued for later, so the
    journal cannot be loaded all at once)."""
    s = _Session(catalog, paced=True)
    s.engine.start()
    for kind, at, args in commands:
        s.run_until(max(0, at - 6 * BLOCK))
        s.engine.post(kind, at=at, **args)
    s.run_until(until)
    kind, cut = s.engine.cut_now()
    assert kind == 'ok' and cut is not None
    return s, cut


# =============================================================================
def test_save_replay_fixed_experiment_in_fresh_process():
    cat = _fresh_catalog('fixed')
    s, cut = _run_experiment(cat)
    try:
        rid = cat.save(cut, "Тест S4: опыт", "Заметка — кириллица ✓")
    finally:
        s.engine.stop()
    rec = cat.load(rid)
    assert rec.title == "Тест S4: опыт" and rec.note == "Заметка — кириллица ✓"
    assert rec.meta['status'] == 'local' and rec.meta['scene'] == s.scene.doc
    # captured WAVs == the blocks the sink received for the same interval
    for o in OUTPUTS:
        got = s.pcm(o, cut.audio_start_sample, cut.end_sample)
        assert np.array_equal(rec.pcm(o), got), f"{o}: saved WAV != sink blocks"
    kinds = [j['kind'] for j in rec.meta['journal']]
    for k in ('stop', 'reset', 'copy_side', 'factory', 'set_engine'):
        assert k in kinds
    assert rec.meta['audio_start_sample'] > 0
    assert rec.engines == ('laplacian', 'laplacian')     # after factory
    # Open in bench = the INITIAL state: scene field + pre-start settings
    # (harm 0.3 on B, side B selected), default volume; instant
    t0 = time.time()
    sc2, vol2 = bench_scene(rec)
    assert time.time() - t0 < 1.0
    assert vol2 == VOL_DEFAULT and sc2.initial_side == 'B' and sc2.title == rec.title
    assert np.array_equal(sc2.initial_grid(), s.scene.initial_grid())
    assert sc2.variants['B'][1]['harm'] == 0.3 and sc2.variants['A'] == s.scene.variants['A']
    r2 = DemoRunner(sc2, vol=vol2)
    assert not r2.running and r2.gen == 0 and np.array_equal(r2.grid, s.scene.initial_grid())
    # the record also keeps the end state for information
    st = rec.meta['state_at_end']
    assert st['gen'] == s.runner.gen and st['vol'] == 0.5
    # replay in a FRESH process, byte-exact, and commands after Stop/Restart ran
    code = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from casynth_lab.catalog import Catalog\n"
        "c = Catalog(%r); r = c.replay(%r)\n"
        "rec = c.load(%r)\n"
        "print(json.dumps(dict(status=r.status, outputs=r.outputs, title=rec.title, note=rec.note)))\n"
        % (ROOT, cat.root, rid, rid))
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out['status'] == 'match', out
    assert out['title'] == "Тест S4: опыт" and out['note'] == "Заметка — кириллица ✓"
    # the replay ran the whole journal incl. everything after stop/reset
    res = cat.replay(rid)
    assert res.status == 'match' and all(res.outputs.values())
    rep = read_wav(os.path.join(res.replay_dir, 'monitor.wav'))
    assert np.array_equal(rep, rec.pcm('monitor'))
    # originals untouched by the replay
    with open(rec.wav_path('A'), 'rb') as f:
        assert len(f.read()) == 44 + rec.meta['audio']['A']['samples'] * 4


def test_cut_is_a_consistent_prefix_and_session_continues():
    cat = _fresh_catalog('prefix')
    s = _Session(cat)
    s.engine.post('start', at=0)
    s.engine.start()
    try:
        s.run_until(_sec(1.0))
        # a command for the future must NOT be in the cut
        s.engine.post('vol', at=_sec(5.0), value=0.2)
        kind, cut1 = s.engine.cut_now()
        assert kind == 'ok'
        assert cut1.end_sample % BLOCK == 0
        assert all(t < cut1.end_sample for (t, _s, _k, _a) in cut1.journal)
        assert not any(k == 'vol' for (_t, _s, k, _a) in cut1.journal)
        rid1 = cat.save(cut1, "first", "")
        # session keeps running; a later save is a NEW record, the first is unchanged
        s.run_until(cut1.end_sample + _sec(0.5))
        s.engine.post('set_cell', r=1, c=1, v=1)
        assert _wait(lambda: any(k == 'set_cell' for (_t, _q, k, _a) in list(s.runner.journal)))
        kind, cut2 = s.engine.cut_now()
        rid2 = cat.save(cut2, "second", "")
        assert rid2 != rid1 and cut2.end_sample > cut1.end_sample
        r1, r2 = cat.load(rid1), cat.load(rid2)
        assert r1.meta['end_sample'] == cut1.end_sample
        assert r2.meta['end_sample'] == cut2.end_sample
        assert np.array_equal(r2.pcm('A')[:len(r1.pcm('A'))], r1.pcm('A'))
        assert any(j['kind'] == 'set_cell' for j in r2.meta['journal'])
        assert not any(j['kind'] == 'set_cell' for j in r1.meta['journal'])
        assert [rr.id for rr, _e in cat.list()] == [rid2, rid1]      # newest first
    finally:
        s.engine.stop()


def test_waiting_before_start_does_not_lengthen_the_wav():
    cat = _fresh_catalog('wait')
    s = _Session(cat)
    s.engine.post('set_param', side='B', name='harm', value=0.7)    # before start
    s.engine.post('start', at=_sec(2.0))
    s.engine.start()
    try:
        assert s.engine.cut_now()[0] == 'none'              # nothing to save yet
        s.run_until(_sec(2.5))
        kind, cut = s.engine.cut_now()
        assert kind == 'ok'
        assert _sec(2.0) <= cut.audio_start_sample < _sec(2.0) + BLOCK
        assert cut.n_frames == cut.end_sample - cut.audio_start_sample < _sec(1.0)
        rid = cat.save(cut)
        rec = cat.load(rid)
        assert rec.meta['audio']['A']['samples'] == cut.n_frames
        assert rec.title.startswith(s.scene.title)
        # the pre-start event survives for replay
        assert rec.meta['journal'][0]['kind'] == 'set_param'
        assert cat.replay(rid).status == 'match'
    finally:
        s.engine.stop()


class _StubEngine(SoundEngine):
    def init(self, grid, exc, gain):
        self.n = 0

    def update_field(self, grid, exc):
        pass

    def render(self, gain, t_samples):
        self.n += 1
        return np.full((self.ctx.block, 2), 300 + self.n % 5, np.int16), 0.01, 0

    def reset(self, gain):
        self.n = 0


def test_record_is_independent_of_scene_file_and_engine_availability():
    cat = _fresh_catalog('indep')
    register(EngineSpec('stub4', 'Stub4', [('amp', 'amp', 0.0, 1.0, False, 0.5)],
                        lambda ctx, p: _StubEngine(ctx, p)))
    tmp_json = os.path.join(CAT_ROOT, "indep_scene.json")
    with open(DEMO_AB, encoding='utf-8') as f:
        d = json.load(f)
    d['variants']['B'] = {'engine_id': 'stub4', 'engine_params': {'amp': 0.5}}
    with open(tmp_json, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    scene = load_scene(tmp_json)
    s = _Session(cat, scene=scene)
    s.engine.post('start', at=0)
    s.engine.post('select', at=_sec(0.4), side='B')
    s.engine.start()
    try:
        s.run_until(_sec(1.0))
        rid = cat.save(s.engine.cut_now()[1], "stub record", "n")
    finally:
        s.engine.stop()
    os.remove(tmp_json)                              # source JSON gone
    assert cat.replay(rid).status == 'match'         # embedded conditions suffice
    unregister('stub4')
    rec = cat.load(rid)                              # listing/loading needs no engine
    assert rec.engine_labels()[1] == 'stub4 (missing)'
    assert rec.pcm('B')[-1, 0] in (300, 301, 302, 303, 304)      # WAV still playable
    res = cat.replay(rid)
    assert res.status == 'unavailable' and 'stub4' in res.reason, res.reason


def test_errors_no_false_success_no_lost_records():
    cat = _fresh_catalog('errors')
    s, cut = _run_experiment(cat, commands=[('start', 0, {})], until=_sec(0.8))
    try:
        good = cat.save(cut, "good", "")
        # 1. interrupted recording: the raw files stop growing -> no success
        s.engine.recorder.close()
        s.run_until(s.engine.snapshot()['out_samples'] + 20 * BLOCK)
        kind, cut2 = s.engine.cut_now()
        assert cut2.record_error is not None
        try:
            cat.save(cut2, "broken", "")
        except CatalogError as e:
            assert "recording" in str(e)
        else:
            raise AssertionError("interrupted recording saved as success")
    finally:
        s.engine.stop()
    assert cat.ids() == [good]
    assert not [n for n in os.listdir(cat.root) if n.endswith('.partial')]
    # 2. disk failure while assembling -> error, no partial record, old ones intact
    s, cut = _run_experiment(cat, commands=[('start', 0, {})], until=_sec(0.6))
    try:
        import casynth_lab.catalog as cm
        orig = cm._raw_to_wav

        def boom(*a, **k):
            raise OSError("disk full")
        cm._raw_to_wav = boom
        try:
            try:
                cat.save(cut, "disk", "")
            except CatalogError as e:
                assert "disk full" in str(e)
            else:
                raise AssertionError("disk failure saved as success")
        finally:
            cm._raw_to_wav = orig
        assert cat.ids() == [good]
        assert not [n for n in os.listdir(cat.root) if n.endswith('.partial')]
        second = cat.save(cut, "second", "")
    finally:
        s.engine.stop()
    # 3. corrupted metadata / WAV: listing survives, replay explains
    bad_json = os.path.join(cat.root, second, 'record.json')
    with open(bad_json, 'w') as f:
        f.write("{corrupt")
    entries = cat.list()
    assert len(entries) == 2
    assert entries[0][0] is None and 'unreadable' in entries[0][1]
    assert entries[1][0].id == good
    with open(bad_json, 'w') as f:
        json.dump(cat.load(good).meta | {'id': second}, f)
    wav_b = os.path.join(cat.root, second, 'B.wav')
    with open(wav_b, 'r+b') as f:
        f.truncate(60)
    assert cat.load(second) is not None
    res = cat.replay(second)
    assert res.status == 'unavailable' and 'B.wav' in res.reason
    # 4. a changed result is reported as a difference; the original is kept
    rec = cat.load(good)
    before = open(rec.wav_path('A'), 'rb').read()
    pcm = rec.pcm('A').copy()
    pcm[1000:1100] += 7
    with wave.open(rec.wav_path('A'), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    res = cat.replay(good)
    assert res.status == 'mismatch' and res.outputs == {'A': False, 'B': True, 'monitor': True}
    assert "differs" in res.text and "A" in res.text
    assert open(rec.wav_path('A'), 'rb').read() != before      # our edit, not a replay overwrite
    assert np.array_equal(read_wav(rec.wav_path('A')), pcm)


def test_replay_progress_and_cancel():
    cat = _fresh_catalog('cancel')
    s, cut = _run_experiment(cat, commands=[('start', 0, {})], until=_sec(2.0))
    try:
        rid = cat.save(cut)
    finally:
        s.engine.stop()
    seen = []
    res = cat.replay(rid, progress=seen.append)
    assert res.status == 'match' and seen and seen[-1] == 1.0
    res = cat.replay(rid, cancel=lambda: True)
    assert res.status == 'cancelled'
    # subprocess variant (used by the live UI): same verdict, progress, cancel
    seen = []
    res = cat.replay_in_subprocess(rid, progress=seen.append)
    assert res.status == 'match' and res.outputs == {'A': True, 'B': True, 'monitor': True}
    assert seen and seen[-1] == 1.0
    assert cat.replay_in_subprocess(rid, cancel=lambda: True).status == 'cancelled'
    assert cat.replay_in_subprocess('nope').status == 'unavailable'
    assert cat.load(rid).pcm('A').shape[0] == cut.n_frames        # untouched


def test_ui_headless_save_catalog_player_replay():
    import pygame
    import demo_bench as db
    cat = _fresh_catalog('ui')
    scene = load_scene(DEMO_AB)
    runner = DemoRunner(scene)
    out_frames = []

    class _Fake:
        def stop(self): pass
        def close(self): pass

    eng = LiveEngine(runner, output_factory=lambda cb: _Fake(), record_root=cat.tmp_root)
    pygame.init()
    app = db.BenchApp(scene, eng, catalog=cat)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    try:
        # pull the audio callback like a device would, so the live queue drains
        stop_pull = threading.Event()

        def pull():
            while not stop_pull.is_set():
                out = np.zeros((512, 2), np.int16)
                eng._audio_cb(out, 512, None, None)
                out_frames.append(out.copy())
                time.sleep(512 / SR)
        th = threading.Thread(target=pull, daemon=True)
        th.start()

        # Save before Start -> short message, nothing saved
        rect = app.lab_buttons['save']
        assert app.press((rect[0] + 3, rect[1] + 3), 1) == 'save'
        assert _wait(lambda: (app.tick(), app.status.startswith("Nothing to save"))[1])
        assert app.mode == 'live' and cat.ids() == []

        rect = app.buttons['start'][0]
        app.press((rect[0] + 3, rect[1] + 3), 1)
        assert _wait(lambda: eng.snapshot()['gen'] >= 2, 30)
        app.set_param('harm', 0.5)
        assert app.key('2') == 'select:B'
        assert _wait(lambda: eng.snapshot()['selected'] == 'B')
        time.sleep(0.3)
        # Save -> cut fixed, form opens; typing does not extend the record
        rect = app.lab_buttons['save']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        assert _wait(lambda: (app.tick(), app.mode == 'save')[1])
        end_fixed = app.save_form['cut'].end_sample
        assert app.key('r') is None                        # hotkeys inert while typing
        for ch in "Проверка S4":
            app.text_input(ch)
        assert app.save_form['title'].endswith("Проверка S4")
        assert app.key('tab') == 'save:field'
        app.text_input("заметка")
        app.draw(screen, font, small)
        time.sleep(0.2)
        assert app.key('return') == 'save:ok'
        assert _wait(lambda: app.status.startswith("Saved"), 20), app.status
        assert app.mode == 'live'
        rid = cat.ids()[0]
        rec = cat.load(rid)
        assert rec.meta['end_sample'] == end_fixed and rec.note == "заметка"
        assert rec.title.endswith("Проверка S4")
        assert eng.snapshot()['running'] and eng.snapshot()['gen'] >= 2     # live continues

        # Catalog: list, select, play (live muted), stop, replay, play replay, back
        rect = app.lab_buttons['catalog']
        assert app.press((rect[0] + 3, rect[1] + 3), 1) == 'catalog'
        assert app.mode == 'catalog' and len(app.cat['entries']) == 1
        app.draw(screen, font, small)
        assert app.press((app._list_rect(0)[0] + 5, app._list_rect(0)[1] + 5), 1) == 'record:0'
        r = app._catalog_rects()
        assert app.press((r['play_mon'][0] + 3, r['play_mon'][1] + 3), 1) == 'play:monitor'
        assert eng.playing
        time.sleep(0.15)
        played = np.concatenate(out_frames[-3:], axis=0)
        ref = rec.pcm('monitor')
        # what the device got is the record's PCM (a contiguous slice of it)
        pos, n = eng.play_pos
        assert pos > 0
        last = out_frames[-1]
        lo = max(0, pos - 8192)
        found = any(np.array_equal(last, ref[k:k + 512])
                    for k in range(lo, min(pos, len(ref) - 512) + 1))
        assert found, "device output is not a slice of the record PCM"
        assert app.press((r['stop'][0] + 3, r['stop'][1] + 3), 1) == 'play:stop'
        assert not eng.playing
        assert app.press((r['play_A'][0] + 3, r['play_A'][1] + 3), 1) == 'play:A'
        assert app.press((r['play_B'][0] + 3, r['play_B'][1] + 3), 1) == 'play:B'
        eng.stop_play()
        assert app.press((r['replay'][0] + 3, r['replay'][1] + 3), 1) == 'replay'
        assert _wait(lambda: (app.tick(), app.cat['thread'] is None)[1], 60)
        app.draw(screen, font, small)
        assert app.status == "Replay matched", app.status
        assert app.cat['play_label'] == "Play replay"
        assert app.press((r['play_replay'][0] + 3, r['play_replay'][1] + 3), 1) == 'play:replay'
        assert eng.playing
        pygame.image.save(screen, os.path.join(ART, "_demo_bench_catalog.png"))
        assert app.key('escape') == 'back'
        assert app.mode == 'live' and not eng.playing
        # Open in bench: new session in the record's INITIAL state, not running
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        app.press((app._list_rect(0)[0] + 5, app._list_rect(0)[1] + 5), 1)
        stop_pull.set()
        th.join(timeout=2)
        assert app.press((r['open'][0] + 3, r['open'][1] + 3), 1) == 'open'
        assert app.mode == 'live' and app.engine is not eng
        eng = app.engine
        snap = eng.snapshot()
        assert not snap['running'] and snap['gen'] == 0
        assert np.array_equal(snap['grid'], scene.initial_grid())     # initial field
        # harm 0.5 / side B were set AFTER Start -> not part of the initial state
        assert snap['sides']['A'][1]['harm'] == 0 and snap['selected'] == 'A'
        assert app.status.startswith("Opened in bench")
        app.draw(screen, font, small)
        # the new session needs its blocks pulled like a device would
        stop_pull.clear()

        def pull2():
            while not stop_pull.is_set():
                out = np.zeros((512, 2), np.int16)
                eng._audio_cb(out, 512, None, None)
                time.sleep(512 / SR)
        th2 = threading.Thread(target=pull2, daemon=True)
        th2.start()
        rect = app.buttons['start'][0]
        app.press((rect[0] + 3, rect[1] + 3), 1)
        assert _wait(lambda: eng.snapshot()['gen'] >= 1, 30)
        stop_pull.set()
        th2.join(timeout=2)
    finally:
        app.engine.stop()
        pygame.quit()


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
    os.makedirs(CAT_ROOT, exist_ok=True)
    sys.exit(1 if _run() else 0)
