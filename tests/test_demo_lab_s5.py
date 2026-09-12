#!/usr/bin/env python3
"""S5 acceptance tests: exact continuation from a snapshot + branches.

    python tests/test_demo_lab_s5.py      # stdlib runner

1. continuity for the five engines (in-process + fresh process)
2. hard boundaries (crossfade, tails, before a step, pause, stop, mixed
   engines, param change, scheduled command); export is side-effect free
3. branch: parent -> continue -> change -> save -> reload -> check -> continue
4. control + clocks after a restore (copy / factory / memory / stop / restart)
5. compatibility (S4 record) + errors (corrupt snapshot, unsupported engine,
   save failure)
6. headless UI: Continue / Open field anew / Derived from / Check reproducibility
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR                                        # noqa: E402
from casynth_lab import load_scene, DemoRunner, BLOCK, registry      # noqa: E402
from casynth_lab.audio_out import LiveEngine                         # noqa: E402
from casynth_lab.catalog import (Catalog, CatalogError, bench_scene,   # noqa: E402
                                 STATE_END, RECORD_FORMAT)
from casynth_lab.engine_api import SoundEngine, supports_snapshot     # noqa: E402
from casynth_lab.registry import EngineSpec, register, unregister    # noqa: E402
from casynth_lab.runner import RUNNER_STATE_VERSION                  # noqa: E402
from casynth_lab.snapshot import save_state, load_state, copy_state, SnapshotError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
ART = os.path.join(ROOT, "artifacts")
CAT_ROOT = os.path.join(ART, "_catalog_test_s5")
OUTPUTS = ('A', 'B', 'monitor')
ENGINES = ('fft2d', 'walsh', 'random', 'laplacian', 'granulo')


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


def _drive(runner, n_blocks, commands=(), collect=True):
    """Post `commands` = [(kind, at_or_None, args)] and render n blocks.
    Returns {output: [blocks]} (or None)."""
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    got = {o: [] for o in OUTPUTS} if collect else None
    for _ in range(n_blocks):
        b = runner.next_block()
        if collect:
            for o in OUTPUTS:
                got[o].append(b.get(o))
    return got


def _same(got1, got2):
    for o in OUTPUTS:
        a, b = np.concatenate(got1[o]), np.concatenate(got2[o])
        if a.shape != b.shape or not np.array_equal(a, b):
            return False, o
    return True, ''


def _loud(got):
    return max(int(np.abs(np.concatenate(got[o])).max()) for o in OUTPUTS)


def _roundtrip(state, name='s'):
    d = os.path.join(CAT_ROOT, '_states')
    os.makedirs(d, exist_ok=True)
    save_state(d, name, state)
    return load_state(d, name)


def _dir_digest(path):
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            h.update(name.encode())
            with open(p, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


# after the snapshot both runners get the SAME commands: control after a
# restore must behave like the continuous session (copy / factory / memory /
# engine switch / stop / restart / pause / paint / volume)
AFTER = [
    ('set_cell', None, {'r': 1, 'c': 1, 'v': 1}),
    ('set_param', 4, {'side': 'A', 'name': 'n', 'value': 7}),
    ('set_engine', 10, {'side': 'B', 'engine_id': 'walsh'}),
    ('set_engine', 20, {'side': 'B', 'engine_id': 'laplacian'}),    # memory
    ('copy_side', 30, {'src': 'A', 'dst': 'B'}),
    ('select', 40, {'side': 'A'}),
    ('vol', 50, {'value': 0.4}),
    ('pause', 60, {'on': True}),
    ('pause', 70, {'on': False}),
    ('factory', 80, {}),
    ('stop', 120, {}),
    ('start', 130, {}),
    ('reset', 160, {}),
]


def _after(base):
    return [(k, (None if at is None else base + at * BLOCK), a) for k, at, a in AFTER]


# =============================================================================
def test_continuity_five_engines_in_process_and_fresh_process():
    scene = load_scene(DEMO_AB)
    states, expect = {}, {}
    for i, eid in enumerate(ENGINES):
        other = ENGINES[(i + 1) % len(ENGINES)]
        r = DemoRunner(scene)
        pre = [('start', 0, {}),
               ('set_engine', 0, {'side': 'A', 'engine_id': eid}),
               ('set_engine', 0, {'side': 'B', 'engine_id': other}),
               ('select', 30 * BLOCK, {'side': 'B'})]
        _drive(r, 120, pre, collect=False)
        assert r.gen >= 1
        st = r.export_state()
        assert supports_snapshot(r.sides['A'].engine) and st['sides']['A']['engine']['engine_id'] == eid
        r2 = DemoRunner.from_state(_roundtrip(st, eid))
        base = r.out_samples
        g1 = _drive(r, 200, _after(base))
        g2 = _drive(r2, 200, _after(base))
        ok, o = _same(g1, g2)
        assert ok, f"{eid}: continued output differs on {o}"
        assert _loud(g1) > 500, f"{eid}: continuation is (nearly) silent"
        assert np.array_equal(r.grid, r2.grid) and r.gen == r2.gen
        assert (r.t_samples, r.ca_samples, r.out_samples) == (r2.t_samples, r2.ca_samples,
                                                              r2.out_samples)
        assert [j for j in r.journal if j[0] >= base] == r2.journal
        assert r.side_settings() == r2.side_settings()
        assert r.param_memory() == r2.param_memory()
        states[eid] = st
        expect[eid] = {o: np.concatenate(g1[o]) for o in OUTPUTS}
    # fresh process: restore from the files and render the same commands
    d = os.path.join(CAT_ROOT, '_states')
    out = os.path.join(d, 'fresh_out.npz')
    code = (
        "import sys, json, numpy as np; sys.path.insert(0, %r)\n"
        "from casynth_lab import DemoRunner, BLOCK\n"
        "from casynth_lab.snapshot import load_state\n"
        "sys.path.insert(0, %r); import test_demo_lab_s5 as T\n"
        "res = {}\n"
        "for eid in T.ENGINES:\n"
        "    r = DemoRunner.from_state(load_state(%r, eid))\n"
        "    g = T._drive(r, 200, T._after(r.out_samples))\n"
        "    for o in T.OUTPUTS: res[eid + '.' + o] = np.concatenate(g[o])\n"
        "    res[eid + '.grid'] = r.grid; res[eid + '.gen'] = np.array(r.gen)\n"
        "np.savez(%r, **res)\n"
        % (ROOT, os.path.dirname(os.path.abspath(__file__)), d, out))
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding='utf-8', cwd=ROOT)
    assert p.returncode == 0, p.stderr
    with np.load(out) as z:
        for eid in ENGINES:
            for o in OUTPUTS:
                assert np.array_equal(z[f"{eid}.{o}"], expect[eid][o]), f"fresh {eid}: {o}"


def test_hard_boundaries_and_side_effect_free_export():
    scene = load_scene(DEMO_AB)

    def check(name, pre, n_pre, after=None, tail_n=150):
        r = DemoRunner(scene)
        _drive(r, n_pre, pre, collect=False)
        st = r.export_state()
        st_again = r.export_state()                     # export twice: same
        assert json.dumps(st['pending']) == json.dumps(st_again['pending'])
        assert np.array_equal(st['grid'], st_again['grid'])
        # the same experiment WITHOUT an export must sound identical
        r_ref = DemoRunner(scene)
        _drive(r_ref, n_pre, pre, collect=False)
        after = after if after is not None else _after(r.out_samples)
        g_ref = _drive(r_ref, tail_n, list(after))
        g_live = _drive(r, tail_n, list(after))
        ok, o = _same(g_ref, g_live)
        assert ok, f"{name}: export changed the live session ({o})"
        # restore twice from the files: identical to each other and to live
        for k in range(2):
            r2 = DemoRunner.from_state(_roundtrip(st, 'edge'))
            g2 = _drive(r2, tail_n, list(after))
            ok, o = _same(g_live, g2)
            assert ok, f"{name} (restore {k}): differs on {o}"
            assert np.array_equal(r2.grid, r.grid) and r2.gen == r.gen
        return r, st

    import math
    step_samples = SR / scene.rate_hz
    start = [('start', 0, {})]
    # inside an A/B crossfade (20 ms = 882 samples > 2 blocks)
    r, st = check("crossfade", start + [('select', 40 * BLOCK, {'side': 'B'})], 41)
    assert st['xfade_from'] == 'A' and 0 < st['xfade_pos'] < 882
    # active tails: a field change just before the cut (old modes ring out)
    r, st = check("tails", start + [('set_cell', 60 * BLOCK, {'r': 4, 'c': 4, 'v': 1}),
                                    ('set_cell', 60 * BLOCK, {'r': 4, 'c': 5, 'v': 1})], 62)
    assert st['exc'] is not None
    # the block right before a CA step: step g happens in block ceil(g*step/BLOCK)
    n = math.ceil(2 * step_samples / BLOCK)               # the NEXT block does gen 2
    r, st = check("before step", start, n)
    assert st['gen'] == 1
    r2 = DemoRunner.from_state(_roundtrip(st, 'step'))
    r2.next_block()
    assert r2.gen == 2
    # Pause CA (running, automaton frozen): sound continues, field stays
    r, st = check("pause", start + [('pause', 50 * BLOCK, {'on': True})], 60,
                  after=_after(60 * BLOCK)[:1] + [('pause', 60 * BLOCK + 20 * BLOCK, {'on': False})])
    assert st['paused'] and st['running']
    # Stop: stopped state survives; a later un-pause starts from the beginning
    r, st = check("stop", start + [('stop', 50 * BLOCK, {})], 55,
                  after=[('pause', 55 * BLOCK + 10 * BLOCK, {'on': False})])
    assert not st['running'] and st['paused']
    # different engines on A/B + a parameter change + a scheduled command
    r, st = check("mixed+scheduled",
                  start + [('set_engine', 0, {'side': 'A', 'engine_id': 'granulo'}),
                           ('set_param', 20 * BLOCK, {'side': 'B', 'name': 'harm', 'value': 0.8}),
                           ('set_param', 200 * BLOCK, {'side': 'B', 'name': 'shape', 'value': 0.5}),
                           ('select', 205 * BLOCK, {'side': 'B'})], 100)
    assert [c['kind'] for c in st['pending']] == ['set_param', 'select']
    assert st['sides']['A']['engine_id'] == 'granulo' and st['sides']['B']['params']['harm'] == 0.8
    # the restored runner really applies the queued commands at their time
    r2 = DemoRunner.from_state(_roundtrip(st, 'sched'))
    _drive(r2, 120, collect=False)
    kinds = [(t, k) for (t, _s, k, _a) in r2.journal if k != 'step']
    assert (200 * BLOCK, 'set_param') in kinds and (205 * BLOCK, 'select') in kinds
    assert r2.sides['B'].params['shape'] == 0.5 and r2.selected == 'B'


class _Session:
    """A live session with a test sink that records every block it receives."""

    def __init__(self, catalog, runner=None, paced=True, **eng_kw):
        self.runner = runner or DemoRunner(load_scene(DEMO_AB))
        self.got = {o: [] for o in OUTPUTS}
        self.first_running = None
        self.n = 0
        self.paced = paced
        self._t0 = None
        self.engine = LiveEngine(self.runner, sink=self._sink, record_root=catalog.tmp_root,
                                 **eng_kw)

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

    def post_at(self, commands):
        for kind, at, args in commands:
            self.run_until(max(0, at - 6 * BLOCK))
            self.engine.post(kind, at=at, **args)

    def pcm(self, o, start, end):
        arr = np.concatenate(self.got[o], axis=0)
        s = start - self.first_running
        return arr[s:s + (end - start)]


def _parent_session(cat, seconds=1.2):
    s = _Session(cat)
    s.engine.start()
    s.post_at([('start', _sec(0.1), {}),
               ('set_param', _sec(0.3), {'side': 'B', 'name': 'harm', 'value': 0.25}),
               ('select', _sec(0.5), {'side': 'B'})])
    s.run_until(_sec(seconds))
    kind, cut = s.engine.cut_now()
    assert kind == 'ok' and cut is not None and cut.end_snapshot is not None
    return s, cut


def test_branch_parent_continue_change_save_reload_check_continue_again():
    cat = _fresh_catalog('branch')
    s, cut = _parent_session(cat)
    try:
        pid = cat.save(cut, "Исходная S5", "родитель")
    finally:
        s.engine.stop()
    parent = cat.load(pid)
    assert parent.meta['format'] == RECORD_FORMAT and parent.origin_kind == 'fresh'
    assert parent.parent_id is None and parent.can_continue() == (True, "")
    assert parent.meta['snapshot']['end'] == {'json': 'state_end.json', 'npz': 'state_end.npz'}
    assert parent.meta['snapshot']['origin'] is None
    parent_digest = _dir_digest(parent.dir)
    n_before = len(cat.ids())

    # Continue: restore (no history replayed), clocks frozen until driven
    runner, state, rec = cat.continue_runner(pid)
    t_out, gen = runner.out_samples, runner.gen
    assert t_out == parent.meta['end_sample'] == state['out_samples']
    assert runner.running and not runner.paused and runner.selected == 'B'
    assert runner.sides['B'].params['harm'] == 0.25
    time.sleep(0.3)
    assert (runner.out_samples, runner.gen) == (t_out, gen)
    assert len(cat.ids()) == n_before                     # opening creates no record

    # the continued session: change harm on B, paint, save -> a branch record
    c = _Session(cat, runner=runner, origin_snapshot=state, parent_record_id=pid)
    c.engine.start()
    c.post_at([('set_param', t_out + _sec(0.3), {'side': 'B', 'name': 'harm', 'value': 0.75}),
               ('set_cell', t_out + _sec(0.5), {'r': 3, 'c': 3, 'v': 1})])
    c.run_until(t_out + _sec(1.0))
    kind, bcut = c.engine.cut_now()
    assert kind == 'ok' and bcut.origin_kind == 'snapshot' and bcut.parent_record_id == pid
    assert bcut.origin_sample == t_out == bcut.audio_start_sample    # audio starts AT the snapshot
    first_blocks = {o: np.concatenate(c.got[o][:20]) for o in OUTPUTS}
    try:
        bid = cat.save(bcut, "Вариант S5", "ветка")
    finally:
        c.engine.stop()
    branch = cat.load(bid)
    assert branch.parent_id == pid and branch.origin_kind == 'snapshot'
    assert branch.meta['snapshot']['origin'] == {'json': 'state_origin.json',
                                                 'npz': 'state_origin.npz'}
    assert branch.meta['snapshot']['origin_seq'] == state['seq']
    assert branch.can_continue() == (True, "")
    assert cat.parent_of(branch) == ("Исходная S5", True)
    # own WAVs == the blocks of the continued session; parent's audio not glued in
    for o in OUTPUTS:
        assert np.array_equal(branch.pcm(o), c.pcm(o, bcut.audio_start_sample, bcut.end_sample))
    assert branch.meta['journal'][0]['out_sample'] >= t_out
    assert any(j['kind'] == 'set_param' and j['args']['value'] == 0.75
               for j in branch.meta['journal'])
    # the parent is untouched
    assert _dir_digest(parent.dir) == parent_digest

    # check reproducibility (in-process + fresh process): all three tracks
    res = cat.replay(bid)
    assert res.status == 'match' and all(res.outputs.values()), res.text
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "casynth_lab.catalog", "replay",
                        cat.root, bid], capture_output=True, text=True, encoding='utf-8',
                       cwd=ROOT)
    assert p.returncode == 0, p.stderr
    last = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith('{')][-1])
    assert last['status'] == 'match', last
    # ... also when the parent's folder is temporarily gone
    hidden = os.path.join(cat.root, '.hidden_' + pid)
    os.rename(parent.dir, hidden)
    try:
        assert cat.replay(bid).status == 'match'
        entries = cat.list()
        assert all(err is None for _r, err in entries) and [r.id for r, _e in entries] == [bid]
        assert cat.parent_of(cat.load(bid)) == (pid, False)
        r_again, _st, _rec = cat.continue_runner(bid)        # continuing needs no parent
        assert r_again.sides['B'].params['harm'] == 0.75
    finally:
        os.rename(hidden, parent.dir)

    # Continue the parent again: starts from the SAME pristine snapshot
    runner2, state2, _ = cat.continue_runner(pid)
    assert runner2.sides['B'].params['harm'] == 0.25
    c2 = _Session(cat, runner=runner2, origin_snapshot=state2, parent_record_id=pid, paced=False)
    c2.engine.start()
    c2.run_until(t_out + 25 * BLOCK)
    c2.engine.stop()
    for o in OUTPUTS:
        assert np.array_equal(np.concatenate(c2.got[o][:20]), first_blocks[o]), o
    # continuing the branch makes the branch the parent of what is saved next
    runner3, state3, _ = cat.continue_runner(bid)
    c3 = _Session(cat, runner=runner3, origin_snapshot=state3, parent_record_id=bid, paced=False)
    c3.engine.start()
    c3.run_until(runner3.out_samples + 30 * BLOCK)
    kind, cut3 = c3.engine.cut_now()
    c3.engine.stop()
    gid = cat.save(cut3, "Внук S5")
    assert cat.load(gid).parent_id == bid
    assert cat.replay(gid).status == 'match'
    # several saves from one continued session share the parent + start point
    assert cat.load(bid).meta['origin_sample'] == t_out


def test_stop_restart_after_continue_start_fresh_recordings_with_parent_link():
    cat = _fresh_catalog('fresh_after')
    s, cut = _parent_session(cat, seconds=0.9)
    try:
        pid = cat.save(cut, "P")
    finally:
        s.engine.stop()
    runner, state, _ = cat.continue_runner(pid)
    t0 = runner.out_samples
    c = _Session(cat, runner=runner, origin_snapshot=state, parent_record_id=pid,
                 record_seconds=1.0)
    c.engine.start()
    c.post_at([('reset', t0 + _sec(0.4), {})])
    c.run_until(t0 + _sec(0.8))
    kind, cut2 = c.engine.cut_now()
    assert cut2.origin_kind == 'fresh' and cut2.parent_record_id == pid
    assert cut2.origin_sample > t0                       # a Restart began a new recording
    rid = cat.save(cut2, "after restart")
    rec = cat.load(rid)
    assert rec.origin_kind == 'fresh' and rec.parent_id == pid
    assert rec.meta['snapshot']['origin'] is None
    assert cat.replay(rid).status == 'match'             # fresh conditions, S4 path
    # Stop in a continued session: the snapshot keeps the stop; a later start is fresh
    c.engine.post('stop')
    _wait(lambda: not c.engine.snapshot()['running'])
    kind, cut3 = c.engine.cut_now()
    assert cut3.end_snapshot['running'] is False
    sid = cat.save(cut3, "stopped")
    r_stop, st_stop, _ = cat.continue_runner(sid)
    assert not r_stop.running
    c.engine.stop()
    # the window of a continued session is still bounded (record_seconds)
    runner4, state4, _ = cat.continue_runner(pid)
    c4 = _Session(cat, runner=runner4, origin_snapshot=state4, parent_record_id=pid,
                  paced=False, record_seconds=0.5)
    c4.engine.start()
    c4.run_until(runner4.out_samples + _sec(1.2))
    kind, cut4 = c4.engine.cut_now()
    c4.engine.stop()
    assert cut4.n_frames <= c4.engine.recorder.ring_frames
    assert cut4.audio_start_sample > cut4.origin_sample     # the ring rolled
    rid4 = cat.save(cut4, "rolled")
    assert cat.replay(rid4).status == 'match'


class _NoSnapEngine(SoundEngine):
    """A bench engine WITHOUT snapshot support (still renders)."""

    def init(self, grid, exc, gain=0.0):
        self.t = 0

    def update_field(self, grid, exc):
        pass

    def render(self, gain, t_samples):
        n = BLOCK
        x = np.sin(2 * np.pi * 220.0 * (np.arange(n) + t_samples) / SR) * 0.2 * gain
        buf = np.repeat((x * 32767).astype(np.int16)[:, None], 2, axis=1)
        return buf, float(abs(x).max()), 0

    def reset(self, gain=0.0):
        pass


def test_compatibility_s4_records_and_errors():
    cat = _fresh_catalog('compat')
    s, cut = _parent_session(cat, seconds=0.9)
    try:
        pid = cat.save(cut, "S5 parent")
    finally:
        s.engine.stop()
    # an S4-style record: format 1, no snapshot keys, no state files
    old_id = pid + 'old'
    shutil.copytree(os.path.join(cat.root, pid), os.path.join(cat.root, old_id))
    for name in ('state_end.json', 'state_end.npz'):
        os.remove(os.path.join(cat.root, old_id, name))
    mp = os.path.join(cat.root, old_id, 'record.json')
    meta = json.load(open(mp, encoding='utf-8'))
    meta['format'] = 1
    for k in ('origin_kind', 'parent_record_id', 'snapshot'):
        meta.pop(k, None)
    json.dump(meta, open(mp, 'w', encoding='utf-8'))
    old = cat.load(old_id)
    assert old.pcm('monitor').shape[0] == old.meta['end_sample'] - old.meta['audio_start_sample']
    ok, why = old.can_continue()
    assert not ok and 'S4' in why
    with _raises(CatalogError):
        cat.continue_runner(old_id)
    sc, vol = bench_scene(old)                              # Open field anew still works
    assert sc.variants['B'][1]['harm'] == 0.25
    assert cat.replay(old_id).status == 'match'
    assert not os.path.exists(os.path.join(cat.root, old_id, 'state_end.json'))  # no migration
    # corrupted / incompatible snapshots are refused, never replaced by a fresh start
    bad = pid + 'bad'
    shutil.copytree(os.path.join(cat.root, pid), os.path.join(cat.root, bad))
    with open(os.path.join(cat.root, bad, 'state_end.npz'), 'r+b') as f:
        f.truncate(100)
    assert cat.load(bad).can_continue()[0]                  # light check passes ...
    with _raises(CatalogError):                             # ... the real load does not
        cat.continue_runner(bad)
    ver = pid + 'ver'
    shutil.copytree(os.path.join(cat.root, pid), os.path.join(cat.root, ver))
    mp = os.path.join(cat.root, ver, 'record.json')
    meta = json.load(open(mp, encoding='utf-8'))
    meta['snapshot']['runner_state_version'] = RUNNER_STATE_VERSION + 1
    json.dump(meta, open(mp, 'w', encoding='utf-8'))
    ok, why = cat.load(ver).can_continue()
    assert not ok and 'version' in why
    grid_ver = pid + 'grid'
    shutil.copytree(os.path.join(cat.root, pid), os.path.join(cat.root, grid_ver))
    st = load_state(os.path.join(cat.root, grid_ver), STATE_END)
    st['grid'] = st['grid'][:-1]
    save_state(os.path.join(cat.root, grid_ver), STATE_END, st)
    with _raises(CatalogError):
        cat.continue_runner(grid_ver)
    st = load_state(os.path.join(cat.root, grid_ver), STATE_END)
    st['sides']['A']['engine']['phase'] = st['sides']['A']['engine']['phase'][:10]
    save_state(os.path.join(cat.root, grid_ver), STATE_END, st)
    with _raises(CatalogError):
        cat.continue_runner(grid_ver)
    # an engine without snapshot support: the bench works, save works, no
    # false "continue" offer; unregistered engine -> continue unavailable
    register(EngineSpec('nosnap', 'NoSnap', [('n', 'part', 1, 20, True, 8)],
                        lambda ctx, params: _NoSnapEngine(ctx, params)))
    try:
        r = DemoRunner(load_scene(DEMO_AB))
        r.post('set_engine', side='A', engine_id='nosnap')
        for _ in range(4):
            r.next_block()               # the engine-switch fade-in ends before Start
        assert r.snapshot_support()[0] is False
        with _raises(ValueError):
            r.export_state()
        s2 = _Session(cat, runner=r, paced=False)
        s2.engine.start()
        s2.engine.post('start')
        s2.run_until(_sec(0.5))
        kind, cut2 = s2.engine.cut_now()
        s2.engine.stop()
        assert cut2.end_snapshot is None and 'nosnap' in cut2.end_snapshot_reason
        nid = cat.save(cut2, "no snapshot engine")
        rec = cat.load(nid)
        ok, why = rec.can_continue()
        assert not ok and 'nosnap' in why and rec.meta['snapshot']['end'] is None
        assert not os.path.exists(os.path.join(rec.dir, 'state_end.json'))
        assert cat.replay(nid).status == 'match'
    finally:
        unregister('nosnap')
    ok, why = cat.load(nid).can_continue()
    assert not ok and 'not registered' in why
    # a failing save leaves no partial record and the parent intact
    import casynth_lab.catalog as catmod
    digest = _dir_digest(os.path.join(cat.root, pid))
    ids = cat.ids()
    real = catmod.save_state

    def boom(*a, **k):
        raise OSError("disk full (simulated)")
    catmod.save_state = boom
    try:
        with _raises(CatalogError):
            cat.save(cut, "will fail")
    finally:
        catmod.save_state = real
    assert cat.ids() == ids and _dir_digest(os.path.join(cat.root, pid)) == digest
    assert not [n for n in os.listdir(cat.root) if n.endswith('.partial')]
    # snapshot files never hold objects; loading never unpickles
    with _raises(SnapshotError):
        save_state(os.path.join(cat.root, pid), 'obj', {'x': np.array([object()])})
    assert copy_state(load_state(os.path.join(cat.root, pid), STATE_END))['seq'] == \
        load_state(os.path.join(cat.root, pid), STATE_END)['seq']


class _raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), f"expected {self.exc.__name__}, got {et}"
        return True


def test_ui_headless_continue_branch_parent_link():
    import pygame
    import demo_bench as db
    cat = _fresh_catalog('ui')
    scene = load_scene(DEMO_AB)

    class _Fake:
        def stop(self): pass
        def close(self): pass

    eng = LiveEngine(DemoRunner(scene), output_factory=lambda cb: _Fake(),
                     record_root=cat.tmp_root)
    pygame.init()
    app = db.BenchApp(scene, eng, catalog=cat)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    stop_pull = threading.Event()
    puller = {'th': None}

    def pull(engine):
        def run():
            while not stop_pull.is_set():
                out = np.zeros((512, 2), np.int16)
                engine._audio_cb(out, 512, None, None)
                time.sleep(512 / SR)
        stop_pull.clear()
        puller['th'] = threading.Thread(target=run, daemon=True)
        puller['th'].start()

    def unpull():
        stop_pull.set()
        puller['th'].join(timeout=2)

    def save_as(title):
        rect = app.lab_buttons['save']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        assert _wait(lambda: (app.tick(), app.mode == 'save')[1], 30)
        app.save_form['title'] = ''
        for ch in title:
            app.text_input(ch)
        assert app.key('return') == 'save:ok'
        assert _wait(lambda: app.status.startswith("Saved"), 30), app.status

    try:
        pull(app.engine)
        rect = app.buttons['pause'][0]
        app.press((rect[0] + 3, rect[1] + 3), 1)              # release pause = go
        assert _wait(lambda: app.engine.snapshot()['gen'] >= 1, 30)
        assert app.key('2') == 'select:B'
        assert _wait(lambda: app.engine.snapshot()['selected'] == 'B')
        app.set_param('harm', 0.25)
        assert _wait(lambda: app.engine.snapshot()['sides']['B'][1]['harm'] == 0.25)
        time.sleep(0.3)
        save_as("Исходная S5")
        pid = cat.ids()[0]
        # catalog -> Continue
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        assert app.mode == 'catalog'
        app.press((app._list_rect(0)[0] + 5, app._list_rect(0)[1] + 5), 1)
        app.draw(screen, font, small)
        r = app._catalog_rects()
        unpull()
        old_engine = app.engine
        assert app.press((r['continue'][0] + 3, r['continue'][1] + 3), 1) == 'continue'
        assert app.mode == 'live' and app.engine is not old_engine
        assert app.status.startswith("Continued"), app.status
        assert app.engine.recorder.origin_kind == 'snapshot'
        assert app.engine.recorder.parent_record_id == pid
        snap = app.engine.snapshot()
        assert snap['running'] and snap['selected'] == 'B' and snap['sides']['B'][1]['harm'] == 0.25
        assert snap['out_samples'] == cat.load(pid).meta['end_sample']
        pull(app.engine)
        assert _wait(lambda: app.engine.snapshot()['out_samples'] > snap['out_samples'] + _sec(0.3), 30)
        app.set_param('harm', 0.75)
        assert _wait(lambda: app.engine.snapshot()['sides']['B'][1]['harm'] == 0.75)
        time.sleep(0.3)
        save_as("Вариант S5")
        (bid,) = set(cat.ids()) - {pid}          # ids of one second sort randomly
        assert cat.load(bid).parent_id == pid
        # catalog: the variant shows its origin; the link selects the parent
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        row = [i for i, (rec, _e) in enumerate(app.cat['entries']) if rec and rec.id == bid][0]
        app.press((app._list_rect(row)[0] + 5, app._list_rect(row)[1] + 5), 1)
        assert app.selected_record().id == bid
        app.draw(screen, font, small)
        pygame.image.save(screen, os.path.join(ART, "_demo_bench_catalog_s5.png"))
        assert cat.parent_of(app.selected_record()) == ("Исходная S5", True)
        r = app._catalog_rects()
        assert app.press((r['parent'][0] + 3, r['parent'][1] + 3), 1) == 'parent'
        assert app.selected_record().id == pid
        # Check reproducibility of the variant (child process)
        app.press((app._list_rect(row)[0] + 5, app._list_rect(row)[1] + 5), 1)
        assert app.selected_record().id == bid
        assert app.press((r['replay'][0] + 3, r['replay'][1] + 3), 1) == 'replay'
        assert app.status.startswith("Checking reproducibility")
        assert _wait(lambda: (app.tick(), app.cat['thread'] is None)[1], 120)
        assert app.status == "Replay matched", app.status
        # Continue the parent again -> harm on B is 0.25 again
        app.press((r['parent'][0] + 3, r['parent'][1] + 3), 1)
        unpull()
        assert app.press((r['continue'][0] + 3, r['continue'][1] + 3), 1) == 'continue'
        assert app.engine.snapshot()['sides']['B'][1]['harm'] == 0.25
        assert cat.load(bid).meta['settings_at_end']['B']['engine_params']['harm'] == 0.75
        pull(app.engine)
        # Open field anew: fresh session, CA paused, parent link kept
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        app.press((app._list_rect(row)[0] + 5, app._list_rect(row)[1] + 5), 1)
        assert app.selected_record().id == bid
        unpull()
        assert app.press((r['open'][0] + 3, r['open'][1] + 3), 1) == 'open'
        snap = app.engine.snapshot()
        assert app.mode == 'live' and not snap['running'] and snap['gen'] == 0
        assert app.engine.recorder.parent_record_id == bid
        assert app.status.startswith("Opened field anew")
        pull(app.engine)
        # an S4-style record (no snapshot): Continue is refused, session kept
        old_id = pid + 'old'
        shutil.copytree(os.path.join(cat.root, pid), os.path.join(cat.root, old_id))
        mp = os.path.join(cat.root, old_id, 'record.json')
        meta = json.load(open(mp, encoding='utf-8'))
        meta['format'] = 1
        meta.pop('snapshot')
        json.dump(meta, open(mp, 'w', encoding='utf-8'))
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        idx = [i for i, (rec, _e) in enumerate(app.cat['entries']) if rec and rec.id == old_id][0]
        app.press((app._list_rect(idx)[0] + 5, app._list_rect(idx)[1] + 5), 1)
        app.draw(screen, font, small)
        cur = app.engine
        assert app.press((r['continue'][0] + 3, r['continue'][1] + 3), 1) == 'continue'
        assert app.engine is cur and app.mode == 'catalog'
        assert app.status.startswith("Cannot continue"), app.status
        assert app.key('escape') == 'back'
        unpull()
    finally:
        stop_pull.set()
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
        except Exception as e:           # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    os.makedirs(CAT_ROOT, exist_ok=True)
    sys.exit(1 if _run() else 0)
