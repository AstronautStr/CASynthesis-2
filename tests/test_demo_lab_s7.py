#!/usr/bin/env python3
"""S7 acceptance tests: catalog check after a code change -- report, pairs,
listening marks, synchronized saved / recomputed playback.

    python tests/test_demo_lab_s7.py      # stdlib runner

1. track classification + diagnostics (exact / A only / monitor only / 1 LSB /
   silence / int16 extremes / different length) -- no audibility verdict
2. batch check == single check == the session's PCM (history longer than the
   window; a branch from a snapshot with interventions and A/B switches)
3. errors have reasons, the catalog sources stay untouched
4. target provenance of the worker, runtime change between records, cancel
   inside a long record, partial failure, single check never overwrites a
   pair, a second run / a restart keep the old report and its mark
5. pair player: one cursor, switch at the expected sample, unity gain, no
   live audio mixed in, silence after the shorter version ends
6. headless UI: the three manual steps
Git work happens in ISOLATED repositories under artifacts/_s7/.
"""
import copy
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
from casynth_lab import load_scene, DemoRunner, BLOCK                # noqa: E402
from casynth_lab.audio_out import LiveEngine, PAIR_XFADE_SAMPLES     # noqa: E402
from casynth_lab.catalog import Catalog, CatalogError, read_wav, _pcm_to_wav  # noqa: E402
from casynth_lab import provenance as prov                           # noqa: E402
from casynth_lab import versions                                     # noqa: E402
from casynth_lab.runner import RUNNER_STATE_VERSION                  # noqa: E402
from casynth_lab.verify import (Verifier, VerifyError, compare_pcm, pcm_sha,   # noqa: E402
                                TRACK_EXACT, TRACK_DIFFERS, TRACK_FAILED, TRACK_UNCHECKED,
                                RUN_COMPLETE, RUN_CANCELLED, RUN_STOPPED, MARK_SAME,
                                MARK_DIFFERENT, finalize_run)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
ART = os.path.join(ROOT, "artifacts")
S7_ROOT = os.path.join(ART, "_s7")
OUTPUTS = ('A', 'B', 'monitor')


def _sec(s):
    return int(round(s * SR))


def _wait(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _rmtree(path):
    import stat

    def onexc(fn, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            fn(p)
        except OSError:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onexc=onexc)
    assert not os.path.exists(path), f"cannot remove {path}"


def _fresh_catalog(name, repo_root=None):
    root = os.path.join(S7_ROOT, name, 'lab_catalog', 'cat') if repo_root else \
        os.path.join(S7_ROOT, name)
    _rmtree(root)
    os.makedirs(os.path.join(root, '.tmp'), exist_ok=True)
    return Catalog(root, repo_root=repo_root)


class _raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), f"expected {self.exc.__name__}, got {et}"
        return True


# -- sessions -------------------------------------------------------------------------
class _Session:
    """A live session with a test sink that keeps every block it receives."""

    def __init__(self, catalog, runner=None, **eng_kw):
        self.runner = runner or DemoRunner(load_scene(DEMO_AB))
        self.got = {o: [] for o in OUTPUTS}
        self.first = None
        self.n = 0
        self._t0 = None
        self.engine = LiveEngine(self.runner, sink=self._sink, record_root=catalog.tmp_root,
                                 **eng_kw)

    def _sink(self, mon, blk):
        snap = self.engine.snapshot()
        if self.first is None:
            self.first = snap['out_samples'] - BLOCK
        if blk is not None:
            for o in OUTPUTS:
                self.got[o].append(blk.get(o))
        # paced like a device: commands posted "shortly before their time"
        # really land near it (an unpaced render runs seconds ahead)
        self.n += BLOCK
        if self._t0 is None:
            self._t0 = time.perf_counter()
        ahead = self.n / SR - (time.perf_counter() - self._t0)
        if ahead > 0.04:
            time.sleep(ahead - 0.04)

    def run_until(self, out_sample):
        assert _wait(lambda: self.engine.snapshot()['out_samples'] >= out_sample, 120)

    def post_at(self, commands):
        for kind, at, args in commands:
            self.run_until(max(0, at - 6 * BLOCK))
            self.engine.post(kind, at=at, **args)

    def pcm(self, o, start, end):
        arr = np.concatenate(self.got[o], axis=0)
        s = start - self.first
        return arr[s:s + (end - start)]


def _make_record(cat, title, seconds=2.0, commands=None, record_seconds=None):
    """A fresh-conditions record; returns (id, {output: the session's PCM of
    the saved window})."""
    kw = {} if record_seconds is None else {'record_seconds': record_seconds}
    s = _Session(cat, **kw)
    s.engine.start()
    cmds = [('start', _sec(0.1), {})] + list(commands or [])
    s.post_at(cmds)
    s.run_until(_sec(seconds))
    kind, cut = s.engine.cut_now()
    assert kind == 'ok' and cut is not None
    try:
        rid = cat.save(cut, title)
    finally:
        s.engine.stop()
    truth = {o: s.pcm(o, cut.audio_start_sample, cut.end_sample) for o in OUTPUTS}
    return rid, truth


def _make_branch(cat, pid, title, seconds=1.5, commands=None):
    runner, state, rec = cat.continue_runner(pid)
    t0 = runner.out_samples
    s = _Session(cat, runner=runner, origin_snapshot=state, parent_record_id=pid)
    s.engine.start()
    s.post_at([(k, t0 + at, a) for (k, at, a) in (commands or [])])
    s.run_until(t0 + _sec(seconds))
    kind, cut = s.engine.cut_now()
    assert kind == 'ok' and cut.origin_kind == 'snapshot'
    try:
        bid = cat.save(cut, title)
    finally:
        s.engine.stop()
    truth = {o: s.pcm(o, cut.audio_start_sample, cut.end_sample) for o in OUTPUTS}
    return bid, truth


def _clone_record(cat, rid, suffix, edit=None):
    """A copy of a record under a new id (optionally with edited WAVs / meta)."""
    src, dst = os.path.join(cat.root, rid), os.path.join(cat.root, rid + suffix)
    _rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(os.path.join(dst, 'replay'), ignore_errors=True)
    if edit is not None:
        edit(dst)
    return rid + suffix


def _rewrite_wav(path, fn):
    pcm = read_wav(path)
    _pcm_to_wav(fn(pcm), path)


def _edit_meta(d, fn):
    mp = os.path.join(d, 'record.json')
    meta = json.load(open(mp, encoding='utf-8'))
    fn(meta)
    json.dump(meta, open(mp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def _dir_digest(path, skip=()):
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, path).encode())
            with open(p, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


def _catalog_digest(cat):
    return {rid: _dir_digest(os.path.join(cat.root, rid), skip=('replay',)) for rid in cat.ids()}


# -- isolated repositories (child worker of another checkout) ------------------------------
def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(text.replace('\r\n', '\n').encode('utf-8'))


def _read(path):
    with open(path, 'rb') as f:
        return f.read().decode('utf-8')


def _make_repo(name):
    root = os.path.join(S7_ROOT, name)
    _rmtree(root)
    os.makedirs(root)
    for rel in prov.runtime_paths(ROOT):
        src, dst = os.path.join(ROOT, rel), os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    _write(os.path.join(root, '.gitignore'), "lab_catalog/\n__pycache__/\n")
    g = lambda *a: prov.git(root, *a)                   # noqa: E731
    g('init', '-q', '-b', 'master')
    g('config', 'user.email', 'test@example.com')
    g('config', 'user.name', 'test')
    g('config', 'core.autocrlf', 'true')
    g('config', 'gc.auto', '0')
    g('add', '-A')
    g('commit', '-q', '-m', 'base')
    return root


def _worker(root, cat_root, on_line=None, hold=None):
    """Run the check worker FROM `root` (its own casynth_lab); JSON lines are
    passed to on_line(msg, proc) as they arrive.  Returns (msgs, proc)."""
    cmd = [sys.executable, '-X', 'utf8', '-m', 'casynth_lab.verify', 'run', cat_root]
    proc = subprocess.Popen(cmd, cwd=root, env=versions.child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8')
    msgs = []
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith('{'):
            continue
        msg = json.loads(line)
        msgs.append(msg)
        if on_line is not None:
            on_line(msg, proc)
    proc.wait(timeout=120)
    return msgs, proc


# =============================================================================
def test_track_classification_and_diagnostics():
    n = 2000
    base = (np.sin(np.arange(n) * 0.01)[:, None] * 12000).astype(np.int16).repeat(2, axis=1)
    c = compare_pcm(base, base.copy())
    assert c['result'] == TRACK_EXACT and c['frac_diff'] == 0.0 and c['max_abs_diff'] == 0
    assert not c['length_mismatch'] and c['n_ref'] == c['n_new'] == n
    # a single sample one PCM unit off: differs, 1/n of the frames, max 1
    one = base.copy()
    one[777, 0] += 1
    c = compare_pcm(base, one)
    assert c['result'] == TRACK_DIFFERS and c['max_abs_diff'] == 1
    assert abs(c['frac_diff'] - 1.0 / n) < 1e-12 and not c['length_mismatch']
    # silence vs silence exact; silence vs anything differs
    z = np.zeros((n, 2), np.int16)
    assert compare_pcm(z, z.copy())['result'] == TRACK_EXACT
    assert compare_pcm(z, base)['result'] == TRACK_DIFFERS
    # int16 extremes: the difference is computed without overflow
    lo = np.full((n, 2), -32768, np.int16)
    hi = np.full((n, 2), 32767, np.int16)
    c = compare_pcm(lo, hi)
    assert c['result'] == TRACK_DIFFERS and c['max_abs_diff'] == 65535 and c['frac_diff'] == 1.0
    assert compare_pcm(lo, lo.copy())['result'] == TRACK_EXACT
    # different length: differs even when the common part is identical; no tail cut
    c = compare_pcm(base, base[:n - 10])
    assert c['result'] == TRACK_DIFFERS and c['length_mismatch']
    assert c['n_ref'] == n and c['n_new'] == n - 10 and c['max_abs_diff'] == 0
    c = compare_pcm(base[:100], base)
    assert c['result'] == TRACK_DIFFERS and c['length_mismatch'] and c['n_new'] == n
    assert compare_pcm(z[:0], z[:0])['result'] == TRACK_EXACT
    # ... and through real records: only A / only monitor / all exact
    cat = _fresh_catalog('classify')
    rid, truth = _make_record(cat, "base", seconds=1.2,
                              commands=[('select', _sec(0.5), {'side': 'B'})])
    a_only = _clone_record(cat, rid, 'a', lambda d: _rewrite_wav(
        os.path.join(d, 'A.wav'), lambda p: np.where(np.arange(len(p))[:, None] == 300, p + 1, p)))
    mon_only = _clone_record(cat, rid, 'm', lambda d: _rewrite_wav(
        os.path.join(d, 'monitor.wav'), lambda p: p[:-BLOCK]))
    run = Verifier(cat).run()
    assert run.status == RUN_COMPLETE and run.counts() == {TRACK_DIFFERS: 2, TRACK_EXACT: 1,
                                                           TRACK_FAILED: 0, TRACK_UNCHECKED: 0}
    e = run.result(rid)
    assert e['status'] == TRACK_EXACT and all(e['tracks'][o]['result'] == TRACK_EXACT
                                              for o in OUTPUTS)
    assert all(e['tracks'][o]['new_file'] is None for o in OUTPUTS)
    e = run.result(a_only)
    assert e['status'] == TRACK_DIFFERS
    assert e['tracks']['A']['result'] == TRACK_DIFFERS and e['tracks']['A']['max_abs_diff'] == 1
    assert e['tracks']['B']['result'] == TRACK_EXACT and e['tracks']['monitor']['result'] == TRACK_EXACT
    assert run.default_track(a_only) == 'A'
    e = run.result(mon_only)
    assert e['tracks']['monitor']['result'] == TRACK_DIFFERS and e['tracks']['monitor']['length_mismatch']
    assert e['tracks']['A']['result'] == TRACK_EXACT and e['tracks']['B']['result'] == TRACK_EXACT
    assert run.default_track(mon_only) == 'monitor'
    # no audibility verdict anywhere in the machine result; marks start empty
    for rr in (rid, a_only, mon_only):
        for o in OUTPUTS:
            assert run.mark(rr, o) is None
            assert 'audible' not in json.dumps(run.result(rr)['tracks'][o]).lower()
    # a differing track keeps the recomputed WAV; it equals the truth (the
    # original of this experiment) and the report stores both fingerprints
    t = e['tracks']['monitor']
    new = read_wav(os.path.join(run.dir, mon_only, t['new_file']))
    assert np.array_equal(new, truth['monitor']) and pcm_sha(new) == t['new_sha256']
    assert t['ref_sha256'] == pcm_sha(cat.load(mon_only).pcm('monitor'))


def test_batch_equals_single_replay_on_long_history_and_branch():
    cat = _fresh_catalog('batch')
    # a fresh record whose history is LONGER than the saved window
    pid, truth_p = _make_record(cat, "long history", seconds=3.0, record_seconds=1.0, commands=[
        ('set_param', _sec(0.4), {'side': 'B', 'name': 'harm', 'value': 0.3}),
        ('select', _sec(0.8), {'side': 'B'}),
        ('set_cell', _sec(1.2), {'r': 3, 'c': 4, 'v': 1}),
        ('set_engine', _sec(1.6), {'side': 'A', 'engine_id': 'walsh'}),
        ('select', _sec(2.1), {'side': 'A'}),
        ('vol', _sec(2.4), {'value': 0.6})])
    rec = cat.load(pid)
    assert rec.meta['audio_start_sample'] > rec.meta['origin_sample']       # the ring rolled
    # a branch from the snapshot with interventions and A/B switches
    bid, truth_b = _make_branch(cat, pid, "branch", seconds=1.6, commands=[
        ('set_cell', _sec(0.2), {'r': 5, 'c': 5, 'v': 1}),
        ('select', _sec(0.5), {'side': 'B'}),
        ('set_param', _sec(0.8), {'side': 'B', 'name': 'shape', 'value': 0.5}),
        ('select', _sec(1.1), {'side': 'A'}),
        ('pause', _sec(1.3), {'on': True})])
    # make every stored track differ (inverted PCM) so BOTH paths write output
    for rr in (pid, bid):
        for o in OUTPUTS:
            _rewrite_wav(cat.load(rr).wav_path(o), lambda p: (-p.astype(np.int32)).clip(
                -32768, 32767).astype(np.int16))
    ver = Verifier(cat)
    run = ver.run_in_subprocess()                        # the real worker path
    assert run.status == RUN_COMPLETE, run.status_label
    for rr, truth in ((pid, truth_p), (bid, truth_b)):
        single = cat.replay(rr)
        assert single.status == 'mismatch' and not any(single.outputs.values())
        e = run.result(rr)
        assert e['status'] == TRACK_DIFFERS
        for o in OUTPUTS:
            batch = read_wav(os.path.join(run.dir, rr, e['tracks'][o]['new_file']))
            one = read_wav(os.path.join(single.replay_dir, f"{o}.wav"))
            assert np.array_equal(batch, one), f"{rr} {o}: batch != single"
            assert np.array_equal(batch, truth[o]), f"{rr} {o}: batch != session"
            assert e['tracks'][o]['n_ref'] == e['tracks'][o]['n_new'] == len(truth[o])
            assert np.abs(truth[o]).max() > 200
    # the in-process run computes the same
    run2 = ver.run()
    for rr in (pid, bid):
        for o in OUTPUTS:
            assert (run2.result(rr)['tracks'][o]['new_sha256']
                    == run.result(rr)['tracks'][o]['new_sha256'])


def test_errors_have_reasons_and_sources_stay_untouched():
    cat = _fresh_catalog('errors')
    pid, _ = _make_record(cat, "parent", seconds=1.2,
                          commands=[('select', _sec(0.5), {'side': 'B'})])
    bid, _ = _make_branch(cat, pid, "branch", seconds=0.8)
    # an old record: format 1, no provenance, no snapshot -> checked, origin unknown
    old = _clone_record(cat, pid, 'old', lambda d: (
        [os.remove(os.path.join(d, n)) for n in ('state_end.json', 'state_end.npz')],
        _edit_meta(d, lambda m: (m.__setitem__('format', 1),
                                 [m.pop(k, None) for k in ('provenance', 'pin', 'status_reason',
                                                           'snapshot', 'origin_kind',
                                                           'parent_record_id')]))))
    # a pinned record whose commit does not exist here -> still checked with this code
    lost = _clone_record(cat, pid, 'lost', lambda d: _edit_meta(d, lambda m: (
        m.__setitem__('provenance', dict(m.get('provenance') or {}, commit='0' * 40,
                                         match='clean', digest='f' * 64)),
        m.__setitem__('status', 'pinned'),
        m.__setitem__('pin', {'commit': '0' * 40, 'ref': prov.pin_ref('0' * 40)}))))
    # a branch whose origin snapshot has another runner state version
    ver_bad = _clone_record(cat, bid, 'ver', lambda d: _edit_meta(d, lambda m: m['snapshot']
                            .__setitem__('runner_state_version', RUNNER_STATE_VERSION + 1)))
    # a branch whose origin snapshot npz is truncated
    npz_bad = _clone_record(cat, bid, 'npz', lambda d: open(
        os.path.join(d, 'state_origin.npz'), 'r+b').truncate(50))
    # a corrupted WAV: only that track fails
    wav_bad = _clone_record(cat, pid, 'wav', lambda d: open(
        os.path.join(d, 'B.wav'), 'r+b').truncate(1000))
    # an unregistered engine
    eng_bad = _clone_record(cat, pid, 'eng', lambda d: _edit_meta(d, lambda m: (
        m['scene']['variants']['A'].__setitem__('engine_id', 'laplacian_v0'),
        m['engines'].__setitem__('A', 'laplacian_v0'))))
    # a broken record.json
    json_bad = _clone_record(cat, pid, 'json', lambda d: open(
        os.path.join(d, 'record.json'), 'w').write('{not json'))
    before = _catalog_digest(cat)
    replay_dirs = {rid: os.path.exists(os.path.join(cat.root, rid, 'replay')) for rid in cat.ids()}
    run = Verifier(cat).run()
    assert run.status == RUN_COMPLETE
    assert set(run.record_ids) == set(cat.ids())
    r = run.result
    assert r(pid)['status'] == TRACK_EXACT and r(bid)['status'] == TRACK_EXACT
    assert r(old)['status'] == TRACK_EXACT
    assert r(old)['origin']['status'] == 'local' and 'not recorded' in r(old)['origin']['status_reason']
    assert r(old)['origin']['commit'] is None and r(old)['inputs']['snapshot'] == {}
    assert r(lost)['status'] == TRACK_EXACT and r(lost)['origin']['commit'] == '0' * 40
    assert r(ver_bad)['status'] == TRACK_FAILED and 'version' in r(ver_bad)['reason']
    assert r(npz_bad)['status'] == TRACK_FAILED and 'snapshot' in r(npz_bad)['reason']
    e = r(wav_bad)
    assert e['status'] == TRACK_FAILED and e['tracks']['B']['result'] == TRACK_FAILED
    assert 'B.wav' in e['tracks']['B']['reason']
    assert e['tracks']['A']['result'] == TRACK_EXACT and e['tracks']['monitor']['result'] == TRACK_EXACT
    assert r(eng_bad)['status'] == TRACK_FAILED and 'laplacian_v0' in r(eng_bad)['reason']
    assert r(json_bad)['status'] == TRACK_FAILED and 'record.json' in r(json_bad)['reason']
    for rid in (ver_bad, npz_bad, eng_bad, json_bad):
        assert all(r(rid)['tracks'][o]['result'] == TRACK_FAILED for o in OUTPUTS)
        assert all(r(rid)['tracks'][o]['reason'] for o in OUTPUTS)
    assert run.counts() == {TRACK_DIFFERS: 0, TRACK_EXACT: 4, TRACK_FAILED: 5, TRACK_UNCHECKED: 0}
    # the catalog's sources are untouched; the batch never writes <record>/replay/
    assert _catalog_digest(cat) == before
    assert {rid: os.path.exists(os.path.join(cat.root, rid, 'replay'))
            for rid in cat.ids()} == replay_dirs
    # the WAVs of a failed record still play (readable)
    assert cat.load(eng_bad).pcm('monitor').shape[0] > 0
    # an empty catalog: a complete report saying so
    empty = _fresh_catalog('empty')
    run0 = Verifier(empty).run()
    assert run0.status == RUN_COMPLETE and 'empty' in run0.doc['reason'] and run0.record_ids == []


def test_target_provenance_runtime_change_cancel_partial_and_persistence():
    # -- the worker's own code is the target (an isolated checkout != this one)
    root = _make_repo('target')
    head = prov.git(root, 'rev-parse', 'HEAD').strip()
    cat = _fresh_catalog('target', repo_root=root)
    ids = [_make_record(cat, f"rec {i}", seconds=6.0,
                        commands=[('select', _sec(1.0 + i), {'side': 'B'})])[0]
           for i in range(3)]
    msgs, proc = _worker(root, cat.root)
    assert proc.returncode == 0, proc.stderr.read()
    assert msgs[0]['provenance']['commit'] == head and msgs[0]['provenance']['match'] == 'clean'
    assert os.path.normcase(msgs[0]['provenance']['root']) == os.path.normcase(root)
    ver = Verifier(cat)
    run = ver.load_run(msgs[-1]['run_id'])
    assert run.status == RUN_COMPLETE and run.target['commit'] == head
    assert run.target['match'] == 'clean' and run.target_label == f"Pinned {head[:7]}"
    assert run.target['commit'] != prov.repo_identity(ROOT)[1]      # not this bench's HEAD
    assert run.target['environment']['python'] and run.target['environment']['packages']
    assert run.counts()[TRACK_EXACT] == 3
    # -- the runtime set changes while record 2 is computed: the run stops,
    #    record 1 stays, record 2 (unfinished) and 3 are unchecked
    cfg = os.path.join(root, 'casynth_config.py')
    orig = _read(cfg)

    def on_line(msg, p):
        if msg.get('record') and msg.get('index') == 1:
            _write(cfg, orig + "\n# changed during the check\n")
    msgs, proc = _worker(root, cat.root, on_line=on_line)
    run2 = ver.load_run(msgs[-1]['run_id'])
    assert run2.status == RUN_STOPPED and 'runtime set changed' in run2.doc['reason']
    assert run2.result(ids[0])['status'] == TRACK_EXACT if run2.record_ids[0] == ids[0] else True
    first, second, third = run2.record_ids
    assert run2.result(first)['status'] == TRACK_EXACT
    assert run2.result(second)['status'] == TRACK_UNCHECKED
    assert run2.result(third)['status'] == TRACK_UNCHECKED
    assert 'runtime set changed' in run2.result(second)['reason']
    assert run2.target['commit'] == head                          # one version per report
    _write(cfg, orig)
    # the same, deterministically, in-process with an injected fingerprint
    calls = {'n': 0}

    def digest():
        calls['n'] += 1
        return 'a' if calls['n'] <= 3 else 'b'      # changes after record 1 finished
    run3 = ver.run(runtime_digest=digest)
    assert run3.status == RUN_STOPPED
    assert [run3.result(r)['status'] for r in run3.record_ids] == [TRACK_EXACT, TRACK_UNCHECKED,
                                                                   TRACK_UNCHECKED]
    # -- cancel INSIDE a long record (worker): terminated, report closed as
    #    cancelled, finished results kept, the rest explicitly unchecked
    seen = {'n': 0}

    def cancel_on_second(i, n, rid):
        seen['n'] += 1
        if seen['n'] == 2:
            flag.set()
    flag = threading.Event()
    run4 = ver.run_in_subprocess(cancel=flag.is_set, on_record=cancel_on_second)
    assert run4.status == RUN_CANCELLED
    sts = [run4.result(r)['status'] for r in run4.record_ids]
    assert sts[0] == TRACK_EXACT and sts[1:] == [TRACK_UNCHECKED, TRACK_UNCHECKED], sts
    assert all(run4.result(r)['reason'] == 'cancelled' for r in run4.record_ids[1:])
    # cancel in-process, mid-record: the unfinished record is not attributed
    blocks = {'n': 0}

    def cancel_mid():
        blocks['n'] += 1
        return blocks['n'] > 40
    run5 = ver.run(cancel=cancel_mid)
    assert run5.status == RUN_CANCELLED and run5.counts()[TRACK_UNCHECKED] == 3
    assert all(run5.result(r)['reason'] == 'cancelled' and run5.result(r)['inputs'] is None
               for r in run5.record_ids)
    # -- partial failure: a corrupted B.wav is 'failed', A / monitor keep their
    #    results; the run is complete but not a full success
    bad = _clone_record(cat, ids[0], 'bad', lambda d: open(
        os.path.join(d, 'B.wav'), 'r+b').truncate(500))
    diff = _clone_record(cat, ids[1], 'diff', lambda d: _rewrite_wav(
        os.path.join(d, 'monitor.wav'), lambda p: (p // 2).astype(np.int16)))
    run6 = ver.run()
    assert run6.status == RUN_COMPLETE
    e = run6.result(bad)
    assert e['status'] == TRACK_FAILED and e['tracks']['B']['result'] == TRACK_FAILED
    assert e['tracks']['A']['result'] == TRACK_EXACT and e['tracks']['monitor']['result'] == TRACK_EXACT
    assert run6.counts()[TRACK_FAILED] == 1 and run6.counts()[TRACK_DIFFERS] == 1
    # -- the single check never overwrites the report's pair
    t = run6.result(diff)['tracks']['monitor']
    pair_path = os.path.join(run6.dir, diff, t['new_file'])
    sha_before = hashlib.sha256(open(pair_path, 'rb').read()).hexdigest()
    saved, new = ver.pair(run6.id, diff, 'monitor')
    assert pcm_sha(new) == t['new_sha256'] and pcm_sha(saved) == t['ref_sha256']
    assert cat.replay(diff).status == 'mismatch'
    assert hashlib.sha256(open(pair_path, 'rb').read()).hexdigest() == sha_before
    assert ver.pair(run6.id, diff, 'monitor')[1].shape == new.shape
    # -- listening mark: per pair and track, survives a "restart", never copied
    ver.set_mark(run6.id, diff, 'monitor', MARK_DIFFERENT)
    with _raises(VerifyError):
        ver.set_mark(run6.id, diff, 'A', MARK_SAME)               # A matched: no pair
    with _raises(VerifyError):
        ver.set_mark(run6.id, diff, 'monitor', 'inaudible')
    again = Verifier(Catalog(cat.root, repo_root=root))            # a new process would do this
    assert again.load_run(run6.id).mark(diff, 'monitor') == MARK_DIFFERENT
    assert again.latest().id == run6.id
    run7 = ver.run()
    assert run7.id != run6.id and run7.mark(diff, 'monitor') is None
    assert ver.load_run(run6.id).mark(diff, 'monitor') == MARK_DIFFERENT
    assert ver.load_run(run6.id).doc == run6.doc                    # the old report is intact
    ver.set_mark(run6.id, diff, 'monitor', None)
    assert ver.load_run(run6.id).mark(diff, 'monitor') is None
    assert len(ver.run_ids()) == 7 and ver.run_ids()[0] == run7.id
    # -- a replaced / lost file invalidates the pair by fingerprint
    _rewrite_wav(cat.load(diff).wav_path('monitor'), lambda p: (p // 4).astype(np.int16))
    with _raises(VerifyError):
        ver.pair(run7.id, diff, 'monitor')
    t7 = run7.result(diff)['tracks']['monitor']
    os.remove(os.path.join(run7.dir, diff, t7['new_file']))
    with _raises(VerifyError):
        ver.pair(run7.id, diff, 'monitor')
    with _raises(VerifyError):
        ver.pair(run7.id, ids[0], 'A')                              # exact: no pair stored
    # -- a report left 'running' by a dead process reads as interrupted and
    #    can be closed explicitly; a closed one is not reopened
    doc = json.load(open(os.path.join(run7.dir, 'run.json'), encoding='utf-8'))
    doc['status'] = 'running'
    doc['results'].pop(ids[2], None)
    json.dump(doc, open(os.path.join(run7.dir, 'run.json'), 'w', encoding='utf-8'))
    stale = ver.load_run(run7.id)
    assert stale.status == 'interrupted' and stale.result(ids[2])['status'] == TRACK_UNCHECKED
    assert finalize_run(run7.dir, 'interrupted', 'bench closed') is True
    assert finalize_run(run7.dir, 'complete', '') is False
    assert ver.load_run(run7.id).result(ids[2])['reason'] == 'bench closed'


def test_pair_player_shared_cursor_switch_gain_no_live():
    class _Fake:
        def stop(self): pass
        def close(self): pass
    n_a, n_b = 3500, 2500
    a = (np.arange(n_a) % 1000 + 100)[:, None].astype(np.int16).repeat(2, axis=1)
    b = (-(np.arange(n_b) % 700) - 200)[:, None].astype(np.int16).repeat(2, axis=1)
    eng = LiveEngine(DemoRunner(load_scene(DEMO_AB)), output_factory=lambda cb: _Fake())
    eng.start()
    try:
        F = 512

        def pull():
            out = np.full((F, 2), 7, np.int16)
            eng._audio_cb(out, F, None, None)
            return out
        eng.post('start')
        t0 = time.time()
        while eng.snapshot()['gen'] < 1 and time.time() - t0 < 30:
            pull()                                   # the device pulls -> the live synth runs
            time.sleep(0.002)
        assert eng.snapshot()['gen'] >= 1 and eng.snapshot()['running']
        eng.play_pair(a, b, 'saved')
        assert eng.playing and eng.pair_version == 'saved' and eng.play_pos == (0, n_a)
        assert np.array_equal(pull(), a[0:F]) and np.array_equal(pull(), a[F:2 * F])
        assert eng.play_pos == (2 * F, n_a)
        # switch at sample 1024: the cursor continues, a short player-only
        # crossfade, then exactly the recomputed version -- unity gain
        assert eng.switch_version('recomputed') == 2 * F
        out = pull()
        x = PAIR_XFADE_SAMPLES
        assert np.array_equal(out[x:], b[2 * F + x:3 * F])
        assert not np.array_equal(out[:x], a[2 * F:2 * F + x]) and not np.array_equal(out[:x], b[2 * F:2 * F + x])
        assert np.all(np.minimum(a[2 * F:2 * F + x], b[2 * F:2 * F + x]) - 1 <= out[:x])
        assert np.all(out[:x] <= np.maximum(a[2 * F:2 * F + x], b[2 * F:2 * F + x]) + 1)
        assert np.array_equal(pull(), b[3 * F:4 * F])
        # the shorter version ends: silence, the cursor goes on to the longer one
        while eng.play_pos[0] < n_b - F:
            pull()
        pos = eng.play_pos[0]
        out = pull()
        assert np.array_equal(out[:n_b - pos], b[pos:n_b]) and not out[n_b - pos:].any()
        assert eng.playing and eng.play_pos == (pos + F, n_a)
        # back to saved beyond b's end: fades from silence into a, same cursor
        pos = eng.play_pos[0]
        assert eng.switch_version('saved') == pos
        out = pull()
        end = min(n_a, pos + F)
        assert np.array_equal(out[x:end - pos], a[pos + x:end]) and not out[end - pos:].any()
        while eng.playing:
            out = pull()
        assert eng.play_pos == (0, 0) and not eng.playing
        # both versions identical -> bit-exact output (no gain, no live mix)
        eng.play_pair(a, a, 'recomputed', pos=100)
        got = [pull() for _ in range(4)]
        assert np.array_equal(np.concatenate(got), a[100:100 + 4 * F])
        eng.switch_version('saved')
        assert np.array_equal(pull(), a[100 + 4 * F:100 + 5 * F])
        eng.stop_play()
        assert not eng.playing
        # while muted / not playing the device gets zeros; the live synth was
        # never mixed into the pair output (it is running and loud)
        eng.muted = True
        assert not pull().any()
        eng.muted = False
        assert _wait(lambda: pull().any(), 10)
        with _raises(ValueError):
            eng.play_pair(a, b[:, :1], 'saved')
        with _raises(ValueError):
            eng.play_pair(a, b, 'other')
    finally:
        eng.stop()


def test_ui_headless_check_catalog_listen_pair_mark_reopen():
    import pygame
    import demo_bench as db
    cat = _fresh_catalog('ui')
    rid, truth = _make_record(cat, "S7 exact", seconds=2.0,
                              commands=[('select', _sec(0.6), {'side': 'B'}),
                                        ('set_param', _sec(1.2), {'side': 'B', 'name': 'harm',
                                                                  'value': 0.5})])
    diff = _clone_record(cat, rid, 'diff', lambda d: (
        [_rewrite_wav(os.path.join(d, f"{o}.wav"), lambda p: (p // 2).astype(np.int16))
         for o in OUTPUTS],
        _edit_meta(d, lambda m: m.__setitem__('title', "S7 differs"))))
    gone = _clone_record(cat, rid, 'gone', lambda d: _edit_meta(d, lambda m: (
        m['scene']['variants']['B'].__setitem__('engine_id', 'laplacian_v0'),
        m['engines'].__setitem__('B', 'laplacian_v0'),
        m.__setitem__('title', "S7 unavailable"))))
    scene = load_scene(DEMO_AB)

    class _Fake:
        def stop(self): pass
        def close(self): pass

    def make_app(catalog):
        eng = LiveEngine(DemoRunner(scene), output_factory=lambda cb: _Fake(),
                         record_root=catalog.tmp_root)
        app = db.BenchApp(scene, eng, catalog=catalog)
        eng.start()
        return app
    pygame.init()
    app = make_app(cat)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    F = 512

    def pull():
        out = np.full((F, 2), 7, np.int16)
        app.engine._audio_cb(out, F, None, None)
        return out

    def click(rect):
        return app.press((rect[0] + 3, rect[1] + 3), 1)
    try:
        # step 1: Check catalog -> short report: a match, a difference, a
        # concrete unavailability reason
        click(app.lab_buttons['catalog'])
        assert app.mode == 'catalog'
        r = app._catalog_rects()
        assert click(r['report']) == 'report' and app.mode == 'catalog'
        assert app.status == "No catalog check yet"
        assert click(r['check']) == 'check' and app.checking
        assert click(r['play_A']) == 'busy'                         # nothing competes
        assert click(r['replay']) == 'busy' and app.cat['thread'] is None
        t0 = time.time()
        while app.checking and time.time() - t0 < 300:
            app.draw(screen, font, small)
            time.sleep(0.03)
        app.tick()
        assert app.mode == 'report', app.status
        run = app.verify['run']
        assert run.status == RUN_COMPLETE
        assert run.counts() == {TRACK_DIFFERS: 1, TRACK_EXACT: 1, TRACK_FAILED: 1,
                                TRACK_UNCHECKED: 0}, run.counts()
        assert app.verify['group'] == TRACK_DIFFERS and app.report_record_id() == diff
        assert app.verify['track'] == 'monitor' and app.verify['version'] == 'saved'
        assert 'laplacian_v0' in run.result(gone)['reason']
        app.draw(screen, font, small)
        pygame.image.save(screen, os.path.join(ART, "_demo_bench_report_s7.png"))
        rr = app._report_rects()
        assert click(rr['group:failed']) == 'group:failed' and app.report_record_id() == gone
        assert click(rr['group:exact']) == 'group:exact' and app.report_record_id() == rid
        assert click(rr['group:differs']) == 'group:differs' and app.report_record_id() == diff
        # step 2: listen 'as heard', switch saved / recomputed on one cursor
        saved, new = app.verifier.pair(run.id, diff, 'monitor')
        assert np.array_equal(new, truth['monitor']) and np.array_equal(saved, truth['monitor'] // 2)
        assert click(rr['play']) == 'play' and app.engine.playing, app.status
        assert np.array_equal(pull(), saved[:F]) and np.array_equal(pull(), saved[F:2 * F])
        assert app.engine.play_pos[0] == 2 * F
        assert click(rr['version:recomputed']) == 'version:recomputed'
        assert app.engine.pair_version == 'recomputed' and app.engine.play_pos[0] == 2 * F
        out = pull()
        assert np.array_equal(out[PAIR_XFADE_SAMPLES:], new[2 * F + PAIR_XFADE_SAMPLES:3 * F])
        assert np.array_equal(pull(), new[3 * F:4 * F])
        assert app.key('space') == 'version:saved' and app.engine.pair_version == 'saved'
        pull()
        assert np.array_equal(pull(), saved[5 * F:6 * F])
        # the track selector keeps the position; the version toggle stays
        assert click(rr['track:A']) == 'track:A' and app.engine.playing
        assert app.engine.play_pos[0] == 6 * F and app.verify['track'] == 'A'
        a_saved, a_new = app.verifier.pair(run.id, diff, 'A')
        assert np.array_equal(pull(), a_saved[6 * F:7 * F])
        assert click(rr['stop']) == 'stop' and not app.engine.playing
        app.draw(screen, font, small)
        # the listening mark: optional, per pair, revertible; never automatic
        assert run.mark(diff, 'A') is None
        assert click(rr['mark:same']) == 'mark:same' and app.status == "Mark saved"
        assert app.verify['run'].mark(diff, 'A') == MARK_SAME
        assert click(rr['track:monitor']) == 'track:monitor'
        assert click(rr['mark:different']) == 'mark:different'
        assert app.verify['run'].mark(diff, 'monitor') == MARK_DIFFERENT
        assert app.verify['run'].result(diff)['tracks']['monitor']['result'] == TRACK_DIFFERS
        assert click(rr['mark:none']) == 'mark:none' and app.verify['run'].mark(diff, 'monitor') is None
        assert click(rr['mark:different']) == 'mark:different'
        # an exact record: the original plays for both versions, no mark offered
        click(rr['group:exact'])
        assert click(rr['play']) == 'play' and np.array_equal(pull(), truth['monitor'][:F])
        click(rr['version:recomputed'])
        pull()
        assert np.array_equal(pull(), truth['monitor'][2 * F:3 * F])
        click(rr['stop'])
        assert click(rr['mark:same']) == 'mark:same' and app.status.startswith("Cannot mark")
        # an unavailable record: the saved WAV plays, no recomputed version
        click(rr['group:failed'])
        click(rr['version:recomputed'])
        assert click(rr['play']) == 'play' and not app.engine.playing
        assert app.status.startswith("No recomputed version")
        click(rr['version:saved'])
        assert click(rr['play']) == 'play' and app.engine.playing
        click(rr['stop'])
        assert app.key('escape') == 'report:back' and app.mode == 'catalog'
        # step 3: reopen the bench: report, pair and mark are there without a
        # recompute; the original record plays as before
        run_dir_digest = _dir_digest(run.dir)
        app.engine.stop()
        app = make_app(Catalog(cat.root, repo_root=None))
        click(app.lab_buttons['catalog'])
        r = app._catalog_rects()
        n_runs = len(app.verifier.run_ids())
        assert click(r['report']) == 'report' and app.mode == 'report'
        assert app.verify['run'].id == run.id and len(app.verifier.run_ids()) == n_runs
        assert app.verify['run'].mark(diff, 'monitor') == MARK_DIFFERENT
        assert app.verify['run'].mark(diff, 'A') == MARK_SAME
        assert app.report_record_id() == diff and app.verify['track'] == 'monitor'
        assert click(rr['play']) == 'play' and np.array_equal(pull(), saved[:F])
        click(rr['stop'])
        assert _dir_digest(run.dir) == run_dir_digest
        app.draw(screen, font, small)
        assert app.key('escape') == 'report:back'
        row = [i for i, (rec, _e) in enumerate(app.cat['entries']) if rec and rec.id == diff][0]
        app.press((app._list_rect(row)[0] + 5, app._list_rect(row)[1] + 5), 1)
        assert click(r['play_A']) == 'play:A' and app.engine.playing
        assert np.array_equal(pull(), a_saved[:F])
        click(r['stop'])
        # a second check makes a NEW report; the first keeps its marks
        assert click(r['check']) == 'check'
        t0 = time.time()
        while app.checking and time.time() - t0 < 300:
            app.tick()
            time.sleep(0.03)
        app.tick()
        assert app.mode == 'report' and app.verify['run'].id != run.id
        assert app.verify['run'].mark(diff, 'monitor') is None
        assert click(rr['older']) == 'report:older' and app.verify['run'].id == run.id
        assert app.verify['run'].mark(diff, 'monitor') == MARK_DIFFERENT
        assert click(rr['newer']) == 'report:newer' and app.verify['run'].id != run.id
        assert app.key('escape') == 'report:back' and app.key('escape') == 'back'
        assert app.mode == 'live'
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
        except Exception as e:           # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    if prov.git_exe() is None:
        print("git executable not found: S7 tests cannot run")
        sys.exit(1)
    os.makedirs(S7_ROOT, exist_ok=True)
    sys.exit(1 if _run() else 0)
