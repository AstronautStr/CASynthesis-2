#!/usr/bin/env python3
"""S6 acceptance tests: execution provenance, pinned / local records, running
a record's original code version.

    python tests/test_demo_lab_s6.py      # stdlib runner

Everything Git-related happens in ISOLATED test repositories under
artifacts/_s6/ (copies of the runtime set); the project's own checkout,
index and history are never touched.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR                                        # noqa: E402
from casynth_lab import load_scene, DemoRunner, BLOCK                # noqa: E402
from casynth_lab.audio_out import LiveEngine                         # noqa: E402
from casynth_lab.catalog import Catalog, CatalogError, RECORD_FORMAT  # noqa: E402
from casynth_lab import provenance as prov                           # noqa: E402
from casynth_lab import versions                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
ART = os.path.join(ROOT, "artifacts")
S6_ROOT = os.path.join(ART, "_s6")
OUTPUTS = ('A', 'B', 'monitor')
GIT = prov.git_exe()


def _sec(s):
    return int(round(s * SR))


def _git(root, *args):
    return prov.git(root, *args)


def _write(path, text, crlf=False):
    data = text.replace('\r\n', '\n')
    if crlf:
        data = data.replace('\n', '\r\n')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data.encode('utf-8'))


def _read(path):
    with open(path, 'rb') as f:
        return f.read().decode('utf-8')


def _rmtree(path):
    """rmtree that copes with Git's read-only object files on Windows."""
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


def _make_repo(name, autocrlf='true'):
    """A fresh git repository holding a copy of the project's runtime set."""
    root = os.path.join(S6_ROOT, name)
    _rmtree(root)
    os.makedirs(root)
    for rel in prov.runtime_paths(ROOT):
        src, dst = os.path.join(ROOT, rel), os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    _write(os.path.join(root, 'README.md'), "# test repo\n")
    _write(os.path.join(root, '.gitignore'), "lab_catalog/\n__pycache__/\n")
    _git(root, 'init', '-q', '-b', 'master')
    _git(root, 'config', 'user.email', 'test@example.com')
    _git(root, 'config', 'user.name', 'test')
    _git(root, 'config', 'core.autocrlf', autocrlf)
    _git(root, 'config', 'gc.auto', '0')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '-m', 'base')
    return root


def _commit_all(root, msg):
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '-m', msg)
    return _git(root, 'rev-parse', 'HEAD').strip()


def _head(root):
    return _git(root, 'rev-parse', 'HEAD').strip()


def _cap(root):
    return prov.capture(root, check_loaded=False)


def _catalog(root, name='cat'):
    cat_root = os.path.join(root, 'lab_catalog', name)
    os.makedirs(os.path.join(cat_root, '.tmp'), exist_ok=True)
    return Catalog(cat_root, repo_root=root)


def _record(cat, doc, seconds=0.6, title="rec"):
    """A record made by a runner whose provenance is `doc`."""
    scene = load_scene(DEMO_AB)
    runner = DemoRunner(scene, provenance=doc)
    got = {o: [] for o in OUTPUTS}
    eng = LiveEngine(runner, sink=lambda m, b: [got[o].append(b.get(o)) for o in OUTPUTS],
                     record_root=cat.tmp_root)
    eng.start()
    eng.post('start')
    n = int(round(seconds * SR / BLOCK))
    t0 = time.time()
    while len(got['A']) < n and time.time() - t0 < 60:
        time.sleep(0.01)
    kind, cut = eng.cut_now()
    eng.stop()
    assert kind == 'ok'
    return cat.save(cut, title)


def _tree_digest(path, skip=('record.json',)):
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isfile(p) and name not in skip:
            h.update(name.encode())
            h.update(open(p, 'rb').read())
    return h.hexdigest()


def _repo_state(root):
    return (_head(root), _git(root, 'status', '--porcelain'), _git(root, 'diff', '--cached'),
            _git(root, 'branch', '--show-current').strip())


class _raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), f"expected {self.exc.__name__}, got {et}"
        return True


# =============================================================================
def test_classification_clean_dirty_docs_crlf():
    root = _make_repo('classify')
    head = _head(root)
    doc = _cap(root)
    assert doc['match'] == 'clean' and doc['commit'] == head and doc['repo_id']
    assert prov.status_of(doc) == ('pinned', '')
    assert len(doc['manifest']) >= 12 and 'casynth_lab/runner.py' in doc['manifest']
    assert 'casynth_engine.py' in doc['manifest'] and 'requirements.txt' in doc['manifest']
    assert not any(k.startswith(('tests/', 'memory/', 'docs/')) for k in doc['manifest'])
    assert doc['digest'] == prov.digest_at_commit(root, head)
    # unstaged change of the shared DSP -> local
    eng_path = os.path.join(root, 'casynth_engine.py')
    orig = _read(eng_path)
    _write(eng_path, orig + "\n# tweak\n")
    d = _cap(root)
    assert d['match'] == 'dirty' and d['dirty'] == ['casynth_engine.py']
    assert prov.status_of(d)[0] == 'local' and 'casynth_engine.py' in prov.status_of(d)[1]
    # staged (not committed) -> still local
    _git(root, 'add', 'casynth_engine.py')
    d = _cap(root)
    assert d['match'] == 'dirty' and d['dirty'] == ['casynth_engine.py']
    _git(root, 'reset', '-q', 'casynth_engine.py')
    _write(eng_path, orig)
    # a NEW runtime file (engine module) -> local, named
    new_mod = os.path.join(root, 'casynth_lab', 'my_engine.py')
    _write(new_mod, "x = 1\n")
    d = _cap(root)
    assert d['match'] == 'dirty' and d['dirty'] == ['casynth_lab/my_engine.py']
    os.remove(new_mod)
    # a document change does not make the sound local
    _write(os.path.join(root, 'README.md'), "# changed docs\n")
    _write(os.path.join(root, 'notes.md'), "notes\n")
    d = _cap(root)
    assert d['match'] == 'clean' and d['commit'] == head
    # CRLF vs LF checkout of the same content: not dirty either way
    for crlf in (True, False):
        for rel in ('casynth_core.py', 'casynth_lab/runner.py', 'demos/laplace_ab.json'):
            p = os.path.join(root, rel)
            _write(p, _read(p), crlf=crlf)
        d = _cap(root)
        assert d['match'] == 'clean', (crlf, d['dirty'])
    # an LF repository (autocrlf false) classifies the same way
    root2 = _make_repo('classify_lf', autocrlf='false')
    d2 = _cap(root2)
    assert d2['match'] == 'clean' and d2['digest'] == doc['digest']    # same code, same digest
    _write(os.path.join(root2, 'casynth_core.py'), _read(os.path.join(root2, 'casynth_core.py')),
           crlf=True)
    assert _cap(root2)['match'] == 'clean'
    # environment is described separately and exactly
    env = doc['environment']
    assert env['python'] and env['packages']['numpy'] and env['machine']
    assert prov.environment_compatible(env) == (True, "")
    assert not prov.environment_compatible(dict(env, python='2.7.0'))[0]
    assert not prov.environment_compatible(dict(env, packages=dict(env['packages'], numpy='0.1')))[0]


def test_process_origin_is_the_code_that_ran_not_head_at_save():
    root = _make_repo('origin')
    x = _head(root)
    cat = _catalog(root)
    doc_x = _cap(root)                                  # captured at runner init (X)
    # ... then the files and HEAD move on to Y while the process keeps running
    cfg = os.path.join(root, 'casynth_config.py')
    _write(cfg, _read(cfg) + "\n# Y\n")
    y = _commit_all(root, 'Y')
    assert y != x and _cap(root)['commit'] == y
    rid = _record(cat, doc_x)
    rec = cat.load(rid)
    assert rec.meta['format'] == RECORD_FORMAT
    assert rec.status == 'pinned' and rec.commit == x and rec.provenance['digest'] == doc_x['digest']
    assert rec.meta['pin'] == {'commit': x, 'ref': prov.pin_ref(x)}
    assert prov.is_held(root, x) and _git(root, 'rev-parse', prov.pin_ref(x)).strip() == x
    assert rec.version_label() == f"Pinned {x[:7]}"
    # a runner created NOW records Y (its own origin), not X
    rid2 = _record(cat, _cap(root))
    assert cat.load(rid2).commit == y
    # a subprocess reports ITS OWN code: the check worker started in the Y
    # checkout says Y (and finds the X-record's PCM reproducible -- the
    # tweak was a comment)
    p = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(root, 'demo_bench.py'),
                        '--catalog', cat.root, '--record', rid, '--action', 'check'],
                       cwd=root, capture_output=True, text=True, encoding='utf-8',
                       env=versions.child_env(), timeout=300)
    lines = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith('{')]
    assert lines and 'provenance' in lines[0], p.stdout + p.stderr
    assert lines[0]['provenance']['commit'] == y and lines[0]['provenance']['match'] == 'clean'
    assert os.path.normcase(lines[0]['provenance']['root']) == os.path.normcase(root)
    assert lines[-1]['status'] == 'match' and lines[-1]['commit'] == y, lines[-1]
    # a dirty checkout -> local with the reason; a pin ref is never created for it
    _write(cfg, _read(cfg) + "\n# dirty\n")
    rid3 = _record(cat, _cap(root))
    rec3 = cat.load(rid3)
    assert rec3.status == 'local' and 'casynth_config.py' in rec3.status_reason
    assert rec3.meta['pin'] is None and rec3.version_label().startswith("Local:")


def test_pin_local_record_after_matching_commit():
    root = _make_repo('pin')
    cat = _catalog(root)
    eng_path = os.path.join(root, 'casynth_engine.py')
    _write(eng_path, _read(eng_path) + "\n# local dsp change\n")
    doc = _cap(root)
    assert doc['match'] == 'dirty'
    rid = _record(cat, doc, title="local find")
    rec = cat.load(rid)
    assert rec.status == 'local'
    before_tree = _tree_digest(rec.dir)
    before_meta = {k: v for k, v in rec.meta.items()
                   if k not in ('provenance', 'status', 'status_reason', 'pin')}
    # pin BEFORE the commit exists: refused, nothing changes
    with _raises(CatalogError):
        cat.pin(rid)
    assert cat.load(rid).status == 'local'
    # commit exactly that code -> pin attaches the record to it
    c = _commit_all(root, 'the dsp change')
    got = cat.pin(rid)
    rec = cat.load(rid)
    assert got == c and rec.status == 'pinned' and rec.commit == c
    assert rec.provenance['match'] == 'clean' and rec.meta['pin']['commit'] == c
    assert prov.is_held(root, c)
    assert _tree_digest(rec.dir) == before_tree                  # WAVs / snapshots untouched
    assert {k: v for k, v in rec.meta.items()
            if k not in ('provenance', 'status', 'status_reason', 'pin')} == before_meta
    assert cat.pin(rid) == c                                     # idempotent
    # the matching commit may be older than HEAD: found in history
    _write(eng_path, _read(eng_path) + "\n# later\n")
    _commit_all(root, 'later')
    rid_old = _record(cat, dict(doc))                            # fingerprint of commit c
    assert cat.load(rid_old).status == 'local'                   # (doc said dirty)
    assert cat.pin(rid_old) == c
    # different code that happens to produce the same PCM does not pin
    _write(eng_path, _read(eng_path) + "\n# other code\n")
    doc2 = _cap(root)
    rid2 = _record(cat, doc2, title="other code")
    rec2 = cat.load(rid2)
    assert np.array_equal(rec2.pcm('A')[:BLOCK * 20], cat.load(rid).pcm('A')[:BLOCK * 20])
    with _raises(CatalogError):
        cat.pin(rid2)
    with _raises(CatalogError):
        cat.pin(rid2, commit=c)                                  # explicit but different code
    assert cat.load(rid2).status == 'local'
    # no fingerprint at all (pre-S6 record) -> refused
    old = dict(rec2.meta)
    old.pop('provenance')
    old['format'] = 2
    cat._rewrite_meta(rec2, old)
    with _raises(CatalogError):
        cat.pin(rid2)
    assert cat.load(rid2).status == 'local' and 'not recorded' in cat.load(rid2).status_reason
    # Git failure while holding -> no false pin, metadata unchanged
    _write(eng_path, _read(eng_path) + "\n# z\n")
    z = _commit_all(root, 'z')
    rid3 = _record(cat, _cap(root))
    assert cat.load(rid3).status == 'pinned'                     # clean save pins itself
    _write(eng_path, _read(eng_path) + "\n# w\n")
    doc_w = _cap(root)
    rid4 = _record(cat, doc_w)
    w = _commit_all(root, 'w')
    real = prov.hold_commit
    prov.hold_commit = lambda *a, **k: (_ for _ in ()).throw(prov.ProvenanceError("refs locked"))
    try:
        with _raises(CatalogError):
            cat.pin(rid4)
    finally:
        prov.hold_commit = real
    assert cat.load(rid4).status == 'local' and cat.load(rid4).meta.get('pin') is None
    assert cat.pin(rid4) == w
    # a clean save whose ref cannot be created is stored as LOCAL with the reason
    prov.hold_commit = lambda *a, **k: (_ for _ in ()).throw(prov.ProvenanceError("refs locked"))
    try:
        rid5 = _record(cat, _cap(root))
    finally:
        prov.hold_commit = real
    r5 = cat.load(rid5)
    assert r5.status == 'local' and 'refs locked' in r5.status_reason and r5.meta['pin'] is None
    assert z != w


def _tweak_gain(root, factor):
    cfg = os.path.join(root, 'casynth_config.py')
    src = _read(cfg)
    assert 'MASTER_GAIN' in src
    _write(cfg, src + f"\nMASTER_GAIN = MASTER_GAIN * {factor}   # S6 test tweak\n")


def _run_child(cat, rec, action, extra=()):
    child = cat.run_version(rec, action, extra)
    assert child.wait(300), "child bench did not finish"
    return child


def test_source_version_runs_the_original_code_and_survives_gc():
    root = _make_repo('versions')
    y = _head(root)                                     # Y = the current code
    _git(root, 'checkout', '-q', '-b', 'tmp')
    _tweak_gain(root, 0.5)                              # X = audibly different DSP
    x = _commit_all(root, 'X: half gain')
    _git(root, 'checkout', '-q', 'master')
    assert _head(root) == y and x != y
    cat = _catalog(root)
    # a record pinned to X (its snapshot was rendered by the code of this
    # process; what matters here is WHICH code continues it)
    doc_x = dict(_cap(root), commit=x, match='clean', dirty=[], reason='')
    doc_x['digest'] = prov.digest_at_commit(root, x)
    rid = _record(cat, doc_x, title="find at X")
    rec = cat.load(rid)
    assert rec.status == 'pinned' and rec.commit == x
    # this bench is Y -> Continue must run X in a separate bench
    cur = _cap(root)
    plan = cat.source_plan(rec, current=cur)
    assert plan['mode'] == 'worktree' and plan['commit'] == x, plan
    plan_y = cat.source_plan(cat.load(_record(cat, cur, title="find at Y")), current=cur)
    assert plan_y['mode'] == 'in-process'
    state_before = _repo_state(root)
    out = os.path.join(cat.root, 'cont_x.npz')          # gitignored (lab_catalog/)
    child = _run_child(cat, rec, 'continue', ('--headless', '--seconds', '0.5', '--out', out))
    assert child.error is None, (child.error, child.lines)
    assert child.provenance['commit'] == x and child.provenance['match'] == 'clean'
    wt = versions.worktree_path(cat.root, x)
    assert os.path.normcase(child.provenance['root']) == os.path.normcase(wt)
    assert child.result['status'] == 'ok' and child.result['commit'] == x
    # the SAME continuation with this (Y) code, in-process
    runner, state, _ = cat.continue_runner(rid)
    got = {o: [] for o in OUTPUTS}
    n = int(round(0.5 * SR / BLOCK))
    while len(got['A']) < n:
        b = runner.next_block()
        for o in OUTPUTS:
            got[o].append(b.get(o))
    # ... and the same continuation with X's DSP change applied to THIS code:
    # the child's output must equal the latter byte-exact and differ from Y
    import casynth_lab.runner as runner_mod
    runner_x, _st, _ = cat.continue_runner(rid)
    got_x = {o: [] for o in OUTPUTS}
    runner_mod.MASTER_GAIN *= 0.5
    try:
        while len(got_x['A']) < n:
            b = runner_x.next_block()
            for o in OUTPUTS:
                got_x[o].append(b.get(o))
    finally:
        runner_mod.MASTER_GAIN *= 2.0
    with np.load(out) as z:
        for o in OUTPUTS:
            px = z[o]
            py = np.concatenate(got[o][:n])
            pxx = np.concatenate(got_x[o][:n])
            assert px.shape == py.shape and np.abs(py).max() > 200
            assert not np.array_equal(px, py), f"{o}: child output equals Y's"
            assert np.array_equal(px, pxx), f"{o}: child output != X's DSP on this snapshot"
    # the branch saved by the X bench is in the SAME catalog, pinned to X, parent = rid
    bid = child.result['saved']
    b = cat.load(bid)
    assert b.parent_id == rid and b.status == 'pinned' and b.commit == x
    assert b.origin_kind == 'snapshot' and b.provenance['root'].lower() == wt.lower()
    assert not os.path.exists(os.path.join(wt, 'lab_catalog'))    # no catalog inside the cache
    # reproducibility check in Y really computes with Y: the X-branch differs
    res = cat.replay(bid)
    assert res.status == 'mismatch', res.text
    # ... while the X worker finds its own branch reproducible
    chk = _run_child(cat, b, 'check')
    assert chk.result['status'] == 'match' and chk.result['commit'] == x
    # the main checkout, branch, index and working copy are untouched
    assert _repo_state(root) == state_before
    # drop the temporary branch and the cache, collect garbage: X is held
    _git(root, 'branch', '-D', 'tmp')
    versions.remove_worktree(root, wt)
    assert not os.path.exists(wt)
    _git(root, 'reflog', 'expire', '--expire=now', '--all')
    _git(root, 'gc', '-q', '--prune=now')
    assert prov.commit_exists(root, x) and prov.is_held(root, x)
    child2 = _run_child(cat, rec, 'continue', ('--headless', '--seconds', '0.2'))
    assert child2.error is None and child2.provenance['commit'] == x
    # a corrupted cache entry is repaired from the held commit
    _write(os.path.join(wt, 'casynth_config.py'), "broken = True\n")
    assert versions.ensure_worktree(root, cat.root, x) == wt
    assert prov.digest(prov.manifest_of_tree(wt)) == prov.digest_at_commit(root, x)
    # a lost commit: continue unavailable with a concrete reason, WAV still readable
    fake = dict(rec.meta)
    fake['provenance'] = dict(rec.provenance, commit='0' * 40)
    fake['pin'] = {'commit': '0' * 40, 'ref': prov.pin_ref('0' * 40)}
    cat._rewrite_meta(rec, fake)
    plan = cat.source_plan(cat.load(rid), current=cur)
    assert plan['mode'] is None and 'not available' in plan['reason']
    assert cat.load(rid).pcm('monitor').shape[0] > 0
    with _raises(versions.VersionError):
        cat.worktree_for(cat.load(rid))
    assert _repo_state(root) == state_before


def test_catalog_compatibility_and_environment():
    root = _make_repo('compat')
    cat = _catalog(root)
    rid = _record(cat, _cap(root))
    rec = cat.load(rid)
    # an S5-era record (format 2, no provenance): local, "not recorded", not
    # attributed to HEAD, still playable / continuable by the S5 rules
    old = dict(rec.meta)
    old.pop('provenance'); old.pop('pin'); old.pop('status_reason')
    old['format'] = 2
    old['status'] = 'local'
    cat._rewrite_meta(rec, old)
    r = cat.load(rid)
    assert r.status == 'local' and r.commit is None and 'not recorded' in r.status_reason
    plan = cat.source_plan(r, current=_cap(root))
    assert plan['mode'] == 'in-process' and 'local' in plan['label']
    assert cat.continue_runner(rid)[0].running
    assert cat.replay(rid).status == 'match'
    with _raises(CatalogError):
        cat.pin(rid)
    assert cat.load(rid).commit is None
    # incompatible environment -> reason, WAV available
    rid2 = _record(cat, _cap(root))
    rec2 = cat.load(rid2)
    other = dict(rec2.meta)
    other['provenance'] = dict(rec2.provenance, commit='1' * 40, digest='f' * 64,
                               environment=dict(rec2.provenance['environment'], python='3.9.1'))
    other['pin'] = {'commit': '1' * 40, 'ref': prov.pin_ref('1' * 40)}
    cat._rewrite_meta(rec2, other)
    assert cat.load(rid2).status == 'pinned'
    _git(root, 'update-ref', prov.pin_ref('1' * 40), _head(root))     # pretend it exists
    plan = cat.source_plan(cat.load(rid2), current=_cap(root))
    assert plan['mode'] is None and 'environment' in plan['reason'] and 'Python' in plan['reason']
    assert cat.load(rid2).pcm('A').shape[0] > 0
    # listing shows both statuses; a broken provenance block never breaks the list
    broken = dict(rec2.meta)
    broken['provenance'] = "garbage"
    cat._rewrite_meta(rec2, broken)
    entries = cat.list()
    assert all(err is None for _r, err in entries)
    assert cat.load(rid2).status == 'local'
    # the project's own checkout is described, never modified, by capture()
    before = (prov.git(ROOT, 'rev-parse', 'HEAD'), prov.git(ROOT, 'status', '--porcelain'))
    doc = prov.capture(ROOT)
    assert doc['commit'] and doc['loaded_from'] and all(doc['loaded_from'].values())
    assert (prov.git(ROOT, 'rev-parse', 'HEAD'), prov.git(ROOT, 'status', '--porcelain')) == before


def test_ui_headless_status_pin_and_continue_in_version():
    import pygame
    import demo_bench as db
    root = _make_repo('ui')
    y = _head(root)
    _git(root, 'checkout', '-q', '-b', 'tmp')
    _tweak_gain(root, 0.5)
    x = _commit_all(root, 'X')
    _git(root, 'checkout', '-q', 'master')
    cat = _catalog(root)
    cur = _cap(root)
    doc_x = dict(cur, commit=x, digest=prov.digest_at_commit(root, x))
    pid_x = _record(cat, doc_x, title="find at X")
    eng_path = os.path.join(root, 'casynth_engine.py')
    eng_orig = _read(eng_path)
    _write(eng_path, eng_orig + "\n# local\n")
    pid_local = _record(cat, _cap(root), title="local find")
    _write(eng_path, eng_orig)
    pid_y = _record(cat, cur, title="find at Y")
    scene = load_scene(DEMO_AB)

    class _Fake:
        def stop(self): pass
        def close(self): pass

    real_current = prov.current
    prov.current = lambda audio=None: cur                        # this bench "is" Y
    eng = LiveEngine(DemoRunner(scene, provenance=cur), output_factory=lambda cb: _Fake(),
                     record_root=cat.tmp_root)
    pygame.init()
    app = db.BenchApp(scene, eng, catalog=cat)
    screen = pygame.Surface((app.width, app.height))
    font = pygame.font.SysFont(db.FONT_NAMES, 17)
    small = pygame.font.SysFont(db.FONT_NAMES, 14)
    eng.start()
    try:
        rect = app.lab_buttons['catalog']
        app.press((rect[0] + 3, rect[1] + 3), 1)
        rows = {rec.id: i for i, (rec, _e) in enumerate(app.cat['entries']) if rec}
        r = app._catalog_rects()

        def select(rid):
            app.press((app._list_rect(rows[rid])[0] + 5, app._list_rect(rows[rid])[1] + 5), 1)
            app.draw(screen, font, small)
            assert app.selected_record().id == rid
        # statuses
        select(pid_y)
        assert app.selected_record().version_label() == f"Pinned {y[:7]}"
        select(pid_local)
        assert app.selected_record().version_label().startswith("Local: runtime files differ")
        # Pin the local record after committing its code
        assert app.press((r['pin'][0] + 3, r['pin'][1] + 3), 1) == 'pin'
        assert app.status.startswith("Cannot pin"), app.status
        _write(eng_path, eng_orig + "\n# local\n")
        c = _commit_all(root, 'local code committed')
        _write(eng_path, eng_orig)
        _commit_all(root, 'back')
        rows = {rec.id: i for i, (rec, _e) in enumerate(app.cat['entries']) if rec}
        select(pid_local)
        assert app.press((r['pin'][0] + 3, r['pin'][1] + 3), 1) == 'pin'
        assert app.status.startswith("Pinned to " + c[:7]), app.status
        rows = {rec.id: i for i, (rec, _e) in enumerate(app.cat['entries']) if rec}
        select(pid_local)
        assert app.selected_record().status == 'pinned'
        assert app.press((r['pin'][0] + 3, r['pin'][1] + 3), 1) is None   # button gone
        pygame.image.save(screen, os.path.join(ART, "_demo_bench_catalog_s6.png"))
        # Continue of the X record from this Y bench: a separate bench of X
        # (headless here), this session kept, catalog refreshed on exit
        real_run = cat.run_version
        cat.run_version = lambda rec, action, extra=(), on_line=None: real_run(
            rec, action, ('--headless', '--seconds', '0.3'), on_line)
        select(pid_x)
        cur_engine = app.engine
        assert app.press((r['continue'][0] + 3, r['continue'][1] + 3), 1) == 'continue'
        assert app.child is not None and app.status.startswith("Running version " + x[:7]), app.status
        assert app.press((r['play_A'][0] + 3, r['play_A'][1] + 3), 1) == 'busy'
        t0 = time.time()
        while app.child is not None and time.time() - t0 < 300:
            app.tick()
            time.sleep(0.05)
        assert app.child is None and app.engine is cur_engine and app.mode == 'catalog'
        assert app.status.startswith("Version bench closed (" + x[:7]), app.status
        branches = [rec for rec, _e in app.cat['entries'] if rec and rec.parent_id == pid_x]
        assert len(branches) == 1 and branches[0].commit == x and branches[0].status == 'pinned'
        cat.run_version = real_run
        # Continue of the Y record: in-process, labelled as this version
        rows = {rec.id: i for i, (rec, _e) in enumerate(app.cat['entries']) if rec}
        select(pid_y)
        assert app.press((r['continue'][0] + 3, r['continue'][1] + 3), 1) == 'continue'
        assert app.mode == 'live' and app.engine is not cur_engine
        assert "this bench is version " + y[:7] in app.status, app.status
    finally:
        prov.current = real_current
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
    if GIT is None:
        print("git executable not found: S6 tests cannot run")
        sys.exit(1)
    os.makedirs(S6_ROOT, exist_ok=True)
    sys.exit(1 if _run() else 0)
