#!/usr/bin/env python3
"""Build the small demo catalog for the S7 manual acceptance:

    python tests/s7_demo_catalog.py            -> lab_catalog/s7_demo/
    run_demo_bench.bat --catalog lab_catalog\\s7_demo     (then: Catalog -> Check catalog)

Three records on the existing Laplace A/B demo scene, with NO change to the
working DSP:
  1. "S7 exact"       -- made by THIS code (A/B switch + a harm change on B):
                         the check reproduces it exactly.
  2. "S7 differs"     -- made by a child process running an isolated COPY of the
                         runtime set with one uncommitted change (MASTER_GAIN
                         halved in casynth_config.py; the copy is a throwaway
                         Git repo under artifacts/_s7_demo_repo, so the record is
                         "Local: runtime files differ from HEAD: casynth_config.py").
                         Recomputed with the current code it is 6 dB louder: a
                         clearly audible, honestly explained difference.
  3. "S7 unavailable" -- a copy of record 1 whose side B names an engine that no
                         longer exists ('laplacian_v0'): the check reports the
                         concrete reason; the WAVs still play.
The user's own catalog (lab_catalog/local) and the project's Git are not touched.
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR                                        # noqa: E402
from casynth_lab import load_scene, DemoRunner, BLOCK                # noqa: E402
from casynth_lab.audio_out import LiveEngine                         # noqa: E402
from casynth_lab.catalog import Catalog                              # noqa: E402
from casynth_lab import provenance as prov                           # noqa: E402
from casynth_lab import versions                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_AB = os.path.join(ROOT, "demos", "laplace_ab.json")
DEMO_ROOT = os.path.join(ROOT, "lab_catalog", "s7_demo")
REPO = os.path.join(ROOT, "artifacts", "_s7_demo_repo")
SECONDS = 8.0
COMMANDS = [('start', 0.1, {}),
            ('select', 2.0, {'side': 'B'}),
            ('set_param', 4.0, {'side': 'B', 'name': 'harm', 'value': 0.5}),
            ('select', 6.0, {'side': 'A'})]


def _sec(s):
    return int(round(s * SR))


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


def record_session(catalog, title, note):
    """A paced live session (test sink) of the demo scene -> saved record id."""
    runner = DemoRunner(load_scene(DEMO_AB))
    clock = {'n': 0, 't0': None}

    def sink(mon, blk):
        clock['n'] += BLOCK
        if clock['t0'] is None:
            clock['t0'] = time.perf_counter()
        ahead = clock['n'] / SR - (time.perf_counter() - clock['t0'])
        if ahead > 0.04:
            time.sleep(ahead - 0.04)
    eng = LiveEngine(runner, sink=sink, record_root=catalog.tmp_root)
    eng.start()
    try:
        for kind, at, args in COMMANDS:
            while eng.snapshot()['out_samples'] < _sec(at) - 6 * BLOCK:
                time.sleep(0.005)
            eng.post(kind, at=_sec(at), **args)
        while eng.snapshot()['out_samples'] < _sec(SECONDS):
            time.sleep(0.01)
        kind, cut = eng.cut_now()
        assert kind == 'ok' and cut is not None
        return catalog.save(cut, title, note)
    finally:
        eng.stop()


def make_isolated_repo():
    """A throwaway Git repo holding a copy of the runtime set, base committed,
    then MASTER_GAIN halved WITHOUT committing (-> a local, dirty version)."""
    _rmtree(REPO)
    os.makedirs(REPO)
    for rel in prov.runtime_paths(ROOT):
        src, dst = os.path.join(ROOT, rel), os.path.join(REPO, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    with open(os.path.join(REPO, '.gitignore'), 'w') as f:
        f.write("lab_catalog/\n__pycache__/\n")
    g = lambda *a: prov.git(REPO, *a)                   # noqa: E731
    g('init', '-q', '-b', 'master')
    g('config', 'user.email', 'demo@example.com')
    g('config', 'user.name', 'demo')
    g('config', 'gc.auto', '0')
    g('add', '-A')
    g('commit', '-q', '-m', 'copy of the runtime set for the S7 demo')
    cfg = os.path.join(REPO, 'casynth_config.py')
    with open(cfg, 'a', encoding='utf-8') as f:
        f.write("\nMASTER_GAIN = MASTER_GAIN * 0.5   # S7 demo: uncommitted local change\n")


def main():
    if prov.git_exe() is None:
        print("git executable not found: the 'differs' record needs an isolated repo")
        return 1
    _rmtree(DEMO_ROOT)
    os.makedirs(os.path.join(DEMO_ROOT, '.tmp'))
    cat = Catalog(DEMO_ROOT)
    print("[1/3] recording 'S7 exact' with this code ...")
    rid = record_session(cat, "S7 exact",
                         "Made by this code: A/B switches + harm 0.5 on B. Check -> exact.")
    print(f"      {rid}  ({cat.load(rid).version_label()})")
    print("[2/3] recording 'S7 differs' in an isolated copy with MASTER_GAIN halved ...")
    make_isolated_repo()
    # self-contained child code: it must import casynth_lab from the COPY
    # (cwd = REPO, PYTHONPATH stripped), never from this checkout
    import inspect
    code = ("import os, sys, time\n"
            "from casynth_config import SR\n"
            "from casynth_lab import load_scene, DemoRunner, BLOCK\n"
            "from casynth_lab.audio_out import LiveEngine\n"
            "from casynth_lab.catalog import Catalog\n"
            "from casynth_lab import provenance as prov\n"
            f"DEMO_AB = {os.path.join(REPO, 'demos', 'laplace_ab.json')!r}\n"
            f"SECONDS = {SECONDS!r}\nCOMMANDS = {COMMANDS!r}\n"
            + inspect.getsource(_sec) + inspect.getsource(record_session)
            + f"assert os.path.normcase(prov.current()['root']) == os.path.normcase({REPO!r})\n"
            f"cat = Catalog({DEMO_ROOT!r}, repo_root={REPO!r})\n"
            "print(record_session(cat, 'S7 differs', 'Made by a local copy of the code with "
            "MASTER_GAIN halved (uncommitted). Recomputed by the current code it is 6 dB louder.'))\n")
    p = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=REPO,
                       env=versions.child_env(), capture_output=True, text=True,
                       encoding='utf-8', timeout=300)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        return 1
    rid2 = p.stdout.strip().splitlines()[-1]
    rec2 = cat.load(rid2)
    print(f"      {rid2}  ({rec2.version_label()})")
    assert rec2.status == 'local' and rec2.status_reason.endswith('casynth_config.py'), \
        rec2.status_reason
    print("[3/3] 'S7 unavailable': a copy of record 1 whose side B engine no longer exists ...")
    d3 = os.path.join(DEMO_ROOT, rid + 'x')
    shutil.copytree(os.path.join(DEMO_ROOT, rid), d3)
    mp = os.path.join(d3, 'record.json')
    meta = json.load(open(mp, encoding='utf-8'))
    meta['title'] = "S7 unavailable"
    meta['note'] = ("Side B names engine 'laplacian_v0', which is not registered any more: "
                    "the check says why; the WAVs still play.")
    meta['scene']['variants']['B']['engine_id'] = 'laplacian_v0'
    meta['engines']['B'] = 'laplacian_v0'
    json.dump(meta, open(mp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"      {rid + 'x'}")
    print(f"\nDemo catalog ready: {DEMO_ROOT}")
    print("Open it:  run_demo_bench.bat --catalog lab_catalog\\s7_demo   (Catalog -> Check catalog)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
