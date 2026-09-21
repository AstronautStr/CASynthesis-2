#!/usr/bin/env python3
"""Does Save turn what was played into an EXPERIMENT the bench can open?  Headless.

WHY THIS EXISTS.  The instrument and the bench kept their work apart: a session
was a pile of _session_*.npz files a person had to convert by hand, while the
bench's catalog -- records with a scene, a WAV, an end snapshot to Continue from
and a pin on the commit -- could only be filled from the bench.  Save (2026-09-22)
closes that: the session the prototype logs all along becomes a scene
(casynth_session.scene_from_session) and that scene is rendered into a record
through the bench's own Recorder / Catalog.save (casynth_lab.offline_record).

So this plays the prototype for a few seconds, clicks Save, and then asks the
BENCH's own reader whether what came out is a record it can list, replay
byte-exact and continue from.

Run: python tests/ui_save_probe.py      (exit 0 = the experiment is in the catalog)
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from casynth_config import GRID_H, CELL                            # noqa: E402

BY = GRID_H * CELL + 8
SAVE_X = 349                               # middle of the Save button
SECONDS = 10

DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_RUN_SECONDS"] = "%(seconds)d"
os.environ["CASYNTH_NO_PERSIST"] = "1"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth
gol_synth._EXPERIMENT_ROOT = %(root_out)r

BY, SAVE_X = %(by)d, %(save_x)d

def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

def drive():
    time.sleep(1.5)
    click(207, BY + 20)                  # Random: a field to sound
    time.sleep(0.3)
    click(52, BY + 20)                   # Play
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0,
                                         unicode='a', scancode=4))
    time.sleep(3.0)                      # play a few generations
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_a, mod=0, scancode=4))
    time.sleep(0.3)
    print("[probe] Save", flush=True)
    click(SAVE_X, BY + 20)
    time.sleep(2.5)                      # the offline render happens on this click
    print("[probe] done", flush=True)

threading.Thread(target=drive, daemon=True).start()
gol_synth.main()
'''


def main():
    out_root = tempfile.mkdtemp(prefix='casynth_save_')
    env = dict(os.environ, PYTHONUTF8='1')
    code = DRIVER % {'seconds': SECONDS, 'root': ROOT, 'by': BY, 'save_x': SAVE_X,
                     'root_out': out_root}
    run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8',
                         timeout=SECONDS + 240)
    said = [l for l in (run.stdout or '').splitlines() if l.startswith('[save]')]
    if run.returncode != 0:
        print(f"[FAIL] the prototype exited {run.returncode}\n{(run.stderr or '')[-1500:]}")
        return 1
    try:
        from casynth_lab.catalog import Catalog
        cat = Catalog(root=out_root)
        ids = cat.ids()
        if not ids:
            print("[FAIL] Save produced no record. What it said: " + ' | '.join(said))
            return 1
        rec = cat.load(ids[-1])
        pcm = rec.pcm('monitor')
        seconds = rec.seconds
        if pcm is None or len(pcm) == 0:
            print("[FAIL] the record holds no audio")
            return 1
        if not rec.can_continue():
            print("[FAIL] the record cannot be continued (no end snapshot)")
            return 1
        res = cat.replay(ids[-1])
        if res.status != 'match':
            print(f"[FAIL] the bench cannot replay the record: {res.status} {res.reason}")
            return 1
        peak = int(abs(pcm).max())
        if peak == 0:
            print("[FAIL] the experiment is silent")
            return 1
        print(f"[PASS] Save wrote {ids[-1]} ({seconds:.1f}s, peak {peak}); "
              f"the bench lists it, replays it byte-exact and can continue it")
        return 0
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
        for stale in os.listdir(ROOT):
            if stale.startswith('_session_'):
                try:
                    os.remove(os.path.join(ROOT, stale))
                except OSError:
                    pass


if __name__ == '__main__':
    sys.exit(main())
