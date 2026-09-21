#!/usr/bin/env python3
"""Does the prototype make a SIGNAL when a person plays it?  One command, headless.

WHY THIS EXISTS.  On 2026-09-21 the user reported no sound, and every gate was
green: check.py renders one frame and compares pixels, tests/test_seam.py checks
the engines in isolation, and the catalog check compares offline renders.  Not
one of them ever asked the question a player asks -- "I pressed Random and Play,
is the level moving?"  A broken hand-over of the engine, a serial that never
advances, a gate stuck shut, a volume stuck at zero or an engine that returns
silence would all have passed every gate while the instrument stood mute.

So this drives the REAL prototype with the REAL mouse events, on every engine it
offers, and asserts there is a signal: the level the synth reports, and the audio
it recorded.  It does not need a sound card -- without one the host paces itself
at real time, exactly as the bench does, so this runs on any machine.

Run:  python tests/ui_sound_probe.py            (exit 0 = the synth made a signal)
      SOUND_PROBE_ENGINES=laplacian,laplace_fm python tests/ui_sound_probe.py
"""
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# a field of a few hundred cells at -11 dBFS rms renders around 0.4..0.6 pre-clip;
# these thresholds only have to separate SOUND from SILENCE, not judge a level
MIN_PEAK = 0.02                 # what the synth reports as its own pre-clip peak
MIN_RMS = 50.0                  # of the recorded int16, out of 32767
SECONDS = 6

DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_RECORD"] = "1"
os.environ["CASYNTH_DEBUG_AUDIO"] = "1"
os.environ["CASYNTH_RUN_SECONDS"] = "%(seconds)d"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth
from casynth_config import GRID_H, CELL
BY = GRID_H * CELL + 8

def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

def drive():
    time.sleep(1.5)
    click(250, BY + 20)            # Random: a field to sound
    time.sleep(0.4)
    click(40, BY + 20)             # Play: let it live
    print("[probe] Random + Play", flush=True)

threading.Thread(target=drive, daemon=True).start()
gol_synth.main()
'''


def run_engine(eid):
    """-> (peak reported by the synth, rms of what it recorded, first dbg line)."""
    for stale in glob.glob(os.path.join(ROOT, '_session_*.wav')) + \
            glob.glob(os.path.join(ROOT, '_session_*.npz')):
        os.remove(stale)
    env = dict(os.environ, PYTHONUTF8='1')
    if eid:
        env['CASYNTH_ENGINE'] = eid
    else:
        env.pop('CASYNTH_ENGINE', None)
    code = DRIVER % {'seconds': SECONDS, 'root': ROOT}
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8',
                         timeout=SECONDS + 180)
    if out.returncode != 0:
        raise AssertionError(f"{eid}: the prototype exited {out.returncode}\n"
                             f"{(out.stderr or '')[-2000:]}")
    peak, cells, voices, last = 0.0, 0, 0, ''
    for line in (out.stdout or '').splitlines():
        if not line.startswith('[dbg]'):
            continue
        last = line
        f = dict(kv.split('=', 1) for kv in line.split()[1:] if '=' in kv)
        peak = max(peak, float(f.get('peak', 0.0)))
        cells = max(cells, int(f.get('cells', 0)))
        voices = max(voices, int(f.get('voices', 0)))
    wavs = sorted(glob.glob(os.path.join(ROOT, '_session_*.wav')))
    rms = 0.0
    if wavs:
        from scipy.io import wavfile
        _sr, pcm = wavfile.read(wavs[-1])
        rms = float(np.sqrt((pcm.astype(float) ** 2).mean())) if len(pcm) else 0.0
        os.remove(wavs[-1])
    for npz in glob.glob(os.path.join(ROOT, '_session_*.npz')):
        os.remove(npz)
    return dict(peak=peak, rms=rms, cells=cells, voices=voices, last=last)


def main():
    want = os.environ.get('SOUND_PROBE_ENGINES')
    if want:
        engines = [e.strip() for e in want.split(',') if e.strip()]
    else:
        from casynth_engines import registry
        engines = [e for e in registry.ids() if registry.get(e).plays_notes]
    bad = []
    for eid in engines:
        r = run_engine(eid)
        ok = r['peak'] >= MIN_PEAK and r['rms'] >= MIN_RMS and r['voices'] > 0
        print("%-24s cells %4d  voices %2d  peak %.3f  rms %7.1f   %s"
              % (eid, r['cells'], r['voices'], r['peak'], r['rms'],
                 'sound' if ok else 'NO SIGNAL'))
        if not ok:
            bad.append((eid, r))
    print()
    if bad:
        print("--- the instrument was MUTE on %d engine(s) ---" % len(bad))
        for eid, r in bad:
            why = ("the field never reached the engine" if r['voices'] == 0 else
                   "the engine rendered silence" if r['peak'] < MIN_PEAK else
                   "the engine had a level but nothing was recorded")
            print(f"  {eid}: {why}")
            print(f"    last state: {r['last']}")
        return 1
    print("all %d engines made a signal after Random + Play" % len(engines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
