#!/usr/bin/env python3
"""Does letting go of a key release the note -- and does HOLD keep it?  Headless.

WHY THIS EXISTS.  Until 2026-09-21 the prototype had no note-off at all from the
keyboard or the mouse: KEYDOWN latched the gate and nothing ever took it down,
so the VOICE release knob was decoration for anyone without a MIDI device.  The
gate below asks the two questions a player asks, on the REAL prototype driven by
REAL events:

    HOLD dark -> I let the key go, does the sound go?
    HOLD lit  -> I let the key go, does the sound STAY? (a note may only be
                 replaced, never released -- that is what the toggle promises)

It needs no sound card: with no device the host paces itself at real time.

Run:  python tests/ui_hold_probe.py         (exit 0 = both answers are right)
"""
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# the release is set to its maximum, so 2 s after the key comes up a released
# note is deep below the held level and a latched one is right where it was
RELEASED_MAX = 0.25             # rms after / rms held, when it must have gone
LATCHED_MIN = 0.60              # ... and when it must have stayed

DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_RECORD"] = "1"
os.environ["CASYNTH_RUN_SECONDS"] = "10"
os.environ["CASYNTH_NO_PERSIST"] = "1"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth
from casynth_config import GRID_H, CELL
BY = GRID_H * CELL + 8
HOLD = %(hold)d

def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

def drive():
    time.sleep(1.5)
    click(250, BY + 20)                 # Random: a field to sound
    time.sleep(0.3)
    click(40, BY + 20)                  # Play
    time.sleep(0.3)
    click(1176, BY + 20 + 3 * 22 + 4)   # VOICE R to its maximum
    if HOLD:
        click(1100, BY + 8)             # the HOLD toggle in the VOICE header
    time.sleep(0.4)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0,
                                         unicode='a', scancode=4))
    print("[probe] key down (hold=%%d)" %% HOLD, flush=True)
    time.sleep(3.0)
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_a, mod=0, scancode=4))
    print("[probe] key up", flush=True)

threading.Thread(target=drive, daemon=True).start()
gol_synth.main()
'''


def _run(hold):
    """-> (rms while the key is held, rms two seconds after it came up)."""
    for stale in glob.glob(os.path.join(ROOT, '_session_*.wav')) + \
            glob.glob(os.path.join(ROOT, '_session_*.npz')):
        os.remove(stale)
    env = dict(os.environ, PYTHONUTF8='1', CASYNTH_ENGINE='laplacian')
    out = subprocess.run([sys.executable, '-c', DRIVER % dict(root=ROOT, hold=hold)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding='utf-8', timeout=300)
    if out.returncode != 0:
        raise AssertionError(f"hold={hold}: the prototype exited {out.returncode}\n"
                             f"{(out.stderr or '')[-2000:]}")
    from scipy.io import wavfile
    wavs = sorted(glob.glob(os.path.join(ROOT, '_session_*.wav')))
    if not wavs:
        raise AssertionError(f"hold={hold}: nothing was recorded")
    _sr, pcm = wavfile.read(wavs[-1])
    pcm = pcm.astype(np.float64)
    held = pcm[int(4.0 * _sr):int(5.2 * _sr)]        # key down at ~2.5 s
    after = pcm[int(7.5 * _sr):int(8.7 * _sr)]       # key up at ~5.5 s
    for f in wavs + glob.glob(os.path.join(ROOT, '_session_*.npz')):
        os.remove(f)
    rms = lambda x: float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    return rms(held), rms(after)


def main():
    bad = []
    for hold, limit, want in ((0, RELEASED_MAX, 'released'),
                              (1, LATCHED_MIN, 'still sounding')):
        held, after = _run(hold)
        ratio = (after / held) if held else float('nan')
        ok = (held > 0.0) and (ratio <= limit if hold == 0 else ratio >= limit)
        print("hold=%d  rms held %8.1f -> after key up %8.1f   ratio %.3f   %s"
              % (hold, held, after, ratio, 'ok' if ok else 'WRONG'))
        if not ok:
            bad.append((hold, want, ratio))
    print()
    if bad:
        for hold, want, ratio in bad:
            print(f"  HOLD {'on' if hold else 'off'}: the note should be {want} "
                  f"two seconds after the key came up (ratio {ratio:.3f})")
        return 1
    print("the keyboard releases a note, and HOLD latches it")
    return 0


if __name__ == '__main__':
    sys.exit(main())
