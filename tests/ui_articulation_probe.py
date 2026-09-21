#!/usr/bin/env python3
"""Is the VOICE envelope audible at all?  The REAL prototype, REAL events.

WHY THIS EXISTS.  Twice in one day the VOICE block turned out to be decoration
and every gate stayed green:

  * until 2026-09-21 there was no note-off from the keyboard or the mouse at all
    (KEYDOWN latched the gate, KEYUP was not handled), so the release knob only
    ever ran for somebody with a MIDI device;
  * and a MELODY never articulated: in the first minute of the shipped MIDI file
    199 notes close the mono gate 16 times and every one of those gaps is
    0.000 s long, so an envelope that fires only on a gate EDGE sat at sustain
    while a line was played -- A, D and R did nothing anyone could hear.

So this asks the three questions a player asks, and needs no sound card (with no
device the host paces itself at real time):

    HOLD dark -> I let the key go, does the sound go?
    HOLD lit  -> I let the key go, does the sound STAY? (a note may only be
                 replaced, never released -- what the toggle promises)
    a file    -> with a slow attack and a low sustain, does the level MOVE from
                 note to note, although the gate never comes up?

Run:  python tests/ui_articulation_probe.py      (exit 0 = all three are right)
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


MIDI_FILE = 'moonlight-sonata.mid'
MIDI_SWING_MIN = 0.05           # how far the VCA must travel while a line plays

MIDI_DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_DEBUG_AUDIO"] = "1"
os.environ["CASYNTH_RUN_SECONDS"] = "12"
os.environ["CASYNTH_NO_PERSIST"] = "1"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth
from casynth_config import GRID_H, CELL
BY = GRID_H * CELL + 8
SLOW = %(slow)d

def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

def drive():
    time.sleep(2.0)
    if SLOW:
        click(1110, BY + 20 + 0 * 22 + 4)   # VOICE A: part-way up the track
        click(1100, BY + 20 + 2 * 22 + 4)   # VOICE S: low
        print("[probe] slow attack, low sustain", flush=True)

threading.Thread(target=drive, daemon=True).start()
gol_synth.main(autoplay_midi=%(midi)r)
'''


def _run_midi(slow):
    """Play the file and read the VCA level the render thread reports.
    -> (min, max) over the blocks after the knobs were moved, and the share of
    blocks whose gate was open."""
    env = dict(os.environ, PYTHONUTF8='1', CASYNTH_ENGINE='laplacian')
    out = subprocess.run([sys.executable, '-c',
                          MIDI_DRIVER % dict(root=ROOT, slow=slow, midi=MIDI_FILE)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding='utf-8', timeout=300)
    if out.returncode != 0:
        raise AssertionError(f"midi slow={slow}: the prototype exited "
                             f"{out.returncode}\n{(out.stderr or '')[-2000:]}")
    vca, gate = [], []
    for line in (out.stdout or '').splitlines():
        if line.startswith('[dbg]'):
            f = dict(kv.split('=', 1) for kv in line.split()[1:] if '=' in kv)
            vca.append(float(f.get('vca', -1)))
            gate.append(int(f.get('gate', -1)))
    vca = vca[3:]                      # after the knobs are moved
    if not vca:
        raise AssertionError(f"midi slow={slow}: the render thread said nothing")
    return min(vca), max(vca), (sum(gate) / len(gate) if gate else 0.0)


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
    # a MELODY: the gate stays open the whole time, and the envelope must still
    # be restarted note by note (the default knobs are a flat 1.0 by design)
    midi = []
    if os.path.exists(os.path.join(ROOT, MIDI_FILE)):
        for slow in (0, 1):
            lo, hi, open_share = _run_midi(slow)
            swing = hi - lo
            moves = swing > MIDI_SWING_MIN
            ok = moves if slow else (swing <= 1e-9)
            print("midi %-14s gate open %3.0f%%  vca %.3f..%.3f  %s"
                  % ('slow A / low S' if slow else 'defaults', open_share * 100,
                     lo, hi, 'ok' if ok else 'WRONG'))
            if not ok:
                midi.append((slow, swing))
    else:
        print(f"midi            skipped ({MIDI_FILE} is not here)")
    print()
    if bad:
        for hold, want, ratio in bad:
            print(f"  HOLD {'on' if hold else 'off'}: the note should be {want} "
                  f"two seconds after the key came up (ratio {ratio:.3f})")
    for slow, swing in midi:
        if slow:
            print(f"  a played line never re-articulated: the VCA moved {swing:.3f} "
                  f"with a slow attack and a low sustain -- the gate never comes up "
                  f"between notes, so the onset must restart the envelope")
        else:
            print(f"  the DEFAULT envelope moved by {swing:.3f}; it must be a flat 1.0")
    if bad or midi:
        return 1
    print("a key releases its note, HOLD latches it, and a played line "
          "re-articulates note by note")
    return 0


if __name__ == '__main__':
    sys.exit(main())
