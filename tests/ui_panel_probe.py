#!/usr/bin/env python3
"""Does a knob that decides another row make that row appear?  Headless.

WHY THIS EXISTS.  On 2026-09-22 the user reported that Laplace+ opens without
the `dyn` row, and that clicking `artic` makes it show up.  The engine says
`dyn` does not act while `shape` is 0 (laplace_unified.inactive) and the panel
leaves an inactive row out -- but the panel was only ever rebuilt when a TAB or
a MODE row was clicked, so raising `shape` with the slider changed the set of
rows and nothing redrew it.  Every gate passed: the frame dump sends no events,
and the click probe only asserts that nothing raised.

So this drives the real prototype, drags `shape` down and back up, and reads the
rows the panel actually built (CASYNTH_PANEL_LOG, a test hook like
CASYNTH_DUMPFRAME).

It then TYPES A RANGE into the Max field of `part` (2026-09-22: every slider has
a Min / Max pair and the track spans between them) and checks the slider took it,
that a refused value leaves the old range standing, and that a right click gives
the whole registry range back.

Run: python tests/ui_panel_probe.py        (exit 0 = the panel obeyed)
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from casynth_config import GRID_H, CELL                            # noqa: E402

BY = GRID_H * CELL + 8                     # the toolbar's first row
CTRL_X = 790                               # inside column A's slider track
MAX_FIELD_X = 860                          # inside the Max field of that row
ROW_H = 22                                 # the panel's row pitch
SHAPE_ROW = 3                              # n, spread, alpha, SHAPE, harm, full, dyn
SECONDS = 8

DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_RUN_SECONDS"] = "%(seconds)d"
os.environ["CASYNTH_NO_PERSIST"] = "1"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth

BY, CTRL_X, ROW_H, SHAPE_ROW = %(by)d, %(x)d, %(row_h)d, %(shape_row)d
Y = BY + 2 + SHAPE_ROW * ROW_H + 4

def drag(x, y, to):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(to, y),
                                         buttons=(1, 0, 0), rel=(to - x, 0)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(to, y), button=1))

PART_Y = BY + 2 + 0 * ROW_H + 4            # the `part` row (first of the panel)

def click(x, y, button=1):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button))

def typed(text, key=pygame.K_RETURN):
    for ch in text:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=ord(ch), mod=0,
                                             unicode=ch, scancode=0))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0,
                                         unicode='', scancode=0))

def drive():
    time.sleep(1.5)
    print("[probe] shape -> 0", flush=True)
    drag(CTRL_X, Y, CTRL_X - 400)          # off the left end of the track -> 0
    time.sleep(0.6)
    print("[probe] shape -> 1", flush=True)
    drag(CTRL_X, Y, CTRL_X + 400)          # off the right end -> full
    time.sleep(0.6)
    print("[probe] part: max -> 8", flush=True)
    click(%(maxx)d, PART_Y)                # the Max field of `part`
    time.sleep(0.2)
    typed("8")                             # replaces the selected text
    time.sleep(0.4)
    print("[probe] part: max -> 999 (refused)", flush=True)
    click(%(maxx)d, PART_Y)
    time.sleep(0.2)
    typed("999")                           # above the registry bound: refused
    time.sleep(0.4)
    print("[probe] part: right click resets", flush=True)
    click(%(maxx)d, PART_Y, button=3)
    time.sleep(0.4)
    print("[probe] done", flush=True)

threading.Thread(target=drive, daemon=True).start()
gol_synth.main()
'''


def main():
    log = os.path.join(tempfile.mkdtemp(prefix='casynth_panel_'), 'panel.log')
    env = dict(os.environ, PYTHONUTF8='1', CASYNTH_PANEL_LOG=log,
               CASYNTH_ENGINE='laplace_unified')
    code = DRIVER % {'seconds': SECONDS, 'root': ROOT, 'by': BY, 'x': CTRL_X,
                     'row_h': ROW_H, 'shape_row': SHAPE_ROW, 'maxx': MAX_FIELD_X}
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8',
                         timeout=SECONDS + 120)
    if out.returncode != 0:
        print(f"[FAIL] the prototype exited {out.returncode}\n{(out.stderr or '')[-1500:]}")
        return 1
    try:
        with open(log, encoding='utf-8') as f:
            rows = [line.split() for line in f if line.strip()]
    except OSError as e:
        print(f"[FAIL] the panel log was never written ({e})")
        return 1
    os.remove(log)
    if not rows:
        print("[FAIL] the panel was never built")
        return 1
    names = [[t.split('=')[0] for t in r[1:]] for r in rows]
    has_dyn = ['dyn' in n for n in names]
    if not has_dyn[0]:
        print("[FAIL] Laplace+ opened WITHOUT the dyn row: " + ' '.join(rows[0][1:]))
        return 1
    if True not in has_dyn or False not in has_dyn:
        print("[FAIL] the dyn row never moved: shape 0 must hide it and shape 1 bring it back")
        for r in rows:
            print("   ", ' '.join(r))
        return 1
    # the last rebuild must be the one where shape is full again
    if not has_dyn[-1]:
        print("[FAIL] raising shape did not bring the dyn row back by itself")
        for r in rows:
            print("   ", ' '.join(r))
        return 1
    # ... and the typed range: `part` spans 1..8, keeps it when 999 is refused,
    # and goes back to the registry range on the right click
    spans = [dict(t.split('=', 1) for t in r[1:] if '=' in t).get('n') for r in rows]
    if '1..8' not in spans:
        print(f"[FAIL] typing 8 into Max never reached the slider: {spans}")
        for r in rows:
            print("   ", ' '.join(r))
        return 1
    after = spans[spans.index('1..8'):]
    if after[-1] != '1..20':
        print(f"[FAIL] the right click did not restore the registry range: {spans}")
        return 1
    if any(s and s not in ('1..8', '1..20') for s in after):
        print(f"[FAIL] a refused range changed the slider anyway: {spans}")
        return 1
    print(f"[PASS] the panel followed the knobs ({len(rows)} rebuilds; dyn hidden at "
          f"shape 0 and back at shape 1; part typed to 1..8, 999 refused, reset to 1..20)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
