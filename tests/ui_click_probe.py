#!/usr/bin/env python3
"""Click every clickable thing in the prototype, headless, and exit 0.

WHY THIS EXISTS.  check.py renders ONE frame of gol_synth and compares pixels --
it never sends the window an event, so nothing in the event loop was ever
executed by a gate.  On 2026-09-21 that let a KeyError reach the user on the
most ordinary action there is: clicking Play.  The hit-test over the knob rows
runs BEFORE the toolbar buttons in the same if/elif chain, so a knob row missing
its hit rectangle crashed every mouse click in the toolbar -- while the keyboard
shortcut for the same action worked, which is exactly why the bug survived.

The probe posts a scripted sequence (never random -- a gate has to fail the same
way twice) that reaches every branch of that chain: the toolbar buttons, the BPM
slider, a note-division button, the volume slider, every knob row of the panel
(engine rows AND the synth-wide VOICE / GEN / Tune rows), every engine tab, the
piano, and painting on the field.  It asserts nothing about the sound -- it
asserts that the prototype survives being used.

Run: python tests/ui_click_probe.py            (exit 0 = nothing raised)
     CASYNTH_ENGINE=<id> python tests/ui_click_probe.py
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('CASYNTH_RUN_SECONDS', '9')

import pygame                                                    # noqa: E402
from casynth_config import (GRID_W, GRID_H, CELL, TOOLBAR_H, PIANO_H)  # noqa: E402
import gol_synth                                                 # noqa: E402

W = GRID_W * CELL
BY = GRID_H * CELL + 8                      # the toolbar's first row
TAB_Y = GRID_H * CELL + TOOLBAR_H - 24      # the engine tabs
MIDI_Y = GRID_H * CELL + TOOLBAR_H - 48     # the MIDI bar
DIV_Y = BY + 46                             # the note-division buttons
INFO_Y = GRID_H * CELL + 76
PIANO_Y = GRID_H * CELL + TOOLBAR_H + PIANO_H // 2
CTRL_X = 740                                # inside column A's slider track
RC_X = 1090                                 # inside the right column's track
VOL_SECTION_Y = BY + 228

# (x, y, what it is) -- every branch of the MOUSEBUTTONDOWN chain in gol_synth
CLICKS = [
    (40, BY + 20, 'play button'),
    (160, BY + 20, 'step button'),
    (250, BY + 20, 'random button'),
    (350, BY + 20, 'clear button'),
    (440, BY + 20, 'BPM slider'),
    (20, DIV_Y + 9, 'note division'),
    (600, BY + 20, 'pattern library button'),
    (RC_X, VOL_SECTION_Y + 66, 'volume slider'),
    (CTRL_X, BY + 2 + 0 * 22 + 4, 'engine knob row 1'),
    (CTRL_X, BY + 2 + 5 * 22 + 4, 'engine knob row 6 (the on/off pill)'),
    (CTRL_X, BY + 2 + 6 * 22 + 4, 'engine knob row 7'),
    (RC_X, BY + 20 + 0 * 22 + 4, 'VOICE A'),
    (RC_X, BY + 20 + 3 * 22 + 4, 'VOICE R'),
    (RC_X, BY + 128 + 0 * 22 + 4, 'GEN A'),
    (RC_X, BY + 128 + 3 * 22 + 4, 'GEN R'),
    (RC_X, BY + 128 + 4 * 22 + 4, 'Tune'),
    (1060, BY + 109, 'GEN amp-slew toggle'),
    (100, 200, 'paint a cell'),
    (120, 220, 'erase a cell'),
    (300, PIANO_Y, 'piano key'),
]
TABS = [45, 100, 150, 210, 280, 350, 440, 545]     # x of each engine tab


def post(kind, **kw):
    pygame.event.post(pygame.event.Event(kind, **kw))


def click(x, y, button=1, drag=0):
    post(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button)
    if drag:
        post(pygame.MOUSEMOTION, pos=(x + drag, y), buttons=(1, 0, 0), rel=(drag, 0))
    post(pygame.MOUSEBUTTONUP, pos=(x + drag, y), button=button)


def drive():
    time.sleep(1.2)                       # let the first frame publish and the device open
    for x, y, what in CLICKS:
        click(x, y, button=(3 if 'erase' in what else 1), drag=(12 if 'slider' in what else 0))
        print(f"[probe] {what}", flush=True)
        time.sleep(0.06)
    for i, x in enumerate(TABS):          # every engine the prototype offers
        click(x, TAB_Y)
        print(f"[probe] engine tab {i}", flush=True)
        time.sleep(0.35)                  # let it be built, handed over and crossfaded
        click(CTRL_X, BY + 2 + 4)         # and its first knob row, whatever widget it is
        time.sleep(0.06)
    click(45, TAB_Y)                      # back to the first engine
    print("[probe] all clicks delivered", flush=True)


def main():
    threading.Thread(target=drive, daemon=True).start()
    gol_synth.main()
    print("[probe] the prototype survived every click")
    return 0


if __name__ == '__main__':
    sys.exit(main())
