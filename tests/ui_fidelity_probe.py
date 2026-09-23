#!/usr/bin/env python3
"""Is what the instrument PLAYED what its session RENDERS offline?  Byte for byte,
the REAL prototype, headless.

WHY THIS EXISTS (2026-09-23, coverage audit).  Everything the prototype's own
render loop does -- when a generation is applied (the device-clock stamp), the
VCA folded into the gain, the gain glide, the note as a transpose, the hosting of
the engine -- was held by no gate at all: the probes asked "is there a signal"
and "does the envelope move", and ui_save_probe proved that the record Save
writes replays byte-exact, which is offline against offline.  The only
live-versus-offline comparison was the manual `python gol_synth.py scene <ts>`,
and on 2026-09-22 it was recorded as "diverging from the first edit of the
field".  It did: on the instrument's opening settings the offline render of a
Random + Play session was 2.7x louder for the first second and shared 5 % of its
samples with the recording.

The whole of that divergence was ONE difference between the two hosts: after a
manual edit of the field (Random, Clear, painting, a dropped pattern) the bench
runner hands the engine an excitation `events_field(prev, new)` -- the edit as
births and deaths -- while the prototype leaves `exc_field` alone (None until
the first automaton step, the previous step's field after).  With `dyn` = 0 the
two are the same sound; on the opening spectrum (`dyn` 1) they are not.  Which
host is right is a design question (memory/questions.md, 2026-09-23); until it
is answered this probe renders the scene the way the PROTOTYPE means it, so
that everything else in the live loop is pinned NOW: with that one convention
applied, live and offline are byte-identical (339 328 of 339 328 samples on
2026-09-23).

So this drives the real prototype (Random, Play, a key, a few generations)
with CASYNTH_RECORD=1, turns the session into a scene exactly as Save does
(casynth_session.scene_from_session), renders that scene offline through the
bench's runner, and asserts the two WAVs are the same bytes.

Run: python tests/ui_fidelity_probe.py      (exit 0 = live == offline, byte for byte)
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np

os.environ.setdefault('CASYNTH_VOLUME', '0.01')   # a gate must not play music at the room
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from casynth_config import SR, CHUNK_S, GRID_H, CELL                   # noqa: E402

BY = GRID_H * CELL + 8
BLOCK = int(CHUNK_S * SR)
SECONDS = 8

DRIVER = r'''
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["CASYNTH_RECORD"] = "1"
os.environ["CASYNTH_RUN_SECONDS"] = "%(seconds)d"
os.environ["CASYNTH_NO_PERSIST"] = "1"
sys.path.insert(0, %(root)r)
import pygame
import gol_synth
BY = %(by)d

def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

def drive():
    time.sleep(1.5)
    click(207, BY + 20)                   # Random: a field to sound (a manual edit)
    time.sleep(0.4)
    click(52, BY + 20)                    # Play: generations on the device clock
    time.sleep(2.0)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0,
                                         unicode='d', scancode=7))    # a note: a transpose
    time.sleep(2.0)
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_d, mod=0, scancode=7))

threading.Thread(target=drive, daemon=True).start()
gol_synth.main()
'''


def _clean():
    for stale in glob.glob(os.path.join(ROOT, '_session_*.wav')) + \
            glob.glob(os.path.join(ROOT, '_session_*.npz')):
        try:
            os.remove(stale)
        except OSError:
            pass


def record_session():
    """Play the prototype headless; -> the timestamp of the session it dumped."""
    _clean()
    env = dict(os.environ, PYTHONUTF8='1')
    code = DRIVER % dict(seconds=SECONDS, root=ROOT, by=BY)
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8',
                         timeout=SECONDS + 180)
    if out.returncode != 0:
        raise AssertionError(f"the prototype exited {out.returncode}\n"
                             f"{(out.stderr or '')[-2000:]}")
    m = re.search(r'_session_(\d{8}_\d{6})\.wav', out.stdout or '')
    if not m:
        raise AssertionError("the prototype dumped no session:\n" + (out.stdout or '')[-1500:])
    return m.group(1)


def render_as_the_prototype_means_it(doc, info):
    """The scene rendered offline through the bench runner, with the ONE host
    convention the prototype has and the bench has not (see the module doc):
    a manual edit of the field leaves the excitation as it was."""
    from casynth_lab import scene_from_doc, render_offline, DemoRunner
    scene = scene_from_doc(doc)
    runner = DemoRunner(scene, vol=info['vol0'])
    orig_apply = runner._apply

    def apply(kind, args):
        orig_apply(kind, args)
        if kind == 'set_cells':
            runner.exc = runner.exc_of_last_step      # None before the first step
            runner._field_changed()
    runner.exc_of_last_step = None
    orig_step = runner._step

    def step():
        orig_step()
        runner.exc_of_last_step = runner.exc
    runner._step = step
    runner._apply = apply
    pcm, _r = render_offline(scene, info['samples'] / float(SR),
                             commands=[('start', 0, {})], runner=runner, output='A')
    return pcm


def main():
    from scipy.io import wavfile
    from casynth_session import scene_from_session
    ts = record_session()
    try:
        doc, info = scene_from_session(ts)
        _sr, live = wavfile.read(os.path.join(ROOT, f'_session_{ts}.wav'))
        live = live[info['offset']:]
        pcm = render_as_the_prototype_means_it(doc, info)
        n = min(len(live), len(pcm))
        print(f"[fidelity] session {ts}: {info['frames']} frames, {info['steps']} generations, "
              f"{info['onsets']} note events, {info['edits']} field edits; "
              f"live {len(live)} / offline {len(pcm)} samples, comparing {n}")
        if n < 4 * SR or info['steps'] < 4 or info['onsets'] < 1 or info['edits'] < 1:
            print("[FAIL] the session is too thin to prove anything: it needs a few seconds, "
                  "generations, a note and the Random edit")
            return 1
        diff = np.abs(live[:n].astype(np.int64) - pcm[:n].astype(np.int64))
        bad = diff.max(axis=1) > 0
        if not bad.any() and len(live) == len(pcm):
            peak = int(np.abs(live).max())
            if peak == 0:
                print("[FAIL] both are silent: the comparison proves nothing")
                return 1
            print(f"[PASS] what the instrument played is what its scene renders: "
                  f"{n} samples byte-identical (peak {peak}), {info['steps']} generations "
                  f"on the device clock, a note and the VCA included")
            return 0
        first = int(np.argmax(bad)) if bad.any() else -1
        same = 100.0 * (1.0 - bad.mean())
        print(f"[FAIL] live and offline differ: {same:.2f} % of samples identical, "
              f"max|diff| {int(diff.max())} LSB, first difference at sample {first} "
              f"({first / SR:.3f} s, block {first // BLOCK}); lengths live {len(live)} "
              f"offline {len(pcm)}.  Diagnose with: python gol_synth.py scene {ts}")
        os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
        wavfile.write(os.path.join(ROOT, 'artifacts', f'_fidelity_offline_{ts}.wav'), SR, pcm)
        return 1
    finally:
        pass


if __name__ == '__main__':
    rc = main()
    if rc == 0:
        _clean()          # a failed run keeps its session for `gol_synth.py scene <ts>`
    sys.exit(rc)
