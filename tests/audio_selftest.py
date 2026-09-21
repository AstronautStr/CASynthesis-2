#!/usr/bin/env python3
"""Does the prototype actually put sound into the audio device?  One command.

"No sound" can mean five different things, and guessing between them by mail is
slow.  This runs the real prototype headless, clicks Random and Play with the
MOUSE the way a person does, and then reports FACTS:

  * did the output device open at all, and which one is it;
  * how many times PortAudio asked the callback for samples, and how many
    seconds of audio it was handed;
  * what the peak of what reached the DEVICE was (0 = the device got silence);
  * what the engine itself rendered (the two can differ: one is the ring, the
    other is what the device pulled out of it);
  * the cost of a block and the underruns.

Run:  python tests/audio_selftest.py
      CASYNTH_ENGINE=laplace_fm python tests/audio_selftest.py
      SELFTEST_SECONDS=12 python tests/audio_selftest.py

It needs a real output device -- it is a diagnosis, not a regression gate, and
is deliberately NOT part of check.py, which must run on a machine with no audio.
"""
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('CASYNTH_RUN_SECONDS', os.environ.get('SELFTEST_SECONDS', '8'))

import pygame                                                       # noqa: E402
import casynth_host                                                 # noqa: E402
from casynth_config import GRID_H, CELL, SR                         # noqa: E402

DEVICE = {'calls': 0, 'frames': 0, 'nonzero': 0, 'peak': 0}
_original_cb = casynth_host.AudioHost._default_cb


def _spy(self, outdata, frames, time_info, status):
    _original_cb(self, outdata, frames, time_info, status)
    DEVICE['calls'] += 1
    DEVICE['frames'] += frames
    a = np.abs(np.asarray(outdata))
    if a.size:
        DEVICE['nonzero'] += int(np.count_nonzero(a))
        DEVICE['peak'] = max(DEVICE['peak'], int(a.max()))


casynth_host.AudioHost._default_cb = _spy

import gol_synth                                                    # noqa: E402

BY = GRID_H * CELL + 8          # the toolbar's first row
RANDOM_X, PLAY_X = 250, 40


def click(x, y):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))


def drive():
    time.sleep(2.0)                      # the window, the engine and the device
    click(RANDOM_X, BY + 20)
    time.sleep(0.4)
    click(PLAY_X, BY + 20)
    print("[selftest] clicked Random, then Play", flush=True)


def device_name():
    try:
        import sounddevice as sd
        d = sd.query_devices(kind='output')
        return f"{d['name']} ({d['default_samplerate']:.0f} Hz, {d['max_output_channels']} ch)"
    except Exception as e:                                          # noqa: BLE001
        return f"unknown ({e})"


def main():
    os.environ['CASYNTH_RECORD'] = '1'
    print(f"[selftest] engine: {os.environ.get('CASYNTH_ENGINE', 'laplacian (default)')}")
    print(f"[selftest] output device: {device_name()}")
    threading.Thread(target=drive, daemon=True).start()
    gol_synth.main()

    secs = DEVICE['frames'] / float(SR)
    print("\n--- what the DEVICE was handed ---")
    print(f"  callback calls : {DEVICE['calls']}")
    print(f"  audio handed   : {secs:.2f} s")
    print(f"  peak           : {DEVICE['peak']} of 32767")
    print(f"  non-zero       : {DEVICE['nonzero']} samples")
    import glob
    wavs = sorted(glob.glob(os.path.join(ROOT, '_session_*.wav')))
    if wavs:
        from scipy.io import wavfile
        _sr, pcm = wavfile.read(wavs[-1])
        a = np.abs(pcm.astype(float))
        print("--- what the ENGINE rendered (the session recording) ---")
        print(f"  file           : {os.path.basename(wavs[-1])}")
        print(f"  peak           : {int(a.max())} of 32767")
        print(f"  rms            : {np.sqrt((a ** 2).mean()):.1f}")
    print("\n--- verdict ---")
    if DEVICE['calls'] == 0:
        print("  THE DEVICE NEVER ASKED FOR SAMPLES -- it did not open, or it is not"
              " running.  Look at the [audio ...] line above.")
        return 1
    if DEVICE['peak'] == 0:
        print("  the device ran, but everything it was handed was SILENCE."
              " The engine, the field or the gain is the place to look.")
        return 1
    print(f"  sound reached the device: peak {DEVICE['peak']}, {secs:.2f} s of it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
