#!/usr/bin/env python3
"""
gol_life_synth_laplacian.py - CASynth v2: Laplacian-mode spectrum.

Same UI and GOL simulation as gol_life_synth.py.  The difference is the
mapping from field objects to audio:

    gol_life_synth.py  : size -> harmonic number k  (harmonic, organ-like)
    THIS FILE          : map_laplacian(object_shape) -> resonant mode freqs
                         (inharmonic, metallic / bell-like)

Each connected object in the GOL field produces a VOICE whose spectrum equals
the Laplacian eigenfrequencies of its graph (sqrt(lambda), normalised so the
lowest mode == carrier f0).  The carrier note is set via the on-screen piano.

Cross-fade on topology change
------------------------------
When GOL steps change an object's shape the set of Laplacian modes changes
abruptly.  Without smoothing this causes audible clicks / "beeps".  Solution:
  - Each voice×mode occupies two physical slots (front / back banks).
  - On a frequency change: the old (front) slot fades out over FADE_CHUNKS audio
    chunks (release); the new frequency immediately starts on the back slot from
    amplitude 0 (attack).  Both slots render in parallel -> overlapping envelopes,
    no amplitude gap ("struck plate" analogy: old resonance still ringing while
    the new one rises).
  - Phase accumulation is continuous -- no phase reset.  Starting the new slot
    at amp_cur=0 means the phase value is inaudible at the moment of assignment.
  - Mode frequencies are NOT glided -- they are discrete resonances; gliding
    would smear the inharmonic character that makes the timbre metallic.

CONTROLS  (identical to gol_life_synth.py)
  Mouse on field : left-drag draw, right-drag erase
  Piano (bottom) : click to set carrier note (latched)
  Volume slider  : drag (top-right toolbar)
  Space ......... play / pause
  S ............. step (key not mapped -- use button)
  R / C ......... random / clear
  Up / Down ..... faster / slower
  Esc ........... quit

RUN
    pip install pygame-ce numpy scipy sounddevice
    python gol_life_synth_laplacian.py
    # headless:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python gol_life_synth_laplacian.py
"""

import os
import colorsys
import queue as _queue
import numpy as np
import pygame
try:
    import sounddevice as sd
except Exception:          # optional; audio is just disabled if unavailable
    sd = None
from scipy import ndimage
from casynth_core import extract, map_laplacian
from patterns import PATTERNS

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  -- all tunable constants here
# ──────────────────────────────────────────────────────────────────────────────
GRID_W, GRID_H = 52, 30
CELL = 20
TOOLBAR_H = 96
PIANO_H = 96
FPS = 60

SR = 44100
STEP_HZ = 6.0            # generations per second (start value)
RANDOM_DENSITY = 0.28
MAX_VOICES = 24          # max connected objects to sonify

CHUNK_S = 0.09           # audio render chunk length (seconds)
# Audio is rendered AHEAD into a ring buffer drained by a sounddevice callback on
# the audio thread (see main()).  This many chunks of look-ahead absorb frame-loop
# jitter so a slow frame shrinks the buffer instead of gapping playback (the cause
# of the live clicks: a session probe showed 36 mixer underruns on normal frames
# under the old pygame play/queue model).  Look-ahead is also added latency, so
# keep it small: measured frames are very steady (max ~23 ms) so 2 chunks
# (~180 ms cushion) is ample.  Total output latency = this buffer + PortAudio's
# own latency (we request 'low' below).
AUDIO_LOOKAHEAD_CHUNKS = 2
# Headroom: up to MAX_VOICES objects, each up to MAX_MODES_PER_OBJ partials, plus
# front+back overlap during cross-fades -> the summed signal can peak well above
# 1.0 and hard-clip (audible distortion).  Measured raw peak on a dense field is
# ~5.6, on the 3-pentadecathlon scene ~3.0.  MASTER_GAIN gives fixed headroom so
# the sum stays below 1.0 without per-chunk auto-gain (which would pump / modulate
# the spectrum).  It only scales the overall level -- partial balance / timbre is
# unchanged.
MASTER_GAIN = 0.04
VOL_DEFAULT = 0.70
VOL_W = 120

# Laplacian mode parameters
MAX_MODES_PER_OBJ = 8   # Laplacian modes kept per object
PATCH_SIZE = 8           # extraction window for map_laplacian

# Total partial slots in the flat audio engine arrays:
#   slot 0 unused (legacy), slots 1..TOTAL_SLOTS used.
#   Each voice×mode occupies TWO slots: front (currently sounding) and back
#   (reserved for overlapping cross-fade).  Layout per voice v, mode m:
#     front slot: 1 + v * MAX_MODES_PER_OBJ * 2 + m
#     back  slot: 1 + v * MAX_MODES_PER_OBJ * 2 + MAX_MODES_PER_OBJ + m
TOTAL_SLOTS = MAX_VOICES * MAX_MODES_PER_OBJ * 2  # 384

# Cross-fade duration: ~50 ms expressed in audio chunks.
# At CHUNK_S=0.09 s: int(round(0.05/0.09)) = 1 chunk (~90 ms).
# Increase CHUNK_S or use a smaller CHUNK_S to get finer resolution.
FADE_MS = 50                                        # target fade duration in ms
FADE_CHUNKS = max(1, round(FADE_MS / 1000.0 / CHUNK_S))  # chunks per fade phase

# keyboard piano
NOTE_DEFAULT = 48         # C3
KB_BASE_MIN  = 24         # C1
KB_BASE_MAX  = 72         # C5

# palette
C_BG = (12, 14, 18)
C_GRID = (26, 30, 36)
C_PANEL = (22, 24, 30)
C_EDGE = (40, 44, 52)
C_TXT = (200, 206, 214)
C_DIM = (120, 126, 134)
C_BTN = (38, 42, 52)
C_BTN_HOT = (58, 64, 78)
C_ACCENT = (111, 208, 224)
C_CAPPED = (74, 78, 86)
C_WHITE = (228, 231, 236)
C_WHITE_ON = (120, 200, 215)
C_BLACK = (28, 30, 36)
C_BLACK_ON = (78, 158, 174)

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
WHITE_PC = {0, 2, 4, 5, 7, 9, 11}
BLACK_PC = {1, 3, 6, 8, 10}
TWO_PI = 2 * np.pi

# Per-chunk amplitude/pan ramp shape.  A plain linear ramp (np.linspace) is
# continuous in value across chunk boundaries but its SLOPE jumps there (each
# chunk re-aims at a new target) -> a corner in the envelope -> an audible click
# ("цык") on every chunk whose target changed (i.e. on every cross-fade /
# topology event).  A raised-cosine (smoothstep) ramp has zero slope at both
# ends, so consecutive chunks meet with matching (zero) slope -> C1-continuous
# envelope -> no corner, no click.  For a steady slot (amp_cur == amp_tgt) the
# ramp is flat either way, so steady tones are bit-unchanged.  For uncorrelated
# cross-fading modes the cosine shape is also closer to equal-power.
_N_CHUNK = int(CHUNK_S * SR)
_RAMP = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, _N_CHUNK))) * 0.5  # 0->1, flat ends

_KB_PIANO = {
    pygame.K_a: 0,  pygame.K_w: 1,
    pygame.K_s: 2,  pygame.K_e: 3,
    pygame.K_d: 4,
    pygame.K_f: 5,  pygame.K_t: 6,
    pygame.K_g: 7,  pygame.K_y: 8,
    pygame.K_h: 9,  pygame.K_u: 10,
    pygame.K_j: 11,
    pygame.K_k: 12, pygame.K_o: 13,
    pygame.K_l: 14, pygame.K_p: 15,
}

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SIDEBAR_W  = 192
_SB_ITEM_H = 40
_SB_PREV_W = 50
_SB_PREV_H = 36
_SB_HDR_H  = 22


def midi_to_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def note_name(n):
    return f"{NAMES[n % 12]}{n // 12 - 1}"


# ──────────────────────────────────────────────────────────────────────────────
# GAME OF LIFE  (toroidal, identical to gol_life_synth.py)
# ──────────────────────────────────────────────────────────────────────────────
_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_S8 = np.ones((3, 3), np.uint8)


def step(grid):
    n = ndimage.convolve(grid, _NEIGH, mode='wrap')
    return ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)


# LAPLACIAN MAPPING  (extract / map_laplacian imported from casynth_core --
# single source of truth shared with mapping_bench.py).  This prototype sounds
# MAX_MODES_PER_OBJ partials per object; the bench uses K_MAX.  The mode count is
# passed explicitly at the call site (see analyse) so the algorithm stays shared.

# ──────────────────────────────────────────────────────────────────────────────
# OBJECT ANALYSIS  (GOL field -> list of voice specs)
# ──────────────────────────────────────────────────────────────────────────────

def hsv(h, v):
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def analyse(grid, f0):
    """Segment the field into connected objects and compute Laplacian voices.

    Returns:
        labels  : labelled grid (0 = background)
        voices  : list of dicts with keys 'freqs', 'amps', 'pan', 'label_idx'
        color   : dict  label -> RGB colour tuple
    """
    labels, n = ndimage.label(grid, structure=_S8)
    if n == 0:
        return labels, [], {}

    objs = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        objs.append((lab, len(xs), ys, xs))
    objs.sort(key=lambda o: -o[1])   # largest first

    voices, color = [], {}
    for idx, (lab, area, ys, xs) in enumerate(objs[:MAX_VOICES]):
        # Build sub-grid containing ONLY this object's cells
        # (raw grid slice may include neighbouring objects if bboxes overlap)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.uint8)
        sub[ys - r0, xs - c0] = 1
        patch = extract(sub, PATCH_SIZE)

        cx = xs.mean() / max(GRID_W - 1, 1)   # stereo pan [0..1]

        freqs, amps = map_laplacian(patch, f0, MAX_MODES_PER_OBJ)

        # Colour: hue by object index (spread across spectrum), brightness by area
        hue = (idx / max(MAX_VOICES - 1, 1)) * 0.72
        val = float(np.clip(0.3 + 0.7 * min(area / 20.0, 1.0), 0.3, 1.0))
        color[lab] = hsv(hue, val)

        voices.append({
            'freqs':     freqs,             # shape (MAX_MODES_PER_OBJ,)
            'amps':      amps,              # shape (MAX_MODES_PER_OBJ,)
            'pan':       float(np.clip(cx, 0.0, 1.0)),
            'label_idx': lab,
        })

    for (lab, area, ys, xs) in objs[MAX_VOICES:]:
        color[lab] = C_CAPPED

    return labels, voices, color


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO ENGINE  -- flat slot pool with per-slot crossfade
# ──────────────────────────────────────────────────────────────────────────────

def render_chunk_laplacian(phase, amp_cur, pan_cur,
                           amp_tgt, pan_tgt, freq_slots, channels):
    """Render one gapless chunk over the flat slot pool.

    Amplitudes and pan are linearly interpolated (glided) from their current
    values to the target values within each chunk.  Phase accumulation is
    continuous across chunks -- there is no phase reset.  Overlapping crossfade
    (old frequency fading out while new frequency rises) is achieved by SlotPool
    keeping two physical slots per voice×mode (front and back banks); both slots
    are rendered here simultaneously during a transition.

    Parameters (all length TOTAL_SLOTS+1, slot 0 unused)
    ---------
    phase     : float64 array, phase accumulator per slot (mutated in place)
    amp_cur   : float64 array, current amplitudes (mutated -> amp_tgt each chunk)
    pan_cur   : float64 array, current pan         (mutated -> pan_tgt each chunk)
    amp_tgt   : float64 array, target amplitudes
    pan_tgt   : float64 array, target pan
    freq_slots: float64 array, frequency in Hz per slot (0 = silent)

    Returns int16 buffer (n_samples x channels).
    """
    n = int(CHUNK_S * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    idx = np.arange(n)
    guard = 0.45 * SR

    for k in range(1, TOTAL_SLOTS + 1):
        if amp_cur[k] < 1e-4 and amp_tgt[k] < 1e-4:
            continue
        freq = freq_slots[k]
        if freq <= 0.0 or freq >= guard:
            amp_cur[k] = 0.0
            continue
        inc = TWO_PI * freq / SR
        wave = np.sin(phase[k] + inc * idx)
        a = amp_cur[k] + (amp_tgt[k] - amp_cur[k]) * _RAMP
        p = pan_cur[k] + (pan_tgt[k] - pan_cur[k]) * _RAMP
        L += wave * a * np.cos(p * np.pi / 2.0)
        R += wave * a * np.sin(p * np.pi / 2.0)
        phase[k] = (phase[k] + inc * n) % TWO_PI
        amp_cur[k] = amp_tgt[k]
        pan_cur[k] = pan_tgt[k]

    L = np.clip(L * MASTER_GAIN, -1.0, 1.0)
    R = np.clip(R * MASTER_GAIN, -1.0, 1.0)

    if channels <= 1:
        data = ((L + R) * 0.5)[:, None]
    else:
        data = np.zeros((n, channels))
        data[:, 0] = L
        data[:, 1] = R

    return np.ascontiguousarray((data * 32767).astype(np.int16))


class SlotPool:
    """Manages assignment of voice×mode pairs to flat audio slots with
    overlapping amplitude cross-fade when mode frequencies change.

    Layout: slots 1..TOTAL_SLOTS.  Slot 0 is reserved / unused.
    Each voice×mode pair occupies TWO slots (front bank and back bank):
        stride = MAX_MODES_PER_OBJ * 2
        front slot for voice v, mode m:  1 + v * stride + bank[v,m] * MAX_MODES_PER_OBJ + m
        back  slot for voice v, mode m:  1 + v * stride + (1 - bank[v,m]) * MAX_MODES_PER_OBJ + m

    On a frequency change for voice v / mode m:
      1. The current front slot is put into RELEASE: amp_tgt -> 0, decaying over
         FADE_CHUNKS audio chunks.
      2. The back slot immediately becomes the new front: freq set to new value,
         amp_cur forced to 0 (inaudible start), amp_tgt set to the target
         amplitude (ATTACK ramp begins immediately).
      3. During the fade both slots are rendered in parallel -> OVERLAP, no gap.
      4. front/back bank index is swapped for that voice×mode.

    Result: the old resonance is still sounding while the new one rises -> the
    "struck plate" analogy from decisions.md is faithfully reproduced.
    """

    def __init__(self):
        sz = TOTAL_SLOTS + 1   # +1 so slot 0 is a harmless dummy
        self.freq_slots  = np.zeros(sz)        # Hz; 0 = silent
        self.amp_tgt     = np.zeros(sz)
        self.pan_tgt     = np.full(sz, 0.5)

        # Per voice×mode: which bank (0 or 1) is the current FRONT.
        # Shape (MAX_VOICES, MAX_MODES_PER_OBJ); int8.
        self._bank = np.zeros((MAX_VOICES, MAX_MODES_PER_OBJ), dtype=np.int8)

        # Release countdown (in chunks) per slot; 0 = not releasing.
        self._release_cnt = np.zeros(sz, dtype=int)
        # Amplitude at the start of release (for linear step computation).
        self._release_amp0 = np.zeros(sz)

        # Last committed frequency per slot (to detect changes).
        self._freq_prev = np.zeros((MAX_VOICES, MAX_MODES_PER_OBJ))

    # ------------------------------------------------------------------
    # Slot index helpers
    # ------------------------------------------------------------------

    def _stride(self):
        return MAX_MODES_PER_OBJ * 2

    def _front_slot(self, v, m):
        """Slot index of the current front bank for voice v, mode m."""
        bank = int(self._bank[v, m])
        return 1 + v * self._stride() + bank * MAX_MODES_PER_OBJ + m

    def _back_slot(self, v, m):
        """Slot index of the back bank for voice v, mode m."""
        bank = 1 - int(self._bank[v, m])
        return 1 + v * self._stride() + bank * MAX_MODES_PER_OBJ + m

    # ------------------------------------------------------------------

    def update(self, voices, amp_cur):
        """Push a new voice list into the slot pool.

        Called once per GOL step (or audio feed cycle).  For each active
        voice/mode:
          - If the frequency is unchanged: update amp/pan targets smoothly on
            the front slot.
          - If the frequency changed (topology event):
              a) Put the front slot into RELEASE: schedule FADE_CHUNKS countdown;
                 amp_tgt will be stepped toward 0 linearly over those chunks.
              b) Assign the new frequency to the BACK slot; set amp_cur=0 (silent
                 start, so the phase accumulator starts at a known value while
                 amplitude is zero -- no audible discontinuity) and amp_tgt to
                 the target amplitude (ATTACK begins immediately).
              c) Swap front/back banks for this voice×mode.
          Both old and new slots are rendered simultaneously during the transition
          -> overlapping envelopes, no amplitude gap.
        Voices no longer present: release their front slots.
        amp_cur is passed in so we can force back slot to silence before attack.
        """
        n_voices = min(len(voices), MAX_VOICES)

        # ── Phase 1: process new voice data (set targets, initiate swaps) ──────
        for v in range(MAX_VOICES):
            for m in range(MAX_MODES_PER_OBJ):
                fslot = self._front_slot(v, m)
                bslot = self._back_slot(v, m)

                if v >= n_voices:
                    # Voice no longer present: start releasing front slot if needed.
                    if self._release_cnt[fslot] == 0:
                        if amp_cur[fslot] >= 0.01:
                            self._release_cnt[fslot] = FADE_CHUNKS
                            self._release_amp0[fslot] = float(amp_cur[fslot])
                        else:
                            self.amp_tgt[fslot] = 0.0  # already silent, just zero tgt
                    # If already releasing, Phase 2 manages amp_tgt via countdown.
                    # Silence the back slot directly (it should already be quiet).
                    self.amp_tgt[bslot] = 0.0
                    self._freq_prev[v, m] = 0.0
                    continue

                voice = voices[v]
                freq_new = float(voice['freqs'][m]) if m < len(voice['freqs']) else 0.0
                amp_new  = float(voice['amps'][m])  if m < len(voice['amps'])  else 0.0
                pan_new  = voice['pan']

                freq_old = self._freq_prev[v, m]
                freq_changed = abs(freq_new - freq_old) > 0.01

                if freq_changed:
                    # ── OVERLAPPING CROSSFADE ────────────────────────────────
                    # a) Put front slot into release (if not already releasing).
                    if self._release_cnt[fslot] == 0:
                        self._release_cnt[fslot] = FADE_CHUNKS
                        self._release_amp0[fslot] = float(amp_cur[fslot])
                    # amp_tgt for the releasing slot is managed by Phase 2 below.

                    # b) Attack on back slot with the new frequency.
                    #    amp_cur[bslot]=0 ensures render starts from silence so the
                    #    phase accumulator value is inconsequential (inaudible start).
                    amp_cur[bslot] = 0.0
                    self.freq_slots[bslot] = freq_new
                    self.amp_tgt[bslot] = amp_new if freq_new > 0 else 0.0
                    self.pan_tgt[bslot] = pan_new
                    self._release_cnt[bslot] = 0   # new front is not releasing

                    # c) Swap banks.
                    self._bank[v, m] = 1 - self._bank[v, m]
                    self._freq_prev[v, m] = freq_new

                else:
                    # ── STABLE: smooth amp/pan update on front slot ──────────
                    self.freq_slots[fslot] = freq_new
                    self.amp_tgt[fslot] = amp_new if freq_new > 0 else 0.0
                    self.pan_tgt[fslot] = pan_new
                    self._freq_prev[v, m] = freq_new

        # ── Phase 2: advance all active release countdowns ────────────────────
        # Iterate over every slot to decrement countdowns and compute the next
        # amp_tgt step.  This is done AFTER the voice-assignment loop so that a
        # slot put into release this cycle gets its FIRST countdown step applied
        # (countdown goes from FADE_CHUNKS to FADE_CHUNKS-1 in this same call,
        # i.e. the very first amp_tgt value already reflects one chunk of decay).
        for slot in range(1, TOTAL_SLOTS + 1):
            cnt = self._release_cnt[slot]
            if cnt <= 0:
                continue
            new_cnt = cnt - 1
            self._release_cnt[slot] = new_cnt
            # Linear step: amp_tgt = amp0 * remaining_fraction
            self.amp_tgt[slot] = self._release_amp0[slot] * new_cnt / FADE_CHUNKS
            # Once countdown hits 0, amp_tgt is 0; render_chunk_laplacian glides
            # amp_cur to 0 within the next chunk.  Silence the freq when done.
            if new_cnt == 0 and amp_cur[slot] < 0.01:
                self.freq_slots[slot] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# PIANO HELPERS  (identical to gol_life_synth.py)
# ──────────────────────────────────────────────────────────────────────────────

def _make_piano(base, piano_top, W):
    end = base + 24
    white_notes = [m for m in range(base, end + 1) if m % 12 in WHITE_PC]
    wkw = W / len(white_notes)
    bkw, bkh = wkw * 0.62, int(PIANO_H * 0.62)
    wk, wi = [], {}
    for i, m in enumerate(white_notes):
        x0, x1 = round(i * wkw), round((i + 1) * wkw)
        wk.append((pygame.Rect(x0, piano_top, x1 - x0, PIANO_H), m))
        wi[m] = i
    bk = []
    for m in range(base, end + 1):
        if m % 12 in BLACK_PC and (m - 1) in wi:
            cx = (wi[m - 1] + 1) * wkw
            bk.append((pygame.Rect(round(cx - bkw / 2), piano_top, round(bkw), bkh), m))
    return wk, bk


def pattern_preview_surf(cells, pw, ph):
    surf = pygame.Surface((pw, ph))
    surf.fill(C_BG)
    if not cells:
        return surf
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    rh = max(rows) - min(rows) + 1
    cw = max(cols) - min(cols) + 1
    cs = max(1, min(pw // max(cw, 1), ph // max(rh, 1)))
    ox = (pw - cs * cw) // 2
    oy = (ph - cs * rh) // 2
    r0, c0 = min(rows), min(cols)
    for r, c in cells:
        x = ox + (c - c0) * cs
        y = oy + (r - r0) * cs
        pygame.draw.rect(surf, C_ACCENT, (x, y, max(1, cs - 1), max(1, cs - 1)))
    return surf


# ──────────────────────────────────────────────────────────────────────────────
# SESSION RECORDING  (opt-in: env CASYNTH_RECORD=1)
# ──────────────────────────────────────────────────────────────────────────────
# Offline renders of a scene sound clean, yet the live demo clicks.  The live
# audio path differs from a clean per-chunk render: feed_audio() runs every frame
# (~60 fps) but only renders a chunk when the mixer queue is empty, and a frame
# hitch (analyse() runs eigvalsh on every object every frame) can starve the
# queue -> playback gap -> click.  To diagnose we capture, during a real user
# session: (1) the exact audio chunks the engine produced (gapless WAV = engine
# output), (2) a per-frame log of timing + queue underruns (clicks not in the WAV
# but flagged here == playback-starvation clicks), (3) the field history + carrier
# note so the session can be replayed offline to reproduce.

def _dump_session(rec, prefix="_session"):
    import time
    from scipy.io import wavfile
    base = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    if rec['chunks']:
        audio = np.concatenate(rec['chunks'], axis=0)
        wavfile.write(f"{base}.wav", SR, audio)
    frames = np.array(rec['frames'], dtype=float) if rec['frames'] else np.zeros((0, 5))
    if rec['steps']:
        gens  = np.array([s[0] for s in rec['steps']])
        grids = np.stack([s[1] for s in rec['steps']])
        notes = np.array([s[2] for s in rec['steps']])
    else:
        gens, grids, notes = (np.zeros(0),
                              np.zeros((0, GRID_H, GRID_W), np.uint8), np.zeros(0))
    np.savez_compressed(f"{base}.npz", frames=frames, gens=gens, grids=grids,
                        notes=notes, sr=SR, chunk_s=CHUNK_S, step_hz=STEP_HZ,
                        max_voices=MAX_VOICES, underruns=rec['underruns'])
    # frames columns: [dt_ms, gen, n_voices, n_rendered, underrun]
    n_hitch = int((frames[:, 0] > CHUNK_S * 1000).sum()) if len(frames) else 0
    print(f"[session saved] {base}.wav ({len(rec['chunks'])} chunks) + {base}.npz "
          f"({len(rec['steps'])} steps)  underruns={rec['underruns']}  "
          f"frames>{int(CHUNK_S*1000)}ms={n_hitch}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    pygame.init()

    # ── audio output: sounddevice callback stream pulling from a ring buffer ──
    # The synth (main thread) renders chunks ahead into audio_q; the PortAudio
    # callback (audio thread) pulls samples at the hardware rate.  This decouples
    # audio from the 60 fps frame loop: a frame hitch only shrinks the buffer, it
    # never gaps playback.  amp_cur/phase/pool are touched only by the main thread;
    # the callback reads finished int16 chunks from the thread-safe queue.
    audio_q = _queue.Queue()
    _resid  = {'buf': None, 'pos': 0}          # partial chunk across callbacks
    _ur     = {'n': 0, 'prev': 0}              # underrun counter (audio thread)
    vol_box = [VOL_DEFAULT]                     # volume, read by the audio thread

    def _audio_cb(outdata, frames, time_info, status):
        filled = 0
        v = vol_box[0]
        while filled < frames:
            if _resid['buf'] is None:
                try:
                    _resid['buf'] = audio_q.get_nowait()
                    _resid['pos'] = 0
                except _queue.Empty:
                    outdata[filled:] = 0       # underrun -> silence (no click)
                    _ur['n'] += 1
                    return
            buf = _resid['buf']
            pos = _resid['pos']
            take = min(frames - filled, len(buf) - pos)
            seg = buf[pos:pos + take]
            outdata[filled:filled + take] = seg if v >= 0.999 else (seg * v).astype(np.int16)
            filled += take
            pos += take
            if pos >= len(buf):
                _resid['buf'] = None
            else:
                _resid['pos'] = pos

    audio_ok = True
    stream = None
    if sd is None:
        audio_ok = False
        print("[audio disabled: sounddevice not installed] - visuals will still run")
    else:
        try:
            # pre-roll silence so the callback never starves before the first
            # feed_audio() tops up the buffer (otherwise 1-2 startup underruns).
            silent = np.zeros((int(CHUNK_S * SR), 2), np.int16)
            for _ in range(AUDIO_LOOKAHEAD_CHUNKS):
                audio_q.put(silent.copy())
            stream = sd.OutputStream(samplerate=SR, channels=2, dtype='int16',
                                     latency='low', callback=_audio_cb)
            stream.start()
            print(f"[audio] sounddevice out latency={stream.latency*1000:.0f}ms "
                  f"+ {AUDIO_LOOKAHEAD_CHUNKS} chunk look-ahead "
                  f"({AUDIO_LOOKAHEAD_CHUNKS*CHUNK_S*1000:.0f}ms)")
        except Exception as e:
            audio_ok = False
            print(f"[audio disabled: {e}] - visuals will still run")

    W = GRID_W * CELL
    H = GRID_H * CELL + TOOLBAR_H + PIANO_H

    screen = pygame.display.set_mode((W + SIDEBAR_W, H))
    pygame.display.set_caption("CASynth — Laplacian timbre")
    font  = pygame.font.SysFont("consolas,menlo,monospace", 16)
    small = pygame.font.SysFont("consolas,menlo,monospace", 13)
    clock = pygame.time.Clock()

    grid = np.zeros((GRID_H, GRID_W), np.uint8)

    # Flat slot pool (audio engine state)
    pool = SlotPool()
    sz = TOTAL_SLOTS + 1
    phase   = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)

    # optional session recording (env CASYNTH_RECORD=1)
    rec = None
    if os.environ.get('CASYNTH_RECORD'):
        rec = dict(chunks=[], frames=[], steps=[], underruns=0)
        print("[CASYNTH_RECORD on] capturing audio + field; quit (Esc) to save")

    state = dict(
        run=False, hz=STEP_HZ, gen=0, acc=0.0,
        note=NOTE_DEFAULT, vol=VOL_DEFAULT,
        kb_base=NOTE_DEFAULT,
        sidebar_open=True,
    )

    # toolbar layout (identical to gol_life_synth.py)
    by = GRID_H * CELL + 8
    info_y = GRID_H * CELL + 58
    defs = [("play", None, 96), ("step", "Step", 70), ("random", "Random", 96),
            ("clear", "Clear", 78), ("slower", "-", 40), ("faster", "+", 40)]
    buttons, bx = [], 12
    for bid, label, bw in defs:
        buttons.append(dict(id=bid, label=label, rect=pygame.Rect(bx, by, bw, 40)))
        bx += bw + 8
    legend_x = 12
    vol_track = pygame.Rect(W - VOL_W - 24, info_y + 14, VOL_W, 8)
    _pat_btn = pygame.Rect(bx + 8, by, 128, 40)

    # sidebar items
    _sb_items = []
    _iy = 32
    for _cat, _pats in PATTERNS:
        _iy += _SB_HDR_H + 4
        for _pname, _pcells in _pats:
            _sb_items.append({
                'rect':  pygame.Rect(W + 4, _iy, SIDEBAR_W - 8, _SB_ITEM_H),
                'cells': _pcells,
                'name':  _pname,
                'prev':  pattern_preview_surf(_pcells, _SB_PREV_W, _SB_PREV_H),
            })
            _iy += _SB_ITEM_H + 2
    _sb_content_h = _iy
    _sb_scroll = 0
    _sb_scroll_min = min(0, H - _sb_content_h)

    piano_top = GRID_H * CELL + TOOLBAR_H
    white_keys, black_keys = _make_piano(state['kb_base'], piano_top, W)

    paint = None
    dragging_vol = False
    drag = {'active': False, 'cells': [], 'name': '', 'snap': None}
    _ghost = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
    _ghost.fill((111, 208, 224, 110))

    def cell_at(mx, my):
        if 0 <= my < GRID_H * CELL and 0 <= mx < GRID_W * CELL:
            return my // CELL, mx // CELL
        return None

    def f0():
        return midi_to_freq(state['note'])

    def set_vol(mx):
        state['vol'] = float(np.clip((mx - vol_track.left) / vol_track.width, 0, 1))
        vol_box[0] = state['vol']   # picked up by the audio callback

    def do(bid):
        nonlocal grid
        if bid == "play":    state['run'] = not state['run']
        elif bid == "step":  grid = step(grid); state['gen'] += 1
        elif bid == "random":
            grid[:] = (np.random.random((GRID_H, GRID_W)) < RANDOM_DENSITY).astype(np.uint8)
        elif bid == "clear": grid[:] = 0; state['gen'] = 0
        elif bid == "slower": state['hz'] = max(1.0, state['hz'] - 1.0)
        elif bid == "faster": state['hz'] = min(30.0, state['hz'] + 1.0)

    def hit_piano(pos):
        for rect, m in black_keys:
            if rect.collidepoint(pos):
                return m
        for rect, m in white_keys:
            if rect.collidepoint(pos):
                return m
        return None

    def feed_audio(voices):
        if not audio_ok:
            return 0, 0
        # Top the ring buffer up to AUDIO_LOOKAHEAD_CHUNKS of look-ahead.
        # pool.update is advanced once PER CHUNK rendered (not per frame), so the
        # cross-fade (FADE_CHUNKS) now counts chunks and is no longer coupled to
        # the frame rate.
        n_rendered = 0
        while audio_q.qsize() < AUDIO_LOOKAHEAD_CHUNKS:
            pool.update(voices, amp_cur)
            buf = render_chunk_laplacian(phase, amp_cur, pan_cur,
                                         pool.amp_tgt, pool.pan_tgt,
                                         pool.freq_slots, 2)
            audio_q.put(buf)
            n_rendered += 1
            if rec is not None:
                rec['chunks'].append(buf)
        ur_delta = _ur['n'] - _ur['prev']      # underruns since last frame
        _ur['prev'] = _ur['n']
        if rec is not None:
            rec['underruns'] = _ur['n']
        return n_rendered, ur_delta

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in _KB_PIANO:
                    state['note'] = state['kb_base'] + _KB_PIANO[e.key]
                elif e.key == pygame.K_z:
                    state['kb_base'] = max(KB_BASE_MIN, state['kb_base'] - 12)
                    white_keys, black_keys = _make_piano(state['kb_base'], piano_top, W)
                elif e.key == pygame.K_x:
                    state['kb_base'] = min(KB_BASE_MAX, state['kb_base'] + 12)
                    white_keys, black_keys = _make_piano(state['kb_base'], piano_top, W)
                elif e.key == pygame.K_ESCAPE:
                    if drag['active']:
                        drag.update(active=False, snap=None, cells=[], name='')
                    else:
                        running = False
                elif e.key == pygame.K_SPACE:  do("play")
                elif e.key == pygame.K_r:       do("random")
                elif e.key == pygame.K_c:       do("clear")
                elif e.key == pygame.K_UP:      do("faster")
                elif e.key == pygame.K_DOWN:    do("slower")

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if state['sidebar_open'] and e.pos[0] >= W and e.button == 1:
                    for item in _sb_items:
                        if item['rect'].move(0, _sb_scroll).collidepoint(e.pos):
                            drag.update(active=True, cells=item['cells'],
                                        name=item['name'], snap=None)
                            break
                elif not drag['active']:
                    rc = cell_at(*e.pos)
                    if rc is not None and e.button in (1, 3):
                        paint = 1 if e.button == 1 else 0
                        grid[rc] = paint
                    elif e.pos[1] >= piano_top:
                        m = hit_piano(e.pos)
                        if m is not None:
                            state['note'] = m
                    else:
                        if _pat_btn.collidepoint(e.pos):
                            state['sidebar_open'] = not state['sidebar_open']
                            nw = W + (SIDEBAR_W if state['sidebar_open'] else 0)
                            screen = pygame.display.set_mode((nw, H))
                        elif vol_track.collidepoint(e.pos):
                            dragging_vol = True
                            set_vol(e.pos[0])
                        else:
                            for b in buttons:
                                if b['rect'].collidepoint(e.pos):
                                    do(b['id']); break

            elif e.type == pygame.MOUSEBUTTONUP:
                if drag['active']:
                    if drag['snap'] is not None:
                        r0, c0 = drag['snap']
                        for dr, dc in drag['cells']:
                            grid[(r0 + dr) % GRID_H, (c0 + dc) % GRID_W] = 1
                    drag.update(active=False, snap=None, cells=[], name='')
                paint = None
                dragging_vol = False

            elif e.type == pygame.MOUSEMOTION:
                if drag['active']:
                    drag['snap'] = cell_at(*e.pos)
                elif paint is not None:
                    rc = cell_at(*e.pos)
                    if rc is not None:
                        grid[rc] = paint
                elif dragging_vol:
                    set_vol(e.pos[0])

            elif e.type == pygame.MOUSEWHEEL:
                if state['sidebar_open'] and pygame.mouse.get_pos()[0] >= W:
                    _sb_scroll = max(_sb_scroll_min, min(0, _sb_scroll + e.y * 20))

        # GOL advance
        if state['run']:
            interval = 1.0 / state['hz']
            state['acc'] += dt
            steps = 0
            while state['acc'] >= interval and steps < 4:
                state['acc'] -= interval
                grid = step(grid)
                state['gen'] += 1
                steps += 1
                if rec is not None:
                    rec['steps'].append((state['gen'], grid.copy(), state['note']))

        labels, voices, color = analyse(grid, f0())
        n_rendered, ur_delta = feed_audio(voices)
        if rec is not None:
            rec['frames'].append((round(dt * 1000.0, 2), state['gen'],
                                  len(voices), n_rendered, ur_delta))

        # ── draw field ──────────────────────────────────────────────────────
        screen.fill(C_BG)
        for x in range(GRID_W + 1):
            pygame.draw.line(screen, C_GRID, (x * CELL, 0), (x * CELL, GRID_H * CELL))
        for y in range(GRID_H + 1):
            pygame.draw.line(screen, C_GRID, (0, y * CELL), (GRID_W * CELL, y * CELL))
        ys, xs = np.where(grid == 1)
        for (r, c) in zip(ys.tolist(), xs.tolist()):
            col = color.get(labels[r, c], C_DIM)
            pygame.draw.rect(screen, col,
                             (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2),
                             border_radius=4)

        # ── drag ghost ──────────────────────────────────────────────────────
        if drag['active'] and drag['snap'] is not None:
            r0, c0 = drag['snap']
            for dr, dc in drag['cells']:
                screen.blit(_ghost, ((c0 + dc) % GRID_W * CELL + 1,
                                     (r0 + dr) % GRID_H * CELL + 1))

        # ── toolbar ─────────────────────────────────────────────────────────
        pygame.draw.rect(screen, C_PANEL, (0, GRID_H * CELL, W, TOOLBAR_H))
        pygame.draw.line(screen, C_EDGE, (0, GRID_H * CELL), (W, GRID_H * CELL))
        mouse = pygame.mouse.get_pos()
        for b in buttons:
            hot = b['rect'].collidepoint(mouse)
            pygame.draw.rect(screen, C_BTN_HOT if hot else C_BTN, b['rect'], border_radius=6)
            label = b['label'] or ("Pause" if state['run'] else "Play")
            txt = font.render(label, True, C_ACCENT if b['id'] == 'play' else C_TXT)
            screen.blit(txt, txt.get_rect(center=b['rect'].center))
        hot_p = _pat_btn.collidepoint(mouse)
        pygame.draw.rect(screen, C_BTN_HOT if hot_p else C_BTN, _pat_btn, border_radius=6)
        plbl = small.render("Laplace | " + ("◀ Lib" if state['sidebar_open'] else "Lib ▶"),
                            True, C_ACCENT)
        screen.blit(plbl, plbl.get_rect(center=_pat_btn.center))

        # info row: legend
        lw, lh = 110, 10
        lx, ly = legend_x, info_y + 5
        for i in range(lw):
            pygame.draw.line(screen, hsv((i / (lw - 1)) * 0.72, 0.95),
                             (lx + i, ly), (lx + i, ly + lh))
        screen.blit(small.render("obj 1", True, C_DIM), (lx, ly + lh + 2))
        hi = small.render(f"obj {MAX_VOICES}", True, C_DIM)
        screen.blit(hi, (lx + lw - hi.get_width(), ly + lh + 2))

        # info row: status
        sx = lx + lw + 16
        st = (f"{note_name(state['note'])} {f0():.0f}Hz  "
              f"gen {state['gen']:>4}  {state['hz']:.0f}/s  "
              f"obj {len(voices)}")
        screen.blit(font.render(st, True, C_TXT), (sx, info_y + 1))
        mode = "RUNNING" if state['run'] else "PAUSED"
        mode_surf = font.render(mode, True, C_ACCENT if state['run'] else C_DIM)
        screen.blit(mode_surf, (sx, info_y + 19))
        kbd_surf = small.render(f"kbd:{note_name(state['kb_base'])}", True, C_DIM)
        screen.blit(kbd_surf, (sx + mode_surf.get_width() + 10, info_y + 21))

        # info row: volume
        screen.blit(small.render(f"vol {int(state['vol'] * 100)}%", True, C_DIM),
                    (vol_track.left, info_y + 1))
        pygame.draw.rect(screen, C_BTN, vol_track, border_radius=4)
        fw = int(vol_track.width * state['vol'])
        pygame.draw.rect(screen, C_ACCENT,
                         (vol_track.left, vol_track.top, fw, vol_track.height),
                         border_radius=4)
        pygame.draw.circle(screen, C_TXT, (vol_track.left + fw, vol_track.centery), 6)
        if not audio_ok:
            screen.blit(small.render("audio disabled", True, C_DIM), (W - 110, 6))

        # ── piano ───────────────────────────────────────────────────────────
        for rect, m in white_keys:
            on = (m == state['note'])
            pygame.draw.rect(screen, C_WHITE_ON if on else C_WHITE, rect)
            pygame.draw.rect(screen, C_EDGE, rect, 1)
            if m % 12 == 0:
                lbl = small.render(note_name(m), True, (70, 74, 82))
                screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.bottom - 18))
        for rect, m in black_keys:
            on = (m == state['note'])
            pygame.draw.rect(screen, C_BLACK_ON if on else C_BLACK, rect, border_radius=3)

        # ── sidebar ─────────────────────────────────────────────────────────
        if state['sidebar_open']:
            pygame.draw.rect(screen, C_PANEL, (W, 0, SIDEBAR_W, H))
            pygame.draw.line(screen, C_EDGE, (W, 0), (W, H))
            ttl = font.render("Patterns", True, C_TXT)
            screen.blit(ttl, (W + (SIDEBAR_W - ttl.get_width()) // 2, 6))
            pygame.draw.line(screen, C_EDGE, (W + 2, 28), (W + SIDEBAR_W - 2, 28))
            screen.set_clip(pygame.Rect(W, 30, SIDEBAR_W, H - 30))
            item_idx = 0
            for _cat, pats in PATTERNS:
                it0 = _sb_items[item_idx]
                cat_y = it0['rect'].top - _SB_HDR_H - 4 + _sb_scroll
                clbl = small.render(_cat.upper(), True, C_DIM)
                screen.blit(clbl, (W + 6, cat_y + (_SB_HDR_H - clbl.get_height()) // 2))
                for _ in pats:
                    it = _sb_items[item_idx]
                    item_idx += 1
                    sr = it['rect'].move(0, _sb_scroll)
                    hot_it = sr.collidepoint(mouse) and not drag['active']
                    pygame.draw.rect(screen, C_BTN_HOT if hot_it else C_BTN,
                                     sr, border_radius=4)
                    screen.blit(it['prev'],
                                (sr.left + 2, sr.top + (sr.height - _SB_PREV_H) // 2))
                    ntxt = small.render(it['name'], True, C_TXT)
                    screen.blit(ntxt, (sr.left + _SB_PREV_W + 6,
                                       sr.centery - ntxt.get_height() // 2))
            screen.set_clip(None)
            if _sb_scroll_min < 0:
                vis = H - 30
                thumb_h = max(18, vis * vis // _sb_content_h)
                thumb_y = 30 + int(-_sb_scroll / -_sb_scroll_min * (vis - thumb_h))
                pygame.draw.rect(screen, C_DIM,
                                 (W + SIDEBAR_W - 4, thumb_y, 3, thumb_h), border_radius=2)

        if drag['active']:
            mx, my = pygame.mouse.get_pos()
            dlbl = small.render(drag['name'], True, C_ACCENT)
            screen.blit(dlbl, (mx + 14, my - dlbl.get_height() // 2))

        pygame.display.flip()

    if rec is not None:
        _dump_session(rec)

    if stream is not None:
        stream.stop()
        stream.close()
    pygame.quit()


if __name__ == '__main__':
    main()
