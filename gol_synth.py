#!/usr/bin/env python3
"""
gol_synth.py - CASynth: multi-engine GOL synth (engine selector).

Same UI and GOL simulation as gol_life_synth_laplacian.py, but the mapping from
field objects to audio is SELECTABLE at runtime from the shared engine registry
(casynth_core.ENGINES): the five mappings that lived on mapping_bench.py --

    FFT / Walsh / Random  : harmonic spectrum (f0*k), amplitudes = transform of shape
    Laplace               : inharmonic resonant modes (sqrt(lambda)), metallic / bell
    Granulo               : scale distribution of the shape -> harmonic amplitudes

A row of engine TABS in the toolbar switches the active engine; the timbre-knob
panel rebuilds to show only THAT engine's attributes (Laplace: part/spread/alpha;
the others: just part).  Per-engine attribute values are remembered across
switches (state['engine_params']) so returning to an engine restores its sound.
The RELEASE knob is synth-wide (mode-tail length), not an engine attribute.

Each connected object in the GOL field produces a VOICE whose spectrum is computed
by the active engine.  The carrier note is set via the on-screen piano; for every
engine the lowest partial is anchored to the carrier f0.

Cross-fade on topology change (active + tail slot pool)
-------------------------------------------------------
When GOL steps change an object's shape the set of Laplacian modes changes
abruptly.  Without smoothing this causes audible clicks / "beeps".  Solution:
  - Each voice×mode has ONE active slot plus a shared pool of release-only TAIL
    slots (see SlotPool).
  - On a frequency change: the old mode is MOVED into a free tail slot that rings
    down over the release-tail length (live knob, RELEASE_MS_*), continuing the
    exact same waveform (same phase + amplitude) so there is no seam; the active
    slot restarts on the new frequency from amplitude 0 (fast attack).  Old (tail)
    and new (active) render in parallel -> overlapping envelopes, no gap ("struck
    plate": old resonance still ringing while the new one rises).  Longer release
    -> tails of many past topologies overlap into a pad.
  - Phase accumulation is continuous -- no phase reset.
  - Mode frequencies are NOT glided -- they are discrete resonances; gliding
    would smear the inharmonic character that makes the timbre metallic.

CONTROLS  (identical to gol_life_synth.py)
  Mouse on field : left-drag draw, right-drag erase
  Piano (bottom) : click to set carrier note (latched)
  Engine tabs    : click to switch the active sound engine (toolbar row)
  Timbre knobs   : current engine's attributes + synth-wide release (live)
  Volume slider  : drag (top-right toolbar) -- PRE-clip, so lowering it is the
                   manual headroom control; watch the level/clip meter above it
                   (turns red "CLIP +X dB" when the sum overshoots the ceiling)
  Space ......... play / pause
  S ............. step (key not mapped -- use button)
  R / C ......... random / clear
  Up / Down ..... faster / slower
  Esc ........... quit

RUN
    pip install pygame-ce numpy scipy sounddevice
    python gol_synth.py
    # headless:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python gol_synth.py
"""

import os
import time
import colorsys
import queue as _queue
import numpy as np
import pygame
try:
    import sounddevice as sd
except Exception:          # optional; audio is just disabled if unavailable
    sd = None
from scipy import ndimage
from casynth_core import extract, ENGINES, ENGINE_BY_ID
from patterns import PATTERNS

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  -- all tunable constants here
# ──────────────────────────────────────────────────────────────────────────────
GRID_W, GRID_H = 52, 30
CELL = 20
# Toolbar: main button row (40px) + note-div row (18px) + status row (~38px) +
# engine-knob panel (Laplace 6×22px, right col: ADSR 4×22px + vol/level) +
# engine-tab strip (24px) + gaps.  Right col ends at by+178; tabs at -24 from
# bottom → 178 + 20 gap + 24 tab = 222 → round up to 230.
TOOLBAR_H = 230
PIANO_H = 96
FPS = 60

SR = 44100
RANDOM_DENSITY = 0.28

# BPM / note-division tempo control  (replaces the old STEP_HZ knob)
BPM_DEFAULT = 120
BPM_MIN     = 40
BPM_MAX     = 240
# Each entry: (UI label, beats-per-step)  where 1 beat = one quarter note.
# Triplet = 2/3 of the straight note value (3 notes in the space of 2).
NOTE_DIVS = [
    ('1/1',   4.0),
    ('1/2',   2.0),
    ('1/4',   1.0),
    ('1/8',   0.5),
    ('1/16',  0.25),
    ('1/2T',  4.0 / 3),
    ('1/4T',  2.0 / 3),
    ('1/8T',  1.0 / 3),
    ('1/16T', 1.0 / 6),
]
DIV_DEFAULT = 2           # index into NOTE_DIVS → 1/4 (quarter note)
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
# Level/clip meter: peak-hold decay per frame (visual ballistics only).  Volume
# is PRE-clip (see render_chunk_laplacian), so lowering it is the manual headroom
# control; the meter shows the pre-clip peak vs the 0 dBFS ceiling.
METER_DECAY = 0.90

# Laplacian mode parameters
# MAX_MODES_PER_OBJ is now the CEILING (slot capacity) -- the live "partials" knob
# (state['n_partials'], 1..MAX_MODES_PER_OBJ) picks how many are actually sounded;
# higher modes are silenced and ring out as tails (SlotPool guards m < len(freqs)).
# All slot-pool figures below (N_ACTIVE/N_TAIL/TOTAL_SLOTS/TAIL_RESERVE) scale with
# this; the empirical steal/click counts in their comments were measured at the
# original 8-mode layout and scale with the same ratios.
MAX_MODES_PER_OBJ = 20  # max Laplacian modes per object (slot capacity / knob max)
PATCH_SIZE = 8           # extraction window for map_laplacian

# Live timbre knobs (on-screen sliders; map_laplacian args, decisions.md 2026-06-16).
# Defaults reproduce the original Laplacian timbre exactly.
SPREAD_DEFAULT = 0.0     # 0 = n lowest modes (dark); 1 = decimate across spectrum (bright)
ALPHA_DEFAULT  = 1.0     # 1/i**alpha rolloff: 0 = flat/bright, >1 = steeper/dark
ALPHA_MIN, ALPHA_MAX = 0.0, 2.0
# Number of partials sounded per object -- a LIVE knob (laplacian_explainer.html
# tweak the researcher demoed; range/default match it).  Fewer = thinner/cleaner,
# more = richer.  Integer, clamped to [N_PARTIALS_MIN, MAX_MODES_PER_OBJ].
N_PARTIALS_DEFAULT = 12
N_PARTIALS_MIN     = 1

# Audio engine slots (slot 0 unused; slots 1..TOTAL_SLOTS used).
#   ACTIVE slots 1..N_ACTIVE: one per voice×mode, the CURRENTLY sounding mode.
#     active slot for voice v, mode m: 1 + v * MAX_MODES_PER_OBJ + m
#   TAIL slots N_ACTIVE+1..TOTAL_SLOTS: a pool of release-only slots that ring
#     out replaced modes (the pad tails).  See SlotPool for why a tail POOL
#     replaced the old front/back 2-bank scheme (2 banks slammed a still-ringing
#     tail to 0 on reuse -> boundary click with long release; DEV-2).
N_ACTIVE = MAX_VOICES * MAX_MODES_PER_OBJ         # 480 (24 voices x 20 modes)
# Tail pool sized 2x the active set: a long release (up to RELEASE_MS_MAX) over a
# fast-changing object stacks many overlapping tails; 192 exhausted on the user's
# single-galaxy snapshot (109 steals -> residual clicks, loudest stolen amp 0.61).
# 384 gives 0 steals there (boundary click peak 1065 -> 94, the short-release
# floor) at negligible render cost (~+5% on the slot loop, slots skip when silent).
# Dense many-object fields with max release can still exhaust it; then the QUIETEST
# tail is stolen (graceful, smallest click).
N_TAIL   = MAX_VOICES * MAX_MODES_PER_OBJ * 4      # 1920 release-only tail slots
TOTAL_SLOTS = N_ACTIVE + N_TAIL                   # 2400
TAIL_MIN_AMP = 0.005     # below this an old mode is too quiet to be worth a tail
# Graceful tail eviction: a dense field with long release wants more overlapping
# tails than any feasible slot count (24 voices x 8 modes x ~24 tails @ 4 s ~= 4600).
# So each update, BEFORE this cycle's spawns, the QUIETEST excess tails are
# fast-faded to zero over FAST_EVICT_CHUNKS (a quick but smooth release -> no
# click), keeping TAIL_RESERVE slots free.  TAIL_RESERVE covers the largest single
# GOL-step spawn burst (every active mode changing at once = N_ACTIVE) so a burst
# never has to steal a loud ringing tail (a stolen loud tail = boundary click;
# measured on a 12-voice snapshot: 2571 steals of amp~0.57 -> clicks).  Under load
# this shortens the pad (oldest/quietest tails dropped first) but stays click-free;
# sparse fields still get the full long pad.
FAST_EVICT_MS = 90
FAST_EVICT_CHUNKS = max(1, round(FAST_EVICT_MS / 1000.0 / CHUNK_S))   # 1 chunk
# Reserve = 2x the active set.  The max single-step spawn burst is N_ACTIVE, but at
# full 24-voice churn overlapping bursts drain the reserve faster than a 1-chunk
# fast-fade refills it; 2x empirically gives 0 steals on the 24-voice random-field
# snapshot (1x left ~10k steals).  Pool 768 suffices -- bigger pools add nothing.
TAIL_RESERVE  = MAX_VOICES * MAX_MODES_PER_OBJ * 2  # 960

# Mode-tail RELEASE duration -- a LIVE knob (decisions.md 2026-06-16 "пэды").
# When a mode's frequency changes (topology event) or its object dies, the OLD
# frequency's slot fades out over RELEASE_MS.  The new mode's onset is shaped by
# the ATTACK/DECAY/SUSTAIN knobs below (was a fixed one-chunk attack); only the
# tail length is user-controlled here.  Short -> percussive "struck plate"; long
# (seconds) -> tails of past topologies overlap into a continuous pad.
#   release length in chunks = max(1, round(RELEASE_MS / 1000 / CHUNK_S))
# computed live from state['release_ms'] each feed cycle (NOT a fixed constant),
# so the knob retunes without restart.
# NOTE (2-bank limit): each voice×mode owns only two physical slots (front/back).
# A tail still ringing when the SAME voice×mode changes frequency again is reused
# for the next attack, cutting that tail short.  So very long tails only fully
# ring out between events (slow / paused fields); under continuous fast stepping
# they are clipped at the next topology change.  A full overlapping pad of many
# stacked tails needs a dedicated tail-slot pool (see questions.md follow-up).
RELEASE_MS_DEFAULT = 50.0
RELEASE_MS_MIN     = 20.0
RELEASE_MS_MAX     = 4000.0

# Per-mode ATTACK/DECAY/SUSTAIN -- LIVE knobs forming a classic ADSR envelope
# keyed to a mode's LIFE (there is no per-note on/off: a "note-on" = a mode onset
# = a new frequency appearing on an active slot at a topology event; "note-off" =
# that mode dying / changing frequency -> the existing RELEASE tail).  The
# envelope is a normalised 0..1 multiplier on the engine's per-mode amplitude
# (amp_new from the shape mapping), advanced one step per audio chunk in SlotPool:
#   attack : level 0 -> 1            over ATTACK_MS   (onset rise to peak)
#   decay  : level 1 -> SUSTAIN      over DECAY_MS    (fall to the held level)
#   sustain: level == SUSTAIN        held while the mode stays alive (generation
#                                    alive / topology stable)
#   release: handled by the tail pool (RELEASE_MS) on mode death/frequency change
# chunk counts: max(1, round(MS / 1000 / CHUNK_S)); SUSTAIN is a level, not a time.
# Defaults reproduce the original sound EXACTLY: ATTACK->1 chunk (the old fixed
# one-chunk onset), SUSTAIN=1 -> level stays 1 for the mode's whole life -> amp_tgt
# == amp_new as before (DECAY then has no effect since peak == sustain level).
# A pure attack->release pluck = SUSTAIN 0 (+ short DECAY); holding the full level
# for the whole generation = SUSTAIN 1.
ATTACK_MS_DEFAULT  = 1.0
ATTACK_MS_MIN      = 0.0
ATTACK_MS_MAX      = 2000.0
DECAY_MS_DEFAULT   = 0.0
DECAY_MS_MIN       = 0.0
DECAY_MS_MAX       = 4000.0
SUSTAIN_DEFAULT    = 1.0
SUSTAIN_MIN        = 0.0
SUSTAIN_MAX        = 1.0

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


def analyse(grid, f0, engine_id, params):
    """Segment the field into connected objects and compute voices via the
    SELECTED engine.

    engine_id : key into ENGINE_BY_ID -- picks the map_* function.
    params    : dict {arg: value} of that engine's live attributes (e.g.
                {'n':12,'spread':0.0,'alpha':1.0} for Laplace, {'n':16} for the
                harmonic engines).  'n' (partial count) is clamped to
                MAX_MODES_PER_OBJ; the engine silences higher slots (they ring
                out as tails).  Forwarded to the map_* as keyword args, so the
                same audio pipeline serves every engine -- harmonic mappings just
                return fixed f0*k freqs (no topology tails, only amp glide).

    Returns:
        labels  : labelled grid (0 = background)
        voices  : list of dicts with keys 'freqs', 'amps', 'pan', 'label_idx'
        color   : dict  label -> RGB colour tuple
    """
    labels, n = ndimage.label(grid, structure=_S8)
    if n == 0:
        return labels, [], {}

    fn = ENGINE_BY_ID[engine_id]['fn']
    kwargs = dict(params)
    kwargs['n'] = int(min(kwargs.get('n', N_PARTIALS_DEFAULT), MAX_MODES_PER_OBJ))
    _fullshape = bool(kwargs.get('fullshape', False))

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
        patch = sub if _fullshape else extract(sub, PATCH_SIZE)

        cx = xs.mean() / max(GRID_W - 1, 1)   # stereo pan [0..1]

        freqs, amps = fn(patch, f0, **kwargs)

        # Colour: hue by object index (spread across spectrum), brightness by area
        hue = (idx / max(MAX_VOICES - 1, 1)) * 0.72
        val = float(np.clip(0.3 + 0.7 * min(area / 20.0, 1.0), 0.3, 1.0))
        color[lab] = hsv(hue, val)

        voices.append({
            'freqs':     freqs,             # shape (n,), n = params['n'] (<= MAX_MODES_PER_OBJ)
            'amps':      amps,              # shape (n,)
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
                           amp_tgt, pan_tgt, freq_slots, channels, gain_prev, gain):
    """Render one gapless chunk over the flat slot pool.

    gain_prev, gain : pre-clip master gain (MASTER_GAIN * volume) at the start and
    end of this chunk.  The gain is GLIDED from gain_prev to gain across the chunk
    (smoothstep) so dragging the volume slider doesn't step the level at a chunk
    boundary (that step was an audible click).  Applied BEFORE the clip, so lowering
    volume pulls the summed signal below the clip ceiling -- an honest headroom
    control (the timbre/spectrum is untouched, only the level).

    Returns (buf, peak, n_clip): buf is int16 (n_samples x channels); peak is the
    PRE-clip absolute peak (1.0 == 0 dBFS ceiling; >1.0 means we clipped and by
    how much); n_clip is the count of samples that exceeded the ceiling.

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

    g = gain_prev + (gain - gain_prev) * _RAMP   # smooth gain ramp -> click-free volume drag
    L *= g
    R *= g
    # Pre-clip metering: peak relative to the 1.0 ceiling (0 dBFS) and how many
    # samples ran over.  Measured BEFORE the clip so the meter shows the true
    # overshoot the user is asked to tame with the volume knob.
    peak = float(max(np.abs(L).max(), np.abs(R).max()))
    n_clip = int(np.count_nonzero(np.abs(L) > 1.0) + np.count_nonzero(np.abs(R) > 1.0))
    L = np.clip(L, -1.0, 1.0)
    R = np.clip(R, -1.0, 1.0)

    if channels <= 1:
        data = ((L + R) * 0.5)[:, None]
    else:
        data = np.zeros((n, channels))
        data[:, 0] = L
        data[:, 1] = R

    return np.ascontiguousarray((data * 32767).astype(np.int16)), peak, n_clip


class SlotPool:
    """Active + tail slot pool with seamless (click-free) release tails.

    Layout (slots 1..TOTAL_SLOTS; slot 0 unused):
      ACTIVE slots 1..N_ACTIVE: one per voice×mode, holding the CURRENT sounding
        mode.  active(v,m) = 1 + v * MAX_MODES_PER_OBJ + m.
      TAIL slots N_ACTIVE+1..TOTAL_SLOTS: a pool of release-only slots.

    On a frequency change (topology event) or voice death, the OLD mode is NOT
    cut.  Its (freq, phase, current amplitude, pan) are MOVED into a free tail
    slot which then rings down over release_chunks, while the active slot restarts
    on the new frequency from amplitude 0 (fast one-chunk attack).  Because the
    tail continues the EXACT waveform the active slot was producing (same phase,
    same amplitude at the seam), the summed signal is continuous across the swap
    -> no click -- and tails of arbitrary length overlap into a pad.

    This replaces the old front/back 2-bank scheme, which had only two slots per
    voice×mode: a still-ringing tail got its amplitude slammed to 0 when the bank
    was reused for the next attack, a step discontinuity heard as a boundary click
    whenever release was long (DEV-2; confirmed on a session snapshot: forcing
    release short cut sharp transients 465 -> 38).

    If the tail pool is exhausted, the QUIETEST tail is stolen (smallest residual
    amplitude -> smallest click) -- graceful degradation under extreme stacking.
    """

    def __init__(self):
        sz = TOTAL_SLOTS + 1   # +1 so slot 0 is a harmless dummy
        self.freq_slots  = np.zeros(sz)        # Hz; 0 = silent
        self.amp_tgt     = np.zeros(sz)
        self.pan_tgt     = np.full(sz, 0.5)

        # Release countdown (in chunks) per slot; 0 = not releasing.
        self._release_cnt = np.zeros(sz, dtype=int)
        # Amplitude at the start of release (for linear step computation).
        self._release_amp0 = np.zeros(sz)
        # Release LENGTH (chunks) captured when each tail started, so a live
        # release-knob change mid-tail does not rescale an in-flight fade.
        self._release_len = np.ones(sz, dtype=int)

        # Last committed frequency per voice×mode active slot (to detect changes).
        self._freq_prev = np.zeros((MAX_VOICES, MAX_MODES_PER_OBJ))

        # ADSR envelope state per ACTIVE slot (tails ring out via the release
        # countdown instead).  phase: 0 idle, 1 attack, 2 decay, 3 sustain.
        # level: current 0..1 multiplier applied to the engine's per-mode amplitude.
        self._env_phase = np.zeros(sz, dtype=int)
        self._env_level = np.zeros(sz)

        # Diagnostics: count tail-pool steals (pool exhausted -> a still-ringing
        # tail had to be overwritten = a residual click) and the loudest stolen
        # amplitude.  Read by the probe; zero in normal (non-exhausted) operation.
        self.steals = 0
        self.steal_amp_max = 0.0

    @staticmethod
    def _active(v, m):
        """Slot index of the active (currently sounding) slot for voice v, mode m."""
        return 1 + v * MAX_MODES_PER_OBJ + m

    def _acquire_tail(self, amp_cur):
        """Return a tail slot for a new ringing-out mode: a free (silent, not
        releasing) one if available, else the quietest currently-ringing one
        (stealing it -> the smallest possible click)."""
        lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
        quietest, quiet_amp = lo, np.inf
        for s in range(lo, hi):
            if self._release_cnt[s] == 0 and amp_cur[s] < 1e-4:
                return s
            if amp_cur[s] < quiet_amp:
                quiet_amp, quietest = amp_cur[s], s
        # No free slot: stealing the quietest still-ringing tail.  Eviction below
        # keeps the quietest near zero, so this steal is normally near-silent.
        self.steals += 1
        self.steal_amp_max = max(self.steal_amp_max, float(quiet_amp))
        return quietest

    def _enforce_tail_budget(self, amp_cur):
        """When ringing tails exceed the pool minus a reserve, fast-fade the
        QUIETEST excess to zero over FAST_EVICT_CHUNKS (a quick, smooth release --
        not a cut) so slots free up before a loud tail must be stolen."""
        lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
        ring = [s for s in range(lo, hi)
                if self._release_cnt[s] > 0 or amp_cur[s] >= 1e-4]
        over = len(ring) - (N_TAIL - TAIL_RESERVE)
        if over <= 0:
            return
        ring.sort(key=lambda s: amp_cur[s])      # quietest first
        for s in ring[:over]:
            if self._release_cnt[s] == 0 or self._release_cnt[s] > FAST_EVICT_CHUNKS:
                self._release_cnt[s]  = FAST_EVICT_CHUNKS
                self._release_len[s]  = FAST_EVICT_CHUNKS
                self._release_amp0[s] = float(amp_cur[s])

    def _advance_env(self, a, attack_chunks, decay_chunks, sustain):
        """Advance the ADSR envelope of active slot `a` by one audio chunk.

        attack/decay are lengths in chunks (>=1); sustain is the held 0..1 level.
        Sets self._env_level[a] in place; release is NOT handled here (tail pool).
        """
        ph = self._env_phase[a]
        if ph == 1:                                   # attack: 0 -> 1 (peak)
            lvl = self._env_level[a] + 1.0 / attack_chunks
            if lvl >= 1.0:
                lvl, self._env_phase[a] = 1.0, 2      # -> decay
            self._env_level[a] = lvl
        elif ph == 2:                                 # decay: 1 -> sustain level
            lvl = self._env_level[a] - (1.0 - sustain) / decay_chunks
            if lvl <= sustain:
                lvl, self._env_phase[a] = sustain, 3  # -> sustain
            self._env_level[a] = lvl
        elif ph == 3:                                 # sustain: track live level
            self._env_level[a] = sustain
        # ph == 0 (idle): level stays at its current value (0 for a silent slot).

    def update(self, voices, phase, amp_cur, pan_cur, release_chunks,
               attack_chunks, decay_chunks, sustain):
        """Push a new voice list into the slot pool.

        phase / amp_cur / pan_cur : the engine's per-slot state arrays (mutated in
            place) -- needed so a replaced mode can be MOVED into a tail slot
            continuing the exact same waveform (phase + amplitude), seamlessly.
        release_chunks : current mode-tail release length in audio chunks (live
            release knob -- see RELEASE_MS_* / feed_audio).
        attack_chunks / decay_chunks / sustain : live ADSR knobs (see
            ATTACK_MS_* / DECAY_MS_* / SUSTAIN_*).  Each active slot carries an
            envelope (0..1) multiplying the engine's per-mode amplitude; it is
            (re)triggered to ATTACK on every mode onset and advanced one chunk per
            update via _advance_env.

        For each voice×mode:
          - frequency unchanged: advance the envelope, glide amp(=env·amp_new)/pan
            targets on the active slot.
          - frequency changed (or voice gone): if the old mode is audible, move it
            into a tail slot (seamless continuation) to ring out over
            release_chunks; then restart the active slot on the new frequency from
            amplitude 0 with the envelope re-triggered to ATTACK, or silence it if
            the mode is gone.
        Old (tail) and new (active) sound in parallel -> overlap, no gap, no click.
        """
        n_voices = min(len(voices), MAX_VOICES)

        # Free up slots BEFORE this cycle's spawns: fast-fade the quietest excess
        # tails so a per-step burst of new tails finds (near-)silent slots instead
        # of stealing loud ones.  Run every update (incl. no-spawn ones) to keep
        # headroom ready for the next GOL-step burst.
        self._enforce_tail_budget(amp_cur)

        # ── Phase 1: assign active slots, spawn tails for replaced modes ──────
        for v in range(MAX_VOICES):
            present = v < n_voices
            voice = voices[v] if present else None
            for m in range(MAX_MODES_PER_OBJ):
                a = self._active(v, m)
                if present:
                    freq_new = float(voice['freqs'][m]) if m < len(voice['freqs']) else 0.0
                    amp_new  = float(voice['amps'][m])  if m < len(voice['amps'])  else 0.0
                    pan_new  = voice['pan']
                else:
                    freq_new, amp_new, pan_new = 0.0, 0.0, pan_cur[a]

                freq_changed = abs(freq_new - self._freq_prev[v, m]) > 0.01
                if not freq_changed:
                    # STABLE: advance the envelope, glide amp(=env·amp_new)/pan.
                    self.freq_slots[a] = freq_new
                    self.pan_tgt[a] = pan_new
                    if freq_new > 0.0:
                        self._advance_env(a, attack_chunks, decay_chunks, sustain)
                        self.amp_tgt[a] = self._env_level[a] * amp_new
                    else:
                        self._env_phase[a] = 0
                        self._env_level[a] = 0.0
                        self.amp_tgt[a] = 0.0
                    self._freq_prev[v, m] = freq_new
                    continue

                # Frequency changed: ring the OLD mode out in a tail slot, if it
                # is still audible -- copy freq/phase/amp/pan so the tail continues
                # the active slot's exact waveform (no discontinuity at the seam).
                if amp_cur[a] > TAIL_MIN_AMP and self.freq_slots[a] > 0.0:
                    t = self._acquire_tail(amp_cur)
                    self.freq_slots[t] = self.freq_slots[a]
                    amp_cur[t]         = amp_cur[a]
                    phase[t]           = phase[a]
                    pan_cur[t]         = pan_cur[a]
                    self.pan_tgt[t]    = self.pan_tgt[a]
                    self._release_cnt[t]  = release_chunks
                    self._release_len[t]  = release_chunks
                    self._release_amp0[t] = float(amp_cur[a])
                    # amp_tgt[t] is stepped down by Phase 2 below.

                # Restart the active slot on the new frequency, re-triggering the
                # ADSR envelope to ATTACK from 0; or silence it if the mode is gone.
                # amp_cur=0 makes the phase value inaudible at the restart.
                amp_cur[a] = 0.0
                self.pan_tgt[a] = pan_new
                if freq_new > 0.0:
                    self.freq_slots[a] = freq_new
                    self._env_phase[a] = 1          # (re)trigger -> attack
                    self._env_level[a] = 0.0
                    self._advance_env(a, attack_chunks, decay_chunks, sustain)
                    self.amp_tgt[a] = self._env_level[a] * amp_new
                else:
                    self.freq_slots[a] = 0.0
                    self._env_phase[a] = 0
                    self._env_level[a] = 0.0
                    self.amp_tgt[a] = 0.0
                self._freq_prev[v, m] = freq_new

        # ── Phase 2: advance tail release countdowns ──────────────────────────
        # Done AFTER assignment so a tail spawned this cycle gets its first decay
        # step immediately (countdown release_chunks -> release_chunks-1 here).
        # The divisor is the per-slot length captured at spawn, so changing the
        # knob mid-tail does not jump an in-flight fade.
        for s in range(N_ACTIVE + 1, TOTAL_SLOTS + 1):
            cnt = self._release_cnt[s]
            if cnt <= 0:
                continue
            new_cnt = cnt - 1
            self._release_cnt[s] = new_cnt
            self.amp_tgt[s] = self._release_amp0[s] * new_cnt / self._release_len[s]
            # When done and silent, free the slot (render glides amp_cur to 0).
            if new_cnt == 0 and amp_cur[s] < 0.01:
                self.freq_slots[s] = 0.0


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
    # Faithful-replay log (per rendered frame): inputs + chunk count.
    # replay (legacy fixed-width, laplacian-probe contract) columns:
    #   [n_rendered, note, spread, alpha, release_ms, vol, n_partials]
    # replay_controls: AUTHORITATIVE per-frame dict of EVERY synth control
    #   (engine id, note, vol, attack/decay/sustain/release, full engine_params) ->
    #   any control bug is reproducible from the recording; use this for faithful
    #   replay (it auto-captures every knob, incl. ones added after this matrix).
    # replay_engines: parallel array of the active engine id per rendered frame
    # (multi-engine; faithful replay must call the matching map_* per frame).
    replay = (np.array(rec.get('replay', []), dtype=float)
              if rec.get('replay') else np.zeros((0, 7)))
    replay_controls = np.array(rec.get('replay_controls', []), dtype=object)
    replay_engines = np.array(rec.get('replay_engines', []), dtype=object)
    replay_grids = (np.stack(rec['replay_grids']).astype(np.uint8)
                    if rec.get('replay_grids')
                    else np.zeros((0, GRID_H, GRID_W), np.uint8))
    np.savez_compressed(f"{base}.npz", frames=frames, gens=gens, grids=grids,
                        notes=notes, sr=SR, chunk_s=CHUNK_S,
                        bpm=rec.get('bpm', BPM_DEFAULT),
                        div_idx=rec.get('div_idx', DIV_DEFAULT),
                        max_voices=MAX_VOICES, master_gain=MASTER_GAIN,
                        max_modes=MAX_MODES_PER_OBJ, patch_size=PATCH_SIZE,
                        underruns=rec['underruns'],
                        replay=replay, replay_grids=replay_grids,
                        replay_engines=replay_engines,
                        replay_controls=replay_controls)
    # frames columns: [dt_ms, gen, n_voices, n_rendered, underrun]
    n_hitch = int((frames[:, 0] > CHUNK_S * 1000).sum()) if len(frames) else 0
    print(f"[session saved] {base}.wav ({len(rec['chunks'])} chunks) + {base}.npz "
          f"({len(rec['steps'])} steps, {len(replay)} replay frames)  "
          f"underruns={rec['underruns']}  frames>{int(CHUNK_S*1000)}ms={n_hitch}")


# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE SESSION REPLAY  (read side of the session log -- see _dump_session)
# ──────────────────────────────────────────────────────────────────────────────
def replay_session(ts, prefix="_session"):
    """Re-render a recorded session (<prefix>_<ts>.npz) OFFLINE from the
    AUTHORITATIVE per-frame control log, deterministically -- so a reported bug
    reproduces on the user's exact snapshot and a fix is validated against it
    (with unfixed code the replay must match the recorded WAV).

    Uses `replay_controls` (object array -> needs allow_pickle), which captures
    the FULL control state per rendered frame (engine id + note + vol + ADSR +
    every engine_params knob, incl. shape).  Because it forwards the whole
    engine_params dict through the SAME analyse()/SlotPool/render path the live
    synth uses, replay tracks the engine exactly and needs no per-knob edits when
    a new control is added (only that the control is in the snapshot).

    Returns (audio int16 (N,2), pre_clip_peak, n_clipped).
    """
    import os
    npz = f"{prefix}_{ts}.npz"
    if not os.path.exists(npz):
        raise SystemExit(f"no session file {npz} (recorded sessions: "
                         f"{prefix}_<ts>.npz from CASYNTH_RECORD=1)")
    d = np.load(npz, allow_pickle=True)
    controls = d["replay_controls"] if "replay_controls" in d.files else np.array([])
    if len(controls) == 0:
        raise SystemExit(
            f"{prefix}_{ts}.npz has no replay_controls log (recorded before "
            "full-control logging was added -- nothing to replay faithfully)")
    grids = d["replay_grids"]
    # Match the recorded run's master gain so old sessions stay faithful even if
    # the MASTER_GAIN constant changes later (live gain = master_gain * vol).
    master_gain = float(d["master_gain"]) if "master_gain" in d.files else MASTER_GAIN

    pool = SlotPool()
    sz = TOTAL_SLOTS + 1
    phase   = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)
    out, pkmax, nclip = [], 0.0, 0
    # gain_prev seeds the first chunk's gain glide (as the live gain_prev_box does).
    gain_prev = master_gain * float(controls[0]["vol"])
    for i, c in enumerate(controls):
        c = dict(c)                              # 0-d object array -> dict
        grid = grids[i]
        f0 = midi_to_freq(int(c["note"]))
        _, voices, _ = analyse(grid, f0, c["engine"], dict(c["engine_params"]))
        release_chunks = max(1, round(float(c["release_ms"]) / 1000.0 / CHUNK_S))
        attack_chunks  = max(1, round(float(c["attack_ms"])  / 1000.0 / CHUNK_S))
        decay_chunks   = max(1, round(float(c["decay_ms"])   / 1000.0 / CHUNK_S))
        sustain        = float(c["sustain"])
        gain = master_gain * float(c["vol"])
        for _ in range(int(c["n_rendered"])):
            pool.update(voices, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain)
            buf, pk, nc = render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                 pool.amp_tgt, pool.pan_tgt,
                                                 pool.freq_slots, 2, gain_prev, gain)
            gain_prev = gain
            out.append(buf)
            pkmax = max(pkmax, pk)
            nclip += nc
    audio = np.concatenate(out) if out else np.zeros((0, 2), np.int16)
    return audio, pkmax, nclip


def _replay_cli(ts, prefix="_session"):
    """`python gol_synth.py replay <ts>`: render the session offline to
    artifacts/_replay_<ts>.wav and, if the recorded WAV is present, report
    sample-level fidelity (max/mean abs diff) -- the regression check."""
    import os
    from scipy.io import wavfile
    audio, pk, nclip = replay_session(ts, prefix)
    os.makedirs("artifacts", exist_ok=True)
    out = os.path.join("artifacts", f"_replay_{ts}.wav")
    wavfile.write(out, SR, audio)
    print(f"[replay] {out}  {len(audio)} samples  pre-clip peak={pk:.3f}  "
          f"clipped={nclip}")
    rec_wav = f"{prefix}_{ts}.wav"
    if os.path.exists(rec_wav):
        _sr, ref = wavfile.read(rec_wav)
        n = min(len(ref), len(audio))
        if n:
            diff = np.abs(ref[:n].astype(np.int64) - audio[:n].astype(np.int64))
            print(f"[fidelity vs {rec_wav}] common={n} samples  "
                  f"max|diff|={int(diff.max())}  mean|diff|={diff.mean():.3f}  "
                  f"(len ref={len(ref)} replay={len(audio)})")
    else:
        print(f"[fidelity] no recorded {rec_wav} to compare against")


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

    def _audio_cb(outdata, frames, time_info, status):
        # Volume is now applied PRE-clip in render_chunk_laplacian, so the callback
        # just copies finished int16 samples -- no per-sample multiply here.
        filled = 0
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
            outdata[filled:filled + take] = buf[pos:pos + take]
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
    pygame.display.set_caption("CASynth — multi-engine")
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

    # Level/clip meter (updated in feed_audio, read in draw -- same thread).
    meter = {'peak': 0.0, 'clip': False}

    # optional session recording (env CASYNTH_RECORD=1).
    # 'replay'/'replay_grids' capture, per RENDERED frame, the FULL engine input
    # (grid + note + every live knob + how many chunks were rendered) so the
    # session can be re-rendered deterministically offline and a fix verified on
    # the user's exact snapshot (see _render_probe.py session mode).
    rec = None
    if os.environ.get('CASYNTH_RECORD'):
        rec = dict(chunks=[], frames=[], steps=[], underruns=0,
                   replay=[], replay_grids=[], replay_engines=[],
                   replay_controls=[],
                   bpm=BPM_DEFAULT, div_idx=DIV_DEFAULT)
        print("[CASYNTH_RECORD on] capturing audio + field + all knobs; "
              "quit (Esc) to save")

    state = dict(
        run=False, gen=0,
        bpm=BPM_DEFAULT, div_idx=DIV_DEFAULT,
        # next_step_time: absolute perf_counter deadline for the next GOL step.
        # None until the first Play press; reset to now+interval on each Play.
        next_step_time=None,
        note=NOTE_DEFAULT, vol=VOL_DEFAULT,
        kb_base=NOTE_DEFAULT,
        sidebar_open=True,
        engine=ENGINES[0]['id'],
        engine_params={e['id']: {arg: default for (arg, _l, _lo, _hi, _i, default)
                                 in e['params']} for e in ENGINES},
        release_ms=RELEASE_MS_DEFAULT,
        attack_ms=ATTACK_MS_DEFAULT,
        decay_ms=DECAY_MS_DEFAULT,
        sustain=SUSTAIN_DEFAULT,
    )

    # toolbar layout
    by     = GRID_H * CELL + 8
    div_y  = by + 46            # note-division button row
    info_y = GRID_H * CELL + 76 # status / volume row (shifted down for div row)
    defs = [("play", None, 96), ("step", "Step", 70),
            ("random", "Random", 96), ("clear", "Clear", 78)]
    buttons, bx = [], 12
    for bid, label, bw in defs:
        buttons.append(dict(id=bid, label=label, rect=pygame.Rect(bx, by, bw, 40)))
        bx += bw + 8

    # BPM widget: "BPM" label + slider track + value  (right of the main buttons)
    _BPM_LBL_W  = 34   # width reserved for "BPM" text
    _BPM_TRACK_W = 84
    _BPM_VAL_W   = 32
    _bpm_x = bx + 8
    bpm_track = pygame.Rect(_bpm_x + _BPM_LBL_W, by + 16, _BPM_TRACK_W, 8)
    _bpm_widget_w = _BPM_LBL_W + _BPM_TRACK_W + 4 + _BPM_VAL_W

    legend_x = 12
    _pat_btn = pygame.Rect(_bpm_x + _bpm_widget_w + 8, by, 96, 40)
    # Right column (former sidebar area): ADSR → level meter → vol slider.
    _rc_x        = W + 8                          # label left edge
    _RC_LABEL_W  = 24                             # width of label area
    _rc_track_x  = _rc_x + _RC_LABEL_W + 4       # track left = W + 36
    _RC_TRACK_W  = 100                            # track width (value label at W+142, 50px to edge)
    _vol_section_y = by + 108                     # starts below 4 ADSR rows + gap
    meter_track  = pygame.Rect(_rc_track_x, _vol_section_y + 13, _RC_TRACK_W, 10)
    vol_track    = pygame.Rect(_rc_track_x, _vol_section_y + 62, _RC_TRACK_W, 8)

    # Note-division buttons  (one compact row below main buttons)
    _DIV_BTN_H = 18
    _DIV_BTN_GAP = 3
    div_buttons = []
    _dx = 12
    for _di, (_dlbl, _) in enumerate(NOTE_DIVS):
        _dw = small.size(_dlbl)[0] + 14   # label + padding
        div_buttons.append(dict(idx=_di, label=_dlbl,
                                rect=pygame.Rect(_dx, div_y, _dw, _DIV_BTN_H)))
        _dx += _dw + _DIV_BTN_GAP

    # ── Engine selector: a row of TABS in the toolbar's bottom strip ──────────
    # Built from the shared engine registry (casynth_core.ENGINES); a click
    # switches state['engine'] and rebuilds the knob panel for that engine.
    _tab_y = GRID_H * CELL + TOOLBAR_H - 24
    engine_tabs = []
    _tx = 12
    for _e in ENGINES:
        _tw = small.size(_e['label'])[0] + 18
        engine_tabs.append(dict(id=_e['id'], label=_e['label'],
                                rect=pygame.Rect(_tx, _tab_y, _tw, 20)))
        _tx += _tw + 4

    # Live knobs -- mouse-drag sliders to the right of the Lib button, in TWO
    # stacked groups: the ACTIVE engine's attributes (top, DYNAMIC per engine via
    # rebuild_ctrls) and the synth-wide ADSR ENVELOPE block A/D/S/R (fixed position
    # below, so it reads as a stable separate block independent of engine).
    # Engine-attribute sliders write state['engine_params'][engine]; the ADSR
    # sliders write state['attack_ms'/'decay_ms'/'sustain'/'release_ms'].
    CTRL_LABEL_W, CTRL_TRACK_W = 56, 120
    _ctrl_x  = _pat_btn.right + 16
    _ctrl_y0 = by + 2
    _CTRL_ROW_H = 22
    # ADSR is in the right column; ENV header and sliders start at _ctrl_y0.
    _ENV_HDR_RC_Y = _ctrl_y0
    _ENV_RC_Y0    = _ctrl_y0 + 18
    ctrls = []

    def _fmt_for(integer, is_ms):
        if is_ms:
            return lambda v: f"{v:.0f}ms"
        if integer:
            return lambda v: f"{int(v)}"
        return lambda v: f"{v:.2f}"

    def rebuild_ctrls():
        """Repopulate `ctrls`: the active engine's params (top) + the synth-wide
        ADSR envelope block A/D/S/R (fixed position below -- see _ENV_* geometry)."""
        ctrls.clear()
        for row, (arg, lbl, lo, hi, integer, _default) in enumerate(
                ENGINE_BY_ID[state['engine']]['params']):
            ctrls.append(dict(
                id=arg, label=lbl, lo=lo, hi=hi, integer=integer, scope='engine',
                block='engine', fmt=_fmt_for(integer, False),
                label_x=_ctrl_x,
                track=pygame.Rect(_ctrl_x + CTRL_LABEL_W,
                                  _ctrl_y0 + row * _CTRL_ROW_H, CTRL_TRACK_W, 8)))
        # ADSR envelope in the right column (synth-wide; A/D/R are durations, S level).
        env_specs = [
            ('attack_ms',  'A', ATTACK_MS_MIN,  ATTACK_MS_MAX,  True),
            ('decay_ms',   'D', DECAY_MS_MIN,   DECAY_MS_MAX,   True),
            ('sustain',    'S', SUSTAIN_MIN,    SUSTAIN_MAX,    False),
            ('release_ms', 'R', RELEASE_MS_MIN, RELEASE_MS_MAX, True),
        ]
        for i, (arg, lbl, lo, hi, is_ms) in enumerate(env_specs):
            ctrls.append(dict(
                id=arg, label=lbl, lo=lo, hi=hi, integer=False, scope='synth',
                block='env', fmt=_fmt_for(False, is_ms),
                label_x=_rc_x,
                track=pygame.Rect(_rc_track_x,
                                  _ENV_RC_Y0 + i * _CTRL_ROW_H, _RC_TRACK_W, 8)))

    rebuild_ctrls()

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
    _sb_scroll_min = min(0, GRID_H * CELL - _sb_content_h)

    piano_top = GRID_H * CELL + TOOLBAR_H
    white_keys, black_keys = _make_piano(state['kb_base'], piano_top, W)

    paint = None
    dragging_vol = False
    dragging_bpm = False
    dragging_ctrl = None          # id of the timbre knob being dragged, or None
    drag = {'active': False, 'cells': [], 'name': '', 'snap': None}
    _ghost = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
    _ghost.fill((111, 208, 224, 110))

    def cell_at(mx, my):
        if 0 <= my < GRID_H * CELL and 0 <= mx < GRID_W * CELL:
            return my // CELL, mx // CELL
        return None

    def f0():
        return midi_to_freq(state['note'])

    def _step_interval():
        """GOL step period in seconds for the current BPM + note division."""
        return NOTE_DIVS[state['div_idx']][1] * 60.0 / state['bpm']

    def set_vol(mx):
        # Pre-clip master volume: lowering it pulls the signal below the clip
        # ceiling (manual headroom).  Read in feed_audio on the main thread.
        state['vol'] = float(np.clip((mx - vol_track.left) / vol_track.width, 0, 1))

    def set_bpm(mx):
        frac = float(np.clip((mx - bpm_track.left) / bpm_track.width, 0, 1))
        state['bpm'] = int(round(BPM_MIN + frac * (BPM_MAX - BPM_MIN)))

    def ctrl_by_id(cid):
        return next(c for c in ctrls if c['id'] == cid)

    def ctrl_value(c):
        """Current value of a knob: engine attributes live in the active engine's
        per-engine dict; the release knob lives directly in state (synth-wide)."""
        if c['scope'] == 'engine':
            return state['engine_params'][state['engine']][c['id']]
        return state[c['id']]

    def set_ctrl(c, mx):
        frac = float(np.clip((mx - c['track'].left) / c['track'].width, 0, 1))
        val = c['lo'] + frac * (c['hi'] - c['lo'])
        val = int(round(val)) if c.get('integer') else val
        if c['scope'] == 'engine':
            state['engine_params'][state['engine']][c['id']] = val
        else:
            state[c['id']] = val

    def do(bid):
        nonlocal grid
        if bid == "play":
            state['run'] = not state['run']
            if state['run']:
                # Schedule first step one interval from now so the user hears the
                # current generation first, then the clock starts ticking.
                state['next_step_time'] = time.perf_counter() + _step_interval()
        elif bid == "step":  grid = step(grid); state['gen'] += 1
        elif bid == "random":
            grid[:] = (np.random.random((GRID_H, GRID_W)) < RANDOM_DENSITY).astype(np.uint8)
        elif bid == "clear": grid[:] = 0; state['gen'] = 0

    def hit_piano(pos):
        for rect, m in black_keys:
            if rect.collidepoint(pos):
                return m
        for rect, m in white_keys:
            if rect.collidepoint(pos):
                return m
        return None

    gain_prev_box = [MASTER_GAIN * state['vol']]   # master gain of the last chunk

    def feed_audio(voices):
        if not audio_ok:
            return 0, 0
        # Top the ring buffer up to AUDIO_LOOKAHEAD_CHUNKS of look-ahead.
        # pool.update is advanced once PER CHUNK rendered (not per frame), so the
        # release-tail countdown counts chunks and is not coupled to the frame rate.
        n_rendered = 0
        # Release-tail length in chunks, derived live from the knob (decoupled
        # from the crossfade attack, which stays one chunk).
        release_chunks = max(1, round(state['release_ms'] / 1000.0 / CHUNK_S))
        # ADSR onset shaping, derived live from the knobs (chunk-quantised like
        # release).  sustain is a level, not a time.
        attack_chunks = max(1, round(state['attack_ms'] / 1000.0 / CHUNK_S))
        decay_chunks  = max(1, round(state['decay_ms']  / 1000.0 / CHUNK_S))
        sustain       = float(state['sustain'])
        gain = MASTER_GAIN * state['vol']      # pre-clip master gain (chunk target)
        frame_peak = 0.0
        frame_clip = 0
        while audio_q.qsize() < AUDIO_LOOKAHEAD_CHUNKS:
            pool.update(voices, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain)
            buf, peak, n_clip = render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                       pool.amp_tgt, pool.pan_tgt,
                                                       pool.freq_slots, 2,
                                                       gain_prev_box[0], gain)
            gain_prev_box[0] = gain            # glide from here next chunk
            audio_q.put(buf)
            n_rendered += 1
            frame_peak = max(frame_peak, peak)
            frame_clip += n_clip
            if rec is not None:
                rec['chunks'].append(buf)
        # Peak-hold with decay so the meter is readable; clip flag is per-frame.
        meter['peak'] = max(frame_peak, meter['peak'] * METER_DECAY)
        meter['clip'] = frame_clip > 0
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
                elif e.key == pygame.K_UP:
                    _d = 10 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 5
                    state['bpm'] = min(BPM_MAX, state['bpm'] + _d)
                elif e.key == pygame.K_DOWN:
                    _d = 10 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 5
                    state['bpm'] = max(BPM_MIN, state['bpm'] - _d)
                elif e.key == pygame.K_RIGHT:
                    state['div_idx'] = min(len(NOTE_DIVS) - 1, state['div_idx'] + 1)
                elif e.key == pygame.K_LEFT:
                    state['div_idx'] = max(0, state['div_idx'] - 1)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if state['sidebar_open'] and e.pos[0] >= W and e.pos[1] < GRID_H * CELL and e.button == 1:
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
                        tab_hit = next((t for t in engine_tabs
                                        if t['rect'].collidepoint(e.pos)), None)
                        if tab_hit is not None:
                            if tab_hit['id'] != state['engine']:
                                state['engine'] = tab_hit['id']
                                rebuild_ctrls()   # swap knob panel to new engine
                        elif _pat_btn.collidepoint(e.pos):
                            state['sidebar_open'] = not state['sidebar_open']
                        elif bpm_track.inflate(0, 16).collidepoint(e.pos):
                            dragging_bpm = True
                            set_bpm(e.pos[0])
                        elif any(b['rect'].inflate(0, 6).collidepoint(e.pos)
                                 for b in div_buttons):
                            for b in div_buttons:
                                if b['rect'].inflate(0, 6).collidepoint(e.pos):
                                    state['div_idx'] = b['idx']
                                    break
                        elif vol_track.collidepoint(e.pos):
                            dragging_vol = True
                            set_vol(e.pos[0])
                        elif any(c['track'].inflate(0, 14).collidepoint(e.pos)
                                 for c in ctrls):
                            for c in ctrls:
                                if c['track'].inflate(0, 14).collidepoint(e.pos):
                                    dragging_ctrl = c['id']
                                    set_ctrl(c, e.pos[0])
                                    break
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
                dragging_bpm = False
                dragging_ctrl = None

            elif e.type == pygame.MOUSEMOTION:
                if drag['active']:
                    drag['snap'] = cell_at(*e.pos)
                elif paint is not None:
                    rc = cell_at(*e.pos)
                    if rc is not None:
                        grid[rc] = paint
                elif dragging_vol:
                    set_vol(e.pos[0])
                elif dragging_bpm:
                    set_bpm(e.pos[0])
                elif dragging_ctrl is not None:
                    set_ctrl(ctrl_by_id(dragging_ctrl), e.pos[0])

            elif e.type == pygame.MOUSEWHEEL:
                if (state['sidebar_open'] and pygame.mouse.get_pos()[0] >= W
                        and pygame.mouse.get_pos()[1] < GRID_H * CELL):
                    _sb_scroll = max(_sb_scroll_min, min(0, _sb_scroll + e.y * 20))

        # GOL advance — strict BPM-aligned timing via perf_counter deadline.
        # next_step_time advances by exactly one interval each step, so the step
        # sequence lands on a perfect periodic grid regardless of frame-time jitter
        # (no float accumulation: += interval, not = now + interval).
        if state['run'] and state['next_step_time'] is not None:
            interval = _step_interval()
            now = time.perf_counter()
            n_steps = 0
            while now >= state['next_step_time'] and n_steps < 4:
                grid = step(grid)
                state['gen'] += 1
                state['next_step_time'] += interval
                n_steps += 1
                if rec is not None:
                    rec['steps'].append((state['gen'], grid.copy(), state['note']))
            # If the clock drifted far behind (e.g. OS pause), snap forward so we
            # don't spiral trying to catch up.
            if state['next_step_time'] < now - interval:
                state['next_step_time'] = now + interval

        _ep = state['engine_params'][state['engine']]
        labels, voices, color = analyse(grid, f0(), state['engine'], _ep)
        n_rendered, ur_delta = feed_audio(voices)
        if rec is not None:
            rec['frames'].append((round(dt * 1000.0, 2), state['gen'],
                                  len(voices), n_rendered, ur_delta))
            # Faithful-replay log: capture inputs ONLY for frames that actually
            # rendered chunks (frames with n_rendered=0 produce no audio and would
            # just bloat the grid log).  Values are constant within a frame, so
            # replaying pool.update/render n_rendered times reproduces the engine
            # output chunk-for-chunk.
            if n_rendered > 0:
                # Legacy fixed-width matrix (laplacian-probe contract; do not drop
                # columns).  spread/alpha default for engines that lack them.
                rec['replay'].append((n_rendered, int(state['note']),
                                      float(_ep.get('spread', 0.0)),
                                      float(_ep.get('alpha', 1.0)),
                                      float(state['release_ms']), float(state['vol']),
                                      int(_ep.get('n', N_PARTIALS_DEFAULT))))
                # AUTHORITATIVE per-frame control snapshot: EVERY synth control so
                # any bug is reproducible from the recording and a fix validated.
                # Copy the whole engine_params dict -> captures every engine knob
                # (incl. shape and any knob added later) with no per-knob edits here.
                rec['replay_controls'].append(dict(
                    n_rendered=int(n_rendered),
                    engine=state['engine'],
                    note=int(state['note']),
                    vol=float(state['vol']),
                    attack_ms=float(state['attack_ms']),
                    decay_ms=float(state['decay_ms']),
                    sustain=float(state['sustain']),
                    release_ms=float(state['release_ms']),
                    engine_params=dict(_ep)))
                rec['replay_engines'].append(state['engine'])
                rec['replay_grids'].append(grid.copy())

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
        plbl = small.render("◀ Lib" if state['sidebar_open'] else "Lib ▶", True, C_ACCENT)
        screen.blit(plbl, plbl.get_rect(center=_pat_btn.center))

        # BPM slider
        screen.blit(small.render("BPM", True, C_DIM),
                    (_bpm_x, bpm_track.centery - 7))
        pygame.draw.rect(screen, C_BTN, bpm_track, border_radius=4)
        _bpm_frac = (state['bpm'] - BPM_MIN) / (BPM_MAX - BPM_MIN)
        _bfw = int(bpm_track.width * _bpm_frac)
        pygame.draw.rect(screen, C_ACCENT,
                         (bpm_track.left, bpm_track.top, _bfw, bpm_track.height),
                         border_radius=4)
        pygame.draw.circle(screen, C_TXT, (bpm_track.left + _bfw, bpm_track.centery), 6)
        screen.blit(small.render(str(state['bpm']), True, C_TXT),
                    (bpm_track.right + 5, bpm_track.centery - 7))

        # Note-division buttons
        for _db in div_buttons:
            _dactive = (_db['idx'] == state['div_idx'])
            _dhot = _db['rect'].collidepoint(mouse)
            pygame.draw.rect(screen,
                             C_ACCENT if _dactive else (C_BTN_HOT if _dhot else C_BTN),
                             _db['rect'], border_radius=3)
            _dtxt = small.render(_db['label'], True, C_BG if _dactive else C_TXT)
            screen.blit(_dtxt, _dtxt.get_rect(center=_db['rect'].center))

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
        _div_lbl = NOTE_DIVS[state['div_idx']][0]
        st = (f"{note_name(state['note'])} {f0():.0f}Hz  "
              f"gen {state['gen']:>4}  "
              f"{state['bpm']} BPM {_div_lbl}  "
              f"obj {len(voices)}")
        screen.blit(font.render(st, True, C_TXT), (sx, info_y + 1))
        mode = "RUNNING" if state['run'] else "PAUSED"
        mode_surf = font.render(mode, True, C_ACCENT if state['run'] else C_DIM)
        screen.blit(mode_surf, (sx, info_y + 19))
        kbd_surf = small.render(f"kbd:{note_name(state['kb_base'])}", True, C_DIM)
        screen.blit(kbd_surf, (sx + mode_surf.get_width() + 10, info_y + 21))

        # info row: volume (right column, below level meter)
        screen.blit(small.render(f"vol {int(state['vol'] * 100)}%", True, C_DIM),
                    (_rc_x, _vol_section_y + 47))
        pygame.draw.rect(screen, C_BTN, vol_track, border_radius=4)
        fw = int(vol_track.width * state['vol'])
        pygame.draw.rect(screen, C_ACCENT,
                         (vol_track.left, vol_track.top, fw, vol_track.height),
                         border_radius=4)
        pygame.draw.circle(screen, C_TXT, (vol_track.left + fw, vol_track.centery), 6)
        if not audio_ok:
            screen.blit(small.render("audio disabled", True, C_DIM), (W - 110, 6))

        # ENV header in the right column, above the ADSR sliders.
        _env_rc_right = _rc_track_x + _RC_TRACK_W + 44
        pygame.draw.line(screen, C_EDGE, (_rc_x, _ENV_HDR_RC_Y),
                         (_env_rc_right, _ENV_HDR_RC_Y))
        screen.blit(small.render("ENV", True, C_DIM), (_rc_x, _ENV_HDR_RC_Y + 2))
        for c in ctrls:
            tr = c['track']
            val = ctrl_value(c)
            screen.blit(small.render(c['label'], True, C_DIM),
                        (c['label_x'], tr.centery - 7))
            pygame.draw.rect(screen, C_BTN, tr, border_radius=4)
            frac = float(np.clip((val - c['lo']) / (c['hi'] - c['lo']), 0, 1))
            cw = int(tr.width * frac)
            pygame.draw.rect(screen, C_ACCENT, (tr.left, tr.top, cw, tr.height),
                             border_radius=4)
            pygame.draw.circle(screen, C_TXT, (tr.left + cw, tr.centery), 5)
            screen.blit(small.render(c['fmt'](val), True, C_DIM),
                        (tr.right + 6, tr.centery - 7))

        # level / clip meter: pre-clip peak vs the 0 dBFS ceiling (right edge).
        # Bar turns red and shows "CLIP +X.XdB" of overshoot when over the ceiling
        # -> lower the volume slider below it for honest (un-clipped) sound.
        pk = meter['peak']
        db = 20.0 * np.log10(pk + 1e-9)
        clipping = meter['clip'] or pk >= 1.0
        screen.blit(small.render("level", True, C_DIM),
                    (_rc_x, meter_track.top - 13))
        pygame.draw.rect(screen, C_BTN, meter_track, border_radius=3)
        fillw = int(meter_track.width * min(pk, 1.0))
        pygame.draw.rect(screen, (212, 76, 76) if clipping else C_ACCENT,
                         (meter_track.left, meter_track.top, fillw, meter_track.height),
                         border_radius=3)
        pygame.draw.line(screen, C_TXT, (meter_track.right - 1, meter_track.top - 1),
                         (meter_track.right - 1, meter_track.bottom + 1))
        mlbl = (f"CLIP +{db:.1f}dB" if clipping else f"{db:.0f}dB")
        msurf = small.render(mlbl, True, (235, 96, 96) if clipping else C_DIM)
        screen.blit(msurf, (meter_track.right - msurf.get_width(), meter_track.bottom + 1))

        # ── engine selector tabs (bottom strip of the toolbar) ────────────────
        for t in engine_tabs:
            active = (t['id'] == state['engine'])
            hot_t = t['rect'].collidepoint(mouse)
            pygame.draw.rect(screen,
                             C_ACCENT if active else (C_BTN_HOT if hot_t else C_BTN),
                             t['rect'], border_radius=4)
            tlbl = small.render(t['label'], True, C_BG if active else C_TXT)
            screen.blit(tlbl, tlbl.get_rect(center=t['rect'].center))

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
            pygame.draw.rect(screen, C_PANEL, (W, 0, SIDEBAR_W, GRID_H * CELL))
            pygame.draw.line(screen, C_EDGE, (W, 0), (W, GRID_H * CELL))
            ttl = font.render("Patterns", True, C_TXT)
            screen.blit(ttl, (W + (SIDEBAR_W - ttl.get_width()) // 2, 6))
            pygame.draw.line(screen, C_EDGE, (W + 2, 28), (W + SIDEBAR_W - 2, 28))
            screen.set_clip(pygame.Rect(W, 30, SIDEBAR_W, GRID_H * CELL - 30))
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
                vis = GRID_H * CELL - 30
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
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'replay':
        _replay_cli(sys.argv[2])
    else:
        main()
