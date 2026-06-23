"""Tunable constants for the gol_synth prototype.

Single home for every tunable: geometry, audio/timing, slot-pool sizing, the
ADSR ranges, the colour palette and note tables.  Pure data -- imports only numpy
(for TWO_PI / the per-chunk ramp); NO pygame, so it can be imported by the audio
engine and offline harnesses without a display.

Consumers do `from casynth_config import *`; __all__ below lists the underscore
names (_RAMP, _SB_* ...) too, which a bare star-import would otherwise skip.
"""
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  -- all tunable constants here
# ──────────────────────────────────────────────────────────────────────────────
GRID_W, GRID_H = 52, 30
CELL = 20
# Toolbar: main button row (40px) + note-div row (18px) + status row (~38px) +
# engine-knob panel (Laplace 6×22px, right col: ADSR 4×22px + vol/level) +
# MIDI device bar (20px) + engine-tab strip (20px) + gaps.
# Height is determined by the ADSR+meter+vol column:
#   _vol_section_y = by+96; vol_track bottom = by+96+62+8 = by+166;
#   MIDI bar at by+170 (4px gap), height 20 → bottom by+190;
#   tabs at by+194 (4px gap), tab height 20, gap 4 → bar bottom = by+218
#   TOOLBAR_H = 8 + 218 = 226.
# (The envelope/vol block is NOT excluded -- it is compacted to sit right after
# the ADSR R-track so the bar height is governed by the vol slider, not air.)
TOOLBAR_H = 226
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

CHUNK_S = 0.008          # audio render sub-chunk length (seconds) = MIDI timing grid
# A NOTE ONSET can only take effect on a render-chunk boundary, so CHUNK_S is the
# quantisation grid for MIDI note timing.  History: 90 ms snapped a steady hardware
# arpeggiator (~225 ms eighths) to an audible 180/270 ms wobble (±45 ms); 20 ms cut
# that but left the per-FRAME MIDI poll (~18 ms) as the floor (sixteenths still
# floated).  The fix (see main()) moves audio rendering and MIDI polling OFF the
# 60 fps frame loop onto dedicated threads, so the grid is now CHUNK_S locked to the
# playback clock -- frame-independent.  8 ms keeps the 1-chunk mode crossfade
# click-safe while giving a ~±8 ms timing grid; shrink only if CPU allows (the
# render thread runs pool.update + render_chunk per sub-chunk -> more Python-loop
# work per second).  ADSR/release/evict are ms-derived and auto-rescale.
# A dedicated RENDER thread renders sub-chunks just-in-time into a small ring buffer
# drained by the sounddevice callback (which only copies int16 -- no heavy Python in
# the RT path).  The ring is small (jitter is one sub-chunk regardless of its depth;
# depth is only constant latency + slack to absorb a main-thread GIL hold).  Size it
# by WALL TIME so it stays constant if CHUNK_S changes.
AUDIO_LOOKAHEAD_MS = 40
AUDIO_LOOKAHEAD_CHUNKS = max(3, round(AUDIO_LOOKAHEAD_MS / 1000.0 / CHUNK_S))
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

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SIDEBAR_W  = 192
_SB_ITEM_H = 40
_SB_PREV_W = 50
_SB_PREV_H = 36
_SB_HDR_H  = 22

# Explicit export list: a bare `from casynth_config import *` skips _underscore
# names, but the engine needs _RAMP/_N_CHUNK and the UI needs _SB_*; list all.
__all__ = [
    'GRID_W', 'GRID_H', 'CELL', 'TOOLBAR_H', 'PIANO_H', 'FPS',
    'SR', 'RANDOM_DENSITY',
    'BPM_DEFAULT', 'BPM_MIN', 'BPM_MAX', 'NOTE_DIVS', 'DIV_DEFAULT', 'MAX_VOICES',
    'CHUNK_S', 'AUDIO_LOOKAHEAD_MS', 'AUDIO_LOOKAHEAD_CHUNKS',
    'MASTER_GAIN', 'VOL_DEFAULT', 'VOL_W', 'METER_DECAY',
    'MAX_MODES_PER_OBJ', 'PATCH_SIZE',
    'SPREAD_DEFAULT', 'ALPHA_DEFAULT', 'ALPHA_MIN', 'ALPHA_MAX',
    'N_PARTIALS_DEFAULT', 'N_PARTIALS_MIN',
    'N_ACTIVE', 'N_TAIL', 'TOTAL_SLOTS', 'TAIL_MIN_AMP',
    'FAST_EVICT_MS', 'FAST_EVICT_CHUNKS', 'TAIL_RESERVE',
    'RELEASE_MS_DEFAULT', 'RELEASE_MS_MIN', 'RELEASE_MS_MAX',
    'ATTACK_MS_DEFAULT', 'ATTACK_MS_MIN', 'ATTACK_MS_MAX',
    'DECAY_MS_DEFAULT', 'DECAY_MS_MIN', 'DECAY_MS_MAX',
    'SUSTAIN_DEFAULT', 'SUSTAIN_MIN', 'SUSTAIN_MAX',
    'NOTE_DEFAULT', 'KB_BASE_MIN', 'KB_BASE_MAX',
    'C_BG', 'C_GRID', 'C_PANEL', 'C_EDGE', 'C_TXT', 'C_DIM', 'C_BTN', 'C_BTN_HOT',
    'C_ACCENT', 'C_CAPPED', 'C_WHITE', 'C_WHITE_ON', 'C_BLACK', 'C_BLACK_ON',
    'NAMES', 'WHITE_PC', 'BLACK_PC', 'TWO_PI',
    '_N_CHUNK', '_RAMP',
    'SIDEBAR_W', '_SB_ITEM_H', '_SB_PREV_W', '_SB_PREV_H', '_SB_HDR_H',
]
