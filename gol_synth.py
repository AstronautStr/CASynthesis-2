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
    pip install pygame-ce numpy scipy sounddevice mido python-rtmidi
    python gol_synth.py
    # headless:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python gol_synth.py
"""


import os
import sys
import ctypes
import time
import queue as _queue
import threading
from types import SimpleNamespace
import numpy as np
import pygame
from casynth_core import ENGINES, ENGINE_BY_ID
from patterns import PATTERNS

# Modularised subsystems (extracted from this monolith -- see casynth_*.py).
from casynth_config import *                                   # noqa: F401,F403
from casynth_engine import (midi_to_freq, note_name, step, hsv, analyse,
                            render_chunk_laplacian, SlotPool, events_field,
                            VoiceEnvelope)
from casynth_session import (_dump_session, replay_session, _replay_cli,
                             _scene_cli)
from casynth_midi import MidiInput, MIDI_AVAILABLE
from casynth_midifile import MidiFilePlayer, MIDIFILE_AVAILABLE
from casynth_host import (AudioHost, open_output_stream, output_device_name,
                          output_devices, audio_choice_load, audio_choice_save)
from casynth_panel import (panel_rows, shared_first, KIND_CHOICES, KIND_INACTIVE,
                           KIND_SLIDER, KIND_TOGGLE)
from casynth_ui import _make_piano, pattern_preview_surf, draw_frame
from casynth_engines import registry as engines
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE
from casynth_tuning import (dissonance_curve, scale_minima, snap_ratio,
                             TUNE_MAX_PARTIALS)


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


# Piano-key geometry + pattern-preview surfaces live in casynth_ui.


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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(autoplay_midi=None):
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    # Render at physical pixels on high-DPI Windows.  Without this, a 125%-scaled
    # desktop reports a 1536x960 logical space to DPI-unaware apps, and our
    # ~1054px-tall window gets its bottom strip (the piano) clipped off-screen.
    if sys.platform == 'win32':
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    pygame.init()

    # ── audio output: the shared device host (casynth_host) ──────────────────
    # The synth (render thread) renders chunks ahead into the host's ring; the
    # PortAudio callback pulls samples at the hardware rate.  This decouples audio
    # from the 60 fps frame loop: a frame hitch only shrinks the ring, it never
    # gaps playback.  The engine's own audio state is touched only by the render
    # thread; the callback copies finished int16 chunks out of the ring, nothing else.
    # The ring, the callback, the look-ahead pre-roll and the underrun count are
    # the same code the bench runs -- sounddevice is imported inside it, so a
    # machine without it just reports no device here.
    # The device is opened AFTER the first frame, not here.  Starting up costs a
    # few hundred milliseconds (pygame, the pattern previews, and importing the
    # engine modules for the tab row), and a device that is already playing
    # drains its whole look-ahead in that time -- which used to be the burst of
    # underruns every session began with.  The render thread starts now and fills
    # the ring with real audio meanwhile.
    # CASYNTH_AUDIO_DEVICE picks the output by index or by a piece of its name;
    # without it the system default is used.  A machine can have a headset, a
    # monitor and onboard speakers at once, and "no sound" is usually a stream
    # that opened somewhere the person is not listening -- so the name of the
    # device that was opened is printed, every time.
    # WHICH OUTPUT.  A machine can have a headset, a monitor and onboard speakers
    # at once, and the system default is not always the one a person is listening
    # to -- on this machine the default endpoint accepts the stream and plays
    # nothing audible at all.  So the choice is the player's: it is remembered
    # between runs (audio_device.json), overridable with CASYNTH_AUDIO_DEVICE, and
    # changeable from the toolbar while the synth runs.
    _req = {'device': audio_choice_load()}
    _env_dev = os.environ.get('CASYNTH_AUDIO_DEVICE') or None
    if _env_dev is not None:
        _req['device'] = (int(_env_dev) if _env_dev.strip().lstrip('-').isdigit()
                          else _env_dev)
    _opened = {'device': _req['device']}
    # The name is asked of the sound system, which costs about 3 ms -- fine once,
    # ruinous 60 times a second.  It only changes when the device does.
    _name_cache = {'device': object(), 'name': ''}

    def _out_name():
        if _name_cache['device'] != _opened['device']:
            _name_cache['device'] = _opened['device']
            _name_cache['name'] = output_device_name(_opened['device'])
        return _name_cache['name']

    def _open_out(cb):
        """Open the requested output; if it cannot be opened, say why, list what
        there is, and fall back to the system default -- somebody who names a
        device wants SOUND, not silence with a reason."""
        want = _req['device']
        try:
            s = open_output_stream(cb, sr=SR, channels=2, device=want)
            _opened['device'] = want
            return s
        except Exception as exc:                  # noqa: BLE001
            if want is None:
                raise
            print(f"[audio] output {want!r} did not open: {exc}")
            print("[audio] outputs on this machine:")
            for i, name, api, ch in output_devices():
                print(f"          {i:3d}  {name}  [{api}]  {ch} ch")
            print("[audio] falling back to the system default")
            s = open_output_stream(cb, sr=SR, channels=2)
            _req['device'] = None
            _opened['device'] = None
            return s

    # pace=True: with no device at all the synth still advances at real time (the
    # bench has always done this) instead of stalling after one look-ahead.
    host = AudioHost(int(CHUNK_S * SR), 2, sr=SR, lookahead=AUDIO_LOOKAHEAD_CHUNKS,
                     output_factory=_open_out, pace=True)
    _ur = {'prev': 0}                          # last underrun count shown in the UI
    # Where the render thread is, in output samples (it writes, the UI thread
    # reads).  A recorded frame stores it, so an event the UI logs per frame --
    # a volume drag, a painted cell -- lands on a real sample when the session is
    # turned into a scene (casynth_session.scene_from_session), instead of being
    # guessed from frame durations.
    _pos = {'cum': 0}
    # What one block of sound COSTS, measured on the render thread (B2): the
    # smoothed and the worst milliseconds per block against the CHUNK_S budget.
    # The UI shows it next to the meter -- an instrument that cannot hold real
    # time must say so out loud, not hide it in an underrun counter.
    budget = {'ms': 0.0, 'max': 0.0, 'underruns': 0, 'lookahead': AUDIO_LOOKAHEAD_CHUNKS}
    # CASYNTH_DEBUG_AUDIO=1: one line a second from the render thread saying what
    # the synth thinks it is doing.  "No sound" has half a dozen causes and they
    # look identical from outside -- this says WHICH one, without guessing.
    _dbg = bool(os.environ.get('CASYNTH_DEBUG_AUDIO'))
    audio_ok = False
    _device = {'opened': False}

    def _open_device():
        """Open the output once the first frame has been drawn and the render
        thread has something to put in the ring."""
        ok = host.start()
        if ok:
            lat = ('?' if host.latency is None else f"{host.latency * 1000:.0f}")
            print(f"[audio] playing into: {output_device_name(_opened['device'])}")
            print(f"[audio] latency={lat}ms "
                  f"+ {AUDIO_LOOKAHEAD_CHUNKS} chunk look-ahead "
                  f"({AUDIO_LOOKAHEAD_CHUNKS*CHUNK_S*1000:.0f}ms)"
                  f"   (CASYNTH_AUDIO_DEVICE=<index|name> picks another output)")
        else:
            print(f"[audio disabled: {host.device_error}] - visuals will still run")
        return ok

    W = GRID_W * CELL
    H = GRID_H * CELL + TOOLBAR_H + PIANO_H

    screen = pygame.display.set_mode((W + SIDEBAR_W, H))
    pygame.display.set_caption("CASynth — multi-engine")
    font  = pygame.font.SysFont("consolas,menlo,monospace", 16)
    small = pygame.font.SysFont("consolas,menlo,monospace", 13)
    clock = pygame.time.Clock()

    # The engines this instrument offers: the ones that can play a NOTE.  An
    # engine whose pitch is a delay length or a scan speed refuses a transpose
    # (casynth_engines/engine_api.py), and an instrument with a piano must not
    # offer a tab that silently ignores the keyboard.
    playable = [eid for eid in engines.ids() if engines.get(eid).plays_notes]

    grid = np.zeros((GRID_H, GRID_W), np.uint8)
    # exc_field: per-generation event excitation (events_field output).
    # Updated ONLY on step() (both auto and manual); held between steps.
    # None until the first step -> analyse falls back to static deg (dyn path).
    exc_field = None

    # Every change of the field gets a SERIAL.  The render thread owns the engine
    # and calls update_field exactly once per new serial, at a block boundary --
    # state['gen'] cannot serve for this, because Clear resets it to 0 and the
    # engine would then miss the change.  The same number is column 5 of the frame
    # log, so a recorded session can be placed on it later.
    _serial = {'n': 0}

    def _touched():
        _serial['n'] += 1

    # The analysis the VISUALS are drawn from, kept until the field or a knob
    # moves (the sound has its own, inside the engine on the render thread).
    _vis = {'key': None, 'out': (None, [], None)}

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
                   # midi_onsets: AUTHORITATIVE sample-accurate timing log written by
                   # the render thread -- (cum_samples, note, gate) on every change.
                   # onset_time = cum_samples / SR exactly -> measures jitter directly
                   # (read side: scratchpad analyse script).  This is the read+write
                   # log path for the threaded render's note timing.
                   midi_onsets=[],
                   # midi_in: device-arrival timestamps (perf_counter, note, gate)
                   # from the MIDI thread -- ground-truth INPUT timing, independent
                   # of rendering.  Comparing its IOIs to midi_onsets' IOIs separates
                   # input jitter (USB-MIDI / arp / swing) from render jitter.
                   midi_in=[],
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
        # CASYNTH_ENGINE preselects a tab (a test hook, like CASYNTH_DUMPFRAME):
        # it is how a headless run measures what one engine costs per block.
        engine=(os.environ.get('CASYNTH_ENGINE') if os.environ.get('CASYNTH_ENGINE') in playable
                else ENGINES[0]['id']),
        engine_params={eid: engines.defaults(eid) for eid in playable},
        # GEN ADSR: per-mode envelope on the AUTOMATON clock; A/D/R are FRACTIONS of
        # one step interval (see casynth_config), S is a 0..1 level.
        gen_attack=GEN_ATTACK_DEFAULT,
        gen_decay=GEN_DECAY_DEFAULT,
        gen_sustain=GEN_SUSTAIN_DEFAULT,
        gen_release=GEN_RELEASE_DEFAULT,
        # GEN amp-slew: when on, per-mode AMPLITUDE changes on a STABLE pitch are
        # smoothed (rise over gen_attack, fall over gen_release) -- kills the shape>0
        # beep where a fixed-frequency mode gates its amplitude and the freq-onset
        # envelope never sees it.  OFF by default = bit-exact; toggle in the GEN
        # header.  (Off for dyn's snappy transients; on for smooth shape>0 timbres.)
        gen_amp_slew=False,
        # VOICE ADSR (note-on/off VCA over the summed signal -- see casynth_config).
        # Defaults (A=0,S=1) make it a no-op multiplier (1.0) while a note is held.
        voice_attack_ms=VOICE_ATTACK_MS_DEFAULT,
        voice_decay_ms=VOICE_DECAY_MS_DEFAULT,
        voice_sustain=VOICE_SUSTAIN_DEFAULT,
        voice_release_ms=VOICE_RELEASE_MS_DEFAULT,
        # Sethares tuning: tune∈[0,1] magnets the note-on carrier toward the nearest
        # dissonance minimum of the current colony spectrum (0 = transparent, bit-exact).
        # tuned_f0: actual sounding carrier in Hz (snapped when tune>0); pre-seeded to
        # the default note so the session log always has a valid float value.
        tune=0.0,
        tuned_f0=midi_to_freq(NOTE_DEFAULT),
        # Gate: note-on/off for the VOICE VCA (True = attack/sustain, False =
        # release).  The oscillator (KA field) is free-running -- gate does NOT empty
        # the voice pool; it only articulates the master VCA (see _render_loop).
        # It goes down when the LAST thing holding a note lets go: a computer key,
        # the mouse on the piano, a MIDI note-off, the file player.  It starts up,
        # so a field sounds the moment the synth opens.
        gate=True,
        # ONSET counter (2026-09-21): every note-on bumps it, whoever struck the
        # note -- keyboard, mouse, MIDI device, file player.  The render thread
        # watches it and re-starts the VOICE attack, because a melody is legato:
        # the gate almost never comes up between two notes, and an envelope that
        # only fires on a gate EDGE never fires at all while a line is played.
        onset=0,
        # HOLD (2026-09-21): while on, a note can only be REPLACED, never released
        # -- the latch the prototype has always had.  Off, the VOICE release runs
        # when the key comes up, which is the only way to hear that block at all
        # from the keyboard or the mouse.
        voice_hold=False,
        midi_held=[],   # stack of currently held MIDI notes (last-note priority)
        midi_port=None, # name of currently open MIDI port
    )

    midi_in = MidiInput()
    _midi_dropdown_open = False
    _audio_dropdown_open = False

    # ── MIDI-file player: reads a .mid and drives the carrier note over time ──
    # Reuses the entire live-MIDI audio path (writes the same state['note']/gate
    # the render thread samples).  Poly->mono reduction = HIGHEST held note
    # (melody); the timbre still comes from the live CA field, not the file.
    midifile = MidiFilePlayer()
    _mf_held = []   # notes currently sounding from the file (for the max() pick)

    # ── who is holding a note right now ───────────────────────────────────────
    # Four things can: a computer key, the mouse on the on-screen piano, a MIDI
    # device (state['midi_held']) and the file player (_mf_held).  The gate goes
    # down only when the LAST of them lets go -- and never while HOLD is lit,
    # where a note can be replaced but not released.
    _kb_held = []        # piano keys physically down, oldest first (last-note priority)
    _mouse_piano = {'on': False}

    def _release_gate():
        """Nothing holds a note any more -> note-off (unless HOLD latches it)."""
        if state['voice_hold']:
            return
        if _kb_held or _mouse_piano['on'] or state['midi_held'] or _mf_held:
            return
        state['gate'] = False

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
    # Right column (former sidebar area): VOICE ADSR → GEN ADSR → Tune → meter → vol.
    _rc_x        = W + 8                          # label left edge
    _RC_LABEL_W  = 24                             # width of label area
    _rc_track_x  = _rc_x + _RC_LABEL_W + 4       # track left = W + 36
    _RC_TRACK_W  = 100                            # track width (value label at W+142, 50px to edge)
    _vol_section_y = by + 228                     # starts 4px below T(tune)-track (by+216+8+4)
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

    # ── MIDI device selector bar (above engine tabs) ─────────────────────────
    _midi_bar_y  = GRID_H * CELL + TOOLBAR_H - 48   # 4px below controls end
    _MIDI_BTN_W  = 260
    _MIDI_DD_ITH = 18   # dropdown item height
    _midi_btn    = pygame.Rect(12, _midi_bar_y, _MIDI_BTN_W, 20)
    # "Play MIDI file" toggle button, just right of the device selector.
    _mf_btn      = pygame.Rect(_midi_btn.right + 8, _midi_bar_y, 168, 20)
    # OUTPUT selector, right of it: which device the sound actually goes to.  A
    # machine can have a headset, a monitor and onboard speakers at once, and the
    # system default may accept the stream and play nothing audible -- so this is
    # the player's choice, visible without opening a console.
    _AUDIO_BTN_W = 300
    _audio_btn   = pygame.Rect(_mf_btn.right + 8, _midi_bar_y, _AUDIO_BTN_W, 20)

    # ── Engine selector: a row of TABS in the toolbar's bottom strip ──────────
    # Built from the SHARED ENGINE REGISTRY (casynth_engines) since 2026-09-21:
    # the prototype hosts the bench's engine instances, so Objects, Laplace waves
    # and Laplace FM are played here as they are, without a line of their sound
    # code being copied.  A click switches state['engine'] and rebuilds the knob
    # panel for that engine.
    _tab_y = GRID_H * CELL + TOOLBAR_H - 24
    engine_tabs = []
    _tx = 12
    for _eid in playable:
        _lbl = engines.label(_eid)
        _tw = small.size(_lbl)[0] + 18
        engine_tabs.append(dict(id=_eid, label=_lbl,
                                rect=pygame.Rect(_tx, _tab_y, _tw, 20)))
        _tx += _tw + 4

    # Live knobs -- mouse-drag sliders to the right of the Lib button, in TWO
    # stacked groups: the ACTIVE engine's attributes (top, DYNAMIC per engine via
    # rebuild_ctrls) and the synth-wide envelope blocks below (fixed position, so
    # they read as stable separate blocks independent of engine): VOICE ADSR (VCA,
    # note-on/off) over GEN ADSR (per-mode, automaton clock) + a Tune row.
    # Engine-attribute sliders write state['engine_params'][engine]; VOICE sliders
    # write state['voice_*'], GEN sliders write state['gen_*'].
    CTRL_LABEL_W, CTRL_TRACK_W = 56, 120
    _ctrl_x  = _pat_btn.right + 16
    _ctrl_y0 = by + 2
    _CTRL_ROW_H = 22
    # Column B (2026-09-21): engines with many settings (Objects has 17) do not fit
    # one column.  Measured geometry of this window (1232 x 1022): column A starts
    # at _ctrl_x = 666, the right column at _rc_x = 1048, the MIDI bar at y = 910
    # -> 13 rows per column and a 172 px gap between the columns.  Column A keeps
    # its pixels, so the four harmonic engines look exactly as before.
    _ctrl2_x       = _ctrl_x + 218                # 884
    CTRL2_LABEL_W  = 40                           # track 924..996
    CTRL2_TRACK_W  = 72                           # value 1002..1044, 4 px to _rc_x
    _CTRL_ROWS_MAX = 13
    # Right column, top to bottom: VOICE ADSR block, GEN ADSR block, Tune row.
    # VOICE header at _ctrl_y0; its 4 knobs at _VOICE_RC_Y0 + row*22.
    # GEN header at by+110; its 4 knobs at _GEN_RC_Y0 + row*22; Tune one row below.
    _VOICE_HDR_RC_Y = _ctrl_y0
    _VOICE_RC_Y0    = _ctrl_y0 + 18
    _GEN_HDR_RC_Y   = by + 110
    _GEN_RC_Y0      = _GEN_HDR_RC_Y + 18
    # Click-toggle for GEN amp-slew, tucked to the right of the "GEN" header text
    # (no extra knob row -> keeps the tight right-column geometry unchanged).
    _slew_btn = pygame.Rect(_rc_x + 34, _GEN_HDR_RC_Y - 1, 46, 15)
    # The same trick in the VOICE header: HOLD.  With it lit a note LATCHES -- it
    # can only be replaced by another note, never let go -- which is how the
    # prototype has always behaved (a field sounds while you work on it).  With it
    # dark, letting go of a key IS a note-off: the VOICE release finally runs, and
    # the block below it stops being decoration.
    _hold_btn = pygame.Rect(_rc_x + 44, _VOICE_HDR_RC_Y - 1, 46, 15)
    ctrls = []

    def _fmt_for(integer, is_ms):
        if is_ms:
            return lambda v: f"{v:.0f}ms"
        if integer:
            return lambda v: f"{int(v)}"
        return lambda v: f"{v:.2f}"

    def _word_rects(words, tr, right):
        """Buttons of a mode selector: each sized to ITS word, left to right from
        the track, using the room up to `right` (a selector needs no value column).
        They are squeezed proportionally only if the words do not fit."""
        gap, x, out = 4, tr.left, []
        for v, word in enumerate(words):
            w = small.size(word)[0] + 10
            out.append((v, pygame.Rect(x, tr.centery - 8, w, 16)))
            x += w + gap
        if x - gap > right:
            room = right - tr.left - gap * (len(out) - 1)
            k = room / float(max(1, sum(r.width for _v, r in out)))
            x = tr.left
            for i, (v, r) in enumerate(out):
                w = max(16, int(r.width * k))
                out[i] = (v, pygame.Rect(x, r.top, w, r.height))
                x += w + gap
        return out

    def rebuild_ctrls():
        """Repopulate `ctrls`: the active engine's parameters (top, two columns)
        + the synth-wide ADSR envelope blocks A/D/S/R (fixed position below -- see
        the _VOICE_/_GEN_ geometry).

        Which widget a parameter gets is decided in casynth_panel, the same call
        the bench makes: a word row for a named-choice parameter, a pill for an
        on/off integer, a slider otherwise (over the user's sub-range when the
        parameter has one).  Parameters that do not act for the current settings
        are left OUT of the panel entirely (the bench shows them as text instead).

        Reading order (user's decision 2026-09-21: "hide the inactive ones,
        compact, the most shared parameters first"): the settings engines have in
        common (the seven of the old Laplace) first, so switching engines moves as
        few rows as possible; then the mode selectors, ahead of the parameters
        whose activity they decide, so changing one moves rows BELOW it and not
        under the cursor; then the engine's own rest."""
        ctrls.clear()
        spec = engines.get(state['engine'])
        params = state['engine_params'][state['engine']]
        order = shared_first(spec, engines.SPECTRUM_KEYS)
        shared = [a for a in order if a in engines.SPECTRUM_KEYS]
        picks = [a for a in order if a not in shared and a in spec.choices]
        order = shared + picks + [a for a in order if a not in shared and a not in picks]
        rows = [r for r in panel_rows(spec, params, order=order)
                if r.kind != KIND_INACTIVE][:2 * _CTRL_ROWS_MAX]
        two_columns = len(rows) > _CTRL_ROWS_MAX
        for i, r in enumerate(rows):
            col = i // _CTRL_ROWS_MAX
            lx = _ctrl_x if col == 0 else _ctrl2_x
            lw = CTRL_LABEL_W if col == 0 else CTRL2_LABEL_W
            tw = CTRL_TRACK_W if col == 0 else CTRL2_TRACK_W
            track = pygame.Rect(lx + lw, _ctrl_y0 + (i % _CTRL_ROWS_MAX) * _CTRL_ROW_H,
                                tw, 8)
            c = dict(id=r.arg, label=r.label, lo=r.lo, hi=r.hi, integer=r.integer,
                     scope='engine', block='engine', kind=r.kind,
                     fmt=_fmt_for(r.integer, False), label_x=lx, track=track,
                     choices=r.choices, hit=track.inflate(0, 14))
            if r.kind == KIND_TOGGLE:
                c['pill'] = pygame.Rect(track.left, track.centery - 8, 34, 16)
                c['hit'] = c['pill'].inflate(4, 4)
            elif r.kind == KIND_CHOICES:
                # a selector may use the room the value column would take; in two
                # columns it must stop before column B
                right = (_ctrl2_x - 8) if (two_columns and col == 0) else (_rc_x - 4)
                c['word_rects'] = _word_rects(r.choices, track, right)
                c['hit'] = c['word_rects'][0][1].unionall(
                    [rr for _v, rr in c['word_rects'][1:]]).inflate(4, 4)
            ctrls.append(c)
        # Right column, synth-wide.  Two ADSR blocks (A/D/R durations, S levels):
        #   VOICE = note-on/off VCA over the summed signal (classic articulation);
        #   GEN   = per-mode envelope on the AUTOMATON clock (the oscillator's life).
        # Then a lone Tune row (Sethares carrier magnet).  block= tags drive the
        # section headers drawn in casynth_ui.
        voice_specs = [
            ('voice_attack_ms',  'A', VOICE_ATTACK_MS_MIN,  VOICE_ATTACK_MS_MAX,  True),
            ('voice_decay_ms',   'D', VOICE_DECAY_MS_MIN,   VOICE_DECAY_MS_MAX,   True),
            ('voice_sustain',    'S', VOICE_SUSTAIN_MIN,    VOICE_SUSTAIN_MAX,    False),
            ('voice_release_ms', 'R', VOICE_RELEASE_MS_MIN, VOICE_RELEASE_MS_MAX, True),
        ]
        for i, (arg, lbl, lo, hi, is_ms) in enumerate(voice_specs):
            _t = pygame.Rect(_rc_track_x, _VOICE_RC_Y0 + i * _CTRL_ROW_H, _RC_TRACK_W, 8)
            ctrls.append(dict(
                id=arg, label=lbl, lo=lo, hi=hi, integer=False, scope='synth',
                block='voice', kind=KIND_SLIDER, fmt=_fmt_for(False, is_ms),
                label_x=_rc_x,
                track=_t, hit=_t.inflate(0, 14)))
        # GEN A/D/R are fractions of a tick (dimensionless), S a 0..1 level.
        # An engine with envelopes of ITS OWN does not read them (Objects has a
        # decay law instead), and the registry says so before an instance exists:
        # then the block is shown as inactive text, the way the bench shows a
        # parameter that does not act, instead of four knobs that do nothing.
        gen_specs = [
            ('gen_attack',  'A', GEN_FRAC_MIN, GEN_FRAC_MAX),
            ('gen_decay',   'D', GEN_FRAC_MIN, GEN_FRAC_MAX),
            ('gen_sustain', 'S', SUSTAIN_MIN,  SUSTAIN_MAX),
            ('gen_release', 'R', GEN_FRAC_MIN, GEN_FRAC_MAX),
        ]
        _gen_live = spec.gen_envelope
        for i, (arg, lbl, lo, hi) in enumerate(gen_specs):
            _t = pygame.Rect(_rc_track_x, _GEN_RC_Y0 + i * _CTRL_ROW_H, _RC_TRACK_W, 8)
            ctrls.append(dict(
                id=arg, label=lbl, lo=lo, hi=hi, integer=False, scope='synth',
                block='gen', kind=(KIND_SLIDER if _gen_live else KIND_INACTIVE),
                text=('' if _gen_live else "engine's own"),
                fmt=_fmt_for(False, False), label_x=_rc_x,
                track=_t, hit=(_t.inflate(0, 14) if _gen_live else pygame.Rect(0, 0, 0, 0))))
        # Tune: one row below the GEN block.
        _t = pygame.Rect(_rc_track_x, _GEN_RC_Y0 + 4 * _CTRL_ROW_H, _RC_TRACK_W, 8)
        ctrls.append(dict(
            id='tune', label='T', lo=0.0, hi=1.0, integer=False, scope='synth',
            block='tune', kind=KIND_SLIDER, fmt=_fmt_for(False, False),
            label_x=_rc_x,
            track=_t, hit=_t.inflate(0, 14)))

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

    # Spectrum strip: engine mode bars (voices' freqs/amps).  Lives in the wide
    # empty band on the LEFT of the toolbar -- below the legend/status row, above
    # the MIDI bar, and stopping short of the engine-knob column (_ctrl_x).  This
    # is roomier than the old right-side pocket and frees that area up.  Read-only
    # viz -> no knob, no session-log path.
    _spec_top = info_y + 40
    _spec_rect = pygame.Rect(legend_x, _spec_top,
                             (_ctrl_x - 12) - legend_x,
                             (_midi_bar_y - 6) - _spec_top)

    piano_top = GRID_H * CELL + TOOLBAR_H
    white_keys, black_keys = _make_piano(state['kb_base'], piano_top, W)

    paint = None
    dragging_vol = False
    dragging_bpm = False
    dragging_ctrl = None          # id of the timbre knob being dragged, or None
    drag = {'active': False, 'cells': [], 'name': '', 'snap': None}
    _ghost = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
    _ghost.fill((111, 208, 224, 110))

    # Static layout geometry bundled for the renderer (casynth_ui.draw_frame).
    # These are the SAME objects the event loop mutates/reads -> no divergence
    # (ctrls is rebuilt in place by rebuild_ctrls; this reference still tracks it).
    lay = SimpleNamespace(
        W=W, buttons=buttons, pat_btn=_pat_btn, bpm_x=_bpm_x, bpm_track=bpm_track,
        div_buttons=div_buttons, legend_x=legend_x, info_y=info_y, rc_x=_rc_x,
        vol_section_y=_vol_section_y, vol_track=vol_track, rc_track_x=_rc_track_x,
        rc_track_w=_RC_TRACK_W, voice_hdr_rc_y=_VOICE_HDR_RC_Y,
        gen_hdr_rc_y=_GEN_HDR_RC_Y, slew_btn=_slew_btn, hold_btn=_hold_btn,
        ctrls=ctrls,
        meter_track=meter_track, midi_btn=_midi_btn, midi_btn_w=_MIDI_BTN_W,
        mf_btn=_mf_btn,
        midi_dd_ith=_MIDI_DD_ITH, audio_btn=_audio_btn, audio_btn_w=_AUDIO_BTN_W,
        engine_tabs=engine_tabs, sb_items=_sb_items,
        sb_content_h=_sb_content_h, sb_scroll_min=_sb_scroll_min,
        spec_rect=_spec_rect)

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
        # None once a mode change has hidden that row (rebuild_ctrls drops the
        # parameters that do not act) -- a drag in flight simply ends
        return next((c for c in ctrls if c['id'] == cid), None)

    def ctrl_value(c):
        """Current value of a knob: engine attributes live in the active engine's
        per-engine dict; the release knob lives directly in state (synth-wide)."""
        if c['scope'] == 'engine':
            return state['engine_params'][state['engine']][c['id']]
        return state[c['id']]

    def put_ctrl(c, val):
        if c['scope'] == 'engine':
            state['engine_params'][state['engine']][c['id']] = val
        else:
            state[c['id']] = val

    def set_ctrl(c, mx):
        """Drag / click at x: a slider reads the fraction of its track, a mode
        selector the word under the cursor (dragging across the words keeps
        picking, so the rows below it move, not the cursor)."""
        if c.get('kind') == KIND_CHOICES:
            val = c['word_rects'][-1][0]
            for v, rect in c['word_rects']:
                if mx < rect.right:
                    val = v
                    break
            put_ctrl(c, int(val))
            rebuild_ctrls()            # a mode decides which rows are shown at all
            return
        frac = float(np.clip((mx - c['track'].left) / c['track'].width, 0, 1))
        val = c['lo'] + frac * (c['hi'] - c['lo'])
        val = int(round(val)) if c.get('integer') else val
        put_ctrl(c, val)

    def toggle_ctrl(c):
        put_ctrl(c, 0 if ctrl_value(c) else 1)
        rebuild_ctrls()                # an on/off setting may hide other rows too

    def do(bid):
        nonlocal grid, exc_field
        if bid == "play":
            state['run'] = not state['run']
            if state['run']:
                # Schedule first step one interval from now so the user hears the
                # current generation first, then the clock starts ticking.
                state['next_step_time'] = time.perf_counter() + _step_interval()
        elif bid == "step":
            prev = grid.copy()
            grid = step(grid)
            exc_field = events_field(prev, grid)
            state['gen'] += 1
            _touched()
            # FIX-A: record button-triggered steps with true prev_grid so replay
            # can reconstruct exc_field exactly (button steps were previously missing).
            if rec is not None:
                rec['steps'].append((state['gen'], grid.copy(), state['note'], prev))
        elif bid == "random":
            grid[:] = (np.random.random((GRID_H, GRID_W)) < RANDOM_DENSITY).astype(np.uint8)
            _touched()
        elif bid == "clear":
            grid[:] = 0
            state['gen'] = 0
            _touched()

    def hit_piano(pos):
        for rect, m in black_keys:
            if rect.collidepoint(pos):
                return m
        for rect, m in white_keys:
            if rect.collidepoint(pos):
                return m
        return None

    # ── Audio render thread + MIDI input thread (decoupled from the 60 fps loop) ──
    # The MAIN thread publishes the current render spec (voices analysed at the fixed
    # REFERENCE f0 -- a PITCH-NORMALIZED oscillator -- plus the live knob values); the
    # RENDER thread renders sub-chunks just-in-time, applying the live carrier as a
    # scalar TRANSPOSE at render (phase-continuous, no per-mode retrigger) and the
    # live gate; the MIDI thread polls the device continuously.  This takes the MIDI
    # poll and the look-ahead off the timing path -> onsets quantise to one CHUNK_S
    # sub-chunk locked to the playback clock, not to the frame rate.
    #
    # Target model (decisions.md 2026-07-06): the KA field is a free-running
    # oscillator whose spectrum is analysed ONCE at REFERENCE_F0.  A note change is a
    # pure transpose of that oscillator (freq × ratio in render_chunk_laplacian), NOT
    # a change of the pool's stored frequencies -- so SlotPool no longer sees the note
    # as a per-mode frequency change (no spurious attack retrigger / tail spawn on
    # every note).  This is what fixes the long-note "hang" and the dense-MIDI churn.
    REFERENCE_F0 = midi_to_freq(NOTE_DEFAULT)   # pitch-normalized analysis anchor
    render_spec = {'cur': None}        # atomic publish point (single ref swap)
    audio_ctl   = {'alive': True}      # threads exit when False

    def _transpose(note):
        """Carrier transpose ratio: sounding f0 / REFERENCE_F0.

        Applied at render as a scalar multiply on every slot's stored frequency
        (see render_chunk_laplacian).  When tune>0 and a snapped carrier is set the
        sounding f0 is state['tuned_f0'] (Sethares-snapped), else midi_to_freq(note).
        On the default note the ratio is 1.0 -> bit-identical to the historical sound.
        """
        tf = state['tuned_f0']
        target_f0 = (tf if (state['tune'] > 0.0 and tf is not None and tf > 0.0)
                     else midi_to_freq(note))
        return (target_f0 / REFERENCE_F0) if REFERENCE_F0 > 0 else 1.0

    # Note-on tracking for Sethares snap (per-frame polling in the main loop).
    # Initialised to the default note so the first frame doesn't trigger snap
    # (tuned_f0 is already pre-seeded to midi_to_freq(NOTE_DEFAULT) in state).
    _last_note_for_snap = [state['note']]   # list wrapper for nonlocal-free mutation

    def _snap_note_on(new_note):
        """Compute state['tuned_f0'] for new_note via Sethares snap.

        Called on every note-on from the main-loop poller (≤1 frame latency).
        Reads render_spec['cur'] for the current colony spectrum (the previous
        frame's analyse() result).  When tune=0 or no prior reference exists the
        raw MIDI frequency is used (transparent / no snap).

        The computation (dissonance_curve + scale_minima) takes ≈1–10 ms for
        N≤24 partials with R=1300 steps — acceptable as a one-shot note-on cost.
        """
        f_raw = midi_to_freq(new_note)
        tune  = state['tune']

        if tune <= 0.0:
            state['tuned_f0'] = f_raw        # no snap; keep reference fresh
            return

        prev_f0 = state['tuned_f0']
        if prev_f0 is None or prev_f0 <= 0.0:
            state['tuned_f0'] = f_raw        # first note: no prior reference
            return

        # Build colony spectrum: all live-voice partials rescaled to prev_f0.
        spec = render_spec['cur']
        if spec is None or not spec['voices']:
            state['tuned_f0'] = f_raw
            return

        base_f0 = spec['base_f0']
        rescale = (prev_f0 / base_f0) if base_f0 > 0.0 else 1.0
        all_f, all_a = [], []
        for voice in spec['voices']:
            for f, a in zip(voice['freqs'], voice['amps']):
                if a > 0.0:
                    all_f.append(f * rescale)
                    all_a.append(float(a))

        if not all_f:
            state['tuned_f0'] = f_raw
            return

        colony_f = np.array(all_f)
        colony_a = np.array(all_a)

        # Top TUNE_MAX_PARTIALS by amplitude
        if len(colony_a) > TUNE_MAX_PARTIALS:
            idx      = np.argpartition(colony_a, -TUNE_MAX_PARTIALS)[-TUNE_MAX_PARTIALS:]
            colony_f = colony_f[idx]
            colony_a = colony_a[idx]

        # Compute dissonance curve, find minima, snap
        ratios_arr, curve_vals = dissonance_curve(colony_f, colony_a)
        minima                 = scale_minima(curve_vals, ratios_arr)
        r_raw                  = f_raw / prev_f0
        r_snapped              = snap_ratio(r_raw, minima, tune)
        state['tuned_f0']      = prev_f0 * r_snapped

    # -- the engine the render thread hosts ------------------------------------
    # The prototype has no slot pool of its own any more: it holds an INSTANCE of
    # a casynth_engines SoundEngine, exactly as the bench does, so Objects,
    # Laplace waves and Laplace FM are played here AS THEY ARE and not one line
    # of their sound code is copied (memory/req-seam-2026-09-21.md).
    _SILENT = np.zeros((int(CHUNK_S * SR), 2), np.int16)
    engine_q = _queue.Queue()          # UI thread -> render thread: a ready engine
    _handed = {'id': None}
    _XFADE_BLOCKS = max(1, round(ENGINE_XFADE_MS / 1000.0 / CHUNK_S))

    def _engine_ctx():
        """The render context.  f0 is the ANALYSIS ANCHOR -- the sounding note is
        a transpose at render -- and the pan is the prototype's own: a voice sits
        where its figure sits (the bench asks for the centre instead)."""
        return EngineContext(SR, int(CHUNK_S * SR), 2, REFERENCE_F0, 1.0,
                             1.0 / _step_interval(), pan=PAN_FIELD_MODE)

    def _engine_tell(e):
        """The live envelope knobs and the live tempo, on a block boundary.  An
        engine with envelopes of its own ignores both (the contract)."""
        e.set_rate(1.0 / _step_interval())
        e.set_envelope(state['gen_attack'], state['gen_decay'], state['gen_sustain'],
                       state['gen_release'], state['gen_amp_slew'])

    def _engine_new(eid, spec, params, gain):
        e = engines.create(eid, _engine_ctx(), dict(params))
        _engine_tell(e)
        e.init(spec['grid'], spec['exc'], gain)
        return e

    def _hand_engine():
        """Build the engine the user picked HERE, on the UI thread, and hand it
        over ready.  Building one can cost a fifth of a second -- Objects compiles
        its kernel the first time it renders -- and that must never happen inside
        a block the device is waiting for.  After the handoff the UI thread never
        touches the instance again; the render thread owns it."""
        spec = render_spec['cur']
        if spec is None:
            return
        eid = state['engine']
        params = dict(state['engine_params'][eid])
        try:
            e = _engine_new(eid, spec, params, MASTER_GAIN * state['vol'])
        except Exception as exc:                 # noqa: BLE001
            print(f"[engine {eid} unavailable: {exc}]")
            return
        _handed['id'] = eid
        engine_q.put((eid, e, spec['serial'], params))

    def _render_block(eng, spec, gain, t_samples, transpose):
        """One block from the hosted engine.  The field, the excitation and the
        knobs come from the snapshot the UI thread published, at a BLOCK boundary
        and never mid-block; update_field runs exactly once per new serial.
        Switching engine under a held note crossfades: both instances render
        while the old one fades out, so the switch has no step in it."""
        try:                                   # a ready engine from the UI thread
            eid, obj, serial, params = engine_q.get_nowait()
        except _queue.Empty:
            pass
        else:
            if eng['obj'] is not None:
                eng['old'] = eng['obj']
                eng['fade'] = _XFADE_BLOCKS
            eng['obj'], eng['id'] = obj, eid
            eng['serial'], eng['params'] = serial, params
            host.lookahead = AUDIO_LOOKAHEAD_CHUNKS     # the new engine earns its own
            budget['lookahead'] = host.lookahead
            budget['max'] = 0.0
        if spec is None or eng['obj'] is None:   # nothing to play yet: silence
            return _SILENT.copy(), 0.0, 0
        if eng['serial'] != spec['serial']:
            eng['obj'].update_field(spec['grid'], spec['exc'])
            eng['serial'] = spec['serial']
        if eng['params'] != spec['params'] and eng['id'] == spec['engine']:
            eng['obj'].set_params(dict(spec['params']))
            eng['params'] = dict(spec['params'])
        _engine_tell(eng['obj'])
        buf, peak, n_clip = eng['obj'].render(gain, t_samples, transpose=transpose)
        if eng['fade'] > 0 and eng['old'] is not None:
            old, o_peak, o_clip = eng['old'].render(gain, t_samples, transpose=transpose)
            k = _XFADE_BLOCKS - eng['fade']
            x = (np.linspace(k, k + 1, len(buf), endpoint=False) / _XFADE_BLOCKS)[:, None]
            buf = np.rint(old.astype(np.float64) * (1.0 - x)
                          + buf.astype(np.float64) * x).astype(np.int16)
            peak = max(peak, o_peak)
            n_clip += o_clip
            eng['fade'] -= 1
            if eng['fade'] == 0:
                eng['old'] = None
        return buf, peak, n_clip

    def _render_loop():
        cum = 0                        # cumulative output samples (onset timestamps)
        eng = {'id': None, 'obj': None, 'params': None, 'serial': None,
               'old': None, 'fade': 0}
        last_note, last_gate = state['note'], bool(state['gate'])
        # VOICE ADSR (VCA): a scalar 0..1 envelope keyed to note-on/off, multiplying
        # the master gain (a classic articulation over the summed oscillator signal).
        # The arithmetic lives in casynth_engine.VoiceEnvelope since 2026-09-21 --
        # the bench articulates a scene's notes with the same class, not a copy.
        venv = VoiceEnvelope(gate=last_gate)
        last_onset = last_rec_onset = state['onset']
        while audio_ctl['alive']:
            if host.full():
                time.sleep(0.001)      # ring full -> idle briefly
                continue
            spec = render_spec['cur']
            note = state['note']
            gate = bool(state['gate'])
            # GEN envelope rides the AUTOMATON clock: A/D/R are fractions of one
            # step interval, so the per-mode texture scales with tempo (BPM/division).
            # ── VOICE ADSR (VCA): advance one chunk, fold into the master gain ────
            # note-on edge -> (re)trigger attack; note-off edge -> release.  This is
            # the ONLY articulation gate: the oscillator (the CA field) is fed to the
            # engine ALWAYS, so a note change never retriggers a per-mode envelope --
            # it just re-articulates this scalar VCA, which lives OUTSIDE the engine.
            # A STRUCK note re-articulates even under a held legato line: the
            # gate edge is not the only thing that starts this envelope, or a
            # melody (where the gate practically never comes up between notes)
            # would never hear the A/D/R knobs at all.
            _attack_s = state['voice_attack_ms'] / 1000.0
            onset = state['onset']
            if onset != last_onset:
                venv.retrigger(_attack_s)
                last_onset = onset
            level = venv.block(gate,
                               _attack_s,
                               state['voice_decay_ms'] / 1000.0,
                               float(state['voice_sustain']),
                               state['voice_release_ms'] / 1000.0)
            gain = MASTER_GAIN * state['vol'] * level
            # The engine is analysed at a FIXED anchor and the live carrier is a
            # scalar transpose at render: phase accumulates, only the increment
            # moves, so a note is phase-continuous and retriggers nothing.
            transpose = _transpose(note)
            _t0 = time.perf_counter()
            buf, peak, n_clip = _render_block(eng, spec, gain, cum, transpose)
            _dt = (time.perf_counter() - _t0) * 1000.0
            budget['ms'] += (_dt - budget['ms']) * BUDGET_SMOOTH
            budget['max'] = max(budget['max'], _dt)
            # A heavy engine earns a deeper ring instead of dropping out, and it
            # EARNS it by actually dropping out: one more block of look-ahead per
            # underrun, up to AUDIO_LOOKAHEAD_MAX_MS.  A machine that keeps up pays
            # no latency at all, and one that cannot buys exactly as much as it
            # needs -- with the price shown in the toolbar beside the cost.  A new
            # engine starts over from the default.
            if host.underruns > budget['underruns'] and host.lookahead < AUDIO_LOOKAHEAD_MAX_CHUNKS:
                host.lookahead += 1
                budget['lookahead'] = host.lookahead
            budget['underruns'] = host.underruns
            if _dbg and cum // int(CHUNK_S * SR) % 125 == 0:
                _cells = int(spec['grid'].sum()) if spec is not None else -1
                _v = len(spec['voices']) if spec is not None else -1
                print(f"[dbg] engine={eng['id']} obj={'yes' if eng['obj'] else 'NONE'} "
                      f"cells={_cells} voices={_v} gate={int(gate)} note={note} "
                      f"vol={state['vol']:.2f} vca={level:.3f} gain={gain:.4f} "
                      f"peak={peak:.3f} clip={n_clip} ring={host.qsize()}", flush=True)
            host.put(buf)
            meter['peak'] = max(peak, meter['peak'] * METER_DECAY)
            meter['clip'] = n_clip > 0
            if rec is not None:
                rec['chunks'].append(buf)
                # An onset with the SAME note under a held gate changes neither of
                # them, and used to leave no trace at all -- so a scene built from
                # the session re-articulated fewer times than the prototype did.
                if note != last_note or gate != last_gate or onset != last_rec_onset:
                    rec['midi_onsets'].append((cum, int(note), bool(gate)))
                    last_rec_onset = onset
                rec['underruns'] = host.underruns
            last_note, last_gate = note, gate
            cum += len(buf)
            _pos['cum'] = cum

    def _on_midi_message(msg):
        # Called from rtmidi's own thread the instant a message arrives (callback
        # mode -> no poll latency, no GIL-starved poll loop).  Last-note priority;
        # updates the shared live note/gate that the render thread samples.
        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note not in state['midi_held']:
                state['midi_held'].append(msg.note)
            state['note'] = msg.note
            state['gate'] = True
            state['onset'] += 1
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in state['midi_held']:
                state['midi_held'].remove(msg.note)
            if state['midi_held']:
                state['note'] = state['midi_held'][-1]
                state['gate'] = True
            else:
                _release_gate()    # only if nothing else holds a note, and not on HOLD
        else:
            return   # clock/sysex/other: not a note event
        if rec is not None:
            # Ground-truth input timing, stamped the instant rtmidi hands it over.
            rec['midi_in'].append((time.perf_counter(),
                                   int(state['note']), bool(state['gate'])))

    def _on_midifile_note(note, on):
        # Called from the MIDI-file player thread.  Poly->mono: the carrier
        # follows the NEWEST onset (last-note priority), falling back to the
        # highest still-held note on release -- so every struck note re-articulates
        # (the engine re-triggers the ADSR attack on each carrier freq change), and
        # the carrier never parks on a multi-second held note.  Writes the same
        # shared note/gate the render thread samples -> reuses the live-MIDI path.
        if on:
            if note in _mf_held:
                _mf_held.remove(note)
            _mf_held.append(note)           # newest at the end
            state['note'] = note            # last-note priority: jump to the onset
            state['gate'] = True
            state['onset'] += 1
        else:
            if note in _mf_held:
                _mf_held.remove(note)
            if _mf_held:
                state['note'] = max(_mf_held)   # fall back to the highest held
                state['gate'] = True
            else:
                _release_gate()
        if rec is not None:
            rec['midi_in'].append((time.perf_counter(),
                                   int(state['note']), bool(state['gate'])))

    def _start_midifile(path):
        # Load + start.  The timbre needs a LIVE colony (an empty field is
        # silent no matter the pitch), so if nothing is alive we seed a random
        # field and start the simulation -- otherwise "play file" looks broken.
        if not midifile.load(path):
            return
        if not grid.any():
            do("random")
        if not state['run']:
            do("play")
        _mf_held.clear()
        midifile.start(_on_midifile_note)

    def _pick_midifile():
        # Native file dialog (tkinter, stdlib).  Runs on the main thread only.
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Open MIDI file",
                filetypes=[("MIDI files", "*.mid *.midi"),
                           ("All files", "*.*")])
            root.destroy()
            return path or None
        except Exception as exc:
            print(f"[MIDIFILE] file dialog unavailable: {exc}")
            return None

    def _toggle_midifile():
        # One button: playing -> stop (gate off); idle -> pick a file and play.
        if midifile.playing:
            midifile.stop()
            _mf_held.clear()
            _release_gate()
        else:
            path = _pick_midifile()
            if path:
                _start_midifile(path)

    _threads = []
    _t = threading.Thread(target=_render_loop, daemon=True, name='render')
    _t.start(); _threads.append(_t)
    # CLI autoplay: `python gol_synth.py play <file.mid>` loads + starts at boot.
    if autoplay_midi:
        _start_midifile(autoplay_midi)
    # MIDI input is delivered by rtmidi's callback (set when a device is opened);
    # there is no MIDI poll thread.

    # UI regression hook: if CASYNTH_DUMPFRAME is set, render exactly one frame,
    # save it to that path and exit (deterministic on the default empty field) --
    # a bit-exact visual baseline for the draw extraction.
    _dumpframe = os.environ.get('CASYNTH_DUMPFRAME')
    # CASYNTH_RUN_SECONDS: quit by itself after that many seconds -- a headless
    # session (SDL_VIDEODRIVER=dummy, CASYNTH_RECORD=1) can then be recorded and
    # compared against its own offline render without a human pressing Esc.
    _run_seconds = os.environ.get('CASYNTH_RUN_SECONDS')
    _deadline = (time.perf_counter() + float(_run_seconds)) if _run_seconds else None
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Precompute MIDI dropdown items + hit-rects (used in events and draw).
        if _midi_dropdown_open and MIDI_AVAILABLE:
            _midi_dd_items = [None] + midi_in.ports()
        else:
            _midi_dd_items = []
        _midi_dd_rects = [
            pygame.Rect(_midi_btn.left + 2,
                        _midi_btn.bottom + 2 + di * _MIDI_DD_ITH,
                        _MIDI_BTN_W - 4, _MIDI_DD_ITH)
            for di in range(len(_midi_dd_items))
        ]
        # outputs: (device or None = system default, label)
        if _audio_dropdown_open:
            _audio_dd_items = [(None, 'System default')] + [
                (i, f"{i:3d}  {name}  [{api}]") for i, name, api, _ch in output_devices()]
        else:
            _audio_dd_items = []
        _audio_dd_rects = [
            pygame.Rect(_audio_btn.left + 2,
                        _audio_btn.bottom + 2 + di * _MIDI_DD_ITH,
                        _AUDIO_BTN_W - 4, _MIDI_DD_ITH)
            for di in range(len(_audio_dd_items))
        ]

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in _KB_PIANO:
                    if e.key in _kb_held:
                        _kb_held.remove(e.key)
                    _kb_held.append(e.key)          # newest last: last-note priority
                    state['note'] = state['kb_base'] + _KB_PIANO[e.key]
                    state['gate'] = True
                    state['onset'] += 1             # a struck note re-articulates
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

            elif e.type == pygame.KEYUP:
                # A key comes up: fall back to the newest key still down, and if
                # none is, let the note GO -- that release is the VOICE envelope's
                # whole point.  HOLD keeps it latched (see _release_gate).
                if e.key in _KB_PIANO:
                    if e.key in _kb_held:
                        _kb_held.remove(e.key)
                    if _kb_held:
                        state['note'] = state['kb_base'] + _KB_PIANO[_kb_held[-1]]
                        state['gate'] = True
                    else:
                        _release_gate()

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if _audio_dropdown_open:
                    # any click closes it; a click on an item moves the sound there
                    for di, item_rect in enumerate(_audio_dd_rects):
                        if item_rect.collidepoint(e.pos):
                            _req['device'] = _audio_dd_items[di][0]
                            if host.reopen():
                                # CASYNTH_NO_PERSIST: a probe may drive this menu
                                # without overwriting the player's own choice
                                if not os.environ.get('CASYNTH_NO_PERSIST'):
                                    audio_choice_save(_opened['device'])
                                audio_ok = True
                                print(f"[audio] playing into: "
                                      f"{output_device_name(_opened['device'])}")
                            else:
                                audio_ok = False
                                print(f"[audio] {_req['device']!r} did not open: "
                                      f"{host.device_error}")
                            break
                    _audio_dropdown_open = False
                elif _midi_dropdown_open:
                    # Any click closes the dropdown; item click also selects.
                    for di, item_rect in enumerate(_midi_dd_rects):
                        if item_rect.collidepoint(e.pos):
                            port = _midi_dd_items[di]
                            if port is None:
                                midi_in.close()
                                state['midi_port'] = None
                            else:
                                midi_in.open(port, _on_midi_message)
                                state['midi_port'] = midi_in.port_name
                            break
                    _midi_dropdown_open = False
                elif state['sidebar_open'] and e.pos[0] >= W and e.pos[1] < GRID_H * CELL and e.button == 1:
                    for item in _sb_items:
                        if item['rect'].move(0, _sb_scroll).collidepoint(e.pos):
                            drag.update(active=True, cells=item['cells'],
                                        name=item['name'], snap=None)
                            break
                elif not drag['active']:
                    rc = cell_at(*e.pos)
                    if rc is not None and e.button in (1, 3):
                        paint = 1 if e.button == 1 else 0
                        if grid[rc] != paint:
                            grid[rc] = paint
                            _touched()
                    elif e.pos[1] >= piano_top:
                        m = hit_piano(e.pos)
                        if m is not None:
                            state['note'] = m
                            state['gate'] = True
                            state['onset'] += 1
                            _mouse_piano['on'] = True    # held until the button comes up
                    else:
                        tab_hit = next((t for t in engine_tabs
                                        if t['rect'].collidepoint(e.pos)), None)
                        if MIDIFILE_AVAILABLE and _mf_btn.collidepoint(e.pos):
                            _toggle_midifile()
                        elif _audio_btn.collidepoint(e.pos):
                            _audio_dropdown_open = not _audio_dropdown_open
                        elif MIDI_AVAILABLE and _midi_btn.collidepoint(e.pos):
                            _midi_dropdown_open = not _midi_dropdown_open
                        elif tab_hit is not None:
                            if tab_hit['id'] != state['engine']:
                                state['engine'] = tab_hit['id']
                                rebuild_ctrls()   # swap knob panel to new engine
                        elif _pat_btn.collidepoint(e.pos):
                            state['sidebar_open'] = not state['sidebar_open']
                        elif _hold_btn.collidepoint(e.pos):
                            state['voice_hold'] = not state['voice_hold']
                            if not state['voice_hold']:
                                # letting HOLD go lets the latched note go too,
                                # if nothing is actually being held
                                _release_gate()
                        elif (_slew_btn.collidepoint(e.pos)
                              and engines.get(state['engine']).gen_amp_slew):
                            state['gen_amp_slew'] = not state['gen_amp_slew']
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
                        elif any(c['hit'].collidepoint(e.pos) for c in ctrls):
                            for c in ctrls:
                                if not c['hit'].collidepoint(e.pos):
                                    continue
                                if c['kind'] == KIND_TOGGLE:
                                    toggle_ctrl(c)      # a pill flips, it is not dragged
                                else:
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
                        _touched()
                    drag.update(active=False, snap=None, cells=[], name='')
                paint = None
                dragging_vol = False
                dragging_bpm = False
                dragging_ctrl = None
                if _mouse_piano['on']:
                    _mouse_piano['on'] = False       # the mouse leaves the piano
                    _release_gate()

            elif e.type == pygame.MOUSEMOTION:
                if drag['active']:
                    drag['snap'] = cell_at(*e.pos)
                elif paint is not None:
                    rc = cell_at(*e.pos)
                    if rc is not None and grid[rc] != paint:
                        grid[rc] = paint
                        _touched()
                elif dragging_vol:
                    set_vol(e.pos[0])
                elif dragging_bpm:
                    set_bpm(e.pos[0])
                elif dragging_ctrl is not None:
                    _c = ctrl_by_id(dragging_ctrl)
                    if _c is None:
                        dragging_ctrl = None
                    else:
                        set_ctrl(_c, e.pos[0])

            elif e.type == pygame.MOUSEWHEEL:
                if (state['sidebar_open'] and pygame.mouse.get_pos()[0] >= W
                        and pygame.mouse.get_pos()[1] < GRID_H * CELL):
                    _sb_scroll = max(_sb_scroll_min, min(0, _sb_scroll + e.y * 20))

        # MIDI device input arrives via rtmidi's callback (_on_midi_message), not
        # here -- that decouples note timing from the 60 fps frame rate.  The
        # on-screen piano / computer keyboard still set state['note']/gate above.

        # GOL advance — strict BPM-aligned timing via perf_counter deadline.
        # next_step_time advances by exactly one interval each step, so the step
        # sequence lands on a perfect periodic grid regardless of frame-time jitter
        # (no float accumulation: += interval, not = now + interval).
        if state['run'] and state['next_step_time'] is not None:
            interval = _step_interval()
            now = time.perf_counter()
            n_steps = 0
            while now >= state['next_step_time'] and n_steps < 4:
                prev_grid = grid.copy()
                grid = step(grid)
                exc_field = events_field(prev_grid, grid)
                state['gen'] += 1
                _touched()
                state['next_step_time'] += interval
                n_steps += 1
                if rec is not None:
                    # FIX-A: 4-tuple includes true prev_grid so replay can reconstruct
                    # exc_field correctly (prev_grid already a copy from line above).
                    rec['steps'].append((state['gen'], grid.copy(), state['note'],
                                         prev_grid))
            # If the clock drifted far behind (e.g. OS pause), snap forward so we
            # don't spiral trying to catch up.
            if state['next_step_time'] < now - interval:
                state['next_step_time'] = now + interval

        # ── Sethares tuning snap (note-on detection, ≤1 frame latency) ──────────
        # Compare note from this frame to last frame's note.  All three input paths
        # (keyboard, screen piano, MIDI callback) write state['note'], so this poll
        # catches them all with at most one frame of latency.
        _curr_note = state['note']
        if _curr_note != _last_note_for_snap[0]:
            if bool(state['gate']):
                _snap_note_on(_curr_note)
            else:
                # Note changed while gate is off (e.g. released MIDI key):
                # update reference without snapping so the next note-on has a
                # consistent previous carrier.
                state['tuned_f0'] = midi_to_freq(_curr_note)
            _last_note_for_snap[0] = _curr_note

        _ep = state['engine_params'][state['engine']]
        base_f0 = REFERENCE_F0            # analyse the field as a pitch-normalized oscillator
        # The VISUALS (the field's colouring and the spectrum strip) are the
        # Laplace analysis of the figures -- the analysis every hosted engine is
        # built on.  The SOUND no longer comes from here: the render thread's
        # engine instance makes it from the snapshot published below.
        _vis_engine = state['engine'] if state['engine'] in ENGINE_BY_ID else 'laplacian'
        _vis_params = (_ep if _vis_engine != 'laplacian'
                       else {k: _ep[k] for k in engines.SPECTRUM_KEYS if k in _ep})
        # ... and only when something it depends on actually changed.  Before the
        # prototype hosted an engine this ran on EVERY frame, 60 times a second,
        # even on a field standing still; now the render thread analyses for the
        # sound as well, and paying for it twice is what starves the device.
        if _handed['id'] != state['engine']:
            _hand_engine()             # a tab was clicked (or this is the first frame)
        _vis_key = (_serial['n'], _vis_engine, tuple(sorted(_vis_params.items())))
        if _vis['key'] != _vis_key:
            _vis['key'] = _vis_key
            _vis['out'] = analyse(grid, base_f0, _vis_engine, _vis_params, exc=exc_field)
        labels, voices, color = _vis['out']
        # Publish an immutable render spec for the render thread (atomic ref swap).
        # Voices are pitch-normalized (analysed at REFERENCE_F0); the render thread
        # applies the live carrier as a transpose and the live gate, so note timing
        # follows the device, not this frame.
        # Everything the render thread needs, in ONE object swapped atomically:
        # the field and its excitation, the engine and its knobs, and the serial
        # that says whether the field is new.  The render thread never reaches
        # into the UI thread's own variables.
        render_spec['cur'] = {'voices': voices, 'base_f0': base_f0,
                              'grid': grid.copy(), 'exc': exc_field,
                              'serial': _serial['n'], 'engine': state['engine'],
                              'params': dict(_ep)}
        _ur_now = host.underruns
        ur_delta = _ur_now - _ur['prev']
        _ur['prev'] = _ur_now
        if rec is not None:
            # columns: [dt_ms, gen, n_voices, n_rendered, underrun, n_steps_done,
            #           out_samples]
            # n_steps_done = len(rec['steps']) at this moment; used by replay to
            # index step history directly (serial-path) so exc reconstruction is
            # correct even when gen resets to 0 after clear() (FIX-F).
            # out_samples (2026-09-21) = how far the render thread had got when this
            # frame was logged -> the sample a frame-timed event happened at.
            rec['frames'].append((round(dt * 1000.0, 2), state['gen'],
                                  len(voices), 0, ur_delta, len(rec['steps']),
                                  _pos['cum']))
            # Per-frame control snapshot -- CONTEXT for the session (engine/knob
            # timeline).  Audio is no longer rendered per frame, so n_rendered=0; the
            # AUTHORITATIVE sample-accurate note timing is rec['midi_onsets'] written
            # by the render thread.  (Faithful per-frame audio replay is superseded
            # by the threaded render -- see memory/log/2026-06-23-midi-timing-jitter.)
            # tuned_f0 in log: actual sounding carrier Hz (snapped when tune>0).
            # Old recordings without tuned_f0 fall back in replay to midi_to_freq(note).
            _tf = state['tuned_f0']
            rec['replay_controls'].append(dict(
                n_rendered=0,
                engine=state['engine'],
                note=int(state['note']),
                gate=bool(state['gate']),
                vol=float(state['vol']),
                # bpm/div_idx are logged so offline replay can reconstruct the tick
                # interval that GEN A/D/R (fractions of a tick) scale against.
                bpm=float(state['bpm']),
                div_idx=int(state['div_idx']),
                gen_attack=float(state['gen_attack']),
                gen_decay=float(state['gen_decay']),
                gen_sustain=float(state['gen_sustain']),
                gen_release=float(state['gen_release']),
                gen_amp_slew=bool(state['gen_amp_slew']),
                voice_attack_ms=float(state['voice_attack_ms']),
                voice_decay_ms=float(state['voice_decay_ms']),
                voice_sustain=float(state['voice_sustain']),
                voice_release_ms=float(state['voice_release_ms']),
                engine_params=dict(_ep),
                tune=float(state['tune']),
                tuned_f0=(float(_tf) if (_tf is not None and _tf > 0.0)
                          else float(midi_to_freq(state['note'])))))
            rec['replay_engines'].append(state['engine'])
            rec['replay_grids'].append(grid.copy())

        # ── render one frame (all drawing lives in casynth_ui.draw_frame) ────
        # The spectrum strip is designed to FOLLOW the carrier (bars shift right as
        # the note rises); voices are now pitch-normalized, so transpose a display
        # copy by the live ratio (== 1.0 on the default note -> dumpframe unchanged).
        _disp_ratio = _transpose(state['note'])
        _disp_voices = (voices if abs(_disp_ratio - 1.0) < 1e-9
                        else [dict(v, freqs=v['freqs'] * _disp_ratio) for v in voices])
        rt = SimpleNamespace(
            grid=grid, color=color, labels=labels, voices=_disp_voices, drag=drag,
            ghost=_ghost, white_keys=white_keys, black_keys=black_keys,
            sb_scroll=_sb_scroll, meter=meter, budget=budget,
            # does the GEN amp-slew toggle act on the engine in front of us?
            slew_live=engines.get(state['engine']).gen_amp_slew,
            # "audio disabled" is a COMPLAINT, and there is nothing to complain
            # about until the device has actually been tried (it is opened after
            # this first frame -- see _open_device)
            audio_ok=(audio_ok or not _device['opened']), midi_in=midi_in,
            midi_dropdown_open=_midi_dropdown_open, midi_dd_items=_midi_dd_items,
            audio_dropdown_open=_audio_dropdown_open, audio_dd_items=_audio_dd_items,
            audio_dd_rects=_audio_dd_rects,
            audio_name=_out_name(),
            # 'ok' it plays there / 'pending' not opened yet / 'failed' it refused
            audio_state=('ok' if audio_ok else
                         ('pending' if not _device['opened'] else 'failed')),
            midi_dd_rects=_midi_dd_rects, midifile=midifile)
        if _dumpframe is not None:
            # The one-frame dump is a PIXEL-EXACT baseline, so nothing in it may
            # depend on how busy the machine happened to be.  The cost indicator
            # (2026-09-21, B2 of the seam) shows live milliseconds per block, and
            # by frame one it has measured whatever the render thread managed to
            # do in the meantime -- which made the gate flaky, not wrong.  At
            # frame one it has measured nothing, and the dump says exactly that.
            budget['ms'] = 0.0
            budget['max'] = 0.0
            budget['underruns'] = 0
            budget['lookahead'] = AUDIO_LOOKAHEAD_CHUNKS
        draw_frame(screen, (font, small), state, lay, rt)
        pygame.display.flip()

        if not _device['opened']:
            _device['opened'] = True
            audio_ok = _open_device()

        if _dumpframe is not None:
            pygame.image.save(screen, _dumpframe)
            running = False
        elif _deadline is not None and time.perf_counter() >= _deadline:
            running = False

    # Stop the render/MIDI threads and the audio stream BEFORE dumping the session,
    # so the render thread is no longer appending to rec['chunks'] when we read it.
    audio_ctl['alive'] = False
    for _th in _threads:
        _th.join(timeout=1.0)
    # Close MIDI BEFORE dumping so the rtmidi callback can't append to rec['midi_in']
    # while _dump_session reads it.
    midifile.stop()
    midi_in.close()
    host.stop()

    print(f"[budget] {budget['ms']:.2f} ms per block (worst {budget['max']:.2f}) "
          f"of {CHUNK_S * 1000.0:.2f} ms; look-ahead {budget['lookahead']} chunks "
          f"({budget['lookahead'] * CHUNK_S * 1000.0:.0f} ms); underruns {host.underruns}")
    if rec is not None:
        _dump_session(rec)

    pygame.quit()


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'replay':
        _replay_cli(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == 'scene':
        # session -> scene -> OFFLINE render, compared with the live WAV
        sys.exit(_scene_cli(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == 'play':
        main(autoplay_midi=sys.argv[2])
    else:
        main()
