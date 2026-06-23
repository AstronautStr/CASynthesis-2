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
import time
import queue as _queue
import threading
from types import SimpleNamespace
import numpy as np
import pygame
try:
    import sounddevice as sd
except Exception:          # optional; audio is just disabled if unavailable
    sd = None
from casynth_core import ENGINES, ENGINE_BY_ID
from patterns import PATTERNS

# Modularised subsystems (extracted from this monolith -- see casynth_*.py).
from casynth_config import *                                   # noqa: F401,F403
from casynth_engine import (midi_to_freq, note_name, step, hsv, analyse,
                            render_chunk_laplacian, SlotPool)
from casynth_session import _dump_session, replay_session, _replay_cli
from casynth_midi import MidiInput, MIDI_AVAILABLE
from casynth_ui import _make_piano, pattern_preview_surf, draw_frame


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
        engine=ENGINES[0]['id'],
        engine_params={e['id']: {arg: default for (arg, _l, _lo, _hi, _i, default)
                                 in e['params']} for e in ENGINES},
        release_ms=RELEASE_MS_DEFAULT,
        attack_ms=ATTACK_MS_DEFAULT,
        decay_ms=DECAY_MS_DEFAULT,
        sustain=SUSTAIN_DEFAULT,
        # MIDI gate: True = voices active; False = all slots in release (note-off).
        # Keyboard/mouse piano always set gate=True (latch). MIDI note-off sets False.
        gate=True,
        midi_held=[],   # stack of currently held MIDI notes (last-note priority)
        midi_port=None, # name of currently open MIDI port
    )

    midi_in = MidiInput()
    _midi_dropdown_open = False

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
    _vol_section_y = by + 96                      # starts 4px below R-track (by+86+8+4)
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

    # Static layout geometry bundled for the renderer (casynth_ui.draw_frame).
    # These are the SAME objects the event loop mutates/reads -> no divergence
    # (ctrls is rebuilt in place by rebuild_ctrls; this reference still tracks it).
    lay = SimpleNamespace(
        W=W, buttons=buttons, pat_btn=_pat_btn, bpm_x=_bpm_x, bpm_track=bpm_track,
        div_buttons=div_buttons, legend_x=legend_x, info_y=info_y, rc_x=_rc_x,
        vol_section_y=_vol_section_y, vol_track=vol_track, rc_track_x=_rc_track_x,
        rc_track_w=_RC_TRACK_W, env_hdr_rc_y=_ENV_HDR_RC_Y, ctrls=ctrls,
        meter_track=meter_track, midi_btn=_midi_btn, midi_btn_w=_MIDI_BTN_W,
        midi_dd_ith=_MIDI_DD_ITH, engine_tabs=engine_tabs, sb_items=_sb_items,
        sb_content_h=_sb_content_h, sb_scroll_min=_sb_scroll_min)

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

    # ── Audio render thread + MIDI input thread (decoupled from the 60 fps loop) ──
    # The MAIN thread publishes the current render spec (voices computed at base_f0
    # + the live knob values); the RENDER thread renders sub-chunks just-in-time,
    # rescaling each voice's freqs to the LIVE MIDI note (so a note change between
    # frames is applied within one sub-chunk, not a frame) and applying the live
    # gate; the MIDI thread polls the device continuously.  This takes the MIDI poll
    # and the look-ahead off the timing path -> onsets quantise to one CHUNK_S
    # sub-chunk locked to the playback clock, not to the frame rate.
    render_spec = {'cur': None}        # atomic publish point (single ref swap)
    audio_ctl   = {'alive': True}      # threads exit when False
    # Cache voices rescaled to the live note so an unchanged (spec, note) -- the
    # common case (live note == the note the spec was built at) -- does not
    # re-allocate every sub-chunk.
    _rescale = {'key': None, 'voices': None}

    def _live_voices(spec, note):
        """`spec` voices with freqs rescaled to the live MIDI note's f0.  freqs are
        linear in f0 for every engine (Laplace f0·√λ/√λ0, harmonic f0·k) so this is
        exact; amps/pan are f0-independent and untouched."""
        base_f0 = spec['base_f0']
        ratio = (midi_to_freq(note) / base_f0) if base_f0 > 0 else 1.0
        if abs(ratio - 1.0) < 1e-9:
            return spec['voices']
        key = (id(spec), note)
        if _rescale['key'] != key:
            _rescale['voices'] = [dict(v, freqs=v['freqs'] * ratio)
                                  for v in spec['voices']]
            _rescale['key'] = key
        return _rescale['voices']

    def _render_loop():
        gain_prev = MASTER_GAIN * state['vol']
        cum = 0                        # cumulative output samples (onset timestamps)
        last_note, last_gate = state['note'], bool(state['gate'])
        while audio_ctl['alive']:
            if audio_q.qsize() >= AUDIO_LOOKAHEAD_CHUNKS:
                time.sleep(0.001)      # ring full -> idle briefly
                continue
            spec = render_spec['cur']
            note = state['note']
            gate = bool(state['gate'])
            # Live knob-derived chunk counts (ms-based -> grid-independent).
            release_chunks = max(1, round(state['release_ms'] / 1000.0 / CHUNK_S))
            attack_chunks  = max(1, round(state['attack_ms']  / 1000.0 / CHUNK_S))
            decay_chunks   = max(1, round(state['decay_ms']   / 1000.0 / CHUNK_S))
            sustain        = float(state['sustain'])
            gain = MASTER_GAIN * state['vol']
            voices_in = (_live_voices(spec, note) if (spec is not None and gate)
                         else [])
            pool.update(voices_in, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain)
            buf, peak, n_clip = render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                       pool.amp_tgt, pool.pan_tgt,
                                                       pool.freq_slots, 2,
                                                       gain_prev, gain)
            gain_prev = gain
            audio_q.put(buf)
            meter['peak'] = max(peak, meter['peak'] * METER_DECAY)
            meter['clip'] = n_clip > 0
            if rec is not None:
                rec['chunks'].append(buf)
                if note != last_note or gate != last_gate:
                    rec['midi_onsets'].append((cum, int(note), bool(gate)))
                rec['underruns'] = _ur['n']
            last_note, last_gate = note, gate
            cum += len(buf)

    def _on_midi_message(msg):
        # Called from rtmidi's own thread the instant a message arrives (callback
        # mode -> no poll latency, no GIL-starved poll loop).  Last-note priority;
        # updates the shared live note/gate that the render thread samples.
        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note not in state['midi_held']:
                state['midi_held'].append(msg.note)
            state['note'] = msg.note
            state['gate'] = True
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in state['midi_held']:
                state['midi_held'].remove(msg.note)
            if state['midi_held']:
                state['note'] = state['midi_held'][-1]
                state['gate'] = True
            else:
                state['gate'] = False
        else:
            return   # clock/sysex/other: not a note event
        if rec is not None:
            # Ground-truth input timing, stamped the instant rtmidi hands it over.
            rec['midi_in'].append((time.perf_counter(),
                                   int(state['note']), bool(state['gate'])))

    _threads = []
    if audio_ok:
        _t = threading.Thread(target=_render_loop, daemon=True, name='render')
        _t.start(); _threads.append(_t)
    # MIDI input is delivered by rtmidi's callback (set when a device is opened);
    # there is no MIDI poll thread.

    # UI regression hook: if CASYNTH_DUMPFRAME is set, render exactly one frame,
    # save it to that path and exit (deterministic on the default empty field) --
    # a bit-exact visual baseline for the draw extraction.
    _dumpframe = os.environ.get('CASYNTH_DUMPFRAME')
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

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in _KB_PIANO:
                    state['note'] = state['kb_base'] + _KB_PIANO[e.key]
                    state['gate'] = True
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
                if _midi_dropdown_open:
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
                        grid[rc] = paint
                    elif e.pos[1] >= piano_top:
                        m = hit_piano(e.pos)
                        if m is not None:
                            state['note'] = m
                            state['gate'] = True
                    else:
                        tab_hit = next((t for t in engine_tabs
                                        if t['rect'].collidepoint(e.pos)), None)
                        if MIDI_AVAILABLE and _midi_btn.collidepoint(e.pos):
                            _midi_dropdown_open = not _midi_dropdown_open
                        elif tab_hit is not None:
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
        base_f0 = f0()
        labels, voices, color = analyse(grid, base_f0, state['engine'], _ep)
        # Publish an immutable render spec for the render thread (atomic ref swap).
        # The render thread rescales these voices to the LIVE MIDI note and applies
        # the live gate, so note timing follows the device, not this frame.
        render_spec['cur'] = {'voices': voices, 'base_f0': base_f0}
        ur_delta = _ur['n'] - _ur['prev']
        _ur['prev'] = _ur['n']
        if rec is not None:
            rec['frames'].append((round(dt * 1000.0, 2), state['gen'],
                                  len(voices), 0, ur_delta))
            # Per-frame control snapshot -- CONTEXT for the session (engine/knob
            # timeline).  Audio is no longer rendered per frame, so n_rendered=0; the
            # AUTHORITATIVE sample-accurate note timing is rec['midi_onsets'] written
            # by the render thread.  (Faithful per-frame audio replay is superseded
            # by the threaded render -- see memory/log/2026-06-23-midi-timing-jitter.)
            rec['replay_controls'].append(dict(
                n_rendered=0,
                engine=state['engine'],
                note=int(state['note']),
                gate=bool(state['gate']),
                vol=float(state['vol']),
                attack_ms=float(state['attack_ms']),
                decay_ms=float(state['decay_ms']),
                sustain=float(state['sustain']),
                release_ms=float(state['release_ms']),
                engine_params=dict(_ep)))
            rec['replay_engines'].append(state['engine'])
            rec['replay_grids'].append(grid.copy())

        # ── render one frame (all drawing lives in casynth_ui.draw_frame) ────
        rt = SimpleNamespace(
            grid=grid, color=color, labels=labels, voices=voices, drag=drag,
            ghost=_ghost, white_keys=white_keys, black_keys=black_keys,
            sb_scroll=_sb_scroll, meter=meter, audio_ok=audio_ok, midi_in=midi_in,
            midi_dropdown_open=_midi_dropdown_open, midi_dd_items=_midi_dd_items,
            midi_dd_rects=_midi_dd_rects)
        draw_frame(screen, (font, small), state, lay, rt)
        pygame.display.flip()

        if _dumpframe is not None:
            pygame.image.save(screen, _dumpframe)
            running = False

    # Stop the render/MIDI threads and the audio stream BEFORE dumping the session,
    # so the render thread is no longer appending to rec['chunks'] when we read it.
    audio_ctl['alive'] = False
    for _th in _threads:
        _th.join(timeout=1.0)
    # Close MIDI BEFORE dumping so the rtmidi callback can't append to rec['midi_in']
    # while _dump_session reads it.
    midi_in.close()
    if stream is not None:
        stream.stop()
        stream.close()

    if rec is not None:
        _dump_session(rec)

    pygame.quit()


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'replay':
        _replay_cli(sys.argv[2])
    else:
        main()
