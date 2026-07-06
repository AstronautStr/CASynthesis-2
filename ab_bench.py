#!/usr/bin/env python3
"""
ab_bench.py — A/B listening bench for CASynth (spec: memory/req-ab-bench.md).

Instant A/B comparison of two preset fields on the SAME hardcoded scene.  Live
A/B is hard in the full synth (by the time you rebuild a scene you have forgotten
how the first variant sounded); this stand freezes the scene into fixed test
cases and lets you flip between two independently-tuned fields with one click.

Nearest use: auditioning the `dyn` knob at shape>0 (the collision and chaos cases
are not yet listened to), so the dynamic-excitation path (events_field -> exc ->
analyse) MUST be live — see the CRITICAL note in the spec.

Design
------
* Reuses casynth_engine / casynth_core / casynth_config — the engine is NOT
  copied.  Audio is a single sounddevice callback stream drained from a ring
  buffer, fed by ONE render thread whose source is the currently ACTIVE field
  (exactly one field sounds at a time).  On a field switch / restart the audio
  state is reset so no tail keeps ringing.
* Two fields A and B share the case geometry (same initial), but each carries its
  own engine + engine params + ADSR/tune.  note / vol / speed are shared.
* Session recorder (CASYNTH_RECORD) is deliberately NOT wired in here.

CLI
---
    python ab_bench.py              # live pygame window
    python ab_bench.py --selftest   # headless: exercises every case + the dyn path
"""

import os
import sys
import time

import numpy as np

from casynth_config import (
    MAX_VOICES, MAX_MODES_PER_OBJ, TOTAL_SLOTS, SR, CHUNK_S, MASTER_GAIN,
    VOL_DEFAULT, NOTE_DEFAULT, KB_BASE_MIN, KB_BASE_MAX,
    ATTACK_MS_DEFAULT, ATTACK_MS_MIN, ATTACK_MS_MAX,
    DECAY_MS_DEFAULT, DECAY_MS_MIN, DECAY_MS_MAX,
    SUSTAIN_DEFAULT, SUSTAIN_MIN, SUSTAIN_MAX,
    RELEASE_MS_DEFAULT, RELEASE_MS_MIN, RELEASE_MS_MAX,
    AUDIO_LOOKAHEAD_CHUNKS,
    C_BG, C_PANEL, C_EDGE, C_TXT, C_DIM, C_BTN, C_BTN_HOT, C_ACCENT,
)
from casynth_core import ENGINES, ENGINE_BY_ID
from casynth_engine import (midi_to_freq, note_name, step, analyse,
                            events_field, render_chunk_laplacian, SlotPool)
from casynth_tuning import (dissonance_curve, scale_minima, snap_ratio,
                            TUNE_MAX_PARTIALS)
from patterns import PATTERNS


# ── Test cases ────────────────────────────────────────────────────────────────
# (name, grid_w, grid_h, [(pattern_name, x, y), ...])
#   x = column, y = row of the pattern's top-left anchor.
# Field sizes: spaceship cases get room so the glider does not reach a toroidal
# seam for ~30 ticks (segmentation is not wrap-aware).
CASES = [
    dict(name='block',          gw=18, gh=14, place=[('Block', 8, 6)]),
    dict(name='beacon',         gw=18, gh=14, place=[('Beacon', 7, 5)]),
    dict(name='r-pentomino',    gw=56, gh=38, place=[('R-pentomino', 26, 17)]),
    dict(name='glider',         gw=44, gh=32, place=[('Glider', 3, 3)]),
    dict(name='glider-vs-block', gw=40, gh=28, place=[('Glider', 2, 2),
                                                      ('Block', 7, 7)]),
]


def pattern_cells(name):
    """Look up a pattern's live-cell offsets by name (from patterns.PATTERNS)."""
    for _cat, plist in PATTERNS:
        for pname, cells in plist:
            if pname == name:
                return cells
    raise KeyError(f"pattern {name!r} not found in patterns.PATTERNS")


def build_grid(case):
    """Materialise a case's initial GOL grid (toroidal geometry gw×gh)."""
    gw, gh = case['gw'], case['gh']
    g = np.zeros((gh, gw), np.uint8)
    for (pname, x, y) in case['place']:
        for (r, c) in pattern_cells(pname):
            rr, cc = y + r, x + c
            if 0 <= rr < gh and 0 <= cc < gw:
                g[rr, cc] = 1
    return g


# ── Field state ───────────────────────────────────────────────────────────────
def _default_engine_params():
    """Per-engine param dicts seeded from the registry defaults (as gol_synth)."""
    return {e['id']: {arg: default for (arg, _l, _lo, _hi, _i, default) in e['params']}
            for e in ENGINES}


def new_field():
    """A fresh field with registry-default engine params + default ADSR/tune."""
    return dict(
        grid=None, exc=None, gen=0,
        engine=ENGINES[0]['id'],
        engine_params=_default_engine_params(),
        attack_ms=ATTACK_MS_DEFAULT, decay_ms=DECAY_MS_DEFAULT,
        sustain=SUSTAIN_DEFAULT, release_ms=RELEASE_MS_DEFAULT,
        tune=0.0, tuned_f0=midi_to_freq(NOTE_DEFAULT),
        # last analyse() result (kept for the tune snap + display); spec_f0 is the
        # carrier the voices were computed at.
        voices=[], spec_f0=midi_to_freq(NOTE_DEFAULT),
    )


def reset_field(f, case):
    """Reset a field's SCENE to the case initial (knob settings are preserved)."""
    f['grid'] = build_grid(case)
    f['exc'] = None
    f['gen'] = 0
    f['voices'] = []


def step_field(f):
    """Advance the field one GOL generation, updating its event-excitation field."""
    prev = f['grid'].copy()
    f['grid'] = step(f['grid'])
    f['exc'] = events_field(prev, f['grid'])
    f['gen'] += 1


def analyse_field(f, f0):
    """Run analyse() on the field's grid with its engine + params + exc field."""
    ep = f['engine_params'][f['engine']]
    return analyse(f['grid'], f0, f['engine'], ep, exc=f['exc'])


def field_spectrum(f, f0):
    """Voices + object colours for the field's CURRENT grid, cached on the scene
    signature (gen / engine / carrier / params).  A frozen field (constant gen)
    hits the cache; a param edit or a step refreshes it.  Also stores the result
    on the field for the audio path and the Sethares snap (single analyse/frame).

    Returns (voices, color).
    """
    ep = f['engine_params'][f['engine']]
    sig = (id(f['grid']), f['gen'], f['engine'], f0, tuple(sorted(ep.items())))
    if f.get('_spec_sig') == sig:
        return f['_spec_voices'], f['_spec_color']
    _labels, voices, color = analyse(f['grid'], f0, f['engine'], ep, exc=f['exc'])
    f['_spec_sig'] = sig
    f['_spec_voices'] = voices
    f['_spec_color'] = color
    f['voices'] = voices
    f['spec_f0'] = f0
    return voices, color


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (headless: no window, no audio)
# ──────────────────────────────────────────────────────────────────────────────
def _amps_concat(voices):
    if not voices:
        return np.array([])
    return np.concatenate([np.asarray(v['amps'], float) for v in voices])


def selftest():
    """Exercise every case and the dyn (exc) path; print a report, return 0/1."""
    ok = True

    # 1) Every case: >=20 ticks, analyse(with exc) + offline render_chunk, no error.
    for case in CASES:
        f = new_field()
        reset_field(f, case)
        pool = SlotPool()
        sz = TOTAL_SLOTS + 1
        phase = np.zeros(sz); amp_cur = np.zeros(sz); pan_cur = np.full(sz, 0.5)
        gain = MASTER_GAIN * VOL_DEFAULT
        f0 = midi_to_freq(NOTE_DEFAULT)
        try:
            for _t in range(24):
                step_field(f)
                _labels, voices, _color = analyse_field(f, f0)
                pool.update(voices, phase, amp_cur, pan_cur,
                            release_chunks=max(1, round(RELEASE_MS_DEFAULT / 1000.0 / CHUNK_S)),
                            attack_chunks=1, decay_chunks=1, sustain=1.0)
                render_chunk_laplacian(phase, amp_cur, pan_cur,
                                       pool.amp_tgt, pool.pan_tgt, pool.freq_slots,
                                       2, gain, gain)
            print(f"[ok]   case {case['name']:<16} 24 ticks: analyse+render clean")
        except Exception as e:                       # noqa: BLE001 -- report & fail
            ok = False
            print(f"[FAIL] case {case['name']:<16} raised {type(e).__name__}: {e}")

    # 2) dyn path numerically:  events case -> dyn=0 vs dyn=1 amps DIFFER;
    #    block (no events after settle) -> amps EQUAL (exc≈0 -> fallback to deg).
    f0 = midi_to_freq(NOTE_DEFAULT)

    def _dyn_amps(grid, exc, dyn):
        params = dict(n=12, spread=0.0, alpha=1.0, shape=1.0, harm=0.0,
                      fullshape=1, dyn=dyn)
        _l, voices, _c = analyse(grid, f0, 'laplacian', params, exc=exc)
        return _amps_concat(voices)

    # events case: r-pentomino a few gens in (births + deaths guaranteed).
    fe = new_field(); reset_field(fe, CASES[2])          # r-pentomino
    for _ in range(10):
        step_field(fe)
    a0 = _dyn_amps(fe['grid'], fe['exc'], 0.0)
    a1 = _dyn_amps(fe['grid'], fe['exc'], 1.0)
    ev_ok = a0.shape == a1.shape and a0.size > 0 and float(np.abs(a0 - a1).max()) > 1e-6
    print(f"[{'ok' if ev_ok else 'FAIL'}]   dyn events  r-pentomino gen{fe['gen']}: "
          f"max|d amp| = {float(np.abs(a0 - a1).max()) if a0.size else 0.0:.4f} "
          f"(expect > 1e-6, events present in exc: {bool(fe['exc'] is not None and fe['exc'].any())})")
    ok = ok and ev_ok

    # control: block, stepped to stability -> exc is all-zero -> dyn has no effect.
    fb = new_field(); reset_field(fb, CASES[0])          # block
    for _ in range(5):
        step_field(fb)
    b0 = _dyn_amps(fb['grid'], fb['exc'], 0.0)
    b1 = _dyn_amps(fb['grid'], fb['exc'], 1.0)
    blk_ok = b0.shape == b1.shape and float(np.abs(b0 - b1).max() if b0.size else 0.0) < 1e-9
    print(f"[{'ok' if blk_ok else 'FAIL'}]   dyn control block gen{fb['gen']}: "
          f"max|d amp| = {float(np.abs(b0 - b1).max()) if b0.size else 0.0:.2e} "
          f"(expect ~0; exc has events: {bool(fb['exc'] is not None and fb['exc'].any())})")
    ok = ok and blk_ok

    # 3) A/B switch simulation (reset both fields, reset audio) does not crash.
    try:
        fa, fbb = new_field(), new_field()
        for case in CASES:
            reset_field(fa, case); reset_field(fbb, case)
            step_field(fa); step_field(fbb)
        print("[ok]   A/B switch simulation: reset both fields across all cases clean")
    except Exception as e:                               # noqa: BLE001
        ok = False
        print(f"[FAIL] A/B switch simulation raised {type(e).__name__}: {e}")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ──────────────────────────────────────────────────────────────────────────────
# LIVE BENCH  (pygame window)
# ──────────────────────────────────────────────────────────────────────────────
# Layout constants (window is fixed; per-case field geometry is fitted into a
# viewport by choosing a cell size, so different case sizes share one window).
TOOLBAR_H = 44
FIELD_VP_W = 470
FIELD_VP_H = 500
SPEC_H = 62          # spectrum strip height at the bottom of each field viewport
GAP = 12
PANEL_W = 250
STATUS_H = 26

WIN_W = GAP + FIELD_VP_W + GAP + FIELD_VP_W + GAP + PANEL_W + GAP
WIN_H = TOOLBAR_H + GAP + FIELD_VP_H + STATUS_H

FIELD_COL = [(111, 208, 224), (224, 168, 96)]     # A = cyan, B = amber


def main():
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    from casynth_ui import _draw_spectrum            # reuse the synth's spectrum view
    try:
        import sounddevice as sd
    except Exception:                                # optional; visuals still run
        sd = None

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("CASynth — A/B bench")
    font = pygame.font.SysFont("consolas,menlo,monospace", 15)
    small = pygame.font.SysFont("consolas,menlo,monospace", 12)
    clock = pygame.time.Clock()

    # ── shared state ─────────────────────────────────────────────────────────
    fields = [new_field(), new_field()]
    common = dict(note=NOTE_DEFAULT, vol=VOL_DEFAULT, speed=6.0)   # speed = ticks/s
    case_idx = [0]
    active = [None]        # which field SOUNDS+runs (0/1/None)
    sel = [0]              # which field the panel EDITS (0/1); == active when running
    next_step_time = [None]
    meter = {'peak': 0.0}
    last_note = [common['note']]

    for f in fields:
        reset_field(f, CASES[case_idx[0]])

    # ── audio state (single pipeline; recreated on reset to kill tails) ───────
    import queue as _queue
    import threading
    AU = {}

    def init_audio():
        AU['pool'] = SlotPool()
        sz = TOTAL_SLOTS + 1
        AU['phase'] = np.zeros(sz)
        AU['amp_cur'] = np.zeros(sz)
        AU['pan_cur'] = np.full(sz, 0.5)
    init_audio()

    audio_q = _queue.Queue()
    _resid = {'buf': None, 'pos': 0}
    render_spec = {'cur': None}
    audio_ctl = {'alive': True, 'reset': False}

    def _audio_cb(outdata, frames, time_info, status):
        filled = 0
        while filled < frames:
            if _resid['buf'] is None:
                try:
                    _resid['buf'] = audio_q.get_nowait()
                    _resid['pos'] = 0
                except _queue.Empty:
                    outdata[filled:] = 0
                    return
            buf, pos = _resid['buf'], _resid['pos']
            take = min(frames - filled, len(buf) - pos)
            outdata[filled:filled + take] = buf[pos:pos + take]
            filled += take; pos += take
            if pos >= len(buf):
                _resid['buf'] = None
            else:
                _resid['pos'] = pos

    stream = None
    if sd is None:
        print("[audio disabled: sounddevice not installed] - visuals still run")
    else:
        try:
            silent = np.zeros((int(CHUNK_S * SR), 2), np.int16)
            for _ in range(AUDIO_LOOKAHEAD_CHUNKS):
                audio_q.put(silent.copy())
            stream = sd.OutputStream(samplerate=SR, channels=2, dtype='int16',
                                     latency='low', callback=_audio_cb)
            stream.start()
            print(f"[audio] out latency={stream.latency * 1000:.0f}ms + "
                  f"{AUDIO_LOOKAHEAD_CHUNKS} chunk look-ahead")
        except Exception as e:                       # noqa: BLE001
            print(f"[audio disabled: {e}] - visuals still run")

    def _render_loop():
        gain_prev = MASTER_GAIN * common['vol']
        while audio_ctl['alive']:
            if audio_ctl['reset']:
                init_audio()
                audio_ctl['reset'] = False
                gain_prev = MASTER_GAIN * common['vol']
            if audio_q.qsize() >= AUDIO_LOOKAHEAD_CHUNKS:
                time.sleep(0.001)
                continue
            spec = render_spec['cur']
            if spec is None:
                voices_in = []
                a_ch = d_ch = 1; sus = 1.0
                r_ch = max(1, round(RELEASE_MS_DEFAULT / 1000.0 / CHUNK_S))
            else:
                voices_in = spec['voices']
                a_ch = max(1, round(spec['attack_ms'] / 1000.0 / CHUNK_S))
                d_ch = max(1, round(spec['decay_ms'] / 1000.0 / CHUNK_S))
                r_ch = max(1, round(spec['release_ms'] / 1000.0 / CHUNK_S))
                sus = float(spec['sustain'])
            gain = MASTER_GAIN * common['vol']
            pool = AU['pool']
            pool.update(voices_in, AU['phase'], AU['amp_cur'], AU['pan_cur'],
                        r_ch, a_ch, d_ch, sus)
            buf, peak, _nclip = render_chunk_laplacian(
                AU['phase'], AU['amp_cur'], AU['pan_cur'],
                pool.amp_tgt, pool.pan_tgt, pool.freq_slots, 2, gain_prev, gain)
            gain_prev = gain
            audio_q.put(buf)
            meter['peak'] = max(peak, meter['peak'] * 0.90)

    render_thread = threading.Thread(target=_render_loop, daemon=True)
    render_thread.start()

    # ── field activation / case selection ────────────────────────────────────
    def select_case(i):
        case_idx[0] = i
        for f in fields:
            reset_field(f, CASES[i])
        active[0] = None
        next_step_time[0] = None
        render_spec['cur'] = None
        audio_ctl['reset'] = True          # silence any ringing tail

    def activate_field(i):
        """Restart field i on the case initial and make it the sounding field."""
        sel[0] = i
        active[0] = i
        reset_field(fields[i], CASES[case_idx[0]])
        next_step_time[0] = time.perf_counter() + 1.0 / max(0.1, common['speed'])
        audio_ctl['reset'] = True          # kill the previous field's tail

    # ── Sethares snap for the active field (mirrors gol_synth._snap_note_on) ──
    def snap_active(new_note):
        f = fields[sel[0]]
        f_raw = midi_to_freq(new_note)
        tune = f['tune']
        if tune <= 0.0:
            f['tuned_f0'] = f_raw
            return
        prev_f0 = f['tuned_f0']
        if not prev_f0 or prev_f0 <= 0.0 or not f['voices']:
            f['tuned_f0'] = f_raw
            return
        base_f0 = f['spec_f0']
        rescale = (prev_f0 / base_f0) if base_f0 > 0.0 else 1.0
        all_f, all_a = [], []
        for v in f['voices']:
            for fr, a in zip(v['freqs'], v['amps']):
                if a > 0.0:
                    all_f.append(fr * rescale); all_a.append(float(a))
        if not all_f:
            f['tuned_f0'] = f_raw
            return
        cf = np.array(all_f); ca = np.array(all_a)
        if len(ca) > TUNE_MAX_PARTIALS:
            idx = np.argpartition(ca, -TUNE_MAX_PARTIALS)[-TUNE_MAX_PARTIALS:]
            cf = cf[idx]; ca = ca[idx]
        ratios_arr, curve_vals = dissonance_curve(cf, ca)
        minima = scale_minima(curve_vals, ratios_arr)
        r_snap = snap_ratio(f_raw / prev_f0, minima, tune)
        f['tuned_f0'] = prev_f0 * r_snap

    # ── panel layout (rebuilt each frame; cheap) ─────────────────────────────
    PANEL_X = GAP + 2 * FIELD_VP_W + 2 * GAP
    LBL_W, TRK_W = 52, 118

    def _fmt(kind, integer):
        if kind == 'ms':
            return lambda v: f"{v:.0f}ms"
        if kind == 'note':
            return lambda v: note_name(int(v))
        if kind == 'speed':
            return lambda v: f"{v:.1f}/s"
        if integer:
            return lambda v: f"{int(v)}"
        return lambda v: f"{v:.2f}"

    def build_panel():
        f = fields[sel[0]]
        ctrls, headers, tabs = [], [], []
        px = PANEL_X + 10
        y = TOOLBAR_H + GAP
        headers.append((f"EDITING FIELD {'AB'[sel[0]]}", y)); y += 22
        # engine tabs (wrap to panel width)
        tx = px
        for e in ENGINES:
            w = small.size(e['label'])[0] + 14
            if tx + w > PANEL_X + PANEL_W - 8:
                tx = px; y += 22
            tabs.append(dict(id=e['id'], label=e['label'],
                             rect=pygame.Rect(tx, y, w, 18)))
            tx += w + 4
        y += 26

        trk_x = px + LBL_W
        def add(cid, kind, lbl, lo, hi, integer, fmtkind):
            nonlocal y
            ctrls.append(dict(id=cid, scope=kind, label=lbl, lo=lo, hi=hi,
                              integer=integer, fmt=_fmt(fmtkind, integer),
                              track=pygame.Rect(trk_x, y + 4, TRK_W, 8)))
            y += 22

        headers.append(("ENGINE", y)); y += 18
        for (arg, lbl, lo, hi, integer, _d) in ENGINE_BY_ID[f['engine']]['params']:
            add(arg, 'engine', lbl, lo, hi, integer, 'int' if integer else 'f')
        y += 6
        headers.append(("ENV", y)); y += 18
        add('attack_ms', 'field', 'A', ATTACK_MS_MIN, ATTACK_MS_MAX, True, 'ms')
        add('decay_ms', 'field', 'D', DECAY_MS_MIN, DECAY_MS_MAX, True, 'ms')
        add('sustain', 'field', 'S', SUSTAIN_MIN, SUSTAIN_MAX, False, 'f')
        add('release_ms', 'field', 'R', RELEASE_MS_MIN, RELEASE_MS_MAX, True, 'ms')
        add('tune', 'field', 'tune', 0.0, 1.0, False, 'f')
        y += 6
        headers.append(("COMMON", y)); y += 18
        add('note', 'common', 'note', KB_BASE_MIN, KB_BASE_MAX, True, 'note')
        add('vol', 'common', 'vol', 0.0, 1.0, False, 'f')
        add('speed', 'common', 'speed', 0.5, 20.0, False, 'speed')
        return ctrls, headers, tabs

    def ctrl_value(c):
        f = fields[sel[0]]
        if c['scope'] == 'engine':
            return f['engine_params'][f['engine']][c['id']]
        if c['scope'] == 'field':
            return f[c['id']]
        return common[c['id']]

    def set_ctrl(c, mx):
        frac = float(np.clip((mx - c['track'].x) / c['track'].w, 0.0, 1.0))
        val = c['lo'] + frac * (c['hi'] - c['lo'])
        val = int(round(val)) if c['integer'] else float(val)
        f = fields[sel[0]]
        if c['scope'] == 'engine':
            f['engine_params'][f['engine']][c['id']] = val
        elif c['scope'] == 'field':
            f[c['id']] = val
        else:
            common[c['id']] = val

    # toolbar case buttons
    def case_buttons():
        btns, bx = [], GAP
        for i, case in enumerate(CASES):
            w = font.size(case['name'])[0] + 20
            btns.append(dict(idx=i, label=case['name'],
                             rect=pygame.Rect(bx, 8, w, 28)))
            bx += w + 8
        return btns

    def field_vp(i):
        x = GAP + i * (FIELD_VP_W + GAP)
        return pygame.Rect(x, TOOLBAR_H + GAP, FIELD_VP_W, FIELD_VP_H)

    dragging = [None]

    def draw_slider(c):
        val = ctrl_value(c)
        tr = c['track']
        screen.blit(small.render(c['label'], True, C_TXT), (PANEL_X + 10, tr.y - 3))
        pygame.draw.rect(screen, C_EDGE, tr, border_radius=3)
        frac = (val - c['lo']) / (c['hi'] - c['lo']) if c['hi'] > c['lo'] else 0.0
        fillw = int(tr.w * float(np.clip(frac, 0, 1)))
        pygame.draw.rect(screen, C_ACCENT, (tr.x, tr.y, fillw, tr.h), border_radius=3)
        kx = tr.x + fillw
        pygame.draw.circle(screen, C_TXT, (kx, tr.y + tr.h // 2), 5)
        screen.blit(small.render(c['fmt'](val), True, C_DIM), (tr.right + 8, tr.y - 3))

    def draw_field(i, base_f0):
        f = fields[i]
        case = CASES[case_idx[0]]
        vp = field_vp(i)
        pygame.draw.rect(screen, (18, 20, 26), vp)
        gw, gh = case['gw'], case['gh']
        # grid area sits between the header and the spectrum strip at the bottom.
        grid_h = vp.h - 24 - SPEC_H - 8
        cs = max(2, min(vp.w // gw, grid_h // gh))
        gx = vp.x + (vp.w - cs * gw) // 2
        gy = vp.y + 24 + (grid_h - cs * gh) // 2
        # grid frame
        pygame.draw.rect(screen, (30, 34, 42),
                         (gx - 1, gy - 1, cs * gw + 2, cs * gh + 2), 1)
        col = FIELD_COL[i]
        ys, xs = np.where(f['grid'] > 0)
        for r, c in zip(ys, xs):
            pygame.draw.rect(screen, col, (gx + c * cs, gy + r * cs,
                                           cs - 1, cs - 1))
        # spectrum strip (raw engine output of THIS field's current grid/params --
        # the shape's spectral fingerprint; makes an A/B difference VISIBLE even
        # when it is hard to hear).  Reuses the synth's _draw_spectrum.
        voices, color = field_spectrum(f, base_f0)
        spec_rect = pygame.Rect(vp.x + 6, vp.bottom - SPEC_H - 4, vp.w - 12, SPEC_H)
        _draw_spectrum(screen, small, spec_rect, voices, color, base_f0)
        # header
        tag = 'AB'[i]
        running = (active[0] == i)
        hdr = f"{tag}  {ENGINE_BY_ID[f['engine']]['label']}  gen {f['gen']}"
        if running:
            hdr += "   ▶ SOUNDING"
        screen.blit(font.render(hdr, True, col if running else C_TXT),
                    (vp.x + 8, vp.y + 4))
        # borders: selected (editing) = bright, running gets a thicker frame
        bw = 3 if running else (2 if sel[0] == i else 1)
        bc = col if running else (C_ACCENT if sel[0] == i else C_EDGE)
        pygame.draw.rect(screen, bc, vp, bw)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        ctrls, headers, tabs = build_panel()
        cbtns = case_buttons()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                # case buttons
                hit = False
                for b in cbtns:
                    if b['rect'].collidepoint(mx, my):
                        select_case(b['idx']); hit = True; break
                if hit:
                    continue
                # engine tabs (edit selected field's engine)
                for tb in tabs:
                    if tb['rect'].collidepoint(mx, my):
                        fields[sel[0]]['engine'] = tb['id']; hit = True; break
                if hit:
                    continue
                # sliders
                for c in ctrls:
                    hot = c['track'].inflate(8, 12)
                    if hot.collidepoint(mx, my):
                        dragging[0] = c['id'] + '|' + c['scope']
                        set_ctrl(c, mx); hit = True; break
                if hit:
                    continue
                # field click -> activate/restart
                for i in (0, 1):
                    if field_vp(i).collidepoint(mx, my):
                        activate_field(i); break
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging[0] = None
            elif e.type == pygame.MOUSEMOTION and dragging[0] is not None:
                for c in ctrls:
                    if c['id'] + '|' + c['scope'] == dragging[0]:
                        set_ctrl(c, e.pos[0]); break

        # note change -> Sethares snap for the active/selected field
        if common['note'] != last_note[0]:
            snap_active(common['note'])
            last_note[0] = common['note']

        # step the active field on the speed clock
        if active[0] is not None and next_step_time[0] is not None:
            interval = 1.0 / max(0.1, common['speed'])
            now = time.perf_counter()
            n = 0
            while now >= next_step_time[0] and n < 8:
                step_field(fields[active[0]])
                next_step_time[0] += interval
                n += 1
            if next_step_time[0] < now - interval:
                next_step_time[0] = now + interval

        # analyse the active field and publish the render spec
        base_f0 = midi_to_freq(common['note'])
        if active[0] is not None:
            f = fields[active[0]]
            voices, _color = field_spectrum(f, base_f0)   # single analyse/frame
            target = (f['tuned_f0'] if (f['tune'] > 0.0 and f['tuned_f0'])
                      else base_f0)
            ratio = (target / base_f0) if base_f0 > 0 else 1.0
            # Publish center-panned voices: NO stereo panning in the A/B bench, so
            # an object's horizontal position never colours the timbre comparison.
            resc = abs(ratio - 1.0) > 1e-9
            pub = [dict(v, pan=0.5,
                        freqs=(np.asarray(v['freqs']) * ratio if resc else v['freqs']))
                   for v in voices]
            render_spec['cur'] = dict(
                voices=pub, attack_ms=f['attack_ms'], decay_ms=f['decay_ms'],
                sustain=f['sustain'], release_ms=f['release_ms'])
        else:
            render_spec['cur'] = None

        # ── draw ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        # toolbar
        pygame.draw.rect(screen, C_PANEL, (0, 0, WIN_W, TOOLBAR_H))
        for b in cbtns:
            hotc = C_ACCENT if b['idx'] == case_idx[0] else C_BTN
            pygame.draw.rect(screen, hotc, b['rect'], border_radius=4)
            tc = C_BG if b['idx'] == case_idx[0] else C_TXT
            screen.blit(font.render(b['label'], True, tc),
                        (b['rect'].x + 10, b['rect'].y + 6))
        # fields (each with its own spectrum strip -> A/B difference is visible)
        draw_field(0, base_f0)
        draw_field(1, base_f0)
        # panel
        pygame.draw.rect(screen, C_PANEL, (PANEL_X, TOOLBAR_H + GAP,
                                           PANEL_W, FIELD_VP_H))
        for (txt, hy) in headers:
            screen.blit(small.render(txt, True, C_DIM), (PANEL_X + 10, hy))
        f = fields[sel[0]]
        for tb in tabs:
            selc = (tb['id'] == f['engine'])
            pygame.draw.rect(screen, C_ACCENT if selc else C_BTN, tb['rect'],
                             border_radius=3)
            screen.blit(small.render(tb['label'], True,
                                     C_BG if selc else C_TXT),
                        (tb['rect'].x + 6, tb['rect'].y + 2))
        for c in ctrls:
            draw_slider(c)
        # status line
        sy = TOOLBAR_H + GAP + FIELD_VP_H
        act = 'AB'[active[0]] if active[0] is not None else '—'
        status = (f"Click a field to play/restart it (only one sounds).  "
                  f"Panel edits field {'AB'[sel[0]]}.  Active: {act}.  "
                  f"peak {meter['peak']:.2f}   Esc quits")
        screen.blit(small.render(status, True, C_DIM), (GAP, sy + 7))

        pygame.display.flip()

    # ── shutdown ─────────────────────────────────────────────────────────────
    audio_ctl['alive'] = False
    render_thread.join(timeout=0.5)
    if stream is not None:
        stream.stop(); stream.close()
    pygame.quit()


if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        sys.exit(selftest())
    main()
