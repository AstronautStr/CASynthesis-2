"""Rendering layer for the gol_synth prototype (all pygame drawing).

Holds the piano-key geometry helpers and the per-frame `draw_frame`, extracted
verbatim from gol_synth.main()'s draw block.  main() still owns layout
construction, event handling and the audio/MIDI threads (input is tied to the
mutable app state); this module is pure output: given the static layout `lay`,
the live `state` and the per-frame runtime `rt`, it paints one frame.

The draw body is character-identical to the original (the only changes are
reading geometry from `lay.*`/`rt.*` instead of main()'s locals), so the rendered
pixels are bit-for-bit unchanged -- verified by the CASYNTH_DUMPFRAME baseline.
"""
import numpy as np
import pygame

from casynth_config import *
from casynth_engine import hsv, note_name, midi_to_freq
from casynth_panel import range_text as panel_range_text
from casynth_midi import MIDI_AVAILABLE
from casynth_midifile import MIDIFILE_AVAILABLE
from patterns import PATTERNS


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


def _ctrl_value(state, c):
    """Current value of a knob (same rule as main()'s ctrl_value closure):
    engine attributes live in the active engine's per-engine dict; synth-wide
    knobs (ADSR) live directly in state."""
    if c['scope'] == 'engine':
        return state['engine_params'][state['engine']][c['id']]
    return state[c['id']]


def _range_text(c, which):
    """The text in one Min / Max field: what the track spans right now.  The same
    rule gol_synth.range_text uses -- this side only draws it."""
    return panel_range_text(c['lo'] if which == 'min' else c['hi'], c.get('integer'))


def _draw_range_fields(screen, small, c, rt):
    """The Min / Max fields of one knob row: a box each, the typed one carrying a
    caret and its selection, a refused one outlined in the accent colour."""
    edit = getattr(rt, 'range_edit', None)
    error = getattr(rt, 'range_error', None)
    for which, rect in c['range_rects'].items():
        typing = edit is not None and edit['id'] == c['id'] and edit['which'] == which
        pygame.draw.rect(screen, C_PANEL if typing else C_BG, rect, border_radius=3)
        edged = (C_ACCENT if typing else
                 (C_WHITE_ON if error == (c['id'], which) else C_EDGE))
        pygame.draw.rect(screen, edged, rect, 1, border_radius=3)
        text = edit['edit'].text if typing else _range_text(c, which)
        surf = small.render(text, True, C_TXT if typing else C_DIM)
        screen.set_clip(rect.inflate(-4, 0))
        screen.blit(surf, (rect.x + 3, rect.centery - 7))
        if typing:
            ed = edit['edit']
            sel = ed.selection
            if sel is not None:
                x0 = rect.x + 3 + small.size(text[:sel[0]])[0]
                x1 = rect.x + 3 + small.size(text[:sel[1]])[0]
                s = pygame.Surface((max(1, x1 - x0), rect.height - 4))
                s.set_alpha(70)
                s.fill(C_ACCENT)
                screen.blit(s, (x0, rect.y + 2))
                screen.blit(surf, (rect.x + 3, rect.centery - 7))
            cx = rect.x + 3 + small.size(text[:ed.cursor])[0]
            pygame.draw.line(screen, C_TXT, (cx, rect.y + 3), (cx, rect.bottom - 3))
        screen.set_clip(None)


def _draw_spectrum(screen, small, rect, voices, color, f0_hz):
    """Engine mode bars: a vertical line per partial (log frequency on X) with
    height = its amplitude (raw engine output, BEFORE the ADSR/tail envelope --
    the shape's spectral fingerprint).  Each voice's bars take its field-object
    colour, tying the spectrum to the shape that produced it.  Read-only."""
    pygame.draw.rect(screen, C_BG, rect)
    pygame.draw.rect(screen, C_EDGE, rect, 1)
    top_y, base_y = rect.top + 2, rect.bottom - 2
    usable_h = base_y - top_y
    x0, w = rect.left + 1, rect.width - 2
    lo = np.log10(SPEC_F_MIN)
    span = np.log10(SPEC_F_MAX) - lo

    def fx(freq):
        lf = np.log10(min(max(freq, SPEC_F_MIN), SPEC_F_MAX))
        return int(x0 + (lf - lo) / span * w)

    # decade gridlines + ticks
    for gf, lbl in ((100.0, "100"), (1000.0, "1k"), (10000.0, "10k")):
        gx = fx(gf)
        pygame.draw.line(screen, C_GRID, (gx, top_y), (gx, base_y))
        screen.blit(small.render(lbl, True, C_DIM), (gx + 2, rect.top + 1))

    # carrier f0 anchor (subtle -- shows where the pitch is anchored)
    if f0_hz > 0.0:
        fxa = fx(f0_hz)
        pygame.draw.line(screen, C_EDGE, (fxa, top_y), (fxa, base_y))

    # per-voice mode bars (coloured by field object)
    for v in voices:
        col = color.get(v['label_idx'], C_DIM)
        for f, a in zip(v['freqs'], v['amps']):
            f, a = float(f), float(a)
            if f <= 0.0 or a <= 0.0:
                continue
            bh = int(min(a, 1.0) * usable_h)
            if bh > 0:
                bx = fx(f)
                pygame.draw.line(screen, col, (bx, base_y), (bx, base_y - bh))


def draw_frame(screen, fonts, state, lay, rt):
    """Paint one frame.  `lay` = static layout geometry (SimpleNamespace built once
    in main()); `rt` = per-frame runtime values (SimpleNamespace).  Does NOT flip
    the display -- the caller does, so a frame can be captured before flipping."""
    font, small = fonts

    # Static geometry (bound to the SAME objects main()'s event loop uses).
    W              = lay.W
    buttons        = lay.buttons
    _pat_btn       = lay.pat_btn
    _bpm_x         = lay.bpm_x
    bpm_track      = lay.bpm_track
    div_buttons    = lay.div_buttons
    legend_x       = lay.legend_x
    info_y         = lay.info_y
    _rc_x          = lay.rc_x
    _vol_section_y = lay.vol_section_y
    vol_track      = lay.vol_track
    _rc_track_x    = lay.rc_track_x
    _RC_TRACK_W    = lay.rc_track_w
    _VOICE_HDR_RC_Y = lay.voice_hdr_rc_y
    _GEN_HDR_RC_Y   = lay.gen_hdr_rc_y
    _slew_btn      = lay.slew_btn
    _hold_btn      = getattr(lay, 'hold_btn', None)
    ctrls          = lay.ctrls
    meter_track    = lay.meter_track
    _midi_btn      = lay.midi_btn
    _mf_btn        = lay.mf_btn
    _audio_btn     = getattr(lay, 'audio_btn', None)
    _AUDIO_BTN_W   = getattr(lay, 'audio_btn_w', 0)
    _MIDI_BTN_W    = lay.midi_btn_w
    _MIDI_DD_ITH   = lay.midi_dd_ith
    engine_tabs    = lay.engine_tabs
    _sb_items      = lay.sb_items
    _sb_content_h  = lay.sb_content_h
    _sb_scroll_min = lay.sb_scroll_min

    # Per-frame runtime values.
    grid                = rt.grid
    color               = rt.color
    labels              = rt.labels
    voices              = rt.voices
    drag                = rt.drag
    _ghost              = rt.ghost
    white_keys          = rt.white_keys
    black_keys          = rt.black_keys
    _sb_scroll          = rt.sb_scroll
    meter               = rt.meter
    audio_ok            = rt.audio_ok
    midi_in             = rt.midi_in
    midifile            = rt.midifile
    _midi_dropdown_open = rt.midi_dropdown_open
    _midi_dd_items      = rt.midi_dd_items
    _audio_dropdown_open = getattr(rt, 'audio_dropdown_open', False)
    _audio_dd_items      = getattr(rt, 'audio_dd_items', [])
    _audio_dd_rects      = getattr(rt, 'audio_dd_rects', [])
    _audio_name          = getattr(rt, 'audio_name', None)
    _audio_state         = getattr(rt, 'audio_state', 'pending')
    _midi_dd_rects      = rt.midi_dd_rects

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
    st = (f"{note_name(state['note'])} {midi_to_freq(state['note']):.0f}Hz  "
          f"gen {state['gen']:>4}  "
          f"{state['bpm']} BPM {_div_lbl}  "
          f"obj {len(voices)}")
    screen.blit(font.render(st, True, C_TXT), (sx, info_y + 1))
    mode = "RUNNING" if state['run'] else "PAUSED"
    mode_surf = font.render(mode, True, C_ACCENT if state['run'] else C_DIM)
    screen.blit(mode_surf, (sx, info_y + 19))
    kbd_surf = small.render(f"kbd:{note_name(state['kb_base'])}", True, C_DIM)
    screen.blit(kbd_surf, (sx + mode_surf.get_width() + 10, info_y + 21))
    # what the last Save did (2026-09-22); absent until something is saved, so a
    # frame dump of a fresh window is unchanged
    _msg = getattr(rt, 'message', '')
    if _msg:
        screen.blit(small.render(_msg[:96], True, C_DIM),
                    (sx + mode_surf.get_width() + 10 + kbd_surf.get_width() + 14,
                     info_y + 21))

    # info row: volume (right column, below level meter)
    screen.blit(small.render(f"vol {int(state['vol'] * 100)}%", True, C_DIM),
                (_rc_x, _vol_section_y + 47))
    pygame.draw.rect(screen, C_BTN, vol_track, border_radius=4)
    fw = int(vol_track.width * state['vol'])
    pygame.draw.rect(screen, C_ACCENT,
                     (vol_track.left, vol_track.top, fw, vol_track.height),
                     border_radius=4)
    pygame.draw.circle(screen, C_TXT, (vol_track.left + fw, vol_track.centery), 6)
    # ── what one block of sound COSTS (2026-09-21) ───────────────────────────
    # An instrument that cannot hold real time has to say so where the player
    # looks, not hide it in a counter nobody reads: the smoothed milliseconds per
    # block against the budget, the worst block since this engine started, the
    # underruns the device reported, and the look-ahead a heavy engine earned.
    _bud = getattr(rt, 'budget', None)
    if _bud is not None:
        _cap = CHUNK_S * 1000.0
        _hot = (_bud['ms'] >= _cap * BUDGET_RAISE_FRAC) or _bud['underruns']
        _txt = (f"{_bud['ms']:.1f}/{_cap:.0f} max{_bud['max']:.0f} ur{_bud['underruns']}")
        if _bud['lookahead'] != AUDIO_LOOKAHEAD_CHUNKS:
            _txt += f" la{_bud['lookahead']}"
        screen.blit(small.render(_txt, True, (235, 96, 96) if _hot else C_DIM),
                    (_rc_x, vol_track.bottom + 10))
    if not audio_ok:
        screen.blit(small.render("audio disabled", True, C_DIM), (W - 110, 6))

    # Two ADSR section headers in the right column: VOICE (note-on/off VCA) above
    # its 4 sliders, GEN (per-mode / automaton-clock envelope) above its 4.
    _env_rc_right = _rc_track_x + _RC_TRACK_W + 44
    for _hy, _hlbl in ((_VOICE_HDR_RC_Y, "VOICE"), (_GEN_HDR_RC_Y, "GEN")):
        pygame.draw.line(screen, C_EDGE, (_rc_x, _hy), (_env_rc_right, _hy))
        screen.blit(small.render(_hlbl, True, C_DIM), (_rc_x, _hy + 2))
    # HOLD (right of the "VOICE" header): lit = a note latches and can only be
    # replaced; dark = letting go of the key is a note-off and the VOICE release
    # runs.  Same widget as slew, one block up.
    _hold_on = bool(state.get('voice_hold', False))
    if _hold_btn is not None:
        pygame.draw.rect(screen, C_ACCENT if _hold_on else C_BTN, _hold_btn,
                         border_radius=3)
        screen.blit(small.render("hold", True, C_BG if _hold_on else C_DIM),
                    (_hold_btn.x + 6, _hold_btn.y + 1))
    # GEN amp-slew toggle (right of the "GEN" header): lit when on.  Smooths
    # stable-pitch amplitude gating (the shape>0 beep); off = bit-exact legacy.
    # An engine that has no slew path says so by going flat instead of offering
    # a toggle that does nothing (rt.slew_live, from the registry).
    _slew_live = bool(getattr(rt, 'slew_live', True))
    _slew_on = state['gen_amp_slew'] and _slew_live
    pygame.draw.rect(screen, C_ACCENT if _slew_on else C_BTN, _slew_btn,
                     border_radius=3)
    screen.blit(small.render("slew", True, (C_BG if _slew_on else
                                            (C_DIM if _slew_live else C_CAPPED))),
                (_slew_btn.x + 6, _slew_btn.y + 1))
    for c in ctrls:
        tr = c['track']
        val = _ctrl_value(state, c)
        kind = c.get('kind', 'slider')
        _lw = c.get('label_w')
        if _lw:                       # a long label stops where its column does
            screen.set_clip(pygame.Rect(c['label_x'], tr.top - 8, _lw, 22))
        screen.blit(small.render(c['label'], True, C_DIM),
                    (c['label_x'], tr.centery - 7))
        screen.set_clip(None)
        if kind == 'inactive':
            # the parameter does not act for this engine: say why, where the
            # widget would have been -- the bench's rule (casynth_panel), so the
            # two panels read the same way
            screen.blit(small.render(c.get('text', ''), True, C_DIM),
                        (tr.left, tr.centery - 7))
        elif kind == 'toggle':
            # an on/off integer is a pill, as on the bench (2026-09-21)
            pill = c['pill']
            pygame.draw.rect(screen, C_ACCENT if val else C_BTN, pill, border_radius=8)
            pygame.draw.rect(screen, C_EDGE, pill, 1, border_radius=8)
            screen.blit(small.render("on" if val else "off", True, C_DIM),
                        (pill.right + 6, tr.centery - 7))
        elif kind == 'choices':
            # a mode named by words: one button per word, the current one lit
            for value, rect in c['word_rects']:
                on = (int(val) == value)
                pygame.draw.rect(screen, C_ACCENT if on else C_BTN, rect, border_radius=3)
                pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=3)
                word = small.render(c['choices'][value], True, C_BG if on else C_DIM)
                screen.set_clip(rect)
                screen.blit(word, (rect.centerx - word.get_width() // 2,
                                   rect.centery - word.get_height() // 2))
                screen.set_clip(None)
        else:
            pygame.draw.rect(screen, C_BTN, tr, border_radius=4)
            span = c['hi'] - c['lo']
            frac = float(np.clip((val - c['lo']) / span, 0, 1)) if span else 0.0
            cw = int(tr.width * frac)
            pygame.draw.rect(screen, C_ACCENT, (tr.left, tr.top, cw, tr.height),
                             border_radius=4)
            pygame.draw.circle(screen, C_TXT, (tr.left + cw, tr.centery), 5)
            rects = c.get('range_rects')
            if not rects:
                screen.blit(small.render(c['fmt'](val), True, C_DIM),
                            (tr.right + 6, tr.centery - 7))
            else:
                # Min / Max around the track, the value on the right INSIDE it:
                # the two fields take the column's spare pixels, and the number
                # has nowhere else to go (gol_synth, 2026-09-22).
                _draw_range_fields(screen, small, c, rt)
                vtxt = small.render(c['fmt'](val), True, C_TXT)
                vx = tr.right - vtxt.get_width() - 4
                shade = pygame.Surface((vtxt.get_width() + 6, tr.height + 6))
                shade.set_alpha(190)
                shade.fill(C_BG)                 # the number stays readable over
                shade_r = shade.get_rect()       # the filled part of the track
                shade_r.topright = (tr.right, tr.top - 3)
                screen.blit(shade, shade_r)
                screen.blit(vtxt, (vx, tr.centery - 7))

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

    # ── spectrum strip: engine mode bars (read-only viz of voices) ─────────
    _draw_spectrum(screen, small, lay.spec_rect, voices, color,
                   midi_to_freq(state['note']))

    # ── MIDI device bar (above engine tabs) ───────────────────────────────
    if MIDI_AVAILABLE:
        hot_midi = _midi_btn.collidepoint(mouse)
        pygame.draw.rect(screen, C_BTN_HOT if hot_midi else C_BTN,
                         _midi_btn, border_radius=3)
        dot_col = (80, 200, 120) if midi_in.port is not None else C_DIM
        pygame.draw.circle(screen, dot_col,
                           (_midi_btn.left + 8, _midi_btn.centery), 4)
        _port_disp = midi_in.port_name or "—"
        if len(_port_disp) > 30:
            _port_disp = _port_disp[:29] + "…"
        _midi_lbl = small.render(f"MIDI: {_port_disp}  ▾", True, C_TXT)
        screen.blit(_midi_lbl,
                    (_midi_btn.left + 18,
                     _midi_btn.centery - _midi_lbl.get_height() // 2))

    # ── MIDI-file transport button (right of the device bar) ──────────────
    if MIDIFILE_AVAILABLE:
        hot_mf = _mf_btn.collidepoint(mouse)
        playing = midifile.playing
        pygame.draw.rect(screen, C_ACCENT if playing
                         else (C_BTN_HOT if hot_mf else C_BTN),
                         _mf_btn, border_radius=3)
        if playing:
            def _ms(s):
                s = max(0, int(s)); return f"{s // 60}:{s % 60:02d}"
            _mf_txt = f"■ {_ms(midifile.pos)}/{_ms(midifile.length)}"
        elif midifile.name:
            _mf_txt = f"▶ {midifile.name}"
        else:
            _mf_txt = "♪ Play MIDI file"
        if len(_mf_txt) > 24:
            _mf_txt = _mf_txt[:23] + "…"
        _mf_lbl = small.render(_mf_txt, True, C_BG if playing else C_TXT)
        screen.blit(_mf_lbl,
                    (_mf_btn.left + 6,
                     _mf_btn.centery - _mf_lbl.get_height() // 2))

    # ── OUTPUT selector (right of the MIDI-file button) ───────────────────
    # Where the sound actually goes.  A machine can have a headset, a monitor and
    # onboard speakers at once, and an endpoint can accept the stream and play
    # nothing audible -- so the device is named here, in the window, not only in
    # a console nobody has open.
    if _audio_btn is not None:
        hot_out = _audio_btn.collidepoint(mouse)
        pygame.draw.rect(screen, C_BTN_HOT if hot_out else C_BTN,
                         _audio_btn, border_radius=3)
        _out_dot = {'ok': (80, 200, 120), 'failed': (212, 76, 76)}.get(_audio_state, C_DIM)
        pygame.draw.circle(screen, _out_dot,
                           (_audio_btn.left + 8, _audio_btn.centery), 4)
        _out_disp = _audio_name or "no device"
        if len(_out_disp) > 34:
            _out_disp = _out_disp[:33] + "…"
        _out_lbl = small.render(f"OUT: {_out_disp}  ▾", True, C_TXT)
        screen.blit(_out_lbl, (_audio_btn.left + 18,
                               _audio_btn.centery - _out_lbl.get_height() // 2))

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

    # ── MIDI dropdown overlay (drawn last so it floats above everything) ──
    if _audio_dropdown_open and _audio_dd_items and _audio_btn is not None:
        _odd = pygame.Rect(_audio_btn.left, _audio_btn.bottom + 1, _AUDIO_BTN_W,
                           len(_audio_dd_items) * _MIDI_DD_ITH + 4)
        pygame.draw.rect(screen, C_PANEL, _odd, border_radius=4)
        pygame.draw.rect(screen, C_EDGE, _odd, 1, border_radius=4)
        for item_rect, (dev, label) in zip(_audio_dd_rects, _audio_dd_items):
            if item_rect.collidepoint(mouse):
                pygame.draw.rect(screen, C_BTN_HOT, item_rect, border_radius=3)
            disp = label if len(label) <= 40 else label[:39] + "…"
            chosen = (_audio_state == 'ok' and _audio_name is not None
                      and dev is not None and _audio_name.split(' [')[0][:20] in label)
            screen.blit(small.render(disp, True, C_ACCENT if chosen else C_TXT),
                        (item_rect.left + 4,
                         item_rect.centery - small.get_height() // 2))

    if _midi_dropdown_open and MIDI_AVAILABLE and _midi_dd_items:
        _dd_rect = pygame.Rect(_midi_btn.left,
                               _midi_btn.bottom + 1,
                               _MIDI_BTN_W,
                               len(_midi_dd_items) * _MIDI_DD_ITH + 4)
        pygame.draw.rect(screen, C_PANEL, _dd_rect, border_radius=4)
        pygame.draw.rect(screen, C_EDGE, _dd_rect, 1, border_radius=4)
        for di, (item_rect, port) in enumerate(zip(_midi_dd_rects, _midi_dd_items)):
            hot_dd = item_rect.collidepoint(mouse)
            if hot_dd:
                pygame.draw.rect(screen, C_BTN_HOT, item_rect, border_radius=3)
            disp = port or "— (none)"
            if len(disp) > 34:
                disp = disp[:33] + "…"
            selected = (port == midi_in.port_name) or (port is None and midi_in.port is None)
            _dlbl = small.render(disp, True, C_ACCENT if selected else C_TXT)
            screen.blit(_dlbl,
                        (item_rect.left + 4,
                         item_rect.centery - _dlbl.get_height() // 2))
