#!/usr/bin/env python3
"""Demo bench (S1 + S2): one live A/B demo window, or an offline render.

    python demo_bench.py --demo demos/laplace_ab.json
    python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor

Window: shared field, demo title + listening hint, transport (Start|Stop /
Pause CA / Restart), volume, generation, audio state; a side panel with the
A/B tabs (select = listen + edit), the selected side's engine and its registry
knobs, and a one-line summary of how A and B differ.  LMB paints, RMB erases;
Clear (C) empties the field; the Patterns column (patterns.py, the synth's
library) drags Life shapes onto the field.  Every text field (save Title /
Note, Notes, the Min / Max range fields of a ranged knob) is one model: caret,
selection, Ctrl+A/X/C/V.  A ranged knob (registry EngineSpec.ranges, Objects
"Radius x", 2026-09-17) gets a "range" row under its slider with two numeric
fields: the slider is linear over min..max of the SELECTED side (runner
set_range, saved with the record / Continue); Enter or leaving the field
commits, Esc cancels, a bad number is refused in place; a committed range that
excludes the current value clamps it once (a normal set_param).
Startup stands on pause (silent); releasing Pause CA starts automaton + sound.
Stop (S) = reset + pause + silence; Pause CA freezes only the automaton while
running; Restart (R) starts again immediately.  Stop/Restart keep engines,
params, volume and side.

All computation lives in casynth_lab (DemoRunner); this file is UI + CLI only.
The offline path imports neither pygame nor sounddevice.
stdout stays ASCII (Windows console codepage).
"""
import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casynth_config import (VOL_DEFAULT, C_BG, C_GRID, C_PANEL, C_EDGE, C_TXT,
                            C_DIM, C_BTN, C_ACCENT)                    # noqa: E402
from casynth_lab import (load_scene, SceneError, DemoRunner, SIDES, registry,
                         describe_difference, validate_param)           # noqa: E402
from casynth_lab.catalog import Catalog, CatalogError, bench_scene     # noqa: E402
from casynth_lab import provenance as prov                             # noqa: E402
from casynth_lab.textedit import TextEdit                              # noqa: E402
from casynth_lab.versions import VersionError                          # noqa: E402
from casynth_lab import notes_window as _notes_win                     # noqa: E402
from patterns import PATTERNS                                          # noqa: E402
from casynth_lab import verify as verify_mod                           # noqa: E402
from casynth_lab.verify import (Verifier, VerifyError, TRACK_EXACT, TRACK_DIFFERS,   # noqa: E402
                                TRACK_FAILED, TRACK_UNCHECKED, TRACK_RESULTS,
                                TRACK_LABELS, RESULT_LABELS, MARK_SAME, MARK_DIFFERENT,
                                track_text)

# -- layout ------------------------------------------------------------------
CELL = 16
MARGIN = 16
TOP_H = 156
BTN_W, BTN_H = 130, 32
BTN_Y = 58
VOL_W = 140
PANEL_W = 270
TAB_W, TAB_H = 96, 30
ARROW_W = 28
ENG_W, ENG_H = 74, 24
ROW_H = 24                   # 26 until 2026-09-17: Objects has 13 knobs + a range row, the window stays < 1000 px
DISPLAY_ROW_H = 14           # one bar row of an engine display (W links / node u)
SLIDER_W = 120
RANGE_FIELD_W = 54           # Min / Max field of a ranged knob (2026-09-17)
RANGE_DECIMALS = 3           # slider rounding / value text of a ranged knob (steps of 0.001)
FONT_NAMES = "segoeui,arial,dejavusans,freesans"
C_ALIVE = (111, 208, 224)
C_ALIVE_PAUSED = (200, 180, 90)
C_BTN_ON = (48, 92, 104)
C_ERR = (230, 120, 90)
C_WARN = (230, 180, 90)
C_OK = (120, 210, 140)
C_OVERLAY = (235, 120, 175)     # S/N demos: reading path / link masks over the field
# N4 (2026-09-16): one stable colour per tracked figure (cells, circle, centre);
# index = display()['figures'][k]['color'] of the engine (id-based, never reused live)
C_FIGURES = ((235, 120, 175), (120, 200, 255), (255, 200, 90), (140, 230, 140),
             (255, 140, 100), (190, 150, 255), (90, 220, 210), (255, 240, 140),
             (240, 150, 210), (150, 190, 120), (200, 170, 130), (130, 160, 240))
LAB_BTN_W = 104
ROW_LIST_H = 50
# pattern library (2026-09-16): a column right of the side panel, items are
# dragged onto the field (the synth's Lib sidebar)
LIB_W = 176
LIB_ITEM_H = 26
LIB_HDR_H = 18
LIB_PREV_W, LIB_PREV_H = 30, 22
C_GHOST = (111, 208, 224, 110)   # dragged pattern over the field (alpha)
C_SEL = (58, 96, 138)            # text selection
FALLBACK_CHAR_W = 7              # text metrics before a font is bound (tests)


def _is_toggle(spec):
    _arg, _label, lo, hi, integer, _d = spec
    return bool(integer) and lo == 0 and hi == 1


class BenchApp:
    """Pure UI state machine over a LiveEngine -- drivable without a display
    (tests call press/drag/release/draw with SDL_VIDEODRIVER=dummy)."""

    def __init__(self, scene, engine, catalog=None):
        self.scene = scene
        self.engine = engine
        self.catalog = catalog            # Catalog or None (S4 disabled)
        self.verifier = Verifier(catalog) if catalog is not None else None   # S7
        self.child = None                 # ChildBench of another version (S6)
        self.mode = 'live'                # 'live' | 'save' | 'catalog' | 'report'
        self.status = ""                  # save / replay status line
        self.save_form = None             # {'cut', 'title', 'note', 'field'}
        self._cut_pending = False
        self._save_thread = None
        self.cat = dict(entries=[], sel=None, result=None, thread=None,
                        progress=0.0, cancel=threading.Event(), play_label="")
        # S7: catalog check state + the report screen
        self.verify = dict(thread=None, cancel=threading.Event(), progress=0.0, index=0,
                           total=0, current='', target='', done=None, error=None,
                           run=None, group=TRACK_DIFFERS, sel=None, track='monitor',
                           version='saved', details=False)
        self.vol = VOL_DEFAULT
        self.paint_value = None          # None / 1 (LMB) / 0 (RMB) while dragging
        self.drag_vol = False
        self.drag_param = None           # param name while dragging its slider
        self.message = ""                # last rejected command, shown in the panel
        self.field_w = scene.cols * CELL
        self.field_h = scene.rows * CELL
        self.field_x = MARGIN
        self.field_y = TOP_H
        self.panel_x = MARGIN + self.field_w + MARGIN
        self.width = self.panel_x + PANEL_W + MARGIN
        self.height = TOP_H + max(self.field_h, 420) + 20 + MARGIN   # +20: overlay caption
        self.buttons = {}
        x = MARGIN
        for key, label in (('stop', 'Stop (S)'), ('pause', 'Pause CA (Space)'),
                           ('reset', 'Restart (R)'), ('clear', 'Clear (C)')):
            self.buttons[key] = ((x, BTN_Y, BTN_W, BTN_H), label)
            x += BTN_W + 10
        self.vol_rect = (MARGIN + 92, BTN_Y + BTN_H + 14, VOL_W, 10)
        self.lab_buttons = {'save': (x + 10, BTN_Y, LAB_BTN_W, BTN_H),
                            'catalog': (x + 20 + LAB_BTN_W, BTN_Y, LAB_BTN_W, BTN_H),
                            'notes': (x + 30 + 2 * LAB_BTN_W, BTN_Y, LAB_BTN_W, BTN_H)}
        # the top row must fit (four transport buttons + three lab buttons)
        self.width = max(self.width, x + 30 + 3 * LAB_BTN_W + MARGIN)
        # listening notes (2026-09-14): the record this session belongs to --
        # the record continued / opened, or the last one saved here
        self.session_record = None
        self.notes_form = None            # {'rid', 'title', 'text'} while the notes window is open
        self.notes_window = None          # casynth_lab.notes_window.NotesWindow (a window of its own, 2026-09-17)
        # text fields (2026-09-16): ONE model for every input (casynth_lab.textedit):
        # caret, click / arrows / Shift selection, Ctrl+A/X/C/V, word rules
        self.save_edits = None            # {'title': TextEdit, 'note': TextEdit} with the form
        self.notes_edit = None            # TextEdit of the Notes window
        self.drag_text = None             # (edit, rect) while the mouse selects text
        self.measure = lambda s: FALLBACK_CHAR_W * len(s)   # text width (font-bound in draw)
        self.line_h = 16
        self._font_bound = None
        self.clipboard = ''               # the bench's own clipboard (no system one hooked)
        self.clip_get = None              # callables hooked by main (pygame.scrap)
        self.clip_put = None
        # side panel
        px, py = self.panel_x, TOP_H
        # row: [A] [<<] [>>] [B]
        self.tabs = {'A': (px, py, TAB_W, TAB_H),
                     'B': (px + TAB_W + 2 * (ARROW_W + 4) + 4, py, TAB_W, TAB_H)}
        self.copy_btns = {('B', 'A'): (px + TAB_W + 4, py, ARROW_W, TAB_H),          # <<
                          ('A', 'B'): (px + TAB_W + ARROW_W + 8, py, ARROW_W, TAB_H)}  # >>
        self.factory_btn = (px + PANEL_W - 100, py + TAB_H + 6, 100, 22)
        # copy spectrum (2026-09-17): only the spectrum settings both sides' engines
        # share (n spread alpha shape harm fullshape dyn); << = B to A, >> = A to B
        sy = py + TAB_H + 44
        self.spec_btns = {('B', 'A'): (px, sy, 118, 20),
                          ('A', 'B'): (px + 124, sy, 118, 20)}
        # engine buttons: 3 per row, as many rows as the registry needs; the
        # parameter rows start below the LAST row (S/N demos: 7 engines)
        self.engine_btns = {}
        ey = sy + 26
        specs = registry.specs()
        for i, e in enumerate(specs):
            col, row = i % 3, i // 3
            self.engine_btns[e.id] = (px + col * (ENG_W + 6), ey + row * (ENG_H + 6),
                                         ENG_W, ENG_H)
        self.engine_rows = (len(specs) + 2) // 3
        self.params_y = ey + self.engine_rows * (ENG_H + 6) + 12
        self.param_rows_max = max([len(e.params) + len(e.ranges) for e in specs] + [1])
        # peak / message line: below the longest parameter list AND below the tallest
        # engine display (gutter_field: 4 parameter rows + a header + 8 node bars)
        self.footer_y = max(self.params_y + self.param_rows_max * ROW_H + 8,
                            self.params_y + 4 * ROW_H + 6 + 18 + 8 * DISPLAY_ROW_H + 8,
                            # N4 Objects: 12 parameter rows + the figure rows of its display
                            self.params_y + self.param_rows_max * ROW_H + 6 + 34 + 8 * DISPLAY_ROW_H + 8)
        self.height = max(self.height, self.footer_y + 44 + MARGIN)
        self.slider_x = px + 82
        # pattern library (2026-09-16): the synth's Lib sidebar -- a column
        # right of the panel; an item is dragged onto the field and dropped
        # (torus wrap) as ONE set_cells; Esc / a release off the field cancels
        self.lib_x = self.panel_x + PANEL_W + MARGIN
        self.width = max(self.width, self.lib_x + LIB_W + MARGIN)
        self.lib_rect = (self.lib_x, TOP_H, LIB_W, self.height - TOP_H - MARGIN)
        self.lib_items = []               # {'rect' (at scroll 0), 'cells', 'name'}
        self.lib_heads = []               # (category, y at scroll 0)
        y = TOP_H + 28
        for cat, pats in PATTERNS:
            self.lib_heads.append((cat, y))
            y += LIB_HDR_H + 2
            for name, cells in pats:
                self.lib_items.append(dict(rect=(self.lib_x + 4, y, LIB_W - 8, LIB_ITEM_H),
                                           cells=[tuple(c) for c in cells], name=name))
                y += LIB_ITEM_H + 2
        self.lib_scroll = 0
        self.lib_scroll_min = min(0, (self.lib_rect[1] + self.lib_rect[3] - 4) - y)
        self.drag_pat = None              # {'item', 'cells', 'name', 'snap'} while dragging
        self._ghost = None                # alpha surface of one ghost cell (made in draw)
        # ranged knobs (2026-09-17): the focused Min / Max field, if any
        self.range_edit = None            # {'name', 'which', 'edit': TextEdit, 'rect'}
        self.range_error = None           # (name, text) shown beside the fields until the next edit
        self._range_pending = {}          # (side, name) -> (min, max) posted, not yet in the snapshot
        self._clamp_pending = {}          # (side, name) -> value posted by a clamp, not yet in the snapshot

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _inside(rect, pos):
        x, y, w, h = rect
        return x <= pos[0] < x + w and y <= pos[1] < y + h

    def cell_at(self, pos):
        c = (pos[0] - self.field_x) // CELL
        r = (pos[1] - self.field_y) // CELL
        if 0 <= r < self.scene.rows and 0 <= c < self.scene.cols:
            return int(r), int(c)
        return None

    def _side(self):
        snap = self.engine.snapshot()
        side = snap['selected']
        eid, params = snap['sides'][side]
        return side, eid, params

    def _param_rows(self, eid):
        """[(spec, slider_rect)] for the selected side's engine; a ranged
        parameter is followed by its extra "range" row."""
        rows = []
        e = registry.get(eid)
        y = self.params_y
        for spec in e.params:
            rows.append((spec, (self.slider_x, y + 6, SLIDER_W, 10)))
            y += ROW_H * (2 if spec[0] in e.ranges else 1)
        return rows

    def _params_height(self, eid):
        e = registry.get(eid)
        return (len(e.params) + len(e.ranges)) * ROW_H

    def _range_rects(self, eid):
        """{name: {'min': rect, 'max': rect, 'y': row y}} of the range rows."""
        out = {}
        e = registry.get(eid)
        for spec, (sx, sy, _w, _h) in self._param_rows(eid):
            if spec[0] in e.ranges:
                y = sy - 6 + ROW_H
                out[spec[0]] = dict(min=(sx, y + 1, RANGE_FIELD_W, 20),
                                    max=(sx + RANGE_FIELD_W + 14, y + 1, RANGE_FIELD_W, 20), y=y)
        return out

    def _range_of(self, snap, side, name):
        """(min, max) of a ranged parameter on a side: the runner's snapshot, or a
        range posted from here that the snapshot has not caught up with yet (two
        commits within one audio block must not overwrite each other)."""
        rr = (snap.get('ranges') or {}).get(side, {})
        got = (float(rr[name][0]), float(rr[name][1])) if name in rr             else registry.default_range(snap['sides'][side][0], name)
        pend = self._range_pending.get((side, name))
        if pend is not None:
            if pend == got:
                del self._range_pending[(side, name)]
            else:
                return pend
        return got

    def _choice_rects(self, eid, name, sy):
        """Word buttons of a named-choice parameter on the row whose slider
        would sit at y = sy: [(value, rect)]."""
        names = registry.get(eid).choices.get(name, ())
        n = max(1, len(names))
        w = min(60, (PANEL_W - 82 - 4 * (n - 1)) // n)
        return [(v, (self.slider_x + v * (w + 4), sy - 4, w, 18)) for v in range(len(names))]

    @staticmethod
    def _inactive(eid, params):
        fn = registry.get(eid).inactive
        return fn(params) if fn is not None else {}

    def _post(self, kind, **args):
        try:
            self.engine.post(kind, **args)
            self.message = ""
            return True
        except ValueError as e:
            self.message = str(e)
            return False

    # -- input ---------------------------------------------------------------
    def press(self, pos, button, now=None, shift=False):
        """Mouse button down. button: 1 = left, 3 = right.  Returns what was hit.
        `now` (seconds, monotonic) only serves double-click detection in the
        catalog; tests may pass it explicitly.  `shift`: Shift is held (a
        click in a text field extends the selection)."""
        if self.mode == 'save':
            return self._press_save_form(pos, button, shift)
        if self.mode == 'catalog':
            return self._press_catalog(pos, button, now)
        if self.mode == 'report':
            return self._press_report(pos, button)
        return self._press_live(pos, button, allow_lab=True, shift=shift)

    def _press_live(self, pos, button, allow_lab=True, shift=False):
        """The live view (transport, A/B, knobs, painting; lab buttons when
        allowed -- the Notes window passes clicks outside it here without them)."""
        if self.range_edit is not None:
            if self._inside(self.range_edit['rect'], pos) and button == 1:
                self._caret_from_pos(self.range_edit['edit'], self.range_edit['rect'], pos,
                                     shift=shift, start_drag=True)
                return f"range:{self.range_edit['name']}:{self.range_edit['which']}"
            self.commit_range()                     # leaving the field commits
        if button == 1 and self.catalog is not None and allow_lab:
            for key, rect in self.lab_buttons.items():
                if self._inside(rect, pos):
                    if key == 'save':
                        self.begin_save()
                    elif key == 'notes':
                        self.open_notes()
                    else:
                        self.open_catalog()
                    return key
        if button == 1 and self._inside(self.lib_rect, pos) and pos[1] >= self.lib_rect[1] + 28:
            for it in self.lib_items:
                x, y, w, h = it['rect']
                if self._inside((x, y + self.lib_scroll, w, h), pos):
                    self.drag_pat = dict(item=it, cells=it['cells'], name=it['name'], snap=None)
                    return f"lib:{it['name']}"
            return None
        if button == 1:
            for key, (rect, _label) in self.buttons.items():
                if self._inside(rect, pos):
                    self._post(key)
                    return key
            vx, vy, vw, vh = self.vol_rect
            if vx - 6 <= pos[0] < vx + vw + 6 and vy - 8 <= pos[1] < vy + vh + 8:
                self.drag_vol = True
                self._set_vol(pos[0])
                return 'vol'
            for s, rect in self.tabs.items():
                if self._inside(rect, pos):
                    self._post('select', side=s)
                    return f'tab:{s}'
            for (src, dst), rect in self.copy_btns.items():
                if self._inside(rect, pos):
                    self._post('copy_side', src=src, dst=dst)
                    return f'copy:{src}{dst}'
            for (src, dst), rect in self.spec_btns.items():
                if self._inside(rect, pos):
                    self._post('copy_spectrum', src=src, dst=dst)
                    return f'spectrum:{src}{dst}'
            if self._inside(self.factory_btn, pos):
                self._post('factory')
                return 'factory'
            side, eid, params = self._side()
            for e_id, rect in self.engine_btns.items():
                if self._inside(rect, pos):
                    self._post('set_engine', side=side, engine_id=e_id)
                    return f'engine:{e_id}'
            inactive = self._inactive(eid, params)
            choices = registry.get(eid).choices
            for name, rr in self._range_rects(eid).items():
                if name in inactive:
                    continue
                for which in ('min', 'max'):
                    if self._inside(rr[which], pos):
                        self._focus_range(name, which, rr[which])
                        return f'range:{name}:{which}'
            for spec, (sx, sy, sw, sh) in self._param_rows(eid):
                name = spec[0]
                if name in inactive:
                    continue                          # shown as text, not editable
                if name in choices:
                    for value, rect in self._choice_rects(eid, name, sy):
                        if self._inside(rect, pos) and value != params[name]:
                            self._post('set_param', side=side, name=name, value=value)
                            return f'param:{name}'
                    continue
                if sx - 6 <= pos[0] < sx + sw + 6 and sy - 8 <= pos[1] < sy + sh + 8:
                    if _is_toggle(spec):
                        self._post('set_param', side=side, name=name,
                                   value=0 if params[name] else 1)
                    else:
                        self.drag_param = name
                        self._set_param_from_x(pos[0])
                    return f'param:{name}'
        if button in (1, 3):
            cell = self.cell_at(pos)
            if cell is not None:
                self.paint_value = 1 if button == 1 else 0
                self._paint(cell)
                return 'paint'
        return None

    def drag(self, pos):
        if self.drag_text is not None:
            edit, rect = self.drag_text
            self._caret_from_pos(edit, rect, pos, shift=True)
        elif self.drag_pat is not None:
            self.drag_pat['snap'] = self.cell_at(pos)
        elif self.drag_vol:
            self._set_vol(pos[0])
        elif self.drag_param is not None:
            self._set_param_from_x(pos[0])
        elif self.paint_value is not None:
            cell = self.cell_at(pos)
            if cell is not None:
                self._paint(cell)

    def key(self, name, ctrl=False, shift=False):
        """Keyboard hotkey by key name ('r' = Restart).  Returns the command or None.
        `ctrl` / `shift`: modifiers held (text editing: Ctrl+Backspace, Ctrl+A/X/C/V,
        Shift+arrows select)."""
        name = name.lower()
        if self.mode == 'save':
            # every text field: the same editing keys before the form's own
            edit, rect = self._active_edit()
            if edit is not None:
                got = self._edit_key(edit, rect, name, ctrl, shift)
                if got is not None:
                    return f"{self.mode}:{got}"
            return self._key_save_form(name, ctrl)
        if self.mode == 'catalog':
            if name == 'escape':
                self.close_catalog()
                return 'back'
            return None
        if self.mode == 'report':
            if name == 'escape':
                self.close_report()
                return 'report:back'
            if name == 'space':
                v = 'recomputed' if self.verify['version'] == 'saved' else 'saved'
                self.set_version(v)
                return f'version:{v}'
            return None
        if name == 'escape' and self.drag_pat is not None:
            self.cancel_drag()
            return 'drag:cancel'
        if self.range_edit is not None:
            # a focused Min / Max field: editing keys, Enter commits, Esc cancels,
            # Tab moves to the other bound; the bench hotkeys stay inert meanwhile
            re = self.range_edit
            if name in ('return', 'enter', 'kp_enter'):
                return 'range:commit' if self.commit_range() else 'range:error'
            if name == 'escape':
                self.cancel_range()
                return 'range:cancel'
            if name == 'tab':
                other = 'max' if re['which'] == 'min' else 'min'
                _side, eid, _params = self._side()
                rect = self._range_rects(eid)[re['name']][other]
                if self.commit_range():
                    self._focus_range(re['name'], other, rect)
                return f"range:{re['name']}:{other}"
            got = self._edit_key(re['edit'], re['rect'], name, ctrl, shift)
            return f"range:{got}" if got else 'range:typing'
        if name == 'r':
            self._post('reset')
            return 'reset'
        if name == 's':
            self._post('stop')
            return 'stop'
        if name == 'c':
            self._post('clear')
            return 'clear'
        if name == 'space':
            self._post('pause')
            return 'pause'
        if name in ('1', '2', '[1]', '[2]'):
            side = 'A' if name.endswith('1') or name == '1' else 'B'
            self._post('select', side=side)
            return f'select:{side}'
        return None

    def release(self):
        """Mouse button up: ends painting / slider / text drags; a dragged
        pattern is dropped where it last snapped (off the field = cancelled)."""
        self.paint_value = None
        self.drag_vol = False
        self.drag_param = None
        self.drag_text = None
        if self.range_edit is not None:
            self.range_edit['edit'].follow = True
        if self.drag_pat is not None:
            d, self.drag_pat = self.drag_pat, None
            if d['snap'] is not None:
                self.stamp(d['cells'], d['snap'])
                return f"drop:{d['name']}"
        return None

    def _paint(self, cell):
        self._post('set_cell', r=cell[0], c=cell[1], v=self.paint_value)

    def _set_vol(self, mx):
        vx, _vy, vw, _vh = self.vol_rect
        self.vol = min(max((mx - vx) / vw, 0.0), 1.0)
        self._post('vol', value=self.vol)

    def set_param(self, name, value):
        """Programmatic knob change on the selected side (validated by the runner)."""
        side, eid, params = self._side()
        return self._post('set_param', side=side, name=name, value=value)

    def _set_param_from_x(self, mx):
        side, eid, params = self._side()
        spec = registry.get(eid).spec_of(self.drag_param)
        _arg, _label, lo, hi, integer, _d = spec
        ranged = self.drag_param in registry.get(eid).ranges
        if ranged:
            lo, hi = self._range_of(self.engine.snapshot(), side, self.drag_param)
        frac = min(max((mx - self.slider_x) / SLIDER_W, 0.0), 1.0)
        v = lo + frac * (hi - lo)
        v = int(round(v)) if integer else round(v, RANGE_DECIMALS if ranged else 3)
        if ranged:
            v = min(max(v, lo), hi)                 # the rounded value stays inside the range
        if v != params[self.drag_param]:
            self._post('set_param', side=side, name=self.drag_param,
                       value=validate_param(eid, self.drag_param, v))

    # -- ranged knobs: the Min / Max fields ------------------------------------------
    @staticmethod
    def range_text(v):
        return f"{float(v):g}"

    def _focus_range(self, name, which, rect):
        side, _eid, _params = self._side()
        lo, hi = self._range_of(self.engine.snapshot(), side, name)
        edit = TextEdit(self.range_text(lo if which == 'min' else hi))
        edit.select_all()                       # the first click: typing replaces the number
        self.range_edit = dict(name=name, which=which, edit=edit, rect=rect)
        self.range_error = None

    def cancel_range(self):
        self.range_edit = None
        self.drag_text = None

    def commit_range(self):
        """Enter / leaving the field: parse the number, validate the pair
        (finite, spec lo <= min < max <= spec hi), post set_range; a value the
        new range excludes is clamped once to the nearest bound (set_param).
        A bad number leaves the range and shows the reason; returns True when
        the field was accepted (and closed)."""
        re = self.range_edit
        if re is None:
            return True
        side, eid, params = self._side()
        name, which = re['name'], re['which']
        lo, hi = self._range_of(self.engine.snapshot(), side, name)
        text = re['edit'].text.strip()
        try:
            v = float(text)
        except ValueError:
            v = float('nan')
        if not math.isfinite(v):
            self.range_error = (name, 'not a number')
            return False
        if which == 'min':
            lo = v
        else:
            hi = v
        spec = registry.get(eid).spec_of(name)
        if lo < spec[2]:
            self.range_error = (name, f'min < {spec[2]:g}')
            return False
        if hi > spec[3]:
            self.range_error = (name, f'max > {spec[3]:g}')
            return False
        if not lo < hi:
            self.range_error = (name, 'min must be < max')
            return False
        self.range_edit = None
        self.drag_text = None
        self.range_error = None
        cur = self._range_of(self.engine.snapshot(), side, name)
        if (lo, hi) != cur:
            if not self._post('set_range', side=side, name=name, lo=lo, hi=hi):
                self.range_error = (name, 'refused')
                return False
            self._range_pending[(side, name)] = (lo, hi)
        value = float(params[name])
        pend = self._clamp_pending.get((side, name))
        if pend is not None:                    # a clamp posted within this block counts once
            if pend == value:
                del self._clamp_pending[(side, name)]
            else:
                value = pend
        clamped = min(max(value, lo), hi)
        if clamped != value:
            self._post('set_param', side=side, name=name, value=validate_param(eid, name, clamped))
            self._clamp_pending[(side, name)] = clamped
        return True

    # -- pattern library -------------------------------------------------------
    def stamp(self, cells, at):
        """Set a pattern's live cells (offsets from its top-left corner) at
        cell `at` = (row, col) -- the field is a torus.  One command."""
        r0, c0 = at
        rows, cols = self.scene.rows, self.scene.cols
        return self._post('set_cells', cells=[[(r0 + dr) % rows, (c0 + dc) % cols, 1]
                                              for dr, dc in cells])

    def cancel_drag(self):
        self.drag_pat = None

    def _draw_library(self, screen, font, small):
        import pygame
        lx, ly, lw, lh = self.lib_rect
        pygame.draw.rect(screen, C_PANEL, self.lib_rect, border_radius=6)
        pygame.draw.rect(screen, C_EDGE, self.lib_rect, 1, border_radius=6)
        t = font.render("Patterns", True, C_TXT)
        screen.blit(t, (lx + (lw - t.get_width()) // 2, ly + 4))
        pygame.draw.line(screen, C_EDGE, (lx + 2, ly + 26), (lx + lw - 2, ly + 26))
        top, bottom = ly + 27, ly + lh
        clip = screen.get_clip()
        screen.set_clip(pygame.Rect(lx + 1, top, lw - 2, lh - 28))
        for cat, y in self.lib_heads:
            y += self.lib_scroll
            if top - LIB_HDR_H <= y <= bottom:
                screen.blit(small.render(cat.upper(), True, C_DIM), (lx + 8, y + 1))
        hot = self.drag_pat['item'] if self.drag_pat is not None else None
        for it in self.lib_items:
            x, y, w, h = it['rect']
            y += self.lib_scroll
            if y + h < top or y > bottom:
                continue
            pygame.draw.rect(screen, C_BTN_ON if it is hot else C_BTN, (x, y, w, h),
                             border_radius=4)
            self._draw_preview(screen, it['cells'], x + 3, y + (h - LIB_PREV_H) // 2)
            screen.blit(small.render(it['name'], True, C_TXT), (x + LIB_PREV_W + 8, y + 5))
        screen.set_clip(clip)
        if self.lib_scroll_min < 0:
            vis = lh - 28
            content = vis - self.lib_scroll_min
            thumb_h = max(18, vis * vis // content)
            thumb_y = top + int(-self.lib_scroll / -self.lib_scroll_min * (vis - thumb_h))
            pygame.draw.rect(screen, C_DIM, (lx + lw - 5, thumb_y, 3, thumb_h), border_radius=2)

    @staticmethod
    def _draw_preview(screen, cells, x, y, pw=LIB_PREV_W, ph=LIB_PREV_H):
        """A pattern thumbnail (the synth's pattern_preview_surf, drawn direct)."""
        import pygame
        pygame.draw.rect(screen, C_BG, (x, y, pw, ph), border_radius=2)
        if not cells:
            return
        r_min, c_min = min(r for r, _c in cells), min(c for _r, c in cells)
        rh = max(r for r, _c in cells) - r_min + 1
        cw = max(c for _r, c in cells) - c_min + 1
        cs = max(1, min(pw // cw, ph // rh))
        ox, oy = x + (pw - cs * cw) // 2, y + (ph - cs * rh) // 2
        for r, c in cells:
            pygame.draw.rect(screen, C_ACCENT, (ox + (c - c_min) * cs, oy + (r - r_min) * cs,
                                                max(1, cs - 1), max(1, cs - 1)))

    # -- drawing -------------------------------------------------------------
    def draw(self, screen, font, small):
        import pygame
        self.tick()
        self._bind_font(small)
        if self.mode == 'catalog':
            screen.fill(C_BG)
            self._draw_catalog(screen, font, small)
            return
        if self.mode == 'report':
            screen.fill(C_BG)
            self._draw_report(screen, font, small)
            return
        snap = self.engine.snapshot()
        screen.fill(C_BG)
        pygame.draw.rect(screen, C_PANEL, (0, 0, self.width, TOP_H))
        screen.blit(font.render(self.scene.title, True, C_TXT), (MARGIN, 10))
        if self.scene.listen:
            screen.blit(small.render(self.scene.listen, True, C_DIM), (MARGIN, 34))
        for key, (rect, label) in self.buttons.items():
            on = (key == 'pause' and snap['paused'])
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = font.render(label, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        vx, vy, vw, vh = self.vol_rect
        screen.blit(small.render("Volume", True, C_DIM), (MARGIN, vy - 4))
        pygame.draw.rect(screen, C_EDGE, (vx, vy, vw, vh), border_radius=3)
        pygame.draw.rect(screen, C_ACCENT, (vx, vy, int(vw * self.vol), vh), border_radius=3)
        screen.blit(small.render(f"{int(self.vol * 100):3d}%", True, C_TXT), (vx + vw + 8, vy - 4))
        gx = vx + vw + 60
        screen.blit(small.render(f"Gen {snap['gen']}   t = {snap['t_seconds']:.2f} s",
                                 True, C_TXT), (gx, vy - 4))
        st = self.engine.status_text()
        col = C_TXT if self.engine.device_ok else C_ERR
        if self.engine.device_ok and self.engine.underruns:
            col = C_WARN
        screen.blit(small.render(f"Audio: {st}", True, col), (MARGIN, TOP_H - 26))
        # field
        fx, fy = self.field_x, self.field_y
        pygame.draw.rect(screen, C_GRID, (fx - 1, fy - 1, self.field_w + 2, self.field_h + 2), 1)
        alive_col = C_ALIVE_PAUSED if snap['paused'] else C_ALIVE
        grid = snap['grid']
        for r in range(self.scene.rows):
            for c in range(self.scene.cols):
                rect = (fx + c * CELL, fy + r * CELL, CELL - 1, CELL - 1)
                pygame.draw.rect(screen, alive_col if grid[r, c] else C_GRID, rect)
        if self.drag_pat is not None and self.drag_pat['snap'] is not None:
            # the dragged pattern's ghost where it would drop (torus wrap)
            if self._ghost is None:
                self._ghost = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
                self._ghost.fill(C_GHOST)
            r0, c0 = self.drag_pat['snap']
            for dr, dc in self.drag_pat['cells']:
                rr, cc = (r0 + dr) % self.scene.rows, (c0 + dc) % self.scene.cols
                screen.blit(self._ghost, (fx + cc * CELL, fy + rr * CELL))
            t = small.render(self.drag_pat['name'], True, C_ACCENT)
            screen.blit(t, (min(fx + c0 * CELL + 14, fx + self.field_w - t.get_width()),
                            max(fy + r0 * CELL - 18, fy)))
        # side panel
        side = snap['selected']
        eid, params = snap['sides'][side]
        disp_side = snap.get('display', {}).get(side)
        if disp_side and 'figures' in disp_side:
            self._draw_figures(screen, small, disp_side, snap['paused'])
        self._draw_overlay(screen, small, eid, params)
        for s, rect in self.tabs.items():
            on = (s == side)
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=4)
            star = '*' if snap['modified'][s] else ''
            hot = '1' if s == 'A' else '2'
            lbl = f"{s}{star}: {registry.label(snap['sides'][s][0])} ({hot})"
            t = font.render(lbl, True, C_TXT)
            if t.get_width() > rect[2] - 6:
                t = small.render(lbl, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        for (src, dst), rect in self.copy_btns.items():
            pygame.draw.rect(screen, C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = font.render('<<' if dst == 'A' else '>>', True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        fr = self.factory_btn
        pygame.draw.rect(screen, C_BTN, fr, border_radius=4)
        pygame.draw.rect(screen, C_EDGE, fr, 1, border_radius=4)
        t = small.render('Factory A+B', True, C_TXT)
        screen.blit(t, (fr[0] + (fr[2] - t.get_width()) // 2, fr[1] + (fr[3] - t.get_height()) // 2))
        px = self.panel_x
        screen.blit(small.render(f"Listening + editing: {side}", True, C_ACCENT),
                    (px, TOP_H + TAB_H + 6))
        screen.blit(small.render(describe_difference(snap['sides']), True, C_TXT),
                    (px, TOP_H + TAB_H + 24))
        ea, eb = snap['sides']['A'][0], snap['sides']['B'][0]
        shared = bool(registry.spectrum_keys(ea, eb))
        for (src, dst), rect in self.spec_btns.items():
            pygame.draw.rect(screen, C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            lbl = '<< spectrum B to A' if dst == 'A' else 'spectrum A to B >>'
            t = small.render(lbl, True, C_TXT if shared else C_DIM)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        for e_id, rect in self.engine_btns.items():
            on = (e_id == eid)
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=3)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=3)
            t = small.render(registry.label(e_id), True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        inactive = self._inactive(eid, params)
        choices = registry.get(eid).choices
        ranges = registry.get(eid).ranges
        range_rects = self._range_rects(eid)
        for spec, (sx, sy, sw, sh) in self._param_rows(eid):
            name, label, lo, hi, integer, _d = spec
            v = params[name]
            ranged = name in ranges
            if ranged:
                lo, hi = self._range_of(snap, side, name)
                rr = range_rects[name]
                screen.blit(small.render("range", True, C_DIM), (px + 12, rr['y']))
                for which in ('min', 'max'):
                    focused = (self.range_edit is not None and self.range_edit['name'] == name
                               and self.range_edit['which'] == which)
                    if focused:
                        self._draw_edit(screen, small, rr[which], self.range_edit['edit'], True)
                    elif name in inactive:
                        screen.blit(small.render(self.range_text(lo if which == 'min' else hi), True, C_DIM),
                                    (rr[which][0] + 6, rr[which][1] + 2))
                    else:
                        self._draw_edit(screen, small, rr[which],
                                        TextEdit(self.range_text(lo if which == 'min' else hi)), False)
                screen.blit(small.render("..", True, C_DIM), (rr['min'][0] + RANGE_FIELD_W + 3, rr['y'] + 2))
                if self.range_error is not None and self.range_error[0] == name:
                    screen.blit(small.render(self.range_error[1][:24], True, C_ERR),
                                (rr['max'][0] + RANGE_FIELD_W + 6, rr['y'] + 2))
            screen.blit(small.render(label, True, C_DIM), (px, sy - 4))
            if name in inactive:
                screen.blit(small.render(inactive[name], True, C_DIM), (sx, sy - 4))
            elif name in choices:
                for value, rect in self._choice_rects(eid, name, sy):
                    on = (value == v)
                    pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=3)
                    pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=3)
                    t = small.render(choices[name][value], True, C_TXT)
                    screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                                    rect[1] + (rect[3] - t.get_height()) // 2))
            elif _is_toggle(spec):
                pygame.draw.rect(screen, C_ACCENT if v else C_BTN, (sx, sy - 3, 34, 16),
                                 border_radius=8)
                pygame.draw.rect(screen, C_EDGE, (sx, sy - 3, 34, 16), 1, border_radius=8)
                screen.blit(small.render("on" if v else "off", True, C_TXT), (sx + 40, sy - 4))
            else:
                frac = (v - lo) / (hi - lo) if hi > lo else 0.0
                frac = min(max(frac, 0.0), 1.0)         # a value outside the user range sits at the end
                pygame.draw.rect(screen, C_EDGE, (sx, sy, sw, sh), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (sx, sy, int(sw * frac), sh), border_radius=3)
                txt = f"{v:d}" if integer else (f"{v:.{RANGE_DECIMALS}f}" if ranged else f"{v:.2f}")
                screen.blit(small.render(txt, True, C_TXT), (sx + sw + 8, sy - 4))
        pk = snap['peak']
        screen.blit(small.render(f"peak A {pk['A']:.2f}   B {pk['B']:.2f}", True, C_DIM),
                    (px, self.footer_y))
        if self.message:
            screen.blit(small.render(self.message[:60], True, C_ERR),
                        (px, self.footer_y + 20))
        self._draw_display(screen, small, snap.get('display', {}).get(side),
                           px, self.params_y + self._params_height(eid) + 6)
        self._draw_library(screen, font, small)
        # S4: lab buttons + status; overlays
        if self.catalog is not None:
            for key, rect in self.lab_buttons.items():
                pygame.draw.rect(screen, C_BTN, rect, border_radius=4)
                pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
                t = font.render({'save': 'Save', 'catalog': 'Catalog', 'notes': 'Notes'}[key],
                                True, C_TXT)
                screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                                rect[1] + (rect[3] - t.get_height()) // 2))
            if self.status:
                col = C_ERR if self.status.lower().startswith(('save failed', 'nothing',
                                                                 'no audio')) else C_OK
                screen.blit(small.render(self.status[:90], True, col), (MARGIN + 300, TOP_H - 26))
        if self.mode == 'save':
            self._draw_save_form(screen, font, small)

    # =====================================================================
    # Listening notes (2026-09-14): free text attached to the session's record
    # =====================================================================
    def notes_target(self):
        """(rid, title) of the record the notes belong to, or (None, reason)."""
        if self.catalog is None:
            return None, "notes need a catalog"
        if self.mode == 'catalog':
            rec = self.selected_record()
            return (rec.id, rec.title) if rec is not None else (None, "select a record first")
        if self.session_record is None:
            return None, "no record in this session yet: Continue a record or Save first"
        try:
            rec = self.catalog.load(self.session_record)
        except CatalogError as e:
            return None, f"record unavailable: {e}"
        return rec.id, rec.title

    def open_notes(self):
        """The notes of the record the session belongs to, in a window of their own
        (the same form as before: casynth_lab.notes_window).  A second press for
        the same record brings that window to the front."""
        rid, why = self.notes_target()
        if rid is None:
            self.status = f"Notes: {why}"
            return False
        if self.notes_window is not None and self.notes_window.alive and self.notes_window.rid == rid:
            self.notes_window.focus()
            return True
        self.close_notes()
        rec = self.catalog.load(rid)
        text = rec.notes
        if text.endswith('\n'):
            text = text[:-1]                  # the file's final newline is added on write
        self.notes_form = dict(rid=rid, title=rec.title, text=text)
        self.notes_edit = TextEdit(text, multiline=True)
        self.notes_window = _notes_win.NotesWindow(rid, rec.title, position=self._notes_window_position())
        self.status = f"Notes: {rec.title[:40]}"
        return True

    def _notes_window_position(self):
        """Right of the bench window when its position is known."""
        try:
            import pygame
            x, y = pygame.display.get_window_position()
            return (x + self.width + 12, y)
        except Exception:                          # noqa: BLE001
            return None

    def close_notes(self):
        """Close the notes window (the text is already written on every edit)."""
        if self.notes_window is not None:
            self.notes_window.close()
        self.notes_window = None
        if self.drag_text is not None and self.notes_edit is not None and self.drag_text[0] is self.notes_edit:
            self.drag_text = None
        self.notes_form = None
        self.notes_edit = None
        if self.mode == 'catalog':
            self.refresh_catalog()

    def sync_notes(self):
        """Once per frame: the window belongs to one record -- closed when the notes
        would go to another record (another record selected, Back to the catalog,
        Continue / Save of another one) or when the OS closed it."""
        w = self.notes_window
        if w is None:
            return
        if not w.alive:
            self.close_notes()
            return
        rid, _why = self.notes_target()
        if rid != w.rid:
            self.close_notes()

    def _notes_edit(self, text):
        """Every edit is written at once (small file, atomic replace): nothing
        typed while listening is lost."""
        f = self.notes_form
        f['text'] = text
        try:
            self.catalog.write_notes(f['rid'], text)
            self.status = f"Notes saved: {f['title'][:40]}"
        except CatalogError as e:
            self.status = f"Notes NOT saved: {e}"

    def _notes_rects(self):
        """The form's rectangles in the notes window's own coordinates."""
        w, h = self.notes_window.size if self.notes_window is not None else _notes_win.DEFAULT_SIZE
        return dict(box=(0, 0, w, h), close=(w - 110, 8, 100, 24), text=(12, 40, w - 24, h - 78))

    # -- events of the notes window (the main loop routes them by window) --------------
    def notes_owns(self, ev):
        return self.notes_window is not None and self.notes_window.owns(ev)

    def notes_event(self, ev):
        """One pygame event that belongs to the notes window."""
        import pygame
        if ev.type == pygame.WINDOWCLOSE:
            self.close_notes()
            return 'notes:close'
        if ev.type == pygame.KEYDOWN:
            return self.notes_key(pygame.key.name(ev.key), ctrl=bool(ev.mod & pygame.KMOD_CTRL),
                                  shift=bool(ev.mod & pygame.KMOD_SHIFT))
        if ev.type == pygame.TEXTINPUT:
            return self.notes_text(ev.text)
        if ev.type == pygame.MOUSEBUTTONDOWN:
            return self.notes_press(ev.pos, ev.button, shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
        if ev.type == pygame.MOUSEWHEEL:
            return self.notes_wheel(pygame.mouse.get_pos(), ev.y)
        if ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
            return self.notes_drag(ev.pos)
        if ev.type == pygame.MOUSEBUTTONUP:
            self.notes_release()
            return 'notes:release'
        return None

    def notes_key(self, name, ctrl=False, shift=False):
        """Keys of the notes window: the editing keys of every text field, then
        Esc (close) and Enter (new line)."""
        if self.notes_form is None:
            return None
        name = name.lower()
        got = self._edit_key(self.notes_edit, self._notes_rects()['text'], name, ctrl, shift)
        if got is not None:
            return f"notes:{got}"
        if name == 'escape':
            self.close_notes()
            return 'notes:close'
        if name in ('return', 'enter', 'kp_enter'):
            self.notes_edit.insert('\n')
            self._after_edit()
            return 'notes:edit'
        return None

    def notes_text(self, text):
        if self.notes_form is None:
            return None
        self.notes_edit.insert(text)
        self._after_edit()
        return 'notes:edit'

    def notes_press(self, pos, button, shift=False):
        if self.notes_form is None:
            return None
        r = self._notes_rects()
        if self._inside(r['close'], pos) and button == 1:
            self.close_notes()
            return 'notes:close'
        if self._inside(r['text'], pos) and button == 1:
            self._caret_from_pos(self.notes_edit, r['text'], pos, shift, start_drag=True)
            return 'notes:caret'
        return None

    def notes_wheel(self, pos, dy):
        """Wheel over the text (dy > 0 = up): scrolls, the caret stays."""
        if self.notes_form is None:
            return None
        r = self._notes_rects()['text']
        if self._inside(r, pos):
            self.notes_edit.scroll = max(0, self.notes_edit.scroll - int(dy) * 3)
            self.notes_edit.follow = False
            return 'notes:scroll'
        return None

    def notes_drag(self, pos):
        if self.drag_text is not None and self.notes_edit is not None and self.drag_text[0] is self.notes_edit:
            self._caret_from_pos(self.notes_edit, self.drag_text[1], pos, shift=True)
            return 'notes:caret'
        return None

    def notes_release(self):
        if self.drag_text is not None and self.notes_edit is not None and self.drag_text[0] is self.notes_edit:
            self.drag_text = None

    def draw_notes(self, font, small):
        """The form -- drawn exactly as before -- into the notes window (every frame)."""
        import pygame
        w = self.notes_window
        if w is None or not w.alive or self.notes_form is None:
            return
        screen = w.surface
        r = self._notes_rects()
        f = self.notes_form
        pygame.draw.rect(screen, C_PANEL, r['box'])
        pygame.draw.rect(screen, C_ACCENT, r['box'], 1)
        screen.blit(font.render(f"Notes: {f['title']}"[:42], True, C_TXT), (12, 10))
        cr = r['close']
        pygame.draw.rect(screen, C_BTN, cr, border_radius=4)
        pygame.draw.rect(screen, C_EDGE, cr, 1, border_radius=4)
        t = small.render('Close (Esc)', True, C_TXT)
        screen.blit(t, (cr[0] + (cr[2] - t.get_width()) // 2, cr[1] + (cr[3] - t.get_height()) // 2))
        self._draw_edit(screen, small, r['text'], self.notes_edit, True)
        screen.blit(small.render("Saved as you type.  Enter = new line.  Ctrl+A/X/C/V.", True, C_DIM),
                    (12, r['box'][3] - 26))
        w.flip()

    # =====================================================================
    # S/N demos (2026-09-14): field overlay of the listened side + engine display
    # =====================================================================
    def _cell_px(self, x, y):
        """Unwrapped cell coordinates (x = column, y = row) -> pixel centre."""
        return (self.field_x + (x + 0.5) * CELL, self.field_y + (y + 0.5) * CELL)

    def _seam_pieces(self, a, b):
        """The unwrapped segment a->b as pieces INSIDE the field: a torus seam
        crossing is split at the seam and each piece is shifted into the field
        (never a fake line across the whole field)."""
        rows, cols = self.scene.rows, self.scene.cols
        (ax, ay), (bx, by) = a, b
        ts = {0.0, 1.0}
        for pa, pb, n in ((ax, bx, cols), (ay, by, rows)):
            if pb != pa:
                lo, hi = min(pa, pb), max(pa, pb)
                for k in range(math.ceil((lo + 0.5) / n), math.floor((hi + 0.5) / n) + 1):
                    t = (k * n - 0.5 - pa) / (pb - pa)
                    if 0.0 < t < 1.0:
                        ts.add(t)
        ts = sorted(ts)
        out = []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            tm = (t0 + t1) / 2.0
            sx = math.floor((ax + (bx - ax) * tm + 0.5) / cols) * cols
            sy = math.floor((ay + (by - ay) * tm + 0.5) / rows) * rows
            out.append(((ax + (bx - ax) * t0 - sx, ay + (by - ay) * t0 - sy),
                        (ax + (bx - ax) * t1 - sx, ay + (by - ay) * t1 - sy)))
        return out

    def _overlay_runs(self, eid, params):
        """Pixel polylines of the engine overlay (cached per settings)."""
        spec = registry.get(eid)
        if spec.overlay is None:
            return None
        key = (eid, tuple(sorted(params.items())), self.scene.rows, self.scene.cols)
        cache = getattr(self, '_overlay_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        ov = spec.overlay(params, self.scene.rows, self.scene.cols)
        runs = []
        pts = ov.get('polyline') or []
        pairs = list(zip(pts[:-1], pts[1:]))
        run = []
        for a, b in pairs:
            for p0, p1 in self._seam_pieces(a, b):
                if run and run[-1] == p0:
                    run.append(p1)
                else:
                    if len(run) >= 2:
                        runs.append(run)
                    run = [p0, p1]
        if len(run) >= 2:
            runs.append(run)
        data = dict(runs=[[self._cell_px(*p) for p in r] for r in runs],
                    start=ov.get('start'), ahead=ov.get('ahead'), text=ov.get('text', ''),
                    circles=ov.get('circles') or [], labels=ov.get('labels') or [],
                    lines=[(self._cell_px(*a), self._cell_px(*b)) for a, b in (ov.get('lines') or [])])
        self._overlay_cache = (key, data)
        return data

    def _draw_overlay(self, screen, small, eid, params):
        import pygame
        data = self._overlay_runs(eid, params)
        if data is None:
            return
        col = C_OVERLAY
        for a, b in data['lines']:                 # straight borders (no torus wrap)
            pygame.draw.line(screen, col, a, b, 1)
        for x, y, radius in data['circles']:
            cx, cy = self._cell_px(x, y)
            pygame.draw.circle(screen, col, (int(cx), int(cy)), int(radius * CELL), 1)
        for x, y, text in data['labels']:
            cx, cy = self._cell_px(x, y)
            t = small.render(text, True, col)
            box = (cx - t.get_width() // 2 - 3, cy - t.get_height() // 2 - 1,
                   t.get_width() + 6, t.get_height() + 2)
            pygame.draw.rect(screen, C_BG, box, border_radius=3)
            pygame.draw.rect(screen, col, box, 1, border_radius=3)
            screen.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))
        for run in data['runs']:
            pygame.draw.lines(screen, col, False, run, 2)
        if data['start'] is not None:
            sx, sy = self._cell_px(*data['start'])
            pygame.draw.circle(screen, col, (int(sx), int(sy)), 5)
            if data['ahead'] is not None:
                hx, hy = self._cell_px(*data['ahead'])
                dx, dy = hx - sx, hy - sy
                norm = math.hypot(dx, dy) or 1.0
                ux, uy = dx / norm, dy / norm
                tip = (sx + ux * 16, sy + uy * 16)
                left = (sx + ux * 6 - uy * 6, sy + uy * 6 + ux * 6)
                right = (sx + ux * 6 + uy * 6, sy + uy * 6 - ux * 6)
                pygame.draw.polygon(screen, col, [tip, left, right])
        if data['text']:
            screen.blit(small.render(data['text'][:90], True, col),
                        (self.field_x, self.field_y + self.field_h + 2))

    def _draw_figures(self, screen, small, disp, paused):
        """N4 ca_object_resonators: the tracked figures of the LISTENED side from the
        engine's display (the same geometry the audio uses): the live cells of each
        figure in its colour, its detector circle (continued across the torus seam;
        Own mode: the cells are the mask -- outlined -- and the circle is a thin
        dotted reference) and its centre.  Never an analysis of its own."""
        import pygame
        fx, fy = self.field_x, self.field_y
        rows, cols = self.scene.rows, self.scene.cols
        own = int(disp.get('detector', 1)) == 0
        clip = screen.get_clip()
        screen.set_clip(pygame.Rect(fx, fy, self.field_w, self.field_h))
        n_cover = 0
        for f in disp['figures']:
            col = C_FIGURES[int(f['color']) % len(C_FIGURES)]
            fill = col if not paused else tuple((a + b) // 2 for a, b in zip(col, C_ALIVE_PAUSED))
            for r, c in f['cells']:
                rect = (fx + c * CELL, fy + r * CELL, CELL - 1, CELL - 1)
                pygame.draw.rect(screen, fill, rect)
                if own:
                    pygame.draw.rect(screen, C_BG, rect, 1)
            cy, cx = float(f['centre'][0]), float(f['centre'][1])
            rad = float(f['radius']) * CELL
            px, py = self._cell_px(cx, cy)
            if f.get('covers_all'):
                # the disk is the whole field (its outline is beyond the screen):
                # a frame in the figure's colour just inside the field edge, one
                # frame per such figure (nested), and its centre as usual
                k = n_cover
                n_cover += 1
                pygame.draw.rect(screen, col, (fx + 2 * k, fy + 2 * k,
                                               self.field_w - 4 * k, self.field_h - 4 * k), 1)
                pygame.draw.circle(screen, C_BG, (int(px), int(py)), 4)
                pygame.draw.circle(screen, col, (int(px), int(py)), 3)
                continue
            for dy in (-rows * CELL, 0, rows * CELL):
                for dx in (-cols * CELL, 0, cols * CELL):
                    ox, oy = px + dx, py + dy
                    if (ox + rad < fx or ox - rad > fx + self.field_w
                            or oy + rad < fy or oy - rad > fy + self.field_h):
                        continue
                    if rad >= 1.0:
                        if own:
                            n = max(12, int(rad / 3))
                            for k in range(n):
                                a = 2.0 * math.pi * k / n
                                screen.set_at((int(ox + rad * math.cos(a)), int(oy + rad * math.sin(a))), col)
                        else:
                            pygame.draw.circle(screen, col, (int(ox), int(oy)), int(round(rad)), 1)
                    pygame.draw.circle(screen, C_BG, (int(ox), int(oy)), 4)
                    pygame.draw.circle(screen, col, (int(ox), int(oy)), 3)
        screen.set_clip(clip)
        cover = f"   {n_cover} circle{'s' if n_cover != 1 else ''} cover the whole field" if n_cover else ""
        head = f"sounding {disp.get('n_sounding', 0)} of {disp.get('n_figures', 0)} figures   tails {disp.get('n_tails', 0)}"
        modes = f"   detector {disp.get('detector_name', '?')}   spectrum {disp.get('spectrum_name', '?')}"
        t = small.render(head + modes + cover, True, C_DIM)
        if t.get_width() > self.field_w - 4:                    # keep it inside the field's width
            t = small.render(head + cover, True, C_DIM)
        screen.blit(t, (fx + self.field_w - t.get_width(), fy - 18))

    def _draw_display(self, screen, small, disp, x, y):
        """Engine display numbers (pm_network: six link depths W as bars,
        the target as a tick, frozen / gate state; gutter_field: the held
        control vector u per node as bars, the field's own u as a tick, the
        frequency ratio, links / resets; ca_event_network: the last event
        packet per node as bars, the response level as a tick, rho;
        ca_tuned_events: the packet as bars, the bank level as a tick, the
        frequency multiplier, tuning mode and decay)."""
        import pygame
        if not disp:
            return
        if 'figures' in disp:
            # N4 ca_object_resonators: one row per sounding figure (colour, id, cells,
            # modes, lowest frequency), the last packet a (bar), the bank level (tick, /5)
            import pygame as _pg
            ramp = "  (ramping)" if int(disp.get('ramp_left', 0)) > 0 else ""
            attack = ""
            if 'attack_ms' in disp:
                attack = (f"   attack {float(disp['attack_ms']):.1f} ms"
                          + (" (ramping)" if int(disp.get('attack_ramp_left', 0)) > 0 else ""))
            extra = ""
            if disp.get('drops') or disp.get('evictions') or disp.get('inplace_fades'):
                extra = (f"   faded {disp.get('evictions', 0)} in place {disp.get('inplace_fades', 0)}"
                         f" dropped {disp.get('drops', 0)}")
            screen.blit(small.render(f"Figures: sounding {disp['n_sounding']} of {disp['n_figures']}"
                                     f"   tails {disp['n_tails']}{attack}{extra}", True, C_DIM), (x, y))
            if int(disp.get('spectrum', 0)) == 1:
                lap = disp.get('laplace', {})
                law = (f"Laplace f0 {float(disp.get('f0', 0.0)):.0f} Hz  n {int(lap.get('n', 0))}"
                       f"  {'full' if int(lap.get('fullshape', 1)) else '8x8'}")
            else:
                law = f"scale {float(disp['frequency_scale']):.0f} Hz"
            screen.blit(small.render(f"R x{float(disp.get('radius_mul', 1.0)):.3f}   {law}   decay "
                                     f"{float(disp['decay_s']):.2f} s{ramp}   a | level", True, C_DIM),
                        (x, y + 14))
            bar_x, bar_w = x + 98, 76
            shown = [f for f in disp['figures'] if f['slot'] >= 0][:8]
            for k, f in enumerate(shown):
                ry = y + 34 + k * DISPLAY_ROW_H
                col = C_FIGURES[int(f['color']) % len(C_FIGURES)]
                _pg.draw.rect(screen, col, (x, ry, 8, 8))
                screen.blit(small.render(f"#{f['id']} {f['n']}c {f['modes']}m", True, C_DIM), (x + 12, ry - 2))
                a, lv = float(f['a']), float(f['level'])
                _pg.draw.rect(screen, C_EDGE, (bar_x, ry, bar_w, 8), border_radius=3)
                _pg.draw.rect(screen, C_ACCENT, (bar_x, ry, int(bar_w * min(a, 1.0)), 8), border_radius=3)
                tx = bar_x + int(bar_w * min(lv / 5.0, 1.0))
                _pg.draw.line(screen, C_TXT, (tx, ry - 2), (tx, ry + 10), 1)
                screen.blit(small.render(f"{float(f['f_low']):.0f} Hz  a {a:.2f}", True, C_TXT),
                            (bar_x + bar_w + 6, ry - 3))
            return
        if 'u' in disp:
            state = "frozen" if disp.get('frozen') else "live"
            screen.blit(small.render(f"Nodes u (held)  {state}   links {disp.get('links', 0):.2f}"
                                     f"   resets {disp.get('resets', 0)}", True, C_DIM), (x, y))
            bar_x, bar_w = x + 44, 150
            for k in range(len(disp['u'])):
                ry = y + 18 + k * DISPLAY_ROW_H
                u, uf = float(disp['u'][k]), float(disp['u_field'][k])
                screen.blit(small.render(f"{k} ({disp['counts'][k]})", True, C_DIM), (x, ry - 2))
                pygame.draw.rect(screen, C_EDGE, (bar_x, ry, bar_w, 8), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (bar_x, ry, int(bar_w * min(u, 1.0)), 8),
                                 border_radius=3)
                tx = bar_x + int(bar_w * min(uf, 1.0))
                pygame.draw.line(screen, C_TXT, (tx, ry - 2), (tx, ry + 10), 1)
                screen.blit(small.render(f"x{disp['ratio'][k]:.3f}", True, C_TXT),
                            (bar_x + bar_w + 8, ry - 3))
            return
        if 'tuning' in disp:
            # N3 ca_tuned_events: the last packet a_i per node (bar), the bank level
            # max |b_i| of the last block (tick, /5), the node's frequency multiplier,
            # the tuning mode and the decay (with its 20 ms ramp)
            mode = "field" if disp.get('tuning') else "fixed"
            ramp = "  (ramping)" if int(disp.get('ramp_left', 0)) > 0 else ""
            screen.blit(small.render(f"Nodes: events a | bank |b|   tuning {mode}   decay "
                                     f"{float(disp.get('decay_s', 0.0)):.2f} s{ramp}", True, C_DIM), (x, y))
            bar_x, bar_w = x + 44, 150
            for k in range(len(disp['events'])):
                ry = y + 18 + k * DISPLAY_ROW_H
                a, lv = float(disp['events'][k]), float(disp['level'][k])
                screen.blit(small.render(f"{k} ({disp['counts'][k]})", True, C_DIM), (x, ry - 2))
                pygame.draw.rect(screen, C_EDGE, (bar_x, ry, bar_w, 8), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (bar_x, ry, int(bar_w * min(a, 1.0)), 8),
                                 border_radius=3)
                tx = bar_x + int(bar_w * min(lv / 5.0, 1.0))
                pygame.draw.line(screen, C_TXT, (tx, ry - 2), (tx, ry + 10), 1)
                screen.blit(small.render(f"x{float(disp['ratio'][k]):.3f}  a {a:.2f}", True, C_TXT),
                            (bar_x + bar_w + 8, ry - 3))
            return
        if 'events' in disp:
            # N2 ca_event_network: the last packet a_i per node (bar), the response
            # level max |l_i| of the last block (tick, x4), rho and its ramp
            rho, tgt = float(disp.get('rho', 0.0)), float(disp.get('rho_target', 0.0))
            arrow = f" -> {tgt:.2f}" if int(disp.get('ramp_left', 0)) > 0 else ""
            screen.blit(small.render(f"Nodes: events a | response |l|   rho {rho:.2f}{arrow}",
                                     True, C_DIM), (x, y))
            bar_x, bar_w = x + 44, 150
            for k in range(len(disp['events'])):
                ry = y + 18 + k * DISPLAY_ROW_H
                a, lv = float(disp['events'][k]), float(disp['level'][k])
                screen.blit(small.render(f"{k}", True, C_DIM), (x, ry - 2))
                pygame.draw.rect(screen, C_EDGE, (bar_x, ry, bar_w, 8), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (bar_x, ry, int(bar_w * min(a, 1.0)), 8),
                                 border_radius=3)
                tx = bar_x + int(bar_w * min(lv * 4.0, 1.0))
                pygame.draw.line(screen, C_TXT, (tx, ry - 2), (tx, ry + 10), 1)
                screen.blit(small.render(f"a {a:.2f}  |l| {lv:.3f}", True, C_TXT),
                            (bar_x + bar_w + 8, ry - 3))
            return
        if 'W' not in disp:
            return
        from casynth_lab.pm_network import EDGES
        state = "frozen" if disp.get('frozen') else "live"
        screen.blit(small.render(f"Links W (j>i)  {state}   gate {disp.get('gate', 0):.2f}",
                                 True, C_DIM), (x, y))
        bar_x, bar_w, full = x + 44, 150, 1.0 / 3.0
        for k, (i, j) in enumerate(EDGES):
            ry = y + 18 + k * DISPLAY_ROW_H
            w, wt = float(disp['W'][k]), float(disp['W_target'][k])
            screen.blit(small.render(f"{j}>{i}", True, C_DIM), (x, ry - 2))
            pygame.draw.rect(screen, C_EDGE, (bar_x, ry, bar_w, 8), border_radius=3)
            pygame.draw.rect(screen, C_ACCENT, (bar_x, ry, int(bar_w * min(w / full, 1.0)), 8),
                             border_radius=3)
            tx = bar_x + int(bar_w * min(wt / full, 1.0))
            pygame.draw.line(screen, C_TXT, (tx, ry - 2), (tx, ry + 10), 1)
            screen.blit(small.render(f"{w:.3f}", True, C_TXT), (bar_x + bar_w + 8, ry - 3))

    # =====================================================================
    # S4: save / catalog / player / replay
    # =====================================================================
    def tick(self):
        """Per-frame housekeeping (also callable from tests): pick up a cut,
        finish background save / replay threads."""
        if self._cut_pending:
            got = self.engine.take_cut()
            if got is not None:
                self._cut_pending = False
                kind, cut = got
                if kind == 'none' or cut is None:
                    self.status = "Nothing to save yet: release Pause CA first"
                else:
                    self._open_save_form(cut)
        child = getattr(self, 'child', None)
        if child is not None and not child.alive:
            child.wait(1.0)
            self.child = None
            if child.error:
                self.status = f"Version bench failed: {child.error}"
            else:
                got = child.provenance or {}
                self.status = f"Version bench closed ({prov.short(got.get('commit')) or '?'})"
            self.refresh_catalog()
        vth = self.verify['thread']
        if vth is not None and not vth.is_alive():
            self.verify['thread'] = None
            err, run = self.verify['error'], self.verify['done']
            if err is not None:
                self.status = f"Check failed: {err}"
            elif run is not None:
                self.status = f"Check {run.status_label}: {run.summary()}"
                if self.mode == 'catalog':
                    self.open_report(run)
        th = self.cat['thread']
        if th is not None and not th.is_alive():
            self.cat['thread'] = None
            pending_open = self.cat.pop('open', None)
            if pending_open is not None:
                rec, state = pending_open
                if isinstance(state, CatalogError):
                    self.status = f"Cannot open: {state}"
                else:
                    self._finish_open(rec, state)
                return
            res = self.cat['result']
            if res is not None:
                self.status = res.text
                self.cat['play_label'] = ("Play replay" if res.status in ('match', 'mismatch')
                                          else "")

    # -- save ----------------------------------------------------------------------
    def begin_save(self):
        """Save button: ask the render thread for a consistent cut NOW; the
        title/note form opens when it arrives (the cut is already fixed)."""
        self.status = ""
        self._cut_pending = True
        self.engine.request_cut()

    def confirm_save(self):
        form, self.save_form = self.save_form, None
        self.mode = 'live'
        if form is None:
            return
        self.status = "Saving..."

        def work():
            try:
                rid = self.catalog.save(form['cut'], form['title'], form['note'])
                self.session_record = rid          # Notes now belong to the saved record
                self.status = f"Saved: {form['title'] or rid}  [{form['cut'].seconds:.1f} s]"
            except CatalogError as e:
                self.status = f"Save failed: {e}"
        self._save_thread = threading.Thread(target=work, daemon=True)
        self._save_thread.start()

    def cancel_save(self):
        self.save_form = None
        self.mode = 'live'
        self.status = "Save cancelled"

    def _open_save_form(self, cut):
        title = Catalog.default_title(self.scene.title)
        self.save_form = dict(cut=cut, field='title', note='', title=title)
        self.save_edits = {'title': TextEdit(title), 'note': TextEdit('')}
        self.mode = 'save'

    # -- text fields: one behaviour for every input --------------------------------
    def _active_edit(self):
        """(TextEdit, rect) of the focused text field in this mode, else (None, None)."""
        if self.mode == 'live' and self.range_edit is not None:
            return self.range_edit['edit'], self.range_edit['rect']
        if self.mode == 'save' and self.save_form is not None:
            field = self.save_form['field']
            edit = self.save_edits[field]
            if edit.text != self.save_form[field]:        # written from outside (scripts)
                edit.set_text(self.save_form[field])
            return edit, self._save_form_rects()[field]
        return None, None

    def _after_edit(self):
        """The model changed: mirror it into the form (Notes: into the file)."""
        if self.mode == 'save' and self.save_form is not None:
            for k, e in self.save_edits.items():
                self.save_form[k] = e.text
        if self.notes_form is not None and self.notes_edit is not None \
                and self.notes_edit.text != self.notes_form['text']:
            self._notes_edit(self.notes_edit.text)

    def text_input(self, text):
        edit, _rect = self._active_edit()
        if edit is not None:
            edit.insert(text)
            self._after_edit()

    def _edit_key(self, edit, rect, name, ctrl=False, shift=False):
        """Editing keys shared by every text field: caret, selection, clipboard.
        Returns 'edit' (text changed), 'caret' (caret / selection moved) or
        None (not an editing key -- the form decides)."""
        spans = edit.layout(self.measure, rect[2] - 12)
        before = edit.text
        if name == 'backspace':
            edit.backspace(ctrl)
        elif name == 'delete':
            edit.delete(ctrl)
        elif name in ('left', 'right'):
            edit.move(name, shift, ctrl)
        elif name in ('home', 'end'):
            edit.move(name, shift, ctrl, spans)
        elif name in ('up', 'down'):
            edit.move_lines(-1 if name == 'up' else 1, shift, spans, self.measure)
        elif ctrl and name == 'a':
            edit.select_all()
        elif ctrl and name == 'c':
            self._clip_put(edit.copy())
        elif ctrl and name == 'x':
            self._clip_put(edit.cut())
        elif ctrl and name == 'v':
            edit.insert(self._clip_get())
        else:
            return None
        if edit.text != before:
            self._after_edit()
            return 'edit'
        return 'caret'

    def _caret_from_pos(self, edit, rect, pos, shift=False, start_drag=False):
        """Put the caret under the mouse (Shift / drag: extend the selection)."""
        spans = edit.layout(self.measure, rect[2] - 12)
        if edit.multiline:
            k = edit.scroll + (pos[1] - rect[1] - 4) // self.line_h
            x = pos[0] - rect[0] - 6
        else:
            k, x = 0, pos[0] - rect[0] - 6 + edit.scroll
        edit.set_cursor(edit.index_at(spans, self.measure, x, k), shift)
        if start_drag:
            self.drag_text = (edit, rect)

    def wheel(self, pos, dy):
        """Mouse wheel (dy > 0 = up): scrolls the Notes text (the caret
        stays) or the pattern library."""
        if self.mode in ('live', 'save') and self._inside(self.lib_rect, pos):
            self.lib_scroll = max(self.lib_scroll_min, min(0, self.lib_scroll + int(dy) * 24))
            return 'lib:scroll'
        return None

    def _clip_get(self):
        if self.clip_get is not None:
            try:
                s = self.clip_get()
                if s:
                    return s
            except Exception:                  # noqa: BLE001 -- no system clipboard
                pass
        return self.clipboard

    def _clip_put(self, s):
        if not s:
            return
        self.clipboard = s
        if self.clip_put is not None:
            try:
                self.clip_put(s)
            except Exception:                  # noqa: BLE001
                pass

    def _bind_font(self, small):
        """Text metrics of the font the fields are drawn with (click -> caret)."""
        if self._font_bound is not small:
            self._font_bound = small
            self.measure = lambda s: small.size(s)[0]
            self.line_h = small.get_height() + 2

    def _draw_edit(self, screen, small, rect, edit, focused):
        """A text field: selection, text, caret; the view follows the caret
        after an edit (multiline: by lines, single-line: by pixels)."""
        import pygame
        x, y, w, h = rect
        pygame.draw.rect(screen, C_BG, rect, border_radius=3)
        pygame.draw.rect(screen, C_ACCENT if focused else C_EDGE, rect, 1, border_radius=3)
        measure, text = self.measure, edit.text
        spans = edit.layout(measure, w - 12)
        line_h = self.line_h
        k, cx = edit.caret_pos(spans, measure)
        if edit.multiline:
            n = max(1, (h - 8) // line_h)
            if edit.follow:
                if k < edit.scroll:
                    edit.scroll = k
                elif k >= edit.scroll + n:
                    edit.scroll = k - n + 1
            edit.scroll = max(0, min(edit.scroll, max(0, len(spans) - n)))
            first, ox = edit.scroll, 0
        else:
            n, first, avail = 1, 0, w - 12
            if edit.follow:
                if cx - edit.scroll > avail:
                    edit.scroll = cx - avail
                if cx < edit.scroll:
                    edit.scroll = cx
            ox = edit.scroll
        edit.follow = False
        sel = edit.selection
        clip = screen.get_clip()
        screen.set_clip(pygame.Rect(x + 1, y + 1, w - 2, h - 2))
        for row, (s, e) in enumerate(spans[first:first + n]):
            ly = y + 4 + row * line_h
            if sel is not None:
                a, b = max(sel[0], s), min(sel[1], e)
                if a < b or (s == e and sel[0] < s < sel[1]):    # (empty line inside)
                    x0 = x + 6 - ox + measure(text[s:a])
                    x1 = x + 6 - ox + measure(text[s:b])
                    pygame.draw.rect(screen, C_SEL, (x0, ly, max(x1 - x0, 3), line_h))
            if e > s:
                screen.blit(small.render(text[s:e], True, C_TXT), (x + 6 - ox, ly))
            if focused and first + row == k:
                px = x + 6 - ox + cx
                pygame.draw.line(screen, C_TXT, (px, ly), (px, ly + line_h - 3), 1)
        screen.set_clip(clip)

    def _key_save_form(self, name, ctrl=False):
        f = self.save_form
        if name in ('return', 'enter', 'kp_enter'):
            self.confirm_save()
            return 'save:ok'
        if name == 'escape':
            self.cancel_save()
            return 'save:cancel'
        if name == 'tab':
            f['field'] = 'note' if f['field'] == 'title' else 'title'
            return 'save:field'
        return None

    def _save_form_rects(self):
        x, y, w = MARGIN + 40, TOP_H + 60, 560
        return dict(title=(x + 70, y + 40, w - 90, 26), note=(x + 70, y + 80, w - 90, 26),
                    ok=(x + 20, y + 130, 120, 30), cancel=(x + 160, y + 130, 120, 30),
                    box=(x, y, w, 180))

    def _press_save_form(self, pos, button, shift=False):
        if button != 1:
            return None
        r = self._save_form_rects()
        if self._inside(r['ok'], pos):
            self.confirm_save()
            return 'save:ok'
        if self._inside(r['cancel'], pos):
            self.cancel_save()
            return 'save:cancel'
        for field in ('title', 'note'):
            if self._inside(r[field], pos):
                self.save_form['field'] = field
                self._caret_from_pos(self.save_edits[field], r[field], pos, shift, start_drag=True)
                return f'save:{field}'
        return None

    def _draw_save_form(self, screen, font, small):
        import pygame
        r = self._save_form_rects()
        f = self.save_form
        pygame.draw.rect(screen, C_PANEL, r['box'], border_radius=6)
        pygame.draw.rect(screen, C_ACCENT, r['box'], 1, border_radius=6)
        bx, by = r['box'][0], r['box'][1]
        screen.blit(font.render(f"Save experiment  (last {f['cut'].seconds:.1f} s since "
                                f"Start/Restart, window {f['cut'].window_seconds:.0f} s)",
                                True, C_TXT), (bx + 20, by + 10))
        for field, label in (('title', 'Title'), ('note', 'Description')):
            rect = r[field]
            screen.blit(small.render(label, True, C_DIM), (bx + 20, rect[1] + 5))
            self._draw_edit(screen, small, rect, self.save_edits[field], f['field'] == field)
        for key, label in (('ok', 'Save (Enter)'), ('cancel', 'Cancel (Esc)')):
            rect = r[key]
            pygame.draw.rect(screen, C_BTN_ON if key == 'ok' else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = small.render(label, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        screen.blit(small.render("Tab = next field.  Ctrl+A/X/C/V, Shift+arrows.  "
                                 "The recording end is already fixed.",
                                 True, C_DIM), (bx + 20, by + 165 - 8))

    # -- catalog ------------------------------------------------------------------
    def open_catalog(self):
        """Catalog screen: the CA is paused and the live output muted -- only
        records may sound here.  Both are restored by close_catalog()."""
        snap = self.engine.snapshot()
        self._was_paused = snap['paused']
        if snap['running'] and not snap['paused']:
            self._post('pause', on=True)
        self.engine.muted = True
        self.close_notes()                 # the field's experiment is left: its notes window goes with it
        self.mode = 'catalog'
        self.refresh_catalog()

    def refresh_catalog(self):
        self.cat['entries'] = self.catalog.list()
        if self.cat['sel'] is not None and self.cat['sel'] >= len(self.cat['entries']):
            self.cat['sel'] = None

    def close_catalog(self):
        self.engine.stop_play()
        self.engine.muted = False
        snap = self.engine.snapshot()
        if snap['running'] and snap['paused'] and not getattr(self, '_was_paused', False):
            self._post('pause', on=False)
        self.mode = 'live'

    def select_record(self, idx):
        if self.cat.get('click') is not None and self.cat['click'][0] != idx:
            self.cat['click'] = None          # a selection change breaks a double click
        self.cat['sel'] = idx
        self.cat['result'] = None
        self.cat['play_label'] = ""
        self.status = ""

    def selected_record(self):
        i = self.cat['sel']
        if i is None or i >= len(self.cat['entries']):
            return None
        return self.cat['entries'][i][0]

    def play_record(self, output, replay=False):
        rec = self.selected_record()
        if rec is None:
            return False
        if not self.engine.device_ok:
            self.status = "No audio device: cannot play"
            return False
        try:
            if replay:
                from casynth_lab.catalog import read_wav
                pcm = read_wav(os.path.join(rec.dir, 'replay', f"{output}.wav"))
            else:
                pcm = rec.pcm(output)
        except CatalogError as e:
            self.status = f"Cannot play: {e}"
            return False
        self.engine.play_pcm(pcm)
        self.status = f"Playing {'replay ' if replay else ''}{output}: {rec.title}"
        return True

    def continue_record(self):
        """S5 Continue: restore the selected record's END snapshot into a
        live session that carries on from the next block (no history is
        replayed).  The current session is replaced only after the snapshot
        has been validated; on failure it stays as it is."""
        rec = self.selected_record()
        if rec is None or self.cat['thread'] is not None:
            return False
        plan = self.catalog.source_plan(rec)
        if plan['mode'] is None:
            self.status = f"Cannot continue: {plan['reason']}"
            return False
        if plan['mode'] == 'worktree':
            return self.continue_in_version(rec, plan)
        ok, why = rec.can_continue()
        if not ok:
            self.status = f"Cannot continue: {why}"
            return False
        try:
            runner, state, rec = self.catalog.continue_runner(rec.id)
        except CatalogError as e:
            self.status = f"Cannot continue: {e}"
            return False
        self.engine.stop_play()
        self.replace_session(runner=runner, origin_snapshot=state, parent_record_id=rec.id)
        self.engine.muted = False
        self.mode = 'live'
        how = ("stopped" if not runner.running else "CA paused" if runner.paused else "running")
        self.status = f"Continued: {rec.title}  ({how}; {plan['label']})"
        return True

    def continue_in_version(self, rec, plan):
        """S6: the record is pinned to another version -> a separate bench of
        that version is started from its cached checkout (it sees the same
        catalog).  This bench stays in the catalog screen, muted, until the
        child exits; a failure keeps the session and explains."""
        self.engine.stop_play()
        self.status = f"Checking out version {prov.short(plan['commit'])}..."
        try:
            child = self.catalog.run_version(rec, 'continue')
        except (VersionError, CatalogError) as e:
            self.status = f"Cannot continue in {prov.short(plan['commit'])}: {e}"
            return False
        self.child = child
        self.status = f"Running version {prov.short(plan['commit'])} in a separate bench..."
        return True

    def pin_record(self):
        """S6 Pin: attach a local record to the commit whose runtime files have
        exactly its fingerprint (only the provenance/status fields change)."""
        rec = self.selected_record()
        if rec is None:
            return False
        try:
            commit = self.catalog.pin(rec.id)
        except CatalogError as e:
            self.status = f"Cannot pin: {e}"
            return False
        self.status = f"Pinned to {prov.short(commit)}: {rec.title}"
        self.refresh_catalog()
        return True

    def open_in_bench(self):
        """Open field anew: load the selected record's END state (field, both
        sides' engine + params, selected side, volume) into a fresh live
        session (fresh sound, CA paused); the user starts it manually."""
        rec = self.selected_record()
        if rec is None or self.cat['thread'] is not None:
            return False
        self.engine.stop_play()
        if rec.meta.get('state_at_end'):
            return self._finish_open(rec, None)
        # older record: the end state is recomputed in a child process with a
        # progress bar; the session is swapped on the UI thread when it is ready
        self.cat['cancel'].clear()
        self.cat['progress'] = 0.0
        self.cat['result'] = None
        self.cat['open'] = None
        self.status = "Opening: rebuilding the end state..."

        def prog(f):
            self.cat['progress'] = f

        def work():
            try:
                self.cat['open'] = (rec, self.catalog.end_state_in_subprocess(
                    rec.id, progress=prog, cancel=self.cat['cancel'].is_set))
            except CatalogError as e:
                self.cat['open'] = (rec, e)
        th = threading.Thread(target=work, daemon=True)
        self.cat['thread'] = th
        th.start()
        return True

    def _finish_open(self, rec, state):
        try:
            scene, vol = bench_scene(rec, state)
        except CatalogError as e:
            self.status = f"Cannot open: {e}"
            return False
        self.replace_session(scene, vol, parent_record_id=rec.id)
        self.engine.muted = False
        self.mode = 'live'
        self.status = f"Opened field anew: {rec.title}  (release Pause CA to start)"
        return True

    def replace_session(self, scene=None, vol=None, runner=None, origin_snapshot=None,
                        parent_record_id=None):
        """Swap the live session for a new LiveEngine: on a fresh DemoRunner
        for `scene` (Open field anew) or on an already restored `runner`
        (Continue; its recording starts at `origin_snapshot`).  New records of
        the session point to `parent_record_id`."""
        from casynth_lab.audio_out import LiveEngine
        old = self.engine
        old.stop()
        if runner is None:
            runner = DemoRunner(scene, vol=vol)
        eng = LiveEngine(runner, output_factory=old._factory, sink=old._sink,
                         record_root=old._record_root, record_seconds=old._record_seconds,
                         origin_snapshot=origin_snapshot, parent_record_id=parent_record_id)
        eng.start()
        self.scene = runner.scene
        self.engine = eng
        self.vol = runner.vol
        self.message = ""
        self.session_record = parent_record_id      # Notes belong to this record now

    def start_replay(self):
        rec = self.selected_record()
        if rec is None or self.cat['thread'] is not None:
            return False
        if self.checking:
            self.status = "Catalog check running: no single check meanwhile"
            return False
        self.cat['cancel'].clear()
        self.cat['result'] = None
        self.cat['progress'] = 0.0
        self.cat['play_label'] = ""
        self.status = "Checking reproducibility..."

        def prog(f):
            self.cat['progress'] = f

        def work():
            self.cat['result'] = self.catalog.replay_in_subprocess(
                rec.id, progress=prog, cancel=self.cat['cancel'].is_set)
        th = threading.Thread(target=work, daemon=True)
        self.cat['thread'] = th
        th.start()
        return True

    def cancel_replay(self):
        self.cat['cancel'].set()

    CAT_TOP = 56

    def _catalog_rects(self):
        px = self.panel_x
        y = self.CAT_TOP
        rects = {}
        rects['continue'] = (px, y, 262, 34)
        y += 40
        rects.update(dict(back=(MARGIN, 12, 110, 30),
                          check=(MARGIN + 120, 12, 130, 30),       # S7: Check catalog / Cancel
                          report=(MARGIN + 260, 12, 110, 30),      # S7: Last report
                          open=(px, y, 262, 28),
                          play_A=(px, y + 44, 90, 26), play_B=(px + 96, y + 44, 90, 26),
                          play_mon=(px, y + 76, 186, 26), stop=(px + 192, y + 44, 70, 58),
                          replay=(px, y + 134, 186, 28), cancel=(px + 192, y + 134, 70, 28),
                          play_replay=(px, y + 194, 186, 26),
                          parent=(px, y + 232, 262, 18),
                          version=(px, y + 252, 262, 18),
                          pin=(px, y + 274, 120, 24),
                          notes=(px + 126, y + 274, 136, 24)))
        return rects

    def _list_rect(self, i):
        return (MARGIN, self.CAT_TOP + i * ROW_LIST_H, self.field_w, ROW_LIST_H - 4)

    CAT_ROWS = 12                    # records listed at once (wheel scrolls)
    DOUBLE_CLICK_S = 0.4             # second click on the same record -> Continue

    def _press_catalog(self, pos, button, now=None):
        if button in (4, 5):             # mouse wheel: scroll the list
            n = len(self.cat['entries'])
            first = self.cat.get('scroll', 0) + (-1 if button == 4 else 1)
            self.cat['scroll'] = max(0, min(first, max(0, n - self.CAT_ROWS)))
            return 'scroll'
        if button != 1:
            return None
        r = self._catalog_rects()
        if self.child is not None and self.child.alive:
            self.status = "A separate version bench is running: close it first"
            return 'busy'
        if self._inside(r['check'], pos):
            if self.checking:
                self.cancel_check()
                return 'check:cancel'
            self.start_check()
            return 'check'
        if self.checking:
            self.status = "Catalog check running: wait or cancel it first"
            return 'busy'
        if self._inside(r['report'], pos):
            self.open_report()
            return 'report'
        if self._inside(r['back'], pos):
            self.close_catalog()
            return 'back'
        first = self.cat.get('scroll', 0)
        now = time.monotonic() if now is None else now
        for i in range(first, min(len(self.cat['entries']), first + self.CAT_ROWS)):
            if self._inside(self._list_rect(i - first), pos):
                last = self.cat.get('click')
                self.select_record(i)
                if last is not None and last[0] == i and 0.0 <= now - last[1] < self.DOUBLE_CLICK_S:
                    # double click = the Continue button of that record
                    self.cat['click'] = None
                    self.continue_record()
                    return 'continue'
                self.cat['click'] = (i, now)
                return f'record:{i}'
        if self._inside(r['continue'], pos):
            self.continue_record()
            return 'continue'
        if self._inside(r['open'], pos):
            self.open_in_bench()
            return 'open'
        if self._inside(r['parent'], pos) and self.goto_parent():
            return 'parent'
        rec = self.selected_record()
        if rec is not None and self._inside(r['notes'], pos):
            self.open_notes()
            return 'notes'
        if rec is not None and rec.status != 'pinned' and self._inside(r['pin'], pos):
            self.pin_record()
            return 'pin'
        if self._inside(r['play_A'], pos):
            self.play_record('A')
            return 'play:A'
        if self._inside(r['play_B'], pos):
            self.play_record('B')
            return 'play:B'
        if self._inside(r['play_mon'], pos):
            self.play_record('monitor')
            return 'play:monitor'
        if self._inside(r['stop'], pos):
            self.engine.stop_play()
            self.status = ""
            return 'play:stop'
        if self._inside(r['replay'], pos):
            self.start_replay()
            return 'replay'
        if self._inside(r['cancel'], pos):
            self.cancel_replay()
            return 'replay:cancel'
        if self._inside(r['play_replay'], pos) and self.cat['play_label']:
            self.play_record('monitor', replay=True)
            return 'play:replay'
        return None

    def goto_parent(self):
        """Select the parent record of the selected one (if it is listed)."""
        rec = self.selected_record()
        if rec is None or not rec.parent_id:
            return False
        for i, (r, _err) in enumerate(self.cat['entries']):
            if r is not None and r.id == rec.parent_id:
                self.select_record(i)
                return True
        self.status = "Parent record is not in the catalog"
        return False

    def _draw_catalog(self, screen, font, small):
        import pygame
        r = self._catalog_rects()
        for key, label in (('check', 'Cancel check' if self.checking else 'Check catalog'),
                           ('report', 'Last report')):
            rect = r[key]
            hot = (key == 'check' and self.checking)
            pygame.draw.rect(screen, C_BTN_ON if hot else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = small.render(label, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        if self.status:
            col = C_ERR if self.status.lower().startswith(('cannot', 'no audio', 'replay unavail',
                                                             'replay differs', 'check failed')) \
                else C_OK
            screen.blit(small.render(self.status[:64], True, col), (MARGIN + 390, 20))
        else:
            screen.blit(small.render("Catalog (local records, newest first)", True, C_DIM),
                        (MARGIN + 390, 20))
        entries = self.cat['entries']
        if not entries:
            screen.blit(small.render("No records yet. Press Save in the live view.",
                                     True, C_DIM), (MARGIN, self.CAT_TOP + 6))
        if self.checking:
            v = self.verify
            y = self.height - 52
            line = (f"Checking {v['index'] + 1} of {v['total']}: {v['current']}"
                    if v['total'] else "Checking catalog: starting...")
            screen.blit(small.render(line[:70], True, C_ACCENT), (MARGIN, y))
            screen.blit(small.render(f"with {v['target']}"[:70], True, C_DIM), (MARGIN, y + 18))
            pygame.draw.rect(screen, C_EDGE, (MARGIN + 320, y + 4, 180, 8), border_radius=3)
            pygame.draw.rect(screen, C_ACCENT, (MARGIN + 320, y + 4, int(180 * v['progress']), 8),
                             border_radius=3)
        first = max(0, min(self.cat.get('scroll', 0), max(0, len(entries) - self.CAT_ROWS)))
        if len(entries) > self.CAT_ROWS:
            screen.blit(small.render(f"{first + 1}-{min(len(entries), first + self.CAT_ROWS)} "
                                     f"of {len(entries)} (wheel scrolls)", True, C_DIM),
                        (MARGIN + 390, self.CAT_TOP - 16))
        for i, (rec, err) in enumerate(entries[first:first + self.CAT_ROWS], start=first):
            rect = self._list_rect(i - first)
            on = (i == self.cat['sel'])
            pygame.draw.rect(screen, C_BTN_ON if on else C_PANEL, rect, border_radius=4)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=4)
            if rec is None:
                screen.blit(small.render(f"(broken record) {err}"[:80], True, C_ERR),
                            (rect[0] + 8, rect[1] + 12))
                continue
            screen.blit(small.render(rec.title[:60], True, C_TXT), (rect[0] + 8, rect[1] + 4))
            la, lb = rec.engine_labels()
            tag = ("Pinned " + prov.short(rec.commit)) if rec.status == 'pinned' else "Local"
            if rec.parent_id:
                tag += ", branch"
            if rec.has_notes:
                tag += ", notes"
            line = (f"{rec.created.replace('T', ' ')}   {rec.seconds:.1f} s   "
                    f"A: {la}  B: {lb}   {tag}")
            screen.blit(small.render(line, True, C_OK if rec.status == 'pinned' else C_DIM),
                        (rect[0] + 8, rect[1] + 22))
        labels = {'back': 'Back (Esc)', 'continue': 'Continue', 'open': 'Open field anew',
                  'play_A': 'Play A', 'play_B': 'Play B', 'play_mon': 'Play as heard',
                  'stop': 'Stop', 'replay': 'Check reproducibility', 'cancel': 'Cancel',
                  'play_replay': self.cat['play_label'], 'parent': '', 'version': '',
                  'pin': 'Pin to commit', 'check': '', 'report': '',
                  'notes': 'Notes'}
        rec = self.selected_record()
        can_cont, cont_why = rec.can_continue() if rec is not None else (False, '')
        plan = None
        if rec is not None:
            plan = self.catalog.source_plan(rec)
            if plan['mode'] is None:
                can_cont, cont_why = False, plan['reason']
            elif plan['mode'] == 'worktree':
                can_cont, cont_why = True, ''
                labels['continue'] = f"Continue in version {prov.short(plan['commit'])}"
            elif rec.status == 'pinned':
                labels['continue'] = "Continue (this version)"
            else:
                labels['continue'] = "Continue (local / current code)"
            labels['version'] = rec.version_label()
            if plan['mode'] is None:
                labels['version'] += f"  |  source: {plan['reason']}"
            elif plan['mode'] == 'worktree':
                labels['version'] += "  |  " + (plan.get('reason') or "source version runnable here")
        if rec is not None:
            ptitle, present = self.catalog.parent_of(rec)
            if ptitle:
                labels['parent'] = f"Derived from: {ptitle}" + ("" if present else " (missing)")
        for key, rect in r.items():
            if key in ('play_replay', 'parent', 'check', 'report') and not labels[key]:
                continue
            if key != 'back' and rec is None:
                continue
            if key == 'parent':
                t = small.render(labels[key][:44], True, C_ACCENT)
                screen.blit(t, (rect[0], rect[1]))
                continue
            if key == 'version':
                col = C_OK if rec.status == 'pinned' else C_WARN
                t = small.render(labels[key][:46], True, col)
                screen.blit(t, (rect[0], rect[1]))
                continue
            if key == 'notes':
                open_here = (self.notes_window is not None and self.notes_window.alive
                             and self.notes_window.rid == rec.id)
                labels[key] = 'Notes' + ('  (open)' if open_here else ('  *' if rec.has_notes else ''))
            if key == 'pin':
                if rec.status == 'pinned':
                    continue
                pygame.draw.rect(screen, C_BTN, rect, border_radius=4)
                pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
                t = small.render(labels[key], True, C_TXT)
                screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                                rect[1] + (rect[3] - t.get_height()) // 2))
                continue
            hot = ((key == 'stop' and self.engine.playing) or (key == 'continue' and can_cont)
                   or (key == 'cancel' and self.cat['thread'] is not None))
            if key == 'continue' and not can_cont:
                pygame.draw.rect(screen, C_PANEL, rect, border_radius=4)
                pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
                t = small.render(f"Continue unavailable: {cont_why}"[:40], True, C_DIM)
                screen.blit(t, (rect[0] + 8, rect[1] + (rect[3] - t.get_height()) // 2))
                continue
            pygame.draw.rect(screen, C_BTN_ON if hot else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = small.render(labels[key], True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        px = self.panel_x
        if self.child is not None and self.child.alive:
            screen.blit(small.render("Separate version bench running (this one is muted)",
                                     True, C_WARN), (MARGIN, self.height - 30))
        if rec is not None:
            y = r['pin'][1] + 30
            screen.blit(small.render("Description:", True, C_DIM), (px, y))
            for j, line in enumerate(_wrap(rec.note, small, PANEL_W)[:6]):
                screen.blit(small.render(line, True, C_TXT), (px, y + 18 + j * 17))
            y2 = y + 130
            if self.cat['thread'] is not None:
                screen.blit(small.render(self.status[:40], True, C_ACCENT), (px, y2 - 18))
                pygame.draw.rect(screen, C_EDGE, (px, y2, 180, 8), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (px, y2, int(180 * self.cat['progress']), 8),
                                 border_radius=3)
            res = self.cat['result']
            if res is not None:
                col = C_OK if res.status == 'match' else C_ERR
                for j, line in enumerate(_wrap(res.text, small, PANEL_W)[:4]):
                    screen.blit(small.render(line, True, col), (px, y2 + 16 + j * 17))
            if self.engine.playing:
                pos, n = self.engine.play_pos
                screen.blit(small.render(f"playing {pos / 44100:.1f} / {n / 44100:.1f} s",
                                         True, C_ACCENT), (px, y2 + 90))


    # =====================================================================
    # S7: catalog check + report screen (saved / recomputed listening)
    # =====================================================================
    @property
    def checking(self):
        th = self.verify['thread']
        return th is not None and th.is_alive()

    def start_check(self):
        """Check catalog: every record listed now is recomputed by ONE worker
        process (its code + environment = the report's target); this bench
        stays responsive in the catalog screen.  One check at a time."""
        if self.verifier is None or self.checking:
            return False
        if self.cat['thread'] is not None:
            self.status = "A single check is running: wait for it first"
            return False
        if not self.catalog.ids():
            self.status = "Catalog is empty: nothing to check"
            return False
        v = self.verify
        v['cancel'].clear()
        v.update(progress=0.0, index=0, total=0, current='', target='', done=None, error=None)
        self.engine.stop_play()
        self.status = "Checking catalog..."

        def on_record(i, n, rid):
            title = rid
            for rec, _e in self.cat['entries']:
                if rec is not None and rec.id == rid:
                    title = rec.title
            v.update(index=i, total=n, current=title, progress=0.0)

        def on_line(f):
            v['progress'] = f

        def work():
            try:
                v['done'] = self.verifier.run_in_subprocess(
                    progress=on_line, cancel=v['cancel'].is_set, on_record=on_record)
            except VerifyError as e:
                v['error'] = str(e)
        th = threading.Thread(target=work, daemon=True)
        v['thread'] = th
        th.start()
        return True

    def cancel_check(self):
        self.verify['cancel'].set()
        self.status = "Cancelling the check..."

    # -- report screen ----------------------------------------------------------
    def open_report(self, run=None):
        """Show a report (default: the latest).  No render happens here."""
        if self.verifier is None:
            return False
        if run is None:
            try:
                run = self.verifier.latest()
            except VerifyError as e:
                self.status = f"Cannot open the report: {e}"
                return False
            if run is None:
                self.status = "No catalog check yet"
                return False
        self.engine.stop_play()
        v = self.verify
        v['run'] = run
        g = run.groups()
        v['group'] = next((k for k in TRACK_RESULTS if g[k]), TRACK_DIFFERS)
        v['sel'] = None
        v['details'] = False
        self.mode = 'report'
        if g[v['group']]:
            self.select_report_record(0)
        self.status = f"Check {run.status_label}: {run.summary()}"
        return True

    def close_report(self):
        self.engine.stop_play()
        self.mode = 'catalog'
        self.refresh_catalog()

    def open_report_neighbour(self, step):
        """Older (+1) / newer (-1) report than the one shown."""
        ids = self.verifier.run_ids()
        cur = self.verify['run']
        if cur is None or cur.id not in ids:
            return False
        i = ids.index(cur.id) + step
        if not (0 <= i < len(ids)):
            return False
        try:
            run = self.verifier.load_run(ids[i])
        except VerifyError as e:
            self.status = f"Cannot open the report: {e}"
            return False
        return self.open_report(run)

    def report_group_ids(self):
        run = self.verify['run']
        return run.groups()[self.verify['group']] if run is not None else []

    def set_report_group(self, group):
        self.verify['group'] = group
        self.verify['sel'] = None
        self.engine.stop_play()
        if self.report_group_ids():
            self.select_report_record(0)

    def report_record_id(self):
        ids = self.report_group_ids()
        i = self.verify['sel']
        return ids[i] if i is not None and i < len(ids) else None

    def report_entry(self):
        rid = self.report_record_id()
        return self.verify['run'].result(rid) if rid else None

    def select_report_record(self, idx):
        """Select a record of the shown group: the first track opened is the
        changed monitor, else the first changed A/B track; version = saved."""
        self.engine.stop_play()
        v = self.verify
        v['sel'] = idx
        rid = self.report_record_id()
        if rid is None:
            return
        v['track'] = v['run'].default_track(rid)
        v['version'] = 'saved'

    def set_track(self, output):
        """A / B / As heard.  A playing pair goes on at the same position."""
        v = self.verify
        if output == v['track']:
            return
        was = self.engine.playing
        pos = self.engine.play_pos[0] if was else 0
        self.engine.stop_play()
        v['track'] = output
        if was:
            self.play_pair(pos)

    def set_version(self, version):
        """Saved / Recomputed: the SAME cursor continues (never a restart)."""
        v = self.verify
        v['version'] = version
        if self.engine.pair_version is not None:
            self.engine.switch_version(version)

    def _pair_for(self, rid, output):
        """(saved, recomputed) PCM of the selected track, checked against the
        report's fingerprints; an exact track plays the original for both."""
        run = self.verify['run']
        t = run.result(rid)['tracks'].get(output) or {}
        if t.get('result') == TRACK_DIFFERS:
            return self.verifier.pair(run.id, rid, output)
        rec = self.catalog.load(rid)
        ref = rec.pcm(output)
        if t.get('result') == TRACK_EXACT:
            if verify_mod.pcm_sha(ref) != t.get('ref_sha256'):
                raise VerifyError("the original WAV changed since the check")
            return ref, ref
        return ref, None            # failed / unchecked: only the original exists

    def play_pair(self, pos=0):
        rid = self.report_record_id()
        if rid is None:
            return False
        if not self.engine.device_ok:
            self.status = "No audio device: cannot play"
            return False
        v = self.verify
        try:
            saved, new = self._pair_for(rid, v['track'])
        except (VerifyError, CatalogError) as e:
            self.status = f"Cannot play: {e}"
            return False
        if new is None:
            if v['version'] == 'recomputed':
                self.status = "No recomputed version for this track (see its result)"
                return False
            new = saved
        try:
            self.engine.play_pair(saved, new, which=v['version'], pos=pos)
        except ValueError as e:
            self.status = f"Cannot play: {e}"
            return False
        self.status = f"Playing {TRACK_LABELS[v['track']]} ({v['version']})"
        return True

    def stop_pair(self):
        self.engine.stop_play()
        self.status = ""

    def set_mark(self, mark):
        """Listening mark for THIS report's pair of the shown track:
        'same' (can't hear a difference) / 'different' / None (not rated)."""
        rid = self.report_record_id()
        v = self.verify
        if rid is None:
            return False
        try:
            marks = self.verifier.set_mark(v['run'].id, rid, v['track'], mark)
        except VerifyError as e:
            self.status = f"Cannot mark: {e}"
            return False
        v['run'].marks = marks
        self.status = "Mark saved" if mark else "Mark cleared"
        return True

    REPORT_LIST_TOP = 96

    def _report_rects(self):
        px = self.panel_x
        rects = dict(back=(MARGIN, 12, 110, 30),
                     older=(self.width - MARGIN - 150, 12, 70, 30),
                     newer=(self.width - MARGIN - 74, 12, 74, 30))
        x = MARGIN
        for k in TRACK_RESULTS:
            rects[f'group:{k}'] = (x, self.CAT_TOP, 124, 30)
            x += 128
        y = self.CAT_TOP + 120
        rects.update({'track:A': (px, y, 80, 28), 'track:B': (px + 84, y, 80, 28),
                      'track:monitor': (px + 168, y, 94, 28),
                      'version:saved': (px, y + 36, 129, 30),
                      'version:recomputed': (px + 133, y + 36, 129, 30),
                      'play': (px, y + 72, 100, 30), 'stop': (px + 104, y + 72, 100, 30),
                      'mark:same': (px, y + 150, 84, 26), 'mark:different': (px + 88, y + 150, 84, 26),
                      'mark:none': (px + 176, y + 150, 86, 26),
                      'details': (px, y + 186, 100, 24)})
        return rects

    def _report_list_rect(self, i):
        return (MARGIN, self.REPORT_LIST_TOP + i * ROW_LIST_H, self.field_w, ROW_LIST_H - 4)

    def _press_report(self, pos, button):
        if button != 1:
            return None
        r = self._report_rects()
        if self._inside(r['back'], pos):
            self.close_report()
            return 'report:back'
        if self._inside(r['older'], pos):
            self.open_report_neighbour(+1)
            return 'report:older'
        if self._inside(r['newer'], pos):
            self.open_report_neighbour(-1)
            return 'report:newer'
        for k in TRACK_RESULTS:
            if self._inside(r[f'group:{k}'], pos):
                self.set_report_group(k)
                return f'group:{k}'
        ids = self.report_group_ids()
        for i in range(min(len(ids), 11)):
            if self._inside(self._report_list_rect(i), pos):
                self.select_report_record(i)
                return f'report:{i}'
        if self.report_record_id() is None:
            return None
        for o in ('A', 'B', 'monitor'):
            if self._inside(r[f'track:{o}'], pos):
                self.set_track(o)
                return f'track:{o}'
        for ver in ('saved', 'recomputed'):
            if self._inside(r[f'version:{ver}'], pos):
                self.set_version(ver)
                return f'version:{ver}'
        if self._inside(r['play'], pos):
            self.play_pair()
            return 'play'
        if self._inside(r['stop'], pos):
            self.stop_pair()
            return 'stop'
        for key, mark in (('mark:same', MARK_SAME), ('mark:different', MARK_DIFFERENT),
                          ('mark:none', None)):
            if self._inside(r[key], pos):
                self.set_mark(mark)
                return key
        if self._inside(r['details'], pos):
            self.verify['details'] = not self.verify['details']
            return 'details'
        return None

    def _draw_report(self, screen, font, small):
        import pygame
        v = self.verify
        run = v['run']
        r = self._report_rects()

        def button(rect, label, hot=False, dim=False):
            pygame.draw.rect(screen, C_BTN_ON if hot else (C_PANEL if dim else C_BTN), rect,
                             border_radius=4)
            pygame.draw.rect(screen, C_ACCENT if hot else C_EDGE, rect, 1, border_radius=4)
            t = small.render(label, True, C_DIM if dim else C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        button(r['back'], 'Back (Esc)')
        button(r['older'], '< older')
        button(r['newer'], 'newer >')
        if run is None:
            screen.blit(small.render("No report", True, C_DIM), (MARGIN + 130, 20))
            return
        head = f"Check {run.id}   {run.status_label}"
        col = C_OK if run.status == 'complete' else C_WARN
        screen.blit(small.render(head[:60], True, col), (MARGIN + 130, 12))
        screen.blit(small.render(f"with {run.target_label}"[:64], True, C_DIM), (MARGIN + 130, 30))
        counts = run.counts()
        names = {TRACK_DIFFERS: 'Differs', TRACK_EXACT: 'Exact', TRACK_FAILED: 'Failed',
                 TRACK_UNCHECKED: 'Unchecked'}
        for k in TRACK_RESULTS:
            button(r[f'group:{k}'], f"{names[k]} ({counts[k]})", hot=(k == v['group']))
        ids = self.report_group_ids()
        if not ids:
            screen.blit(small.render("Nothing in this group.", True, C_DIM),
                        (MARGIN, self.REPORT_LIST_TOP + 6))
        for i, rid in enumerate(ids[:11]):
            e = run.result(rid)
            rect = self._report_list_rect(i)
            on = (i == v['sel'])
            pygame.draw.rect(screen, C_BTN_ON if on else C_PANEL, rect, border_radius=4)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=4)
            screen.blit(small.render(e.get('title', rid)[:60], True, C_TXT),
                        (rect[0] + 8, rect[1] + 4))
            tr = e['tracks']
            line = "   ".join(f"{TRACK_LABELS[o]}: {RESULT_LABELS[tr[o]['result']].lower()}"
                              for o in ('A', 'B', 'monitor') if o in tr)
            if e['status'] in (TRACK_FAILED, TRACK_UNCHECKED) and e.get('reason'):
                line = f"{RESULT_LABELS[e['status']]}: {e['reason']}"
            screen.blit(small.render(line[:88], True, C_DIM), (rect[0] + 8, rect[1] + 22))
        rid = self.report_record_id()
        if rid is None:
            return
        e = run.result(rid)
        px, y = self.panel_x, self.CAT_TOP
        screen.blit(font.render(e.get('title', rid)[:30], True, C_TXT), (px, y))
        origin = e.get('origin') or {}
        if origin.get('status') == 'pinned':
            olab, ocol = f"Origin: Pinned {prov.short(origin.get('commit'))}", C_OK
        elif origin:
            olab, ocol = f"Origin: Local: {origin.get('status_reason') or '?'}", C_WARN
        else:
            olab, ocol = "Origin: unknown", C_DIM
        for j, line in enumerate(_wrap(olab, small, PANEL_W)[:2]):
            screen.blit(small.render(line, True, ocol), (px, y + 24 + j * 17))
        for j, o in enumerate(('A', 'B', 'monitor')):
            t = e['tracks'].get(o)
            if t and t['result'] == TRACK_DIFFERS:
                what = (f"differs: {100.0 * (t['frac_diff'] or 0):.1f}% smp, max {t['max_abs_diff']}"
                        + ("; length!" if t.get('length_mismatch') else ""))
            else:
                what = track_text(t) if t else 'not checked'
            txt = f"{TRACK_LABELS[o]}: {what}"
            tcol = {TRACK_EXACT: C_OK, TRACK_DIFFERS: C_WARN, TRACK_FAILED: C_ERR}.get(
                t['result'] if t else None, C_DIM)
            screen.blit(small.render(txt[:44], True, tcol), (px, y + 62 + j * 17))
        for o in ('A', 'B', 'monitor'):
            button(r[f'track:{o}'], TRACK_LABELS[o], hot=(o == v['track']))
        t = e['tracks'].get(v['track']) or {}
        has_new = t.get('result') == TRACK_DIFFERS
        for ver, label in (('saved', 'Saved'), ('recomputed', 'Recomputed')):
            button(r[f'version:{ver}'], label, hot=(ver == v['version']),
                   dim=(ver == 'recomputed' and not has_new and t.get('result') != TRACK_EXACT))
        button(r['play'], 'Play', hot=self.engine.playing)
        button(r['stop'], 'Stop')
        yy = r['play'][1] + 36
        if self.engine.playing:
            pos, n = self.engine.play_pos
            screen.blit(small.render(f"{pos / 44100:.1f} / {n / 44100:.1f} s  "
                                     f"({self.engine.pair_version or v['version']})",
                                     True, C_ACCENT), (px, yy))
        elif t.get('length_mismatch'):
            screen.blit(small.render(f"lengths differ: saved {t['n_ref']}, "
                                     f"recomputed {t['n_new']} samples", True, C_WARN), (px, yy))
        screen.blit(small.render("Listening mark (this pair, this report):", True, C_DIM),
                    (px, yy + 24))
        if has_new:
            mark = run.mark(rid, v['track'])
            button(r['mark:same'], "Can't hear", hot=(mark == MARK_SAME))
            button(r['mark:different'], "Hear it", hot=(mark == MARK_DIFFERENT))
            button(r['mark:none'], "Not rated", hot=(mark is None))
        else:
            screen.blit(small.render("(only for a differing track)", True, C_DIM),
                        (px, yy + 44))
        button(r['details'], 'Details', hot=v['details'])
        if v['details']:
            dy = r['details'][1] + 30
            for j, o in enumerate(('A', 'B', 'monitor')):
                tt = e['tracks'].get(o) or {}
                if tt.get('n_ref') is None:
                    line = f"{TRACK_LABELS[o]}: {tt.get('reason') or '-'}"
                else:
                    line = (f"{TRACK_LABELS[o]}: {tt['n_ref']}/{tt['n_new']} smp  "
                            f"{100.0 * (tt['frac_diff'] or 0):.2f}% differ  max|d| {tt['max_abs_diff']}")
                screen.blit(small.render(line[:46], True, C_TXT), (px, dy + j * 17))
        if self.status:
            col = C_ERR if self.status.lower().startswith(('cannot', 'no audio', 'no recomputed',
                                                             'check failed')) else C_OK
            screen.blit(small.render(self.status[:44], True, col), (px, self.height - 30))


def _wrap(text, font, width):
    words, lines, cur = (text or '').split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if font.size(t)[0] > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def _print_provenance():
    """First line of the launch contract: what code THIS process runs."""
    doc = prov.current()
    print(json.dumps({'provenance': {k: doc.get(k) for k in
                                     ('commit', 'digest', 'match', 'reason', 'root',
                                      'repo_id', 'environment')}}), flush=True)
    st, why = prov.status_of(doc)
    print(f"[code] {st} {prov.short(doc.get('commit')) or '?'} {why}".rstrip(), flush=True)


def run_ui(scene, vol, catalog=None, runner=None, origin_snapshot=None, parent_record_id=None,
           caption='', start='catalog'):
    """`start`: 'catalog' (default) opens on the catalog screen -- the live
    scene stands paused and muted behind it, Esc / Back reaches it; 'live'
    opens the sounding field directly (--live; autotests, continued records)."""
    import pygame
    from casynth_lab.audio_out import LiveEngine
    catalog = catalog or Catalog()
    os.makedirs(catalog.tmp_root, exist_ok=True)
    if runner is None:
        runner = DemoRunner(scene, vol=vol)
    engine = LiveEngine(runner, record_root=catalog.tmp_root, origin_snapshot=origin_snapshot,
                        parent_record_id=parent_record_id)
    engine.start()
    print(f"[audio] {engine.status_text()}", flush=True)     # (a parent bench reads the pipe)
    pygame.init()
    pygame.key.set_repeat(400, 35)     # held keys repeat (Backspace, arrows in text fields)
    app = BenchApp(runner.scene, engine, catalog=catalog)
    app.vol = runner.vol
    app.session_record = parent_record_id
    if origin_snapshot is not None:
        app.status = f"Continued: {runner.scene.title}  ({caption})"
    elif start == 'catalog':
        app.open_catalog()
    screen = pygame.display.set_mode((app.width, app.height))
    pygame.display.set_caption(f"CASynth demo bench - {runner.scene.title}"
                               + (f"  [{caption}]" if caption else ''))
    try:                               # system clipboard for the text fields (Ctrl+X/C/V)
        pygame.scrap.init()
        app.clip_get, app.clip_put = pygame.scrap.get_text, pygame.scrap.put_text
    except Exception:                  # noqa: BLE001 -- none: the bench keeps its own
        pass
    font = pygame.font.SysFont(FONT_NAMES, 17)
    small = pygame.font.SysFont(FONT_NAMES, 14)
    clock = pygame.time.Clock()
    try:
        alive = True
        while alive:
            for ev in pygame.event.get():
                if app.notes_owns(ev):             # the notes window: its own form
                    app.notes_event(ev)
                    continue
                if ev.type == pygame.QUIT or ev.type == pygame.WINDOWCLOSE:
                    # with the notes window open SDL reports the bench window's close
                    # button as WINDOWCLOSE (no QUIT until the last window goes)
                    alive = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE \
                        and app.mode == 'live' and app.drag_pat is None and app.range_edit is None:
                    alive = False
                elif ev.type == pygame.KEYDOWN:
                    app.key(pygame.key.name(ev.key), ctrl=bool(ev.mod & pygame.KMOD_CTRL),
                            shift=bool(ev.mod & pygame.KMOD_SHIFT))
                elif ev.type == pygame.TEXTINPUT:
                    app.text_input(ev.text)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    app.press(ev.pos, ev.button,
                              shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
                elif ev.type == pygame.MOUSEWHEEL:
                    app.wheel(pygame.mouse.get_pos(), ev.y)
                elif ev.type == pygame.MOUSEMOTION and (ev.buttons[0] or ev.buttons[2]):
                    app.drag(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    app.release()
            app.draw(screen, font, small)
            pygame.display.flip()
            app.draw_notes(font, small)
            app.sync_notes()
            clock.tick(60)
    finally:
        app.close_notes()
        engine.stop()
        pygame.quit()
        if app.child is not None:
            app.child.terminate()
        if app.checking:                   # S7: the worker stops, its report closes as cancelled
            app.cancel_check()
            app.verify['thread'].join(timeout=10)
    return 0


def run_record(catalog_root, rid, action, headless=False, seconds=2.0, out=None):
    """Launch contract of a version bench / worker (S6):
        demo_bench.py --catalog ROOT --record ID --action continue|check
    The first stdout line is this process's provenance (JSON); a `status`
    JSON line ends a worker.  `continue` opens the live bench on the record's
    end snapshot (or, headless, renders `seconds` from it into `out`.npz);
    `check` = reproducibility check with THIS code."""
    _print_provenance()
    catalog = Catalog(os.path.abspath(catalog_root))
    doc = prov.current()
    caption = f"version {prov.short(doc.get('commit')) or 'local'}"
    if action == 'check':
        res = catalog.replay(rid, yield_cpu=False)
        print(json.dumps(dict(status=res.status, reason=res.reason, outputs=res.outputs,
                              commit=doc.get('commit'))), flush=True)
        return 0 if res.status == 'match' else 1
    try:
        runner, state, rec = catalog.continue_runner(rid)
    except CatalogError as e:
        print(json.dumps(dict(status='unavailable', reason=str(e))), flush=True)
        return 3
    if headless:
        import numpy as np
        from casynth_lab import BLOCK, OUTPUTS
        from casynth_lab.audio_out import LiveEngine
        got = {o: [] for o in OUTPUTS}
        n = int(round(seconds * 44100 / BLOCK))
        eng = LiveEngine(runner, sink=lambda m, b: [got[o].append(b.get(o)) for o in OUTPUTS],
                         record_root=catalog.tmp_root, origin_snapshot=state,
                         parent_record_id=rec.id)
        eng.start()
        import time
        t0 = time.time()
        while len(got['A']) < n and time.time() - t0 < 120:
            time.sleep(0.01)
        kind, cut = eng.cut_now()
        eng.stop()
        rid2 = catalog.save(cut, f"headless continuation of {rec.title}") if cut else None
        if out:
            np.savez(out, **{o: np.concatenate(got[o][:n]) for o in OUTPUTS})
        print(json.dumps(dict(status='ok', blocks=len(got['A'][:n]), saved=rid2,
                              commit=doc.get('commit'))), flush=True)
        return 0
    return run_ui(None, runner.vol, catalog=catalog, runner=runner, origin_snapshot=state,
                  parent_record_id=rec.id, caption=caption, start='live')


def run_render(scene, out_path, seconds, vol, side):
    from casynth_lab import render_offline, write_wav
    if seconds <= 0:
        print("error: --seconds must be > 0")
        return 2
    pcm, runner = render_offline(scene, seconds, vol=vol, output=side)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    write_wav(out_path, pcm)
    peak = float(abs(pcm).max()) / 32767.0 if len(pcm) else 0.0
    clip = runner.snapshot()['clip_blocks']
    print(f"[render] {out_path} ({side}): {len(pcm)} samples x {pcm.shape[1]} ch, "
          f"{seconds:g} s, gen {runner.gen}, peak {peak:.3f}, "
          f"clip blocks A {clip['A']} / B {clip['B']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="CASynth demo bench (S1/S2)")
    ap.add_argument('--demo', help="scene JSON (demos/*.json)")
    ap.add_argument('--render', metavar='WAV', help="offline render to WAV (no window)")
    ap.add_argument('--seconds', type=float, default=8.0, help="render length (s)")
    ap.add_argument('--side', choices=('A', 'B', 'monitor'), default='A',
                    help="offline output: raw side A/B or the monitor mix")
    ap.add_argument('--vol', type=float, default=VOL_DEFAULT, help="initial volume 0..1")
    ap.add_argument('--catalog', metavar='ROOT', help="catalog root (absolute; S6 contract)")
    ap.add_argument('--live', action='store_true',
                    help="open the sounding field directly instead of the catalog screen")
    ap.add_argument('--record', metavar='ID', help="record to act on (with --catalog)")
    ap.add_argument('--action', choices=('continue', 'check'), default='continue')
    ap.add_argument('--headless', action='store_true',
                    help="with --record continue: render --seconds from the snapshot, no window")
    ap.add_argument('--out', metavar='NPZ', help="headless continuation output (npz)")
    a = ap.parse_args(argv)
    if a.record:
        if not a.catalog:
            print("error: --record needs --catalog ROOT")
            return 2
        return run_record(a.catalog, a.record, a.action, headless=a.headless,
                          seconds=a.seconds, out=a.out)
    if not a.demo:
        print("error: --demo is required (or --catalog/--record)")
        return 2
    try:
        scene = load_scene(a.demo)
    except SceneError as e:
        print(f"error: {e}")
        return 2
    if a.render:
        return run_render(scene, a.render, a.seconds, a.vol, a.side)
    _print_provenance()
    catalog = Catalog(os.path.abspath(a.catalog)) if a.catalog else None
    return run_ui(scene, a.vol, catalog=catalog, start='live' if a.live else 'catalog')


if __name__ == '__main__':
    sys.exit(main())
