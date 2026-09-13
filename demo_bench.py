#!/usr/bin/env python3
"""Demo bench (S1 + S2): one live A/B demo window, or an offline render.

    python demo_bench.py --demo demos/laplace_ab.json
    python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor

Window: shared field, demo title + listening hint, transport (Start|Stop /
Pause CA / Restart), volume, generation, audio state; a side panel with the
A/B tabs (select = listen + edit), the selected side's engine and its registry
knobs, and a one-line summary of how A and B differ.  LMB paints, RMB erases.
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
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casynth_config import (VOL_DEFAULT, C_BG, C_GRID, C_PANEL, C_EDGE, C_TXT,
                            C_DIM, C_BTN, C_ACCENT)                    # noqa: E402
from casynth_lab import (load_scene, SceneError, DemoRunner, SIDES, registry,
                         describe_difference, validate_param)           # noqa: E402
from casynth_lab.catalog import Catalog, CatalogError, bench_scene     # noqa: E402
from casynth_lab import provenance as prov                             # noqa: E402
from casynth_lab.versions import VersionError                          # noqa: E402

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
ROW_H = 26
SLIDER_W = 120
FONT_NAMES = "segoeui,arial,dejavusans,freesans"
C_ALIVE = (111, 208, 224)
C_ALIVE_PAUSED = (200, 180, 90)
C_BTN_ON = (48, 92, 104)
C_ERR = (230, 120, 90)
C_WARN = (230, 180, 90)
C_OK = (120, 210, 140)
LAB_BTN_W = 104
ROW_LIST_H = 50


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
        self.child = None                 # ChildBench of another version (S6)
        self.mode = 'live'                # 'live' | 'save' | 'catalog'
        self.status = ""                  # save / replay status line
        self.save_form = None             # {'cut', 'title', 'note', 'field'}
        self._cut_pending = False
        self._save_thread = None
        self.cat = dict(entries=[], sel=None, result=None, thread=None,
                        progress=0.0, cancel=threading.Event(), play_label="")
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
        self.height = TOP_H + max(self.field_h, 420) + MARGIN
        self.buttons = {}
        x = MARGIN
        for key, label in (('stop', 'Stop (S)'), ('pause', 'Pause CA (Space)'), ('reset', 'Restart (R)')):
            self.buttons[key] = ((x, BTN_Y, BTN_W, BTN_H), label)
            x += BTN_W + 10
        self.vol_rect = (MARGIN + 92, BTN_Y + BTN_H + 14, VOL_W, 10)
        self.lab_buttons = {'save': (x + 10, BTN_Y, LAB_BTN_W, BTN_H),
                            'catalog': (x + 20 + LAB_BTN_W, BTN_Y, LAB_BTN_W, BTN_H)}
        # side panel
        px, py = self.panel_x, TOP_H
        # row: [A] [<<] [>>] [B]
        self.tabs = {'A': (px, py, TAB_W, TAB_H),
                     'B': (px + TAB_W + 2 * (ARROW_W + 4) + 4, py, TAB_W, TAB_H)}
        self.copy_btns = {('B', 'A'): (px + TAB_W + 4, py, ARROW_W, TAB_H),          # <<
                          ('A', 'B'): (px + TAB_W + ARROW_W + 8, py, ARROW_W, TAB_H)}  # >>
        self.factory_btn = (px + PANEL_W - 100, py + TAB_H + 6, 100, 22)
        self.engine_btns = {}
        ey = py + TAB_H + 48
        for i, e in enumerate(registry.specs()):
            col, row = i % 3, i // 3
            self.engine_btns[e.id] = (px + col * (ENG_W + 6), ey + row * (ENG_H + 6),
                                         ENG_W, ENG_H)
        self.params_y = ey + 2 * (ENG_H + 6) + 12
        self.slider_x = px + 60

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
        """[(spec, slider_rect)] for the selected side's engine."""
        rows = []
        for i, spec in enumerate(registry.get(eid).params):
            y = self.params_y + i * ROW_H
            rows.append((spec, (self.slider_x, y + 6, SLIDER_W, 10)))
        return rows

    def _post(self, kind, **args):
        try:
            self.engine.post(kind, **args)
            self.message = ""
            return True
        except ValueError as e:
            self.message = str(e)
            return False

    # -- input ---------------------------------------------------------------
    def press(self, pos, button):
        """Mouse button down. button: 1 = left, 3 = right.  Returns what was hit."""
        if self.mode == 'save':
            return self._press_save_form(pos, button)
        if self.mode == 'catalog':
            return self._press_catalog(pos, button)
        if button == 1 and self.catalog is not None:
            for key, rect in self.lab_buttons.items():
                if self._inside(rect, pos):
                    if key == 'save':
                        self.begin_save()
                    else:
                        self.open_catalog()
                    return key
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
            if self._inside(self.factory_btn, pos):
                self._post('factory')
                return 'factory'
            side, eid, params = self._side()
            for e_id, rect in self.engine_btns.items():
                if self._inside(rect, pos):
                    self._post('set_engine', side=side, engine_id=e_id)
                    return f'engine:{e_id}'
            for spec, (sx, sy, sw, sh) in self._param_rows(eid):
                if sx - 6 <= pos[0] < sx + sw + 6 and sy - 8 <= pos[1] < sy + sh + 8:
                    name = spec[0]
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
        if self.drag_vol:
            self._set_vol(pos[0])
        elif self.drag_param is not None:
            self._set_param_from_x(pos[0])
        elif self.paint_value is not None:
            cell = self.cell_at(pos)
            if cell is not None:
                self._paint(cell)

    def key(self, name, ctrl=False):
        """Keyboard hotkey by key name ('r' = Restart).  Returns the command or None.
        `ctrl`: a Control modifier is held (text editing: Ctrl+Backspace)."""
        name = name.lower()
        if self.mode == 'save':
            return self._key_save_form(name, ctrl)
        if self.mode == 'catalog':
            if name == 'escape':
                self.close_catalog()
                return 'back'
            return None
        if name == 'r':
            self._post('reset')
            return 'reset'
        if name == 's':
            self._post('stop')
            return 'stop'
        if name == 'space':
            self._post('pause')
            return 'pause'
        if name in ('1', '2', '[1]', '[2]'):
            side = 'A' if name.endswith('1') or name == '1' else 'B'
            self._post('select', side=side)
            return f'select:{side}'
        return None

    def release(self):
        self.paint_value = None
        self.drag_vol = False
        self.drag_param = None

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
        frac = min(max((mx - self.slider_x) / SLIDER_W, 0.0), 1.0)
        v = lo + frac * (hi - lo)
        v = int(round(v)) if integer else round(v, 3)
        if v != params[self.drag_param]:
            self._post('set_param', side=side, name=self.drag_param,
                       value=validate_param(eid, self.drag_param, v))

    # -- drawing -------------------------------------------------------------
    def draw(self, screen, font, small):
        import pygame
        self.tick()
        if self.mode == 'catalog':
            screen.fill(C_BG)
            self._draw_catalog(screen, font, small)
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
        # side panel
        side = snap['selected']
        eid, params = snap['sides'][side]
        for s, rect in self.tabs.items():
            on = (s == side)
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=4)
            star = '*' if snap['modified'][s] else ''
            hot = '1' if s == 'A' else '2'
            lbl = f"{s}{star}: {registry.label(snap['sides'][s][0])} ({hot})"
            t = font.render(lbl, True, C_TXT)
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
        for e_id, rect in self.engine_btns.items():
            on = (e_id == eid)
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=3)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=3)
            t = small.render(registry.label(e_id), True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        for spec, (sx, sy, sw, sh) in self._param_rows(eid):
            name, label, lo, hi, integer, _d = spec
            v = params[name]
            screen.blit(small.render(label, True, C_DIM), (px, sy - 4))
            if _is_toggle(spec):
                pygame.draw.rect(screen, C_ACCENT if v else C_BTN, (sx, sy - 3, 34, 16),
                                 border_radius=8)
                pygame.draw.rect(screen, C_EDGE, (sx, sy - 3, 34, 16), 1, border_radius=8)
                screen.blit(small.render("on" if v else "off", True, C_TXT), (sx + 40, sy - 4))
            else:
                frac = (v - lo) / (hi - lo) if hi > lo else 0.0
                pygame.draw.rect(screen, C_EDGE, (sx, sy, sw, sh), border_radius=3)
                pygame.draw.rect(screen, C_ACCENT, (sx, sy, int(sw * frac), sh), border_radius=3)
                txt = f"{v:d}" if integer else f"{v:.2f}"
                screen.blit(small.render(txt, True, C_TXT), (sx + sw + 8, sy - 4))
        pk = snap['peak']
        screen.blit(small.render(f"peak A {pk['A']:.2f}   B {pk['B']:.2f}", True, C_DIM),
                    (px, self.params_y + 7 * ROW_H + 8))
        if self.message:
            screen.blit(small.render(self.message[:60], True, C_ERR),
                        (px, self.params_y + 7 * ROW_H + 28))
        # S4: lab buttons + status; overlays
        if self.catalog is not None:
            for key, rect in self.lab_buttons.items():
                pygame.draw.rect(screen, C_BTN, rect, border_radius=4)
                pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
                t = font.render('Save' if key == 'save' else 'Catalog', True, C_TXT)
                screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                                rect[1] + (rect[3] - t.get_height()) // 2))
            if self.status:
                col = C_ERR if self.status.lower().startswith(('save failed', 'nothing',
                                                                 'no audio')) else C_OK
                screen.blit(small.render(self.status[:90], True, col), (MARGIN + 300, TOP_H - 26))
        if self.mode == 'save':
            self._draw_save_form(screen, font, small)

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
                    self.save_form = dict(cut=cut, field='title', note='',
                                          title=Catalog.default_title(self.scene.title))
                    self.mode = 'save'
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
                self.status = f"Saved: {form['title'] or rid}  [{form['cut'].seconds:.1f} s]"
            except CatalogError as e:
                self.status = f"Save failed: {e}"
        self._save_thread = threading.Thread(target=work, daemon=True)
        self._save_thread.start()

    def cancel_save(self):
        self.save_form = None
        self.mode = 'live'
        self.status = "Save cancelled"

    def text_input(self, text):
        f = self.save_form
        if f is not None:
            f[f['field']] += text

    @staticmethod
    def _erase_word(text):
        """Ctrl+Backspace: drop trailing spaces, then the last word."""
        t = text.rstrip(' ')
        i = t.rfind(' ')
        return t[:i + 1] if i >= 0 else ''

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
        if name == 'backspace':
            cur = f[f['field']]
            f[f['field']] = self._erase_word(cur) if ctrl else cur[:-1]
            return 'save:edit'
        return None

    def _save_form_rects(self):
        x, y, w = MARGIN + 40, TOP_H + 60, 560
        return dict(title=(x + 70, y + 40, w - 90, 26), note=(x + 70, y + 80, w - 90, 26),
                    ok=(x + 20, y + 130, 120, 30), cancel=(x + 160, y + 130, 120, 30),
                    box=(x, y, w, 180))

    def _press_save_form(self, pos, button):
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
        for field, label in (('title', 'Title'), ('note', 'Note')):
            rect = r[field]
            screen.blit(small.render(label, True, C_DIM), (bx + 20, rect[1] + 5))
            on = (f['field'] == field)
            pygame.draw.rect(screen, C_BG, rect, border_radius=3)
            pygame.draw.rect(screen, C_ACCENT if on else C_EDGE, rect, 1, border_radius=3)
            txt = f[field] + ('|' if on else '')
            screen.blit(small.render(txt[-70:], True, C_TXT), (rect[0] + 6, rect[1] + 5))
        for key, label in (('ok', 'Save (Enter)'), ('cancel', 'Cancel (Esc)')):
            rect = r[key]
            pygame.draw.rect(screen, C_BTN_ON if key == 'ok' else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = small.render(label, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        screen.blit(small.render("Tab switches field. The recording end is already fixed.",
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

    def start_replay(self):
        rec = self.selected_record()
        if rec is None or self.cat['thread'] is not None:
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
                          open=(px, y, 262, 28),
                          play_A=(px, y + 44, 90, 26), play_B=(px + 96, y + 44, 90, 26),
                          play_mon=(px, y + 76, 186, 26), stop=(px + 192, y + 44, 70, 58),
                          replay=(px, y + 134, 186, 28), cancel=(px + 192, y + 134, 70, 28),
                          play_replay=(px, y + 194, 186, 26),
                          parent=(px, y + 232, 262, 18),
                          version=(px, y + 252, 262, 18),
                          pin=(px, y + 274, 120, 24)))
        return rects

    def _list_rect(self, i):
        return (MARGIN, self.CAT_TOP + i * ROW_LIST_H, self.field_w, ROW_LIST_H - 4)

    def _press_catalog(self, pos, button):
        if button != 1:
            return None
        r = self._catalog_rects()
        if self.child is not None and self.child.alive:
            self.status = "A separate version bench is running: close it first"
            return 'busy'
        if self._inside(r['back'], pos):
            self.close_catalog()
            return 'back'
        for i in range(len(self.cat['entries'])):
            if self._inside(self._list_rect(i), pos):
                self.select_record(i)
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
        screen.blit(font.render("Catalog  (local records, newest first)", True, C_TXT),
                    (MARGIN + 130, 18))
        if self.status:
            col = C_ERR if self.status.lower().startswith(('cannot', 'no audio', 'replay unavail',
                                                             'replay differs')) else C_OK
            screen.blit(small.render(self.status[:70], True, col), (MARGIN + 420, 20))
        entries = self.cat['entries']
        if not entries:
            screen.blit(small.render("No records yet. Press Save in the live view.",
                                     True, C_DIM), (MARGIN, self.CAT_TOP + 6))
        for i, (rec, err) in enumerate(entries[:12]):
            rect = self._list_rect(i)
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
            line = (f"{rec.created.replace('T', ' ')}   {rec.seconds:.1f} s   "
                    f"A: {la}  B: {lb}   {tag}")
            screen.blit(small.render(line, True, C_OK if rec.status == 'pinned' else C_DIM),
                        (rect[0] + 8, rect[1] + 22))
        r = self._catalog_rects()
        labels = {'back': 'Back (Esc)', 'continue': 'Continue', 'open': 'Open field anew',
                  'play_A': 'Play A', 'play_B': 'Play B', 'play_mon': 'Play as heard',
                  'stop': 'Stop', 'replay': 'Check reproducibility', 'cancel': 'Cancel',
                  'play_replay': self.cat['play_label'], 'parent': '', 'version': '',
                  'pin': 'Pin to commit'}
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
                labels['version'] += "  |  source version runnable here"
        if rec is not None:
            ptitle, present = self.catalog.parent_of(rec)
            if ptitle:
                labels['parent'] = f"Derived from: {ptitle}" + ("" if present else " (missing)")
        for key, rect in r.items():
            if key in ('play_replay', 'parent') and not labels[key]:
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
            screen.blit(small.render("Note:", True, C_DIM), (px, y))
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
           caption=''):
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
    pygame.key.set_repeat(400, 35)     # held keys repeat (Backspace in the save form)
    app = BenchApp(runner.scene, engine, catalog=catalog)
    app.vol = runner.vol
    if origin_snapshot is not None:
        app.status = f"Continued: {runner.scene.title}  ({caption})"
    screen = pygame.display.set_mode((app.width, app.height))
    pygame.display.set_caption(f"CASynth demo bench - {runner.scene.title}"
                               + (f"  [{caption}]" if caption else ''))
    font = pygame.font.SysFont(FONT_NAMES, 17)
    small = pygame.font.SysFont(FONT_NAMES, 14)
    clock = pygame.time.Clock()
    try:
        alive = True
        while alive:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    alive = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE \
                        and app.mode == 'live':
                    alive = False
                elif ev.type == pygame.KEYDOWN:
                    app.key(pygame.key.name(ev.key), ctrl=bool(ev.mod & pygame.KMOD_CTRL))
                elif ev.type == pygame.TEXTINPUT:
                    app.text_input(ev.text)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    app.press(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEMOTION and (ev.buttons[0] or ev.buttons[2]):
                    app.drag(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    app.release()
            app.draw(screen, font, small)
            pygame.display.flip()
            clock.tick(60)
    finally:
        engine.stop()
        pygame.quit()
        if app.child is not None:
            app.child.terminate()
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
                  parent_record_id=rec.id, caption=caption)


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
    return run_ui(scene, a.vol, catalog=catalog)


if __name__ == '__main__':
    sys.exit(main())
