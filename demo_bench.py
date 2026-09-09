#!/usr/bin/env python3
"""Demo bench (S1 + S2): one live A/B demo window, or an offline render.

    python demo_bench.py --demo demos/laplace_ab.json
    python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor

Window: shared field, demo title + listening hint, transport (Start|Stop /
Pause CA / Restart), volume, generation, audio state; a side panel with the
A/B tabs (select = listen + edit), the selected side's engine and its registry
knobs, and a one-line summary of how A and B differ.  LMB paints, RMB erases.
Startup is silent and frozen; Start launches automaton + sound (and turns into
Stop = full stop, initial scene); Pause CA freezes only the automaton; Restart
starts again immediately.  Stop/Restart keep engines, params, volume and side.

All computation lives in casynth_lab (DemoRunner); this file is UI + CLI only.
The offline path imports neither pygame nor sounddevice.
stdout stays ASCII (Windows console codepage).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casynth_config import (VOL_DEFAULT, C_BG, C_GRID, C_PANEL, C_EDGE, C_TXT,
                            C_DIM, C_BTN, C_ACCENT)                    # noqa: E402
from casynth_lab import (load_scene, SceneError, DemoRunner, SIDES, registry,
                         describe_difference, validate_param)           # noqa: E402

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


def _is_toggle(spec):
    _arg, _label, lo, hi, integer, _d = spec
    return bool(integer) and lo == 0 and hi == 1


class BenchApp:
    """Pure UI state machine over a LiveEngine -- drivable without a display
    (tests call press/drag/release/draw with SDL_VIDEODRIVER=dummy)."""

    def __init__(self, scene, engine):
        self.scene = scene
        self.engine = engine
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
        for key, label in (('start', 'Start'), ('pause', 'Pause CA (Space)'), ('reset', 'Restart (R)')):
            self.buttons[key] = ((x, BTN_Y, BTN_W, BTN_H), label)
            x += BTN_W + 10
        self.vol_rect = (MARGIN + 92, BTN_Y + BTN_H + 14, VOL_W, 10)
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
        if button == 1:
            for key, (rect, _label) in self.buttons.items():
                if self._inside(rect, pos):
                    if key == 'start' and self.engine.snapshot()['running']:
                        key = 'stop'
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

    def key(self, name):
        """Keyboard hotkey by key name ('r' = Restart).  Returns the command or None."""
        name = name.lower()
        if name == 'r':
            self._post('reset')
            return 'reset'
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
        snap = self.engine.snapshot()
        screen.fill(C_BG)
        pygame.draw.rect(screen, C_PANEL, (0, 0, self.width, TOP_H))
        screen.blit(font.render(self.scene.title, True, C_TXT), (MARGIN, 10))
        if self.scene.listen:
            screen.blit(small.render(self.scene.listen, True, C_DIM), (MARGIN, 34))
        for key, (rect, label) in self.buttons.items():
            on = ((key == 'start' and snap['running']) or
                  (key == 'pause' and snap['paused']))
            if key == 'start' and snap['running']:
                label = 'Stop'
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


def run_ui(scene, vol):
    import pygame
    from casynth_lab.audio_out import LiveEngine
    runner = DemoRunner(scene, vol=vol)
    engine = LiveEngine(runner)
    engine.start()
    print(f"[audio] {engine.status_text()}")
    pygame.init()
    app = BenchApp(scene, engine)
    app.vol = vol
    screen = pygame.display.set_mode((app.width, app.height))
    pygame.display.set_caption(f"CASynth demo bench - {scene.title}")
    font = pygame.font.SysFont(FONT_NAMES, 17)
    small = pygame.font.SysFont(FONT_NAMES, 14)
    clock = pygame.time.Clock()
    try:
        alive = True
        while alive:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    alive = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    alive = False
                elif ev.type == pygame.KEYDOWN:
                    app.key(pygame.key.name(ev.key))
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
    return 0


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
    ap.add_argument('--demo', required=True, help="scene JSON (demos/*.json)")
    ap.add_argument('--render', metavar='WAV', help="offline render to WAV (no window)")
    ap.add_argument('--seconds', type=float, default=8.0, help="render length (s)")
    ap.add_argument('--side', choices=('A', 'B', 'monitor'), default='A',
                    help="offline output: raw side A/B or the monitor mix")
    ap.add_argument('--vol', type=float, default=VOL_DEFAULT, help="initial volume 0..1")
    a = ap.parse_args(argv)
    try:
        scene = load_scene(a.demo)
    except SceneError as e:
        print(f"error: {e}")
        return 2
    if a.render:
        return run_render(scene, a.render, a.seconds, a.vol, a.side)
    return run_ui(scene, a.vol)


if __name__ == '__main__':
    sys.exit(main())
