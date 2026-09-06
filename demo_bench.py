#!/usr/bin/env python3
"""Demo bench (S1): one live Laplace demo window, or an offline render.

    python demo_bench.py --demo demos/laplace_basic.json
    python demo_bench.py --demo demos/laplace_basic.json --render artifacts/s1.wav --seconds 8

Window: field, demo title, three buttons (Start / Pause CA / Restart), volume,
generation counter, audio-output state.  LMB paints, RMB erases.  Startup is
silent and frozen; Start launches both the automaton and the sound; Pause CA
freezes only the automaton (sound keeps running, edits still apply); Restart
resets field + clocks + every audio tail and starts the scene again.

All computation lives in casynth_lab (DemoRunner); this file is UI + CLI only.
The offline path imports neither pygame nor sounddevice.
stdout stays ASCII (Windows console codepage).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casynth_config import (VOL_DEFAULT, C_BG, C_GRID, C_PANEL, C_EDGE, C_TXT,
                            C_DIM, C_BTN, C_ACCENT)   # noqa: E402
from casynth_lab import load_scene, SceneError, DemoRunner      # noqa: E402

# -- layout ------------------------------------------------------------------
CELL = 16
MARGIN = 16
TOP_H = 136
BTN_W, BTN_H = 130, 32
BTN_Y = 40
VOL_W = 140
FONT_NAMES = "segoeui,arial,dejavusans,freesans"
C_ALIVE = (111, 208, 224)
C_ALIVE_PAUSED = (200, 180, 90)
C_BTN_ON = (48, 92, 104)


class BenchApp:
    """Pure UI state machine over a LiveEngine -- drivable without a display
    (tests call press/drag/draw with SDL_VIDEODRIVER=dummy)."""

    def __init__(self, scene, engine):
        self.scene = scene
        self.engine = engine
        self.vol = VOL_DEFAULT
        self.paint_value = None          # None / 1 (LMB) / 0 (RMB) while dragging
        self.drag_vol = False
        self.field_w = scene.cols * CELL
        self.field_h = scene.rows * CELL
        self.width = max(self.field_w + 2 * MARGIN, 3 * (BTN_W + 10) + 2 * MARGIN)
        self.height = TOP_H + self.field_h + MARGIN
        self.field_x = (self.width - self.field_w) // 2
        self.field_y = TOP_H
        self.buttons = {}
        x = MARGIN
        for key, label in (('start', 'Start'), ('pause', 'Pause CA'), ('reset', 'Restart')):
            self.buttons[key] = ((x, BTN_Y, BTN_W, BTN_H), label)
            x += BTN_W + 10
        self.vol_rect = (MARGIN + 92, BTN_Y + BTN_H + 14, VOL_W, 10)

    # -- geometry ------------------------------------------------------------
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

    # -- input ---------------------------------------------------------------
    def press(self, pos, button):
        """Mouse button down. button: 1 = left, 3 = right."""
        if button == 1:
            for key, (rect, _label) in self.buttons.items():
                if self._inside(rect, pos):
                    self.engine.post(key)
                    return key
            vx, vy, vw, vh = self.vol_rect
            if vx - 6 <= pos[0] < vx + vw + 6 and vy - 8 <= pos[1] < vy + vh + 8:
                self.drag_vol = True
                self._set_vol(pos[0])
                return 'vol'
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
        elif self.paint_value is not None:
            cell = self.cell_at(pos)
            if cell is not None:
                self._paint(cell)

    def release(self):
        self.paint_value = None
        self.drag_vol = False

    def _paint(self, cell):
        self.engine.post('set_cell', r=cell[0], c=cell[1], v=self.paint_value)

    def _set_vol(self, mx):
        vx, _vy, vw, _vh = self.vol_rect
        self.vol = min(max((mx - vx) / vw, 0.0), 1.0)
        self.engine.post('vol', value=self.vol)

    # -- drawing -------------------------------------------------------------
    def draw(self, screen, font, small):
        import pygame
        snap = self.engine.snapshot()
        screen.fill(C_BG)
        pygame.draw.rect(screen, C_PANEL, (0, 0, self.width, TOP_H))
        screen.blit(font.render(self.scene.title, True, C_TXT), (MARGIN, 10))
        for key, (rect, label) in self.buttons.items():
            on = ((key == 'start' and snap['running']) or
                  (key == 'pause' and snap['paused']))
            pygame.draw.rect(screen, C_BTN_ON if on else C_BTN, rect, border_radius=4)
            pygame.draw.rect(screen, C_EDGE, rect, 1, border_radius=4)
            t = font.render(label, True, C_TXT)
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2,
                            rect[1] + (rect[3] - t.get_height()) // 2))
        # volume
        vx, vy, vw, vh = self.vol_rect
        screen.blit(small.render("Volume", True, C_DIM), (MARGIN, vy - 4))
        pygame.draw.rect(screen, C_EDGE, (vx, vy, vw, vh), border_radius=3)
        pygame.draw.rect(screen, C_ACCENT, (vx, vy, int(vw * self.vol), vh), border_radius=3)
        screen.blit(small.render(f"{int(self.vol * 100):3d}%", True, C_TXT), (vx + vw + 8, vy - 4))
        # generation + clock + audio state
        gx = vx + vw + 60
        screen.blit(small.render(f"Gen {snap['gen']}   t = {snap['t_seconds']:.2f} s",
                                 True, C_TXT), (gx, vy - 4))
        st = self.engine.status_text()
        col = C_TXT if self.engine.device_ok else (230, 120, 90)
        if self.engine.device_ok and self.engine.underruns:
            col = (230, 180, 90)
        screen.blit(small.render(f"Audio: {st}", True, col), (MARGIN, TOP_H - 28))
        # field
        fx, fy = self.field_x, self.field_y
        pygame.draw.rect(screen, C_GRID, (fx - 1, fy - 1, self.field_w + 2, self.field_h + 2), 1)
        alive_col = C_ALIVE_PAUSED if snap['paused'] else C_ALIVE
        grid = snap['grid']
        for r in range(self.scene.rows):
            for c in range(self.scene.cols):
                rect = (fx + c * CELL, fy + r * CELL, CELL - 1, CELL - 1)
                pygame.draw.rect(screen, alive_col if grid[r, c] else C_GRID, rect)


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


def run_render(scene, out_path, seconds, vol):
    from casynth_lab import render_offline, write_wav
    if seconds <= 0:
        print("error: --seconds must be > 0")
        return 2
    pcm, runner = render_offline(scene, seconds, vol=vol)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    write_wav(out_path, pcm)
    peak = float(abs(pcm).max()) / 32767.0 if len(pcm) else 0.0
    print(f"[render] {out_path}: {len(pcm)} samples x {pcm.shape[1]} ch, "
          f"{seconds:g} s, gen {runner.gen}, peak {peak:.3f}, "
          f"clip blocks {runner.clip_blocks}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="CASynth demo bench (S1)")
    ap.add_argument('--demo', required=True, help="scene JSON (demos/*.json)")
    ap.add_argument('--render', metavar='WAV', help="offline render to WAV (no window)")
    ap.add_argument('--seconds', type=float, default=8.0, help="render length (s)")
    ap.add_argument('--vol', type=float, default=VOL_DEFAULT, help="initial volume 0..1")
    a = ap.parse_args(argv)
    try:
        scene = load_scene(a.demo)
    except SceneError as e:
        print(f"error: {e}")
        return 2
    if a.render:
        return run_render(scene, a.render, a.seconds, a.vol)
    return run_ui(scene, a.vol)


if __name__ == '__main__':
    sys.exit(main())
