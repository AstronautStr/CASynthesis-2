#!/usr/bin/env python3
"""
mapping_bench.py — Batch listening stand for shape→timbre mapping experiments.

Matrix: oscillator (row) × mapping algorithm (col).
Click a cell to hear that combination loop. All cells animate simultaneously.
Space: pause/resume GoL animations. Esc: quit.

Mapping contract:  map_*(patch: ndarray[8,8], f0: float) -> (freqs: ndarray[K_MAX], amps: ndarray[K_MAX])
  - freqs in Hz; harmonic mappings return f0 * np.arange(1, K_MAX+1)
  - Laplacian returns inharmonic freqs derived from sqrt(eigenvalue)

Run:
    python mapping_bench.py
"""

import os
import numpy as np
import pygame

from gol_life_synth import render_chunk, step as _gol_step
from gol_life_synth import SR, K_MAX
from casynth_core import (PATCH_SIZE, extract,
                          map_fft2d, map_walsh, map_random,
                          map_laplacian, map_granulo)

# ── Config ──────────────────────────────────────────────────────────────────
GOL_HZ       = 3.0    # oscillator speed (generations/sec)
CARRIER_MIDI = 48     # C3 = 261 Hz
GRID_SZ      = 20     # simulation grid (avoids edge effects)
WINDOW       = PATCH_SIZE  # shape extraction window (alias for drawing/preview code)
FPS          = 30

CELL_W, CELL_H = 180, 140  # matrix cell (pixels)
LABEL_W        = 100       # row-label column
HEADER_H       = 40        # column-header row
STATUS_H       = 22        # status bar at bottom

PAT_PX = 8    # pixels per GoL cell in preview
BAR_H  = 26   # bar chart height inside cell

C_BG       = (12,  14,  18)
C_PANEL    = (22,  24,  30)
C_EDGE     = (40,  44,  52)
C_TXT      = (200, 206, 214)
C_DIM      = (100, 106, 114)
C_ACCENT   = (111, 208, 224)
C_LIVE     = (90,  180,  90)
C_DEAD     = (28,  32,  40)
C_SEL      = (111, 208, 224)
C_BAR      = (80,  150, 200)

def _midi_to_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)

# ── Oscillator definitions ──────────────────────────────────────────────────
# Clock_C: 6-cell period-2 oscillator, 1 connected component (8-connectivity).
# Verified by verify_clock.py. The 12-cell coordinates from the original spec
# turned out to be disconnected and non-oscillating — see questions.md P0-4.
_OSC_DEFS = [
    ("Blinker",
     [(0,0),(0,1),(0,2)]),
    ("Toad",
     [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]),
    ("Beacon",
     [(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)]),
    ("Clock",
     [(0,2),(1,0),(1,1),(2,2),(2,3),(3,1)]),  # Clock_C: period=2, 1 component
]
N_OSC = len(_OSC_DEFS)

def _make_grid(cells):
    g = np.zeros((GRID_SZ, GRID_SZ), np.uint8)
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    r0 = (GRID_SZ - (max(rs)-min(rs)+1)) // 2 - min(rs)
    c0 = (GRID_SZ - (max(cs)-min(cs)+1)) // 2 - min(cs)
    for r, c in cells:
        g[r0+r, c0+c] = 1
    return g

# Mapping algorithms and extract() live in casynth_core (single source of truth,
# imported above).  This bench passes n=K_MAX to each map_* (see _MAPPINGS calls).

_MAPPINGS = [
    ("2D-FFT",  map_fft2d),
    ("Walsh",   map_walsh),
    ("Random",  map_random),
    ("Laplace", map_laplacian),
    ("Granulo", map_granulo),
]
N_MAP = len(_MAPPINGS)

# ── Layout helpers ──────────────────────────────────────────────────────────

def _cell_rect(row, col):
    return pygame.Rect(LABEL_W + col*CELL_W, HEADER_H + row*CELL_H, CELL_W, CELL_H)

# ── Drawing ─────────────────────────────────────────────────────────────────

def _draw_cell(surf, row, col, patch, amp_vec, selected, small):
    r = _cell_rect(row, col)
    pygame.draw.rect(surf, C_PANEL, r)
    is_sel = (row, col) == selected
    pygame.draw.rect(surf, C_SEL if is_sel else C_EDGE, r, 2 if is_sel else 1)

    # 8×8 pattern preview
    pw = WINDOW * PAT_PX
    px_off = (CELL_W - pw) // 2
    py_off = 10
    for pr in range(WINDOW):
        for pc in range(WINDOW):
            col_px = C_LIVE if patch[pr, pc] else C_DEAD
            pygame.draw.rect(surf, col_px,
                             (r.left + px_off + pc*PAT_PX,
                              r.top  + py_off + pr*PAT_PX,
                              PAT_PX-1, PAT_PX-1))

    # Amplitude bar chart
    bar_top = r.top + py_off + pw + 8
    bw_total = CELL_W - 16
    bar_bw = max(1, bw_total // K_MAX)
    for k in range(K_MAX):
        h_bar = int(amp_vec[k] * BAR_H)
        bx = r.left + 8 + k * bar_bw
        by = bar_top + BAR_H - h_bar
        if h_bar:
            pygame.draw.rect(surf, C_BAR, (bx, by, max(1, bar_bw-1), h_bar))

    if is_sel:
        lbl = small.render("▶ playing", True, C_ACCENT)
        surf.blit(lbl, (r.left + 4, r.bottom - lbl.get_height() - 3))


def main():
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

    W = LABEL_W + N_MAP * CELL_W
    H = HEADER_H + N_OSC * CELL_H + STATUS_H

    pygame.init()

    audio_ok = True
    try:
        pygame.mixer.quit()
        try:
            pygame.mixer.init(SR, -16, 2, 512, allowedchanges=0)
        except TypeError:
            pygame.mixer.init(SR, -16, 2, 512)
        pygame.mixer.set_num_channels(4)
    except Exception as e:
        audio_ok = False
        print(f"[audio disabled: {e}]")

    MIX_CH = (pygame.mixer.get_init() or (0, 0, 2))[2]
    chan = (pygame.mixer.Channel(0) if audio_ok else None)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Mapping Bench — shape→timbre")
    font  = pygame.font.SysFont("consolas,menlo,monospace", 15)
    small = pygame.font.SysFont("consolas,menlo,monospace", 11)
    clock = pygame.time.Clock()

    # Simulation state: one 20×20 grid per oscillator
    grids   = [_make_grid(cells) for _, cells in _OSC_DEFS]
    patches = [extract(g, PATCH_SIZE) for g in grids]
    f0      = _midi_to_freq(CARRIER_MIDI)

    # fa[i][j] = (freqs_arr, amps_arr) for oscillator i, mapping j
    fa = [[mfn(patches[i], f0, K_MAX) for _, mfn in _MAPPINGS] for i in range(N_OSC)]

    selected = (0, 0)
    paused   = False
    acc      = 0.0

    # Audio state
    phase    = np.zeros(K_MAX + 1)
    amp_cur  = np.zeros(K_MAX + 1)
    pan_cur  = np.full(K_MAX + 1, 0.5)
    pan_tgt  = np.full(K_MAX + 1, 0.5)
    freq_cur = np.zeros(K_MAX + 1)   # tracks previous slot frequencies for phase-reset

    def _freq_tgt():
        row, col = selected
        f = np.zeros(K_MAX + 1)
        f[1:K_MAX+1] = fa[row][col][0][:K_MAX]   # freqs from mapping
        return f

    def _amp_tgt():
        row, col = selected
        a = np.zeros(K_MAX + 1)
        a[1:K_MAX+1] = fa[row][col][1][:K_MAX]   # amps from mapping
        return a

    def _feed_audio():
        if not audio_ok:
            return
        tgt_a = _amp_tgt()
        tgt_f = _freq_tgt()
        try:
            if not chan.get_busy():
                chan.play(pygame.sndarray.make_sound(
                    render_chunk(phase, amp_cur, pan_cur, tgt_a, pan_tgt, tgt_f, MIX_CH,
                                 freq_cur=freq_cur)))
            if chan.get_queue() is None:
                chan.queue(pygame.sndarray.make_sound(
                    render_chunk(phase, amp_cur, pan_cur, tgt_a, pan_tgt, tgt_f, MIX_CH,
                                 freq_cur=freq_cur)))
        except Exception as ex:
            print(f"[audio error] {ex!r}")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                for row in range(N_OSC):
                    for col in range(N_MAP):
                        if _cell_rect(row, col).collidepoint(mx, my):
                            selected = (row, col)

        if not paused:
            acc += dt
            interval = 1.0 / GOL_HZ
            while acc >= interval:
                acc -= interval
                for i in range(N_OSC):
                    grids[i]   = _gol_step(grids[i])
                    patches[i] = extract(grids[i], PATCH_SIZE)
                    fa[i]      = [mfn(patches[i], f0, K_MAX) for _, mfn in _MAPPINGS]

        _feed_audio()

        # ── Draw ────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Column headers
        for col, (mname, _) in enumerate(_MAPPINGS):
            x = LABEL_W + col * CELL_W
            pygame.draw.rect(screen, C_PANEL, (x, 0, CELL_W, HEADER_H))
            pygame.draw.rect(screen, C_EDGE,  (x, 0, CELL_W, HEADER_H), 1)
            lbl = font.render(mname, True, C_ACCENT)
            screen.blit(lbl, (x + (CELL_W - lbl.get_width())//2,
                               (HEADER_H - font.get_height())//2))

        # Row labels
        for row, (oname, _) in enumerate(_OSC_DEFS):
            y = HEADER_H + row * CELL_H
            pygame.draw.rect(screen, C_PANEL, (0, y, LABEL_W, CELL_H))
            pygame.draw.rect(screen, C_EDGE,  (0, y, LABEL_W, CELL_H), 1)
            lines = oname.split('\n')
            lh = font.get_height()
            ty = y + (CELL_H - len(lines)*lh) // 2
            for li, line in enumerate(lines):
                lbl = font.render(line, True, C_TXT)
                screen.blit(lbl, (8, ty + li*lh))

        # Matrix cells
        for row in range(N_OSC):
            for col in range(N_MAP):
                _draw_cell(screen, row, col,
                           patches[row], fa[row][col][1],
                           selected, small)

        # Status bar
        oname, _ = _OSC_DEFS[selected[0]]
        mname, _ = _MAPPINGS[selected[1]]
        state_str = "PAUSED" if paused else "RUNNING"
        st = (f"▶ {oname.replace(chr(10),' ')} / {mname}   "
              f"[Space] pause   [Esc] quit   C3 {f0:.0f}Hz   {state_str}")
        screen.blit(small.render(st, True, C_DIM),
                    (4, H - STATUS_H + (STATUS_H - small.get_height())//2))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
