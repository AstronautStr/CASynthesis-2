#!/usr/bin/env python3
"""
gol_life_synth.py - Game of Life as a CONTINUOUS additive synthesizer.

The field is segmented into OBJECTS (8-connected components). Every object is a
HARMONIC of one carrier note you play on the on-screen piano:

    size (area)   -> harmonic number k   (big object = low harmonic, near f0;
                                          small object = high harmonic)
    density       -> that harmonic's amplitude
    centroid x    -> that harmonic's stereo pan

The audio is CONTINUOUS: harmonic phases run unbroken across audio chunks and
amplitudes glide smoothly toward their targets, so the loudness stays constant
while the automaton reshapes the timbre (no per-generation pulsing). The whole
field = the evolving SPECTRUM of the held note. Empty field = silence.

Cells are coloured by their harmonic (low=red .. high=blue), brightness=amplitude.
Grey = objects beyond the voice cap (not sounded).

CONTROLS
  Mouse on field : left-drag draw, right-drag erase (any time)
  Piano (bottom) : click a key to set the carrier note (latched)
  Volume slider  : drag (top-right of the toolbar)
  Play / Pause ... run or freeze the automaton            [Space]
  Step ........... one generation                         [S]
  Random / Clear . fill / empty                           [R] / [C]
  - / + .......... slower / faster                        [Down] / [Up]
  quit ....................................... [Esc] / window close

RUN
    pip install pygame-ce numpy scipy
    python gol_life_synth.py
"""

import os
import colorsys
import numpy as np
import pygame
from scipy import ndimage

# ----------------------------------------------------------------------
# CONFIG  -- the tunable feature->parameter table (tweak by ear)
# ----------------------------------------------------------------------
GRID_W, GRID_H = 52, 30
CELL = 20
TOOLBAR_H = 70
PIANO_H = 96
FPS = 60

SR = 44100
STEP_HZ = 6.0                  # generations per second (start value)
RANDOM_DENSITY = 0.28
MAX_VOICES = 24

CHUNK_S = 0.09                 # audio render chunk (continuous streaming)
MASTER_GAIN = 0.25
VOL_DEFAULT = 0.70
VOL_W = 120

# size (area) -> harmonic number k in [1 .. K_MAX]
K_MAX = 20
HARM_AREA_MIN, HARM_AREA_MAX = 1, 18

# density -> harmonic amplitude;  1/k^ROLLOFF tames upper partials
AMP_FLOOR = 0.30
ROLLOFF = 0.70

# piano range (MIDI): C3 .. C5
NOTE_LO, NOTE_HI, NOTE_DEFAULT = 48, 72, 48

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


def midi_to_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def note_name(n):
    return f"{NAMES[n % 12]}{n // 12 - 1}"


# ----------------------------------------------------------------------
# GAME OF LIFE  (toroidal)
# ----------------------------------------------------------------------
_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_S8 = np.ones((3, 3), np.uint8)


def step(grid):
    n = ndimage.convolve(grid, _NEIGH, mode='wrap')
    return ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)


# ----------------------------------------------------------------------
# OBJECTS -> HARMONICS (+ colour encoding the harmonic)
# ----------------------------------------------------------------------
def analyse(grid):
    labels, n = ndimage.label(grid, structure=_S8)
    if n == 0:
        return labels, [], {}
    objs = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        objs.append((lab, len(xs), ys, xs))
    objs.sort(key=lambda o: -o[1])

    voices, color = [], {}
    for (lab, area, ys, xs) in objs[:MAX_VOICES]:
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        density = area / (h * w)
        cx = xs.mean() / (GRID_W - 1)

        frac = float(np.clip((area - HARM_AREA_MIN) / (HARM_AREA_MAX - HARM_AREA_MIN), 0, 1))
        k = 1 + int(round((1.0 - frac) * (K_MAX - 1)))
        amp = (AMP_FLOOR + (1 - AMP_FLOOR) * density) / (k ** ROLLOFF)
        voices.append(dict(k=k, amp=amp, pan=float(np.clip(cx, 0, 1))))

        hue = ((k - 1) / (K_MAX - 1)) * 0.72
        val = float(np.clip(0.3 + 0.7 * density, 0.3, 1.0))
        color[lab] = hsv(hue, val)
    for (lab, area, ys, xs) in objs[MAX_VOICES:]:
        color[lab] = C_CAPPED
    return labels, voices, color


def hsv(h, v):
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def voices_to_targets(voices):
    """Aggregate object voices into per-harmonic target amp/pan arrays."""
    amp_raw = np.zeros(K_MAX + 1)
    pan_w = np.zeros(K_MAX + 1)
    for v in voices:
        k = v['k']
        amp_raw[k] += v['amp']
        pan_w[k] += v['amp'] * v['pan']
    pan = np.where(amp_raw > 1e-9, pan_w / np.maximum(amp_raw, 1e-9), 0.5)
    amp = np.clip(amp_raw, 0.0, 1.0)
    return amp, pan


# ----------------------------------------------------------------------
# CONTINUOUS SYNTH  -- phase-continuous, amplitude-glided chunks
# ----------------------------------------------------------------------
def render_chunk(phase, amp_cur, pan_cur, amp_tgt, pan_tgt, f0, channels):
    """Render one gapless chunk. Mutates phase/amp_cur/pan_cur in place so the
    next chunk continues seamlessly. Returns an int16 buffer (n x channels)."""
    n = int(CHUNK_S * SR)
    L = np.zeros(n); R = np.zeros(n)
    idx = np.arange(n)
    guard = 0.45 * SR
    for k in range(1, K_MAX + 1):
        if amp_cur[k] < 1e-4 and amp_tgt[k] < 1e-4:
            continue
        freq = k * f0
        if freq >= guard:                         # anti-alias: silence too-high partials
            amp_cur[k] = 0.0
            continue
        inc = TWO_PI * freq / SR
        wave = np.sin(phase[k] + inc * idx)       # phase carries over -> no clicks
        a = np.linspace(amp_cur[k], amp_tgt[k], n)   # glide amplitude -> constant level
        p = np.linspace(pan_cur[k], pan_tgt[k], n)
        L += wave * a * np.cos(p * np.pi / 2)
        R += wave * a * np.sin(p * np.pi / 2)
        phase[k] = (phase[k] + inc * n) % TWO_PI
        amp_cur[k] = amp_tgt[k]
        pan_cur[k] = pan_tgt[k]
    L = np.clip(L * MASTER_GAIN, -1.0, 1.0)
    R = np.clip(R * MASTER_GAIN, -1.0, 1.0)
    if channels <= 1:
        data = ((L + R) * 0.5)[:, None]
    else:
        data = np.zeros((n, channels))
        data[:, 0] = L; data[:, 1] = R
    return np.ascontiguousarray((data * 32767).astype(np.int16))


# ----------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------
def main():
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    pygame.init()
    audio_ok = True
    try:
        pygame.mixer.quit()
        try:
            pygame.mixer.init(SR, -16, 2, 512, allowedchanges=0)   # force true stereo
        except TypeError:
            pygame.mixer.init(SR, -16, 2, 512)
        pygame.mixer.set_num_channels(16)
    except Exception as e:
        audio_ok = False
        print(f"[audio disabled: {e}] - visuals will still run")
    MIX_CH = (pygame.mixer.get_init() or (0, 0, 2))[2]

    W = GRID_W * CELL
    H = GRID_H * CELL + TOOLBAR_H + PIANO_H
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Game of Life - continuous synth")
    font = pygame.font.SysFont("consolas,menlo,monospace", 16)
    small = pygame.font.SysFont("consolas,menlo,monospace", 13)
    clock = pygame.time.Clock()

    grid = np.zeros((GRID_H, GRID_W), np.uint8)
    state = dict(run=False, hz=STEP_HZ, gen=0, acc=0.0, note=NOTE_DEFAULT, vol=VOL_DEFAULT,
                 phase=np.zeros(K_MAX + 1), amp=np.zeros(K_MAX + 1), pan=np.full(K_MAX + 1, 0.5),
                 chan=(pygame.mixer.Channel(0) if audio_ok else None), audio_err=False)
    if audio_ok:
        state['chan'].set_volume(state['vol'])

    # toolbar buttons
    by = GRID_H * CELL + 14
    defs = [("play", None, 96), ("step", "Step", 70), ("random", "Random", 96),
            ("clear", "Clear", 78), ("slower", "-", 40), ("faster", "+", 40)]
    buttons, bx = [], 12
    for bid, label, bw in defs:
        buttons.append(dict(id=bid, label=label, rect=pygame.Rect(bx, by, bw, 40)))
        bx += bw + 8
    legend_x = bx + 12
    vol_track = pygame.Rect(W - VOL_W - 24, by + 16, VOL_W, 10)

    # piano geometry
    piano_top = GRID_H * CELL + TOOLBAR_H
    white_notes = [m for m in range(NOTE_LO, NOTE_HI + 1) if m % 12 in WHITE_PC]
    wkw = W / len(white_notes)
    white_keys, white_index = [], {}
    for i, m in enumerate(white_notes):
        x0, x1 = round(i * wkw), round((i + 1) * wkw)
        white_keys.append((pygame.Rect(x0, piano_top, x1 - x0, PIANO_H), m))
        white_index[m] = i
    black_keys, bkw, bkh = [], wkw * 0.62, int(PIANO_H * 0.62)
    for m in range(NOTE_LO, NOTE_HI + 1):
        if m % 12 in BLACK_PC and (m - 1) in white_index:
            cx = (white_index[m - 1] + 1) * wkw
            black_keys.append((pygame.Rect(round(cx - bkw / 2), piano_top, round(bkw), bkh), m))

    paint = None
    dragging_vol = False

    def cell_at(mx, my):
        if 0 <= my < GRID_H * CELL and 0 <= mx < GRID_W * CELL:
            return my // CELL, mx // CELL
        return None

    def f0():
        return midi_to_freq(state['note'])

    def set_vol(mx):
        state['vol'] = float(np.clip((mx - vol_track.left) / vol_track.width, 0, 1))
        if audio_ok:
            state['chan'].set_volume(state['vol'])

    def do(bid):
        nonlocal grid
        if bid == "play":   state['run'] = not state['run']
        elif bid == "step": grid = step(grid); state['gen'] += 1
        elif bid == "random":
            grid[:] = (np.random.random((GRID_H, GRID_W)) < RANDOM_DENSITY).astype(np.uint8)
        elif bid == "clear": grid[:] = 0; state['gen'] = 0
        elif bid == "slower": state['hz'] = max(1.0, state['hz'] - 1.0)
        elif bid == "faster": state['hz'] = min(30.0, state['hz'] + 1.0)

    def hit_piano(pos):
        for rect, m in black_keys:
            if rect.collidepoint(pos):
                return m
        for rect, m in white_keys:
            if rect.collidepoint(pos):
                return m
        return None

    def feed_audio(amp_tgt, pan_tgt):
        if not audio_ok:
            return
        ch = state['chan']
        try:
            if not ch.get_busy():
                ch.play(pygame.sndarray.make_sound(
                    render_chunk(state['phase'], state['amp'], state['pan'], amp_tgt, pan_tgt, f0(), MIX_CH)))
            if ch.get_queue() is None:
                ch.queue(pygame.sndarray.make_sound(
                    render_chunk(state['phase'], state['amp'], state['pan'], amp_tgt, pan_tgt, f0(), MIX_CH)))
        except Exception as e:
            if not state['audio_err']:
                state['audio_err'] = True
                print(f"[audio error] {e!r}")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_SPACE: do("play")
                elif e.key == pygame.K_s: do("step")
                elif e.key == pygame.K_r: do("random")
                elif e.key == pygame.K_c: do("clear")
                elif e.key == pygame.K_UP: do("faster")
                elif e.key == pygame.K_DOWN: do("slower")
            elif e.type == pygame.MOUSEBUTTONDOWN:
                rc = cell_at(*e.pos)
                if rc is not None and e.button in (1, 3):
                    paint = 1 if e.button == 1 else 0
                    grid[rc] = paint
                elif e.pos[1] >= piano_top:
                    m = hit_piano(e.pos)
                    if m is not None:
                        state['note'] = m
                else:
                    if vol_track.collidepoint(e.pos):
                        dragging_vol = True; set_vol(e.pos[0])
                    else:
                        for b in buttons:
                            if b['rect'].collidepoint(e.pos):
                                do(b['id']); break
            elif e.type == pygame.MOUSEBUTTONUP:
                paint = None; dragging_vol = False
            elif e.type == pygame.MOUSEMOTION:
                if paint is not None:
                    rc = cell_at(*e.pos)
                    if rc is not None:
                        grid[rc] = paint
                elif dragging_vol:
                    set_vol(e.pos[0])

        if state['run']:
            interval = 1.0 / state['hz']
            state['acc'] += dt
            steps = 0
            while state['acc'] >= interval and steps < 4:
                state['acc'] -= interval
                grid = step(grid); state['gen'] += 1; steps += 1

        labels, voices, color = analyse(grid)
        amp_tgt, pan_tgt = voices_to_targets(voices)
        feed_audio(amp_tgt, pan_tgt)

        # ---- field ----
        screen.fill(C_BG)
        for x in range(GRID_W + 1):
            pygame.draw.line(screen, C_GRID, (x * CELL, 0), (x * CELL, GRID_H * CELL))
        for y in range(GRID_H + 1):
            pygame.draw.line(screen, C_GRID, (0, y * CELL), (GRID_W * CELL, y * CELL))
        ys, xs = np.where(grid == 1)
        for (r, c) in zip(ys.tolist(), xs.tolist()):
            col = color.get(labels[r, c], C_DIM)
            pygame.draw.rect(screen, col, (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2),
                             border_radius=4)

        # ---- toolbar ----
        pygame.draw.rect(screen, C_PANEL, (0, GRID_H * CELL, W, TOOLBAR_H))
        pygame.draw.line(screen, C_EDGE, (0, GRID_H * CELL), (W, GRID_H * CELL))
        mouse = pygame.mouse.get_pos()
        for b in buttons:
            hot = b['rect'].collidepoint(mouse)
            pygame.draw.rect(screen, C_BTN_HOT if hot else C_BTN, b['rect'], border_radius=6)
            label = b['label'] or ("Pause" if state['run'] else "Play")
            txt = font.render(label, True, C_ACCENT if b['id'] == 'play' else C_TXT)
            screen.blit(txt, txt.get_rect(center=b['rect'].center))

        lw, lh, lx, ly = 130, 14, legend_x, by + 4
        for i in range(lw):
            pygame.draw.line(screen, hsv((i / (lw - 1)) * 0.72, 0.95), (lx + i, ly), (lx + i, ly + lh))
        screen.blit(small.render("big / low", True, C_DIM), (lx, ly + lh + 3))
        hi = small.render("small / high", True, C_DIM)
        screen.blit(hi, (lx + lw - hi.get_width(), ly + lh + 3))

        sx = lx + lw + 22
        st = f"{note_name(state['note'])} {f0():.0f}Hz  gen {state['gen']:>4}  {state['hz']:.0f}/s"
        screen.blit(font.render(st, True, C_TXT), (sx, by + 2))
        mode = "RUNNING" if state['run'] else "PAUSED"
        screen.blit(font.render(mode, True, C_ACCENT if state['run'] else C_DIM), (sx, by + 24))

        # volume slider
        pygame.draw.rect(screen, C_BTN, vol_track, border_radius=5)
        fw = int(vol_track.width * state['vol'])
        pygame.draw.rect(screen, C_ACCENT, (vol_track.left, vol_track.top, fw, vol_track.height), border_radius=5)
        pygame.draw.circle(screen, C_TXT, (vol_track.left + fw, vol_track.centery), 7)
        screen.blit(small.render(f"vol {int(state['vol'] * 100)}%", True, C_DIM), (vol_track.left, by - 1))
        if not audio_ok:
            screen.blit(small.render("audio disabled", True, C_DIM), (W - 110, 6))

        # ---- piano ----
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

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
