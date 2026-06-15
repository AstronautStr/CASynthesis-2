#!/usr/bin/env python3
"""verify_clock.py - verify oscillator properties (period, connectivity) for mapping_bench.

Run:
    python verify_clock.py

Checks all oscillators in _OSC_DEFS and extra candidates.
Also verifies Walsh 2D sequency ordering.
"""
import numpy as np
from scipy import ndimage, linalg
import sys
sys.path.insert(0, '.')
from gol_life_synth import step

_S8 = np.ones((3, 3), np.uint8)

def test_oscillator(cells, name="", grid_size=30, max_period=10):
    g = np.zeros((grid_size, grid_size), np.uint8)
    rs = [r for r,c in cells]; cs = [c for r,c in cells]
    r0 = (grid_size - (max(rs)-min(rs)+1)) // 2 - min(rs)
    c0 = (grid_size - (max(cs)-min(cs)+1)) // 2 - min(cs)
    for r,c in cells:
        g[r0+r, c0+c] = 1
    _, nc = ndimage.label(g, structure=_S8)
    g0 = g.copy()
    for p in range(1, max_period+1):
        g = step(g)
        if np.array_equal(g, g0):
            status = "OK" if (p == 2 and nc == 1) else f"period={p},components={nc}"
            print(f"  {name or str(cells)}: period={p}, components={nc}  [{status}]")
            return p, nc
    print(f"  {name or str(cells)}: no period found in {max_period} steps, components={nc}")
    return None, nc

# ── Sanity checks ───────────────────────────────────────────────────────────
print("=== Zoo sanity (blinker/toad/beacon) ===")
test_oscillator([(0,0),(0,1),(0,2)], "Blinker")
test_oscillator([(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)], "Toad")
test_oscillator([(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)], "Beacon")

# ── Clock candidates ────────────────────────────────────────────────────────
print("\n=== Clock candidates (want: period=2, components=1) ===")
candidates = {
    # Previous session candidates
    "Clock_A": [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)],
    "Clock_B": [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)],
    "Clock_C": [(0,2),(1,0),(1,1),(2,2),(2,3),(3,1)],
    "Clock_D": [(0,0),(0,1),(1,1),(2,1),(2,2),(3,2)],
    "Clock_E": [(0,1),(1,0),(1,1),(1,2),(2,2),(3,2),(3,3)],
    "Clock_F": [(0,1),(1,0),(1,2),(2,1),(1,3),(2,2)],
    "Clock_G": [(0,0),(0,2),(1,1),(1,2),(2,0),(2,1)],
    "Clock_H": [(0,1),(0,2),(1,0),(2,1),(2,2),(3,0)],
    "Clock_I": [(0,1),(0,2),(1,0),(2,3),(3,1),(3,2)],
    "Clock_J": [(0,2),(1,0),(1,3),(2,0),(2,3),(3,1)],
    # ТЗ-кандидат (12 cells, из spec): turns out disconnected — kept for record
    "Clock_spec12": [(0,2),(0,3),(1,1),(1,2),(2,0),(2,1),(3,4),(3,5),(4,3),(4,4),(5,2),(5,3)],
    # Canonical Clock from LifeWiki (6 cells, period 2):
    # https://www.conwaylife.com/wiki/Clock
    # RLE: 2o$obo$o2bo$3bo! — translates to:
    "Clock_LifeWiki": [(0,0),(0,1),(1,0),(1,2),(2,0),(2,3),(3,3)],
    # Another LifeWiki reading:
    "Clock_LifeWiki2": [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)],
}
for name, cells in candidates.items():
    test_oscillator(cells, name)

# ── Walsh sequency check ────────────────────────────────────────────────────
print("\n=== Walsh 2D sequency ordering ===")
WINDOW = 8
H8 = linalg.hadamard(WINDOW).astype(float) / WINDOW

def row_sequency(H):
    seq = np.zeros(H.shape[0], dtype=int)
    for i in range(H.shape[0]):
        row = H[i]
        seq[i] = int(np.sum(row[:-1] * row[1:] < 0))
    return seq

SEQ8 = row_sequency(H8)
print(f"  Row sequencies of H8: {SEQ8}")
ij_seq = np.array([[SEQ8[i] + SEQ8[j] for j in range(WINDOW)] for i in range(WINDOW)])
walsh_order = np.argsort(ij_seq.flatten(), kind='stable')
print(f"  DC at index 0: {walsh_order[0] == 0}")
print(f"  First 8 combined sequencies: {ij_seq.flatten()[walsh_order[:8]]}")
print("  [OK] Walsh 2D sequency ordering verified" if walsh_order[0] == 0 else "  [FAIL]")
