#!/usr/bin/env python3
"""One-shot diagnostic: how distinct are the amplitude vectors across
oscillators and mappings? Researcher tool, not part of the engine."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=140)

import mapping_bench as mb

f0 = mb._midi_to_freq(mb.CARRIER_MIDI)

# Collect both phases of each period-2 oscillator.
def phases_of(cells, n=4):
    g = mb._make_grid(cells)
    out = []
    for _ in range(n):
        out.append(mb.extract(g, mb.PATCH_SIZE))
        g = mb._gol_step(g)
    return out

mappings = mb._MAPPINGS  # [(name, fn), ...]

print("=" * 100)
print("AMPLITUDE VECTORS per oscillator (phase 0) per mapping")
print("=" * 100)
osc_vectors = {}  # name -> {mapname -> [amp vectors over phases]}
for oname, cells in mb._OSC_DEFS:
    patches = phases_of(cells)
    osc_vectors[oname] = {}
    print(f"\n### {oname}  ({len(patches)} sampled phases)")
    for mname, mfn in mappings:
        vecs = []
        for p in patches:
            freqs, amps = mfn(p, f0)
            vecs.append(amps.copy())
        osc_vectors[oname][mname] = vecs
        print(f"  {mname:8s} phase0 amps: {vecs[0]}")

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float('nan')
    return float(np.dot(a, b) / (na * nb))

print("\n" + "=" * 100)
print("PAIRWISE COSINE SIMILARITY between oscillators (phase0), per mapping")
print("  1.00 = identical direction (indistinguishable spectral shape)")
print("=" * 100)
onames = [o for o, _ in mb._OSC_DEFS]
for mname, _ in mappings:
    print(f"\n### {mname}")
    print("          " + "".join(f"{o[:7]:>9s}" for o in onames))
    for oi in onames:
        row = []
        for oj in onames:
            row.append(cos(osc_vectors[oi][mname][0],
                           osc_vectors[oj][mname][0]))
        print(f"  {oi[:7]:>7s} " + "".join(f"{v:9.2f}" for v in row))

print("\n" + "=" * 100)
print("WITHIN-OSCILLATOR phase-to-phase cosine (how much the timbre MOVES")
print("  over time for a single oscillator; ~1.00 = static timbre, motionless)")
print("=" * 100)
for mname, _ in mappings:
    print(f"\n### {mname}")
    for oname in onames:
        vecs = osc_vectors[oname][mname]
        sims = [cos(vecs[0], vecs[t]) for t in range(1, len(vecs))]
        sims = [s for s in sims if not np.isnan(s)]
        smin = min(sims) if sims else float('nan')
        print(f"  {oname:8s} phase-sims vs phase0: "
              + " ".join(f"{s:.2f}" for s in sims)
              + f"   (min {smin:.2f})")

print("\n" + "=" * 100)
print("ENERGY CONCENTRATION: fraction of total amplitude in top-3 partials")
print("  (high => few dominant partials => sparse/pure; low => spread/noisy)")
print("=" * 100)
for mname, _ in mappings:
    print(f"\n### {mname}")
    for oname in onames:
        v = np.sort(osc_vectors[oname][mname][0])[::-1]
        tot = v.sum()
        frac = v[:3].sum() / (tot + 1e-9)
        nnz = int((v > 0.05).sum())
        print(f"  {oname:8s} top3-frac={frac:.2f}  nnz(>0.05)={nnz}")
