#!/usr/bin/env python3
"""Decisive click test: isolate the envelope-corner click from carrier curvature.

Part of the `audio-artifact-probe` skill (.claude/skills/).

(A) Micro-test: one slot, one amplitude step then steady.  Measure the slope
    jump at the chunk boundary for a LINEAR ramp vs the SMOOTHSTEP ramp.  The
    carrier is identical, so the difference is purely the envelope click.

(B) Scene-test: compare per-boundary |d2| of the linear render (gainfix) vs the
    smoothstep render (clickfix) at the SAME boundaries.  Carrier curvature
    cancels; what remains is the envelope click removed by the fix.
"""
import os
import numpy as np
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import gol_life_synth_laplacian as L

n = int(L.CHUNK_S * L.SR)
FREQ = 261.6   # a carrier in-band


def render_two_chunks(ramp):
    """Two chunks for one slot: chunk1 amp 0.2->0.8 (a step), chunk2 steady 0.8.
    `ramp` is a length-n array 0->1.  Returns concatenated mono float signal."""
    phase = 0.0
    inc = L.TWO_PI * FREQ / L.SR
    idx = np.arange(n)
    out = []
    # chunk 1: amp_cur=0.2 -> amp_tgt=0.8
    for (amp_cur, amp_tgt) in [(0.2, 0.8), (0.8, 0.8)]:
        wave = np.sin(phase + inc * idx)
        a = amp_cur + (amp_tgt - amp_cur) * ramp
        out.append(wave * a)
        phase = (phase + inc * n) % L.TWO_PI
    return np.concatenate(out)


def env_two_chunks(ramp):
    """Envelope alone (no carrier): step 0.2->0.8 then steady 0.8."""
    out = []
    for (amp_cur, amp_tgt) in [(0.2, 0.8), (0.8, 0.8)]:
        out.append(amp_cur + (amp_tgt - amp_cur) * ramp)
    return np.concatenate(out)


def junction_d2(sig):
    """Max |second difference| scanning the chunk junction (n-3..n+2)."""
    d2 = np.abs(np.diff(sig, 2))
    return float(d2[n - 4:n + 1].max())


def main():
    linear = np.linspace(0.0, 1.0, n)
    smooth = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n))) * 0.5

    # envelope-only: pure corner, no carrier-curvature contamination
    el = junction_d2(env_two_chunks(linear))
    es = junction_d2(env_two_chunks(smooth))
    print("=== (A) micro-test: ENVELOPE corner at chunk junction (no carrier) ===")
    print(f"LINEAR ramp envelope |d2| at junction = {el:.3e}")
    print(f"SMOOTH ramp envelope |d2| at junction = {es:.3e}")
    print(f"corner reduction: {el/max(es,1e-15):.0f}x")

    print("\n=== (B) scene-test: per-boundary |d2|, linear vs smooth ===")
    import _render_probe as P
    g = P.make_grid()
    L.MASTER_GAIN = 0.04
    a_lin = P.render(g)              # current source already smoothstep...
    # To get a true LINEAR render we reload with linspace; instead reuse saved wavs.
    from scipy.io import wavfile
    sr1, w_lin = wavfile.read("_probe_gainfix.wav")   # linear-ramp render
    sr2, w_smo = wavfile.read("_probe_clickfix.wav")  # smoothstep render
    m_lin = w_lin.astype(np.float64).mean(axis=1)
    m_smo = w_smo.astype(np.float64).mean(axis=1)
    N = min(len(m_lin), len(m_smo))
    d2_lin = np.abs(np.diff(m_lin[:N], 2))
    d2_smo = np.abs(np.diff(m_smo[:N], 2))
    bnd = np.arange(n, N - 2, n)
    bl = np.array([d2_lin[max(0, b - 2):b + 2].max() for b in bnd])
    bs = np.array([d2_smo[max(0, b - 2):b + 2].max() for b in bnd])
    # transition boundaries = those where linear had elevated d2
    order = np.argsort(bl)[::-1][:8]   # 8 worst (transition) boundaries
    print(f"all boundaries: linear mean={bl.mean():.2f} smooth mean={bs.mean():.2f}")
    print("8 worst (transition) boundaries: linear -> smooth |d2|")
    for o in order:
        print(f"  boundary @sample {bnd[o]:>7}: {bl[o]:6.1f} -> {bs[o]:6.1f}  "
              f"({100*bs[o]/max(bl[o],1e-9):.0f}%)")


if __name__ == "__main__":
    main()
