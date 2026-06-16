#!/usr/bin/env python3
"""Offline render probe for gol_life_synth_laplacian.py.

Regression harness for the `audio-artifact-probe` skill (.claude/skills/).
Renders a scene to a WAV offline (no pygame display/audio) using the CURRENT
source code, then measures (a) clipping/distortion and (b) chunk-boundary clicks
objectively, and saves a normalised magnitude spectrum so timbre can be compared
across fixes.

    python _render_probe.py session <ts> [label]  # PRIMARY: replay a recorded
                                                   # user session snapshot
                                                   # (_session_<ts>.npz) with
                                                   # full knobs; verify fidelity
                                                   # vs the recorded WAV
    python _render_probe.py <label>         # FALLBACK: 3-pentadecathlon scene
    python _render_probe.py compare A B      # compare two saved spectra (shape)

The session-replay mode is the primary pipeline: a click that only shows up with
a specific field + knob combination must be reproduced on the user's actual
snapshot, not on a synthetic scene.  The pentadecathlon scene is only a fallback
when no session is available.
"""
import os
import sys
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gol_life_synth_laplacian as L
from scipy.io import wavfile

PENTA = [(0, 1), (1, 1), (2, 0), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1),
         (7, 0), (7, 2), (8, 1), (9, 1)]
COLS = [8, 25, 42]          # ~equal spacing on 52-wide torus (gaps ~15)
ROW0 = 10                   # vertical centre (rows 10..19 of 30)
DURATION = 6.0


def make_grid():
    g = np.zeros((L.GRID_H, L.GRID_W), np.uint8)
    for c0 in COLS:
        for (r, c) in PENTA:
            g[ROW0 + r, c0 + c] = 1
    return g


def render(grid0, gain=None):
    """Render the scene with current source; returns int16 (N,2). Deterministic."""
    if gain is not None:
        L.MASTER_GAIN = gain
    grid = grid0.copy()
    pool = L.SlotPool()
    sz = L.TOTAL_SLOTS + 1
    phase = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)
    f0 = L.midi_to_freq(L.NOTE_DEFAULT)

    n_chunks = int(DURATION / L.CHUNK_S)
    acc = 0.0
    interval = 1.0 / L.STEP_HZ
    # default-knob render: lowest modes, 1/i rolloff, short release, vol=1.0 so the
    # probe measures full-scale clipping against MASTER_GAIN (as before).
    release_chunks = max(1, round(L.RELEASE_MS_DEFAULT / 1000.0 / L.CHUNK_S))
    out = []
    for _ in range(n_chunks):
        acc += L.CHUNK_S
        while acc >= interval:
            acc -= interval
            grid = L.step(grid)
        _, voices, _ = L.analyse(grid, f0)
        pool.update(voices, phase, amp_cur, pan_cur, release_chunks)
        buf, _pk, _nc = L.render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                 pool.amp_tgt, pool.pan_tgt,
                                                 pool.freq_slots, 2,
                                                 L.MASTER_GAIN, L.MASTER_GAIN)
        out.append(buf)
    return np.concatenate(out)


def replay_session(ts):
    """Re-render a recorded session (_session_<ts>.npz) with the CURRENT engine,
    using the exact logged per-frame inputs (grid + note + all knobs + chunk
    count).  Returns (audio int16 (N,2), pre_clip_peak, n_clipped).  Deterministic
    -> with UNFIXED code it must match the recorded WAV (fidelity check)."""
    d = np.load(f"_session_{ts}.npz")
    rp = d["replay"]
    grids = d["replay_grids"]
    if len(rp) == 0:
        raise SystemExit(f"_session_{ts}.npz has no replay log "
                         "(recorded before per-frame knob logging was added)")
    pool = L.SlotPool()
    sz = L.TOTAL_SLOTS + 1
    phase = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)
    out = []
    pkmax = 0.0
    nclip = 0
    gain_prev = L.MASTER_GAIN * float(rp[0][5])    # initial volume
    for i in range(len(rp)):
        row = rp[i]
        n_rendered, note, spread, alpha, rel_ms, vol = row[:6]
        # n_partials added as a 7th column later; old logs default to the FIXED mode
        # count recorded in the npz (d["max_modes"]), so they replay faithfully even
        # though the current engine's MAX_MODES_PER_OBJ ceiling has since changed.
        n_partials = int(round(row[6])) if len(row) > 6 else int(d["max_modes"])
        n_rendered = int(round(n_rendered))
        grid = grids[i]
        f0 = L.midi_to_freq(int(round(note)))
        _, voices, _ = L.analyse(grid, f0, float(spread), float(alpha), n_partials)
        rc = max(1, round(rel_ms / 1000.0 / L.CHUNK_S))
        gain = L.MASTER_GAIN * float(vol)
        for _ in range(n_rendered):
            pool.update(voices, phase, amp_cur, pan_cur, rc)
            buf, pk, nc = L.render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                   pool.amp_tgt, pool.pan_tgt,
                                                   pool.freq_slots, 2, gain_prev, gain)
            gain_prev = gain
            out.append(buf)
            pkmax = max(pkmax, pk)
            nclip += nc
    return np.concatenate(out), pkmax, nclip


def fidelity_vs_recorded(ts, replayed):
    """Compare the offline replay against the recorded WAV.  With unfixed code
    they should be (near-)identical, proving the replay reproduces the live
    artifact; a fix then changes this on purpose."""
    try:
        sr, rec = wavfile.read(f"_session_{ts}.wav")
    except FileNotFoundError:
        print("  (no recorded WAV to compare fidelity)")
        return
    n = min(len(rec), len(replayed))
    if len(rec) != len(replayed):
        print(f"  [fidelity] length differs: recorded={len(rec)} replay={len(replayed)} "
              f"(comparing first {n})")
    diff = np.abs(rec[:n].astype(np.int32) - replayed[:n].astype(np.int32))
    print(f"  [fidelity vs recorded WAV] max|diff|={int(diff.max())} "
          f"mean|diff|={diff.mean():.2f}  "
          f"({'IDENTICAL' if diff.max() == 0 else 'differs'})")


def analyse_audio(audio, label):
    mono = audio.astype(np.float64).mean(axis=1)
    N = len(mono)
    n = int(L.CHUNK_S * L.SR)

    # ---- distortion: clipping ----
    sat = int(np.sum(np.abs(audio) >= 32767))
    clipped = np.abs(audio) >= 32767
    runs = 0
    for ch in range(audio.shape[1]):
        m = clipped[:, ch].astype(int)
        if m.any():
            d = np.diff(np.concatenate(([0], m, [0])))
            s = np.where(d == 1)[0]
            e = np.where(d == -1)[0]
            runs = max(runs, int((e - s).max()))

    # ---- clicks: slope discontinuity (2nd difference) ----
    d2 = np.abs(np.diff(mono, 2))           # index i -> sample i+1
    boundaries = np.arange(n, N - 2, n)
    bvals = np.array([d2[max(0, b - 3):b + 3].max() for b in boundaries])
    imed = float(np.median(d2))

    # locate top-10 d2 spikes; report distance to nearest chunk boundary
    top = np.argsort(d2)[-10:][::-1]
    dist = [int(min(idx % n, n - (idx % n))) for idx in top]

    print(f"\n=== {label} ===")
    print(f"samples={N}  peak_int16={int(np.abs(audio).max())}")
    print(f"[DISTORTION] saturated(|x|>=32767)={sat} ({100*sat/audio.size:.3f}%)  "
          f"longest clip run={runs}")
    print(f"[CLICKS] |d2| median(interior)={imed:.2f}  "
          f"boundary mean={bvals.mean():.2f} max={bvals.max():.1f}  "
          f"ratio(bnd/int)={bvals.mean()/max(imed,1e-9):.2f}")
    print(f"[CLICKS] top-10 |d2| values: {[round(float(d2[i]),0) for i in top]}")
    print(f"[CLICKS] their dist to nearest chunk boundary (0=on boundary): {dist}")

    # normalised spectrum (shape only)
    w = np.hanning(N)
    S = np.abs(np.fft.rfft(mono * w))
    Snorm = S / max(S.max(), 1e-12)
    return Snorm


def compare(a, b):
    Sa = np.load(f"_probe_spec_{a}.npy")
    Sb = np.load(f"_probe_spec_{b}.npy")
    freqs = np.fft.rfftfreq((len(Sa) - 1) * 2, 1.0 / L.SR)
    # focus on audible band
    band = (freqs >= 20) & (freqs <= 20000)
    Sa, Sb, fr = Sa[band], Sb[band], freqs[band]
    # partial region = bins with significant energy in A (the reference)
    mask = Sa > 0.02
    rel = np.abs(Sb[mask] - Sa[mask]) / np.maximum(Sa[mask], 1e-6)
    # broadband floor (non-partial bins) energy: artifact proxy
    floor_a = float(np.sqrt(np.mean(Sa[~mask] ** 2)))
    floor_b = float(np.sqrt(np.mean(Sb[~mask] ** 2)))
    cos = float(np.dot(Sa, Sb) / (np.linalg.norm(Sa) * np.linalg.norm(Sb) + 1e-12))
    print(f"\n=== compare {a} vs {b} (normalised spectra) ===")
    print(f"cosine similarity (1.0=identical shape): {cos:.6f}")
    print(f"max relative diff on partials (>2% bins): {rel.max()*100:.2f}%")
    print(f"mean relative diff on partials: {rel.mean()*100:.3f}%")
    print(f"broadband floor RMS (artifact proxy): {a}={floor_a:.5e}  "
          f"{b}={floor_b:.5e}  ({'cleaner' if floor_b < floor_a else 'dirtier'})")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "session":
        ts = sys.argv[2]
        label = sys.argv[3] if len(sys.argv) > 3 else f"session_{ts}"
        audio, peak, nclip = replay_session(ts)
        print(f"[replay] pre-clip peak={peak:.3f} ({20*np.log10(peak+1e-9):+.1f} dBFS)  "
              f"clipped samples={nclip}")
        Snorm = analyse_audio(audio, label)
        fidelity_vs_recorded(ts, audio)
        wavfile.write(f"_probe_{label}.wav", L.SR, audio)
        np.save(f"_probe_spec_{label}.npy", Snorm)
        print(f"\nsaved _probe_{label}.wav and _probe_spec_{label}.npy")
        return
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    audio = render(make_grid())
    Snorm = analyse_audio(audio, label)
    wavfile.write(f"_probe_{label}.wav", L.SR, audio)
    np.save(f"_probe_spec_{label}.npy", Snorm)
    print(f"\nsaved _probe_{label}.wav and _probe_spec_{label}.npy")


if __name__ == "__main__":
    main()
