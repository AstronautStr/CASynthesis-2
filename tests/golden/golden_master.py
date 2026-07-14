#!/usr/bin/env python3
"""Golden-master audio regression gate for gol_synth.

WHY: every knob/refactor promises "default = bit-exact". This harness renders a
fixed deterministic programme through the real engine functions and compares the
output byte-for-byte against a committed reference. It moved here from
artifacts/_golden_master.py (2026-07-14) so the reference is VERSION-CONTROLLED
(artifacts/ is gitignored and swept during /dream consolidation).

HOW: mirrors the offline render path (analyse -> SlotPool.update ->
render_chunk_laplacian) over deterministic scenes covering all 5 engines, ADSR
(pluck S=0 / pad S=1), large shapes (>8x8 and the MAX_LAPLACIAN_NODES decimation
path) and scene changes (mode birth/death -> tail crossfade). Symbols are pulled
via `import gol_synth`, so the harness survives module reshuffles as long as
gol_synth re-exports them.

USAGE:
  python tests/golden/golden_master.py           # render and compare (PASS/FAIL)
  python tests/golden/golden_master.py --save    # re-bless the reference
                                                 # (ONLY after an agreed audible
                                                 #  change; mention it in the log)
The reference lives next to this file: golden_master_ref.npz (int16 samples)
+ golden_master_ref.wav for listening. Comparison is np.array_equal (0 diff = PASS).
"""
import os
import sys
import hashlib

import numpy as np

# Run from anywhere: make the repo root importable (tests/golden -> repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gol_synth as G   # noqa: E402  (after sys.path setup)

REF_NPZ = os.path.join(_HERE, "golden_master_ref.npz")
REF_WAV = os.path.join(_HERE, "golden_master_ref.wav")
OUT_WAV = os.path.join(_ROOT, "artifacts", "_golden_master_out.wav")


# -- deterministic scenes -----------------------------------------------------
def _grid():
    return np.zeros((G.GRID_H, G.GRID_W), np.uint8)


def _blinker():
    g = _grid()
    g[5, 5:8] = 1                      # 3-cell horizontal line (small oscillator)
    return g


def _block(top, left, h, w):
    g = _grid()
    g[top:top + h, left:left + w] = 1
    return g


def _two():
    g = _blinker()
    g[10:20, 20:30] = 1               # blinker + a 10x10 block (two voices, >8x8)
    return g


SCENE_BLINKER = _blinker()
SCENE_BLOCK   = _block(10, 20, 10, 10)     # 10x10 > 8x8 -> fullshape != extract
SCENE_BIG     = _block(4, 4, 20, 20)       # 20x20 = 400 cells > 256 -> decimation path
SCENE_TWO     = _two()
SCENE_EMPTY   = _grid()                    # silence -> release tails ring out


def _params(engine_id, alt=False):
    """Default params for an engine, or a non-default variant when alt=True."""
    specs = G.ENGINE_BY_ID[engine_id]['params']
    p = {arg: default for (arg, _l, _lo, _hi, _i, default) in specs}
    if not alt:
        return p
    if engine_id == 'laplacian':
        p.update(n=8, spread=0.5, alpha=0.7, shape=0.7, harm=0.4)
    else:
        p['n'] = 10
    return p


# Each frame: (scene, engine, params, note, adsr, vol, n_rendered)
def _build_frames():
    f = []
    # ADSR variants exercised on the laplacian engine + scene changes.
    adsr_default = dict(attack_ms=1.0, decay_ms=0.0, sustain=1.0, release_ms=50.0)
    adsr_pluck   = dict(attack_ms=5.0, decay_ms=50.0, sustain=0.0, release_ms=200.0)
    adsr_pad     = dict(attack_ms=200.0, decay_ms=0.0, sustain=1.0, release_ms=1000.0)

    # All five engines: default params then an alt-params pass, on a two-voice scene.
    for eng in [e['id'] for e in G.ENGINES]:
        f.append((SCENE_TWO, eng, _params(eng), 48, adsr_default, 0.70, 3))
        f.append((SCENE_BLOCK, eng, _params(eng, alt=True), 60, adsr_default, 0.55, 3))

    # Laplacian-specific paths: big-shape decimation + ADSR + scene-change tails.
    lap = 'laplacian'
    f.append((SCENE_BIG, lap, _params(lap), 36, adsr_default, 0.70, 2))           # decimation
    f.append((SCENE_TWO, lap, _params(lap), 48, adsr_pluck, 0.70, 3))             # pluck S=0
    f.append((SCENE_BLINKER, lap, _params(lap), 48, adsr_pluck, 0.70, 3))         # topology change
    f.append((SCENE_TWO, lap, _params(lap), 55, adsr_pad, 0.80, 4))              # pad S=1, vol up
    f.append((SCENE_EMPTY, lap, _params(lap), 55, adsr_pad, 0.80, 4))             # release tails
    f.append((SCENE_BLINKER, lap, _params(lap), 48, adsr_default, 0.65, 3))       # voices reborn
    return f


# -- render (mirror of the offline replay loop) --------------------------------
def render():
    sr, chunk_s, master = G.SR, G.CHUNK_S, G.MASTER_GAIN
    sz = G.TOTAL_SLOTS + 1
    pool = G.SlotPool()
    phase   = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)
    frames = _build_frames()
    gain_prev = master * frames[0][5]
    out = []
    for (grid, eng, params, note, adsr, vol, n_rendered) in frames:
        f0 = G.midi_to_freq(int(note))
        _, voices, _ = G.analyse(grid, f0, eng, dict(params))
        release_chunks = max(1, round(float(adsr['release_ms']) / 1000.0 / chunk_s))
        attack_chunks  = max(1, round(float(adsr['attack_ms'])  / 1000.0 / chunk_s))
        decay_chunks   = max(1, round(float(adsr['decay_ms'])   / 1000.0 / chunk_s))
        sustain        = float(adsr['sustain'])
        gain = master * float(vol)
        for _ in range(int(n_rendered)):
            pool.update(voices, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain)
            buf, _pk, _nc = G.render_chunk_laplacian(
                phase, amp_cur, pan_cur, pool.amp_tgt, pool.pan_tgt,
                pool.freq_slots, 2, gain_prev, gain)
            gain_prev = gain
            out.append(buf)
    return np.concatenate(out) if out else np.zeros((0, 2), np.int16)


def _save(audio):
    from scipy.io import wavfile
    np.savez_compressed(REF_NPZ, audio=audio, sr=G.SR, chunk_s=G.CHUNK_S)
    wavfile.write(REF_WAV, G.SR, audio)


def main():
    force_save = "--save" in sys.argv
    audio = render()
    digest = hashlib.sha256(audio.tobytes()).hexdigest()[:16]
    print(f"[golden-master] rendered {len(audio)} samples  "
          f"CHUNK_S={G.CHUNK_S}  sha256[:16]={digest}")

    if force_save or not os.path.exists(REF_NPZ):
        _save(audio)
        print(f"[baseline saved] {REF_NPZ} (+ {REF_WAV})")
        return 0

    ref = np.load(REF_NPZ)["audio"]
    if audio.shape == ref.shape and np.array_equal(audio, ref):
        print(f"[PASS] byte-exact match vs {os.path.basename(REF_NPZ)} "
              f"({len(ref)} samples)")
        return 0
    # mismatch -> write the divergent output and report.
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    wavfile.write(OUT_WAV, G.SR, audio)
    n = min(len(audio), len(ref))
    if n and audio.shape[1:] == ref.shape[1:]:
        diff = np.abs(audio[:n].astype(np.int64) - ref[:n].astype(np.int64))
        print(f"[FAIL] differs from baseline: shapes ref={ref.shape} "
              f"out={audio.shape}  common={n}  max|diff|={int(diff.max())}  "
              f"mean|diff|={diff.mean():.4f}  (wrote {OUT_WAV})")
    else:
        print(f"[FAIL] shape mismatch ref={ref.shape} out={audio.shape} "
              f"(wrote {OUT_WAV})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
