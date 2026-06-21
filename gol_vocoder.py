#!/usr/bin/env python3
"""
gol_vocoder.py — GoL-vocoder: offline creative resynthesis via SA search.

Finds a binary GoL field whose Laplacian spectrum (casynth_core.map_laplacian)
approximates the Bark-band envelope of a target sound.  The result is "target
sound through GoL-metal" — a creative effect, not a faithful copy.  Exact
inversion is impossible (co-spectral graphs; Kac 1966 / Gordon-Webb-Wolpert 1992).
See memory/decisions.md 2026-06-21 "GoL-вокодер (T3)".

Usage
-----
  python gol_vocoder.py --selftest              # bring-up self-test (no WAV needed)
  python gol_vocoder.py target.wav              # search for a matching field
  python gol_vocoder.py target.wav --n 16 --shape 0.8 --steps 12000 --restarts 8
  python gol_vocoder.py target.wav --f0 220 --seed 7 --out my_run/

All parameters are recorded in log.json so the run is fully reproducible.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
import casynth_core as cc

# ── Defaults ──────────────────────────────────────────────────────────────
_DEF = dict(
    window   = 8,      # field size (window × window binary grid)
    n        = 16,     # map_laplacian partials
    spread   = 0.3,    # spread knob
    alpha    = 1.0,    # amplitude rolloff exponent
    shape    = 0.5,    # shape knob (>0 engages amplitude matching in metric)
    harm     = 0.0,    # harmonic quantisation knob
    steps    = 8000,   # SA steps per restart
    restarts = 5,      # number of independent random restarts
    t_start  = 0.5,    # initial SA temperature
    t_end    = 0.005,  # final SA temperature
    seed     = 0,      # RNG seed (for reproducibility)
    dur      = 2.0,    # render duration (seconds)
    sr       = 44100,  # audio sample rate
)

_N_BARK = 24  # number of perceptual Bark bands

# ── Perceptual metric ─────────────────────────────────────────────────────

def _hz_to_bark(f: float) -> float:
    """Zwicker/Traunmüller approximation.  0 Hz → 0 Bark, ~15.5 kHz → 24 Bark."""
    return 13.0 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500.0) ** 2)


# Uniform Bark-scale bin edges (0..24 Bark → _N_BARK bins)
_BARK_EDGES = np.linspace(0.0, 24.0, _N_BARK + 1)


def _bark_envelope(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Map (freqs, amps) → normalised Bark-band energy vector (length _N_BARK)."""
    env = np.zeros(_N_BARK)
    for f, a in zip(freqs, amps):
        if f <= 0.0 or a <= 0.0:
            continue
        b   = _hz_to_bark(float(f))
        idx = int(np.searchsorted(_BARK_EDGES, b, side='right')) - 1
        if 0 <= idx < _N_BARK:
            env[idx] += float(a)
    mx = env.max()
    if mx > 1e-9:
        env /= mx
    return env


def _bark_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine similarity.  0 = identical, 1 = orthogonal / one side silent."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ── WAV I/O ───────────────────────────────────────────────────────────────

def _load_wav_mono(path: Path):
    """Load WAV → (float32 samples, sample_rate)."""
    with wave.open(str(path), 'rb') as w:
        sr        = w.getframerate()
        n_frames  = w.getnframes()
        n_ch      = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw       = w.readframes(n_frames)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f'Unsupported WAV sample width: {sampwidth} bytes')
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


def _save_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """Save float64/float32 samples in [-1, 1] to 16-bit mono WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    data16  = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data16.tobytes())


def _analyze_wav(samples: np.ndarray, sr: int,
                 n_peaks: int = 48, f_min: float = 50.0):
    """FFT → top peaks → Bark envelope + f0 anchor.

    Returns (bark_env: ndarray[_N_BARK], f0: float).
    f0 = lowest notable peak above f_min; used to anchor the Laplacian mapping.
    """
    win_len = min(sr, len(samples))
    start   = max(0, len(samples) // 2 - win_len // 2)
    seg     = samples[start:start + win_len]

    hann    = np.hanning(len(seg))
    mag     = np.abs(np.fft.rfft(seg * hann))
    faxis   = np.fft.rfftfreq(len(seg), 1.0 / sr)

    min_bin         = max(1, int(f_min * len(seg) / sr))
    peak_bins, _    = find_peaks(mag[min_bin:],
                                  height=mag[min_bin:].max() * 0.05,
                                  distance=3)
    peak_bins += min_bin

    if len(peak_bins) == 0:
        peak_bins = np.array([int(np.argmax(mag[min_bin:])) + min_bin])

    top_idx  = peak_bins[np.argsort(mag[peak_bins])[::-1][:n_peaks]]
    pk_freqs = faxis[top_idx]
    pk_amps  = mag[top_idx] / mag[top_idx].max()

    # f0 = lowest-frequency peak that is also "notable" (>= 20 % of loudest peak).
    # Using plain min() risks anchoring on a weak sub-harmonic noise peak.
    notable_mask = pk_amps >= 0.20
    if not np.any(notable_mask):
        notable_mask = np.ones(len(pk_amps), dtype=bool)  # all, if none qualify
    f0 = float(pk_freqs[notable_mask].min())
    return _bark_envelope(pk_freqs, pk_amps), f0


# ── Additive synthesis render ─────────────────────────────────────────────

def _render(freqs: np.ndarray, amps: np.ndarray, dur: float, sr: int) -> np.ndarray:
    """Additive synthesis: sum of sine partials, short fade to avoid clicks."""
    t   = np.linspace(0.0, dur, int(dur * sr), endpoint=False)
    out = np.zeros(len(t), dtype=np.float64)
    for f, a in zip(freqs, amps):
        if f > 0.0 and a > 0.0:
            out += float(a) * np.sin(2.0 * math.pi * float(f) * t)
    fade = max(1, int(0.02 * sr))
    out[:fade]  *= np.linspace(0.0, 1.0, fade)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    mx = np.abs(out).max()
    if mx > 1e-9:
        out /= mx * 1.05
    return out.astype(np.float32)


# ── SA cost function ──────────────────────────────────────────────────────

def _eval_field(field: np.ndarray, target_env: np.ndarray,
                f0: float, cfg: dict) -> float:
    """Bark distance between the field's Laplacian spectrum and the target."""
    patch       = field.reshape(cfg['window'], cfg['window']).astype(np.uint8)
    freqs, amps = cc.map_laplacian(
        patch, f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    return _bark_distance(_bark_envelope(freqs, amps), target_env)


# ── Simulated annealing ───────────────────────────────────────────────────

def _sa_search(target_env: np.ndarray, f0: float, cfg: dict, rng,
               verbose: bool = True):
    """SA over a binary (window²,) field.  Returns (best_field, best_cost)."""
    size    = cfg['window'] ** 2
    steps   = cfg['steps']
    T_start = cfg['t_start']
    T_end   = cfg['t_end']

    best_field = np.zeros(size, dtype=np.uint8)
    best_cost  = float('inf')

    for restart in range(cfg['restarts']):
        field = (rng.random(size) < 0.25).astype(np.uint8)
        cost  = _eval_field(field, target_env, f0, cfg)

        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))

            idx       = int(rng.integers(0, size))
            field[idx] ^= 1
            new_cost  = _eval_field(field, target_env, f0, cfg)
            delta     = new_cost - cost

            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                cost = new_cost
            else:
                field[idx] ^= 1  # revert

            if cost < best_cost:
                best_cost  = cost
                best_field = field.copy()

        if verbose:
            print(f'  restart {restart + 1}/{cfg["restarts"]}:  '
                  f'local={cost:.4f}  global_best={best_cost:.4f}')

    return best_field, best_cost


# ── Output helpers ────────────────────────────────────────────────────────

def _field_str(field: np.ndarray, w: int) -> str:
    patch = field.reshape(w, w)
    return '\n'.join(''.join('#' if c else '.' for c in row) for row in patch)


def _field_cells(field: np.ndarray, w: int):
    patch = field.reshape(w, w)
    return [(int(r), int(c))
            for r in range(w) for c in range(w) if patch[r, c]]


# ── Self-test ─────────────────────────────────────────────────────────────

def run_selftest(cfg: dict, rng) -> bool:
    """Bring-up self-test: feed own spectrum of a known field as the target.

    Pass criterion: SA achieves Bark distance < 50 % of the random baseline.
    This validates that the analysis→search→render loop is closed and the
    metric is meaningful — without requiring a real audio file.
    """
    print('=' * 60)
    print('SELF-TEST: synthesise own spectrum as search target')
    print('=' * 60)

    w    = cfg['window']
    size = w * w
    f0   = 261.0  # C4

    # Known shape: cross + two extra arms (7 live cells, moderately complex spectrum)
    known = np.zeros(size, dtype=np.uint8)
    mid   = w // 2
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (0, 2)]:
        r, c = mid + dr, mid + dc
        if 0 <= r < w and 0 <= c < w:
            known[r * w + c] = 1

    print(f'Known field ({int(known.sum())} live cells):')
    print(_field_str(known, w))

    fk, ak     = cc.map_laplacian(
        known.reshape(w, w).astype(np.uint8), f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    target_env = _bark_envelope(fk, ak)
    print(f'Target freqs (non-zero): {fk[fk > 0].round(1)}')

    # Random baseline
    rand_costs = [
        _eval_field((rng.random(size) < 0.25).astype(np.uint8), target_env, f0, cfg)
        for _ in range(30)
    ]
    baseline = float(np.mean(rand_costs))
    print(f'\nRandom baseline cost (mean of 30): {baseline:.4f}')

    print('\nRunning SA ...')
    best_field, best_cost = _sa_search(target_env, f0, cfg, rng, verbose=True)

    bf, ba = cc.map_laplacian(
        best_field.reshape(w, w).astype(np.uint8), f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    print(f'\nBest found field (cost={best_cost:.4f}, baseline={baseline:.4f}):')
    print(_field_str(best_field, w))
    print(f'Best freqs (non-zero): {bf[bf > 0].round(1)}')

    pct = (1.0 - best_cost / (baseline + 1e-9)) * 100.0
    print(f'\nImprovement over baseline: {pct:.1f}%')

    # Bark-bin match report (how many of the 24 bins differ by > 0.1)
    bf_env = _bark_envelope(bf, ba)
    mismatch_bins = int(np.sum(np.abs(bf_env - target_env) > 0.1))
    print(f'Bark-bin mismatch (|diff|>0.1): {mismatch_bins}/{_N_BARK}')

    ok_relative = best_cost < baseline * 0.5
    ok_absolute = best_cost < 0.05
    ok = ok_relative and ok_absolute
    print(f'\n{"PASS" if ok else "WARN"}:')
    print(f'  relative: best_cost {"<" if ok_relative else ">="} 0.5 * baseline  '
          f'({best_cost:.4f} vs {baseline * 0.5:.4f})')
    print(f'  absolute: best_cost {"<" if ok_absolute else ">="} 0.05  '
          f'({best_cost:.4f})')
    return ok


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='GoL-vocoder: creative resynthesis via SA search.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('target',     nargs='?',   help='Target WAV file')
    p.add_argument('--selftest', action='store_true',
                   help='Run bring-up self-test (no WAV needed)')
    p.add_argument('--window',   type=int,   default=_DEF['window'])
    p.add_argument('--n',        type=int,   default=_DEF['n'])
    p.add_argument('--spread',   type=float, default=_DEF['spread'])
    p.add_argument('--alpha',    type=float, default=_DEF['alpha'])
    p.add_argument('--shape',    type=float, default=_DEF['shape'])
    p.add_argument('--harm',     type=float, default=_DEF['harm'])
    p.add_argument('--steps',    type=int,   default=_DEF['steps'])
    p.add_argument('--restarts', type=int,   default=_DEF['restarts'])
    p.add_argument('--t-start',  type=float, default=_DEF['t_start'], dest='t_start')
    p.add_argument('--t-end',    type=float, default=_DEF['t_end'],   dest='t_end')
    p.add_argument('--seed',     type=int,   default=_DEF['seed'])
    p.add_argument('--dur',      type=float, default=_DEF['dur'],
                   help='Render duration in seconds')
    p.add_argument('--f0',       type=float, default=None,
                   help='Override auto-detected f0 (Hz)')
    p.add_argument('--out',      type=str,   default=None,
                   help='Output directory (default: vocoder_YYYYMMDD_HHMMSS/)')
    args = p.parse_args()

    cfg = dict(
        window   = args.window,
        n        = args.n,
        spread   = args.spread,
        alpha    = args.alpha,
        shape    = args.shape,
        harm     = args.harm,
        steps    = args.steps,
        restarts = args.restarts,
        t_start  = args.t_start,
        t_end    = args.t_end,
        seed     = args.seed,
        dur      = args.dur,
        sr       = _DEF['sr'],
    )
    rng = np.random.default_rng(cfg['seed'])

    # ── Self-test ──────────────────────────────────────────────────────
    if args.selftest:
        ok = run_selftest(cfg, rng)
        sys.exit(0 if ok else 1)

    # ── Normal run ─────────────────────────────────────────────────────
    if not args.target:
        p.error('Provide a target WAV file or pass --selftest')

    target_path = Path(args.target)
    if not target_path.exists():
        sys.exit(f'Error: {target_path} not found')

    ts      = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out) if args.out else Path(f'vocoder_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('GoL-vocoder')
    print(f'  target  : {target_path}')
    print(f'  output  : {out_dir}')
    print(f'  n={cfg["n"]}  spread={cfg["spread"]}  alpha={cfg["alpha"]}  '
          f'shape={cfg["shape"]}  harm={cfg["harm"]}')
    print(f'  steps={cfg["steps"]}  restarts={cfg["restarts"]}  '
          f'T={cfg["t_start"]}→{cfg["t_end"]}  seed={cfg["seed"]}')

    # 1. Analyse target
    print('\n[1/3] Analysing target ...')
    samples, wav_sr = _load_wav_mono(target_path)
    target_env, f0_auto = _analyze_wav(samples, wav_sr)
    f0 = args.f0 if args.f0 is not None else f0_auto
    print(f'  f0 = {f0:.1f} Hz  ({"manual" if args.f0 else "auto-detected"})')
    print(f'  non-zero Bark bins: {int(np.sum(target_env > 0))}')

    # 2. SA search
    print('\n[2/3] SA search ...')
    t0 = time.time()
    best_field, best_cost = _sa_search(target_env, f0, cfg, rng, verbose=True)
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s  |  best Bark distance = {best_cost:.4f}')

    # 3. Render and save
    print('\n[3/3] Rendering ...')
    w     = cfg['window']
    patch = best_field.reshape(w, w).astype(np.uint8)
    freqs, amps = cc.map_laplacian(
        patch, f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    result_samples = _render(freqs, amps, cfg['dur'], cfg['sr'])

    result_path = out_dir / 'result.wav'
    target_copy = out_dir / 'target.wav'
    _save_wav(result_path, result_samples, cfg['sr'])
    shutil.copy2(target_path, target_copy)

    field_repr = _field_str(best_field, w)
    print(f'\nBest field (Bark distance = {best_cost:.4f}):')
    print(field_repr)
    print(f'Freqs (non-zero): {freqs[freqs > 0].round(1)}')
    print(f'Amps  (non-zero): {amps[amps > 0].round(3)}')

    # Log — full record, sufficient for deterministic replay
    log = {
        'timestamp'     : ts,
        'target'        : str(target_path.resolve()),
        'f0_hz'         : float(f0),
        'f0_source'     : 'manual' if args.f0 else 'auto',
        'bark_distance' : float(best_cost),
        'cfg'           : cfg,
        'field'         : best_field.tolist(),
        'cells'         : _field_cells(best_field, w),
        'freqs_hz'      : [round(float(f), 3) for f in freqs],
        'amps'          : [round(float(a), 6) for a in amps],
        'field_str'     : field_repr,
    }
    log_path = out_dir / 'log.json'
    log_path.write_text(json.dumps(log, indent=2))

    print(f'\nSaved:')
    print(f'  {result_path}  ← best field rendered through GoL-metal')
    print(f'  {target_copy}  ← original target (A/B listening)')
    print(f'  {log_path}  ← full run log (reproducible)')


if __name__ == '__main__':
    main()
