#!/usr/bin/env python3
"""
gol_vocoder.py — GoL-vocoder: offline creative resynthesis via SA search.

Finds a binary GoL field whose Laplacian spectrum (casynth_core.map_laplacian)
approximates the log-mel envelope of a target sound.  The result is "target
sound through GoL-metal" — a creative effect, not a faithful copy.  Exact
inversion is impossible (co-spectral graphs; Kac 1966 / Gordon-Webb-Wolpert 1992).
See memory/decisions.md 2026-06-21 "GoL-вокодер (T3)".

Usage
-----
  python gol_vocoder.py --selftest
  python gol_vocoder.py target.wav
  python gol_vocoder.py target.wav --n 16 --shape 0.8 --steps 12000 --restarts 8
  python gol_vocoder.py speech.wav --dynamic --frame-steps 500 --max-hamming 8 --ema-alpha 0.7
"""

import argparse
import json
import math
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np
from scipy.fftpack import dct as _scipy_dct
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
import casynth_core as cc

# ── Defaults ──────────────────────────────────────────────────────────────
_DEF = dict(
    window   = 8,
    n        = 16,
    spread   = 0.3,
    alpha    = 0.6,    # lowered from 1.0: more upper-partial energy in search
    shape    = 0.8,    # raised from 0.5: amplitude coupled to field shape
    harm     = 0.0,
    steps    = 8000,
    restarts = 5,
    t_start  = 0.5,
    t_end    = 0.005,
    seed     = 0,
    dur      = 2.0,
    sr       = 44100,
)

# ── Bark-scale metric (kept for legacy display in self-test) ──────────────

_N_BARK = 24

def _hz_to_bark(f: float) -> float:
    return 13.0 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500.0) ** 2)

_BARK_EDGES = np.linspace(0.0, 24.0, _N_BARK + 1)

def _bark_envelope(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
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


# ── Mel-scale metric (primary search metric) ──────────────────────────────

_N_MEL     = 40
_MEL_F_MIN = 80.0
_MEL_F_MAX = 8000.0

def _hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + max(f, 1e-9) / 700.0)

def _mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

_MEL_HZ = np.array([
    _mel_to_hz(m)
    for m in np.linspace(_hz_to_mel(_MEL_F_MIN), _hz_to_mel(_MEL_F_MAX), _N_MEL + 2)
])  # shape (_N_MEL+2,): left/center/right edges for each triangular filter


def _mel_logenv(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """(freqs, amps) -> log(1+energy) in _N_MEL mel bands. Vectorised."""
    env  = np.zeros(_N_MEL)
    mask = (freqs > 0.0) & (amps > 0.0)
    if not mask.any():
        return env
    f_v = freqs[mask].astype(np.float64)   # (M,)
    a_v = amps[mask].astype(np.float64)    # (M,)
    # Broadcast (N_MEL, M) for all filters at once
    f_lo  = _MEL_HZ[:-2, None]   # (N, 1)
    f_ctr = _MEL_HZ[1:-1, None]  # (N, 1)
    f_hi  = _MEL_HZ[2:, None]    # (N, 1)
    f_mat = f_v[None, :]          # (1, M)
    left  = (f_mat > f_lo)  & (f_mat <= f_ctr)
    right = (f_mat > f_ctr) & (f_mat <  f_hi)
    w = (np.where(left,  (f_mat - f_lo)  / np.maximum(f_ctr - f_lo,  1e-9), 0.0) +
         np.where(right, (f_hi  - f_mat) / np.maximum(f_hi  - f_ctr, 1e-9), 0.0))
    env = w @ a_v   # (N,)
    return np.log1p(env)


def _mel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised L2 in log-mel space. 0 = identical."""
    d = a - b
    return float(math.sqrt(float(np.dot(d, d)) / _N_MEL))


# ── WAV I/O ───────────────────────────────────────────────────────────────

def _load_wav_mono(path: Path):
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
    clipped = np.clip(samples, -1.0, 1.0)
    data16  = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data16.tobytes())


# ── Analysis ──────────────────────────────────────────────────────────────

def _analyze_wav(samples: np.ndarray, sr: int,
                 n_peaks: int = 48, f_min: float = 50.0):
    """FFT -> top peaks -> mel log-envelope + f0 anchor.
    Returns (mel_env: ndarray[_N_MEL], f0: float).
    """
    win_len = min(sr, len(samples))
    start   = max(0, len(samples) // 2 - win_len // 2)
    seg     = samples[start:start + win_len]
    hann    = np.hanning(len(seg))
    mag     = np.abs(np.fft.rfft(seg * hann))
    faxis   = np.fft.rfftfreq(len(seg), 1.0 / sr)
    min_bin      = max(1, int(f_min * len(seg) / sr))
    peak_bins, _ = find_peaks(mag[min_bin:],
                               height=mag[min_bin:].max() * 0.05,
                               distance=3)
    peak_bins += min_bin
    if len(peak_bins) == 0:
        peak_bins = np.array([int(np.argmax(mag[min_bin:])) + min_bin])
    top_idx  = peak_bins[np.argsort(mag[peak_bins])[::-1][:n_peaks]]
    pk_freqs = faxis[top_idx]
    pk_amps  = mag[top_idx] / mag[top_idx].max()
    notable  = pk_amps >= 0.20
    if not np.any(notable):
        notable = np.ones(len(pk_amps), dtype=bool)
    f0 = float(pk_freqs[notable].min())
    return _mel_logenv(pk_freqs, pk_amps), f0


# ── Additive synthesis ────────────────────────────────────────────────────

def _render(freqs: np.ndarray, amps: np.ndarray, dur: float, sr: int) -> np.ndarray:
    """Static render with fade (no phase tracking needed)."""
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


def _render_raw_phase(freqs: np.ndarray, amps: np.ndarray,
                      n_samples: int, phase_acc: list, sr: int) -> np.ndarray:
    """Additive synthesis with phase continuity across frames.

    phase_acc is a per-slot list of phase offsets (radians), updated in-place.
    Phases are carried across consecutive chunks even when frequencies jump,
    eliminating OLA beating artifacts.
    """
    n = len(freqs)
    while len(phase_acc) < n:
        phase_acc.append(0.0)
    t   = np.arange(n_samples, dtype=np.float64) / sr
    out = np.zeros(n_samples, dtype=np.float64)
    for k in range(n):
        f = float(freqs[k]); a = float(amps[k])
        if f > 0.0 and a > 0.0:
            out += a * np.sin(2.0 * math.pi * f * t + phase_acc[k])
        if f > 0.0:
            phase_acc[k] = (phase_acc[k] + 2.0 * math.pi * f * n_samples / sr) % (2.0 * math.pi)
    return out


# ── SA cost function ──────────────────────────────────────────────────────

def _eval_field(field: np.ndarray, target_env: np.ndarray,
                f0: float, cfg: dict) -> float:
    """Log-mel L2 distance between field's Laplacian spectrum and target."""
    patch       = field.reshape(cfg['window'], cfg['window']).astype(np.uint8)
    freqs, amps = cc.map_laplacian(
        patch, f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    return _mel_l2(_mel_logenv(freqs, amps), target_env)


# ── Simulated annealing ───────────────────────────────────────────────────

def _sa_search(target_env: np.ndarray, f0: float, cfg: dict, rng,
               verbose: bool = True):
    """SA over binary (window^2,) field. Returns (best_field, best_cost)."""
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
            T         = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
            idx       = int(rng.integers(0, size))
            field[idx] ^= 1
            new_cost  = _eval_field(field, target_env, f0, cfg)
            delta     = new_cost - cost
            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                cost = new_cost
            else:
                field[idx] ^= 1
            if cost < best_cost:
                best_cost  = cost
                best_field = field.copy()
        if verbose:
            print(f'  restart {restart + 1}/{cfg["restarts"]}:  '
                  f'local={cost:.4f}  global_best={best_cost:.4f}')
    return best_field, best_cost


def _sa_warm(field_init: np.ndarray, target_env: np.ndarray, f0: float,
             cfg: dict, rng, steps: int = 400,
             max_hamming: int = None) -> tuple:
    """Short SA warm-started from field_init.

    max_hamming: hard limit on Hamming distance from field_init.
    Bounds timbral jump between consecutive frames.
    """
    field     = field_init.copy()
    cost      = _eval_field(field, target_env, f0, cfg)
    best_f    = field.copy()
    best_cost = cost
    size      = len(field)
    T_start, T_end = 0.15, 0.002
    n_changed = 0   # Hamming distance from field_init

    for step in range(steps):
        T   = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
        idx = int(rng.integers(0, size))

        before_same = (field[idx] == field_init[idx])
        if max_hamming is not None and before_same and n_changed >= max_hamming:
            continue  # hard Hamming budget: skip moves that would exceed it

        field[idx] ^= 1
        new_cost   = _eval_field(field, target_env, f0, cfg)
        delta      = new_cost - cost

        if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cost = new_cost
            if before_same:
                n_changed += 1   # bit now differs from init
            else:
                n_changed -= 1   # bit now matches init
        else:
            field[idx] ^= 1  # revert

        if cost < best_cost:
            best_cost = cost
            best_f    = field.copy()

    return best_f, best_cost


# ── Offline MCD evaluation ────────────────────────────────────────────────

def compute_mcd(path_ref: Path, path_test: Path, n_mfcc: int = 13) -> tuple:
    """MCD and delta-MCD between two WAV files (offline eval, not in SA loop).

    Both resampled to 16 kHz. MCD (dB) = timbral distance per frame (c1..c13).
    delta-MCD = temporal jitter mismatch (first-difference cepstra).
    Lower = closer to target.
    """
    target_sr_a = 16000

    def _resample(x, sr_in, sr_out):
        if sr_in == sr_out:
            return x
        n_out = int(len(x) * sr_out / sr_in)
        return np.interp(np.linspace(0, len(x) - 1, n_out),
                         np.arange(len(x)), x).astype(np.float32)

    ref_raw, sr_ref = _load_wav_mono(path_ref)
    tst_raw, sr_tst = _load_wav_mono(path_test)
    for s in (ref_raw, tst_raw):
        mx = np.abs(s).max()
        if mx > 1e-9:
            s /= mx
    ref = _resample(ref_raw, sr_ref, target_sr_a)
    tst = _resample(tst_raw, sr_tst, target_sr_a)

    win  = int(0.025 * target_sr_a)  # 25 ms
    hop  = int(0.010 * target_sr_a)  # 10 ms

    # Mel filterbank for cepstrum analysis
    n_mel_a  = 40
    mel_min_ = _hz_to_mel(80.0)
    mel_max_ = _hz_to_mel(min(8000.0, target_sr_a / 2 * 0.95))
    mel_pts_ = np.linspace(mel_min_, mel_max_, n_mel_a + 2)
    hz_pts_  = np.array([_mel_to_hz(m) for m in mel_pts_])
    faxis_a  = np.fft.rfftfreq(win, 1.0 / target_sr_a)
    f_lo_ = hz_pts_[:-2, None];  f_ctr_ = hz_pts_[1:-1, None];  f_hi_ = hz_pts_[2:, None]
    fv_   = faxis_a[None, :]
    fb    = (np.where((fv_ > f_lo_)  & (fv_ <= f_ctr_),
                      (fv_ - f_lo_)  / np.maximum(f_ctr_ - f_lo_,  1e-9), 0.0) +
             np.where((fv_ > f_ctr_) & (fv_ <  f_hi_),
                      (f_hi_ - fv_)  / np.maximum(f_hi_  - f_ctr_, 1e-9), 0.0))

    hann_a = np.hanning(win)

    def _cepstra(sig):
        frames = []
        for start in range(0, max(1, len(sig) - win + 1), hop):
            frm = sig[start:start + win]
            if len(frm) < win:
                frm = np.pad(frm, (0, win - len(frm)))
            mag   = np.abs(np.fft.rfft(frm.astype(np.float64) * hann_a))
            mel_e = fb @ mag
            log_m = np.log(np.maximum(mel_e, 1e-8))
            c     = _scipy_dct(log_m, type=2, norm='ortho')
            frames.append(c[1:n_mfcc + 1])  # skip c0 (energy)
        return np.array(frames) if frames else np.zeros((1, n_mfcc))

    c_ref = _cepstra(ref);  c_tst = _cepstra(tst)
    n     = min(len(c_ref), len(c_tst))
    c_ref = c_ref[:n];  c_tst = c_tst[:n]

    diff = c_ref - c_tst
    mcd  = (10.0 / math.log(10.0)) * float(
        np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    )
    if n > 1:
        dd        = np.diff(c_ref, axis=0) - np.diff(c_tst, axis=0)
        delta_mcd = (10.0 / math.log(10.0)) * float(
            np.mean(np.sqrt(2.0 * np.sum(dd ** 2, axis=1)))
        )
    else:
        delta_mcd = 0.0
    return mcd, delta_mcd


# ── Dynamic (frame-by-frame) ──────────────────────────────────────────────

def run_dynamic(target_path: Path, f0: float, cfg: dict, rng,
                out_dir: Path, frame_steps: int = 400,
                max_hamming: int = None,
                ema_alpha: float = 1.0) -> Path:
    """Frame-by-frame SA with phase continuity and optional temporal smoothing.

    ema_alpha: weight on current frame envelope (1.0=no smooth, 0.7=slight smooth).
    max_hamming: max bits changed per frame from previous field (None=unlimited).
    Phase accumulators carry oscillator phases across frames.
    """
    samples, wav_sr = _load_wav_mono(target_path)
    sr = cfg['sr']

    hop_wav  = int(0.075 * wav_sr)
    win_wav  = hop_wav * 2
    hop_sr   = int(0.075 * sr)
    chunk_sr = hop_sr * 2

    positions = list(range(0, max(1, len(samples) - win_wav + 1), hop_wav))
    n_frames  = len(positions)
    print(f'  {n_frames} frames | {win_wav * 1000 // wav_sr}ms window, '
          f'{hop_wav * 1000 // wav_sr}ms hop, {frame_steps} SA steps/frame')
    if max_hamming is not None:
        print(f'  Hamming budget: {max_hamming} bits/frame')
    if ema_alpha < 1.0:
        print(f'  EMA alpha: {ema_alpha:.2f}')

    # ── Analyse all frames ─────────────────────────────────────────────
    frame_envs     = []
    frame_energies = []
    for pos in positions:
        seg   = samples[pos:pos + win_wav]
        h     = np.hanning(len(seg))
        mag   = np.abs(np.fft.rfft(seg * h))
        faxis = np.fft.rfftfreq(len(seg), 1.0 / wav_sr)
        min_b = max(1, int(50.0 * len(seg) / wav_sr))
        sub   = mag[min_b:]
        if sub.max() > 1e-9:
            pkb, _ = find_peaks(sub, height=sub.max() * 0.05, distance=3)
            if len(pkb) == 0:
                pkb = np.array([int(np.argmax(sub))])
            pkb += min_b
            pa  = mag[pkb] / mag[pkb].max()
            env = _mel_logenv(faxis[pkb], pa)
        else:
            env = np.zeros(_N_MEL)
        frame_envs.append(env)
        frame_energies.append(float(np.sqrt(np.mean(seg ** 2))))

    max_e = max(frame_energies) if max(frame_energies) > 1e-9 else 1.0
    frame_energies = [e / max_e for e in frame_energies]

    # Optional EMA smoothing of target envelopes
    if ema_alpha < 1.0:
        env_ema = frame_envs[0].copy()
        for i in range(len(frame_envs)):
            env_ema       = ema_alpha * frame_envs[i] + (1.0 - ema_alpha) * env_ema
            frame_envs[i] = env_ema.copy()

    # ── SA frame-by-frame ──────────────────────────────────────────────
    out_len   = n_frames * hop_sr + chunk_sr
    out_buf   = np.zeros(out_len, dtype=np.float64)
    hann      = np.hanning(chunk_sr)
    field     = (rng.random(cfg['window'] ** 2) < 0.25).astype(np.uint8)
    phase_acc = []  # per-slot oscillator phase accumulators

    for i, (env, energy) in enumerate(zip(frame_envs, frame_energies)):
        print(f'  frame {i + 1:3d}/{n_frames}  energy={energy:.2f}', end='\r')

        if env.max() < 1e-9 or energy < 0.005:
            chunk = np.zeros(chunk_sr, dtype=np.float64)
        else:
            field, _ = _sa_warm(field, env, f0, cfg, rng,
                                 steps=frame_steps, max_hamming=max_hamming)
            patch = field.reshape(cfg['window'], cfg['window']).astype(np.uint8)
            freqs, amps = cc.map_laplacian(
                patch, f0,
                n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
                shape=cfg['shape'], harm=cfg['harm'],
            )
            raw   = _render_raw_phase(freqs, amps, chunk_sr, phase_acc, sr)
            chunk = raw * hann * energy

        start = i * hop_sr
        out_buf[start:start + chunk_sr] += chunk

    print()

    mx = np.abs(out_buf).max()
    if mx > 1e-9:
        out_buf /= mx * 1.05

    out_path = out_dir / 'result_dynamic.wav'
    _save_wav(out_path, out_buf.astype(np.float32), sr)
    return out_path


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
    """Bring-up self-test using log-mel L2 metric.
    Pass: SA achieves mel-L2 < 50% of random baseline AND < 0.05.
    """
    print('=' * 60)
    print('SELF-TEST: synthesise own spectrum as search target')
    print('=' * 60)

    w    = cfg['window']
    size = w * w
    f0   = 261.0  # C4

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
    target_env = _mel_logenv(fk, ak)
    print(f'Target freqs (non-zero): {fk[fk > 0].round(1)}')

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

    bf_env        = _mel_logenv(bf, ba)
    mismatch_bins = int(np.sum(np.abs(bf_env - target_env) > 0.1))
    print(f'Mel-bin mismatch (|diff|>0.1): {mismatch_bins}/{_N_MEL}')

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
    p.add_argument('target',       nargs='?', help='Target WAV file')
    p.add_argument('--selftest',   action='store_true')
    p.add_argument('--window',     type=int,   default=_DEF['window'])
    p.add_argument('--n',          type=int,   default=_DEF['n'])
    p.add_argument('--spread',     type=float, default=_DEF['spread'])
    p.add_argument('--alpha',      type=float, default=_DEF['alpha'])
    p.add_argument('--shape',      type=float, default=_DEF['shape'])
    p.add_argument('--harm',       type=float, default=_DEF['harm'])
    p.add_argument('--steps',      type=int,   default=_DEF['steps'])
    p.add_argument('--restarts',   type=int,   default=_DEF['restarts'])
    p.add_argument('--t-start',    type=float, default=_DEF['t_start'], dest='t_start')
    p.add_argument('--t-end',      type=float, default=_DEF['t_end'],   dest='t_end')
    p.add_argument('--seed',       type=int,   default=_DEF['seed'])
    p.add_argument('--dur',        type=float, default=_DEF['dur'],
                   help='Render duration in seconds (static mode)')
    p.add_argument('--f0',         type=float, default=None,
                   help='Override auto-detected f0 (Hz)')
    p.add_argument('--dynamic',    action='store_true',
                   help='Frame-by-frame dynamic mode')
    p.add_argument('--frame-steps', type=int, default=400, dest='frame_steps',
                   help='SA steps per frame in --dynamic mode')
    p.add_argument('--max-hamming', type=int, default=None, dest='max_hamming',
                   help='Max bits changed per frame (None=unlimited; try 8-16)')
    p.add_argument('--ema-alpha',   type=float, default=1.0, dest='ema_alpha',
                   help='EMA weight on current frame envelope (1.0=off, 0.7=smooth)')
    p.add_argument('--out',        type=str, default=None,
                   help='Output directory (default: vocoder_YYYYMMDD_HHMMSS/)')
    args = p.parse_args()

    cfg = dict(
        window      = args.window,
        n           = args.n,
        spread      = args.spread,
        alpha       = args.alpha,
        shape       = args.shape,
        harm        = args.harm,
        steps       = args.steps,
        restarts    = args.restarts,
        t_start     = args.t_start,
        t_end       = args.t_end,
        seed        = args.seed,
        dur         = args.dur,
        sr          = _DEF['sr'],
        dynamic     = args.dynamic,
        frame_steps = args.frame_steps,
    )
    rng = np.random.default_rng(cfg['seed'])

    if args.selftest:
        ok = run_selftest(cfg, rng)
        sys.exit(0 if ok else 1)

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
          f'T={cfg["t_start"]}->{cfg["t_end"]}  seed={cfg["seed"]}')

    print('\n[1/3] Analysing target ...')
    samples, wav_sr = _load_wav_mono(target_path)
    target_env, f0_auto = _analyze_wav(samples, wav_sr)
    f0 = args.f0 if args.f0 is not None else f0_auto
    print(f'  f0 = {f0:.1f} Hz  ({"manual" if args.f0 else "auto-detected"})')
    print(f'  non-zero mel bins: {int(np.sum(target_env > 0))}')

    # ── Dynamic mode ───────────────────────────────────────────────────────
    if cfg.get('dynamic'):
        print('\n[2/2] Dynamic SA (frame-by-frame) ...')
        t0 = time.time()
        dyn_path = run_dynamic(
            target_path, f0, cfg, rng, out_dir,
            frame_steps = cfg['frame_steps'],
            max_hamming = args.max_hamming,
            ema_alpha   = args.ema_alpha,
        )
        target_copy = out_dir / 'target.wav'
        shutil.copy2(target_path, target_copy)
        print(f'  Done in {time.time() - t0:.1f}s')

        print('\nComputing MCD ...')
        mcd, delta_mcd = compute_mcd(target_copy, dyn_path)
        print(f'  MCD       = {mcd:.2f} dB  (timbral distance, lower = closer)')
        print(f'  delta-MCD = {delta_mcd:.2f} dB  (jitter, lower = smoother)')

        log = {
            'timestamp'    : ts,
            'target'       : str(target_path.resolve()),
            'f0_hz'        : float(f0),
            'f0_source'    : 'manual' if args.f0 else 'auto',
            'mode'         : 'dynamic',
            'cfg'          : cfg,
            'max_hamming'  : args.max_hamming,
            'ema_alpha'    : args.ema_alpha,
            'mcd_db'       : round(mcd, 3),
            'delta_mcd_db' : round(delta_mcd, 3),
        }
        (out_dir / 'log.json').write_text(json.dumps(log, indent=2))
        print(f'\nSaved:')
        print(f'  {dyn_path}  (dynamic GoL-metal vocoder)')
        print(f'  {target_copy}  (original, A/B listening)')
        return

    # ── Static mode ────────────────────────────────────────────────────────
    print('\n[2/3] SA search ...')
    t0 = time.time()
    best_field, best_cost = _sa_search(target_env, f0, cfg, rng, verbose=True)
    print(f'  Done in {time.time() - t0:.1f}s  |  best mel-L2 = {best_cost:.4f}')

    print('\n[3/3] Rendering ...')
    w     = cfg['window']
    patch = best_field.reshape(w, w).astype(np.uint8)
    freqs, amps = cc.map_laplacian(
        patch, f0,
        n=cfg['n'], spread=cfg['spread'], alpha=cfg['alpha'],
        shape=cfg['shape'], harm=cfg['harm'],
    )
    result_samples = _render(freqs, amps, cfg['dur'], cfg['sr'])
    result_path    = out_dir / 'result.wav'
    target_copy    = out_dir / 'target.wav'
    _save_wav(result_path, result_samples, cfg['sr'])
    shutil.copy2(target_path, target_copy)

    field_repr = _field_str(best_field, w)
    print(f'\nBest field (mel-L2 = {best_cost:.4f}):')
    print(field_repr)
    print(f'Freqs (non-zero): {freqs[freqs > 0].round(1)}')
    print(f'Amps  (non-zero): {amps[amps > 0].round(3)}')

    log = {
        'timestamp' : ts,
        'target'    : str(target_path.resolve()),
        'f0_hz'     : float(f0),
        'f0_source' : 'manual' if args.f0 else 'auto',
        'mel_l2'    : float(best_cost),
        'mode'      : 'static',
        'cfg'       : cfg,
        'field'     : best_field.tolist(),
        'cells'     : _field_cells(best_field, w),
        'freqs_hz'  : [round(float(f), 3) for f in freqs],
        'amps'      : [round(float(a), 6) for a in amps],
        'field_str' : field_repr,
    }
    log_path = out_dir / 'log.json'
    log_path.write_text(json.dumps(log, indent=2))
    print(f'\nSaved:')
    print(f'  {result_path}  (best field rendered through GoL-metal)')
    print(f'  {target_copy}  (original target, A/B listening)')
    print(f'  {log_path}  (full run log, reproducible)')


if __name__ == '__main__':
    main()
