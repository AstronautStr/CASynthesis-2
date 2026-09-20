"""Independent references for the Laplace FM gates (REQ memory/req-laplace-fm-2026-09-20.md
section 4) -- used by tests/test_laplace_fm.py and demos/laplace_fm_report.py.

Two different references, deliberately not the engine's own code path:

  stationary(...)   the phase formula A*sin(theta_c + sum beta_j*sin(theta_j)) written
                    out directly at an arbitrary oversampling, then the SAME band limit
                    and the SAME DC blocker as the engine (REQ section 4: "when checking
                    against the mathematical reference, apply the same filter").  Raising
                    the rate until two consecutive ones agree is what makes it converged.

  expansion(...)    the multi-modulator Jacobi-Anger expansion: every line
                    f0 + sum_j n_j*f_j with the product of Bessel coefficients, negative
                    frequencies reflected WITH their sign and coincident lines added
                    coherently.  The Bessel coefficients are taken the way the Researcher
                    preflight took them -- an FFT of exp(i*beta*sin(theta)) -- so the
                    check does not lean on the same series the engine could.

Nothing here imports scipy: the coefficients come from numpy FFTs.
"""
import itertools
import math

import numpy as np

from casynth_config import SR
from casynth_lab import laplace_fm as lfm


# ── the phase formula at an arbitrary rate ────────────────────────────────────

def raw(sources, f0, n_out, oversample, sr=SR, t0=0.0):
    """The bare oversampled FM sum (no band limit, no DC blocker).

    sources: [(A, theta_c0, [(f_j, beta_j, theta_j0), ...]), ...] -- one per figure.
    """
    n_os = int(n_out) * int(oversample)
    sr_os = float(sr) * float(oversample)
    idx = np.arange(n_os, dtype=float) + float(t0) * sr_os
    y = np.zeros(n_os)
    for amp, th_c, mods in sources:
        acc = np.zeros(n_os)
        for f, beta, th in mods:
            acc += float(beta) * np.sin(float(th) + (2.0 * math.pi * float(f) / sr_os) * idx)
        y += float(amp) * np.sin(float(th_c) + (2.0 * math.pi * float(f0) / sr_os) * idx + acc)
    return y


def stationary(sources, f0, n_out, oversample, sr=SR, dc=True):
    """`raw` brought to sr through the engine's own filter chain (same kernel, same
    fixed DC blocker), starting from an empty history."""
    kernel = lfm.fir_kernel(oversample, sr)
    y = raw(sources, f0, n_out, oversample, sr)
    out, _hist = decimated(y, kernel, oversample)
    if dc:
        out, _x, _y = lfm.dc_block(out, math.exp(-2.0 * math.pi * lfm.DC_HZ / float(sr)), 0.0, 0.0)
    return out


def decimated(y_os, kernel, oversample):
    return lfm.decimate(y_os, kernel, oversample, np.zeros(len(kernel) - 1))


# ── the Bessel expansion ──────────────────────────────────────────────────────

def bessel_coeffs(beta, order, n=16384):
    """(J_{-order}(beta) ... J_{order}(beta)) from one FFT of exp(i*beta*sin(theta)) --
    the Researcher preflight's own method."""
    t = 2.0 * math.pi * np.arange(n) / n
    c = np.fft.fft(np.exp(1j * float(beta) * np.sin(t))) / n
    k = np.arange(-int(order), int(order) + 1)
    return np.real(c[k % n])


def expansion_lines(f0, mods, order=12, floor=1e-12):
    """{frequency: (sin amplitude, cos amplitude)} of A=1 * sin(theta_c + sum beta*sin),
    from the product expansion.  A line at -f is reflected as sin(-x+p) = -sin(x-p), and
    coincident lines are summed coherently (never as powers).

    mods: [(f_j, beta_j, theta_j0), ...]; theta_c0 is taken as the caller's phase below.
    """
    coeffs = [bessel_coeffs(b, order) for _f, b, _th in mods]
    ns = range(-int(order), int(order) + 1)
    lines = {}
    for combo in itertools.product(ns, repeat=len(mods)):
        p = 1.0
        for c, n in zip(coeffs, combo):
            p *= c[n + order]
            if abs(p) < floor:
                break
        if abs(p) < floor:
            continue
        freq = float(f0) + sum(n * float(m[0]) for n, m in zip(combo, mods))
        phase = sum(n * float(m[2]) for n, m in zip(combo, mods))
        if freq < 0.0:
            freq, phase, p = -freq, -phase, -p
        key = round(freq, 6)
        s, c = lines.get(key, (0.0, 0.0))
        lines[key] = (s + p * math.cos(phase), c + p * math.sin(phase))
    return lines


def expansion_signal(amp, f0, th_c, mods, n, sr=SR, order=12, floor=1e-12):
    """The same signal rebuilt from the lines above -- an independent route to the time
    series that `raw` computes from the phase formula."""
    t = np.arange(int(n), dtype=float) / float(sr)
    y = np.zeros(int(n))
    for freq, (s, c) in expansion_lines(f0, mods, order, floor).items():
        w = 2.0 * math.pi * freq * t + float(th_c)
        y += float(amp) * (s * np.sin(w) + c * np.cos(w))
    return y


# ── small measurement helpers ─────────────────────────────────────────────────

def seam_ratios(engine, grid, rate_hz, gain, blocks=400, commands=4, step=0.001,
                param='harm', drag_from=30, warmup=40):
    """Drag `param` on a live field and measure, for every block, how much the phase sum
    JUMPS at the block seam relative to how much it moves inside the block.

    A click in this engine is a step in the oversampled sum -- an index that disappears at
    a boundary instead of being released -- so it is measured there, before the band limit
    spreads it over the filter's length.  A continuous signal keeps the ratio around 1.
    Returns (ratios, engine); the engine carries the tail counters afterwards.
    """
    from casynth_engine import step as ca_step, events_field
    from casynth_lab import BLOCK
    g = np.asarray(grid).copy()
    exc = None
    engine.init(g, exc, gain)
    n_os = BLOCK * engine.oversample
    value = float(engine.params[param])
    prev, rows, ca, gen, way = None, [], 0, 0, -1.0
    for i in range(int(blocks)):
        if ca >= (gen + 1) * (SR / float(rate_hz)):
            new = ca_step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            engine.update_field(g, exc)
        if i >= drag_from:
            for _k in range(int(commands)):
                value += way * step
                if not (0.0 <= value <= 1.0):
                    way = -way
                    value = min(max(value, 0.0), 1.0)
                engine.set_params(dict(engine.params, **{param: round(value, 3)}))
        if engine._pending_analysis:
            engine._analyse()
        engine.src.update(engine._mods, engine._index(), engine._release_chunks,
                          engine._attack_chunks, engine._decay_chunks, engine._sustain)
        y_os = engine.src.render(n_os, engine._ramp_os)
        if prev is not None:
            seam = abs(y_os[0] - 2.0 * prev[-1] + prev[-2])
            inside = float(np.percentile(np.abs(np.diff(y_os, 2)), 99.9))
            rows.append(seam / max(inside, 1e-18))
        prev = y_os[-2:].copy()
        ca += BLOCK
    return np.array(rows[warmup:]), engine


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return math.sqrt(float(np.mean(x * x))) if x.size else 0.0


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-300))


def mean_dc(x):
    """The constant component, measured through a Hann window: a plain mean over a
    non-integer number of periods of a 110 Hz line leaks ~1e-3 and would hide the
    blocker's real suppression."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    w = np.hanning(x.size)
    return float(np.sum(w * x) / np.sum(w))


def rel_db(got, ref):
    """Relative RMS error in dB of `got` against `ref`."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = min(len(got), len(ref))
    return db(rms(got[:n] - ref[:n]) / max(rms(ref[:n]), 1e-300))


def instantaneous_freq(y, sr=SR):
    """Instantaneous frequency (Hz) of a real signal through its analytic signal --
    used to measure the stationary peak deviation beta*f of one modulator."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    spec = np.fft.fft(y)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    z = np.fft.ifft(spec * h)
    ph = np.unwrap(np.angle(z))
    return np.diff(ph) * float(sr) / (2.0 * math.pi)


def sources_of(engine, depth=None):
    """[(A_object, theta_c, [(f_j, beta_j, theta_j)])] of a fresh engine -- the
    stationary configuration its analysis asks for, with zero phases."""
    index = float(engine.params.get('fm_depth', 1.0)) if depth is None else float(depth)
    out = []
    for item in engine._mods:
        if item is None:
            continue
        fr, am, amp = item
        fr = np.asarray(fr, dtype=float)
        am = np.asarray(am, dtype=float)
        live = (fr > 0.0) & (am > 0.0)
        out.append((float(amp), 0.0,
                    [(float(f), index * float(a), 0.0) for f, a in zip(fr[live], am[live])]))
    return out
