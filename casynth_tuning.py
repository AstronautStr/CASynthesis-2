"""Sethares dissonance-curve tuning for CASynth.

Implements the Plomp–Levelt dissonance model (Sethares 1993) to magnetise
note-on carrier frequency toward the consonance minima of the current colony
spectrum.  Four public functions + three module-level constants.

Formula parity: pair_dissonance matches frontiers_explainer/src/linalg.js:130-134
exactly (same float constants 0.24/0.021/19/3.5/5.75).

All functions are pure numpy — no pygame/sounddevice dependency.
"""

import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
TUNE_CURVE_STEPS  = 1300   # r-grid points over [r_min, r_max]; ~1¢/step
TUNE_MAX_PARTIALS = 24     # top-N partials by amplitude from colony spectrum
TUNE_SKIP_BELOW   = 1.02   # ignore minima below this ratio (~34¢); skip unison


# ── Core model ────────────────────────────────────────────────────────────────

def pair_dissonance(f1, a1, f2, a2):
    """Plomp–Levelt roughness for a single frequency pair.

    Parity with frontiers_explainer/src/linalg.js:130-134:
        fmin = min(f1,f2)  df = |f2-f1|
        s    = 0.24 / (0.021*fmin + 19)
        d    = min(a1,a2) * (exp(-3.5*s*df) - exp(-5.75*s*df))

    Parameters
    ----------
    f1, f2 : float  – frequencies in Hz (positive)
    a1, a2 : float  – amplitudes in [0, 1]

    Returns
    -------
    float – dissonance contribution (non-negative; 0 at f1==f2 or df→∞)
    """
    fmin = min(f1, f2)
    df   = abs(f2 - f1)
    s    = 0.24 / (0.021 * fmin + 19)
    return min(a1, a2) * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))


def dissonance_curve(freqs, amps, r_min=1.0, r_max=2.1, steps=TUNE_CURVE_STEPS):
    """Total Sethares dissonance D(r) for combined spectrum {S, r·S}.

    Matches JS dissonanceCurve(freqs, amps, alphaMin, alphaMax, steps=440)
    exactly (same pair enumeration: upper-triangle of the 2N×2N combined
    spectrum at each r).

    Vectorised three-group decomposition (no Python loop over r):
      SS pairs   – constant (independent of r), computed once.
      rSrS pairs – depend on r via frequency scaling, shape (P, R).
      Cross pairs – S×rS, all N×N combinations, shape (N, N, R).

    Memory: ~30 MB peak for N=24, R=1301 (acceptable for a note-on one-shot).

    Parameters
    ----------
    freqs : (N,) float array – partial frequencies in Hz
    amps  : (N,) float array – partial amplitudes in [0, 1]
    r_min, r_max : float – ratio range (default 1.0–2.1 = one octave + 1/10)
    steps : int – number of intervals; curve has (steps+1) points

    Returns
    -------
    ratios : (steps+1,) float array
    curve  : (steps+1,) float array – total dissonance at each ratio
    """
    freqs = np.asarray(freqs, dtype=float)
    amps  = np.asarray(amps,  dtype=float)
    N     = len(freqs)

    ratios = r_min + (r_max - r_min) * np.arange(steps + 1) / steps   # (R,)
    R      = len(ratios)

    if N == 0:
        return ratios, np.zeros(R)

    # Upper-triangle pair indices (p < q), P = N*(N-1)/2
    pi, pj = np.triu_indices(N, k=1)

    # ── SS pairs (constant, independent of r) ──
    if len(pi) > 0:
        fi_ss   = freqs[pi];   fj_ss   = freqs[pj]
        ai_ss   = amps[pi];    aj_ss   = amps[pj]
        fmin_ss = np.minimum(fi_ss, fj_ss)
        df_ss   = np.abs(fi_ss - fj_ss)
        s_ss    = 0.24 / (0.021 * fmin_ss + 19)
        d_ss    = float(
            (np.minimum(ai_ss, aj_ss)
             * (np.exp(-3.5 * s_ss * df_ss) - np.exp(-5.75 * s_ss * df_ss))).sum()
        )
    else:
        d_ss = 0.0

    # ── rSrS pairs (r-scaled spectrum, upper triangle) ──
    if len(pi) > 0:
        fi_p  = freqs[pi][:, None]   # (P, 1)
        fj_p  = freqs[pj][:, None]   # (P, 1)
        rr    = ratios[None, :]       # (1, R)
        rfip  = fi_p * rr             # (P, R)
        rfjp  = fj_p * rr             # (P, R)
        fmin_rr = np.minimum(rfip, rfjp)
        df_rr   = np.abs(rfip - rfjp)
        s_rr    = 0.24 / (0.021 * fmin_rr + 19)
        ma_rr   = np.minimum(amps[pi], amps[pj])[:, None]   # (P, 1)
        d_rr    = (ma_rr * (np.exp(-3.5 * s_rr * df_rr)
                            - np.exp(-5.75 * s_rr * df_rr))).sum(axis=0)  # (R,)
    else:
        d_rr = np.zeros(R)

    # ── Cross pairs S×rS (all N×N combinations) ──
    # In the 2N combined spectrum, p-indices [0,N) always precede q-indices [N,2N)
    # → all N*N cross pairs are (p, q) with p < q; no filtering needed.
    fi_c  = freqs[:, None, None]   # (N, 1, 1)
    fj_c  = freqs[None, :, None]   # (1, N, 1)
    rr_c  = ratios[None, None, :]  # (1, 1, R)
    rfj_c = fj_c * rr_c            # (1, N, R)

    ai_c  = amps[:, None, None]    # (N, 1, 1)
    aj_c  = amps[None, :, None]    # (1, N, 1)

    fmin_c = np.minimum(fi_c, rfj_c)                 # (N, N, R)
    df_c   = np.abs(fi_c - rfj_c)                    # (N, N, R)
    s_c    = 0.24 / (0.021 * fmin_c + 19)            # (N, N, R)
    ma_c   = np.minimum(ai_c, aj_c)                  # (N, N, 1) → broadcasts to (N, N, R)
    d_c    = (ma_c * (np.exp(-3.5 * s_c * df_c)
                      - np.exp(-5.75 * s_c * df_c))).sum(axis=(0, 1))  # (R,)

    curve = d_ss + d_rr + d_c
    return ratios, curve


# ── Minima detection ───────────────────────────────────────────────────────────

def scale_minima(curve, ratios, skip_below=TUNE_SKIP_BELOW):
    """Strict interior local minima of the dissonance curve.

    Matches JS localMinima condition (curve[i] <= neighbours with at least one
    neighbour gap > 1e-6) but WITHOUT the 6-cap.  Only minima with
    ratios[i] > skip_below are returned (skips the unison region).

    Parameters
    ----------
    curve  : (R,) float array – dissonance values from dissonance_curve()
    ratios : (R,) float array – corresponding ratio values
    skip_below : float – ignore minima at ratios ≤ this threshold

    Returns
    -------
    (M,) float array – ratio values at local minima, sorted ascending.
                       Empty array when no minima found.
    """
    minima = []
    for i in range(1, len(curve) - 1):
        if ratios[i] <= skip_below:
            continue
        d      = curve[i]
        d_prev = curve[i - 1]
        d_next = curve[i + 1]
        # Non-strict neighbour comparison + meaningful depth threshold
        if (d <= d_prev and d <= d_next
                and (d_prev - d > 1e-6 or d_next - d > 1e-6)):
            minima.append(float(ratios[i]))
    return np.array(minima)


# ── Geometric snap ─────────────────────────────────────────────────────────────

def snap_ratio(r_raw, minima, tune):
    """Magnetise r_raw toward the nearest dissonance minimum.

    Octave fold: map r_raw to [1, 2) by dividing by 2^k (k = floor(log2(r_raw))).
    When the fold pushes r_folded below TUNE_SKIP_BELOW (e.g. r_raw=2.0 → 1.0),
    no snap is applied — the interval is returned unchanged (conservative: exact
    octaves and other powers of 2 are stable without tuning interference).

    Geometric snap (linear in cents):
        cents_snapped = cents_folded + tune * (cents_nearest - cents_folded)
        r_snapped     = r_folded * (r_nearest / r_folded)^tune

    Parameters
    ----------
    r_raw  : float – raw interval ratio (target_f / prev_carrier_f); any positive value
    minima : (M,) float array – consonance snap targets from scale_minima()
    tune   : float in [0, 1] – 0 = transparent (no snap), 1 = full snap

    Returns
    -------
    float – snapped ratio; equals r_raw when tune=0 or minima is empty
    """
    r_raw = float(r_raw)
    if tune <= 0.0 or len(minima) == 0 or r_raw <= 0.0:
        return r_raw

    # Octave fold: k = floor(log2(r_raw))
    k        = int(np.floor(np.log2(r_raw)))
    r_folded = r_raw / (2.0 ** k)

    # Guard: if the fold pushed us below the skip threshold, no snap makes sense
    # (e.g. r_raw=2.0 → r_folded=1.0 < TUNE_SKIP_BELOW; exact octaves stay intact).
    if r_folded < TUNE_SKIP_BELOW:
        return r_raw

    # Find nearest minimum by cents distance (geometric = linear in cent space)
    cents_folded = 1200.0 * np.log2(r_folded)
    cents_minima = 1200.0 * np.log2(np.asarray(minima, dtype=float))
    nearest_idx  = int(np.argmin(np.abs(cents_minima - cents_folded)))
    r_nearest    = float(minima[nearest_idx])

    # Geometric interpolation: r_snapped_folded = r_folded * (r_nearest/r_folded)^tune
    r_snapped_folded = r_folded * ((r_nearest / r_folded) ** tune)

    # Undo octave fold and return
    return r_snapped_folded * (2.0 ** k)
