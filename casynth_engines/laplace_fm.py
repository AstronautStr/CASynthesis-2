"""Laplace FM (REQ memory/req-laplace-fm-2026-09-20.md): the modes of a figure
modulate ONE sine carrier of that figure.

Every figure of the field gets one carrier at the note frequency f0.  The
Laplacian modes of THAT figure are its modulators: mode j becomes a sine at
f_j whose modulation index is beta_j = I * a_j (I = the `FM depth` knob).  The
modulators enter the SAME phase (never separate carriers, never each other),
and the outputs of the figures are summed.  A common carrier for the whole
field is deliberately NOT part of this trial.

Law (REQ section 2), written as phase modulation -- the standard form of musical
digital FM, with the index in radians:

    beta_j     = I * a_j
    theta_c'   = 2*pi*f0        theta_j' = 2*pi*f_j
    A_object   = sqrt(sum_j a_j^2)                  (over the non-zero modes)
    y_object   = A_object * sin(theta_c + sum_j beta_j * sin(theta_j))
    y          = DC( LP( sum_objects y_object ) )

`beta_j` is a phase deviation in radians; in a stationary state one mode
contributes a peak frequency deviation of beta_j*f_j Hz.  Adding I*a_j Hz to
the frequency instead would be a different law (its index falling as 1/f_j) and
is NOT what this engine does.  While the weights move, the phase formula above
is what runs -- with continuous phases and smoothed indices; the derivative of a
moving index is part of the instantaneous frequency and no claim of equality
with an integrated arbitrary FM signal in Hz is made.  Reference for the naming
and the index: Julius O. Smith, "Sinusoidal Frequency Modulation (FM)",
https://dsprelated.com/freebooks/mdft/Sinusoidal_Frequency_Modulation_FM.html

Nothing is normalised away: the index is never divided by the number of modes,
the sum of the weights or the RMS of the modulators, and the weights are read
BEFORE the bench gain and the record calibration (Gain never reaches the
timbre).  Coincident mode frequencies stay coherent -- two equal modulators
sharing a phase simply add their indices.

  FM depth = 0 leaves the sine carrier of every sounding figure (its own scale
  and envelope), NOT the old Laplacian sum of sines.  A figure with no non-zero
  mode -- an empty field or a single cell -- stays silent even at I = 0.

Band limit (REQ section 4).  Even sine modulators create new lines at
f0 + sum_j n_j*f_j with products of Bessel coefficients, and the highest
instantaneous frequency is not a bound of that spectrum.  The sum of the
carriers is therefore rendered at OVERSAMPLE x sr and brought down through one
linear-phase Kaiser FIR (flat to 0.40*sr, -118 dB from 0.50*sr, constant delay
FIR_DELAY_OUT = 39 output samples at ANY oversampling).  Indices and mode
frequencies are never quietly reduced to save the budget.

DC (REQ section 3).  PM puts a constant term wherever a combination frequency
lands on zero with a phase that does not cancel (a modulator at exactly f0 whose
phase has run away from the carrier's -- the Jam does produce this).  A FIXED
blocker, with no user knob, runs on the summed FM output after the decimation
and before the master gain:  y[n] = x[n] - x[n-1] + r*y[n-1], r = exp(-2*pi*5/sr).

Time behaviour (REQ section 3) -- the baseline's own scales, no new excitation:
  - a figure that persists keeps its carrier phase; knobs never reset the field
    or an audio phase; a NEW source starts from phase zero, deterministically.
  - mode frequencies never glide.  A changed mode moves its OLD modulator into a
    modulator tail that rings its index down over the baseline release while the
    new one enters over the baseline attack -- the amplitude transitions of the
    old spectrum, applied here to the indices INSIDE the one phase sum.
  - a figure that disappears releases its carrier output while the tail keeps the
    indices and goes on advancing every modulator phase (never a bare sine).
  - I and A_object glide to their targets on the pool's own per-block ramp and
    are not restarted while a state persists.
Tail rule (both pools): a free slot first -- not releasing and already silent --
otherwise the quietest one is stolen and counted (display / snapshot `steals`).
"""
import math

import numpy as np

from casynth_config import (SR, TWO_PI, _RAMP, MAX_VOICES, MAX_MODES_PER_OBJ,
                            TAIL_MIN_AMP,
                            GEN_ATTACK_DEFAULT, GEN_DECAY_DEFAULT,
                            GEN_SUSTAIN_DEFAULT, GEN_RELEASE_DEFAULT)
from casynth_core import ENGINE_BY_ID
from casynth_engine import analyse
from .engine_api import SoundEngine
from .legacy_engine import PAN_CENTER, _gen_chunks
from .registry import EngineSpec, register

ENGINE_ID = 'laplace_fm'
LABEL = 'Laplace FM'

# the seven spectrum settings of the old Laplace, metadata from the core registry
LAPLACE_PARAMS = tuple(tuple(p) for p in ENGINE_BY_ID['laplacian']['params'])
SPECTRUM_KEYS = tuple(p[0] for p in LAPLACE_PARAMS)   # n spread alpha shape harm fullshape dyn

PARAMS = [('fm_depth', 'FM depth', 0.0, 4.0, False, 1.0)] + list(LAPLACE_PARAMS)
# absent from an older parameter set / snapshot -> these defaults
OPTIONAL_PARAMS = {'fm_depth': 1.0}
OPTIONAL_PARAMS.update({p[0]: p[5] for p in LAPLACE_PARAMS})

# ── the output band (REQ section 4) ───────────────────────────────────────────
OVERSAMPLE = 8                # the FM sum is built at OVERSAMPLE * sr
BAND_PASS = 0.40              # * sr: kept flat
BAND_STOP = 0.50              # * sr: rejected (anything above aliases into the band)
FIR_DELAY_OUT = 39            # group delay in OUTPUT samples -- the same at every
                              # oversampling, so a higher-rate reference aligns exactly
FIR_ATT_DB = 120.0            # Kaiser design attenuation (measured: -118 dB from 0.50*sr)
DC_HZ = 5.0                   # fixed DC blocker pole (no user knob)

# ── pool sizes / thresholds ───────────────────────────────────────────────────
N_FM_TAILS = MAX_VOICES * 4       # released carriers ringing out (with their modulators)
# Per source: old modulators ringing their index down.  Dragging a spectrum knob moves
# EVERY mode of a figure on EVERY block, and each one rings for `release` blocks, so one
# figure can want `n x release` tails at once -- 60 at the widest spectrum on a 4
# generations/s scene.  It was MAX_MODES_PER_OBJ until 2026-09-21, and a full pool used to
# overwrite a tail that was still sounding: that step is the click heard while tweaking
# `harm`.  Beyond the pool the engine now fades a modulator out where it stands.
N_MOD_TAILS = MAX_MODES_PER_OBJ * 3
N_MODS = MAX_MODES_PER_OBJ + N_MOD_TAILS
AMP_EPS = 1e-4                    # a source below this on both ends of the block is skipped
BETA_EPS = 1e-9                   # an index below this is not worth a sine (-180 dB)
BETA_TAIL_MIN = 1e-4              # below this an old modulator is not worth a tail (-80 dB step)
BETA_FREE = 1e-4                  # a spent modulator tail this quiet frees its slot
# Column chunking of the render (speed only, never the values): aim for this many
# (modulator x sample) cells per pass, never fewer than this many samples.
CHUNK_CELLS = 12000
CHUNK_MIN = 128


# ── the output filter ─────────────────────────────────────────────────────────

def fir_taps(oversample=OVERSAMPLE):
    """Length of the decimation FIR: odd, and a multiple of 2*oversample plus one,
    so the group delay is exactly FIR_DELAY_OUT OUTPUT samples."""
    return 2 * int(oversample) * FIR_DELAY_OUT + 1


def fir_kernel(oversample=OVERSAMPLE, sr=SR):
    """Linear-phase Kaiser low-pass for OVERSAMPLE*sr -> sr: unity DC gain, flat to
    0.40*sr, -118 dB from 0.50*sr (measured), no resonant colouring."""
    n = fir_taps(oversample)
    fs = float(oversample) * float(sr)
    fc = 0.5 * (BAND_PASS + BAND_STOP) * float(sr)          # 0.45*sr, the -6 dB point
    m = np.arange(n, dtype=float) - (n - 1) / 2.0
    h = 2.0 * (fc / fs) * np.sinc(2.0 * (fc / fs) * m)
    h = h * np.kaiser(n, 0.1102 * (FIR_ATT_DB - 8.7))
    return h / h.sum()


_KERNELS = {}


def _kernel(oversample, sr):
    key = (int(oversample), float(sr))
    got = _KERNELS.get(key)
    if got is None:
        got = _KERNELS[key] = fir_kernel(oversample, sr)
    return got


def decimate(y_os, kernel, oversample, history):
    """One block: (out, new history).  `history` holds the last len(kernel)-1
    oversampled samples, so the filter runs across block boundaries."""
    buf = np.concatenate((history, y_os))
    w = np.lib.stride_tricks.sliding_window_view(buf, len(kernel))[::int(oversample)]
    return w @ kernel, buf[1 - len(kernel):].copy()


_DC_POW = {}


def _dc_powers(n, r):
    """(r**k, r**-k) for k = 0..n-1 -- the closed form of the blocker's recursion
    (r**-351 = 1.285: perfectly conditioned at this pole)."""
    key = (int(n), float(r))
    got = _DC_POW.get(key)
    if got is None:
        k = np.arange(int(n), dtype=float)
        got = _DC_POW[key] = (np.power(r, k), np.power(r, -k))
    return got


def dc_block(x, r, x_prev, y_prev):
    """y[n] = x[n] - x[n-1] + r*y[n-1] over one block: (y, x[-1], y[-1]).

    Evaluated in closed form (y[n] = r**(n+1)*y_prev + sum_k r**(n-k)*d[k]), which
    is the same filter as the scalar recursion -- tests check both agree to 1e-15."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy(), x_prev, y_prev
    d = np.empty(x.shape)
    d[0] = x[0] - x_prev
    d[1:] = x[1:] - x[:-1]
    p, q = _dc_powers(x.size, r)
    y = p * (np.cumsum(d * q) + r * y_prev)
    return y, float(x[-1]), float(y[-1])


# ── the carriers and their modulators ─────────────────────────────────────────

class _FMSources:
    """One source per voice channel (carrier + its modulator slots) plus a pool of
    released sources ringing out.  Every rule below is the baseline SlotPool's,
    with the per-mode AMPLITUDE of the old spectrum reading as the per-mode INDEX
    inside one phase sum."""

    def __init__(self, f0, sr, oversample=OVERSAMPLE):
        self.f0 = float(f0)
        self.sr = float(sr)
        self.oversample = int(oversample)
        self.sr_os = float(sr) * float(oversample)
        self.guard = 0.5 * self.sr_os          # nothing above the oversampled Nyquist
        n = MAX_VOICES + N_FM_TAILS
        self.n_src = n
        self.th_c = np.zeros(n)                # carrier phase
        self.amp_cur = np.zeros(n)
        self.amp_tgt = np.zeros(n)
        self.env_phase = np.zeros(n, dtype=int)
        self.env_level = np.zeros(n)
        self.alive = np.zeros(MAX_VOICES, dtype=bool)
        self.rel_cnt = np.zeros(n, dtype=int)
        self.rel_len = np.ones(n, dtype=int)
        self.rel_amp0 = np.zeros(n)
        self.f_mod = np.zeros((n, N_MODS))     # 0 = the slot is free
        self.th_mod = np.zeros((n, N_MODS))
        self.beta_cur = np.zeros((n, N_MODS))
        self.beta_tgt = np.zeros((n, N_MODS))
        self.m_env_phase = np.zeros((n, N_MODS), dtype=int)
        self.m_env_level = np.zeros((n, N_MODS))
        self.m_rel_cnt = np.zeros((n, N_MODS), dtype=int)
        self.m_rel_len = np.ones((n, N_MODS), dtype=int)
        self.m_rel_beta0 = np.zeros((n, N_MODS))
        self.f_prev = np.zeros((MAX_VOICES, MAX_MODES_PER_OBJ))
        self.steals = 0
        self.mod_steals = 0        # carriers: the quietest tail taken (never in a delivered scene)
        self.mod_inplace = 0      # modulators: faded where they stood, the pool being full

    # -- envelopes (the SlotPool rule) -------------------------------------------
    @staticmethod
    def _env(phase, level, v, attack_chunks, decay_chunks, sustain):
        ph = phase[v]
        if ph == 1:
            lvl = level[v] + 1.0 / attack_chunks
            if lvl >= 1.0:
                lvl, phase[v] = 1.0, 2
            level[v] = lvl
        elif ph == 2:
            lvl = level[v] - (1.0 - sustain) / decay_chunks
            if lvl <= sustain:
                lvl, phase[v] = sustain, 3
            level[v] = lvl
        elif ph == 3:
            level[v] = sustain

    # -- tails --------------------------------------------------------------------
    def _acquire_source_tail(self):
        lo, hi = MAX_VOICES, self.n_src
        quietest, quiet_amp = lo, np.inf
        for s in range(lo, hi):
            if self.rel_cnt[s] == 0 and self.amp_cur[s] < AMP_EPS:
                return s
            if self.amp_cur[s] < quiet_amp:
                quiet_amp, quietest = self.amp_cur[s], s
        self.steals += 1
        return quietest

    def _acquire_mod_tail(self, s):
        """A free modulator tail of source `s`, or -1 when the pool is full.

        A still-sounding tail is NEVER taken: its index would leave the phase sum in one
        step, which is exactly the click heard on 2026-09-21 while `harm` was dragged
        (dragging a spectrum knob moves every mode on every block, so one figure needs
        `n x release` tails at once).  The caller fades the old modulator out in place
        instead -- see _release_modulator_in_place."""
        for m in range(MAX_MODES_PER_OBJ, N_MODS):
            if self.m_rel_cnt[s, m] == 0 and abs(float(self.beta_cur[s, m])) < BETA_FREE:
                return m
        return -1

    def _start_source(self, v):
        """A NEW figure on this channel: deterministic zero phases, no leftovers."""
        self.th_c[v] = 0.0
        self.th_mod[v] = 0.0
        self.f_mod[v] = 0.0
        self.beta_cur[v] = 0.0
        self.beta_tgt[v] = 0.0
        self.m_env_phase[v] = 0
        self.m_env_level[v] = 0.0
        self.m_rel_cnt[v] = 0
        self.m_rel_len[v] = 1
        self.m_rel_beta0[v] = 0.0
        self.f_prev[v] = 0.0
        self.amp_cur[v] = 0.0
        self.env_phase[v] = 1
        self.env_level[v] = 0.0

    def _clear_channel(self, v):
        self.amp_cur[v] = 0.0
        self.amp_tgt[v] = 0.0
        self.env_phase[v] = 0
        self.env_level[v] = 0.0
        self.alive[v] = False
        self.f_mod[v] = 0.0
        self.beta_cur[v] = 0.0
        self.beta_tgt[v] = 0.0
        self.m_env_phase[v] = 0
        self.m_env_level[v] = 0.0
        self.m_rel_cnt[v] = 0
        self.m_rel_len[v] = 1
        self.m_rel_beta0[v] = 0.0
        self.f_prev[v] = 0.0

    # -- one block of targets ------------------------------------------------------
    def update(self, mods, index, release_chunks, attack_chunks, decay_chunks, sustain):
        """mods[v] = (freqs, amps, A_object) of channel v, or None when it is silent;
        `index` = the FM depth knob I."""
        index = float(index)
        for v in range(MAX_VOICES):
            item = mods[v] if v < len(mods) else None
            if item is None:
                self._release_channel(v, release_chunks)
                continue
            fr, am, scale = item
            fresh = not self.alive[v]
            if fresh:
                self._start_source(v)
            self._env(self.env_phase, self.env_level, v, attack_chunks, decay_chunks, sustain)
            self.amp_tgt[v] = self.env_level[v] * float(scale)
            self.alive[v] = True
            n_fr, n_am = len(fr), len(am)
            for m in range(MAX_MODES_PER_OBJ):
                f_new = float(fr[m]) if m < n_fr else 0.0
                a_new = float(am[m]) if m < n_am else 0.0
                if not (f_new > 0.0) or not np.isfinite(f_new) or f_new >= self.guard:
                    f_new, a_new = 0.0, 0.0
                beta_new = index * a_new
                if abs(f_new - self.f_prev[v, m]) <= 0.01:      # STABLE mode
                    self.f_mod[v, m] = f_new
                    if f_new > 0.0:
                        self._env_mod(v, m, attack_chunks, decay_chunks, sustain)
                        self.beta_tgt[v, m] = self.m_env_level[v, m] * beta_new
                    else:
                        self.m_env_phase[v, m] = 0
                        self.m_env_level[v, m] = 0.0
                        self.beta_tgt[v, m] = 0.0
                    self.f_prev[v, m] = f_new
                    continue
                # the mode changed: ring the OLD modulator's index out in a tail slot
                # (its phase continues there), start the new one from index 0
                if abs(self.beta_cur[v, m]) > BETA_TAIL_MIN and self.f_mod[v, m] > 0.0:
                    t = self._acquire_mod_tail(v)
                    if t < 0:
                        # the pool is full: fade this modulator out where it is, on the
                        # OLD frequency, over this one block, and take the new frequency
                        # on the next one (f_prev is deliberately left alone).  A block
                        # of glide instead of a step -- never an index cut in two.
                        self.m_env_phase[v, m] = 0
                        self.m_env_level[v, m] = 0.0
                        self.beta_tgt[v, m] = 0.0
                        self.mod_inplace += 1
                        continue
                    self.f_mod[v, t] = self.f_mod[v, m]
                    self.th_mod[v, t] = self.th_mod[v, m]
                    self.beta_cur[v, t] = self.beta_cur[v, m]
                    self.beta_tgt[v, t] = self.beta_cur[v, m]   # stepped down below
                    self.m_rel_cnt[v, t] = release_chunks
                    self.m_rel_len[v, t] = release_chunks
                    self.m_rel_beta0[v, t] = float(self.beta_cur[v, m])
                    self.m_env_phase[v, t] = 0
                    self.m_env_level[v, t] = 0.0
                self.beta_cur[v, m] = 0.0
                self.f_mod[v, m] = f_new
                if f_new > 0.0:
                    self.m_env_phase[v, m] = 1                  # (re)trigger -> attack
                    self.m_env_level[v, m] = 0.0
                    self._env_mod(v, m, attack_chunks, decay_chunks, sustain)
                    self.beta_tgt[v, m] = self.m_env_level[v, m] * beta_new
                else:
                    self.m_env_phase[v, m] = 0
                    self.m_env_level[v, m] = 0.0
                    self.beta_tgt[v, m] = 0.0
                self.f_prev[v, m] = f_new
            if fresh:
                # nothing is audible yet (the carrier fades in), so the colour is set
                # rather than glided -- the Filter rule of the carriers engine
                self.beta_cur[v] = self.beta_tgt[v]
        self._advance_mod_tails()
        self._advance_source_tails()

    def _env_mod(self, v, m, attack_chunks, decay_chunks, sustain):
        ph = self.m_env_phase[v, m]
        if ph == 1:
            lvl = self.m_env_level[v, m] + 1.0 / attack_chunks
            if lvl >= 1.0:
                lvl, self.m_env_phase[v, m] = 1.0, 2
            self.m_env_level[v, m] = lvl
        elif ph == 2:
            lvl = self.m_env_level[v, m] - (1.0 - sustain) / decay_chunks
            if lvl <= sustain:
                lvl, self.m_env_phase[v, m] = sustain, 3
            self.m_env_level[v, m] = lvl
        elif ph == 3:
            self.m_env_level[v, m] = sustain

    def _release_channel(self, v, release_chunks):
        """The figure is gone: move the WHOLE source (carrier phase, every modulator
        phase and index, any modulator release under way) into a tail that rings the
        carrier output down -- never a bare, unmodulated sine."""
        if self.alive[v] and self.amp_cur[v] > TAIL_MIN_AMP:
            t = self._acquire_source_tail()
            self.th_c[t] = self.th_c[v]
            self.amp_cur[t] = self.amp_cur[v]
            self.amp_tgt[t] = self.amp_cur[v]
            self.f_mod[t] = self.f_mod[v]
            self.th_mod[t] = self.th_mod[v]
            self.beta_cur[t] = self.beta_cur[v]
            self.beta_tgt[t] = self.beta_cur[v]      # frozen indices, no step
            self.m_rel_cnt[t] = self.m_rel_cnt[v]
            self.m_rel_len[t] = self.m_rel_len[v]
            self.m_rel_beta0[t] = self.m_rel_beta0[v]
            self.m_env_phase[t] = 0
            self.m_env_level[t] = 0.0
            self.rel_cnt[t] = release_chunks
            self.rel_len[t] = release_chunks
            self.rel_amp0[t] = float(self.amp_cur[v])
            self.env_phase[t] = 0
            self.env_level[t] = 0.0
        self._clear_channel(v)

    def _advance_mod_tails(self):
        """One release step of every modulator tail (of live channels AND of released
        sources), after the assignment -- so a tail spawned this block decays at once."""
        rr, cc = np.nonzero(self.m_rel_cnt[:, MAX_MODES_PER_OBJ:] > 0)
        if not len(rr):
            return
        cc = cc + MAX_MODES_PER_OBJ
        cnt = self.m_rel_cnt[rr, cc] - 1
        self.m_rel_cnt[rr, cc] = cnt
        self.beta_tgt[rr, cc] = self.m_rel_beta0[rr, cc] * cnt / self.m_rel_len[rr, cc]
        free = (cnt == 0) & (np.abs(self.beta_cur[rr, cc]) < BETA_FREE)
        if free.any():
            self.f_mod[rr[free], cc[free]] = 0.0

    def _advance_source_tails(self):
        s = np.arange(MAX_VOICES, self.n_src)
        s = s[self.rel_cnt[s] > 0]
        if not len(s):
            return
        cnt = self.rel_cnt[s] - 1
        self.rel_cnt[s] = cnt
        self.amp_tgt[s] = self.rel_amp0[s] * cnt / self.rel_len[s]
        free = s[(cnt == 0) & (self.amp_cur[s] < 0.01)]
        if len(free):
            self.f_mod[free] = 0.0
            self.beta_cur[free] = 0.0
            self.beta_tgt[free] = 0.0
            self.m_rel_cnt[free] = 0

    # -- one block of audio --------------------------------------------------------
    def render(self, n_os, ramp_os):
        """Mono FM sum at OVERSAMPLE*sr: sum over the sounding sources of
        A(t)*sin(theta_c + sum_j beta_j(t)*sin(theta_j)), A and every beta glided
        across the block on the pool's own ramp.

        Evaluated in COLUMN chunks (2026-09-20).  One modulator per row and the whole
        block per column is 1.6 MB of temporaries at the modulator counts a dragged
        spectrum knob produces (every mode changes frequency every block, so every
        modulator also has its tails sounding), and the block then costs more in memory
        traffic than in arithmetic.  Chunking the SAMPLES keeps the working set in
        cache; every output sample is still computed by exactly the same operations in
        exactly the same order, so the result is bit-for-bit what the full-width
        version produced (the pinned records replay unchanged)."""
        act = np.nonzero((self.amp_cur >= AMP_EPS) | (self.amp_tgt >= AMP_EPS))[0]
        if not len(act):
            self.advance(n_os)
            return np.zeros(n_os)
        idx = np.arange(n_os, dtype=float)
        fm = self.f_mod[act]
        b0 = self.beta_cur[act]
        db = self.beta_tgt[act] - b0
        live = (fm > 0.0) & ((np.abs(b0) > BETA_EPS) | (np.abs(b0 + db) > BETA_EPS))
        rows = cols = uniq = starts = None
        inc = th_m = bb0 = bdb = None
        if live.any():
            rows, cols = np.nonzero(live)                     # rows ascending
            inc = (TWO_PI / self.sr_os) * fm[rows, cols]
            th_m = self.th_mod[act][rows, cols]
            bb0 = b0[rows, cols]
            bdb = db[rows, cols]
            uniq, starts = np.unique(rows, return_index=True)
        inc_c = TWO_PI * self.f0 / self.sr_os
        th_c = self.th_c[act]
        a0 = self.amp_cur[act]
        da = self.amp_tgt[act] - a0
        y = np.empty(n_os)
        chunk = self._chunk(0 if rows is None else len(rows), n_os)
        for c0 in range(0, n_os, chunk):
            sl = slice(c0, min(c0 + chunk, n_os))
            idx_c = idx[sl]
            ramp_c = ramp_os[sl]
            acc = np.zeros((len(act), len(idx_c)))
            if rows is not None:
                s = np.sin(th_m[:, None] + inc[:, None] * idx_c[None, :])
                s *= bb0[:, None] + bdb[:, None] * ramp_c[None, :]
                acc[uniq] = np.add.reduceat(s, starts, axis=0)
            y[sl] = ((a0[:, None] + da[:, None] * ramp_c[None, :])
                     * np.sin(th_c[:, None] + inc_c * idx_c[None, :] + acc)).sum(axis=0)
        self.advance(n_os)
        return y

    @staticmethod
    def _chunk(n_rows, n_os):
        """Samples per pass: enough rows x columns to keep the vector units busy, few
        enough to stay in cache.  Only the SPEED depends on this number."""
        if n_rows <= 0:
            return n_os
        return int(min(n_os, max(CHUNK_MIN, CHUNK_CELLS // n_rows)))

    def advance(self, n_os):
        """Commit the block: every phase moves on (a free-running oscillator per
        carrier and per modulator), the glided values reach their targets."""
        self.th_c[:] = (self.th_c + (TWO_PI * self.f0 / self.sr_os) * n_os) % TWO_PI
        m = self.f_mod > 0.0
        if m.any():
            self.th_mod[m] = (self.th_mod[m]
                              + (TWO_PI / self.sr_os) * self.f_mod[m] * n_os) % TWO_PI
        self.amp_cur[:] = self.amp_tgt
        self.beta_cur[:] = self.beta_tgt

    # -- diagnostics ----------------------------------------------------------------
    def counts(self):
        sounding = int(np.count_nonzero(self.alive))
        tails = int(np.count_nonzero(self.rel_cnt[MAX_VOICES:] > 0))
        mods = int(np.count_nonzero((self.f_mod[:MAX_VOICES, :MAX_MODES_PER_OBJ] > 0.0)
                                    & (np.abs(self.beta_tgt[:MAX_VOICES, :MAX_MODES_PER_OBJ])
                                       > BETA_EPS)))
        mod_tails = int(np.count_nonzero(self.m_rel_cnt[:, MAX_MODES_PER_OBJ:] > 0))
        return sounding, tails, mods, mod_tails


# ── the engine ────────────────────────────────────────────────────────────────

class LaplaceFMEngine(SoundEngine):
    """The Laplacian analysis of the baseline, sounded as one FM carrier per figure."""

    STATE_VERSION = 2          # 2: the wider modulator tail pool (2026-09-21); a version 1
                               # state is accepted and widened -- see restore_state

    def __init__(self, ctx, params, oversample=OVERSAMPLE):
        """`oversample` is OVERSAMPLE for the registered engine; a gate builds the SAME
        engine at a higher rate as the converged reference of section 4 (a state made at
        one rate is refused at another)."""
        super().__init__(ctx, params)
        self.engine_id = ENGINE_ID
        interval = 1.0 / ctx.rate_hz
        self._release_chunks = _gen_chunks(GEN_RELEASE_DEFAULT, interval)
        self._attack_chunks = _gen_chunks(GEN_ATTACK_DEFAULT, interval)
        self._decay_chunks = _gen_chunks(GEN_DECAY_DEFAULT, interval)
        self._sustain = float(GEN_SUSTAIN_DEFAULT)
        self.oversample = int(oversample)
        self.kernel = _kernel(self.oversample, ctx.sr)
        self.taps = len(self.kernel)
        self.dc_r = float(math.exp(-TWO_PI * DC_HZ / float(ctx.sr)))
        n_os = int(ctx.block) * self.oversample
        self._ramp_os = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_os))) * 0.5
        self.voices = []
        self._grid = None
        self._exc = None
        self._clear_audio(gain=0.0)

    # -- parameters ----------------------------------------------------------------
    def _p(self, name):
        return self.params.get(name, OPTIONAL_PARAMS[name])

    def _index(self):
        return float(self._p('fm_depth'))

    def _spectrum_params(self):
        return {k: self._p(k) for k in SPECTRUM_KEYS}

    # -- SoundEngine ---------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._clear_audio(gain)
        self.update_field(grid, exc)

    def update_field(self, grid, exc):
        self._grid, self._exc = grid, exc
        self._analyse()

    def set_params(self, params):
        """The analysis is deferred to the next render (2026-09-20).

        The bench drains every pending mouse event at 60 Hz and the runner applies all
        the commands that are due in ONE block, so dragging a spectrum knob delivers
        several `set_param` commands per block -- each of which used to re-analyse the
        whole field (the most expensive thing the engine does).  Only the LAST value of
        a block can reach the audio, so analysing once, at the render, is the same
        spectrum bit for bit and costs one analysis instead of four."""
        super().set_params(params)
        if self._grid is not None:
            self._pending_analysis = True

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        self._check_transpose(transpose)
        if gain_prev is not None:
            self.gain_prev = float(gain_prev)      # the host overrides the glide start
        n = self.ctx.block
        if self._pending_analysis:
            self._analyse()
        self.src.update(self._mods, self._index(), self._release_chunks,
                        self._attack_chunks, self._decay_chunks, self._sustain)
        y_os = self.src.render(n * self.oversample, self._ramp_os)
        y, self.fir_hist = decimate(y_os, self.kernel, self.oversample, self.fir_hist)
        y, self.dc_x, self.dc_y = dc_block(y, self.dc_r, self.dc_x, self.dc_y)
        self.mono = y            # read-only diagnostic: the float FM sum of this block,
                                 # after the band limit and the DC blocker, before pan,
                                 # gain and int16 -- what the gates compare with the
                                 # high-rate reference (never read back by the sound)
        pan = PAN_CENTER * np.pi / 2.0
        return self._finish(y * math.cos(pan), y * math.sin(pan), gain)

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

    def display(self):
        """Read-only numbers for the bench panel (never part of the sound)."""
        sounding, tails, mods, mod_tails = self.src.counts()
        rows = []
        for i, item in enumerate(self._mods):
            if item is None:
                continue
            fr, am, scale = item
            live = (np.asarray(fr) > 0.0) & (np.asarray(am) > 0.0)
            rows.append(dict(n=int(np.count_nonzero(live)),
                             f_low=float(np.asarray(fr)[live][0]) if live.any() else 0.0,
                             beta=float(self._index() * np.asarray(am)[live].max()) if live.any() else 0.0,
                             a=float(scale)))
        return dict(fm=True, f0=float(self.ctx.f0), depth=self._index(),
                    mod_inplace=int(self.src.mod_inplace),
                    oversample=int(self.oversample), taps=int(self.taps),
                    delay_samples=int(FIR_DELAY_OUT),
                    voices=int(len(self.voices)), sounding=sounding, tails=tails,
                    modulators=mods, mod_tails=mod_tails,
                    steals=int(self.src.steals), mod_steals=int(self.src.mod_steals),
                    lines=rows)

    # -- snapshot --------------------------------------------------------------------
    _SRC_ARRAYS = ('th_c', 'amp_cur', 'amp_tgt', 'env_phase', 'env_level', 'alive',
                   'rel_cnt', 'rel_len', 'rel_amp0', 'f_mod', 'th_mod', 'beta_cur',
                   'beta_tgt', 'm_env_phase', 'm_env_level', 'm_rel_cnt', 'm_rel_len',
                   'm_rel_beta0', 'f_prev')

    def export_state(self):
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  params=dict(self.params), gain_prev=float(self.gain_prev),
                  voices=int(MAX_VOICES), modes=int(MAX_MODES_PER_OBJ),
                  tails=int(N_FM_TAILS), mod_slots=int(N_MODS),
                  oversample=int(self.oversample), taps=int(self.taps),
                  f0=float(self.ctx.f0), dc_r=float(self.dc_r),
                  dc_x=float(self.dc_x), dc_y=float(self.dc_y),
                  fir_hist=self.fir_hist.copy(),
                  steals=int(self.src.steals), mod_steals=int(self.src.mod_steals),
                  mod_inplace=int(self.src.mod_inplace))
        for name in self._SRC_ARRAYS:
            st['src_' + name] = getattr(self.src, name).copy()
        return st

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') not in (1, self.STATE_VERSION):
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        state = _widen_v1(state)
        if state.get('engine_id') != self.engine_id:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {self.engine_id!r}")
        if (state.get('voices'), state.get('modes'), state.get('tails'), state.get('mod_slots')) != \
                (MAX_VOICES, MAX_MODES_PER_OBJ, N_FM_TAILS, N_MODS):
            raise ValueError(f"engine {ENGINE_ID}: pool layout of the state does not match this build")
        if (int(state.get('oversample', -1)), int(state.get('taps', -1))) != \
                (self.oversample, self.taps):
            raise ValueError(f"engine {ENGINE_ID}: the state carries oversampling "
                             f"{state.get('oversample')}/{state.get('taps')} taps, this build has "
                             f"{self.oversample}/{self.taps}")
        n = MAX_VOICES + N_FM_TAILS
        want = {'src_th_c': (n,), 'src_amp_cur': (n,), 'src_amp_tgt': (n,),
                'src_env_phase': (n,), 'src_env_level': (n,), 'src_alive': (MAX_VOICES,),
                'src_rel_cnt': (n,), 'src_rel_len': (n,), 'src_rel_amp0': (n,),
                'src_f_mod': (n, N_MODS), 'src_th_mod': (n, N_MODS),
                'src_beta_cur': (n, N_MODS), 'src_beta_tgt': (n, N_MODS),
                'src_m_env_phase': (n, N_MODS), 'src_m_env_level': (n, N_MODS),
                'src_m_rel_cnt': (n, N_MODS), 'src_m_rel_len': (n, N_MODS),
                'src_m_rel_beta0': (n, N_MODS),
                'src_f_prev': (MAX_VOICES, MAX_MODES_PER_OBJ),
                'fir_hist': (self.taps - 1,)}
        arrays = {}
        for name, shape in want.items():
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != shape:
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or shape "
                                 f"{getattr(a, 'shape', None)} != {shape}")
            if a.dtype != bool and not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            arrays[name] = a
        self._clear_audio(float(state['gain_prev']))
        for name in self._SRC_ARRAYS:
            getattr(self.src, name)[:] = arrays['src_' + name]
        self.src.steals = int(state.get('steals', 0))
        self.src.mod_steals = int(state.get('mod_steals', 0))
        self.src.mod_inplace = int(state.get('mod_inplace', 0))
        self.fir_hist[:] = arrays['fir_hist']
        self.dc_x = float(state['dc_x'])
        self.dc_y = float(state['dc_y'])
        self.params = dict(state['params'])
        self.update_field(grid, exc)

    # -- internals ---------------------------------------------------------------------
    def _clear_audio(self, gain):
        self._pending_analysis = False
        self.src = _FMSources(self.ctx.f0, self.ctx.sr, self.oversample)
        self.fir_hist = np.zeros(self.taps - 1)
        self.dc_x = 0.0
        self.dc_y = 0.0
        self.mono = np.zeros(self.ctx.block)
        self.gain_prev = gain
        self._mods = [None] * MAX_VOICES

    def _analyse(self):
        """voices = the baseline analysis; a channel carries (freqs, amps, A_object).
        A figure without a single non-zero mode is silent -- no bare carrier, even at
        FM depth 0."""
        _labels, voices, _color = analyse(self._grid, self.ctx.f0, 'laplacian',
                                          self._spectrum_params(), exc=self._exc)
        for v in voices:
            v['pan'] = PAN_CENTER
        self.voices = voices
        self._pending_analysis = False
        mods = [None] * MAX_VOICES
        for i, v in enumerate(voices[:MAX_VOICES]):
            fr = np.asarray(v['freqs'], dtype=float)
            am = np.asarray(v['amps'], dtype=float)
            live = (fr > 0.0) & (am > 0.0)
            if not live.any():
                continue
            mods[i] = (fr, am, float(math.sqrt(float((am[live] * am[live]).sum()))))
        self._mods = mods

    def _finish(self, L, R, gain):
        """The baseline block tail: one pre-clip gain ramp, honest metering, clip,
        int16 (render_chunk_laplacian)."""
        g = self.gain_prev + (gain - self.gain_prev) * _RAMP
        L = L * g
        R = R * g
        peak = float(max(np.abs(L).max(), np.abs(R).max()))
        n_clip = int(np.count_nonzero(np.abs(L) > 1.0) + np.count_nonzero(np.abs(R) > 1.0))
        L = np.clip(L, -1.0, 1.0)
        R = np.clip(R, -1.0, 1.0)
        ch = self.ctx.channels
        if ch <= 1:
            data = ((L + R) * 0.5)[:, None]
        else:
            data = np.zeros((len(L), ch))
            data[:, 0] = L
            data[:, 1] = R
        self.gain_prev = gain
        return np.ascontiguousarray((data * 32767).astype(np.int16)), peak, n_clip


def _widen_v1(state):
    """A version 1 snapshot (modulator pool of MAX_MODES_PER_OBJ tails) read into this
    build's wider pool: the active slots are 0..MAX_MODES_PER_OBJ-1 in both, and the old
    tail slots keep their own indices, so the copy is exact and the sound continues
    unchanged.  Records made before 2026-09-21 stay continuable."""
    if state.get('version') == LaplaceFMEngine.STATE_VERSION:
        return state
    old_slots = int(state.get('mod_slots', 0))
    if old_slots > N_MODS:
        raise ValueError(f"engine {ENGINE_ID}: state carries {old_slots} modulator slots, "
                         f"this build has {N_MODS}")
    out = dict(state)
    for name in ('f_mod', 'th_mod', 'beta_cur', 'beta_tgt', 'm_env_phase', 'm_env_level',
                 'm_rel_cnt', 'm_rel_len', 'm_rel_beta0'):
        key = 'src_' + name
        a = state.get(key)
        if not isinstance(a, np.ndarray) or a.ndim != 2 or a.shape[1] != old_slots:
            raise ValueError(f"engine {ENGINE_ID}: version 1 state array {key!r} missing or "
                             f"shape {getattr(a, 'shape', None)} != (*, {old_slots})")
        wide = np.zeros((a.shape[0], N_MODS), dtype=a.dtype)
        if name == 'm_rel_len':
            wide[:] = 1
        wide[:, :MAX_MODES_PER_OBJ] = a[:, :MAX_MODES_PER_OBJ]
        wide[:, MAX_MODES_PER_OBJ:old_slots] = a[:, MAX_MODES_PER_OBJ:]
        out[key] = wide
    out['version'] = LaplaceFMEngine.STATE_VERSION
    out['mod_slots'] = int(N_MODS)
    return out


def inactive(params):
    """Settings that do not act for the current mode (the bench shows them as text).

    `fm_depth` is NEVER listed here: a parameter shown as text is not editable, and
    the depth has to stay draggable at 0 (what 0 means is said in the display line)."""
    out = {}
    if float(params.get('shape', 0.0)) <= 0.0:
        out['dyn'] = 'acts only with shape > 0'
    return out


def overlay(params, rows, cols):
    i = float(params.get('fm_depth', OPTIONAL_PARAMS['fm_depth']))
    return dict(text=f"Laplace FM [depth {i:.2f}]" + ("  carrier only" if i <= 0.0 else ""))


register(EngineSpec(ENGINE_ID, LABEL, PARAMS,
                    lambda ctx, params: LaplaceFMEngine(ctx, params),
                    inactive=inactive, overlay=overlay))
