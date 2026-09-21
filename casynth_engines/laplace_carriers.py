"""Laplace carriers (REQ memory/req-laplace-carriers-2026-09-20.md): the SAME
Laplacian frequencies and weights sounded by two other laws.

  Filter    -- one periodic wave per figure at the note frequency f0; the Laplacian
               profile shapes the amplitudes of ITS harmonics (k * f0).
  Wave bank -- one periodic wave per selected MODE at that mode's frequency f_j,
               its amplitude the weight a_j; every wave brings its own harmonics
               (h * f_j).  Sine is the historical sum of modes.

Both take their geometry, voices, audio clock, phases, envelopes and release
times from the existing baseline path (casynth_engine.analyse + SlotPool + the
GEN_* envelopes of LegacySynthEngine): only the WAVEFORM of a sounding line, or
the mapping spectrum -> carrier colouring, is new.  Wave bank / Sine is the
legacy `laplacian` engine bit-for-bit on any field whose mode frequencies stay
below 0.40 * sr (the band limit below is the only difference, and it is 1 there).

Waveform law (REQ section 3), theta = phase of the FUNDAMENTAL of that wave:

    c_h(Sine) = 1 at h=1 else 0;  c_h(Saw) = 1/h;  c_h(Square) = 1/h for odd h
    b(nu) = 1 below 0.40*sr, a cosine taper to 0 at 0.45*sr, 0 above
    W(theta; f) = sum_h c_h * b(h*f) * sin(h*theta)

    Wave bank:  y = sum_j a_j * W(theta_j; f_j),          theta_j' = 2*pi*f_j
    Filter:     P_k = sum_j a_j * exp(-0.5*(log2(k/r_j)/sigma)^2),  r_j = f_j/f0
                E_k = P_k / max_m P_m;  H_k = 10^(-D*(1-E_k)/20)
                A    = sqrt(sum_j a_j^2)
                y = A * sum_k c_k * b(k*f0) * H_k * sin(k*theta_0)

The harmonics of one source keep the coherent phases h*theta (never independent
random phases), and a figure with no non-zero weight -- or a flat P below 1e-12
-- is silent (a bare carrier is never started, not even at D = 0).

Implementation notes
  - Saw / Square are played from a band-limited WAVETABLE built for the EXACT
    frequency of that wave (so b(h*f) is the law's, not an octave's), linearly
    interpolated; TABLE_N is sized so the table's error stays far below the -60 dB
    the REQ asks for (tests measure it against the direct finite sum).  Sine is
    rendered directly (no table), which also keeps the bit-exact baseline path.
  - Filter sums its harmonics through two cached basis matrices cos/sin(k*omega*i)
    -- one carrier frequency serves every voice, only the phase and the mask differ.
  - Any change of the spectrum (mask, waveform coefficients, weights) glides over
    ONE block on the pool's own _RAMP, like a baseline amplitude change; switching
    the METHOD crossfades over the voice release time (both laws are rendered while
    it runs and a new switch turns the crossfade around instead of cutting it).
"""
import math

import numpy as np

from casynth_config import (SR, TWO_PI, _RAMP, TOTAL_SLOTS, N_ACTIVE, MAX_VOICES,
                            MAX_MODES_PER_OBJ, TAIL_MIN_AMP)
from casynth_core import ENGINE_BY_ID
from casynth_engine import analyse, SlotPool
from .engine_api import SoundEngine
from .legacy_engine import (PAN_CENTER, GenEnvelopeKnobs, _SLOT_ARRAYS, _POOL_ARRAYS)
from .registry import EngineSpec, register

ENGINE_ID = 'laplace_carriers'
LABEL = 'Laplace waves'

# the seven spectrum settings of the old Laplace, metadata from the core registry
LAPLACE_PARAMS = tuple(tuple(p) for p in ENGINE_BY_ID['laplacian']['params'])
SPECTRUM_KEYS = tuple(p[0] for p in LAPLACE_PARAMS)   # n spread alpha shape harm fullshape dyn

METHOD_FILTER, METHOD_BANK = 0, 1
WF_SINE, WF_SAW, WF_SQUARE = 0, 1, 2
METHOD_NAMES = ('Filter', 'Wave bank')
WAVE_NAMES = ('Sine', 'Saw', 'Square')

PARAMS = [('method', 'Method', 0, 1, True, METHOD_BANK),
          ('waveform', 'Wave', 0, 2, True, WF_SINE),
          ('filter_width_oct', 'width', 0.10, 1.00, False, 0.35),
          ('filter_depth_db', 'depth', 0.0, 36.0, False, 24.0)] + list(LAPLACE_PARAMS)
CHOICES = {'method': METHOD_NAMES, 'waveform': WAVE_NAMES}
# absent from an older parameter set / snapshot -> these defaults
OPTIONAL_PARAMS = {'method': METHOD_BANK, 'waveform': WF_SINE,
                   'filter_width_oct': 0.35, 'filter_depth_db': 24.0}
OPTIONAL_PARAMS.update({p[0]: p[5] for p in LAPLACE_PARAMS})

# band limit b(nu) (REQ section 3): flat, cosine taper, zero
BAND_FLAT = 0.40                  # * sr
BAND_ZERO = 0.45                  # * sr
GUARD = BAND_ZERO * SR            # the pool's anti-alias guard, as in render_chunk_laplacian
TABLE_N = 8192                    # wavetable length (power of two); error << -60 dB
TABLE_CACHE_MAX = 256             # tables kept (cleared when full); the sound never depends on a hit
MASK_FLOOR = 1e-12                # max P at or below this -> the figure is silent
AMP_EPS = 1e-4                    # a source below this on both ends of the block is skipped
N_FILTER_TAILS = MAX_VOICES * 4   # frozen Filter carriers ringing out


# ── the waveform law ──────────────────────────────────────────────────────────

def band_limit(nu, sr=SR):
    """b(nu): 1 up to 0.40*sr, cosine taper to 0 at 0.45*sr, 0 above (REQ 3)."""
    x = np.asarray(nu, dtype=float)
    flat, zero = BAND_FLAT * sr, BAND_ZERO * sr
    out = np.ones(x.shape, dtype=float)
    taper = (x > flat) & (x < zero)
    out[taper] = 0.5 * (1.0 + np.cos(np.pi * (x[taper] - flat) / (zero - flat)))
    out[x >= zero] = 0.0
    out[x <= 0.0] = 0.0
    return out if out.ndim else float(out)


def band_limit1(nu, sr=SR):
    """Scalar b(nu)."""
    return float(band_limit(np.array([float(nu)]), sr)[0])


def harmonic_indices(waveform, f, sr=SR):
    """Harmonic numbers h of one wave at fundamental f: every h with b(h*f) > 0
    (Saw: all, Square: odd, Sine: h = 1 only)."""
    f = float(f)
    if not (f > 0.0) or not np.isfinite(f):
        return np.zeros(0, np.int64)
    h_max = int(math.floor((BAND_ZERO * sr) / f))
    while h_max > 0 and h_max * f >= BAND_ZERO * sr:      # strict inequality, float-safe
        h_max -= 1
    if h_max < 1:
        return np.zeros(0, np.int64)
    if waveform == WF_SINE:
        return np.ones(1, np.int64)
    h = np.arange(1, h_max + 1, dtype=np.int64)
    if waveform == WF_SQUARE:
        h = h[h % 2 == 1]
    return h


def harmonic_coeffs(waveform, f, sr=SR):
    """(h, coef) of one wave at fundamental f: coef = c_h * b(h*f) (REQ 3).

    The amplitude of the fundamental is 1 before the band limit; there is no
    per-wave peak/RMS normalisation."""
    h = harmonic_indices(waveform, f, sr)
    if len(h) == 0:
        return h, np.zeros(0)
    c = 1.0 / h.astype(float) if waveform != WF_SINE else np.ones(1)
    return h, c * band_limit(h.astype(float) * float(f), sr)


def direct_wave(waveform, f, phase0, n, sr=SR):
    """The wave as the exact finite sum sum_h c_h*b(h*f)*sin(h*theta) over n samples
    starting at theta = phase0 (the reference the wavetable is checked against)."""
    h, c = harmonic_coeffs(waveform, f, sr)
    idx = np.arange(int(n), dtype=float)
    th = float(phase0) + (TWO_PI * float(f) / sr) * idx
    if len(h) == 0:
        return np.zeros(int(n))
    return (c[:, None] * np.sin(h[:, None].astype(float) * th[None, :])).sum(axis=0)


_TABLES = {}


def wavetable(waveform, f, sr=SR):
    """One period of the band-limited wave of fundamental f, TABLE_N samples
    (cached by the EXACT frequency: b(h*f) belongs to this wave, not to an octave)."""
    key = (int(waveform), float(f), float(sr))
    tab = _TABLES.get(key)
    if tab is not None:
        return tab
    h, c = harmonic_coeffs(waveform, f, sr)
    spec = np.zeros(TABLE_N // 2 + 1, dtype=complex)
    keep = h[h <= TABLE_N // 2 - 1]
    if len(keep):
        spec[keep] = -1j * c[:len(keep)] * (TABLE_N / 2.0)
    tab = np.fft.irfft(spec, TABLE_N)
    if len(_TABLES) >= TABLE_CACHE_MAX:
        _TABLES.clear()                    # bounded memory; identical values on a miss
    _TABLES[key] = tab
    return tab


def table_wave(tab, phase0, inc, n):
    """Linearly interpolated table lookup over n samples from phase0 with the
    per-sample phase increment inc (radians)."""
    x = (float(phase0) + inc * np.arange(int(n), dtype=float)) * (len(tab) / TWO_PI)
    base = np.floor(x)
    frac = x - base
    i0 = base.astype(np.int64) % len(tab)
    i1 = (i0 + 1) % len(tab)
    return tab[i0] * (1.0 - frac) + tab[i1] * frac


def slot_wave(waveform, f, phase0, n, sr=SR):
    """One sounding line: Sine directly (bit-exact legacy path when b = 1),
    Saw / Square from the band-limited table of that exact frequency."""
    idx = np.arange(int(n), dtype=float)
    ph = float(phase0) + (TWO_PI * float(f) / sr) * idx
    if waveform == WF_SINE:
        b = band_limit1(f, sr)
        return np.sin(ph) if b == 1.0 else b * np.sin(ph)
    return table_wave(wavetable(waveform, f, sr), phase0, TWO_PI * float(f) / sr, n)


# ── the Filter mask ───────────────────────────────────────────────────────────

def carrier_harmonics(f0, sr=SR):
    """K: the harmonic numbers 1..K of the carrier, K*f0 < 0.45*sr (REQ 3.2)."""
    return int(len(harmonic_indices(WF_SAW, f0, sr)))


def filter_mask(freqs, amps, f0, sigma, depth_db, sr=SR):
    """(H, A) of one figure (REQ 3.2), or (None, 0.0) when it is silent.

    H : (K,) mask of the carrier harmonics k = 1..K, 10^(-D/20) <= H <= 1
    A : sqrt(sum a_j^2) over the non-zero modes -- the figure's own scale, NOT a
        loudness normalisation.
    The normalisation of E runs over ALL k (even ones included), so choosing Saw
    or Square never re-shapes the mask itself."""
    fr = np.asarray(freqs, dtype=float)
    am = np.asarray(amps, dtype=float)
    live = (fr > 0.0) & (am > 0.0)
    if not live.any():
        return None, 0.0
    r = fr[live] / float(f0)
    a = am[live]
    K = carrier_harmonics(f0, sr)
    if K < 1:
        return None, 0.0
    k = np.arange(1, K + 1, dtype=float)
    z = np.log2(k[:, None] / r[None, :]) / float(sigma)
    P = (a[None, :] * np.exp(-0.5 * z * z)).sum(axis=1)
    pmax = float(P.max())
    if not (pmax > MASK_FLOOR):
        return None, 0.0
    E = P / pmax
    H = np.power(10.0, -float(depth_db) * (1.0 - E) / 20.0)
    return H, float(math.sqrt(float((a * a).sum())))


_BASIS = {}


def carrier_basis(f0, block, sr=SR):
    """(C, S) with C[k-1, i] = cos(k*omega*i), S[k-1, i] = sin(k*omega*i),
    omega = 2*pi*f0/sr -- the basis every Filter carrier of this scene shares."""
    key = (float(f0), int(block), float(sr))
    got = _BASIS.get(key)
    if got is not None:
        return got
    K = carrier_harmonics(f0, sr)
    k = np.arange(1, K + 1, dtype=float)[:, None]
    i = np.arange(int(block), dtype=float)[None, :]
    arg = k * (TWO_PI * float(f0) / sr) * i
    got = (np.cos(arg), np.sin(arg))
    if len(_BASIS) > 8:
        _BASIS.clear()
    _BASIS[key] = got
    return got


def carrier_coeffs(waveform, f0, sr=SR):
    """g_k = c_k * b(k*f0) for k = 1..K (zero where the waveform has no harmonic)."""
    K = carrier_harmonics(f0, sr)
    g = np.zeros(K)
    h, c = harmonic_coeffs(waveform, f0, sr)
    keep = h <= K
    g[h[keep] - 1] = c[keep]
    return g


# ── Wave bank: the legacy slot pool with another wave per slot ────────────────

class _BankVoices:
    """The baseline voice machinery (SlotPool + GEN envelopes + release tails) with
    the sine of every slot replaced by W(theta; f) of the current waveform."""

    def __init__(self):
        self.pool = SlotPool()
        sz = TOTAL_SLOTS + 1
        self.phase = np.zeros(sz)
        self.amp_cur = np.zeros(sz)
        self.pan_cur = np.full(sz, PAN_CENTER)

    def update(self, voices, release_chunks, attack_chunks, decay_chunks, sustain):
        self.pool.update(voices, self.phase, self.amp_cur, self.pan_cur,
                         release_chunks, attack_chunks, decay_chunks, sustain,
                         amp_slew=False)

    def render(self, waveform, wave_prev, n, sr=SR, transpose=1.0):
        """(L, R) of this block; the slot order, ramps and phase bookkeeping are
        the baseline's (render_chunk_laplacian), only `wave` differs.

        `transpose` multiplies every line's frequency: the band limit and the
        wavetable are then built for the frequency that actually sounds, and the
        phase keeps accumulating, so a note change is continuous."""
        L = np.zeros(n)
        R = np.zeros(n)
        idx = np.arange(n, dtype=float)
        pool = self.pool
        for k in range(1, TOTAL_SLOTS + 1):
            if self.amp_cur[k] < AMP_EPS and pool.amp_tgt[k] < AMP_EPS:
                continue
            freq = pool.freq_slots[k] * transpose
            if freq <= 0.0 or freq >= GUARD:
                self.amp_cur[k] = 0.0
                continue
            inc = TWO_PI * freq / sr
            ph = self.phase[k] + inc * idx
            if waveform == WF_SINE:
                b = band_limit1(freq, sr)
                wave = np.sin(ph) if b == 1.0 else b * np.sin(ph)
            else:
                wave = table_wave(wavetable(waveform, freq, sr), self.phase[k], inc, n)
            if wave_prev != waveform:                 # one-block glide between waveforms
                if wave_prev == WF_SINE:
                    b = band_limit1(freq, sr)
                    old = np.sin(ph) if b == 1.0 else b * np.sin(ph)
                else:
                    old = table_wave(wavetable(wave_prev, freq, sr), self.phase[k], inc, n)
                wave = old * (1.0 - _RAMP) + wave * _RAMP
            a = self.amp_cur[k] + (pool.amp_tgt[k] - self.amp_cur[k]) * _RAMP
            p = self.pan_cur[k] + (pool.pan_tgt[k] - self.pan_cur[k]) * _RAMP
            L += wave * a * np.cos(p * np.pi / 2.0)
            R += wave * a * np.sin(p * np.pi / 2.0)
            self.phase[k] = (self.phase[k] + inc * n) % TWO_PI
            self.amp_cur[k] = pool.amp_tgt[k]
            self.pan_cur[k] = pool.pan_tgt[k]
        return L, R

    def advance_silent(self, n, sr=SR, transpose=1.0):
        """Keep the state moving while this law is not heard (method switched away):
        same bookkeeping as render(), no audio."""
        pool = self.pool
        for k in range(1, TOTAL_SLOTS + 1):
            if self.amp_cur[k] < AMP_EPS and pool.amp_tgt[k] < AMP_EPS:
                continue
            freq = pool.freq_slots[k] * transpose
            if freq <= 0.0 or freq >= GUARD:
                self.amp_cur[k] = 0.0
                continue
            inc = TWO_PI * freq / sr
            self.phase[k] = (self.phase[k] + inc * n) % TWO_PI
            self.amp_cur[k] = pool.amp_tgt[k]
            self.pan_cur[k] = pool.pan_tgt[k]


# ── Filter: one carrier per voice channel + frozen tails ──────────────────────

class _FilterVoices:
    """One carrier (f0) per voice channel, its harmonics weighted by the figure's
    mask; a channel that loses its figure is frozen into a tail that rings out over
    the baseline release time.  The envelopes, the release length and the glide of
    every changing value are the baseline's."""

    def __init__(self, f0, block, sr=SR):
        self.f0 = float(f0)
        self.sr = float(sr)
        self.block = int(block)
        self.K = carrier_harmonics(f0, sr)
        self.f0_base = float(f0)              # the anchor; self.f0 is what sounds
        n = MAX_VOICES + N_FILTER_TAILS
        self.th = np.zeros(n)                 # carrier phase of every source
        self.amp_cur = np.zeros(n)
        self.amp_tgt = np.zeros(n)
        self.spec_cur = np.zeros((n, self.K))  # c_k*b(k*f0)*H_k in force now
        self.spec_tgt = np.zeros((n, self.K))
        self.env_phase = np.zeros(n, dtype=int)
        self.env_level = np.zeros(n)
        self.alive = np.zeros(MAX_VOICES, dtype=bool)
        self.rel_cnt = np.zeros(n, dtype=int)
        self.rel_len = np.ones(n, dtype=int)
        self.rel_amp0 = np.zeros(n)
        self.steals = 0

    def retune(self, f0):
        """A new sounding carrier (the note).  Phases, amplitudes, envelopes and
        tails are KEPT -- only the harmonic grid is re-derived, because how many
        harmonics fit under the band limit depends on the pitch.  The spectra
        themselves are handed back by the engine's next analysis."""
        f0 = float(f0)
        if f0 == self.f0:
            return
        K = carrier_harmonics(f0, self.sr)
        if K != self.K:
            for name in ('spec_cur', 'spec_tgt'):
                old = getattr(self, name)
                new = np.zeros((old.shape[0], K))
                keep = min(K, self.K)
                new[:, :keep] = old[:, :keep]
                setattr(self, name, new)
            self.K = K
        self.f0 = f0

    # -- envelope (the SlotPool rule) -------------------------------------------
    def _advance_env(self, v, attack_chunks, decay_chunks, sustain):
        ph = self.env_phase[v]
        if ph == 1:
            lvl = self.env_level[v] + 1.0 / attack_chunks
            if lvl >= 1.0:
                lvl, self.env_phase[v] = 1.0, 2
            self.env_level[v] = lvl
        elif ph == 2:
            lvl = self.env_level[v] - (1.0 - sustain) / decay_chunks
            if lvl <= sustain:
                lvl, self.env_phase[v] = sustain, 3
            self.env_level[v] = lvl
        elif ph == 3:
            self.env_level[v] = sustain

    def _acquire_tail(self):
        lo, hi = MAX_VOICES, MAX_VOICES + N_FILTER_TAILS
        quietest, quiet_amp = lo, np.inf
        for s in range(lo, hi):
            if self.rel_cnt[s] == 0 and self.amp_cur[s] < AMP_EPS:
                return s
            if self.amp_cur[s] < quiet_amp:
                quiet_amp, quietest = self.amp_cur[s], s
        self.steals += 1
        return quietest

    def update(self, specs, scales, release_chunks, attack_chunks, decay_chunks, sustain):
        """specs[v] : (K,) target spectrum of channel v or None (silent);
        scales[v]  : A_object of that channel."""
        for v in range(MAX_VOICES):
            spec = specs[v] if v < len(specs) else None
            scale = float(scales[v]) if v < len(scales) else 0.0
            present = spec is not None and scale > 0.0
            if present:
                if not self.alive[v]:
                    self.env_phase[v] = 1               # onset: re-trigger, phase continues
                    self.env_level[v] = 0.0
                    self.amp_cur[v] = 0.0
                    self.spec_cur[v] = spec             # no glide while silent
                self._advance_env(v, attack_chunks, decay_chunks, sustain)
                self.spec_tgt[v] = spec
                self.amp_tgt[v] = self.env_level[v] * scale
                self.alive[v] = True
                continue
            if self.alive[v] and self.amp_cur[v] > TAIL_MIN_AMP:
                t = self._acquire_tail()
                self.th[t] = self.th[v]
                self.amp_cur[t] = self.amp_cur[v]
                self.spec_cur[t] = self.spec_cur[v]     # the tail keeps its colour
                self.spec_tgt[t] = self.spec_cur[v]
                self.rel_cnt[t] = release_chunks
                self.rel_len[t] = release_chunks
                self.rel_amp0[t] = float(self.amp_cur[v])
                self.amp_tgt[t] = float(self.amp_cur[v])
            self.amp_cur[v] = 0.0
            self.amp_tgt[v] = 0.0
            self.env_phase[v] = 0
            self.env_level[v] = 0.0
            self.alive[v] = False
        for s in range(MAX_VOICES, MAX_VOICES + N_FILTER_TAILS):
            cnt = self.rel_cnt[s]
            if cnt <= 0:
                continue
            cnt -= 1
            self.rel_cnt[s] = cnt
            self.amp_tgt[s] = self.rel_amp0[s] * cnt / self.rel_len[s]
            if cnt == 0 and self.amp_cur[s] < 0.01:
                self.spec_cur[s] = 0.0
                self.spec_tgt[s] = 0.0

    def render(self, n, sr=SR):
        """(L, R): sum over the sounding carriers of A(t) * sum_k spec_k(t) *
        sin(k*theta), with A and spec glided across the block on _RAMP."""
        act = np.nonzero((self.amp_cur >= AMP_EPS) | (self.amp_tgt >= AMP_EPS))[0]
        omega = TWO_PI * self.f0 / sr
        if len(act) == 0:
            self.th[:] = (self.th + omega * n) % TWO_PI
            self.amp_cur[:] = self.amp_tgt
            self.spec_cur[:] = self.spec_tgt
            return np.zeros(n), np.zeros(n)
        C, S = carrier_basis(self.f0, n, sr)
        k = np.arange(1, self.K + 1, dtype=float)[None, :]
        kth = k * self.th[act][:, None]
        sin_kth, cos_kth = np.sin(kth), np.cos(kth)
        spec0 = self.spec_cur[act]
        dspec = self.spec_tgt[act] - spec0
        base = (spec0 * sin_kth) @ C + (spec0 * cos_kth) @ S
        delta = (dspec * sin_kth) @ C + (dspec * cos_kth) @ S
        wave = base + delta * _RAMP[None, :]
        a0 = self.amp_cur[act][:, None]
        da = (self.amp_tgt[act] - self.amp_cur[act])[:, None]
        y = ((a0 + da * _RAMP[None, :]) * wave).sum(axis=0)
        self.th[:] = (self.th + omega * n) % TWO_PI
        self.amp_cur[:] = self.amp_tgt
        self.spec_cur[:] = self.spec_tgt
        pan = PAN_CENTER * np.pi / 2.0
        return y * math.cos(pan), y * math.sin(pan)

    def advance_silent(self, n, sr=SR):
        self.th[:] = (self.th + (TWO_PI * self.f0 / sr) * n) % TWO_PI
        self.amp_cur[:] = self.amp_tgt
        self.spec_cur[:] = self.spec_tgt


# ── the engine ────────────────────────────────────────────────────────────────

class LaplaceCarriersEngine(GenEnvelopeKnobs, SoundEngine):
    """Both laws on one analysis; `method` picks which one is heard."""

    STATE_VERSION = 1

    # the Filter carriers run their own per-source envelope, which has no
    # amplitude-slew path -- the GEN slew toggle would act on one method and not
    # on the other, so this engine declines it for both (GenEnvelopeKnobs)
    SUPPORTS_AMP_SLEW = False

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        self.engine_id = ENGINE_ID
        # the GEN envelope is the SlotPool's and its knobs are the host's, live
        # (2026-09-21); _recount below keeps the method crossfade tied to release
        self._init_gen_env(ctx.rate_hz)
        self.voices = []
        self._grid = None
        self._exc = None
        self._clear_audio(gain=0.0)

    def _recount(self):
        """Plus the rule this engine adds: the method crossfade lasts exactly as
        long as a voice takes to let go, so a switch never outlives its tails."""
        super()._recount()
        self._xfade_blocks = max(1, self._release_chunks)

    # -- parameters --------------------------------------------------------------
    def _p(self, name):
        return self.params.get(name, OPTIONAL_PARAMS[name])

    def _method(self):
        return int(self._p('method'))

    def _waveform(self):
        return int(self._p('waveform'))

    def _spectrum_params(self):
        return {k: self._p(k) for k in SPECTRUM_KEYS}

    # -- SoundEngine -------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._clear_audio(gain)
        self.update_field(grid, exc)

    def update_field(self, grid, exc):
        self._grid, self._exc = grid, exc
        self._analyse()

    def set_params(self, params):
        super().set_params(params)
        if self._grid is not None:
            self._analyse()

    SUPPORTS_TRANSPOSE = True      # the note is a scalar on every frequency

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        if gain_prev is not None:
            self.gain_prev = float(gain_prev)      # the host overrides the glide start
        if transpose != self._transpose:
            self._retune(transpose)
        n = self.ctx.block
        self.bank.update(self.voices, self._release_chunks, self._attack_chunks,
                         self._decay_chunks, self._sustain)
        self.filt.update(self._specs, self._scales, self._release_chunks,
                         self._attack_chunks, self._decay_chunks, self._sustain)
        wf, prev = self._waveform(), self._wave_prev
        mix_prev = self._mix
        target = 1.0 if self._method() == METHOD_BANK else 0.0
        step = 1.0 / self._xfade_blocks
        if self._mix < target:
            self._mix = min(target, self._mix + step)
        elif self._mix > target:
            self._mix = max(target, self._mix - step)
        mix = self._mix
        if mix_prev >= 1.0 and mix >= 1.0:
            L, R = self.bank.render(wf, prev, n, self.ctx.sr, self._transpose)
            self.filt.advance_silent(n, self.ctx.sr)
        elif mix_prev <= 0.0 and mix <= 0.0:
            L, R = self.filt.render(n, self.ctx.sr)
            self.bank.advance_silent(n, self.ctx.sr, self._transpose)
        else:
            Lb, Rb = self.bank.render(wf, prev, n, self.ctx.sr, self._transpose)
            Lf, Rf = self.filt.render(n, self.ctx.sr)
            w = mix_prev + (mix - mix_prev) * _RAMP
            L = Lf * (1.0 - w) + Lb * w
            R = Rf * (1.0 - w) + Rb * w
        self._wave_prev = wf
        return self._finish(L, R, gain)

    def _retune(self, transpose):
        """A new note.  Both laws keep every phase, amplitude and envelope; the
        Filter's harmonic grid and the analysed spectra are re-derived at the
        pitch that now sounds."""
        self._transpose = float(transpose)
        self.filt.retune(self.ctx.f0 * self._transpose)
        if self._grid is not None:
            self._analyse()

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

    def display(self):
        """Read-only numbers for the bench panel (never part of the sound)."""
        h = int(len(harmonic_indices(self._waveform(), self.ctx.f0, self.ctx.sr)))
        sounding = int(sum(1 for s in self._specs if s is not None))
        tails = int(np.count_nonzero(self.filt.rel_cnt[MAX_VOICES:] > 0))
        bank_slots = int(np.count_nonzero(self.bank.pool.amp_tgt[1:] >= AMP_EPS))
        return dict(carriers=True, method=self._method(), method_name=METHOD_NAMES[self._method()],
                    waveform=self._waveform(), wave_name=WAVE_NAMES[self._waveform()],
                    mix=float(self._mix), f0=float(self.ctx.f0), harmonics=h,
                    carrier_harmonics=int(self.filt.K), voices=int(len(self.voices)),
                    sounding=sounding, filter_tails=tails, bank_lines=bank_slots,
                    width_oct=float(self._p('filter_width_oct')),
                    depth_db=float(self._p('filter_depth_db')),
                    lines=[dict(n=int(np.count_nonzero(np.asarray(v['freqs']) > 0)),
                                f_low=float(np.asarray(v['freqs'])[0]),
                                a=float(self._scales[i]))
                           for i, v in enumerate(self.voices[:MAX_VOICES])])

    # -- snapshot ----------------------------------------------------------------
    _BANK_ARRAYS = tuple('bank_' + n for n in _SLOT_ARRAYS + _POOL_ARRAYS)
    _FILTER_ARRAYS = ('th', 'amp_cur', 'amp_tgt', 'spec_cur', 'spec_tgt',
                      'env_phase', 'env_level', 'alive', 'rel_cnt', 'rel_len', 'rel_amp0')

    def export_state(self):
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  params=dict(self.params), gain_prev=float(self.gain_prev),
                  mix=float(self._mix), wave_prev=int(self._wave_prev),
                  slots=int(TOTAL_SLOTS), voices=int(MAX_VOICES),
                  modes=int(MAX_MODES_PER_OBJ), tails=int(N_FILTER_TAILS),
                  carrier_k=int(self.filt.K), f0=float(self.ctx.f0),
                  steals=int(self.bank.pool.steals),
                  steal_amp_max=float(self.bank.pool.steal_amp_max),
                  filter_steals=int(self.filt.steals))
        for name in _SLOT_ARRAYS:
            st['bank_' + name] = getattr(self.bank, name).copy()
        for name in _POOL_ARRAYS:
            st['bank_' + name] = getattr(self.bank.pool, name).copy()
        st['bank__freq_prev'] = self.bank.pool._freq_prev.copy()
        for name in self._FILTER_ARRAYS:
            st['filter_' + name] = getattr(self.filt, name).copy()
        return st

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != self.engine_id:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {self.engine_id!r}")
        if (state.get('slots'), state.get('voices'), state.get('modes'), state.get('tails')) != \
                (TOTAL_SLOTS, MAX_VOICES, MAX_MODES_PER_OBJ, N_FILTER_TAILS):
            raise ValueError(f"engine {ENGINE_ID}: pool layout of the state does not match this build")
        if int(state.get('carrier_k', -1)) != carrier_harmonics(self.ctx.f0, self.ctx.sr):
            raise ValueError(f"engine {ENGINE_ID}: the state carries {state.get('carrier_k')} carrier "
                             f"harmonics, this scene has {carrier_harmonics(self.ctx.f0, self.ctx.sr)}")
        sz = TOTAL_SLOTS + 1
        want = {'bank_' + n: (sz,) for n in _SLOT_ARRAYS + _POOL_ARRAYS}
        want['bank__freq_prev'] = (MAX_VOICES, MAX_MODES_PER_OBJ)
        nsrc = MAX_VOICES + N_FILTER_TAILS
        K = int(state['carrier_k'])
        for name in self._FILTER_ARRAYS:
            want['filter_' + name] = ((nsrc, K) if name.startswith('spec') else
                                      ((MAX_VOICES,) if name == 'alive' else (nsrc,)))
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
        for name in _SLOT_ARRAYS:
            getattr(self.bank, name)[:] = arrays['bank_' + name]
        for name in _POOL_ARRAYS:
            getattr(self.bank.pool, name)[:] = arrays['bank_' + name]
        self.bank.pool._freq_prev[:] = arrays['bank__freq_prev']
        self.bank.pool.steals = int(state.get('steals', 0))
        self.bank.pool.steal_amp_max = float(state.get('steal_amp_max', 0.0))
        for name in self._FILTER_ARRAYS:
            getattr(self.filt, name)[:] = arrays['filter_' + name]
        self.filt.steals = int(state.get('filter_steals', 0))
        self._mix = float(state['mix'])
        self._wave_prev = int(state['wave_prev'])
        self.params = dict(state['params'])
        self.update_field(grid, exc)

    # -- internals ---------------------------------------------------------------
    def _clear_audio(self, gain):
        self.bank = _BankVoices()
        self._transpose = 1.0
        self.filt = _FilterVoices(self.ctx.f0, self.ctx.block, self.ctx.sr)
        self.gain_prev = gain
        self._mix = 1.0 if self._method() == METHOD_BANK else 0.0
        self._wave_prev = self._waveform()
        self._specs = [None] * MAX_VOICES
        self._scales = [0.0] * MAX_VOICES

    def _analyse(self):
        """voices = the baseline analysis; the Filter targets follow from the same
        (freqs, amps) -- both laws always describe the SAME spectrum."""
        _labels, voices, _color = analyse(self._grid, self.ctx.f0, 'laplacian',
                                          self._spectrum_params(), exc=self._exc)
        for v in voices:
            v['pan'] = PAN_CENTER
        self.voices = voices
        sigma = float(self._p('filter_width_oct'))
        depth = float(self._p('filter_depth_db'))
        # the whole spectrum moves with the note: the carrier AND the modes it
        # filters, so the mask a figure casts on its carrier is the same shape at
        # any pitch -- only how many harmonics fit under the band limit changes
        t = self._transpose
        f0 = self.ctx.f0 * t
        g = carrier_coeffs(self._waveform(), f0, self.ctx.sr)
        specs, scales = [None] * MAX_VOICES, [0.0] * MAX_VOICES
        for i, v in enumerate(voices[:MAX_VOICES]):
            H, A = filter_mask(v['freqs'] * t, v['amps'], f0, sigma, depth, self.ctx.sr)
            if H is None:
                continue
            specs[i] = g * H
            scales[i] = A
        self._specs = specs
        self._scales = scales

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


def inactive(params):
    """Settings that do not act for the current mode (the bench shows them as text)."""
    out = {}
    if int(params.get('method', OPTIONAL_PARAMS['method'])) == METHOD_BANK:
        out['filter_width_oct'] = f"{float(params.get('filter_width_oct', 0.35)):.2f}  Filter only"
        out['filter_depth_db'] = f"{float(params.get('filter_depth_db', 24.0)):.0f} dB  Filter only"
    if float(params.get('shape', 0.0)) <= 0.0:
        out['dyn'] = 'acts only with shape > 0'
    return out


def overlay(params, rows, cols):
    m = METHOD_NAMES[int(params.get('method', OPTIONAL_PARAMS['method']))]
    w = WAVE_NAMES[int(params.get('waveform', OPTIONAL_PARAMS['waveform']))]
    return dict(text=f"Laplace waves [{m}, {w}]")




def _unified(ctx, params):
    """REQ memory/req-unified-laplace-2026-09-21.md section 5: this id is an
    ALIAS of the unified Laplace engine with its axes pinned.  The factory asks
    that engine which cell the id means, and the cell IS this law -- so the
    parameters, the snapshot and every byte of a record stay what they were, and
    only ONE place has to know what an id stands for.  The import is deferred to
    the first instance, so the registry still costs nothing to import."""
    from . import laplace_unified as lu
    return lu.create(ctx, params, ENGINE_ID)


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, _unified,
                    choices=CHOICES, inactive=inactive, overlay=overlay,
                    # the SlotPool envelope, but no slew path in the Filter method
                    gen_envelope=True, gen_amp_slew=False,
                    plays_notes=True))
