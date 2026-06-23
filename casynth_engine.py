"""Audio engine + field analysis for the gol_synth prototype.

Pure DSP/simulation, no pygame: the Game-of-Life step, the field->voices analysis
(engine dispatch via casynth_core.ENGINE_BY_ID), the gapless chunk renderer, and
the SlotPool (active + release-tail slots with click-free crossfades and the ADSR
envelope).  Shared by the live synth (gol_synth.py), offline replay
(casynth_session.py) and the golden-master harness.

Defaults reproduce the historical sound bit-for-bit (see casynth_config ADSR notes).
"""
import colorsys

import numpy as np
from scipy import ndimage

from casynth_config import *
from casynth_core import extract, ENGINE_BY_ID


def midi_to_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def note_name(n):
    return f"{NAMES[n % 12]}{n // 12 - 1}"


# ──────────────────────────────────────────────────────────────────────────────
# GAME OF LIFE  (toroidal, identical to gol_life_synth.py)
# ──────────────────────────────────────────────────────────────────────────────
_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_S8 = np.ones((3, 3), np.uint8)


def step(grid):
    n = ndimage.convolve(grid, _NEIGH, mode='wrap')
    return ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)


# LAPLACIAN MAPPING  (extract / map_laplacian imported from casynth_core --
# single source of truth shared with mapping_bench.py).  This prototype sounds
# MAX_MODES_PER_OBJ partials per object; the bench uses K_MAX.  The mode count is
# passed explicitly at the call site (see analyse) so the algorithm stays shared.

# ──────────────────────────────────────────────────────────────────────────────
# OBJECT ANALYSIS  (GOL field -> list of voice specs)
# ──────────────────────────────────────────────────────────────────────────────

def hsv(h, v):
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def analyse(grid, f0, engine_id, params):
    """Segment the field into connected objects and compute voices via the
    SELECTED engine.

    engine_id : key into ENGINE_BY_ID -- picks the map_* function.
    params    : dict {arg: value} of that engine's live attributes (e.g.
                {'n':12,'spread':0.0,'alpha':1.0} for Laplace, {'n':16} for the
                harmonic engines).  'n' (partial count) is clamped to
                MAX_MODES_PER_OBJ; the engine silences higher slots (they ring
                out as tails).  Forwarded to the map_* as keyword args, so the
                same audio pipeline serves every engine -- harmonic mappings just
                return fixed f0*k freqs (no topology tails, only amp glide).

    Returns:
        labels  : labelled grid (0 = background)
        voices  : list of dicts with keys 'freqs', 'amps', 'pan', 'label_idx'
        color   : dict  label -> RGB colour tuple
    """
    labels, n = ndimage.label(grid, structure=_S8)
    if n == 0:
        return labels, [], {}

    fn = ENGINE_BY_ID[engine_id]['fn']
    kwargs = dict(params)
    kwargs['n'] = int(min(kwargs.get('n', N_PARTIALS_DEFAULT), MAX_MODES_PER_OBJ))
    _fullshape = bool(kwargs.get('fullshape', False))

    objs = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        objs.append((lab, len(xs), ys, xs))
    objs.sort(key=lambda o: -o[1])   # largest first

    voices, color = [], {}
    for idx, (lab, area, ys, xs) in enumerate(objs[:MAX_VOICES]):
        # Build sub-grid containing ONLY this object's cells
        # (raw grid slice may include neighbouring objects if bboxes overlap)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.uint8)
        sub[ys - r0, xs - c0] = 1
        patch = sub if _fullshape else extract(sub, PATCH_SIZE)

        cx = xs.mean() / max(GRID_W - 1, 1)   # stereo pan [0..1]

        freqs, amps = fn(patch, f0, **kwargs)

        # Colour: hue by object index (spread across spectrum), brightness by area
        hue = (idx / max(MAX_VOICES - 1, 1)) * 0.72
        val = float(np.clip(0.3 + 0.7 * min(area / 20.0, 1.0), 0.3, 1.0))
        color[lab] = hsv(hue, val)

        voices.append({
            'freqs':     freqs,             # shape (n,), n = params['n'] (<= MAX_MODES_PER_OBJ)
            'amps':      amps,              # shape (n,)
            'pan':       float(np.clip(cx, 0.0, 1.0)),
            'label_idx': lab,
        })

    for (lab, area, ys, xs) in objs[MAX_VOICES:]:
        color[lab] = C_CAPPED

    return labels, voices, color


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO ENGINE  -- flat slot pool with per-slot crossfade
# ──────────────────────────────────────────────────────────────────────────────

def render_chunk_laplacian(phase, amp_cur, pan_cur,
                           amp_tgt, pan_tgt, freq_slots, channels, gain_prev, gain):
    """Render one gapless chunk over the flat slot pool.

    gain_prev, gain : pre-clip master gain (MASTER_GAIN * volume) at the start and
    end of this chunk.  The gain is GLIDED from gain_prev to gain across the chunk
    (smoothstep) so dragging the volume slider doesn't step the level at a chunk
    boundary (that step was an audible click).  Applied BEFORE the clip, so lowering
    volume pulls the summed signal below the clip ceiling -- an honest headroom
    control (the timbre/spectrum is untouched, only the level).

    Returns (buf, peak, n_clip): buf is int16 (n_samples x channels); peak is the
    PRE-clip absolute peak (1.0 == 0 dBFS ceiling; >1.0 means we clipped and by
    how much); n_clip is the count of samples that exceeded the ceiling.

    Amplitudes and pan are linearly interpolated (glided) from their current
    values to the target values within each chunk.  Phase accumulation is
    continuous across chunks -- there is no phase reset.  Overlapping crossfade
    (old frequency fading out while new frequency rises) is achieved by SlotPool
    keeping two physical slots per voice×mode (front and back banks); both slots
    are rendered here simultaneously during a transition.

    Parameters (all length TOTAL_SLOTS+1, slot 0 unused)
    ---------
    phase     : float64 array, phase accumulator per slot (mutated in place)
    amp_cur   : float64 array, current amplitudes (mutated -> amp_tgt each chunk)
    pan_cur   : float64 array, current pan         (mutated -> pan_tgt each chunk)
    amp_tgt   : float64 array, target amplitudes
    pan_tgt   : float64 array, target pan
    freq_slots: float64 array, frequency in Hz per slot (0 = silent)

    Returns int16 buffer (n_samples x channels).
    """
    n = int(CHUNK_S * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    idx = np.arange(n)
    guard = 0.45 * SR

    for k in range(1, TOTAL_SLOTS + 1):
        if amp_cur[k] < 1e-4 and amp_tgt[k] < 1e-4:
            continue
        freq = freq_slots[k]
        if freq <= 0.0 or freq >= guard:
            amp_cur[k] = 0.0
            continue
        inc = TWO_PI * freq / SR
        wave = np.sin(phase[k] + inc * idx)
        a = amp_cur[k] + (amp_tgt[k] - amp_cur[k]) * _RAMP
        p = pan_cur[k] + (pan_tgt[k] - pan_cur[k]) * _RAMP
        L += wave * a * np.cos(p * np.pi / 2.0)
        R += wave * a * np.sin(p * np.pi / 2.0)
        phase[k] = (phase[k] + inc * n) % TWO_PI
        amp_cur[k] = amp_tgt[k]
        pan_cur[k] = pan_tgt[k]

    g = gain_prev + (gain - gain_prev) * _RAMP   # smooth gain ramp -> click-free volume drag
    L *= g
    R *= g
    # Pre-clip metering: peak relative to the 1.0 ceiling (0 dBFS) and how many
    # samples ran over.  Measured BEFORE the clip so the meter shows the true
    # overshoot the user is asked to tame with the volume knob.
    peak = float(max(np.abs(L).max(), np.abs(R).max()))
    n_clip = int(np.count_nonzero(np.abs(L) > 1.0) + np.count_nonzero(np.abs(R) > 1.0))
    L = np.clip(L, -1.0, 1.0)
    R = np.clip(R, -1.0, 1.0)

    if channels <= 1:
        data = ((L + R) * 0.5)[:, None]
    else:
        data = np.zeros((n, channels))
        data[:, 0] = L
        data[:, 1] = R

    return np.ascontiguousarray((data * 32767).astype(np.int16)), peak, n_clip


class SlotPool:
    """Active + tail slot pool with seamless (click-free) release tails.

    Layout (slots 1..TOTAL_SLOTS; slot 0 unused):
      ACTIVE slots 1..N_ACTIVE: one per voice×mode, holding the CURRENT sounding
        mode.  active(v,m) = 1 + v * MAX_MODES_PER_OBJ + m.
      TAIL slots N_ACTIVE+1..TOTAL_SLOTS: a pool of release-only slots.

    On a frequency change (topology event) or voice death, the OLD mode is NOT
    cut.  Its (freq, phase, current amplitude, pan) are MOVED into a free tail
    slot which then rings down over release_chunks, while the active slot restarts
    on the new frequency from amplitude 0 (fast one-chunk attack).  Because the
    tail continues the EXACT waveform the active slot was producing (same phase,
    same amplitude at the seam), the summed signal is continuous across the swap
    -> no click -- and tails of arbitrary length overlap into a pad.

    This replaces the old front/back 2-bank scheme, which had only two slots per
    voice×mode: a still-ringing tail got its amplitude slammed to 0 when the bank
    was reused for the next attack, a step discontinuity heard as a boundary click
    whenever release was long (DEV-2; confirmed on a session snapshot: forcing
    release short cut sharp transients 465 -> 38).

    If the tail pool is exhausted, the QUIETEST tail is stolen (smallest residual
    amplitude -> smallest click) -- graceful degradation under extreme stacking.
    """

    def __init__(self):
        sz = TOTAL_SLOTS + 1   # +1 so slot 0 is a harmless dummy
        self.freq_slots  = np.zeros(sz)        # Hz; 0 = silent
        self.amp_tgt     = np.zeros(sz)
        self.pan_tgt     = np.full(sz, 0.5)

        # Release countdown (in chunks) per slot; 0 = not releasing.
        self._release_cnt = np.zeros(sz, dtype=int)
        # Amplitude at the start of release (for linear step computation).
        self._release_amp0 = np.zeros(sz)
        # Release LENGTH (chunks) captured when each tail started, so a live
        # release-knob change mid-tail does not rescale an in-flight fade.
        self._release_len = np.ones(sz, dtype=int)

        # Last committed frequency per voice×mode active slot (to detect changes).
        self._freq_prev = np.zeros((MAX_VOICES, MAX_MODES_PER_OBJ))

        # ADSR envelope state per ACTIVE slot (tails ring out via the release
        # countdown instead).  phase: 0 idle, 1 attack, 2 decay, 3 sustain.
        # level: current 0..1 multiplier applied to the engine's per-mode amplitude.
        self._env_phase = np.zeros(sz, dtype=int)
        self._env_level = np.zeros(sz)

        # Diagnostics: count tail-pool steals (pool exhausted -> a still-ringing
        # tail had to be overwritten = a residual click) and the loudest stolen
        # amplitude.  Read by the probe; zero in normal (non-exhausted) operation.
        self.steals = 0
        self.steal_amp_max = 0.0

    @staticmethod
    def _active(v, m):
        """Slot index of the active (currently sounding) slot for voice v, mode m."""
        return 1 + v * MAX_MODES_PER_OBJ + m

    def _acquire_tail(self, amp_cur):
        """Return a tail slot for a new ringing-out mode: a free (silent, not
        releasing) one if available, else the quietest currently-ringing one
        (stealing it -> the smallest possible click)."""
        lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
        quietest, quiet_amp = lo, np.inf
        for s in range(lo, hi):
            if self._release_cnt[s] == 0 and amp_cur[s] < 1e-4:
                return s
            if amp_cur[s] < quiet_amp:
                quiet_amp, quietest = amp_cur[s], s
        # No free slot: stealing the quietest still-ringing tail.  Eviction below
        # keeps the quietest near zero, so this steal is normally near-silent.
        self.steals += 1
        self.steal_amp_max = max(self.steal_amp_max, float(quiet_amp))
        return quietest

    def _enforce_tail_budget(self, amp_cur):
        """When ringing tails exceed the pool minus a reserve, fast-fade the
        QUIETEST excess to zero over FAST_EVICT_CHUNKS (a quick, smooth release --
        not a cut) so slots free up before a loud tail must be stolen."""
        lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
        ring = [s for s in range(lo, hi)
                if self._release_cnt[s] > 0 or amp_cur[s] >= 1e-4]
        over = len(ring) - (N_TAIL - TAIL_RESERVE)
        if over <= 0:
            return
        ring.sort(key=lambda s: amp_cur[s])      # quietest first
        for s in ring[:over]:
            if self._release_cnt[s] == 0 or self._release_cnt[s] > FAST_EVICT_CHUNKS:
                self._release_cnt[s]  = FAST_EVICT_CHUNKS
                self._release_len[s]  = FAST_EVICT_CHUNKS
                self._release_amp0[s] = float(amp_cur[s])

    def _advance_env(self, a, attack_chunks, decay_chunks, sustain):
        """Advance the ADSR envelope of active slot `a` by one audio chunk.

        attack/decay are lengths in chunks (>=1); sustain is the held 0..1 level.
        Sets self._env_level[a] in place; release is NOT handled here (tail pool).
        """
        ph = self._env_phase[a]
        if ph == 1:                                   # attack: 0 -> 1 (peak)
            lvl = self._env_level[a] + 1.0 / attack_chunks
            if lvl >= 1.0:
                lvl, self._env_phase[a] = 1.0, 2      # -> decay
            self._env_level[a] = lvl
        elif ph == 2:                                 # decay: 1 -> sustain level
            lvl = self._env_level[a] - (1.0 - sustain) / decay_chunks
            if lvl <= sustain:
                lvl, self._env_phase[a] = sustain, 3  # -> sustain
            self._env_level[a] = lvl
        elif ph == 3:                                 # sustain: track live level
            self._env_level[a] = sustain
        # ph == 0 (idle): level stays at its current value (0 for a silent slot).

    def update(self, voices, phase, amp_cur, pan_cur, release_chunks,
               attack_chunks, decay_chunks, sustain):
        """Push a new voice list into the slot pool.

        phase / amp_cur / pan_cur : the engine's per-slot state arrays (mutated in
            place) -- needed so a replaced mode can be MOVED into a tail slot
            continuing the exact same waveform (phase + amplitude), seamlessly.
        release_chunks : current mode-tail release length in audio chunks (live
            release knob -- see RELEASE_MS_* / feed_audio).
        attack_chunks / decay_chunks / sustain : live ADSR knobs (see
            ATTACK_MS_* / DECAY_MS_* / SUSTAIN_*).  Each active slot carries an
            envelope (0..1) multiplying the engine's per-mode amplitude; it is
            (re)triggered to ATTACK on every mode onset and advanced one chunk per
            update via _advance_env.

        For each voice×mode:
          - frequency unchanged: advance the envelope, glide amp(=env·amp_new)/pan
            targets on the active slot.
          - frequency changed (or voice gone): if the old mode is audible, move it
            into a tail slot (seamless continuation) to ring out over
            release_chunks; then restart the active slot on the new frequency from
            amplitude 0 with the envelope re-triggered to ATTACK, or silence it if
            the mode is gone.
        Old (tail) and new (active) sound in parallel -> overlap, no gap, no click.
        """
        n_voices = min(len(voices), MAX_VOICES)

        # Free up slots BEFORE this cycle's spawns: fast-fade the quietest excess
        # tails so a per-step burst of new tails finds (near-)silent slots instead
        # of stealing loud ones.  Run every update (incl. no-spawn ones) to keep
        # headroom ready for the next GOL-step burst.
        self._enforce_tail_budget(amp_cur)

        # ── Phase 1: assign active slots, spawn tails for replaced modes ──────
        for v in range(MAX_VOICES):
            present = v < n_voices
            voice = voices[v] if present else None
            for m in range(MAX_MODES_PER_OBJ):
                a = self._active(v, m)
                if present:
                    freq_new = float(voice['freqs'][m]) if m < len(voice['freqs']) else 0.0
                    amp_new  = float(voice['amps'][m])  if m < len(voice['amps'])  else 0.0
                    pan_new  = voice['pan']
                else:
                    freq_new, amp_new, pan_new = 0.0, 0.0, pan_cur[a]

                freq_changed = abs(freq_new - self._freq_prev[v, m]) > 0.01
                if not freq_changed:
                    # STABLE: advance the envelope, glide amp(=env·amp_new)/pan.
                    self.freq_slots[a] = freq_new
                    self.pan_tgt[a] = pan_new
                    if freq_new > 0.0:
                        self._advance_env(a, attack_chunks, decay_chunks, sustain)
                        self.amp_tgt[a] = self._env_level[a] * amp_new
                    else:
                        self._env_phase[a] = 0
                        self._env_level[a] = 0.0
                        self.amp_tgt[a] = 0.0
                    self._freq_prev[v, m] = freq_new
                    continue

                # Frequency changed: ring the OLD mode out in a tail slot, if it
                # is still audible -- copy freq/phase/amp/pan so the tail continues
                # the active slot's exact waveform (no discontinuity at the seam).
                if amp_cur[a] > TAIL_MIN_AMP and self.freq_slots[a] > 0.0:
                    t = self._acquire_tail(amp_cur)
                    self.freq_slots[t] = self.freq_slots[a]
                    amp_cur[t]         = amp_cur[a]
                    phase[t]           = phase[a]
                    pan_cur[t]         = pan_cur[a]
                    self.pan_tgt[t]    = self.pan_tgt[a]
                    self._release_cnt[t]  = release_chunks
                    self._release_len[t]  = release_chunks
                    self._release_amp0[t] = float(amp_cur[a])
                    # amp_tgt[t] is stepped down by Phase 2 below.

                # Restart the active slot on the new frequency, re-triggering the
                # ADSR envelope to ATTACK from 0; or silence it if the mode is gone.
                # amp_cur=0 makes the phase value inaudible at the restart.
                amp_cur[a] = 0.0
                self.pan_tgt[a] = pan_new
                if freq_new > 0.0:
                    self.freq_slots[a] = freq_new
                    self._env_phase[a] = 1          # (re)trigger -> attack
                    self._env_level[a] = 0.0
                    self._advance_env(a, attack_chunks, decay_chunks, sustain)
                    self.amp_tgt[a] = self._env_level[a] * amp_new
                else:
                    self.freq_slots[a] = 0.0
                    self._env_phase[a] = 0
                    self._env_level[a] = 0.0
                    self.amp_tgt[a] = 0.0
                self._freq_prev[v, m] = freq_new

        # ── Phase 2: advance tail release countdowns ──────────────────────────
        # Done AFTER assignment so a tail spawned this cycle gets its first decay
        # step immediately (countdown release_chunks -> release_chunks-1 here).
        # The divisor is the per-slot length captured at spawn, so changing the
        # knob mid-tail does not jump an in-flight fade.
        for s in range(N_ACTIVE + 1, TOTAL_SLOTS + 1):
            cnt = self._release_cnt[s]
            if cnt <= 0:
                continue
            new_cnt = cnt - 1
            self._release_cnt[s] = new_cnt
            self.amp_tgt[s] = self._release_amp0[s] * new_cnt / self._release_len[s]
            # When done and silent, free the slot (render glides amp_cur to 0).
            if new_cnt == 0 and amp_cur[s] < 0.01:
                self.freq_slots[s] = 0.0
