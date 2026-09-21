"""Adapter: the existing five-method synthesizer behind the SoundEngine API.

Wraps casynth_engine's analyse -> SlotPool.update -> render_chunk_laplacian
path exactly as the S1/S2 SideState did (same call order, same rounding, gain
applied once pre-clip), so the sound is byte-identical.  Partials, object
segmentation, SlotPool, phases and GEN envelopes stay INSIDE this adapter;
the bench only sees blocks.
"""
import numpy as np

from casynth_config import (CHUNK_S, TOTAL_SLOTS, MAX_VOICES, MAX_MODES_PER_OBJ,
                            GEN_ATTACK_DEFAULT, GEN_DECAY_DEFAULT,
                            GEN_SUSTAIN_DEFAULT, GEN_RELEASE_DEFAULT)
from casynth_engine import analyse, SlotPool, render_chunk_laplacian
from .engine_api import SoundEngine, PAN_FIELD_MODE

PAN_CENTER = 0.5                   # both channels identical (S1 rule)

# Snapshot (S5): everything render() reads besides grid/exc/params.  `voices`
# is NOT stored: it is a pure function of (grid, exc, params) recomputed by
# restore_state through the same analyse() call -- the arrays below are the
# only audio memory.  Slot arrays: (TOTAL_SLOTS + 1,); _freq_prev: (MAX_VOICES,
# MAX_MODES_PER_OBJ).
_SLOT_ARRAYS = ('phase', 'amp_cur', 'pan_cur')
_POOL_ARRAYS = ('freq_slots', 'amp_tgt', 'pan_tgt', '_release_cnt', '_release_amp0',
                '_release_len', '_env_phase', '_env_level', '_amp_smooth')


def _gen_chunks(frac, interval_s):
    """Fraction of one automaton tick -> chunk count, as in gol_synth."""
    return max(1, round(frac * interval_s / CHUNK_S))


class GenEnvelopeKnobs:
    """The host's LIVE GEN ADSR knobs, for an engine whose per-mode envelope is
    the SlotPool's: A/D/R are FRACTIONS of one automaton tick (so the texture
    scales with the tempo), S is a 0..1 level.

    ONE copy of that arithmetic, mixed into every engine that has the envelope.
    Until 2026-09-21 only LegacySynthEngine read the knobs, so on the
    prototype's "Laplace waves" and "Laplace FM" tabs the whole GEN block was
    dead: those engines run the same SlotPool envelope but held the defaults
    they were built with, and neither followed BPM.  A host that says nothing
    still gets exactly the defaults, so a scene without an `envelope` renders
    byte-for-byte as before.

    SUPPORTS_AMP_SLEW: amplitude slew is a SlotPool feature; an engine with its
    own per-source envelope (the Filter carriers, the FM sources) has nowhere to
    put it and says so, instead of swallowing the toggle in silence."""

    SUPPORTS_AMP_SLEW = True

    def _init_gen_env(self, rate_hz):
        self._rate_hz = float(rate_hz)
        self._attack = float(GEN_ATTACK_DEFAULT)
        self._decay = float(GEN_DECAY_DEFAULT)
        self._sustain = float(GEN_SUSTAIN_DEFAULT)
        self._release = float(GEN_RELEASE_DEFAULT)
        self._amp_slew = False
        self._recount()

    def _recount(self):
        """Chunk counts of the GEN envelope at the current fractions and tempo --
        the same arithmetic the prototype ran every block before it hosted an
        engine (gol_synth._render_loop)."""
        interval = 1.0 / self._rate_hz
        self._release_chunks = _gen_chunks(self._release, interval)
        self._attack_chunks = _gen_chunks(self._attack, interval)
        self._decay_chunks = _gen_chunks(self._decay, interval)

    def set_envelope(self, attack, decay, sustain, release, amp_slew):
        self._attack = float(attack)
        self._decay = float(decay)
        self._sustain = float(sustain)
        self._release = float(release)
        self._amp_slew = bool(amp_slew) and self.SUPPORTS_AMP_SLEW
        self._recount()

    def set_rate(self, rate_hz):
        rate_hz = float(rate_hz)
        if rate_hz > 0.0 and rate_hz != self._rate_hz:
            self._rate_hz = rate_hz
            self._recount()


class LegacySynthEngine(GenEnvelopeKnobs, SoundEngine):
    """One casynth_core map_* method (engine_id) rendered through the shared
    SlotPool additive path.

    The note is a `transpose` at render: the pool holds the frequencies analysed
    at ctx.f0 and render_chunk_laplacian multiplies them per block, so a note
    change is phase-continuous and retriggers nothing (see engine_api)."""

    SUPPORTS_TRANSPOSE = True

    def __init__(self, ctx, params, engine_id):
        super().__init__(ctx, params)
        self.engine_id = engine_id
        # GEN A/D/R are FRACTIONS of one automaton tick, so the per-mode texture
        # scales with the tempo.  Both the fractions and the tempo are LIVE knobs
        # in the prototype: set_envelope / set_rate hand them over on a block
        # boundary (2026-09-21, GenEnvelopeKnobs).  The defaults reproduce the
        # bench exactly.
        self._init_gen_env(ctx.rate_hz)
        self.voices = []
        self._grid = None
        self._exc = None
        self._clear_audio(gain=0.0)

    # -- SoundEngine ------------------------------------------------------------
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

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        if gain_prev is not None:
            self.gain_prev = float(gain_prev)      # the host overrides the glide start
        self.pool.update(self.voices, self.phase, self.amp_cur, self.pan_cur,
                         self._release_chunks, self._attack_chunks,
                         self._decay_chunks, self._sustain, amp_slew=self._amp_slew)
        buf, peak, n_clip = render_chunk_laplacian(
            self.phase, self.amp_cur, self.pan_cur, self.pool.amp_tgt,
            self.pool.pan_tgt, self.pool.freq_slots, self.ctx.channels,
            self.gain_prev, gain, transpose)
        self.gain_prev = gain
        return buf, peak, n_clip

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

    # -- snapshot -----------------------------------------------------------------
    STATE_VERSION = 1

    def export_state(self):
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  params=dict(self.params), gain_prev=float(self.gain_prev),
                  steals=int(self.pool.steals), steal_amp_max=float(self.pool.steal_amp_max),
                  slots=int(TOTAL_SLOTS), voices=int(MAX_VOICES), modes=int(MAX_MODES_PER_OBJ))
        for name in _SLOT_ARRAYS:
            st[name] = getattr(self, name).copy()
        for name in _POOL_ARRAYS:
            st[name] = getattr(self.pool, name).copy()
        st['_freq_prev'] = self.pool._freq_prev.copy()
        return st

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {self.engine_id}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r} "
                             f"!= {self.STATE_VERSION}")
        if state.get('engine_id') != self.engine_id:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, "
                             f"not {self.engine_id!r}")
        if (state.get('slots'), state.get('voices'), state.get('modes')) !=                 (TOTAL_SLOTS, MAX_VOICES, MAX_MODES_PER_OBJ):
            raise ValueError(f"engine {self.engine_id}: slot layout "
                             f"{state.get('slots')}/{state.get('voices')}/{state.get('modes')}"
                             f" != {TOTAL_SLOTS}/{MAX_VOICES}/{MAX_MODES_PER_OBJ}")
        sz = TOTAL_SLOTS + 1
        arrays = {}
        for name in _SLOT_ARRAYS + _POOL_ARRAYS + ('_freq_prev',):
            a = state.get(name)
            want = (MAX_VOICES, MAX_MODES_PER_OBJ) if name == '_freq_prev' else (sz,)
            if not isinstance(a, np.ndarray) or a.shape != want:
                raise ValueError(f"engine {self.engine_id}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {want}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {self.engine_id}: state array {name!r} not finite")
            arrays[name] = a
        # fresh pool + arrays, then overwrite (dtypes of the live pool are kept)
        self._clear_audio(float(state['gain_prev']))
        for name in _SLOT_ARRAYS:
            getattr(self, name)[:] = arrays[name]
        for name in _POOL_ARRAYS:
            getattr(self.pool, name)[:] = arrays[name]
        self.pool._freq_prev[:] = arrays['_freq_prev']
        self.pool.steals = int(state.get('steals', 0))
        self.pool.steal_amp_max = float(state.get('steal_amp_max', 0.0))
        self.params = dict(state['params'])
        self.update_field(grid, exc)          # voices = analyse(grid, exc, params)

    # -- internals --------------------------------------------------------------
    def _clear_audio(self, gain):
        self.pool = SlotPool()
        sz = TOTAL_SLOTS + 1
        self.phase = np.zeros(sz)
        self.amp_cur = np.zeros(sz)
        self.pan_cur = np.full(sz, PAN_CENTER)
        self.gain_prev = gain

    def _analyse(self):
        _labels, voices, _color = analyse(self._grid, self.ctx.f0, self.engine_id,
                                          self.params, exc=self._exc)
        if self.ctx.pan != PAN_FIELD_MODE:
            # the bench's rule since S1: both channels identical, so an A/B
            # difference is never a difference of position.  A host that asks for
            # 'field' keeps the position analyse() gave each voice -- which is
            # what the prototype has always played.
            for v in voices:
                v['pan'] = PAN_CENTER
        self.voices = voices
