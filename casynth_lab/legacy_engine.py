"""Adapter: the existing five-method synthesizer behind the SoundEngine API.

Wraps casynth_engine's analyse -> SlotPool.update -> render_chunk_laplacian
path exactly as the S1/S2 SideState did (same call order, same rounding, gain
applied once pre-clip), so the sound is byte-identical.  Partials, object
segmentation, SlotPool, phases and GEN envelopes stay INSIDE this adapter;
the bench only sees blocks.
"""
import numpy as np

from casynth_config import (CHUNK_S, TOTAL_SLOTS, GEN_ATTACK_DEFAULT,
                            GEN_DECAY_DEFAULT, GEN_SUSTAIN_DEFAULT,
                            GEN_RELEASE_DEFAULT)
from casynth_engine import analyse, SlotPool, render_chunk_laplacian
from .engine_api import SoundEngine

PAN_CENTER = 0.5                   # both channels identical (S1 rule)


def _gen_chunks(frac, interval_s):
    """Fraction of one automaton tick -> chunk count, as in gol_synth."""
    return max(1, round(frac * interval_s / CHUNK_S))


class LegacySynthEngine(SoundEngine):
    """One casynth_core map_* method (engine_id) rendered through the shared
    SlotPool additive path."""

    def __init__(self, ctx, params, engine_id):
        super().__init__(ctx, params)
        self.engine_id = engine_id
        interval = 1.0 / ctx.rate_hz
        self._release_chunks = _gen_chunks(GEN_RELEASE_DEFAULT, interval)
        self._attack_chunks = _gen_chunks(GEN_ATTACK_DEFAULT, interval)
        self._decay_chunks = _gen_chunks(GEN_DECAY_DEFAULT, interval)
        self._sustain = float(GEN_SUSTAIN_DEFAULT)
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

    def render(self, gain, t_samples):
        self.pool.update(self.voices, self.phase, self.amp_cur, self.pan_cur,
                         self._release_chunks, self._attack_chunks,
                         self._decay_chunks, self._sustain, amp_slew=False)
        buf, peak, n_clip = render_chunk_laplacian(
            self.phase, self.amp_cur, self.pan_cur, self.pool.amp_tgt,
            self.pool.pan_tgt, self.pool.freq_slots, self.ctx.channels,
            self.gain_prev, gain, 1.0)
        self.gain_prev = gain
        return buf, peak, n_clip

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

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
        for v in voices:
            v['pan'] = PAN_CENTER
        self.voices = voices
