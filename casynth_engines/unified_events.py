"""The Events articulation of the unified Laplace engine
(REQ memory/req-unified-laplace-2026-09-21.md).

The articulation is object_resonators unchanged -- its tracker, its packets
(a = e/(e+2)), its detector disk, its Fixed decay (gamma = ln 1000 / T60), its
attack, its slots and its tails.  A cell here only says WHICH of those settings
the axis offers (objects_params) and reads the bank back; at Sine it reads it
back through the law's own kernel, so the cell is `ca_object_resonators` bit for
bit and none of its five gates is re-blessed.

The module lives apart from laplace_unified so that the Env cells -- and with
them the prototype's default tab -- never pull numba or the 2000-line resonator
bank just to play a Laplace line (the seam's A1 win).
"""
import numpy as np

from . import object_resonators as orz
from . import laplace_carriers as lc


def objects_params(params):
    """The Objects parameter set the Events articulation pins (REQ section 4):
    the Disk detector, the Laplace spectrum, a uniform packet, the Fixed decay --
    and the four knobs the axis does offer.  `frequency_scale` never acts under
    the Laplace law; ctx.f0 and the note's transpose play its part."""
    out = {k: params[k] for k in orz.SPECTRUM_KEYS}
    out.update(detector=orz.DET_DISK, spectrum=orz.SPEC_LAPLACE,
               excitation=orz.EXC_UNIFORM, birth_strength=1.0,
               decay_law=orz.LAW_FIXED, frequency_scale=220.0,
               events=int(params['events']),
               radius_mul=float(params['radius_mul']),
               decay_s=float(params['decay_s']),
               attack_ms=float(params['attack_ms']))
    return out


class _EventsCell:
    """What every Events voicing shares: the Objects engine underneath."""

    def __init__(self, ctx, params, grid, exc, gain, strike):
        self.ctx = ctx
        self.n = int(ctx.block)
        self.sr = float(ctx.sr)
        self.obj = orz.ObjectResonatorsEngine(ctx, objects_params(params))
        self.obj.init(grid, exc, gain)
        if not strike:
            # a switch of an axis is not an event: the articulation takes the
            # current field as its OWN previous field and waits for a change
            self.obj.prime_silent()

    def set_field(self, grid, exc):
        self.obj.update_field(grid, exc)

    def set_params(self, params):
        self.obj.set_params(objects_params(params))

    def display(self):
        return self.obj.display()

    def _selection(self, transpose, audible_only):
        """The live pairs of the bank: a mode inside n_live whose frequency
        sounds, and -- for a voicing that carries trajectories -- one that is
        either driven or still audible."""
        obj = self.obj
        M = obj.zre.shape[1]
        j = np.arange(M)[None, :]
        live = (j < obj.nlive[:, None]) & (obj.role != orz.ROLE_FREE)[:, None]
        freq = obj.ffreq * float(transpose)
        sel = live & (freq > 0.0) & (freq < lc.BAND_ZERO * self.sr)
        if audible_only:
            fed = np.where(obj.role == orz.ROLE_ACTIVE, obj.ndrive, obj.npulse)
            mag = np.hypot(obj.zre, obj.zim)
            sel = sel & ((j < fed[:, None]) | (mag > orz.TAIL_FLOOR))
        return sel, freq


class EventsBankCell(_EventsCell):
    """Events + Bank: the bank of the struck figure, mode by mode.

    At Sine the cell runs the law's own kernel on the law's own state, so it IS
    `ca_object_resonators` -- the byte anchor of REQ section 3 is the gate that
    engine already has, not a new claim."""

    def __init__(self, ctx, params, grid, exc, gain, strike):
        super().__init__(ctx, params, grid, exc, gain, strike)
        self.wave = int(params['waveform'])
        self._out = np.zeros((self.n, 2))

    def set_params(self, params):
        super().set_params(params)
        self.wave = int(params['waveform'])

    def render_float(self, gain, gain_prev, transpose, env=None):
        obj = self.obj
        obj.begin_block(gain, transpose, gain_prev)
        out = self._out
        obj._kernel(self.n, out)                    # the law's own kernel
        obj.end_block()
        return out.copy()

    def display(self):
        d = super().display()
        d.update(voicing='bank', wave=int(self.wave),
                 wave_name=lc.WAVE_NAMES[int(self.wave)])
        return d
