"""Sound-engine interface of the demo bench (S3).

The BENCH owns the shared CA field, clocks, command queue/journal, transport,
the A/B instances, the monitor switch, live output and WAV export.
An ENGINE instance (one per side) receives the full field + its excitation,
its applied parameters and the render context, and returns finished stereo
int16 blocks with level/clip diagnostics.  It never touches the shared field
and never uses wall-clock time.

Contract (all calls on the render thread, in journal order, block-aligned):
    engine = factory(ctx, params)      # ctx: EngineContext, params: full dict
    engine.init(grid, exc)             # (re)start on the CURRENT field, silent
    engine.update_field(grid, exc)     # field changed (step or painting)
    engine.set_params(params)          # full, registry-validated dict
    buf, peak, n_clip = engine.render(gain, t_samples)
                                       # buf: int16 (ctx.block, 2); gain applied
                                       # ONCE inside (pre-clip master gain);
                                       # peak = pre-clip abs peak, n_clip = count
    engine.reset()                     # == init on the same field, all memory gone
The bench validates every returned block (check_block) before it can reach
the audio output or a WAV.
"""
import numpy as np


class EngineContext:
    """Render context handed to an engine at construction (immutable)."""
    __slots__ = ('sr', 'block', 'channels', 'f0', 'level', 'rate_hz')

    def __init__(self, sr, block, channels, f0, level, rate_hz):
        self.sr = int(sr)
        self.block = int(block)
        self.channels = int(channels)
        self.f0 = float(f0)
        self.level = float(level)
        self.rate_hz = float(rate_hz)     # automaton steps / s (tick-relative envelopes)


class SoundEngine:
    """Base class: subclasses implement the five methods below."""

    def __init__(self, ctx, params):
        self.ctx = ctx
        self.params = dict(params)

    def init(self, grid, exc):
        raise NotImplementedError

    def update_field(self, grid, exc):
        raise NotImplementedError

    def set_params(self, params):
        self.params = dict(params)

    def render(self, gain, t_samples):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class EngineBlockError(RuntimeError):
    """An engine returned a block that cannot be sent to the audio output."""


def check_block(buf, ctx, engine_id=''):
    """Reject anything but a contiguous int16 (block, channels) array."""
    if not isinstance(buf, np.ndarray) or buf.dtype != np.int16:
        raise EngineBlockError(f"engine {engine_id!r}: block must be an int16 ndarray, "
                               f"got {type(buf).__name__} {getattr(buf, 'dtype', '')}")
    if buf.shape != (ctx.block, ctx.channels):
        raise EngineBlockError(f"engine {engine_id!r}: block shape {buf.shape} != "
                               f"{(ctx.block, ctx.channels)}")
    return buf
