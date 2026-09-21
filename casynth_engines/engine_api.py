"""Sound-engine interface of CASynth (S3; extended 2026-09-21 for a host that
plays notes -- the seam, memory/req-seam-2026-09-21.md).

A HOST owns the CA field, the clocks, the transport, the note, the articulation
and the output device.  An ENGINE instance receives the full field + its
excitation, its applied parameters and the render context, and returns finished
stereo int16 blocks with level/clip diagnostics.  It never touches the shared
field and never uses wall-clock time.  Two hosts exist: the demo bench
(casynth_lab: A/B sides, journal, catalog) and the prototype gol_synth.py
(piano, MIDI, VCA); an engine sees no difference between them.

Contract (all calls on the render thread, block-aligned, in host order):
    engine = factory(ctx, params)      # ctx: EngineContext, params: full dict
    engine.init(grid, exc, gain=0.0)   # (re)start on the CURRENT field, silent
    engine.update_field(grid, exc)     # field changed (step or painting)
    engine.set_params(params)          # full, registry-validated dict
    engine.set_envelope(attack, decay, sustain, release, amp_slew)   # optional
    engine.set_rate(rate_hz)                                         # optional
    buf, peak, n_clip = engine.render(gain, t_samples,
                                      gain_prev=None, transpose=1.0)
                                       # buf: int16 (ctx.block, 2); gain applied
                                       # ONCE inside (pre-clip master gain);
                                       # peak = pre-clip abs peak, n_clip = count
    engine.reset(gain=0.0)             # == init on the same field, all memory gone
    engine.display()                   # optional: read-only numbers for a UI
The host validates every returned block (check_block) before it can reach the
audio output or a WAV.

PITCH -- ctx.f0 is an ANCHOR, `transpose` is the note
    ctx.f0 is the fixed frequency the field is ANALYSED at: it is part of the
    identity of a scene and of a snapshot, and engines may cache anything
    derived from it at construction.  It is NOT the sounding note.
    The sounding pitch arrives per block as `transpose`, a scalar multiplier on
    every frequency the engine renders.  Phase is accumulated, only the
    increment changes, so a note change is phase-continuous and does NOT
    retrigger per-mode envelopes (the model of the prototype: the field is a
    free-running oscillator, the note is a transpose -- see
    casynth_engine.render_chunk_laplacian).
    Re-analysing the field at a live f0 instead is FORBIDDEN: the frequencies in
    the pool would change with the note, the pool would read that as a change of
    modes, and every note would retrigger an attack and spawn a tail.
    transpose=1.0 reproduces the engine's historical sound bit-for-bit.  An
    engine that cannot transpose (its pitch is a delay length, a table index...)
    leaves SUPPORTS_TRANSPOSE False and raises ValueError when asked for one, so
    a host never plays the wrong pitch silently; ask supports_transpose(engine).

GAIN -- one pre-clip multiply, glided across the block
    `gain` is the host's master gain for the END of this block.  The engine
    applies it ONCE, before the clip, and GLIDES to it from the previous block's
    gain across the block -- otherwise a dragged volume slider steps at every
    block boundary, which is an audible click.  The engine remembers the
    previous value itself; `gain_prev` is the host OVERRIDING that memory (a
    fresh instance under a crossfade, a continuation from a snapshot), and then
    the glide must start at exactly that value.

WHAT STAYS OUTSIDE THE ENGINE
    * The VCA and the note gate.  Note-on / note-off articulation is the host's
      envelope folded into `gain` (gol_synth: gain = MASTER_GAIN * vol *
      voice_env).  An engine knows nothing about notes being held.
    * The meter.  `peak` is the RAW pre-clip peak of THIS block and `n_clip` the
      count of samples over the ceiling; the decay of a level meter and the
      "clip" lamp are the host's (gol_synth keeps METER_DECAY, the bench counts
      clipped blocks).
    * The transport, the field and the automaton clock.

Envelopes and tempo (optional, 2026-09-21)
    set_envelope(attack, decay, sustain, release, amp_slew) and set_rate(rate_hz)
    are called on a block boundary when the host's envelope knobs or its tempo
    change.  A/D/R are FRACTIONS of one automaton tick, S a 0..1 level,
    amp_slew a bool (see casynth_config GEN_*).  The base class ignores both:
    an engine with envelopes of its own works exactly as before.

Snapshot (S5, optional -- an engine without it still works in a host, it just
cannot be continued exactly):
    state = engine.export_state()      # plain dict: JSON scalars/lists/dicts +
                                       # numpy arrays (no objects); copied, so the
                                       # live engine may keep rendering afterwards
    engine.restore_state(grid, exc, state)
                                       # rebuild the instance so that the next
                                       # render() equals the one the exported
                                       # engine would have produced; ValueError
                                       # on an incompatible state.  Caches may be
                                       # recomputed from grid/exc/params only if
                                       # that cannot change the output.
    engine.STATE_VERSION               # class attribute, part of the state
Both calls run on the render thread at a block boundary (never in the audio
callback).  supports_snapshot(engine) tells whether a class implements them.
"""
import numpy as np


PAN_CENTER_MODE = 'center'
PAN_FIELD_MODE = 'field'
PAN_MODES = (PAN_CENTER_MODE, PAN_FIELD_MODE)


class EngineContext:
    """Render context handed to an engine at construction (immutable).

    `f0` is the analysis anchor, not the sounding note -- the note is the
    per-block `transpose` of render() (see the module doc).

    `pan` (2026-09-21) is how the HOST wants the field laid across the stereo
    image: 'center' -- both channels identical, which is what the bench has
    always rendered so that an A/B difference is never a difference of position
    -- or 'field', where a voice sits where its figure sits, which is what the
    prototype has always played.  It is a property of the host, not of the
    engine's spectrum law, and only engines that have a per-voice position read
    it.  Default 'center': every scene and record made so far is unchanged."""
    __slots__ = ('sr', 'block', 'channels', 'f0', 'level', 'rate_hz', 'pan')

    def __init__(self, sr, block, channels, f0, level, rate_hz, pan=PAN_CENTER_MODE):
        self.sr = int(sr)
        self.block = int(block)
        self.channels = int(channels)
        self.f0 = float(f0)
        self.level = float(level)
        self.rate_hz = float(rate_hz)     # automaton steps / s (tick-relative envelopes)
        if pan not in PAN_MODES:
            raise ValueError(f"EngineContext: pan must be one of {PAN_MODES}, got {pan!r}")
        self.pan = pan


class SoundEngine:
    """Base class: subclasses implement the methods below."""

    #: the engine renders a per-block `transpose` (see the module doc)
    SUPPORTS_TRANSPOSE = False

    def __init__(self, ctx, params):
        self.ctx = ctx
        self.params = dict(params)

    def init(self, grid, exc, gain=0.0):
        """(Re)start on the CURRENT field: all audio memory gone, silent.
        `gain` seeds the gain glide of the first block."""
        raise NotImplementedError

    def update_field(self, grid, exc):
        raise NotImplementedError

    def set_params(self, params):
        self.params = dict(params)

    def set_envelope(self, attack, decay, sustain, release, amp_slew):
        """Live envelope settings of the host (optional; ignored by default --
        see the module doc).  A/D/R are fractions of one automaton tick."""

    def set_rate(self, rate_hz):
        """Live automaton tempo of the host (optional; ignored by default)."""

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        """One block: (int16 (ctx.block, ctx.channels), pre-clip peak, n_clip).

        gain_prev : start the block's gain glide from exactly this value instead
                    of the remembered one (None = the engine's own memory).
        transpose : scalar multiplier on every rendered frequency (the note)."""
        raise NotImplementedError

    def reset(self, gain=0.0):
        """== init on the same field, all audio memory gone."""
        raise NotImplementedError

    def display(self):
        """Read-only numbers a UI may show (None = none).  The display never
        takes part in the sound."""
        return None

    # -- helpers for subclasses -------------------------------------------------
    def _check_transpose(self, transpose):
        """Refuse a note an engine cannot play instead of playing a wrong one."""
        if transpose != 1.0 and not self.SUPPORTS_TRANSPOSE:
            raise ValueError(f"engine {type(self).__name__} cannot transpose "
                             f"(transpose={transpose!r}); it renders at ctx.f0 only")

    # -- snapshot (optional) ----------------------------------------------------
    STATE_VERSION = None       # set by engines that implement export/restore

    def export_state(self):
        raise NotImplementedError

    def restore_state(self, grid, exc, state):
        raise NotImplementedError


def supports_snapshot(engine):
    """True when the engine CLASS overrides both snapshot methods and
    declares a STATE_VERSION."""
    cls = type(engine)
    return (cls.export_state is not SoundEngine.export_state
            and cls.restore_state is not SoundEngine.restore_state
            and getattr(cls, 'STATE_VERSION', None) is not None)


def supports_transpose(engine):
    """True when the engine plays a per-block `transpose` (a host may give it
    a note); False = it renders at ctx.f0 only."""
    return bool(getattr(type(engine), 'SUPPORTS_TRANSPOSE', False))


def reads_gen_envelope(engine):
    """True when the engine CLASS implements set_envelope -- the host's live GEN
    A/D/S/R knobs act on it.  False = it carries envelopes of its own and the
    knobs are ignored, which a UI must SAY instead of offering four knobs that
    do nothing (the registry's `gen_envelope` mirrors this so a host can ask
    before it builds an instance)."""
    return type(engine).set_envelope is not SoundEngine.set_envelope


def supports_amp_slew(engine):
    """True when the GEN amplitude-slew toggle acts on this engine: it reads the
    knobs AND has somewhere to put the slew (the SlotPool has; an engine with a
    per-source envelope of its own has not)."""
    return (reads_gen_envelope(engine)
            and bool(getattr(type(engine), 'SUPPORTS_AMP_SLEW', False)))


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
