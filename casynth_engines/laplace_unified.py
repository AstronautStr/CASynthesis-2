"""The unified Laplace engine (REQ memory/req-unified-laplace-2026-09-21.md):
ONE engine whose timbre is two independent axes -- how a mode is ARTICULATED and
what SOUNDS it.

    field -> figures -> modes (f_j, phi_j, pan)
       |
    articulation:  Env | Events            -> the amplitude trajectory a_j(t)
       |
    voicing:       Bank (Sine|Saw|Square) | FM
       |
    output: pan, gain glide, clip, meter

Until 2026-09-21 the four Laplace engines were monoliths in which the decaying
mode WAS the oscillator, so "Events + Saw" or "Events + FM" simply did not exist
as code.  The split above is possible because in the Objects kernel a mode lives
as  z <- r e^{i omega} z + strike  and sounds as Re(z) (object_resonators._render):
between strikes that is exactly a0 * r^n * cos(omega n + phi0), so the amplitude
|z| and the phase are known in closed form and the decay can be handed to a layer
of its own.

THE SIX CELLS -- four are older engines and keep their bytes, two are new

  Env    + Bank + Sine       == `laplacian`              (golden master)
  Env    + Bank + Saw/Square == `laplace_carriers` Wave bank      (gate 1r)
  Env    + FM                == `laplace_fm`                      (gate 1s)
  Events + Bank + Sine       == `ca_object_resonators`            (gates 1l-1q)
  Events + Bank + Saw/Square == NEW (section 6 of the REQ)
  Events + FM                == NEW (section 6 of the REQ)

The first four are not re-derived here: the cell IS the implementation those
gates already pin (_BankVoices of laplace_carriers, _FMSources of laplace_fm,
ObjectResonatorsEngine itself and, inside it, the very kernel `_render`).  A
byte anchor is therefore not a claim to be proved once -- it is a gate that is
already written, and tests/test_laplace_unified.py runs each of the four against
its old engine block by block.

THE TWO NEW CELLS -- the Events articulation, sounded by something else

The Objects law is NOT rewritten.  The tracker, the packets (a = e/(e+2)), the
detector disk, the Fixed decay (gamma = ln 1000 / T60), the attack smoothing, the
slots and the tails are its own; only the READOUT of a mode changes:

    Sine        : b_s = sum_j w_j Re(z_j)                    (the law's own kernel)
    Saw/Square  : b_s = sum_j w_j |z_j| W(theta_j; f_j)
    FM          : y_s = A_s(t) sin(theta_c + sum_j beta_j(t) sin(theta_j))
                  a_j(t) = w_j |z_j(t)|,  beta_j = I a_j,  A_s = sqrt(sum_j a_j^2)

theta_j = arg(z_j) + pi/2 is the mode's OWN phase, read out of the state rather
than kept beside it, so the h = 1 term of W is |z| sin(theta) = Re(z) exactly: a
wave is the sine line with harmonics on top.  W, the band limit b(h f) taken at
the frequency of every harmonic of every wave, and the whole FM law (beta in
RADIANS, divided by nothing, one carrier per figure, one phase sum, OVERSAMPLE x
sr and the fixed DC blocker) are the two older REQs unchanged.  What is new --
and what makes these a new law rather than a consequence of an old one -- is that
beta_j now DECAYS with the mode: a struck figure is an FM percussion whose index
falls with its own amplitude.  See casynth_engines/unified_events.py.

SWITCHING AN AXIS never makes an impulse and never cuts.  The outgoing cell keeps
rendering and is faded out over one voice release while the incoming one fades in;
a cell that has finished fading out is dropped, and a cell that is created for a
switch starts SILENT -- the Events articulation is primed with the current field
as its own previous field, so it has nothing to strike until the next change of
the automaton.  (A fresh init of the whole engine keeps the Objects rule: the
first boundary compares the field with zeros and every live cell is a birth.)
Inside the Bank cell a change of the waveform is a one-block cross-blend of the
two readouts on the pool's own ramp, the rule the wave bank already uses.

WHAT IS DELIBERATELY NOT HERE (user's decision, REQ section 4): the Modal / Common
age decay laws, Birth position and Birth strength, the Own detector, the Figure
spectrum, `frequency_scale` (ctx.f0 and the note's transpose play its part) and
the Filter method of the carriers.  Torus segmentation stays INSIDE Events: the
Env articulation keeps reading casynth_engine.analyse, which labels without the
seam, until that is changed and its golden master re-taken.

OLD IDS ARE ALIASES (REQ section 5).  `laplacian`, `laplace_carriers`,
`laplace_fm` and `ca_object_resonators` stay in the registry with their own
parameter sets, their own labels and their own snapshots -- their FACTORY now
comes through create() here, which resolves the id to the cell its axes pin and
hands back that cell's engine.  So one place knows which law an id means, scenes,
snapshots and the records of lab_catalog/ replay byte for byte, and no gate of
those four laws is re-blessed.

Only sr 44100 / block 352 / stereo: the Events articulation is the Objects engine
and that is its restriction.
"""
import math

import numpy as np

from casynth_config import CHUNK_S, TWO_PI, _RAMP, MAX_VOICES
from casynth_core import ENGINE_BY_ID
from casynth_engine import analyse
from .engine_api import SoundEngine, PAN_FIELD_MODE
from .legacy_engine import PAN_CENTER, GenEnvelopeKnobs, LegacySynthEngine
from .registry import EngineSpec, register
from . import laplace_carriers as lc
from . import laplace_fm as lfm

ENGINE_ID = 'laplace_unified'
LABEL = 'Laplace+'

# the seven spectrum settings of the old Laplace, metadata from the core registry
LAPLACE_PARAMS = tuple(tuple(p) for p in ENGINE_BY_ID['laplacian']['params'])
SPECTRUM_KEYS = tuple(p[0] for p in LAPLACE_PARAMS)   # n spread alpha shape harm fullshape dyn

ARTIC_ENV, ARTIC_EVENTS = 0, 1
VOICE_BANK, VOICE_FM = 0, 1
ARTIC_NAMES = ('Env', 'Events')
VOICE_NAMES = ('Bank', 'FM')
WAVE_NAMES = lc.WAVE_NAMES                       # Sine / Saw / Square
EVENT_NAMES = ('Both', 'Births', 'Deaths')
WF_SINE, WF_SAW, WF_SQUARE = lc.WF_SINE, lc.WF_SAW, lc.WF_SQUARE

RADIUS_RANGE = (0.25, 4.0)                       # the slider range of `rad` (REQ section 2)

# The seven spectral settings act in EVERY cell, so they come first (REQ section 2);
# then the two axes, then the knobs each axis brings.
PARAMS = list(LAPLACE_PARAMS) + [
    ('artic', 'artic', 0, 1, True, ARTIC_ENV),
    ('voice', 'voice', 0, 1, True, VOICE_BANK),
    ('waveform', 'wave', 0, 2, True, WF_SINE),
    ('fm_depth', 'FM d', 0.0, 4.0, False, 1.0),
    ('events', 'ev', 0, 2, True, 0),
    ('radius_mul', 'rad', 0.0, math.inf, False, 1.0),
    ('decay_s', 'dec', 0.20, 1.50, False, 0.80),
    ('attack_ms', 'atk', 0.0, 20.0, False, 0.0),
]
CHOICES = {'artic': ARTIC_NAMES, 'voice': VOICE_NAMES, 'waveform': WAVE_NAMES,
           'events': EVENT_NAMES}
RANGES = {'radius_mul': RADIUS_RANGE}
OPTIONAL_PARAMS = {p[0]: p[5] for p in PARAMS}

RAMP_MS = 20.0                    # the ramp every engine of this family uses
XFADE_MIN_BLOCKS = max(1, int(round(RAMP_MS / 1000.0 / CHUNK_S)))

OVERLAY_TEXT = "Laplace+: articulation x voicing on one Laplacian spectrum"


# ── the axes an old id pins (REQ section 5) ───────────────────────────────────
#
# An alias keeps its OWN parameter set; `pin` says which cell of this engine its
# values mean, and `axes` how to read a unified parameter dict out of them.

def _pin_laplacian(params):
    return dict(artic=ARTIC_ENV, voice=VOICE_BANK, waveform=WF_SINE)


def _pin_carriers(params):
    """Wave bank -> Env + Bank with its waveform.  The Filter method is not on any
    axis of this engine (REQ section 5) and keeps its own implementation."""
    if int(params.get('method', lc.METHOD_BANK)) != lc.METHOD_BANK:
        return None
    return dict(artic=ARTIC_ENV, voice=VOICE_BANK,
                waveform=int(params.get('waveform', WF_SINE)))


def _pin_fm(params):
    return dict(artic=ARTIC_ENV, voice=VOICE_FM,
                fm_depth=float(params.get('fm_depth', 1.0)))


def _pin_objects(params):
    """Objects -> Events + Bank + Sine.  The settings REQ section 4 leaves out
    (Own, Figure, Birth position / strength, the age decay laws, frequency_scale)
    are outside the axes: such a parameter set is not a cell of this engine."""
    import casynth_engines.object_resonators as orz
    if (int(params.get('detector', orz.DET_DISK)) != orz.DET_DISK
            or int(params.get('spectrum', orz.SPEC_FIGURE)) != orz.SPEC_LAPLACE
            or int(params.get('excitation', orz.EXC_UNIFORM)) != orz.EXC_UNIFORM
            or int(params.get('decay_law', orz.LAW_FIXED)) != orz.LAW_FIXED
            or float(params.get('birth_strength', 1.0)) != 1.0):
        return None
    return dict(artic=ARTIC_EVENTS, voice=VOICE_BANK, waveform=WF_SINE,
                events=int(params.get('events', 0)),
                radius_mul=float(params.get('radius_mul', 1.0)),
                decay_s=float(params.get('decay_s', 0.80)),
                attack_ms=float(params.get('attack_ms', 0.0)))


PINNED = {
    'laplacian': _pin_laplacian,
    'laplace_carriers': _pin_carriers,
    'laplace_fm': _pin_fm,
    'ca_object_resonators': _pin_objects,
}


def unified_params(engine_id, params):
    """The unified parameter dict an alias's own parameter set means, or None when
    those values are outside the axes (the Filter method, an Objects setting REQ
    section 4 leaves out).  The seven spectral settings are shared verbatim."""
    fn = PINNED.get(engine_id)
    if fn is None:
        return None
    axes = fn(params)
    if axes is None:
        return None
    out = dict(OPTIONAL_PARAMS)
    for k in SPECTRUM_KEYS:
        if k in params:
            out[k] = params[k]
    out.update(axes)
    return out


def create(ctx, params, engine_id=ENGINE_ID):
    """The factory of this engine AND of every id it collapsed.

    An alias resolves to the cell its axes pin, which is that law's own engine --
    so `registry.create('ca_object_resonators', ...)` still hands back an
    ObjectResonatorsEngine, with its parameters, its snapshot and its bytes."""
    if engine_id == ENGINE_ID:
        return UnifiedLaplaceEngine(ctx, params)
    if engine_id == 'laplacian':
        return LegacySynthEngine(ctx, params, 'laplacian')
    if engine_id == lc.ENGINE_ID:
        return lc.LaplaceCarriersEngine(ctx, params)
    if engine_id == lfm.ENGINE_ID:
        return lfm.LaplaceFMEngine(ctx, params)
    if engine_id == 'ca_object_resonators':
        import casynth_engines.object_resonators as orz
        return orz.ObjectResonatorsEngine(ctx, params)
    raise KeyError(f"{ENGINE_ID}: {engine_id!r} is not an id of this engine")


# ── Env cells: the pool of the baseline, voiced two ways ─────────────────────

class _EnvCell:
    """The Env articulation: casynth_engine.analyse + the GEN envelope of the
    SlotPool, which is what the baseline, the wave bank and the FM engine have
    always shared.  The cell adds nothing to it -- the voicings below ARE those
    engines' own machinery, so each of the three Env cells is its old engine."""

    def __init__(self, ctx, params, grid, exc, gain):
        self.ctx = ctx
        self.n = int(ctx.block)
        self.sr = float(ctx.sr)
        self.params = dict(params)
        self.gain_prev = float(gain)
        self._grid = grid
        self._exc = exc
        self.voices = []
        self._analyse()

    def set_field(self, grid, exc):
        self._grid, self._exc = grid, exc
        self._analyse()

    def set_params(self, params):
        self.params = dict(params)
        self._analyse()

    def _spectrum(self):
        return {k: self.params[k] for k in SPECTRUM_KEYS}

    def _analyse(self):
        if self._grid is None:
            self.voices = []
            return
        _labels, voices, _color = analyse(self._grid, self.ctx.f0, 'laplacian',
                                          self._spectrum(), exc=self._exc)
        if self.ctx.pan != PAN_FIELD_MODE:
            for v in voices:
                v['pan'] = PAN_CENTER
        self.voices = voices

    def _glide(self, gain, gain_prev):
        """The baseline's one pre-clip gain ramp (render_chunk_laplacian)."""
        if gain_prev is not None:
            self.gain_prev = float(gain_prev)
        g = self.gain_prev + (gain - self.gain_prev) * _RAMP
        self.gain_prev = gain
        return g

    def display(self):
        return dict(voices=int(len(self.voices)))


class EnvBankCell(_EnvCell):
    """Env + Bank: the wave bank of laplace_carriers on the baseline analysis.

    At Sine this is the legacy `laplacian` path bit for bit (the gate
    test_laplace_carriers.test_bank_sine_is_the_baseline_byte_exact), and at
    Saw / Square it is the Wave bank method of `laplace_carriers`."""

    def __init__(self, ctx, params, grid, exc, gain):
        super().__init__(ctx, params, grid, exc, gain)
        self.bank = lc._BankVoices()
        self.wave = int(params['waveform'])
        self.wave_prev = self.wave

    def set_params(self, params):
        super().set_params(params)
        self.wave = int(params['waveform'])

    def render_float(self, gain, gain_prev, transpose, env=None):
        rel, atk, dec, sus = env
        self.bank.update(self.voices, rel, atk, dec, sus)
        L, R = self.bank.render(self.wave, self.wave_prev, self.n, self.sr, transpose)
        self.wave_prev = self.wave
        g = self._glide(gain, gain_prev)
        out = np.empty((self.n, 2))
        out[:, 0] = L * g
        out[:, 1] = R * g
        return out

    def display(self):
        d = super().display()
        d.update(voicing='bank', wave=int(self.wave),
                 wave_name=WAVE_NAMES[int(self.wave)],
                 bank_lines=int(np.count_nonzero(self.bank.pool.amp_tgt[1:] >= lc.AMP_EPS)))
        return d


class EnvFMCell(_EnvCell):
    """Env + FM: the sources of laplace_fm on the baseline analysis -- one carrier
    per figure at f0, its modes the modulators, the sum decimated from
    OVERSAMPLE x sr and DC-blocked.  Both channels are identical, as that engine
    has always rendered: its carriers are summed to mono before the pan."""

    def __init__(self, ctx, params, grid, exc, gain, oversample=None):
        self.oversample = int(lfm.OVERSAMPLE if oversample is None else oversample)
        self._mods = [None] * MAX_VOICES
        super().__init__(ctx, params, grid, exc, gain)
        self.src = lfm._FMSources(ctx.f0, ctx.sr, self.oversample)
        self.kernel = lfm._kernel(self.oversample, ctx.sr)
        self.taps = len(self.kernel)
        self.dc_r = float(math.exp(-TWO_PI * lfm.DC_HZ / float(ctx.sr)))
        self.fir_hist = np.zeros(self.taps - 1)
        self.dc_x = 0.0
        self.dc_y = 0.0
        n_os = self.n * self.oversample
        self._ramp_os = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_os))) * 0.5

    def _analyse(self):
        super()._analyse()
        # the FM engine pans its summed carriers centre, whatever the host asks
        for v in self.voices:
            v['pan'] = PAN_CENTER
        mods = [None] * MAX_VOICES
        for i, v in enumerate(self.voices[:MAX_VOICES]):
            fr = np.asarray(v['freqs'], dtype=float)
            am = np.asarray(v['amps'], dtype=float)
            live = (fr > 0.0) & (am > 0.0)
            if not live.any():
                continue
            mods[i] = (fr, am, float(math.sqrt(float((am[live] * am[live]).sum()))))
        self._mods = mods

    def render_float(self, gain, gain_prev, transpose, env=None):
        rel, atk, dec, sus = env
        self.src.update(self._mods, float(self.params['fm_depth']), rel, atk, dec, sus)
        y_os = self.src.render(self.n * self.oversample, self._ramp_os, transpose)
        y, self.fir_hist = lfm.decimate(y_os, self.kernel, self.oversample, self.fir_hist)
        y, self.dc_x, self.dc_y = lfm.dc_block(y, self.dc_r, self.dc_x, self.dc_y)
        self.mono = y
        pan = PAN_CENTER * np.pi / 2.0
        g = self._glide(gain, gain_prev)
        out = np.empty((self.n, 2))
        out[:, 0] = y * math.cos(pan) * g
        out[:, 1] = y * math.sin(pan) * g
        return out

    def display(self):
        d = super().display()
        sounding, tails, mods, mod_tails = self.src.counts()
        d.update(voicing='fm', depth=float(self.params['fm_depth']),
                 oversample=int(self.oversample), taps=int(self.taps),
                 delay_samples=int(lfm.FIR_DELAY_OUT), sounding=sounding,
                 tails=tails, modulators=mods, mod_tails=mod_tails)
        return d


# ── the router ────────────────────────────────────────────────────────────────

class _Layer:
    """One cell with its fade weight: the active one climbs to 1, a cell an axis
    switch left behind falls to 0 and is dropped when it gets there."""
    __slots__ = ('cell', 'key', 'w', 'target')

    def __init__(self, cell, key, w, target):
        self.cell = cell
        self.key = key
        self.w = float(w)
        self.target = float(target)


class UnifiedLaplaceEngine(GenEnvelopeKnobs, SoundEngine):
    """Articulation x voicing on one Laplacian spectrum (module doc)."""

    SUPPORTS_TRANSPOSE = True          # every cell renders the note as a scalar
    # the Events articulation has its own decay law and the FM sources their own
    # envelope: neither has a SlotPool slew path, so the toggle is declined for
    # all of them rather than acting on one cell and not on the next
    SUPPORTS_AMP_SLEW = False

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        for k, v in OPTIONAL_PARAMS.items():
            self.params.setdefault(k, v)
        self.engine_id = ENGINE_ID
        self._init_gen_env(ctx.rate_hz)
        self.n = int(ctx.block)
        self._grid = None
        self._exc = None
        self._layers = []
        self._tables = None                 # one wave-table pool for every Events cell
        self._fresh = True                  # the next cell is an init, not a switch
        self.gain_prev = float(0.0)
        self.switches = 0

    def _recount(self):
        """Plus this engine's rule: an axis switch crossfades for as long as a
        voice takes to let go, so it never outlives its own tails -- and never
        for less than the 20 ms every ramp of this family takes, so a switch at a
        fast tempo is still a glide and not a step."""
        super()._recount()
        self._xfade_blocks = max(XFADE_MIN_BLOCKS, self._release_chunks)

    # -- parameters ---------------------------------------------------------------
    def _p(self, name):
        return self.params.get(name, OPTIONAL_PARAMS[name])

    def _key(self):
        """Which cell the axes ask for: the waveform lives INSIDE the bank cell,
        so changing it is a blend of readouts, not a change of cell."""
        return (int(self._p('artic')), int(self._p('voice')))

    def _env(self):
        return (self._release_chunks, self._attack_chunks, self._decay_chunks, self._sustain)

    def _make(self, key, strike):
        artic, voice = key
        if artic == ARTIC_ENV:
            if voice == VOICE_FM:
                return EnvFMCell(self.ctx, self.params, self._grid, self._exc, self.gain_prev)
            return EnvBankCell(self.ctx, self.params, self._grid, self._exc, self.gain_prev)
        from . import unified_events as uev
        if voice == VOICE_FM:
            return uev.EventsFMCell(self.ctx, self.params, self._grid, self._exc,
                                    self.gain_prev, strike)
        if self._tables is None:
            self._tables = uev.WaveTables()
        return uev.EventsBankCell(self.ctx, self.params, self._grid, self._exc,
                                  self.gain_prev, strike, tables=self._tables)

    def _active(self):
        for lay in self._layers:
            if lay.target >= 1.0:
                return lay
        return None

    def _switch(self, key):
        """An axis moved.  The cell that was heard keeps rendering and falls to
        silence over one release; the new one is created SILENT and climbs.  A
        switch straight back turns that crossfade around instead of stacking a
        third cell on top of it."""
        cur = self._active()
        if cur is not None:
            cur.target = 0.0
        for lay in self._layers:
            if lay.key == key and lay is not cur:
                lay.target = 1.0                   # the crossfade turns around
                self.switches += 1
                return
        self._layers.append(_Layer(self._make(key, strike=False), key, 0.0, 1.0))
        self.switches += 1

    # -- SoundEngine ---------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._layers = []
        self._fresh = True
        self.gain_prev = float(gain)
        self.update_field(grid, exc)

    def update_field(self, grid, exc):
        self._grid, self._exc = grid, exc
        for lay in self._layers:
            lay.cell.set_field(grid, exc)

    def set_params(self, params):
        key_old = self._key()
        super().set_params(params)
        for k, v in OPTIONAL_PARAMS.items():
            self.params.setdefault(k, v)
        for lay in self._layers:
            lay.cell.set_params(self.params)
        if self._key() != key_old and self._grid is not None:
            self._switch(self._key())

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        if gain_prev is not None:
            self.gain_prev = float(gain_prev)
        if not self._layers:
            self._layers.append(_Layer(self._make(self._key(), strike=self._fresh),
                                       self._key(), 1.0, 1.0))
            self._fresh = False
        env = self._env()
        step = 1.0 / self._xfade_blocks
        acc = None
        for lay in self._layers:
            w0 = lay.w
            w1 = min(1.0, w0 + step) if lay.target >= 1.0 else max(0.0, w0 - step)
            y = lay.cell.render_float(gain, gain_prev, transpose, env)
            if w0 >= 1.0 and w1 >= 1.0:
                acc = y if acc is None else acc + y      # the single-cell path: untouched
            elif w0 <= 0.0 and w1 <= 0.0:
                pass                                     # rendered to keep its state moving
            else:
                ramp = (w0 + (w1 - w0) * _RAMP)[:, None]
                acc = y * ramp if acc is None else acc + y * ramp
            lay.w = w1
        self._layers = [l for l in self._layers if l.target >= 1.0 or l.w > 0.0]
        self.gain_prev = gain
        if acc is None:
            acc = np.zeros((self.n, 2))
        return _finish(acc, self.ctx.channels)

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

    def display(self):
        lay = self._active() or (self._layers[0] if self._layers else None)
        d = dict(unified=True, artic=int(self._p('artic')), voice=int(self._p('voice')),
                 artic_name=ARTIC_NAMES[int(self._p('artic'))],
                 voice_name=VOICE_NAMES[int(self._p('voice'))],
                 f0=float(self.ctx.f0), layers=len(self._layers),
                 switches=int(self.switches),
                 mix=float(lay.w) if lay is not None else 0.0)
        if lay is not None:
            cell = lay.cell.display()
            if cell:
                d.update(cell)
        return d


def _finish(y, channels):
    """The block tail every Laplace engine shares: the honest pre-clip meter, the
    clip, int16.  The gain is already inside -- each cell applies it once, glided,
    the way its own law does."""
    peak = float(np.abs(y).max()) if y.size else 0.0
    n_clip = int(np.count_nonzero(np.abs(y) > 1.0))
    y = np.clip(y, -1.0, 1.0)
    if channels <= 1:
        data = ((y[:, 0] + y[:, 1]) * 0.5)[:, None]
    elif channels == 2:
        data = y
    else:                                                  # pragma: no cover
        data = np.zeros((len(y), channels))
        data[:, 0] = y[:, 0]
        data[:, 1] = y[:, 1]
    return np.ascontiguousarray((data * 32767).astype(np.int16)), peak, n_clip


# ── panel metadata ────────────────────────────────────────────────────────────

def inactive(params):
    """Knobs that do not act for the current axes.  The prototype leaves them out
    of the panel entirely, so Env + Bank shows 9 rows and Events + FM 13 (REQ
    section 2)."""
    out = {}
    artic = int(params.get('artic', ARTIC_ENV))
    voice = int(params.get('voice', VOICE_BANK))
    if voice == VOICE_FM:
        out['waveform'] = f"{WAVE_NAMES[int(params.get('waveform', WF_SINE))]}  Bank only"
    else:
        out['fm_depth'] = f"{float(params.get('fm_depth', 1.0)):.2f}  FM only"
    if artic == ARTIC_ENV:
        out['events'] = f"{EVENT_NAMES[int(params.get('events', 0))]}  Events only"
        out['radius_mul'] = f"{float(params.get('radius_mul', 1.0)):.2f}  Events only"
        out['decay_s'] = f"{float(params.get('decay_s', 0.80)):.2f} s  Events only"
        out['attack_ms'] = f"{float(params.get('attack_ms', 0.0)):.0f} ms  Events only"
    if artic == ARTIC_EVENTS:
        # not a parameter of this engine: the key a PANEL reads to grey out the
        # host's GEN block, the way the registry hint does it for Objects
        out['gen_envelope'] = "Events: the engine's own"
    if float(params.get('shape', 0.0)) <= 0.0:
        out['dyn'] = 'acts only with shape > 0'
    return out


def overlay(params, rows, cols):
    a = ARTIC_NAMES[int(params.get('artic', ARTIC_ENV))]
    v = VOICE_NAMES[int(params.get('voice', VOICE_BANK))]
    if int(params.get('voice', VOICE_BANK)) == VOICE_FM:
        i = float(params.get('fm_depth', 1.0))
        v = f"FM {i:.2f}" + ("  carrier only" if i <= 0.0 else "")
    else:
        v = f"Bank {WAVE_NAMES[int(params.get('waveform', WF_SINE))]}"
    return dict(text=f"Laplace+ [{a}, {v}]")


register(EngineSpec(ENGINE_ID, LABEL, PARAMS,
                    lambda ctx, params: UnifiedLaplaceEngine(ctx, params),
                    choices=CHOICES, ranges=RANGES,
                    inactive=inactive, overlay=overlay,
                    # the Env cells run the SlotPool envelope, so the host's GEN
                    # knobs act; the Events articulation has a decay law of its own
                    # and inactive() says so.  No cell has a slew path.
                    gen_envelope=True, gen_amp_slew=False,
                    plays_notes=True))
