"""DemoRunner -- the single live/offline executor (S1 + S2 A/B + S3 engines).

Owns the BENCH state: the shared field, excitation, clocks, command queue and
journal, and two SIDES (A / B) each holding its own engine INSTANCE created
from the bench registry (casynth_lab.registry).  Commands arrive through
post() (a queue) tagged with the output sample index at which they may be
applied; each is applied at the first block boundary >= that time, and its
actual time + order is recorded in `journal`.

Clocks (all in samples, SR from casynth_config):
  out_samples : transport clock -- every block ever produced by next_block(),
                monotone, never reset (command timestamps refer to it).
  t_samples   : scene audio clock -- advances while the scene is running;
                reset by 'reset' / 'stop'.
  ca_samples  : automaton clock -- advances while running AND not paused;
                frozen by 'pause', so un-pausing never replays missed steps.
Tempo is derived from the number of rendered samples, never from wall time.

Both sides are rendered every block on the same timeline (one engine.render
per side); the MONITOR is the listened side with a 20 ms linear crossfade
(coefficients sum to 1) applied only in the output mixer -- the raw side PCM
is untouched.  Every engine block is validated (engine_api.check_block)
before it can reach the output.
"""
import math
import wave

import numpy as np

from casynth_config import SR, CHUNK_S, MASTER_GAIN, VOL_DEFAULT
from casynth_engine import step, events_field
from . import registry
from .engine_api import EngineContext, check_block

BLOCK = int(CHUNK_S * SR)          # samples per audio block (== engine chunk)
CHANNELS = 2
SIDES = ('A', 'B')
OUTPUTS = ('A', 'B', 'monitor')
XFADE_MS = 20.0                    # A/B switch crossfade (output mixer only)
XFADE_SAMPLES = int(round(XFADE_MS / 1000.0 * SR))   # 882
COMMANDS = ('start', 'stop', 'pause', 'set_cell', 'reset', 'vol',
            'select', 'set_param', 'set_engine', 'copy_side', 'factory')

# re-exported for callers that only import the runner
validate_param = registry.validate_param
engine_defaults = registry.defaults


class SideState:
    """One A/B side: the engine id + applied params + the engine instance."""

    def __init__(self, name, engine_id, params, ctx):
        self.name = name
        self.ctx = ctx
        self.clip_blocks = 0
        self.peak = 0.0
        self.engine_id = engine_id
        self.params = dict(params)
        self.engine = registry.create(engine_id, ctx, self.params)

    def restart(self, grid, exc, gain):
        """(Re)initialise the engine on the current field: all audio memory gone."""
        self.engine.init(grid, exc, gain)

    def switch(self, engine_id, params, grid, exc, gain):
        """Replace the engine instance (tails of the old one are not mixed in)."""
        self.engine_id = engine_id
        self.params = dict(params)
        self.engine = registry.create(engine_id, self.ctx, self.params)
        self.engine.init(grid, exc, gain)

    def set_params(self, params):
        self.params = dict(params)
        self.engine.set_params(self.params)

    def update_field(self, grid, exc):
        self.engine.update_field(grid, exc)

    def render(self, gain, t_samples):
        buf, peak, n_clip = self.engine.render(gain, t_samples)
        check_block(buf, self.ctx, self.engine_id)
        self.peak = float(peak)
        if n_clip > 0:
            self.clip_blocks += 1
        return buf


class Block:
    """One rendered block: raw A, raw B and the monitor (listened) mix."""
    __slots__ = ('A', 'B', 'monitor')

    def __init__(self, a, b, monitor):
        self.A, self.B, self.monitor = a, b, monitor

    def get(self, output):
        return getattr(self, output)


_SILENT = np.zeros((BLOCK, CHANNELS), np.int16)


class DemoRunner:
    def __init__(self, scene, vol=VOL_DEFAULT):
        self.scene = scene
        self.vol = float(vol)
        self.out_samples = 0
        self.journal = []              # (out_sample, seq, kind, args)
        self._pending = []             # [(seq, at, kind, args)]
        self._seq = 0
        self.selected = scene.initial_side
        self.ctx = EngineContext(SR, BLOCK, CHANNELS, scene.f0_hz, scene.level,
                                 scene.rate_hz)
        # per (side, engine) parameter memory: first pick = defaults / scene,
        # returning to an engine restores the previous values
        self._memory = {}
        self.sides = {}
        for name in SIDES:
            eid, params = scene.variants[name]
            if eid not in registry.REGISTRY:
                raise ValueError(f"scene variant {name}: unknown engine {eid!r}")
            self.sides[name] = SideState(name, eid, params, self.ctx)
            self._memory[(name, eid)] = dict(params)
        # crossfade state (mixer only): from-side and samples done
        self._xfade_from = None
        self._xfade_pos = 0
        self._init_scene_state()

    # -- state ----------------------------------------------------------------
    def _init_scene_state(self):
        """Field + clocks + audio memory of BOTH sides back to the start.
        Engines, params, volume and the selected side are preserved."""
        sc = self.scene
        self.grid = sc.initial_grid()
        self.exc = None
        self.gen = 0
        self.t_samples = 0
        self.ca_samples = 0
        self.running = False
        self.paused = False
        self.step_samples = SR / sc.rate_hz
        for s in self.sides.values():
            s.restart(self.grid, self.exc, self._gain())
        self._xfade_from = None
        self._xfade_pos = 0

    def _gain(self):
        return MASTER_GAIN * self.vol * self.scene.level

    def _field_changed(self):
        for s in self.sides.values():
            s.update_field(self.grid, self.exc)

    # -- commands -------------------------------------------------------------
    def post(self, kind, at=None, **args):
        """Queue a command.  `at` = output-sample index not before which it may
        apply (None = next block boundary).  Addressed commands are validated
        HERE (before any state change); a bad one raises ValueError and leaves
        the runner untouched.  Returns the command's sequence no."""
        if kind not in COMMANDS:
            raise ValueError(f"unknown command {kind!r}")
        args = self._check(kind, dict(args))
        self._seq += 1
        self._pending.append((self._seq, at, kind, args))
        return self._seq

    def _check(self, kind, args):
        if kind in ('select', 'set_param', 'set_engine'):
            side = args.get('side')
            if side not in SIDES:
                raise ValueError(f"unknown side {side!r} (expected A or B)")
        if kind == 'copy_side':
            src, dst = args.get('src'), args.get('dst')
            if src not in SIDES or dst not in SIDES or src == dst:
                raise ValueError(f"copy_side: need two different sides, got {src!r}->{dst!r}")
        if kind == 'set_engine':
            if args.get('engine_id') not in registry.REGISTRY:
                raise ValueError(f"unknown engine {args.get('engine_id')!r} "
                                 f"(registered: {registry.ids()})")
        elif kind == 'set_param':
            # validated against the engine the side WILL have when applied:
            # a queued set_engine for that side counts
            eid = self.sides[args['side']].engine_id
            for _seq, _at, k, a in self._pending:
                if k == 'set_engine' and a['side'] == args['side']:
                    eid = a['engine_id']
            args['value'] = validate_param(eid, args.get('name'), args.get('value'))
        elif kind == 'vol':
            v = args.get('value')
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"vol: value must be a finite number, got {v!r}")
        return args

    def _apply_due(self):
        due = [c for c in self._pending
               if c[1] is None or c[1] <= self.out_samples]
        if not due:
            return
        due.sort(key=lambda c: ((-1 if c[1] is None else c[1]), c[0]))
        for c in due:
            self._pending.remove(c)
        for seq, _at, kind, args in due:
            self._apply(kind, args)
            self.journal.append((self.out_samples, seq, kind, dict(args)))
            if kind in ('reset', 'stop'):
                # Reset/stop discard everything still queued for the future
                # (those events belong to the previous run).
                self._pending.clear()

    def _apply(self, kind, args):
        if kind == 'start':
            self.running = True
        elif kind == 'pause':
            self.paused = bool(args.get('on', not self.paused))
        elif kind == 'set_cell':
            r, c, v = int(args['r']), int(args['c']), int(bool(args['v']))
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                return
            if self.grid[r, c] != v:
                prev = self.grid.copy()
                self.grid[r, c] = v
                self.exc = events_field(prev, self.grid)
                self._field_changed()
        elif kind == 'reset':
            self._init_scene_state()
            self.running = True
        elif kind == 'stop':
            # Full stop (player semantics): back to the initial scene, silent,
            # not running -- Start begins again from the beginning.
            self._init_scene_state()
        elif kind == 'vol':
            self.vol = float(min(max(args['value'], 0.0), 1.0))
        elif kind == 'select':
            side = args['side']
            if side != self.selected:
                self._begin_xfade(self.selected)
                self.selected = side
        elif kind == 'set_param':
            s = self.sides[args['side']]
            v = validate_param(s.engine_id, args['name'], args['value'])
            params = dict(s.params)
            params[args['name']] = v
            s.set_params(params)
            self._memory[(s.name, s.engine_id)] = dict(s.params)
        elif kind == 'set_engine':
            s = self.sides[args['side']]
            eid = args['engine_id']
            if eid == s.engine_id:
                return
            params = self._memory.get((s.name, eid)) or engine_defaults(eid)
            self._set_side(s, eid, params)
        elif kind == 'copy_side':
            src = self.sides[args['src']]
            self._set_side(self.sides[args['dst']], src.engine_id, src.params)
        elif kind == 'factory':
            for name in SIDES:
                eid, params = self.scene.variants[name]
                self._set_side(self.sides[name], eid, params)

    def _set_side(self, s, eid, params):
        """Put engine+params on a side (engine change / copy / factory).
        Same engine -> the usual param update path; different engine -> a new
        engine instance on the current shared field (only THIS side restarts)
        + a monitor fade-in when it is the listened side."""
        if eid == s.engine_id:
            s.set_params(params)
        else:
            s.switch(eid, params, self.grid, self.exc, self._gain())
            if s.name == self.selected:
                self._begin_xfade(None)
        self._memory[(s.name, eid)] = dict(s.params)

    def _begin_xfade(self, from_side):
        """Start a mixer crossfade from `from_side` (None = from silence)."""
        self._xfade_from = from_side if from_side is not None else 'silence'
        self._xfade_pos = 0

    # -- audio ----------------------------------------------------------------
    def _step(self):
        prev = self.grid
        self.grid = step(prev)
        self.exc = events_field(prev, self.grid)
        self.gen += 1
        self._field_changed()
        self.journal.append((self.out_samples, 0, 'step', {'gen': self.gen}))

    def _mix_monitor(self, raw):
        cur = raw[self.selected]
        if self._xfade_from is None:
            return cur
        old = _SILENT if self._xfade_from == 'silence' else raw[self._xfade_from]
        pos = self._xfade_pos
        x = (np.arange(BLOCK) + pos + 1) / float(XFADE_SAMPLES)
        x = np.minimum(x, 1.0)[:, None]
        out = (old.astype(np.float64) * (1.0 - x) + cur.astype(np.float64) * x)
        self._xfade_pos = pos + BLOCK
        if self._xfade_pos >= XFADE_SAMPLES:
            self._xfade_from = None
        return np.rint(out).astype(np.int16)

    def next_block(self):
        """Produce the next Block (raw A, raw B, monitor), each int16
        (BLOCK x 2).  The ONLY audio entry point for both the live thread and
        the offline renderer; every side is rendered exactly once per block."""
        self._apply_due()
        if not self.running:
            self.out_samples += BLOCK
            z = _SILENT.copy()
            return Block(z, z.copy(), z.copy())
        if not self.paused and self.ca_samples >= (self.gen + 1) * self.step_samples:
            self._step()
        gain = self._gain()
        raw = {name: self.sides[name].render(gain, self.t_samples) for name in SIDES}
        mon = self._mix_monitor(raw)
        self.out_samples += BLOCK
        self.t_samples += BLOCK
        if not self.paused:
            self.ca_samples += BLOCK
        return Block(raw['A'], raw['B'], mon)

    # -- introspection ----------------------------------------------------------
    def side_settings(self):
        return {n: (s.engine_id, dict(s.params)) for n, s in self.sides.items()}

    def side_modified(self):
        """True per side when engine/params differ from the scene's defaults."""
        return {n: (s.engine_id, s.params) != (self.scene.variants[n][0],
                                                 self.scene.variants[n][1])
                for n, s in self.sides.items()}

    def snapshot(self):
        return dict(grid=self.grid.copy(), gen=self.gen, running=self.running,
                    paused=self.paused, t_seconds=self.t_samples / SR,
                    vol=self.vol, out_samples=self.out_samples,
                    selected=self.selected,
                    sides=self.side_settings(), modified=self.side_modified(),
                    peak={n: s.peak for n, s in self.sides.items()},
                    clip_blocks={n: s.clip_blocks for n, s in self.sides.items()})


def describe_difference(settings):
    """Short human summary of how A and B differ (no JSON)."""
    (ea, pa), (eb, pb) = settings['A'], settings['B']
    la, lb = registry.label(ea), registry.label(eb)
    if ea != eb:
        return f"A: {la}  /  B: {lb}"
    diffs = [f"{k}: A={pa[k]:g}, B={pb[k]:g}" for k in pa if pa[k] != pb.get(k)]
    return "   ".join(diffs) if diffs else f"A = B ({la})"


def render_offline(scene, seconds, commands=None, vol=VOL_DEFAULT, runner=None,
                   output='A'):
    """Render `seconds` of audio through DemoRunner, trimmed to the exact
    sample count, no normalisation.  `commands` = [(kind, at_samples, {args})];
    default = start at 0.  `output` = 'A' | 'B' | 'monitor' (or a tuple of
    those -> dict).  Returns (pcm or {output: pcm}, runner)."""
    n = int(round(seconds * SR))
    if runner is None:
        runner = DemoRunner(scene, vol=vol)
    if commands is None:
        commands = [('start', 0, {})]
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    outputs = (output,) if isinstance(output, str) else tuple(output)
    for o in outputs:
        if o not in OUTPUTS:
            raise ValueError(f"unknown output {o!r} (expected A, B or monitor)")
    blocks = {o: [] for o in outputs}
    got = 0
    while got < n:
        b = runner.next_block()
        for o in outputs:
            blocks[o].append(b.get(o))
        got += BLOCK
    result = {o: (np.concatenate(blocks[o], axis=0)[:n] if blocks[o]
                  else np.zeros((0, CHANNELS), np.int16)) for o in outputs}
    if isinstance(output, str):
        return result[output], runner
    return result, runner


def write_wav(path, pcm):
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
