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
from . import provenance as _prov
from .engine_api import EngineContext, check_block, supports_snapshot

BLOCK = int(CHUNK_S * SR)          # samples per audio block (== engine chunk)
CHANNELS = 2
SIDES = ('A', 'B')
OUTPUTS = ('A', 'B', 'monitor')
XFADE_MS = 20.0                    # A/B switch crossfade (output mixer only)
XFADE_SAMPLES = int(round(XFADE_MS / 1000.0 * SR))   # 882
COMMANDS = ('start', 'stop', 'pause', 'set_cell', 'reset', 'vol',
            'select', 'set_param', 'set_engine', 'copy_side', 'factory',
            'clear', 'set_cells',      # 2026-09-16: Clear button, pattern drop
            'copy_spectrum',           # 2026-09-17: the shared spectrum settings only
            'set_range')               # 2026-09-17: the slider range of a ranged parameter (UI, no sound)
RUNNER_STATE_VERSION = 1           # export_state() / from_state() format

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

    def display(self):
        """Read-only numbers an engine offers the UI (None when it has none);
        the display never takes part in the sound."""
        fn = getattr(self.engine, 'display', None)
        return fn() if callable(fn) else None

    @property
    def snapshot_ok(self):
        return supports_snapshot(self.engine)

    def export_state(self):
        """engine_id, params, diagnostics + the engine's own state (None when
        the engine class has no snapshot support)."""
        return dict(engine_id=self.engine_id, params=dict(self.params),
                    peak=float(self.peak), clip_blocks=int(self.clip_blocks),
                    engine_state_version=(type(self.engine).STATE_VERSION
                                          if self.snapshot_ok else None),
                    engine=(self.engine.export_state() if self.snapshot_ok else None))

    def restore(self, st, grid, exc):
        if not self.snapshot_ok:
            raise ValueError(f"engine {self.engine_id!r} has no snapshot support")
        if st.get('engine') is None:
            raise ValueError(f"side {self.name}: no engine state in the snapshot")
        self.engine.restore_state(grid, exc, st['engine'])
        self.params = dict(self.engine.params)
        self.peak = float(st.get('peak', 0.0))
        self.clip_blocks = int(st.get('clip_blocks', 0))


class Block:
    """One rendered block: raw A, raw B and the monitor (listened) mix."""
    __slots__ = ('A', 'B', 'monitor')

    def __init__(self, a, b, monitor):
        self.A, self.B, self.monitor = a, b, monitor

    def get(self, output):
        return getattr(self, output)


_SILENT = np.zeros((BLOCK, CHANNELS), np.int16)


class DemoRunner:
    def __init__(self, scene, vol=VOL_DEFAULT, provenance=None):
        self.scene = scene
        self.vol = float(vol)
        # S6: the code this executor runs on -- captured once per process at
        # first use (the imported code), never re-read at Save time
        self.provenance = provenance if provenance is not None else _prov.current(
            audio=dict(sr=SR, block=BLOCK, channels=CHANNELS))
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
        for name, per_engine in getattr(scene, 'param_memory', {}).items():
            for eid, pp in per_engine.items():
                if eid in registry.REGISTRY:
                    self._memory[(name, eid)] = dict(pp)
        # per (side, engine) slider ranges of ranged parameters (registry EngineSpec.ranges,
        # 2026-09-17): edited by the user, never a sound change; absent = the registry default
        self._ranges = {}
        for name, per_engine in getattr(scene, 'param_ranges', {}).items():
            for eid, rr in per_engine.items():
                if eid in registry.REGISTRY:
                    for pname, (lo, hi) in rr.items():
                        self._ranges.setdefault((name, eid), {})[pname] = registry.validate_range(
                            eid, pname, lo, hi)
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
        self.paused = True             # a stopped scene stands on pause
        self.step_samples = SR / sc.rate_hz
        for s in self.sides.values():
            s.restart(self.grid, self.exc, self._side_gain(s.name))
        self._xfade_from = None
        self._xfade_pos = 0

    def _gain(self):
        return MASTER_GAIN * self.vol * self.scene.level

    def _side_gain(self, name):
        """The pre-clip gain of one side: the bench gain times the scene's
        optional per-side calibration (1.0 when absent)."""
        return self._gain() * self.scene.side_gain.get(name, 1.0)

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
        if kind in ('select', 'set_param', 'set_engine', 'set_range'):
            side = args.get('side')
            if side not in SIDES:
                raise ValueError(f"unknown side {side!r} (expected A or B)")
        if kind in ('copy_side', 'copy_spectrum'):
            src, dst = args.get('src'), args.get('dst')
            if src not in SIDES or dst not in SIDES or src == dst:
                raise ValueError(f"{kind}: need two different sides, got {src!r}->{dst!r}")
        if kind == 'copy_spectrum':
            # the engines the sides WILL have when applied (queued set_engine counts)
            eids = {n: self.sides[n].engine_id for n in (args['src'], args['dst'])}
            for _seq, _at, k, a in self._pending:
                if k == 'set_engine' and a['side'] in eids:
                    eids[a['side']] = a['engine_id']
            if not registry.spectrum_keys(eids[args['src']], eids[args['dst']]):
                raise ValueError(f"copy_spectrum: {registry.label(eids[args['src']])} and "
                                 f"{registry.label(eids[args['dst']])} share no spectrum settings")
        if kind == 'set_engine':
            if args.get('engine_id') not in registry.REGISTRY:
                raise ValueError(f"unknown engine {args.get('engine_id')!r} "
                                 f"(registered: {registry.ids()})")
        elif kind == 'set_param':
            # validated against the engine the side WILL have when applied:
            # a queued set_engine for that side counts
            eid = self._engine_when_applied(args['side'])
            args['value'] = validate_param(eid, args.get('name'), args.get('value'))
        elif kind == 'set_range':
            eid = self._engine_when_applied(args['side'])
            lo, hi = registry.validate_range(eid, args.get('name'), args.get('lo'), args.get('hi'))
            args['lo'], args['hi'] = lo, hi
        elif kind == 'vol':
            v = args.get('value')
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"vol: value must be a finite number, got {v!r}")
        elif kind == 'set_cells':
            try:
                cells = [[int(r), int(c), int(bool(v))] for r, c, v in args.get('cells') or ()]
            except (TypeError, ValueError):
                raise ValueError("set_cells: cells must be [[row, col, value], ...]") from None
            rows, cols = self.grid.shape
            for r, c, _v in cells:
                if not (0 <= r < rows and 0 <= c < cols):
                    raise ValueError(f"set_cells: ({r},{c}) outside the {rows}x{cols} field")
            args['cells'] = cells                  # plain lists: the journal is JSON
        return args

    def _engine_when_applied(self, side):
        eid = self.sides[side].engine_id
        for _seq, _at, k, a in self._pending:
            if k == 'set_engine' and a['side'] == side:
                eid = a['engine_id']
        return eid

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
            self.paused = False
        elif kind == 'pause':
            on = bool(args.get('on', not self.paused))
            if not on and not self.running:
                # releasing the pause of a stopped scene STARTS it (transport
                # model: Stop = reset + pause + silence; un-pause = go)
                self.running = True
            self.paused = on
        elif kind == 'set_cell':
            r, c, v = int(args['r']), int(args['c']), int(bool(args['v']))
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                return
            if self.grid[r, c] != v:
                prev = self.grid.copy()
                self.grid[r, c] = v
                self.exc = events_field(prev, self.grid)
                self._field_changed()
        elif kind == 'set_cells':
            # one edit for a whole dropped pattern: one events field, one
            # engine update, one journal entry (validated in _check)
            prev = self.grid.copy()
            for r, c, v in args['cells']:
                self.grid[r, c] = v
            if not np.array_equal(prev, self.grid):
                self.exc = events_field(prev, self.grid)
                self._field_changed()
        elif kind == 'clear':
            # Clear button: an empty field; transport / engines / params stay
            if self.grid.any():
                prev = self.grid.copy()
                self.grid[:] = 0
                self.exc = events_field(prev, self.grid)
                self._field_changed()
        elif kind == 'reset':
            self._init_scene_state()
            self.running = True
            self.paused = False
        elif kind == 'stop':
            # Full stop: back to the initial scene, silent, on pause --
            # releasing the pause begins again from the beginning.
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
        elif kind == 'copy_spectrum':
            # only the spectrum settings both engines offer; engines and every
            # other setting of the destination stay (no engine restart)
            src, dst = self.sides[args['src']], self.sides[args['dst']]
            keys = registry.spectrum_keys(src.engine_id, dst.engine_id)
            if keys:
                params = dict(dst.params)
                for k in keys:
                    params[k] = validate_param(dst.engine_id, k, src.params[k])
                if params != dst.params:
                    dst.set_params(params)
                    self._memory[(dst.name, dst.engine_id)] = dict(dst.params)
        elif kind == 'factory':
            for name in SIDES:
                eid, params = self.scene.factory_variants[name]
                self._set_side(self.sides[name], eid, params)
        elif kind == 'set_range':
            # the slider range only: the parameter value is untouched (the bench clamps
            # it with a set_param of its own when the new range excludes it)
            s = self.sides[args['side']]
            lo, hi = registry.validate_range(s.engine_id, args['name'], args['lo'], args['hi'])
            self._ranges.setdefault((s.name, s.engine_id), {})[args['name']] = (lo, hi)

    def _set_side(self, s, eid, params):
        """Put engine+params on a side (engine change / copy / factory).
        Same engine -> the usual param update path; different engine -> a new
        engine instance on the current shared field (only THIS side restarts)
        + a monitor fade-in when it is the listened side."""
        if eid == s.engine_id:
            s.set_params(params)
        else:
            s.switch(eid, params, self.grid, self.exc, self._side_gain(s.name))
            if s.name == self.selected:
                self._begin_xfade(None)
        self._memory[(s.name, eid)] = dict(s.params)

    def _begin_xfade(self, from_side):
        """Start a mixer crossfade from `from_side` (None = from silence).
        Nothing to fade from while the scene is not running (the mixer is
        idle): a switch made before Start / after Stop starts clean -- and a
        fresh-conditions replay (which has no mixer memory) stays exact."""
        if not self.running:
            self._xfade_from = None
            self._xfade_pos = 0
            return
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
        raw = {name: self.sides[name].render(self._side_gain(name), self.t_samples)
               for name in SIDES}
        mon = self._mix_monitor(raw)
        self.out_samples += BLOCK
        self.t_samples += BLOCK
        if not self.paused:
            self.ca_samples += BLOCK
        return Block(raw['A'], raw['B'], mon)

    # -- snapshot (S5) ------------------------------------------------------------
    def snapshot_support(self):
        """(ok, reason): False when a side's engine cannot be exported."""
        bad = [f"{n}: {s.engine_id}" for n, s in self.sides.items() if not s.snapshot_ok]
        if bad:
            return False, "engine without snapshot support: " + ", ".join(bad)
        return True, ""

    def export_state(self):
        """Complete, explicit state at the current block boundary: everything
        the next next_block() reads.  Arrays are copied (the live runner may go
        on).  Not included: the journal (a continuation starts a new one) and
        nothing outside the runner (device queues, UI).  Raises ValueError when
        an engine has no snapshot support (see snapshot_support())."""
        ok, why = self.snapshot_support()
        if not ok:
            raise ValueError(why)
        return dict(
            version=RUNNER_STATE_VERSION,
            scene=self.scene.doc,
            ctx=dict(sr=SR, block=BLOCK, channels=CHANNELS, f0=self.ctx.f0,
                     level=self.ctx.level, rate_hz=self.ctx.rate_hz),
            grid=self.grid.copy(),
            exc=(None if self.exc is None else np.asarray(self.exc, np.float64).copy()),
            gen=int(self.gen), out_samples=int(self.out_samples),
            t_samples=int(self.t_samples), ca_samples=int(self.ca_samples),
            running=bool(self.running), paused=bool(self.paused),
            vol=float(self.vol), selected=self.selected,
            xfade_from=self._xfade_from, xfade_pos=int(self._xfade_pos),
            seq=int(self._seq),
            pending=[dict(seq=int(q), at=(None if at is None else int(at)), kind=k,
                          args=dict(a)) for (q, at, k, a) in self._pending],
            memory={n: {eid: dict(pp) for (nn, eid), pp in self._memory.items() if nn == n}
                    for n in SIDES},
            ranges=self.param_ranges(),
            sides={n: s.export_state() for n, s in self.sides.items()},
        )

    @classmethod
    def from_state(cls, state, scene=None, provenance=None):
        """A runner whose next next_block() equals the exported runner's.
        The state is validated completely BEFORE anything is built; any
        problem raises ValueError (nothing half-restored is returned)."""
        from .scene import scene_from_doc, SceneError
        if not isinstance(state, dict):
            raise ValueError("snapshot: not a mapping")
        if state.get('version') != RUNNER_STATE_VERSION:
            raise ValueError(f"snapshot: runner state version {state.get('version')!r} "
                             f"!= {RUNNER_STATE_VERSION}")
        for key in ('scene', 'ctx', 'grid', 'gen', 'out_samples', 't_samples', 'ca_samples',
                    'running', 'paused', 'vol', 'selected', 'seq', 'pending', 'memory', 'sides'):
            if key not in state:
                raise ValueError(f"snapshot: missing '{key}'")
        if scene is None:
            try:
                scene = scene_from_doc(state['scene'])
            except SceneError as e:
                raise ValueError(f"snapshot: scene: {e}")
        ctx = state['ctx']
        if not isinstance(ctx, dict):
            raise ValueError("snapshot: ctx must be a mapping")
        want = dict(sr=SR, block=BLOCK, channels=CHANNELS)
        got = {k: ctx.get(k) for k in want}
        if got != want:
            raise ValueError(f"snapshot: render context {got} != {want}")
        if (ctx.get('f0'), ctx.get('level'), ctx.get('rate_hz')) != \
                (scene.f0_hz, scene.level, scene.rate_hz):
            raise ValueError("snapshot: audio context does not match the scene")
        grid = state['grid']
        if not isinstance(grid, np.ndarray) or grid.shape != (scene.rows, scene.cols):
            raise ValueError(f"snapshot: grid shape {getattr(grid, 'shape', None)} != "
                             f"{(scene.rows, scene.cols)}")
        exc = state.get('exc')
        if exc is not None and (not isinstance(exc, np.ndarray) or exc.shape != grid.shape):
            raise ValueError("snapshot: exc shape does not match the grid")
        if state['selected'] not in SIDES:
            raise ValueError(f"snapshot: selected side {state['selected']!r}")
        if state.get('xfade_from') not in (None, 'silence', 'A', 'B'):
            raise ValueError(f"snapshot: xfade_from {state.get('xfade_from')!r}")
        sides = state['sides']
        if not isinstance(sides, dict) or sorted(sides) != sorted(SIDES):
            raise ValueError("snapshot: sides must be A and B")
        for n in SIDES:
            eid = sides[n].get('engine_id')
            if eid not in registry.REGISTRY:
                raise ValueError(f"snapshot: side {n}: engine {eid!r} is not registered")
            spec = registry.get(eid)
            params = sides[n].get('params')
            if not isinstance(params, dict) or sorted(params) != sorted(spec.defaults()):
                raise ValueError(f"snapshot: side {n}: parameters of {eid} do not match "
                                 f"the registry")
            for k, v in params.items():
                registry.validate_param(eid, k, v)
        if not isinstance(state['memory'], dict):
            raise ValueError("snapshot: memory must be a mapping")
        for n, per_engine in state['memory'].items():
            if n not in SIDES or not isinstance(per_engine, dict):
                raise ValueError("snapshot: memory keys must be sides A/B")
        ranges = state.get('ranges') or {}            # older snapshots: the registry defaults
        if not isinstance(ranges, dict) or any(n not in SIDES or not isinstance(v, dict)
                                               for n, v in ranges.items()):
            raise ValueError("snapshot: ranges must be a mapping of sides A/B")
        for n, per_engine in ranges.items():
            for eid, rr in per_engine.items():
                if eid not in registry.REGISTRY or not isinstance(rr, dict):
                    raise ValueError(f"snapshot: ranges of side {n}: engine {eid!r} / not a mapping")
                for pname, pair in rr.items():
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError(f"snapshot: range of {eid}.{pname} must be [min, max]")
                    registry.validate_range(eid, pname, pair[0], pair[1])
        # build: a plain runner on the scene, then overwrite everything
        r = cls(scene, vol=state['vol'], provenance=provenance)
        r.grid = np.ascontiguousarray(grid, dtype=np.uint8).copy()
        r.exc = None if exc is None else np.ascontiguousarray(exc, np.float64).copy()
        r.gen = int(state['gen'])
        r.out_samples = int(state['out_samples'])
        r.t_samples = int(state['t_samples'])
        r.ca_samples = int(state['ca_samples'])
        r.running = bool(state['running'])
        r.paused = bool(state['paused'])
        r.vol = float(state['vol'])
        r.selected = state['selected']
        r._xfade_from = state.get('xfade_from')
        r._xfade_pos = int(state.get('xfade_pos', 0))
        r._seq = int(state['seq'])
        r._memory = {}
        for n, per_engine in state['memory'].items():
            for eid, pp in per_engine.items():
                if eid in registry.REGISTRY:
                    r._memory[(n, eid)] = dict(pp)
        r._ranges = {}
        for n, per_engine in ranges.items():
            for eid, rr in per_engine.items():
                for pname, pair in rr.items():
                    r._ranges.setdefault((n, eid), {})[pname] = (float(pair[0]), float(pair[1]))
        r.journal = []
        r._pending = []
        for n in SIDES:
            st = sides[n]
            side = SideState(n, st['engine_id'], st['params'], r.ctx)
            side.restore(st, r.grid, r.exc)
            r.sides[n] = side
        # queued-but-not-yet-applied commands keep their time and order
        for c in state['pending']:
            kind, args = c.get('kind'), dict(c.get('args') or {})
            if kind not in COMMANDS:
                raise ValueError(f"snapshot: pending command {kind!r}")
            args = r._check(kind, args)
            r._pending.append((int(c['seq']), c.get('at'), kind, args))
        return r

    # -- introspection ----------------------------------------------------------
    def param_memory(self):
        """{side: {engine_id: params}} -- remembered settings of every engine
        each side has used (part of the reproducible conditions)."""
        out = {n: {} for n in SIDES}
        for (name, eid), pp in self._memory.items():
            out[name][eid] = dict(pp)
        return out

    def param_ranges(self):
        """{side: {engine_id: {param: [min, max]}}} -- the user-edited slider ranges
        (only what was edited / loaded; absent = the registry default)."""
        out = {n: {} for n in SIDES}
        for (name, eid), rr in self._ranges.items():
            if rr:
                out[name][eid] = {k: [float(v[0]), float(v[1])] for k, v in rr.items()}
        return {n: v for n, v in out.items() if v}

    def range_of(self, side, name):
        """(min, max) of the slider of a ranged parameter on the side's current
        engine (the edited range, else the registry default); None when the
        parameter has no user range."""
        s = self.sides[side]
        rr = self._ranges.get((s.name, s.engine_id), {})
        if name in rr:
            return tuple(rr[name])
        return registry.default_range(s.engine_id, name)

    def side_ranges(self):
        return {n: {k: self.range_of(n, k) for k in registry.get(s.engine_id).ranges}
                for n, s in self.sides.items()}

    def side_settings(self):
        return {n: (s.engine_id, dict(s.params)) for n, s in self.sides.items()}

    def side_modified(self):
        """True per side when engine/params differ from the scene's defaults."""
        return {n: (s.engine_id, s.params) != (self.scene.factory_variants[n][0],
                                                 self.scene.factory_variants[n][1])
                for n, s in self.sides.items()}

    def snapshot(self):
        return dict(grid=self.grid.copy(), gen=self.gen, running=self.running,
                    paused=self.paused, t_seconds=self.t_samples / SR,
                    vol=self.vol, out_samples=self.out_samples,
                    selected=self.selected,
                    sides=self.side_settings(), modified=self.side_modified(),
                    ranges=self.side_ranges(),
                    peak={n: s.peak for n, s in self.sides.items()},
                    clip_blocks={n: s.clip_blocks for n, s in self.sides.items()},
                    display={n: s.display() for n, s in self.sides.items()})


def describe_difference(settings):
    """Short human summary of how A and B differ (no JSON)."""
    (ea, pa), (eb, pb) = settings['A'], settings['B']
    la, lb = registry.label(ea), registry.label(eb)
    if ea != eb:
        return f"A: {la}  /  B: {lb}"
    diffs = [f"{k}: A={registry.value_text(ea, k, pa[k])}, B={registry.value_text(ea, k, pb[k])}"
             for k in pa if pa[k] != pb.get(k)]
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
