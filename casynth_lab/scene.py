"""Scene (demo package) formats v1 / v2: load + strict validation.

v1: one engine (engine_id/engine_params) -> loaded as A = B.
v2: `variants` {A, B} each with a full engine_id/engine_params, plus `listen`
(short instruction) and optional `initial_side` (default A).
Optional `script` (2026-09-18, Objects Decay law D2): scripted commands of the
experiment, [{"at": samples of the scene clock > 0, "kind": ..., "args": ...},
...] -- the runner queues them every time the scene starts from its beginning
(Start, the pause of a stopped scene released, Restart); absent = none (older
scenes / records unchanged).  Kinds: `pause` {on}, and since 2026-09-21 `note`
{note: 0..127}, `gate` {on}, `vol` {value: 0..1} and `set_cells`
{cells: [[row, col, 0|1], ...]} (an edit of the field the user made by hand).

ARTICULATION (2026-09-21, the seam): a scene may also describe what a host that
plays NOTES does, so a prototype session can be re-rendered offline -- immune to
a CPU dip, unlike a live run.  In `audio`:
  note      MIDI number the scene sounds at (absent = none: the engines render
            at f0_hz, transpose 1.0, exactly as every scene so far);
  gate      is the note held at the start (default true);
  pan       "center" (default: both channels identical, the bench's rule) or
            "field" (a voice sits where its figure sits, what the prototype
            plays);
  envelope  {"voice": {attack_ms, decay_ms, sustain, release_ms},
             "gen":   {attack, decay, sustain, release, amp_slew}} -- the VCA
            over the sum and the per-mode envelope on the automaton clock
            (GEN A/D/R are FRACTIONS of one tick).  Absent = the defaults of
            casynth_config, which is what the bench has always rendered.
A scene without any of this renders byte-for-byte as before.
Unknown engine / unknown or missing engine parameter / out-of-range cell ->
SceneError with a readable message.  No silent fallbacks.  The loaded JSON is
never mutated by playback (the runner copies the settings).
"""
import copy
import json
import os

import numpy as np

from . import registry

FORMAT_VERSIONS = (1, 2)
SIDES = ('A', 'B')
SUPPORTED_RULES = ('B3/S23',)
SUPPORTED_BOUNDARIES = ('torus',)
# commands a scene script may schedule (2026-09-18; note / gate / vol / set_cells 2026-09-21)
SCRIPT_KINDS = ('pause', 'note', 'gate', 'vol', 'set_cells', 'step')
STEP_SOURCES = ('clock', 'script')   # who advances the automaton (2026-09-21)
VOICE_KEYS = ('attack_ms', 'decay_ms', 'sustain', 'release_ms')
GEN_KEYS = ('attack', 'decay', 'sustain', 'release', 'amp_slew')


class SceneError(ValueError):
    pass


class Scene:
    def __init__(self, d, path=None):
        self.path = path
        self.doc = copy.deepcopy(d)      # self-contained copy (records embed it)
        self.id = d['id']
        self.title = d['title']
        self.rows = int(d['grid']['rows'])
        self.cols = int(d['grid']['cols'])
        self.cells = [(int(r), int(c)) for r, c in d['cells']]
        self.rule = d['rule']
        self.boundary = d['boundary']
        self.rate_hz = float(d['rate_hz'])
        # who advances the automaton: its own sample clock at rate_hz ('clock',
        # every scene so far), or the script ('script') -- a converted prototype
        # session, whose steps happened on a WALL clock and are replayed at the
        # output samples they were recorded at.  rate_hz still scales the
        # tick-relative envelopes either way.
        self.step_source = d.get('steps', 'clock')
        self.format = int(d['format'])
        if self.format == 1:
            one = (d['engine_id'], dict(d['engine_params']))
            self.variants = {'A': one, 'B': (one[0], dict(one[1]))}
            self.listen = ''
            self.factory_variants = {k: (v[0], dict(v[1])) for k, v in self.variants.items()}
            self.param_memory = {}
            self.initial_side = 'A'
        else:
            self.variants = {k: (d['variants'][k]['engine_id'],
                                 dict(d['variants'][k]['engine_params']))
                             for k in SIDES}
            self.listen = d.get('listen', '')
            self.initial_side = d.get('initial_side', 'A')
            fv = d.get('factory_variants') or d['variants']
            self.factory_variants = {k: (fv[k]['engine_id'], dict(fv[k]['engine_params']))
                                     for k in SIDES}
            # per side: {engine_id: params} remembered for engines not currently on it
            self.param_memory = {k: {eid: dict(pp) for eid, pp in v.items()}
                                 for k, v in (d.get('param_memory') or {}).items()}
        # per side: {engine_id: {param: (min, max)}} user slider ranges of ranged
        # parameters (2026-09-17, Objects Radius x); absent = the registry defaults
        self.param_ranges = {k: {eid: {p: (float(v[0]), float(v[1])) for p, v in rr.items()}
                                 for eid, rr in per.items()}
                             for k, per in (d.get('param_ranges') or {}).items()}
        # scripted commands (2026-09-18): (at, kind, args) on the scene clock, sorted
        self.script = sorted(((int(c['at']), c['kind'], dict(c.get('args') or {}))
                              for c in (d.get('script') or [])), key=lambda c: c[0])
        # articulation (2026-09-21): the note, the gate and the envelopes of a host
        # that plays notes; absent = nothing to articulate (see the module doc)
        _a = d['audio']
        self.note = None if _a.get('note') is None else int(_a['note'])
        self.gate = bool(_a.get('gate', True))
        self.pan = _a.get('pan', 'center')
        env = _a.get('envelope')
        self.envelope = ({k: dict(v) for k, v in env.items()} if env else None)
        self.articulated = (self.note is not None or self.envelope is not None
                            or any(k in ('note', 'gate') for _at, k, _ar in self.script))
        # v1 convenience (single engine)
        self.engine_id, self.engine_params = self.variants['A']
        self.f0_hz = float(d['audio']['f0_hz'])
        self.level = float(d['audio'].get('level', 1.0))
        # optional per-side level calibration (2026-09-17): a constant factor on the
        # side's pre-clip gain, absent = 1.0 (older scenes / records: bit-exact)
        sg = d['audio'].get('side_gain') or {}
        self.side_gain = {k: float(sg.get(k, 1.0)) for k in SIDES}

    def initial_grid(self):
        g = np.zeros((self.rows, self.cols), np.uint8)
        for r, c in self.cells:
            g[r, c] = 1
        return g


def _fail(msg):
    raise SceneError(msg)


def validate(d):
    if not isinstance(d, dict):
        _fail("scene: top-level JSON must be an object")
    fmt = d.get('format')
    if fmt not in FORMAT_VERSIONS:
        _fail(f"scene: 'format' must be one of {FORMAT_VERSIONS}, got {fmt!r}")
    keys = ['id', 'title', 'grid', 'cells', 'rule', 'boundary', 'rate_hz', 'audio']
    keys += ['engine_id', 'engine_params'] if fmt == 1 else ['variants']
    for key in keys:
        if key not in d:
            _fail(f"scene: missing key '{key}'")
    g = d['grid']
    if not (isinstance(g, dict) and 'rows' in g and 'cols' in g):
        _fail("scene: 'grid' must be {rows, cols}")
    rows, cols = g['rows'], g['cols']
    if not (isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0):
        _fail(f"scene: grid rows/cols must be positive ints, got {rows!r}x{cols!r}")
    for i, cell in enumerate(d['cells']):
        if not (isinstance(cell, (list, tuple)) and len(cell) == 2
                and all(isinstance(v, int) for v in cell)):
            _fail(f"scene: cells[{i}] must be [row, col] ints, got {cell!r}")
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            _fail(f"scene: cells[{i}]=({r},{c}) outside {rows}x{cols} grid")
    if d['rule'] not in SUPPORTED_RULES:
        _fail(f"scene: unsupported rule {d['rule']!r} (supported: {SUPPORTED_RULES})")
    if d['boundary'] not in SUPPORTED_BOUNDARIES:
        _fail(f"scene: unsupported boundary {d['boundary']!r} "
              f"(supported: {SUPPORTED_BOUNDARIES})")
    rate = d['rate_hz']
    if not (isinstance(rate, (int, float)) and rate > 0):
        _fail(f"scene: rate_hz must be > 0, got {rate!r}")
    if d.get('steps', 'clock') not in STEP_SOURCES:
        _fail(f"scene: 'steps' must be one of {STEP_SOURCES}, got {d.get('steps')!r}")
    if fmt == 1:
        _validate_engine(d.get('engine_id'), d.get('engine_params'), where='')
    else:
        var = d['variants']
        if not isinstance(var, dict):
            _fail("scene: 'variants' must be an object with keys A and B")
        extra = sorted(set(var) - set(SIDES))
        if extra:
            _fail(f"scene: unknown variants {extra} (expected exactly A and B)")
        for k in SIDES:
            if k not in var or not isinstance(var[k], dict):
                _fail(f"scene: variants.{k} missing or not an object")
            for key in ('engine_id', 'engine_params'):
                if key not in var[k]:
                    _fail(f"scene: variants.{k} missing '{key}'")
            _validate_engine(var[k]['engine_id'], var[k]['engine_params'],
                             where=f"variants.{k}.")
        if 'listen' in d and not isinstance(d['listen'], str):
            _fail("scene: 'listen' must be a string")
        fv = d.get('factory_variants')
        if fv is not None:
            if not isinstance(fv, dict) or sorted(fv) != sorted(SIDES):
                _fail("scene: 'factory_variants' must have exactly A and B")
            for k in SIDES:
                if not isinstance(fv[k], dict) or 'engine_id' not in fv[k] \
                        or 'engine_params' not in fv[k]:
                    _fail(f"scene: factory_variants.{k} needs engine_id/engine_params")
                _validate_engine(fv[k]['engine_id'], fv[k]['engine_params'],
                                 where=f"factory_variants.{k}.")
        pm = d.get('param_memory')
        if pm is not None:
            if not isinstance(pm, dict) or any(k not in SIDES for k in pm):
                _fail("scene: 'param_memory' keys must be sides A/B")
            for k, per_engine in pm.items():
                if not isinstance(per_engine, dict):
                    _fail(f"scene: param_memory.{k} must be an object")
                for eid, pp in per_engine.items():
                    _validate_engine(eid, pp, where=f"param_memory.{k}.{eid}.")
        if d.get('initial_side', 'A') not in SIDES:
            _fail(f"scene: initial_side must be A or B, got {d.get('initial_side')!r}")
    pr = d.get('param_ranges')
    if pr is not None:
        if not isinstance(pr, dict) or any(k not in SIDES for k in pr):
            _fail("scene: 'param_ranges' keys must be sides A/B")
        for k, per_engine in pr.items():
            if not isinstance(per_engine, dict):
                _fail(f"scene: param_ranges.{k} must be an object")
            for eid, rr in per_engine.items():
                if not registry.has(eid):
                    _fail(f"scene: param_ranges.{k}: unknown engine {eid!r}")
                if not isinstance(rr, dict):
                    _fail(f"scene: param_ranges.{k}.{eid} must be an object")
                for pname, pair in rr.items():
                    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                        _fail(f"scene: param_ranges.{k}.{eid}.{pname} must be [min, max]")
                    try:
                        registry.validate_range(eid, pname, pair[0], pair[1])
                    except ValueError as e:
                        _fail(f"scene: param_ranges.{k}.{eid}.{pname}: {e}")
    sc = d.get('script')
    if sc is not None:
        if not isinstance(sc, list):
            _fail("scene: 'script' must be a list of {at, kind, args}")
        for i, c in enumerate(sc):
            if not isinstance(c, dict) or sorted(c) not in (['at', 'kind'], ['args', 'at', 'kind']):
                _fail(f"scene: script[{i}] must be {{at, kind, args}}")
            at = c['at']
            if isinstance(at, bool) or not isinstance(at, int) or at <= 0:
                _fail(f"scene: script[{i}].at must be a positive int (samples), got {at!r}")
            if c['kind'] not in SCRIPT_KINDS:
                _fail(f"scene: script[{i}].kind must be one of {SCRIPT_KINDS}, got {c['kind']!r}")
            args = c.get('args') or {}
            if not isinstance(args, dict):
                _fail(f"scene: script[{i}].args must be an object")
            _validate_script_args(i, c['kind'], args, rows, cols)
    a = d['audio']
    if not (isinstance(a, dict) and isinstance(a.get('f0_hz'), (int, float))
            and a['f0_hz'] > 0):
        _fail("scene: 'audio' must be {f0_hz > 0, level?}")
    _validate_articulation(a)
    lvl = a.get('level', 1.0)
    if not (isinstance(lvl, (int, float)) and 0.0 <= lvl <= 1.0):
        _fail(f"scene: audio.level must be within [0, 1], got {lvl!r}")
    sg = a.get('side_gain')
    if sg is not None:
        if not isinstance(sg, dict) or any(k not in SIDES for k in sg):
            _fail("scene: audio.side_gain must be an object with keys A / B")
        for k, v in sg.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < v <= 16.0):
                _fail(f"scene: audio.side_gain.{k} must be a number within (0, 16], got {v!r}")


def _validate_script_args(i, kind, args, rows, cols):
    """Strict per-kind arguments of a scripted command (no silent defaults)."""
    where = f"scene: script[{i}].args"
    if kind == 'step':
        if args:
            _fail(f"{where} of a step must be empty")
    elif kind in ('pause', 'gate'):
        if sorted(args) != ['on'] or not isinstance(args['on'], bool):
            _fail(f"{where} of a {kind} must be {{on: true|false}}")
    elif kind == 'note':
        n = args.get('note')
        if sorted(args) != ['note'] or isinstance(n, bool) or not isinstance(n, int) \
                or not (0 <= n <= 127):
            _fail(f"{where} of a note must be {{note: 0..127}}, got {args!r}")
    elif kind == 'vol':
        v = args.get('value')
        if sorted(args) != ['value'] or isinstance(v, bool) \
                or not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            _fail(f"{where} of a vol must be {{value: 0..1}}, got {args!r}")
    elif kind == 'set_cells':
        cells = args.get('cells')
        if sorted(args) != ['cells'] or not isinstance(cells, list) or not cells:
            _fail(f"{where} of a set_cells must be {{cells: [[row, col, 0|1], ...]}}")
        for j, cell in enumerate(cells):
            if not (isinstance(cell, (list, tuple)) and len(cell) == 3
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in cell)):
                _fail(f"{where}.cells[{j}] must be [row, col, 0|1] ints, got {cell!r}")
            r, c, v = cell
            if not (0 <= r < rows and 0 <= c < cols):
                _fail(f"{where}.cells[{j}]=({r},{c}) outside the {rows}x{cols} field")
            if v not in (0, 1):
                _fail(f"{where}.cells[{j}] value must be 0 or 1, got {v!r}")


def _number(where, v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        _fail(f"scene: {where} must be a number, got {v!r}")
    if not (lo <= v <= hi):
        _fail(f"scene: {where}={v!r} outside [{lo}, {hi}]")


def _validate_articulation(a):
    """audio.note / audio.gate / audio.envelope (2026-09-21) -- all optional,
    all strict; absent means the bench's own rendering, unchanged."""
    n = a.get('note')
    if n is not None and (isinstance(n, bool) or not isinstance(n, int)
                          or not (0 <= n <= 127)):
        _fail(f"scene: audio.note must be a MIDI number 0..127, got {n!r}")
    if not isinstance(a.get('gate', True), bool):
        _fail(f"scene: audio.gate must be true or false, got {a.get('gate')!r}")
    if a.get('pan', 'center') not in ('center', 'field'):
        _fail(f"scene: audio.pan must be 'center' or 'field', got {a.get('pan')!r}")
    env = a.get('envelope')
    if env is None:
        return
    if not isinstance(env, dict) or sorted(env) not in (['gen'], ['voice'], ['gen', 'voice']):
        _fail("scene: audio.envelope must be {voice?: {...}, gen?: {...}}")
    v = env.get('voice')
    if v is not None:
        if not isinstance(v, dict) or sorted(v) != sorted(VOICE_KEYS):
            _fail(f"scene: audio.envelope.voice must have exactly {list(VOICE_KEYS)}")
        for k in ('attack_ms', 'decay_ms', 'release_ms'):
            _number(f"audio.envelope.voice.{k}", v[k], 0.0, 60000.0)
        _number("audio.envelope.voice.sustain", v['sustain'], 0.0, 1.0)
    g = env.get('gen')
    if g is not None:
        if not isinstance(g, dict) or sorted(g) != sorted(GEN_KEYS):
            _fail(f"scene: audio.envelope.gen must have exactly {list(GEN_KEYS)}")
        for k in ('attack', 'decay', 'release'):
            _number(f"audio.envelope.gen.{k}", g[k], 0.0, 64.0)
        _number("audio.envelope.gen.sustain", g['sustain'], 0.0, 1.0)
        if not isinstance(g['amp_slew'], bool):
            _fail(f"scene: audio.envelope.gen.amp_slew must be true or false, got {g['amp_slew']!r}")


def _validate_engine(eid, params, where):
    if not registry.has(eid):
        _fail(f"scene: unknown {where}engine_id {eid!r} (registered: {registry.ids()})")
    spec = {p[0]: p for p in registry.get(eid).params}
    if not isinstance(params, dict):
        _fail(f"scene: '{where}engine_params' must be an object")
    unknown = sorted(set(params) - set(spec))
    if unknown:
        _fail(f"scene: unknown {where}engine_params for {eid}: {unknown}")
    missing = sorted(set(spec) - set(params))
    if missing:
        _fail(f"scene: missing {where}engine_params for {eid}: {missing}")
    for name, (arg, label, lo, hi, integer, default) in spec.items():
        v = params[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            _fail(f"scene: {where}engine_params.{name} must be a number, got {v!r}")
        if v != v or v in (float('inf'), float('-inf')):
            _fail(f"scene: {where}engine_params.{name} must be finite, got {v!r}")
        if integer and int(v) != v:
            _fail(f"scene: {where}engine_params.{name} must be an integer, got {v!r}")
        if not (lo <= v <= hi):
            _fail(f"scene: {where}engine_params.{name}={v!r} outside [{lo}, {hi}]")
    try:
        registry.validate_params(eid, params)          # the engine's combination rule (2026-09-18)
    except ValueError as e:
        _fail(f"scene: {where}engine_params: {e}")


def scene_from_doc(d):
    """Validate + build a Scene from an in-memory document (records)."""
    validate(d)
    return Scene(copy.deepcopy(d))


def load_scene(path):
    if not os.path.isfile(path):
        _fail(f"scene file not found: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"scene: invalid JSON in {path}: {e}")
    validate(d)
    return Scene(d, path=path)
