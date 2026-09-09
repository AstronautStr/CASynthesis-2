"""Scene (demo package) formats v1 / v2: load + strict validation.

v1: one engine (engine_id/engine_params) -> loaded as A = B.
v2: `variants` {A, B} each with a full engine_id/engine_params, plus `listen`
(short instruction) and optional `initial_side` (default A).
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
        # v1 convenience (single engine)
        self.engine_id, self.engine_params = self.variants['A']
        self.f0_hz = float(d['audio']['f0_hz'])
        self.level = float(d['audio'].get('level', 1.0))

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
    a = d['audio']
    if not (isinstance(a, dict) and isinstance(a.get('f0_hz'), (int, float))
            and a['f0_hz'] > 0):
        _fail("scene: 'audio' must be {f0_hz > 0, level?}")
    lvl = a.get('level', 1.0)
    if not (isinstance(lvl, (int, float)) and 0.0 <= lvl <= 1.0):
        _fail(f"scene: audio.level must be within [0, 1], got {lvl!r}")


def _validate_engine(eid, params, where):
    if eid not in registry.REGISTRY:
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
