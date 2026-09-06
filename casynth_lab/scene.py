"""Scene (demo package) format v1: load + strict validation.

Unknown engine / unknown or missing engine parameter / out-of-range cell ->
SceneError with a readable message.  No silent fallbacks.
"""
import json
import os

import numpy as np

from casynth_core import ENGINE_BY_ID

FORMAT_VERSION = 1
SUPPORTED_RULES = ('B3/S23',)
SUPPORTED_BOUNDARIES = ('torus',)


class SceneError(ValueError):
    pass


class Scene:
    def __init__(self, d, path=None):
        self.path = path
        self.id = d['id']
        self.title = d['title']
        self.rows = int(d['grid']['rows'])
        self.cols = int(d['grid']['cols'])
        self.cells = [(int(r), int(c)) for r, c in d['cells']]
        self.rule = d['rule']
        self.boundary = d['boundary']
        self.rate_hz = float(d['rate_hz'])
        self.engine_id = d['engine_id']
        self.engine_params = dict(d['engine_params'])
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
    if d.get('format') != FORMAT_VERSION:
        _fail(f"scene: 'format' must be {FORMAT_VERSION}, got {d.get('format')!r}")
    for key in ('id', 'title', 'grid', 'cells', 'rule', 'boundary', 'rate_hz',
                'engine_id', 'engine_params', 'audio'):
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
    eid = d['engine_id']
    if eid not in ENGINE_BY_ID:
        _fail(f"scene: unknown engine_id {eid!r} (known: {sorted(ENGINE_BY_ID)})")
    spec = {p[0]: p for p in ENGINE_BY_ID[eid]['params']}
    params = d['engine_params']
    if not isinstance(params, dict):
        _fail("scene: 'engine_params' must be an object")
    unknown = sorted(set(params) - set(spec))
    if unknown:
        _fail(f"scene: unknown engine_params for {eid}: {unknown}")
    missing = sorted(set(spec) - set(params))
    if missing:
        _fail(f"scene: missing engine_params for {eid}: {missing}")
    for name, (arg, label, lo, hi, integer, default) in spec.items():
        v = params[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            _fail(f"scene: engine_params.{name} must be a number, got {v!r}")
        if integer and int(v) != v:
            _fail(f"scene: engine_params.{name} must be an integer, got {v!r}")
        if not (lo <= v <= hi):
            _fail(f"scene: engine_params.{name}={v!r} outside [{lo}, {hi}]")
    a = d['audio']
    if not (isinstance(a, dict) and isinstance(a.get('f0_hz'), (int, float))
            and a['f0_hz'] > 0):
        _fail("scene: 'audio' must be {f0_hz > 0, level?}")
    lvl = a.get('level', 1.0)
    if not (isinstance(lvl, (int, float)) and 0.0 <= lvl <= 1.0):
        _fail(f"scene: audio.level must be within [0, 1], got {lvl!r}")


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
