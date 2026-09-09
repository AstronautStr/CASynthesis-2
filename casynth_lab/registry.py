"""Bench engine registry: id -> (label, parameter specs, factory).

The UI, scene validation, commands and offline render all read from here.
The five existing methods are registered from casynth_core.ENGINES (their
metadata is NOT duplicated) through LegacySynthEngine.  A new engine module
registers itself with register() -- explicitly, in one place (see
demos/README.md); no directory scanning or hot loading.

Param spec tuple (same as casynth_core): (arg, label, lo, hi, integer, default).
"""
import math

from casynth_core import ENGINES as _CORE_ENGINES
from .engine_api import EngineContext, SoundEngine   # noqa: F401  (re-export for engine modules)
from .legacy_engine import LegacySynthEngine


class EngineSpec:
    __slots__ = ('id', 'label', 'params', 'factory')

    def __init__(self, id, label, params, factory):
        self.id = id
        self.label = label
        self.params = tuple(tuple(p) for p in params)
        self.factory = factory       # factory(ctx, params) -> SoundEngine

    def defaults(self):
        return {p[0]: p[5] for p in self.params}

    def spec_of(self, name):
        for p in self.params:
            if p[0] == name:
                return p
        return None


REGISTRY = {}


def register(spec, replace=False):
    """Add an EngineSpec.  Re-registering an existing id needs replace=True."""
    if not isinstance(spec, EngineSpec):
        raise TypeError("register() expects an EngineSpec")
    if spec.id in REGISTRY and not replace:
        raise ValueError(f"engine id {spec.id!r} already registered")
    for p in spec.params:
        if len(p) != 6:
            raise ValueError(f"engine {spec.id!r}: bad param spec {p!r}")
    REGISTRY[spec.id] = spec
    return spec


def unregister(engine_id):
    REGISTRY.pop(engine_id, None)


def get(engine_id):
    if engine_id not in REGISTRY:
        raise KeyError(engine_id)
    return REGISTRY[engine_id]


def ids():
    return list(REGISTRY)


def specs():
    return list(REGISTRY.values())


def label(engine_id):
    return get(engine_id).label


def defaults(engine_id):
    return get(engine_id).defaults()


def create(engine_id, ctx, params):
    return get(engine_id).factory(ctx, params)


def validate_param(engine_id, name, value):
    """Registry-driven check; returns the coerced value or raises ValueError."""
    if engine_id not in REGISTRY:
        raise ValueError(f"unknown engine {engine_id!r}")
    spec = get(engine_id).spec_of(name)
    if spec is None:
        raise ValueError(f"unknown parameter {name!r} for engine {engine_id}")
    _arg, _label, lo, hi, integer, _default = spec
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{engine_id}.{name}: value must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{engine_id}.{name}: value must be finite, got {value!r}")
    if integer:
        if int(value) != value:
            raise ValueError(f"{engine_id}.{name}: must be an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if not (lo <= value <= hi):
        raise ValueError(f"{engine_id}.{name}={value!r} outside [{lo}, {hi}]")
    return value


def _legacy_factory(engine_id):
    return lambda ctx, params: LegacySynthEngine(ctx, params, engine_id)


for _e in _CORE_ENGINES:
    register(EngineSpec(_e['id'], _e['label'], _e['params'], _legacy_factory(_e['id'])))
