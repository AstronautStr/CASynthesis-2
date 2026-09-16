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
    """Optional display hints (S/N demos, 2026-09-14; the sound never depends
    on them):
      choices  : {param: (name, ...)} -- an integer parameter whose values are
                 modes named by words (index = value); the bench shows buttons
                 with those words instead of a slider / on-off toggle.
      inactive : inactive(params) -> {param: text} -- parameters that do not
                 act for the current settings (shown as that text, not editable).
      overlay  : overlay(params, rows, cols) -> dict drawn over the field for
                 the listened side (see demo_bench: 'polyline' of unwrapped
                 cell coordinates, 'start'/'ahead', 'labels', 'circles',
                 'text').  It must use the SAME geometry as the engine."""
    __slots__ = ('id', 'label', 'params', 'factory', 'choices', 'inactive', 'overlay')

    def __init__(self, id, label, params, factory, choices=None, inactive=None, overlay=None):
        self.id = id
        self.label = label
        self.params = tuple(tuple(p) for p in params)
        self.factory = factory       # factory(ctx, params) -> SoundEngine
        self.choices = {k: tuple(v) for k, v in (choices or {}).items()}
        self.inactive = inactive
        self.overlay = overlay

    def defaults(self):
        return {p[0]: p[5] for p in self.params}

    def spec_of(self, name):
        for p in self.params:
            if p[0] == name:
                return p
        return None

    def value_text(self, name, value):
        """Human text of a parameter value (mode word when the parameter has
        named choices)."""
        names = self.choices.get(name)
        if names is not None and 0 <= int(value) < len(names) and int(value) == value:
            return names[int(value)]
        spec = self.spec_of(name)
        return f"{int(value):d}" if spec is not None and spec[4] else f"{value:g}"


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
    for name, names in spec.choices.items():
        p = spec.spec_of(name)
        if p is None or not p[4] or p[2] != 0 or p[3] != len(names) - 1:
            raise ValueError(f"engine {spec.id!r}: choices of {name!r} do not match its "
                             f"integer range 0..{len(names) - 1}")
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


def value_text(engine_id, name, value):
    return get(engine_id).value_text(name, value)


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

# S/N demo engines (2026-09-14): registered after the built-in five, in this
# one place (each module calls register() at import).
from . import scan_surface    # noqa: E402,F401
from . import pm_network      # noqa: E402,F401
# N1 (2026-09-15): the field-driven Gutter network.
from . import gutter_field    # noqa: E402,F401
from . import gutter_controls    # noqa: E402,F401
# N2 (2026-09-16): the periodic-readout Gutter (A) and the event-driven delay network (B).
from . import gutter_field_periodic    # noqa: E402,F401
from . import event_network            # noqa: E402,F401
