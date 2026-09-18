"""Bench engine registry: id -> (label, parameter specs, factory).

The UI, scene validation, commands and offline render all read from here.
The five existing methods are registered from casynth_core.ENGINES (their
metadata is NOT duplicated) through LegacySynthEngine.  A new engine module
registers itself with register() -- explicitly, in one place (see
demos/README.md); no directory scanning or hot loading.

Param spec tuple (same as casynth_core): (arg, label, lo, hi, integer, default).
"""
import math

from casynth_core import ENGINES as _CORE_ENGINES, ENGINE_BY_ID as _CORE_BY_ID
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
                 'text').  It must use the SAME geometry as the engine.
      ranges   : {param: (min, max)} -- a float parameter whose slider covers a
                 USER-EDITABLE sub-range (Objects "Radius x", 2026-09-17): the
                 pair is the default range of the slider, the spec's lo / hi
                 stay the validation bounds (lo <= min < max <= hi).  The bench
                 keeps the edited range per side (runner set_range, scene
                 `param_ranges`); the registry entry itself is never changed.
      validate : validate(params) -> None, raising ValueError for a COMBINATION
                 of values the engine does not define (Objects Decay law,
                 2026-09-18); every single value has already passed its range.
                 The scene loader, the runner (before a command changes anything)
                 and the snapshot call it through validate_params()."""
    __slots__ = ('id', 'label', 'params', 'factory', 'choices', 'inactive', 'overlay', 'ranges', 'validate')

    def __init__(self, id, label, params, factory, choices=None, inactive=None, overlay=None,
                 ranges=None, validate=None):
        self.id = id
        self.label = label
        self.params = tuple(tuple(p) for p in params)
        self.factory = factory       # factory(ctx, params) -> SoundEngine
        self.choices = {k: tuple(v) for k, v in (choices or {}).items()}
        self.inactive = inactive
        self.overlay = overlay
        self.ranges = {k: (float(v[0]), float(v[1])) for k, v in (ranges or {}).items()}
        self.validate = validate

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

# The seven spectrum settings of the old Laplace (n spread alpha shape harm
# fullshape dyn): engines that offer them under the same names share them
# through the bench's "copy spectrum" command (2026-09-17).
SPECTRUM_KEYS = tuple(p[0] for p in _CORE_BY_ID['laplacian']['params'])


def spectrum_keys(engine_a, engine_b):
    """The spectrum settings both engines offer (in SPECTRUM_KEYS order)."""
    a = {p[0] for p in get(engine_a).params}
    b = {p[0] for p in get(engine_b).params}
    return tuple(k for k in SPECTRUM_KEYS if k in a and k in b)


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
    for name, (rmin, rmax) in spec.ranges.items():
        p = spec.spec_of(name)
        if p is None or p[4] or not (math.isfinite(rmin) and math.isfinite(rmax))                 or not (p[2] <= rmin < rmax <= p[3]):
            raise ValueError(f"engine {spec.id!r}: default range of {name!r} must be a finite "
                             f"lo <= min < max <= hi of a float parameter")
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


def validate_range(engine_id, name, lo, hi):
    """A user range of a ranged parameter (EngineSpec.ranges): finite numbers with
    spec lo <= lo < hi <= spec hi; returns (lo, hi) as floats or raises ValueError."""
    if engine_id not in REGISTRY:
        raise ValueError(f"unknown engine {engine_id!r}")
    spec = get(engine_id)
    if name not in spec.ranges:
        raise ValueError(f"{engine_id}.{name}: not a parameter with a user range")
    _arg, _label, slo, shi, _integer, _default = spec.spec_of(name)
    out = []
    for label, v in (('min', lo), ('max', hi)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{engine_id}.{name} range {label}: must be a finite number, got {v!r}")
        out.append(float(v))
    lo, hi = out
    if not (slo <= lo):
        raise ValueError(f"{engine_id}.{name} range min {lo!r} below {slo}")
    if not (hi <= shi):
        raise ValueError(f"{engine_id}.{name} range max {hi!r} above {shi}")
    if not (lo < hi):
        raise ValueError(f"{engine_id}.{name} range: min {lo!r} must be below max {hi!r}")
    return lo, hi


def default_range(engine_id, name):
    """(min, max) default slider range of a ranged parameter, else None."""
    return get(engine_id).ranges.get(name)


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


def validate_params(engine_id, params):
    """The engine's combination rule on a FULL parameter dict (after the per-value
    checks): returns the dict, raises ValueError.  Engines without a rule accept
    everything."""
    fn = get(engine_id).validate
    if fn is not None:
        fn(params)
    return params


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
# N3 (2026-09-16): the field tunes the resonances, its events strike them.
from . import tuned_events             # noqa: E402,F401
# N4 (2026-09-16): every figure a resonator bank of its own Laplacian, struck inside its circle.
from . import object_resonators        # noqa: E402,F401
