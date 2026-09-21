"""Which widget an engine parameter gets -- the one answer both panels read.

The spec of a parameter has been the same tuple in both UIs from the start
(`(arg, label, lo, hi, integer, default)` -- casynth_core and the registry), but
the DECISION on top of it was written twice: the bench grew four widget kinds
(a word row for named choices, a pill for an on/off integer, a line of text for
a parameter that does not act, a slider otherwise, with the user's sub-range
when the parameter has one) while the prototype only ever drew a slider.

This module holds that decision and nothing else.  Geometry, colours, fonts and
number formatting stay in each UI -- they are what makes the two panels look
like what they are, and unifying them would move pixels.

    for row in panel_rows(spec, params, ranges=..., inactive=...):
        row.kind    -- 'slider' | 'toggle' | 'choices' | 'inactive'
        row.lo, row.hi   -- the bounds the widget spans (the user range if any)
        row.bound_lo, row.bound_hi -- what the spec allows (a range stays inside)
        row.value   -- the applied value
        row.choices -- the words of a named-choice parameter (else ())
        row.text    -- why an inactive parameter does not act (else '')
"""

KIND_SLIDER = 'slider'
KIND_TOGGLE = 'toggle'
KIND_CHOICES = 'choices'
KIND_INACTIVE = 'inactive'


class PanelRow:
    """One parameter of one engine, ready to be drawn."""
    __slots__ = ('arg', 'label', 'kind', 'lo', 'hi', 'integer', 'value',
                 'choices', 'text', 'ranged', 'bound_lo', 'bound_hi')

    def __init__(self, arg, label, kind, lo, hi, integer, value,
                 choices=(), text='', ranged=False, bounds=None):
        self.arg = arg
        self.label = label
        self.kind = kind
        self.lo = lo
        self.hi = hi
        # what the SPEC allows, which a user range may narrow but never leave
        self.bound_lo, self.bound_hi = bounds if bounds is not None else (lo, hi)
        self.integer = bool(integer)
        self.value = value
        self.choices = tuple(choices)
        self.text = text
        self.ranged = bool(ranged)     # the slider spans a user-editable sub-range

    def __repr__(self):              # diagnostics only
        return (f"PanelRow({self.arg!r}, {self.kind}, {self.lo}..{self.hi}, "
                f"value={self.value!r})")


def range_text(value, integer):
    """How an end of a slider's range is written: whole for an integer knob,
    shortest exact form otherwise (0, 0.5, 1.97).  Both panels print it and the
    prototype's Min / Max fields parse what they print."""
    return f"{int(round(value)):d}" if integer else f"{float(value):g}"


def is_toggle(spec_tuple):
    """An integer parameter with exactly two values is an on/off switch."""
    _arg, _label, lo, hi, integer, _default = spec_tuple
    return bool(integer) and lo == 0 and hi == 1


def panel_rows(spec, params, ranges=None, inactive=None, order=None):
    """[PanelRow] for the parameters of `spec` (an EngineSpec).

    params   : the applied parameter dict (every parameter of the engine).
    ranges   : {arg: (lo, hi)} the slider ranges in force for ranged parameters
               (a runner's edited range); absent entries use the registry default.
    inactive : {arg: text} already computed by the caller, or None to ask the
               engine itself (spec.inactive(params)).
    order    : the arg names to emit, in that order (default: registry order).
               A caller may reorder or drop rows -- the prototype puts the
               settings engines SHARE first and leaves inactive ones out.
    """
    if inactive is None:
        fn = spec.inactive
        inactive = fn(params) if fn is not None else {}
    ranges = ranges or {}
    by_arg = {p[0]: p for p in spec.params}
    names = list(order) if order is not None else [p[0] for p in spec.params]
    out = []
    for name in names:
        p = by_arg.get(name)
        if p is None:
            continue                       # not a parameter of this engine
        arg, label, lo, hi, integer, _default = p
        value = params[arg]
        # the bounds a parameter's widget spans do not depend on whether it acts:
        # a ranged parameter keeps the user's sub-range even while shown as text
        bounds = (lo, hi)
        ranged = arg in spec.ranges
        if ranged or arg in ranges:
            # the bench gives a sub-range to the parameters that declare one; the
            # prototype gives every slider a Min / Max pair (2026-09-22), and both
            # arrive here as `ranges`
            rlo, rhi = ranges.get(arg) or spec.ranges[arg]
            lo, hi = float(rlo), float(rhi)
        if arg in inactive:
            kind, extra = KIND_INACTIVE, dict(text=inactive[arg])
        elif arg in spec.choices:
            kind, extra = KIND_CHOICES, dict(choices=spec.choices[arg])
        elif is_toggle(p):
            kind, extra = KIND_TOGGLE, {}
        else:
            kind, extra = KIND_SLIDER, {}
        out.append(PanelRow(arg, label, kind, lo, hi, integer, value,
                            ranged=ranged, bounds=bounds, **extra))
    return out


def shared_first(spec, shared):
    """Arg names of `spec` with the settings in `shared` first, in THAT order,
    then the engine's own in registry order -- the prototype's reading order
    (the settings most engines have in common sit at the top of column A, so
    switching engines moves as few rows as possible)."""
    own = [p[0] for p in spec.params]
    head = [k for k in shared if k in own]
    return head + [k for k in own if k not in head]
