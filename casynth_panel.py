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
                 'choices', 'text', 'ranged')

    def __init__(self, arg, label, kind, lo, hi, integer, value,
                 choices=(), text='', ranged=False):
        self.arg = arg
        self.label = label
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.integer = bool(integer)
        self.value = value
        self.choices = tuple(choices)
        self.text = text
        self.ranged = bool(ranged)     # the slider spans a user-editable sub-range

    def __repr__(self):              # diagnostics only
        return (f"PanelRow({self.arg!r}, {self.kind}, {self.lo}..{self.hi}, "
                f"value={self.value!r})")


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
        ranged = arg in spec.ranges
        if ranged:
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
                            ranged=ranged, **extra))
    return out


def shared_first(spec, shared):
    """Arg names of `spec` with the settings in `shared` first, in THAT order,
    then the engine's own in registry order -- the prototype's reading order
    (the settings most engines have in common sit at the top of column A, so
    switching engines moves as few rows as possible)."""
    own = [p[0] for p in spec.params]
    head = [k for k in shared if k in own]
    return head + [k for k in own if k not in head]
