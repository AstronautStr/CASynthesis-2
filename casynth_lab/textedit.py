"""Moved to casynth_textedit (2026-09-22): the text-field model is shared by the
bench AND the prototype (the Min / Max fields of a knob row), and the prototype
must not import the bench (the seam, memory/req-seam-2026-09-21.md).

It hands the SAME module object over (sys.modules), not a copy of its public
names: casynth_lab.textedit is casynth_textedit, private names and all.
"""
import sys as _sys

import casynth_textedit as _moved

_sys.modules[__name__] = _moved
