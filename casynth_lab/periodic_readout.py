"""Moved to casynth_engines.periodic_readout (2026-09-21): the sound ENGINES are a
package of their own, so a host can import them without the bench.  This shim
keeps the old name working for the bench, its tests and the pinned worktrees.

It hands the SAME module object over (sys.modules), not a copy of its public
names: casynth_lab.periodic_readout is casynth_engines.periodic_readout, private names and all.
"""
import sys as _sys

from casynth_engines import periodic_readout as _moved

_sys.modules[__name__] = _moved
