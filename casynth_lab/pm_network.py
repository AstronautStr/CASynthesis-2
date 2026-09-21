"""Moved to casynth_engines.pm_network (2026-09-21): the sound ENGINES are a
package of their own, so a host can import them without the bench.  This shim
keeps the old name working for the bench, its tests and the pinned worktrees.

It hands the SAME module object over (sys.modules), not a copy of its public
names: casynth_lab.pm_network is casynth_engines.pm_network, private names and all.
"""
import sys as _sys

from casynth_engines import pm_network as _moved

_sys.modules[__name__] = _moved
