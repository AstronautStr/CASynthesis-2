"""casynth_engines -- the SOUND ENGINES, as a package of their own (2026-09-21).

A host (the prototype gol_synth.py, the demo bench, an offline render) gets a
sound engine from here: the registry, the SoundEngine contract and the engine
modules.  Nothing in this package knows about a bench, a catalog, a device or a
UI -- that is casynth_lab.  The split exists so a host can play the engines
without importing the development bench (the seam, memory/req-seam-2026-09-21).

Importing this package stays cheap: registration is lazy (registry._LAZY), so
the numba kernels and the big resonator banks are imported by the first
get(engine_id) that needs them, not here.  casynth_lab keeps a shim on every
moved name, so `from casynth_lab import gutter_field` is still the same module.
"""
from . import registry
from .engine_api import (EngineContext, SoundEngine, EngineBlockError, check_block,
                         supports_snapshot)
from .registry import EngineSpec, register

__all__ = ['registry', 'EngineContext', 'SoundEngine', 'EngineBlockError', 'check_block',
           'supports_snapshot', 'EngineSpec', 'register']
