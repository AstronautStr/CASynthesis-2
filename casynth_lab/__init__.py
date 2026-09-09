"""casynth_lab -- demo-bench execution library (S1).

Shared by the live UI (demo_bench.py) and offline rendering: one DemoRunner
turns a scene + timed commands into audio blocks; the UI and the offline path
call the same next_block().  DSP is imported from casynth_engine (not copied).
"""
from . import registry
from .engine_api import EngineContext, SoundEngine, EngineBlockError, check_block
from .registry import EngineSpec, register
from .scene import Scene, SceneError, load_scene, scene_from_doc
from .runner import (DemoRunner, Block, BLOCK, SIDES, OUTPUTS, render_offline,
                     write_wav, describe_difference, validate_param, engine_defaults)

__all__ = ['Scene', 'SceneError', 'load_scene', 'scene_from_doc', 'DemoRunner', 'Block', 'BLOCK',
           'SIDES', 'OUTPUTS', 'render_offline', 'write_wav', 'describe_difference',
           'validate_param', 'engine_defaults', 'registry', 'EngineContext',
           'SoundEngine', 'EngineBlockError', 'check_block', 'EngineSpec', 'register']
