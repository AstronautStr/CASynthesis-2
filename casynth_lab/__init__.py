"""casynth_lab -- demo-bench execution library (S1).

Shared by the live UI (demo_bench.py) and offline rendering: one DemoRunner
turns a scene + timed commands into audio blocks; the UI and the offline path
call the same next_block().  DSP is imported from casynth_engine (not copied).
"""
from .scene import Scene, SceneError, load_scene
from .runner import (DemoRunner, Block, BLOCK, SIDES, OUTPUTS, render_offline,
                     write_wav, describe_difference, validate_param, engine_defaults)

__all__ = ['Scene', 'SceneError', 'load_scene', 'DemoRunner', 'Block', 'BLOCK',
           'SIDES', 'OUTPUTS', 'render_offline', 'write_wav', 'describe_difference',
           'validate_param', 'engine_defaults']
