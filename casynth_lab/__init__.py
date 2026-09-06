"""casynth_lab -- demo-bench execution library (S1).

Shared by the live UI (demo_bench.py) and offline rendering: one DemoRunner
turns a scene + timed commands into audio blocks; the UI and the offline path
call the same next_block().  DSP is imported from casynth_engine (not copied).
"""
from .scene import Scene, SceneError, load_scene
from .runner import DemoRunner, BLOCK, render_offline, write_wav

__all__ = ['Scene', 'SceneError', 'load_scene', 'DemoRunner', 'BLOCK',
           'render_offline', 'write_wav']
