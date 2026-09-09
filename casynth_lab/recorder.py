"""Recorder: streams every rendered Block (raw A / raw B / monitor) of a live
DemoRunner to temporary raw PCM files, and produces consistent CUTS.

Runs on the render thread (never in the audio callback): one append per
output per block.  Recording starts automatically with the first block of a
RUNNING scene (`audio_start_sample`); everything after it -- pauses, Stop
silence, Restart -- is part of the timeline.  Events before the start are
kept in the runner's journal (they restore the initial settings on replay).

A cut = (end_sample at a completed block boundary, the journal applied
before it, the byte counts of the three raw files, the scene document and
the runner's initial settings + diagnostics) -- one interval, one truth.
The catalog turns a cut into a record (WAVs + record.json) off-thread.
"""
import os
import uuid

import numpy as np

from casynth_config import SR
from .runner import BLOCK, CHANNELS, OUTPUTS, XFADE_SAMPLES

BYTES_PER_FRAME = 2 * CHANNELS


class RecorderError(RuntimeError):
    pass


class Cut:
    """Immutable description of a saved interval (see module docstring)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def n_frames(self):
        return self.end_sample - self.audio_start_sample

    @property
    def seconds(self):
        return self.n_frames / SR


class Recorder:
    def __init__(self, runner, tmp_root):
        self.runner = runner
        self.session_id = uuid.uuid4().hex[:8]
        self.dir = os.path.join(tmp_root, self.session_id)
        self.audio_start_sample = None
        self.frames = 0                    # frames written to each raw file
        self.error = None                  # first write failure (recording stops)
        self.vol_initial = float(runner.vol)
        self.scene_doc = runner.scene.doc
        self._files = {}
        try:
            os.makedirs(self.dir, exist_ok=True)
            for o in OUTPUTS:
                self._files[o] = open(self.raw_path(o), 'wb')
        except OSError as e:
            self.error = f"cannot open temp recording files: {e}"
            self._files = {}

    def raw_path(self, output):
        return os.path.join(self.dir, f"{output}.raw")

    @property
    def started(self):
        return self.audio_start_sample is not None

    def on_block(self, blk, running, out_sample_before):
        """Called by the render thread after each next_block().
        `blk` may be None (rejected engine block -> silence was output)."""
        if not running and not self.started:
            return
        if not self.started:
            self.audio_start_sample = out_sample_before
        if self.error is not None:
            return
        silent = None
        try:
            for o in OUTPUTS:
                buf = blk.get(o) if blk is not None else None
                if buf is None:
                    if silent is None:
                        silent = np.zeros((BLOCK, CHANNELS), np.int16)
                    buf = silent
                self._files[o].write(np.ascontiguousarray(buf, dtype=np.int16).tobytes())
            self.frames += BLOCK
        except (OSError, ValueError, KeyError) as e:
            self.error = f"recording write failed at frame {self.frames}: {e}"
            self.close()

    def cut(self, diagnostics):
        """Consistent cut at the current block boundary (render thread)."""
        if not self.started:
            return None
        try:
            for f in self._files.values():
                f.flush()
        except OSError as e:
            self.error = f"flush failed: {e}"
        r = self.runner
        end = r.out_samples
        expected = end - self.audio_start_sample
        return Cut(
            session_id=self.session_id,
            scene_doc=self.scene_doc,
            vol_initial=self.vol_initial,
            audio_start_sample=self.audio_start_sample,
            end_sample=end,
            journal=[(int(t), int(s), k, dict(a)) for (t, s, k, a) in r.journal
                     if t < end],
            raw_paths={o: self.raw_path(o) for o in OUTPUTS},
            n_bytes=self.frames * BYTES_PER_FRAME,
            frames_written=self.frames,
            frames_expected=expected,
            record_error=self.error,
            side_settings=r.side_settings(),
            selected=r.selected,
            state_at_end=dict(cells=[[int(a), int(b)] for a, b in np.argwhere(r.grid > 0)],
                              gen=int(r.gen), vol=float(r.vol), paused=bool(r.paused),
                              running=bool(r.running)),
            runner_settings=dict(sr=SR, block=BLOCK, channels=CHANNELS,
                                 xfade_samples=XFADE_SAMPLES),
            diagnostics=dict(diagnostics),
        )

    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except OSError:
                pass
        self._files = {}

    def discard(self):
        """Remove the temp files (session end).  Finished records are elsewhere."""
        self.close()
        for o in OUTPUTS:
            try:
                os.remove(self.raw_path(o))
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass
