"""Recorder: keeps the LAST `window_seconds` of every rendered Block (raw A /
raw B / monitor) of a live DemoRunner in bounded ring buffers, and produces
consistent CUTS.

Runs on the render thread (never in the audio callback).

ORIGIN = the moment the current recording started: the first Start, a Start
after Stop, or a Restart.  Each of these begins a NEW recording: the origin
state (field, both sides' engine + params, selected side, volume -- i.e. the
conditions a replay starts from, with all audio memory at zero) is captured,
the command journal counts from there, and the audio window rolls behind the
current block boundary, keeping at most `window_seconds`.

A cut = (origin state, journal since the origin, the PCM window
[audio_start_sample, end_sample), diagnostics) -- one interval, one truth.
Replay recomputes from the origin and compares the saved window byte-exact.

S5: a session CONTINUED from a record's end snapshot starts its recording at
that snapshot (origin_kind = 'snapshot', origin_snapshot = the pristine loaded
state, parent_record_id = the record); a later Restart / Start-after-Stop
begins a fresh recording again (origin_kind = 'fresh') but keeps the parent
link.  Every cut also carries the runner's END snapshot (export_state at the
cut boundary) when all engines support it -- the next "Continue" point.
"""
import uuid

import numpy as np

from casynth_config import SR
from .runner import BLOCK, CHANNELS, OUTPUTS, XFADE_SAMPLES

WINDOW_SECONDS_DEFAULT = 30.0


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


def _origin_state(runner):
    return dict(cells=[[int(a), int(b)] for a, b in np.argwhere(runner.grid > 0)],
                vol=float(runner.vol), selected=runner.selected,
                settings={side: {'engine_id': eid, 'engine_params': dict(p)}
                          for side, (eid, p) in runner.side_settings().items()},
                param_memory=runner.param_memory(),
                param_ranges=runner.param_ranges(),
                factory_variants={side: {'engine_id': eid, 'engine_params': dict(p)}
                                  for side, (eid, p) in runner.scene.factory_variants.items()})


class Recorder:
    def __init__(self, runner, tmp_root=None, window_seconds=WINDOW_SECONDS_DEFAULT,
                 origin_snapshot=None, parent_record_id=None):
        self.runner = runner
        self.parent_record_id = parent_record_id
        self.origin_kind = None            # 'fresh' | 'snapshot' while started
        self.origin_snapshot = None        # pristine state (snapshot origins only)
        self.session_id = uuid.uuid4().hex[:8]
        self.window_seconds = float(window_seconds)
        n_blocks = max(1, int(np.ceil(self.window_seconds * SR / BLOCK)))
        self.ring_frames = n_blocks * BLOCK
        self._ring = {o: np.zeros((self.ring_frames, CHANNELS), np.int16) for o in OUTPUTS}
        self._w = 0                        # ring write position (frames)
        self.origin_sample = None          # out_sample of the current recording's origin
        self.origin_state = None
        self.frames_since_origin = 0
        self.error = None                  # set by close(): recording stopped
        self.scene_doc = runner.scene.doc
        self._prev_running = bool(runner.running)
        self._closed = False
        if origin_snapshot is not None and runner.running:
            # continuation: the recording starts right here, at the snapshot
            self.origin_sample = runner.out_samples
            self.origin_state = _origin_state(runner)
            self.origin_kind = 'snapshot'
            self.origin_snapshot = origin_snapshot

    @property
    def started(self):
        return self.origin_sample is not None

    @property
    def audio_start_sample(self):
        if not self.started:
            return None
        return self.runner.out_samples - min(self.frames_since_origin, self.ring_frames)

    def _is_new_origin(self, running, out_sample_before):
        """A Restart at this boundary, or the scene turning on (Start, or the
        pause of a stopped scene released)."""
        if running and not self._prev_running:
            return True
        journal = self.runner.journal
        for t, _seq, kind, _args in reversed(journal):
            if t != out_sample_before:
                break
            if kind == 'reset':
                return True
        return False

    def on_block(self, blk, running, out_sample_before):
        """Called by the render thread after each next_block().
        `blk` may be None (rejected engine block -> silence was output)."""
        if self._closed:
            return
        if self._is_new_origin(running, out_sample_before):
            self.origin_sample = out_sample_before
            self.origin_state = _origin_state(self.runner)
            self.origin_kind = 'fresh'
            self.origin_snapshot = None
            self.frames_since_origin = 0
            self._w = 0
        self._prev_running = running
        if not self.started:
            return
        for o in OUTPUTS:
            buf = blk.get(o) if blk is not None else None
            dst = self._ring[o][self._w:self._w + BLOCK]
            if buf is None:
                dst[:] = 0
            else:
                dst[:] = buf
        self._w = (self._w + BLOCK) % self.ring_frames
        self.frames_since_origin += BLOCK

    def _window(self, o, n):
        """Contiguous copy of the last n frames of output o."""
        if n <= 0:
            return np.zeros((0, CHANNELS), np.int16)
        start = (self._w - n) % self.ring_frames
        ring = self._ring[o]
        if start + n <= self.ring_frames:
            return ring[start:start + n].copy()
        return np.concatenate([ring[start:], ring[:start + n - self.ring_frames]], axis=0)

    def cut(self, diagnostics):
        """Consistent cut at the current block boundary (render thread)."""
        if not self.started:
            return None
        r = self.runner
        end = r.out_samples
        n = min(self.frames_since_origin, self.ring_frames)
        snap_ok, snap_why = r.snapshot_support()
        end_snapshot = r.export_state() if snap_ok else None
        return Cut(
            provenance=r.provenance,
            origin_kind=self.origin_kind,
            origin_snapshot=self.origin_snapshot,
            parent_record_id=self.parent_record_id,
            end_snapshot=end_snapshot,
            end_snapshot_reason=snap_why,
            session_id=self.session_id,
            scene_doc=self.scene_doc,
            origin_sample=self.origin_sample,
            origin_state=dict(self.origin_state),
            vol_initial=float(self.origin_state['vol']),
            audio_start_sample=end - n,
            end_sample=end,
            window_seconds=self.window_seconds,
            journal=[(int(t), int(s), k, dict(a)) for (t, s, k, a) in r.journal
                     if self.origin_sample <= t < end],
            pcm={o: self._window(o, n) for o in OUTPUTS},
            record_error=self.error,
            side_settings=r.side_settings(),
            selected=r.selected,
            state_at_end=dict(_origin_state(r),
                              gen=int(r.gen), paused=bool(r.paused),
                              running=bool(r.running)),
            runner_settings=dict(sr=SR, block=BLOCK, channels=CHANNELS,
                                 xfade_samples=XFADE_SAMPLES),
            diagnostics=dict(diagnostics),
        )

    def close(self):
        """Stop recording (session end / failure): later cuts report an error."""
        self._closed = True
        if self.error is None:
            self.error = "recording stopped"

    def discard(self):
        self.close()
        self._ring = {}
