"""LiveEngine: render thread + sounddevice output for DemoRunner.

  UI thread   --post()-->  cmd queue  -->  render thread (owns DemoRunner)
                                              |  next_block() just-in-time
                                              v
                                          block queue  -->  audio callback
                                                            (copies int16 only)
The UI reads snapshot() (a copy published by the render thread) and never
touches the runner.  Slow frames / painting cannot change the render tempo:
the tempo is the device pulling blocks, the runner counts samples.

Transport fade: 'reset' flushes queued blocks and applies a short explicit
fade-in to the fresh audio (transport only; the runner's PCM is untouched).
No device -> a wall-clock pacer drains blocks so the demo still runs, and the
status says so (never a fake "audio OK").
"""
import queue
import threading
import time

import numpy as np

from casynth_config import SR, AUDIO_LOOKAHEAD_CHUNKS
from .runner import BLOCK, CHANNELS
from .engine_api import EngineBlockError
from .recorder import Recorder

TRANSPORT_FADE_BLOCKS = 3      # ~24 ms fade-in after reset


def _default_output_factory(callback):
    import sounddevice as sd     # imported lazily: offline path never needs it
    stream = sd.OutputStream(samplerate=SR, channels=CHANNELS, dtype='int16',
                             latency='low', callback=callback)
    stream.start()
    return stream


class LiveEngine:
    def __init__(self, runner, output_factory=_default_output_factory,
                 sink=None, lookahead=AUDIO_LOOKAHEAD_CHUNKS, record_root=None,
                 record_seconds=None):
        """output_factory(callback) -> object with .stop()/.close(); raises if
        no device.  sink(monitor_buf, block) -- test hook: receives every block
        (monitor as heard + the full Block with raw A/B) instead of a device.
        record_root: temp dir for the streaming Recorder (None = no recording)."""
        self.runner = runner
        self._record_root = record_root
        self._record_seconds = record_seconds
        self.muted = False           # live output silenced (catalog screen)
        self.recorder = None
        self._cut_req = False
        self._cut_q = queue.Queue()
        self._player = None            # {'pcm': int16 (n,2), 'pos': int} while a record plays
        self._factory = output_factory
        self._sink = sink
        self._lookahead = lookahead
        self._cmd_q = queue.Queue()
        self._blk_q = queue.Queue()
        self._resid = {'buf': None, 'pos': 0}
        self._alive = False
        self._thread = None
        self._pacer = None
        self._stream = None
        self.device_ok = False
        self.device_error = None
        self.underruns = 0
        self.block_errors = 0        # engine blocks rejected (replaced by silence)
        self.last_error = None
        self._fade_left = 0
        self._snap = runner.snapshot()

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        self._alive = True
        if self._record_root is not None:
            kw = {} if self._record_seconds is None else {'window_seconds': self._record_seconds}
            self.recorder = Recorder(self.runner, self._record_root, **kw)
        if self._sink is None:
            try:
                silent = np.zeros((BLOCK, CHANNELS), np.int16)
                for _ in range(self._lookahead):
                    self._blk_q.put(silent.copy())
                self._stream = self._factory(self._audio_cb)
                self.device_ok = True
            except Exception as e:           # noqa: BLE001
                self.device_ok = False
                self.device_error = str(e) or type(e).__name__
                self._pacer = threading.Thread(target=self._pace_loop, daemon=True)
                self._pacer.start()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pacer is not None:
            self._pacer.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                # noqa: BLE001
                pass
        if self.recorder is not None:
            self.recorder.discard()

    # -- recording / cut (S4) ----------------------------------------------------
    def diagnostics(self):
        s = self._snap
        return dict(device_ok=self.device_ok, device_error=self.device_error,
                    underruns=self.underruns, block_errors=self.block_errors,
                    last_error=self.last_error, clip_blocks=dict(s['clip_blocks']),
                    record_error=(self.recorder.error if self.recorder else None))

    def request_cut(self):
        """Ask the render thread for a consistent cut at the next completed
        block boundary; fetch it with take_cut()."""
        self._cut_req = True

    def take_cut(self, timeout=None):
        """-> ('ok', Cut) | ('none', None) (nothing started yet) | None (not yet)."""
        try:
            return self._cut_q.get(timeout=timeout) if timeout else self._cut_q.get_nowait()
        except queue.Empty:
            return None

    def cut_now(self, timeout=5.0):
        """Convenience for tests / scripts: request + wait."""
        self.request_cut()
        return self.take_cut(timeout=timeout)

    # -- record playback through the same output (S4) ------------------------------
    def play_pcm(self, pcm):
        """Play an int16 (n, 2) array instead of the live monitor.  The live
        experiment keeps running (its blocks are drained) but is muted, so the
        record and the synth never sound together."""
        pcm = np.ascontiguousarray(pcm, dtype=np.int16)
        if pcm.ndim != 2 or pcm.shape[1] != CHANNELS:
            raise ValueError("play_pcm expects (n, 2) int16")
        self._player = {'pcm': pcm, 'pos': 0}

    def stop_play(self):
        self._player = None

    @property
    def playing(self):
        return self._player is not None

    @property
    def play_pos(self):
        p = self._player
        return (p['pos'], len(p['pcm'])) if p else (0, 0)

    # -- UI side --------------------------------------------------------------
    def post(self, kind, at=None, **args):
        """Validate (registry ranges etc.) on the caller's thread -- a bad
        command raises here and never reaches the runner -- then queue."""
        self.runner._check(kind, dict(args))
        self._cmd_q.put((kind, at, args))

    def snapshot(self):
        return self._snap

    def status_text(self):
        err = (f"ENGINE ERROR x{self.block_errors} ({self.last_error})  "
               if self.block_errors else "")
        if self._sink is not None:
            return err + "test sink"
        if not self.device_ok:
            return err + f"NO AUDIO DEVICE (silent): {self.device_error}"
        c = self._snap['clip_blocks']
        return err + f"OK  underrun {self.underruns}  clip A {c['A']} / B {c['B']}"

    # -- render thread --------------------------------------------------------
    def _flush_blocks(self):
        while True:
            try:
                self._blk_q.get_nowait()
            except queue.Empty:
                return

    def _render_loop(self):
        r = self.runner
        while self._alive:
            if self._sink is None and self._blk_q.qsize() >= self._lookahead:
                time.sleep(0.001)
                continue
            while True:
                try:
                    kind, at, args = self._cmd_q.get_nowait()
                except queue.Empty:
                    break
                if kind in ('reset', 'stop'):
                    self._flush_blocks()
                    self._fade_left = TRANSPORT_FADE_BLOCKS
                r.post(kind, at=at, **args)
            try:
                blk = r.next_block()
                buf = blk.monitor
            except EngineBlockError as e:
                # a malformed engine block never reaches the device: silence
                # instead, counted + shown in the status
                self.block_errors += 1
                self.last_error = str(e)
                blk = None
                buf = np.zeros((BLOCK, CHANNELS), np.int16)
            if self._fade_left > 0:
                k = TRANSPORT_FADE_BLOCKS - self._fade_left
                ramp = (np.linspace(k, k + 1, len(buf), endpoint=False)
                        / TRANSPORT_FADE_BLOCKS)
                buf = (buf.astype(np.float64) * ramp[:, None]).astype(np.int16)
                self._fade_left -= 1
            self._snap = r.snapshot()
            if self.recorder is not None:
                self.recorder.on_block(blk, r.running, r.out_samples - BLOCK)
            if self._cut_req:
                self._cut_req = False
                cut = (self.recorder.cut(self.diagnostics())
                       if self.recorder is not None else None)
                self._cut_q.put(('ok', cut) if cut is not None else ('none', None))
            if self._sink is not None:
                self._sink(buf, blk)
            else:
                self._blk_q.put(buf)

    # -- device side ----------------------------------------------------------
    def _audio_cb(self, outdata, frames, time_info, status):
        if status and getattr(status, 'output_underflow', False):
            self.underruns += 1
        player = self._player
        if player is None and self.muted:
            self._drain_live(frames)
            outdata[:] = 0
            return
        if player is not None:
            # record playback: the live monitor is drained (kept running) but
            # NOT mixed in; the record's PCM goes to the device alone
            self._drain_live(frames)
            pcm, pos = player['pcm'], player['pos']
            take = min(frames, len(pcm) - pos)
            outdata[:take] = pcm[pos:pos + take]
            if take < frames:
                outdata[take:] = 0
            player['pos'] = pos + take
            if player['pos'] >= len(pcm):
                self._player = None
            return
        filled = 0
        while filled < frames:
            if self._resid['buf'] is None:
                try:
                    self._resid['buf'] = self._blk_q.get_nowait()
                    self._resid['pos'] = 0
                except queue.Empty:
                    outdata[filled:] = 0
                    self.underruns += 1
                    return
            buf, pos = self._resid['buf'], self._resid['pos']
            take = min(frames - filled, len(buf) - pos)
            outdata[filled:filled + take] = buf[pos:pos + take]
            filled += take
            pos += take
            if pos >= len(buf):
                self._resid['buf'] = None
            else:
                self._resid['pos'] = pos

    def _drain_live(self, frames):
        """Consume `frames` of live blocks without outputting them."""
        left = frames
        while left > 0:
            if self._resid['buf'] is None:
                try:
                    self._resid['buf'] = self._blk_q.get_nowait()
                    self._resid['pos'] = 0
                except queue.Empty:
                    return
            buf, pos = self._resid['buf'], self._resid['pos']
            take = min(left, len(buf) - pos)
            pos += take
            left -= take
            if pos >= len(buf):
                self._resid['buf'] = None
            else:
                self._resid['pos'] = pos

    def _pace_loop(self):
        """No device: consume blocks at real time so the scene still advances."""
        period = BLOCK / SR
        nxt = time.perf_counter()
        while self._alive:
            nxt += period
            try:
                self._blk_q.get(timeout=period)
            except queue.Empty:
                pass
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()
