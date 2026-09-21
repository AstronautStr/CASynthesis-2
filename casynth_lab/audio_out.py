"""LiveEngine: render thread + sounddevice output for DemoRunner.

  UI thread   --post()-->  cmd queue  -->  render thread (owns DemoRunner)
                                              |  next_block() just-in-time
                                              v
                                    casynth_host.AudioHost  -->  audio callback
                                    (ring + look-ahead)          (copies int16 only)
The UI reads snapshot() (a copy published by the render thread) and never
touches the runner.  Slow frames / painting cannot change the render tempo:
the tempo is the device pulling blocks, the runner counts samples.

The ring, the look-ahead pre-roll, the underrun count and the device belong to
casynth_host (shared with the prototype since 2026-09-21); what is left here is
what only a BENCH does: the A/B runner, the command queue, the transport fade,
the streaming Recorder and the players of a saved take.

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
from casynth_host import AudioHost, open_output_stream, audio_choice_load
from .runner import BLOCK, CHANNELS
from .engine_api import EngineBlockError
from .recorder import Recorder

TRANSPORT_FADE_BLOCKS = 3      # ~24 ms fade-in after reset
PAIR_XFADE_MS = 10.0           # S7 pair player: saved <-> recomputed switch (player only)
PAIR_XFADE_SAMPLES = int(round(PAIR_XFADE_MS / 1000.0 * SR))   # 441


def _default_output_factory(callback):
    """The bench's device: casynth_host opens it, the bench only names its own
    rate and channels (sounddevice stays a lazy import -- the offline path never
    needs a device).

    WHICH output is the player's REMEMBERED choice -- the same audio_device.json
    the prototype writes, because it is the same person and the same pair of
    ears.  Until 2026-09-21 the bench always took the system default, so choosing
    an output in the synth changed nothing here at all, and "I picked the device
    and the bench is still silent" was the honest outcome rather than a mystery.
    A remembered device that will not open falls back to the default: somebody
    who named a device wants SOUND, not silence with a reason (the rule
    gol_synth._open_out already follows)."""
    want = audio_choice_load()
    if want is None:
        return open_output_stream(callback, sr=SR, channels=CHANNELS)
    try:
        return open_output_stream(callback, sr=SR, channels=CHANNELS, device=want)
    except Exception as exc:                        # noqa: BLE001
        print(f"[audio] remembered output {want!r} did not open: {exc}")
        print("[audio] falling back to the system default")
        return open_output_stream(callback, sr=SR, channels=CHANNELS)


class LiveEngine:
    def __init__(self, runner, output_factory=_default_output_factory,
                 sink=None, lookahead=AUDIO_LOOKAHEAD_CHUNKS, record_root=None,
                 record_seconds=None, origin_snapshot=None, parent_record_id=None):
        """output_factory(callback) -> object with .stop()/.close(); raises if
        no device.  sink(monitor_buf, block) -- test hook: receives every block
        (monitor as heard + the full Block with raw A/B) instead of a device.
        record_root: temp dir for the streaming Recorder (None = no recording).
        origin_snapshot / parent_record_id (S5): the runner was restored from
        that record's snapshot -- the recording starts at the snapshot."""
        self.runner = runner
        self._record_root = record_root
        self._record_seconds = record_seconds
        self._origin_snapshot = origin_snapshot
        self._parent_record_id = parent_record_id
        self.muted = False           # live output silenced (catalog screen)
        self.recorder = None
        self._cut_req = False
        self._cut_q = queue.Queue()
        self._player = None            # {'pcm': int16 (n,2), 'pos': int} while a record plays
        self._factory = output_factory
        self._sink = sink
        self._lookahead = lookahead
        self._cmd_q = queue.Queue()
        # the ring + the device: shared with the prototype.  pace=True -> without a
        # device a wall-clock pacer drains it, so the scene still advances.
        self.host = AudioHost(BLOCK, CHANNELS, sr=SR, lookahead=lookahead,
                              output_factory=output_factory, callback=self._audio_cb,
                              pace=True)
        self._alive = False
        self._thread = None
        self.block_errors = 0        # engine blocks rejected (replaced by silence)
        self.last_error = None
        self._fade_left = 0
        self._snap = runner.snapshot()

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        self._alive = True
        if self._record_root is not None:
            kw = {} if self._record_seconds is None else {'window_seconds': self._record_seconds}
            self.recorder = Recorder(self.runner, self._record_root,
                                     origin_snapshot=self._origin_snapshot,
                                     parent_record_id=self._parent_record_id, **kw)
        if self._sink is None:
            self.host.start()                # sets device_ok / device_error itself
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.host.stop()
        if self.recorder is not None:
            self.recorder.discard()

    # -- the device, as the UI reads it ---------------------------------------
    @property
    def device_ok(self):
        return self.host.device_ok

    @property
    def device_error(self):
        return self.host.device_error

    @property
    def underruns(self):
        return self.host.underruns

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

    # -- pair playback: saved / recomputed on ONE cursor (S7) ------------------------
    def play_pair(self, saved, recomputed, which='saved', pos=0):
        """Play one of two int16 (n, 2) versions of the same track, both on a
        shared position: switch_version() continues from the same sample, it
        never restarts.  Beyond the end of the shorter version that version is
        silence (nothing is stretched or hidden); playback ends at the end of
        the longer one.  Unity gain for both, no normalisation; the live synth
        is drained, never mixed in."""
        vers = {}
        for name, pcm in (('saved', saved), ('recomputed', recomputed)):
            pcm = np.ascontiguousarray(pcm, dtype=np.int16)
            if pcm.ndim != 2 or pcm.shape[1] != CHANNELS:
                raise ValueError("play_pair expects (n, 2) int16 arrays")
            vers[name] = pcm
        if which not in vers:
            raise ValueError(f"unknown version {which!r}")
        n = max(len(v) for v in vers.values())
        self._player = {'pcm': None, 'versions': vers, 'which': which,
                        'pos': int(min(max(pos, 0), n)), 'n': n,
                        'fade_from': None, 'fade_left': 0}

    def switch_version(self, which):
        """Switch the pair player to 'saved' / 'recomputed' at the current
        sample (a short player-only crossfade smooths the switch; the PCM
        being compared is untouched).  Returns the position, or None when no
        pair is playing."""
        p = self._player
        if p is None or 'versions' not in p:
            return None
        if which not in p['versions']:
            raise ValueError(f"unknown version {which!r}")
        if which != p['which']:
            p['fade_from'] = p['which']
            p['fade_left'] = PAIR_XFADE_SAMPLES
            p['which'] = which
        return p['pos']

    @property
    def pair_version(self):
        p = self._player
        return p['which'] if p is not None and 'versions' in p else None

    def stop_play(self):
        self._player = None

    @property
    def playing(self):
        return self._player is not None

    @property
    def play_pos(self):
        p = self._player
        if not p:
            return (0, 0)
        if 'versions' in p:
            return (p['pos'], p['n'])
        return (p['pos'], len(p['pcm']))

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
        self.host.flush()

    def _render_loop(self):
        r = self.runner
        while self._alive:
            if self._sink is None and self.host.full():
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
                self.host.put(buf)

    # -- device side ----------------------------------------------------------
    def _audio_cb(self, outdata, frames, time_info, status):
        self.host.count_status(status)
        player = self._player
        if player is None and self.muted:
            self._drain_live(frames)
            outdata[:] = 0
            return
        if player is not None:
            # record playback: the live monitor is drained (kept running) but
            # NOT mixed in; the record's PCM goes to the device alone
            self._drain_live(frames)
            if 'versions' in player:
                self._pair_fill(player, outdata, frames)
                return
            pcm, pos = player['pcm'], player['pos']
            take = min(frames, len(pcm) - pos)
            outdata[:take] = pcm[pos:pos + take]
            if take < frames:
                outdata[take:] = 0
            player['pos'] = pos + take
            if player['pos'] >= len(pcm):
                self._player = None
            return
        self.host.ring.fill(outdata, frames)

    @staticmethod
    def _slice_or_silence(pcm, pos, frames):
        """`frames` samples of pcm from pos, zero-padded past its end."""
        take = max(0, min(frames, len(pcm) - pos))
        if take == frames:
            return pcm[pos:pos + frames]
        out = np.zeros((frames, CHANNELS), np.int16)
        if take > 0:
            out[:take] = pcm[pos:pos + take]
        return out

    def _pair_fill(self, player, outdata, frames):
        pos = player['pos']
        cur = self._slice_or_silence(player['versions'][player['which']], pos, frames)
        left = player['fade_left']
        if left > 0 and player['fade_from'] is not None:
            old = self._slice_or_silence(player['versions'][player['fade_from']], pos, frames)
            done = PAIR_XFADE_SAMPLES - left
            x = np.minimum((np.arange(frames) + done + 1) / float(PAIR_XFADE_SAMPLES), 1.0)[:, None]
            mix = old.astype(np.float64) * (1.0 - x) + cur.astype(np.float64) * x
            outdata[:] = np.rint(mix).astype(np.int16)
            player['fade_left'] = max(0, left - frames)
            if player['fade_left'] == 0:
                player['fade_from'] = None
        else:
            outdata[:] = cur
        player['pos'] = pos + frames
        if player['pos'] >= player['n']:
            self._player = None

    def _drain_live(self, frames):
        """Consume `frames` of live blocks without outputting them."""
        self.host.ring.drain(frames)
