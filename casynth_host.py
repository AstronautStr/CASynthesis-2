"""The DEVICE side of a CASynth host: one ring of finished blocks, one callback.

Both hosts had the same ~120 lines: a queue of int16 blocks, a callback that
copies samples out of it across block boundaries, a look-ahead pre-roll of
silence so the first callbacks never starve, an underrun counter and a render
loop throttled by the ring's depth (gol_synth._audio_cb / casynth_lab.audio_out
before 2026-09-21).  That code lives here now; both hosts hold an AudioHost.

What is NOT here (it is the host's own): what a block CONTAINS, the transport,
the note and its VCA, the level meter's decay, recording, and any playback of
a saved take -- the bench keeps its record player by giving its own `callback`
and calling ring.fill() / ring.drain() from it.

Threads: the render thread only calls put() / full(); the audio callback only
calls ring.fill() / ring.drain().  A queue.Queue is the only shared object, so
nothing but finished int16 blocks crosses the boundary.
"""
import json
import os
import queue
import threading
import time

import numpy as np

from casynth_config import SR, AUDIO_LOOKAHEAD_CHUNKS


def open_output_stream(callback, sr=SR, channels=2, device=None):
    """A started sounddevice OutputStream (int16, low latency) -- the one device
    both hosts open.  sounddevice is imported here: an offline render never
    needs it.

    `device` = None means the system default, which is not always the thing the
    person is listening to: a machine can have a headset, a monitor and onboard
    speakers all at once, and "no sound" usually means the stream opened
    somewhere else.  An index or a name substring picks one (see
    output_devices())."""
    import sounddevice as sd
    stream = sd.OutputStream(samplerate=sr, channels=channels, dtype='int16',
                             latency='low', callback=callback, device=device)
    stream.start()
    return stream


def output_device_name(device=None):
    """Human name of the output that would be opened (never raises)."""
    try:
        import sounddevice as sd
        info = sd.query_devices(device, kind='output')
        api = sd.query_hostapis(info['hostapi'])['name']
        return f"{info['name']} [{api}]"
    except Exception as e:                    # noqa: BLE001
        return f"unknown ({e})"


def output_devices():
    """[(index, name, host api, channels)] of every output -- for a person who
    cannot hear anything and needs to know where the sound went."""
    try:
        import sounddevice as sd
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_output_channels'] > 0:
                out.append((i, d['name'], sd.query_hostapis(d['hostapi'])['name'],
                            int(d['max_output_channels'])))
        return out
    except Exception:                         # noqa: BLE001
        return []


# -- which output, remembered between runs --------------------------------------
# The system default is not always the device a person is listening to: a machine
# can have a headset, a monitor and onboard speakers at once, and an endpoint can
# accept a stream and render nothing audible.  So the choice is the player's, and
# it outlives the session.  Stored by NAME (indices move when devices come and
# go) with the index as a fallback hint.
AUDIO_CHOICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'audio_device.json')


def audio_choice_save(device, path=AUDIO_CHOICE_FILE):
    """Remember the chosen output (None = follow the system default)."""
    doc = {'device': None}
    if device is not None:
        try:
            import sounddevice as sd
            info = sd.query_devices(device, kind='output')
            doc = {'device': {'name': info['name'],
                              'hostapi': sd.query_hostapis(info['hostapi'])['name'],
                              'index': int(device) if isinstance(device, int) else None}}
        except Exception:                        # noqa: BLE001
            doc = {'device': {'name': str(device), 'hostapi': None, 'index': None}}
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
    except OSError:
        pass
    return doc


def audio_choice_load(path=AUDIO_CHOICE_FILE):
    """The remembered output as an index, or None to follow the system default.
    A device that is gone (unplugged, renamed) resolves to None rather than to
    somebody else's speakers."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    want = (doc or {}).get('device')
    if not isinstance(want, dict):
        return None
    name, api, idx = want.get('name'), want.get('hostapi'), want.get('index')
    for i, dev_name, dev_api, _ch in output_devices():
        if dev_name == name and (api is None or dev_api == api):
            return i
    if isinstance(idx, int):
        for i, _n, _a, _c in output_devices():
            if i == idx:
                return i
    return None


class BlockRing:
    """Finished int16 blocks between the render thread and the audio callback.

    The blocks a host renders and the frames a device asks for are different
    sizes, so the reader keeps a partially consumed block (`_resid`) across
    callbacks.  An empty ring is an underrun: the rest of the device buffer is
    silence (never a repeated block -- that clicks) and the count goes up."""

    def __init__(self, block, channels):
        self.block = int(block)
        self.channels = int(channels)
        self.underruns = 0
        self._q = queue.Queue()
        self._resid = {'buf': None, 'pos': 0}

    # -- render side ----------------------------------------------------------
    def put(self, buf):
        self._q.put(buf)

    def qsize(self):
        return self._q.qsize()

    def full(self, lookahead):
        return self._q.qsize() >= lookahead

    def prefill(self, n):
        """Top the ring UP TO n blocks with silence: the look-ahead the callback
        starts on.  Only what is missing -- a host whose render thread has already
        put real audio in must not have silence queued behind it."""
        silent = np.zeros((self.block, self.channels), np.int16)
        for _ in range(max(0, int(n) - self._q.qsize())):
            self._q.put(silent.copy())

    def flush(self):
        """Drop every queued block (a transport reset); the block being consumed
        in the callback is left alone."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def take(self, timeout=None):
        """One block off the ring, or None within `timeout` (the pacer)."""
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            return None

    # -- device side ----------------------------------------------------------
    def fill(self, outdata, frames):
        """Write `frames` samples into outdata from the ring; silence + one
        underrun when it runs dry."""
        filled = 0
        while filled < frames:
            if self._resid['buf'] is None:
                try:
                    self._resid['buf'] = self._q.get_nowait()
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

    def drain(self, frames):
        """Consume `frames` without outputting them: the live synth keeps
        running while something else is heard (the bench's record player)."""
        left = frames
        while left > 0:
            if self._resid['buf'] is None:
                try:
                    self._resid['buf'] = self._q.get_nowait()
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


class AudioHost:
    """A BlockRing plus the output device it feeds.

    start() pre-rolls `lookahead` blocks of silence and opens the device; the
    device then pulls at the hardware rate and a slow frame in the host only
    shrinks the ring.  With pace=True a host that could NOT open a device gets a
    wall-clock pacer draining the ring at real time, so its scene still advances
    while device_ok stays False (never a fake "audio OK").
    A host that renders into a test sink instead of a device simply never starts
    this one: an unstarted ring is empty, so full() is False (no throttle) and
    underruns stays 0 -- there is no real time to underrun against.
    `callback` replaces the default one (which is just ring.fill) when the host
    needs to decide per callback what is heard (the bench's record player)."""

    def __init__(self, block, channels, sr=SR, lookahead=AUDIO_LOOKAHEAD_CHUNKS,
                 output_factory=open_output_stream, callback=None, pace=False):
        self.ring = BlockRing(block, channels)
        self.sr = int(sr)
        self.lookahead = int(lookahead)
        self.device_ok = False
        self.device_error = None
        self.latency = None            # device latency in seconds (None = unknown)
        self._factory = output_factory
        self._callback = callback if callback is not None else self._default_cb
        self._pace = bool(pace)
        self._stream = None
        self._pacer = None
        self._alive = False

    # -- ring (the render thread's view) --------------------------------------
    @property
    def underruns(self):
        return self.ring.underruns

    def put(self, buf):
        self.ring.put(buf)

    def full(self):
        """The ring holds the whole look-ahead: the render loop may idle."""
        return self.ring.full(self.lookahead)

    def qsize(self):
        return self.ring.qsize()

    def flush(self):
        self.ring.flush()

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        """Pre-roll + open the device.  Returns device_ok; the error (if any) is
        in device_error -- a host decides itself whether to run silently."""
        self._alive = True
        try:
            self.ring.prefill(self.lookahead)
            self._stream = self._factory(self._callback)
            self.device_ok = True
            self.latency = getattr(self._stream, 'latency', None)
        except Exception as e:                    # noqa: BLE001
            self.device_ok = False
            self.device_error = str(e) or type(e).__name__
            if self._pace:
                self._pacer = threading.Thread(target=self._pace_loop, daemon=True)
                self._pacer.start()
        return self.device_ok

    def reopen(self):
        """Open the output again -- after the host's factory has been pointed at
        another device.  The render thread and the ring are untouched: only who
        DRAINS the ring changes.  Whatever the old device had queued is dropped,
        because it belongs to a stream nobody is listening to any more.

        Returns device_ok; the reason for a failure is in device_error, and the
        host keeps running silently rather than dying on a bad choice."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                     # noqa: BLE001
                pass
            self._stream = None
        self.device_ok = False
        self.device_error = None
        self.latency = None
        self.ring.flush()
        try:
            self.ring.prefill(self.lookahead)
            self._stream = self._factory(self._callback)
            self.device_ok = True
            self.latency = getattr(self._stream, 'latency', None)
        except Exception as e:                    # noqa: BLE001
            self.device_error = str(e) or type(e).__name__
        return self.device_ok

    def stop(self):
        self._alive = False
        if self._pacer is not None:
            self._pacer.join(timeout=2.0)
            self._pacer = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                     # noqa: BLE001
                pass
            self._stream = None

    # -- device side ----------------------------------------------------------
    def _default_cb(self, outdata, frames, time_info, status):
        self.count_status(status)
        self.ring.fill(outdata, frames)

    def count_status(self, status):
        """PortAudio told us it ran dry: count it like an empty ring."""
        if status and getattr(status, 'output_underflow', False):
            self.ring.underruns += 1

    def _pace_loop(self):
        """No device: consume blocks at real time so the host still advances."""
        period = self.ring.block / float(self.sr)
        nxt = time.perf_counter()
        while self._alive:
            nxt += period
            self.ring.take(timeout=period)
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()
