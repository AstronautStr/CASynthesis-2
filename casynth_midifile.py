"""Offline MIDI-file playback for gol_synth (optional dependency: mido).

Reads a Standard MIDI File and, on a background daemon thread, delivers
note-on / note-off events to a callback in real time (respecting the file's
tempo map).  Poly->mono reduction is the CALLER's concern -- the callback decides
which held note becomes the carrier; this module only schedules the raw events.

If mido is unavailable, MIDIFILE_AVAILABLE is False and MidiFilePlayer degrades
to a no-op (load returns False) so the synth still runs without it.
"""
import os
import threading
import time

try:
    import mido
    MIDIFILE_AVAILABLE = True
except ImportError:
    mido = None
    MIDIFILE_AVAILABLE = False


class MidiFilePlayer:
    """Load a .mid, then stream its note events to a callback in real time."""

    def __init__(self):
        self.path = None
        self.name = None
        self.length = 0.0     # total duration, seconds (from mido)
        self.pos = 0.0        # elapsed playback time, seconds (approx., for UI)
        self._msgs = None     # [(t_abs_seconds, note, is_on), ...]
        self._thread = None
        self._alive = False
        # Woken by stop() so the scheduler waits ONCE per event (Event.wait) rather
        # than polling in 20 ms slices -- the previous 1 ms MIDI poll thread was
        # removed precisely because its wakeups starved the render thread of the
        # GIL and caused underruns (see log/2026-06-23-midi-timing-jitter.md).
        self._stop_evt = threading.Event()

    @property
    def playing(self):
        return (self._alive and self._thread is not None
                and self._thread.is_alive())

    def load(self, path):
        """Parse `path` into an absolute-time note-event list.  Returns True on
        success (a file with at least one note), False otherwise."""
        if not MIDIFILE_AVAILABLE:
            print("[MIDIFILE] mido not installed")
            return False
        try:
            mid = mido.MidiFile(path)
        except Exception as exc:
            print(f"[MIDIFILE] cannot load {path!r}: {exc}")
            return False
        msgs, t = [], 0.0
        # Iterating a MidiFile merges all tracks and yields each message with
        # .time already in SECONDS (tempo-aware), so a running sum is the
        # absolute onset time on the playback clock.
        for msg in mid:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                msgs.append((t, msg.note, True))
            elif msg.type == 'note_off' or (msg.type == 'note_on'
                                            and msg.velocity == 0):
                msgs.append((t, msg.note, False))
        if not msgs:
            print(f"[MIDIFILE] no notes in {path!r}")
            return False
        self.path = path
        self.name = os.path.basename(path)
        self._msgs = msgs
        self.length = float(mid.length)
        self.pos = 0.0
        return True

    def start(self, callback):
        """(Re)start playback from the top, delivering callback(note, is_on)."""
        self.stop()
        if not self._msgs:
            return
        self._stop_evt.clear()
        self._alive = True
        self._thread = threading.Thread(target=self._run, args=(callback,),
                                        daemon=True, name='midifile')
        self._thread.start()

    def stop(self):
        """Stop playback.  Any notes still held get a final note-off through the
        callback (in _run's cleanup) so the synth never hangs a gate."""
        self._alive = False
        self._stop_evt.set()          # wake the scheduler's wait() immediately
        th = self._thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=1.0)
        self._thread = None
        self.pos = 0.0

    def _run(self, callback):
        t0 = time.perf_counter()
        held = set()
        try:
            for (t, note, on) in self._msgs:
                # Wait ONCE until this event's scheduled time.  Event.wait sleeps
                # the whole interval and returns the instant stop() fires, so the
                # thread wakes only per-event (not ~50x/s), keeping GIL pressure off
                # the render thread.  Absolute-time anchoring (t vs perf-t0) means
                # the wait self-corrects; no cumulative drift.
                ahead = t - (time.perf_counter() - t0)
                if ahead > 0 and self._stop_evt.wait(ahead):
                    break                     # stop() fired
                if not self._alive:
                    break
                if on:
                    held.add(note)
                else:
                    held.discard(note)
                self.pos = time.perf_counter() - t0
                callback(note, on)
        finally:
            # Release anything still sounding so the carrier gate returns to off.
            for note in list(held):
                callback(note, False)
            self._alive = False
