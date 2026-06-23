"""Non-blocking MIDI input for gol_synth (optional dependency: mido).

If mido is unavailable, MIDI_AVAILABLE is False and MidiInput degrades to a no-op
(empty port list, open/poll do nothing) so the synth still runs without MIDI.
"""
import threading

try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    mido = None
    MIDI_AVAILABLE = False


class MidiInput:
    """Non-blocking MIDI input: enumerate ports, open one, poll messages."""

    def __init__(self):
        self.port = None
        self.port_name = None
        # open()/close() run on the main thread (device dropdown); poll() runs on
        # the dedicated MIDI thread.  A lock keeps a close mid-poll from racing.
        self._lock = threading.Lock()

    def ports(self):
        if not MIDI_AVAILABLE:
            return []
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def open(self, name, callback=None):
        self.close()
        if name is None or not MIDI_AVAILABLE:
            return
        with self._lock:
            try:
                # callback mode: rtmidi delivers each message on its own thread the
                # instant it arrives (no polling latency, no GIL-starved poll loop).
                self.port = mido.open_input(name, callback=callback)
                self.port_name = name
            except Exception as exc:
                print(f"[MIDI] cannot open {name!r}: {exc}")
                self.port = None
                self.port_name = None

    def close(self):
        with self._lock:
            if self.port is not None:
                try:
                    self.port.close()
                except Exception:
                    pass
                self.port = None
                self.port_name = None

    def poll(self):
        """Return all pending messages without blocking."""
        with self._lock:
            if self.port is None:
                return []
            try:
                return list(self.port.iter_pending())
            except Exception:
                return []
