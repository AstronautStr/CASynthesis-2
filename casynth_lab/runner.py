"""DemoRunner -- the single live/offline executor (S1).

Owns ALL mutable state: field, excitation, clocks, SlotPool, phases, current
amps/pan, gain.  Commands arrive through post() (a queue) tagged with the output
sample index at which they may be applied; each is applied at the first block
boundary >= that time, and its actual time + order is recorded in `journal`.

Clocks (all in samples, SR from casynth_config):
  out_samples : transport clock -- every block ever produced by next_block(),
                monotone, never reset (command timestamps refer to it).
  t_samples   : scene audio clock -- advances while the scene is running;
                reset by 'reset'.
  ca_samples  : automaton clock -- advances while running AND not paused;
                frozen by 'pause', so un-pausing never replays missed steps.
Tempo is derived from the number of rendered samples, never from wall time.

DSP is imported from casynth_engine (step / events_field / analyse /
SlotPool.update / render_chunk_laplacian); nothing is re-implemented here.
"""
import wave

import numpy as np

from casynth_config import (SR, CHUNK_S, MASTER_GAIN, VOL_DEFAULT, TOTAL_SLOTS,
                            GEN_ATTACK_DEFAULT, GEN_DECAY_DEFAULT,
                            GEN_SUSTAIN_DEFAULT, GEN_RELEASE_DEFAULT)
from casynth_engine import (step, events_field, analyse, SlotPool,
                            render_chunk_laplacian)

BLOCK = int(CHUNK_S * SR)          # samples per audio block (== engine chunk)
CHANNELS = 2
PAN_CENTER = 0.5                   # S1: both channels identical
COMMANDS = ('start', 'pause', 'set_cell', 'reset', 'vol')


def _gen_chunks(frac, interval_s):
    """Fraction of one automaton tick -> chunk count, as in gol_synth."""
    return max(1, round(frac * interval_s / CHUNK_S))


class DemoRunner:
    def __init__(self, scene, vol=VOL_DEFAULT):
        self.scene = scene
        self.vol = float(vol)
        self.out_samples = 0
        self.journal = []              # (out_sample, seq, kind, args)
        self._pending = []             # [(seq, at, kind, args)]
        self._seq = 0
        self.clip_blocks = 0
        self.peak = 0.0
        self._init_scene_state()

    # -- state ----------------------------------------------------------------
    def _init_scene_state(self):
        sc = self.scene
        self.grid = sc.initial_grid()
        self.exc = None
        self.gen = 0
        self.t_samples = 0
        self.ca_samples = 0
        self.running = False
        self.paused = False
        self.step_samples = SR / sc.rate_hz
        interval = 1.0 / sc.rate_hz
        self._release_chunks = _gen_chunks(GEN_RELEASE_DEFAULT, interval)
        self._attack_chunks = _gen_chunks(GEN_ATTACK_DEFAULT, interval)
        self._decay_chunks = _gen_chunks(GEN_DECAY_DEFAULT, interval)
        self._sustain = float(GEN_SUSTAIN_DEFAULT)
        self.pool = SlotPool()
        sz = TOTAL_SLOTS + 1
        self.phase = np.zeros(sz)
        self.amp_cur = np.zeros(sz)
        self.pan_cur = np.full(sz, PAN_CENTER)
        self.gain_prev = self._gain()
        self._analyse()

    def _gain(self):
        return MASTER_GAIN * self.vol * self.scene.level

    def _analyse(self):
        sc = self.scene
        _labels, voices, _color = analyse(self.grid, sc.f0_hz, sc.engine_id,
                                          sc.engine_params, exc=self.exc)
        for v in voices:
            v['pan'] = PAN_CENTER
        self.voices = voices

    # -- commands -------------------------------------------------------------
    def post(self, kind, at=None, **args):
        """Queue a command.  `at` = output-sample index not before which it may
        apply (None = next block boundary).  Returns the command's sequence no."""
        if kind not in COMMANDS:
            raise ValueError(f"unknown command {kind!r}")
        self._seq += 1
        self._pending.append((self._seq, at, kind, args))
        return self._seq

    def _apply_due(self):
        due = [c for c in self._pending
               if c[1] is None or c[1] <= self.out_samples]
        if not due:
            return
        due.sort(key=lambda c: ((-1 if c[1] is None else c[1]), c[0]))
        for c in due:
            self._pending.remove(c)
        for seq, _at, kind, args in due:
            self._apply(kind, args)
            self.journal.append((self.out_samples, seq, kind, dict(args)))
            if kind == 'reset':
                # Reset discards everything still queued for the future (those
                # events belong to the previous run).
                self._pending.clear()

    def _apply(self, kind, args):
        if kind == 'start':
            self.running = True
        elif kind == 'pause':
            self.paused = bool(args.get('on', not self.paused))
        elif kind == 'set_cell':
            r, c, v = int(args['r']), int(args['c']), int(bool(args['v']))
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                return
            if self.grid[r, c] != v:
                prev = self.grid.copy()
                self.grid[r, c] = v
                self.exc = events_field(prev, self.grid)
                self._analyse()
        elif kind == 'reset':
            self._init_scene_state()
            self.running = True
        elif kind == 'vol':
            self.vol = float(min(max(args['value'], 0.0), 1.0))

    # -- audio ----------------------------------------------------------------
    def _step(self):
        prev = self.grid
        self.grid = step(prev)
        self.exc = events_field(prev, self.grid)
        self.gen += 1
        self._analyse()
        self.journal.append((self.out_samples, 0, 'step', {'gen': self.gen}))

    def next_block(self):
        """Produce the next int16 (BLOCK x 2) block.  The ONLY audio entry point
        for both the live thread and the offline renderer."""
        self._apply_due()
        if not self.running:
            self.out_samples += BLOCK
            return np.zeros((BLOCK, CHANNELS), np.int16)
        if not self.paused and self.ca_samples >= (self.gen + 1) * self.step_samples:
            self._step()
        gain = self._gain()
        self.pool.update(self.voices, self.phase, self.amp_cur, self.pan_cur,
                         self._release_chunks, self._attack_chunks,
                         self._decay_chunks, self._sustain, amp_slew=False)
        buf, peak, n_clip = render_chunk_laplacian(
            self.phase, self.amp_cur, self.pan_cur, self.pool.amp_tgt,
            self.pool.pan_tgt, self.pool.freq_slots, CHANNELS,
            self.gain_prev, gain, 1.0)
        self.gain_prev = gain
        self.peak = peak
        if n_clip > 0:
            self.clip_blocks += 1
        self.out_samples += BLOCK
        self.t_samples += BLOCK
        if not self.paused:
            self.ca_samples += BLOCK
        return buf

    def snapshot(self):
        return dict(grid=self.grid.copy(), gen=self.gen, running=self.running,
                    paused=self.paused, t_seconds=self.t_samples / SR,
                    vol=self.vol, peak=self.peak, clip_blocks=self.clip_blocks,
                    out_samples=self.out_samples)


def render_offline(scene, seconds, commands=None, vol=VOL_DEFAULT, runner=None):
    """Render `seconds` of audio through DemoRunner, trimmed to the exact sample
    count, no normalisation.  `commands` = [(kind, at_samples, {args})...];
    default = start at 0.  Returns (int16 (N x 2), runner)."""
    n = int(round(seconds * SR))
    if runner is None:
        runner = DemoRunner(scene, vol=vol)
    if commands is None:
        commands = [('start', 0, {})]
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    blocks, got = [], 0
    while got < n:
        b = runner.next_block()
        blocks.append(b)
        got += len(b)
    if blocks:
        pcm = np.concatenate(blocks, axis=0)[:n]
    else:
        pcm = np.zeros((0, CHANNELS), np.int16)
    return pcm, runner


def write_wav(path, pcm):
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
