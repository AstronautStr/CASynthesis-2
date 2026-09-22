#!/usr/bin/env python3
"""Golden-master audio gate for the EVENTS cells of the unified Laplace engine.

WHY (2026-09-22): the four older Laplace laws are pinned byte for byte -- the
golden master for `laplacian`, the catalog records for the carriers, FM and
Objects -- but the Events articulation sounded by a wave (Saw / Square) or by FM
had no versioned reference at all.  Only tests/test_laplace_unified.ThreadedRender
checks them, and it checks the kernel against ITSELF at another thread count.
So the speed work of the performance audit (memory/log/2026-09-22-perf-audit.md)
had nothing to prove "bit for bit" against on exactly the cells it touches most:
the tail pool, the block boundary, the wave-table pool, the FM readout.

This file renders a fixed deterministic programme through every Events cell
(Sine as well, so the Objects kernel itself is pinned here too) and compares the
int16 output byte for byte with a committed reference, the way
tests/golden/golden_master.py does for the baseline.  The programme deliberately
exercises what a speed change could break: a crowded Random field at twelve
generations a second (the tail pool fills and evicts), a spectrum knob moved
under a running field (`_retune_all`), a note change (`_retune`), a waveform
blend (both table readouts in one block), a gain glide, and a sparse field of
library figures that continue, split and die (tails, fades, drops).

The wave-table pool is process-global and LRU: which frequencies it can serve
depends on what was asked before.  The cells are therefore rendered in a FIXED
order from a fresh process, and the reference is only meaningful for that order.

USAGE:
  python tests/golden/events_master.py           # render and compare (PASS/FAIL)
  python tests/golden/events_master.py --save    # re-bless the reference
                                                 # (ONLY after an agreed audible
                                                 #  change; say so in the log)
"""
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, CHUNK_S, MASTER_GAIN                      # noqa: E402
from casynth_engine import step, events_field, midi_to_freq              # noqa: E402
from casynth_engines import registry                                     # noqa: E402
from casynth_engines import laplace_unified as lu                        # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE     # noqa: E402
from patterns import PATTERNS                                            # noqa: E402

REF_NPZ = os.path.join(_HERE, "events_master_ref.npz")
OUT_DIR = os.path.join(_ROOT, "artifacts")

BLOCK = int(CHUNK_S * SR)
ROWS, COLS = 30, 52
STEPS_PER_S = 12.0                      # bpm 120 at 1/16T -- the player's session
# the instrument's start position (gol_synth.START_SPECTRUM): every mode, the
# shape law on, harm fully on -- the settings the audit measured on
SPECTRUM = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=1.0, fullshape=1, dyn=1.0)
EVENTS = dict(events=0, radius_mul=1.0, decay_s=0.8, attack_ms=0.0)
GAIN = MASTER_GAIN * 0.7

# the cells, in the order they are rendered (the table pool is shared)
CELLS = [
    ('events_sine',   dict(voice=lu.VOICE_BANK, waveform=lu.WF_SINE, fm_depth=1.0)),
    ('events_saw',    dict(voice=lu.VOICE_BANK, waveform=lu.WF_SAW, fm_depth=1.0)),
    ('events_square', dict(voice=lu.VOICE_BANK, waveform=lu.WF_SQUARE, fm_depth=1.0)),
    ('events_fm',     dict(voice=lu.VOICE_FM, waveform=lu.WF_SINE, fm_depth=1.0)),
]


# -- the two fields -------------------------------------------------------------
def crowd():
    """About a hundred figures: the tracker's hardest case, the tail pool full."""
    rng = np.random.default_rng(7)
    return (rng.random((ROWS, COLS)) < 0.35).astype(np.uint8)


def library():
    """A few library figures far apart: continuations, an oscillator, a glider
    that walks, and a shape that dies -- tails and identities, not crowds."""
    by_name = {name: cells for _cat, pats in PATTERNS for name, cells in pats}
    g = np.zeros((ROWS, COLS), np.uint8)
    wanted = [n for n in ('Glider', 'Blinker', 'Pulsar', 'R-pentomino', 'Block')
              if n in by_name]
    if not wanted:                                   # pragma: no cover
        wanted = list(by_name)[:4]
    at = [(3, 3), (20, 8), (6, 30), (22, 40), (12, 20)]
    for (r0, c0), name in zip(at, wanted):
        for dr, dc in by_name[name]:
            g[(r0 + dr) % ROWS, (c0 + dc) % COLS] = 1
    return g


# -- the programme --------------------------------------------------------------
# (blocks, note, knob changes at the START of the block) per field; the field
# steps at STEPS_PER_S all along
PROGRAMME = [
    # field factory, blocks, {block: (note, params update)}
    (crowd, 72, {0: (48, {}), 24: (48, dict(harm=0.5)), 36: (55, {}),
                 48: (55, dict(waveform='next')), 60: (48, dict(shape=0.4))}),
    (library, 60, {0: (48, {}), 30: (60, dict(decay_s=0.4)),
                   45: (60, dict(events=1))}),
]


def _params(axes):
    p = dict(SPECTRUM)
    p.update(EVENTS)
    p.update(artic=lu.ARTIC_EVENTS)
    p.update(axes)
    return p


def render_cell(axes):
    """int16 (n, 2) of the whole programme on one cell, plus the digest of the
    Objects state at the end (slots, tails, trackers)."""
    out = []
    for field, blocks, cues in PROGRAMME:
        ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(48), 1.0, STEPS_PER_S,
                            pan=PAN_FIELD_MODE)
        p = _params(axes)
        eng = registry.create('laplace_unified', ctx, dict(p))
        eng.set_rate(STEPS_PER_S)
        eng.set_envelope(0.0, 0.0, 1.0, 0.1, False)
        g = field()
        exc = None
        eng.init(g, exc, GAIN)
        sps = SR / STEPS_PER_S
        cum = gen = 0
        note = 48
        gain_prev = None
        for i in range(blocks):
            if cum >= (gen + 1) * sps:
                new = step(g)
                exc = events_field(g, new)
                g = new
                gen += 1
                eng.update_field(g, exc)
            if i in cues:
                note, upd = cues[i]
                upd = dict(upd)
                if upd.get('waveform') == 'next':
                    upd['waveform'] = (int(p['waveform']) + 1) % 3
                if upd:
                    p.update(upd)
                    eng.set_params(dict(p))
            # a gain glide over blocks 10..20 of every field (the ramp path)
            gain = GAIN * (0.5 if 10 <= i < 20 else 1.0)
            buf, _peak, _clip = eng.render(gain, cum, gain_prev=gain_prev,
                                           transpose=midi_to_freq(note) / midi_to_freq(48))
            gain_prev = None
            out.append(buf.copy())
            cum += BLOCK
    audio = np.concatenate(out)
    lay = eng._active()
    st = lay.cell.obj.export_state()
    h = hashlib.sha256()
    for k in sorted(st):
        v = st[k]
        if isinstance(v, np.ndarray):
            h.update(k.encode()); h.update(np.ascontiguousarray(v).tobytes())
    return audio, h.hexdigest()[:16]


def render():
    return {name: render_cell(axes) for name, axes in CELLS}


def main():
    force_save = "--save" in sys.argv
    got = render()
    for name, (audio, state) in got.items():
        digest = hashlib.sha256(audio.tobytes()).hexdigest()[:16]
        print(f"[events-master] {name:14s} {len(audio)} samples  sha256[:16]={digest}  "
              f"state={state}")
    if force_save or not os.path.exists(REF_NPZ):
        np.savez_compressed(REF_NPZ,
                            **{f"{n}_audio": a for n, (a, _s) in got.items()},
                            **{f"{n}_state": np.array(s) for n, (_a, s) in got.items()},
                            sr=SR, chunk_s=CHUNK_S)
        print(f"[baseline saved] {REF_NPZ}")
        return 0
    ref = np.load(REF_NPZ)
    rc = 0
    for name, (audio, state) in got.items():
        ra = ref[f"{name}_audio"]
        rs = str(ref[f"{name}_state"])
        if audio.shape == ra.shape and np.array_equal(audio, ra) and state == rs:
            print(f"[PASS] {name}: byte-exact ({len(ra)} samples), state {state}")
            continue
        rc = 1
        os.makedirs(OUT_DIR, exist_ok=True)
        from scipy.io import wavfile
        path = os.path.join(OUT_DIR, f"_events_master_{name}.wav")
        wavfile.write(path, SR, audio)
        n = min(len(audio), len(ra))
        if n and audio.shape[1:] == ra.shape[1:]:
            diff = np.abs(audio[:n].astype(np.int64) - ra[:n].astype(np.int64))
            first = int(np.argmax(diff.max(axis=1) > 0)) if diff.max() else -1
            n_diff = int(np.count_nonzero(diff))
            print(f"[FAIL] {name}: audio differs from baseline: shapes ref={ra.shape} "
                  f"out={audio.shape} max|diff|={int(diff.max())} LSB, "
                  f"{n_diff} of {diff.size} values differ ({100.0 * n_diff / diff.size:.3f} %), "
                  f"first at sample {first} (block {first // BLOCK if first >= 0 else '-'}); "
                  f"state {'same' if state == rs else 'DIFFERS'} (wrote {path})")
        else:
            print(f"[FAIL] {name}: shape mismatch ref={ra.shape} out={audio.shape} "
                  f"(wrote {path})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
