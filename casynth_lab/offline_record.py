"""Scene -> catalog record, offline: the one path from a scene document to a
saved experiment (2026-09-22).

It drives a DemoRunner block by block -- no device, no wall clock -- through the
SAME Recorder / Cut / Catalog.save the live bench uses, so a record made here is
indistinguishable from one a listener saved by hand: same metadata, same
provenance and pin, same end snapshot to Continue from, and `verify` replays it
the same way.

It was written twice in demos/ (build_sn_demos, build_n1_demos) and is now
needed a third time, by the prototype's Save button (a session the player
recorded becomes an experiment), so it lives here.

    rid, snapshot = record(catalog, doc, seconds, title='...', note='...',
                           commands=[('pause', 0, {'on': True})])
"""
import math

import numpy as np

from casynth_config import SR
from .recorder import Recorder, WINDOW_SECONDS_DEFAULT
from .runner import BLOCK, DemoRunner
from .scene import scene_from_doc


def record(catalog, doc, seconds, title=None, note='', commands=(), on_block=None):
    """Render `doc` for `seconds` and save it as a record.  -> (record id, snapshot).

    commands : [(kind, at_samples, args)] posted before the first block, on top of
               the 'start' this always posts (the scene's own script still runs).
    on_block : optional callback(i, n_blocks) for a progress bar.
    """
    runner = DemoRunner(scene_from_doc(doc))
    n_blocks = int(math.ceil(float(seconds) * SR / BLOCK))
    # The Recorder keeps only the LAST `window_seconds` of audio in a ring -- the
    # live bench records a ROLLING window.  An offline build is the whole render,
    # so the ring is sized to it; with the default 30 s window a longer render
    # came back cut to 30 s and the frame check below refused the record
    # (2026-09-22: the prototype's Save on a session longer than half a minute).
    # Shorter renders keep the default window, so their cuts are unchanged.
    rec = Recorder(runner, None,
                   window_seconds=max(WINDOW_SECONDS_DEFAULT,
                                      n_blocks * BLOCK / float(SR)))
    runner.post('start', at=0)
    for kind, at, args in commands:
        runner.post(kind, at=at, **dict(args or {}))
    for i in range(n_blocks):
        before = runner.out_samples
        blk = runner.next_block()
        rec.on_block(blk, runner.running, before)
        if on_block is not None:
            on_block(i, n_blocks)
    snap = runner.snapshot()
    diagnostics = dict(device_ok=False, device_error='offline build', underruns=0,
                       block_errors=0, last_error=None,
                       clip_blocks=dict(snap['clip_blocks']), record_error=None)
    cut = rec.cut(diagnostics)
    if cut is None or cut.n_frames != n_blocks * BLOCK:
        raise RuntimeError(f"offline record: cut {cut and cut.n_frames} frames, "
                           f"expected {n_blocks * BLOCK}")
    return catalog.save(cut, title, note), snap


__all__ = ['record', 'BLOCK', 'SR', 'np']
