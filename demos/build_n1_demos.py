#!/usr/bin/env python3
"""Build the prepared listening material of N1 (REQ memory/req-network-ca-n1-2026-09-15.md;
user rule 2026-09-15: every listening goes through the experiment catalog):

    python demos/build_n1_demos.py                 -> scenes demos/n1_*.json (rewritten,
                                                      identical content) + catalog
                                                      lab_catalog/network_n1_2026_09_15/
    python demos/build_n1_demos.py --scenes-only   -> only the scene files
    python demos/build_n1_demos.py --root DIR      -> catalog under another directory

The target catalog may exist only while it holds no records (a bench that was
opened on it creates `.tmp`); otherwise an explicit error -- records the user
added are never overwritten.  `run_network_n1.bat` opens the bench on this
catalog's screen: double click / Continue = the live field from the record's
end state (byte-exact), Notes = the user's feedback (`python -m
casynth_lab.catalog notes lab_catalog/network_n1_2026_09_15`).

Field (all records): 32 x 32, B3/S23, torus, four vertical blinkers in columns
4 / 12 / 20 / 28 (rows 14..16), zero-based -- the fixtures' `vertical` field.
Records (12 s of evolution at 2 generations / s unless noted; both sides are in
every record, 1 / 2 switch the listened side):
  N1-follow   A follows the field, B = Freeze CA (the REQ's A / B)
  N1-edits    the fixtures' edit schedule: the field moved by 8 rows and back every
              4 s, 20 s, CA paused, no restart (A follows, B = Freeze CA)
  N1-depth    A CA amount 1, B CA amount 0 (manual scale only)
  N1-links    A Links 127, B Links 230 (both follow the field)
  N1-routes   A the N1 routes (review R1: node 6 has no right outlet), B the N0
              routing -- the same network otherwise
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR                                          # noqa: E402
from casynth_lab import scene_from_doc, DemoRunner, BLOCK              # noqa: E402
from casynth_lab.catalog import Catalog                                # noqa: E402
from casynth_lab.recorder import Recorder                              # noqa: E402
from casynth_lab.gutter_field import ENGINE_ID, ENGINE_ID_N0R          # noqa: E402

DEMOS_DIR = os.path.join(ROOT, 'demos')
CATALOG_ROOT = os.path.join(ROOT, 'lab_catalog', 'network_n1_2026_09_15')
ROWS = COLS = 32
RATE_HZ = 2.0
SEC_EVOLVE = 12.0
SEC_EDITS = 20.0
EDIT_PERIOD_S = 4.0
EDIT_SHIFT_ROWS = 8
COLUMNS = (4, 12, 20, 28)
LISTEN = "1 / 2 = listen A / B. Space pauses the CA; paint anywhere -- each region tunes its node."


def field_vertical(shift=0):
    return [[r + shift, c] for c in COLUMNS for r in (14, 15, 16)]


def gutter(**over):
    p = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)
    p.update(over)
    return ENGINE_ID, p


def gutter_n0_routes(**over):
    return ENGINE_ID_N0R, gutter(**over)[1]


def scene_doc(sid, title, A, B, listen=LISTEN):
    return dict(format=2, id=sid, title=title, grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in field_vertical()],
                rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=110.0, level=1.0),
                variants=dict(A=dict(engine_id=A[0], engine_params=A[1]),
                              B=dict(engine_id=B[0], engine_params=B[1])),
                initial_side='A', listen=listen)


# (scene id, record title, note, A, B, kind)   kind: 'evolve' | 'edits'
RECORDS = [
    ('network_n1_blinkers', "N1-follow: follows the field / Freeze CA",
     "Four blinkers evolving at 2 steps/s. A: the eight resonator banks follow the live-cell "
     "count of their regions; B: Freeze CA holds the initial control. Same start, same knobs.",
     gutter(), gutter(freeze_ca=1), 'evolve'),
    ('n1_edits', "N1-edits: the field moved and back every 4 s",
     "CA paused; the blinkers are moved 8 rows down and back every 4 s for 20 s without a "
     "restart. A follows the edits, B = Freeze CA (control).",
     gutter(), gutter(freeze_ca=1), 'edits'),
    ('n1_depth', "N1-depth: CA amount 1 / 0",
     "Same evolution. A: CA amount 1 (the field tunes the banks); B: CA amount 0 (manual "
     "Resonators scale only, the field does not act).",
     gutter(depth=1.0), gutter(depth=0.0), 'evolve'),
    ('n1_links', "N1-links: Links 127 / 230",
     "Same evolution, both follow the field. A: Links 127 (source default); B: Links 230 "
     "(strong coupling between the nodes through the 2064-sample delay).",
     gutter(interaction=127), gutter(interaction=230), 'evolve'),
    ('n1_routes', "N1-routes: R1 fix / N0 routing",
     "Same evolution and knobs. A: node 6 has no right outlet (review R1, the source patch); "
     "B: node 6 also on the right, as in the N0 package. Nothing else differs.",
     gutter(), gutter_n0_routes(), 'evolve'),
]


def scene_for(rec):
    sid, title, note, A, B, kind = rec
    return scene_doc(sid, title, A, B)


def write_scenes():
    paths = []
    for rec in RECORDS:
        doc = scene_for(rec)
        scene_from_doc(doc)                     # strict validation before writing
        path = os.path.join(DEMOS_DIR, doc['id'] + '.json')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(doc, f, indent=1)
            f.write('\n')
        paths.append(path)
    return paths


def edit_commands(seconds=SEC_EDITS, period=EDIT_PERIOD_S, shift=EDIT_SHIFT_ROWS):
    """The fixtures' schedule: vertical at 0, moved at 4, vertical at 8, ... (block-aligned
    by the runner: applied at the first block boundary at or after the nominal time)."""
    cmds = [('pause', 0, dict(on=True))]
    k = 1
    while k * period < seconds:
        at = int(round(k * period * SR))
        cur = field_vertical(shift) if k % 2 == 1 else field_vertical(0)
        prev = field_vertical(0) if k % 2 == 1 else field_vertical(shift)
        cmds += [('set_cell', at, dict(r=r, c=c, v=0)) for r, c in prev]
        cmds += [('set_cell', at, dict(r=r, c=c, v=1)) for r, c in cur]
        k += 1
    return cmds


def commands_for(kind):
    if kind == 'evolve':
        return []
    if kind == 'edits':
        return edit_commands()
    raise ValueError(kind)


def seconds_for(kind):
    return SEC_EDITS if kind == 'edits' else SEC_EVOLVE


# -- recording (offline, same Recorder / Cut / Catalog.save path as the live bench) --
def record_offline(catalog, doc, seconds, commands, title, note):
    """Drive a DemoRunner block by block (no device, no wall clock) with 'start' at
    block 0 plus `commands` [(kind, at_samples, args)], feed the same Recorder the
    live bench uses, cut at the last block boundary and save."""
    runner = DemoRunner(scene_from_doc(doc))
    rec = Recorder(runner, None)
    runner.post('start', at=0)
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    n_blocks = int(np.ceil(seconds * SR / BLOCK))
    for _ in range(n_blocks):
        before = runner.out_samples
        blk = runner.next_block()
        rec.on_block(blk, runner.running, before)
    snap = runner.snapshot()
    diagnostics = dict(device_ok=False, device_error='offline build', underruns=0,
                       block_errors=0, last_error=None, clip_blocks=dict(snap['clip_blocks']),
                       record_error=None)
    cut = rec.cut(diagnostics)
    assert cut is not None and cut.n_frames == n_blocks * BLOCK
    rid = catalog.save(cut, title, note)
    return rid, snap


def build(root):
    if os.path.exists(root):
        present = [n for n in os.listdir(root) if not n.startswith('.')]
        if present:
            raise SystemExit(f"error: target catalog already holds records: {root}\n"
                             f"(rebuild into a new directory with --root, or remove it yourself; "
                             f"records added by the user are never overwritten)")
    os.makedirs(os.path.join(root, '.tmp'), exist_ok=True)
    cat = Catalog(root)
    # the catalog lists newest first: save in reverse so that the order above
    # is the order on screen (ids carry a 1 s timestamp -> wait between saves)
    results = []
    for rec in reversed(RECORDS):
        sid, title, note, A, B, kind = rec
        doc = scene_for(rec)
        secs = seconds_for(kind)
        t0 = time.time()
        rid, snap = record_offline(cat, doc, secs, commands_for(kind), title, note)
        r = cat.load(rid)
        results.append((rid, title, r.version_label(), snap['clip_blocks'],
                        snap['running'], snap['paused'], snap['gen']))
        print(f"  {rid}  {title}  ({secs:.0f} s, {r.version_label()}, clip {snap['clip_blocks']}, "
              f"running={snap['running']} paused={snap['paused']} gen={snap['gen']})")
        wait = 1.05 - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the N1 listening scenes and catalog")
    ap.add_argument('--root', default=CATALOG_ROOT, help="catalog root (must hold no records)")
    ap.add_argument('--scenes-only', action='store_true')
    a = ap.parse_args(argv)
    paths = write_scenes()
    print(f"scenes: {len(paths)} files in {DEMOS_DIR}")
    if a.scenes_only:
        return 0
    res = build(a.root)
    print(f"catalog: {len(res)} records in {a.root}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
