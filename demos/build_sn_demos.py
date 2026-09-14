#!/usr/bin/env python3
"""Build the prepared material of the S / N sound demos (REQ 2026-09-14, section 5):

    python demos/build_sn_demos.py                 -> scenes demos/sn_*.json (rewritten,
                                                      identical content) + catalog
                                                      lab_catalog/sn_demos_2026_09_14/{scan,network}/
    python demos/build_sn_demos.py --scenes-only   -> only the scene files
    python demos/build_sn_demos.py --root DIR      -> catalog under another directory

The target catalog must NOT exist yet (an explicit error otherwise): a rebuild
goes to a new directory or after the user removes the old one -- records the
user added are never overwritten.  lab_catalog/local/ is never touched.

Fields (32x32, zero-based [row, col]; cell lists are expanded into the JSON,
nothing is generated at listening time):
  F1  : g[r, c] = 1 if (r//4 + 2*(c//5)) % 3 == 0   (asymmetric structure, whole field)
  F2  : F1 transposed
  F3  : Pulsar (patterns.py), bbox 13x13, anchor (9, 9)
  F4L : Pond, anchor (6, 6);   F4R : Pond, anchor (6, 21)
  F5T : rows r < 16 live;      F5B : rows r >= 16 live
  KOK : Kok's galaxy (period 8) centred -- the REQ's reserve evolution case
Records (8 s static with the CA paused from the first block, running=True /
paused=True in the end snapshot; 12 s evolution at 2 steps/s for F3):
  scan/     S-path (Ellipse/Lissajous, Ellipse/Raster), S-surface (F1, F2),
            S-Laplace (Smooth; Distance; harm=1 control)
  network/  N-links (top, bottom), N-field (F4 left, F4 right), N-frozen
            (Pulsar; Kok's galaxy reserve), N-Laplace (coupling 2; harm=1 control)
Laplace control parameters: n=12, spread=0, alpha=1, shape=1, harm=0,
fullshape=1, dyn=0 (harm=1 in the named extra record).  Constant level
corrections: Scan trim_db = +6 dB, Network trim_db = +3 dB (registry defaults;
RMS matched to the Laplace side on F3, never per record).
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
from casynth_lab import registry, scene_from_doc, DemoRunner, BLOCK   # noqa: E402
from casynth_lab.catalog import Catalog                                # noqa: E402
from casynth_lab.recorder import Recorder                              # noqa: E402
from patterns import PATTERNS                                          # noqa: E402

DEMOS_DIR = os.path.join(ROOT, 'demos')
CATALOG_ROOT = os.path.join(ROOT, 'lab_catalog', 'sn_demos_2026_09_14')
SUBCATALOGS = ('scan', 'network')
ROWS = COLS = 32
F0 = 110.0
RATE_STATIC = 4.0        # never steps: the CA is paused from the first block
RATE_EVOLVE = 2.0        # F3 evolution: 2 generations / s
SEC_STATIC = 8.0
SEC_EVOLVE = 12.0
LAPLACE = dict(n=12, spread=0, alpha=1, shape=1, harm=0, fullshape=1, dyn=0)
LAPLACE_HARM = dict(LAPLACE, harm=1)
SCAN_TRIM_DB = 6.0
NETWORK_TRIM_DB = 3.0

_LIB = {name: cells for _cat, items in PATTERNS for name, cells in items}


# -- fields -----------------------------------------------------------------------
def _place(cells, r0, c0):
    g = np.zeros((ROWS, COLS), np.uint8)
    for r, c in cells:
        g[r + r0, c + c0] = 1
    return g


def field_f1():
    g = np.zeros((ROWS, COLS), np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            g[r, c] = 1 if (r // 4 + 2 * (c // 5)) % 3 == 0 else 0
    return g


def field_f2():
    return np.ascontiguousarray(field_f1().T)


def field_f3():
    return _place(_LIB['Pulsar'], 9, 9)


def field_f4(anchor):
    return _place(_LIB['Pond'], *anchor)


def field_f5(top):
    g = np.zeros((ROWS, COLS), np.uint8)
    if top:
        g[:16] = 1
    else:
        g[16:] = 1
    return g


def field_kok():
    """Reserve case (section 5): Kok's galaxy centred in the field."""
    cells = _LIB["Kok's galaxy"]
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    return _place(cells, (ROWS - h) // 2, (COLS - w) // 2)


FIELDS = {
    'F1': field_f1, 'F2': field_f2, 'F3': field_f3,
    'F4L': lambda: field_f4((6, 6)), 'F4R': lambda: field_f4((6, 21)),
    'F5T': lambda: field_f5(True), 'F5B': lambda: field_f5(False),
    'KOK': field_kok,
}


# -- engines ------------------------------------------------------------------------
def scan(**over):
    p = dict(registry.defaults('scan_surface'))
    p['trim_db'] = SCAN_TRIM_DB
    p.update(over)
    return ('scan_surface', p)


def network(**over):
    p = dict(registry.defaults('pm_network'))
    p['trim_db'] = NETWORK_TRIM_DB
    p.update(over)
    return ('pm_network', p)


def laplace(harm=0):
    return ('laplacian', dict(LAPLACE, harm=harm))


# -- scenes ---------------------------------------------------------------------------
def scene_doc(sid, title, field, A, B, rate_hz, listen, f0=F0):
    g = FIELDS[field]()
    return dict(format=2, id=sid, title=title,
                grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in np.argwhere(g > 0)],
                rule='B3/S23', boundary='torus', rate_hz=rate_hz,
                audio=dict(f0_hz=f0, level=1.0),
                variants=dict(A=dict(engine_id=A[0], engine_params=A[1]),
                              B=dict(engine_id=B[0], engine_params=B[1])),
                initial_side='A', listen=listen)


# (catalog, scene id, record title, note, field, A, B, static?)
RECORDS = [
    ('scan', 'sn_s_path_ellipse_lissajous', "S-path: Ellipse / Lissajous",
     "F1 frozen. Same surface (Smooth, width 1.5) and field; only the reading path differs: "
     "A Ellipse, B Lissajous 2:3 (radii 0.8).",
     'F1', scan(path=0), scan(path=1), True),
    ('scan', 'sn_s_path_ellipse_raster', "S-path: Ellipse / Raster",
     "F1 frozen. A Ellipse (radii 0.8), B Raster (all cells). Surface, width and level equal.",
     'F1', scan(path=0), scan(path=2), True),
    ('scan', 'sn_s_surface_f1', "S-surface: Smooth / Distance (F1)",
     "F1 frozen, Raster path. A Smooth, B Distance, width 1.5 on both.",
     'F1', scan(surface=0, path=2), scan(surface=1, path=2), True),
    ('scan', 'sn_s_surface_f2', "S-surface: Smooth / Distance (F2)",
     "F2 = F1 transposed (same cell count), same settings as the F1 pair: a field change.",
     'F2', scan(surface=0, path=2), scan(surface=1, path=2), True),
    ('scan', 'sn_s_laplace', "S-Laplace: Laplacian / Scan Smooth Raster",
     "F3 Pulsar evolving at 2 steps/s. A Laplacian (n=12, shape=1, harm=0, fullshape=1), "
     "B Scan Smooth Raster width 1.5.",
     'F3', laplace(), scan(surface=0, path=2), False),
    ('scan', 'sn_s_laplace_distance', "S-Laplace (Distance): Laplacian / Scan Distance Raster",
     "Named extra: as S-Laplace, B reads the Distance surface.",
     'F3', laplace(), scan(surface=1, path=2), False),
    ('scan', 'sn_s_laplace_harm1', "S-Laplace (harm=1): Laplacian harm=1 / Scan Smooth Raster",
     "Harmonicity control: A Laplacian with harm=1 (integer ratios), B as in S-Laplace.",
     'F3', laplace(harm=1), scan(surface=0, path=2), False),
    ('network', 'sn_n_links_top', "N-links: independent / connected (top)",
     "F5 top half live, frozen CA. A coupling 0 (four independent generators), "
     "B coupling 2; both live links.",
     'F5T', network(coupling=0.0), network(coupling=2.0), True),
    ('network', 'sn_n_links_bottom', "N-links: independent / connected (bottom)",
     "F5 bottom half live: same settings and cell count as 'top', other link weights.",
     'F5B', network(coupling=0.0), network(coupling=2.0), True),
    ('network', 'sn_n_field_left', "N-field: Pond left (6,6)",
     "Auxiliary control: one Pond at (6,6), A coupling 0 / B coupling 2. "
     "A equals the 'right' record after the gate; B may differ only slightly.",
     'F4L', network(coupling=0.0), network(coupling=2.0), True),
    ('network', 'sn_n_field_right', "N-field: Pond right (6,21)",
     "Auxiliary control: the same Pond at (6,21) (equal cell count), same settings.",
     'F4R', network(coupling=0.0), network(coupling=2.0), True),
    ('network', 'sn_n_frozen', "N-frozen: live / frozen links",
     "F3 Pulsar evolving at 2 steps/s, coupling 2 on both. A live links (the field updates W), "
     "B frozen links (W of the initial field held). Initial W equal.",
     'F3', network(coupling=2.0, freeze_links=0), network(coupling=2.0, freeze_links=1), False),
    ('network', 'sn_n_frozen_kok', "N-frozen (Kok's galaxy): live / frozen links",
     "Reserve case of the REQ: Kok's galaxy (period 8) centred in the field, coupling 2, "
     "A live / B frozen links. The Pulsar moves W by at most 0.017; the galaxy by 0.025 "
     "(scale 0..1/3): neither library oscillator modulates W strongly under the six masks.",
     'KOK', network(coupling=2.0, freeze_links=0), network(coupling=2.0, freeze_links=1), False),
    ('network', 'sn_n_laplace', "N-Laplace: Laplacian / Network",
     "F3 Pulsar evolving at 2 steps/s. A Laplacian (n=12, shape=1, harm=0, fullshape=1), "
     "B Network coupling 2, live links.",
     'F3', laplace(), network(coupling=2.0), False),
    ('network', 'sn_n_laplace_harm1', "N-Laplace (harm=1): Laplacian harm=1 / Network",
     "Harmonicity control: A Laplacian with harm=1, B as in N-Laplace.",
     'F3', laplace(harm=1), network(coupling=2.0), False),
]

LISTEN = {
    'scan': "1 / 2 = listen A / B. Space pauses the CA; paint cells inside the path.",
    'network': "1 / 2 = listen A / B. Space pauses the CA; paint inside the six link marks.",
}


def scene_for(rec):
    cat, sid, title, note, field, A, B, static = rec
    return scene_doc(sid, title, field, A, B, RATE_STATIC if static else RATE_EVOLVE,
                     LISTEN[cat])


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


# -- recording (offline, same Recorder / Cut / Catalog.save path as the live bench) --
def record_offline(catalog, doc, seconds, static, title, note):
    """Drive a DemoRunner block by block (no device, no wall clock) with the
    journal 'start' [+ 'pause on'] at block 0, feed the same Recorder the live
    bench uses, cut at the last block boundary and save."""
    runner = DemoRunner(scene_from_doc(doc))
    rec = Recorder(runner, None)
    runner.post('start', at=0)
    if static:
        runner.post('pause', at=0, on=True)
    n_blocks = int(np.ceil(seconds * SR / BLOCK))
    clip = {'A': 0, 'B': 0}
    for _ in range(n_blocks):
        before = runner.out_samples
        blk = runner.next_block()
        rec.on_block(blk, runner.running, before)
    snap = runner.snapshot()
    clip = dict(snap['clip_blocks'])
    diagnostics = dict(device_ok=False, device_error='offline build', underruns=0,
                       block_errors=0, last_error=None, clip_blocks=clip, record_error=None)
    cut = rec.cut(diagnostics)
    assert cut is not None and cut.n_frames == n_blocks * BLOCK
    rid = catalog.save(cut, title, note)
    return rid, snap


def build(root):
    if os.path.exists(root):
        raise SystemExit(f"error: target catalog already exists: {root}\n"
                         f"(rebuild into a new directory with --root, or remove it yourself; "
                         f"records added by the user are never overwritten)")
    cats = {}
    for name in SUBCATALOGS:
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, '.tmp'))
        cats[name] = Catalog(d)
    # the catalog lists newest first: save in reverse so that the order above
    # is the order on screen (ids carry a 1 s timestamp -> wait between saves)
    results = []
    for rec in reversed(RECORDS):
        cat, sid, title, note, field, A, B, static = rec
        doc = scene_for(rec)
        secs = SEC_STATIC if static else SEC_EVOLVE
        t0 = time.time()
        rid, snap = record_offline(cats[cat], doc, secs, static, title, note)
        r = cats[cat].load(rid)
        results.append((cat, rid, title, r.version_label(), snap['clip_blocks'],
                        snap['running'], snap['paused'], snap['gen']))
        print(f"  [{cat:7s}] {rid}  {title}  ({secs:.0f} s, {r.version_label()}, "
              f"clip {snap['clip_blocks']}, running={snap['running']} paused={snap['paused']} "
              f"gen={snap['gen']})")
        wait = 1.05 - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the S/N demo scenes and catalog")
    ap.add_argument('--root', default=CATALOG_ROOT, help="catalog root (must not exist)")
    ap.add_argument('--scenes-only', action='store_true')
    a = ap.parse_args(argv)
    paths = write_scenes()
    print(f"scenes: {len(paths)} files in {DEMOS_DIR} (sn_*.json)")
    if a.scenes_only:
        return 0
    root = os.path.abspath(a.root)
    print(f"catalog: {root}")
    results = build(root)
    bad = [r for r in results if r[4]['A'] or r[4]['B']]
    print(f"done: {len(results)} records; clipped records: {len(bad)}")
    print("open:  run_scan_demo.bat  /  run_network_demo.bat")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
