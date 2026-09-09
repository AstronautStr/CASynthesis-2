#!/usr/bin/env python3
"""S2 audio reference for the demo bench (recorded BEFORE the S3 engine-interface
refactor, from the accepted S2 code).

For each of the five engines on side B of demos/laplace_ab.json a fixed journal
(painting, a parameter change, transport, volume, monitor switches) is rendered
offline and the sha256 of raw A, raw B and monitor is stored in
demo_lab_s2_ref.json.  tests/test_demo_lab.py compares against it byte-exact.

    python tests/golden/demo_lab_s2_ref.py --save   # bless (ONLY from accepted code)
    python tests/golden/demo_lab_s2_ref.py          # compare
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from casynth_config import SR                                        # noqa: E402
from casynth_lab import load_scene, render_offline                   # noqa: E402

REF = os.path.join(_HERE, "demo_lab_s2_ref.json")
SCENE = os.path.join(_ROOT, "demos", "laplace_ab.json")
ENGINE_IDS = ('laplacian', 'fft2d', 'walsh', 'random', 'granulo')
SECONDS = 4.0


def _sec(s):
    return int(round(s * SR))


def journal(engine_id):
    return [
        ('start', 0, {}),
        ('set_engine', 0, {'side': 'B', 'engine_id': engine_id}),
        ('select', _sec(0.5), {'side': 'B'}),
        ('set_param', _sec(1.0), {'side': 'B', 'name': 'n', 'value': 10}),
        ('set_cell', _sec(1.5), {'r': 3, 'c': 3, 'v': 1}),
        ('set_cell', _sec(1.5), {'r': 3, 'c': 4, 'v': 1}),
        ('pause', _sec(2.0), {'on': True}),
        ('set_cell', _sec(2.3), {'r': 14, 'c': 15, 'v': 0}),
        ('pause', _sec(2.6), {'on': False}),
        ('vol', _sec(3.0), {'value': 0.5}),
        ('select', _sec(3.4), {'side': 'A'}),
        ('copy_side', _sec(3.7), {'src': 'B', 'dst': 'A'}),
    ]


def compute():
    scene = load_scene(SCENE)
    out = {}
    for eid in ENGINE_IDS:
        res, r = render_offline(scene, SECONDS, commands=journal(eid),
                                output=('A', 'B', 'monitor'))
        out[eid] = {o: hashlib.sha256(res[o].tobytes()).hexdigest()[:16]
                    for o in ('A', 'B', 'monitor')}
        out[eid]['gen'] = r.gen
        out[eid]['cells'] = int(r.grid.sum())
    return out


def main():
    got = compute()
    if '--save' in sys.argv:
        with open(REF, 'w') as f:
            json.dump(got, f, indent=1, sort_keys=True)
        print(f"[blessed] {REF}")
        return 0
    with open(REF) as f:
        ref = json.load(f)
    ok = (got == ref)
    for eid in ENGINE_IDS:
        print(f"  {'PASS' if got[eid] == ref.get(eid) else 'FAIL'}  {eid}  {got[eid]}")
    print("[PASS] demo lab S2 reference" if ok else "[FAIL] demo lab S2 reference")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
