"""Reproduce the revised N1 evidence and verify the actual delivered catalog.

python demos/n1_hypotheses_report.py
No audio device; measurements are technical evidence, not listening verdicts.
"""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_lab import DemoRunner, scene_from_doc, BLOCK
from casynth_lab.catalog import Catalog
from demos.build_n1_hypotheses import CASES, CATALOG_ROOT, scene_for, commands_for
from demos.network_reference_n0.analysis import describe, compare


def audio(runner, seconds):
    out = {s: [] for s in ('A', 'B')}
    times = []
    for _ in range(math.ceil(seconds * 44100 / BLOCK)):
        before = time.perf_counter()
        block = runner.next_block()
        times.append((time.perf_counter() - before) * 1000)
        for side in out:
            out[side].append(block.get(side))
    return {s: np.concatenate(x).astype(float) / 32767 for s, x in out.items()}, times


def probe(case):
    runner = DemoRunner(scene_from_doc(scene_for(case)))
    runner.post('start', at=0)
    for kind, at, args in commands_for(case):
        runner.post(kind, at=at, **args)
    y, times = audio(runner, 48)
    info = dict(scene=scene_for(case), windows={})
    for side in ('A', 'B'):
        info['windows'][side] = []
        for start in (1, 8, 16, 28, 44):
            d, _ = describe(y[side][start * 44100:(start + 3) * 44100].mean(1))
            info['windows'][side].append(dict(start_seconds=start, **d))
    info['ab'], *_ = compare(y['A'][4 * 44100:12 * 44100].mean(1),
                            y['B'][4 * 44100:12 * 44100].mean(1))
    state = runner.export_state()
    edited, control = DemoRunner.from_state(state), DemoRunner.from_state(state)
    edited.post('pause', on=True)
    control.post('pause', on=True)
    rows, cols = runner.grid.shape
    moved = np.roll(runner.grid, rows // 2, axis=0)
    for r in range(rows):
        for c in range(cols):
            if moved[r, c] != runner.grid[r, c]:
                edited.post('set_cell', r=r, c=c, v=int(moved[r, c]))
    ey, _ = audio(edited, 4)
    cy, _ = audio(control, 4)
    info['late_edit'] = {}
    for side in ('A', 'B'):
        assert not np.array_equal(ey[side], cy[side]), (case['id'], side, 'field is inert')
        info['late_edit'][side], *_ = compare(ey[side][44100:].mean(1), cy[side][44100:].mean(1))
    info['clip_blocks'] = runner.snapshot()['clip_blocks']
    info['node_resets'] = {s: int(runner.sides[s].engine.resets.sum()) for s in ('A', 'B')}
    info['block_ms'] = dict(p95=float(np.percentile(times, 95)), p99=float(np.percentile(times, 99)),
                            budget=BLOCK / 44100 * 1000)
    assert not any(info['clip_blocks'].values()) and not any(info['node_resets'].values())
    return info


def verify_catalog():
    catalog = Catalog(str(CATALOG_ROOT))
    checked = []
    for record, error in catalog.list():
        assert not error, error
        result = catalog.replay(record.id, yield_cpu=False)
        assert result.status == 'match', (record.id, result.reason)
        runner, state, _ = catalog.continue_runner(record.id)
        restored = DemoRunner.from_state(state)
        for _ in range(65):
            a, b = runner.next_block(), restored.next_block()
            for side in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(side), b.get(side))
        case = next(c for c in CASES if c['id'] == record.meta['scene']['id'])
        assert record.notes.startswith(case['hypothesis'])
        assert catalog.source_plan(record)['mode'] == 'in-process'
        checked.append(dict(id=record.id, scene=case['id'], replay='match', continue_exact=True,
                            hypothesis_in_notes=True))
    assert len(checked) == len(CASES)
    return checked


def main():
    results = dict(cases={})
    for case in CASES:
        results['cases'][case['id']] = probe(case)
        print(case['id'], 'field response / limits: OK', flush=True)
    results['catalog'] = verify_catalog()
    out = ROOT / 'demos' / 'results' / 'n1_hypotheses'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n',
                                    encoding='utf-8')
    print('All delivered records: exact replay, continuation, hypotheses in Notes', flush=True)


if __name__ == '__main__':
    main()
