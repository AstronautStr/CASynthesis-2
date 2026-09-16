#!/usr/bin/env python3
"""N2 gates (REQ memory/req-network-events-n2-2026-09-16.md, "Проверки до передачи"):
the shared periodic readout, side A (gutter_field_periodic) against gutter_field,
side B (ca_event_network) against an independent scalar reference, event
semantics, causality through the delays, snapshot / continue, limits, timing,
the three scenes and the prepared catalog.

    python tests/test_n2_events.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_n2_tests/.
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN                                    # noqa: E402
from casynth_lab import BLOCK, registry                                       # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot          # noqa: E402
from casynth_lab.snapshot import save_state, load_state                      # noqa: E402
from casynth_lab import periodic_readout as pr                               # noqa: E402
from casynth_lab import gutter_field as gf                                   # noqa: E402
from casynth_lab import gutter_field_periodic as gp                          # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_n2_tests')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7
GUTTER_PARAMS = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r % rows, c % cols] = 1
    return g


def ref_weights(rows, cols):
    """The REQ formulas written out per cell (independent of periodic_readout)."""
    K = np.zeros((8, rows, cols))
    for i in range(8):
        j, l = i % 4, i // 4
        ax, ay = (j + 0.5) / 4, (l + 0.5) / 2
        for r in range(rows):
            for c in range(cols):
                x, y = (c + 0.5) / cols, (r + 0.5) / rows
                dx = ((x - ax + 0.5) % 1.0) - 0.5
                dy = ((y - ay + 0.5) % 1.0) - 0.5
                hx = (1 + math.cos(math.pi * dx / 0.25)) / 2 if abs(dx) <= 0.25 else 0.0
                hy = (1 + math.cos(math.pi * dy / 0.5)) / 2 if abs(dy) <= 0.5 else 0.0
                K[i, r, c] = hx * hy
    return K


class PeriodicReadoutTests(unittest.TestCase):
    def test_partition_of_unity_and_formulas(self):
        for rows, cols in ((32, 32), (16, 24), (7, 9)):
            K = pr.weights(rows, cols)
            self.assertEqual(K.shape, (8, rows, cols))
            self.assertTrue(np.all(K >= 0.0))
            self.assertLess(np.abs(K.sum(0) - 1.0).max(), 1e-12)
            self.assertLess(np.abs(K - ref_weights(rows, cols)).max(), 1e-12)

    def test_periodic_on_the_torus(self):
        K = pr.weights(32, 32)
        # a shift by one node spacing (8 columns / 16 rows) maps node i onto its neighbour
        for i in range(8):
            j, l = i % 4, i // 4
            right = 4 * l + (j + 1) % 4
            self.assertLess(np.abs(np.roll(K[i], 8, axis=1) - K[right]).max(), 1e-12)
            down = 4 * ((l + 1) % 2) + j
            self.assertLess(np.abs(np.roll(K[i], 16, axis=0) - K[down]).max(), 1e-12)
        # continuity across the outer edge: the step between columns 31 and 0 is the
        # same size as any interior step of the same weight
        steps = np.abs(np.diff(np.concatenate([K[0, 16], K[0, 16, :1]])))
        self.assertLess(steps[-1], steps.max() + 1e-12)
        self.assertGreater(K[0, 16, 0], 0.0)          # node 0 (centre column 3.5) reaches column 0
        self.assertGreater(K[3, 16, 0], 0.0)          # ... and so does node 3 across the edge

    def test_weighted_counts_and_centres(self):
        K = pr.weights(32, 32)
        g = grid([(16, 3), (16, 4), (0, 31)])
        n = pr.weighted_counts(K, g)
        self.assertLess(abs(n.sum() - 3.0), 1e-12)
        self.assertGreater(n[0], n[1])
        self.assertEqual(pr.centre_cells(32, 32)[0], (3.5, 7.5))
        self.assertEqual(pr.centre_cells(32, 32)[7], (27.5, 23.5))
        with self.assertRaises(ValueError):
            pr.weighted_counts(K, np.zeros((16, 16)))


class GutterPeriodicTests(unittest.TestCase):
    def _pair(self, g, **over):
        params = dict(GUTTER_PARAMS)
        params.update(over)
        a = registry.create(gp.ENGINE_ID, CTX, params)
        b = registry.create(gf.ENGINE_ID, CTX, params)
        a.init(g, None, GAIN)
        b.init(g, None, GAIN)
        return a, b

    def test_same_model_as_gutter_field_only_the_readout_and_trim_differ(self):
        # empty field: every count is 0 on both readouts -> identical messages, identical raw
        a, b = self._pair(np.zeros((32, 32), np.uint8))
        self.assertEqual(a.model_version, gp.MODEL_VERSION)
        self.assertNotEqual(a.model_version, b.model_version)
        for _ in range(6):
            np.testing.assert_array_equal(a.raw_block(), b.raw_block())
        ya, _, _ = a.render_float(GAIN)
        yb, _, _ = b.render_float(GAIN)
        self.assertGreater(np.abs(yb).max(), 0.0)
        self.assertLess(np.abs(ya - yb * gp.A_TRIM).max(), 1e-15)
        self.assertAlmostEqual(20 * math.log10(gp.A_TRIM), -16.5, places=12)

    def test_weighted_counts_drive_the_frequencies(self):
        g = grid([(9, 11), (9, 12), (9, 13), (25, 20)])
        a, b = self._pair(g)
        K = ref_weights(32, 32)
        want = np.tensordot(K, g.astype(float), axes=([1, 2], [0, 1]))
        self.assertLess(np.abs(a.counts - want).max(), 1e-12)
        self.assertLess(np.abs(a.u_field - want / (want + 2.0)).max(), 1e-12)
        ratio = np.power(2.0, np.clip(2.0 * a.u_field - 1.0, -1.0, 1.0))
        base = np.array(a.cfg['filters_hz'])
        np.testing.assert_array_equal(a.ffreq, np.minimum(base * ratio[:, None], 19000.0)
                                      .astype(np.float32).astype(np.float64))
        # a cell moved INSIDE one N1 region: N1 does not see it, the periodic readout does
        moved = grid([(9, 11), (9, 12), (9, 14), (25, 20)])
        a.update_field(moved, None)
        b.update_field(moved, None)
        self.assertLess(np.abs(b.u_field - gf.field_u(gf.region_counts(g))).max(), 1e-15)
        self.assertNotEqual(list(a.counts), list(want))
        self.assertEqual(a.display()['readout'], 'periodic')
        self.assertIsInstance(a.display()['counts'][0], float)

    def test_snapshot_roundtrip_and_registry(self):
        spec = registry.get(gp.ENGINE_ID)
        self.assertEqual([p[0] for p in spec.params], [p[0] for p in gf.PARAMS])
        ov = spec.overlay(GUTTER_PARAMS, 32, 32)
        self.assertEqual(len(ov['labels']), 8)
        self.assertNotIn('lines', ov)
        g = grid([(9, 11), (9, 12), (9, 13)])
        a, _ = self._pair(g)
        self.assertTrue(supports_snapshot(a))
        for _ in range(3):
            a.render(GAIN, 0)
        a.set_params(dict(GUTTER_PARAMS, interaction=200))       # in-flight 50 ms ramp
        a.render(GAIN, 0)
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 's', a.export_state())
            st = load_state(tmp, 's')
        r = registry.create(gp.ENGINE_ID, CTX, GUTTER_PARAMS)
        r.restore_state(g, None, st)
        for _ in range(8):
            np.testing.assert_array_equal(a.render(GAIN, 0)[0], r.render(GAIN, 0)[0])
        bad = dict(st, model_version=gf.MODEL_VERSION)
        with self.assertRaises(ValueError):
            registry.create(gp.ENGINE_ID, CTX, GUTTER_PARAMS).restore_state(g, None, bad)


if __name__ == '__main__':
    unittest.main()
