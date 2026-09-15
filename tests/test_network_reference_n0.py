"""N0 gates: vectorised network == scalar node port (bit-exact), determinism, chunk invariance,
state round-trip, fdlibm atan, source manifest.  stdlib runner (no pytest here)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from demos.network_reference_n0 import gutter_network as gn  # noqa: E402
from demos.network_reference_n0 import jmath, source_manifest as sm  # noqa: E402
from demos.network_reference_n0.gutter_node import GutterOsc  # noqa: E402

SR = 44100.0


def _scalar_run(cfg, node, controls, n):
    """Feed one node of the config through the scalar port with the SAME float32 inlet values."""
    osc = GutterOsc()
    osc.setLowpass(gn.map_soften(cfg["soften_raw"][node]))
    osc.setHighpass(cfg["highpass_hz"])
    osc.filters(cfg["filter_count"][node])
    for i, f in enumerate(cfg["filters_hz"][node]):
        osc.setFreqN(i, min(f * cfg["pitch_shift"], 19000.0))
    for i in range(24):
        osc.setQN(i, gn.map_q(cfg["q_raw"][node]))
    osc.setDistortionMethod(cfg["dist_method"])
    osc.dspsetup(SR)
    o0 = np.zeros(n); dx = np.zeros(n)
    for i in range(n):
        g, w, c, dt, gain = controls[i]
        o0[i], _ = osc.perform_sample(g, w, c, dt, gain, 0.0)
        dx[i] = osc.duffX                     # double state (out1 would be the float32 cast)
    return o0, dx, osc


class TestNodeEquivalence(unittest.TestCase):
    def test_vector_engine_matches_scalar_port_bit_exact(self):
        """8 nodes, links ON, 3000 samples: the vectorised node core must equal the scalar port
        sample-for-sample when driven by the same float32 inlet values (recorded from the engine)."""
        cfg = gn.default_config()
        net = gn.GutterNetwork(cfg)
        n = 3000
        controls = {j: [] for j in range(net.n)}
        duffX = np.zeros((n, net.n))
        for i in range(n):
            net.step()
            g, w, c, dt, gain = net.last_inlets
            for j in range(net.n):
                controls[j].append((float(g[j]), float(w), float(c[j]), float(dt[j]), float(gain[j])))
            duffX[i] = net.duffX
        self.assertGreater(np.abs(duffX).max(), 0.0)
        for j in range(net.n):
            o0, o1, osc = _scalar_run(cfg, j, controls[j], n)
            np.testing.assert_array_equal(o1, duffX[:, j], err_msg=f"node {j} duffX differs")
            self.assertEqual(osc.duffY, net.duffY[j])
            self.assertEqual(osc.t, net.t[j])

    def test_links_reach_damping(self):
        cfg = gn.default_config()
        net = gn.GutterNetwork(cfg)
        net.render(cfg["matrix_delay_samples"] + cfg["extra_feedback_samples"] + 200)
        self.assertGreater(np.abs(net.ring).max(), 0.0)


class TestReproducibility(unittest.TestCase):
    def test_two_runs_identical_and_chunk_invariant(self):
        cfg = gn.default_config()
        a, _ = gn.GutterNetwork(cfg).render(2500)
        net = gn.GutterNetwork(cfg)
        parts = [net.render(700)[0], net.render(1)[0], net.render(1799)[0]]
        b = np.concatenate(parts)
        np.testing.assert_array_equal(a, b)

    def test_state_roundtrip_continues_bit_exact(self):
        cfg = gn.default_config()
        net = gn.GutterNetwork(cfg)
        net.set_slider("mod", 120)
        net.render(1500)
        st = net.export_state()
        st = {k: np.array(v) for k, v in st.items()}
        cont, _ = gn.GutterNetwork.from_state(cfg, st).render(1200)
        ref, _ = net.render(1200)
        np.testing.assert_array_equal(cont, ref)

    def test_no_nan_in_base_config(self):
        cfg = gn.default_config()
        net = gn.GutterNetwork(cfg)
        out, _ = net.render(4000)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(int(net.resets.sum()), 0)


class TestMath(unittest.TestCase):
    def test_fdlibm_atan_scalar_equals_vector_and_within_1ulp_of_libm(self):
        import math
        rng = np.random.default_rng(3)
        xs = np.concatenate([rng.standard_normal(20000) * 3, rng.standard_normal(500) * 1e-9, [0.0, 0.4375, 0.6875, 1.1875, 2.4375, 1e30, -1e30]])
        a = np.array([jmath.atan(float(x)) for x in xs])
        b = jmath.atan_np(xs)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, np.array([jmath.atan_fast(float(x)) for x in xs]))
        np.testing.assert_array_equal(a, jmath.atan_vec(xs))
        c = np.array([math.atan(float(x)) for x in xs])
        self.assertLessEqual(int(np.abs(a.view(np.int64) - c.view(np.int64)).max()), 1)

    def test_slider_mappings_at_restore_values(self):
        self.assertAlmostEqual(gn.map_gain(162), 2.2148, 3)
        self.assertAlmostEqual(gn.map_damp(138), 0.2906, 3)
        self.assertAlmostEqual(gn.map_mod(46), 0.3229, 3)
        self.assertAlmostEqual(gn.map_rate(39), 0.002693, 5)
        self.assertAlmostEqual(gn.map_q(149), 135.83, 1)
        self.assertAlmostEqual(gn.map_soften(103), 6036.7, 0)
        self.assertAlmostEqual(gn.map_interaction(127), 1.2306, 3)


class TestManifest(unittest.TestCase):
    def test_preset_banks_parse(self):
        banks = sm.preset_filter_banks()
        self.assertEqual(len(banks), 20)
        self.assertEqual(len(banks[5]), 24)
        self.assertEqual(banks[0][0], 97.0)

    def test_random_bank_in_source_range(self):
        rng = np.random.default_rng(0)
        f = sm.random_filter_bank(rng)
        self.assertEqual(len(f), 24)
        self.assertTrue(all(50.0 <= x <= 5000.0 for x in f))


if __name__ == "__main__":
    unittest.main(verbosity=1)
