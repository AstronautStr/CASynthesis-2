#!/usr/bin/env python3
"""The seam: what the prototype and the bench now share (REQ req-seam-2026-09-21).

A1  the engines are a package a host can import without the bench;
A2  the SoundEngine contract carries a note, a glide start and live envelopes;
A3  one ring and one callback (casynth_host) feed both hosts;
A4  one decision (casynth_panel) builds both panels' rows;
A5  a scene carries the note, the gate, the volume and the envelopes, so a
    prototype session can be re-rendered OFFLINE -- the comparison a CPU dip
    cannot touch.

Run: python tests/test_seam.py        (stdlib runner; pytest is not installed)
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_config import (SR, CHUNK_S, VOICE_ATTACK_MS_DEFAULT,        # noqa: E402
                            VOICE_DECAY_MS_DEFAULT, VOICE_SUSTAIN_DEFAULT,
                            VOICE_RELEASE_MS_DEFAULT, GEN_ATTACK_DEFAULT,
                            GEN_DECAY_DEFAULT, GEN_SUSTAIN_DEFAULT,
                            GEN_RELEASE_DEFAULT)
from casynth_engine import VoiceEnvelope, midi_to_freq                    # noqa: E402
from casynth_host import AudioHost, BlockRing                             # noqa: E402
import casynth_panel as panel                                            # noqa: E402
from casynth_engines import registry                                      # noqa: E402
from casynth_engines.engine_api import supports_transpose                 # noqa: E402
from casynth_lab import (DemoRunner, BLOCK, scene_from_doc, render_offline,  # noqa: E402
                         SceneError)

BLOCKS = 40


def base_scene(**audio):
    a = dict(f0_hz=110.0, level=1.0)
    a.update(audio)
    cells = [[5, 5], [5, 6], [5, 7], [10, 10], [10, 11], [11, 10], [11, 11]]
    return {
        'format': 2, 'id': 'seam_test', 'title': 'seam',
        'grid': {'rows': 32, 'cols': 32}, 'cells': cells,
        'rule': 'B3/S23', 'boundary': 'torus', 'rate_hz': 4.0,
        'audio': a,
        'variants': {s: {'engine_id': 'laplacian',
                         'engine_params': dict(registry.defaults('laplacian'))}
                     for s in ('A', 'B')},
    }


def render(doc, blocks=BLOCKS, commands=None):
    scene = scene_from_doc(doc)
    pcm, _r = render_offline(scene, blocks * BLOCK / float(SR),
                             commands=commands or [('start', 0, {})], output='A')
    return pcm


# -- A2 / A4 / A3: the shared pieces -------------------------------------------------
class SharedPieces(unittest.TestCase):
    def test_every_engine_answers_the_contract(self):
        """Every registered engine takes the contract's render() and either plays a
        note or refuses one -- it never renders a wrong pitch silently."""
        from casynth_engines import EngineContext
        ctx = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 4.0)
        g = np.zeros((32, 32), np.uint8)
        g[5, 5:8] = 1
        g[10:14, 10:14] = 1
        for eid in registry.ids():
            e = registry.create(eid, ctx, registry.defaults(eid))
            e.init(g, None, 0.0)
            buf, _pk, _nc = e.render(0.04, 0, gain_prev=0.0)
            self.assertEqual(buf.shape, (BLOCK, 2), eid)
            self.assertEqual(buf.dtype, np.int16, eid)
            if supports_transpose(e):
                e.render(0.04, BLOCK, transpose=1.5)
            else:
                with self.assertRaises(ValueError, msg=eid):
                    e.render(0.04, BLOCK, transpose=1.5)

    def test_importing_the_engines_does_not_drag_the_bench_in(self):
        """A1: a host imports casynth_engines without numba or the resonator bank."""
        import subprocess
        code = ("import sys, casynth_engines;"
                "print('numba' in sys.modules,"
                "'casynth_engines.object_resonators' in sys.modules,"
                "'casynth_lab.runner' in sys.modules)")
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ['False', 'False', 'False'], out.stdout)

    def test_the_ring_fills_across_block_boundaries_and_counts_a_dry_read(self):
        """A3: the callback's view of the ring -- what both hosts used to copy."""
        ring = BlockRing(4, 2)
        ring.put(np.arange(8, dtype=np.int16).reshape(4, 2))
        ring.put(np.arange(100, 108, dtype=np.int16).reshape(4, 2))
        out = np.zeros((6, 2), np.int16)
        ring.fill(out, 6)
        self.assertEqual(out[:, 0].tolist(), [0, 2, 4, 6, 100, 102])
        self.assertEqual(ring.underruns, 0)
        ring.fill(out, 6)                     # 2 samples left, then dry
        self.assertEqual(ring.underruns, 1)
        self.assertEqual(out[2:, 0].tolist(), [0, 0, 0, 0])

    def test_a_host_without_a_device_says_so(self):
        def broken(_cb):
            raise RuntimeError("no device here")
        host = AudioHost(BLOCK, 2, output_factory=broken)
        self.assertFalse(host.start())
        self.assertIn("no device here", host.device_error)
        host.stop()

    def test_one_decision_builds_both_panels(self):
        """A4: the widget kinds come from casynth_panel, not from two copies."""
        spec = registry.get('laplacian')
        rows = {r.arg: r for r in panel.panel_rows(spec, registry.defaults('laplacian'))}
        self.assertEqual(rows['fullshape'].kind, panel.KIND_TOGGLE)
        self.assertEqual(rows['n'].kind, panel.KIND_SLIDER)
        spec = registry.get('ca_object_resonators')
        params = registry.defaults('ca_object_resonators')
        rows = {r.arg: r for r in panel.panel_rows(spec, params)}
        self.assertEqual(rows['decay_law'].kind, panel.KIND_CHOICES)
        self.assertEqual(rows['decay_law'].choices, spec.choices['decay_law'])
        inactive = {k: v for k, v in (spec.inactive(params) or {}).items()}
        for name, text in inactive.items():
            self.assertEqual(rows[name].kind, panel.KIND_INACTIVE)
            self.assertEqual(rows[name].text, text)
        # the settings engines share come first in the prototype's reading order
        order = panel.shared_first(spec, registry.SPECTRUM_KEYS)
        self.assertEqual(order[:len([k for k in registry.SPECTRUM_KEYS
                                     if k in {p[0] for p in spec.params}])],
                         [k for k in registry.SPECTRUM_KEYS
                          if k in {p[0] for p in spec.params}])


# -- A5: the scene articulates -------------------------------------------------------
class SceneArticulation(unittest.TestCase):
    def test_a_scene_without_articulation_renders_exactly_as_before(self):
        plain = render(base_scene())
        with_defaults = render(base_scene(
            note=None, gate=True,
            envelope=dict(voice=dict(attack_ms=VOICE_ATTACK_MS_DEFAULT,
                                     decay_ms=VOICE_DECAY_MS_DEFAULT,
                                     sustain=VOICE_SUSTAIN_DEFAULT,
                                     release_ms=VOICE_RELEASE_MS_DEFAULT),
                          gen=dict(attack=GEN_ATTACK_DEFAULT, decay=GEN_DECAY_DEFAULT,
                                   sustain=GEN_SUSTAIN_DEFAULT, release=GEN_RELEASE_DEFAULT,
                                   amp_slew=False))))
        np.testing.assert_array_equal(plain, with_defaults)

    def test_a_note_at_the_anchor_is_the_same_sound_as_no_note(self):
        """transpose == 1.0 must be bit-exact, whichever way it is reached."""
        anchor = base_scene()
        anchor['audio']['f0_hz'] = float(midi_to_freq(48))
        plain = render(anchor)
        noted = dict(anchor)
        noted['audio'] = dict(anchor['audio'], note=48)
        np.testing.assert_array_equal(plain, render(noted))

    def test_a_note_transposes_and_a_higher_note_sounds_higher(self):
        low = render(base_scene(f0_hz=float(midi_to_freq(48)), note=48))
        high = render(base_scene(f0_hz=float(midi_to_freq(48)), note=60))
        self.assertFalse(np.array_equal(low, high))
        centroid = []
        for pcm in (low, high):
            x = pcm[:, 0].astype(float)
            mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            f = np.fft.rfftfreq(len(x), 1.0 / SR)
            centroid.append(float((mag * f).sum() / max(mag.sum(), 1e-9)))
        self.assertGreater(centroid[1], centroid[0] * 1.5)

    def test_a_closed_gate_is_silent_and_the_script_opens_it(self):
        shut = render(base_scene(note=48, gate=False,
                                 envelope=dict(voice=dict(attack_ms=0.0, decay_ms=0.0,
                                                          sustain=1.0, release_ms=0.0))))
        self.assertEqual(int(np.abs(shut).max()), 0)
        doc = base_scene(note=48, gate=False,
                         envelope=dict(voice=dict(attack_ms=0.0, decay_ms=0.0,
                                                  sustain=1.0, release_ms=0.0)))
        doc['script'] = [dict(at=10 * BLOCK, kind='gate', args=dict(on=True))]
        opened = render(doc)
        self.assertEqual(int(np.abs(opened[:10 * BLOCK]).max()), 0)
        self.assertGreater(int(np.abs(opened[10 * BLOCK:]).max()), 0)

    def test_the_script_can_play_a_phrase_set_cells_and_move_the_volume(self):
        doc = base_scene(note=48)
        doc['script'] = [
            dict(at=5 * BLOCK, kind='note', args=dict(note=55)),
            dict(at=9 * BLOCK, kind='vol', args=dict(value=0.2)),
            dict(at=12 * BLOCK, kind='set_cells', args=dict(cells=[[20, 20, 1], [20, 21, 1],
                                             [21, 20, 1], [21, 21, 1]])),
        ]
        scene = scene_from_doc(doc)
        self.assertEqual(len(scene.script), 3)
        runner = DemoRunner(scene)
        runner.post('start', at=0)
        for _ in range(BLOCKS):
            runner.next_block()
        self.assertEqual(runner.note, 55)
        self.assertAlmostEqual(runner.vol, 0.2)
        self.assertEqual(int(runner.grid[20, 20]), 1)

    def test_the_scene_refuses_what_it_cannot_express(self):
        for audio, why in ((dict(note=200), 'note out of range'),
                           (dict(note=1.5), 'note not an int'),
                           (dict(gate='yes'), 'gate not a bool'),
                           (dict(envelope=dict(voice=dict(attack_ms=0.0))), 'partial voice'),
                           (dict(envelope=dict(gen=dict(attack=0.1))), 'partial gen')):
            with self.assertRaises(SceneError, msg=why):
                scene_from_doc(base_scene(**audio))
        for script, why in (([dict(at=BLOCK, kind='note', args=dict(note=-1))], 'bad note'),
                            ([dict(at=BLOCK, kind='vol', args=dict(value=2.0))], 'bad vol'),
                            ([dict(at=BLOCK, kind='set_cells',
                                   args=dict(cells=[[99, 0, 1]]))], 'cell outside'),
                            ([dict(at=BLOCK, kind='fly', args={})], 'unknown kind')):
            doc = base_scene()
            doc['script'] = script
            with self.assertRaises(SceneError, msg=why):
                scene_from_doc(doc)

    def test_the_pan_mode_is_the_hosts_choice_and_center_is_the_default(self):
        """The bench renders both channels identical (its rule since S1); the
        prototype puts a voice where its figure is.  The default keeps every
        scene and record made so far byte-for-byte."""
        centred = render(base_scene())
        np.testing.assert_array_equal(centred[:, 0], centred[:, 1])
        np.testing.assert_array_equal(centred, render(base_scene(pan='center')))
        spread = render(base_scene(pan='field'))
        self.assertFalse(np.array_equal(spread[:, 0], spread[:, 1]))
        with self.assertRaises(SceneError):
            scene_from_doc(base_scene(pan='left'))

    def test_an_articulated_scene_survives_a_snapshot(self):
        doc = base_scene(note=48)
        doc['script'] = [dict(at=4 * BLOCK, kind='note', args=dict(note=53))]
        scene = scene_from_doc(doc)
        runner = DemoRunner(scene)
        runner.post('start', at=0)
        for _ in range(12):
            runner.next_block()
        state = runner.export_state()
        self.assertEqual(state['note'], 53)
        cont = DemoRunner.from_state(state)
        self.assertEqual(cont.note, 53)
        for _ in range(8):
            np.testing.assert_array_equal(cont.next_block().A, runner.next_block().A)


# -- A5: the VCA is one class, not two copies ----------------------------------------
def _inline_vca(gates, va, vd, vs, vr, gate0=True):
    """The arithmetic exactly as gol_synth._render_loop carried it until
    2026-09-21 -- the reference VoiceEnvelope must reproduce."""
    venv = {'level': (1.0 if gate0 else 0.0), 'phase': (3 if gate0 else 0), 'rel0': 0.0}
    last_gate = gate0
    out = []
    for gate in gates:
        if gate and not last_gate:
            venv['phase'] = 1
            if va <= 0.0:
                venv['level'], venv['phase'] = 1.0, 2
        elif last_gate and not gate:
            venv['phase'] = 4
            venv['rel0'] = venv['level']
            if vr <= 0.0:
                venv['level'], venv['phase'] = 0.0, 0
        ph = venv['phase']
        if ph == 1:
            venv['level'] += CHUNK_S / va
            if venv['level'] >= 1.0:
                venv['level'], venv['phase'] = 1.0, 2
        elif ph == 2:
            if vd <= 0.0:
                venv['level'], venv['phase'] = vs, 3
            else:
                venv['level'] -= (1.0 - vs) * CHUNK_S / vd
                if venv['level'] <= vs:
                    venv['level'], venv['phase'] = vs, 3
        elif ph == 3:
            venv['level'] = vs
        elif ph == 4:
            venv['level'] -= venv['rel0'] * CHUNK_S / vr
            if venv['level'] <= 0.0:
                venv['level'], venv['phase'] = 0.0, 0
        last_gate = gate
        out.append(venv['level'])
    return out


class VoiceEnvelopeIsOneLaw(unittest.TestCase):
    def test_the_class_reproduces_the_arithmetic_it_replaced(self):
        rng = np.random.RandomState(11)
        for (va, vd, vs, vr) in ((0.0, 0.0, 1.0, 0.05),        # the defaults
                                 (0.02, 0.05, 0.3, 0.2),
                                 (0.2, 0.0, 1.0, 1.0),
                                 (0.0, 0.01, 0.0, 0.001)):
            for gate0 in (True, False):
                gates = [bool(b) for b in rng.random_sample(300) > 0.3]
                want = _inline_vca(gates, va, vd, vs, vr, gate0)
                venv = VoiceEnvelope(gate=gate0)
                got = [venv.block(g, va, vd, vs, vr) for g in gates]
                self.assertEqual(got, want, f"A={va} D={vd} S={vs} R={vr} gate0={gate0}")

    def test_the_default_envelope_is_a_no_op_multiplier(self):
        venv = VoiceEnvelope(gate=True)
        for _ in range(50):
            self.assertEqual(venv.block(True, VOICE_ATTACK_MS_DEFAULT / 1000.0,
                                        VOICE_DECAY_MS_DEFAULT / 1000.0,
                                        VOICE_SUSTAIN_DEFAULT,
                                        VOICE_RELEASE_MS_DEFAULT / 1000.0), 1.0)


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in
                               (SharedPieces, SceneArticulation, VoiceEnvelopeIsOneLaw))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    n = res.testsRun
    print(f"\n{n - len(res.failures) - len(res.errors)}/{n} passed")
    return 0 if res.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
