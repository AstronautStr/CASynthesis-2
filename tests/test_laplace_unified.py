#!/usr/bin/env python3
"""The unified Laplace engine (REQ memory/req-unified-laplace-2026-09-21.md).

The axes, the four byte anchors, the aliases and the two new cells:

  - the axes: the seven spectral settings first, then artic / voice and the knobs
    each axis brings, with the ranges, defaults and named choices of section 2
  - the anchors: every cell of an OLD engine renders byte for byte what that
    engine renders, over a running field, through a note and through a gain drag
        Env    + Bank + Sine       == `laplacian`
        Env    + Bank + Saw/Square == `laplace_carriers` (Wave bank)
        Env    + FM                == `laplace_fm`
        Events + Bank + Sine       == `ca_object_resonators`
    -- which is the whole proof that collapsing four engines into two axes
    changed no sound: the law of each one is still the gate it always had.
  - the alias map: what an old id's OWN parameter set means on the axes, and
    which parameter sets are outside them (the Filter method, the Objects
    settings section 4 leaves out)
  - switching an axis: never an impulse, never a cut, and never a hole -- the
    cell that was heard rings out while the new one comes in
  - the combinations that are NOT implemented yet are refused out loud
  - the aliases: the factory of every old id resolves through this engine, and
    what comes back is that id's own law with its own parameters and its own
    bytes -- so scenes, snapshots and the records of lab_catalog/ replay
  - the NEW cells: the wave kernel is the law's kernel with one branch changed
    (bit for bit at Sine), one mode sounded as a wave follows c_h = 1/h under
    the band limit of ITS own harmonics, and the FM phase sum is the law of
    req-laplace-fm with an index that decays with the mode

    python tests/test_laplace_unified.py

Stdlib runner (pytest is not installed)."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN                                   # noqa: E402
from casynth_engine import step, events_field                                # noqa: E402
from casynth_engines import registry                                         # noqa: E402
from casynth_engines.engine_api import EngineContext                         # noqa: E402
from casynth_engines import laplace_unified as lu                            # noqa: E402
from casynth_engines import laplace_carriers as lc                           # noqa: E402
from casynth_engines import object_resonators as orz                         # noqa: E402
from casynth_engines import unified_events as uev                            # noqa: E402
from casynth_engines import event_network as en                              # noqa: E402

BLOCK = 352
F0 = 110.0
RATE = 6.0
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE)
GAIN = MASTER_GAIN * 0.35
SPECTRUM = dict(n=12, spread=0.2, alpha=1.0, shape=0.0, harm=0.0, fullshape=1, dyn=0.0)


def scene():
    """A blinker, a 10x10 block and a bar: several figures, several voices, and
    a field that keeps changing under Life."""
    g = np.zeros((48, 64), np.uint8)
    g[5, 5:8] = 1
    g[10:20, 20:30] = 1
    g[30:34, 40:48] = 1
    return g


def uni(**axes):
    p = dict(lu.OPTIONAL_PARAMS)
    p.update(SPECTRUM)
    p.update(axes)
    return lu.UnifiedLaplaceEngine(CTX, p)


def run(engine, blocks=16, gains=None, transpose=None):
    """The same deterministic programme for both sides: Life every other block,
    an optional gain drag and an optional note."""
    g = scene()
    engine.init(g, None, 0.0)
    out = []
    for i in range(blocks):
        if i % 2 == 1:
            g = step(g)
            engine.update_field(g, events_field(g, g))
        gain = GAIN if gains is None else gains[i % len(gains)]
        t = 1.0 if transpose is None else transpose[i % len(transpose)]
        buf, _peak, _clip = engine.render(gain, i * BLOCK, transpose=t)
        out.append(buf.copy())
    return np.concatenate(out)


class Axes(unittest.TestCase):
    """REQ section 2: what the panel offers and in which order."""

    def test_the_seven_spectral_settings_come_first(self):
        names = [p[0] for p in lu.PARAMS]
        self.assertEqual(names[:7], list(lu.SPECTRUM_KEYS))
        self.assertEqual(names[7:], ['artic', 'voice', 'waveform', 'fm_depth',
                                     'events', 'radius_mul', 'decay_s', 'attack_ms'])

    def test_ranges_defaults_and_choices(self):
        by = {p[0]: p for p in lu.PARAMS}
        self.assertEqual(by['artic'][5], lu.ARTIC_ENV)
        self.assertEqual(by['voice'][5], lu.VOICE_BANK)
        self.assertEqual(by['waveform'][5], lu.WF_SINE)
        self.assertEqual(by['fm_depth'][2:6], (0.0, 4.0, False, 1.0))
        self.assertEqual(by['decay_s'][2:6], (0.20, 1.50, False, 0.80))
        self.assertEqual(by['attack_ms'][2:6], (0.0, 20.0, False, 0.0))
        self.assertEqual(by['events'][2:5], (0, 2, True))
        self.assertEqual(lu.RANGES['radius_mul'], (0.25, 4.0))
        self.assertEqual(lu.CHOICES['artic'], ('Env', 'Events'))
        self.assertEqual(lu.CHOICES['voice'], ('Bank', 'FM'))
        self.assertEqual(lu.CHOICES['waveform'], ('Sine', 'Saw', 'Square'))
        # the seven spectral settings keep the ranges and defaults of the old
        # Laplace -- they are not re-declared, they are the registry's own
        self.assertEqual(lu.PARAMS[:7], list(lu.LAPLACE_PARAMS))

    def test_the_panel_is_compact(self):
        """A knob that does not act is not drawn (REQ section 2: 9 rows for
        Env + Bank, 13 for Events + FM)."""
        def rows(**axes):
            p = dict(lu.OPTIONAL_PARAMS)
            p.update(SPECTRUM)
            p.update(axes)
            off = lu.inactive(p)
            return [n for n in (q[0] for q in lu.PARAMS) if n not in off]
        env_bank = rows(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        ev_fm = rows(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_FM)
        self.assertEqual(len(env_bank), 9, env_bank)
        self.assertEqual(len(ev_fm), 13, ev_fm)
        self.assertNotIn('fm_depth', env_bank)
        self.assertNotIn('waveform', ev_fm)
        for name in ('events', 'radius_mul', 'decay_s', 'attack_ms'):
            self.assertNotIn(name, env_bank)
            self.assertIn(name, ev_fm)


class ByteAnchors(unittest.TestCase):
    """REQ section 3: every anchor is a gate that already exists."""

    def _same(self, a, b, what):
        self.assertEqual(a.shape, b.shape, what)
        if not np.array_equal(a, b):
            d = np.abs(a.astype(np.int64) - b.astype(np.int64))
            self.fail(f"{what}: {int(np.count_nonzero(d))} of {d.size} samples differ, "
                      f"max {int(d.max())}")

    def test_env_bank_sine_is_the_baseline(self):
        self._same(run(uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK, waveform=lu.WF_SINE)),
                   run(registry.create('laplacian', CTX, dict(SPECTRUM))),
                   'Env + Bank + Sine vs laplacian')

    def test_env_bank_waves_are_the_wave_bank(self):
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            old = registry.create(lc.ENGINE_ID, CTX,
                                  dict(method=lc.METHOD_BANK, waveform=wf,
                                       filter_width_oct=0.35, filter_depth_db=24.0,
                                       **SPECTRUM))
            self._same(run(uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK, waveform=wf)),
                       run(old), f"Env + Bank + {lc.WAVE_NAMES[wf]} vs laplace_carriers")

    def test_env_fm_is_laplace_fm(self):
        for depth in (0.0, 1.0, 2.5):
            self._same(run(uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_FM, fm_depth=depth)),
                       run(registry.create('laplace_fm', CTX, dict(fm_depth=depth, **SPECTRUM))),
                       f"Env + FM (depth {depth}) vs laplace_fm")

    def _objects(self, **kw):
        p = dict(detector=orz.DET_DISK, radius_mul=1.0, spectrum=orz.SPEC_LAPLACE,
                 events=0, excitation=orz.EXC_UNIFORM, birth_strength=1.0,
                 frequency_scale=220.0, decay_s=0.80, decay_law=orz.LAW_FIXED,
                 attack_ms=0.0, **SPECTRUM)
        p.update(kw)
        return registry.create('ca_object_resonators', CTX, p)

    def test_events_bank_sine_is_objects(self):
        for ev, rad, dec, atk in ((0, 1.0, 0.80, 0.0), (1, 2.0, 0.35, 8.0),
                                  (2, 0.5, 1.40, 20.0)):
            a = run(uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK, waveform=lu.WF_SINE,
                        events=ev, radius_mul=rad, decay_s=dec, attack_ms=atk))
            b = run(self._objects(events=ev, radius_mul=rad, decay_s=dec, attack_ms=atk))
            self._same(a, b, f"Events + Bank + Sine vs Objects (ev={ev}, rad={rad})")

    def test_the_anchors_hold_through_a_note_and_a_gain_drag(self):
        """A host plays: the gain is dragged every block and the note moves."""
        drag = [0.05, 0.4, 0.22, 0.31]
        notes = [1.0, 1.0, 1.25, 1.25, 0.75, 0.75]
        pairs = (
            (uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK, waveform=lu.WF_SINE),
             registry.create('laplacian', CTX, dict(SPECTRUM)), 'laplacian'),
            (uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK, waveform=lc.WF_SAW),
             registry.create(lc.ENGINE_ID, CTX, dict(method=lc.METHOD_BANK, waveform=lc.WF_SAW,
                                                     filter_width_oct=0.35,
                                                     filter_depth_db=24.0, **SPECTRUM)),
             'laplace_carriers'),
            (uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_FM, fm_depth=1.0),
             registry.create('laplace_fm', CTX, dict(fm_depth=1.0, **SPECTRUM)), 'laplace_fm'),
            (uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK, waveform=lu.WF_SINE),
             self._objects(), 'ca_object_resonators'),
        )
        for new, old, name in pairs:
            self._same(run(new, gains=drag, transpose=notes),
                       run(old, gains=drag, transpose=notes), f"{name} under a played line")


class AliasMap(unittest.TestCase):
    """REQ section 5: what an old id's own parameter set means on the axes."""

    def test_every_old_id_is_on_the_axes(self):
        got = lu.unified_params('laplacian', dict(SPECTRUM))
        self.assertEqual((got['artic'], got['voice'], got['waveform']),
                         (lu.ARTIC_ENV, lu.VOICE_BANK, lu.WF_SINE))
        got = lu.unified_params(lc.ENGINE_ID, dict(method=lc.METHOD_BANK,
                                                   waveform=lc.WF_SQUARE, **SPECTRUM))
        self.assertEqual((got['artic'], got['voice'], got['waveform']),
                         (lu.ARTIC_ENV, lu.VOICE_BANK, lc.WF_SQUARE))
        got = lu.unified_params('laplace_fm', dict(fm_depth=2.0, **SPECTRUM))
        self.assertEqual((got['artic'], got['voice'], got['fm_depth']),
                         (lu.ARTIC_ENV, lu.VOICE_FM, 2.0))
        got = lu.unified_params('ca_object_resonators',
                                dict(detector=orz.DET_DISK, radius_mul=2.0,
                                     spectrum=orz.SPEC_LAPLACE, events=1,
                                     excitation=orz.EXC_UNIFORM, birth_strength=1.0,
                                     frequency_scale=220.0, decay_s=1.2,
                                     decay_law=orz.LAW_FIXED, attack_ms=5.0, **SPECTRUM))
        self.assertEqual((got['artic'], got['voice'], got['events'],
                          got['radius_mul'], got['decay_s'], got['attack_ms']),
                         (lu.ARTIC_EVENTS, lu.VOICE_BANK, 1, 2.0, 1.2, 5.0))

    def test_the_spectral_settings_travel_unchanged(self):
        got = lu.unified_params('laplacian', dict(SPECTRUM))
        for k in lu.SPECTRUM_KEYS:
            self.assertEqual(got[k], SPECTRUM[k], k)

    def test_a_parameter_set_outside_the_axes_says_so(self):
        """The Filter method and the Objects settings section 4 leaves out are not
        cells of this engine -- and are never silently read as something else."""
        self.assertIsNone(lu.unified_params(lc.ENGINE_ID,
                                            dict(method=lc.METHOD_FILTER, waveform=0, **SPECTRUM)))
        base = dict(detector=orz.DET_DISK, radius_mul=1.0, spectrum=orz.SPEC_LAPLACE,
                    events=0, excitation=orz.EXC_UNIFORM, birth_strength=1.0,
                    frequency_scale=220.0, decay_s=0.8, decay_law=orz.LAW_FIXED,
                    attack_ms=0.0, **SPECTRUM)
        for key, value in (('detector', orz.DET_OWN), ('spectrum', orz.SPEC_FIGURE),
                           ('excitation', orz.EXC_POSITION), ('decay_law', orz.LAW_MODAL),
                           ('birth_strength', 2.0)):
            p = dict(base)
            p[key] = value
            self.assertIsNone(lu.unified_params('ca_object_resonators', p), key)
        self.assertIsNone(lu.unified_params('gutter_field', {}))


class Switching(unittest.TestCase):
    """REQ section 6: an axis switch is never an impulse and never a cut."""

    def _play(self, engine, switch_at, change, blocks=30):
        g = scene()
        engine.init(g, None, 0.0)
        out = []
        for i in range(blocks):
            if i % 2 == 1:
                g = step(g)
                engine.update_field(g, events_field(g, g))
            if i == switch_at:
                p = dict(engine.params)
                p.update(change)
                engine.set_params(p)
            buf, _peak, _clip = engine.render(GAIN, i * BLOCK)
            out.append(buf.copy())
        return np.concatenate(out)

    def _seam(self, y, switch_at, span=2):
        """The step ACROSS the switch against the steps around it: a crossfade
        that clicks puts its biggest jump exactly on that seam.  (Comparing the
        loudest step of the whole run would only measure how much harder the new
        cell hits, which is a level difference, not a discontinuity.)"""
        d = np.abs(np.diff(y[:, 0].astype(np.int64)))
        seam = switch_at * BLOCK - 1
        lo, hi = seam - span * BLOCK, seam + span * BLOCK
        around = np.concatenate((d[lo:seam], d[seam + 1:hi]))
        return int(d[seam]), int(around.max())

    def test_switching_an_axis_makes_no_step(self):
        for change, what in ((dict(artic=lu.ARTIC_EVENTS), 'articulation'),
                             (dict(voice=lu.VOICE_FM), 'voicing')):
            y = self._play(uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK),
                           switch_at=14, change=change)
            step_at, around = self._seam(y, 14)
            self.assertLessEqual(step_at, around,
                                 f"the {what} switch stepped by {step_at}, more than "
                                 f"the {around} the samples around it step")

    def test_switching_over_a_still_field_is_silent(self):
        """Nothing has happened in the automaton, so the Events articulation has
        nothing to strike: the switch itself never makes a packet."""
        e = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        g = scene()
        e.init(g, None, 0.0)
        for i in range(8):
            e.render(GAIN, i * BLOCK)
        p = dict(e.params)
        p['artic'] = lu.ARTIC_EVENTS
        e.set_params(p)
        tail = [e.render(GAIN, (8 + i) * BLOCK)[0] for i in range(16)]
        # the Env cell rings out over its release, then silence -- no strike
        self.assertTrue(np.array_equal(np.concatenate(tail[-8:]),
                                       np.zeros((8 * BLOCK, 2), np.int16)),
                        "the switch to Events struck the field")

    def test_the_outgoing_cell_rings_out_instead_of_being_cut(self):
        e = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        g = scene()
        e.init(g, None, 0.0)
        for i in range(8):
            e.render(GAIN, i * BLOCK)
        self.assertEqual(len(e._layers), 1)
        p = dict(e.params)
        p['voice'] = lu.VOICE_FM
        e.set_params(p)
        e.render(GAIN, 8 * BLOCK)
        self.assertEqual(len(e._layers), 2, "the old cell was dropped, not faded")
        for i in range(e._xfade_blocks + 2):
            e.render(GAIN, (9 + i) * BLOCK)
        self.assertEqual(len(e._layers), 1, "the faded cell was never released")

    def test_a_switch_straight_back_turns_the_crossfade_around(self):
        e = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        g = scene()
        e.init(g, None, 0.0)
        for i in range(6):
            e.render(GAIN, i * BLOCK)
        first = e._layers[0].cell
        p = dict(e.params)
        p['voice'] = lu.VOICE_FM
        e.set_params(p)
        e.render(GAIN, 6 * BLOCK)
        p = dict(e.params)
        p['voice'] = lu.VOICE_BANK
        e.set_params(p)
        e.render(GAIN, 7 * BLOCK)
        self.assertIs(e._active().cell, first, "a third cell was stacked on the fade")
        self.assertLessEqual(len(e._layers), 2, "a third cell was stacked on the fade")
        cells = {id(lay.cell) for lay in e._layers}
        self.assertIn(id(first), cells)

    def test_every_combination_of_the_axes_is_a_cell(self):
        for artic in (lu.ARTIC_ENV, lu.ARTIC_EVENTS):
            for voice in (lu.VOICE_BANK, lu.VOICE_FM):
                for wf in (lu.WF_SINE, lc.WF_SAW, lc.WF_SQUARE):
                    e = uni(artic=artic, voice=voice, waveform=wf)
                    g = scene()
                    e.init(g, None, 0.0)
                    buf, _p, _c = e.render(GAIN, 0)
                    self.assertEqual(buf.shape, (BLOCK, 2), (artic, voice, wf))


class Contract(unittest.TestCase):
    """The engine contract the host relies on (casynth_engines.engine_api)."""

    def test_a_block_is_a_contiguous_int16_stereo_array(self):
        from casynth_engines.engine_api import check_block
        for axes in (dict(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK),
                     dict(artic=lu.ARTIC_ENV, voice=lu.VOICE_FM),
                     dict(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK)):
            e = uni(**axes)
            g = scene()
            e.init(g, None, 0.0)
            buf, peak, n_clip = e.render(GAIN, 0)
            check_block(buf, CTX, lu.ENGINE_ID)
            self.assertGreaterEqual(peak, 0.0)
            self.assertGreaterEqual(n_clip, 0)

    def test_an_empty_field_is_exactly_silent(self):
        for axes in (dict(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK),
                     dict(artic=lu.ARTIC_ENV, voice=lu.VOICE_FM),
                     dict(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK)):
            e = uni(**axes)
            e.init(np.zeros((48, 64), np.uint8), None, 0.0)
            for i in range(6):
                buf, peak, _clip = e.render(GAIN, i * BLOCK)
                self.assertEqual(int(np.abs(buf).max()), 0)
                self.assertEqual(peak, 0.0)

    def test_reset_restarts_the_whole_engine(self):
        e = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        a = run(e)
        e.reset(0.0)
        # reset() restarts on the field the run ended on; a fresh run on the same
        # programme is the honest comparison
        b = run(uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK))
        self.assertTrue(np.array_equal(a, b))

    def test_the_host_may_override_the_gain_glide(self):
        e = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        g = scene()
        e.init(g, None, 0.0)
        e.render(GAIN, 0)
        a, _p, _c = e.render(GAIN, BLOCK, gain_prev=0.0)
        e2 = uni(artic=lu.ARTIC_ENV, voice=lu.VOICE_BANK)
        e2.init(g, None, 0.0)
        e2.render(GAIN, 0)
        b, _p, _c = e2.render(GAIN, BLOCK)
        self.assertFalse(np.array_equal(a, b), "gain_prev did not reach the cell")


class ObjectsHooks(unittest.TestCase):
    """The three points the Events articulation needed in object_resonators --
    all additive, and the old path must be untouched."""

    def _objects(self):
        return registry.create('ca_object_resonators', CTX,
                               dict(detector=orz.DET_DISK, radius_mul=1.0,
                                    spectrum=orz.SPEC_LAPLACE, events=0,
                                    excitation=orz.EXC_UNIFORM, birth_strength=1.0,
                                    frequency_scale=220.0, decay_s=0.8,
                                    decay_law=orz.LAW_FIXED, attack_ms=0.0, **SPECTRUM))

    def test_begin_and_end_block_are_render_floats_own_halves(self):
        a, b = self._objects(), self._objects()
        g = scene()
        a.init(g, None, 0.0)
        b.init(g, None, 0.0)
        for i in range(8):
            g2 = step(g) if i % 2 else g
            if i % 2:
                g = g2
                a.update_field(g, events_field(g, g))
                b.update_field(g, events_field(g, g))
            ya, _p, _c = a.render_float(GAIN)
            b.begin_block(GAIN)
            out = np.zeros((BLOCK, 2))
            b._kernel(BLOCK, out)
            b.end_block()
            self.assertTrue(np.array_equal(ya, out), f"block {i}")

    def test_prime_silent_takes_the_field_as_its_own_past(self):
        e = self._objects()
        g = scene()
        e.init(g, None, 0.0)
        e.prime_silent()
        for i in range(6):
            buf, peak, _c = e.render(GAIN, i * BLOCK)
            self.assertEqual(int(np.abs(buf).max()), 0, "a primed engine struck the field")

    def test_an_attached_slot_array_follows_the_slots(self):
        e = self._objects()
        extra = e.attach_slot_state(np.zeros(orz.N_SLOTS))
        g = scene()
        e.init(g, None, 0.0)
        for i in range(4):
            e.render(GAIN, i * BLOCK)
        live = [f.slot for f in e.figures.values() if f.slot >= 0]
        self.assertTrue(live)
        for s in live:
            extra[s] = 100.0 + s
        moved = False
        for i in range(4, 40):
            g = step(g)
            e.update_field(g, events_field(g, g))
            e.render(GAIN, i * BLOCK)
            tails = np.nonzero(e.role == orz.ROLE_TAIL)[0]
            if len(tails):
                moved = True
                for t in tails:
                    self.assertNotEqual(float(extra[t]), 0.0,
                                        "a bank moved to a tail without its voicing")
                break
        self.assertTrue(moved, "no tail appeared: the hook was never exercised")
        e.init(g, None, 0.0)
        self.assertEqual(float(np.abs(extra).max()), 0.0, "a restart left the voicing behind")

    def test_a_wrong_sized_attachment_is_refused(self):
        with self.assertRaises(ValueError):
            self._objects().attach_slot_state(np.zeros(3))


class AliasFactories(unittest.TestCase):
    """REQ section 5: the four old ids are ALIASES of this engine.

    The id resolves HERE -- one place knows which law an id stands for -- and
    what comes back is the cell its axes pin, which is that law's own engine.
    Its registry entry, its parameters, its snapshot and its bytes are untouched,
    which is what lets every scene, every snapshot and every record of
    lab_catalog/ go on replaying."""

    IDS = ('laplacian', 'laplace_carriers', 'laplace_fm', 'ca_object_resonators')

    def test_every_old_id_resolves_through_this_engine(self):
        seen = []
        real = lu.create

        def spy(ctx, params, engine_id=lu.ENGINE_ID):
            seen.append(engine_id)
            return real(ctx, params, engine_id)
        lu.create = spy
        try:
            for eid in self.IDS:
                registry.create(eid, CTX, registry.defaults(eid))
        finally:
            lu.create = real
        self.assertEqual(sorted(seen), sorted(self.IDS))

    def test_an_alias_hands_back_its_own_law(self):
        from casynth_engines.legacy_engine import LegacySynthEngine
        from casynth_engines import laplace_fm as lfm
        want = {'laplacian': LegacySynthEngine,
                'laplace_carriers': lc.LaplaceCarriersEngine,
                'laplace_fm': lfm.LaplaceFMEngine,
                'ca_object_resonators': orz.ObjectResonatorsEngine}
        for eid, cls in want.items():
            self.assertIs(type(registry.create(eid, CTX, registry.defaults(eid))), cls, eid)

    def test_an_alias_renders_what_its_class_renders(self):
        """The factory changed, the sound did not: the same parameters through
        the alias and through the class itself are the same bytes."""
        from casynth_engines.legacy_engine import LegacySynthEngine
        from casynth_engines import laplace_fm as lfm
        direct = {
            'laplacian': lambda p: LegacySynthEngine(CTX, p, 'laplacian'),
            'laplace_carriers': lambda p: lc.LaplaceCarriersEngine(CTX, p),
            'laplace_fm': lambda p: lfm.LaplaceFMEngine(CTX, p),
            'ca_object_resonators': lambda p: orz.ObjectResonatorsEngine(CTX, p),
        }
        for eid, build in direct.items():
            p = dict(registry.defaults(eid))
            a = run(registry.create(eid, CTX, dict(p)))
            b = run(build(dict(p)))
            self.assertTrue(np.array_equal(a, b), f"{eid}: the alias changed the sound")

    def test_the_registry_entries_are_untouched(self):
        """A host reads the entry BEFORE it builds anything; an alias may not
        move a parameter, a default, a named choice or a capability."""
        expect = {
            'laplacian': dict(label='Laplace', params=list(lu.SPECTRUM_KEYS),
                              notes=True, gen=True, slew=True),
            'laplace_carriers': dict(label='Laplace waves',
                                     params=['method', 'waveform', 'filter_width_oct',
                                             'filter_depth_db'] + list(lu.SPECTRUM_KEYS),
                                     notes=True, gen=True, slew=False),
            'laplace_fm': dict(label='Laplace FM',
                               params=['fm_depth'] + list(lu.SPECTRUM_KEYS),
                               notes=True, gen=True, slew=False),
            'ca_object_resonators': dict(label='Objects', params=None,
                                         notes=True, gen=False, slew=False),
        }
        for eid, want in expect.items():
            spec = registry.get(eid)
            self.assertEqual(spec.label, want['label'], eid)
            if want['params'] is not None:
                self.assertEqual([p[0] for p in spec.params], want['params'], eid)
            self.assertEqual(spec.plays_notes, want['notes'], eid)
            self.assertEqual(spec.gen_envelope, want['gen'], eid)
            self.assertEqual(spec.gen_amp_slew, want['slew'], eid)
        self.assertEqual(registry.get('ca_object_resonators').ranges['radius_mul'],
                         orz.RADIUS_RANGE)
        self.assertIsNotNone(registry.get('ca_object_resonators').validate)
        self.assertEqual(registry.get(lc.ENGINE_ID).choices['method'], lc.METHOD_NAMES)

    def test_the_order_of_the_ids_is_unchanged(self):
        """The bench buttons and the prototype's tabs are built from it."""
        ids = registry.ids()
        self.assertEqual(ids[0], 'laplacian')
        for eid in self.IDS:
            self.assertIn(eid, ids)
        self.assertEqual(ids.index('ca_object_resonators') + 1, ids.index('laplace_carriers'))
        self.assertEqual(ids.index('laplace_carriers') + 1, ids.index('laplace_fm'))

    def test_an_id_this_engine_never_collapsed_is_refused(self):
        with self.assertRaises(KeyError):
            lu.create(CTX, {}, 'gutter_field')

    def test_a_laplace_engine_still_costs_no_numba(self):
        """A1 of the seam: a host that plays one Laplace line must not pull the
        resonator bank.  Routing the id through the unified engine may not undo
        that -- the Events cells are imported only when an axis asks for them."""
        import subprocess
        code = ("import sys;"
                "from casynth_engines import registry;"
                "from casynth_engines.engine_api import EngineContext;"
                "registry.create('laplacian', EngineContext(44100, 352, 2, 110.0, 1.0, 6.0),"
                "                registry.defaults('laplacian'));"
                "print('numba' in sys.modules,"
                "      'casynth_engines.object_resonators' in sys.modules)")
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.split(), ['False', 'False'], out.stderr[-400:])


class TheWaveKernelIsTheLawsKernel(unittest.TestCase):
    """REQ section 3 / 6: the Saw / Square readout is object_resonators._render
    with one branch changed -- not "the same idea", the same arithmetic."""

    ARRAYS = ('role', 'ndrive', 'npulse', 'nlive', 'zre', 'zim', 'cth', 'sth',
              'wcur', 'winc', 'wtgt', 'wleft', 'zf', 'zs', 'zu', 'zfm', 'zsm',
              'zum', 'pan', 'pleft', 'rr', 'gg', 'qq', 'ints', 'level',
              'slaw', 'gam_a', 'gam_t', 'hp')
    ORDER = ('role', 'ndrive', 'npulse', 'nlive', 'zre', 'zim', 'cth', 'sth',
             'wcur', 'winc', 'wtgt', 'wleft', 'zf', 'zs', 'zu', 'zfm', 'zsm',
             'zum', 'pan', 'pleft', 'rr', 'gg', 'qq', 'ints')

    def _state(self, seed=3, S=7, M=5):
        rng = np.random.RandomState(seed)
        omega = rng.uniform(0.01, 2.0, (S, M))
        return dict(
            role=rng.randint(0, 4, S).astype(np.int64),
            ndrive=rng.randint(0, M + 1, S).astype(np.int64),
            npulse=rng.randint(0, M + 1, S).astype(np.int64),
            nlive=rng.randint(0, M + 1, S).astype(np.int64),
            zre=rng.randn(S, M), zim=rng.randn(S, M),
            cth=np.cos(omega), sth=np.sin(omega),
            wcur=rng.rand(S, M), winc=rng.randn(S, M) * 1e-4, wtgt=rng.rand(S, M),
            wleft=rng.randint(0, 40, S).astype(np.int64),
            zf=rng.randn(S), zs=rng.randn(S), zu=rng.randn(S),
            zfm=rng.randn(S, M), zsm=rng.randn(S, M), zum=rng.randn(S, M),
            pan=rng.rand(S, 6), pleft=rng.randint(0, 40, S).astype(np.int64),
            rr=np.array([0.9995, 0.9990, -1e-7]),
            gg=np.array([0.40, 0.55, 1e-4]),
            qq=np.array([0.30, 0.20, -1e-4]),
            ints=np.array([0, 17, 23, 11], np.int64),
            level=np.zeros(S), slaw=rng.randint(0, 2, S).astype(np.int64),
            gam_a=rng.rand(S, M) * 12.0, gam_t=rng.rand(S, M) * 12.0,
            hp=np.array([[0.01, -0.02], [0.03, 0.004]]))

    def _run(self, st, wave, tabs=None, tab_a=None, tab_b=None, blending=0):
        n = BLOCK
        out = np.zeros((n, 2))
        head = [st[k] for k in self.ORDER]
        tail = [en.consts(float(SR)),
                math.exp(-2.0 * math.pi * orz.HP_HZ / SR), st['hp'],
                orz.OUT_SCALE, st['level'], st['slaw'], st['gam_a'], st['gam_t'],
                orz.gamma_smooth_k(float(SR)), float(SR)]
        if not wave:
            orz._render(n, out, *head, *tail)
            return out
        if tabs is None:
            tabs = np.zeros((1, lc.TABLE_N))
        if tab_a is None:
            tab_a = np.full(st['zre'].shape, -1, np.int64)
        if tab_b is None:
            tab_b = tab_a
        uev._render_wave(n, out, *head, *tail, tab_a, tab_b, tabs,
                         tabs.shape[1], blending, lu._RAMP)
        return out

    def test_the_sine_readout_is_bit_for_bit_the_original_kernel(self):
        law = self._state()
        wave = {k: v.copy() for k, v in law.items()}
        a = self._run(law, wave=False)
        b = self._run(wave, wave=True)
        self.assertTrue(np.array_equal(a, b), "the samples differ")
        for name in self.ARRAYS:
            self.assertTrue(np.array_equal(law[name], wave[name]),
                            f"the kernel left {name} in a different state")

    def test_a_wave_is_the_amplitude_of_the_mode_times_its_table(self):
        """One undriven slot, one mode: the readout must be |z| W(arg z + pi/2)
        sample by sample, through the engine's own DC filter and OUT_SCALE."""
        S, M, n = 1, 1, BLOCK
        f, r = 220.0, 0.9995
        w = 2.0 * math.pi * f / SR
        st = dict(role=np.array([orz.ROLE_TAIL], np.int64),
                  ndrive=np.zeros(S, np.int64), npulse=np.zeros(S, np.int64),
                  nlive=np.ones(S, np.int64),
                  zre=np.array([[1.0]]), zim=np.array([[0.0]]),
                  cth=np.array([[math.cos(w)]]), sth=np.array([[math.sin(w)]]),
                  wcur=np.array([[0.5]]), winc=np.zeros((S, M)), wtgt=np.array([[0.5]]),
                  wleft=np.zeros(S, np.int64),
                  zf=np.zeros(S), zs=np.zeros(S), zu=np.zeros(S),
                  zfm=np.zeros((S, M)), zsm=np.zeros((S, M)), zum=np.zeros((S, M)),
                  pan=np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                  pleft=np.zeros(S, np.int64),
                  rr=np.array([r, r, 0.0]), gg=np.array([1.0, 1.0, 0.0]),
                  qq=np.zeros(3), ints=np.zeros(4, np.int64), level=np.zeros(S),
                  slaw=np.zeros(S, np.int64), gam_a=np.zeros((S, M)),
                  gam_t=np.zeros((S, M)), hp=np.zeros((2, 2)))
        tabs = np.zeros((2, lc.TABLE_N))
        tabs[1] = lc.wavetable(lc.WF_SAW, f, SR)
        got = self._run({k: v.copy() for k, v in st.items()}, wave=True, tabs=tabs,
                        tab_a=np.array([[1]], np.int64))
        z = complex(1.0, 0.0)
        rot = complex(math.cos(w), math.sin(w))
        h = math.exp(-2.0 * math.pi * orz.HP_HZ / SR)
        prev = mix = 0.0
        ref = np.zeros(n)
        for t in range(n):
            z = r * rot * z
            th = (math.atan2(z.imag, z.real) + math.pi / 2.0) % (2.0 * math.pi)
            x = th * (lc.TABLE_N / (2.0 * math.pi))
            i0 = int(x)
            fr = x - i0
            i1 = (i0 + 1) % lc.TABLE_N
            b = 0.5 * abs(z) * (tabs[1][i0] * (1.0 - fr) + tabs[1][i1] * fr)
            prev = h * ((prev + b) - mix)
            mix = b
            ref[t] = prev * orz.OUT_SCALE
        self.assertLess(float(np.abs(got[:, 0] - ref).max()), 1e-12)


class TheNewWaves(unittest.TestCase):
    """REQ section 6: the wave law of req-laplace-carriers section 3 on a bank
    the automaton struck."""

    ONE_MODE = dict(SPECTRUM, n=1)

    def _ring(self, waveform, blocks=28, spectrum=None, **axes):
        """One compact figure, struck once by the opening packet, left to ring on
        a still field -- so the window holds the wave and not an attack."""
        g = np.zeros((32, 32), np.uint8)
        g[12:18, 12:18] = 1
        p = dict(lu.OPTIONAL_PARAMS)
        p.update(spectrum or SPECTRUM)
        p.update(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK, waveform=waveform,
                 decay_s=1.5, **axes)
        e = lu.UnifiedLaplaceEngine(CTX, p)
        e.init(g, None, 0.0)
        out = [e.render(GAIN, i * BLOCK)[0] for i in range(blocks)]
        return e, np.concatenate(out)

    def _spectrum_of(self, y, skip=8):
        m = y[skip * BLOCK:, 0].astype(float) / 32767.0
        sp = np.abs(np.fft.rfft(m * np.hanning(len(m))))
        return np.fft.rfftfreq(len(m), 1.0 / SR), sp

    def _mode_frequency(self, e):
        obj = e._active().cell.obj
        s = next(f.slot for f in obj.figures.values() if f.slot >= 0)
        return float(obj.ffreq[s, 0])

    def _harmonic_levels(self, y, f1, n_h=6):
        f, sp = self._spectrum_of(y)
        out = []
        for h in range(1, n_h + 1):
            k = int(np.argmin(np.abs(f - h * f1)))
            out.append(float(sp[max(0, k - 3):k + 4].max()))
        return np.array(out) / max(out[0], 1e-30)

    def test_one_mode_sounded_as_a_saw_is_c_h_equal_one_over_h(self):
        e, saw = self._ring(lc.WF_SAW, spectrum=self.ONE_MODE)
        got = self._harmonic_levels(saw, self._mode_frequency(e))
        want = 1.0 / np.arange(1, len(got) + 1)
        self.assertLess(float(np.abs(20.0 * np.log10(got / want)).max()), 2.0,
                        f"Saw harmonics {np.round(got, 4)} vs 1/h {np.round(want, 4)}")

    def test_one_mode_sounded_as_a_square_keeps_only_the_odd_harmonics(self):
        e, sq = self._ring(lc.WF_SQUARE, spectrum=self.ONE_MODE)
        got = self._harmonic_levels(sq, self._mode_frequency(e))
        for h in range(2, len(got) + 1):
            level = 20.0 * math.log10(max(got[h - 1], 1e-12))
            if h % 2:
                self.assertLess(abs(level - 20.0 * math.log10(1.0 / h)), 2.0,
                                f"odd harmonic {h} at {level:.1f} dB")
            else:
                self.assertLess(level, -40.0, f"Square sounded harmonic {h}")

    def test_one_mode_sounded_as_a_sine_has_no_harmonics(self):
        e, sine = self._ring(lu.WF_SINE, spectrum=self.ONE_MODE)
        got = self._harmonic_levels(sine, self._mode_frequency(e))
        self.assertLess(20.0 * math.log10(max(float(got[1:].max()), 1e-12)), -40.0)

    def test_the_band_limit_is_taken_per_wave(self):
        """b(nu) is the law's, taken at the frequency of every harmonic of every
        wave: a ringing bank has nothing above 0.45 sr."""
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            _e, y = self._ring(wf, blocks=40)
            f, sp = self._spectrum_of(y, skip=16)
            p = sp ** 2
            over = float(p[f >= 0.45 * SR].sum() / max(p.sum(), 1e-30))
            self.assertLess(10.0 * math.log10(max(over, 1e-30)), -60.0,
                            f"{lc.WAVE_NAMES[wf]}: energy above 0.45 sr")

    def test_a_wave_sounds_more_than_the_sine_bank(self):
        _e, sine = self._ring(lu.WF_SINE)
        _e, saw = self._ring(lc.WF_SAW)
        self.assertGreater(float(np.abs(saw.astype(float)).mean()),
                           float(np.abs(sine.astype(float)).mean()))

    def test_changing_the_waveform_is_a_blend_not_a_step(self):
        g = np.zeros((32, 32), np.uint8)
        g[12:18, 12:18] = 1
        e = uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK, waveform=lu.WF_SINE,
                decay_s=1.5)
        e.init(g, None, 0.0)
        out = []
        for i in range(16):
            if i == 8:
                p = dict(e.params)
                p['waveform'] = lc.WF_SAW
                e.set_params(p)
            out.append(e.render(GAIN, i * BLOCK)[0])
        y = np.concatenate(out)
        d = np.abs(np.diff(y[:, 0].astype(np.int64)))
        seam = 8 * BLOCK - 1
        around = np.concatenate((d[seam - 2 * BLOCK:seam], d[seam + 1:seam + 2 * BLOCK]))
        self.assertLessEqual(int(d[seam]), int(around.max()),
                             "the waveform change stepped")


class TheNewFM(unittest.TestCase):
    """REQ section 6 and the FM law of req-laplace-fm sections 2-4."""

    def _fm(self, depth, blocks=28, still=True, decay_s=1.5):
        g = np.zeros((32, 32), np.uint8)
        g[12:18, 12:18] = 1
        e = uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_FM, fm_depth=depth,
                decay_s=decay_s)
        e.init(g, None, 0.0)
        out = []
        for i in range(blocks):
            if not still and i % 2 == 1:
                g = step(g)
                e.update_field(g, events_field(g, g))
            out.append(e.render(GAIN, i * BLOCK)[0])
        return e, np.concatenate(out)

    def test_the_phase_sum_is_the_law(self):
        """_fm_sum against a direct evaluation of A sin(theta_c + sum I a sin theta):
        the index is I*a in RADIANS, every modulator of a slot enters the SAME
        phase, and a trajectory is read linearly between audio samples."""
        rng = np.random.RandomState(11)
        os_, n, V, K = 4, 16, 2, 5
        n_os = n * os_
        amp, carr = rng.rand(K, n), rng.rand(V, n)
        panl, panr = rng.rand(V, n), rng.rand(V, n)
        amp_prev, carr_prev = rng.rand(K), rng.rand(V)
        panl_prev, panr_prev = rng.rand(V), rng.rand(V)
        row_start = np.array([0, 3, K], np.int64)
        inc_mod = rng.uniform(0.01, 0.4, K)
        th_mod = rng.uniform(0.0, 6.0, K)
        inc_c, index = 0.07, 1.7
        th_c = rng.uniform(0.0, 6.0, V)
        got_l, got_r = np.zeros(n_os), np.zeros(n_os)
        uev._fm_sum(n_os, os_, index, amp, amp_prev, carr, carr_prev, panl,
                    panl_prev, panr, panr_prev, row_start, inc_mod,
                    th_mod.copy(), inc_c, th_c.copy(), got_l, got_r)
        want_l, want_r = np.zeros(n_os), np.zeros(n_os)
        tm, tc = th_mod.copy(), th_c.copy()
        for m in range(n_os):
            t, fr = m // os_, (m % os_ + 1) / os_
            L = R = 0.0
            for v in range(V):
                acc = 0.0
                for k in range(row_start[v], row_start[v + 1]):
                    a0 = amp_prev[k] if t == 0 else amp[k, t - 1]
                    acc += index * (a0 + (amp[k, t] - a0) * fr) * math.sin(tm[k])
                    tm[k] = (tm[k] + inc_mod[k]) % (2.0 * math.pi)
                A0 = carr_prev[v] if t == 0 else carr[v, t - 1]
                y = (A0 + (carr[v, t] - A0) * fr) * math.sin(tc[v] + acc)
                tc[v] = (tc[v] + inc_c) % (2.0 * math.pi)
                l0 = panl_prev[v] if t == 0 else panl[v, t - 1]
                r0 = panr_prev[v] if t == 0 else panr[v, t - 1]
                L += (l0 + (panl[v, t] - l0) * fr) * y
                R += (r0 + (panr[v, t] - r0) * fr) * y
            want_l[m], want_r[m] = L, R
        self.assertLess(float(np.abs(got_l - want_l).max()), 1e-12)
        self.assertLess(float(np.abs(got_r - want_r).max()), 1e-12)

    def test_the_index_is_divided_by_nothing(self):
        """Not by the number of modes, not by sum a, not by an RMS: with the
        carrier standing still, arcsin of the output IS the phase sum."""
        os_, n = 2, 8
        n_os = n * os_
        amp = np.full((4, n), 0.2)
        one_row = np.ones((1, n))

        def run(index, rows):
            l, r = np.zeros(n_os), np.zeros(n_os)
            uev._fm_sum(n_os, os_, index, amp[:rows], np.full(rows, 0.2),
                        one_row, np.ones(1), one_row, np.ones(1),
                        one_row, np.ones(1), np.array([0, rows], np.int64),
                        np.full(rows, 0.2), np.zeros(rows), 0.0, np.zeros(1), l, r)
            return np.arcsin(np.clip(l, -1.0, 1.0))
        one = run(1.0, 1)
        self.assertGreater(float(np.abs(one).max()), 1e-3, "nothing was modulated")
        self.assertLess(float(np.abs(run(2.0, 1) - 2.0 * one).max()), 1e-12,
                        "the index does not scale with the depth")
        self.assertLess(float(np.abs(run(1.0, 4) - 4.0 * one).max()), 1e-12,
                        "the index was normalised by the number of modes")

    def test_depth_zero_is_a_decaying_sine_on_f0(self):
        """REQ section 2: FM d = 0 is the carrier of every sounding figure at its
        own scale -- not a return to the sum of modes."""
        _e, y = self._fm(0.0, blocks=32)
        m = y[BLOCK * 6:, 0].astype(float)
        sp = np.abs(np.fft.rfft(m * np.hanning(len(m))))
        f = np.fft.rfftfreq(len(m), 1.0 / SR)
        peak = f[int(np.argmax(sp))]
        self.assertLess(abs(peak - F0), 8.0, f"the carrier sits at {peak:.1f} Hz")
        half = len(y) // 2
        self.assertLess(float(np.abs(y[half:]).max()), float(np.abs(y[:half]).max()),
                        "the carrier did not decay with its figure")

    def test_a_deeper_index_is_a_wider_spectrum(self):
        widths = []
        for depth in (0.0, 1.0, 3.0):
            _e, y = self._fm(depth, blocks=28)
            m = y[BLOCK * 6:, 0].astype(float)
            sp = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
            f = np.fft.rfftfreq(len(m), 1.0 / SR)
            tot = float(sp.sum())
            centre = float((f * sp).sum() / tot)
            widths.append(math.sqrt(float((sp * (f - centre) ** 2).sum() / tot)))
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])

    def test_no_energy_above_the_band(self):
        """REQ section 6: the sum is built at OVERSAMPLE x sr and brought down
        through the Kaiser FIR, so a ringing figure stays inside the band."""
        for depth in (0.0, 1.0, 3.0):
            _e, y = self._fm(depth, blocks=32)
            m = y[BLOCK * 8:, 0].astype(float)
            sp = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
            f = np.fft.rfftfreq(len(m), 1.0 / SR)
            over = float(sp[f >= 0.45 * SR].sum() / max(sp.sum(), 1e-30))
            self.assertLess(10.0 * math.log10(max(over, 1e-30)), -60.0,
                            f"depth {depth}: energy above 0.45 sr")

    def test_the_dc_blocker_is_on_the_output(self):
        _e, y = self._fm(2.0, blocks=32, still=False)
        m = y[BLOCK * 6:, 0].astype(float)
        self.assertLess(abs(float(m.mean())), 0.01 * float(np.abs(m).max()) + 1.0)

    def test_an_empty_field_is_silent_at_every_depth(self):
        for depth in (0.0, 2.0):
            e = uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_FM, fm_depth=depth)
            e.init(np.zeros((32, 32), np.uint8), None, 0.0)
            for i in range(8):
                buf, _p, _c = e.render(GAIN, i * BLOCK)
                self.assertEqual(int(np.abs(buf).max()), 0)

    def test_the_carrier_phase_is_attached_to_the_bank(self):
        """attach_slot_state: the carrier has no state to be read out of, so its
        phase is handed to the tracker and moves, splits and clears WITH the
        slots -- a figure that leaves keeps its carrier in the tail it went to."""
        e = uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_FM, fm_depth=1.0)
        g = scene()
        e.init(g, None, 0.0)
        e.render(GAIN, 0)
        cell = e._active().cell
        self.assertIs(cell.th_c, cell.obj.extra_slot[0])
        src = next(f.slot for f in cell.obj.figures.values() if f.slot >= 0)
        cell.th_c[src] = 1.25
        cell.obj._move_slot(src, orz.N_ACTIVE)
        self.assertEqual(float(cell.th_c[orz.N_ACTIVE]), 1.25)
        self.assertEqual(float(cell.th_c[src]), 0.0)


class ThreadedRender(unittest.TestCase):
    """Both Events voicings build their block on SEVERAL THREADS -- and that must
    be a change of speed only.

    The wave readout divides its SLOTS (2026-09-22, the `artic` case) and the FM
    readout divides its CARRIERS (2026-09-22, the Events + FM case): in both the
    samples cannot be divided -- a mode is a recurrence and a modulator's phase
    free-runs -- but two slots share nothing except the sum at the end, and that
    sum is still taken in slot order.  So every output sample is computed by the
    same operations in the same order whatever the thread count is.

    This gate renders the SAME field at 1, 2 and 4 threads and compares the
    blocks byte for byte.  The field is crowded on purpose: the split is skipped
    when there are fewer slots than two per worker, and a gate that never split
    would prove nothing -- so it also counts the blocks that really went through
    the pool."""

    BLOCKS = 12

    @staticmethod
    def crowd():
        """A field of about a hundred figures -- what a Random field gives the
        tracker, and the only kind of scene where the pool is used at all."""
        rng = np.random.default_rng(7)
        return (rng.random((30, 52)) < 0.35).astype(np.uint8)

    def _blocks(self, threads, axes, split=None):
        old = uev.render_pool.THREADS
        old_ranges = uev.render_pool.ranges
        uev.render_pool.THREADS = int(threads)
        if split is not None:
            def counted(n, parts):
                split['n'] += 1
                return old_ranges(n, parts)
            uev.render_pool.ranges = counted
        try:
            e = uni(artic=lu.ARTIC_EVENTS, **axes)
            g = self.crowd()
            e.init(g, None, 0.0)
            out = []
            for i in range(self.BLOCKS):
                if i % 2 == 1:
                    new = step(g)
                    e.update_field(new, events_field(g, new))
                    g = new
                buf, _peak, _clip = e.render(GAIN, i * BLOCK)
                out.append(buf.copy())
            return out
        finally:
            uev.render_pool.THREADS = old
            uev.render_pool.ranges = old_ranges

    def _same(self, axes, what):
        ref = self._blocks(1, axes)
        for threads in (2, 4):
            split = {'n': 0}
            got = self._blocks(threads, axes, split)
            self.assertGreater(
                split['n'], self.BLOCKS // 2,
                f"{what}: only {split['n']} of {self.BLOCKS} blocks were divided at "
                f"{threads} threads -- the gate would prove nothing")
            for i, (a, b) in enumerate(zip(ref, got)):
                self.assertTrue(
                    np.array_equal(a, b),
                    f"{what}: block {i} on {threads} threads differs from the single "
                    f"loop at {int(np.argmax(a != b))} of {a.size} samples")

    def test_the_wave_block_is_the_single_threaded_block(self):
        """Events + Bank + Square: the slots divided over the pool."""
        self._same(dict(voice=lu.VOICE_BANK, waveform=lc.WF_SQUARE), 'wave')

    def test_the_fm_block_is_the_single_threaded_block(self):
        """Events + FM: the carriers divided over the pool."""
        self._same(dict(voice=lu.VOICE_FM, fm_depth=1.0), 'FM')


class RegistryEntry(unittest.TestCase):
    """The engine as a host reads it before building an instance."""

    def test_the_engine_is_registered_with_its_axes(self):
        spec = registry.get(lu.ENGINE_ID)
        self.assertEqual(spec.label, lu.LABEL)
        self.assertEqual([p[0] for p in spec.params], [p[0] for p in lu.PARAMS])
        self.assertEqual(spec.choices['artic'], lu.ARTIC_NAMES)
        self.assertEqual(spec.choices['voice'], lu.VOICE_NAMES)
        self.assertEqual(spec.ranges['radius_mul'], lu.RADIUS_RANGE)
        self.assertTrue(spec.plays_notes)
        self.assertTrue(spec.gen_envelope)
        self.assertFalse(spec.gen_amp_slew)

    def test_the_gen_block_is_the_engines_own_under_events(self):
        """The Env cells run the SlotPool envelope, so the host's GEN knobs act;
        the Events articulation has a decay law of its own and the panel says so
        instead of offering four knobs that do nothing."""
        p = dict(lu.OPTIONAL_PARAMS)
        p.update(SPECTRUM)
        self.assertNotIn('gen_envelope', lu.inactive(p))
        p['artic'] = lu.ARTIC_EVENTS
        self.assertIn('gen_envelope', lu.inactive(p))

    def test_every_cell_renders_through_the_registry(self):
        for axes in (dict(artic=0, voice=0, waveform=0), dict(artic=0, voice=0, waveform=1),
                     dict(artic=0, voice=1), dict(artic=1, voice=0, waveform=0),
                     dict(artic=1, voice=0, waveform=2), dict(artic=1, voice=1)):
            p = dict(registry.defaults(lu.ENGINE_ID))
            p.update(SPECTRUM)
            p.update(axes)
            e = registry.create(lu.ENGINE_ID, CTX, p)
            e.init(scene(), None, 0.0)
            for i in range(4):
                buf, peak, _n = e.render(GAIN, i * BLOCK)
                self.assertEqual(buf.shape, (BLOCK, 2), axes)
                self.assertTrue(math.isfinite(peak))


if __name__ == '__main__':
    unittest.main()
