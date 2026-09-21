#!/usr/bin/env python3
"""The unified Laplace engine (REQ memory/req-unified-laplace-2026-09-21.md).

Steps C1 and C2 -- the axes, the four byte anchors and the aliases:

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

    def test_the_cells_that_are_not_here_yet_are_refused(self):
        with self.assertRaises(ValueError):
            uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_FM)
        with self.assertRaises(ValueError):
            uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK, waveform=lc.WF_SAW)
        e = uni(artic=lu.ARTIC_EVENTS, voice=lu.VOICE_BANK)
        before = dict(e.params)
        with self.assertRaises(ValueError):
            p = dict(e.params)
            p['voice'] = lu.VOICE_FM
            e.set_params(p)
        self.assertEqual(e.params, before, "a refused change was applied anyway")


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


if __name__ == '__main__':
    unittest.main()
