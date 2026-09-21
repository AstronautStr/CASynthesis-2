#!/usr/bin/env python3
"""Do the host's GEN envelope knobs actually reach the engine?

WHY THIS EXISTS.  The prototype hands its live GEN A/D/S/R and its tempo to the
hosted engine on every block (gol_synth._engine_tell -> set_envelope/set_rate).
`set_envelope` is OPTIONAL in the contract, so an engine that simply does not
implement it swallows the knobs in silence -- and that is exactly what happened
on 2026-09-21: "Laplace waves" and "Laplace FM" run the SlotPool envelope but
never read the knobs, so the whole GEN block of the panel was dead on those
tabs while every gate stayed green.  Nothing had ever asked the question.

So this asks it, per engine and in both directions:
  * an engine the registry says reads the knobs must SOUND different when they
    move (and the registry hint must agree with the class);
  * an engine with envelopes of its own must be untouched by them -- a dead
    knob is fine only when the panel says the block is inactive (`gen_envelope`
    drives that text, so the two can never drift apart);
  * an engine nobody ever tells anything must render exactly what it rendered
    before the knobs existed -- that is what keeps every scene and catalog
    record byte-for-byte.

Run:  python tests/test_gen_envelope.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from casynth_config import CHUNK_S, NOTE_DEFAULT                    # noqa: E402
from casynth_engine import step, events_field, midi_to_freq         # noqa: E402
from casynth_engines import registry                                # noqa: E402
from casynth_engines.engine_api import (EngineContext, reads_gen_envelope,   # noqa: E402
                                        supports_amp_slew)

SR = 44100
BLOCK = int(CHUNK_S * SR)
F0 = midi_to_freq(NOTE_DEFAULT)
RATE = 4.0                      # automaton steps / s
BLOCKS = 100                    # ~1.2 s of audio: several automaton steps at RATE

# the defaults an engine is built with (a host that says nothing) ...
DEFAULT_ENV = None
# ... against a clearly different envelope: a slow attack onto a low sustain
SLOW_ENV = dict(attack=0.9, decay=0.9, sustain=0.2, release=0.9, amp_slew=False)


def _field():
    """One step of a dense random field: live figures, and an excitation that
    makes the per-mode envelopes start rather than sit at sustain."""
    rng = np.random.RandomState(7)
    g0 = (rng.random_sample((30, 52)) < 0.28).astype(np.uint8)
    g1 = step(g0)
    return g0, g1


def _render(eid, env, rate=RATE, blocks=BLOCKS, params=None):
    """A LIVING field, as a host drives one: the automaton steps on its own clock
    while the engine renders.  A frozen field would hide half of what the GEN
    envelope does -- attacks, release tails and the amplitude slew only exist
    where the modes actually move."""
    prev, grid = _field()
    exc = events_field(prev, grid)
    ctx = EngineContext(SR, BLOCK, 2, F0, 1.0, rate, pan='field')
    e = registry.create(eid, ctx, dict(params or registry.defaults(eid)))
    if env is not None:
        e.set_rate(rate)
        e.set_envelope(env['attack'], env['decay'], env['sustain'],
                       env['release'], env['amp_slew'])
    e.init(grid, exc, 0.5)
    per_tick = max(1, int(round((1.0 / rate) / CHUNK_S)))
    out = []
    for i in range(blocks):
        if i and i % per_tick == 0:
            prev, grid = grid, step(grid)
            exc = events_field(prev, grid)
            e.update_field(grid, exc)
        out.append(e.render(0.5, i * BLOCK, transpose=1.0)[0])
    return np.concatenate(out)


def _maxdiff(a, b):
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())


class TheRegistryHintMatchesTheEngine(unittest.TestCase):
    """`gen_envelope` / `gen_amp_slew` are read by a UI BEFORE it builds an
    instance (that is why they live in the registry), so they must not drift
    away from what the class actually does."""

    def test_every_engine_agrees_with_its_registry_entry(self):
        ctx = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE)
        for eid in registry.ids():
            spec = registry.get(eid)
            inst = registry.create(eid, ctx, registry.defaults(eid))
            self.assertEqual(spec.gen_envelope, reads_gen_envelope(inst),
                             f"{eid}: the registry hint and the engine disagree "
                             f"about the GEN envelope knobs")
            self.assertEqual(spec.gen_amp_slew, supports_amp_slew(inst),
                             f"{eid}: the registry hint and the engine disagree "
                             f"about the GEN amp-slew toggle")

    def test_slew_is_never_claimed_without_the_envelope(self):
        for eid in registry.ids():
            spec = registry.get(eid)
            if spec.gen_amp_slew:
                self.assertTrue(spec.gen_envelope,
                                f"{eid}: claims the slew toggle but not the knobs")

    def test_the_engines_the_prototype_plays_are_covered(self):
        by_id = {eid: registry.get(eid) for eid in registry.ids()}
        for want in ('laplacian', 'laplace_carriers', 'laplace_fm'):
            self.assertTrue(by_id[want].gen_envelope,
                            f"{want}: its per-mode envelope IS the SlotPool's -- "
                            f"the host's GEN knobs must reach it")
        # Objects has a decay law of its own; the panel says so instead of
        # offering four knobs that do nothing
        self.assertFalse(by_id['ca_object_resonators'].gen_envelope)


class TheKnobsReachTheSound(unittest.TestCase):
    """The question a player asks: I moved the GEN block, did anything happen?"""

    def test_an_engine_that_reads_them_sounds_different(self):
        for eid in registry.ids():
            if not registry.get(eid).gen_envelope:
                continue
            with self.subTest(engine=eid):
                d = _maxdiff(_render(eid, DEFAULT_ENV), _render(eid, SLOW_ENV))
                self.assertGreater(d, 0, f"{eid}: the GEN knobs changed nothing")

    def test_an_engine_with_its_own_envelopes_is_untouched(self):
        for eid in registry.ids():
            if registry.get(eid).gen_envelope:
                continue
            with self.subTest(engine=eid):
                d = _maxdiff(_render(eid, DEFAULT_ENV), _render(eid, SLOW_ENV))
                self.assertEqual(d, 0, f"{eid}: says it ignores the GEN knobs "
                                       f"but its sound moved")

    def test_saying_nothing_is_the_same_as_saying_the_defaults(self):
        """What keeps the catalog: a scene without an `envelope` block renders
        exactly as it did before the knobs were live."""
        from casynth_config import (GEN_ATTACK_DEFAULT, GEN_DECAY_DEFAULT,
                                    GEN_SUSTAIN_DEFAULT, GEN_RELEASE_DEFAULT)
        defaults = dict(attack=GEN_ATTACK_DEFAULT, decay=GEN_DECAY_DEFAULT,
                        sustain=GEN_SUSTAIN_DEFAULT, release=GEN_RELEASE_DEFAULT,
                        amp_slew=False)
        for eid in registry.ids():
            with self.subTest(engine=eid):
                self.assertEqual(_maxdiff(_render(eid, DEFAULT_ENV),
                                          _render(eid, defaults)), 0,
                                 f"{eid}: an untold engine differs from one told "
                                 f"the defaults")

    def test_the_tempo_reaches_the_engines_that_read_it(self):
        """A/D/R are fractions of a TICK: at half the tempo they last twice as
        long, and an engine that froze ctx.rate_hz would not notice."""
        env = dict(SLOW_ENV)
        for eid in registry.ids():
            if not registry.get(eid).gen_envelope:
                continue
            with self.subTest(engine=eid):
                d = _maxdiff(_render(eid, env, rate=RATE),
                             _render(eid, env, rate=RATE / 4.0))
                self.assertGreater(d, 0, f"{eid}: the tempo changed nothing")


class TheAmpSlewToggleIsHonest(unittest.TestCase):
    """The slew smooths an amplitude change on a STABLE pitch -- the shape>0 beep
    it was added for -- so it is asked on a setting that produces one (a mode
    whose frequency stays put while its weight is gated), not on the defaults,
    where there is nothing for it to smooth."""

    @staticmethod
    def _gated_params(eid):
        p = dict(registry.defaults(eid))
        if registry.get(eid).spec_of('shape') is not None:
            p['shape'] = 0.6               # weights gated by the figure's shape
        return p

    def test_it_acts_where_it_is_claimed_and_nowhere_else(self):
        on = dict(SLOW_ENV, amp_slew=True)
        off = dict(SLOW_ENV, amp_slew=False)
        for eid in registry.ids():
            spec = registry.get(eid)
            if not spec.gen_envelope:
                continue
            with self.subTest(engine=eid):
                p = self._gated_params(eid)
                d = _maxdiff(_render(eid, on, params=p), _render(eid, off, params=p))
                if spec.gen_amp_slew:
                    self.assertGreater(d, 0, f"{eid}: claims the slew toggle "
                                             f"but it changes nothing")
                else:
                    self.assertEqual(d, 0, f"{eid}: declines the slew toggle "
                                           f"but it changed the sound")


if __name__ == '__main__':
    unittest.main(verbosity=2)
