#!/usr/bin/env python3
"""Golden-master gate for the tail pool UNDER OVERLOAD (SlotPool, casynth_engine).

WHY (2026-09-23, coverage audit memory/log/2026-09-23-test-coverage-audit.md):
tests/golden/golden_master.py plays two voices, so the tail pool never fills:
the two branches that decide what happens when it does -- `_enforce_tail_budget`
fast-fading the quietest excess tails to keep a reserve, and `_acquire_tail`
STEALING the quietest ringing tail because no slot is free at all -- were
executed by no test.  They are the "graceful degradation without a click" law a
player hits on a dense Random field with a long GEN release, and commit 9849350
(2026-09-22) rewrote both of them vectorised with the claim "the same slot, bit
for bit".  For those branches nothing checked the claim.

Three programmes, each through the public path (analyse -> SlotPool.update ->
render_chunk_laplacian) on the prototype's field size:
  distinct  a dense Random field, every mode of every figure on a frequency of
            its own (harm 0) and moving at every generation, a release far longer
            than the generation interval: tails pile up and the budget EVICTS.
  player    the instrument's opening spectrum (harm 1 collapses frequencies
            onto harmonics, so fewer modes move), a shorter release: what a
            Random + Play does.  Evicts too.
  prefilled the same field, but the pool starts with EVERY tail slot ringing
            (a prepared state, deterministic): the budget evicts its quietest
            half, and the first generation's burst finds no free slot -- every
            tail it spawns has to STEAL.  The reserve (TAIL_RESERVE = 2 x the
            active set) is designed so that a field alone never gets there, which
            is why the branch needs a prepared pool to be reached at all.
The int16 output of each is compared byte for byte with a committed reference,
and the gate insists that the programmes still REACH the branches (evictions in
every programme, steals in `prefilled`): a reference that quietly stopped
reaching the code would be worth nothing.

The reference was blessed on commit f3e8518 -- the code BEFORE the vectorised
pool -- and compared with HEAD, so it pins the ORIGINAL slot-by-slot law, not
the rewrite (user's request, 2026-09-23).

USAGE:
  python tests/golden/tail_pool_master.py           # render, check the branches, compare
  python tests/golden/tail_pool_master.py --save    # re-bless (ONLY after an agreed
                                                    # audible change; say so in the log)
  python tests/golden/tail_pool_master.py --dry     # render and print, no compare
"""
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from casynth_config import (SR, CHUNK_S, MASTER_GAIN, TOTAL_SLOTS, N_ACTIVE,   # noqa: E402
                            N_TAIL, TAIL_RESERVE, FAST_EVICT_CHUNKS, TWO_PI)
from casynth_engine import (step, events_field, analyse, SlotPool,               # noqa: E402
                            render_chunk_laplacian, midi_to_freq)

REF_NPZ = os.path.join(_HERE, "tail_pool_master_ref.npz")
OUT_DIR = os.path.join(_ROOT, "artifacts")

ROWS, COLS = 30, 52                     # the prototype's field
GAIN = MASTER_GAIN * 0.7
F0 = float(midi_to_freq(48))
SPECTRUM_DISTINCT = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=0.0, fullshape=1, dyn=1.0)
SPECTRUM_PLAYER = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=1.0, fullshape=1, dyn=1.0)

# (name, laplacian params, density, seed, release_chunks, chunks per generation,
#  chunks, prefill the pool)
PROGRAMMES = [
    ('distinct', SPECTRUM_DISTINCT, 0.35, 7, 250, 10, 420, False),
    ('player', SPECTRUM_PLAYER, 0.30, 11, 125, 10, 300, False),
    ('prefilled', SPECTRUM_DISTINCT, 0.35, 7, 250, 8, 200, True),
]
MUST_STEAL = ('prefilled',)


def field(seed, density):
    rng = np.random.default_rng(seed)
    return (rng.random((ROWS, COLS)) < density).astype(np.uint8)


def _ringing(pool, amp_cur):
    """How many tail slots ring right now -- the count _enforce_tail_budget
    compares with the reserve (read-only)."""
    lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
    return int(np.count_nonzero((pool._release_cnt[lo:hi] > 0) | (amp_cur[lo:hi] >= 1e-4)))


def prefill(pool, phase, amp_cur, pan_cur, seed):
    """Every tail slot ringing: a deterministic spread of amplitudes (so the
    quietest is unambiguous), frequencies and phases, all mid-release."""
    rng = np.random.default_rng(seed)
    lo, hi = N_ACTIVE + 1, TOTAL_SLOTS + 1
    n = hi - lo
    amps = rng.permutation(np.linspace(0.02, 0.5, n))
    pool.freq_slots[lo:hi] = rng.uniform(200.0, 4000.0, n)
    phase[lo:hi] = rng.uniform(0.0, TWO_PI, n)
    amp_cur[lo:hi] = amps
    pool.amp_tgt[lo:hi] = amps
    pan_cur[lo:hi] = 0.5
    pool.pan_tgt[lo:hi] = 0.5
    pool._release_cnt[lo:hi] = 200
    pool._release_len[lo:hi] = 200
    pool._release_amp0[lo:hi] = amps


def render_programme(params, density, seed, release_chunks, gen_every, chunks, prefilled):
    """-> (int16 (n, 2), stats dict)."""
    pool = SlotPool()
    sz = TOTAL_SLOTS + 1
    phase, amp_cur, pan_cur = np.zeros(sz), np.zeros(sz), np.full(sz, 0.5)
    if prefilled:
        prefill(pool, phase, amp_cur, pan_cur, seed + 100)
    g, exc = field(seed, density), None
    _l, voices, _c = analyse(g, F0, 'laplacian', dict(params), exc=exc)
    out = []
    evictions = 0
    max_ring = 0
    for i in range(chunks):
        if i and i % gen_every == 0:
            new = step(g)
            exc = events_field(g, new)
            g = new
            _l, voices, _c = analyse(g, F0, 'laplacian', dict(params), exc=exc)
        ring = _ringing(pool, amp_cur)
        max_ring = max(max_ring, ring)
        if ring > N_TAIL - TAIL_RESERVE:
            evictions += 1                 # this update runs the budget's eviction branch
        pool.update(voices, phase, amp_cur, pan_cur, release_chunks, 1, 1, 1.0)
        buf, _pk, _nc = render_chunk_laplacian(phase, amp_cur, pan_cur, pool.amp_tgt,
                                               pool.pan_tgt, pool.freq_slots, 2, GAIN, GAIN)
        out.append(buf)
    audio = np.concatenate(out)
    stats = dict(steals=int(pool.steals), steal_amp_max=float(pool.steal_amp_max),
                 evictions=int(evictions), max_ring=int(max_ring),
                 voices=int(len(voices)))
    return audio, stats


def render():
    return {name: render_programme(p, d, s, rel, ge, ch, pre)
            for (name, p, d, s, rel, ge, ch, pre) in PROGRAMMES}


def main():
    force_save = "--save" in sys.argv
    dry = "--dry" in sys.argv
    got = render()
    rc = 0
    for name, (audio, st) in got.items():
        digest = hashlib.sha256(audio.tobytes()).hexdigest()[:16]
        print(f"[tail-pool] {name:9s} {len(audio)} samples sha256[:16]={digest}  "
              f"steals={st['steals']} (loudest {st['steal_amp_max']:.3f})  "
              f"evictions={st['evictions']}  max ringing={st['max_ring']} of {N_TAIL}  "
              f"voices={st['voices']}  fast-evict={FAST_EVICT_CHUNKS} chunks")
        # the programmes must still reach the overload branches, or the
        # reference means nothing
        if st['evictions'] == 0 or (name in MUST_STEAL and st['steals'] == 0):
            print(f"[FAIL] {name}: the programme no longer exercises the overload branches "
                  f"(steals={st['steals']}, evictions={st['evictions']})")
            rc = 1
    if dry:
        return rc
    if force_save or not os.path.exists(REF_NPZ):
        np.savez_compressed(REF_NPZ,
                            **{f"{n}_audio": a for n, (a, _s) in got.items()},
                            **{f"{n}_steals": np.array(s['steals']) for n, (_a, s) in got.items()},
                            **{f"{n}_evictions": np.array(s['evictions'])
                               for n, (_a, s) in got.items()},
                            sr=SR, chunk_s=CHUNK_S)
        print(f"[baseline saved] {REF_NPZ}")
        return rc
    ref = np.load(REF_NPZ)
    for name, (audio, st) in got.items():
        ra = ref[f"{name}_audio"]
        r_steals, r_ev = int(ref[f"{name}_steals"]), int(ref[f"{name}_evictions"])
        same_counts = (st['steals'] == r_steals and st['evictions'] == r_ev)
        if audio.shape == ra.shape and np.array_equal(audio, ra) and same_counts:
            print(f"[PASS] {name}: byte-exact ({len(ra)} samples), "
                  f"steals {r_steals}, evictions {r_ev}")
            continue
        rc = 1
        os.makedirs(OUT_DIR, exist_ok=True)
        from scipy.io import wavfile
        path = os.path.join(OUT_DIR, f"_tail_pool_master_{name}.wav")
        wavfile.write(path, SR, audio)
        n = min(len(audio), len(ra))
        if n and audio.shape[1:] == ra.shape[1:]:
            diff = np.abs(audio[:n].astype(np.int64) - ra[:n].astype(np.int64))
            first = int(np.argmax(diff.max(axis=1) > 0)) if diff.max() else -1
            print(f"[FAIL] {name}: differs from the reference: shapes ref={ra.shape} "
                  f"out={audio.shape} max|diff|={int(diff.max())} LSB, "
                  f"{int(np.count_nonzero(diff))} of {diff.size} values, first at sample "
                  f"{first} (chunk {first // int(CHUNK_S * SR) if first >= 0 else '-'}); "
                  f"steals {st['steals']} vs {r_steals}, evictions {st['evictions']} vs {r_ev} "
                  f"(wrote {path})")
        else:
            print(f"[FAIL] {name}: shape mismatch ref={ra.shape} out={audio.shape} "
                  f"(wrote {path})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
