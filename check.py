#!/usr/bin/env python3
"""Run ALL regression gates in one command.

    python check.py              # every gate: a parallel wave, then the timed ones alone
    python check.py --fast       # the inner loop (~1 min): unit, golden master, the seam,
                                 # the unified engine, the click and panel probes
    python check.py --jobs=1     # one at a time (the old behaviour, for a slow machine)
    python check.py -v           # print every gate's output, not just its last line
    python check.py --bless-ui   # re-bless the UI baseline (ONLY after an agreed,
                                 # legitimate UI change -- in the SAME change, not
                                 # "later"; mention re-blessing in the session log)

HOW IT RUNS (2026-09-22).  The gates are independent child processes and the
machine has cores to spare, so they run SEVERAL AT A TIME -- except the ones that
MEASURE wall-clock time (a block budget, a real-time replay), which are skipped in
that wave (CASYNTH_SKIP_TIMING, see tests/timing_gate.py) and run afterwards,
alone, with nothing else on the cores.  A p99 measured against five other test
processes measures the scheduler, not the engine.  Every gate is timed and the
slowest are printed at the end.

Gates (any FAIL -> exit 1):
  1. unit tests        python tests/test_casynth_core.py          (core invariants)
  1b. demo lab tests   python tests/test_demo_lab.py              (S1-S3 demo bench)
  1c. demo lab S4      python tests/test_demo_lab_s4.py           (catalog / replay)
  1d. demo lab S5      python tests/test_demo_lab_s5.py           (snapshot / continue)
  1e. demo lab S6      python tests/test_demo_lab_s6.py           (provenance / versions)
  1f. demo lab S7      python tests/test_demo_lab_s7.py           (catalog check / report)
  1g. demo lab S/N     python tests/test_demo_lab_sn.py           (Scan / Network engines)
  1h. network ref N0   python tests/test_network_reference_n0.py  (gutterOsc port == engine)
  1i. network N1       python tests/test_gutter_field_n1.py       (live kernel == model, bench)
  1j. network N2       python tests/test_n2_events.py             (periodic readout, event network)
  1k. network N3       python tests/test_n3_tuned_events.py       (tuned banks struck by events)
  1l. objects N4       python tests/test_n4_object_resonators.py (figure banks, circular detectors)
  1m. objects/Laplace  python tests/test_objects_laplace.py       (Laplace law in Objects, tail rules, side gain)
  1n. objects radius/attack python tests/test_objects_radius_attack.py (radius range, full masks, attack)
  1o. objects events/modal python tests/test_objects_event_source.py (Births / Deaths, Birth position)
  1p. objects birth strength python tests/test_objects_birth_strength.py (Birth strength law, M2)
  1q. objects decay law python tests/test_objects_decay.py          (history -> losses, D1-D3, scene script)
  1r. laplace carriers python tests/test_laplace_carriers.py      (carrier filter / wave bank, band limit)
  1s. laplace FM       python tests/test_laplace_fm.py            (phase law, band / DC, records)
  1t. the seam         python tests/test_seam.py                  (engines package, contract,
                       shared ring / panel, a scene that plays a note)
  1u. GEN envelope     python tests/test_gen_envelope.py          (the host's live GEN
                       A/D/S/R + tempo reach every engine that claims them, reach no
                       engine that does not, and an untold engine is byte-identical)
  1v. unified Laplace  python tests/test_laplace_unified.py       (articulation x
                       voicing on one spectrum: the axes, the four byte anchors of the
                       engines it collapsed, and an axis switch that never clicks)
  1w. live budget      python tests/test_live_budget.py           (the two cases the
                       user's own sessions caught on 2026-09-22 -- switching `artic` to
                       Events, and dragging `harm` -- must come in time on his field
                       and his knobs, and the paths that were fine must stay fine;
                       MEASURES TIME)
  2. golden master     python tests/golden/golden_master.py       (audio, byte-exact)
  3. ui frame + smoke  CASYNTH_DUMPFRAME render of gol_synth.py   (pixel-exact vs
                       tests/golden/ui_frame.png; doubles as the import/init smoke)
  3b. ui clicks        python tests/ui_click_probe.py             (every branch of the
                       prototype's mouse chain: buttons, sliders, knob rows, tabs,
                       piano, painting -- the frame dump sends no events at all)
  3c. it makes a sound python tests/ui_sound_probe.py             (Random + Play with the
                       MOUSE on every engine the prototype offers, then: did the level
                       move and was anything recorded -- the question a player asks)
  3e. panel rows       python tests/ui_panel_probe.py             (a knob that decides
                       whether ANOTHER row acts -- Laplace+ `shape` over `dyn` --
                       must make that row appear, without clicking anything else)
  3f. save an experiment python tests/ui_save_probe.py           (play PAST the
                       recorder's rolling window, click Save, and ask the BENCH
                       whether the record it wrote is one it can list, replay
                       byte-exact and continue from -- while the instrument keeps
                       drawing frames throughout the render, which it did not
                       until 2026-09-22)
  3d. articulation     python tests/ui_articulation_probe.py      (letting a key go
                       releases the note; HOLD lit keeps it sounding; and a played
                       MIDI line re-articulates note by note although its gate
                       never comes up)

Keep stdout ASCII-only: the default Windows console codepage (cp1251) chokes on
fancy glyphs, and agents run this a lot.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_REF = os.path.join(ROOT, "tests", "golden", "ui_frame.png")
UI_OUT = os.path.join(ROOT, "artifacts", "_ui_check.png")
DEFAULT_JOBS = max(1, min(6, (os.cpu_count() or 2) // 3))
COOLDOWN_S = 6.0            # idle before each gate that measures time
# CASYNTH_VOLUME: the probes drive the real prototype, which opens the real sound
# card -- a gate run must not play music at the person sitting there (user,
# 2026-09-22).  1% is audible to a meter and inaudible in the room.
DUMMY = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYTHONUTF8": "1",
         "CASYNTH_VOLUME": "0.01"}

TIMES = {}
_T0 = time.perf_counter()


def _compare_ui(ref_path, out_path):
    """Pixel-exact comparison of two frame dumps (ignores PNG encoder details)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import numpy as np
    import pygame
    a = pygame.surfarray.array3d(pygame.image.load(ref_path))
    b = pygame.surfarray.array3d(pygame.image.load(out_path))
    if a.shape != b.shape:
        print(f"[FAIL] ui frame: size differs ref={a.shape[:2]} out={b.shape[:2]}")
        return False
    if not np.array_equal(a, b):
        diff = int(np.count_nonzero((a != b).any(axis=2)))
        print(f"[FAIL] ui frame: {diff} pixels differ (dump kept at {out_path};"
              f" if the UI change is legitimate, run: python check.py --bless-ui)")
        return False
    print("[PASS] ui frame: pixel-exact vs tests/golden/ui_frame.png")
    return True


class Gate:
    """One child process.

    timing : the names of the tests in this file that MEASURE time (unittest -k
             patterns).  They are skipped in the parallel wave and run afterwards,
             alone; None means the file has none.  `timing=()` means the whole file
             is re-run alone (a file with its own runner, which knows no -k).
    fast   : part of the --fast inner loop.
    """
    __slots__ = ('label', 'name', 'argv', 'env', 'timeout', 'timing', 'fast')

    def __init__(self, label, name, argv, env=None, timeout=300, timing=None, fast=False):
        self.label = label
        self.name = name
        self.argv = list(argv)
        self.env = dict(DUMMY if env is None else env)
        self.timeout = timeout
        self.timing = timing
        self.fast = fast


def _t(name):
    return [sys.executable, os.path.join("tests", name)]


def gates():
    py = sys.executable
    return [
        Gate("unit tests", "unit tests", _t("test_casynth_core.py"), env={}, fast=True),
        Gate("demo lab tests (S1-S3)", "demo lab tests", _t("test_demo_lab.py"), timeout=600),
        Gate("demo lab tests (S4 catalog)", "demo lab S4 tests", _t("test_demo_lab_s4.py"),
             timeout=900),
        Gate("demo lab tests (S5 continue)", "demo lab S5 tests", _t("test_demo_lab_s5.py"),
             timeout=900),
        Gate("demo lab tests (S6 versions)", "demo lab S6 tests", _t("test_demo_lab_s6.py"),
             timeout=900),
        Gate("demo lab tests (S7 catalog check)", "demo lab S7 tests",
             _t("test_demo_lab_s7.py"), timeout=900),
        Gate("demo lab tests (S/N engines)", "demo lab S/N tests", _t("test_demo_lab_sn.py"),
             timeout=900),
        Gate("network reference N0", "N0 network reference tests",
             _t("test_network_reference_n0.py"), timeout=600),
        Gate("network N1 (gutter_field)", "N1 gutter_field tests",
             _t("test_gutter_field_n1.py"), timeout=600, timing=()),
        Gate("N1 hypothesis experiments", "N1 hypothesis tests", _t("test_n1_hypotheses.py"),
             timeout=600),
        Gate("network N2 (events)", "N2 event-network tests", _t("test_n2_events.py"),
             timeout=600, timing=("test_engine_timing_budget",)),
        Gate("network N3 (tuned events)", "N3 tuned-events tests",
             _t("test_n3_tuned_events.py"), timeout=600,
             timing=("test_engine_timing_budget",)),
        Gate("objects N4 (figure resonators)", "N4 object-resonator tests",
             _t("test_n4_object_resonators.py"), timeout=600,
             timing=("test_engine_timing_budget_on_both_scenes",)),
        Gate("objects / Laplace (spectrum law, tails, side gain)", "Objects / Laplace tests",
             _t("test_objects_laplace.py"), timeout=600,
             timing=("test_block_budget_of_both_sides_on_the_three_scenes",)),
        Gate("objects radius / attack (radius range, full masks, attack)",
             "Objects radius / attack tests", _t("test_objects_radius_attack.py"), timeout=900),
        Gate("objects events / modal excitation (Births / Deaths, Birth position)",
             "Objects event-source tests", _t("test_objects_event_source.py"), timeout=900),
        Gate("objects birth strength (the force of the birth position, M2)",
             "Objects birth-strength tests", _t("test_objects_birth_strength.py"), timeout=900),
        Gate("objects decay law (losses from the history of the cells, D1-D3)",
             "Objects decay-law tests", _t("test_objects_decay.py"), timeout=900),
        Gate("laplace carriers (the carrier filter / the wave bank on the Laplacian modes)",
             "Laplace carriers tests", _t("test_laplace_carriers.py"), timeout=900,
             timing=("test_block_budget_of_the_six_scenes",)),
        Gate("laplace FM (the modes of a figure modulate its one carrier)",
             "Laplace FM tests", _t("test_laplace_fm.py"), timeout=900,
             timing=("KnobDragBudget", "test_block_budget_of_the_two_new_scenes")),
        Gate("the seam (engines package, contract, shared ring / panel, articulated scene)",
             "seam tests", _t("test_seam.py"), timeout=600, fast=True),
        Gate("GEN envelope knobs reach the engines that claim them", "GEN envelope tests",
             _t("test_gen_envelope.py"), timeout=600),
        Gate("the unified Laplace engine (axes x voicings, the byte anchors)",
             "unified Laplace tests", _t("test_laplace_unified.py"), timeout=900, fast=True),
        Gate("the instrument comes in time on the fields the user played",
             "live budget", _t("test_live_budget.py"), timeout=600, timing=()),
        Gate("a spectrum knob does not decompose the field again (counted, not timed)",
             "knob drag work", _t("test_knob_drag_work.py"), timeout=300, fast=True),
        Gate("the tail pool is not walked slot by slot in Python (counted, not timed)",
             "slot pool work", _t("test_slot_pool_work.py"), timeout=300, fast=True),
        Gate("the wave-table pool holds every wave a standing field sounds (counted)",
             "wave pool holds the field", _t("test_wave_pool_holds_the_field.py"),
             timeout=300, fast=True),
        Gate("switching an axis compiles nothing on the render thread (counted)",
             "events kernels warm", _t("test_events_kernels_warm.py"), timeout=300, fast=True),
        Gate("a full garbage collection cannot stall the instrument",
             "gc pause", _t("test_gc_pause.py"), timeout=300, fast=True),
        Gate("a frame in which nothing changed costs almost nothing to draw (counted)",
             "frame work", _t("test_frame_work.py"), timeout=300, fast=True),
        Gate("the automaton keeps its tempo on the audio clock while the render thread stalls",
             "generation clock", _t("test_generation_clock.py"), timeout=300, timing=()),
        Gate("golden master", "golden master",
             [py, os.path.join("tests", "golden", "golden_master.py")], env={}, fast=True),
        Gate("events master (the Events cells of Laplace+, byte-exact)", "events master",
             [py, os.path.join("tests", "golden", "events_master.py")], env={}, fast=True),
        Gate("ui clicks (the prototype's event loop)", "ui click probe",
             _t("ui_click_probe.py"), timeout=300, fast=True),
        Gate("it makes a sound (Random + Play on every engine)", "sound probe",
             _t("ui_sound_probe.py"), timeout=900),
        Gate("the panel follows the knobs (a row that starts acting appears)",
             "panel probe", _t("ui_panel_probe.py"), timeout=300, fast=True),
        Gate("Save turns a session into an experiment the bench can open", "save probe",
             _t("ui_save_probe.py"), timeout=600),
        Gate("the VOICE envelope is audible (note off, HOLD, a played line)",
             "articulation probe", _t("ui_articulation_probe.py"), timeout=900),
    ]


def _spawn(name, argv, env, timeout, skip_timing):
    """One child, its output captured whole so parallel gates do not interleave."""
    e = dict(os.environ)
    e.update(env)
    if skip_timing:
        e['CASYNTH_SKIP_TIMING'] = '1'
    t0 = time.perf_counter()
    try:
        r = subprocess.run(argv, cwd=ROOT, env=e, timeout=timeout, capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        ok, out = (r.returncode == 0), (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired as exc:
        ok = False
        out = f"timed out after {timeout}s\n{(exc.stdout or b'')!r}"
    dt = time.perf_counter() - t0
    TIMES[name] = TIMES.get(name, 0.0) + dt
    return ok, out, dt


def _report(name, ok, out, dt, verbose):
    print(f"--- {name} ---")
    if not ok or verbose:
        print(out.rstrip())
    else:
        tail = [l for l in out.splitlines() if l.strip()]
        if tail:
            print(tail[-1])
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({dt:.1f}s)", flush=True)


def main():
    bless_ui = "--bless-ui" in sys.argv
    fast = "--fast" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    jobs = DEFAULT_JOBS
    for a in sys.argv[1:]:
        if a.startswith("--jobs="):
            jobs = max(1, int(a.split("=", 1)[1]))
    py = sys.executable
    results = []
    chosen = [g for g in gates() if g.fast or not fast]

    # -- wave 1: the tests that MEASURE TIME, alone, on a machine nobody is using -
    # They come first on purpose: run them after two hundred seconds of six busy
    # cores and the same code measures 3.3 ms or 5.6 ms depending on how hot the
    # package got (measured 2026-09-22).
    if jobs > 1 and len(chosen) > 1:
        timed = [g for g in chosen if g.timing is not None]
        if timed:
            print(f"--- {len(timed)} timed gates first, alone ---", flush=True)
        for g in timed:
            # ... and a breath between them.  These gates assert milliseconds, and
            # a package that has just run a minute of dense arithmetic clocks lower
            # than a cold one: the same FM drag measures 2.96 ms rested and 5.6 ms
            # in a row of heavy neighbours (2026-09-22).  COOLDOWN is not a fix for
            # slow code, it is what makes the measurement mean what it says.
            time.sleep(COOLDOWN_S)
            argv = list(g.argv)
            for pattern in g.timing:
                argv += ["-k", pattern]
            ok, out, dt = _spawn(g.name + " (timed)", argv, g.env, g.timeout, False)
            _report(g.name + " (timed)", ok, out, dt, verbose)
            results.append((g.label + " -- the timed tests", ok))

    # -- wave 2: every gate, several at a time, with the timed tests skipped ----
    if jobs > 1 and len(chosen) > 1:
        print(f"--- {len(chosen)} gates on {jobs} processes; the timed tests follow, alone ---",
              flush=True)
        with cf.ThreadPoolExecutor(jobs) as pool:
            futures = {pool.submit(_spawn, g.name, g.argv, g.env, g.timeout, True): g
                       for g in chosen}
            for fut in cf.as_completed(futures):
                g = futures[fut]
                ok, out, dt = fut.result()
                _report(g.name, ok, out, dt, verbose)
                results.append((g.label, ok))
    else:
        for g in chosen:
            ok, out, dt = _spawn(g.name, g.argv, g.env, g.timeout, False)
            _report(g.name, ok, out, dt, verbose)
            results.append((g.label, ok))

    # -- the UI frame: a child renders it, this process compares the pixels ----
    if not fast or bless_ui:
        target = UI_REF if bless_ui else UI_OUT
        os.makedirs(os.path.dirname(target), exist_ok=True)
        dump_ok, out, dt = _spawn(
            "ui dump + smoke", [py, "gol_synth.py"],
            dict(DUMMY, CASYNTH_DUMPFRAME=target), 300, False)
        _report("ui dump + smoke", dump_ok, out, dt, verbose)
        if bless_ui:
            results.append(("ui baseline", dump_ok))
            if dump_ok:
                print(f"[blessed] new UI baseline written to {UI_REF}")
        elif dump_ok and os.path.exists(UI_REF):
            results.append(("ui frame", _compare_ui(UI_REF, UI_OUT)))
        elif dump_ok:
            results.append(("ui frame", False))
            print(f"[FAIL] ui frame: baseline missing at {UI_REF} "
                  f"(bless one with: python check.py --bless-ui)")
        else:
            results.append(("ui dump + smoke", False))

    print("--- summary ---")
    failed = [n for (n, ok) in results if not ok]
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    wall = time.perf_counter() - _T0
    print(f"--- time: {wall:.0f}s wall ({sum(TIMES.values()):.0f}s of gate time), "
          f"slowest first ---")
    for n, t in sorted(TIMES.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {t:6.1f}s  {n}")
    if failed:
        print(f"RESULT: FAIL ({', '.join(failed)})")
        return 1
    print("RESULT: ALL GATES PASS" + (" (--fast subset)" if fast else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
