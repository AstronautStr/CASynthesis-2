#!/usr/bin/env python3
"""Run the regression gates in one command.

    python check.py              # AFTER EVERY CHANGE: the fast set + the gates the
                                 # files changed since HEAD can affect (~0.5-2 min)
    python check.py --all        # BEFORE A COMMIT: every gate, the timed ones alone
                                 # first (~4 min)
    python check.py --fast       # the fast set only (~30 s)
    python check.py --dry-run    # say which gates a run would pick, run nothing
    python check.py --since=REV  # "changed" = against REV instead of HEAD (a commit
                                 # just made: --since=HEAD~1)
    python check.py --files=a,b  # "changed" = these paths (what would this touch?)
    python check.py --jobs=1     # one at a time (the old behaviour, for a slow machine)
    python check.py -v           # print every gate's output, not just its last line
    python check.py --bless-ui   # re-bless the UI baseline (ONLY after an agreed,
                                 # legitimate UI change -- in the SAME change, not
                                 # "later"; mention re-blessing in the session log)

WHICH GATES (2026-09-23).  The default run picks gates by the changed files: a
map in this file (an engine -> the gates on its law + the gates that walk every
engine; a bench module -> the bench tests; a prototype module -> the probes and
the UI frame; a core module -> everything) plus every gate whose script names
the changed file (a scene, a builder, a golden reference; a catalog RECORD is a
listener's data and runs nothing).  The map over-selects on purpose and a file
it does not know runs everything; what it still misses, --all catches before
the commit.  See gates_for().

HOW IT RUNS (2026-09-22).  The gates are independent child processes and the
machine has cores to spare, so they run SEVERAL AT A TIME -- except the ones that
MEASURE wall-clock time (a block budget, a real-time replay), which are skipped in
that wave (CASYNTH_SKIP_TIMING, see tests/timing_gate.py) and run afterwards,
alone, with nothing else on the cores.  A p99 measured against five other test
processes measures the scheduler, not the engine.  Every gate is timed and the
slowest are printed at the end.

ORDER (2026-09-23).  The parallel wave starts its LONGEST gates first: a pool
that takes the list as written starts the two-minute gates last and then waits
for them alone (measured: 205 s for a wave whose lower bound was 153 s).  The
lengths come from the previous run (artifacts/gate_times.json, written at the
end of every full run) and, before there is one, from the estimate each gate
carries.  The timed wave sleeps COOLDOWN_S only after a gate that actually
heated the cores (ran longer than COOLDOWN_AFTER_S) and before the first one;
a six-second breath after a one-second test cooled nothing.  The wave's
processes run OpenBLAS on BLAS_THREADS_IN_WAVE threads, or its per-core buffer
reservations exhaust the commit charge of a 16 GB machine (see the constant).
Measured 2026-09-23: 304 s -> 230 s, the same two known reds, byte-exact gates
unchanged.

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
  2b. tail pool master python tests/golden/tail_pool_master.py    (the SlotPool under
                       OVERLOAD: the budget's eviction and the steal of a ringing
                       tail, byte-exact against a reference blessed on the code
                       BEFORE the vectorised pool -- golden_master never fills it)
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
  3g. fidelity         python tests/ui_fidelity_probe.py          (what the instrument
                       PLAYED is what its session renders offline, byte for byte:
                       a Random + Play + note session recorded from the real
                       prototype, turned into a scene as Save does, rendered
                       through the bench runner -- the live loop's generation
                       clock, VCA, gain glide and hosting, held at once)
  3d. articulation     python tests/ui_articulation_probe.py      (letting a key go
                       releases the note; HOLD lit keeps it sounding; opened
                       with HOLD dark and untouched, it is silent; and a played
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
COOLDOWN_S = 6.0            # idle before a gate that measures time ...
COOLDOWN_AFTER_S = 2.0      # ... when the gate before it ran longer than this
TIMES_FILE = os.path.join(ROOT, "artifacts", "gate_times.json")   # last run's seconds per gate
# OpenBLAS threads for the gates of the PARALLEL wave (and the child benches and
# workers they start, which inherit it).  OpenBLAS reserves its buffers for every
# core at load -- twenty threads' worth in each of the fifteen-odd processes the
# wave runs at once -- and on this 16 GB machine that is what "OpenBLAS error:
# Memory allocation still failed" in S7 / S/N / the articulation probe was
# (2026-09-23): the commit charge, not the physical memory.  The sound does not
# depend on it: the three byte-exact masters pass unchanged at 1 and at 2
# threads, and the full wave is the proof for the rest.  The timed wave keeps
# the machine's default -- it measures the instrument as it is played.
BLAS_THREADS_IN_WAVE = "2"
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
    expect : roughly how long it runs in the parallel wave, in seconds (the
             2026-09-22 measurement) -- only the ORDER of the wave depends on it,
             and only until a run has written artifacts/gate_times.json.
    """
    __slots__ = ('label', 'name', 'argv', 'env', 'timeout', 'timing', 'fast', 'expect')

    def __init__(self, label, name, argv, env=None, timeout=300, timing=None, fast=False,
                 expect=None):
        self.label = label
        self.name = name
        self.argv = list(argv)
        self.env = dict(DUMMY if env is None else env)
        self.timeout = timeout
        self.timing = timing
        self.fast = fast
        self.expect = float(EXPECT_S.get(name, 10.0) if expect is None else expect)


# Seconds each gate took in the parallel wave on 2026-09-22 (six processes, 20
# cores).  Only the START ORDER of the wave reads this, and only until the first
# full run has written TIMES_FILE; a gate missing here is assumed to take 10 s.
EXPECT_S = {
    'demo lab S7 tests': 123, 'demo lab tests': 119, 'save probe': 89, 'sound probe': 69,
    'demo lab S/N tests': 62, 'GEN envelope tests': 59, 'articulation probe': 50,
    'Objects decay-law tests': 50, 'Laplace FM tests': 47, 'demo lab S5 tests': 45,
    'demo lab S4 tests': 40, 'demo lab S6 tests': 30, 'Laplace carriers tests': 30,
    'Objects / Laplace tests': 13, 'ui click probe': 13, 'Objects birth-strength tests': 12,
    'live budget': 10, 'panel probe': 10, 'N4 object-resonator tests': 8,
    'N1 gutter_field tests': 6, 'Objects event-source tests': 6, 'N3 tuned-events tests': 5,
    'seam tests': 5, 'unified Laplace tests': 5, 'N1 hypothesis tests': 4,
    'Objects radius / attack tests': 4, 'N2 event-network tests': 3,
    'N0 network reference tests': 2, 'unit tests': 1, 'golden master': 1,
}


def _recorded_times():
    """{gate name: seconds} of the last full run, or {} (no file, or unreadable)."""
    try:
        import json
        with open(TIMES_FILE, encoding='utf-8') as f:
            got = json.load(f)
        return {str(k): float(v) for k, v in got.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _save_times():
    """Remember this run's seconds per gate for the next run's start order."""
    try:
        import json
        os.makedirs(os.path.dirname(TIMES_FILE), exist_ok=True)
        with open(TIMES_FILE, 'w', encoding='utf-8') as f:
            json.dump({k: round(v, 1) for k, v in TIMES.items()}, f, indent=1, sort_keys=True)
    except OSError as exc:
        print(f"(gate times not saved: {exc})")


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
        Gate("tail pool master (SlotPool under overload: eviction and steal, byte-exact)",
             "tail pool master",
             [py, os.path.join("tests", "golden", "tail_pool_master.py")], env={}, fast=True),
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
        Gate("what the instrument played is what its session renders (byte-exact)",
             "fidelity probe", _t("ui_fidelity_probe.py"), timeout=600),
    ]


# -- which gates a change touches (the default run, 2026-09-23) ------------------------
# `python check.py` with no flag runs the FAST set plus every gate that the files
# changed since HEAD can affect; `--all` runs everything and is what runs before
# a commit.  The map is a judgment, not an analysis: the tests reach the engines
# through the registry, by id, so no import graph can say which engine a test
# exercises.  A gate too many costs seconds, so the map over-selects on purpose;
# a gate too few is caught by --all.  A file the map does not know at all runs
# EVERYTHING, and says so.  Besides the map, a changed file's name is looked
# for in the text of every gate's script (a scene, a catalog, a builder, a
# golden reference), and every gate that names it runs too.
UI_FRAME = 'ui frame'                   # the pixel gate, which is not a Gate object
PROTOTYPE = ['ui click probe', 'sound probe', 'panel probe', 'save probe', 'articulation probe',
             'fidelity probe', 'golden master', 'generation clock', 'frame work', 'gc pause',
             UI_FRAME]
BENCH = ['demo lab tests', 'demo lab S4 tests', 'demo lab S5 tests', 'demo lab S6 tests',
         'demo lab S7 tests', 'demo lab S/N tests']
EVERY_ENGINE = ['seam tests', 'GEN envelope tests', 'demo lab tests']   # they walk every engine
OBJECTS = ['N4 object-resonator tests', 'Objects / Laplace tests', 'Objects radius / attack tests',
           'Objects event-source tests', 'Objects birth-strength tests', 'Objects decay-law tests']
UNIFIED = ['unified Laplace tests', 'live budget', 'events master', 'events kernels warm',
           'wave pool holds the field', 'knob drag work', 'fidelity probe']
CATALOG_USERS = OBJECTS + ['Laplace carriers tests', 'Laplace FM tests', 'N1 hypothesis tests']
NETWORK = ['N1 gutter_field tests', 'N1 hypothesis tests', 'N2 event-network tests',
           'N3 tuned-events tests']
# the files everything stands on: any change runs every gate
CORE_FILES = frozenset((
    'casynth_core.py', 'casynth_engine.py', 'casynth_config.py', 'check.py', 'requirements.txt',
    'casynth_engines/__init__.py', 'casynth_engines/engine_api.py', 'casynth_engines/registry.py',
    'casynth_engines/legacy_engine.py', 'casynth_engines/render_pool.py',
    'casynth_lab/__init__.py', 'casynth_lab/runner.py', 'casynth_lab/scene.py',
    'casynth_lab/snapshot.py', 'tests/timing_gate.py'))
ENGINE_GATES = {   # casynth_engines/<module>.py, or its casynth_lab shim -> the gates on that law
    'gutter_field': NETWORK, 'gutter_field_n1_config': NETWORK, 'gutter_controls': NETWORK,
    'gutter_field_periodic': NETWORK[2:], 'periodic_readout': NETWORK[2:],
    'event_network': NETWORK[2:],
    'tuned_events': ['N3 tuned-events tests', 'N4 object-resonator tests'],
    'figures': OBJECTS + UNIFIED, 'object_resonators': OBJECTS + UNIFIED,
    'laplace_carriers': ['Laplace carriers tests', 'Laplace FM tests'] + UNIFIED,
    'laplace_fm': ['Laplace FM tests'] + UNIFIED,
    'laplace_unified': UNIFIED, 'unified_events': UNIFIED,
    'scan_surface': ['demo lab S/N tests'], 'pm_network': ['demo lab S/N tests'],
}
LAB_GATES = {      # casynth_lab/<module>.py that is not an engine
    'audio_out': BENCH + ['save probe'],
    'recorder': BENCH + CATALOG_USERS + ['save probe'],
    'offline_record': BENCH + CATALOG_USERS + ['save probe'],
    'catalog': BENCH + CATALOG_USERS + ['save probe'],
    'verify': BENCH + CATALOG_USERS + ['save probe'],
    'versions': BENCH + CATALOG_USERS,
    'provenance': BENCH + CATALOG_USERS + ['save probe'],
    'notes_window': ['demo lab tests', 'demo lab S/N tests'],
    'textedit': ['demo lab tests'],
}
ROOT_GATES = {     # <module>.py in the repo root
    'gol_synth': PROTOTYPE, 'casynth_ui': PROTOTYPE, 'casynth_midi': PROTOTYPE,
    'casynth_midifile': PROTOTYPE,
    'casynth_tuning': PROTOTYPE + ['unit tests'],
    'casynth_session': PROTOTYPE + ['unit tests'],
    'casynth_host': PROTOTYPE + ['seam tests', 'demo lab tests', 'demo lab S4 tests'],
    'casynth_panel': PROTOTYPE + BENCH + ['seam tests'],
    'casynth_textedit': PROTOTYPE + BENCH,
    'patterns': PROTOTYPE + BENCH + ['events master'],
    'demo_bench': BENCH + CATALOG_USERS + NETWORK,
}
# what runs no gate at all: notes, docs, launchers, pictures
QUIET_PREFIXES = ('memory/', 'docs/', 'obsidian/', '.claude/', '.agents/', '.obsidian/',
                  'artifacts/', 'output/', 'tmp/', 'frontiers_explainer/', 'laplacian_explainer/',
                  'demos/results/', 'build/', 'dist/')
QUIET_SUFFIXES = ('.md', '.bat', '.canvas', '.txt', '.code-workspace', '.gitignore', '.mid',
                  '.spec', '.pyc')
_SCRIPTS = {}


def _git_exe():
    import shutil
    g = shutil.which('git')
    if g:
        return g
    for cand in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe"):
        if os.path.isfile(cand):
            return cand
    return None


def changed_files(since=None):
    """Repo-relative paths (forward slashes) changed in the working tree against
    HEAD (staged, unstaged and untracked), or against `since` when given.
    None when git cannot answer."""
    exe = _git_exe()
    if exe is None:
        return None
    try:
        if since:
            r = subprocess.run([exe, 'diff', '--name-only', since], cwd=ROOT, capture_output=True,
                               text=True, encoding='utf-8', errors='replace', timeout=60)
            files = [l.strip() for l in r.stdout.splitlines() if l.strip()]
            u = subprocess.run([exe, 'ls-files', '--others', '--exclude-standard'], cwd=ROOT,
                               capture_output=True, text=True, encoding='utf-8', errors='replace',
                               timeout=60)
            files += [l.strip() for l in u.stdout.splitlines() if l.strip()]
        else:
            r = subprocess.run([exe, 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT,
                               capture_output=True, text=True, encoding='utf-8', errors='replace',
                               timeout=60)
            files = []
            for line in r.stdout.splitlines():
                if len(line) < 4:
                    continue
                path = line[3:].strip()
                if ' -> ' in path:                 # a rename: the new name
                    path = path.split(' -> ', 1)[1]
                files.append(path.strip('"'))
        if r.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sorted(set(f.replace('\\', '/') for f in files))


def _script_text(g):
    """The text of a gate's script, read once, with `#` comments and
    triple-quoted docstrings cut off: a name in prose ("the field of record X")
    is not a dependency, a name in code is."""
    import re
    path = g.argv[1] if len(g.argv) > 1 else ''
    if path not in _SCRIPTS:
        try:
            with open(os.path.join(ROOT, path), encoding='utf-8', errors='replace') as f:
                text = re.sub(r'("""|\'\'\')[\s\S]*?\1', '', f.read())
                _SCRIPTS[path] = re.sub(r'#[^\n]*', '', text)
        except OSError:
            _SCRIPTS[path] = ''
    return _SCRIPTS[path]


def _gates_naming(token, all_gates):
    """The gates whose script names `token` as a word (underscores count as
    boundaries: 'object_resonators' is found inside 'ca_object_resonators')."""
    import re
    pat = re.compile(r'(?<![A-Za-z0-9])' + re.escape(token) + r'(?![A-Za-z0-9])')
    return [g.name for g in all_gates if pat.search(_script_text(g))]


def gates_for(path, all_gates):
    """(gate names, note) for one changed file; None as names means EVERYTHING."""
    p = path.replace('\\', '/')
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    if p in CORE_FILES:
        return None, 'core: everything runs'
    if p.startswith(QUIET_PREFIXES) or p.endswith(QUIET_SUFFIXES):
        return [], ''
    names = []
    if p.startswith('casynth_engines/') or p.startswith('casynth_lab/'):
        # the map only: an engine's name turns up in unrelated tests ("5 rows
        # since laplace_fm"), and the gates that walk every engine are listed
        if stem in ENGINE_GATES:
            names += ENGINE_GATES[stem] + EVERY_ENGINE
        elif stem in LAB_GATES:
            names += LAB_GATES[stem]
        else:
            return None, 'not in the map: everything runs'
    elif '/' not in p and p.endswith('.py'):
        if stem in ROOT_GATES:
            names += ROOT_GATES[stem] + _gates_naming(stem, all_gates)
        else:
            return None, 'not in the map: everything runs'
    elif p.startswith('tests/'):
        for g in all_gates:
            if len(g.argv) > 1 and g.argv[1].replace('\\', '/') == p:
                names.append(g.name)
        if base == 'ui_frame.png':
            names.append(UI_FRAME)
        names += _gates_naming(base, all_gates)      # a reference file
        if p.endswith('.py'):
            names += _gates_naming(stem, all_gates)  # a helper module the tests import
    elif p.startswith('lab_catalog/'):
        # a record (a listening experiment) is the listener's data, not code: the
        # gates that replay delivered records name them by id (2026-09-23), so a
        # record saved into any catalog runs nothing
        return [], ''
    elif p.startswith('demos/'):
        parts = p.split('/')
        token = stem if len(parts) == 2 else parts[1]  # a scene / builder, or a reference dir
        names += _gates_naming(token, all_gates)     # the tests that load or import it
    else:
        return None, 'not in the map: everything runs'
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, ''


def select(files, all_gates):
    """-> (set of gate names or None for everything, [(file, names, note)])."""
    chosen, plan, everything = set(), [], False
    for f in files:
        names, note = gates_for(f, all_gates)
        plan.append((f, names, note))
        if names is None:
            everything = True
        else:
            chosen.update(names)
    return (None if everything else chosen), plan


def _spawn(name, argv, env, timeout, skip_timing):
    """One child, its output captured whole so parallel gates do not interleave."""
    e = dict(os.environ)
    e.update(env)
    if skip_timing:
        e['CASYNTH_SKIP_TIMING'] = '1'
        e.setdefault('OPENBLAS_NUM_THREADS', BLAS_THREADS_IN_WAVE)
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
    run_all = "--all" in sys.argv
    dry = "--dry-run" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    jobs = DEFAULT_JOBS
    since, files = None, None
    for a in sys.argv[1:]:
        if a.startswith("--jobs="):
            jobs = max(1, int(a.split("=", 1)[1]))
        elif a.startswith("--since="):
            since = a.split("=", 1)[1]
        elif a.startswith("--files="):
            files = [f for f in a.split("=", 1)[1].split(",") if f]
    py = sys.executable
    results = []
    every = gates()
    want_ui = (not fast) or bless_ui
    if run_all or fast:
        chosen = [g for g in every if g.fast or not fast]
    else:
        # the default: the fast set + what the changed files touch
        if files is None:
            files = changed_files(since)
        if files is None:
            print("--- git could not say what changed: everything runs ---")
            chosen = list(every)
        else:
            picked, plan = select(files, every)
            quiet = [f for f, names, _n in plan if names == []]
            print(f"--- changed{' since ' + since if since else ''}: {len(files)} file(s)"
                  f"{f', {len(quiet)} of them run no gate (notes, docs, launchers)' if quiet else ''}"
                  " ---")
            for f, names, note in plan:
                if names == []:
                    continue
                print(f"  {f} -> {note if names is None else ', '.join(names)}")
            if picked is None:
                chosen = list(every)
                print("--- everything runs ---")
            else:
                chosen = [g for g in every if g.fast or g.name in picked]
                want_ui = bless_ui or UI_FRAME in picked
                extra = [g.name for g in chosen if not g.fast]
                print(f"--- the fast set + {len(extra)} gate(s)"
                      f"{' + the UI frame' if want_ui else ''}; "
                      f"`python check.py --all` runs everything (before a commit) ---")
                if not files:
                    print("    (nothing changed against HEAD: the fast set only)")
        if dry:
            for g in chosen:
                print(f"  would run: {g.name}")
            return 0

    # -- wave 1: the tests that MEASURE TIME, alone, on a machine nobody is using -
    # They come first on purpose: run them after two hundred seconds of six busy
    # cores and the same code measures 3.3 ms or 5.6 ms depending on how hot the
    # package got (measured 2026-09-22).
    if jobs > 1 and len(chosen) > 1:
        timed = [g for g in chosen if g.timing is not None]
        if timed:
            print(f"--- {len(timed)} timed gates first, alone ---", flush=True)
        prev_dt = None
        for g in timed:
            # ... and a breath between them.  These gates assert milliseconds, and
            # a package that has just run a minute of dense arithmetic clocks lower
            # than a cold one: the same FM drag measures 2.96 ms rested and 5.6 ms
            # in a row of heavy neighbours (2026-09-22).  COOLDOWN is not a fix for
            # slow code, it is what makes the measurement mean what it says.  It is
            # owed after a gate that ran long enough to heat the cores, and before
            # the first one (whatever ran before check.py is unknown); a gate that
            # was over in a second left nothing to cool (2026-09-23).
            if prev_dt is None or prev_dt > COOLDOWN_AFTER_S:
                time.sleep(COOLDOWN_S)
            argv = list(g.argv)
            for pattern in g.timing:
                argv += ["-k", pattern]
            ok, out, dt = _spawn(g.name + " (timed)", argv, g.env, g.timeout, False)
            _report(g.name + " (timed)", ok, out, dt, verbose)
            results.append((g.label + " -- the timed tests", ok))
            prev_dt = dt

    # -- wave 2: every gate, several at a time, with the timed tests skipped ----
    if jobs > 1 and len(chosen) > 1:
        # Longest first: the wave ends when its last long gate ends, so a long
        # gate that starts late is waited for alone.  Last run's seconds where
        # there are any, the estimates otherwise.
        recorded = _recorded_times()
        order = sorted(chosen, key=lambda g: -recorded.get(g.name, g.expect))
        src = "last run's times" if recorded else "estimated times"
        print(f"--- {len(chosen)} gates on {jobs} processes, longest first ({src}); "
              f"the timed tests ran alone ---", flush=True)
        with cf.ThreadPoolExecutor(jobs) as pool:
            futures = {pool.submit(_spawn, g.name, g.argv, g.env, g.timeout, True): g
                       for g in order}
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
    if want_ui:
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
    if run_all:
        _save_times()          # the next run starts its wave in this run's order
    if failed:
        print(f"RESULT: FAIL ({', '.join(failed)})")
        return 1
    print("RESULT: ALL GATES PASS" + (" (--fast subset)" if fast else
                                      "" if run_all else " (the gates of this change)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
