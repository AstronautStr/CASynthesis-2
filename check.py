#!/usr/bin/env python3
"""Run ALL regression gates in one command.

    python check.py              # unit tests + golden-master audio + UI frame + smoke
    python check.py --bless-ui   # re-bless the UI baseline (ONLY after an agreed,
                                 # legitimate UI change -- in the SAME change, not
                                 # "later"; mention re-blessing in the session log)

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
  2. golden master     python tests/golden/golden_master.py       (audio, byte-exact)
  3. ui frame + smoke  CASYNTH_DUMPFRAME render of gol_synth.py   (pixel-exact vs
                       tests/golden/ui_frame.png; doubles as the import/init smoke)
  3b. ui clicks        python tests/ui_click_probe.py             (every branch of the
                       prototype's mouse chain: buttons, sliders, knob rows, tabs,
                       piano, painting -- the frame dump sends no events at all)

Keep stdout ASCII-only: the default Windows console codepage (cp1251) chokes on
fancy glyphs, and agents run this a lot.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_REF = os.path.join(ROOT, "tests", "golden", "ui_frame.png")
UI_OUT = os.path.join(ROOT, "artifacts", "_ui_check.png")


def _run(name, argv, env=None, timeout=180):
    print(f"--- {name} ---")
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        r = subprocess.run(argv, cwd=ROOT, env=e, timeout=timeout)
        ok = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {name}: timed out after {timeout}s")
        return False
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


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


def main():
    bless_ui = "--bless-ui" in sys.argv
    py = sys.executable
    results = []

    results.append(("unit tests",
                    _run("unit tests", [py, os.path.join("tests", "test_casynth_core.py")])))

    results.append(("demo lab tests (S1-S3)",
                    _run("demo lab tests", [py, os.path.join("tests", "test_demo_lab.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                         timeout=300)))

    results.append(("demo lab tests (S4 catalog)",
                    _run("demo lab S4 tests", [py, os.path.join("tests", "test_demo_lab_s4.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=600)))

    results.append(("demo lab tests (S5 continue)",
                    _run("demo lab S5 tests", [py, os.path.join("tests", "test_demo_lab_s5.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=600)))

    results.append(("demo lab tests (S6 versions)",
                    _run("demo lab S6 tests", [py, os.path.join("tests", "test_demo_lab_s6.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("demo lab tests (S7 catalog check)",
                    _run("demo lab S7 tests", [py, os.path.join("tests", "test_demo_lab_s7.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("demo lab tests (S/N engines)",
                    _run("demo lab S/N tests", [py, os.path.join("tests", "test_demo_lab_sn.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("network reference N0",
                    _run("N0 network reference tests", [py, os.path.join("tests", "test_network_reference_n0.py")],
                         env={"PYTHONUTF8": "1"}, timeout=300)))

    results.append(("network N1 (gutter_field)",
                    _run("N1 gutter_field tests", [py, os.path.join("tests", "test_gutter_field_n1.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=600)))

    results.append(("N1 hypothesis experiments",
                    _run("N1 hypothesis tests", [py, os.path.join("tests", "test_n1_hypotheses.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=300)))

    results.append(("network N2 (events)",
                    _run("N2 event-network tests", [py, os.path.join("tests", "test_n2_events.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=600)))

    results.append(("network N3 (tuned events)",
                    _run("N3 tuned-events tests", [py, os.path.join("tests", "test_n3_tuned_events.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=600)))

    results.append(("objects N4 (figure resonators)",
                    _run("N4 object-resonator tests", [py, os.path.join("tests", "test_n4_object_resonators.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=600)))

    results.append(("objects / Laplace (spectrum law, tails, side gain)",
                    _run("Objects / Laplace tests", [py, os.path.join("tests", "test_objects_laplace.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"}, timeout=600)))

    results.append(("objects radius / attack (radius range, full masks, attack)",
                    _run("Objects radius / attack tests", [py, os.path.join("tests", "test_objects_radius_attack.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("objects events / modal excitation (Births / Deaths, Birth position)",
                    _run("Objects event-source tests", [py, os.path.join("tests", "test_objects_event_source.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("objects birth strength (the force of the birth position, M2)",
                    _run("Objects birth-strength tests", [py, os.path.join("tests", "test_objects_birth_strength.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("objects decay law (losses from the history of the cells, D1-D3)",
                    _run("Objects decay-law tests", [py, os.path.join("tests", "test_objects_decay.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("laplace carriers (the carrier filter / the wave bank on the Laplacian modes)",
                    _run("Laplace carriers tests", [py, os.path.join("tests", "test_laplace_carriers.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("laplace FM (the modes of a figure modulate its one carrier)",
                    _run("Laplace FM tests", [py, os.path.join("tests", "test_laplace_fm.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=900)))

    results.append(("the seam (engines package, contract, shared ring / panel, articulated scene)",
                    _run("seam tests", [py, os.path.join("tests", "test_seam.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=600)))

    results.append(("golden master",
                    _run("golden master", [py, os.path.join("tests", "golden", "golden_master.py")])))

    # UI frame dump doubles as the smoke test: it imports, inits pygame/audio,
    # renders exactly one frame and exits 0.
    target = UI_REF if bless_ui else UI_OUT
    os.makedirs(os.path.dirname(target), exist_ok=True)
    dump_ok = _run("ui dump + smoke", [py, "gol_synth.py"],
                   env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                        "CASYNTH_DUMPFRAME": target}, timeout=120)
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

    results.append(("ui clicks (the prototype's event loop)",
                    _run("ui click probe", [py, os.path.join("tests", "ui_click_probe.py")],
                         env={"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                              "PYTHONUTF8": "1"},
                         timeout=300)))

    print("--- summary ---")
    failed = [n for (n, ok) in results if not ok]
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if failed:
        print(f"RESULT: FAIL ({', '.join(failed)})")
        return 1
    print("RESULT: ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
