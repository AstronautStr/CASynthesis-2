"""Verify the Python node port against the ORIGINAL ``gutterOsc.class`` executed on a JVM.

The reference never calls the port: ``javaref/RefRunner.java`` drives the shipped class file
(with two tiny stand-ins for the Max Java API, ``javaref/com/cycling74/msp``) on input signal
arrays written by this script; the same arrays are fed to ``gutter_node.GutterOsc``.

    python -m demos.network_reference_n0.verify_node --source <clone of guttersynthesis> \
        --jdk <JDK bin dir> [--out demos/results/network_reference_n0/verification_node.json]

Compared per case: float32 outlets (bit-exact or 1 ulp), the double state trace of the first
samples (duffX, duffY, finalY, t, dx, dy), the final state, and block-split invariance
(block 1 / 64 / 512 in Java must be bit-identical).  Tolerances are argued from number
precision only: the port and the JVM share IEEE-754 double arithmetic with the same
statement order; the only known platform difference is the last-ulp behaviour of
``sin``/``atan``/``tan``/``exp`` in the Java intrinsics vs. the C runtime, so a relative
tolerance of 1e-12 on states is allowed and any drift beyond that in the first samples fails.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from demos.network_reference_n0.gutter_node import GutterOsc  # noqa: E402
from demos.network_reference_n0 import source_manifest as sm  # noqa: E402

SR = 44100.0
N = 6000          # samples per case
TRACE = 3000      # per-sample state trace compared

# Patch-like initial messages (see source_manifest.PARAMS for where the values come from)
FREQS = [50.0 + 4950.0 * v * v for v in np.linspace(0.02, 0.6, 24)]     # 24 fixed filter freqs
Q_ALL = 30.0


def _inputs(kind, rng):
    x = np.zeros((6, N), np.float32)
    if kind == "const":
        x[0] = 0.32; x[1] = 0.002; x[2] = 0.29; x[3] = 0.0027; x[4] = 2.2
    elif kind == "ramps":       # what line~ objects produce at load
        x[0] = np.linspace(0, 0.32, N); x[1] = np.linspace(0, 0.002, N); x[2] = np.linspace(0, 0.29, N)
        x[3] = np.linspace(0, 0.0027, N); x[4] = np.linspace(0, 2.2, N)
    elif kind == "noisy_c":     # damping input driven like the matrix link (bipolar, clipped 0.0001..1)
        x[0] = 0.32; x[1] = 0.002; x[3] = 0.0027; x[4] = 2.2
        x[2] = np.clip(0.29 + 0.8 * rng.standard_normal(N).astype(np.float32), 0.0001, 1.0)
    elif kind == "audio":
        x[0] = 0.5; x[1] = 0.002; x[2] = 0.29; x[3] = 0.0027; x[4] = 2.2
        x[5] = 0.3 * rng.standard_normal(N).astype(np.float32)
    return x


def _msgs_common():
    m = [("setLowpass", 6036.7), ("setHighpass", 5.0), ("filters", 24)]
    for i, f in enumerate(FREQS):
        m.append(("setFreqN", i, float(f)))
    for i in range(24):
        m.append(("setQN", i, Q_ALL))
    return m


CASES = {
    # name: (inputs kind, init msgs, in-DSP msgs {block: [msg...]}, description)
    "filters_on_atan":   ("const", _msgs_common(), {}, "patch default: 24 filters, distortion 2 (atan), highpass bypassed"),
    "ramps_at_load":     ("ramps", _msgs_common(), {}, "line~-style ramps on all inlets"),
    "noisy_damping":     ("noisy_c", _msgs_common(), {}, "damping inlet driven like the network link"),
    "audio_input":       ("audio", _msgs_common() + [("toggleAudioInput", 1)], {}, "gamma*audioInput forcing"),
    "highpass_active":   ("const", _msgs_common() + [("setHighpass", 20.0)], {}, "one-pole highpass in the loop"),
    "filters_off":       ("const", [("setLowpass", 6036.7), ("setHighpass", 5.0), ("toggleFilters", 0)], {}, "raw Duffing: clamp +-100, reset at |x|>99"),
    "filter_count_12":   ("const", _msgs_common() + [("filters", 12)], {}, "only the first 12 filters"),
    "live_messages":     ("const", _msgs_common(), {1024: [("setQN", 3, 120.0), ("setFreqN", 3, 1500.0)], 2048: [("setLowpass", 2000.0)], 4096: [("filters", 8)]}, "messages while DSP runs (at sample multiples of 512)"),
}
for mode in (0, 1, 3, 4, 5):
    CASES[f"dist_mode_{mode}"] = ("const", _msgs_common() + [("setDistortionMethod", mode)], {}, f"distortion method {mode}")


def _fmt(m):
    return " ".join(str(a) for a in m)


def run_java(jdk, source, build, spec_lines, inputs, prefix, block):
    inputs.astype("<f4").tofile(prefix + "_in.f32")
    with open(prefix + ".spec", "w") as fh:
        fh.write(f"sr {int(SR)}\nblock {block}\nnsamples {N}\ninputs {prefix}_in.f32\nout {prefix}\ntrace {TRACE}\n")
        fh.write("\n".join(spec_lines) + "\n")
    java = os.path.join(jdk, "java.exe" if os.name == "nt" else "java")
    cp = os.pathsep.join([build, source])
    r = subprocess.run([java, "-cp", cp, "RefRunner", prefix + ".spec"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    o0 = np.fromfile(prefix + "_out0.f32", "<f4")
    o1 = np.fromfile(prefix + "_out1.f32", "<f4")
    tr = np.fromfile(prefix + "_trace.f64", "<f8").reshape(-1, 6)
    st = {}
    for line in open(prefix + "_state.txt"):
        k, v = line.split()
        st[k] = float(v)
    return o0, o1, tr, st


def _live_blocks(live, block):
    """Messages keyed by SAMPLE index -> by block index (sample must sit on a block boundary)."""
    out = {}
    for smp, ms in live.items():
        assert smp % block == 0, (smp, block)
        out[smp // block] = ms
    return out


def run_port(x, init_msgs, live, block):
    live = _live_blocks(live, block)
    osc = GutterOsc()
    for m in init_msgs:
        getattr(osc, m[0])(*m[1:])
    osc.dspsetup(SR)
    o0 = np.zeros(N, np.float32); o1 = np.zeros(N, np.float32)
    tr = np.zeros((TRACE, 6))
    nb = (N + block - 1) // block
    for b in range(nb):
        for m in live.get(b, []):
            getattr(osc, m[0])(*m[1:])
        base = b * block
        for i in range(base, min(N, base + block)):
            o0[i], o1[i] = osc.perform_sample(*[float(x[r, i]) for r in range(6)])
            if i < TRACE:
                tr[i] = (osc.duffX, osc.duffY, osc.finalY, osc.t, osc.dx, osc.dy)
    st = {k: getattr(osc, k) for k in ("duffX", "duffY", "dx", "dy", "finalY", "t", "dt", "gamma", "omega", "c", "singleGain", "sampleRate")}
    return o0, o1, tr, st, osc.resets


def compile_ref(jdk, work):
    build = os.path.join(work, "build")
    os.makedirs(build, exist_ok=True)
    src = os.path.join(HERE, "javaref")
    javac = os.path.join(jdk, "javac.exe" if os.name == "nt" else "javac")
    r = subprocess.run([javac, "-d", build,
                        os.path.join(src, "com", "cycling74", "msp", "MSPSignal.java"),
                        os.path.join(src, "com", "cycling74", "msp", "MSPPerformer.java"),
                        os.path.join(src, "RefRunner.java")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return build


def ulp_diff(a, b):
    """Max distance in float32 ulps between two float32 arrays."""
    ia = a.view(np.int32).astype(np.int64); ib = b.view(np.int32).astype(np.int64)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0, np.abs(ia - ib))
    return int(np.max(d)) if a.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="checkout of tommmmudd/guttersynthesis (class files)")
    ap.add_argument("--jdk", required=True, help="JDK bin directory (javac + java)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "network_reference_n0", "verification_node.json"))
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix="n0_verify_")
    os.makedirs(work, exist_ok=True)
    hashes = sm.hash_source(a.source)
    build = compile_ref(a.jdk, work)
    rng = np.random.default_rng(20260915)
    report = {"source_commit": sm.SOURCE_COMMIT, "source_hashes_match": hashes == sm.SOURCE_HASHES,
              "sr": SR, "samples": N, "trace_samples": TRACE, "cases": {}, "all_pass": True}
    if not report["source_hashes_match"]:
        print("[WARN] source file hashes differ from the manifest:", {k: (v, sm.SOURCE_HASHES.get(k)) for k, v in hashes.items() if v != sm.SOURCE_HASHES.get(k)})
    for name, (kind, init, live, desc) in CASES.items():
        x = _inputs(kind, rng)
        def spec_for(block):
            return [f"msg {_fmt(m)}" for m in init] + [f"at {b} {_fmt(m)}" for b, ms in _live_blocks(live, block).items() for m in ms]
        j64 = run_java(a.jdk, a.source, build, spec_for(64), x, os.path.join(work, name + "_b64"), 64)
        p64 = run_port(x, init, live, 64)
        # block-split invariance (reference only, cheap): 1 and 512
        j1 = run_java(a.jdk, a.source, build, spec_for(1), x, os.path.join(work, name + "_b1"), 1)
        j512 = run_java(a.jdk, a.source, build, spec_for(512), x, os.path.join(work, name + "_b512"), 512)
        java_block_invariant = bool(np.array_equal(j64[0], j1[0], equal_nan=True) and np.array_equal(j64[1], j1[1], equal_nan=True)
                                    and np.array_equal(j64[0], j512[0], equal_nan=True) and np.array_equal(j64[1], j512[1], equal_nan=True))
        o0_ulp = ulp_diff(j64[0], p64[0]); o1_ulp = ulp_diff(j64[1], p64[1])
        tr_j, tr_p = j64[2], p64[2]
        # state error normalised by the magnitude of each state column (relative-to-signal);
        # NaN states (post-overflow) compare as equal when both are NaN
        both_nan = np.isnan(tr_j) & np.isnan(tr_p)
        err = np.where(both_nan, 0.0, np.abs(tr_j - tr_p))
        col_scale = np.maximum(np.nanmax(np.abs(np.where(np.isinf(tr_j), np.nan, tr_j)), axis=0), 1e-300)
        nerr = np.where(np.isinf(err), np.inf, err) / col_scale
        max_nerr = float(np.nanmax(nerr))
        diff_rows = np.where(np.nanmax(nerr, axis=1) > 0)[0]
        first_diff = int(diff_rows[0]) if diff_rows.size else -1
        st_j, st_p = j64[3], p64[3]
        st_rel = max(0.0 if (math.isnan(st_j[k]) and math.isnan(st_p[k])) or st_j[k] == st_p[k]
                     else abs(st_j[k] - st_p[k]) / max(abs(st_j[k]), 1e-30) for k in st_j)
        exact_o0 = bool(np.array_equal(j64[0], p64[0], equal_nan=True)); exact_o1 = bool(np.array_equal(j64[1], p64[1], equal_nan=True))
        # criterion: block-invariant reference; float32 outlets bit-exact (or <= 1 ulp);
        # normalised state error <= 1e-12 over the whole trace (pure IEEE arithmetic in the
        # same order should be exact; 1e-12 leaves room for a last-ulp libm difference in
        # sin/tan without letting a real algorithmic deviation through)
        # dist mode 5 uses Math.exp, a HotSpot intrinsic whose last ulp differs from the C runtime
        # (not used by the patch: default method is 2); it gets the libm-level tolerance only.
        libm_dep = name == "dist_mode_5"
        tol = 1e-9 if libm_dep else 1e-12
        ok = java_block_invariant and o0_ulp <= 1 and o1_ulp <= 1 and max_nerr <= tol and st_rel <= (1e-6 if libm_dep else 1e-9)
        report["cases"][name] = {
            "description": desc, "pass": bool(ok), "tolerance_normalised": tol, "libm_dependent": libm_dep,
            "java_block_invariant_1_64_512": java_block_invariant,
            "out0_bit_exact": exact_o0, "out1_bit_exact": exact_o1, "out0_max_ulp": o0_ulp, "out1_max_ulp": o1_ulp,
            "trace_max_normalised_err": max_nerr, "trace_first_differing_sample": first_diff,
            "final_state_max_rel_err": float(st_rel),
            "port_resets": int(p64[4]), "ref_out0_absmax": float(np.nanmax(np.abs(j64[0]))), "ref_out1_absmax": float(np.nanmax(np.abs(j64[1]))),
        }
        report["all_pass"] &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:18s} out0 ulp={o0_ulp} out1 ulp={o1_ulp} exact={exact_o0 and exact_o1} trace nerr={max_nerr:.2e} first_diff={first_diff} state rel={st_rel:.2e} block-inv={java_block_invariant}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("ALL PASS" if report["all_pass"] else "SOME CASES FAILED", "->", a.out)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
