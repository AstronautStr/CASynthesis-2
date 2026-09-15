"""One reproducible run builds the N0 package  demos/results/network_reference_n0/

    python -m demos.network_reference_n0.render_n0            # everything (probes in parallel)
    python -m demos.network_reference_n0.render_n0 --quick    # skip probes (examples only)

Outputs
  01_palette.wav   three configurations x 8 s, 0.5 s pauses, each from the zero state
  02_control.wav   12 s: 4 s base, 4 s one slider moved, 4 s moved back (no restart)
  02_control_reference.wav   the same run without the intervention
  03_links.wav     8 s links OFF | 0.5 s pause | 8 s links ON, both continued from ONE saved state
  raw/*.f32.wav    float32 output before PCM conversion (16-bit files = raw * MASTER_GAIN)
  raw/probes/      float32 probe renders + probes.json / probes.md (measurements)
  manifest.json, README.md, config.json, verification_node.json (copied if present)
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from demos.network_reference_n0 import gutter_network as gn  # noqa: E402
from demos.network_reference_n0 import analysis as an  # noqa: E402
from demos.network_reference_n0 import source_manifest as sm  # noqa: E402

SR = 44100
OUT_DEFAULT = os.path.join(os.path.dirname(HERE), "results", "network_reference_n0")   # demos/results/...
SEED = 20260915
MASTER_GAIN = 0.85          # ONE fixed coefficient for every 16-bit file (raw sum of 8 nodes -> PCM); raw peaks ~0.85 -> ~-3 dBFS
PEAK_TARGET_MAX = 0.9       # the manifest reports the real peak; nothing adapts per file

# ---- probe plan: one axis at a time, raw slider units of the source patch ------------------
PROBE_T_SET = 4.0           # s: intervention (source line~ ramps apply)
PROBE_LEN = 12.0            # s
ANALYSIS_WINDOW = (6.0, 12.0)
PROBES = {
    "mod":         [0, 23, 92, 138, 184],       # gamma (forcing amplitude): base raw 46
    "damp":        [40, 90, 190, 230, 256],     # c base (damping): base raw 138
    "interaction": [0, 64, 190, 230, 256],      # link gain: base raw 127
}
BASE_RAW = dict(mod=46, damp=138, interaction=127)

# ---- examples ------------------------------------------------------------------------------
PALETTE = [  # (label, slider changes applied at t=0 with source ramps) -- chosen from the probe sweep
    ("A_base", {}),                        # resonant lines of the filter banks + low rumble, steady
    ("B_low_damping", {"damp": 40}),       # c = 0.024: sub-100 Hz growl, broadband, moving
    ("C_strong_links", {"interaction": 230}),  # link gain 4.0: slow amplitude breathing, spectrum sinks
]
CONTROL_PARAM = ("damp", 40)     # 02_control: base -> value at 4 s -> base at 8 s
LINKS_WARMUP = 4.0               # s of links-OFF warm-up before the saved state


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_wavs(out_dir, name, raw, master_gain):
    """raw: (n,2) float64 before master gain -> raw/<name>.f32.wav and <name>.wav (int16)."""
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    fp = os.path.join(out_dir, "raw", name + ".f32.wav")
    wavfile.write(fp, SR, raw.astype(np.float32))
    y = raw * master_gain
    peak = float(np.abs(y).max())
    pcm = np.round(np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    p16 = os.path.join(out_dir, name + ".wav")
    wavfile.write(p16, SR, pcm)
    return dict(file=name + ".wav", raw_file="raw/" + name + ".f32.wav", seconds=len(raw) / SR,
                peak_after_gain=peak, clipped_samples=int((np.abs(y) > 1.0).sum()),
                sha256=sha256(p16), raw_sha256=sha256(fp))


def silence(seconds=0.5):
    return np.zeros((int(seconds * SR), 2))


# ---------------------------------------------------------------------------- probe worker
def run_probe(args):
    cfg, axis, raw = args
    net = gn.GutterNetwork(cfg)
    n_set = int(PROBE_T_SET * SR); n_tot = int(PROBE_LEN * SR)
    a, _ = net.render(n_set)
    if axis is not None:
        net.set_slider(axis, raw)                 # ramped like the patch (line~)
    b, _ = net.render(n_tot - n_set)
    out = np.concatenate([a, b])
    return dict(axis=axis, raw=raw, audio=out.astype(np.float32), resets=int(net.resets.sum()),
                nan=bool(~np.isfinite(out).all()))


def run_probes(cfg, out_dir, workers):
    pdir = os.path.join(out_dir, "raw", "probes")
    os.makedirs(pdir, exist_ok=True)
    jobs = [(cfg, None, None)] + [(cfg, ax, v) for ax, vals in PROBES.items() for v in vals]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_probe, jobs))
    print(f"[probes] {len(jobs)} renders in {time.time() - t0:.0f} s")
    ctrl = res[0]
    w0, w1 = int(ANALYSIS_WINDOW[0] * SR), int(ANALYSIS_WINDOW[1] * SR)
    ref = an.mono(ctrl["audio"].astype(np.float64))[w0:w1]
    # warm-up from the control run: first 0.25 s frame after which the level stays within
    # +-3 dB of the median of the last 4 s (spectral centroid likewise within +-25 %)
    fr = an.frame_rms_db(an.mono(ctrl["audio"].astype(np.float64)))
    tail = np.median(fr[-16:])
    settle = None
    for i in range(len(fr)):
        if np.all(np.abs(fr[i:] - tail) <= 3.0):
            settle = i * 0.25
            break
    cm = an.mono(ctrl["audio"].astype(np.float64))
    cent_per_s = [round(an.spectral_features(*an.spectrum(cm[i * SR:(i + 1) * SR]))["centroid_hz"]) for i in range(int(PROBE_LEN))]
    report = dict(schedule=dict(t_set_s=PROBE_T_SET, length_s=PROBE_LEN, analysis_window_s=list(ANALYSIS_WINDOW)),
                  control=dict(resets=ctrl["resets"], nan=ctrl["nan"], frame_rms_db_250ms=[round(float(v), 2) for v in fr],
                               settle_time_s_level_within_3db=settle, centroid_hz_per_second=cent_per_s,
                               window_features=an.describe(ref)[0]),
                  probes=[])
    wavfile.write(os.path.join(pdir, "control.f32.wav"), SR, ctrl["audio"])
    for r in res[1:]:
        x = an.mono(r["audio"].astype(np.float64))[w0:w1]
        cmp_, dx, dr = an.compare(x, ref)
        fname = f"{r['axis']}_{r['raw']:03d}.f32.wav"
        wavfile.write(os.path.join(pdir, fname), SR, r["audio"])
        report["probes"].append(dict(axis=r["axis"], raw=r["raw"], mapped=_mapped(r["axis"], r["raw"]), file="raw/probes/" + fname,
                                     resets=r["resets"], nan=r["nan"], features=dx, vs_control=cmp_))
    with open(os.path.join(pdir, "probes.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    _probes_md(report, os.path.join(pdir, "probes.md"))
    return report


def _mapped(axis, raw):
    return {"mod": gn.map_mod, "damp": gn.map_damp, "interaction": gn.map_interaction}[axis](raw)


def _probes_md(rep, path):
    L = ["# N0 probes (one axis at a time; window %s s; control = same run without intervention)" % rep["schedule"]["analysis_window_s"], ""]
    c = rep["control"]["window_features"]
    L.append(f"Control window: rms {c['rms_db']:.1f} dB, centroid {c['centroid_hz']:.0f} Hz, flatness {c['flatness_db']:.1f} dB, "
             f"flux {c['spectral_flux']:.4f}, env std(250 ms) {c['env_std_db_250ms']:.2f} dB, crest {c['crest_db']:.1f} dB; "
             f"settle (level within 3 dB of the last 4 s): {rep['control']['settle_time_s_level_within_3db']} s; "
             f"control centroid per second (Hz): {rep['control']['centroid_hz_per_second']}")
    L += ["", "Readings: 'level' = |d level| > 3 dB; 'spectrum' = log-spectral distance (level removed) > 3 dB or centroid outside x0.8..x1.25; "
          "'time_behaviour' = spectral flux outside x0.67..x1.5 or |d env std| > 2 dB.  'pcm diff' is the RMS of (probe - control) re control: "
          "it is > 0 dB for every probe because trajectories diverge -- it says nothing about what changed."]
    L += ["", "| axis | raw | mapped | d level dB | log-spec dist dB | centroid x | flux x | d env std dB | pcm diff dB | resets | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in rep["probes"]:
        v = p["vs_control"]; rd = v["reading"]
        tags = [k for k in ("level", "spectrum", "time_behaviour") if rd[k]] or ["none"]
        L.append(f"| {p['axis']} | {p['raw']} | {p['mapped']:.4g} | {v['level_change_db']:+.1f} | {v['log_spectral_distance_db']:.1f} | "
                 f"{v['centroid_change_ratio']:.2f} | {v['flux_change_ratio']:.2f} | {v['env_std_change_db']:+.2f} | "
                 f"{v['pcm_difference_db_re_control']:+.1f} | {p['resets']} | {', '.join(tags)} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------- examples
def render_palette(cfg):
    segs, info = [], []
    for label, changes in PALETTE:
        net = gn.GutterNetwork(cfg)
        for k, v in changes.items():
            net.set_slider(k, v)                 # at t=0, ramped like the patch
        out, _ = net.render(int(8.0 * SR))
        segs += [out, silence()]
        info.append(dict(label=label, changes=changes, start_s=sum(len(s) for s in segs[:-2]) / SR, seconds=8.0,
                         resets=int(net.resets.sum()), features=an.describe(an.mono(out))[0]))
    return np.concatenate(segs[:-1]), info


def render_control(cfg):
    axis, raw = CONTROL_PARAM
    net = gn.GutterNetwork(cfg)
    a, _ = net.render(int(4.0 * SR))
    net.set_slider(axis, raw)
    b, _ = net.render(int(4.0 * SR))
    net.set_slider(axis, BASE_RAW[axis])
    c, _ = net.render(int(4.0 * SR))
    ref_net = gn.GutterNetwork(cfg)
    ref, _ = ref_net.render(int(12.0 * SR))
    out = np.concatenate([a, b, c])
    info = dict(axis=axis, raw_base=BASE_RAW[axis], raw_set=raw, mapped_base=_mapped(axis, BASE_RAW[axis]), mapped_set=_mapped(axis, raw),
                t_set_s=4.0, t_back_s=8.0, ramp_ms=cfg["ramp_ms"][axis], resets=int(net.resets.sum()), resets_reference=int(ref_net.resets.sum()),
                segments={k: an.describe(an.mono(out[int(s * SR):int(e * SR)]))[0] for k, (s, e) in dict(base=(0, 4), set=(4, 8), back=(8, 12)).items()},
                reference_segments={k: an.describe(an.mono(ref[int(s * SR):int(e * SR)]))[0] for k, (s, e) in dict(base=(0, 4), set=(4, 8), back=(8, 12)).items()})
    return out, ref, info


def render_links(cfg, out_dir, interaction_raw=None, tag=""):
    cfg = dict(cfg)
    if interaction_raw is not None:
        cfg["interaction_raw"] = interaction_raw
    cfg_off = dict(cfg); cfg_off["matrix"] = gn.preset_matrix(3).tolist()
    net = gn.GutterNetwork(cfg_off)
    net.render(int(LINKS_WARMUP * SR))
    st = net.export_state()
    np.savez(os.path.join(out_dir, "raw", f"links{tag}_saved_state.npz"), **st)
    a_net = gn.GutterNetwork.from_state(cfg_off, st)
    a, _ = a_net.render(int(8.0 * SR))
    b_net = gn.GutterNetwork.from_state(cfg_off, st)
    b_net.set_matrix(gn.preset_matrix(1), ramp=False)     # instant (the patch would ramp 2000 ms)
    b, _ = b_net.render(int(8.0 * SR))
    out = np.concatenate([a, silence(), b])
    info = dict(warmup_s=LINKS_WARMUP, saved_state=f"raw/links{tag}_saved_state.npz",
                saved_state_sha256=sha256(os.path.join(out_dir, "raw", f"links{tag}_saved_state.npz")),
                off=dict(start_s=0.0, seconds=8.0, resets=int(a_net.resets.sum()), features=an.describe(an.mono(a))[0]),
                on=dict(start_s=8.5, seconds=8.0, resets=int(b_net.resets.sum()), features=an.describe(an.mono(b))[0],
                        matrix="preset 1 (all-to-all, no self)", interaction_raw=cfg["interaction_raw"]),
                on_vs_off=an.compare(an.mono(b), an.mono(a))[0],
                node_dynamics_kept="both runs use the same node parameters and saved state; only matrix~ cells differ")
    return out, info


def figures(out_dir, manifest):
    """Spectrogram + waveform of the three examples and the probe grid (matplotlib, optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import stft
    except Exception as e:  # pragma: no cover
        print("[figures] skipped:", e)
        return []
    fdir = os.path.join(out_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    made = []

    def spec(ax, x, title):
        f, t, Z = stft(x, fs=SR, nperseg=2048, noverlap=1536)
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-100, vmax=-20, shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=100); ax.set_ylim(20, 16000); ax.set_title(title, fontsize=9)

    for name in ("01_palette", "02_control", "02_control_reference", "03_links", "03b_links_strong"):
        sr_, y = wavfile.read(os.path.join(out_dir, "raw", name + ".f32.wav"))
        x = an.mono(y.astype(np.float64))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw=dict(height_ratios=[3, 1]))
        spec(a1, x, name)
        a2.plot(np.arange(len(x)) / SR, x * MASTER_GAIN, lw=0.3); a2.set_ylim(-1, 1); a2.set_xlim(0, len(x) / SR)
        plt.tight_layout(); fp = os.path.join(fdir, name + ".png"); plt.savefig(fp, dpi=80); plt.close(fig); made.append("figures/" + name + ".png")
    pdir = os.path.join(out_dir, "raw", "probes")
    if os.path.isdir(pdir) and "probes" in manifest:
        items = [("control", os.path.join(pdir, "control.f32.wav"))] + [(f"{p['axis']} {p['raw']}", os.path.join(out_dir, p["file"])) for p in manifest["probes"]["probes"]]
        fig, axes = plt.subplots(len(items), 1, figsize=(12, 1.6 * len(items)))
        for ax, (title, fp) in zip(axes, items):
            sr_, y = wavfile.read(fp)
            spec(ax, an.mono(y.astype(np.float64)), title + "  (intervention at 4 s)")
        plt.tight_layout(); fp = os.path.join(fdir, "probes.png"); plt.savefig(fp, dpi=70); plt.close(fig); made.append("figures/probes.png")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--quick", action="store_true", help="skip the probe sweep")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    t_start = time.time()
    cfg = gn.default_config(seed=a.seed)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    manifest = dict(
        package="network_reference_n0", generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        command="python -m demos.network_reference_n0.render_n0" + (" --quick" if a.quick else ""),
        source=dict(url=sm.SOURCE_URL, commit=sm.SOURCE_COMMIT, license=sm.SOURCE_LICENSE, file_sha256=sm.SOURCE_HASHES,
                    executable_node="gutterOsc.class (top level) -- NOT gutterOsc.java"),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__,
                         platform=platform.platform(), machine=platform.machine()),
        sample_rate=SR, seed=a.seed, master_gain=MASTER_GAIN, config="config.json",
        model_constants=dict(matrix_delay_samples=cfg["matrix_delay_samples"], extra_feedback_samples=cfg["extra_feedback_samples"],
                             total_link_delay_samples=cfg["matrix_delay_samples"] + cfg["extra_feedback_samples"],
                             ramps_ms=cfg["ramp_ms"], svf_hp_hz=cfg["svf_hp_hz"], node_clip=cfg["node_clip"]),
        parameters=sm.PARAMS, files={}, examples={},
    )
    if not a.quick:
        manifest["probes"] = run_probes(cfg, out_dir, a.workers)
    pal, pal_info = render_palette(cfg)
    manifest["files"]["01_palette"] = write_wavs(out_dir, "01_palette", pal, MASTER_GAIN)
    manifest["examples"]["01_palette"] = pal_info
    ctl, ref, ctl_info = render_control(cfg)
    manifest["files"]["02_control"] = write_wavs(out_dir, "02_control", ctl, MASTER_GAIN)
    manifest["files"]["02_control_reference"] = write_wavs(out_dir, "02_control_reference", ref, MASTER_GAIN)
    manifest["examples"]["02_control"] = ctl_info
    lnk, lnk_info = render_links(cfg, out_dir)
    manifest["files"]["03_links"] = write_wavs(out_dir, "03_links", lnk, MASTER_GAIN)
    manifest["examples"]["03_links"] = lnk_info
    # explicitly labelled extra: the same OFF/ON pair with the link gain slider at 230 (source range 0..256)
    lnk2, lnk2_info = render_links(cfg, out_dir, interaction_raw=230, tag="_strong")
    manifest["files"]["03b_links_strong"] = write_wavs(out_dir, "03b_links_strong", lnk2, MASTER_GAIN)
    manifest["examples"]["03b_links_strong"] = lnk2_info
    manifest["peak_after_master_gain_max"] = max(v["peak_after_gain"] for v in manifest["files"].values())
    manifest["clipped_samples_total"] = sum(v["clipped_samples"] for v in manifest["files"].values())
    ver = os.path.join(out_dir, "verification_node.json")
    manifest["verification_node"] = "verification_node.json" if os.path.exists(ver) else None
    manifest["figures"] = figures(out_dir, manifest)
    manifest["render_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"done in {manifest['render_seconds']} s; peak after gain {manifest['peak_after_master_gain_max']:.3f}; clipped {manifest['clipped_samples_total']}")


if __name__ == "__main__":
    main()
