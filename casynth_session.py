"""Session recording dump + offline replay for gol_synth.

The reproducibility contract: a real user session records (CASYNTH_RECORD=1) the
per-frame audio + the AUTHORITATIVE `replay_controls` snapshot of every synth
control; `replay_session` re-renders that snapshot OFFLINE through the SAME
analyse()/SlotPool/render path, so a reported bug reproduces on the user's exact
snapshot and a fix is validated against the recorded WAV (with unfixed code the
replay must match bit-for-bit).

This module holds the READ side (dump + replay + CLI).  The WRITE side (appending
to rec['replay_controls'] each rendered frame) lives in gol_synth.main(); the
shared piece is the schema, unchanged by this split.
"""
import numpy as np

from casynth_config import *
from casynth_engine import SlotPool, analyse, render_chunk_laplacian, midi_to_freq


def _dump_session(rec, prefix="_session"):
    import time
    from scipy.io import wavfile
    base = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    if rec['chunks']:
        audio = np.concatenate(rec['chunks'], axis=0)
        wavfile.write(f"{base}.wav", SR, audio)
    frames = np.array(rec['frames'], dtype=float) if rec['frames'] else np.zeros((0, 5))
    if rec['steps']:
        gens  = np.array([s[0] for s in rec['steps']])
        grids = np.stack([s[1] for s in rec['steps']])
        notes = np.array([s[2] for s in rec['steps']])
    else:
        gens, grids, notes = (np.zeros(0),
                              np.zeros((0, GRID_H, GRID_W), np.uint8), np.zeros(0))
    # Faithful-replay log (per rendered frame): inputs + chunk count.
    # replay (legacy fixed-width, laplacian-probe contract) columns:
    #   [n_rendered, note, spread, alpha, release_ms, vol, n_partials]
    # replay_controls: AUTHORITATIVE per-frame dict of EVERY synth control
    #   (engine id, note, vol, attack/decay/sustain/release, full engine_params) ->
    #   any control bug is reproducible from the recording; use this for faithful
    #   replay (it auto-captures every knob, incl. ones added after this matrix).
    # replay_engines: parallel array of the active engine id per rendered frame
    # (multi-engine; faithful replay must call the matching map_* per frame).
    replay = (np.array(rec.get('replay', []), dtype=float)
              if rec.get('replay') else np.zeros((0, 7)))
    replay_controls = np.array(rec.get('replay_controls', []), dtype=object)
    replay_engines = np.array(rec.get('replay_engines', []), dtype=object)
    replay_grids = (np.stack(rec['replay_grids']).astype(np.uint8)
                    if rec.get('replay_grids')
                    else np.zeros((0, GRID_H, GRID_W), np.uint8))
    # midi_onsets: sample-accurate note-timing log (cum_samples, note, gate) from
    # the render thread -> onset_time = col0 / SR.  Authoritative for jitter measurement.
    midi_onsets = (np.array(rec.get('midi_onsets', []), dtype=float)
                   if rec.get('midi_onsets') else np.zeros((0, 3)))
    # midi_in: device-arrival timing (perf_counter, note, gate) -- ground-truth input.
    midi_in_log = (np.array(rec.get('midi_in', []), dtype=float)
                   if rec.get('midi_in') else np.zeros((0, 3)))
    np.savez_compressed(f"{base}.npz", frames=frames, gens=gens, grids=grids,
                        notes=notes, sr=SR, chunk_s=CHUNK_S,
                        bpm=rec.get('bpm', BPM_DEFAULT),
                        div_idx=rec.get('div_idx', DIV_DEFAULT),
                        max_voices=MAX_VOICES, master_gain=MASTER_GAIN,
                        max_modes=MAX_MODES_PER_OBJ, patch_size=PATCH_SIZE,
                        underruns=rec['underruns'],
                        replay=replay, replay_grids=replay_grids,
                        replay_engines=replay_engines,
                        replay_controls=replay_controls,
                        midi_onsets=midi_onsets, midi_in=midi_in_log)
    # frames columns: [dt_ms, gen, n_voices, n_rendered, underrun]
    n_hitch = int((frames[:, 0] > CHUNK_S * 1000).sum()) if len(frames) else 0
    print(f"[session saved] {base}.wav ({len(rec['chunks'])} chunks) + {base}.npz "
          f"({len(rec['steps'])} steps, {len(midi_onsets)} midi onsets)  "
          f"underruns={rec['underruns']}")


# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE SESSION REPLAY  (read side of the session log -- see _dump_session)
# ──────────────────────────────────────────────────────────────────────────────
def replay_session(ts, prefix="_session"):
    """Re-render a recorded session (<prefix>_<ts>.npz) OFFLINE from the
    AUTHORITATIVE per-frame control log, deterministically -- so a reported bug
    reproduces on the user's exact snapshot and a fix is validated against it
    (with unfixed code the replay must match the recorded WAV).

    Uses `replay_controls` (object array -> needs allow_pickle), which captures
    the FULL control state per rendered frame (engine id + note + vol + ADSR +
    every engine_params knob, incl. shape).  Because it forwards the whole
    engine_params dict through the SAME analyse()/SlotPool/render path the live
    synth uses, replay tracks the engine exactly and needs no per-knob edits when
    a new control is added (only that the control is in the snapshot).

    Returns (audio int16 (N,2), pre_clip_peak, n_clipped).
    """
    import os
    npz = f"{prefix}_{ts}.npz"
    if not os.path.exists(npz):
        raise SystemExit(f"no session file {npz} (recorded sessions: "
                         f"{prefix}_<ts>.npz from CASYNTH_RECORD=1)")
    d = np.load(npz, allow_pickle=True)
    controls = d["replay_controls"] if "replay_controls" in d.files else np.array([])
    if len(controls) == 0:
        raise SystemExit(
            f"{prefix}_{ts}.npz has no replay_controls log (recorded before "
            "full-control logging was added -- nothing to replay faithfully)")
    grids = d["replay_grids"]
    # Match the recorded run's master gain so old sessions stay faithful even if
    # the MASTER_GAIN constant changes later (live gain = master_gain * vol).
    master_gain = float(d["master_gain"]) if "master_gain" in d.files else MASTER_GAIN

    pool = SlotPool()
    sz = TOTAL_SLOTS + 1
    phase   = np.zeros(sz)
    amp_cur = np.zeros(sz)
    pan_cur = np.full(sz, 0.5)
    out, pkmax, nclip = [], 0.0, 0
    # gain_prev seeds the first chunk's gain glide (as the live gain_prev_box does).
    gain_prev = master_gain * float(controls[0]["vol"])
    for i, c in enumerate(controls):
        c = dict(c)                              # 0-d object array -> dict
        grid = grids[i]
        f0 = midi_to_freq(int(c["note"]))
        _, voices, _ = analyse(grid, f0, c["engine"], dict(c["engine_params"]))
        gate = bool(c.get("gate", True))   # default True for sessions recorded pre-MIDI
        voices_for_replay = voices if gate else []
        release_chunks = max(1, round(float(c["release_ms"]) / 1000.0 / CHUNK_S))
        attack_chunks  = max(1, round(float(c["attack_ms"])  / 1000.0 / CHUNK_S))
        decay_chunks   = max(1, round(float(c["decay_ms"])   / 1000.0 / CHUNK_S))
        sustain        = float(c["sustain"])
        gain = master_gain * float(c["vol"])
        for _ in range(int(c["n_rendered"])):
            pool.update(voices_for_replay, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain)
            buf, pk, nc = render_chunk_laplacian(phase, amp_cur, pan_cur,
                                                 pool.amp_tgt, pool.pan_tgt,
                                                 pool.freq_slots, 2, gain_prev, gain)
            gain_prev = gain
            out.append(buf)
            pkmax = max(pkmax, pk)
            nclip += nc
    audio = np.concatenate(out) if out else np.zeros((0, 2), np.int16)
    return audio, pkmax, nclip


def _replay_cli(ts, prefix="_session"):
    """`python gol_synth.py replay <ts>`: render the session offline to
    artifacts/_replay_<ts>.wav and, if the recorded WAV is present, report
    sample-level fidelity (max/mean abs diff) -- the regression check."""
    import os
    from scipy.io import wavfile
    audio, pk, nclip = replay_session(ts, prefix)
    os.makedirs("artifacts", exist_ok=True)
    out = os.path.join("artifacts", f"_replay_{ts}.wav")
    wavfile.write(out, SR, audio)
    print(f"[replay] {out}  {len(audio)} samples  pre-clip peak={pk:.3f}  "
          f"clipped={nclip}")
    rec_wav = f"{prefix}_{ts}.wav"
    if os.path.exists(rec_wav):
        _sr, ref = wavfile.read(rec_wav)
        n = min(len(ref), len(audio))
        if n:
            diff = np.abs(ref[:n].astype(np.int64) - audio[:n].astype(np.int64))
            print(f"[fidelity vs {rec_wav}] common={n} samples  "
                  f"max|diff|={int(diff.max())}  mean|diff|={diff.mean():.3f}  "
                  f"(len ref={len(ref)} replay={len(audio)})")
    else:
        print(f"[fidelity] no recorded {rec_wav} to compare against")
