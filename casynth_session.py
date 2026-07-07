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
from casynth_engine import (SlotPool, analyse, render_chunk_laplacian,
                            midi_to_freq, events_field)


def _dump_session(rec, prefix="_session"):
    import time
    from scipy.io import wavfile
    base = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    if rec['chunks']:
        audio = np.concatenate(rec['chunks'], axis=0)
        wavfile.write(f"{base}.wav", SR, audio)
    frames = np.array(rec['frames'], dtype=float) if rec['frames'] else np.zeros((0, 6))
    if rec['steps']:
        gens  = np.array([s[0] for s in rec['steps']])
        grids = np.stack([s[1] for s in rec['steps']])
        notes = np.array([s[2] for s in rec['steps']])
        # FIX-A: 4-element tuples (gen, grid_after, note, grid_prev) record the true
        # predecessor grid per step so replay can reconstruct exc_field exactly.
        # Legacy 3-element tuples: step_prevs shape (0,H,W) → legacy fallback in replay.
        if len(rec['steps'][0]) >= 4:
            step_prevs = np.stack([s[3] for s in rec['steps']]).astype(np.uint8)
        else:
            step_prevs = np.zeros((0, GRID_H, GRID_W), np.uint8)
    else:
        gens = np.zeros(0)
        grids = np.zeros((0, GRID_H, GRID_W), np.uint8)
        notes = np.zeros(0)
        step_prevs = np.zeros((0, GRID_H, GRID_W), np.uint8)
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
                        notes=notes, step_prevs=step_prevs,
                        sr=SR, chunk_s=CHUNK_S,
                        bpm=rec.get('bpm', BPM_DEFAULT),
                        div_idx=rec.get('div_idx', DIV_DEFAULT),
                        max_voices=MAX_VOICES, master_gain=MASTER_GAIN,
                        max_modes=MAX_MODES_PER_OBJ, patch_size=PATCH_SIZE,
                        underruns=rec['underruns'],
                        replay=replay, replay_grids=replay_grids,
                        replay_engines=replay_engines,
                        replay_controls=replay_controls,
                        midi_onsets=midi_onsets, midi_in=midi_in_log)
    # frames columns: [dt_ms, gen, n_voices, n_rendered, underrun, n_steps_done]
    n_hitch = int((frames[:, 0] > CHUNK_S * 1000).sum()) if len(frames) else 0
    print(f"[session saved] {base}.wav ({len(rec['chunks'])} chunks) + {base}.npz "
          f"({len(rec['steps'])} steps, {len(midi_onsets)} midi onsets)  "
          f"underruns={rec['underruns']}")


# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE SESSION REPLAY  (read side of the session log -- see _dump_session)
# ──────────────────────────────────────────────────────────────────────────────

def _exc_for_frames(step_gens, step_grids, step_prevs, frame_gens,
                    frame_serials=None):
    """Reconstruct per-frame exc_field from the recorded step history.

    Mirrors the live-synth policy: exc is computed via events_field() on every
    GOL step and held unchanged until the next step.

    Parameters
    ----------
    step_gens  : 1-D int array (K,) -- generation number when each step was taken.
                 NOTE: may be non-monotonic when the user pressed clear() mid-session
                 (state['gen'] resets to 0).  Only reliable when frame_serials=None
                 AND the user never pressed clear during the session.
    step_grids : uint8 array (K, H, W) -- grid AFTER each step (rec['grids'])
    step_prevs : uint8 array (K, H, W) or None -- grid BEFORE each step.
                 None OR shape (0, H, W) = legacy recording without prev_grid;
                 falls back to step_grids[k-1] as approximate predecessor (k=0 → zeros).
                 With the FIX-A schema (4-tuple steps), step_prevs holds the TRUE prev
                 (including manual edits and random-fills between steps).
    frame_gens : 1-D int array (F,) -- generation at each recorded frame.
    frame_serials : 1-D int array (F,) or None.
                 When provided (FIX-F; frames column 5 = n_steps_done at record time),
                 uses SERIAL-PATH: direct index k = serial - 1.  This is correct even
                 when step_gens are non-monotonic (clear scenario).  serial=0 → None
                 (no step taken yet).
                 When None, falls back to LEGACY GEN-PATH: searchsorted on step_gens.
                 Safe only when step_gens are monotonic (user never pressed clear).

    Returns
    -------
    list of F items: None (no step seen yet) or float (H, W) exc_field array.
    Items are shared references; callers that need independent copies must copy.
    """
    if len(step_gens) == 0 or len(frame_gens) == 0:
        return [None] * len(frame_gens)

    has_prevs = (step_prevs is not None) and (len(step_prevs) == len(step_grids))
    K = len(step_grids)
    exc_list = []
    last_exc = None
    last_step_idx = -1

    use_serials = (frame_serials is not None
                   and len(frame_serials) == len(frame_gens))

    if use_serials:
        # Serial-path: k = serial - 1 (direct, no sorting required).
        # serial = len(rec['steps']) at frame record time (FIX-F 6th column).
        # serial=0  → no step taken yet → None.
        # k >= K    → safety clamp (malformed recording; keeps last valid exc).
        for serial in frame_serials:
            k = int(serial) - 1
            if k < 0:
                exc_list.append(None)
                continue
            k = min(k, K - 1)   # safety clamp for malformed recordings
            if k != last_step_idx:
                new_g  = step_grids[k]
                prev_g = (step_prevs[k] if has_prevs else
                          (step_grids[k - 1] if k > 0
                           else np.zeros_like(new_g)))
                last_exc = events_field(prev_g, new_g)
                last_step_idx = k
            exc_list.append(last_exc)
    else:
        # Legacy gen-path: searchsorted requires MONOTONIC step_gens.
        # For sessions recorded before FIX-F (5-column frames) where the user
        # did NOT press clear(), step_gens are monotonic and this is correct.
        for gen in frame_gens:
            step_idx = int(np.searchsorted(step_gens, int(gen), side='right')) - 1
            if step_idx >= 0 and step_idx != last_step_idx:
                new_g = step_grids[step_idx]
                if has_prevs:
                    prev_g = step_prevs[step_idx]      # true prev (FIX-A)
                else:
                    # Legacy: use the preceding step's grid as approximate prev.
                    # First step -> zeros (all live cells look like births).
                    prev_g = (step_grids[step_idx - 1] if step_idx > 0
                              else np.zeros_like(new_g))
                last_exc = events_field(prev_g, new_g)
                last_step_idx = step_idx
            exc_list.append(last_exc)

    return exc_list


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

    # --- exc_field reconstruction from step history (decisions.md 2026-07-05) ---
    # FIX-A: step_prevs (shape K×H×W) records the TRUE predecessor grid for each step,
    # including manual edits and random-fills between auto-steps.  Legacy recordings
    # without step_prevs (shape 0×H×W) use the approximate legacy fallback in
    # _exc_for_frames (prev = preceding step's grid; first step -> zeros).
    step_grids_all = (d['grids'] if 'grids' in d.files
                      else np.zeros((0, GRID_H, GRID_W), np.uint8))
    step_gens_all  = (d['gens'].astype(int) if 'gens' in d.files
                      else np.array([], dtype=int))
    step_prevs_all = (d['step_prevs'] if 'step_prevs' in d.files
                      else np.zeros((0, GRID_H, GRID_W), np.uint8))
    frame_data     = (d['frames'] if 'frames' in d.files
                      else np.zeros((0, 5)))
    frame_gens     = (frame_data[:, 1].astype(int) if len(frame_data) > 0
                      else np.array([], dtype=int))
    # FIX-F: 6-column frames (n_steps_done) → serial-path (correct after clear()).
    # 5-column frames (legacy, before FIX-F) → gen-path (requires monotonic gens).
    frame_serials  = (frame_data[:, 5].astype(int)
                      if (len(frame_data) > 0 and frame_data.shape[1] >= 6)
                      else None)
    # Reconstruct per-frame exc_field list via the shared helper (same logic as live).
    exc_per_frame = _exc_for_frames(step_gens_all, step_grids_all,
                                    step_prevs_all, frame_gens,
                                    frame_serials=frame_serials)

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
        exc_field = exc_per_frame[i] if i < len(exc_per_frame) else None

        # tuned_f0: actual sounding carrier (Sethares-snapped when tune>0).
        # Old recordings without this key fall back to midi_to_freq(note) for compat.
        f0 = float(c.get("tuned_f0", midi_to_freq(int(c["note"]))))
        _, voices, _ = analyse(grid, f0, c["engine"], dict(c["engine_params"]),
                               exc=exc_field)
        gate = bool(c.get("gate", True))   # default True for sessions recorded pre-MIDI
        voices_for_replay = voices if gate else []
        # GEN A/D/R chunk counts.  New recordings store gen_* as FRACTIONS of a tick;
        # legacy recordings store *_ms.  (This is the superseded per-frame path --
        # threaded-era recordings log n_rendered=0, so it renders nothing; the VCA
        # articulation is not reproduced here.  Kept tolerant for legacy ms sessions.)
        if "gen_release" in c:
            _bpm = float(c.get("bpm", BPM_DEFAULT))
            _div = int(c.get("div_idx", DIV_DEFAULT))
            _interval = NOTE_DIVS[_div][1] * 60.0 / _bpm
            release_chunks = max(1, round(float(c["gen_release"]) * _interval / CHUNK_S))
            attack_chunks  = max(1, round(float(c["gen_attack"])  * _interval / CHUNK_S))
            decay_chunks   = max(1, round(float(c["gen_decay"])   * _interval / CHUNK_S))
            sustain        = float(c["gen_sustain"])
        else:
            release_chunks = max(1, round(float(c["release_ms"]) / 1000.0 / CHUNK_S))
            attack_chunks  = max(1, round(float(c["attack_ms"])  / 1000.0 / CHUNK_S))
            decay_chunks   = max(1, round(float(c["decay_ms"])   / 1000.0 / CHUNK_S))
            sustain        = float(c["sustain"])
        gain = master_gain * float(c["vol"])
        amp_slew = bool(c.get("gen_amp_slew", False))   # legacy sessions -> False
        for _ in range(int(c["n_rendered"])):
            pool.update(voices_for_replay, phase, amp_cur, pan_cur, release_chunks,
                        attack_chunks, decay_chunks, sustain, amp_slew=amp_slew)
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
