"""Measurements for the N0 probes: level, long-term spectrum, time variation -- and a comparison
of a probe with the untouched control run.  Plain numpy/scipy; nothing perceptual is claimed."""
import numpy as np
from scipy.signal import stft, welch

SR = 44100


def mono(stereo):
    return 0.5 * (stereo[:, 0] + stereo[:, 1])


def db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def frame_rms_db(x, sr=SR, frame=0.25):
    n = int(frame * sr)
    m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    return db(np.sqrt(np.mean(fr * fr, axis=1)))


def spectrum(x, sr=SR, nfft=4096):
    f, p = welch(x, fs=sr, nperseg=nfft, noverlap=nfft // 2, window="hann", scaling="spectrum")
    return f, p


def spectral_features(f, p):
    """Centroid (Hz), spectral spread (Hz), 85% roll-off (Hz), flatness (dB, 20 Hz..16 kHz)."""
    p = np.maximum(p, 1e-30)
    band = (f >= 20) & (f <= 16000)
    fb, pb = f[band], p[band]
    tot = pb.sum()
    cen = float((fb * pb).sum() / tot)
    spread = float(np.sqrt(((fb - cen) ** 2 * pb).sum() / tot))
    cum = np.cumsum(pb) / tot
    roll = float(fb[np.searchsorted(cum, 0.85)])
    flat = float(10.0 * np.log10(np.exp(np.mean(np.log(pb))) / np.mean(pb)))
    return dict(centroid_hz=cen, spread_hz=spread, rolloff85_hz=roll, flatness_db=flat)


def temporal_features(x, sr=SR):
    """Spectral flux (mean L2 change of the normalised magnitude spectrogram per frame),
    frame-level std (dB, 50 ms frames), crest factor (dB)."""
    f, t, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536, window="hann")
    M = np.abs(Z)
    M = M / (M.sum(axis=0, keepdims=True) + 1e-12)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(M, axis=1) ** 2, axis=0))))
    fr = frame_rms_db(x, sr, frame=0.05)
    env_std = float(np.std(fr))
    rms = np.sqrt(np.mean(x * x)) + 1e-12
    crest = float(db(np.abs(x).max() / rms))
    # slow modulation: std of 0.25 s frames (level movement on the scale of gestures)
    env_std_slow = float(np.std(frame_rms_db(x, sr, frame=0.25)))
    return dict(spectral_flux=flux, env_std_db_50ms=env_std, env_std_db_250ms=env_std_slow, crest_db=crest)


def describe(x, sr=SR):
    f, p = spectrum(x, sr)
    d = dict(rms_db=float(db(np.sqrt(np.mean(x * x)))), peak=float(np.abs(x).max()))
    d.update(spectral_features(f, p))
    d.update(temporal_features(x, sr))
    return d, (f, p)


def compare(x, ref, sr=SR):
    """Probe vs control on the same window: level difference, log-spectral distance after
    level normalisation (dB RMS over 20 Hz..16 kHz), PCM difference (dB re control), changes
    of the temporal features."""
    dx, (f, px) = describe(x, sr)
    dr, (_, pr) = describe(ref, sr)
    band = (f >= 20) & (f <= 16000)
    lx = 10 * np.log10(np.maximum(px[band], 1e-30)); lr = 10 * np.log10(np.maximum(pr[band], 1e-30))
    lx = lx - lx.mean(); lr = lr - lr.mean()                     # remove level
    lsd = float(np.sqrt(np.mean((lx - lr) ** 2)))
    n = min(len(x), len(ref))
    diff = x[:n] - ref[:n]
    pcm_diff_db = float(db(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref[:n] ** 2)) + 1e-12)))
    out = dict(level_change_db=dx["rms_db"] - dr["rms_db"], log_spectral_distance_db=lsd,
               pcm_difference_db_re_control=pcm_diff_db,
               centroid_change_ratio=dx["centroid_hz"] / max(dr["centroid_hz"], 1e-9),
               flux_change_ratio=dx["spectral_flux"] / max(dr["spectral_flux"], 1e-12),
               env_std_change_db=dx["env_std_db_250ms"] - dr["env_std_db_250ms"],
               crest_change_db=dx["crest_db"] - dr["crest_db"])
    # coarse reading of WHAT changed (thresholds are engineering conventions, not hearing)
    out["reading"] = dict(
        level=abs(out["level_change_db"]) > 3.0,
        spectrum=lsd > 3.0 or not (0.8 < out["centroid_change_ratio"] < 1.25),
        time_behaviour=not (0.67 < out["flux_change_ratio"] < 1.5) or abs(out["env_std_change_db"]) > 2.0,
    )
    return out, dx, dr
