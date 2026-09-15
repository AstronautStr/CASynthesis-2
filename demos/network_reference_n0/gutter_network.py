"""Vectorised reproduction of the Gutter Synthesis 8-node network (Max patch + gutterOsc.class).

Node arithmetic is the SAME statement order as ``gutter_node.GutterOsc`` (bit-identical, see
tests); the Max-side objects around it are transcribed from ``mxjGutterCompact.maxpat`` and
``Gutter Synth.maxpat`` (the Max runtime itself was NOT executed -- see README limitations):

  per node n (bpatcher mxjGutterCompact #n):
    gamma_n  = mapMod(mod_n)      -> line~ 30 ms  -> mxj inlet 0
    omega    = 0.002              -> line~ 50 ms  -> mxj inlet 1
    c_n      = clip~(mapDamp(damp_n) [line~ 15 ms] + link_n, 0.0001, 1) -> mxj inlet 2
    dt_n     = mapRate(rate_n)    -> line~ 30 ms  -> mxj inlet 3
    gain_n   = mapGain(gain_n)    -> line~ 30 ms  -> mxj inlet 4
    audio    = 0                                  -> mxj inlet 5
    out0 -> clip~ -5 5 -> svf~ 20 (HP) -> svf~ 30 (HP) -> tanh~ -> pan (equal power)
                                                       -> L_n, R_n (send~ nL / nR)
  network (p matrix):
    in_i  = 0.5 * (L_i + R_i)
    mo_j  = sum_i G[i, j] * in_i           (matrix~ 8 8, cells from matrixctrl preset 1)
    link_j(k) = interaction * mo_j(k - D)  (delay~ D samples, then *~ line~ 50 ms)
              D = MATRIX_DELAY (2000 in N0; the patch draws 2000..3999 at load)
                + EXTRA_FEEDBACK_SAMPLES (one signal vector added by send~/receive~ in a loop)
  master: (sum_n L_n, sum_n R_n) * MASTER_GAIN   (patch: * 1.0 * 0.4 -> dac~)

All signals in the Max chain are float64 (Max 6+ audio); the mxj~ boundary is float32 in
both directions (MSPSignal.vec is float[]), and that cast is applied here too.

Explicit configuration keys added for N1 (2026-09-15; absent = the N0 behaviour, so the N0
package stays reproducible from its own config.json):
  output_routes : {"L": [0/1 x 8], "R": [0/1 x 8]} -- which node outlets reach the master
                  sum AND the matrix input.  In the source patch (p individual_controls)
                  node 6 (index 5) has no right connection (review R1): R[5] = 0.
  post_math     : "numpy" (N0) | "scalar" (N1) -- how tanh~, the master sums and the matrix~
                  product are evaluated.  numpy's vectorised tanh differs from the C runtime
                  in the last ulp for ~20 % of inputs, and its reductions / BLAS do not sum
                  in index order; "scalar" = C-runtime tanh per node, sums and the matrix
                  product accumulated in index order from 0.0 (what a per-sample kernel does).
"""
import math

import numpy as np

from . import jmath
from . import source_manifest as sm

N_NODES = 8
MAX_FILTERS = 24
EXTRA_FEEDBACK_SAMPLES = 64      # one Max signal vector (default sigvs) in the send~/receive~ loop
MATRIX_DELAY_SAMPLES = 2000      # delay~ 8000 2000 initial value; the patch randomises 2000..3999 at load
PAN_LOOKUP = [0.0, 1.0, 0.15, 0.85, 0.3, 0.7, 0.46, 0.54]
RAMP_MS = dict(gain=30.0, damp=15.0, mod=30.0, rate=30.0, omega=50.0, interaction=50.0, matrix=2000.0)


# ---- source slider mappings (Max [scale] etc.; raw slider values 0..256) --------------------
def map_gain(raw):
    return raw / 256.0 * 3.5


def map_damp(raw):
    x = raw / 256.0
    return x * x


def map_mod(raw):
    x = raw / 256.0
    return x * x * 10.0


def map_rate(raw):
    x = raw / 256.0
    x2 = x * x
    return x2 * x2 * 5.0


def map_q(raw):
    x = raw / 256.0
    return x * x * 399.5 + 0.5


def map_soften(raw):
    x = 1.0 - raw / 256.0
    return x * x * 15500.0 + 500.0


def map_interaction(raw):
    x = raw / 256.0
    return x * x * 5.0


def preset_matrix(number=1):
    """matrixctrl preset -> G[in, out] (0/1).  Preset 1 = every node to every other node."""
    G = np.zeros((N_NODES, N_NODES))
    if number == 1:
        G[:] = 1.0
        np.fill_diagonal(G, 0.0)
    elif number == 3:
        pass
    else:
        raise ValueError("only presets 1 (all-to-all) and 3 (none) are transcribed")
    return G


def default_config(seed=20260915, filter_bank="random"):
    """The N0 base configuration = patch state after load (autopattr restore values)."""
    rng = np.random.default_rng(seed)
    if filter_bank == "random":
        filters = [sm.random_filter_bank(rng, MAX_FILTERS, 1.0) for _ in range(N_NODES)]
    else:
        banks = sm.preset_filter_banks()
        filters = []
        for n in range(N_NODES):
            b = list(banks[int(filter_bank)])
            b = (b + [b[-1]] * MAX_FILTERS)[:MAX_FILTERS]     # source: extra filters copy the last freq
            filters.append(b)
    return dict(
        sr=44100.0, n_nodes=N_NODES, seed=seed, filter_bank=filter_bank,
        gain_raw=[162] * N_NODES, damp_raw=[138] * N_NODES, mod_raw=[46] * N_NODES, rate_raw=[39] * N_NODES,
        q_raw=[149] * N_NODES, soften_raw=[103] * N_NODES, omega=0.002, highpass_hz=5.0,
        filter_count=[24] * N_NODES, dist_method=2, filters_hz=filters, pitch_shift=1.0,
        pan=list(PAN_LOOKUP), matrix=preset_matrix(1).tolist(), interaction_raw=127,
        matrix_delay_samples=MATRIX_DELAY_SAMPLES, extra_feedback_samples=EXTRA_FEEDBACK_SAMPLES,
        svf_hp_hz=[20.0, 30.0], svf_res=0.01, node_clip=5.0, ramp_ms=dict(RAMP_MS),
    )


class _Ramp:
    """line~ on a vector: linear move from the current value to the target over N samples."""

    def __init__(self, value):
        self.cur = np.array(value, dtype=np.float64)
        self.target = self.cur.copy()
        self.inc = np.zeros_like(self.cur)
        self.left = 0

    def set(self, target, samples):
        self.target = np.array(target, dtype=np.float64).reshape(self.cur.shape)
        samples = int(samples)
        if samples <= 0:
            self.cur = self.target.copy(); self.inc[:] = 0.0; self.left = 0
        else:
            self.inc = (self.target - self.cur) / samples; self.left = samples

    def step(self):
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.cur = self.target.copy()
            else:
                self.cur = self.cur + self.inc
        return self.cur

    def export(self):
        return dict(cur=self.cur, target=self.target, inc=self.inc, left=np.array(self.left))

    def restore(self, d):
        self.cur = np.array(d["cur"], dtype=np.float64); self.target = np.array(d["target"], dtype=np.float64)
        self.inc = np.array(d["inc"], dtype=np.float64); self.left = int(d["left"])


class _SVF:
    """Chamberlin state-variable filter (Max svf~ is documented as this design); HP outlet used.
    Approximation: Max's exact coefficient mapping of 'resonance' is not published -> q = 1 - res.
    """

    def __init__(self, n, fc, res, sr):
        self.f = 2.0 * math.sin(math.pi * fc / sr)
        self.q = 1.0 - res
        self.low = np.zeros(n); self.band = np.zeros(n)

    def hp(self, x):
        self.low = self.low + self.f * self.band
        high = x - self.low - self.q * self.band
        self.band = self.band + self.f * high
        return high


class GutterNetwork:
    def __init__(self, config=None):
        cfg = default_config() if config is None else config
        self.cfg = cfg
        self.sr = float(cfg["sr"])
        n = self.n = int(cfg["n_nodes"])
        self.k = 0                                  # sample clock
        # --- node state (as in gutterOsc) ---
        self.duffX = np.zeros(n); self.duffY = np.zeros(n); self.dx = np.zeros(n); self.dy = np.zeros(n)
        self.t = np.zeros(n); self.finalY = np.zeros(n)
        self.resets = np.zeros(n, dtype=np.int64)
        self.filtersOn = True
        self.dist = int(cfg["dist_method"])
        self.count = np.array(cfg["filter_count"], dtype=np.int64)
        self.fmask = (np.arange(MAX_FILTERS)[None, :] < self.count[:, None])
        self.ffreq = np.array(cfg["filters_hz"], dtype=np.float64)
        self.fQ = np.full((n, MAX_FILTERS), 30.0)
        for i in range(n):
            self.fQ[i, :] = jmath.f32(map_q(cfg["q_raw"][i]))
        # setFreqN / setQN messages carry float32 values (Max -> Java float)
        self.ffreq = np.array([[jmath.f32(min(f * cfg["pitch_shift"], 19000.0)) for f in row] for row in self.ffreq])
        self.px1 = np.zeros((n, MAX_FILTERS)); self.px2 = np.zeros((n, MAX_FILTERS))
        self.py1 = np.zeros((n, MAX_FILTERS)); self.py2 = np.zeros((n, MAX_FILTERS))
        self.a1 = np.zeros((n, MAX_FILTERS))
        self._calc_coeffs()
        # one-pole lowpass / highpass of the class
        self.lp_prev = np.zeros(n)
        self.lp_a0 = np.zeros(n); self.lp_b1 = np.ones(n)
        for i in range(n):
            self._set_lowpass(i, map_soften(cfg["soften_raw"][i]))
        self.hp_state = np.zeros(n)
        self.hp_cut = np.full(n, jmath.f32(cfg["highpass_hz"]))
        self.hp_ratio = self.hp_cut / (6.283188 * self.sr)
        # --- control ramps (line~) ---
        ms = cfg["ramp_ms"]; sr = self.sr
        self._ramp_n = {k: int(round(ms[k] * sr / 1000.0)) for k in ms}
        self.r_gamma = _Ramp(np.zeros(n)); self.r_gamma.set([map_mod(v) for v in cfg["mod_raw"]], self._ramp_n["mod"])
        self.r_dt = _Ramp(np.zeros(n)); self.r_dt.set([map_rate(v) for v in cfg["rate_raw"]], self._ramp_n["rate"])
        self.r_gain = _Ramp(np.zeros(n)); self.r_gain.set([map_gain(v) for v in cfg["gain_raw"]], self._ramp_n["gain"])
        self.r_damp = _Ramp(np.ones(n)); self.r_damp.set([map_damp(v) for v in cfg["damp_raw"]], self._ramp_n["damp"])
        self.r_omega = _Ramp(np.zeros(1)); self.r_omega.set([cfg["omega"]], self._ramp_n["omega"])
        self.r_gint = _Ramp(np.zeros(1)); self.r_gint.set([map_interaction(cfg["interaction_raw"])], self._ramp_n["interaction"])
        self.G = np.array(cfg["matrix"], dtype=np.float64)          # G[in, out]
        self.G_target = self.G.copy(); self.G_inc = np.zeros_like(self.G); self.G_left = 0
        # --- Max post chain ---
        self.node_clip = float(cfg["node_clip"])
        self.svf1 = _SVF(n, cfg["svf_hp_hz"][0], cfg["svf_res"], sr)
        self.svf2 = _SVF(n, cfg["svf_hp_hz"][1], cfg["svf_res"], sr)
        v = 0.25 * np.array(cfg["pan"], dtype=np.float64)
        self.panL = np.cos(2.0 * np.pi * v); self.panR = np.cos(2.0 * np.pi * (v + 0.75))
        routes = cfg.get("output_routes") or {"L": [1] * n, "R": [1] * n}
        self.routeL = np.array(routes["L"], dtype=np.float64); self.routeR = np.array(routes["R"], dtype=np.float64)
        if self.routeL.shape != (n,) or self.routeR.shape != (n,):
            raise ValueError("output_routes: need L and R masks of length n_nodes")
        post = cfg.get("post_math", "numpy")
        if post not in ("numpy", "scalar"):
            raise ValueError(f"post_math must be 'numpy' or 'scalar', got {post!r}")
        self.scalar_post = (post == "scalar")
        # --- delay lines of the matrix outputs ---
        self.D = int(cfg["matrix_delay_samples"]) + int(cfg["extra_feedback_samples"])
        self.ring = np.zeros((n, self.D + 1))
        self._ar = np.arange(n)

    # ---------------------------------------------------------------- messages / controls
    def _calc_coeffs(self):
        d = np.tan(np.pi * self.ffreq / self.sr)          # per element: same expression as Java
        d2 = 1.0 / (1.0 + d / self.fQ + d * d)
        self.a0 = d / self.fQ * d2
        self.a2 = -self.a0
        self.b1 = 2.0 * (d * d - 1.0) * d2
        self.b2 = (1.0 - d / self.fQ + d * d) * d2

    def _set_lowpass(self, i, hz):
        hz = jmath.f32(hz)
        self.lp_a0[i] = math.sin(6.283188 * (hz / self.sr))
        self.lp_b1[i] = self.lp_a0[i] - 1.0

    def set_slider(self, name, raw, nodes=None, ramp=True):
        """Move a source slider (raw 0..256) of the given nodes (None = all) like the patch would."""
        idx = list(range(self.n)) if nodes is None else list(nodes)
        if name == "gain":
            tgt = self.r_gain.target.copy()
            for i in idx: tgt[i] = map_gain(raw)
            self.r_gain.set(tgt, self._ramp_n["gain"] if ramp else 0)
        elif name == "damp":
            tgt = self.r_damp.target.copy()
            for i in idx: tgt[i] = map_damp(raw)
            self.r_damp.set(tgt, self._ramp_n["damp"] if ramp else 0)
        elif name == "mod":
            tgt = self.r_gamma.target.copy()
            for i in idx: tgt[i] = map_mod(raw)
            self.r_gamma.set(tgt, self._ramp_n["mod"] if ramp else 0)
        elif name == "rate":
            tgt = self.r_dt.target.copy()
            for i in idx: tgt[i] = map_rate(raw)
            self.r_dt.set(tgt, self._ramp_n["rate"] if ramp else 0)
        elif name == "Q":
            for i in idx:
                self.fQ[i, :] = jmath.f32(map_q(raw))
            self._calc_coeffs()
        elif name == "soften":
            for i in idx:
                self._set_lowpass(i, map_soften(raw))
        elif name == "interaction":
            self.r_gint.set([map_interaction(raw)], self._ramp_n["interaction"] if ramp else 0)
        else:
            raise KeyError(name)

    def set_matrix(self, G, ramp=True):
        """matrix~ cell gains (G[in, out]); the patch ramps cell changes over 2000 ms."""
        self.G_target = np.array(G, dtype=np.float64)
        steps = self._ramp_n["matrix"] if ramp else 0
        if steps <= 0:
            self.G = self.G_target.copy(); self.G_left = 0; self.G_inc[:] = 0.0
        else:
            self.G_inc = (self.G_target - self.G) / steps; self.G_left = steps

    def set_filters(self, node, freqs_hz):
        row = (list(freqs_hz) + [freqs_hz[-1]] * MAX_FILTERS)[:MAX_FILTERS]
        self.ffreq[node, :] = [jmath.f32(min(f * self.cfg["pitch_shift"], 19000.0)) for f in row]
        self._calc_coeffs()

    def set_filter_count(self, node, count):
        self.count[node] = int(count)
        self.fmask = (np.arange(MAX_FILTERS)[None, :] < self.count[:, None])

    # ---------------------------------------------------------------- one sample
    def _distort(self, d):
        m = self.dist
        with np.errstate(all="ignore"):
            if m == 0:
                return np.maximum(np.minimum(d, 1.0), -1.0)
            if m == 1:
                return np.where(self.finalY <= -1.0, -0.666666667, np.where(d <= 1.0, d - d * d * d / 3.0, 0.666666667))
            if m == 2:
                return jmath.atan_vec(d)
            if m == 3:
                return 0.75 * (np.sqrt(d * 1.3 * (d * 1.3) + 1.0) * 1.65 - 1.65) / d
            if m == 4:
                return (0.1076 * d * d * d + 3.029 * d) / (d * d + 3.124)
            if m == 5:
                return 2.0 / (1.0 + np.exp(-1.0 * d))
        return np.zeros_like(d)

    def step(self):
        """Advance one sample; returns (L, R) master-summed (before MASTER_GAIN) and node outputs."""
        k = self.k
        n = self.n
        # ---- controls (line~ outputs) -> float32 mxj inlets
        gamma = self.r_gamma.step(); dt = self.r_dt.step(); gain = self.r_gain.step(); damp = self.r_damp.step()
        omega = self.r_omega.step()[0]; gint = self.r_gint.step()[0]
        if self.G_left > 0:
            self.G_left -= 1
            self.G = self.G_target.copy() if self.G_left == 0 else self.G + self.G_inc
        link = self.ring[:, (k + 1) % (self.D + 1)] * gint            # = interaction * mo(k - D)
        c = np.minimum(np.maximum(damp + link, 0.0001), 1.0)
        gamma32 = gamma.astype(np.float32).astype(np.float64)
        omega32 = float(np.float32(omega))
        c32 = c.astype(np.float32).astype(np.float64)
        dt32 = dt.astype(np.float32).astype(np.float64)
        gain32 = gain.astype(np.float32).astype(np.float64)
        self.last_inlets = (gamma32, omega32, c32, dt32, gain32)     # diagnostic: what the nodes saw
        # ---- gutterOsc.perform, one sample, all nodes
        x = self.duffX
        y = self.a0 * x[:, None] + self.a1 * self.px1 + self.a2 * self.px2 - self.b1 * self.py1 - self.b2 * self.py2
        fm = self.fmask
        self.px2 = np.where(fm, self.px1, self.px2)
        self.px1 = np.where(fm, x[:, None], self.px1)
        self.py2 = np.where(fm, self.py1, self.py2)
        self.py1 = np.where(fm, y, self.py1)
        cs = np.cumsum(y * gain32[:, None], axis=1)                   # sequential sum == Java loop order
        fy = np.where(self.count > 0, cs[self._ar, np.maximum(self.count - 1, 0)], 0.0)
        if not self.filtersOn:
            fy = x.copy()
        self.finalY = fy
        s = np.array([jmath.sin(v) for v in (omega32 * self.t)])       # java.lang.Math.sin per node
        self.dy = fy - fy * fy * fy - c32 * self.duffY + gamma32 * s
        self.duffY = self.duffY + self.dy
        self.dx = self.duffY
        d2 = self.lp_a0 * (self.finalY + self.dx) + self.lp_b1 * self.lp_prev
        self.lp_prev = d2
        d3 = d2 - self.hp_state
        self.hp_state = self.hp_state + d3 * self.hp_ratio
        duffX = np.where(self.hp_cut < 10.0, d2, d3)
        if self.filtersOn:
            duffX = self._distort(duffX)
            out0 = (self.finalY * 0.125).astype(np.float32).astype(np.float64)
        else:
            duffX = np.maximum(np.minimum(duffX, 100.0), -100.0)
            big = np.abs(duffX) > 99.0
            if big.any():
                self._reset(big)
                duffX = np.where(big, 0.0, duffX)
            out0 = np.maximum(np.minimum(duffX * gain32, 1.0), -1.0).astype(np.float32).astype(np.float64)
        self.duffX = duffX
        self.t = self.t + dt32
        nan = np.isnan(self.duffX)
        if nan.any():
            self._reset(nan)
            self.duffX = np.where(nan, 0.0, self.duffX)
        # ---- Max post chain: clip~ -5 5 -> svf~ 20 HP -> svf~ 30 HP -> tanh~ -> pan
        v = np.maximum(np.minimum(out0, self.node_clip), -self.node_clip)
        v = self.svf1.hp(v)
        v = self.svf2.hp(v)
        if self.scalar_post:
            o = np.array([math.tanh(float(x)) for x in v])
        else:
            o = np.tanh(v)
        L = np.where(self.routeL > 0, o * self.panL, 0.0)
        R = np.where(self.routeR > 0, o * self.panR, 0.0)
        # ---- matrix~ -> delay~ (written now, read D samples later)
        m_in = 0.5 * (L + R)
        if self.scalar_post:
            col = self.ring[:, k % (self.D + 1)]
            for j in range(n):
                acc = 0.0
                for i in range(n):
                    acc += float(m_in[i]) * float(self.G[i, j])
                col[j] = acc
        else:
            self.ring[:, k % (self.D + 1)] = m_in @ self.G             # mo[j] = sum_i G[i,j] in_i
        self.k = k + 1
        return L, R, o

    def _reset(self, mask):
        self.duffY = np.where(mask, 0.0, self.duffY); self.dx = np.where(mask, 0.0, self.dx)
        self.dy = np.where(mask, 0.0, self.dy); self.t = np.where(mask, 0.0, self.t)
        self.resets = self.resets + mask.astype(np.int64)

    def render(self, n_samples, node_out=False):
        """Render n samples -> (stereo (n,2) float64 before master gain, per-node (n,8) or None)."""
        out = np.zeros((n_samples, 2))
        nodes = np.zeros((n_samples, self.n)) if node_out else None
        for i in range(n_samples):
            L, R, o = self.step()
            if self.scalar_post:
                sl = 0.0; sr_ = 0.0
                for v in L: sl += float(v)
                for v in R: sr_ += float(v)
                out[i, 0] = sl; out[i, 1] = sr_
            else:
                out[i, 0] = L.sum(); out[i, 1] = R.sum()
            if node_out:
                nodes[i] = o
        return out, nodes

    # ---------------------------------------------------------------- state
    def export_state(self):
        st = dict(k=np.array(self.k), duffX=self.duffX, duffY=self.duffY, dx=self.dx, dy=self.dy, t=self.t,
                  finalY=self.finalY, resets=self.resets, px1=self.px1, px2=self.px2, py1=self.py1, py2=self.py2,
                  lp_prev=self.lp_prev, hp_state=self.hp_state, svf1_low=self.svf1.low, svf1_band=self.svf1.band,
                  svf2_low=self.svf2.low, svf2_band=self.svf2.band, ring=self.ring, G=self.G, G_target=self.G_target,
                  G_inc=self.G_inc, G_left=np.array(self.G_left), ffreq=self.ffreq, fQ=self.fQ, count=self.count,
                  lp_a0=self.lp_a0, lp_b1=self.lp_b1, hp_cut=self.hp_cut, hp_ratio=self.hp_ratio,
                  filtersOn=np.array(self.filtersOn), dist=np.array(self.dist))
        for name in ("r_gamma", "r_dt", "r_gain", "r_damp", "r_omega", "r_gint"):
            for kk, vv in getattr(self, name).export().items():
                st[f"{name}.{kk}"] = vv
        return {k: np.array(v) for k, v in st.items()}

    def restore_state(self, st):
        self.k = int(st["k"])
        for name in ("duffX", "duffY", "dx", "dy", "t", "finalY", "resets", "px1", "px2", "py1", "py2", "lp_prev",
                     "hp_state", "ring", "G", "G_target", "G_inc", "ffreq", "fQ", "count", "lp_a0", "lp_b1", "hp_cut", "hp_ratio"):
            setattr(self, name, np.array(st[name]).copy())
        self.svf1.low = np.array(st["svf1_low"]).copy(); self.svf1.band = np.array(st["svf1_band"]).copy()
        self.svf2.low = np.array(st["svf2_low"]).copy(); self.svf2.band = np.array(st["svf2_band"]).copy()
        self.G_left = int(st["G_left"]); self.filtersOn = bool(st["filtersOn"]); self.dist = int(st["dist"])
        self.fmask = (np.arange(MAX_FILTERS)[None, :] < self.count[:, None])
        for name in ("r_gamma", "r_dt", "r_gain", "r_damp", "r_omega", "r_gint"):
            getattr(self, name).restore({kk: st[f"{name}.{kk}"] for kk in ("cur", "target", "inc", "left")})
        self._calc_coeffs()

    @classmethod
    def from_state(cls, config, st):
        net = cls(config)
        net.restore_state(st)
        return net
