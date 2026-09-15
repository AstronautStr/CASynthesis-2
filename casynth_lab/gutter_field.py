"""N1 -- the Gutter Synthesis eight-node network sonifying the field (engine id
`gutter_field`, REQ memory/req-network-ca-n1-2026-09-15.md).

Model.  The network of demos/network_reference_n0/gutter_network.py (eight
``gutterOsc`` nodes, matrix~ -> delay~ 2064 -> x interaction -> damping input;
clip~ / svf~ 20 / svf~ 30 / tanh~ / pan per node) under the FIXED N1
configuration `gutter_field_n1_config.CONFIG` (fixtures 2026-09-15): explicit
bank frequencies, N0 node values, review-R1 output routes (node 6 has no right
outlet -- neither in the master sum nor at the matrix input), `post_math =
"scalar"`.  The per-sample arithmetic here is the SAME statement order as the
scalar node port (Java-verified) and as the slow model in its scalar mode;
tests/test_gutter_field_n1.py checks both bit-exactly.  Only the execution
changed: one numba kernel per block instead of ~50 numpy calls per sample
(~0.7 us / sample instead of ~52; without numba the same function runs as
plain Python -- correct, far too slow for the live bench).

Field -> sound (REQ "Field -> sound").  The H x W field is cut into 2 rows x 4
columns of regions [floor(rH/2), floor((r+1)H/2)) x [floor(cW/4), floor((c+1)W/4));
node i = 4r + c.  Per region:

    n_i     = live cells
    u_i     = n_i / (n_i + 2)
    ratio_i = 2 ** clip(log2(scale) + depth * (2 u_i - 1), -1, 1)
    f_ij    = min(19000, base_ij * ratio_i)          base_ij from the config, never accumulated

The eight targets are applied atomically at a block boundary (update_field /
set_params run there) exactly like the source's setFreqN messages: float32 value,
coefficients recomputed, all filter / node states kept, no interpolation.
depth = 0 -> ratio = scale for every node.  An empty region is n_i = 0.  Nothing
resets the network: painting, evolution, parameter moves and the CA pause all
keep it running; Stop / Restart are the bench transport (init on the field).

Controls: `scale` ("Resonators", 0.5..2), `depth` ("CA amount", 0..1),
`interaction` ("Links", raw 0..256 -> (v/256)^2 * 5 with the source 50 ms line~),
`freeze_ca` ("Freeze CA": holds the control vector u reached so far; manual
parameters keep acting on it; switching it off takes the current field; at init
the held vector is the initial field's).

Level: the raw master sum (eight tanh~ outputs, |sum| <= 8) is multiplied by
FIXED_SCALE = 20 and then ONCE by the bench gain (MASTER_GAIN * vol * level =
0.028 at the defaults -> 0.56 effective), ramped across the block; peak /
n_clip are measured before the int16 clip.  Nothing adapts to the field or the
scene.
"""
import math

import numpy as np

from .engine_api import SoundEngine
from .registry import EngineSpec, register
from .gutter_field_n1_config import CONFIG as N1_CONFIG, config as n1_config

try:                                   # the kernel is plain Python when numba is missing
    import numba as _numba
    _jit = _numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except ImportError:                    # pragma: no cover  (environment without numba)
    def _jit(f):
        return f
    HAVE_NUMBA = False

ENGINE_ID = 'gutter_field'
LABEL = 'Gutter'
PARAMS = [('scale', 'Resonators', 0.5, 2.0, False, 1.0),
          ('depth', 'CA amount', 0.0, 1.0, False, 1.0),
          ('interaction', 'Links', 0, 256, True, 127),
          ('freeze_ca', 'Freeze CA', 0, 1, True, 0)]
FIXED_SCALE = 20.0                     # raw master sum -> bench gain (REQ: 20)
REGION_ROWS, REGION_COLS = 2, 4
N_NODES = 8
MAX_FILTERS = 24
FREQ_MAX = 19000.0                     # source: pitch-shifted frequencies are capped at 19 kHz
STATE_VERSION = 1
MODEL_VERSION = N1_CONFIG['model_version']

# node state rows (float64 (NS, 8)); ramp indices (float64 (6, 3, 8): cur / target / inc)
DUFFX, DUFFY, DX, DY, T, FINALY, LP_PREV, HP_STATE, S1L, S1B, S2L, S2B = range(12)
NS = 12
R_GAMMA, R_DT, R_GAIN, R_DAMP, R_OMEGA, R_GINT = range(6)
NR = 6
C_CLIP, C_F1, C_Q1, C_F2, C_Q2 = range(5)          # consts (float64 (5,))
I_K, I_GLEFT = range(2)                            # ints (int64 (2,))


# -- source slider mappings (Max [scale]; copies of gutter_network.map_* -- asserted equal) --
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


def f32(x):
    """Java ``(float)`` cast of a message argument."""
    return float(np.float32(x))


def calc_coeffs(ffreq, fQ, sr):
    """gutterOsc$BPFilter.calcCoeffs on arrays (same expressions as the node port)."""
    d = np.tan(np.pi * ffreq / sr)
    d2 = 1.0 / (1.0 + d / fQ + d * d)
    a0 = d / fQ * d2
    a2 = -a0
    b1 = 2.0 * (d * d - 1.0) * d2
    b2 = (1.0 - d / fQ + d * d) * d2
    return a0, a2, b1, b2


# -- field -> control ------------------------------------------------------------------
def region_bounds(rows, cols):
    """[(r0, r1, c0, c1)] for node i = 4 r + c (half-open, floor division as in the REQ)."""
    out = []
    for r in range(REGION_ROWS):
        for c in range(REGION_COLS):
            out.append((r * rows // REGION_ROWS, (r + 1) * rows // REGION_ROWS,
                        c * cols // REGION_COLS, (c + 1) * cols // REGION_COLS))
    return out


def region_counts(grid):
    """Live cells per region, float64 (8,)."""
    g = np.asarray(grid)
    return np.array([float(np.count_nonzero(g[r0:r1, c0:c1]))
                     for r0, r1, c0, c1 in region_bounds(*g.shape)])


def field_u(counts):
    n = np.asarray(counts, np.float64)
    return n / (n + 2.0)


def ratio_of(u, scale, depth):
    x = np.clip(math.log2(float(scale)) + float(depth) * (2.0 * np.asarray(u, np.float64) - 1.0),
                -1.0, 1.0)
    return np.power(2.0, x)


def bank_freqs(base, ratio):
    """f32(min(19000, base_ij * ratio_i)) -- the value a setFreqN message carries."""
    f = np.minimum(np.asarray(base, np.float64) * np.asarray(ratio, np.float64)[:, None], FREQ_MAX)
    return f.astype(np.float32).astype(np.float64)


# -- fdlibm atan (java.lang.Math.atan), scalar; see demos/network_reference_n0/jmath.py -------
_ATANHI = (4.63647609000806093515e-01, 7.85398163397448278999e-01,
           9.82793723247329054082e-01, 1.57079632679489655800e+00)
_ATANLO = (2.26987774529616870924e-17, 3.06161699786838301793e-17,
           1.39033110312309984516e-17, 6.12323399573676603587e-17)
_AT = (3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01,
       -1.11111104054623557880e-01, 9.09088713343650656196e-02, -7.69187620504482999495e-02,
       6.66107313738753120669e-02, -5.83357013379057348645e-02, 4.97687799461593236017e-02,
       -3.65315727442169155270e-02, 1.62858201153657823623e-02)
_HUGE = 1.0e300


@_jit
def _atan(x):
    if x != x:
        return x + x
    ax = -x if x < 0.0 else x
    if ax >= 7.378697629483821e19:             # 2^66 (also +-inf)
        return _ATANHI[3] + _ATANLO[3] if x > 0.0 else -_ATANHI[3] - _ATANLO[3]
    if ax < 0.4375:
        if ax < 1.862645149230957e-09:         # 2^-29
            if _HUGE + x > 1.0:
                return x
        z = x * x
        w = z * z
        s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
        s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
        return x - x * (s1 + s2)
    if ax < 1.1875:
        if ax < 0.6875:
            idx = 0
            ax = (2.0 * ax - 1.0) / (2.0 + ax)
        else:
            idx = 1
            ax = (ax - 1.0) / (ax + 1.0)
    else:
        if ax < 2.4375:
            idx = 2
            ax = (ax - 1.5) / (1.0 + 1.5 * ax)
        else:
            idx = 3
            ax = -1.0 / ax
    z = ax * ax
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    z = _ATANHI[idx] - ((ax * (s1 + s2) - _ATANLO[idx]) - ax)
    return -z if x < 0.0 else z


@_jit
def _f32(x):
    return np.float64(np.float32(x))


@_jit
def _render(n, out, node, resets, px1, px2, py1, py2, a0, a1, a2, b1, b2, count,
            lp_a0, lp_b1, hp_ratio, hp_cut, ramps, ramp_left, G, G_tgt, G_inc, ints,
            ring, consts, panL, panR, routeL, routeR, scratch):
    """`n` samples of the network into out (n, 2): raw master L / R sums.
    Statement order per node = gutterOsc.perform (see gutter_node.py); the Max
    chain and the line~ ramps = gutter_network.GutterNetwork.step (scalar mode)."""
    nn = node.shape[1]
    D1 = ring.shape[1]
    clip = consts[C_CLIP]
    f1 = consts[C_F1]; q1 = consts[C_Q1]; f2 = consts[C_F2]; q2 = consts[C_Q2]
    for s in range(n):
        k = ints[I_K]
        # ---- line~ ramps (cur -> target over `left` samples)
        for w in range(NR):
            left = ramp_left[w]
            if left > 0:
                left -= 1
                if left == 0:
                    for i in range(nn):
                        ramps[w, 0, i] = ramps[w, 1, i]
                else:
                    for i in range(nn):
                        ramps[w, 0, i] = ramps[w, 0, i] + ramps[w, 2, i]
                ramp_left[w] = left
        gl = ints[I_GLEFT]
        if gl > 0:
            gl -= 1
            if gl == 0:
                for i in range(nn):
                    for j in range(nn):
                        G[i, j] = G_tgt[i, j]
            else:
                for i in range(nn):
                    for j in range(nn):
                        G[i, j] = G[i, j] + G_inc[i, j]
            ints[I_GLEFT] = gl
        idx_r = (k + 1) % D1
        idx_w = k % D1
        gint = ramps[R_GINT, 0, 0]
        omega32 = _f32(ramps[R_OMEGA, 0, 0])
        sumL = 0.0
        sumR = 0.0
        for i in range(nn):
            # ---- damping input: clip~(damp + interaction * mo(k - D), 0.0001, 1) -> float32
            c = ramps[R_DAMP, 0, i] + ring[i, idx_r] * gint
            if c < 0.0001:
                c = 0.0001
            if c > 1.0:
                c = 1.0
            gamma32 = _f32(ramps[R_GAMMA, 0, i])
            c32 = _f32(c)
            dt32 = _f32(ramps[R_DT, 0, i])
            gain32 = _f32(ramps[R_GAIN, 0, i])
            # ---- gutterOsc.perform, one sample
            x = node[DUFFX, i]
            fy = 0.0
            for j in range(count[i]):
                y = a0[i, j] * x + a1[i, j] * px1[i, j] + a2[i, j] * px2[i, j] \
                    - b1[i, j] * py1[i, j] - b2[i, j] * py2[i, j]
                px2[i, j] = px1[i, j]
                px1[i, j] = x
                py2[i, j] = py1[i, j]
                py1[i, j] = y
                fy += y * gain32
            node[FINALY, i] = fy
            sn = math.sin(omega32 * node[T, i])
            dyv = fy - fy * fy * fy - c32 * node[DUFFY, i] + gamma32 * sn
            node[DY, i] = dyv
            duffY = node[DUFFY, i] + dyv
            node[DUFFY, i] = duffY
            node[DX, i] = duffY
            d2 = lp_a0[i] * (fy + duffY) + lp_b1[i] * node[LP_PREV, i]
            node[LP_PREV, i] = d2
            d3 = d2 - node[HP_STATE, i]
            node[HP_STATE, i] = node[HP_STATE, i] + d3 * hp_ratio[i]
            dX = d2 if hp_cut[i] < 10.0 else d3
            dX = _atan(dX)                                   # distortion method 2
            out0 = _f32(fy * 0.125)
            node[DUFFX, i] = dX
            node[T, i] = node[T, i] + dt32
            if dX != dX:                                     # resetDuff()
                node[DUFFX, i] = 0.0
                node[DUFFY, i] = 0.0
                node[DX, i] = 0.0
                node[DY, i] = 0.0
                node[T, i] = 0.0
                resets[i] += 1
            # ---- Max chain: clip~ -5 5 -> svf~ 20 HP -> svf~ 30 HP -> tanh~ -> pan -> routes
            v = out0
            if v > clip:
                v = clip
            if v < -clip:
                v = -clip
            low = node[S1L, i] + f1 * node[S1B, i]
            node[S1L, i] = low
            high = v - low - q1 * node[S1B, i]
            node[S1B, i] = node[S1B, i] + f1 * high
            v = high
            low = node[S2L, i] + f2 * node[S2B, i]
            node[S2L, i] = low
            high = v - low - q2 * node[S2B, i]
            node[S2B, i] = node[S2B, i] + f2 * high
            o = math.tanh(high)
            Li = o * panL[i] if routeL[i] > 0.0 else 0.0
            Ri = o * panR[i] if routeR[i] > 0.0 else 0.0
            sumL += Li
            sumR += Ri
            scratch[i] = 0.5 * (Li + Ri)
        # ---- matrix~ -> delay~ (written now, read D samples later)
        for j in range(nn):
            acc = 0.0
            for i in range(nn):
                acc += scratch[i] * G[i, j]
            ring[j, idx_w] = acc
        out[s, 0] = sumL
        out[s, 1] = sumR
        ints[I_K] = k + 1


def overlay(params, rows, cols):
    """Region borders + node numbers over the field (same geometry as region_bounds)."""
    lines, labels = [], []
    for c in range(1, REGION_COLS):
        x = c * cols // REGION_COLS - 0.5
        lines.append(((x, -0.5), (x, rows - 0.5)))
    for r in range(1, REGION_ROWS):
        y = r * rows // REGION_ROWS - 0.5
        lines.append(((-0.5, y), (cols - 0.5, y)))
    for i, (r0, r1, c0, c1) in enumerate(region_bounds(rows, cols)):
        labels.append(((c0 + c1 - 1) / 2.0, (r0 + r1 - 1) / 2.0, str(i)))
    return dict(lines=lines, labels=labels,
                text="regions 0..7 = nodes: live cells in a region tune that node's resonator bank")


# -- engine --------------------------------------------------------------------------------
class GutterFieldEngine(SoundEngine):
    STATE_VERSION = STATE_VERSION

    def __init__(self, ctx, params, config=None):
        super().__init__(ctx, params)
        cfg = n1_config() if config is None else config
        if int(cfg['n_nodes']) != N_NODES or float(cfg['sr']) != float(ctx.sr) \
                or int(cfg['dist_method']) != 2 or cfg.get('post_math') != 'scalar' \
                or any(int(c) > MAX_FILTERS or int(c) < 0 for c in cfg['filter_count']) \
                or ctx.channels != 2:
            raise ValueError(f"engine {ENGINE_ID}: configuration outside the N1 model "
                             f"(8 nodes, sr {ctx.sr}, distortion 2, scalar post, stereo)")
        self.cfg = cfg
        self.sr = float(cfg['sr'])
        self.base = np.array(cfg['filters_hz'], np.float64)
        if self.base.shape != (N_NODES, MAX_FILTERS):
            raise ValueError(f"engine {ENGINE_ID}: filters_hz must be 8 x 24")
        self.model_version = str(cfg['model_version'])
        n = int(ctx.block)
        self._ramp = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n))) * 0.5
        self._out = np.zeros((n, 2))
        self._grid = None
        self._build(int(self.params['interaction']))
        self.u_field = np.zeros(N_NODES)
        self.u_held = np.zeros(N_NODES)
        self.counts = np.zeros(N_NODES)
        self.ratio = np.ones(N_NODES)
        self.frozen = False
        self.gain_prev = 0.0
        self._kernel(0, np.zeros((0, 2)))          # compile / load the cached kernel now

    # -- model state (== GutterNetwork.__init__ of the same config) ----------------------
    def _build(self, interaction_raw):
        cfg, sr, n = self.cfg, self.sr, N_NODES
        self.node = np.zeros((NS, n))
        self.resets = np.zeros(n, np.int64)
        self.count = np.array(cfg['filter_count'], np.int64)
        self.fQ = np.full((n, MAX_FILTERS), 30.0)
        for i in range(n):
            self.fQ[i, :] = f32(map_q(cfg['q_raw'][i]))
        self.ffreq = np.array([[f32(min(f * cfg['pitch_shift'], FREQ_MAX)) for f in row]
                               for row in self.base])
        self.px1 = np.zeros((n, MAX_FILTERS)); self.px2 = np.zeros((n, MAX_FILTERS))
        self.py1 = np.zeros((n, MAX_FILTERS)); self.py2 = np.zeros((n, MAX_FILTERS))
        self.a1 = np.zeros((n, MAX_FILTERS))
        self.a0, self.a2, self.b1, self.b2 = calc_coeffs(self.ffreq, self.fQ, sr)
        self.lp_a0 = np.zeros(n); self.lp_b1 = np.ones(n)
        for i in range(n):
            hz = f32(map_soften(cfg['soften_raw'][i]))
            self.lp_a0[i] = math.sin(6.283188 * (hz / sr))
            self.lp_b1[i] = self.lp_a0[i] - 1.0
        self.hp_cut = np.full(n, f32(cfg['highpass_hz']))
        self.hp_ratio = self.hp_cut / (6.283188 * sr)
        ms = cfg['ramp_ms']
        self._ramp_n = {k: int(round(ms[k] * sr / 1000.0)) for k in ms}
        self.ramps = np.zeros((NR, 3, n))
        self.ramp_left = np.zeros(NR, np.int64)
        self._ramp_set(R_GAMMA, [map_mod(v) for v in cfg['mod_raw']], self._ramp_n['mod'])
        self._ramp_set(R_DT, [map_rate(v) for v in cfg['rate_raw']], self._ramp_n['rate'])
        self._ramp_set(R_GAIN, [map_gain(v) for v in cfg['gain_raw']], self._ramp_n['gain'])
        self.ramps[R_DAMP, 0, :] = 1.0
        self._ramp_set(R_DAMP, [map_damp(v) for v in cfg['damp_raw']], self._ramp_n['damp'])
        self._ramp_set(R_OMEGA, [cfg['omega']] * n, self._ramp_n['omega'])
        self._ramp_set(R_GINT, [map_interaction(interaction_raw)] * n, self._ramp_n['interaction'])
        self.G = np.array(cfg['matrix'], np.float64)
        if self.G.shape != (n, n):
            raise ValueError(f"engine {ENGINE_ID}: matrix must be 8 x 8")
        self.G_tgt = self.G.copy(); self.G_inc = np.zeros_like(self.G)
        self.ints = np.zeros(2, np.int64)
        self.consts = np.array([float(cfg['node_clip']),
                                2.0 * math.sin(math.pi * cfg['svf_hp_hz'][0] / sr), 1.0 - cfg['svf_res'],
                                2.0 * math.sin(math.pi * cfg['svf_hp_hz'][1] / sr), 1.0 - cfg['svf_res']])
        v = 0.25 * np.array(cfg['pan'], np.float64)
        self.panL = np.cos(2.0 * np.pi * v); self.panR = np.cos(2.0 * np.pi * (v + 0.75))
        routes = cfg['output_routes']
        self.routeL = np.array(routes['L'], np.float64); self.routeR = np.array(routes['R'], np.float64)
        self.D = int(cfg['matrix_delay_samples']) + int(cfg['extra_feedback_samples'])
        self.ring = np.zeros((n, self.D + 1))
        self._scratch = np.zeros(n)

    def _ramp_set(self, which, target, samples):
        """line~: move from the current value to `target` over `samples` (0 = jump)."""
        tgt = np.array(target, np.float64)
        cur = self.ramps[which, 0]
        if samples <= 0:
            self.ramps[which, 0, :] = tgt; self.ramps[which, 2, :] = 0.0; self.ramp_left[which] = 0
        else:
            self.ramps[which, 2, :] = (tgt - cur) / samples; self.ramp_left[which] = samples
        self.ramps[which, 1, :] = tgt

    def _kernel(self, n, out):
        _render(n, out, self.node, self.resets, self.px1, self.px2, self.py1, self.py2,
                self.a0, self.a1, self.a2, self.b1, self.b2, self.count,
                self.lp_a0, self.lp_b1, self.hp_ratio, self.hp_cut, self.ramps, self.ramp_left,
                self.G, self.G_tgt, self.G_inc, self.ints, self.ring, self.consts,
                self.panL, self.panR, self.routeL, self.routeR, self._scratch)

    # -- field -> frequencies ---------------------------------------------------------
    def _set_grid(self, grid):
        self._grid = np.array(grid, np.uint8, copy=True)
        self.counts = region_counts(self._grid)
        self.u_field = field_u(self.counts)

    def _apply_freqs(self):
        """The eight setFreqN targets from the held vector + scale / depth (atomic)."""
        self.ratio = ratio_of(self.u_held, self.params['scale'], self.params['depth'])
        self.ffreq = bank_freqs(self.base, self.ratio)
        a0, a2, b1, b2 = calc_coeffs(self.ffreq, self.fQ, self.sr)
        self.a0[:] = a0; self.a2[:] = a2; self.b1[:] = b1; self.b2[:] = b2

    # -- SoundEngine ------------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._build(int(self.params['interaction']))
        self._set_grid(grid)
        self.frozen = bool(int(self.params['freeze_ca']))
        self.u_held = self.u_field.copy()
        self._apply_freqs()
        self.gain_prev = self._eff(gain)

    def update_field(self, grid, exc):
        self._set_grid(grid)
        if not self.frozen:
            self.u_held = self.u_field.copy()
        self._apply_freqs()

    def set_params(self, params):
        old = dict(self.params)
        super().set_params(params)
        if self._grid is None:
            return
        if int(self.params['interaction']) != int(old['interaction']):
            self._ramp_set(R_GINT, [map_interaction(int(self.params['interaction']))] * N_NODES,
                           self._ramp_n['interaction'])
        frozen = bool(int(self.params['freeze_ca']))
        if frozen != self.frozen:
            self.frozen = frozen
            if not frozen:
                self.u_held = self.u_field.copy()
        self._apply_freqs()

    def render(self, gain, t_samples):
        y, peak, n_clip = self.render_float(gain)
        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(pcm), peak, n_clip

    def render_float(self, gain):
        """One block (block, 2) after FIXED_SCALE and the ramped bench gain, before the
        int16 clip, + pre-clip peak / clipped sample count."""
        out = self._out
        self._kernel(out.shape[0], out)
        eff = self._eff(gain)
        y = out * (self.gain_prev + (eff - self.gain_prev) * self._ramp)[:, None]
        self.gain_prev = eff
        a = np.abs(y)
        peak = float(a.max()) if a.size else 0.0
        return y, peak, int(np.count_nonzero(a > 1.0))

    def raw_block(self):
        """One block of the raw master sums (no gain) -- verification helper."""
        out = np.zeros_like(self._out)
        self._kernel(out.shape[0], out)
        return out

    def reset(self, gain=0.0):
        self.init(self._grid, None, gain)

    def display(self):
        return dict(u=[float(v) for v in self.u_held], u_field=[float(v) for v in self.u_field],
                    counts=[int(v) for v in self.counts], ratio=[float(v) for v in self.ratio],
                    frozen=bool(self.frozen), links=float(self.ramps[R_GINT, 0, 0]),
                    resets=int(self.resets.sum()), model=self.model_version)

    # -- snapshot -----------------------------------------------------------------------
    _ARRAYS = ('node', 'resets', 'px1', 'px2', 'py1', 'py2', 'a0', 'a1', 'a2', 'b1', 'b2',
               'count', 'ffreq', 'fQ', 'lp_a0', 'lp_b1', 'hp_ratio', 'hp_cut', 'ramps',
               'ramp_left', 'G', 'G_tgt', 'G_inc', 'ints', 'ring', 'u_held', 'u_field',
               'counts', 'ratio')

    def export_state(self):
        st = dict(version=self.STATE_VERSION, engine_id=ENGINE_ID, model_version=self.model_version,
                  params=dict(self.params), frozen=bool(self.frozen), gain_prev=float(self.gain_prev),
                  delay=int(self.D))
        for name in self._ARRAYS:
            st[name] = np.array(getattr(self, name), copy=True)
        return st

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != ENGINE_ID:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {ENGINE_ID!r}")
        if state.get('model_version') != self.model_version:
            raise ValueError(f"engine {ENGINE_ID}: model version {state.get('model_version')!r}"
                             f" != {self.model_version!r}")
        if int(state.get('delay', -1)) != self.D:
            raise ValueError(f"engine {ENGINE_ID}: matrix delay {state.get('delay')} != {self.D}")
        params = state.get('params')
        if not isinstance(params, dict) or sorted(params) != sorted(p[0] for p in PARAMS):
            raise ValueError(f"engine {ENGINE_ID}: state params do not match the registry")
        want = {name: getattr(self, name).shape for name in self._ARRAYS}
        arrays = {}
        for name, shape in want.items():
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != shape:
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            kind = getattr(self, name).dtype
            arrays[name] = np.array(a, kind, copy=True)
        gp = float(state.get('gain_prev', 0.0))
        if not math.isfinite(gp):
            raise ValueError(f"engine {ENGINE_ID}: bad scalar in the state")
        self.params = dict(params)
        self._build(int(self.params['interaction']))
        for name, a in arrays.items():
            setattr(self, name, a)
        self._grid = np.array(grid, np.uint8, copy=True)
        self.frozen = bool(state.get('frozen', False))
        self.gain_prev = gp

    # -- internals ----------------------------------------------------------------------
    def _eff(self, gain):
        return float(gain) * FIXED_SCALE


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: GutterFieldEngine(ctx, params),
                    overlay=overlay))
