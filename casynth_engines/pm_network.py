"""N -- tonal phase-modulation network (engine id `pm_network`, REQ 2026-09-14
section 4).

Four generators with frequency ratios [1, 2, 3, 4] to f0 share ONE base phase
theta (fixed zero initial offsets).  Six directed links run only from a higher
index to a lower one; per internal audio sample, i = 3, 2, 1, 0:

    x_i = sin((i+1)*theta + beta * sum_{j > i} W_ij * x_j)      (x_j of the SAME sample)
    raw = (x_0 + x_1 + x_2 + x_3) / 4
    theta += 2*pi*f0 / internal_sr

No closed feedback, no delayed edges: with constant W the signal is a function
of theta, periodic in 1/f0.  beta = 0 is the plain sum of the four independent
generators with the same output weights.

Field -> six weights (section 4.2): six fixed Gaussian masks K_k on the torus,
centres x = (W-1)*[1/6, 1/2, 5/6], y = (H-1)*[1/4, 3/4], sigma = min(H, W)/6
cells, shortest torus distance per axis; listed row-major and mapped to
(i, j) = (0,1), (0,2), (0,3), (1,2), (1,3), (2,3):

    W_ij = sum(K_k * g) / sum(K_k) / 3          (denominators fixed by the size)

New target weights are reached causally with tau = 30 ms (one-pole, internal
rate); beta the same; the first W at init come straight from the initial
field; a field edit never resets the phase.  Frozen links (`freeze_links`):
on -> the W reached so far are held and new fields do not move them; off ->
the target is the current field again (smoothed).  A gate (a separate
"field present" channel, NOT density -> loudness) closes over 20 ms on an
empty field and opens over 20 ms on a non-empty one; sound enters through it
after init.

Rendering (section 4.3): the network runs at OVERSAMPLE x sr (4x); a
linear-phase Kaiser FIR low-pass (cut-off FIR_CUTOFF*sr) decimates to sr with
an exact delay of FIR_HALF output samples (the 8x variant used by the quality
check has the same delay); a fixed first-order 5 Hz high-pass removes DC;
both filters run continuously and their memory is part of the snapshot.
`gain` (already MASTER_GAIN*vol*level) times 10^(trim_db/20) is applied once,
ramped across the block; peak / n_clip are measured before the int16 clip.
Fixed raw headroom: |raw| <= 1 (mean of four sines) scaled by RAW_SCALE = 8
(+18 dB) before the master gain -> worst case 8 * MASTER_GAIN = 0.32 of full
scale at trim 0, no overflow (only the +trim range can clip; counted).  The
constant level correction found for the prepared comparisons (RMS matched to
the Laplace side on the F3 Pulsar scene, 2026-09-14) is the registry default
trim_db.
"""
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from casynth_config import _RAMP
from .engine_api import SoundEngine
from .registry import EngineSpec, register

ENGINE_ID = 'pm_network'
LABEL = 'Network'
PARAMS = [('coupling', 'Coupling', 0.0, 4.0, False, 2.0),
          ('freeze_links', 'Frozen links', 0, 1, True, 0),
          ('trim_db', 'Level (dB)', -24.0, 24.0, False, 3.0)]
RATIOS = (1, 2, 3, 4)
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))     # (i, j): j modulates i
MASK_X = (1.0 / 6.0, 0.5, 5.0 / 6.0)
MASK_Y = (0.25, 0.75)
OVERSAMPLE = 4
RAW_SCALE = 8.0               # fixed raw headroom scale (see the module docstring)
FIR_HALF = 32                 # decimator delay in OUTPUT samples (taps = 2*R*FIR_HALF + 1)
FIR_CUTOFF = 0.476            # low-pass cut-off, fraction of sr (21 kHz at 44.1 kHz)
KAISER_BETA = 8.0
HPF_HZ = 5.0
TAU_W = 0.030                 # link smoothing (s)
TAU_BETA = 0.030              # coupling smoothing (s)
GATE_S = 0.020                # field-present gate ramp (s)
STATE_VERSION = 1


# -- field -> weights ----------------------------------------------------------
def mask_centres(rows, cols):
    """Six (x, y) centres in cell coordinates, row-major (y outer, x inner)."""
    return [((cols - 1) * fx, (rows - 1) * fy) for fy in MASK_Y for fx in MASK_X]


def mask_sigma(rows, cols):
    return min(rows, cols) / 6.0


def masks(rows, cols):
    """(6, rows, cols) Gaussian masks on the torus."""
    sigma = mask_sigma(rows, cols)
    r = np.arange(rows, dtype=np.float64)[:, None]
    c = np.arange(cols, dtype=np.float64)[None, :]
    out = np.empty((len(EDGES), rows, cols))
    for k, (x, y) in enumerate(mask_centres(rows, cols)):
        dx = (c - x + cols / 2.0) % cols - cols / 2.0
        dy = (r - y + rows / 2.0) % rows - rows / 2.0
        out[k] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    return out


def field_weights(grid, K=None):
    """Six weights W_ij (EDGES order) of a 0/1 field."""
    g = np.asarray(grid, np.float64)
    K = masks(*g.shape) if K is None else K
    return np.array([(K[k] * g).sum() / K[k].sum() / 3.0 for k in range(len(EDGES))])


# -- the network ---------------------------------------------------------------
def pm_network(theta, W, beta):
    """Vectorised network: theta in CYCLES (N,), W (6,) or (6, N) in EDGES
    order, beta scalar or (N,).  Returns (x (4, N), raw (N,))."""
    th = np.asarray(theta, np.float64)
    W = np.asarray(W, np.float64)
    if W.ndim == 1:
        W = W[:, None]
    beta = np.asarray(beta, np.float64)
    two_pi = 2.0 * np.pi
    w01, w02, w03, w12, w13, w23 = W
    x3 = np.sin(two_pi * ((4.0 * th) % 1.0))
    x2 = np.sin(two_pi * ((3.0 * th) % 1.0) + beta * (w23 * x3))
    x1 = np.sin(two_pi * ((2.0 * th) % 1.0) + beta * (w12 * x2 + w13 * x3))
    x0 = np.sin(two_pi * (th % 1.0) + beta * (w01 * x1 + w02 * x2 + w03 * x3))
    x = np.stack([x0, x1, x2, x3])
    return x, x.sum(axis=0) * 0.25


def decimator_taps(sr, oversample):
    """Linear-phase Kaiser FIR low-pass for the internal rate oversample*sr:
    2*oversample*FIR_HALF + 1 taps -> delay FIR_HALF output samples."""
    taps = 2 * int(oversample) * FIR_HALF + 1
    return signal.firwin(taps, FIR_CUTOFF * sr, fs=oversample * sr, window=('kaiser', KAISER_BETA))


def hpf_coeffs(sr):
    """First-order high-pass at HPF_HZ (bilinear), (b, a) for scipy.signal.lfilter."""
    t = math.tan(math.pi * HPF_HZ / sr)
    a1 = (1.0 - t) / (1.0 + t)
    b0 = (1.0 + a1) / 2.0
    return np.array([b0, -b0]), np.array([1.0, -a1])


# -- overlay (masks) + link display ---------------------------------------------
def overlay(params, rows, cols):
    sigma = mask_sigma(rows, cols)
    circles, labels = [], []
    for k, (x, y) in enumerate(mask_centres(rows, cols)):
        i, j = EDGES[k]
        circles.append((x, y, sigma))
        labels.append((x, y, f"{j}>{i}"))
    return dict(circles=circles, labels=labels,
                text="links j>i: the field around each mark sets how much j modulates i")


# -- engine ----------------------------------------------------------------------
class PMNetworkEngine(SoundEngine):
    STATE_VERSION = STATE_VERSION

    def __init__(self, ctx, params, oversample=OVERSAMPLE):
        super().__init__(ctx, params)
        self.R = int(oversample)
        self.isr = float(ctx.sr * self.R)
        self.taps = decimator_taps(ctx.sr, self.R)
        self._taps_rev = self.taps[::-1].copy()
        self._hpf_b, self._hpf_a = hpf_coeffs(ctx.sr)
        n = ctx.block * self.R
        j = np.arange(1, n + 1, dtype=np.float64)
        self._decay_w = np.exp(-j / (TAU_W * self.isr))
        self._decay_beta = np.exp(-j / (TAU_BETA * self.isr))
        self._gate_step = j / (GATE_S * self.isr)
        self._grid = None
        self._K = None
        self._clear_audio(0.0)

    # -- SoundEngine ------------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._clear_audio(self._eff(gain))
        self._set_grid(grid)
        w = field_weights(self._grid, self._K)
        self.W = w.copy()
        self.W_target = w.copy()
        self.frozen = bool(int(self.params['freeze_links']))
        self.beta = float(self.params['coupling'])

    def update_field(self, grid, exc):
        self._set_grid(grid)
        if not self.frozen:
            self.W_target = field_weights(self._grid, self._K)

    def set_params(self, params):
        was_frozen = self.frozen
        super().set_params(params)
        self.frozen = bool(int(self.params['freeze_links']))
        if self._grid is None:
            return
        if self.frozen and not was_frozen:
            self.W_target = self.W.copy()                # hold what has been reached
        elif was_frozen and not self.frozen:
            self.W_target = field_weights(self._grid, self._K)

    def render(self, gain, t_samples):
        y, peak, n_clip = self.render_float(gain)
        out = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(np.repeat(out[:, None], self.ctx.channels, axis=1)), \
            peak, n_clip

    def render_float(self, gain):
        """One block as float (mono, before the int16 clip) + peak / n_clip."""
        n = self.ctx.block * self.R
        step = self.ctx.f0 / self.isr
        theta = (self.theta + np.arange(n) * step) % 1.0
        self.theta = (self.theta + n * step) % 1.0
        W_n = self.W_target[:, None] + (self.W - self.W_target)[:, None] * self._decay_w[None, :]
        beta_t = float(self.params['coupling'])
        beta_n = beta_t + (self.beta - beta_t) * self._decay_beta
        gate_n = np.clip(self.gate + (1.0 if self.gate_target else -1.0) * self._gate_step,
                         0.0, 1.0)
        self.W = W_n[:, -1].copy()
        self.beta = float(beta_n[-1])
        self.gate = float(gate_n[-1])
        _x, raw = pm_network(theta, W_n, beta_n)
        raw = raw * gate_n
        buf = np.concatenate([self.fir_mem, raw])
        y = sliding_window_view(buf, len(self.taps))[::self.R] @ self._taps_rev
        self.fir_mem = buf[-(len(self.taps) - 1):].copy()
        y, self.hpf_zi = signal.lfilter(self._hpf_b, self._hpf_a, y, zi=self.hpf_zi)
        eff = self._eff(gain)
        y = y * (self.gain_prev + (eff - self.gain_prev) * _RAMP)
        self.gain_prev = eff
        peak = float(np.abs(y).max()) if len(y) else 0.0
        n_clip = int(np.count_nonzero(np.abs(y) > 1.0))
        return y, peak, n_clip

    def reset(self, gain=0.0):
        self.init(self._grid, None, gain)

    def display(self):
        return dict(W=[float(v) for v in self.W], W_target=[float(v) for v in self.W_target],
                    frozen=bool(self.frozen), gate=float(self.gate), beta=float(self.beta))

    # -- snapshot ------------------------------------------------------------------------
    def export_state(self):
        return dict(version=self.STATE_VERSION, engine_id=ENGINE_ID, params=dict(self.params),
                    oversample=int(self.R), taps=int(len(self.taps)),
                    theta=float(self.theta), beta=float(self.beta), gate=float(self.gate),
                    frozen=bool(self.frozen), gain_prev=float(self.gain_prev),
                    W=self.W.copy(), W_target=self.W_target.copy(),
                    fir_mem=self.fir_mem.copy(), hpf_zi=self.hpf_zi.copy())

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != ENGINE_ID:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {ENGINE_ID!r}")
        if (state.get('oversample'), state.get('taps')) != (self.R, len(self.taps)):
            raise ValueError(f"engine {ENGINE_ID}: render layout "
                             f"{state.get('oversample')}x/{state.get('taps')} taps != "
                             f"{self.R}x/{len(self.taps)} taps")
        want = {'W': (len(EDGES),), 'W_target': (len(EDGES),),
                'fir_mem': (len(self.taps) - 1,), 'hpf_zi': (1,)}
        arrays = {}
        for name, shape in want.items():
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != shape:
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            arrays[name] = np.array(a, np.float64, copy=True)
        scalars = {k: float(state[k]) for k in ('theta', 'beta', 'gate', 'gain_prev')}
        if not all(math.isfinite(v) for v in scalars.values()) \
                or not (0.0 <= scalars['theta'] < 1.0) or not (0.0 <= scalars['gate'] <= 1.0):
            raise ValueError(f"engine {ENGINE_ID}: bad scalar in the state")
        self.params = dict(state['params'])
        self._set_grid(grid)
        self.theta, self.beta = scalars['theta'], scalars['beta']
        self.gate, self.gain_prev = scalars['gate'], scalars['gain_prev']
        self.frozen = bool(state['frozen'])
        self.W, self.W_target = arrays['W'], arrays['W_target']
        self.fir_mem, self.hpf_zi = arrays['fir_mem'], arrays['hpf_zi']

    # -- internals --------------------------------------------------------------------------
    def _clear_audio(self, gain):
        self.theta = 0.0
        self.W = np.zeros(len(EDGES))
        self.W_target = np.zeros(len(EDGES))
        self.frozen = False
        self.beta = 0.0
        self.gate = 0.0
        self.gate_target = False
        self.fir_mem = np.zeros(len(self.taps) - 1)
        self.hpf_zi = np.zeros(1)
        self.gain_prev = float(gain)

    def _set_grid(self, grid):
        self._grid = np.array(grid, np.uint8, copy=True)
        if self._K is None or self._K.shape[1:] != self._grid.shape:
            self._K = masks(*self._grid.shape)
        self.gate_target = bool(self._grid.any())

    def _eff(self, gain):
        return float(gain) * RAW_SCALE * 10.0 ** (float(self.params['trim_db']) / 20.0)


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: PMNetworkEngine(ctx, params),
                    overlay=overlay))
