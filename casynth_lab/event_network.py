"""N2 side B -- field events excite a decaying delay network (engine id
`ca_event_network`, REQ memory/req-network-events-n2-2026-09-16.md, "Б — событийная
сеть").  A new model, NOT a port of Gutter: N2 is the experiment number.

Events (block boundary, before the samples of the block):
    E   = (G != G_prev)                 birth and death both count, no sign
    e_i = sum(K_i * E)                  periodic weights of periodic_readout
    a_i = e_i / (e_i + 2)
    G_prev = copy(G)
    z_fast[i] += a_i ;  z_slow[i] += a_i
G_prev is the field the previous render has processed; G the field after all
commands / CA steps of the block (several edits inside one block merge into
the final change).  Fresh init / reset: all states zero and G_prev = 0, so the
initial live cells give ONE packet.  restore_state brings back G_prev and the
pending field: no packet.  The bench's `exc` argument (the held excitation of
another algorithm) is ignored.  Parameter changes are not events.

Per sample k (float64), first the excitation, then the decay of the states:
    exc_i = 0.75 ((1 - q_fast) z_fast[i] - (1 - q_slow) z_slow[i]) / (q_slow - q_fast)
    z_fast[i] *= q_fast ; z_slow[i] *= q_slow        q = exp(-1 / (SR tau)), tau 0.25 / 2 ms
then ALL delays are read and ALL loss filters updated, then the new values:
    d_i[k] = v_i[k - D_i]                            history before k = 0 is zero
    l_i[k] = (1 - q) d_i[k] + q l_i[k-1]             q = exp(-2 pi 6000 / SR)
    v_i[k] = tanh(exc_i[k] + rho sum_j M_ij l_j[k])   M_ij = 0.25 - delta_ij (orthogonal)
    D = [149, 211, 293, 401, 547, 743, 1009, 1361]
Output from the loss filters, p_i = (j + 0.5) / 4 for i = 4 l + j:
    L = sum_i cos(pi p_i / 2) l_i / 8 ;  R = sum_i sin(pi p_i / 2) l_i / 8
then OUT_SCALE = 96 and ONCE the bench gain (ramped across the block as in
gutter_field; gain_prev is part of the state).  No direct path from the pulse
to the output: without events the output is exactly zero, after them the
network decays (rho < 1, |tanh x| <= |x|, orthogonal M, loss filter).

One control: rho ("Response", 0.60..0.95, default 0.88); a manual change is a
linear ramp of 20 ms on the sample path; the ramp is part of the snapshot.
Delays, matrix, pulse strength / shape and the readout are fixed for N2.
Only SR 44100 / block 352 / stereo are supported (the configuration was
calibrated there); anything else raises at construction.
"""
import math

import numpy as np

from .engine_api import SoundEngine
from .registry import EngineSpec, register
from . import periodic_readout as pr

try:
    import numba as _numba
    _jit = _numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except ImportError:                    # pragma: no cover
    def _jit(f):
        return f
    HAVE_NUMBA = False

ENGINE_ID = 'ca_event_network'
LABEL = 'Events'
PARAMS = [('rho', 'Response', 0.60, 0.95, False, 0.88)]
MODEL_VERSION = 'ca_event_network_n2_v1'
STATE_VERSION = 1
SR_REQUIRED = 44100
BLOCK_REQUIRED = 352
N_NODES = pr.N_NODES
DELAYS = (149, 211, 293, 401, 547, 743, 1009, 1361)
LOSS_HZ = 6000.0
PULSE_FAST_S = 0.00025
PULSE_SLOW_S = 0.002
PULSE_STRENGTH = 0.75
EVENT_SAT = 2.0                        # a = e / (e + EVENT_SAT)
MATRIX_OFF = 0.25                      # M_ij = 0.25 - delta_ij
RHO_RAMP_MS = 20.0
OUT_SCALE = 96.0
OVERLAY_TEXT = "8 nodes: a cell change near a node strikes its delay line (periodic weights)"

C_Q, C_QF, C_QS, C_STRENGTH = range(4)
I_K, I_RHO_LEFT = range(2)
R_CUR, R_TGT, R_INC = range(3)


def matrix():
    return np.full((N_NODES, N_NODES), MATRIX_OFF) - np.eye(N_NODES)


def pans():
    p = np.array([(j + 0.5) / pr.NODE_COLS for _l in range(pr.NODE_ROWS)
                  for j in range(pr.NODE_COLS)])
    return np.cos(np.pi * p / 2.0), np.sin(np.pi * p / 2.0)


def consts(sr):
    return np.array([math.exp(-2.0 * math.pi * LOSS_HZ / sr),
                     math.exp(-1.0 / (sr * PULSE_FAST_S)),
                     math.exp(-1.0 / (sr * PULSE_SLOW_S)),
                     PULSE_STRENGTH])


def packet(K, prev, cur):
    """(e, a): weighted event counts and the saturated packet for prev -> cur."""
    E = np.asarray(prev, np.uint8) != np.asarray(cur, np.uint8)
    e = pr.weighted_counts(K, E)
    return e, e / (e + EVENT_SAT)


@_jit
def _render(n, out, ring, D, lfilt, zf, zs, M, rho, ints, cst, panL, panR, exc, level):
    """`n` samples into out (n, 2) = L / R of the formulas (before OUT_SCALE / gain).
    level[i] = max |l_i| over the block (display only)."""
    nn = ring.shape[0]
    q = cst[C_Q]; qf = cst[C_QF]; qs = cst[C_QS]; strength = cst[C_STRENGTH]
    for i in range(nn):
        level[i] = 0.0
    for s in range(n):
        k = ints[I_K]
        left = ints[I_RHO_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                rho[R_CUR] = rho[R_TGT]
            else:
                rho[R_CUR] = rho[R_CUR] + rho[R_INC]
            ints[I_RHO_LEFT] = left
        r = rho[R_CUR]
        # excitation, then the decay of the pulse states
        for i in range(nn):
            exc[i] = strength * ((1.0 - qf) * zf[i] - (1.0 - qs) * zs[i]) / (qs - qf)
            zf[i] = zf[i] * qf
            zs[i] = zs[i] * qs
        # read ALL delays, update ALL loss filters
        for i in range(nn):
            d = ring[i, k % D[i]]
            lfilt[i] = (1.0 - q) * d + q * lfilt[i]
            a = lfilt[i] if lfilt[i] >= 0.0 else -lfilt[i]
            if a > level[i]:
                level[i] = a
        # new values, written into the delays
        for i in range(nn):
            acc = 0.0
            for j in range(nn):
                acc += M[i, j] * lfilt[j]
            ring[i, k % D[i]] = math.tanh(exc[i] + r * acc)
        L = 0.0
        R = 0.0
        for i in range(nn):
            L += panL[i] * lfilt[i]
            R += panR[i] * lfilt[i]
        out[s, 0] = L / nn
        out[s, 1] = R / nn
        ints[I_K] = k + 1


class EventNetworkEngine(SoundEngine):
    STATE_VERSION = STATE_VERSION

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        if int(ctx.sr) != SR_REQUIRED or int(ctx.block) != BLOCK_REQUIRED or ctx.channels != 2:
            raise ValueError(f"engine {ENGINE_ID}: only sr {SR_REQUIRED} / block {BLOCK_REQUIRED} / "
                             f"stereo are supported (got {ctx.sr} / {ctx.block} / {ctx.channels})")
        self.engine_id = ENGINE_ID
        self.model_version = MODEL_VERSION
        self.sr = float(ctx.sr)
        n = int(ctx.block)
        self._ramp = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n))) * 0.5
        self._out = np.zeros((n, 2))
        self._exc = np.zeros(N_NODES)
        self.level = np.zeros(N_NODES)
        self.rho_ramp_n = int(round(RHO_RAMP_MS / 1000.0 * self.sr))
        self._K = None
        self._grid = None
        self._zero()
        self.gain_prev = 0.0
        self._kernel(0, np.zeros((0, 2)))          # compile / load the cached kernel

    # -- model state ----------------------------------------------------------------
    def _zero(self):
        self.D = np.array(DELAYS, np.int64)
        self.ring = np.zeros((N_NODES, int(self.D.max())))
        self.lfilt = np.zeros(N_NODES)
        self.zf = np.zeros(N_NODES)
        self.zs = np.zeros(N_NODES)
        self.M = matrix()
        self.panL, self.panR = pans()
        self.consts = consts(self.sr)
        self.ints = np.zeros(2, np.int64)
        r = float(self.params['rho'])
        self.rho = np.array([r, r, 0.0])
        self.last_e = np.zeros(N_NODES)
        self.last_a = np.zeros(N_NODES)
        self.G_prev = None

    def _kernel(self, n, out):
        _render(n, out, self.ring, self.D, self.lfilt, self.zf, self.zs, self.M, self.rho,
                self.ints, self.consts, self.panL, self.panR, self._exc, self.level)

    def _set_grid(self, grid):
        self._grid = np.array(grid, np.uint8, copy=True)
        if self._K is None or self._K.shape[1:] != self._grid.shape:
            self._K = pr.weights(*self._grid.shape)

    def _events(self):
        """The packet of the pending field against G_prev (block boundary)."""
        if self.G_prev is None or self.G_prev.shape != self._grid.shape:
            self.G_prev = np.zeros_like(self._grid)
        e, a = packet(self._K, self.G_prev, self._grid)
        self.last_e = e
        self.last_a = a
        self.zf += a
        self.zs += a
        self.G_prev = self._grid.copy()

    # -- SoundEngine ----------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._zero()
        self._set_grid(grid)
        self.level[:] = 0.0
        self.gain_prev = self._eff(gain)

    def update_field(self, grid, exc):
        self._set_grid(grid)

    def set_params(self, params):
        old = float(self.params['rho'])
        super().set_params(params)
        new = float(self.params['rho'])
        if new != old:
            self.rho[R_TGT] = new
            if self.rho_ramp_n > 0:
                self.rho[R_INC] = (new - self.rho[R_CUR]) / self.rho_ramp_n
                self.ints[I_RHO_LEFT] = self.rho_ramp_n
            else:
                self.rho[R_CUR] = new
                self.rho[R_INC] = 0.0
                self.ints[I_RHO_LEFT] = 0

    def render(self, gain, t_samples):
        y, peak, n_clip = self.render_float(gain)
        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(pcm), peak, n_clip

    def render_float(self, gain):
        """One block (block, 2) after OUT_SCALE and the ramped bench gain, before the
        int16 clip, + pre-clip peak / clipped sample count.  The events of the block
        are applied first."""
        self._events()
        out = self._out
        self._kernel(out.shape[0], out)
        eff = self._eff(gain)
        y = out * (self.gain_prev + (eff - self.gain_prev) * self._ramp)[:, None]
        self.gain_prev = eff
        a = np.abs(y)
        peak = float(a.max()) if a.size else 0.0
        return y, peak, int(np.count_nonzero(a > 1.0))

    def raw_block(self, events=True):
        """One block of the L / R formulas (no OUT_SCALE, no gain) -- verification helper;
        events=False renders the sample path only (no packet, G_prev untouched)."""
        if events:
            self._events()
        out = np.zeros_like(self._out)
        self._kernel(out.shape[0], out)
        return out

    def inject(self, a):
        """Test hook: add a packet vector directly to the pulse states (no field)."""
        a = np.asarray(a, np.float64)
        self.zf += a
        self.zs += a

    def reset(self, gain=0.0):
        self.init(self._grid, None, gain)

    def display(self):
        return dict(events=[float(v) for v in self.last_a], e=[float(v) for v in self.last_e],
                    level=[float(v) for v in self.level], rho=float(self.rho[R_CUR]),
                    rho_target=float(self.rho[R_TGT]), ramp_left=int(self.ints[I_RHO_LEFT]),
                    model=self.model_version)

    # -- snapshot ---------------------------------------------------------------------
    _ARRAYS = ('ring', 'D', 'lfilt', 'zf', 'zs', 'M', 'ints', 'rho', 'last_e', 'last_a', 'level')

    def export_state(self):
        if self._grid is None:
            raise ValueError(f"engine {ENGINE_ID}: no field yet (init first)")
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  model_version=self.model_version, params=dict(self.params),
                  sr=int(self.sr), block=int(self._out.shape[0]),
                  gain_prev=float(self.gain_prev), out_scale=OUT_SCALE,
                  grid_pending=self._grid.copy(),
                  grid_prev=(np.zeros_like(self._grid) if self.G_prev is None
                             else self.G_prev.copy()))
        for name in self._ARRAYS:
            st[name] = np.array(getattr(self, name), copy=True)
        return st

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != self.engine_id:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {self.engine_id!r}")
        if state.get('model_version') != self.model_version:
            raise ValueError(f"engine {ENGINE_ID}: model version {state.get('model_version')!r}"
                             f" != {self.model_version!r}")
        if int(state.get('sr', -1)) != int(self.sr) or int(state.get('block', -1)) != self._out.shape[0]:
            raise ValueError(f"engine {ENGINE_ID}: state sr / block do not match the context")
        params = state.get('params')
        if not isinstance(params, dict) or sorted(params) != sorted(p[0] for p in PARAMS):
            raise ValueError(f"engine {ENGINE_ID}: state params do not match the registry")
        fresh = dict(D=np.array(DELAYS, np.int64), ring=np.zeros((N_NODES, max(DELAYS))),
                     lfilt=np.zeros(N_NODES), zf=np.zeros(N_NODES), zs=np.zeros(N_NODES),
                     M=matrix(), ints=np.zeros(2, np.int64), rho=np.zeros(3),
                     last_e=np.zeros(N_NODES), last_a=np.zeros(N_NODES), level=np.zeros(N_NODES))
        arrays = {}
        for name, proto in fresh.items():
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != proto.shape:
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {proto.shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            arrays[name] = np.array(a, proto.dtype, copy=True)
        if not np.array_equal(arrays['D'], fresh['D']) or not np.array_equal(arrays['M'], fresh['M']):
            raise ValueError(f"engine {ENGINE_ID}: delays / matrix of the state are not the N2 model")
        if arrays['ints'][I_K] < 0 or arrays['ints'][I_RHO_LEFT] < 0:
            raise ValueError(f"engine {ENGINE_ID}: negative counters in the state")
        g = np.asarray(grid, np.uint8)
        gp = state.get('grid_pending')
        gv = state.get('grid_prev')
        for name, arr in (('grid_pending', gp), ('grid_prev', gv)):
            if not isinstance(arr, np.ndarray) or arr.shape != g.shape:
                raise ValueError(f"engine {ENGINE_ID}: {name} missing or shape "
                                 f"{getattr(arr, 'shape', None)} != {g.shape}")
        if not np.array_equal(np.asarray(gp, np.uint8), g):
            raise ValueError(f"engine {ENGINE_ID}: the pending field of the state differs "
                             f"from the bench field")
        gain_prev = float(state.get('gain_prev', 0.0))
        if not math.isfinite(gain_prev) or float(state.get('out_scale', OUT_SCALE)) != OUT_SCALE:
            raise ValueError(f"engine {ENGINE_ID}: bad scalar in the state")
        self.params = dict(params)
        self._zero()
        for name, a in arrays.items():
            setattr(self, name, a)
        self.panL, self.panR = pans()
        self.consts = consts(self.sr)
        self._set_grid(g)
        self.G_prev = np.array(gv, np.uint8, copy=True)
        self.gain_prev = gain_prev

    # -- internals ----------------------------------------------------------------------
    def _eff(self, gain):
        return float(gain) * OUT_SCALE


def overlay(params, rows, cols):
    return pr.overlay(rows, cols, OVERLAY_TEXT)


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: EventNetworkEngine(ctx, params),
                    overlay=overlay))
