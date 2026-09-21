"""N3 -- the field tunes the resonances, its events strike them (engine id
`ca_tuned_events`, REQ memory/req-network-combined-n3-2026-09-16.md).  One model,
two modes of the comparison parameter `field_tuning`:

    0  "Hits, fixed resonances"  -- every frequency multiplier is 1; the field acts
                                    on the sound through its events only
    1  "Hits + field tuning"     -- the same hits, plus the resonators follow the
                                    live cells (the N1 / N2-A law, scale 1, depth 1)

    cell changes -> pulses and the N2-B delay network -> eight tuned banks -> sound
    live cells   -> eight frequency multipliers  ---------------^

Field readout (block boundary, before the samples of the block; the shared
periodic weights of periodic_readout for the field G the bench holds):
    n_i     = sum(K_i * G)            u_i = n_i / (n_i + 2)
    ratio_i = 2 ** (2 u_i - 1)        (field_tuning = 1; otherwise 1)
    f_ij    = float32(min(19000, base_ij * ratio_i))     base = CONFIG['filters_hz'] of N1,
                                                         all 24 frequencies of every node
Events exactly as event_network (N2 side B): E = (G != G_prev_rendered), e_i = sum(K_i E),
a_i = e_i / (e_i + 2) into the fast / slow pulse pair; fresh init compares the initial
field with zeros once, restore brings back both fields without a packet, parameter
changes are never events, the bench's `exc` is ignored.

Sample path (float64), per sample k:
    the N2-B network step at rho = 0.88 FIXED (pulse 0.25 / 2 ms x 0.75, delays
    D = 149..1361, M_ij = 0.25 - delta_ij, 6 kHz loss filter, tanh on the write) --
    the same statement order as event_network._render, so the network states of
    this engine equal those of `ca_event_network` for the same field history;
    then every node's loss-filter value l_i[k] drives its bank of 24 resonators
        z_ij[k+1] = r * exp(i theta_ij) * z_ij[k] + l_i[k]      theta_ij = 2 pi f_ij / SR
        b_i[k]    = sum_j Re(z_ij[k+1]) / 24                     r = 10 ** (-3 / (SR T60))
    (two real arrays; both new coordinates from the old Re / Im), then the N2
    panning p_i = (j + 0.5) / 4 for i = 4 l + j:
        mix_L = sum_i b_i cos(pi p_i / 2) / 8 ;  mix_R = sum_i b_i sin(pi p_i / 2) / 8
    then one DC-removal filter per channel, h = exp(-2 pi 20 / SR):
        hp[k] = h * (hp[k-1] + mix[k] - mix[k-1])
    output = hp * OUT_SCALE (4) * bench gain (ramped across the block, gain_prev in the state).
The banks' output never returns into the delay network; nothing but the banks is
heard (no dry N2, no Gutter).  Frequency coefficients are applied at the block
boundary before its first sample; a retune keeps every bank state (an excited tail
goes on with the new tuning; a retune of a silent bank makes no sound).

Controls: `field_tuning` (0 / 1, retunes without reset and without an event) and
`decay_s` ("Decay" T60 0.20..1.50 s, default 0.80): a manual change drives r
linearly to the new value over 882 samples (20 ms); init starts at the tuned r.
Only SR 44100 / block 352 / stereo (as N2).  Own model_version / STATE_VERSION.
"""
import math

import numpy as np

from .engine_api import SoundEngine
from .registry import EngineSpec, register
from . import periodic_readout as pr
from . import event_network as en
from . import gutter_field as gf
from .gutter_field_n1_config import CONFIG as N1_CONFIG

try:
    import numba as _numba
    _jit = _numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except ImportError:                    # pragma: no cover
    def _jit(f):
        return f
    HAVE_NUMBA = False

ENGINE_ID = 'ca_tuned_events'
LABEL = 'Tuned'
PARAMS = [('field_tuning', 'Field tuning', 0, 1, True, 1),
          ('decay_s', 'Decay', 0.20, 1.50, False, 0.80)]
CHOICES = {'field_tuning': ('Fixed', 'Field')}
MODEL_VERSION = 'ca_tuned_events_n3_v1'
STATE_VERSION = 1
SR_REQUIRED = en.SR_REQUIRED
BLOCK_REQUIRED = en.BLOCK_REQUIRED
N_NODES = pr.N_NODES
N_BANK = 24
RHO = 0.88                             # fixed for N3 (the N2 default)
DELAYS = en.DELAYS
HP_HZ = 20.0
DECAY_RAMP_MS = 20.0
OUT_SCALE = 4.0
FREQ_MAX = gf.FREQ_MAX
BASE_HZ = tuple(tuple(float(v) for v in row) for row in N1_CONFIG['filters_hz'])
OVERLAY_TEXT = "8 nodes: cell changes near a node strike its bank; live cells tune it (periodic weights)"

I_K, I_R_LEFT = range(2)
R_CUR, R_TGT, R_INC = range(3)
H_PREV, H_MIX = range(2)               # hp[ch, .] = previous output / previous input


def base_freqs():
    """float64 (8, 24): the explicit N1 bank frequencies (never regenerated, never sorted)."""
    b = np.array(BASE_HZ, np.float64)
    if b.shape != (N_NODES, N_BANK):
        raise ValueError(f"N3: base frequencies {b.shape} != {(N_NODES, N_BANK)}")
    return b


def decay_r(t60_s, sr=SR_REQUIRED):
    """r = 10 ** (-3 / (SR T60))."""
    return 10.0 ** (-3.0 / (float(sr) * float(t60_s)))


def field_ratio(u, tuning):
    """ratio_i: the N1 law with scale 1 / depth 1 when the field tunes, else ones."""
    u = np.asarray(u, np.float64)
    return gf.ratio_of(u, 1.0, 1.0) if tuning else np.ones_like(u)


def trig_tables(ffreq, sr):
    """cos / sin of theta_ij = 2 pi f_ij / SR, elementwise with math.* (the scalar
    reference computes the same values)."""
    ffreq = np.asarray(ffreq, np.float64)
    c = np.empty_like(ffreq)
    s = np.empty_like(ffreq)
    for i in range(ffreq.shape[0]):
        for j in range(ffreq.shape[1]):
            th = 2.0 * math.pi * ffreq[i, j] / sr
            c[i, j] = math.cos(th)
            s[i, j] = math.sin(th)
    return c, s


@_jit
def _render(n, out, ring, D, lfilt, zf, zs, M, rho, ints, cst, zre, zim, cth, sth, rr,
            hp_h, hp, panL, panR, exc, bank, level):
    """`n` samples into out (n, 2): the formulas before OUT_SCALE / gain.
    level[i] = max |b_i| over the block (display only)."""
    nn = ring.shape[0]
    nb = zre.shape[1]
    q = cst[en.C_Q]; qf = cst[en.C_QF]; qs = cst[en.C_QS]; strength = cst[en.C_STRENGTH]
    for i in range(nn):
        level[i] = 0.0
    for s in range(n):
        k = ints[I_K]
        left = ints[I_R_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                rr[R_CUR] = rr[R_TGT]
            else:
                rr[R_CUR] = rr[R_CUR] + rr[R_INC]
            ints[I_R_LEFT] = left
        r = rr[R_CUR]
        # ---- the N2-B network step (event_network._render, rho fixed)
        for i in range(nn):
            exc[i] = strength * ((1.0 - qf) * zf[i] - (1.0 - qs) * zs[i]) / (qs - qf)
            zf[i] = zf[i] * qf
            zs[i] = zs[i] * qs
        for i in range(nn):
            d = ring[i, k % D[i]]
            lfilt[i] = (1.0 - q) * d + q * lfilt[i]
        for i in range(nn):
            acc = 0.0
            for j in range(nn):
                acc += M[i, j] * lfilt[j]
            ring[i, k % D[i]] = math.tanh(exc[i] + rho * acc)
        # ---- the tuned banks, driven by the loss-filter values of this sample
        L = 0.0
        R = 0.0
        for i in range(nn):
            drive = lfilt[i]
            acc = 0.0
            for j in range(nb):
                re = zre[i, j]
                im = zim[i, j]
                c = cth[i, j]
                sn = sth[i, j]
                nre = r * (c * re - sn * im) + drive
                nim = r * (sn * re + c * im)
                zre[i, j] = nre
                zim[i, j] = nim
                acc += nre
            b = acc / nb
            bank[i] = b
            a = b if b >= 0.0 else -b
            if a > level[i]:
                level[i] = a
            L += panL[i] * b
            R += panR[i] * b
        mixL = L / nn
        mixR = R / nn
        # ---- DC removal per channel
        yL = hp_h * ((hp[0, H_PREV] + mixL) - hp[0, H_MIX])
        hp[0, H_PREV] = yL
        hp[0, H_MIX] = mixL
        yR = hp_h * ((hp[1, H_PREV] + mixR) - hp[1, H_MIX])
        hp[1, H_PREV] = yR
        hp[1, H_MIX] = mixR
        out[s, 0] = yL
        out[s, 1] = yR
        ints[I_K] = k + 1


class TunedEventsEngine(SoundEngine):
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
        self._bank = np.zeros(N_NODES)
        self.level = np.zeros(N_NODES)
        self.base = base_freqs()
        self.hp_h = math.exp(-2.0 * math.pi * HP_HZ / self.sr)
        self.r_ramp_n = int(round(DECAY_RAMP_MS / 1000.0 * self.sr))
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
        self.M = en.matrix()
        self.panL, self.panR = en.pans()
        self.consts = en.consts(self.sr)
        self.ints = np.zeros(2, np.int64)
        r = decay_r(self.params['decay_s'], self.sr)
        self.rr = np.array([r, r, 0.0])
        self.zre = np.zeros((N_NODES, N_BANK))
        self.zim = np.zeros((N_NODES, N_BANK))
        self.hp = np.zeros((2, 2))
        self.counts = np.zeros(N_NODES)
        self.u = np.zeros(N_NODES)
        self.ratio = np.ones(N_NODES)
        self.ffreq = gf.bank_freqs(self.base, self.ratio)
        self.cth, self.sth = trig_tables(self.ffreq, self.sr)
        self.last_e = np.zeros(N_NODES)
        self.last_a = np.zeros(N_NODES)
        self.G_prev = None

    def _kernel(self, n, out):
        _render(n, out, self.ring, self.D, self.lfilt, self.zf, self.zs, self.M, RHO,
                self.ints, self.consts, self.zre, self.zim, self.cth, self.sth, self.rr,
                self.hp_h, self.hp, self.panL, self.panR, self._exc, self._bank, self.level)

    def _set_grid(self, grid):
        self._grid = np.array(grid, np.uint8, copy=True)
        if self._K is None or self._K.shape[1:] != self._grid.shape:
            self._K = pr.weights(*self._grid.shape)
        self.counts = pr.weighted_counts(self._K, self._grid)
        self.u = gf.field_u(self.counts)
        self._retune()

    def _retune(self):
        """The eight multipliers -> 8 x 24 float32 frequency messages -> rotation
        tables (states untouched)."""
        self.ratio = field_ratio(self.u, bool(int(self.params['field_tuning'])))
        self.ffreq = gf.bank_freqs(self.base, self.ratio)
        self.cth, self.sth = trig_tables(self.ffreq, self.sr)

    def _events(self):
        """The packet of the pending field against G_prev (block boundary)."""
        if self.G_prev is None or self.G_prev.shape != self._grid.shape:
            self.G_prev = np.zeros_like(self._grid)
        e, a = en.packet(self._K, self.G_prev, self._grid)
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
        old = dict(self.params)
        super().set_params(params)
        if self._grid is None:
            return
        if int(self.params['field_tuning']) != int(old['field_tuning']):
            self._retune()
        if float(self.params['decay_s']) != float(old['decay_s']):
            tgt = decay_r(self.params['decay_s'], self.sr)
            self.rr[R_TGT] = tgt
            if self.r_ramp_n > 0:
                self.rr[R_INC] = (tgt - self.rr[R_CUR]) / self.r_ramp_n
                self.ints[I_R_LEFT] = self.r_ramp_n
            else:
                self.rr[R_CUR] = tgt
                self.rr[R_INC] = 0.0
                self.ints[I_R_LEFT] = 0

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        self._check_transpose(transpose)
        if gain_prev is not None:
            self.gain_prev = self._eff(gain_prev)  # the host overrides the glide start
        y, peak, n_clip = self.render_float(gain)
        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(pcm), peak, n_clip

    def render_float(self, gain):
        """One block (block, 2) after OUT_SCALE and the ramped bench gain, before the
        int16 clip, + pre-clip peak / clipped sample count.  The events of the block
        are applied first; the frequency tables are those of the pending field."""
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
        """One block of the formulas (after the DC filters, no OUT_SCALE, no gain) --
        verification helper; events=False renders the sample path only."""
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

    def network_state(self):
        """Copies of the N2 network states (for the equality check with ca_event_network)."""
        return dict(ring=self.ring.copy(), lfilt=self.lfilt.copy(), zf=self.zf.copy(),
                    zs=self.zs.copy(), k=int(self.ints[I_K]))

    def reset(self, gain=0.0):
        self.init(self._grid, None, gain)

    def display(self):
        return dict(events=[float(v) for v in self.last_a], e=[float(v) for v in self.last_e],
                    level=[float(v) for v in self.level], ratio=[float(v) for v in self.ratio],
                    counts=[round(float(v), 2) for v in self.counts],
                    tuning=bool(int(self.params['field_tuning'])),
                    decay_s=float(self.params['decay_s']), r=float(self.rr[R_CUR]),
                    r_target=float(self.rr[R_TGT]), ramp_left=int(self.ints[I_R_LEFT]),
                    model=self.model_version)

    # -- snapshot ---------------------------------------------------------------------
    _ARRAYS = ('ring', 'D', 'lfilt', 'zf', 'zs', 'M', 'ints', 'rr', 'zre', 'zim', 'ffreq', 'hp',
               'last_e', 'last_a', 'level', 'ratio', 'counts')

    def export_state(self):
        if self._grid is None:
            raise ValueError(f"engine {ENGINE_ID}: no field yet (init first)")
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  model_version=self.model_version, params=dict(self.params),
                  sr=int(self.sr), block=int(self._out.shape[0]),
                  gain_prev=float(self.gain_prev), out_scale=OUT_SCALE, rho=RHO,
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
                     M=en.matrix(), ints=np.zeros(2, np.int64), rr=np.zeros(3),
                     zre=np.zeros((N_NODES, N_BANK)), zim=np.zeros((N_NODES, N_BANK)),
                     ffreq=np.zeros((N_NODES, N_BANK)), hp=np.zeros((2, 2)),
                     last_e=np.zeros(N_NODES), last_a=np.zeros(N_NODES), level=np.zeros(N_NODES),
                     ratio=np.zeros(N_NODES), counts=np.zeros(N_NODES))
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
            raise ValueError(f"engine {ENGINE_ID}: delays / matrix of the state are not the N2 network")
        if arrays['ints'][I_K] < 0 or arrays['ints'][I_R_LEFT] < 0:
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
        if (not math.isfinite(gain_prev) or float(state.get('out_scale', OUT_SCALE)) != OUT_SCALE
                or float(state.get('rho', RHO)) != RHO):
            raise ValueError(f"engine {ENGINE_ID}: bad scalar in the state")
        self.params = dict(params)
        self._zero()
        for name, a in arrays.items():
            setattr(self, name, a)
        self.panL, self.panR = en.pans()
        self.consts = en.consts(self.sr)
        self._set_grid(g)                       # recomputes ratio / ffreq / tables from the field
        if not np.array_equal(self.ffreq, arrays['ffreq']) or not np.array_equal(self.ratio, arrays['ratio']):
            raise ValueError(f"engine {ENGINE_ID}: the frequencies of the state do not follow "
                             f"from its field and parameters")
        self.G_prev = np.array(gv, np.uint8, copy=True)
        self.gain_prev = gain_prev

    # -- internals ----------------------------------------------------------------------
    def _eff(self, gain):
        return float(gain) * OUT_SCALE


def overlay(params, rows, cols):
    return pr.overlay(rows, cols, OVERLAY_TEXT)


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: TunedEventsEngine(ctx, params),
                    choices=CHOICES, overlay=overlay))
