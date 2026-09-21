"""S -- scanned surface (engine id `scan_surface`, REQ 2026-09-14 section 3).

The whole 0/1 field is ONE sound object: a surface z[y, x] is built from the
mask, a fixed periodic path P(p) (p = 0..1 over one note period) reads it
bilinearly in periodic coordinates, and the reading v(p) = z(P(p)) -- mean
removed, band-limited -- is the periodic wave played at f0 with a continuous
phase.  Position and orientation on the field matter by design.

Surfaces (`surface`):
  0 Smooth   : z = 2 * Gaussian(g, sigma = width) - 1   (normalised kernel,
               periodic borders, width in cells)
  1 Distance : z = tanh((d_dead - d_live) / width), d_* = Euclidean distance
               from the cell centre to the nearest dead / live cell centre on
               the torus (positive inside live areas; all-dead -> -1,
               all-live -> +1, never inf / NaN)
Paths (`path`), cell-centre coordinates x = 0..W-1 (column), y = 0..H-1 (row),
cx = (W-1)/2, cy = (H-1)/2, rx = radius_x*(W-1)/2, ry = radius_y*(H-1)/2,
phi = 2*pi*p, phi uniform in p:
  0 Ellipse   : x = cx + rx*cos(phi),   y = cy + ry*sin(phi)
  1 Lissajous : x = cx + rx*cos(2 phi), y = cy + ry*sin(3 phi)
  2 Raster    : closed snake through the centres of ALL cells (even rows left
                to right, odd rows right to left, rows top-down), linear
                between centres at constant speed along the path length; the
                last centre returns to (0, 0) across the torus seam by the
                shortest segment (even H: one vertical step; odd H: a diagonal
                step).  radius_x / radius_y do not apply.
Rendering (section 3.3): the period is sampled at table_size() uniform phase
points (8192 for 32x32, more for larger fields), the mean is removed and the
COMPLEX coefficients of harmonics 1..K, K = floor(0.45*sr/f0), are kept (K < 1
-> silence).  Playback is additive from those coefficients (no table
interpolation, nothing above K exists -> no aliasing by construction); the
phase accumulator never resets.  Any change of surface / path / radii / field
makes a new target coefficient vector reached by a linear 20 ms (sample-time)
crossfade of the coefficient vectors; an edit during a transition starts from
the mix actually reached.  No per-table energy normalisation: an empty or
constant surface is silence after DC removal.  `gain` (already
MASTER_GAIN*vol*level) times 10^(trim_db/20) is applied once, ramped across
the block like the legacy engine; peak / n_clip are measured before the int16
clip.  Fixed raw headroom: the reading |v| <= 2 (z in [-1, 1] minus its
mean) is scaled by RAW_SCALE = 8 (+18 dB) before the master gain, so the
worst case is 16 * MASTER_GAIN = 0.64 of full scale at trim 0 -- no overflow
(only the +trim range can clip, and that is counted).  The constant level
correction found for the prepared comparisons (RMS matched to the Laplace
side on the F3 Pulsar scene, 2026-09-14) is the registry default trim_db.
"""
import math

import numpy as np
from scipy import ndimage

from casynth_config import _RAMP
from .engine_api import SoundEngine
from .registry import EngineSpec, register

ENGINE_ID = 'scan_surface'
LABEL = 'Scan'
SURFACES = ('Smooth', 'Distance')
PATHS = ('Ellipse', 'Lissajous', 'Raster')
PARAMS = [('surface', 'Surface', 0, 1, True, 0),
          ('width', 'Width (cells)', 0.5, 3.0, False, 1.5),
          ('path', 'Path', 0, 2, True, 2),
          ('radius_x', 'Radius X', 0.1, 0.95, False, 0.8),
          ('radius_y', 'Radius Y', 0.1, 0.95, False, 0.8),
          ('trim_db', 'Level (dB)', -24.0, 24.0, False, 6.0)]
CHOICES = {'surface': SURFACES, 'path': PATHS}
BAND = 0.45                   # top of the kept band, fraction of sr
XFADE_S = 0.020               # coefficient crossfade (sample time)
RAW_SCALE = 8.0               # fixed raw headroom scale (see the module docstring)
TABLE_MIN = 8192              # phase samples per period (32x32 and smaller)
TABLE_PER_CELL = 8            # larger fields: >= 8 samples per raster cell
OVERLAY_POINTS = 720          # analytic paths: polyline density of the overlay
STATE_VERSION = 1


# -- surfaces -----------------------------------------------------------------
def smooth_surface(grid, width):
    g = np.asarray(grid, np.float64)
    return 2.0 * ndimage.gaussian_filter(g, sigma=float(width), mode='wrap') - 1.0


def torus_distance_to(mask):
    """Euclidean distance from every cell centre to the nearest centre of a
    True cell, the shortest way round the torus (inf when no cell is True).
    The 3x3 tiling holds every torus-nearest copy (|dx| <= W/2, |dy| <= H/2)."""
    m = np.asarray(mask, bool)
    h, w = m.shape
    if not m.any():
        return np.full((h, w), np.inf)
    d = ndimage.distance_transform_edt(~np.tile(m, (3, 3)))
    return d[h:2 * h, w:2 * w]


def distance_surface(grid, width):
    live = np.asarray(grid) > 0
    h, w = live.shape
    if live.all():
        return np.ones((h, w))
    if not live.any():
        return -np.ones((h, w))
    return np.tanh((torus_distance_to(~live) - torus_distance_to(live)) / float(width))


def surface(grid, mode, width):
    return smooth_surface(grid, width) if int(mode) == 0 else distance_surface(grid, width)


# -- paths --------------------------------------------------------------------
def _seam_delta(d, n):
    """Shortest signed displacement congruent to d modulo n."""
    if n <= 1:
        return 0.0
    return float((d + n / 2.0) % n - n / 2.0)


def raster_nodes(rows, cols):
    """(N+1, 2) unwrapped (x, y) vertices of the closed snake: N = rows*cols
    cell centres, then the closing point = (0, 0) + the shortest torus
    displacement from the last centre (pts[-1] - pts[-2] is the seam step)."""
    pts = []
    for r in range(rows):
        xs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        pts.extend((float(c), float(r)) for c in xs)
    pts = np.array(pts, np.float64).reshape(-1, 2)
    last = pts[-1]
    close = last + (_seam_delta(-last[0], cols), _seam_delta(-last[1], rows))
    return np.vstack([pts, close])


def _polyline_cum(nodes):
    seg = np.diff(nodes, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    return length, np.concatenate([[0.0], np.cumsum(length)])


def path_points(path, rows, cols, radius_x, radius_y, p, nodes=None):
    """(x, y) arrays (unwrapped cell coordinates) of the path at phases p
    (cycles; taken modulo 1).  `nodes` = cached raster_nodes(rows, cols)."""
    p = np.asarray(p, np.float64) % 1.0
    path = int(path)
    if path == 2:
        nodes = raster_nodes(rows, cols) if nodes is None else nodes
        length, cum = _polyline_cum(nodes)
        total = cum[-1]
        if total <= 0.0:
            return np.full(p.shape, nodes[0, 0]), np.full(p.shape, nodes[0, 1])
        s = p * total
        i = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(length) - 1)
        t = (s - cum[i]) / np.where(length[i] > 0, length[i], 1.0)
        pts = nodes[i] + (nodes[i + 1] - nodes[i]) * t[:, None]
        return pts[:, 0], pts[:, 1]
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    rx, ry = float(radius_x) * cx, float(radius_y) * cy
    phi = 2.0 * np.pi * p
    if path == 0:
        return cx + rx * np.cos(phi), cy + ry * np.sin(phi)
    return cx + rx * np.cos(2.0 * phi), cy + ry * np.sin(3.0 * phi)


def read_surface(z, x, y):
    """Bilinear reading of z at (x, y) with periodic (torus) coordinates."""
    h, w = z.shape
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    c0 = x0.astype(np.int64) % w
    r0 = y0.astype(np.int64) % h
    c1, r1 = (c0 + 1) % w, (r0 + 1) % h
    return (z[r0, c0] * (1.0 - fx) * (1.0 - fy) + z[r0, c1] * fx * (1.0 - fy)
            + z[r1, c0] * (1.0 - fx) * fy + z[r1, c1] * fx * fy)


def table_size(rows, cols):
    n = TABLE_PER_CELL * rows * cols
    return max(TABLE_MIN, 1 << max(0, (int(n) - 1).bit_length()))


def read_period(z, params, n, nodes=None):
    """v(p) at n uniform phases: the raw (unfiltered) reading of one period."""
    h, w = z.shape
    p = np.arange(n, dtype=np.float64) / float(n)
    x, y = path_points(params['path'], h, w, params['radius_x'], params['radius_y'], p,
                       nodes=nodes)
    return read_surface(z, x, y)


# -- spectrum -----------------------------------------------------------------
def harmonics(sr, f0):
    """K = floor(BAND*sr/f0) (0 -> silence)."""
    return max(0, int(math.floor(BAND * float(sr) / float(f0)))) if f0 > 0 else 0


def band_limit(v, K):
    """Mean removed, complex coefficients of harmonics 1..K (a[k-1] such that
    v_bl(p) = Re sum_k a_k exp(2 pi i k p)); harmonics above K and DC = 0."""
    v = np.asarray(v, np.float64)
    a = np.zeros(int(K), np.complex128)
    if K < 1 or len(v) < 2:
        return a
    v = v - v.mean()
    if np.abs(v).max() < 1e-12:            # a constant reading is silence, exactly
        return a
    c = np.fft.rfft(v) / len(v)
    kmax = min(int(K), len(c) - 1)
    a[:kmax] = 2.0 * c[1:kmax + 1]
    return a


def evaluate(a, p):
    """Exact additive evaluation of the band-limited wave at phases p (cycles).
    `a` may be one vector (K,) or several (m, K) sharing the same phases."""
    a = np.asarray(a, np.complex128)
    p = np.asarray(p, np.float64) % 1.0
    K = a.shape[-1]
    if K == 0 or len(p) == 0:
        return np.zeros(a.shape[:-1] + p.shape)
    base = np.exp(2j * np.pi * p)
    E = np.cumprod(np.broadcast_to(base, (K, len(p))), axis=0)   # E[k-1, n] = e^{2 pi i k p_n}
    return (a @ E).real


def target_coeffs(grid, params, sr, f0, nodes=None):
    z = surface(grid, params['surface'], params['width'])
    v = read_period(z, params, table_size(*z.shape), nodes=nodes)
    return band_limit(v, harmonics(sr, f0))


# -- overlay (drawn by the bench with the SAME geometry as the reading) -------
def overlay(params, rows, cols):
    """Closed polyline of the reading path in unwrapped cell coordinates
    (consecutive points may cross the torus seam: the bench wraps them), the
    start point and a point a little further along (direction arrow), and the
    label of the path."""
    path = int(params['path'])
    if path == 2:
        pts = raster_nodes(rows, cols)
        text = 'Raster: full field'
    else:
        p = np.arange(OVERLAY_POINTS + 1) / float(OVERLAY_POINTS)
        x, y = path_points(path, rows, cols, params['radius_x'], params['radius_y'], p)
        pts = np.column_stack([x, y])
        text = PATHS[path]
    x0, y0 = path_points(path, rows, cols, params['radius_x'], params['radius_y'],
                         np.array([0.0, 0.004]))
    return dict(polyline=[(float(a), float(b)) for a, b in pts], closed=True,
                start=(float(x0[0]), float(y0[0])), ahead=(float(x0[1]), float(y0[1])),
                text=text)


def inactive(params):
    """Parameters that do not act for the current settings -> shown text."""
    if int(params['path']) == 2:
        return {'radius_x': 'Full field', 'radius_y': 'Full field'}
    return {}


# -- engine -------------------------------------------------------------------
class ScanSurfaceEngine(SoundEngine):
    STATE_VERSION = STATE_VERSION

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        self.K = harmonics(ctx.sr, ctx.f0)
        self.xfade = max(1, int(round(XFADE_S * ctx.sr)))
        self._grid = None
        self._nodes = None
        self._clear_audio(0.0)

    # -- SoundEngine --------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._clear_audio(self._eff(gain))
        self._grid = np.array(grid, np.uint8, copy=True)
        self._nodes = raster_nodes(*self._grid.shape)
        a = self._target()
        self.a_from = a
        self.a_to = a.copy()
        self.xpos = self.xfade                      # no transition in flight

    def update_field(self, grid, exc):
        self._grid = np.array(grid, np.uint8, copy=True)
        if self._nodes is None or len(self._nodes) != self._grid.size + 1:
            self._nodes = raster_nodes(*self._grid.shape)
        self._retarget()

    def set_params(self, params):
        super().set_params(params)
        if self._grid is not None:
            self._retarget()

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        self._check_transpose(transpose)
        if gain_prev is not None:
            self.gain_prev = self._eff(gain_prev)  # the host overrides the glide start
        y, peak, n_clip = self.render_float(gain)
        out = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(np.repeat(out[:, None], self.ctx.channels, axis=1)), \
            peak, n_clip

    def render_float(self, gain):
        """One block as float (mono, before the int16 clip) + peak / n_clip."""
        n = self.ctx.block
        p = (self.phase + np.arange(n) * (self.ctx.f0 / self.ctx.sr)) % 1.0
        self.phase = (self.phase + n * (self.ctx.f0 / self.ctx.sr)) % 1.0
        if self.xpos >= self.xfade:
            y = evaluate(self.a_from, p)
        else:
            w = np.minimum((self.xpos + 1 + np.arange(n)) / float(self.xfade), 1.0)
            both = evaluate(np.stack([self.a_from, self.a_to]), p)
            y = both[0] * (1.0 - w) + both[1] * w
            self.xpos += n
            if self.xpos >= self.xfade:
                self.a_from = self.a_to.copy()
        eff = self._eff(gain)
        y = y * (self.gain_prev + (eff - self.gain_prev) * _RAMP)
        self.gain_prev = eff
        peak = float(np.abs(y).max()) if n else 0.0
        n_clip = int(np.count_nonzero(np.abs(y) > 1.0))
        return y, peak, n_clip

    def reset(self, gain=0.0):
        self.init(self._grid, None, gain)

    # -- snapshot -----------------------------------------------------------------
    def export_state(self):
        return dict(version=self.STATE_VERSION, engine_id=ENGINE_ID, params=dict(self.params),
                    K=int(self.K), phase=float(self.phase), xpos=int(self.xpos),
                    gain_prev=float(self.gain_prev),
                    a_from=self.a_from.copy(), a_to=self.a_to.copy())

    def restore_state(self, grid, exc, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != ENGINE_ID:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {ENGINE_ID!r}")
        if int(state.get('K', -1)) != self.K:
            raise ValueError(f"engine {ENGINE_ID}: {state.get('K')} harmonics in the state, "
                             f"{self.K} for this context")
        arrays = {}
        for name in ('a_from', 'a_to'):
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != (self.K,) or a.dtype.kind != 'c':
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {(self.K,)}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            arrays[name] = np.array(a, np.complex128, copy=True)
        phase, xpos, gp = float(state['phase']), int(state['xpos']), float(state['gain_prev'])
        if not (0.0 <= phase < 1.0) or xpos < 0 or not math.isfinite(gp):
            raise ValueError(f"engine {ENGINE_ID}: bad phase / transition / gain in the state")
        self.params = dict(state['params'])
        self._grid = np.array(grid, np.uint8, copy=True)
        self._nodes = raster_nodes(*self._grid.shape)
        self.phase, self.xpos, self.gain_prev = phase, min(xpos, self.xfade), gp
        self.a_from, self.a_to = arrays['a_from'], arrays['a_to']

    # -- internals ----------------------------------------------------------------
    def _clear_audio(self, gain):
        self.phase = 0.0
        self.a_from = np.zeros(self.K, np.complex128)
        self.a_to = np.zeros(self.K, np.complex128)
        self.xpos = self.xfade
        self.gain_prev = float(gain)

    def _eff(self, gain):
        return float(gain) * RAW_SCALE * 10.0 ** (float(self.params['trim_db']) / 20.0)

    def _target(self):
        return target_coeffs(self._grid, self.params, self.ctx.sr, self.ctx.f0, nodes=self._nodes)

    def _retarget(self):
        """New target wave; a transition in flight restarts from the mix
        actually reached (the last output sample's mix)."""
        a = self._target()
        if np.array_equal(a, self.a_to):
            return                                   # same target: nothing to restart
        if self.xpos < self.xfade:
            w = min(self.xpos / float(self.xfade), 1.0)
            self.a_from = self.a_from * (1.0 - w) + self.a_to * w
        else:
            self.a_from = self.a_to
        self.a_to = a
        self.xpos = 0


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: ScanSurfaceEngine(ctx, params),
                    choices=CHOICES, inactive=inactive, overlay=overlay))
