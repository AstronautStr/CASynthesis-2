"""N2 side A -- the Gutter network with the periodic field readout (engine id
`gutter_field_periodic`, REQ memory/req-network-events-n2-2026-09-16.md, "А — частотный
способ").

Everything is gutter_field (N1): the eight nodes, the internal source, the
matrix delay, the routes, the states and the field -> frequency law

    u_i     = n_i / (n_i + 2)
    ratio_i = 2 ** clip(log2(scale) + depth * (2 u_i - 1), -1, 1)
    f_ij    = float32(min(19000, base_ij * ratio_i))

Only two things differ, both fixed for the N2 comparison:
  * n_i = sum(K_i * G) -- the weighted (fractional) live-cell count under the
    periodic raised-cosine weights of periodic_readout instead of the hard
    2 x 4 regions;
  * the output trim A_TRIM = 10 ** (-16.5 / 20) applied after the existing
    FIXED_SCALE = 20 and before the bench gain (REQ "Уровни сравнения").
Same controls, same snapshot layout, own model_version.
"""
import numpy as np

from . import gutter_field as gf
from . import periodic_readout as pr
from .registry import EngineSpec, register

ENGINE_ID = 'gutter_field_periodic'
LABEL = 'Gutter K'
MODEL_VERSION = 'gutter_field_periodic_n2_v1'
A_TRIM_DB = -16.5
A_TRIM = 10.0 ** (A_TRIM_DB / 20.0)
OVERLAY_TEXT = "8 nodes: periodic weights (no borders) -- cells near a node tune its resonators"


def periodic_config():
    cfg = gf.n1_config()
    cfg['model_version'] = MODEL_VERSION
    return cfg


class GutterFieldPeriodicEngine(gf.GutterFieldEngine):
    def __init__(self, ctx, params):
        self._K = None
        super().__init__(ctx, params, config=periodic_config(), engine_id=ENGINE_ID)

    def _set_grid(self, grid):
        self._grid = np.array(grid, np.uint8, copy=True)
        if self._K is None or self._K.shape[1:] != self._grid.shape:
            self._K = pr.weights(*self._grid.shape)
        self.counts = pr.weighted_counts(self._K, self._grid)
        self.u_field = gf.field_u(self.counts)

    def _eff(self, gain):
        return float(gain) * gf.FIXED_SCALE * A_TRIM

    def display(self):
        d = super().display()
        d['counts'] = [round(float(v), 2) for v in self.counts]
        d['readout'] = 'periodic'
        return d


def overlay(params, rows, cols):
    return pr.overlay(rows, cols, OVERLAY_TEXT)


register(EngineSpec(ENGINE_ID, LABEL, gf.PARAMS,
                    lambda ctx, params: GutterFieldPeriodicEngine(ctx, params),
                    overlay=overlay))
