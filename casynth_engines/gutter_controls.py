"""Field-controlled alternatives for the revised N1 experiments.

The verified Gutter sample kernel, eight regions and delay (2064 samples) are
shared with gutter_field. Both alternatives retain its field -> resonator
frequency mapping and add a second field channel:
  damping: raw_i = 138 + depth * (160 + 80*u_i - 138), then (raw_i/256)**2;
  links:   G_ij = G0_ij * (1 + depth * (0.15 + 2.85*u_i - 1)).
Links are directed: row i is source i, column j is destination j. Matrix
messages ramp over 50 ms, damping over 15 ms. Existing filter, node and delay
state survives every field update. These are explicit research mappings,
not claims about the original Max patch's field control.
"""
import numpy as np

from . import gutter_field as gf
from .registry import EngineSpec, register

DAMP_ID = 'gutter_ca_damping'
LINKS_ID = 'gutter_ca_links'


class GutterControlEngine(gf.GutterFieldEngine):
    def __init__(self, ctx, params, mode):
        if mode not in ('damping', 'links'):
            raise ValueError(mode)
        self.mode = mode
        cfg = gf.n1_config()
        cfg['model_version'] = 'gutter_ca_' + mode + '_v1'
        cfg['ramp_ms']['matrix'] = 50.0
        eid = DAMP_ID if mode == 'damping' else LINKS_ID
        super().__init__(ctx, params, config=cfg, engine_id=eid)

    def _apply_freqs(self):
        super()._apply_freqs()
        depth = float(self.params['depth'])
        if self.mode == 'damping':
            raw = 138.0 + depth * (160.0 + 80.0 * self.u_held - 138.0)
            self._ramp_set(gf.R_DAMP, [gf.map_damp(v) for v in raw], self._ramp_n['damp'])
        else:
            weights = 1.0 + depth * (0.15 + 2.85 * self.u_held - 1.0)
            target = np.asarray(self.cfg['matrix']) * weights[:, None]
            self.G_tgt[:] = target
            samples = self._ramp_n['matrix']
            self.G_inc[:] = (target - self.G) / samples
            self.ints[gf.I_GLEFT] = samples

    def display(self):
        d = super().display()
        d['channel'] = self.mode
        return d


def _overlay(mode, params, rows, cols):
    d = gf.overlay(params, rows, cols)
    d['text'] = ('regions 0..7: cells tune resonators + node damping' if mode == 'damping'
                 else 'regions 0..7: cells tune resonators + outgoing delayed links')
    return d


register(EngineSpec(DAMP_ID, 'G+Damp', gf.PARAMS,
                    lambda ctx, params: GutterControlEngine(ctx, params, 'damping'),
                    overlay=lambda p, r, c: _overlay('damping', p, r, c)))
register(EngineSpec(LINKS_ID, 'G+Links', gf.PARAMS,
                    lambda ctx, params: GutterControlEngine(ctx, params, 'links'),
                    overlay=lambda p, r, c: _overlay('links', p, r, c)))
