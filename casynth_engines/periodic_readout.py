"""N2 -- the shared periodic field readout (REQ memory/req-network-events-n2-2026-09-16.md,
"Общий способ чтения положения").

Both N2 sides (A: gutter_field_periodic, B: ca_event_network) read the field
through the SAME eight overlapping raised-cosine weights instead of the hard
2 x 4 regions of N1, so that region borders cannot decide the comparison.
For an H x W field, cell (r, c) and node i = 4 l + j (j = 0..3 columns,
l = 0..1 rows):

    x  = (c + 0.5) / W          y  = (r + 0.5) / H
    ax = (j + 0.5) / 4          ay = (l + 0.5) / 2
    dx = ((x - ax + 0.5) mod 1) - 0.5        dy likewise (torus distance)
    h(d, s) = (1 + cos(pi d / s)) / 2  for |d| <= s, else 0
    K_i(r, c) = h(dx, 1/4) * h(dy, 1/2)

K_i >= 0 and the eight weights of every cell sum to 1 (a partition of unity,
continuous on the torus).  The weights depend only on the field size; they
are computed once per engine and never on the sample path.
"""
import numpy as np

NODE_COLS, NODE_ROWS = 4, 2
N_NODES = NODE_COLS * NODE_ROWS
SX, SY = 1.0 / NODE_COLS, 1.0 / NODE_ROWS       # half-widths of the supports (1/4, 1/2)


def node_centres():
    """[(ax, ay)] for i = 0..7 in field-relative units (0..1)."""
    return [((j + 0.5) / NODE_COLS, (l + 0.5) / NODE_ROWS)
            for l in range(NODE_ROWS) for j in range(NODE_COLS)]


def _h(d, s):
    d = np.asarray(d, np.float64)
    return np.where(np.abs(d) <= s, (1.0 + np.cos(np.pi * d / s)) * 0.5, 0.0)


def _torus_delta(x, a):
    return ((x - a + 0.5) % 1.0) - 0.5


def weights(rows, cols):
    """float64 (8, rows, cols): K_i(r, c)."""
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("periodic readout: empty field")
    x = (np.arange(cols) + 0.5) / cols
    y = (np.arange(rows) + 0.5) / rows
    K = np.zeros((N_NODES, rows, cols))
    for i, (ax, ay) in enumerate(node_centres()):
        hx = _h(_torus_delta(x, ax), SX)
        hy = _h(_torus_delta(y, ay), SY)
        K[i] = hy[:, None] * hx[None, :]
    return K


def weighted_counts(K, field):
    """(8,) float64: sum over cells of K_i * field (field: 0/1 or bool)."""
    f = np.asarray(field)
    if f.shape != K.shape[1:]:
        raise ValueError(f"periodic readout: field {f.shape} != weights {K.shape[1:]}")
    return np.tensordot(K, f.astype(np.float64), axes=([1, 2], [0, 1]))


def centre_cells(rows, cols):
    """Node centres in CELL coordinates (x = column, y = row; half-integers allowed)."""
    return [(ax * cols - 0.5, ay * rows - 0.5) for ax, ay in node_centres()]


def overlay(rows, cols, text):
    """Bench overlay: the eight node centres (dot + node number) and the ring where the
    weight of a cell in the node's row direction has fallen to one half (SX / 2 of the
    width).  No borders: the weights overlap and wrap around the torus."""
    circles, labels = [], []
    half = 0.5 * SX * cols            # h(d, 1/4) = 1/2 at |d| = 1/8 of the width
    for i, (cx, cy) in enumerate(centre_cells(rows, cols)):
        circles.append((cx, cy, 0.6))
        circles.append((cx, cy, half))
        labels.append((cx, cy - 2.2, str(i)))
    return dict(circles=circles, labels=labels, text=text)
