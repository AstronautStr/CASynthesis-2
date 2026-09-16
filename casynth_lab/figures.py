"""N4 -- connected figures of the field (REQ memory/req-object-resonators-n4-2026-09-16.md,
"Геометрия, спектр и возбуждение"): 8-connected components ACROSS the torus seam,
the periodic centre, the radius, the two detector masks, the Laplacian spectrum
of a figure and the deterministic matching of figures between two fields.

Everything here is pure geometry / linear algebra on the field; no audio state.
The engine (object_resonators.py) owns the banks and calls these functions at the
block boundary.

Components: ndimage.label with the 3 x 3 structure, then the labels of cells that
touch across the outer edges (last row <-> first row, last column <-> first
column, diagonals included) are united.  Cells of a component are (row, col)
int64 pairs sorted lexicographically; the components themselves are sorted by
their first cell.

Periodic centre (per axis, n = rows or cols): the m that minimises
    F(m) = sum_i d(x_i, m)^2        d = ((x - m + n/2) mod n) - n/2  (shortest way)
F is piecewise quadratic with cusps (local maxima) at x_i + n/2, so its minimum is
the plain mean of one of the n "cut" unwrappings x' = s + ((x - s) mod n),
s = 0..n-1; every cut is tried and F evaluated exactly.  Several minima within
1e-9: the one nearest (shortest way) to the previous centre, for a new figure the
smallest coordinate.  For a compact figure this is exactly its unwrapped mean.

Radius: R = max over the cells of the torus distance between the cell and the
centre (cell centres; a single cell has R = 0).  Disk mask: every cell whose
torus distance to the centre is <= R (+ 1e-9, the boundary included).  Own mask:
exactly the figure's cells.

Spectrum: the full unweighted 8-connectivity graph of the component (adjacency
modulo the torus), L = D - A, eigvalsh; the zero mode is dropped (a connected
graph has exactly one), the remaining eigenvalues (with multiplicities) are the
first `max_modes` in ascending order, returned as sqrt(lambda).  A component of N
cells has at most N - 1 modes; a single cell has none.  The matrix is built from
the CANONICAL placement of the component (circular offsets from its first cell,
shifted to (0, 0)), so the same shape anywhere on the torus gives the same matrix
and the same eigenvalues bit for bit; the spectrum is cached by that key and the
cache never changes a result.

Matching (fields prev -> cur): the overlap graph between the previous figures and
the current components (shared cells).  A connected piece of that graph with
exactly one previous figure and one component continues that figure; every other
piece (split, merge, chains) sends all its previous figures to tails and gives
every component a new identity -- the rule of the first model.  Previous figures
and components with NO overlap are then paired by the torus distance of their
centres when it is <= MOVE_MAX (1.5 cells), nearest first (ties: lower previous
index, then lower component index).  Everything left is a tail / a new figure.
"""
import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), np.uint8)
MOVE_MAX = 1.5                 # zero-overlap continuation: centre displacement <= this
TIE_EPS = 1e-9
ZERO_EPS = 1e-9                # eigenvalues below this are the zero mode (safety)
MASK_EPS = 1e-9                # disk boundary inclusion


# -- components ----------------------------------------------------------------------
def components(grid):
    """[int64 (N_k, 2)] 8-connected components of the 0/1 field on the torus."""
    g = np.asarray(grid, np.uint8)
    rows, cols = g.shape
    lab, n = ndimage.label(g, structure=STRUCT8)
    if n == 0:
        return []
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    if rows > 1:
        top, bot = lab[0], lab[rows - 1]
        for c in range(cols):
            if bot[c]:
                for dc in (-1, 0, 1):
                    t = top[(c + dc) % cols]
                    if t:
                        union(bot[c], t)
    if cols > 1:
        left, right = lab[:, 0], lab[:, cols - 1]
        for r in range(rows):
            if right[r]:
                for dr in (-1, 0, 1):
                    t = left[(r + dr) % rows]
                    if t:
                        union(right[r], t)
    roots = np.array([find(i) for i in range(n + 1)])
    root_of_cell = roots[lab]
    out = []
    for root in np.unique(roots[1:]):
        rr, cc = np.nonzero(root_of_cell == root)
        cells = np.stack([rr, cc], axis=1).astype(np.int64)
        out.append(cells[np.lexsort((cells[:, 1], cells[:, 0]))])
    out.sort(key=lambda a: (int(a[0, 0]), int(a[0, 1])))
    return out


# -- geometry ------------------------------------------------------------------------
def torus_delta(x, a, n):
    """Shortest signed way from a to x on a circle of length n (in [-n/2, n/2))."""
    return np.mod(np.asarray(x, np.float64) - a + n / 2.0, n) - n / 2.0


def periodic_mean(x, n, prev=None):
    """The periodic centre of integer coordinates x on a circle of length n
    (docstring of the module); `prev` = the previous centre for the tie rule."""
    x = np.asarray(x, np.float64)
    n = int(n)
    if x.size == 0:
        raise ValueError("periodic_mean: no cells")
    s = np.arange(n, dtype=np.float64)[:, None]
    xu = s + np.mod(x[None, :] - s, n)
    mu = xu.mean(axis=1)
    d = np.mod(x[None, :] - mu[:, None] + n / 2.0, n) - n / 2.0
    F = (d * d).sum(axis=1)
    best = float(F.min())
    cand = np.nonzero(F <= best + TIE_EPS * max(1.0, best))[0]
    mus = np.mod(mu[cand], n)
    if prev is not None and cand.size > 1:
        dist = np.abs(torus_delta(mus, float(prev), n))
        order = np.lexsort((mus, dist))
        return float(mus[order[0]])
    return float(mus.min())


def centre_of(cells, rows, cols, prev=None):
    """(cy, cx) of a component (row / col units, cell (r, c) sits at (r, c))."""
    cells = np.asarray(cells, np.int64)
    py = None if prev is None else float(prev[0])
    px = None if prev is None else float(prev[1])
    return (periodic_mean(cells[:, 0], rows, py), periodic_mean(cells[:, 1], cols, px))


def radius_of(cells, centre, rows, cols):
    cells = np.asarray(cells, np.int64)
    dr = torus_delta(cells[:, 0], centre[0], rows)
    dc = torus_delta(cells[:, 1], centre[1], cols)
    return float(np.sqrt(dr * dr + dc * dc).max())


def disk_mask(centre, radius, rows, cols):
    """bool (rows, cols): torus distance from the cell centre to `centre` <= radius."""
    dr = torus_delta(np.arange(rows), centre[0], rows)
    dc = torus_delta(np.arange(cols), centre[1], cols)
    dist = np.sqrt(dr[:, None] ** 2 + dc[None, :] ** 2)
    return dist <= float(radius) + MASK_EPS


def own_mask(cells, rows, cols):
    m = np.zeros((rows, cols), bool)
    cells = np.asarray(cells, np.int64)
    m[cells[:, 0], cells[:, 1]] = True
    return m


# -- spectrum ------------------------------------------------------------------------
def canonical_cells(cells, rows, cols):
    """The component translated to a canonical place: the circular offsets of its
    cells from its first cell, shifted to start at (0, 0), sorted.  The same shape
    at any position (across the seam too) gives the SAME array, so the Laplacian
    built from it -- and eigvalsh of it -- is identical bit for bit."""
    cells = np.asarray(cells, np.int64)
    dr = np.mod(cells[:, 0] - cells[0, 0] + rows // 2, rows) - rows // 2
    dc = np.mod(cells[:, 1] - cells[0, 1] + cols // 2, cols) - cols // 2
    dr = dr - dr.min()
    dc = dc - dc.min()
    rel = np.stack([dr, dc], axis=1)
    return np.ascontiguousarray(rel[np.lexsort((rel[:, 1], rel[:, 0]))], dtype=np.int64)


def shape_key(cells, rows, cols):
    """Translation- and seam-invariant key of a component (canonical_cells bytes)."""
    return (int(rows), int(cols), canonical_cells(cells, rows, cols).tobytes())


def laplacian_matrix(cells, rows, cols):
    """float64 (N, N): L = D - A of the unweighted 8-connectivity graph of the
    cells, neighbours taken modulo the torus (each neighbour counted once)."""
    cells = np.asarray(cells, np.int64)
    n = len(cells)
    index = np.full((rows, cols), -1, np.int64)
    index[cells[:, 0], cells[:, 1]] = np.arange(n)
    L = np.zeros((n, n))
    seen = np.zeros((n, n), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            j = index[np.mod(cells[:, 0] + dr, rows), np.mod(cells[:, 1] + dc, cols)]
            i = np.nonzero(j >= 0)[0]
            j = j[i]
            keep = (j != i) & ~seen[i, j]
            i, j = i[keep], j[keep]
            seen[i, j] = True
            L[i, j] -= 1.0
            np.add.at(L, (i, i), 1.0)
    return L


def spectrum(cells, rows, cols, max_modes=24):
    """float64 (m,): sqrt of the first `max_modes` positive Laplacian eigenvalues
    of the component (ascending, multiplicities kept; empty for a single cell)."""
    cells = np.asarray(cells, np.int64)
    if len(cells) < 2:
        return np.zeros(0)
    lam = np.linalg.eigvalsh(laplacian_matrix(canonical_cells(cells, rows, cols), rows, cols))
    lam = lam[1:]                            # the single zero mode of a connected graph
    lam = lam[lam > ZERO_EPS][:int(max_modes)]
    return np.sqrt(lam)


class SpectrumCache:
    """spectrum() memoised by shape_key (bounded, FIFO)."""

    def __init__(self, max_modes=24, capacity=4096):
        self.max_modes = int(max_modes)
        self.capacity = int(capacity)
        self._d = {}
        self.hits = 0
        self.misses = 0

    def get(self, cells, rows, cols):
        key = shape_key(cells, rows, cols)
        v = self._d.get(key)
        if v is None:
            v = spectrum(cells, rows, cols, self.max_modes)
            if len(self._d) >= self.capacity:
                self._d.pop(next(iter(self._d)))
            self._d[key] = v
            self.misses += 1
        else:
            self.hits += 1
        return v.copy()


# -- matching --------------------------------------------------------------------------
def overlap_matrix(prev_cells, comps, rows, cols):
    """int64 (P, C): shared cells between previous figure p and component c."""
    P, C = len(prev_cells), len(comps)
    ov = np.zeros((P, C), np.int64)
    if P == 0 or C == 0:
        return ov
    lab = np.full((rows, cols), -1, np.int64)
    for p, cells in enumerate(prev_cells):
        cells = np.asarray(cells, np.int64)
        lab[cells[:, 0], cells[:, 1]] = p
    for c, cells in enumerate(comps):
        hit = lab[cells[:, 0], cells[:, 1]]
        hit = hit[hit >= 0]
        if hit.size:
            np.add.at(ov[:, c], hit, 1)
    return ov


def match(prev_cells, prev_centres, comps, rows, cols):
    """(continued {c: p}, tails [p], new [c]) for previous figures (cells,
    centres) -> current components (module docstring)."""
    P, C = len(prev_cells), len(comps)
    ov = overlap_matrix(prev_cells, comps, rows, cols)
    rel = ov > 0
    continued, tails, new = {}, [], []
    # connected pieces of the bipartite overlap graph
    seen_p = np.zeros(P, bool)
    seen_c = np.zeros(C, bool)
    for p0 in range(P):
        if seen_p[p0] or not rel[p0].any():
            continue
        ps, cs = {p0}, set()
        stack = [('p', p0)]
        seen_p[p0] = True
        while stack:
            kind, i = stack.pop()
            if kind == 'p':
                for c in np.nonzero(rel[i])[0]:
                    c = int(c)
                    if not seen_c[c]:
                        seen_c[c] = True
                        cs.add(c)
                        stack.append(('c', c))
            else:
                for p in np.nonzero(rel[:, i])[0]:
                    p = int(p)
                    if not seen_p[p]:
                        seen_p[p] = True
                        ps.add(p)
                        stack.append(('p', p))
        if len(ps) == 1 and len(cs) == 1:
            continued[next(iter(cs))] = next(iter(ps))
        else:
            tails.extend(sorted(ps))
            new.extend(sorted(cs))
    # zero-overlap displacement pairing among the isolated ones
    free_p = [p for p in range(P) if not rel[p].any()]
    free_c = [c for c in range(C) if not rel[:, c].any()]
    if free_p and free_c:
        cand = []
        for p in free_p:
            for c in free_c:
                cy, cx = centre_of(comps[c], rows, cols, prev_centres[p])
                dy = torus_delta(cy, prev_centres[p][0], rows)
                dx = torus_delta(cx, prev_centres[p][1], cols)
                dist = float(np.hypot(dy, dx))
                if dist <= MOVE_MAX:
                    cand.append((dist, p, c))
        cand.sort()
        used_p, used_c = set(), set()
        for _dist, p, c in cand:
            if p in used_p or c in used_c:
                continue
            used_p.add(p)
            used_c.add(c)
            continued[c] = p
        tails.extend(p for p in free_p if p not in used_p)
        new.extend(c for c in free_c if c not in used_c)
    else:
        tails.extend(free_p)
        new.extend(free_c)
    return continued, sorted(tails), sorted(new)
