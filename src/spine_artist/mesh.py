"""Alpha to Spine mesh: hull polygons that contain every drawn pixel, and their triangles.

A layer's mask (alpha above a threshold) is max-pooled onto a cell grid, grown by the padding plus the
simplification tolerance, traced into one outer polygon per connected component and simplified with that
tolerance. Simplification moves the outline inwards by at most the tolerance, so the result still contains
the mask grown by the padding. Holes are kept inside the hull (they are transparent texels, not geometry).

A band (y0, y1) keeps only the rows between two cut lines on cell boundaries: the outline runs exactly along a
cut, and vertices on it are pinned through simplification, so the meshes of neighbouring bands share the cut
line with no overlap and no gap (pieces.py).

Coordinates are canvas pixels, x right and y down.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MeshParams:
    threshold: int = 0        # alpha above this counts as drawn (0..255)
    padding: float = 2.       # px of transparent margin kept around the drawing
    tolerance: float = 3.     # px the outline may deviate when simplified
    cell: int = 2             # px per grid cell when tracing
    min_area: float = 4.      # components smaller than this (px²) are absorbed by padding, not dropped

    def as_dict(self) -> dict:
        return {'threshold': self.threshold, 'padding': self.padding, 'tolerance': self.tolerance,
                'cell': self.cell, 'min_area': self.min_area}


@dataclass
class Mesh:
    polygons: list[np.ndarray]                 # each (n, 2) float, counter-clockwise in y-down canvas space
    vertices: np.ndarray                       # (n, 2) all polygon vertices, concatenated
    triangles: np.ndarray                      # (m, 3) vertex indices
    bbox: tuple[int, int, int, int]            # x0, y0, x1, y1 (exclusive), integer canvas pixels
    stats: dict = field(default_factory=dict)


class MeshError(ValueError):
    pass


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a disc of the given radius (in cells)."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + radius:
                continue
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            out[yd, xd] |= mask[ys, xs]
    return out


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels from row runs and union-find (numpy plus a small Python loop over runs)."""
    h, w = mask.shape
    parent: list[int] = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    runs = []                                   # (row, x0, x1, run id)
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        cur = []
        j = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            while j < len(prev) and prev[j][1] <= x0:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < x1:
                ra, rb = find(rid), find(prev[k][2])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                k += 1
            cur.append((x0, x1, rid))
            runs.append((y, x0, x1, rid))
        prev = cur
    labels = np.zeros((h, w), dtype=np.int32)
    roots: dict[int, int] = {}
    for y, x0, x1, rid in runs:
        r = find(rid)
        labels[y, x0:x1] = roots.setdefault(r, len(roots) + 1)
    return labels, len(roots)


def _trace(mask: np.ndarray) -> list[tuple[int, int]]:
    """Outer boundary of a 4-connected-filled single component, as grid-corner points (x, y), clockwise in y-down.

    Walks cell edges with the drawn side on the right; pinch points are resolved by preferring right turns,
    which keeps the polygon simple.
    """
    h, w = mask.shape
    m = np.zeros((h + 2, w + 2), dtype=bool)
    m[1:-1, 1:-1] = mask
    ys, xs = np.nonzero(m)
    i = np.lexsort((xs, ys))[0]
    sy, sx = int(ys[i]), int(xs[i])          # top-most, then left-most cell: its top edge is on the boundary
    def cell(x, y):
        return m[y, x]
    # Corner coordinates: corner (x, y) is the top-left corner of cell (x, y). Direction vectors in y-down.
    start = (sx, sy)
    pos, d = start, (1, 0)                   # walk right along the top edge of the start cell
    pts = [pos]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for _ in range(4 * (h + 2) * (w + 2)):
        x, y = pos
        k = dirs.index(d)
        # Cells around the corner ahead: for direction d, the drawn cell must be on the right-hand side.
        def right_cell(px, py, dd):
            # cell to the right of the edge from (px, py) in direction dd
            if dd == (1, 0): return cell(px, py)
            if dd == (0, 1): return cell(px - 1, py)
            if dd == (-1, 0): return cell(px - 1, py - 1)
            return cell(px, py - 1)
        def left_cell(px, py, dd):
            if dd == (1, 0): return cell(px, py - 1)
            if dd == (0, 1): return cell(px, py)
            if dd == (-1, 0): return cell(px - 1, py)
            return cell(px - 1, py - 1)
        nx, ny = x + d[0], y + d[1]
        # at the new corner, choose the next direction: try right turn, straight, left turn
        chosen = None
        for turn in (1, 0, -1):
            nd = dirs[(k + turn) % 4]
            if right_cell(nx, ny, nd) and not left_cell(nx, ny, nd):
                chosen = nd
                break
        if chosen is None:
            raise MeshError('contour trace failed')
        pos, d = (nx, ny), chosen
        if pos == start and d == (1, 0):
            break
        if pos != pts[-1]:
            pts.append(pos)
    else:
        raise MeshError('contour trace did not close')
    # back to unpadded grid coordinates
    return [(x - 1, y - 1) for x, y in pts]


def _collinear_free(pts: np.ndarray) -> np.ndarray:
    keep = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-9:
            keep.append(i)
    return pts[keep]


def _dp(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker for an open polyline; returns kept points including both ends."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = points[a], points[b]
        seg = q - p
        n2 = float(seg @ seg)
        mid = points[a + 1:b]
        if n2 == 0:
            dist = np.hypot(*(mid - p).T)
        else:
            t = np.clip(((mid - p) @ seg) / n2, 0, 1)
            dist = np.hypot(*(mid - (p + t[:, None] * seg)).T)
        j = int(np.argmax(dist))
        if dist[j] > tol:
            k = a + 1 + j
            keep[k] = True
            stack += [(a, k), (k, b)]
    return points[keep]


def _simplify_closed(poly: np.ndarray, tol: float) -> np.ndarray:
    if len(poly) <= 4 or tol <= 0:
        return poly
    # split at the two mutually farthest-ish points so both halves are open polylines
    i = 0
    j = int(np.argmax(np.hypot(*(poly - poly[i]).T)))
    i = int(np.argmax(np.hypot(*(poly - poly[j]).T)))
    i, j = sorted((i, j))
    a = _dp(poly[i:j + 1], tol)
    b = _dp(np.concatenate([poly[j:], poly[:i + 1]]), tol)
    return np.concatenate([a[:-1], b[:-1]])


def _simplify_pinned(poly: np.ndarray, tol: float, pinned_y: list[float]) -> np.ndarray:
    """Simplify a closed outline, keeping every vertex on a pinned row (a cut line) where it is."""
    pins = np.nonzero(np.isin(poly[:, 1], pinned_y))[0] if pinned_y else np.zeros(0, int)
    if not len(pins):
        return _simplify_closed(poly, tol)
    if tol <= 0:
        return poly
    poly = np.roll(poly, -int(pins[0]), axis=0)
    pins = np.append((pins - pins[0]) % len(poly), len(poly))
    ring = np.concatenate([poly, poly[:1]])
    parts = [_dp(ring[a:b + 1], tol)[:-1] for a, b in zip(pins[:-1], pins[1:])]
    return np.concatenate(parts)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 < 0 and d3 * d4 < 0


def is_simple(poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if _segments_cross(a, b, c, d):
                return False
    return True


def triangulate(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple polygon; returns index triples into poly, consistently oriented."""
    n = len(poly)
    if n < 3:
        raise MeshError('polygon needs three vertices')
    sign = 1. if _area(poly) > 0 else -1.
    idx = list(range(n))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def reflex_set():
        m = len(idx)
        return [idx[k] for k in range(m) if sign * cross(poly[idx[k - 1]], poly[idx[k]], poly[idx[(k + 1) % m]]) <= 1e-12]
    tris = []
    reflex = reflex_set()
    while len(idx) > 3:
        m = len(idx)
        rp = poly[reflex] if reflex else np.zeros((0, 2))
        found = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if sign * cross(a, b, c) <= 1e-12:
                continue
            if len(rp):
                others = np.array([r not in (i0, i1, i2) for r in reflex])
                q = rp[others]
                q = q[~(np.all(q == a, axis=1) | np.all(q == b, axis=1) | np.all(q == c, axis=1))]
                if len(q):
                    d1 = sign * ((b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0]))
                    d2 = sign * ((c[0] - b[0]) * (q[:, 1] - b[1]) - (c[1] - b[1]) * (q[:, 0] - b[0]))
                    d3 = sign * ((a[0] - c[0]) * (q[:, 1] - c[1]) - (a[1] - c[1]) * (q[:, 0] - c[0]))
                    if np.any((d1 >= 0) & (d2 >= 0) & (d3 >= 0)):
                        continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:
            raise MeshError('ear clipping found no ear (polygon not simple)')
        reflex = reflex_set()
    tris.append(tuple(idx))
    return tris


def build_mesh(alpha: np.ndarray, params: MeshParams = MeshParams(), band: tuple | None = None) -> Mesh:
    """Hull polygons and triangles for one layer's alpha (uint8, H x W), or for its rows in band (y0, y1).

    A band edge that is None is open (the layer's own outline); the others are cut lines on cell boundaries.
    """
    if alpha.ndim != 2:
        raise MeshError('alpha must be 2-D')
    h, w = alpha.shape
    c = max(1, int(params.cell))
    lo, hi = band or (None, None)
    for v in (lo, hi):
        if v is not None and v % c:
            raise MeshError(f'cut line {v} is not on a {c} px cell boundary')
    drawn = alpha > params.threshold
    if lo is not None or hi is not None:
        drawn = drawn.copy()
        drawn[:max(0, lo or 0)] = False
        if hi is not None:
            drawn[hi:] = False
    if not drawn.any():
        raise MeshError('empty layer')
    gh, gw = -(-h // c), -(-w // c)
    pad = np.zeros((gh * c, gw * c), dtype=bool)
    pad[:h, :w] = drawn
    grid = pad.reshape(gh, c, gw, c).any(axis=(1, 3))
    grow_cells = int(np.ceil((params.padding + params.tolerance) / c))
    margin = grow_cells + 1
    g = np.zeros((gh + 2 * margin, gw + 2 * margin), dtype=bool)
    g[margin:margin + gh, margin:margin + gw] = grid
    g = _grow(g, grow_cells)
    if lo is not None:
        g[:lo // c + margin] = False
    if hi is not None:
        g[hi // c + margin:] = False
    pinned = [float(v) for v in (lo, hi) if v is not None]
    labels, count = _label(g)
    polygons = []
    for lab in range(1, count + 1):
        comp = labels == lab
        corners = np.array(_trace(comp), dtype=np.float64)
        pts = (corners - margin) * c                       # grid corners to canvas pixels
        pts = _collinear_free(pts)
        tol = float(params.tolerance)
        simple = _simplify_pinned(pts, tol, pinned)
        while not is_simple(simple) and tol > 0.25:
            tol /= 2
            simple = _simplify_pinned(pts, tol, pinned)
        if not is_simple(simple):
            simple = pts
        if _area(simple) < 0:
            simple = simple[::-1]
        polygons.append(simple)
    vertices = np.concatenate(polygons)
    triangles, base = [], 0
    for poly in polygons:
        triangles += [(a + base, b + base, cc + base) for a, b, cc in triangulate(poly)]
        base += len(poly)
    x0, y0 = np.floor(vertices.min(axis=0)).astype(int)
    x1, y1 = np.ceil(vertices.max(axis=0)).astype(int)
    mesh = Mesh(polygons, vertices, np.array(triangles, dtype=np.int64), (int(x0), int(y0), int(x1), int(y1)))
    mesh.stats = {'components': len(polygons), 'vertices': int(len(vertices)), 'triangles': len(triangles),
                  'drawn_pixels': int(drawn.sum()), 'hull_area': float(sum(abs(_area(p)) for p in polygons))}
    return mesh


def rasterize(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Pixel-centre inside test for the union of polygons (even-odd per polygon, union across)."""
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    ys = np.arange(h) + .5
    for poly in polygons:
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        if y0 >= y1 or x0 >= x1:
            continue
        xs = np.arange(x0, x1) + .5
        inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        n = len(poly)
        for i in range(n):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
            if ay == by:
                continue
            yy = ys[y0:y1]
            cross = ((ay <= yy) & (yy < by)) | ((by <= yy) & (yy < ay))
            t = (yy - ay) / (by - ay)
            xcross = ax + t * (bx - ax)
            flip = cross[:, None] & (xs[None, :] < xcross[:, None])
            inside ^= flip
        out[y0:y1, x0:x1] |= inside
    return out


def coverage(alpha: np.ndarray, mesh: Mesh, threshold: int = 0) -> dict:
    """Drawn pixels outside the hull (must be zero) and the hull's waste."""
    drawn = alpha > threshold
    hull = rasterize(mesh.polygons, alpha.shape)
    outside = int((drawn & ~hull).sum())
    return {'outside_pixels': outside, 'hull_pixels': int(hull.sum()), 'drawn_pixels': int(drawn.sum())}
