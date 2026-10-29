"""Cutting a layer into pieces that each fit a page, joined without a seam.

One region must fit one page. A full figure drawn as one body layer is taller than an official page at the
official texel density, so the packer would have to shrink every region to fit it. Official portraits are cut
into several regions instead; a layer here is cut the same way, along rows.

Each piece is its own slot and region on the layer's bone. Its mesh ends exactly on the cut line (a cell
boundary, vertices pinned there), so the pieces tile the layer's hull with no overlap and no gap. Its texture
region carries APRON px of the real neighbouring pixels across the cut, so bilinear filtering and the page
resampling read the same pixels at the cut as one region would.

Cuts are decisions (`cuts`: layer id -> canvas rows); missing ones are proposed where the drawing is narrowest,
near equal shares, and recorded in the manifest.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .atlas import PackParams
from .mesh import Mesh, MeshParams, build_mesh, rasterize

APRON = 16          # canvas px across a cut: 4 texels at the lowest packing density (0.25), Lanczos support 3
KEEP = ('features', 'eyeclose')     # face patches are small and switch as one attachment: never cut
ROOM = .72          # share of the page area hulls fill at best (atlas._pack_with_masks starts there)
FILL = .9           # a piece takes at most this share of the page side, leaving room to interlock
WINDOW = .15        # a cut may move this share of a piece's height towards a narrower row


def expected_density(meshes: dict[str, Mesh], pack: PackParams) -> float:
    total = sum(m.stats['hull_area'] for m in meshes.values())
    return min(pack.density, (ROOM * pack.max_pages * pack.page_width * pack.page_height / max(1., total)) ** .5)


def _limit(width: float, pack: PackParams) -> float | None:
    """The tallest piece of this width (at density) that fits the page, or None if the width itself does not."""
    long_side = max(pack.page_width, pack.page_height) - 2 * pack.padding
    short_side = min(pack.page_width, pack.page_height) - 2 * pack.padding
    if width <= short_side:
        return long_side
    if width <= long_side:
        return short_side
    return None


def propose(data: dict, meshes: dict[str, Mesh], images: dict[str, Image.Image], pack: PackParams,
            mesh_params) -> dict[str, list[int]]:
    """Cut rows for every layer too large for a page at the expected density."""
    d = expected_density(meshes, pack)
    out = {}
    for l in data['layers']:
        if l['role'] in KEEP:
            continue
        x0, y0, x1, y1 = meshes[l['id']].bbox
        limit = _limit((x1 - x0) * d, pack)
        if limit is None or (y1 - y0) * d <= limit:
            continue
        n = math.ceil((y1 - y0) * d / (FILL * limit))
        cell = max(1, int(mesh_params(l).cell))
        alpha = np.asarray(images[l['id']].getchannel('A'))
        widths = (alpha > mesh_params(l).threshold).sum(axis=1)
        share = (y1 - y0) / n
        cuts = []
        for k in range(1, n):
            target = y0 + k * share
            lo, hi = int(target - WINDOW * share), int(target + WINDOW * share)
            rows = np.arange(-(-lo // cell) * cell, hi + 1, cell)
            rows = rows[(rows > y0) & (rows < y1)]
            best = int(rows[np.argmin(widths[np.minimum(rows, len(widths) - 1)])]) if len(rows) else \
                int(round(target / cell) * cell)
            cuts.append(best)
        bounds = [y0, *cuts, y1]
        if any((b - a) * d > limit for a, b in zip(bounds, bounds[1:])):
            cuts = [int(round((y0 + k * share) / cell) * cell) for k in range(1, n)]
        out[l['id']] = cuts
    return out


def validate(cuts, data: dict, meshes: dict[str, Mesh]) -> list[str]:
    if cuts is None:
        return []
    if not isinstance(cuts, dict):
        return ['cuts must map layer ids to lists of canvas rows']
    errors = []
    layers = {l['id']: l for l in data['layers']}
    for lid, rows in cuts.items():
        if lid not in layers:
            errors.append(f'cuts: unknown layer {lid!r}')
            continue
        if layers[lid]['role'] in KEEP and rows:
            errors.append(f'cuts: {lid} is a face patch and is never cut')
            continue
        if not isinstance(rows, list) or not all(isinstance(v, int) for v in rows):
            errors.append(f'cuts.{lid} must be a list of canvas rows (integers)')
            continue
        y0, y1 = meshes[lid].bbox[1], meshes[lid].bbox[3]
        if rows != sorted(set(rows)) or any(not y0 < v < y1 for v in rows):
            errors.append(f'cuts.{lid}: rows must rise and lie inside the layer ({y0}..{y1})')
    return errors


def snap(cuts: dict[str, list[int]], data: dict, mesh_params) -> dict[str, list[int]]:
    """Cut rows on the mesh cell boundaries."""
    layers = {l['id']: l for l in data['layers']}
    out = {}
    for lid, rows in cuts.items():
        cell = max(1, int(mesh_params(layers[lid]).cell))
        out[lid] = sorted({int(round(v / cell) * cell) for v in rows})
    return out


def split(data: dict, cuts: dict[str, list[int]], meshes: dict[str, Mesh], images: dict[str, Image.Image],
          mesh_params) -> dict[str, dict]:
    """Replace each cut layer by its pieces in data['layers'], meshes and images; returns piece id -> source.

    images[piece] holds only the piece's rows (for the instruments); the source keeps the whole layer for the
    texture crop with its apron.
    """
    layers, pieces = [], {}
    for l in data['layers']:
        rows = cuts.get(l['id'])
        if not rows:
            layers.append(l)
            continue
        whole, image = meshes.pop(l['id']), images.pop(l['id'])
        alpha = np.asarray(image.getchannel('A'))
        bounds = [None, *rows, None]
        for k, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
            pid = f'{l["id"]}_{k}'
            piece = {**l, 'id': pid, 'slot': f'{l["slot"]}_{k}', 'piece': {'of': l['id'], 'band': [a, b]}}
            meshes[pid] = build_mesh(alpha, mesh_params(l), band=(a, b))
            rgba = np.asarray(image).copy()
            if a is not None:
                rgba[:a, :, 3] = 0
            if b is not None:
                rgba[b:, :, 3] = 0
            images[pid] = Image.fromarray(rgba, 'RGBA')
            pieces[pid] = {'image': image, 'whole': whole, 'band': (a, b)}
            layers.append(piece)
    data['layers'] = layers
    return pieces


def crop(source: dict, mesh: Mesh) -> tuple[Image.Image, tuple[int, int, int, int], np.ndarray]:
    """The piece's texture crop, its canvas box and its packing mask: the piece's hull plus the apron rows of
    the whole layer's hull across each cut, so those page pixels are written and kept for filtering."""
    a, b = source['band']
    whole = source['whole']
    x0, y0, x1, y1 = mesh.bbox
    top = max(whole.bbox[1], y0 - APRON) if a is not None else y0
    bottom = min(whole.bbox[3], y1 + APRON) if b is not None else y1
    box = (min(x0, whole.bbox[0]), top, max(x1, whole.bbox[2]), bottom)
    shape = (box[3] - box[1], box[2] - box[0])
    origin = np.array(box[:2], dtype=np.float64)
    mask = rasterize([p - origin for p in mesh.polygons], shape)
    apron = rasterize([p - origin for p in whole.polygons], shape)
    rows = np.zeros(shape[0], bool)
    if a is not None:
        rows[:max(0, a - box[1])] = True
    if b is not None:
        rows[max(0, b - box[1]):] = True
    mask |= apron & rows[:, None]
    return source['image'].crop(box), box, mask
