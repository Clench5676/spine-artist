"""Texture packing: layer crops onto pages, with padding and colour bleed, plus the Spine 4.x atlas text.

Regions are the layers' hull bounding boxes, cropped from the canvas and scaled by one global density
factor. Pixels stay straight alpha. Bleed copies the nearest drawn colour into fully transparent pixels
(alpha stays zero), as Spine's texture packer does, so bilinear filtering never pulls in black.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PackParams:
    page_width: int = 2048
    page_height: int = 2048
    max_pages: int = 1
    padding: int = 2          # px between regions and to the page edge
    bleed: int = 4            # px of colour bled into transparent pixels
    density: float = 1.       # texels per canvas pixel; reduced automatically when the layers do not fit
    min_density: float = .25
    allow_shrink: bool = True
    rotate: bool = True       # allow 90-degree rotation on the page (atlas `rotate: 90`)
    mode: str = 'polygon'     # 'polygon': hulls interlock, boxes may overlap (as Spine's polygon packing); 'rect'
    grid: int = 4             # px per occupancy cell in polygon mode

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ('page_width', 'page_height', 'max_pages', 'padding', 'bleed',
                                               'density', 'min_density', 'allow_shrink', 'rotate', 'mode', 'grid')}


@dataclass
class Placement:
    name: str
    page: int
    x: int
    y: int
    width: int
    height: int
    box: tuple[int, int, int, int]       # canvas crop x0, y0, x1, y1
    degrees: int = 0                     # stored rotated counter-clockwise by this (atlas `rotate:`); width/height unrotated

    @property
    def rotated(self) -> bool:
        return self.degrees != 0


@dataclass
class Packed:
    pages: list[Image.Image]
    placements: dict[str, Placement]
    density: float
    stats: dict = field(default_factory=dict)


class PackError(ValueError):
    pass


class _MaxRects:
    """MaxRects bin packing, best short side fit, optionally trying the 90-degree rotation."""

    def __init__(self, width: int, height: int):
        self.free = [(0, 0, width, height)]

    def insert(self, w: int, h: int, rotate: bool = False):
        best, score = None, None
        for fx, fy, fw, fh in self.free:
            for rw, rh, rot in ((w, h, False), (h, w, True)) if rotate and w != h else ((w, h, False),):
                if rw <= fw and rh <= fh:
                    s = (min(fw - rw, fh - rh), max(fw - rw, fh - rh), rot)
                    if score is None or s < score:
                        best, score = (fx, fy, rw, rh, rot), s
        if best is None:
            return None
        self._split(best[:4])
        return best

    def _split(self, used):
        ux, uy, uw, uh = used
        out = []
        for fx, fy, fw, fh in self.free:
            if ux >= fx + fw or ux + uw <= fx or uy >= fy + fh or uy + uh <= fy:
                out.append((fx, fy, fw, fh))
                continue
            if ux > fx:
                out.append((fx, fy, ux - fx, fh))
            if ux + uw < fx + fw:
                out.append((ux + uw, fy, fx + fw - ux - uw, fh))
            if uy > fy:
                out.append((fx, fy, fw, uy - fy))
            if uy + uh < fy + fh:
                out.append((fx, uy + uh, fw, fy + fh - uy - uh))
        pruned = []
        for i, a in enumerate(out):
            if any(i != j and a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2]
                   and a[1] + a[3] <= b[1] + b[3] and (a != b or j < i) for j, b in enumerate(out)):
                continue
            pruned.append(a)
        self.free = pruned


def bleed(image: Image.Image, radius: int) -> Image.Image:
    """Copy the nearest drawn RGB into transparent pixels, up to radius px; alpha is unchanged."""
    a = np.asarray(image.convert('RGBA')).copy()
    if radius <= 0:
        return Image.fromarray(a, 'RGBA')
    rgb = a[..., :3].astype(np.float32)
    known = a[..., 3] > 0
    h, w = known.shape
    for _ in range(radius):
        acc = np.zeros_like(rgb)
        cnt = np.zeros((h, w), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            k = np.zeros((h, w), dtype=bool)
            k[yd, xd] = known[ys, xs]
            src = np.zeros_like(rgb)
            src[yd, xd] = rgb[ys, xs]
            acc += src * k[..., None]
            cnt += k
        grow = (~known) & (cnt > 0)
        if not grow.any():
            break
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    out = a.copy()
    clear = a[..., 3] == 0
    out[..., :3][clear] = np.clip(np.rint(rgb[clear]), 0, 255).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')


def _scaled(box, density):
    x0, y0, x1, y1 = box
    return max(1, int(round((x1 - x0) * density))), max(1, int(round((y1 - y0) * density)))


def _grown(mask: np.ndarray, grow: int) -> np.ndarray:
    """The mask on a canvas expanded by `grow` px per side, dilated by a disc of that radius.

    Rotation-equivariant: rot90(_grown(m)) == _grown(rot90(m)).
    """
    h, w = mask.shape
    m = np.zeros((h + 2 * grow, w + 2 * grow), bool)
    m[grow:grow + h, grow:grow + w] = mask
    if grow <= 0:
        return m
    out = m.copy()
    H, W = m.shape
    for dy in range(-grow, grow + 1):
        for dx in range(-grow, grow + 1):
            if dx * dx + dy * dy <= grow * grow:
                out[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] |= \
                    m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return out


def _cells(mask: np.ndarray, g: int) -> np.ndarray:
    """Max-pool a pixel mask to g px cells (anchored at the top-left)."""
    h, w = mask.shape
    gh, gw = -(-h // g), -(-w // g)
    pad = np.zeros((gh * g, gw * g), bool)
    pad[:h, :w] = mask
    return pad.reshape(gh, g, gw, g).any(axis=(1, 3))


def _pack_polygon(crops, masks, params: PackParams, density: float, rule: str = 'top', order=None):
    """Place each region's hull mask where it overlaps nothing placed before (FFT correlation per page).

    Orientations are 0, 90, 180 and 270 degrees counter-clockwise (verified against the runtime's atlas
    `rotate:` storage). rule 'top' takes the top-most, then left-most free cell; 'low' the lowest bottom edge.
    """
    g = max(1, int(params.grid))
    pw, ph = params.page_width, params.page_height
    gw, gh = pw // g, ph // g
    pages_occ: list[np.ndarray] = []
    order = order or sorted(crops, key=lambda n: -int(masks[n].sum()))
    spots = {}
    for name in order:
        w, h = _scaled(crops[name][1], density)
        m = masks[name]
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        placed = None
        for p in range(params.max_pages):
            if p == len(pages_occ):
                pages_occ.append(np.zeros((gh, gw), bool))
            occ = pages_occ[p]
            F = np.fft.rfft2(occ.astype(np.float32))
            best = None
            for k in ((0, 1, 2, 3) if params.rotate else (0,)):
                cell = _cells(_grown(np.rot90(m, k), params.padding), g)   # hull plus padding, expanded box
                ch, cw = cell.shape
                if ch > gh or cw > gw:
                    continue
                kern = np.zeros((gh, gw), np.float32)
                kern[:ch, :cw] = cell
                corr = np.fft.irfft2(F * np.conj(np.fft.rfft2(kern)), s=(gh, gw))
                free = corr[:gh - ch + 1, :gw - cw + 1] < .5
                ys, xs = np.nonzero(free)
                if not len(ys):
                    continue
                if rule == 'low':
                    i = np.lexsort((xs, ys + ch))[0]
                    key = (int(ys[i]) + ch, int(xs[i]))
                elif rule == 'left':
                    i = np.lexsort((ys, xs))[0]
                    key = (int(xs[i]), int(ys[i]))
                else:
                    i = np.lexsort((xs, ys))[0]
                    key = (int(ys[i]), int(xs[i]))
                if best is None or key < best[0]:
                    best = (key, int(ys[i]), int(xs[i]), k, cell)
            if best is not None:
                _, cy, cx, k, cell = best
                occ[cy:cy + cell.shape[0], cx:cx + cell.shape[1]] |= cell
                # the region's own (rotated) box starts `padding` px inside the expanded box
                spots[name] = (p, cx * g + params.padding, cy * g + params.padding, w, h, 90 * k)
                placed = True
                break
        if not placed:
            return None, len(pages_occ)
    return spots, len(pages_occ)


def pack(crops: dict[str, tuple[Image.Image, tuple[int, int, int, int]]], params: PackParams = PackParams(),
         masks: dict[str, np.ndarray] | None = None) -> Packed:
    """Pack named crops (image cut from the canvas, canvas box) onto pages; shrinks density until they fit.

    With masks (each region's hull in crop pixels) and mode 'polygon', regions interlock and only their hull
    pixels are written, so their boxes may overlap on the page.
    """
    if not crops:
        raise PackError('nothing to pack')
    if params.mode == 'polygon' and masks is not None:
        return _pack_with_masks(crops, masks, params)
    density = params.density
    order = sorted(crops, key=lambda n: -max(_scaled(crops[n][1], 1.)))
    while True:
        bins = [_MaxRects(params.page_width - params.padding, params.page_height - params.padding)]
        spots, ok = {}, True
        for name in order:
            w, h = _scaled(crops[name][1], density)
            spot = None
            for p, b in enumerate(bins):
                got = b.insert(w + params.padding, h + params.padding, params.rotate)
                if got is not None:
                    spot = (p, got)
                    break
            if spot is None and len(bins) < params.max_pages:
                bins.append(_MaxRects(params.page_width - params.padding, params.page_height - params.padding))
                got = bins[-1].insert(w + params.padding, h + params.padding, params.rotate)
                spot = (len(bins) - 1, got) if got is not None else None
            if spot is None:
                ok = False
                break
            spots[name] = (spot[0], spot[1][0] + params.padding, spot[1][1] + params.padding, w, h, spot[1][4])
        if ok:
            break
        if not params.allow_shrink or density * .95 < params.min_density:
            raise PackError(f'layers do not fit {params.max_pages} page(s) of {params.page_width}x{params.page_height}')
        density *= .95
    pages = [Image.new('RGBA', (params.page_width, params.page_height), (0, 0, 0, 0)) for _ in range(len(bins))]
    placements = {}
    used = 0
    for name, (p, x, y, w, h, rot) in spots.items():
        image, box = crops[name]
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        image = bleed(image, params.bleed)
        if rot:
            image = image.transpose(Image.Transpose.ROTATE_90)    # counter-clockwise, the atlas `rotate: 90` storage
        pages[p].paste(image, (x, y))   # copy: alpha_composite would drop bled colour
        placements[name] = Placement(name, p, x, y, w, h, box, 90 if rot else 0)
        used += w * h
    stats = {'density': density, 'pages': len(pages), 'regions': len(placements),
             'occupancy': used / (len(pages) * params.page_width * params.page_height)}
    return Packed(pages, placements, density, stats)


def _pack_with_masks(crops, masks, params: PackParams) -> Packed:
    from itertools import permutations
    by_area = sorted(crops, key=lambda n: -int(masks[n].sum()))
    head, tail = by_area[:3], by_area[3:]
    orders = [list(p) + tail for p in permutations(head)]
    # start near the largest density the page area allows (hulls rarely fill more than about 70 % of a page)
    total = sum(int(m.sum()) for m in masks.values())
    room = .72 * params.max_pages * params.page_width * params.page_height
    estimate = min(params.density, (room / max(1, total)) ** .5)
    density = params.density
    first = True
    while True:
        spots = None
        for order in orders:
            for rule in ('top', 'left', 'low'):
                spots, n_pages = _pack_polygon(crops, masks, params, density, rule, order)
                if spots is not None:
                    break
            if spots is not None:
                break
        if spots is not None:
            break
        if not params.allow_shrink or density * .97 < params.min_density:
            raise PackError(f'layers do not fit {params.max_pages} page(s) of {params.page_width}x{params.page_height}')
        # the requested density first, then straight to the area estimate, then down in small steps
        density = estimate if first and estimate < density * .97 else density * .97
        first = False
    pages = [np.zeros((params.page_height, params.page_width, 4), np.uint8) for _ in range(n_pages)]
    owner = [np.zeros((params.page_height, params.page_width), bool) for _ in range(n_pages)]
    placements, used = {}, 0
    for name, (p, x, y, w, h, deg) in spots.items():
        image, box = crops[name]
        m = masks[name]
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        g_ = params.padding
        own = _grown(m, g_)                                     # hull plus padding, on the expanded box
        img = np.zeros(own.shape + (4,), np.uint8)
        img[g_:g_ + h, g_:g_ + w] = np.asarray(bleed(image, params.bleed))
        if deg:
            img, own = np.rot90(img, deg // 90), np.rot90(own, deg // 90)   # counter-clockwise storage
        X, Y = x - g_, y - g_
        hh, ww = own.shape
        if X < 0 or Y < 0 or Y + hh > params.page_height or X + ww > params.page_width:
            raise PackError(f'region {name} leaves the page')
        if np.any(owner[p][Y:Y + hh, X:X + ww] & own):
            raise PackError(f'region {name} overlaps another region')
        sub = pages[p][Y:Y + hh, X:X + ww]
        sub[own] = img[own]
        owner[p][Y:Y + hh, X:X + ww] |= own
        placements[name] = Placement(name, p, x, y, w, h, box, deg)
        used += int(m.sum())
    stats = {'density': density, 'pages': n_pages, 'regions': len(placements), 'mode': 'polygon',
             'occupancy': used / (n_pages * params.page_width * params.page_height)}
    return Packed([Image.fromarray(a, 'RGBA') for a in pages], placements, density, stats)


def atlas_text(packed: Packed, page_names: list[str]) -> str:
    """Spine 4.x atlas text: one block per page, regions as bounds without rotation or trim offsets."""
    lines = []
    for p, page in enumerate(packed.pages):
        if lines:
            lines.append('')
        lines += [page_names[p], f'\tsize: {page.width}, {page.height}', '\tfilter: Linear, Linear']
        for name in sorted(n for n, pl in packed.placements.items() if pl.page == p):
            pl = packed.placements[name]
            lines += [name, f'\tbounds: {pl.x}, {pl.y}, {pl.width}, {pl.height}']
            if pl.degrees:
                lines.append(f'\trotate: {pl.degrees}')
    return '\n'.join(lines) + '\n'
